// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Throughput benchmarks for [`PixelGrid`] operators, four groups:
//!
//! * **`gradient` / `laplacian`** — finite-difference central stencils
//!   (2nd/4th/6th order) against their FD-equivalent spectral counterparts and
//!   the exact spectral derivative. Matching-order `fd-*` and `spectral-*`
//!   compute the same discrete operator, so this isolates the FD-vs-FFT cost.
//! * **`poisson`** — solving $\nabla^2\psi = f$ by one FFT divide-by-$-k^2$
//!   against a finite-difference conjugate-gradient solve of the periodic
//!   5-point Laplacian.
//! * **`interp`** — bilinear resampling at a raster of query points, serial
//!   against Rayon-parallel.
//!
//! Grid sides are powers of two (required by FFT for optimal performance). The
//! derivative field is smooth and periodic; the Poisson source is narrow
//! Gaussian clumps (broad-spectrum, so CG does representative work).

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lens_forge::grid::{DerivativeStencil, PixelGrid, Point2D, PoissonConfig};

/// A smooth, periodic test field on `[-π, π]²`: `f(x, y) = sin(x)·cos(y)`.
fn field(n: usize) -> PixelGrid {
    PixelGrid::from_func_sq(n, std::f64::consts::PI, |x, y| x.sin() * y.cos())
}

// Parameter sweeps: a quick representative subset by default, the full sweep
// under `--features full-bench`.
#[cfg(feature = "full-bench")]
const STENCILS: &[(&str, DerivativeStencil)] = &[
    ("central2", DerivativeStencil::Central),
    ("central4", DerivativeStencil::Central4),
    ("central6", DerivativeStencil::Central6),
];
#[cfg(not(feature = "full-bench"))]
const STENCILS: &[(&str, DerivativeStencil)] = &[
    ("central4", DerivativeStencil::Central4),
    ("central6", DerivativeStencil::Central6),
];

#[cfg(feature = "full-bench")]
const SIDES: &[usize] = &[64, 128, 256];
#[cfg(not(feature = "full-bench"))]
const SIDES: &[usize] = &[128];

#[cfg(feature = "full-bench")]
const POISSON_SIDES: &[usize] = &[32, 64, 128];
#[cfg(not(feature = "full-bench"))]
const POISSON_SIDES: &[usize] = &[64];

fn bench_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelgrid/gradient");
    for &n in SIDES {
        let pg = field(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        for &(label, stencil) in STENCILS {
            group.bench_with_input(BenchmarkId::new(format!("fd-{label}"), n), &pg, |b, pg| {
                b.iter(|| black_box(pg.gradient_field(black_box(stencil)).unwrap()));
            });
            group.bench_with_input(
                BenchmarkId::new(format!("spectral-{label}"), n),
                &pg,
                |b, pg| b.iter(|| black_box(pg.gradient_spectral(black_box(stencil)).unwrap())),
            );
        }
        group.bench_with_input(BenchmarkId::new("spectral-exact", n), &pg, |b, pg| {
            b.iter(|| black_box(pg.gradient_spectral_exact().unwrap()));
        });
    }
    group.finish();
}

fn bench_laplacian(c: &mut Criterion) {
    let mut group = c.benchmark_group("pixelgrid/laplacian");
    for &n in SIDES {
        let pg = field(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        for &(label, stencil) in STENCILS {
            group.bench_with_input(BenchmarkId::new(format!("fd-{label}"), n), &pg, |b, pg| {
                b.iter(|| black_box(pg.laplacian_field(black_box(stencil), None).unwrap()));
            });
            group.bench_with_input(
                BenchmarkId::new(format!("spectral-{label}"), n),
                &pg,
                |b, pg| b.iter(|| black_box(pg.laplacian_spectral(black_box(stencil)).unwrap())),
            );
        }
        group.bench_with_input(BenchmarkId::new("spectral-exact", n), &pg, |b, pg| {
            b.iter(|| black_box(pg.laplacian_spectral_exact().unwrap()));
        });
    }
    group.finish();
}

/* ============ Poisson: FFT vs finite-difference CG ======================= */

/// A clumpy Poisson source on `[-π, π]²`: a few narrow Gaussians. Broad-spectrum,
/// so the CG solver does representative work (a single smooth mode is nearly a
/// Laplacian eigenvector and would let CG converge in ~1 iteration).
fn poisson_source(n: usize) -> PixelGrid {
    let centers = [(-1.3, 0.7), (0.9, -1.1), (1.6, 1.4), (-0.4, -1.7)];
    let s2 = 0.15 * 0.15;
    PixelGrid::from_func_sq(n, std::f64::consts::PI, |x, y| {
        centers
            .iter()
            .map(|&(cx, cy)| {
                let r2 = (x - cx).powi(2) + (y - cy).powi(2);
                (-0.5 * r2 / s2).exp()
            })
            .sum()
    })
}

fn bench_poisson(c: &mut Criterion) {
    // Match the FFT solver's default accuracy so the comparison is fair.
    let cfg = PoissonConfig::default()
        .with_tol(1e-6)
        .with_max_iter(20_000);
    let mut group = c.benchmark_group("pixelgrid/poisson");
    for &n in POISSON_SIDES {
        let src = poisson_source(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("fft", n), &src, |b, src| {
            b.iter(|| black_box(src.poisson_spectral().unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("fd-cg", n), &src, |b, src| {
            b.iter(|| black_box(src.poisson_cg(black_box(cfg)).unwrap()));
        });
    }
    group.finish();
}

/* ============ Interpolation: serial vs Rayon bilinear resampling ========= */

/// A `side × side` raster of query points covering `[-π, π]²`, offset by half a
/// step so no point lands exactly on a pixel centre (the generic interp path).
fn raster_targets(side: usize) -> Vec<Point2D> {
    let pi = std::f64::consts::PI;
    let step = 2.0 * pi / side as f64;
    let mut out = Vec::with_capacity(side * side);
    for j in 0..side {
        let y = -pi + (j as f64 + 0.5) * step;
        for i in 0..side {
            let x = -pi + (i as f64 + 0.5) * step;
            out.push(Point2D::new(x, y));
        }
    }
    out
}

fn bench_interp(c: &mut Criterion) {
    const RASTER: usize = 256; // query grid side
    let query = raster_targets(RASTER);
    let mut group = c.benchmark_group("pixelgrid/interp");
    group.throughput(Throughput::Elements((RASTER * RASTER) as u64));
    for &n in SIDES {
        let pg = field(n);
        group.bench_with_input(BenchmarkId::new("serial", n), &pg, |b, pg| {
            b.iter(|| black_box(pg.interpolate_batch(black_box(&query))));
        });
        group.bench_with_input(BenchmarkId::new("parallel", n), &pg, |b, pg| {
            b.iter(|| black_box(pg.par_interpolate_batch(black_box(&query))));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // The spectral and Poisson ops run long enough that 100 samples overflow the
    // default 5 s window; 50 fits comfortably and keeps the statistics sound.
    config = Criterion::default().sample_size(50);
    targets = bench_gradient, bench_laplacian, bench_poisson, bench_interp
}
criterion_main!(benches);
