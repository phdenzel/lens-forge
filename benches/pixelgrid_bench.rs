// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Throughput benchmarks for [`PixelGrid`] derivative and Poisson operators.
//!
//! Three comparisons:
//!
//! * **`gradient` / `laplacian`** — the finite-difference (FD) central stencils
//!   (2nd/4th/6th order) against each other, each FD stencil against its
//!   *FD-equivalent* spectral (FFT) counterpart, and against the *exact*
//!   spectral derivative (`spectral-exact`). Because `fd-*` and `spectral-*` of
//!   the same order compute the identical discrete operator, this shows the
//!   direct FD-vs-FFT cost of the *same* result; `spectral-exact` is the extra
//!   route that buys spectral accuracy (see the accuracy tests in
//!   `src/grid/spectral.rs`).
//! * **`poisson`** — solving $\nabla^2\psi = f$ by one FFT divide-by-$-k^2$
//!   ([`PixelGrid::poisson_spectral`]) versus a finite-difference conjugate-
//!   gradient (CG) solve of the periodic 5-point Laplacian. This is the regime
//!   where the FFT wins decisively: one transform pair vs. many CG sweeps.
//!
//! Derivative inputs are a smooth periodic field; the Poisson source is a set of
//! narrow Gaussian clumps (broad spectrum, so CG does representative work —
//! a single smooth mode is nearly a Laplacian eigenvector and would let CG
//! converge in ~1 iteration). Grid sides are powers of two (FFT's good case).

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lens_forge::grid::{DerivativeStencil, PixelGrid, PoissonConfig};

/// A smooth, periodic test field on `[-π, π]²`: `f(x, y) = sin(x)·cos(y)`.
fn field(n: usize) -> PixelGrid {
    PixelGrid::from_func_sq(n, std::f64::consts::PI, |x, y| x.sin() * y.cos())
}

/// Central FD stencils to compare (label, stencil). The spectral methods reuse
/// the same stencil to form matching Fourier symbols.
const STENCILS: [(&str, DerivativeStencil); 3] = [
    ("central2", DerivativeStencil::Central),
    ("central4", DerivativeStencil::Central4),
    ("central6", DerivativeStencil::Central6),
];

const SIDES: [usize; 3] = [64, 128, 256];

fn bench_gradient(c: &mut Criterion) {
    for &n in &SIDES {
        let pg = field(n);
        let mut group = c.benchmark_group(format!("pixelgrid/gradient/N{n}"));
        group.throughput(Throughput::Elements((n * n) as u64));
        for (label, stencil) in STENCILS {
            group.bench_function(format!("fd-{label}"), |b| {
                b.iter(|| black_box(pg.gradient_field(black_box(stencil)).unwrap()));
            });
            group.bench_function(format!("spectral-{label}"), |b| {
                b.iter(|| black_box(pg.gradient_spectral(black_box(stencil)).unwrap()));
            });
        }
        group.bench_function("spectral-exact", |b| {
            b.iter(|| black_box(pg.gradient_spectral_exact().unwrap()));
        });
        group.finish();
    }
}

fn bench_laplacian(c: &mut Criterion) {
    for &n in &SIDES {
        let pg = field(n);
        let mut group = c.benchmark_group(format!("pixelgrid/laplacian/N{n}"));
        group.throughput(Throughput::Elements((n * n) as u64));
        for (label, stencil) in STENCILS {
            group.bench_function(format!("fd-{label}"), |b| {
                b.iter(|| black_box(pg.laplacian_field(black_box(stencil), None).unwrap()));
            });
            group.bench_function(format!("spectral-{label}"), |b| {
                b.iter(|| black_box(pg.laplacian_spectral(black_box(stencil)).unwrap()));
            });
        }
        group.bench_function("spectral-exact", |b| {
            b.iter(|| black_box(pg.laplacian_spectral_exact().unwrap()));
        });
        group.finish();
    }
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
    for &n in &[32usize, 64, 128] {
        let src = poisson_source(n);

        let mut group = c.benchmark_group(format!("pixelgrid/poisson/N{n}"));
        group.throughput(Throughput::Elements((n * n) as u64));

        group.bench_function("fft", |b| {
            b.iter(|| black_box(src.poisson_spectral().unwrap()));
        });
        group.bench_function("fd-cg", |b| {
            b.iter(|| black_box(src.poisson_cg(black_box(cfg)).unwrap()));
        });

        group.finish();
    }
}

criterion_group!(benches, bench_gradient, bench_laplacian, bench_poisson);
criterion_main!(benches);
