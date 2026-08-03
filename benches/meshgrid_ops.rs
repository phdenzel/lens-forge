// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Throughput benchmarks for [`MeshGrid`] construction and field operators,
//! swept over site counts:
//!
//! * **`build`** — full mesh construction (Bowyer–Watson triangulation + the
//!   dual Voronoi tessellation and adjacency) from a raw site cloud.
//! * **`gradient` / `laplacian`** — the two-point-flux finite-volume
//!   [`MeshGrid::gradient_field`] and [`MeshGrid::laplacian_field`] on a smooth
//!   field.
//! * **`poisson`** — a Jacobi-preconditioned CG solve of $\nabla^2\psi = f$
//!   ([`MeshGrid::poisson_cg`], Dirichlet) on a clumpy source.
//! * **`interp`** — barycentric resampling at a raster of query points, serial
//!   ([`MeshGrid::interpolate_batch`]) vs Rayon
//!   ([`MeshGrid::par_interpolate_batch`]).
//!
//! Sites concentrate toward the domain centre (adaptive-mesh clustering);
//! `--features full-bench` widens the site-count sweep.

use std::hint::black_box;

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use lens_forge::Lcg;
use lens_forge::grid::{BBox, BoundaryCondition, MeshGrid, Point2D, PoissonConfig};

const DOMAIN: f64 = 1.0;
const RASTER: usize = 128; // interp query grid side (W = H)

#[cfg(feature = "full-bench")]
const SITE_COUNTS: &[usize] = &[500, 2_000, 8_000];
#[cfg(not(feature = "full-bench"))]
const SITE_COUNTS: &[usize] = &[2_000];

fn bbox() -> BBox {
    BBox::from_lims(-DOMAIN, DOMAIN, -DOMAIN, DOMAIN)
}

/// `n` sites on `[-1, 1]²`, radially concentrated toward the centre (`r = u²`).
fn sites(n: usize) -> Vec<Point2D> {
    let mut rng = Lcg::new(0x5EED);
    (0..n)
        .map(|_| {
            let u = rng.f();
            let r = 0.98 * DOMAIN * u * u;
            let theta = std::f64::consts::TAU * rng.f();
            Point2D::new(r * theta.cos(), r * theta.sin())
        })
        .collect()
}

/// A smooth field `f(x, y) = sin(πx)·cos(πy)` sampled at the sites.
fn smooth_mesh(n: usize) -> MeshGrid {
    let s = sites(n);
    let pi = std::f64::consts::PI;
    MeshGrid::from_func(&s, bbox(), |x, y| (pi * x).sin() * (pi * y).cos()).unwrap()
}

/// A clumpy Poisson source: a few narrow Gaussians (broad spectrum, so CG does
/// representative work rather than converging in ~1 iteration on a smooth mode).
fn clumpy_mesh(n: usize) -> MeshGrid {
    let s = sites(n);
    let centers = [(-0.4, 0.25), (0.3, -0.35), (0.55, 0.5), (-0.15, -0.6)];
    let s2 = 0.08 * 0.08;
    MeshGrid::from_func(&s, bbox(), |x, y| {
        centers
            .iter()
            .map(|&(cx, cy)| {
                let r2 = (x - cx).powi(2) + (y - cy).powi(2);
                (-0.5 * r2 / s2).exp()
            })
            .sum()
    })
    .unwrap()
}

/// A `side × side` raster of query points covering the domain.
fn raster_targets(side: usize) -> Vec<Point2D> {
    let step = 2.0 * DOMAIN / side as f64;
    let mut out = Vec::with_capacity(side * side);
    for j in 0..side {
        let y = -DOMAIN + (j as f64 + 0.5) * step;
        for i in 0..side {
            let x = -DOMAIN + (i as f64 + 0.5) * step;
            out.push(Point2D::new(x, y));
        }
    }
    out
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("meshgrid/build");
    group.sampling_mode(SamplingMode::Flat);
    for &n in SITE_COUNTS {
        let s = sites(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("constant", n), &s, |b, s| {
            b.iter(|| black_box(MeshGrid::constant(black_box(s), bbox(), 0.0).unwrap()));
        });
    }
    group.finish();
}

fn bench_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("meshgrid/gradient");
    group.sampling_mode(SamplingMode::Flat);
    for &n in SITE_COUNTS {
        let m = smooth_mesh(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &m, |b, m| {
            b.iter(|| black_box(m.gradient_field()));
        });
    }
    group.finish();
}

fn bench_laplacian(c: &mut Criterion) {
    let mut group = c.benchmark_group("meshgrid/laplacian");
    group.sampling_mode(SamplingMode::Flat);
    for &n in SITE_COUNTS {
        let m = smooth_mesh(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &m, |b, m| {
            b.iter(|| black_box(m.laplacian_field()));
        });
    }
    group.finish();
}

fn bench_poisson(c: &mut Criterion) {
    // Dirichlet: non-singular SPD system; match the uniform bench's accuracy.
    let cfg = PoissonConfig::default()
        .with_tol(1e-6)
        .with_max_iter(20_000)
        .with_bc(BoundaryCondition::Dirichlet);
    let mut group = c.benchmark_group("meshgrid/poisson");
    group.sampling_mode(SamplingMode::Flat);
    for &n in SITE_COUNTS {
        let src = clumpy_mesh(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("cg", n), &src, |b, src| {
            b.iter(|| black_box(src.poisson_cg(black_box(cfg)).unwrap()));
        });
    }
    group.finish();
}

fn bench_interp(c: &mut Criterion) {
    let query = raster_targets(RASTER);
    let mut group = c.benchmark_group("meshgrid/interp");
    group.sampling_mode(SamplingMode::Flat);
    group.throughput(Throughput::Elements((RASTER * RASTER) as u64));
    for &n in SITE_COUNTS {
        let m = smooth_mesh(n);
        group.bench_with_input(BenchmarkId::new("serial", n), &m, |b, m| {
            b.iter(|| black_box(m.interpolate_batch(black_box(&query))));
        });
        group.bench_with_input(BenchmarkId::new("parallel", n), &m, |b, m| {
            b.iter(|| black_box(m.par_interpolate_batch(black_box(&query))));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_build, bench_gradient, bench_laplacian, bench_poisson, bench_interp
}
criterion_main!(benches);
