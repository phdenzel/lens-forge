// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-point vs batched ray-tracing of an image-plane grid to the source plane.
//!
//! Two labels per `(backend, target-count)`:
//!
//! * **`per-point`** — the current loop, one `ray_trace` per target.
//! * **`batched`** — one `RayTracer::Direct.trace` over all targets.
//!
//! The gap depends on the `jet_batch` override: `spectral` parallelizes the
//! interpolation (rayon), the fastsum backends reuse the tree across the batch,
//! and a parametric lens (none) is the null baseline. `--features full-bench`
//! widens the target-count sweep.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lens_forge::fastsum::Backend;
use lens_forge::grid::{Grid2D, PixelGrid, Point2D};
use lens_forge::lens::freeform::{FieldMethod, PixelLens};
use lens_forge::lens::{FreeFormLens, LensModel, SIS};
use lens_forge::solver::mapping::raytrace::RayTracer;

/// Side of the κ grid backing the free-form lenses.
const KAPPA_SIDE: usize = 64;

#[cfg(feature = "full-bench")]
const TARGET_SIDES: &[usize] = &[32, 64, 96];
#[cfg(not(feature = "full-bench"))]
const TARGET_SIDES: &[usize] = &[64];

/// A smooth, positive convergence field for the free-form lenses.
fn kappa() -> PixelGrid {
    PixelGrid::from_func(Grid2D::square(KAPPA_SIDE, 2.0).unwrap(), |x, y| {
        0.3 * (-(x * x + y * y) / 0.5).exp() + 0.05
    })
}

/// A `side × side` grid of image-plane targets over the field of view.
fn targets(side: usize) -> Vec<Point2D> {
    let mut out = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let x = -1.5 + 3.0 * (i as f64) / (side as f64 - 1.0);
            let y = -1.5 + 3.0 * (j as f64) / (side as f64 - 1.0);
            out.push(Point2D::new(x, y));
        }
    }
    out
}

/// Bench per-point vs batched tracing of `lens` over each target-count.
fn bench_lens(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, name: &str, lens: &dyn LensModel) {
    for &side in TARGET_SIDES {
        let q = targets(side);
        let n = q.len();
        group.bench_with_input(BenchmarkId::new(format!("{name}/per-point"), n), &q, |b, q| {
            b.iter(|| {
                let out: Vec<Point2D> = q.iter().map(|&t| lens.ray_trace(t)).collect();
                black_box(out)
            });
        });
        group.bench_with_input(BenchmarkId::new(format!("{name}/batched"), n), &q, |b, q| {
            b.iter(|| black_box(RayTracer::Direct.trace(lens, black_box(q))));
        });
    }
}

fn bench_raytrace(c: &mut Criterion) {
    let spectral = PixelLens::build(kappa(), FieldMethod::Spectral).unwrap();
    let analytic = PixelLens::build(kappa(), FieldMethod::analytic(Backend::BarnesHut)).unwrap();
    let fastsum = PixelLens::build(kappa(), FieldMethod::fast_sum(Backend::BarnesHut)).unwrap();
    let sis = SIS::new(Point2D::origin(), 0.6); // parametric null baseline

    let mut group = c.benchmark_group("raytrace");
    bench_lens(&mut group, "spectral", &spectral);
    bench_lens(&mut group, "analytic-bh", &analytic);
    bench_lens(&mut group, "fastsum-bh", &fastsum);
    bench_lens(&mut group, "sis", &sis);
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30);
    targets = bench_raytrace
}
criterion_main!(benches);
