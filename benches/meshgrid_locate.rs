// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-location throughput for [`MeshGrid`] — two methods × two query
//! orderings (all four return the same cell):
//!
//! * **`cold-kdtree`** — one independent `O(log n)` query ([`MeshGrid::cell_of`])
//!   against the k-d tree; insensitive to query order.
//! * **`warm-walk`** — a greedy walk on the Voronoi adjacency
//!   ([`MeshGrid::cell_of_walk`]) seeded from the previous query; ~`O(1)` only
//!   when consecutive queries are spatially close.
//! * **`raster`** — a coherent scanline sweep of pixel centres, the renderer's
//!   access pattern (`MeshGrid::to_raster`) and the walk's best case.
//! * **`scatter`** — random points in random order: the walk degrades toward
//!   `O(√n)` while the k-d tree is unaffected — the crossover where it wins.
//!
//! Sites concentrate toward the domain centre (adaptive-mesh clustering); site
//! counts sweep the benchmark id, and `--features full-bench` widens it.

use std::hint::black_box;

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use lens_forge::Lcg;
use lens_forge::grid::{BBox, MeshGrid, Point2D};

#[cfg(feature = "full-bench")]
const SITE_COUNTS: &[usize] = &[512, 2_048, 8_192];
#[cfg(not(feature = "full-bench"))]
const SITE_COUNTS: &[usize] = &[2_048];

const RASTER: usize = 256; // W = H
const QUERIES: usize = RASTER * RASTER; // scatter count == raster pixels
const SAMPLES: usize = 50; // for the fast variants
const SAMPLES_SLOW: usize = 10; // the O(√n) warm-walk/scatter runs seconds/iter

/// `QUERIES` points drawn uniformly over `[-1, 1]²` in random order, so the walk
/// never gets a spatially-close hint from its predecessor.
fn scattered_targets() -> Vec<Point2D> {
    let mut rng = Lcg::new(0xC0FFEE);
    (0..QUERIES)
        .map(|_| Point2D::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)))
        .collect()
}

/// `n` sites on `[-1, 1]²`, radially concentrated toward the centre (`r = u²`).
fn concentrated_mesh(n: usize) -> MeshGrid {
    let mut rng = Lcg::new(0x5EED);
    let sites: Vec<Point2D> = (0..n)
        .map(|_| {
            let u = rng.f();
            let r = 0.98 * u * u;
            let theta = std::f64::consts::TAU * rng.f();
            Point2D::new(r * theta.cos(), r * theta.sin())
        })
        .collect();
    let bbox = BBox::from_lims(-1.0, 1.0, -1.0, 1.0);
    MeshGrid::constant(&sites, bbox, 0.0).unwrap()
}

fn bench_locate(c: &mut Criterion) {
    let (min, max) = ((-1.0, -1.0), (1.0, 1.0));
    let step = (max.0 - min.0) / RASTER as f64;
    let scatter = scattered_targets();

    let mut group = c.benchmark_group("meshgrid/locate");
    group.throughput(Throughput::Elements(QUERIES as u64));
    // ms-scale macro-benchmarks: flat sampling avoids the linear scheme's
    // per_iter × (1+…+sample_size) blow-up past the 5 s window.
    group.sampling_mode(SamplingMode::Flat);

    for &n in SITE_COUNTS {
        let m = concentrated_mesh(n);
        group.sample_size(SAMPLES); // reset after the slow variant lowered it

        // Coherent raster scan: the renderer's access pattern, the walk's best case.
        group.bench_with_input(BenchmarkId::new("cold-kdtree/raster", n), &m, |b, m| {
            b.iter(|| {
                let mut acc = 0usize;
                for j in 0..RASTER {
                    let y = min.1 + (j as f64 + 0.5) * step;
                    for i in 0..RASTER {
                        let x = min.0 + (i as f64 + 0.5) * step;
                        acc ^= m.cell_of(black_box(Point2D::new(x, y)));
                    }
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("warm-walk/raster", n), &m, |b, m| {
            b.iter(|| {
                let mut acc = 0usize;
                let mut row_seed = 0usize;
                for j in 0..RASTER {
                    let y = min.1 + (j as f64 + 0.5) * step;
                    let mut hint = row_seed;
                    for i in 0..RASTER {
                        let x = min.0 + (i as f64 + 0.5) * step;
                        let cell = m.cell_of_walk(black_box(Point2D::new(x, y)), hint);
                        if i == 0 {
                            row_seed = cell; // seed next row from this row's start
                        }
                        hint = cell;
                        acc ^= cell;
                    }
                }
                black_box(acc)
            });
        });

        // Incoherent scattered queries: the walk gets no useful warm-start hint.
        group.bench_with_input(BenchmarkId::new("cold-kdtree/scatter", n), &m, |b, m| {
            b.iter(|| {
                let mut acc = 0usize;
                for &q in &scatter {
                    acc ^= m.cell_of(black_box(q));
                }
                black_box(acc)
            });
        });

        group.sample_size(SAMPLES_SLOW);
        group.bench_with_input(BenchmarkId::new("warm-walk/scatter", n), &m, |b, m| {
            b.iter(|| {
                let mut acc = 0usize;
                let mut hint = 0usize;
                for &q in &scatter {
                    hint = m.cell_of_walk(black_box(q), hint);
                    acc ^= hint;
                }
                black_box(acc)
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_locate
}
criterion_main!(benches);
