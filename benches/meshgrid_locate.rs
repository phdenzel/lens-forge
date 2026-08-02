// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-location throughput for [`MeshGrid`]: cold k-d tree vs warm-start walk.
//!
//! Both methods return the same cell (a `debug_assert` in the walk's tests
//! guards that); this measures the cost of a *raster scan* — the hot path in
//! `viz::render::raster`, which samples the field at every pixel in scanline
//! order.
//!
//! * **`cold-kdtree`** — one independent `O(log n)` k-d tree descent per pixel
//!   ([`MeshGrid::cell_of`]), the current `sample` path.
//! * **`warm-walk`** — a greedy walk on the Voronoi adjacency
//!   ([`MeshGrid::cell_of_walk`]) seeded with the previous pixel's cell, and
//!   each row seeded from the previous row's first cell. Consecutive pixels
//!   fall in the same or an adjacent cell, so each walk is ~`O(1)`.
//!
//! Sites are radially concentrated toward the domain centre (a stand-in for the
//! high-magnification clustering an adaptive lensing mesh targets), so the k-d
//! tree faces the non-uniform density it is supposed to handle well.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lens_forge::grid::{BBox, MeshGrid, Point2D};

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `n` sites on `[-1, 1]²`, radially concentrated toward the centre (`r = u²`).
fn concentrated_mesh(n: usize) -> MeshGrid {
    let mut rng = Lcg(0x5EED);
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

const SITE_COUNTS: [usize; 3] = [500, 2_000, 8_000];
const RASTER: usize = 256; // W = H

fn bench_locate(c: &mut Criterion) {
    for &n in &SITE_COUNTS {
        let m = concentrated_mesh(n);
        let (min, max) = ((-1.0, -1.0), (1.0, 1.0));
        let step = (max.0 - min.0) / RASTER as f64;

        let mut group = c.benchmark_group(format!("meshgrid/locate/N{n}"));
        group.throughput(Throughput::Elements((RASTER * RASTER) as u64));

        group.bench_function("cold-kdtree", |b| {
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

        group.bench_function("warm-walk", |b| {
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

        group.finish();
    }
}

criterion_group!(benches, bench_locate);
criterion_main!(benches);
