// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Head-to-head benchmarks of the fast-summation backends for the lensing
//! deflection field.
//!
//! Times both the build (tree/precompute) and a batched query over a grid of
//! targets, per backend, across a range of source counts. Accuracy comparisons
//! against the exact direct sum live in the unit tests; this measures speed.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lens_forge::grid::Point2D;
use lens_forge::fastsum::{build_solver, Monopole, SolverConfig, SolverKind};

/// Deterministic LCG so benches are reproducible without an rng dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn sources(n: usize) -> Vec<Monopole> {
    let mut rng = Lcg(0xC0FFEE);
    (0..n)
        .map(|_| {
            Monopole::new(
                Point2D::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)),
                rng.range(0.0, 0.5),
            )
        })
        .collect()
}

/// A square grid of query targets covering the field of view.
fn targets(side: usize) -> Vec<Point2D> {
    let mut out = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let x = -2.0 + 4.0 * (i as f64) / (side as f64 - 1.0);
            let y = -2.0 + 4.0 * (j as f64) / (side as f64 - 1.0);
            out.push(Point2D::new(x, y));
        }
    }
    out
}

fn bench_backends(c: &mut Criterion) {
    let cfg = SolverConfig::default().with_theta(0.5);
    let query = targets(64); // 4096 targets

    for &n in &[1_000usize, 5_000, 20_000] {
        let src = sources(n);

        // Skip the O(N²) direct sum at the largest size to keep the run bounded.
        let kinds: &[SolverKind] = if n <= 5_000 {
            &[SolverKind::Direct, SolverKind::BarnesHut]
        } else {
            &[SolverKind::BarnesHut]
        };

        let mut group = c.benchmark_group(format!("fastsum/query_N{n}"));
        for &kind in kinds {
            let solver = build_solver(kind, &src, cfg);
            group.bench_with_input(BenchmarkId::new(solver.name(), n), &query, |b, q| {
                b.iter(|| black_box(solver.field_batch(black_box(q))));
            });
        }
        group.finish();

        let mut build = c.benchmark_group(format!("fastsum/build_N{n}"));
        for &kind in kinds {
            build.bench_with_input(BenchmarkId::new(kind_name(kind), n), &src, |b, s| {
                b.iter(|| black_box(build_solver(kind, black_box(s), cfg)));
            });
        }
        build.finish();
    }
}

fn kind_name(kind: SolverKind) -> &'static str {
    match kind {
        SolverKind::Direct => "direct",
        SolverKind::BarnesHut => "barnes-hut",
    }
}

criterion_group!(benches, bench_backends);
criterion_main!(benches);
