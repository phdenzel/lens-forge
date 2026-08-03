// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Head-to-head benchmarks of the fast-summation backends for the lensing
//! deflection field.
//!
//! Two groups per backend, swept over source counts:
//!
//! * **`build`** — the tree/precompute construction cost.
//! * **`query`** — a batched deflection query over a fixed grid of targets.
//!
//! Accuracy against the exact direct sum lives in the unit tests; this measures
//! speed. The `O(N²)` direct sum is skipped above 5k to bound the run;
//! `--features full-bench` widens the source-count sweep.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};
use lens_forge::Lcg;
use lens_forge::fastsum::{Monopole, SolverConfig, SolverKind, build_solver};
use lens_forge::grid::Point2D;

const TARGET_SIDE: usize = 64; // 64×64 = 4096 query targets

#[cfg(feature = "full-bench")]
const SOURCE_COUNTS: &[usize] = &[1_024, 8_192, 16_384];
#[cfg(not(feature = "full-bench"))]
const SOURCE_COUNTS: &[usize] = &[8_192];

fn sources(n: usize) -> Vec<Monopole> {
    let mut rng = Lcg::new(0xC0FFEE);
    (0..n)
        .map(|_| {
            Monopole::new(
                Point2D::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)),
                rng.range(0.0, 0.5),
            )
        })
        .collect()
}

/// A `side × side` grid of query targets covering the field of view.
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

fn kind_name(kind: SolverKind) -> &'static str {
    match kind {
        SolverKind::Direct => "direct",
        SolverKind::BarnesHut => "barnes-hut",
        SolverKind::Fmm => "fmm",
        SolverKind::Kifmm => "kifmm",
    }
}

/// Backends benched at source count `n`: the `O(N²)` direct sum is skipped above
/// 5k to keep the run bounded; the tree/expansion backends run at every `n`.
fn kinds_for(n: usize) -> &'static [SolverKind] {
    if n <= 5_000 {
        &[
            SolverKind::Direct,
            SolverKind::BarnesHut,
            SolverKind::Fmm,
            SolverKind::Kifmm,
        ]
    } else {
        &[SolverKind::BarnesHut, SolverKind::Fmm, SolverKind::Kifmm]
    }
}

fn bench_build(c: &mut Criterion) {
    let cfg = SolverConfig::default().with_theta(0.5);
    let mut group = c.benchmark_group("fastsum/build");
    group.sampling_mode(SamplingMode::Flat);
    // Builds are slow (kifmm ~seconds) but near-deterministic, so few samples.
    group.sample_size(10);
    for &n in SOURCE_COUNTS {
        let src = sources(n);
        for &kind in kinds_for(n) {
            group.bench_with_input(BenchmarkId::new(kind_name(kind), n), &src, |b, s| {
                b.iter(|| black_box(build_solver(kind, black_box(s), cfg)));
            });
        }
    }
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let cfg = SolverConfig::default().with_theta(0.5);
    let query = targets(TARGET_SIDE);
    let mut group = c.benchmark_group("fastsum/query");
    group.sampling_mode(SamplingMode::Flat);
    for &n in SOURCE_COUNTS {
        let src = sources(n);
        for &kind in kinds_for(n) {
            let solver = build_solver(kind, &src, cfg);
            group.bench_with_input(BenchmarkId::new(kind_name(kind), n), &query, |b, q| {
                b.iter(|| black_box(solver.jet_batch(black_box(q))));
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_build, bench_query
}
criterion_main!(benches);
