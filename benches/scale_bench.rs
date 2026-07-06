// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Throughput benchmarks for the `viz` render-scaling and normalization ops.
//!
//! Compares four implementation strategies — per-value, slice, in-place, and
//! Rayon-parallel — for [`RenderScale`] (linear/log10/symlog) and [`Normalize`]
//! across array sizes from 100 to 100k, to guide which path the viewer should
//! use at a given size. Uses simple deterministic input data (no rng).

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lens_forge::viz::scale::{Normalize, RenderScale};


/* ============ Helper functions =========================================== */
fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.01).collect()
}

fn generate_test_data_mixed(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let v = (i as f32) * 0.01;
            if i % 100 == 0 {
                f32::NAN
            } else if i % 200 == 0 {
                -v
            } else {
                v
            }
        })
        .collect()
}


/* ============ RenderScale benchmarks ===================================== */
fn bench_render_scale_linear(c: &mut Criterion) {
    let scale = RenderScale::Linear;
    let sizes = vec![100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("render_scale_linear");

    for size in sizes {
        let data = generate_test_data(size);
        let fallback = 0.0;

        group.throughput(Throughput::Elements(size as u64));

        // Per-value approach
        group.bench_with_input(BenchmarkId::new("per_value", size), &size, |b, _| {
            b.iter(|| {
                let result: Vec<f32> = data
                    .iter()
                    .map(|&v| black_box(scale.apply_or(v, fallback)))
                    .collect();
                black_box(result);
            });
        });

        // Slice approach
        group.bench_with_input(BenchmarkId::new("slice", size), &size, |b, _| {
            b.iter(|| {
                let mut output = vec![0.0; size];
                scale.apply_slice(black_box(&data), &mut output, fallback);
                black_box(output);
            });
        });

        // Inplace approach
        group.bench_with_input(BenchmarkId::new("inplace", size), &size, |b, _| {
            b.iter(|| {
                let mut values = data.clone();
                scale.apply_slice_inplace(black_box(&mut values), fallback);
                black_box(values);
            });
        });

        // Parallel slice approach (only for larger sizes)
        if size >= 10000 {
            group.bench_with_input(BenchmarkId::new("par_slice", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0; size];
                    scale.par_apply_slice(black_box(&data), &mut output, fallback);
                    black_box(output);
                });
            });

            group.bench_with_input(BenchmarkId::new("par_inplace", size), &size, |b, _| {
                b.iter(|| {
                    let mut values = data.clone();
                    scale.par_apply_slice_inplace(black_box(&mut values), fallback);
                    black_box(values);
                });
            });
        }
    }

    group.finish();
}

fn bench_render_scale_log10(c: &mut Criterion) {
    let scale = RenderScale::Log10;
    let sizes = vec![100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("render_scale_log10");

    for size in sizes {
        let data = generate_test_data(size);
        let fallback = 0.0;

        group.throughput(Throughput::Elements(size as u64));

        // Per-value approach
        group.bench_with_input(BenchmarkId::new("per_value", size), &size, |b, _| {
            b.iter(|| {
                let result: Vec<f32> = data
                    .iter()
                    .map(|&v| black_box(scale.apply_or(v, fallback)))
                    .collect();
                black_box(result);
            });
        });

        // Slice approach
        group.bench_with_input(BenchmarkId::new("slice", size), &size, |b, _| {
            b.iter(|| {
                let mut output = vec![0.0; size];
                scale.apply_slice(black_box(&data), &mut output, fallback);
                black_box(output);
            });
        });

        // Inplace approach
        group.bench_with_input(BenchmarkId::new("inplace", size), &size, |b, _| {
            b.iter(|| {
                let mut values = data.clone();
                scale.apply_slice_inplace(black_box(&mut values), fallback);
                black_box(values);
            });
        });

        // Parallel approaches for larger sizes
        if size >= 10000 {
            group.bench_with_input(BenchmarkId::new("par_slice", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0; size];
                    scale.par_apply_slice(black_box(&data), &mut output, fallback);
                    black_box(output);
                });
            });
        }
    }

    group.finish();
}

fn bench_render_scale_symlog(c: &mut Criterion) {
    let scale = RenderScale::SymLog { linthresh: 2.0 };
    let sizes = vec![100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("render_scale_symlog");

    for size in sizes {
        let data = generate_test_data_mixed(size);
        let fallback = 0.0;

        group.throughput(Throughput::Elements(size as u64));

        // Per-value approach
        group.bench_with_input(BenchmarkId::new("per_value", size), &size, |b, _| {
            b.iter(|| {
                let result: Vec<f32> = data
                    .iter()
                    .map(|&v| black_box(scale.apply_or(v, fallback)))
                    .collect();
                black_box(result);
            });
        });

        // Slice approach
        group.bench_with_input(BenchmarkId::new("slice", size), &size, |b, _| {
            b.iter(|| {
                let mut output = vec![0.0; size];
                scale.apply_slice(black_box(&data), &mut output, fallback);
                black_box(output);
            });
        });

        // Inplace approach
        group.bench_with_input(BenchmarkId::new("inplace", size), &size, |b, _| {
            b.iter(|| {
                let mut values = data.clone();
                scale.apply_slice_inplace(black_box(&mut values), fallback);
                black_box(values);
            });
        });

        // Parallel approaches for larger sizes
        if size >= 10000 {
            group.bench_with_input(BenchmarkId::new("par_slice", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0; size];
                    scale.par_apply_slice(black_box(&data), &mut output, fallback);
                    black_box(output);
                });
            });
        }
    }

    group.finish();
}


/* ============ Normalize benchmarks ======================================= */
fn bench_normalize(c: &mut Criterion) {
    let norm = Normalize::default()
        .with_lims(0.0, 100.0)
        .unwrap()
        .with_clamp(true);
    let sizes = vec![100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("normalize");

    for size in sizes {
        let data = generate_test_data(size);

        group.throughput(Throughput::Elements(size as u64));

        // Per-value approach
        group.bench_with_input(BenchmarkId::new("per_value", size), &size, |b, _| {
            b.iter(|| {
                let result: Vec<f32> = data.iter().map(|&v| black_box(norm.apply(v))).collect();
                black_box(result);
            });
        });

        // Slice approach
        group.bench_with_input(BenchmarkId::new("slice", size), &size, |b, _| {
            b.iter(|| {
                let mut output = vec![0.0; size];
                norm.apply_slice(black_box(&data), &mut output);
                black_box(output);
            });
        });

        // Inplace approach
        group.bench_with_input(BenchmarkId::new("inplace", size), &size, |b, _| {
            b.iter(|| {
                let mut values = data.clone();
                norm.apply_slice_inplace(black_box(&mut values));
                black_box(values);
            });
        });

        // Parallel approaches for larger sizes
        if size >= 10000 {
            group.bench_with_input(BenchmarkId::new("par_slice", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0; size];
                    norm.par_apply_slice(black_box(&data), &mut output);
                    black_box(output);
                });
            });

            group.bench_with_input(BenchmarkId::new("par_inplace", size), &size, |b, _| {
                b.iter(|| {
                    let mut values = data.clone();
                    norm.par_apply_slice_inplace(black_box(&mut values));
                    black_box(values);
                });
            });
        }
    }

    group.finish();
}

fn bench_normalize_unclamped(c: &mut Criterion) {
    let norm = Normalize::default()
        .with_lims(0.0, 100.0)
        .unwrap()
        .with_clamp(false);
    let sizes = vec![100, 1000, 10000, 100000];

    let mut group = c.benchmark_group("normalize_unclamped");

    for size in sizes {
        let data = generate_test_data_mixed(size);

        group.throughput(Throughput::Elements(size as u64));

        // Per-value approach
        group.bench_with_input(BenchmarkId::new("per_value", size), &size, |b, _| {
            b.iter(|| {
                let result: Vec<f32> = data.iter().map(|&v| black_box(norm.apply(v))).collect();
                black_box(result);
            });
        });

        // Slice approach
        group.bench_with_input(BenchmarkId::new("slice", size), &size, |b, _| {
            b.iter(|| {
                let mut output = vec![0.0; size];
                norm.apply_slice(black_box(&data), &mut output);
                black_box(output);
            });
        });

        // Parallel approaches for larger sizes
        if size >= 10000 {
            group.bench_with_input(BenchmarkId::new("par_slice", size), &size, |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0; size];
                    norm.par_apply_slice(black_box(&data), &mut output);
                    black_box(output);
                });
            });
        }
    }

    group.finish();
}


/* ============ Criterion configuration ==================================== */
criterion_group!(
    benches,
    bench_render_scale_linear,
    bench_render_scale_log10,
    bench_render_scale_symlog,
    bench_normalize,
    bench_normalize_unclamped,
);

criterion_main!(benches);
