// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Throughput benchmarks for the `viz` render-scaling and normalization ops.
//!
//! For [`RenderScale`] (linear/log10/symlog) and [`Normalize`], compares five
//! strategies — per-value, slice, in-place, and the Rayon-parallel slice/in-place
//! variants (parallel only above a size threshold) — across array sizes, to guide
//! which path the viewer should use at a given size. `--features full-bench`
//! widens the size sweep.

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use lens_forge::viz::scale::{Normalize, RenderScale};
use std::hint::black_box;

/// Array sizes swept as the benchmark id: a small representative subset by
/// default, the full range under `--features full-bench`. The full sweep
/// brackets both serial/parallel crossovers — ~10k–100k for the transcendental
/// modes (log10/symlog), well above 100k for the cheap ones (linear/normalize).
#[cfg(feature = "full-bench")]
const SIZES: &[usize] = &[100, 1_000, 10_000, 30_000, 100_000, 300_000, 1_000_000];
#[cfg(not(feature = "full-bench"))]
const SIZES: &[usize] = &[1_000, 100_000];

/// Below this size the Rayon-parallel variants lose to thread overhead, so they
/// are only benchmarked at or above it.
const PAR_THRESHOLD: usize = 10_000;

/* ============ Test data ================================================== */
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

/* ============ Strategy sweeps ============================================= */
/// The five [`RenderScale`] strategies over each size in [`SIZES`].
fn bench_scale(c: &mut Criterion, name: &str, scale: RenderScale, data_fn: fn(usize) -> Vec<f32>) {
    let fallback = 0.0;
    let mut group = c.benchmark_group(name);
    group.sampling_mode(SamplingMode::Flat); // ms-scale at large sizes
    for &size in SIZES {
        let data = data_fn(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("per_value", size), &size, |b, _| {
            b.iter(|| {
                let result: Vec<f32> = data
                    .iter()
                    .map(|&v| black_box(scale.apply_or(v, fallback)))
                    .collect();
                black_box(result);
            });
        });
        group.bench_with_input(BenchmarkId::new("slice", size), &size, |b, _| {
            b.iter(|| {
                let mut output = vec![0.0; size];
                scale.apply_slice(black_box(&data), &mut output, fallback);
                black_box(output);
            });
        });
        group.bench_with_input(BenchmarkId::new("inplace", size), &size, |b, _| {
            b.iter(|| {
                let mut values = data.clone();
                scale.apply_slice_inplace(black_box(&mut values), fallback);
                black_box(values);
            });
        });

        if size >= PAR_THRESHOLD {
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

/// The five [`Normalize`] strategies over each size in [`SIZES`].
fn bench_norm(c: &mut Criterion, name: &str, norm: Normalize, data_fn: fn(usize) -> Vec<f32>) {
    let mut group = c.benchmark_group(name);
    group.sampling_mode(SamplingMode::Flat); // ms-scale at large sizes
    for &size in SIZES {
        let data = data_fn(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("per_value", size), &size, |b, _| {
            b.iter(|| {
                let result: Vec<f32> = data.iter().map(|&v| black_box(norm.apply(v))).collect();
                black_box(result);
            });
        });
        group.bench_with_input(BenchmarkId::new("slice", size), &size, |b, _| {
            b.iter(|| {
                let mut output = vec![0.0; size];
                norm.apply_slice(black_box(&data), &mut output);
                black_box(output);
            });
        });
        group.bench_with_input(BenchmarkId::new("inplace", size), &size, |b, _| {
            b.iter(|| {
                let mut values = data.clone();
                norm.apply_slice_inplace(black_box(&mut values));
                black_box(values);
            });
        });

        if size >= PAR_THRESHOLD {
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

/* ============ RenderScale benchmarks ===================================== */
fn bench_render_scale_linear(c: &mut Criterion) {
    bench_scale(c, "render_scale_linear", RenderScale::Linear, generate_test_data);
}

fn bench_render_scale_log10(c: &mut Criterion) {
    bench_scale(c, "render_scale_log10", RenderScale::Log10, generate_test_data);
}

fn bench_render_scale_symlog(c: &mut Criterion) {
    let scale = RenderScale::SymLog { linthresh: 2.0 };
    bench_scale(c, "render_scale_symlog", scale, generate_test_data_mixed);
}

/* ============ Normalize benchmarks ======================================= */
fn bench_normalize(c: &mut Criterion) {
    let norm = Normalize::default()
        .with_lims(0.0, 100.0)
        .unwrap()
        .with_clamp(true);
    bench_norm(c, "normalize", norm, generate_test_data);
}

fn bench_normalize_unclamped(c: &mut Criterion) {
    let norm = Normalize::default()
        .with_lims(0.0, 100.0)
        .unwrap()
        .with_clamp(false);
    bench_norm(c, "normalize_unclamped", norm, generate_test_data_mixed);
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
