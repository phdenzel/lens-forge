<!--
SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
SPDX-FileNotice: Part of lens-forge
SPDX-License-Identifier: GPL-3.0-or-later
-->

# lens-forge

[![CI](https://github.com/phdenzel/lens-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/phdenzel/lens-forge/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/phdenzel/lens-forge/branch/main/graph/badge.svg?token=R7QX8UEROS)](https://codecov.io/gh/phdenzel/lens-forge)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://phdenzel.github.io/lens-forge/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

A blazingly fast Rust-based gravitational-lens modelling library.

`lens-forge` targets **both** strong-lensing use cases:

- **Forward simulation** — given a lens mass distribution and a source, produce
  lensed images, magnification maps, critical curves, and caustics.
- **Inverse reconstruction** — given observed lensed images (with PSF and noise),
  reconstruct the lens mass and/or source light.

Both the lens and the source support **parametric** (analytic profiles) and
**free-form** (pixelated / grid-based) representations behind a common trait, so
solvers, rendering, and inference are agnostic to the underlying representation.


## Status

Early development. The numerical and visualization foundations are in place; the
physics and solver layers are being built out.

| Area | Module | State |
| --- | --- | --- |
| 2D grids, pixel fields, FFT, gradients, Laplacian, interpolation | `grid` | mature (~5.5k LOC, well tested) |
| Color mapping, scaling, rasterization, live `winit`/`wgpu` viewer | `viz` | mature |
| Lens & source models (profiles + pixelated) | `model` | in progress |
| Forward/inverse lens-equation solvers | `solver` | in progress |


## Build

The project uses `wgpu`/`winit` for the viewer. A Nix flake provides the dev
environment with all system dependencies (Vulkan/Metal, `libiconv`, X11/Wayland
on Linux):

```sh
nix develop        # enter the dev shell
cargo build
cargo test
```

Without Nix, ensure the platform equivalents of the flake's `buildInputs` are
available to the linker (notably `libiconv` on macOS).


### Examples

```sh
cargo run --example pixel_grid_viz
cargo run --example gradient_viz
```


## Benchmarks

Performance is tracked with [Criterion](https://github.com/bheisler/criterion.rs)
(statistical benchmarking with HTML reports written to `target/criterion/`; open
`target/criterion/report/index.html` to compare methods within a suite).

```sh
cargo bench                          # all suites, quick representative sweep
cargo bench --features full-bench    # exhaustive sweep (all sizes/stencils; slow)
cargo bench --bench pixelgrid_ops    # run one suite
cargo bench -- pixelgrid/gradient    # filter by benchmark id
```

Benches that support it run a small representative subset by default; the
`full-bench` feature widens their parameter sweep for a thorough (but much
slower) run.

| Bench | Measures |
| --- | --- |
| `viz_scale` | Throughput of the `viz` render-scaling (`RenderScale`: linear/log10/symlog) and `Normalize` operations, comparing five strategies — per-value, slice, in-place, and their Rayon-parallel slice/in-place variants — across array sizes (100 up to 1M under `full-bench`). Locates the serial/parallel crossover per mode: parallel never pays off for the cheap linear/normalize paths, and crosses ~30k for the transcendental log10/symlog modes — which sets `RenderOptions::threshold_parallel`. Also times `auto_range`, the serial min/max reduction behind range determination. |
| `pixelgrid_ops` | Throughput of `PixelGrid` operators. **Gradient/Laplacian:** FD central stencils (2nd/4th/6th order) vs their FD-equivalent spectral counterparts vs the *exact* spectral derivative, over 64²–256². For the *same* discrete operator FD wins (spectral does an FFT to get the same numbers); the exact-spectral route is the one that buys accuracy (~10 orders of magnitude on smooth periodic data — see the tests in `src/grid/spectral.rs`). **Poisson (`∇²ψ = f`):** one FFT divide-by-`−k²` vs a finite-difference conjugate-gradient solve — the regime where the FFT wins decisively (~11× at 64², widening with size). **Interp:** serial vs Rayon-parallel bilinear resampling at a raster of query points. |
| `meshgrid_ops` | Throughput of `MeshGrid` (unstructured Voronoi) construction and operators, swept over site counts. **Build:** full mesh construction (Bowyer–Watson triangulation + the dual Voronoi tessellation and adjacency) from a raw site cloud. **Gradient/Laplacian:** the two-point-flux finite-volume operators on a smooth field. **Poisson (`∇²ψ = f`):** a Jacobi-preconditioned CG solve (Dirichlet). **Interp:** barycentric resampling at a raster of query points, serial vs Rayon-parallel. Sites are centre-concentrated, mirroring an adaptive lensing mesh. |
| `meshgrid_locate` | Point-location throughput for `MeshGrid`: a cold `O(log n)` k-d tree query per point vs a warm-start greedy walk on the Voronoi adjacency, each under two query orderings — a coherent **raster** scan (the `viz` render hot path, the walk's best case) and **scattered** random points (where the walk degrades toward `O(√n)` and the cold tree wins). Swept over site counts. |
| `fastsum_bench` | Head-to-head speed of the fast-summation backends (`DirectSum`, `BarnesHut`, `Fmm`, `Kifmm`) for the lensing deflection field. Times both the build (tree/precompute) and a batched query over a 64×64 grid of targets, sweeping source counts (1k/5k/20k); the exact `O(N²)` direct sum is skipped above 5k to bound the run. Accuracy is verified separately in the unit tests — this suite is speed only. |

`fastsum_bench` and `meshgrid_locate` use a small deterministic LCG for
reproducible inputs (no `rand` dependency).


## License

GPL-3.0-or-later. Copyright © 2025-present Philipp Denzel.
