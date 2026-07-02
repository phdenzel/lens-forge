<!--
SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
SPDX-FileNotice: Part of lens-forge
SPDX-License-Identifier: GPL-3.0-or-later
-->

# lens-forge

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
(statistical benchmarking with HTML reports written to `target/criterion/`).

```sh
cargo bench                         # run all benchmark suites
cargo bench --bench fastsum_bench   # run one suite
cargo bench -- fastsum/query_N1000  # filter by benchmark id
```

| Bench | Measures |
| --- | --- |
| `fastsum_bench` | Head-to-head speed of the fast-summation backends (`DirectSum` vs `BarnesHut`) for the lensing deflection field. Times both the build (tree/precompute) and a batched query over a 64×64 grid of targets, sweeping source counts (1k/5k/20k); the exact `O(N²)` direct sum is skipped at 20k to bound the run. Accuracy is verified separately in the unit tests — this suite is speed only. |
| `scale_bench` | Throughput of the `viz` render-scaling (`RenderScale`: linear/log10/symlog) and `Normalize` operations, comparing per-value, slice, in-place, and Rayon-parallel implementations across array sizes (100–100k). Guides which code path the viewer should use at a given size. |
| `pixelgrid_bench` | Throughput of `PixelGrid` derivatives and Poisson solves. **Gradient/Laplacian:** FD central stencils (2nd/4th/6th order) vs their FD-equivalent spectral counterparts vs the *exact* spectral derivative, over 64²–256². For the *same* discrete operator FD wins (spectral does an FFT to get the same numbers); the exact-spectral route is the one that buys accuracy (~10 orders of magnitude on smooth periodic data — see the tests in `src/grid/spectral.rs`). **Poisson (`∇²ψ = f`):** one FFT divide-by-`−k²` vs a finite-difference conjugate-gradient solve — the regime where the FFT wins decisively (~11× at 64², widening with size). |

`fastsum_bench` uses a small deterministic LCG for reproducible inputs (no
`rand` dependency).


## License

GPL-3.0-or-later. Copyright © 2025-present Philipp Denzel.
