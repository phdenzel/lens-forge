// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Uniform (regular pixel) grid — the structured counterpart to
//! [`MeshGrid`](crate::grid::MeshGrid).
//!
//! [`PixelGrid`] carries the grid geometry and its sampled field; its operators
//! live in sibling submodules as inherent `PixelGrid` methods:
//! * `grid2d` — [`Grid2D`], the regular-grid coordinate geometry backing
//!   [`PixelGrid`] (the structured analogue of `VoronoiMesh` in `mesh`).
//! * `pixelgrid` — [`PixelGrid`], the public uniform field type.
//! * `interp` — [`BilinearKernel`] bilinear interpolation.
//! * `spectral` — FFT transforms and `k`-space derivatives.
//! * `diff` — finite-difference derivatives (gradient, Laplacian).
//! * `poisson` — Poisson solvers ($\nabla^2 \psi = f$).

mod grid2d;
mod pixelgrid;
mod interp;
mod spectral;
mod diff;
mod poisson;
mod fieldops;

pub use grid2d::Grid2D;
pub use pixelgrid::PixelGrid;
pub use interp::BilinearKernel;
pub use diff::{BoundaryCondition, DerivativeStencil};
pub use poisson::PoissonConfig;
