// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Uniform (regular pixel) grid — a lattice of uniform and square cells.
//!
//! [`PixelGrid`] carries the grid geometry and its sampled field, along with the
//! operators behind the shared scalar-field trait.
//!
//! * `grid2d` — [`Grid2D`], the regular-grid coordinate geometry backing
//!   [`PixelGrid`] (the structured analogue of `VoronoiMesh` in `mesh`).
//! * `pixelgrid` — [`PixelGrid`], the public uniform field type.
//! * `interp` — [`BilinearKernel`] bilinear interpolation.
//! * `spectral` — FFT transforms and `k`-space derivatives.
//! * `diff` — finite-difference derivatives (gradient, Laplacian).
//! * `poisson` — Poisson solvers ($\nabla^2 \psi = f$).

mod contour;
mod diff;
mod fieldops;
mod grid2d;
mod interp;
mod pixelgrid;
mod poisson;
mod spectral;

pub use diff::{BoundaryCondition, DerivativeStencil};
pub use grid2d::Grid2D;
pub use interp::BilinearKernel;
pub use pixelgrid::PixelGrid;
