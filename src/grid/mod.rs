// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Grid structures and discretized fields for lens modelling.
//!
//! This module provides the geometry and sampled-field types that the physics
//! layers ([`crate::model`], [`crate::solver`]) build on. Everything is 2D and
//! `f64`; arrays are row-major and indexed `[j, i]` (row = y, column = x).
//!
//! # Core types
//!
//! * [`Point2D`] — a point in the plane, with the vector arithmetic
//!   (add/sub/scale, dot, norm, polar, bounding box, …) used throughout.
//! * [`Grid2D`] — the coordinate geometry of a regular grid: pixel counts,
//!   physical extents, and derived pixel sizes. Coordinate arrays (meshgrids,
//!   radius maps) are computed lazily on demand. Holds no field data itself.
//! * [`PixelGrid`] — a [`Grid2D`] paired with a [`FieldArray2D`] of samples,
//!   i.e. an actual scalar field on the plane. This is the main workhorse:
//!   constructors, element access, statistics, resampling, and interpolation.
//! * [`BilinearKernel`] — precomputed indices and weights for bilinear
//!   interpolation of a field at an arbitrary point (see
//!   [`Grid2D::bilinear_interp`]).
//!
//! # Differentiation
//!
//! [`PixelGrid`] gains derivative operators via inherent-`impl` blocks split
//! across two submodules (both operate on the same type):
//!
//! * **Finite differences** — [`PixelGrid::gradient`], [`PixelGrid::laplacian`],
//!   and their field/parallel variants, with a selectable
//!   [`DerivativeStencil`] (central 2nd/4th/6th order).
//! * **Spectral (FFT)** — [`PixelGrid::fft2d`] / [`PixelGrid::ifft2d`] plus
//!   `k`-space derivatives ([`PixelGrid::gradient_spectral`],
//!   [`PixelGrid::laplacian_spectral`]). Well suited to the periodic Poisson
//!   solves ($\nabla^2 \psi = 2\kappa$) of the pixelated lens.
//!
//! # Type aliases
//!
//! [`FieldArray2D`], [`VectorField2D`], [`ComplexArray1D`],
//! [`ComplexFieldArray2D`], and [`FourierSymbols`] name the `ndarray` types used
//! for scalar/vector fields and their Fourier transforms.
//!
//! Adaptive (non-uniform) grids are planned but not yet implemented.

/// A 2D scalar field sampled on a grid (row-major, indexed `[j, i]`).
pub type FieldArray2D = ndarray::Array2<f64>;
/// A 2D vector field as a pair of scalar-field components `(x, y)`.
pub type VectorField2D = (FieldArray2D, FieldArray2D);
/// A 1D complex array (e.g. a row/column spectrum).
pub type ComplexArray1D = ndarray::Array1<num::complex::Complex<f64>>;
/// A 2D complex field, e.g. the Fourier transform of a [`FieldArray2D`].
pub type ComplexFieldArray2D = ndarray::Array2<num::complex::Complex<f64>>;
/// Per-axis Fourier symbols (wavenumbers) as `(kx, ky)`.
pub type FourierSymbols = (ComplexArray1D, ComplexArray1D);

mod adaptive;
mod diff;
mod grid2d;
mod interp;
mod mesh;
mod point;
mod poisson;
mod spectral;
mod uniform;

pub use diff::DerivativeStencil;
pub use grid2d::Grid2D;
pub use interp::BilinearKernel;
pub use point::Point2D;
pub use poisson::PoissonConfig;
pub use uniform::PixelGrid;
// pub use adaptive::AdaGrid;
