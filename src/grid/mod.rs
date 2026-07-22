// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Grid structures and discretized fields for lens modelling.
//!
//! This module provides the geometry and sampled-field types that the physics
//! layers ([`crate::model`], [`crate::solver`]) build on. Everything is 2D and
//! `f64`; arrays are row-major and indexed `[j, i]` (row = y, column = x).
//!
//! # Geometry primitives
//!
//! * [`Point2D`] — a point in the plane, with the vector arithmetic
//!   (add/sub/scale, dot, norm, polar, bounding box, …) used throughout.
//! * [`Grid2D`] (module `uniform`) — the coordinate geometry of a regular grid:
//!   pixel counts, physical extents, and derived pixel sizes. Coordinate arrays
//!   (coordinate matrices, radius maps) are computed lazily on demand. Holds no
//!   field data itself; it backs [`PixelGrid`], the structured analogue of the
//!   mesh's `VoronoiMesh`.
//!
//! # Scalar fields
//!
//! [`ScalarField2D`] is the object-safe trait that unifies the field
//! representations: cell count, per-cell value/centre/area, domain bounds, and
//! pointwise [`sample`](ScalarField2D::sample), plus default aggregates (flux,
//! sum, mean, min, max). Downstream code can accept `&dyn ScalarField2D` and
//! treat either representation identically. [`FieldOps`] extends it with the
//! operator workflow — continuous interpolation, gradient, Laplacian and the
//! Poisson solve — for generic (`T: FieldOps`) code that must run on either
//! representation; being `Self`-valued it is a generic bound, not a trait
//! object. Two implementations:
//!
//! * [`PixelGrid`] (module `uniform`) — a [`Grid2D`] paired with a
//!   [`FieldArray2D`] of samples: a scalar field on a regular grid. The main
//!   workhorse (constructors, element access, statistics, resampling,
//!   interpolation) and the only fully-featured representation.
//! * [`MeshGrid`] (module `mesh`) — the unstructured counterpart: one value per
//!   cell of a Voronoi tessellation, with the geometry shared via `Arc` so
//!   value-producing operations clone cheaply. A work in progress. Fully
//!   adaptive (moving-mesh) grids are planned but not yet implemented.
//!
//! # Operators on [`PixelGrid`]
//!
//! [`PixelGrid`] gains operators from sibling submodules under `uniform`, all
//! inherent `PixelGrid` methods:
//!
//! * **Finite differences** — [`PixelGrid::gradient`], [`PixelGrid::laplacian`],
//!   and their field/parallel variants, with a selectable
//!   [`DerivativeStencil`] (central 2nd/4th/6th order).
//! * **Spectral (FFT)** — [`PixelGrid::fft2d`] / [`PixelGrid::ifft2d`] plus
//!   `k`-space derivatives ([`PixelGrid::gradient_spectral`],
//!   [`PixelGrid::laplacian_spectral`]).
//! * **Poisson** — [`PixelGrid::poisson_spectral`] / [`PixelGrid::poisson_cg`]
//!   (configured via [`PoissonConfig`]) solve $\nabla^2 \psi = f$; passing
//!   $f = 2\kappa$ recovers the lens potential of the pixelated lens.
//! * **Interpolation** — [`BilinearKernel`], precomputed indices and weights for
//!   bilinear sampling at an arbitrary point (see [`Grid2D::bilinear_interp`]).
//!
//! # Type aliases
//!
//! [`FieldArray1D`], [`FieldArray2D`], [`VectorField2D`], [`ComplexArray1D`],
//! [`ComplexFieldArray2D`], and [`FourierSymbols`] name the `ndarray` types used
//! for scalar/vector fields and their Fourier transforms. [`FieldArray1D`] backs
//! the mesh field ([`MeshGrid`]); [`FieldArray2D`] backs the regular grid
//! ([`PixelGrid`]).

/// A 1D scalar field sampled on an unstructured mesh (one value per cell).
pub type FieldArray1D = ndarray::Array1<f64>;
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

mod point;
mod bbox;
mod field;
mod ops;
mod uniform;
mod adaptive;
mod mesh;

pub use point::Point2D;
pub use bbox::BBox;
pub use field::ScalarField2D;
pub use ops::{FieldOps, PoissonConfig};
pub use uniform::{BilinearKernel, BoundaryCondition, DerivativeStencil, Grid2D, PixelGrid};
// pub use adaptive::AdaGrid;
pub use mesh::{MeshGrid, VoronoiMesh};
