// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Grid structures and discretized fields for lens modelling.
//!
//! This module provides the geometry and sampled-field types that the physics
//! layers ([`crate::lens`], [`crate::solver`]) build on. Everything is 2D and
//! `f64`; arrays are row-major and indexed `[j, i]` (row = y, column = x).
//!
//! # Geometry primitives
//!
//! The coordinate geometry, holding no field data:
//!
//! * [`Point2D`] — a point in the plane, with the vector arithmetic
//!   (add/sub/scale, dot, norm, polar, …) used throughout.
//! * [`BBox`] — an axis-aligned bounding box, i.e. a rectangular domain extent.
//! * [`Grid2D`] (module `uniform`) — the structured geometry of a regular grid:
//!   pixel counts, physical extents, derived pixel sizes, and lazily-computed
//!   coordinate/radius arrays. Backs [`PixelGrid`].
//! * [`VoronoiMesh`] (module `mesh`) — the unstructured analogue of [`Grid2D`]:
//!   generating sites, clipped Voronoi cells (polygon, area, centroid), and site
//!   adjacency. Backs [`MeshGrid`].
//!
//! # Scalar fields
//!
//! A scalar field pairs one of the geometries above with a value per cell.
//! [`ScalarField2D`] is the object-safe trait unifying the representations: cell
//! count, per-cell value/centre/area, domain bounds, and pointwise
//! [`sample`](ScalarField2D::sample), plus default aggregates (flux, sum, mean,
//! min, max). Downstream code can accept `&dyn ScalarField2D` and treat either
//! identically. Two implementations:
//!
//! * [`PixelGrid`] (module `uniform`) — a [`Grid2D`] paired with a
//!   [`FieldArray2D`] of samples: a field on a regular grid, and the main
//!   workhorse (constructors, element access, statistics, resampling).
//! * [`MeshGrid`] (module `mesh`) — the unstructured counterpart: one value per
//!   Voronoi cell, with the geometry shared via `Arc` so value-producing
//!   operations clone cheaply. A work in progress; fully adaptive (moving-mesh)
//!   grids are planned but not yet implemented.
//!
//! # Operators
//!
//! [`FieldOps`] extends [`ScalarField2D`] with the operator workflow shared by
//! both representations: continuous interpolation, gradient (and gradient
//! magnitude), Laplacian, and the Poisson solve $\nabla^2 \psi = f$ (configured
//! via [`PoissonConfig`]; passing $f = 2\kappa$ recovers the lens potential).
//! Each implementation uses its natural discretization — finite differences / FFT
//! for the grid, finite volume for the mesh — behind the common contract, so
//! generic `T: FieldOps` code runs on either. Being `Self`-valued it is a generic
//! bound, not a trait object. Representation-specific extras (selectable
//! [`DerivativeStencil`] orders, spectral transforms, the [`BilinearKernel`]) are
//! inherent methods on [`PixelGrid`].
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

mod adaptive;
mod bbox;
mod contour;
mod field;
mod kernel;
mod mesh;
mod ops;
mod point;
mod uniform;

pub use bbox::BBox;
pub use contour::ContourSegment;
pub use field::{CellAdjacency, ScalarField2D};
pub use kernel::Kernel;
pub use ops::{FieldOps, PoissonConfig};
pub use point::Point2D;
pub use uniform::{
    BilinearKernel, BoundaryCondition, DerivativeStencil, Grid2D, PixelGrid, SpectralJet,
    square_log_kernel,
};
// pub use adaptive::AdaGrid;
pub use mesh::{MeshCell, MeshGrid, VoronoiMesh, polygon_log_kernel};
