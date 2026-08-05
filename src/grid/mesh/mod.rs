// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Voronoi mesh grid — a free-form tessellation partitioning a plane into
//! polygonal cells.
//!
//! [`MeshGrid`] carries a scalar value per Voronoi cell, along with the geometry
//! (cell areas, adjacency, point location) and operators behind the shared
//! scalar-field trait.
//!
//! * `predicates` — robust orientation and in-circle tests, the numerical
//!   foundation of the Delaunay triangulation.
//! * `triangulate` — incremental Delaunay triangulation with Lawson edge-flips.
//! * `voronoi` — the Voronoi dual: clipped cells (polygon, area, centroid),
//!   site adjacency, and Lloyd relaxation.
//! * `generators` — site-layout constructors (lattice, hexagonal, concentric,
//!   density-graded).
//! * `meshgrid` — [`MeshGrid`], the public Voronoi field type.
//! * `faces` — shared Voronoi facet lengths, the geometry the finite-volume
//!   operators discretize over.
//! * `interp` — continuous interpolation (Laplace natural-neighbour, with a
//!   barycentric-linear fallback).
//! * `diff` — finite-volume gradient (least-squares) and Laplacian (TPFA).
//! * `poisson` — conjugate-gradient Poisson solver ($\nabla^2 \psi = f$).

mod generators;
mod predicates;
mod triangulate;
mod voronoi;
pub use voronoi::VoronoiMesh;

mod faces;
mod meshgrid;
pub use meshgrid::{MeshCell, MeshGrid};

mod contour;
mod diff;
mod fieldops;
mod interp;
mod poisson;
