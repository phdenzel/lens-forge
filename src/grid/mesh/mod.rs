// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Voronoi / Delaunay mesh grid — the free-form, unstructured counterpart to
//! [`PixelGrid`](crate::grid::PixelGrid).
//!
//! Work in progress. The end goal is `MeshGrid`: a set of sites carrying a
//! scalar value per Voronoi cell, plus the geometry (cell areas, adjacency,
//! point location) and operators needed to match `PixelGrid` usage behind a
//! shared scalar-field trait.
//!
//! Built bottom-up. Implemented so far:
//! * `predicates` — robust orientation and in-circle tests, the numerical
//!   foundation of the Delaunay triangulation.
//! * `triangulate` — incremental Delaunay triangulation with Lawson edge-flips.
//! * `voronoi` — the Voronoi dual: clipped cells (polygon, area, centroid),
//!   site adjacency, and Lloyd relaxation.
//! * `generators` — site-layout constructors (lattice, hexagonal, concentric,
//!   density-graded).
//! * `kdtree` — nearest-site point location.
//! * `meshgrid` — [`MeshGrid`], the public Voronoi field type.
//! * `faces` — shared Voronoi facet lengths, the geometry the finite-volume
//!   operators discretize over.
//! * `interp` — continuous interpolation (Laplace natural-neighbour, with a
//!   barycentric-linear fallback).
//! * `diff` — finite-volume gradient (least-squares) and Laplacian (TPFA).
//! * `poisson` — conjugate-gradient Poisson solver ($\nabla^2 \psi = f$).

mod generators;
mod kdtree;
mod predicates;
#[allow(dead_code)]
mod triangulate;
mod voronoi;
pub use voronoi::VoronoiMesh;

mod faces;
mod meshgrid;
pub use meshgrid::MeshGrid;

mod diff;
mod interp;
mod poisson;
