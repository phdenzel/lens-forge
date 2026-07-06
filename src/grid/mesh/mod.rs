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
//! * `voronoi` — the Voronoi dual: clipped cells (polygon, area, centroid) and
//!   site adjacency.
//! * `kdtree` — nearest-site point location.
//! * `meshgrid` — [`MeshGrid`], the public Voronoi field type.

mod kdtree;
mod predicates;
#[allow(dead_code)]
mod triangulate;
mod voronoi;

mod meshgrid;
pub use meshgrid::MeshGrid;
