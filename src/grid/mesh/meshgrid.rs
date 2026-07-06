// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! `MeshGrid`: a scalar field on a Voronoi tessellation.
//!
//! The unstructured counterpart to [`PixelGrid`](crate::grid::PixelGrid): a
//! [`VoronoiMesh`] geometry paired with one value per cell. The immutable
//! geometry is shared via [`Arc`], so value-producing operations (`scale`,
//! `map`, …) clone cheaply without duplicating the tessellation.

use std::sync::Arc;

use super::voronoi::VoronoiMesh;
use crate::grid::{Point2D, ScalarField2D};
use crate::{LensForgeError, Result};

/// A scalar field sampled on a Voronoi tessellation (one value per cell).
#[derive(Clone)]
pub struct MeshGrid {
    mesh: Arc<VoronoiMesh>,
    values: Vec<f64>,
}

impl MeshGrid {
    /// Build a mesh grid from `sites` clipped to `[min, max]`, with one value
    /// per site.
    ///
    /// # Errors
    /// Returns an error if `values.len() != sites.len()` or the geometry is
    /// invalid (empty sites, `min >= max`, non-finite sites).
    pub fn new(sites: &[Point2D], min: Point2D, max: Point2D, values: Vec<f64>) -> Result<Self> {
        if values.len() != sites.len() {
            return Err(LensForgeError::dimension_mismatch(
                format!("{} values", sites.len()),
                format!("{} values", values.len()),
            ));
        }
        let mesh = VoronoiMesh::new(sites, min, max)?;
        Ok(Self { mesh: Arc::new(mesh), values })
    }

    /// Mesh grid of zeros over `sites`.
    pub fn zeros(sites: &[Point2D], min: Point2D, max: Point2D) -> Result<Self> {
        Self::new(sites, min, max, vec![0.0; sites.len()])
    }

    /// Mesh grid with a constant value over `sites`.
    pub fn constant(sites: &[Point2D], min: Point2D, max: Point2D, value: f64) -> Result<Self> {
        Self::new(sites, min, max, vec![value; sites.len()])
    }

    /// Mesh grid sampling `f` at each site.
    pub fn from_point_func<F: Fn(Point2D) -> f64>(
        sites: &[Point2D],
        min: Point2D,
        max: Point2D,
        f: F,
    ) -> Result<Self> {
        let values = sites.iter().map(|&s| f(s)).collect();
        Self::new(sites, min, max, values)
    }

    /// Reuse an existing geometry with a new set of values (shares the
    /// tessellation, no rebuild).
    pub fn with_values(&self, values: Vec<f64>) -> Result<Self> {
        if values.len() != self.values.len() {
            return Err(LensForgeError::dimension_mismatch(
                format!("{} values", self.values.len()),
                format!("{} values", values.len()),
            ));
        }
        Ok(Self { mesh: self.mesh.clone(), values })
    }

    /* ---- access ---- */

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there are no cells.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Cell values as a slice.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Value of cell `i`.
    pub fn value(&self, i: usize) -> f64 {
        self.values[i]
    }

    /// Set the value of cell `i`.
    pub fn set(&mut self, i: usize, value: f64) {
        self.values[i] = value;
    }

    /// Generating site of cell `i`.
    pub fn site(&self, i: usize) -> Point2D {
        self.mesh.sites[i]
    }

    /// Area-weighted centroid of cell `i`.
    pub fn centroid(&self, i: usize) -> Point2D {
        self.mesh.cells[i].centroid
    }

    /// Area of cell `i`.
    pub fn area(&self, i: usize) -> f64 {
        self.mesh.cells[i].area
    }

    /// Clipped polygon (counter-clockwise) of cell `i`.
    pub fn cell_polygon(&self, i: usize) -> &[Point2D] {
        &self.mesh.cells[i].polygon
    }

    /// Voronoi neighbours of cell `i`.
    pub fn neighbors(&self, i: usize) -> &[usize] {
        &self.mesh.neighbors[i]
    }

    /// Index of the cell containing `p` (nearest site).
    pub fn cell_of(&self, p: Point2D) -> usize {
        self.mesh.cell_of(p)
    }

    /// Total covered area (equals the domain area).
    pub fn total_area(&self) -> f64 {
        self.mesh.total_area()
    }

    /* ---- value operations ---- */

    /// A copy with every value scaled by `factor` (shares geometry).
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            mesh: self.mesh.clone(),
            values: self.values.iter().map(|v| v * factor).collect(),
        }
    }

    /// A copy with `scalar` added to every value (shares geometry).
    pub fn shift(&self, scalar: f64) -> Self {
        Self {
            mesh: self.mesh.clone(),
            values: self.values.iter().map(|v| v + scalar).collect(),
        }
    }

    /// A copy with `f` applied to every value (shares geometry).
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self {
            mesh: self.mesh.clone(),
            values: self.values.iter().map(|&v| f(v)).collect(),
        }
    }

    /// Apply `f` to every value in place.
    pub fn apply<F: Fn(f64) -> f64>(&mut self, f: F) {
        for v in &mut self.values {
            *v = f(*v);
        }
    }
}

impl ScalarField2D for MeshGrid {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn value_at(&self, i: usize) -> f64 {
        self.values[i]
    }

    fn center(&self, i: usize) -> Point2D {
        self.mesh.sites[i]
    }

    fn cell_area(&self, i: usize) -> f64 {
        self.mesh.cells[i].area
    }

    fn bounds(&self) -> (Point2D, Point2D) {
        (self.mesh.min, self.mesh.max)
    }

    fn sample(&self, p: Point2D) -> f64 {
        // Piecewise-constant: value of the containing (nearest-site) cell.
        self.values[self.mesh.cell_of(p)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn demo() -> MeshGrid {
        let sites = [p(0.25, 0.25), p(0.75, 0.25), p(0.75, 0.75), p(0.25, 0.75)];
        MeshGrid::from_point_func(&sites, p(0.0, 0.0), p(1.0, 1.0), |s| s.x + s.y).unwrap()
    }

    #[test]
    fn construction_and_access() {
        let m = demo();
        assert_eq!(m.len(), 4);
        assert!((m.value(0) - 0.5).abs() < 1e-12);
        assert!((m.total_area() - 1.0).abs() < 1e-12);
        assert!((m.area(0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn flux_of_constant_is_value_times_area() {
        let sites = [p(0.2, 0.3), p(0.8, 0.1), p(0.5, 0.9), p(0.1, 0.6), p(0.9, 0.7)];
        let m = MeshGrid::constant(&sites, p(0.0, 0.0), p(1.0, 1.0), 2.5).unwrap();
        assert!((m.flux() - 2.5 * 1.0).abs() < 1e-9);
    }

    #[test]
    fn sample_returns_nearest_cell_value() {
        let m = demo();
        // A point deep in the lower-left quadrant is nearest to site 0.
        assert_eq!(m.sample(p(0.1, 0.1)), m.value(0));
        assert_eq!(m.sample(p(0.9, 0.9)), m.value(2));
    }

    #[test]
    fn value_ops_share_geometry() {
        let m = demo();
        let scaled = m.scale(2.0);
        assert!((scaled.value(0) - 1.0).abs() < 1e-12);
        // Geometry is shared (same Arc), not rebuilt.
        assert!(Arc::ptr_eq(&m.mesh, &scaled.mesh));
    }

    #[test]
    fn works_as_dyn_scalar_field() {
        let m = demo();
        let f: &dyn ScalarField2D = &m;
        assert_eq!(f.len(), 4);
        assert!((f.flux() - m.flux()).abs() < 1e-12);
    }
}
