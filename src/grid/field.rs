// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! A representation-agnostic scalar field sampled on a 2D domain.
//!
//! Both [`PixelGrid`] (regular) and
//! [`MeshGrid`](crate::grid::MeshGrid) (Voronoi) implement [`ScalarField2D`], so
//! downstream code — solvers, models, rendering — have an interface behind one
//! trait. The trait is object-safe (`&dyn ScalarField2D`).

use crate::grid::{BBox, PixelGrid, Point2D};

/// A scalar field sampled at discrete cells over a 2D domain.
///
/// A "cell" is a pixel for a regular grid or a Voronoi cell for a mesh; it has a
/// value, a representative centre, and an area. Implementors provide the sampled
/// data and pointwise evaluation; aggregate quantities have default definitions.
pub trait ScalarField2D {
    /// Number of cells.
    fn len(&self) -> usize;

    /// Whether the field has no cells.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value stored at cell `i`.
    fn value_at(&self, i: usize) -> f64;

    /// Representative point of cell `i` (pixel centre / Voronoi site).
    fn center(&self, i: usize) -> Point2D;

    /// Area of cell `i` (pixel area / Voronoi cell area).
    fn cell_area(&self, i: usize) -> f64;

    /// Domain bounding box.
    fn bounds(&self) -> BBox;

    /// Field value at an arbitrary point (interpolation for a pixel grid,
    /// nearest-cell for a mesh).
    fn sample(&self, p: Point2D) -> f64;

    /// Area-weighted integral of the field, `Σ vᵢ · areaᵢ`.
    fn flux(&self) -> f64 {
        (0..self.len()).map(|i| self.value_at(i) * self.cell_area(i)).sum()
    }

    /// Sum of the cell values.
    fn sum(&self) -> f64 {
        (0..self.len()).map(|i| self.value_at(i)).sum()
    }

    /// Mean of the cell values (`0` for an empty field).
    fn mean(&self) -> f64 {
        let n = self.len();
        if n == 0 { 0.0 } else { self.sum() / n as f64 }
    }

    /// Minimum cell value.
    fn min(&self) -> f64 {
        (0..self.len()).map(|i| self.value_at(i)).fold(f64::INFINITY, f64::min)
    }

    /// Maximum cell value.
    fn max(&self) -> f64 {
        (0..self.len()).map(|i| self.value_at(i)).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Sample the field at many points.
    fn sample_batch(&self, points: &[Point2D]) -> Vec<f64> {
        points.iter().map(|&p| self.sample(p)).collect()
    }
}

impl ScalarField2D for PixelGrid {
    fn len(&self) -> usize {
        self.grid.n_pixels()
    }

    fn value_at(&self, i: usize) -> f64 {
        let (ix, jy) = self.grid.unflatten_index(i);
        self.get(ix, jy)
    }

    fn center(&self, i: usize) -> Point2D {
        let (ix, jy) = self.grid.unflatten_index(i);
        self.grid.pixel_center(ix, jy)
    }

    fn cell_area(&self, _i: usize) -> f64 {
        self.grid.pixel_area()
    }

    fn bounds(&self) -> BBox {
        self.grid.bbox
    }

    fn sample(&self, p: Point2D) -> f64 {
        self.interpolate(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;

    #[test]
    fn pixelgrid_flux_equals_value_times_area() {
        let grid = Grid2D::square(16, 1.0).unwrap(); // domain [-1,1]², area 4
        let pg = PixelGrid::constant(grid, 3.0);
        assert_eq!(pg.len(), 256);
        assert!((ScalarField2D::flux(&pg) - 3.0 * 4.0).abs() < 1e-9);
        assert!((ScalarField2D::mean(&pg) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn pixelgrid_sample_and_dyn() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x + 2.0 * y);
        let f: &dyn ScalarField2D = &pg;
        // Linear field is reproduced by bilinear interpolation in the interior.
        let v = f.sample(Point2D::new(0.5, -0.5));
        assert!((v - (0.5 - 1.0)).abs() < 1e-9, "sample {v}");
    }
}
