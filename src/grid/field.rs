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
        (0..self.len())
            .map(|i| self.value_at(i) * self.cell_area(i))
            .sum()
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
        (0..self.len())
            .map(|i| self.value_at(i))
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum cell value.
    fn max(&self) -> f64 {
        (0..self.len())
            .map(|i| self.value_at(i))
            .fold(f64::NEG_INFINITY, f64::max)
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

    #[test]
    fn pixelgrid_is_empty_and_sum() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::constant(grid, 2.0);
        let f: &dyn ScalarField2D = &pg;
        assert_eq!(f.len(), 16);
        assert!(!f.is_empty());
        assert!((f.sum() - 2.0 * 16.0).abs() < 1e-12);
    }

    #[test]
    fn pixelgrid_min_max() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x + 2.0 * y);
        let f: &dyn ScalarField2D = &pg;
        // On the symmetric domain [-2,2]², the linear field's extrema at pixel
        // centres are ± of the largest |x + 2y|.
        let n = f.len();
        let expected_min = (0..n).map(|i| pg.value_at(i)).fold(f64::INFINITY, f64::min);
        let expected_max = (0..n)
            .map(|i| pg.value_at(i))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((f.min() - expected_min).abs() < 1e-12);
        assert!((f.max() - expected_max).abs() < 1e-12);
        assert!(f.min() < f.max());
    }

    #[test]
    fn pixelgrid_center_area_bounds() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::constant(grid, 1.0);
        let f: &dyn ScalarField2D = &pg;
        // center(i) agrees with the grid's own pixel centre, area is uniform,
        // and bounds match the grid bbox.
        for i in 0..f.len() {
            let (ix, jy) = pg.grid.unflatten_index(i);
            assert_eq!(f.center(i), pg.grid.pixel_center(ix, jy));
            assert!((f.cell_area(i) - pg.grid.pixel_area()).abs() < 1e-12);
        }
        assert_eq!(f.bounds(), pg.grid.bbox);
    }

    /// Minimal in-test field over an explicit list of values, used to exercise
    /// the trait defaults on the empty case a non-empty `PixelGrid` can't reach.
    struct VecField(Vec<f64>);

    impl ScalarField2D for VecField {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn value_at(&self, i: usize) -> f64 {
            self.0[i]
        }
        fn center(&self, _i: usize) -> Point2D {
            Point2D::new(0.0, 0.0)
        }
        fn cell_area(&self, _i: usize) -> f64 {
            1.0
        }
        fn bounds(&self) -> BBox {
            BBox::from_lims(0.0, 1.0, 0.0, 1.0)
        }
        fn sample(&self, _p: Point2D) -> f64 {
            0.0
        }
    }

    #[test]
    fn empty_field_defaults() {
        let f = VecField(vec![]);
        assert_eq!(f.len(), 0);
        assert!(f.is_empty());
        assert_eq!(f.mean(), 0.0); // n == 0 branch
        assert_eq!(f.sum(), 0.0);
        assert_eq!(f.flux(), 0.0);
        assert_eq!(f.min(), f64::INFINITY);
        assert_eq!(f.max(), f64::NEG_INFINITY);
    }

    #[test]
    fn vecfield_required_methods() {
        // Exercise every VecField method so the helper impl is fully covered.
        let f = VecField(vec![1.0, 4.0, 2.0]);
        assert!(!f.is_empty());
        assert_eq!(f.value_at(1), 4.0);
        assert_eq!(f.center(0), Point2D::new(0.0, 0.0));
        assert_eq!(f.cell_area(0), 1.0);
        assert_eq!(f.bounds(), BBox::from_lims(0.0, 1.0, 0.0, 1.0));
        assert_eq!(f.sample(Point2D::new(0.5, 0.5)), 0.0);
        assert_eq!(f.min(), 1.0);
        assert_eq!(f.max(), 4.0);
        assert_eq!(f.sum(), 7.0);
        assert_eq!(f.mean(), 7.0 / 3.0); // n != 0 branch
    }

    #[test]
    fn pixelgrid_sample_batch_matches_sample() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x + 2.0 * y);
        let f: &dyn ScalarField2D = &pg;
        let pts = [
            Point2D::new(0.5, -0.5),
            Point2D::new(-1.0, 0.25),
            Point2D::new(0.0, 0.0),
        ];
        let batch = f.sample_batch(&pts);
        assert_eq!(batch.len(), pts.len());
        for (b, &p) in batch.iter().zip(pts.iter()) {
            assert!((b - f.sample(p)).abs() < 1e-12);
        }
    }
}
