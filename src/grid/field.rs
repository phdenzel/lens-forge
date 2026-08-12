// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! A representation-agnostic scalar field sampled on a 2D domain.
//!
//! [`PixelGrid`](crate::grid::PixelGrid) (regular) and
//! [`MeshGrid`](crate::grid::MeshGrid) (Voronoi) implement [`ScalarField2D`], so
//! downstream code — solvers, models, rendering — has an interface behind one
//! trait.

use crate::grid::{BBox, Point2D};

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

    /// Cells and weights of the linear stencil reproducing [`sample`](Self::sample)
    /// at `p`: `sample(p) == Σ weightᵢ · value_at(indexᵢ)`. Empty where `sample`
    /// reads as zero (outside a pixel grid).
    ///
    /// # Arguments
    /// * `p` - the query point.
    fn sample_stencil(&self, p: Point2D) -> Vec<(usize, f64)>;

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

/// Per-cell adjacency of a discretized field: the cells sharing an edge with cell `i`.
///
/// The grid-topology seam reused by the convergence basis (gradient-limit priors) and
/// the source-plane smoothness regularizer, so both run on either representation.
pub trait CellAdjacency {
    /// Indices of the cells adjacent to cell `i`.
    ///
    /// # Arguments
    /// * `i` - cell index.
    fn cell_neighbors(&self, i: usize) -> Vec<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

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
        fn sample_stencil(&self, _p: Point2D) -> Vec<(usize, f64)> {
            Vec::new()
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
        assert!(f.sample_stencil(Point2D::new(0.5, 0.5)).is_empty());
        assert_eq!(f.min(), 1.0);
        assert_eq!(f.max(), 4.0);
        assert_eq!(f.sum(), 7.0);
        assert_eq!(f.mean(), 7.0 / 3.0); // n != 0 branch
    }
}
