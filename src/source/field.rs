// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Free-form source: a surface-brightness field.
//!
//! The counterpart to the parametric profiles ([`Sersic`](crate::source::Sersic),
//! [`Gaussian`](crate::source::Gaussian)): rather than a closed form, the brightness
//! is read from a stored [`ScalarField2D`] by sampling. Generic over the field type,
//! so the same source wraps a [`PixelGrid`](crate::grid::PixelGrid) (bilinear) or a
//! [`MeshGrid`](crate::grid::MeshGrid) (nearest-cell) — whatever the field's
//! [`sample`](ScalarField2D::sample) provides.

use crate::grid::{Point2D, ScalarField2D};
use crate::source::model::SourceModel;

/// A free-form source backed by any scalar surface-brightness field.
pub struct SourceField<F: ScalarField2D>(F);

impl<F: ScalarField2D> SourceField<F> {
    /// Wrap a scalar field as a surface-brightness source.
    ///
    /// # Arguments
    /// * `field` - surface brightness in any representation.
    pub fn new(field: F) -> Self {
        Self(field)
    }

    /// The wrapped brightness field.
    pub fn field(&self) -> &F {
        &self.0
    }

    /// Consume the source, returning the wrapped field.
    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: ScalarField2D> SourceModel for SourceField<F> {
    fn brightness(&self, beta: Point2D) -> f64 {
        self.0.sample(beta)
    }

    fn brightness_batch(&self, betas: &[Point2D]) -> Vec<f64> {
        self.0.sample_batch(betas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid};

    #[test]
    fn brightness_matches_pixelgrid_sample() {
        let grid = Grid2D::square(32, 4.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (0.3 * x).sin() + 0.5 * y);
        let src = SourceField::new(pg.clone());
        for p in [Point2D::new(0.7, -1.2), Point2D::new(-2.1, 1.5)] {
            assert_eq!(src.brightness(p), pg.sample(p));
        }
    }

    #[test]
    fn brightness_matches_meshgrid_sample() {
        // A MeshGrid drives the same source type — nearest-cell brightness.
        let bbox = BBox::new(Point2D::new(-2.0, -2.0), Point2D::new(2.0, 2.0));
        let sites = [
            Point2D::new(-1.0, -1.0),
            Point2D::new(1.0, -1.0),
            Point2D::new(-1.0, 1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 0.0),
        ];
        let mesh = MeshGrid::from_func(&sites, bbox, |x, y| x + 2.0 * y).unwrap();
        let src = SourceField::new(mesh.clone());
        for p in [Point2D::new(0.6, -0.8), Point2D::new(-1.3, 0.9)] {
            assert_eq!(src.brightness(p), mesh.sample(p));
        }
    }

    #[test]
    fn brightness_batch_matches_pointwise() {
        let grid = Grid2D::square(16, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x * x - y);
        let src = SourceField::new(pg);
        let pts = [
            Point2D::new(0.1, 0.2),
            Point2D::new(-0.7, 1.1),
            Point2D::new(1.3, -0.4),
        ];
        let batch = src.brightness_batch(&pts);
        for (b, &p) in batch.iter().zip(&pts) {
            assert_eq!(*b, src.brightness(p));
        }
    }

    #[test]
    fn field_accessor_and_into_inner_round_trip() {
        let grid = Grid2D::square(8, 1.0).unwrap();
        let pg = PixelGrid::constant(grid, 2.5);
        let src = SourceField::new(pg);
        assert_eq!(src.field().sample(Point2D::origin()), 2.5);
        assert_eq!(src.into_inner().sample(Point2D::origin()), 2.5);
    }
}
