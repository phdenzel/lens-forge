// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Magnification map μ(θ) = 1/det A of a lens, on any field representation.

use crate::grid::FieldOps;
use crate::lens::LensModel;

/// A field of the signed magnification μ(θ) = 1/det A of `lens`, on the same
/// geometry as `field` (only its geometry is used).
///
/// Diverges (±∞) at cells on the critical curves where det A = 0.
///
/// # Arguments
/// * `lens` - the lens model.
/// * `field` - the sampling geometry (a [`PixelGrid`](crate::grid::PixelGrid) or
///   [`MeshGrid`](crate::grid::MeshGrid)).
pub fn magnification_map<F: FieldOps>(lens: &dyn LensModel, field: &F) -> F {
    field.map_centers(|p| lens.magnification(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid, Point2D};
    use crate::lens::SIS;
    use crate::utils::rng::Lcg;

    /// Each cell of the map holds `lens.magnification` at that cell's centre.
    fn check_matches_lens<F: FieldOps>(lens: &dyn LensModel, field: &F) {
        let mu = magnification_map(lens, field);
        for i in 0..mu.len() {
            assert_eq!(mu.value_at(i), lens.magnification(mu.center(i)));
        }
    }

    #[test]
    fn samples_the_lens_on_a_pixel_grid() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let field = PixelGrid::constant(Grid2D::square(16, 2.0).unwrap(), 0.0);
        check_matches_lens(&lens, &field);
    }

    #[test]
    fn samples_the_lens_on_a_mesh() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let mut rng = Lcg::new(0x5A3);
        let bbox = BBox::from_lims(-2.0, 2.0, -2.0, 2.0);
        let sites: Vec<Point2D> = (0..300)
            .map(|_| Point2D::new(3.6 * rng.f() - 1.8, 3.6 * rng.f() - 1.8))
            .collect();
        let field = MeshGrid::zeros(&sites, bbox).unwrap();
        check_matches_lens(&lens, &field);
    }
}
