// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Convergence map κ(θ) = ½∇²ψ of a lens, on any field representation.

use crate::grid::FieldOps;
use crate::lens::LensModel;

/// A field of the convergence κ(θ) = ½∇²ψ of `lens`, on the same geometry as
/// `field` (only its geometry is used).
///
/// # Arguments
/// * `lens` - the lens model.
/// * `field` - the sampling geometry (a [`PixelGrid`](crate::grid::PixelGrid) or
///   [`MeshGrid`](crate::grid::MeshGrid)).
pub fn convergence_map<F: FieldOps>(lens: &dyn LensModel, field: &F) -> F {
    field.map_centers(|p| lens.convergence(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid, Point2D};
    use crate::lens::SIS;
    use crate::utils::rng::Lcg;

    /// Each cell of the map holds `lens.convergence` at that cell's centre.
    fn check_matches_lens<F: FieldOps>(lens: &dyn LensModel, field: &F) {
        let kappa = convergence_map(lens, field);
        for i in 0..kappa.len() {
            assert_eq!(kappa.value_at(i), lens.convergence(kappa.center(i)));
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
