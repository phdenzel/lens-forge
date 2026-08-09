// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The lens Jacobian determinant det A sampled onto a field.
//!
//! Vanishes on the critical curves and its reciprocal is the magnification,
//! so it underlies both [`magnification`](super::magnification) and
//! [`critical`](super::critical).

use crate::grid::FieldOps;
use crate::lens::LensModel;

/// A field of det A(θ) = det(I − H(ψ)) of `lens`, on the same geometry as `field`
/// (only its geometry is used).
///
/// # Arguments
/// * `lens` - the lens model.
/// * `field` - the sampling geometry (a [`PixelGrid`](crate::grid::PixelGrid) or
///   [`MeshGrid`](crate::grid::MeshGrid)).
pub fn det_a_map<F: FieldOps>(lens: &dyn LensModel, field: &F) -> F {
    field.map_centers(|p| lens.jacobian_det(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, PixelGrid, Point2D};
    use crate::lens::SIS;

    /// Each cell of the map holds `lens.jacobian_det` at that cell's centre.
    fn check_det_a_map<F: FieldOps>(lens: &dyn LensModel, field: &F) {
        let da = det_a_map(lens, field);
        for i in 0..da.len() {
            assert_eq!(da.value_at(i), lens.jacobian_det(da.center(i)));
        }
    }

    #[test]
    fn det_a_map_samples_the_jacobian_det() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let field = PixelGrid::constant(Grid2D::square(16, 2.0).unwrap(), 0.0);
        check_det_a_map(&lens, &field);
    }
}
