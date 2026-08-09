// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended-source imaging: forming the lensed surface-brightness field.
//!
//! Surface brightness is conserved along rays (Liouville), so the lensed image is
//! I(θ) = S(β(θ)) = S(θ − α(θ)): ray-trace each image point to the source plane and
//! read the source there. [`ImageSimulation::Direct`] does this per cell centre; adaptive
//! supersampling and triangle mapping join as further variants. PSF and noise are a
//! separate instrument stage (a convolution), not part of forming the image.

use crate::grid::FieldOps;
use crate::lens::LensModel;
use crate::solver::mapping::raytrace::RayTracer;
use crate::source::SourceModel;

/// An extended-source imaging strategy: (lens, source) → lensed surface-brightness
/// field, selected by variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSimulation {
    /// Direct ray-shooting: one ray per cell centre, sampling the source at
    /// β = θ − α(θ). Exact per cell; aliases near caustics where |μ| is large.
    Direct,
}

impl ImageSimulation {
    /// Form the lensed surface-brightness field of `source` seen through `lens`,
    /// sampled onto the geometry of `field`.
    ///
    /// # Arguments
    /// * `lens` - the lens model.
    /// * `source` - the source surface-brightness model.
    /// * `field` - the image-plane sampling geometry (only its geometry is used).
    pub fn image<F: FieldOps>(
        &self,
        lens: &dyn LensModel,
        source: &dyn SourceModel,
        field: &F,
    ) -> F {
        match self {
            // Ray-trace all cell centres in one batch (RayTracer), then sample the
            // source at each landing point.
            ImageSimulation::Direct => field
                .map_centers_batch(|centers| source.brightness_batch(&RayTracer::Direct.trace(lens, centers))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid, Point2D};
    use crate::lens::SIS;
    use crate::rng::Lcg;
    use crate::source::Gaussian;

    /// Each cell holds the source brightness at the ray-traced position.
    fn check_matches_source<F: FieldOps>(lens: &dyn LensModel, source: &dyn SourceModel, field: &F) {
        let img = ImageSimulation::Direct.image(lens, source, field);
        for i in 0..img.len() {
            assert_eq!(img.value_at(i), source.brightness(lens.ray_trace(img.center(i))));
        }
    }

    #[test]
    fn direct_image_samples_the_source_through_the_lens_on_a_pixel_grid() {
        let lens = SIS::new(Point2D::origin(), 0.6);
        let source = Gaussian::new(Point2D::new(0.1, 0.0), 1.0, 0.2, 1.0, 0.0);
        let field = PixelGrid::constant(Grid2D::square(32, 2.0).unwrap(), 0.0);
        check_matches_source(&lens, &source, &field);
    }

    #[test]
    fn direct_image_samples_the_source_through_the_lens_on_a_mesh() {
        let lens = SIS::new(Point2D::origin(), 0.6);
        let source = Gaussian::new(Point2D::new(0.1, 0.0), 1.0, 0.2, 1.0, 0.0);
        let mut rng = Lcg::new(0x5A3);
        let bbox = BBox::from_lims(-2.0, 2.0, -2.0, 2.0);
        let sites: Vec<Point2D> = (0..300)
            .map(|_| Point2D::new(3.6 * rng.f() - 1.8, 3.6 * rng.f() - 1.8))
            .collect();
        let field = MeshGrid::zeros(&sites, bbox).unwrap();
        check_matches_source(&lens, &source, &field);
    }

    #[test]
    fn on_axis_source_images_to_a_rotationally_symmetric_ring() {
        // SIS + a source at the origin: I(θ) depends only on |θ| (an Einstein ring),
        // so it is invariant under the grid's 90° rotation.
        let lens = SIS::new(Point2D::origin(), 0.8);
        let source = Gaussian::new(Point2D::origin(), 1.0, 0.15, 1.0, 0.0);
        let field = PixelGrid::constant(Grid2D::square(128, 2.0).unwrap(), 0.0);
        let img = ImageSimulation::Direct.image(&lens, &source, &field);
        for &r in &[0.4, 0.8, 1.2] {
            let a = img.interpolate_or(Point2D::new(r, 0.0), 0.0);
            let b = img.interpolate_or(Point2D::new(0.0, r), 0.0);
            assert!((a - b).abs() < 1e-6, "ring asymmetric at r={r}: {a} vs {b}");
        }
    }
}
