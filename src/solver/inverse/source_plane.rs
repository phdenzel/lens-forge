// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-member source construction on a [`SourceBasis`](crate::solver::basis::SourceBasis).
//!
//! [`source_field`] builds a member's source plane over its back-projected generators, and
//! [`reconstruct_source`] solves the semi-linear inversion and pours the coefficients back into
//! a [`SourceModel`](crate::source::SourceModel). The representation (mesh or pixel) is the
//! caller's choice.

use crate::grid::{BBox, PixelGrid, ScalarField2D};
use crate::instrument::mask::Mask;
use crate::instrument::noise::Noise;
use crate::lens::LensModel;
use crate::solver::basis::{SourceBasis, SourceConfig};
use crate::solver::extended::operator::LensingOperator;
use crate::source::SourceField;
use crate::Result;

use super::backproject::back_project;
use super::semilinear::{invert_source, Regularizer};

/// CG tolerance and cap for the per-member source inversion.
const CG_TOL: f64 = 1e-10;
const CG_MAX_ITER: usize = 500;

/// Build a per-member source plane whose generators are the back-projected active pixels
/// (values initialized to zero — the semi-linear inversion solves for them). The
/// representation `S` is chosen by the caller (mesh or pixel).
///
/// # Arguments
/// * `lens` - the lens model.
/// * `image` - the image-plane field (only geometry is read).
/// * `mask` - the pixel mask.
/// * `cfg` - the source-plane build config (padding and pixel resolution).
///
/// # Errors
/// Propagates a construction failure (e.g. fewer than three distinct generators).
pub fn source_field<S: SourceBasis>(
    lens: &dyn LensModel,
    image: &impl ScalarField2D,
    mask: &Mask,
    cfg: &SourceConfig,
) -> Result<S> {
    let sites = back_project(lens, image, mask);
    let bbox = BBox::from_points(&sites)
        .map(|b| b.padded_square(cfg.pad.max(0.0), 1.0))
        .unwrap_or_else(|| BBox::from_lims(-1.0, 1.0, -1.0, 1.0));
    S::from_generators(&sites, bbox, cfg)
}

/// Invert one member's source: build its source plane, the lensing operator, and solve the
/// semi-linear system, returning the source geometry, the operator, and the coefficients.
#[allow(clippy::too_many_arguments)]
pub(super) fn invert_member<S: SourceBasis>(
    lens: &dyn LensModel,
    image: &PixelGrid,
    data: &[f64],
    noise: &Noise,
    mask: &Mask,
    lambda: f64,
    source_cfg: &SourceConfig,
) -> Result<(S, LensingOperator, Vec<f64>)> {
    let source = source_field::<S>(lens, image, mask, source_cfg)?;
    let f = LensingOperator::build(lens, image, &source);
    let reg = Regularizer::from_adjacency(&source);
    let (s, _) = invert_source(&f, data, noise, mask, &reg, lambda, CG_TOL, CG_MAX_ITER);
    Ok((source, f, s))
}

/// Reconstruct the source of one member as a [`SourceModel`](crate::source::SourceModel):
/// back-project the arc mask, solve the semi-linear inversion, and pour the coefficients
/// back onto the source plane. The representation `S` (mesh or pixel) is the caller's choice.
///
/// # Arguments
/// * `lens` - the member's lens model.
/// * `image` - the image-plane geometry.
/// * `data` - the observed image (length `image.len()`).
/// * `noise` - the noise model.
/// * `mask` - the arc mask.
/// * `lambda` - source-inversion regularization strength.
/// * `source_cfg` - the source-plane build config (padding, pixel resolution).
///
/// # Errors
/// Propagates a source-plane construction failure (e.g. too few active pixels).
pub fn reconstruct_source<S: SourceBasis>(
    lens: &dyn LensModel,
    image: &PixelGrid,
    data: &[f64],
    noise: &Noise,
    mask: &Mask,
    lambda: f64,
    source_cfg: &SourceConfig,
) -> Result<SourceField<S>> {
    let (source, _f, s) = invert_member::<S>(lens, image, data, noise, mask, lambda, source_cfg)?;
    Ok(SourceField::new(source.with_coeffs(&s)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, MeshGrid, Point2D};
    use crate::lens::SIS;
    use crate::solver::extended::imaging::ImageSimulation;
    use crate::source::SourceModel;

    #[test]
    fn source_field_has_one_generator_per_active_pixel() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let image = PixelGrid::constant(Grid2D::square(8, 2.0).unwrap(), 1.0);
        let mask = Mask::all(image.len());
        let cfg = SourceConfig { pad: 0.1, resolution: None };
        let mesh = source_field::<MeshGrid>(&lens, &image, &mask, &cfg).unwrap();
        assert_eq!(mesh.len(), image.len());
    }

    #[test]
    fn source_field_fails_without_active_pixels() {
        // No active pixels → no generators → mesh construction fails (via the fallback bbox).
        let lens = SIS::new(Point2D::origin(), 0.5);
        let image = PixelGrid::constant(Grid2D::square(6, 2.0).unwrap(), 1.0);
        let mask = Mask::from_flags(&vec![false; image.len()]);
        let cfg = SourceConfig { pad: 0.1, resolution: None };
        assert!(source_field::<MeshGrid>(&lens, &image, &mask, &cfg).is_err());
    }

    /// The arc of an SIS(θ_E = 0.5) lensing a compact off-centre Gaussian, with an arc mask.
    fn arc_scenario() -> (PixelGrid, Vec<f64>, Noise, Mask) {
        let truth = SIS::new(Point2D::origin(), 0.5);
        let src = PixelGrid::from_func(Grid2D::square(24, 1.5).unwrap(), |x, y| {
            (-((x - 0.1).powi(2) + y * y) / 0.03).exp()
        });
        let image = PixelGrid::constant(Grid2D::square(16, 2.0).unwrap(), 0.0);
        let data_img = ImageSimulation::Direct.image(&truth, &SourceField::new(src), &image);
        let data: Vec<f64> = (0..image.len()).map(|i| data_img.value_at(i)).collect();
        let mask = Mask::from_flags(&data.iter().map(|&d| d > 1e-3).collect::<Vec<_>>());
        (image, data, Noise::uniform(0.05), mask)
    }

    #[test]
    fn reconstruct_source_yields_a_source_model() {
        // The truth-matching lens reconstructs a source with brightness near the true
        // off-centre Gaussian at β ≈ (0.1, 0); a mesh and a pixel source both work.
        let (image, data, noise, mask) = arc_scenario();
        let lens = SIS::new(Point2D::origin(), 0.5);
        let cfg = SourceConfig::default();

        let mesh_src =
            reconstruct_source::<MeshGrid>(&lens, &image, &data, &noise, &mask, 1e-4, &cfg).unwrap();
        let pix_src =
            reconstruct_source::<PixelGrid>(&lens, &image, &data, &noise, &mask, 1e-4, &cfg)
                .unwrap();

        assert!(mesh_src.brightness(Point2D::new(0.1, 0.0)).abs() > 0.0);
        assert!(pix_src.brightness(Point2D::new(0.1, 0.0)).abs() > 0.0);
    }

    #[test]
    fn reconstruct_source_fails_without_active_pixels() {
        let (image, data, noise, _) = arc_scenario();
        let lens = SIS::new(Point2D::origin(), 0.5);
        let mask = Mask::from_flags(&vec![false; image.len()]);
        let cfg = SourceConfig::default();
        assert!(
            reconstruct_source::<MeshGrid>(&lens, &image, &data, &noise, &mask, 1e-4, &cfg).is_err()
        );
    }
}
