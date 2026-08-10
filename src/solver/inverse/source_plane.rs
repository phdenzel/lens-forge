// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The source-plane representation seam and per-member source construction.
//!
//! [`SourceBasis`] is to the source plane what `KappaBasis` is to the lens plane: the single
//! representation-specific hook that lets a Voronoi [`MeshGrid`] or a regular [`PixelGrid`]
//! carry the reconstructed source. It builds a zero-initialized field over the back-projected
//! generators and — via [`CellAdjacency`] — feeds the smoothness regularizer. On top of it,
//! [`source_field`] constructs a member's source plane for the semi-linear inversion.

use crate::grid::{
    BBox, CellAdjacency, FieldOps, Grid2D, MeshGrid, PixelGrid, Point2D, ScalarField2D,
};
use crate::instrument::mask::Mask;
use crate::instrument::noise::Noise;
use crate::lens::LensModel;
use crate::solver::extended::operator::LensingOperator;
use crate::Result;

use super::backproject::back_project;
use super::semilinear::{invert_source, Regularizer};

/// CG tolerance and cap for the per-member source inversion.
const CG_TOL: f64 = 1e-10;
const CG_MAX_ITER: usize = 500;

/// How to build a per-member source plane from its back-projected generators.
#[derive(Debug, Clone, Copy)]
pub struct SourceConfig {
    /// Fractional padding added around the generators' bounding box.
    pub pad: f64,
    /// Per-axis pixel count for a pixel source; `None` picks `≈√(#generators)` so the pixel
    /// count matches the mesh cell count. Ignored by a mesh source.
    pub resolution: Option<usize>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self { pad: 0.1, resolution: None }
    }
}

/// A source-plane field that a semi-linear inversion can solve for: it builds itself from
/// back-projected generators and exposes cell adjacency for the regularizer.
pub trait SourceBasis: FieldOps + CellAdjacency + Sized {
    /// A zero-initialized source field over `sites`, bounded by `bbox`.
    ///
    /// # Arguments
    /// * `sites` - the back-projected generators (Voronoi sites / pixel-count hint).
    /// * `bbox` - the padded source-plane bounding box.
    /// * `cfg` - the build config (a pixel source reads `resolution`; a mesh ignores it).
    ///
    /// # Errors
    /// Propagates a construction failure (e.g. too few distinct generators).
    fn from_generators(sites: &[Point2D], bbox: BBox, cfg: &SourceConfig) -> Result<Self>;
}

impl SourceBasis for MeshGrid {
    fn from_generators(sites: &[Point2D], bbox: BBox, _cfg: &SourceConfig) -> Result<Self> {
        MeshGrid::zeros(sites, bbox)
    }
}

impl SourceBasis for PixelGrid {
    fn from_generators(sites: &[Point2D], bbox: BBox, cfg: &SourceConfig) -> Result<Self> {
        let n = cfg
            .resolution
            .unwrap_or_else(|| (sites.len() as f64).sqrt().round() as usize)
            .max(2);
        Ok(PixelGrid::zeros(Grid2D::new(n, n, bbox)?))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use crate::lens::SIS;

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
}
