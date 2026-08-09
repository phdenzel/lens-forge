// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Back-projecting the arc mask to the source plane (Stage 1).
//!
//! Ray-tracing every active image pixel through a member's deflection field lands it at
//! `β = θ − α(θ)`. Those positions are the Voronoi generators of a per-member source mesh —
//! adaptivity is automatic, since a fixed image pixel maps to source-plane area ∝ 1/|μ|, so
//! generators pile up near caustics exactly where resolution is needed. A correct model maps
//! the several images of one source coherently on top of each other; a wrong one scatters
//! their brightness — the cheap [`source_plane_dispersion`] cull.

use crate::grid::{BBox, MeshGrid, Point2D, ScalarField2D};
use crate::instrument::mask::Mask;
use crate::lens::LensModel;
use crate::Result;

/// Back-project the active image pixels of `image` to source-plane positions under `lens`.
///
/// # Arguments
/// * `lens` - the lens model.
/// * `image` - the image-plane field (only geometry is read).
/// * `mask` - the pixel mask; pixels with zero weight are skipped.
pub fn back_project(lens: &dyn LensModel, image: &impl ScalarField2D, mask: &Mask) -> Vec<Point2D> {
    (0..image.len())
        .filter(|&i| mask.weight(i) > 0.0)
        .map(|i| lens.ray_trace(image.center(i)))
        .collect()
}

/// Build a per-member source mesh whose generators are the back-projected active pixels
/// (values initialized to zero — the semi-linear inversion solves for them).
///
/// # Arguments
/// * `lens` - the lens model.
/// * `image` - the image-plane field (only geometry is read).
/// * `mask` - the pixel mask.
/// * `pad` - fractional padding added around the generators' bounding box.
///
/// # Errors
/// Propagates a mesh-construction failure (e.g. fewer than three distinct generators).
pub fn source_mesh(
    lens: &dyn LensModel,
    image: &impl ScalarField2D,
    mask: &Mask,
    pad: f64,
) -> Result<MeshGrid> {
    let sites = back_project(lens, image, mask);
    let bbox = BBox::from_points(&sites)
        .map(|b| b.padded_square(pad.max(0.0), 1.0))
        .unwrap_or_else(|| BBox::from_lims(-1.0, 1.0, -1.0, 1.0));
    MeshGrid::zeros(&sites, bbox)
}

/// Source-plane brightness dispersion: the within-bin variance of image brightness after
/// back-projection, summed over a coarse `nbins × nbins` source-plane grid. A correct model
/// overlaps the images coherently (low dispersion); a wrong one scatters them (high).
///
/// # Arguments
/// * `lens` - the lens model.
/// * `image` - the image-plane field (geometry and brightness).
/// * `mask` - the pixel mask.
/// * `nbins` - coarse source-plane binning per axis.
pub fn source_plane_dispersion(
    lens: &dyn LensModel,
    image: &impl ScalarField2D,
    mask: &Mask,
    nbins: usize,
) -> f64 {
    let active: Vec<(Point2D, f64)> = (0..image.len())
        .filter(|&i| mask.weight(i) > 0.0)
        .map(|i| (lens.ray_trace(image.center(i)), image.value_at(i)))
        .collect();
    let Some(bbox) = BBox::from_points(&active.iter().map(|(b, _)| *b).collect::<Vec<_>>()) else {
        return 0.0;
    };
    let (xmin, xmax) = bbox.extent_x();
    let (ymin, ymax) = bbox.extent_y();
    let (wx, wy) = ((xmax - xmin).max(1e-12), (ymax - ymin).max(1e-12));

    // Welford within-bin accumulation of brightness; sum the M2 (within-bin SS).
    let (mut count, mut mean, mut m2) = (vec![0usize; nbins * nbins], vec![0.0; nbins * nbins], vec![0.0; nbins * nbins]);
    for (b, v) in &active {
        let bx = (((b.x - xmin) / wx * nbins as f64) as usize).min(nbins - 1);
        let by = (((b.y - ymin) / wy * nbins as f64) as usize).min(nbins - 1);
        let idx = by * nbins + bx;
        count[idx] += 1;
        let d = v - mean[idx];
        mean[idx] += d / count[idx] as f64;
        m2[idx] += d * (v - mean[idx]);
    }
    m2.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, PixelGrid};
    use crate::lens::SIS;

    #[test]
    fn back_projection_keeps_only_active_pixels() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let image = PixelGrid::constant(Grid2D::square(6, 2.0).unwrap(), 1.0);
        let flags: Vec<bool> = (0..image.len()).map(|i| i % 3 == 0).collect();
        let mask = Mask::from_flags(&flags);
        let sites = back_project(&lens, &image, &mask);
        assert_eq!(sites.len(), mask.n_active());
    }

    #[test]
    fn source_mesh_has_one_generator_per_active_pixel() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let image = PixelGrid::constant(Grid2D::square(8, 2.0).unwrap(), 1.0);
        let mask = Mask::all(image.len());
        let mesh = source_mesh(&lens, &image, &mask, 0.1).unwrap();
        assert_eq!(mesh.len(), image.len());
    }

    #[test]
    fn a_uniform_image_has_no_dispersion() {
        // Every pixel shares the same brightness, so within-bin variance is zero for any lens.
        let lens = SIS::new(Point2D::origin(), 0.7);
        let image = PixelGrid::constant(Grid2D::square(10, 2.0).unwrap(), 2.5);
        let mask = Mask::all(image.len());
        assert!(source_plane_dispersion(&lens, &image, &mask, 4) < 1e-12);
    }

    #[test]
    fn a_structured_image_has_positive_dispersion() {
        let lens = SIS::new(Point2D::origin(), 0.7);
        let image = PixelGrid::from_func(Grid2D::square(10, 2.0).unwrap(), |x, y| x * x + y);
        let mask = Mask::all(image.len());
        assert!(source_plane_dispersion(&lens, &image, &mask, 4) > 0.0);
    }
}
