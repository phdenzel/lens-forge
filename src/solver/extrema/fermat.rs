// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The Fermat (arrival-time) surface and point-source total magnification.
//!
//! The Fermat potential τ(θ; β) = ½|θ − β|² − ψ(θ) is the arrival-time surface of
//! a source at β; its stationary points are the images (minima, saddles and maxima
//! by parity). Sampling it onto a field gives a surface to inspect; summing |μ|
//! over the images gives the total magnification. Physical delays live in `delays`.
//!
//! Dropping the source-dependent tilting term −θ·β leaves the β-independent
//! lensing Roche potential ½|θ|² − ψ(θ), a lensing analogue of the binary-star
//! Roche potential.

use crate::grid::{FieldOps, Point2D};
use crate::lens::LensModel;
use crate::solver::image::Image;

/// The Fermat potential τ(θ; β) = ½|θ − β|² − ψ(θ) of `lens` at `theta`.
///
/// # Arguments
/// * `lens` - the lens model (supplies the potential ψ).
/// * `theta` - image-plane position θ.
/// * `beta` - source-plane position β.
pub fn fermat_potential(lens: &dyn LensModel, theta: Point2D, beta: Point2D) -> f64 {
    0.5 * (theta - beta).norm_squared() - lens.potential(theta)
}

/// The lensing Roche potential R(θ) = ½|θ|² − ψ(θ) of `lens` at `theta`.
///
/// The arrival-time surface with the source-dependent tilting term −θ·β removed:
/// a β-independent surface with ½|θ|² the centrifugal term and −ψ the
/// gravitational one. Equals [`fermat_potential`] for a source at the origin.
///
/// # Arguments
/// * `lens` - the lens model (supplies the potential ψ).
/// * `theta` - image-plane position θ.
pub fn roche_potential(lens: &dyn LensModel, theta: Point2D) -> f64 {
    0.5 * theta.norm_squared() - lens.potential(theta)
}

/// The Fermat surface τ(θ; β) of `lens` for a source at `beta`, sampled onto the
/// geometry of `field` (only its geometry is used).
///
/// # Arguments
/// * `lens` - the lens model.
/// * `beta` - source-plane position β.
/// * `field` - the sampling geometry (a [`PixelGrid`](crate::grid::PixelGrid) or
///   [`MeshGrid`](crate::grid::MeshGrid)).
pub fn fermat_surface<F: FieldOps>(lens: &dyn LensModel, beta: Point2D, field: &F) -> F {
    field.map_centers(|theta| fermat_potential(lens, theta, beta))
}

/// The total magnification μ_tot = Σ_i |μ_i| of a point source with the given
/// `images` — the observable flux ratio for an unresolved source.
///
/// # Arguments
/// * `images` - the images of the point source (from an
///   [`ImageFinder`](super::images::ImageFinder)).
pub fn total_magnification(images: &[Image]) -> f64 {
    images.iter().map(|im| im.mu.abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid};
    use crate::lens::SIS;
    use crate::rng::Lcg;
    use crate::solver::image::Parity;

    /// Each cell of the surface holds `fermat_potential` at that cell's centre.
    fn check_surface<F: FieldOps>(lens: &dyn LensModel, beta: Point2D, field: &F) {
        let tau = fermat_surface(lens, beta, field);
        for i in 0..tau.len() {
            assert_eq!(tau.value_at(i), fermat_potential(lens, tau.center(i), beta));
        }
    }

    #[test]
    fn fermat_potential_matches_the_sis_analytic() {
        // SIS ψ = θ_E r; at θ = (1.3, 0), β = (0.3, 0): τ = ½·1² − 1·1.3 = −0.8.
        let lens = SIS::new(Point2D::origin(), 1.0);
        let tau = fermat_potential(&lens, Point2D::new(1.3, 0.0), Point2D::new(0.3, 0.0));
        assert!((tau + 0.8).abs() < 1e-12);
    }

    #[test]
    fn roche_potential_matches_the_sis_analytic() {
        // SIS ψ = θ_E r; at θ = (2, 0), θ_E = 1: R = ½·2² − 1·2 = 0.
        let lens = SIS::new(Point2D::origin(), 1.0);
        assert!(roche_potential(&lens, Point2D::new(2.0, 0.0)).abs() < 1e-12);
    }

    #[test]
    fn roche_potential_is_the_fermat_surface_at_the_origin() {
        // Dropping the tilting term is exactly a source at β = 0.
        let lens = SIS::new(Point2D::origin(), 0.7);
        let theta = Point2D::new(0.4, -0.9);
        let roche = roche_potential(&lens, theta);
        assert_eq!(roche, fermat_potential(&lens, theta, Point2D::origin()));
    }

    #[test]
    fn fermat_surface_samples_the_potential_on_a_pixel_grid() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let field = PixelGrid::constant(Grid2D::square(16, 2.0).unwrap(), 0.0);
        check_surface(&lens, Point2D::new(0.1, -0.2), &field);
    }

    #[test]
    fn fermat_surface_samples_the_potential_on_a_mesh() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let mut rng = Lcg::new(0x5A3);
        let bbox = BBox::from_lims(-2.0, 2.0, -2.0, 2.0);
        let sites: Vec<Point2D> = (0..300)
            .map(|_| Point2D::new(3.6 * rng.f() - 1.8, 3.6 * rng.f() - 1.8))
            .collect();
        let field = MeshGrid::zeros(&sites, bbox).unwrap();
        check_surface(&lens, Point2D::new(0.1, -0.2), &field);
    }

    #[test]
    fn minimum_image_arrives_before_the_saddle() {
        // SIS θ_E = 1, β = (0.3, 0): outer (1.3, 0) is the minimum, inner (−0.7, 0)
        // the saddle. The minimum has the lower Fermat potential (arrives first).
        let lens = SIS::new(Point2D::origin(), 1.0);
        let beta = Point2D::new(0.3, 0.0);
        let tau_min = fermat_potential(&lens, Point2D::new(1.3, 0.0), beta);
        let tau_saddle = fermat_potential(&lens, Point2D::new(-0.7, 0.0), beta);
        assert!(tau_min < tau_saddle);
    }

    #[test]
    fn total_magnification_sums_absolute_values() {
        // A minimum (μ = +4) and a saddle (μ = −2.5): μ_tot = 6.5.
        let images = [
            Image { theta: Point2D::new(1.0, 0.0), mu: 4.0, parity: Parity::MINIMUM, tau: 0.0 },
            Image { theta: Point2D::new(-0.5, 0.0), mu: -2.5, parity: Parity::SADDLE, tau: 0.1 },
        ];
        assert!((total_magnification(&images) - 6.5).abs() < 1e-12);
    }

    #[test]
    fn no_images_have_zero_total_magnification() {
        assert_eq!(total_magnification(&[]), 0.0);
    }
}
