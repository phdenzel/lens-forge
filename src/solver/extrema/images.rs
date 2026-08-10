// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-source image finding: solve β = θ − α(θ) for all images of β.
//!
//! An [`ImageFinder`] trait returns every image with its magnification, parity,
//! and Fermat potential. Every [`MapDiscretization`] has an [`ImageFinder`]
//! implementation: its seeds are [`newton`]-inferred, deduplicated, and 
//! classified by [`Image::at`]. Build a discretization once and query it for
//! many source positions.

use crate::grid::Point2D;
use crate::lens::LensModel;
use crate::solver::basis::MapDiscretization;
use crate::solver::image::Image;
use super::newton::{newton, NewtonConfig};

/// Image-plane distance below which two polished images are treated as one.
const DEDUP_TOL: f64 = 1e-6;

/// Solve the lens equation for a point source: all images of `beta` with parity
/// and magnification.
pub trait ImageFinder {
    /// All images of the source at `beta` under `lens`.
    ///
    /// # Arguments
    /// * `lens` - the lens model.
    /// * `beta` - the source-plane position.
    fn images(&self, lens: &dyn LensModel, beta: Point2D) -> Vec<Image>;
}

/// Every discretized lens map is an image finder: its seeds are Newton-polished,
/// deduplicated within `DEDUP_TOL`, and classified.
///
/// The discretization must have been built from the same `lens` passed to
/// [`images`](ImageFinder::images) (Newton re-evaluates the jet at fresh θ).
impl<D: MapDiscretization + ?Sized> ImageFinder for D {
    fn images(&self, lens: &dyn LensModel, beta: Point2D) -> Vec<Image> {
        let mut images: Vec<Image> = Vec::new();
        for seed in self.images_of(beta) {
            let Some(theta) = newton(lens, beta, seed, NewtonConfig::default()) else { continue };
            // Many cells seed the same image; keep only the first of each.
            if !images.iter().any(|im| (im.theta - theta).norm() < DEDUP_TOL) {
                images.push(Image::at(lens, theta, beta));
            }
        }
        images
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D};
    use crate::lens::SIS;
    use crate::rng::Lcg;
    use crate::solver::image::Parity;
    use crate::solver::basis::{MapPixellation, MapTessellation};

    /// The image of `images` nearest `theta`, if any is within `tol`.
    fn image_at<'a>(images: &'a [Image], theta: Point2D, tol: f64) -> Option<&'a Image> {
        images.iter().find(|im| (im.theta - theta).norm() < tol)
    }

    /// Assert the two analytic SIS images of β = (0.3, 0) under θ_E = 1.
    fn assert_sis_pair(images: &[Image]) {
        assert_eq!(images.len(), 2, "SIS off-axis source has two images");
        // Outer image: minimum, μ = r/(r − θ_E) = 1.3/0.3.
        let outer = image_at(images, Point2D::new(1.3, 0.0), 1e-6).expect("outer image");
        assert_eq!(outer.parity, Parity::MINIMUM);
        assert!((outer.mu - 1.3 / 0.3).abs() < 1e-6);
        // Inner image: saddle, μ = −r/(θ_E − r) = −0.7/0.3.
        let inner = image_at(images, Point2D::new(-0.7, 0.0), 1e-6).expect("inner image");
        assert_eq!(inner.parity, Parity::SADDLE);
        assert!((inner.mu + 0.7 / 0.3).abs() < 1e-6);
    }

    /// A jittered `n × n` lattice of image-plane samples over `[-half, half]²`.
    fn jittered_samples(n: usize, half: f64) -> (Vec<Point2D>, BBox) {
        let mut rng = Lcg::new(0x1A2B);
        let step = 2.0 * half / n as f64;
        let mut samples = Vec::new();
        for gy in 0..n {
            for gx in 0..n {
                let x = -half + (gx as f64 + 0.5) * step + 0.2 * step * (rng.f() - 0.5);
                let y = -half + (gy as f64 + 0.5) * step + 0.2 * step * (rng.f() - 0.5);
                samples.push(Point2D::new(x, y));
            }
        }
        (samples, BBox::from_lims(-half, half, -half, half))
    }

    #[test]
    fn pixellation_finds_the_sis_pair() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let grid = Grid2D::square(80, 2.0).unwrap();
        let finder = MapPixellation::build(&lens, &grid);
        assert_sis_pair(&finder.images(&lens, Point2D::new(0.3, 0.0)));
    }

    #[test]
    fn tessellation_finds_the_sis_pair() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let (samples, bbox) = jittered_samples(60, 2.0);
        let finder = MapTessellation::build(&lens, &samples, bbox).unwrap();
        assert_sis_pair(&finder.images(&lens, Point2D::new(0.3, 0.0)));
    }

    #[test]
    fn source_far_outside_has_no_images() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let grid = Grid2D::square(24, 2.0).unwrap();
        let finder = MapPixellation::build(&lens, &grid);
        assert!(finder.images(&lens, Point2D::new(50.0, 50.0)).is_empty());
    }

    #[test]
    fn one_discretization_serves_many_sources() {
        // Build the map once, query several β: each on-axis-offset source is a pair.
        let lens = SIS::new(Point2D::origin(), 1.0);
        let grid = Grid2D::square(80, 2.0).unwrap();
        let finder = MapPixellation::build(&lens, &grid);
        for &bx in &[0.2, 0.4, 0.6] {
            let images = finder.images(&lens, Point2D::new(bx, 0.0));
            assert_eq!(images.len(), 2);
            // Every returned image ray-traces back to its source.
            for im in &images {
                assert!((lens.ray_trace(im.theta) - Point2D::new(bx, 0.0)).norm() < 1e-8);
            }
        }
    }
}
