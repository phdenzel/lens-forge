// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-source image finding: solve β = θ − α(θ) for all images of β.
//!
//! An [`ImageFinder`] returns every image with its magnification, parity, and Fermat
//! potential: every [`MapDiscretization`] has an implementation whose seeds are
//! Newton-polished ([`newton`](super::newton)), deduplicated, and classified by
//! [`Image::at`]. [`PointImageSolver`] wraps the same solve behind the [`Solve`] seam,
//! reporting the aggregate Newton convergence as a [`Solution`]. Build a discretization
//! once and query it for many source positions.

use crate::grid::Point2D;
use crate::lens::LensModel;
use crate::solver::basis::MapDiscretization;
use crate::solver::image::Image;
use crate::solver::solve::{Solve, Solution};
use crate::{Result, SolveInfo};
use super::newton::{newton_with_info, NewtonConfig};

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
        solve_images(lens, self, beta, NewtonConfig::default()).0
    }
}

/// Solve for all images of `beta`: Newton-polish every discretization seed, deduplicate
/// within `DEDUP_TOL`, classify, and aggregate the per-seed Newton convergence.
///
/// The record: `iterations` is the total Newton work over all seed attempts, `residual` the
/// worst source-plane residual among the kept images, and `converged` is `true` unless seeds
/// were produced yet none polished to a valid image (a genuine failure, e.g. a mismatched lens
/// or every seed on a critical curve). A source with no seeds is vacuously converged.
pub(super) fn solve_images<D: MapDiscretization + ?Sized>(
    lens: &dyn LensModel,
    finder: &D,
    beta: Point2D,
    config: NewtonConfig,
) -> (Vec<Image>, SolveInfo) {
    let mut images: Vec<Image> = Vec::new();
    let mut iterations = 0usize;
    let mut residual = 0.0f64;
    let mut had_seed = false;
    for seed in finder.images_of(beta) {
        had_seed = true;
        let (theta, info) = newton_with_info(lens, beta, seed, config);
        iterations += info.iterations;
        if !info.converged {
            continue;
        }
        // Many cells seed the same image; keep only the first of each.
        if !images.iter().any(|im| (im.theta - theta).norm() < DEDUP_TOL) {
            residual = residual.max(info.residual);
            images.push(Image::at(lens, theta, beta));
        }
    }
    let converged = !had_seed || !images.is_empty();
    (images, SolveInfo { iterations, residual, converged })
}

/// A configured point-image solver: solves β = θ − α(θ) for all images of a source and
/// reports the aggregate Newton convergence via [`Solve`] — the forward-side counterpart of
/// [`FreeFormInversion`](crate::solver::inverse::FreeFormInversion).
///
/// Configure it once over a lens and a built discretization, then [`solve`](Solve::solve) many
/// source positions.
pub struct PointImageSolver<'a, D: MapDiscretization + ?Sized> {
    lens: &'a dyn LensModel,
    finder: &'a D,
    config: NewtonConfig,
}

impl<'a, D: MapDiscretization + ?Sized> PointImageSolver<'a, D> {
    /// A solver over `lens` using the built discretization `finder` (default Newton config).
    ///
    /// # Arguments
    /// * `lens` - the lens model (must be the one `finder` was built from).
    /// * `finder` - the built discretization supplying image seeds.
    pub fn new(lens: &'a dyn LensModel, finder: &'a D) -> Self {
        Self { lens, finder, config: NewtonConfig::default() }
    }

    /// Set the Newton stopping criteria.
    ///
    /// # Arguments
    /// * `config` - the Newton tolerance and iteration cap.
    pub fn with_config(mut self, config: NewtonConfig) -> Self {
        self.config = config;
        self
    }
}

impl<D: MapDiscretization + ?Sized> Solve for PointImageSolver<'_, D> {
    /// The source-plane position β to find images of.
    type Input<'a> = Point2D;
    /// The images of β.
    type Output = Vec<Image>;

    fn solve(&self, beta: Point2D) -> Result<Solution<Vec<Image>>> {
        let (images, info) = solve_images(self.lens, self.finder, beta, self.config);
        Ok(Solution::new(images, info))
    }
}

/// The total magnification μ_tot = Σ_i |μ_i| of a point source with the given
/// `images` — the observable flux ratio for an unresolved source.
///
/// # Arguments
/// * `images` - the images of the point source (from an [`ImageFinder`]).
pub fn total_magnification(images: &[Image]) -> f64 {
    images.iter().map(|im| im.mu.abs()).sum()
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

    #[test]
    fn point_image_solver_solves_the_sis_pair_with_convergence() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let finder = MapPixellation::build(&lens, &Grid2D::square(80, 2.0).unwrap());
        let sol = PointImageSolver::new(&lens, &finder).solve(Point2D::new(0.3, 0.0)).unwrap();
        assert_sis_pair(&sol.output);
        assert!(sol.info.converged);
        assert!(sol.info.iterations > 0);
        assert!(sol.info.residual < 1e-9, "residual {}", sol.info.residual); // ≤ Newton tol
    }

    #[test]
    fn point_image_solver_matches_the_finder() {
        // The Solve facade returns exactly the ImageFinder images (same solve, plus a record).
        let lens = SIS::new(Point2D::origin(), 1.0);
        let finder = MapPixellation::build(&lens, &Grid2D::square(64, 2.0).unwrap());
        let beta = Point2D::new(0.25, 0.1);
        let via_solve = PointImageSolver::new(&lens, &finder).solve(beta).unwrap().output;
        let via_finder = finder.images(&lens, beta);
        assert_eq!(via_solve.len(), via_finder.len());
        for (a, b) in via_solve.iter().zip(&via_finder) {
            assert_eq!(a.theta, b.theta);
        }
    }

    #[test]
    fn point_image_solver_reports_no_images_outside() {
        // A source with no covering seeds yields an empty, vacuously-converged solution.
        let lens = SIS::new(Point2D::origin(), 1.0);
        let finder = MapPixellation::build(&lens, &Grid2D::square(24, 2.0).unwrap());
        let sol = PointImageSolver::new(&lens, &finder).solve(Point2D::new(50.0, 50.0)).unwrap();
        assert!(sol.output.is_empty());
        assert!(sol.info.converged);
        assert_eq!(sol.info.iterations, 0);
    }
}
