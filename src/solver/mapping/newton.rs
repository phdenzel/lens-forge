// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Newton's method on the lens map: refine a seed θ to an image of β.
//!
//! An image is a root of F(θ) = θ − α(θ) − β, whose Jacobian is the lens Jacobian
//! A = I − H(ψ) from the [`jet`](LensModel::jet).

use crate::grid::Point2D;
use crate::lens::LensModel;
use crate::numerical::newton::newton2;

/// Stopping criteria for [`newton`]: the source-plane residual tolerance and the
/// iteration cap.
#[derive(Debug, Clone, Copy)]
pub struct NewtonConfig {
    /// Convergence tolerance on the source-plane residual |θ − α(θ) − β|.
    pub tol: f64,
    /// Maximum number of Newton iterations before giving up.
    pub max_iter: usize,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self { tol: 1e-10, max_iter: 50 }
    }
}

impl NewtonConfig {
    /// Set the source-plane residual tolerance.
    ///
    /// # Arguments
    /// * `tol` - convergence tolerance on |θ − α(θ) − β|.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the iteration cap.
    ///
    /// # Arguments
    /// * `max_iter` - maximum number of Newton iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

/// Refine `seed` to an image θ of the source `beta` under `lens` by Newton's
/// method, or `None` if it fails to converge (including at a near-critical seed
/// where A is singular).
///
/// # Arguments
/// * `lens` - the lens model (its jet supplies both α and the Jacobian A).
/// * `beta` - source-plane position β.
/// * `seed` - starting image-plane position θ₀.
/// * `cfg` - stopping criteria.
pub fn newton(
    lens: &dyn LensModel,
    beta: Point2D,
    seed: Point2D,
    cfg: NewtonConfig,
) -> Option<Point2D> {
    let (theta, info) = newton2(
        |theta| {
            let jet = lens.jet(theta);
            let h = jet.hessian;
            let a = [[1.0 - h[0][0], -h[0][1]], [-h[1][0], 1.0 - h[1][1]]];
            (theta - jet.alpha - beta, a) // (F(θ) = θ − α(θ) − β, A = I − H)
        },
        seed,
        cfg.tol,
        cfg.max_iter,
    );
    info.converged.then_some(theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::{PointMass, SIS};

    #[test]
    fn sis_outer_image_from_a_nearby_seed() {
        // Source at β = (0.3, 0) inside θ_E = 1: outer image at θ = (β + θ_E, 0).
        let lens = SIS::new(Point2D::origin(), 1.0);
        let beta = Point2D::new(0.3, 0.0);
        let theta = newton(&lens, beta, Point2D::new(1.0, 0.1), NewtonConfig::default()).unwrap();
        assert!((theta - Point2D::new(1.3, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn sis_inner_saddle_image_from_a_nearby_seed() {
        // The inner (saddle) SIS image sits at θ = (β − θ_E, 0) = (−0.7, 0).
        let lens = SIS::new(Point2D::origin(), 1.0);
        let beta = Point2D::new(0.3, 0.0);
        let theta = newton(&lens, beta, Point2D::new(-0.8, 0.1), NewtonConfig::default()).unwrap();
        assert!((theta - Point2D::new(-0.7, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn converged_image_ray_traces_back_to_the_source() {
        let lens = PointMass::new(Point2D::origin(), 0.5);
        let beta = Point2D::new(0.2, 0.1);
        let theta = newton(&lens, beta, Point2D::new(0.9, 0.5), NewtonConfig::default()).unwrap();
        assert!((lens.ray_trace(theta) - beta).norm() < 1e-9);
    }

    #[test]
    fn singular_seed_returns_none() {
        // Seeding exactly on the SIS Einstein ring makes A singular (λ_t = 0).
        let lens = SIS::new(Point2D::origin(), 1.0);
        let beta = Point2D::new(0.3, 0.0);
        assert!(newton(&lens, beta, Point2D::new(1.0, 0.0), NewtonConfig::default()).is_none());
    }

    #[test]
    fn config_builder_sets_fields() {
        let cfg = NewtonConfig::default().with_tol(1e-6).with_max_iter(5);
        assert_eq!(cfg.tol, 1e-6);
        assert_eq!(cfg.max_iter, 5);
    }
}
