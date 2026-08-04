// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Singular isothermal sphere (SIS) lens.
//!
//! Resource: Turner, Ostriker & Gott, The statistics of gravitational lenses: the
//! distributions of image angular separations and lens redshifts, ApJ 284 (1984),
//! [DOI:10.1086/162379](https://doi.org/10.1086/162379)
//!
//! An SIS of Einstein radius $\theta_E$ at centre $c$ (with $r = |\theta - c|$):
//! potential $\psi = \theta_E r$, deflection $\alpha = \theta_E (\theta - c)/r$, and
//! convergence $\kappa = \theta_E/(2r)$.

use crate::grid::Point2D;
use crate::lens::model::{LensJet, LensModel};

/// A singular isothermal sphere of Einstein radius `theta_e` centred at `center`.
pub struct SIS {
    center: Point2D,
    theta_e: f64,
}

impl SIS {
    /// Create an SIS of Einstein radius `theta_e` centred at `center`.
    ///
    /// # Arguments
    /// * `center` - lens centre.
    /// * `theta_e` - Einstein radius θ_E.
    pub fn new(center: Point2D, theta_e: f64) -> Self {
        Self { center, theta_e }
    }
}

impl LensModel for SIS {
    fn jet(&self, theta: Point2D) -> LensJet {
        let d = theta - self.center;
        let r2 = d.x * d.x + d.y * d.y;
        let r = r2.sqrt();
        let s = self.theta_e / r; // θ_E / r
        LensJet {
            psi: self.theta_e * r,
            alpha: Point2D::new(s * d.x, s * d.y),
            hessian: [
                [s * (1.0 - d.x * d.x / r2), -s * d.x * d.y / r2],
                [-s * d.x * d.y / r2, s * (1.0 - d.y * d.y / r2)],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflection_and_convergence() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let a = lens.deflection(Point2D::new(2.0, 0.0));
        assert!((a.x - 1.0).abs() < 1e-12 && a.y.abs() < 1e-12); // |α| = θ_E
        assert!((lens.convergence(Point2D::new(2.0, 0.0)) - 0.25).abs() < 1e-12); // θ_E/(2r)
    }

    #[test]
    fn magnification_matches_sis_formula() {
        let (te, r) = (1.0, 2.0);
        let lens = SIS::new(Point2D::origin(), te);
        let mu = lens.magnification(Point2D::new(r, 0.0));
        assert!((mu - 1.0 / (1.0 - te / r)).abs() < 1e-12, "mu={mu}"); // μ = 1/(1 − θ_E/r)
    }

    #[test]
    fn constant_deflection_magnitude() {
        let lens = SIS::new(Point2D::new(-0.5, 0.3), 1.4);
        for p in [Point2D::new(2.0, 1.0), Point2D::new(-3.0, 2.5)] {
            assert!((lens.deflection(p).norm() - 1.4).abs() < 1e-12);
        }
    }
}
