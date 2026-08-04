// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-mass lens.

use crate::grid::Point2D;
use crate::lens::model::{LensJet, LensModel};

/// A point mass of Einstein radius θ_E: ψ = θ_E² ln|θ−c|, α = θ_E² (θ−c)/|θ−c|²,
/// with κ = 0 everywhere except the (singular) centre.
pub struct PointMass {
    center: Point2D,
    theta_e: f64,
}

impl PointMass {
    /// Create a point mass of Einstein radius `theta_e` centred at `center`.
    ///
    /// # Arguments
    /// * `center` - lens centre.
    /// * `theta_e` - Einstein radius θ_E.
    pub fn new(center: Point2D, theta_e: f64) -> Self {
        Self { center, theta_e }
    }
}

impl LensModel for PointMass {
    fn jet(&self, theta: Point2D) -> LensJet {
        let d = theta - self.center;
        let r2 = d.x * d.x + d.y * d.y;
        let te2 = self.theta_e * self.theta_e;
        let s = te2 / r2; // θ_E² / r²
        LensJet {
            psi: 0.5 * te2 * r2.ln(),
            alpha: Point2D::new(s * d.x, s * d.y),
            hessian: [
                [s * (1.0 - 2.0 * d.x * d.x / r2), -2.0 * s * d.x * d.y / r2],
                [-2.0 * s * d.x * d.y / r2, s * (1.0 - 2.0 * d.y * d.y / r2)],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflection_and_convergence() {
        let lens = PointMass::new(Point2D::origin(), 1.5);
        let p = Point2D::new(3.0, 0.0);
        let a = lens.deflection(p);
        assert!((a.x - 1.5 * 1.5 / 3.0).abs() < 1e-12); // θ_E²/r = 2.25/3
        assert!(a.y.abs() < 1e-12);
        assert!(lens.convergence(p).abs() < 1e-12); // κ = 0 away from the centre
    }

    #[test]
    fn magnification_matches_point_mass_formula() {
        let te = 1.0;
        let lens = PointMass::new(Point2D::origin(), te);
        let r = 2.0;
        let mu = lens.magnification(Point2D::new(r, 0.0));
        let expect = 1.0 / (1.0 - te.powi(4) / r.powi(4)); // μ = 1/(1 − θ_E⁴/r⁴)
        assert!((mu - expect).abs() < 1e-12, "mu={mu} exp={expect}");
    }

    #[test]
    fn center_offset_shifts_field() {
        let c = Point2D::new(1.0, -0.5);
        let lens = PointMass::new(c, 1.0);
        // At centre + (2, 0) the deflection is radial with magnitude θ_E²/r = 0.5.
        let a = lens.deflection(Point2D::new(3.0, -0.5));
        assert!((a.x - 0.5).abs() < 1e-12 && a.y.abs() < 1e-12);
    }
}
