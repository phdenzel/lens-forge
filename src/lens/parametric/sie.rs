// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Singular isothermal ellipsoid (SIE) lens.
//!
//! Resource: Kormann, Schneider & Bartelmann, Isothermal elliptical gravitational
//! lens models, A&A 284 (1994),
//! [ADS:1994A&A...284..285K](https://ui.adsabs.harvard.edu/abs/1994A%26A...284..285K)
//!
//! In the lens-aligned frame with axis ratio $q$ and $\psi = \sqrt{q^2 x^2 + y^2}$, the
//! deflection is $\alpha_x = a \arctan(b x/\psi)$,
//! $\alpha_y = a\,\operatorname{arctanh}(b y/\psi)$ with $a = \theta_E\sqrt{q}/\sqrt{1-q^2}$
//! and $b = \sqrt{1-q^2}$, and the convergence is $\kappa = \theta_E\sqrt{q}/(2\psi)$.
//! The $q \to 1$ limit reduces to the [`SIS`](super::SIS).

use crate::grid::Point2D;
use crate::lens::model::{LensJet, LensModel};

/// A singular isothermal ellipsoid of Einstein radius `theta_e`, axis ratio `q`
/// ∈ (0, 1], and major-axis position angle `phi`, centred at `center`.
pub struct SIE {
    center: Point2D,
    theta_e: f64,
    q: f64,
    phi: f64,
}

impl SIE {
    /// Create an SIE centred at `center`.
    ///
    /// # Arguments
    /// * `center` - lens centre.
    /// * `theta_e` - Einstein radius θ_E.
    /// * `q` - minor/major axis ratio in (0, 1].
    /// * `phi` - major-axis position angle (radians).
    pub fn new(center: Point2D, theta_e: f64, q: f64, phi: f64) -> Self {
        Self {
            center,
            theta_e,
            q,
            phi,
        }
    }
}

impl LensModel for SIE {
    fn jet(&self, theta: Point2D) -> LensJet {
        let dd = theta - self.center;
        let (c, s) = (self.phi.cos(), self.phi.sin());
        // Rotate into the lens-aligned frame (major axis along x).
        let x = c * dd.x + s * dd.y;
        let y = -s * dd.x + c * dd.y;

        let q = self.q;
        let r2 = x * x + y * y;
        let psi = (q * q * x * x + y * y).sqrt();
        let b = (1.0 - q * q).sqrt();
        // Deflection in the aligned frame (SIS branch when q ≈ 1).
        let (ax, ay) = if b < 1e-8 {
            let r = r2.sqrt();
            (self.theta_e * x / r, self.theta_e * y / r)
        } else {
            let a = self.theta_e * q.sqrt() / b;
            (a * (b * x / psi).atan(), a * (b * y / psi).atanh())
        };
        // Hessian in the aligned frame: θ_E√q/(r²ψ) · [[y², -xy], [-xy, x²]].
        let pref = self.theta_e * q.sqrt() / (r2 * psi);
        let (hxx, hxy, hyy) = (pref * y * y, -pref * x * y, pref * x * x);

        // Rotate deflection (vector) and Hessian (tensor) back to the world frame.
        let alpha = Point2D::new(c * ax - s * ay, s * ax + c * ay);
        let (cc, ss, cs) = (c * c, s * s, c * s);
        let hessian = [
            [
                cc * hxx - 2.0 * cs * hxy + ss * hyy,
                cs * (hxx - hyy) + (cc - ss) * hxy,
            ],
            [
                cs * (hxx - hyy) + (cc - ss) * hxy,
                ss * hxx + 2.0 * cs * hxy + cc * hyy,
            ],
        ];
        // ψ is homogeneous of degree 1, so ψ = θ · α (frame-invariant).
        LensJet {
            psi: x * ax + y * ay,
            alpha,
            hessian,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::SIS;

    #[test]
    fn reduces_to_sis_at_q_one() {
        let sie = SIE::new(Point2D::origin(), 1.0, 1.0, 0.0);
        let sis = SIS::new(Point2D::origin(), 1.0);
        for p in [
            Point2D::new(2.0, 0.0),
            Point2D::new(1.0, 1.5),
            Point2D::new(-3.0, 2.0),
        ] {
            assert!(sie.deflection(p).distance(&sis.deflection(p)) < 1e-10);
            assert!((sie.convergence(p) - sis.convergence(p)).abs() < 1e-10);
            assert!((sie.potential(p) - sis.potential(p)).abs() < 1e-10);
        }
    }

    #[test]
    fn convergence_matches_formula() {
        let (te, q) = (1.0, 0.6);
        let sie = SIE::new(Point2D::origin(), te, q, 0.0);
        let p = Point2D::new(1.0, 0.5);
        let psi = (q * q * 1.0 + 0.25_f64).sqrt();
        assert!((sie.convergence(p) - te * q.sqrt() / (2.0 * psi)).abs() < 1e-12);
    }

    #[test]
    fn orientation_rotates_the_field() {
        let (phi, q) = (0.4, 0.6);
        let rotated = SIE::new(Point2D::origin(), 1.0, q, phi);
        let aligned = SIE::new(Point2D::origin(), 1.0, q, 0.0);
        let p = Point2D::new(1.3, 0.7);
        let (c, s) = (phi.cos(), phi.sin());
        // Deflection of the rotated lens = R(φ) · [deflection of aligned lens at R(-φ)p].
        let p0 = Point2D::new(c * p.x + s * p.y, -s * p.x + c * p.y);
        let a0 = aligned.deflection(p0);
        let expect = Point2D::new(c * a0.x - s * a0.y, s * a0.x + c * a0.y);
        assert!(rotated.deflection(p).distance(&expect) < 1e-12);
    }
}
