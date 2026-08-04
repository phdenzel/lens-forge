// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! External shear.
//!
//! Resource: Keeton, Kochanek & Seljak, Shear and Ellipticity in Gravitational
//! Lenses, ApJ 482 (1997),
//! [DOI:10.1086/304172](https://doi.org/10.1086/304172)
//!
//! A constant tidal field with components $(\gamma_1, \gamma_2)$ about centre $c$
//! (with $d = \theta - c$): potential
//! $\psi = \tfrac12[\gamma_1 (d_x^2 - d_y^2) + 2\gamma_2 d_x d_y]$, deflection
//! $\alpha = (\gamma_1 d_x + \gamma_2 d_y,\ \gamma_2 d_x - \gamma_1 d_y)$, constant
//! Hessian $\begin{pmatrix} \gamma_1 & \gamma_2 \\ \gamma_2 & -\gamma_1 \end{pmatrix}$,
//! and $\kappa = 0$.

use crate::grid::Point2D;
use crate::lens::model::{LensJet, LensModel};

/// A constant external shear $(\gamma_1, \gamma_2)$ about `center`.
pub struct ExternalShear {
    center: Point2D,
    gamma1: f64,
    gamma2: f64,
}

impl ExternalShear {
    /// Create an external shear of components `gamma1`, `gamma2` about `center`.
    ///
    /// # Arguments
    /// * `center` - reference point (deflection zero-point).
    /// * `gamma1` - shear component γ₁.
    /// * `gamma2` - shear component γ₂.
    pub fn new(center: Point2D, gamma1: f64, gamma2: f64) -> Self {
        Self {
            center,
            gamma1,
            gamma2,
        }
    }
}

impl LensModel for ExternalShear {
    fn jet(&self, theta: Point2D) -> LensJet {
        let d = theta - self.center;
        let (g1, g2) = (self.gamma1, self.gamma2);
        LensJet {
            psi: 0.5 * (g1 * (d.x * d.x - d.y * d.y) + 2.0 * g2 * d.x * d.y),
            alpha: Point2D::new(g1 * d.x + g2 * d.y, g2 * d.x - g1 * d.y),
            hessian: [[g1, g2], [g2, -g1]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflection_convergence_and_hessian() {
        let lens = ExternalShear::new(Point2D::origin(), 0.1, 0.05);
        let a = lens.deflection(Point2D::new(2.0, 1.0));
        assert!((a.x - 0.25).abs() < 1e-12 && a.y.abs() < 1e-12); // (γ1·2+γ2·1, γ2·2−γ1·1)
        assert!(lens.convergence(Point2D::new(2.0, 1.0)).abs() < 1e-12); // κ = 0
        let h = lens.hessian(Point2D::new(-3.0, 4.0)); // constant, independent of θ
        assert_eq!(h, [[0.1, 0.05], [0.05, -0.1]]);
    }

    #[test]
    fn magnification_matches_shear_formula() {
        let (g1, g2) = (0.12, -0.08);
        let lens = ExternalShear::new(Point2D::new(1.0, -2.0), g1, g2);
        let mu = lens.magnification(Point2D::new(0.5, 0.5));
        assert!((mu - 1.0 / (1.0 - g1 * g1 - g2 * g2)).abs() < 1e-12); // μ = 1/(1−γ₁²−γ₂²)
    }

    #[test]
    fn potential_gradient_matches_deflection() {
        let lens = ExternalShear::new(Point2D::origin(), 0.1, 0.05);
        let h = 1e-5;
        let p = Point2D::new(1.3, -0.7);
        let dx = (lens.potential(Point2D::new(p.x + h, p.y))
            - lens.potential(Point2D::new(p.x - h, p.y)))
            / (2.0 * h);
        let a = lens.deflection(p);
        assert!((dx - a.x).abs() < 1e-6);
    }
}
