// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Navarro–Frenk–White (NFW) lens.
//!
//! Resource: Wright & Brainerd, Gravitational Lensing by NFW Halos, ApJ 534 (2000),
//! [DOI:10.1086/308744](https://doi.org/10.1086/308744)
//!
//! Circularly symmetric, with scale radius $r_s$, characteristic convergence
//! $\kappa_s = \rho_s r_s / \Sigma_\mathrm{cr}$, and $x = |\theta - c|/r_s$. Writing
//! $F(x) = \operatorname{arccosh}(1/x)/\sqrt{1-x^2}$ for $x<1$ and
//! $\arccos(1/x)/\sqrt{x^2-1}$ for $x>1$, the convergence is
//! $\kappa = 2\kappa_s (1-F)/(x^2-1)$ and the radial deflection is
//! $\alpha = 4\kappa_s r_s\,[\ln(x/2) + F]/x$.

use crate::grid::Point2D;
use crate::lens::model::{LensJet, LensModel};

/// An NFW halo of scale radius `r_s` and characteristic convergence `kappa_s`.
pub struct NFW {
    center: Point2D,
    r_s: f64,
    kappa_s: f64,
}

impl NFW {
    /// Create an NFW halo centred at `center`.
    ///
    /// # Arguments
    /// * `center` - lens centre.
    /// * `r_s` - scale radius r_s.
    /// * `kappa_s` - characteristic convergence κ_s = ρ_s r_s / Σ_cr.
    pub fn new(center: Point2D, r_s: f64, kappa_s: f64) -> Self {
        Self {
            center,
            r_s,
            kappa_s,
        }
    }
}

impl LensModel for NFW {
    fn jet(&self, theta: Point2D) -> LensJet {
        let d = theta - self.center;
        let r2 = d.x * d.x + d.y * d.y;
        let r = r2.sqrt();
        let x = r / self.r_s;
        let ks = self.kappa_s;

        // Radial function F(x) and the potential's extra term ∫F/x dx.
        let (fx, psi_extra) = if x < 1.0 - 1e-6 {
            let ac = (1.0 / x).acosh(); // arccosh(1/x)
            (ac / (1.0 - x * x).sqrt(), -0.5 * ac * ac)
        } else if x > 1.0 + 1e-6 {
            let ac = (1.0 / x).acos(); // arccos(1/x)
            (ac / (x * x - 1.0).sqrt(), 0.5 * ac * ac)
        } else {
            (1.0, 0.0) // x ≈ 1
        };
        let g = (0.5 * x).ln() + fx; // g(x) = ln(x/2) + F(x)

        // Convergence κ(x) (limit 2κ_s/3 at x = 1) and mean convergence α/R.
        let kappa = if (x - 1.0).abs() < 1e-4 {
            2.0 * ks / 3.0
        } else {
            2.0 * ks * (1.0 - fx) / (x * x - 1.0)
        };
        let mean_kappa = 4.0 * ks * g / (x * x); // = |α|/R
        let gamma = kappa - mean_kappa; // Cartesian shear coeff (B'R/2)

        // α = mean_κ · d; H = κ I + γ · [[cos2φ, sin2φ], [sin2φ, -cos2φ]].
        let cos2 = (d.x * d.x - d.y * d.y) / r2;
        let sin2 = 2.0 * d.x * d.y / r2;
        LensJet {
            psi: 4.0 * ks * self.r_s * self.r_s * (0.5 * (0.5 * x).ln().powi(2) + psi_extra),
            alpha: Point2D::new(mean_kappa * d.x, mean_kappa * d.y),
            hessian: [
                [kappa + gamma * cos2, gamma * sin2],
                [gamma * sin2, kappa - gamma * cos2],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens() -> NFW {
        NFW::new(Point2D::new(0.2, -0.1), 1.3, 0.8)
    }

    #[test]
    fn convergence_at_scale_radius() {
        // κ(r_s) = 2κ_s/3.
        let l = NFW::new(Point2D::origin(), 1.0, 0.8);
        assert!((l.convergence(Point2D::new(1.0, 0.0)) - 2.0 * 0.8 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn potential_gradient_matches_deflection() {
        let l = lens();
        let h = 1e-5;
        for p in [Point2D::new(0.9, 0.3), Point2D::new(2.4, -1.1)] {
            let dx = (l.potential(Point2D::new(p.x + h, p.y))
                - l.potential(Point2D::new(p.x - h, p.y)))
                / (2.0 * h);
            let dy = (l.potential(Point2D::new(p.x, p.y + h))
                - l.potential(Point2D::new(p.x, p.y - h)))
                / (2.0 * h);
            let a = l.deflection(p);
            assert!(
                (dx - a.x).abs() < 1e-5 && (dy - a.y).abs() < 1e-5,
                "∇ψ ≠ α at {p:?}"
            );
        }
    }

    #[test]
    fn deflection_divergence_matches_convergence() {
        let l = lens();
        let h = 1e-5;
        // κ = ½ ∇·α, covering both the x<1 and x>1 branches.
        for p in [Point2D::new(0.7, 0.2), Point2D::new(2.0, -0.9)] {
            let dax = (l.deflection(Point2D::new(p.x + h, p.y)).x
                - l.deflection(Point2D::new(p.x - h, p.y)).x)
                / (2.0 * h);
            let day = (l.deflection(Point2D::new(p.x, p.y + h)).y
                - l.deflection(Point2D::new(p.x, p.y - h)).y)
                / (2.0 * h);
            assert!(
                (0.5 * (dax + day) - l.convergence(p)).abs() < 1e-5,
                "κ ≠ ½∇·α at {p:?}"
            );
        }
    }

    #[test]
    fn hessian_matches_finite_difference() {
        let l = lens();
        let h = 1e-5;
        let p = Point2D::new(1.5, 0.6);
        let hij = l.hessian(p);
        let dxx = (l.deflection(Point2D::new(p.x + h, p.y)).x
            - l.deflection(Point2D::new(p.x - h, p.y)).x)
            / (2.0 * h);
        let dxy = (l.deflection(Point2D::new(p.x, p.y + h)).x
            - l.deflection(Point2D::new(p.x, p.y - h)).x)
            / (2.0 * h);
        assert!((hij[0][0] - dxx).abs() < 1e-5 && (hij[0][1] - dxy).abs() < 1e-5);
    }
}
