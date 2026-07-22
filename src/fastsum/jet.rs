// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`PotentialJet`]: the potential and its derivatives at a query point.
//!
//! Holds the 2-jet of ψ — value, gradient (deflection) and Hessian — the
//! order-≤2 truncation of a [`LocalExpansion`](super::local::LocalExpansion).
//! The lensing observables κ, shear and magnification are derived from it.

use crate::grid::Point2D;
use std::f64::consts::FRAC_1_PI;

/// The potential ψ and its derivatives (deflection, Hessian) at one point.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PotentialJet {
    /// Potential ψ.
    pub potential: f64,
    /// Deflection α = ∇ψ.
    pub deflection: Point2D,
    /// Row-major Hessian $\begin{bmatrix} \psi_{xx} & \psi_{xy} \\\\ \psi_{yx} & \psi_{yy} \end{bmatrix}$ (symmetric).
    pub hessian: [[f64; 2]; 2],
}

impl PotentialJet {
    /// Accumulate the contribution of a (softened) monopole at `source`.
    ///
    /// The shared near-field / leaf kernel used by every backend, so they stay
    /// numerically consistent.
    ///
    /// # Arguments
    /// * `theta` - query point.
    /// * `source` - monopole position.
    /// * `strength` - monopole coefficient.
    /// * `eps2` - squared softening length added to $r^2$.
    #[inline]
    pub fn add_monopole(&mut self, theta: Point2D, source: Point2D, strength: f64, eps2: f64) {
        let dx = theta.x - source.x;
        let dy = theta.y - source.y;
        let r2 = dx * dx + dy * dy + eps2;
        if r2 <= 0.0 {
            return; // coincident with an un-softened source: skip the singularity
        }
        let inv = 1.0 / r2;
        let k = strength * FRAC_1_PI;
        self.potential += 0.5 * k * r2.ln();
        self.deflection.x += k * dx * inv;
        self.deflection.y += k * dy * inv;
        let inv2 = inv * inv;
        let hxx = k * (inv - 2.0 * dx * dx * inv2);
        let hxy = k * (-2.0 * dx * dy * inv2);
        let hyy = k * (inv - 2.0 * dy * dy * inv2);
        self.hessian[0][0] += hxx;
        self.hessian[0][1] += hxy;
        self.hessian[1][0] += hxy;
        self.hessian[1][1] += hyy;
    }

    /// Convergence κ = ½∇²ψ = ½ tr H.
    #[inline]
    pub fn convergence(&self) -> f64 {
        0.5 * (self.hessian[0][0] + self.hessian[1][1])
    }

    /// Shear `(γ₁, γ₂) = (½(ψxx − ψyy), ψxy)`.
    #[inline]
    pub fn shear(&self) -> (f64, f64) {
        (0.5 * (self.hessian[0][0] - self.hessian[1][1]), self.hessian[0][1])
    }

    /// Signed magnification μ = 1/det(A) with A = I − H.
    #[inline]
    pub fn magnification(&self) -> f64 {
        let a11 = 1.0 - self.hessian[0][0];
        let a22 = 1.0 - self.hessian[1][1];
        1.0 / (a11 * a22 - self.hessian[0][1] * self.hessian[1][0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_observables_match_the_hessian() {
        // A jet with a known Hessian: κ, shear and μ follow directly.
        let mut j = PotentialJet::default();
        j.hessian = [[0.3, 0.1], [0.1, -0.1]];
        assert!((j.convergence() - 0.1).abs() < 1e-15);
        let (g1, g2) = j.shear();
        assert!((g1 - 0.2).abs() < 1e-15 && (g2 - 0.1).abs() < 1e-15);
        // det(A) = (1-0.3)(1+0.1) - 0.1² = 0.77 - 0.01 = 0.76.
        assert!((j.magnification() - 1.0 / 0.76).abs() < 1e-12);
    }

    #[test]
    fn single_monopole_is_trace_free_away_from_the_source() {
        // Away from the (unsoftened) source ∇²ψ = 0, so κ = ½ tr H = 0.
        let mut j = PotentialJet::default();
        j.add_monopole(Point2D::new(3.0, -2.0), Point2D::origin(), 0.5, 0.0);
        assert!(j.convergence().abs() < 1e-15);
    }
}
