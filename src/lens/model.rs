// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens models.
//!
//! Every lens implements [`LensModel`]. Its one required
//! method [`jet`](LensModel::jet) returns the potential ψ together with its gradient
//! (deflection α) and Hessian at a point; the derived quantities (convergence,
//! magnification, ray-tracing) fall out as defaults. Consumers (solver, rendering) are
//! written against the trait, agnostic to the representation.
//!
//! Conventions (scaled/dimensionless angular units):
//! * lens equation: $\beta = \theta - \alpha(\theta)$
//! * convergence: $\kappa = \tfrac12 \nabla^2\psi = \tfrac12\,\mathrm{tr}\,H$
//! * deflection: $\alpha = \nabla\psi$
//! * Jacobian: $A = \partial\beta/\partial\theta = I - H(\psi)$, magnification $\mu = 1/\det A$

use crate::fastsum::{Jet, Monopole};
use crate::grid::Point2D;
use std::ops::{Add, AddAssign};

/// A point-mass element `m = κ · area` — the lens-domain name for a fastsum
/// [`Monopole`], the mass a cell contributes to the deflection field.
pub type MassElement = Monopole;

/// Lensing potential ψ and its derivatives at one point: ψ, deflection α = ∇ψ, Hessian H.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct LensJet {
    /// Lensing potential ψ(θ).
    pub psi: f64,
    /// Scaled deflection α = ∇ψ.
    pub alpha: Point2D,
    /// Hessian H = ∂²ψ, row-major `[[ψxx, ψxy], [ψyx, ψyy]]` (symmetric).
    pub hessian: [[f64; 2]; 2],
}

impl From<Jet> for LensJet {
    fn from(j: Jet) -> Self {
        Self {
            psi: j.scalar,
            alpha: j.gradient,
            hessian: j.hessian,
        }
    }
}

impl AddAssign for LensJet {
    /// Superpose another lens's jet: ψ, α and H are all additive.
    fn add_assign(&mut self, rhs: LensJet) {
        self.psi += rhs.psi;
        self.alpha += rhs.alpha;
        for i in 0..2 {
            for j in 0..2 {
                self.hessian[i][j] += rhs.hessian[i][j];
            }
        }
    }
}

impl Add for LensJet {
    type Output = LensJet;
    fn add(mut self, rhs: LensJet) -> LensJet {
        self += rhs;
        self
    }
}

/// A gravitational lens.
pub trait LensModel {
    /// Potential ψ, deflection α = ∇ψ, and Hessian H at `theta`.
    fn jet(&self, theta: Point2D) -> LensJet;

    /// The [`jet`](Self::jet) at each of `thetas`.
    ///
    /// Default maps [`jet`](Self::jet); override to batch when the backend supports it.
    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<LensJet> {
        thetas.iter().map(|&t| self.jet(t)).collect()
    }

    /// Convergence κ(θ) = ½∇²ψ = ½ tr H.
    ///
    /// Override where κ is a primary input rather than the solved trace.
    fn convergence(&self, theta: Point2D) -> f64 {
        let h = self.jet(theta).hessian;
        0.5 * (h[0][0] + h[1][1])
    }

    /// Scaled deflection angle α(θ) = ∇ψ.
    fn deflection(&self, theta: Point2D) -> Point2D {
        self.jet(theta).alpha
    }

    /// Lensing potential ψ(θ).
    fn potential(&self, theta: Point2D) -> f64 {
        self.jet(theta).psi
    }

    /// Hessian of the potential H = ∂²ψ, row-major `[[ψxx, ψxy], [ψyx, ψyy]]`.
    fn hessian(&self, theta: Point2D) -> [[f64; 2]; 2] {
        self.jet(theta).hessian
    }

    /// Map an image-plane position to the source plane via the lens equation.
    fn ray_trace(&self, theta: Point2D) -> Point2D {
        theta - self.deflection(theta)
    }

    /// Signed magnification μ = 1/det(A) with A = I − H(ψ).
    ///
    /// Diverges (returns ±∞) on critical curves where det(A) = 0.
    fn magnification(&self, theta: Point2D) -> f64 {
        let h = self.jet(theta).hessian;
        let a11 = 1.0 - h[0][0];
        let a22 = 1.0 - h[1][1];
        let a12 = -h[0][1];
        let a21 = -h[1][0];
        1.0 / (a11 * a22 - a12 * a21)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lensjet_addition_is_componentwise() {
        let a = LensJet {
            psi: 1.0,
            alpha: Point2D::new(2.0, 3.0),
            hessian: [[1.0, 0.5], [0.5, -1.0]],
        };
        let b = LensJet {
            psi: 0.5,
            alpha: Point2D::new(-1.0, 4.0),
            hessian: [[2.0, 1.0], [1.0, 3.0]],
        };
        let mut c = a;
        c += b;
        assert_eq!(c, a + b);
        assert_eq!(c.psi, 1.5);
        assert_eq!(c.alpha, Point2D::new(1.0, 7.0));
        assert_eq!(c.hessian, [[3.0, 1.5], [1.5, 2.0]]);
    }
}
