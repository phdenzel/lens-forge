// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens models.
//!
//! Every lens — parametric or free-form — implements [`LensModel`]. Its one required
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

use crate::fastsum::Jet;
use crate::grid::Point2D;

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

/// A gravitational lens — parametric, external-field, or free-form.
pub trait LensModel {
    /// Potential ψ, deflection α = ∇ψ, and Hessian H at `theta`.
    fn jet(&self, theta: Point2D) -> LensJet;

    /// The [`jet`](Self::jet) at each of `thetas`.
    ///
    /// Default maps [`jet`](Self::jet); free-form lenses override to batch their solver.
    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<LensJet> {
        thetas.iter().map(|&t| self.jet(t)).collect()
    }

    /// Convergence κ(θ) = ½∇²ψ = ½ tr H.
    ///
    /// Free-form lenses override with their input κ (primary there, not the solved trace).
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
