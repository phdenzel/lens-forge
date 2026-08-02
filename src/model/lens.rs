// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens models.
//!
//! Every lens — parametric or free-form — implements [`LensModel`], exposing the
//! four quantities the rest of the library needs (convergence $\kappa$, deflection
//! $\alpha$, potential $\psi$, Hessian $\partial\alpha/\partial\theta$); consumers
//! (solver, magnification, rendering) are written against the trait, agnostic to
//! the representation.
//!
//! Conventions (scaled/dimensionless angular units):
//! * lens equation: $\beta = \theta - \alpha(\theta)$
//! * convergence: $\kappa = \tfrac12 \nabla^2\psi$
//! * deflection: $\alpha = \nabla\psi$
//! * Jacobian: $A = \partial\beta/\partial\theta = I - H(\psi)$, magnification $\mu = 1/\det A$

use crate::grid::Point2D;

/// A gravitational lens — analytic or pixelated/adaptive.
pub trait LensModel {
    /// Convergence κ(θ) = Σ/Σ_cr = ½∇²ψ.
    fn convergence(&self, theta: Point2D) -> f64;

    /// Scaled deflection angle α(θ) = ∇ψ.
    fn deflection(&self, theta: Point2D) -> Point2D;

    /// Lensing potential ψ(θ).
    fn potential(&self, theta: Point2D) -> f64;

    /// Hessian of the potential H = ∂α/∂θ = ∂²ψ/∂θ_i∂θ_j, row-major `[[ψxx, ψxy], [ψyx, ψyy]]`.
    fn hessian(&self, theta: Point2D) -> [[f64; 2]; 2];

    /// Map an image-plane position to the source plane via the lens equation.
    fn ray_trace(&self, theta: Point2D) -> Point2D {
        theta - self.deflection(theta)
    }

    /// Signed magnification μ = 1/det(A) with A = I − H(ψ).
    ///
    /// Diverges (returns ±∞) on critical curves where det(A) = 0.
    fn magnification(&self, theta: Point2D) -> f64 {
        let h = self.hessian(theta);
        let a11 = 1.0 - h[0][0];
        let a22 = 1.0 - h[1][1];
        let a12 = -h[0][1];
        let a21 = -h[1][0];
        1.0 / (a11 * a22 - a12 * a21)
    }
}
