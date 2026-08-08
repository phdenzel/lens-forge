// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Newton's method for a 2-vector root, supplied as a single closure.
//!
//! Solves F(x) = 0 for F: ℝ² → ℝ² by the multivariate Newton iteration
//! x ← x − J(x)⁻¹ F(x), the closed-form 2×2 solve at each step. The residual and
//! its Jacobian are returned *together* by one `eval` callback, so a caller whose
//! value and derivative share work (e.g. a lens jet) computes them once per step.
//! Representation-agnostic: any 2D root-find — image finding on the lens map —
//! reduces to this.

use crate::grid::Point2D;
use crate::SolveInfo;

/// Below this |det J| the step is treated as singular and the iteration stops.
const SINGULAR_EPS: f64 = 1e-14;

/// Find a root of `F` near `x0` by Newton's method, `eval(x)` returning the pair
/// `(F(x), J(x))` with `J` row-major `[[∂Fx/∂x, ∂Fx/∂y], [∂Fy/∂x, ∂Fy/∂y]]`.
///
/// Stops when the residual `‖F(x)‖ ≤ tol`, when `J` turns singular
/// (`|det J| ≤ 1e-14`, e.g. a critical point), on a non-finite iterate, or after
/// `max_iter` steps.
///
/// # Arguments
/// * `eval` - evaluates the residual and its Jacobian: `eval(x) = (F(x), J(x))`.
/// * `x0` - the initial guess.
/// * `tol` - convergence tolerance on the residual norm `‖F(x)‖`.
/// * `max_iter` - maximum iterations.
///
/// Returns the final iterate and a [`SolveInfo`] (iterations, residual `‖F(x)‖`,
/// convergence flag); the iterate is a root only when `converged` is set.
pub(crate) fn newton2<G>(
    mut eval: G,
    x0: Point2D,
    tol: f64,
    max_iter: usize,
) -> (Point2D, SolveInfo)
where
    G: FnMut(Point2D) -> (Point2D, [[f64; 2]; 2]),
{
    let mut x = x0;
    let mut iterations = 0;
    let (mut residual, converged);
    loop {
        let (fx, j) = eval(x);
        residual = fx.norm();
        if residual <= tol {
            converged = true;
            break;
        }
        if iterations >= max_iter {
            converged = false;
            break;
        }
        // Solve J Δ = F(x) via the closed-form 2×2 inverse; x ← x − Δ.
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det.abs() <= SINGULAR_EPS {
            converged = false;
            break;
        }
        let dx = (j[1][1] * fx.x - j[0][1] * fx.y) / det;
        let dy = (j[0][0] * fx.y - j[1][0] * fx.x) / det;
        let next = x - Point2D::new(dx, dy);
        if !next.is_finite() {
            converged = false;
            break;
        }
        x = next;
        iterations += 1;
    }
    (x, SolveInfo { iterations, residual, converged })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F(x) = x − c is linear: Newton reaches the root `c` in one step.
    #[test]
    fn linear_root_in_one_step() {
        let c = Point2D::new(1.5, -2.0);
        let (x, info) = newton2(
            |p| (p - c, [[1.0, 0.0], [0.0, 1.0]]),
            Point2D::origin(),
            1e-12,
            10,
        );
        assert!((x - c).norm() < 1e-12);
        assert!(info.converged && info.iterations == 1);
    }

    /// A nonlinear system: F = (x² − y − 1, x − y). Root at x = y = φ (golden ratio).
    #[test]
    fn nonlinear_system_converges() {
        let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
        let (x, info) = newton2(
            |p| {
                let f = Point2D::new(p.x * p.x - p.y - 1.0, p.x - p.y);
                let j = [[2.0 * p.x, -1.0], [1.0, -1.0]];
                (f, j)
            },
            Point2D::new(1.0, 1.0),
            1e-12,
            50,
        );
        assert!((x - Point2D::new(phi, phi)).norm() < 1e-10);
        assert!(info.converged);
    }

    /// A singular Jacobian stops the iteration unconverged.
    #[test]
    fn singular_jacobian_stops() {
        let (_, info) = newton2(
            |_| (Point2D::new(1.0, 1.0), [[1.0, 1.0], [1.0, 1.0]]),
            Point2D::origin(),
            1e-12,
            10,
        );
        assert!(!info.converged && info.iterations == 0);
    }

    /// Starting at the root reports convergence with zero iterations.
    #[test]
    fn seed_at_root_takes_no_steps() {
        let (x, info) = newton2(
            |p| (p, [[1.0, 0.0], [0.0, 1.0]]),
            Point2D::origin(),
            1e-12,
            10,
        );
        assert_eq!(x, Point2D::origin());
        assert!(info.converged && info.iterations == 0);
    }
}
