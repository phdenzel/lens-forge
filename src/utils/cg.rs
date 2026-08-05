// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Conjugate gradient for a symmetric positive-(semi)definite operator supplied
//! as a closure, with an optional Jacobi preconditioner.
//!
//! The operator `M` is passed as an `apply(x, out)` callback (so callers keep
//! their own sparse structure and scratch); `diag` is the diagonal of `M` used as
//! a Jacobi preconditioner, or `None` for plain CG. Representation-agnostic: every
//! SPD system in the crate — diffusion steps, Poisson solves — reduces to this.

use crate::SolveInfo;

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve `M x = rhs` for a symmetric positive-(semi)definite `M` by conjugate
/// gradient, warm-started at `x0`.
///
/// Stops on the relative residual `‖r‖/‖rhs‖ ≤ tol`, on a non-positive curvature
/// direction `pᵀMp ≤ 0` (a null-space direction of a singular `M`), or after
/// `max_iter` steps. A zero `rhs` returns the zero vector immediately.
///
/// # Arguments
/// * `apply` - applies the operator: `apply(x, out)` writes `out = M x`.
/// * `diag` - Jacobi preconditioner (diagonal of `M`, entries `> 0`), or `None`
///   for unpreconditioned CG.
/// * `rhs` - the right-hand side `b`.
/// * `x0` - the initial guess, consumed and returned as the solution.
/// * `tol` - relative residual tolerance `‖r‖ / ‖b‖`.
/// * `max_iter` - maximum iterations.
///
/// Returns the solution and a [`SolveInfo`] (iterations, relative residual `‖r‖/‖b‖`,
/// convergence flag).
pub(crate) fn jacobi_pcg<A>(
    mut apply: A,
    diag: Option<&[f64]>,
    rhs: &[f64],
    x0: Vec<f64>,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, SolveInfo)
where
    A: FnMut(&[f64], &mut [f64]),
{
    let n = rhs.len();
    let rhs_norm2 = dot(rhs, rhs);
    let mut x = x0;
    if rhs_norm2 == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        let info = SolveInfo {
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
        return (x, info);
    }
    let thresh = tol * tol * rhs_norm2;

    // z ← M⁻¹ r: divide by the Jacobi diagonal, or copy for plain CG.
    let precondition = |r: &[f64], z: &mut [f64]| match diag {
        Some(d) => {
            for k in 0..n {
                z[k] = r[k] / d[k];
            }
        }
        None => z.copy_from_slice(r),
    };

    let mut ax = vec![0.0; n];
    apply(&x, &mut ax);
    let mut r: Vec<f64> = (0..n).map(|k| rhs[k] - ax[k]).collect();
    let mut z = vec![0.0; n];
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut ap = vec![0.0; n];
    let mut rs = dot(&r, &r);
    let mut rz = dot(&r, &z);
    let mut iterations = 0;
    let mut converged = rs <= thresh;
    while iterations < max_iter && !converged {
        apply(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            break; // singular null-space direction; nothing more to resolve
        }
        let alpha = rz / pap;
        for k in 0..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        let rs_new = dot(&r, &r);
        iterations += 1;
        if rs_new <= thresh {
            rs = rs_new;
            converged = true;
            break;
        }
        precondition(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        for k in 0..n {
            p[k] = z[k] + beta * p[k];
        }
        rz = rz_new;
        rs = rs_new;
    }
    let info = SolveInfo {
        iterations,
        residual: (rs / rhs_norm2).sqrt(),
        converged,
    };
    (x, info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_spd_system() {
        // M = [[4, 1], [1, 3]], b = [1, 2] -> x = [1/11, 7/11]; Jacobi-preconditioned.
        let apply = |x: &[f64], out: &mut [f64]| {
            out[0] = 4.0 * x[0] + x[1];
            out[1] = x[0] + 3.0 * x[1];
        };
        let (x, info) = jacobi_pcg(apply, Some(&[4.0, 3.0]), &[1.0, 2.0], vec![0.0, 0.0], 1e-12, 100);
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-10);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-10);
        assert!(info.converged && info.residual < 1e-12);
    }

    #[test]
    fn unpreconditioned_solves_and_zero_rhs_returns_zero() {
        // Identity operator M = I, no preconditioner. A real solve exercises `apply`,
        // then a zero right-hand side short-circuits to x = 0 from a non-zero guess.
        let mut apply = |x: &[f64], out: &mut [f64]| out.copy_from_slice(x);
        let (y, _) = jacobi_pcg(&mut apply, None, &[2.0, -3.0], vec![0.0, 0.0], 1e-12, 10);
        assert!((y[0] - 2.0).abs() < 1e-10 && (y[1] + 3.0).abs() < 1e-10);
        let (x, info) = jacobi_pcg(&mut apply, None, &[0.0, 0.0], vec![3.0, -2.0], 1e-12, 10);
        assert_eq!(x, vec![0.0, 0.0]);
        assert!(info.converged && info.iterations == 0);
    }

    #[test]
    fn warm_start_at_the_solution_skips_the_loop() {
        // M = diag(2, 3), b = [2, 6] -> x = [1, 2]. Starting there, the initial
        // residual already meets the tolerance, so no iterations run.
        let apply = |x: &[f64], out: &mut [f64]| {
            out[0] = 2.0 * x[0];
            out[1] = 3.0 * x[1];
        };
        let (x, info) = jacobi_pcg(apply, Some(&[2.0, 3.0]), &[2.0, 6.0], vec![1.0, 2.0], 1e-10, 10);
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
        assert!(info.converged && info.iterations == 0);
    }

    #[test]
    fn breaks_on_non_positive_curvature() {
        // Singular M = diag(1, 0) with the right-hand side in its null space: the
        // search direction has pᵀMp = 0, so the solver bails out unconverged.
        let apply = |x: &[f64], out: &mut [f64]| {
            out[0] = x[0];
            out[1] = 0.0;
        };
        let (_, info) = jacobi_pcg(apply, None, &[0.0, 1.0], vec![0.0, 0.0], 1e-12, 10);
        assert!(!info.converged && info.iterations == 0);
    }
}
