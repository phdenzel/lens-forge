// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Jacobi-preconditioned conjugate gradient for a symmetric positive-definite
//! operator supplied as a closure.
//!
//! The operator `M` is passed as an `apply(x, out)` callback (so callers keep
//! their own sparse structure and scratch), with the diagonal of `M` as the
//! Jacobi preconditioner. Representation-agnostic: any SPD system reduces to this.

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve `M x = rhs` for a symmetric positive-definite `M` by Jacobi-preconditioned
/// conjugate gradient, warm-started at `x0`.
///
/// # Arguments
/// * `apply` - applies the operator: `apply(x, out)` writes `out = M x`.
/// * `diag` - the diagonal of `M`, the Jacobi preconditioner (entries `> 0`).
/// * `rhs` - the right-hand side `b`.
/// * `x0` - the initial guess, consumed and returned as the solution.
/// * `tol` - relative residual tolerance `‖r‖ / ‖b‖`.
/// * `max_iter` - maximum iterations.
///
/// Returns the solution `x` (the zero vector when `rhs` is zero).
pub(crate) fn jacobi_pcg<A>(
    mut apply: A,
    diag: &[f64],
    rhs: &[f64],
    x0: Vec<f64>,
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    A: FnMut(&[f64], &mut [f64]),
{
    let n = rhs.len();
    let rhs_norm2 = dot(rhs, rhs);
    let mut x = x0;
    if rhs_norm2 == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        return x;
    }
    let thresh = tol * tol * rhs_norm2;

    let mut mx = vec![0.0; n];
    apply(&x, &mut mx);
    let mut r: Vec<f64> = (0..n).map(|i| rhs[i] - mx[i]).collect();
    let mut z: Vec<f64> = (0..n).map(|i| r[i] / diag[i]).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut mp = vec![0.0; n];
    for _ in 0..max_iter {
        if dot(&r, &r) <= thresh {
            break;
        }
        apply(&p, &mut mp);
        let alpha = rz / dot(&p, &mp);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * mp[i];
        }
        for i in 0..n {
            z[i] = r[i] / diag[i];
        }
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_spd_system() {
        // M = [[4, 1], [1, 3]], b = [1, 2] -> x = [1/11, 7/11].
        let apply = |x: &[f64], out: &mut [f64]| {
            out[0] = 4.0 * x[0] + x[1];
            out[1] = x[0] + 3.0 * x[1];
        };
        let x = jacobi_pcg(apply, &[4.0, 3.0], &[1.0, 2.0], vec![0.0, 0.0], 1e-12, 100);
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-10);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-10);
    }

    #[test]
    fn zero_rhs_returns_zero() {
        // Identity operator M = I. A real solve exercises `apply`, and a zero
        // right-hand side then short-circuits to x = 0 (from a non-zero guess).
        let mut apply = |x: &[f64], out: &mut [f64]| out.copy_from_slice(x);
        let y = jacobi_pcg(&mut apply, &[1.0, 1.0], &[2.0, -3.0], vec![0.0, 0.0], 1e-12, 10);
        assert!((y[0] - 2.0).abs() < 1e-10 && (y[1] + 3.0).abs() < 1e-10);
        let x = jacobi_pcg(&mut apply, &[1.0, 1.0], &[0.0, 0.0], vec![3.0, -2.0], 1e-12, 10);
        assert_eq!(x, vec![0.0, 0.0]);
    }
}
