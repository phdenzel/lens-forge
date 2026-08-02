// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Small dense linear algebra: a one-sided Jacobi SVD and the Tikhonov-
//! regularized pseudo-inverse built on it.
//!
//! Sized for the tiny, ill-conditioned check-from-equivalent maps a
//! kernel-independent FMM inverts once per box geometry and then applies as a
//! matvec. Storage and the `dot` products come from [`ndarray`]; only the SVD is
//! hand-rolled. One-sided Jacobi rotates the columns of `A` (m × n, `m ≥ n`) to
//! mutual orthogonality: the rotated columns are `σᵢ uᵢ` and the accumulated
//! rotations form `V`, so `A = U Σ Vᵀ`.

use ndarray::Array2;

/// Rotate the columns of `u` (m × n, `m ≥ n`) to orthogonality by one-sided
/// Jacobi, accumulating the applied rotations into `v`. On return the columns of
/// `u` are `σᵢ uᵢ` and `v` is the orthogonal right-singular-vector matrix.
///
/// # Arguments
/// * `u` - matrix whose columns are orthogonalized in place (starts as `A`).
/// * `v` - accumulator, updated in place (starts as the identity).
fn one_sided_jacobi(u: &mut Array2<f64>, v: &mut Array2<f64>) {
    let (m, n) = u.dim();
    const EPS: f64 = 1e-15;
    const MAX_SWEEPS: usize = 60;
    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for i in 0..n {
            for j in (i + 1)..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for k in 0..m {
                    let (a, b) = (u[[k, i]], u[[k, j]]);
                    alpha += a * a;
                    beta += b * b;
                    gamma += a * b;
                }
                if gamma.abs() <= EPS * (alpha * beta).sqrt() {
                    continue; // columns already orthogonal to tolerance
                }
                rotated = true;
                // Symmetric Schur rotation diagonalizing [[alpha, gamma], [gamma, beta]].
                // signum(0) = 1 in Rust, so alpha == beta gives the 45° rotation.
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for k in 0..m {
                    let (a, b) = (u[[k, i]], u[[k, j]]);
                    u[[k, i]] = c * a - s * b;
                    u[[k, j]] = s * a + c * b;
                }
                for k in 0..n {
                    let (a, b) = (v[[k, i]], v[[k, j]]);
                    v[[k, i]] = c * a - s * b;
                    v[[k, j]] = s * a + c * b;
                }
            }
        }
        if !rotated {
            break;
        }
    }
}

/// Tikhonov-regularized pseudo-inverse of `a` (m × n, `m ≥ n`): the n × m matrix
/// `Aλ⁺ = V diag(σᵢ / (σᵢ² + λ²)) Uᵀ`, damping the ill-conditioned directions.
///
/// # Arguments
/// * `a` - the matrix to invert; must have at least as many rows as columns.
/// * `lambda` - regularization strength (`0` gives the plain pseudo-inverse).
pub(crate) fn pseudo_inverse(a: &Array2<f64>, lambda: f64) -> Array2<f64> {
    let (m, n) = a.dim();
    assert!(m >= n, "pseudo_inverse expects m >= n, got {m}x{n}");
    let mut u = a.clone();
    let mut v = Array2::<f64>::eye(n);
    one_sided_jacobi(&mut u, &mut v);

    // Aλ⁺ = Σᵢ vᵢ gᵢᵀ / (σᵢ² + λ²), with gᵢ = σᵢ uᵢ the i-th rotated column, so
    // σᵢ² = gᵢᵀgᵢ and no division by a tiny σᵢ is needed.
    let lam2 = lambda * lambda;
    let mut pinv = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        let g = u.column(i);
        let denom = g.dot(&g) + lam2;
        if denom == 0.0 {
            continue; // a zero singular direction with no regularization
        }
        let vi = v.column(i);
        for r in 0..n {
            let vr = vi[r] / denom;
            for c in 0..m {
                pinv[[r, c]] += vr * g[c];
            }
        }
    }
    pinv
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    /// Frobenius distance between two equally-shaped matrices.
    fn dist(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn inverts_a_square_nonsingular_matrix() {
        // With lambda = 0 the pseudo-inverse is the true inverse: both products
        // recover the identity.
        let a = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let inv = pseudo_inverse(&a, 0.0);
        let id = Array2::<f64>::eye(3);
        assert!(dist(&inv.dot(&a), &id) < 1e-10, "left inverse");
        assert!(dist(&a.dot(&inv), &id) < 1e-10, "right inverse");
    }

    #[test]
    fn solves_overdetermined_least_squares() {
        // A tall full-rank system: the pseudo-inverse gives the least-squares
        // solution, matching the exact answer for a consistent right-hand side.
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let x_true = array![2.0, -3.0];
        let b: Array1<f64> = a.dot(&x_true);
        let x = pseudo_inverse(&a, 0.0).dot(&b);
        assert!((&x - &x_true).iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-10);
        // It is a left inverse (n x n identity), not a right inverse (m > n).
        assert!(dist(&pseudo_inverse(&a, 0.0).dot(&a), &Array2::eye(2)) < 1e-10);
    }

    #[test]
    fn reconstructs_the_original_matrix() {
        // A A⁺ A = A holds for any pseudo-inverse (checks the SVD factors).
        let a = array![
            [2.0, -1.0, 0.5],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, -2.0],
            [4.0, 0.0, 1.0]
        ];
        let pinv = pseudo_inverse(&a, 0.0);
        assert!(dist(&a.dot(&pinv).dot(&a), &a) < 1e-9);
    }

    #[test]
    fn regularization_bounds_an_ill_conditioned_inverse() {
        // A nearly rank-deficient matrix: lambda = 0 blows the inverse up, while a
        // small lambda keeps its norm bounded.
        let a = array![[1.0, 1.0], [1.0, 1.0 + 1e-8]];
        let raw = pseudo_inverse(&a, 0.0)
            .iter()
            .fold(0.0_f64, |m, &x| m.max(x.abs()));
        let reg = pseudo_inverse(&a, 1e-3)
            .iter()
            .fold(0.0_f64, |m, &x| m.max(x.abs()));
        assert!(raw > 1e6, "unregularized inverse should be huge, got {raw}");
        assert!(
            reg < 1e3,
            "regularized inverse should stay bounded, got {reg}"
        );
    }

    #[test]
    fn identity_is_its_own_inverse() {
        let id = Array2::<f64>::eye(5);
        assert!(dist(&pseudo_inverse(&id, 0.0), &id) < 1e-12);
    }
}
