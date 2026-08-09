// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The convex polytope input and the equality null-space reduction shared by the samplers.
//!
//! Every sampler here works over `{x : A_ub x ≤ b_ub, A_eq x = b_eq}`. The equalities are
//! removed once, up front, by moving to an orthonormal basis `Z` of their null space:
//! `x = x₀ + Z y` keeps every draw on the constraint hyperplanes, and (because `Z` is
//! orthonormal) a uniform draw in `y` is a uniform draw in `x` on the subspace. These
//! helpers build `x₀`, `Z`, and map back — the substrate a walk then runs on.

use ndarray::{Array1, Array2};

use crate::numerical::svd::pseudo_inverse;

/// A convex polytope `{x ∈ ℝⁿ : A_ub x ≤ b_ub, A_eq x = b_eq}` as borrowed matrices.
pub(crate) struct Polytope<'a> {
    /// Inequality matrix `A_ub` (`k_ub × n`).
    pub a_ub: &'a Array2<f64>,
    /// Inequality right-hand sides.
    pub b_ub: &'a [f64],
    /// Equality matrix `A_eq` (`k_eq × n`; use a `0 × n` array for none).
    pub a_eq: &'a Array2<f64>,
    /// Equality right-hand sides.
    pub b_eq: &'a [f64],
    /// Number of unknowns `n`.
    pub n: usize,
}

/// The result of a polytope sampling run.
pub(crate) enum PolytopeSample {
    /// Drawn points (each length `n`) with the equality residual of the base point `x₀`.
    Ok {
        /// The drawn points.
        points: Vec<Vec<f64>>,
        /// `‖A_eq x₀ − b_eq‖_∞`.
        residual: f64,
    },
    /// The equalities have no solution.
    InconsistentEqualities,
    /// The polytope has no full-dimensional interior (empty or lower-dimensional).
    EmptyInterior,
}

/// Minimum-norm particular solution `x₀ = Aᵀ(AAᵀ)⁺ b` of `A x = b` (zero when no equalities).
pub(super) fn particular_solution(a: &Array2<f64>, b: &[f64], n: usize) -> Array1<f64> {
    if a.nrows() == 0 {
        return Array1::zeros(n);
    }
    let aat = a.dot(&a.t());
    let pinv = pseudo_inverse(&aat, 1e-10);
    a.t().dot(&pinv.dot(&Array1::from(b.to_vec())))
}

/// Worst-case equality residual `‖A x − b‖_∞` (zero when there are no equalities).
pub(super) fn equality_residual(a: &Array2<f64>, b: &[f64], x: &Array1<f64>) -> f64 {
    if a.nrows() == 0 {
        return 0.0;
    }
    (a.dot(x) - Array1::from(b.to_vec())).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Orthonormal basis (`n × k`) of the null space of `A`, by modified Gram–Schmidt: first
/// orthonormalize the rows, then complete the standard basis against them.
pub(super) fn null_space(a: &Array2<f64>, n: usize) -> Array2<f64> {
    let tol = 1e-9;
    let mut row_basis: Vec<Array1<f64>> = Vec::new();
    for i in 0..a.nrows() {
        let mut v = a.row(i).to_owned();
        for q in &row_basis {
            let p = v.dot(q);
            v = &v - &(q * p);
        }
        let nrm = v.dot(&v).sqrt();
        if nrm > tol {
            row_basis.push(v / nrm);
        }
    }
    let mut null: Vec<Array1<f64>> = Vec::new();
    for j in 0..n {
        let mut v = Array1::<f64>::zeros(n);
        v[j] = 1.0;
        for q in row_basis.iter().chain(null.iter()) {
            let p = v.dot(q);
            v = &v - &(q * p);
        }
        let nrm = v.dot(&v).sqrt();
        if nrm > tol {
            null.push(v / nrm);
        }
    }
    let mut z = Array2::<f64>::zeros((n, null.len()));
    for (col, v) in null.iter().enumerate() {
        for r in 0..n {
            z[[r, col]] = v[r];
        }
    }
    z
}

/// Reconstruct `x = x₀ + Z y` from null-space coordinates.
pub(super) fn reconstruct(x0: &Array1<f64>, z: &Array2<f64>, y: &[f64]) -> Vec<f64> {
    if y.is_empty() {
        return x0.to_vec();
    }
    (x0 + &z.dot(&Array1::from(y.to_vec()))).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn null_space_is_orthonormal_and_annihilates_the_rows() {
        // rank-2 rows in ℝ³ → a one-dimensional null space.
        let a = array![[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let z = null_space(&a, 3);
        assert_eq!(z.ncols(), 1);
        assert!(a.dot(&z).iter().all(|v| v.abs() < 1e-9), "A Z ≠ 0");
        assert!((z.t().dot(&z)[[0, 0]] - 1.0).abs() < 1e-9, "columns not unit-norm");
    }

    #[test]
    fn particular_solution_satisfies_the_equalities() {
        let a = array![[1.0, 1.0]];
        let x = particular_solution(&a, &[1.0], 2);
        assert!((a.dot(&x)[0] - 1.0).abs() < 1e-9);
        assert!(equality_residual(&a, &[1.0], &x) < 1e-9);
    }

    #[test]
    fn no_equalities_give_the_trivial_reduction() {
        let a = Array2::<f64>::zeros((0, 2));
        assert_eq!(particular_solution(&a, &[], 2), Array1::<f64>::zeros(2));
        assert_eq!(equality_residual(&a, &[], &Array1::<f64>::zeros(2)), 0.0);
        assert_eq!(null_space(&a, 2).ncols(), 2); // the whole space is free
    }

    #[test]
    fn reconstruct_maps_reduced_coordinates_to_full() {
        let x0 = array![1.0, 2.0];
        let z = array![[1.0], [0.0]];
        assert_eq!(reconstruct(&x0, &z, &[3.0]), vec![4.0, 2.0]);
        assert_eq!(reconstruct(&x0, &z, &[]), vec![1.0, 2.0]); // empty y → x₀
    }
}
