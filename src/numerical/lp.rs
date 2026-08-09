// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Dense linear programming: two-phase primal simplex, plus the Chebyshev centre
//! of a polytope — the deepest interior point used to seed the ensemble walk.
//!
//! [`lpsolve`] minimizes `cᵀx` over `A_ub x ≤ b_ub`, `A_eq x = b_eq` with per-variable
//! non-negativity (free variables are split `x = x⁺ − x⁻`). It uses Bland's rule so it
//! terminates on degenerate vertices, at the cost of speed — fine for the modest systems
//! (10²–10³ cells) the point-image polytope produces. [`chebyshev_center`] builds the LP
//! `max r s.t. aᵢ·x + ‖aᵢ‖ r ≤ bᵢ` on top: a positive `r` certifies a full-dimensional
//! feasible interior and returns its centre.

use ndarray::{Array1, Array2};

/// Numerical tolerance for reduced costs, ratios and pivots.
const EPS: f64 = 1e-9;
/// Cap on the inscribed-ball radius, keeping the Chebyshev LP bounded on an
/// unbounded (monopole/MSD) polytope.
const R_MAX: f64 = 1.0e6;

/// Outcome of a linear program.
pub(crate) enum LpStatus {
    /// Optimal vertex/solution `x` with objective value `cᵀx` (the value is part of the
    /// contract; not every caller reads it).
    Optimal(Vec<f64>, #[allow(dead_code)] f64),
    /// The constraints admit no feasible point (polyhedron is empty).
    Infeasible,
    /// The objective is unbounded below on the feasible region.
    Unbounded,
}

/// Minimize `cᵀx` subject to `A_ub x ≤ b_ub`, `A_eq x = b_eq`, with `x[j] ≥ 0` where
/// `nonneg[j]` and `x[j]` free otherwise.
///
/// # Arguments
/// * `c` - objective coefficients (length `n`).
/// * `a_ub` - inequality matrix (`k_ub × n`); pass a `0 × n` array for none.
/// * `b_ub` - inequality right-hand sides (length `k_ub`).
/// * `a_eq` - equality matrix (`k_eq × n`); pass a `0 × n` array for none.
/// * `b_eq` - equality right-hand sides (length `k_eq`).
/// * `nonneg` - per-variable non-negativity flags (length `n`).
pub(crate) fn lpsolve(
    c: &[f64],
    a_ub: &Array2<f64>,
    b_ub: &[f64],
    a_eq: &Array2<f64>,
    b_eq: &[f64],
    nonneg: &[bool],
) -> LpStatus {
    let n = c.len();
    let (k_ub, k_eq) = (b_ub.len(), b_eq.len());
    let m = k_ub + k_eq;

    // Map each original variable to its standard-form column(s): one column when
    // non-negative, a (plus, minus) pair when free.
    let mut cols: Vec<(usize, Option<usize>)> = Vec::with_capacity(n);
    let mut n_struct = 0;
    for &nn in nonneg {
        let plus = n_struct;
        n_struct += 1;
        if nn {
            cols.push((plus, None));
        } else {
            cols.push((plus, Some(n_struct)));
            n_struct += 1;
        }
    }
    let slack0 = n_struct;
    n_struct += k_ub; // one slack per inequality row

    let mut a = Array2::<f64>::zeros((m, n_struct));
    let mut b = Array1::<f64>::zeros(m);
    let mut cost = Array1::<f64>::zeros(n_struct);
    for j in 0..n {
        let (plus, minus) = cols[j];
        cost[plus] = c[j];
        if let Some(minus) = minus {
            cost[minus] = -c[j];
        }
    }
    for i in 0..k_ub {
        for j in 0..n {
            let (plus, minus) = cols[j];
            a[[i, plus]] = a_ub[[i, j]];
            if let Some(minus) = minus {
                a[[i, minus]] = -a_ub[[i, j]];
            }
        }
        a[[i, slack0 + i]] = 1.0;
        b[i] = b_ub[i];
    }
    for i in 0..k_eq {
        for j in 0..n {
            let (plus, minus) = cols[j];
            a[[k_ub + i, plus]] = a_eq[[i, j]];
            if let Some(minus) = minus {
                a[[k_ub + i, minus]] = -a_eq[[i, j]];
            }
        }
        b[k_ub + i] = b_eq[i];
    }
    // Standard form needs b ≥ 0; flip any negative row.
    for i in 0..m {
        if b[i] < 0.0 {
            b[i] = -b[i];
            for j in 0..n_struct {
                a[[i, j]] = -a[[i, j]];
            }
        }
    }

    match two_phase(a, b, &cost) {
        None => LpStatus::Infeasible,
        Some(None) => LpStatus::Unbounded,
        Some(Some((z, obj))) => {
            let x = (0..n)
                .map(|j| {
                    let (plus, minus) = cols[j];
                    z[plus] - minus.map_or(0.0, |mi| z[mi])
                })
                .collect();
            LpStatus::Optimal(x, obj)
        }
    }
}

/// The Chebyshev centre of `{x : A x ≤ b}`: the centre of its largest inscribed ball.
///
/// Returns the centre when the polytope has a non-empty interior; `None` when it is
/// empty or lower-dimensional (largest ball radius ≤ 0).
///
/// # Arguments
/// * `a` - inequality matrix (`k × n`).
/// * `b` - right-hand sides (length `k`).
pub(crate) fn chebyshev_center(a: &Array2<f64>, b: &[f64]) -> Option<Vec<f64>> {
    let (k, n) = a.dim();
    // Variables (x, r); rows aᵢ·x + ‖aᵢ‖ r ≤ bᵢ, plus r ≤ R_MAX; maximize r.
    let mut a_ub = Array2::<f64>::zeros((k + 1, n + 1));
    let mut b_ub = vec![0.0; k + 1];
    for i in 0..k {
        let mut norm = 0.0;
        for j in 0..n {
            a_ub[[i, j]] = a[[i, j]];
            norm += a[[i, j]] * a[[i, j]];
        }
        a_ub[[i, n]] = norm.sqrt();
        b_ub[i] = b[i];
    }
    a_ub[[k, n]] = 1.0;
    b_ub[k] = R_MAX;

    let mut c = vec![0.0; n + 1];
    c[n] = -1.0; // minimize −r
    let mut nonneg = vec![false; n + 1];
    nonneg[n] = true; // r ≥ 0

    let empty = Array2::<f64>::zeros((0, n + 1));
    match lpsolve(&c, &a_ub, &b_ub, &empty, &[], &nonneg) {
        LpStatus::Optimal(x, _) => {
            let r = x[n];
            (r > EPS).then(|| x[..n].to_vec())
        }
        _ => None,
    }
}

/// Two-phase primal simplex on standard form `min cᵀx s.t. A x = b, x ≥ 0, b ≥ 0`.
///
/// Returns `None` if infeasible, `Some(None)` if unbounded, else `Some(Some((x, obj)))`.
fn two_phase(a: Array2<f64>, b: Array1<f64>, c: &Array1<f64>) -> Option<Option<(Vec<f64>, f64)>> {
    let (m, n) = a.dim();
    // Phase 1: append an artificial per row, minimize their sum from the trivial basis.
    let mut t = Array2::<f64>::zeros((m, n + m));
    t.slice_mut(ndarray::s![.., ..n]).assign(&a);
    for i in 0..m {
        t[[i, n + i]] = 1.0;
    }
    let mut rhs = b;
    let mut basis: Vec<usize> = (0..m).map(|i| n + i).collect();
    let mut phase1 = Array1::<f64>::zeros(n + m);
    for i in 0..m {
        phase1[n + i] = 1.0;
    }
    // Phase-1 cost (sum of artificials) is bounded below by zero, so it never runs off.
    simplex(&mut t, &mut rhs, &phase1, &mut basis)
        .expect("phase-1 objective is bounded below by zero");
    let infeasibility: f64 = basis
        .iter()
        .zip(rhs.iter())
        .filter(|&(&col, _)| col >= n)
        .map(|(_, &v)| v)
        .sum();
    if infeasibility > 1e-7 {
        return None;
    }

    // Phase 2: drop the artificial columns, optimize the real objective.
    let mut t2 = t.slice(ndarray::s![.., ..n]).to_owned();
    // Any artificial still basic (at value 0) is pivoted out onto a real column if possible.
    for i in 0..m {
        if basis[i] >= n {
            if let Some(q) = (0..n).find(|&j| t2[[i, j]].abs() > EPS) {
                pivot(&mut t2, &mut rhs, i, q);
                basis[i] = q;
            }
        }
    }
    match simplex(&mut t2, &mut rhs, c, &mut basis) {
        None => Some(None),
        Some(()) => {
            let mut x = vec![0.0; n];
            for i in 0..m {
                if basis[i] < n {
                    x[basis[i]] = rhs[i];
                }
            }
            let obj = (0..n).map(|j| c[j] * x[j]).sum();
            Some(Some((x, obj)))
        }
    }
}

/// Run primal simplex (Bland's rule) to optimality in place. Returns `None` if the
/// objective is unbounded below.
///
/// # Arguments
/// * `t` - constraint tableau (`m × n`), canonical for `basis` on entry and exit.
/// * `rhs` - basic-variable values (length `m`).
/// * `cost` - objective coefficients (length `n`).
/// * `basis` - basic column index per row (length `m`).
fn simplex(
    t: &mut Array2<f64>,
    rhs: &mut Array1<f64>,
    cost: &Array1<f64>,
    basis: &mut [usize],
) -> Option<()> {
    let (m, n) = t.dim();
    loop {
        // Reduced costs r_j = c_j − c_B · T_j, with c_B the basic costs.
        let entering = (0..n).find(|&j| {
            let rc = cost[j] - (0..m).map(|i| cost[basis[i]] * t[[i, j]]).sum::<f64>();
            rc < -EPS
        });
        let Some(q) = entering else { return Some(()) };

        // Ratio test: smallest rhs_i / T_iq over positive pivots (Bland tie-break by row).
        let mut leave = None;
        let mut best = f64::INFINITY;
        for i in 0..m {
            if t[[i, q]] > EPS {
                let ratio = rhs[i] / t[[i, q]];
                if ratio < best - EPS {
                    best = ratio;
                    leave = Some(i);
                }
            }
        }
        let Some(r) = leave else { return None };
        pivot(t, rhs, r, q);
        basis[r] = q;
    }
}

/// Gauss–Jordan pivot on element `(r, q)`: scale row `r` to a unit pivot, then clear
/// column `q` from every other row (tableau and right-hand side alike).
fn pivot(t: &mut Array2<f64>, rhs: &mut Array1<f64>, r: usize, q: usize) {
    let (m, n) = t.dim();
    let p = t[[r, q]];
    for j in 0..n {
        t[[r, j]] /= p;
    }
    rhs[r] /= p;
    for i in 0..m {
        if i == r {
            continue;
        }
        let f = t[[i, q]];
        if f.abs() > EPS {
            for j in 0..n {
                t[[i, j]] -= f * t[[r, j]];
            }
            rhs[i] -= f * rhs[r];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn empty(n: usize) -> Array2<f64> {
        Array2::<f64>::zeros((0, n))
    }

    impl LpStatus {
        /// The optimal `(x, objective)`, panicking on any other status.
        fn unwrap_optimal(self) -> (Vec<f64>, f64) {
            match self {
                LpStatus::Optimal(x, obj) => (x, obj),
                _ => panic!("expected Optimal"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "expected Optimal")]
    fn unwrap_optimal_panics_on_non_optimal() {
        LpStatus::Infeasible.unwrap_optimal();
    }

    #[test]
    fn solves_a_basic_bounded_program() {
        // max x + y  s.t. x + 2y ≤ 4, 3x + y ≤ 6, x,y ≥ 0  → optimum at (8/5, 6/5).
        let a = array![[1.0, 2.0], [3.0, 1.0]];
        let c = [-1.0, -1.0];
        let (x, obj) =
            lpsolve(&c, &a, &[4.0, 6.0], &empty(2), &[], &[true, true]).unwrap_optimal();
        assert!((x[0] - 1.6).abs() < 1e-6 && (x[1] - 1.2).abs() < 1e-6, "{x:?}");
        assert!((obj + 2.8).abs() < 1e-6);
    }

    #[test]
    fn detects_infeasibility() {
        // x ≥ 0 with x ≤ −1 (as −x ≥ 1, i.e. x ≤ −1) is infeasible.
        let a = array![[1.0]];
        assert!(matches!(
            lpsolve(&[1.0], &a, &[-1.0], &empty(1), &[], &[true]),
            LpStatus::Infeasible
        ));
    }

    #[test]
    fn detects_unboundedness() {
        // min −x with no upper bound on the free-to-grow x ≥ 0 is unbounded.
        assert!(matches!(
            lpsolve(&[-1.0], &empty(1), &[], &empty(1), &[], &[true]),
            LpStatus::Unbounded
        ));
    }

    #[test]
    fn handles_equalities_and_free_variables() {
        // min x + y s.t. x + y = 1, x free, y ≥ 0 → x = 1 − y and objective = 1,
        // constant → optimum 1 at any feasible point.
        let aeq = array![[1.0, 1.0]];
        let (x, obj) =
            lpsolve(&[1.0, 1.0], &empty(2), &[], &aeq, &[1.0], &[false, true]).unwrap_optimal();
        assert!((x[0] + x[1] - 1.0).abs() < 1e-9);
        assert!((obj - 1.0).abs() < 1e-9);
    }

    #[test]
    fn phase_two_pivots_out_a_basic_artificial() {
        // x + s = 1 together with x = 1 leaves the second row's artificial basic at zero
        // on a row with a nonzero structural coefficient — the phase-2 pivot-out path.
        let aeq = array![[1.0, 1.0], [1.0, 0.0]];
        let (x, _) =
            lpsolve(&[0.0, 0.0], &empty(2), &[], &aeq, &[1.0, 1.0], &[true, true]).unwrap_optimal();
        assert!((x[0] - 1.0).abs() < 1e-9 && x[1].abs() < 1e-9, "{x:?}");
    }

    #[test]
    fn chebyshev_center_of_the_unit_square() {
        // 0 ≤ x ≤ 1, 0 ≤ y ≤ 1 → centre (0.5, 0.5), inscribed radius 0.5.
        let a = array![[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]];
        let c = chebyshev_center(&a, &[0.0, 1.0, 0.0, 1.0]).expect("interior");
        assert!((c[0] - 0.5).abs() < 1e-6 && (c[1] - 0.5).abs() < 1e-6, "{c:?}");
    }

    #[test]
    fn chebyshev_center_of_a_triangle() {
        // x ≥ 0, y ≥ 0, x + y ≤ 1: incentre lies on the line y = x by symmetry.
        let a = array![[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]];
        let c = chebyshev_center(&a, &[0.0, 0.0, 1.0]).expect("interior");
        assert!((c[0] - c[1]).abs() < 1e-6, "{c:?}");
        assert!(c[0] > 0.0 && c[0] + c[1] < 1.0);
    }

    #[test]
    fn chebyshev_center_rejects_an_empty_polytope() {
        // x ≤ 0 and x ≥ 1 (−x ≤ −1) have no interior.
        let a = array![[1.0], [-1.0]];
        assert!(chebyshev_center(&a, &[0.0, -1.0]).is_none());
    }
}
