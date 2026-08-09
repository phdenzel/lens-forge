// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The hit-and-run sampler.
//!
//! Hit-and-run is a Markov chain whose stationary distribution is uniform on the body: from
//! the current point, pick an isotropic direction and jump to a uniform point on the chord
//! it cuts. After the equality [`reduction`](super::polytope) the inequalities are
//! `Ã y ≤ b̃`; walkers start from the Chebyshev centre and run independently (rayon).
//! Unbounded (recession) directions are clamped to a finite reach.

use ndarray::Array2;
use rayon::prelude::*;

use crate::numerical::lp::chebyshev_center;
use crate::rng::Lcg;

use super::polytope::{
    equality_residual, null_space, particular_solution, reconstruct, Polytope, PolytopeSample,
};

/// Clamp on an unbounded hit-and-run segment (a recession direction), keeping draws finite.
const REACH: f64 = 1.0e6;

/// Draw `walkers × per_walker` points uniformly from `poly` by hit-and-run.
///
/// # Arguments
/// * `poly` - the polytope to sample.
/// * `walkers` - number of independent walkers (run in parallel).
/// * `burn_in` - steps discarded before recording, per walker.
/// * `per_walker` - recorded samples per walker.
/// * `seed` - base RNG seed (each walker derives a distinct stream).
pub(crate) fn hit_and_run(
    poly: &Polytope,
    walkers: usize,
    burn_in: usize,
    per_walker: usize,
    seed: u64,
) -> PolytopeSample {
    let x0 = particular_solution(poly.a_eq, poly.b_eq, poly.n);
    let residual = equality_residual(poly.a_eq, poly.b_eq, &x0);
    if residual > 1e-6 {
        return PolytopeSample::InconsistentEqualities;
    }
    let z = null_space(poly.a_eq, poly.n);

    // Reduced inequalities Ã y ≤ b̃ in the null-space coordinates.
    let ax0 = poly.a_ub.dot(&x0);
    let b_tilde: Vec<f64> = poly.b_ub.iter().zip(ax0.iter()).map(|(bi, a)| bi - a).collect();
    let a_tilde = poly.a_ub.dot(&z);
    let Some(y_c) = chebyshev_center(&a_tilde, &b_tilde) else {
        return PolytopeSample::EmptyInterior;
    };

    let points: Vec<Vec<f64>> = (0..walkers)
        .into_par_iter()
        .map(|w| {
            let mut rng = Lcg::new(seed.wrapping_add((w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let mut y = y_c.clone();
            for _ in 0..burn_in {
                step(&a_tilde, &b_tilde, &mut y, &mut rng);
            }
            let mut out = Vec::with_capacity(per_walker);
            for _ in 0..per_walker {
                step(&a_tilde, &b_tilde, &mut y, &mut rng);
                out.push(reconstruct(&x0, &z, &y));
            }
            out
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();

    PolytopeSample::Ok { points, residual }
}

/// One hit-and-run step: draw an isotropic direction, find the feasible segment against
/// `Ã y ≤ b̃`, and move to a uniform point on it (clamping unbounded reach).
fn step(a: &Array2<f64>, b: &[f64], y: &mut [f64], rng: &mut Lcg) {
    let k = y.len();
    if k == 0 {
        return;
    }
    let mut u: Vec<f64> = (0..k).map(|_| rng.gaussian()).collect();
    let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return;
    }
    for v in &mut u {
        *v /= norm;
    }
    let (mut t_lo, mut t_hi) = (-REACH, REACH);
    for i in 0..a.nrows() {
        let (mut ay, mut au) = (0.0, 0.0);
        for j in 0..k {
            ay += a[[i, j]] * y[j];
            au += a[[i, j]] * u[j];
        }
        let slack = b[i] - ay;
        if au > 1e-12 {
            t_hi = t_hi.min(slack / au);
        } else if au < -1e-12 {
            t_lo = t_lo.max(slack / au);
        }
    }
    if t_hi <= t_lo {
        return; // no room along this direction; keep the point
    }
    let t = rng.range(t_lo, t_hi);
    for j in 0..k {
        y[j] += t * u[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn points_of(s: PolytopeSample) -> Vec<Vec<f64>> {
        match s {
            PolytopeSample::Ok { points, .. } => points,
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn samples_stay_in_the_box_and_fill_it() {
        // The unit square 0 ≤ x,y ≤ 1 (two cells, no equalities).
        let a = array![[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]];
        let b = [0.0, 1.0, 0.0, 1.0];
        let eq = Array2::<f64>::zeros((0, 2));
        let poly = Polytope { a_ub: &a, b_ub: &b, a_eq: &eq, b_eq: &[], n: 2 };
        let pts = points_of(hit_and_run(&poly, 8, 200, 200, 0x5EED));
        assert_eq!(pts.len(), 8 * 200);
        let mut mean = [0.0; 2];
        for p in &pts {
            for d in 0..2 {
                assert!(p[d] >= -1e-9 && p[d] <= 1.0 + 1e-9, "{p:?} out of box");
                mean[d] += p[d];
            }
        }
        for d in 0..2 {
            assert!((mean[d] / pts.len() as f64 - 0.5).abs() < 0.05, "axis {d} off-centre");
        }
    }

    #[test]
    fn inconsistent_equalities_are_reported() {
        let a = array![[-1.0]];
        let eq = array![[1.0], [1.0]];
        let poly = Polytope { a_ub: &a, b_ub: &[0.0], a_eq: &eq, b_eq: &[0.3, 0.7], n: 1 };
        assert!(matches!(hit_and_run(&poly, 1, 1, 1, 1), PolytopeSample::InconsistentEqualities));
    }

    #[test]
    fn an_empty_interior_is_reported() {
        // x ≤ 0 and −x ≤ −1 (x ≥ 1): no interior.
        let a = array![[1.0], [-1.0]];
        let eq = Array2::<f64>::zeros((0, 1));
        let poly = Polytope { a_ub: &a, b_ub: &[0.0, -1.0], a_eq: &eq, b_eq: &[], n: 1 };
        assert!(matches!(hit_and_run(&poly, 1, 1, 1, 1), PolytopeSample::EmptyInterior));
    }

    #[test]
    fn a_fully_determined_system_returns_the_unique_point() {
        // x = 0.5 with x ≥ 0: the null space is empty, so every draw is the unique solution.
        let a = array![[-1.0]];
        let eq = array![[1.0]];
        let poly = Polytope { a_ub: &a, b_ub: &[0.0], a_eq: &eq, b_eq: &[0.5], n: 1 };
        let pts = points_of(hit_and_run(&poly, 2, 1, 3, 1));
        assert!(pts.iter().all(|p| (p[0] - 0.5).abs() < 1e-9), "not unique: {pts:?}");
    }
}
