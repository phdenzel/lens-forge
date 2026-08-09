// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Hit-and-run sampling of the point-image polytope (flat-in-mass).
//!
//! The equalities `C x = d` are removed by working in an orthonormal basis `Z` of their
//! null space: `x = x₀ + Z y` keeps every draw on the constraint hyperplanes, and the
//! inequalities become `Ã y ≤ b̃` with `Ã = A Z`, `b̃ = b − A x₀`. Because `Z` is
//! orthonormal, a uniform draw in `y` is a uniform draw in `x` on the subspace — the
//! flat-in-mass prior. Walkers start from the Chebyshev centre and run independently
//! (rayon); unbounded (monopole/MSD) directions are clamped to a finite reach.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::grid::Point2D;
use crate::numerical::lp::chebyshev_center;
use crate::numerical::svd::pseudo_inverse;
use crate::solver::constraints::ConstraintSet;
use crate::{LensForgeError, Result, SolveInfo};
use crate::rng::Lcg;

/// Clamp on an unbounded hit-and-run segment (a degeneracy direction), keeping draws finite.
const REACH: f64 = 1.0e6;

/// One ensemble draw: the full unknown `x = (m, {β_s})`.
#[derive(Debug, Clone)]
pub struct EnsembleMember {
    /// The unknown vector: `n_cells` masses followed by two coordinates per source.
    pub x: Vec<f64>,
}

impl EnsembleMember {
    /// The cell masses `m` (leading `n_cells` entries).
    ///
    /// # Arguments
    /// * `n_cells` - number of κ cells.
    pub fn masses(&self, n_cells: usize) -> &[f64] {
        &self.x[..n_cells]
    }

    /// The recovered position `β_s` of source `s`.
    ///
    /// # Arguments
    /// * `n_cells` - number of κ cells.
    /// * `s` - source index.
    pub fn source(&self, n_cells: usize, s: usize) -> Point2D {
        Point2D::new(self.x[n_cells + 2 * s], self.x[n_cells + 2 * s + 1])
    }
}

/// A sample of the constraint polytope: the degenerate-solution ensemble.
#[derive(Debug, Clone)]
pub struct Ensemble {
    /// Number of κ cells (leading block of every member's `x`).
    pub n_cells: usize,
    /// Number of sources (each a 2-vector after the cells).
    pub n_sources: usize,
    /// The drawn members.
    pub members: Vec<EnsembleMember>,
}

impl Ensemble {
    /// Number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the ensemble is empty.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// Hit-and-run polytope sampler with independent parallel walkers.
#[derive(Debug, Clone, Copy)]
pub struct HitAndRun {
    /// Number of independent walkers (parallelized across).
    pub walkers: usize,
    /// Steps discarded before recording, per walker.
    pub burn_in: usize,
    /// Recorded samples per walker.
    pub per_walker: usize,
    /// Base RNG seed (walkers derive distinct streams from it).
    pub seed: u64,
}

impl Default for HitAndRun {
    fn default() -> Self {
        Self { walkers: 8, burn_in: 200, per_walker: 200, seed: 0x5EED }
    }
}

impl HitAndRun {
    /// Draw an ensemble from the polytope of `cs` (flat-in-mass).
    ///
    /// # Arguments
    /// * `cs` - the assembled constraint polytope.
    ///
    /// # Errors
    /// Returns a [`ComputationError`](LensForgeError::ComputationError) when the equalities
    /// are inconsistent, or when the polytope has no full-dimensional interior.
    pub fn sample(&self, cs: &ConstraintSet) -> Result<(Ensemble, SolveInfo)> {
        let n = cs.n_unknowns();
        let x0 = particular_solution(&cs.a_eq, &cs.b_eq, n);
        let residual = equality_residual(&cs.a_eq, &cs.b_eq, &x0);
        if residual > 1e-6 {
            return Err(LensForgeError::computation_error(
                "inconsistent equality constraints (no solution)",
            ));
        }
        let z = null_space(&cs.a_eq, n);

        // Reduced inequalities Ã y ≤ b̃ in the null-space coordinates.
        let ax0 = cs.a_ub.dot(&x0);
        let b_tilde: Vec<f64> = cs.b_ub.iter().zip(ax0.iter()).map(|(bi, a)| bi - a).collect();
        let a_tilde = cs.a_ub.dot(&z);
        let y_c = chebyshev_center(&a_tilde, &b_tilde).ok_or_else(|| {
            LensForgeError::computation_error("empty or lower-dimensional constraint polytope")
        })?;

        let per_walker = self.per_walker;
        let burn_in = self.burn_in;
        let members: Vec<EnsembleMember> = (0..self.walkers)
            .into_par_iter()
            .map(|w| {
                let mut rng =
                    Lcg::new(self.seed.wrapping_add((w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                let mut y = y_c.clone();
                for _ in 0..burn_in {
                    hit_and_run_step(&a_tilde, &b_tilde, &mut y, &mut rng);
                }
                let mut out = Vec::with_capacity(per_walker);
                for _ in 0..per_walker {
                    hit_and_run_step(&a_tilde, &b_tilde, &mut y, &mut rng);
                    out.push(EnsembleMember { x: reconstruct(&x0, &z, &y) });
                }
                out
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect();

        let info = SolveInfo { iterations: members.len(), residual, converged: true };
        Ok((Ensemble { n_cells: cs.n_cells, n_sources: cs.n_sources, members }, info))
    }
}

/// Minimum-norm particular solution `x₀ = Cᵀ(CCᵀ)⁺ d` of `C x = d` (zero when no equalities).
fn particular_solution(c: &Array2<f64>, d: &[f64], n: usize) -> Array1<f64> {
    if c.nrows() == 0 {
        return Array1::zeros(n);
    }
    let cct = c.dot(&c.t());
    let pinv = pseudo_inverse(&cct, 1e-10);
    c.t().dot(&pinv.dot(&Array1::from(d.to_vec())))
}

/// Worst-case equality residual `‖C x − d‖_∞` (zero when there are no equalities).
fn equality_residual(c: &Array2<f64>, d: &[f64], x: &Array1<f64>) -> f64 {
    if c.nrows() == 0 {
        return 0.0;
    }
    (c.dot(x) - Array1::from(d.to_vec())).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Orthonormal basis (`n × k`) of the null space of `C`, by modified Gram–Schmidt: first
/// orthonormalize the rows, then complete the standard basis against them.
fn null_space(c: &Array2<f64>, n: usize) -> Array2<f64> {
    let tol = 1e-9;
    let mut row_basis: Vec<Array1<f64>> = Vec::new();
    for i in 0..c.nrows() {
        let mut v = c.row(i).to_owned();
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
fn reconstruct(x0: &Array1<f64>, z: &Array2<f64>, y: &[f64]) -> Vec<f64> {
    if y.is_empty() {
        return x0.to_vec();
    }
    (x0 + &z.dot(&Array1::from(y.to_vec()))).to_vec()
}

/// One hit-and-run step: draw an isotropic direction, find the feasible segment against
/// `Ã y ≤ b̃`, and move to a uniform point on it (clamping unbounded reach).
fn hit_and_run_step(a: &Array2<f64>, b: &[f64], y: &mut [f64], rng: &mut Lcg) {
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
    use crate::grid::{Grid2D, PixelGrid, Point2D, ScalarField2D};
    use crate::lens::SIS;
    use crate::solver::constraints::{ConstraintConfig, ConstraintSet};
    use crate::solver::image::{Image, PointImageSystem};

    /// The unit square `[0,1]²` as a bare polytope (two mass cells, no sources).
    fn unit_square() -> ConstraintSet {
        use ndarray::array;
        ConstraintSet {
            n_cells: 2,
            n_sources: 0,
            a_ub: array![[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
            b_ub: vec![0.0, 1.0, 0.0, 1.0],
            a_eq: Array2::zeros((0, 2)),
            b_eq: vec![],
        }
    }

    #[test]
    fn samples_stay_in_the_box_and_fill_it() {
        let cs = unit_square();
        let (ens, info) = HitAndRun::default().sample(&cs).unwrap();
        assert_eq!(ens.len(), 8 * 200);
        assert!(info.converged);
        let mut mean = [0.0; 2];
        for m in &ens.members {
            for d in 0..2 {
                assert!(m.x[d] >= -1e-9 && m.x[d] <= 1.0 + 1e-9, "{:?} out of box", m.x);
                mean[d] += m.x[d];
            }
        }
        for d in 0..2 {
            mean[d] /= ens.len() as f64;
            assert!((mean[d] - 0.5).abs() < 0.05, "mean {} not centred: {}", d, mean[d]);
        }
    }

    #[test]
    fn member_accessors_split_masses_and_source_positions() {
        let m = EnsembleMember { x: vec![1.0, 2.0, 0.3, -0.4, 0.7, 0.8] };
        assert_eq!(m.masses(2), &[1.0, 2.0]);
        assert_eq!(m.source(2, 0), Point2D::new(0.3, -0.4));
        assert_eq!(m.source(2, 1), Point2D::new(0.7, 0.8));
    }

    #[test]
    fn inconsistent_equalities_are_rejected() {
        use ndarray::array;
        // x = 0.3 and x = 0.7 cannot both hold.
        let cs = ConstraintSet {
            n_cells: 1,
            n_sources: 0,
            a_ub: array![[-1.0]],
            b_ub: vec![0.0],
            a_eq: array![[1.0], [1.0]],
            b_eq: vec![0.3, 0.7],
        };
        assert!(HitAndRun::default().sample(&cs).is_err());
    }

    #[test]
    fn an_empty_interior_is_rejected() {
        use ndarray::array;
        // x ≤ 0 and x ≥ 1 (−x ≤ −1): no full-dimensional interior.
        let cs = ConstraintSet {
            n_cells: 1,
            n_sources: 0,
            a_ub: array![[1.0], [-1.0]],
            b_ub: vec![0.0, -1.0],
            a_eq: Array2::zeros((0, 1)),
            b_eq: vec![],
        };
        assert!(HitAndRun::default().sample(&cs).is_err());
    }

    #[test]
    fn a_fully_determined_system_returns_the_unique_point() {
        use ndarray::array;
        // x = 0.5 with x ≥ 0: the null space is empty, so every member is the unique solution.
        let cs = ConstraintSet {
            n_cells: 1,
            n_sources: 0,
            a_ub: array![[-1.0]],
            b_ub: vec![0.0],
            a_eq: array![[1.0]],
            b_eq: vec![0.5],
        };
        let (ens, _) =
            HitAndRun { walkers: 2, burn_in: 1, per_walker: 3, seed: 1 }.sample(&cs).unwrap();
        assert!(!ens.is_empty());
        assert!(ens.members.iter().all(|m| (m.x[0] - 0.5).abs() < 1e-9), "not unique");
    }

    #[test]
    fn every_member_satisfies_the_lens_polytope() {
        // A real point-image polytope (SIS double on a coarse pixel basis): each draw must
        // satisfy the inequalities and lie on the equality hyperplanes.
        let lens = SIS::new(Point2D::origin(), 1.0);
        let src = Point2D::new(0.15, 0.0);
        let sys = PointImageSystem::from_images(vec![
            Image::at(&lens, Point2D::new(1.15, 0.0), src),
            Image::at(&lens, Point2D::new(-0.85, 0.0), src),
        ]);
        let basis = PixelGrid::constant(Grid2D::square(5, 2.5).unwrap(), 0.3);
        let cs = ConstraintSet::assemble(&basis, &[sys], &ConstraintConfig::default());
        let sampler = HitAndRun { walkers: 4, burn_in: 100, per_walker: 50, seed: 7 };
        let (ens, _) = sampler.sample(&cs).unwrap();
        assert!(!ens.is_empty());
        for m in &ens.members {
            let x = Array1::from(m.x.clone());
            // Inequalities A x ≤ b.
            let ax = cs.a_ub.dot(&x);
            for (i, axi) in ax.iter().enumerate() {
                assert!(*axi <= cs.b_ub[i] + 1e-6, "row {i}: {axi} > {}", cs.b_ub[i]);
            }
            // Equalities C x = d.
            let cxd = cs.a_eq.dot(&x) - Array1::from(cs.b_eq.clone());
            assert!(cxd.iter().all(|v| v.abs() < 1e-6), "equality residual {cxd:?}");
            // Non-negative masses.
            assert!(m.masses(basis.len()).iter().all(|&mc| mc >= -1e-6));
        }
    }
}
