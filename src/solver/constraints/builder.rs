// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Assembling the point-image polytope `A x ≤ b, C x = d` from images and a κ basis.
//!
//! The unknown is `x = (m, {β_s})`: the per-cell masses `m_c = κ_c A_c` plus two free
//! coordinates per source. Sampling in mass coordinates makes the flat-in-mass prior exact
//! (uniform-in-`m`). Every row is affine because α and ψ are linear in the masses and each
//! Fermat difference is linear once the source position β is an explicit free unknown.

use ndarray::Array2;

use crate::numerical::lp::{self, LpStatus};
use crate::solver::image::PointImageSystem;

use super::influence::{InfluenceMatrix, KappaBasis};

/// Options controlling which regularizing inequalities the builder emits.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintConfig {
    /// Optional convergence-gradient bound `g`: `|κ_i − κ_j| ≤ g · d_ij` on adjacent cells.
    pub grad_bound: Option<f64>,
}

/// The point-image polytope: inequalities `A x ≤ b` and equalities `C x = d` over the
/// unknown `x = (m, {β_s})`.
pub struct ConstraintSet {
    /// Number of κ cells (leading block of `x`).
    pub n_cells: usize,
    /// Number of sources (each contributes a 2-vector `β_s` after the cells).
    pub n_sources: usize,
    /// Inequality matrix `A` (`k_ub × n_unknowns`).
    pub a_ub: Array2<f64>,
    /// Inequality right-hand sides.
    pub b_ub: Vec<f64>,
    /// Equality matrix `C` (`k_eq × n_unknowns`).
    pub a_eq: Array2<f64>,
    /// Equality right-hand sides.
    pub b_eq: Vec<f64>,
}

impl ConstraintSet {
    /// Assemble the polytope for `basis` constrained by `systems`.
    ///
    /// Emits, per source: two position equalities per image (`θ − α(θ) = β_s`); a Fermat
    /// time-delay equality per measured delay, or arrival-order inequalities when only the
    /// order is known. Globally: `κ ≥ 0` per cell, and optional adjacent-cell gradient
    /// bounds. β-coordinates are appended after the cells in source order.
    ///
    /// # Arguments
    /// * `basis` - the convergence basis (pixel or mesh).
    /// * `systems` - the point-image systems sharing this basis.
    /// * `config` - which regularizing inequalities to add.
    pub fn assemble<B: KappaBasis>(
        basis: &B,
        systems: &[PointImageSystem],
        config: &ConstraintConfig,
    ) -> Self {
        let n_cells = basis.len();
        let n_sources = systems.len();
        let n = n_cells + 2 * n_sources;
        let areas: Vec<f64> = (0..n_cells).map(|c| basis.cell_area(c)).collect();

        let mut ub: Vec<Vec<f64>> = Vec::new();
        let mut b_ub: Vec<f64> = Vec::new();
        let mut eq: Vec<Vec<f64>> = Vec::new();
        let mut b_eq: Vec<f64> = Vec::new();

        for (s, sys) in systems.iter().enumerate() {
            let thetas: Vec<_> = sys.images.iter().map(|im| im.theta).collect();
            let infl = InfluenceMatrix::build(basis, &thetas);
            let (bx, by) = (n_cells + 2 * s, n_cells + 2 * s + 1);

            // Position equalities: α(θ_i) + β_s = θ_i.
            for i in 0..thetas.len() {
                let (ax, ay) = infl.alpha_rows(i);
                let mut rx = vec![0.0; n];
                let mut ry = vec![0.0; n];
                for c in 0..n_cells {
                    rx[c] = ax[c] / areas[c];
                    ry[c] = ay[c] / areas[c];
                }
                rx[bx] = 1.0;
                ry[by] = 1.0;
                eq.push(rx);
                b_eq.push(thetas[i].x);
                eq.push(ry);
                b_eq.push(thetas[i].y);
            }

            // Fermat difference τ_j − τ_i as an affine form L(x) with the constant carried
            // to the right-hand side (the β² term cancels in the difference).
            let fermat_form = |i: usize, j: usize| -> (Vec<f64>, f64) {
                let mut row = vec![0.0; n];
                let pi = infl.psi_row(i);
                let pj = infl.psi_row(j);
                for c in 0..n_cells {
                    row[c] = -(pj[c] - pi[c]) / areas[c];
                }
                let (ti, tj) = (thetas[i], thetas[j]);
                row[bx] = -(tj.x - ti.x);
                row[by] = -(tj.y - ti.y);
                let constant = 0.5 * (tj.norm_squared() - ti.norm_squared());
                (row, constant)
            };

            if !sys.time_delays.is_empty() {
                for td in &sys.time_delays {
                    let (row, constant) = fermat_form(td.from, td.to);
                    eq.push(row);
                    b_eq.push(td.delay / sys.delay_scale - constant);
                }
            } else if let Some(order) = &sys.order {
                // Consecutive arrival order: τ_earlier − τ_later ≤ 0.
                for w in order.windows(2) {
                    let (row, constant) = fermat_form(w[1], w[0]);
                    ub.push(row);
                    b_ub.push(-constant);
                }
            }
        }

        // κ ≥ 0 per cell: −m_c ≤ 0.
        for c in 0..n_cells {
            let mut row = vec![0.0; n];
            row[c] = -1.0;
            ub.push(row);
            b_ub.push(0.0);
        }

        // Adjacent-cell gradient bound: |m_i/A_i − m_j/A_j| ≤ g · d_ij.
        if let Some(g) = config.grad_bound {
            for i in 0..n_cells {
                let ci = basis.center(i);
                for j in basis.cell_neighbors(i) {
                    if j <= i {
                        continue;
                    }
                    let d = (basis.center(j) - ci).norm();
                    let mut row = vec![0.0; n];
                    row[i] = 1.0 / areas[i];
                    row[j] = -1.0 / areas[j];
                    ub.push(row.clone());
                    b_ub.push(g * d);
                    for v in row.iter_mut() {
                        *v = -*v;
                    }
                    ub.push(row);
                    b_ub.push(g * d);
                }
            }
        }

        ConstraintSet {
            n_cells,
            n_sources,
            a_ub: stack(ub, n),
            b_ub,
            a_eq: stack(eq, n),
            b_eq,
        }
    }

    /// Number of unknowns `n = n_cells + 2 · n_sources`.
    pub fn n_unknowns(&self) -> usize {
        self.n_cells + 2 * self.n_sources
    }

    /// Whether the polytope is non-empty (the constraints admit a solution).
    pub fn feasible(&self) -> bool {
        let c = vec![0.0; self.n_unknowns()];
        let nonneg = vec![false; self.n_unknowns()];
        !matches!(
            lp::lpsolve(&c, &self.a_ub, &self.b_ub, &self.a_eq, &self.b_eq, &nonneg),
            LpStatus::Infeasible
        )
    }
}

/// Stack accumulated rows (each length `n`) into an `m × n` matrix.
fn stack(rows: Vec<Vec<f64>>, n: usize) -> Array2<f64> {
    let m = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((m, n), flat).expect("row lengths equal n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, PixelGrid, Point2D, ScalarField2D};
    use crate::lens::SIS;
    use crate::solver::extrema::delays::TimeDelay;
    use crate::solver::image::{Image, PointImageSystem};

    /// The two collinear images of a point source under an SIS, as a system.
    fn sis_double(beta: f64) -> (SIS, PointImageSystem) {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let src = Point2D::new(beta, 0.0);
        // SIS images of an axial source: θ_± = β ± θ_E on the x-axis.
        let images = vec![
            Image::at(&lens, Point2D::new(beta + 1.0, 0.0), src),
            Image::at(&lens, Point2D::new(beta - 1.0, 0.0), src),
        ];
        (lens, PointImageSystem::from_images(images))
    }

    fn uniform_basis() -> PixelGrid {
        PixelGrid::constant(Grid2D::square(6, 2.5).unwrap(), 0.3)
    }

    #[test]
    fn assembles_expected_row_and_column_counts() {
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let cs = ConstraintSet::assemble(&basis, &[sys], &ConstraintConfig::default());
        assert_eq!(cs.n_cells, basis.len());
        assert_eq!(cs.n_sources, 1);
        assert_eq!(cs.n_unknowns(), basis.len() + 2);
        // Two images → four position equalities; κ≥0 → one inequality per cell.
        assert_eq!(cs.a_eq.nrows(), 4);
        assert_eq!(cs.b_ub.len(), basis.len());
        assert_eq!(cs.a_ub.ncols(), cs.n_unknowns());
    }

    #[test]
    fn the_generating_lens_lies_in_the_polytope() {
        // The κ field that produced the images must satisfy every position equality, so the
        // polytope is feasible.
        let (_lens, sys) = sis_double(0.15);
        let basis = uniform_basis();
        let cs = ConstraintSet::assemble(&basis, &[sys], &ConstraintConfig::default());
        assert!(cs.feasible());
    }

    #[test]
    fn a_time_delay_adds_one_equality() {
        let (_lens, sys) = sis_double(0.2);
        let sys = sys.with_time_delays(vec![TimeDelay { from: 1, to: 0, delay: 0.05 }], 1.0);
        let basis = uniform_basis();
        let cs = ConstraintSet::assemble(&basis, &[sys], &ConstraintConfig::default());
        // Four position rows plus one time-delay row.
        assert_eq!(cs.a_eq.nrows(), 5);
    }

    #[test]
    fn gradient_bound_adds_inequalities() {
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let plain = ConstraintSet::assemble(&basis, &[sys.clone()], &ConstraintConfig::default());
        let bounded =
            ConstraintSet::assemble(&basis, &[sys], &ConstraintConfig { grad_bound: Some(0.5) });
        assert!(bounded.a_ub.nrows() > plain.a_ub.nrows());
    }
}
