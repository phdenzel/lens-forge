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

use crate::grid::Point2D;
use crate::numerical::lp::{self, LpStatus};
use crate::numerical::matrix::stack_rows;
use crate::solver::image::PointImageSystem;

use crate::solver::basis::{FieldCoefficients, KappaBasis};

use super::data::{arrival_order_inequalities, position_equalities, time_delay_equalities};
use super::priors::{
    curvature_bounds, gradient_bounds, mean_kappa_equality, nonneg_kappa_bounds,
    radial_monotonic_bounds, symmetry_equalities,
};

/// A symmetry the convergence field is required to obey, as equalities `κ(θ) = κ(Sθ)`.
///
/// The symmetry centre is the coordinate origin (the usual lens centre). Cells are paired
/// by matching each cell centre's image `Sθ` to the nearest cell centre; only mutually
/// nearest pairs are constrained, so a basis not symmetric under `S` simply yields fewer
/// (or no) pairings rather than spurious ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    /// 180° rotation about the origin: `κ(θ) = κ(−θ)`.
    Inversion,
    /// Reflection across the x-axis: `κ(x, y) = κ(x, −y)`.
    MirrorX,
    /// Reflection across the y-axis: `κ(x, y) = κ(−x, y)`.
    MirrorY,
}

impl Symmetry {
    /// The image `Sθ` of `p` under this symmetry (about the origin).
    ///
    /// # Arguments
    /// * `p` - the point to map.
    pub(super) fn map(self, p: Point2D) -> Point2D {
        match self {
            Symmetry::Inversion => Point2D::new(-p.x, -p.y),
            Symmetry::MirrorX => Point2D::new(p.x, -p.y),
            Symmetry::MirrorY => Point2D::new(-p.x, p.y),
        }
    }
}

/// Options composing which constraint families the builder emits.
///
/// The [`Default`] emits the full physical polytope: the lens-equation position equalities,
/// non-negative convergence, and any time-delay / arrival-order data a system carries. The
/// adjacent-cell gradient bound is off by default.
#[derive(Debug, Clone, Copy)]
pub struct ConstraintConfig {
    /// Emit the lens-equation position equalities `α(θ) + β = θ` (default `true`). Disabling
    /// them leaves the polytope with no lens constraint — a warning is logged when so.
    pub position_equalities: bool,
    /// Emit the non-negative convergence bounds `κ ≥ 0` per cell (default `true`).
    pub nonneg_kappa: bool,
    /// Honour measured time delays a system carries (default `true`).
    pub use_time_delays: bool,
    /// Honour a known arrival order a system carries (default `true`).
    pub use_arrival_order: bool,
    /// Optional convergence-gradient bound `g`: `|κ_i − κ_j| ≤ g · d_ij` on adjacent cells.
    pub grad_bound: Option<f64>,
    /// Optional curvature bound `l`: `|Σ_{j∈N(i)}(κ_i − κ_j)| ≤ l` per cell (the discrete
    /// Laplacian), capping substructure amplitude — small `l` smooth, large `l` clumpy.
    pub curvature_bound: Option<f64>,
    /// Optional symmetry the convergence field must obey (`κ(θ) = κ(Sθ)` per paired cells).
    pub symmetry: Option<Symmetry>,
    /// Optional mass-sheet pin: fix the mean convergence `(1/C)·Σ_c κ_c` to this value,
    /// removing the mass-sheet-degeneracy (MSD) direction from the polytope.
    pub mean_kappa: Option<f64>,
    /// Optional radial-monotonicity prior about this centre: convergence may not increase
    /// outward, `κ_outer ≤ κ_inner` on each radial adjacency edge (a centrally-concentrated
    /// profile). Near-equidistant (azimuthal) edges are left unconstrained.
    pub radial_monotonic: Option<Point2D>,
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            position_equalities: true,
            nonneg_kappa: true,
            use_time_delays: true,
            use_arrival_order: true,
            grad_bound: None,
            curvature_bound: None,
            symmetry: None,
            mean_kappa: None,
            radial_monotonic: None,
        }
    }
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
    /// Assemble the polytope for `basis` constrained by `systems`, composing the families
    /// `config` enables.
    ///
    /// Per source (when enabled): two position equalities per image (`α(θ) + β_s = θ`); a
    /// Fermat time-delay equality per measured delay, or arrival-order inequalities when only
    /// the order is known. Globally (when enabled): `κ ≥ 0` per cell, and optional
    /// adjacent-cell gradient bounds. β-coordinates are appended after the cells in source
    /// order.
    ///
    /// # Arguments
    /// * `basis` - the convergence basis (pixel or mesh).
    /// * `systems` - the point-image systems sharing this basis.
    /// * `config` - which constraint families to emit.
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

        if !config.position_equalities {
            eprintln!(
                "Warning: position equalities are disabled; the constraint polytope carries no lens-equation constraint."
            );
        }

        for (s, sys) in systems.iter().enumerate() {
            let thetas: Vec<Point2D> = sys.images.iter().map(|im| im.theta).collect();
            let infl = FieldCoefficients::build(basis, &thetas);
            let (bx, by) = (n_cells + 2 * s, n_cells + 2 * s + 1);

            if config.position_equalities {
                let (rows, rhs) = position_equalities(&infl, &thetas, &areas, n, bx, by);
                eq.extend(rows);
                b_eq.extend(rhs);
            }

            if config.use_time_delays && !sys.time_delays.is_empty() {
                let (rows, rhs) = time_delay_equalities(sys, &infl, &thetas, &areas, n, bx, by);
                eq.extend(rows);
                b_eq.extend(rhs);
            } else if let (true, Some(order)) = (config.use_arrival_order, &sys.order) {
                let (rows, rhs) = arrival_order_inequalities(order, &infl, &thetas, &areas, n, bx, by);
                ub.extend(rows);
                b_ub.extend(rhs);
            }
        }

        if config.nonneg_kappa {
            let (rows, rhs) = nonneg_kappa_bounds(n_cells, n);
            ub.extend(rows);
            b_ub.extend(rhs);
        }

        if let Some(g) = config.grad_bound {
            let (rows, rhs) = gradient_bounds(basis, &areas, g, n);
            ub.extend(rows);
            b_ub.extend(rhs);
        }

        if let Some(l) = config.curvature_bound {
            let (rows, rhs) = curvature_bounds(basis, &areas, l, n);
            ub.extend(rows);
            b_ub.extend(rhs);
        }

        if let Some(center) = config.radial_monotonic {
            let (rows, rhs) = radial_monotonic_bounds(basis, center, &areas, n);
            ub.extend(rows);
            b_ub.extend(rhs);
        }

        if let Some(sym) = config.symmetry {
            let (rows, rhs) = symmetry_equalities(basis, sym, &areas, n);
            eq.extend(rows);
            b_eq.extend(rhs);
        }

        if let Some(mean) = config.mean_kappa {
            let (row, rhs) = mean_kappa_equality(mean, &areas, n);
            eq.push(row);
            b_eq.push(rhs);
        }

        ConstraintSet {
            n_cells,
            n_sources,
            a_ub: stack_rows(ub, n),
            b_ub,
            a_eq: stack_rows(eq, n),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, PixelGrid, Point2D, ScalarField2D};
    use crate::lens::SIS;
    use crate::solver::forward::extrema::delays::TimeDelay;
    use crate::solver::image::{Image, PointImageSystem};
    use ndarray::Array1;

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
        let bounded = ConstraintSet::assemble(
            &basis,
            &[sys],
            &ConstraintConfig { grad_bound: Some(0.5), ..Default::default() },
        );
        assert!(bounded.a_ub.nrows() > plain.a_ub.nrows());
    }

    #[test]
    fn disabling_nonneg_kappa_drops_its_inequalities() {
        // Default emits one κ≥0 inequality per cell; disabling it (with no gradient bound)
        // leaves the inequality block empty.
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let cfg = ConstraintConfig { nonneg_kappa: false, ..Default::default() };
        let cs = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert_eq!(cs.a_ub.nrows(), 0);
    }

    #[test]
    fn disabling_position_equalities_drops_the_equalities() {
        // With no delays/order and position equalities off, the equality block is empty.
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let cfg = ConstraintConfig { position_equalities: false, ..Default::default() };
        let cs = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert_eq!(cs.a_eq.nrows(), 0);
    }

    #[test]
    fn disabling_time_delays_ignores_measured_delays() {
        // A delay-carrying system: with delays honoured there are 5 equality rows (4 position
        // + 1 delay); gating them off leaves only the 4 position rows.
        let (_lens, sys) = sis_double(0.2);
        let sys = sys.with_time_delays(vec![TimeDelay { from: 1, to: 0, delay: 0.05 }], 1.0);
        let basis = uniform_basis();
        let cfg = ConstraintConfig { use_time_delays: false, ..Default::default() };
        let cs = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert_eq!(cs.a_eq.nrows(), 4);
    }

    #[test]
    fn disabling_arrival_order_ignores_the_order() {
        // An order-carrying system adds one inequality per consecutive pair by default;
        // gating the order off leaves only the κ≥0 inequalities.
        let (_lens, sys) = sis_double(0.2);
        let sys = sys.with_order(vec![0, 1]);
        let basis = uniform_basis();
        let honoured = ConstraintSet::assemble(&basis, &[sys.clone()], &ConstraintConfig::default());
        let cfg = ConstraintConfig { use_arrival_order: false, ..Default::default() };
        let ignored = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert_eq!(honoured.a_ub.nrows(), ignored.a_ub.nrows() + 1);
    }

    #[test]
    fn symmetry_adds_paired_equalities() {
        // A 6×6 origin-centred basis pairs every cell antipodally (no on-axis self-pairs)
        // under each symmetry → n_cells/2 equalities added.
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let base = ConstraintSet::assemble(&basis, &[sys.clone()], &ConstraintConfig::default());
        for sym in [Symmetry::Inversion, Symmetry::MirrorX, Symmetry::MirrorY] {
            let cfg = ConstraintConfig { symmetry: Some(sym), ..Default::default() };
            let cs = ConstraintSet::assemble(&basis, &[sys.clone()], &cfg);
            assert_eq!(cs.a_eq.nrows(), base.a_eq.nrows() + basis.len() / 2, "sym {sym:?}");
        }
    }

    #[test]
    fn inversion_symmetry_keeps_the_symmetric_lens_feasible() {
        // The SIS that generated the images is inversion-symmetric, so imposing the symmetry
        // equalities leaves the polytope non-empty.
        let (_lens, sys) = sis_double(0.15);
        let basis = uniform_basis();
        let cfg = ConstraintConfig { symmetry: Some(Symmetry::Inversion), ..Default::default() };
        assert!(ConstraintSet::assemble(&basis, &[sys], &cfg).feasible());
    }

    #[test]
    fn mean_kappa_adds_one_equality() {
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let base = ConstraintSet::assemble(&basis, &[sys.clone()], &ConstraintConfig::default());
        let cfg = ConstraintConfig { mean_kappa: Some(0.3), ..Default::default() };
        let cs = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert_eq!(cs.a_eq.nrows(), base.a_eq.nrows() + 1);
    }

    #[test]
    fn mean_kappa_pin_respects_non_negativity() {
        // With κ≥0 the mean pin is feasible for a non-negative target (a uniform field) and
        // infeasible for a negative one — the pin genuinely bites. (Position equalities off to
        // isolate the pin.)
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let feasible = ConstraintConfig {
            position_equalities: false,
            mean_kappa: Some(0.5),
            ..Default::default()
        };
        assert!(ConstraintSet::assemble(&basis, &[sys.clone()], &feasible).feasible());
        let infeasible = ConstraintConfig {
            position_equalities: false,
            mean_kappa: Some(-1.0),
            ..Default::default()
        };
        assert!(!ConstraintSet::assemble(&basis, &[sys], &infeasible).feasible());
    }

    #[test]
    fn radial_monotonicity_adds_inequalities() {
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let base = ConstraintSet::assemble(&basis, &[sys.clone()], &ConstraintConfig::default());
        let cfg =
            ConstraintConfig { radial_monotonic: Some(Point2D::origin()), ..Default::default() };
        let cs = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert!(cs.a_ub.nrows() > base.a_ub.nrows());
    }

    #[test]
    fn radial_monotonicity_orders_convergence_outward() {
        // With only the radial bounds (positions and κ≥0 off): a field rising outward
        // violates a bound, one falling outward satisfies them all.
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let cfg = ConstraintConfig {
            position_equalities: false,
            nonneg_kappa: false,
            radial_monotonic: Some(Point2D::origin()),
            ..Default::default()
        };
        let cs = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert!(cs.a_ub.nrows() > 0);

        let n = cs.n_unknowns();
        let radius = |c: usize| ScalarField2D::center(&basis, c).norm();
        let field = |ascending: bool| -> Array1<f64> {
            Array1::from((0..n).map(|k| {
                if k < basis.len() {
                    let kappa = if ascending { radius(k) } else { 5.0 - radius(k) };
                    kappa * basis.cell_area(k)
                } else {
                    0.0 // the source β columns
                }
            }).collect::<Vec<_>>())
        };
        let rising = cs.a_ub.dot(&field(true));
        let falling = cs.a_ub.dot(&field(false));
        assert!(rising.iter().any(|&v| v > 1e-9), "rising field must violate a bound");
        assert!(falling.iter().all(|&v| v <= 1e-9), "falling field must satisfy all bounds");
    }

    #[test]
    fn curvature_bound_adds_inequalities() {
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let base = ConstraintSet::assemble(&basis, &[sys.clone()], &ConstraintConfig::default());
        let cfg = ConstraintConfig { curvature_bound: Some(0.5), ..Default::default() };
        let cs = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert!(cs.a_ub.nrows() > base.a_ub.nrows());
    }

    #[test]
    fn curvature_bound_caps_peakiness() {
        // With only the curvature bounds (positions and κ≥0 off): a flat field has zero
        // curvature and passes; a single-cell spike has large curvature and violates it.
        let (_lens, sys) = sis_double(0.2);
        let basis = uniform_basis();
        let cfg = ConstraintConfig {
            position_equalities: false,
            nonneg_kappa: false,
            curvature_bound: Some(0.5),
            ..Default::default()
        };
        let cs = ConstraintSet::assemble(&basis, &[sys], &cfg);
        assert!(cs.a_ub.nrows() > 0);

        let n = cs.n_unknowns();
        let mass = |kappa: &dyn Fn(usize) -> f64| -> Array1<f64> {
            Array1::from(
                (0..n)
                    .map(|k| if k < basis.len() { kappa(k) * basis.cell_area(k) } else { 0.0 })
                    .collect::<Vec<_>>(),
            )
        };
        let flat = mass(&|_| 1.0);
        assert!(cs.a_ub.dot(&flat).iter().all(|&v| v <= 0.5 + 1e-9));

        let center = basis.cell_of(&Point2D::new(0.4, 0.4)).unwrap();
        let spike = mass(&|c| if c == center { 1.0 } else { 0.0 });
        assert!(cs.a_ub.dot(&spike).iter().any(|&v| v > 0.5), "a spike must break the bound");
    }
}
