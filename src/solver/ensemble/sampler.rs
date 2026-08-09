// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The lensing overlay on hit-and-run polytope sampling (flat-in-mass).
//!
//! The generic Markov-chain engine lives in [`numerical::mcmc`](crate::numerical::mcmc);
//! this is the thin domain layer: it adapts a
//! [`ConstraintSet`](crate::solver::constraints::ConstraintSet) into the polytope the engine
//! samples and wraps the raw draws as an [`Ensemble`] of [`EnsembleMember`]s (each the full
//! `x = (m, {β_s})` — cell masses plus recovered source positions). Sampling in mass
//! coordinates makes the uniform draw a flat-in-mass prior.

use crate::grid::Point2D;
use crate::numerical::mcmc::{self, Polytope, PolytopeSample};
use crate::solver::constraints::ConstraintSet;
use crate::{LensForgeError, Result, SolveInfo};

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
        let poly = Polytope {
            a_ub: &cs.a_ub,
            b_ub: &cs.b_ub,
            a_eq: &cs.a_eq,
            b_eq: &cs.b_eq,
            n: cs.n_unknowns(),
        };
        match mcmc::hit_and_run(&poly, self.walkers, self.burn_in, self.per_walker, self.seed) {
            PolytopeSample::InconsistentEqualities => Err(LensForgeError::computation_error(
                "inconsistent equality constraints (no solution)",
            )),
            PolytopeSample::EmptyInterior => Err(LensForgeError::computation_error(
                "empty or lower-dimensional constraint polytope",
            )),
            PolytopeSample::Ok { points, residual } => {
                let members: Vec<EnsembleMember> =
                    points.into_iter().map(|x| EnsembleMember { x }).collect();
                let iterations = members.len();
                Ok((
                    Ensemble { n_cells: cs.n_cells, n_sources: cs.n_sources, members },
                    SolveInfo { iterations, residual, converged: true },
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, PixelGrid, Point2D, ScalarField2D};
    use crate::lens::SIS;
    use crate::solver::constraints::{ConstraintConfig, ConstraintSet};
    use crate::solver::image::{Image, PointImageSystem};
    use ndarray::{array, Array1, Array2};

    #[test]
    fn member_accessors_split_masses_and_source_positions() {
        let m = EnsembleMember { x: vec![1.0, 2.0, 0.3, -0.4, 0.7, 0.8] };
        assert_eq!(m.masses(2), &[1.0, 2.0]);
        assert_eq!(m.source(2, 0), Point2D::new(0.3, -0.4));
        assert_eq!(m.source(2, 1), Point2D::new(0.7, 0.8));
    }

    #[test]
    fn inconsistent_equalities_map_to_an_error() {
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
    fn an_empty_interior_maps_to_an_error() {
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
        let (ens, info) = sampler.sample(&cs).unwrap();
        assert!(!ens.is_empty() && info.converged);
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
