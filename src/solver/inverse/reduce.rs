// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Reducing the ensemble by extended-image observation fits.
//!
//! [`reweight`](super::reweight) turns each member's forward χ² into an importance weight;
//! [`reduce_ensemble`] collapses that [`WeightedEnsemble`] into the observation-constrained
//! posterior sub-ensemble — either by systematic SIR resampling to a fixed size or by a
//! weight-threshold cull.

use crate::rng::Lcg;
use crate::solver::ensemble::{Ensemble, EnsembleMember};

use super::reweight::WeightedEnsemble;

/// How to reduce a [`WeightedEnsemble`] to a posterior sub-ensemble.
#[derive(Debug, Clone, Copy)]
pub enum ReduceMethod {
    /// Systematic sampling-importance-resampling to `target` equally-weighted members.
    Resample {
        /// Number of members to draw.
        target: usize,
    },
    /// Keep every member whose weight is at least this threshold.
    Threshold(f64),
}

/// Configuration for [`reduce_ensemble`].
#[derive(Debug, Clone, Copy)]
pub struct ReduceConfig {
    /// The reduction rule.
    pub method: ReduceMethod,
    /// Base RNG seed for the resampling draw (ignored by [`ReduceMethod::Threshold`]).
    pub seed: u64,
}

/// Reduce `weighted` to the observation-constrained posterior sub-ensemble.
///
/// # Arguments
/// * `weighted` - the re-weighted ensemble (weights need not be normalized).
/// * `cfg` - the reduction rule and seed.
///
/// Returns an [`Ensemble`] of the retained (or resampled) members; empty when every weight is
/// zero (a fully-culled input).
pub fn reduce_ensemble(weighted: &WeightedEnsemble, cfg: &ReduceConfig) -> Ensemble {
    let members = match cfg.method {
        ReduceMethod::Resample { target } => resample(weighted, target, cfg.seed),
        ReduceMethod::Threshold(t) => weighted
            .ensemble
            .members
            .iter()
            .zip(&weighted.weights)
            .filter(|(_, w)| **w >= t)
            .map(|(m, _)| m.clone())
            .collect(),
    };
    Ensemble { n_cells: weighted.ensemble.n_cells, n_sources: weighted.ensemble.n_sources, members }
}

/// Systematic SIR: one low-variance sweep of the weight CDF drawing `target` members.
fn resample(weighted: &WeightedEnsemble, target: usize, seed: u64) -> Vec<EnsembleMember> {
    let weights = &weighted.weights;
    let total: f64 = weights.iter().sum();
    if target == 0 || total <= 0.0 || weights.is_empty() {
        return Vec::new();
    }
    let step = total / target as f64;
    let mut rng = Lcg::new(seed);
    let u0 = rng.f() * step;
    let mut out = Vec::with_capacity(target);
    let (mut j, mut cum) = (0usize, weights[0]);
    for k in 0..target {
        let pos = u0 + k as f64 * step;
        while pos > cum && j + 1 < weights.len() {
            j += 1;
            cum += weights[j];
        }
        out.push(weighted.ensemble.members[j].clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ensemble::summaries::effective_sample_size;

    /// A weighted ensemble whose member `i` carries the marker value `i` in its single cell.
    fn weighted(weights: Vec<f64>) -> WeightedEnsemble {
        let members = (0..weights.len()).map(|i| EnsembleMember { x: vec![i as f64] }).collect();
        let ess = effective_sample_size(&weights);
        WeightedEnsemble { ensemble: Ensemble { n_cells: 1, n_sources: 0, members }, weights, ess }
    }

    #[test]
    fn resample_draws_exactly_target_members() {
        let we = weighted(vec![0.1, 0.7, 0.2]);
        let cfg = ReduceConfig { method: ReduceMethod::Resample { target: 10 }, seed: 42 };
        let out = reduce_ensemble(&we, &cfg);
        assert_eq!(out.len(), 10);
        assert_eq!(out.n_cells, 1);
    }

    #[test]
    fn resample_favours_high_weight_members() {
        // Member 1 dominates the weight, so it should fill most resampled slots.
        let we = weighted(vec![0.02, 0.96, 0.02]);
        let cfg = ReduceConfig { method: ReduceMethod::Resample { target: 100 }, seed: 7 };
        let out = reduce_ensemble(&we, &cfg);
        let from_dominant = out.members.iter().filter(|m| m.x[0] == 1.0).count();
        assert!(from_dominant > 80, "dominant filled {from_dominant}/100");
    }

    #[test]
    fn threshold_keeps_only_heavy_members() {
        let we = weighted(vec![0.05, 0.6, 0.35]);
        let cfg = ReduceConfig { method: ReduceMethod::Threshold(0.3), seed: 0 };
        let out = reduce_ensemble(&we, &cfg);
        assert_eq!(out.len(), 2); // 0.6 and 0.35 survive
        assert!(out.members.iter().all(|m| m.x[0] != 0.0));
    }

    #[test]
    fn a_fully_culled_ensemble_reduces_to_empty() {
        let we = weighted(vec![0.0, 0.0]);
        let cfg = ReduceConfig { method: ReduceMethod::Resample { target: 5 }, seed: 1 };
        assert!(reduce_ensemble(&we, &cfg).is_empty());
    }
}
