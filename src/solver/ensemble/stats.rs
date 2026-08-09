// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Reducing the ensemble to physical summaries.
//!
//! The ensemble is a cloud of mass distributions; these functions project it onto the
//! quantities that matter: the mean/variance convergence maps, the mass-sheet level (the
//! MSD direction), the radial mass profile whose per-annulus scatter exposes the monopole
//! degeneracy, and the effective sample size for the re-weighting stage.

use crate::grid::Point2D;
use crate::lens::RadialProfile;

use super::sampler::Ensemble;

/// Per-cell mean convergence `⟨κ_c⟩` over the ensemble (`κ_c = m_c / A_c`).
///
/// # Arguments
/// * `ensemble` - the sampled ensemble.
/// * `areas` - cell areas (length `n_cells`).
pub fn kappa_mean(ensemble: &Ensemble, areas: &[f64]) -> Vec<f64> {
    let n = ensemble.len().max(1) as f64;
    let mut mean = vec![0.0; ensemble.n_cells];
    for m in &ensemble.members {
        for (c, mc) in m.masses(ensemble.n_cells).iter().enumerate() {
            mean[c] += mc / areas[c];
        }
    }
    for v in &mut mean {
        *v /= n;
    }
    mean
}

/// Per-cell convergence variance `Var(κ_c)` over the ensemble.
///
/// # Arguments
/// * `ensemble` - the sampled ensemble.
/// * `areas` - cell areas (length `n_cells`).
pub fn kappa_variance(ensemble: &Ensemble, areas: &[f64]) -> Vec<f64> {
    let mean = kappa_mean(ensemble, areas);
    let n = ensemble.len().max(1) as f64;
    let mut var = vec![0.0; ensemble.n_cells];
    for m in &ensemble.members {
        for (c, mc) in m.masses(ensemble.n_cells).iter().enumerate() {
            let d = mc / areas[c] - mean[c];
            var[c] += d * d;
        }
    }
    for v in &mut var {
        *v /= n;
    }
    var
}

/// The mass-sheet level of each member: its mean convergence `(1/C) Σ_c κ_c`. The spread
/// of these values measures the ensemble's freedom along the mass-sheet (MSD) direction.
///
/// # Arguments
/// * `ensemble` - the sampled ensemble.
/// * `areas` - cell areas (length `n_cells`).
pub fn msd_levels(ensemble: &Ensemble, areas: &[f64]) -> Vec<f64> {
    let c = ensemble.n_cells.max(1) as f64;
    ensemble
        .members
        .iter()
        .map(|m| m.masses(ensemble.n_cells).iter().enumerate().map(|(i, mc)| mc / areas[i]).sum::<f64>() / c)
        .collect()
}

/// The ensemble's mean radial mass profile (a [`RadialProfile`]) together with the
/// per-annulus mass variance across members — large variance in image-free annuli is the
/// monopole degeneracy.
///
/// # Arguments
/// * `ensemble` - the sampled ensemble.
/// * `centers` - cell centres (length `n_cells`).
/// * `origin` - the centre the annuli are measured from.
/// * `nbins` - number of annuli spanning `[0, max radius]`.
pub fn radial_mass_profile(
    ensemble: &Ensemble,
    centers: &[Point2D],
    origin: Point2D,
    nbins: usize,
) -> (RadialProfile, Vec<f64>) {
    let n = ensemble.len().max(1) as f64;
    // The binning geometry is member-independent; take the radii once (also handles an
    // empty ensemble, where the loop below never runs).
    let radii = RadialProfile::binned(centers, &vec![0.0; centers.len()], origin, nbins).radii;
    let (mut mean, mut sq) = (vec![0.0; nbins], vec![0.0; nbins]);
    for m in &ensemble.members {
        let binned = RadialProfile::binned(centers, m.masses(ensemble.n_cells), origin, nbins);
        for b in 0..nbins {
            mean[b] += binned.values[b];
            sq[b] += binned.values[b] * binned.values[b];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let variance = (0..nbins).map(|b| (sq[b] / n - mean[b] * mean[b]).max(0.0)).collect();
    (RadialProfile { radii, values: mean }, variance)
}

/// The effective sample size `(Σ w)² / Σ w²` of importance weights (equal weights give the
/// full count; concentration collapses it). Zero for empty or all-zero weights.
///
/// # Arguments
/// * `weights` - per-member importance weights.
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let s: f64 = weights.iter().sum();
    let s2: f64 = weights.iter().map(|w| w * w).sum();
    if s2 <= 0.0 { 0.0 } else { s * s / s2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sampler::{Ensemble, EnsembleMember};

    /// A tiny ensemble of two 2-cell members (no sources), for hand-checkable stats.
    fn toy() -> Ensemble {
        Ensemble {
            n_cells: 2,
            n_sources: 0,
            members: vec![
                EnsembleMember { x: vec![1.0, 3.0] },
                EnsembleMember { x: vec![3.0, 5.0] },
            ],
        }
    }

    #[test]
    fn mean_and_variance_match_hand_computation() {
        let ens = toy();
        let areas = [1.0, 1.0];
        assert_eq!(kappa_mean(&ens, &areas), vec![2.0, 4.0]); // (1+3)/2, (3+5)/2
        assert_eq!(kappa_variance(&ens, &areas), vec![1.0, 1.0]); // ((±1)²+(±1)²)/2
    }

    #[test]
    fn areas_scale_convergence() {
        // Doubling a cell's area halves its convergence.
        let ens = toy();
        let areas = [2.0, 1.0];
        assert_eq!(kappa_mean(&ens, &areas), vec![1.0, 4.0]);
    }

    #[test]
    fn msd_level_is_the_mean_convergence_per_member() {
        let ens = toy();
        let areas = [1.0, 1.0];
        assert_eq!(msd_levels(&ens, &areas), vec![2.0, 4.0]); // (1+3)/2, (3+5)/2
    }

    #[test]
    fn radial_profile_bins_and_reduces() {
        let ens = toy();
        let centers = [Point2D::new(0.5, 0.0), Point2D::new(2.0, 0.0)];
        let (mean, variance) = radial_mass_profile(&ens, &centers, Point2D::origin(), 2);
        assert_eq!(mean.values.len(), 2);
        // Inner annulus holds cell 0 (mass 1,3 → mean 2), outer holds cell 1 (3,5 → mean 4).
        assert!((mean.values[0] - 2.0).abs() < 1e-12);
        assert!((mean.values[1] - 4.0).abs() < 1e-12);
        assert!((variance[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn effective_sample_size_spans_the_extremes() {
        assert!((effective_sample_size(&[1.0, 1.0, 1.0, 1.0]) - 4.0).abs() < 1e-12);
        assert!((effective_sample_size(&[1.0, 0.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
        assert_eq!(effective_sample_size(&[0.0, 0.0]), 0.0);
    }
}
