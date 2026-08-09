// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Radial profiles of a lens's mass distribution.
//!
//! A [`RadialProfile`] bins a per-cell quantity (mass, convergence) into concentric annuli
//! about a centre, indexed by each annulus's outer radius — a property of a single mass map.
//! The ensemble statistics reuse it to expose the monopole degeneracy (per-annulus scatter).

use crate::grid::Point2D;

/// A quantity binned into concentric annuli about a centre, indexed by each annulus's outer
/// radius.
#[derive(Debug, Clone)]
pub struct RadialProfile {
    /// Outer radius of each annulus.
    pub radii: Vec<f64>,
    /// The binned quantity in each annulus.
    pub values: Vec<f64>,
}

impl RadialProfile {
    /// Bin per-cell `values` at `centers` into `nbins` annuli spanning `[0, max radius]`
    /// about `origin`, summing the values that fall in each annulus.
    ///
    /// # Arguments
    /// * `centers` - cell centres.
    /// * `values` - the per-cell quantity (length matches `centers`).
    /// * `origin` - the centre the annuli are measured from.
    /// * `nbins` - number of annuli (at least one).
    pub fn binned(centers: &[Point2D], values: &[f64], origin: Point2D, nbins: usize) -> Self {
        let r: Vec<f64> = centers.iter().map(|c| (*c - origin).norm()).collect();
        let r_max = r.iter().cloned().fold(0.0_f64, f64::max).max(1e-12);
        let mut values_binned = vec![0.0; nbins];
        for (c, v) in values.iter().enumerate() {
            let b = ((r[c] / r_max * nbins as f64) as usize).min(nbins - 1);
            values_binned[b] += *v;
        }
        let radii = (0..nbins).map(|b| (b + 1) as f64 / nbins as f64 * r_max).collect();
        Self { radii, values: values_binned }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binned_sums_values_into_annuli() {
        // Two cells at r = 0.5 and r = 2.0 land in the inner and outer of two annuli.
        let centers = [Point2D::new(0.5, 0.0), Point2D::new(2.0, 0.0)];
        let prof = RadialProfile::binned(&centers, &[3.0, 5.0], Point2D::origin(), 2);
        assert_eq!(prof.values, vec![3.0, 5.0]);
        assert_eq!(prof.radii.len(), 2);
        assert!((prof.radii[1] - 2.0).abs() < 1e-12); // outer edge = max radius
    }

    #[test]
    fn coincident_cells_accumulate_in_one_bin() {
        let centers = [Point2D::new(1.0, 0.0), Point2D::new(1.0, 0.0)];
        let prof = RadialProfile::binned(&centers, &[2.0, 4.0], Point2D::origin(), 1);
        assert_eq!(prof.values, vec![6.0]);
    }
}
