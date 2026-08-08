// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Time delays between the images of a point source.
//!
//! The relative arrival time of an image is set by its Fermat potential τ: the
//! earliest (lowest τ) arrives first, and the dimensionless differences Δτ scale
//! to physical delays through the cosmological time-delay distance D_Δt,
//! Δt = (D_Δt / c) Δτ. Measured delays feed the inverse H₀ inference
//! ([`h0_from_time_delay_distance`](crate::cosmology::Cosmology::h0_from_time_delay_distance)).

use crate::cosmology::model::delay_from_time_delay_distance;
use crate::cosmology::Cosmology;
use crate::solver::image::Image;

/// Dimensionless arrival-time differences Δτ_i = τ_i − τ_earliest of `images`, in
/// input order (the earliest image is 0).
///
/// # Arguments
/// * `images` - the images of the point source (from an
///   [`ImageFinder`](super::images::ImageFinder)).
pub fn arrival_delays(images: &[Image]) -> Vec<f64> {
    let tau_min = images.iter().map(|im| im.tau).fold(f64::INFINITY, f64::min);
    images.iter().map(|im| im.tau - tau_min).collect()
}

/// Physical time delays (seconds) of `images` relative to the earliest, scaling
/// their Fermat differences by the time-delay distance D_Δt(z_lens, z_source).
///
/// # Arguments
/// * `images` - the images of the point source.
/// * `cosmo` - the background cosmology supplying D_Δt.
/// * `z_lens` - lens redshift.
/// * `z_source` - source redshift.
pub fn time_delays(
    images: &[Image],
    cosmo: &dyn Cosmology,
    z_lens: f64,
    z_source: f64,
) -> Vec<f64> {
    let d_dt = cosmo.time_delay_distance(z_lens, z_source);
    arrival_delays(images)
        .into_iter()
        .map(|dtau| delay_from_time_delay_distance(d_dt, dtau))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::model::time_delay_distance_from_delay;
    use crate::cosmology::FlatLCDM;
    use crate::grid::Point2D;
    use crate::solver::image::Parity;

    /// An image carrying only the Fermat potential `tau` (the rest is irrelevant here).
    fn image_with_tau(tau: f64) -> Image {
        Image { theta: Point2D::origin(), mu: 1.0, parity: Parity::MINIMUM, tau }
    }

    #[test]
    fn arrival_delays_are_relative_to_the_earliest() {
        let images = [image_with_tau(0.5), image_with_tau(-0.3), image_with_tau(0.1)];
        let delays = arrival_delays(&images);
        assert_eq!(delays, vec![0.8, 0.0, 0.4]);
    }

    #[test]
    fn empty_images_have_no_delays() {
        assert!(arrival_delays(&[]).is_empty());
        assert!(time_delays(&[], &FlatLCDM::planck18(), 0.5, 2.0).is_empty());
    }

    #[test]
    fn physical_delays_invert_back_to_the_time_delay_distance() {
        // For each image, D_Δt recovered from (Δt, Δτ) matches the model's D_Δt.
        let cosmo = FlatLCDM::planck18();
        let (z_lens, z_source) = (0.5, 2.0);
        let images = [image_with_tau(0.0), image_with_tau(0.03)];
        let d_dt = cosmo.time_delay_distance(z_lens, z_source);
        let dt = time_delays(&images, &cosmo, z_lens, z_source);
        let dtau = arrival_delays(&images);
        // Image 0 is the earliest (zero delay); check the second against the distance.
        assert_eq!(dt[0], 0.0);
        assert!((time_delay_distance_from_delay(dt[1], dtau[1]) - d_dt).abs() / d_dt < 1e-12);
    }
}
