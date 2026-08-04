// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Flat ΛCDM background cosmology.
//!
//! A spatially flat model with matter, radiation, and a cosmological constant
//! (Saha, *magicenv* §8.3, eq. 8.17):
//! $E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_r(1+z)^4 + \Omega_\Lambda}$ with
//! $\Omega_\Lambda = 1 - \Omega_m - \Omega_r$ and $\Omega_k = 0$. The radiation term is small
//! today ($\Omega_r \sim 10^{-4}$) but grows as $(1+z)^4$ at high redshift. The first
//! [`Cosmology`] implementation.

use crate::cosmology::model::Cosmology;
use crate::cosmology::units;

/// A flat ΛCDM cosmology of Hubble constant `h0` (km/s/Mpc), matter density `omega_m`,
/// and radiation density `omega_r`.
pub struct FlatLCDM {
    h0: f64,
    omega_m: f64,
    omega_r: f64,
}

impl FlatLCDM {
    /// Create a flat ΛCDM model ($\Omega_\Lambda = 1 - \Omega_m - \Omega_r$).
    ///
    /// # Arguments
    /// * `h0_km_s_mpc` - Hubble constant H₀ (km/s/Mpc).
    /// * `omega_m` - matter density parameter Ω_m.
    /// * `omega_r` - radiation density parameter Ω_r.
    pub fn new(h0_km_s_mpc: f64, omega_m: f64, omega_r: f64) -> Self {
        Self {
            h0: h0_km_s_mpc,
            omega_m,
            omega_r,
        }
    }

    /// Create a flat ΛCDM model with H₀ given in atto-Hertz (10⁻¹⁸ s⁻¹).
    ///
    /// # Arguments
    /// * `h0_ahz` - Hubble constant H₀ (atto-Hertz).
    /// * `omega_m` - matter density parameter Ω_m.
    /// * `omega_r` - radiation density parameter Ω_r.
    pub fn from_h0_ahz(h0_ahz: f64, omega_m: f64, omega_r: f64) -> Self {
        Self::new(h0_ahz * 1e-18 * units::MPC / 1e3, omega_m, omega_r)
    }

    /// The Planck 2018 flat ΛCDM parameters (H₀ = 67.66, Ω_m = 0.3111, Ω_r = 9.14e-5;
    /// Planck 2018 VI, radiation from photons + neutrinos at Neff = 3.046).
    pub fn planck18() -> Self {
        Self::new(67.66, 0.3111, 9.14e-5)
    }

    /// Matter density parameter Ω_m.
    pub fn omega_m(&self) -> f64 {
        self.omega_m
    }

    /// Radiation density parameter Ω_r.
    pub fn omega_r(&self) -> f64 {
        self.omega_r
    }

    /// Dark-energy density parameter $\Omega_\Lambda = 1 - \Omega_m - \Omega_r$.
    pub fn omega_l(&self) -> f64 {
        1.0 - self.omega_m - self.omega_r
    }
}

impl Cosmology for FlatLCDM {
    fn h0(&self) -> f64 {
        self.h0
    }

    fn e(&self, z: f64) -> f64 {
        (self.omega_m * (1.0 + z).powi(3) + self.omega_r * (1.0 + z).powi(4) + self.omega_l())
            .sqrt()
    }

    fn omega_k(&self) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Einstein–de Sitter (Ω_m = 1, no radiation): closed-form comoving distance is ground truth.
    #[test]
    fn einstein_de_sitter_matches_closed_form() {
        let cosmo = FlatLCDM::new(70.0, 1.0, 0.0);
        let dh = cosmo.hubble_distance();
        for &z in &[0.5f64, 1.0, 2.5] {
            let dc = 2.0 * dh * (1.0 - 1.0 / (1.0 + z).sqrt()); // D_C = 2 D_H (1 - 1/√(1+z))
            assert!((cosmo.comoving_distance(z) - dc).abs() < 1e-6, "z={z}");
            assert!((cosmo.angular_diameter_distance(z) - dc / (1.0 + z)).abs() < 1e-6);
        }
    }

    #[test]
    fn distance_duality_holds() {
        let cosmo = FlatLCDM::planck18();
        for &z in &[0.2, 1.0, 3.0] {
            let dl = cosmo.luminosity_distance(z);
            let da = cosmo.angular_diameter_distance(z);
            assert!((dl - (1.0 + z).powi(2) * da).abs() < 1e-9, "z={z}");
        }
    }

    #[test]
    fn trivial_limits() {
        let cosmo = FlatLCDM::planck18();
        assert!((cosmo.e(0.0) - 1.0).abs() < 1e-12); // densities sum to unity
        assert!(cosmo.angular_diameter_distance(0.0).abs() < 1e-9);
        assert!((cosmo.omega_m() - 0.3111).abs() < 1e-12);
        assert!((cosmo.omega_r() - 9.14e-5).abs() < 1e-12);
        assert!((cosmo.omega_l() - (1.0 - 0.3111 - 9.14e-5)).abs() < 1e-12);
    }

    #[test]
    fn radiation_term_grows_at_high_redshift() {
        // With Ω_r, E(z) exceeds the radiation-free value, and increasingly so at high z.
        let with_r = FlatLCDM::new(70.0, 0.3, 1e-3);
        let without_r = FlatLCDM::new(70.0, 0.3, 0.0);
        let low = with_r.e(1.0) - without_r.e(1.0);
        let high = with_r.e(10.0) - without_r.e(10.0);
        assert!(low > 0.0 && high > low, "low={low} high={high}");
    }

    #[test]
    fn h0_unit_views_round_trip() {
        let cosmo = FlatLCDM::planck18();
        assert!((cosmo.little_h() - 0.6766).abs() < 1e-12);
        let round = FlatLCDM::from_h0_ahz(cosmo.h0_ahz(), cosmo.omega_m(), cosmo.omega_r());
        assert!((round.h0() - cosmo.h0()).abs() < 1e-9);
    }

    /// ΛCDM regression: cross-check comoving_distance against an independent composite
    /// Simpson quadrature (a different method from the crate's adaptive integrator).
    #[test]
    fn lambda_cdm_comoving_distance_matches_independent_quadrature() {
        let cosmo = FlatLCDM::planck18();
        let (om, or) = (0.3111, 9.14e-5);
        let e = |z: f64| (om * (1.0 + z).powi(3) + or * (1.0 + z).powi(4) + (1.0 - om - or)).sqrt();
        let (n, z) = (2000usize, 1.0);
        let h = z / n as f64;
        let mut s = 1.0 / e(0.0) + 1.0 / e(z);
        for i in 1..n {
            s += (if i % 2 == 1 { 4.0 } else { 2.0 }) / e(i as f64 * h);
        }
        let reference = cosmo.hubble_distance() * s * h / 3.0; // ≈ 3395.5 Mpc
        assert!((cosmo.comoving_distance(1.0) - reference).abs() / reference < 2e-3);
    }
}
