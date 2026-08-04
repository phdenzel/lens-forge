// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Flat ΛCDM background cosmology.
//!
//! A spatially flat model with matter and a cosmological constant:
//! $E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda}$ with $\Omega_\Lambda = 1 - \Omega_m$ and
//! $\Omega_k = 0$. The first [`Cosmology`] implementation.

use crate::cosmology::model::Cosmology;
use crate::cosmology::units;

/// A flat ΛCDM cosmology of Hubble constant `h0` (km/s/Mpc) and matter density `omega_m`.
pub struct FlatLCDM {
    h0: f64,
    omega_m: f64,
}

impl FlatLCDM {
    /// Create a flat ΛCDM model ($\Omega_\Lambda = 1 - \Omega_m$).
    ///
    /// # Arguments
    /// * `h0_km_s_mpc` - Hubble constant H₀ (km/s/Mpc).
    /// * `omega_m` - matter density parameter Ω_m.
    pub fn new(h0_km_s_mpc: f64, omega_m: f64) -> Self {
        Self {
            h0: h0_km_s_mpc,
            omega_m,
        }
    }

    /// Create a flat ΛCDM model with H₀ given in atto-Hertz (10⁻¹⁸ s⁻¹).
    ///
    /// # Arguments
    /// * `h0_ahz` - Hubble constant H₀ (atto-Hertz).
    /// * `omega_m` - matter density parameter Ω_m.
    pub fn from_h0_ahz(h0_ahz: f64, omega_m: f64) -> Self {
        Self::new(h0_ahz * 1e-18 * units::MPC / 1e3, omega_m)
    }

    /// The Planck 2018 flat ΛCDM parameters (H₀ = 67.66, Ω_m = 0.3111; Planck 2018 VI).
    pub fn planck18() -> Self {
        Self::new(67.66, 0.3111)
    }

    /// Matter density parameter Ω_m.
    pub fn omega_m(&self) -> f64 {
        self.omega_m
    }

    /// Dark-energy density parameter $\Omega_\Lambda = 1 - \Omega_m$.
    pub fn omega_l(&self) -> f64 {
        1.0 - self.omega_m
    }
}

impl Cosmology for FlatLCDM {
    fn h0(&self) -> f64 {
        self.h0
    }

    fn e(&self, z: f64) -> f64 {
        (self.omega_m * (1.0 + z).powi(3) + self.omega_l()).sqrt()
    }

    fn omega_k(&self) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Einstein–de Sitter (Ω_m = 1): closed-form comoving distance is the ground truth.
    #[test]
    fn einstein_de_sitter_matches_closed_form() {
        let cosmo = FlatLCDM::new(70.0, 1.0);
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
        assert!((cosmo.e(0.0) - 1.0).abs() < 1e-12);
        assert!(cosmo.angular_diameter_distance(0.0).abs() < 1e-9);
        assert!((cosmo.omega_m() - 0.3111).abs() < 1e-12);
        assert!((cosmo.omega_l() - (1.0 - 0.3111)).abs() < 1e-12);
    }

    #[test]
    fn h0_unit_views_round_trip() {
        let cosmo = FlatLCDM::planck18();
        assert!((cosmo.little_h() - 0.6766).abs() < 1e-12);
        let round = FlatLCDM::from_h0_ahz(cosmo.h0_ahz(), 0.3111);
        assert!((round.h0() - cosmo.h0()).abs() < 1e-9);
    }

    /// ΛCDM regression: cross-check comoving_distance against an independent composite
    /// Simpson quadrature (a different method from the crate's adaptive integrator).
    #[test]
    fn lambda_cdm_comoving_distance_matches_independent_quadrature() {
        let cosmo = FlatLCDM::planck18();
        let e = |z: f64| (0.3111 * (1.0 + z).powi(3) + (1.0 - 0.3111)).sqrt();
        let (n, z) = (2000usize, 1.0);
        let h = z / n as f64;
        let mut s = 1.0 / e(0.0) + 1.0 / e(z);
        for i in 1..n {
            s += (if i % 2 == 1 { 4.0 } else { 2.0 }) / e(i as f64 * h);
        }
        let reference = cosmo.hubble_distance() * s * h / 3.0; // ≈ 3395.87 Mpc
        assert!((cosmo.comoving_distance(1.0) - reference).abs() / reference < 2e-3);
    }
}
