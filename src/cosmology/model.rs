// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Background cosmology and cosmological distances.
//!
//! Every model implements [`Cosmology`]. Its required primitives — `h0` (km/s/Mpc), the
//! dimensionless expansion rate `e(z)` = $H(z)/H_0$, and `omega_k` — determine all the
//! provided distance measures (comoving, transverse-comoving, angular-diameter, luminosity),
//! computed by integrating $1/E(z)$ (Hogg, *Distance measures in cosmology*, arXiv:astro-ph/9905116).
//! Distances are in Mpc; constants come from [`units`].

use crate::cosmology::units;
use crate::numerical::quad::integrate;

/// A homogeneous, isotropic background cosmology.
pub trait Cosmology {
    /// Hubble constant $H_0$ (km/s/Mpc).
    fn h0(&self) -> f64;

    /// Dimensionless expansion rate $E(z) = H(z)/H_0$.
    fn e(&self, z: f64) -> f64;

    /// Curvature density parameter $\Omega_k$ (0 when flat).
    fn omega_k(&self) -> f64;

    /// Dimensionless Hubble parameter $h = H_0/100$.
    fn little_h(&self) -> f64 {
        self.h0() / 100.0
    }

    /// Hubble constant in inverse seconds (s⁻¹).
    fn h0_hz(&self) -> f64 {
        self.h0() * 1e3 / units::MPC
    }

    /// Hubble constant in atto-Hertz (10⁻¹⁸ s⁻¹).
    fn h0_ahz(&self) -> f64 {
        self.h0_hz() * 1e18
    }

    /// Hubble distance $D_H = c/H_0$ (Mpc).
    fn hubble_distance(&self) -> f64 {
        (units::C / 1e3) / self.h0()
    }

    /// Hubble time $H_0^{-1}$ (Gyr).
    fn hubble_time_gyr(&self) -> f64 {
        1.0 / self.h0_hz() / units::GYR
    }

    /// Line-of-sight comoving distance to redshift `z` (Mpc).
    ///
    /// # Arguments
    /// * `z` - redshift.
    fn comoving_distance(&self, z: f64) -> f64 {
        self.hubble_distance() * integrate(|zp| 1.0 / self.e(zp), 0.0, z)
    }

    /// Line-of-sight comoving distance between redshifts (Mpc).
    ///
    /// # Arguments
    /// * `z1` - nearer redshift.
    /// * `z2` - farther redshift.
    fn comoving_distance_between(&self, z1: f64, z2: f64) -> f64 {
        self.hubble_distance() * integrate(|zp| 1.0 / self.e(zp), z1, z2)
    }

    /// Transverse comoving distance $D_M$ to redshift `z` (Mpc), curvature-aware.
    ///
    /// # Arguments
    /// * `z` - redshift.
    fn transverse_comoving_distance(&self, z: f64) -> f64 {
        let dc = self.comoving_distance(z);
        let ok = self.omega_k();
        let dh = self.hubble_distance();
        if ok.abs() < 1e-8 {
            dc
        } else if ok > 0.0 {
            dh / ok.sqrt() * (ok.sqrt() * dc / dh).sinh()
        } else {
            let s = (-ok).sqrt();
            dh / s * (s * dc / dh).sin()
        }
    }

    /// Angular-diameter distance to redshift `z` (Mpc).
    ///
    /// # Arguments
    /// * `z` - redshift.
    fn angular_diameter_distance(&self, z: f64) -> f64 {
        self.transverse_comoving_distance(z) / (1.0 + z)
    }

    /// Angular-diameter distance between two redshifts (Mpc; Hogg eq. 19, valid for $\Omega_k \ge 0$).
    ///
    /// # Arguments
    /// * `z1` - lens (nearer) redshift.
    /// * `z2` - source (farther) redshift.
    fn angular_diameter_distance_between(&self, z1: f64, z2: f64) -> f64 {
        let dm1 = self.transverse_comoving_distance(z1);
        let dm2 = self.transverse_comoving_distance(z2);
        let dh = self.hubble_distance();
        let ok = self.omega_k();
        (dm2 * (1.0 + ok * dm1 * dm1 / (dh * dh)).sqrt()
            - dm1 * (1.0 + ok * dm2 * dm2 / (dh * dh)).sqrt())
            / (1.0 + z2)
    }

    /// Luminosity distance to redshift `z` (Mpc).
    ///
    /// # Arguments
    /// * `z` - redshift.
    fn luminosity_distance(&self, z: f64) -> f64 {
        (1.0 + z) * self.transverse_comoving_distance(z)
    }

    /// Lensing efficiency $D_{ds}/D_s$ (dimensionless; physical → reduced deflection factor).
    ///
    /// # Arguments
    /// * `z_lens` - lens redshift.
    /// * `z_source` - source redshift.
    fn lensing_efficiency(&self, z_lens: f64, z_source: f64) -> f64 {
        self.angular_diameter_distance_between(z_lens, z_source)
            / self.angular_diameter_distance(z_source)
    }

    /// Time-delay distance $D_{\Delta t} = (1+z_l)\,D_d D_s/D_{ds}$ (Mpc).
    ///
    /// # Arguments
    /// * `z_lens` - lens redshift.
    /// * `z_source` - source redshift.
    fn time_delay_distance(&self, z_lens: f64, z_source: f64) -> f64 {
        let dd = self.angular_diameter_distance(z_lens);
        let ds = self.angular_diameter_distance(z_source);
        let dds = self.angular_diameter_distance_between(z_lens, z_source);
        (1.0 + z_lens) * dd * ds / dds
    }

    /// Infer $H_0$ (km/s/Mpc) from a measured time-delay distance.
    ///
    /// Every distance scales as $1/H_0$ at fixed density parameters, so $D_{\Delta t}\propto
    /// 1/H_0$ and the inference is an exact rescale of this model's `h0`.
    ///
    /// # Arguments
    /// * `z_lens` - lens redshift.
    /// * `z_source` - source redshift.
    /// * `d_dt_mpc` - measured time-delay distance (Mpc).
    fn h0_from_time_delay_distance(&self, z_lens: f64, z_source: f64, d_dt_mpc: f64) -> f64 {
        self.h0() * self.time_delay_distance(z_lens, z_source) / d_dt_mpc
    }
}

/// Time-delay distance (Mpc) from an observed delay and a modelled Fermat-potential difference.
///
/// Inverts $\Delta t = \frac{D_{\Delta t}}{c}\Delta\tau$. The Fermat difference `delta_fermat`
/// ($\tfrac12|\theta-\beta|^2-\psi$, rad²) is produced by the lens/solver layer and `delta_t` is
/// the observable; feed the result into [`Cosmology::h0_from_time_delay_distance`] to infer $H_0$
/// (Denzel et al. 2021).
///
/// # Arguments
/// * `delta_t_seconds` - observed time delay between images (s).
/// * `delta_fermat` - Fermat-potential difference between images (rad²).
pub fn time_delay_distance_from_delay(delta_t_seconds: f64, delta_fermat: f64) -> f64 {
    units::C * delta_t_seconds / (delta_fermat * units::MPC)
}

/// Physical time delay (s) from a time-delay distance and a modelled Fermat-potential difference.
///
/// The forward of [`time_delay_distance_from_delay`]: $\Delta t = \frac{D_{\Delta t}}{c}\Delta\tau$.
/// `d_dt_mpc` is the time-delay distance (e.g. from [`Cosmology::time_delay_distance`]) and
/// `delta_fermat` ($\tfrac12|\theta-\beta|^2-\psi$, rad²) comes from the lens/solver layer.
///
/// # Arguments
/// * `d_dt_mpc` - time-delay distance (Mpc).
/// * `delta_fermat` - Fermat-potential difference between images (rad²).
pub fn delay_from_time_delay_distance(d_dt_mpc: f64, delta_fermat: f64) -> f64 {
    d_dt_mpc * units::MPC * delta_fermat / units::C
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::FlatLCDM;

    /// General (possibly curved) test cosmology to exercise the curvature branches.
    struct CurvedTest {
        h0: f64,
        omega_m: f64,
        omega_k: f64,
    }
    impl Cosmology for CurvedTest {
        fn h0(&self) -> f64 {
            self.h0
        }
        fn e(&self, z: f64) -> f64 {
            let ol = 1.0 - self.omega_m - self.omega_k;
            (self.omega_m * (1.0 + z).powi(3) + self.omega_k * (1.0 + z).powi(2) + ol).sqrt()
        }
        fn omega_k(&self) -> f64 {
            self.omega_k
        }
    }

    #[test]
    fn curvature_bends_transverse_distance() {
        let z = 1.0;
        let flat = CurvedTest {
            h0: 70.0,
            omega_m: 0.3,
            omega_k: 0.0,
        };
        let open = CurvedTest {
            h0: 70.0,
            omega_m: 0.3,
            omega_k: 0.2,
        };
        let closed = CurvedTest {
            h0: 70.0,
            omega_m: 0.3,
            omega_k: -0.2,
        };
        // Flat: D_M == D_C. Open (Ωk>0, sinh): D_M > D_C. Closed (Ωk<0, sin): D_M < D_C.
        assert!((flat.transverse_comoving_distance(z) - flat.comoving_distance(z)).abs() < 1e-9);
        assert!(open.transverse_comoving_distance(z) > open.comoving_distance(z));
        assert!(closed.transverse_comoving_distance(z) < closed.comoving_distance(z));
        // Pair distance with curvature stays positive and finite.
        assert!(open.angular_diameter_distance_between(0.5, z) > 0.0);
        assert!(closed.angular_diameter_distance_between(0.5, z) > 0.0);
    }

    #[test]
    fn comoving_distance_between_and_hubble_time() {
        let cosmo = FlatLCDM::planck18();
        let between = cosmo.comoving_distance_between(0.5, 2.0);
        let diff = cosmo.comoving_distance(2.0) - cosmo.comoving_distance(0.5);
        assert!((between - diff).abs() < 1e-6);
        // H0⁻¹ ≈ 14.5 Gyr for Planck18.
        assert!((cosmo.hubble_time_gyr() - 14.46).abs() < 0.1);
    }

    #[test]
    fn time_delay_distance_is_positive_and_inverse_in_h0() {
        let (zl, zs) = (0.5, 2.0);
        let a = FlatLCDM::new(70.0, 0.3, 0.0);
        let b = FlatLCDM::new(140.0, 0.3, 0.0); // only H0 differs
        assert!(a.time_delay_distance(zl, zs) > 0.0);
        // D_Δt ∝ 1/H0: doubling H0 halves the time-delay distance.
        let ratio = a.time_delay_distance(zl, zs) / b.time_delay_distance(zl, zs);
        assert!((ratio - 2.0).abs() < 1e-9, "ratio={ratio}");
    }

    #[test]
    fn h0_inference_round_trips() {
        let cosmo = FlatLCDM::planck18();
        let (zl, zs) = (0.4, 1.8);
        let d_dt = cosmo.time_delay_distance(zl, zs);
        assert!((cosmo.h0_from_time_delay_distance(zl, zs, d_dt) - cosmo.h0()).abs() < 1e-9);
    }

    #[test]
    fn lensing_efficiency_bounds() {
        let cosmo = FlatLCDM::planck18();
        let eff = cosmo.lensing_efficiency(0.5, 2.0);
        assert!(eff > 0.0 && eff < 1.0, "eff={eff}");
        // D_ds/D_s → 0 as the lens approaches the source.
        assert!(cosmo.lensing_efficiency(1.99, 2.0) < eff);
    }

    #[test]
    fn pair_distance_reduces_to_single() {
        let cosmo = FlatLCDM::planck18();
        let z = 1.3;
        assert!(
            (cosmo.angular_diameter_distance_between(0.0, z) - cosmo.angular_diameter_distance(z))
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn time_delay_distance_from_delay_arithmetic() {
        // 100-day delay, Fermat difference 1e-10 rad²; independent SI constants.
        let (dt, dtau) = (100.0 * 86_400.0, 1e-10);
        let expected = 2.997_924_58e8 * dt / (dtau * 3.085_677_581_491_367_3e22);
        assert!((time_delay_distance_from_delay(dt, dtau) - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn delay_and_distance_conversions_are_inverse() {
        // D_Δt → Δt → D_Δt round-trips exactly.
        let (d_dt, dtau) = (1234.5, 0.02);
        let dt = delay_from_time_delay_distance(d_dt, dtau);
        assert!((time_delay_distance_from_delay(dt, dtau) - d_dt).abs() / d_dt < 1e-12);
    }
}

