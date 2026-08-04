// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Background cosmology and cosmological distances.
//!
//! Every model implements [`Cosmology`]. Its required primitives — `h0` (km/s/Mpc), the
//! dimensionless expansion rate `e(z)` = $H(z)/H_0$, and `omega_k` — determine all the
//! provided distance measures (comoving, transverse-comoving, angular-diameter, luminosity),
//! computed by integrating $1/E(z)$ (Hogg, *Distance measures in cosmology*, arXiv:astro-ph/9905116).
//! Distances are in Mpc; constants come from [`units`](crate::cosmology::units).

use crate::cosmology::units;

/// Recursive adaptive Simpson step: refine until the split matches `whole` within `eps`.
fn adaptive_simpson<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    eps: f64,
    depth: u32,
) -> f64 {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = f(lm);
    let frm = f(rm);
    let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    if depth == 0 || (left + right - whole).abs() <= 15.0 * eps {
        return left + right + (left + right - whole) / 15.0;
    }
    adaptive_simpson(f, a, m, fa, fm, flm, left, 0.5 * eps, depth - 1)
        + adaptive_simpson(f, m, b, fm, fb, frm, right, 0.5 * eps, depth - 1)
}

/// Integrate `f` over `[a, b]` by adaptive Simpson (`0` for an empty interval).
fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    if b <= a {
        return 0.0;
    }
    let m = 0.5 * (a + b);
    let (fa, fb, fm) = (f(a), f(b), f(m));
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    adaptive_simpson(&f, a, b, fa, fb, fm, whole, 1e-11, 50)
}

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
}
