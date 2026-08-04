// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Physical constants, astrophysical units, and Planckian-unit conversions.
//!
//! All constants are in SI. The Planckian ("natural") system sets $c = \hbar = G = k_B = 1$;
//! its base units follow Saha (*Appendix D*): *marble* (mass) $=\sqrt{\hbar c/G}$, *lap*
//! (length) $=\sqrt{\hbar G/c^3}$, *tick* (time) $=\sqrt{\hbar G/c^5}$, *therm* (temperature).
//! With $c = 1$ a length equals a light-travel time, so `m_to_lightsec` gives e.g. 1 au ≈ 499 s.

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s, exact).
pub const C: f64 = 2.997_924_58e8;
/// Newtonian constant of gravitation (m³ kg⁻¹ s⁻², CODATA 2018).
pub const G: f64 = 6.674_30e-11;
/// Reduced Planck constant ℏ (J s).
pub const H_BAR: f64 = 1.054_571_817e-34;
/// Boltzmann constant (J K⁻¹, SI 2019).
pub const K_B: f64 = 1.380_649e-23;

/// Planck mass √(ℏc/G) — the Planckian *marble* (kg).
pub const PLANCK_MASS: f64 = 2.176_434e-8;
/// Planck length √(ℏG/c³) — the Planckian *lap* (m).
pub const PLANCK_LENGTH: f64 = 1.616_255e-35;
/// Planck time √(ℏG/c⁵) — the Planckian *tick* (s).
pub const PLANCK_TIME: f64 = 5.391_247e-44;
/// Planck temperature √(ℏc⁵/G)/k_B — the Planckian *therm* (K).
pub const PLANCK_TEMPERATURE: f64 = 1.416_784e32;

/// Solar mass (kg).
pub const M_SUN: f64 = 1.988_47e30;
/// Parsec (m).
pub const PARSEC: f64 = 3.085_677_581_491_367_3e16;
/// Megaparsec (m).
pub const MPC: f64 = PARSEC * 1e6;
/// Astronomical unit (m, exact).
pub const AU: f64 = 1.495_978_707e11;
/// Gigayear (s, Julian year × 10⁹).
pub const GYR: f64 = 365.25 * 86_400.0 * 1e9;
/// Arcsecond in radians (π/648000).
pub const ARCSEC: f64 = PI / 648_000.0;
/// Day (s).
pub const DAY: f64 = 86_400.0;

/// Mass in kg → Planckian *marbles*.
#[inline]
pub fn kg_to_marble(m: f64) -> f64 {
    m / PLANCK_MASS
}
/// Mass in Planckian *marbles* → kg.
#[inline]
pub fn marble_to_kg(m: f64) -> f64 {
    m * PLANCK_MASS
}
/// Length in m → Planckian *laps*.
#[inline]
pub fn m_to_lap(l: f64) -> f64 {
    l / PLANCK_LENGTH
}
/// Length in Planckian *laps* → m.
#[inline]
pub fn lap_to_m(l: f64) -> f64 {
    l * PLANCK_LENGTH
}
/// Time in s → Planckian *ticks*.
#[inline]
pub fn s_to_tick(t: f64) -> f64 {
    t / PLANCK_TIME
}
/// Time in Planckian *ticks* → s.
#[inline]
pub fn tick_to_s(t: f64) -> f64 {
    t * PLANCK_TIME
}

/// Solar masses → Planckian *marbles*.
#[inline]
pub fn msun_to_marble(m: f64) -> f64 {
    kg_to_marble(m * M_SUN)
}
/// Megaparsecs → Planckian *laps*.
#[inline]
pub fn mpc_to_lap(l: f64) -> f64 {
    m_to_lap(l * MPC)
}
/// Gigayears → Planckian *ticks*.
#[inline]
pub fn gyr_to_tick(t: f64) -> f64 {
    s_to_tick(t * GYR)
}

/// Length in m → light-travel time in s (the $c=1$ bridge).
#[inline]
pub fn m_to_lightsec(l: f64) -> f64 {
    l / C
}
/// Light-travel time in s → length in m.
#[inline]
pub fn lightsec_to_m(t: f64) -> f64 {
    t * C
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planck_units_are_self_consistent() {
        // marble = √(ℏc/G), lap = √(ℏG/c³), tick = √(ℏG/c⁵).
        assert!(((PLANCK_MASS - (H_BAR * C / G).sqrt()) / PLANCK_MASS).abs() < 1e-4);
        assert!(((PLANCK_LENGTH - (H_BAR * G / C.powi(3)).sqrt()) / PLANCK_LENGTH).abs() < 1e-4);
        assert!(((PLANCK_TIME - (H_BAR * G / C.powi(5)).sqrt()) / PLANCK_TIME).abs() < 1e-4);
    }

    #[test]
    fn lengths_as_light_travel_times_match_appendix_d() {
        assert!((m_to_lightsec(AU) - 499.0).abs() < 0.1); // Saha D.6: au = 499.00 s
        assert!((m_to_lightsec(PARSEC) / 1.029e8 - 1.0).abs() < 1e-3); // parsec = 1.029e8 s
    }

    #[test]
    fn solar_mass_in_marbles_matches_saha_landau_cross_check() {
        // Saha D.4–D.5: M_sun = 0.54 M_L, M_L = m_b⁻² marble, m_b = 7.688e-20 marble.
        let expected = 0.54 * (7.688e-20f64).powi(-2);
        assert!((msun_to_marble(1.0) / expected - 1.0).abs() < 1e-2);
    }

    #[test]
    fn conversions_round_trip() {
        assert!((marble_to_kg(kg_to_marble(3.5)) - 3.5).abs() < 1e-12);
        assert!((lap_to_m(m_to_lap(2.0)) - 2.0).abs() < 1e-20);
        assert!((tick_to_s(s_to_tick(7.0)) - 7.0).abs() < 1e-40);
        assert!((lightsec_to_m(m_to_lightsec(MPC)) / MPC - 1.0).abs() < 1e-12);
    }

    #[test]
    fn astro_helpers_compose_base_conversions() {
        assert!((mpc_to_lap(1.0) - m_to_lap(MPC)).abs() < 1.0);
        assert!((gyr_to_tick(1.0) - s_to_tick(GYR)).abs() < 1.0);
        assert!((msun_to_marble(2.0) - kg_to_marble(2.0 * M_SUN)).abs() < 1.0);
    }
}
