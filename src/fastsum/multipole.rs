// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`Multipole`]: a source cluster's far-field expansion.
//!
//! Laurent expansion of the holomorphic potential Φ(z) = (1/π) Σ mᵢ ln(z − zᵢ)
//! about a centre `z_c`: with `t_i = z_i − z_c` and `w = z − z_c`,
//! Φ = (1/π)[M₀ ln w − Σ_{k≥1} (M_k/k) w^{−k}], `M_k = Σ mᵢ tᵢᵏ`. The evaluated
//! field is read from Φ (ψ = Re Φ, α = conj Φ′, shear = conj Φ″). It is the
//! order-0 case (`M₀` only) that the Barnes–Hut node acceptance already uses.

use crate::fastsum::{Monopole, PotentialJet};
use crate::grid::Point2D;
use num::complex::Complex;
use std::f64::consts::FRAC_1_PI;

/// A cluster's multipole (Laurent) expansion of Φ about [`center`](Self::center);
/// `coeffs[k]` is the moment `M_k`, with `coeffs[0]` the total strength `M₀`.
pub struct Multipole {
    /// Expansion centre.
    pub center: Point2D,
    /// Complex moments `M_k`, `k = 0..=order`.
    pub coeffs: Vec<Complex<f64>>,
}

impl Multipole {
    /// An order-0 multipole: a single aggregate monopole of `strength`.
    ///
    /// # Arguments
    /// * `center` - expansion centre.
    /// * `strength` - total (order-0) moment.
    pub fn monopole(center: Point2D, strength: f64) -> Self {
        Self { center, coeffs: vec![Complex::new(strength, 0.0)] }
    }

    /// Expand `sources` about `center` up to `order` (`coeffs` has `order + 1`
    /// entries), `M_k = Σ mᵢ (zᵢ − center)ᵏ`.
    ///
    /// # Arguments
    /// * `sources` - the cluster's monopoles.
    /// * `center` - expansion centre.
    /// * `order` - highest moment retained.
    pub fn from_monopoles(sources: &[Monopole], center: Point2D, order: usize) -> Self {
        let zc = center.to_complex();
        let mut coeffs = vec![Complex::new(0.0, 0.0); order + 1];
        for s in sources {
            let t = s.position.to_complex() - zc;
            let mut tk = Complex::new(1.0, 0.0); // t^0
            for c in coeffs.iter_mut() {
                *c += tk * s.strength;
                tk *= t;
            }
        }
        Self { center, coeffs }
    }

    /// Evaluate the far-field jet this expansion induces at `theta`.
    ///
    /// Valid only outside the cluster (`|theta − center|` beyond its radius);
    /// returns a zero jet exactly at the centre, where the series has a pole.
    ///
    /// # Arguments
    /// * `theta` - query point.
    pub fn eval_jet(&self, theta: Point2D) -> PotentialJet {
        let w = theta.to_complex() - self.center.to_complex();
        if w.norm_sqr() == 0.0 {
            return PotentialJet::default();
        }
        let inv_w = w.inv();
        let m0 = self.coeffs[0];
        // Φ, Φ', Φ'' as Laurent sums (prefactor 1/π folded in at the end).
        let mut phi = m0 * w.ln();
        let mut dphi = m0 * inv_w;
        let mut ddphi = -m0 * inv_w * inv_w;
        let mut wk = inv_w; // w^{-k}, k starting at 1
        for (k, &mk) in self.coeffs.iter().enumerate().skip(1) {
            let kf = k as f64;
            phi -= mk * wk / kf;
            dphi += mk * wk * inv_w;
            ddphi -= mk * wk * inv_w * inv_w * (kf + 1.0);
            wk *= inv_w;
        }
        PotentialJet::from_holomorphic(phi * FRAC_1_PI, dphi * FRAC_1_PI, ddphi * FRAC_1_PI)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::{DirectSum, FieldSolver, SolverConfig};
    use crate::utils::rng::Lcg;

    /// A small cluster of monopoles within `radius` of the origin.
    fn cluster(n: usize, seed: u64, radius: f64) -> Vec<Monopole> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let p = Point2D::from_polar(radius * rng.f().sqrt(), std::f64::consts::TAU * rng.f());
                Monopole::new(p, rng.range(0.1, 0.6))
            })
            .collect()
    }

    #[test]
    fn monopole_constructor_matches_a_single_source() {
        // Centred on the source, the order-0 expansion is exact, not approximate.
        let s = Monopole::new(Point2D::new(0.1, -0.2), 0.7);
        let mp = Multipole::monopole(s.position, s.strength);
        let direct = DirectSum::build(&[s], SolverConfig::default());
        let theta = Point2D::new(3.0, 1.0);
        let a = mp.eval_jet(theta);
        let b = direct.field(theta);
        assert!(a.deflection.distance(&b.deflection) < 1e-12);
        assert!((a.potential - b.potential).abs() < 1e-12);
        assert!(a.convergence().abs() < 1e-12); // trace-free away from the source
    }

    #[test]
    fn multipole_converges_to_direct_far_field() {
        // Error at a far point must shrink as the retained order grows.
        let bodies = cluster(40, 0xB19, 0.3);
        let direct = DirectSum::build(&bodies, SolverConfig::default());
        let theta = Point2D::new(5.0, 3.0);
        let exact = direct.field(theta);
        let err = |order: usize| {
            Multipole::from_monopoles(&bodies, Point2D::origin(), order)
                .eval_jet(theta)
                .deflection
                .distance(&exact.deflection)
        };
        let (e1, e4, e8) = (err(1), err(4), err(8));
        assert!(e4 < e1 && e8 < e4, "not monotone: {e1}, {e4}, {e8}");
        assert!(e8 < 1e-8, "order-8 error {e8} too large");
        let hi = Multipole::from_monopoles(&bodies, Point2D::origin(), 14).eval_jet(theta);
        assert!((hi.potential - exact.potential).abs() < 1e-8);
    }
}
