// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`Multipole`]: a source's far-field expansion.
//!
//! Laurent expansion of the holomorphic potential Φ(z) = (1/π) Σ mᵢ ln(z − zᵢ)
//! about a centre `z_c`: with `t_i = z_i − z_c` and `w = z − z_c`,
//! Φ = (1/π)[M₀ ln w − Σ_{k≥1} (M_k/k) w^{−k}], `M_k = Σ mᵢ tᵢᵏ`. The evaluated
//! field is read from Φ (ψ = Re Φ, α = conj Φ′, shear = conj Φ″). It is the
//! order-0 case (`M₀` only) that the Barnes–Hut node acceptance already uses.

use crate::fastsum::{Monopole, Jet};
use crate::grid::Point2D;
use num::complex::Complex;
use std::f64::consts::FRAC_1_PI;

/// A source's multipole (Laurent) expansion of Φ about [`center`](Self::center):
/// the real M₀ lives in [`strength`](Self::strength); `moments[i]` is `M_{i+1}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Multipole {
    /// Expansion centre.
    pub center: Point2D,
    /// The order-0 moment `M₀ = Σ mᵢ` (real).
    pub strength: f64,
    /// Complex moments `M_k` for `k ≥ 1`; `moments[i]` is `M_{i+1}`.
    pub moments: Vec<Complex<f64>>,
}

impl Multipole {
    /// Expand `sources` about `center` up to `order` (`moments` has `order`
    /// entries `M_1..=M_order`; M₀ in [`strength`](Self::strength)),
    /// `M_k = Σ mᵢ (zᵢ − center)ᵏ`.
    ///
    /// # Arguments
    /// * `sources` - the monopoles.
    /// * `center` - expansion centre.
    /// * `order` - highest moment retained.
    pub fn from_monopoles(sources: &[Monopole], center: Point2D, order: usize) -> Self {
        let zc = center.to_complex();
        let mut strength = 0.0;
        let mut moments = vec![Complex::new(0.0, 0.0); order];
        for s in sources {
            let t = s.position.to_complex() - zc;
            strength += s.strength; // M₀
            let mut tk = t; // t^1
            for c in moments.iter_mut() {
                *c += tk * s.strength; // M_k = Σ mᵢ tᵢᵏ
                tk *= t;
            }
        }
        Self { center, strength, moments }
    }

    /// Evaluate the far-field jet this expansion induces at `theta`.
    ///
    /// Valid only outside the field (`|theta − center|` beyond its radius);
    /// returns a zero jet exactly at the centre, where the series has a pole.
    ///
    /// # Arguments
    /// * `theta` - query point.
    pub fn eval_jet(&self, theta: Point2D) -> Jet {
        let w = theta.to_complex() - self.center.to_complex();
        if w.norm_sqr() == 0.0 {
            return Jet::default();
        }
        let inv_w = w.inv();
        let m0 = Complex::new(self.strength, 0.0);
        // Φ, Φ', Φ'' as Laurent sums (prefactor 1/π folded in at the end).
        let mut phi = m0 * w.ln();
        let mut dphi = m0 * inv_w;
        let mut ddphi = -m0 * inv_w * inv_w;
        let mut wk = inv_w; // w^{-k}, k starting at 1
        for (i, &mk) in self.moments.iter().enumerate() {
            let kf = (i + 1) as f64; // moments[i] = M_{i+1}
            phi -= mk * wk / kf;
            dphi += mk * wk * inv_w;
            ddphi -= mk * wk * inv_w * inv_w * (kf + 1.0);
            wk *= inv_w;
        }
        Jet::from_holomorphic(phi * FRAC_1_PI, dphi * FRAC_1_PI, ddphi * FRAC_1_PI)
    }
}

impl From<Monopole> for Multipole {
    /// Promote a point mass to an order-0 multipole centred on itself.
    fn from(m: Monopole) -> Self {
        Self { center: m.position, strength: m.strength, moments: Vec::new() }
    }
}

impl TryFrom<Multipole> for Monopole {
    /// The un-collapsed multipole, returned when it carries higher moments.
    type Error = Multipole;

    /// Collapse an order-0 multipole to its point mass; fails (lossy) if any
    /// higher moment is present.
    fn try_from(m: Multipole) -> Result<Self, Self::Error> {
        if m.moments.is_empty() {
            Ok(Monopole::new(m.center, m.strength))
        } else {
            Err(m)
        }
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
        let mp = Multipole::from(s);
        let direct = DirectSum::build(&[s], SolverConfig::default());
        let theta = Point2D::new(3.0, 1.0);
        let a = mp.eval_jet(theta);
        let b = direct.jet(theta);
        assert!(a.gradient.distance(&b.gradient) < 1e-12);
        assert!((a.scalar - b.scalar).abs() < 1e-12);
        assert!(a.isotropic().abs() < 1e-12); // trace-free away from the source
    }

    #[test]
    fn monopole_multipole_round_trip() {
        // Monopole → order-0 Multipole → Monopole is lossless; a multipole with
        // higher moments refuses to collapse and is handed back.
        let s = Monopole::new(Point2D::new(0.1, -0.2), 0.7);
        let back = Monopole::try_from(Multipole::from(s)).unwrap();
        assert_eq!(back, s);
        let higher = Multipole::from_monopoles(&cluster(8, 0x5, 0.3), Point2D::origin(), 2);
        assert!(Monopole::try_from(higher).is_err());
    }

    #[test]
    fn multipole_converges_to_direct_far_field() {
        // Error at a far point must shrink as the retained order grows.
        let bodies = cluster(40, 0xB19, 0.3);
        let direct = DirectSum::build(&bodies, SolverConfig::default());
        let theta = Point2D::new(5.0, 3.0);
        let exact = direct.jet(theta);
        let err = |order: usize| {
            Multipole::from_monopoles(&bodies, Point2D::origin(), order)
                .eval_jet(theta)
                .gradient
                .distance(&exact.gradient)
        };
        let (e1, e4, e8) = (err(1), err(4), err(8));
        assert!(e4 < e1 && e8 < e4, "not monotone: {e1}, {e4}, {e8}");
        assert!(e8 < 1e-8, "order-8 error {e8} too large");
        let hi = Multipole::from_monopoles(&bodies, Point2D::origin(), 14).eval_jet(theta);
        assert!((hi.scalar - exact.scalar).abs() < 1e-8);
    }
}
