// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`Multipole`]: a source's far-field expansion.
//!
//! Laurent expansion of the holomorphic potential
//! $\Phi(z) = \frac{1}{\pi} \sum_i m_i \ln(z - z_i)$ about a centre $z_c$: with
//! $t_i = z_i - z_c$ and $w = z - z_c$,
//! $$ \Phi = \frac{1}{\pi}\Big[ M_0 \ln w - \sum_{k \ge 1} \frac{M_k}{k} w^{-k} \Big], \qquad M_k = \sum_i m_i t_i^k. $$
//! The evaluated field is read from $\Phi$ ($\psi = \operatorname{Re}\Phi$,
//! $\alpha = \operatorname{conj}\Phi'$, shear $= \operatorname{conj}\Phi''$). It
//! is the order-0 case ($M_0$ only) that the Barnes–Hut node acceptance already
//! uses.

use crate::fastsum::{Jet, LocalExpansion, Monopole};
use crate::grid::Point2D;
use crate::numerical::binomial::pascal;
use num::complex::Complex;
use std::f64::consts::FRAC_1_PI;
use std::ops::Add;

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
        Self {
            center,
            strength,
            moments,
        }
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

    /// Shift this multipole to `new_center`, same order (M2M translation).
    ///
    /// With `d = center − new_center`, `M_l' = Σ_{k=0}^{l} C(l,k) d^{l−k} M_k`
    /// (a lower-triangular binomial mix; the total mass `M₀` is preserved).
    ///
    /// # Arguments
    /// * `new_center` - the target expansion centre.
    pub fn translate(&self, new_center: Point2D) -> Multipole {
        let p = self.moments.len();
        let binom = pascal(p);
        let d = self.center.to_complex() - new_center.to_complex();
        let mut dpow = Vec::with_capacity(p + 1); // d^0..=d^p
        let mut acc = Complex::new(1.0, 0.0);
        for _ in 0..=p {
            dpow.push(acc);
            acc *= d;
        }
        let moment = |k: usize| {
            if k == 0 {
                Complex::new(self.strength, 0.0)
            } else {
                self.moments[k - 1]
            }
        };
        let mut moments = vec![Complex::new(0.0, 0.0); p];
        for l in 1..=p {
            let mut sum = Complex::new(0.0, 0.0);
            for k in 0..=l {
                sum += dpow[l - k] * binom[l][k] * moment(k);
            }
            moments[l - 1] = sum;
        }
        Multipole {
            center: new_center,
            strength: self.strength,
            moments,
        }
    }

    /// Convert this far-field multipole into a local (Taylor) expansion about a
    /// well-separated `target` (M2L translation).
    ///
    /// With `D = target − center`: `b₀ = M₀ ln D − Σ_{k≥1} (M_k/k) D^{−k}` and,
    /// for `l ≥ 1`, `b_l = (−1)^{l+1}[ M₀/(l D^l) + Σ_{k≥1} (M_k/k) C(k+l−1,l)
    /// D^{−k−l} ]`. Valid only when the boxes are well separated (`|D|` beyond
    /// both radii).
    ///
    /// # Arguments
    /// * `target` - the local expansion centre.
    pub fn to_local(&self, target: Point2D) -> LocalExpansion {
        let p = self.moments.len();
        let binom = pascal(2 * p);
        let d = target.to_complex() - self.center.to_complex();
        let inv_d = d.inv();
        let mut dneg = Vec::with_capacity(2 * p + 1); // D^{−k}, k=0..=2p
        let mut acc = Complex::new(1.0, 0.0);
        for _ in 0..=2 * p {
            dneg.push(acc);
            acc *= inv_d;
        }
        let mut coeffs = vec![Complex::new(0.0, 0.0); p + 1];
        let mut b0 = Complex::new(self.strength, 0.0) * d.ln();
        for k in 1..=p {
            b0 -= self.moments[k - 1] * dneg[k] / k as f64;
        }
        coeffs[0] = b0;
        for l in 1..=p {
            let mut inner = Complex::new(self.strength / l as f64, 0.0) * dneg[l];
            for k in 1..=p {
                inner += self.moments[k - 1] / k as f64 * binom[k + l - 1][l] * dneg[k + l];
            }
            let sign = if l % 2 == 0 { -1.0 } else { 1.0 }; // (−1)^{l+1}
            coeffs[l] = inner * sign;
        }
        for c in coeffs.iter_mut() {
            *c *= FRAC_1_PI;
        }
        LocalExpansion {
            center: target,
            coeffs,
        }
    }
}

impl From<Monopole> for Multipole {
    /// Promote a point mass to an order-0 multipole centred on itself.
    fn from(m: Monopole) -> Self {
        Self {
            center: m.position,
            strength: m.strength,
            moments: Vec::new(),
        }
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

impl Add for Multipole {
    type Output = Multipole;

    /// Superpose two expansions about the **same** centre (the M2M merge of
    /// co-centred children); moments add term by term.
    ///
    /// # Arguments
    /// * `rhs` - the multipole to add; must share `self`'s centre and order.
    fn add(mut self, rhs: Multipole) -> Multipole {
        debug_assert_eq!(
            self.center, rhs.center,
            "Multipole::add needs a shared centre"
        );
        debug_assert_eq!(
            self.moments.len(),
            rhs.moments.len(),
            "Multipole::add needs equal order"
        );
        self.strength += rhs.strength;
        for (m, r) in self.moments.iter_mut().zip(rhs.moments.iter()) {
            *m += *r;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::{DirectSum, Summation, SumConfig};
    use crate::rng::Lcg;

    /// A small cluster of monopoles within `radius` of the origin.
    fn cluster(n: usize, seed: u64, radius: f64) -> Vec<Monopole> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let p =
                    Point2D::from_polar(radius * rng.f().sqrt(), std::f64::consts::TAU * rng.f());
                Monopole::new(p, rng.range(0.1, 0.6))
            })
            .collect()
    }

    #[test]
    fn monopole_constructor_matches_a_single_source() {
        // Centred on the source, the order-0 expansion is exact, not approximate.
        let s = Monopole::new(Point2D::new(0.1, -0.2), 0.7);
        let mp = Multipole::from(s);
        let direct = DirectSum::build(&[s], SumConfig::default());
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
        let direct = DirectSum::build(&bodies, SumConfig::default());
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

    #[test]
    fn add_superposes_co_centred_expansions() {
        // A shared-centre sum of two clusters' multipoles equals the multipole of
        // their union (the M2M merge relies on this linearity).
        let center = Point2D::new(0.1, -0.05);
        let a = cluster(12, 0x1, 0.3);
        let b = cluster(9, 0x2, 0.3);
        let ma = Multipole::from_monopoles(&a, center, 8);
        let mb = Multipole::from_monopoles(&b, center, 8);
        let union: Vec<Monopole> = a.iter().chain(b.iter()).copied().collect();
        let expected = Multipole::from_monopoles(&union, center, 8);
        let sum = ma + mb;
        assert_eq!(sum.center, expected.center);
        assert!((sum.strength - expected.strength).abs() < 1e-12);
        for (s, e) in sum.moments.iter().zip(expected.moments.iter()) {
            assert!((s - e).norm() < 1e-12);
        }
    }

    #[test]
    fn m2m_preserves_the_far_field() {
        // Shifting a multipole to a new centre (M2M) leaves the far field it
        // induces unchanged, still matching the direct sum.
        let bodies = cluster(30, 0x7, 0.3);
        let mp1 = Multipole::from_monopoles(&bodies, Point2D::new(0.05, -0.02), 10);
        let mp2 = mp1.translate(Point2D::new(-0.1, 0.08));
        let direct = DirectSum::build(&bodies, SumConfig::default());
        let theta = Point2D::new(6.0, -4.0);
        let a = mp2.eval_jet(theta);
        let b = direct.jet(theta);
        assert!(a.gradient.distance(&b.gradient) < 1e-8);
        assert!((a.scalar - b.scalar).abs() < 1e-8);
    }

    #[test]
    fn m2l_matches_direct_near_the_target() {
        // A source cluster's multipole, converted to a local expansion about a
        // well-separated target (M2L), matches the direct sum near that target.
        let bodies = cluster(30, 0x9, 0.3); // about the origin
        let mp = Multipole::from_monopoles(&bodies, Point2D::origin(), 12);
        let target = Point2D::new(8.0, 0.0);
        let le = mp.to_local(target);
        let direct = DirectSum::build(&bodies, SumConfig::default());
        let theta = target + Point2D::new(0.3, -0.2);
        let a = le.eval_jet(theta);
        let b = direct.jet(theta);
        assert!(a.gradient.distance(&b.gradient) < 1e-8, "grad mismatch");
        assert!((a.scalar - b.scalar).abs() < 1e-8, "scalar mismatch");
    }
}
