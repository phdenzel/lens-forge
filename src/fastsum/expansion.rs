// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`LocalExpansion`]: the field's Taylor tower about a target centre.
//!
//! Taylor expansion of the holomorphic potential Φ about a source-free centre
//! `z_t`: with `s_i = z_i − z_t` and `w = z − z_t`,
//! Φ = (1/π)[Σ mᵢ ln(−sᵢ) − Σ_{k≥1} (1/k)(Σ mᵢ sᵢ^{−k}) wᵏ] = Σ bₖ wᵏ. The
//! [`Jet`] is the order-≤2 truncation — b₀, b₁, b₂ give ψ, α, H — so
//! this is the general-order form of the field-attribute abstraction.

use crate::fastsum::{Monopole, Jet};
use crate::grid::Point2D;
use crate::utils::binomial::pascal;
use num::complex::Complex;
use std::f64::consts::FRAC_1_PI;

/// A field's local (Taylor) expansion of Φ about [`center`](Self::center);
/// `coeffs[k]` is the Taylor coefficient `b_k`.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalExpansion {
    /// Expansion centre.
    pub center: Point2D,
    /// Complex Taylor coefficients `b_k`, `k = 0..=order`.
    pub coeffs: Vec<Complex<f64>>,
}

impl LocalExpansion {
    /// Expand the field of `sources` about `center` up to `order` (`coeffs` has
    /// `order + 1` entries). Sources coincident with `center` are skipped.
    ///
    /// # Arguments
    /// * `sources` - the (far) monopoles generating the field.
    /// * `center` - source-free expansion centre.
    /// * `order` - highest Taylor coefficient retained.
    pub fn from_monopoles(sources: &[Monopole], center: Point2D, order: usize) -> Self {
        let zt = center.to_complex();
        let mut coeffs = vec![Complex::new(0.0, 0.0); order + 1];
        for src in sources {
            let s = src.position.to_complex() - zt; // s_i = z_i − z_t
            if s.norm_sqr() == 0.0 {
                continue; // cannot expand about a source
            }
            let m = src.strength;
            coeffs[0] += (-s).ln() * m; // b_0 = Σ mᵢ ln(−sᵢ)
            let inv_s = s.inv();
            let mut p = inv_s; // s_i^{-k}, k starting at 1
            for (k, c) in coeffs.iter_mut().enumerate().skip(1) {
                *c += p * (-m / k as f64); // b_k = −(1/k) Σ mᵢ sᵢ^{−k}
                p *= inv_s;
            }
        }
        for c in coeffs.iter_mut() {
            *c = *c * FRAC_1_PI;
        }
        Self { center, coeffs }
    }

    /// Evaluate the jet this expansion represents at `theta`.
    ///
    /// Exact at the centre (b₀, b₁, b₂ are the true derivatives there); the
    /// truncation error grows as `theta` moves out toward the nearest source.
    ///
    /// # Arguments
    /// * `theta` - query point (near the centre).
    pub fn eval_jet(&self, theta: Point2D) -> Jet {
        let w = theta.to_complex() - self.center.to_complex();
        let mut phi = Complex::new(0.0, 0.0);
        let mut dphi = Complex::new(0.0, 0.0);
        let mut ddphi = Complex::new(0.0, 0.0);
        // Φ = Σ bₖwᵏ, Φ′ = Σ k bₖwᵏ⁻¹, Φ″ = Σ k(k−1) bₖwᵏ⁻², via running powers.
        let mut wk = Complex::new(1.0, 0.0); // w^k
        let mut wk1 = Complex::new(0.0, 0.0); // w^{k-1}
        let mut wk2 = Complex::new(0.0, 0.0); // w^{k-2}
        for (k, &bk) in self.coeffs.iter().enumerate() {
            phi += bk * wk;
            if k >= 1 {
                dphi += bk * (k as f64) * wk1;
            }
            if k >= 2 {
                ddphi += bk * ((k * (k - 1)) as f64) * wk2;
            }
            wk2 = wk1;
            wk1 = wk;
            wk *= w;
        }
        Jet::from_holomorphic(phi, dphi, ddphi)
    }

    /// Shift this local expansion to `new_center`, same order (L2L translation).
    ///
    /// With `e = new_center − center`, `b_l' = Σ_{k≥l} C(k,l) e^{k−l} b_k` (an
    /// upper-triangular binomial mix; exact for the retained orders).
    ///
    /// # Arguments
    /// * `new_center` - the target expansion centre.
    pub fn translate(&self, new_center: Point2D) -> LocalExpansion {
        let n = self.coeffs.len(); // p + 1
        let binom = pascal(n.saturating_sub(1));
        let e = new_center.to_complex() - self.center.to_complex();
        let mut epow = Vec::with_capacity(n); // e^0..=e^p
        let mut acc = Complex::new(1.0, 0.0);
        for _ in 0..n {
            epow.push(acc);
            acc *= e;
        }
        let mut coeffs = vec![Complex::new(0.0, 0.0); n];
        for l in 0..n {
            let mut sum = Complex::new(0.0, 0.0);
            for k in l..n {
                sum += self.coeffs[k] * binom[k][l] * epow[k - l];
            }
            coeffs[l] = sum;
        }
        LocalExpansion { center: new_center, coeffs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::{DirectSum, FieldSolver, SolverConfig};
    use crate::utils::rng::Lcg;

    /// A ring of monopoles at `radius`, far from the origin.
    fn far_ring(n: usize, seed: u64, radius: f64) -> Vec<Monopole> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                Monopole::new(Point2D::from_polar(radius, a), rng.range(0.1, 0.6))
            })
            .collect()
    }

    #[test]
    fn local_expansion_is_exact_at_the_center() {
        // b₀,b₁,b₂ are the true Taylor coefficients, so the order-2 jet at the
        // centre matches the direct sum exactly.
        let bodies = far_ring(24, 0x11, 5.0);
        let center = Point2D::new(0.2, -0.1);
        let le = LocalExpansion::from_monopoles(&bodies, center, 2);
        let direct = DirectSum::build(&bodies, SolverConfig::default());
        let a = le.eval_jet(center);
        let b = direct.jet(center);
        assert!(a.gradient.distance(&b.gradient) < 1e-12);
        assert!((a.scalar - b.scalar).abs() < 1e-12);
        for r in 0..2 {
            for c in 0..2 {
                assert!((a.hessian[r][c] - b.hessian[r][c]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn local_expansion_converges_near_center() {
        // Away from the centre the truncated series improves with order.
        let bodies = far_ring(24, 0x22, 5.0);
        let direct = DirectSum::build(&bodies, SolverConfig::default());
        let theta = Point2D::new(0.4, 0.3);
        let exact = direct.jet(theta);
        let err = |order: usize| {
            LocalExpansion::from_monopoles(&bodies, Point2D::origin(), order)
                .eval_jet(theta)
                .gradient
                .distance(&exact.gradient)
        };
        let (e1, e4, e8) = (err(1), err(4), err(8));
        assert!(e4 < e1 && e8 < e4, "not monotone: {e1}, {e4}, {e8}");
        assert!(e8 < 1e-6, "order-8 error {e8} too large");
    }

    #[test]
    fn l2l_preserves_the_field_near_the_new_center() {
        // Re-centring a local expansion (L2L) must reproduce the same field near
        // the shifted centre, matching the direct sum.
        let bodies = far_ring(24, 0x33, 6.0);
        let le1 = LocalExpansion::from_monopoles(&bodies, Point2D::origin(), 12);
        let t2 = Point2D::new(0.3, -0.2);
        let le2 = le1.translate(t2);
        let direct = DirectSum::build(&bodies, SolverConfig::default());
        let theta = t2 + Point2D::new(0.1, 0.05);
        let a = le2.eval_jet(theta);
        let b = direct.jet(theta);
        assert!(a.gradient.distance(&b.gradient) < 1e-6);
        assert!((a.scalar - b.scalar).abs() < 1e-6);
    }
}
