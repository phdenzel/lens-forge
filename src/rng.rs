// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Small deterministic pseudo-random generator shared across the crate.
//!
//! A 64-bit linear congruential generator: not cryptographic, but fast,
//! dependency-free, and reproducible from a seed. Used for seeded site sampling
//! and as the fixture generator in tests.

/// A seedable 64-bit linear congruential generator.
pub struct Lcg(u64);

impl Lcg {
    /// New generator seeded with `seed`.
    ///
    /// # Arguments
    /// * `seed` - the initial state; equal seeds reproduce the same sequence.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next uniform sample in `[0, 1)`.
    pub fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Next uniform sample in `[lo, hi)`.
    ///
    /// # Arguments
    /// * `lo`, `hi` - the half-open interval to sample.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }

    /// Next standard-normal sample (mean 0, variance 1), by the Box–Muller
    /// transform. Each call consumes two uniforms and returns one deviate.
    pub fn gaussian(&mut self) -> f64 {
        // Draw u1 in (0, 1] so ln(u1) is finite.
        let u1 = 1.0 - self.f();
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_stays_in_range() {
        let mut rng = Lcg::new(0xC0FFEE);
        for _ in 0..1000 {
            let x = rng.f();
            assert!((0.0..1.0).contains(&x));
            let y = rng.range(-2.0, 5.0);
            assert!((-2.0..5.0).contains(&y));
        }
    }

    #[test]
    fn gaussian_has_zero_mean_and_unit_variance() {
        let mut rng = Lcg::new(0x9E3779B9);
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| rng.gaussian()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn same_seed_reproduces_the_sequence() {
        let (mut a, mut b) = (Lcg::new(42), Lcg::new(42));
        for _ in 0..100 {
            assert_eq!(a.gaussian(), b.gaussian());
        }
    }
}
