// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Small deterministic pseudo-random generator shared across the crate.
//!
//! A 64-bit linear congruential generator: not cryptographic, but fast,
//! dependency-free, and reproducible from a seed. Used for seeded site sampling
//! and as the fixture generator in tests.

/// A seedable 64-bit linear congruential generator.
pub(crate) struct Lcg(u64);

impl Lcg {
    /// New generator seeded with `seed`.
    ///
    /// # Arguments
    /// * `seed` - the initial state; equal seeds reproduce the same sequence.
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next uniform sample in `[0, 1)`.
    pub(crate) fn f(&mut self) -> f64 {
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
    pub(crate) fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}
