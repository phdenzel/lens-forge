// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Error-free transformations for `f64` arithmetic.
//!
//! Each primitive computes a floating-point `+`/`*` and *exactly* recovers the
//! rounding error, so the true result is the sum of the returned pair: for a
//! result `r`, `r = s + e` with `s = fl(op)` and `e` the discarded round-off.

/// Exact sum.
#[inline]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bv = s - a;
    let e = (a - (s - bv)) + (b - bv);
    (s, e)
}

/// Exact sum when `|a| >= |b|` (Dekker).
#[inline]
pub(crate) fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// Exact product via FMA.
#[inline]
pub(crate) fn two_product(a: f64, b: f64) -> (f64, f64) {
    let s = a * b;
    let e = a.mul_add(b, -s);
    (s, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `two_sum` recovers the round-off lost to catastrophic cancellation.
    #[test]
    fn two_sum_recovers_lost_bits() {
        let (s, e) = two_sum(1.0, 1e-20);
        // The naive sum drops `b` entirely; the error term recovers it exactly.
        assert_eq!(s, 1.0);
        assert_eq!(e, 1e-20);
    }

    /// `fast_two_sum` agrees with `two_sum` when `|a| >= |b|`.
    #[test]
    fn fast_two_sum_matches_two_sum() {
        let (s1, e1) = two_sum(3.0, 1e-16);
        let (s2, e2) = fast_two_sum(3.0, 1e-16);
        assert_eq!((s1, e1), (s2, e2));
    }

    /// `two_product` stores the low half of the product that `f64` cannot hold.
    #[test]
    fn two_product_recovers_low_bits() {
        let a = 1.0 + 2f64.powi(-30);
        let (p, e) = two_product(a, a);
        assert_eq!(p, a * a);
        // The dropped tail is non-zero and captured in `e`.
        assert!(e != 0.0);
    }
}
