// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Double-double (`hi + lo`) arithmetic.
//!
//! A value is represented by an unevaluated pair `(hi, lo)` whose exact value is
//! `hi + lo`, giving ~106 bits of precision. Add/sub/mul are built on the
//! [`error_free`](super::error_free) transformations, which carry the round-off
//! of each `f64` operation so it is not lost between steps.

use super::error_free::{fast_two_sum, two_product, two_sum};

/// A double-double number represented as an unevaluated pair `(hi, lo)` whose
/// exact value is `hi + lo`.
pub(crate) type Dd = (f64, f64);

/// Exact difference `a - b` as a double-double.
#[inline]
pub(crate) fn dd_diff(a: f64, b: f64) -> Dd {
    two_sum(a, -b)
}

#[inline]
pub(crate) fn dd_add(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum(a.0, b.0);
    let e = e + (a.1 + b.1);
    fast_two_sum(s, e)
}

#[inline]
pub(crate) fn dd_sub(a: Dd, b: Dd) -> Dd {
    dd_add(a, (-b.0, -b.1))
}

#[inline]
pub(crate) fn dd_mul(a: Dd, b: Dd) -> Dd {
    let (p, e) = two_product(a.0, b.0);
    let e = e + (a.0 * b.1 + a.1 * b.0);
    fast_two_sum(p, e)
}

/// Exact square `a * a` as a double-double. Bit-identical to `dd_mul(a, a)` but
/// exploits the symmetry of the cross term (`a.0*a.1 + a.1*a.0 == 2*a.0*a.1`,
/// exactly) to save a multiply.
#[inline]
pub(crate) fn dd_sqr(a: Dd) -> Dd {
    let (p, e) = two_product(a.0, a.0);
    let e = e + 2.0 * (a.0 * a.1);
    fast_two_sum(p, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Double-double multiply keeps the low bits a plain product would drop.
    #[test]
    fn dd_mul_carries_low_bits() {
        let x = dd_diff(1.0 + 2f64.powi(-52), 1.0); // exactly 2^-52
        let y = dd_diff(1.0 + 2f64.powi(-52), 1.0);
        let prod = dd_mul(x, y);
        let expected = 2f64.powi(-52) * 2f64.powi(-52);
        assert!((prod.0 + prod.1 - expected).abs() <= expected * 1e-15);
    }

    /// `dd_sqr` is bit-identical to the general `dd_mul(a, a)`.
    #[test]
    fn dd_sqr_matches_dd_mul() {
        for a in [
            dd_diff(3.0, 0.1),
            dd_diff(1.0 + 2f64.powi(-52), 1.0),
            dd_diff(-7.5, 2.25),
        ] {
            assert_eq!(dd_sqr(a), dd_mul(a, a));
        }
    }

    /// `dd_add` then `dd_sub` of the same operand round-trips.
    #[test]
    fn dd_add_sub_roundtrip() {
        let a = dd_diff(3.0, 0.1);
        let b = dd_diff(0.7, 2.0);
        let back = dd_sub(dd_add(a, b), b);
        assert!((back.0 + back.1 - (a.0 + a.1)).abs() < 1e-15);
    }
}
