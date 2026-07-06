// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Robust geometric predicates for Delaunay triangulation.
//!
//! [`orient2d`] and [`incircle`] each evaluate a determinant. They first try a
//! plain `f64` computation, and only when the sign is numerically uncertain (the
//! magnitude is within a relative error bound of zero) fall back to an
//! error-free-transformation (double-double) recomputation using fused
//! multiply-add. This keeps the common case cheap while staying reliable on the
//! near-collinear / near-cocircular inputs a naive determinant would
//! misclassify — the failure mode that otherwise corrupts the triangulation.

use crate::grid::Point2D;

/// Machine epsilon for `f64` ($2^{-53}$).
const EPS: f64 = 1.1102230246251565e-16;
/// Relative error bound for the 2×2 orientation determinant (Shewchuk).
const ORIENT_BOUND: f64 = (3.0 + 16.0 * EPS) * EPS;
/// Relative error bound for the in-circle determinant (Shewchuk).
const INCIRCLE_BOUND: f64 = (10.0 + 96.0 * EPS) * EPS;

/* ============ Error-free transformations ================================= */

/// Exact sum: returns `(s, e)` with `s = fl(a + b)` and `a + b = s + e`.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bv = s - a;
    let e = (a - (s - bv)) + (b - bv);
    (s, e)
}

/// Exact sum when `|a| >= |b|` (Dekker): `a + b = s + e`.
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// Exact product via FMA: returns `(p, e)` with `p = fl(a * b)` and `a*b = p+e`.
#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/* ============ Double-double arithmetic (value = hi + lo) ================== */

type Dd = (f64, f64);

/// Exact difference `a - b` as a double-double.
#[inline]
fn dd_diff(a: f64, b: f64) -> Dd {
    two_sum(a, -b)
}

#[inline]
fn dd_add(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum(a.0, b.0);
    let e = e + (a.1 + b.1);
    fast_two_sum(s, e)
}

#[inline]
fn dd_sub(a: Dd, b: Dd) -> Dd {
    dd_add(a, (-b.0, -b.1))
}

#[inline]
fn dd_mul(a: Dd, b: Dd) -> Dd {
    let (p, e) = two_product(a.0, b.0);
    let e = e + (a.0 * b.1 + a.1 * b.0);
    fast_two_sum(p, e)
}

/* ============ Predicates ================================================= */

/// Orientation of `c` relative to the directed line `a → b`.
///
/// Returns a value with the sign of the signed area of triangle `(a, b, c)`:
/// `> 0` if `c` lies to the left (counter-clockwise winding), `< 0` to the
/// right (clockwise), and `0` if the three points are collinear. Only the sign
/// is meaningful.
pub(crate) fn orient2d(a: Point2D, b: Point2D, c: Point2D) -> f64 {
    let acx = a.x - c.x;
    let bcx = b.x - c.x;
    let acy = a.y - c.y;
    let bcy = b.y - c.y;

    let detleft = acx * bcy;
    let detright = acy * bcx;
    let det = detleft - detright;

    let errbound = ORIENT_BOUND * (detleft.abs() + detright.abs());
    if det.abs() > errbound {
        return det;
    }

    // Uncertain sign: recompute exactly in double-double.
    let acx = dd_diff(a.x, c.x);
    let bcx = dd_diff(b.x, c.x);
    let acy = dd_diff(a.y, c.y);
    let bcy = dd_diff(b.y, c.y);
    let det = dd_sub(dd_mul(acx, bcy), dd_mul(acy, bcx));
    det.0 + det.1
}

/// In-circle test: is `d` inside the circumcircle of triangle `(a, b, c)`?
///
/// Assumes `(a, b, c)` is counter-clockwise ([`orient2d`] `> 0`). Returns a
/// value that is `> 0` if `d` is strictly inside the circumcircle, `< 0` if
/// outside, and `0` if the four points are cocircular. Only the sign is
/// meaningful.
pub(crate) fn incircle(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> f64 {
    let adx = a.x - d.x;
    let ady = a.y - d.y;
    let bdx = b.x - d.x;
    let bdy = b.y - d.y;
    let cdx = c.x - d.x;
    let cdy = c.y - d.y;

    let bc = bdx * cdy - cdx * bdy;
    let ca = cdx * ady - adx * cdy;
    let ab = adx * bdy - bdx * ady;

    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    let det = alift * bc + blift * ca + clift * ab;

    let permanent = (bdx.abs() * cdy.abs() + cdx.abs() * bdy.abs()) * alift
        + (cdx.abs() * ady.abs() + adx.abs() * cdy.abs()) * blift
        + (adx.abs() * bdy.abs() + bdx.abs() * ady.abs()) * clift;
    let errbound = INCIRCLE_BOUND * permanent;
    if det.abs() > errbound {
        return det;
    }

    // Uncertain sign: recompute exactly in double-double.
    let adx = dd_diff(a.x, d.x);
    let ady = dd_diff(a.y, d.y);
    let bdx = dd_diff(b.x, d.x);
    let bdy = dd_diff(b.y, d.y);
    let cdx = dd_diff(c.x, d.x);
    let cdy = dd_diff(c.y, d.y);

    let bc = dd_sub(dd_mul(bdx, cdy), dd_mul(cdx, bdy));
    let ca = dd_sub(dd_mul(cdx, ady), dd_mul(adx, cdy));
    let ab = dd_sub(dd_mul(adx, bdy), dd_mul(bdx, ady));

    let alift = dd_add(dd_mul(adx, adx), dd_mul(ady, ady));
    let blift = dd_add(dd_mul(bdx, bdx), dd_mul(bdy, bdy));
    let clift = dd_add(dd_mul(cdx, cdx), dd_mul(cdy, cdy));

    let det = dd_add(dd_add(dd_mul(alift, bc), dd_mul(blift, ca)), dd_mul(clift, ab));
    det.0 + det.1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    #[test]
    fn orient2d_ccw_cw_collinear() {
        // Counter-clockwise.
        assert!(orient2d(p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0)) > 0.0);
        // Clockwise.
        assert!(orient2d(p(0.0, 0.0), p(0.0, 1.0), p(1.0, 0.0)) < 0.0);
        // Exactly collinear.
        assert_eq!(orient2d(p(0.0, 0.0), p(1.0, 1.0), p(2.0, 2.0)), 0.0);
    }

    #[test]
    fn orient2d_near_collinear_uses_exact_path() {
        // c is a hair above the line y = x through a and b: must be ccw (> 0),
        // a magnitude where a naive determinant is at risk of cancellation.
        let a = p(0.0, 0.0);
        let b = p(1.0, 1.0);
        let c = p(0.5, 0.5 + 1e-15);
        assert!(orient2d(a, b, c) > 0.0);
        // Mirror below the line: clockwise.
        let c = p(0.5, 0.5 - 1e-15);
        assert!(orient2d(a, b, c) < 0.0);
    }

    #[test]
    fn incircle_inside_outside_on() {
        // Unit right triangle, ccw; circumcircle centre (0.5, 0.5), r = √0.5.
        let (a, b, c) = (p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0));
        assert!(orient2d(a, b, c) > 0.0, "triangle must be ccw");

        // Centre point: strictly inside.
        assert!(incircle(a, b, c, p(0.5, 0.5)) > 0.0);
        // Far corner: outside.
        assert!(incircle(a, b, c, p(2.0, 2.0)) < 0.0);
        // Fourth square corner (1, 1) lies exactly on the circumcircle.
        assert_eq!(incircle(a, b, c, p(1.0, 1.0)), 0.0);
    }

    #[test]
    fn incircle_near_cocircular_uses_exact_path() {
        let (a, b, c) = (p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0));
        // Just inside / outside the on-circle point (1, 1).
        assert!(incircle(a, b, c, p(1.0, 1.0 - 1e-12)) > 0.0);
        assert!(incircle(a, b, c, p(1.0, 1.0 + 1e-12)) < 0.0);
    }
}
