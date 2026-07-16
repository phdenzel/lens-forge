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
use crate::utils::double_double::{dd_add, dd_diff, dd_mul, dd_sqr, dd_sub};

/// Machine epsilon for `f64` ($2^{-53}$).
const EPS: f64 = 1.1102230246251565e-16;
/// Relative error bound for the 2x2 orientation determinant (Shewchuk).
const ORIENT_BOUND: f64 = (3.0 + 16.0 * EPS) * EPS;
/// Relative error bound for the in-circle determinant (Shewchuk).
const INCIRCLE_BOUND: f64 = (10.0 + 96.0 * EPS) * EPS;

/* ============ Predicates ================================================= */

/// Orientation of `c` relative to the directed line `a→b`.
///
/// Returns a value with the sign of the signed area of triangle `(a, b, c)`:
/// `> 0` if `c` lies to the left (counter-clockwise winding), `< 0` to the
/// right (clockwise), and `0` if the three points are collinear. Only the sign
/// is meaningful.
pub(crate) fn orient2d(a: Point2D, b: Point2D, c: Point2D) -> f64 {
    let ac = a - c;
    let bc = b - c;

    let detleft = ac.x * bc.y;
    let detright = ac.y * bc.x;
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
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;

    let bc = bd.cross(&cd);
    let ca = cd.cross(&ad);
    let ab = ad.cross(&bd);

    let alift = ad.norm_squared();
    let blift = bd.norm_squared();
    let clift = cd.norm_squared();

    let det = alift * bc + blift * ca + clift * ab;

    let permanent = (bd.x.abs() * cd.y.abs() + cd.x.abs() * bd.y.abs()) * alift
        + (cd.x.abs() * ad.y.abs() + ad.x.abs() * cd.y.abs()) * blift
        + (ad.x.abs() * bd.y.abs() + bd.x.abs() * ad.y.abs()) * clift;
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

    let alift = dd_add(dd_sqr(adx), dd_sqr(ady));
    let blift = dd_add(dd_sqr(bdx), dd_sqr(bdy));
    let clift = dd_add(dd_sqr(cdx), dd_sqr(cdy));

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
