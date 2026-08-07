// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Free-space log-potential kernel of a uniform Voronoi (convex-polygon) cell.
//!
//! The unstructured counterpart to the square-pixel kernel
//! ([`square_log_kernel`](crate::grid::square_log_kernel)): a uniform-value convex
//! polygon contributes the log integral $\psi = \frac{1}{\pi}\int_\text{cell}
//! \ln|\theta-\theta'|\,d^2\theta'$, reduced by Green's theorem (the 2D
//! Talwani–Hubbert construction) to a sum of elementary per-edge line integrals of
//! $\ln r$ and $R/r^2$. Same $1/\pi$ normalization as `fastsum`, so the two agree in
//! the far field, and it reproduces the square kernel on a square cell.

use crate::grid::{MeshGrid, Point2D};
use std::f64::consts::FRAC_1_PI;

/// `1/(2π)` — the symmetrized Hessian off-diagonal normalization.
const FRAC_1_2PI: f64 = 0.5 * FRAC_1_PI;

impl MeshGrid {
    /// Free-space log-potential kernel of Voronoi cell `i` at `theta`, per unit cell
    /// value: `[ψ, ψ_x, ψ_y, ψ_xx, ψ_xy, ψ_yy]` with the `1/π` free-space
    /// normalization ($\nabla^2\psi = 2\cdot\text{value}$).
    ///
    /// # Arguments
    /// * `i` - cell index.
    /// * `theta` - evaluation point.
    pub fn log_kernel(&self, i: usize, theta: Point2D) -> [f64; 6] {
        polygon_log_kernel(self.cell_polygon(i), theta)
    }
}

/// Free-space log-potential kernel of one uniform convex polygon (CCW `verts`) at
/// `theta`, per unit value: `[ψ, ψ_x, ψ_y, ψ_xx, ψ_xy, ψ_yy]` with the `1/π`
/// normalization — the Green's-theorem sum of per-edge line integrals. Returns zeros
/// for a degenerate polygon (`< 3` vertices).
///
/// # Arguments
/// * `verts` - convex-polygon vertices, counter-clockwise.
/// * `theta` - evaluation point.
pub fn polygon_log_kernel(verts: &[Point2D], theta: Point2D) -> [f64; 6] {
    let n = verts.len();
    if n < 3 {
        return [0.0; 6];
    }
    let mut acc = [0.0f64; 6];
    for k in 0..n {
        let e = edge_kernel(verts[k], verts[(k + 1) % n], theta);
        for j in 0..6 {
            acc[j] += e[j];
        }
    }
    acc
}

/// Contribution `[ψ, ∂x, ∂y, ∂xx, ∂xy, ∂yy]` of one CCW edge `a → b` to the
/// polygon log-potential kernel at `theta` (the counterpart of `pixel_kernel` on the
/// pixel side). The Hessian off-diagonal is symmetrized here: the per-edge
/// antisymmetric part `½ln(r_b/r_a)` telescopes to zero around the closed loop.
fn edge_kernel(a: Point2D, b: Point2D, theta: Point2D) -> [f64; 6] {
    let e = b - a;
    let ell = e.norm();
    if ell == 0.0 {
        return [0.0; 6]; // degenerate edge
    }
    let (tx, ty) = (e.x / ell, e.y / ell); // unit tangent
    let (nx, ny) = (ty, -tx); // outward normal (CCW → right of travel)
    let (da, db) = (a - theta, b - theta);
    let (ra, rb) = (da.norm(), db.norm());
    let q = da.x * nx + da.y * ny; // signed perpendicular offset of the edge
    let (ua, ub) = (da.x * tx + da.y * ty, db.x * tx + db.y * ty);
    let phi = da.cross(&db).atan2(da.dot(&db)); // signed subtended angle
    let ln_ra = if ra > 0.0 { ra.ln() } else { 0.0 };
    let ln_rb = if rb > 0.0 { rb.ln() } else { 0.0 };
    // Edge integral of ln r; and L = ln(r_b/r_a) for the Hessian.
    let j = ub * ln_rb - ua * ln_ra - ell + q * phi;
    let l = ln_rb - ln_ra;
    let (gx, gy) = (nx * phi + tx * l, ny * phi + ty * l);
    [
        FRAC_1_PI * q * (0.5 * j - 0.25 * ell),
        -FRAC_1_PI * nx * j,
        -FRAC_1_PI * ny * j,
        FRAC_1_PI * nx * gx,
        FRAC_1_2PI * (nx * gy + ny * gx),
        FRAC_1_PI * ny * gy,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, FieldArray1D, square_log_kernel};
    use std::f64::consts::PI;

    /// A unit square, counter-clockwise, spanning [-0.5, 0.5]².
    fn unit_square() -> [Point2D; 4] {
        [
            Point2D::new(-0.5, -0.5),
            Point2D::new(0.5, -0.5),
            Point2D::new(0.5, 0.5),
            Point2D::new(-0.5, 0.5),
        ]
    }

    /// ½·tr H = value at an interior point (∇²ψ = 2κ), 0 outside.
    #[test]
    fn trace_recovers_value_inside_and_vanishes_outside() {
        let sq = unit_square();
        let inside = polygon_log_kernel(&sq, Point2D::new(0.1, -0.2));
        assert!((0.5 * (inside[3] + inside[5]) - 1.0).abs() < 1e-12);
        let outside = polygon_log_kernel(&sq, Point2D::new(2.0, 1.3));
        assert!((0.5 * (outside[3] + outside[5])).abs() < 1e-12);
    }

    /// Far away, a unit-area cell matches its point-mass monopole `α = (m/π)θ/r²`.
    #[test]
    fn far_field_matches_monopole() {
        let sq = unit_square(); // area 1, centred at origin
        let theta = Point2D::new(30.0, 20.0);
        let k = polygon_log_kernel(&sq, theta);
        let r2 = theta.x * theta.x + theta.y * theta.y;
        let (ax, ay) = (theta.x / PI / r2, theta.y / PI / r2);
        assert!((k[1] - ax).abs() < 1e-5 && (k[2] - ay).abs() < 1e-5, "α=({},{})", k[1], k[2]);
    }

    /// Cross-validation: the polygon kernel on a square reproduces the independently
    /// derived square-pixel kernel to round-off (off the boundary, where the Hessian
    /// jumps and the pointwise value is ambiguous).
    #[test]
    fn square_polygon_matches_pixel_kernel() {
        let sq = unit_square();
        for &p in &[
            Point2D::new(0.13, -0.07), // interior
            Point2D::new(1.7, 0.9),    // exterior
            Point2D::new(-2.0, 3.0),   // far
        ] {
            let poly = polygon_log_kernel(&sq, p);
            let pix = square_log_kernel(Point2D::new(0.0, 0.0), 0.5, 0.5, p);
            for c in 0..6 {
                assert!((poly[c] - pix[c]).abs() < 1e-9, "comp {c} at {p:?}: {} vs {}", poly[c], pix[c]);
            }
        }
    }

    /// On an edge/vertex the atan2/ln guards keep every component finite.
    #[test]
    fn edges_and_vertices_stay_finite() {
        let sq = unit_square();
        for &p in &[
            Point2D::new(0.5, 0.2),  // on an edge
            Point2D::new(0.5, 0.5),  // at a vertex
            Point2D::new(-0.5, 0.1), // on the opposite edge
        ] {
            assert!(polygon_log_kernel(&sq, p).iter().all(|v| v.is_finite()), "{p:?}");
        }
    }

    /// A degenerate polygon (< 3 vertices) contributes nothing.
    #[test]
    fn degenerate_polygon_is_zero() {
        assert_eq!(polygon_log_kernel(&[Point2D::new(0.0, 0.0)], Point2D::new(1.0, 1.0)), [0.0; 6]);
    }

    /// A repeated vertex (zero-length edge) is skipped, leaving the result unchanged.
    #[test]
    fn zero_length_edge_is_skipped() {
        let p = Point2D::new(0.13, -0.07);
        let clean = polygon_log_kernel(&unit_square(), p);
        let with_dup = polygon_log_kernel(
            &[
                Point2D::new(-0.5, -0.5),
                Point2D::new(0.5, -0.5),
                Point2D::new(0.5, -0.5), // duplicate → zero-length edge
                Point2D::new(0.5, 0.5),
                Point2D::new(-0.5, 0.5),
            ],
            p,
        );
        for c in 0..6 {
            assert!((clean[c] - with_dup[c]).abs() < 1e-12, "comp {c}");
        }
    }

    /// `MeshGrid::log_kernel` drives the polygon kernel from a real cell; ½·tr H = κ
    /// inside the cell containing the query.
    #[test]
    fn meshgrid_log_kernel_recovers_kappa_inside() {
        let sites = [
            Point2D::new(-1.0, -1.0),
            Point2D::new(1.0, -1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(-1.0, 1.0),
            Point2D::new(0.0, 0.0),
        ];
        let bbox = BBox::from_lims(-3.0, 3.0, -3.0, 3.0);
        let mg = MeshGrid::new(&sites, bbox, FieldArray1D::from_vec(vec![1.0; 5])).unwrap();
        // The central cell (site 4) contains the origin; ½·tr H there is its κ = 1.
        let k = mg.log_kernel(4, Point2D::new(0.05, -0.03));
        assert!((0.5 * (k[3] + k[5]) - 1.0).abs() < 1e-9, "½trH = {}", 0.5 * (k[3] + k[5]));
    }
}
