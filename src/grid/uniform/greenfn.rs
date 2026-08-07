// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Free-space log-potential kernel of a uniform square pixel.
//!
//! The exact real-space counterpart to the spectral Poisson solve, and the
//! extended-source counterpart to `fastsum`'s point-mass monopole: a uniform-value
//! square cell contributes the log integral $\psi = \frac{1}{\pi}\int_\text{cell}
//! \ln|\theta-\theta'|\,d^2\theta'$, evaluated in closed form as a four-corner
//! difference of the mixed antiderivative (and likewise its gradient and Hessian).
//! Same $1/\pi$ normalization as `fastsum`, so the two agree in the far field.

use crate::grid::{PixelGrid, Point2D};
use std::f64::consts::FRAC_1_PI;

/// `1/(2π)`
const FRAC_1_2PI: f64 = 0.5 * FRAC_1_PI;

impl PixelGrid {
    /// Four-corner difference of the uniform square-pixel log-potential kernel for
    /// cell `i` at `theta`, per unit cell value: `[ψ, ψ_x, ψ_y, ψ_xx, ψ_xy, ψ_yy]`
    /// with the `1/π` free-space normalization ($\nabla^2\psi = 2\cdot\text{value}$).
    ///
    /// # Arguments
    /// * `i` - flat (row-major) cell index.
    /// * `theta` - evaluation point.
    pub fn log_kernel(&self, i: usize, theta: Point2D) -> [f64; 6] {
        let (ix, jy) = self.grid.unflatten_index(i);
        let c = self.grid.pixel_center(ix, jy);
        square_log_kernel(c, 0.5 * self.dx(), 0.5 * self.dy(), theta)
    }
}

/// Free-space log-potential kernel of one uniform square cell (centre `center`,
/// half-widths `hx`, `hy`) at `theta`, per unit value: `[ψ, ψ_x, ψ_y, ψ_xx, ψ_xy,
/// ψ_yy]` with the `1/π` normalization — the four-corner difference of the mixed
/// antiderivative.
///
/// # Arguments
/// * `center` - cell centre.
/// * `hx` - half-width in x.
/// * `hy` - half-width in y.
/// * `theta` - evaluation point.
pub fn square_log_kernel(center: Point2D, hx: f64, hy: f64, theta: Point2D) -> [f64; 6] {
    let (dx, dy) = (theta.x - center.x, theta.y - center.y);
    // Relative coordinates θ − pixel corner, on the left/right and bottom/top edges.
    let (left, right) = (dx + hx, dx - hx);
    let (bottom, top) = (dy + hy, dy - hy);
    let top_right = pixel_kernel(right, top);
    let bottom_right = pixel_kernel(right, bottom);
    let top_left = pixel_kernel(left, top);
    let bottom_left = pixel_kernel(left, bottom);
    std::array::from_fn(|j| top_right[j] - bottom_right[j] - top_left[j] + bottom_left[j])
}

/// Mixed antiderivatives `[ψ̃, ∂x, ∂y, ∂xx, ∂xy, ∂yy]` of the square-pixel log
/// integral at relative coordinates `(x, y)`, whose pixel integral is the
/// four-corner difference of each.
///
/// Uses the principal `(y/x).atan()` (not `atan2`): the builtin already gives `±π/2`
/// for `x = 0` (via `±∞`), so only the `0/0` corner (`r² = 0`, also the `ln`
/// singularity) needs guarding, taken as zero.
fn pixel_kernel(x: f64, y: f64) -> [f64; 6] {
    let r2 = x * x + y * y;
    if r2 == 0.0 {
        return [0.0; 6];
    }
    let ln = r2.ln();
    let atan_yx = (y / x).atan();
    let atan_xy = (x / y).atan();
    [
        FRAC_1_2PI * (x * x * atan_yx + y * y * atan_xy + x * y * ln - 3.0 * x * y),
        FRAC_1_PI * (x * atan_yx + 0.5 * y * ln - y),
        FRAC_1_PI * (y * atan_xy + 0.5 * x * ln - x),
        FRAC_1_PI * atan_yx,
        FRAC_1_2PI * ln,
        FRAC_1_PI * atan_xy,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use std::f64::consts::PI;

    /// A single uniform cell has ½·tr H = value at an interior point (∇²ψ = 2κ).
    #[test]
    fn trace_recovers_value_inside_a_cell() {
        let grid = Grid2D::square(1, 0.5).unwrap(); // one pixel spanning [-0.5, 0.5]²
        let pg = PixelGrid::constant(grid, 1.0);
        let k = pg.log_kernel(0, Point2D::new(0.1, -0.2));
        assert!((0.5 * (k[3] + k[5]) - 1.0).abs() < 1e-12, "tr/2 = {}", 0.5 * (k[3] + k[5]));
    }

    /// Outside the cell the source-free Laplacian vanishes: ½·tr H = 0.
    #[test]
    fn trace_vanishes_outside_a_cell() {
        let grid = Grid2D::square(1, 0.5).unwrap();
        let pg = PixelGrid::constant(grid, 1.0);
        let k = pg.log_kernel(0, Point2D::new(2.0, 1.3));
        assert!((0.5 * (k[3] + k[5])).abs() < 1e-12);
    }

    /// Far away the exact cell integral matches its point-mass monopole `m = area`:
    /// α = (m/π)·(θ−c)/|θ−c|².
    #[test]
    fn far_field_matches_monopole() {
        let grid = Grid2D::square(1, 0.5).unwrap();
        let pg = PixelGrid::constant(grid, 1.0);
        let theta = Point2D::new(30.0, 20.0);
        let k = pg.log_kernel(0, theta);
        let (m, r2) = (pg.pixel_area(), theta.x * theta.x + theta.y * theta.y);
        let (ax, ay) = (m / PI * theta.x / r2, m / PI * theta.y / r2);
        assert!((k[1] - ax).abs() < 1e-6 && (k[2] - ay).abs() < 1e-6, "α=({},{})", k[1], k[2]);
    }

    /// On a pixel edge/corner the `b = 0` arctan branches and the `r² = 0` log
    /// guard are hit; every component must stay finite.
    #[test]
    fn edges_and_corners_stay_finite() {
        let grid = Grid2D::square(1, 0.5).unwrap(); // cell spanning [-0.5, 0.5]²
        let pg = PixelGrid::constant(grid, 1.0);
        for &p in &[
            Point2D::new(0.5, -0.3), // right edge: atan(∓, 0) → ±π/2
            Point2D::new(0.5, 0.5),  // corner: atan(0, 0) → 0 and ln(0) guard
            Point2D::new(-0.3, 0.5), // top edge
        ] {
            assert!(pg.log_kernel(0, p).iter().all(|v| v.is_finite()), "{p:?}");
        }
    }
}
