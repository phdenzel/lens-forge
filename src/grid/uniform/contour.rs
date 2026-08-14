// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Marching-squares iso-contours for [`PixelGrid`].
//!
//! Each grid cell (four adjacent pixel centres) is split into two triangles and
//! solved by [`iso_segment`], giving a
//! watertight, saddle-free contour that shares the mesh's crossing code.

use super::pixelgrid::PixelGrid;
use crate::grid::contour::iso_segment;
use crate::grid::ContourSegment;

impl PixelGrid {
    /// The iso-contour `value = level` as unordered line segments (marching squares).
    ///
    /// # Arguments
    /// * `level` - the contour value.
    pub fn contour(&self, level: f64) -> Vec<ContourSegment> {
        let (nx, ny) = (self.grid.nx, self.grid.ny);
        let mut segments = Vec::new();
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx.saturating_sub(1) {
                let p00 = self.grid.pixel_center(i, j);
                let p10 = self.grid.pixel_center(i + 1, j);
                let p01 = self.grid.pixel_center(i, j + 1);
                let p11 = self.grid.pixel_center(i + 1, j + 1);
                let v00 = self.get(i, j) - level;
                let v10 = self.get(i + 1, j) - level;
                let v01 = self.get(i, j + 1) - level;
                let v11 = self.get(i + 1, j + 1) - level;
                // Split the cell along the (p00, p11) diagonal.
                if let Some(s) = iso_segment([p00, p10, p11], [v00, v10, v11]) {
                    segments.push(s);
                }
                if let Some(s) = iso_segment([p00, p11, p01], [v00, v11, v01]) {
                    segments.push(s);
                }
            }
        }
        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;

    #[test]
    fn contour_of_a_circle_lies_on_the_circle() {
        // f = x² + y² − r₀²; the zero contour is the circle of radius r₀.
        let r0 = 0.6_f64;
        let g = PixelGrid::from_func_sq(128, 1.0, |x, y| x * x + y * y - r0 * r0);
        let segments = g.contour(0.0);
        assert!(!segments.is_empty());
        // Endpoints sit on the circle to within a pixel.
        let dx = g.grid.dx();
        for seg in &segments {
            for p in seg {
                assert!((p.norm() - r0).abs() < 2.0 * dx, "off-circle point {p:?}");
            }
        }
    }

    #[test]
    fn no_contour_when_level_is_outside_the_range() {
        let g = PixelGrid::from_func_sq(32, 1.0, |x, y| x * x + y * y);
        // The field is in [0, 2]; a level of 10 is never crossed.
        assert!(g.contour(10.0).is_empty());
    }

    #[test]
    fn contour_of_a_plane_is_a_straight_line() {
        // 2x − 3y + 5 = level ⇒ endpoints satisfy the linear equation.
        let grid = Grid2D::square(48, 1.0).unwrap();
        let g = PixelGrid::from_func(grid, |x, y| 2.0 * x - 3.0 * y + 5.0);
        let segments = g.contour(5.0);
        assert!(!segments.is_empty());
        for seg in &segments {
            for p in seg {
                assert!((2.0 * p.x - 3.0 * p.y).abs() < 1e-9, "off-line point {p:?}");
            }
        }
    }
}
