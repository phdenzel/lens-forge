// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Marching-triangles iso-contours for [`MeshGrid`].
//!
//! The mesh's cached Delaunay triangulation is the natural cell complex; each
//! triangle is solved by [`iso_segment`], the
//! same crossing code the pixel grid uses.

use super::meshgrid::MeshGrid;
use crate::grid::contour::iso_segment;
use crate::grid::ContourSegment;

impl MeshGrid {
    /// The iso-contour `value = level` as unordered line segments (marching triangles).
    ///
    /// # Arguments
    /// * `level` - the contour value.
    pub fn contour(&self, level: f64) -> Vec<ContourSegment> {
        let triangles = self.triangles();
        let mut segments = Vec::with_capacity(triangles.len());
        for &[a, b, c] in triangles {
            let pts = [self.site(a), self.site(b), self.site(c)];
            let vals = [
                self.value(a) - level,
                self.value(b) - level,
                self.value(c) - level,
            ];
            if let Some(s) = iso_segment(pts, vals) {
                segments.push(s);
            }
        }
        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Point2D};
    use crate::rng::Lcg;

    #[test]
    fn contour_of_a_circle_lies_on_the_circle() {
        // f = x² + y² − r₀² on an unstructured mesh; the zero contour is the circle.
        let r0 = 0.6_f64;
        let mut rng = Lcg::new(0xC047);
        let bbox = BBox::from_lims(-1.0, 1.0, -1.0, 1.0);
        let sites: Vec<Point2D> = (0..2000)
            .map(|_| Point2D::new(1.9 * rng.f() - 0.95, 1.9 * rng.f() - 0.95))
            .collect();
        let m = MeshGrid::from_func(&sites, bbox, |x, y| x * x + y * y - r0 * r0).unwrap();
        let segments = m.contour(0.0);
        assert!(!segments.is_empty());
        // Linear interpolation between sites keeps the tolerance ~ site spacing.
        for seg in &segments {
            for p in seg {
                assert!((p.norm() - r0).abs() < 0.1, "off-circle point {p:?}");
            }
        }
    }

    #[test]
    fn no_contour_when_level_is_outside_the_range() {
        let bbox = BBox::from_lims(-1.0, 1.0, -1.0, 1.0);
        let sites: Vec<Point2D> = (0..64)
            .map(|i| {
                let t = i as f64 / 64.0;
                Point2D::new(1.6 * t - 0.8, 1.6 * ((7 * i) % 64) as f64 / 64.0 - 0.8)
            })
            .collect();
        let m = MeshGrid::from_func(&sites, bbox, |x, y| x * x + y * y).unwrap();
        assert!(m.contour(10.0).is_empty());
    }
}
