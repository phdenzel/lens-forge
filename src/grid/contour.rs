// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Iso-contour extraction shared by the field representations.
//!
//! [`FieldOps::contour`](crate::grid::FieldOps::contour) marches each grid's
//! natural cells, but both back-ends reduce to the same linear zero-crossing of a
//! triangle, so that per-triangle solver lives here. [`PixelGrid`] splits each cell
//! into two triangles (a watertight, saddle-free marching squares), [`MeshGrid`]
//! marches the Delaunay triangles directly.
//!
//! [`PixelGrid`]: crate::grid::PixelGrid
//! [`MeshGrid`]: crate::grid::MeshGrid

use crate::grid::Point2D;

/// A single line segment of an extracted iso-contour (its two endpoints).
pub type ContourSegment = [Point2D; 2];

/// The zero-crossing segment of the linear field interpolated over one triangle.
///
/// # Arguments
/// * `pts` - the triangle corners.
/// * `vals` - the field at each corner, already offset by the contour level.
///
/// Returns the crossing segment, or `None` when all corners lie on one side (a
/// linear field crosses a triangle on exactly zero or two edges).
pub(crate) fn iso_segment(pts: [Point2D; 3], vals: [f64; 3]) -> Option<ContourSegment> {
    let mut crossings = [Point2D::origin(); 2];
    let mut n = 0;
    for &(a, b) in &[(0usize, 1usize), (1, 2), (2, 0)] {
        let (va, vb) = (vals[a], vals[b]);
        // Strictly opposite sides of the level (>= 0 counts as "above").
        if (va < 0.0) != (vb < 0.0) {
            let t = va / (va - vb); // va + t·(vb − va) = 0
            if n < 2 {
                crossings[n] = pts[a].lerp(&pts[b], t);
            }
            n += 1;
        }
    }
    if n == 2 { Some(crossings) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_crossing_when_all_above_or_below() {
        let pts = [
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        assert!(iso_segment(pts, [1.0, 2.0, 0.5]).is_none());
        assert!(iso_segment(pts, [-1.0, -2.0, -0.5]).is_none());
    }

    #[test]
    fn single_negative_corner_cuts_its_two_edges() {
        let pts = [
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(0.0, 2.0),
        ];
        // Corner 0 below, 1 and 2 above: cut the (0,1) and (2,0) edges at their
        // midpoints (equal-magnitude values).
        let seg = iso_segment(pts, [-1.0, 1.0, 1.0]).unwrap();
        let mids = [Point2D::new(1.0, 0.0), Point2D::new(0.0, 1.0)];
        for endpoint in seg {
            assert!(mids.iter().any(|m| (endpoint - *m).norm() < 1e-12));
        }
    }
}
