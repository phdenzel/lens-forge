// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! A discretized lens map θ ↔ β, queried for the images of a point source.
//!
//! A cloud of image-plane samples is connected into cells and each cell ray-traced
//! to the source plane; the source-plane cells tile a warped copy of the map. To
//! find the images of β, ask which source cells contain it — one image per cell —
//! and read the image-plane position back off the same cell. The two realizations
//! differ only in cell topology: [`MapTessellation`](super::tessellate) uses
//! Delaunay triangles, [`MapPixellation`](super::pixellate) uses lattice quads;
//! both reduce to the shared triangle test [`seed_if_contains`].

use crate::grid::Point2D;

/// Barycentric coordinates below `-CONTAIN_EPS` count as outside the triangle; the
/// small slack keeps images that fall on a shared cell edge (a neighbour catches
/// any that slip through, and duplicates are pruned downstream).
const CONTAIN_EPS: f64 = 1e-9;

/// A discretized lens map: query it for the image-plane seeds of a source point.
pub trait MapDiscretization {
    /// Image-plane seeds for every cell whose source image contains `beta` — one
    /// coarse seed per predicted image, to be polished by
    /// [`newton`](super::newton::newton).
    ///
    /// # Arguments
    /// * `beta` - the source-plane position to find images of.
    fn images_of(&self, beta: Point2D) -> Vec<Point2D>;
}

/// If `beta` lies in the source triangle `source`, return the image-plane seed:
/// the same barycentric combination of the corresponding image triangle `image`.
///
/// Affine-invariant, so it also catches folded (inverted) source triangles — the
/// negative-parity images near critical curves.
///
/// # Arguments
/// * `beta` - the source-plane query point.
/// * `source` - the source-plane triangle β_a, β_b, β_c.
/// * `image` - the image-plane triangle θ_a, θ_b, θ_c, in the same vertex order.
pub(crate) fn seed_if_contains(
    beta: Point2D,
    source: [Point2D; 3],
    image: [Point2D; 3],
) -> Option<Point2D> {
    let (u, v, w) = beta.barycentric(&source[0], &source[1], &source[2]);
    if u < -CONTAIN_EPS || v < -CONTAIN_EPS || w < -CONTAIN_EPS {
        return None;
    }
    Some(Point2D::new(
        u * image[0].x + v * image[1].x + w * image[2].x,
        u * image[0].y + v * image[1].y + w * image[2].y,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A right triangle in the image plane, sheared to a different one in the source
    // plane; the seed must be the barycentric-consistent image-plane point.
    const IMAGE: [Point2D; 3] = [
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 1.0, y: 0.0 },
        Point2D { x: 0.0, y: 1.0 },
    ];
    const SOURCE: [Point2D; 3] = [
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 2.0, y: 0.0 },
        Point2D { x: 0.0, y: 2.0 },
    ];

    #[test]
    fn interior_point_maps_back_by_barycentric() {
        // β = (1, 1) is the source centroid-ish point ½·b + ½·c; its image seed is
        // ½·(1,0) + ½·(0,1) = (0.5, 0.5).
        let seed = seed_if_contains(Point2D::new(1.0, 1.0), SOURCE, IMAGE).unwrap();
        assert!((seed - Point2D::new(0.5, 0.5)).norm() < 1e-12);
    }

    #[test]
    fn exterior_point_is_rejected() {
        assert!(seed_if_contains(Point2D::new(3.0, 3.0), SOURCE, IMAGE).is_none());
    }

    #[test]
    fn folded_triangle_still_contains() {
        // Mirror the source triangle (negative area): containment is orientation-free.
        let folded = [SOURCE[0], SOURCE[2], SOURCE[1]];
        assert!(seed_if_contains(Point2D::new(0.5, 0.5), folded, IMAGE).is_some());
    }
}
