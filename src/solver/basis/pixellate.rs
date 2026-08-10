// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The pixel (lattice-quad) discretization of the lens mapping.
//!
//! [`MapPixellation`] samples the image plane on a regular [`Grid2D`] lattice and
//! ray-traces the nodes to the source plane. Each lattice cell is a source-plane
//! quad; [`images_of`](MapDiscretization::images_of) splits it into two triangles
//! and reuses the shared containment test.

use super::discretization::{seed_if_contains, MapDiscretization};
use crate::grid::{Grid2D, Point2D};
use crate::solver::mapping::raytrace::RayTracer;
use crate::lens::LensModel;

/// A lattice-quad discretization of the lens map: `nx × ny` image-plane nodes,
/// their source-plane images, and the implicit quad connectivity between them.
pub struct MapPixellation {
    nx: usize,
    ny: usize,
    image_vertices: Vec<Point2D>,
    source_vertices: Vec<Point2D>,
}

impl MapPixellation {
    /// Build the pixellation of `lens` over the image-plane lattice `grid`:
    /// ray-trace each of its pixel centres to the source plane.
    ///
    /// # Arguments
    /// * `lens` - the lens whose map is discretized.
    /// * `grid` - the image-plane sampling lattice.
    pub fn build(lens: &dyn LensModel, grid: &Grid2D) -> Self {
        let image_vertices: Vec<Point2D> = grid.iter_centers().collect();
        let source_vertices = RayTracer::Direct.trace(lens, &image_vertices);
        Self { nx: grid.nx, ny: grid.ny, image_vertices, source_vertices }
    }

    /// The image-plane vertices θ_i, row-major (index = j·nx + i).
    pub fn image_vertices(&self) -> &[Point2D] {
        &self.image_vertices
    }

    /// The source-plane images β_i = θ_i − α(θ_i).
    pub fn source_vertices(&self) -> &[Point2D] {
        &self.source_vertices
    }

    /// Flat row-major index of lattice node `(i, j)`.
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Test one node-triangle against `beta`, returning an image seed if it is
    /// contained.
    fn seed(&self, beta: Point2D, a: usize, b: usize, c: usize) -> Option<Point2D> {
        let source = [self.source_vertices[a], self.source_vertices[b], self.source_vertices[c]];
        let image = [self.image_vertices[a], self.image_vertices[b], self.image_vertices[c]];
        seed_if_contains(beta, source, image)
    }
}

impl MapDiscretization for MapPixellation {
    fn images_of(&self, beta: Point2D) -> Vec<Point2D> {
        let mut seeds = Vec::new();
        for j in 0..self.ny.saturating_sub(1) {
            for i in 0..self.nx.saturating_sub(1) {
                // Split the cell (v00, v10, v11, v01) into two triangles.
                let (v00, v10) = (self.idx(i, j), self.idx(i + 1, j));
                let (v11, v01) = (self.idx(i + 1, j + 1), self.idx(i, j + 1));
                seeds.extend(self.seed(beta, v00, v10, v11));
                seeds.extend(self.seed(beta, v00, v11, v01));
            }
        }
        seeds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::SIS;
    use crate::solver::extrema::newton::{newton, NewtonConfig};

    /// Newton-polish the seeds and collapse them to distinct images.
    fn distinct_images(lens: &dyn LensModel, beta: Point2D, seeds: &[Point2D]) -> Vec<Point2D> {
        let mut images: Vec<Point2D> = Vec::new();
        for &seed in seeds {
            if let Some(theta) = newton(lens, beta, seed, NewtonConfig::default()) {
                if !images.iter().any(|p| (*p - theta).norm() < 1e-6) {
                    images.push(theta);
                }
            }
        }
        images
    }

    #[test]
    fn vertices_ray_trace_to_the_source_plane() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let grid = Grid2D::square(16, 2.0).unwrap();
        let pix = MapPixellation::build(&lens, &grid);
        for (theta, beta) in pix.image_vertices().iter().zip(pix.source_vertices()) {
            assert_eq!(*beta, lens.ray_trace(*theta));
        }
    }

    #[test]
    fn finds_both_sis_images_of_an_offset_source() {
        // SIS θ_E = 1, source at β = (0.3, 0): images at (1.3, 0) and (−0.7, 0).
        let lens = SIS::new(Point2D::origin(), 1.0);
        let grid = Grid2D::square(80, 2.0).unwrap();
        let pix = MapPixellation::build(&lens, &grid);
        let beta = Point2D::new(0.3, 0.0);
        let images = distinct_images(&lens, beta, &pix.images_of(beta));
        assert_eq!(images.len(), 2, "SIS off-axis source has two images");
        assert!(images.iter().any(|p| (*p - Point2D::new(1.3, 0.0)).norm() < 1e-6));
        assert!(images.iter().any(|p| (*p - Point2D::new(-0.7, 0.0)).norm() < 1e-6));
    }

    #[test]
    fn source_far_outside_has_no_images() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let grid = Grid2D::square(24, 2.0).unwrap();
        let pix = MapPixellation::build(&lens, &grid);
        assert!(pix.images_of(Point2D::new(50.0, 50.0)).is_empty());
    }
}
