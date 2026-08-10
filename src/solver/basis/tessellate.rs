// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The mesh (Delaunay-triangle) discretization of the lens mapping.
//!
//! [`MapTessellation`] Delaunay-triangulates a cloud of image-plane samples
//! (reusing [`MeshGrid`] geometry), ray-traces each vertex to the source plane,
//! and answers [`images_of`](MapDiscretization::images_of) by the shared triangle
//! containment test.

use super::discretization::{seed_if_contains, MapDiscretization};
use crate::grid::{BBox, MeshGrid, Point2D};
use crate::solver::mapping::raytrace::RayTracer;
use crate::lens::LensModel;
use crate::Result;

/// A Delaunay-triangle discretization of the lens map: image-plane vertices, their
/// source-plane images, and the triangles connecting them (shared indexing).
pub struct MapTessellation {
    image_vertices: Vec<Point2D>,
    source_vertices: Vec<Point2D>,
    triangles: Vec<[usize; 3]>,
}

impl MapTessellation {
    /// Build the tessellation of `lens` over the image-plane `samples`: Delaunay-
    /// triangulate them and ray-trace each vertex to the source plane.
    ///
    /// # Arguments
    /// * `lens` - the lens whose map is discretized.
    /// * `samples` - image-plane sample points to triangulate.
    /// * `bbox` - bounding box for the triangulation.
    pub fn build(lens: &dyn LensModel, samples: &[Point2D], bbox: BBox) -> Result<Self> {
        let mesh = MeshGrid::zeros(samples, bbox)?;
        // Trace the mesh's own sites so vertex indexing matches `triangles()`.
        let source_vertices = RayTracer::Direct.trace(lens, mesh.sites());
        Ok(Self {
            image_vertices: mesh.sites().to_vec(),
            source_vertices,
            triangles: mesh.triangles().to_vec(),
        })
    }

    /// The image-plane vertices θ_i.
    pub fn image_vertices(&self) -> &[Point2D] {
        &self.image_vertices
    }

    /// The source-plane images β_i = θ_i − α(θ_i).
    pub fn source_vertices(&self) -> &[Point2D] {
        &self.source_vertices
    }

    /// The triangles, as index triples into the vertex arrays.
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }
}

impl MapDiscretization for MapTessellation {
    fn images_of(&self, beta: Point2D) -> Vec<Point2D> {
        self.triangles
            .iter()
            .filter_map(|&[a, b, c]| {
                let source = [self.source_vertices[a], self.source_vertices[b], self.source_vertices[c]];
                let image = [self.image_vertices[a], self.image_vertices[b], self.image_vertices[c]];
                seed_if_contains(beta, source, image)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::SIS;
    use crate::rng::Lcg;
    use crate::solver::mapping::newton::{newton, NewtonConfig};

    /// A jittered `n × n` lattice of image-plane samples over `[-half, half]²`.
    fn jittered_samples(n: usize, half: f64) -> (Vec<Point2D>, BBox) {
        let mut rng = Lcg::new(0x5EED);
        let step = 2.0 * half / n as f64;
        let mut samples = Vec::new();
        for gy in 0..n {
            for gx in 0..n {
                let x = -half + (gx as f64 + 0.5) * step + 0.2 * step * (rng.f() - 0.5);
                let y = -half + (gy as f64 + 0.5) * step + 0.2 * step * (rng.f() - 0.5);
                samples.push(Point2D::new(x, y));
            }
        }
        (samples, BBox::from_lims(-half, half, -half, half))
    }

    /// Newton-polish the seeds and collapse them to distinct images.
    fn distinct_images(lens: &dyn LensModel, beta: Point2D, seeds: &[Point2D]) -> Vec<Point2D> {
        let mut images: Vec<Point2D> = Vec::new();
        for theta in seeds.iter().filter_map(|&seed| newton(lens, beta, seed, NewtonConfig::default())) {
            if !images.iter().any(|p| (*p - theta).norm() < 1e-6) {
                images.push(theta);
            }
        }
        images
    }

    #[test]
    fn vertices_ray_trace_to_the_source_plane() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let (samples, bbox) = jittered_samples(20, 2.0);
        let tess = MapTessellation::build(&lens, &samples, bbox).unwrap();
        for (theta, beta) in tess.image_vertices().iter().zip(tess.source_vertices()) {
            assert_eq!(*beta, lens.ray_trace(*theta));
        }
        // The triangulation indexes into the shared vertex arrays.
        let n = tess.image_vertices().len();
        assert!(!tess.triangles().is_empty());
        assert!(tess.triangles().iter().flatten().all(|&v| v < n));
    }

    #[test]
    fn finds_both_sis_images_of_an_offset_source() {
        // SIS θ_E = 1, source at β = (0.3, 0): images at (1.3, 0) and (−0.7, 0).
        let lens = SIS::new(Point2D::origin(), 1.0);
        let (samples, bbox) = jittered_samples(60, 2.0);
        let tess = MapTessellation::build(&lens, &samples, bbox).unwrap();
        let beta = Point2D::new(0.3, 0.0);
        let images = distinct_images(&lens, beta, &tess.images_of(beta));
        assert_eq!(images.len(), 2, "SIS off-axis source has two images");
        assert!(images.iter().any(|p| (*p - Point2D::new(1.3, 0.0)).norm() < 1e-6));
        assert!(images.iter().any(|p| (*p - Point2D::new(-0.7, 0.0)).norm() < 1e-6));
    }

    #[test]
    fn source_far_outside_has_no_images() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let (samples, bbox) = jittered_samples(30, 2.0);
        let tess = MapTessellation::build(&lens, &samples, bbox).unwrap();
        // A source well beyond the traced footprint is covered by no source triangle.
        assert!(tess.images_of(Point2D::new(50.0, 50.0)).is_empty());
    }
}
