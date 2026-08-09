// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Critical curves (where det A = 0) and their caustics.
//!
//! det A = λ_t·λ_r is sampled onto a field — a [`PixelGrid`](crate::grid::PixelGrid)
//! or a [`MeshGrid`](crate::grid::MeshGrid) — and its zero contour extracted with
//! [`contour`](crate::grid::FieldOps::contour) (marching squares / triangles). A
//! [`Stretch`] selector picks the full locus or the tangential (λ_t=0) / radial
//! (λ_r=0) branch of the eigen-split. The caustics are the chosen curve ray-traced
//! to the source plane. All are returned as unordered line segments.

use super::jacobian::det_a_map;
use super::raytrace::RayTracer;
use crate::grid::{ContourSegment, FieldOps, Point2D};
use crate::lens::LensModel;

/// Which branch of the critical locus det A = λ_t·λ_r to trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stretch {
    /// The full critical locus det A = 0 (either eigenvalue vanishes).
    Total,
    /// The tangential critical curve λ_t = 1 − κ − γ = 0 (tangential arcs).
    Tangential,
    /// The radial critical curve λ_r = 1 − κ + γ = 0 (radially stretched images).
    Radial,
}

/// The field whose zero contour is the `stretch` critical curve of `lens`.
fn stretch_field<F: FieldOps>(lens: &dyn LensModel, field: &F, stretch: Stretch) -> F {
    match stretch {
        Stretch::Total => det_a_map(lens, field),
        Stretch::Tangential => field.map_centers(|t| 1.0 - lens.convergence(t) - lens.shear(t)),
        Stretch::Radial => field.map_centers(|t| 1.0 - lens.convergence(t) + lens.shear(t)),
    }
}

/// The critical curves of `lens` for the `stretch` branch, as line segments,
/// extracted on the geometry of `field` (only its geometry is used).
///
/// # Arguments
/// * `lens` - the lens model.
/// * `field` - the sampling geometry; it must span the curves.
/// * `stretch` - which branch of the critical locus to trace.
pub fn critical_curves<F: FieldOps>(
    lens: &dyn LensModel,
    field: &F,
    stretch: Stretch,
) -> Vec<ContourSegment> {
    stretch_field(lens, field, stretch).contour(0.0)
}

/// The caustics of `lens`: the `stretch` critical curves mapped to the source
/// plane by β = θ − α(θ).
///
/// # Arguments
/// * `lens` - the lens model.
/// * `field` - the sampling geometry; it must span the critical curves.
/// * `stretch` - which branch of the critical locus to map.
pub fn caustics<F: FieldOps>(
    lens: &dyn LensModel,
    field: &F,
    stretch: Stretch,
) -> Vec<ContourSegment> {
    // Trace every endpoint in one batch, then reassemble the segments.
    let segments = critical_curves(lens, field, stretch);
    let endpoints: Vec<Point2D> = segments.iter().flatten().copied().collect();
    RayTracer::Direct
        .trace(lens, &endpoints)
        .chunks_exact(2)
        .map(|s| [s[0], s[1]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid, Point2D};
    use crate::lens::{PointMass, SIS};
    use crate::rng::Lcg;

    /// Every segment endpoint lies at radius `radius` (within `tol`).
    fn endpoints_on_circle(segments: &[ContourSegment], radius: f64, tol: f64) {
        assert!(!segments.is_empty());
        for [a, b] in segments {
            assert!((a.norm() - radius).abs() < tol, "endpoint {a:?} off r={radius}");
            assert!((b.norm() - radius).abs() < tol, "endpoint {b:?} off r={radius}");
        }
    }

    /// A jittered `n × n` lattice of sites over `[-radius, radius]²`.
    fn jittered_mesh(n: usize, radius: f64) -> MeshGrid {
        let mut rng = Lcg::new(0xC717);
        let step = 2.0 * radius / n as f64;
        let mut sites = Vec::new();
        for gy in 0..n {
            for gx in 0..n {
                let x = -radius + (gx as f64 + 0.5) * step + 0.2 * step * (rng.f() - 0.5);
                let y = -radius + (gy as f64 + 0.5) * step + 0.2 * step * (rng.f() - 0.5);
                sites.push(Point2D::new(x, y));
            }
        }
        MeshGrid::zeros(&sites, BBox::from_lims(-radius, radius, -radius, radius)).unwrap()
    }

    #[test]
    fn sis_einstein_ring_on_a_pixel_grid() {
        // The SIS critical curve is the circle r = θ_E; the whole ring maps to the
        // origin, so its caustic is a point.
        let theta_e = 0.6;
        let lens = SIS::new(Point2D::origin(), theta_e);
        let field = PixelGrid::constant(Grid2D::square(256, 1.5).unwrap(), 0.0);
        let tol = 2.0 * field.grid.dx();
        endpoints_on_circle(&critical_curves(&lens, &field, Stretch::Total), theta_e, tol);
        endpoints_on_circle(&caustics(&lens, &field, Stretch::Total), 0.0, tol);
    }

    #[test]
    fn sis_einstein_ring_on_a_mesh() {
        // The same lens, the same physics, on an unstructured Voronoi mesh.
        let theta_e = 0.6;
        let lens = SIS::new(Point2D::origin(), theta_e);
        let field = jittered_mesh(60, 1.5); // site spacing ≈ 0.05
        let tol = 0.15;
        endpoints_on_circle(&critical_curves(&lens, &field, Stretch::Total), theta_e, tol);
        endpoints_on_circle(&caustics(&lens, &field, Stretch::Total), 0.0, tol);
    }

    #[test]
    fn point_mass_einstein_ring_on_a_pixel_grid() {
        let theta_e = 0.5;
        let lens = PointMass::new(Point2D::origin(), theta_e);
        let field = PixelGrid::constant(Grid2D::square(256, 1.5).unwrap(), 0.0);
        let tol = 2.0 * field.grid.dx();
        endpoints_on_circle(&critical_curves(&lens, &field, Stretch::Total), theta_e, tol);
        endpoints_on_circle(&caustics(&lens, &field, Stretch::Total), 0.0, tol);
    }

    #[test]
    fn sis_eigen_split_is_tangential_only() {
        // SIS has κ = γ = θ_E/2r, so λ_t = 1 − θ_E/r (ring at r = θ_E) and λ_r = 1
        // everywhere: a tangential critical curve, no radial one. Total ≡ tangential.
        let theta_e = 0.6;
        let lens = SIS::new(Point2D::origin(), theta_e);
        let field = PixelGrid::constant(Grid2D::square(256, 1.5).unwrap(), 0.0);
        let tol = 2.0 * field.grid.dx();
        endpoints_on_circle(&critical_curves(&lens, &field, Stretch::Tangential), theta_e, tol);
        endpoints_on_circle(&critical_curves(&lens, &field, Stretch::Total), theta_e, tol);
        assert!(critical_curves(&lens, &field, Stretch::Radial).is_empty());
        assert!(caustics(&lens, &field, Stretch::Radial).is_empty());
    }
}
