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

use super::raytrace::RayTracer;
use crate::grid::{ContourSegment, FieldOps, Point2D};
use crate::lens::LensModel;

/// Maximum corrector iterations per predictor step in [`CriticalTracer`].
const CORRECTOR_ITERS: usize = 32;

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

/// The scalar whose zero is the `stretch` critical curve of `lens`, at `theta`.
fn critical_value(lens: &dyn LensModel, theta: Point2D, stretch: Stretch) -> f64 {
    match stretch {
        Stretch::Total => lens.jacobian_det(theta),
        Stretch::Tangential => 1.0 - lens.convergence(theta) - lens.shear(theta),
        Stretch::Radial => 1.0 - lens.convergence(theta) + lens.shear(theta),
    }
}

/// The field whose zero contour is the `stretch` critical curve of `lens`.
fn critical_field<F: FieldOps>(lens: &dyn LensModel, field: &F, stretch: Stretch) -> F {
    field.map_centers(|t| critical_value(lens, t, stretch))
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
    critical_field(lens, field, stretch).contour(0.0)
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

/// Predictor–corrector tracer producing an *ordered* critical curve.
///
/// Follows the curve by arc-length continuation: step along the tangent (⟂ ∇g),
/// then Newton-correct back onto g = 0, where g is the [`Stretch`] scalar. ∇g is
/// central-finite-differenced from the lens, so it needs nothing beyond the jet.
/// Unlike [`critical_curves`] (unordered segments) the output is one ordered
/// polyline, closed when the curve loops — resolving cusps and folds cleanly.
#[derive(Debug, Clone, Copy)]
pub struct CriticalTracer {
    /// Arc-length spacing.
    step: f64,
    /// Convergence tolerance on the critical scalar value.
    tol: f64,
    /// Maximum polyline length before giving up.
    max_points: usize,
    /// Finite-difference epsilon.
    fd_eps: f64,
}

impl Default for CriticalTracer {
    fn default() -> Self {
        Self { step: 0.01, tol: 1e-10, max_points: 100_000, fd_eps: 1e-6 }
    }
}

impl CriticalTracer {
    /// A tracer with default step, tolerance and finite-difference epsilon.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the arc-length step between curve points.
    ///
    /// # Arguments
    /// * `step` - arc-length spacing.
    pub fn with_step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// Set the corrector tolerance on |g|.
    ///
    /// # Arguments
    /// * `tol` - convergence tolerance on the stretch scalar.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the cap on the number of traced points.
    ///
    /// # Arguments
    /// * `max_points` - maximum polyline length before giving up.
    pub fn with_max_points(mut self, max_points: usize) -> Self {
        self.max_points = max_points;
        self
    }

    /// Central finite-difference gradient ∇g of the stretch scalar at `theta`.
    fn gradient(&self, lens: &dyn LensModel, theta: Point2D, stretch: Stretch) -> Point2D {
        let (ex, ey) = (Point2D::new(self.fd_eps, 0.0), Point2D::new(0.0, self.fd_eps));
        let gx = critical_value(lens, theta + ex, stretch) - critical_value(lens, theta - ex, stretch);
        let gy = critical_value(lens, theta + ey, stretch) - critical_value(lens, theta - ey, stretch);
        Point2D::new(gx, gy) * (0.5 / self.fd_eps)
    }

    /// Newton-project `p` onto g = 0 along ∇g, or `None` if the gradient stalls
    /// (not near a critical curve).
    fn correct(&self, lens: &dyn LensModel, mut p: Point2D, stretch: Stretch) -> Option<Point2D> {
        for _ in 0..CORRECTOR_ITERS {
            let g = critical_value(lens, p, stretch);
            if g.abs() <= self.tol {
                return Some(p);
            }
            let grad = self.gradient(lens, p, stretch);
            let gn2 = grad.norm_squared();
            if gn2 < 1e-30 {
                return None;
            }
            p = p - grad * (g / gn2);
        }
        (critical_value(lens, p, stretch).abs() <= self.tol).then_some(p)
    }

    /// Trace the critical curve of `lens` through `seed` for the `stretch` branch,
    /// as an ordered polyline (closed when it loops); empty if `seed` corrects to
    /// no curve.
    ///
    /// # Arguments
    /// * `lens` - the lens model.
    /// * `seed` - a point on or near the target curve.
    /// * `stretch` - which branch of the critical locus to trace.
    pub fn trace(&self, lens: &dyn LensModel, seed: Point2D, stretch: Stretch) -> Vec<Point2D> {
        let Some(start) = self.correct(lens, seed, stretch) else {
            return Vec::new();
        };
        let mut curve = vec![start];
        let (mut p, mut left) = (start, false);
        let mut prev: Option<Point2D> = None;
        for _ in 0..self.max_points {
            let Some(mut tangent) = self.gradient(lens, p, stretch).perp().normalize() else {
                break; // singular gradient
            };
            // Keep marching the same way around the curve.
            if let Some(pt) = prev {
                if tangent.dot(&pt) < 0.0 {
                    tangent = -tangent;
                }
            }
            let Some(next) = self.correct(lens, p + tangent * self.step, stretch) else {
                break;
            };
            let d = (next - start).norm();
            left |= d > 3.0 * self.step;
            if left && d < self.step {
                curve.push(start); // close the loop
                break;
            }
            curve.push(next);
            prev = Some(tangent);
            p = next;
        }
        curve
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid, Point2D};
    use crate::lens::{PointMass, SIE, SIS};
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

    #[test]
    fn tracer_walks_the_sis_ring_as_an_ordered_closed_loop() {
        let theta_e = 0.6;
        let lens = SIS::new(Point2D::origin(), theta_e);
        // Seed slightly off the ring to exercise the corrector.
        let curve = CriticalTracer::new()
            .with_step(0.02)
            .trace(&lens, Point2D::new(0.55, 0.0), Stretch::Tangential);
        assert!(curve.len() > 20, "len {}", curve.len());
        for p in &curve {
            assert!((p.norm() - theta_e).abs() < 1e-4, "point {p:?} off the ring");
        }
        // Consecutive points are ordered (~step apart), and the loop closes.
        for w in curve.windows(2) {
            assert!((w[1] - w[0]).norm() < 3.0 * 0.02);
        }
        assert_eq!(curve.first(), curve.last());
    }

    #[test]
    fn tracer_returns_empty_where_there_is_no_curve() {
        // The SIS radial scalar λ_r = 1 is constant, so ∇g = 0 and no curve exists.
        let lens = SIS::new(Point2D::origin(), 0.6);
        let curve = CriticalTracer::new().trace(&lens, Point2D::new(0.6, 0.0), Stretch::Radial);
        assert!(curve.is_empty());
    }

    #[test]
    fn corrector_rejects_a_seed_it_cannot_refine() {
        // On an elliptical (SIE) oval the corrected point carries a ~1e-14 residual, so a
        // tolerance below attainable precision exhausts the corrector and rejects the seed:
        // tracing never starts. Exercises `with_tol` and the corrector fall-through.
        let lens = SIE::new(Point2D::origin(), 0.6, 0.7, 0.0);
        let curve = CriticalTracer::new()
            .with_tol(1e-18)
            .trace(&lens, Point2D::new(0.55, 0.0), Stretch::Tangential);
        assert!(curve.is_empty());
    }

    #[test]
    fn tracer_respects_the_max_points_cap() {
        // The polyline never exceeds the cap plus the seed (exercises `with_max_points`).
        let lens = SIS::new(Point2D::origin(), 0.6);
        let curve = CriticalTracer::new()
            .with_step(0.02)
            .with_max_points(5)
            .trace(&lens, Point2D::new(0.55, 0.0), Stretch::Tangential);
        assert!(!curve.is_empty() && curve.len() <= 6, "len {}", curve.len());
    }
}
