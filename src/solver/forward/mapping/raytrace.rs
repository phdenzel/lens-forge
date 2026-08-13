// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Ray-tracing: mapping image-plane points to the source plane, in batch.
//!
//! The lens equation β = θ − α(θ) is a point map; a [`RayTracer`] applies it to a
//! whole set of rays at once, batched through [`jet_batch`](LensModel::jet_batch).
//! [`Direct`](RayTracer::Direct) (one jet per ray) is the baseline; adaptive
//! supersampling and inverse ray-shooting join as further variants.

use crate::grid::Point2D;
use crate::lens::LensModel;

/// A ray-tracing strategy: image-plane points → source-plane points.
///
/// Backends share the [`trace`](RayTracer::trace) signature and are selected by
/// variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayTracer {
    /// Direct per-ray evaluation β = θ − α(θ), batched through
    /// [`jet_batch`](LensModel::jet_batch): exact, one jet per ray.
    Direct,
}

impl RayTracer {
    /// Map each of `thetas` to the source plane under `lens`.
    ///
    /// # Arguments
    /// * `lens` - the lens model supplying the deflection field.
    /// * `thetas` - the image-plane rays to trace.
    pub fn trace(&self, lens: &dyn LensModel, thetas: &[Point2D]) -> Vec<Point2D> {
        match self {
            RayTracer::Direct => lens
                .jet_batch(thetas)
                .into_iter()
                .zip(thetas)
                .map(|(jet, &theta)| theta - jet.alpha)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::{PointMass, SIS};

    #[test]
    fn direct_maps_sis_rays_to_the_analytic_source() {
        // SIS: β = θ − θ_E θ̂. On the x-axis at r > 0 this is (r − θ_E, 0).
        let lens = SIS::new(Point2D::origin(), 0.4);
        let thetas = [Point2D::new(1.0, 0.0), Point2D::new(2.0, 0.0)];
        let betas = RayTracer::Direct.trace(&lens, &thetas);
        assert!((betas[0] - Point2D::new(0.6, 0.0)).norm() < 1e-12);
        assert!((betas[1] - Point2D::new(1.6, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn direct_matches_per_ray_ray_trace() {
        let lens = PointMass::new(Point2D::new(0.1, -0.2), 0.5);
        let thetas: Vec<Point2D> =
            (0..7).map(|k| Point2D::new(0.3 * k as f64 + 0.5, 0.2 * k as f64 - 0.4)).collect();
        for (traced, &theta) in RayTracer::Direct.trace(&lens, &thetas).iter().zip(&thetas) {
            assert_eq!(*traced, lens.ray_trace(theta));
        }
    }

    #[test]
    fn empty_input_traces_to_empty() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        assert!(RayTracer::Direct.trace(&lens, &[]).is_empty());
    }
}
