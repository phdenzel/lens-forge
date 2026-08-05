// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! [`FieldOps`] for [`MeshGrid`]: expose its inherent operators through the
//! shared, representation-agnostic contract.
//!
//! The mesh operators are already `Self`-valued and infallible, so each adapter
//! is a trivial `Ok`-wrap delegating to the inherent method of the same name
//! (inherent methods win method resolution over trait methods).

use super::meshgrid::MeshGrid;
use crate::grid::{ContourSegment, FieldArray1D, FieldOps, Kernel, Point2D, PoissonConfig};
use crate::{Result, SolveInfo};

impl FieldOps for MeshGrid {
    fn interpolate_or(&self, p: Point2D, default: f64) -> f64 {
        MeshGrid::interpolate_or(self, p, default)
    }

    fn par_interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        MeshGrid::par_interpolate_batch(self, points)
    }

    fn gradient_field(&self) -> Result<(Self, Self)> {
        Ok(MeshGrid::gradient_field(self))
    }

    fn gradient_magnitude(&self) -> Result<Self> {
        Ok(MeshGrid::gradient_magnitude(self))
    }

    fn laplacian_field(&self) -> Result<Self> {
        Ok(MeshGrid::laplacian_field(self))
    }

    fn map_centers<G: Fn(Point2D) -> f64>(&self, f: G) -> Self {
        let values = FieldArray1D::from_vec(self.sites().iter().map(|&s| f(s)).collect());
        self.with_values(values)
            .expect("resampling shares this geometry")
    }

    fn poisson(&self, cfg: Option<PoissonConfig>) -> Result<(Self, SolveInfo)> {
        self.poisson_cg(cfg)
    }

    fn contour(&self, level: f64) -> Vec<ContourSegment> {
        MeshGrid::contour(self, level)
    }

    fn convolve(&self, kernel: &Kernel) -> Result<Self> {
        MeshGrid::convolve(self, kernel)
    }
}
