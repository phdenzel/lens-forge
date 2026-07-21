// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! [`FieldOps`] for [`MeshGrid`]: expose its inherent operators through the
//! shared, representation-agnostic contract.
//!
//! The mesh operators are already `Self`-valued and infallible, so the adapters
//! are trivial `Ok`-wraps. Each body delegates to the inherent method of the
//! same name (inherent methods take method-resolution priority over trait
//! methods), keeping this a thin adapter over the finite-volume operators.

use super::meshgrid::MeshGrid;
use crate::grid::{FieldOps, Point2D, PoissonConfig};
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

    fn poisson(&self, cfg: Option<PoissonConfig>) -> Result<(Self, SolveInfo)> {
        self.poisson_cg(cfg)
    }
}
