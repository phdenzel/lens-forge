// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! [`FieldOps`] for [`PixelGrid`]: adapt its inherent operators to the shared,
//! representation-agnostic contract.
//!
//! The inherent operators return raw [`FieldArray2D`](crate::grid::FieldArray2D) data with selectable
//! stencils and boundary conditions; here they are called with their defaults and
//! rewrapped as [`PixelGrid`]s, so generic `FieldOps` code sees the same
//! `Self`-valued results as the mesh. Each body delegates to the inherent method
//! of the same name (inherent methods win method resolution over trait methods).

use super::pixelgrid::PixelGrid;
use crate::grid::{FieldOps, Point2D, PoissonConfig};
use crate::{Result, SolveInfo};

impl FieldOps for PixelGrid {
    fn interpolate_or(&self, p: Point2D, default: f64) -> f64 {
        PixelGrid::interpolate_or(self, p, default)
    }

    fn par_interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        PixelGrid::par_interpolate_batch(self, points)
    }

    fn gradient_field(&self) -> Result<(Self, Self)> {
        // Inherent `gradient_field` takes a stencil argument; `None` is the default.
        let (gx, gy) = PixelGrid::gradient_field(self, None)?;
        Ok((self.with_values(gx)?, self.with_values(gy)?))
    }

    fn gradient_magnitude(&self) -> Result<Self> {
        // Inherent takes a stencil argument; `None` selects the default.
        self.with_values(PixelGrid::gradient_magnitude(self, None)?)
    }

    fn laplacian_field(&self) -> Result<Self> {
        self.with_values(PixelGrid::laplacian_field(self, None, None)?)
    }

    fn poisson(&self, cfg: Option<PoissonConfig>) -> Result<(Self, SolveInfo)> {
        let (psi, info) = self.poisson_cg(cfg)?;
        Ok((self.with_values(psi)?, info))
    }
}
