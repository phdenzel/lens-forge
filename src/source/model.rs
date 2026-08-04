// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Source models.
//!
//! Every source implements [`SourceModel`]. Its one required method
//! [`brightness`](SourceModel::brightness) returns the surface brightness at a
//! source-plane position β; consumers (solver, rendering) are written against the
//! trait, agnostic to the representation (parametric profile or free-form field).
//!
//! Conventions (scaled/dimensionless angular units): β is a source-plane position,
//! the ray-traced image β = θ − α(θ) of the [`lens`](crate::lens) layer.

use crate::grid::Point2D;

/// A source-plane surface-brightness distribution.
pub trait SourceModel {
    /// Surface brightness at source-plane position `beta`.
    fn brightness(&self, beta: Point2D) -> f64;

    /// The [`brightness`](Self::brightness) at each of `betas`.
    ///
    /// Default maps [`brightness`](Self::brightness); override to batch when the
    /// backend supports it.
    fn brightness_batch(&self, betas: &[Point2D]) -> Vec<f64> {
        betas.iter().map(|&b| self.brightness(b)).collect()
    }
}
