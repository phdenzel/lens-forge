// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`SourceKernel`] trait: a source the summation backends sum over.
//!
//! A backend uses a source two ways: its `(center, strength)` moment builds the
//! far-field expansions (tree aggregation, M2M/M2L), while [`accumulate_near`] adds
//! the exact near/direct contribution in place. [`Monopole`] is the point-source
//! kernel; extended sources (e.g. the free-form lens's pixel/Voronoi cells) implement
//! the same trait with an exact-integral near field, so a backend accelerates either
//! identically.
//!
//! [`accumulate_near`]: SourceKernel::accumulate_near

use crate::fastsum::{Jet, Monopole};
use crate::grid::Point2D;

/// A source the summation backends aggregate: a far-field monopole moment plus an
/// exact near-field contribution.
pub trait SourceKernel: Send + Sync + Clone {
    /// Position of the source's monopole moment (its far-field expansion centre).
    fn center(&self) -> Point2D;

    /// Order-0 moment (strength) used to build the far-field expansions.
    fn strength(&self) -> f64;

    /// Add this source's exact near/direct contribution to `field` at `theta`.
    ///
    /// # Arguments
    /// * `field` - jet accumulator, updated in place.
    /// * `theta` - query point.
    /// * `eps2` - squared softening (used by singular kernels; ignored by finite ones).
    fn accumulate_near(&self, field: &mut Jet, theta: Point2D, eps2: f64);
}

impl SourceKernel for Monopole {
    #[inline]
    fn center(&self) -> Point2D {
        self.position
    }

    #[inline]
    fn strength(&self) -> f64 {
        self.strength
    }

    #[inline]
    fn accumulate_near(&self, field: &mut Jet, theta: Point2D, eps2: f64) {
        field.add_monopole(theta, self.position, self.strength, eps2);
    }
}
