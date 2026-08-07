// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`SourceKernel`] trait: a source the summation backends sum over.
//!
//! A backend uses a source two ways: its `(center, strength)` moment builds the
//! far-field expansions (tree aggregation, M2M/M2L), while [`near_jet`] gives the
//! exact near/direct contribution. [`Monopole`] is the point-source kernel;
//! extended sources (e.g. the free-form lens's pixel/Voronoi cells) implement the
//! same trait with an exact-integral `near_jet`, so a backend accelerates either
//! identically.
//!
//! [`near_jet`]: SourceKernel::near_jet

use crate::fastsum::{Jet, Monopole};
use crate::grid::Point2D;

/// A source the summation backends aggregate: a far-field monopole moment plus an
/// exact near-field contribution. `Clone` so a backend can own its sources.
pub trait SourceKernel: Send + Sync + Clone {
    /// Position of the source's monopole moment (its far-field expansion centre).
    fn center(&self) -> Point2D;

    /// Order-0 moment (strength) used to build the far-field expansions.
    fn strength(&self) -> f64;

    /// Exact near/direct contribution to the field 2-jet at `theta`.
    ///
    /// # Arguments
    /// * `theta` - query point.
    /// * `eps2` - squared softening (used by singular kernels; ignored by finite ones).
    fn near_jet(&self, theta: Point2D, eps2: f64) -> Jet;
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
    fn near_jet(&self, theta: Point2D, eps2: f64) -> Jet {
        let mut jet = Jet::default();
        jet.add_monopole(theta, self.position, self.strength, eps2);
        jet
    }
}
