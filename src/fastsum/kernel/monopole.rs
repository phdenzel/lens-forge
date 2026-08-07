// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`Monopole`] source element: a position with a scalar coefficient.

use crate::grid::Point2D;

/// A point source — the order-0 multipole; e.g. a cell reduced to
/// its monopole with `strength = density · area`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Monopole {
    /// Source position.
    pub position: Point2D,
    /// Scalar coefficient (the order-0 moment).
    pub strength: f64,
}

impl Monopole {
    /// Create a monopole at `position` with the given `strength`.
    ///
    /// # Arguments
    /// * `position` - source position.
    /// * `strength` - scalar coefficient.
    #[inline]
    pub fn new(position: Point2D, strength: f64) -> Self {
        Self { position, strength }
    }
}
