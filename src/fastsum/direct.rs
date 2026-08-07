// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact $O(N^2)$ direct summation — the accuracy oracle for the other backends.

use crate::fastsum::{Jet, SourceKernel, SumConfig, Summation};
use crate::grid::Point2D;

/// Brute-force summation over all source elements. Exact (up to summation order),
/// used to validate and benchmark the accelerated backends.
#[derive(Debug, Clone)]
pub struct DirectSum<S> {
    elements: Vec<S>,
    eps2: f64,
}

impl<S: SourceKernel> DirectSum<S> {
    /// Build the backend from source elements and a configuration.
    ///
    /// # Arguments
    /// * `sources` - the source kernels to sum.
    /// * `cfg` - solver configuration (only `eps2` is used).
    pub fn build(sources: &[S], cfg: SumConfig) -> Self {
        Self {
            elements: sources.to_vec(),
            eps2: cfg.eps2,
        }
    }
}

impl<S: SourceKernel> Summation for DirectSum<S> {
    fn jet(&self, theta: Point2D) -> Jet {
        let mut f = Jet::default();
        for e in &self.elements {
            e.accumulate_near(&mut f, theta, self.eps2);
        }
        f
    }

    fn name(&self) -> &'static str {
        "direct"
    }
}
