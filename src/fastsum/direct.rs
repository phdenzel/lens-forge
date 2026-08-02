// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact $O(N^2)$ direct summation — the accuracy oracle for the other backends.

use crate::fastsum::{FieldSolver, Jet, Monopole, SolverConfig};
use crate::grid::Point2D;

/// Brute-force summation over all source elements. Exact (up to summation order),
/// used to validate and benchmark the accelerated backends.
#[derive(Debug, Clone)]
pub struct DirectSum {
    elements: Vec<Monopole>,
    eps2: f64,
}

impl FieldSolver for DirectSum {
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        Self {
            elements: sources.to_vec(),
            eps2: cfg.eps2,
        }
    }

    fn jet(&self, theta: Point2D) -> Jet {
        let mut f = Jet::default();
        for e in &self.elements {
            f.add_monopole(theta, e.position, e.strength, self.eps2);
        }
        f
    }

    fn name(&self) -> &'static str {
        "direct"
    }
}
