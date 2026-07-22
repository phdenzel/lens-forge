// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact $O(N^2)$ direct summation — the accuracy oracle for the other backends.

use crate::grid::Point2D;
use crate::fastsum::{FieldSolver, Monopole, PotentialJet, SolverConfig};

/// Brute-force summation over all source elements. Exact (up to summation order),
/// used to validate and benchmark the accelerated backends.
#[derive(Debug, Clone)]
pub struct DirectSum {
    bodies: Vec<Monopole>,
    eps2: f64,
}

impl FieldSolver for DirectSum {
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        Self {
            bodies: sources.to_vec(),
            eps2: cfg.eps2,
        }
    }

    fn field(&self, theta: Point2D) -> PotentialJet {
        let mut f = PotentialJet::default();
        for b in &self.bodies {
            f.add_monopole(theta, b.position, b.strength, self.eps2);
        }
        f
    }

    fn name(&self) -> &'static str {
        "direct"
    }
}
