// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The shared result and driver conventions for the solver public-facing components.
//!
//! [`Solution`] pairs a solver's output with its [`SolveInfo`] convergence record; [`Solve`]
//! is the common shape an iterative solver implements — a configured struct that consumes
//! borrowed inputs and yields a [`Solution`]. New solver façades slot in by implementing
//! [`Solve`] and re-exporting from [`solver`](crate::solver).

use crate::Result;
use crate::SolveInfo;

/// A solver's output together with its convergence record.
pub struct Solution<T> {
    /// The produced output.
    pub output: T,
    /// The iteration/residual/convergence record of the solve.
    pub info: SolveInfo,
}

impl<T> Solution<T> {
    /// Pair `output` with its solve `info`.
    ///
    /// # Arguments
    /// * `output` - the produced output.
    /// * `info` - the convergence record.
    pub fn new(output: T, info: SolveInfo) -> Self {
        Self { output, info }
    }
}

/// A configured, iterative solver: it consumes borrowed inputs and yields a [`Solution`].
pub trait Solve {
    /// The borrowed inputs the solve reads.
    type Input<'a>;
    /// The output the solve produces.
    type Output;

    /// Run the solve on `input`.
    ///
    /// # Arguments
    /// * `input` - the borrowed inputs.
    ///
    /// # Errors
    /// Propagates a solver-specific failure (e.g. an infeasible or empty problem).
    fn solve(&self, input: Self::Input<'_>) -> Result<Solution<Self::Output>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial solver summing its inputs with an offset, to exercise the seam.
    struct Adder(f64);
    impl Solve for Adder {
        type Input<'a> = &'a [f64];
        type Output = f64;
        fn solve(&self, input: &[f64]) -> Result<Solution<f64>> {
            let sum: f64 = input.iter().sum::<f64>() + self.0;
            Ok(Solution::new(sum, SolveInfo { iterations: 1, residual: 0.0, converged: true }))
        }
    }

    #[test]
    fn solution_pairs_output_with_info() {
        let sol = Adder(1.0).solve(&[1.0, 2.0]).unwrap();
        assert_eq!(sol.output, 4.0);
        assert_eq!(sol.info.iterations, 1);
        assert!(sol.info.converged);
    }
}
