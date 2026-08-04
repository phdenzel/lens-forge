// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
#![doc = include_str!("../README.md")]
// Warn on any undocumented public API item so the docs stay complete as the
// crate grows. Use `#[allow(missing_docs)]` locally for deliberate exceptions.
#![warn(missing_docs)]

pub mod fastsum;
pub mod grid;
pub mod lens;
pub mod solver;
pub mod source;
pub(crate) mod tree;
pub(crate) mod utils;
pub mod viz;

pub use utils::rng::Lcg;

use std::error::Error;
use std::fmt;

/// Common result type for the lens_forge library
pub type Result<T> = std::result::Result<T, LensForgeError>;

/// Error types for the lens_forge library
#[derive(Debug, Clone, PartialEq)]
pub enum LensForgeError {
    /// Invalid grid dimensions
    InvalidGridDimensions(String),
    /// Mismatch between dimensions of grid and values
    DimensionMismatch {
        /// Expected dimensions
        expected: String,
        /// Actual dimensions
        got: String,
    },
    /// Invalid parameter provided
    InvalidParameter(String),
    /// Invalid setting/option provided
    InvalidOption(String),
    /// Computation error occurred
    NumericalError(String),
    /// Computation error occurred
    ComputationError(String),
    /// Solver failed to converge
    ConvergenceError {
        /// Number of iterations performed before giving up
        iterations: usize,
    },
}

impl fmt::Display for LensForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LensForgeError::InvalidGridDimensions(msg) => {
                write!(f, "Invalid grid dimensions: {}", msg)
            }
            LensForgeError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, but got {}",
                    expected, got
                )
            }
            LensForgeError::InvalidParameter(msg) => {
                write!(f, "Invalid parameter: {}", msg)
            }
            LensForgeError::InvalidOption(msg) => {
                write!(f, "Invalid option: {}", msg)
            }
            LensForgeError::NumericalError(msg) => {
                write!(f, "Numerical error: {}", msg)
            }
            LensForgeError::ComputationError(msg) => {
                write!(f, "Computation error: {}", msg)
            }
            LensForgeError::ConvergenceError { iterations } => {
                write!(
                    f,
                    "Solver failed to converge after {} iterations",
                    iterations
                )
            }
        }
    }
}

impl Error for LensForgeError {}

// Convenience constructors for creating errors
impl LensForgeError {
    /// Create a new InvalidGridDimension error
    pub fn invalid_grid_dimension(msg: impl Into<String>) -> Self {
        LensForgeError::InvalidGridDimensions(msg.into())
    }

    /// Create a new DimensionMismatch error
    pub fn dimension_mismatch(expected: String, got: String) -> Self {
        LensForgeError::DimensionMismatch { expected, got }
    }

    /// Create a new InvalidParameter error
    pub fn invalid_parameter(msg: impl Into<String>) -> Self {
        LensForgeError::InvalidParameter(msg.into())
    }

    /// Create a new InvalidOption error
    pub fn invalid_option(msg: impl Into<String>) -> Self {
        LensForgeError::InvalidOption(msg.into())
    }

    /// Create a new NumericalError
    pub fn numerical_error(msg: impl Into<String>) -> Self {
        LensForgeError::NumericalError(msg.into())
    }

    /// Create a new ComputationError
    pub fn computation_error(msg: impl Into<String>) -> Self {
        LensForgeError::ComputationError(msg.into())
    }

    /// Create a new ConvergenceError
    pub fn convergence_error(iterations: usize) -> Self {
        LensForgeError::ConvergenceError { iterations }
    }
}

/// Outcome of an iterative solve (e.g. [`PixelGrid::poisson_cg`]).
///
/// [`PixelGrid::poisson_cg`]: crate::grid::PixelGrid::poisson_cg
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveInfo {
    /// Iterations performed.
    pub iterations: usize,
    /// Final relative residual, `‖r‖ / ‖b‖`.
    pub residual: f64,
    /// Whether the tolerance was reached within the iteration budget.
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LensForgeError::invalid_parameter("theta_e must be positive");
        assert_eq!(
            err.to_string(),
            "Invalid parameter: theta_e must be positive"
        );

        let err = LensForgeError::convergence_error(1000);
        assert_eq!(
            err.to_string(),
            "Solver failed to converge after 1000 iterations"
        );
    }

    #[test]
    fn test_error_is_error_trait() {
        let err: Box<dyn Error> = Box::new(LensForgeError::computation_error("test"));
        assert!(err.to_string().contains("test"));
    }
}
