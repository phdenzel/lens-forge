// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
#![doc = include_str!("../README.md")]
// Warn on any undocumented public API item so the docs stay complete as the
// crate grows. Use `#[allow(missing_docs)]` locally for deliberate exceptions.
#![warn(missing_docs)]
// Lets the winit/wgpu viewer opt out of coverage instrumentation (`#![coverage(off)]`).
#![feature(coverage_attribute)]

pub mod fastsum;
pub(crate) mod fft;
pub mod grid;
pub mod cosmology;
pub mod instrument;
pub mod io;
pub mod lens;
pub(crate) mod numerical;
pub(crate) mod rng;
pub mod solver;
pub mod source;
pub(crate) mod tree;
pub(crate) mod arithmetic;
pub mod viz;

pub use rng::Lcg;
pub use solver::solve::SolveInfo;

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
    /// Filesystem or byte-stream I/O failure (stringified, as `std::io::Error`
    /// is neither `Clone` nor `PartialEq`).
    IoError(String),
    /// Malformed or unsupported external data format (e.g. an invalid FITS card).
    FormatError(String),
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
            LensForgeError::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            LensForgeError::FormatError(msg) => {
                write!(f, "Format error: {}", msg)
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

    /// Create a new IoError
    pub fn io_error(msg: impl Into<String>) -> Self {
        LensForgeError::IoError(msg.into())
    }

    /// Create a new FormatError
    pub fn format_error(msg: impl Into<String>) -> Self {
        LensForgeError::FormatError(msg.into())
    }
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
