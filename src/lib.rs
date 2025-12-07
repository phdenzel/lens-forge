// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens_forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! # lens_forge
//!
//! A blazingly fast Rust-based gravitational lens modelling library.

pub mod model;
pub mod grid;
// pub mod solver;
pub mod viz;

use std::error::Error;
use std::fmt;

/// Common result type for the lens_forge library
pub type Result<T> = std::result::Result<T, LensForgeError>;

/// Error types for the lens_forge library
#[derive(Debug, Clone, PartialEq)]
pub enum LensForgeError {
    /// Invalid grid dimensions
    InvalidGridDimensions(String),
    /// Invalid parameter provided
    InvalidParameter(String),
    /// Computation error occurred
    ComputationError(String),
    /// Solver failed to converge
    ConvergenceError { iterations: usize },
}


impl fmt::Display for LensForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
	match self {
	    LensForgeError::InvalidGridDimensions(msg) => {
		write!(f, "Invalid grid dimensions: {}", msg)
	    }
	    LensForgeError::InvalidParameter(msg) => {
		write!(f, "Invalid parameter: {}", msg)
	    }
	    LensForgeError::ComputationError(msg) => {
		write!(f, "Computation error: {}", msg)
	    }
	    LensForgeError::ConvergenceError { iterations } => {
		write!(f, "Solver failed to converge after {} iterations", iterations)
	    }
	}
    }
}

impl Error for LensForgeError {}

// Convenience constructors for creating errors
impl LensForgeError {
    /// Create a new InvalidParameter error
    pub fn invalid_parameter(msg: impl Into<String>) -> Self {
        LensForgeError::InvalidParameter(msg.into())
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
