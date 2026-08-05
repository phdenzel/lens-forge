// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Convolution kernels for [`FieldOps::convolve`](crate::grid::FieldOps::convolve).
//!
//! A kernel is the smoothing profile convolved with a field, in the field's own
//! length units. The isotropic Gaussian comes first — the common case and the one
//! both representations handle natively (an analytic k-space multiplier on the
//! pixel grid, diffusion on the mesh); a [`Sampled`] kernel then carries a general
//! kernel given as a pixel grid (direct convolution on the grid, kernel-gather on
//! the mesh).
//!
//! [`Sampled`]: Kernel::Sampled

use crate::grid::PixelGrid;

/// A convolution kernel, normalized to unit integral.
#[derive(Debug, Clone)]
pub enum Kernel {
    /// Isotropic Gaussian of standard deviation `sigma`, in domain length units.
    Gaussian {
        /// Standard deviation (> 0).
        sigma: f64,
    },
    /// Sum of isotropic Gaussians as `(weight, sigma)` pairs — a Gaussian mixture
    /// (e.g. a Moffat fit). Weights should sum to one for unit integral.
    GaussianMixture(Vec<(f64, f64)>),
    /// A general kernel sampled on a pixel grid centred on the origin, with values
    /// summing (times pixel area) to one for unit integral.
    Sampled(PixelGrid),
}

impl Kernel {
    /// An isotropic Gaussian kernel of standard deviation `sigma`.
    ///
    /// # Arguments
    /// * `sigma` - standard deviation in domain length units (> 0).
    pub fn gaussian(sigma: f64) -> Self {
        Self::Gaussian { sigma }
    }

    /// A Gaussian-mixture kernel from `(weight, sigma)` components.
    ///
    /// # Arguments
    /// * `components` - the `(weight, sigma)` pairs; weights should sum to one.
    pub fn gaussian_mixture(components: Vec<(f64, f64)>) -> Self {
        Self::GaussianMixture(components)
    }

    /// A general kernel sampled on a pixel grid (`kernel`, centred on the origin,
    /// unit integral).
    ///
    /// # Arguments
    /// * `kernel` - the sampled kernel image.
    pub fn sampled(kernel: PixelGrid) -> Self {
        Self::Sampled(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_constructor_sets_sigma() {
        match Kernel::gaussian(1.5) {
            Kernel::Gaussian { sigma } => assert_eq!(sigma, 1.5),
            _ => panic!("expected a Gaussian kernel"),
        }
    }
}
