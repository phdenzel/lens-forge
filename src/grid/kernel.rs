// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Convolution kernels for [`FieldOps::convolve`](crate::grid::FieldOps::convolve).
//!
//! A kernel is the point-spread / smoothing profile applied to a field, in the
//! field's own length units. The isotropic Gaussian comes first: it is the common
//! PSF and the one both representations handle natively — an analytic k-space
//! multiplier on the pixel grid, diffusion on the mesh.

/// A smoothing / point-spread kernel, normalized to unit integral.
#[derive(Debug, Clone, PartialEq)]
pub enum Kernel {
    /// Isotropic Gaussian of standard deviation `sigma`, in domain length units.
    Gaussian {
        /// Standard deviation (> 0).
        sigma: f64,
    },
}

impl Kernel {
    /// An isotropic Gaussian kernel of standard deviation `sigma`.
    ///
    /// # Arguments
    /// * `sigma` - standard deviation in domain length units (> 0).
    pub fn gaussian(sigma: f64) -> Self {
        Self::Gaussian { sigma }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_constructor_sets_sigma() {
        assert_eq!(Kernel::gaussian(1.5), Kernel::Gaussian { sigma: 1.5 });
    }
}
