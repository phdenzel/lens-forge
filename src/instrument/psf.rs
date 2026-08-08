// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-spread functions: the observation-side blur.
//!
//! A [`Psf`] is the instrument-facing parameterization of a convolution — a
//! Gaussian or Gaussian mixture by full-width-half-maximum, or an empirical pixel
//! image. It is a thin wrapper over a [`Kernel`]: the astronomy lives here (FWHM →
//! σ), the numerics stay in [`grid`](crate::grid), and the blur itself is the
//! representation-native [`convolve`](FieldOps::convolve) (FFT on a grid, diffusion
//! on a mesh).

use crate::grid::{FieldOps, Kernel, PixelGrid};
use crate::Result;

/// FWHM → standard deviation for a Gaussian: σ = FWHM / (2√(2 ln 2)).
///
/// # Arguments
/// * `fwhm` - full width at half maximum, in domain length units.
fn fwhm_to_sigma(fwhm: f64) -> f64 {
    fwhm / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt())
}

/// A point-spread function: the instrument blur, parameterized for astronomy and
/// carried as a convolution [`Kernel`].
#[derive(Debug, Clone)]
pub struct Psf(Kernel);

impl Psf {
    /// A circular Gaussian PSF of full-width-half-maximum `fwhm`.
    ///
    /// # Arguments
    /// * `fwhm` - full width at half maximum (> 0), in domain length units.
    pub fn gaussian(fwhm: f64) -> Self {
        Psf(Kernel::gaussian(fwhm_to_sigma(fwhm)))
    }

    /// A Gaussian-mixture PSF from `(weight, fwhm)` components (e.g. a Moffat fit).
    ///
    /// # Arguments
    /// * `components` - the `(weight, fwhm)` pairs; weights should sum to one.
    pub fn gaussian_mixture(components: &[(f64, f64)]) -> Self {
        Psf(Kernel::gaussian_mixture(
            components.iter().map(|&(w, fwhm)| (w, fwhm_to_sigma(fwhm))).collect(),
        ))
    }

    /// An empirical PSF from a sampled pixel image.
    ///
    /// # Arguments
    /// * `image` - the PSF image (should integrate to one).
    pub fn pixel(image: PixelGrid) -> Self {
        Psf(Kernel::sampled(image))
    }

    /// Wrap an existing convolution [`Kernel`] (already σ-parameterized) as a PSF.
    ///
    /// # Arguments
    /// * `kernel` - the convolution kernel.
    pub fn from_kernel(kernel: Kernel) -> Self {
        Psf(kernel)
    }

    /// The underlying convolution [`Kernel`].
    pub fn kernel(&self) -> &Kernel {
        &self.0
    }

    /// Apply this PSF to `field`, via the representation-native convolution.
    ///
    /// # Arguments
    /// * `field` - the field to blur (a [`PixelGrid`] or a mesh).
    pub fn apply<F: FieldOps>(&self, field: &F) -> Result<F> {
        field.convolve(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, ScalarField2D};

    #[test]
    fn gaussian_fwhm_compiles_to_the_matching_sigma() {
        // FWHM = 2√(2 ln 2) ≈ 2.3548 gives σ = 1.
        match Psf::gaussian(2.0 * (2.0 * std::f64::consts::LN_2).sqrt()).kernel() {
            Kernel::Gaussian { sigma } => assert!((sigma - 1.0).abs() < 1e-12),
            other => panic!("expected a Gaussian kernel, got {other:?}"),
        }
    }

    #[test]
    fn mixture_converts_each_component_fwhm() {
        match Psf::gaussian_mixture(&[(0.6, 2.0), (0.4, 4.0)]).kernel() {
            Kernel::GaussianMixture(c) => {
                assert_eq!(c.len(), 2);
                assert!((c[0].0 - 0.6).abs() < 1e-12);
                assert!((c[0].1 - fwhm_to_sigma(2.0)).abs() < 1e-12);
                assert!((c[1].1 - fwhm_to_sigma(4.0)).abs() < 1e-12);
            }
            other => panic!("expected a Gaussian-mixture kernel, got {other:?}"),
        }
    }

    #[test]
    fn pixel_psf_passes_through_as_a_sampled_kernel() {
        let image = PixelGrid::constant(Grid2D::square(4, 1.0).unwrap(), 0.25);
        assert!(matches!(Psf::pixel(image).kernel(), Kernel::Sampled(_)));
    }

    #[test]
    fn apply_is_convolution_with_the_kernel() {
        let field = PixelGrid::from_func(Grid2D::square(32, 2.0).unwrap(), |x, y| {
            (2.0 * x).sin() + y
        });
        let psf = Psf::gaussian(0.3);
        let blurred = psf.apply(&field).unwrap();
        let expected = field.convolve(psf.kernel()).unwrap();
        for i in 0..field.len() {
            assert_eq!(blurred.value_at(i), expected.value_at(i));
        }
    }
}
