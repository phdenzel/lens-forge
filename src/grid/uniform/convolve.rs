// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Convolution for [`PixelGrid`] via the FFT.
//!
//! A Gaussian PSF is an analytic k-space multiplier: convolving with an isotropic
//! Gaussian of width σ multiplies the spectrum by `exp(-k²σ²/2)`, so no kernel
//! stamp is sampled. The zero-frequency mode is left unchanged, so total flux is
//! preserved. Requires power-of-two dimensions (the `phastft` backend).

use super::pixelgrid::PixelGrid;
use crate::grid::Kernel;
use crate::Result;
use ndarray::{Axis, Zip};

impl PixelGrid {
    /// Convolve the field with `kernel`, returning a new grid of the same geometry.
    ///
    /// # Arguments
    /// * `kernel` - the convolution kernel.
    ///
    /// Requires power-of-two dimensions; errors otherwise (via [`Self::fft2d`]).
    pub fn convolve(&self, kernel: &Kernel) -> Result<Self> {
        match kernel {
            Kernel::Gaussian { sigma } => self.convolve_gaussian(*sigma),
        }
    }

    /// Convolve with an isotropic Gaussian of standard deviation `sigma` by
    /// multiplying the spectrum with `exp(-k²σ²/2)`.
    ///
    /// # Arguments
    /// * `sigma` - Gaussian standard deviation in domain length units.
    fn convolve_gaussian(&self, sigma: f64) -> Result<Self> {
        let mut spec = self.fft2d()?;
        let (kx, ky) = self.grid.wavenumbers();
        let half_s2 = 0.5 * sigma * sigma;
        for (j, mut row) in spec.axis_iter_mut(Axis(0)).enumerate() {
            let ky2 = ky[j] * ky[j];
            Zip::from(&mut row).and(&kx).for_each(|fhat, &kxi| {
                *fhat *= (-(kxi * kxi + ky2) * half_s2).exp();
            });
        }
        self.with_values(Self::ifft2d_unchecked(&spec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, ScalarField2D};
    use std::f64::consts::PI;

    #[test]
    fn single_mode_is_attenuated_by_the_gaussian_factor() {
        // On [-π, π]² the wavenumber of mode m is exactly m, so a pure cos(3x) is
        // scaled by exactly exp(-3²σ²/2) — a direct check of the k-space multiplier.
        let grid = Grid2D::square(32, PI).unwrap();
        let pg = PixelGrid::from_func(grid, |x, _y| (3.0 * x).cos());
        let sigma = 0.5_f64;
        let factor = (-9.0 * sigma * sigma / 2.0).exp();
        let out = pg.convolve(&Kernel::gaussian(sigma)).unwrap();
        for (o, v) in out.values.iter().zip(pg.values.iter()) {
            assert!((o - factor * v).abs() < 1e-9);
        }
    }

    #[test]
    fn constant_field_is_unchanged() {
        let grid = Grid2D::square(16, 2.0).unwrap();
        let pg = PixelGrid::constant(grid, 3.0);
        let out = pg.convolve(&Kernel::gaussian(0.7)).unwrap();
        for v in out.values.iter() {
            assert!((v - 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn gaussian_preserves_total_flux_and_lowers_the_peak() {
        // A single spike spreads out: total sum is conserved (DC mode untouched)
        // while the peak drops.
        let grid = Grid2D::square(32, 4.0).unwrap();
        let mut pg = PixelGrid::constant(grid, 0.0);
        pg.values[[16, 16]] = 1.0;
        let out = pg.convolve(&Kernel::gaussian(0.8)).unwrap();
        assert!((ScalarField2D::sum(&out) - 1.0).abs() < 1e-9);
        assert!(out.values[[16, 16]] < 1.0);
        assert!(out.values[[16, 17]] > 0.0);
    }

    #[test]
    fn requires_power_of_two_dimensions() {
        let grid = Grid2D::square(30, 2.0).unwrap();
        let pg = PixelGrid::constant(grid, 1.0);
        assert!(pg.convolve(&Kernel::gaussian(0.5)).is_err());
    }
}
