// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Convolution for [`PixelGrid`] via the FFT.
//!
//! A Gaussian kernel is an analytic k-space multiplier: convolving with an
//! isotropic Gaussian of width σ multiplies the spectrum by `exp(-k²σ²/2)`, so no
//! kernel is sampled. The zero-frequency mode is left unchanged, so flux is preserved.
//! Requires power-of-two dimensions (the `phastft` backend). A [`Sampled`] kernel
//! instead convolves directly with its pixel values.
//!
//! [`Sampled`]: crate::grid::Kernel::Sampled

use super::pixelgrid::PixelGrid;
use crate::grid::{FieldArray2D, Kernel, Point2D};
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
            Kernel::Gaussian { sigma } => self.convolve_gaussian_mixture(&[(1.0, *sigma)]),
            Kernel::GaussianMixture(components) => self.convolve_gaussian_mixture(components),
            Kernel::Sampled(kernel) => self.convolve_sampled(kernel),
        }
    }

    /// Convolve with a sampled `kernel` by direct summation over its pixels, with
    /// out-of-domain samples treated as zero. Works on any grid size (no FFT).
    ///
    /// # Arguments
    /// * `kernel` - the sampled kernel image, centred on the origin.
    fn convolve_sampled(&self, kernel: &PixelGrid) -> Result<Self> {
        let da = kernel.grid.pixel_area();
        // Each kernel pixel is an offset and a weight wᵦ = Kᵦ·(pixel area) (Σ wᵦ = 1).
        let taps: Vec<(Point2D, f64)> = kernel
            .grid
            .iter_pixels()
            .map(|(a, b)| (kernel.grid.pixel_center(a, b), kernel.get(a, b) * da))
            .collect();
        let (nx, ny) = (self.grid.nx, self.grid.ny);
        let mut out = FieldArray2D::zeros((ny, nx));
        for j in 0..ny {
            for i in 0..nx {
                let x = self.grid.pixel_center(i, j);
                out[[j, i]] = taps
                    .iter()
                    .map(|&(o, w)| self.interpolate_or(x - o, 0.0) * w)
                    .sum();
            }
        }
        self.with_values(out)
    }

    /// Convolve with a Gaussian mixture by multiplying the spectrum with
    /// `Σ wₖ exp(-k²σₖ²/2)` — a single FFT pass for any number of components.
    ///
    /// # Arguments
    /// * `components` - the `(weight, sigma)` Gaussian components.
    fn convolve_gaussian_mixture(&self, components: &[(f64, f64)]) -> Result<Self> {
        let mut spec = self.fft2d()?;
        let (kx, ky) = self.grid.wavenumbers();
        for (j, mut row) in spec.axis_iter_mut(Axis(0)).enumerate() {
            let ky2 = ky[j] * ky[j];
            Zip::from(&mut row).and(&kx).for_each(|fhat, &kxi| {
                let k2 = kxi * kxi + ky2;
                let m: f64 = components
                    .iter()
                    .map(|&(w, s)| w * (-k2 * s * s * 0.5).exp())
                    .sum();
                *fhat *= m;
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
    fn mixture_equals_the_weighted_sum_of_its_components() {
        // A Gaussian mixture convolution is the weighted sum of single-Gaussian
        // convolutions of the same field.
        let grid = Grid2D::square(32, 3.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (x + 0.5 * y).sin());
        let mix = pg
            .convolve(&Kernel::gaussian_mixture(vec![(0.25, 0.3), (0.75, 0.6)]))
            .unwrap();
        let a = pg.convolve(&Kernel::gaussian(0.3)).unwrap();
        let b = pg.convolve(&Kernel::gaussian(0.6)).unwrap();
        for ((m, va), vb) in mix.values.iter().zip(a.values.iter()).zip(b.values.iter()) {
            assert!((m - (0.25 * va + 0.75 * vb)).abs() < 1e-9);
        }
    }

    #[test]
    fn requires_power_of_two_dimensions() {
        let grid = Grid2D::square(30, 2.0).unwrap();
        let pg = PixelGrid::constant(grid, 1.0);
        assert!(pg.convolve(&Kernel::gaussian(0.5)).is_err());
    }

    #[test]
    fn trait_convolve_matches_inherent() {
        // The FieldOps trait path delegates to the inherent method.
        use crate::grid::FieldOps;
        let grid = Grid2D::square(16, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (x + 0.5 * y).sin());
        let k = Kernel::gaussian(0.3);
        let via_trait = FieldOps::convolve(&pg, &k).unwrap();
        let inherent = pg.convolve(&k).unwrap();
        for (a, b) in via_trait.values.iter().zip(inherent.values.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn delta_kernel_is_identity() {
        // A single-pixel unit-integral kernel reproduces the field.
        let grid = Grid2D::square(16, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (x - 0.3 * y).cos());
        let sg = Grid2D::square(1, 0.5).unwrap();
        let kernel = PixelGrid::constant(sg, 1.0 / sg.pixel_area());
        let out = pg.convolve(&Kernel::sampled(kernel)).unwrap();
        for (o, v) in out.values.iter().zip(pg.values.iter()) {
            assert!((o - v).abs() < 1e-9);
        }
    }

    #[test]
    fn box_kernel_reproduces_a_linear_field_in_the_interior() {
        // A symmetric unit box convolves a linear field to itself (interior),
        // and the kernel taps land exactly on neighbouring pixel centres.
        let grid = Grid2D::square(16, 2.0).unwrap(); // dx = 0.25
        let pg = PixelGrid::from_func(grid, |x, y| 2.0 * x - 3.0 * y + 1.0);
        let sg = Grid2D::square(3, 0.375).unwrap(); // kernel dx = 0.25, commensurate
        let kernel = PixelGrid::constant(sg, 1.0 / (9.0 * sg.pixel_area()));
        let out = pg.convolve(&Kernel::sampled(kernel)).unwrap();
        for j in 2..14 {
            for i in 2..14 {
                assert!((out.values[[j, i]] - pg.values[[j, i]]).abs() < 1e-9);
            }
        }
    }
}
