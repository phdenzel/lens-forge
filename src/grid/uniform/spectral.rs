// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Spectral (FFT-based) transforms and differentiation for [`PixelGrid`].
//!
//! Split out of `uniform.rs`; these remain inherent `PixelGrid` methods.
//! Finite-difference derivatives live in `diff.rs`.

use super::diff::DerivativeStencil;
use crate::grid::{
    ComplexArray1D, ComplexFieldArray2D, FieldArray2D, FourierSymbols, PixelGrid, VectorField2D,
};
use crate::{LensForgeError, Result};
use ndarray::{Array2, Axis, Zip};
use num::complex::Complex;
use phastft::{
    fft_64_with_opts_and_plan,
    options::Options,
    planner::{Direction, Planner64},
};

impl PixelGrid {
    /* ================ Spectral methods =================================== */
    /// Convert real ndarray (row-major) to contiguous complex vector
    ///
    /// # Arguments
    /// * `values` - 2D array to be converted
    fn ndarray_to_complex_buffer(values: &FieldArray2D) -> Vec<Complex<f64>> {
        values.iter().map(|&r| Complex::new(r, 0.0)).collect()
    }

    /// Convert contiguous complex vector to a complex 2D ndarray
    ///
    /// # Arguments
    /// * `buffer` - complex contiguous vector
    /// * `ny` - size of y axis
    /// * `nx` - size of x axis
    fn complex_buffer_to_ndarray(
        buffer: Vec<Complex<f64>>,
        ny: usize,
        nx: usize,
    ) -> ComplexFieldArray2D {
        Array2::from_shape_vec((ny, nx), buffer).expect("buffer size mismatch")
    }

    /// Test whether number is a power of two
    ///
    /// # Arguments
    /// * `n` - number to be tested as a power of two
    fn is_base2(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Compute 2D FFT (using phastft backend)
    pub fn fft2d(&self) -> Result<ComplexFieldArray2D> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        if !Self::is_base2(nx) || !Self::is_base2(ny) {
            return Err(LensForgeError::invalid_grid_dimension(format!(
                "FFT (phastft) requires power-of-two dimensions, got ({}, {})",
                ny, nx
            )));
        }
        let mut buffer = Self::ndarray_to_complex_buffer(&self.values);
        let opts = Options::default();
        let row_planner = Planner64::new(nx, Direction::Forward);
        let col_planner = Planner64::new(ny, Direction::Forward);
        // row-wise 1D-FFT
        for j in 0..ny {
            let ini = j * nx;
            let fin = ini + nx;
            let row_slice = &mut buffer[ini..fin];
            let mut re: Vec<f64> = row_slice.iter().map(|c| c.re).collect();
            let mut im: Vec<f64> = row_slice.iter().map(|c| c.im).collect();
            fft_64_with_opts_and_plan(&mut re, &mut im, &opts, &row_planner);
            for (i, c) in row_slice.iter_mut().enumerate() {
                *c = Complex::new(re[i], im[i]);
            }
        }
        // column-wise 1D-FFT
        for i in 0..nx {
            let mut col_re: Vec<f64> = (0..ny).map(|j| buffer[j * nx + i].re).collect();
            let mut col_im: Vec<f64> = (0..ny).map(|j| buffer[j * nx + i].im).collect();
            fft_64_with_opts_and_plan(&mut col_re, &mut col_im, &opts, &col_planner);
            for j in 0..ny {
                buffer[j * nx + i] = Complex::new(col_re[j], col_im[j]);
            }
        }

        Ok(Self::complex_buffer_to_ndarray(buffer, ny, nx))
    }

    /// Compute inverse 2D FFT (using phastft backend)
    ///
    /// # Arguments
    /// * `arr` - complex ndarray to be inverse transformed
    pub fn ifft2d(arr: &ComplexFieldArray2D) -> Result<FieldArray2D> {
        let nx = arr.ncols();
        let ny = arr.nrows();

        if !Self::is_base2(nx) || !Self::is_base2(ny) {
            return Err(LensForgeError::invalid_grid_dimension(format!(
                "IFFT (phastft) requires power-of-two dimensions, got ({}, {})",
                ny, nx
            )));
        }
        Ok(Self::ifft2d_unchecked(arr))
    }

    /// Inverse 2D FFT assuming power-of-two dimensions (no validation).
    ///
    /// Internal fast path for callers that produced `arr` from [`Self::fft2d`],
    /// so the dimensions are already power-of-two and the transform cannot fail.
    ///
    /// # Arguments
    /// * `arr` - complex ndarray to be inverse transformed
    pub(crate) fn ifft2d_unchecked(arr: &ComplexFieldArray2D) -> FieldArray2D {
        let nx = arr.ncols();
        let ny = arr.nrows();

        let mut buffer: Vec<Complex<f64>> = arr.iter().cloned().collect();
        let opts = Options::default();
        let row_planner = Planner64::new(nx, Direction::Reverse);
        let col_planner = Planner64::new(ny, Direction::Reverse);
        // row-wise inverse 1D-FFT
        for j in 0..ny {
            let ini = j * nx;
            let fin = ini + nx;
            let row_slice = &mut buffer[ini..fin];
            let mut re: Vec<f64> = row_slice.iter().map(|c| c.re).collect();
            let mut im: Vec<f64> = row_slice.iter().map(|c| c.im).collect();
            fft_64_with_opts_and_plan(&mut re, &mut im, &opts, &row_planner);
            for (i, c) in row_slice.iter_mut().enumerate() {
                *c = Complex::new(re[i], im[i]);
            }
        }
        // column-wise inverse 1D-FFT
        for i in 0..nx {
            let mut col_re: Vec<f64> = (0..ny).map(|j| buffer[j * nx + i].re).collect();
            let mut col_im: Vec<f64> = (0..ny).map(|j| buffer[j * nx + i].im).collect();
            fft_64_with_opts_and_plan(&mut col_re, &mut col_im, &opts, &col_planner);
            for j in 0..ny {
                buffer[j * nx + i] = Complex::new(col_re[j], col_im[j]);
            }
        }

        let values_re: Vec<f64> = buffer.iter().map(|c| c.re).collect();
        // Length equals ny*nx by construction, so the reshape cannot fail.
        Array2::from_shape_vec((ny, nx), values_re).expect("buffer length matches (ny, nx)")
    }

    /// Compute FFT shift (move zero-frequency component to centre)
    ///
    /// # Arguments
    /// * `arr` - complex array to be shifted
    pub fn fftshift(arr: &ComplexFieldArray2D) -> ComplexFieldArray2D {
        let nx = arr.ncols();
        let ny = arr.nrows();
        let nx_half = nx / 2;
        let ny_half = ny / 2;

        let mut shifted = Array2::zeros((ny, nx));

        // swap quadrants
        for j in 0..ny {
            for i in 0..nx {
                let i_shift = (i + nx_half) % nx;
                let j_shift = (j + ny_half) % ny;
                shifted[[j_shift, i_shift]] = arr[[j, i]];
            }
        }

        shifted
    }

    /// Compute inverse FFT shift (undo fftshift)
    ///
    /// # Arguments
    /// * `arr` - complex array to be unshifted
    pub fn ifftshift(arr: &ComplexFieldArray2D) -> ComplexFieldArray2D {
        let nx = arr.ncols();
        let ny = arr.nrows();
        let nx_half = nx.div_ceil(2);
        let ny_half = ny.div_ceil(2);

        let mut unshifted = Array2::zeros((ny, nx));

        // unswap quadrants
        for j in 0..ny {
            for i in 0..nx {
                let i_shift = (i + nx_half) % nx;
                let j_shift = (j + ny_half) % ny;
                unshifted[[j_shift, i_shift]] = arr[[j, i]];
            }
        }

        unshifted
    }

    /// Fourier symbols (for gradients) for given finite difference stencil types
    ///
    /// # Arguments
    /// * `stencil`: derivative stencil type
    #[inline]
    fn fourier_symbols_fd1(&self, stencil: DerivativeStencil) -> FourierSymbols {
        let (kx, ky) = self.grid.wavenumbers();
        let dx = self.grid.dx();
        let dy = self.grid.dy();
        let dkx = kx * dx;
        let dky = ky * dy;

        let (kxd1, kyd1) = match stencil {
            DerivativeStencil::Central => {
                (
                    // (exp(ikh) - exp(-ikh)) / 2h = i*sin(kh) / h
                    dkx.mapv(|theta| Complex::new(0.0, theta.sin() / dx)),
                    dky.mapv(|theta| Complex::new(0.0, theta.sin() / dy)),
                )
            }
            DerivativeStencil::Forward => {
                (
                    // (exp(ikh) - 1) / h
                    dkx.mapv(|theta| Complex::new(theta.cos() - 1.0, theta.sin()) / dx),
                    dky.mapv(|theta| Complex::new(theta.cos() - 1.0, theta.sin()) / dy),
                )
            }
            DerivativeStencil::Backward => {
                (
                    // (1 - exp(-ikh)) / h
                    dkx.mapv(|theta| Complex::new(1.0 - theta.cos(), theta.sin()) / dx),
                    dky.mapv(|theta| Complex::new(1.0 - theta.cos(), theta.sin()) / dy),
                )
            }
            DerivativeStencil::Central4 => {
                (
                    // (-exp(2ikh) + 8*exp(ikh) - 8*exp(-ikh) + exp(-2ikh)) / 12h
                    // = i/h * (4/3 * sin(kh) - 1/6 * sin(2kh))
                    dkx.mapv(|theta| {
                        Complex::new(
                            0.0,
                            ((4.0 / 3.0) * theta.sin() - (1.0 / 6.0) * (2.0 * theta).sin()) / dx,
                        )
                    }),
                    dky.mapv(|theta| {
                        Complex::new(
                            0.0,
                            ((4.0 / 3.0) * theta.sin() - (1.0 / 6.0) * (2.0 * theta).sin()) / dy,
                        )
                    }),
                )
            }
            DerivativeStencil::Central6 => {
                (
                    // (exp(3ikh) - 9*exp(2ikh) + 45*exp(ikh) - 45*exp(-ikh) + 9*exp(-2ikh) - exp(-3ikh)) / 60h
                    // = i/h * (3/2 * sin(kh) - 3/10 * sin(2kh) + 1/30 * sin(3kh))
                    dkx.mapv(|theta| {
                        Complex::new(
                            0.0,
                            (1.5 * theta.sin() - 0.3 * (2.0 * theta).sin()
                                + (1.0 / 30.0) * (3.0 * theta).sin())
                                / dx,
                        )
                    }),
                    dky.mapv(|theta| {
                        Complex::new(
                            0.0,
                            (1.5 * theta.sin() - 0.3 * (2.0 * theta).sin()
                                + (1.0 / 30.0) * (3.0 * theta).sin())
                                / dy,
                        )
                    }),
                )
            }
        };

        (kxd1, kyd1)
    }

    /// Fourier symbols (for Laplacian) for given finite difference stencil types
    ///
    /// # Arguments
    /// * `stencil`: Laplacian stencil type
    #[inline]
    fn fourier_symbols_fd2(&self, stencil: DerivativeStencil) -> FourierSymbols {
        let (kx, ky) = self.grid.wavenumbers();
        let dx = self.grid.dx();
        let dy = self.grid.dy();
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let dkx = kx * dx;
        let dky = ky * dy;

        let (kxd2, kyd2) = match stencil {
            DerivativeStencil::Central => {
                (
                    // (exp(ikh) - 2 + exp(-ikh)) / h^2
                    // = (2*cos(kh)-2) / h^2
                    dkx.mapv(|theta| Complex::new((2.0 * theta.cos() - 2.0) / dx2, 0.0)),
                    dky.mapv(|theta| Complex::new((2.0 * theta.cos() - 2.0) / dy2, 0.0)),
                )
            }
            DerivativeStencil::Forward => {
                (
                    // (exp(ikh) - 1)^2 / h^2
                    // = (2*cos(kh)*(cos(kh)-1) + 2i*(cos(kh)-1)*sin(kh)) / h^2
                    dkx.mapv(|theta| {
                        let cos_theta = theta.cos();
                        Complex::new(
                            2.0 * ((cos_theta - 1.0) * cos_theta),
                            2.0 * (cos_theta - 1.0) * theta.sin(),
                        ) / dx2
                    }),
                    dky.mapv(|theta| {
                        let cos_theta = theta.cos();
                        Complex::new(
                            2.0 * ((cos_theta - 1.0) * cos_theta),
                            2.0 * (cos_theta - 1.0) * theta.sin(),
                        ) / dy2
                    }),
                )
            }
            DerivativeStencil::Backward => {
                (
                    // (1 - exp(-ikh))^2 / h^2 = conjugate of the forward symbol
                    // = (2*cos(kh)*(cos(kh)-1) + 2i*(1-cos(kh))*sin(kh)) / h^2
                    dkx.mapv(|theta| {
                        let cos_theta = theta.cos();
                        Complex::new(
                            2.0 * ((cos_theta - 1.0) * cos_theta),
                            2.0 * (1.0 - cos_theta) * theta.sin(),
                        ) / dx2
                    }),
                    dky.mapv(|theta| {
                        let cos_theta = theta.cos();
                        Complex::new(
                            2.0 * ((cos_theta - 1.0) * cos_theta),
                            2.0 * (1.0 - cos_theta) * theta.sin(),
                        ) / dy2
                    }),
                )
            }
            DerivativeStencil::Central4 => {
                let c2 = -1.0 / 6.0;
                let c1 = 8.0 / 3.0;
                (
                    // (-exp(2ikh) + 16*exp(ikh) - 30 + 16*exp(-ikh) - exp(-2ikh)) / 12h^2
                    // = (-1/6 * cos(2kh) + 8/3 * cos(kh) - 5/2 ) / h^2
                    dkx.mapv(|theta| {
                        Complex::new(
                            (c2 * (2.0 * theta).cos() + c1 * theta.cos() - 2.5) / dx2,
                            0.0,
                        )
                    }),
                    dky.mapv(|theta| {
                        Complex::new(
                            (c2 * (2.0 * theta).cos() + c1 * theta.cos() - 2.5) / dy2,
                            0.0,
                        )
                    }),
                )
            }
            DerivativeStencil::Central6 => {
                let c3 = 1.0 / 45.0;
                let c0 = 49.0 / 18.0;
                (
                    // (2*exp(3ikh) - 27*exp(2ikh) + 270*exp(ikh) - 490 + 270*exp(-ikh) - 27*exp(-2ikh) + 2*exp(-3ikh)) / 180h^2
                    // (1/45*cos(3kh) - 3/10*cos(2kh) + 3*cos(kh) - 49/18) / h^2
                    dkx.mapv(|theta| {
                        Complex::new(
                            (c3 * (3.0 * theta).cos() - 0.3 * (2.0 * theta).cos()
                                + 3.0 * theta.cos()
                                - c0)
                                / dx2,
                            0.0,
                        )
                    }),
                    dky.mapv(|theta| {
                        Complex::new(
                            (c3 * (3.0 * theta).cos() - 0.3 * (2.0 * theta).cos()
                                + 3.0 * theta.cos()
                                - c0)
                                / dy2,
                            0.0,
                        )
                    }),
                )
            }
        };

        (kxd2, kyd2)
    }

    /// Compute gradient using spectral method (using FFT)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn gradient_spectral(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<VectorField2D> {
        let stencil = stencil.into().unwrap_or_default();
        let (kx_fd, ky_fd) = self.fourier_symbols_fd1(stencil);
        self.gradient_spectral_k(&kx_fd, &ky_fd)
    }

    /// Compute gradient using spectral method with given Fourier symbols (using FFT)
    ///
    /// # Arguments
    /// * `kx_fd` - pre-computed Fourier symbols (translation operators) along the x axis
    /// * `ky_fd` - pre-computed Fourier symbols (translation operators) along the y axis
    pub fn gradient_spectral_k(
        &self,
        kx_fd: &ComplexArray1D,
        ky_fd: &ComplexArray1D,
    ) -> Result<VectorField2D> {
        let mut dvdx_fft = self.fft2d()?;
        let mut dvdy_fft = dvdx_fft.clone();

        for mut row in dvdx_fft.axis_iter_mut(Axis(0)) {
            Zip::from(&mut row)
                .and(kx_fd)
                .for_each(|fhat, &kx| *fhat *= kx);
        }

        for (j, mut row) in dvdy_fft.axis_iter_mut(Axis(0)).enumerate() {
            let ky = ky_fd[j];
            row.mapv_inplace(|fhat| fhat * ky);
        }

        // dvdx_fft came from fft2d, so the dimensions are already power-of-two.
        let dvdx = Self::ifft2d_unchecked(&dvdx_fft);
        let dvdy = Self::ifft2d_unchecked(&dvdy_fft);

        Ok((dvdx, dvdy))
    }

    /// Compute Laplacian using spectral method (using FFT)
    ///
    /// Note: assumes periodic boundaries
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn laplacian_spectral(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FieldArray2D> {
        let stencil = stencil.into().unwrap_or_default();
        let (kx_fd2, ky_fd2) = self.fourier_symbols_fd2(stencil);
        self.laplacian_spectral_k(&kx_fd2, &ky_fd2)
    }

    /// Compute Laplacian using spectral method with given squared Fourier symbols (using FFT)
    ///
    /// Note: assumes periodic boundaries
    ///
    /// # Arguments
    /// * `kx_fd2` - pre-computed Fourier symbols (translation operators) along the x axis
    /// * `ky_fd2` - pre-computed Fourier symbols (translation operators) along the x axis
    pub fn laplacian_spectral_k(
        &self,
        kx_fd2: &ComplexArray1D,
        ky_fd2: &ComplexArray1D,
    ) -> Result<FieldArray2D> {
        let mut fft_data = self.fft2d()?;
        for (j, mut row) in fft_data.axis_iter_mut(Axis(0)).enumerate() {
            let ky2 = ky_fd2[j];
            Zip::from(&mut row)
                .and(kx_fd2)
                .for_each(|fhat, &kx2| *fhat *= kx2 + ky2);
        }

        // fft_data came from fft2d, so the dimensions are already power-of-two.
        Ok(Self::ifft2d_unchecked(&fft_data))
    }

    /* ============ Exact spectral operators =============================== */
    /// Exact spectral (Fourier) symbols for the gradient: $ik$ per axis.
    ///
    /// Unlike [`Self::fourier_symbols_fd1`], which reproduces a finite-difference
    /// stencil, these are the *true* derivative symbols. Applied to a field's
    /// spectrum they yield exponentially convergent derivatives for smooth
    /// periodic data.
    #[inline]
    fn fourier_symbols_grad_exact(&self) -> FourierSymbols {
        let (kx, ky) = self.grid.wavenumbers();
        (
            kx.mapv(|k| Complex::new(0.0, k)),
            ky.mapv(|k| Complex::new(0.0, k)),
        )
    }

    /// Exact spectral (Fourier) symbols for the Laplacian: $-k^2$ per axis.
    #[inline]
    fn fourier_symbols_lap_exact(&self) -> FourierSymbols {
        let (kx, ky) = self.grid.wavenumbers();
        (
            kx.mapv(|k| Complex::new(-k * k, 0.0)),
            ky.mapv(|k| Complex::new(-k * k, 0.0)),
        )
    }

    /// Gradient via the exact spectral symbol $ik$ (true spectral derivative).
    ///
    /// For smooth periodic fields this converges exponentially in the grid
    /// resolution, unlike the finite-difference-equivalent
    /// [`Self::gradient_spectral`]. Assumes periodic boundaries.
    pub fn gradient_spectral_exact(&self) -> Result<VectorField2D> {
        let (kx, ky) = self.fourier_symbols_grad_exact();
        self.gradient_spectral_k(&kx, &ky)
    }

    /// Laplacian via the exact spectral symbol $-k^2$ (true spectral Laplacian).
    ///
    /// Assumes periodic boundaries.
    pub fn laplacian_spectral_exact(&self) -> Result<FieldArray2D> {
        let (kx2, ky2) = self.fourier_symbols_lap_exact();
        self.laplacian_spectral_k(&kx2, &ky2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;

    fn approx_eq(a: f64, b: f64, eps: impl Into<Option<f64>>) -> bool {
        let eps = eps.into().unwrap_or(1e-12);
        (a - b).abs() < eps
    }

    #[test]
    fn test_pixel_grid_fft_roundtrip() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (x * std::f64::consts::PI).cos() + y);

        let fft_result = pg.fft2d().unwrap();
        let reconstructed = PixelGrid::ifft2d(&fft_result).unwrap();

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(pg.values[[j, i]], reconstructed[[j, i]], 1e-12));
            }
        }
    }

    #[test]
    fn test_pixel_grid_fft_non_power_of_two() {
        let grid = Grid2D::square(5, 2.0).unwrap();
        let pg = PixelGrid::ones(grid);

        let result = pg.fft2d();
        assert!(result.is_err());
    }

    #[test]
    fn test_pixel_grid_fftshift() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::ones(grid);
        let fft_result = pg.fft2d().unwrap();

        let shifted = PixelGrid::fftshift(&fft_result);
        let unshifted = PixelGrid::ifftshift(&shifted);

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(
                    fft_result[[j, i]].re,
                    unshifted[[j, i]].re,
                    1e-12
                ));
                assert!(approx_eq(
                    fft_result[[j, i]].im,
                    unshifted[[j, i]].im,
                    1e-12
                ));
            }
        }
    }

    #[test]
    fn test_pixel_grid_gradient_spectral() {
        let grid = Grid2D::square(64, 2.0).unwrap();
        //
        let pg = PixelGrid::from_func(grid, |x, y| x + 2.0 * y);

        let (dvdx, dvdy) = pg.gradient_spectral(DerivativeStencil::Central).unwrap();

        // Check interior
        for j in 16..48 {
            for i in 16..48 {
                assert!(approx_eq(dvdx[[j, i]], 1.0, 1e-12));
                assert!(approx_eq(dvdy[[j, i]], 2.0, 1e-12));
            }
        }
    }

    #[test]
    fn test_pixel_grid_laplacian_spectral() {
        let size: usize = 64;
        let grid = Grid2D::square(size, 2.0).unwrap();
        // Quadratic function: Laplacian should be constant
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let laplace_fft = pg.laplacian_spectral(DerivativeStencil::Central).unwrap();

        let border_size = size / 32;
        for j in border_size..size - border_size {
            for i in border_size..size - border_size {
                assert!(approx_eq(laplace_fft[[j, i]], 4.0, 1e-12));
            }
        }
    }

    #[test]
    fn test_laplacian_spectral_vs_finite_diff() {
        let size: usize = 128;
        let grid = Grid2D::square(size, 2.0).unwrap();
        // Quadratic function: Laplacian should be constant
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let lap_fd = pg.laplacian_field(DerivativeStencil::Central, None).unwrap();
        let lap_spec = pg.laplacian_spectral(DerivativeStencil::Central).unwrap();

        let mut max_err: f64 = 0.0;

        let border_size = size / 32;
        for j in border_size..size - border_size {
            for i in border_size..size - border_size {
                let err = (lap_fd[[j, i]] - lap_spec[[j, i]]).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(max_err < 1e-10);
    }

    /// Max absolute difference between two fields.
    fn max_abs_err(a: &FieldArray2D, b: &FieldArray2D) -> f64 {
        Zip::from(a)
            .and(b)
            .fold(0.0_f64, |m, &x, &y| m.max((x - y).abs()))
    }

    #[test]
    fn exact_spectral_gradient_beats_finite_diff() {
        use std::f64::consts::PI;
        // Smooth, periodic on [-π, π]²: f = sin(x)·cos(y), ∂f/∂x = cos(x)·cos(y).
        let grid = Grid2D::square(64, PI).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x.sin() * y.cos());
        let analytic = PixelGrid::from_func(grid, |x, y| x.cos() * y.cos());

        let (sx, _) = pg.gradient_spectral_exact().unwrap();
        let (fx, _) = pg.gradient_field(DerivativeStencil::Central).unwrap();

        let spec_err = max_abs_err(&sx, &analytic.values);
        let fd_err = max_abs_err(&fx, &analytic.values);

        // Exact spectral recovers a band-limited derivative to round-off, while
        // 2nd-order central differences carry an O(dx²) truncation error.
        assert!(spec_err < 1e-10);
        assert!(fd_err > 1e3 * spec_err);
    }

    #[test]
    fn exact_spectral_laplacian_beats_finite_diff() {
        use std::f64::consts::PI;
        // ∇²[sin(x)·cos(y)] = -2 sin(x)·cos(y).
        let grid = Grid2D::square(64, PI).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x.sin() * y.cos());
        let analytic = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());

        let spec = pg.laplacian_spectral_exact().unwrap();
        let fd = pg.laplacian_field(DerivativeStencil::Central, None).unwrap();

        let spec_err = max_abs_err(&spec, &analytic.values);
        let fd_err = max_abs_err(&fd, &analytic.values);

        assert!(spec_err < 1e-10);
        assert!(fd_err > 1e3 * spec_err);
    }

    const ALL_STENCILS: [DerivativeStencil; 5] = [
        DerivativeStencil::Central,
        DerivativeStencil::Forward,
        DerivativeStencil::Backward,
        DerivativeStencil::Central4,
        DerivativeStencil::Central6,
    ];

    #[test]
    fn gradient_spectral_all_stencils_match_linear_interior() {
        // The spectral (circular) evaluation reproduces each FD stencil, exact for a
        // linear field, so the interior recovers ∇ for all orders (fourier_symbols_fd1).
        let grid = Grid2D::square(64, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x + 2.0 * y);
        for stencil in ALL_STENCILS {
            let (dvdx, dvdy) = pg.gradient_spectral(stencil).unwrap();
            for j in 20..44 {
                for i in 20..44 {
                    assert!(approx_eq(dvdx[[j, i]], 1.0, 1e-9));
                    assert!(approx_eq(dvdy[[j, i]], 2.0, 1e-9));
                }
            }
        }
    }

    #[test]
    fn laplacian_spectral_all_stencils_match_quadratic_interior() {
        // Each fourier_symbols_fd2 stencil reproduces its FD 2nd-derivative (exact
        // for a quadratic), so ∇²(x²+y²)=4 in the interior — guards the fd2 coeffs.
        let size = 64usize;
        let grid = Grid2D::square(size, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);
        for stencil in ALL_STENCILS {
            let lap = pg.laplacian_spectral(stencil).unwrap();
            for j in 20..44 {
                for i in 20..44 {
                    assert!(approx_eq(lap[[j, i]], 4.0, 1e-8));
                }
            }
        }
    }

    #[test]
    fn fd2_symbols_vanish_at_dc() {
        // The k=0 (DC) symbol of any consistent 2nd-derivative stencil must be zero
        // (∇² annihilates constants); a non-zero DC term signalled the old fd2 bug.
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::ones(grid);
        for stencil in ALL_STENCILS {
            let (kx2, ky2) = pg.fourier_symbols_fd2(stencil);
            // wavenumbers()[0] is the k=0 mode on both axes.
            assert!(kx2[0].norm() < 1e-12);
            assert!(ky2[0].norm() < 1e-12);
        }
    }

    #[test]
    fn laplacian_spectral_higher_order_beats_second_order() {
        use std::f64::consts::PI;
        // On a smooth periodic field the FD-equivalent spectral Laplacian carries its
        // stencil's truncation error, so Central6 must beat Central (truly higher order).
        let grid = Grid2D::square(32, PI).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x.sin() * y.cos());
        let analytic = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());

        let e2 = max_abs_err(&pg.laplacian_spectral(DerivativeStencil::Central).unwrap(), &analytic.values);
        let e6 = max_abs_err(&pg.laplacian_spectral(DerivativeStencil::Central6).unwrap(), &analytic.values);
        assert!(e6 < e2 * 1e-2);
    }

    #[test]
    fn ifft2d_rejects_non_power_of_two() {
        let arr = Array2::from_elem((5, 5), Complex::new(1.0, 0.0));
        assert!(PixelGrid::ifft2d(&arr).is_err());
    }

    #[test]
    fn spectral_ops_propagate_fft_error() {
        // Non-power-of-two dimensions make the underlying fft2d fail; every
        // spectral operator must surface that error rather than panic.
        let grid = Grid2D::square(5, 2.0).unwrap();
        let pg = PixelGrid::ones(grid);
        assert!(pg.gradient_spectral(None).is_err());
        assert!(pg.laplacian_spectral(None).is_err());
        assert!(pg.gradient_spectral_exact().is_err());
        assert!(pg.laplacian_spectral_exact().is_err());
    }
}
