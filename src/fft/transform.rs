// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Forward/inverse 2D FFT (row-then-column, `phastft` backend) and the
//! real↔complex buffer conversions it needs.

use crate::{LensForgeError, Result};
use ndarray::Array2;
use num::complex::Complex;
use phastft::{
    fft_64_with_opts_and_plan,
    options::Options,
    planner::{Direction, Planner64},
};

/// Convert a real (row-major) array into a contiguous complex buffer.
fn to_complex_buffer(values: &Array2<f64>) -> Vec<Complex<f64>> {
    values.iter().map(|&r| Complex::new(r, 0.0)).collect()
}

/// Reshape a contiguous complex buffer into an `(ny, nx)` complex array.
fn complex_buffer_to_array(
    buffer: Vec<Complex<f64>>,
    ny: usize,
    nx: usize,
) -> Array2<Complex<f64>> {
    Array2::from_shape_vec((ny, nx), buffer).expect("buffer size mismatch")
}

/// Error unless both dimensions are powers of two (the `phastft` requirement).
///
/// # Arguments
/// * `ny`, `nx` - array dimensions to validate
/// * `op` - operation name for the error message
fn guard_pow2(ny: usize, nx: usize, op: &str) -> Result<()> {
    if !nx.is_power_of_two() || !ny.is_power_of_two() {
        return Err(LensForgeError::invalid_grid_dimension(format!(
            "{op} (phastft) requires power-of-two dimensions, got ({ny}, {nx})"
        )));
    }
    Ok(())
}

/// In-place row-wise then column-wise 1D FFT over a contiguous `(ny, nx)` buffer.
///
/// # Arguments
/// * `buffer` - `ny*nx` row-major complex samples, transformed in place
/// * `ny`, `nx` - buffer dimensions
/// * `dir` - transform direction (forward or inverse)
pub(crate) fn transform2d(buffer: &mut [Complex<f64>], ny: usize, nx: usize, dir: Direction) {
    let opts = Options::default();
    let row_planner = Planner64::new(nx, dir);
    let col_planner = Planner64::new(ny, dir);
    // row-wise 1D-FFT
    for j in 0..ny {
        let ini = j * nx;
        let row_slice = &mut buffer[ini..ini + nx];
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
}

/// Forward 2D FFT of a real field. Errors on non-power-of-two dimensions.
///
/// # Arguments
/// * `values` - real (row-major) field to transform
pub(crate) fn fft2d(values: &Array2<f64>) -> Result<Array2<Complex<f64>>> {
    let (ny, nx) = (values.nrows(), values.ncols());
    guard_pow2(ny, nx, "FFT")?;
    let mut buffer = to_complex_buffer(values);
    transform2d(&mut buffer, ny, nx, Direction::Forward);
    Ok(complex_buffer_to_array(buffer, ny, nx))
}

/// Inverse 2D FFT to a real field. Errors on non-power-of-two dimensions.
///
/// # Arguments
/// * `arr` - complex spectrum to inverse-transform
pub(crate) fn ifft2d(arr: &Array2<Complex<f64>>) -> Result<Array2<f64>> {
    guard_pow2(arr.nrows(), arr.ncols(), "IFFT")?;
    Ok(ifft2d_unchecked(arr))
}

/// Inverse 2D FFT assuming power-of-two dimensions (no validation).
///
/// Fast path for callers whose input came from [`fft2d`], so the dimensions are
/// already power-of-two and the transform cannot fail.
///
/// # Arguments
/// * `arr` - complex spectrum to inverse-transform
pub(crate) fn ifft2d_unchecked(arr: &Array2<Complex<f64>>) -> Array2<f64> {
    let (ny, nx) = (arr.nrows(), arr.ncols());
    let mut buffer: Vec<Complex<f64>> = arr.iter().cloned().collect();
    transform2d(&mut buffer, ny, nx, Direction::Reverse);
    let values_re: Vec<f64> = buffer.iter().map(|c| c.re).collect();
    // Length equals ny*nx by construction, so the reshape cannot fail.
    Array2::from_shape_vec((ny, nx), values_re).expect("buffer length matches (ny, nx)")
}
