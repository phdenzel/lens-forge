// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! FFT quadrant swaps moving the zero-frequency component to/from the centre.

use ndarray::Array2;
use num::complex::Complex;

/// Move the zero-frequency component to the centre.
///
/// # Arguments
/// * `arr` - complex array to be shifted
pub(crate) fn fftshift(arr: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = (arr.nrows(), arr.ncols());
    let (nx_half, ny_half) = (nx / 2, ny / 2);
    let mut shifted = Array2::zeros((ny, nx));
    for j in 0..ny {
        for i in 0..nx {
            let i_shift = (i + nx_half) % nx;
            let j_shift = (j + ny_half) % ny;
            shifted[[j_shift, i_shift]] = arr[[j, i]];
        }
    }
    shifted
}

/// Undo [`fftshift`].
///
/// # Arguments
/// * `arr` - complex array to be unshifted
pub(crate) fn ifftshift(arr: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = (arr.nrows(), arr.ncols());
    let (nx_half, ny_half) = (nx.div_ceil(2), ny.div_ceil(2));
    let mut unshifted = Array2::zeros((ny, nx));
    for j in 0..ny {
        for i in 0..nx {
            let i_shift = (i + nx_half) % nx;
            let j_shift = (j + ny_half) % ny;
            unshifted[[j_shift, i_shift]] = arr[[j, i]];
        }
    }
    unshifted
}
