// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Colorization functions for value buffers from rasters and meshes.

use crate::viz::MIN_CHUNK;
use colorgrad::Gradient;
//use image::{ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;

/// Core function to map a normalized value to RGBA using gradient.
///
/// # Arguments
/// * `v` - Normalized value between [0, 1]
/// * `gradient` - Color gradient to use for mapping
#[inline(always)]
fn map_to_rgba(t: f32, gradient: &dyn Gradient) -> [u8; 4] {
    let color = gradient.at(t);
    [
        (color.r * 255.0) as u8,
        (color.g * 255.0) as u8,
        (color.b * 255.0) as u8,
        (color.a * 255.0) as u8,
    ]
}

/// Safely convert a normalized value to RGBA using gradient.
///
/// # Arguments
/// * `v` - Normalized value between [0, 1]
/// * `gradient` - Color gradient to use for mapping
/// * `invalid_rgba` - Optional fallback colour for non-finite values
#[inline(always)]
fn map_to_rgba_or(v: f32, gradient: &dyn Gradient, invalid_rgba: Option<[u8; 4]>) -> [u8; 4] {
    let fallback = invalid_rgba.unwrap_or([0_u8, 0_u8, 0_u8, 0_u8]);
    if v.is_finite() {
        let t = v.clamp(0.0, 1.0);
        map_to_rgba(t, gradient)
    } else {
        fallback
    }
}

/// Map a value slice to a given RGBA colour buffer using gradient.
///
/// # Arguments
/// * `values` - Slice of normalized values between [0, 1]
/// * `gradient` - Color gradient to use for mapping
/// * `output` - Output buffer slice
#[inline]
pub fn map_to_rgba_inplace(values: &[f32], gradient: &dyn Gradient, output: &mut [[u8; 4]]) {
    assert_eq!(
        values.len(),
        output.len(),
        "Input and output buffers must have same length"
    );
    for (out, &v) in output.iter_mut().zip(values.iter()) {
        *out = map_to_rgba(v, gradient);
    }
}

/// Safely map a value slice to a given RGBA colour buffer using gradient.
///
/// # Arguments
/// * `values` - Slice of normalized values between [0, 1]
/// * `gradient` - Color gradient to use for mapping
/// * `invalid_rgba` - Optional fallback colour for non-finite values
/// * `output` - Output buffer slice
#[inline]
pub fn map_to_rgba_inplace_or(
    values: &[f32],
    gradient: &dyn Gradient,
    invalid_rgba: Option<[u8; 4]>,
    output: &mut [[u8; 4]],
) {
    assert_eq!(
        values.len(),
        output.len(),
        "Input and output buffers must have same length"
    );
    for (out, &v) in output.iter_mut().zip(values.iter()) {
        *out = map_to_rgba_or(v, gradient, invalid_rgba);
    }
}

/// Map a value slice to a new RGBA colour buffer using gradient.
///
/// # Arguments
/// * `values` - Slice of normalized values between [0, 1]
/// * `gradient` - Color gradient to use for mapping
#[inline]
pub fn map_to_rgba_slice(values: &[f32], gradient: &dyn Gradient) -> Vec<[u8; 4]> {
    values.iter().map(|&v| map_to_rgba(v, gradient)).collect()
}

/// Map a value slice to a new RGBA colour buffer using gradient.
///
/// # Arguments
/// * `values` - Slice of normalized values
/// * `gradient` - Color gradient to use for mapping
/// * `invalid_rgba` - Optional fallback colour for non-finite values
#[inline]
pub fn map_to_rgba_slice_or(
    values: &[f32],
    gradient: &dyn Gradient,
    invalid_rgba: Option<[u8; 4]>,
) -> Vec<[u8; 4]> {
    values
        .iter()
        .map(|&v| map_to_rgba_or(v, gradient, invalid_rgba))
        .collect()
}

/// Map a value slice to a given RGBA colour buffer using gradient in parallel (using rayon backend).
///
/// # Arguments
/// * `values` - Slice of normalized values between [0, 1]
/// * `gradient` - Color gradient to use for mapping
/// * `output` - Output buffer slice
#[inline]
pub fn par_map_to_rgba_inplace(
    values: &[f32],
    gradient: &(dyn Gradient + Send + Sync),
    output: &mut [[u8; 4]],
) {
    assert_eq!(
        values.len(),
        output.len(),
        "Input and output slices must have same length"
    );

    values
        .par_chunks(MIN_CHUNK)
        .zip(output.par_chunks_mut(MIN_CHUNK))
        .for_each(|(val_chunk, out_chunk)| {
            for (out, &val) in out_chunk.iter_mut().zip(val_chunk.iter()) {
                *out = map_to_rgba(val, gradient);
            }
        });
}

/// Map a value slice to a given RGBA colour buffer using gradient in parallel (using rayon backend).
///
/// # Arguments
/// * `values` - Slice of normalized values between [0, 1]
/// * `gradient` - Color gradient to use for mapping
/// * `output` - Output buffer slice
#[inline]
pub fn par_map_to_rgba_inplace_or(
    values: &[f32],
    gradient: &(dyn Gradient + Send + Sync),
    invalid_rgba: Option<[u8; 4]>,
    output: &mut [[u8; 4]],
) {
    assert_eq!(
        values.len(),
        output.len(),
        "Input and output slices must have same length"
    );

    values
        .par_chunks(MIN_CHUNK)
        .zip(output.par_chunks_mut(MIN_CHUNK))
        .for_each(|(val_chunk, out_chunk)| {
            for (out, &val) in out_chunk.iter_mut().zip(val_chunk.iter()) {
                *out = map_to_rgba_or(val, gradient, invalid_rgba);
            }
        });
}

/// Map a value slice to a new RGBA colour buffer using gradient in parallel (using rayon backend).
///
/// # Arguments
/// * `values` - Slice of normalized values between [0, 1]
/// * `gradient` - Color gradient to use for mapping
#[inline]
pub fn par_map_to_rgba_slice(
    values: &[f32],
    gradient: &(dyn Gradient + Send + Sync),
) -> Vec<[u8; 4]> {
    values
        .par_iter()
        .map(|&v| map_to_rgba(v, gradient))
        .collect()
}

/// Map a value slice to a new RGBA colour buffer using gradient in parallel (using rayon backend).
///
/// # Arguments
/// * `values` - Slice of normalized values
/// * `gradient` - Color gradient to use for mapping
/// * `invalid_rgba` - Optional fallback colour for non-finite values
#[inline]
pub fn par_map_to_rgba_slice_or(
    values: &[f32],
    gradient: &(dyn Gradient + Send + Sync),
    invalid_rgba: Option<[u8; 4]>,
) -> Vec<[u8; 4]> {
    values
        .par_iter()
        .map(|&v| map_to_rgba_or(v, gradient, invalid_rgba))
        .collect()
}

/* ============ Tests ====================================================== */
#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::gradients::get_gradient;

    #[test]
    fn test_map_to_rgba_or_valid() {
        let gradient = get_gradient("viridis");
        let invalid = Some([0_u8, 0_u8, 0_u8, 0_u8]);

        let c0 = map_to_rgba_or(0.0, gradient.as_ref(), invalid);
        let c5 = map_to_rgba_or(0.5, gradient.as_ref(), invalid);
        let c1 = map_to_rgba_or(1.0, gradient.as_ref(), invalid);

        assert_ne!(c0, [0, 0, 0, 0]);
        assert_ne!(c5, [0, 0, 0, 0]);
        assert_ne!(c1, [0, 0, 0, 0]);
        assert_eq!(c0[3], 255);
        assert_eq!(c5[3], 255);
        assert_eq!(c1[3], 255);
    }

    #[test]
    fn test_map_to_rgba_or_invalid() {
        let gradient = get_gradient("viridis");
        let invalid = Some([255_u8, 0_u8, 255_u8, 128_u8]);

        assert_eq!(
            map_to_rgba_or(f32::NAN, gradient.as_ref(), invalid),
            invalid.unwrap()
        );
        assert_eq!(
            map_to_rgba_or(f32::INFINITY, gradient.as_ref(), invalid),
            invalid.unwrap()
        );
    }

    #[test]
    fn test_map_to_rgba() {
        let gradient = get_gradient("viridis");

        // Fast and safe should match for valid inputs
        let val = 0.5;
        let c_safe = map_to_rgba_or(val, gradient.as_ref(), Some([0_u8, 0_u8, 0_u8, 0_u8]));
        let c_fast = map_to_rgba(val, gradient.as_ref());

        assert_eq!(c_safe, c_fast);
    }

    #[test]
    fn test_map_to_rgba_inplace_or() {
        let gradient = get_gradient("viridis");
        let invalid = Some([255_u8, 0_u8, 0_u8, 255_u8]);

        let values = vec![0.0, 0.5, 1.0, f32::NAN];
        let mut output = vec![[0_u8; 4]; 4];

        map_to_rgba_inplace_or(&values, gradient.as_ref(), invalid, &mut output);

        assert_ne!(output[0], invalid.unwrap());
        assert_ne!(output[1], invalid.unwrap());
        assert_ne!(output[2], invalid.unwrap());
        assert_eq!(output[3], invalid.unwrap()); // NaN handled correctly
    }

    #[test]
    fn test_map_to_rgba_inplace() {
        let gradient = get_gradient("viridis");
        let invalid = Some([0_u8, 0_u8, 0_u8, 0_u8]);

        // All valid values
        let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut output_safe = vec![[0_u8; 4]; 5];
        let mut output_fast = vec![[0_u8; 4]; 5];

        map_to_rgba_inplace_or(&values, gradient.as_ref(), invalid, &mut output_safe);
        map_to_rgba_inplace(&values, gradient.as_ref(), &mut output_fast);

        // Should produce identical results for valid inputs
        assert_eq!(output_safe, output_fast);
    }

    #[test]
    fn test_par_map_to_rgba_inplace_or() {
        let gradient = get_gradient("viridis");
        let invalid = Some([0_u8, 0_u8, 0_u8, 0_u8]);

        let values: Vec<f32> = (0..10000).map(|i| i as f32 / 9999.0).collect();
        let mut output_seq = vec![[0_u8; 4]; 10000];
        let mut output_par = vec![[0_u8; 4]; 10000];

        map_to_rgba_inplace_or(&values, gradient.as_ref(), invalid, &mut output_seq);
        let gradient_sync = gradient.as_ref();
        par_map_to_rgba_inplace_or(&values, gradient_sync, invalid, &mut output_par);

        assert_eq!(output_seq, output_par);
    }

    #[test]
    fn test_par_map_to_rgba_inplace() {
        let gradient = get_gradient("viridis");

        let values: Vec<f32> = (0..10000).map(|i| i as f32 / 9999.0).collect();
        let mut output_seq = vec![[0_u8; 4]; 10000];
        let mut output_par = vec![[0_u8; 4]; 10000];

        map_to_rgba_inplace(&values, gradient.as_ref(), &mut output_seq);
        let gradient_sync = gradient.as_ref();
        par_map_to_rgba_inplace(&values, gradient_sync, &mut output_par);

        assert_eq!(output_seq, output_par);
    }

    #[test]
    fn test_map_to_rgba_slice_safe_and_unsafe() {
        let gradient = get_gradient("magma");
        let invalid = Some([0_u8, 0_u8, 0_u8, 0_u8]);
        let values = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        let colors_safe = map_to_rgba_slice_or(&values, gradient.as_ref(), invalid);
        let colors_fast = map_to_rgba_slice(&values, gradient.as_ref());

        assert_eq!(colors_safe, colors_fast);
    }

    #[test]
    fn test_map_to_rgba_slice_empty() {
        let gradient = get_gradient("viridis");
        let invalid = Some([0_u8, 0_u8, 0_u8, 0_u8]);

        let colors = map_to_rgba_slice_or(&[], gradient.as_ref(), invalid);
        assert_eq!(colors.len(), 0);

        let colors_fast = map_to_rgba_slice(&[], gradient.as_ref());
        assert_eq!(colors_fast.len(), 0);
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_length_mismatch_panics() {
        let gradient = get_gradient("viridis");
        let values = vec![0.0, 0.5, 1.0];
        let invalid = Some([0_u8, 0_u8, 0_u8, 0_u8]);
        let mut output = vec![[0u8; 4]; 2];

        map_to_rgba_inplace_or(&values, gradient.as_ref(), invalid, &mut output);
    }
}
