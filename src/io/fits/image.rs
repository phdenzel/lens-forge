// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! An image HDU's pixels: `BITPIX`, axis lengths, scaling, and the `f64` bridge.
//!
//! [`FitsImage`] pairs a [`FitsData`] buffer with its `NAXISn` lengths and the
//! `BSCALE`/`BZERO`/`BLANK` scaling keywords. It decodes to an
//! [`ndarray::ArrayD`] of physical values and, for 2D data, to a
//! [`FieldArray2D`]/[`PixelGrid`]. Axis order is `NAXIS1` (x) fastest, matching
//! `ndarray` `[.., NAXIS2, NAXIS1]`; there is no row flip, so the array keeps the
//! FITS-native bottom-up orientation.

use super::data::{FitsData, bytes_per_element};
use super::error::FitsError;
use crate::grid::{FieldArray2D, Grid2D, PixelGrid};
use ndarray::{Array2, ArrayD, Axis, Ix2, IxDyn};

/// The default `BITPIX` for writing an `f64` field (IEEE single precision).
pub const DEFAULT_BITPIX: i32 = -32;

/// An image HDU's pixel array with its shape and physical-scaling keywords.
#[derive(Debug, Clone, PartialEq)]
pub struct FitsImage {
    /// The element type selector (`BITPIX`).
    pub bitpix: i32,
    /// The axis lengths `[NAXIS1, NAXIS2, …]` (x fastest).
    pub naxes: Vec<usize>,
    /// The raw element buffer (FITS order, `NAXIS1` fastest).
    pub data: FitsData,
    /// Linear scale factor (`BSCALE`, default 1).
    pub bscale: f64,
    /// Additive offset (`BZERO`, default 0).
    pub bzero: f64,
    /// Integer sentinel for undefined pixels (`BLANK`), mapped to `NaN`.
    pub blank: Option<i64>,
}

impl FitsImage {
    /// Assemble an image from decoded parts, checking the element count.
    ///
    /// # Arguments
    /// * `naxes` - axis lengths `[NAXIS1, NAXIS2, …]`.
    /// * `data` - the element buffer (length must equal the axis product).
    /// * `bscale` - the `BSCALE` factor.
    /// * `bzero` - the `BZERO` offset.
    /// * `blank` - the optional `BLANK` sentinel.
    pub fn new(
        naxes: Vec<usize>,
        data: FitsData,
        bscale: f64,
        bzero: f64,
        blank: Option<i64>,
    ) -> Result<Self, FitsError> {
        let expected: usize = naxes.iter().product();
        if data.len() != expected {
            return Err(FitsError::structure(format!(
                "element count {} does not match axis product {}",
                data.len(),
                expected
            )));
        }
        Ok(Self {
            bitpix: data.bitpix(),
            naxes,
            data,
            bscale,
            bzero,
            blank,
        })
    }

    /// The number of dimensions (`NAXIS`).
    pub fn ndim(&self) -> usize {
        self.naxes.len()
    }

    /// The total number of pixels.
    pub fn n_elements(&self) -> usize {
        self.naxes.iter().product()
    }

    /// The data byte size (unpadded) implied by `BITPIX` and the axes.
    pub fn data_bytes(&self) -> usize {
        let esize = bytes_per_element(self.bitpix).unwrap_or(0);
        self.n_elements() * esize
    }

    /// Decode to physical `f64` values as an [`ndarray::ArrayD`].
    ///
    /// The shape is the reversed axis order `[.., NAXIS2, NAXIS1]` so that, in
    /// `ndarray`'s row-major layout, `NAXIS1` varies fastest.
    pub fn as_f64_array(&self) -> ArrayD<f64> {
        let values = self.data.as_f64(self.bscale, self.bzero, self.blank);
        let shape: Vec<usize> = self.naxes.iter().rev().copied().collect();
        ArrayD::from_shape_vec(IxDyn(&shape), values)
            .expect("element count matches axis product by construction")
    }

    /// A single 2D plane `[NAXIS2, NAXIS1]`, fixing the outer axes.
    ///
    /// # Arguments
    /// * `fixed` - one index per axis beyond the leading 2D plane, outermost
    ///   first; empty for a 2D image.
    pub fn plane(&self, fixed: &[usize]) -> Result<Array2<f64>, FitsError> {
        if self.ndim() < 2 {
            return Err(FitsError::structure("image has fewer than 2 axes"));
        }
        if fixed.len() != self.ndim() - 2 {
            return Err(FitsError::structure(format!(
                "expected {} plane index(es), got {}",
                self.ndim() - 2,
                fixed.len()
            )));
        }
        let arr = self.as_f64_array();
        let mut view = arr.view();
        for &idx in fixed {
            let n = view.len_of(Axis(0));
            if idx >= n {
                return Err(FitsError::structure(format!(
                    "plane index {} out of bounds (0..{})",
                    idx, n
                )));
            }
            view = view.index_axis_move(Axis(0), idx);
        }
        Ok(view
            .into_dimensionality::<Ix2>()
            .expect("2 axes remain after fixing outer indices")
            .to_owned())
    }

    /// Every 2D `[NAXIS2, NAXIS1]` slice along one `ndarray` axis of a 3D cube.
    ///
    /// The channel/band bridge (e.g. Lupton RGB) combines the returned planes.
    ///
    /// # Arguments
    /// * `axis` - the `ndarray` axis to iterate (0 for a channel-first cube).
    pub fn planes(&self, axis: usize) -> Result<Vec<Array2<f64>>, FitsError> {
        if self.ndim() != 3 {
            return Err(FitsError::unsupported(
                "planes() requires a 3D image cube",
            ));
        }
        let arr = self.as_f64_array();
        if axis >= arr.ndim() {
            return Err(FitsError::structure(format!(
                "axis {} out of bounds for a 3D cube",
                axis
            )));
        }
        Ok(arr
            .axis_iter(Axis(axis))
            .map(|v| {
                v.into_dimensionality::<Ix2>()
                    .expect("a slice of a 3D cube is 2D")
                    .to_owned()
            })
            .collect())
    }

    /// The 2D physical field `[NAXIS2, NAXIS1]`; errors unless the image is 2D.
    pub fn to_field_array(&self) -> Result<FieldArray2D, FitsError> {
        if self.ndim() != 2 {
            return Err(FitsError::structure(format!(
                "to_field_array requires a 2D image, got {} axes",
                self.ndim()
            )));
        }
        self.plane(&[])
    }

    /// A [`PixelGrid`] over `grid` carrying the 2D physical field.
    ///
    /// # Arguments
    /// * `grid` - the geometry to attach (shape must match `[NAXIS2, NAXIS1]`).
    pub fn to_pixelgrid(&self, grid: Grid2D) -> Result<PixelGrid, FitsError> {
        let values = self.to_field_array()?;
        PixelGrid::from_values(grid, values).map_err(|e| FitsError::structure(e.to_string()))
    }

    /// Build an image for writing from a 2D `f64` field.
    ///
    /// Float `bitpix` (`-32`/`-64`) writes values directly with no scaling;
    /// integer `bitpix` uses `BSCALE = 1`, `BZERO = 0` unless set afterwards.
    /// Row 0 stays the bottom row (no flip).
    ///
    /// # Arguments
    /// * `bitpix` - the target `BITPIX`; defaults to [`DEFAULT_BITPIX`] (`-32`).
    /// * `field` - the 2D field, shape `[NAXIS2, NAXIS1] = [ny, nx]`.
    pub fn from_field_array(
        bitpix: Option<i32>,
        field: &FieldArray2D,
    ) -> Result<Self, FitsError> {
        let bitpix = bitpix.unwrap_or(DEFAULT_BITPIX);
        let (ny, nx) = field.dim();
        // Row-major iteration is NAXIS1-fastest FITS order already.
        let values: Vec<f64> = field.iter().copied().collect();
        let data = FitsData::from_f64(bitpix, &values, 1.0, 0.0, None)?;
        Self::new(vec![nx, ny], data, 1.0, 0.0, None)
    }

    /// Build an image for writing from a [`PixelGrid`]'s values.
    ///
    /// # Arguments
    /// * `bitpix` - the target `BITPIX`; defaults to [`DEFAULT_BITPIX`] (`-32`).
    /// * `pg` - the pixel grid whose values are written.
    pub fn from_pixelgrid(bitpix: Option<i32>, pg: &PixelGrid) -> Result<Self, FitsError> {
        Self::from_field_array(bitpix, &pg.values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::BBox;
    use ndarray::array;

    #[test]
    fn field_array_roundtrip_orientation() {
        // A ramp so [j, i] layout is unambiguous.
        let field = array![[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]; // [ny=2, nx=3]
        let img = FitsImage::from_field_array(Some(-64), &field).unwrap();
        assert_eq!(img.naxes, vec![3, 2]); // [NAXIS1, NAXIS2]
        let back = img.to_field_array().unwrap();
        assert_eq!(back, field);
    }

    #[test]
    fn default_bitpix_is_single_float() {
        let field = array![[1.0, 2.0]];
        let img = FitsImage::from_field_array(None, &field).unwrap();
        assert_eq!(img.bitpix, -32);
    }

    #[test]
    fn plane_of_cube() {
        // 2 channels, each [ny=2, nx=2]; FITS order NAXIS1 fastest.
        // channel 0: [[0,1],[2,3]], channel 1: [[100,101],[102,103]]
        let data = FitsData::F64(vec![
            0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0,
        ]);
        let img = FitsImage::new(vec![2, 2, 2], data, 1.0, 0.0, None).unwrap();
        assert_eq!(img.ndim(), 3);
        let p0 = img.plane(&[0]).unwrap();
        assert_eq!(p0, array![[0.0, 1.0], [2.0, 3.0]]);
        let p1 = img.plane(&[1]).unwrap();
        assert_eq!(p1, array![[100.0, 101.0], [102.0, 103.0]]);
        let planes = img.planes(0).unwrap();
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[1], p1);
    }

    #[test]
    fn to_pixelgrid_preserves_values() {
        let field = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let img = FitsImage::from_field_array(Some(-64), &field).unwrap();
        let grid = Grid2D::new(3, 2, BBox::from_lims(-1.5, 1.5, -1.0, 1.0)).unwrap();
        let pg = img.to_pixelgrid(grid).unwrap();
        assert_eq!(pg.values, field);
    }

    #[test]
    fn new_rejects_count_mismatch() {
        let data = FitsData::F64(vec![1.0, 2.0, 3.0]);
        assert!(FitsImage::new(vec![2, 2], data, 1.0, 0.0, None).is_err());
    }

    #[test]
    fn ndim_and_data_bytes() {
        let img = FitsImage::from_field_array(Some(-32), &array![[1.0, 2.0]]).unwrap();
        assert_eq!(img.ndim(), 2);
        assert_eq!(img.n_elements(), 2);
        assert_eq!(img.data_bytes(), 2 * 4); // 2 elements × 4 bytes (BITPIX -32)
    }

    #[test]
    fn plane_errors() {
        // 1D image: fewer than 2 axes.
        let one_d = FitsImage::new(vec![3], FitsData::F64(vec![0.0; 3]), 1.0, 0.0, None).unwrap();
        assert!(one_d.plane(&[]).is_err());
        // Wrong number of fixed indices.
        let two_d = FitsImage::from_field_array(Some(-64), &array![[1.0, 2.0]]).unwrap();
        assert!(two_d.plane(&[0]).is_err());
        // Index out of bounds on a cube.
        let cube = FitsImage::new(vec![1, 1, 2], FitsData::F64(vec![0.0; 2]), 1.0, 0.0, None)
            .unwrap();
        assert!(cube.plane(&[5]).is_err());
    }

    #[test]
    fn planes_errors() {
        // planes() requires 3D.
        let two_d = FitsImage::from_field_array(Some(-64), &array![[1.0, 2.0]]).unwrap();
        assert!(two_d.planes(0).is_err());
        // Axis out of bounds on a 3D cube.
        let cube = FitsImage::new(vec![1, 1, 2], FitsData::F64(vec![0.0; 2]), 1.0, 0.0, None)
            .unwrap();
        assert!(cube.planes(9).is_err());
    }

    #[test]
    fn to_field_array_requires_2d() {
        let cube = FitsImage::new(vec![1, 1, 2], FitsData::F64(vec![0.0; 2]), 1.0, 0.0, None)
            .unwrap();
        assert!(cube.to_field_array().is_err());
    }

    #[test]
    fn to_pixelgrid_rejects_mismatched_grid() {
        let img = FitsImage::from_field_array(Some(-64), &array![[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let grid = Grid2D::new(3, 3, BBox::from_lims(-1.0, 1.0, -1.0, 1.0)).unwrap();
        assert!(img.to_pixelgrid(grid).is_err());
    }

    #[test]
    fn from_pixelgrid_uses_values() {
        let grid = Grid2D::new(2, 2, BBox::from_lims(-1.0, 1.0, -1.0, 1.0)).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x + y);
        let img = FitsImage::from_pixelgrid(Some(-64), &pg).unwrap();
        assert_eq!(img.to_field_array().unwrap(), pg.values);
    }
}
