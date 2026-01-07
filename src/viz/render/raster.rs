// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Raster adapters and rasterization interfaces for rendering grids.
//! This module implements a single canonical `RasterField` that can be
//! retrieved from several conversion patterns.
//!
//! # Key types
//!
//! * `TryFrom<&G> for RasterField` if supported by grids (such as `PixelGrid`):
//!     - direct borrowing the underlying slice if contiguous (copy-on-write)
//!     - falls back to copying if not contiguous
//!
//! * `ToRaster` trait for rasterization of already dense grids like `PixelGrid`:
//!     - implementation for grids that intrinsically provide rasterization parameters
//!
//! * `RasterizeToField` trait with `RasterizeParams` for grids that are not inherently uniform:
//!     - allows configuring target resolution, bounds, and sampling methods
//!
//! # Typical entrypoint usage
//!
//! `rasterize` and `rasterize_with` are provided as a top-level entrypoints.
//!
//! For dense grids, use:
//! ```ignore
//! let rf = rasterize(&pixel_grid)?; // no parameters needed
//! ```
//!
//! For non-uniform grids (that require sampling):
//! ```ignore
//! let params = RasterizeParams::from_dims(512, 512);
//! let rf = rasterize_with(&adaptive_grid, &params)?;
//! ```

use crate::grid::PixelGrid;
use crate::viz::render::RenderBounds;
use crate::{LensForgeError, Result};
use std::borrow::Cow;
use std::convert::TryFrom;
use std::num::NonZeroU32;

/* ============ Raster field =============================================== */
/// A uniform raster interface with dimensions and value buffer.
#[derive(Clone, Debug)]
pub struct RasterField<'a> {
    /// Raster width in pixels (number of columns)
    pub width: NonZeroU32,
    /// Raster height in pixels (number of rows)
    pub height: NonZeroU32,
    /// Flat row-major buffer of length
    pub values: Cow<'a, [f64]>,
}

impl<'a> RasterField<'a> {
    /// The expected number of pixels (`width * height`) as `usize`.
    #[inline]
    pub fn len_expected(&self) -> usize {
        (self.width.get() as usize) * (self.height.get() as usize)
    }

    /// The effective number of pixels as `usize`.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if the buffer length matches `width * height`.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.len() == self.len_expected()
    }

    /// Compute the flat index for 2D coordinates in row-major order.
    ///
    /// # Arguments
    /// * `x` - x coordinate
    /// * `y` - y coordinate
    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        let w = self.width.get();
        let h = self.height.get();
        assert!(x < w && y < h, "RasterField index out of bounds");
        (y as usize) * (w as usize) + (x as usize)
    }

    /// Get a value at given coordinates with bounds checking.
    ///
    /// # Arguments
    /// * `x` - x coordinate
    /// * `y` - y coordinate
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Option<f64> {
        let w = self.width.get();
        let h = self.height.get();
        if x >= w || y >= h {
            return None;
        }
        self.values
            .get((y as usize) * (w as usize) + (x as usize))
            .copied()
    }

    /// A borrowed slice view of the underlying buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Convert into an owned raster field (materializes a copy only if needed).
    #[inline]
    pub fn into_owned(self) -> RasterField<'static> {
        RasterField {
            width: self.width,
            height: self.height,
            values: Cow::Owned(self.values.into_owned()),
        }
    }

    /// Validate shape invariants and return a typed error on mismatch.
    pub fn validate(&self) -> Result<()> {
        if !self.is_valid() {
            return Err(LensForgeError::dimension_mismatch(
                format!(
                    "{}*{} = {}",
                    self.width.get(),
                    self.height.get(),
                    self.len_expected()
                ),
                format!("{}", self.len()),
            ));
        }
        Ok(())
    }

    /// Construct a raster field from an owned `Vec<f64>` and explicit non-zero dimensions.
    pub fn from_vec(
        width: NonZeroU32,
        height: NonZeroU32,
        values: Vec<f64>,
    ) -> Result<RasterField<'static>> {
        let rf = RasterField {
            width,
            height,
            values: Cow::Owned(values),
        };
        rf.validate()?;
        Ok(rf)
    }

    /// Construct a raster field from an owned `Vec<f64>` and explicit dimensions that are checked to be non-zero.
    pub fn from_vec_safe(
        width: u32,
        height: u32,
        values: Vec<f64>,
    ) -> Result<RasterField<'static>> {
        let width = NonZeroU32::new(width)
            .ok_or_else(|| LensForgeError::invalid_option("Raster width must be > 0"))?;
        let height = NonZeroU32::new(height)
            .ok_or_else(|| LensForgeError::invalid_option("Raster height must be > 0"))?;
        RasterField::from_vec(width, height, values)
    }
}

/// Trait for a uniform raster view: dimensions and a row-major value buffer.
pub trait ToRaster {
    /// Raster width in pixels (number of columns)
    fn width(&self) -> NonZeroU32;
    /// Raster height in pixels (number of rows)
    fn height(&self) -> NonZeroU32;
    /// Flat row-major scalar sample buffer
    fn values(&self) -> &[f64]; // row-major w*h
}

impl ToRaster for RasterField<'_> {
    fn width(&self) -> NonZeroU32 {
        self.width
    }
    fn height(&self) -> NonZeroU32 {
        self.height
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
}

/* ============ Raster parameters ========================================== */
/// Sampling modes used when rasterizing non-uniform grids.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RasterSampling {
    /// Nearest-neighbor sampling (fast, good for quick inspection).
    Nearest,
    /// Bilinear sampling (smoother, slightly more expensive).
    Bilinear,
    /// Area/box averaging (best for downsampling, most expensive).
    Area,
}

impl Default for RasterSampling {
    fn default() -> Self {
        RasterSampling::Nearest
    }
}

/// Rasterization parameters for conversions into raster fields.
#[derive(Clone, Copy, Debug)]
pub struct RasterizeParams {
    /// Output raster width
    pub width: NonZeroU32,
    /// Output raster height
    pub height: NonZeroU32,
    /// Optional coordinate bounds to rasterize
    pub bounds: Option<RenderBounds>,
    /// Sampling strategy for conversions requiring rasterization.
    pub sampling: RasterSampling,
}

impl Default for RasterizeParams {
    fn default() -> Self {
        Self {
            width: NonZeroU32::new(512).unwrap(),
            height: NonZeroU32::new(512).unwrap(),
            bounds: None,
            sampling: RasterSampling::Nearest,
        }
    }
}

impl RasterizeParams {
    pub fn from_dims(width: u32, height: u32) -> Result<Self> {
        let width = NonZeroU32::new(width)
            .ok_or_else(|| LensForgeError::invalid_option("Raster width must be > 0"))?;
        let height = NonZeroU32::new(height)
            .ok_or_else(|| LensForgeError::invalid_option("Raster height must be > 0"))?;
        Ok(Self {
            width,
            height,
            bounds: None,
            sampling: RasterSampling::Nearest,
        })
    }
}

/// Trait for (non-uniform) grids that can be rasterized given `RasterizeParams`.
pub trait RasterizeToField {
    fn to_raster(&self, params: &RasterizeParams) -> Result<RasterField<'static>>;
}

/* ============ Pixel grids implementations ================================ */
/// Convert a `&PixelGrid` into `RasterField`
///
/// Note: This is not the recommended conversion pattern for `PixelGrid`;
/// rather use the `ToRaster` adapter.
impl<'a> TryFrom<&'a PixelGrid> for RasterField<'a> {
    type Error = LensForgeError;

    fn try_from(pg: &'a PixelGrid) -> Result<Self> {
        let width = NonZeroU32::new(pg.grid.nx as u32)
            .ok_or_else(|| LensForgeError::invalid_option("Grid width must be > 0"))?;
        let height = NonZeroU32::new(pg.grid.ny as u32)
            .ok_or_else(|| LensForgeError::invalid_option("Grid height must be > 0"))?;
        let values: Cow<'a, [f64]> = match pg.values.as_slice_memory_order() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(pg.values.iter().copied().collect()),
        };
        let rf = RasterField {
            width,
            height,
            values,
        };
        rf.validate()?;
        Ok(rf)
    }
}

/// Convert dense raster sources `&dyn ToRaster` into a borrowed `RasterField`.
impl<'a> TryFrom<&'a dyn ToRaster> for RasterField<'a> {
    type Error = LensForgeError;

    fn try_from(src: &'a dyn ToRaster) -> Result<Self> {
        let rf = RasterField {
            width: src.width(),
            height: src.height(),
            values: Cow::Borrowed(src.values()),
        };
        rf.validate()?;
        Ok(rf)
    }
}

/// Convert a `&PixelGrid` into `RasterField` leveraging TryFrom<&PixelGrid> without casting.
impl RasterizeToField for PixelGrid {
    fn to_raster(&self, _params: &RasterizeParams) -> Result<RasterField<'static>> {
        // Ignore bounds/sampling for now
        let rf: RasterField = self.try_into()?;
        Ok(rf.into_owned())
    }
}

/* ============ Entrypoints ================================================ */
/// Rasterize any supported grid into a raster field for rendering.
///
/// This is the main entrypoint for uniform grids.
pub fn rasterize<G>(grid: &G) -> Result<RasterField<'static>>
where
    for<'a> RasterField<'a>: TryFrom<&'a G, Error = LensForgeError>,
{
    let rf: RasterField = grid.try_into()?;
    Ok(rf.into_owned())
}

/// Rasterize a non-uniform grid into a raster field using explicit parameters.
pub fn rasterize_with<G: RasterizeToField>(
    grid: &G,
    params: &RasterizeParams,
) -> Result<RasterField<'static>> {
    // Delegate to the grid's rasterizer
    grid.to_raster(params)
}

/// Validate a raster buffer against its dimensions.
///
/// # Arguments
/// * `r` - raster buffer
pub fn validate_raster(r: &dyn ToRaster) -> Result<()> {
    let w = r.width().get() as usize;
    let h = r.height().get() as usize;
    let expected = w.saturating_mul(h);
    let got = r.values().len();
    if got != expected {
        return Err(LensForgeError::dimension_mismatch(
            format!("({w}, {h}) = {expected}"),
            format!("{got}"),
        ));
    }
    Ok(())
}

/* ============ Unit tests ================================================= */
#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    /* ============ Mock data structures =================================== */
    /// Simple mock that implements `ToRaster` for testing borrowing behavior.
    struct MockRaster {
        w: NonZeroU32,
        h: NonZeroU32,
        buf: Vec<f64>,
    }
    impl ToRaster for MockRaster {
        fn width(&self) -> NonZeroU32 {
            self.w
        }
        fn height(&self) -> NonZeroU32 {
            self.h
        }
        fn values(&self) -> &[f64] {
            &self.buf
        }
    }

    /// Simple mock "grid" type to exercise the generic `rasterize<G>` path via `TryFrom<&G>`.
    struct MockGrid {
        w: NonZeroU32,
        h: NonZeroU32,
        buf: Vec<f64>,
    }
    impl<'a> std::convert::TryFrom<&'a MockGrid> for RasterField<'a> {
        type Error = LensForgeError;
        fn try_from(mg: &'a MockGrid) -> Result<Self> {
            let rf = RasterField {
                width: mg.w,
                height: mg.h,
                values: std::borrow::Cow::Borrowed(&mg.buf),
            };
            rf.validate()?;
            Ok(rf)
        }
    }

    /// Mock non-uniform grid that implements `RasterizeToField`.
    struct MockNonUniform;
    impl RasterizeToField for MockNonUniform {
        fn to_raster(&self, params: &RasterizeParams) -> Result<RasterField<'static>> {
            let (w, h) = (params.width.get(), params.height.get());
            let len = (w as usize) * (h as usize);
            // Fill with a simple pattern: v[x,y] = x + y as f64
            let mut values = Vec::with_capacity(len);
            for y in 0..h {
                for x in 0..w {
                    values.push((x + y) as f64);
                }
            }
            RasterField::from_vec(params.width, params.height, values)
        }
    }

    /* ============ RasterField tests ====================================== */
    #[test]
    fn raster_field_len_and_valid() {
        let w = NonZeroU32::new(4).unwrap();
        let h = NonZeroU32::new(3).unwrap();
        let values = (0..12).map(|i| i as f64).collect::<Vec<_>>();
        let rf = RasterField::from_vec(w, h, values.clone()).unwrap();

        assert_eq!(rf.len_expected(), 12);
        assert_eq!(rf.len(), 12);
        assert!(rf.is_valid());
        assert_eq!(rf.as_slice(), &values);
    }

    #[test]
    fn raster_field_validate_mismatch_error() {
        let w = NonZeroU32::new(3).unwrap();
        let h = NonZeroU32::new(3).unwrap();
        // Wrong length (8 instead of 9) -> validate() must error with DimensionMismatch
        let rf = RasterField {
            width: w,
            height: h,
            values: std::borrow::Cow::Owned(vec![0.0; 8]),
        };
        let err = rf.validate().unwrap_err();
        assert!(matches!(err, LensForgeError::DimensionMismatch { .. }));
    }

    #[test]
    fn raster_field_idx_and_get() {
        let w = NonZeroU32::new(3).unwrap();
        let h = NonZeroU32::new(2).unwrap();
        // Row-major: y=0 -> [0,1,2], y=1 -> [3,4,5]
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let rf = RasterField::from_vec(w, h, values.clone()).unwrap();

        assert_eq!(rf.idx(0, 0), 0); // (x=0,y=0)
        assert_eq!(rf.idx(2, 0), 2); // (x=2,y=0)
        assert_eq!(rf.idx(0, 1), 3); // (x=0,y=1)
        assert_eq!(rf.get(2, 1), Some(5.0)); // bounds-checked get
        assert_eq!(rf.get(3, 0), None); // out-of-bounds -> None
        assert_eq!(rf.get(0, 2), None); // out-of-bounds -> None
    }

    #[test]
    #[should_panic(expected = "RasterField index out of bounds")]
    fn raster_field_idx_out_of_bounds_panics() {
        let w = NonZeroU32::new(2).unwrap();
        let h = NonZeroU32::new(2).unwrap();
        let rf = RasterField::from_vec(w, h, vec![0.0; 4]).unwrap();
        // idx() uses assert! to guard bounds and panics on violation
        let _ = rf.idx(5, 1);
    }

    #[test]
    fn raster_field_into_owned_materializes() {
        // Start with borrowed via TryFrom<&dyn ToRaster>
        let mr = MockRaster {
            w: NonZeroU32::new(2).unwrap(),
            h: NonZeroU32::new(2).unwrap(),
            buf: vec![10.0, 11.0, 12.0, 13.0],
        };
        let borrowed: RasterField = (&mr as &dyn ToRaster).try_into().unwrap();
        assert!(matches!(borrowed.values, std::borrow::Cow::Borrowed(_)));

        let owned = borrowed.into_owned(); // materialize a copy only if needed
        assert!(matches!(owned.values, std::borrow::Cow::Owned(_)));
        assert_eq!(owned.as_slice(), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(owned.width.get(), 2);
        assert_eq!(owned.height.get(), 2);
    }

    #[test]
    fn raster_field_from_vec_safe_checks_zero_dims() {
        // width=0 -> InvalidOption
        let err = RasterField::from_vec_safe(0, 2, vec![0.0; 0]).unwrap_err();
        assert!(matches!(err, LensForgeError::InvalidOption(_)));

        // height=0 -> InvalidOption
        let err = RasterField::from_vec_safe(2, 0, vec![0.0; 0]).unwrap_err();
        assert!(matches!(err, LensForgeError::InvalidOption(_)));

        // Non-zero dims but wrong buffer length -> DimensionMismatch
        let err = RasterField::from_vec_safe(2, 2, vec![0.0; 3]).unwrap_err();
        assert!(matches!(err, LensForgeError::DimensionMismatch { .. }));
    }

    /* ============ TryFrom<&dyn ToRaster> ================================= */
    #[test]
    fn tryfrom_dyn_to_raster_borrows_and_validates() {
        let mr = MockRaster {
            w: NonZeroU32::new(3).unwrap(),
            h: NonZeroU32::new(1).unwrap(),
            buf: vec![1.0, 2.0, 3.0],
        };
        // implements TryFrom<&dyn ToRaster>
        let rf: RasterField = (&mr as &dyn ToRaster).try_into().unwrap();
        assert!(rf.is_valid());
        assert_eq!(rf.len(), 3);
        assert!(matches!(rf.values, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn tryfrom_dyn_to_raster_rejects_mismatch() {
        let mr = MockRaster {
            w: NonZeroU32::new(3).unwrap(),
            h: NonZeroU32::new(2).unwrap(),
            buf: vec![0.0; 5], // should be 6
        };
        let err = TryInto::<RasterField>::try_into(&mr as &dyn ToRaster).unwrap_err();
        assert!(matches!(err, LensForgeError::DimensionMismatch { .. }));
    }

    /* ============ RasterSampling ========================================= */
    #[test]
    fn raster_sampling_default_is_nearest() {
        let s = RasterSampling::default(); // default() -> Nearest
        assert_eq!(s, RasterSampling::Nearest);
        assert_ne!(s, RasterSampling::Bilinear);
        assert_ne!(s, RasterSampling::Area);
    }

    /* ============ RasterizeParams ======================================== */
    #[test]
    fn rasterize_params_default_and_from_dims() {
        let d = RasterizeParams::default(); // width=512, height=512, bounds=None, sampling=Nearest
        assert_eq!(d.width.get(), 512);
        assert_eq!(d.height.get(), 512);
        assert!(d.bounds.is_none());
        assert_eq!(d.sampling, RasterSampling::Nearest);

        let p = RasterizeParams::from_dims(8, 4).unwrap(); // non-zero dims are accepted
        assert_eq!(p.width.get(), 8);
        assert_eq!(p.height.get(), 4);
        assert!(p.bounds.is_none());
        assert_eq!(p.sampling, RasterSampling::Nearest);

        // Zero dims -> InvalidOption
        let err = RasterizeParams::from_dims(0, 4).unwrap_err();
        assert!(matches!(err, LensForgeError::InvalidOption(_)));
        let err = RasterizeParams::from_dims(8, 0).unwrap_err();
        assert!(matches!(err, LensForgeError::InvalidOption(_)));
    }

    /* ============ validate_raster ======================================== */
    #[test]
    fn validate_raster_checks_shape() {
        // Correct shape: should Ok(())
        let mr_ok = MockRaster {
            w: NonZeroU32::new(2).unwrap(),
            h: NonZeroU32::new(2).unwrap(),
            buf: vec![0.0; 4],
        };
        validate_raster(&mr_ok as &dyn ToRaster).unwrap(); // shape matches w*h

        // Wrong shape: should error
        let mr_bad = MockRaster {
            w: NonZeroU32::new(2).unwrap(),
            h: NonZeroU32::new(2).unwrap(),
            buf: vec![0.0; 3],
        };
        let err = validate_raster(&mr_bad as &dyn ToRaster).unwrap_err();
        assert!(matches!(err, LensForgeError::DimensionMismatch { .. }));
    }

    /* ============ entry points =========================================== */
    #[test]
    fn rasterize_generic_uses_tryfrom_and_returns_owned() {
        let mg = MockGrid {
            w: NonZeroU32::new(3).unwrap(),
            h: NonZeroU32::new(2).unwrap(),
            buf: (0..6).map(|i| i as f64).collect(),
        };
        let rf = rasterize(&mg).unwrap(); // calls RasterField<'a>: TryFrom<&'a G> then into_owned()
        assert_eq!(rf.len_expected(), 6);
        assert!(matches!(rf.values, std::borrow::Cow::Owned(_))); // owned 'static buffer
        assert_eq!(rf.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn rasterize_with_params_on_non_uniform_grid() {
        let params = RasterizeParams::from_dims(4, 3).unwrap();
        let rf = rasterize_with(&MockNonUniform, &params).unwrap(); // delegates to grid.to_raster(params)
        assert_eq!(rf.len_expected(), 12);
        // Verify simple pattern: v[x,y] = x + y
        let w = params.width.get();
        for y in 0..params.height.get() {
            for x in 0..params.width.get() {
                let idx = (y as usize) * (w as usize) + (x as usize);
                assert_eq!(rf.as_slice()[idx], (x + y) as f64);
            }
        }
    }
}
