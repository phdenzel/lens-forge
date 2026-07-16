// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Reusable rendering contexts for efficient processing.
//!
//! This module provides rendering contexts that maintain reusable buffers,
//! avoiding repeated allocations when rendering multiple fields.

use crate::viz::colorize::{map_to_rgba_inplace, par_map_to_rgba_inplace};
use crate::viz::render::{MeshField, RasterField, RenderOptions};
use crate::viz::scale::{Normalize, RenderScale, auto_range};
use crate::{LensForgeError, Result};
use image::RgbaImage;
use image::imageops::{FilterType, resize};

/// Context for rendering raster fields.
///
/// This context maintains internal buffers that are reused across multiple
/// render calls, avoiding repeated allocations and improving performance
/// when rendering many fields.
pub struct RasterContext {
    /// Buffer for values transformations (scaling/normalization)
    stage_buffer: Vec<f32>,
    /// Buffer for colours (RGBA as [u8; 4])
    color_buffer: Vec<[u8; 4]>,
}

impl RasterContext {
    /// Create a new rendering context with empty buffers.
    pub fn new() -> Self {
        Self {
            stage_buffer: Vec::new(),
            color_buffer: Vec::new(),
        }
    }

    /// Create a context pre-allocated for a specific size.
    ///
    /// This avoids the initial allocation if you know the size ahead of time.
    ///
    /// # Arguments
    /// * `capacity` - Expected number of pixels
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            stage_buffer: Vec::with_capacity(capacity),
            color_buffer: Vec::with_capacity(capacity),
        }
    }

    /// Get current buffer capacity (useful for monitoring memory usage).
    pub fn capacity(&self) -> usize {
        self.stage_buffer.capacity()
    }

    /// Shrink buffers to fit current size (frees excess memory).
    pub fn shrink_to_fit(&mut self) {
        self.stage_buffer.shrink_to_fit();
        self.color_buffer.shrink_to_fit();
    }

    /// Clear buffers (keeps capacity for reuse).
    pub fn clear(&mut self) {
        self.stage_buffer.clear();
        self.color_buffer.clear();
    }

    /// Apply configured scaling transformation to raster values.
    ///
    /// # Arguments:
    /// * `opt` - Rendering configuration
    fn apply_scaling(&mut self, opt: &RenderOptions) -> Result<()> {
        let fallback = match opt.scale {
            RenderScale::Log10 => opt.vmin.unwrap_or(-10.0),
            _ => 0.0,
        };

        if self.stage_buffer.len() > opt.threshold_parallel {
            opt.scale
                .par_apply_slice_inplace(&mut self.stage_buffer, fallback);
        } else {
            opt.scale
                .apply_slice_inplace(&mut self.stage_buffer, fallback);
        }

        Ok(())
    }

    /// Apply configured normalization to raster values.
    ///
    /// # Arguments:
    /// * `opt` - Rendering configuration
    fn apply_normalization(&mut self, opt: &RenderOptions) -> Result<()> {
        let (vmin, vmax) = self.determine_range(opt)?;

        let norm = Normalize::default()
            .with_clamp(opt.clamp)
            .with_lims(vmin, vmax)?;

        if self.stage_buffer.len() > opt.threshold_parallel {
            norm.par_apply_slice_inplace(&mut self.stage_buffer);
        } else {
            norm.apply_slice_inplace(&mut self.stage_buffer);
        }

        Ok(())
    }

    /// Determine normalization range from options or auto-detect from data.
    ///
    /// # Arguments
    /// * `opt` - Rendering options
    fn determine_range(&self, opt: &RenderOptions) -> Result<(f32, f32)> {
        let vmin = if let Some(v) = opt.vmin {
            v
        } else {
            auto_range(&self.stage_buffer)
                .map(|(min, _)| min)
                .unwrap_or(0.0)
        };

        let vmax = if let Some(v) = opt.vmax {
            v
        } else {
            auto_range(&self.stage_buffer)
                .map(|(_, max)| max)
                .unwrap_or(1.0)
        };

        if vmin > vmax {
            return Err(LensForgeError::invalid_option(
                "vmin must be less than vmax",
            ));
        }
        Ok((vmin, vmax))
    }

    /// Apply configured colormap to raster values.
    ///
    /// # Arguments:
    /// * `opt` - Rendering configuration
    fn apply_colormap(&mut self, opt: &RenderOptions) -> Result<()> {
        let gradient = &*opt.gradient();
        if self.color_buffer.len() > opt.threshold_parallel {
            par_map_to_rgba_inplace(&self.stage_buffer, gradient, &mut self.color_buffer);
        } else {
            map_to_rgba_inplace(&self.stage_buffer, gradient, &mut self.color_buffer);
        }
        Ok(())
    }

    /// Create image from the RGBA buffer.
    ///
    /// # Arguments:
    /// * `width` - target width
    /// * `height` - target height
    fn to_image(&self, width: u32, height: u32) -> Result<RgbaImage> {
        let target_len = (width * height) as usize;
        if target_len != self.color_buffer.len() {
            return Err(LensForgeError::dimension_mismatch(
                format!("Buffer length {}", self.color_buffer.len()),
                format!("Buffer length {}", target_len),
            ));
        }
        let rgba_buffer: Vec<u8> = self.color_buffer.iter().flat_map(|&c| c).collect();
        RgbaImage::from_raw(width, height, rgba_buffer)
            .ok_or_else(|| LensForgeError::computation_error("Failed to create image from buffer"))
    }

    /// Resize an image
    /// # Arguments:
    /// * `img` - image reference
    /// * `nwidth` - target width
    /// * `nheight` - target height
    fn resize_image(
        img: &RgbaImage,
        nwidth: u32,
        nheight: u32,
        filter: Option<image::imageops::FilterType>,
    ) -> Result<RgbaImage> {
        let filter = filter.unwrap_or(FilterType::CatmullRom);
        let resized = resize(img, nwidth, nheight, filter);
        Ok(resized)
    }

    /// Render a raster field using this context's buffers.
    ///
    /// This method reuses internal buffers, avoiding allocations if the
    /// buffer sizes match the field size.
    ///
    /// # Arguments
    /// * `field` - Raster field to render
    /// * `opt` - Rendering options
    pub fn render(&mut self, field: &RasterField, opt: &RenderOptions) -> Result<RgbaImage> {
        // Validate
        field.validate()?;
        opt.validate()?;

        let width = field.width.get();
        let height = field.height.get();
        let values: &[f32] = &field.as_slice_f32();
        let n = values.len();

        // Ensure capacity and copy raster values into allocation
        if self.stage_buffer.len() != n {
            self.stage_buffer.resize(n, 0.0);
        }
        if self.color_buffer.len() != n {
            self.color_buffer.resize(n, [0u8; 4]);
        }
        self.stage_buffer.copy_from_slice(&values);

        // Apply rendering stages in order: scaling, normalization, colormap
        self.apply_scaling(opt)?;
        self.apply_normalization(opt)?;
        self.apply_colormap(opt)?;
        // Convert to RGBA image
        let mut image = self.to_image(width, height)?;

        if opt.width_u32() != image.width() || opt.height_u32() != image.height() {
            image = Self::resize_image(&image, opt.width_u32(), opt.height_u32(), opt.filter())?;
        }

        Ok(image)
    }
}

impl Default for RasterContext {
    fn default() -> Self {
        Self::new()
    }
}


/* ============ Unit tests ================================================= */
#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::render::{RasterField, RenderOptions};
    use crate::viz::scale::RenderScale;
    use std::num::NonZeroU32;

    /* ============ Test setup ============================================= */
    /// Create a test raster field with given dimensions and value pattern
    fn create_test_field(width: u32, height: u32) -> RasterField<'static> {
        let w = NonZeroU32::new(width).unwrap();
        let h = NonZeroU32::new(height).unwrap();
        let size = (width * height) as usize;

        // Create a gradient pattern: values from 0.0 to 1.0
        let values: Vec<f64> = (0..size).map(|i| i as f64 / size as f64).collect();

        RasterField::from_vec(w, h, values).unwrap()
    }

    /// Create a test field with constant value
    fn create_constant_field(width: u32, height: u32, value: f64) -> RasterField<'static> {
        let w = NonZeroU32::new(width).unwrap();
        let h = NonZeroU32::new(height).unwrap();
        let size = (width * height) as usize;
        let values = vec![value; size];
        RasterField::from_vec(w, h, values).unwrap()
    }

    /// Create a test field with specific value range
    fn create_ranged_field(width: u32, height: u32, min: f64, max: f64) -> RasterField<'static> {
        let w = NonZeroU32::new(width).unwrap();
        let h = NonZeroU32::new(height).unwrap();
        let size = (width * height) as usize;

        let values: Vec<f64> = (0..size)
            .map(|i| min + (max - min) * (i as f64 / size as f64))
            .collect();

        RasterField::from_vec(w, h, values).unwrap()
    }

    /// Create default render options for testing
    fn default_options() -> RenderOptions {
        RenderOptions::default()
    }

    /// Create render options with specific dimensions
    fn options_with_dims(width: u32, height: u32) -> RenderOptions {
        let mut opt = RenderOptions::default();
        opt.width = NonZeroU32::new(width).unwrap();
        opt.height = NonZeroU32::new(height).unwrap();
        opt
    }

    /* ============ Constructor tests ====================================== */
    #[test]
    fn new_creates_empty_context() {
        let ctx = RasterContext::new();
        assert_eq!(ctx.capacity(), 0);
    }

    #[test]
    fn with_capacity_preallocates_buffers() {
        let ctx = RasterContext::with_capacity(1000);
        // Capacity should be at least the requested size
        assert!(ctx.capacity() >= 1000);
    }

    #[test]
    fn with_capacity_zero_is_valid() {
        let ctx = RasterContext::with_capacity(0);
        assert_eq!(ctx.capacity(), 0);
    }

    #[test]
    fn default_trait_creates_empty_context() {
        let ctx: RasterContext = Default::default();
        assert_eq!(ctx.capacity(), 0);
    }

    /* ============ Buffer management tests ================================ */
    #[test]
    fn clear_keeps_capacity() {
        let mut ctx = RasterContext::with_capacity(1000);
        let initial_capacity = ctx.capacity();
        ctx.clear();
        // Capacity should remain the same after clear
        assert_eq!(ctx.capacity(), initial_capacity);
    }

    #[test]
    fn shrink_to_fit_reduces_capacity() {
        let mut ctx = RasterContext::with_capacity(10000);
        assert!(ctx.capacity() >= 10000);
        ctx.shrink_to_fit();
        // After shrinking an empty context, capacity should be 0 or minimal
        assert!(ctx.capacity() < 100);
    }

    #[test]
    fn capacity_grows_with_usage() {
        let mut ctx = RasterContext::new();
        assert_eq!(ctx.capacity(), 0);
        let field = create_test_field(100, 100);
        let opt = default_options();
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok(), "Render failed: {:?}", result.err());
        // Capacity should have grown to accommodate the field
        assert!(ctx.capacity() >= 10000);
    }

    /* ============ Basic rendering tests ================================== */
    #[test]
    fn render_small_field_succeeds() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(10, 10);
        let opt = options_with_dims(10, 10);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok(), "Render failed: {:?}", result.err());
        let img = result.unwrap();
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
    }

    #[test]
    fn render_medium_field_succeeds() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(256, 256);
        let opt = options_with_dims(256, 256);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok(), "Render failed: {:?}", result.err());
        let img = result.unwrap();
        assert_eq!(img.width(), 256);
        assert_eq!(img.height(), 256);
    }

    #[test]
    fn render_large_field_succeeds() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(512, 512);
        let opt = options_with_dims(512, 512);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok(), "Render failed: {:?}", result.err());
    }

    #[test]
    fn render_rectangular_field_succeeds() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(640, 480);
        let opt = options_with_dims(640, 480);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok(), "Render failed: {:?}", result.err());
        let img = result.unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
    }

    #[test]
    fn render_minimal_field_succeeds() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(1, 1);
        let opt = options_with_dims(1, 1);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok(), "Render failed: {:?}", result.err());
        let img = result.unwrap();
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
    }

    /* ============ Scaling tests ========================================== */
    #[test]
    fn render_with_linear_scale() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let mut opt = options_with_dims(100, 100);
        opt.scale = RenderScale::Linear;
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_with_log10_scale() {
        let mut ctx = RasterContext::new();
        // Use positive values for log scale
        let field = create_ranged_field(100, 100, 0.1, 100.0);
        let mut opt = options_with_dims(100, 100);
        opt.scale = RenderScale::Log10;
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_with_symlog_scale() {
        let mut ctx = RasterContext::new();
        // SymLog handles both positive and negative values
        let field = create_ranged_field(100, 100, -10.0, 10.0);
        let mut opt = options_with_dims(100, 100);
        opt.scale = RenderScale::SymLog { linthresh: 1.0 };
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    /* ============ Normalization tests ==================================== */
    #[test]
    fn render_with_explicit_vmin_vmax() {
        let mut ctx = RasterContext::new();
        let field = create_ranged_field(100, 100, 0.0, 10.0);
        let mut opt = options_with_dims(100, 100);
        opt.vmin = Some(0.0);
        opt.vmax = Some(10.0);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_with_auto_vmin_vmax() {
        let mut ctx = RasterContext::new();
        let field = create_ranged_field(100, 100, 5.0, 15.0);
        let mut opt = options_with_dims(100, 100);
        opt.vmin = None;
        opt.vmax = None;
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_with_clamping_enabled() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(50, 50);
        let mut opt = options_with_dims(50, 50);
        opt.clamp = true;
        opt.vmin = Some(0.2);
        opt.vmax = Some(0.8);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_with_clamping_disabled() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(50, 50);
        let mut opt = options_with_dims(50, 50);
        opt.clamp = false;
        opt.vmin = Some(0.2);
        opt.vmax = Some(0.8);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    /* ============ Gradient/colormap tests ================================ */
    #[test]
    fn render_with_viridis_gradient() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let mut opt = options_with_dims(100, 100);
        opt.gradient_name = "viridis".to_string();
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_with_magma_gradient() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let mut opt = options_with_dims(100, 100);
        opt.gradient_name = "magma".to_string();
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_with_custom_gradient() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let mut opt = options_with_dims(100, 100);
        opt.gradient_name = "hertzsprung".to_string();
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    /* ============ Resizing tests ========================================= */
    #[test]
    fn render_with_upscaling() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let opt = options_with_dims(200, 200);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
        let img = result.unwrap();
        assert_eq!(img.width(), 200);
        assert_eq!(img.height(), 200);
    }

    #[test]
    fn render_with_downscaling() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(200, 200);
        let opt = options_with_dims(100, 100);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
        let img = result.unwrap();
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn render_with_aspect_ratio_change() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let opt = options_with_dims(200, 100);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
        let img = result.unwrap();
        assert_eq!(img.width(), 200);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn render_without_resizing() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(128, 128);
        let opt = options_with_dims(128, 128);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
        let img = result.unwrap();
        assert_eq!(img.width(), 128);
        assert_eq!(img.height(), 128);
    }

    /* ============ Buffer reuse tests ===================================== */
    #[test]
    fn render_multiple_fields_reuses_buffers() {
        let mut ctx = RasterContext::new();
        let field1 = create_test_field(100, 100);
        let field2 = create_test_field(100, 100);
        let opt = options_with_dims(100, 100);
        let _ = ctx.render(&field1, &opt).unwrap();
        let capacity_after_first = ctx.capacity();
        let _ = ctx.render(&field2, &opt).unwrap();
        let capacity_after_second = ctx.capacity();
        // Capacity should not grow if rendering same-sized fields
        assert_eq!(capacity_after_first, capacity_after_second);
    }

    #[test]
    fn render_different_sizes_grows_buffers() {
        let mut ctx = RasterContext::new();
        let small_field = create_test_field(50, 50);
        let large_field = create_test_field(200, 200);
        let opt_small = options_with_dims(50, 50);
        let opt_large = options_with_dims(200, 200);
        let _ = ctx.render(&small_field, &opt_small).unwrap();
        let capacity_after_small = ctx.capacity();
        let _ = ctx.render(&large_field, &opt_large).unwrap();
        let capacity_after_large = ctx.capacity();
        // Capacity should grow when rendering larger field
        assert!(capacity_after_large > capacity_after_small);
    }

    #[test]
    fn render_after_clear_still_works() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let opt = options_with_dims(100, 100);
        let _ = ctx.render(&field, &opt).unwrap();
        ctx.clear();
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_after_shrink_still_works() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let opt = options_with_dims(100, 100);
        let _ = ctx.render(&field, &opt).unwrap();
        ctx.shrink_to_fit();
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    /* ============ Edge case tests ======================================== */
    #[test]
    fn render_constant_field() {
        let mut ctx = RasterContext::new();
        let field = create_constant_field(100, 100, 0.5);
        let opt = options_with_dims(100, 100);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_zero_field() {
        let mut ctx = RasterContext::new();
        let field = create_constant_field(100, 100, 0.0);
        let opt = options_with_dims(100, 100);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_field_with_negative_values() {
        let mut ctx = RasterContext::new();
        let field = create_ranged_field(100, 100, -5.0, 5.0);
        let opt = options_with_dims(100, 100);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_field_with_large_values() {
        let mut ctx = RasterContext::new();
        let field = create_ranged_field(100, 100, 1e6, 1e9);
        let opt = options_with_dims(100, 100);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_field_with_small_values() {
        let mut ctx = RasterContext::new();
        let field = create_ranged_field(100, 100, 1e-9, 1e-6);
        let opt = options_with_dims(100, 100);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    /* ============ Invalid color handling ================================= */
    #[test]
    fn render_field_with_nan_values() {
        let mut ctx = RasterContext::new();
        let w = NonZeroU32::new(10).unwrap();
        let h = NonZeroU32::new(10).unwrap();
        let mut values = vec![0.5; 100];
        values[50] = f64::NAN;
        let field = RasterField::from_vec(w, h, values).unwrap();
        let opt = options_with_dims(10, 10);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_field_with_inf_values() {
        let mut ctx = RasterContext::new();
        let w = NonZeroU32::new(10).unwrap();
        let h = NonZeroU32::new(10).unwrap();
        let mut values = vec![0.5; 100];
        values[25] = f64::INFINITY;
        values[75] = f64::NEG_INFINITY;
        let field = RasterField::from_vec(w, h, values).unwrap();
        let opt = options_with_dims(10, 10);
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    /* ============ Error handling tests =================================== */
    #[test]
    fn render_fails_with_invalid_vmin_vmax() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let mut opt = options_with_dims(100, 100);
        opt.vmin = Some(10.0);
        opt.vmax = Some(5.0); // vmax < vmin
        let result = ctx.render(&field, &opt);
        assert!(result.is_err());
    }

    /* ============ Integration tests ====================================== */
    #[test]
    fn complete_rendering_pipeline() {
        let mut ctx = RasterContext::new();
        let field = create_ranged_field(256, 256, 0.0, 100.0);
        let mut opt = options_with_dims(512, 512);
        opt.scale = RenderScale::Log10;
        opt.vmin = Some(0.1);
        opt.vmax = Some(100.0);
        opt.clamp = true;
        opt.gradient_name = "viridis".to_string();
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
        let img = result.unwrap();
        assert_eq!(img.width(), 512);
        assert_eq!(img.height(), 512);
    }

    #[test]
    fn render_sequence_with_different_configs() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        // First render with linear scale
        let mut opt1 = options_with_dims(100, 100);
        opt1.scale = RenderScale::Linear;
        assert!(ctx.render(&field, &opt1).is_ok());
        // Second render with log scale
        let mut opt2 = options_with_dims(100, 100);
        opt2.scale = RenderScale::Log10;
        opt2.vmin = Some(0.1);
        opt2.vmax = Some(1.0);
        assert!(ctx.render(&field, &opt2).is_ok());
        // Third render with different gradient
        let mut opt3 = options_with_dims(100, 100);
        opt3.gradient_name = "magma".to_string();
        assert!(ctx.render(&field, &opt3).is_ok());
    }

    #[test]
    fn render_multiple_fields_consecutively() {
        let mut ctx = RasterContext::new();
        let opt = options_with_dims(100, 100);   
        for i in 0..10 {
            let field = create_ranged_field(100, 100, i as f64, (i + 1) as f64);
            let result = ctx.render(&field, &opt);
            assert!(result.is_ok(), "Render #{} failed", i);
        }
    }

    /* ============ Performance-related tests ============================== */
    #[test]
    fn preallocated_context_avoids_reallocation() {
        let size = 512 * 512;
        let ctx = RasterContext::with_capacity(size);
        // Capacity should be sufficient from the start
        assert!(ctx.capacity() >= size);
    }

    #[test]
    fn context_grows_appropriately() {
        let mut ctx = RasterContext::new();
        // Start with small field
        let small_field = create_test_field(10, 10);
        let opt_small = options_with_dims(10, 10);
        let _ = ctx.render(&small_field, &opt_small).unwrap();
        let cap_small = ctx.capacity();
        // Render medium field
        let medium_field = create_test_field(100, 100);
        let opt_medium = options_with_dims(100, 100);
        let _ = ctx.render(&medium_field, &opt_medium).unwrap();
        let cap_medium = ctx.capacity();   
        // Render large field
        let large_field = create_test_field(500, 500);
        let opt_large = options_with_dims(500, 500);
        let _ = ctx.render(&large_field, &opt_large).unwrap();
        let cap_large = ctx.capacity();
        
        assert!(cap_medium > cap_small);
        assert!(cap_large > cap_medium);
    }

    /* ============ Special configuration tests ============================ */

    #[test]
    fn render_with_invalid_rgba_handling() {
        let mut ctx = RasterContext::new();
        let field = create_test_field(100, 100);
        let mut opt = options_with_dims(100, 100);
        opt.invalid_rgba = [255, 0, 255, 128]; // Magenta with alpha
        
        let result = ctx.render(&field, &opt);
        assert!(result.is_ok());
    }

    #[test]
    fn render_preserves_image_dimensions() {
        let mut ctx = RasterContext::new();

        let test_cases = vec![
            (100, 100),
            (256, 128),
            (128, 256),
            (640, 480),
            (1, 1000),
            (1000, 1),
        ];

        for (w, h) in test_cases {
            let field = create_test_field(w, h);
            let opt = options_with_dims(w, h);       
            let img = ctx.render(&field, &opt).unwrap();
            assert_eq!(img.width(), w, "Width mismatch for {}x{}", w, h);
            assert_eq!(img.height(), h, "Height mismatch for {}x{}", w, h);
        }
    }
}
