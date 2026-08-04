// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Rendering interfaces and shared configuration for implemented data structures.
//!
//! Two complementary rendering strategies are provided:
//! * **raster** high-performance path for uniform pixel rendering
//! * **mesh** path for explicit rendering of adaptive and irregular (polygonal cell) grids
pub mod context;
mod mesh;
pub mod raster;
use crate::viz::gradients::get_gradient;
use crate::viz::scale::RenderScale;
use crate::{LensForgeError, Result};
use colorgrad::Gradient;
use image::imageops::FilterType;
pub use raster::{RasterField, RasterizeParams, rasterize, rasterize_with, validate_raster};
use std::num::NonZeroU32;

/// Shared rendering configuration.
#[derive(Clone, Debug)]
pub struct RenderOptions {
    /// Output width in pixels (if different from field, image is resized)
    pub width: NonZeroU32,
    /// Output height in pixels (if different from field, image is resized)
    pub height: NonZeroU32,
    /// Value scaling applied prior to normalization
    pub scale: RenderScale,
    /// Explicit lower normalization bound (in scaled space)
    pub vmin: Option<f32>,
    /// Explicit higher normalization bound (in scaled space)
    pub vmax: Option<f32>,
    /// Clamp normalized values to `[0, 1]`
    pub clamp: bool,
    /// RGBA colour used for invalid samples (NaN/inf, or log-invalid values)
    pub invalid_rgba: [u8; 4],
    /// Filter name for resampling (one of nearest, triangle, catmullrom, gaussian, lanczos3)
    pub sampling_type: String,
    /// Whether to draw cell edges (only relevant for cell-based rendering)
    pub show_edges: bool,
    /// RGBA for cell edges, if `show_edges == true`
    pub edge_rgba: [u8; 4],
    /// Gradient used for colour mapping of normalized values in `[0, 1]`
    pub gradient_name: String,
    /// Element count above which the `Log10`/`SymLog` scaling and colormap
    /// stages switch to the parallel path. `Linear` scaling and normalization
    /// always run serial — parallel never wins for them (see
    /// `benches/viz_scale.rs`).
    pub threshold_parallel: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: NonZeroU32::new(512).unwrap(),
            height: NonZeroU32::new(512).unwrap(),
            scale: RenderScale::Linear,
            vmin: None,
            vmax: None,
            clamp: true,
            invalid_rgba: [0, 0, 0, 0],
            sampling_type: "CatmullRom".to_string(),
            show_edges: false,
            edge_rgba: [255, 255, 255, 32],
            gradient_name: "viridis".to_string(),
            threshold_parallel: 30_000,
        }
    }
}

impl RenderOptions {
    /// Output width as a plain `u32`.
    #[inline]
    pub fn width_u32(&self) -> u32 {
        self.width.get()
    }
    /// Output height as a plain `u32`.
    #[inline]
    pub fn height_u32(&self) -> u32 {
        self.height.get()
    }
    /// Resolve the configured colour gradient by name.
    #[inline]
    pub fn gradient(&self) -> Box<dyn Gradient + Send + Sync> {
        get_gradient(&self.gradient_name)
    }
    /// Resampling filter for the configured `sampling_type`, or `None` if unknown.
    #[inline]
    pub fn filter(&self) -> Option<FilterType> {
        match self.sampling_type.to_lowercase().as_str() {
            "nearest" => Some(FilterType::Nearest),
            "triangle" => Some(FilterType::Triangle),
            "catmullrom" => Some(FilterType::CatmullRom),
            "gaussian" => Some(FilterType::Gaussian),
            "lanczos3" => Some(FilterType::Lanczos3),
            _ => None,
        }
    }

    /// Validate common option invariants.
    pub fn validate(&self) -> Result<()> {
        if self.vmin > self.vmax {
            return Err(LensForgeError::invalid_option(
                "It is required to have vmin < vmax",
            ));
        }
        if let RenderScale::SymLog { linthresh } = self.scale {
            if !linthresh.is_finite() || linthresh <= 0.0 {
                return Err(LensForgeError::invalid_option(
                    "SymLog linthresh must be finite and > 0",
                ));
            }
        }
        Ok(())
    }
}

/// Validate common option invariants.
///
/// # Arguments
/// * `opt` - Shared rendering configuration
pub fn validate_options(opt: &RenderOptions) -> Result<()> {
    opt.validate()
}

/* ============ Unit tests ================================================= */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_maps_every_supported_name() {
        let cases = [
            ("nearest", FilterType::Nearest),
            ("Triangle", FilterType::Triangle),
            ("CatmullRom", FilterType::CatmullRom),
            ("GAUSSIAN", FilterType::Gaussian),
            ("lanczos3", FilterType::Lanczos3),
        ];
        for (name, expected) in cases {
            let mut opt = RenderOptions::default();
            opt.sampling_type = name.to_string();
            assert_eq!(opt.filter(), Some(expected));
        }
        // Unknown filter name yields None.
        let mut opt = RenderOptions::default();
        opt.sampling_type = "bogus".to_string();
        assert_eq!(opt.filter(), None);
    }

    #[test]
    fn dims_and_gradient_getters() {
        let opt = RenderOptions::default();
        assert_eq!(opt.width_u32(), 512);
        assert_eq!(opt.height_u32(), 512);
        // gradient() resolves the configured name without panicking.
        let _ = opt.gradient();
    }

    #[test]
    fn validate_rejects_bad_ranges_and_symlog() {
        // vmin > vmax is rejected (via both the method and the free function).
        let mut opt = RenderOptions::default();
        opt.vmin = Some(1.0);
        opt.vmax = Some(0.0);
        assert!(opt.validate().is_err());
        assert!(validate_options(&opt).is_err());

        // Non-positive / non-finite SymLog linthresh is rejected.
        for bad in [0.0, -1.0, f32::NAN] {
            let opt = RenderOptions {
                scale: RenderScale::SymLog { linthresh: bad },
                ..RenderOptions::default()
            };
            assert!(opt.validate().is_err());
        }

        // A valid configuration passes.
        assert!(RenderOptions::default().validate().is_ok());
    }
}
