// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Rendering interfaces and shared configuration for implemented data structures.
//!
//! Two complementary rendering strategies are provided:
//! * **raster** high-performance path for uniform pixel rendering
//! * **mesh** path for explicit rendering of adaptive and irregular (polygonal cell) grids
mod mesh;
mod raster;
use crate::viz::gradients::get_gradient;
use crate::viz::scale::RenderScale;
use crate::{LensForgeError, Result};
use colorgrad::Gradient;
pub use mesh::MeshField; //, texturize, texturize_with};
use mesh::ToMesh;
use raster::ToRaster;
pub use raster::{RasterField, RasterizeParams, rasterize, rasterize_with, validate_raster};
use std::num::NonZeroU32;

/// Coordinate bounds for rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderBounds {
    /// Lower-left corner (xmin, ymin)
    pub min: [f64; 2],
    /// Upper-right corner (xmax, ymax)
    pub max: [f64; 2],
}

impl RenderBounds {
    /// Returns `true` if bounds are finite and non-degenerate
    pub fn is_valid(&self) -> bool {
        self.min[0].is_finite()
            && self.min[1].is_finite()
            && self.max[0].is_finite()
            && self.max[1].is_finite()
            && (self.max[0] > self.min[0])
            && (self.max[1] > self.min[1])
    }
}

/// Shared rendering configuration.
#[derive(Clone, Debug)]
pub struct RenderOptions {
    /// Output width in pixels
    pub width: NonZeroU32,
    /// Output height in pixels
    pub height: NonZeroU32,
    /// Optional downsampling resolution
    pub downsample_to: Option<(NonZeroU32, NonZeroU32)>,
    /// Value scaling applied prior to normalization
    pub scale: RenderScale,
    /// Explicit lower normalization bound (in scaled space)
    pub vmin: Option<f64>,
    pub vmax: Option<f64>,
    /// Clamp normalized values to `[0, 1]`
    pub clamp: bool,
    /// RGBA color used for invalid samples (NaN/inf, or log-invalid values)
    pub invalid_rgba: [u8; 4],
    /// Whether to draw cell edges (only relevant for cell-based rendering)
    pub show_edges: bool,
    /// RGBA for cell edges, if `show_edges == true`
    pub edge_rgba: [u8; 4],
    /// Gradient used for color mapping of normalized values in `[0, 1]`
    pub gradient_name: String,
    /// Optional; if `Some(n)`, draw a raster grid overlay every `n` pixels
    pub grid_overlay_every: Option<u32>,
    /// RGBA color for raster grid overlay lines
    pub grid_overlay_rgba: [u8; 4],
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: NonZeroU32::new(512).unwrap(),
            height: NonZeroU32::new(512).unwrap(),
            downsample_to: None,
            scale: RenderScale::Linear,
            vmin: None,
            vmax: None,
            clamp: true,
            invalid_rgba: [0, 0, 0, 0],
            show_edges: false,
            edge_rgba: [255, 255, 255, 32],
            gradient_name: "viridis".to_string(),
            grid_overlay_every: None,
            grid_overlay_rgba: [255, 255, 255, 24],
        }
    }
}

impl RenderOptions {
    #[inline]
    pub fn width_u32(&self) -> u32 {
        self.width.get()
    }
    #[inline]
    pub fn height_u32(&self) -> u32 {
        self.height.get()
    }
    #[inline]
    pub fn downsample_u32(&self) -> Option<(u32, u32)> {
        self.downsample_to.map(|(w, h)| (w.get(), h.get()))
    }
    #[inline]
    pub fn gradient(&self) -> Box<dyn Gradient> {
        get_gradient(&self.gradient_name)
    }
}

/// High-level: a thing we can render
pub enum RenderInput<'a> {
    Raster(&'a dyn ToRaster),
    Mesh(&'a dyn ToMesh),
}

/// Validate common option invariants.
///
/// # Arguments
/// * `opt` - Shared rendering configuration
pub fn validate_options(opt: &RenderOptions) -> Result<()> {
    if let RenderScale::SymLog { linthresh } = opt.scale {
        if !linthresh.is_finite() || linthresh <= 0.0 {
            return Err(LensForgeError::invalid_option(
                "SymLog linthresh must be finite and > 0",
            ));
        }
    }
    Ok(())
}
