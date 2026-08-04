// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Visualization: colour mapping, scaling, rasterization, and a live viewer.
mod colorize;
mod gradients;
pub mod io;
pub mod render;
pub mod scale;
pub mod view;

pub use gradients::get_gradient;
pub use render::{RasterField, RasterizeParams, rasterize, rasterize_with};
pub use scale::{Normalize, RenderScale, auto_range};

const MIN_CHUNK: usize = 1024;
