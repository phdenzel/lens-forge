// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
mod gradients;
//mod colorize;
pub mod io;
pub mod render;
pub mod scale;

pub use gradients::get_gradient;
pub use render::{MeshField, RasterField, RasterizeParams, rasterize, rasterize_with};
pub use scale::{Normalize, RenderScale};
