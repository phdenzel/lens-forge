// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Visualization: colour mapping, scaling, rasterization, and a live viewer.
//!
//! The pipeline turns a scalar field into pixels: scale the values, normalize to
//! `[0, 1]`, map them through a colour gradient, and rasterize (with optional resize).
//!
//! * `scale` — value scaling ([`RenderScale`]: linear/log10/symlog), [`Normalize`],
//!   and [`auto_range`].
//! * `gradients` — named colour gradients ([`get_gradient`]).
//! * `colorize` — apply a gradient to normalized values.
//! * `render` — rasterize a field to an image ([`rasterize`]; uniform-grid and mesh paths).
//! * `view` — the live `winit`/`wgpu` interactive viewer.
//! * `io` — save/load image artefacts.
pub mod colorize;
mod gradients;
pub mod io;
pub mod render;
pub mod scale;
pub mod view;

pub use gradients::get_gradient;
pub use render::{RasterField, RasterizeParams, rasterize, rasterize_with};
pub use scale::{Normalize, RenderScale, auto_range};

const MIN_CHUNK: usize = 1024;
