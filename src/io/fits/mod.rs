// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! A pure-Rust, from-scratch FITS image reader and writer.
//!
//! Implements the image subset of the FITS standard with no `cfitsio` binding
//! and no extra crates. A file is a sequence of 2880-byte blocks; the low-level
//! byte handling and error types live here, with the higher layers added in
//! sibling modules.
//!
//! * [`block`] — the 2880-byte block, a big-endian byte cursor/sink, padding.
//! * [`error`] — the context-carrying [`FitsError`], bridged to `LensForgeError`.
//! * [`card`] — one 80-byte card ([`Card`]/[`CardValue`]): parse and encode.
//! * [`header`] — a [`Header`] of cards: parse to `END`, encode block-padded.
//! * [`data`]/[`image`] — [`FitsData`] element buffers and a [`FitsImage`]
//!   (`BITPIX`, axes, scaling), with the `f64`/[`grid`](crate::grid) bridge.
//!
//! Physical values follow `physical = BZERO + BSCALE·array`, with an integer
//! `BLANK` mapping to `NaN` in the `f64` view. Arrays keep the FITS-native
//! bottom-up orientation (no row flip); axis order is `NAXIS1` (x) fastest,
//! matching `ndarray` `[NAXIS2, NAXIS1] = [j, i]`.

pub mod block;
pub mod card;
pub mod data;
pub mod error;
pub mod header;
pub mod image;

pub use card::{Card, CardValue};
pub use data::FitsData;
pub use error::FitsError;
pub use header::Header;
pub use image::FitsImage;
