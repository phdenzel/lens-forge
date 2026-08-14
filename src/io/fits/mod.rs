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

pub mod block;
pub mod card;
pub mod error;

pub use card::{Card, CardValue};
pub use error::FitsError;
