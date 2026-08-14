// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Loading and saving external data as lens-forge field types.
//!
//! The observational inputs a lens model needs — science images, PSFs, masks —
//! and the fields it produces — simulated images, convergence maps,
//! reconstructed sources — are naturally stored on disk. This module bridges
//! those files to the crate's [`grid`](crate::grid) types.
//!
//! Its first (and currently only) member is [`fits`], a pure-Rust, from-scratch
//! reader and writer for the image subset of the FITS standard. Further formats
//! (HDF5, plain ASCII) may join it later under the same roof.

pub mod fits;
