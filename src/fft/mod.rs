// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Representation-agnostic 2D FFT engine over `ndarray` complex arrays.
//!
//! * `transform` — forward/inverse 2D FFT (row-then-column, `phastft` backend)
//!   and its real↔complex buffer helpers.
//! * `shift` — `fftshift`/`ifftshift` quadrant swaps.
//!
//! [`PixelGrid`](crate::grid::PixelGrid) re-exposes these as inherent methods.

mod shift;
mod transform;

pub(crate) use shift::{fftshift, ifftshift};
pub(crate) use transform::{fft2d, ifft2d, ifft2d_unchecked};
