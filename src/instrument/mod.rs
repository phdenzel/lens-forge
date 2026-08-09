// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The instrument: how the sky is observed, orthogonal to the lens/source model.
//!
//! The astrophysical model ([`lens`](crate::lens) + [`source`](crate::source)) is
//! the sky; how it is measured — the point-spread function, the noise, the mask —
//! is a distinct concern shared by forward imaging and every inverse likelihood.
//!
//! * `psf` — point-spread functions, compiled to a [`grid`](crate::grid) `Kernel`
//!   and applied through [`FieldOps::convolve`](crate::grid::FieldOps::convolve).
//! * `noise` — the data covariance C: per-cell inverse-variance weighting for the
//!   likelihood, and Gaussian noise realizations for mock data.
//! * `mask` — the pixel mask / weight map: which cells enter the fit, composing
//!   multiplicatively with the noise weight.

pub mod mask;
pub mod noise;
pub mod psf;
