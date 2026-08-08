// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-image world: the stationary points of the Fermat surface.
//!
//! * `images` — the [`ImageFinder`](images::ImageFinder) trait and its
//!   discretization-based backend: all images of a point source, each with
//!   magnification, parity and Fermat potential.
//! * `fermat` — the Fermat (arrival-time) surface τ(θ; β) and the point-source
//!   total magnification Σ|μ_i|.
//! * `delays` — arrival-time differences between images, dimensionless and
//!   cosmology-scaled to physical time delays.
//!
//! The inverse point-image constraints reuse the same
//! [`Image`](crate::solver::image::Image).

pub mod delays;
pub mod fermat;
pub mod images;
