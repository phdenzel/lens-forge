// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-image world: the stationary points of the Fermat surface.
//!
//! * `images` — the [`ImageFinder`](images::ImageFinder) trait and its
//!   discretization-based backend: all images of a point source, each with
//!   magnification, parity and Fermat potential.
//!
//! The arrival-time surface and time delays (`fermat`, `delays`) follow; the
//! inverse point-image constraints reuse the same [`Image`](crate::solver::image::Image).

pub mod images;
