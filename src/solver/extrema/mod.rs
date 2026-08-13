// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-source interpretation: the stationary points of the Fermat surface.
//!
//! * `images` — the [`ImageFinder`](images::ImageFinder) trait and its backend: it drives a
//!   [`MapDiscretization`](crate::solver::basis::MapDiscretization) for seeds, polishes them
//!   with `newton`, and returns all images of a point source, each with magnification, parity
//!   and Fermat potential; plus the point-source total magnification Σ|μ_i|.
//! * `newton` — Newton's method on the lens map (Jacobian = A), refining a seed to an image.
//! * `fermat` — the Fermat (arrival-time) surface τ(θ; β), the Roche potential and its contours.
//! * `delays` — arrival-time differences between images, dimensionless and
//!   cosmology-scaled to physical time delays.
//!
//! The inverse point-image constraints reuse the same
//! [`Image`](crate::solver::image::Image).

pub mod delays;
pub mod fermat;
pub mod images;
pub mod newton;
