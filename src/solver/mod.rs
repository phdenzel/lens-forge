// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens-equation solvers — a bag of interchangeable variants.
//!
//! * `image` — the [`Image`](image::Image)/[`Parity`](image::Parity) currency
//!   shared by the forward image finders and the inverse point-image constraints.
//! * `mapping` — the lens mapping itself: ray-tracing, magnification, critical
//!   curves and caustics; quantities derived from the deflection field alone.
//!
//! Forward first; the inverse layers (point-image `extrema`, extended-image
//! `extended`) follow.

pub mod image;
pub mod mapping;
