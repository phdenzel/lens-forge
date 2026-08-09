// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens-equation solvers — a bag of interchangeable variants.
//!
//! * `image` — the [`Image`](image::Image)/[`Parity`](image::Parity) currency
//!   shared by the forward image finders and the inverse point-image constraints.
//! * `mapping` — the lens mapping itself: ray-tracing, magnification, critical
//!   curves and caustics; quantities derived from the deflection field alone.
//! * `extrema` — stationary points of the Fermat surface: point-source image
//!   finding, the arrival-time surface and time delays.
//! * `extended` — extended-source imaging: the lensed surface-brightness field.
//! * `constraints` — the inverse point-image polytope: images + a κ basis → `A x ≤ b`,
//!   `C x = d` over the cell masses and source positions.
//!
//! Forward first; the inverse layers grow in place beside these forward primitives.

pub mod constraints;
pub mod extended;
pub mod extrema;
pub mod image;
pub mod mapping;
