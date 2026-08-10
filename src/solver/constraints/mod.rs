// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-image constraints: turning observed images into a convex polytope.
//!
//! The free-form κ basis makes every point-image observable *linear* in the cell masses,
//! so the feasible set of mass distributions is a polytope `A x ≤ b, C x = d` in the
//! unknown `x = (m, {β_s})` — the cell masses together with each source's (free) position.
//!
//! * `influence` — the [`KappaBasis`](influence::KappaBasis) bridge (pixel or mesh) and the
//!   [`FieldCoefficients`](influence::FieldCoefficients): per-cell ψ and deflection coefficients
//!   precomputed at the image positions, once per basis geometry.
//! * `builder` — assembles the [`ConstraintSet`](builder::ConstraintSet) from a basis and a
//!   set of [`PointImageSystem`](crate::solver::image::PointImageSystem)s (the observed-data
//!   currency, defined in [`image`](crate::solver::image)).

pub mod builder;
pub mod influence;

pub use builder::{ConstraintConfig, ConstraintSet};
pub use influence::{FieldCoefficients, KappaBasis};
