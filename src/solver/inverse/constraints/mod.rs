// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-image constraints: turning observed images into a convex polytope.
//!
//! The free-form κ basis makes every point-image observable *linear* in the cell masses,
//! so the feasible set of mass distributions is a polytope `A x ≤ b, C x = d` in the
//! unknown `x = (m, {β_s})` — the cell masses together with each source's (free) position.
//!
//! `builder` assembles the [`ConstraintSet`] from a
//! [`KappaBasis`](crate::solver::basis::KappaBasis) and a set of
//! [`PointImageSystem`](crate::solver::image::PointImageSystem)s (the observed-data unit,
//! defined in [`image`](crate::solver::image)); the per-cell ψ and deflection coefficients come
//! from [`FieldCoefficients`](crate::solver::basis::FieldCoefficients).

pub mod builder;
mod data;
mod priors;

pub use builder::{ConstraintConfig, ConstraintSet, Symmetry};
