// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Source models, all behind the common [`SourceModel`] trait.
//!
//! Parametric profiles and a free-form field representation share the trait; all
//! work in scaled/dimensionless angular units (β a source-plane position).
//!
//! * `model` — the [`SourceModel`] trait every source implements.
//! * `parametric` — closed-form profiles: [`Sersic`], [`Gaussian`].
//! * `field` — [`SourceField`], a free-form pixelated/meshed source plane.

pub mod field;
pub mod model;
pub mod parametric;

pub use field::SourceField;
pub use model::SourceModel;
pub use parametric::{Gaussian, Sersic};
