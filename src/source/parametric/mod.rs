// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Parametric surface-brightness profiles, each a closed-form [`SourceModel`](crate::source::SourceModel).
//!
//! * `sersic` — [`Sersic`], the Sérsic R^(1/n) profile (galaxies, bulges).
//! * `gaussian` — [`Gaussian`], an elliptical Gaussian blob.

pub mod gaussian;
pub mod sersic;

pub use gaussian::Gaussian;
pub use sersic::Sersic;
