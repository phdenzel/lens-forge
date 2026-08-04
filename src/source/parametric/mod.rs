// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Parametric surface-brightness profiles, each a closed-form [`SourceModel`](crate::source::SourceModel).

pub mod gaussian;
pub mod sersic;

pub use gaussian::Gaussian;
pub use sersic::Sersic;
