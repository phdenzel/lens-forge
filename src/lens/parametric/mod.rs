// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Parametric mass profiles, each a closed-form [`LensModel`](crate::lens::LensModel).

pub mod point_mass;
pub mod sie;
pub mod sis;

pub use point_mass::PointMass;
pub use sie::SIE;
pub use sis::SIS;
