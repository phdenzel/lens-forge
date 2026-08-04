// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Cosmology: the physical-scales boundary layer.
//!
//! Turns redshifts + a background model into the angular-diameter distances and lensing
//! quantities the otherwise dimensionless [`lens`](crate::lens)/[`source`](crate::source)
//! core lacks. [`units`] carries SI constants, astrophysical units, and Planckian-unit
//! conversions.

pub mod flat_lcdm;
pub mod model;
pub mod units;

pub use flat_lcdm::FlatLCDM;
pub use model::Cosmology;
