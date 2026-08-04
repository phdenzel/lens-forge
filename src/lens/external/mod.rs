// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! External-field (κ = 0) perturbations: the environment's contribution to the
//! deflection, not a localized mass. Each is a closed-form
//! [`LensModel`](crate::lens::LensModel).

pub mod external_shear;

pub use external_shear::ExternalShear;
