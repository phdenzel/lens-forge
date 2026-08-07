// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens models, all behind the common [`LensModel`] trait.

pub mod composite;
pub mod external;
pub mod freeform;
pub mod model;
pub mod parametric;

pub use composite::CompositeLens;
pub use external::ExternalShear;
pub use freeform::{FieldMethod, FreeFormLens, MassElement, MeshLens, PixelLens};
pub use model::{LensJet, LensModel};
pub use parametric::{NFW, PointMass, SIE, SIS};
