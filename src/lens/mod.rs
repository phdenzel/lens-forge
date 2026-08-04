// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens models — parametric, external-field, and free-form, all behind [`LensModel`].

pub mod mesh;
pub mod model;

pub use mesh::{MeshLens, VoronoiCell};
pub use model::{LensJet, LensModel};
