// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Free-form lenses — pixel-grid and Voronoi-mesh convergence, each a
//! [`LensModel`](crate::lens::LensModel).
//!
//! The convergence κ is carried per cell rather than by a closed form, so the
//! deflection is summed over cells via a selectable [`FieldMethod`].
//!
//! * `model` — [`FreeFormLens`] and the [`FieldMethod`] (analytic kernel vs a
//!   `fastsum` backend) shared by both representations.
//! * `pixel` — [`PixelLens`], κ on a structured `PixelGrid`.
//! * `mesh` — [`MeshLens`], κ on an unstructured Voronoi `MeshGrid`.
//! * `source` — [`MassElement`], the per-cell monopole a `fastsum` backend sums.

pub mod mesh;
pub mod model;
pub mod pixel;
pub mod source;

pub use mesh::MeshLens;
pub use model::{FieldMethod, FreeFormLens};
pub use pixel::PixelLens;
pub use source::MassElement;
