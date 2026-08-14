// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The pixel-or-mesh representation seams the solvers are generic over, and their realizations.
//!
//! Every solver layer plugs into one of three seams so a structured [`PixelGrid`] and an
//! unstructured [`MeshGrid`] stay interchangeable:
//!
//! * `lens` — [`KappaBasis`]: lens-plane κ → field, and its precomputed
//!   [`FieldCoefficients`].
//! * `source` — [`SourceBasis`]: the reconstructed source plane.
//! * `discretization` — [`MapDiscretization`]: a discretized
//!   θ↔β map, realized by [`MapTessellation`] (mesh) and
//!   [`MapPixellation`] (pixel).
//!
//! [`PixelGrid`]: crate::grid::PixelGrid
//! [`MeshGrid`]: crate::grid::MeshGrid

pub mod discretization;
pub mod lens;
pub mod pixellate;
pub mod source;
pub mod tessellate;

pub use discretization::MapDiscretization;
pub use lens::{FieldCoefficients, KappaBasis};
pub use pixellate::MapPixellation;
pub use source::{SourceBasis, SourceConfig};
pub use tessellate::MapTessellation;
