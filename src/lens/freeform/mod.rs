// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Free-form lenses — pixel-grid and Voronoi-mesh convergence, each a
//! [`LensModel`](crate::lens::LensModel).

pub mod mesh;
pub mod pixel;

pub use mesh::MeshLens;
pub use pixel::PixelLens;
