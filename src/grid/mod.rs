// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens_forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Grid structures for lens modelling

mod point;
mod grid2d;
mod uniform;
mod adaptive;
mod mesh;

pub use point::Point2D;
pub use grid2d::Grid2D;
pub use grid2d::BilinearKernel;
pub use uniform::PixelGrid;
// pub use adaptive::AdaGrid;

