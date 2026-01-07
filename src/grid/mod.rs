// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Grid structures for lens modelling

pub type FieldArray2D = ndarray::Array2<f64>;
pub type VectorField2D = (FieldArray2D, FieldArray2D);
pub type ComplexArray1D = ndarray::Array1<num::complex::Complex<f64>>;
pub type ComplexFieldArray2D = ndarray::Array2<num::complex::Complex<f64>>;
pub type FourierSymbols = (ComplexArray1D, ComplexArray1D);

mod adaptive;
mod grid2d;
mod mesh;
mod point;
mod uniform;

pub use grid2d::BilinearKernel;
pub use grid2d::Grid2D;
pub use point::Point2D;
pub use uniform::PixelGrid;
// pub use adaptive::AdaGrid;
