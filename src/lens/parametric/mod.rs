// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Parametric mass profiles, each a closed-form [`LensModel`](crate::lens::LensModel).
//!
//! * `sis` — [`SIS`], the singular isothermal sphere.
//! * `sie` — [`SIE`], its elliptical generalization (singular isothermal ellipsoid).
//! * `nfw` — [`NFW`], the Navarro–Frenk–White dark-matter halo.
//! * `point_mass` — [`PointMass`], a single Schwarzschild point deflector.

pub mod nfw;
pub mod point_mass;
pub mod sie;
pub mod sis;

pub use nfw::NFW;
pub use point_mass::PointMass;
pub use sie::SIE;
pub use sis::SIS;
