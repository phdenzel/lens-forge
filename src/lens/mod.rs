// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens models, all behind the common [`LensModel`] trait.
//!
//! Every lens exposes the same jet (potential ψ, deflection α, Hessian H) at a
//! point, so the solvers stay agnostic to how the mass is represented.
//!
//! * `model` — the [`LensModel`] trait (and its [`LensJet`]) every lens implements.
//! * `parametric` — closed-form analytic profiles: [`SIS`], [`SIE`], [`NFW`], [`PointMass`].
//! * `freeform` — pixel-grid and Voronoi-mesh convergence lenses ([`PixelLens`], [`MeshLens`]).
//! * `external` — external-field perturbations such as [`ExternalShear`] (κ = 0).
//! * `composite` — [`CompositeLens`], the additive sum of several components.
//! * `profile` — [`RadialProfile`], concentric-annulus binning of a mass map.

pub mod composite;
pub mod external;
pub mod freeform;
pub mod model;
pub mod parametric;
pub mod profile;

pub use composite::CompositeLens;
pub use external::ExternalShear;
pub use freeform::{FieldMethod, FreeFormLens, MassElement, MeshLens, PixelLens};
pub use model::{LensJet, LensModel};
pub use parametric::{NFW, PointMass, SIE, SIS};
pub use profile::RadialProfile;
