// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The forward map: (lens, source) → observables.
//!
//! * `mapping` — quantities read off the deflection field alone (source-independent):
//!   ray-tracing β = θ − α(θ), the Jacobian det A, magnification μ = 1/det A, convergence
//!   κ = ½∇²ψ, and the critical curves and their caustics.
//! * `extrema` — the point-source world: solve β = θ − α(θ) for every image (`ImageFinder`),
//!   classify each on the Fermat arrival-time surface, form time delays.
//! * `extended` — the extended-source world: the lensed surface-brightness field
//!   I(θ) = S(θ − α(θ)) and its sparse linear operator F (source → image).
//! * `model` — the [`ForwardModel`] composer tying the three into observables.

pub mod extended;
pub mod extrema;
pub mod mapping;
pub mod model;

pub use model::ForwardModel;
