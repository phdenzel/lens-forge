// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Sampling the point-image polytope: the degenerate-solution ensemble and its statistics.
//!
//! The [`ConstraintSet`](crate::solver::constraints::ConstraintSet) is a convex polytope of
//! admissible mass distributions. This module draws a uniform (flat-in-mass) sample of it
//! and reduces the sample to physical summaries.
//!
//! * `sampler` — [`HitAndRun`](sampler::HitAndRun): equality-null-space reduction, a
//!   Chebyshev-centre seed, and parallel hit-and-run walkers producing an
//!   [`Ensemble`](sampler::Ensemble).
//! * `stats` — mean/variance κ maps, the mass-sheet level, the radial mass profile
//!   exposing the monopole degeneracy, and the effective sample size.

pub mod sampler;
pub mod stats;

pub use sampler::{Ensemble, EnsembleMember, HitAndRun};
pub use stats::RadialProfile;
