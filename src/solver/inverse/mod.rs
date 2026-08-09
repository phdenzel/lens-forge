// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended-image inversion and re-weighting: the phase-2 bridge from point images to arcs.
//!
//! The point-image [`ensemble`](crate::solver::ensemble) is a proposal set; extended (arc)
//! data re-weight it. For a fixed member the source is a linear least-squares solve, and its
//! forward image-plane fit is a Gaussian-evidence importance weight.
//!
//! * `backproject` — Stage 1: ray-trace the arc mask to source-plane generators, build the
//!   per-member source mesh, and the cheap dispersion cull.
//! * `semilinear` — Warren & Dye source inversion `(Fᵀ W F + λ R) s = Fᵀ W d` and its mesh
//!   regularizer.
//! * `reweight` — Stage 2: per-member forward χ² → importance weights + effective sample size.

pub mod backproject;
pub mod reweight;
pub mod semilinear;

pub use reweight::{image_chi_squared, importance_weights, reweight_ensemble};
pub use semilinear::{invert_source, Regularizer};
