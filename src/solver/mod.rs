// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens-equation solvers: the forward map (model → observables) and the inverse
//! reconstruction (observables → mass + source), over a shared foundation.
//!
//! The shared foundation lives at this level; the forward and inverse machinery are grouped
//! under [`forward`] and [`inverse`] so it is clear which belongs to which.
//!
//! ## Shared foundation
//! * `basis` — the three representation seams both directions are generic over: `KappaBasis`
//!   (lens-plane κ → field), `SourceBasis` (the reconstructed source plane) and
//!   `MapDiscretization` (a discretized θ↔β map).
//! * `image` — the point-image types `Image`/`PointImageSystem` shared by both directions.
//! * `solve` — the `Solve`/`Solution`/`SolveInfo` convention iterative solvers report through.
//!
//! ## Forward: model → observables
//! [`forward`] bundles the source-independent `mapping` quantities, the point-source `extrema`
//! world, the extended-source `extended` world, and the `ForwardModel` composer.
//!
//! ## Inverse: observables → ensemble → source
//! [`inverse`] bundles the point-image `constraints` polytope, its `ensemble` sampling, the
//! per-member semi-linear source inversion and re-weighting, and the `FreeFormInversion`
//! driver that runs the whole pipeline.

pub mod basis;
pub mod forward;
pub mod image;
pub mod inverse;
pub mod solve;

pub use forward::ForwardModel;
pub use inverse::FreeFormInversion;
pub use solve::{SolveInfo, Solution, Solve};
