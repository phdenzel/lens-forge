// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended-image inversion and re-weighting: from point images to arcs.
//!
//! The point-image [`ensemble`](crate::solver::ensemble) is a proposal set; extended (arc)
//! data constrain it further. For a fixed member the source is a linear least-squares solve,
//! and its forward image-plane fit is a Gaussian-evidence importance weight.
//!
//! * `source_plane` — per-member source construction on a
//!   [`SourceBasis`](crate::solver::basis::SourceBasis):
//!   [`source_field`](source_plane::source_field) builds a member's source plane and
//!   [`reconstruct_source`](source_plane::reconstruct_source) turns its inversion into a
//!   [`SourceModel`](crate::source::SourceModel).
//! * `backproject` — ray-trace the arc mask to source-plane generators and the cheap
//!   dispersion cull.
//! * `semilinear` — Warren & Dye source inversion `(Fᵀ W F + λ R) s = Fᵀ W d` and its
//!   graph-Laplacian regularizer.
//! * `reweight` — per-member forward χ² → a [`WeightedEnsemble`](reweight::WeightedEnsemble)
//!   (importance weights + effective sample size).
//! * `reduce` — collapse that weighted ensemble to the observation-constrained posterior
//!   sub-ensemble (SIR resampling or a weight-threshold cull).
//! * `freeform` — the [`FreeFormInversion`](freeform::FreeFormInversion) driver running the
//!   whole pipeline (polytope → ensemble → re-weight → reduce → optional source).

pub mod backproject;
pub mod freeform;
pub mod reduce;
pub mod reweight;
pub mod semilinear;
pub mod source_plane;

pub use freeform::{EnsembleSummary, ExtendedData, FreeFormInput, FreeFormInversion, FreeFormResult};
pub use reduce::{reduce_ensemble, ReduceConfig, ReduceMethod};
pub use reweight::{image_chi_squared, importance_weights, reweight_ensemble, WeightedEnsemble};
pub use semilinear::{invert_source, Regularizer};
pub use source_plane::{reconstruct_source, source_field};
