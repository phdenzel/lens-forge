// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended-image inversion and re-weighting: from point images to arcs.
//!
//! The point-image [`ensemble`](crate::solver::ensemble) is a proposal set; extended (arc)
//! data constrain it further. For a fixed member the source is a linear least-squares solve,
//! and its forward image-plane fit is a Gaussian-evidence importance weight.
//!
//! * `source_plane` — the pixel-or-mesh source representation seam (`SourceBasis`), plus
//!   [`source_field`](source_plane::source_field) to build a member's source plane and
//!   [`reconstruct_source`](source_plane::reconstruct_source) turning its inversion into a
//!   [`SourceModel`](crate::source::SourceModel).
//! * `backproject` — ray-trace the arc mask to source-plane generators and the cheap
//!   dispersion cull.
//! * `semilinear` — Warren & Dye source inversion `(Fᵀ W F + λ R) s = Fᵀ W d` and its
//!   graph-Laplacian regularizer.
//! * `reweight` — per-member forward χ² → importance weights + effective sample size.

pub mod backproject;
pub mod reweight;
pub mod semilinear;
pub mod source_plane;

pub use reweight::{image_chi_squared, importance_weights, reweight_ensemble};
pub use semilinear::{invert_source, Regularizer};
pub use source_plane::{reconstruct_source, source_field, SourceBasis, SourceConfig};
