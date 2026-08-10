// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens-equation solvers: the forward map (model → observables) and the inverse
//! free-form reconstruction (observables → mass + source), over a shared currency.
//!
//! Every layer is generic over the field representation — a structured `PixelGrid`
//! or an unstructured `MeshGrid` — through three seams collected in `basis`:
//! `KappaBasis` (lens-plane κ → field), `SourceBasis` (the reconstructed source
//! plane) and `MapDiscretization` (a discretized θ↔β map). The point-image currency
//! is `Image`/`PointImageSystem` in `image`; iterative solvers report a `Solution`.
//!
//! ## Forward: model → observables
//! * `mapping` — quantities read off the deflection field alone (source-independent):
//!   ray-tracing β = θ − α(θ), the Jacobian det A, magnification μ = 1/det A,
//!   convergence κ = ½∇²ψ, and the critical curves and their caustics.
//! * `extrema` — the point-source world: solve β = θ − α(θ) for every image
//!   (`ImageFinder`), classify each on the Fermat arrival-time surface, form delays.
//! * `extended` — the extended-source world: the lensed surface-brightness field
//!   I(θ) = S(θ − α(θ)) and its sparse linear operator F (source → image).
//! * `forward` — the `ForwardModel` composer tying the three into observables.
//!
//! ## Inverse: observables → ensemble → source
//! The free-form κ basis makes every point-image observable linear in the cell
//! masses, so the admissible models form a convex polytope; it is sampled for a
//! degenerate-solution ensemble, re-weighted by the extended (arc) data, and reduced
//! to the observation-constrained posterior.
//! * `constraints` — images + a κ basis → the polytope A x ≤ b, C x = d.
//! * `ensemble` — hit-and-run sampling of that polytope and its physical summaries.
//! * `inverse` — the inverse direction: per-member semi-linear source reconstruction,
//!   extended-image re-weighting, ensemble reduction, and the `FreeFormInversion`
//!   driver (`inverse::freeform`) that runs the whole pipeline.

pub mod basis;
pub mod constraints;
pub mod ensemble;
pub mod extended;
pub mod extrema;
pub mod image;
pub mod inverse;
pub mod mapping;
