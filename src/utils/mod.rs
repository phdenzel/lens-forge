// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Representation-agnostic numeric helpers shared across the crate.
//!
//! * `error_free` — error-free transformations recovering the exact round-off
//!   of a single `f64` `+`/`*`.
//! * `double_double` — `hi + lo` compensated arithmetic built on error-free
//!   transformations.
//! * `rng` — a small seedable LCG for site sampling and test fixtures.
//! * `binomial` — Pascal's-triangle binomial-coefficient table.
//! * `svd` — one-sided Jacobi SVD and a Tikhonov-regularized pseudo-inverse
//!   for small, ill-conditioned dense systems.
//! * `quad` — adaptive Simpson quadrature for smooth 1D integrands.
//! * `cg` — Jacobi-preconditioned conjugate gradient for an SPD operator given
//!   as a closure.

pub(crate) mod binomial;
pub(crate) mod cg;
pub(crate) mod double_double;
pub(crate) mod error_free;
pub(crate) mod quad;
pub(crate) mod rng;
pub(crate) mod svd;
