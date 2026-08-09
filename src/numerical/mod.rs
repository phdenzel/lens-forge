// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Numerical and linear-algebra algorithms shared across the crate.
//!
//! * `binomial` — Pascal's-triangle binomial-coefficient table.
//! * `cg` — Jacobi-preconditioned conjugate gradient for an SPD operator given
//!   as a closure.
//! * `lp` — dense two-phase simplex and the Chebyshev centre of a polytope.
//! * `newton` — multivariate Newton's method for a 2-vector root, given a
//!   residual-and-Jacobian closure.
//! * `quad` — adaptive Simpson quadrature for smooth 1D integrands.
//! * `svd` — one-sided Jacobi SVD and a Tikhonov-regularized pseudo-inverse
//!   for small, ill-conditioned dense systems.

pub(crate) mod binomial;
pub(crate) mod cg;
pub(crate) mod lp;
pub(crate) mod newton;
pub(crate) mod quad;
pub(crate) mod svd;
