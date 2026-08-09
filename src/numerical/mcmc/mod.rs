// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Markov-chain Monte Carlo samplers of a convex polytope.
//!
//! * `polytope` — the [`Polytope`](polytope::Polytope) input, the
//!   [`PolytopeSample`](polytope::PolytopeSample) result, and the equality null-space
//!   reduction every sampler shares.
//! * `hit_and_run` — the [`hit_and_run`](hit_and_run::hit_and_run) sampler: uniform draws by
//!   isotropic-direction chords. Further chains (Dikin walk, reversible-jump) join here.

mod hit_and_run;
mod polytope;

pub(crate) use hit_and_run::hit_and_run;
pub(crate) use polytope::{Polytope, PolytopeSample};
