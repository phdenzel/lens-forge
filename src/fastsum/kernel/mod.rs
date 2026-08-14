// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Representations of the 2D lensing Green's-function kernel, near and far.
//!
//! The layer the summation backends sum over, separate from the algorithms that
//! aggregate it:
//!
//! * `source` — [`SourceKernel`], the trait a backend sums over.
//! * `monopole` — [`Monopole`], the point/near source kernel.
//! * `multipole` — [`Multipole`], the far-field multipole expansion.
//! * `local_expansion` — [`LocalExpansion`], the local (Taylor) expansion.
//! * `expansion` — `ExpansionOps`, the shared multipole/local expansion operations.

pub(crate) mod expansion;
pub mod local_expansion;
pub mod monopole;
pub mod multipole;
pub mod source;

pub use local_expansion::LocalExpansion;
pub use monopole::Monopole;
pub use multipole::Multipole;
pub use source::SourceKernel;

pub(crate) use expansion::ExpansionOps;
