// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Fast-summation backends for the 2D lensing Green's function.
//!
//! Given a set of [`Monopole`] sources (order-0 multipoles; in lensing
//! $\mathrm{strength} = \kappa_i \cdot \mathrm{area}_i$), every backend evaluates
//! the same [`Jet`] — ψ and its derivatives — at a query point $\theta$:
//!
//! $$ \psi(\theta) = \frac{1}{\pi} \sum_i m_i \tfrac{1}{2} \ln\left( \lvert \theta - \theta_i \rvert^2 + \varepsilon^2 \right) $$
//!
//! $$ \alpha(\theta) = \frac{1}{\pi} \sum_i m_i \frac{\theta - \theta_i}{\lvert \theta - \theta_i \rvert^2 + \varepsilon^2} $$
//!
//! $$ H(\theta) = \frac{\partial \alpha}{\partial \theta} \quad (\text{Hessian of } \psi) $$
//!
//! The backends trade accuracy for speed differently and are meant to be
//! **compared** against each other and against the exact [`DirectSum`] oracle:
//!
//! * [`DirectSum`]  — exact $O(N^2)$ reference.
//! * [`BarnesHut`]  — quadtree, monopole acceptance, $O(N \log N)$.
//! * [`Fmm`]        — 2D analytic (complex/Cauchy) multipole+local expansions, $O(N)$.
//! * [`Kifmm`]      — kernel-independent FMM (equivalent densities), $O(N)$.
//! * *(planned)* $\mathcal{H}$-matrix — hierarchical low-rank matrix (also invertible).
//!
//! All backends implement [`Summation`], so the lens models are method-agnostic.

pub mod barnes_hut;
pub mod direct;
pub mod fmm;
pub mod jet;
pub mod kernel;
pub mod kifmm;
pub mod sum;
pub(crate) mod tree_fmm;

pub use barnes_hut::BarnesHut;
pub use direct::DirectSum;
pub use fmm::Fmm;
pub use jet::Jet;
pub use kernel::{LocalExpansion, Monopole, Multipole};
pub use kifmm::Kifmm;
pub use sum::{Backend, SumConfig, Summation, build_summation};

pub(crate) use kernel::ExpansionOps;
pub(crate) use tree_fmm::TreeFmm;
