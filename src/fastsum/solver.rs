// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Backend configuration, the [`FieldSolver`] trait, and runtime selection.

use crate::fastsum::{BarnesHut, DirectSum, Jet, Monopole};
use crate::grid::Point2D;

/// Shared configuration for the fast-summation backends. Not every field is used
/// by every backend (e.g. `order` is ignored by [`DirectSum`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverConfig {
    /// Barnes–Hut / FMM opening angle $\theta_\mathrm{bh}$ (multipole acceptance
    /// criterion). $0$ forces exact evaluation.
    pub theta_bh: f64,
    /// Multipole / local expansion order (FMM, KIFMM).
    pub order: usize,
    /// Squared softening length $\varepsilon^2$ added to $r^2$.
    pub eps2: f64,
    /// Maximum bodies per leaf before subdividing.
    pub leaf_size: usize,
    /// Target relative tolerance (ℋ-matrix / ACA).
    pub tol: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            theta_bh: 0.5,
            order: 8,
            eps2: 0.0,
            leaf_size: 16,
            tol: 1e-6,
        }
    }
}

impl SolverConfig {
    #[inline]
    pub fn with_theta(mut self, theta_bh: f64) -> Self {
        self.theta_bh = theta_bh.max(0.0);
        self
    }
    #[inline]
    pub fn with_softening(mut self, eps: f64) -> Self {
        self.eps2 = eps * eps;
        self
    }
    #[inline]
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }
    #[inline]
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size.max(1);
        self
    }
    #[inline]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

/// A fast-summation backend for 2D Green's functions.
pub trait FieldSolver {
    /// Build the backend from source elements and a configuration.
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self
    where
        Self: Sized;

    /// The field's full 2-jet (scalar, gradient, hessian) at `theta`.
    fn jet(&self, theta: Point2D) -> Jet;

    /// Gradient only from the 2-jet.
    fn gradient(&self, theta: Point2D) -> Point2D {
        self.jet(theta).gradient
    }

    /// Evaluate the full field jets at many targets. Backends with a cheaper
    /// batched path (e.g. FMM target trees) may override this.
    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<Jet> {
        thetas.iter().map(|&t| self.jet(t)).collect()
    }

    /// Short identifier of the type of solver.
    fn name(&self) -> &'static str;
}

/// Runtime selector over the available backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    Direct,
    BarnesHut,
    // planned: Fmm, Kifmm, HMatrix
}

impl SolverKind {
    /// All currently-implemented backends (handy for comparison loops).
    pub fn implemented() -> &'static [SolverKind] {
        &[SolverKind::Direct, SolverKind::BarnesHut]
    }
}

/// Build a boxed backend by kind.
pub fn build_solver(
    kind: SolverKind,
    sources: &[Monopole],
    cfg: SolverConfig,
) -> Box<dyn FieldSolver + Send + Sync> {
    match kind {
        SolverKind::Direct => Box::new(DirectSum::build(sources, cfg)),
        SolverKind::BarnesHut => Box::new(BarnesHut::build(sources, cfg)),
    }
}
