// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Fast-summation backends for the 2D lensing Green's function.
//!
//! Given a set of point-mass elements $m_i = \kappa_i \cdot \mathrm{area}_i$
//! (e.g. Voronoi cells reduced to their monopole), every backend evaluates the
//! same field quantities at a query point $\theta$:
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
//! * *(planned)* FMM — 2D analytic (complex/Cauchy) multipole+local expansions, $O(N)$.
//! * *(planned)* KIFMM — kernel-independent FMM (equivalent densities).
//! * *(planned)* $\mathcal{H}$-matrix — hierarchical low-rank matrix (also invertible).
//!
//! All backends implement [`FieldSolver`], so the lens models and the benchmark
//! harness are agnostic to which one is in use.

pub mod barnes_hut;
pub mod direct;
pub mod monopole;

pub use barnes_hut::BarnesHut;
pub use direct::DirectSum;
pub use monopole::Monopole;

use crate::grid::Point2D;
use std::f64::consts::FRAC_1_PI;

/// Field quantities (potential, deflection, Hessian) at a query point.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Field {
    pub psi: f64,
    pub alpha: Point2D,
    /// Row-major $\begin{bmatrix} \psi_{xx} & \psi_{xy} \\\\ \psi_{yx} & \psi_{yy} \end{bmatrix}$ (symmetric).
    pub hessian: [[f64; 2]; 2],
}

impl Field {
    /// Accumulate the contribution of a (softened) point mass at `com`.
    ///
    /// This is the shared kernel used by every backend for near-field / leaf /
    /// monopole terms, so they are numerically consistent.
    #[inline]
    pub fn add_point(&mut self, theta: Point2D, com: Point2D, mass: f64, eps2: f64) {
        let dx = theta.x - com.x;
        let dy = theta.y - com.y;
        let r2 = dx * dx + dy * dy + eps2;
        if r2 <= 0.0 {
            return; // coincident with an un-softened mass: skip the singularity
        }
        let inv = 1.0 / r2;
        let k = mass * FRAC_1_PI;
        self.psi += 0.5 * k * r2.ln();
        self.alpha.x += k * dx * inv;
        self.alpha.y += k * dy * inv;
        let inv2 = inv * inv;
        let hxx = k * (inv - 2.0 * dx * dx * inv2);
        let hxy = k * (-2.0 * dx * dy * inv2);
        let hyy = k * (inv - 2.0 * dy * dy * inv2);
        self.hessian[0][0] += hxx;
        self.hessian[0][1] += hxy;
        self.hessian[1][0] += hxy;
        self.hessian[1][1] += hyy;
    }
}

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

/// A fast-summation backend for the 2D lensing Green's function.
pub trait FieldSolver {
    /// Build the backend from source elements and a configuration.
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self
    where
        Self: Sized;

    /// Full field (ψ, α, H) at `theta`.
    fn field(&self, theta: Point2D) -> Field;

    /// Deflection only — the hot path for ray tracing.
    fn deflection(&self, theta: Point2D) -> Point2D {
        self.field(theta).alpha
    }

    /// Evaluate the full field at many targets. Backends with a cheaper batched
    /// path (e.g. FMM target trees) may override this.
    fn field_batch(&self, thetas: &[Point2D]) -> Vec<Field> {
        thetas.iter().map(|&t| self.field(t)).collect()
    }

    /// Short identifier used in benchmarks/reports.
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
