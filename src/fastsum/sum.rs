// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Summation configuration ([`SumConfig`]), the [`Summation`] trait, and runtime
//! [`Backend`] selection.

use crate::fastsum::{BarnesHut, DirectSum, Fmm, Jet, Kifmm, SourceKernel};
use crate::grid::Point2D;

/// Shared configuration for the fast-summation backends. Not every field is used
/// by every backend (e.g. `order` is ignored by [`DirectSum`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumConfig {
    /// Barnes–Hut / FMM opening angle $\theta_\mathrm{bh}$ (multipole acceptance
    /// criterion).
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

impl Default for SumConfig {
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

impl SumConfig {
    /// Set the opening angle (clamped to `>= 0`).
    ///
    /// # Arguments
    /// * `theta_bh` - Barnes–Hut / FMM acceptance angle
    #[inline]
    pub fn with_theta(mut self, theta_bh: f64) -> Self {
        self.theta_bh = theta_bh.max(0.0);
        self
    }
    /// Set the softening length (stored squared).
    ///
    /// # Arguments
    /// * `eps` - softening length added in quadrature to `r`
    #[inline]
    pub fn with_softening(mut self, eps: f64) -> Self {
        self.eps2 = eps * eps;
        self
    }
    /// Set the multipole/local expansion order.
    ///
    /// # Arguments
    /// * `order` - expansion order
    #[inline]
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }
    /// Set the maximum bodies per leaf (clamped to `>= 1`).
    ///
    /// # Arguments
    /// * `leaf_size` - bodies per leaf before subdividing
    #[inline]
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size.max(1);
        self
    }
    /// Set the target relative tolerance.
    ///
    /// # Arguments
    /// * `tol` - relative tolerance
    #[inline]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

/// A fast-summation backend for 2D Green's functions — the query interface shared
/// by every backend. Construct one with each backend's inherent `build` (generic
/// over the [`SourceKernel`]), or [`build_summation`] for a boxed backend by kind.
pub trait Summation {
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

    /// Short identifier of the summation backend.
    fn name(&self) -> &'static str;
}

/// Runtime selector over the available backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Exact $O(N^2)$ direct summation.
    Direct,
    /// Quadtree treecode with monopole acceptance, $O(N \log N)$.
    BarnesHut,
    /// Analytic (Cauchy) multipole/local expansions, $O(N)$.
    Fmm,
    /// Kernel-independent FMM with equivalent densities, $O(N)$.
    Kifmm,
    // planned: HMatrix
}

impl Backend {
    /// All currently-implemented backends (handy for comparison loops).
    pub fn implemented() -> &'static [Backend] {
        &[
            Backend::Direct,
            Backend::BarnesHut,
            Backend::Fmm,
            Backend::Kifmm,
        ]
    }
}

/// Build a boxed backend by kind over any [`SourceKernel`].
///
/// # Arguments
/// * `kind` - which backend to build.
/// * `sources` - the source kernels to sum.
/// * `cfg` - solver configuration.
pub fn build_summation<S: SourceKernel + 'static>(
    kind: Backend,
    sources: &[S],
    cfg: SumConfig,
) -> Box<dyn Summation + Send + Sync> {
    match kind {
        Backend::Direct => Box::new(DirectSum::build(sources, cfg)),
        Backend::BarnesHut => Box::new(BarnesHut::build(sources, cfg)),
        Backend::Fmm => Box::new(Fmm::build(sources, cfg)),
        Backend::Kifmm => Box::new(Kifmm::build(sources, cfg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::Monopole;

    fn bodies() -> Vec<Monopole> {
        vec![
            Monopole::new(Point2D::new(-0.3, 0.1), 0.4),
            Monopole::new(Point2D::new(0.2, -0.4), 0.6),
            Monopole::new(Point2D::new(0.05, 0.3), 0.5),
        ]
    }

    #[test]
    fn config_defaults_and_builders() {
        let d = SumConfig::default();
        assert_eq!(
            (d.theta_bh, d.order, d.eps2, d.leaf_size, d.tol),
            (0.5, 8, 0.0, 16, 1e-6)
        );
        // Builders clamp / transform their inputs.
        assert_eq!(SumConfig::default().with_theta(-1.0).theta_bh, 0.0);
        assert_eq!(SumConfig::default().with_softening(0.5).eps2, 0.25);
        assert_eq!(SumConfig::default().with_order(3).order, 3);
        assert_eq!(SumConfig::default().with_leaf_size(0).leaf_size, 1);
        assert_eq!(SumConfig::default().with_tol(1e-9).tol, 1e-9);
    }

    #[test]
    fn implemented_lists_every_backend() {
        let ks = Backend::implemented();
        assert!(ks.contains(&Backend::Direct));
        assert!(ks.contains(&Backend::BarnesHut));
        assert!(ks.contains(&Backend::Fmm));
        assert!(ks.contains(&Backend::Kifmm));
    }

    #[test]
    fn build_summation_dispatches_named_backends() {
        let src = bodies();
        let cfg = SumConfig::default();
        let names: Vec<_> = Backend::implemented()
            .iter()
            .map(|&k| build_summation(k, &src, cfg).name())
            .collect();
        assert_eq!(names, vec!["direct", "barnes-hut", "fmm", "kifmm"]);
    }

    #[test]
    fn every_backend_agrees_with_direct_in_the_far_field() {
        // Validates that build_summation wires each kind to a working field; each
        // backend's precision is pinned in its own module.
        let src = bodies();
        let cfg = SumConfig::default().with_order(12);
        let theta = Point2D::new(6.0, -4.0);
        let reference = build_summation(Backend::Direct, &src, cfg).jet(theta);
        for &kind in Backend::implemented() {
            let s = build_summation(kind, &src, cfg);
            let rel = s.jet(theta).gradient.distance(&reference.gradient)
                / reference.gradient.norm().max(1e-6);
            assert!(rel < 5e-2, "{} rel err {rel}", s.name());
        }
    }

    #[test]
    fn default_batch_and_gradient_are_consistent() {
        let src = bodies();
        let cfg = SumConfig::default().with_order(10);
        let targets = [
            Point2D::new(3.0, 1.0),
            Point2D::new(-2.0, 2.5),
            Point2D::new(5.0, -1.0),
        ];
        for &kind in Backend::implemented() {
            let s = build_summation(kind, &src, cfg);
            let batch = s.jet_batch(&targets);
            for (i, &t) in targets.iter().enumerate() {
                let j = s.jet(t);
                assert_eq!(batch[i], j, "{} batch vs pointwise", s.name());
                assert_eq!(s.gradient(t), j.gradient, "{} gradient default", s.name());
            }
        }
    }
}
