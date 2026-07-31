// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Analytic Carrier–Greengard–Rokhlin FMM: the `Cauchy` expansion scheme and
//! the [`Fmm`] backend built on the generic `TreeFmm` driver, $O(N)$.
//!
//! Resource: Carrier, Greengard & Rokhlin, A Fast Adaptive Multipole Algorithm
//! for Particle Simulations, SIAM J. Sci. Stat. Comput. 9(4),
//! [DOI:10.1137/0909044](https://doi.org/10.1137/0909044)
//!
//! `Cauchy` realizes the six `ExpansionOps` operators for the holomorphic
//! log-kernel with closed-form complex expansions: [`Multipole`] (Laurent) as
//! the outgoing carrier and [`LocalExpansion`] (Taylor) as the incoming one. The
//! operators are scale-free, so the box half-size is unused; the scheme carries
//! just the expansion order.

use crate::fastsum::{
    ExpansionOps, FieldSolver, Jet, LocalExpansion, Monopole, Multipole, SolverConfig, TreeFmm,
};
use crate::grid::Point2D;
use crate::tree::TreeBox;
use num::complex::Complex;

/// Analytic multipole scheme for the 2D log kernel: Laurent multipoles and
/// Taylor locals with the closed-form CGR translation operators.
pub(crate) struct Cauchy {
    /// Expansion order (the operators are otherwise scale-free).
    order: usize,
}

impl ExpansionOps for Cauchy {
    type Outgoing = Multipole;
    type Incoming = LocalExpansion;

    fn build(cfg: SolverConfig, _root_half: f64, _max_depth: usize) -> Self {
        Cauchy { order: cfg.order }
    }

    fn s2m(&self, b: TreeBox, sources: &[Monopole]) -> Multipole {
        Multipole::from_monopoles(sources, b.center, self.order)
    }

    fn m2m(&self, b: TreeBox, child: &Multipole, acc: &mut Option<Multipole>) {
        let shifted = child.translate(b.center);
        *acc = Some(match acc.take() {
            Some(a) => a + shifted,
            None => shifted,
        });
    }

    fn zero_incoming(&self, b: TreeBox) -> LocalExpansion {
        LocalExpansion { center: b.center, coeffs: vec![Complex::new(0.0, 0.0); self.order + 1] }
    }

    fn l2l(&self, b: TreeBox, parent: &LocalExpansion, into: &mut LocalExpansion) {
        *into += &parent.translate(b.center);
    }

    fn m2l(&self, b: TreeBox, _src: TreeBox, out: &Multipole, into: &mut LocalExpansion) {
        *into += &out.to_local(b.center);
    }

    fn p2l(&self, b: TreeBox, sources: &[Monopole], into: &mut LocalExpansion) {
        *into += &LocalExpansion::from_monopoles(sources, b.center, self.order);
    }

    fn l2p(&self, inc: &LocalExpansion, theta: Point2D) -> Jet {
        inc.eval_jet(theta)
    }

    fn m2p(&self, out: &Multipole, theta: Point2D) -> Jet {
        out.eval_jet(theta)
    }
}

/// Fast Multipole Method backend (adaptive quadtree, analytic log-kernel scheme).
pub struct Fmm(TreeFmm<Cauchy>);

impl FieldSolver for Fmm {
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        Fmm(TreeFmm::build(sources, cfg))
    }

    fn jet(&self, theta: Point2D) -> Jet {
        self.0.jet(theta)
    }

    fn name(&self) -> &'static str {
        "fmm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::DirectSum;
    use crate::utils::rng::Lcg;

    fn random_bodies(n: usize, seed: u64) -> Vec<Monopole> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                Monopole::new(
                    Point2D::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)),
                    rng.range(0.0, 0.5),
                )
            })
            .collect()
    }

    #[test]
    fn matches_direct_within_tolerance() {
        // Full three-pass FMM against the exact sum, at targets inside the source
        // region (exercises the local expansion + near-field path).
        let bodies = random_bodies(3000, 24680);
        let cfg = SolverConfig::default().with_order(12);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let mut rng = Lcg::new(13);
        for _ in 0..40 {
            // Interior targets (well inside the source domain): the local +
            // near-field path. Outside the domain is the root-multipole regime.
            let theta = Point2D::new(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9));
            let a = fmm.gradient(theta);
            let b = direct.gradient(theta);
            let rel = a.distance(&b) / b.norm().max(1e-6);
            assert!(rel < 1e-5, "rel err {rel} too large");
        }
    }

    #[test]
    fn far_target_uses_root_multipole() {
        let bodies = random_bodies(200, 7);
        let cfg = SolverConfig::default().with_order(12);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let theta = Point2D::new(20.0, -15.0);
        let a = fmm.jet(theta);
        let b = direct.jet(theta);
        assert!(a.gradient.distance(&b.gradient) < 1e-8);
        assert!((a.scalar - b.scalar).abs() < 1e-8);
    }

    #[test]
    fn empty_and_single_body() {
        let fmm = Fmm::build(&[], SolverConfig::default());
        assert_eq!(fmm.gradient(Point2D::new(1.0, 1.0)), Point2D::origin());

        let one = [Monopole::new(Point2D::new(0.1, 0.1), 0.5)];
        let fmm = Fmm::build(&one, SolverConfig::default());
        let direct = DirectSum::build(&one, SolverConfig::default());
        let theta = Point2D::new(3.0, 2.0);
        assert!(fmm.gradient(theta).distance(&direct.gradient(theta)) < 1e-9);
    }

    #[test]
    fn hessian_matches_direct() {
        // The Hessian (κ/shear in lensing) follows the same expansions; check it
        // against the exact sum at an interior target.
        let bodies = random_bodies(3000, 555);
        let cfg = SolverConfig::default().with_order(12);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let theta = Point2D::new(0.4, -0.35);
        let a = fmm.jet(theta).hessian;
        let b = direct.jet(theta).hessian;
        let scale = b.iter().flatten().fold(0.0_f64, |m, &x| m.max(x.abs())).max(1e-6);
        for r in 0..2 {
            for c in 0..2 {
                assert!((a[r][c] - b[r][c]).abs() < 1e-4 * scale, "H[{r}][{c}] mismatch");
            }
        }
    }

    #[test]
    fn accuracy_improves_with_order() {
        // The near field is exact regardless of order, so a shrinking error with
        // order confirms the far-field M2L/L2L truncation is wired correctly.
        let bodies = random_bodies(2000, 321);
        let direct = DirectSum::build(&bodies, SolverConfig::default());
        let theta = Point2D::new(0.5, 0.2);
        let exact = direct.gradient(theta);
        let err = |order: usize| {
            Fmm::build(&bodies, SolverConfig::default().with_order(order))
                .gradient(theta)
                .distance(&exact)
        };
        let (e2, e6, e10) = (err(2), err(6), err(10));
        assert!(e6 < e2 && e10 < e6, "not monotone: {e2}, {e6}, {e10}");
    }

    #[test]
    fn deep_tree_matches_direct() {
        // leaf_size = 1 forces the deepest tree, exercising the multi-level M2M
        // (upward) and L2L (downward) translation chains.
        let bodies = random_bodies(1500, 909);
        let cfg = SolverConfig::default().with_order(12).with_leaf_size(1);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let mut rng = Lcg::new(4242);
        for _ in 0..20 {
            let theta = Point2D::new(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9));
            let exact = direct.gradient(theta);
            let rel = fmm.gradient(theta).distance(&exact) / exact.norm().max(1e-6);
            assert!(rel < 1e-4, "rel err {rel} too large");
        }
    }

    #[test]
    fn query_at_a_source_is_finite() {
        let bodies = random_bodies(500, 111);
        // eps2 = 0: the coincident source is skipped, the result stays finite.
        let fmm = Fmm::build(&bodies, SolverConfig::default().with_order(8));
        let j = fmm.jet(bodies[0].position);
        assert!(j.gradient.x.is_finite() && j.gradient.y.is_finite() && j.scalar.is_finite());
        // eps2 > 0: the near-field softening also keeps it finite.
        let fmm = Fmm::build(&bodies, SolverConfig::default().with_softening(0.05));
        let j = fmm.jet(bodies[0].position);
        assert!(j.gradient.x.is_finite() && j.gradient.y.is_finite() && j.scalar.is_finite());
    }

    #[test]
    fn clustered_sources_match_direct() {
        // Non-uniform sources (two tight clusters) exercise mixed-depth leaves,
        // the X-list (coarse-leaf P2L) and coarse-neighbour near lists that a
        // uniform tree never produces.
        let mut rng = Lcg::new(2024);
        let mut bodies = Vec::new();
        for _ in 0..800 {
            let p = Point2D::new(-0.6 + 0.05 * rng.range(-1.0, 1.0), 0.5 + 0.05 * rng.range(-1.0, 1.0));
            bodies.push(Monopole::new(p, rng.range(0.0, 0.5)));
        }
        for _ in 0..800 {
            let p = Point2D::new(0.7 + 0.05 * rng.range(-1.0, 1.0), -0.4 + 0.05 * rng.range(-1.0, 1.0));
            bodies.push(Monopole::new(p, rng.range(0.0, 0.5)));
        }
        // A few scattered sources so the tree has coarse leaves between clusters.
        for _ in 0..40 {
            bodies.push(Monopole::new(Point2D::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)), rng.range(0.0, 0.3)));
        }
        let cfg = SolverConfig::default().with_order(12);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        for _ in 0..30 {
            let theta = Point2D::new(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9));
            let a = fmm.gradient(theta);
            let b = direct.gradient(theta);
            let rel = a.distance(&b) / b.norm().max(1e-6);
            assert!(rel < 1e-4, "rel err {rel} too large");
        }
    }
}
