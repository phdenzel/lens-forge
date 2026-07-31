// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Adaptive Carrier–Greengard–Rokhlin FMM on a [`QuadTree`], $O(N)$.
//!
//! Resource: Carrier, Greengard & Rokhlin, A Fast Adaptive Multipole Algorithm
//! for Particle Simulations, SIAM J. Sci. Stat. Comput. 9(4),
//! [DOI:10.1137/0909044](https://doi.org/10.1137/0909044)
//!
//! Solves the holomorphic log-kernel on the same adaptive [`QuadTree`] as
//! Barnes–Hut (leaves at mixed depths). Three source-only passes run once at
//! [`build`](FieldSolver::build):
//!
//! * **upward** — build leaf multipoles ([`S2M`](Multipole::from_monopoles)),
//!   then merge children into parents ([`M2M`](Multipole::translate)).
//! * **colleagues** — each box's same-level adjacent neighbours.
//! * **downward** — each box's local expansion is its parent's
//!   ([`L2L`](LocalExpansion::translate)) plus well-separated same-level
//!   multipoles ([`M2L`](Multipole::to_local), V-list) and coarse-leaf sources
//!   ([`P2L`](LocalExpansion::from_monopoles), X-list).
//!
//! A multipole expansion *sends* a box's sources' influence outward, a local
//! expansion *receives* it at a target; the operators above relay that
//! information from senders to receivers. Only the sources in the query box,
//! its neighbours, and adjacent coarser leaves are summed directly.

use crate::fastsum::{FieldSolver, Jet, LocalExpansion, Monopole, Multipole, SolverConfig};
use crate::grid::{BBox, Point2D};
use crate::tree::QuadTree;
use num::complex::Complex;

/// Fast Multipole Method backend (adaptive quadtree).
#[derive(Debug)]
pub struct Fmm {
    /// Source bodies (for the direct near-field).
    sources: Vec<Monopole>,
    /// Spatial partitioning; leaves hold source indices into `sources`.
    tree: QuadTree,
    /// Precomputed local expansion per tree node.
    locals: Vec<LocalExpansion>,
    /// Same-level adjacent boxes per node (the colleague list).
    colleagues: Vec<Vec<usize>>,
    /// Root multipole, used for query points outside the root box.
    root_multipole: Option<Multipole>,
    /// Square, padded root box; queries outside it are pure far-field.
    root_box: BBox,
    /// Squared softening added to $r^2$ in the near field.
    eps2: f64,
    /// Whether the solver holds no sources.
    empty: bool,
}

/// All leaves adjacent to node `x`: its same-level leaf colleagues and the
/// coarser leaves adjacent to it (colleagues of its ancestors). No duplicates —
/// each leaf sits at a single level.
///
/// # Arguments
/// * `tree` - the quadtree.
/// * `colleagues` - same-level neighbour lists, indexed by node.
/// * `x` - node whose adjacent leaves are collected.
fn leaf_neighbors(tree: &QuadTree, colleagues: &[Vec<usize>], x: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let nx = &tree.nodes[x];
    for &c in &colleagues[x] {
        if tree.nodes[c].is_leaf() {
            out.push(c);
        }
    }
    let mut ancestor = tree.nodes[x].parent;
    while let Some(a) = ancestor {
        for &c in &colleagues[a] {
            let cn = &tree.nodes[c];
            if cn.is_leaf() && cn.is_adjacent(nx) {
                out.push(c);
            }
        }
        ancestor = tree.nodes[a].parent;
    }
    out
}

impl Fmm {
    /// Locate the deepest existing box containing `theta` (a leaf, or the box
    /// whose target quadrant is empty).
    ///
    /// # Arguments
    /// * `theta` - query point, assumed inside the root box.
    fn locate(&self, theta: Point2D) -> usize {
        let mut idx = self.tree.root;
        loop {
            let node = &self.tree.nodes[idx];
            if node.is_leaf() {
                return idx;
            }
            let c = node.center;
            let q = usize::from(theta.x >= c.x) | (usize::from(theta.y >= c.y) << 1);
            match node.children[q] {
                Some(child) => idx = child,
                None => return idx, // empty quadrant: stop here
            }
        }
    }

    /// Near-region seed boxes of `b`: `b` itself, its same-level neighbours, and
    /// the coarser leaves adjacent to `b` (colleagues of `b`'s ancestors).
    ///
    /// # Arguments
    /// * `b` - the box a query landed in.
    fn near_seeds(&self, b: usize) -> Vec<usize> {
        let mut seeds = vec![b];
        seeds.extend_from_slice(&self.colleagues[b]);
        let nb = &self.tree.nodes[b];
        let mut ancestor = self.tree.nodes[b].parent;
        while let Some(a) = ancestor {
            for &c in &self.colleagues[a] {
                let cn = &self.tree.nodes[c];
                if cn.is_leaf() && !cn.indices.is_empty() && cn.is_adjacent(nb) {
                    seeds.push(c);
                }
            }
            ancestor = self.tree.nodes[a].parent;
        }
        seeds
    }

    /// Direct-sum every source in the subtree rooted at `node_idx` into `f`.
    ///
    /// # Arguments
    /// * `node_idx` - subtree root.
    /// * `theta` - query point.
    /// * `f` - jet accumulator, updated in place.
    fn sum_subtree(&self, node_idx: usize, theta: Point2D, f: &mut Jet) {
        let node = &self.tree.nodes[node_idx];
        if node.is_leaf() {
            for &i in &node.indices {
                let e = self.sources[i];
                f.add_monopole(theta, e.position, e.strength, self.eps2);
            }
        } else {
            for &c in node.children.iter().flatten() {
                self.sum_subtree(c, theta, f);
            }
        }
    }
}

impl FieldSolver for Fmm {
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        let order = cfg.order;
        let positions: Vec<Point2D> = sources.iter().map(|s| s.position).collect();
        let tree = QuadTree::build(&positions, cfg.leaf_size.max(1));
        let n = tree.nodes.len();
        let root = &tree.nodes[tree.root];
        let root_box = BBox::new(root.center - root.half, root.center + root.half);

        // Upward pass: build multipoles for every node at every level (bottom-up)
	// S2M at leaves, M2M up to the root
        let mut mp: Vec<Option<Multipole>> = vec![None; n];
        for i in (0..n).rev() {
            let node = &tree.nodes[i];
            if node.is_leaf() {
		// S2M: build multipole directly from sources
                if node.indices.is_empty() {
                    continue;
                }
                let subset: Vec<Monopole> = node.indices.iter().map(|&k| sources[k]).collect();
                mp[i] = Some(Multipole::from_monopoles(&subset, node.center, order));
            } else {
		// M2M: build multipole upwards from children (previously built)
		// by shifting their center
                mp[i] = node
                    .children
                    .iter()
                    .flatten()
                    .filter_map(|&ch| mp[ch].as_ref().map(|cm| cm.translate(node.center)))
                    .reduce(|a, b| a + b);
            }
        }
        let root_multipole = mp[tree.root].clone();

        // Colleagues: same-level adjacent boxes
        let mut colleagues: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in 0..n {
            let Some(p) = tree.nodes[b].parent else {
                continue;
            };
	    // Candidates are all siblings of `b` (U-list) and children of its parent colleagues
            let mut candidates: Vec<usize> = tree.nodes[p].children.iter().flatten().copied().collect();
            for &pc in &colleagues[p] {
                candidates.extend(tree.nodes[pc].children.iter().flatten().copied());
            }
	    // Only keep adjacent candidates as colleagues
            let mut list = Vec::new();
            for c in candidates {
                if c != b && tree.nodes[c].is_adjacent(&tree.nodes[b]) {
                    list.push(c);
                }
            }
            colleagues[b] = list;
        }

        // Downward pass: build local expansions for every node (top-down)
	// L2L from the parent, then M2L (V-list) and P2L (X-list) from the
        // coarse leaves of the parent's colleagues.
        let mut locals: Vec<LocalExpansion> = (0..n)
            .map(|i| LocalExpansion {
                center: tree.nodes[i].center,
                coeffs: vec![Complex::new(0.0, 0.0); order + 1],
            })
            .collect();
        for b in 0..n {
            let cb = tree.nodes[b].center;
            if let Some(p) = tree.nodes[b].parent {
                let parent_local = locals[p].translate(cb);
                locals[b] += &parent_local;
                // V-list: same-level boxes well-separated from b
                for &pc in &colleagues[p] {
                    if tree.nodes[pc].is_leaf() {
                        continue;
                    }
                    for &c in tree.nodes[pc].children.iter().flatten() {
                        if !tree.nodes[c].is_adjacent(&tree.nodes[b]) {
                            if let Some(cm) = &mp[c] {
                                locals[b] += &cm.to_local(cb);
                            }
                        }
                    }
                }
                // X-list: leaves adjacent to the parent but well-separated from b
                for l in leaf_neighbors(&tree, &colleagues, p) {
                    let ln = &tree.nodes[l];
                    if !ln.indices.is_empty() && !ln.is_adjacent(&tree.nodes[b]) {
                        let subset: Vec<Monopole> = ln.indices.iter().map(|&k| sources[k]).collect();
                        locals[b] += &LocalExpansion::from_monopoles(&subset, cb, order);
                    }
                }
            }
        }

        Fmm {
            sources: sources.to_vec(),
            tree,
            locals,
            colleagues,
            root_multipole,
            root_box,
            eps2: cfg.eps2,
            empty: sources.is_empty(),
        }
    }

    fn jet(&self, theta: Point2D) -> Jet {
        if self.empty {
            return Jet::default();
        }
        // Outside the root box: every source is far, use the root multipole.
        if !self.root_box.contains(&theta) {
            return match &self.root_multipole {
                Some(mp) => mp.eval_jet(theta),
                None => Jet::default(),
            };
        }
        let b = self.locate(theta);
        let mut f = self.locals[b].eval_jet(theta);
        // Near field: direct sum over the box, its neighbours, and coarse
        // adjacent leaves (everything the local expansion omits).
        for s in self.near_seeds(b) {
            self.sum_subtree(s, theta, &mut f);
        }
        f
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
