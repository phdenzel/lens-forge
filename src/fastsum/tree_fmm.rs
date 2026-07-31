// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Generic adaptive FMM driver on a [`QuadTree`], parameterized over an
//! [`ExpansionOps`] scheme, $O(N)$.
//!
//! Three source-only passes run once at [`build`](TreeFmm::build), on the same
//! adaptive tree as Barnes–Hut (leaves at mixed depths):
//!
//! * **upward** — build leaf outgoing expansions (S2M), then merge children into
//!   parents (M2M).
//! * **colleagues** — each box's same-level adjacent neighbours.
//! * **downward** — each box's incoming expansion is its parent's (L2L) plus
//!   well-separated same-level outgoing expansions (M2L, V-list) and coarse-leaf
//!   sources (P2L, X-list).
//!
//! At query time only the local expansion plus the sources in the query box, its
//! neighbours, and adjacent coarser leaves are summed directly. The tree walk
//! and near field are kernel-independent; the [`ExpansionOps`] scheme supplies
//! the rest.

use crate::fastsum::{ExpansionOps, Jet, Monopole, SolverConfig};
use crate::grid::{BBox, Point2D};
use crate::tree::QuadTree;

/// Adaptive-quadtree FMM engine over an [`ExpansionOps`] scheme `E`.
pub(crate) struct TreeFmm<E: ExpansionOps> {
    /// Source bodies (for the direct near-field).
    sources: Vec<Monopole>,
    /// Spatial partitioning; leaves hold source indices into `sources`.
    tree: QuadTree,
    /// Kernel- and scale-dependent operator data.
    ctx: E::Ctx,
    /// Precomputed incoming expansion per tree node.
    incoming: Vec<E::Incoming>,
    /// Same-level adjacent boxes per node (the colleague list).
    colleagues: Vec<Vec<usize>>,
    /// Root outgoing expansion, used for query points outside the root box.
    root_outgoing: Option<E::Outgoing>,
    /// Square, padded root box; queries outside it are pure far-field.
    root_box: BBox,
    /// Squared softening added to $r^2$ in the near field.
    eps2: f64,
    /// Whether the driver holds no sources.
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

impl<E: ExpansionOps> TreeFmm<E> {
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

    /// Build the engine from source elements and a configuration.
    ///
    /// # Arguments
    /// * `sources` - the monopole bodies.
    /// * `cfg` - solver configuration (order, leaf size, softening).
    pub(crate) fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        let order = cfg.order;
        let positions: Vec<Point2D> = sources.iter().map(|s| s.position).collect();
        let tree = QuadTree::build(&positions, cfg.leaf_size.max(1));
        let n = tree.nodes.len();
        let root = &tree.nodes[tree.root];
        let root_box = BBox::new(root.center - root.half, root.center + root.half);
        let root_half = root.half;

        // Node depths (parent precedes child in the arena) and the tree's depth.
        let mut depth = vec![0usize; n];
        let mut max_depth = 0;
        for i in 0..n {
            if let Some(p) = tree.nodes[i].parent {
                depth[i] = depth[p] + 1;
                max_depth = max_depth.max(depth[i]);
            }
        }
        let ctx = E::context(root_half, max_depth, order, cfg.eps2);

        // Upward pass: outgoing expansion for every node, bottom-up (S2M at
        // leaves, M2M up to the root).
        let mut out: Vec<Option<E::Outgoing>> = vec![None; n];
        for i in (0..n).rev() {
            let node = &tree.nodes[i];
            let b = node.to_box();
            if node.is_leaf() {
                if node.indices.is_empty() {
                    continue;
                }
                let subset: Vec<Monopole> = node.indices.iter().map(|&k| sources[k]).collect();
                out[i] = Some(E::s2m(&ctx, b, &subset));
            } else {
                let mut acc: Option<E::Outgoing> = None;
                for &ch in node.children.iter().flatten() {
                    if let Some(cm) = &out[ch] {
                        E::m2m(&ctx, b, cm, &mut acc);
                    }
                }
                out[i] = acc;
            }
        }
        let root_outgoing = out[tree.root].clone();

        // Colleagues: same-level adjacent boxes.
        let mut colleagues: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in 0..n {
            let Some(p) = tree.nodes[b].parent else {
                continue;
            };
            // Candidates are all siblings of `b` and children of `p`'s colleagues.
            let mut candidates: Vec<usize> =
                tree.nodes[p].children.iter().flatten().copied().collect();
            for &pc in &colleagues[p] {
                candidates.extend(tree.nodes[pc].children.iter().flatten().copied());
            }
            let mut list = Vec::new();
            for c in candidates {
                if c != b && tree.nodes[c].is_adjacent(&tree.nodes[b]) {
                    list.push(c);
                }
            }
            colleagues[b] = list;
        }

        // Downward pass: incoming expansion for every node, top-down (L2L from
        // the parent, then M2L (V-list) and P2L (X-list) from coarse leaves).
        let mut incoming: Vec<E::Incoming> =
            (0..n).map(|i| E::zero_incoming(&ctx, tree.nodes[i].to_box())).collect();
        for b in 0..n {
            let bb = tree.nodes[b].to_box();
            let Some(p) = tree.nodes[b].parent else {
                continue;
            };
            let parent = incoming[p].clone();
            E::l2l(&ctx, bb, &parent, &mut incoming[b]);
            // V-list: same-level boxes well-separated from b.
            for &pc in &colleagues[p] {
                if tree.nodes[pc].is_leaf() {
                    continue;
                }
                for &c in tree.nodes[pc].children.iter().flatten() {
                    if !tree.nodes[c].is_adjacent(&tree.nodes[b]) {
                        if let Some(cm) = &out[c] {
                            E::m2l(&ctx, bb, tree.nodes[c].to_box(), cm, &mut incoming[b]);
                        }
                    }
                }
            }
            // X-list: leaves adjacent to the parent but well-separated from b.
            for l in leaf_neighbors(&tree, &colleagues, p) {
                let ln = &tree.nodes[l];
                if !ln.indices.is_empty() && !ln.is_adjacent(&tree.nodes[b]) {
                    let subset: Vec<Monopole> = ln.indices.iter().map(|&k| sources[k]).collect();
                    E::p2l(&ctx, bb, &subset, &mut incoming[b]);
                }
            }
        }

        TreeFmm {
            sources: sources.to_vec(),
            tree,
            ctx,
            incoming,
            colleagues,
            root_outgoing,
            root_box,
            eps2: cfg.eps2,
            empty: sources.is_empty(),
        }
    }

    /// The field's full 2-jet (scalar, gradient, hessian) at `theta`.
    ///
    /// # Arguments
    /// * `theta` - query point.
    pub(crate) fn jet(&self, theta: Point2D) -> Jet {
        if self.empty {
            return Jet::default();
        }
        // Outside the root box: every source is far, use the root expansion.
        if !self.root_box.contains(&theta) {
            return match &self.root_outgoing {
                Some(out) => E::m2p(&self.ctx, out, theta),
                None => Jet::default(),
            };
        }
        let b = self.locate(theta);
        let mut f = E::l2p(&self.ctx, &self.incoming[b], theta);
        // Near field: direct sum over the box, its neighbours, and coarse
        // adjacent leaves (everything the local expansion omits).
        for s in self.near_seeds(b) {
            self.sum_subtree(s, theta, &mut f);
        }
        f
    }
}
