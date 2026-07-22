// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Square quadtree over a 2D point set (topology only).
//!
//! Stores just the spatial subdivision — square bounds and, for leaves, the
//! indices of the points they hold. Callers attach their own per-node aggregates
//! (e.g. Barnes–Hut monopoles) keyed by node index. Nodes are pushed
//! parent-before-children, so every descendant has a larger index than its
//! parent — a single reverse pass over [`QuadTree::nodes`] visits children
//! before parents.

use crate::grid::Point2D;

/// Deepest subdivision level, bounding recursion when points coincide.
const MAX_DEPTH: usize = 64;

/// A node of the [`QuadTree`]: a square region and its children or points.
#[derive(Debug)]
pub(crate) struct QuadNode {
    /// Centre of the square region this node covers.
    pub center: Point2D,
    /// Half the side length of the region.
    pub half: f64,
    /// Child node indices by quadrant (`x>=cx` bit 0, `y>=cy` bit 1), or `None`.
    pub children: [Option<usize>; 4],
    /// Indices of the points held here; non-empty only for leaves.
    pub indices: Vec<usize>,
}

impl QuadNode {
    /// Whether this node has no children (and so stores points directly).
    #[inline]
    pub(crate) fn is_leaf(&self) -> bool {
        self.children.iter().all(Option::is_none)
    }
}

/// A square quadtree over a fixed set of points, stored in an arena.
#[derive(Debug)]
pub(crate) struct QuadTree {
    /// Arena of nodes; children and the root reference entries here by index.
    pub nodes: Vec<QuadNode>,
    /// Arena index of the root node.
    pub root: usize,
}

impl QuadTree {
    /// Build a square quadtree over `points`.
    ///
    /// # Arguments
    /// * `points` - the points to index; slice positions become the indices
    ///   stored in leaf [`QuadNode::indices`].
    /// * `leaf_size` - subdivide a node until it holds at most this many points
    ///   (or [`MAX_DEPTH`] is reached).
    ///
    /// Returns the built tree; an empty `points` yields a single unit-sized root.
    pub(crate) fn build(points: &[Point2D], leaf_size: usize) -> Self {
        let mut nodes = Vec::new();
        if points.is_empty() {
            nodes.push(QuadNode {
                center: Point2D::origin(),
                half: 1.0,
                children: [None; 4],
                indices: Vec::new(),
            });
            return Self { nodes, root: 0 };
        }

        let (lo, hi) = Point2D::bounding_box(points).expect("non-empty");
        let center = lo.midpoint(&hi);
        let mut half = 0.5 * (hi.x - lo.x).max(hi.y - lo.y);
        if !(half > 0.0) {
            half = 1.0; // all points coincide
        }
        half *= 1.0 + 1e-9;

        let all: Vec<usize> = (0..points.len()).collect();
        let root = build_node(&mut nodes, points, &all, center, half, 0, leaf_size);
        Self { nodes, root }
    }
}

/// Recursively build the subtree covering `idx`, appending nodes to `nodes` and
/// returning the arena index of the subtree root.
///
/// # Arguments
/// * `nodes` - arena that built nodes are appended to.
/// * `points` - the backing points, read only.
/// * `idx` - indices into `points` covered by this node.
/// * `center` - centre of this node's square region.
/// * `half` - half the side length of the region.
/// * `depth` - current subdivision level; stops at [`MAX_DEPTH`].
/// * `leaf_size` - leaf point-count threshold.
fn build_node(
    nodes: &mut Vec<QuadNode>,
    points: &[Point2D],
    idx: &[usize],
    center: Point2D,
    half: f64,
    depth: usize,
    leaf_size: usize,
) -> usize {
    let node_idx = nodes.len();
    nodes.push(QuadNode {
        center,
        half,
        children: [None; 4],
        indices: Vec::new(),
    });

    if idx.len() <= leaf_size || depth >= MAX_DEPTH {
        nodes[node_idx].indices = idx.to_vec();
        return node_idx;
    }

    let mut quads: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for &i in idx {
        let p = points[i];
        let q = usize::from(p.x >= center.x) | (usize::from(p.y >= center.y) << 1);
        quads[q].push(i);
    }

    let hh = 0.5 * half;
    let offs = [
        Point2D::new(-hh, -hh),
        Point2D::new(hh, -hh),
        Point2D::new(-hh, hh),
        Point2D::new(hh, hh),
    ];
    let mut children = [None; 4];
    for (q, quad) in quads.iter().enumerate() {
        if !quad.is_empty() {
            let cc = center + offs[q];
            children[q] = Some(build_node(
                nodes, points, quad, cc, hh, depth + 1, leaf_size,
            ));
        }
    }
    nodes[node_idx].children = children;
    node_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rng::Lcg;

    /// Every point lands in exactly one leaf whose region contains it.
    #[test]
    fn points_partition_into_containing_leaves() {
        let mut rng = Lcg::new(0xC0FFEE);
        let pts: Vec<Point2D> = (0..400).map(|_| Point2D::new(rng.f(), rng.f())).collect();
        let tree = QuadTree::build(&pts, 8);

        let mut seen = vec![0usize; pts.len()];
        for node in &tree.nodes {
            if node.is_leaf() {
                assert!(node.indices.len() <= 8 || node.half < 1e-12);
                for &i in &node.indices {
                    seen[i] += 1;
                    let p = pts[i];
                    assert!((p.x - node.center.x).abs() <= node.half);
                    assert!((p.y - node.center.y).abs() <= node.half);
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "each point in exactly one leaf");
    }

    /// Parents precede their descendants in the arena (reverse-pass invariant).
    #[test]
    fn children_have_larger_index_than_parent() {
        let mut rng = Lcg::new(7);
        let pts: Vec<Point2D> = (0..200).map(|_| Point2D::new(rng.f(), rng.f())).collect();
        let tree = QuadTree::build(&pts, 4);
        for (i, node) in tree.nodes.iter().enumerate() {
            for &child in node.children.iter().flatten() {
                assert!(child > i);
            }
        }
    }

    #[test]
    fn empty_yields_single_root() {
        let tree = QuadTree::build(&[], 4);
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.root, 0);
        assert!(tree.nodes[0].is_leaf());
    }
}
