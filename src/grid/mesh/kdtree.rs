// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Static 2D k-d tree for nearest-site queries.
//!
//! For a Voronoi tessellation, the cell containing a query point is the cell of
//! the *nearest site* — so point location reduces to a nearest-neighbour search.
//! This is a balanced, build-once k-d tree over the sites giving `O(log n)`
//! queries.

use crate::grid::Point2D;

const NONE: usize = usize::MAX;

struct Node {
    point: usize,
    left: usize,
    right: usize,
}

/// A balanced 2D k-d tree over a fixed set of points.
pub(crate) struct KdTree {
    pts: Vec<Point2D>,
    nodes: Vec<Node>,
    root: usize,
}

impl KdTree {
    /// Build a k-d tree over `pts`.
    pub(crate) fn new(pts: &[Point2D]) -> Self {
        let mut idx: Vec<usize> = (0..pts.len()).collect();
        let mut nodes = Vec::with_capacity(pts.len());
        let root = build(&mut idx, pts, 0, &mut nodes);
        Self { pts: pts.to_vec(), nodes, root }
    }

    /// Index of the point nearest to `q` (the containing Voronoi cell).
    pub(crate) fn nearest(&self, q: Point2D) -> usize {
        let mut best = NONE;
        let mut best_d2 = f64::INFINITY;
        self.search(self.root, q, 0, &mut best, &mut best_d2);
        best
    }

    fn search(&self, node: usize, q: Point2D, depth: usize, best: &mut usize, best_d2: &mut f64) {
        if node == NONE {
            return;
        }
        let n = &self.nodes[node];
        let pt = self.pts[n.point];
        let d2 = pt.distance_squared(&q);
        if d2 < *best_d2 {
            *best_d2 = d2;
            *best = n.point;
        }
        let axis = depth % 2;
        let diff = q[axis] - pt[axis];
        let (near, far) = if diff < 0.0 { (n.left, n.right) } else { (n.right, n.left) };
        self.search(near, q, depth + 1, best, best_d2);
        if diff * diff < *best_d2 {
            self.search(far, q, depth + 1, best, best_d2);
        }
    }
}

/// Recursively build a balanced subtree over `idx`, splitting on the median
/// along the axis chosen by `depth`. Returns the node index of the subtree root.
fn build(idx: &mut [usize], pts: &[Point2D], depth: usize, nodes: &mut Vec<Node>) -> usize {
    if idx.is_empty() {
        return NONE;
    }
    let axis = depth % 2;
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        pts[a][axis]
            .partial_cmp(&pts[b][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let point = idx[mid];
    let node_id = nodes.len();
    nodes.push(Node { point, left: NONE, right: NONE });

    let (left, rest) = idx.split_at_mut(mid);
    let right = &mut rest[1..];
    let l = build(left, pts, depth + 1, nodes);
    let r = build(right, pts, depth + 1, nodes);
    nodes[node_id].left = l;
    nodes[node_id].right = r;
    node_id
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn brute(pts: &[Point2D], q: Point2D) -> usize {
        (0..pts.len())
            .min_by(|&a, &b| {
                pts[a].distance_squared(&q).partial_cmp(&pts[b].distance_squared(&q)).unwrap()
            })
            .unwrap()
    }

    #[test]
    fn matches_brute_force() {
        let mut rng = Lcg(0xF00D);
        let pts: Vec<Point2D> = (0..500).map(|_| Point2D::new(rng.f(), rng.f())).collect();
        let tree = KdTree::new(&pts);
        for _ in 0..500 {
            let q = Point2D::new(rng.f(), rng.f());
            let got = tree.nearest(q);
            let want = brute(&pts, q);
            // Ties are possible in principle but vanishingly unlikely for random
            // points; compare by distance to be safe.
            assert!(
                (pts[got].distance_squared(&q) - pts[want].distance_squared(&q)).abs() < 1e-15,
                "kd {got} vs brute {want}"
            );
        }
    }

    #[test]
    fn single_point() {
        let pts = [Point2D::new(3.0, 4.0)];
        let tree = KdTree::new(&pts);
        assert_eq!(tree.nearest(Point2D::new(0.0, 0.0)), 0);
    }
}
