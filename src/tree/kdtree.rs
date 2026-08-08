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
use std::ops::Index;

/// Point the [`KdTree`] can index: fixed dimension and squared-distance metric.
pub(crate) trait KdPoint: Copy + Index<usize, Output = f64> {
    /// Number of coordinate axes (the tree cycles splits over these).
    const DIM: usize;

    /// Squared Euclidean distance to `other` (no square root).
    ///
    /// # Arguments
    /// * `other` - the point to measure against.
    fn distance_squared(&self, other: &Self) -> f64;
}

impl KdPoint for Point2D {
    const DIM: usize = 2;

    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        Point2D::distance_squared(self, other)
    }
}

/// Node in the tree; children (`left`/`right`) are `None` for an absent subtree
struct Node {
    /// Index into [`KdTree::pts`] of the point stored at this node.
    index: usize,
    /// Arena index of the left child (points below the split), or `None`.
    left: Option<u32>,
    /// Arena index of the right child (points above the split), or `None`.
    right: Option<u32>,
}

/// A balanced k-d tree over a fixed set of points
pub(crate) struct KdTree<P: KdPoint> {
    /// The indexed points, kept in their original insertion order.
    pts: Vec<P>,
    /// Arena of tree nodes; children reference entries here by index.
    nodes: Vec<Node>,
    /// Arena index of the root node, or `None` for an empty tree.
    root: Option<u32>,
}

impl<P: KdPoint> KdTree<P> {
    /// Build a balanced k-d tree over a fixed set of points
    ///
    /// # Arguments
    /// * `pts` - the points to index; their positions in the slice become the
    ///   indices returned by [`nearest`](Self::nearest).
    ///
    /// Returns the built tree, ready for nearest-neighbour queries.
    pub(crate) fn new(pts: &[P]) -> Self {
        let mut idx: Vec<usize> = (0..pts.len()).collect();
        let mut nodes = Vec::with_capacity(pts.len());
        let root = build(&mut idx, pts, 0, &mut nodes);
        Self {
            pts: pts.to_vec(),
            nodes,
            root,
        }
    }

    /// Index of the point nearest to the query point `q`.
    ///
    /// # Arguments
    /// * `q` - the query point to locate.
    ///
    /// Returns the index of the nearest point, or `None` for an empty tree.
    pub(crate) fn nearest(&self, q: P) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut dist_sq = f64::INFINITY;
        self.search(self.root, q, 0, &mut best, &mut dist_sq);
        best
    }

    /// Indices of all indexed points within Euclidean `radius` of `q`.
    ///
    /// # Arguments
    /// * `q` - the query point.
    /// * `radius` - the search radius; points at distance `≤ radius` are returned.
    ///
    /// Returns the matching indices in unspecified order.
    pub(crate) fn within_radius(&self, q: P, radius: f64) -> Vec<usize> {
        let mut hits = Vec::new();
        self.range_search(self.root, q, 0, radius * radius, &mut hits);
        hits
    }

    /// Collect into `hits` every point of the subtree at `node` within `r2`
    /// (squared radius) of `q`.
    ///
    /// # Arguments
    /// * `node` - root of the subtree to search (`None` ends the branch).
    /// * `q` - the query point.
    /// * `depth` - current tree depth; selects the split axis (`depth % DIM`).
    /// * `r2` - squared search radius.
    /// * `hits` - accumulator of matching point indices.
    fn range_search(&self, node: Option<u32>, q: P, depth: usize, r2: f64, hits: &mut Vec<usize>) {
        let Some(node) = node else {
            return;
        };
        let n = &self.nodes[node as usize];
        let pt = self.pts[n.index];
        if pt.distance_squared(&q) <= r2 {
            hits.push(n.index);
        }
        let axis = depth % P::DIM;
        let diff = q[axis] - pt[axis];
        let (near, far) = if diff < 0.0 {
            (n.left, n.right)
        } else {
            (n.right, n.left)
        };
        self.range_search(near, q, depth + 1, r2, hits);
        // The far side can only hold matches if the split plane is within radius.
        if diff * diff <= r2 {
            self.range_search(far, q, depth + 1, r2, hits);
        }
    }

    /// Recursively refine the nearest-point search over the subtree at `node`.
    ///
    /// # Arguments
    /// * `node` - root of the subtree to search (`None` ends the branch).
    /// * `q` - the query point.
    /// * `depth` - current tree depth; selects the split axis (`depth % DIM`).
    /// * `best` - index of the closest point found so far, updated in place.
    /// * `dist2` - squared distance to `best`, updated in place.
    fn search(
        &self,
        node: Option<u32>,
        q: P,
        depth: usize,
        best: &mut Option<usize>,
        dist2: &mut f64,
    ) {
        let Some(node) = node else {
            return;
        };
        // Measure distance from query point to node
        let n = &self.nodes[node as usize];
        let pt = self.pts[n.index];
        let d2 = pt.distance_squared(&q);
        if d2 < *dist2 {
            *dist2 = d2;
            *best = Some(n.index);
        }
        // Search nearest branch
        let axis = depth % P::DIM;
        let diff = q[axis] - pt[axis];
        let (near, far) = if diff < 0.0 {
            (n.left, n.right)
        } else {
            (n.right, n.left)
        };
        self.search(near, q, depth + 1, best, dist2);
        // Search far branch in case split plane is closer
        if diff * diff < *dist2 {
            self.search(far, q, depth + 1, best, dist2);
        }
    }
}

/// Recursively build a balanced subtree by splitting `idx` at its median along
/// the axis chosen by `depth`.
///
/// # Arguments
/// * `idx` - indices into `pts` for this subtree; reordered in place.
/// * `pts` - the backing points, read only.
/// * `depth` - current tree depth; selects the split axis (`depth % DIM`).
/// * `nodes` - the arena that built nodes are appended to.
///
/// Returns the arena index of the subtree root, or `None` for an empty slice.
fn build<P: KdPoint>(
    idx: &mut [usize],
    pts: &[P],
    depth: usize,
    nodes: &mut Vec<Node>,
) -> Option<u32> {
    if idx.is_empty() {
        return None;
    }
    // Reorder point index `idx` at midpoint
    let axis = depth % P::DIM;
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        pts[a][axis]
            .partial_cmp(&pts[b][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Add mid-point node
    let point = idx[mid];
    let node_id = nodes.len() as u32;
    nodes.push(Node {
        index: point,
        left: None,
        right: None,
    });
    // Divide and build subtrees (left and right)
    let (left, rest) = idx.split_at_mut(mid);
    let right = &mut rest[1..]; // leave out mid-point
    let l = build(left, pts, depth + 1, nodes);
    let r = build(right, pts, depth + 1, nodes);
    nodes[node_id as usize].left = l;
    nodes[node_id as usize].right = r;
    Some(node_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Lcg;

    fn brute(pts: &[Point2D], q: Point2D) -> usize {
        (0..pts.len())
            .min_by(|&a, &b| {
                pts[a]
                    .distance_squared(&q)
                    .partial_cmp(&pts[b].distance_squared(&q))
                    .unwrap()
            })
            .unwrap()
    }

    #[test]
    fn matches_brute_force() {
        let mut rng = Lcg::new(0xF00D);
        let pts: Vec<Point2D> = (0..500).map(|_| Point2D::new(rng.f(), rng.f())).collect();
        let tree = KdTree::new(&pts);
        for _ in 0..500 {
            let q = Point2D::new(rng.f(), rng.f());
            let got = tree.nearest(q).unwrap();
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
        assert_eq!(tree.nearest(Point2D::new(0.0, 0.0)), Some(0));
    }

    #[test]
    fn within_radius_matches_brute_force() {
        let mut rng = Lcg::new(0xBEEF);
        let pts: Vec<Point2D> = (0..500).map(|_| Point2D::new(rng.f(), rng.f())).collect();
        let tree = KdTree::new(&pts);
        for _ in 0..200 {
            let q = Point2D::new(rng.f(), rng.f());
            let r = 0.15;
            let mut got = tree.within_radius(q, r);
            got.sort_unstable();
            let mut want: Vec<usize> = (0..pts.len())
                .filter(|&i| pts[i].distance_squared(&q) <= r * r)
                .collect();
            want.sort_unstable();
            assert_eq!(got, want);
        }
    }
}
