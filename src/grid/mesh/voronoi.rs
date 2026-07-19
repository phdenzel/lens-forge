// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Voronoi tessellation: the geometric dual of the Delaunay triangulation.
//!
//! [`VoronoiMesh`] is the unstructured analogue of [`Grid2D`](crate::grid::Grid2D).
//! It holds the generating sites, a clipped Voronoi cell (polygon, area, centroid)
//! per site, and the site adjacency. It carries no field data itself; a value
//! per cell turns it into the mesh counterpart of `PixelGrid`.
//!
//! Cells are built by **half-plane clipping**: each cell starts as the domain
//! box and is clipped by the perpendicular bisector between its site and every
//! Delaunay neighbour. This handles hull (unbounded) and interior cells
//! uniformly, needs no circumcentre/ray bookkeeping, and — because the clipped
//! cells exactly tile the box — their areas sum to the domain area.

use super::kdtree::KdTree;
use super::triangulate::triangulate;
use crate::grid::{BBox, Point2D};
use crate::{LensForgeError, Result};

/// A single Voronoi cell: its clipped edges, centroid, and area.
pub(crate) struct VoronoiCell {
    /// Edge points of the cell (ordering: ccw)
    pub polygon: Vec<Point2D>,
    /// Centroid of the cell
    pub centroid: Point2D,
    /// Area of the cell
    pub area: f64,
}

/// A Voronoi tessellation of sites, clipped to a rectangular domain.
pub(crate) struct VoronoiMesh {
    /// Generating points (one per cell).
    pub sites: Vec<Point2D>,
    /// Domain box the cells are clipped to.
    pub bbox: BBox,
    /// Voronoi cell per site.
    pub cells: Vec<VoronoiCell>,
    /// Delaunay 1-ring neighbours per site (== Voronoi adjacency).
    pub neighbors: Vec<Vec<usize>>,
    /// Spatial index over the sites for `O(log n)` point location.
    kdtree: KdTree<Point2D>,
}

impl VoronoiMesh {
    /// Build the Voronoi tessellation of `sites` clipped to a rectangular domain.
    ///
    /// # Arguments
    /// * `sites` - the generating points, one per cell; must be non-empty and finite.
    /// * `bbox` - the domain box the cells are clipped to; must be non-degenerate.
    ///
    /// Returns the [`VoronoiMesh`], or an error for empty or non-finite sites or a
    /// degenerate (`min >= max`) domain.
    pub fn new(sites: &[Point2D], bbox: BBox) -> Result<Self> {
        if sites.is_empty() {
            return Err(LensForgeError::invalid_parameter("mesh needs at least one site"));
        }
        if !bbox.is_positive() {
            return Err(LensForgeError::invalid_option("mesh domain must have min < max"));
        }
        if sites.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return Err(LensForgeError::numerical_error("mesh site is not finite"));
        }
        let mut cells = Vec::with_capacity(sites.len());
	// Build the triangulated Delaunay diagram
        let neighbors = build_adjacency(sites);
        let boxpoly = bbox.corners_ccw();
        // Squared distance of each site from the origin for the bisector offset
        let mag2: Vec<f64> = sites.iter().map(|p| p.x * p.x + p.y * p.y).collect();
        // Scratch buffers reused across every cell and every half-plane clip
        let mut src: Vec<Point2D> = Vec::with_capacity(8);
        let mut dst: Vec<Point2D> = Vec::with_capacity(8);
        for (s, &site) in sites.iter().enumerate() {
	    // Load starting polygon from bbox
            src.clear();
            src.extend_from_slice(&boxpoly);
            let half_mag2_s = 0.5 * mag2[s];
            for &t in &neighbors[s] {
		// Bisector half-plane: {x : n·x <= c};
		// all points closer to site than Delaunay neighbour
                let other = sites[t];
		// Normal vector (nx, ny)
                let n = other - site;
		// Bisector offset
                let c = 0.5 * mag2[t] - half_mag2_s;
		// Clip to Voronoi membership
                clip_halfplane(&src, n.x, n.y, c, &mut dst);
                std::mem::swap(&mut src, &mut dst);
                if src.is_empty() {
                    break;
                }
            }
            let (area, centroid) = polygon_area_centroid(&src, site);
            cells.push(VoronoiCell { polygon: src.clone(), area, centroid });
        }

        let kdtree = KdTree::new(sites);
        Ok(Self { sites: sites.to_vec(), bbox, cells, neighbors, kdtree })
    }

    /// Total area covered by the cells (equals the domain area).
    pub fn total_area(&self) -> f64 {
        self.cells.iter().map(|c| c.area).sum()
    }

    /// Index of the cell containing `p`: its nearest site, by the definition of a
    /// Voronoi tessellation, found by an `O(log n)` k-d tree query.
    ///
    /// # Arguments
    /// * `p` - the query point to locate.
    ///
    /// Returns the index of the containing (nearest-site) cell.
    pub fn cell_of(&self, p: Point2D) -> usize {
        self.kdtree.nearest(p).expect("VoronoiMesh always has at least one site")
    }

    /// Index of the cell containing `q`, found by a warm-started greedy walk on
    /// the Voronoi adjacency: step to the nearest neighbouring site until none is
    /// closer. Exact — a cell is the intersection of its neighbours' bisector
    /// half-planes, so a local distance minimum is the containing cell — and
    /// terminating, since each step strictly decreases the distance to `q`.
    ///
    /// # Arguments
    /// * `q` - the query point to locate.
    /// * `hint` - cell to start the walk from; the previous result for
    ///   spatially-coherent queries (a raster scan) keeps each walk ~`O(1)`,
    ///   versus the `O(log n)` cold descent of [`cell_of`](Self::cell_of).
    ///
    /// Returns the index of the containing (nearest-site) cell.
    pub fn cell_of_walk(&self, q: Point2D, hint: usize) -> usize {
        let mut cur = hint;
        let mut cur_d2 = self.sites[cur].distance_squared(&q);
        loop {
            let mut best = cur;
            let mut best_d2 = cur_d2;
            for &t in &self.neighbors[cur] {
                let d2 = self.sites[t].distance_squared(&q);
                if d2 < best_d2 {
                    best = t;
                    best_d2 = d2;
                }
            }
            if best == cur {
                return cur;
            }
            cur = best;
            cur_d2 = best_d2;
        }
    }
}

/// Voronoi adjacency: the deduplicated Delaunay 1-ring of each site.
///
/// # Arguments
/// * `sites` - the generating points; counts below three (which cannot be
///   triangulated) are handled directly.
///
/// Returns, per site, the sorted indices of its Delaunay neighbours.
fn build_adjacency(sites: &[Point2D]) -> Vec<Vec<usize>> {
    let n = sites.len();
    match n {
        1 => return vec![Vec::new()],
        2 => return vec![vec![1], vec![0]],
        _ => {}
    }

    let mut neighbors = vec![Vec::new(); n];
    let tri = triangulate(sites);
    for t in &tri.triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    }
    for list in &mut neighbors {
        list.sort_unstable();
        list.dedup();
    }
    neighbors
}

/// Sutherland–Hodgman clip of a convex polygon by the half-plane
/// `{(x,y) : nx·x + ny·y <= c}`.
///
/// # Arguments
/// * `poly` - the polygon to clip, assumed to be convex.
/// * `nx` - the half-plane normal's x component.
/// * `ny` - the half-plane normal's y component.
/// * `c` - the half-plane offset.
/// * `out` - reused output buffer; cleared then filled with the clipped polygon
///   (left empty if `poly` is fully outside). Must not alias `poly`.
fn clip_halfplane(poly: &[Point2D], nx: f64, ny: f64, c: f64, out: &mut Vec<Point2D>) {
    out.clear();
    let n = poly.len();
    if n == 0 {
        return;
    }
    for i in 0..n {
	// Walk the polygon's edge: current→next
        let cur = poly[i];
        let nxt = poly[(i + 1) % n];
	// Compute intersections
        let dcur = nx * cur.x + ny * cur.y - c;
        let dnxt = nx * nxt.x + ny * nxt.y - c;
	// Test inside -> keep
        let cur_in = dcur <= 0.0;
        let nxt_in = dnxt <= 0.0;
        if cur_in {
            out.push(cur);
        }
	// if outside: emit intersection point via linear interpolation onto cut
        if cur_in != nxt_in {
            let s = dcur / (dcur - dnxt);
            out.push(Point2D::new(cur.x + s * (nxt.x - cur.x), cur.y + s * (nxt.y - cur.y)));
        }
    }
}

/// Area and centroid of a simple polygon by the shoelace formula.
///
/// # Arguments
/// * `poly` - the polygon vertices in order.
/// * `fallback` - centroid returned for a degenerate (< 3 vertex) polygon.
///
/// Returns `(area, centroid)` with a non-negative area.
fn polygon_area_centroid(poly: &[Point2D], fallback: Point2D) -> (f64, Point2D) {
    let n = poly.len();
    if n < 3 {
        return (0.0, fallback);
    }
    let mut a2 = 0.0; // 2 × signed area
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        let cross = p0.x * p1.y - p1.x * p0.y;
        a2 += cross;
        cx += (p0.x + p1.x) * cross;
        cy += (p0.y + p1.y) * cross;
    }
    if a2.abs() < 1e-300 {
        return (0.0, fallback);
    }
    (0.5 * a2.abs(), Point2D::new(cx / (3.0 * a2), cy / (3.0 * a2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn unit() -> BBox {
        BBox::from_lims(0.0, 1.0, 0.0, 1.0)
    }

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

    #[test]
    fn single_site_fills_box() {
        let bbox = unit();
        let m = VoronoiMesh::new(&[p(0.3, 0.7)], bbox).unwrap();
        assert!((m.total_area() - 1.0).abs() < 1e-12);
        assert_eq!(m.cells[0].polygon.len(), 4);
    }

    #[test]
    fn two_sites_split_box_in_half() {
        let bbox = unit();
        let m = VoronoiMesh::new(&[p(0.25, 0.5), p(0.75, 0.5)], bbox).unwrap();
        assert!((m.cells[0].area - 0.5).abs() < 1e-12);
        assert!((m.cells[1].area - 0.5).abs() < 1e-12);
        assert_eq!(m.neighbors[0], vec![1]);
    }

    #[test]
    fn four_symmetric_sites_are_quadrants() {
        let bbox = unit();
        let sites = [p(0.25, 0.25), p(0.75, 0.25), p(0.75, 0.75), p(0.25, 0.75)];
        let m = VoronoiMesh::new(&sites, bbox).unwrap();
        for c in &m.cells {
            assert!((c.area - 0.25).abs() < 1e-12, "quadrant area {}", c.area);
        }
        // Each quadrant centroid sits between its site and the box centre.
        assert!(m.cells[0].centroid.x < 0.5 && m.cells[0].centroid.y < 0.5);
    }

    #[test]
    fn cells_tile_the_domain() {
        // The defining property: clipped Voronoi cells partition the box exactly.
        let mut rng = Lcg(0x1234_5678);
        let sites: Vec<Point2D> = (0..200).map(|_| p(0.02 + 0.96 * rng.f(), 0.02 + 0.96 * rng.f())).collect();
        let bbox = unit();
        let m = VoronoiMesh::new(&sites, bbox).unwrap();

        assert!((m.total_area() - 1.0).abs() < 1e-9, "area sum {}", m.total_area());
        for (s, c) in m.cells.iter().enumerate() {
            assert!(c.area > 0.0, "empty cell {s}");
            assert!(c.polygon.len() >= 3, "degenerate cell {s}");
            // Centroid lies inside the domain.
            assert!(c.centroid.x >= -1e-9 && c.centroid.x <= 1.0 + 1e-9);
            assert!(c.centroid.y >= -1e-9 && c.centroid.y <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn adjacency_is_symmetric() {
        let mut rng = Lcg(0x99);
        let sites: Vec<Point2D> = (0..60).map(|_| p(rng.f(), rng.f())).collect();
        let bbox = unit();
        let m = VoronoiMesh::new(&sites, bbox).unwrap();
        for (s, list) in m.neighbors.iter().enumerate() {
            for &t in list {
                assert!(m.neighbors[t].contains(&s), "adjacency not symmetric: {s}-{t}");
            }
        }
    }

    #[test]
    fn walk_matches_kdtree_from_any_hint() {
        // The greedy walk must land on the same cell as the exact k-d tree query,
        // regardless of the starting hint — including a deliberately bad one.
        let mut rng = Lcg(0xC0FFEE);
        let sites: Vec<Point2D> = (0..300).map(|_| p(rng.f(), rng.f())).collect();
        let bbox = unit();
        let m = VoronoiMesh::new(&sites, bbox).unwrap();
        for _ in 0..2000 {
            let q = p(rng.f(), rng.f());
            let want = m.cell_of(q);
            // Warm hint (previous-ish) and a worst-case far hint both converge.
            for hint in [want, 0, sites.len() - 1, (rng.f() * sites.len() as f64) as usize] {
                assert_eq!(m.cell_of_walk(q, hint), want, "hint {hint}, q {q:?}");
            }
        }
    }
}
