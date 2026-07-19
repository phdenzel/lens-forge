// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Voronoi tessellation: the geometric dual of the Delaunay triangulation.
//!
//! [`VoronoiMesh`] is the unstructured analogue of [`Grid2D`](crate::grid::Grid2D).
//! It holds the generating sites, a half-plane-clipped Voronoi cell per site
//! (polygon, area, centroid), and the site adjacency.

use super::kdtree::KdTree;
use super::triangulate::triangulate;
use crate::grid::{BBox, Point2D};
use crate::{LensForgeError, Result};

/// A single Voronoi cell: its clipped edges, centroid, and area.
pub(crate) struct VoronoiCell {
    /// Vertices of the cell (ordering: ccw)
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

    /// Voronoi mesh of an `n × n` lattice of sites centred at the origin.
    ///
    /// The mesh analogue of [`Grid2D::square`](crate::grid::Grid2D::square): the
    /// sites are pixel centres, so the clipped cells reproduce a regular square
    /// grid spanning `-radius..radius` in each axis.
    ///
    /// # Arguments
    /// * `n` - number of sites per axis (`n²` cells total).
    /// * `radius` - half the domain extent.
    pub fn square(n: usize, radius: f64) -> Result<Self> {
        Self::rectangular(n, n, radius, radius)
    }

    /// Voronoi mesh of an `nx × ny` lattice of sites centred at the origin.
    ///
    /// # Arguments
    /// * `nx`, `ny` - number of sites along x and y (`nx · ny` cells total).
    /// * `rx`, `ry` - half the domain extent in x and y.
    pub fn rectangular(nx: usize, ny: usize, rx: f64, ry: f64) -> Result<Self> {
        if rx <= 0.0 || ry <= 0.0 {
            return Err(LensForgeError::invalid_option("mesh extents must be positive"));
        }
        let bbox = BBox::from_lims(-rx, rx, -ry, ry);
        let sites = lattice_sites(nx, ny, &bbox);
        Self::new(&sites, bbox)
    }

    /// Voronoi mesh of a triangular lattice of sites centred at the origin, whose
    /// cells form a hexagonal honeycomb (boundary cells clipped to the box).
    ///
    /// # Arguments
    /// * `nx` - sites per row.
    /// * `ny` - number of rows (odd rows offset by half the column spacing).
    /// * `rx`, `ry` - half the domain extent in x and y.
    pub fn hexagonal(nx: usize, ny: usize, rx: f64, ry: f64) -> Result<Self> {
        if rx <= 0.0 || ry <= 0.0 {
            return Err(LensForgeError::invalid_option("mesh extents must be positive"));
        }
        let bbox = BBox::from_lims(-rx, rx, -ry, ry);
        let sites = hexagonal_sites(nx, ny, &bbox);
        Self::new(&sites, bbox)
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

    /// Index of the cell containing `q`, via a warm-started greedy walk on the
    /// Voronoi adjacency: step to the nearest neighbouring site until none is
    /// closer.
    ///
    /// # Arguments
    /// * `q` - the query point to locate.
    /// * `hint` - cell to start the walk from; a spatially-close hint (e.g. the
    ///   previous result in a raster scan) keeps each walk ~`O(1)`.
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

/// Pixel-centre sites of an `nx × ny` lattice over `bbox`, in row-major order.
///
/// # Arguments
/// * `nx`, `ny` - number of sites along x and y.
/// * `bbox` - the domain the lattice fills; sites sit at cell centres.
///
/// Returns the `nx · ny` site positions (empty if either count is zero).
fn lattice_sites(nx: usize, ny: usize, bbox: &BBox) -> Vec<Point2D> {
    let dx = bbox.width() / nx as f64;
    let dy = bbox.height() / ny as f64;
    let mut sites = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        let y = bbox.min.y + (j as f64 + 0.5) * dy;
        for i in 0..nx {
            sites.push(Point2D::new(bbox.min.x + (i as f64 + 0.5) * dx, y));
        }
    }
    sites
}

/// Triangular-lattice sites over `bbox`, odd rows offset by half a column — the
/// generators whose Voronoi cells form a hexagonal honeycomb.
///
/// # Arguments
/// * `nx`, `ny` - number of columns and rows.
/// * `bbox` - the domain the lattice fills.
///
/// Returns the site positions inside `bbox` (offset rows may hold `nx - 1`).
fn hexagonal_sites(nx: usize, ny: usize, bbox: &BBox) -> Vec<Point2D> {
    let dx = bbox.width() / nx as f64;
    let dy = bbox.height() / ny as f64;
    let mut sites = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        let y = bbox.min.y + (j as f64 + 0.5) * dy;
        // Offset odd rows by half a column to build the triangular lattice.
        let x0 = bbox.min.x + if j % 2 == 1 { dx } else { 0.5 * dx };
        for i in 0..nx {
            let x = x0 + i as f64 * dx;
            if x < bbox.max.x {
                sites.push(Point2D::new(x, y));
            }
        }
    }
    sites
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
    let mut a2 = 0.0; // double signed area
    let mut c_acc = Point2D::new(0.0, 0.0); // centroid moment accumulator
    for i in 0..n {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % n];
        let cross = p0.cross(&p1);
        a2 += cross;
        c_acc += (p0 + p1) * cross;
    }
    if a2.abs() < 1e-300 {
        return (0.0, fallback);
    }
    (0.5 * a2.abs(), c_acc / (3.0 * a2))
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
    fn square_lattice_tiles_into_equal_cells() {
        // A perfect lattice is maximally cocircular; the tessellation must still
        // tile the box, and every cell must be an identical square pixel.
        let m = VoronoiMesh::square(5, 1.0).unwrap();
        assert_eq!(m.cells.len(), 25);
        assert!((m.total_area() - 4.0).abs() < 1e-9, "area {}", m.total_area());
        let pixel = (2.0 / 5.0) * (2.0 / 5.0); // side 2r/n = 0.4
        for (s, c) in m.cells.iter().enumerate() {
            assert!((c.area - pixel).abs() < 1e-9, "cell {s} area {}", c.area);
        }
    }

    #[test]
    fn rectangular_lattice_tiles_and_counts() {
        let m = VoronoiMesh::rectangular(4, 3, 2.0, 1.0).unwrap();
        assert_eq!(m.cells.len(), 12);
        assert!((m.total_area() - 8.0).abs() < 1e-9, "area {}", m.total_area());
    }

    #[test]
    fn rectangular_rejects_nonpositive_extent() {
        assert!(VoronoiMesh::rectangular(3, 3, 0.0, 1.0).is_err());
        assert!(VoronoiMesh::square(3, -1.0).is_err());
    }

    #[test]
    fn hexagonal_interior_cells_are_regular_hexagons() {
        // Aspect chosen so row height = column spacing · √3⁄2: dx = 1, dy = √3⁄2,
        // giving unit nearest-neighbour spacing and regular hexagon cells of area
        // √3⁄2. rx = nx·dx/2 = 3, ry = ny·dy/2 = √3.
        let m = VoronoiMesh::hexagonal(6, 4, 3.0, 3.0_f64.sqrt()).unwrap();
        assert!((m.total_area() - 12.0 * 3.0_f64.sqrt()).abs() < 1e-9, "area {}", m.total_area());
        let hex_area = 3.0_f64.sqrt() / 2.0;
        let n_regular = m
            .cells
            .iter()
            .filter(|c| c.polygon.len() == 6 && (c.area - hex_area).abs() < 1e-9)
            .count();
        assert!(n_regular >= 1, "expected regular hexagon cells, got {n_regular}");
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
