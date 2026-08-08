// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Voronoi tessellation: the geometric dual of the Delaunay triangulation.
//!
//! [`VoronoiMesh`] is the unstructured analogue of [`Grid2D`](crate::grid::Grid2D).
//! It holds the generating sites, a half-plane-clipped Voronoi cell per site
//! (polygon, area, centroid), and the site adjacency.

use super::triangulate::triangulate;
use crate::grid::{BBox, Point2D};
use crate::tree::KdTree;
use crate::{LensForgeError, Result};

/// The Voronoi cell polygons held in compressed-sparse-row (CSR) form.
struct VoronoiCells {
    /// Concatenated ccw polygon vertices of every cell, sliced by `offsets`.
    vertices: Vec<Point2D>,
    /// CSR offsets: cell `i` spans `vertices[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<usize>,
    /// Area-weighted centroid per cell.
    centroids: Vec<Point2D>,
    /// Area per cell.
    areas: Vec<f64>,
}

impl VoronoiCells {
    /// Clipped polygon (ccw) of cell `i`.
    #[inline]
    fn polygon(&self, i: usize) -> &[Point2D] {
        &self.vertices[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Centroid of cell `i`.
    #[inline]
    fn centroid(&self, i: usize) -> Point2D {
        self.centroids[i]
    }

    /// Area of cell `i`.
    #[inline]
    fn area(&self, i: usize) -> f64 {
        self.areas[i]
    }

    /// Total area of all cells.
    #[inline]
    fn total_area(&self) -> f64 {
        self.areas.iter().sum()
    }
}

/// The Voronoi adjacency (Delaunay 1-ring) of cell `i` stored in CSR format.
struct VoronoiAdjacency {
    /// Cell neighbour indices.
    indices: Vec<usize>,
    /// CSR offsets: cell `i` spans `indices[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<usize>,
}

impl VoronoiAdjacency {
    /// Flatten a per-cell list of neighbour lists into CSR form.
    fn from_lists(lists: &[Vec<usize>]) -> Self {
        let total: usize = lists.iter().map(Vec::len).sum();
        let mut indices = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        offsets.push(0);
        for list in lists {
            indices.extend_from_slice(list);
            offsets.push(indices.len());
        }
        Self { indices, offsets }
    }

    /// Neighbours of cell `i`.
    #[inline]
    fn of(&self, i: usize) -> &[usize] {
        &self.indices[self.offsets[i]..self.offsets[i + 1]]
    }
}

/// A Voronoi tessellation of sites, clipped to a rectangular domain.
pub struct VoronoiMesh {
    /// Generating points (one per cell).
    pub sites: Vec<Point2D>,
    /// Domain box the cells are clipped to.
    pub bbox: BBox,
    /// Per-cell clipped geometry (reach it via the accessors).
    cells: VoronoiCells,
    /// Delaunay 1-ring adjacency per site.
    neighbors: VoronoiAdjacency,
    /// Delaunay triangles (site-indexed, counter-clockwise); empty for < 3 sites.
    triangles: Vec<[usize; 3]>,
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
            return Err(LensForgeError::invalid_parameter(
                "mesh needs at least one site",
            ));
        }
        if !bbox.is_positive() {
            return Err(LensForgeError::invalid_option(
                "mesh domain must have min < max",
            ));
        }
        if sites.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return Err(LensForgeError::numerical_error("mesh site is not finite"));
        }
        let n = sites.len();
        // Triangulate once; keep the CSR adjacency and the triangles for reuse.
        let (adjacency, triangles) = build_delaunay(sites);
        let neighbors = VoronoiAdjacency::from_lists(&adjacency);
        let boxpoly = bbox.corners_ccw();
        // Squared distance of each site from the origin for the bisector offset
        let mag2: Vec<f64> = sites.iter().map(|p| p.x * p.x + p.y * p.y).collect();
        // Per-cell geometry accumulated struct-of-arrays; polygons packed CSR-style
        let mut centroids = Vec::with_capacity(n);
        let mut areas = Vec::with_capacity(n);
        let mut vertices: Vec<Point2D> = Vec::with_capacity(n * 6);
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        // Scratch buffers reused across every cell and every half-plane clip
        let mut src: Vec<Point2D> = Vec::with_capacity(8);
        let mut dst: Vec<Point2D> = Vec::with_capacity(8);
        for (s, &site) in sites.iter().enumerate() {
            // Load starting polygon from bbox
            src.clear();
            src.extend_from_slice(&boxpoly);
            let half_mag2_s = 0.5 * mag2[s];
            for &t in neighbors.of(s) {
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
            // Append this cell's polygon to the shared buffer (no per-cell alloc)
            vertices.extend_from_slice(&src);
            offsets.push(vertices.len());
            areas.push(area);
            centroids.push(centroid);
        }
        let cells = VoronoiCells {
            centroids,
            areas,
            vertices,
            offsets,
        };

        let kdtree = KdTree::new(sites);
        Ok(Self {
            sites: sites.to_vec(),
            bbox,
            cells,
            neighbors,
            triangles,
            kdtree,
        })
    }

    /// Relax the mesh with `iterations` Lloyd sweeps, moving each site to its cell
    /// centroid toward a centroidal (CVT) tessellation of more uniform,
    /// near-hexagonal cells. Chainable after any constructor.
    ///
    /// # Arguments
    /// * `iterations` - number of relaxation sweeps (`0` returns it unchanged).
    pub fn with_lloyd_iters(self, iterations: usize) -> Result<Self> {
        let mut mesh = self;
        for _ in 0..iterations {
            let relaxed: Vec<Point2D> = mesh.cells.centroids.clone();
            mesh = Self::new(&relaxed, mesh.bbox)?;
        }
        Ok(mesh)
    }

    /* ============ Mesh attributes ======================================== */
    /// Number of cells (one per generating site).
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.sites.len()
    }

    /// x extent of the domain as `(min, max)`.
    #[inline]
    pub fn extent_x(&self) -> (f64, f64) {
        self.bbox.extent_x()
    }

    /// y extent of the domain as `(min, max)`.
    #[inline]
    pub fn extent_y(&self) -> (f64, f64) {
        self.bbox.extent_y()
    }

    /// Domain width.
    #[inline]
    pub fn width(&self) -> f64 {
        self.bbox.width()
    }

    /// Domain height.
    #[inline]
    pub fn height(&self) -> f64 {
        self.bbox.height()
    }

    /// Domain centre.
    #[inline]
    pub fn center(&self) -> Point2D {
        self.bbox.center()
    }

    /// Aspect ratio (width / height).
    #[inline]
    pub fn aspect_ratio(&self) -> f64 {
        self.width() / self.height()
    }

    /// Total area covered by the cells (equals the domain area).
    pub fn total_area(&self) -> f64 {
        self.cells.total_area()
    }

    /* ============ Per-cell accessors ===================================== */
    /// Generating site of cell `i`.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    #[inline]
    pub fn site(&self, i: usize) -> Point2D {
        self.sites[i]
    }

    /// Area-weighted centroid of cell `i`.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    #[inline]
    pub fn centroid(&self, i: usize) -> Point2D {
        self.cells.centroid(i)
    }

    /// Area of cell `i`.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    #[inline]
    pub fn cell_area(&self, i: usize) -> f64 {
        self.cells.area(i)
    }

    /// Clipped polygon (counter-clockwise vertices) of cell `i`.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    #[inline]
    pub fn cell_polygon(&self, i: usize) -> &[Point2D] {
        self.cells.polygon(i)
    }

    /// Voronoi neighbours (Delaunay 1-ring) of cell `i`.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    #[inline]
    pub fn neighbors_of(&self, i: usize) -> &[usize] {
        self.neighbors.of(i)
    }

    /// All generating sites, one per cell (the mesh's coordinate array).
    #[inline]
    pub fn sites(&self) -> &[Point2D] {
        &self.sites
    }

    /// The Delaunay triangles (site-indexed, counter-clockwise), computed once at
    /// construction; empty for fewer than three sites.
    #[inline]
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }

    /// Whether `p` lies within the domain box.
    ///
    /// # Arguments
    /// * `p` - the point to test.
    #[inline]
    pub fn contains(&self, p: Point2D) -> bool {
        self.bbox.contains(&p)
    }

    /* ============ Maps ==================================================== */
    /// The area-weighted centroids of every cell.
    pub fn centroids(&self) -> Vec<Point2D> {
        self.cells.centroids.clone()
    }

    /// Apply `f` to every site, collecting one value per cell — the unstructured
    /// analogue of [`Grid2D::map_centers`](crate::grid::Grid2D::map_centers).
    ///
    /// # Arguments
    /// * `f` - mapped over each generating site.
    pub fn map_sites<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(Point2D) -> T,
    {
        self.sites.iter().map(|&s| f(s)).collect()
    }

    /* ============ Transformations ======================================== */
    /// A copy translated by `(dx, dy)`: sites and domain shifted, then rebuilt.
    ///
    /// # Arguments
    /// * `dx`, `dy` - the shift along each axis.
    pub fn translate(&self, dx: f64, dy: f64) -> Result<Self> {
        let shift = Point2D::new(dx, dy);
        let sites: Vec<Point2D> = self.sites.iter().map(|&s| s + shift).collect();
        Self::new(
            &sites,
            BBox::new(self.bbox.min + shift, self.bbox.max + shift),
        )
    }

    /// Index of the cell containing `p`: its nearest site, by the definition of a
    /// Voronoi tessellation, found by an `O(log n)` k-d tree query.
    ///
    /// # Arguments
    /// * `p` - the query point to locate.
    ///
    /// Returns the index of the containing (nearest-site) cell.
    pub fn cell_of(&self, p: Point2D) -> usize {
        self.kdtree
            .nearest(p)
            .expect("VoronoiMesh always has at least one site")
    }

    /// Indices of all sites within Euclidean `radius` of `q`, via an `O(log n)`
    /// k-d tree range query (order unspecified).
    ///
    /// # Arguments
    /// * `q` - the query point.
    /// * `radius` - the search radius (sites at distance `≤ radius` are returned).
    pub fn sites_within(&self, q: Point2D, radius: f64) -> Vec<usize> {
        self.kdtree.within_radius(q, radius)
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
            for &t in self.neighbors.of(cur) {
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

impl std::fmt::Display for VoronoiMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VoronoiMesh({} sites, [{:.3}, {:.3}]x[{:.3}, {:.3}])",
            self.sites.len(),
            self.bbox.min.x,
            self.bbox.max.x,
            self.bbox.min.y,
            self.bbox.max.y,
        )
    }
}

/// Delaunay connectivity of `sites`: the deduplicated 1-ring adjacency per site
/// and the triangle list, from a single triangulation.
///
/// # Arguments
/// * `sites` - the generating points; counts below three (which cannot be
///   triangulated) are handled directly, yielding no triangles.
///
/// Returns the per-site sorted neighbour indices and the site-indexed triangles.
fn build_delaunay(sites: &[Point2D]) -> (Vec<Vec<usize>>, Vec<[usize; 3]>) {
    let n = sites.len();
    match n {
        1 => return (vec![Vec::new()], Vec::new()),
        2 => return (vec![vec![1], vec![0]], Vec::new()),
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
    (neighbors, tri.triangles)
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
pub(super) fn clip_halfplane(poly: &[Point2D], nx: f64, ny: f64, c: f64, out: &mut Vec<Point2D>) {
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
            out.push(Point2D::new(
                cur.x + s * (nxt.x - cur.x),
                cur.y + s * (nxt.y - cur.y),
            ));
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
    use crate::rng::Lcg;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn unit() -> BBox {
        BBox::from_lims(0.0, 1.0, 0.0, 1.0)
    }

    #[test]
    fn sites_within_matches_brute_force() {
        let mut rng = Lcg::new(0x51DE5);
        let sites: Vec<Point2D> = (0..300).map(|_| p(rng.f(), rng.f())).collect();
        let m = VoronoiMesh::new(&sites, unit()).unwrap();
        let q = p(0.5, 0.5);
        let r = 0.2;
        let mut got = m.sites_within(q, r);
        got.sort_unstable();
        let mut want: Vec<usize> = (0..sites.len())
            .filter(|&i| sites[i].distance(&q) <= r)
            .collect();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn with_lloyd_iters_reduces_area_variance() {
        // Random sites are uneven; Lloyd relaxation should even the cell areas out
        // (converging toward a centroidal Voronoi tessellation).
        let mut rng = Lcg::new(0xF00D);
        let sites: Vec<Point2D> = (0..80)
            .map(|_| p(0.05 + 0.9 * rng.f(), 0.05 + 0.9 * rng.f()))
            .collect();
        let bbox = unit();
        let before = VoronoiMesh::new(&sites, bbox).unwrap();
        let after = VoronoiMesh::new(&sites, bbox)
            .unwrap()
            .with_lloyd_iters(15)
            .unwrap();
        assert!((after.total_area() - 1.0).abs() < 1e-9);
        let area_var = |m: &VoronoiMesh| {
            let mean = m.total_area() / m.n_cells() as f64;
            (0..m.n_cells())
                .map(|i| (m.cell_area(i) - mean).powi(2))
                .sum::<f64>()
                / m.n_cells() as f64
        };
        assert!(area_var(&after) < area_var(&before), "variance not reduced");
    }

    #[test]
    fn with_lloyd_iters_zero_is_identity() {
        let m = VoronoiMesh::new(&[p(0.2, 0.3), p(0.8, 0.7), p(0.4, 0.9)], unit()).unwrap();
        let sites_before = m.sites.clone();
        let relaxed = m.with_lloyd_iters(0).unwrap();
        assert_eq!(relaxed.sites, sites_before);
    }

    #[test]
    fn attributes_match_domain_and_sites() {
        let m = VoronoiMesh::new(&[p(0.25, 0.5), p(0.75, 0.5)], unit()).unwrap();
        assert_eq!(m.n_cells(), 2);
        assert_eq!(m.extent_x(), (0.0, 1.0));
        assert_eq!(m.extent_y(), (0.0, 1.0));
        assert!((m.width() - 1.0).abs() < 1e-12);
        assert!((m.height() - 1.0).abs() < 1e-12);
        assert!((m.aspect_ratio() - 1.0).abs() < 1e-12);
        assert_eq!(m.center(), p(0.5, 0.5));
        assert!(m.contains(p(0.5, 0.5)));
        assert!(!m.contains(p(2.0, 0.0)));
    }

    #[test]
    fn per_cell_accessors_are_consistent() {
        let sites = [p(0.25, 0.25), p(0.75, 0.25), p(0.5, 0.75)];
        let m = VoronoiMesh::new(&sites, unit()).unwrap();
        assert_eq!(m.sites(), sites.as_slice());
        for i in 0..m.n_cells() {
            assert_eq!(m.site(i), m.sites[i]);
            // The area accessor matches a shoelace recompute of the polygon accessor,
            // and the centroid accessor agrees with the maps-level `centroids()`.
            let poly = m.cell_polygon(i);
            assert!(poly.len() >= 3, "degenerate polygon for cell {i}");
            let (area, centroid) = polygon_area_centroid(poly, m.site(i));
            assert!((m.cell_area(i) - area).abs() < 1e-12);
            assert_eq!(m.centroid(i), centroid);
            assert_eq!(m.centroid(i), m.centroids()[i]);
            // Neighbours are valid, distinct cell indices.
            for &t in m.neighbors_of(i) {
                assert!(t < m.n_cells() && t != i, "bad neighbour {t} of {i}");
            }
        }
        assert_eq!(m.centroids().len(), 3);
    }

    #[test]
    fn map_sites_collects_one_value_per_cell() {
        let sites = [p(0.1, 0.2), p(0.8, 0.9)];
        let m = VoronoiMesh::new(&sites, unit()).unwrap();
        assert_eq!(m.map_sites(|s| s.x), vec![0.1, 0.8]);
    }

    #[test]
    fn translate_shifts_sites_and_preserves_area() {
        let m = VoronoiMesh::new(&[p(0.25, 0.25), p(0.75, 0.75)], unit()).unwrap();
        let t = m.translate(1.0, 2.0).unwrap();
        assert_eq!(t.site(0), p(1.25, 2.25));
        assert_eq!(t.extent_x(), (1.0, 2.0));
        assert_eq!(t.extent_y(), (2.0, 3.0));
        assert!((t.total_area() - m.total_area()).abs() < 1e-12);
    }

    #[test]
    fn display_reports_sites_and_extent() {
        let m = VoronoiMesh::new(&[p(0.3, 0.7)], unit()).unwrap();
        assert!(format!("{m}").starts_with("VoronoiMesh(1 sites"));
    }

    #[test]
    fn single_site_fills_box() {
        let bbox = unit();
        let m = VoronoiMesh::new(&[p(0.3, 0.7)], bbox).unwrap();
        assert!((m.total_area() - 1.0).abs() < 1e-12);
        assert_eq!(m.cell_polygon(0).len(), 4);
    }

    #[test]
    fn two_sites_split_box_in_half() {
        let bbox = unit();
        let m = VoronoiMesh::new(&[p(0.25, 0.5), p(0.75, 0.5)], bbox).unwrap();
        assert!((m.cell_area(0) - 0.5).abs() < 1e-12);
        assert!((m.cell_area(1) - 0.5).abs() < 1e-12);
        assert_eq!(m.neighbors_of(0), &[1]);
    }

    #[test]
    fn four_symmetric_sites_are_quadrants() {
        let bbox = unit();
        let sites = [p(0.25, 0.25), p(0.75, 0.25), p(0.75, 0.75), p(0.25, 0.75)];
        let m = VoronoiMesh::new(&sites, bbox).unwrap();
        for i in 0..m.n_cells() {
            assert!(
                (m.cell_area(i) - 0.25).abs() < 1e-12,
                "quadrant area {}",
                m.cell_area(i)
            );
        }
        // Each quadrant centroid sits between its site and the box centre.
        let c0 = m.centroid(0);
        assert!(c0.x < 0.5 && c0.y < 0.5);
    }

    #[test]
    fn cells_tile_the_domain() {
        // The defining property: clipped Voronoi cells partition the box exactly.
        let mut rng = Lcg::new(0x1234_5678);
        let sites: Vec<Point2D> = (0..200)
            .map(|_| p(0.02 + 0.96 * rng.f(), 0.02 + 0.96 * rng.f()))
            .collect();
        let bbox = unit();
        let m = VoronoiMesh::new(&sites, bbox).unwrap();

        assert!(
            (m.total_area() - 1.0).abs() < 1e-9,
            "area sum {}",
            m.total_area()
        );
        for i in 0..m.n_cells() {
            assert!(m.cell_area(i) > 0.0, "empty cell {i}");
            assert!(m.cell_polygon(i).len() >= 3, "degenerate cell {i}");
            // Centroid lies inside the domain.
            let c = m.centroid(i);
            assert!(c.x >= -1e-9 && c.x <= 1.0 + 1e-9);
            assert!(c.y >= -1e-9 && c.y <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn adjacency_is_symmetric() {
        let mut rng = Lcg::new(0x99);
        let sites: Vec<Point2D> = (0..60).map(|_| p(rng.f(), rng.f())).collect();
        let bbox = unit();
        let m = VoronoiMesh::new(&sites, bbox).unwrap();
        for s in 0..m.n_cells() {
            for &t in m.neighbors_of(s) {
                assert!(
                    m.neighbors_of(t).contains(&s),
                    "adjacency not symmetric: {s}-{t}"
                );
            }
        }
    }

    #[test]
    fn walk_matches_kdtree_from_any_hint() {
        // The greedy walk must land on the same cell as the exact k-d tree query,
        // regardless of the starting hint — including a deliberately bad one.
        let mut rng = Lcg::new(0xC0FFEE);
        let sites: Vec<Point2D> = (0..300).map(|_| p(rng.f(), rng.f())).collect();
        let bbox = unit();
        let m = VoronoiMesh::new(&sites, bbox).unwrap();
        for _ in 0..2000 {
            let q = p(rng.f(), rng.f());
            let want = m.cell_of(q);
            // Warm hint (previous-ish) and a worst-case far hint both converge.
            for hint in [
                want,
                0,
                sites.len() - 1,
                (rng.f() * sites.len() as f64) as usize,
            ] {
                assert_eq!(m.cell_of_walk(q, hint), want, "hint {hint}, q {q:?}");
            }
        }
    }
}
