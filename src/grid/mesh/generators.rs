// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Site-layout generators for [`VoronoiMesh`].
//!
//! Ways to place the generating sites — lattice, hexagonal honeycomb, concentric
//! rings, density-graded sampling.

use super::voronoi::VoronoiMesh;
use crate::grid::{BBox, Point2D};
use crate::utils::rng::Lcg;
use crate::{LensForgeError, Result};

impl VoronoiMesh {
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
            return Err(LensForgeError::invalid_option(
                "mesh extents must be positive",
            ));
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
            return Err(LensForgeError::invalid_option(
                "mesh extents must be positive",
            ));
        }
        let bbox = BBox::from_lims(-rx, rx, -ry, ry);
        let sites = hexagonal_sites(nx, ny, &bbox);
        Self::new(&sites, bbox)
    }

    /// Voronoi mesh of sites laid out on concentric rings about the origin, a
    /// centre site plus `rings` rings of `points_per_ring` (finer near the centre).
    ///
    /// # Arguments
    /// * `rings` - number of rings around the centre site.
    /// * `points_per_ring` - sites per ring (rings are angularly staggered).
    /// * `radius` - half the domain extent; the outer ring sits just inside it.
    pub fn concentric(rings: usize, points_per_ring: usize, radius: f64) -> Result<Self> {
        if radius <= 0.0 {
            return Err(LensForgeError::invalid_option(
                "mesh radius must be positive",
            ));
        }
        let bbox = BBox::from_lims(-radius, radius, -radius, radius);
        let sites = concentric_sites(rings, points_per_ring, radius);
        Self::new(&sites, bbox)
    }

    /// Voronoi mesh of `n` sites rejection-sampled so their spatial density
    /// follows `density`, concentrating cells where `density` is large.
    ///
    /// # Arguments
    /// * `n` - number of sites to place.
    /// * `bbox` - the domain box the sites are sampled in and clipped to.
    /// * `seed` - seed for the deterministic sampler.
    /// * `density` - non-negative weight per point; higher means denser sampling.
    pub fn density_graded<F: Fn(Point2D) -> f64>(
        n: usize,
        bbox: BBox,
        seed: u64,
        density: F,
    ) -> Result<Self> {
        if !bbox.is_positive() {
            return Err(LensForgeError::invalid_option(
                "mesh domain must have min < max",
            ));
        }
        let sites = density_graded_sites(n, &bbox, seed, density)?;
        Self::new(&sites, bbox)
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

/// Sites on concentric rings about the origin: a centre point plus `rings` rings
/// of `points_per_ring`, each ring angularly staggered to avoid radial spokes.
///
/// # Arguments
/// * `rings` - number of rings around the centre site.
/// * `points_per_ring` - sites per ring.
/// * `radius` - the outer ring sits at `radius · rings / (rings + 1)`.
///
/// Returns `1 + rings · points_per_ring` site positions inside the radius.
fn concentric_sites(rings: usize, points_per_ring: usize, radius: f64) -> Vec<Point2D> {
    let mut sites = Vec::with_capacity(1 + rings * points_per_ring);
    sites.push(Point2D::new(0.0, 0.0));
    let step = std::f64::consts::TAU / points_per_ring.max(1) as f64;
    for k in 1..=rings {
        let r = radius * k as f64 / (rings as f64 + 1.0);
        let offset = 0.5 * step * k as f64; // stagger successive rings
        for i in 0..points_per_ring {
            let theta = offset + i as f64 * step;
            sites.push(Point2D::new(r * theta.cos(), r * theta.sin()));
        }
    }
    sites
}

/// Rejection-sampled sites whose spatial density follows `density`.
///
/// # Arguments
/// * `n` - number of sites to place.
/// * `bbox` - the domain to sample within.
/// * `seed` - seed for the deterministic sampler.
/// * `density` - non-negative weight per point; negatives are clamped to zero.
///
/// Returns `n` sampled sites, or an error if `density` is nowhere positive or the
/// sampler fails to converge.
fn density_graded_sites<F: Fn(Point2D) -> f64>(
    n: usize,
    bbox: &BBox,
    seed: u64,
    density: F,
) -> Result<Vec<Point2D>> {
    if n == 0 {
        return Err(LensForgeError::invalid_parameter(
            "mesh needs at least one site",
        ));
    }
    let (w, h) = (bbox.width(), bbox.height());
    // Estimate the peak density on a coarse grid to set the acceptance scale.
    let grid = 32;
    let mut dmax = 0.0_f64;
    for j in 0..grid {
        for i in 0..grid {
            let x = bbox.min.x + (i as f64 + 0.5) / grid as f64 * w;
            let y = bbox.min.y + (j as f64 + 0.5) / grid as f64 * h;
            dmax = dmax.max(density(Point2D::new(x, y)).max(0.0));
        }
    }
    if dmax <= 0.0 {
        return Err(LensForgeError::invalid_parameter(
            "density must be positive somewhere",
        ));
    }

    let mut rng = Lcg::new(seed);
    let mut sites = Vec::with_capacity(n);
    let max_attempts = 1000 * n + 10_000;
    let mut attempts = 0;
    while sites.len() < n {
        attempts += 1;
        if attempts > max_attempts {
            return Err(LensForgeError::numerical_error(
                "density sampling failed to converge",
            ));
        }
        let p = Point2D::new(
            rng.range(bbox.min.x, bbox.max.x),
            rng.range(bbox.min.y, bbox.max.y),
        );
        // Accept with probability density(p) / dmax.
        if rng.f() * dmax <= density(p).max(0.0) {
            sites.push(p);
        }
    }
    Ok(sites)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    #[test]
    fn square_lattice_tiles_into_equal_cells() {
        // A perfect lattice is maximally cocircular; the tessellation must still
        // tile the box, and every cell must be an identical square pixel.
        let m = VoronoiMesh::square(5, 1.0).unwrap();
        assert_eq!(m.n_cells(), 25);
        assert!(
            (m.total_area() - 4.0).abs() < 1e-9,
            "area {}",
            m.total_area()
        );
        let pixel = (2.0 / 5.0) * (2.0 / 5.0); // side 2r/n = 0.4
        for s in 0..m.n_cells() {
            assert!(
                (m.cell_area(s) - pixel).abs() < 1e-9,
                "cell {s} area {}",
                m.cell_area(s)
            );
        }
    }

    #[test]
    fn rectangular_lattice_tiles_and_counts() {
        let m = VoronoiMesh::rectangular(4, 3, 2.0, 1.0).unwrap();
        assert_eq!(m.n_cells(), 12);
        assert!(
            (m.total_area() - 8.0).abs() < 1e-9,
            "area {}",
            m.total_area()
        );
    }

    #[test]
    fn rectangular_rejects_nonpositive_extent() {
        assert!(VoronoiMesh::rectangular(3, 3, 0.0, 1.0).is_err());
        assert!(VoronoiMesh::square(3, -1.0).is_err());
    }

    #[test]
    fn hexagonal_interior_cells_are_regular_hexagons() {
        // Aspect dx = 1, dy = √3⁄2 gives unit nearest-neighbour spacing and regular
        // hexagons of area √3⁄2; rx = nx·dx/2 = 3, ry = ny·dy/2 = √3.
        let m = VoronoiMesh::hexagonal(6, 4, 3.0, 3.0_f64.sqrt()).unwrap();
        assert!(
            (m.total_area() - 12.0 * 3.0_f64.sqrt()).abs() < 1e-9,
            "area {}",
            m.total_area()
        );
        let hex_area = 3.0_f64.sqrt() / 2.0;
        let n_regular = (0..m.n_cells())
            .filter(|&i| m.cell_polygon(i).len() == 6 && (m.cell_area(i) - hex_area).abs() < 1e-9)
            .count();
        assert!(
            n_regular >= 1,
            "expected regular hexagon cells, got {n_regular}"
        );
    }

    #[test]
    fn concentric_is_finer_near_the_centre() {
        let m = VoronoiMesh::concentric(4, 8, 1.0).unwrap();
        assert_eq!(m.n_cells(), 1 + 4 * 8);
        assert!(
            (m.total_area() - 4.0).abs() < 1e-9,
            "area {}",
            m.total_area()
        );
        // The central site's cell is smaller than an outer-ring cell.
        let center = m.cell_of(p(0.0, 0.0));
        let outer = m.cell_of(p(0.7, 0.0));
        assert!(m.cell_area(center) < m.cell_area(outer));
    }

    #[test]
    fn density_graded_concentrates_where_density_is_high() {
        // A centrally-peaked density must place more sites in the central disk than
        // out in the corners (of comparable area).
        let bbox = BBox::from_lims(-1.0, 1.0, -1.0, 1.0);
        let m = VoronoiMesh::density_graded(400, bbox, 0xBEEF, |q| {
            (-4.0 * (q.x * q.x + q.y * q.y)).exp()
        })
        .unwrap();
        assert_eq!(m.n_cells(), 400);
        assert!((m.total_area() - 4.0).abs() < 1e-9);
        let r2 = |q: &Point2D| q.x * q.x + q.y * q.y;
        let inner = m.sites.iter().filter(|s| r2(s) < 0.25).count(); // disk r < 0.5
        let corners = m.sites.iter().filter(|s| r2(s) > 1.0).count(); // beyond unit circle
        assert!(inner > corners, "inner {inner} vs corners {corners}");
    }

    #[test]
    fn density_graded_rejects_zero_density() {
        let bbox = BBox::from_lims(0.0, 1.0, 0.0, 1.0);
        assert!(VoronoiMesh::density_graded(10, bbox, 1, |_| 0.0).is_err());
    }
}
