// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared Voronoi face geometry: per-cell facet lengths and site distances.
//!
//! The finite-volume operators (the TPFA Laplacian in `diff` and the Poisson
//! solver built on it) discretize a face integral over each Voronoi cell. The
//! transmissibility across the facet shared by cells `i` and `j` is
//! $\ell_{ij} / d_{ij}$ — facet length over site distance
//! $d_{ij} = \lvert s_i - s_j \rvert$.

use super::voronoi::VoronoiMesh;

/// Precomputed shared-facet lengths of a [`VoronoiMesh`], in CSR form aligned
/// with the mesh adjacency.
pub(crate) struct FaceGeometry {
    /// Per interior face, `(neighbour index, facet length)`. Cell `i` owns
    /// `entries[offsets[i]..offsets[i + 1]]`, one entry per Voronoi neighbour.
    entries: Vec<(usize, f64)>,
    /// CSR offsets: cell `i` spans `entries[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<usize>,
}

impl FaceGeometry {
    /// Recover the symmetrized shared-facet length of every cell face.
    ///
    /// # Arguments
    /// * `mesh` - the tessellation whose facet lengths are measured.
    pub(crate) fn build(mesh: &VoronoiMesh) -> Self {
        let n = mesh.n_cells();
        // Flatten the adjacency into CSR, keeping a parallel length buffer.
        let mut neighbors: Vec<usize> = Vec::new();
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        for i in 0..n {
            neighbors.extend_from_slice(mesh.neighbors_of(i));
            offsets.push(neighbors.len());
        }
        let mut lengths = vec![0.0f64; neighbors.len()];

        // Directed pass: attribute each polygon edge to its owning neighbour.
        for i in 0..n {
            let nb = &neighbors[offsets[i]..offsets[i + 1]];
            directed_facet_lengths(mesh, i, nb, &mut lengths[offsets[i]..offsets[i + 1]]);
        }

        // Symmetrize: average the two directed lengths of every undirected face.
        let mut sym = lengths.clone();
        for i in 0..n {
            for k1 in offsets[i]..offsets[i + 1] {
                let j = neighbors[k1];
                if i >= j {
                    continue;
                }
                let jnb = &neighbors[offsets[j]..offsets[j + 1]];
                if let Ok(rel) = jnb.binary_search(&i) {
                    let k2 = offsets[j] + rel;
                    let avg = 0.5 * (lengths[k1] + lengths[k2]);
                    sym[k1] = avg;
                    sym[k2] = avg;
                }
            }
        }

        let entries = neighbors.into_iter().zip(sym).collect();
        Self { entries, offsets }
    }

    /// The interior faces of cell `i` as `(neighbour index, facet length)` pairs.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    #[inline]
    pub(crate) fn faces_of(&self, i: usize) -> &[(usize, f64)] {
        &self.entries[self.offsets[i]..self.offsets[i + 1]]
    }
}

/// Shared-facet lengths of cell `i` recovered from its own clipped polygon,
/// written into `out` aligned with `nb` (its Voronoi neighbours).
///
/// Each polygon edge is attributed to the neighbour whose site is nearest —
/// hence equidistant with the owning site — to the edge midpoint; edges on the
/// domain box match no neighbour and are skipped. These are the *directed*
/// lengths (from cell `i`'s side of each facet); [`FaceGeometry::build`]
/// symmetrizes them, while the pointwise
/// [`MeshGrid::laplacian`](crate::grid::MeshGrid::laplacian) uses them as-is.
///
/// # Arguments
/// * `mesh` - the tessellation supplying the polygon and sites.
/// * `i` - the cell index.
/// * `nb` - the Voronoi neighbours of cell `i`.
/// * `out` - per-neighbour length buffer (same length as `nb`, overwritten).
pub(super) fn directed_facet_lengths(mesh: &VoronoiMesh, i: usize, nb: &[usize], out: &mut [f64]) {
    out.fill(0.0);
    let site = mesh.site(i);
    let poly = mesh.cell_polygon(i);
    let m = poly.len();
    if m < 2 {
        return;
    }
    for e in 0..m {
        let a = poly[e];
        let b = poly[(e + 1) % m];
        let mid = a.midpoint(&b);
        let d_own = mid.distance_squared(&site);
        // The owning neighbour is the site nearest the edge midpoint.
        let mut best_k = usize::MAX;
        let mut best_d = f64::INFINITY;
        for (k, &j) in nb.iter().enumerate() {
            let d = mid.distance_squared(&mesh.site(j));
            if d < best_d {
                best_d = d;
                best_k = k;
            }
        }
        // Interior facet iff the midpoint is equidistant from the owning site and
        // that neighbour; otherwise it is a domain-box edge.
        if best_k != usize::MAX && (best_d - d_own).abs() <= 1e-9 * d_own.max(1e-300) {
            out[best_k] += a.distance(&b);
        }
    }
}

/// Per-cell homogeneous-Dirichlet boundary transmissibility `Σ ℓ_b / d_b` over
/// the domain-box facets, with `d_b` the perpendicular site-to-box distance;
/// cells with no box facet get `0`.
///
/// # Arguments
/// * `mesh` - the tessellation supplying polygons, sites and the domain box.
pub(super) fn boundary_transmissibilities(mesh: &VoronoiMesh) -> Vec<f64> {
    let (x0, x1) = mesh.extent_x();
    let (y0, y1) = mesh.extent_y();
    let eps = 1e-9 * (x1 - x0).max(y1 - y0).max(1e-300);
    let n = mesh.n_cells();
    let mut out = vec![0.0f64; n];
    for (i, o) in out.iter_mut().enumerate() {
        let site = mesh.site(i);
        let poly = mesh.cell_polygon(i);
        let m = poly.len();
        if m < 2 {
            continue;
        }
        let mut acc = 0.0;
        for e in 0..m {
            let a = poly[e];
            let b = poly[(e + 1) % m];
            // A box facet has both endpoints on one axis-aligned box side; its
            // ghost lies at twice the perpendicular site-to-side distance.
            let dist = if (a.x - x0).abs() < eps && (b.x - x0).abs() < eps {
                site.x - x0
            } else if (a.x - x1).abs() < eps && (b.x - x1).abs() < eps {
                x1 - site.x
            } else if (a.y - y0).abs() < eps && (b.y - y0).abs() < eps {
                site.y - y0
            } else if (a.y - y1).abs() < eps && (b.y - y1).abs() < eps {
                y1 - site.y
            } else {
                continue;
            };
            // Boundary value sits on the box face, so d is the site-to-face
            // distance (one-sided Dirichlet), not a doubled mirror-ghost distance.
            if dist > 0.0 {
                acc += a.distance(&b) / dist;
            }
        }
        *o = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Point2D};
    use crate::utils::rng::Lcg;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn unit() -> BBox {
        BBox::from_lims(0.0, 1.0, 0.0, 1.0)
    }

    #[test]
    fn two_sites_share_the_full_height_facet() {
        // Sites split the unit box vertically; the shared facet is the segment
        // x = 0.5, spanning the full height 1.0.
        let m = VoronoiMesh::new(&[p(0.25, 0.5), p(0.75, 0.5)], unit()).unwrap();
        let fg = FaceGeometry::build(&m);
        let faces = fg.faces_of(0);
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].0, 1);
        assert!(
            (faces[0].1 - 1.0).abs() < 1e-12,
            "facet length {}",
            faces[0].1
        );
    }

    #[test]
    fn quadrant_facets_are_half_the_box() {
        // Four symmetric quadrants: axis-adjacent facets are half-edges of length
        // 0.5; the diagonal Delaunay pair meets only at the centre (zero length).
        let sites = [p(0.25, 0.25), p(0.75, 0.25), p(0.75, 0.75), p(0.25, 0.75)];
        let m = VoronoiMesh::new(&sites, unit()).unwrap();
        let fg = FaceGeometry::build(&m);
        for i in 0..m.n_cells() {
            for &(_, len) in fg.faces_of(i) {
                assert!(
                    len < 1e-12 || (len - 0.5).abs() < 1e-12,
                    "facet length {len}"
                );
            }
        }
    }

    #[test]
    fn directed_facet_lengths_are_symmetric() {
        // ℓ_ij recovered from cell i must equal ℓ_ji recovered from cell j.
        let mut rng = Lcg::new(0xFACE);
        let sites: Vec<Point2D> = (0..80)
            .map(|_| p(0.02 + 0.96 * rng.f(), 0.02 + 0.96 * rng.f()))
            .collect();
        let m = VoronoiMesh::new(&sites, unit()).unwrap();
        let fg = FaceGeometry::build(&m);
        for i in 0..m.n_cells() {
            for &(j, len_ij) in fg.faces_of(i) {
                let len_ji = fg
                    .faces_of(j)
                    .iter()
                    .find(|&&(k, _)| k == i)
                    .map(|&(_, l)| l)
                    .expect("adjacency is symmetric");
                assert!(
                    (len_ij - len_ji).abs() < 1e-12,
                    "{i}-{j}: {len_ij} vs {len_ji}"
                );
            }
        }
    }
}
