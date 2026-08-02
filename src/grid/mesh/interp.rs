// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Continuous interpolation of a [`MeshGrid`] at arbitrary points — the
//! unstructured analogue of
//! [`PixelGrid::interpolate`](crate::grid::PixelGrid::interpolate), unlike the
//! piecewise-constant
//! [`ScalarField2D::sample`](crate::grid::ScalarField2D::sample).
//!
//! The primary scheme is **Laplace (non-Sibsonian) natural-neighbour
//! interpolation**: each site `i` whose cell would border the inserted cell of
//! the query point `p` contributes with weight `λ_i = ℓ_i / d_i`, giving
//! `value(p) = Σ λ_i v_i / Σ λ_i`, where `d_i = |p − sᵢ|` and `ℓ_i` is the shared
//! Voronoi facet length. It is exact for linear fields and C¹ away from the
//! sites, and needs only facet lengths — no re-tessellation.
//!
//! Where `p` is not fully enclosed by its natural neighbours (at or beyond the
//! edge of the site cloud) the weights would extrapolate, so the interpolant
//! falls back to **barycentric-linear** over the enclosing Delaunay triangle,
//! and finally — outside the convex hull — to the nearest cell value.

use super::meshgrid::MeshGrid;
use super::voronoi::clip_halfplane;
use crate::grid::Point2D;
use rayon::prelude::*;

impl MeshGrid {
    /// Continuously interpolate the field at `p`, returning `0.0` outside the
    /// domain box.
    ///
    /// # Arguments
    /// * `p` - the query point.
    #[inline]
    pub fn interpolate(&self, p: Point2D) -> f64 {
        self.interpolate_or(p, 0.0)
    }

    /// Continuously interpolate the field at `p` with an explicit out-of-domain
    /// fallback.
    ///
    /// Uses Laplace natural-neighbour interpolation where `p` is fully enclosed
    /// by its natural neighbours, degrading to barycentric-linear near the edge
    /// of the site cloud and to nearest-cell sampling outside its convex hull.
    ///
    /// # Arguments
    /// * `p` - the query point.
    /// * `default` - value returned when `p` lies outside the domain box.
    pub fn interpolate_or(&self, p: Point2D, default: f64) -> f64 {
        if !self.mesh().contains(p) {
            return default;
        }
        self.laplace_nn(p)
            .or_else(|| self.barycentric(p))
            .unwrap_or_else(|| self.value(self.cell_of(p)))
    }

    /// Interpolate the field at each of `points` (out-of-domain points yield
    /// `0.0`).
    ///
    /// # Arguments
    /// * `points` - the query points.
    pub fn interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        points
            .iter()
            .map(|&p| self.interpolate_or(p, 0.0))
            .collect()
    }

    /// Interpolate the field at each of `points` in parallel.
    ///
    /// # Arguments
    /// * `points` - the query points.
    pub fn par_interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        points
            .par_iter()
            .map(|&p| self.interpolate_or(p, 0.0))
            .collect()
    }

    /// Laplace (non-Sibsonian) natural-neighbour value at `p`, or `None` when
    /// `p` is not fully enclosed by its natural neighbours (the virtual cell
    /// borders the domain box), where the weights would extrapolate.
    ///
    /// # Arguments
    /// * `p` - the query point.
    fn laplace_nn(&self, p: Point2D) -> Option<f64> {
        let seed = self.cell_of(p);
        // Snap to a coincident site: the weights are singular there anyway.
        if self.site(seed).distance_squared(&p) < 1e-24 {
            return Some(self.value(seed));
        }

        // Grow the candidate set until every facet of `p`'s clipped cell is owned
        // by a candidate — no uncovered site closer to a facet midpoint than `p`.
        let mut candidates = vec![seed];
        candidates.extend_from_slice(self.neighbors(seed));
        candidates.sort_unstable();
        candidates.dedup();

        let boxpoly = self.mesh().bbox.corners_ccw();
        let mut cell: Vec<Point2D> = Vec::new();
        loop {
            self.clip_virtual_cell(p, &candidates, &boxpoly, &mut cell);
            let mut grew = false;
            for e in 0..cell.len() {
                let mid = cell[e].midpoint(&cell[(e + 1) % cell.len()]);
                let dp = mid.distance_squared(&p);
                let nearest = self.cell_of(mid);
                // A site strictly closer than `p` should have clipped this facet;
                // if it is not yet a candidate, add it and rebuild the cell.
                if mid.distance_squared(&self.site(nearest)) < dp - 1e-12
                    && candidates.binary_search(&nearest).is_err()
                {
                    let ins = candidates.partition_point(|&c| c < nearest);
                    candidates.insert(ins, nearest);
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }

        // Accumulate facet lengths per neighbour, weighted by 1/d. A box facet (no
        // owning site) means `p` isn't enclosed; defer to the caller's fallback.
        let mut num = 0.0;
        let mut den = 0.0;
        for e in 0..cell.len() {
            let a = cell[e];
            let b = cell[(e + 1) % cell.len()];
            let mid = a.midpoint(&b);
            let dp = mid.distance(&p);
            // Owning site = candidate nearest the facet midpoint: equidistant with
            // `p` on an interior facet (bisector), farther on a domain-box edge.
            let mut best = candidates[0];
            let mut best_d = mid.distance(&self.site(best));
            for &c in &candidates[1..] {
                let d = mid.distance(&self.site(c));
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if (best_d - dp).abs() > 1e-7 * dp.max(1e-300) {
                return None; // domain-box facet: `p` is not fully surrounded
            }
            // `d_i > 0` (coincident queries returned early); a surrounded query has
            // a non-degenerate cell, so `den` accrues a positive weight.
            let d_i = self.site(best).distance(&p);
            let lambda = a.distance(&b) / d_i;
            num += lambda * self.value(best);
            den += lambda;
        }
        Some(num / den)
    }

    /// Clip the domain box down to the Voronoi cell that `p` would occupy against
    /// `candidates`, writing the polygon into `out`.
    ///
    /// # Arguments
    /// * `p` - the query point whose virtual cell is built.
    /// * `candidates` - the sites whose bisectors with `p` clip the cell.
    /// * `boxpoly` - the domain-box polygon the clip starts from.
    /// * `out` - reused output buffer, overwritten with the clipped cell.
    fn clip_virtual_cell(
        &self,
        p: Point2D,
        candidates: &[usize],
        boxpoly: &[Point2D],
        out: &mut Vec<Point2D>,
    ) {
        let mut src: Vec<Point2D> = boxpoly.to_vec();
        let mut dst: Vec<Point2D> = Vec::with_capacity(src.len() + 1);
        let p_mag2 = p.x * p.x + p.y * p.y;
        for &c in candidates {
            let s = self.site(c);
            // Half-plane of points closer to `p` than to site `s`.
            let n = s - p;
            let off = 0.5 * (s.x * s.x + s.y * s.y - p_mag2);
            clip_halfplane(&src, n.x, n.y, off, &mut dst);
            std::mem::swap(&mut src, &mut dst);
        }
        std::mem::swap(out, &mut src);
    }

    /// Barycentric-linear value at `p` over the Delaunay triangle enclosing it,
    /// or `None` when `p` lies outside every triangle incident to its cell (the
    /// convex hull).
    ///
    /// # Arguments
    /// * `p` - the query point.
    fn barycentric(&self, p: Point2D) -> Option<f64> {
        let c = self.cell_of(p);
        let sc = self.site(c);
        let nb = self.neighbors(c);
        // Enclosing Delaunay triangle: (c, a, b) with a, b mutually adjacent
        // neighbours of c.
        for (ia, &a) in nb.iter().enumerate() {
            for &b in &nb[ia + 1..] {
                if !self.neighbors(a).contains(&b) {
                    continue;
                }
                let (sa, sb) = (self.site(a), self.site(b));
                if p.in_triangle(&sc, &sa, &sb) {
                    let (u, v, w) = p.barycentric(&sc, &sa, &sb);
                    return Some(u * self.value(c) + v * self.value(a) + w * self.value(b));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, FieldArray1D};
    use crate::utils::rng::Lcg;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn unit() -> BBox {
        BBox::from_lims(0.0, 1.0, 0.0, 1.0)
    }

    /// A jittered lattice of sites carrying a linear field `v = 2x + 3y + 1`.
    fn linear_mesh() -> MeshGrid {
        let mut rng = Lcg::new(0x11CE);
        let mut sites = Vec::new();
        for gy in 0..10 {
            for gx in 0..10 {
                let x = 0.05 + 0.1 * gx as f64 + 0.03 * (rng.f() - 0.5);
                let y = 0.05 + 0.1 * gy as f64 + 0.03 * (rng.f() - 0.5);
                sites.push(p(x, y));
            }
        }
        let values: FieldArray1D = sites.iter().map(|s| 2.0 * s.x + 3.0 * s.y + 1.0).collect();
        MeshGrid::new(&sites, unit(), values).unwrap()
    }

    #[test]
    fn reproduces_linear_fields_exactly() {
        // Laplace NN has linear precision: interior queries must recover the
        // underlying plane to rounding.
        let m = linear_mesh();
        let mut rng = Lcg::new(0x5EED);
        for _ in 0..300 {
            // Stay well inside the site convex hull (roughly [0.05, 0.95]).
            let q = p(0.15 + 0.7 * rng.f(), 0.15 + 0.7 * rng.f());
            let exact = 2.0 * q.x + 3.0 * q.y + 1.0;
            let got = m.interpolate(q);
            assert!((got - exact).abs() < 1e-9, "q {q:?}: {got} vs {exact}");
        }
    }

    #[test]
    fn interpolation_at_a_site_returns_its_value() {
        let m = linear_mesh();
        for i in 0..m.len() {
            assert!((m.interpolate(m.site(i)) - m.value(i)).abs() < 1e-9);
        }
    }

    #[test]
    fn out_of_domain_uses_the_fallback() {
        let m = linear_mesh();
        assert_eq!(m.interpolate_or(p(-1.0, 0.5), -7.0), -7.0);
        assert_eq!(m.interpolate(p(2.0, 2.0)), 0.0);
    }

    #[test]
    fn is_continuous_across_a_facet() {
        // Straddling a Voronoi facet, the interpolant must not jump (unlike the
        // piecewise-constant `sample`).
        let m = linear_mesh();
        let a = m.interpolate(p(0.5, 0.5));
        let b = m.interpolate(p(0.5 + 1e-6, 0.5));
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }

    #[test]
    fn parallel_batch_matches_serial() {
        let m = linear_mesh();
        let mut rng = Lcg::new(0xB47C4);
        let pts: Vec<Point2D> = (0..400).map(|_| p(rng.f(), rng.f())).collect();
        assert_eq!(m.interpolate_batch(&pts), m.par_interpolate_batch(&pts));
    }

    #[test]
    fn barycentric_fallback_reproduces_linear_field() {
        // Directly exercise the barycentric path on an interior point.
        let m = linear_mesh();
        let q = p(0.5, 0.5);
        let got = m
            .barycentric(q)
            .expect("interior point has an enclosing triangle");
        assert!((got - (2.0 * q.x + 3.0 * q.y + 1.0)).abs() < 1e-9);
    }

    #[test]
    fn near_boundary_query_uses_barycentric_fallback() {
        // Just inside the cloud's edge, Laplace NN bails (cell borders the box) and
        // the barycentric fallback takes over — both exact for the linear field.
        let m = linear_mesh();
        let q = p(0.06, 0.5);
        assert!(
            m.laplace_nn(q).is_none(),
            "expected Laplace NN to defer near the edge"
        );
        assert!(m.barycentric(q).is_some(), "expected an enclosing triangle");
        let got = m.interpolate(q);
        assert!((got - (2.0 * q.x + 3.0 * q.y + 1.0)).abs() < 1e-9, "{got}");
    }

    #[test]
    fn outside_hull_falls_back_to_nearest() {
        // A query in an empty corner is inside the domain but outside the site
        // convex hull, so neither NN nor barycentric applies: nearest-cell result.
        let sites = [p(0.4, 0.4), p(0.6, 0.4), p(0.5, 0.6), p(0.5, 0.5)];
        let vals = FieldArray1D::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let m = MeshGrid::new(&sites, unit(), vals).unwrap();
        let q = p(0.02, 0.02);
        assert!(m.laplace_nn(q).is_none());
        assert!(m.barycentric(q).is_none());
        assert_eq!(m.interpolate(q), m.value(m.cell_of(q)));
    }
}
