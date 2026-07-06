// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Incremental Delaunay triangulation with Lawson edge-flips (AREPO-style).
//!
//! Points are inserted one at a time into a triangulation seeded with a large
//! enclosing super-triangle. Each insertion locates the containing triangle by
//! walking the neighbour graph, splits it (1→3), and restores the Delaunay
//! property by flipping edges whose opposite vertex fails the [`incircle`] test.
//! The mesh is stored as triangles with vertex indices and per-edge neighbour
//! links, so inserting further points later (refinement) is a local operation.
//!
//! Robustness rests on the exact [`orient2d`]/[`incircle`] predicates. Points
//! that land exactly on an existing edge or vertex are currently handled by a
//! tiny deterministic perturbation (a stand-in for symbolic perturbation); the
//! proper 2→4 split is a planned hardening step.

use super::predicates::{incircle, orient2d};
use crate::grid::Point2D;

/// Sentinel for "no adjacent triangle" (a hull edge).
pub(crate) const NONE: usize = usize::MAX;

/// A triangle: counter-clockwise vertex indices `v`, and neighbour triangle
/// indices `n`, where `n[i]` is the triangle across the edge opposite `v[i]`
/// (i.e. sharing edge `v[(i+1)%3] → v[(i+2)%3]`).
#[derive(Clone, Copy, Debug)]
struct Tri {
    v: [usize; 3],
    n: [usize; 3],
}

/// A Delaunay triangulation of a point set: triangles as counter-clockwise
/// index triples into the input points, with matching neighbour links (`NONE`
/// on the convex hull). The enclosing super-triangle has been removed.
pub(crate) struct Triangulation {
    pub triangles: Vec<[usize; 3]>,
    pub neighbors: Vec<[usize; 3]>,
}

enum Located {
    Inside(usize),
    Edge(usize, usize),
    Vertex,
}

/// Point `pi` (already inside triangle `t`) splits it into three (1→3 flip).
/// Returns the reused slot and pushes the three outer edges for legalisation.
fn insert_inside(tris: &mut Vec<Tri>, pi: usize, t: usize, stack: &mut Vec<(usize, usize)>) -> usize {
    let old = tris[t];
    let [a, b, c] = old.v;
    let [na, nb, nc] = old.n; // opposite a,b,c → edges (b,c),(c,a),(a,b)

    let t1 = t;
    let t2 = tris.len();
    let t3 = t2 + 1;

    tris[t1] = Tri { v: [a, b, pi], n: [t2, t3, nc] };
    tris.push(Tri { v: [b, c, pi], n: [t3, t1, na] });
    tris.push(Tri { v: [c, a, pi], n: [t1, t2, nb] });

    relink(tris, nc, t, t1);
    relink(tris, na, t, t2);
    relink(tris, nb, t, t3);

    stack.push((t1, 2)); // edge (a,b) opposite the new vertex
    stack.push((t2, 2)); // edge (b,c)
    stack.push((t3, 2)); // edge (c,a)
    t1
}

/// Repoint neighbour `t`'s link that referenced `from` to `to`.
#[inline]
fn relink(tris: &mut [Tri], t: usize, from: usize, to: usize) {
    if t == NONE {
        return;
    }
    for k in 0..3 {
        if tris[t].n[k] == from {
            tris[t].n[k] = to;
            return;
        }
    }
}

/// Process the legalisation stack, flipping any edge whose opposite vertex lies
/// inside the circumcircle. Each stack entry `(t, e)` names the edge in slot `e`
/// of triangle `t`, opposite the just-inserted vertex `v[e]`.
fn legalize(tris: &mut Vec<Tri>, coords: &[Point2D], stack: &mut Vec<(usize, usize)>) {
    while let Some((t, e)) = stack.pop() {
        let tn = tris[t].n[e];
        if tn == NONE {
            continue;
        }
        let j = (0..3).find(|&k| tris[tn].n[k] == t).expect("neighbour symmetry");
        let d = tris[tn].v[j];

        let a = coords[tris[t].v[0]];
        let b = coords[tris[t].v[1]];
        let c = coords[tris[t].v[2]];
        if incircle(a, b, c, coords[d]) > 0.0 {
            flip(tris, t, e, tn, j, stack);
        }
    }
}

/// Flip the shared edge between triangles `t` (apex slot `e`) and `tn` (apex
/// slot `j`), reusing both slots, and queue the two new outer edges.
fn flip(tris: &mut [Tri], t: usize, e: usize, tn: usize, j: usize, stack: &mut Vec<(usize, usize)>) {
    let p = tris[t].v[e];
    let u = tris[t].v[(e + 1) % 3];
    let w = tris[t].v[(e + 2) % 3];
    let t_wp = tris[t].n[(e + 1) % 3]; // across edge (w,p)
    let t_pu = tris[t].n[(e + 2) % 3]; // across edge (p,u)

    let d = tris[tn].v[j];
    let t_ud = tris[tn].n[(j + 1) % 3]; // across edge (u,d)
    let t_dw = tris[tn].n[(j + 2) % 3]; // across edge (d,w)

    // Diagonal (u,w) → (p,d): new triangles (p,u,d) at slot t and (p,d,w) at tn.
    tris[t] = Tri { v: [p, u, d], n: [t_ud, tn, t_pu] };
    tris[tn] = Tri { v: [p, d, w], n: [t_dw, t_wp, t] };

    relink(tris, t_ud, tn, t);
    relink(tris, t_wp, t, tn);

    stack.push((t, 0)); // (u,d) opposite p
    stack.push((tn, 0)); // (d,w) opposite p
}

/// Walk the neighbour graph from `start` to the triangle containing `p`.
fn locate(tris: &[Tri], coords: &[Point2D], p: Point2D, start: usize) -> Located {
    let mut t = start;
    for _ in 0..(tris.len() * 3 + 8) {
        let tri = tris[t];
        let a = coords[tri.v[0]];
        let b = coords[tri.v[1]];
        let c = coords[tri.v[2]];
        let d0 = orient2d(b, c, p); // edge opposite v0
        let d1 = orient2d(c, a, p); // edge opposite v1
        let d2 = orient2d(a, b, p); // edge opposite v2

        if d0 < 0.0 {
            t = tri.n[0];
            continue;
        }
        if d1 < 0.0 {
            t = tri.n[1];
            continue;
        }
        if d2 < 0.0 {
            t = tri.n[2];
            continue;
        }
        // Inside or on the boundary of `t`.
        let zeros = (d0 == 0.0) as u8 + (d1 == 0.0) as u8 + (d2 == 0.0) as u8;
        if zeros >= 2 {
            return Located::Vertex;
        }
        if d0 == 0.0 {
            return Located::Edge(t, 0);
        }
        if d1 == 0.0 {
            return Located::Edge(t, 1);
        }
        if d2 == 0.0 {
            return Located::Edge(t, 2);
        }
        return Located::Inside(t);
    }
    Located::Inside(t)
}

/// Deterministic sub-ULP-scale perturbation used when a point lands exactly on
/// an edge/vertex (temporary stand-in for a proper 2→4 split).
fn jitter(p: Point2D, scale: f64, seed: usize) -> Point2D {
    let h = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let ax = (h >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let ay = (h >> 33) as f64 / (1u64 << 21) as f64 - 0.5;
    let e = scale * 1e-9;
    Point2D::new(p.x + e * ax, p.y + e * ay)
}

/// Build the Delaunay triangulation of `points`.
pub(crate) fn triangulate(points: &[Point2D]) -> Triangulation {
    let n = points.len();
    if n < 3 {
        return Triangulation {
            triangles: Vec::new(),
            neighbors: Vec::new(),
        };
    }

    let (mut minx, mut miny) = (f64::INFINITY, f64::INFINITY);
    let (mut maxx, mut maxy) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in points {
        minx = minx.min(p.x);
        miny = miny.min(p.y);
        maxx = maxx.max(p.x);
        maxy = maxy.max(p.y);
    }
    let dmax = (maxx - minx).max(maxy - miny).max(1.0);
    let cx = 0.5 * (minx + maxx);
    let cy = 0.5 * (miny + maxy);
    let m = 1000.0 * dmax;

    let mut coords: Vec<Point2D> = points.to_vec();
    let (s0, s1, s2) = (n, n + 1, n + 2);
    coords.push(Point2D::new(cx - m, cy - m));
    coords.push(Point2D::new(cx + m, cy - m));
    coords.push(Point2D::new(cx, cy + m));

    // super-triangle (bottom-left, bottom-right, top) is counter-clockwise
    let mut tris: Vec<Tri> = vec![Tri { v: [s0, s1, s2], n: [NONE, NONE, NONE] }];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut last = 0usize;

    for pi in 0..n {
        let mut p = coords[pi];
        let t = match locate(&tris, &coords, p, last) {
            Located::Inside(t) => t,
            Located::Vertex => continue, // duplicate site
            Located::Edge(t, _) => {
                // Perturb off the edge and relocate (temporary degeneracy path).
                p = jitter(p, dmax, pi + 1);
                coords[pi] = p;
                match locate(&tris, &coords, p, t) {
                    Located::Inside(t2) => t2,
                    _ => continue,
                }
            }
        };
        last = insert_inside(&mut tris, pi, t, &mut stack);
        legalize(&mut tris, &coords, &mut stack);
    }

    finalize(&tris, n)
}

/// Drop super-triangle triangles and remap neighbour indices to the survivors.
fn finalize(tris: &[Tri], n: usize) -> Triangulation {
    let mut new_index = vec![NONE; tris.len()];
    let mut count = 0usize;
    for (i, t) in tris.iter().enumerate() {
        if t.v.iter().all(|&v| v < n) {
            new_index[i] = count;
            count += 1;
        }
    }

    let mut triangles = Vec::with_capacity(count);
    let mut neighbors = Vec::with_capacity(count);
    for (i, t) in tris.iter().enumerate() {
        if new_index[i] == NONE {
            continue;
        }
        triangles.push(t.v);
        let mut nb = [NONE; 3];
        for k in 0..3 {
            nb[k] = if t.n[k] == NONE { NONE } else { new_index[t.n[k]] };
        }
        neighbors.push(nb);
    }
    Triangulation { triangles, neighbors }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
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

    /// Every triangle is counter-clockwise and its circumcircle contains no
    /// other input point (the defining Delaunay property).
    fn assert_delaunay(points: &[Point2D], tri: &Triangulation) {
        for t in &tri.triangles {
            let (a, b, c) = (points[t[0]], points[t[1]], points[t[2]]);
            assert!(orient2d(a, b, c) > 0.0, "triangle not ccw: {t:?}");
            for (q, &pt) in points.iter().enumerate() {
                if q == t[0] || q == t[1] || q == t[2] {
                    continue;
                }
                assert!(
                    incircle(a, b, c, pt) <= 0.0,
                    "point {q} inside circumcircle of {t:?}"
                );
            }
        }
    }

    /// Neighbour links are symmetric.
    fn assert_neighbor_symmetry(tri: &Triangulation) {
        for (i, nb) in tri.neighbors.iter().enumerate() {
            for &t in nb {
                if t != NONE {
                    assert!(
                        tri.neighbors[t].iter().any(|&b| b == i),
                        "neighbour {t} does not point back to {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn single_triangle() {
        let pts = [p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0)];
        let t = triangulate(&pts);
        assert_eq!(t.triangles.len(), 1);
        assert_delaunay(&pts, &t);
    }

    #[test]
    fn square_two_triangles() {
        let pts = [p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)];
        let t = triangulate(&pts);
        assert_eq!(t.triangles.len(), 2);
        assert_delaunay(&pts, &t);
        assert_neighbor_symmetry(&t);
    }

    #[test]
    fn random_points_are_delaunay() {
        let mut rng = Lcg(0xABCDEF);
        let pts: Vec<Point2D> = (0..150).map(|_| p(rng.f(), rng.f())).collect();
        let t = triangulate(&pts);
        // For points in general position: #triangles = 2n - 2 - h (h = hull pts).
        assert!(t.triangles.len() > 200, "got {} triangles", t.triangles.len());
        assert_delaunay(&pts, &t);
        assert_neighbor_symmetry(&t);
    }

    #[test]
    fn grid_points_do_not_panic() {
        // A regular grid is maximally degenerate (cocircular quads); exercises
        // the on-edge perturbation path.
        let mut pts = Vec::new();
        for j in 0..8 {
            for i in 0..8 {
                pts.push(p(i as f64, j as f64));
            }
        }
        let t = triangulate(&pts);
        assert!(!t.triangles.is_empty());
        assert_neighbor_symmetry(&t);
    }
}
