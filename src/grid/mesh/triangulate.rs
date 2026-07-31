// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Delaunay triangulation with incremental edge-flips.
//!
//! Resource: Berg et al., Computational Geometry (Ch. 9.3),
//! [DOI:10.1007/978-3-540-77974-2](https://doi.org/10.1007/978-3-540-77974-2)
//!
//! Points are inserted one at a time into a triangulation seeded with a large
//! enclosing super-triangle. Each insertion locates the containing triangle by
//! walking the neighbour graph and splits it (1→3 interior, 2→4 on an edge,
//! 1→2 on the hull); the Delaunay property is restored by flipping edges whose
//! opposite vertex fails an [`incircle`] test. Triangles store vertex indices and
//! per-edge neighbour links, so later insertion (refinement) stays local.
//!
//! [`TriangulationBuilder`] holds the mutable working state (coordinates behind an
//! enclosing super-triangle) and supports
//! [`insert`](TriangulationBuilder::insert) one site at a time — for the initial
//! build or later local refinement — reusing
//! [`locate`]/[`insert_inside`]/[`insert_on_edge`]/[`legalize`].
//! [`finalize`](TriangulationBuilder::finalize) snapshots it into the read-only
//! [`Triangulation`] (super-triangle removed, indices remapped) without consuming
//! the builder. Moving-mesh (Lloyd/CVT) regularization rebuilds from scratch.

use super::predicates::{incircle, orient2d};
use crate::grid::{BBox, Point2D};

/// Sentinel for "no adjacent triangle" (a hull edge).
pub(crate) const NO_NEIGHBOR: usize = usize::MAX;

/// A triangle: counter-clockwise `vertices`, and `neighbors` triangle indices,
/// where `neighbors[i]` is the triangle across the edge opposite `vertices[i]`
/// (i.e. sharing edge `vertices[(i+1)%3] to vertices[(i+2)%3]`).
#[derive(Clone, Copy, Debug)]
struct Triangle {
    vertices: [usize; 3],
    neighbors: [usize; 3],
}

/// A Delaunay triangulation of a point set: triangles as counter-clockwise
/// index triples into the input points, with matching neighbour links.
pub(crate) struct Triangulation {
    pub triangles: Vec<[usize; 3]>,
    pub neighbors: Vec<[usize; 3]>,
}

/// Mutable Delaunay triangulation state for incremental construction and local
/// refinement.
///
/// Holds the site coordinates behind an enclosing super-triangle plus the working
/// triangle list, so [`insert`](Self::insert) can add one site at a time — during
/// the initial build or later, to refine locally — reusing
/// [`locate`]/[`insert_inside`]/[`insert_on_edge`]/[`legalize`].
/// [`finalize`](Self::finalize) snapshots it into a read-only [`Triangulation`]
/// without consuming the builder, so refinement can continue afterwards.
pub(crate) struct TriangulationBuilder {
    /// Super-triangle vertices at `[0, 1, 2]`, inserted sites at `[3..]`.
    coords: Vec<Point2D>,
    /// Working triangles over `coords` (still including the super-triangle).
    tris: Vec<Triangle>,
    /// Last-touched triangle, warm-starting the next [`locate`] walk.
    last: usize,
    /// Reused legalization work list.
    stack: Vec<(usize, usize)>,
}

impl TriangulationBuilder {
    /// Create an empty triangulation whose super-triangle encloses `domain`.
    ///
    /// # Arguments
    /// * `domain` - a box containing every point that will later be inserted; the
    ///   super-triangle is sized generously around it.
    pub(crate) fn new(domain: BBox) -> Self {
        let dmax = domain.width().max(domain.height()).max(1.0);
        let c = domain.center();
        let m = 1000.0 * dmax;
        // super-triangle (bottom-left, bottom-right, top) is counter-clockwise
        let coords = vec![
            Point2D::new(c.x - m, c.y - m),
            Point2D::new(c.x + m, c.y - m),
            Point2D::new(c.x, c.y + m),
        ];
        let tris = vec![Triangle {
            vertices: [0, 1, 2],
            neighbors: [NO_NEIGHBOR; 3],
        }];
        Self { coords, tris, last: 0, stack: Vec::new() }
    }

    /// Insert `p`, restore the Delaunay property, and return the site index
    /// assigned to it (dense, in insertion order).
    ///
    /// `p` must lie within the `domain` passed to [`new`](Self::new). A point
    /// coinciding with an existing site is not triangulated; its returned index is
    /// then an orphan carrying no triangle (the handling [`triangulate`] gives
    /// duplicate inputs).
    ///
    /// # Arguments
    /// * `p` - the site to insert.
    pub(crate) fn insert(&mut self, p: Point2D) -> usize {
        let pidx = self.coords.len();
        self.coords.push(p);
        self.last = match locate(&self.tris, &self.coords, p, self.last) {
            Located::Inside(t) => insert_inside(&mut self.tris, t, pidx, &mut self.stack),
            Located::Edge(t, e) => insert_on_edge(&mut self.tris, t, e, pidx, &mut self.stack),
            Located::Vertex => self.last, // duplicate: orphan coord, mesh unchanged
        };
        legalize(&mut self.tris, &self.coords, &mut self.stack);
        pidx - 3
    }

    /// Snapshot the current triangulation over the inserted sites as a read-only
    /// [`Triangulation`], dropping the super-triangle and remapping vertex indices
    /// to site indices. Leaves the builder intact for further insertion.
    pub(crate) fn finalize(&self) -> Triangulation {
        finalize(&self.tris)
    }
}

/// Point locations relative to triangles.
enum Located {
    // interior points: 1→3 flip
    Inside(usize),
    // edge points: 2→4 or 1→2 split
    Edge(usize, usize),
    // duplicate site (skip)
    Vertex,
}

/// Repoint neighbour `t`'s link that referenced `from` to `to`.
#[inline]
fn relink(tris: &mut [Triangle], t: usize, from: usize, to: usize) {
    if t == NO_NEIGHBOR {
        return;
    }
    for k in 0..3 {
        if tris[t].neighbors[k] == from {
            tris[t].neighbors[k] = to;
            return;
        }
    }
}

/// Insert a point lying strictly inside triangle `t`, replacing it with three
/// triangles fanning out from `pidx` (the 1→3 flip).
///
/// ```text
///        before insert            after insert (1→3 split)
///              a                           a
///             / \                         /|\
///            /   \                       / | \
///           /     \                     /  |  \
///          /       \                   /   |   \
///         /         \        →        /    p    \
///        /           \               / t1 ╱ ╲ t3 \
///       /      t      \             /   ╱     ╲   \
///      /               \           /  ╱         ╲  \
///     /                 \         / ╱     t2      ╲ \
///    /                   \       /╱                 ╲\
///   b─────────────────────c     b─────────────────────c
/// ```
///
/// # Arguments
/// * `tris` - working triangle list, grown in place (two triangles appended).
/// * `t` - the containing triangle; reused as one child's slot.
/// * `pidx` - index of the point to insert, known to lie strictly inside `t`.
/// * `stack` - legalization work list; the three outer edges are pushed as
///   `(triangle, slot)` pairs, each slot naming the edge opposite `pidx`.
///
/// Returns the slot of one child (`t`) as the start for the next [`locate`] walk.
fn insert_inside(
    tris: &mut Vec<Triangle>,
    t: usize,
    pidx: usize,
    stack: &mut Vec<(usize, usize)>,
) -> usize {
    // Gather vertices and edges
    let [a, b, c] = tris[t].vertices;
    let [na, nb, nc] = tris[t].neighbors; // opposite i, edges (i+1)%3 to (i+2)%3

    // Allocate slot for new triangles
    let t1 = t;
    let t2 = tris.len();
    let t3 = t2 + 1;

    // Build the new triangles around `pidx`
    tris[t1] = Triangle {
        vertices: [a, b, pidx],
        neighbors: [t2, t3, nc],
    };
    tris.push(Triangle {
        vertices: [b, c, pidx],
        neighbors: [t3, t1, na],
    });
    tris.push(Triangle {
        vertices: [c, a, pidx],
        neighbors: [t1, t2, nb],
    });

    // Update neighbour pointing to new triangles
    relink(tris, nc, t, t1);
    relink(tris, na, t, t2);
    relink(tris, nb, t, t3);
    // Push edges to legalization stack
    stack.push((t1, 2)); // edge (a,b) opposite the new vertex
    stack.push((t2, 2)); // edge (b,c)
    stack.push((t3, 2)); // edge (c,a)
    t1 // next locate
}

/// Insert a point lying exactly on the edge in slot `e` of triangle `t`,
/// splitting both triangles sharing that edge into four, a 2→4 flip; on a hull
/// edge only the one triangle is split into two, a 1→2 flip.
///
/// ```text
///   before insert (b–c is hull)   after insert (1→2 split)
///          a                             a
///         / \                           /│\
///        /   \                         / │ \
///       /  t  \           →           /t1│t2\
///      /       \                     /   │   \
///     b─────────c                   b────p────c
///      hull edge                     hull: b–p, p–c
///
///
///   before insert                 after insert (2→4 split)
///          a                             a
///         / \                           /│\
///        /   \                         / │ \
///       /  t  \                       /t1│t2\
///      /       \                     /   │   \
///     b─────────c         →         b────p────c
///      \       /                     \   │   /
///       \ t_bc/                       \t4│t3/
///        \   /                         \ │ /
///         \ /                           \│/
///          d                             d
/// ```
///
/// # Arguments
/// * `tris` - working triangle list, grown in place (one or two triangles appended).
/// * `t` - the triangle whose edge is being split; reused as one child's slot.
/// * `e` - slot of the split edge (the edge opposite vertex `t.vertices[e]`).
/// * `pidx` - index of the point to insert, known to lie exactly on edge `e` of `t`.
/// * `stack` - legalization work list; the outer edges are pushed as
///   `(triangle, slot)` pairs, each slot naming the edge opposite `pidx`.
///
/// Returns the slot of one child (`t`) as the start for the next [`locate`] walk.
fn insert_on_edge(
    tris: &mut Vec<Triangle>,
    t: usize,
    e: usize,
    pidx: usize,
    stack: &mut Vec<(usize, usize)>,
) -> usize {
    // Gather vertices and edges
    let a = tris[t].vertices[e]; // apex opposite the split edge (b,c)
    let b = tris[t].vertices[(e + 1) % 3];
    let c = tris[t].vertices[(e + 2) % 3];
    let t_bc = tris[t].neighbors[e]; // across the split edge (b,c)
    let t_ca = tris[t].neighbors[(e + 1) % 3]; // across edge (c,a)
    let t_ab = tris[t].neighbors[(e + 2) % 3]; // across edge (a,b)

    // Allocate slot for new triangles
    let t1 = t;
    let t2 = tris.len();

    // Hull edge: 1→2 split with `pidx` as new vertex
    if t_bc == NO_NEIGHBOR {
        tris[t1] = Triangle {
            vertices: [a, b, pidx],
            neighbors: [NO_NEIGHBOR, t2, t_ab],
        };
        tris.push(Triangle {
            vertices: [a, pidx, c],
            neighbors: [NO_NEIGHBOR, t_ca, t1],
        });

        relink(tris, t_ca, t, t2);

        stack.push((t1, 2)); // edge (a,b) opposite the new vertex
        stack.push((t2, 1)); // edge (c,a)
        return t1;
    }

    // Interior edge: 2→4 split around `pidx`
    let apex_slot = (0..3)
        .find(|&k| tris[t_bc].neighbors[k] == t)
        .expect("neighbour symmetry");
    let opposite_t = tris[t_bc];
    let d = opposite_t.vertices[apex_slot];
    let t_bd = opposite_t.neighbors[(apex_slot + 1) % 3]; // across edge (b,d)
    let t_dc = opposite_t.neighbors[(apex_slot + 2) % 3]; // across edge (d,c)

    // Allocate slot for other triangles
    let t3 = t_bc;
    let t4 = tris.len() + 1;

    // Build the new triangles around `pidx`
    tris[t1] = Triangle {
        vertices: [a, b, pidx],
        neighbors: [t4, t2, t_ab],
    };
    tris.push(Triangle {
        vertices: [a, pidx, c],
        neighbors: [t3, t_ca, t1],
    });
    tris[t3] = Triangle {
        vertices: [d, c, pidx],
        neighbors: [t2, t4, t_dc],
    };
    tris.push(Triangle {
        vertices: [d, pidx, b],
        neighbors: [t1, t_bd, t3],
    });

    // Update neighbour pointing to new triangles
    relink(tris, t_ca, t, t2);
    relink(tris, t_bd, t_bc, t4);
    // Push edges to legalization stack
    stack.push((t1, 2)); // edge (a,b) opposite pidx
    stack.push((t2, 1)); // edge (c,a)
    stack.push((t3, 2)); // edge (d,c)
    stack.push((t4, 1)); // edge (b,d)
    t1
}

/// Flip the edge shared by triangles `t` and `tn`, swapping the diagonal `(u,w)`
/// for `(p,d)`, reusing both slots and queuing the two new edges opposite `p`.
///
/// ```text
///   before flip         after flip
///        p                   p
///       / \                 /|\
///      /   \               / | \
///     /     \             /  |  \
///    /   t   \           /   |   \
///   u─────────w    →    u  t | tn w
///    \       /           \   |   /
///     \  tn /             \  |  /
///      \   /               \ | /
///       \ /                 \|/
///        d                   d
/// ```
///
/// # Arguments
/// * `tris` - working triangle list; both triangles are rewritten in place.
/// * `t` - triangle on the inserted-vertex side; reused for new triangle `(p,u,d)`.
/// * `e` - slot of apex `p` in `t` (the shared edge is opposite it).
/// * `tn` - the adjacent triangle across the shared edge; reused for `(p,d,w)`.
/// * `apex` - slot of apex `d` in `tn` (the shared edge is opposite it).
/// * `stack` - legalization work list; the two new edges opposite `p` are pushed.
fn flip(
    tris: &mut [Triangle],
    t: usize,
    e: usize,
    tn: usize,
    apex_slot: usize,
    stack: &mut Vec<(usize, usize)>,
) {
    // Gather vertices and edges
    let p = tris[t].vertices[e];
    let u = tris[t].vertices[(e + 1) % 3];
    let w = tris[t].vertices[(e + 2) % 3];
    let t_wp = tris[t].neighbors[(e + 1) % 3]; // across edge (w,p)
    let t_pu = tris[t].neighbors[(e + 2) % 3]; // across edge (p,u)

    let d = tris[tn].vertices[apex_slot];
    let t_ud = tris[tn].neighbors[(apex_slot + 1) % 3]; // across edge (u,d)
    let t_dw = tris[tn].neighbors[(apex_slot + 2) % 3]; // across edge (d,w)

    // Build triangles with new edge (p,d)
    tris[t] = Triangle {
        vertices: [p, u, d],
        neighbors: [t_ud, tn, t_pu],
    };
    tris[tn] = Triangle {
        vertices: [p, d, w],
        neighbors: [t_dw, t_wp, t],
    };
    // Update neighbour pointing to new triangles
    relink(tris, t_ud, tn, t);
    relink(tris, t_wp, t, tn);
    // Push edges to legalization stack
    stack.push((t, 0)); // (u,d) opposite p
    stack.push((tn, 0)); // (d,w) opposite p
}

/// Restore the Delaunay property after an insertion, [`flip`]ping every stacked
/// edge whose opposite vertex fails the empty-circumcircle ([`incircle`]) test.
///
/// # Arguments
/// * `tris` - working triangle list; edges are flipped in place.
/// * `coords` - point coordinates, indexed by the vertex indices stored in `tris`.
/// * `stack` - legalization working list; the outer edges are stored as
///   `(triangle, slot)` pairs, opposite the inserted vertex `tris[t].vertices[e]`.
fn legalize(tris: &mut Vec<Triangle>, coords: &[Point2D], stack: &mut Vec<(usize, usize)>) {
    while let Some((t, e)) = stack.pop() {
        let tn = tris[t].neighbors[e];
        // edge on the convex hull
        if tn == NO_NEIGHBOR {
            continue;
        }
        // find opposite triangle sharing edge with `tn`
        let apex_slot = (0..3)
            .find(|&k| tris[tn].neighbors[k] == t)
            .expect("neighbour symmetry");
        let d = coords[tris[tn].vertices[apex_slot]];
        // test whether `d` (far apex) lies strictly inside the circumcircle
        // of `t` (consisting of vertices a,b,c) → illegal
        let a = coords[tris[t].vertices[0]];
        let b = coords[tris[t].vertices[1]];
        let c = coords[tris[t].vertices[2]];
        if incircle(a, b, c, d) > 0.0 {
            flip(tris, t, e, tn, apex_slot, stack);
        }
    }
}

/// Turn edge orientations `d` into a [`Located`] category for the triangle `t`:
/// two zeros → vertex (duplicate), one zero → on that edge, none → interior.
///
/// # Arguments
/// * `t` - the triangle known to contain the query point.
/// * `d` - the three signed edge orientations, `d[i]` being [`orient2d`] of the
///   point against the edge opposite `vertices[i]`.
fn classify(t: usize, d: [f64; 3]) -> Located {
    let zeros = (d[0] == 0.0) as u8 + (d[1] == 0.0) as u8 + (d[2] == 0.0) as u8;
    if zeros >= 2 {
        return Located::Vertex;
    }
    if d[0] == 0.0 {
        return Located::Edge(t, 0);
    }
    if d[1] == 0.0 {
        return Located::Edge(t, 1);
    }
    if d[2] == 0.0 {
        return Located::Edge(t, 2);
    }
    Located::Inside(t)
}

/// Locate the triangle containing `p` with a straight (Green-Sibson) walk along
/// the segment from `start`'s centroid toward `p`, crossing one edge per step
/// until `p` is contained.
///
/// # Arguments
/// * `tris` - triangle list to walk through.
/// * `coords` - coordinates, indexed by the vertex indices stored in `tris`.
/// * `p` - the query point to locate.
/// * `start` - triangle slot to begin the walk from (passing the previously
///   touched triangle exploits spatial locality to keep successive walks short).
///
/// Returns a [`Located`] category: {`Inside(t)`, `Edge(t, e)`, or `Vertex`}.
fn locate(tris: &[Triangle], coords: &[Point2D], p: Point2D, start: usize) -> Located {
    // Source point: centroid of `start`
    let start_vertices = tris[start].vertices;
    let source = Point2D::centroid(&[
        coords[start_vertices[0]],
        coords[start_vertices[1]],
        coords[start_vertices[2]],
    ])
    .expect("a triangle has three vertices");

    // Orientation closure of `p` against the ccw edge opposite `slot` of `tri`
    let edge_orient = |tri: &Triangle, slot: usize| {
        let u = coords[tri.vertices[(slot + 1) % 3]];
        let w = coords[tri.vertices[(slot + 2) % 3]];
        orient2d(u, w, p)
    };
    // Whether vertex `v` lies to the left of the directed ray source→p
    let left_of_walk = |v: usize| orient2d(source, p, coords[v]) > 0.0;

    let mut current = start;
    // Slot of the edge just crossed into `current` (inner by construction).
    let mut entry_slot: Option<usize> = None;
    // Debug canary: a straight walk crosses each triangle at most once.
    #[cfg(debug_assertions)]
    let mut steps_remaining = tris.len() + 8;

    loop {
        #[cfg(debug_assertions)]
        {
            steps_remaining = steps_remaining
                .checked_sub(1)
                .expect("locate: straight walk did not terminate");
        }
        let tri = tris[current];

        let exit_slot = match entry_slot {
            // Full containment test; if outside, leave along the ray source→p
            None => {
                let orient = [
                    edge_orient(&tri, 0),
                    edge_orient(&tri, 1),
                    edge_orient(&tri, 2),
                ];
                if orient[0] >= 0.0 && orient[1] >= 0.0 && orient[2] >= 0.0 {
                    return classify(current, orient);
                }
                (0..3)
                    .find(|&i| {
                        orient[i] < 0.0
                            && left_of_walk(tri.vertices[(i + 1) % 3])
                                != left_of_walk(tri.vertices[(i + 2) % 3])
                    })
                    .expect("segment from interior source exits through one edge")
            }
	    // Only two edges sharing apex opposite entry edge can exclude `p`
            Some(entry) => {
                let (edge1, edge2) = ((entry + 1) % 3, (entry + 2) % 3);
                let orient1 = edge_orient(&tri, edge1);
                let orient2 = edge_orient(&tri, edge2);
                match (orient1 < 0.0, orient2 < 0.0) {
                    // Inner to both: classify with all three.
                    (false, false) => {
                        let mut orient = [0.0; 3];
                        orient[edge1] = orient1;
                        orient[edge2] = orient2;
                        orient[entry] = edge_orient(&tri, entry);
                        return classify(current, orient);
                    }
		    // Beyond `edge1`
                    (true, false) => edge1,
		    // Beyond `edge2`
                    (false, true) => edge2,
                    // Outer to both: disambiguate with the ray against both
                    (true, true) => {
                        if left_of_walk(tri.vertices[entry])
                            != left_of_walk(tri.vertices[(entry + 2) % 3])
                        {
                            edge1
                        } else {
                            edge2
                        }
                    }
                }
            }
        };
	// The ray source→p crosses finitely many edges (termination guaranteed)
        let next = tri.neighbors[exit_slot];
        debug_assert!(next != NO_NEIGHBOR, "locate: walk left the triangulation");
        entry_slot = Some(
            (0..3)
                .find(|&k| tris[next].neighbors[k] == current)
                .expect("neighbour symmetry"),
        );
        current = next;
    }
}

/// Axis-aligned bounding box of `points` (assumed non-empty).
fn initialize_domain(points: &[Point2D]) -> BBox {
    let (mut minx, mut miny) = (f64::INFINITY, f64::INFINITY);
    let (mut maxx, mut maxy) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in points {
        minx = minx.min(p.x);
        miny = miny.min(p.y);
        maxx = maxx.max(p.x);
        maxy = maxy.max(p.y);
    }
    BBox::from_lims(minx, maxx, miny, maxy)
}

/// Snapshot `tris` into a [`Triangulation`] over the sites: drop every triangle
/// still touching the super-triangle (any vertex `< 3`), remap the remaining
/// vertices to site indices (`v - 3`), and turn links to dropped triangles into
/// [`NO_NEIGHBOR`] hull edges.
///
/// # Arguments
/// * `tris` - the working triangle list, super-triangle at vertices `[0, 1, 2]`.
fn finalize(tris: &[Triangle]) -> Triangulation {
    let mut new_index: Vec<Option<usize>> = vec![None; tris.len()];
    let mut count = 0usize;
    for (i, t) in tris.iter().enumerate() {
        if t.vertices.iter().all(|&v| v >= 3) {
            new_index[i] = Some(count);
            count += 1;
        }
    }
    // Allocate triangulation components (vertices and neighbours)
    let mut triangles = Vec::with_capacity(count);
    let mut neighbors = Vec::with_capacity(count);
    for (i, t) in tris.iter().enumerate() {
        if new_index[i].is_none() {
            continue;
        }
        // Site indices drop the three super-triangle vertices.
        triangles.push([t.vertices[0] - 3, t.vertices[1] - 3, t.vertices[2] - 3]);
        // Links to dropped triangles become hull edges.
        let mut nb = [NO_NEIGHBOR; 3];
        for k in 0..3 {
            if t.neighbors[k] != NO_NEIGHBOR {
                nb[k] = new_index[t.neighbors[k]].unwrap_or(NO_NEIGHBOR);
            }
        }
        neighbors.push(nb);
    }
    Triangulation { triangles, neighbors }
}

/// Build the Delaunay triangulation of `points` by incremental insertion.
///
/// # Arguments
/// * `points` - the sites to triangulate, indexed by the returned index triples.
///
/// Returns the [`Triangulation`] (counter-clockwise index triples with matching
/// neighbour links); empty when fewer than three points are given.
pub(crate) fn triangulate(points: &[Point2D]) -> Triangulation {
    // Only triangulate with enough points
    if points.len() < 3 {
        return Triangulation {
            triangles: Vec::new(),
            neighbors: Vec::new(),
        };
    }
    // Insert every point into a builder sized to enclose them, then snapshot.
    let mut builder = TriangulationBuilder::new(initialize_domain(points));
    for &p in points {
        builder.insert(p);
    }
    builder.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rng::Lcg;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
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

    /// Neighbor links are symmetric.
    fn assert_neighbor_symmetry(tri: &Triangulation) {
        for (i, nb) in tri.neighbors.iter().enumerate() {
            for &t in nb {
                if t != NO_NEIGHBOR {
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
        let mut rng = Lcg::new(0xABCDEF);
        let pts: Vec<Point2D> = (0..150).map(|_| p(rng.f(), rng.f())).collect();
        let t = triangulate(&pts);
        // For points in general position: #triangles = 2n - 2 - h (h = hull pts).
        assert!(
            t.triangles.len() > 200,
            "got {} triangles",
            t.triangles.len()
        );
        assert_delaunay(&pts, &t);
        assert_neighbor_symmetry(&t);
    }

    #[test]
    fn duplicate_points_are_skipped() {
        // A repeated site must be located as `Vertex` and skipped, exercising the
        // straight walk's arrival-on-a-vertex path rather than being reinserted.
        let pts = [
            p(0.0, 0.0),
            p(1.0, 0.0),
            p(0.0, 1.0),
            p(1.0, 0.0), // duplicate of index 1
            p(0.5, 0.25),
        ];
        let t = triangulate(&pts);
        assert!(!t.triangles.is_empty());
        assert_delaunay(&pts, &t);
        assert_neighbor_symmetry(&t);
    }

    #[test]
    fn grid_points_are_delaunay() {
        // A regular grid is maximally degenerate (collinear rows, cocircular
        // quads), stressing the exact orientation and in-circle predicates.
        let mut pts = Vec::new();
        for j in 0..8 {
            for i in 0..8 {
                pts.push(p(i as f64, j as f64));
            }
        }
        let t = triangulate(&pts);
        assert!(!t.triangles.is_empty());
        assert_delaunay(&pts, &t);
        assert_neighbor_symmetry(&t);
    }

    #[test]
    fn point_exactly_on_edge_triggers_split() {
        // The last point lies on the interior edge (0,0)-(2,0), so it is located as
        // `Edge` and inserted via the on-edge (2→4) split, not the interior 1→3 fan.
        let pts = [
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(1.0, 2.0),
            p(1.0, 0.0), // collinear with and between the first two
        ];
        let t = triangulate(&pts);
        assert_delaunay(&pts, &t);
        assert_neighbor_symmetry(&t);
        // The on-edge point was inserted (used as a vertex), not skipped.
        assert!(t.triangles.iter().any(|tri| tri.contains(&3)), "on-edge point missing");
    }

    #[test]
    fn points_on_all_three_edges() {
        // A midpoint on each edge of the seed triangle, so the on-edge split is
        // reached for edges in different slots of the located triangle.
        let pts = [
            p(0.0, 0.0),
            p(4.0, 0.0),
            p(0.0, 4.0),
            p(2.0, 0.0), // on edge (0,0)-(4,0)
            p(0.0, 2.0), // on edge (0,0)-(0,4)
            p(2.0, 2.0), // on edge (4,0)-(0,4)
        ];
        let t = triangulate(&pts);
        assert_delaunay(&pts, &t);
        assert_neighbor_symmetry(&t);
        for v in 3..6 {
            assert!(t.triangles.iter().any(|tri| tri.contains(&v)), "point {v} missing");
        }
    }

    #[test]
    fn fewer_than_three_points_is_empty() {
        for pts in [&[][..], &[p(0.0, 0.0)][..], &[p(0.0, 0.0), p(1.0, 1.0)][..]] {
            let t = triangulate(pts);
            assert!(t.triangles.is_empty() && t.neighbors.is_empty());
        }
    }

    #[test]
    fn insert_after_finalize_refines_locally() {
        // Build a coarse mesh, snapshot it, then keep inserting sites into the same
        // builder (local refinement). Each snapshot stays Delaunay over its sites,
        // and finalize does not consume the builder.
        let bbox = BBox::from_lims(0.0, 1.0, 0.0, 1.0);
        let mut b = TriangulationBuilder::new(bbox);
        let mut all: Vec<Point2D> = Vec::new();
        for &q in &[p(0.05, 0.05), p(0.95, 0.05), p(0.95, 0.95), p(0.05, 0.95), p(0.5, 0.5)] {
            b.insert(q);
            all.push(q);
        }
        let coarse = b.finalize();
        assert_delaunay(&all, &coarse);

        // Refine near one corner; the warm-started walk re-enters the built mesh.
        for &q in &[p(0.2, 0.2), p(0.3, 0.25), p(0.25, 0.35)] {
            let idx = b.insert(q);
            assert_eq!(idx, all.len()); // dense site index in insertion order
            all.push(q);
        }
        let refined = b.finalize();
        assert!(refined.triangles.len() > coarse.triangles.len());
        assert_delaunay(&all, &refined);
        assert_neighbor_symmetry(&refined);
        for v in 0..all.len() {
            assert!(refined.triangles.iter().any(|tri| tri.contains(&v)), "site {v} missing");
        }
    }

    #[test]
    fn reinserting_an_existing_site_is_a_noop() {
        // A point coinciding with an existing site is located as a vertex and left
        // out (the `insert` duplicate path), so the mesh is unchanged.
        let bbox = BBox::from_lims(0.0, 1.0, 0.0, 1.0);
        let sites = [p(0.1, 0.1), p(0.9, 0.1), p(0.9, 0.9), p(0.1, 0.9), p(0.5, 0.5)];
        let mut b = TriangulationBuilder::new(bbox);
        for &q in &sites {
            b.insert(q);
        }
        let before = b.finalize();
        b.insert(p(0.5, 0.5)); // duplicate of site 4
        let after = b.finalize();
        assert_eq!(before.triangles, after.triangles, "duplicate insert changed the mesh");
        assert_delaunay(&sites, &after);
    }
}
