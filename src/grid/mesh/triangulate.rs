// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Delaunay triangulation with incremental edge-flips.
//!
//! Points are inserted one at a time into a triangulation seeded with a large
//! enclosing super-triangle. Each insertion locates the containing triangle by
//! walking the neighbour graph and splits it: 1→3 for interior points, or
//! 2→4 for edge points, 1→2 on the hull; the Delaunay property is
//! restored by flipping edges whose opposite vertex fails an [`incircle`] test.
//! The mesh is stored as triangles with vertex indices and per-edge neighbour
//! links, so inserting further points later (refinement) is a local operation.
//!
//! Resource:
//! Berg et al., Computational Geometry (Ch. 9.3), DOI:10.1007/978-3-540-77974-2
//!
//! Design note (refinement): [`Triangulation`] is the *finalized, immutable*
//! snapshot — super-triangle removed, indices remapped to the input points — and
//! is intentionally not mutable in place. The working state needed to insert more
//! points (the coordinates and the enclosing super-triangle) is dropped by
//! [`finalize`]. When refinement lands, the plan is to promote that working state
//! into a persistent mutable mesh type owning `Vec<Triangle>` + coordinates, with
//! an `insert` method reusing [`locate`]/[`insert_inside`]/[`insert_on_edge`]/
//! [`legalize`], and a `finalize` producing the read-only [`Triangulation`] for
//! consumers. Moving-mesh regularization (Lloyd/CVT), by contrast, is expected to
//! rebuild from scratch each step rather than mutate.

use super::predicates::{incircle, orient2d};
use crate::grid::Point2D;

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

/// Seed the working triangulation state with a large enclosing super-triangle.
///
/// # Arguments
/// * `points` - the input sites; copied into the returned coordinates unchanged.
///
/// Returns `(coords, tris)`: the input points followed by the three super-triangle
/// vertices, and a one-element triangle list for the (counter-clockwise) super-triangle.
fn initialize_super_triangle(points: &[Point2D]) -> (Vec<Point2D>, Vec<Triangle>) {
    let n = points.len();
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
    coords.push(Point2D::new(cx - m, cy - m));
    coords.push(Point2D::new(cx + m, cy - m));
    coords.push(Point2D::new(cx, cy + m));

    // super-triangle (bottom-left, bottom-right, top) is counter-clockwise
    let tris = vec![Triangle {
        vertices: [n, n + 1, n + 2],
        neighbors: [NO_NEIGHBOR, NO_NEIGHBOR, NO_NEIGHBOR],
    }];
    (coords, tris)
}

/// Drop super-triangle triangles and remap neighbour indices as hull edge.
///
/// # Arguments
/// * `tris` - the finished triangle list, still holding the super-triangle.
/// * `n` - number of input points; vertices `>= n` are super-triangle vertices.
///
/// Returns the [`Triangulation`] over the input points, with links to dropped
/// triangles turned into [`NO_NEIGHBOR`] hull edges.
fn finalize(tris: &[Triangle], n: usize) -> Triangulation {
    let mut new_index: Vec<Option<usize>> = vec![None; tris.len()];
    let mut count = 0usize;
    for (i, t) in tris.iter().enumerate() {
        if t.vertices.iter().all(|&v| v < n) {
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
	// Write triangle vertices
        triangles.push(t.vertices);
        // Neighbors of dropped vertices become a hull edges
        let mut nb = [NO_NEIGHBOR; 3];
        for k in 0..3 {
            if t.neighbors[k] != NO_NEIGHBOR {
                nb[k] = new_index[t.neighbors[k]].unwrap_or(NO_NEIGHBOR);
            }
        }
	// Write neighbours
        neighbors.push(nb);
    }
    // Build final triangulation struct
    Triangulation {
        triangles,
        neighbors,
    }
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
    let n = points.len();
    if n < 3 {
        return Triangulation {
            triangles: Vec::new(),
            neighbors: Vec::new(),
        };
    }
    // Initialize all-encompassing super triangle
    let (coords, mut tris) = initialize_super_triangle(points);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut last = 0usize;
    // Build new triangles depending on location, and check Delaunay property
    for pidx in 0..n {
        let p = coords[pidx];
        last = match locate(&tris, &coords, p, last) {
            Located::Inside(t) => insert_inside(&mut tris, t, pidx, &mut stack),
            Located::Edge(t, e) => insert_on_edge(&mut tris, t, e, pidx, &mut stack),
            Located::Vertex => continue, // duplicate site
        };
        legalize(&mut tris, &coords, &mut stack);
    }
    // Remove super triangle
    finalize(&tris, n)
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
        let mut rng = Lcg(0xABCDEF);
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
        // quads); exercises the on-edge 2→4 split path exactly.
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
        // The last point lies exactly on the interior edge (0,0)-(2,0), so it is
        // located as `Edge` and inserted via the on-edge (2→4) split rather than
        // the interior 1→3 fan.
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
}
