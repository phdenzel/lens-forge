// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Barnes–Hut quadtree with monopole (centre-of-mass) acceptance, $O(N \log N)$.
//!
//! A square quadtree is built over the source elements; each node caches its
//! aggregate monopole (total mass and centre of mass). At a query point, a node
//! whose region satisfies the opening criterion
//! $\mathrm{size} / \mathrm{distance} < \theta_\mathrm{bh}$ is summarised by its
//! monopole; otherwise its children are visited. Leaves sum their bodies
//! directly. $\theta_\mathrm{bh} = 0$ never accepts, recovering the exact sum.

use crate::grid::Point2D;
use crate::fastsum::{FieldSolver, Monopole, PotentialJet, SolverConfig};

const MAX_DEPTH: usize = 64;

#[derive(Debug)]
struct QNode {
    mass: f64,
    com: Point2D,
    half: f64,
    children: [Option<usize>; 4],
    bodies: Vec<usize>, // non-empty only for leaves
}

impl QNode {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.children.iter().all(Option::is_none)
    }
}

/// Barnes–Hut backend.
#[derive(Debug)]
pub struct BarnesHut {
    bodies: Vec<Monopole>,
    nodes: Vec<QNode>,
    root: usize,
    theta2: f64,
    eps2: f64,
    leaf_size: usize,
}

impl BarnesHut {
    /// Build the quadtree over `bodies`, returning `(nodes, root_index)`.
    fn build_tree(bodies: &[Monopole], leaf_size: usize) -> (Vec<QNode>, usize) {
        let mut nodes = Vec::new();
        if bodies.is_empty() {
            nodes.push(QNode {
                mass: 0.0,
                com: Point2D::origin(),
                half: 1.0,
                children: [None; 4],
                bodies: Vec::new(),
            });
            return (nodes, 0);
        }

        let positions: Vec<Point2D> = bodies.iter().map(|b| b.position).collect();
        let (lo, hi) = Point2D::bounding_box(&positions).expect("non-empty");
        let center = lo.midpoint(&hi);
        let mut half = 0.5 * (hi.x - lo.x).max(hi.y - lo.y);
        if !(half > 0.0) {
            half = 1.0; // all points coincide
        }
        half *= 1.0 + 1e-9;

        let all: Vec<usize> = (0..bodies.len()).collect();
        let root = Self::build_node(&mut nodes, bodies, &all, center, half, 0, leaf_size);
        (nodes, root)
    }

    fn build_node(
        nodes: &mut Vec<QNode>,
        bodies: &[Monopole],
        idx: &[usize],
        center: Point2D,
        half: f64,
        depth: usize,
        leaf_size: usize,
    ) -> usize {
        let mut m = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;
        for &b in idx {
            let e = bodies[b];
            m += e.strength;
            cx += e.strength * e.position.x;
            cy += e.strength * e.position.y;
        }
        let com = if m != 0.0 {
            Point2D::new(cx / m, cy / m)
        } else {
            center
        };

        let node_idx = nodes.len();
        nodes.push(QNode {
            mass: m,
            com,
            half,
            children: [None; 4],
            bodies: Vec::new(),
        });

        if idx.len() <= leaf_size || depth >= MAX_DEPTH {
            nodes[node_idx].bodies = idx.to_vec();
            return node_idx;
        }

        let mut quads: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for &b in idx {
            let p = bodies[b].position;
            let q = usize::from(p.x >= center.x) | (usize::from(p.y >= center.y) << 1);
            quads[q].push(b);
        }

        let hh = 0.5 * half;
        let offs = [
            Point2D::new(-hh, -hh),
            Point2D::new(hh, -hh),
            Point2D::new(-hh, hh),
            Point2D::new(hh, hh),
        ];
        let mut children = [None; 4];
        for (q, bodies_q) in quads.iter().enumerate() {
            if !bodies_q.is_empty() {
                let cc = center + offs[q];
                children[q] = Some(Self::build_node(
                    nodes, bodies, bodies_q, cc, hh, depth + 1, leaf_size,
                ));
            }
        }
        nodes[node_idx].children = children;
        node_idx
    }

    fn accumulate(&self, node_idx: usize, theta: Point2D, field: &mut PotentialJet) {
        let node = &self.nodes[node_idx];
        if node.is_leaf() {
            for &b in &node.bodies {
                let e = self.bodies[b];
                field.add_monopole(theta, e.position, e.strength, self.eps2);
            }
            return;
        }
        let dx = theta.x - node.com.x;
        let dy = theta.y - node.com.y;
        let d2 = dx * dx + dy * dy;
        let size = 2.0 * node.half;
        if d2 > 0.0 && size * size < self.theta2 * d2 {
            // Accept the node as its order-0 multipole (aggregate monopole).
            field.add_monopole(theta, node.com, node.mass, self.eps2);
        } else {
            for &child in node.children.iter().flatten() {
                self.accumulate(child, theta, field);
            }
        }
    }
}

impl FieldSolver for BarnesHut {
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        let (nodes, root) = Self::build_tree(sources, cfg.leaf_size.max(1));
        Self {
            bodies: sources.to_vec(),
            nodes,
            root,
            theta2: cfg.theta_bh * cfg.theta_bh,
            eps2: cfg.eps2,
            leaf_size: cfg.leaf_size.max(1),
        }
    }

    fn field(&self, theta: Point2D) -> PotentialJet {
        let mut f = PotentialJet::default();
        self.accumulate(self.root, theta, &mut f);
        f
    }

    fn name(&self) -> &'static str {
        "barnes-hut"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::DirectSum;
    use crate::utils::rng::Lcg;
    use std::f64::consts::FRAC_1_PI;

    fn random_bodies(n: usize, seed: u64) -> Vec<Monopole> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                Monopole::new(
                    Point2D::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)),
                    rng.range(0.0, 0.5),
                )
            })
            .collect()
    }

    #[test]
    fn single_mass_matches_analytic() {
        let m = 0.4;
        let bh = BarnesHut::build(
            &[Monopole::new(Point2D::origin(), m)],
            SolverConfig::default(),
        );
        let a = bh.deflection(Point2D::new(2.0, 0.0));
        let expected = m * FRAC_1_PI * 0.5; // θ_x/|θ|² = 2/4
        assert!((a.x - expected).abs() < 1e-12);
        assert!(a.y.abs() < 1e-12);
    }

    #[test]
    fn matches_direct_within_tolerance() {
        let bodies = random_bodies(2000, 12345);
        let cfg = SolverConfig::default().with_theta(0.5);
        let bh = BarnesHut::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let mut rng = Lcg::new(999);
        for _ in 0..50 {
            let theta = Point2D::from_polar(rng.range(1.5, 4.0), rng.range(0.0, std::f64::consts::TAU));
            let approx = bh.deflection(theta);
            let exact = direct.deflection(theta);
            let rel = approx.distance(&exact) / exact.norm().max(1e-6);
            assert!(rel < 2e-2, "rel err {rel} too large");
        }
    }

    #[test]
    fn zero_opening_angle_is_exact() {
        let bodies = random_bodies(500, 7);
        let cfg = SolverConfig::default().with_theta(0.0);
        let bh = BarnesHut::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let theta = Point2D::new(0.3, -0.7);
        assert!(bh.deflection(theta).distance(&direct.deflection(theta)) < 1e-9);
    }

    #[test]
    fn hessian_is_trace_free_away_from_masses() {
        let bodies = random_bodies(300, 42);
        let bh = BarnesHut::build(&bodies, SolverConfig::default().with_theta(0.0));
        let h = bh.field(Point2D::new(5.0, 5.0)).hessian;
        assert!((h[0][0] + h[1][1]).abs() < 1e-9);
        assert!((h[0][1] - h[1][0]).abs() < 1e-15);
    }
}
