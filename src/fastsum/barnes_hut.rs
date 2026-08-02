// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Barnes–Hut quadtree with monopole (centre-of-mass) acceptance, $O(N \log N)$.
//!
//! Resource: Barnes & Hut, A hierarchical $O(N \log N)$ force-calculation
//! algorithm, Nature 324, [DOI:10.1038/324446a0](https://doi.org/10.1038/324446a0)
//!
//! A square quadtree is built over the source elements; each node caches its
//! aggregate monopole (total mass and centre of mass). At a query point, a node
//! whose region satisfies the opening criterion
//! $\mathrm{size} / \mathrm{distance} < \theta_\mathrm{bh}$ is summarised by its
//! monopole; otherwise its children are visited. Leaves sum their elements
//! directly.

use crate::fastsum::{FieldSolver, Jet, Monopole, SolverConfig};
use crate::grid::Point2D;
use crate::tree::QuadTree;

/// Barnes–Hut backend.
#[derive(Debug)]
pub struct BarnesHut {
    /// Bodies in the simulation
    sources: Vec<Monopole>,
    /// Spatial partitioning
    tree: QuadTree,
    /// Center of mass per node (indexed by node id)
    com: Vec<Point2D>,
    /// Total mass per node (indexed by node id)
    mass: Vec<f64>,
    /// opening angle (squared)
    theta2: f64,
    /// softening (squared)
    eps2: f64,
}

impl BarnesHut {
    /// Aggregate each node's monopole (total mass and centre of mass).
    ///
    /// Nodes are pushed parent-before-children, so a reverse pass visits every
    /// child before its parent: leaves sum their bodies, internal nodes sum
    /// their children.
    fn aggregate(tree: &QuadTree, sources: &[Monopole]) -> (Vec<f64>, Vec<Point2D>) {
        let n = tree.nodes.len();
        let mut mass = vec![0.0; n];
        let mut com = vec![Point2D::origin(); n];
        for i in (0..n).rev() {
            let node = &tree.nodes[i];
            let (mut m, mut c) = (0.0, Point2D::origin());
            if node.is_leaf() {
                for &b in &node.indices {
                    let e = sources[b];
                    m += e.strength;
                    c += e.position * e.strength;
                }
            } else {
                for &ch in node.children.iter().flatten() {
                    m += mass[ch];
                    c += com[ch] * mass[ch];
                }
            }
            mass[i] = m;
            com[i] = if m != 0.0 { c / m } else { node.center };
        }
        (mass, com)
    }

    fn accumulate(&self, node_idx: usize, theta: Point2D, field: &mut Jet) {
        let node = &self.tree.nodes[node_idx];
        if node.is_leaf() {
            for &b in &node.indices {
                let e = self.sources[b];
                field.add_monopole(theta, e.position, e.strength, self.eps2);
            }
            return;
        }
        let com = self.com[node_idx];
        let d2 = theta.distance_squared(&com);
        let size = 2.0 * node.half;
        if d2 > 0.0 && size * size < self.theta2 * d2 {
            // Accept the node as its order-0 multipole (aggregate monopole).
            field.add_monopole(theta, com, self.mass[node_idx], self.eps2);
        } else {
            for &child in node.children.iter().flatten() {
                self.accumulate(child, theta, field);
            }
        }
    }
}

impl FieldSolver for BarnesHut {
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        let positions: Vec<Point2D> = sources.iter().map(|b| b.position).collect();
        let tree = QuadTree::build(&positions, cfg.leaf_size.max(1));
        let (mass, com) = Self::aggregate(&tree, sources);
        Self {
            sources: sources.to_vec(),
            tree,
            mass,
            com,
            theta2: cfg.theta_bh * cfg.theta_bh,
            eps2: cfg.eps2,
        }
    }

    fn jet(&self, theta: Point2D) -> Jet {
        let mut f = Jet::default();
        self.accumulate(self.tree.root, theta, &mut f);
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
        let a = bh.gradient(Point2D::new(2.0, 0.0));
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
            let theta =
                Point2D::from_polar(rng.range(1.5, 4.0), rng.range(0.0, std::f64::consts::TAU));
            let approx = bh.gradient(theta);
            let exact = direct.gradient(theta);
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
        assert!(bh.gradient(theta).distance(&direct.gradient(theta)) < 1e-9);
    }

    #[test]
    fn hessian_is_trace_free_away_from_masses() {
        let bodies = random_bodies(300, 42);
        let bh = BarnesHut::build(&bodies, SolverConfig::default().with_theta(0.0));
        let h = bh.jet(Point2D::new(5.0, 5.0)).hessian;
        assert!((h[0][0] + h[1][1]).abs() < 1e-9);
        assert!((h[0][1] - h[1][0]).abs() < 1e-15);
    }
}
