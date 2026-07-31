// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Kernel-independent FMM: the `KernelIndependent` expansion scheme and the [`Kifmm`] backend
//! on the generic `TreeFmm` engine, $O(N)$.
//!
//! Resource: Ying, Biros & Zorin, A kernel-independent adaptive fast multipole
//! algorithm in two and three dimensions, J. Comput. Phys. 196(2),
//! [DOI:10.1016/j.jcp.2003.11.021](https://doi.org/10.1016/j.jcp.2003.11.021)
//!
//! Each box carries equivalent monopole densities on box-scaled proxy circles;
//! the operators solve the check-from-equivalent map (a precomputed
//! `pseudo_inverse`) rather than deriving anything analytically. Evaluation
//! reuses [`Jet::add_monopole`], as the densities are just monopoles.

use crate::fastsum::{ExpansionOps, FieldSolver, Jet, Monopole, SolverConfig, TreeFmm};
use crate::grid::Point2D;
use crate::tree::TreeBox;
use crate::utils::dense::pseudo_inverse;
use ndarray::{Array1, Array2};
use std::f64::consts::TAU;

/// Inner proxy-circle radius, in units of a box's half side-length. Encloses the
/// box's sources (circumscribed radius $\sqrt2$) with margin.
const R_IN: f64 = 1.5;
/// Outer (well-separated) proxy-circle radius, in units of the half side-length.
const R_OUT: f64 = 3.0;
/// Check points per equivalent point: an oversampled, overdetermined fit.
const OVERSAMPLE: usize = 2;
/// Tikhonov regularization for the check-from-equivalent pseudo-inverses.
const LAMBDA: f64 = 1e-12;

/// Kernel-independent multipole scheme for the 2D log kernel: equivalent proxy
/// densities on box-scaled circles with pseudo-inverse translation operators.
/// The two unit circles are shared across levels; the `n × m` pinvs are
/// precomputed per level. Outgoing and incoming expansions are both just the
/// proxy monopoles.
pub(crate) struct KernelIndependent {
    /// Root box half side-length, for mapping a box half to its level index.
    root_half: f64,
    /// Unit equivalent circle: `n` points (equivalent surfaces).
    equiv: Vec<Point2D>,
    /// Unit check circle: `m = OVERSAMPLE · n` points (check surfaces).
    check: Vec<Point2D>,
    /// Per level, the up-check → up-equivalent inverse (S2M, M2M).
    up_pinv: Vec<Array2<f64>>,
    /// Per level, the down-check → down-equivalent inverse (M2L, L2L, S2L).
    down_pinv: Vec<Array2<f64>>,
}

impl KernelIndependent {
    /// Level index for a box of the given half side-length.
    ///
    /// # Arguments
    /// * `half` - the box half side-length.
    fn level_index(&self, half: f64) -> usize {
        let l = (self.root_half / half).log2().round() as usize;
        l.min(self.up_pinv.len() - 1)
    }
}

/// Unit-circle offsets (`n` points) scaled by `radius`.
///
/// # Arguments
/// * `n` - number of proxy points.
/// * `radius` - circle radius.
fn circle(n: usize, radius: f64) -> Vec<Point2D> {
    (0..n)
        .map(|i| {
            let a = TAU * i as f64 / n as f64;
            Point2D::new(radius * a.cos(), radius * a.sin())
        })
        .collect()
}

/// A unit circle scaled by `radius` and centred on `center`.
///
/// # Arguments
/// * `unit` - unit-circle offsets.
/// * `radius` - circle radius.
/// * `center` - box centre (the origin gives a pure offset).
fn place(unit: &[Point2D], radius: f64, center: Point2D) -> Vec<Point2D> {
    unit.iter().map(|o| center + *o * radius).collect()
}

/// Bare log-kernel matrix `K[i][j] = ln|check_i − equiv_j|` (concentric surfaces,
/// so the shared centre cancels and only the offsets matter).
///
/// # Arguments
/// * `check` - check-surface offsets (rows).
/// * `equiv` - equivalent-surface offsets (columns).
fn kernel_matrix(check: &[Point2D], equiv: &[Point2D]) -> Array2<f64> {
    let mut k = Array2::zeros((check.len(), equiv.len()));
    for (i, c) in check.iter().enumerate() {
        for (j, e) in equiv.iter().enumerate() {
            let (dx, dy) = (c.x - e.x, c.y - e.y);
            k[[i, j]] = 0.5 * (dx * dx + dy * dy).ln();
        }
    }
    k
}

/// Bare check potential `φ_k = Σ_i mᵢ ln|target_k − yᵢ|` of `srcs` at `targets`
/// (no `1/π`; it cancels against the pseudo-inverse's kernel).
///
/// # Arguments
/// * `targets` - check-surface points (absolute coordinates).
/// * `srcs` - the source monopoles.
fn check_potential(targets: &[Point2D], srcs: &[Monopole]) -> Array1<f64> {
    Array1::from_iter(targets.iter().map(|t| {
        srcs.iter().fold(0.0, |acc, m| {
            let (dx, dy) = (t.x - m.position.x, t.y - m.position.y);
            let r2 = dx * dx + dy * dy;
            if r2 > 0.0 {
                acc + 0.5 * r2.ln() * m.strength
            } else {
                acc
            }
        })
    }))
}

/// Proxy monopoles at `positions` carrying densities `w`.
///
/// # Arguments
/// * `positions` - proxy-surface points.
/// * `w` - densities, one per position.
fn density(positions: &[Point2D], w: &Array1<f64>) -> Vec<Monopole> {
    positions
        .iter()
        .zip(w.iter())
        .map(|(&p, &wi)| Monopole::new(p, wi))
        .collect()
}

/// Evaluate proxy monopoles' jet at `theta` (the `1/π` kernel via `add_monopole`).
///
/// # Arguments
/// * `monos` - the proxy monopoles.
/// * `theta` - query point.
fn eval(monos: &[Monopole], theta: Point2D) -> Jet {
    let mut f = Jet::default();
    for m in monos {
        f.add_monopole(theta, m.position, m.strength, 0.0);
    }
    f
}

impl ExpansionOps for KernelIndependent {
    type Outgoing = Vec<Monopole>;
    type Incoming = Vec<Monopole>;

    fn build(cfg: SolverConfig, root_half: f64, max_depth: usize) -> Self {
        let n = (4 * cfg.order).max(16);
        let equiv = circle(n, 1.0);
        let check = circle(OVERSAMPLE * n, 1.0);
        let o = Point2D::origin();
        let mut up_pinv = Vec::with_capacity(max_depth + 1);
        let mut down_pinv = Vec::with_capacity(max_depth + 1);
        for l in 0..=max_depth {
            let h = root_half * 0.5f64.powi(l as i32);
            // Up: check at R_OUT, equivalent at R_IN; down swaps the radii.
            let up = kernel_matrix(&place(&check, R_OUT * h, o), &place(&equiv, R_IN * h, o));
            let down = kernel_matrix(&place(&check, R_IN * h, o), &place(&equiv, R_OUT * h, o));
            up_pinv.push(pseudo_inverse(&up, LAMBDA));
            down_pinv.push(pseudo_inverse(&down, LAMBDA));
        }
        KernelIndependent {
            root_half,
            equiv,
            check,
            up_pinv,
            down_pinv,
        }
    }

    fn s2m(&self, b: TreeBox, sources: &[Monopole]) -> Vec<Monopole> {
        let l = self.level_index(b.half);
        let check = place(&self.check, R_OUT * b.half, b.center);
        let w = self.up_pinv[l].dot(&check_potential(&check, sources));
        density(&place(&self.equiv, R_IN * b.half, b.center), &w)
    }

    fn m2m(&self, b: TreeBox, child: &Vec<Monopole>, acc: &mut Option<Vec<Monopole>>) {
        let l = self.level_index(b.half);
        let check = place(&self.check, R_OUT * b.half, b.center);
        let w = self.up_pinv[l].dot(&check_potential(&check, child));
        let monos = density(&place(&self.equiv, R_IN * b.half, b.center), &w);
        match acc {
            None => *acc = Some(monos),
            Some(a) => {
                for (m, c) in a.iter_mut().zip(monos) {
                    m.strength += c.strength;
                }
            }
        }
    }

    fn zero_incoming(&self, b: TreeBox) -> Vec<Monopole> {
        place(&self.equiv, R_OUT * b.half, b.center)
            .into_iter()
            .map(|p| Monopole::new(p, 0.0))
            .collect()
    }

    fn l2l(&self, b: TreeBox, parent: &Vec<Monopole>, into: &mut Vec<Monopole>) {
        let l = self.level_index(b.half);
        let check = place(&self.check, R_IN * b.half, b.center);
        let w = self.down_pinv[l].dot(&check_potential(&check, parent));
        add_weights(into, &w);
    }

    fn m2l(&self, b: TreeBox, _src: TreeBox, out: &Vec<Monopole>, into: &mut Vec<Monopole>) {
        let l = self.level_index(b.half);
        let check = place(&self.check, R_IN * b.half, b.center);
        let w = self.down_pinv[l].dot(&check_potential(&check, out));
        add_weights(into, &w);
    }

    fn s2l(&self, b: TreeBox, sources: &[Monopole], into: &mut Vec<Monopole>) {
        let l = self.level_index(b.half);
        let check = place(&self.check, R_IN * b.half, b.center);
        let w = self.down_pinv[l].dot(&check_potential(&check, sources));
        add_weights(into, &w);
    }

    fn l2t(&self, inc: &Vec<Monopole>, theta: Point2D) -> Jet {
        eval(inc, theta)
    }

    fn m2t(&self, out: &Vec<Monopole>, theta: Point2D) -> Jet {
        eval(out, theta)
    }
}

/// Add densities `w` to an incoming expansion's fixed proxy positions.
///
/// # Arguments
/// * `into` - the incoming proxy monopoles, updated in place.
/// * `w` - densities to add, one per proxy point.
fn add_weights(into: &mut [Monopole], w: &Array1<f64>) {
    for (m, &wi) in into.iter_mut().zip(w.iter()) {
        m.strength += wi;
    }
}

/// Kernel-independent Fast Multipole Method backend (adaptive quadtree).
pub struct Kifmm(TreeFmm<KernelIndependent>);

impl FieldSolver for Kifmm {
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        Kifmm(TreeFmm::build(sources, cfg))
    }

    fn jet(&self, theta: Point2D) -> Jet {
        self.0.jet(theta)
    }

    fn name(&self) -> &'static str {
        "kifmm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::DirectSum;
    use crate::utils::rng::Lcg;

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
    fn matches_direct_within_tolerance() {
        // Full three-pass kiFMM against the exact sum at interior targets: the
        // equivalent-density local expansion plus the near field.
        let bodies = random_bodies(3000, 24680);
        let cfg = SolverConfig::default().with_order(12);
        let ki = Kifmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let mut rng = Lcg::new(13);
        for _ in 0..40 {
            let theta = Point2D::new(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9));
            let a = ki.gradient(theta);
            let b = direct.gradient(theta);
            let rel = a.distance(&b) / b.norm().max(1e-6);
            assert!(rel < 1e-4, "rel err {rel} too large");
        }
    }

    #[test]
    fn far_target_uses_root_expansion() {
        let bodies = random_bodies(200, 7);
        let cfg = SolverConfig::default().with_order(12);
        let ki = Kifmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let theta = Point2D::new(20.0, -15.0);
        let a = ki.jet(theta);
        let b = direct.jet(theta);
        let rel = a.gradient.distance(&b.gradient) / b.gradient.norm().max(1e-6);
        assert!(rel < 1e-5, "rel err {rel} too large");
    }

    #[test]
    fn empty_and_single_body() {
        let ki = Kifmm::build(&[], SolverConfig::default());
        assert_eq!(ki.gradient(Point2D::new(1.0, 1.0)), Point2D::origin());

        let one = [Monopole::new(Point2D::new(0.1, 0.1), 0.5)];
        let ki = Kifmm::build(&one, SolverConfig::default());
        let direct = DirectSum::build(&one, SolverConfig::default());
        let theta = Point2D::new(3.0, 2.0);
        let rel =
            ki.gradient(theta).distance(&direct.gradient(theta)) / direct.gradient(theta).norm();
        assert!(rel < 1e-6, "rel err {rel} too large");
    }

    #[test]
    fn hessian_matches_direct() {
        // The Hessian (κ/shear) follows the same equivalent densities.
        let bodies = random_bodies(3000, 555);
        let cfg = SolverConfig::default().with_order(12);
        let ki = Kifmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let theta = Point2D::new(0.4, -0.35);
        let a = ki.jet(theta).hessian;
        let b = direct.jet(theta).hessian;
        let scale = b
            .iter()
            .flatten()
            .fold(0.0_f64, |m, &x| m.max(x.abs()))
            .max(1e-6);
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (a[r][c] - b[r][c]).abs() < 1e-3 * scale,
                    "H[{r}][{c}] mismatch"
                );
            }
        }
    }

    #[test]
    fn query_at_a_source_is_finite() {
        let bodies = random_bodies(500, 111);
        let ki = Kifmm::build(&bodies, SolverConfig::default().with_order(8));
        let j = ki.jet(bodies[0].position);
        assert!(j.gradient.x.is_finite() && j.gradient.y.is_finite() && j.scalar.is_finite());
    }

    #[test]
    fn clustered_sources_match_direct() {
        // Two tight clusters exercise mixed-depth leaves, the X-list (S2L) and
        // coarse-neighbour near lists.
        let mut rng = Lcg::new(2024);
        let mut bodies = Vec::new();
        for _ in 0..800 {
            let p = Point2D::new(
                -0.6 + 0.05 * rng.range(-1.0, 1.0),
                0.5 + 0.05 * rng.range(-1.0, 1.0),
            );
            bodies.push(Monopole::new(p, rng.range(0.0, 0.5)));
        }
        for _ in 0..800 {
            let p = Point2D::new(
                0.7 + 0.05 * rng.range(-1.0, 1.0),
                -0.4 + 0.05 * rng.range(-1.0, 1.0),
            );
            bodies.push(Monopole::new(p, rng.range(0.0, 0.5)));
        }
        for _ in 0..40 {
            bodies.push(Monopole::new(
                Point2D::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)),
                rng.range(0.0, 0.3),
            ));
        }
        let cfg = SolverConfig::default().with_order(12);
        let ki = Kifmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        for _ in 0..30 {
            let theta = Point2D::new(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9));
            let a = ki.gradient(theta);
            let b = direct.gradient(theta);
            let rel = a.distance(&b) / b.norm().max(1e-6);
            assert!(rel < 1e-4, "rel err {rel} too large");
        }
    }
}
