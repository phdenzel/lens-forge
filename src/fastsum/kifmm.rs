// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Kernel-independent FMM: the `KiScheme` expansion and the [`Kifmm`] backend
//! on the generic `TreeFmm` driver, $O(N)$.
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

/// Equivalent monopole densities on a box-scaled proxy circle; carries the
/// outgoing (inner circle) and incoming (outer circle) expansions alike.
#[derive(Clone)]
pub(crate) struct EquivDensity {
    /// Proxy monopoles: fixed circle positions with the current densities.
    monos: Vec<Monopole>,
}

/// Precomputed proxy geometry and operator inverses for one tree level. Each
/// pass has an `n`-point equivalent surface and an oversampled `m`-point check
/// surface; the pinvs are `n × m`.
pub(crate) struct LevelOps {
    /// Up-equivalent offsets: `n` points at `R_IN * half` (outgoing positions).
    up_equiv: Vec<Point2D>,
    /// Up-check offsets: `m` points at `R_OUT * half` (S2M, M2M).
    up_check: Vec<Point2D>,
    /// Down-equivalent offsets: `n` points at `R_OUT * half` (incoming positions).
    down_equiv: Vec<Point2D>,
    /// Down-check offsets: `m` points at `R_IN * half` (M2L, L2L, S2L).
    down_check: Vec<Point2D>,
    /// Up pass: up-check → up-equivalent map inverse (S2M, M2M).
    up_pinv: Array2<f64>,
    /// Down pass: down-check → down-equivalent map inverse (M2L, L2L, S2L).
    down_pinv: Array2<f64>,
}

/// Kernel-independent multipole scheme for the 2D log kernel: equivalent proxy
/// densities with pseudo-inverse translation operators, precomputed per level.
pub(crate) struct KiScheme {
    /// Root box half side-length, for mapping a box half to its level index.
    root_half: f64,
    /// Operator data indexed by tree level.
    level: Vec<LevelOps>,
}

impl KiScheme {
    /// Operator data for a box of the given half side-length.
    ///
    /// # Arguments
    /// * `half` - the box half side-length.
    fn level_for(&self, half: f64) -> &LevelOps {
        let l = (self.root_half / half).log2().round() as usize;
        &self.level[l.min(self.level.len() - 1)]
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

/// Proxy monopoles at `center + offset` carrying densities `w`.
///
/// # Arguments
/// * `center` - the box centre.
/// * `offsets` - circle offsets from the centre.
/// * `w` - densities, one per offset.
fn density(center: Point2D, offsets: &[Point2D], w: &Array1<f64>) -> Vec<Monopole> {
    offsets
        .iter()
        .zip(w.iter())
        .map(|(o, &wi)| Monopole::new(center + *o, wi))
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

impl ExpansionOps for KiScheme {
    type Outgoing = EquivDensity;
    type Incoming = EquivDensity;

    fn build(cfg: SolverConfig, root_half: f64, max_depth: usize) -> Self {
        let n = (4 * cfg.order).max(16);
        let m = OVERSAMPLE * n;
        let equiv_unit = circle(n, 1.0);
        let check_unit = circle(m, 1.0);
        let level = (0..=max_depth)
            .map(|l| {
                let h = root_half * 0.5f64.powi(l as i32);
                let up_equiv: Vec<Point2D> = equiv_unit.iter().map(|o| *o * (R_IN * h)).collect();
                let up_check: Vec<Point2D> = check_unit.iter().map(|o| *o * (R_OUT * h)).collect();
                let down_equiv: Vec<Point2D> = equiv_unit.iter().map(|o| *o * (R_OUT * h)).collect();
                let down_check: Vec<Point2D> = check_unit.iter().map(|o| *o * (R_IN * h)).collect();
                let up_pinv = pseudo_inverse(&kernel_matrix(&up_check, &up_equiv), LAMBDA);
                let down_pinv = pseudo_inverse(&kernel_matrix(&down_check, &down_equiv), LAMBDA);
                LevelOps { up_equiv, up_check, down_equiv, down_check, up_pinv, down_pinv }
            })
            .collect();
        KiScheme { root_half, level }
    }

    fn s2m(&self, b: TreeBox, sources: &[Monopole]) -> EquivDensity {
        let lo = self.level_for(b.half);
        let check: Vec<Point2D> = lo.up_check.iter().map(|o| b.center + *o).collect();
        let w = lo.up_pinv.dot(&check_potential(&check, sources));
        EquivDensity { monos: density(b.center, &lo.up_equiv, &w) }
    }

    fn m2m(&self, b: TreeBox, child: &EquivDensity, acc: &mut Option<EquivDensity>) {
        let lo = self.level_for(b.half);
        let check: Vec<Point2D> = lo.up_check.iter().map(|o| b.center + *o).collect();
        let w = lo.up_pinv.dot(&check_potential(&check, &child.monos));
        let monos = density(b.center, &lo.up_equiv, &w);
        match acc {
            None => *acc = Some(EquivDensity { monos }),
            Some(a) => {
                for (m, c) in a.monos.iter_mut().zip(monos) {
                    m.strength += c.strength;
                }
            }
        }
    }

    fn zero_incoming(&self, b: TreeBox) -> EquivDensity {
        let lo = self.level_for(b.half);
        let monos = lo.down_equiv.iter().map(|o| Monopole::new(b.center + *o, 0.0)).collect();
        EquivDensity { monos }
    }

    fn l2l(&self, b: TreeBox, parent: &EquivDensity, into: &mut EquivDensity) {
        let lo = self.level_for(b.half);
        let check: Vec<Point2D> = lo.down_check.iter().map(|o| b.center + *o).collect();
        let w = lo.down_pinv.dot(&check_potential(&check, &parent.monos));
        add_weights(into, &w);
    }

    fn m2l(&self, b: TreeBox, _src: TreeBox, out: &EquivDensity, into: &mut EquivDensity) {
        let lo = self.level_for(b.half);
        let check: Vec<Point2D> = lo.down_check.iter().map(|o| b.center + *o).collect();
        let w = lo.down_pinv.dot(&check_potential(&check, &out.monos));
        add_weights(into, &w);
    }

    fn s2l(&self, b: TreeBox, sources: &[Monopole], into: &mut EquivDensity) {
        let lo = self.level_for(b.half);
        let check: Vec<Point2D> = lo.down_check.iter().map(|o| b.center + *o).collect();
        let w = lo.down_pinv.dot(&check_potential(&check, sources));
        add_weights(into, &w);
    }

    fn l2t(&self, inc: &EquivDensity, theta: Point2D) -> Jet {
        eval(&inc.monos, theta)
    }

    fn m2t(&self, out: &EquivDensity, theta: Point2D) -> Jet {
        eval(&out.monos, theta)
    }
}

/// Add densities `w` to an incoming expansion's fixed proxy positions.
///
/// # Arguments
/// * `into` - the incoming expansion, updated in place.
/// * `w` - densities to add, one per proxy point.
fn add_weights(into: &mut EquivDensity, w: &Array1<f64>) {
    for (m, &wi) in into.monos.iter_mut().zip(w.iter()) {
        m.strength += wi;
    }
}

/// Kernel-independent Fast Multipole Method backend (adaptive quadtree).
pub struct Kifmm(TreeFmm<KiScheme>);

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
        let rel = ki.gradient(theta).distance(&direct.gradient(theta)) / direct.gradient(theta).norm();
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
        let scale = b.iter().flatten().fold(0.0_f64, |m, &x| m.max(x.abs())).max(1e-6);
        for r in 0..2 {
            for c in 0..2 {
                assert!((a[r][c] - b[r][c]).abs() < 1e-3 * scale, "H[{r}][{c}] mismatch");
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
            let p = Point2D::new(-0.6 + 0.05 * rng.range(-1.0, 1.0), 0.5 + 0.05 * rng.range(-1.0, 1.0));
            bodies.push(Monopole::new(p, rng.range(0.0, 0.5)));
        }
        for _ in 0..800 {
            let p = Point2D::new(0.7 + 0.05 * rng.range(-1.0, 1.0), -0.4 + 0.05 * rng.range(-1.0, 1.0));
            bodies.push(Monopole::new(p, rng.range(0.0, 0.5)));
        }
        for _ in 0..40 {
            bodies.push(Monopole::new(Point2D::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)), rng.range(0.0, 0.3)));
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
