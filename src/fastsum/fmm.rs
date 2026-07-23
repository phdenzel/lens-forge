// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Fast Multipole Method on a uniform quadtree, $O(N)$.
//!
//! The classic Greengard–Rokhlin 2D FMM for the holomorphic log-kernel. A
//! uniform quadtree of depth $L$ is laid over the sources; the field is built in
//! three passes whose expansions depend only on the sources, so they run once at
//! [`build`](FieldSolver::build) time:
//!
//! * **upward** — form each leaf's multipole ([`S2M`](Multipole::from_monopoles)),
//!   then merge children into parents ([`M2M`](Multipole::translate)).
//! * **interaction** — for every box, convert each well-separated box's multipole
//!   into a local expansion ([`M2L`](Multipole::to_local)) over the interaction
//!   list.
//! * **downward** — push each box's local expansion to its children
//!   ([`L2L`](LocalExpansion::translate)).
//!
//! A query then locates its leaf, evaluates that leaf's local expansion, and adds
//! the direct near-field over the leaf and its neighbours (the shared
//! [`Jet::add_monopole`] kernel). The far field uses the exact log-kernel, so
//! softening (`eps2`) is a near-field-only correction — consistent with the far
//! field only at `eps2 = 0`, which is harmless since far sources sit at least one
//! box away.

use crate::fastsum::{FieldSolver, Jet, LocalExpansion, Monopole, Multipole, SolverConfig};
use crate::grid::Point2D;
use num::complex::Complex;

/// Deepest uniform-tree level (caps memory for very large source counts).
const MAX_LEVEL: usize = 8;
/// Shallowest level that still has a non-trivial interaction list.
const MIN_LEVEL: usize = 2;

/// Fast Multipole Method backend.
#[derive(Debug)]
pub struct Fmm {
    /// Source bodies (for the direct near-field).
    elements: Vec<Monopole>,
    /// Squared softening added to $r^2$ in the near field.
    eps2: f64,
    /// Lower corner of the (square, padded) root box.
    lo: Point2D,
    /// Boxes per axis at the leaf level (`2^leaf_level`).
    n_leaf: usize,
    /// Box side length at the leaf level.
    cell: f64,
    /// Source indices per leaf box, row-major `iy * n_leaf + ix`.
    leaf_sources: Vec<Vec<usize>>,
    /// Precomputed local expansion per leaf box.
    leaf_locals: Vec<LocalExpansion>,
    /// Root multipole, used for query points outside the root box.
    root_multipole: Option<Multipole>,
    /// Whether the solver holds no sources.
    empty: bool,
}

/// Well-separated boxes of `(ix, iy)` at `level`: children of the parent's
/// neighbours that are not themselves neighbours of `(ix, iy)`.
///
/// # Arguments
/// * `ix` - box column index at `level`.
/// * `iy` - box row index at `level`.
/// * `level` - tree level the box lives on.
///
/// Returns the box indices `(x, y)` at the same level to reach by M2L.
fn interaction_list(ix: usize, iy: usize, level: usize) -> Vec<(usize, usize)> {
    if level < MIN_LEVEL {
        return Vec::new();
    }
    let pn = 1usize << (level - 1); // parent boxes per axis
    let (px, py) = (ix / 2, iy / 2);
    let mut list = Vec::new();
    for qy in py.saturating_sub(1)..=(py + 1).min(pn - 1) {
        for qx in px.saturating_sub(1)..=(px + 1).min(pn - 1) {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (cx, cy) = (2 * qx + dx, 2 * qy + dy);
                    let far = (cx as i64 - ix as i64).abs() > 1 || (cy as i64 - iy as i64).abs() > 1;
                    if far {
                        list.push((cx, cy));
                    }
                }
            }
        }
    }
    list
}

impl Fmm {
    /// Add a local expansion's coefficients into an accumulator in place.
    ///
    /// # Arguments
    /// * `acc` - coefficient accumulator, updated in place.
    /// * `le` - local expansion whose coefficients are added.
    fn add_coeffs(acc: &mut [Complex<f64>], le: &LocalExpansion) {
        for (a, b) in acc.iter_mut().zip(le.coeffs.iter()) {
            *a += *b;
        }
    }
}

impl FieldSolver for Fmm {
    fn build(sources: &[Monopole], cfg: SolverConfig) -> Self {
        let order = cfg.order;
        if sources.is_empty() {
            return Fmm {
                elements: Vec::new(),
                eps2: cfg.eps2,
                lo: Point2D::origin(),
                n_leaf: 1 << MIN_LEVEL,
                cell: 1.0,
                leaf_sources: Vec::new(),
                leaf_locals: Vec::new(),
                root_multipole: None,
                empty: true,
            };
        }

        // Square, slightly padded root box (padding keeps boundary sources inside).
        let positions: Vec<Point2D> = sources.iter().map(|s| s.position).collect();
        let (blo, bhi) = Point2D::bounding_box(&positions).expect("non-empty");
        let center = blo.midpoint(&bhi);
        let extent = bhi - blo;
        let mut half = 0.5 * extent.x.max(extent.y);
        if !(half > 0.0) {
            half = 1.0; // all sources coincide
        }
        half *= 1.0 + 1e-9;
        let lo = center - half;
        let root_size = 2.0 * half;

        // Tree depth: aim for ~leaf_size bodies per leaf box.
        let ratio = (sources.len() as f64 / cfg.leaf_size.max(1) as f64).max(1.0);
        let leaf_level =
            (ratio.log(4.0).round() as i64).clamp(MIN_LEVEL as i64, MAX_LEVEL as i64) as usize;
        let n_leaf = 1usize << leaf_level;
        let cell = root_size / n_leaf as f64;
        let box_center = |level: usize, ix: usize, iy: usize| {
            let s = root_size / (1usize << level) as f64;
            lo + Point2D::new(ix as f64 + 0.5, iy as f64 + 0.5) * s
        };

        // Assign sources to leaf boxes.
        let mut leaf_sources: Vec<Vec<usize>> = vec![Vec::new(); n_leaf * n_leaf];
        for (i, p) in positions.iter().enumerate() {
            let rel = (*p - lo) / cell;
            let ix = (rel.x.floor() as i64).clamp(0, n_leaf as i64 - 1) as usize;
            let iy = (rel.y.floor() as i64).clamp(0, n_leaf as i64 - 1) as usize;
            leaf_sources[iy * n_leaf + ix].push(i);
        }

        // Upward pass: S2M at leaves, M2M up to the root.
        let mut mp: Vec<Vec<Option<Multipole>>> = (0..=leaf_level)
            .map(|l| vec![None; (1usize << l) * (1usize << l)])
            .collect();
        for iy in 0..n_leaf {
            for ix in 0..n_leaf {
                let flat = iy * n_leaf + ix;
                if leaf_sources[flat].is_empty() {
                    continue;
                }
                let subset: Vec<Monopole> = leaf_sources[flat].iter().map(|&i| sources[i]).collect();
                mp[leaf_level][flat] =
                    Some(Multipole::from_monopoles(&subset, box_center(leaf_level, ix, iy), order));
            }
        }
        for level in (0..leaf_level).rev() {
            let n = 1usize << level;
            for iy in 0..n {
                for ix in 0..n {
                    let c = box_center(level, ix, iy);
                    let mut strength = 0.0;
                    let mut moments = vec![Complex::new(0.0, 0.0); order];
                    let mut any = false;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let child = &mp[level + 1][(2 * iy + dy) * (2 * n) + (2 * ix + dx)];
                            if let Some(ch) = child {
                                let t = ch.translate(c);
                                strength += t.strength;
                                for (m, tm) in moments.iter_mut().zip(t.moments.iter()) {
                                    *m += *tm;
                                }
                                any = true;
                            }
                        }
                    }
                    if any {
                        mp[level][iy * n + ix] = Some(Multipole { center: c, strength, moments });
                    }
                }
            }
        }
        let root_multipole = mp[0][0].clone();

        // Interaction + downward pass: M2L over the interaction list,
        // then L2L from the parent, level by level.
        let mut local: Vec<Vec<Vec<Complex<f64>>>> = (0..=leaf_level)
            .map(|l| vec![vec![Complex::new(0.0, 0.0); order + 1]; (1usize << l) * (1usize << l)])
            .collect();
        for level in MIN_LEVEL..=leaf_level {
            let n = 1usize << level;
            for iy in 0..n {
                for ix in 0..n {
                    let flat = iy * n + ix;
                    let c = box_center(level, ix, iy);
                    // L2L from the parent (parent locals are zero at MIN_LEVEL).
                    if level > MIN_LEVEL {
                        let pn = n / 2;
                        let (px, py) = (ix / 2, iy / 2);
                        let parent = LocalExpansion {
                            center: box_center(level - 1, px, py),
                            coeffs: local[level - 1][py * pn + px].clone(),
                        };
                        let shifted = parent.translate(c);
                        Self::add_coeffs(&mut local[level][flat], &shifted);
                    }
                    // M2L from every well-separated box.
                    for (ax, ay) in interaction_list(ix, iy, level) {
                        if let Some(source_mp) = &mp[level][ay * n + ax] {
                            let le = source_mp.to_local(c);
                            Self::add_coeffs(&mut local[level][flat], &le);
                        }
                    }
                }
            }
        }

        // Keep only the leaf-level local expansions.
        let mut leaf_locals = Vec::with_capacity(n_leaf * n_leaf);
        for iy in 0..n_leaf {
            for ix in 0..n_leaf {
                let flat = iy * n_leaf + ix;
                leaf_locals.push(LocalExpansion {
                    center: box_center(leaf_level, ix, iy),
                    coeffs: std::mem::take(&mut local[leaf_level][flat]),
                });
            }
        }

        Fmm {
            elements: sources.to_vec(),
            eps2: cfg.eps2,
            lo,
            n_leaf,
            cell,
            leaf_sources,
            leaf_locals,
            root_multipole,
            empty: false,
        }
    }

    fn jet(&self, theta: Point2D) -> Jet {
        if self.empty {
            return Jet::default();
        }
        let rel = (theta - self.lo) / self.cell;
        // Outside the root box: every source is far, use the root multipole.
        let n = self.n_leaf as f64;
        if rel.x < 0.0 || rel.y < 0.0 || rel.x >= n || rel.y >= n {
            return match &self.root_multipole {
                Some(mp) => mp.eval_jet(theta),
                None => Jet::default(),
            };
        }
        let ix = rel.x.floor() as usize;
        let iy = rel.y.floor() as usize;
        let mut f = self.leaf_locals[iy * self.n_leaf + ix].eval_jet(theta);
        // Near field: direct sum over this box and its neighbours (the U-list).
        for ny in iy.saturating_sub(1)..=(iy + 1).min(self.n_leaf - 1) {
            for nx in ix.saturating_sub(1)..=(ix + 1).min(self.n_leaf - 1) {
                for &i in &self.leaf_sources[ny * self.n_leaf + nx] {
                    let e = self.elements[i];
                    f.add_monopole(theta, e.position, e.strength, self.eps2);
                }
            }
        }
        f
    }

    fn name(&self) -> &'static str {
        "fmm"
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
        // Full three-pass FMM against the exact sum, at targets inside the source
        // region (exercises the local expansion + near-field path).
        let bodies = random_bodies(3000, 24680);
        let cfg = SolverConfig::default().with_order(12);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let mut rng = Lcg::new(13);
        for _ in 0..40 {
            // Interior targets (well inside the source domain): the local +
            // near-field path. Outside the domain is the root-multipole regime.
            let theta = Point2D::new(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9));
            let a = fmm.gradient(theta);
            let b = direct.gradient(theta);
            let rel = a.distance(&b) / b.norm().max(1e-6);
            assert!(rel < 1e-5, "rel err {rel} too large");
        }
    }

    #[test]
    fn far_target_uses_root_multipole() {
        let bodies = random_bodies(200, 7);
        let cfg = SolverConfig::default().with_order(12);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let theta = Point2D::new(20.0, -15.0);
        let a = fmm.jet(theta);
        let b = direct.jet(theta);
        assert!(a.gradient.distance(&b.gradient) < 1e-8);
        assert!((a.scalar - b.scalar).abs() < 1e-8);
    }

    #[test]
    fn empty_and_single_body() {
        let fmm = Fmm::build(&[], SolverConfig::default());
        assert_eq!(fmm.gradient(Point2D::new(1.0, 1.0)), Point2D::origin());

        let one = [Monopole::new(Point2D::new(0.1, 0.1), 0.5)];
        let fmm = Fmm::build(&one, SolverConfig::default());
        let direct = DirectSum::build(&one, SolverConfig::default());
        let theta = Point2D::new(3.0, 2.0);
        assert!(fmm.gradient(theta).distance(&direct.gradient(theta)) < 1e-9);
    }

    #[test]
    fn hessian_matches_direct() {
        // The Hessian (κ/shear in lensing) follows the same expansions; check it
        // against the exact sum at an interior target.
        let bodies = random_bodies(3000, 555);
        let cfg = SolverConfig::default().with_order(12);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let theta = Point2D::new(0.4, -0.35);
        let a = fmm.jet(theta).hessian;
        let b = direct.jet(theta).hessian;
        let scale = b.iter().flatten().fold(0.0_f64, |m, &x| m.max(x.abs())).max(1e-6);
        for r in 0..2 {
            for c in 0..2 {
                assert!((a[r][c] - b[r][c]).abs() < 1e-4 * scale, "H[{r}][{c}] mismatch");
            }
        }
    }

    #[test]
    fn accuracy_improves_with_order() {
        // The near field is exact regardless of order, so a shrinking error with
        // order confirms the far-field M2L/L2L truncation is wired correctly.
        let bodies = random_bodies(2000, 321);
        let direct = DirectSum::build(&bodies, SolverConfig::default());
        let theta = Point2D::new(0.5, 0.2);
        let exact = direct.gradient(theta);
        let err = |order: usize| {
            Fmm::build(&bodies, SolverConfig::default().with_order(order))
                .gradient(theta)
                .distance(&exact)
        };
        let (e2, e6, e10) = (err(2), err(6), err(10));
        assert!(e6 < e2 && e10 < e6, "not monotone: {e2}, {e6}, {e10}");
    }

    #[test]
    fn deep_tree_matches_direct() {
        // leaf_size = 1 forces the deepest tree, exercising the multi-level M2M
        // (upward) and L2L (downward) translation chains.
        let bodies = random_bodies(1500, 909);
        let cfg = SolverConfig::default().with_order(12).with_leaf_size(1);
        let fmm = Fmm::build(&bodies, cfg);
        let direct = DirectSum::build(&bodies, cfg);
        let mut rng = Lcg::new(4242);
        for _ in 0..20 {
            let theta = Point2D::new(rng.range(-0.9, 0.9), rng.range(-0.9, 0.9));
            let exact = direct.gradient(theta);
            let rel = fmm.gradient(theta).distance(&exact) / exact.norm().max(1e-6);
            assert!(rel < 1e-4, "rel err {rel} too large");
        }
    }

    #[test]
    fn query_at_a_source_is_finite() {
        let bodies = random_bodies(500, 111);
        // eps2 = 0: the coincident source is skipped, the result stays finite.
        let fmm = Fmm::build(&bodies, SolverConfig::default().with_order(8));
        let j = fmm.jet(bodies[0].position);
        assert!(j.gradient.x.is_finite() && j.gradient.y.is_finite() && j.scalar.is_finite());
        // eps2 > 0: the near-field softening also keeps it finite.
        let fmm = Fmm::build(&bodies, SolverConfig::default().with_softening(0.05));
        let j = fmm.jet(bodies[0].position);
        assert!(j.gradient.x.is_finite() && j.gradient.y.is_finite() && j.scalar.is_finite());
    }
}
