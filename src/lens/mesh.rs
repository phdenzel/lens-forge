// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens on an adaptive/Voronoi lens plane.
//!
//! Its irregular sampling rules out the spectral (FFT) solve the uniform
//! [`PixelGrid`](crate::grid::PixelGrid) lens uses for $\nabla^2\psi = 2\kappa$.
//! Instead each cell is a mass element $m_i = \kappa_i \cdot \mathrm{area}_i$ at its
//! centroid, and the field is summed from the 2D Poisson Green's function by one of
//! the interchangeable [`fastsum`](crate::fastsum) backends (Barnes–Hut by default;
//! also the exact direct sum, and — planned — FMM / KIFMM / $\mathcal{H}$-matrix).

use crate::fastsum::{FieldSolver, SolverConfig, SolverKind, build_solver};
use crate::grid::{Point2D, ScalarField2D};
use crate::lens::model::{LensJet, LensModel, MassElement};
use crate::{LensForgeError, Result};

/// Lens on a Voronoi/adaptive plane.
///
/// Each field cell becomes a mass element `m = κ · area` at its centroid; the
/// potential and its derivatives are delegated to a pluggable [`FieldSolver`]
/// backend. Convergence is read back from the nearest cell (the Voronoi definition).
pub struct MeshLens {
    sources: Vec<MassElement>,
    kappa: Vec<f64>,
    kind: SolverKind,
    cfg: SolverConfig,
    solver: Box<dyn FieldSolver + Send + Sync>,
}

impl MeshLens {
    /// Build a lens from a scalar κ field using the direct summation backend
    /// (exact but does not scale).
    ///
    /// Chain [`with_solver`](Self::with_solver) and [`configure`](Self::configure)
    /// to pick a faster-to-query backend (e.g. Barnes–Hut) or a configuration.
    pub fn new(field: &impl ScalarField2D) -> Result<Self> {
        Self::from_field(field, SolverKind::Direct, SolverConfig::default())
    }

    /// Switch the fast-summation backend, rebuilding the solver (chainable).
    ///
    /// # Arguments
    /// * `kind` - fast-summation backend to switch to.
    pub fn with_solver(mut self, kind: SolverKind) -> Self {
        self.kind = kind;
        self.rebuild();
        self
    }

    /// Replace the backend configuration, rebuilding the solver (chainable).
    ///
    /// # Arguments
    /// * `cfg` - backend configuration to apply.
    pub fn configure(mut self, cfg: SolverConfig) -> Self {
        self.cfg = cfg;
        self.rebuild();
        self
    }

    /// Rebuild the boxed backend from the stored sources, kind and config.
    fn rebuild(&mut self) {
        self.solver = build_solver(self.kind, &self.sources, self.cfg);
    }

    /// Build a lens from a scalar κ field, choosing the fast-summation backend.
    ///
    /// Reads each cell's centroid, value (κ) and area straight off the
    /// [`ScalarField2D`] — a [`MeshGrid`](crate::grid::MeshGrid) κ field drops in
    /// with no intermediate representation.
    ///
    /// # Arguments
    /// * `field` - convergence κ sampled per cell (centroid, value, area).
    /// * `kind` - fast-summation backend.
    /// * `cfg` - backend configuration.
    ///
    /// # Errors
    /// Returns [`LensForgeError::InvalidParameter`] if the field is empty or any
    /// cell has a non-finite centroid/κ or a non-positive, non-finite area.
    pub fn from_field(
        field: &impl ScalarField2D,
        kind: SolverKind,
        cfg: SolverConfig,
    ) -> Result<Self> {
        let n = field.len();
        if n == 0 {
            return Err(LensForgeError::invalid_parameter(
                "MeshLens requires a non-empty field",
            ));
        }
        let mut sources = Vec::with_capacity(n);
        let mut kappa = Vec::with_capacity(n);
        for i in 0..n {
            let center = field.center(i);
            let k = field.value_at(i);
            let area = field.cell_area(i);
            if !center.is_finite() || !k.is_finite() {
                return Err(LensForgeError::invalid_parameter(format!(
                    "cell {i} has a non-finite centre or κ"
                )));
            }
            if !(area.is_finite() && area > 0.0) {
                return Err(LensForgeError::invalid_parameter(format!(
                    "cell {i} has a non-positive or non-finite area"
                )));
            }
            kappa.push(k);
            sources.push(MassElement::new(center, k * area));
        }
        let solver = build_solver(kind, &sources, cfg);
        Ok(Self {
            sources,
            kappa,
            kind,
            cfg,
            solver,
        })
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether the lens has no cells (never true for a constructed lens).
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl LensModel for MeshLens {
    fn jet(&self, theta: Point2D) -> LensJet {
        self.solver.jet(theta).into()
    }

    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<LensJet> {
        self.solver
            .jet_batch(thetas)
            .into_iter()
            .map(Into::into)
            .collect()
    }

    /// Deflection only — cheaper than a full [`jet`](Self::jet) (skips the Hessian).
    fn deflection(&self, theta: Point2D) -> Point2D {
        self.solver.gradient(theta)
    }

    /// Convergence at `theta` — the κ of the containing Voronoi cell, i.e. the
    /// nearest site (that is exactly the Voronoi definition).
    fn convergence(&self, theta: Point2D) -> f64 {
        let mut best = 0;
        let mut best_d2 = f64::INFINITY;
        for (i, s) in self.sources.iter().enumerate() {
            let d2 = theta.distance_squared(&s.position);
            if d2 < best_d2 {
                best_d2 = d2;
                best = i;
            }
        }
        self.kappa[best]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lcg;
    use crate::grid::{BBox, FieldArray1D, MeshGrid};
    use std::f64::consts::FRAC_1_PI;

    fn bbox() -> BBox {
        BBox::from_lims(-3.0, 3.0, -3.0, 3.0)
    }

    /// A κ field on a Voronoi mesh with explicit per-site values.
    fn mesh(sites: &[Point2D], kappa: &[f64]) -> MeshGrid {
        MeshGrid::new(sites, bbox(), FieldArray1D::from_vec(kappa.to_vec())).unwrap()
    }

    /// `n` scattered sites in the box, radially concentrated toward the centre.
    fn scattered(n: usize) -> Vec<Point2D> {
        let mut rng = Lcg::new(0x5EED);
        (0..n)
            .map(|_| {
                let r = 2.9 * rng.f() * rng.f();
                let t = std::f64::consts::TAU * rng.f();
                Point2D::new(r * t.cos(), r * t.sin())
            })
            .collect()
    }

    #[test]
    fn deflection_matches_direct_monopole_sum() {
        let sites = scattered(64);
        let kappa: Vec<f64> = (0..64).map(|i| 0.1 + 0.01 * i as f64).collect();
        let mg = mesh(&sites, &kappa);
        let lens = MeshLens::from_field(&mg, SolverKind::Direct, SolverConfig::default()).unwrap();
        // Reference: sum monopole deflections from the mesh's own cells (m = κ·area).
        let field: &dyn ScalarField2D = &mg;
        let theta = Point2D::new(4.0, 3.0);
        let (mut ax, mut ay) = (0.0, 0.0);
        for i in 0..field.len() {
            let c = field.center(i);
            let m = field.value_at(i) * field.cell_area(i);
            let r2 = theta.distance_squared(&c);
            ax += m * FRAC_1_PI * (theta.x - c.x) / r2;
            ay += m * FRAC_1_PI * (theta.y - c.y) / r2;
        }
        let a = lens.deflection(theta);
        assert!(
            (a.x - ax).abs() < 1e-9 && (a.y - ay).abs() < 1e-9,
            "a=({},{}) exp=({},{})",
            a.x,
            a.y,
            ax,
            ay
        );
    }

    #[test]
    fn convergence_is_nearest_site_kappa() {
        let sites = [
            Point2D::new(-1.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.5),
        ];
        let mg = mesh(&sites, &[0.2, 0.9, 0.5]);
        let lens = MeshLens::new(&mg).unwrap();
        assert_eq!(lens.convergence(Point2D::new(-0.9, 0.0)), 0.2);
        assert_eq!(lens.convergence(Point2D::new(0.9, 0.0)), 0.9);
    }

    #[test]
    fn direct_and_barnes_hut_backends_agree() {
        let sites = scattered(200);
        let kappa: Vec<f64> = (0..200).map(|i| 0.1 + 0.01 * i as f64).collect();
        let mg = mesh(&sites, &kappa);
        let bh = MeshLens::from_field(&mg, SolverKind::BarnesHut, SolverConfig::default()).unwrap();
        let direct =
            MeshLens::from_field(&mg, SolverKind::Direct, SolverConfig::default()).unwrap();
        let theta = Point2D::new(3.0, 2.0);
        let rel = bh.deflection(theta).distance(&direct.deflection(theta))
            / direct.deflection(theta).norm().max(1e-6);
        assert!(rel < 2e-2, "rel err {rel}");
    }

    #[test]
    fn chaining_selects_backend() {
        let sites = scattered(80);
        let kappa: Vec<f64> = (0..80).map(|i| 0.2 + 0.005 * i as f64).collect();
        let mg = mesh(&sites, &kappa);
        let theta = Point2D::new(3.0, 2.0);
        let direct =
            MeshLens::from_field(&mg, SolverKind::Direct, SolverConfig::default()).unwrap();
        // new() defaults to the exact direct sum: matches from_field(Direct) exactly.
        assert!(
            MeshLens::new(&mg)
                .unwrap()
                .deflection(theta)
                .distance(&direct.deflection(theta))
                < 1e-12
        );
        // A switch to Barnes–Hut then back must rebuild and again equal Direct exactly.
        let chained = MeshLens::new(&mg)
            .unwrap()
            .with_solver(SolverKind::BarnesHut)
            .with_solver(SolverKind::Direct)
            .configure(SolverConfig::default());
        assert!(
            chained
                .deflection(theta)
                .distance(&direct.deflection(theta))
                < 1e-12
        );
    }

    #[test]
    fn ray_trace_and_magnification_are_finite() {
        let sites = scattered(100);
        let mg = MeshGrid::from_func(&sites, bbox(), |_, _| 0.3).unwrap();
        let lens = MeshLens::new(&mg).unwrap();
        let theta = Point2D::new(2.5, 0.0);
        assert!(lens.ray_trace(theta).is_finite());
        assert!(lens.magnification(theta).is_finite());
    }

    /// Minimal field to exercise the defensive validation a well-formed mesh can't hit.
    struct BadField {
        kappa: Vec<f64>,
        area: Vec<f64>,
    }
    impl ScalarField2D for BadField {
        fn len(&self) -> usize {
            self.kappa.len()
        }
        fn value_at(&self, i: usize) -> f64 {
            self.kappa[i]
        }
        fn center(&self, _i: usize) -> Point2D {
            Point2D::origin()
        }
        fn cell_area(&self, i: usize) -> f64 {
            self.area[i]
        }
        fn bounds(&self) -> BBox {
            bbox()
        }
        fn sample(&self, _p: Point2D) -> f64 {
            0.0
        }
    }

    #[test]
    fn rejects_empty_and_bad_fields() {
        let empty = BadField {
            kappa: vec![],
            area: vec![],
        };
        assert!(MeshLens::new(&empty).is_err());
        let zero_area = BadField {
            kappa: vec![0.5],
            area: vec![0.0],
        };
        assert!(MeshLens::new(&zero_area).is_err());
        let neg_area = BadField {
            kappa: vec![0.5],
            area: vec![-1.0],
        };
        assert!(MeshLens::new(&neg_area).is_err());
        let nan_kappa = BadField {
            kappa: vec![f64::NAN],
            area: vec![1.0],
        };
        assert!(MeshLens::new(&nan_kappa).is_err());
    }
}
