// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Free-form lens on an adaptive/Voronoi lens plane.
//!
//! Its irregular sampling rules out the spectral (FFT) solve the uniform
//! [`PixelGrid`](crate::grid::PixelGrid) lens uses for $\nabla^2\psi = 2\kappa$.
//! Instead each cell is a mass element $m_i = \kappa_i \cdot \mathrm{area}_i$ at its
//! centroid, and the field is summed from the 2D Poisson Green's function by one of
//! the interchangeable [`fastsum`](crate::fastsum) backends (Barnes–Hut by default;
//! also the exact direct sum, and — planned — FMM / KIFMM / $\mathcal{H}$-matrix).

use crate::fastsum::{FieldSolver, Monopole, SolverConfig, SolverKind, build_solver};
use crate::grid::Point2D;
use crate::lens::model::{LensJet, LensModel};
use crate::{LensForgeError, Result};

/// A single Voronoi cell of the lens plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoronoiCell {
    /// Cell site / centroid (also defines the Voronoi region).
    pub center: Point2D,
    /// Convergence κ = Σ/Σ_cr, assumed constant over the cell.
    pub kappa: f64,
    /// Cell area (in the same angular units as the coordinates).
    pub area: f64,
}

impl VoronoiCell {
    /// Create a new cell.
    pub fn new(center: Point2D, kappa: f64, area: f64) -> Self {
        Self {
            center,
            kappa,
            area,
        }
    }

    /// Dimensionless mass element `m = κ · area`.
    #[inline]
    fn mass(&self) -> f64 {
        self.kappa * self.area
    }
}

/// Free-form lens on a Voronoi/adaptive plane.
///
/// The deflection field is delegated to a pluggable [`FieldSolver`] backend; the
/// convergence is read directly from the containing cell.
pub struct MeshLens {
    sites: Vec<Point2D>,
    kappa: Vec<f64>,
    solver: Box<dyn FieldSolver + Send + Sync>,
}

impl MeshLens {
    /// Build a lens from Voronoi cells using the default backend (Barnes–Hut).
    pub fn new(cells: &[VoronoiCell]) -> Result<Self> {
        Self::with_solver(cells, SolverKind::BarnesHut, SolverConfig::default())
    }

    /// Build a lens choosing the fast-summation backend and its configuration.
    ///
    /// # Errors
    /// Returns [`LensForgeError::InvalidParameter`] if there are no cells or any
    /// cell has a non-finite coordinate/κ or a non-positive, non-finite area.
    pub fn with_solver(cells: &[VoronoiCell], kind: SolverKind, cfg: SolverConfig) -> Result<Self> {
        if cells.is_empty() {
            return Err(LensForgeError::invalid_parameter(
                "MeshLens requires at least one cell",
            ));
        }
        for (i, c) in cells.iter().enumerate() {
            if !c.center.is_finite() || !c.kappa.is_finite() {
                return Err(LensForgeError::invalid_parameter(format!(
                    "cell {i} has a non-finite centre or κ"
                )));
            }
            if !(c.area.is_finite() && c.area > 0.0) {
                return Err(LensForgeError::invalid_parameter(format!(
                    "cell {i} has a non-positive or non-finite area"
                )));
            }
        }

        let sites: Vec<Point2D> = cells.iter().map(|c| c.center).collect();
        let kappa: Vec<f64> = cells.iter().map(|c| c.kappa).collect();
        let sources: Vec<Monopole> = cells
            .iter()
            .map(|c| Monopole::new(c.center, c.mass()))
            .collect();
        let solver = build_solver(kind, &sources, cfg);
        Ok(Self {
            sites,
            kappa,
            solver,
        })
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Whether the lens has no cells (never true for a constructed lens).
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Name of the active fast-summation backend.
    pub fn solver_name(&self) -> &'static str {
        self.solver.name()
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
        for (i, &p) in self.sites.iter().enumerate() {
            let d2 = theta.distance_squared(&p);
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
    use std::f64::consts::FRAC_1_PI;

    #[test]
    fn single_cell_matches_point_mass() {
        let m = 0.5 * 0.8;
        let lens = MeshLens::new(&[VoronoiCell::new(Point2D::origin(), 0.5, 0.8)]).unwrap();
        let a = lens.deflection(Point2D::new(2.0, 0.0));
        let expected = m * FRAC_1_PI * 0.5;
        assert!(
            (a.x - expected).abs() < 1e-12,
            "ax={} exp={}",
            a.x,
            expected
        );
        assert!(a.y.abs() < 1e-12);
    }

    #[test]
    fn convergence_is_nearest_site_kappa() {
        let cells = vec![
            VoronoiCell::new(Point2D::new(-1.0, 0.0), 0.2, 1.0),
            VoronoiCell::new(Point2D::new(1.0, 0.0), 0.9, 1.0),
        ];
        let lens = MeshLens::new(&cells).unwrap();
        assert_eq!(lens.convergence(Point2D::new(-0.9, 0.1)), 0.2);
        assert_eq!(lens.convergence(Point2D::new(0.9, -0.3)), 0.9);
    }

    #[test]
    fn direct_and_barnes_hut_backends_agree() {
        let cells: Vec<VoronoiCell> = (0..200)
            .map(|i| {
                let t = i as f64 * 0.3;
                VoronoiCell::new(
                    Point2D::new(t.cos(), (2.0 * t).sin()),
                    0.1 + 0.01 * i as f64,
                    0.2,
                )
            })
            .collect();
        let bh = MeshLens::with_solver(&cells, SolverKind::BarnesHut, SolverConfig::default())
            .unwrap();
        let direct =
            MeshLens::with_solver(&cells, SolverKind::Direct, SolverConfig::default()).unwrap();
        assert_eq!(bh.solver_name(), "barnes-hut");
        assert_eq!(direct.solver_name(), "direct");
        let theta = Point2D::new(3.0, 2.0);
        let rel = bh.deflection(theta).distance(&direct.deflection(theta))
            / direct.deflection(theta).norm().max(1e-6);
        assert!(rel < 2e-2, "rel err {rel}");
    }

    #[test]
    fn ray_trace_and_magnification_are_finite() {
        let cells: Vec<VoronoiCell> = (0..100)
            .map(|i| VoronoiCell::new(Point2D::new((i as f64).cos(), (i as f64).sin()), 0.3, 0.2))
            .collect();
        let lens = MeshLens::new(&cells).unwrap();
        let theta = Point2D::new(2.5, 0.0);
        assert!(lens.ray_trace(theta).is_finite());
        assert!(lens.magnification(theta).is_finite());
    }

    #[test]
    fn rejects_empty_and_bad_cells() {
        assert!(MeshLens::new(&[]).is_err());
        assert!(MeshLens::new(&[VoronoiCell::new(Point2D::origin(), 0.5, 0.0)]).is_err());
        assert!(MeshLens::new(&[VoronoiCell::new(Point2D::origin(), 0.5, -1.0)]).is_err());
        assert!(MeshLens::new(&[VoronoiCell::new(Point2D::origin(), f64::NAN, 1.0)]).is_err());
    }
}
