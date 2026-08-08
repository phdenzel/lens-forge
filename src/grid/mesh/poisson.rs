// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Poisson solver for [`MeshGrid`]: solve $\nabla^2 \psi = f$ on the mesh.
//!
//! The unstructured counterpart to
//! [`PixelGrid::poisson_cg`](crate::grid::PixelGrid::poisson_cg): Jacobi
//! (diagonal) preconditioned conjugate gradient on the symmetric two-point-flux
//! stiffness matrix (`apply_stiffness` in [`diff`](super::diff)), with this
//! grid's field as the source $f$.
//!
//! The default boundary is **natural zero-flux (Neumann)** — box facets carry no
//! flux — so the problem is singular: the source is projected to a zero
//! area-weighted mean (solvability) and $\psi$ is returned with a zero
//! area-weighted mean (the gauge freedom). Setting `cfg.bc = Dirichlet` instead
//! pins $\psi = 0$ on the box (one-sided box-facet fluxes), giving a unique,
//! non-singular solution; every other `bc` value keeps the Neumann default. For
//! lensing, pass $f = 2\kappa$. [`PoissonConfig`] is shared with the uniform
//! solver; only `tol`, `max_iter` and the Dirichlet/Neumann split of `bc` apply.

use super::diff::apply_stiffness;
use super::faces::{FaceGeometry, boundary_transmissibilities};
use super::meshgrid::MeshGrid;
use crate::grid::{BoundaryCondition, FieldArray1D, PoissonConfig};
use crate::numerical::cg::jacobi_pcg;
use crate::{Result, SolveInfo};

impl MeshGrid {
    /// Solve $\nabla^2 \psi = f$ by conjugate gradient, taking this grid's field
    /// as the source $f$ and returning the potential $\psi$ as a mesh grid
    /// sharing this geometry.
    ///
    /// With the default natural zero-flux (Neumann) boundary the problem is
    /// singular: the source is projected to a zero area-weighted mean and the
    /// returned $\psi$ likewise has one. `cfg.bc = Dirichlet` instead pins
    /// $\psi = 0$ on the box for a unique solution. Returns the solution together
    /// with a [`SolveInfo`] (iterations, relative residual, convergence flag).
    ///
    /// # Arguments
    /// * `cfg` - solver configuration; only `tol`, `max_iter` and the
    ///   Dirichlet/Neumann split of `bc` are used (defaults if `None`).
    pub fn poisson_cg(
        &self,
        cfg: impl Into<Option<PoissonConfig>>,
    ) -> Result<(MeshGrid, SolveInfo)> {
        self.poisson_cg_impl(cfg.into().unwrap_or_default())
    }

    /// Concrete solver body, kept non-generic so it is monomorphized once.
    ///
    /// # Arguments
    /// * `cfg` - solver configuration (tolerance, iterations).
    fn poisson_cg_impl(&self, cfg: PoissonConfig) -> Result<(MeshGrid, SolveInfo)> {
        let faces = FaceGeometry::build(self.mesh());
        let n = self.len();
        let areas: Vec<f64> = (0..n).map(|i| self.area(i)).collect();
        let total_area: f64 = areas.iter().sum();

        // Dirichlet pins ψ = 0 on the box (one-sided fluxes) so A = -K + diag(w)
        // is positive definite; any other bc keeps the singular zero-flux boundary.
        let dirichlet = cfg.bc == BoundaryCondition::Dirichlet;
        let bweight = if dirichlet {
            boundary_transmissibilities(self.mesh())
        } else {
            Vec::new()
        };

        // RHS b = -Aᵢ(fᵢ - f̄). Neumann projects f to a zero area-weighted mean
        // for solvability (b ⟂ K's null space); Dirichlet is non-singular, f̄ = 0.
        let fbar = if !dirichlet && total_area > 0.0 {
            (0..n).map(|i| areas[i] * self.value(i)).sum::<f64>() / total_area
        } else {
            0.0
        };
        let b: Vec<f64> = (0..n).map(|i| -areas[i] * (self.value(i) - fbar)).collect();

        // Jacobi preconditioner M = diag(A): the per-cell transmissibility sum
        // (plus the Dirichlet boundary term). Positive on any connected cell; the
        // fallback keeps an isolated cell non-singular.
        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let site = self.site(i);
                let mut d = 0.0;
                for &(j, len) in faces.faces_of(i) {
                    let dist = site.distance(&self.site(j));
                    if dist > 0.0 {
                        d += len / dist;
                    }
                }
                if dirichlet {
                    d += bweight[i];
                }
                if d > 0.0 { d } else { 1.0 }
            })
            .collect();

        // A p = -K p (+ diag(w) p for the pinned Dirichlet boundary).
        let mut kp = vec![0.0f64; n];
        let (mut x, info) = jacobi_pcg(
            |p, ap| {
                apply_stiffness(self, &faces, p, &mut kp);
                for k in 0..n {
                    ap[k] = -kp[k];
                    if dirichlet {
                        ap[k] += bweight[k] * p[k];
                    }
                }
            },
            Some(&diag),
            &b,
            vec![0.0f64; n],
            cfg.tol,
            cfg.max_iter,
        );

        // Neumann is singular: fix the additive-constant gauge (zero area-weighted
        // mean). Dirichlet is unique, so the solution is kept as solved.
        if !dirichlet && total_area > 0.0 {
            let mean = (0..n).map(|i| areas[i] * x[i]).sum::<f64>() / total_area;
            for xi in &mut x {
                *xi -= mean;
            }
        }
        let psi = self.with_values(FieldArray1D::from_vec(x))?;
        Ok((psi, info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Point2D};
    use crate::rng::Lcg;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn unit() -> BBox {
        BBox::from_lims(0.0, 1.0, 0.0, 1.0)
    }

    fn is_interior(m: &MeshGrid, i: usize) -> bool {
        let (x0, x1) = m.extent_x();
        let (y0, y1) = m.extent_y();
        m.cell_polygon(i)
            .iter()
            .all(|v| v.x > x0 + 1e-3 && v.x < x1 - 1e-3 && v.y > y0 + 1e-3 && v.y < y1 - 1e-3)
    }

    /// A jittered lattice over the unit box, valued by `f`.
    fn mesh_from<F: Fn(f64, f64) -> f64>(f: F) -> MeshGrid {
        let mut rng = Lcg::new(0x0155);
        let mut sites = Vec::new();
        for gy in 0..18 {
            for gx in 0..18 {
                let x = 0.5 / 18.0 + gx as f64 / 18.0 + 0.015 * (rng.f() - 0.5);
                let y = 0.5 / 18.0 + gy as f64 / 18.0 + 0.015 * (rng.f() - 0.5);
                sites.push(p(x, y));
            }
        }
        MeshGrid::from_func(&sites, unit(), f).unwrap()
    }

    #[test]
    fn recovers_a_manufactured_potential() {
        // ψ = cos(πx)cos(πy) is a Neumann eigenfunction (∇²ψ = -2π²ψ, ∂ψ/∂n = 0 on
        // the unit box); solving ∇²ψ = f must recover ψ up to the mean-zero gauge.
        use std::f64::consts::PI;
        let psi_exact = |x: f64, y: f64| (PI * x).cos() * (PI * y).cos();
        let m = mesh_from(|x, y| -2.0 * PI * PI * psi_exact(x, y));
        let (psi, info) = m
            .poisson_cg(PoissonConfig::default().with_tol(1e-10))
            .unwrap();
        assert!(info.converged, "did not converge: {info:?}");

        // Compare against the mean-subtracted exact solution on interior cells.
        let total_area = m.total_area();
        let exact_mean = (0..m.len())
            .map(|i| m.area(i) * psi_exact(m.site(i).x, m.site(i).y))
            .sum::<f64>()
            / total_area;
        let mut max_err = 0.0f64;
        for i in 0..m.len() {
            if is_interior(&m, i) {
                let want = psi_exact(m.site(i).x, m.site(i).y) - exact_mean;
                max_err = max_err.max((psi.value(i) - want).abs());
            }
        }
        assert!(max_err < 0.05, "max interior error {max_err}");
    }

    #[test]
    fn dirichlet_recovers_a_manufactured_potential() {
        // ψ = sin(πx)sin(πy) vanishes on the box (homogeneous Dirichlet) and
        // solves ∇²ψ = -2π²ψ, so the pinned-box solve recovers it uniquely.
        use std::f64::consts::PI;
        let psi_exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
        let m = mesh_from(|x, y| -2.0 * PI * PI * psi_exact(x, y));
        let cfg = PoissonConfig::default()
            .with_bc(crate::grid::BoundaryCondition::Dirichlet)
            .with_tol(1e-10);
        let (psi, info) = m.poisson_cg(cfg).unwrap();
        assert!(info.converged, "did not converge: {info:?}");

        // The one-sided box facets are first-order, so error concentrates near the
        // boundary; the interior recovers ψ to the same tolerance as the Neumann case.
        let mut max_err = 0.0f64;
        for i in 0..m.len() {
            if is_interior(&m, i) {
                let want = psi_exact(m.site(i).x, m.site(i).y);
                max_err = max_err.max((psi.value(i) - want).abs());
            }
        }
        assert!(max_err < 0.05, "max interior error {max_err}");
    }

    #[test]
    fn dirichlet_solution_is_not_mean_zero() {
        // A single-signed source pins a non-zero potential on the box, so unlike
        // the Neumann solve the Dirichlet solution keeps a non-zero mean.
        let m = mesh_from(|_, _| 1.0);
        let cfg = PoissonConfig::default()
            .with_bc(crate::grid::BoundaryCondition::Dirichlet)
            .with_tol(1e-10);
        let (psi, info) = m.poisson_cg(cfg).unwrap();
        assert!(info.converged);
        let mean = (0..m.len()).map(|i| m.area(i) * psi.value(i)).sum::<f64>() / m.total_area();
        assert!(mean.abs() > 1e-3, "expected non-zero mean, got {mean}");
    }

    #[test]
    fn solution_has_zero_area_weighted_mean() {
        let m = mesh_from(|x, y| x - y);
        let (psi, _) = m.poisson_cg(None).unwrap();
        let mean = (0..m.len()).map(|i| m.area(i) * psi.value(i)).sum::<f64>() / m.total_area();
        assert!(mean.abs() < 1e-8, "mean {mean}");
    }

    #[test]
    fn converges_on_an_anisotropic_mesh() {
        // Cells stretched 8:1 make the operator ill-conditioned; the Jacobi
        // preconditioner keeps CG converging and recovering the potential.
        use std::f64::consts::PI;
        let mut rng = Lcg::new(0x0A15);
        let bbox = BBox::from_lims(0.0, 8.0, 0.0, 1.0);
        let mut sites = Vec::new();
        for gy in 0..14 {
            for gx in 0..14 {
                let x = 8.0 * (0.5 + gx as f64) / 14.0 + 0.05 * (rng.f() - 0.5);
                let y = (0.5 + gy as f64) / 14.0 + 0.006 * (rng.f() - 0.5);
                sites.push(p(x, y));
            }
        }
        // ψ = cos(πx/8)cos(πy) is a Neumann eigenfunction on this box.
        let psi_exact = |x: f64, y: f64| (PI * x / 8.0).cos() * (PI * y).cos();
        let lambda = PI * PI / 64.0 + PI * PI;
        let m = MeshGrid::from_func(&sites, bbox, |x, y| -lambda * psi_exact(x, y)).unwrap();
        let (psi, info) = m
            .poisson_cg(PoissonConfig::default().with_tol(1e-9))
            .unwrap();
        assert!(info.converged, "did not converge: {info:?}");
        assert!(
            info.iterations < m.len(),
            "too many iterations: {}",
            info.iterations
        );

        let mean = (0..m.len())
            .map(|i| m.area(i) * psi_exact(m.site(i).x, m.site(i).y))
            .sum::<f64>()
            / m.total_area();
        let mut max_err = 0.0f64;
        for i in 0..m.len() {
            if is_interior(&m, i) {
                let want = psi_exact(m.site(i).x, m.site(i).y) - mean;
                max_err = max_err.max((psi.value(i) - want).abs());
            }
        }
        assert!(max_err < 0.1, "max interior error {max_err}");
    }

    #[test]
    fn zero_source_yields_zero_potential() {
        let m = mesh_from(|_, _| 0.0);
        let (psi, info) = m.poisson_cg(None).unwrap();
        assert!(info.converged && info.iterations == 0);
        assert!(psi.as_array1().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn laplacian_of_the_solution_matches_the_source() {
        // End-to-end: ∇²(poisson_cg(f)) ≈ f - f̄ on interior cells.
        let m = mesh_from(|x, y| (2.0 * x - 1.0) * (2.0 * y - 1.0));
        let (psi, info) = m
            .poisson_cg(PoissonConfig::default().with_tol(1e-10))
            .unwrap();
        assert!(info.converged);
        let fbar = (0..m.len()).map(|i| m.area(i) * m.value(i)).sum::<f64>() / m.total_area();
        let lap = psi.laplacian_field();
        for i in 0..m.len() {
            if is_interior(&m, i) {
                let err = (lap.value(i) - (m.value(i) - fbar)).abs();
                assert!(
                    err < 0.15,
                    "cell {i}: lap {} vs f {}",
                    lap.value(i),
                    m.value(i) - fbar
                );
            }
        }
    }
}
