// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Poisson solver for [`MeshGrid`]: solve $\nabla^2 \psi = f$ on the mesh.
//!
//! The unstructured counterpart to
//! [`PixelGrid::poisson_cg`](crate::grid::PixelGrid::poisson_cg): conjugate
//! gradient on the symmetric two-point-flux stiffness matrix (`apply_stiffness`
//! in [`diff`](super::diff)), with this grid's field as the source $f$.
//!
//! Box facets carry no flux — a **natural zero-flux (Neumann)** boundary — so the
//! problem is singular: the source is projected to a zero area-weighted mean
//! (solvability) and $\psi$ is returned with a zero area-weighted mean (the gauge
//! freedom). For lensing, pass $f = 2\kappa$. [`PoissonConfig`] is shared with the
//! uniform solver; only `tol` and `max_iter` apply here.

use super::diff::apply_stiffness;
use super::faces::FaceGeometry;
use super::meshgrid::MeshGrid;
use crate::grid::{FieldArray1D, PoissonConfig};
use crate::{Result, SolveInfo};

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl MeshGrid {
    /// Solve $\nabla^2 \psi = f$ by conjugate gradient, taking this grid's field
    /// as the source $f$ and returning the potential $\psi$ as a mesh grid
    /// sharing this geometry.
    ///
    /// The boundary is the natural zero-flux (Neumann) one, so the problem is
    /// singular: the source is projected to a zero area-weighted mean and the
    /// returned $\psi$ likewise has a zero area-weighted mean. Returns the
    /// solution together with a [`SolveInfo`] (iterations, relative residual,
    /// convergence flag).
    ///
    /// # Arguments
    /// * `cfg` - solver configuration; only `tol` and `max_iter` are used
    ///   (defaults if `None`).
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

        // Solvability: project the source to a zero area-weighted mean, forming
        // b = -Aᵢ(fᵢ - f̄) so A = -K is PSD and b ⟂ constants (K's null space).
        let fbar = if total_area > 0.0 {
            (0..n).map(|i| areas[i] * self.value(i)).sum::<f64>() / total_area
        } else {
            0.0
        };
        let b: Vec<f64> = (0..n).map(|i| -areas[i] * (self.value(i) - fbar)).collect();
        let bnorm2 = dot(&b, &b);

        let mut x = vec![0.0f64; n];
        if bnorm2 == 0.0 {
            let psi = self.with_values(FieldArray1D::from_vec(x))?;
            return Ok((psi, SolveInfo { iterations: 0, residual: 0.0, converged: true }));
        }
        let thresh = cfg.tol * cfg.tol * bnorm2;

        // Standard CG with A p = -K p.
        let mut r = b;
        let mut p = r.clone();
        let mut kp = vec![0.0f64; n];
        let mut rs = dot(&r, &r);
        let mut iterations = 0;
        let mut converged = rs <= thresh;
        while iterations < cfg.max_iter && !converged {
            apply_stiffness(self, &faces, &p, &mut kp);
            let pap: f64 = p.iter().zip(&kp).map(|(pi, ki)| -pi * ki).sum();
            if pap <= 0.0 {
                break; // constant (null-space) direction; nothing more to resolve
            }
            let alpha = rs / pap;
            for k in 0..n {
                x[k] += alpha * p[k];
                r[k] += alpha * kp[k]; // r -= alpha * A p, with A p = -K p
            }
            let rs_new = dot(&r, &r);
            iterations += 1;
            if rs_new <= thresh {
                rs = rs_new;
                converged = true;
                break;
            }
            let beta = rs_new / rs;
            for k in 0..n {
                p[k] = r[k] + beta * p[k];
            }
            rs = rs_new;
        }

        // Fix the additive-constant gauge: zero area-weighted mean.
        if total_area > 0.0 {
            let mean = (0..n).map(|i| areas[i] * x[i]).sum::<f64>() / total_area;
            for xi in &mut x {
                *xi -= mean;
            }
        }
        let psi = self.with_values(FieldArray1D::from_vec(x))?;
        let info = SolveInfo { iterations, residual: (rs / bnorm2).sqrt(), converged };
        Ok((psi, info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Point2D};
    use crate::utils::rng::Lcg;

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
        let (psi, info) = m.poisson_cg(PoissonConfig::default().with_tol(1e-10)).unwrap();
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
    fn solution_has_zero_area_weighted_mean() {
        let m = mesh_from(|x, y| x - y);
        let (psi, _) = m.poisson_cg(None).unwrap();
        let mean = (0..m.len()).map(|i| m.area(i) * psi.value(i)).sum::<f64>() / m.total_area();
        assert!(mean.abs() < 1e-8, "mean {mean}");
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
        let (psi, info) = m.poisson_cg(PoissonConfig::default().with_tol(1e-10)).unwrap();
        assert!(info.converged);
        let fbar = (0..m.len()).map(|i| m.area(i) * m.value(i)).sum::<f64>() / m.total_area();
        let lap = psi.laplacian_field();
        for i in 0..m.len() {
            if is_interior(&m, i) {
                let err = (lap.value(i) - (m.value(i) - fbar)).abs();
                assert!(err < 0.15, "cell {i}: lap {} vs f {}", lap.value(i), m.value(i) - fbar);
            }
        }
    }
}
