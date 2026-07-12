// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Poisson solvers for [`PixelGrid`]: solve $\nabla^2 \psi = f$.
//!
//! Two backends, both taking the grid's field as the source $f$ and returning
//! the potential $\psi$ on a periodic domain:
//!
//! * [`PixelGrid::poisson_spectral`] — one FFT divide-by-$-k^2$, $O(N \log N)$.
//!   Exact for band-limited periodic data; the fast default.
//! * [`PixelGrid::poisson_cg`] — conjugate gradient on the 5-point Laplacian,
//!   $O(N)$ memory with iterative sweeps. Slower here, but the route that
//!   generalises to non-periodic boundary conditions where the FFT does not
//!   apply.
//!
//! The periodic problem needs a zero-mean source and is unique up to a constant.
//! [`PixelGrid::poisson_spectral`] always projects to zero mean;
//! [`PixelGrid::poisson_cg`] takes a [`BoundaryCondition`] and projects only for
//! the singular cases (`Periodic`, `Neumann`), returning the unique solution for
//! `Dirichlet`.
//!
//! For lensing, pass $f = 2\kappa$ to recover the potential from the
//! convergence $\kappa$.

use super::diff::{apply_laplacian_5pt_symm, BoundaryCondition};
use crate::grid::{FieldArray2D, PixelGrid};
use crate::{LensForgeError, Result, SolveInfo};
use ndarray::{Array2, Axis, Zip};
use num::complex::Complex;

/// Configuration for the iterative [`PixelGrid::poisson_cg`] solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonConfig {
    /// Relative residual tolerance $\lVert r \rVert / \lVert b \rVert$ at which
    /// the solve is considered converged.
    pub tol: f64,
    /// Maximum number of conjugate-gradient iterations.
    pub max_iter: usize,
    /// Boundary condition for the discrete Laplacian. Must be symmetric;
    /// `Natural` is rejected by [`PixelGrid::poisson_cg`].
    pub bc: BoundaryCondition,
}

impl Default for PoissonConfig {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 10_000,
            bc: BoundaryCondition::Periodic,
        }
    }
}

impl PoissonConfig {
    /// Set the relative residual tolerance.
    ///
    /// # Arguments
    /// * `tol` - relative residual tolerance
    #[inline]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Set the maximum iteration count.
    ///
    /// # Arguments
    /// * `max_iter` - maximum number of CG iterations
    #[inline]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Set the boundary condition.
    ///
    /// # Arguments
    /// * `bc` - boundary condition for the discrete Laplacian
    #[inline]
    pub fn with_bc(mut self, bc: BoundaryCondition) -> Self {
        self.bc = bc;
        self
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Apply $A = -\nabla^2$ into `out` (the operator CG solves against). Negated
/// spacings fold the sign into [`apply_laplacian_5pt_symm`].
#[inline]
fn neg_laplacian(
    u: &[f64],
    nx: usize,
    ny: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    bc: BoundaryCondition,
    out: &mut [f64],
) {
    apply_laplacian_5pt_symm(u, nx, ny, -inv_dx2, -inv_dy2, bc, out);
}

/// Reshape a row-major buffer into a `(ny, nx)` field.
#[inline]
fn to_field(data: Vec<f64>, ny: usize, nx: usize) -> Result<FieldArray2D> {
    Array2::from_shape_vec((ny, nx), data)
        .map_err(|e| LensForgeError::computation_error(e.to_string()))
}

impl PixelGrid {
    /// Solve $\nabla^2 \psi = f$ on a periodic domain by FFT, taking this grid's
    /// field as the source $f$ and returning $\psi$.
    ///
    /// In Fourier space the Laplacian is diagonal, so the solve is a pointwise
    /// division: $\hat\psi(k) = -\hat f(k) / k^2$. The $k = 0$ (mean) mode is set
    /// to zero, returning the zero-mean solution. The whole solve is one forward
    /// and one inverse FFT, $O(N \log N)$.
    ///
    /// See also [`Self::poisson_cg`] for the iterative (non-FFT) counterpart.
    pub fn poisson_spectral(&self) -> Result<FieldArray2D> {
        let (kx, ky) = self.grid.wavenumbers();
        let mut fhat = self.fft2d()?;
        for (j, mut row) in fhat.axis_iter_mut(Axis(0)).enumerate() {
            let ky2 = ky[j] * ky[j];
            Zip::from(&mut row).and(&kx).for_each(|fh, &kxi| {
                let k2 = kxi * kxi + ky2;
                *fh = if k2 > 0.0 {
                    *fh / Complex::new(-k2, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
            });
        }
        Self::ifft2d(&fhat)
    }

    /// Solve $\nabla^2 \psi = f$ by conjugate gradient on the 5-point Laplacian,
    /// taking this grid's field as the source $f$.
    ///
    /// Solves $A\psi = b$ with $A = -\nabla^2$ under [`PoissonConfig::bc`]. The
    /// singular cases (`Periodic`, `Neumann`) project the source to zero mean and
    /// return a zero-mean $\psi$; `Dirichlet` returns the unique solution.
    /// `Natural` is asymmetric and rejected. Returns the solution with a
    /// [`SolveInfo`] (iterations, residual, convergence flag).
    ///
    /// Slower than [`Self::poisson_spectral`], but generalises to non-periodic BCs.
    ///
    /// # Arguments
    /// * `cfg` - solver configuration (tolerance, iterations, BC; defaults if `None`)
    pub fn poisson_cg(
        &self,
        cfg: impl Into<Option<PoissonConfig>>,
    ) -> Result<(FieldArray2D, SolveInfo)> {
        let cfg = cfg.into().unwrap_or_default();
        if cfg.bc == BoundaryCondition::Natural {
            return Err(LensForgeError::computation_error(
                "poisson_cg requires a symmetric boundary condition (Periodic, Dirichlet, or \
                 Neumann); Natural is not supported"
                    .to_string(),
            ));
        }
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let m = nx * ny;
        let inv_dx2 = 1.0 / (self.grid.dx * self.grid.dx);
        let inv_dy2 = 1.0 / (self.grid.dy * self.grid.dy);

        // Source in row-major order. Singular BCs need a zero-mean source for the
        // problem to be solvable; Dirichlet is non-singular so it is left as-is.
        let mut f = vec![0.0f64; m];
        for j in 0..ny {
            for i in 0..nx {
                f[j * nx + i] = self.values[[j, i]];
            }
        }
        let mean = if cfg.bc.is_singular() {
            f.iter().sum::<f64>() / m as f64
        } else {
            0.0
        };
        // A x = b with A = -∇², b = -(f - mean).
        let b: Vec<f64> = f.iter().map(|&v| -(v - mean)).collect();
        let bnorm2 = dot(&b, &b);

        let mut x = vec![0.0f64; m];
        if bnorm2 == 0.0 {
            let psi = to_field(x, ny, nx)?;
            return Ok((
                psi,
                SolveInfo {
                    iterations: 0,
                    residual: 0.0,
                    converged: true,
                },
            ));
        }
        let thresh = cfg.tol * cfg.tol * bnorm2;

        let mut r = b;
        let mut p = r.clone();
        let mut ap = vec![0.0f64; m];
        let mut rs = dot(&r, &r);
        let mut iterations = 0;
        let mut converged = rs <= thresh;

        while iterations < cfg.max_iter && !converged {
            neg_laplacian(&p, nx, ny, inv_dx2, inv_dy2, cfg.bc, &mut ap);
            let alpha = rs / dot(&p, &ap);
            for k in 0..m {
                x[k] += alpha * p[k];
                r[k] -= alpha * ap[k];
            }
            let rs_new = dot(&r, &r);
            iterations += 1;
            if rs_new <= thresh {
                rs = rs_new;
                converged = true;
                break;
            }
            let beta = rs_new / rs;
            for k in 0..m {
                p[k] = r[k] + beta * p[k];
            }
            rs = rs_new;
        }

        let psi = to_field(x, ny, nx)?;
        let info = SolveInfo {
            iterations,
            residual: (rs / bnorm2).sqrt(),
            converged,
        };
        Ok((psi, info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use std::f64::consts::PI;

    fn max_abs_err(a: &FieldArray2D, b: &FieldArray2D) -> f64 {
        Zip::from(a)
            .and(b)
            .fold(0.0_f64, |m, &x, &y| m.max((x - y).abs()))
    }

    #[test]
    fn poisson_spectral_inverts_laplacian() {
        // Solve ∇²ψ = f with f = -2 sin(x)cos(y); expect ψ = sin(x)cos(y).
        let grid = Grid2D::square(64, PI).unwrap();
        let rhs = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());
        let expected = PixelGrid::from_func(grid, |x, y| x.sin() * y.cos());

        let psi = rhs.poisson_spectral().unwrap();
        let err = max_abs_err(&psi, &expected.values);
        assert!(err < 1e-10, "poisson err {err:e}");
    }

    #[test]
    fn poisson_spectral_ignores_constant_offset() {
        // The k=0 mode is dropped, so a constant added to the source (which only
        // shifts the mean) must not change the recovered zero-mean solution.
        let grid = Grid2D::square(32, PI).unwrap();
        let a = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());
        let b = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos() + 5.0);
        assert!(max_abs_err(&a.poisson_spectral().unwrap(), &b.poisson_spectral().unwrap()) < 1e-10);
    }

    #[test]
    fn poisson_cg_solves_smooth_periodic() {
        // Same equation as the spectral test; the 5-point FD solution matches the
        // analytic potential to O(h²).
        let grid = Grid2D::square(64, PI).unwrap();
        let rhs = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());
        let expected = PixelGrid::from_func(grid, |x, y| x.sin() * y.cos());

        let (psi, report) = rhs.poisson_cg(None).unwrap();
        assert!(report.converged, "CG did not converge: {report:?}");
        assert!(report.residual < 1e-8, "residual {}", report.residual);
        let err = max_abs_err(&psi, &expected.values);
        assert!(err < 5e-3, "CG solution err {err:e}");
    }

    #[test]
    fn poisson_cg_ignores_constant_offset() {
        // CG subtracts the mean internally, so a constant offset in the source
        // leaves the solution unchanged.
        let grid = Grid2D::square(32, PI).unwrap();
        let a = PixelGrid::from_func(grid, |x, y| (2.0 * x).sin() * (y).cos());
        let b = PixelGrid::from_func(grid, |x, y| (2.0 * x).sin() * (y).cos() + 3.0);

        let cfg = PoissonConfig::default().with_tol(1e-10);
        let (pa, _) = a.poisson_cg(cfg).unwrap();
        let (pb, _) = b.poisson_cg(cfg).unwrap();
        assert!(max_abs_err(&pa, &pb) < 1e-9);
    }

    /// Build the discrete source `f = ∇²ψ` for a chosen `psi` under `bc`, so a
    /// solve of `∇²ψ = f` should return `psi` (the operator is its own forward
    /// map — a round-trip consistency check independent of any analytic edge
    /// mismatch).
    fn source_from_potential(
        grid: Grid2D,
        psi: &[f64],
        bc: BoundaryCondition,
    ) -> PixelGrid {
        let nx = grid.nx;
        let ny = grid.ny;
        let inv_dx2 = 1.0 / (grid.dx * grid.dx);
        let inv_dy2 = 1.0 / (grid.dy * grid.dy);
        let mut f = vec![0.0; nx * ny];
        apply_laplacian_5pt_symm(psi, nx, ny, inv_dx2, inv_dy2, bc, &mut f);
        PixelGrid::from_values(grid, to_field(f, ny, nx).unwrap()).unwrap()
    }

    fn row_major(pg: &PixelGrid) -> Vec<f64> {
        let nx = pg.grid.nx;
        let ny = pg.grid.ny;
        let mut v = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                v[j * nx + i] = pg.values[[j, i]];
            }
        }
        v
    }

    #[test]
    fn poisson_cg_rejects_natural_bc() {
        // Natural (one-sided) is asymmetric, so CG must refuse it.
        let grid = Grid2D::square(8, PI).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x.sin() * y.cos());
        let cfg = PoissonConfig::default().with_bc(BoundaryCondition::Natural);
        assert!(pg.poisson_cg(cfg).is_err());
    }

    #[test]
    fn poisson_cg_dirichlet_recovers_solution() {
        // Dirichlet is SPD → unique solution, recovered exactly (no projection).
        let grid = Grid2D::square(24, PI).unwrap();
        let psi = row_major(&PixelGrid::from_func(grid, |x, y| x.sin() * (2.0 * y).sin()));
        let src = source_from_potential(grid, &psi, BoundaryCondition::Dirichlet);

        let cfg = PoissonConfig::default()
            .with_bc(BoundaryCondition::Dirichlet)
            .with_tol(1e-12);
        let (sol, info) = src.poisson_cg(cfg).unwrap();
        assert!(info.converged, "CG did not converge: {info:?}");
        let expected = to_field(psi, grid.ny, grid.nx).unwrap();
        let err = max_abs_err(&sol, &expected);
        assert!(err < 1e-8, "dirichlet err {err:e}");
    }

    #[test]
    fn poisson_cg_neumann_recovers_up_to_constant() {
        // Neumann is singular (constants in the null space): recover ψ up to an
        // additive constant, so compare the zero-mean parts.
        let grid = Grid2D::square(24, PI).unwrap();
        let psi = row_major(&PixelGrid::from_func(grid, |x, y| x.cos() * y.cos()));
        let src = source_from_potential(grid, &psi, BoundaryCondition::Neumann);

        let cfg = PoissonConfig::default()
            .with_bc(BoundaryCondition::Neumann)
            .with_tol(1e-12);
        let (sol, info) = src.poisson_cg(cfg).unwrap();
        assert!(info.converged, "CG did not converge: {info:?}");

        let psi_arr = to_field(psi, grid.ny, grid.nx).unwrap();
        let sol_mean = sol.mean().unwrap();
        let psi_mean = psi_arr.mean().unwrap();
        let err = Zip::from(&sol).and(&psi_arr).fold(0.0_f64, |m, &a, &b| {
            m.max(((a - sol_mean) - (b - psi_mean)).abs())
        });
        assert!(err < 1e-8, "neumann err {err:e}");
    }
}
