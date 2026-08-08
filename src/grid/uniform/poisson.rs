// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Poisson solvers for [`PixelGrid`]: solve $\nabla^2 \psi = f$.
//!
//! Two backends, both taking the grid's field as the source $f$ and returning
//! the potential $\psi$:
//!
//! * [`PixelGrid::poisson_spectral`] — one FFT divide-by-$-k^2$, $O(N \log N)$;
//!   exact for band-limited periodic data, the fast default. Always projects the
//!   source to zero mean.
//! * [`PixelGrid::poisson_cg`] — conjugate gradient on the 5-point Laplacian;
//!   slower, but takes a [`BoundaryCondition`] and so generalizes to non-periodic
//!   BCs, projecting to zero mean only for the singular cases (`Periodic`,
//!   `Neumann`).
//!
//! For lensing, pass $f = 2\kappa$ to recover the potential from the convergence.

use super::diff::{BoundaryCondition, apply_laplacian_5pt_symm};
use crate::grid::{FieldArray2D, PixelGrid, PoissonConfig};
use crate::numerical::cg::jacobi_pcg;
use crate::{LensForgeError, Result, SolveInfo};
use ndarray::{Array2, Axis, Zip};
use num::complex::Complex;

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
        // fhat came from fft2d, so the dimensions are already power-of-two.
        Ok(Self::ifft2d_unchecked(&fhat))
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
    /// Slower than [`Self::poisson_spectral`], but generalizes to non-periodic BCs.
    ///
    /// # Arguments
    /// * `cfg` - solver configuration (tolerance, iterations, BC; defaults if `None`)
    pub fn poisson_cg(
        &self,
        cfg: impl Into<Option<PoissonConfig>>,
    ) -> Result<(FieldArray2D, SolveInfo)> {
        self.poisson_cg_impl(cfg.into().unwrap_or_default())
    }

    /// Concrete implementation of [`Self::poisson_cg`]. Keeping the solver body in
    /// a single non-generic function (rather than in the `impl Into<…>` wrapper)
    /// avoids re-monomorphizing it per argument type.
    ///
    /// # Arguments
    /// * `cfg` - solver configuration (tolerance, iterations, BC)
    fn poisson_cg_impl(&self, cfg: PoissonConfig) -> Result<(FieldArray2D, SolveInfo)> {
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
        let inv_dx2 = 1.0 / (self.grid.dx() * self.grid.dx());
        let inv_dy2 = 1.0 / (self.grid.dy() * self.grid.dy());

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
        // A x = b with A = -∇² (negated spacings fold the sign in), b = -(f - mean).
        // Plain CG: the 5-point Laplacian has a constant diagonal, so Jacobi
        // preconditioning would be a no-op.
        let b: Vec<f64> = f.iter().map(|&v| -(v - mean)).collect();
        let (x, info) = jacobi_pcg(
            |p, ap| apply_laplacian_5pt_symm(p, nx, ny, -inv_dx2, -inv_dy2, cfg.bc, ap),
            None,
            &b,
            vec![0.0f64; m],
            cfg.tol,
            cfg.max_iter,
        );
        let psi = to_field(x, ny, nx).expect("x has length nx*ny");
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
        assert!(err < 1e-10);
    }

    #[test]
    fn poisson_spectral_ignores_constant_offset() {
        // The k=0 mode is dropped, so a constant added to the source (which only
        // shifts the mean) must not change the recovered zero-mean solution.
        let grid = Grid2D::square(32, PI).unwrap();
        let a = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());
        let b = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos() + 5.0);
        assert!(
            max_abs_err(
                &a.poisson_spectral().unwrap(),
                &b.poisson_spectral().unwrap()
            ) < 1e-10
        );
    }

    #[test]
    fn poisson_cg_solves_smooth_periodic() {
        // Same equation as the spectral test; the 5-point FD solution matches the
        // analytic potential to O(h²).
        let grid = Grid2D::square(64, PI).unwrap();
        let rhs = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());
        let expected = PixelGrid::from_func(grid, |x, y| x.sin() * y.cos());

        let (psi, report) = rhs.poisson_cg(None).unwrap();
        assert!(report.converged);
        assert!(report.residual < 1e-8);
        let err = max_abs_err(&psi, &expected.values);
        assert!(err < 5e-3);
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
    fn source_from_potential(grid: Grid2D, psi: &[f64], bc: BoundaryCondition) -> PixelGrid {
        let nx = grid.nx;
        let ny = grid.ny;
        let inv_dx2 = 1.0 / (grid.dx() * grid.dx());
        let inv_dy2 = 1.0 / (grid.dy() * grid.dy());
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
        let psi = row_major(&PixelGrid::from_func(grid, |x, y| {
            x.sin() * (2.0 * y).sin()
        }));
        let src = source_from_potential(grid, &psi, BoundaryCondition::Dirichlet);

        let cfg = PoissonConfig::default()
            .with_bc(BoundaryCondition::Dirichlet)
            .with_tol(1e-12);
        let (sol, info) = src.poisson_cg(cfg).unwrap();
        assert!(info.converged);
        let expected = to_field(psi, grid.ny, grid.nx).unwrap();
        let err = max_abs_err(&sol, &expected);
        assert!(err < 1e-8);
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
        assert!(info.converged);

        let psi_arr = to_field(psi, grid.ny, grid.nx).unwrap();
        let sol_mean = sol.mean().unwrap();
        let psi_mean = psi_arr.mean().unwrap();
        let err = Zip::from(&sol).and(&psi_arr).fold(0.0_f64, |m, &a, &b| {
            m.max(((a - sol_mean) - (b - psi_mean)).abs())
        });
        assert!(err < 1e-8);
    }

    #[test]
    fn to_field_rejects_length_mismatch() {
        // The reshape helper surfaces a computation error when the buffer length
        // does not match ny*nx.
        assert!(to_field(vec![0.0; 3], 2, 2).is_err());
        assert!(to_field(vec![0.0; 4], 2, 2).is_ok());
    }

    #[test]
    fn poisson_cg_zero_source_short_circuits() {
        // A constant field under a singular BC has a zero-mean source (b == 0), so
        // CG returns the trivial zero solution without iterating.
        let grid = Grid2D::square(8, PI).unwrap();
        let pg = PixelGrid::constant(grid, 3.5);
        let (psi, info) = pg.poisson_cg(None).unwrap();
        assert_eq!(info.iterations, 0);
        assert_eq!(info.residual, 0.0);
        assert!(info.converged);
        assert!(psi.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn poisson_spectral_propagates_fft_error() {
        // The spectral solver relies on an FFT, so non-power-of-two dimensions
        // must surface an error rather than panic.
        let grid = Grid2D::square(6, PI).unwrap();
        let pg = PixelGrid::ones(grid);
        assert!(pg.poisson_spectral().is_err());
    }

    #[test]
    fn poisson_cg_respects_max_iter_cap() {
        // A tiny iteration cap with an impossible tolerance stops unconverged.
        let grid = Grid2D::square(16, PI).unwrap();
        let rhs = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());
        let cfg = PoissonConfig::default().with_tol(1e-16).with_max_iter(2);
        let (_, info) = rhs.poisson_cg(cfg).unwrap();
        assert!(!info.converged);
        assert_eq!(info.iterations, 2);
    }
}
