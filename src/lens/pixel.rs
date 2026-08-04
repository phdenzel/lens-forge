// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens on a uniform pixel grid.
//!
//! The structured counterpart to [`MeshLens`](crate::lens::MeshLens): the potential
//! solves $\nabla^2\psi = 2\kappa$ on the [`PixelGrid`] by FFT (the same k-space
//! divide-by-$-k^2$ as [`poisson_spectral`](crate::grid::PixelGrid::poisson_spectral)),
//! and the deflection and Hessian are formed in that same k-space
//! ($\hat\alpha = i\mathbf{k}\hat\psi$, $\hat H = -\mathbf{k}\mathbf{k}^\top\hat\psi$),
//! then bilinearly interpolated. The FFT solve is periodic and zero-mean — a different
//! boundary condition from the free-space [`MeshLens`](crate::lens::MeshLens), so the two suit different
//! setups. Grid side lengths must be powers of two.

use crate::Result;
use crate::grid::{ComplexFieldArray2D, Grid2D, PixelGrid, Point2D, ScalarField2D};
use crate::lens::model::{LensJet, LensModel};
use num::complex::Complex;

/// Lens on a uniform pixel grid, its potential solved by FFT.
///
/// The deflection and Hessian are precomputed as fields at construction and read
/// back by interpolation; convergence is the interpolated input κ.
pub struct PixelLens {
    kappa: PixelGrid,
    psi: PixelGrid,
    alpha_x: PixelGrid,
    alpha_y: PixelGrid,
    psi_xx: PixelGrid,
    psi_xy: PixelGrid,
    psi_yy: PixelGrid,
}

impl PixelLens {
    /// Build a lens from a scalar κ field, resampled onto a grid derived from the
    /// field — the counterpart to [`MeshLens::new`](crate::lens::MeshLens::new).
    ///
    /// The grid spans the field's [`bounds`](ScalarField2D::bounds) at a per-side
    /// resolution matching its sampling density (≈√N for an N-cell field). Chain
    /// [`with_resolution`](Self::with_resolution) / [`with_grid`](Self::with_grid)
    /// to override.
    ///
    /// # Arguments
    /// * `field` - convergence κ in any representation.
    ///
    /// # Errors
    /// Returns an error if a valid power-of-two grid cannot be built.
    pub fn new(field: &impl ScalarField2D) -> Result<Self> {
        let n = ((field.len() as f64).sqrt().ceil() as usize).next_power_of_two();
        Self::from_field(field, Grid2D::new(n, n, field.bounds())?)
    }

    /// Build a lens by resampling any scalar κ field onto an explicit `grid`.
    ///
    /// The field is sampled at each pixel centre ([`ScalarField2D::sample`]), so a
    /// [`MeshGrid`](crate::grid::MeshGrid) κ (or any representation) can drive a
    /// pixel lens — the point of the [`ScalarField2D`] abstraction.
    ///
    /// # Arguments
    /// * `field` - convergence κ in any representation.
    /// * `grid` - target uniform grid (side lengths powers of two).
    ///
    /// # Errors
    /// Returns an error if the grid dimensions are not powers of two (FFT requirement).
    pub fn from_field(field: &impl ScalarField2D, grid: Grid2D) -> Result<Self> {
        let kappa = PixelGrid::from_func(grid, |x, y| field.sample(Point2D::new(x, y)));
        Self::solve(kappa)
    }

    /// Build a lens directly from a κ [`PixelGrid`], using its own grid (no resampling).
    ///
    /// # Arguments
    /// * `kappa` - convergence κ on a uniform grid (side lengths powers of two).
    ///
    /// # Errors
    /// Returns an error if the grid dimensions are not powers of two (FFT requirement).
    pub fn from_grid(kappa: PixelGrid) -> Result<Self> {
        Self::solve(kappa)
    }

    /// Rebuild at a square resolution over the same bounds (chainable).
    ///
    /// # Arguments
    /// * `n` - per-side resolution (rounded up to a power of two).
    ///
    /// # Errors
    /// Returns an error if a valid grid cannot be built.
    pub fn with_resolution(self, n: usize) -> Result<Self> {
        let n = n.next_power_of_two();
        let grid = Grid2D::new(n, n, self.kappa.grid.bbox)?;
        Self::from_field(&self.kappa, grid)
    }

    /// Rebuild on an explicit `grid`, resampling the current κ (chainable).
    ///
    /// # Arguments
    /// * `grid` - target uniform grid (side lengths powers of two).
    ///
    /// # Errors
    /// Returns an error if the grid dimensions are not powers of two.
    pub fn with_grid(self, grid: Grid2D) -> Result<Self> {
        Self::from_field(&self.kappa, grid)
    }

    /// Solve $\nabla^2\psi = 2\kappa$ by FFT and precompute the deflection and Hessian
    /// fields in k-space for interpolation.
    fn solve(kappa: PixelGrid) -> Result<Self> {
        let grid = kappa.grid;
        let (kx, ky) = grid.wavenumbers();
        let (ny, nx) = (grid.ny, grid.nx);
        let khat = kappa.fft2d()?;
        // ψ̂ = -2κ̂/k² (∇²ψ = 2κ); the k = 0 (mean) mode is set to zero.
        let mut psih = ComplexFieldArray2D::zeros((ny, nx));
        for j in 0..ny {
            for i in 0..nx {
                let k2 = kx[i] * kx[i] + ky[j] * ky[j];
                if k2 > 0.0 {
                    psih[[j, i]] = khat[[j, i]] * Complex::new(-2.0 / k2, 0.0);
                }
            }
        }
        // Multiply ψ̂ by a k-space symbol and transform back to a real field.
        let deriv = |mult: &dyn Fn(f64, f64) -> Complex<f64>| -> Result<PixelGrid> {
            let mut spec = ComplexFieldArray2D::zeros((ny, nx));
            for j in 0..ny {
                for i in 0..nx {
                    spec[[j, i]] = psih[[j, i]] * mult(kx[i], ky[j]);
                }
            }
            PixelGrid::from_values(grid, PixelGrid::ifft2d_unchecked(&spec))
        };
        let psi = deriv(&|_, _| Complex::new(1.0, 0.0))?;
        let alpha_x = deriv(&|kxi, _| Complex::new(0.0, kxi))?;
        let alpha_y = deriv(&|_, kyj| Complex::new(0.0, kyj))?;
        let psi_xx = deriv(&|kxi, _| Complex::new(-kxi * kxi, 0.0))?;
        let psi_xy = deriv(&|kxi, kyj| Complex::new(-kxi * kyj, 0.0))?;
        let psi_yy = deriv(&|_, kyj| Complex::new(-kyj * kyj, 0.0))?;
        Ok(Self {
            kappa,
            psi,
            alpha_x,
            alpha_y,
            psi_xx,
            psi_xy,
            psi_yy,
        })
    }
}

impl LensModel for PixelLens {
    fn jet(&self, theta: Point2D) -> LensJet {
        let hxy = self.psi_xy.interpolate_or(theta, 0.0);
        LensJet {
            psi: self.psi.interpolate_or(theta, 0.0),
            alpha: Point2D::new(
                self.alpha_x.interpolate_or(theta, 0.0),
                self.alpha_y.interpolate_or(theta, 0.0),
            ),
            hessian: [
                [self.psi_xx.interpolate_or(theta, 0.0), hxy],
                [hxy, self.psi_yy.interpolate_or(theta, 0.0)],
            ],
        }
    }

    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<LensJet> {
        let psi = self.psi.par_interpolate_batch(thetas);
        let ax = self.alpha_x.par_interpolate_batch(thetas);
        let ay = self.alpha_y.par_interpolate_batch(thetas);
        let hxx = self.psi_xx.par_interpolate_batch(thetas);
        let hxy = self.psi_xy.par_interpolate_batch(thetas);
        let hyy = self.psi_yy.par_interpolate_batch(thetas);
        (0..thetas.len())
            .map(|i| LensJet {
                psi: psi[i],
                alpha: Point2D::new(ax[i], ay[i]),
                hessian: [[hxx[i], hxy[i]], [hxy[i], hyy[i]]],
            })
            .collect()
    }

    /// Convergence at `theta` — the interpolated input κ field.
    fn convergence(&self, theta: Point2D) -> f64 {
        self.kappa.interpolate_or(theta, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const N: usize = 64;
    const L: f64 = 5.0; // domain [-L, L]²

    /// κ(x,y) = A cos(k x), zero-mean and periodic over the box; ψ, α, H analytic:
    /// ψ = -2A/k² cos(kx) ⇒ α_x = (2A/k) sin(kx), α_y = 0, ½ tr H = A cos(kx) = κ.
    fn cosine_lens(a: f64) -> (PixelLens, f64, f64) {
        let k = 2.0 * PI / (2.0 * L); // fundamental mode across the box
        let kappa = PixelGrid::from_func_sq(N, L, |x, _y| a * (k * x).cos());
        (PixelLens::from_grid(kappa).unwrap(), a, k)
    }

    #[test]
    fn from_grid_matches_from_field_on_native_grid() {
        // from_grid uses the grid directly; resampling onto the same grid is identical.
        let (a, k) = (0.3, 2.0 * PI / (2.0 * L));
        let kappa = PixelGrid::from_func_sq(N, L, |x, _| a * (k * x).cos());
        let via_grid = PixelLens::from_grid(kappa.clone()).unwrap();
        let via_field = PixelLens::from_field(&kappa, kappa.grid).unwrap();
        let p = Point2D::new(0.7, 0.4);
        assert!((via_grid.jet(p).alpha.x - via_field.jet(p).alpha.x).abs() < 1e-12);
    }

    #[test]
    fn deflection_and_hessian_match_analytic_cosine() {
        let (lens, a, k) = cosine_lens(0.3);
        for &x in &[-1.3, 0.0, 0.7, 2.1] {
            let p = Point2D::new(x, 0.4);
            let j = lens.jet(p);
            assert!(
                (j.alpha.x - 2.0 * a / k * (k * x).sin()).abs() < 5e-3,
                "αx at x={x}: {}",
                j.alpha.x
            );
            assert!(j.alpha.y.abs() < 1e-6, "αy at x={x}: {}", j.alpha.y);
            let kappa = 0.5 * (j.hessian[0][0] + j.hessian[1][1]);
            assert!(
                (kappa - a * (k * x).cos()).abs() < 5e-3,
                "κ at x={x}: {kappa}"
            );
        }
    }

    #[test]
    fn convergence_returns_input_kappa() {
        let (lens, a, k) = cosine_lens(0.25);
        assert!((lens.convergence(Point2D::new(0.6, -0.2)) - a * (k * 0.6).cos()).abs() < 1e-2);
    }

    #[test]
    fn new_derives_grid_from_field() {
        // An N×N source ⇒ new() picks an N×N grid over its bounds (≈√len per side),
        // matching from_field on the native grid.
        let (a, k) = (0.3, 2.0 * PI / (2.0 * L));
        let src = PixelGrid::from_func_sq(N, L, |x, _| a * (k * x).cos());
        let via_new = PixelLens::new(&src).unwrap();
        let native = PixelLens::from_field(&src, src.grid).unwrap();
        let p = Point2D::new(0.7, 0.4);
        assert!((via_new.jet(p).alpha.x - native.jet(p).alpha.x).abs() < 1e-9);
    }

    #[test]
    fn chaining_rebuilds_at_resolution() {
        // new() then a finer resolution rebuilds and still recovers the analytic α.
        let (a, k) = (0.3, 2.0 * PI / (2.0 * L));
        let src = PixelGrid::from_func_sq(N, L, |x, _| a * (k * x).cos());
        let lens = PixelLens::new(&src).unwrap().with_resolution(128).unwrap();
        let x = 0.7;
        let j = lens.jet(Point2D::new(x, 0.4));
        assert!(
            (j.alpha.x - 2.0 * a / k * (k * x).sin()).abs() < 5e-3,
            "αx: {}",
            j.alpha.x
        );
    }

    #[test]
    fn jet_batch_matches_jet() {
        let (lens, _, _) = cosine_lens(0.2);
        let pts = [
            Point2D::new(0.1, 0.2),
            Point2D::new(-1.0, 0.5),
            Point2D::new(2.0, -1.5),
        ];
        let batch = lens.jet_batch(&pts);
        for (b, &p) in batch.iter().zip(&pts) {
            let s = lens.jet(p);
            assert_eq!(b.psi, s.psi);
            assert_eq!(b.alpha.x, s.alpha.x);
            assert_eq!(b.hessian[0][1], s.hessian[0][1]);
        }
    }
}
