// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens on a uniform pixel grid.
//!
//! The structured counterpart to [`MeshLens`](crate::lens::MeshLens), with the field
//! produced by a chosen [`FieldMethod`]. The default
//! [`Spectral`](FieldMethod::Spectral) solves $\nabla^2\psi = 2\kappa$ on the
//! [`PixelGrid`] by FFT (the same k-space divide-by-$-k^2$ as
//! [`poisson_spectral`](crate::grid::PixelGrid::poisson_spectral)), forms the
//! deflection and Hessian in that same k-space
//! ($\hat\alpha = i\mathbf{k}\hat\psi$, $\hat H = -\mathbf{k}\mathbf{k}^\top\hat\psi$),
//! then interpolates — periodic and zero-mean, so it needs power-of-two side lengths.
//! The two free-space methods instead impose an isolated (non-periodic) boundary:
//! [`FastSum`](FieldMethod::FastSum) sums a monopole per pixel (shared with
//! [`MeshLens`](crate::lens::MeshLens)), while [`Analytic`](FieldMethod::Analytic)
//! sums the exact square-pixel log integral
//! ([`log_kernel`](crate::grid::PixelGrid::log_kernel)) — they agree in the far
//! field but the latter is exact near and inside cells.

use crate::Result;
use crate::fastsum::{Summation, build_summation};
use crate::grid::{ComplexFieldArray2D, Grid2D, PixelGrid, Point2D, ScalarField2D};
use crate::lens::model::{LensJet, LensModel};
use crate::lens::{FieldMethod, FreeFormLens, MassElement};
use num::complex::Complex;
use rayon::prelude::*;

/// Lens on a uniform pixel grid.
///
/// The convergence κ is stored; how its potential/deflection/Hessian are produced
/// is chosen by [`FieldMethod`] (default [`Spectral`](FieldMethod::Spectral)).
/// Convergence is always the interpolated input κ.
pub struct PixelLens {
    kappa: PixelGrid,
    method: FieldMethod,
    field: PixelField,
}

/// The realised field behind a [`PixelLens`], one variant per [`FieldMethod`].
enum PixelField {
    /// FFT solve: potential, deflection and Hessian precomputed as grids (boxed to
    /// balance the enum), read back by interpolation.
    Spectral(Box<SpectralGrids>),
    /// On-demand free-space summation over pixel-centre mass elements — cannot be
    /// baked onto the grid (each pixel's monopole is singular at its own node).
    FastSum(Box<dyn Summation + Send + Sync>),
    /// On-demand exact per-pixel log integral, evaluated from the stored κ grid.
    Analytic,
}

/// Precomputed potential, deflection and Hessian grids from the spectral solve.
struct SpectralGrids {
    psi: PixelGrid,
    alpha_x: PixelGrid,
    alpha_y: PixelGrid,
    psi_xx: PixelGrid,
    psi_xy: PixelGrid,
    psi_yy: PixelGrid,
}

impl PixelLens {
    /// Build by resampling any scalar κ field onto an explicit `grid` (default method).
    ///
    /// The field is sampled at each pixel centre ([`ScalarField2D::sample`]), so a
    /// [`MeshGrid`](crate::grid::MeshGrid) κ (or any representation) can drive a pixel
    /// lens. Chain [`with_method`](FreeFormLens::with_method) for a non-default method.
    ///
    /// # Arguments
    /// * `field` - convergence κ in any representation.
    /// * `grid` - target uniform grid (side lengths powers of two).
    ///
    /// # Errors
    /// Returns an error if the grid dimensions are not powers of two (FFT requirement).
    pub fn from_field(field: &impl ScalarField2D, grid: Grid2D) -> Result<Self> {
        Self::new(Self::pixelate(field, grid))
    }

    /// Rebuild at a square resolution over the same bounds, preserving the method.
    ///
    /// # Arguments
    /// * `n` - per-side resolution (rounded up to a power of two).
    ///
    /// # Errors
    /// Returns an error if a valid grid cannot be built.
    pub fn with_resolution(self, n: usize) -> Result<Self> {
        let n = n.next_power_of_two();
        let grid = Grid2D::new(n, n, self.kappa.grid.bbox)?;
        self.with_grid(grid)
    }

    /// Rebuild on an explicit `grid`, resampling the current κ and preserving the method.
    ///
    /// # Arguments
    /// * `grid` - target uniform grid (side lengths powers of two).
    ///
    /// # Errors
    /// Returns an error if the grid dimensions are not powers of two.
    pub fn with_grid(self, grid: Grid2D) -> Result<Self> {
        let kappa = Self::pixelate(&self.kappa, grid);
        Self::build(kappa, self.method)
    }

    /// Resample any scalar κ field onto `grid` at pixel centres — the structured
    /// counterpart to [`MeshLens`](crate::lens::MeshLens)'s tessellation.
    fn pixelate(field: &impl ScalarField2D, grid: Grid2D) -> PixelGrid {
        PixelGrid::from_func(grid, |x, y| field.sample(Point2D::new(x, y)))
    }

    /// Pixel-centre mass elements `m = κ·area` for the free-space summation methods.
    fn mass_elements(field: &impl ScalarField2D) -> Vec<MassElement> {
        (0..field.len())
            .map(|i| MassElement::new(field.center(i), field.value_at(i) * field.cell_area(i)))
            .collect()
    }

    /// Exact per-pixel log integral: the κ-weighted sum of the square-pixel
    /// [`log_kernel`](PixelGrid::log_kernel) over every cell of the κ grid.
    fn analytic_jet(&self, theta: Point2D) -> LensJet {
        let mut acc = [0.0f64; 6];
        for i in 0..self.kappa.len() {
            let k = self.kappa.value_at(i);
            if k == 0.0 {
                continue;
            }
            let ker = self.kappa.log_kernel(i, theta);
            for j in 0..6 {
                acc[j] += k * ker[j];
            }
        }
        LensJet {
            psi: acc[0],
            alpha: Point2D::new(acc[1], acc[2]),
            hessian: [[acc[3], acc[4]], [acc[4], acc[5]]],
        }
    }

    /// Solve $\nabla^2\psi = 2\kappa$ by FFT and precompute the deflection and Hessian
    /// fields in k-space for interpolation.
    fn solve_spectral(kappa: &PixelGrid) -> Result<SpectralGrids> {
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
        Ok(SpectralGrids {
            psi: deriv(&|_, _| Complex::new(1.0, 0.0))?,
            alpha_x: deriv(&|kxi, _| Complex::new(0.0, kxi))?,
            alpha_y: deriv(&|_, kyj| Complex::new(0.0, kyj))?,
            psi_xx: deriv(&|kxi, _| Complex::new(-kxi * kxi, 0.0))?,
            psi_xy: deriv(&|kxi, kyj| Complex::new(-kxi * kyj, 0.0))?,
            psi_yy: deriv(&|_, kyj| Complex::new(-kyj * kyj, 0.0))?,
        })
    }
}

impl FreeFormLens for PixelLens {
    type Repr = PixelGrid;

    /// The native structured FFT solve.
    fn default_method() -> FieldMethod {
        FieldMethod::Spectral
    }

    /// A pixel grid supports every method.
    fn supports(_method: FieldMethod) -> bool {
        true
    }

    fn build(kappa: PixelGrid, method: FieldMethod) -> Result<Self> {
        let field = match method {
            FieldMethod::Spectral => PixelField::Spectral(Box::new(Self::solve_spectral(&kappa)?)),
            FieldMethod::FastSum { backend, config } => {
                let sources = Self::mass_elements(&kappa);
                PixelField::FastSum(build_summation(backend, &sources, config))
            }
            FieldMethod::Analytic => PixelField::Analytic,
        };
        Ok(Self {
            kappa,
            method,
            field,
        })
    }

    fn method(&self) -> FieldMethod {
        self.method
    }

    fn with_method(self, method: FieldMethod) -> Result<Self> {
        Self::build(self.kappa, method)
    }
}

impl LensModel for PixelLens {
    fn jet(&self, theta: Point2D) -> LensJet {
        match &self.field {
            PixelField::Spectral(g) => {
                let hxy = g.psi_xy.interpolate_or(theta, 0.0);
                LensJet {
                    psi: g.psi.interpolate_or(theta, 0.0),
                    alpha: Point2D::new(
                        g.alpha_x.interpolate_or(theta, 0.0),
                        g.alpha_y.interpolate_or(theta, 0.0),
                    ),
                    hessian: [
                        [g.psi_xx.interpolate_or(theta, 0.0), hxy],
                        [hxy, g.psi_yy.interpolate_or(theta, 0.0)],
                    ],
                }
            }
            PixelField::FastSum(s) => s.jet(theta).into(),
            PixelField::Analytic => self.analytic_jet(theta),
        }
    }

    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<LensJet> {
        match &self.field {
            PixelField::Spectral(g) => {
                let psi = g.psi.par_interpolate_batch(thetas);
                let ax = g.alpha_x.par_interpolate_batch(thetas);
                let ay = g.alpha_y.par_interpolate_batch(thetas);
                let hxx = g.psi_xx.par_interpolate_batch(thetas);
                let hxy = g.psi_xy.par_interpolate_batch(thetas);
                let hyy = g.psi_yy.par_interpolate_batch(thetas);
                (0..thetas.len())
                    .map(|i| LensJet {
                        psi: psi[i],
                        alpha: Point2D::new(ax[i], ay[i]),
                        hessian: [[hxx[i], hxy[i]], [hxy[i], hyy[i]]],
                    })
                    .collect()
            }
            PixelField::FastSum(s) => s.jet_batch(thetas).into_iter().map(Into::into).collect(),
            PixelField::Analytic => thetas.par_iter().map(|&t| self.analytic_jet(t)).collect(),
        }
    }

    /// Convergence at `theta` — the interpolated input κ field.
    fn convergence(&self, theta: Point2D) -> f64 {
        self.kappa.interpolate_or(theta, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::Backend;
    use std::f64::consts::{FRAC_1_PI, PI};

    const N: usize = 64;
    const L: f64 = 5.0; // domain [-L, L]²

    /// κ(x,y) = A cos(k x), zero-mean and periodic over the box; ψ, α, H analytic:
    /// ψ = -2A/k² cos(kx) ⇒ α_x = (2A/k) sin(kx), α_y = 0, ½ tr H = A cos(kx) = κ.
    fn cosine_lens(a: f64) -> (PixelLens, f64, f64) {
        let k = 2.0 * PI / (2.0 * L); // fundamental mode across the box
        let kappa = PixelGrid::from_func_sq(N, L, |x, _y| a * (k * x).cos());
        (PixelLens::new(kappa).unwrap(), a, k)
    }

    #[test]
    fn new_defaults_to_spectral() {
        let (lens, _, _) = cosine_lens(0.3);
        assert_eq!(lens.method(), FieldMethod::Spectral);
    }

    #[test]
    fn from_field_matches_new_on_native_grid() {
        // from_field resamples onto the grid; on the native grid it matches new().
        let (a, k) = (0.3, 2.0 * PI / (2.0 * L));
        let kappa = PixelGrid::from_func_sq(N, L, |x, _| a * (k * x).cos());
        let via_new = PixelLens::new(kappa.clone()).unwrap();
        let via_field = PixelLens::from_field(&kappa, kappa.grid).unwrap();
        let p = Point2D::new(0.7, 0.4);
        assert!((via_new.jet(p).alpha.x - via_field.jet(p).alpha.x).abs() < 1e-12);
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
    fn chaining_rebuilds_at_resolution() {
        // new() then a finer resolution rebuilds and still recovers the analytic α.
        let (a, k) = (0.3, 2.0 * PI / (2.0 * L));
        let src = PixelGrid::from_func_sq(N, L, |x, _| a * (k * x).cos());
        let lens = PixelLens::new(src).unwrap().with_resolution(128).unwrap();
        let x = 0.7;
        let j = lens.jet(Point2D::new(x, 0.4));
        assert!(
            (j.alpha.x - 2.0 * a / k * (k * x).sin()).abs() < 5e-3,
            "αx: {}",
            j.alpha.x
        );
    }

    #[test]
    fn with_grid_rebuilds_on_explicit_grid() {
        let (a, k) = (0.3, 2.0 * PI / (2.0 * L));
        let src = PixelGrid::from_func_sq(N, L, |x, _| a * (k * x).cos());
        let grid = Grid2D::square(128, L).unwrap();
        let lens = PixelLens::new(src).unwrap().with_grid(grid).unwrap();
        let x = 0.7;
        let j = lens.jet(Point2D::new(x, 0.4));
        assert!(
            (j.alpha.x - 2.0 * a / k * (k * x).sin()).abs() < 5e-3,
            "αx: {}",
            j.alpha.x
        );
    }

    #[test]
    fn rejects_non_power_of_two_grid() {
        let src = PixelGrid::from_func_sq(N, L, |_, _| 0.3);
        let grid = Grid2D::new(100, 100, src.grid.bbox).unwrap(); // 100 is not a power of two
        assert!(PixelLens::from_field(&src, grid).is_err());
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

    #[test]
    fn fast_sum_matches_direct_monopole_sum() {
        // FastSum turns each pixel into a monopole m = κ·area at its centre; the
        // free-space field must match a direct sum of those monopoles to round-off.
        let kappa = PixelGrid::from_func_sq(N, L, |x, y| 0.2 + 0.05 * x - 0.03 * y);
        let pix = PixelLens::new(kappa.clone())
            .unwrap()
            .with_method(FieldMethod::fast_sum(Backend::Direct))
            .unwrap();
        let theta = Point2D::new(7.0, 5.0); // exterior point, clear of every pixel centre
        let field: &dyn ScalarField2D = &kappa;
        let (mut ax, mut ay) = (0.0, 0.0);
        for i in 0..field.len() {
            let c = field.center(i);
            let m = field.value_at(i) * field.cell_area(i);
            let r2 = theta.distance_squared(&c);
            ax += m * FRAC_1_PI * (theta.x - c.x) / r2;
            ay += m * FRAC_1_PI * (theta.y - c.y) / r2;
        }
        let a = pix.deflection(theta);
        assert!(
            (a.x - ax).abs() < 1e-9 && (a.y - ay).abs() < 1e-9,
            "a=({},{}) exp=({},{})",
            a.x,
            a.y,
            ax,
            ay
        );
    }

    /// Build a linear-ramp κ lens with the exact per-pixel Analytic method.
    fn analytic_ramp_lens() -> PixelLens {
        let kappa = PixelGrid::from_func_sq(N, L, |x, y| 0.2 + 0.05 * x - 0.03 * y);
        PixelLens::new(kappa)
            .unwrap()
            .with_method(FieldMethod::Analytic)
            .unwrap()
    }

    #[test]
    fn analytic_convergence_inside_matches_kappa() {
        // ½ tr H = κ inside a uniform field, since ∇²ψ = 2κ pointwise.
        let kappa = PixelGrid::from_func_sq(N, L, |_, _| 0.3);
        let lens = PixelLens::new(kappa)
            .unwrap()
            .with_method(FieldMethod::Analytic)
            .unwrap();
        let h = lens.jet(Point2D::new(0.13, -0.07)).hessian;
        assert!((0.5 * (h[0][0] + h[1][1]) - 0.3).abs() < 1e-9);
    }

    #[test]
    fn analytic_matches_fast_sum_far_field() {
        // Each pixel's exact integral tends to its monopole; the two agree far away.
        let analytic = analytic_ramp_lens();
        let kappa = PixelGrid::from_func_sq(N, L, |x, y| 0.2 + 0.05 * x - 0.03 * y);
        let fast = PixelLens::new(kappa)
            .unwrap()
            .with_method(FieldMethod::fast_sum(Backend::Direct))
            .unwrap();
        let theta = Point2D::new(40.0, 30.0);
        let (a, b) = (analytic.deflection(theta), fast.deflection(theta));
        assert!((a.x - b.x).abs() < 1e-5 && (a.y - b.y).abs() < 1e-5, "a=({},{})", a.x, a.y);
    }

    #[test]
    fn analytic_alpha_is_gradient_of_psi() {
        // Central-difference ψ recovers α — validates the potential against its gradient.
        let lens = analytic_ramp_lens();
        let (p, e) = (Point2D::new(0.13, -0.07), 1e-4);
        let dpsi_dx =
            (lens.jet(Point2D::new(p.x + e, p.y)).psi - lens.jet(Point2D::new(p.x - e, p.y)).psi)
                / (2.0 * e);
        assert!((dpsi_dx - lens.jet(p).alpha.x).abs() < 1e-6, "{dpsi_dx}");
    }

    #[test]
    fn analytic_skips_zero_pixels() {
        // Empty (κ = 0) cells are skipped; the field is still finite and non-trivial.
        let kappa = PixelGrid::from_func_sq(N, L, |x, _| if x > 0.0 { 0.3 } else { 0.0 });
        let lens = PixelLens::new(kappa)
            .unwrap()
            .with_method(FieldMethod::Analytic)
            .unwrap();
        let j = lens.jet(Point2D::new(1.0, 0.5));
        assert!(j.psi.is_finite() && j.alpha.x.abs() > 0.0);
    }

    #[test]
    fn analytic_jet_batch_matches_jet() {
        let lens = analytic_ramp_lens();
        let pts = [Point2D::new(0.13, -0.07), Point2D::new(2.1, 1.4)];
        for (b, &p) in lens.jet_batch(&pts).iter().zip(&pts) {
            let s = lens.jet(p);
            assert_eq!(b.psi, s.psi);
            assert_eq!(b.alpha.x, s.alpha.x);
            assert_eq!(b.hessian, s.hessian);
        }
    }

    #[test]
    fn supports_every_method() {
        assert!(PixelLens::supports(FieldMethod::Spectral));
        assert!(PixelLens::supports(FieldMethod::Analytic));
        assert!(PixelLens::supports(FieldMethod::fast_sum(Backend::Direct)));
    }

    #[test]
    fn fast_sum_jet_batch_matches_jet() {
        let kappa = PixelGrid::from_func_sq(N, L, |x, y| 0.2 + 0.05 * x - 0.03 * y);
        let lens = PixelLens::new(kappa)
            .unwrap()
            .with_method(FieldMethod::fast_sum(Backend::Direct))
            .unwrap();
        let pts = [Point2D::new(7.0, 5.0), Point2D::new(-6.0, 4.0)]; // exterior points
        for (b, &p) in lens.jet_batch(&pts).iter().zip(&pts) {
            let s = lens.jet(p);
            assert_eq!(b.psi, s.psi);
            assert_eq!(b.alpha.x, s.alpha.x);
            assert_eq!(b.hessian, s.hessian);
        }
    }
}
