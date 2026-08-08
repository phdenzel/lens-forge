// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens on a uniform pixel grid.
//!
//! The structured counterpart to [`MeshLens`](crate::lens::MeshLens), with the field
//! produced by a chosen [`FieldMethod`]. The default
//! [`Analytic`](FieldMethod::Analytic) sums the exact square-pixel log integral
//! ([`square_log_kernel`](crate::grid::square_log_kernel)) — exact near and inside
//! cells; like [`FastSum`](FieldMethod::FastSum) (a monopole per pixel, shared with
//! [`MeshLens`](crate::lens::MeshLens)) it imposes an isolated free-space boundary and
//! is accelerable by any fastsum backend, the two agreeing in the far field.
//! [`Spectral`](FieldMethod::Spectral) instead solves $\nabla^2\psi = 2\kappa$ on the
//! [`PixelGrid`] by FFT (the same k-space divide-by-$-k^2$ as
//! [`poisson_spectral`](crate::grid::PixelGrid::poisson_spectral)), forming the
//! deflection and Hessian in that k-space
//! ($\hat\alpha = i\mathbf{k}\hat\psi$, $\hat H = -\mathbf{k}\mathbf{k}^\top\hat\psi$)
//! then interpolating — periodic and zero-mean, so it needs power-of-two side lengths.

use super::source::{MassElement, PixelCellSource};
use crate::Result;
use crate::fastsum::{Backend, Summation, build_summation};
use crate::grid::{Grid2D, PixelGrid, Point2D, ScalarField2D, SpectralJet};
use crate::lens::model::{LensJet, LensModel};
use crate::lens::{FieldMethod, FreeFormLens};

/// Lens on a uniform pixel grid.
///
/// The convergence κ is stored; how its potential/deflection/Hessian are produced
/// is chosen by [`FieldMethod`] (default [`Analytic`](FieldMethod::Analytic) by
/// direct summation). Convergence is always the interpolated input κ.
pub struct PixelLens {
    kappa: PixelGrid,
    method: FieldMethod,
    field: PixelField,
}

/// The realised field behind a [`PixelLens`], one variant per [`FieldMethod`].
enum PixelField {
    /// FFT solve: potential, deflection and Hessian precomputed as grids (boxed to
    /// balance the enum), read back by interpolation.
    Spectral(Box<SpectralJet>),
    /// On-demand free-space summation over pixel-centre mass elements — cannot be
    /// baked onto the grid (each pixel's monopole is singular at its own node).
    FastSum(Box<dyn Summation + Send + Sync>),
    /// On-demand summation over exact per-pixel log-integral sources (same backends
    /// as [`FastSum`](Self::FastSum), exact near/inside cells).
    Analytic(Box<dyn Summation + Send + Sync>),
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
}

impl FreeFormLens for PixelLens {
    type Repr = PixelGrid;

    /// The exact free-space per-pixel integral by direct summation.
    fn default_method() -> FieldMethod {
        FieldMethod::analytic(Backend::Direct)
    }

    /// A pixel grid supports every method.
    fn supports(_method: FieldMethod) -> bool {
        true
    }

    fn build(kappa: PixelGrid, method: FieldMethod) -> Result<Self> {
        let field = match method {
            FieldMethod::Spectral => {
                // ∇²ψ = 2κ, so the FFT Poisson solve takes 2κ as its source.
                let source = PixelGrid::from_values(kappa.grid, &kappa.values * 2.0)?;
                PixelField::Spectral(Box::new(source.poisson_spectral_jet()?))
            }
            FieldMethod::FastSum { backend, config } => {
                let sources = Self::mass_elements(&kappa);
                PixelField::FastSum(build_summation(backend, &sources, config))
            }
            FieldMethod::Analytic { backend, config } => {
                let (hx, hy) = (0.5 * kappa.dx(), 0.5 * kappa.dy());
                let sources = PixelCellSource::from_field(&kappa, hx, hy);
                PixelField::Analytic(build_summation(backend, &sources, config))
            }
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
                let hxy = g.hess_xy.interpolate_or(theta, 0.0);
                LensJet {
                    psi: g.psi.interpolate_or(theta, 0.0),
                    alpha: Point2D::new(
                        g.grad_x.interpolate_or(theta, 0.0),
                        g.grad_y.interpolate_or(theta, 0.0),
                    ),
                    hessian: [
                        [g.hess_xx.interpolate_or(theta, 0.0), hxy],
                        [hxy, g.hess_yy.interpolate_or(theta, 0.0)],
                    ],
                }
            }
            PixelField::FastSum(s) => s.jet(theta).into(),
            PixelField::Analytic(s) => s.jet(theta).into(),
        }
    }

    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<LensJet> {
        match &self.field {
            PixelField::Spectral(g) => {
                let psi = g.psi.par_interpolate_batch(thetas);
                let ax = g.grad_x.par_interpolate_batch(thetas);
                let ay = g.grad_y.par_interpolate_batch(thetas);
                let hxx = g.hess_xx.par_interpolate_batch(thetas);
                let hxy = g.hess_xy.par_interpolate_batch(thetas);
                let hyy = g.hess_yy.par_interpolate_batch(thetas);
                (0..thetas.len())
                    .map(|i| LensJet {
                        psi: psi[i],
                        alpha: Point2D::new(ax[i], ay[i]),
                        hessian: [[hxx[i], hxy[i]], [hxy[i], hyy[i]]],
                    })
                    .collect()
            }
            PixelField::FastSum(s) => s.jet_batch(thetas).into_iter().map(Into::into).collect(),
            PixelField::Analytic(s) => s.jet_batch(thetas).into_iter().map(Into::into).collect(),
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
        // The periodic cosine's analytic α/κ is the Spectral (periodic) solution.
        let k = 2.0 * PI / (2.0 * L); // fundamental mode across the box
        let kappa = PixelGrid::from_func_sq(N, L, |x, _y| a * (k * x).cos());
        (PixelLens::build(kappa, FieldMethod::Spectral).unwrap(), a, k)
    }

    #[test]
    fn new_defaults_to_analytic_direct() {
        let kappa = PixelGrid::from_func_sq(N, L, |_, _| 0.3);
        assert_eq!(
            PixelLens::new(kappa).unwrap().method(),
            FieldMethod::analytic(Backend::Direct)
        );
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
        // A finer resolution rebuilds and still recovers the periodic (Spectral) α.
        let (a, k) = (0.3, 2.0 * PI / (2.0 * L));
        let src = PixelGrid::from_func_sq(N, L, |x, _| a * (k * x).cos());
        let lens = PixelLens::build(src, FieldMethod::Spectral)
            .unwrap()
            .with_resolution(128)
            .unwrap();
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
        let lens = PixelLens::build(src, FieldMethod::Spectral)
            .unwrap()
            .with_grid(grid)
            .unwrap();
        let x = 0.7;
        let j = lens.jet(Point2D::new(x, 0.4));
        assert!(
            (j.alpha.x - 2.0 * a / k * (k * x).sin()).abs() < 5e-3,
            "αx: {}",
            j.alpha.x
        );
    }

    #[test]
    fn spectral_rejects_non_power_of_two_grid() {
        let src = PixelGrid::from_func_sq(N, L, |_, _| 0.3);
        let grid = Grid2D::new(100, 100, src.grid.bbox).unwrap(); // 100 is not a power of two
        // The default (Analytic) needs no FFT, so it builds on any grid …
        let lens = PixelLens::from_field(&src, grid).unwrap();
        // … but Spectral requires power-of-two side lengths.
        assert!(lens.with_method(FieldMethod::Spectral).is_err());
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

    /// Build a linear-ramp κ lens with the default exact Analytic (Direct) method.
    fn analytic_ramp_lens() -> PixelLens {
        let kappa = PixelGrid::from_func_sq(N, L, |x, y| 0.2 + 0.05 * x - 0.03 * y);
        PixelLens::new(kappa).unwrap()
    }

    #[test]
    fn analytic_convergence_inside_matches_kappa() {
        // ½ tr H = κ inside a uniform field, since ∇²ψ = 2κ pointwise.
        let kappa = PixelGrid::from_func_sq(N, L, |_, _| 0.3);
        let lens = PixelLens::new(kappa).unwrap();
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
    fn analytic_handles_zero_kappa_cells() {
        // Zero-κ cells contribute nothing; the field stays finite and non-trivial.
        let kappa = PixelGrid::from_func_sq(N, L, |x, _| if x > 0.0 { 0.3 } else { 0.0 });
        let lens = PixelLens::new(kappa).unwrap();
        let j = lens.jet(Point2D::new(1.0, 0.5));
        assert!(j.psi.is_finite() && j.alpha.x.abs() > 0.0);
    }

    #[test]
    fn analytic_barnes_hut_matches_direct() {
        // Accelerating Analytic with Barnes–Hut agrees with the exact direct sum
        // (and exercises the cell sources' far-field monopole moment).
        let kappa = PixelGrid::from_func_sq(N, L, |x, y| 0.2 + 0.05 * x - 0.03 * y);
        let direct = PixelLens::new(kappa.clone()).unwrap(); // Analytic(Direct)
        let bh = PixelLens::new(kappa)
            .unwrap()
            .with_method(FieldMethod::analytic(Backend::BarnesHut))
            .unwrap();
        let theta = Point2D::new(6.0, 4.0); // exterior point
        let (a, b) = (direct.deflection(theta), bh.deflection(theta));
        assert!(a.distance(&b) / a.norm().max(1e-6) < 2e-2, "a=({},{})", a.x, a.y);
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
        assert!(PixelLens::supports(FieldMethod::analytic(Backend::Direct)));
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
