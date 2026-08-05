// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Convolution for [`MeshGrid`] via diffusion.
//!
//! A Gaussian blur of width σ is the heat equation $\partial_t u = \nabla^2 u$ run
//! to time $t = \sigma^2/2$. We integrate it with implicit (backward-Euler)
//! sub-steps $(A - \Delta t\,K)\,u^{n+1} = A\,u^n$, where $K$ is the symmetric
//! stiffness ([`apply_stiffness`](super::diff)) and $A$ the diagonal of cell areas.
//! The step operator $A - \Delta t K$ is a symmetric positive-definite M-matrix, so
//! a Jacobi-preconditioned CG solves it; the scheme is unconditionally stable,
//! **self-adjoint**, and conserves the area-weighted flux $\sum_i A_i u_i$ exactly.

use super::diff::apply_stiffness;
use super::faces::FaceGeometry;
use super::meshgrid::MeshGrid;
use crate::grid::{FieldArray1D, Kernel, PixelGrid};
use crate::Result;

/// Backward-Euler sub-steps used to reach the diffusion time $t = \sigma^2/2$;
/// more steps sharpen the Gaussian approximation (each is one implicit solve).
const DIFFUSION_STEPS: usize = 16;

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Apply the step operator `M = A − Δt·K` to `x`, writing `out` (with `kbuf` scratch).
fn apply_step(
    mesh: &MeshGrid,
    faces: &FaceGeometry,
    areas: &[f64],
    dt: f64,
    x: &[f64],
    out: &mut [f64],
    kbuf: &mut [f64],
) {
    apply_stiffness(mesh, faces, x, kbuf);
    for i in 0..x.len() {
        out[i] = areas[i] * x[i] - dt * kbuf[i];
    }
}

impl MeshGrid {
    /// Convolve the field with `kernel`, returning a new mesh of the same geometry.
    ///
    /// # Arguments
    /// * `kernel` - the convolution kernel.
    pub fn convolve(&self, kernel: &Kernel) -> Result<Self> {
        match kernel {
            Kernel::Gaussian { sigma } => self.convolve_gaussian_mixture(&[(1.0, *sigma)]),
            Kernel::GaussianMixture(components) => self.convolve_gaussian_mixture(components),
            Kernel::Sampled(kernel) => self.convolve_sampled(kernel),
        }
    }

    /// Convolve with a sampled `kernel` by kernel-gather: for each cell, gather the
    /// cells within the kernel's support (a k-d tree range query) and form the
    /// kernel-weighted average `Σ K·Aⱼ·vⱼ / Σ K·Aⱼ`. Normalizing per cell keeps a
    /// constant field exactly fixed and the output within the input's range.
    ///
    /// # Arguments
    /// * `kernel` - the sampled kernel image, centred on the origin.
    fn convolve_sampled(&self, kernel: &PixelGrid) -> Result<Self> {
        let n = self.len();
        // Support radius = the kernel's farthest corner from the origin.
        let bb = kernel.grid.bbox;
        let hx = bb.min.x.abs().max(bb.max.x.abs());
        let hy = bb.min.y.abs().max(bb.max.y.abs());
        let radius = (hx * hx + hy * hy).sqrt();

        let out: Vec<f64> = (0..n)
            .map(|i| {
                let xi = self.site(i);
                let (mut num, mut den) = (0.0, 0.0);
                for j in self.cells_within(xi, radius) {
                    let w = kernel.interpolate_or(xi - self.site(j), 0.0) * self.area(j);
                    num += w * self.value(j);
                    den += w;
                }
                if den > 0.0 { num / den } else { self.value(i) }
            })
            .collect();
        self.with_values(FieldArray1D::from_vec(out))
    }

    /// Convolve with a Gaussian mixture: each component diffuses the original
    /// field to time $t = \sigma_k^2/2$, and the results are summed with weights.
    ///
    /// # Arguments
    /// * `components` - the `(weight, sigma)` Gaussian components.
    fn convolve_gaussian_mixture(&self, components: &[(f64, f64)]) -> Result<Self> {
        let n = self.len();
        if n == 0 {
            return self.with_values(FieldArray1D::from_vec(self.values().to_vec()));
        }
        let faces = FaceGeometry::build(self.mesh());
        let areas: Vec<f64> = (0..n).map(|i| self.area(i)).collect();
        // Transmissibility sum per cell: Σⱼ Tᵢⱼ, the off-diagonal weight the
        // Jacobi preconditioner needs (Δt-independent, so built once).
        let tsum: Vec<f64> = (0..n)
            .map(|i| {
                let si = self.site(i);
                faces
                    .faces_of(i)
                    .iter()
                    .map(|&(j, len)| {
                        let d = si.distance(&self.site(j));
                        if d > 0.0 { len / d } else { 0.0 }
                    })
                    .sum()
            })
            .collect();

        let orig = self.values();
        let mut acc = vec![0.0f64; n];
        for &(weight, sigma) in components {
            if sigma <= 0.0 {
                // A zero-width Gaussian is the identity.
                for i in 0..n {
                    acc[i] += weight * orig[i];
                }
                continue;
            }
            let diffused = self.diffuse(&faces, &areas, &tsum, sigma, orig);
            for i in 0..n {
                acc[i] += weight * diffused[i];
            }
        }
        self.with_values(FieldArray1D::from_vec(acc))
    }

    /// Diffuse `u0` to time $t = \sigma^2/2$ by backward-Euler sub-steps, returning
    /// the diffused values (one per cell).
    fn diffuse(
        &self,
        faces: &FaceGeometry,
        areas: &[f64],
        tsum: &[f64],
        sigma: f64,
        u0: &[f64],
    ) -> Vec<f64> {
        let n = u0.len();
        let dt = 0.5 * sigma * sigma / DIFFUSION_STEPS as f64;
        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let d = areas[i] + dt * tsum[i];
                if d > 0.0 { d } else { 1.0 }
            })
            .collect();
        let mut u = u0.to_vec();
        for _ in 0..DIFFUSION_STEPS {
            let rhs: Vec<f64> = (0..n).map(|i| areas[i] * u[i]).collect();
            u = solve_step(self, faces, areas, &diag, dt, &rhs, u);
        }
        u
    }
}

/// Solve `(A − Δt·K) x = rhs` by Jacobi-preconditioned CG, warm-started at `x0`.
fn solve_step(
    mesh: &MeshGrid,
    faces: &FaceGeometry,
    areas: &[f64],
    diag: &[f64],
    dt: f64,
    rhs: &[f64],
    x0: Vec<f64>,
) -> Vec<f64> {
    let n = rhs.len();
    let rhs_norm2 = dot(rhs, rhs);
    let mut x = x0;
    if rhs_norm2 == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        return x;
    }
    let thresh = 1e-16 * rhs_norm2; // relative residual tolerance 1e-8
    let max_iter = 10 * n + 50;

    let mut kbuf = vec![0.0; n];
    let mut ax = vec![0.0; n];
    apply_step(mesh, faces, areas, dt, &x, &mut ax, &mut kbuf);
    let mut r: Vec<f64> = (0..n).map(|i| rhs[i] - ax[i]).collect();
    let mut z: Vec<f64> = (0..n).map(|i| r[i] / diag[i]).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut mp = vec![0.0; n];
    for _ in 0..max_iter {
        if dot(&r, &r) <= thresh {
            break;
        }
        apply_step(mesh, faces, areas, dt, &p, &mut mp, &mut kbuf);
        let alpha = rz / dot(&p, &mp);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * mp[i];
        }
        for i in 0..n {
            z[i] = r[i] / diag[i];
        }
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, Point2D, ScalarField2D};
    use crate::utils::rng::Lcg;

    /// A positive Gaussian kernel image (unnormalized: the gather normalizes per cell).
    fn gaussian_kernel() -> PixelGrid {
        let sg = Grid2D::square(7, 0.2).unwrap();
        PixelGrid::from_func(sg, |x, y| (-(x * x + y * y) / (2.0 * 0.08 * 0.08)).exp())
    }

    /// A jittered lattice over the unit box, valued by `f`.
    fn mesh_from<F: Fn(f64, f64) -> f64>(f: F) -> MeshGrid {
        let mut rng = Lcg::new(0x9C7AD);
        let mut sites = Vec::new();
        for gy in 0..14 {
            for gx in 0..14 {
                let x = 0.5 / 14.0 + gx as f64 / 14.0 + 0.02 * (rng.f() - 0.5);
                let y = 0.5 / 14.0 + gy as f64 / 14.0 + 0.02 * (rng.f() - 0.5);
                sites.push(Point2D::new(x, y));
            }
        }
        MeshGrid::from_func(&sites, BBox::from_lims(0.0, 1.0, 0.0, 1.0), f).unwrap()
    }

    #[test]
    fn constant_field_is_unchanged() {
        // ∇²(const) = 0, so diffusion leaves a constant field fixed.
        let m = mesh_from(|_, _| 2.5);
        let out = m.convolve(&Kernel::gaussian(0.1)).unwrap();
        for i in 0..out.len() {
            assert!((out.value(i) - 2.5).abs() < 1e-8, "cell {i}: {}", out.value(i));
        }
    }

    #[test]
    fn preserves_area_weighted_flux() {
        // Backward Euler conserves Σ Aᵢ uᵢ exactly (K has zero row sums).
        let m = mesh_from(|x, y| x * x - 0.3 * y);
        let out = m.convolve(&Kernel::gaussian(0.08)).unwrap();
        assert!((ScalarField2D::flux(&out) - ScalarField2D::flux(&m)).abs() < 1e-8);
    }

    #[test]
    fn smooths_a_peak_and_stays_nonnegative() {
        // A single hot cell spreads: its value drops, a neighbour rises, and the
        // M-matrix step keeps every value non-negative.
        let mut m = mesh_from(|_, _| 0.0);
        let peak = m.len() / 2;
        m.set(peak, 1.0);
        let nb = m.neighbors(peak)[0];
        let out = m.convolve(&Kernel::gaussian(0.06)).unwrap();
        assert!(out.value(peak) < 1.0);
        assert!(out.value(nb) > 0.0);
        for i in 0..out.len() {
            assert!(out.value(i) >= -1e-12, "cell {i} went negative: {}", out.value(i));
        }
    }

    #[test]
    fn mixture_equals_the_weighted_sum_of_its_components() {
        // The mixture diffuses each component off the original field and sums them.
        let m = mesh_from(|x, y| x * x - 0.3 * y);
        let mix = m
            .convolve(&Kernel::gaussian_mixture(vec![(0.4, 0.05), (0.6, 0.09)]))
            .unwrap();
        let a = m.convolve(&Kernel::gaussian(0.05)).unwrap();
        let b = m.convolve(&Kernel::gaussian(0.09)).unwrap();
        for i in 0..m.len() {
            let want = 0.4 * a.value(i) + 0.6 * b.value(i);
            assert!((mix.value(i) - want).abs() < 1e-9, "cell {i}");
        }
    }

    #[test]
    fn sampled_gather_leaves_a_constant_field_unchanged() {
        // The per-cell normalization makes the weighted gather exact on constants.
        let m = mesh_from(|_, _| 4.0);
        let out = m.convolve(&Kernel::sampled(gaussian_kernel())).unwrap();
        for i in 0..out.len() {
            assert!((out.value(i) - 4.0).abs() < 1e-9, "cell {i}: {}", out.value(i));
        }
    }

    #[test]
    fn sampled_gather_smooths_a_peak_and_stays_bounded() {
        // A hot cell spreads to its support; the normalized average keeps every
        // value within the input's [0, 1] range.
        let mut m = mesh_from(|_, _| 0.0);
        let peak = m.len() / 2;
        m.set(peak, 1.0);
        let out = m.convolve(&Kernel::sampled(gaussian_kernel())).unwrap();
        assert!(out.value(peak) < 1.0);
        assert!((0..m.len()).any(|i| i != peak && out.value(i) > 0.0));
        for i in 0..m.len() {
            let v = out.value(i);
            assert!((-1e-12..=1.0 + 1e-12).contains(&v), "cell {i}: {v}");
        }
    }

    #[test]
    fn trait_convolve_matches_inherent() {
        use crate::grid::FieldOps;
        let m = mesh_from(|x, y| x + 2.0 * y);
        let via_trait = FieldOps::convolve(&m, &Kernel::gaussian(0.05)).unwrap();
        let inherent = m.convolve(&Kernel::gaussian(0.05)).unwrap();
        for i in 0..m.len() {
            assert!((via_trait.value(i) - inherent.value(i)).abs() < 1e-12);
        }
    }
}
