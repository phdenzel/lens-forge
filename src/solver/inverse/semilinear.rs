// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Semi-linear source inversion (Warren & Dye 2003).
//!
//! For a fixed lens the lensed image is linear in the source, so the source solves the
//! normal equations `(Fᵀ W F + λ R) s = Fᵀ W d`, with `F` the sparse lensing operator, `W`
//! the mask-weighted inverse-noise covariance, and `R` a smoothness regularizer on the
//! source mesh. The system is SPD, so a conjugate gradient (from a matrix-free operator)
//! solves it without ever forming `Fᵀ W F`.

use crate::grid::MeshGrid;
use crate::instrument::mask::Mask;
use crate::instrument::noise::Noise;
use crate::numerical::cg::jacobi_pcg;
use crate::solver::extended::operator::LensingOperator;
use crate::SolveInfo;

/// Graph-Laplacian smoothness regularizer on a source mesh: `(R s)_i = Σ_{j∈N(i)}(s_i − s_j)`.
pub struct Regularizer {
    neighbors: Vec<Vec<usize>>,
}

impl Regularizer {
    /// Build the regularizer from a source mesh's Delaunay adjacency.
    ///
    /// # Arguments
    /// * `mesh` - the source mesh (columns of `F`).
    pub fn from_mesh(mesh: &MeshGrid) -> Self {
        let neighbors = (0..mesh.len()).map(|i| mesh.neighbors(i).to_vec()).collect();
        Self { neighbors }
    }

    /// The number of source cells.
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Whether the mesh is empty.
    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    /// Apply `R` in place: `out_i = Σ_{j∈N(i)}(s_i − s_j)`.
    ///
    /// # Arguments
    /// * `s` - source coefficients.
    /// * `out` - destination for `R s` (length matches `s`).
    pub fn apply(&self, s: &[f64], out: &mut [f64]) {
        for (i, nbrs) in self.neighbors.iter().enumerate() {
            out[i] = nbrs.iter().map(|&j| s[i] - s[j]).sum();
        }
    }
}

/// Solve `(Fᵀ W F + λ R) s = Fᵀ W d` for the source coefficients `s`.
///
/// `W` is the mask weight times the noise inverse-variance, per image cell.
///
/// # Arguments
/// * `f` - the sparse lensing operator (image ← source).
/// * `data` - the observed image (length `f.n_image()`).
/// * `noise` - the data noise model (supplies `C⁻¹`).
/// * `mask` - the pixel mask (composes multiplicatively with the noise weight).
/// * `reg` - the source smoothness regularizer.
/// * `lambda` - regularization strength.
/// * `tol` - CG relative-residual tolerance.
/// * `max_iter` - CG iteration cap.
pub fn invert_source(
    f: &LensingOperator,
    data: &[f64],
    noise: &Noise,
    mask: &Mask,
    reg: &Regularizer,
    lambda: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, SolveInfo) {
    let n_image = f.n_image();
    let weight: Vec<f64> =
        (0..n_image).map(|i| mask.weight(i) * noise.inverse_variance(i)).collect();

    // rhs = Fᵀ (W d).
    let wd: Vec<f64> = data.iter().zip(&weight).map(|(d, w)| d * w).collect();
    let rhs = f.apply_transpose(&wd);

    // Matrix-free SPD operator s ↦ Fᵀ W F s + λ R s.
    let apply = |s: &[f64], out: &mut [f64]| {
        let fs = f.apply(s);
        let wfs: Vec<f64> = fs.iter().zip(&weight).map(|(v, w)| v * w).collect();
        let ftwfs = f.apply_transpose(&wfs);
        reg.apply(s, out);
        for (o, a) in out.iter_mut().zip(&ftwfs) {
            *o = a + lambda * *o;
        }
    };
    jacobi_pcg(apply, None, &rhs, vec![0.0; f.n_source()], tol, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid, Point2D, ScalarField2D};
    use crate::lens::SIS;
    use crate::rng::Lcg;

    #[test]
    fn regularizer_penalizes_roughness_not_flatness() {
        let mut rng = Lcg::new(0x9);
        let sites: Vec<Point2D> =
            (0..60).map(|_| Point2D::new(2.0 * rng.f() - 1.0, 2.0 * rng.f() - 1.0)).collect();
        let mesh = MeshGrid::zeros(&sites, BBox::from_lims(-1.0, 1.0, -1.0, 1.0)).unwrap();
        let reg = Regularizer::from_mesh(&mesh);
        let mut out = vec![0.0; reg.len()];
        // A constant source has zero graph-Laplacian everywhere.
        reg.apply(&vec![3.0; reg.len()], &mut out);
        assert!(out.iter().all(|v| v.abs() < 1e-12));
        // A non-constant source does not.
        let s: Vec<f64> = (0..reg.len()).map(|i| i as f64).collect();
        reg.apply(&s, &mut out);
        assert!(out.iter().any(|v| v.abs() > 1e-9));
    }

    #[test]
    fn inversion_recovers_a_known_source() {
        // Simulate d = F s_true for a fixed lens/source, then invert with light
        // regularization: the recovered source matches the truth.
        let lens = SIS::new(Point2D::origin(), 0.5);
        let mut rng = Lcg::new(0x1234);
        let bbox = BBox::from_lims(-1.5, 1.5, -1.5, 1.5);
        let sites: Vec<Point2D> =
            (0..120).map(|_| Point2D::new(3.0 * rng.f() - 1.5, 3.0 * rng.f() - 1.5)).collect();
        let source = MeshGrid::zeros(&sites, bbox).unwrap();
        let image = PixelGrid::constant(Grid2D::square(24, 2.0).unwrap(), 0.0);
        let f = LensingOperator::build(&lens, &image, &source);

        let s_true: Vec<f64> = (0..source.len()).map(|i| 1.0 + 0.5 * (i as f64 * 0.3).sin()).collect();
        let data = f.apply(&s_true);

        let reg = Regularizer::from_mesh(&source);
        let noise = Noise::uniform(1.0);
        let mask = Mask::all(image.len());
        let (s, info) = invert_source(&f, &data, &noise, &mask, &reg, 1e-6, 1e-12, 500);
        assert!(info.converged, "CG did not converge: {info:?}");
        // Only source cells actually hit by the image rays are constrained; check those.
        let mut checked = 0;
        for j in 0..source.len() {
            let column_hit = (0..image.len()).any(|i| {
                source.sample_stencil(lens_ray(&lens, &image, i)).iter().any(|(c, _)| *c == j)
            });
            if column_hit {
                assert!((s[j] - s_true[j]).abs() < 1e-3, "cell {j}: {} vs {}", s[j], s_true[j]);
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    fn lens_ray(lens: &SIS, image: &PixelGrid, i: usize) -> Point2D {
        use crate::lens::LensModel;
        lens.ray_trace(ScalarField2D::center(image, i))
    }
}
