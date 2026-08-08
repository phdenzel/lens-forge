// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Finite-volume differential operators on a [`MeshGrid`].
//!
//! * [`MeshGrid::gradient`] / [`MeshGrid::gradient_field`] — a per-cell
//!   **least-squares** gradient over the Voronoi 1-ring, exact for linear fields
//!   and degrading gracefully at short-ring boundary cells.
//! * [`MeshGrid::laplacian`] / [`MeshGrid::laplacian_field`] — the
//!   **two-point-flux (TPFA)** Laplacian
//!   $(\nabla^2 v)_i = \frac{1}{A_i} \sum_j \frac{\ell_{ij}}{d_{ij}}(v_j - v_i)$,
//!   summing the flux across each shared facet (see
//!   [`FaceGeometry`](super::faces)); box facets carry no flux.
//!
//! The area-free stiffness sum is the symmetric operator the Poisson solver
//! applies; see [`apply_stiffness`].

use super::faces::{FaceGeometry, directed_facet_lengths};
use super::meshgrid::MeshGrid;
use crate::grid::{FieldArray1D, Point2D};

/// Apply the symmetric mesh stiffness operator `K` to `v`, writing `out`.
///
/// `outᵢ = Σ_j (ℓ_ij / d_ij)(v_j − vᵢ)` over the interior facets of cell `i`.
/// This is the area-weighted Laplacian (`Aᵢ · (∇²v)ᵢ`); it is symmetric
/// negative-semidefinite, the form the conjugate-gradient Poisson solve needs.
///
/// # Arguments
/// * `mesh` - the tessellation supplying sites and adjacency.
/// * `faces` - precomputed shared-facet lengths for `mesh`.
/// * `v` - input values, one per cell.
/// * `out` - output buffer, one per cell (overwritten).
pub(super) fn apply_stiffness(mesh: &MeshGrid, faces: &FaceGeometry, v: &[f64], out: &mut [f64]) {
    for i in 0..v.len() {
        let site = mesh.site(i);
        let mut acc = 0.0;
        for &(j, len) in faces.faces_of(i) {
            let d = site.distance(&mesh.site(j));
            if d > 0.0 {
                acc += (len / d) * (v[j] - v[i]);
            }
        }
        out[i] = acc;
    }
}

impl MeshGrid {
    /// Gradient `(∂v/∂x, ∂v/∂y)` at cell `i`, fitted by weighted least squares to
    /// its Voronoi 1-ring.
    ///
    /// Solves the `2×2` normal equations of
    /// `min_g Σ_j w_j [ (v_j − vᵢ) − g·(s_j − sᵢ) ]²` with inverse-distance
    /// weights `w_j = 1/|s_j − sᵢ|²`. Exact for linear fields on any mesh; a cell
    /// whose 1-ring is empty or collinear (a singular system) yields a zero
    /// gradient.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    pub fn gradient(&self, i: usize) -> Point2D {
        let site = self.site(i);
        let vi = self.value(i);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        let (mut bx, mut by) = (0.0, 0.0);
        for &j in self.neighbors(i) {
            let d = self.site(j) - site;
            let d2 = d.x * d.x + d.y * d.y;
            if d2 <= 0.0 {
                continue;
            }
            let w = 1.0 / d2;
            let dv = self.value(j) - vi;
            sxx += w * d.x * d.x;
            sxy += w * d.x * d.y;
            syy += w * d.y * d.y;
            bx += w * d.x * dv;
            by += w * d.y * dv;
        }
        let det = sxx * syy - sxy * sxy;
        if det.abs() > 1e-300 {
            Point2D::new((syy * bx - sxy * by) / det, (sxx * by - sxy * bx) / det)
        } else {
            Point2D::new(0.0, 0.0)
        }
    }

    /// The whole-field gradient `(∂/∂x, ∂/∂y)` as two mesh grids sharing this
    /// geometry — [`gradient`](Self::gradient) evaluated at every cell.
    pub fn gradient_field(&self) -> (MeshGrid, MeshGrid) {
        let n = self.len();
        let mut gx = FieldArray1D::zeros(n);
        let mut gy = FieldArray1D::zeros(n);
        for i in 0..n {
            let g = self.gradient(i);
            gx[i] = g.x;
            gy[i] = g.y;
        }
        (
            self.with_values(gx).expect("gradient shares this geometry"),
            self.with_values(gy).expect("gradient shares this geometry"),
        )
    }

    /// The whole-field gradient magnitude `|∇v|`, one value per cell.
    pub fn gradient_magnitude(&self) -> MeshGrid {
        let (gx, gy) = self.gradient_field();
        let mag: FieldArray1D = gx
            .as_array1()
            .iter()
            .zip(gy.as_array1().iter())
            .map(|(&x, &y)| Point2D::new(x, y).norm())
            .collect();
        self.with_values(mag)
            .expect("gradient shares this geometry")
    }

    /// The finite-volume (TPFA) Laplacian `∇²v` at cell `i`.
    ///
    /// `(∇²v)ᵢ = (1/Aᵢ) Σ_j (ℓ_ij / d_ij)(v_j − vᵢ)` over the interior facets of
    /// cell `i`, with `Aᵢ` the cell area, `ℓ_ij` the shared-facet length and
    /// `d_ij` the site distance. Facets on the domain box carry no flux, and a
    /// zero-area cell yields `0`.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    pub fn laplacian(&self, i: usize) -> f64 {
        let area = self.area(i);
        if area <= 0.0 {
            return 0.0;
        }
        let nb = self.neighbors(i);
        let mut lengths = vec![0.0f64; nb.len()];
        directed_facet_lengths(self.mesh(), i, nb, &mut lengths);
        let site = self.site(i);
        let vi = self.value(i);
        let mut acc = 0.0;
        for (k, &j) in nb.iter().enumerate() {
            let d = site.distance(&self.site(j));
            if d > 0.0 {
                acc += (lengths[k] / d) * (self.value(j) - vi);
            }
        }
        acc / area
    }

    /// The whole-field Laplacian `∇²v`, one value per cell.
    ///
    /// Uses the symmetrized `FaceGeometry` (matching the Poisson
    /// solver's stiffness operator); agrees with the pointwise
    /// [`laplacian`](Self::laplacian) to rounding on interior cells.
    pub fn laplacian_field(&self) -> MeshGrid {
        let faces = FaceGeometry::build(self.mesh());
        let n = self.len();
        let mut out = vec![0.0f64; n];
        apply_stiffness(self, &faces, self.values(), &mut out);
        for (i, o) in out.iter_mut().enumerate() {
            let area = self.area(i);
            if area > 0.0 {
                *o /= area;
            } else {
                *o = 0.0;
            }
        }
        self.with_values(FieldArray1D::from_vec(out))
            .expect("laplacian shares this geometry")
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

    /// A jittered lattice over the unit box, valued by `f`.
    fn mesh_from<F: Fn(f64, f64) -> f64>(f: F) -> MeshGrid {
        let mut rng = Lcg::new(0x9C7AD);
        let mut sites = Vec::new();
        for gy in 0..14 {
            for gx in 0..14 {
                let x = 0.5 / 14.0 + gx as f64 / 14.0 + 0.02 * (rng.f() - 0.5);
                let y = 0.5 / 14.0 + gy as f64 / 14.0 + 0.02 * (rng.f() - 0.5);
                sites.push(p(x, y));
            }
        }
        MeshGrid::from_func(&sites, unit(), f).unwrap()
    }

    /// Whether cell `i` is an interior cell (no facet on the domain box), so its
    /// operators are unbiased by the zero-flux boundary.
    fn is_interior(m: &MeshGrid, i: usize) -> bool {
        let (x0, x1) = m.extent_x();
        let (y0, y1) = m.extent_y();
        m.cell_polygon(i)
            .iter()
            .all(|v| v.x > x0 + 1e-3 && v.x < x1 - 1e-3 && v.y > y0 + 1e-3 && v.y < y1 - 1e-3)
    }

    #[test]
    fn gradient_of_a_linear_field_is_constant() {
        // v = 2x - 5y + 7  ->  ∇v = (2, -5) everywhere, exactly.
        let m = mesh_from(|x, y| 2.0 * x - 5.0 * y + 7.0);
        let (gx, gy) = m.gradient_field();
        for i in 0..m.len() {
            assert!((gx.value(i) - 2.0).abs() < 1e-9, "gx {}", gx.value(i));
            assert!((gy.value(i) + 5.0).abs() < 1e-9, "gy {}", gy.value(i));
        }
        let mag = m.gradient_magnitude();
        assert!((mag.value(0) - (29.0f64).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn single_cell_mesh_has_zero_gradient_and_laplacian() {
        // One site, no 1-ring: the least-squares system is singular (zero
        // gradient) and there are no interior facets (zero Laplacian).
        let m = MeshGrid::from_func(&[p(0.5, 0.5)], unit(), |x, y| x + y).unwrap();
        assert_eq!(m.gradient(0), p(0.0, 0.0));
        assert_eq!(m.laplacian(0), 0.0);
        let (gx, gy) = m.gradient_field();
        assert_eq!((gx.value(0), gy.value(0)), (0.0, 0.0));
    }

    #[test]
    fn pointwise_gradient_matches_the_field() {
        // The per-cell gradient must equal the corresponding cell of the
        // whole-field gradient.
        let m = mesh_from(|x, y| x * x - 2.0 * y);
        let (gx, gy) = m.gradient_field();
        for i in 0..m.len() {
            let g = m.gradient(i);
            assert!((g.x - gx.value(i)).abs() < 1e-12 && (g.y - gy.value(i)).abs() < 1e-12);
        }
    }

    #[test]
    fn laplacian_of_a_linear_field_vanishes_in_the_interior() {
        let m = mesh_from(|x, y| 3.0 * x + 4.0 * y - 1.0);
        let lap = m.laplacian_field();
        for i in 0..m.len() {
            if is_interior(&m, i) {
                assert!(lap.value(i).abs() < 1e-6, "cell {i}: {}", lap.value(i));
            }
        }
    }

    #[test]
    fn pointwise_laplacian_matches_the_field_in_the_interior() {
        // On interior cells the directed (pointwise) and symmetrized (field)
        // facet lengths coincide, so the two Laplacians agree to rounding.
        let m = mesh_from(|x, y| x * x + y * y);
        let lap = m.laplacian_field();
        for i in 0..m.len() {
            if is_interior(&m, i) {
                assert!((m.laplacian(i) - lap.value(i)).abs() < 1e-9, "cell {i}");
            }
        }
    }

    #[test]
    fn laplacian_of_a_quadratic_is_the_trace_of_the_hessian() {
        // v = x² + y²  ->  ∇²v = 4 in the interior (consistency of the TPFA
        // stencil on a near-uniform mesh).
        let m = mesh_from(|x, y| x * x + y * y);
        let lap = m.laplacian_field();
        let interior: Vec<f64> = (0..m.len())
            .filter(|&i| is_interior(&m, i))
            .map(|i| lap.value(i))
            .collect();
        assert!(!interior.is_empty());
        let mean = interior.iter().sum::<f64>() / interior.len() as f64;
        assert!((mean - 4.0).abs() < 0.2, "mean interior laplacian {mean}");
    }

    #[test]
    fn stiffness_is_symmetric() {
        // K must satisfy aᵀ K b == bᵀ K a for the CG solver to be valid.
        let m = mesh_from(|x, y| x + y);
        let faces = FaceGeometry::build(m.mesh());
        let n = m.len();
        let mut rng = Lcg::new(0x5717);
        let a: Vec<f64> = (0..n).map(|_| rng.f()).collect();
        let b: Vec<f64> = (0..n).map(|_| rng.f()).collect();
        let mut ka = vec![0.0; n];
        let mut kb = vec![0.0; n];
        apply_stiffness(&m, &faces, &a, &mut ka);
        apply_stiffness(&m, &faces, &b, &mut kb);
        let akb: f64 = a.iter().zip(&kb).map(|(x, y)| x * y).sum();
        let bka: f64 = b.iter().zip(&ka).map(|(x, y)| x * y).sum();
        assert!((akb - bka).abs() < 1e-9, "{akb} vs {bka}");
    }
}
