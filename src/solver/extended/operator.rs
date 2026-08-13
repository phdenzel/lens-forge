// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The sparse lensing operator F: source coefficients → image, with its transpose.
//!
//! With the source held on a field of unknown cell values s, the lensed image is
//! linear in s: dᵢ = S(βᵢ) = Σⱼ wᵢⱼ sⱼ, so F is the sparse matrix whose row i is the
//! source-plane sample stencil at βᵢ = θᵢ − α(θᵢ). It is stored in compressed sparse
//! row format with an explicit transpose, so the semi-linear source inversion (normal
//! equations FᵀC⁻¹F) can build on it.

use crate::grid::FieldOps;
use crate::lens::LensModel;

/// The sparse lensing operator F mapping source-cell coefficients to image-cell
/// brightness (d = F s), in compressed sparse row form.
pub struct LensingOperator {
    /// Number of image cells (rows of F).
    n_image: usize,
    /// Number of source cells (columns of F).
    n_source: usize,
    /// CSR row offsets into `col_idx`/`values` (length `n_image + 1`).
    row_ptr: Vec<usize>,
    /// Source-cell column index of each stored nonzero.
    col_idx: Vec<usize>,
    /// Stencil weight of each stored nonzero.
    values: Vec<f64>,
}

impl LensingOperator {
    /// Build F for `lens`: ray-trace each image cell to the source plane and take
    /// the source's sample stencil there as that row.
    ///
    /// # Arguments
    /// * `lens` - the lens model.
    /// * `image` - the image-plane sampling geometry (rows; only geometry is used).
    /// * `source` - the source-plane basis (columns; only geometry is used).
    pub fn build<I: FieldOps, S: FieldOps>(lens: &dyn LensModel, image: &I, source: &S) -> Self {
        let n_image = image.len();
        let mut row_ptr = Vec::with_capacity(n_image + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for i in 0..n_image {
            let beta = lens.ray_trace(image.center(i));
            for (col, w) in source.sample_stencil(beta) {
                col_idx.push(col);
                values.push(w);
            }
            row_ptr.push(col_idx.len());
        }
        Self { n_image, n_source: source.len(), row_ptr, col_idx, values }
    }

    /// Apply F: the lensed image d = F s of a source with cell values `s`.
    ///
    /// # Arguments
    /// * `s` - source-cell coefficients (length [`n_source`](Self::n_source)).
    pub fn apply(&self, s: &[f64]) -> Vec<f64> {
        assert_eq!(s.len(), self.n_source, "source vector length mismatch");
        (0..self.n_image)
            .map(|i| {
                (self.row_ptr[i]..self.row_ptr[i + 1])
                    .map(|k| self.values[k] * s[self.col_idx[k]])
                    .sum()
            })
            .collect()
    }

    /// Apply the transpose Fᵀ: back-project an image `d` onto the source basis.
    ///
    /// # Arguments
    /// * `d` - image-cell values (length [`n_image`](Self::n_image)).
    pub fn apply_transpose(&self, d: &[f64]) -> Vec<f64> {
        assert_eq!(d.len(), self.n_image, "image vector length mismatch");
        let mut out = vec![0.0; self.n_source];
        for i in 0..self.n_image {
            let di = d[i];
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                out[self.col_idx[k]] += self.values[k] * di;
            }
        }
        out
    }

    /// Number of image cells (rows of F).
    pub fn n_image(&self) -> usize {
        self.n_image
    }

    /// Number of source cells (columns of F).
    pub fn n_source(&self) -> usize {
        self.n_source
    }

    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Grid2D, MeshGrid, PixelGrid, Point2D, ScalarField2D};
    use crate::lens::SIS;
    use crate::rng::Lcg;
    use crate::solver::extended::imaging::ImageSimulation;
    use crate::source::SourceField;

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn f_matches_direct_imaging_of_a_pixel_source() {
        // F·s must equal directly imaging the same source field: both are sample(β_i).
        let lens = SIS::new(Point2D::origin(), 0.5);
        let source = PixelGrid::from_func(Grid2D::square(24, 2.0).unwrap(), |x, y| {
            (0.5 * x).sin() + 0.3 * y + 1.0
        });
        let image = PixelGrid::constant(Grid2D::square(20, 1.5).unwrap(), 0.0);
        let s: Vec<f64> = (0..source.len()).map(|j| source.value_at(j)).collect();
        let f = LensingOperator::build(&lens, &image, &source);
        let d = f.apply(&s);
        let img = ImageSimulation::Direct.image(&lens, &SourceField::new(source), &image);
        for i in 0..image.len() {
            assert!((d[i] - img.value_at(i)).abs() < 1e-12, "row {i}: {} vs {}", d[i], img.value_at(i));
        }
    }

    #[test]
    fn f_matches_direct_imaging_of_a_mesh_source() {
        // Same equivalence with a nearest-cell mesh source (single-cell stencils).
        let lens = SIS::new(Point2D::origin(), 0.5);
        let mut rng = Lcg::new(0x5A3);
        let bbox = BBox::from_lims(-2.0, 2.0, -2.0, 2.0);
        let sites: Vec<Point2D> =
            (0..200).map(|_| Point2D::new(3.6 * rng.f() - 1.8, 3.6 * rng.f() - 1.8)).collect();
        let source = MeshGrid::from_func(&sites, bbox, |x, y| x - 0.5 * y + 2.0).unwrap();
        let image = PixelGrid::constant(Grid2D::square(16, 1.5).unwrap(), 0.0);
        let s: Vec<f64> = (0..source.len()).map(|j| source.value_at(j)).collect();
        let f = LensingOperator::build(&lens, &image, &source);
        let d = f.apply(&s);
        let img = ImageSimulation::Direct.image(&lens, &SourceField::new(source), &image);
        for i in 0..image.len() {
            assert!((d[i] - img.value_at(i)).abs() < 1e-12);
        }
    }

    #[test]
    fn transpose_is_the_adjoint() {
        // ⟨F s, d⟩ == ⟨s, Fᵀ d⟩ for random vectors.
        let lens = SIS::new(Point2D::origin(), 0.6);
        let source = PixelGrid::constant(Grid2D::square(16, 2.0).unwrap(), 0.0);
        let image = PixelGrid::constant(Grid2D::square(12, 1.5).unwrap(), 0.0);
        let f = LensingOperator::build(&lens, &image, &source);
        let mut rng = Lcg::new(0xF00D);
        let s: Vec<f64> = (0..f.n_source()).map(|_| rng.f() - 0.5).collect();
        let d: Vec<f64> = (0..f.n_image()).map(|_| rng.f() - 0.5).collect();
        let lhs = dot(&f.apply(&s), &d);
        let rhs = dot(&s, &f.apply_transpose(&d));
        assert!((lhs - rhs).abs() < 1e-9 * (1.0 + lhs.abs()), "{lhs} vs {rhs}");
    }

    #[test]
    fn dimensions_and_sparsity_match_the_fields() {
        let lens = SIS::new(Point2D::origin(), 0.5);
        let source = PixelGrid::constant(Grid2D::square(10, 2.0).unwrap(), 0.0);
        let image = PixelGrid::constant(Grid2D::square(8, 1.5).unwrap(), 0.0);
        let f = LensingOperator::build(&lens, &image, &source);
        assert_eq!(f.n_image(), 64);
        assert_eq!(f.n_source(), 100);
        // Each row has at most four (bilinear) entries.
        assert!(f.nnz() > 0 && f.nnz() <= 4 * f.n_image());
    }
}
