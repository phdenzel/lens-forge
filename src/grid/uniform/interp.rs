// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Grid interpolation kernels

use crate::grid::{Grid2D, Point2D};
use ndarray::Array2;

/* ================ Interpolators ========================================== */
/// Bilinear interpolation kernel.
///
/// Precomputed indices and weights of the four grid cells surrounding a query
/// point, so the interpolated value is the weighted sum
/// $f = w_{00} f_{i_0 j_0} + w_{10} f_{i_1 j_0} + w_{01} f_{i_0 j_1} + w_{11} f_{i_1 j_1}$.
/// The weights sum to 1. Build one with [`Grid2D::bilinear_interp`] and evaluate
/// it against a data array with [`BilinearKernel::apply`].
#[derive(Debug, Clone, Copy)]
pub struct BilinearKernel {
    /// Column index of the left cell edge.
    pub i0: usize,
    /// Row index of the bottom cell edge.
    pub j0: usize,
    /// Column index of the right cell edge (`i0 + 1`, clamped to the grid).
    pub i1: usize,
    /// Row index of the top cell edge (`j0 + 1`, clamped to the grid).
    pub j1: usize,
    /// Weight of the bottom-left corner `(i0, j0)`.
    pub w00: f64,
    /// Weight of the bottom-right corner `(i1, j0)`.
    pub w10: f64,
    /// Weight of the top-left corner `(i0, j1)`.
    pub w01: f64,
    /// Weight of the top-right corner `(i1, j1)`.
    pub w11: f64,
}

impl BilinearKernel {
    /// Apply kernel to a 2D array for bilinear interpolation
    ///
    /// # Arguments
    /// * `data` - 2D data array to be interpolated
    #[inline]
    pub fn apply(&self, data: &Array2<f64>) -> f64 {
        self.w00 * data[[self.j0, self.i0]]
            + self.w10 * data[[self.j0, self.i1]]
            + self.w01 * data[[self.j1, self.i0]]
            + self.w11 * data[[self.j1, self.i1]]
    }
}

/* ================ Display ================================================ */
impl std::fmt::Display for BilinearKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BilinearKernel(({}, {})-({}, {}), k:[{:.3}, {:.3}, {:.3}, {:.3}])",
            self.i0, self.j0, self.i1, self.j1, self.w00, self.w01, self.w10, self.w11,
        )
    }
}

impl Grid2D {
    /// Compute bilinear interpolation for a point
    #[inline]
    pub fn bilinear_interp(&self, p: &Point2D) -> Option<BilinearKernel> {
        let (fx, fy) = self.relative_to(p);

        let i0 = fx.floor() as i64;
        let j0 = fy.floor() as i64;

        if i0 < -1 || i0 >= self.nx as i64 || j0 < -1 || j0 >= self.ny as i64 {
            return None;
        }

        let tx = fx - i0 as f64;
        let ty = fy - j0 as f64;

        let i0 = i0.max(0) as usize;
        let j0 = j0.max(0) as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);

        Some(BilinearKernel {
            i0,
            j0,
            i1,
            j1,
            w00: (1.0 - tx) * (1.0 - ty),
            w10: tx * (1.0 - ty),
            w01: (1.0 - tx) * ty,
            w11: tx * ty,
        })
    }
}

/* ================ Unit tests ============================================= */
#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_bilinear_interp_at_center() {
        let grid = Grid2D::new(4, 4, (0.0, 4.0), (0.0, 4.0)).unwrap();

        // Point at pixel center
        let w = grid.bilinear_interp(&Point2D::new(0.5, 0.5)).unwrap();
        assert!(approx_eq(w.w00, 1.0));
        assert!(approx_eq(w.w10 + w.w01 + w.w11, 0.0));
        println!("{}", w);
    }

    #[test]
    fn test_bilinear_interp_between() {
        let grid = Grid2D::new(4, 4, (-2.0, 2.0), (-2.0, 2.0)).unwrap();

        // Point between four pixels
        let w = grid.bilinear_interp(&Point2D::new(1.0, 1.0)).unwrap();
        assert!(approx_eq(w.w00, 0.25));
        assert!(approx_eq(w.w10, 0.25));
        assert!(approx_eq(w.w01, 0.25));
        assert!(approx_eq(w.w11, 0.25));
        println!("{}", w);
    }

    #[test]
    fn test_bilinear_interpolate() {
        let grid = Grid2D::new(4, 4, (0.0, 4.0), (0.0, 4.0)).unwrap();
        let data = grid.map_centers(|x, _| x); // f(x,y) = x

        let p = Point2D::new(2.0, 2.0);
        let w = grid.bilinear_interp(&p).unwrap();
        println!("{}", w);
        let interp = w.apply(&data);

        // Should be close to 2.0
        println!("{}", interp);
        assert!((interp - 2.0).abs() < 0.5);
    }
}
