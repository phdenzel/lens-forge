// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! [`FieldOps`] for [`PixelGrid`]: adapt its inherent operators to the shared,
//! representation-agnostic contract.
//!
//! The inherent operators return raw [`FieldArray2D`](crate::grid::FieldArray2D) data with selectable
//! stencils and boundary conditions; here they are called with their defaults and
//! rewrapped as [`PixelGrid`]s, so generic `FieldOps` code sees the same
//! `Self`-valued results as the mesh. Each body delegates to the inherent method
//! of the same name (inherent methods win method resolution over trait methods).

use super::pixelgrid::PixelGrid;
use crate::grid::{ContourSegment, FieldOps, Kernel, Point2D, PoissonConfig};
use crate::{Result, SolveInfo};

impl FieldOps for PixelGrid {
    fn interpolate_or(&self, p: Point2D, default: f64) -> f64 {
        PixelGrid::interpolate_or(self, p, default)
    }

    fn par_interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        PixelGrid::par_interpolate_batch(self, points)
    }

    fn gradient_field(&self) -> Result<(Self, Self)> {
        // Inherent `gradient_field` takes a stencil argument; `None` is the default.
        let (gx, gy) = PixelGrid::gradient_field(self, None)?;
        Ok((self.with_values(gx)?, self.with_values(gy)?))
    }

    fn gradient_magnitude(&self) -> Result<Self> {
        // Inherent takes a stencil argument; `None` selects the default.
        self.with_values(PixelGrid::gradient_magnitude(self, None)?)
    }

    fn laplacian_field(&self) -> Result<Self> {
        self.with_values(PixelGrid::laplacian_field(self, None, None)?)
    }

    fn map_centers<G: Fn(Point2D) -> f64>(&self, f: G) -> Self {
        PixelGrid::from_func(self.grid, |x, y| f(Point2D::new(x, y)))
    }

    fn map_centers_batch<G: FnOnce(&[Point2D]) -> Vec<f64>>(&self, f: G) -> Self {
        // Cell centres in row-major (flat) order, matching `from_vec`/`value_at`.
        let centers: Vec<Point2D> = (0..self.grid.n_pixels())
            .map(|i| {
                let (ix, jy) = self.grid.unflatten_index(i);
                self.grid.pixel_center(ix, jy)
            })
            .collect();
        PixelGrid::from_vec(self.grid, f(&centers)).expect("one value per cell shares this geometry")
    }

    fn poisson(&self, cfg: Option<PoissonConfig>) -> Result<(Self, SolveInfo)> {
        let (psi, info) = self.poisson_cg(cfg)?;
        Ok((self.with_values(psi)?, info))
    }

    fn contour(&self, level: f64) -> Vec<ContourSegment> {
        PixelGrid::contour(self, level)
    }

    fn convolve(&self, kernel: &Kernel) -> Result<Self> {
        PixelGrid::convolve(self, kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, ScalarField2D};

    #[test]
    fn map_centers_batch_matches_map_centers() {
        let pg = PixelGrid::constant(Grid2D::square(8, 2.0).unwrap(), 0.0);
        let g = |p: Point2D| 0.5 * p.x - 1.3 * p.y + 0.2;
        let a = pg.map_centers(g);
        let b = pg.map_centers_batch(|cs| cs.iter().map(|&p| g(p)).collect());
        for i in 0..ScalarField2D::len(&a) {
            assert_eq!(a.value_at(i), b.value_at(i));
        }
    }
}
