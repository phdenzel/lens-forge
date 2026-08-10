// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The linear map from cell convergences to the lensing field at the image positions.
//!
//! [`KappaBasis`] is the single representation-specific seam of the inverse layers: it
//! exposes the per-unit-κ kernel (the square-pixel or Voronoi-polygon log integral) and
//! cell adjacency, so [`PixelGrid`] and [`MeshGrid`] feed the same polytope machinery.
//! [`InfluenceMatrix`] evaluates that kernel once per (cell, image position) pair and stores
//! the ψ and deflection coefficients — the rows every linear constraint contracts against.

use crate::grid::{
    CellAdjacency, MeshGrid, PixelGrid, Point2D, ScalarField2D, polygon_log_kernel,
    square_log_kernel,
};

/// A convergence basis whose field is linear in the per-cell κ: it reports each cell's
/// per-unit-κ jet at a point, plus its neighbouring cells (via [`CellAdjacency`]).
///
/// The jet is `[ψ, ψ_x, ψ_y, ψ_xx, ψ_xy, ψ_yy]` per unit κ — the same array the free-form
/// lens kernels return, so a coefficient row is `Σ_c κ_c · cell_jet_c`.
pub trait KappaBasis: ScalarField2D + CellAdjacency {
    /// Per-unit-κ jet `[ψ, ψ_x, ψ_y, ψ_xx, ψ_xy, ψ_yy]` of cell `i` at `theta`.
    ///
    /// # Arguments
    /// * `i` - cell index.
    /// * `theta` - evaluation point.
    fn cell_jet(&self, i: usize, theta: Point2D) -> [f64; 6];
}

impl KappaBasis for MeshGrid {
    fn cell_jet(&self, i: usize, theta: Point2D) -> [f64; 6] {
        polygon_log_kernel(self.cell_polygon(i), theta)
    }
}

impl KappaBasis for PixelGrid {
    fn cell_jet(&self, i: usize, theta: Point2D) -> [f64; 6] {
        let c = ScalarField2D::center(self, i);
        square_log_kernel(c, 0.5 * self.dx(), 0.5 * self.dy(), theta)
    }
}

/// The precomputed ψ and deflection coefficients of a [`KappaBasis`] at a fixed set of
/// image positions: row `q` maps cell convergences to ψ, α_x, α_y at position `q`.
pub struct InfluenceMatrix {
    n_cells: usize,
    points: Vec<Point2D>,
    psi: Vec<f64>,
    alpha_x: Vec<f64>,
    alpha_y: Vec<f64>,
}

impl InfluenceMatrix {
    /// Evaluate `basis` at every point of `points`, storing the ψ and deflection
    /// (`α = ∇ψ`) coefficients per cell.
    ///
    /// # Arguments
    /// * `basis` - the convergence basis (pixel or mesh).
    /// * `points` - the image positions to precompute at.
    pub fn build<B: KappaBasis>(basis: &B, points: &[Point2D]) -> Self {
        let n_cells = basis.len();
        let mut psi = Vec::with_capacity(points.len() * n_cells);
        let mut alpha_x = Vec::with_capacity(points.len() * n_cells);
        let mut alpha_y = Vec::with_capacity(points.len() * n_cells);
        for &p in points {
            for c in 0..n_cells {
                let j = basis.cell_jet(c, p);
                psi.push(j[0]);
                alpha_x.push(j[1]);
                alpha_y.push(j[2]);
            }
        }
        Self { n_cells, points: points.to_vec(), psi, alpha_x, alpha_y }
    }

    /// Number of cells (columns of each coefficient row).
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// Number of image positions (rows).
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// The image position of row `q`.
    ///
    /// # Arguments
    /// * `q` - row (image-position) index.
    pub fn point(&self, q: usize) -> Point2D {
        self.points[q]
    }

    /// Per-cell ψ coefficients at position `q`: `ψ(θ_q) = Σ_c κ_c · row[c]`.
    ///
    /// # Arguments
    /// * `q` - row index.
    pub fn psi_row(&self, q: usize) -> &[f64] {
        &self.psi[q * self.n_cells..(q + 1) * self.n_cells]
    }

    /// Per-cell deflection coefficients `(α_x, α_y)` at position `q`.
    ///
    /// # Arguments
    /// * `q` - row index.
    pub fn alpha_rows(&self, q: usize) -> (&[f64], &[f64]) {
        let r = q * self.n_cells..(q + 1) * self.n_cells;
        (&self.alpha_x[r.clone()], &self.alpha_y[r])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastsum::Backend;
    use crate::grid::{BBox, Grid2D, MeshGrid};
    use crate::lens::freeform::{FieldMethod, MeshLens, PixelLens};
    use crate::lens::{FreeFormLens, LensModel};
    use crate::rng::Lcg;

    fn dot(row: &[f64], kappa: &[f64]) -> f64 {
        row.iter().zip(kappa).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn pixel_influence_matches_the_analytic_lens_field() {
        // The exact per-cell square kernel summed against κ must equal a PixelLens built
        // with the Analytic(Direct) method (which sums the same kernel over all cells).
        let grid = Grid2D::square(8, 2.0).unwrap();
        let kappa = PixelGrid::from_func(grid, |x, y| 0.4 + 0.05 * x - 0.03 * y);
        let kv: Vec<f64> = (0..kappa.len()).map(|i| kappa.value_at(i)).collect();
        let probes = [Point2D::new(1.3, -0.7), Point2D::new(-0.4, 0.9)];
        let infl = InfluenceMatrix::build(&kappa, &probes);
        assert_eq!(infl.n_cells(), kv.len());
        assert_eq!(infl.n_points(), probes.len());
        assert_eq!(infl.point(0), probes[0]);
        let lens = PixelLens::build(kappa, FieldMethod::analytic(Backend::Direct)).unwrap();
        for (q, &p) in probes.iter().enumerate() {
            let (ax, ay) = infl.alpha_rows(q);
            let a = lens.deflection(p);
            assert!((dot(ax, &kv) - a.x).abs() < 1e-9, "alpha_x at {p:?}");
            assert!((dot(ay, &kv) - a.y).abs() < 1e-9, "alpha_y at {p:?}");
            assert!((dot(infl.psi_row(q), &kv) - lens.potential(p)).abs() < 1e-9, "psi at {p:?}");
        }
    }

    #[test]
    fn mesh_influence_matches_the_analytic_lens_field() {
        let mut rng = Lcg::new(0xB17);
        let bbox = BBox::from_lims(-2.0, 2.0, -2.0, 2.0);
        let sites: Vec<Point2D> =
            (0..120).map(|_| Point2D::new(3.6 * rng.f() - 1.8, 3.6 * rng.f() - 1.8)).collect();
        let kappa = MeshGrid::from_func(&sites, bbox, |x, _y| 0.5 + 0.1 * x).unwrap();
        let kv: Vec<f64> = (0..kappa.len()).map(|i| kappa.value_at(i)).collect();
        let probes = [Point2D::new(1.1, 0.3)];
        let infl = InfluenceMatrix::build(&kappa, &probes);
        assert!(!kappa.cell_neighbors(0).is_empty()); // Delaunay adjacency on the mesh
        let lens = MeshLens::build(kappa, FieldMethod::analytic(Backend::Direct)).unwrap();
        let (ax, ay) = infl.alpha_rows(0);
        let a = lens.deflection(probes[0]);
        assert!((dot(ax, &kv) - a.x).abs() < 1e-9);
        assert!((dot(ay, &kv) - a.y).abs() < 1e-9);
        assert!((dot(infl.psi_row(0), &kv) - lens.potential(probes[0])).abs() < 1e-9);
    }

    #[test]
    fn pixel_neighbors_are_the_four_edge_cells_interior() {
        let grid = Grid2D::square(5, 2.0).unwrap();
        let kappa = PixelGrid::constant(grid, 1.0);
        // The centre cell of a 5×5 grid has all four neighbours.
        let mid = kappa.cell_of(&Point2D::new(0.0, 0.0)).unwrap();
        assert_eq!(kappa.cell_neighbors(mid).len(), 4);
    }
}
