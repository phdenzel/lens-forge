// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! [`FieldOps`]: the representation-agnostic operator surface shared by
//! [`PixelGrid`](crate::grid::PixelGrid) and [`MeshGrid`](crate::grid::MeshGrid).
//!
//! [`ScalarField2D`] is the object-safe *data* trait (cell access, sampling,
//! aggregates); [`FieldOps`] extends it with the *operator* workflow — continuous
//! interpolation, gradient, Laplacian, and the Poisson solve — so generic
//! `T: FieldOps` code runs unchanged on either representation. It is `Self`-valued
//! and thus a generic bound, not a trait object.
//!
//! The shared [`PoissonConfig`] configures both Poisson solvers, so it lives here
//! rather than in either representation's module; the mesh solver reads only its
//! `tol` and `max_iter`.

use crate::grid::{BoundaryCondition, ContourSegment, Kernel, Point2D, ScalarField2D};
use crate::{Result, SolveInfo};

/// Configuration for the iterative conjugate-gradient Poisson solvers
/// ([`PixelGrid::poisson_cg`](crate::grid::PixelGrid::poisson_cg) and
/// [`MeshGrid::poisson_cg`](crate::grid::MeshGrid::poisson_cg)).
///
/// Shared by both representations: the mesh solver reads `tol`, `max_iter` and
/// the `Dirichlet`/non-`Dirichlet` split of `bc` (pinned box vs. natural
/// zero-flux boundary), while the grid solver honours the full `bc`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonConfig {
    /// Relative residual tolerance $\lVert r \rVert / \lVert b \rVert$ at which
    /// the solve is considered converged.
    pub tol: f64,
    /// Maximum number of conjugate-gradient iterations.
    pub max_iter: usize,
    /// Boundary condition for the discrete Laplacian. Must be symmetric;
    /// `Natural` is rejected by
    /// [`PixelGrid::poisson_cg`](crate::grid::PixelGrid::poisson_cg). The mesh
    /// solver reads only the `Dirichlet`/non-`Dirichlet` split: `Dirichlet` pins
    /// $\psi = 0$ on the box, any other value keeps its natural zero-flux boundary.
    pub bc: BoundaryCondition,
}

impl Default for PoissonConfig {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_iter: 10_000,
            bc: BoundaryCondition::Periodic,
        }
    }
}

impl PoissonConfig {
    /// Set the relative residual tolerance.
    ///
    /// # Arguments
    /// * `tol` - relative residual tolerance
    #[inline]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Set the maximum iteration count.
    ///
    /// # Arguments
    /// * `max_iter` - maximum number of CG iterations
    #[inline]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Set the boundary condition.
    ///
    /// # Arguments
    /// * `bc` - boundary condition for the discrete Laplacian
    #[inline]
    pub fn with_bc(mut self, bc: BoundaryCondition) -> Self {
        self.bc = bc;
        self
    }
}

/// The continuous-interpolation and differential/Poisson operators shared by the
/// scalar-field representations.
pub trait FieldOps: ScalarField2D + Sized {
    /// Continuously interpolate the field at `p`, returning `default` outside the
    /// domain.
    ///
    /// # Arguments
    /// * `p` - the query point.
    /// * `default` - value returned when `p` lies outside the domain.
    fn interpolate_or(&self, p: Point2D, default: f64) -> f64;

    /// Interpolate the field at each of `points` in parallel (out-of-domain
    /// points yield `0.0`).
    ///
    /// # Arguments
    /// * `points` - the query points.
    fn par_interpolate_batch(&self, points: &[Point2D]) -> Vec<f64>;

    /// The gradient field `(∂/∂x, ∂/∂y)` as a pair of same-representation fields.
    fn gradient_field(&self) -> Result<(Self, Self)>;

    /// The gradient-magnitude field `|∇v|`.
    fn gradient_magnitude(&self) -> Result<Self>;

    /// The Laplacian field `∇²v`.
    fn laplacian_field(&self) -> Result<Self>;

    /// A new field of the same geometry whose value at each cell is `f` evaluated
    /// at that cell's centre (pixel centre / mesh site). The dual of
    /// [`sample`](crate::grid::ScalarField2D::sample): build a field from a
    /// function rather than read one at a point.
    ///
    /// # Arguments
    /// * `f` - evaluated at each cell centre.
    fn map_centers<G: Fn(Point2D) -> f64>(&self, f: G) -> Self;

    /// Solve `∇²ψ = f` with this field as the source `f`, returning the potential
    /// `ψ` and solver diagnostics. The boundary handling is
    /// representation-specific (see the [`grid`](crate::grid) module docs).
    ///
    /// # Arguments
    /// * `cfg` - solver configuration (defaults if `None`).
    fn poisson(&self, cfg: Option<PoissonConfig>) -> Result<(Self, SolveInfo)>;

    /// Extract the iso-contour `v = level` as unordered line segments, each
    /// representation marching its natural cells.
    ///
    /// # Arguments
    /// * `level` - the contour value.
    fn contour(&self, level: f64) -> Vec<ContourSegment>;

    /// Convolve the field with `kernel`, returning a same-representation field.
    /// Each representation uses its natural transform (FFT on the grid, diffusion
    /// on the mesh).
    ///
    /// # Arguments
    /// * `kernel` - the convolution kernel.
    fn convolve(&self, kernel: &Kernel) -> Result<Self>;

    /// Continuously interpolate the field at `p`, returning `0.0` outside the
    /// domain.
    ///
    /// # Arguments
    /// * `p` - the query point.
    fn interpolate(&self, p: Point2D) -> f64 {
        self.interpolate_or(p, 0.0)
    }

    /// Interpolate the field at each of `points` (out-of-domain points yield
    /// `0.0`).
    ///
    /// # Arguments
    /// * `points` - the query points.
    fn interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        points
            .iter()
            .map(|&p| self.interpolate_or(p, 0.0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, BoundaryCondition, MeshGrid, PixelGrid, Point2D};
    use crate::utils::rng::Lcg;

    fn plane(x: f64, y: f64) -> f64 {
        2.0 * x - 3.0 * y + 5.0
    }

    #[test]
    fn poisson_config_builders() {
        // Exercise the builder chain, including with_max_iter.
        let cfg = PoissonConfig::default()
            .with_tol(1e-6)
            .with_max_iter(42)
            .with_bc(BoundaryCondition::Dirichlet);
        assert_eq!(cfg.tol, 1e-6);
        assert_eq!(cfg.max_iter, 42);
        assert_eq!(cfg.bc, BoundaryCondition::Dirichlet);
    }

    /// A generic routine written once against `FieldOps`, run on both
    /// representations below. It touches every operator in the trait.
    fn exercise<F: FieldOps>(f: &F, sample: Point2D) -> (f64, f64, f64, f64) {
        let interp = f.interpolate(sample);
        let batch = f.interpolate_batch(&[sample]);
        assert_eq!(batch, f.par_interpolate_batch(&[sample]));
        assert_eq!(batch[0], interp);
        let (gx, gy) = f.gradient_field().unwrap();
        let gmag = f.gradient_magnitude().unwrap();
        let lap = f.laplacian_field().unwrap();
        // The trait's aggregate view (via ScalarField2D) also stays available.
        assert_eq!(gx.len(), f.len());
        assert_eq!(gmag.len(), f.len());
        let (psi, info) = f.poisson(None).unwrap();
        assert_eq!(psi.len(), f.len());
        assert!(info.residual >= 0.0); // solver ran and reported diagnostics
        // The mean level crosses the field, so the contour returns segments.
        assert!(!f.contour(f.mean()).is_empty());
        // map_centers rebuilds a same-geometry field from a point function.
        assert_eq!(f.map_centers(|p| p.x).len(), f.len());
        // Return interior-ish scalars so callers can sanity-check the values.
        let mid = f.len() / 2;
        (
            interp,
            gx.value_at(mid),
            gy.value_at(mid),
            lap.value_at(mid),
        )
    }

    #[test]
    fn pixelgrid_satisfies_fieldops() {
        // Linear field on a grid: interpolation is exact, gradient is (2, -3).
        let g = PixelGrid::from_func_sq(48, 1.0, plane);
        let (interp, gx, gy, _lap) = exercise(&g, Point2D::new(0.1, -0.2));
        assert!((interp - plane(0.1, -0.2)).abs() < 1e-9);
        assert!((gx - 2.0).abs() < 1e-6 && (gy + 3.0).abs() < 1e-6);
    }

    #[test]
    fn pixelgrid_fieldops_poisson_surfaces_solver_error() {
        // The `poisson` delegate propagates the inherent solver's rejection of the
        // asymmetric Natural boundary — exercising its error path.
        let g = PixelGrid::from_func_sq(8, 1.0, plane);
        let cfg = PoissonConfig::default().with_bc(BoundaryCondition::Natural);
        assert!(FieldOps::poisson(&g, Some(cfg)).is_err());
    }

    #[test]
    fn meshgrid_satisfies_fieldops() {
        // The same generic routine, same linear field, on an unstructured mesh.
        let mut rng = Lcg::new(0x0F5);
        let bbox = BBox::from_lims(-1.0, 1.0, -1.0, 1.0);
        let sites: Vec<Point2D> = (0..400)
            .map(|_| Point2D::new(1.8 * rng.f() - 0.9, 1.8 * rng.f() - 0.9))
            .collect();
        let m = MeshGrid::from_func(&sites, bbox, plane).unwrap();
        let (interp, gx, gy, _lap) = exercise(&m, Point2D::new(0.1, -0.2));
        assert!((interp - plane(0.1, -0.2)).abs() < 1e-6);
        assert!((gx - 2.0).abs() < 1e-6 && (gy + 3.0).abs() < 1e-6);
    }
}
