// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! [`FieldOps`]: the representation-agnostic operator surface shared by
//! [`PixelGrid`](crate::grid::PixelGrid) and [`MeshGrid`](crate::grid::MeshGrid).
//!
//! [`ScalarField2D`] is the object-safe *data* trait (cell access, nearest-cell
//! sampling, aggregates). [`FieldOps`] extends it with the *operator* workflow
//! that differs by representation but shares a common contract: continuous
//! interpolation, the differential operators (gradient, Laplacian), and the
//! Poisson solve. Generic code bounded on `T: FieldOps` therefore runs
//! unchanged on either the structured grid or the unstructured mesh.
//!
//! Because the operators return field-valued results (`Self`, `(Self, Self)`),
//! `FieldOps` is `Sized` and not object-safe — use it as a generic bound
//! (`fn f<T: FieldOps>(…)`), not as `&dyn FieldOps`; reach for
//! [`ScalarField2D`] where trait objects are needed.
//!
//! # Representation-specific semantics
//!
//! The contract is the *operation*, not bit-identical numerics. Each
//! implementation uses its natural discretization and boundary handling: the
//! grid uses finite differences (default central stencil) with the
//! [`poisson_cg`](crate::grid::PixelGrid::poisson_cg) boundary condition, while
//! the mesh uses finite-volume operators with a natural zero-flux (Neumann)
//! boundary. Results agree in the continuum limit, not to the last bit.

use crate::grid::{Point2D, PoissonConfig, ScalarField2D};
use crate::{Result, SolveInfo};

/// The continuous-interpolation and differential/Poisson operators shared by the
/// scalar-field representations.
///
/// A supertrait of [`ScalarField2D`]; see the [module docs](self) for why it is
/// `Sized` (and thus generic-only, not a trait object) and how the numerics vary
/// by representation.
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

    /// Solve `∇²ψ = f` with this field as the source `f`, returning the potential
    /// `ψ` and solver diagnostics. The boundary handling is
    /// representation-specific (see the [module docs](self)).
    ///
    /// # Arguments
    /// * `cfg` - solver configuration (defaults if `None`).
    fn poisson(&self, cfg: Option<PoissonConfig>) -> Result<(Self, SolveInfo)>;

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
        points.iter().map(|&p| self.interpolate_or(p, 0.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, MeshGrid, PixelGrid, Point2D};
    use crate::utils::rng::Lcg;

    fn plane(x: f64, y: f64) -> f64 {
        2.0 * x - 3.0 * y + 5.0
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
        // Return interior-ish scalars so callers can sanity-check the values.
        let mid = f.len() / 2;
        (interp, gx.value_at(mid), gy.value_at(mid), lap.value_at(mid))
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
    fn meshgrid_satisfies_fieldops() {
        // The same generic routine, same linear field, on an unstructured mesh.
        let mut rng = Lcg::new(0x0F5);
        let bbox = BBox::from_lims(-1.0, 1.0, -1.0, 1.0);
        let sites: Vec<Point2D> =
            (0..400).map(|_| Point2D::new(1.8 * rng.f() - 0.9, 1.8 * rng.f() - 0.9)).collect();
        let m = MeshGrid::from_func(&sites, bbox, plane).unwrap();
        let (interp, gx, gy, _lap) = exercise(&m, Point2D::new(0.1, -0.2));
        assert!((interp - plane(0.1, -0.2)).abs() < 1e-6);
        assert!((gx - 2.0).abs() < 1e-6 && (gy + 3.0).abs() < 1e-6);
    }
}
