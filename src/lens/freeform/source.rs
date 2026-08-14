// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Free-form cells as `fastsum` sources — the bridge from a κ representation to the
//! [`SourceKernel`]s a summation backend aggregates.
//!
//! [`FastSum`](crate::lens::FieldMethod::FastSum) reduces each cell to a point
//! [`MassElement`] (its monopole moment). [`Analytic`](crate::lens::FieldMethod::Analytic)
//! keeps the cell's exact shape: [`PixelCellSource`] / [`VoronoiCellSource`] carry the
//! same `(center, strength)` far-field moment but an exact-integral `near_jet` (the
//! square / polygon log kernel), so a backend accelerates either identically.

use crate::fastsum::{Jet, Monopole, SourceKernel};
use crate::grid::{MeshGrid, Point2D, ScalarField2D, polygon_log_kernel, square_log_kernel};

/// A convergence cell reduced to its monopole moment (`strength = κ · area`) at the
/// cell centre — the [`FastSum`](crate::lens::FieldMethod::FastSum) source of a
/// free-form lens (a fastsum [`Monopole`]).
pub type MassElement = Monopole;

/// Add a per-unit-κ kernel `[ψ, ψ_x, ψ_y, ψ_xx, ψ_xy, ψ_yy]` scaled by `kappa` into
/// `field` (in place, for the backends' hot near-field loop).
fn add_kernel(field: &mut Jet, k: [f64; 6], kappa: f64) {
    field.scalar += kappa * k[0];
    field.gradient.x += kappa * k[1];
    field.gradient.y += kappa * k[2];
    field.hessian[0][0] += kappa * k[3];
    field.hessian[0][1] += kappa * k[4];
    field.hessian[1][0] += kappa * k[4];
    field.hessian[1][1] += kappa * k[5];
}

/// A uniform-κ square pixel as an [`Analytic`](crate::lens::FieldMethod::Analytic)
/// source: the exact square-pixel log integral in the near field, its monopole moment
/// far.
#[derive(Debug, Clone)]
pub struct PixelCellSource {
    center: Point2D,
    hx: f64,
    hy: f64,
    kappa: f64,
}

impl PixelCellSource {
    /// One source per pixel of a κ field with half-widths `hx`, `hy`.
    ///
    /// # Arguments
    /// * `field` - convergence κ (pixel centres and values).
    /// * `hx` - pixel half-width in x.
    /// * `hy` - pixel half-width in y.
    pub fn from_field(field: &impl ScalarField2D, hx: f64, hy: f64) -> Vec<Self> {
        (0..field.len())
            .map(|i| Self {
                center: field.center(i),
                hx,
                hy,
                kappa: field.value_at(i),
            })
            .collect()
    }
}

impl SourceKernel for PixelCellSource {
    fn center(&self) -> Point2D {
        self.center
    }

    fn strength(&self) -> f64 {
        self.kappa * 4.0 * self.hx * self.hy
    }

    fn accumulate_near(&self, field: &mut Jet, theta: Point2D, _eps2: f64) {
        add_kernel(
            field,
            square_log_kernel(self.center, self.hx, self.hy, theta),
            self.kappa,
        );
    }
}

/// A uniform-κ Voronoi cell as an [`Analytic`](crate::lens::FieldMethod::Analytic)
/// source: the exact convex-polygon log integral in the near field, its monopole
/// moment (at the centroid) far.
#[derive(Debug, Clone)]
pub struct VoronoiCellSource {
    verts: Vec<Point2D>,
    centroid: Point2D,
    area: f64,
    kappa: f64,
}

impl VoronoiCellSource {
    /// One source per Voronoi cell of a mesh κ field.
    ///
    /// # Arguments
    /// * `mesh` - the convergence mesh (clipped cell polygons and values).
    pub fn from_mesh(mesh: &MeshGrid) -> Vec<Self> {
        (0..mesh.len())
            .map(|i| Self {
                verts: mesh.cell_polygon(i).to_vec(),
                centroid: mesh.centroid(i),
                area: mesh.cell_area(i),
                kappa: mesh.value(i),
            })
            .collect()
    }
}

impl SourceKernel for VoronoiCellSource {
    fn center(&self) -> Point2D {
        self.centroid
    }

    fn strength(&self) -> f64 {
        self.kappa * self.area
    }

    fn accumulate_near(&self, field: &mut Jet, theta: Point2D, _eps2: f64) {
        add_kernel(field, polygon_log_kernel(&self.verts, theta), self.kappa);
    }
}
