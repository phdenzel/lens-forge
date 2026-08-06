// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Lens on an adaptive/Voronoi lens plane.
//!
//! The unstructured counterpart to [`PixelLens`](crate::lens::PixelLens). Its
//! irregular sampling rules out the pixel-only [`Spectral`](FieldMethod::Spectral)
//! and [`Analytic`](FieldMethod::Analytic) methods, so it supports only
//! [`FastSum`](FieldMethod::FastSum): each cell is a mass element
//! $m_i = \kappa_i \cdot \mathrm{area}_i$ at its centroid, and the field is summed
//! from the 2D Poisson Green's function by one of the interchangeable
//! [`fastsum`](crate::fastsum) backends. The `MeshGrid` is stored, so a future
//! mesh-native method (e.g. a Voronoi Poisson solve) has the geometry it needs.

use crate::fastsum::{Backend, Summation, build_summation};
use crate::grid::{FieldArray1D, MeshGrid, Point2D, ScalarField2D, VoronoiMesh};
use crate::lens::model::{LensJet, LensModel, MassElement};
use crate::lens::{FieldMethod, FreeFormLens};
use crate::{LensForgeError, Result};

/// Lens on a Voronoi/adaptive plane.
///
/// Stores the [`MeshGrid`]; the field is produced by a chosen [`FieldMethod`]
/// (only [`FastSum`](FieldMethod::FastSum) is valid on an unstructured mesh).
/// Convergence is the nearest-cell κ (the Voronoi definition).
pub struct MeshLens {
    mesh: MeshGrid,
    method: FieldMethod,
    field: MeshField,
}

/// The realised field behind a [`MeshLens`], one variant per (mesh) [`FieldMethod`].
enum MeshField {
    /// On-demand free-space monopole summation over cell mass elements.
    FastSum(Box<dyn Summation + Send + Sync>),
    // extension point: a Voronoi-native Poisson solve (the mesh counterpart to pixel
    // `Spectral`) would land here as a further variant, plus a new `FieldMethod`.
}

impl MeshLens {
    /// Build from any scalar κ field, tessellating its cell centres into a
    /// [`MeshGrid`] (default method) — so `MeshLens` always holds a real mesh.
    ///
    /// A [`MeshGrid`] κ round-trips to the same tessellation; any other
    /// [`ScalarField2D`] (e.g. a [`PixelGrid`](crate::grid::PixelGrid)) is meshed
    /// from its cell centres. Chain [`with_method`](FreeFormLens::with_method) for a
    /// non-default backend/config.
    ///
    /// # Arguments
    /// * `field` - convergence κ sampled per cell (centroid + value).
    ///
    /// # Errors
    /// Propagates [`MeshGrid::new`] failure (empty or degenerate cell centres).
    pub fn from_field(field: &impl ScalarField2D) -> Result<Self> {
        Self::new(Self::tessellate(field)?)
    }

    /// Rebuild on new Voronoi generators `sites`, resampling the current κ and
    /// preserving the method — the mesh counterpart to
    /// [`PixelLens::with_grid`](crate::lens::PixelLens::with_grid).
    ///
    /// # Arguments
    /// * `sites` - new generating points; the current κ is sampled at each.
    ///
    /// # Errors
    /// Propagates [`MeshGrid::new`] failure (empty or degenerate sites).
    pub fn with_sites(self, sites: &[Point2D]) -> Result<Self> {
        let values: Vec<f64> = sites.iter().map(|&p| self.mesh.sample(p)).collect();
        let mesh = MeshGrid::new(sites, self.mesh.bounds(), FieldArray1D::from_vec(values))?;
        Self::build(mesh, self.method)
    }

    /// Rebuild after `iterations` Lloyd (CVT) relaxation sweeps, resampling the current
    /// κ onto the relaxed generators and preserving the method.
    ///
    /// Each sweep moves every site to its cell centroid, regularizing toward uniform,
    /// near-hexagonal cells — so it *removes* any adaptive concentration (it regularizes
    /// an irregular mesh, it does not adapt one to the lensing field).
    ///
    /// # Arguments
    /// * `iterations` - number of relaxation sweeps (`0` resamples onto the current sites).
    ///
    /// # Errors
    /// Propagates the mesh rebuild.
    pub fn with_lloyd_iters(self, iterations: usize) -> Result<Self> {
        let relaxed =
            VoronoiMesh::new(self.mesh.sites(), self.mesh.bounds())?.with_lloyd_iters(iterations)?;
        let sites = relaxed.sites;
        self.with_sites(&sites)
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.mesh.len()
    }

    /// Whether the lens has no cells (never true for a constructed lens).
    pub fn is_empty(&self) -> bool {
        self.mesh.is_empty()
    }

    /// Tessellate a scalar field's cell centres/values into a [`MeshGrid`].
    fn tessellate(field: &impl ScalarField2D) -> Result<MeshGrid> {
        let n = field.len();
        let mut sites = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            sites.push(field.center(i));
            values.push(field.value_at(i));
        }
        MeshGrid::new(&sites, field.bounds(), FieldArray1D::from_vec(values))
    }

    /// Per-cell mass elements `m = κ·area` at cell centroids for the summation.
    fn mass_elements(field: &impl ScalarField2D) -> Vec<MassElement> {
        (0..field.len())
            .map(|i| MassElement::new(field.center(i), field.value_at(i) * field.cell_area(i)))
            .collect()
    }
}

impl FreeFormLens for MeshLens {
    type Repr = MeshGrid;

    /// The exact direct summation — its only method family's reference backend.
    fn default_method() -> FieldMethod {
        FieldMethod::fast_sum(Backend::Direct)
    }

    /// A mesh supports only [`FastSum`](FieldMethod::FastSum).
    fn supports(method: FieldMethod) -> bool {
        matches!(method, FieldMethod::FastSum { .. })
    }

    fn build(mesh: MeshGrid, method: FieldMethod) -> Result<Self> {
        let field = match method {
            FieldMethod::FastSum { backend, config } => {
                let sources = Self::mass_elements(&mesh);
                MeshField::FastSum(build_summation(backend, &sources, config))
            }
            FieldMethod::Spectral | FieldMethod::Analytic => {
                return Err(LensForgeError::invalid_parameter(
                    "MeshLens supports only the FastSum method (no Spectral/Analytic on an unstructured mesh)",
                ));
            }
        };
        Ok(Self {
            mesh,
            method,
            field,
        })
    }

    fn method(&self) -> FieldMethod {
        self.method
    }

    fn with_method(self, method: FieldMethod) -> Result<Self> {
        Self::build(self.mesh, method)
    }
}

impl LensModel for MeshLens {
    fn jet(&self, theta: Point2D) -> LensJet {
        match &self.field {
            MeshField::FastSum(s) => s.jet(theta).into(),
        }
    }

    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<LensJet> {
        match &self.field {
            MeshField::FastSum(s) => s.jet_batch(thetas).into_iter().map(Into::into).collect(),
        }
    }

    /// Deflection only — cheaper than a full [`jet`](Self::jet) (skips the Hessian).
    fn deflection(&self, theta: Point2D) -> Point2D {
        match &self.field {
            MeshField::FastSum(s) => s.gradient(theta),
        }
    }

    /// Convergence at `theta` — the κ of the containing Voronoi cell (nearest site).
    fn convergence(&self, theta: Point2D) -> f64 {
        self.mesh.sample(theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lcg;
    use crate::fastsum::SumConfig;
    use crate::grid::{BBox, PixelGrid};
    use std::f64::consts::FRAC_1_PI;

    fn bbox() -> BBox {
        BBox::from_lims(-3.0, 3.0, -3.0, 3.0)
    }

    /// A κ field on a Voronoi mesh with explicit per-site values.
    fn mesh(sites: &[Point2D], kappa: &[f64]) -> MeshGrid {
        MeshGrid::new(sites, bbox(), FieldArray1D::from_vec(kappa.to_vec())).unwrap()
    }

    /// `n` scattered sites in the box, radially concentrated toward the centre.
    fn scattered(n: usize) -> Vec<Point2D> {
        let mut rng = Lcg::new(0x5EED);
        (0..n)
            .map(|_| {
                let r = 2.9 * rng.f() * rng.f();
                let t = std::f64::consts::TAU * rng.f();
                Point2D::new(r * t.cos(), r * t.sin())
            })
            .collect()
    }

    #[test]
    fn new_defaults_to_direct_fast_sum() {
        let mg = mesh(&scattered(24), &[0.3; 24]);
        assert_eq!(
            MeshLens::new(mg).unwrap().method(),
            FieldMethod::fast_sum(Backend::Direct)
        );
    }

    #[test]
    fn deflection_matches_direct_monopole_sum() {
        let sites = scattered(64);
        let kappa: Vec<f64> = (0..64).map(|i| 0.1 + 0.01 * i as f64).collect();
        let mg = mesh(&sites, &kappa);
        let lens = MeshLens::new(mg.clone()).unwrap();
        // Reference: sum monopole deflections from the mesh's own cells (m = κ·area).
        let field: &dyn ScalarField2D = &mg;
        let theta = Point2D::new(4.0, 3.0);
        let (mut ax, mut ay) = (0.0, 0.0);
        for i in 0..field.len() {
            let c = field.center(i);
            let m = field.value_at(i) * field.cell_area(i);
            let r2 = theta.distance_squared(&c);
            ax += m * FRAC_1_PI * (theta.x - c.x) / r2;
            ay += m * FRAC_1_PI * (theta.y - c.y) / r2;
        }
        let a = lens.deflection(theta);
        assert!(
            (a.x - ax).abs() < 1e-9 && (a.y - ay).abs() < 1e-9,
            "a=({},{}) exp=({},{})",
            a.x,
            a.y,
            ax,
            ay
        );
    }

    #[test]
    fn convergence_is_nearest_site_kappa() {
        let sites = [
            Point2D::new(-1.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.5),
        ];
        let mg = mesh(&sites, &[0.2, 0.9, 0.5]);
        let lens = MeshLens::new(mg).unwrap();
        assert_eq!(lens.convergence(Point2D::new(-0.9, 0.0)), 0.2);
        assert_eq!(lens.convergence(Point2D::new(0.9, 0.0)), 0.9);
    }

    #[test]
    fn len_reports_cell_count() {
        let mg = mesh(&scattered(24), &[0.3; 24]);
        let lens = MeshLens::new(mg).unwrap();
        assert_eq!(lens.len(), 24);
        assert!(!lens.is_empty());
    }

    #[test]
    fn jet_batch_matches_jet() {
        let kappa: Vec<f64> = (0..50).map(|i| 0.2 + 0.01 * i as f64).collect();
        let mg = mesh(&scattered(50), &kappa);
        let lens = MeshLens::new(mg).unwrap();
        let pts = [Point2D::new(3.0, 2.0), Point2D::new(-2.0, 1.5)];
        for (b, &p) in lens.jet_batch(&pts).iter().zip(&pts) {
            let s = lens.jet(p);
            assert!((b.psi - s.psi).abs() < 1e-12);
            assert!((b.alpha.x - s.alpha.x).abs() < 1e-12);
            assert_eq!(b.hessian, s.hessian);
        }
    }

    #[test]
    fn direct_and_barnes_hut_backends_agree() {
        let sites = scattered(200);
        let kappa: Vec<f64> = (0..200).map(|i| 0.1 + 0.01 * i as f64).collect();
        let mg = mesh(&sites, &kappa);
        let bh = MeshLens::build(mg.clone(), FieldMethod::fast_sum(Backend::BarnesHut)).unwrap();
        let direct = MeshLens::build(mg, FieldMethod::fast_sum(Backend::Direct)).unwrap();
        let theta = Point2D::new(3.0, 2.0);
        let rel = bh.deflection(theta).distance(&direct.deflection(theta))
            / direct.deflection(theta).norm().max(1e-6);
        assert!(rel < 2e-2, "rel err {rel}");
    }

    #[test]
    fn switching_method_rebuilds_the_backend() {
        let sites = scattered(80);
        let kappa: Vec<f64> = (0..80).map(|i| 0.2 + 0.005 * i as f64).collect();
        let mg = mesh(&sites, &kappa);
        let theta = Point2D::new(3.0, 2.0);
        let direct = MeshLens::new(mg.clone()).unwrap();
        // new() defaults to Direct; switching to Barnes–Hut then back equals it exactly.
        let chained = MeshLens::new(mg)
            .unwrap()
            .with_method(FieldMethod::fast_sum(Backend::BarnesHut))
            .unwrap()
            .with_method(FieldMethod::fast_sum(Backend::Direct))
            .unwrap();
        assert!(chained.deflection(theta).distance(&direct.deflection(theta)) < 1e-12);
    }

    #[test]
    fn softening_changes_the_field() {
        let mg = mesh(&scattered(60), &[0.4; 60]);
        let sharp = MeshLens::new(mg.clone()).unwrap();
        let softened = MeshLens::build(
            mg,
            FieldMethod::FastSum {
                backend: Backend::Direct,
                config: SumConfig::default().with_softening(0.5),
            },
        )
        .unwrap();
        // Softening the kernel measurably changes the deflection inside the cloud.
        let p = Point2D::new(0.8, 0.4);
        assert!(sharp.deflection(p).distance(&softened.deflection(p)) > 1e-3);
    }

    #[test]
    fn ray_trace_and_magnification_are_finite() {
        let sites = scattered(100);
        let mg = MeshGrid::from_func(&sites, bbox(), |_, _| 0.3).unwrap();
        let lens = MeshLens::new(mg).unwrap();
        let theta = Point2D::new(2.5, 0.0);
        assert!(lens.ray_trace(theta).is_finite());
        assert!(lens.magnification(theta).is_finite());
    }

    #[test]
    fn rejects_pixel_only_methods() {
        let mg = mesh(&scattered(24), &[0.3; 24]);
        assert!(MeshLens::build(mg.clone(), FieldMethod::Spectral).is_err());
        assert!(
            MeshLens::new(mg)
                .unwrap()
                .with_method(FieldMethod::Analytic)
                .is_err()
        );
    }

    #[test]
    fn with_sites_resamples_onto_new_generators() {
        // Uniform κ: resampling onto any generators preserves it; cell count follows sites.
        let mg = mesh(&scattered(40), &[0.3; 40]);
        let relensed = MeshLens::new(mg).unwrap().with_sites(&scattered(60)).unwrap();
        assert_eq!(relensed.len(), 60);
        assert!((relensed.convergence(Point2D::new(0.5, 0.2)) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn with_lloyd_iters_regularizes_and_preserves_kappa() {
        // Relaxing a uniform-κ mesh keeps κ and cell count and yields a finite field.
        let mg = mesh(&scattered(80), &[0.3; 80]);
        let relaxed = MeshLens::new(mg).unwrap().with_lloyd_iters(3).unwrap();
        assert_eq!(relaxed.len(), 80);
        assert!((relaxed.convergence(Point2D::new(0.4, -0.3)) - 0.3).abs() < 1e-12);
        assert!(relaxed.deflection(Point2D::new(2.5, 1.0)).norm().is_finite());
    }

    #[test]
    fn builds_from_a_generic_scalar_field() {
        // Any ScalarField2D drives a MeshLens: a PixelGrid κ is tessellated into a mesh.
        let pg = PixelGrid::from_func_sq(8, 2.0, |_, _| 0.5);
        let lens = MeshLens::from_field(&pg).unwrap();
        assert_eq!(lens.len(), 64);
        assert!(!lens.is_empty());
    }

    #[test]
    fn from_field_rejects_empty() {
        // An empty field has no cell centres to tessellate: MeshGrid::new rejects it.
        assert!(MeshLens::from_field(&PixelGrid::default()).is_err());
    }

    #[test]
    fn supports_only_fast_sum() {
        assert!(MeshLens::supports(FieldMethod::fast_sum(Backend::Direct)));
        assert!(!MeshLens::supports(FieldMethod::Spectral));
        assert!(!MeshLens::supports(FieldMethod::Analytic));
    }
}
