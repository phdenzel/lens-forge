// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! `MeshGrid`: a scalar field on a Voronoi tessellation.
//!
//! The unstructured counterpart to [`PixelGrid`](crate::grid::PixelGrid): a
//! [`VoronoiMesh`] geometry paired with one value per cell. The immutable
//! geometry is shared via [`Arc`], so value-producing operations (`scale`,
//! `map`, …) clone cheaply without duplicating the tessellation.

use std::sync::Arc;
use ndarray::ArrayView1;
use super::voronoi::VoronoiMesh;
use crate::grid::{BBox, FieldArray1D, Point2D, ScalarField2D};
use crate::{LensForgeError, Result};

/// A scalar field sampled on a Voronoi tessellation (one value per cell).
#[derive(Clone)]
pub struct MeshGrid {
    mesh: Arc<VoronoiMesh>,
    /// The per-cell values, one per site.
    values: FieldArray1D,
}

impl MeshGrid {
    /// Build a mesh grid from `sites` clipped to `bbox`, with a value per site.
    ///
    /// # Errors
    /// Returns an error if `values.len() != sites.len()` or the geometry is
    /// invalid (empty sites, degenerate `bbox`, non-finite sites).
    pub fn new(sites: &[Point2D], bbox: BBox, values: FieldArray1D) -> Result<Self> {
        if values.len() != sites.len() {
            return Err(LensForgeError::dimension_mismatch(
                format!("{} values", sites.len()),
                format!("{} values", values.len()),
            ));
        }
        let mesh = VoronoiMesh::new(sites, bbox)?;
        Ok(Self { mesh: Arc::new(mesh), values })
    }

    /// Mesh grid of zeros over `sites`.
    pub fn zeros(sites: &[Point2D], bbox: BBox) -> Result<Self> {
        Self::new(sites, bbox, FieldArray1D::zeros(sites.len()))
    }

    /// Mesh grid of ones over `sites`.
    pub fn ones(sites: &[Point2D], bbox: BBox) -> Result<Self> {
        Self::new(sites, bbox, FieldArray1D::ones(sites.len()))
    }

    /// Mesh grid with a constant value over `sites`.
    pub fn constant(sites: &[Point2D], bbox: BBox, value: f64) -> Result<Self> {
        Self::new(sites, bbox, FieldArray1D::from_elem(sites.len(), value))
    }

    /// Mesh grid sampling `f` at each site.
    ///
    /// # Arguments
    /// * `sites` - the generating points, one per cell.
    /// * `bbox` - the domain box the cells are clipped to.
    /// * `f` - function mapping a site point into its cell value.
    pub fn from_point_func<F: Fn(Point2D) -> f64>(
        sites: &[Point2D],
        bbox: BBox,
        f: F,
    ) -> Result<Self> {
        let values: FieldArray1D = sites.iter().map(|&s| f(s)).collect();
        Self::new(sites, bbox, values)
    }

    /// Mesh grid sampling `f` at each site's `(x, y)` coordinates.
    ///
    /// # Arguments
    /// * `sites` - the generating points, one per cell.
    /// * `bbox` - the domain box the cells are clipped to.
    /// * `f` - function mapping site coordinates into a cell value.
    pub fn from_func<F: Fn(f64, f64) -> f64>(
        sites: &[Point2D],
        bbox: BBox,
        f: F,
    ) -> Result<Self> {
        let values: FieldArray1D = sites.iter().map(|s| f(s.x, s.y)).collect();
        Self::new(sites, bbox, values)
    }

    /// Reuse an existing geometry with a new set of values (shares the
    /// tessellation, no rebuild).
    pub fn with_values(&self, values: FieldArray1D) -> Result<Self> {
        if values.len() != self.values.len() {
            return Err(LensForgeError::dimension_mismatch(
                format!("{} values", self.values.len()),
                format!("{} values", values.len()),
            ));
        }
        Ok(Self { mesh: self.mesh.clone(), values })
    }

    /* ================ Getter & Setter ==================================== */
    /// Number of cells.
    pub fn len(&self) -> usize {
        self.mesh.n_cells()
    }

    /// Whether there are no cells.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Cell values as a contiguous slice.
    pub fn values(&self) -> &[f64] {
        self.values.as_slice().expect("mesh values are contiguous")
    }

    /// Value of cell `i`.
    pub fn value(&self, i: usize) -> f64 {
        self.values[i]
    }

    /// Value of cell `i` with bounds checking.
    ///
    /// # Arguments
    /// * `i` - the cell index.
    pub fn value_safely(&self, i: usize) -> Option<f64> {
        self.values.get(i).copied()
    }

    /// A view of the underlying value array.
    pub fn as_array1(&self) -> ArrayView1<'_, f64> {
        self.values.view()
    }

    /// A mutable reference to the underlying value array.
    pub fn as_mut_array1(&mut self) -> &mut FieldArray1D {
        &mut self.values
    }

    /// Cell values as an owned `Vec` (copy).
    pub fn as_vec(&self) -> Vec<f64> {
        self.values.to_vec()
    }

    /// Set the value of cell `i`.
    pub fn set(&mut self, i: usize, value: f64) {
        self.values[i] = value;
    }

    /// The underlying Voronoi geometry (shared, immutable). Use it to reach the
    /// domain attributes, per-cell geometry, and point location.
    pub fn mesh(&self) -> &VoronoiMesh {
        &self.mesh
    }

    /// Generating site of cell `i`.
    pub fn site(&self, i: usize) -> Point2D {
        self.mesh.site(i)
    }

    /// Area-weighted centroid of cell `i`.
    pub fn centroid(&self, i: usize) -> Point2D {
        self.mesh.centroid(i)
    }

    /// Area of cell `i`.
    pub fn area(&self, i: usize) -> f64 {
        self.mesh.cell_area(i)
    }

    /// Clipped polygon (counter-clockwise) of cell `i`.
    pub fn cell_polygon(&self, i: usize) -> &[Point2D] {
        self.mesh.cell_polygon(i)
    }

    /// Voronoi neighbours of cell `i`.
    pub fn neighbors(&self, i: usize) -> &[usize] {
        self.mesh.neighbors_of(i)
    }

    /* ============ Mesh attributes ======================================== */
    /// Number of cells (one per generating site).
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.mesh.n_cells()
    }

    /// Generating sites, one per cell.
    #[inline]
    pub fn sites(&self) -> &[Point2D] {
        self.mesh.sites()
    }

    /// Area-weighted centroids of every cell.
    #[inline]
    pub fn centroids(&self) -> Vec<Point2D> {
        self.mesh.centroids()
    }

    /// Domain width.
    #[inline]
    pub fn width(&self) -> f64 {
        self.mesh.width()
    }

    /// Domain height.
    #[inline]
    pub fn height(&self) -> f64 {
        self.mesh.height()
    }

    /// Domain centre.
    #[inline]
    pub fn center(&self) -> Point2D {
        self.mesh.center()
    }

    /// x extent as `(min, max)`.
    #[inline]
    pub fn extent_x(&self) -> (f64, f64) {
        self.mesh.extent_x()
    }

    /// y extent as `(min, max)`.
    #[inline]
    pub fn extent_y(&self) -> (f64, f64) {
        self.mesh.extent_y()
    }

    /// Aspect ratio (width / height).
    #[inline]
    pub fn aspect_ratio(&self) -> f64 {
        self.mesh.aspect_ratio()
    }

    /// Whether `p` lies within the domain box.
    ///
    /// # Arguments
    /// * `p` - the point to test.
    #[inline]
    pub fn contains(&self, p: Point2D) -> bool {
        self.mesh.contains(p)
    }

    /// Total covered area (equals the domain area).
    pub fn total_area(&self) -> f64 {
        self.mesh.total_area()
    }

    /// Index of the cell containing `p` (nearest site), via an `O(log n)` k-d
    /// tree query.
    pub fn cell_of(&self, p: Point2D) -> usize {
        self.mesh.cell_of(p)
    }

    /// Index of the cell containing `p`, via a warm-started greedy walk seeded at
    /// `hint`.
    ///
    /// Exact and equal to [`cell_of`](Self::cell_of), but ~`O(1)` per query when
    /// `hint` is the previous result for spatially-coherent queries (a raster
    /// scan, a batch of nearby points).
    ///
    /// # Arguments
    /// * `p` - the query point to locate.
    /// * `hint` - cell to start the walk from (any valid cell index).
    ///
    /// Returns the index of the containing (nearest-site) cell.
    pub fn cell_of_walk(&self, p: Point2D, hint: usize) -> usize {
        self.mesh.cell_of_walk(p, hint)
    }

    /* ================ Statistics ========================================= */
    /// Sum of all cell values.
    pub fn sum(&self) -> f64 {
        self.values.iter().sum()
    }

    /// Mean of all cell values (`0` for an empty mesh).
    pub fn mean(&self) -> f64 {
        self.values.mean().unwrap_or(0.0)
    }

    /// Minimum cell value.
    pub fn min(&self) -> f64 {
        self.values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Maximum cell value.
    pub fn max(&self) -> f64 {
        self.values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Variance of all cell values.
    pub fn var(&self) -> f64 {
        let mean = self.mean();
        self.values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.values.len() as f64
    }

    /// Standard deviation of all cell values.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Flux (area-weighted integral) of the field, `Σ vᵢ · areaᵢ`.
    pub fn flux(&self) -> f64 {
        self.values
            .iter()
            .enumerate()
            .map(|(i, &v)| v * self.mesh.cell_area(i))
            .sum()
    }

    /* ================ Mappings =========================================== */
    /// A copy with every value scaled by `factor` (shares geometry).
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            mesh: self.mesh.clone(),
            values: &self.values * factor,
        }
    }

    /// A copy with `scalar` added to every value (shares geometry).
    pub fn shift(&self, scalar: f64) -> Self {
        Self {
            mesh: self.mesh.clone(),
            values: &self.values + scalar,
        }
    }

    /// A copy with `f` applied to every value (shares geometry).
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self {
            mesh: self.mesh.clone(),
            values: self.values.mapv(f),
        }
    }

    /// Apply `f` to every value in place.
    pub fn apply<F: Fn(f64) -> f64>(&mut self, f: F) {
        self.values.mapv_inplace(f);
    }

    /// Whether `other` shares this mesh's geometry (same tessellation).
    ///
    /// Cheap when the two grids descend from a common geometry (shared `Arc`);
    /// otherwise compares the domain box and every site.
    fn same_geometry(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.mesh, &other.mesh)
            || (self.mesh.bbox == other.mesh.bbox && self.sites() == other.sites())
    }

    /// Add two mesh grids sharing the same geometry (value-wise).
    ///
    /// # Arguments
    /// * `other` - other mesh grid, which must share this one's tessellation.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if !self.same_geometry(other) {
            return Err(LensForgeError::dimension_mismatch(
                format!("{} cells", self.len()),
                format!("{} cells", other.len()),
            ));
        }
        Ok(Self {
            mesh: self.mesh.clone(),
            values: &self.values + &other.values,
        })
    }

    /// Subtract two mesh grids sharing the same geometry (value-wise).
    ///
    /// # Arguments
    /// * `other` - other mesh grid, which must share this one's tessellation.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        if !self.same_geometry(other) {
            return Err(LensForgeError::dimension_mismatch(
                format!("{} cells", self.len()),
                format!("{} cells", other.len()),
            ));
        }
        Ok(Self {
            mesh: self.mesh.clone(),
            values: &self.values - &other.values,
        })
    }

    /// Multiply two mesh grids sharing the same geometry (value-wise).
    ///
    /// # Arguments
    /// * `other` - other mesh grid, which must share this one's tessellation.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        if !self.same_geometry(other) {
            return Err(LensForgeError::dimension_mismatch(
                format!("{} cells", self.len()),
                format!("{} cells", other.len()),
            ));
        }
        Ok(Self {
            mesh: self.mesh.clone(),
            values: &self.values * &other.values,
        })
    }

    /// Clamp every cell value into `[min, max]` (in place).
    ///
    /// # Arguments
    /// * `min` - minimum value.
    /// * `max` - maximum value.
    pub fn clamp(&mut self, min: f64, max: f64) {
        self.values.mapv_inplace(|x| x.clamp(min, max));
    }

    /// Normalize the field so its value sum equals `num` (default `1`).
    ///
    /// # Arguments
    /// * `num` - target sum (`None` normalizes to `1`).
    pub fn normalize_sum(&mut self, num: Option<f64>) {
        let c = num.unwrap_or(1.0);
        let total = self.sum();
        if total.abs() > 1e-15 {
            self.values *= c / total;
        }
    }

    /// Normalize the field so its maximum value equals `num` (default `1`).
    ///
    /// # Arguments
    /// * `num` - target maximum (`None` normalizes to `1`).
    pub fn normalize_max(&mut self, num: Option<f64>) {
        let c = num.unwrap_or(1.0);
        let max = self.max();
        if max.abs() > 1e-15 {
            self.values *= c / max;
        }
    }

    /* ================ Resampling ========================================= */
    /// Resample onto a new set of `sites` by nearest-cell sampling, building a
    /// fresh tessellation clipped to `bbox`.
    ///
    /// # Arguments
    /// * `sites` - the new generating points.
    /// * `bbox` - the domain box the new cells are clipped to.
    pub fn resample(&self, sites: &[Point2D], bbox: BBox) -> Result<Self> {
        let values = FieldArray1D::from_vec(self.sample_batch(sites));
        Self::new(sites, bbox, values)
    }
}

impl ScalarField2D for MeshGrid {
    fn len(&self) -> usize {
        self.mesh.n_cells()
    }

    fn value_at(&self, i: usize) -> f64 {
        self.values[i]
    }

    fn center(&self, i: usize) -> Point2D {
        self.mesh.site(i)
    }

    fn cell_area(&self, i: usize) -> f64 {
        self.mesh.cell_area(i)
    }

    fn bounds(&self) -> BBox {
        self.mesh.bbox
    }

    fn sample(&self, p: Point2D) -> f64 {
        // Piecewise-constant: value of the containing (nearest-site) cell.
        self.values[self.mesh.cell_of(p)]
    }

    fn sample_batch(&self, points: &[Point2D]) -> Vec<f64> {
        // Locate by warm-started walk: the first point cold (k-d tree), then each
        // subsequent point seeded with the previous result. ~O(1) per query when
        // `points` are spatially coherent (a raster scan, a path); still exact for
        // any order, but scattered inputs are better served point-wise by
        // `sample`/`cell_of` (an unconditional O(log n) k-d tree query).
        let mut out = Vec::with_capacity(points.len());
        let mut hint = None;
        for &p in points {
            let cell = match hint {
                None => self.mesh.cell_of(p),
                Some(h) => self.mesh.cell_of_walk(p, h),
            };
            hint = Some(cell);
            out.push(self.values[cell]);
        }
        out
    }
}

/* ================ Display ================================================ */
impl std::fmt::Display for MeshGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MeshGrid({}, range: [{:.4}, {:.4}], mean: {:.4})",
            self.mesh(),
            self.min(),
            self.max(),
            self.mean()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rng::Lcg;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn demo() -> MeshGrid {
        let sites = [p(0.25, 0.25), p(0.75, 0.25), p(0.75, 0.75), p(0.25, 0.75)];
        MeshGrid::from_point_func(&sites, BBox::from_lims(0.0, 1.0, 0.0, 1.0), |s| s.x + s.y).unwrap()
    }

    #[test]
    fn construction_and_access() {
        let m = demo();
        assert_eq!(m.len(), 4);
        assert!((m.value(0) - 0.5).abs() < 1e-12);
        assert!((m.total_area() - 1.0).abs() < 1e-12);
        assert!((m.area(0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn mesh_accessor_exposes_geometry() {
        // Geometry is reached through the shared mesh (mirroring PixelGrid::grid),
        // and the delegating cell accessors agree with it.
        let m = demo();
        assert_eq!(m.mesh().n_cells(), m.len());
        assert!((m.mesh().width() - 1.0).abs() < 1e-12);
        for i in 0..m.len() {
            assert_eq!(m.site(i), m.mesh().site(i));
            assert_eq!(m.area(i), m.mesh().cell_area(i));
        }
    }

    #[test]
    fn delegated_getters_match_the_mesh() {
        let m = demo();
        assert_eq!(m.n_cells(), m.mesh().n_cells());
        assert_eq!(m.width(), m.mesh().width());
        assert_eq!(m.height(), m.mesh().height());
        assert_eq!(m.center(), m.mesh().center());
        assert_eq!(m.extent_x(), m.mesh().extent_x());
        assert_eq!(m.extent_y(), m.mesh().extent_y());
        assert_eq!(m.aspect_ratio(), m.mesh().aspect_ratio());
        assert_eq!(m.sites(), m.mesh().sites());
        assert!(m.contains(p(0.5, 0.5)) && !m.contains(p(9.0, 9.0)));
    }

    #[test]
    fn flux_of_constant_is_value_times_area() {
        let sites = [p(0.2, 0.3), p(0.8, 0.1), p(0.5, 0.9), p(0.1, 0.6), p(0.9, 0.7)];
        let m = MeshGrid::constant(&sites, BBox::from_lims(0.0, 1.0, 0.0, 1.0), 2.5).unwrap();
        assert!((m.flux() - 2.5 * 1.0).abs() < 1e-9);
    }

    #[test]
    fn sample_returns_nearest_cell_value() {
        let m = demo();
        // A point deep in the lower-left quadrant is nearest to site 0.
        assert_eq!(m.sample(p(0.1, 0.1)), m.value(0));
        assert_eq!(m.sample(p(0.9, 0.9)), m.value(2));
    }

    #[test]
    fn sample_batch_matches_pointwise_sample() {
        // The warm-started batch override must agree with the cold, point-wise
        // path for every point, coherent or not.
        let mut rng = Lcg::new(0xBA7C4);
        let sites: Vec<Point2D> = (0..120).map(|_| p(rng.f(), rng.f())).collect();
        let values: FieldArray1D = (0..sites.len()).map(|i| i as f64).collect();
        let m = MeshGrid::new(&sites, BBox::from_lims(0.0, 1.0, 0.0, 1.0), values).unwrap();

        let pts: Vec<Point2D> = (0..500).map(|_| p(rng.f(), rng.f())).collect();
        let batch = m.sample_batch(&pts);
        for (k, &q) in pts.iter().enumerate() {
            assert_eq!(batch[k], m.sample(q), "point {k} {q:?}");
        }
    }

    #[test]
    fn value_ops_share_geometry() {
        let m = demo();
        let scaled = m.scale(2.0);
        assert!((scaled.value(0) - 1.0).abs() < 1e-12);
        // Geometry is shared (same Arc), not rebuilt.
        assert!(Arc::ptr_eq(&m.mesh, &scaled.mesh));
    }

    #[test]
    fn works_as_dyn_scalar_field() {
        let m = demo();
        let f: &dyn ScalarField2D = &m;
        assert_eq!(f.len(), 4);
        assert!((f.flux() - m.flux()).abs() < 1e-12);
    }

    #[test]
    fn ones_and_from_func_construct_expected_values() {
        let sites = [p(0.25, 0.25), p(0.75, 0.75)];
        let bbox = BBox::from_lims(0.0, 1.0, 0.0, 1.0);
        let ones = MeshGrid::ones(&sites, bbox).unwrap();
        assert_eq!(ones.as_vec(), vec![1.0, 1.0]);

        let m = MeshGrid::from_func(&sites, bbox, |x, y| x * x + y * y).unwrap();
        assert!((m.value(0) - 0.125).abs() < 1e-12);
        assert!((m.value(1) - 1.125).abs() < 1e-12);
    }

    #[test]
    fn value_safely_bounds_checks() {
        let m = demo();
        assert_eq!(m.value_safely(0), Some(m.value(0)));
        assert_eq!(m.value_safely(99), None);
    }

    #[test]
    fn statistics_over_cell_values() {
        // demo values: [0.5, 1.0, 1.5, 1.0]; each cell area 0.25.
        let m = demo();
        assert!((m.sum() - 4.0).abs() < 1e-12);
        assert!((m.mean() - 1.0).abs() < 1e-12);
        assert!((m.min() - 0.5).abs() < 1e-12);
        assert!((m.max() - 1.5).abs() < 1e-12);
        assert!((m.var() - 0.125).abs() < 1e-12);
        assert!((m.std() - 0.125_f64.sqrt()).abs() < 1e-12);
        assert!((m.flux() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn add_sub_mul_share_geometry() {
        let m1 = demo();
        let m2 = m1.scale(2.0); // shares the Arc geometry
        let added = m1.add(&m2).unwrap();
        let subbed = m1.sub(&m2).unwrap();
        let mulled = m1.mul(&m2).unwrap();
        for i in 0..m1.len() {
            assert!((added.value(i) - 3.0 * m1.value(i)).abs() < 1e-12);
            assert!((subbed.value(i) + m1.value(i)).abs() < 1e-12);
            assert!((mulled.value(i) - 2.0 * m1.value(i).powi(2)).abs() < 1e-12);
        }
        // Geometry is shared, not rebuilt.
        assert!(Arc::ptr_eq(&m1.mesh, &added.mesh));
    }

    #[test]
    fn binary_ops_reject_mismatched_geometry() {
        let m1 = demo();
        let other_sites = [p(0.1, 0.1), p(0.9, 0.9)];
        let m2 = MeshGrid::ones(&other_sites, BBox::from_lims(0.0, 1.0, 0.0, 1.0)).unwrap();
        assert!(m1.add(&m2).is_err());
    }

    #[test]
    fn same_geometry_holds_for_independent_but_identical_meshes() {
        // Distinct Arcs, but identical sites and box: add must succeed.
        let sites = [p(0.25, 0.25), p(0.75, 0.25), p(0.75, 0.75), p(0.25, 0.75)];
        let bbox = BBox::from_lims(0.0, 1.0, 0.0, 1.0);
        let a = MeshGrid::ones(&sites, bbox).unwrap();
        let b = MeshGrid::constant(&sites, bbox, 3.0).unwrap();
        assert!(!Arc::ptr_eq(&a.mesh, &b.mesh));
        let sum = a.add(&b).unwrap();
        assert!(sum.as_vec().iter().all(|&v| (v - 4.0).abs() < 1e-12));
    }

    #[test]
    fn clamp_and_normalize() {
        let mut m = demo();
        m.clamp(0.75, 1.25);
        assert!((m.min() - 0.75).abs() < 1e-12);
        assert!((m.max() - 1.25).abs() < 1e-12);

        let mut s = demo();
        s.normalize_sum(Some(1.0));
        assert!((s.sum() - 1.0).abs() < 1e-12);

        let mut x = demo();
        x.normalize_max(Some(1.0));
        assert!((x.max() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn resample_takes_nearest_cell_values() {
        let m = demo();
        // A coarser mesh: one site deep in each of two quadrants.
        let new_sites = [p(0.1, 0.1), p(0.9, 0.9)];
        let r = m.resample(&new_sites, BBox::from_lims(0.0, 1.0, 0.0, 1.0)).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.value(0), m.sample(p(0.1, 0.1)));
        assert_eq!(r.value(1), m.sample(p(0.9, 0.9)));
    }

    #[test]
    fn display_reports_range_and_mean() {
        let m = demo();
        let s = format!("{m}");
        assert!(s.starts_with("MeshGrid(VoronoiMesh("));
        assert!(s.contains("mean: 1.0000"));
    }

    #[test]
    fn zeros_constructs_empty_valued_mesh() {
        let sites = [p(0.25, 0.25), p(0.75, 0.75)];
        let z = MeshGrid::zeros(&sites, BBox::from_lims(0.0, 1.0, 0.0, 1.0)).unwrap();
        assert_eq!(z.as_vec(), vec![0.0, 0.0]);
        assert!(!z.is_empty());
    }

    #[test]
    fn new_rejects_value_count_mismatch() {
        let sites = [p(0.2, 0.2), p(0.8, 0.8)];
        let bbox = BBox::from_lims(0.0, 1.0, 0.0, 1.0);
        assert!(MeshGrid::new(&sites, bbox, FieldArray1D::from_vec(vec![1.0])).is_err());
    }

    #[test]
    fn with_values_shares_geometry_and_checks_length() {
        let m = demo();
        let updated = m.with_values(FieldArray1D::from_vec(vec![10.0, 20.0, 30.0, 40.0])).unwrap();
        assert_eq!(updated.values(), &[10.0, 20.0, 30.0, 40.0]);
        assert!(Arc::ptr_eq(&m.mesh, &updated.mesh)); // geometry reused, not rebuilt
        assert!(m.with_values(FieldArray1D::from_vec(vec![1.0])).is_err());
    }

    #[test]
    fn ndarray_accessors_view_and_mutate_values() {
        let mut m = demo();
        assert_eq!(m.as_array1().len(), 4);
        assert_eq!(m.as_array1()[0], m.value(0));
        m.set(0, 99.0);
        assert_eq!(m.value(0), 99.0);
        m.as_mut_array1()[1] = 7.0;
        assert_eq!(m.value(1), 7.0);
    }

    #[test]
    fn per_cell_geometry_accessors_delegate_to_mesh() {
        let m = demo();
        for i in 0..m.len() {
            assert_eq!(m.centroid(i), m.mesh().centroid(i));
            assert_eq!(m.neighbors(i), m.mesh().neighbors_of(i));
        }
        assert_eq!(m.centroids(), m.mesh().centroids());
    }

    #[test]
    fn shift_map_apply_transform_values() {
        // demo values: [0.5, 1.0, 1.5, 1.0].
        let m = demo();
        assert!((m.shift(2.0).value(0) - 2.5).abs() < 1e-12);
        assert!((m.map(|v| v * 10.0).value(0) - 5.0).abs() < 1e-12);
        let mut a = demo();
        a.apply(|v| v + 1.0);
        assert!((a.value(0) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn sub_and_mul_reject_mismatched_geometry() {
        let m1 = demo();
        let other = [p(0.1, 0.1), p(0.9, 0.9)];
        let m2 = MeshGrid::ones(&other, BBox::from_lims(0.0, 1.0, 0.0, 1.0)).unwrap();
        assert!(m1.sub(&m2).is_err());
        assert!(m1.mul(&m2).is_err());
    }

    #[test]
    fn scalar_field_center_returns_the_site() {
        let m = demo();
        let f: &dyn ScalarField2D = &m;
        for i in 0..m.len() {
            assert_eq!(f.center(i), m.site(i));
        }
    }

    #[test]
    fn constructors_propagate_geometry_errors() {
        // Value count matches the sites, but the domain is degenerate, so the `?`
        // on VoronoiMesh::new inside `new` must propagate the error.
        let bad = BBox::from_lims(0.0, 0.0, 0.0, 1.0); // min.x == max.x
        assert!(MeshGrid::zeros(&[p(0.5, 0.5)], bad).is_err());
        assert!(MeshGrid::new(&[p(0.5, 0.5)], bad, FieldArray1D::from_vec(vec![1.0])).is_err());
    }

    #[test]
    fn normalize_is_a_noop_on_degenerate_totals() {
        let bbox = BBox::from_lims(0.0, 1.0, 0.0, 1.0);
        let sites = [p(0.25, 0.25), p(0.75, 0.75)];
        // Zero sum / zero max: normalization must leave the values untouched.
        let mut s = MeshGrid::zeros(&sites, bbox).unwrap();
        s.normalize_sum(None);
        assert_eq!(s.as_vec(), vec![0.0, 0.0]);
        let mut m = MeshGrid::zeros(&sites, bbox).unwrap();
        m.normalize_max(None);
        assert_eq!(m.as_vec(), vec![0.0, 0.0]);
    }
}
