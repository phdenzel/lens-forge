// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! `PixelGrid`: a scalar field on a uniform Cartesian grid.
//!
//! The structured counterpart to [`MeshGrid`](crate::grid::MeshGrid): a
//! [`Grid2D`] geometry paired with a 2D array of sampled values. Operators
//! (derivatives, FFTs, Poisson solvers, interpolation) are inherent methods
//! added in sibling submodules.

use core::f64;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use crate::grid::{BBox, FieldArray2D, Grid2D, Point2D};
use crate::{LensForgeError, Result};

/// A 2D pixel grid with associated values
#[derive(Debug, Clone, Default)]
pub struct PixelGrid {
    /// The underlying grid geometry
    pub grid: Grid2D,
    /// The pixel values stored as a 2D array [ny, nx] (row-major)
    pub values: FieldArray2D,
}

impl PixelGrid {
    /// Create a pixel grid from a grid geometry's arguments and a value array.
    ///
    /// # Arguments
    /// * `nx` - number of pixels in the x direction
    /// * `ny` - number of pixels in the y direction
    /// * `bbox` - physical domain of the grid
    /// * `values` - pixel value array; must match the grid shape `[ny, nx]`
    ///
    /// # Errors
    /// Returns an error if the geometry is invalid (see [`Grid2D::new`]) or the
    /// value array does not match the grid shape.
    pub fn new(nx: usize, ny: usize, bbox: BBox, values: FieldArray2D) -> Result<Self> {
        let grid = Grid2D::new(nx, ny, bbox)?;
        Self::from_values(grid, values)
    }

    /// Create a new pixel grid initialized with zeros.
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    pub fn zeros(grid: Grid2D) -> Self {
        Self {
            grid,
            values: Array2::zeros(grid.shape()),
        }
    }

    /// Create a new pixel grid initialized with ones.
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    pub fn ones(grid: Grid2D) -> Self {
        Self {
            grid,
            values: Array2::ones(grid.shape()),
        }
    }

    /// Create a new pixel grid initialized with a constant value.
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `value` - constant value for the pixels
    pub fn constant(grid: Grid2D, value: f64) -> Self {
        Self {
            grid,
            values: Array2::from_elem(grid.shape(), value),
        }
    }

    /// Create a new pixel grid initialized from an ndarray.
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `values` - pixel value array
    pub fn from_values(grid: Grid2D, values: FieldArray2D) -> Result<Self> {
        if values.nrows() != grid.ny || values.ncols() != grid.nx {
            return Err(LensForgeError::dimension_mismatch(
                format!("({}, {})", grid.ny, grid.nx),
                format!("({}, {})", values.nrows(), values.ncols()),
            ));
        }
        Ok(Self { grid, values })
    }

    /// Create from a flat vector (row-major order).
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `vec` - pixel value vector
    pub fn from_vec(grid: Grid2D, v: Vec<f64>) -> Result<Self> {
        // `from_shape_vec` already rejects a length that does not match the shape,
        // so map its error to a dimension mismatch rather than checking twice.
        let (expected, got) = (grid.n_pixels(), v.len());
        let values = Array2::from_shape_vec(grid.shape(), v)
            .map_err(|_| LensForgeError::dimension_mismatch(expected.to_string(), got.to_string()))?;
        Ok(Self { grid, values })
    }

    /// Create from a 1D array (row-major order).
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `arr` - flat pixel value array
    pub fn from_array1(grid: Grid2D, arr: &Array1<f64>) -> Result<Self> {
        Self::from_vec(grid, arr.to_vec())
    }

    /// Reuse this grid's geometry with a new value array (no geometry rebuild).
    ///
    /// # Arguments
    /// * `values` - replacement values; must match the grid shape `[ny, nx]`.
    pub fn with_values(&self, values: FieldArray2D) -> Result<Self> {
        Self::from_values(self.grid, values)
    }

    /// Create from a function using 2D points.
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `f` - function mapping 2D points into values
    pub fn from_point_func<F: Fn(&Point2D) -> f64>(grid: Grid2D, f: F) -> Self {
        let values = grid.map_points(f);
        Self { grid, values }
    }

    /// Create from a function using pixel coordinates.
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `f` - function mapping coordinates into values
    pub fn from_func<F: Fn(f64, f64) -> f64>(grid: Grid2D, f: F) -> Self {
        let values = grid.map_centers(f);
        Self { grid, values }
    }

    /// Create from a function using pixel coordinates, dimensions, and extent.
    ///
    /// # Arguments
    /// * `nx` - Number of pixels in the x direction
    /// * `ny` - Number of pixels in the y direction
    /// * `rx` - Half the physical extent in x (grid spans from -rx to +rx)
    /// * `ry` - Half the physical extent in y (grid spans from -ry to +ry)
    /// * `f` - function mapping coordinates into values
    pub fn from_func_sq<F>(n: usize, radius: f64, f: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        let grid = Grid2D::square(n, radius).unwrap();
        let values = grid.map_centers(f);
        Self { grid, values }
    }

    /// Create from a function using pixel coordinates, dimensions, and extent.
    ///
    /// # Arguments
    /// * `nx` - Number of pixels in the x direction
    /// * `ny` - Number of pixels in the y direction
    /// * `rx` - Half the physical extent in x (grid spans from -rx to +rx)
    /// * `ry` - Half the physical extent in y (grid spans from -ry to +ry)
    /// * `f` - function mapping coordinates into values
    pub fn from_func_rect<F>(nx: usize, ny: usize, rx: f64, ry: f64, f: F) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        let grid = Grid2D::rectangular(nx, ny, rx, ry).unwrap();
        let values = grid.map_centers(f);
        Self { grid, values }
    }

    /* ================ Getter & Setter ==================================== */
    /// Get value at pixel (i, j).
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[[j, i]]
    }

    /// Get value at pixel (i, j) with bounds checking.
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    #[inline]
    pub fn get_safely(&self, i: usize, j: usize) -> Option<f64> {
        if i < self.grid.nx && j < self.grid.ny {
            Some(self.values[[j, i]])
        } else {
            None
        }
    }

    /// Set value at pixel (i, j).
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    /// * `value` - pixel grid value to be set
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.values[[j, i]] = value;
    }

    /* ============ Grid attributes ======================================== */
    /// Total number of pixels.
    #[inline]
    pub fn n_pixels(&self) -> usize {
        self.grid.n_pixels()
    }

    /// Shape as `(ny, nx)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.grid.shape()
    }

    /// Grid width.
    #[inline]
    pub fn width(&self) -> f64 {
        self.grid.width()
    }

    /// Grid height.
    #[inline]
    pub fn height(&self) -> f64 {
        self.grid.height()
    }

    /// Pixel size in x.
    #[inline]
    pub fn dx(&self) -> f64 {
        self.grid.dx()
    }

    /// Pixel size in y.
    #[inline]
    pub fn dy(&self) -> f64 {
        self.grid.dy()
    }

    /// Pixel area.
    #[inline]
    pub fn pixel_area(&self) -> f64 {
        self.grid.pixel_area()
    }

    /// Total grid area (`width × height`).
    #[inline]
    pub fn total_area(&self) -> f64 {
        self.grid.total_area()
    }

    /// Grid centre.
    #[inline]
    pub fn center(&self) -> Point2D {
        self.grid.center()
    }

    /// x extent as `(min, max)`.
    #[inline]
    pub fn extent_x(&self) -> (f64, f64) {
        self.grid.extent_x()
    }

    /// y extent as `(min, max)`.
    #[inline]
    pub fn extent_y(&self) -> (f64, f64) {
        self.grid.extent_y()
    }

    /// Aspect ratio (width / height).
    #[inline]
    pub fn aspect_ratio(&self) -> f64 {
        self.grid.aspect_ratio()
    }

    /// Whether `p` lies within the grid domain.
    ///
    /// # Arguments
    /// * `p` - the point to test.
    #[inline]
    pub fn contains(&self, p: &Point2D) -> bool {
        self.grid.contains(p)
    }

    /// Flat (row-major) index of the pixel containing `p`, or `None` if `p` lies
    /// outside the domain.
    ///
    /// # Arguments
    /// * `p` - the query point to locate.
    #[inline]
    pub fn cell_of(&self, p: &Point2D) -> Option<usize> {
        self.grid.point_to_pixel(p).map(|(i, j)| self.grid.flatten_index(i, j))
    }

    /* ============ Values ================================================= */
    /// Get values as a flat 1D array (row-major order).
    pub fn as_array1(&self) -> Array1<f64> {
        self.values
            .clone()
            .into_shape_with_order(self.grid.n_pixels())
            .expect("Shape conversion should not fail")
    }

    /// Get values as a flat Vec (row-major order).
    pub fn as_vec(&self) -> Vec<f64> {
        self.values.iter().cloned().collect()
    }

    /// Get a view of the 2D array.
    pub fn as_array2(&self) -> ArrayView2<'_, f64> {
        self.values.view()
    }

    /// Get a mutable reference to the underlying array.
    pub fn as_mut_array2(&mut self) -> &mut FieldArray2D {
        &mut self.values
    }

    /// Get row `j` as a slice.
    ///
    /// # Arguments
    /// * `j` - pixel grid index along y axis
    pub fn row(&self, j: usize) -> ArrayView1<'_, f64> {
        self.values.row(j)
    }

    /// Get column `i` as a slice.
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    pub fn column(&self, i: usize) -> ArrayView1<'_, f64> {
        self.values.column(i)
    }

    /* ================ Statistics ========================================= */
    /// Sum of all pixel grid values.
    pub fn sum(&self) -> f64 {
        self.values.sum()
    }

    /// Average of all pixel grid values.
    pub fn mean(&self) -> f64 {
        self.values.mean().unwrap_or(0.0)
    }

    /// Minimum of all pixel grid values.
    pub fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Maximum of all pixel grid values.
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Variance of all pixel grid values.
    pub fn var(&self) -> f64 {
        let mean = self.mean();
        self.values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.grid.n_pixels() as f64
    }

    /// Standard deviation of all pixel grid values.
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Flux (per-area sum) of all pixel grid values.
    pub fn flux(&self) -> f64 {
        self.sum() * self.grid.pixel_area()
    }

    /* ================ Mappings =========================================== */
    /// Scale the pixel grid values by a factor.
    ///
    /// # Arguments
    /// * `scale` - scaling factor
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            grid: self.grid,
            values: &self.values * factor,
        }
    }

    /// Shift the pixel grid values by a scalar.
    ///
    /// # Arguments
    /// * `scalar` - scalar addend
    pub fn shift(&self, scalar: f64) -> Self {
        Self {
            grid: self.grid,
            values: &self.values + scalar,
        }
    }

    /// Add two pixel grids of same dimensions.
    ///
    /// # Arguments
    /// * `other` - other pixel grid
    pub fn add(&self, other: &Self) -> Result<Self> {
        if !self.grid.same_geometry(&other.grid) {
            return Err(LensForgeError::dimension_mismatch(
                format!("({}, {})", self.grid.ny, self.grid.nx),
                format!("{}, {}", other.grid.ny, other.grid.nx),
            ));
        }
        Ok(Self {
            grid: self.grid,
            values: &self.values + &other.values,
        })
    }

    /// Subtract two pixel grids of same dimensions.
    ///
    /// # Arguments
    /// * `other` - other pixel grid
    pub fn sub(&self, other: &Self) -> Result<Self> {
        if !self.grid.same_geometry(&other.grid) {
            return Err(LensForgeError::dimension_mismatch(
                format!("({}, {})", self.grid.ny, self.grid.nx),
                format!("{}, {}", other.grid.ny, other.grid.nx),
            ));
        }
        Ok(Self {
            grid: self.grid,
            values: &self.values - &other.values,
        })
    }

    /// Multiply two pixel grids of same dimensions.
    ///
    /// # Arguments
    /// * `other` - other pixel grid
    pub fn mul(&self, other: &Self) -> Result<Self> {
        if !self.grid.same_geometry(&other.grid) {
            return Err(LensForgeError::dimension_mismatch(
                format!("({}, {})", self.grid.ny, self.grid.nx),
                format!("{}, {}", other.grid.ny, other.grid.nx),
            ));
        }
        Ok(Self {
            grid: self.grid,
            values: &self.values * &other.values,
        })
    }

    /// Apply function to the pixel grid values (in-place).
    ///
    /// # Arguments
    /// * `f` - function to be applied
    pub fn apply<F: Fn(f64) -> f64>(&mut self, f: F) {
        self.values.mapv_inplace(f);
    }

    /// Map function to the pixel grid values.
    ///
    /// # Arguments
    /// * `f` - function to be mapped
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self {
            grid: self.grid,
            values: self.values.mapv(f),
        }
    }

    /// Clamp the pixel grid values.
    ///
    /// # Arguments
    /// * `min` - minimum value
    /// * `max` - maximum value
    pub fn clamp(&mut self, min: f64, max: f64) {
        self.values.mapv_inplace(|x| x.clamp(min, max));
    }

    /// Normalize the map's sum to a given total.
    ///
    /// # Arguments
    /// * `num` - total sum normalization
    pub fn normalize_sum(&mut self, num: Option<f64>) {
        let c = num.unwrap_or(1.0);
        let total = self.sum();
        if total.abs() > 1e-15 {
            self.values *= c / total;
        }
    }

    /// Normalize the map's maximum to a given number.
    ///
    /// # Arguments
    /// * `num` - total sum normalization
    pub fn normalize_max(&mut self, num: Option<f64>) {
        let c = num.unwrap_or(1.0);
        let max = self.max();
        if max.abs() > 1e-15 {
            self.values *= c / max;
        }
    }

    /* ================ Resampling ========================================= */
    /// Resample to a different grid using bilinear interpolation.
    ///
    /// # Arguments
    /// * `grid` -
    pub fn resample(&self, grid: Grid2D) -> Self {
        Self::from_point_func(grid, |p| self.interpolate_or(*p, 0.0))
    }

    /// Resample to a different grid using bilinear interpolation in parallel
    /// (using rayon backend).
    ///
    /// # Arguments
    /// * `grid` -
    pub fn par_resample(&self, grid: Grid2D) -> Self {
        let nx = grid.nx;
        let ny = grid.ny;

        let values: Vec<f64> = (0..ny * nx)
            .into_par_iter()
            .map(|idx| {
                let i = idx % nx;
                let j = idx / nx;
                let p = grid.pixel_center(i, j);
                self.interpolate_or(p, 0.0)
            })
            .collect();

        Self {
            grid,
            values: Array2::from_shape_vec((ny, nx), values).unwrap(),
        }
    }

    /* ================ Interpolation ====================================== */
    /// Bilinear interpolation at arbitrary point.
    ///
    /// # Arguments
    /// * `p` - point for interpolation
    #[inline]
    pub fn interpolate(&self, p: Point2D) -> f64 {
        match self.grid.bilinear_interp(&p) {
            Some(kernel) => kernel.apply(&self.values),
            None => 0.0,
        }
    }

    /// Bilinear interpolation with explicit fallback value.
    ///
    /// # Arguments
    /// * `p` - point for interpolation
    /// * `default` - out-of-bounds/fallback value
    #[inline]
    pub fn interpolate_or(&self, p: Point2D, default: f64) -> f64 {
        match self.grid.bilinear_interp(&p) {
            Some(kernel) => kernel.apply(&self.values),
            None => default,
        }
    }

    /// Batch interpolation for multiple points.
    ///
    /// # Arguments
    /// * `points` - batch array of points
    pub fn interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        points.iter().map(|&p| self.interpolate_or(p, 0.0)).collect()
    }

    /// Batch interpolation for multiple points in parallel.
    ///
    /// # Arguments
    /// * `points` - batch array of points
    pub fn par_interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        points
            .par_iter()
            .map(|&p| self.interpolate_or(p, 0.0))
            .collect()
    }
}

/* ================ Display ================================================ */
impl std::fmt::Display for PixelGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PixelGrid({}, range: [{:.4}, {:.4}], mean: {:.4})",
            self.grid,
            self.min(),
            self.max(),
            self.mean()
        )
    }
}

/* ================ Unit tests ============================================= */
#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: impl Into<Option<f64>>) -> bool {
        let eps = eps.into().unwrap_or(1e-12);
        (a - b).abs() < eps
    }

    #[test]
    fn test_pixel_grid_zeros() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::zeros(grid);

        assert_eq!(pg.values.shape(), &[4, 4]);
        assert!(approx_eq(pg.sum(), 0.0, 1e-12));
    }

    #[test]
    fn delegated_getters_match_the_grid() {
        let grid = Grid2D::rectangular(4, 3, 2.0, 1.0).unwrap();
        let pg = PixelGrid::zeros(grid);
        assert_eq!(pg.n_pixels(), grid.n_pixels());
        assert_eq!(pg.shape(), grid.shape());
        assert_eq!(pg.width(), grid.width());
        assert_eq!(pg.height(), grid.height());
        assert_eq!(pg.dx(), grid.dx());
        assert_eq!(pg.dy(), grid.dy());
        assert_eq!(pg.pixel_area(), grid.pixel_area());
        assert_eq!(pg.center(), grid.center());
        assert_eq!(pg.extent_x(), grid.extent_x());
        assert_eq!(pg.extent_y(), grid.extent_y());
        assert_eq!(pg.aspect_ratio(), grid.aspect_ratio());
        assert_eq!(pg.contains(&Point2D::origin()), grid.contains(&Point2D::origin()));
    }

    #[test]
    fn new_builds_geometry_from_args() {
        let bbox = BBox::from_lims(-2.0, 2.0, -1.0, 1.0);
        let values = Array2::from_shape_fn((3, 4), |(j, i)| (i + 4 * j) as f64);
        let pg = PixelGrid::new(4, 3, bbox, values.clone()).unwrap();
        assert_eq!(pg.grid, Grid2D::new(4, 3, bbox).unwrap());
        assert_eq!(pg.values, values);
        // Shape mismatch against the built geometry is rejected.
        assert!(PixelGrid::new(4, 3, bbox, Array2::zeros((2, 2))).is_err());
        // Invalid geometry is rejected too.
        assert!(PixelGrid::new(0, 3, bbox, Array2::zeros((3, 0))).is_err());
    }

    #[test]
    fn test_pixel_grid_ones() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::ones(grid);

        assert_eq!(pg.values.shape(), &[4, 4]);
        assert!(approx_eq(pg.sum(), 16.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_constant() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::constant(grid, 3.5);

        assert!(approx_eq(pg.mean(), 3.5, 1e-12));
    }

    #[test]
    fn test_pixel_grid_from_func() {
        let grid = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let center_val = pg.get(2, 2);
        let expected = grid.x_at(2).powi(2) + grid.y_at(2).powi(2);
        assert!(approx_eq(center_val, expected, 1e-12));
    }

    #[test]
    fn test_pixel_grid_get_set() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let mut pg = PixelGrid::zeros(grid);

        pg.set(1, 2, 5.0);
        assert!(approx_eq(pg.get(1, 2), 5.0, 1e-12));

        assert!(pg.get_safely(1, 2).is_some());
        assert!(pg.get_safely(10, 10).is_none());
    }

    #[test]
    fn test_pixel_grid_statistics() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::from_func(grid, |_, _| 2.0);

        assert!(approx_eq(pg.sum(), 32.0, 1e-12));
        assert!(approx_eq(pg.mean(), 2.0, 1e-12));
        assert!(approx_eq(pg.min(), 2.0, 1e-12));
        assert!(approx_eq(pg.max(), 2.0, 1e-12));
        assert!(approx_eq(pg.var(), 0.0, 1e-12));
        assert!(approx_eq(pg.std(), 0.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_scale_shift() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::ones(grid);

        let scaled = pg.scale(2.0);
        assert!(approx_eq(scaled.mean(), 2.0, 1e-12));

        let shifted = pg.shift(3.0);
        assert!(approx_eq(shifted.mean(), 4.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_add_sub_mul() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg1 = PixelGrid::constant(grid, 2.0);
        let pg2 = PixelGrid::constant(grid, 3.0);

        let added = pg1.add(&pg2).unwrap();
        assert!(approx_eq(added.mean(), 5.0, 1e-12));

        let subtracted = pg1.sub(&pg2).unwrap();
        assert!(approx_eq(subtracted.mean(), -1.0, 1e-12));

        let multiplied = pg1.mul(&pg2).unwrap();
        assert!(approx_eq(multiplied.mean(), 6.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_normalize() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let mut pg = PixelGrid::constant(grid, 2.0);

        pg.normalize_sum(Some(1.0));
        assert!(approx_eq(pg.sum(), 1.0, 1e-12));

        let mut pg2 = PixelGrid::constant(grid, 2.0);
        pg2.normalize_max(Some(1.0));
        assert!(approx_eq(pg2.max(), 1.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_resample() {
        let grid1 = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid1, |x, y| x + y);

        let grid2 = Grid2D::square(8, 2.0).unwrap();
        let resampled = pg.resample(grid2);

        assert_eq!(resampled.grid.nx, 8);
        assert_eq!(resampled.grid.ny, 8);

        let center = Point2D::origin();
        let orig_val = pg.interpolate(center);
        let resampled_val = resampled.interpolate(center);
        assert!(approx_eq(orig_val, resampled_val, 1e-12));
    }

    #[test]
    fn test_pixel_grid_par_resample() {
        let grid1 = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid1, |x, y| x + y);

        let grid2 = Grid2D::square(8, 2.0).unwrap();
        let resampled_seq = pg.resample(grid2);
        let resampled_par = pg.par_resample(grid2);

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(
                    resampled_seq.get(i, j),
                    resampled_par.get(i, j),
                    1e-12
                ));
            }
        }
    }

    #[test]
    fn total_area_matches_grid() {
        let grid = Grid2D::rectangular(4, 3, 2.0, 1.0).unwrap();
        let pg = PixelGrid::zeros(grid);
        assert!(approx_eq(pg.total_area(), grid.total_area(), 1e-12));
        assert!(approx_eq(pg.total_area(), pg.width() * pg.height(), 1e-12));
    }

    #[test]
    fn cell_of_locates_pixel_or_none() {
        let grid = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::zeros(grid);
        let idx = pg.cell_of(&Point2D::origin()).unwrap();
        let (i, j) = grid.unflatten_index(idx);
        assert!(grid.is_valid_pixel(i, j));
        assert_eq!(pg.cell_of(&Point2D::new(100.0, 100.0)), None);
    }

    #[test]
    fn with_values_reuses_geometry() {
        let grid = Grid2D::square(3, 1.0).unwrap();
        let pg = PixelGrid::zeros(grid);
        let updated = pg.with_values(Array2::from_elem(grid.shape(), 7.0)).unwrap();
        assert!(approx_eq(updated.mean(), 7.0, 1e-12));
        assert!(updated.grid.same_geometry(&pg.grid));
        // Shape mismatch is rejected.
        assert!(pg.with_values(Array2::zeros((1, 1))).is_err());
    }

    #[test]
    fn test_pixel_grid_interpolation() {
        let grid = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, _| x);

        let center = Point2D::origin();
        let interp = pg.interpolate(center);

        assert!(approx_eq(interp, 0.0, 1e-12));
    }

    #[test]
    fn from_vec_array1_and_errors() {
        let grid = Grid2D::new(2, 2, BBox::from_lims(0.0, 2.0, 0.0, 2.0)).unwrap();
        let pg = PixelGrid::from_vec(grid, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(approx_eq(pg.get(0, 0), 1.0, None));
        assert!(approx_eq(pg.get(1, 1), 4.0, None));
        // Wrong length is rejected.
        assert!(PixelGrid::from_vec(grid, vec![1.0, 2.0]).is_err());

        let arr = ndarray::Array1::from(vec![5.0, 6.0, 7.0, 8.0]);
        let pg2 = PixelGrid::from_array1(grid, &arr).unwrap();
        assert!(approx_eq(pg2.get(1, 0), 6.0, None));
    }

    #[test]
    fn from_func_sq_and_rect() {
        let sq = PixelGrid::from_func_sq(4, 2.0, |x, y| x + y);
        assert_eq!(sq.shape(), (4, 4));
        let rect = PixelGrid::from_func_rect(5, 3, 2.0, 1.0, |x, _| x);
        assert_eq!(rect.shape(), (3, 5));
        assert!(approx_eq(rect.get(0, 0), rect.grid.x_at(0), None));
    }

    #[test]
    fn value_accessors() {
        let grid = Grid2D::new(3, 2, BBox::from_lims(0.0, 3.0, 0.0, 2.0)).unwrap();
        let mut pg = PixelGrid::from_vec(grid, (0..6).map(|k| k as f64).collect()).unwrap();

        assert_eq!(pg.as_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(pg.as_array1().to_vec(), pg.as_vec());
        assert_eq!(pg.as_array2().shape(), &[2, 3]);
        assert_eq!(pg.row(1).to_vec(), vec![3.0, 4.0, 5.0]);
        assert_eq!(pg.column(0).to_vec(), vec![0.0, 3.0]);

        // Mutable view lets us edit the backing store in place.
        pg.as_mut_array2()[[0, 0]] = 42.0;
        assert!(approx_eq(pg.get(0, 0), 42.0, None));
    }

    #[test]
    fn variance_flux_and_extrema() {
        let grid = Grid2D::new(2, 2, BBox::from_lims(0.0, 2.0, 0.0, 2.0)).unwrap();
        let pg = PixelGrid::from_vec(grid, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(approx_eq(pg.min(), 1.0, None));
        assert!(approx_eq(pg.max(), 4.0, None));
        assert!(approx_eq(pg.mean(), 2.5, None));
        // Population variance of [1,2,3,4] = 1.25.
        assert!(approx_eq(pg.var(), 1.25, 1e-12));
        assert!(approx_eq(pg.std(), 1.25_f64.sqrt(), 1e-12));
        // Flux = sum * pixel_area; pixel_area here is 1.
        assert!(approx_eq(pg.flux(), pg.sum() * grid.pixel_area(), 1e-12));
    }

    #[test]
    fn apply_map_and_clamp() {
        let grid = Grid2D::square(3, 1.0).unwrap();
        let base = PixelGrid::constant(grid, 2.0);

        let mapped = base.map(|v| v * 3.0);
        assert!(approx_eq(mapped.mean(), 6.0, None));
        // map leaves the original untouched.
        assert!(approx_eq(base.mean(), 2.0, None));

        let mut pg = base.clone();
        pg.apply(|v| v + 1.0);
        assert!(approx_eq(pg.mean(), 3.0, None));

        let mut clamped = PixelGrid::from_vec(
            Grid2D::new(2, 2, BBox::from_lims(0.0, 2.0, 0.0, 2.0)).unwrap(),
            vec![-1.0, 0.5, 2.0, 5.0],
        )
        .unwrap();
        clamped.clamp(0.0, 1.0);
        assert!(approx_eq(clamped.min(), 0.0, None));
        assert!(approx_eq(clamped.max(), 1.0, None));
    }

    #[test]
    fn normalize_ignores_degenerate_totals() {
        let grid = Grid2D::square(3, 1.0).unwrap();
        // Zero sum: normalize_sum must be a no-op (no division by ~0).
        let mut zero = PixelGrid::zeros(grid);
        zero.normalize_sum(None);
        assert!(approx_eq(zero.sum(), 0.0, None));
        // Zero max: normalize_max must be a no-op too.
        zero.normalize_max(None);
        assert!(approx_eq(zero.max(), 0.0, None));
    }

    #[test]
    fn arithmetic_rejects_geometry_mismatch() {
        let a = PixelGrid::constant(Grid2D::square(4, 1.0).unwrap(), 1.0);
        let b = PixelGrid::constant(Grid2D::square(5, 1.0).unwrap(), 1.0);
        assert!(a.add(&b).is_err());
        assert!(a.sub(&b).is_err());
        assert!(a.mul(&b).is_err());
    }

    #[test]
    fn interpolation_out_of_bounds_and_batch() {
        let grid = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, _| x);

        let outside = Point2D::new(100.0, 100.0);
        // interpolate falls back to 0.0, interpolate_or to the given default.
        assert!(approx_eq(pg.interpolate(outside), 0.0, None));
        assert!(approx_eq(pg.interpolate_or(outside, -9.0), -9.0, None));

        let pts = [Point2D::origin(), outside, grid.pixel_center(1, 1)];
        let seq = pg.interpolate_batch(&pts);
        let par = pg.par_interpolate_batch(&pts);
        assert_eq!(seq.len(), 3);
        for (a, b) in seq.iter().zip(par.iter()) {
            assert!(approx_eq(*a, *b, 1e-12));
        }
        assert!(approx_eq(seq[1], 0.0, None));
    }

    #[test]
    fn display_formatting() {
        let grid = Grid2D::square(3, 1.0).unwrap();
        let pg = PixelGrid::constant(grid, 1.0);
        let s = format!("{pg}");
        assert!(s.contains("PixelGrid"));
    }
}
