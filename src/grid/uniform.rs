// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Uniform grid with pixels

use crate::grid::{
    ComplexArray1D, ComplexFieldArray2D, FieldArray2D, FourierSymbols, Grid2D, Point2D,
    VectorField2D,
};
use crate::{LensForgeError, Result};
use core::f64;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use num::complex::Complex;
use phastft::{
    fft_64_with_opts_and_plan,
    options::Options,
    planner::{Direction, Planner64},
};
use rayon::prelude::*;

macro_rules! zip3_for_each {
    ($par:expr, $out:expr, $a:expr, $b:expr, $body:expr) => {{
        let z = Zip::from($out).and($a).and($b);
        if $par {
            z.par_for_each($body);
        } else {
            z.for_each($body);
        }
    }};
}

macro_rules! zip4_for_each {
    ($par:expr, $out:expr, $a:expr, $b:expr, $c:expr, $body:expr) => {{
        let z = Zip::from($out).and($a).and($b).and($c);
        if $par {
            z.par_for_each($body);
        } else {
            z.for_each($body);
        }
    }};
}

macro_rules! zip5_for_each {
    ($par:expr, $out:expr, $a:expr, $b:expr, $c:expr, $d:expr, $body:expr) => {{
        let z = ndarray::Zip::from($out).and($a).and($b).and($c).and($d);
        if $par {
            z.par_for_each($body);
        } else {
            z.for_each($body);
        }
    }};
}

/// A 2D pixel grid with associated values
#[derive(Debug, Clone, Default)]
pub struct PixelGrid {
    /// The underlying grid geometry
    pub grid: Grid2D,
    /// The pixel values stored as a 2D array [ny, nx] (row-major)
    pub values: FieldArray2D,
}

impl PixelGrid {
    /// Create a new pixel grid initialized with zeros
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    pub fn zeros(grid: Grid2D) -> Self {
        Self {
            grid,
            values: Array2::zeros(grid.shape()),
        }
    }

    /// Create a new pixel grid initialized with ones
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    pub fn ones(grid: Grid2D) -> Self {
        Self {
            grid,
            values: Array2::ones(grid.shape()),
        }
    }

    /// Create a new pixel grid initialized with a constant value
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

    /// Create a new pixel grid initialized from an ndarray
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

    /// Create from a flat vector (row-major order)
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `vec` - pixel value vector
    pub fn from_vec(grid: Grid2D, v: Vec<f64>) -> Result<Self> {
        if v.len() != grid.n_pixels() {
            return Err(LensForgeError::dimension_mismatch(
                format!("{}", grid.n_pixels()),
                format!("{}", v.len()),
            ));
        }
        let values = Array2::from_shape_vec(grid.shape(), v)
            .map_err(|e| LensForgeError::numerical_error(e.to_string()))?;
        Ok(Self { grid, values })
    }

    /// Create from a 1D array (row-major order)
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `arr` - flat pixel value array
    pub fn from_array1(grid: Grid2D, arr: &Array1<f64>) -> Result<Self> {
        Self::from_vec(grid, arr.to_vec())
    }

    /// Create from a function using 2D points
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `f` - function mapping 2D points into values
    pub fn from_point_func<F: Fn(&Point2D) -> f64>(grid: Grid2D, f: F) -> Self {
        let values = Array2::from_shape_fn(grid.shape(), |(j, i)| f(&grid.pixel_center(i, j)));
        Self { grid, values }
    }

    /// Create from a function using pixel coordinates
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `f` - function mapping coordinates into values
    pub fn from_func<F: Fn(f64, f64) -> f64>(grid: Grid2D, f: F) -> Self {
        let values = Array2::from_shape_fn(grid.shape(), |(j, i)| f(grid.x_at(i), grid.y_at(j)));
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
        let values = Array2::from_shape_fn(grid.shape(), |(j, i)| f(grid.x_at(i), grid.y_at(j)));
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
        let values = Array2::from_shape_fn(grid.shape(), |(j, i)| f(grid.x_at(i), grid.y_at(j)));
        Self { grid, values }
    }

    /* ================ Getter & Setter ==================================== */
    /// Get value at pixel (i, j)
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[[j, i]]
    }

    /// Get value at pixel (i, j) with bounds checking
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

    /// Set value at pixel (i, j)
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    /// * `value` - pixel grid value to be set
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.values[[j, i]] = value;
    }

    /// Get values as a flat 1D array (row-major order)
    pub fn as_array1(&self) -> Array1<f64> {
        self.values
            .clone()
            .into_shape_with_order(self.grid.n_pixels())
            .expect("Shape conversion should not fail")
    }

    /// Get values as a flat Vec (row-major order)
    pub fn as_vec(&self) -> Vec<f64> {
        self.values.iter().cloned().collect()
    }

    /// Get a view of the 2D array
    pub fn as_array2(&self) -> ArrayView2<'_, f64> {
        self.values.view()
    }

    /// Get a mutable reference to the underlying array
    pub fn as_mut_array2(&mut self) -> &mut FieldArray2D {
        &mut self.values
    }

    /// Get row j as a slice
    ///
    /// # Arguments
    /// * `j` - pixel grid index along y axis
    pub fn row(&self, j: usize) -> ArrayView1<'_, f64> {
        self.values.row(j)
    }

    /// Get column i as a slice
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    pub fn column(&self, i: usize) -> ArrayView1<'_, f64> {
        self.values.column(i)
    }

    /* ================ Statistics ========================================= */
    /// Sum of all pixel grid values
    pub fn sum(&self) -> f64 {
        self.values.sum()
    }

    /// Average of all pixel grid values
    pub fn mean(&self) -> f64 {
        self.values.mean().unwrap_or(0.0)
    }

    /// Minimum of all pixel grid values
    pub fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Maximum of all pixel grid values
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Variance of all pixel grid values
    pub fn var(&self) -> f64 {
        let mean = self.mean();
        self.values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.grid.n_pixels() as f64
    }

    /// Standard deviation of all pixel grid values
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Flux (per-area sum) of all pixel grid values
    pub fn flux(&self) -> f64 {
        self.sum() * self.grid.pixel_area()
    }

    /* ================ Mappings =========================================== */
    /// Scale the pixel grid values by a factor
    ///
    /// # Arguments
    /// * `scale` - scaling factor
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            grid: self.grid,
            values: &self.values * factor,
        }
    }

    /// Shift the pixel grid values by a scalar
    ///
    /// # Arguments
    /// * `scalar` - scalar addend
    pub fn shift(&self, scalar: f64) -> Self {
        Self {
            grid: self.grid,
            values: &self.values + scalar,
        }
    }

    /// Add two pixel grids of same dimensions
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

    /// Subtract two pixel grids of same dimensions
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

    /// Multiply two pixel grids of same dimensions
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

    /// Apply function to the pixel grid values (in-place)
    ///
    /// # Arguments
    /// * `f` - function to be applied
    pub fn apply<F: Fn(f64) -> f64>(&mut self, f: F) {
        self.values.mapv_inplace(f);
    }

    /// Map function to the pixel grid values
    ///
    /// # Arguments
    /// * `f` - function to be mapped
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self {
            grid: self.grid,
            values: self.values.mapv(f),
        }
    }

    /// Clamp the pixel grid values
    ///
    /// # Arguments
    /// * `min` - minimum value
    /// * `max` - maximum value
    pub fn clamp(&mut self, min: f64, max: f64) {
        self.values.mapv_inplace(|x| x.clamp(min, max));
    }

    /// Normalize the map's sum to a given total
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

    /// Normalize the map's maximum to a given number
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
    /// Resample to a different grid using bilinear interpolation
    ///
    /// # Arguments
    /// * `grid` -
    pub fn resample(&self, grid: Grid2D) -> Self {
        Self::from_point_func(grid, |p| self.interpolate_or(p, 0.0))
    }

    /// Resample to a different grid using bilinear interpolation in parallel (using rayon backend)
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
                self.interpolate_or(&p, 0.0)
            })
            .collect();

        Self {
            grid,
            values: Array2::from_shape_vec((ny, nx), values).unwrap(),
        }
    }

    /* ================ Differentiation ==================================== */
    /// Bilinear interpolation at arbitrary point
    ///
    /// # Arguments
    /// * `p` - point for interpolation
    #[inline]
    pub fn interpolate(&self, p: &Point2D) -> f64 {
        match self.grid.bilinear_interp(p) {
            Some(kernel) => kernel.apply(&self.values),
            None => 0.0,
        }
    }

    /// Bilinear interpolation with explicit fallback value
    ///
    /// # Arguments
    /// * `p` - point for interpolation
    /// * `default` - out-of-bounds/fallback value
    #[inline]
    pub fn interpolate_or(&self, p: &Point2D, default: f64) -> f64 {
        match self.grid.bilinear_interp(p) {
            Some(kernel) => kernel.apply(&self.values),
            None => default,
        }
    }

    /// Batch interpolation for multiple points
    ///
    /// # Arguments
    /// * `points` - batch array of points
    pub fn interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        points.iter().map(|p| self.interpolate_or(p, 0.0)).collect()
    }

    /// Batch interpolation for multiple points in parallel
    ///
    /// # Arguments
    /// * `points` - batch array of points
    pub fn par_interpolate_batch(&self, points: &[Point2D]) -> Vec<f64> {
        points
            .par_iter()
            .map(|p| self.interpolate_or(p, 0.0))
            .collect()
    }

    /// Compute gradient at a pixel (i, j) using central differences
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    #[inline]
    pub fn gradient(
        &self,
        i: usize,
        j: usize,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<Point2D> {
        let stencil = stencil.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        let dvdx = match stencil {
            DerivativeStencil::Central => {
                if i == 0 {
                    (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx
                } else if i == nx - 1 {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx
                } else {
                    (self.values[[j, i + 1]] - self.values[[j, i - 1]]) / (2.0 * self.grid.dx)
                }
            }
            DerivativeStencil::Forward => {
                if i < nx - 1 {
                    (self.values[[j, i + 1]] - self.values[[j, i]]) / self.grid.dx
                } else {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx
                }
            }
            DerivativeStencil::Backward => {
                if i > 0 {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx
                } else {
                    (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx
                }
            }
            DerivativeStencil::Central4 => {
                if i >= 2 && i < nx - 2 {
                    (-self.values[[j, i + 2]] + 8.0 * self.values[[j, i + 1]]
                        - 8.0 * self.values[[j, i - 1]]
                        + self.values[[j, i - 2]])
                        / (12.0 * self.grid.dx)
                } else if i >= 1 && i < nx - 1 {
                    // Fall back to 2nd-order central
                    (self.values[[j, i + 1]] - self.values[[j, i - 1]]) / (2.0 * self.grid.dx)
                } else if i == 0 {
                    (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx
                } else {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx
                }
            }
            DerivativeStencil::Central6 => {
                if i >= 3 && i < nx - 3 {
                    (self.values[[j, i + 3]] - 9.0 * self.values[[j, i + 2]]
                        + 45.0 * self.values[[j, i + 1]]
                        - 45.0 * self.values[[j, i - 1]]
                        + 9.0 * self.values[[j, i - 2]]
                        - self.values[[j, i - 3]])
                        / (60.0 * self.grid.dx)
                } else if i >= 2 && i < nx - 2 {
                    // Fall back to 4th-order
                    (-self.values[[j, i + 2]] + 8.0 * self.values[[j, i + 1]]
                        - 8.0 * self.values[[j, i - 1]]
                        + self.values[[j, i - 2]])
                        / (12.0 * self.grid.dx)
                } else if i >= 1 && i < nx - 1 {
                    (self.values[[j, i + 1]] - self.values[[j, i - 1]]) / (2.0 * self.grid.dx)
                } else if i == 0 {
                    (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx
                } else {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx
                }
            }
        };

        let dvdy = match stencil {
            DerivativeStencil::Central => {
                if j == 0 {
                    (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy
                } else if j == ny - 1 {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy
                } else {
                    (self.values[[j + 1, i]] - self.values[[j - 1, i]]) / (2.0 * self.grid.dy)
                }
            }
            DerivativeStencil::Forward => {
                if j < ny - 1 {
                    (self.values[[j + 1, i]] - self.values[[j, i]]) / self.grid.dy
                } else {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy
                }
            }
            DerivativeStencil::Backward => {
                if j > 0 {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy
                } else {
                    (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy
                }
            }
            DerivativeStencil::Central4 => {
                if j >= 2 && j < ny - 2 {
                    (-self.values[[j + 2, i]] + 8.0 * self.values[[j + 1, i]]
                        - 8.0 * self.values[[j - 1, i]]
                        + self.values[[j - 2, i]])
                        / (12.0 * self.grid.dy)
                } else if j >= 1 && j < ny - 1 {
                    (self.values[[j + 1, i]] - self.values[[j - 1, i]]) / (2.0 * self.grid.dy)
                } else if j == 0 {
                    (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy
                } else {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy
                }
            }
            DerivativeStencil::Central6 => {
                if j >= 3 && j < ny - 3 {
                    (self.values[[j + 3, i]] - 9.0 * self.values[[j + 2, i]]
                        + 45.0 * self.values[[j + 1, i]]
                        - 45.0 * self.values[[j - 1, i]]
                        + 9.0 * self.values[[j - 2, i]]
                        - self.values[[j - 3, i]])
                        / (60.0 * self.grid.dy)
                } else if j >= 2 && j < ny - 2 {
                    (-self.values[[j + 2, i]] + 8.0 * self.values[[j + 1, i]]
                        - 8.0 * self.values[[j - 1, i]]
                        + self.values[[j - 2, i]])
                        / (12.0 * self.grid.dy)
                } else if j >= 1 && j < ny - 1 {
                    (self.values[[j + 1, i]] - self.values[[j - 1, i]]) / (2.0 * self.grid.dy)
                } else if j == 0 {
                    (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy
                } else {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy
                }
            }
        };

        Ok(Point2D::new(dvdx, dvdy))
    }

    /// Compute gradient at arbitrary point using interpolation
    ///
    /// # Arguments
    /// * `p` - point coordinate for gradient calculation
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn gradient_interp(
        &self,
        p: &Point2D,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<Point2D> {
        let stencil = stencil.into().unwrap_or_default();
        let dx = self.grid.dx;
        let dy = self.grid.dy;

        let dvdx = match stencil {
            DerivativeStencil::Central => {
                (self.interpolate(&Point2D::new(p.x + dx, p.y))
                    - self.interpolate(&Point2D::new(p.x - dx, p.y)))
                    / (2.0 * dx)
            }
            DerivativeStencil::Forward => {
                (self.interpolate(&Point2D::new(p.x + dx, p.y)) - self.interpolate(p)) / dx
            }
            DerivativeStencil::Backward => {
                (self.interpolate(p) - self.interpolate(&Point2D::new(p.x - dx, p.y))) / dx
            }
            DerivativeStencil::Central4 => {
                let f_p2 = self.interpolate(&Point2D::new(p.x + 2.0 * dx, p.y));
                let f_p1 = self.interpolate(&Point2D::new(p.x + dx, p.y));
                let f_m1 = self.interpolate(&Point2D::new(p.x - dx, p.y));
                let f_m2 = self.interpolate(&Point2D::new(p.x - 2.0 * dx, p.y));
                (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * dx)
            }
            DerivativeStencil::Central6 => {
                let f_p3 = self.interpolate(&Point2D::new(p.x + 3.0 * dx, p.y));
                let f_p2 = self.interpolate(&Point2D::new(p.x + 2.0 * dx, p.y));
                let f_p1 = self.interpolate(&Point2D::new(p.x + dx, p.y));
                let f_m1 = self.interpolate(&Point2D::new(p.x - dx, p.y));
                let f_m2 = self.interpolate(&Point2D::new(p.x - 2.0 * dx, p.y));
                let f_m3 = self.interpolate(&Point2D::new(p.x - 3.0 * dx, p.y));
                (f_p3 - 9.0 * f_p2 + 45.0 * f_p1 - 45.0 * f_m1 + 9.0 * f_m2 - f_m3) / (60.0 * dx)
            }
        };

        let dvdy = match stencil {
            DerivativeStencil::Central => {
                (self.interpolate(&Point2D::new(p.x, p.y + dy))
                    - self.interpolate(&Point2D::new(p.x, p.y - dy)))
                    / (2.0 * dy)
            }
            DerivativeStencil::Forward => {
                (self.interpolate(&Point2D::new(p.x, p.y + dy)) - self.interpolate(p)) / dy
            }
            DerivativeStencil::Backward => {
                (self.interpolate(p) - self.interpolate(&Point2D::new(p.x, p.y - dy))) / dy
            }
            DerivativeStencil::Central4 => {
                let f_p2 = self.interpolate(&Point2D::new(p.x, p.y + 2.0 * dy));
                let f_p1 = self.interpolate(&Point2D::new(p.x, p.y + dy));
                let f_m1 = self.interpolate(&Point2D::new(p.x, p.y - dy));
                let f_m2 = self.interpolate(&Point2D::new(p.x, p.y - 2.0 * dy));
                (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * dy)
            }
            DerivativeStencil::Central6 => {
                let f_p3 = self.interpolate(&Point2D::new(p.x, p.y + 3.0 * dy));
                let f_p2 = self.interpolate(&Point2D::new(p.x, p.y + 2.0 * dy));
                let f_p1 = self.interpolate(&Point2D::new(p.x, p.y + dy));
                let f_m1 = self.interpolate(&Point2D::new(p.x, p.y - dy));
                let f_m2 = self.interpolate(&Point2D::new(p.x, p.y - 2.0 * dy));
                let f_m3 = self.interpolate(&Point2D::new(p.x, p.y - 3.0 * dy));
                (f_p3 - 9.0 * f_p2 + 45.0 * f_p1 - 45.0 * f_m1 + 9.0 * f_m2 - f_m3) / (60.0 * dy)
            }
        };

        Ok(Point2D::new(dvdx, dvdy))
    }

    /// Compute gradient field over the entire grid
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn gradient_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<VectorField2D> {
        self.gradient_field_impl::<false>(stencil)
    }

    /// Compute gradient field over the entire grid in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn par_gradient_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<VectorField2D> {
        self.gradient_field_impl::<true>(stencil)
    }

    /// Base implementation to compute gradient fields (parallel or sequential)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    fn gradient_field_impl<const PAR: bool>(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<VectorField2D> {
        let stencil = stencil.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx = 1.0 / self.grid.dx;
        let inv_dy = 1.0 / self.grid.dy;

        let mut dvdx = Array2::zeros((ny, nx));
        let mut dvdy = Array2::zeros((ny, nx));
        let v = self.values.as_standard_layout();
        let v = v.view();

        if nx == 0 || ny == 0 {
            return Ok((dvdx, dvdy));
        }

        match stencil {
            DerivativeStencil::Central => {
                let inv_2dx = 0.5 * inv_dx;
                let inv_2dy = 0.5 * inv_dy;
                // interior
                if nx >= 3 {
                    let mut out = dvdx.slice_mut(s![.., 1..nx - 1]);
                    let vr = v.slice(s![.., 2..nx]);
                    let vl = v.slice(s![.., 0..nx - 2]);
                    zip3_for_each!(PAR, &mut out, vr, vl, |o, &r, &l| *o = (r - l) * inv_2dx);
                }
                if ny >= 3 {
                    let mut out = dvdy.slice_mut(s![1..ny - 1, ..]);
                    let vb = v.slice(s![2..ny, ..]);
                    let vt = v.slice(s![0..ny - 2, ..]);
                    zip3_for_each!(PAR, &mut out, vb, vt, |o, &b, &t| *o = (b - t) * inv_2dy);
                }
                // boundary fallbacks (forward / backward)
                if nx >= 2 {
                    {
                        let mut out = dvdx.slice_mut(s![.., 0..1]);
                        let right = v.slice(s![.., 1..2]);
                        let left = v.slice(s![.., 0..1]);
                        zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                            (rr - ll) * inv_dx);
                    }
                    {
                        let mut out = dvdx.slice_mut(s![.., nx - 1..nx]);
                        let right = v.slice(s![.., nx - 1..nx]);
                        let left = v.slice(s![.., nx - 2..nx - 1]);
                        zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                            (rr - ll) * inv_dx);
                    }
                }
                if ny >= 2 {
                    {
                        let mut out = dvdy.slice_mut(s![0..1, ..]);
                        let bottom = v.slice(s![1..2, ..]);
                        let top = v.slice(s![0..1, ..]);
                        zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                            (bb - tt) * inv_dy);
                    }
                    {
                        let mut out = dvdy.slice_mut(s![ny - 1..ny, ..]);
                        let bottom = v.slice(s![ny - 1..ny, ..]);
                        let top = v.slice(s![ny - 2..ny - 1, ..]);
                        zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                            (bb - tt) * inv_dy);
                    }
                }
            }
            DerivativeStencil::Forward => {
                if nx >= 2 {
                    // interior
                    {
                        let mut out = dvdx.slice_mut(s![.., 0..nx - 1]);
                        let right = v.slice(s![.., 1..nx]);
                        let left = v.slice(s![.., 0..nx - 1]);
                        zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                            (rr - ll) * inv_dx);
                    }
                    // boundary fallback (forward)
                    {
                        let mut out = dvdx.slice_mut(s![.., nx - 1..nx]);
                        let right = v.slice(s![.., nx - 1..nx]);
                        let left = v.slice(s![.., nx - 2..nx - 1]);
                        zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                            (rr - ll) * inv_dx);
                    }
                }

                if ny >= 2 {
                    // interior
                    {
                        let mut out = dvdy.slice_mut(s![0..ny - 1, ..]);
                        let bottom = v.slice(s![1..ny, ..]);
                        let top = v.slice(s![0..ny - 1, ..]);
                        zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                            (bb - tt) * inv_dy);
                    }
                    // boundary fallback (forward)
                    {
                        let mut out = dvdy.slice_mut(s![ny - 1..ny, ..]);
                        let bottom = v.slice(s![ny - 1..ny, ..]);
                        let top = v.slice(s![ny - 2..ny - 1, ..]);
                        zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                            (bb - tt) * inv_dy);
                    }
                }
            }
            DerivativeStencil::Backward => {
                if nx >= 2 {
                    // interior
                    {
                        let mut out = dvdx.slice_mut(s![.., 1..nx]);
                        let right = v.slice(s![.., 1..nx]);
                        let left = v.slice(s![.., 0..nx - 1]);
                        zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                            (rr - ll) * inv_dx);
                    }
                    // boundary fallback (backward)
                    {
                        let mut out = dvdx.slice_mut(s![.., 0..1]);
                        let right = v.slice(s![.., 1..2]);
                        let left = v.slice(s![.., 0..1]);
                        zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                            (rr - ll) * inv_dx);
                    }
                }

                if ny >= 2 {
                    // interior
                    {
                        let mut out = dvdy.slice_mut(s![1..ny, ..]);
                        let bottom = v.slice(s![1..ny, ..]);
                        let top = v.slice(s![0..ny - 1, ..]);
                        zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                            (bb - tt) * inv_dy);
                    }
                    // boundary fallback (backward)
                    {
                        let mut out = dvdy.slice_mut(s![0..1, ..]);
                        let bottom = v.slice(s![1..2, ..]);
                        let top = v.slice(s![0..1, ..]);
                        zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                            (bb - tt) * inv_dy);
                    }
                }
            }
            DerivativeStencil::Central4 => {
                let inv_2dx = 0.5 * inv_dx;
                let inv_2dy = 0.5 * inv_dy;
                let c1x = (2.0 / 3.0) * inv_dx;
                let c2x = inv_dx / 12.0;
                let c1y = (2.0 / 3.0) * inv_dy;
                let c2y = inv_dy / 12.0;

                // interior
                if nx >= 5 {
                    let mut out = dvdx.slice_mut(s![.., 2..nx - 2]);
                    let r1 = v.slice(s![.., 3..nx - 1]);
                    let l1 = v.slice(s![.., 1..nx - 3]);
                    let r2 = v.slice(s![.., 4..nx]);
                    let l2 = v.slice(s![.., 0..nx - 4]);
                    zip5_for_each!(
                        PAR,
                        &mut out,
                        r1,
                        l1,
                        r2,
                        l2,
                        |o, &rr1, &ll1, &rr2, &ll2| {
                            *o = c1x * (rr1 - ll1) - c2x * (rr2 - ll2);
                        }
                    );
                }
                if ny >= 5 {
                    let mut out = dvdy.slice_mut(s![2..ny - 2, ..]);
                    let b1 = v.slice(s![3..ny - 1, ..]);
                    let t1 = v.slice(s![1..ny - 3, ..]);
                    let b2 = v.slice(s![4..ny, ..]);
                    let t2 = v.slice(s![0..ny - 4, ..]);
                    zip5_for_each!(
                        PAR,
                        &mut out,
                        b1,
                        t1,
                        b2,
                        t2,
                        |o, &bb1, &tt1, &bb2, &tt2| {
                            *o = c1y * (bb1 - tt1) - c2y * (bb2 - tt2);
                        }
                    );
                }
                // boundary fallbacks (forward, 2nd central, backward)
                if nx >= 2 {
                    {
                        let mut out = dvdx.slice_mut(s![.., 0..1]);
                        let right = v.slice(s![.., 1..2]);
                        let left = v.slice(s![.., 0..1]);
                        zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                            (rr - ll) * inv_dx);
                    }
                    if nx >= 3 {
                        {
                            let mut out = dvdx.slice_mut(s![.., 1..2]);
                            let right = v.slice(s![.., 2..3]);
                            let left = v.slice(s![.., 0..1]);
                            zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                                (rr - ll) * inv_2dx);
                        }
                        {
                            let mut out = dvdx.slice_mut(s![.., nx - 2..nx - 1]);
                            let right = v.slice(s![.., nx - 1..nx]);
                            let left = v.slice(s![.., nx - 3..nx - 2]);
                            zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                                (rr - ll) * inv_2dx);
                        }
                    }
                    {
                        let mut out = dvdx.slice_mut(s![.., nx - 1..nx]);
                        let right = v.slice(s![.., nx - 1..nx]);
                        let left = v.slice(s![.., nx - 2..nx - 1]);
                        zip3_for_each!(PAR, &mut out, right, left, |o, &rr, &ll| *o =
                            (rr - ll) * inv_dx);
                    }
                }
                if ny >= 2 {
                    {
                        let mut out = dvdy.slice_mut(s![0..1, ..]);
                        let bottom = v.slice(s![1..2, ..]);
                        let top = v.slice(s![0..1, ..]);
                        zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                            (bb - tt) * inv_dy);
                    }
                    if ny >= 3 {
                        {
                            let mut out = dvdy.slice_mut(s![1..2, ..]);
                            let bottom = v.slice(s![2..3, ..]);
                            let top = v.slice(s![0..1, ..]);
                            zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                                (bb - tt) * inv_2dy);
                        }
                        {
                            let mut out = dvdy.slice_mut(s![ny - 2..ny - 1, ..]);
                            let bottom = v.slice(s![ny - 1..ny, ..]);
                            let top = v.slice(s![ny - 3..ny - 2, ..]);
                            zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                                (bb - tt) * inv_2dy);
                        }
                    }
                    {
                        let mut out = dvdy.slice_mut(s![ny - 1..ny, ..]);
                        let bottom = v.slice(s![ny - 1..ny, ..]);
                        let top = v.slice(s![ny - 2..ny - 1, ..]);
                        zip3_for_each!(PAR, &mut out, bottom, top, |o, &bb, &tt| *o =
                            (bb - tt) * inv_dy);
                    }
                }
            }
            DerivativeStencil::Central6 => {
                let inv_2dx = 0.5 * inv_dx;
                let inv_2dy = 0.5 * inv_dy;
                let inv_12dx = inv_dx / 12.0;
                let inv_12dy = inv_dy / 12.0;
                let c1x = 0.75 * inv_dx;
                let c2x = 0.15 * inv_dx;
                let c3x = 0.2 * inv_12dx;
                let c1y = 0.75 * inv_dy;
                let c2y = 0.15 * inv_dy;
                let c3y = 0.2 * inv_12dy;

                // interior
                if nx >= 7 {
                    let mut out = dvdx.slice_mut(s![.., 3..nx - 3]);
                    // Note: Zip arity limit is 5
                    //     (future versions might manage a single Zip pass)
                    {
                        let r1 = v.slice(s![.., 4..nx - 2]);
                        let l1 = v.slice(s![.., 2..nx - 4]);
                        zip3_for_each!(PAR, &mut out, r1, l1, |o, &rr1, &ll1| *o =
                            c1x * (rr1 - ll1));
                    }
                    {
                        let r2 = v.slice(s![.., 5..nx - 1]);
                        let l2 = v.slice(s![.., 1..nx - 5]);
                        zip3_for_each!(PAR, &mut out, r2, l2, |o, &rr2, &ll2| *o +=
                            -c2x * (rr2 - ll2));
                    }

                    {
                        let r3 = v.slice(s![.., 6..nx]);
                        let l3 = v.slice(s![.., 0..nx - 6]);
                        zip3_for_each!(PAR, &mut out, r3, l3, |o, &rr3, &ll3| *o +=
                            c3x * (rr3 - ll3));
                    }
                }
                if ny >= 7 {
                    let mut out = dvdy.slice_mut(s![3..ny - 3, ..]);

                    {
                        let b1 = v.slice(s![4..ny - 2, ..]);
                        let t1 = v.slice(s![2..ny - 4, ..]);
                        zip3_for_each!(PAR, &mut out, b1, t1, |o, &bb1, &tt1| *o =
                            c1y * (bb1 - tt1));
                    }
                    {
                        let b2 = v.slice(s![5..ny - 1, ..]);
                        let t2 = v.slice(s![1..ny - 5, ..]);
                        zip3_for_each!(PAR, &mut out, b2, t2, |o, &bb2, &tt2| *o +=
                            -c2y * (bb2 - tt2));
                    }
                    {
                        let b3 = v.slice(s![6..ny, ..]);
                        let t3 = v.slice(s![0..ny - 6, ..]);
                        zip3_for_each!(PAR, &mut out, b3, t3, |o, &bb3, &tt3| *o +=
                            c3y * (bb3 - tt3));
                    }
                }

                // boundary fallbacks on x axis
                //   (forward, 2nd central, 4th/2nd central2,
                //    4th/2nd central, 2nd central, backward)
                if nx >= 2 {
                    {
                        let mut out = dvdx.slice_mut(s![.., 0..1]);
                        let r = v.slice(s![.., 1..2]);
                        let l = v.slice(s![.., 0..1]);
                        zip3_for_each!(PAR, &mut out, r, l, |o, &rr, &ll| *o = (rr - ll) * inv_dx);
                    }
                    if nx >= 3 {
                        {
                            let mut out = dvdx.slice_mut(s![.., 1..2]);
                            let r = v.slice(s![.., 2..3]);
                            let l = v.slice(s![.., 0..1]);
                            zip3_for_each!(PAR, &mut out, r, l, |o, &rr, &ll| *o =
                                (rr - ll) * inv_2dx);
                        }
                        {
                            let mut out = dvdx.slice_mut(s![.., nx - 2..nx - 1]);
                            let r = v.slice(s![.., nx - 1..nx]);
                            let l = v.slice(s![.., nx - 3..nx - 2]);
                            zip3_for_each!(PAR, &mut out, r, l, |o, &rr, &ll| *o =
                                (rr - ll) * inv_2dx);
                        }
                    }
                    if nx > 4 {
                        {
                            let mut out = dvdx.slice_mut(s![.., 2..3]);
                            let l3 = v.slice(s![.., 3..4]);
                            let l1 = v.slice(s![.., 1..2]);
                            let l4 = v.slice(s![.., 4..5]);
                            let l0 = v.slice(s![.., 0..1]);
                            zip5_for_each!(
                                PAR,
                                &mut out,
                                l3,
                                l1,
                                l4,
                                l0,
                                |o, &ll3, &ll1, &ll4, &ll0| {
                                    *o = (8.0 * (ll3 - ll1) - (ll4 - ll0)) * inv_12dx;
                                }
                            );
                        }
                        {
                            let mut out = dvdx.slice_mut(s![.., nx - 3..nx - 2]);
                            let r1 = v.slice(s![.., nx - 2..nx - 1]);
                            let r3 = v.slice(s![.., nx - 4..nx - 3]);
                            let r0 = v.slice(s![.., nx - 1..nx]);
                            let r4 = v.slice(s![.., nx - 5..nx - 4]);
                            zip5_for_each!(
                                PAR,
                                &mut out,
                                r1,
                                r3,
                                r0,
                                r4,
                                |o, &rr1, &rr3, &rr0, &rr4| {
                                    *o = (8.0 * (rr1 - rr3) - (rr0 - rr4)) * inv_12dx;
                                }
                            );
                        }
                    } else if nx >= 4 {
                        {
                            let mut out = dvdx.slice_mut(s![.., 2..3]);
                            let r = v.slice(s![.., 3..4]);
                            let l = v.slice(s![.., 1..2]);
                            zip3_for_each!(PAR, &mut out, r, l, |o, &rr, &ll| *o =
                                (rr - ll) * inv_2dx);
                        }
                        {
                            let mut out = dvdx.slice_mut(s![.., nx - 3..nx - 2]);
                            let r = v.slice(s![.., nx - 2..nx - 1]);
                            let l = v.slice(s![.., nx - 4..nx - 3]);
                            zip3_for_each!(PAR, &mut out, r, l, |o, &rr, &ll| *o =
                                (rr - ll) * inv_2dx);
                        }
                    }
                    {
                        let mut out = dvdx.slice_mut(s![.., nx - 1..nx]);
                        let r = v.slice(s![.., nx - 1..nx]);
                        let l = v.slice(s![.., nx - 2..nx - 1]);
                        zip3_for_each!(PAR, &mut out, r, l, |o, &rr, &ll| *o = (rr - ll) * inv_dx);
                    }
                }

                // boundary fallbacks on y axis
                //   (forward, 2nd central, 4th/2nd central2,
                //    4th/2nd central, 2nd central, backward)
                if ny >= 2 {
                    {
                        let mut out = dvdy.slice_mut(s![0..1, ..]);
                        let b = v.slice(s![1..2, ..]);
                        let t = v.slice(s![0..1, ..]);
                        zip3_for_each!(PAR, &mut out, b, t, |o, &bb, &tt| *o = (bb - tt) * inv_dy);
                    }
                    if ny >= 3 {
                        {
                            let mut out = dvdy.slice_mut(s![1..2, ..]);
                            let b = v.slice(s![2..3, ..]);
                            let t = v.slice(s![0..1, ..]);
                            zip3_for_each!(PAR, &mut out, b, t, |o, &bb, &tt| *o =
                                (bb - tt) * inv_2dy);
                        }
                        {
                            let mut out = dvdy.slice_mut(s![ny - 2..ny - 1, ..]);
                            let b = v.slice(s![ny - 1..ny, ..]);
                            let t = v.slice(s![ny - 3..ny - 2, ..]);
                            zip3_for_each!(PAR, &mut out, b, t, |o, &bb, &tt| *o =
                                (bb - tt) * inv_2dy);
                        }
                    }
                    if ny > 4 {
                        {
                            let mut out = dvdy.slice_mut(s![2..3, ..]);
                            let t3 = v.slice(s![3..4, ..]);
                            let t1 = v.slice(s![1..2, ..]);
                            let t4 = v.slice(s![4..5, ..]);
                            let t0 = v.slice(s![0..1, ..]);
                            zip5_for_each!(
                                PAR,
                                &mut out,
                                t3,
                                t1,
                                t4,
                                t0,
                                |o, &tt3, &tt1, &tt4, &tt0| {
                                    *o = (8.0 * (tt3 - tt1) - (tt4 - tt0)) * inv_12dy;
                                }
                            );
                        }
                        {
                            let mut out = dvdy.slice_mut(s![ny - 3..ny - 2, ..]);
                            let b1 = v.slice(s![ny - 2..ny - 1, ..]);
                            let b3 = v.slice(s![ny - 4..ny - 3, ..]);
                            let b0 = v.slice(s![ny - 1..ny, ..]);
                            let b4 = v.slice(s![ny - 5..ny - 4, ..]);
                            zip5_for_each!(
                                PAR,
                                &mut out,
                                b1,
                                b3,
                                b0,
                                b4,
                                |o, &bb1, &bb3, &bb0, &bb4| {
                                    *o = (8.0 * (bb1 - bb3) - (bb0 - bb4)) * inv_12dy;
                                }
                            );
                        }
                    } else if ny >= 4 {
                        {
                            let mut out = dvdy.slice_mut(s![2..3, ..]);
                            let b = v.slice(s![3..4, ..]);
                            let t = v.slice(s![1..2, ..]);
                            zip3_for_each!(PAR, &mut out, b, t, |o, &bb, &tt| *o =
                                (bb - tt) * inv_2dy);
                        }
                        {
                            let mut out = dvdy.slice_mut(s![ny - 3..ny - 2, ..]);
                            let b = v.slice(s![ny - 2..ny - 1, ..]);
                            let t = v.slice(s![ny - 4..ny - 3, ..]);
                            zip3_for_each!(PAR, &mut out, b, t, |o, &bb, &tt| *o =
                                (bb - tt) * inv_2dy);
                        }
                    }
                    {
                        let mut out = dvdy.slice_mut(s![ny - 1..ny, ..]);
                        let b = v.slice(s![ny - 1..ny, ..]);
                        let t = v.slice(s![ny - 2..ny - 1, ..]);
                        zip3_for_each!(PAR, &mut out, b, t, |o, &bb, &tt| *o = (bb - tt) * inv_dy);
                    }
                }
            }
        }

        Ok((dvdx, dvdy))
    }

    /// Compute gradient magnitude of the entire grid
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn gradient_magnitude(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FieldArray2D> {
        self.gradient_magnitude_impl::<false>(stencil)
    }

    /// Compute gradient magnitude of the entire grid in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn par_gradient_magnitude(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FieldArray2D> {
        self.gradient_magnitude_impl::<true>(stencil)
    }

    fn gradient_magnitude_impl<const PAR: bool>(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FieldArray2D> {
        let stencil = stencil.into().unwrap_or_default();
        let (dvdx, dvdy) = self.gradient_field_impl::<PAR>(stencil)?;
        let mut magnitude = Array2::zeros((self.grid.ny, self.grid.nx));
        zip3_for_each!(PAR, &mut magnitude, &dvdx, &dvdy, |m, &gx, &gy| {
            *m = (gx * gx + gy * gy).sqrt();
        });
        Ok(magnitude)
    }

    /// Compute Laplacian at a pixel (i, j) using 5-point stencil
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    #[inline]
    pub fn laplacian(
        &self,
        i: usize,
        j: usize,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<f64> {
        let stencil = stencil.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let v_ij = self.values[[j, i]];

        let d2vdx2 = match stencil {
            DerivativeStencil::Central => {
                if i > 0 && i < nx - 1 {
                    (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) / dx2
                } else if i == 0 && nx > 2 {
                    (self.values[[j, 2]] - 2.0 * self.values[[j, 1]] + v_ij) / dx2
                } else if i == nx - 1 && nx > 2 {
                    (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]]) / dx2
                } else {
                    0.0
                }
            }
            DerivativeStencil::Forward => {
                if i + 2 < nx {
                    (self.values[[j, i + 2]] - 2.0 * self.values[[j, i + 1]] + v_ij) / dx2
                } else if i + 1 < nx && i > 0 {
                    (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) / dx2
                } else if i > 1 {
                    (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]]) / dx2
                } else {
                    0.0
                }
            }
            DerivativeStencil::Backward => {
                if i >= 2 {
                    (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]]) / dx2
                } else if i > 0 && i + 1 < nx {
                    (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) / dx2
                } else if i + 2 < nx {
                    (self.values[[j, i + 2]] - 2.0 * self.values[[j, i + 1]] + v_ij) / dx2
                } else {
                    0.0
                }
            }
            DerivativeStencil::Central4 => {
                if i >= 2 && i < nx - 2 {
                    (-self.values[[j, i + 2]] + 16.0 * self.values[[j, i + 1]] - 30.0 * v_ij
                        + 16.0 * self.values[[j, i - 1]]
                        - self.values[[j, i - 2]])
                        / (12.0 * dx2)
                } else if i > 0 && i < nx - 1 {
                    // fall back to central
                    (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) / dx2
                } else {
                    0.0
                }
            }
            DerivativeStencil::Central6 => {
                if i >= 3 && i < nx - 3 {
                    (2.0 * self.values[[j, i + 3]] - 27.0 * self.values[[j, i + 2]]
                        + 270.0 * self.values[[j, i + 1]]
                        - 490.0 * v_ij
                        + 270.0 * self.values[[j, i - 1]]
                        - 27.0 * self.values[[j, i - 2]]
                        + 2.0 * self.values[[j, i - 3]])
                        / (180.0 * dx2)
                } else if i >= 2 && i < nx - 2 {
                    // fall back to 4-th order
                    (-self.values[[j, i + 2]] + 16.0 * self.values[[j, i + 1]] - 30.0 * v_ij
                        + 16.0 * self.values[[j, i - 1]]
                        - self.values[[j, i - 2]])
                        / (12.0 * dx2)
                } else if i > 0 && i < nx - 1 {
                    // fall back to central
                    (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) / dx2
                } else {
                    0.0
                }
            }
        };

        let d2vdy2 = match stencil {
            DerivativeStencil::Central => {
                if j > 0 && j < ny - 1 {
                    (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) / dy2
                } else if j == 0 && ny > 2 {
                    (self.values[[2, i]] - 2.0 * self.values[[1, i]] + v_ij) / dy2
                } else if j == ny - 1 && ny > 2 {
                    (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) / dy2
                } else {
                    0.0
                }
            }
            DerivativeStencil::Forward => {
                if j + 2 < ny {
                    (self.values[[j + 2, i]] - 2.0 * self.values[[j + 1, i]] + v_ij) / dy2
                } else if j + 1 < ny && j > 0 {
                    (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) / dy2
                } else if j > 1 {
                    (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) / dy2
                } else {
                    0.0
                }
            }
            DerivativeStencil::Backward => {
                if j + 2 < ny {
                    (self.values[[j + 2, i]] - 2.0 * self.values[[j + 1, i]] + v_ij) / dy2
                } else if j + 1 < ny && j > 0 {
                    (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) / dy2
                } else if j > 1 {
                    (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) / dy2
                } else {
                    0.0
                }
            }
            DerivativeStencil::Central4 => {
                if j >= 2 && j < ny - 2 {
                    (-self.values[[j + 2, i]] + 16.0 * self.values[[j + 1, i]] - 30.0 * v_ij
                        + 16.0 * self.values[[j - 1, i]]
                        - self.values[[j - 2, i]])
                        / (12.0 * dy2)
                } else if j > 0 && j < ny - 1 {
                    (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) / dy2
                } else {
                    0.0
                }
            }
            DerivativeStencil::Central6 => {
                if j >= 3 && j < ny - 3 {
                    (2.0 * self.values[[j + 3, i]] - 27.0 * self.values[[j + 2, i]]
                        + 270.0 * self.values[[j + 1, i]]
                        - 490.0 * v_ij
                        + 270.0 * self.values[[j - 1, i]]
                        - 27.0 * self.values[[j - 2, i]]
                        + 2.0 * self.values[[j - 3, i]])
                        / (180.0 * dy2)
                } else if j >= 2 && j < ny - 2 {
                    (-self.values[[j + 2, i]] + 16.0 * self.values[[j + 1, i]] - 30.0 * v_ij
                        + 16.0 * self.values[[j - 1, i]]
                        - self.values[[j - 2, i]])
                        / (12.0 * dy2)
                } else if j > 0 && j < ny - 1 {
                    (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) / dy2
                } else {
                    0.0
                }
            }
        };

        Ok(d2vdx2 + d2vdy2)
    }

    /// Compute Laplacian at arbitrary point using interpolation
    ///
    /// # Arguments
    /// * `p` - point coordinate for gradient calculation
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn laplacian_interp(
        &self,
        p: &Point2D,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<f64> {
        let stencil = stencil.into().unwrap_or_default();
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let v_00 = self.interpolate(p);

        let d2vdx2 = match stencil {
            // 3-point
            DerivativeStencil::Central => {
                let v_mx = self.interpolate(&Point2D::new(p.x - dx, p.y));
                let v_px = self.interpolate(&Point2D::new(p.x + dx, p.y));
                (v_px - 2.0 * v_00 + v_mx) / dx2
            }
            DerivativeStencil::Forward => {
                let v_px = self.interpolate(&Point2D::new(p.x + dx, p.y));
                let v_2px = self.interpolate(&Point2D::new(p.x + 2.0 * dx, p.y));
                (v_2px - 2.0 * v_px + v_00) / dx2
            }
            DerivativeStencil::Backward => {
                let v_mx = self.interpolate(&Point2D::new(p.x - dx, p.y));
                let v_2mx = self.interpolate(&Point2D::new(p.x - 2.0 * dx, p.y));
                (v_00 - 2.0 * v_mx + v_2mx) / dx2
            }
            DerivativeStencil::Central4 => {
                // 5-point
                let v_00 = self.interpolate(p);
                let v_p1 = self.interpolate(&Point2D::new(p.x + dx, p.y));
                let v_p2 = self.interpolate(&Point2D::new(p.x + 2.0 * dx, p.y));
                let v_m1 = self.interpolate(&Point2D::new(p.x - dx, p.y));
                let v_m2 = self.interpolate(&Point2D::new(p.x - 2.0 * dx, p.y));
                (-v_p2 + 16.0 * v_p1 - 30.0 * v_00 + 16.0 * v_m1 - v_m2) / (12.0 * dx2)
            }
            DerivativeStencil::Central6 => {
                // 7-point
                let v_00 = self.interpolate(p);
                let v_p1 = self.interpolate(&Point2D::new(p.x + dx, p.y));
                let v_p2 = self.interpolate(&Point2D::new(p.x + 2.0 * dx, p.y));
                let v_p3 = self.interpolate(&Point2D::new(p.x + 3.0 * dx, p.y));
                let v_m1 = self.interpolate(&Point2D::new(p.x - dx, p.y));
                let v_m2 = self.interpolate(&Point2D::new(p.x - 2.0 * dx, p.y));
                let v_m3 = self.interpolate(&Point2D::new(p.x - 3.0 * dx, p.y));
                (2.0 * v_p3 - 27.0 * v_p2 + 270.0 * v_p1 - 490.0 * v_00 + 270.0 * v_m1
                    - 27.0 * v_m2
                    + 2.0 * v_m3)
                    / (180.0 * dx2)
            }
        };

        let d2vdy2 = match stencil {
            // 3-point
            DerivativeStencil::Central => {
                let v_my = self.interpolate(&Point2D::new(p.x, p.y - dy));
                let v_py = self.interpolate(&Point2D::new(p.x, p.y + dy));
                (v_py - 2.0 * v_00 + v_my) / dy2
            }
            DerivativeStencil::Forward => {
                let v_py = self.interpolate(&Point2D::new(p.x, p.y + dy));
                let v_2py = self.interpolate(&Point2D::new(p.x, p.y + 2.0 * dy));
                (v_2py - 2.0 * v_py + v_00) / dy2
            }
            DerivativeStencil::Backward => {
                let v_my = self.interpolate(&Point2D::new(p.x, p.y - dy));
                let v_2my = self.interpolate(&Point2D::new(p.x, p.y - 2.0 * dy));
                (v_00 - 2.0 * v_my + v_2my) / dy2
            }
            // 5-point
            DerivativeStencil::Central4 => {
                let v_00 = self.interpolate(p);
                let v_p1 = self.interpolate(&Point2D::new(p.x, p.y + dy));
                let v_p2 = self.interpolate(&Point2D::new(p.x, p.y + 2.0 * dy));
                let v_m1 = self.interpolate(&Point2D::new(p.x, p.y - dy));
                let v_m2 = self.interpolate(&Point2D::new(p.x, p.y - 2.0 * dy));
                (-v_p2 + 16.0 * v_p1 - 30.0 * v_00 + 16.0 * v_m1 - v_m2) / (12.0 * dy2)
            }
            // 8-point
            DerivativeStencil::Central6 => {
                let v_00 = self.interpolate(p);
                let v_p1 = self.interpolate(&Point2D::new(p.x, p.y + dy));
                let v_p2 = self.interpolate(&Point2D::new(p.x, p.y + 2.0 * dy));
                let v_p3 = self.interpolate(&Point2D::new(p.x, p.y + 3.0 * dy));
                let v_m1 = self.interpolate(&Point2D::new(p.x, p.y - dy));
                let v_m2 = self.interpolate(&Point2D::new(p.x, p.y - 2.0 * dy));
                let v_m3 = self.interpolate(&Point2D::new(p.x, p.y - 3.0 * dy));
                (2.0 * v_p3 - 27.0 * v_p2 + 270.0 * v_p1 - 490.0 * v_00 + 270.0 * v_m1
                    - 27.0 * v_m2
                    + 2.0 * v_m3)
                    / (180.0 * dy2)
            }
        };

        Ok(d2vdx2 + d2vdy2)
    }

    /// Base implementation to compute Laplacian fields (parallel or sequential)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    fn laplacian_field_impl<const PAR: bool>(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FieldArray2D> {
        let stencil = stencil.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx2 = 1.0 / (self.grid.dx * self.grid.dx);
        let inv_dy2 = 1.0 / (self.grid.dy * self.grid.dy);

        let mut laplace = Array2::zeros((ny, nx));
        if nx == 0 || ny == 0 {
            return Ok(laplace);
        }

        let v = self.values.as_standard_layout();
        let v = v.view();

        match stencil {
            DerivativeStencil::Central => {
                // interior
                if nx >= 3 && ny >= 3 {
                    {
                        let mut out = laplace.slice_mut(s![1..ny - 1, 1..nx - 1]);
                        let c = v.slice(s![1..ny - 1, 1..nx - 1]);
                        let xl = v.slice(s![1..ny - 1, 0..nx - 2]);
                        let xr = v.slice(s![1..ny - 1, 2..nx]);
                        zip4_for_each!(PAR, &mut out, xr, c, xl, |o, &r, &cc, &l| {
                            *o = (r - 2.0 * cc + l) * inv_dx2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![1..ny - 1, 1..nx - 1]);
                        let c = v.slice(s![1..ny - 1, 1..nx - 1]);
                        let yt = v.slice(s![0..ny - 2, 1..nx - 1]);
                        let yb = v.slice(s![2..ny, 1..nx - 1]);
                        zip4_for_each!(PAR, &mut out, yt, c, yb, |o, &t, &cc, &b| {
                            *o += (b - 2.0 * cc + t) * inv_dy2;
                        });
                    }
                }

                // boundary fallbacks (forward, backward)
                if ny > 2 && nx > 2 {
                    {
                        let mut out = laplace.slice_mut(s![0..1, 1..nx - 1]);
                        let xc = v.slice(s![0..1, 1..nx - 1]);
                        let xl = v.slice(s![0..1, 0..nx - 2]);
                        let xr = v.slice(s![0..1, 2..nx]);
                        zip4_for_each!(PAR, &mut out, xr, xc, xl, |o, &r, &cc, &l| {
                            *o = (r - 2.0 * cc + l) * inv_dx2;
                        });
                        let y0 = v.slice(s![0..1, 1..nx - 1]);
                        let y1 = v.slice(s![1..2, 1..nx - 1]);
                        let y2 = v.slice(s![2..3, 1..nx - 1]);
                        zip4_for_each!(PAR, &mut out, y2, y1, y0, |o, &yy2, &yy1, &yy0| {
                            *o += (yy2 - 2.0 * yy1 + yy0) * inv_dy2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![ny - 1..ny, 1..nx - 1]);
                        let c = v.slice(s![ny - 1..ny, 1..nx - 1]);
                        let xl = v.slice(s![ny - 1..ny, 0..nx - 2]);
                        let xr = v.slice(s![ny - 1..ny, 2..nx]);
                        zip4_for_each!(PAR, &mut out, xr, c, xl, |o, &r, &cc, &l| {
                            *o = (r - 2.0 * cc + l) * inv_dx2;
                        });
                        let y0 = v.slice(s![ny - 3..ny - 2, 1..nx - 1]);
                        let y1 = v.slice(s![ny - 2..ny - 1, 1..nx - 1]);
                        let y2 = v.slice(s![ny - 1..ny, 1..nx - 1]);
                        zip4_for_each!(PAR, &mut out, y2, y1, y0, |o, &yy2, &yy1, &yy0| {
                            *o += (yy2 - 2.0 * yy1 + yy0) * inv_dy2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![1..ny - 1, 0..1]);
                        let c = v.slice(s![1..ny - 1, 0..1]);
                        let yt = v.slice(s![0..ny - 2, 0..1]);
                        let yb = v.slice(s![2..ny, 0..1]);
                        zip4_for_each!(PAR, &mut out, yb, c, yt, |o, &b, &cc, &t| {
                            *o = (b - 2.0 * cc + t) * inv_dy2;
                        });
                        let x0 = v.slice(s![1..ny - 1, 0..1]);
                        let x1 = v.slice(s![1..ny - 1, 1..2]);
                        let x2 = v.slice(s![1..ny - 1, 2..3]);
                        zip4_for_each!(PAR, &mut out, x2, x1, x0, |o, &xx2, &xx1, &xx0| {
                            *o += (xx2 - 2.0 * xx1 + xx0) * inv_dx2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![1..ny - 1, nx - 1..nx]);
                        let c = v.slice(s![1..ny - 1, nx - 1..nx]);
                        let yt = v.slice(s![0..ny - 2, nx - 1..nx]);
                        let yb = v.slice(s![2..ny, nx - 1..nx]);
                        zip4_for_each!(PAR, &mut out, yb, c, yt, |o, &b, &cc, &t| {
                            *o = (b - 2.0 * cc + t) * inv_dy2;
                        });
                        let x0 = v.slice(s![1..ny - 1, nx - 3..nx - 2]);
                        let x1 = v.slice(s![1..ny - 1, nx - 2..nx - 1]);
                        let x2 = v.slice(s![1..ny - 1, nx - 1..nx]);
                        zip4_for_each!(PAR, &mut out, x2, x1, x0, |o, &xx2, &xx1, &xx0| {
                            *o += (xx2 - 2.0 * xx1 + xx0) * inv_dx2;
                        });
                    }
                    if nx >= 3 && ny >= 3 {
                        let v00 = v[[0, 0]];
                        laplace[[0, 0]] = (v[[0, 2]] - 2.0 * v[[0, 1]] + v00) * inv_dx2
                            + (v[[2, 0]] - 2.0 * v[[1, 0]] + v00) * inv_dy2;

                        let v0n = v[[0, nx - 1]];
                        laplace[[0, nx - 1]] = (v0n - 2.0 * v[[0, nx - 2]] + v[[0, nx - 3]])
                            * inv_dx2
                            + (v[[2, nx - 1]] - 2.0 * v[[1, nx - 1]] + v0n) * inv_dy2;

                        let vn0 = v[[ny - 1, 0]];
                        laplace[[ny - 1, 0]] = (v[[ny - 1, 2]] - 2.0 * v[[ny - 1, 1]] + vn0)
                            * inv_dx2
                            + (vn0 - 2.0 * v[[ny - 2, 0]] + v[[ny - 3, 0]]) * inv_dy2;

                        let vnn = v[[ny - 1, nx - 1]];
                        laplace[[ny - 1, nx - 1]] =
                            (vnn - 2.0 * v[[ny - 1, nx - 2]] + v[[ny - 1, nx - 3]]) * inv_dx2
                                + (vnn - 2.0 * v[[ny - 2, nx - 1]] + v[[ny - 3, nx - 1]]) * inv_dy2;
                    }
                }
            }
            DerivativeStencil::Forward => {
                if nx >= 3 {
                    // interior along x axis
                    let mut out = laplace.slice_mut(s![.., 0..nx - 2]);
                    let x0 = v.slice(s![.., 0..nx - 2]);
                    let x1 = v.slice(s![.., 1..nx - 1]);
                    let x2 = v.slice(s![.., 2..nx]);
                    zip4_for_each!(PAR, &mut out, x2, x1, x0, |o, &xx2, &xx1, &xx0| {
                        *o = (xx2 - 2.0 * xx1 + xx0) * inv_dx2;
                    });
                    // boundary fallbacks (2nd central, backward)
                    if nx >= 3 {
                        let mut col = laplace.slice_mut(s![.., nx - 2..nx - 1]);
                        let xl = v.slice(s![.., nx - 3..nx - 2]);
                        let xc = v.slice(s![.., nx - 2..nx - 1]);
                        let xr = v.slice(s![.., nx - 1..nx]);
                        zip4_for_each!(PAR, &mut col, xr, xc, xl, |o, &r, &c, &l| {
                            *o = (r - 2.0 * c + l) * inv_dx2;
                        });
                    }
                    {
                        let mut col = laplace.slice_mut(s![.., nx - 1..nx]);
                        let x0 = v.slice(s![.., nx - 3..nx - 2]);
                        let x1 = v.slice(s![.., nx - 2..nx - 1]);
                        let x2 = v.slice(s![.., nx - 1..nx]);
                        zip4_for_each!(PAR, &mut col, x2, x1, x0, |o, &xx2, &xx1, &xx0| {
                            *o = (xx2 - 2.0 * xx1 + xx0) * inv_dx2;
                        });
                    }
                }
                if ny >= 3 {
                    // interior along y axis
                    let mut out = laplace.slice_mut(s![0..ny - 2, ..]);
                    let y0 = v.slice(s![0..ny - 2, ..]);
                    let y1 = v.slice(s![1..ny - 1, ..]);
                    let y2 = v.slice(s![2..ny, ..]);
                    zip4_for_each!(PAR, &mut out, y2, y1, y0, |o, &yy2, &yy1, &yy0| {
                        *o += (yy2 - 2.0 * yy1 + yy0) * inv_dy2;
                    });
                    // boundary fallbacks (2nd central, backward)
                    {
                        let mut row = laplace.slice_mut(s![ny - 2..ny - 1, ..]);
                        let yt = v.slice(s![ny - 3..ny - 2, ..]);
                        let yc = v.slice(s![ny - 2..ny - 1, ..]);
                        let yb = v.slice(s![ny - 1..ny, ..]);
                        zip4_for_each!(PAR, &mut row, yb, yc, yt, |o, &b, &c, &t| {
                            *o += (b - 2.0 * c + t) * inv_dy2;
                        });
                    }
                    {
                        let mut row = laplace.slice_mut(s![ny - 1..ny, ..]);
                        let y0 = v.slice(s![ny - 3..ny - 2, ..]);
                        let y1 = v.slice(s![ny - 2..ny - 1, ..]);
                        let y2 = v.slice(s![ny - 1..ny, ..]);
                        zip4_for_each!(PAR, &mut row, y2, y1, y0, |o, &yy2, &yy1, &yy0| {
                            *o += (yy2 - 2.0 * yy1 + yy0) * inv_dy2;
                        });
                    }
                }
            }
            DerivativeStencil::Backward => {
                if nx >= 3 {
                    // interior along x axis
                    let mut out = laplace.slice_mut(s![.., 2..nx]);
                    let x2 = v.slice(s![.., 2..nx]);
                    let x1 = v.slice(s![.., 1..nx - 1]);
                    let x0 = v.slice(s![.., 0..nx - 2]);
                    zip4_for_each!(PAR, &mut out, x2, x1, x0, |o, &xx2, &xx1, &xx0| {
                        *o = (xx2 - 2.0 * xx1 + xx0) * inv_dx2;
                    });
                    // boundary fallback (2nd central, forward)
                    {
                        let mut col = laplace.slice_mut(s![.., 0..1]);
                        let xl = v.slice(s![.., 0..1]);
                        let xc = v.slice(s![.., 1..2]);
                        let xr = v.slice(s![.., 2..3]);
                        zip4_for_each!(PAR, &mut col, xr, xc, xl, |o, &r, &c, &l| {
                            *o = (r - 2.0 * c + l) * inv_dx2;
                        });
                    }
                    {
                        let mut col = laplace.slice_mut(s![.., 1..2]);
                        let x0 = v.slice(s![.., 0..1]);
                        let x1 = v.slice(s![.., 1..2]);
                        let x2 = v.slice(s![.., 2..3]);
                        zip4_for_each!(PAR, &mut col, x2, x1, x0, |o, &xx2, &xx1, &xx0| {
                            *o = (xx2 - 2.0 * xx1 + xx0) * inv_dx2;
                        });
                    }
                }
                if ny >= 3 {
                    // interior along y axis
                    let mut out = laplace.slice_mut(s![2..ny, ..]);
                    let y2 = v.slice(s![2..ny, ..]);
                    let y1 = v.slice(s![1..ny - 1, ..]);
                    let y0 = v.slice(s![0..ny - 2, ..]);
                    zip4_for_each!(PAR, &mut out, y2, y1, y0, |o, &yy2, &yy1, &yy0| {
                        *o += (yy2 - 2.0 * yy1 + yy0) * inv_dy2;
                    });

                    // boundary fallbacks (2nd central, forward)
                    {
                        let mut row = laplace.slice_mut(s![0..1, ..]);
                        let yl = v.slice(s![0..1, ..]);
                        let yc = v.slice(s![1..2, ..]);
                        let yr = v.slice(s![2..3, ..]);
                        zip4_for_each!(PAR, &mut row, yr, yc, yl, |o, &r, &c, &l| {
                            *o += (r - 2.0 * c + l) * inv_dy2;
                        });
                    }
                    {
                        let mut row = laplace.slice_mut(s![1..2, ..]);
                        let y0 = v.slice(s![0..1, ..]);
                        let y1 = v.slice(s![1..2, ..]);
                        let y2 = v.slice(s![2..3, ..]);
                        zip4_for_each!(PAR, &mut row, y2, y1, y0, |o, &yy2, &yy1, &yy0| {
                            *o += (yy2 - 2.0 * yy1 + yy0) * inv_dy2;
                        });
                    }
                }
            }
            DerivativeStencil::Central4 => {
                let inv_12dx2 = inv_dx2 / 12.0;
                let inv_12dy2 = inv_dy2 / 12.0;
                // interior along x axis
                if nx >= 5 {
                    let mut out = laplace.slice_mut(s![.., 2..nx - 2]);
                    let c = v.slice(s![.., 2..nx - 2]);
                    zip3_for_each!(PAR, &mut out, c, c, |o, &cc, &_| {
                        *o = (-30.0 * cc) * inv_12dx2;
                    });
                    {
                        let p1 = v.slice(s![.., 3..nx - 1]);
                        let m1 = v.slice(s![.., 1..nx - 3]);
                        zip3_for_each!(PAR, &mut out, p1, m1, |o, &pp, &mm| {
                            *o += (16.0 * (pp + mm)) * inv_12dx2;
                        });
                    }
                    {
                        let p2 = v.slice(s![.., 4..nx]);
                        let m2 = v.slice(s![.., 0..nx - 4]);
                        zip3_for_each!(PAR, &mut out, p2, m2, |o, &pp, &mm| {
                            *o += (-(pp + mm)) * inv_12dx2;
                        });
                    }
                }
                // boundary fallbacks (2nd central, leave edge cases 0)
                if nx >= 3 {
                    {
                        let mut out = laplace.slice_mut(s![.., 1..2]);
                        let l = v.slice(s![.., 0..1]);
                        let c = v.slice(s![.., 1..2]);
                        let r = v.slice(s![.., 2..3]);
                        zip4_for_each!(PAR, &mut out, r, c, l, |o, &rr, &cc, &ll| {
                            *o += (rr - 2.0 * cc + ll) * inv_dx2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![.., nx - 2..nx - 1]);
                        let l = v.slice(s![.., nx - 3..nx - 2]);
                        let c = v.slice(s![.., nx - 2..nx - 1]);
                        let r = v.slice(s![.., nx - 1..nx]);
                        zip4_for_each!(PAR, &mut out, r, c, l, |o, &rr, &cc, &ll| {
                            *o += (rr - 2.0 * cc + ll) * inv_dx2;
                        });
                    }
                }

                // interior along y axis
                if ny >= 5 {
                    let mut out = laplace.slice_mut(s![2..ny - 2, ..]);
                    let c = v.slice(s![2..ny - 2, ..]);
                    zip3_for_each!(PAR, &mut out, c, c, |o, &cc, &_| {
                        *o += (-30.0 * cc) * inv_12dy2;
                    });
                    {
                        let p1 = v.slice(s![3..ny - 1, ..]);
                        let m1 = v.slice(s![1..ny - 3, ..]);
                        zip3_for_each!(PAR, &mut out, p1, m1, |o, &pp, &mm| {
                            *o += (16.0 * (pp + mm)) * inv_12dy2;
                        });
                    }
                    {
                        let p2 = v.slice(s![4..ny, ..]);
                        let m2 = v.slice(s![0..ny - 4, ..]);
                        zip3_for_each!(PAR, &mut out, p2, m2, |o, &pp, &mm| {
                            *o += (-(pp + mm)) * inv_12dy2;
                        });
                    }
                }
                // boundary fallbacks (2nd central, leave edge cases 0)
                if ny >= 3 {
                    {
                        let mut out = laplace.slice_mut(s![1..2, ..]);
                        let t = v.slice(s![0..1, ..]);
                        let c = v.slice(s![1..2, ..]);
                        let b = v.slice(s![2..3, ..]);
                        zip4_for_each!(PAR, &mut out, b, c, t, |o, &bb, &cc, &tt| {
                            *o += (bb - 2.0 * cc + tt) * inv_dy2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![ny - 2..ny - 1, ..]);
                        let t = v.slice(s![ny - 3..ny - 2, ..]);
                        let c = v.slice(s![ny - 2..ny - 1, ..]);
                        let b = v.slice(s![ny - 1..ny, ..]);
                        zip4_for_each!(PAR, &mut out, b, c, t, |o, &bb, &cc, &tt| {
                            *o += (bb - 2.0 * cc + tt) * inv_dy2;
                        });
                    }
                }
            }
            DerivativeStencil::Central6 => {
                let inv_12dx2 = inv_dx2 / 12.0;
                let inv_12dy2 = inv_dy2 / 12.0;
                let c3 = 2.0 / 15.0;
                let c2 = 9.0 / 5.0;
                let c0 = 98.0 / 3.0;

                let c0x = (-c0) * inv_12dx2;
                let c1x = 18.0 * inv_12dx2;
                let c2x = (-c2) * inv_12dx2;
                let c3x = c3 * inv_12dx2;
                let c0y = (-c0) * inv_12dy2;
                let c1y = 18.0 * inv_12dy2;
                let c2y = (-c2) * inv_12dy2;
                let c3y = c3 * inv_12dy2;

                // interior along x axis
                if nx >= 7 {
                    let mut out = laplace.slice_mut(s![.., 3..nx - 3]);
                    let c = v.slice(s![.., 3..nx - 3]);
                    zip3_for_each!(PAR, &mut out, c, c, |o, &cc, &_| {
                        *o = c0x * cc;
                    });
                    {
                        let p1 = v.slice(s![.., 4..nx - 2]);
                        let m1 = v.slice(s![.., 2..nx - 4]);
                        zip3_for_each!(PAR, &mut out, p1, m1, |o, &pp, &mm| {
                            *o += c1x * (pp + mm);
                        });
                    }
                    {
                        let p2 = v.slice(s![.., 5..nx - 1]);
                        let m2 = v.slice(s![.., 1..nx - 5]);
                        zip3_for_each!(PAR, &mut out, p2, m2, |o, &pp, &mm| {
                            *o += c2x * (pp + mm);
                        });
                    }
                    {
                        let p3 = v.slice(s![.., 6..nx]);
                        let m3 = v.slice(s![.., 0..nx - 6]);
                        zip3_for_each!(PAR, &mut out, p3, m3, |o, &pp, &mm| {
                            *o += c3x * (pp + mm);
                        });
                    }
                }

                // boundary fallbacks (4th central)
                if nx >= 5 {
                    {
                        let mut out = laplace.slice_mut(s![.., 2..3]);
                        let l0 = v.slice(s![.., 0..1]);
                        let l1 = v.slice(s![.., 1..2]);
                        let xc = v.slice(s![.., 2..3]);
                        let r3 = v.slice(s![.., 3..4]);
                        let r4 = v.slice(s![.., 4..5]);
                        zip3_for_each!(PAR, &mut out, xc, xc, |o, &cc, &_| {
                            *o += (-30.0 * cc) * inv_12dx2;
                        });
                        zip3_for_each!(PAR, &mut out, r3, l1, |o, &p, &m| {
                            *o += (16.0 * (p + m)) * inv_12dx2;
                        });
                        zip3_for_each!(PAR, &mut out, r4, l0, |o, &p2, &m2| {
                            *o += (-(p2 + m2)) * inv_12dx2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![.., nx - 3..nx - 2]);
                        let l0 = v.slice(s![.., nx - 5..nx - 4]);
                        let l1 = v.slice(s![.., nx - 4..nx - 3]);
                        let xc = v.slice(s![.., nx - 3..nx - 2]);
                        let r3 = v.slice(s![.., nx - 2..nx - 1]);
                        let r4 = v.slice(s![.., nx - 1..nx]);

                        zip3_for_each!(PAR, &mut out, xc, xc, |o, &cc, &_| {
                            *o += (-30.0 * cc) * inv_12dx2;
                        });
                        zip3_for_each!(PAR, &mut out, r3, l1, |o, &p, &m| {
                            *o += (16.0 * (p + m)) * inv_12dx2;
                        });
                        zip3_for_each!(PAR, &mut out, r4, l0, |o, &p2, &m2| {
                            *o += (-(p2 + m2)) * inv_12dx2;
                        });
                    }
                }
                // boundary fallbacks (2nd central)
                if nx >= 3 {
                    {
                        let mut out = laplace.slice_mut(s![.., 1..2]);
                        let l = v.slice(s![.., 0..1]);
                        let c = v.slice(s![.., 1..2]);
                        let r = v.slice(s![.., 2..3]);
                        zip4_for_each!(PAR, &mut out, r, c, l, |o, &rr, &cc, &ll| {
                            *o += (rr - 2.0 * cc + ll) * inv_dx2;
                        });
                    }
                    // i = nx-2
                    {
                        let mut out = laplace.slice_mut(s![.., nx - 2..nx - 1]);
                        let l = v.slice(s![.., nx - 3..nx - 2]);
                        let c = v.slice(s![.., nx - 2..nx - 1]);
                        let r = v.slice(s![.., nx - 1..nx]);
                        zip4_for_each!(PAR, &mut out, r, c, l, |o, &rr, &cc, &ll| {
                            *o += (rr - 2.0 * cc + ll) * inv_dx2;
                        });
                    }
                }

                // interior along y axis
                if ny >= 7 {
                    let mut out = laplace.slice_mut(s![3..ny - 3, ..]);
                    let c = v.slice(s![3..ny - 3, ..]);
                    zip3_for_each!(PAR, &mut out, c, c, |o, &cc, &_| {
                        *o += c0y * cc;
                    });
                    {
                        let p1 = v.slice(s![4..ny - 2, ..]);
                        let m1 = v.slice(s![2..ny - 4, ..]);
                        zip3_for_each!(PAR, &mut out, p1, m1, |o, &pp, &mm| {
                            *o += c1y * (pp + mm);
                        });
                    }
                    {
                        let p2 = v.slice(s![5..ny - 1, ..]);
                        let m2 = v.slice(s![1..ny - 5, ..]);
                        zip3_for_each!(PAR, &mut out, p2, m2, |o, &pp, &mm| {
                            *o += c2y * (pp + mm);
                        });
                    }
                    {
                        let p3 = v.slice(s![6..ny, ..]);
                        let m3 = v.slice(s![0..ny - 6, ..]);
                        zip3_for_each!(PAR, &mut out, p3, m3, |o, &pp, &mm| {
                            *o += c3y * (pp + mm);
                        });
                    }
                }

                // boundary fallbacks (4th central)
                if ny >= 5 {
                    {
                        let mut out = laplace.slice_mut(s![2..3, ..]);
                        let t0 = v.slice(s![0..1, ..]);
                        let t1 = v.slice(s![1..2, ..]);
                        let yc = v.slice(s![2..3, ..]);
                        let b3 = v.slice(s![3..4, ..]);
                        let b4 = v.slice(s![4..5, ..]);
                        zip3_for_each!(PAR, &mut out, yc, yc, |o, &cc, &_| {
                            *o += (-30.0 * cc) * inv_12dy2;
                        });
                        zip3_for_each!(PAR, &mut out, b3, t1, |o, &p, &m| {
                            *o += (16.0 * (p + m)) * inv_12dy2;
                        });
                        zip3_for_each!(PAR, &mut out, b4, t0, |o, &p2, &m2| {
                            *o += (-(p2 + m2)) * inv_12dy2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![ny - 3..ny - 2, ..]);
                        let t0 = v.slice(s![ny - 5..ny - 4, ..]);
                        let t1 = v.slice(s![ny - 4..ny - 3, ..]);
                        let yc = v.slice(s![ny - 3..ny - 2, ..]);
                        let b3 = v.slice(s![ny - 2..ny - 1, ..]);
                        let b4 = v.slice(s![ny - 1..ny, ..]);
                        zip3_for_each!(PAR, &mut out, yc, yc, |o, &cc, &_| {
                            *o += (-30.0 * cc) * inv_12dy2;
                        });
                        zip3_for_each!(PAR, &mut out, b3, t1, |o, &p, &m| {
                            *o += (16.0 * (p + m)) * inv_12dy2;
                        });
                        zip3_for_each!(PAR, &mut out, b4, t0, |o, &p2, &m2| {
                            *o += (-(p2 + m2)) * inv_12dy2;
                        });
                    }
                }
                // boundary fallbacks (2nd central)
                if ny >= 3 {
                    {
                        let mut out = laplace.slice_mut(s![1..2, ..]);
                        let d = v.slice(s![0..1, ..]);
                        let c = v.slice(s![1..2, ..]);
                        let u = v.slice(s![2..3, ..]);
                        zip4_for_each!(PAR, &mut out, u, c, d, |o, &uu, &cc, &dd| {
                            *o += (uu - 2.0 * cc + dd) * inv_dy2;
                        });
                    }
                    {
                        let mut out = laplace.slice_mut(s![ny - 2..ny - 1, ..]);
                        let d = v.slice(s![ny - 3..ny - 2, ..]);
                        let c = v.slice(s![ny - 2..ny - 1, ..]);
                        let u = v.slice(s![ny - 1..ny, ..]);
                        zip4_for_each!(PAR, &mut out, u, c, d, |o, &uu, &cc, &dd| {
                            *o += (uu - 2.0 * cc + dd) * inv_dy2;
                        });
                    }
                }
            }
        }

        Ok(laplace)
    }

    /// Compute Laplace field over the entire grid
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn laplacian_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FieldArray2D> {
        self.laplacian_field_impl::<false>(stencil)
    }

    /// Compute Laplace field over the entire grid (in parallel)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn par_laplacian_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FieldArray2D> {
        self.laplacian_field_impl::<true>(stencil)
    }

    /* ================ Spectral methods =================================== */
    /// Convert real ndarray (row-major) to contiguous complex vector
    ///
    /// # Arguments
    /// * `values` - 2D array to be converted
    fn ndarray_to_complex_buffer(values: &FieldArray2D) -> Vec<Complex<f64>> {
        values.iter().map(|&r| Complex::new(r, 0.0)).collect()
    }

    /// Convert contiguous complex vector to a complex 2D ndarray
    ///
    /// # Arguments
    /// * `buffer` - complex contiguous vector
    /// * `ny` - size of y axis
    /// * `nx` - size of x axis
    fn complex_buffer_to_ndarray(
        buffer: Vec<Complex<f64>>,
        ny: usize,
        nx: usize,
    ) -> ComplexFieldArray2D {
        Array2::from_shape_vec((ny, nx), buffer).expect("buffer size mismatch")
    }

    /// Test whether number is a power of two
    ///
    /// # Arguments
    /// * `n` - number to be tested as a power of two
    fn is_base2(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Compute 2D FFT (using phastft backend)
    pub fn fft2d(&self) -> Result<ComplexFieldArray2D> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        if !Self::is_base2(nx) || !Self::is_base2(ny) {
            return Err(LensForgeError::invalid_grid_dimension(format!(
                "FFT (phastft) requires power-of-two dimensions, got ({}, {})",
                ny, nx
            )));
        }
        let mut buffer = Self::ndarray_to_complex_buffer(&self.values);
        let opts = Options::default();
        let row_planner = Planner64::new(nx, Direction::Forward);
        let col_planner = Planner64::new(ny, Direction::Forward);
        // row-wise 1D-FFT
        for j in 0..ny {
            let ini = j * nx;
            let fin = ini + nx;
            let row_slice = &mut buffer[ini..fin];
            let mut re: Vec<f64> = row_slice.iter().map(|c| c.re).collect();
            let mut im: Vec<f64> = row_slice.iter().map(|c| c.im).collect();
            fft_64_with_opts_and_plan(&mut re, &mut im, &opts, &row_planner);
            for (i, c) in row_slice.iter_mut().enumerate() {
                *c = Complex::new(re[i], im[i]);
            }
        }
        // column-wise 1D-FFT
        for i in 0..nx {
            let mut col_re: Vec<f64> = (0..ny).map(|j| buffer[j * nx + i].re).collect();
            let mut col_im: Vec<f64> = (0..ny).map(|j| buffer[j * nx + i].im).collect();
            fft_64_with_opts_and_plan(&mut col_re, &mut col_im, &opts, &col_planner);
            for j in 0..ny {
                buffer[j * nx + i] = Complex::new(col_re[j], col_im[j]);
            }
        }

        Ok(Self::complex_buffer_to_ndarray(buffer, ny, nx))
    }

    /// Compute inverse 2D FFT (using phastft backend)
    ///
    /// # Arguments
    /// * `arr` - complex ndarray to be inverse transformed
    pub fn ifft2d(arr: &ComplexFieldArray2D) -> Result<FieldArray2D> {
        let nx = arr.ncols();
        let ny = arr.nrows();

        if !Self::is_base2(nx) || !Self::is_base2(ny) {
            return Err(LensForgeError::invalid_grid_dimension(format!(
                "IFFT (phastft) requires power-of-two dimensions, got ({}, {})",
                ny, nx
            )));
        }

        let mut buffer: Vec<Complex<f64>> = arr.iter().cloned().collect();
        let opts = Options::default();
        let row_planner = Planner64::new(nx, Direction::Reverse);
        let col_planner = Planner64::new(ny, Direction::Reverse);
        // row-wise inverse 1D-FFT
        for j in 0..ny {
            let ini = j * nx;
            let fin = ini + nx;
            let row_slice = &mut buffer[ini..fin];
            let mut re: Vec<f64> = row_slice.iter().map(|c| c.re).collect();
            let mut im: Vec<f64> = row_slice.iter().map(|c| c.im).collect();
            fft_64_with_opts_and_plan(&mut re, &mut im, &opts, &row_planner);
            for (i, c) in row_slice.iter_mut().enumerate() {
                *c = Complex::new(re[i], im[i]);
            }
        }
        // column-wise inverse 1D-FFT
        for i in 0..nx {
            let mut col_re: Vec<f64> = (0..ny).map(|j| buffer[j * nx + i].re).collect();
            let mut col_im: Vec<f64> = (0..ny).map(|j| buffer[j * nx + i].im).collect();
            fft_64_with_opts_and_plan(&mut col_re, &mut col_im, &opts, &col_planner);
            for j in 0..ny {
                buffer[j * nx + i] = Complex::new(col_re[j], col_im[j]);
            }
        }

        let values_re: Vec<f64> = buffer.iter().map(|c| c.re).collect();
        Array2::from_shape_vec((ny, nx), values_re)
            .map_err(|e| LensForgeError::numerical_error(e.to_string()))
    }

    /// Compute FFT shift (move zero-frequency component to center)
    ///
    /// # Arguments
    /// * `arr` - complex array to be shifted
    pub fn fftshift(arr: &ComplexFieldArray2D) -> ComplexFieldArray2D {
        let nx = arr.ncols();
        let ny = arr.nrows();
        let nx_half = nx / 2;
        let ny_half = ny / 2;

        let mut shifted = Array2::zeros((ny, nx));

        // swap quadrants
        for j in 0..ny {
            for i in 0..nx {
                let i_shift = (i + nx_half) % nx;
                let j_shift = (j + ny_half) % ny;
                shifted[[j_shift, i_shift]] = arr[[j, i]];
            }
        }

        shifted
    }

    /// Compute inverse FFT shift (undo fftshift)
    ///
    /// # Arguments
    /// * `arr` - complex array to be unshifted
    pub fn ifftshift(arr: &ComplexFieldArray2D) -> ComplexFieldArray2D {
        let nx = arr.ncols();
        let ny = arr.nrows();
        let nx_half = nx.div_ceil(2);
        let ny_half = ny.div_ceil(2);

        let mut unshifted = Array2::zeros((ny, nx));

        // unswap quadrants
        for j in 0..ny {
            for i in 0..nx {
                let i_shift = (i + nx_half) % nx;
                let j_shift = (j + ny_half) % ny;
                unshifted[[j_shift, i_shift]] = arr[[j, i]];
            }
        }

        unshifted
    }

    /// Fourier symbols (for gradients) for given finite difference stencil types
    ///
    /// # Arguments
    /// * `stencil`: derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    #[inline]
    fn fourier_symbols_fd1(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FourierSymbols> {
        let stencil = stencil.into().unwrap_or_default();
        let (kx, ky) = self.grid.wavenumbers();
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dkx = kx * dx;
        let dky = ky * dy;

        let (kxd1, kyd1) = match stencil {
            DerivativeStencil::Central => {
                (
                    // (exp(ikh) - exp(-ikh)) / 2h = i*sin(kh) / h
                    dkx.mapv(|theta| Complex::new(0.0, theta.sin() / dx)),
                    dky.mapv(|theta| Complex::new(0.0, theta.sin() / dy)),
                )
            }
            DerivativeStencil::Forward => {
                (
                    // (exp(ikh) - 1) / h
                    dkx.mapv(|theta| Complex::new(theta.cos() - 1.0, theta.sin()) / dx),
                    dky.mapv(|theta| Complex::new(theta.cos() - 1.0, theta.sin()) / dy),
                )
            }
            DerivativeStencil::Backward => {
                (
                    // (1 - exp(-ikh)) / h
                    dkx.mapv(|theta| Complex::new(1.0 - theta.cos(), theta.sin()) / dx),
                    dky.mapv(|theta| Complex::new(1.0 - theta.cos(), theta.sin()) / dy),
                )
            }
            DerivativeStencil::Central4 => {
                (
                    // (-exp(2ikh) + 8*exp(ikh) - 8*exp(-ikh) + exp(-2ikh)) / 12h
                    // = i/h * (4/3 * sin(kh) - 1/6 * sin(2kh))
                    dkx.mapv(|theta| {
                        Complex::new(
                            0.0,
                            ((4.0 / 3.0) * theta.sin() - (1.0 / 6.0) * (2.0 * theta).sin()) / dx,
                        )
                    }),
                    dky.mapv(|theta| {
                        Complex::new(
                            0.0,
                            ((4.0 / 3.0) * theta.sin() - (1.0 / 6.0) * (2.0 * theta).sin()) / dy,
                        )
                    }),
                )
            }
            DerivativeStencil::Central6 => {
                (
                    // (exp(3ikh) - 9*exp(2ikh) + 45*exp(ikh) - 45*exp(-ikh) + 9*exp(-2ikh) - exp(-3ikh)) / 60h
                    // = i/h * (3/2 * sin(kh) - 3/10 * sin(2kh) + 1/30 * sin(3kh))
                    dkx.mapv(|theta| {
                        Complex::new(
                            0.0,
                            (1.5 * theta.sin() - 0.3 * (2.0 * theta).sin()
                                + (1.0 / 30.0) * (3.0 * theta).sin())
                                / dx,
                        )
                    }),
                    dky.mapv(|theta| {
                        Complex::new(
                            0.0,
                            (1.5 * theta.sin() - 0.3 * (2.0 * theta).sin()
                                + (1.0 / 30.0) * (3.0 * theta).sin())
                                / dy,
                        )
                    }),
                )
            }
        };

        Ok((kxd1, kyd1))
    }

    /// Fourier symbols (for Laplacian) for given finite difference stencil types
    ///
    /// # Arguments
    /// * `stencil`: Laplacian stencil type (default: DerivativeStencil::Central; defaults if `None`)
    #[inline]
    fn fourier_symbols_fd2(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FourierSymbols> {
        let stencil = stencil.into().unwrap_or_default();
        let (kx, ky) = self.grid.wavenumbers();
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let dkx = kx * dx;
        let dky = ky * dy;

        let (kxd2, kyd2) = match stencil {
            DerivativeStencil::Central => {
                (
                    // (exp(ikh) - 2 + exp(-ikh)) / h^2
                    // = (2*cos(kh)-2) / h^2
                    dkx.mapv(|theta| Complex::new((2.0 * theta.cos() - 2.0) / dx2, 0.0)),
                    dky.mapv(|theta| Complex::new((2.0 * theta.cos() - 2.0) / dy2, 0.0)),
                )
            }
            DerivativeStencil::Forward => {
                (
                    // (exp(ikh) - 1)^2 / h^2
                    dkx.mapv(|theta| {
                        let cos_theta = theta.cos();
                        Complex::new(
                            2.0 * ((cos_theta - 1.0) * cos_theta),
                            2.0 - 2.0 * cos_theta * theta.sin(),
                        ) / dx2
                    }),
                    dky.mapv(|theta| {
                        let cos_theta = theta.cos();
                        Complex::new(
                            2.0 * ((cos_theta - 1.0) * cos_theta),
                            2.0 - 2.0 * cos_theta * theta.sin(),
                        ) / dy2
                    }),
                )
            }
            DerivativeStencil::Backward => {
                (
                    // (1 - exp(-ikh))^2 / h^2
                    dkx.mapv(|theta| {
                        let cos_theta = theta.cos();
                        Complex::new(
                            2.0 * ((cos_theta - 1.0) * cos_theta),
                            2.0 - 2.0 * cos_theta * theta.sin(),
                        ) / dx2
                    }),
                    dky.mapv(|theta| {
                        let cos_theta = theta.cos();
                        Complex::new(
                            2.0 * ((cos_theta - 1.0) * cos_theta),
                            2.0 - 2.0 * cos_theta * theta.sin(),
                        ) / dy2
                    }),
                )
            }
            DerivativeStencil::Central4 => {
                let c2 = 1.0 / 6.0;
                let c1 = 8.0 / 3.0;
                (
                    // (-exp(2ikh) + 16*exp(ikh) - 30 + 16*exp(-ikh) - exp(-2ikh)) / 12h^2
                    // = (-1/6 * cos(2kh) + 8/3 * cos(kh) - 5/2 ) / h^2
                    dkx.mapv(|theta| {
                        Complex::new(
                            (c2 * (2.0 * theta).cos() + c1 * theta.cos() - 2.5) / dx2,
                            0.0,
                        )
                    }),
                    dky.mapv(|theta| {
                        Complex::new(
                            (c2 * (2.0 * theta).cos() + c1 * theta.cos() - 2.5) / dy2,
                            0.0,
                        )
                    }),
                )
            }
            DerivativeStencil::Central6 => {
                let c3 = 1.0 / 45.0;
                let c0 = 49.0 / 18.0;
                (
                    // (2*exp(3ikh) - 27*exp(2ikh) + 270*exp(ikh) - 490 + 270*exp(-ikh) - 27*exp(-2ikh) + 2*exp(-3ikh)) / 180h^2
                    // (1/45*cos(3kh) - 3/10*cos(2kh) + 2*cos(kh) - 49/18) / h^2
                    dkx.mapv(|theta| {
                        Complex::new(
                            (c3 * (3.0 * theta).cos() - 0.3 * (2.0 * theta).cos()
                                + 2.0 * theta.cos()
                                - c0)
                                / dx2,
                            0.0,
                        )
                    }),
                    dky.mapv(|theta| {
                        Complex::new(
                            (c3 * (3.0 * theta).cos() - 0.3 * (2.0 * theta).cos()
                                + 2.0 * theta.cos()
                                - c0)
                                / dy2,
                            0.0,
                        )
                    }),
                )
            }
        };

        Ok((kxd2, kyd2))
    }

    /// Compute gradient using spectral method (using FFT)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn gradient_spectral(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<VectorField2D> {
        let stencil = stencil.into().unwrap_or_default();
        let (kx_fd, ky_fd) = self.fourier_symbols_fd1(stencil)?;
        self.gradient_spectral_k(&kx_fd, &ky_fd)
    }

    /// Compute gradient using spectral method with given Fourier symbols (using FFT)
    ///
    /// # Arguments
    /// * `kx_fd` - pre-computed Fourier symbols (translation operators) along the x axis
    /// * `ky_fd` - pre-computed Fourier symbols (translation operators) along the y axis
    pub fn gradient_spectral_k(
        &self,
        kx_fd: &ComplexArray1D,
        ky_fd: &ComplexArray1D,
    ) -> Result<VectorField2D> {
        let mut dvdx_fft = self.fft2d()?;
        let mut dvdy_fft = dvdx_fft.clone();

        for mut row in dvdx_fft.axis_iter_mut(Axis(0)) {
            Zip::from(&mut row)
                .and(kx_fd)
                .for_each(|fhat, &kx| *fhat *= kx);
        }

        for (j, mut row) in dvdy_fft.axis_iter_mut(Axis(0)).enumerate() {
            let ky = ky_fd[j];
            row.mapv_inplace(|fhat| fhat * ky);
        }

        let dvdx = Self::ifft2d(&dvdx_fft)?;
        let dvdy = Self::ifft2d(&dvdy_fft)?;

        Ok((dvdx, dvdy))
    }

    /// Compute Laplacian using spectral method (using FFT)
    ///
    /// Note: assumes periodic boundaries
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn laplacian_spectral(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Result<FieldArray2D> {
        let stencil = stencil.into().unwrap_or_default();
        let (kx_fd2, ky_fd2) = self.fourier_symbols_fd2(stencil)?;
        self.laplacian_spectral_k(&kx_fd2, &ky_fd2)
    }

    /// Compute Laplacian using spectral method with given squared Fourier symbols (using FFT)
    ///
    /// Note: assumes periodic boundaries
    ///
    /// # Arguments
    /// * `kx_fd2` - pre-computed Fourier symbols (translation operators) along the x axis
    /// * `ky_fd2` - pre-computed Fourier symbols (translation operators) along the x axis
    pub fn laplacian_spectral_k(
        &self,
        kx_fd2: &ComplexArray1D,
        ky_fd2: &ComplexArray1D,
    ) -> Result<FieldArray2D> {
        let mut fft_data = self.fft2d()?;
        for (j, mut row) in fft_data.axis_iter_mut(Axis(0)).enumerate() {
            let ky2 = ky_fd2[j];
            Zip::from(&mut row)
                .and(kx_fd2)
                .for_each(|fhat, &kx2| *fhat *= kx2 + ky2);
        }

        Self::ifft2d(&fft_data)
    }
}

/* ================ Derivative methods =============================== */
/// Discrete derivative stencil type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DerivativeStencil {
    /// Forward difference: ∇ - (f[i+1] - f[i]) / dx
    Forward,
    /// Backward difference: ∇ - (f[i] - f[i-1]) / dx
    Backward,
    /// Central difference:
    /// ∇ - (f[i+1] - f[i-1]) / (2*dx)
    /// Δ - (f[i+1] - 2*f[i] + f[i-1]) / (dx*dx)
    #[default]
    Central,
    /// Central difference (4-th order):
    /// * ∇ - (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)
    /// * Δ - (-f[i+2] + 16*f[i+1] - 30*f[i] + 16*f[i-1] - f[i-2]) / (12*dx*dx)
    Central4,
    /// Central difference (6-th order):
    /// * ∇ - (f[i+3] - 9*f[i+2] + 45*f[i+1] - 45*f[i-1] + 9*f[i-2] - f[i-3]) / (60*dx)
    /// * Δ - (2*f[i+3] - 27*f[i+2] + 270*f[i+1] - 490*f[i] + 270*f[i-1] - 27*f[i-2] + 2*f[i-3]) / (180*dx*dx)
    Central6,
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
        let orig_val = pg.interpolate(&center);
        let resampled_val = resampled.interpolate(&center);
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
    fn test_pixel_grid_fft_roundtrip() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (x * std::f64::consts::PI).cos() + y);

        let fft_result = pg.fft2d().unwrap();
        let reconstructed = PixelGrid::ifft2d(&fft_result).unwrap();

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(pg.values[[j, i]], reconstructed[[j, i]], 1e-12));
            }
        }
    }

    #[test]
    fn test_pixel_grid_fft_non_power_of_two() {
        let grid = Grid2D::square(5, 2.0).unwrap();
        let pg = PixelGrid::ones(grid);

        let result = pg.fft2d();
        assert!(result.is_err());
    }

    #[test]
    fn test_pixel_grid_fftshift() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::ones(grid);
        let fft_result = pg.fft2d().unwrap();

        let shifted = PixelGrid::fftshift(&fft_result);
        let unshifted = PixelGrid::ifftshift(&shifted);

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(
                    fft_result[[j, i]].re,
                    unshifted[[j, i]].re,
                    1e-12
                ));
                assert!(approx_eq(
                    fft_result[[j, i]].im,
                    unshifted[[j, i]].im,
                    1e-12
                ));
            }
        }
    }

    #[test]
    fn test_pixel_grid_interpolation() {
        let grid = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, _| x);

        let center = Point2D::origin();
        let interp = pg.interpolate(&center);

        assert!(approx_eq(interp, 0.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_gradient() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| 2.0 * x + 3.0 * y);

        let grad = pg.gradient(4, 4, DerivativeStencil::Central).unwrap();
        assert!(approx_eq(grad.x, 2.0, 1e-12));
        assert!(approx_eq(grad.y, 3.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_gradient_field() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| 2.0 * x + 3.0 * y);

        let (dvdx, dvdy) = pg.gradient_field(None).unwrap();
        let (par_dvdx, par_dvdy) = pg.par_gradient_field(None).unwrap();

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(dvdx[[j, i]], par_dvdx[[j, i]], 1e-12));
                assert!(approx_eq(dvdy[[j, i]], par_dvdy[[j, i]], 1e-12));
            }
        }
    }

    #[test]
    fn test_pixel_grid_gradient_spectral() {
        let grid = Grid2D::square(64, 2.0).unwrap();
        //
        let pg = PixelGrid::from_func(grid, |x, y| x + 2.0 * y);

        let (dvdx, dvdy) = pg.gradient_spectral(DerivativeStencil::Central).unwrap();

        // Check interior
        for j in 16..48 {
            for i in 16..48 {
                assert!(
                    approx_eq(dvdx[[j, i]], 1.0, 1e-12),
                    "dvdx at ({}, {}) = {}, expected 1.0",
                    i,
                    j,
                    dvdx[[j, i]]
                );
                assert!(
                    approx_eq(dvdy[[j, i]], 2.0, 1e-12),
                    "dvdy at ({}, {}) = {}, expected 2.0",
                    i,
                    j,
                    dvdy[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_pixel_grid_laplacian() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        // Quadratic function: Laplacian should be constant
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let lap = pg.laplacian(4, 4, DerivativeStencil::Central).unwrap();
        assert!(approx_eq(lap, 4.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_laplacian_field() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        // Quadratic function: Laplacian should be constant
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let laplace = pg.laplacian_field(None).unwrap();
        let par_laplace = pg.par_laplacian_field(None).unwrap();

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(laplace[[j, i]], par_laplace[[j, i]], 1e-12));
            }
        }
    }

    #[test]
    fn test_pixel_grid_laplacian_spectral() {
        let size: usize = 64;
        let grid = Grid2D::square(size, 2.0).unwrap();
        // Quadratic function: Laplacian should be constant
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let laplace_fft = pg.laplacian_spectral(DerivativeStencil::Central).unwrap();

        let border_size = size / 32;
        for j in border_size..size - border_size {
            for i in border_size..size - border_size {
                assert!(
                    approx_eq(laplace_fft[[j, i]], 4.0, 1e-12),
                    "Laplacian at ({}, {}) = {}, expected {}",
                    i,
                    j,
                    laplace_fft[[j, i]],
                    4.0
                );
            }
        }
    }

    #[test]
    fn test_laplacian_spectral_vs_finite_diff() {
        let size: usize = 128;
        let grid = Grid2D::square(size, 2.0).unwrap();
        // Quadratic function: Laplacian should be constant
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let lap_fd = pg.laplacian_field(DerivativeStencil::Central).unwrap();
        let lap_spec = pg.laplacian_spectral(DerivativeStencil::Central).unwrap();

        let mut max_err: f64 = 0.0;

        let border_size = size / 32;
        for j in border_size..size - border_size {
            for i in border_size..size - border_size {
                let err = (lap_fd[[j, i]] - lap_spec[[j, i]]).abs();
                max_err = max_err.max(err);
            }
        }
        println!("Max Laplacian error: {}", max_err);
        assert!(max_err < 1e-10, "Laplacian error too large: {}", max_err);
    }
}
