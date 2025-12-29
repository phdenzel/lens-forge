// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens_forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Uniform grid with pixels

use crate::grid::{Grid2D, Point2D};
use crate::{LensForgeError, Result};
use core::f64;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num::complex::Complex;
use phastft::{
    fft_64_with_opts_and_plan,
    options::Options,
    planner::{Direction, Planner64},
};
use rayon::prelude::*;

/// A 2D pixel grid with associated values
#[derive(Debug, Clone)]
pub struct PixelGrid {
    /// The underlying grid geometry
    pub grid: Grid2D,
    /// The pixel values stored as a 2D array [ny, nx] (row-major)
    pub values: Array2<f64>,
}

impl PixelGrid {
    /// Create a new pixel grid initialized with zeros
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    pub fn zeros(grid: Grid2D) -> Self {
        Self {
            grid: grid,
            values: Array2::zeros(grid.shape()),
        }
    }

    /// Create a new pixel grid initialized with ones
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    pub fn ones(grid: Grid2D) -> Self {
        Self {
            grid: grid,
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
            grid: grid,
            values: Array2::from_elem(grid.shape(), value),
        }
    }

    /// Create a new pixel grid initialized from an ndarray
    ///
    /// # Arguments
    /// * `grid` - grid struct holding information about the geometry
    /// * `values` - pixel value array
    pub fn from_values(grid: Grid2D, values: Array2<f64>) -> Result<Self> {
        if values.nrows() != grid.ny || values.ncols() != grid.nx {
            return Err(LensForgeError::DimensionMismatch {
                expected: format!("({}, {})", grid.ny, grid.nx),
                got: format!("({}, {})", values.nrows(), values.ncols()),
            });
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
            return Err(LensForgeError::DimensionMismatch {
                expected: format!("{}", grid.n_pixels()),
                got: format!("{}", v.len()),
            });
        }
        let values = Array2::from_shape_vec(grid.shape(), v)
            .map_err(|e| LensForgeError::NumericalError(e.to_string()))?;
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
    pub fn as_mut_array2(&mut self) -> &mut Array2<f64> {
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
            return Err(LensForgeError::DimensionMismatch {
                expected: format!("({}, {})", self.grid.ny, self.grid.nx),
                got: format!("{}, {}", other.grid.ny, other.grid.nx),
            });
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
            return Err(LensForgeError::DimensionMismatch {
                expected: format!("({}, {})", self.grid.ny, self.grid.nx),
                got: format!("{}, {}", other.grid.ny, other.grid.nx),
            });
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
            return Err(LensForgeError::DimensionMismatch {
                expected: format!("({}, {})", self.grid.ny, self.grid.nx),
                got: format!("{}, {}", other.grid.ny, other.grid.nx),
            });
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
        self.values.mapv_inplace(|x| f(x));
    }

    /// Map function to the pixel grid values
    ///
    /// # Arguments
    /// * `f` - function to be mapped
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self {
            grid: self.grid,
            values: self.values.mapv(|x| f(x)),
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
        let c = num.unwrap_or_else(|| 1.0);
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
        let c = num.unwrap_or_else(|| 1.0);
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
            grid: grid,
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
    ) -> Point2D {
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
        };

        Point2D::new(dvdx, dvdy)
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
    ) -> Point2D {
        let stencil = stencil.into().unwrap_or_default();
        let dx = self.grid.dx;
        let dy = self.grid.dy;

        let dvdx = match stencil {
            DerivativeStencil::Forward => {
                (self.interpolate(&Point2D::new(p.x + dx, p.y)) - self.interpolate(p)) / dx
            }
            DerivativeStencil::Backward => {
                (self.interpolate(p) - self.interpolate(&Point2D::new(p.x - dx, p.y))) / dx
            }
            DerivativeStencil::Central => {
                (self.interpolate(&Point2D::new(p.x + dx, p.y))
                    - self.interpolate(&Point2D::new(p.x - dx, p.y)))
                    / (2.0 * dx)
            }
        };

        let dvdy = match stencil {
            DerivativeStencil::Forward => {
                (self.interpolate(&Point2D::new(p.x, p.y + dy)) - self.interpolate(p)) / dy
            }
            DerivativeStencil::Backward => {
                (self.interpolate(p) - self.interpolate(&Point2D::new(p.x, p.y - dy))) / dy
            }
            DerivativeStencil::Central => {
                (self.interpolate(&Point2D::new(p.x, p.y + dy))
                    - self.interpolate(&Point2D::new(p.x, p.y - dy)))
                    / (2.0 * dy)
            }
        };

        Point2D::new(dvdx, dvdy)
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
    ) -> f64 {
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
        };

        d2vdx2 + d2vdy2
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
    ) -> f64 {
        let stencil = stencil.into().unwrap_or_default();
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let v_00 = self.interpolate(p);

        let d2vdx2 = match stencil {
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
        };

        let d2vdy2 = match stencil {
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
        };

        d2vdx2 + d2vdy2
    }

    /// Compute gradient field over the entire grid
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn gradient_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> (Array2<f64>, Array2<f64>) {
        let stencil = stencil.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx = 1.0 / self.grid.dx;
        let inv_dy = 1.0 / self.grid.dy;

        let mut dvdx = Array2::zeros((ny, nx));
        let mut dvdy = Array2::zeros((ny, nx));

        match stencil {
            DerivativeStencil::Central => {
                let inv_2dx = 0.5 * inv_dx;
                for j in 0..ny {
                    for i in 1..nx - 1 {
                        dvdx[[j, i]] =
                            (self.values[[j, i + 1]] - self.values[[j, i - 1]]) * inv_2dx;
                    }
                    if nx > 1 {
                        dvdx[[j, 0]] = (self.values[[j, 1]] - self.values[[j, 0]]) * inv_dx;
                        dvdx[[j, nx - 1]] =
                            (self.values[[j, nx - 1]] - self.values[[j, nx - 2]]) * inv_dx;
                    }
                }
                let inv_2dy = 0.5 * inv_dy;
                for i in 0..nx {
                    for j in 1..ny - 1 {
                        dvdy[[j, i]] =
                            (self.values[[j + 1, i]] - self.values[[j - 1, i]]) * inv_2dy;
                    }
                    if ny > 1 {
                        dvdy[[0, i]] = (self.values[[1, i]] - self.values[[0, i]]) * inv_dy;
                        dvdy[[ny - 1, i]] =
                            (self.values[[ny - 1, i]] - self.values[[ny - 2, i]]) * inv_dy;
                    }
                }
            }
            DerivativeStencil::Forward => {
                for j in 0..ny {
                    for i in 0..nx - 1 {
                        dvdx[[j, i]] = (self.values[[j, i + 1]] - self.values[[j, i]]) * inv_dx;
                    }
                    dvdx[[j, nx - 1]] =
                        (self.values[[j, nx - 1]] - self.values[[j, nx - 2]]) * inv_dx;
                }
                for i in 0..nx {
                    for j in 0..ny - 1 {
                        dvdy[[j, i]] = (self.values[[j + 1, i]] - self.values[[j, i]]) * inv_dy;
                    }
                    dvdy[[ny - 1, i]] =
                        (self.values[[ny - 1, i]] - self.values[[ny - 2, i]]) * inv_dy;
                }
            }
            DerivativeStencil::Backward => {
                for j in 0..ny {
                    dvdx[[j, 0]] = (self.values[[j, 1]] - self.values[[j, 0]]) * inv_dx;
                    for i in 1..nx {
                        dvdx[[j, i]] = (self.values[[j, i]] - self.values[[j, i - 1]]) * inv_dx;
                    }
                }

                for i in 0..nx {
                    dvdy[[0, i]] = (self.values[[1, i]] - self.values[[0, i]]) * inv_dy;
                    for j in 1..ny {
                        dvdy[[j, i]] = (self.values[[j, i]] - self.values[[j - 1, i]]) * inv_dy;
                    }
                }
            }
        }

        (dvdx, dvdy)
    }

    /// Compute gradient field over the entire grid in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn par_gradient_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> (Array2<f64>, Array2<f64>) {
        let stencil = stencil.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx = 1.0 / self.grid.dx;
        let inv_dy = 1.0 / self.grid.dy;
        let inv_2dx = 0.5 * inv_dx;
        let inv_2dy = 0.5 * inv_dy;

        let gradients: Vec<(f64, f64)> = (0..ny * nx)
            .into_par_iter()
            .map(|idx| {
                let i = idx % nx;
                let j = idx / nx;

                let dvdx = match stencil {
                    DerivativeStencil::Central => {
                        if i == 0 {
                            if nx > 1 {
                                (self.values[[j, 1]] - self.values[[j, 0]]) * inv_dx
                            } else {
                                0.0
                            }
                        } else if i == nx - 1 {
                            (self.values[[j, i]] - self.values[[j, i - 1]]) * inv_dx
                        } else {
                            (self.values[[j, i + 1]] - self.values[[j, i - 1]]) * inv_2dx
                        }
                    }
                    DerivativeStencil::Forward => {
                        if i < nx - 1 {
                            (self.values[[j, i + 1]] - self.values[[j, i]]) * inv_dx
                        } else {
                            (self.values[[j, i]] - self.values[[j, i - 1]]) * inv_dx
                        }
                    }
                    DerivativeStencil::Backward => {
                        if i > 0 {
                            (self.values[[j, i]] - self.values[[j, i - 1]]) * inv_dx
                        } else {
                            (self.values[[j, i + 1]] - self.values[[j, i]]) * inv_dx
                        }
                    }
                };

                let dvdy = match stencil {
                    DerivativeStencil::Central => {
                        if j == 0 {
                            if ny > 1 {
                                (self.values[[1, i]] - self.values[[0, i]]) * inv_dy
                            } else {
                                0.0
                            }
                        } else if j == ny - 1 {
                            (self.values[[ny - 1, i]] - self.values[[ny - 2, i]]) * inv_dy
                        } else {
                            (self.values[[j + 1, i]] - self.values[[j - 1, i]]) * inv_2dy
                        }
                    }
                    DerivativeStencil::Forward => {
                        if j < ny - 1 {
                            (self.values[[j + 1, i]] - self.values[[j, i]]) * inv_dy
                        } else {
                            (self.values[[j, i]] - self.values[[j - 1, i]]) * inv_dy
                        }
                    }
                    DerivativeStencil::Backward => {
                        if j > 0 {
                            (self.values[[j, i]] - self.values[[j - 1, i]]) * inv_dy
                        } else {
                            (self.values[[j + 1, i]] - self.values[[j, i]]) * inv_dy
                        }
                    }
                };

                (dvdx, dvdy)
            })
            .collect();

        let dvdx_data: Vec<f64> = gradients.iter().map(|(gx, _)| *gx).collect();
        let dvdy_data: Vec<f64> = gradients.iter().map(|(_, gy)| *gy).collect();

        (
            Array2::from_shape_vec((ny, nx), dvdx_data).unwrap(),
            Array2::from_shape_vec((ny, nx), dvdy_data).unwrap(),
        )
    }

    /// Compute gradient magnitude of the entire grid
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn gradient_magnitude(&self, stencil: impl Into<Option<DerivativeStencil>>) -> Array2<f64> {
        let stencil = stencil.into().unwrap_or_default();
        let (dvdx, dvdy) = self.gradient_field(stencil);
        let mut magnitude = Array2::zeros((self.grid.ny, self.grid.nx));

        ndarray::Zip::from(&mut magnitude)
            .and(&dvdx)
            .and(&dvdy)
            .for_each(|m, &gx, &gy| {
                *m = (gx * gx + gy * gy).sqrt();
            });

        magnitude
    }

    /// Compute gradient magnitude of the entire grid in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn par_gradient_magnitude(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Array2<f64> {
        let stencil = stencil.into().unwrap_or_default();
        let (dvdx, dvdy) = self.par_gradient_field(stencil);
        let mut magnitude = Array2::zeros((self.grid.ny, self.grid.nx));

        ndarray::Zip::from(&mut magnitude)
            .and(&dvdx)
            .and(&dvdy)
            .par_for_each(|m, &gx, &gy| {
                *m = (gx * gx + gy * gy).sqrt();
            });

        magnitude
    }

    /// Compute Laplace field over the entire grid
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn laplacian_field(&self, stencil: impl Into<Option<DerivativeStencil>>) -> Array2<f64> {
        let stencil = stencil.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx2 = 1.0 / (self.grid.dx * self.grid.dx);
        let inv_dy2 = 1.0 / (self.grid.dy * self.grid.dy);

        let mut laplace = Array2::zeros((ny, nx));

        match stencil {
            DerivativeStencil::Central => {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let v_ij = self.values[[j, i]];
                        laplace[[j, i]] = (self.values[[j, i + 1]] - 2.0 * v_ij
                            + self.values[[j, i - 1]])
                            * inv_dx2
                            + (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]])
                                * inv_dy2;
                    }
                }
                // Fallback to forward/backward stencils
                if nx > 2 && ny > 2 {
                    // Top and bottom rows (excluding corners)
                    for i in 1..nx - 1 {
                        let v_0i = self.values[[0, i]];
                        laplace[[0, i]] = (self.values[[0, i + 1]] - 2.0 * v_0i
                            + self.values[[0, i - 1]])
                            * inv_dx2
                            + (self.values[[2, i]] - 2.0 * self.values[[1, i]] + v_0i) * inv_dy2;

                        let v_ni = self.values[[ny - 1, i]];
                        laplace[[ny - 1, i]] = (self.values[[ny - 1, i + 1]] - 2.0 * v_ni
                            + self.values[[ny - 1, i - 1]])
                            * inv_dx2
                            + (v_ni - 2.0 * self.values[[ny - 2, i]] + self.values[[ny - 3, i]])
                                * inv_dy2;
                    }

                    // Left and right columns (excluding corners)
                    for j in 1..ny - 1 {
                        let v_j0 = self.values[[j, 0]];
                        laplace[[j, 0]] = (self.values[[j, 2]] - 2.0 * self.values[[j, 1]] + v_j0)
                            * inv_dx2
                            + (self.values[[j + 1, 0]] - 2.0 * v_j0 + self.values[[j - 1, 0]])
                                * inv_dy2;

                        let v_jn = self.values[[j, nx - 1]];
                        laplace[[j, nx - 1]] = (v_jn - 2.0 * self.values[[j, nx - 2]]
                            + self.values[[j, nx - 3]])
                            * inv_dx2
                            + (self.values[[j + 1, nx - 1]] - 2.0 * v_jn
                                + self.values[[j - 1, nx - 1]])
                                * inv_dy2;
                    }

                    // Corners
                    let v00 = self.values[[0, 0]];
                    laplace[[0, 0]] = (self.values[[0, 2]] - 2.0 * self.values[[0, 1]] + v00)
                        * inv_dx2
                        + (self.values[[2, 0]] - 2.0 * self.values[[1, 0]] + v00) * inv_dy2;

                    let v0n = self.values[[0, nx - 1]];
                    laplace[[0, nx - 1]] =
                        (v0n - 2.0 * self.values[[0, nx - 2]] + self.values[[0, nx - 3]]) * inv_dx2
                            + (self.values[[2, nx - 1]] - 2.0 * self.values[[1, nx - 1]] + v0n)
                                * inv_dy2;

                    let vn0 = self.values[[ny - 1, 0]];
                    laplace[[ny - 1, 0]] =
                        (self.values[[ny - 1, 2]] - 2.0 * self.values[[ny - 1, 1]] + vn0) * inv_dx2
                            + (vn0 - 2.0 * self.values[[ny - 2, 0]] + self.values[[ny - 3, 0]])
                                * inv_dy2;

                    let vnn = self.values[[ny - 1, nx - 1]];
                    laplace[[ny - 1, nx - 1]] = (vnn - 2.0 * self.values[[ny - 1, nx - 2]]
                        + self.values[[ny - 1, nx - 3]])
                        * inv_dx2
                        + (vnn - 2.0 * self.values[[ny - 2, nx - 1]]
                            + self.values[[ny - 3, nx - 1]])
                            * inv_dy2;
                }
            }
            DerivativeStencil::Forward => {
                for j in 0..ny {
                    for i in 0..nx {
                        let v_ij = self.values[[j, i]];

                        let d2vdx2 = if i + 2 < nx {
                            (self.values[[j, i + 2]] - 2.0 * self.values[[j, i + 1]] + v_ij)
                                * inv_dx2
                        } else if i + 1 < nx && i > 0 {
                            (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]])
                                * inv_dx2
                        } else if i > 1 {
                            (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]])
                                * inv_dx2
                        } else {
                            0.0
                        };

                        let d2vdy2 = if j + 2 < ny {
                            (self.values[[j + 2, i]] - 2.0 * self.values[[j + 1, i]] + v_ij)
                                * inv_dy2
                        } else if j + 1 < ny && j > 0 {
                            (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]])
                                * inv_dy2
                        } else if j > 1 {
                            (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]])
                                * inv_dy2
                        } else {
                            0.0
                        };

                        laplace[[j, i]] = d2vdx2 + d2vdy2;
                    }
                }
            }
            DerivativeStencil::Backward => {
                for j in 0..ny {
                    for i in 0..nx {
                        let v_ij = self.values[[j, i]];

                        let d2vdx2 = if i >= 2 {
                            (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]])
                                * inv_dx2
                        } else if i > 0 && i + 1 < nx {
                            (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]])
                                * inv_dx2
                        } else if i + 2 < nx {
                            (self.values[[j, i + 2]] - 2.0 * self.values[[j, i + 1]] + v_ij)
                                * inv_dx2
                        } else {
                            0.0
                        };

                        let d2vdy2 = if j >= 2 {
                            (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]])
                                * inv_dy2
                        } else if j > 0 && j + 1 < ny {
                            (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]])
                                * inv_dy2
                        } else if j + 2 < ny {
                            (self.values[[j + 2, i]] - 2.0 * self.values[[j + 1, i]] + v_ij)
                                * inv_dy2
                        } else {
                            0.0
                        };

                        laplace[[j, i]] = d2vdx2 + d2vdy2;
                    }
                }
            }
        }

        laplace
    }

    /// Compute Laplace field over the entire grid
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    pub fn par_laplacian_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
    ) -> Array2<f64> {
        let stencil = stencil.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx2 = 1.0 / (self.grid.dx * self.grid.dx);
        let inv_dy2 = 1.0 / (self.grid.dy * self.grid.dy);

        // Compute as flat vec in parallel, then reshape
        let laplace: Vec<f64> = (0..ny * nx)
            .into_par_iter()
            .map(|idx| {
                let i = idx % nx;
                let j = idx / nx;
                let v_ij = self.values[[j, i]];

		let d2vdx2 = match stencil {
                    DerivativeStencil:: Central => {
                        if i > 0 && i < nx - 1 {
                            (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) * inv_dx2
                        } else if i == 0 && nx > 2 {
                            (self. values[[j, 2]] - 2.0 * self.values[[j, 1]] + v_ij) * inv_dx2
                        } else if i == nx - 1 && nx > 2 {
                            (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]]) * inv_dx2
                        } else {
                            0.0
                        }
                    }
                    DerivativeStencil::Forward => {
                        if i + 2 < nx {
                            (self.values[[j, i + 2]] - 2.0 * self.values[[j, i + 1]] + v_ij) * inv_dx2
                        } else if i + 1 < nx && i > 0 {
                            (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) * inv_dx2
                        } else if i > 1 {
                            (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]]) * inv_dx2
                        } else {
                            0.0
                        }
                    }
                    DerivativeStencil::Backward => {
                        if i >= 2 {
                            (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]]) * inv_dx2
                        } else if i > 0 && i + 1 < nx {
                            (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) * inv_dx2
                        } else if i + 2 < nx {
                            (self.values[[j, i + 2]] - 2.0 * self.values[[j, i + 1]] + v_ij) * inv_dx2
                        } else {
                            0.0
                        }
                    }
                };

                let d2vdy2 = match stencil {
                    DerivativeStencil::Central => {
                        if j > 0 && j < ny - 1 {
                            (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) * inv_dy2
                        } else if j == 0 && ny > 2 {
                            (self.values[[2, i]] - 2.0 * self.values[[1, i]] + v_ij) * inv_dy2
                        } else if j == ny - 1 && ny > 2 {
                            (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) * inv_dy2
                        } else {
                            0.0
                        }
                    }
                    DerivativeStencil:: Forward => {
                        if j + 2 < ny {
                            (self.values[[j + 2, i]] - 2.0 * self.values[[j + 1, i]] + v_ij) * inv_dy2
                        } else if j + 1 < ny && j > 0 {
                            (self. values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) * inv_dy2
                        } else if j > 1 {
                            (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) * inv_dy2
                        } else {
                            0.0
                        }
                    }
                    DerivativeStencil:: Backward => {
                        if j >= 2 {
                            (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) * inv_dy2
                        } else if j > 0 && j + 1 < ny {
                            (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) * inv_dy2
                        } else if j + 2 < ny {
                            (self.values[[j + 2, i]] - 2.0 * self.values[[j + 1, i]] + v_ij) * inv_dy2
                        } else {
                            0.0
                        }
                    }
                };

                d2vdx2 + d2vdy2
            })
            .collect();

        Array2::from_shape_vec((ny, nx), laplace).unwrap()
    }

    /* ================ Spectral methods =================================== */
    /// Convert real ndarray (row-major) to contiguous complex vector
    ///
    /// # Arguments
    /// * `values` - 2D array to be converted
    fn ndarray_to_complex_buffer(values: &Array2<f64>) -> Vec<Complex<f64>> {
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
    ) -> Array2<Complex<f64>> {
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
    pub fn fft2d(&self) -> Result<Array2<Complex<f64>>> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        if !Self::is_base2(nx) || !Self::is_base2(ny) {
            return Err(LensForgeError::InvalidGridDimensions(format!(
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
    pub fn ifft2d(arr: &Array2<Complex<f64>>) -> Result<Array2<f64>> {
        let nx = arr.ncols();
        let ny = arr.nrows();

        if !Self::is_base2(nx) || !Self::is_base2(ny) {
            return Err(LensForgeError::InvalidGridDimensions(format!(
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
        Ok(Array2::from_shape_vec((ny, nx), values_re)
            .map_err(|e| LensForgeError::NumericalError(e.to_string()))?)
    }

    /// Compute FFT shift (move zero-frequency component to center)
    ///
    /// # Arguments
    /// * `arr` - complex array to be shifted
    pub fn fftshift(arr: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
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
    pub fn ifftshift(arr: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let nx = arr.ncols();
        let ny = arr.nrows();
        let nx_half = (nx + 1) / 2;
        let ny_half = (ny + 1) / 2;

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

    /// Compute Laplacian using spectral method (using FFT)
    ///
    /// Note: assumes periodic boundaries
    pub fn laplacian_spectral(&self) -> Result<Array2<f64>> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        let mut fft_data = self.fft2d()?;

        let (kx, ky) = self.grid.wavenumbers();
        let kx2 = kx.mapv(|k| k * k);
        let ky2 = ky.mapv(|k| k * k);
        for j in 0..ny {
            for i in 0..nx {
                let k2 = kx2[i] + ky2[j];
                fft_data[[j, i]] *= Complex::new(-k2, 0.0);
            }
        }

        Self::ifft2d(&fft_data)
    }

    /// Compute Laplacian using spectral method with given squared wavenumbers (using FFT)
    ///
    /// Note: assumes periodic boundaries
    pub fn laplacian_spectral_k(
        &self,
        kx2: &Array1<f64>,
        ky2: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        let mut fft_data = self.fft2d()?;

        for j in 0..ny {
            for i in 0..nx {
                let k2 = kx2[i] + ky2[j];
                fft_data[[j, i]] *= Complex::new(-k2, 0.0);
            }
        }

        Self::ifft2d(&fft_data)
    }

    /// Compute gradient using spectral method (using FFT)
    pub fn gradient_spectral(&self) -> Result<(Array2<f64>, Array2<f64>)> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        let fft_data = self.fft2d()?;

        let (kx, ky) = self.grid.wavenumbers();

        let mut dvdx_fft = fft_data.clone();
        let mut dvdy_fft = fft_data.clone();
        for j in 0..ny {
            for i in 0..nx {
                let f_hat = fft_data[[j, i]];
                dvdx_fft[[j, i]] = Complex::new(-f_hat.im * kx[i], f_hat.re * kx[i]);
                dvdy_fft[[j, i]] = Complex::new(-f_hat.im * ky[j], f_hat.re * ky[j]);
            }
        }

        let dvdx = Self::ifft2d(&dvdx_fft)?;
        let dvdy = Self::ifft2d(&dvdy_fft)?;

        Ok((dvdx, dvdy))
    }

    /// Compute gradient using spectral method with given wavenumbers (using FFT)
    ///
    /// # Arguments
    /// * `kx` - wavenumbers along the x axis
    /// * `ky` - wavenumbers along the y axis
    pub fn gradient_spectral_k(
        &self,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        let fft_data = self.fft2d()?;

        let mut dvdx_fft = fft_data.clone();
        let mut dvdy_fft = fft_data.clone();
        for j in 0..ny {
            for i in 0..nx {
                let f_hat = fft_data[[j, i]];
                dvdx_fft[[j, i]] = Complex::new(-f_hat.im * kx[i], f_hat.re * kx[i]);
                dvdy_fft[[j, i]] = Complex::new(-f_hat.im * ky[j], f_hat.re * ky[j]);
            }
        }

        let dvdx = Self::ifft2d(&dvdx_fft)?;
        let dvdy = Self::ifft2d(&dvdy_fft)?;

        Ok((dvdx, dvdy))
    }
}

/* ================ Derivative methods =============================== */
/// Discrete derivative stencil type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DerivativeStencil {
    /// Forward difference:  (f[i+1] - f[i]) / dx
    Forward,
    /// Backward difference: (f[i] - f[i-1]) / dx
    Backward,
    /// Central difference:  (f[i+1] - f[i-1]) / (2*dx)
    #[default]
    Central,
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

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_pixel_grid_zeros() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::zeros(grid);

        assert_eq!(pg.values.shape(), &[4, 4]);
        assert!(approx_eq(pg.sum(), 0.0));
    }

    #[test]
    fn test_pixel_grid_ones() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::ones(grid);

        assert_eq!(pg.values.shape(), &[4, 4]);
        assert!(approx_eq(pg.sum(), 16.0));
    }

    #[test]
    fn test_pixel_grid_constant() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::constant(grid, 3.5);

        assert!(approx_eq(pg.mean(), 3.5));
    }

    #[test]
    fn test_pixel_grid_from_func() {
        let grid = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let center_val = pg.get(2, 2);
        let expected = grid.x_at(2).powi(2) + grid.y_at(2).powi(2);
        assert!(approx_eq(center_val, expected));
    }

    #[test]
    fn test_pixel_grid_get_set() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let mut pg = PixelGrid::zeros(grid);

        pg.set(1, 2, 5.0);
        assert!(approx_eq(pg.get(1, 2), 5.0));

        assert!(pg.get_safely(1, 2).is_some());
        assert!(pg.get_safely(10, 10).is_none());
    }

    #[test]
    fn test_pixel_grid_statistics() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::from_func(grid, |_, _| 2.0);

        assert!(approx_eq(pg.sum(), 32.0));
        assert!(approx_eq(pg.mean(), 2.0));
        assert!(approx_eq(pg.min(), 2.0));
        assert!(approx_eq(pg.max(), 2.0));
        assert!(approx_eq(pg.var(), 0.0));
        assert!(approx_eq(pg.std(), 0.0));
    }

    #[test]
    fn test_pixel_grid_scale_shift() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg = PixelGrid::ones(grid);

        let scaled = pg.scale(2.0);
        assert!(approx_eq(scaled.mean(), 2.0));

        let shifted = pg.shift(3.0);
        assert!(approx_eq(shifted.mean(), 4.0));
    }

    #[test]
    fn test_pixel_grid_add_sub_mul() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let pg1 = PixelGrid::constant(grid, 2.0);
        let pg2 = PixelGrid::constant(grid, 3.0);

        let added = pg1.add(&pg2).unwrap();
        assert!(approx_eq(added.mean(), 5.0));

        let subtracted = pg1.sub(&pg2).unwrap();
        assert!(approx_eq(subtracted.mean(), -1.0));

        let multiplied = pg1.mul(&pg2).unwrap();
        assert!(approx_eq(multiplied.mean(), 6.0));
    }

    #[test]
    fn test_pixel_grid_normalize() {
        let grid = Grid2D::square(4, 1.0).unwrap();
        let mut pg = PixelGrid::constant(grid, 2.0);

        pg.normalize_sum(Some(1.0));
        assert!(approx_eq(pg.sum(), 1.0));

        let mut pg2 = PixelGrid::constant(grid, 2.0);
        pg2.normalize_max(Some(1.0));
        assert!(approx_eq(pg2.max(), 1.0));
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
        assert!(approx_eq(orig_val, resampled_val));
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
                assert!(approx_eq(resampled_seq.get(i, j), resampled_par.get(i, j)));
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
                assert!(approx_eq(pg.values[[j, i]], reconstructed[[j, i]]));
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
                assert!(approx_eq(fft_result[[j, i]].re, unshifted[[j, i]].re));
                assert!(approx_eq(fft_result[[j, i]].im, unshifted[[j, i]].im));
            }
        }
    }

    #[test]
    fn test_pixel_grid_interpolation() {
        let grid = Grid2D::square(4, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, _| x);

        let center = Point2D::origin();
        let interp = pg.interpolate(&center);

        assert!(approx_eq(interp, 0.0));
    }

    #[test]
    fn test_pixel_grid_gradient() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| 2.0 * x + 3.0 * y);

        let grad = pg.gradient(4, 4, DerivativeStencil::Central);
        assert!(approx_eq(grad.x, 2.0));
        assert!(approx_eq(grad.y, 3.0));
    }

    #[test]
    fn test_pixel_grid_laplacian() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let lap = pg.laplacian(4, 4, DerivativeStencil::Central);
        assert!(approx_eq(lap, 4.0));
    }

    #[test]
    fn test_pixel_grid_gradient_field() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| 2.0 * x + 3.0 * y);

        let (dvdx, dvdy) = pg.gradient_field(None);
        let (par_dvdx, par_dvdy) = pg.par_gradient_field(None);

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(dvdx[[j, i]], par_dvdx[[j, i]]));
                assert!(approx_eq(dvdy[[j, i]], par_dvdy[[j, i]]));
            }
        }
    }

    #[test]
    fn test_pixel_grid_laplacian_field() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let laplace = pg.laplacian_field(None);
        let par_laplace = pg.par_laplacian_field(None);

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(laplace[[j, i]], par_laplace[[j, i]]));
            }
        }
    }

    #[test]
    fn test_pixel_grid_gradient_spectral() {
        let grid = Grid2D::square(64, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x + 2.0 * y);

        let (dvdx, dvdy) = pg.gradient_spectral().unwrap();

        // Check interior
        for j in 16..48 {
            for i in 16..48 {
                assert!(
                    approx_eq(dvdx[[j, i]], 1.0),
                    "dvdx at ({}, {}) = {}, expected 1.0",
                    i,
                    j,
                    dvdx[[j, i]]
                );
                assert!(
                    approx_eq(dvdy[[j, i]], 2.0),
                    "dvdy at ({}, {}) = {}, expected 2.0",
                    i,
                    j,
                    dvdy[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_pixel_grid_laplacian_spectral() {
        let grid = Grid2D::square(64, 2.0).unwrap();
        let k = 2.0 * std::f64::consts::PI / 4.0;
        let pg = PixelGrid::from_func(grid, |x, y| (k * x).cos() * (k * y).cos());

        let laplace_fft = pg.laplacian_spectral().unwrap();
        let f_expected = -2.0 * k * k;

        for j in 0..64 {
            for i in 0..64 {
                let expected = f_expected * pg.values[[j, i]];
                assert!(
                    approx_eq(laplace_fft[[j, i]], expected),
                    "Laplacian at ({}, {}) = {}, expected {}",
                    i,
                    j,
                    laplace_fft[[j, i]],
                    expected
                );
            }
        }
    }
}
