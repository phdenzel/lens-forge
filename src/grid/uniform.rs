// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens_forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Uniform grid with pixels

use core::f64;

use crate::grid::{Grid2D, Point2D};
use crate::{LensForgeError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use num::complex::Complex;
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
    pub fn as_array2(&self) -> ArrayView2<f64> {
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
    pub fn row(&self, j: usize) -> ArrayView1<f64> {
        self.values.row(j)
    }

    /// Get column i as a slice
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    pub fn column(&self, i: usize) -> ArrayView1<f64> {
        self.values.column(i)
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
    #[inline]
    pub fn gradient(&self, i: usize, j: usize) -> Point2D {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        let dvdx = if i == 0 {
            (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx
        } else if i == nx - 1 {
            (self.values[[j, nx - 1]] - self.values[[j, nx - 2]]) / self.grid.dx
        } else {
            (self.values[[j, i + 1]] - self.values[[j, i - 1]]) / (2.0 * self.grid.dx)
        };

        let dvdy = if j == 0 {
            (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy
        } else if j == ny - 1 {
            (self.values[[ny - 1, i]] - self.values[[ny - 2, i]]) / self.grid.dy
        } else {
            (self.values[[j + 1, i]] - self.values[[j - 1, i]]) / (2.0 * self.grid.dy)
        };

        Point2D::new(dvdx, dvdy)
    }

    /// Compute gradient at arbitrary point using interpolation
    ///
    /// # Arguments
    /// * `p` - point coordinate for gradient calculation
    pub fn gradient_interp(&self, p: &Point2D) -> Point2D {
        let h_x = self.grid.dx;
        let h_y = self.grid.dy;

        let dvdx = (self.interpolate(&Point2D::new(p.x + h_x, p.y))
            - self.interpolate(&Point2D::new(p.x - h_x, p.y)))
            / (2.0 * h_x);
        let dvdy = (self.interpolate(&Point2D::new(p.x, p.y + h_y))
            - self.interpolate(&Point2D::new(p.x, p.y - h_y)))
            / (2.0 * h_y);

        Point2D::new(dvdx, dvdy)
    }

    /// Compute Laplacian at a pixel (i, j) using 5-point stencil
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    #[inline]
    pub fn laplacian(&self, i: usize, j: usize) -> f64 {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let v_ij = self.values[[j, i]];

        let d2vdx2 = if i == 0 {
            if nx > 2 {
                (self.values[[j, 2]] - 2.0 * self.values[[j, 1]] + v_ij) / dx2
            } else {
                0.0
            }
        } else if i == nx - 1 {
            if nx > 2 {
                (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]]) / dx2
            } else {
                0.0
            }
        } else {
            self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]] / dx2
        };

        let d2vdy2 = if j == 0 {
            if ny > 2 {
                (self.values[[2, i]] - 2.0 * self.values[[1, i]] + v_ij) / dy2
            } else {
                0.0
            }
        } else if j == ny - 1 {
            if ny > 2 {
                (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) / dy2
            } else {
                0.0
            }
        } else {
            (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) / dy2
        };

        d2vdx2 + d2vdy2
    }

    /// Compute Laplacian at arbitrary point using interpolation
    ///
    /// # Arguments
    /// * `p` - point coordinate for gradient calculation
    pub fn laplacian_interp(&self, p: &Point2D) -> f64 {
        let h_x = self.grid.dx;
        let h_y = self.grid.dy;

        let v_00 = self.interpolate(p);
        let v_mx = self.interpolate(&Point2D::new(p.x - h_x, p.y));
        let v_px = self.interpolate(&Point2D::new(p.x + h_x, p.y));
        let v_my = self.interpolate(&Point2D::new(p.x, p.y - h_y));
        let v_py = self.interpolate(&Point2D::new(p.x, p.y + h_y));

        (v_mx + v_px - 2.0 * v_00) / (h_x * h_x) + (v_my + v_py - 2.0 * v_00) / (h_y * h_y)
    }

    /// Compute gradient field over the entire grid
    pub fn gradient_field(&self) -> (Array2<f64>, Array2<f64>) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_2dx = 1.0 / (2.0 * self.grid.dx);
        let inv_2dy = 1.0 / (2.0 * self.grid.dy);
        let inv_dx = 1.0 / self.grid.dx;
        let inv_dy = 1.0 / self.grid.dy;

        let mut dvdx = Array2::zeros((ny, nx));
        let mut dvdy = Array2::zeros((ny, nx));

        for j in 0..ny {
            for i in 1..nx - 1 {
                dvdx[[j, i]] = (self.values[[j, i + 1]] - self.values[[j, i - 1]]) * inv_2dx;
            }
            if nx > 1 {
                dvdx[[j, 0]] = (self.values[[j, 1]] - self.values[[j, 0]]) * inv_dx;
                dvdx[[j, nx - 1]] = (self.values[[j, nx - 1]] - self.values[[j, nx - 2]]) * inv_dx;
            }
        }

        for i in 0..nx {
            for j in 1..ny - 1 {
                dvdy[[j, i]] = (self.values[[j + 1, i]] - self.values[[j - 1, i]]) * inv_2dy;
            }
            if ny > 1 {
                dvdy[[0, i]] = (self.values[[1, i]] - self.values[[0, i]]) * inv_dy;
                dvdy[[ny - 1, i]] = (self.values[[ny - 1, i]] - self.values[[ny - 2, i]]) * inv_dy;
            }
        }

        (dvdx, dvdy)
    }

    /// Compute gradient field over the entire grid in parallel (using rayon backend)
    pub fn par_gradient_field(&self) -> (Array2<f64>, Array2<f64>) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_2dx = 1.0 / (2.0 * self.grid.dx);
        let inv_2dy = 1.0 / (2.0 * self.grid.dy);
        let inv_dx = 1.0 / self.grid.dx;
        let inv_dy = 1.0 / self.grid.dy;

        let gradients: Vec<(f64, f64)> = (0..ny * nx)
            .into_par_iter()
            .map(|idx| {
                let i = idx % nx;
                let j = idx / nx;

                let dvdx = if i == 0 {
                    if nx > 1 {
                        (self.values[[j, 1]] - self.values[[j, 0]]) * inv_dx
                    } else {
                        0.0
                    }
                } else if i == nx - 1 {
                    (self.values[[j, nx - 1]] - self.values[[j, nx - 2]]) * inv_dx
                } else {
                    (self.values[[j, i + 1]] - self.values[[j, i - 1]]) * inv_2dx
                };

                let dvdy = if j == 0 {
                    if ny > 1 {
                        (self.values[[1, i]] - self.values[[0, i]]) * inv_dy
                    } else {
                        0.0
                    }
                } else if j == ny - 1 {
                    (self.values[[ny - 1, i]] - self.values[[ny - 2, i]]) * inv_dy
                } else {
                    (self.values[[j + 1, i]] - self.values[[j - 1, i]]) * inv_2dy
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
    pub fn gradient_magnitude(&self) -> Array2<f64> {
        let (dvdx, dvdy) = self.gradient_field();
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
    pub fn par_gradient_magnitude(&self) -> Array2<f64> {
        let (dvdx, dvdy) = self.par_gradient_field();
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
    pub fn laplacian_field(&self) -> Array2<f64> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx2 = 1.0 / (self.grid.dx * self.grid.dx);
        let inv_dy2 = 1.0 / (self.grid.dy * self.grid.dy);

        let mut laplace = Array2::zeros((ny, nx));

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let v_ij2 = 2.0 * self.values[[j, i]];
                laplace[[j, i]] = (self.values[[j, i + 1]] + self.values[[j, i - 1]] - v_ij2)
                    * inv_dx2
                    + (self.values[[j + 1, i]] + self.values[[j - 1, i]] - v_ij2) * inv_dy2;
            }
        }
        if ny > 2 {
            for i in 1..nx - 1 {
                let v_0i = self.values[[0, i]];
                laplace[[0, i]] = (self.values[[0, i + 1]] + self.values[[0, i - 1]] - 2.0 * v_0i)
                    * inv_dx2
                    + (self.values[[2, i]] - 2.0 * self.values[[1, i]] + v_0i) * inv_dy2;
                let v_ni = self.values[[ny - 1, i]];
                laplace[[ny - 1, i]] = (self.values[[ny - 1, i + 1]]
                    + self.values[[ny - 1, i - 1]]
                    - 2.0 * v_ni)
                    * inv_dx2
                    + (v_ni - 2.0 * self.values[[ny - 2, i]] + self.values[[ny - 3, i]]) * inv_dy2;
            }
        }
        if nx > 2 {
            for j in 1..ny - 1 {
                let v_j0 = self.values[[j, 0]];
                laplace[[j, 0]] = (self.values[[j, 2]] - 2.0 * self.values[[j, 1]] + v_j0)
                    * inv_dx2
                    + (self.values[[j + 1, 0]] + self.values[[j - 1, 0]] - 2.0 * v_j0) * inv_dy2;

                let v_jn = self.values[[j, nx - 1]];
                laplace[[j, nx - 1]] = (v_jn - 2.0 * self.values[[j, nx - 2]]
                    + self.values[[j, nx - 3]])
                    * inv_dx2
                    + (self.values[[j + 1, nx - 1]] + self.values[[j - 1, nx - 1]] - 2.0 * v_jn)
                        * inv_dy2;
            }
        }
        if nx > 2 && ny > 2 {
            let v00 = self.values[[0, 0]];
            let v0n = self.values[[0, nx - 1]];
            let vn0 = self.values[[ny - 1, 0]];
            let vnn = self.values[[ny - 1, nx - 1]];
            laplace[[0, 0]] = (self.values[[0, 2]] - 2.0 * self.values[[0, 1]] + v00) * inv_dx2
                + (self.values[[2, 0]] - 2.0 * self.values[[1, 0]] + v00) * inv_dy2;

            laplace[[0, nx - 1]] =
                (v0n - 2.0 * self.values[[0, nx - 2]] + self.values[[0, nx - 3]]) * inv_dx2
                    + (self.values[[2, nx - 1]] - 2.0 * self.values[[1, nx - 1]] + v0n) * inv_dy2;

            laplace[[ny - 1, 0]] =
                (self.values[[ny - 1, 2]] - 2.0 * self.values[[ny - 1, 1]] + vn0) * inv_dx2
                    + (vn0 - 2.0 * self.values[[ny - 2, 0]] + self.values[[ny - 3, 0]]) * inv_dy2;

            laplace[[ny - 1, nx - 1]] = (vnn - 2.0 * self.values[[ny - 1, nx - 2]]
                + self.values[[ny - 1, nx - 3]])
                * inv_dx2
                + (vnn - 2.0 * self.values[[ny - 2, nx - 1]] + self.values[[ny - 3, nx - 1]])
                    * inv_dy2;
        }

        laplace
    }

    /// Compute Laplace field over the entire grid
    pub fn par_laplacian_field(&self) -> Array2<f64> {
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

                let d2vdx2 = if i == 0 {
                    if nx > 2 {
                        (self.values[[j, 2]] - 2.0 * self.values[[j, 1]] + v_ij) * inv_dx2
                    } else if nx > 1 {
                        (self.values[[j, 1]] - v_ij) * inv_dx2
                    } else {
                        0.0
                    }
                } else if i == nx - 1 {
                    if nx > 2 {
                        (v_ij - 2.0 * self.values[[j, i - 1]] + self.values[[j, i - 2]]) * inv_dx2
                    } else {
                        (v_ij - self.values[[j, i - 1]]) * inv_dx2
                    }
                } else {
                    (self.values[[j, i + 1]] - 2.0 * v_ij + self.values[[j, i - 1]]) * inv_dx2
                };

                let d2vdy2 = if j == 0 {
                    if ny > 2 {
                        (self.values[[2, i]] - 2.0 * self.values[[1, i]] + v_ij) * inv_dy2
                    } else if ny > 1 {
                        (self.values[[1, i]] - v_ij) * inv_dy2
                    } else {
                        0.0
                    }
                } else if j == ny - 1 {
                    if ny > 2 {
                        (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) * inv_dy2
                    } else {
                        (v_ij - self.values[[j - 1, i]]) * inv_dy2
                    }
                } else {
                    (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) * inv_dy2
                };

                d2vdx2 + d2vdy2
            })
            .collect();

        Array2::from_shape_vec((ny, nx), laplace).unwrap()
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
    // /// Resample to a different grid using bilinear interpolation
    // ///
    // /// # Arguments
    // /// * `grid` -
    // pub fn resample(&mut self, grid: &Grid2D) -> Self {
    // 	Self::from_point_func(
    // 	    // TODO
    // 	)
    // }

    // /// Resample to a different grid using bilinear interpolation in parallel (using rayon backend)
    // ///
    // /// # Arguments
    // /// * `grid` -
    // pub fn par_resample(&mut self, grid: &Grid2D) -> Self {
    // 	Self::from_point_func(
    // 	    // TODO
    // 	)
    // }

    /* ================ FFT adapters ======================================= */
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

    // ///Compute 2D FFT (using phastft backend)
    // pub fn fft2d(&self) -> Result<Array2<Complex<f64>>> {
    // 	// TODO
    // }

    // /// Compute inverse 2D FFT (using phastft backend)
    // ///
    // /// # Arguments
    // /// * `complex` - complex ndarray to be inverse transformed
    // pub fn ifft2d(complex:  &Array2<Complex<f64>>) -> Result<Array2<f64>> {
    // 	// TODO
    // }
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
}
