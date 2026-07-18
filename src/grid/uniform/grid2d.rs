// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Basic 2D grid type

use crate::grid::{BBox, Point2D};
use crate::{LensForgeError, Result};
use ndarray::{Array1, Array2, Axis, Zip, parallel::prelude::*};

/// A 2D grid for discretizing planes
///
/// All coordinate arrays are computed lazily on demand.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Grid2D {
    /// Number of pixels in x direction
    pub nx: usize,
    /// Number of pixels in y direction
    pub ny: usize,
    /// Physical domain (lower-left = `min`, upper-right = `max`).
    pub bbox: BBox,
}

impl Grid2D {
    /// Create a new 2D grid
    ///
    /// # Arguments
    /// * `nx` - Number of pixels in the x direction
    /// * `ny` - Number of pixels in the y direction
    /// * `bbox` - Physical domain of the grid
    ///
    /// # Errors
    /// Returns an error if dimensions are zero or the domain is degenerate
    #[inline]
    pub fn new(nx: usize, ny: usize, bbox: BBox) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(LensForgeError::invalid_grid_dimension(
                "Grid size must be strictly positive",
            ));
        }
        if !bbox.is_positive() {
            return Err(LensForgeError::invalid_grid_dimension(
                "Grid extent minimum must be less than maximum",
            ));
        }

        Ok(Self { nx, ny, bbox })
    }

    /// Create a square grid centred at the origin
    ///
    /// # Arguments
    /// * `n` - Number of pixels in each direction
    /// * `radius` - Half the physical extent (grid spans from -radius to +radius)
    pub fn square(n: usize, radius: f64) -> Result<Self> {
        if radius <= 0.0 {
            return Err(LensForgeError::invalid_grid_dimension(
                "Radius must be positive",
            ));
        }
        Self::new(n, n, BBox::from_lims(-radius, radius, -radius, radius))
    }

    /// Create a rectangular grid centred at the origin
    ///
    /// # Arguments
    /// * `nx` - Number of pixels in the x direction
    /// * `ny` - Number of pixels in the y direction
    /// * `rx` - Half the physical extent in x (grid spans from -rx to +rx)
    /// * `ry` - Half the physical extent in y (grid spans from -ry to +ry)
    pub fn rectangular(nx: usize, ny: usize, rx: f64, ry: f64) -> Result<Self> {
        if rx <= 0.0 || ry <= 0.0 {
            return Err(LensForgeError::invalid_grid_dimension(
                "Extents must be positive",
            ));
        }
        Self::new(nx, ny, BBox::from_lims(-rx, rx, -ry, ry))
    }

    /// Create a grid with given physical extent and resolution
    ///
    /// # Arguments
    /// * `extent_x` - Physical extent in x as (x_min, x_max)
    /// * `extent_y` - Physical extent in y as (y_min, y_max)
    /// * `resolution` - Grid resolution in x and y
    pub fn with_resolution(
        extent_x: (f64, f64),
        extent_y: (f64, f64),
        resolution: f64,
    ) -> Result<Self> {
        if resolution <= 0.0 {
            return Err(LensForgeError::invalid_grid_dimension(
                "Resolution must be positive",
            ));
        }
        let nx = ((extent_x.1 - extent_x.0) * resolution).ceil() as usize;
        let ny = ((extent_y.1 - extent_y.0) * resolution).ceil() as usize;
        if nx < 1 {
            eprintln!("Warning: calculated non-positive grid size in x, setting to 1.");
        }
        if ny < 1 {
            eprintln!("Warning: calculated non-positive grid size in y, setting to 1.");
        }
        Self::new(
            nx.max(1),
            ny.max(1),
            BBox::from_lims(extent_x.0, extent_x.1, extent_y.0, extent_y.1),
        )
    }

    /* ============ Grid attributes ======================================== */
    /// Get x extent as tuple
    #[inline]
    pub fn extent_x(&self) -> (f64, f64) {
        self.bbox.extent_x()
    }

    /// Get y extent as tuple
    #[inline]
    pub fn extent_y(&self) -> (f64, f64) {
        self.bbox.extent_y()
    }

    /// Pixel size in x direction (`= bbox.width() / nx`)
    #[inline]
    pub fn dx(&self) -> f64 {
        self.bbox.width() / self.nx as f64
    }

    /// Pixel size in y direction (`= bbox.height() / ny`)
    #[inline]
    pub fn dy(&self) -> f64 {
        self.bbox.height() / self.ny as f64
    }

    /// Grid width
    #[inline]
    pub fn width(&self) -> f64 {
        self.bbox.width()
    }

    /// Grid height
    #[inline]
    pub fn height(&self) -> f64 {
        self.bbox.height()
    }

    /// Grid centre
    #[inline]
    pub fn center(&self) -> Point2D {
        self.bbox.center()
    }

    /// Aspect ratio (width / height)
    #[inline]
    pub fn aspect_ratio(&self) -> f64 {
        self.width() / self.height()
    }

    /// Total number of pixels
    #[inline(always)]
    pub fn n_pixels(&self) -> usize {
        self.nx * self.ny
    }

    /// Pixel area
    #[inline(always)]
    pub fn pixel_area(&self) -> f64 {
        self.dx() * self.dy()
    }

    /// Total grid area
    #[inline(always)]
    pub fn total_area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Wavenumber of the grid
    #[inline(always)]
    pub fn wavenumbers(&self) -> (Array1<f64>, Array1<f64>) {
        let nx = self.nx;
        let ny = self.ny;
        let dkx = 2.0 * std::f64::consts::PI / self.width();
        let dky = 2.0 * std::f64::consts::PI / self.height();

        let kx = Array1::from_shape_fn(nx, |i| {
            if i <= nx / 2 {
                i as f64 * dkx
            } else {
                (i as i64 - nx as i64) as f64 * dkx
            }
        });
        let ky = Array1::from_shape_fn(ny, |j| {
            if j <= ny / 2 {
                j as f64 * dky
            } else {
                (j as i64 - ny as i64) as f64 * dky
            }
        });

        (kx, ky)
    }

    /// Shape as (ny, nx) tuple
    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    /// Shape as [ny, nx] array
    #[inline(always)]
    pub fn shape_as_array(&self) -> [usize; 2] {
        [self.ny, self.nx]
    }

    /* ============ Coordinates ============================================ */
    /// Get x-coordinate of pixel centre
    ///
    /// # Arguments
    /// * `i` - index along x axis
    #[inline(always)]
    pub fn x_at(&self, i: usize) -> f64 {
        self.bbox.min.x + (i as f64 + 0.5) * self.dx()
    }

    /// Get y-coordinate of pixel centre
    ///
    /// # Arguments
    /// * `j` - index along y axis
    #[inline(always)]
    pub fn y_at(&self, j: usize) -> f64 {
        self.bbox.min.y + (j as f64 + 0.5) * self.dy()
    }

    /// Get pixel centre coordinates
    ///
    /// # Arguments
    /// * `i` - index along x axis
    /// * `j` - index along y axis
    #[inline(always)]
    pub fn pixel_center(&self, i: usize, j: usize) -> Point2D {
        Point2D::new(self.x_at(i), self.y_at(j))
    }

    /// Get x-coordinate of pixel corner
    ///
    /// # Arguments
    /// * `i` - index along x axis
    #[inline(always)]
    pub fn x_corner_at(&self, i: usize) -> f64 {
        self.bbox.min.x + i as f64 * self.dx()
    }

    /// Get y-coordinate of pixel corner
    ///
    /// # Arguments
    /// * `j` - index along y axis
    #[inline(always)]
    pub fn y_corner_at(&self, j: usize) -> f64 {
        self.bbox.min.y + j as f64 * self.dy()
    }

    /// Get pixel corner (bottom-left)
    ///
    /// # Arguments
    /// * `i` - index along x axis
    /// * `j` - index along y axis
    #[inline(always)]
    pub fn pixel_corner(&self, i: usize, j: usize) -> Point2D {
        Point2D::new(self.x_corner_at(i), self.y_corner_at(j))
    }

    /// Convert point to pixel indices
    ///
    /// # Arguments
    /// * `p` - point to be mapped
    #[inline(always)]
    pub fn point_to_pixel(&self, p: &Point2D) -> Option<(usize, usize)> {
        let fi = (p.x - self.bbox.min.x) / self.dx();
        let fj = (p.y - self.bbox.min.y) / self.dy();

        if fi >= 0.0 && fi < self.nx as f64 && fj >= 0.0 && fj < self.ny as f64 {
            Some((fi as usize, fj as usize))
        } else {
            None
        }
    }

    /// Convert point to fractional pixel coordinates (relative to centres)
    ///
    /// # Arguments
    /// * `p` - point to be mapped
    #[inline(always)]
    pub fn relative_to(&self, p: &Point2D) -> (f64, f64) {
        (
            (p.x - self.bbox.min.x) / self.dx() - 0.5,
            (p.y - self.bbox.min.y) / self.dy() - 0.5,
        )
    }

    /// Convert point to fractional pixel coordinates (relative to corners)
    ///
    /// # Arguments
    /// * `p` - point to be mapped
    #[inline(always)]
    pub fn relative_to_corner(&self, p: &Point2D) -> (f64, f64) {
        ((p.x - self.bbox.min.x) / self.dx(), (p.y - self.bbox.min.y) / self.dy())
    }

    /// Check if point is inside grid
    ///
    /// # Arguments
    /// * `p` - point to be checked
    #[inline(always)]
    pub fn contains(&self, p: &Point2D) -> bool {
        self.bbox.contains(p)
    }

    /// Check if pixel indices are valid
    ///
    /// # Arguments
    /// * `i` - index along x axis
    /// * `j` - index along y axis
    #[inline(always)]
    pub fn is_valid_pixel(&self, i: usize, j: usize) -> bool {
        i < self.nx && j < self.ny
    }

    /// Convert (i, j) to flat index (row-major)
    ///
    /// # Arguments
    /// * `i` - index along x axis
    /// * `j` - index along y axis
    #[inline(always)]
    pub fn flatten_index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Convert flat index to (i, j)
    ///
    /// # Arguments
    /// * `idx` - index to be unflattened
    #[inline(always)]
    pub fn unflatten_index(&self, idx: usize) -> (usize, usize) {
        (idx % self.nx, idx / self.nx)
    }

    /// Create 1D array of x coordinates for pixel centres
    #[inline]
    pub fn x_coords(&self) -> Array1<f64> {
        let (min_x, dx) = (self.bbox.min.x, self.dx());
        Array1::from_shape_fn(self.nx, |i| min_x + (i as f64 + 0.5) * dx)
    }

    /// Create 1D array of y coordinates for pixel centres
    #[inline]
    pub fn y_coords(&self) -> Array1<f64> {
        let (min_y, dy) = (self.bbox.min.y, self.dy());
        Array1::from_shape_fn(self.ny, |j| min_y + (j as f64 + 0.5) * dy)
    }

    /// Create 1D array of x coordinates for pixel edges
    #[inline]
    pub fn x_edges(&self) -> Array1<f64> {
        let (min_x, dx) = (self.bbox.min.x, self.dx());
        Array1::from_shape_fn(self.nx + 1, |i| min_x + i as f64 * dx)
    }

    /// Create 1D array of y coordinates for pixel edges
    #[inline]
    pub fn y_edges(&self) -> Array1<f64> {
        let (min_y, dy) = (self.bbox.min.y, self.dy());
        Array1::from_shape_fn(self.ny + 1, |j| min_y + j as f64 * dy)
    }

    /* ============ Coordinate matrices ============================================== */
    /// Create coordinate matrix of x coordinates (non-broadcasting)
    #[inline]
    pub fn prealloc_coord_matrix_x(&self) -> Array2<f64> {
        let x_coords = self.x_coords();
        // pre-allocate array
        let mut result = Array2::zeros((self.ny, self.nx));
        // row-by-row assignment
        for mut row in result.rows_mut() {
            row.assign(&x_coords);
        }
        result
    }

    /// Create coordinate matrix of x coordinates using broadcasting
    #[inline]
    pub fn coord_matrix_x(&self) -> Array2<f64> {
        let x = self.x_coords();
        // reshape to (1, nx) and broadcast to (ny, nx)
        let x_row = x.into_shape_with_order((1, self.nx)).unwrap();
        x_row.broadcast((self.ny, self.nx)).unwrap().to_owned()
    }

    /// Create coordinate matrix of y coordinates (non-broadcasting)
    #[inline]
    pub fn prealloc_coord_matrix_y(&self) -> Array2<f64> {
        let y_coords = self.y_coords();
        // pre-allocate array
        let mut transposed = Array2::zeros((self.ny, self.nx));
        // row-by-row assignment
        for mut row in transposed.rows_mut() {
            row.assign(&y_coords);
        }
        transposed.reversed_axes().as_standard_layout().into_owned()
    }

    /// Create coordinate matrix of y coordinates using broadcasting
    #[inline]
    pub fn coord_matrix_y(&self) -> Array2<f64> {
        let y = self.y_coords();
        // reshape to (ny, 1) and broadcast to (ny, nx)
        let y_col = y.into_shape_with_order((self.ny, 1)).unwrap();
        y_col.broadcast((self.ny, self.nx)).unwrap().to_owned()
    }

    /// Create x and y coordinate matrices (non-broadcasting)
    pub fn prealloc_coord_matrix(&self) -> (Array2<f64>, Array2<f64>) {
        (self.prealloc_coord_matrix_x(), self.prealloc_coord_matrix_y())
    }

    /// Create x and y coordinate matrices using broadcasting
    pub fn coord_matrix(&self) -> (Array2<f64>, Array2<f64>) {
        (self.coord_matrix_x(), self.coord_matrix_y())
    }

    /// Compute squared radial distance coordinate matrix
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid centre)
    pub fn coord_matrix_r2(&self, center: Option<Point2D>) -> Array2<f64> {
        let c = center.unwrap_or_else(|| self.center());
        let cx = c.x;
        let cy = c.y;
        // Precompute (x - cx)^2 for each column and (y - cy)^2 for each row
        let dx2: Array1<f64> = self.x_coords().mapv(|x| {
            let d = x - cx;
            d * d
        });
        let dy2: Array1<f64> = self.y_coords().mapv(|y| {
            let d = y - cy;
            d * d
        });

        let mut result = Array2::zeros((self.ny, self.nx));
        for (j, mut row) in result.rows_mut().into_iter().enumerate() {
            let dy2_j = dy2[j];
            Zip::from(&mut row).and(&dx2).for_each(|r, &dx2_i| {
                *r = dx2_i + dy2_j;
            });
        }
        result
    }

    /// Compute radial distance coordinate matrix
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid centre)
    pub fn coord_matrix_r(&self, center: Option<Point2D>) -> Array2<f64> {
        self.coord_matrix_r2(center).mapv(f64::sqrt)
    }

    /// Compute angular coordinate matrix with theta = atan2(y - cy, x - cx)
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid centre)
    pub fn coord_matrix_theta(&self, center: Option<Point2D>) -> Array2<f64> {
        let c = center.unwrap_or_else(|| self.center());
        let cx = c.x;
        let cy = c.y;
        let xs = self.x_coords();
        let ys = self.y_coords();

        Array2::from_shape_fn((self.ny, self.nx), |(j, i)| {
            let x = xs[i] - cx;
            let y = ys[j] - cy;
            y.atan2(x)
        })
    }

    /// Compute polar coordinate matrices (r, theta)
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid centre)
    pub fn coord_matrix_polar(&self, center: Option<Point2D>) -> (Array2<f64>, Array2<f64>) {
        let c = center.unwrap_or_else(|| self.center());

        // Precompute deltas
        let dx_vec: Array1<f64> = self.x_coords().mapv(|x| x - c.x);
        let dy_vec: Array1<f64> = self.y_coords().mapv(|y| y - c.y);
        let dx2_vec: Array1<f64> = dx_vec.mapv(|d| d * d);

        let mut r_grid = Array2::zeros((self.ny, self.nx));
        let mut theta_grid = Array2::zeros((self.ny, self.nx));

        for (j, (mut r_row, mut theta_row)) in r_grid
            .rows_mut()
            .into_iter()
            .zip(theta_grid.rows_mut())
            .enumerate()
        {
            let dy_j = dy_vec[j];
            let dy2_j = dy_j * dy_j;

            Zip::from(&mut r_row)
                .and(&mut theta_row)
                .and(&dx_vec)
                .and(&dx2_vec)
                .for_each(|r, theta, &dx_i, &dx2_i| {
                    *r = (dx2_i + dy2_j).sqrt();
                    *theta = dy_j.atan2(dx_i);
                });
        }

        (r_grid, theta_grid)
    }

    /// Create coordinate matrix of x coordinates in parallel (using rayon backend)
    pub fn par_coord_matrix_x(&self) -> Array2<f64> {
        let x_coords = self.x_coords();

        let mut result = Array2::zeros((self.ny, self.nx));
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                row.assign(&x_coords);
            });
        result
    }

    /// Create coordinate matrix of y coordinates in parallel (using rayon backend)
    pub fn par_coord_matrix_y(&self) -> Array2<f64> {
        let y_coords = self.y_coords();

        let mut result = Array2::zeros((self.ny, self.nx));
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut row)| {
                row.fill(y_coords[j]);
            });
        result
    }

    /// Create coordinate matrix of x and y coordinates in parallel (using rayon backend)
    pub fn par_coord_matrix(&self) -> (Array2<f64>, Array2<f64>) {
        // Run both in parallel using rayon's join
        rayon::join(|| self.par_coord_matrix_x(), || self.par_coord_matrix_y())
    }

    /// Compute squared radial distance coordinate matrix in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid centre)
    pub fn par_coord_matrix_r2(&self, center: Option<Point2D>) -> Array2<f64> {
        let c = center.unwrap_or_else(|| self.center());
        let dx2: Array1<f64> = self.x_coords().mapv(|x| {
            let d = x - c.x;
            d * d
        });
        let ys = self.y_coords();

        let mut result = Array2::zeros((self.ny, self.nx));
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut row)| {
                let dy = ys[j] - c.y;
                let dy2 = dy * dy;
                Zip::from(&mut row).and(&dx2).for_each(|r, &dx2_i| {
                    *r = dx2_i + dy2;
                });
            });
        result
    }

    /// Compute radial distance coordinate matrix in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid centre)
    pub fn par_coord_matrix_r(&self, center: Option<Point2D>) -> Array2<f64> {
        self.par_coord_matrix_r2(center).mapv(f64::sqrt)
    }

    /// Apply a function to each pixel centre coordinate
    ///
    /// # Arguments
    /// * `f` - function to be applied to each grid coordinate
    #[inline]
    pub fn map_centers<F, T>(&self, f: F) -> Array2<T>
    where
        F: Fn(f64, f64) -> T,
    {
        let xs = self.x_coords();
        let ys = self.y_coords();
        Array2::from_shape_fn((self.ny, self.nx), |(j, i)| f(xs[i], ys[j]))
    }

    /// Apply a function that takes Point2D
    ///
    /// # Arguments
    /// * `f` - function to be applied to each grid point
    #[inline]
    pub fn map_points<F, T>(&self, f: F) -> Array2<T>
    where
        F: Fn(&Point2D) -> T,
    {
        self.map_centers(|x, y| f(&Point2D::new(x, y)))
    }

    /// Apply a function to each pixel centre coordinate in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `f` - function to be applied to each grid coordinate
    pub fn par_map_centers<F, T>(&self, f: F) -> Array2<T>
    where
        F: Fn(f64, f64) -> T + Sync + Send,
        T: Send,
    {
        let x_coords = self.x_coords();
        let y_coords = self.y_coords();

        let rows: Vec<Vec<T>> = y_coords
            .par_iter()
            .map(|&y| x_coords.iter().map(|&x| f(x, y)).collect())
            .collect();

        let data: Vec<T> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((self.ny, self.nx), data).unwrap()
    }

    /* ============ Iterators ============================================== */
    /// Iterate over pixel indices
    #[inline]
    pub fn iter_pixels(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.ny).flat_map(move |j| (0..self.nx).map(move |i| (i, j)))
    }

    /// Iterate over pixel centres
    #[inline]
    pub fn iter_centers(&self) -> impl Iterator<Item = Point2D> + '_ {
        self.iter_pixels().map(|(i, j)| self.pixel_center(i, j))
    }

    /// Iterate with coordinates
    #[inline]
    pub fn iter_coords(&self) -> impl Iterator<Item = (usize, usize, f64, f64)> + '_ {
        let (min_x, dx) = (self.bbox.min.x, self.dx());
        let (min_y, dy) = (self.bbox.min.y, self.dy());
        (0..self.ny).flat_map(move |j| {
            let y = min_y + (j as f64 + 0.5) * dy;
            (0..self.nx).map(move |i| (i, j, min_x + (i as f64 + 0.5) * dx, y))
        })
    }

    /// Iterate over neighbour indices (E, W, S, N) from given position
    ///
    /// # Arguments
    /// * `i` - index at x axis
    /// * `j` - index at y axis
    #[inline]
    pub fn neighbors_4(&self, i: usize, j: usize) -> impl Iterator<Item = (usize, usize)> {
        [
            if i > 0 { Some((i - 1, j)) } else { None },
            if i < self.nx - 1 {
                Some((i + 1, j))
            } else {
                None
            },
            if j > 0 { Some((i, j - 1)) } else { None },
            if j < self.ny - 1 {
                Some((i, j + 1))
            } else {
                None
            },
        ]
        .into_iter()
        .flatten()
    }

    /// Iterate over neighbour indices (SW, S, SE, W, E, NW, N, NE) from given position
    ///
    /// # Arguments
    /// * `i` - index at x axis
    /// * `j` - index at y axis
    #[inline]
    pub fn neighbors_8(&self, i: usize, j: usize) -> impl Iterator<Item = (usize, usize)> {
        [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
        .into_iter()
        .filter_map(move |(di, dj)| {
            let ni = i as i64 + di;
            let nj = j as i64 + dj;
            if ni >= 0 && ni < self.nx as i64 && nj >= 0 && nj < self.ny as i64 {
                Some((ni as usize, nj as usize))
            } else {
                None
            }
        })
    }

    /* ============ Grid Transformations =================================== */
    /// Refine the grid by a factor
    ///
    /// # Arguments
    /// * `factor` - refinement factor
    #[inline]
    pub fn refine(&self, factor: usize) -> Result<Self> {
        if factor == 0 {
            return Err(LensForgeError::invalid_grid_dimension(
                "Refinement factor must be positive",
            ));
        }
        Ok(Self {
            nx: self.nx * factor,
            ny: self.ny * factor,
            bbox: self.bbox,
        })
    }

    /// Coarsen the grid by a factor
    ///
    /// # Arguments
    /// * `factor` - coarsing factor
    #[inline]
    pub fn coarsen(&self, factor: usize) -> Result<Self> {
        if factor == 0 {
            return Err(LensForgeError::invalid_grid_dimension(
                "Coarsening factor must be positive",
            ));
        }
        if !self.nx.is_multiple_of(factor) || !self.ny.is_multiple_of(factor) {
            return Err(LensForgeError::invalid_grid_dimension(format!(
                "Dimensions ({}, {}) not divisible by {}",
                self.nx, self.ny, factor
            )));
        }
        Ok(Self {
            nx: self.nx / factor,
            ny: self.ny / factor,
            bbox: self.bbox,
        })
    }

    /// Expand the grid to the left, right, bottom, and top
    ///
    /// # Arguments
    /// * `left` - number of pixels to expand on the left
    /// * `right` - number of pixels to expand on the right
    /// * `bottom` - number of pixels to expand on the bottom
    /// * `top` - number of pixels to expand on the top
    #[inline]
    pub fn expand(&self, left: usize, right: usize, bottom: usize, top: usize) -> Self {
        Self {
            nx: self.nx + left + right,
            ny: self.ny + bottom + top,
            bbox: BBox::new(
                Point2D::new(
                    self.bbox.min.x - left as f64 * self.dx(),
                    self.bbox.min.y - bottom as f64 * self.dy(),
                ),
                Point2D::new(
                    self.bbox.max.x + right as f64 * self.dx(),
                    self.bbox.max.y + top as f64 * self.dy(),
                ),
            ),
        }
    }

    /// Shift the grid by (dx, dy)
    ///
    /// # Arguments
    /// * `dx` - shift along the x axis
    /// * `dy` - shift along the y axis
    #[inline]
    pub fn translate(&self, dx: f64, dy: f64) -> Self {
        let shift = Point2D::new(dx, dy);
        Self {
            bbox: BBox::new(self.bbox.min + shift, self.bbox.max + shift),
            ..*self
        }
    }

    /// Check whether another grid has the same geometry
    ///
    /// # Arguments
    /// * `other` - other grid
    #[inline]
    pub fn same_geometry(&self, other: &Self) -> bool {
        self.nx == other.nx
            && self.ny == other.ny
            && (self.bbox.min.x - other.bbox.min.x).abs() < 1e-10
            && (self.bbox.min.y - other.bbox.min.y).abs() < 1e-10
            && (self.dx() - other.dx()).abs() < 1e-10
            && (self.dy() - other.dy()).abs() < 1e-10
    }

}

impl std::fmt::Display for Grid2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Grid2D({}x{}, [{:.3}, {:.3}]x[{:.3}, {:.3}])",
            self.nx,
            self.ny,
            self.bbox.min.x,
            self.bbox.min.x + self.width(),
            self.bbox.min.y,
            self.bbox.min.y + self.height()
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
    fn test_coord_matrix_x_values() {
        let grid = Grid2D::new(4, 3, BBox::from_lims(0.0, 4.0, 0.0, 3.0)).unwrap();
        let mgx = grid.coord_matrix_x();

        assert_eq!(mgx.shape(), &[3, 4]);

        // Check values
        for i in 0..4 {
            let expected = 0.5 + i as f64;
            for j in 0..3 {
                assert!(approx_eq(mgx[[j, i]], expected));
            }
        }
    }

    #[test]
    fn test_coord_matrix_y_values() {
        let grid = Grid2D::new(4, 3, BBox::from_lims(0.0, 4.0, 0.0, 3.0)).unwrap();
        let mgy = grid.coord_matrix_y();

        for j in 0..3 {
            let expected = 0.5 + j as f64;
            for i in 0..4 {
                assert!(approx_eq(mgy[[j, i]], expected));
            }
        }
    }

    #[test]
    fn test_coord_matrix_implementations_equal() {
        let grid = Grid2D::square(10, 2.0).unwrap();

        let x1 = grid.coord_matrix_x();
        let x2 = grid.prealloc_coord_matrix_x();
        let x3 = grid.par_coord_matrix_x();
        let y1 = grid.coord_matrix_y();
        let y2 = grid.prealloc_coord_matrix_y();
        let y3 = grid.par_coord_matrix_y();

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                assert!(approx_eq(x1[[j, i]], x2[[j, i]]));
                assert!(approx_eq(x1[[j, i]], x3[[j, i]]));
                assert!(approx_eq(y1[[j, i]], y2[[j, i]]));
                assert!(approx_eq(y1[[j, i]], y3[[j, i]]));
            }
        }
    }

    #[test]
    fn test_coord_matrix() {
        let grid = Grid2D::square(5, 1.0).unwrap();

        let (x1, y1) = grid.coord_matrix();
        let (x2, y2) = grid.prealloc_coord_matrix();
        let (x3, y3) = grid.par_coord_matrix();

        assert_eq!(x1, x2);
        assert_eq!(x1, x3);
        assert_eq!(y1, y2);
        assert_eq!(y1, y3);
    }

    #[test]
    fn test_coord_matrix_r2() {
        let grid = Grid2D::square(5, 2.0).unwrap();

        let r2_1 = grid.coord_matrix_r2(None);
        let r2_2 = grid.par_coord_matrix_r2(None);

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                assert!(approx_eq(r2_1[[j, i]], r2_2[[j, i]]));
            }
        }
    }

    #[test]
    fn test_coord_matrix_r2_manual() {
        let grid = Grid2D::square(5, 2.0).unwrap();
        let center = Point2D::origin();

        let r2 = grid.coord_matrix_r2(Some(center));
        let (x, y) = grid.coord_matrix();
        let r2_manual = &x * &x + &y * &y;

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                assert!(approx_eq(r2[[j, i]], r2_manual[[j, i]]));
            }
        }
    }

    #[test]
    fn test_map_centers() {
        let grid = Grid2D::square(4, 1.0).unwrap();

        let r2_map = grid.map_centers(|x, y| x * x + y * y);
        let r2_coord_matrix = grid.coord_matrix_r2(Some(Point2D::origin()));

        assert_eq!(r2_map, r2_coord_matrix);
    }

}
