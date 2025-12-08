// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens_forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Basic 2D grid type

use ndarray::{Array1, Array2, Axis, Zip, parallel::prelude::*};
use crate::grid::Point2D;
use crate::{LensForgeError, Result};


/// A 2D grid for discretizing planes
///
/// All coordinate arrays are computed lazily on demand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid2D {
    /// Number of pixels in x direction
    pub nx: usize,
    /// Number of pixels in y direction
    pub ny: usize,
    /// Grid origin (bottom-left corner)
    x0: f64,
    y0: f64,
    /// Pixel sizes (precomputed)
    pub dx: f64,
    pub dy: f64,
}

impl Grid2D {
    /// Create a new 2D grid
    ///
    /// # Arguments
    /// * `nx` - Number of pixels in the x direction
    /// * `ny` - Number of pixels in the y direction
    /// * `extent_x` - Physical extent in x as (x_min, x_max)
    /// * `extent_y` - Physical extent in y as (y_min, y_max)
    ///
    /// # Errors
    /// Returns an error if dimensions are zero or extents are invalid
    #[inline]
    pub fn new(nx: usize, ny: usize, extent_x: (f64, f64), extent_y: (f64, f64)) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(LensForgeError::InvalidGridDimensions(
                "Grid size must be strictly positive".into(),
            ));
        }
        if extent_x.0 >= extent_x.1 || extent_y.0 >= extent_y.1 {
            return Err(LensForgeError::InvalidGridDimensions(
                "Grid extent minimum must be less than maximum".into(),
            ));
        }

        Ok(Self {
            nx,
            ny,
            x0: extent_x.0,
            y0: extent_y.0,
            dx: (extent_x.1 - extent_x.0) / nx as f64,
            dy: (extent_y.1 - extent_y.0) / ny as f64,
        })
    }

    /// Create a square grid centered at the origin
    ///
    /// # Arguments
    /// * `n` - Number of pixels in each direction
    /// * `radius` - Half the physical extent (grid spans from -radius to +radius)
    pub fn square(n: usize, radius: f64) -> Result<Self> {
	if radius <= 0.0 {
	    return Err(LensForgeError::InvalidGridDimensions(
		"Radius must be positive".into()
	    ));
	}
	Self::new(n, n, (-radius, radius), (-radius, radius))
    }

    /// Create a rectangular grid centered at the origin
    ///
    /// # Arguments
    /// * `nx` - Number of pixels in the x direction
    /// * `ny` - Number of pixels in the y direction
    /// * `rx` - Half the physical extent in x (grid spans from -rx to +rx)
    /// * `ry` - Half the physical extent in y (grid spans from -ry to +ry)
    pub fn rectangular(nx: usize, ny: usize, rx: f64, ry: f64) -> Result<Self> {
	if rx <= 0.0  || ry <= 0.0 {
	    return Err(LensForgeError::InvalidGridDimensions(
		"Extents must be positive".into()
	    ));
	}
	Self::new(nx, ny, (-rx, rx), (-ry, ry))
    }

    /// Create a grid with given physical extent and resolution
    ///
    /// # Arguments
    /// * `extent_x` - Physical extent in x as (x_min, x_max)
    /// * `extent_y` - Physical extent in y as (y_min, y_max)
    /// * `resolution` - Grid resolution in x and y
    pub fn with_resolution(extent_x: (f64, f64), extent_y: (f64, f64), resolution: f64) -> Result<Self> {
	if resolution <= 0.0 {
	    return Err(LensForgeError::InvalidGridDimensions(
		"Resolution must be positive".into()
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
        Self::new(nx.max(1), ny.max(1), extent_x, extent_y)
    }

    /* ============ Grid attributes ======================================== */
    /// Get x extent as tuple
    #[inline]
    pub fn extent_x(&self) -> (f64, f64) {
        (self.x0, self.x0 + self.nx as f64 * self.dx)
    }

    /// Get y extent as tuple
    #[inline]
    pub fn extent_y(&self) -> (f64, f64) {
        (self.y0, self.y0 + self.ny as f64 * self.dy)
    }

    /// Grid width
    #[inline]
    pub fn width(&self) -> f64 {
        self.nx as f64 * self.dx
    }

    /// Grid height
    #[inline]
    pub fn height(&self) -> f64 {
	self.ny as f64 * self.dy
    }

    /// Grid center
    #[inline]
    pub fn center(&self) -> Point2D {
        Point2D::new(
            self.x0 + self.width() * 0.5,
            self.y0 + self.height() * 0.5,
        )
    }

    /// Aspect ratio (width / height)
    #[inline]
    pub fn aspect_ratio(&self) -> f64 {
        self.width() / self. height()
    }

    /// Total number of pixels
    #[inline(always)]
    pub fn n_pixels(&self) -> usize {
        self.nx * self.ny
    }

    /// Pixel area
    #[inline(always)]
    pub fn pixel_area(&self) -> f64 {
        self.dx * self.dy
    }

    /// Total grid area
    #[inline(always)]
    pub fn total_area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Shape as (ny, nx) for ndarray compatibility
    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    /// Shape as array [ny, nx]
    #[inline(always)]
    pub fn shape_as_array(&self) -> [usize; 2] {
        [self.ny, self. nx]
    }

    /* ============ Coordinates ============================================ */
    /// Get x-coordinate of pixel center
    ///
    /// # Arguments
    /// * `i` - index along x axis
    #[inline(always)]
    pub fn x_at(&self, i: usize) -> f64 {
        self.x0 + (i as f64 + 0.5) * self.dx
    }

    /// Get y-coordinate of pixel center
    ///
    /// # Arguments
    /// * `j` - index along y axis
    #[inline(always)]
    pub fn y_at(&self, j: usize) -> f64 {
        self.y0 + (j as f64 + 0.5) * self.dy
    }

     /// Get pixel center coordinates
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
        self.x0 + i as f64 * self.dx
    }

    /// Get y-coordinate of pixel corner
    ///
    /// # Arguments
    /// * `j` - index along y axis
    #[inline(always)]
    pub fn y_corner_at(&self, j: usize) -> f64 {
        self.y0 + j as f64 * self.dy
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
        let fi = (p.x - self.x0) / self.dx;
        let fj = (p.y - self.y0) / self.dy;
        
        if fi >= 0.0 && fi < self.nx as f64 && fj >= 0.0 && fj < self.ny as f64 {
            Some((fi as usize, fj as usize))
        } else {
            None
        }
    }

    /// Convert point to fractional pixel coordinates (relative to centers)
    ///
    /// # Arguments
    /// * `p` - point to be mapped
    #[inline(always)]
    pub fn relative_to(&self, p: &Point2D) -> (f64, f64) {
        (
            (p.x - self.x0) / self.dx - 0.5,
            (p.y - self.y0) / self.dy - 0.5,
        )
    }

    /// Convert point to fractional pixel coordinates (relative to corners)
    ///
    /// # Arguments
    /// * `p` - point to be mapped
    #[inline(always)]
    pub fn relative_to_corner(&self, p: &Point2D) -> (f64, f64) {
        (
            (p.x - self.x0) / self.dx,
            (p.y - self.y0) / self.dy,
        )
    }

    /// Check if point is inside grid
    ///
    /// # Arguments
    /// * `p` - point to be checked
    #[inline(always)]
    pub fn contains(&self, p: &Point2D) -> bool {
        let (x1, y1) = (self.x0 + self.width(), self.y0 + self.height());
        p.x >= self.x0 && p.x <= x1 && p.y >= self.y0 && p.y <= y1
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
        (idx % self.nx, idx / self. nx)
    }

    /// Create 1D array of x coordinates for pixel centers
    #[inline]
    pub fn x_coords(&self) -> Array1<f64> {
	Array1::from_shape_fn(self.nx, |i| self.x_at(i))
    }

    /// Create 1D array of y coordinates for pixel centers
    #[inline]
    pub fn y_coords(&self) -> Array1<f64> {
	Array1::from_shape_fn(self.ny, |j| self.y_at(j))
    }

    /// Create 1D array of x coordinates for pixel edges
    #[inline]
    pub fn x_edges(&self) -> Array1<f64> {
	Array1::from_shape_fn(self.nx + 1, |i| self.x_corner_at(i))
    }

    /// Create 1D array of y coordinates for pixel edges
    #[inline]
    pub fn y_edges(&self) -> Array1<f64> {
	Array1::from_shape_fn(self.ny + 1, |j| self.y_corner_at(j))
    }

    /* ============ Meshgrids ============================================== */
    /// Create meshgrid of x coordinates (non-broadcasting)
    #[inline]
    pub fn prealloc_meshgrid_x(&self) -> Array2<f64> {
        let x_coords = self.x_coords();
	// pre-allocate array
	let mut result = Array2::zeros((self.ny, self.nx));
        // row-by-row assignment
        for mut row in result.rows_mut() {
            row.assign(&x_coords);
        }
        result
    }

    /// Create meshgrid of x coordinates using broadcasting
    #[inline]
    pub fn meshgrid_x(&self) -> Array2<f64> {
        let x = self.x_coords();
	// reshape to (1, nx) and broadcast to (ny, nx)
	let x_row = x.into_shape_with_order((1, self.nx)).unwrap();
        x_row.broadcast((self.ny, self.nx)).unwrap().to_owned()
    }

    /// Create meshgrid of y coordinates (non-broadcasting)
    #[inline]
    pub fn prealloc_meshgrid_y(&self) -> Array2<f64> {
        let y_coords = self.y_coords();
	// pre-allocate array
	let mut transposed = Array2::zeros((self.ny, self.nx));
        // row-by-row assignment
        for mut row in transposed.rows_mut() {
            row.assign(&y_coords);
        }
        transposed.reversed_axes().as_standard_layout().into_owned()
    }

    /// Create meshgrid of y coordinates using broadcasting
    #[inline]
    pub fn meshgrid_y(&self) -> Array2<f64> {
        let y = self.y_coords();
	// reshape to (ny, 1) and broadcast to (ny, nx)
	let y_col = y.into_shape_with_order((self.ny, 1)).unwrap();
        y_col.broadcast((self.ny, self.nx)).unwrap().to_owned()
    }

    /// Create x and y meshgrids (non-broadcasting)
    pub fn prealloc_meshgrid(&self) -> (Array2<f64>, Array2<f64>) {
        (self.prealloc_meshgrid_x(), self.prealloc_meshgrid_y())
    }

    /// Create x and y meshgrids using broadcasting
    pub fn meshgrid(&self) -> (Array2<f64>, Array2<f64>) {
        (self.meshgrid_x(), self.meshgrid_y())
    }

    /// Compute squared radial distance meshgrid
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid center)
    pub fn meshgrid_r2(&self, center: Option<Point2D>) -> Array2<f64> {
        let c = center.unwrap_or_else(|| self.center());
	let cx = c.x;
        let cy = c.y;
        // Precompute (x - cx)^2 for each column
        let dx2: Array1<f64> = Array1::from_shape_fn(self.nx, |i| {
            let d = self.x_at(i) - cx;
            d * d
        });
        // Precompute (y - cy)^2 for each row
        let dy2: Array1<f64> = Array1::from_shape_fn(self.ny, |j| {
            let d = self.y_at(j) - cy;
            d * d
        });

        let mut result = Array2::zeros((self.ny, self.nx));
        for (j, mut row) in result.rows_mut().into_iter().enumerate() {
            let dy2_j = dy2[j];
            Zip::from(&mut row)
                .and(&dx2)
                .for_each(|r, &dx2_i| {
                    *r = dx2_i + dy2_j;
                });
        }
        result
    }

    /// Compute radial distance meshgrid
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid center)
    pub fn meshgrid_r(&self, center: Option<Point2D>) -> Array2<f64> {
        self.meshgrid_r2(center).mapv(f64::sqrt)
    }

    /// Compute angular meshgrid with theta = atan2(y - cy, x - cx)
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid center)
    pub fn meshgrid_theta(&self, center: Option<Point2D>) -> Array2<f64> {
        let c = center.unwrap_or_else(|| self.center());
        let cx = c.x;
        let cy = c.y;
        
        Array2::from_shape_fn((self.ny, self.nx), |(j, i)| {
            let x = self.x_at(i) - cx;
            let y = self.y_at(j) - cy;
            y.atan2(x)
        })
    }

    /// Compute polar coordinate meshgrids (r, theta)
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid center)
    pub fn meshgrid_polar(&self, center: Option<Point2D>) -> (Array2<f64>, Array2<f64>) {
        let c = center.unwrap_or_else(|| self.center());

        // Precompute deltas
        let dx_vec: Array1<f64> = Array1::from_shape_fn(self.nx, |i| self.x_at(i) - c.x);
        let dy_vec: Array1<f64> = Array1::from_shape_fn(self.ny, |j| self.y_at(j) - c.y);
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
                    *r = (dx2_i + dy2_j). sqrt();
                    *theta = dy_j.atan2(dx_i);
                });
        }

        (r_grid, theta_grid)
    }

    /// Create meshgrid of x coordinates in parallel (using rayon backend)
    pub fn pmeshgrid_x(&self) -> Array2<f64> {
        let x_coords = self.x_coords();

        let mut result = Array2::zeros((self.ny, self.nx));
	result.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                row.assign(&x_coords);
            });
        result
    }

    /// Create meshgrid of y coordinates in parallel (using rayon backend)
    pub fn pmeshgrid_y(&self) -> Array2<f64> {
        let y_coords = self.y_coords();

        let mut result = Array2::zeros((self.ny, self.nx));
        result.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut row)| {
                row.fill(y_coords[j]);
            });
        result
    }

    /// Create meshgrid of x and y coordinates in parallel (using rayon backend)
    pub fn pmeshgrid(&self) -> (Array2<f64>, Array2<f64>) {
        // Run both in parallel using rayon's join
        rayon::join(
            || self.pmeshgrid_x(),
            || self.pmeshgrid_y(),
        )
    }

    /// Compute squared radial distance meshgrid in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid center)
    pub fn pmeshgrid_r2(&self, center: Option<Point2D>) -> Array2<f64> {
        let c = center.unwrap_or_else(|| self.center());
        let dx2: Array1<f64> = Array1::from_shape_fn(self.nx, |i| {
            let d = self.x_at(i) - c.x;
            d * d
        });

	let mut result = Array2::zeros((self.ny, self.nx));
        result.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut row)| {
		let dy = self.y_at(j) - c.y;
                let dy2 = dy * dy;
                Zip::from(&mut row)
                    .and(&dx2)
                    .for_each(|r, &dx2_i| {
                        *r = dx2_i + dy2;
                    });
            });
        result
    }

    /// Compute radial distance meshgrid in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `center` - Optional reference point (default: grid center)
    pub fn pmeshgrid_r(&self, center: Option<Point2D>) -> Array2<f64> {
	self.pmeshgrid_r2(center).mapv(f64::sqrt)
    }

    /// Apply a function to each pixel center coordinate
    ///
    /// # Arguments
    /// * `f` - function to be applied to each grid coordinate
    #[inline]
    pub fn map_centers<F, T>(&self, f: F) -> Array2<T>
    where
        F: Fn(f64, f64) -> T,
    {   
        Array2::from_shape_fn((self.ny, self. nx), |(j, i)| {
            let x = self.x_at(i);
            let y = self.y_at(j);
            f(x, y)
        })
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

    /// Apply a function to each pixel center coordinate in parallel (using rayon backend)
    ///
    /// # Arguments
    /// * `f` - function to be applied to each grid coordinate
    pub fn pmap_centers<F, T>(&self, f: F) -> Array2<T>
    where
        F: Fn(f64, f64) -> T + Sync + Send,
        T: Send,
    {
	let x_coords = self.x_coords();
	let y_coords = self.y_coords();

	let rows: Vec<Vec<T>> = y_coords
	    .par_iter()
	    .map(|&y| {
		x_coords.iter().map(|&x| f(x, y)).collect()
	    })
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

    /// Iterate over pixel centers
    #[inline]
    pub fn iter_centers(&self) -> impl Iterator<Item = Point2D> + '_ {
        self.iter_pixels().map(|(i, j)| self.pixel_center(i, j))
    }

    /// Iterate with coordinates
    #[inline]
    pub fn iter_coords(&self) -> impl Iterator<Item = (usize, usize, f64, f64)> + '_ {
        (0.. self.ny).flat_map(move |j| {
            let y = self.y_at(j);
            (0..self. nx).map(move |i| (i, j, self.x_at(i), y))
        })
    }

    /// Iterate over neighbor indices (E, W, S, N) from given position
    ///
    /// # Arguments
    /// * `i` - index at x axis
    /// * `j` - index at y axis
    #[inline]
    pub fn neighbors_4(&self, i: usize, j: usize) -> impl Iterator<Item = (usize, usize)> {
        [
            if i > 0 { Some((i - 1, j)) } else { None },
            if i < self.nx - 1 { Some((i + 1, j)) } else { None },
            if j > 0 { Some((i, j - 1)) } else { None },
            if j < self.ny - 1 { Some((i, j + 1)) } else { None },
        ]
            .into_iter()
            .flatten()
    }

    /// Iterate over neighbor indices (SW, S, SE, W, E, NW, N, NE) from given position
    ///
    /// # Arguments
    /// * `i` - index at x axis
    /// * `j` - index at y axis
    #[inline]
    pub fn neighbors_8(&self, i: usize, j: usize) -> impl Iterator<Item = (usize, usize)> {
        [
	    (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)
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
            return Err(LensForgeError::InvalidGridDimensions(
                "Refinement factor must be positive".into()
            ));
        }
        Ok(Self {
            nx: self.nx * factor,
            ny: self.ny * factor,
            x0: self.x0,
            y0: self.y0,
            dx: self.dx / factor as f64,
            dy: self.dy / factor as f64,
        })
    }

    /// Coarsen the grid by a factor
    ///
    /// # Arguments
    /// * `factor` - coarsing factor
    #[inline]
    pub fn coarsen(&self, factor: usize) -> Result<Self> {
        if factor == 0 {
            return Err(LensForgeError::InvalidGridDimensions(
                "Coarsening factor must be positive". into()
            ));
        }
        if self.nx % factor != 0 || self.ny % factor != 0 {
            return Err(LensForgeError::InvalidGridDimensions(
                format!("Dimensions ({}, {}) not divisible by {}", self.nx, self.ny, factor)
            ));
        }
        Ok(Self {
            nx: self.nx / factor,
            ny: self.ny / factor,
            x0: self.x0,
            y0: self.y0,
            dx: self.dx * factor as f64,
            dy: self.dy * factor as f64,
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
            x0: self.x0 - left as f64 * self.dx,
            y0: self.y0 - bottom as f64 * self.dy,
            dx: self.dx,
            dy: self.dy,
        }
    }

    /// Shift the grid by (dx, dy)
    ///
    /// # Arguments
    /// * `dx` - shift along the x axis
    /// * `dy` - shift along the y axis
    #[inline]
    pub fn translate(&self, dx: f64, dy: f64) -> Self {
        Self {
            x0: self.x0 + dx,
            y0: self.y0 + dy,
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
            && (self.x0 - other.x0).abs() < 1e-10
            && (self.y0 - other.y0).abs() < 1e-10
            && (self.dx - other.dx).abs() < 1e-10
            && (self.dy - other.dy).abs() < 1e-10
    }

    /// Get bounding box corners at the lower left and upper right
    #[inline]
    pub fn bounding_box(&self) -> (Point2D, Point2D) {
        (
            Point2D::new(self.x0, self.y0),
            Point2D::new(self.x0 + self.width(), self.y0 + self.height()),
        )
    }
}

impl std::fmt::Display for Grid2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Grid2D({}x{}, [{:.3}, {:.3}]x[{:.3}, {:.3}])",
            self.nx, self.ny,
            self.x0, self.x0 + self.width(),
            self.y0, self.y0 + self.height()
        )
    }
}

/* ================ Interpolators ========================================== */
/// Bilinear interpolation kernel
#[derive(Debug, Clone, Copy)]
pub struct BilinearKernel {
    pub i0: usize,
    pub j0: usize,
    pub i1: usize,
    pub j1: usize,
    pub w00: f64,
    pub w10: f64,
    pub w01: f64,
    pub w11: f64,
}

impl BilinearKernel {
    /// Apply kernel to a 2D array for bilinear interpolation
    ///
    /// # Arguments
    /// * `data` - 2D data array to be interpolated
    #[inline]
    pub fn apply(&self, data: &Array2<f64>) -> f64 {
        self.w00 * data[[self.j0, self.i0]]
            + self.w10 * data[[self.j0, self.i1]]
            + self.w01 * data[[self.j1, self.i0]]
            + self.w11 * data[[self.j1, self.i1]]
    }
}

impl std::fmt::Display for BilinearKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BilinearKernel(({}, {})-({}, {}), k:[{:.3}, {:.3}, {:.3}, {:.3}])",
            self.i0, self.j0, self.i1, self.j1,
            self.w00, self.w01, self.w10, self.w11,
        )
    }
}

impl Grid2D {
    /// Compute bilinear interpolation for a point
    #[inline]
    pub fn bilinear_interp(&self, p: &Point2D) -> Option<BilinearKernel> {
        let (fx, fy) = self.relative_to(p);
        
        let i0 = fx.floor() as i64;
        let j0 = fy.floor() as i64;
        
        if i0 < -1 || i0 >= self.nx as i64 || j0 < -1 || j0 >= self.ny as i64 {
            return None;
        }
        
        let tx = fx - i0 as f64;
        let ty = fy - j0 as f64;
        
        let i0 = i0.max(0) as usize;
        let j0 = j0.max(0) as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        
        Some(BilinearKernel {
            i0, j0, i1, j1,
            w00: (1.0 - tx) * (1.0 - ty),
            w10: tx * (1.0 - ty),
            w01: (1.0 - tx) * ty,
            w11: tx * ty,
        })
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
    fn test_meshgrid_x_values() {
        let grid = Grid2D::new(4, 3, (0.0, 4.0), (0.0, 3.0)).unwrap();
        let mgx = grid.meshgrid_x();
        
        assert_eq!(mgx. shape(), &[3, 4]);
        
        // Check values
        for i in 0..4 {
            let expected = 0.5 + i as f64;
            for j in 0..3 {
                assert!(approx_eq(mgx[[j, i]], expected));
            }
        }
    }

    #[test]
    fn test_meshgrid_y_values() {
        let grid = Grid2D::new(4, 3, (0.0, 4.0), (0.0, 3.0)).unwrap();
        let mgy = grid.meshgrid_y();
        
        for j in 0..3 {
            let expected = 0.5 + j as f64;
            for i in 0.. 4 {
                assert!(approx_eq(mgy[[j, i]], expected));
            }
        }
    }

    #[test]
    fn test_meshgrid_implementations_equal() {
        let grid = Grid2D::square(10, 2.0).unwrap();
        
        let x1 = grid.meshgrid_x();
        let x2 = grid.prealloc_meshgrid_x();
        let x3 = grid.pmeshgrid_x();
	let y1 = grid.meshgrid_y();
	let y2 = grid.prealloc_meshgrid_y();
        let y3 = grid.pmeshgrid_y();

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
    fn test_meshgrid() {
        let grid = Grid2D::square(5, 1.0).unwrap();
        
        let (x1, y1) = grid.meshgrid();
	let (x2, y2) = grid.prealloc_meshgrid();
        let (x3, y3) = grid.pmeshgrid();
        
        assert_eq!(x1, x2);
	assert_eq!(x1, x3);
        assert_eq!(y1, y2);
        assert_eq!(y1, y3);
    }

    #[test]
    fn test_meshgrid_r2() {
        let grid = Grid2D::square(5, 2.0).unwrap();
        
        let r2_1 = grid.meshgrid_r2(None);
	let r2_2 = grid.pmeshgrid_r2(None);
        
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                assert!(approx_eq(r2_1[[j, i]], r2_2[[j, i]]));
            }
        }
    }

    #[test]
    fn test_meshgrid_r2_manual() {
        let grid = Grid2D::square(5, 2.0).unwrap();
        let center = Point2D::origin();
        
        let r2 = grid.meshgrid_r2(Some(center));
        let (x, y) = grid.meshgrid();
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
        let r2_meshgrid = grid.meshgrid_r2(Some(Point2D::origin()));
        
        assert_eq!(r2_map, r2_meshgrid);
    }

    #[test]
    fn test_bilinear_interp_at_center() {
        let grid = Grid2D::new(4, 4, (0.0, 4.0), (0.0, 4.0)).unwrap();
        
        // Point at pixel center
        let w = grid.bilinear_interp(&Point2D::new(0.5, 0.5)).unwrap();
        assert!(approx_eq(w.w00, 1.0));
        assert!(approx_eq(w.w10 + w.w01 + w.w11, 0.0));
	println!("{}", w);
    }

    #[test]
    fn test_bilinear_interp_between() {
        let grid = Grid2D::new(4, 4, (-2.0, 2.0), (-2.0, 2.0)).unwrap();
        
        // Point between four pixels
        let w = grid.bilinear_interp(&Point2D::new(1.0, 1.0)).unwrap();
        assert!(approx_eq(w.w00, 0.25));
        assert!(approx_eq(w.w10, 0.25));
        assert!(approx_eq(w.w01, 0.25));
        assert!(approx_eq(w.w11, 0.25));
	println!("{}", w);
    }

    #[test]
    fn test_bilinear_interpolate() {
        let grid = Grid2D::new(4, 4, (0.0, 4.0), (0.0, 4.0)).unwrap();
        let data = grid.map_centers(|x, _| x); // f(x,y) = x
        
        let p = Point2D::new(2.0, 2.0);
        let w = grid.bilinear_interp(&p).unwrap();
	println!("{}", w);
        let interp = w.apply(&data);
        
        // Should be close to 2.0
	println!("{}", interp);
        assert!((interp - 2.0).abs() < 0.5);
    }
}
