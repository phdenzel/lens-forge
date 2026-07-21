// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Finite-difference differentiation for [`PixelGrid`] (gradient, Laplacian).

use crate::grid::{FieldArray2D, PixelGrid, Point2D, VectorField2D};
use crate::Result;
use ndarray::{Array2, Zip, s};

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

impl PixelGrid {
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
                    (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx()
                } else if i == nx - 1 {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx()
                } else {
                    (self.values[[j, i + 1]] - self.values[[j, i - 1]]) / (2.0 * self.grid.dx())
                }
            }
            DerivativeStencil::Forward => {
                if i < nx - 1 {
                    (self.values[[j, i + 1]] - self.values[[j, i]]) / self.grid.dx()
                } else {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx()
                }
            }
            DerivativeStencil::Backward => {
                if i > 0 {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx()
                } else {
                    (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx()
                }
            }
            DerivativeStencil::Central4 => {
                if i >= 2 && i < nx - 2 {
                    (-self.values[[j, i + 2]] + 8.0 * self.values[[j, i + 1]]
                        - 8.0 * self.values[[j, i - 1]]
                        + self.values[[j, i - 2]])
                        / (12.0 * self.grid.dx())
                } else if i >= 1 && i < nx - 1 {
                    // Fall back to 2nd-order central
                    (self.values[[j, i + 1]] - self.values[[j, i - 1]]) / (2.0 * self.grid.dx())
                } else if i == 0 {
                    (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx()
                } else {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx()
                }
            }
            DerivativeStencil::Central6 => {
                if i >= 3 && i < nx - 3 {
                    (self.values[[j, i + 3]] - 9.0 * self.values[[j, i + 2]]
                        + 45.0 * self.values[[j, i + 1]]
                        - 45.0 * self.values[[j, i - 1]]
                        + 9.0 * self.values[[j, i - 2]]
                        - self.values[[j, i - 3]])
                        / (60.0 * self.grid.dx())
                } else if i >= 2 && i < nx - 2 {
                    // Fall back to 4th-order
                    (-self.values[[j, i + 2]] + 8.0 * self.values[[j, i + 1]]
                        - 8.0 * self.values[[j, i - 1]]
                        + self.values[[j, i - 2]])
                        / (12.0 * self.grid.dx())
                } else if i >= 1 && i < nx - 1 {
                    (self.values[[j, i + 1]] - self.values[[j, i - 1]]) / (2.0 * self.grid.dx())
                } else if i == 0 {
                    (self.values[[j, 1]] - self.values[[j, 0]]) / self.grid.dx()
                } else {
                    (self.values[[j, i]] - self.values[[j, i - 1]]) / self.grid.dx()
                }
            }
        };

        let dvdy = match stencil {
            DerivativeStencil::Central => {
                if j == 0 {
                    (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy()
                } else if j == ny - 1 {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy()
                } else {
                    (self.values[[j + 1, i]] - self.values[[j - 1, i]]) / (2.0 * self.grid.dy())
                }
            }
            DerivativeStencil::Forward => {
                if j < ny - 1 {
                    (self.values[[j + 1, i]] - self.values[[j, i]]) / self.grid.dy()
                } else {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy()
                }
            }
            DerivativeStencil::Backward => {
                if j > 0 {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy()
                } else {
                    (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy()
                }
            }
            DerivativeStencil::Central4 => {
                if j >= 2 && j < ny - 2 {
                    (-self.values[[j + 2, i]] + 8.0 * self.values[[j + 1, i]]
                        - 8.0 * self.values[[j - 1, i]]
                        + self.values[[j - 2, i]])
                        / (12.0 * self.grid.dy())
                } else if j >= 1 && j < ny - 1 {
                    (self.values[[j + 1, i]] - self.values[[j - 1, i]]) / (2.0 * self.grid.dy())
                } else if j == 0 {
                    (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy()
                } else {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy()
                }
            }
            DerivativeStencil::Central6 => {
                if j >= 3 && j < ny - 3 {
                    (self.values[[j + 3, i]] - 9.0 * self.values[[j + 2, i]]
                        + 45.0 * self.values[[j + 1, i]]
                        - 45.0 * self.values[[j - 1, i]]
                        + 9.0 * self.values[[j - 2, i]]
                        - self.values[[j - 3, i]])
                        / (60.0 * self.grid.dy())
                } else if j >= 2 && j < ny - 2 {
                    (-self.values[[j + 2, i]] + 8.0 * self.values[[j + 1, i]]
                        - 8.0 * self.values[[j - 1, i]]
                        + self.values[[j - 2, i]])
                        / (12.0 * self.grid.dy())
                } else if j >= 1 && j < ny - 1 {
                    (self.values[[j + 1, i]] - self.values[[j - 1, i]]) / (2.0 * self.grid.dy())
                } else if j == 0 {
                    (self.values[[1, i]] - self.values[[0, i]]) / self.grid.dy()
                } else {
                    (self.values[[j, i]] - self.values[[j - 1, i]]) / self.grid.dy()
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
        let dx = self.grid.dx();
        let dy = self.grid.dy();

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
        let inv_dx = 1.0 / self.grid.dx();
        let inv_dy = 1.0 / self.grid.dy();

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
    /// * `bc` - boundary condition (default: [`BoundaryCondition::Natural`]; defaults if `None`)
    #[inline]
    pub fn laplacian(
        &self,
        i: usize,
        j: usize,
        stencil: impl Into<Option<DerivativeStencil>>,
        bc: impl Into<Option<BoundaryCondition>>,
    ) -> Result<f64> {
        Ok(self.laplacian_at(
            i,
            j,
            stencil.into().unwrap_or_default(),
            bc.into().unwrap_or_default(),
        ))
    }

    /// Laplacian at a pixel (infallible core): `Natural` dispatches to
    /// [`Self::laplacian_natural`], the ghost conditions to [`Self::laplacian_ghost`].
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    /// * `stencil` - derivative stencil type
    /// * `bc` - boundary condition
    #[inline]
    fn laplacian_at(
        &self,
        i: usize,
        j: usize,
        stencil: DerivativeStencil,
        bc: BoundaryCondition,
    ) -> f64 {
        match bc {
            BoundaryCondition::Natural => self.laplacian_natural(i, j, stencil),
            _ => self.laplacian_ghost(i, j, stencil, bc),
        }
    }

    /// Laplacian via ghost-cell substitution: the full-order stencil at every
    /// pixel, out-of-domain samples resolved by [`ghost_index`].
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    /// * `stencil` - derivative stencil type
    /// * `bc` - ghost boundary condition (must not be [`BoundaryCondition::Natural`])
    fn laplacian_ghost(
        &self,
        i: usize,
        j: usize,
        stencil: DerivativeStencil,
        bc: BoundaryCondition,
    ) -> f64 {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx2 = 1.0 / (self.grid.dx() * self.grid.dx());
        let inv_dy2 = 1.0 / (self.grid.dy() * self.grid.dy());
        let sx = |o: isize| match ghost_index(i as isize + o, nx, bc) {
            Some(k) => self.values[[j, k]],
            None => 0.0,
        };
        let sy = |o: isize| match ghost_index(j as isize + o, ny, bc) {
            Some(k) => self.values[[k, i]],
            None => 0.0,
        };
        let (num_x, den_x) = stencil_second_deriv(stencil, sx);
        let (num_y, den_y) = stencil_second_deriv(stencil, sy);
        num_x / den_x * inv_dx2 + num_y / den_y * inv_dy2
    }

    /// Laplacian with one-sided edge fallbacks (the [`BoundaryCondition::Natural`]
    /// operator): the interior stencil reduced in order near the boundary.
    ///
    /// # Arguments
    /// * `i` - pixel grid index along x axis
    /// * `j` - pixel grid index along y axis
    /// * `stencil` - derivative stencil type
    fn laplacian_natural(&self, i: usize, j: usize, stencil: DerivativeStencil) -> f64 {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx2 = self.grid.dx() * self.grid.dx();
        let dy2 = self.grid.dy() * self.grid.dy();
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
                if j >= 2 {
                    (v_ij - 2.0 * self.values[[j - 1, i]] + self.values[[j - 2, i]]) / dy2
                } else if j > 0 && j + 1 < ny {
                    (self.values[[j + 1, i]] - 2.0 * v_ij + self.values[[j - 1, i]]) / dy2
                } else if j + 2 < ny {
                    (self.values[[j + 2, i]] - 2.0 * self.values[[j + 1, i]] + v_ij) / dy2
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
    ) -> Result<f64> {
        let stencil = stencil.into().unwrap_or_default();
        let dx = self.grid.dx();
        let dy = self.grid.dy();
        // Same 1-D second-derivative weights as the on-grid Laplacian; here the
        // samples come from bilinear interpolation at offset points rather than
        // array indices.
        let (num_x, den_x) = stencil_second_deriv(stencil, |o| {
            self.interpolate(&Point2D::new(p.x + o as f64 * dx, p.y))
        });
        let (num_y, den_y) = stencil_second_deriv(stencil, |o| {
            self.interpolate(&Point2D::new(p.x, p.y + o as f64 * dy))
        });
        Ok(num_x / den_x / (dx * dx) + num_y / den_y / (dy * dy))
    }

    /// Base implementation to compute Laplacian fields (parallel or sequential)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    /// * `bc` - boundary condition (default: [`BoundaryCondition::Natural`]; defaults if `None`)
    fn laplacian_field_impl<const PAR: bool>(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
        bc: impl Into<Option<BoundaryCondition>>,
    ) -> Result<FieldArray2D> {
        let stencil = stencil.into().unwrap_or_default();
        let bc = bc.into().unwrap_or_default();
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let inv_dx2 = 1.0 / (self.grid.dx() * self.grid.dx());
        let inv_dy2 = 1.0 / (self.grid.dy() * self.grid.dy());

        let mut laplace = Array2::zeros((ny, nx));
        if nx == 0 || ny == 0 {
            return Ok(laplace);
        }

        // Ghost-cell boundary conditions are computed per-pixel via the shared
        // order-generic path; the interior needs no ghosts so this stays exact,
        // and the boundary bands simply read substituted neighbours. The
        // `Natural` (one-sided fallback) operator keeps the vectorised paths below.
        if bc != BoundaryCondition::Natural {
            if PAR {
                ndarray::Zip::indexed(&mut laplace)
                    .par_for_each(|(j, i), o| *o = self.laplacian_ghost(i, j, stencil, bc));
            } else {
                ndarray::Zip::indexed(&mut laplace)
                    .for_each(|(j, i), o| *o = self.laplacian_ghost(i, j, stencil, bc));
            }
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
                    // Skip when nx == 3: column nx-2 coincides with column 1 above,
                    // and `+=` would double-apply the same 2nd-central stencil.
                    if nx > 3 {
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
                    // Skip when ny == 3: row ny-2 coincides with row 1 above.
                    if ny > 3 {
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
                    // Skip when nx == 5: column nx-3 coincides with column 2 above.
                    if nx > 5 {
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
                    // i = nx-2; skip when nx == 3 (coincides with column 1 above).
                    if nx > 3 {
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
                    // Skip when ny == 5: row ny-3 coincides with row 2 above.
                    if ny > 5 {
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
                    // j = ny-2; skip when ny == 3 (coincides with row 1 above).
                    if ny > 3 {
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
    /// * `bc` - boundary condition (default: [`BoundaryCondition::Natural`]; defaults if `None`)
    pub fn laplacian_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
        bc: impl Into<Option<BoundaryCondition>>,
    ) -> Result<FieldArray2D> {
        self.laplacian_field_impl::<false>(stencil, bc)
    }

    /// Compute Laplace field over the entire grid (in parallel)
    ///
    /// # Arguments
    /// * `stencil` - derivative stencil type (default: DerivativeStencil::Central; defaults if `None`)
    /// * `bc` - boundary condition (default: [`BoundaryCondition::Natural`]; defaults if `None`)
    pub fn par_laplacian_field(
        &self,
        stencil: impl Into<Option<DerivativeStencil>>,
        bc: impl Into<Option<BoundaryCondition>>,
    ) -> Result<FieldArray2D> {
        self.laplacian_field_impl::<true>(stencil, bc)
    }
}


/* ================ Derivative methods =============================== */
/// Discrete derivative stencil type.
///
/// Selects the finite-difference formula used for the gradient ($\nabla f$) and
/// Laplacian ($\Delta f$) on a uniform grid of spacing $\Delta x$.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DerivativeStencil {
    /// Forward difference: $\nabla f = (f_{i+1} - f_i) / \Delta x$
    Forward,
    /// Backward difference: $\nabla f = (f_i - f_{i-1}) / \Delta x$
    Backward,
    /// Central difference (2nd order):
    /// * $\nabla f = (f_{i+1} - f_{i-1}) / (2 \Delta x)$
    /// * $\Delta f = (f_{i+1} - 2 f_i + f_{i-1}) / \Delta x^2$
    #[default]
    Central,
    /// Central difference (4th order):
    /// * $\nabla f = (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 \Delta x)$
    /// * $\Delta f = (-f_{i+2} + 16 f_{i+1} - 30 f_i + 16 f_{i-1} - f_{i-2}) / (12 \Delta x^2)$
    Central4,
    /// Central difference (6th order):
    /// * $\nabla f = (f_{i+3} - 9 f_{i+2} + 45 f_{i+1} - 45 f_{i-1} + 9 f_{i-2} - f_{i-3}) / (60 \Delta x)$
    /// * $\Delta f = (2 f_{i+3} - 27 f_{i+2} + 270 f_{i+1} - 490 f_i + 270 f_{i-1} - 27 f_{i-2} + 2 f_{i-3}) / (180 \Delta x^2)$
    Central6,
}

/// 1-D second derivative as `(numerator, denominator)` → `num / (den * spacing²)`.
/// `s(o)` samples at offset `o` (ghost-resolved by the caller).
#[inline]
fn stencil_second_deriv(stencil: DerivativeStencil, s: impl Fn(isize) -> f64) -> (f64, f64) {
    match stencil {
        DerivativeStencil::Central => (s(-1) - 2.0 * s(0) + s(1), 1.0),
        DerivativeStencil::Forward => (s(0) - 2.0 * s(1) + s(2), 1.0),
        DerivativeStencil::Backward => (s(0) - 2.0 * s(-1) + s(-2), 1.0),
        DerivativeStencil::Central4 => (
            -s(-2) + 16.0 * s(-1) - 30.0 * s(0) + 16.0 * s(1) - s(2),
            12.0,
        ),
        DerivativeStencil::Central6 => (
            2.0 * s(-3) - 27.0 * s(-2) + 270.0 * s(-1) - 490.0 * s(0) + 270.0 * s(1)
                - 27.0 * s(2)
                + 2.0 * s(3),
            180.0,
        ),
    }
}

/// Apply the 2nd-order 5-point Laplacian $\nabla^2 u$.
/// Requires symmetric ghost boundary conditions.
pub(crate) fn apply_laplacian_5pt_symm(
    u: &[f64],
    nx: usize,
    ny: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    bc: BoundaryCondition,
    out: &mut [f64],
) {
    debug_assert_ne!(
        bc,
        BoundaryCondition::Natural,
        "apply_laplacian_5pt_symm requires a symmetric (ghost) boundary condition"
    );
    let sample = |k: Option<usize>, u: &[f64]| k.map_or(0.0, |k| u[k]);
    for j in 0..ny {
        let row = j * nx;
        for i in 0..nx {
            let c = u[row + i];
            let xl = sample(ghost_index(i as isize - 1, nx, bc).map(|k| row + k), u);
            let xr = sample(ghost_index(i as isize + 1, nx, bc).map(|k| row + k), u);
            let yl = sample(ghost_index(j as isize - 1, ny, bc).map(|k| k * nx + i), u);
            let yr = sample(ghost_index(j as isize + 1, ny, bc).map(|k| k * nx + i), u);
            out[row + i] = (xl + xr - 2.0 * c) * inv_dx2 + (yl + yr - 2.0 * c) * inv_dy2;
        }
    }
}

/* ================ Boundary conditions ============================== */
/// Edge handling for the discrete Laplacian, orthogonal to [`DerivativeStencil`]:
/// what to do when a stencil neighbour falls outside the domain. The three ghost
/// conditions keep the operator symmetric; `Natural` does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryCondition {
    /// No ghost cells; one-sided/reduced-order stencils at the edge. Asymmetric. Default.
    #[default]
    Natural,
    /// Wrap-around. Symmetric; singular (constants in the null space).
    Periodic,
    /// Ghost cells read as $0$. Symmetric, positive-definite.
    Dirichlet,
    /// Ghost cells mirror their neighbour (zero normal derivative). Symmetric; singular.
    Neumann,
}

impl BoundaryCondition {
    /// Whether constants lie in the null space (every condition but [`Dirichlet`](Self::Dirichlet)).
    #[inline]
    pub(crate) fn is_singular(self) -> bool {
        !matches!(self, BoundaryCondition::Dirichlet)
    }
}

/// Resolve a possibly out-of-range 1-D index under `bc`: wrap for `Periodic`,
/// even reflection for `Neumann`, `None` when the ghost is $0$ (`Dirichlet`,
/// `Natural`). Shared ghost semantics for the on-grid and flat-buffer Laplacians.
#[inline]
pub(crate) fn ghost_index(idx: isize, n: usize, bc: BoundaryCondition) -> Option<usize> {
    let ni = n as isize;
    if idx >= 0 && idx < ni {
        return Some(idx as usize);
    }
    match bc {
        BoundaryCondition::Periodic => Some(idx.rem_euclid(ni) as usize),
        BoundaryCondition::Neumann => {
            let reflected = if idx < 0 { -idx - 1 } else { 2 * ni - idx - 1 };
            Some(reflected.clamp(0, ni - 1) as usize)
        }
        BoundaryCondition::Dirichlet | BoundaryCondition::Natural => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;

    fn approx_eq(a: f64, b: f64, eps: impl Into<Option<f64>>) -> bool {
        let eps = eps.into().unwrap_or(1e-12);
        (a - b).abs() < eps
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
    fn test_pixel_grid_laplacian() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        // Quadratic function: Laplacian should be constant
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let lap = pg.laplacian(4, 4, DerivativeStencil::Central, None).unwrap();
        assert!(approx_eq(lap, 4.0, 1e-12));
    }

    #[test]
    fn test_pixel_grid_laplacian_field() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        // Quadratic function: Laplacian should be constant
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);

        let laplace = pg.laplacian_field(None, None).unwrap();
        let par_laplace = pg.par_laplacian_field(None, None).unwrap();

        for j in 0..8 {
            for i in 0..8 {
                assert!(approx_eq(laplace[[j, i]], par_laplace[[j, i]], 1e-12));
            }
        }
    }

    #[test]
    fn laplacian_interp_matches_on_grid() {
        // At a pixel centre, interpolation returns the stored value, so the
        // interpolated Laplacian must equal the on-grid (Natural) Laplacian for
        // every stencil — a lock-in that the `stencil_second_deriv` reuse in
        // `laplacian_interp` preserves the original per-order formulas.
        let grid = Grid2D::square(16, 3.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (0.6 * x).sin() * (0.4 * y).cos() + 0.2 * x * y);
        let (i, j) = (7, 9);
        let p = grid.pixel_center(i, j);
        for stencil in [
            DerivativeStencil::Central,
            DerivativeStencil::Forward,
            DerivativeStencil::Backward,
            DerivativeStencil::Central4,
            DerivativeStencil::Central6,
        ] {
            let on_grid = pg.laplacian(i, j, stencil, None).unwrap();
            let interp = pg.laplacian_interp(&p, stencil).unwrap();
            assert!(
                (on_grid - interp).abs() < 1e-9,
                "stencil {stencil:?}: on-grid {on_grid} vs interp {interp}"
            );
        }
    }

    #[test]
    fn ghost_index_rules() {
        use BoundaryCondition::{Dirichlet, Neumann, Periodic};
        // Periodic wraps around.
        assert_eq!(ghost_index(-1, 4, Periodic), Some(3));
        assert_eq!(ghost_index(4, 4, Periodic), Some(0));
        assert_eq!(ghost_index(2, 4, Periodic), Some(2));
        // Neumann reflects evenly about the boundary.
        assert_eq!(ghost_index(-1, 4, Neumann), Some(0));
        assert_eq!(ghost_index(-2, 4, Neumann), Some(1));
        assert_eq!(ghost_index(4, 4, Neumann), Some(3));
        assert_eq!(ghost_index(5, 4, Neumann), Some(2));
        // Dirichlet reads out-of-domain as zero (None → caller substitutes 0).
        assert_eq!(ghost_index(-1, 4, Dirichlet), None);
        assert_eq!(ghost_index(4, 4, Dirichlet), None);
        assert_eq!(ghost_index(1, 4, Dirichlet), Some(1));
    }

    #[test]
    fn laplacian_field_periodic_matches_analytic() {
        use std::f64::consts::PI;
        // sin(x)cos(y) has period 2π; on the [-π,π) periodic grid its Laplacian is
        // -2 sin(x)cos(y) *everywhere*, edges included — unlike the Natural
        // one-sided fallback which loses order at the boundary.
        let grid = Grid2D::square(64, PI).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x.sin() * y.cos());
        let expected = PixelGrid::from_func(grid, |x, y| -2.0 * x.sin() * y.cos());
        let lap = pg
            .laplacian_field(DerivativeStencil::Central, BoundaryCondition::Periodic)
            .unwrap();
        let err = lap
            .iter()
            .zip(expected.values.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 5e-3, "periodic laplacian err {err:e}");
    }

    #[test]
    fn flat_kernel_matches_field_second_order() {
        // The flat CG kernel and the Array2 field path share ghost_index, so they
        // must agree exactly for every ghost boundary condition (2nd order).
        let grid = Grid2D::square(16, 1.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (1.3 * x).sin() + (0.7 * y).cos() + x * y);
        let nx = grid.nx;
        let ny = grid.ny;
        let inv_dx2 = 1.0 / (grid.dx() * grid.dx());
        let inv_dy2 = 1.0 / (grid.dy() * grid.dy());
        let mut u = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = pg.values[[j, i]];
            }
        }
        for bc in [
            BoundaryCondition::Periodic,
            BoundaryCondition::Dirichlet,
            BoundaryCondition::Neumann,
        ] {
            let field = pg.laplacian_field(DerivativeStencil::Central, bc).unwrap();
            let mut out = vec![0.0; nx * ny];
            apply_laplacian_5pt_symm(&u, nx, ny, inv_dx2, inv_dy2, bc, &mut out);
            for j in 0..ny {
                for i in 0..nx {
                    assert!(
                        (out[j * nx + i] - field[[j, i]]).abs() < 1e-12,
                        "bc {bc:?} mismatch at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn laplacian_field_and_point_agree_for_ghost_bcs() {
        // The field path delegates to laplacian_ghost per pixel; sanity-check that
        // the public point method reports the same values under a ghost BC.
        let grid = Grid2D::square(12, 1.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| (x * x - y).cos());
        for stencil in [DerivativeStencil::Central, DerivativeStencil::Central4] {
            let field = pg
                .laplacian_field(stencil, BoundaryCondition::Neumann)
                .unwrap();
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let point = pg
                        .laplacian(i, j, stencil, BoundaryCondition::Neumann)
                        .unwrap();
                    assert!((field[[j, i]] - point).abs() < 1e-12);
                }
            }
        }
    }

    const ALL_STENCILS: [DerivativeStencil; 5] = [
        DerivativeStencil::Central,
        DerivativeStencil::Forward,
        DerivativeStencil::Backward,
        DerivativeStencil::Central4,
        DerivativeStencil::Central6,
    ];

    /// A smooth, non-polynomial test field.
    fn smooth(grid: Grid2D) -> PixelGrid {
        PixelGrid::from_func(grid, |x, y| (0.5 * x).sin() * (0.3 * y).cos() + 0.1 * x * y)
    }

    fn max_field_diff(a: &FieldArray2D, b: &FieldArray2D) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn gradient_point_exact_for_linear_all_stencils() {
        // Every stencil (interior and one-sided edge fallbacks) is exact for a
        // linear field, so the pixel gradient is (2, 3) everywhere. Sweeping the
        // whole 8×8 grid hits the i==0/1, interior, nx-2/nx-1 branches.
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| 2.0 * x + 3.0 * y);
        for stencil in ALL_STENCILS {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let g = pg.gradient(i, j, stencil).unwrap();
                    assert!(approx_eq(g.x, 2.0, 1e-9), "{stencil:?} gx at ({i},{j})");
                    assert!(approx_eq(g.y, 3.0, 1e-9), "{stencil:?} gy at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn gradient_interp_exact_for_linear_all_stencils() {
        // gradient_interp samples the bilinear surface at offset points; on a
        // linear field every stencil recovers the exact analytic gradient.
        let grid = Grid2D::square(16, 3.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| 2.0 * x - 0.5 * y);
        let p = grid.pixel_center(8, 8);
        for stencil in ALL_STENCILS {
            let g = pg.gradient_interp(&p, stencil).unwrap();
            assert!(approx_eq(g.x, 2.0, 1e-9), "{stencil:?} interp gx");
            assert!(approx_eq(g.y, -0.5, 1e-9), "{stencil:?} interp gy");
        }
    }

    #[test]
    fn gradient_field_matches_point_and_parallel_all_sizes() {
        // The vectorised field path must agree with the per-pixel gradient for
        // every stencil, and the parallel path must match the serial one. Sweeping
        // sizes 2..=9 exercises the nx>=3/5/7 interior guards and their edge
        // fallbacks (including the nx==4 vs nx>4 Central6 branch split).
        for n in 2..=9usize {
            let grid = Grid2D::square(n, 2.0).unwrap();
            let pg = smooth(grid);
            for stencil in ALL_STENCILS {
                let (fx, fy) = pg.gradient_field(stencil).unwrap();
                let (px, py) = pg.par_gradient_field(stencil).unwrap();
                assert!(max_field_diff(&fx, &px) < 1e-12, "n={n} {stencil:?} par x");
                assert!(max_field_diff(&fy, &py) < 1e-12, "n={n} {stencil:?} par y");
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let g = pg.gradient(i, j, stencil).unwrap();
                        assert!(
                            (fx[[j, i]] - g.x).abs() < 1e-9,
                            "n={n} {stencil:?} field vs point gx at ({i},{j})"
                        );
                        assert!(
                            (fy[[j, i]] - g.y).abs() < 1e-9,
                            "n={n} {stencil:?} field vs point gy at ({i},{j})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gradient_magnitude_matches_field_all_stencils() {
        let grid = Grid2D::square(8, 2.0).unwrap();
        let pg = smooth(grid);
        for stencil in ALL_STENCILS {
            let mag = pg.gradient_magnitude(stencil).unwrap();
            let par_mag = pg.par_gradient_magnitude(stencil).unwrap();
            let (gx, gy) = pg.gradient_field(stencil).unwrap();
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let expected = (gx[[j, i]].powi(2) + gy[[j, i]].powi(2)).sqrt();
                    assert!(approx_eq(mag[[j, i]], expected, 1e-12));
                    assert!(approx_eq(mag[[j, i]], par_mag[[j, i]], 1e-12));
                }
            }
        }
    }

    #[test]
    fn laplacian_field_par_matches_serial_all_sizes() {
        // The parallel and serial field paths differ only in the const-generic
        // PAR flag, so they must produce identical results for every stencil,
        // boundary condition and grid size. Sweeping sizes 2..=9 executes all the
        // interior guards and edge fallbacks (Natural and every ghost BC).
        for n in 2..=9usize {
            let grid = Grid2D::square(n, 2.0).unwrap();
            let pg = smooth(grid);
            for stencil in ALL_STENCILS {
                for bc in [
                    BoundaryCondition::Natural,
                    BoundaryCondition::Periodic,
                    BoundaryCondition::Dirichlet,
                    BoundaryCondition::Neumann,
                ] {
                    let field = pg.laplacian_field(stencil, bc).unwrap();
                    let par = pg.par_laplacian_field(stencil, bc).unwrap();
                    assert!(
                        max_field_diff(&field, &par) < 1e-12,
                        "n={n} {stencil:?} {bc:?} par"
                    );
                }
            }
        }
    }

    #[test]
    fn laplacian_field_natural_matches_point_all_sizes() {
        // The vectorised Natural field path must reproduce the per-pixel laplacian
        // for every stencil at every size. Small grids (n = 3, 5) previously
        // double-applied the Central4/Central6 boundary bands; this is the
        // regression guard for that fix.
        for n in 2..=12usize {
            let grid = Grid2D::square(n, 2.0).unwrap();
            let pg = smooth(grid);
            for stencil in ALL_STENCILS {
                let field = pg.laplacian_field(stencil, None).unwrap();
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let point = pg.laplacian(i, j, stencil, None).unwrap();
                        assert!(
                            (field[[j, i]] - point).abs() < 1e-9,
                            "n={n} {stencil:?} field vs point at ({i},{j}) = {} vs {}",
                            field[[j, i]],
                            point
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn laplacian_field_ghost_bcs_match_point_all_stencils() {
        // Ghost boundary conditions dispatch per pixel; the field and point paths
        // share laplacian_ghost, so they agree exactly for every stencil and BC.
        let grid = Grid2D::square(10, 1.5).unwrap();
        let pg = smooth(grid);
        for bc in [
            BoundaryCondition::Periodic,
            BoundaryCondition::Dirichlet,
            BoundaryCondition::Neumann,
        ] {
            for stencil in ALL_STENCILS {
                let field = pg.laplacian_field(stencil, bc).unwrap();
                let par = pg.par_laplacian_field(stencil, bc).unwrap();
                assert!(max_field_diff(&field, &par) < 1e-12, "{bc:?} {stencil:?} par");
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let point = pg.laplacian(i, j, stencil, bc).unwrap();
                        assert!(
                            (field[[j, i]] - point).abs() < 1e-12,
                            "{bc:?} {stencil:?} at ({i},{j})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn laplacian_quadratic_interior_is_exact_all_stencils() {
        // ∇²(x²+y²) = 4; all 2nd-difference stencils are exact in the interior.
        let grid = Grid2D::square(12, 3.0).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y);
        for stencil in ALL_STENCILS {
            let lap = pg.laplacian_field(stencil, None).unwrap();
            // Stay well inside so every stencil uses its full interior formula.
            for j in 4..8 {
                for i in 4..8 {
                    assert!(
                        approx_eq(lap[[j, i]], 4.0, 1e-9),
                        "{stencil:?} lap {} at ({i},{j})",
                        lap[[j, i]]
                    );
                }
            }
        }
    }

    #[test]
    fn degenerate_single_pixel_grids() {
        // A 1×1 grid can apply no stencil: the point laplacian falls through to the
        // zero fallbacks for every stencil, and the field operators leave the
        // (nx<2 / ny<2) guarded bands untouched. Exercises the degenerate branches.
        let tiny = Grid2D::square(1, 1.0).unwrap();
        let pg = PixelGrid::constant(tiny, 2.0);
        for stencil in ALL_STENCILS {
            assert_eq!(pg.laplacian(0, 0, stencil, None).unwrap(), 0.0);
            let (gx, gy) = pg.gradient_field(stencil).unwrap();
            assert_eq!(gx[[0, 0]], 0.0);
            assert_eq!(gy[[0, 0]], 0.0);
            assert_eq!(pg.gradient_magnitude(stencil).unwrap()[[0, 0]], 0.0);
            for bc in [BoundaryCondition::Natural, BoundaryCondition::Dirichlet] {
                let lap = pg.laplacian_field(stencil, bc).unwrap();
                assert!(lap[[0, 0]].is_finite());
            }
        }
    }

    #[test]
    fn thin_strip_grids_run_all_stencils() {
        // 1-wide strips exercise the axis-asymmetric guards: one axis has no room
        // for a stencil while the other does. Par and serial must still agree.
        for (nx, ny) in [(1usize, 6usize), (6usize, 1usize)] {
            let grid = Grid2D::rectangular(nx, ny, 2.0, 2.0).unwrap();
            let pg = smooth(grid);
            for stencil in ALL_STENCILS {
                let (fx, _) = pg.gradient_field(stencil).unwrap();
                let (px, _) = pg.par_gradient_field(stencil).unwrap();
                assert!(max_field_diff(&fx, &px) < 1e-12);
                for bc in [
                    BoundaryCondition::Natural,
                    BoundaryCondition::Periodic,
                    BoundaryCondition::Neumann,
                ] {
                    let field = pg.laplacian_field(stencil, bc).unwrap();
                    let par = pg.par_laplacian_field(stencil, bc).unwrap();
                    assert!(max_field_diff(&field, &par) < 1e-12);
                }
            }
        }
    }

    #[test]
    fn is_singular_flags_null_space() {
        assert!(BoundaryCondition::Periodic.is_singular());
        assert!(BoundaryCondition::Neumann.is_singular());
        assert!(BoundaryCondition::Natural.is_singular());
        assert!(!BoundaryCondition::Dirichlet.is_singular());
    }
}
