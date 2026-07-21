// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Polygonal mesh adapters for rendering non-uniform grids.
use crate::grid::MeshGrid;
use crate::viz::colorize::map_to_rgba_slice_or;
use crate::viz::render::{RenderBounds, RenderOptions};
use crate::viz::scale::{Normalize, RenderScale, auto_range};
use crate::{LensForgeError, Result};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_line_segment_mut, draw_polygon_mut};
use imageproc::point::Point;

/// A single cell for cell-based mesh rendering.
#[derive(Clone, Debug)]
pub struct Cell {
    /// Polygon vertex coordinates
    pub vertices: Vec<[f64; 2]>,
    /// Scalar value associated with the cell
    pub value: f64,
}

impl Cell {
    pub fn is_valid(&self) -> bool {
        if self.vertices.len() < 3 {
            return false;
        }
        if self
            .vertices
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return false;
        }
        // check whether cell is non-degenerate
        self.area().abs() > 0.0
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.vertices.len();
        if n < 3 {
            return Err(LensForgeError::dimension_mismatch(
                "3+ vertices".to_string(),
                format!("{} vertices", n),
            ));
        }
        if self
            .vertices
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return Err(LensForgeError::numerical_error(
                "Cell has non-finite vertex coodinates",
            ));
        }
        let a = self.area().abs();
        if !(a > 0.0) {
            return Err(LensForgeError::numerical_error(
                "Cell is degenerate (has zero area)",
            ));
        }
        Ok(())
    }

    /// Signed polygon area (2D) using Gauss's formula.
    ///
    /// For clockwise (cw) < 0 , for counter-clockwise (ccw) > 0, and
    /// zero if colinear or degenerate.
    pub fn area(&self) -> f64 {
        let poly = &self.vertices;
        let n = poly.len();
        if n < 3 {
            return 0.0;
        }
        let mut sum: f64 = 0.0;
        for i in 0..n {
            let i_next = (i + 1) % n;
            let (x0, y0) = (poly[i][0], poly[i][1]);
            let (x1, y1) = (poly[i_next][0], poly[i_next][1]);
            sum += x0 * y1 - x1 * y0;
        }
        0.5 * sum
    }
}

/// A polygonal mesh view.
pub trait ToMesh {
    fn cells(&self) -> Result<Vec<Cell>>;
}

/// A polygonal mesh interface storing cells and (optionally) explicit bounds.
#[derive(Clone, Debug, Default)]
pub struct MeshField {
    /// Cells to render
    pub cells: Vec<Cell>,
    /// Optional coordinate bounds
    pub bounds: Option<RenderBounds>,
}

impl MeshField {
    /// Construct from cells without explicit bounds.
    pub fn from_cells(cells: Vec<Cell>) -> Self {
        Self {
            cells,
            bounds: None,
        }
    }

    /// Builder for setting `bounds` field.
    pub fn with_bounds(mut self, bounds: RenderBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        if let Some(b) = self.bounds {
            if !b.is_valid() {
                return false;
            }
        }
        self.cells.iter().all(|c| c.is_valid())
    }

    pub fn validate(&self) -> Result<()> {
        if let Some(b) = self.bounds {
            if !b.is_valid() {
                return Err(LensForgeError::invalid_option(
                    "Mesh bounds must be finite and non-degenerate",
                ));
            }
        }
        for (i, c) in self.cells.iter().enumerate() {
            c.validate()
                .map_err(|e| LensForgeError::invalid_parameter(format!("Cell #{i}: {e}")))?;
        }
        Ok(())
    }

    /// Compute tight axis-aligned bounds over all cells.
    pub fn compute_bounds(&self) -> Option<RenderBounds> {
        let mut xmin = f64::INFINITY;
        let mut ymin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymax = f64::NEG_INFINITY;

        if self.cells.is_empty() {
            return None;
        }

        for cell in &self.cells {
            for &p in &cell.vertices {
                let (x, y) = (p[0], p[1]);
                if !x.is_finite() || !y.is_finite() {
                    return None;
                }
                xmin = xmin.min(x);
                ymin = ymin.min(y);
                xmax = xmax.max(x);
                ymax = ymax.max(y);
            }
        }

        let bounds = RenderBounds {
            min: [xmin, ymin],
            max: [xmax, ymax],
        };
        if bounds.is_valid() {
            Some(bounds)
        } else {
            None
        }
    }
}

impl ToMesh for MeshField {
    fn cells(&self) -> Result<Vec<Cell>> {
        // Return a shallow clone of all cells (vertices are owned in each cell).
        self.validate()?;
        Ok(self.cells.clone())
    }
}

/// Render a `MeshGrid` as its Voronoi polygons: each cell becomes a filled
/// polygon carrying the cell value. Degenerate cells (fewer than 3 vertices)
/// are skipped.
impl ToMesh for MeshGrid {
    fn cells(&self) -> Result<Vec<Cell>> {
        let mut cells = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            let poly = self.cell_polygon(i);
            if poly.len() < 3 {
                continue;
            }
            cells.push(Cell {
                vertices: poly.iter().map(|p| [p.x, p.y]).collect(),
                value: self.value(i),
            });
        }
        Ok(cells)
    }
}

/// Render a polygonal mesh to an RGBA image by filling each cell with the
/// colour of its value.
///
/// This is the faithful visualization for Voronoi / irregular grids: cell
/// boundaries stay crisp at any resolution (unlike sampling the field onto a
/// pixel raster). The value pipeline — scale → normalize → colormap — matches
/// the raster path, so a `MeshGrid` and its rasterization colour-map identically.
/// Set [`RenderOptions::show_edges`] to stroke the cell outlines.
pub fn texturize<M: ToMesh + ?Sized>(mesh: &M, opt: &RenderOptions) -> Result<RgbaImage> {
    opt.validate()?;
    let cells = mesh.cells()?;
    let width = opt.width_u32();
    let height = opt.height_u32();
    let mut img = RgbaImage::from_pixel(width, height, Rgba(opt.invalid_rgba));
    if cells.is_empty() {
        return Ok(img);
    }

    // Domain bounds from all cell vertices.
    let (mut minx, mut miny) = (f64::INFINITY, f64::INFINITY);
    let (mut maxx, mut maxy) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for c in &cells {
        for v in &c.vertices {
            minx = minx.min(v[0]);
            miny = miny.min(v[1]);
            maxx = maxx.max(v[0]);
            maxy = maxy.max(v[1]);
        }
    }
    if !(maxx > minx && maxy > miny) {
        return Err(LensForgeError::invalid_option("mesh has degenerate bounds"));
    }

    // Value pipeline: scale → normalize → colormap (mirrors the raster path).
    let fallback = match opt.scale {
        RenderScale::Log10 => opt.vmin.unwrap_or(-10.0),
        _ => 0.0,
    };
    let scaled: Vec<f32> = cells.iter().map(|c| opt.scale.apply_or(c.value as f32, fallback)).collect();
    let vmin = opt.vmin.unwrap_or_else(|| auto_range(&scaled).map(|(a, _)| a).unwrap_or(0.0));
    let vmax = opt.vmax.unwrap_or_else(|| auto_range(&scaled).map(|(_, b)| b).unwrap_or(1.0));
    let norm = Normalize::default().with_clamp(opt.clamp).with_lims(vmin, vmax)?;
    let normed: Vec<f32> = scaled.iter().map(|&v| norm.apply(v)).collect();
    let gradient = opt.gradient();
    let colors = map_to_rgba_slice_or(&normed, &*gradient, Some(opt.invalid_rgba));

    // World → pixel transform (row 0 = min.y, matching the mesh raster path).
    let sx = width as f64 / (maxx - minx);
    let sy = height as f64 / (maxy - miny);
    let to_px = |v: [f64; 2]| -> Point<i32> {
        let px = ((v[0] - minx) * sx).clamp(0.0, (width - 1) as f64);
        let py = ((v[1] - miny) * sy).clamp(0.0, (height - 1) as f64);
        Point::new(px as i32, py as i32)
    };

    for (c, &color) in cells.iter().zip(&colors) {
        let mut pts: Vec<Point<i32>> = c.vertices.iter().map(|&v| to_px(v)).collect();
        pts.dedup();
        if pts.len() >= 2 && pts.first() == pts.last() {
            pts.pop(); // draw_polygon_mut closes the loop itself
        }
        if pts.len() < 3 {
            continue;
        }
        draw_polygon_mut(&mut img, &pts, Rgba(color));

        if opt.show_edges {
            let edge = Rgba(opt.edge_rgba);
            for k in 0..pts.len() {
                let a = pts[k];
                let b = pts[(k + 1) % pts.len()];
                draw_line_segment_mut(&mut img, (a.x as f32, a.y as f32), (b.x as f32, b.y as f32), edge);
            }
        }
    }

    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(a: [f64; 2], b: [f64; 2], c: [f64; 2], value: f64) -> Cell {
        Cell {
            vertices: vec![a, b, c],
            value,
        }
    }

    #[test]
    fn area_ccw_positive() {
        let c = tri([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], 0.5);
        assert!(c.area() > 0.0);
    }

    #[test]
    fn area_cw_negative() {
        let c = tri([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], 0.5);
        assert!(c.area() < 0.0);
    }

    #[test]
    fn mesh_validate_ok_and_compute_bounds() {
        let cells = vec![
            tri([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], 0.5),
            tri([1.0, 1.0], [2.0, 1.0], [1.0, 2.0], 1.5),
        ];
        let mf = MeshField::from_cells(cells);
        mf.validate().unwrap();

        let b = mf.compute_bounds().unwrap();
        assert!(b.is_valid());
        assert_eq!(b.min, [0.0, 0.0]);
        assert_eq!(b.max, [2.0, 2.0]);
    }

    #[test]
    fn mesh_validate_fails_on_degenerate() {
        // Colinear triangle (area = 0)
        let cells = vec![tri([0.0, 0.0], [1.0, 1.0], [2.0, 2.0], 0.0)];
        let mf = MeshField::from_cells(cells);
        assert!(mf.validate().is_err());
        assert!(!mf.is_valid());
    }

    #[test]
    fn mesh_validate_fails_on_non_finite() {
        let cells = vec![Cell {
            vertices: vec![[f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]],
            value: 0.0,
        }];
        let mf = MeshField::from_cells(cells);
        assert!(mf.validate().is_err());
        assert!(mf.compute_bounds().is_none());
    }

    #[test]
    fn to_mesh_returns_cells_after_validation() {
        let cells = vec![tri([0.0, 0.0], [2.0, 0.0], [0.0, 2.0], 7.0)];
        let mf = MeshField::from_cells(cells.clone());
        let out = mf.cells().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].vertices.len(), 3);
        assert_eq!(out[0].value, 7.0);
    }

    #[test]
    fn texturize_fills_cells_with_distinct_colors() {
        use crate::grid::Point2D;
        use std::num::NonZeroU32;

        let sites = [
            Point2D::new(0.25, 0.25),
            Point2D::new(0.75, 0.25),
            Point2D::new(0.75, 0.75),
            Point2D::new(0.25, 0.75),
        ];
        let mesh = MeshGrid::new(
            &sites,
            crate::grid::BBox::from_lims(0.0, 1.0, 0.0, 1.0),
            ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();

        let mut opt = RenderOptions::default();
        opt.width = NonZeroU32::new(64).unwrap();
        opt.height = NonZeroU32::new(64).unwrap();
        opt.vmin = Some(1.0);
        opt.vmax = Some(4.0);

        let img = texturize(&mesh, &opt).unwrap();
        assert_eq!(img.dimensions(), (64, 64));

        // Sample the four quadrant centres: all filled (opaque), all distinct.
        let centers = [(16, 16), (48, 16), (16, 48), (48, 48)];
        let colors: Vec<[u8; 4]> = centers.iter().map(|&(x, y)| img.get_pixel(x, y).0).collect();
        for c in &colors {
            assert_eq!(c[3], 255, "quadrant pixel not opaque: {c:?}");
        }
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j], "quadrants {i},{j} share a colour");
            }
        }
    }

    #[test]
    fn meshgrid_to_mesh_yields_polygon_cells() {
        use crate::grid::Point2D;
        let sites = [
            Point2D::new(0.25, 0.25),
            Point2D::new(0.75, 0.25),
            Point2D::new(0.75, 0.75),
            Point2D::new(0.25, 0.75),
        ];
        let mesh = MeshGrid::new(
            &sites,
            crate::grid::BBox::from_lims(0.0, 1.0, 0.0, 1.0),
            ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();

        let cells = mesh.cells().unwrap();
        assert_eq!(cells.len(), 4);
        // Each quadrant cell is a valid, unit-quarter-area quad.
        for c in &cells {
            assert!(c.vertices.len() >= 3);
            assert!((c.area().abs() - 0.25).abs() < 1e-9, "cell area {}", c.area());
        }
        let mf = MeshField::from_cells(cells);
        mf.validate().unwrap();
    }
}
