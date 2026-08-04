// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Polygonal rendering of a `MeshGrid`: fill each Voronoi cell with the colour
//! of its value.
use crate::grid::{MeshGrid, Point2D, ScalarField2D};
use crate::viz::colorize::map_to_rgba_slice_or;
use crate::viz::render::RenderOptions;
use crate::viz::scale::{Normalize, RenderScale, auto_range};
use crate::{LensForgeError, Result};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_line_segment_mut, draw_polygon_mut};
use imageproc::point::Point;

impl MeshGrid {
    /// Render this mesh to an RGBA image by filling each Voronoi cell with the
    /// colour of its value.
    ///
    /// This is the faithful visualization for Voronoi / irregular grids: cell
    /// boundaries stay crisp at any resolution (unlike sampling the field onto a
    /// pixel raster). Set [`RenderOptions::show_edges`] to stroke the cell
    /// outlines.
    ///
    /// # Arguments
    /// * `opt` - shared rendering configuration.
    pub fn texturize(&self, opt: &RenderOptions) -> Result<RgbaImage> {
        opt.validate()?;
        let width = opt.width_u32();
        let height = opt.height_u32();
        let mut img = RgbaImage::from_pixel(width, height, Rgba(opt.invalid_rgba));
        if self.is_empty() {
            return Ok(img);
        }

        let bbox = self.bounds();
        if !bbox.is_positive() {
            return Err(LensForgeError::invalid_option("mesh has degenerate bounds"));
        }

        // Value pipeline: scale → normalize → colormap (mirrors the raster path).
        let fallback = match opt.scale {
            RenderScale::Log10 => opt.vmin.unwrap_or(-10.0),
            _ => 0.0,
        };
        let scaled: Vec<f32> = self
            .values()
            .iter()
            .map(|&v| opt.scale.apply_or(v as f32, fallback))
            .collect();
        let vmin = opt
            .vmin
            .unwrap_or_else(|| auto_range(&scaled).map(|(a, _)| a).unwrap_or(0.0));
        let vmax = opt
            .vmax
            .unwrap_or_else(|| auto_range(&scaled).map(|(_, b)| b).unwrap_or(1.0));
        let norm = Normalize::default()
            .with_clamp(opt.clamp)
            .with_lims(vmin, vmax)?;
        let normed: Vec<f32> = scaled.iter().map(|&v| norm.apply(v)).collect();
        let gradient = opt.gradient();
        let colors = map_to_rgba_slice_or(&normed, &*gradient, Some(opt.invalid_rgba));

        // World → pixel transform (row 0 = min.y)
        let min = bbox.min;
        let sx = width as f64 / bbox.width();
        let sy = height as f64 / bbox.height();
        let to_px = |v: Point2D| -> Point<i32> {
            let px = ((v.x - min.x) * sx).clamp(0.0, (width - 1) as f64);
            let py = ((v.y - min.y) * sy).clamp(0.0, (height - 1) as f64);
            Point::new(px as i32, py as i32)
        };

        for i in 0..self.len() {
            let mut pts: Vec<Point<i32>> = self.cell(i).polygon.iter().map(|&v| to_px(v)).collect();
            pts.dedup();
            if pts.len() >= 2 && pts.first() == pts.last() {
                pts.pop(); // draw_polygon_mut closes the loop itself
            }
            if pts.len() < 3 {
                continue; // degenerate cell (fewer than 3 distinct pixels)
            }
            draw_polygon_mut(&mut img, &pts, Rgba(colors[i]));

            if opt.show_edges {
                let edge = Rgba(opt.edge_rgba);
                for k in 0..pts.len() {
                    let a = pts[k];
                    let b = pts[(k + 1) % pts.len()];
                    draw_line_segment_mut(
                        &mut img,
                        (a.x as f32, a.y as f32),
                        (b.x as f32, b.y as f32),
                        edge,
                    );
                }
            }
        }

        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{BBox, Point2D};
    use std::num::NonZeroU32;

    fn quad_mesh() -> MeshGrid {
        let sites = [
            Point2D::new(0.25, 0.25),
            Point2D::new(0.75, 0.25),
            Point2D::new(0.75, 0.75),
            Point2D::new(0.25, 0.75),
        ];
        MeshGrid::new(
            &sites,
            BBox::from_lims(0.0, 1.0, 0.0, 1.0),
            ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap()
    }

    #[test]
    fn texturize_fills_cells_with_distinct_colors() {
        let mesh = quad_mesh();
        let mut opt = RenderOptions::default();
        opt.width = NonZeroU32::new(64).unwrap();
        opt.height = NonZeroU32::new(64).unwrap();
        opt.vmin = Some(1.0);
        opt.vmax = Some(4.0);

        let img = mesh.texturize(&opt).unwrap();
        assert_eq!(img.dimensions(), (64, 64));

        // Sample the four quadrant centres: all filled (opaque), all distinct.
        let centers = [(16, 16), (48, 16), (16, 48), (48, 48)];
        let colors: Vec<[u8; 4]> = centers
            .iter()
            .map(|&(x, y)| img.get_pixel(x, y).0)
            .collect();
        for c in &colors {
            assert_eq!(c[3], 255, "quadrant pixel not opaque: {c:?}");
        }
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j], "quadrants {i},{j} share a colour");
            }
        }
    }
}
