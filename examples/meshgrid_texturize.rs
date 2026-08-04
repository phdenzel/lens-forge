// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Example plotting a MeshGrid via the polygon-fill (texturize) path.
//! Run with cargo run --example meshgrid_viz_texturize

use lens_forge::Lcg;
use lens_forge::grid::{BBox, MeshGrid, Point2D};
use lens_forge::viz::render::RenderOptions;
use lens_forge::viz::scale::RenderScale;
use lens_forge::viz::view::view_image;
use std::num::NonZeroU32;

fn main() {
    let n_sites: usize = 4_096; // average resolution of 64x64
    let render_size: usize = 2_048;
    let r: f64 = 2.0;

    // Scatter sites uniformly over [-r, r]; the Voronoi cells tessellate them.
    let mut rng = Lcg::new(0x5EED);
    let sites: Vec<Point2D> = (0..n_sites)
        .map(|_| Point2D::new(rng.range(-r, r), rng.range(-r, r)))
        .collect();
    let bbox = BBox::from_lims(-r, r, -r, r);

    let mg = MeshGrid::from_func(&sites, bbox, |x, y| {
        let r2 = x * x + y * y;
        let sigma = 2.0;
        (-0.5 * r2 / (sigma * sigma)).exp()
    })
    .expect("valid mesh geometry");

    // texturize fills each Voronoi polygon directly (no raster resampling), so
    // the cell geometry stays crisp at any size; show_edges strokes the outlines.
    let opt = RenderOptions {
        width: NonZeroU32::new(render_size as u32).unwrap(),
        height: NonZeroU32::new(render_size as u32).unwrap(),
        gradient_name: "obscura".to_string(),
        scale: RenderScale::Linear,
        show_edges: true,
        ..Default::default()
    };

    let img = mg.texturize(&opt).expect("valid mesh render");
    view_image(img);
}
