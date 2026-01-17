// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Example showing how to plot a PixelGrid.

use lens_forge::Result;
use lens_forge::grid::PixelGrid;
use lens_forge::viz::render::RenderOptions;
use lens_forge::viz::scale::RenderScale;
use lens_forge::viz::view::view_grid;
use std::num::NonZeroU32;

fn main() -> Result<()> {
    let size: usize = 2048;
    let r: f64 = 2.0;

    let pg = PixelGrid::from_func_sq(size, r, |x, y| {
        let r2 = x * x + y * y;
        let sigma = 2.0;
        (-0.5 * r2 / (sigma * sigma)).exp()
    });

    let opt = RenderOptions {
        width: NonZeroU32::new((size as u32) / 2).unwrap(),
        height: NonZeroU32::new((size as u32) / 2).unwrap(),
        gradient_name: "obscura".to_string(),
        scale: RenderScale::Linear,
        ..Default::default()
    };

    view_grid(&pg, &opt);

    Ok(())
}
