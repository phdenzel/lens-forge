// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Example plotting various gradients/colormaps
//! Run with cargo run --example gradient_viz -- <gradient-name>
//! Available color gradient names are:
//! * viridis
//! * magma
//! * inferno
//! * plasma
//! * turbo
//! * cividis
//! * hertzsprung
//! * obscura
//! * gaseous

use lens_forge::Result;
use lens_forge::viz::get_gradient;
use lens_forge::viz::view::view_image;
use std::env;

fn plot_colorbar(name: &str) -> Result<()> {
    let width = 1300.00;
    let height = 70.00;
    let gradient = get_gradient(name);
    let imgbuf = image::ImageBuffer::from_fn(width as u32, height as u32, |x, _| {
        image::Rgba(gradient.at(x as f32 / width).to_rgba8())
    });
    view_image(imgbuf);
    Ok(())
}

fn main() -> Result<()> {
    let gradient_name = env::args().nth(1).unwrap_or_else(|| "magma".to_string());

    println!("Displaying gradient: {}", gradient_name);
    plot_colorbar(&gradient_name)?;
    Ok(())
}
