// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Gradients for mapping values to colours
use colorgrad::Gradient;

/// Colormap inspired by the stellar temperature colours in the Hertzsprung-Russell diagram
pub fn hertzsprung() -> impl Gradient {
    colorgrad::GradientBuilder::new()
        .html_colors(&[
            "#101016", "#6A1B36", "#852F4F", "#8F385B", "#D98647", "#EBC79A", "#FFFFFF",
        ])
        .build::<colorgrad::CatmullRomGradient>()
        .unwrap()
}

/// Colormap with two colour features (green and yellow), increasing in lightness
pub fn obscura() -> impl Gradient {
    colorgrad::GradientBuilder::new()
        .html_colors(&[
            "#101016", "#103F39", "#27524C", "#52746F", "#6F795D", "#D98647", "#EBC79A", "#FFFFFF",
        ])
        .build::<colorgrad::CatmullRomGradient>()
        .unwrap()
}

/// Colormap with two colour features (blue and yellow), increasing in lightness
pub fn gaseous() -> impl Gradient {
    colorgrad::GradientBuilder::new()
        .html_colors(&[
            "#101016", "#1B366A", "#2F4F85", "#385B8F", "#79695d", "#D98647", "#EBC79A", "#FFFFFF",
        ])
        .build::<colorgrad::CatmullRomGradient>()
        .unwrap()
}

/// Getter for pre-built and custom colormapping gradients
///
/// # Arguments
/// * `name` - Name of the colormapping gradient
pub fn get_gradient(name: &str) -> Box<dyn Gradient + Send + Sync> {
    match name.to_lowercase().as_str() {
        "viridis" => Box::new(colorgrad::preset::viridis()),
        "magma" => Box::new(colorgrad::preset::magma()),
        "inferno" => Box::new(colorgrad::preset::inferno()),
        "plasma" => Box::new(colorgrad::preset::plasma()),
        "turbo" => Box::new(colorgrad::preset::turbo()),
        "cividis" => Box::new(colorgrad::preset::cividis()),
        "hertzsprung" => Box::new(hertzsprung()),
        "obscura" => Box::new(obscura()),
        "gaseous" => Box::new(gaseous()),
        _ => Box::new(colorgrad::preset::viridis()),
    }
}
