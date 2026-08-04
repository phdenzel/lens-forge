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

/* ============ Unit tests ================================================= */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_gradients_build_and_span_endpoints() {
        // Each custom gradient builds without panicking and its domain is [0, 1].
        for g in [
            Box::new(hertzsprung()) as Box<dyn Gradient>,
            Box::new(obscura()),
            Box::new(gaseous()),
        ] {
            let (lo, hi) = g.domain();
            assert!(approx(lo, 0.0) && approx(hi, 1.0));
            // Endpoints match the first/last html colours (#101016 → #FFFFFF).
            let a = g.at(0.0).to_rgba8();
            let b = g.at(1.0).to_rgba8();
            assert_eq!(a, [0x10, 0x10, 0x16, 0xFF]);
            assert_eq!(b, [0xFF, 0xFF, 0xFF, 0xFF]);
        }
    }

    #[test]
    fn get_gradient_dispatches_all_names() {
        // Every named preset/custom gradient resolves; case-insensitive.
        for name in [
            "viridis",
            "magma",
            "inferno",
            "plasma",
            "turbo",
            "cividis",
            "Hertzsprung",
            "OBSCURA",
            "gaseous",
        ] {
            let _ = get_gradient(name);
        }
        // Unknown names fall back to viridis.
        let fallback = get_gradient("does-not-exist");
        let viridis = get_gradient("viridis");
        assert_eq!(fallback.at(0.5).to_rgba8(), viridis.at(0.5).to_rgba8());
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }
}
