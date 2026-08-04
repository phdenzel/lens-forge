// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Gaussian surface-brightness profile.
//!
//! In the source-aligned frame (major axis along x', axis ratio $q$) with elliptical
//! radius $r = \sqrt{x'^2 + (y'/q)^2}$, the brightness is
//! $I(r) = I_0 \exp\!\big[-r^2/(2\sigma^2)\big]$, with peak $I_0$ at the centre. A
//! Sérsic profile with $n = 1/2$, kept as its own type for the common case.

use crate::grid::Point2D;
use crate::source::model::SourceModel;

/// A Gaussian source of peak brightness `amplitude`, scale `sigma`, axis ratio `q`
/// ∈ (0, 1], and major-axis position angle `phi`, centred at `center`.
pub struct Gaussian {
    center: Point2D,
    amplitude: f64,
    sigma: f64,
    q: f64,
    phi: f64,
}

impl Gaussian {
    /// Create a Gaussian source centred at `center`.
    ///
    /// # Arguments
    /// * `center` - source centre.
    /// * `amplitude` - peak brightness I_0 at the centre.
    /// * `sigma` - Gaussian scale along the major axis.
    /// * `q` - minor/major axis ratio in (0, 1].
    /// * `phi` - major-axis position angle (radians).
    pub fn new(center: Point2D, amplitude: f64, sigma: f64, q: f64, phi: f64) -> Self {
        Self {
            center,
            amplitude,
            sigma,
            q,
            phi,
        }
    }
}

impl SourceModel for Gaussian {
    fn brightness(&self, beta: Point2D) -> f64 {
        let d = beta - self.center;
        let (c, s) = (self.phi.cos(), self.phi.sin());
        // Rotate into the source-aligned frame (major axis along x').
        let x = c * d.x + s * d.y;
        let y = -s * d.x + c * d.y;
        let r2 = x * x + (y / self.q) * (y / self.q);
        self.amplitude * (-r2 / (2.0 * self.sigma * self.sigma)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn central_brightness_is_amplitude() {
        let src = Gaussian::new(Point2D::new(0.3, -0.4), 5.0, 1.0, 0.8, 0.6);
        assert!((src.brightness(Point2D::new(0.3, -0.4)) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn brightness_at_one_sigma() {
        // I(σ along major axis) = amplitude·exp(-1/2).
        let src = Gaussian::new(Point2D::origin(), 2.0, 1.5, 1.0, 0.0);
        let expected = 2.0 * (-0.5f64).exp();
        assert!((src.brightness(Point2D::new(1.5, 0.0)) - expected).abs() < 1e-12);
    }

    #[test]
    fn circular_is_direction_independent() {
        let src = Gaussian::new(Point2D::origin(), 1.0, 1.0, 1.0, 0.4);
        let a = src.brightness(Point2D::new(0.9, 0.0));
        let b = src.brightness(Point2D::new(0.0, 0.9));
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn phi_rotates_the_major_axis() {
        let horiz = Gaussian::new(Point2D::origin(), 1.0, 1.0, 0.5, 0.0);
        let vert = Gaussian::new(Point2D::origin(), 1.0, 1.0, 0.5, FRAC_PI_2);
        let p = Point2D::new(0.8, 0.0);
        assert!((horiz.brightness(p) - vert.brightness(Point2D::new(0.0, 0.8))).abs() < 1e-12);
    }
}
