// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Sérsic surface-brightness profile.
//!
//! Resource: Ciotti & Bertin, Analytical properties of the R^{1/m} law, A&A 352
//! (1999), [ADS:1999A&A...352..447C](https://ui.adsabs.harvard.edu/abs/1999A%26A...352..447C)
//!
//! In the source-aligned frame (major axis along x', axis ratio $q$) with elliptical
//! radius $r = \sqrt{x'^2 + (y'/q)^2}$, the brightness is
//! $I(r) = I_e \exp\!\big[-b_n\big((r/r_e)^{1/n} - 1\big)\big]$, where $I_e$ is the
//! brightness at the half-light radius $r_e$ and $b_n$ satisfies $\Gamma(2n) = 2\gamma(2n, b_n)$
//! (Ciotti & Bertin series approximation).

use crate::grid::Point2D;
use crate::source::model::SourceModel;

/// A Sérsic profile of index `n`, half-light radius `r_e`, brightness `amplitude`
/// at `r_e`, axis ratio `q` ∈ (0, 1], and major-axis position angle `phi`, centred
/// at `center`.
pub struct Sersic {
    center: Point2D,
    amplitude: f64,
    r_e: f64,
    n: f64,
    q: f64,
    phi: f64,
}

impl Sersic {
    /// Create a Sérsic source centred at `center`.
    ///
    /// # Arguments
    /// * `center` - source centre.
    /// * `amplitude` - brightness I_e at the half-light radius.
    /// * `r_e` - half-light (effective) radius.
    /// * `n` - Sérsic index.
    /// * `q` - minor/major axis ratio in (0, 1].
    /// * `phi` - major-axis position angle (radians).
    pub fn new(center: Point2D, amplitude: f64, r_e: f64, n: f64, q: f64, phi: f64) -> Self {
        Self {
            center,
            amplitude,
            r_e,
            n,
            q,
            phi,
        }
    }

    /// The Sérsic constant b_n such that r_e encloses half the light (Ciotti &
    /// Bertin 1999 series).
    fn b_n(&self) -> f64 {
        let n = self.n;
        2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)
            + 131.0 / (1148175.0 * n * n * n)
    }
}

impl SourceModel for Sersic {
    fn brightness(&self, beta: Point2D) -> f64 {
        let d = beta - self.center;
        let (c, s) = (self.phi.cos(), self.phi.sin());
        // Rotate into the source-aligned frame (major axis along x').
        let x = c * d.x + s * d.y;
        let y = -s * d.x + c * d.y;
        let r = (x * x + (y / self.q) * (y / self.q)).sqrt();
        self.amplitude * (-self.b_n() * ((r / self.r_e).powf(1.0 / self.n) - 1.0)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn brightness_at_effective_radius_is_amplitude() {
        // I(r_e) = amplitude by definition, along any direction (q = 1).
        let src = Sersic::new(Point2D::origin(), 3.0, 2.0, 4.0, 1.0, 0.0);
        for p in [Point2D::new(2.0, 0.0), Point2D::new(0.0, 2.0)] {
            assert!((src.brightness(p) - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn central_brightness_is_amplitude_times_exp_bn() {
        let src = Sersic::new(Point2D::origin(), 1.0, 2.0, 4.0, 1.0, 0.0);
        assert!((src.brightness(Point2D::origin()) - src.b_n().exp()).abs() < 1e-9);
    }

    #[test]
    fn circular_is_direction_independent() {
        let src = Sersic::new(Point2D::origin(), 1.0, 1.5, 2.5, 1.0, 0.7);
        let a = src.brightness(Point2D::new(1.1, 0.0));
        let b = src.brightness(Point2D::new(0.0, 1.1));
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn ellipticity_brightens_along_major_axis() {
        // q < 1: a point at distance d along the major axis is brighter (smaller
        // elliptical radius) than the same distance along the minor axis.
        let src = Sersic::new(Point2D::origin(), 1.0, 2.0, 2.0, 0.5, 0.0);
        let major = src.brightness(Point2D::new(1.5, 0.0));
        let minor = src.brightness(Point2D::new(0.0, 1.5));
        assert!(major > minor, "major {major} minor {minor}");
    }

    #[test]
    fn phi_rotates_the_major_axis() {
        // Rotating the profile by π/2 swaps the major/minor brightness at a point.
        let horiz = Sersic::new(Point2D::origin(), 1.0, 2.0, 2.0, 0.5, 0.0);
        let vert = Sersic::new(Point2D::origin(), 1.0, 2.0, 2.0, 0.5, FRAC_PI_2);
        let p = Point2D::new(1.5, 0.0);
        assert!((horiz.brightness(p) - vert.brightness(Point2D::new(0.0, 1.5))).abs() < 1e-12);
    }

    #[test]
    fn exponential_disk_decays_as_expected() {
        // n = 1: I(2 r_e)/I(r_e) = exp(-b_1 (2 - 1)) = exp(-b_1).
        let src = Sersic::new(Point2D::origin(), 1.0, 1.0, 1.0, 1.0, 0.0);
        let ratio = src.brightness(Point2D::new(2.0, 0.0)) / src.brightness(Point2D::new(1.0, 0.0));
        assert!((ratio - (-src.b_n()).exp()).abs() < 1e-12);
    }
}
