// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The point-image data types shared by the forward and inverse layers.
//!
//! An [`Image`] records one solution of the lens equation: its position, signed
//! magnification, [`Parity`], and Fermat potential. The image finders in
//! [`extrema`](super::forward::extrema) produce it; the inverse point-image constraints consume it. A
//! [`PointImageSystem`] gathers the images of one source, with any measured time delays
//! ([`TimeDelay`](super::forward::extrema::delays::TimeDelay)) and arrival order — the observed-data
//! unit the constraint builder takes.
//!
//! Parity is the arrival-time surface's stationary type encoded as `1 − m`, where
//! `m` is the Morse index (the number of negative eigenvalues of A = I − H): `+1`
//! minimum, `0` saddle, `−1` maximum.

use crate::grid::Point2D;
use crate::lens::LensModel;
use crate::solver::forward::extrema::delays::TimeDelay;

/// Image parity as `1 − Morse index` of the Fermat surface: `+1` minimum, `0`
/// saddle, `−1` maximum (one minus the number of negative eigenvalues of A = I − H).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parity(pub i8);

impl Parity {
    /// Minimum of the arrival-time surface — both eigenvalues of A positive.
    pub const MINIMUM: Parity = Parity(1);
    /// Saddle point — eigenvalues of A of opposite sign.
    pub const SADDLE: Parity = Parity(0);
    /// Maximum — both eigenvalues of A negative.
    pub const MAXIMUM: Parity = Parity(-1);

    /// Classify a stationary point from the two eigenvalues of A = I − H.
    ///
    /// # Arguments
    /// * `lambda_t` - tangential eigenvalue λ_t = 1 − κ − γ.
    /// * `lambda_r` - radial eigenvalue λ_r = 1 − κ + γ.
    pub fn from_eigenvalues(lambda_t: f64, lambda_r: f64) -> Self {
        Self::from_invariants(lambda_t * lambda_r, lambda_t + lambda_r)
    }

    /// Classify a stationary point from the invariants of A = I − H: its
    /// determinant and trace which together fix the signs of both eigenvalues.
    ///
    /// # Arguments
    /// * `det_a` - Jacobian determinant det A.
    /// * `trace_a` - Jacobian trace tr A.
    pub fn from_invariants(det_a: f64, trace_a: f64) -> Self {
        if det_a < 0.0 {
            Parity::SADDLE
        } else if trace_a >= 0.0 {
            Parity::MINIMUM
        } else {
            Parity::MAXIMUM
        }
    }
}

/// Source point image: one solution θ of β = θ − α(θ), with its magnification,
/// parity, and Fermat potential.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Image {
    /// Image-plane position θ.
    pub theta: Point2D,
    /// Signed magnification μ = 1/det A.
    pub mu: f64,
    /// Parity (Fermat stationary-point type).
    pub parity: Parity,
    /// Fermat potential τ(θ; β) = ½|θ − β|² − ψ(θ).
    pub tau: f64,
}

impl Image {
    /// The image record at `theta` for a source at `beta` under `lens`: one jet
    /// evaluation yields μ, parity and the Fermat potential τ.
    ///
    /// # Arguments
    /// * `lens` - the lens model.
    /// * `theta` - image-plane position.
    /// * `beta` - source-plane position.
    pub fn at(lens: &dyn LensModel, theta: Point2D, beta: Point2D) -> Self {
        let jet = lens.jet(theta);
        let h = jet.hessian;
        let (a11, a22) = (1.0 - h[0][0], 1.0 - h[1][1]);
        let det_a = a11 * a22 - h[0][1] * h[1][0];
        Image {
            theta,
            mu: 1.0 / det_a,
            parity: Parity::from_invariants(det_a, a11 + a22),
            tau: 0.5 * (theta - beta).norm_squared() - jet.psi,
        }
    }
}

/// The images of one source, with optional arrival order and measured time delays.
///
/// Several systems can share one κ basis; each adds its own source-position unknowns to
/// the point-image polytope. Only the image positions and parities enter as constraints —
/// magnification and τ are model-derived.
#[derive(Debug, Clone)]
pub struct PointImageSystem {
    /// The observed images (positions and parities used; τ/μ ignored).
    pub images: Vec<Image>,
    /// Measured time delays between image pairs (empty if none).
    pub time_delays: Vec<TimeDelay>,
    /// Delay scale `k` in `Δt = k · Δτ` linking a physical delay to the dimensionless
    /// Fermat difference (`1.0` when delays are already dimensionless). See
    /// [`time_delay_scale`](super::forward::extrema::delays::time_delay_scale).
    pub delay_scale: f64,
    /// Explicit arrival order as image indices earliest-first (`None` if unknown).
    pub order: Option<Vec<usize>>,
}

impl PointImageSystem {
    /// A system with just image positions/parities — no delays or order.
    ///
    /// # Arguments
    /// * `images` - the observed images of the source.
    pub fn from_images(images: Vec<Image>) -> Self {
        Self { images, time_delays: Vec::new(), delay_scale: 1.0, order: None }
    }

    /// Attach measured time delays and their scale `k` (`Δt = k · Δτ`).
    ///
    /// # Arguments
    /// * `delays` - measured image-pair delays.
    /// * `scale` - the delay scale `k`.
    pub fn with_time_delays(mut self, delays: Vec<TimeDelay>, scale: f64) -> Self {
        self.time_delays = delays;
        self.delay_scale = scale;
        self
    }

    /// Attach an explicit arrival order (image indices, earliest first).
    ///
    /// # Arguments
    /// * `order` - arrival order as image indices.
    pub fn with_order(mut self, order: Vec<usize>) -> Self {
        self.order = Some(order);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::model::LensJet;
    use crate::lens::SIS;
    use crate::solver::forward::extrema::delays::TimeDelay;

    #[test]
    fn point_image_system_builders_populate_every_field() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let src = Point2D::new(0.2, 0.0);
        let images = vec![
            Image::at(&lens, Point2D::new(1.2, 0.0), src),
            Image::at(&lens, Point2D::new(-0.8, 0.0), src),
        ];
        // Bare system: no delays, unit scale, unknown order.
        let bare = PointImageSystem::from_images(images.clone());
        assert_eq!(bare.images.len(), 2);
        assert!(bare.time_delays.is_empty());
        assert_eq!(bare.delay_scale, 1.0);
        assert!(bare.order.is_none());
        // Chained builders set delays, scale and order.
        let full = PointImageSystem::from_images(images)
            .with_time_delays(vec![TimeDelay { from: 1, to: 0, delay: 3.5 }], 2.0)
            .with_order(vec![1, 0]);
        assert_eq!(full.time_delays.len(), 1);
        assert_eq!(full.delay_scale, 2.0);
        assert_eq!(full.order.as_deref(), Some(&[1usize, 0][..]));
    }

    /// A lens with a fixed jet everywhere, for exercising the derived quantities.
    struct ConstJet(LensJet);
    impl LensModel for ConstJet {
        fn jet(&self, _theta: Point2D) -> LensJet {
            self.0
        }
    }

    #[test]
    fn from_invariants_covers_the_three_types() {
        assert_eq!(Parity::from_invariants(-0.5, 0.5), Parity::SADDLE); // det A < 0
        assert_eq!(Parity::from_invariants(0.25, 1.0), Parity::MINIMUM); // both λ > 0
        assert_eq!(Parity::from_invariants(0.25, -1.0), Parity::MAXIMUM); // both λ < 0
    }

    #[test]
    fn from_eigenvalues_reads_the_sign_pattern() {
        assert_eq!(Parity::from_eigenvalues(0.5, 2.0), Parity::MINIMUM); // (+, +)
        assert_eq!(Parity::from_eigenvalues(-0.5, 2.0), Parity::SADDLE); // (−, +)
        assert_eq!(Parity::from_eigenvalues(0.5, -2.0), Parity::SADDLE); // (+, −)
        assert_eq!(Parity::from_eigenvalues(-0.5, -2.0), Parity::MAXIMUM); // (−, −)
    }

    #[test]
    fn sis_outer_image_is_a_minimum() {
        // Outside the Einstein radius det A = 1 − θ_E/r > 0 with λ_r = 1 > 0.
        let lens = SIS::new(Point2D::origin(), 1.0);
        let img = Image::at(&lens, Point2D::new(2.0, 0.0), Point2D::origin());
        assert_eq!(img.parity, Parity::MINIMUM);
        assert!((img.mu - 2.0).abs() < 1e-12); // μ = 1/(1 − θ_E/r) = 2
    }

    #[test]
    fn sis_inner_image_is_a_saddle() {
        // Inside the Einstein radius λ_t = 1 − θ_E/r < 0, so det A < 0.
        let lens = SIS::new(Point2D::origin(), 1.0);
        let img = Image::at(&lens, Point2D::new(0.5, 0.0), Point2D::origin());
        assert_eq!(img.parity, Parity::SADDLE);
        assert!(img.mu < 0.0);
    }

    #[test]
    fn central_over_focused_image_is_a_maximum() {
        // H = diag(1.5, 1.5) ⇒ A = diag(−0.5, −0.5): det A = 0.25, tr A = −1.
        let jet = LensJet {
            psi: 0.75,
            alpha: Point2D::origin(),
            hessian: [[1.5, 0.0], [0.0, 1.5]],
        };
        let lens = ConstJet(jet);
        let img = Image::at(&lens, Point2D::new(0.2, 0.1), Point2D::new(0.2, 0.1));
        assert_eq!(img.parity, Parity::MAXIMUM);
        assert!((img.mu - 4.0).abs() < 1e-12); // μ = 1/0.25
        assert!((img.tau + 0.75).abs() < 1e-12); // τ = ½·0 − ψ = −0.75
    }

    #[test]
    fn fermat_potential_includes_the_geometric_term() {
        // τ = ½|θ − β|² − ψ; with ψ = 0 it is the pure geometric delay.
        let jet = LensJet {
            psi: 0.0,
            alpha: Point2D::origin(),
            hessian: [[0.0, 0.0], [0.0, 0.0]],
        };
        let lens = ConstJet(jet);
        let img = Image::at(&lens, Point2D::new(3.0, 4.0), Point2D::origin());
        assert!((img.tau - 12.5).abs() < 1e-12); // ½·25
    }
}
