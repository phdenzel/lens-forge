// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended-image re-weighting of the point-image ensemble (Stage 2).
//!
//! Each polytope member is a proposal; the extended-image data supply the weight. Per
//! member: back-project the arc mask, solve for the source on it, re-lens and blur with the
//! PSF **in the image plane**, and compare to the data there with the proper noise
//! covariance. The forward χ² becomes a Gaussian-evidence importance weight; a cheap
//! source-plane dispersion cull kills badly wrong members before any inversion. The
//! effective sample size ([`effective_sample_size`]) flags weight collapse.

use rayon::prelude::*;

use crate::grid::{PixelGrid, ScalarField2D};
use crate::instrument::mask::Mask;
use crate::instrument::noise::Noise;
use crate::instrument::psf::Psf;
use crate::lens::LensModel;
use crate::solver::ensemble::stats::effective_sample_size;
use crate::solver::ensemble::Ensemble;
use crate::solver::extended::operator::LensingOperator;

use super::backproject::{source_mesh, source_plane_dispersion};
use super::semilinear::{invert_source, Regularizer};

/// Padding fraction for the per-member source mesh bounding box.
const PAD: f64 = 0.1;
/// Coarse source-plane binning for the dispersion cull.
const DISP_BINS: usize = 8;
/// CG tolerance and cap for the per-member source inversion.
const CG_TOL: f64 = 1e-10;
const CG_MAX_ITER: usize = 500;

/// The image-plane χ² of a `model` against `data` under the mask-weighted noise covariance.
///
/// # Arguments
/// * `model` - the modelled image.
/// * `data` - the observed image.
/// * `noise` - the noise model (supplies `C⁻¹`).
/// * `mask` - the pixel mask.
pub fn image_chi_squared(model: &[f64], data: &[f64], noise: &Noise, mask: &Mask) -> f64 {
    (0..model.len())
        .map(|i| {
            let r = model[i] - data[i];
            mask.weight(i) * r * r * noise.inverse_variance(i)
        })
        .sum()
}

/// Importance weights from per-member χ²: the normalized Gaussian evidence
/// `w_i ∝ exp(−½ χ²_i)`, computed stably relative to the best member. Non-finite χ²
/// (culled members) get zero weight.
///
/// # Arguments
/// * `chi2` - per-member image χ².
pub fn importance_weights(chi2: &[f64]) -> Vec<f64> {
    let min = chi2.iter().cloned().fold(f64::INFINITY, f64::min);
    if !min.is_finite() {
        return vec![0.0; chi2.len()];
    }
    let mut w: Vec<f64> = chi2.iter().map(|c| (-0.5 * (c - min)).exp()).collect();
    let sum: f64 = w.iter().sum();
    if sum > 0.0 {
        for wi in &mut w {
            *wi /= sum;
        }
    }
    w
}

/// Re-weight `ensemble` by extended-image evidence, returning the normalized weights and
/// their effective sample size.
///
/// # Arguments
/// * `ensemble` - the point-image ensemble to re-weight.
/// * `build_lens` - builds a lens model from a member's cell masses.
/// * `image` - the image-plane geometry (its grid carries the PSF convolution).
/// * `data` - the observed image (length `image.len()`).
/// * `noise` - the noise model.
/// * `mask` - the arc mask.
/// * `psf` - optional PSF applied to the model in the image plane.
/// * `lambda` - source-inversion regularization strength.
/// * `dispersion_cut` - optional source-plane dispersion above which a member is culled
///   (weight zero) before inversion.
pub fn reweight_ensemble<B>(
    ensemble: &Ensemble,
    build_lens: B,
    image: &PixelGrid,
    data: &[f64],
    noise: &Noise,
    mask: &Mask,
    psf: Option<&Psf>,
    lambda: f64,
    dispersion_cut: Option<f64>,
) -> (Vec<f64>, f64)
where
    B: Fn(&[f64]) -> Box<dyn LensModel> + Sync,
{
    let n_cells = ensemble.n_cells;
    let chi2: Vec<f64> = ensemble
        .members
        .par_iter()
        .map(|m| {
            let lens = build_lens(m.masses(n_cells));
            member_chi_squared(lens.as_ref(), image, data, noise, mask, psf, lambda, dispersion_cut)
        })
        .collect();
    let weights = importance_weights(&chi2);
    let ess = effective_sample_size(&weights);
    (weights, ess)
}

/// The image χ² of one member, or `+∞` if it is culled or its inversion cannot be set up.
#[allow(clippy::too_many_arguments)]
fn member_chi_squared(
    lens: &dyn LensModel,
    image: &PixelGrid,
    data: &[f64],
    noise: &Noise,
    mask: &Mask,
    psf: Option<&Psf>,
    lambda: f64,
    dispersion_cut: Option<f64>,
) -> f64 {
    if let Some(cut) = dispersion_cut {
        if source_plane_dispersion(lens, image, mask, DISP_BINS) > cut {
            return f64::INFINITY;
        }
    }
    let Ok(source) = source_mesh(lens, image, mask, PAD) else {
        return f64::INFINITY;
    };
    let f = LensingOperator::build(lens, image, &source);
    let reg = Regularizer::from_mesh(&source);
    let (s, _) = invert_source(&f, data, noise, mask, &reg, lambda, CG_TOL, CG_MAX_ITER);

    let model = f.apply(&s);
    let model = match psf {
        Some(p) => match PixelGrid::from_vec(image.grid.clone(), model).and_then(|g| p.apply(&g)) {
            Ok(blurred) => (0..blurred.len()).map(|i| blurred.value_at(i)).collect(),
            Err(_) => return f64::INFINITY,
        },
        None => model,
    };
    image_chi_squared(&model, data, noise, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Grid2D, Point2D};
    use crate::lens::SIS;
    use crate::solver::ensemble::{Ensemble, EnsembleMember};
    use crate::solver::extended::imaging::ImageSimulation;
    use crate::source::SourceField;

    #[test]
    fn chi_squared_is_zero_for_a_perfect_match() {
        let noise = Noise::uniform(0.5);
        let mask = Mask::all(4);
        let d = [1.0, 2.0, 3.0, 4.0];
        assert!(image_chi_squared(&d, &d, &noise, &mask) < 1e-15);
        assert!(image_chi_squared(&[1.0, 2.0, 3.0, 5.0], &d, &noise, &mask) > 0.0);
    }

    #[test]
    fn weights_favour_the_lower_chi_squared_and_normalize() {
        let w = importance_weights(&[2.0, 0.0, 6.0]);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(w[1] > w[0] && w[0] > w[2]);
        // A fully-culled ensemble has zero weights.
        assert_eq!(importance_weights(&[f64::INFINITY, f64::INFINITY]), vec![0.0, 0.0]);
    }

    /// Data: the arc of an SIS(θ_E = 0.5) lensing a compact off-centre Gaussian, an arc mask
    /// (the bright pixels only — the singular centre is excluded), and a two-member ensemble
    /// whose lenses are SIS built from each member's first mass.
    fn arc_scenario() -> (PixelGrid, Vec<f64>, Ensemble, Noise, Mask) {
        let truth = SIS::new(Point2D::origin(), 0.5);
        let src = PixelGrid::from_func(Grid2D::square(24, 1.5).unwrap(), |x, y| {
            (-((x - 0.1).powi(2) + y * y) / 0.03).exp()
        });
        let image = PixelGrid::constant(Grid2D::square(16, 2.0).unwrap(), 0.0);
        let data_img = ImageSimulation::Direct.image(&truth, &SourceField::new(src), &image);
        let data: Vec<f64> = (0..image.len()).map(|i| data_img.value_at(i)).collect();
        let mask = Mask::from_flags(&data.iter().map(|&d| d > 1e-3).collect::<Vec<_>>());
        let ensemble = Ensemble {
            n_cells: 1,
            n_sources: 0,
            members: vec![
                EnsembleMember { x: vec![0.5] },  // matches the truth
                EnsembleMember { x: vec![0.85] }, // wrong Einstein radius
            ],
        };
        (image, data, ensemble, Noise::uniform(0.05), mask)
    }

    fn sis_from_mass(m: &[f64]) -> Box<dyn LensModel> {
        Box::new(SIS::new(Point2D::origin(), m[0]))
    }

    #[test]
    fn reweighting_returns_valid_normalized_weights() {
        // A free per-pixel source fits either lens, so without a cull the weights merely
        // stay a valid normalized distribution — the honest baseline (the plan's caveat that
        // source-plane fitting only weakly breaks the lens degeneracy).
        let (image, data, ensemble, noise, mask) = arc_scenario();
        let (w, ess) =
            reweight_ensemble(&ensemble, sis_from_mass, &image, &data, &noise, &mask, None, 1e-4, None);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9, "weights: {w:?}");
        assert!(w.iter().all(|&x| x >= 0.0));
        assert!(ess > 0.0 && ess <= 2.0 + 1e-9, "ess {ess}");
    }

    #[test]
    fn a_fully_masked_ensemble_gets_zero_weights() {
        // No active pixels → every member's source mesh fails → all χ² are ∞ → zero weights.
        let (image, data, ensemble, noise, _) = arc_scenario();
        let mask = Mask::from_flags(&vec![false; image.len()]);
        let (w, ess) =
            reweight_ensemble(&ensemble, sis_from_mass, &image, &data, &noise, &mask, None, 1e-4, None);
        assert!(w.iter().all(|&x| x == 0.0));
        assert_eq!(ess, 0.0);
    }

    #[test]
    fn reweighting_applies_the_psf_in_the_image_plane() {
        // The PSF branch: the model is convolved before comparison; weights stay valid.
        let (image, data, ensemble, noise, mask) = arc_scenario();
        let psf = Psf::gaussian(0.15);
        let (w, ess) = reweight_ensemble(
            &ensemble, sis_from_mass, &image, &data, &noise, &mask, Some(&psf), 1e-4, None,
        );
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(ess > 0.0 && ess <= 2.0 + 1e-9);
    }

    #[test]
    fn a_generous_dispersion_cut_keeps_every_member() {
        // The cull is evaluated but passed (dispersion ≤ cut), so members proceed to the fit.
        let (image, data, ensemble, noise, mask) = arc_scenario();
        let (w, ess) = reweight_ensemble(
            &ensemble, sis_from_mass, &image, &data, &noise, &mask, None, 1e-4, Some(1e12),
        );
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(ess > 0.0);
    }

    #[test]
    fn the_dispersion_cull_zeroes_culled_members() {
        // A cut below every (non-negative) dispersion culls the whole ensemble to zero
        // weight — the deterministic test of the cull plumbing.
        let (image, data, ensemble, noise, mask) = arc_scenario();
        let (w, ess) = reweight_ensemble(
            &ensemble, sis_from_mass, &image, &data, &noise, &mask, None, 1e-4, Some(-1.0),
        );
        assert!(w.iter().all(|&x| x == 0.0), "everything culled: {w:?}");
        assert_eq!(ess, 0.0);
    }
}
