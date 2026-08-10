// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The free-form inversion driver: images (+ arcs) → ensemble → source.
//!
//! [`FreeFormInversion`] runs the whole inverse pipeline behind the [`Solve`] seam: assemble
//! the point-image polytope from a κ [`basis`](crate::solver::basis) and the observed image
//! systems, sample it for the degenerate-solution [`Ensemble`], and — when extended (arc) data
//! are supplied — re-weight the members by their forward fit, reduce to the
//! observation-constrained posterior, and optionally reconstruct each member's source.
//!
//! The driver is parameterized by the lens-plane basis `B` and the source-plane
//! representation `S`, e.g. `FreeFormInversion::<PixelGrid, MeshGrid>`.

use std::marker::PhantomData;

use crate::grid::PixelGrid;
use crate::instrument::mask::Mask;
use crate::instrument::noise::Noise;
use crate::instrument::psf::Psf;
use crate::lens::LensModel;
use crate::solver::basis::{KappaBasis, SourceBasis, SourceConfig};
use crate::solver::constraints::{ConstraintConfig, ConstraintSet};
use crate::solver::ensemble::summaries::{kappa_mean, kappa_variance, msd_levels};
use crate::solver::ensemble::{Ensemble, HitAndRun};
use crate::solver::image::PointImageSystem;
use crate::solver::solve::{Solution, Solve};
use crate::source::SourceField;
use crate::Result;

use super::reduce::{reduce_ensemble, ReduceConfig};
use super::reweight::reweight_ensemble;
use super::source_plane::reconstruct_source;

/// A per-member lens builder: cell masses `m` → a lens model.
pub type BuildLens<'a> = &'a (dyn Fn(&[f64]) -> Box<dyn LensModel> + Sync);

/// The extended (arc) data that turn the point-image ensemble into a weighted posterior.
pub struct ExtendedData<'a> {
    /// Builds a lens model from a member's cell masses.
    pub build_lens: BuildLens<'a>,
    /// The image-plane geometry (its grid carries any PSF convolution).
    pub image: &'a PixelGrid,
    /// The observed image (length `image.len()`).
    pub data: &'a [f64],
    /// The noise model.
    pub noise: &'a Noise,
    /// The arc mask.
    pub mask: &'a Mask,
    /// Optional PSF applied to each member's model in the image plane.
    pub psf: Option<&'a Psf>,
    /// Source-inversion regularization strength.
    pub lambda: f64,
    /// Optional source-plane dispersion cut culling bad members before inversion.
    pub dispersion_cut: Option<f64>,
}

/// The inputs to a [`FreeFormInversion`] solve.
pub struct FreeFormInput<'a, B: KappaBasis> {
    /// The κ basis (pixel or mesh) the polytope is built on.
    pub basis: &'a B,
    /// The observed point-image systems sharing this basis.
    pub systems: &'a [PointImageSystem],
    /// Optional extended data; when present, the ensemble is re-weighted and reduced.
    pub extended: Option<ExtendedData<'a>>,
}

/// Physical summaries of an ensemble's κ maps.
pub struct EnsembleSummary {
    /// Per-cell mean convergence.
    pub kappa_mean: Vec<f64>,
    /// Per-cell convergence variance.
    pub kappa_variance: Vec<f64>,
    /// Per-member mass-sheet level (the monopole/MSD degeneracy indicator).
    pub msd_levels: Vec<f64>,
}

/// The result of a free-form inversion.
pub struct FreeFormResult<S: SourceBasis> {
    /// The final ensemble — reduced to the posterior when extended data were supplied,
    /// otherwise the raw polytope sample.
    pub ensemble: Ensemble,
    /// κ summaries of [`ensemble`](Self::ensemble).
    pub summary: EnsembleSummary,
    /// Per-member importance weights, `Some` when extended data were supplied.
    pub weights: Option<Vec<f64>>,
    /// Effective sample size of the weights, `Some` when extended data were supplied.
    pub ess: Option<f64>,
    /// Per-member reconstructed sources, `Some` when reconstruction was requested.
    pub sources: Option<Vec<SourceField<S>>>,
}

/// The free-form inversion driver, parameterized by the κ basis `B` and source
/// representation `S`.
pub struct FreeFormInversion<B: KappaBasis, S: SourceBasis> {
    /// The polytope sampler.
    pub sampler: HitAndRun,
    /// Which regularizing inequalities the polytope builder emits.
    pub constraints: ConstraintConfig,
    /// Per-member source-plane build config.
    pub source: SourceConfig,
    /// Optional ensemble reduction applied after re-weighting.
    pub reduce: Option<ReduceConfig>,
    /// Whether to reconstruct each surviving member's source.
    pub reconstruct: bool,
    _marker: PhantomData<fn() -> (B, S)>,
}

impl<B: KappaBasis, S: SourceBasis> Default for FreeFormInversion<B, S> {
    fn default() -> Self {
        Self {
            sampler: HitAndRun::default(),
            constraints: ConstraintConfig::default(),
            source: SourceConfig::default(),
            reduce: None,
            reconstruct: false,
            _marker: PhantomData,
        }
    }
}

impl<B: KappaBasis, S: SourceBasis> FreeFormInversion<B, S> {
    /// A driver with default sampler/constraint/source settings and no reduction.
    pub fn new() -> Self {
        Self::default()
    }

    /// κ summaries of `ensemble` over the `basis` cell areas.
    fn summarize(basis: &B, ensemble: &Ensemble) -> EnsembleSummary {
        let areas: Vec<f64> = (0..basis.len()).map(|c| basis.cell_area(c)).collect();
        EnsembleSummary {
            kappa_mean: kappa_mean(ensemble, &areas),
            kappa_variance: kappa_variance(ensemble, &areas),
            msd_levels: msd_levels(ensemble, &areas),
        }
    }

    /// Reconstruct every member's source under `ext`.
    fn reconstruct_all(&self, ensemble: &Ensemble, ext: &ExtendedData) -> Result<Vec<SourceField<S>>> {
        ensemble
            .members
            .iter()
            .map(|m| {
                let lens = (ext.build_lens)(m.masses(ensemble.n_cells));
                reconstruct_source::<S>(
                    lens.as_ref(), ext.image, ext.data, ext.noise, ext.mask, ext.lambda, &self.source,
                )
            })
            .collect()
    }
}

impl<B: KappaBasis + 'static, S: SourceBasis> Solve for FreeFormInversion<B, S> {
    type Input<'a> = FreeFormInput<'a, B>;
    type Output = FreeFormResult<S>;

    fn solve(&self, input: FreeFormInput<'_, B>) -> Result<Solution<FreeFormResult<S>>> {
        let cs = ConstraintSet::assemble(input.basis, input.systems, &self.constraints);
        let (sample, info) = self.sampler.sample(&cs)?;

        let (ensemble, weights, ess) = match &input.extended {
            None => (sample, None, None),
            Some(ext) => {
                let weighted = reweight_ensemble::<S, _>(
                    &sample, ext.build_lens, ext.image, ext.data, ext.noise, ext.mask, ext.psf,
                    ext.lambda, &self.source, ext.dispersion_cut,
                );
                let ensemble = match &self.reduce {
                    Some(cfg) => reduce_ensemble(&weighted, cfg),
                    None => weighted.ensemble.clone(),
                };
                (ensemble, Some(weighted.weights), Some(weighted.ess))
            }
        };

        let summary = Self::summarize(input.basis, &ensemble);
        let sources = match (&input.extended, self.reconstruct) {
            (Some(ext), true) => Some(self.reconstruct_all(&ensemble, ext)?),
            _ => None,
        };

        Ok(Solution::new(FreeFormResult { ensemble, summary, weights, ess, sources }, info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reduce::ReduceMethod;
    use crate::grid::{Grid2D, MeshGrid, Point2D, ScalarField2D};
    use crate::lens::SIS;
    use crate::solver::extended::imaging::ImageSimulation;
    use crate::solver::image::Image;

    /// A coarse κ basis and the SIS double-image system of an axial source.
    fn double_scenario() -> (PixelGrid, Vec<PointImageSystem>) {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let src = Point2D::new(0.15, 0.0);
        let sys = PointImageSystem::from_images(vec![
            Image::at(&lens, Point2D::new(1.15, 0.0), src),
            Image::at(&lens, Point2D::new(-0.85, 0.0), src),
        ]);
        (PixelGrid::constant(Grid2D::square(5, 2.5).unwrap(), 0.3), vec![sys])
    }

    fn small_sampler() -> HitAndRun {
        HitAndRun { walkers: 3, burn_in: 60, per_walker: 20, seed: 11 }
    }

    fn sis_from_mass(m: &[f64]) -> Box<dyn LensModel> {
        Box::new(SIS::new(Point2D::origin(), 0.5 + m[0]))
    }

    #[test]
    fn point_image_only_yields_an_ensemble_and_summary() {
        let (basis, systems) = double_scenario();
        let driver = FreeFormInversion::<PixelGrid, MeshGrid> { sampler: small_sampler(), ..Default::default() };
        let sol = driver.solve(FreeFormInput { basis: &basis, systems: &systems, extended: None }).unwrap();
        assert!(sol.info.converged && !sol.output.ensemble.is_empty());
        assert_eq!(sol.output.summary.kappa_mean.len(), basis.len());
        assert!(sol.output.weights.is_none() && sol.output.sources.is_none());
    }

    /// An SIS arc scenario: image geometry, data, noise and an arc mask.
    fn arc_scenario() -> (PixelGrid, Vec<f64>, Noise, Mask) {
        let truth = SIS::new(Point2D::origin(), 0.5);
        let src = PixelGrid::from_func(Grid2D::square(24, 1.5).unwrap(), |x, y| {
            (-((x - 0.1).powi(2) + y * y) / 0.03).exp()
        });
        let image = PixelGrid::constant(Grid2D::square(16, 2.0).unwrap(), 0.0);
        let data_img = ImageSimulation::Direct.image(&truth, &SourceField::new(src), &image);
        let data: Vec<f64> = (0..image.len()).map(|i| data_img.value_at(i)).collect();
        let mask = Mask::from_flags(&data.iter().map(|&d| d > 1e-3).collect::<Vec<_>>());
        (image, data, Noise::uniform(0.05), mask)
    }

    #[test]
    fn extended_data_reweights_and_reduces_the_ensemble() {
        let (basis, systems) = double_scenario();
        let (image, data, noise, mask) = arc_scenario();
        let driver = FreeFormInversion::<PixelGrid, MeshGrid> {
            sampler: small_sampler(),
            reduce: Some(ReduceConfig { method: ReduceMethod::Resample { target: 8 }, seed: 3 }),
            ..Default::default()
        };
        let ext = ExtendedData {
            build_lens: &sis_from_mass,
            image: &image,
            data: &data,
            noise: &noise,
            mask: &mask,
            psf: None,
            lambda: 1e-4,
            dispersion_cut: None,
        };
        let sol = driver
            .solve(FreeFormInput { basis: &basis, systems: &systems, extended: Some(ext) })
            .unwrap();
        // Re-weighted, then reduced to the requested target.
        assert_eq!(sol.output.ensemble.len(), 8);
        let w = sol.output.weights.expect("weights present");
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(sol.output.ess.unwrap() > 0.0);
        assert!(sol.output.sources.is_none()); // reconstruct not requested
    }

    #[test]
    fn reconstruction_yields_one_source_per_member() {
        let (basis, systems) = double_scenario();
        let (image, data, noise, mask) = arc_scenario();
        let driver = FreeFormInversion::<PixelGrid, MeshGrid> {
            sampler: small_sampler(),
            reduce: Some(ReduceConfig { method: ReduceMethod::Resample { target: 3 }, seed: 5 }),
            reconstruct: true,
            ..Default::default()
        };
        let ext = ExtendedData {
            build_lens: &sis_from_mass,
            image: &image,
            data: &data,
            noise: &noise,
            mask: &mask,
            psf: None,
            lambda: 1e-4,
            dispersion_cut: None,
        };
        let sol = driver
            .solve(FreeFormInput { basis: &basis, systems: &systems, extended: Some(ext) })
            .unwrap();
        let sources = sol.output.sources.expect("sources present");
        assert_eq!(sources.len(), sol.output.ensemble.len());
    }
}
