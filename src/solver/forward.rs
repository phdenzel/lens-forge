// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The forward-model composer: (lens, source) → observables.
//!
//! [`ForwardModel`] bundles a lens (and optionally a source) and offers the forward
//! observables as direct methods, delegating to [`mapping`](crate::solver::mapping),
//! [`extrema`](crate::solver::extrema) and [`extended`](crate::solver::extended): point
//! images, magnification/convergence maps, critical curves and caustics, time delays, the
//! lensed surface-brightness field, and — with an instrument — a simulated observation
//! (PSF + noise).

use crate::cosmology::Cosmology;
use crate::grid::{ContourSegment, FieldOps, PixelGrid, Point2D, ScalarField2D};
use crate::instrument::noise::Noise;
use crate::instrument::psf::Psf;
use crate::lens::LensModel;
use crate::rng::Lcg;
use crate::solver::extended::imaging::ImageSimulation;
use crate::solver::extrema::delays::time_delays;
use crate::solver::extrema::images::ImageFinder;
use crate::solver::image::Image;
use crate::solver::mapping::critical::{caustics, critical_curves, Stretch};
use crate::solver::mapping::convergence::convergence_map;
use crate::solver::mapping::magnification::magnification_map;
use crate::source::SourceModel;
use crate::{LensForgeError, Result};

/// A lens (and optional source) exposing the forward observables as direct methods.
pub struct ForwardModel<'a> {
    lens: &'a dyn LensModel,
    source: Option<&'a dyn SourceModel>,
}

impl<'a> ForwardModel<'a> {
    /// A forward model over `lens` alone (source-dependent methods will error).
    ///
    /// # Arguments
    /// * `lens` - the lens model.
    pub fn new(lens: &'a dyn LensModel) -> Self {
        Self { lens, source: None }
    }

    /// A forward model over `lens` lensing `source`.
    ///
    /// # Arguments
    /// * `lens` - the lens model.
    /// * `source` - the source surface-brightness model.
    pub fn with_source(lens: &'a dyn LensModel, source: &'a dyn SourceModel) -> Self {
        Self { lens, source: Some(source) }
    }

    /// All images of a point source at `beta`, found via `finder`.
    ///
    /// # Arguments
    /// * `finder` - the image finder (a built discretization).
    /// * `beta` - the source-plane position.
    pub fn images(&self, finder: &dyn ImageFinder, beta: Point2D) -> Vec<Image> {
        finder.images(self.lens, beta)
    }

    /// The magnification map μ(θ) = 1/det A on `field`'s geometry.
    ///
    /// # Arguments
    /// * `field` - the sampling geometry.
    pub fn magnification_map<F: FieldOps>(&self, field: &F) -> F {
        magnification_map(self.lens, field)
    }

    /// The convergence map κ(θ) = ½∇²ψ on `field`'s geometry.
    ///
    /// # Arguments
    /// * `field` - the sampling geometry.
    pub fn convergence_map<F: FieldOps>(&self, field: &F) -> F {
        convergence_map(self.lens, field)
    }

    /// The `stretch` critical curves (det A = 0 locus) on `field`'s geometry.
    ///
    /// # Arguments
    /// * `field` - the sampling geometry; it must span the critical curves.
    /// * `stretch` - which branch of the critical locus.
    pub fn critical_curves<F: FieldOps>(&self, field: &F, stretch: Stretch) -> Vec<ContourSegment> {
        critical_curves(self.lens, field, stretch)
    }

    /// The caustics: the `stretch` critical curves mapped to the source plane.
    ///
    /// # Arguments
    /// * `field` - the sampling geometry; it must span the critical curves.
    /// * `stretch` - which branch of the critical locus to map.
    pub fn caustics<F: FieldOps>(&self, field: &F, stretch: Stretch) -> Vec<ContourSegment> {
        caustics(self.lens, field, stretch)
    }

    /// Physical time delays (seconds) of `images` relative to the earliest.
    ///
    /// # Arguments
    /// * `images` - the images of the point source (from [`images`](Self::images)).
    /// * `cosmo` - the background cosmology supplying D_Δt.
    /// * `z_lens` - lens redshift.
    /// * `z_source` - source redshift.
    pub fn time_delays(
        &self,
        images: &[Image],
        cosmo: &dyn Cosmology,
        z_lens: f64,
        z_source: f64,
    ) -> Vec<f64> {
        time_delays(images, cosmo, z_lens, z_source)
    }

    /// The lensed surface-brightness field I(θ) = S(θ − α(θ)) on `field`'s geometry.
    ///
    /// # Arguments
    /// * `field` - the image-plane sampling geometry.
    /// * `sim` - the imaging strategy.
    ///
    /// # Errors
    /// Errors if the model has no source (built via [`new`](Self::new)).
    pub fn image<F: FieldOps>(&self, field: &F, sim: ImageSimulation) -> Result<F> {
        Ok(sim.image(self.lens, self.source()?, field))
    }

    /// A simulated observation: the lensed image, optionally blurred by `psf`, and — when
    /// `noise` is given — with a noise realization added in the image plane. With both `psf`
    /// and `noise` `None` this is the idealized (clean) lensed image.
    ///
    /// # Arguments
    /// * `field` - the image-plane sampling geometry (a pixel grid carries the PSF).
    /// * `sim` - the imaging strategy.
    /// * `psf` - optional PSF convolved into the image before adding noise.
    /// * `noise` - optional noise model paired with the random stream that realizes it;
    ///   `None` leaves the image noiseless.
    ///
    /// # Errors
    /// Errors if the model has no source, or if the PSF/grid construction fails.
    pub fn observe(
        &self,
        field: &PixelGrid,
        sim: ImageSimulation,
        psf: Option<&Psf>,
        noise: Option<(&Noise, &mut Lcg)>,
    ) -> Result<PixelGrid> {
        let clean: PixelGrid = sim.image(self.lens, self.source()?, field);
        let blurred = match psf {
            Some(p) => p.apply(&clean)?,
            None => clean,
        };
        match noise {
            Some((noise, rng)) => {
                let draw = noise.realize(rng, blurred.len());
                let values: Vec<f64> =
                    (0..blurred.len()).map(|i| blurred.value_at(i) + draw[i]).collect();
                PixelGrid::from_vec(blurred.grid, values)
            }
            None => Ok(blurred),
        }
    }

    /// The stored source, or an error if the model was built without one.
    fn source(&self) -> Result<&dyn SourceModel> {
        self.source.ok_or_else(|| {
            LensForgeError::invalid_parameter("ForwardModel has no source for imaging")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use crate::lens::SIS;
    use crate::solver::basis::MapPixellation;
    use crate::source::Gaussian;

    fn sis_and_source() -> (SIS, Gaussian) {
        (
            SIS::new(Point2D::origin(), 0.6),
            Gaussian::new(Point2D::new(0.1, 0.0), 1.0, 0.2, 1.0, 0.0),
        )
    }

    #[test]
    fn images_match_the_finder() {
        let lens = SIS::new(Point2D::origin(), 1.0);
        let fm = ForwardModel::new(&lens);
        let finder = MapPixellation::build(&lens, &Grid2D::square(80, 2.0).unwrap());
        let images = fm.images(&finder, Point2D::new(0.3, 0.0));
        assert_eq!(images.len(), 2); // the SIS double
    }

    #[test]
    fn image_requires_a_source() {
        let lens = SIS::new(Point2D::origin(), 0.6);
        let field = PixelGrid::constant(Grid2D::square(8, 2.0).unwrap(), 0.0);
        // Source-free model errors on imaging.
        assert!(ForwardModel::new(&lens).image(&field, ImageSimulation::Direct).is_err());
    }

    #[test]
    fn image_matches_direct_simulation() {
        let (lens, source) = sis_and_source();
        let field = PixelGrid::constant(Grid2D::square(32, 2.0).unwrap(), 0.0);
        let fm = ForwardModel::with_source(&lens, &source);
        let img = fm.image(&field, ImageSimulation::Direct).unwrap();
        let direct = ImageSimulation::Direct.image(&lens, &source, &field);
        for i in 0..img.len() {
            assert_eq!(img.value_at(i), direct.value_at(i));
        }
    }

    #[test]
    fn observe_without_psf_or_noise_matches_the_clean_image() {
        // No PSF and no noise: observe() is exactly the idealized clean lensed image.
        let (lens, source) = sis_and_source();
        let field = PixelGrid::constant(Grid2D::square(24, 2.0).unwrap(), 0.0);
        let fm = ForwardModel::with_source(&lens, &source);
        let obs = fm.observe(&field, ImageSimulation::Direct, None, None).unwrap();
        let clean = fm.image(&field, ImageSimulation::Direct).unwrap();
        for i in 0..obs.len() {
            assert!((obs.value_at(i) - clean.value_at(i)).abs() < 1e-12);
        }
    }

    #[test]
    fn observe_adds_noise_and_applies_the_psf() {
        // With a PSF and non-zero noise the observation departs from the clean image.
        let (lens, source) = sis_and_source();
        // The PSF FFT needs power-of-two dimensions.
        let field = PixelGrid::constant(Grid2D::square(32, 2.0).unwrap(), 0.0);
        let fm = ForwardModel::with_source(&lens, &source);
        let psf = Psf::gaussian(0.15);
        let mut rng = Lcg::new(0x77);
        let obs = fm
            .observe(&field, ImageSimulation::Direct, Some(&psf), Some((&Noise::uniform(0.05), &mut rng)))
            .unwrap();
        let clean = fm.image(&field, ImageSimulation::Direct).unwrap();
        let diff: f64 = (0..obs.len()).map(|i| (obs.value_at(i) - clean.value_at(i)).abs()).sum();
        assert!(diff > 0.0, "observation should differ from the clean image");
        assert_eq!(obs.len(), clean.len());
    }
}
