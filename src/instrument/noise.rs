// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The data covariance C: inverse-variance weighting and noise realizations.
//!
//! A [`Noise`] model is the observation's covariance, used both ways: it supplies
//! the diagonal of C⁻¹ for the likelihood χ² = rᵀC⁻¹r and the normal equations
//! FᵀC⁻¹F of the inverse, and it draws Gaussian realizations n ~ N(0, C) to build
//! mock data d = model + n.

use crate::rng::Lcg;

/// A diagonal data-covariance model: Gaussian pixel noise with a per-cell σ.
#[derive(Debug, Clone, PartialEq)]
pub enum Noise {
    /// The same standard deviation `sigma` on every cell (homoscedastic).
    Uniform {
        /// Per-cell standard deviation σ (> 0).
        sigma: f64,
    },
    /// A per-cell standard deviation map (an rms/error image), σ per cell (> 0).
    SigmaMap(Vec<f64>),
}

impl Noise {
    /// Homoscedastic noise with the same σ on every cell.
    ///
    /// # Arguments
    /// * `sigma` - per-cell standard deviation (> 0).
    pub fn uniform(sigma: f64) -> Self {
        Noise::Uniform { sigma }
    }

    /// Per-cell noise from a σ map.
    ///
    /// # Arguments
    /// * `sigmas` - per-cell standard deviations (> 0).
    pub fn sigma_map(sigmas: Vec<f64>) -> Self {
        Noise::SigmaMap(sigmas)
    }

    /// Standard deviation σ at cell `i`.
    fn sigma_at(&self, i: usize) -> f64 {
        match self {
            Noise::Uniform { sigma } => *sigma,
            Noise::SigmaMap(sigmas) => sigmas[i],
        }
    }

    /// The diagonal of C⁻¹ at cell `i`: the inverse variance 1/σ².
    ///
    /// # Arguments
    /// * `i` - cell index.
    pub fn inverse_variance(&self, i: usize) -> f64 {
        let s = self.sigma_at(i);
        1.0 / (s * s)
    }

    /// The inverse-variance-weighted residual C⁻¹ r, i.e. `r_i / σ_i²`.
    ///
    /// # Arguments
    /// * `residual` - the residual vector r (its length sets the cell count; for a
    ///   [`SigmaMap`](Noise::SigmaMap) it must match the map length).
    pub fn weight(&self, residual: &[f64]) -> Vec<f64> {
        self.check_len(residual.len());
        residual.iter().enumerate().map(|(i, &r)| r * self.inverse_variance(i)).collect()
    }

    /// The χ² = rᵀ C⁻¹ r of a residual r.
    ///
    /// # Arguments
    /// * `residual` - the residual vector r (see [`weight`](Noise::weight) for the
    ///   length rule).
    pub fn chi_squared(&self, residual: &[f64]) -> f64 {
        self.check_len(residual.len());
        residual.iter().enumerate().map(|(i, &r)| r * r * self.inverse_variance(i)).sum()
    }

    /// A noise realization of `n` cells: each `n_i ~ N(0, σ_i²)`.
    ///
    /// # Arguments
    /// * `rng` - the random source.
    /// * `n` - number of cells (must match the map length for a
    ///   [`SigmaMap`](Noise::SigmaMap)).
    pub fn realize(&self, rng: &mut Lcg, n: usize) -> Vec<f64> {
        self.check_len(n);
        (0..n).map(|i| self.sigma_at(i) * rng.gaussian()).collect()
    }

    /// Panic if `len` disagrees with a `SigmaMap`'s length (a no-op for `Uniform`).
    fn check_len(&self, len: usize) {
        if let Noise::SigmaMap(sigmas) = self {
            assert_eq!(len, sigmas.len(), "vector length must match the σ map");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_weights_by_a_single_inverse_variance() {
        let noise = Noise::uniform(2.0); // σ² = 4
        assert_eq!(noise.inverse_variance(7), 0.25);
        assert_eq!(noise.weight(&[2.0, 4.0]), vec![0.5, 1.0]);
        assert_eq!(noise.chi_squared(&[2.0, 4.0]), 5.0); // (4 + 16) / 4
    }

    #[test]
    fn sigma_map_weights_per_cell() {
        let noise = Noise::sigma_map(vec![1.0, 2.0]);
        assert_eq!(noise.inverse_variance(0), 1.0);
        assert_eq!(noise.inverse_variance(1), 0.25);
        assert_eq!(noise.weight(&[3.0, 8.0]), vec![3.0, 2.0]);
        assert_eq!(noise.chi_squared(&[3.0, 8.0]), 25.0); // 9/1 + 64/4
    }

    #[test]
    fn chi_squared_is_the_residual_dotted_with_its_weight() {
        let noise = Noise::sigma_map(vec![0.5, 1.5, 3.0]);
        let r = [0.7, -1.2, 2.1];
        let dot: f64 = r.iter().zip(noise.weight(&r)).map(|(a, b)| a * b).sum();
        assert!((noise.chi_squared(&r) - dot).abs() < 1e-12);
    }

    #[test]
    fn realization_recovers_the_cell_sigma() {
        // Repeated one-cell realizations of σ = 3 have variance ≈ 9.
        let noise = Noise::sigma_map(vec![3.0]);
        let mut rng = Lcg::new(0x0B1A5E);
        let xs: Vec<f64> = (0..100_000).map(|_| noise.realize(&mut rng, 1)[0]).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!((var - 9.0).abs() < 0.2, "var {var}");
    }

    #[test]
    fn uniform_realization_has_the_requested_length() {
        let noise = Noise::uniform(1.0);
        assert_eq!(noise.realize(&mut Lcg::new(1), 16).len(), 16);
    }

    #[test]
    #[should_panic(expected = "σ map")]
    fn sigma_map_rejects_a_length_mismatch() {
        Noise::sigma_map(vec![1.0, 2.0]).chi_squared(&[1.0]);
    }
}
