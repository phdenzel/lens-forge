// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Scaling transformations and mappings for rendering.
use crate::{LensForgeError, Result};
use rayon::prelude::*;

/// Compute automatic min/max from a value slice.
///
/// # Arguments
/// * `values` - Value array slice
pub fn auto_range(values: &[f32]) -> Option<(f32, f32)> {
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    if vmin.is_finite() && vmax.is_finite() {
        Some((vmin, vmax))
    } else {
        None
    }
}

/// Scaling applied to map values for visualizations.
#[derive(Clone, Copy, Debug, Default)]
pub enum RenderScale {
    /// no scaling
    #[default]
    Linear,
    /// Base-10 logarithmic scaling
    Log10,
    /// Symmetric log-like scaling with a linear threshold around zero.
    SymLog {
        /// Half-width of linear region around zero.
        linthresh: f32,
    },
}

/// Various scaling kernels and transformations.
///
/// # Kernels
/// Functions that scale values `v` and return `None` for non-mappable results;
/// unwrap_or variants additionally return `fallback` values instead of `None`.
/// * `apply_linear` / `apply_linear_or`
/// * `apply_log10` / `apply_log10_or`
/// * `apply_symlog` / `apply_symlog_or`
/// * `revert_linear` / `revert_linear_or`
/// * `revert_log10` / `revert_log10_or`
/// * `revert_symlog` / `revert_symlog_or`
impl RenderScale {
    #[inline]
    pub fn apply_linear(v: f32) -> Option<f32> {
        if !v.is_finite() { None } else { Some(v) }
    }
    #[inline]
    pub fn apply_linear_or(v: f32, fallback: f32) -> f32 {
        RenderScale::apply_linear(v).unwrap_or(fallback)
    }
    #[inline]
    pub fn revert_linear(s: f32) -> Option<f32> {
        if !s.is_finite() { None } else { Some(s) }
    }
    #[inline]
    pub fn revert_linear_or(s: f32, fallback: f32) -> f32 {
        RenderScale::revert_linear(s).unwrap_or(fallback)
    }
    #[inline]
    pub fn apply_log10(v: f32) -> Option<f32> {
        if v > 0.0 && v.is_finite() {
            Some(v.log10())
        } else {
            None
        }
    }
    #[inline]
    pub fn apply_log10_or(v: f32, fallback: f32) -> f32 {
        RenderScale::apply_log10(v).unwrap_or(fallback)
    }
    #[inline]
    pub fn revert_log10(s: f32) -> Option<f32> {
        let v = 10.0_f32.powf(s);
        if v.is_finite() { Some(v) } else { None }
    }
    #[inline]
    pub fn revert_log10_or(s: f32, fallback: f32) -> f32 {
        RenderScale::revert_log10(s).unwrap_or(fallback)
    }
    #[inline]
    pub fn apply_symlog(v: f32, linthresh: f32) -> Option<f32> {
        if !linthresh.is_finite() || linthresh <= 0.0 {
            return None;
        }
        let vabs = v.abs();
        if vabs <= linthresh {
            RenderScale::apply_linear(v)
        } else {
            let sgn = v.signum();
            let scaled = sgn * linthresh * (1.0 + (vabs / linthresh).log10());
            if scaled.is_finite() {
                Some(scaled)
            } else {
                None
            }
        }
    }
    #[inline]
    pub fn apply_symlog_or(v: f32, linthresh: f32, fallback: f32) -> f32 {
        RenderScale::apply_symlog(v, linthresh).unwrap_or(fallback)
    }
    #[inline]
    pub fn revert_symlog(s: f32, linthresh: f32) -> Option<f32> {
        if !linthresh.is_finite() || linthresh <= 0.0 {
            return None;
        }
        let sabs = s.abs();
        if sabs <= linthresh {
            RenderScale::revert_linear(s)
        } else {
            let sgn = s.signum();
            let exp = sabs / linthresh - 1.0;
            let v = sgn * linthresh * 10.0_f32.powf(exp);
            if v.is_finite() { Some(v) } else { None }
        }
    }
    #[inline]
    pub fn revert_symlog_or(s: f32, linthresh: f32, fallback: f32) -> f32 {
        RenderScale::revert_symlog(s, linthresh).unwrap_or(fallback)
    }

    /// Apply the specified scaling and return `None` if result is invalid.
    ///
    /// # Arguments
    /// * `v` - Value to scale
    #[inline]
    pub fn apply(&self, v: f32) -> Option<f32> {
        if !v.is_finite() {
            return None;
        }
        match *self {
            RenderScale::Linear => RenderScale::apply_linear(v),
            RenderScale::Log10 => RenderScale::apply_log10(v),
            RenderScale::SymLog { linthresh } => RenderScale::apply_symlog(v, linthresh),
        }
    }

    /// Safely apply going into failure mode for invalid values.
    ///
    /// # Arguments
    /// * `v` - Value to scale
    #[inline]
    #[must_use]
    pub fn try_apply(&self, v: f32) -> Result<f32> {
        if !v.is_finite() {
            return Err(LensForgeError::numerical_error("Need valid, finite input."));
        }
        match *self {
            RenderScale::Linear => Ok(RenderScale::apply_linear(v).unwrap()),
            RenderScale::Log10 => {
                if v <= 0.0 {
                    Err(LensForgeError::numerical_error(
                        "RenderScale::Log10 requires v > 0",
                    ))
                } else {
                    Ok(RenderScale::apply_log10(v).unwrap())
                }
            }
            RenderScale::SymLog { linthresh } => {
                if !linthresh.is_finite() || linthresh <= 0.0 {
                    return Err(LensForgeError::invalid_option(
                        "RenderScale::SymLog requires finite linthresh > 0",
                    ));
                }
                Ok(RenderScale::apply_symlog(v, linthresh).unwrap())
            }
        }
    }

    /// Apply the specified scaling with fallback.
    ///
    /// # Arguments
    /// * `v` - Value to scale
    /// * `fallback` - Fallback value for invalid scaling results
    #[inline]
    pub fn apply_or(&self, v: f32, fallback: f32) -> f32 {
        self.try_apply(v).unwrap_or(fallback)
    }

    /// Inverse of the specified scaling and return `None` if result is invalid.
    ///
    /// # Arguments
    /// * `s` - Value to unscale
    #[inline]
    pub fn revert(&self, s: f32) -> Option<f32> {
        if !s.is_finite() {
            return None;
        }
        match *self {
            RenderScale::Linear => RenderScale::revert_linear(s),
            RenderScale::Log10 => RenderScale::revert_log10(s),
            RenderScale::SymLog { linthresh } => RenderScale::revert_symlog(s, linthresh),
        }
    }

    /// Safely apply going into failure mode for invalid values.
    ///
    /// # Arguments
    /// * `s` - Value to unscale
    #[inline]
    #[must_use]
    pub fn try_revert(&self, s: f32) -> Result<f32> {
        if !s.is_finite() {
            return Err(LensForgeError::numerical_error("Need valid, finite input."));
        }
        match *self {
            RenderScale::Linear => Ok(RenderScale::revert_linear(s).unwrap()),
            RenderScale::Log10 => {
                let v = RenderScale::revert_log10(s);
                if v.is_none() {
                    return Err(LensForgeError::numerical_error(
                        "RenderScale::Log10 produced non-finite result.",
                    ));
                }
                Ok(v.unwrap())
            }
            RenderScale::SymLog { linthresh } => {
                if !linthresh.is_finite() || linthresh <= 0.0 {
                    return Err(LensForgeError::invalid_option(
                        "RenderScale::SymLog requires finite linthresh > 0",
                    ));
                }
                Ok(RenderScale::revert_symlog(s, linthresh).unwrap())
            }
        }
    }

    /// Inverse of the specified scaling.
    ///
    /// # Arguments
    /// * `s` - Value to unscale
    /// * `fallback` - Fallback value for invalid results
    #[inline]
    pub fn revert_or(&self, s: f32, fallback: f32) -> f32 {
        self.try_revert(s).unwrap_or(fallback)
    }

    /// Apply scaling to an entire slice, writing results to output buffer.
    ///
    /// # Arguments
    /// * `values` - Input slice
    /// * `output` - Output buffer (must have same length as input)
    /// * `fallback` - Value to use for invalid inputs
    pub fn apply_slice(&self, values: &[f32], output: &mut [f32], fallback: f32) {
        assert_eq!(
            values.len(),
            output.len(),
            "Input and output slices must have same length"
        );

        match *self {
            RenderScale::Linear => {
                for (v, out) in values.iter().zip(output.iter_mut()) {
                    *out = if v.is_finite() { *v } else { fallback };
                }
            }
            RenderScale::Log10 => {
                for (v, out) in values.iter().zip(output.iter_mut()) {
                    *out = if *v > 0.0 && v.is_finite() {
                        v.log10()
                    } else {
                        fallback
                    };
                }
            }
            RenderScale::SymLog { linthresh } => {
                if !linthresh.is_finite() || linthresh <= 0.0 {
                    output.fill(fallback);
                    return;
                }
                for (v, out) in values.iter().zip(output.iter_mut()) {
                    *out = Self::apply_symlog_or(*v, linthresh, fallback);
                }
            }
        }
    }

    /// Apply scaling to slice in parallel (using rayon backend).
    ///
    /// # Arguments
    /// * `values` - Input slice
    /// * `output` - Output buffer (must have same length as input)
    /// * `fallback` - Value to use for invalid inputs
    pub fn par_apply_slice(&self, values: &[f32], output: &mut [f32], fallback: f32) {
        assert_eq!(
            values.len(),
            output.len(),
            "Input and output slices must have same length"
        );

        match *self {
            RenderScale::Linear => {
                values
                    .par_iter()
                    .zip(output.par_iter_mut())
                    .for_each(|(v, out)| {
                        *out = if v.is_finite() { *v } else { fallback };
                    });
            }
            RenderScale::Log10 => {
                values
                    .par_iter()
                    .zip(output.par_iter_mut())
                    .for_each(|(v, out)| {
                        *out = if *v > 0.0 && v.is_finite() {
                            v.log10()
                        } else {
                            fallback
                        };
                    });
            }
            RenderScale::SymLog { linthresh } => {
                if !linthresh.is_finite() || linthresh <= 0.0 {
                    output.par_iter_mut().for_each(|out| *out = fallback);
                    return;
                }
                values
                    .par_iter()
                    .zip(output.par_iter_mut())
                    .for_each(|(v, out)| {
                        *out = Self::apply_symlog_or(*v, linthresh, fallback);
                    });
            }
        }
    }

    /// Apply scaling in-place to a mutable slice.
    ///
    /// # Arguments
    /// * `values` - Mutable slice to transform in-place
    /// * `fallback` - Value to use for invalid inputs
    pub fn apply_slice_inplace(&self, values: &mut [f32], fallback: f32) {
        match *self {
            RenderScale::Linear => {
                for v in values.iter_mut() {
                    if !v.is_finite() {
                        *v = fallback;
                    }
                }
            }
            RenderScale::Log10 => {
                for v in values.iter_mut() {
                    *v = if *v > 0.0 && v.is_finite() {
                        v.log10()
                    } else {
                        fallback
                    };
                }
            }
            RenderScale::SymLog { linthresh } => {
                if !linthresh.is_finite() || linthresh <= 0.0 {
                    values.fill(fallback);
                    return;
                }
                for v in values.iter_mut() {
                    *v = Self::apply_symlog_or(*v, linthresh, fallback);
                }
            }
        }
    }

    /// Apply scaling in-place to a mutable slice in parallel (using rayon backend).
    ///
    /// # Arguments
    /// * `values` - Mutable slice to transform in-place
    /// * `fallback` - Value to use for invalid inputs
    pub fn par_apply_slice_inplace(&self, values: &mut [f32], fallback: f32) {
        match *self {
            RenderScale::Linear => {
                values.par_iter_mut().for_each(|v| {
                    if !v.is_finite() {
                        *v = fallback;
                    }
                });
            }
            RenderScale::Log10 => {
                values.par_iter_mut().for_each(|v| {
                    *v = if *v > 0.0 && v.is_finite() {
                        v.log10()
                    } else {
                        fallback
                    };
                });
            }
            RenderScale::SymLog { linthresh } => {
                if !linthresh.is_finite() || linthresh <= 0.0 {
                    values.par_iter_mut().for_each(|v| *v = fallback);
                    return;
                }
                values.par_iter_mut().for_each(|v| {
                    *v = Self::apply_symlog_or(*v, linthresh, fallback);
                });
            }
        }
    }
}

/* ============ Normalize ================================================== */
/// Normalizer that maps values between specified range to [0, 1].
#[derive(Clone, Copy, Debug)]
pub struct Normalize {
    /// Normalization range limits [`min`, `max`]
    pub min: f32,
    pub max: f32,
    /// Whether to clamp values beyond normalization range
    pub clamp: bool,
    /// Numerical precision parameter
    pub eps: f32,
}

impl Default for Normalize {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            clamp: true,
            eps: 1e-12,
        }
    }
}

impl Normalize {
    /// Builder for setting `clamp` field.
    pub fn with_clamp(mut self, clamp: bool) -> Self {
        self.clamp = clamp;
        self
    }
    /// Builder for setting numerical stability `eps` field.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    /// Construct normalizer from limits `min` and `max`.
    pub fn with_lims(mut self, min: f32, max: f32) -> Result<Self> {
        if !min.is_finite() || !max.is_finite() {
            return Err(LensForgeError::invalid_option("Limits must be finite"));
        }
        self.min = min;
        self.max = max;
        let range = self.range();
        if range < 0.0 {
            return Err(LensForgeError::invalid_option(
                "Normalization range must be > 0",
            ));
        }
        Ok(self)
    }
    /// Construct normalizer from `min` and `range`.
    pub fn from_range(min: f32, range: f32) -> Result<Self> {
        let max = min + range;
        Self::default().with_lims(min, max)
    }
    /// Getter for the normalization range.
    #[inline]
    pub fn range(&self) -> f32 {
        self.max - self.min
    }
    /// Getter for normalization range with numerical stability.
    #[inline]
    pub fn range_eps(&self) -> f32 {
        let range = self.range();
        if range.abs() < self.eps {
            self.eps
        } else {
            range
        }
    }
    /// Clamp value `v` iff normalizer is configured to do so.
    #[inline]
    pub fn clamp_or(&self, v: f32) -> f32 {
        if self.clamp { v.clamp(0.0, 1.0) } else { v }
    }

    pub fn apply(&self, v: f32) -> f32 {
        if !v.is_finite() {
            return f32::NAN;
        }
        let range = self.range_eps();
        let normalized = (v - self.min) / range;
        self.clamp_or(normalized)
    }

    pub fn try_apply(&self, v: f32) -> Result<f32> {
        if !v.is_finite() {
            return Err(LensForgeError::numerical_error(
                "Normalize::apply requires finite `v`",
            ));
        }
        let normalized = self.apply(v);
        println!("Normalized {}", normalized);
        if !normalized.is_finite() {
            return Err(LensForgeError::numerical_error(
                "Normalize::apply returned non-finite result.",
            ));
        }
        Ok(normalized)
    }

    pub fn revert(&self, n: f32) -> f32 {
        if !n.is_finite() {
            return f32::NAN;
        }
        let range = self.range_eps();
        n * range + self.min
    }

    pub fn try_revert(&self, n: f32) -> Result<f32> {
        if !n.is_finite() {
            return Err(LensForgeError::numerical_error(
                "Normalize::revert requires finite `n`",
            ));
        }
        if self.clamp {
            return Err(LensForgeError::invalid_option(
                "Normalize::revert is not possible when `clamp` is `true`",
            ));
        }
        let v = self.revert(n);
        if !v.is_finite() {
            return Err(LensForgeError::numerical_error(
                "Normalize::revert returned non-finite result.",
            ));
        }
        Ok(v)
    }

    /// Normalize an entire slice, writing results to output buffer.
    ///
    /// # Arguments
    /// * `values` - Input slice
    /// * `output` - Output buffer (must have same length as input)
    pub fn apply_slice(&self, values: &[f32], output: &mut [f32]) {
        assert_eq!(values.len(), output.len());
        let range = self.range_eps();
        let inv_range = 1.0 / range;

        if self.clamp {
            for (v, out) in values.iter().zip(output.iter_mut()) {
                if v.is_finite() {
                    let normalized = (*v - self.min) * inv_range;
                    *out = normalized.clamp(0.0, 1.0);
                } else {
                    *out = f32::NAN;
                }
            }
        } else {
            for (v, out) in values.iter().zip(output.iter_mut()) {
                *out = if v.is_finite() {
                    (*v - self.min) * inv_range
                } else {
                    f32::NAN
                };
            }
        }
    }

    /// Normalize slice in parallel (using rayon backend).
    ///
    /// # Arguments
    /// * `values` - Input slice
    /// * `output` - Output buffer (must have same length as input)
    pub fn par_apply_slice(&self, values: &[f32], output: &mut [f32]) {
        assert_eq!(values.len(), output.len());
        let range = self.range_eps();
        let inv_range = 1.0 / range;

        if self.clamp {
            values
                .par_iter()
                .zip(output.par_iter_mut())
                .for_each(|(v, out)| {
                    if v.is_finite() {
                        let normalized = (*v - self.min) * inv_range;
                        *out = normalized.clamp(0.0, 1.0);
                    } else {
                        *out = f32::NAN;
                    }
                });
        } else {
            values
                .par_iter()
                .zip(output.par_iter_mut())
                .for_each(|(v, out)| {
                    *out = if v.is_finite() {
                        (*v - self.min) * inv_range
                    } else {
                        f32::NAN
                    };
                });
        }
    }

    /// Normalize in-place.
    ///
    /// # Arguments
    /// * `values` - Mutable slice to normalize in-place
    pub fn apply_slice_inplace(&self, values: &mut [f32]) {
        let range = self.range_eps();
        let inv_range = 1.0 / range;

        if self.clamp {
            for v in values.iter_mut() {
                if v.is_finite() {
                    let normalized = (*v - self.min) * inv_range;
                    *v = normalized.clamp(0.0, 1.0);
                } else {
                    *v = f32::NAN;
                }
            }
        } else {
            for v in values.iter_mut() {
                *v = if v.is_finite() {
                    (*v - self.min) * inv_range
                } else {
                    f32::NAN
                };
            }
        }
    }

    /// Normalize in-place in parallel (using rayon backend).
    ///
    /// # Arguments
    /// * `values` - Mutable slice to normalize in-place
    pub fn par_apply_slice_inplace(&self, values: &mut [f32]) {
        let range = self.range_eps();
        let inv_range = 1.0 / range;

        if self.clamp {
            values.par_iter_mut().for_each(|v| {
                if v.is_finite() {
                    let normalized = (*v - self.min) * inv_range;
                    *v = normalized.clamp(0.0, 1.0);
                } else {
                    *v = f32::NAN;
                }
            });
        } else {
            values.par_iter_mut().for_each(|v| {
                *v = if v.is_finite() {
                    (*v - self.min) * inv_range
                } else {
                    f32::NAN
                };
            });
        }
    }
}

/* ============ Unit tests ================================================= */
#[cfg(test)]
mod tests {
    use super::{LensForgeError, Normalize, RenderScale};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    /* ============ RenderScale tests ====================================== */
    #[test]
    fn linear_roundtrip() {
        let s = RenderScale::Linear;
        for &v in &[0.0, 1.0, -3.5, 1e6] {
            let f = s.apply(v).unwrap();
            let b = s.revert(f).unwrap();
            assert!(approx(v, b, 1e-10));
        }
    }

    #[test]
    fn log10_validity() {
        let s = RenderScale::Log10;
        assert!(s.apply(-1.0).is_none());
        assert!(s.apply(0.0).is_none());
        assert!(s.apply(f32::NAN).is_none());
    }

    #[test]
    fn log10_roundtrip() {
        let s = RenderScale::Log10;
        for &v in &[1e-6, 1.0, 10.0, 1e6] {
            let f = s.apply(v).unwrap();
            let b = s.revert(f).unwrap();
            assert!(approx(v, b, 1e-10));
        }
    }

    #[test]
    fn symlog_continuity() {
        let s = RenderScale::SymLog { linthresh: 2.0 };

        // continuity at boundary
        let f_pos = s.apply(2.0).unwrap();
        let f_neg = s.apply(-2.0).unwrap();
        assert!(approx(f_pos, 2.0, 1e-10));
        assert!(approx(f_neg, -2.0, 1e-10));
    }

    #[test]
    fn symlog_roundtrip() {
        let s = RenderScale::SymLog { linthresh: 2.0 };

        // roundtrip across ranges
        for &v in &[-1000.0, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 1000.0] {
            let f = s.apply(v).unwrap();
            let b = s.revert(f).unwrap();
            assert!(approx(v, b, 1e-6));
        }
    }

    #[test]
    fn linear_try_apply_revert_errors_on_non_finite() {
        let s = RenderScale::Linear;
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                s.try_apply(bad),
                Err(LensForgeError::NumericalError(_))
            ));
            assert!(matches!(
                s.try_revert(bad),
                Err(LensForgeError::NumericalError(_))
            ));
        }
    }

    #[test]
    fn linear_fallbacks_work() {
        let s = RenderScale::Linear;
        let fb = 42.0;
        assert_eq!(s.apply_or(f32::NAN, fb), fb);
        assert_eq!(s.revert_or(f32::INFINITY, fb), fb);
    }

    #[test]
    fn log10_try_apply_on_non_finite() {
        let s = RenderScale::Log10;
        // try_apply returns typed error
        assert!(matches!(
            s.try_apply(0.0),
            Err(LensForgeError::NumericalError(_))
        ));
        assert!(matches!(
            s.try_apply(-1.0),
            Err(LensForgeError::NumericalError(_))
        ));
        // Non-finite input
        assert!(matches!(
            s.try_apply(f32::NAN),
            Err(LensForgeError::NumericalError(_))
        ));
    }

    #[test]
    fn log10_revert_overflow_error() {
        let s = RenderScale::Log10;
        // 10^s overflows for sufficiently large s (e.g., > 309 on f32)
        let big = 310.0;
        assert!(matches!(
            s.try_revert(big),
            Err(LensForgeError::NumericalError(_))
        ));
        // Fallback path
        let fb = 1.23;
        assert_eq!(s.revert_or(big, fb), fb);
    }

    #[test]
    fn symlog_threshold_validation() {
        let s0 = RenderScale::SymLog { linthresh: 0.0 };
        assert!(s0.apply(1.0).is_none());
        assert!(matches!(
            s0.try_apply(1.0),
            Err(LensForgeError::InvalidOption(_))
        ));
        assert!(matches!(
            s0.try_revert(1.0),
            Err(LensForgeError::InvalidOption(_))
        ));

        let snan = RenderScale::SymLog {
            linthresh: f32::NAN,
        };
        assert!(matches!(
            snan.try_apply(1.0),
            Err(LensForgeError::InvalidOption(_))
        ));
    }

    #[test]
    fn symlog_fallbacks_on_invalid_threshold() {
        let s = RenderScale::SymLog { linthresh: 0.0 };
        let fb = 7.7;
        assert_eq!(s.apply_or(10.0, fb), fb);
        assert_eq!(s.revert_or(10.0, fb), fb);
    }

    #[test]
    fn render_scale_non_finite_inputs_rejected() {
        let scales = [
            RenderScale::Linear,
            RenderScale::Log10,
            RenderScale::SymLog { linthresh: 2.0 },
        ];
        for &scale in &scales {
            for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(scale.apply(bad).is_none());
                assert!(matches!(
                    scale.try_apply(bad),
                    Err(LensForgeError::NumericalError(_))
                ));
                assert!(scale.revert(bad).is_none());
                assert!(matches!(
                    scale.try_revert(bad),
                    Err(LensForgeError::NumericalError(_))
                ));
            }
        }
    }

    #[test]
    fn slice_apply_linear() {
        let scale = RenderScale::Linear;
        let input = vec![1.0, 2.0, f32::NAN, 4.0, f32::INFINITY];
        let mut output = vec![0.0; 5];
        let fallback = -99.0;
        scale.apply_slice(&input, &mut output, fallback);

        assert!(approx(output[0], 1.0, 1e-10));
        assert!(approx(output[1], 2.0, 1e-10));
        assert!(approx(output[2], fallback, 1e-10));
        assert!(approx(output[3], 4.0, 1e-10));
        assert!(approx(output[4], fallback, 1e-10));
    }

    #[test]
    fn slice_apply_log10() {
        let scale = RenderScale::Log10;
        let input = vec![1.0, 10.0, 100.0, -1.0, 0.0];
        let mut output = vec![0.0; 5];
        let fallback = -99.0;

        scale.apply_slice(&input, &mut output, fallback);

        assert!(approx(output[0], 0.0, 1e-10));
        assert!(approx(output[1], 1.0, 1e-10));
        assert!(approx(output[2], 2.0, 1e-10));
        assert!(approx(output[3], fallback, 1e-10));
        assert!(approx(output[4], fallback, 1e-10));
    }

    #[test]
    fn slice_apply_symlog() {
        let scale = RenderScale::SymLog { linthresh: 2.0 };
        let input = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        let mut output = vec![0.0; 5];
        let fallback = -99.0;

        scale.apply_slice(&input, &mut output, fallback);

        // Values in linear region [-2, 2] should pass through
        assert!(approx(output[2], 0.0, 1e-10));
        assert!(approx(output[3], 1.0, 1e-10));

        // Check symmetry
        assert!(approx(output[0], -output[4], 1e-10));
    }

    #[test]
    fn slice_apply_inplace() {
        let scale = RenderScale::Log10;
        let mut values = vec![1.0, 10.0, 100.0, -1.0];
        let fallback = -99.0;

        scale.apply_slice_inplace(&mut values, fallback);

        assert!(approx(values[0], 0.0, 1e-10));
        assert!(approx(values[1], 1.0, 1e-10));
        assert!(approx(values[2], 2.0, 1e-10));
        assert!(approx(values[3], fallback, 1e-10));
    }

    #[test]
    fn slice_par_apply() {
        let scale = RenderScale::Linear;
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let mut output_seq = vec![0.0; 1000];
        let mut output_par = vec![0.0; 1000];
        let fallback = 0.0;

        scale.apply_slice(&input, &mut output_seq, fallback);
        scale.par_apply_slice(&input, &mut output_par, fallback);

        for i in 0..1000 {
            assert!(approx(output_seq[i], output_par[i], 1e-10));
        }
    }

    #[test]
    fn slice_consistency_with_per_value() {
        let scale = RenderScale::SymLog { linthresh: 2.0 };
        let input: Vec<f32> = vec![-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
        let mut output_slice = vec![0.0; input.len()];
        let fallback = 0.0;

        scale.apply_slice(&input, &mut output_slice, fallback);

        for (i, &v) in input.iter().enumerate() {
            let per_value = scale.apply_or(v, fallback);
            assert!(
                approx(output_slice[i], per_value, 1e-10),
                "Mismatch at index {}: slice={}, per_value={}",
                i,
                output_slice[i],
                per_value
            );
        }
    }

    #[test]
    #[should_panic(expected = "Input and output slices must have same length")]
    fn slice_apply_panics_on_length_mismatch() {
        let scale = RenderScale::Linear;
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 2];
        scale.apply_slice(&input, &mut output, 0.0);
    }

    /* ============ Normalize tests ======================================== */
    #[test]
    fn normalize_default_behavior() {
        let n = Normalize::default();
        // Default maps [0, 1] to [0, 1] with `clamp: true`
        assert!(approx(n.apply(0.0), 0.0, 1e-10));
        assert!(approx(n.apply(1.0), 1.0, 1e-10));
        // Values outside range get clamped
        assert!(approx(n.apply(-10.0), 0.0, 1e-10));
        assert!(approx(n.apply(10.0), 1.0, 1e-10));
    }

    #[test]
    fn normalize_with_lims_validation() {
        // Finite, valid range
        let n = Normalize::default().with_lims(10.0, 20.0).unwrap();
        assert!(approx(n.range(), 10.0, 1e-10));

        // Invalid: non-finite
        assert!(Normalize::default().with_lims(f32::NAN, 1.0).is_err());
        assert!(Normalize::default().with_lims(0.0, f32::INFINITY).is_err());

        // Invalid: max < min
        assert!(Normalize::default().with_lims(5.0, 5.0).is_ok());
        assert!(Normalize::default().with_lims(5.0, 4.0).is_err());
    }

    #[test]
    fn normalize_from_range_validation() {
        let n = Normalize::from_range(-2.0, 4.0).unwrap();
        assert!(approx(n.min, -2.0, 1e-10));
        assert!(approx(n.max, 2.0, 1e-10));
        assert!(approx(n.range(), 4.0, 1e-10));
    }

    #[test]
    fn normalize_apply_unclamped() {
        let n = Normalize::default()
            .with_clamp(false)
            .with_lims(-2.0, 2.0)
            .unwrap();
        // No clamping; values outside [-2, 2] map outside [0, 1]
        assert!(approx(n.apply(-3.0), -0.25, 1e-10));
        assert!(approx(n.apply(3.0), 1.25, 1e-10));
        assert!(approx(n.apply(0.0), 0.5, 1e-10));
    }

    #[test]
    fn normalize_eps_stability() {
        // Create a near-zero range smaller than eps
        let mut n = Normalize::default();
        n.eps = 1e-4;
        let n = n.with_lims(1.0, 1.0 + 1e-6).unwrap();
        // range_eps returns eps when |range| < eps
        assert!(approx(n.range_eps(), 1e-4, 1e-8));
        assert!(approx(n.range(), 1e-6, 1e-6));
        // Mapping uses eps in denominator
        let u = n.try_apply(1.0 + 1e-2).unwrap();
        assert!(approx(u, 1.0, 1e-6));
    }

    #[test]
    fn normalize_nan_input_returns_nan() {
        let n = Normalize::default().with_lims(10.0, 20.0).unwrap();
        let u = n.apply(f32::NAN);
        assert!(u.is_nan());
    }

    #[test]
    fn normalize_builder_chain() {
        // Builder chain with custom clamp and eps
        let n = Normalize::default()
            .with_clamp(false)
            .with_eps(1e-6)
            .with_lims(2.0, 6.0)
            .unwrap();
        assert!(!n.clamp);
        assert!(approx(n.eps, 1e-6, 0.0));
        assert!(approx(n.range(), 4.0, 1e-10));
        assert!(approx(n.apply(4.0), 0.5, 1e-10));
    }

    #[test]
    fn normalize_slice_apply() {
        let norm = Normalize::default().with_lims(0.0, 100.0).unwrap();
        let input = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        let mut output = vec![0.0; 5];

        norm.apply_slice(&input, &mut output);

        assert!(approx(output[0], 0.0, 1e-10));
        assert!(approx(output[1], 0.25, 1e-10));
        assert!(approx(output[2], 0.5, 1e-10));
        assert!(approx(output[3], 0.75, 1e-10));
        assert!(approx(output[4], 1.0, 1e-10));
    }

    #[test]
    fn normalize_slice_with_clamping() {
        let norm = Normalize::default()
            .with_clamp(true)
            .with_lims(0.0, 10.0)
            .unwrap();
        let input = vec![-5.0, 0.0, 5.0, 10.0, 15.0];
        let mut output = vec![0.0; 5];

        norm.apply_slice(&input, &mut output);

        assert!(approx(output[0], 0.0, 1e-10)); // clamped
        assert!(approx(output[1], 0.0, 1e-10));
        assert!(approx(output[2], 0.5, 1e-10));
        assert!(approx(output[3], 1.0, 1e-10));
        assert!(approx(output[4], 1.0, 1e-10)); // clamped
    }

    #[test]
    fn normalize_slice_without_clamping() {
        let norm = Normalize::default()
            .with_clamp(false)
            .with_lims(0.0, 10.0)
            .unwrap();
        let input = vec![-5.0, 0.0, 5.0, 10.0, 15.0];
        let mut output = vec![0.0; 5];

        norm.apply_slice(&input, &mut output);

        assert!(approx(output[0], -0.5, 1e-10)); // not clamped
        assert!(approx(output[1], 0.0, 1e-10));
        assert!(approx(output[2], 0.5, 1e-10));
        assert!(approx(output[3], 1.0, 1e-10));
        assert!(approx(output[4], 1.5, 1e-10)); // not clamped
    }

    #[test]
    fn normalize_slice_inplace() {
        let norm = Normalize::default().with_lims(0.0, 100.0).unwrap();
        let mut values = vec![0.0, 50.0, 100.0];

        norm.apply_slice_inplace(&mut values);

        assert!(approx(values[0], 0.0, 1e-10));
        assert!(approx(values[1], 0.5, 1e-10));
        assert!(approx(values[2], 1.0, 1e-10));
    }

    #[test]
    fn normalize_par_apply_slice() {
        let norm = Normalize::default().with_lims(0.0, 1000.0).unwrap();
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let mut output_seq = vec![0.0; 1000];
        let mut output_par = vec![0.0; 1000];

        norm.apply_slice(&input, &mut output_seq);
        norm.par_apply_slice(&input, &mut output_par);

        for i in 0..1000 {
            assert!(approx(output_seq[i], output_par[i], 1e-10));
        }
    }

    #[test]
    fn normalize_slice_consistency_with_per_value() {
        let norm = Normalize::default()
            .with_clamp(false)
            .with_lims(-10.0, 10.0)
            .unwrap();
        let input: Vec<f32> = vec![-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0];
        let mut output_slice = vec![0.0; input.len()];

        norm.apply_slice(&input, &mut output_slice);

        for (i, &v) in input.iter().enumerate() {
            let per_value = norm.apply(v);
            assert!(
                approx(output_slice[i], per_value, 1e-10),
                "Mismatch at index {}: slice={}, per_value={}",
                i,
                output_slice[i],
                per_value
            );
        }
    }

    #[test]
    fn normalize_slice_handles_nan() {
        let norm = Normalize::default().with_lims(0.0, 10.0).unwrap();
        let input = vec![0.0, f32::NAN, 5.0, f32::INFINITY, 10.0];
        let mut output = vec![0.0; 5];

        norm.apply_slice(&input, &mut output);

        assert!(approx(output[0], 0.0, 1e-10));
        assert!(output[1].is_nan());
        assert!(approx(output[2], 0.5, 1e-10));
        assert!(output[3].is_nan());
        assert!(approx(output[4], 1.0, 1e-10));
    }

    /* ============ Chained scaling tests ================================== */
    #[test]
    fn normalize_then_scale() {
        // Normalize an input to [0, 1], then scale using SymLog for display coordinates.
        let norm = Normalize::default()
            .with_clamp(false)
            .with_lims(0.0, 100.0)
            .unwrap();
        let scale = RenderScale::SymLog { linthresh: 0.2 };

        // Data points: low, mid, high
        for &v in &[0.0, 50.0, 100.0] {
            let u = scale.try_apply(v).unwrap();
            let x = norm.try_apply(u).unwrap();
            // Revert and renormalize back should give original u
            let u_rev = norm.try_revert(x).unwrap();
            let v_rev = scale.try_revert(u_rev).unwrap();
            assert!(approx(u, u_rev, 1e-6));
            assert!(approx(v, v_rev, 1e-6));
        }
    }
}
