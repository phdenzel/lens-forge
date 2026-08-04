// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Adaptive Simpson quadrature for smooth 1D integrands.

/// Recursive adaptive Simpson step: refine until the split matches `whole` within `eps`.
fn adaptive_simpson<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    eps: f64,
    depth: u32,
) -> f64 {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = f(lm);
    let frm = f(rm);
    let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    if depth == 0 || (left + right - whole).abs() <= 15.0 * eps {
        return left + right + (left + right - whole) / 15.0;
    }
    adaptive_simpson(f, a, m, fa, fm, flm, left, 0.5 * eps, depth - 1)
        + adaptive_simpson(f, m, b, fm, fb, frm, right, 0.5 * eps, depth - 1)
}

/// Integrate `f` over `[a, b]` by adaptive Simpson (`0` for an empty interval).
///
/// # Arguments
/// * `f` - integrand.
/// * `a` - lower limit.
/// * `b` - upper limit.
pub(crate) fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    if b <= a {
        return 0.0;
    }
    let m = 0.5 * (a + b);
    let (fa, fb, fm) = (f(a), f(b), f(m));
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    adaptive_simpson(&f, a, b, fa, fb, fm, whole, 1e-11, 50)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_polynomial_exactly() {
        // ∫₀² 3x² dx = 8; Simpson is exact for cubics.
        assert!((integrate(|x| 3.0 * x * x, 0.0, 2.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn integrates_smooth_transcendental() {
        // ∫₀^π sin x dx = 2.
        assert!((integrate(|x| x.sin(), 0.0, std::f64::consts::PI) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn empty_interval_is_zero() {
        assert_eq!(integrate(|x| x, 1.0, 1.0), 0.0);
        assert_eq!(integrate(|x| x, 2.0, 1.0), 0.0);
    }
}
