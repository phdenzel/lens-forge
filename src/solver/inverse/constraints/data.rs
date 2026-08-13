// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Point-image (data) constraint families: observed images turned into affine rows.
//!
//! Per source, these produce the `(rows, rhs)` blocks the [`builder`](super::builder)
//! composes — the lens-equation position equalities and the Fermat time-delay /
//! arrival-order relations. All are affine in `x = (m, {β_s})`.

use crate::grid::Point2D;
use crate::solver::basis::FieldCoefficients;
use crate::solver::image::PointImageSystem;

/// Position equalities for one source: `α(θ_i) + β_s = θ_i`, an x and a y row per image.
///
/// # Arguments
/// * `infl` - the per-cell field coefficients at this source's images.
/// * `thetas` - the image positions.
/// * `areas` - cell areas (length `n_cells`, converting the mass column to convergence).
/// * `n` - number of unknowns (row length).
/// * `bx` - the unknown index of this source's β_x column.
/// * `by` - the unknown index of this source's β_y column.
pub(super) fn position_equalities(
    infl: &FieldCoefficients,
    thetas: &[Point2D],
    areas: &[f64],
    n: usize,
    bx: usize,
    by: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_cells = areas.len();
    let mut rows = Vec::with_capacity(2 * thetas.len());
    let mut rhs = Vec::with_capacity(2 * thetas.len());
    for i in 0..thetas.len() {
        let (ax, ay) = infl.alpha_rows(i);
        let mut rx = vec![0.0; n];
        let mut ry = vec![0.0; n];
        for c in 0..n_cells {
            rx[c] = ax[c] / areas[c];
            ry[c] = ay[c] / areas[c];
        }
        rx[bx] = 1.0;
        ry[by] = 1.0;
        rows.push(rx);
        rhs.push(thetas[i].x);
        rows.push(ry);
        rhs.push(thetas[i].y);
    }
    (rows, rhs)
}

/// Position inequalities for one source with slack `eps`: `|α(θ_i) + β_s − θ_i| ≤ eps` per
/// coordinate — two ± inequality rows per coordinate (four per image). The soft counterpart of
/// [`position_equalities`], for extended-feature correspondences with centroiding error.
///
/// # Arguments
/// * `infl` - the per-cell field coefficients at this source's images.
/// * `thetas` - the image positions.
/// * `areas` - cell areas (length `n_cells`).
/// * `n` - number of unknowns (row length).
/// * `bx` - the unknown index of this source's β_x column.
/// * `by` - the unknown index of this source's β_y column.
/// * `eps` - the per-coordinate positional slack.
pub(super) fn position_inequalities(
    infl: &FieldCoefficients,
    thetas: &[Point2D],
    areas: &[f64],
    n: usize,
    bx: usize,
    by: usize,
    eps: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_cells = areas.len();
    let mut rows = Vec::with_capacity(4 * thetas.len());
    let mut rhs = Vec::with_capacity(4 * thetas.len());
    for i in 0..thetas.len() {
        let (ax, ay) = infl.alpha_rows(i);
        let mut rx = vec![0.0; n];
        let mut ry = vec![0.0; n];
        for c in 0..n_cells {
            rx[c] = ax[c] / areas[c];
            ry[c] = ay[c] / areas[c];
        }
        rx[bx] = 1.0;
        ry[by] = 1.0;
        push_band(&mut rows, &mut rhs, rx, thetas[i].x, eps);
        push_band(&mut rows, &mut rhs, ry, thetas[i].y, eps);
    }
    (rows, rhs)
}

/// Push `|row·x − target| ≤ eps` as a ± pair: `row·x ≤ target + eps` and
/// `−row·x ≤ eps − target`.
fn push_band(rows: &mut Vec<Vec<f64>>, rhs: &mut Vec<f64>, row: Vec<f64>, target: f64, eps: f64) {
    let neg: Vec<f64> = row.iter().map(|v| -v).collect();
    rows.push(row);
    rhs.push(target + eps);
    rows.push(neg);
    rhs.push(eps - target);
}

/// The Fermat difference `τ_j − τ_i` as an affine form in `x` plus the constant carried to
/// the right-hand side (the β² term cancels in the difference).
///
/// # Arguments
/// * `infl` - the per-cell field coefficients at this source's images.
/// * `thetas` - the image positions.
/// * `areas` - cell areas (length `n_cells`).
/// * `n` - number of unknowns (row length).
/// * `bx` - the unknown index of this source's β_x column.
/// * `by` - the unknown index of this source's β_y column.
/// * `i` - the reference image index.
/// * `j` - the delayed image index.
fn fermat_form(
    infl: &FieldCoefficients,
    thetas: &[Point2D],
    areas: &[f64],
    n: usize,
    bx: usize,
    by: usize,
    i: usize,
    j: usize,
) -> (Vec<f64>, f64) {
    let n_cells = areas.len();
    let mut row = vec![0.0; n];
    let pi = infl.psi_row(i);
    let pj = infl.psi_row(j);
    for c in 0..n_cells {
        row[c] = -(pj[c] - pi[c]) / areas[c];
    }
    let (ti, tj) = (thetas[i], thetas[j]);
    row[bx] = -(tj.x - ti.x);
    row[by] = -(tj.y - ti.y);
    let constant = 0.5 * (tj.norm_squared() - ti.norm_squared());
    (row, constant)
}

/// Time-delay equalities: one Fermat equality `τ_to − τ_from = Δt / k` per measured delay.
///
/// # Arguments
/// * `sys` - the source's point-image system (supplies the delays and their scale).
/// * `infl` - the per-cell field coefficients at this source's images.
/// * `thetas` - the image positions.
/// * `areas` - cell areas (length `n_cells`).
/// * `n` - number of unknowns (row length).
/// * `bx` - the unknown index of this source's β_x column.
/// * `by` - the unknown index of this source's β_y column.
pub(super) fn time_delay_equalities(
    sys: &PointImageSystem,
    infl: &FieldCoefficients,
    thetas: &[Point2D],
    areas: &[f64],
    n: usize,
    bx: usize,
    by: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rows = Vec::with_capacity(sys.time_delays.len());
    let mut rhs = Vec::with_capacity(sys.time_delays.len());
    for td in &sys.time_delays {
        let (row, constant) = fermat_form(infl, thetas, areas, n, bx, by, td.from, td.to);
        rows.push(row);
        rhs.push(td.delay / sys.delay_scale - constant);
    }
    (rows, rhs)
}

/// Arrival-order inequalities: consecutive `τ_earlier − τ_later ≤ 0`.
///
/// # Arguments
/// * `order` - the arrival order as image indices, earliest first.
/// * `infl` - the per-cell field coefficients at this source's images.
/// * `thetas` - the image positions.
/// * `areas` - cell areas (length `n_cells`).
/// * `n` - number of unknowns (row length).
/// * `bx` - the unknown index of this source's β_x column.
/// * `by` - the unknown index of this source's β_y column.
pub(super) fn arrival_order_inequalities(
    order: &[usize],
    infl: &FieldCoefficients,
    thetas: &[Point2D],
    areas: &[f64],
    n: usize,
    bx: usize,
    by: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rows = Vec::with_capacity(order.len().saturating_sub(1));
    let mut rhs = Vec::with_capacity(order.len().saturating_sub(1));
    for w in order.windows(2) {
        let (row, constant) = fermat_form(infl, thetas, areas, n, bx, by, w[1], w[0]);
        rows.push(row);
        rhs.push(-constant);
    }
    (rows, rhs)
}
