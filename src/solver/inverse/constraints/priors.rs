// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Prior / regularization constraint families: model priors independent of the data.
//!
//! These produce the `(rows, rhs)` blocks the [`builder`](super::builder) composes — `κ ≥ 0`,
//! adjacent-cell gradient bounds, symmetry equalities, the mass-sheet pin, and radial
//! monotonicity. All are affine in `x = (m, {β_s})`.

use crate::grid::Point2D;
use crate::solver::basis::KappaBasis;

use super::builder::Symmetry;

/// Non-negative convergence bounds: `−m_c ≤ 0` per cell.
///
/// # Arguments
/// * `n_cells` - number of κ cells.
/// * `n` - number of unknowns (row length).
pub(super) fn nonneg_kappa_bounds(n_cells: usize, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rows = Vec::with_capacity(n_cells);
    for c in 0..n_cells {
        let mut row = vec![0.0; n];
        row[c] = -1.0;
        rows.push(row);
    }
    (rows, vec![0.0; n_cells])
}

/// Adjacent-cell gradient bounds: `|m_i/A_i − m_j/A_j| ≤ g · d_ij`, a ± pair per unique edge.
///
/// # Arguments
/// * `basis` - the convergence basis (supplies cell centres and adjacency).
/// * `areas` - cell areas (length `n_cells`).
/// * `g` - the gradient bound.
/// * `n` - number of unknowns (row length).
pub(super) fn gradient_bounds<B: KappaBasis>(
    basis: &B,
    areas: &[f64],
    g: f64,
    n: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_cells = areas.len();
    let mut rows = Vec::new();
    let mut rhs = Vec::new();
    for i in 0..n_cells {
        let ci = basis.center(i);
        for j in basis.cell_neighbors(i) {
            if j <= i {
                continue;
            }
            let d = (basis.center(j) - ci).norm();
            let mut row = vec![0.0; n];
            row[i] = 1.0 / areas[i];
            row[j] = -1.0 / areas[j];
            rows.push(row.clone());
            rhs.push(g * d);
            for v in row.iter_mut() {
                *v = -*v;
            }
            rows.push(row);
            rhs.push(g * d);
        }
    }
    (rows, rhs)
}

/// Symmetry equalities `κ_i = κ_j` for each pair of cells mapped onto each other by `sym`.
///
/// Cells are paired by nearest centre to the symmetry image; only mutually nearest pairs are
/// emitted (each once, `i < j`), so on-axis cells and an imperfectly symmetric basis simply
/// contribute no spurious constraint. `O(n_cells²)` — assembly is a one-off.
///
/// # Arguments
/// * `basis` - the convergence basis (supplies cell centres).
/// * `sym` - the required symmetry.
/// * `areas` - cell areas (length `n_cells`).
/// * `n` - number of unknowns (row length).
pub(super) fn symmetry_equalities<B: KappaBasis>(
    basis: &B,
    sym: Symmetry,
    areas: &[f64],
    n: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_cells = areas.len();
    let centers: Vec<Point2D> = (0..n_cells).map(|c| basis.center(c)).collect();
    // The cell whose centre is nearest the symmetry image of cell `i`'s centre.
    let partner = |i: usize| -> usize {
        let image = sym.map(centers[i]);
        (0..n_cells)
            .min_by(|&a, &b| {
                (centers[a] - image)
                    .norm_squared()
                    .total_cmp(&(centers[b] - image).norm_squared())
            })
            .unwrap()
    };
    let mut rows = Vec::new();
    let mut rhs = Vec::new();
    for i in 0..n_cells {
        let j = partner(i);
        // One equality per mutually-nearest pair; skip self-pairs (on-axis cells).
        if j > i && partner(j) == i {
            let mut row = vec![0.0; n];
            row[i] = 1.0 / areas[i];
            row[j] = -1.0 / areas[j];
            rows.push(row);
            rhs.push(0.0);
        }
    }
    (rows, rhs)
}

/// Mass-sheet pin: a single equality fixing the mean convergence `(1/C)·Σ_c κ_c = mean`.
///
/// # Arguments
/// * `mean` - the target mean convergence.
/// * `areas` - cell areas (length `n_cells`).
/// * `n` - number of unknowns (row length).
pub(super) fn mean_kappa_equality(mean: f64, areas: &[f64], n: usize) -> (Vec<f64>, f64) {
    let n_cells = areas.len();
    let mut row = vec![0.0; n];
    for c in 0..n_cells {
        row[c] = 1.0 / areas[c];
    }
    (row, mean * n_cells as f64)
}

/// Radial-monotonicity bounds about `center`: `κ_outer − κ_inner ≤ 0` on each radial
/// adjacency edge (the farther cell is the outer one). Near-equidistant edges are skipped.
///
/// # Arguments
/// * `basis` - the convergence basis (supplies cell centres and adjacency).
/// * `center` - the centre radii are measured from.
/// * `areas` - cell areas (length `n_cells`).
/// * `n` - number of unknowns (row length).
pub(super) fn radial_monotonic_bounds<B: KappaBasis>(
    basis: &B,
    center: Point2D,
    areas: &[f64],
    n: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_cells = areas.len();
    let mut rows = Vec::new();
    let mut rhs = Vec::new();
    for i in 0..n_cells {
        let ri = (basis.center(i) - center).norm();
        for j in basis.cell_neighbors(i) {
            if j <= i {
                continue; // each edge once
            }
            let rj = (basis.center(j) - center).norm();
            // Skip azimuthal (equidistant) edges — monotonicity is a radial statement only.
            if (ri - rj).abs() <= 1e-9 {
                continue;
            }
            let (outer, inner) = if ri > rj { (i, j) } else { (j, i) };
            let mut row = vec![0.0; n];
            row[outer] = 1.0 / areas[outer];
            row[inner] = -1.0 / areas[inner];
            rows.push(row);
            rhs.push(0.0);
        }
    }
    (rows, rhs)
}

/// Curvature bounds: `|Σ_{j∈N(i)} (κ_i − κ_j)| ≤ l` per cell (the discrete Laplacian), a ±
/// pair per cell.
///
/// Caps the peakiness — the substructure amplitude — of the convergence field: a small `l`
/// forces a smooth halo, a large `l` admits localized clumps. Reuses the same graph-Laplacian
/// stencil as the source-plane smoothness regularizer, here as a hard constraint on κ.
///
/// # Arguments
/// * `basis` - the convergence basis (supplies cell adjacency).
/// * `areas` - cell areas (length `n_cells`).
/// * `l` - the curvature bound.
/// * `n` - number of unknowns (row length).
pub(super) fn curvature_bounds<B: KappaBasis>(
    basis: &B,
    areas: &[f64],
    l: f64,
    n: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_cells = areas.len();
    let mut rows = Vec::new();
    let mut rhs = Vec::new();
    for i in 0..n_cells {
        let neighbors = basis.cell_neighbors(i);
        if neighbors.is_empty() {
            continue;
        }
        // (Δκ)_i = Σ_j (κ_i − κ_j): degree on the cell, −1 on each neighbour (in κ = m/A).
        let mut row = vec![0.0; n];
        row[i] = neighbors.len() as f64 / areas[i];
        for &j in &neighbors {
            row[j] -= 1.0 / areas[j];
        }
        rows.push(row.clone());
        rhs.push(l);
        for v in row.iter_mut() {
            *v = -*v;
        }
        rows.push(row);
        rhs.push(l);
    }
    (rows, rhs)
}
