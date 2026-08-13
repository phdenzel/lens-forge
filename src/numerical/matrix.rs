// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Dense-matrix assembly helpers shared across the crate.

use ndarray::Array2;

/// Stack `rows` (each of length `ncols`) into a row-major `rows.len() × ncols` matrix.
///
/// # Arguments
/// * `rows` - the row vectors, each of length `ncols`.
/// * `ncols` - the number of columns (the length of every row).
pub(crate) fn stack_rows<T>(rows: Vec<Vec<T>>, ncols: usize) -> Array2<T> {
    let nrows = rows.len();
    let flat: Vec<T> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((nrows, ncols), flat).expect("every row has ncols elements")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacks_rows_row_major() {
        let m = stack_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], 3);
        assert_eq!(m.shape(), &[2, 3]);
        assert_eq!(m[[0, 2]], 3.0);
        assert_eq!(m[[1, 0]], 4.0);
    }

    #[test]
    fn empty_rows_make_an_empty_matrix() {
        let m: Array2<f64> = stack_rows(Vec::new(), 4);
        assert_eq!(m.shape(), &[0, 4]);
    }

    #[test]
    #[should_panic]
    fn mismatched_row_length_panics() {
        let _ = stack_rows(vec![vec![1.0, 2.0], vec![3.0]], 2);
    }
}
