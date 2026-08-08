// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Binomial coefficients as a Pascal's triangle table.

/// Pascal's triangle up to row `max`: `table[n][k] = C(n, k)`, `0 ≤ k ≤ n`.
///
/// # Arguments
/// * `max` - highest row index retained.
pub(crate) fn pascal(max: usize) -> Vec<Vec<f64>> {
    let mut table: Vec<Vec<f64>> = Vec::with_capacity(max + 1);
    for n in 0..=max {
        let mut row = vec![1.0; n + 1];
        for k in 1..n {
            row[k] = table[n - 1][k - 1] + table[n - 1][k];
        }
        table.push(row);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pascal_rows_are_binomials() {
        let t = pascal(5);
        assert_eq!(t[0], vec![1.0]);
        assert_eq!(t[4], vec![1.0, 4.0, 6.0, 4.0, 1.0]);
        assert_eq!(t[5], vec![1.0, 5.0, 10.0, 10.0, 5.0, 1.0]);
    }
}
