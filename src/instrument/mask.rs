// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The pixel mask / weight map: a [`Mask`] carries a per-cell weight `w_i ≥ 0`
//! in image-cell order — binary (in/out) or soft. It selects regions, such as
//! arcs, excluding sky, the lens-galaxy light, and bad pixels, and composes
//! multiplicatively with noise.

/// A per-cell fit weight `w_i ≥ 0` (typically `0`/`1`), in image-cell order.
#[derive(Debug, Clone, PartialEq)]
pub struct Mask(Vec<f64>);

impl Mask {
    /// A mask that includes every one of `n` cells (all weights `1`).
    ///
    /// # Arguments
    /// * `n` - number of cells.
    pub fn all(n: usize) -> Self {
        Mask(vec![1.0; n])
    }

    /// A binary mask from include/exclude flags (`true` → `1`, `false` → `0`).
    ///
    /// # Arguments
    /// * `flags` - per-cell inclusion flags.
    pub fn from_flags(flags: &[bool]) -> Self {
        Mask(flags.iter().map(|&f| if f { 1.0 } else { 0.0 }).collect())
    }

    /// A soft mask from explicit per-cell weights (`≥ 0`).
    ///
    /// # Arguments
    /// * `weights` - per-cell weights.
    pub fn from_weights(weights: Vec<f64>) -> Self {
        Mask(weights)
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the mask has no cells.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The per-cell weights.
    pub fn weights(&self) -> &[f64] {
        &self.0
    }

    /// The weight `w_i` of cell `i`.
    ///
    /// # Arguments
    /// * `i` - cell index.
    pub fn weight(&self, i: usize) -> f64 {
        self.0[i]
    }

    /// The number of active cells (weight `> 0`) — the size of the fitted region.
    pub fn n_active(&self) -> usize {
        self.0.iter().filter(|&&w| w > 0.0).count()
    }

    /// Weight `values` by the mask: the element-wise product `w_i · values_i`
    /// (masked cells become `0`).
    ///
    /// # Arguments
    /// * `values` - a cell vector the same length as the mask.
    pub fn apply(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(values.len(), self.0.len(), "vector length must match the mask");
        self.0.iter().zip(values).map(|(&w, &v)| w * v).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_includes_every_cell() {
        let mask = Mask::all(4);
        assert_eq!(mask.len(), 4);
        assert!(!mask.is_empty());
        assert_eq!(mask.n_active(), 4);
        assert_eq!(mask.apply(&[1.0, 2.0, 3.0, 4.0]), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn flags_zero_the_excluded_cells() {
        let mask = Mask::from_flags(&[true, false, true]);
        assert_eq!(mask.weights(), &[1.0, 0.0, 1.0]);
        assert_eq!(mask.weight(1), 0.0);
        assert_eq!(mask.n_active(), 2);
        assert_eq!(mask.apply(&[5.0, 6.0, 7.0]), vec![5.0, 0.0, 7.0]);
    }

    #[test]
    fn soft_weights_count_only_the_positive_cells() {
        let mask = Mask::from_weights(vec![0.5, 0.0, 1.0]);
        assert_eq!(mask.n_active(), 2);
        assert_eq!(mask.apply(&[2.0, 9.0, 4.0]), vec![1.0, 0.0, 4.0]);
    }

    #[test]
    fn empty_mask_is_empty() {
        let mask = Mask::all(0);
        assert!(mask.is_empty());
        assert_eq!(mask.n_active(), 0);
    }

    #[test]
    #[should_panic(expected = "match the mask")]
    fn apply_rejects_a_length_mismatch() {
        Mask::all(3).apply(&[1.0, 2.0]);
    }
}
