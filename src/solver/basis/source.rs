// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The source-plane representation seam a semi-linear inversion solves on.
//!
//! [`SourceBasis`] is to the source plane what [`KappaBasis`](super::KappaBasis) is to the lens
//! plane: the single representation-specific hook that lets a regular [`PixelGrid`] or Voronoi
//! [`MeshGrid`] carry the reconstructed source. It builds a zero-initialized field over the
//! back-projected generators and — via [`CellAdjacency`] — feeds the smoothness regularizer.

use crate::grid::{
    BBox, CellAdjacency, FieldArray1D, FieldOps, Grid2D, MeshGrid, PixelGrid, Point2D,
};
use crate::Result;

/// Parameters to build a per-member source plane from its back-projected generators.
#[derive(Debug, Clone, Copy)]
pub struct SourceConfig {
    /// Fractional padding added around the generators' bounding box.
    pub pad: f64,
    /// Per-axis pixel count for a pixel source; `None` picks `≈√(#generators)` so the pixel
    /// count matches the mesh cell count. Ignored by a mesh source.
    pub resolution: Option<usize>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self { pad: 0.1, resolution: None }
    }
}

/// A source-plane field that a semi-linear inversion can solve for: it builds itself from
/// back-projected generators and exposes cell adjacency for the regularizer.
pub trait SourceBasis: FieldOps + CellAdjacency + Sized {
    /// A zero-initialized source field over `sites`, bounded by `bbox`.
    ///
    /// # Arguments
    /// * `sites` - the back-projected generators (Voronoi sites / pixel-count hint).
    /// * `bbox` - the padded source-plane bounding box.
    /// * `cfg` - the build config (a pixel source reads `resolution`; a mesh ignores it).
    ///
    /// # Errors
    /// Propagates a construction failure (e.g. too few distinct generators).
    fn from_generators(sites: &[Point2D], bbox: BBox, cfg: &SourceConfig) -> Result<Self>;

    /// This geometry carrying the solved surface-brightness coefficients `s`.
    ///
    /// # Arguments
    /// * `s` - one coefficient per cell (in cell-index order), length [`len`](crate::grid::ScalarField2D::len).
    ///
    /// # Errors
    /// Fails on a length mismatch against this field's cell count.
    fn with_coeffs(&self, s: &[f64]) -> Result<Self>;
}

impl SourceBasis for MeshGrid {
    fn from_generators(sites: &[Point2D], bbox: BBox, _cfg: &SourceConfig) -> Result<Self> {
        MeshGrid::zeros(sites, bbox)
    }

    fn with_coeffs(&self, s: &[f64]) -> Result<Self> {
        self.with_values(FieldArray1D::from_vec(s.to_vec()))
    }
}

impl SourceBasis for PixelGrid {
    fn from_generators(sites: &[Point2D], bbox: BBox, cfg: &SourceConfig) -> Result<Self> {
        let n = cfg
            .resolution
            .unwrap_or_else(|| (sites.len() as f64).sqrt().round() as usize)
            .max(2);
        Ok(PixelGrid::zeros(Grid2D::new(n, n, bbox)?))
    }

    fn with_coeffs(&self, s: &[f64]) -> Result<Self> {
        PixelGrid::from_vec(self.grid, s.to_vec())
    }
}
