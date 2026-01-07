// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Polygonal mesh adapters for rendering non-uniform grids.
use crate::viz::render::RenderBounds;
use crate::{LensForgeError, Result};

/// A single cell for cell-based mesh rendering.
#[derive(Clone, Debug)]
pub struct Cell {
    /// Polygon vertex coordinates
    pub vertices: Vec<[f64; 2]>,
    /// Scalar value associated with the cell
    pub value: f64,
}

impl Cell {
    pub fn is_valid(&self) -> bool {
        if self.vertices.len() < 3 {
            return false;
        }
        if self
            .vertices
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return false;
        }
        // check whether cell is non-degenerate
        self.area().abs() > 0.0
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.vertices.len();
        if n < 3 {
            return Err(LensForgeError::dimension_mismatch(
                "3+ vertices".to_string(),
                format!("{} vertices", n),
            ));
        }
        if self
            .vertices
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return Err(LensForgeError::numerical_error(
                "Cell has non-finite vertex coodinates",
            ));
        }
        let a = self.area().abs();
        if !(a > 0.0) {
            return Err(LensForgeError::numerical_error(
                "Cell is degenerate (has zero area)",
            ));
        }
        Ok(())
    }

    /// Signed polygon area (2D) using Gauss's formula.
    ///
    /// For clockwise (cw) < 0 , for counter-clockwise (ccw) > 0, and
    /// zero if colinear or degenerate.
    pub fn area(&self) -> f64 {
        let poly = &self.vertices;
        let n = poly.len();
        if n < 3 {
            return 0.0;
        }
        let mut sum: f64 = 0.0;
        for i in 0..n {
            let i_next = (i + 1) % n;
            let (x0, y0) = (poly[i][0], poly[i][1]);
            let (x1, y1) = (poly[i_next][0], poly[i_next][1]);
            sum += x0 * y1 - x1 * y0;
        }
        0.5 * sum
    }
}

/// A polygonal mesh view.
pub trait ToMesh {
    fn cells(&self) -> Result<Vec<Cell>>;
}

/// A polygonal mesh interface storing cells and (optionally) explicit bounds.
#[derive(Clone, Debug, Default)]
pub struct MeshField {
    /// Cells to render
    pub cells: Vec<Cell>,
    /// Optional coordinate bounds
    pub bounds: Option<RenderBounds>,
}

impl MeshField {
    /// Construct from cells without explicit bounds.
    pub fn from_cells(cells: Vec<Cell>) -> Self {
        Self {
            cells,
            bounds: None,
        }
    }

    /// Builder for setting `bounds` field.
    pub fn with_bounds(mut self, bounds: RenderBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        if let Some(b) = self.bounds {
            if !b.is_valid() {
                return false;
            }
        }
        self.cells.iter().all(|c| c.is_valid())
    }

    pub fn validate(&self) -> Result<()> {
        if let Some(b) = self.bounds {
            if !b.is_valid() {
                return Err(LensForgeError::invalid_option(
                    "Mesh bounds must be finite and non-degenerate",
                ));
            }
        }
        for (i, c) in self.cells.iter().enumerate() {
            c.validate()
                .map_err(|e| LensForgeError::invalid_parameter(format!("Cell #{i}: {e}")))?;
        }
        Ok(())
    }

    /// Compute tight axis-aligned bounds over all cells.
    pub fn compute_bounds(&self) -> Option<RenderBounds> {
        let mut xmin = f64::INFINITY;
        let mut ymin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymax = f64::NEG_INFINITY;

        if self.cells.is_empty() {
            return None;
        }

        for cell in &self.cells {
            for &p in &cell.vertices {
                let (x, y) = (p[0], p[1]);
                if !x.is_finite() || !y.is_finite() {
                    return None;
                }
                xmin = xmin.min(x);
                ymin = ymin.min(y);
                xmax = xmax.max(x);
                ymax = ymax.max(y);
            }
        }

        let bounds = RenderBounds {
            min: [xmin, ymin],
            max: [xmax, ymax],
        };
        if bounds.is_valid() {
            Some(bounds)
        } else {
            None
        }
    }
}

impl ToMesh for MeshField {
    fn cells(&self) -> Result<Vec<Cell>> {
        // Return a shallow clone of all cells (vertices are owned in each cell).
        self.validate()?;
        Ok(self.cells.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(a: [f64; 2], b: [f64; 2], c: [f64; 2], value: f64) -> Cell {
        Cell {
            vertices: vec![a, b, c],
            value,
        }
    }

    #[test]
    fn area_ccw_positive() {
        let c = tri([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], 0.5);
        assert!(c.area() > 0.0);
    }

    #[test]
    fn area_cw_negative() {
        let c = tri([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], 0.5);
        assert!(c.area() < 0.0);
    }

    #[test]
    fn mesh_validate_ok_and_compute_bounds() {
        let cells = vec![
            tri([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], 0.5),
            tri([1.0, 1.0], [2.0, 1.0], [1.0, 2.0], 1.5),
        ];
        let mf = MeshField::from_cells(cells);
        mf.validate().unwrap();

        let b = mf.compute_bounds().unwrap();
        assert!(b.is_valid());
        assert_eq!(b.min, [0.0, 0.0]);
        assert_eq!(b.max, [2.0, 2.0]);
    }

    #[test]
    fn mesh_validate_fails_on_degenerate() {
        // Colinear triangle (area = 0)
        let cells = vec![tri([0.0, 0.0], [1.0, 1.0], [2.0, 2.0], 0.0)];
        let mf = MeshField::from_cells(cells);
        assert!(mf.validate().is_err());
        assert!(!mf.is_valid());
    }

    #[test]
    fn mesh_validate_fails_on_non_finite() {
        let cells = vec![Cell {
            vertices: vec![[f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]],
            value: 0.0,
        }];
        let mf = MeshField::from_cells(cells);
        assert!(mf.validate().is_err());
        assert!(mf.compute_bounds().is_none());
    }

    #[test]
    fn to_mesh_returns_cells_after_validation() {
        let cells = vec![tri([0.0, 0.0], [2.0, 0.0], [0.0, 2.0], 7.0)];
        let mf = MeshField::from_cells(cells.clone());
        let out = mf.cells().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].vertices.len(), 3);
        assert_eq!(out[0].value, 7.0);
    }
}
