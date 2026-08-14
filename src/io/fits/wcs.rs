// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! A linear-WCS bridge between FITS pixel/world coordinates and [`Grid2D`].
//!
//! The affine map `world_i = CRVAL_i + Σ_j M_ij (pixel_j − CRPIX_j)` is read
//! from `CRPIX/CRVAL/CDELT` (or the `CD`/`PC` matrix), with `CTYPE`/`CUNIT`
//! carried for context. [`LinearWcs::to_grid2d`] builds a grid whose physical
//! extent is `NAXIS_i · |CDELT_i|` (converting angular `CUNIT` to arcsec);
//! [`LinearWcs::from_grid2d`] emits the inverse cards. Sky projections and SIP
//! distortions are out of scope — only the linear terms drive the bridge.

use super::card::{Card, CardValue};
use super::error::FitsError;
use super::header::Header;
use crate::grid::{BBox, Grid2D};

/// A linear world-coordinate system for the first two image axes.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearWcs {
    /// Reference pixel `[CRPIX1, CRPIX2]` (1-based).
    pub crpix: [f64; 2],
    /// World value at the reference pixel `[CRVAL1, CRVAL2]`.
    pub crval: [f64; 2],
    /// The linear transform matrix `M` (rows are world axes).
    pub m: [[f64; 2]; 2],
    /// Coordinate types `[CTYPE1, CTYPE2]`.
    pub ctype: [String; 2],
    /// Coordinate units `[CUNIT1, CUNIT2]`.
    pub cunit: [String; 2],
}

impl Default for LinearWcs {
    fn default() -> Self {
        Self {
            crpix: [1.0, 1.0],
            crval: [0.0, 0.0],
            m: [[1.0, 0.0], [0.0, 1.0]],
            ctype: ["LINEAR".into(), "LINEAR".into()],
            cunit: ["".into(), "".into()],
        }
    }
}

/// The scale from an angular `CUNIT` to arcseconds (1 for linear/blank units).
///
/// # Arguments
/// * `cunit` - the unit string (case-insensitive).
pub fn cunit_scale(cunit: &str) -> f64 {
    match cunit.trim().to_ascii_lowercase().as_str() {
        "deg" | "degree" | "degrees" => 3600.0,
        "arcmin" | "arcmins" => 60.0,
        "rad" | "radian" | "radians" => 180.0 / std::f64::consts::PI * 3600.0,
        "mas" => 1e-3,
        _ => 1.0, // arcsec, pixel, or blank
    }
}

impl LinearWcs {
    /// Read a linear WCS from a header, or `None` if no WCS keywords are present.
    ///
    /// # Arguments
    /// * `header` - the parsed header.
    pub fn from_header(header: &Header) -> Option<Self> {
        let present = ["CRPIX1", "CDELT1", "CD1_1", "PC1_1", "CTYPE1"]
            .iter()
            .any(|k| header.get(k).is_some());
        if !present {
            return None;
        }
        let mut wcs = LinearWcs::default();
        for (i, axis) in [1usize, 2].into_iter().enumerate() {
            if let Some(v) = header.get_float(&format!("CRPIX{}", axis)) {
                wcs.crpix[i] = v;
            }
            if let Some(v) = header.get_float(&format!("CRVAL{}", axis)) {
                wcs.crval[i] = v;
            }
            if let Some(s) = header.get_str(&format!("CTYPE{}", axis)) {
                wcs.ctype[i] = s.to_string();
            }
            if let Some(s) = header.get_str(&format!("CUNIT{}", axis)) {
                wcs.cunit[i] = s.to_string();
            }
        }
        // Prefer an explicit CD matrix; else PC·CDELT; else diagonal CDELT.
        let cd = |r: usize, c: usize| header.get_float(&format!("CD{}_{}", r + 1, c + 1));
        if cd(0, 0).is_some() || cd(0, 1).is_some() || cd(1, 0).is_some() || cd(1, 1).is_some() {
            wcs.m = [
                [cd(0, 0).unwrap_or(0.0), cd(0, 1).unwrap_or(0.0)],
                [cd(1, 0).unwrap_or(0.0), cd(1, 1).unwrap_or(0.0)],
            ];
        } else {
            let cdelt1 = header.get_float("CDELT1").unwrap_or(1.0);
            let cdelt2 = header.get_float("CDELT2").unwrap_or(1.0);
            let pc = |r: usize, c: usize| {
                header
                    .get_float(&format!("PC{}_{}", r + 1, c + 1))
                    .unwrap_or(if r == c { 1.0 } else { 0.0 })
            };
            wcs.m = [
                [cdelt1 * pc(0, 0), cdelt1 * pc(0, 1)],
                [cdelt2 * pc(1, 0), cdelt2 * pc(1, 1)],
            ];
        }
        Some(wcs)
    }

    /// A pixel-coordinate WCS (unit spacing, origin at the first pixel corner).
    ///
    /// Used as the fallback geometry when a header carries no WCS.
    pub fn pixel_default() -> Self {
        LinearWcs::default()
    }

    /// Per-axis pixel scale (column norm of `M`), in the header's `CUNIT`.
    fn pixel_scales(&self) -> [f64; 2] {
        [
            self.m[0][0].hypot(self.m[1][0]),
            self.m[0][1].hypot(self.m[1][1]),
        ]
    }

    /// Build a [`Grid2D`] of `nx × ny` pixels from this WCS.
    ///
    /// The physical extent is `NAXIS_i · |scale_i|`, angular `CUNIT` converted to
    /// arcseconds, centred on the world coordinate of the grid's middle pixel.
    ///
    /// # Arguments
    /// * `nx` - pixel count along axis 1.
    /// * `ny` - pixel count along axis 2.
    pub fn to_grid2d(&self, nx: usize, ny: usize) -> Result<Grid2D, FitsError> {
        let ux = cunit_scale(&self.cunit[0]);
        let uy = cunit_scale(&self.cunit[1]);
        let scales = self.pixel_scales();
        let (sx, sy) = (scales[0].abs() * ux, scales[1].abs() * uy);
        if sx == 0.0 || sy == 0.0 {
            return Err(FitsError::structure("degenerate WCS pixel scale"));
        }
        // World coordinate of the geometric centre pixel (1-based).
        let cx = 0.5 * (nx as f64 + 1.0);
        let cy = 0.5 * (ny as f64 + 1.0);
        let dx = cx - self.crpix[0];
        let dy = cy - self.crpix[1];
        let wx = (self.crval[0] + self.m[0][0] * dx + self.m[0][1] * dy) * ux;
        let wy = (self.crval[1] + self.m[1][0] * dx + self.m[1][1] * dy) * uy;
        let (half_w, half_h) = (0.5 * nx as f64 * sx, 0.5 * ny as f64 * sy);
        let bbox = BBox::from_lims(wx - half_w, wx + half_w, wy - half_h, wy + half_h);
        Grid2D::new(nx, ny, bbox).map_err(|e| FitsError::structure(e.to_string()))
    }

    /// Emit linear-WCS cards describing `grid` (arcsec units, centred reference).
    ///
    /// # Arguments
    /// * `grid` - the grid geometry to stamp.
    ///
    /// Returns the cards `CRPIX/CRVAL/CDELT/CTYPE/CUNIT` for both axes.
    pub fn from_grid2d(grid: &Grid2D) -> Vec<Card> {
        let center = grid.bbox.center();
        vec![
            Card::real("CRPIX1", 0.5 * (grid.nx as f64 + 1.0)),
            Card::real("CRPIX2", 0.5 * (grid.ny as f64 + 1.0)),
            Card::real("CRVAL1", center.x),
            Card::real("CRVAL2", center.y),
            Card::real("CDELT1", grid.dx()),
            Card::real("CDELT2", grid.dy()),
            Card::new("CTYPE1", CardValue::Str("LINEAR".into()), None),
            Card::new("CTYPE2", CardValue::Str("LINEAR".into()), None),
            Card::new("CUNIT1", CardValue::Str("arcsec".into()), None),
            Card::new("CUNIT2", CardValue::Str("arcsec".into()), None),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_roundtrip_recovers_extent_and_scale() {
        let grid = Grid2D::new(64, 32, BBox::from_lims(-3.2, 3.2, -1.6, 1.6)).unwrap();
        let cards = LinearWcs::from_grid2d(&grid);
        let mut h = Header::new();
        for c in cards {
            h.push(c);
        }
        let wcs = LinearWcs::from_header(&h).unwrap();
        let back = wcs.to_grid2d(64, 32).unwrap();
        assert_eq!(back.nx, 64);
        assert_eq!(back.ny, 32);
        assert!((back.dx() - grid.dx()).abs() < 1e-9);
        assert!((back.dy() - grid.dy()).abs() < 1e-9);
        assert!((back.bbox.center().x - grid.bbox.center().x).abs() < 1e-9);
        assert!((back.width() - grid.width()).abs() < 1e-9);
    }

    #[test]
    fn no_wcs_keywords_gives_none() {
        let mut h = Header::new();
        h.push(Card::logical("SIMPLE", true));
        assert!(LinearWcs::from_header(&h).is_none());
    }

    #[test]
    fn cd_matrix_drives_pixel_scale() {
        let mut h = Header::new();
        h.push(Card::real("CRPIX1", 1.0));
        h.push(Card::real("CRPIX2", 1.0));
        h.push(Card::real("CD1_1", 0.5));
        h.push(Card::real("CD2_2", 0.25));
        let wcs = LinearWcs::from_header(&h).unwrap();
        let grid = wcs.to_grid2d(4, 8).unwrap();
        assert!((grid.width() - 4.0 * 0.5).abs() < 1e-9);
        assert!((grid.height() - 8.0 * 0.25).abs() < 1e-9);
    }

    #[test]
    fn degenerate_scale_errors() {
        let mut h = Header::new();
        h.push(Card::real("CRPIX1", 1.0));
        h.push(Card::real("CDELT1", 0.0));
        h.push(Card::real("CDELT2", 1.0));
        let wcs = LinearWcs::from_header(&h).unwrap();
        assert!(wcs.to_grid2d(4, 4).is_err());
    }

    #[test]
    fn pixel_default_gives_unit_grid() {
        let grid = LinearWcs::pixel_default().to_grid2d(10, 5).unwrap();
        assert!((grid.dx() - 1.0).abs() < 1e-9);
        assert!((grid.dy() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_dimension_grid_errors() {
        // A zero pixel count makes Grid2D::new reject the geometry.
        assert!(LinearWcs::pixel_default().to_grid2d(0, 4).is_err());
    }

    #[test]
    fn degree_units_convert_to_arcsec() {
        let mut h = Header::new();
        h.push(Card::real("CRPIX1", 1.0));
        h.push(Card::real("CRPIX2", 1.0));
        h.push(Card::real("CDELT1", 1.0 / 3600.0)); // 1 arcsec in degrees
        h.push(Card::real("CDELT2", 1.0 / 3600.0));
        h.push(Card::string("CUNIT1", "deg"));
        h.push(Card::string("CUNIT2", "deg"));
        let wcs = LinearWcs::from_header(&h).unwrap();
        let grid = wcs.to_grid2d(10, 10).unwrap();
        // 10 px × 1 arcsec/px = 10 arcsec extent.
        assert!((grid.width() - 10.0).abs() < 1e-9);
    }
}
