// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The whole-file FITS write API.
//!
//! [`Fits::to_bytes`] concatenates each HDU's encoded header and data segment
//! into a block-aligned buffer (advertising `EXTEND` in the primary when
//! extensions follow); [`Fits::write`] saves it. Builder conveniences construct
//! a single-HDU file directly from a [`FitsImage`], a [`PixelGrid`], or a
//! [`FieldArray2D`], stamping linear-WCS cards from the grid geometry.

use super::block::ByteSink;
use super::error::FitsError;
use super::hdu::Hdu;
use super::image::FitsImage;
use super::reader::Fits;
use super::wcs::LinearWcs;
use crate::grid::{FieldArray2D, Grid2D, PixelGrid};
use std::path::Path;

impl Fits {
    /// An empty file with no HDUs.
    pub fn empty() -> Self {
        Self { hdus: Vec::new() }
    }

    /// A single-HDU file whose Primary carries `image`.
    ///
    /// # Arguments
    /// * `image` - the image payload.
    pub fn from_image(image: FitsImage) -> Self {
        Self {
            hdus: vec![Hdu::primary(image)],
        }
    }

    /// A single-HDU file from a 2D field, stamping `grid`'s linear-WCS cards.
    ///
    /// # Arguments
    /// * `field` - the 2D field, shape `[ny, nx]`.
    /// * `grid` - the geometry whose WCS is written.
    /// * `bitpix` - the target `BITPIX`; defaults to `-32`.
    pub fn from_field_array(
        field: &FieldArray2D,
        grid: &Grid2D,
        bitpix: Option<i32>,
    ) -> Result<Self, FitsError> {
        let image = FitsImage::from_field_array(bitpix, field)?;
        let mut hdu = Hdu::primary(image);
        for card in LinearWcs::from_grid2d(grid) {
            hdu.header.push(card);
        }
        Ok(Self { hdus: vec![hdu] })
    }

    /// A single-HDU file from a [`PixelGrid`], stamping its linear-WCS cards.
    ///
    /// # Arguments
    /// * `pg` - the pixel grid to write.
    /// * `bitpix` - the target `BITPIX`; defaults to `-32`.
    pub fn from_pixelgrid(pg: &PixelGrid, bitpix: Option<i32>) -> Result<Self, FitsError> {
        Self::from_field_array(&pg.values, &pg.grid, bitpix)
    }

    /// Append an `IMAGE` extension carrying `image`.
    ///
    /// # Arguments
    /// * `image` - the image payload.
    pub fn push_image(&mut self, image: FitsImage) {
        self.hdus.push(Hdu::image_ext(image));
    }

    /// Encode the whole file to a block-aligned byte buffer.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut sink = ByteSink::new();
        let multi = self.hdus.len() > 1;
        for (i, hdu) in self.hdus.iter().enumerate() {
            if i == 0 && multi && !hdu.extend {
                let mut primary = hdu.clone();
                primary.extend = true;
                primary.encode(&mut sink);
            } else {
                hdu.encode(&mut sink);
            }
        }
        sink.into_bytes()
    }

    /// Encode and write the file to disk.
    ///
    /// # Arguments
    /// * `path` - the destination path.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), FitsError> {
        std::fs::write(path, self.to_bytes()).map_err(|e| FitsError::io(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::BBox;
    use ndarray::array;

    #[test]
    fn image_roundtrip_bytes() {
        let field = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let img = FitsImage::from_field_array(Some(-64), &field).unwrap();
        let bytes = Fits::from_image(img).to_bytes();
        assert_eq!(bytes.len() % 2880, 0);
        let back = Fits::from_bytes(&bytes).unwrap();
        assert_eq!(back.primary_image().unwrap().to_field_array().unwrap(), field);
    }

    #[test]
    fn integer_bitpix_is_exact() {
        let field = array![[0.0, 100.0], [200.0, 300.0]];
        let img = FitsImage::from_field_array(Some(32), &field).unwrap();
        let bytes = Fits::from_image(img).to_bytes();
        let back = Fits::from_bytes(&bytes).unwrap();
        assert_eq!(back.primary_image().unwrap().to_field_array().unwrap(), field);
    }

    #[test]
    fn pixelgrid_roundtrip_with_wcs() {
        let grid = Grid2D::new(4, 3, BBox::from_lims(-2.0, 2.0, -1.5, 1.5)).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x + 10.0 * y);
        let bytes = Fits::from_pixelgrid(&pg, None).unwrap().to_bytes();
        let back = Fits::from_bytes(&bytes).unwrap();
        let pg2 = back.to_pixelgrid(0, &[]).unwrap();
        // -32 default: float tolerance.
        assert!((pg2.grid.dx() - grid.dx()).abs() < 1e-6);
        for (a, b) in pg2.values.iter().zip(pg.values.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn empty_has_no_hdus() {
        assert!(Fits::empty().is_empty());
    }

    #[test]
    fn write_then_open_from_disk() {
        let field = array![[1.0, 2.0], [3.0, 4.0]];
        let img = FitsImage::from_field_array(Some(-64), &field).unwrap();
        let fits = Fits::from_image(img);
        let path = std::env::temp_dir().join(format!("lens_forge_w_{}.fits", std::process::id()));
        fits.write(&path).unwrap();
        let back = Fits::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back.primary_image().unwrap().to_field_array().unwrap(), field);
        let _ = std::fs::remove_file(&path);
        // Writing to an unwritable path is an I/O error.
        let bad = std::path::PathBuf::from("/nonexistent_dir_xyz/out.fits");
        assert!(fits.write(&bad).is_err());
    }

    #[test]
    fn multi_hdu_roundtrip_sets_extend() {
        let a = FitsImage::from_field_array(Some(-64), &array![[1.0, 2.0]]).unwrap();
        let b = FitsImage::from_field_array(Some(-64), &array![[3.0, 4.0, 5.0]]).unwrap();
        let mut fits = Fits::from_image(a);
        fits.push_image(b);
        let bytes = fits.to_bytes();
        let back = Fits::from_bytes(&bytes).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.hdus[0].header.get_logical("EXTEND"), Some(true));
        assert_eq!(
            back.image(1).unwrap().to_field_array().unwrap(),
            array![[3.0, 4.0, 5.0]]
        );
    }
}
