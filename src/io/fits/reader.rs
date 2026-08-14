// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The whole-file FITS read API.
//!
//! [`Fits::open`]/[`Fits::from_bytes`] parse a file into a list of [`Hdu`]s: for
//! each unit the header is parsed, the data byte size computed, and image data
//! decoded (Primary + `IMAGE`) or skipped (other extensions). Convenience
//! accessors return the primary image and bridge a chosen plane to a
//! [`PixelGrid`] via the header's linear WCS.

use super::block::{BLOCK_SIZE, ByteCursor, padded_len};
use super::data::{FitsData, bytes_per_element};
use super::error::FitsError;
use super::hdu::{Hdu, HduKind};
use super::header::Header;
use super::image::FitsImage;
use super::wcs::LinearWcs;
use crate::grid::PixelGrid;
use std::path::Path;

/// A parsed FITS file: a sequence of HDUs.
#[derive(Debug, Clone, Default)]
pub struct Fits {
    /// The Header+Data Units, in file order (index 0 is the Primary HDU).
    pub hdus: Vec<Hdu>,
}

impl Fits {
    /// Read and parse a FITS file from disk.
    ///
    /// # Arguments
    /// * `path` - the file to open.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FitsError> {
        let bytes = std::fs::read(path).map_err(|e| FitsError::io(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    /// Parse a FITS file already held in memory.
    ///
    /// # Arguments
    /// * `bytes` - the whole-file byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FitsError> {
        let mut cursor = ByteCursor::new(bytes);
        let mut hdus = Vec::new();
        while cursor.remaining() >= BLOCK_SIZE {
            let is_primary = hdus.is_empty();
            let hdu = parse_hdu(&mut cursor, is_primary)?;
            hdus.push(hdu);
        }
        if hdus.is_empty() {
            return Err(FitsError::structure("no HDU found (file too short)"));
        }
        Ok(Self { hdus })
    }

    /// The number of HDUs.
    pub fn len(&self) -> usize {
        self.hdus.len()
    }

    /// Whether there are no HDUs.
    pub fn is_empty(&self) -> bool {
        self.hdus.is_empty()
    }

    /// The image of the `i`-th HDU, if it carries one.
    ///
    /// # Arguments
    /// * `i` - the HDU index.
    pub fn image(&self, i: usize) -> Option<&FitsImage> {
        self.hdus.get(i).and_then(|h| h.image.as_ref())
    }

    /// The first HDU (searching from the primary) that carries an image.
    pub fn primary_image(&self) -> Option<&FitsImage> {
        self.hdus.iter().find_map(|h| h.image.as_ref())
    }

    /// Bridge a 2D plane of the `i`-th HDU's image to a [`PixelGrid`].
    ///
    /// The grid geometry comes from the header's linear WCS, falling back to
    /// pixel coordinates when no WCS is present.
    ///
    /// # Arguments
    /// * `i` - the HDU index.
    /// * `fixed` - one index per axis beyond the 2D plane (empty for 2D images).
    pub fn to_pixelgrid(&self, i: usize, fixed: &[usize]) -> Result<PixelGrid, FitsError> {
        let hdu = self
            .hdus
            .get(i)
            .ok_or_else(|| FitsError::structure(format!("no HDU at index {}", i)))?;
        let image = hdu
            .image
            .as_ref()
            .ok_or_else(|| FitsError::structure(format!("HDU {} carries no image", i)))?;
        let (nx, ny) = (image.naxes[0], image.naxes[1]);
        let wcs = LinearWcs::from_header(&hdu.header).unwrap_or_else(LinearWcs::pixel_default);
        let grid = wcs.to_grid2d(nx, ny)?;
        let field = image.plane(fixed)?;
        PixelGrid::from_values(grid, field).map_err(|e| FitsError::structure(e.to_string()))
    }
}

/// Parse one HDU at the cursor, decoding or skipping its data segment.
fn parse_hdu(cursor: &mut ByteCursor, is_primary: bool) -> Result<Hdu, FitsError> {
    let header = Header::parse(cursor)?;
    let kind = hdu_kind(&header, is_primary);
    let nbytes = data_byte_size(&header)?;
    let padded = padded_len(nbytes);

    let decode = nbytes > 0 && matches!(kind, HduKind::Primary | HduKind::Image);
    let image = if decode {
        let bitpix = header.require_int("BITPIX")? as i32;
        let naxes = axis_lengths(&header)?;
        let count: usize = naxes.iter().product();
        let seg = cursor.take(nbytes)?;
        let data = FitsData::decode(bitpix, seg, count)?;
        let bscale = header.get_float("BSCALE").unwrap_or(1.0);
        let bzero = header.get_float("BZERO").unwrap_or(0.0);
        let blank = header.get_int("BLANK");
        let img = FitsImage::new(naxes, data, bscale, bzero, blank)?;
        // Skip the zero padding after the data.
        cursor.skip((padded - nbytes).min(cursor.remaining()))?;
        Some(img)
    } else {
        cursor.skip(padded.min(cursor.remaining()))?;
        None
    };
    Ok(Hdu::from_parts(header, image, kind))
}

/// Classify an HDU from its header.
fn hdu_kind(header: &Header, is_primary: bool) -> HduKind {
    if is_primary {
        return HduKind::Primary;
    }
    match header.get_str("XTENSION").map(|s| s.trim()) {
        Some("IMAGE") => HduKind::Image,
        Some(other) => HduKind::Other(other.to_string()),
        None => HduKind::Other(String::new()),
    }
}

/// The per-axis lengths `[NAXIS1, …, NAXISn]` from a header.
fn axis_lengths(header: &Header) -> Result<Vec<usize>, FitsError> {
    let naxis = header.require_int("NAXIS")?;
    if naxis < 0 {
        return Err(FitsError::bad_value("NAXIS", "negative axis count"));
    }
    (1..=naxis)
        .map(|k| {
            let n = header.require_int(&format!("NAXIS{}", k))?;
            if n < 0 {
                Err(FitsError::bad_value(format!("NAXIS{}", k), "negative length"))
            } else {
                Ok(n as usize)
            }
        })
        .collect()
}

/// The unpadded data byte size implied by the header (FITS `Nbytes` formula).
fn data_byte_size(header: &Header) -> Result<usize, FitsError> {
    let bitpix = header.require_int("BITPIX")? as i32;
    let naxes = axis_lengths(header)?;
    if naxes.is_empty() || naxes.contains(&0) {
        return Ok(0);
    }
    let esize = bytes_per_element(bitpix)?;
    let nelem: usize = naxes.iter().product();
    let pcount = header.get_int("PCOUNT").unwrap_or(0).max(0) as usize;
    let gcount = header.get_int("GCOUNT").unwrap_or(1).max(1) as usize;
    Ok(esize * gcount * (pcount + nelem))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::fits::block::ByteSink;
    use crate::io::fits::card::Card;
    use ndarray::array;

    /// Assemble a minimal single-HDU FITS buffer in memory.
    fn primary_bytes() -> Vec<u8> {
        let img = FitsImage::from_field_array(Some(-64), &array![[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let mut sink = ByteSink::new();
        Hdu::primary(img).encode(&mut sink);
        sink.into_bytes()
    }

    #[test]
    fn read_primary_image() {
        let bytes = primary_bytes();
        let fits = Fits::from_bytes(&bytes).unwrap();
        assert_eq!(fits.len(), 1);
        let img = fits.primary_image().unwrap();
        assert_eq!(img.naxes, vec![2, 2]);
        assert_eq!(img.to_field_array().unwrap(), array![[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn skips_unknown_extension() {
        // Primary (no data) + a BINTABLE-like extension whose data must be skipped.
        let mut sink = ByteSink::new();
        let mut primary = Header::new();
        primary.push(Card::logical("SIMPLE", true));
        primary.push(Card::integer("BITPIX", 8));
        primary.push(Card::integer("NAXIS", 0));
        primary.push(Card::logical("EXTEND", true));
        primary.encode(&mut sink);
        // Extension: 3 bytes of data, XTENSION unknown.
        let mut ext = Header::new();
        ext.push(Card::string("XTENSION", "BINTABLE"));
        ext.push(Card::integer("BITPIX", 8));
        ext.push(Card::integer("NAXIS", 1));
        ext.push(Card::integer("NAXIS1", 3));
        ext.push(Card::integer("PCOUNT", 0));
        ext.push(Card::integer("GCOUNT", 1));
        ext.encode(&mut sink);
        sink.pad(0, 3);
        sink.pad_to_block(0);
        let bytes = sink.into_bytes();

        let fits = Fits::from_bytes(&bytes).unwrap();
        assert_eq!(fits.len(), 2);
        assert!(fits.image(0).is_none());
        assert!(fits.image(1).is_none());
        assert!(matches!(fits.hdus[1].kind, HduKind::Other(_)));
    }

    #[test]
    fn to_pixelgrid_with_wcs() {
        use crate::io::fits::wcs::LinearWcs;
        use crate::grid::{BBox, Grid2D};
        let grid = Grid2D::new(2, 2, BBox::from_lims(-1.0, 1.0, -1.0, 1.0)).unwrap();
        let img = FitsImage::from_field_array(Some(-64), &array![[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let mut hdu = Hdu::primary(img);
        for c in LinearWcs::from_grid2d(&grid) {
            hdu.header.push(c);
        }
        let mut sink = ByteSink::new();
        hdu.encode(&mut sink);
        let bytes = sink.into_bytes();
        let fits = Fits::from_bytes(&bytes).unwrap();
        let pg = fits.to_pixelgrid(0, &[]).unwrap();
        assert_eq!(pg.values, array![[1.0, 2.0], [3.0, 4.0]]);
        assert!((pg.grid.dx() - grid.dx()).abs() < 1e-9);
    }

    #[test]
    fn len_is_empty_and_short_buffer() {
        let fits = Fits::from_bytes(&primary_bytes()).unwrap();
        assert_eq!(fits.len(), 1);
        assert!(!fits.is_empty());
        // A sub-block buffer yields no HDU.
        assert!(Fits::from_bytes(&[0u8; 10]).is_err());
    }

    #[test]
    fn open_reads_from_disk() {
        let path = std::env::temp_dir().join(format!("lens_forge_{}.fits", std::process::id()));
        std::fs::write(&path, primary_bytes()).unwrap();
        let fits = Fits::open(&path).unwrap();
        assert_eq!(fits.len(), 1);
        assert!(fits.primary_image().is_some());
        let _ = std::fs::remove_file(&path);
        // Opening a missing file is an I/O error (both &str and &PathBuf paths).
        assert!(Fits::open("/nonexistent/path/definitely.fits").is_err());
        let missing = std::path::PathBuf::from("/nonexistent/path/definitely.fits");
        assert!(Fits::open(&missing).is_err());
    }

    fn header_bytes(cards: &[Card]) -> Vec<u8> {
        let mut sink = ByteSink::new();
        let mut h = Header::new();
        for c in cards {
            h.push(c.clone());
        }
        h.encode(&mut sink);
        sink.into_bytes()
    }

    #[test]
    fn negative_axis_counts_error() {
        let bytes = header_bytes(&[
            Card::logical("SIMPLE", true),
            Card::integer("BITPIX", 8),
            Card::integer("NAXIS", -1),
        ]);
        assert!(Fits::from_bytes(&bytes).is_err());

        let bytes = header_bytes(&[
            Card::logical("SIMPLE", true),
            Card::integer("BITPIX", 8),
            Card::integer("NAXIS", 1),
            Card::integer("NAXIS1", -3),
        ]);
        assert!(Fits::from_bytes(&bytes).is_err());
    }

    #[test]
    fn truncated_data_segment_errors() {
        // Header declares 1000 pixels but no data follows the header block.
        let bytes = header_bytes(&[
            Card::logical("SIMPLE", true),
            Card::integer("BITPIX", -64),
            Card::integer("NAXIS", 1),
            Card::integer("NAXIS1", 1000),
        ]);
        assert!(Fits::from_bytes(&bytes).is_err());
    }

    #[test]
    fn extension_without_xtension_is_other() {
        let mut bytes = header_bytes(&[
            Card::logical("SIMPLE", true),
            Card::integer("BITPIX", 8),
            Card::integer("NAXIS", 0),
        ]);
        bytes.extend(header_bytes(&[
            Card::integer("BITPIX", 8),
            Card::integer("NAXIS", 0),
        ]));
        let fits = Fits::from_bytes(&bytes).unwrap();
        assert_eq!(fits.len(), 2);
        assert!(matches!(&fits.hdus[1].kind, HduKind::Other(s) if s.is_empty()));
    }

    #[test]
    fn to_pixelgrid_error_paths() {
        let fits = Fits::from_bytes(&primary_bytes()).unwrap();
        assert!(fits.to_pixelgrid(9, &[]).is_err()); // no such HDU
        // A data-less primary carries no image.
        let mut sink = ByteSink::new();
        let mut h = Header::new();
        h.push(Card::logical("SIMPLE", true));
        h.push(Card::integer("BITPIX", 8));
        h.push(Card::integer("NAXIS", 0));
        h.encode(&mut sink);
        let empty = Fits::from_bytes(&sink.into_bytes()).unwrap();
        assert!(empty.to_pixelgrid(0, &[]).is_err());
    }
}
