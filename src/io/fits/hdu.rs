// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! A Header+Data Unit: a [`Header`] and an optional image payload.
//!
//! [`Hdu`] retains the cards it was read with for verbatim rewrite; on
//! [`Hdu::encode`] the mandatory structural keywords (`SIMPLE`/`XTENSION`,
//! `BITPIX`, `NAXIS`, `NAXISn`, `PCOUNT`/`GCOUNT`) and the scaling keywords are
//! (re)emitted first and in order from the current image, then the retained
//! user cards (`COMMENT`/`HISTORY`/WCS/provenance), then the data segment.

use super::block::ByteSink;
use super::card::{Card, CardValue};
use super::header::Header;
use super::image::FitsImage;

/// What kind of HDU this is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HduKind {
    /// The Primary HDU (`SIMPLE = T`).
    Primary,
    /// An `IMAGE` extension.
    Image,
    /// Another extension type, retained by header but with data skipped.
    Other(String),
}

/// A Header+Data Unit.
#[derive(Debug, Clone)]
pub struct Hdu {
    /// The (retained) header cards.
    pub header: Header,
    /// The decoded image, if this HDU carries image data we handle.
    pub image: Option<FitsImage>,
    /// The HDU's kind.
    pub kind: HduKind,
    /// Whether to advertise extensions in the primary header (`EXTEND = T`).
    pub extend: bool,
}

impl Hdu {
    /// Assemble an HDU from its parsed parts.
    ///
    /// # Arguments
    /// * `header` - the retained header.
    /// * `image` - the decoded image, if any.
    /// * `kind` - the HDU kind.
    pub fn from_parts(header: Header, image: Option<FitsImage>, kind: HduKind) -> Self {
        Self {
            header,
            image,
            kind,
            extend: false,
        }
    }

    /// A Primary HDU carrying `image` with a freshly generated minimal header.
    ///
    /// # Arguments
    /// * `image` - the image payload.
    pub fn primary(image: FitsImage) -> Self {
        Self {
            header: Header::new(),
            image: Some(image),
            kind: HduKind::Primary,
            extend: false,
        }
    }

    /// An `IMAGE` extension carrying `image` with a minimal header.
    ///
    /// # Arguments
    /// * `image` - the image payload.
    pub fn image_ext(image: FitsImage) -> Self {
        Self {
            header: Header::new(),
            image: Some(image),
            kind: HduKind::Image,
            extend: false,
        }
    }

    /// Whether this HDU holds image data this module can encode.
    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }

    /// Build the header to write: structural keywords first, then user cards.
    fn build_header(&self) -> Header {
        let mut h = Header::new();
        match &self.kind {
            HduKind::Primary => h.push(Card::logical("SIMPLE", true)),
            HduKind::Image => h.push(Card::string("XTENSION", "IMAGE")),
            HduKind::Other(x) => h.push(Card::string("XTENSION", x.clone())),
        }
        let (bitpix, naxes) = match &self.image {
            Some(img) => (img.bitpix, img.naxes.clone()),
            None => (8, Vec::new()),
        };
        h.push(Card::integer("BITPIX", bitpix as i64));
        h.push(Card::integer("NAXIS", naxes.len() as i64));
        for (k, &n) in naxes.iter().enumerate() {
            h.push(Card::integer(format!("NAXIS{}", k + 1), n as i64));
        }
        if matches!(self.kind, HduKind::Primary) && self.extend {
            h.push(Card::logical("EXTEND", true));
        }
        if !matches!(self.kind, HduKind::Primary) {
            h.push(Card::integer("PCOUNT", 0));
            h.push(Card::integer("GCOUNT", 1));
        }
        if let Some(img) = &self.image {
            if img.bscale != 1.0 {
                h.push(Card::real("BSCALE", img.bscale));
            }
            if img.bzero != 0.0 {
                h.push(Card::real("BZERO", img.bzero));
            }
            if let Some(b) = img.blank {
                h.push(Card::integer("BLANK", b));
            }
        }
        // Append retained user cards, skipping the structural ones we manage.
        for card in self.header.cards() {
            if !is_structural(&card.keyword) {
                h.push(card.clone());
            }
        }
        h
    }

    /// Encode this HDU (header block(s) + data segment) into `sink`.
    ///
    /// # Arguments
    /// * `sink` - the output sink to append to.
    pub fn encode(&self, sink: &mut ByteSink) {
        self.build_header().encode(sink);
        if let Some(img) = &self.image {
            img.data.encode(sink);
        }
    }

    /// Set a string card in the retained header (builder convenience).
    ///
    /// # Arguments
    /// * `keyword` - the keyword to set.
    /// * `value` - the string value.
    pub fn set_str(&mut self, keyword: &str, value: impl Into<String>) {
        self.header.set(keyword, CardValue::Str(value.into()));
    }
}

/// Whether a keyword is a structural/scaling keyword managed on encode.
fn is_structural(keyword: &str) -> bool {
    matches!(
        keyword,
        "SIMPLE" | "XTENSION" | "EXTEND" | "BITPIX" | "NAXIS" | "PCOUNT" | "GCOUNT" | "BSCALE"
            | "BZERO" | "BLANK" | "END"
    ) || is_naxis_n(keyword)
}

/// Whether a keyword is `NAXISn` for some positive integer `n`.
fn is_naxis_n(keyword: &str) -> bool {
    keyword
        .strip_prefix("NAXIS")
        .filter(|r| !r.is_empty())
        .is_some_and(|r| r.chars().all(|c| c.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::fits::block::ByteSink;
    use crate::io::fits::header::Header;
    use ndarray::array;

    fn tiny_image() -> FitsImage {
        FitsImage::from_field_array(Some(-32), &array![[1.0, 2.0], [3.0, 4.0]]).unwrap()
    }

    #[test]
    fn primary_header_has_mandatory_first() {
        let hdu = Hdu::primary(tiny_image());
        let h = hdu.build_header();
        let kws: Vec<&str> = h.cards().iter().map(|c| c.keyword.as_str()).collect();
        assert_eq!(&kws[..4], &["SIMPLE", "BITPIX", "NAXIS", "NAXIS1"]);
        assert_eq!(kws[4], "NAXIS2");
    }

    #[test]
    fn extension_has_pcount_gcount() {
        let hdu = Hdu::image_ext(tiny_image());
        let h = hdu.build_header();
        assert_eq!(h.get_str("XTENSION"), Some("IMAGE"));
        assert_eq!(h.get_int("PCOUNT"), Some(0));
        assert_eq!(h.get_int("GCOUNT"), Some(1));
    }

    #[test]
    fn retained_cards_kept_structural_dropped() {
        let mut header = Header::new();
        header.push(Card::integer("BITPIX", 999)); // stale structural, must be dropped
        header.push(Card::comment("hello"));
        let hdu = Hdu::from_parts(header, Some(tiny_image()), HduKind::Primary);
        let h = hdu.build_header();
        assert_eq!(h.get_int("BITPIX"), Some(-32)); // corrected from image
        assert!(h.cards().iter().any(|c| c.keyword == "COMMENT"));
    }

    #[test]
    fn encode_is_block_aligned() {
        let mut sink = ByteSink::new();
        Hdu::primary(tiny_image()).encode(&mut sink);
        assert_eq!(sink.len() % 2880, 0);
    }

    #[test]
    fn image_ext_and_has_image() {
        let hdu = Hdu::image_ext(tiny_image());
        assert!(hdu.has_image());
        assert_eq!(hdu.kind, HduKind::Image);
        let empty = Hdu::from_parts(Header::new(), None, HduKind::Primary);
        assert!(!empty.has_image());
    }

    #[test]
    fn set_str_writes_card() {
        let mut hdu = Hdu::primary(tiny_image());
        hdu.set_str("OBJECT", "M87");
        assert_eq!(hdu.header.get_str("OBJECT"), Some("M87"));
    }

    #[test]
    fn other_kind_keeps_xtension_name() {
        let hdu = Hdu::from_parts(
            Header::new(),
            None,
            HduKind::Other("BINTABLE".into()),
        );
        assert_eq!(hdu.build_header().get_str("XTENSION"), Some("BINTABLE"));
    }

    #[test]
    fn naxis_prefixed_keywords_classified() {
        // "NAXIS1" is structural (dropped); "NAXISBAD" is not (retained).
        let mut header = Header::new();
        header.push(Card::integer("NAXIS1", 999));
        header.push(Card::string("NAXISBAD", "x"));
        let hdu = Hdu::from_parts(header, Some(tiny_image()), HduKind::Primary);
        let h = hdu.build_header();
        assert_eq!(h.get_int("NAXIS1"), Some(2)); // corrected from image, not 999
        assert!(h.get("NAXISBAD").is_some()); // non-digit suffix → retained
    }

    #[test]
    fn scaling_keywords_emitted_when_set() {
        let mut img = tiny_image();
        img.bscale = 2.0;
        img.bzero = 32768.0;
        img.blank = Some(-1);
        let h = Hdu::primary(img).build_header();
        assert_eq!(h.get_float("BSCALE"), Some(2.0));
        assert_eq!(h.get_float("BZERO"), Some(32768.0));
        assert_eq!(h.get_int("BLANK"), Some(-1));
    }
}
