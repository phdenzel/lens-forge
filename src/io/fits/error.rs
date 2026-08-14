// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Rich, context-carrying errors for FITS parse/encode, bridged to
//! [`LensForgeError`].
//!
//! Filesystem failures stringify into [`LensForgeError::IoError`]; every
//! parse/encode failure into [`LensForgeError::FormatError`]. Each variant keeps
//! enough context (a byte offset, the offending 80-byte card, a missing keyword)
//! to locate the problem.

use crate::LensForgeError;
use std::fmt;

/// A FITS-specific error with diagnostic context.
#[derive(Debug, Clone, PartialEq)]
pub enum FitsError {
    /// A filesystem operation failed (path open/read/write).
    Io(String),
    /// The buffer ended before an expected read completed.
    UnexpectedEof {
        /// Byte offset at which the read was attempted.
        offset: usize,
        /// Bytes requested.
        needed: usize,
        /// Bytes actually available.
        available: usize,
    },
    /// An 80-byte card image could not be parsed.
    BadCard {
        /// Byte offset of the card in the file.
        offset: usize,
        /// The offending card image, lossily decoded as ASCII.
        card: String,
        /// What specifically went wrong.
        reason: String,
    },
    /// A required keyword was absent from a header.
    MissingKeyword(String),
    /// A keyword held a value of an unexpected type or an unparseable literal.
    BadValue {
        /// The keyword whose value was rejected.
        keyword: String,
        /// What specifically went wrong.
        reason: String,
    },
    /// A `BITPIX` value outside the FITS-allowed set.
    BadBitpix(i64),
    /// A structural inconsistency (dimension count, data size, …).
    Structure(String),
    /// A supported-but-not-yet-implemented or out-of-scope feature.
    Unsupported(String),
}

impl FitsError {
    /// Construct an [`FitsError::Io`].
    ///
    /// # Arguments
    /// * `msg` - the failure description.
    pub fn io(msg: impl Into<String>) -> Self {
        FitsError::Io(msg.into())
    }

    /// Construct an [`FitsError::UnexpectedEof`].
    ///
    /// # Arguments
    /// * `offset` - byte offset of the attempted read.
    /// * `needed` - bytes requested.
    /// * `available` - bytes available.
    pub fn unexpected_eof(offset: usize, needed: usize, available: usize) -> Self {
        FitsError::UnexpectedEof {
            offset,
            needed,
            available,
        }
    }

    /// Construct a [`FitsError::BadCard`] from a raw 80-byte image.
    ///
    /// # Arguments
    /// * `offset` - byte offset of the card.
    /// * `card` - the raw card bytes.
    /// * `reason` - what went wrong.
    pub fn bad_card(offset: usize, card: &[u8], reason: impl Into<String>) -> Self {
        FitsError::BadCard {
            offset,
            card: String::from_utf8_lossy(card).into_owned(),
            reason: reason.into(),
        }
    }

    /// Construct a [`FitsError::MissingKeyword`].
    ///
    /// # Arguments
    /// * `keyword` - the absent keyword.
    pub fn missing_keyword(keyword: impl Into<String>) -> Self {
        FitsError::MissingKeyword(keyword.into())
    }

    /// Construct a [`FitsError::BadValue`].
    ///
    /// # Arguments
    /// * `keyword` - the keyword whose value was rejected.
    /// * `reason` - what went wrong.
    pub fn bad_value(keyword: impl Into<String>, reason: impl Into<String>) -> Self {
        FitsError::BadValue {
            keyword: keyword.into(),
            reason: reason.into(),
        }
    }

    /// Construct a [`FitsError::Structure`].
    ///
    /// # Arguments
    /// * `msg` - the inconsistency description.
    pub fn structure(msg: impl Into<String>) -> Self {
        FitsError::Structure(msg.into())
    }

    /// Construct a [`FitsError::Unsupported`].
    ///
    /// # Arguments
    /// * `msg` - the unsupported feature.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        FitsError::Unsupported(msg.into())
    }

    /// Whether this error is a filesystem failure (vs a format problem).
    fn is_io(&self) -> bool {
        matches!(self, FitsError::Io(_))
    }
}

impl fmt::Display for FitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FitsError::Io(msg) => write!(f, "FITS I/O: {}", msg),
            FitsError::UnexpectedEof {
                offset,
                needed,
                available,
            } => write!(
                f,
                "unexpected end of FITS data at byte {}: needed {}, {} available",
                offset, needed, available
            ),
            FitsError::BadCard {
                offset,
                card,
                reason,
            } => write!(
                f,
                "bad FITS card at byte {} ({}): {:?}",
                offset, reason, card
            ),
            FitsError::MissingKeyword(kw) => write!(f, "missing FITS keyword {:?}", kw),
            FitsError::BadValue { keyword, reason } => {
                write!(f, "bad value for FITS keyword {:?}: {}", keyword, reason)
            }
            FitsError::BadBitpix(b) => write!(f, "invalid BITPIX {}", b),
            FitsError::Structure(msg) => write!(f, "FITS structure error: {}", msg),
            FitsError::Unsupported(msg) => write!(f, "unsupported FITS feature: {}", msg),
        }
    }
}

impl std::error::Error for FitsError {}

impl From<FitsError> for LensForgeError {
    fn from(e: FitsError) -> Self {
        if e.is_io() {
            LensForgeError::IoError(e.to_string())
        } else {
            LensForgeError::FormatError(e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_maps_to_io_variant() {
        let e = FitsError::io("disk gone");
        let l: LensForgeError = e.into();
        assert!(matches!(l, LensForgeError::IoError(_)));
    }

    #[test]
    fn format_maps_to_format_variant() {
        let e = FitsError::missing_keyword("BITPIX");
        let l: LensForgeError = e.into();
        assert!(matches!(l, LensForgeError::FormatError(_)));
        assert!(l.to_string().contains("BITPIX"));
    }

    #[test]
    fn bad_card_keeps_context() {
        let e = FitsError::bad_card(160, b"BOGUS", "no value indicator");
        let s = e.to_string();
        assert!(s.contains("160"));
        assert!(s.contains("BOGUS"));
    }

    #[test]
    fn every_variant_displays_and_converts_to_format() {
        let format_variants = [
            FitsError::unexpected_eof(10, 4, 2),
            FitsError::bad_card(0, b"X", "why"),
            FitsError::missing_keyword("NAXIS"),
            FitsError::bad_value("BSCALE", "nope"),
            FitsError::BadBitpix(7),
            FitsError::structure("mismatch"),
            FitsError::unsupported("BINTABLE"),
        ];
        for v in format_variants {
            assert!(!v.to_string().is_empty());
            let l: LensForgeError = v.clone().into();
            assert!(matches!(l, LensForgeError::FormatError(_)));
        }
    }
}
