// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! One 80-byte FITS header card: parse and fixed-format encode.
//!
//! A card is `keyword` (cols 1–8), an optional value indicator `"= "`
//! (cols 9–10), then the value field and an optional `/ comment`. Commentary
//! keywords (`COMMENT`, `HISTORY`, blank) carry free text and no value; `END`
//! terminates the header. Values are logical (`T`/`F`), integer, real, complex,
//! or a quoted string in which a doubled quote `''` escapes an embedded one.
//! Parsing accepts fixed and free format; encoding emits fixed format.
//!
//! The multi-card `CONTINUE` long-string convention is stitched and split one
//! level up, in [`header`](super::header); each `CONTINUE` line is an ordinary
//! card here.

use super::error::FitsError;
use num::complex::Complex;

/// The width of a card image in bytes.
pub const CARD_LEN: usize = 80;
/// The width of the keyword field in bytes (cols 1–8).
pub const KEYWORD_LEN: usize = 8;

/// The typed value carried by a valued card.
#[derive(Debug, Clone, PartialEq)]
pub enum CardValue {
    /// A logical `T`/`F`.
    Logical(bool),
    /// A signed integer.
    Integer(i64),
    /// A floating-point real.
    Real(f64),
    /// A complex value `(re, im)` (rare).
    Complex(Complex<f64>),
    /// A character string (quotes stripped, `''` unescaped).
    Str(String),
    /// No value (commentary card, or a valued keyword left blank).
    None,
}

/// One parsed header card.
#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    /// The keyword, trailing spaces stripped (empty for a blank commentary card).
    pub keyword: String,
    /// The card's value.
    pub value: CardValue,
    /// The `/ comment`, if any (leading `/` and surrounding spaces stripped).
    pub comment: Option<String>,
}

impl Card {
    /// A valued card with an optional comment.
    ///
    /// # Arguments
    /// * `keyword` - the keyword (uppercased, truncated to 8 chars on encode).
    /// * `value` - the typed value.
    /// * `comment` - an optional trailing comment.
    pub fn new(keyword: impl Into<String>, value: CardValue, comment: Option<String>) -> Self {
        Self {
            keyword: keyword.into(),
            value,
            comment,
        }
    }

    /// A logical-valued card.
    ///
    /// # Arguments
    /// * `keyword` - the keyword.
    /// * `value` - the boolean value.
    pub fn logical(keyword: impl Into<String>, value: bool) -> Self {
        Self::new(keyword, CardValue::Logical(value), None)
    }

    /// An integer-valued card.
    ///
    /// # Arguments
    /// * `keyword` - the keyword.
    /// * `value` - the integer value.
    pub fn integer(keyword: impl Into<String>, value: i64) -> Self {
        Self::new(keyword, CardValue::Integer(value), None)
    }

    /// A real-valued card.
    ///
    /// # Arguments
    /// * `keyword` - the keyword.
    /// * `value` - the floating-point value.
    pub fn real(keyword: impl Into<String>, value: f64) -> Self {
        Self::new(keyword, CardValue::Real(value), None)
    }

    /// A string-valued card.
    ///
    /// # Arguments
    /// * `keyword` - the keyword.
    /// * `value` - the string value.
    pub fn string(keyword: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(keyword, CardValue::Str(value.into()), None)
    }

    /// A `COMMENT` card carrying free text.
    ///
    /// # Arguments
    /// * `text` - the comment text.
    pub fn comment(text: impl Into<String>) -> Self {
        Self::new("COMMENT", CardValue::None, Some(text.into()))
    }

    /// A `HISTORY` card carrying free text.
    ///
    /// # Arguments
    /// * `text` - the history text.
    pub fn history(text: impl Into<String>) -> Self {
        Self::new("HISTORY", CardValue::None, Some(text.into()))
    }

    /// The `END` card.
    pub fn end() -> Self {
        Self::new("END", CardValue::None, None)
    }

    /// Whether this is the `END` card.
    pub fn is_end(&self) -> bool {
        self.keyword == "END"
    }

    /// Whether this is a commentary card (`COMMENT`, `HISTORY`, or blank keyword).
    pub fn is_commentary(&self) -> bool {
        self.keyword.is_empty() || self.keyword == "COMMENT" || self.keyword == "HISTORY"
    }

    /// The value as a boolean, if logical.
    pub fn as_bool(&self) -> Option<bool> {
        match self.value {
            CardValue::Logical(b) => Some(b),
            _ => None,
        }
    }

    /// The value as an integer, if integer.
    pub fn as_i64(&self) -> Option<i64> {
        match self.value {
            CardValue::Integer(i) => Some(i),
            _ => None,
        }
    }

    /// The value as an `f64`, accepting integer or real.
    pub fn as_f64(&self) -> Option<f64> {
        match self.value {
            CardValue::Real(x) => Some(x),
            CardValue::Integer(i) => Some(i as f64),
            _ => None,
        }
    }

    /// The value as a string slice, if a string.
    pub fn as_str(&self) -> Option<&str> {
        match &self.value {
            CardValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Parse one 80-byte card image.
    ///
    /// # Arguments
    /// * `bytes` - exactly [`CARD_LEN`] bytes.
    /// * `offset` - the card's byte offset in the file (for diagnostics).
    ///
    /// Returns the parsed [`Card`].
    pub fn parse(bytes: &[u8], offset: usize) -> Result<Self, FitsError> {
        if bytes.len() != CARD_LEN {
            return Err(FitsError::bad_card(offset, bytes, "card is not 80 bytes"));
        }
        // Keywords are ASCII; reject non-ASCII up front so slicing is safe.
        if !bytes.is_ascii() {
            return Err(FitsError::bad_card(offset, bytes, "non-ASCII card"));
        }
        let keyword = std::str::from_utf8(&bytes[..KEYWORD_LEN])
            .unwrap()
            .trim_end()
            .to_string();
        let rest = std::str::from_utf8(&bytes[KEYWORD_LEN..]).unwrap();

        if keyword == "END" {
            return Ok(Card::end());
        }
        // Commentary keywords (and blank) carry free text, no value indicator.
        if keyword.is_empty() || keyword == "COMMENT" || keyword == "HISTORY" {
            let text = rest.trim_end().to_string();
            let comment = if text.is_empty() { None } else { Some(text) };
            return Ok(Card::new(keyword, CardValue::None, comment));
        }
        // CONTINUE carries a continuation string with no `= ` indicator.
        if keyword == "CONTINUE" {
            let (value, comment) = parse_value_field(rest, &keyword, offset)?;
            return Ok(Card::new(keyword, value, comment));
        }
        // A valued keyword must show `= ` in columns 9–10.
        if &bytes[KEYWORD_LEN..KEYWORD_LEN + 2] == b"= " {
            let (value, comment) = parse_value_field(&rest[2..], &keyword, offset)?;
            Ok(Card::new(keyword, value, comment))
        } else {
            // No value indicator: keep the keyword, treat the remainder as a comment.
            let text = rest.trim_end().to_string();
            let comment = if text.is_empty() { None } else { Some(text) };
            Ok(Card::new(keyword, CardValue::None, comment))
        }
    }

    /// Encode this card as a single fixed-format 80-byte image.
    ///
    /// A long string value is truncated to fit; the [`header`](super::header)
    /// layer splits such values across `CONTINUE` cards before calling this.
    pub fn to_card_image(&self) -> [u8; CARD_LEN] {
        format_card_image(&self.keyword, &self.value, self.comment.as_deref())
    }
}

/// Parse the value field (everything after the keyword+indicator) into a value
/// and optional comment.
fn parse_value_field(
    field: &str,
    keyword: &str,
    offset: usize,
) -> Result<(CardValue, Option<String>), FitsError> {
    let field = field.trim_start();
    if field.is_empty() {
        return Ok((CardValue::None, None));
    }
    if field.starts_with('\'') {
        return parse_string_value(field, offset);
    }
    // Split off the comment at the first '/'.
    let (token, comment) = match field.find('/') {
        Some(i) => (field[..i].trim(), clean_comment(&field[i + 1..])),
        None => (field.trim(), None),
    };
    if token.is_empty() {
        return Ok((CardValue::None, comment));
    }
    if token.starts_with('(') {
        return Ok((parse_complex(token, keyword)?, comment));
    }
    let value = match token {
        "T" => CardValue::Logical(true),
        "F" => CardValue::Logical(false),
        _ => parse_numeric(token, keyword)?,
    };
    Ok((value, comment))
}

/// Parse a quoted string value, unescaping `''` and finding a trailing comment.
fn parse_string_value(
    field: &str,
    offset: usize,
) -> Result<(CardValue, Option<String>), FitsError> {
    let bytes = field.as_bytes();
    let mut s = String::new();
    let mut i = 1; // skip opening quote
    loop {
        if i >= bytes.len() {
            return Err(FitsError::bad_card(
                offset,
                field.as_bytes(),
                "unterminated string value",
            ));
        }
        if bytes[i] == b'\'' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                s.push('\'');
                i += 2;
            } else {
                i += 1; // closing quote
                break;
            }
        } else {
            s.push(bytes[i] as char);
            i += 1;
        }
    }
    // Trailing spaces in a FITS string are not significant.
    while s.ends_with(' ') {
        s.pop();
    }
    let comment = field[i..].find('/').map(|j| &field[i..][j + 1..]);
    Ok((CardValue::Str(s), comment.and_then(clean_comment)))
}

/// Parse an integer or real numeric token (accepting a Fortran `D` exponent).
fn parse_numeric(token: &str, keyword: &str) -> Result<CardValue, FitsError> {
    if let Ok(i) = token.parse::<i64>() {
        return Ok(CardValue::Integer(i));
    }
    let normalized = token.replace(['D', 'd'], "E");
    if let Ok(x) = normalized.parse::<f64>() {
        return Ok(CardValue::Real(x));
    }
    Err(FitsError::bad_value(
        keyword,
        format!("unparseable numeric literal {:?}", token),
    ))
}

/// Parse a complex token of the form `(re, im)`.
fn parse_complex(token: &str, keyword: &str) -> Result<CardValue, FitsError> {
    let inner = token
        .strip_prefix('(')
        .and_then(|t| t.strip_suffix(')'))
        .ok_or_else(|| FitsError::bad_value(keyword, "malformed complex value"))?;
    let mut parts = inner.split(',');
    let re = parts.next().unwrap_or("").trim().replace(['D', 'd'], "E");
    let im = parts.next().unwrap_or("").trim().replace(['D', 'd'], "E");
    let re: f64 = re
        .parse()
        .map_err(|_| FitsError::bad_value(keyword, "bad complex real part"))?;
    let im: f64 = im
        .parse()
        .map_err(|_| FitsError::bad_value(keyword, "bad complex imaginary part"))?;
    Ok(CardValue::Complex(Complex::new(re, im)))
}

/// Trim a raw comment slice; `None` if it is empty after trimming.
fn clean_comment(raw: &str) -> Option<String> {
    let t = raw.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

/// Fixed-format encode of keyword + value + comment into one 80-byte image.
pub fn format_card_image(
    keyword: &str,
    value: &CardValue,
    comment: Option<&str>,
) -> [u8; CARD_LEN] {
    let mut card = [b' '; CARD_LEN];
    let kw = keyword.to_ascii_uppercase();
    let kw_bytes = kw.as_bytes();
    let n = kw_bytes.len().min(KEYWORD_LEN);
    card[..n].copy_from_slice(&kw_bytes[..n]);

    if kw == "END" {
        return card;
    }
    // Commentary cards: free text from column 9, no value indicator.
    if kw.is_empty() || kw == "COMMENT" || kw == "HISTORY" {
        if let Some(c) = comment {
            write_ascii(&mut card, KEYWORD_LEN, c);
        }
        return card;
    }
    // CONTINUE: string value from column 11, no value indicator.
    if kw == "CONTINUE" {
        let mut col = KEYWORD_LEN + 2;
        if let CardValue::Str(s) = value {
            col = write_quoted_string(&mut card, col, s);
        }
        write_inline_comment(&mut card, col, comment);
        return card;
    }

    // Valued keyword: "= " indicator in columns 9–10.
    card[KEYWORD_LEN] = b'=';
    card[KEYWORD_LEN + 1] = b' ';
    let value_start = KEYWORD_LEN + 2; // column 11 (byte 10)
    let col = match value {
        CardValue::Str(s) => write_quoted_string(&mut card, value_start, s),
        CardValue::None => value_start,
        _ => write_fixed_numeric(&mut card, value_start, value),
    };
    write_inline_comment(&mut card, col, comment);
    card
}

/// Write a quoted, `''`-escaped string starting at `start`; pad the content to
/// at least 8 chars (closing quote no earlier than column 20). Returns the next
/// free column.
fn write_quoted_string(card: &mut [u8; CARD_LEN], start: usize, s: &str) -> usize {
    let escaped = s.replace('\'', "''");
    let mut content: Vec<u8> = escaped.bytes().collect();
    if content.len() < 8 {
        content.resize(8, b' ');
    }
    // Leave room for the closing quote within the 80-byte card.
    let max_content = CARD_LEN.saturating_sub(start + 2);
    if content.len() > max_content {
        content.truncate(max_content);
    }
    card[start] = b'\'';
    let mut col = start + 1;
    for &b in &content {
        card[col] = b;
        col += 1;
    }
    card[col] = b'\'';
    col + 1
}

/// Write a logical/integer/real/complex value right-justified in columns 11–30
/// (or as far as it fits). Returns the next free column.
fn write_fixed_numeric(card: &mut [u8; CARD_LEN], start: usize, value: &CardValue) -> usize {
    let s = render_numeric(value);
    let field_end = (start + 20).min(CARD_LEN); // column 30
    let field_width = field_end - start;
    let bytes = s.as_bytes();
    if bytes.len() <= field_width {
        let pad = field_width - bytes.len();
        let at = start + pad;
        card[at..at + bytes.len()].copy_from_slice(bytes);
        field_end
    } else {
        let n = bytes.len().min(CARD_LEN - start);
        card[start..start + n].copy_from_slice(&bytes[..n]);
        start + n
    }
}

/// Render a logical/integer/real/complex value as its literal token.
fn render_numeric(value: &CardValue) -> String {
    match value {
        CardValue::Logical(true) => "T".to_string(),
        CardValue::Logical(false) => "F".to_string(),
        CardValue::Integer(i) => i.to_string(),
        CardValue::Real(x) => render_real(*x),
        CardValue::Complex(c) => format!("({}, {})", render_real(c.re), render_real(c.im)),
        CardValue::Str(_) | CardValue::None => String::new(),
    }
}

/// Render an `f64` so it reads back as a real (always carrying a `.` or `E`).
fn render_real(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let s = format!("{}", x);
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s.replace('e', "E")
    } else {
        format!("{}.0", s)
    }
}

/// Append ` / comment` starting at `col` if there is room.
fn write_inline_comment(card: &mut [u8; CARD_LEN], col: usize, comment: Option<&str>) {
    let Some(c) = comment else { return };
    let c = c.trim();
    if c.is_empty() {
        return;
    }
    // " / " separator then as much of the comment as fits.
    let sep_at = col + 1;
    if sep_at + 3 >= CARD_LEN {
        return;
    }
    card[sep_at] = b'/';
    write_ascii(card, sep_at + 2, c);
}

/// Copy ASCII `text` into `card` from `start`, truncating at the card end.
fn write_ascii(card: &mut [u8; CARD_LEN], start: usize, text: &str) {
    let n = text.len().min(CARD_LEN - start);
    card[start..start + n].copy_from_slice(&text.as_bytes()[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(card: &Card) -> Card {
        let image = card.to_card_image();
        assert_eq!(image.len(), CARD_LEN);
        Card::parse(&image, 0).unwrap()
    }

    #[test]
    fn logical_roundtrip() {
        let c = Card::logical("SIMPLE", true);
        let r = roundtrip(&c);
        assert_eq!(r.keyword, "SIMPLE");
        assert_eq!(r.as_bool(), Some(true));
    }

    #[test]
    fn integer_roundtrip() {
        let c = Card::new("BITPIX", CardValue::Integer(-32), Some("floats".into()));
        let r = roundtrip(&c);
        assert_eq!(r.as_i64(), Some(-32));
        assert_eq!(r.comment.as_deref(), Some("floats"));
    }

    #[test]
    fn real_roundtrip() {
        for x in [0.0, 1.5, -3.25, 6.02e23, 1e-20, 32768.0] {
            let c = Card::real("BSCALE", x);
            let r = roundtrip(&c);
            match r.value {
                CardValue::Real(y) => assert!((y - x).abs() <= x.abs() * 1e-12 + 1e-30),
                CardValue::Integer(i) => assert_eq!(i as f64, x),
                other => panic!("expected real, got {:?}", other),
            }
        }
    }

    #[test]
    fn string_roundtrip_and_escape() {
        let c = Card::string("OBJECT", "M31's core");
        let r = roundtrip(&c);
        assert_eq!(r.as_str(), Some("M31's core"));
    }

    #[test]
    fn short_string_padded_to_eight() {
        let c = Card::string("O", "AB");
        let image = c.to_card_image();
        // Opening quote at col 11 (byte 10), closing quote no earlier than col 20.
        assert_eq!(image[10], b'\'');
        assert_eq!(image[19], b'\'');
    }

    #[test]
    fn commentary_card() {
        let c = Card::comment("processed by lens-forge");
        let r = roundtrip(&c);
        assert!(r.is_commentary());
        assert_eq!(r.comment.as_deref(), Some("processed by lens-forge"));
    }

    #[test]
    fn end_card() {
        let c = Card::end();
        let image = c.to_card_image();
        assert_eq!(&image[..3], b"END");
        assert!(image[3..].iter().all(|&b| b == b' '));
        assert!(Card::parse(&image, 0).unwrap().is_end());
    }

    #[test]
    fn free_format_parse() {
        let mut img = [b' '; CARD_LEN];
        img[..8].copy_from_slice(b"NAXIS1  ");
        img[8] = b'=';
        img[9] = b' ';
        img[10..14].copy_from_slice(b"  64"); // not right-justified to col 30
        let c = Card::parse(&img, 0).unwrap();
        assert_eq!(c.as_i64(), Some(64));
    }

    #[test]
    fn fortran_d_exponent() {
        let mut img = [b' '; CARD_LEN];
        img[..8].copy_from_slice(b"VAL     ");
        img[8] = b'=';
        img[9] = b' ';
        img[10..17].copy_from_slice(b"1.5D3  ");
        let c = Card::parse(&img, 0).unwrap();
        assert_eq!(c.as_f64(), Some(1500.0));
    }

    /// An 80-byte card image from a left-justified ASCII string.
    fn card_img(s: &str) -> [u8; CARD_LEN] {
        let mut c = [b' '; CARD_LEN];
        let b = s.as_bytes();
        let n = b.len().min(CARD_LEN);
        c[..n].copy_from_slice(&b[..n]);
        c
    }

    #[test]
    fn parse_rejects_wrong_length_and_non_ascii() {
        assert!(Card::parse(b"too short", 0).is_err());
        let mut bad = [b' '; CARD_LEN];
        bad[0] = 0xFF;
        assert!(Card::parse(&bad, 0).is_err());
    }

    #[test]
    fn history_and_blank_commentary() {
        let h = Card::history("reduced");
        assert!(h.is_commentary());
        let r = roundtrip(&h);
        assert_eq!(r.keyword, "HISTORY");
        assert_eq!(r.comment.as_deref(), Some("reduced"));
        // Blank-keyword commentary card.
        let c = Card::parse(&card_img("         some free text"), 0).unwrap();
        assert!(c.is_commentary());
        assert_eq!(c.keyword, "");
    }

    #[test]
    fn accessors_return_none_for_mismatch() {
        let s = Card::string("K", "v");
        assert_eq!(s.as_bool(), None);
        assert_eq!(s.as_i64(), None);
        assert_eq!(s.as_f64(), None);
        let i = Card::integer("K", 5);
        assert_eq!(i.as_str(), None);
        assert_eq!(i.as_f64(), Some(5.0)); // integer accepted as f64
        assert_eq!(i.as_bool(), None);
    }

    #[test]
    fn valued_without_indicator_becomes_comment() {
        let c = Card::parse(&card_img("FOO     no indicator here"), 0).unwrap();
        assert_eq!(c.keyword, "FOO");
        assert_eq!(c.value, CardValue::None);
        assert!(c.comment.as_deref().unwrap().contains("no indicator"));
    }

    #[test]
    fn empty_value_field_variants() {
        let c = Card::parse(&card_img("EMPTY   = "), 0).unwrap();
        assert_eq!(c.value, CardValue::None);
        let c2 = Card::parse(&card_img("EMPTY   =   / just a comment"), 0).unwrap();
        assert_eq!(c2.value, CardValue::None);
        assert_eq!(c2.comment.as_deref(), Some("just a comment"));
    }

    #[test]
    fn complex_value_roundtrip_and_errors() {
        let c = Card::new("CPLX", CardValue::Complex(Complex::new(1.5, -2.0)), None);
        let r = roundtrip(&c);
        assert_eq!(r.value, CardValue::Complex(Complex::new(1.5, -2.0)));
        assert!(Card::parse(&card_img("CPLX    = (oops, 1.0)"), 0).is_err()); // bad real
        assert!(Card::parse(&card_img("CPLX    = (1.0, oops)"), 0).is_err()); // bad imaginary
        assert!(Card::parse(&card_img("CPLX    = (1.0"), 0).is_err()); // malformed
    }

    #[test]
    fn parse_numeric_and_string_errors() {
        assert!(Card::parse(&card_img("BAD     = 12abc"), 0).is_err());
        assert!(Card::parse(&card_img("OBJ     = 'unclosed"), 0).is_err());
    }

    #[test]
    fn continue_card_parses_as_string() {
        let c = Card::parse(&card_img("CONTINUE  'more text&'"), 0).unwrap();
        assert_eq!(c.keyword, "CONTINUE");
        assert_eq!(c.as_str(), Some("more text&"));
    }

    #[test]
    fn render_logical_false_and_nan_inf() {
        let img = Card::logical("BOOL", false).to_card_image();
        assert_eq!(img[29], b'F'); // right-justified to column 30
        let nan = Card::real("X", f64::NAN).to_card_image();
        assert!(String::from_utf8_lossy(&nan).contains("NaN"));
        let inf = Card::real("X", f64::INFINITY).to_card_image();
        assert!(String::from_utf8_lossy(&inf).contains("inf"));
        let ninf = Card::real("X", f64::NEG_INFINITY).to_card_image();
        assert!(String::from_utf8_lossy(&ninf).contains("-inf"));
    }

    #[test]
    fn blank_comment_not_written() {
        let c = Card::new("K", CardValue::Integer(1), Some("   ".into()));
        assert!(!c.to_card_image().contains(&b'/'));
    }

    #[test]
    fn valued_none_encodes_indicator_and_comment() {
        // A valued keyword with no value but a comment.
        let c = Card::new("FOO", CardValue::None, Some("note".into()));
        let img = c.to_card_image();
        assert_eq!(&img[8..10], b"= ");
        assert!(String::from_utf8_lossy(&img).contains("/ note"));
    }

    #[test]
    fn empty_comment_after_slash_parses_as_none() {
        let c = Card::parse(&card_img("K       = 5 /   "), 0).unwrap();
        assert_eq!(c.as_i64(), Some(5));
        assert_eq!(c.comment, None);
    }

    #[test]
    fn overlong_value_and_string_are_truncated_to_one_card() {
        // A string far longer than one card is truncated (header splits instead).
        let long = "x".repeat(200);
        let img = Card::string("OBJECT", &long).to_card_image();
        assert_eq!(img.len(), CARD_LEN);
        assert_eq!(img[CARD_LEN - 1], b'\''); // closing quote at the very end
        // A real whose decimal expansion exceeds the fixed field still fits.
        let img = Card::real("BIG", 1e300).to_card_image();
        assert_eq!(img.len(), CARD_LEN);
    }
}
