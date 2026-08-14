// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! A FITS header: an ordered list of [`Card`]s with keyword lookup.
//!
//! [`Header::parse`] consumes whole 2880-byte blocks up to the `END` card and
//! stitches `CONTINUE` long strings back together; [`Header::encode`] writes the
//! cards verbatim, splitting any over-long string back across `CONTINUE` cards,
//! then appends `END` and space-pads to the block boundary. Callers that build a
//! header fresh are responsible for placing the mandatory structural keywords
//! (`SIMPLE`/`XTENSION`, `BITPIX`, `NAXIS`, …) first and in order.

use super::block::{ByteCursor, ByteSink, HEADER_PAD};
use super::card::{Card, CardValue, format_card_image};
use super::error::FitsError;
use std::collections::HashMap;

/// Number of 80-byte cards in one 2880-byte block.
const CARDS_PER_BLOCK: usize = 36;

/// Encoded-content budget (chars) for a continued-string card chunk.
const CONTINUE_BUDGET: usize = 67;

/// An ordered collection of header cards with a keyword index.
#[derive(Debug, Clone, Default)]
pub struct Header {
    cards: Vec<Card>,
    index: HashMap<String, usize>,
}

impl Header {
    /// An empty header.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a header from a card list, indexing the first of each keyword.
    ///
    /// # Arguments
    /// * `cards` - the cards, in order.
    pub fn from_cards(cards: Vec<Card>) -> Self {
        let mut h = Self {
            cards,
            index: HashMap::new(),
        };
        h.rebuild_index();
        h
    }

    /// The cards in order.
    pub fn cards(&self) -> &[Card] {
        &self.cards
    }

    /// Number of cards (excluding the implicit `END`).
    pub fn len(&self) -> usize {
        self.cards.len()
    }

    /// Whether the header holds no cards.
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// The first card for `keyword`, if present.
    ///
    /// # Arguments
    /// * `keyword` - the keyword to look up (case-sensitive, uppercase).
    pub fn get(&self, keyword: &str) -> Option<&Card> {
        self.index.get(keyword).map(|&i| &self.cards[i])
    }

    /// The logical value of `keyword`, if present and logical.
    ///
    /// # Arguments
    /// * `keyword` - the keyword to look up.
    pub fn get_logical(&self, keyword: &str) -> Option<bool> {
        self.get(keyword).and_then(Card::as_bool)
    }

    /// The integer value of `keyword`, if present and integer.
    ///
    /// # Arguments
    /// * `keyword` - the keyword to look up.
    pub fn get_int(&self, keyword: &str) -> Option<i64> {
        self.get(keyword).and_then(Card::as_i64)
    }

    /// The floating-point value of `keyword` (accepting an integer literal).
    ///
    /// # Arguments
    /// * `keyword` - the keyword to look up.
    pub fn get_float(&self, keyword: &str) -> Option<f64> {
        self.get(keyword).and_then(Card::as_f64)
    }

    /// The string value of `keyword`, if present and a string.
    ///
    /// # Arguments
    /// * `keyword` - the keyword to look up.
    pub fn get_str(&self, keyword: &str) -> Option<&str> {
        self.get(keyword).and_then(Card::as_str)
    }

    /// The integer value of `keyword`, erroring if absent.
    ///
    /// # Arguments
    /// * `keyword` - the required keyword.
    pub fn require_int(&self, keyword: &str) -> Result<i64, FitsError> {
        self.get_int(keyword)
            .ok_or_else(|| FitsError::missing_keyword(keyword))
    }

    /// Append a card at the end, updating the index.
    ///
    /// # Arguments
    /// * `card` - the card to append.
    pub fn push(&mut self, card: Card) {
        let i = self.cards.len();
        if !card.is_commentary() && !self.index.contains_key(&card.keyword) {
            self.index.insert(card.keyword.clone(), i);
        }
        self.cards.push(card);
    }

    /// Set a valued keyword, replacing an existing card or appending a new one.
    ///
    /// # Arguments
    /// * `keyword` - the keyword to set.
    /// * `value` - the value to store.
    pub fn set(&mut self, keyword: &str, value: CardValue) {
        if let Some(&i) = self.index.get(keyword) {
            self.cards[i].value = value;
        } else {
            self.push(Card::new(keyword, value, None));
        }
    }

    /// Remove all cards for `keyword` (no-op if absent).
    ///
    /// # Arguments
    /// * `keyword` - the keyword to remove.
    pub fn remove(&mut self, keyword: &str) {
        self.cards.retain(|c| c.keyword != keyword);
        self.rebuild_index();
    }

    /// Rebuild the keyword→index map from the current cards.
    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, c) in self.cards.iter().enumerate() {
            if !c.is_commentary() {
                self.index.entry(c.keyword.clone()).or_insert(i);
            }
        }
    }

    /// Parse a header from the cursor, consuming whole blocks up to `END`.
    ///
    /// Stitches `CONTINUE` long strings; the cursor is left at the block
    /// boundary following the header.
    ///
    /// # Arguments
    /// * `cursor` - positioned at the start of a header block.
    ///
    /// Returns the parsed [`Header`].
    pub fn parse(cursor: &mut ByteCursor) -> Result<Self, FitsError> {
        let mut raw = Vec::new();
        let mut saw_end = false;
        while !saw_end {
            let block_offset = cursor.pos();
            let block = cursor.take_block()?;
            for i in 0..CARDS_PER_BLOCK {
                let bytes = &block[i * 80..(i + 1) * 80];
                let card = Card::parse(bytes, block_offset + i * 80)?;
                if card.is_end() {
                    saw_end = true;
                    break;
                }
                raw.push(card);
            }
        }
        Ok(Self::from_cards(stitch_continue(raw)))
    }

    /// Encode the header: cards (splitting long strings), `END`, block-padded.
    ///
    /// # Arguments
    /// * `sink` - the output sink to append to.
    pub fn encode(&self, sink: &mut ByteSink) {
        for card in &self.cards {
            encode_card(sink, card);
        }
        sink.extend(&Card::end().to_card_image());
        sink.pad_to_block(HEADER_PAD);
    }
}

/// Stitch `CONTINUE` cards into the string value of the card they extend.
fn stitch_continue(cards: Vec<Card>) -> Vec<Card> {
    let mut out: Vec<Card> = Vec::with_capacity(cards.len());
    for card in cards {
        if card.keyword == "CONTINUE"
            && let (Some(prev), CardValue::Str(more)) = (out.last_mut(), &card.value)
            && let CardValue::Str(prefix) = &mut prev.value
        {
            if prefix.ends_with('&') {
                prefix.pop();
            }
            prefix.push_str(more);
            if card.comment.is_some() {
                prev.comment = card.comment.clone();
            }
            continue;
        }
        out.push(card);
    }
    out
}

/// Encode one card, splitting an over-long string across `CONTINUE` cards.
fn encode_card(sink: &mut ByteSink, card: &Card) {
    if let CardValue::Str(s) = &card.value {
        let escaped_len = s.len() + s.matches('\'').count();
        if !card.is_commentary() && card.keyword != "CONTINUE" && escaped_len > CONTINUE_BUDGET {
            encode_long_string(sink, &card.keyword, s, card.comment.as_deref());
            return;
        }
    }
    sink.extend(&card.to_card_image());
}

/// Emit a value card plus `CONTINUE` cards for a string exceeding one card.
fn encode_long_string(sink: &mut ByteSink, keyword: &str, s: &str, comment: Option<&str>) {
    let chunks = chunk_string(s);
    let last = chunks.len() - 1;
    for (i, chunk) in chunks.iter().enumerate() {
        let mut text = chunk.clone();
        let is_last = i == last;
        if !is_last {
            text.push('&');
        }
        let value = CardValue::Str(text);
        let kw = if i == 0 { keyword } else { "CONTINUE" };
        let cmt = if is_last { comment } else { None };
        sink.extend(&format_card_image(kw, &value, cmt));
    }
}

/// Split `s` into chunks whose escaped length fits [`CONTINUE_BUDGET`].
fn chunk_string(s: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    let mut len = 0;
    for ch in s.chars() {
        let enc = if ch == '\'' { 2 } else { 1 };
        if len + enc > CONTINUE_BUDGET {
            chunks.push(std::mem::take(&mut cur));
            len = 0;
        }
        cur.push(ch);
        len += enc;
    }
    chunks.push(cur);
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::fits::block::BLOCK_SIZE;

    fn minimal() -> Header {
        let mut h = Header::new();
        h.push(Card::logical("SIMPLE", true));
        h.push(Card::integer("BITPIX", -32));
        h.push(Card::integer("NAXIS", 2));
        h.push(Card::integer("NAXIS1", 4));
        h.push(Card::integer("NAXIS2", 3));
        h
    }

    #[test]
    fn encode_is_block_aligned_with_end() {
        let mut sink = ByteSink::new();
        minimal().encode(&mut sink);
        let bytes = sink.into_bytes();
        assert_eq!(bytes.len() % BLOCK_SIZE, 0);
        assert_eq!(bytes.len(), BLOCK_SIZE);
        // END present.
        let has_end = bytes
            .chunks(80)
            .any(|c| &c[..3] == b"END" && c[3..].iter().all(|&b| b == b' '));
        assert!(has_end);
    }

    #[test]
    fn parse_roundtrip() {
        let mut sink = ByteSink::new();
        minimal().encode(&mut sink);
        let bytes = sink.into_bytes();
        let mut cur = ByteCursor::new(&bytes);
        let h = Header::parse(&mut cur).unwrap();
        assert_eq!(h.get_logical("SIMPLE"), Some(true));
        assert_eq!(h.require_int("BITPIX").unwrap(), -32);
        assert_eq!(h.get_int("NAXIS1"), Some(4));
        assert_eq!(h.get_int("NAXIS2"), Some(3));
        assert_eq!(cur.pos(), BLOCK_SIZE);
    }

    #[test]
    fn set_and_remove() {
        let mut h = minimal();
        h.set("BITPIX", CardValue::Integer(16));
        assert_eq!(h.get_int("BITPIX"), Some(16));
        h.remove("NAXIS2");
        assert!(h.get("NAXIS2").is_none());
        assert_eq!(h.get_int("NAXIS1"), Some(4)); // index still valid
    }

    #[test]
    fn accessors_and_set_new_keyword() {
        let h = minimal();
        assert_eq!(h.len(), 5);
        assert!(!h.is_empty());
        assert_eq!(h.cards().len(), 5);
        assert!(Header::new().is_empty());
        assert_eq!(h.get_str("BITPIX"), None); // present but not a string
        assert_eq!(h.get_float("NAXIS1"), Some(4.0));
        assert!(h.require_int("MISSING").is_err());

        let mut h = minimal();
        h.set("OBJECT", CardValue::Integer(7)); // keyword not yet present
        assert_eq!(h.get_int("OBJECT"), Some(7));
    }

    #[test]
    fn stray_continue_is_kept_verbatim() {
        // A CONTINUE with no preceding string card cannot be stitched.
        let cards = vec![
            Card::logical("SIMPLE", true),
            Card::string("CONTINUE", "orphan"),
        ];
        let stitched = stitch_continue(cards);
        assert_eq!(stitched.len(), 2);
        assert_eq!(stitched[1].keyword, "CONTINUE");
    }

    #[test]
    fn continue_long_string_roundtrip() {
        let long = "a very long OBJECT name that certainly exceeds one \
                    eighty-byte FITS card and forces the CONTINUE convention \
                    to kick in for the writer and reader alike";
        let mut h = Header::new();
        h.push(Card::logical("SIMPLE", true));
        h.push(Card::string("OBJECT", long));
        let mut sink = ByteSink::new();
        h.encode(&mut sink);
        let bytes = sink.into_bytes();
        // Multiple cards were emitted for OBJECT (CONTINUE present).
        assert!(bytes.chunks(80).any(|c| &c[..8] == b"CONTINUE"));
        let mut cur = ByteCursor::new(&bytes);
        let parsed = Header::parse(&mut cur).unwrap();
        assert_eq!(parsed.get_str("OBJECT"), Some(long));
    }

    #[test]
    fn continue_carries_comment_to_stitched_value() {
        let long = "z".repeat(120);
        let mut h = Header::new();
        h.push(Card::new(
            "OBJECT",
            CardValue::Str(long.clone()),
            Some("a note".into()),
        ));
        let mut sink = ByteSink::new();
        h.encode(&mut sink);
        let bytes = sink.into_bytes();
        let mut cur = ByteCursor::new(&bytes);
        let parsed = Header::parse(&mut cur).unwrap();
        assert_eq!(parsed.get_str("OBJECT"), Some(long.as_str()));
        assert_eq!(
            parsed.get("OBJECT").unwrap().comment.as_deref(),
            Some("a note")
        );
    }
}
