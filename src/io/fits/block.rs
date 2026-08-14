// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Block-aligned byte handling shared by the FITS reader and writer.
//!
//! A FITS file is a sequence of [`BLOCK_SIZE`]-byte blocks; every header and
//! data segment is padded to a whole number of blocks — headers with ASCII
//! spaces, data with zero bytes. [`ByteCursor`] reads big-endian primitives off
//! a `&[u8]`; [`ByteSink`] writes them into a growing `Vec<u8>`. Both keep the
//! block alignment explicit so the two directions stay symmetric.

use super::error::FitsError;

/// The FITS block size in bytes; all I/O is aligned to this boundary.
pub const BLOCK_SIZE: usize = 2880;

/// The fill byte for header padding (ASCII space).
pub const HEADER_PAD: u8 = b' ';

/// The fill byte for data padding (zero).
pub const DATA_PAD: u8 = 0;

/// Number of blocks needed to hold `len` bytes (rounded up).
#[inline]
pub fn blocks_for(len: usize) -> usize {
    len.div_ceil(BLOCK_SIZE)
}

/// `len` rounded up to a whole number of [`BLOCK_SIZE`]-byte blocks.
#[inline]
pub fn padded_len(len: usize) -> usize {
    blocks_for(len) * BLOCK_SIZE
}

/// A forward-only cursor over a byte buffer, reading big-endian primitives.
///
/// Tracks a byte offset for block-aligned advancing and for the diagnostic
/// offsets carried by [`FitsError`].
pub struct ByteCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    /// Wrap a byte buffer at offset zero.
    ///
    /// # Arguments
    /// * `buf` - the whole-file bytes to read from.
    #[inline]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Current byte offset from the start of the buffer.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Total buffer length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Bytes remaining after the current position.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// Whether the cursor has consumed the whole buffer.
    #[inline]
    pub fn at_end(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Borrow the next `n` bytes and advance, erroring on a short buffer.
    ///
    /// # Arguments
    /// * `n` - number of bytes to take.
    #[inline]
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], FitsError> {
        if self.pos + n > self.buf.len() {
            return Err(FitsError::unexpected_eof(self.pos, n, self.remaining()));
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Borrow the next [`BLOCK_SIZE`] bytes as one block.
    #[inline]
    pub fn take_block(&mut self) -> Result<&'a [u8], FitsError> {
        self.take(BLOCK_SIZE)
    }

    /// Skip `n` bytes without reading them, erroring past the end.
    ///
    /// # Arguments
    /// * `n` - number of bytes to skip.
    #[inline]
    pub fn skip(&mut self, n: usize) -> Result<(), FitsError> {
        if self.pos + n > self.buf.len() {
            return Err(FitsError::unexpected_eof(self.pos, n, self.remaining()));
        }
        self.pos += n;
        Ok(())
    }
}

/// A growing byte buffer that writes big-endian primitives and pads to blocks.
#[derive(Debug, Default)]
pub struct ByteSink {
    buf: Vec<u8>,
}

impl ByteSink {
    /// A fresh, empty sink.
    #[inline]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Current length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Append raw bytes verbatim.
    ///
    /// # Arguments
    /// * `bytes` - the bytes to append.
    #[inline]
    pub fn extend(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Append `count` copies of `fill`.
    ///
    /// # Arguments
    /// * `fill` - the byte to repeat.
    /// * `count` - how many copies to append.
    #[inline]
    pub fn pad(&mut self, fill: u8, count: usize) {
        self.buf.resize(self.buf.len() + count, fill);
    }

    /// Pad with `fill` up to the next [`BLOCK_SIZE`] boundary (no-op if aligned).
    ///
    /// # Arguments
    /// * `fill` - the byte to pad with (spaces for headers, zeros for data).
    #[inline]
    pub fn pad_to_block(&mut self, fill: u8) {
        let rem = self.buf.len() % BLOCK_SIZE;
        if rem != 0 {
            self.pad(fill, BLOCK_SIZE - rem);
        }
    }

    /// Consume the sink and return the accumulated bytes.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_rounding() {
        assert_eq!(blocks_for(0), 0);
        assert_eq!(blocks_for(1), 1);
        assert_eq!(blocks_for(BLOCK_SIZE), 1);
        assert_eq!(blocks_for(BLOCK_SIZE + 1), 2);
        assert_eq!(padded_len(1), BLOCK_SIZE);
        assert_eq!(padded_len(BLOCK_SIZE), BLOCK_SIZE);
        assert_eq!(padded_len(BLOCK_SIZE + 1), 2 * BLOCK_SIZE);
    }

    #[test]
    fn cursor_take_and_skip() {
        let data: Vec<u8> = (0..10u8).collect();
        let mut cur = ByteCursor::new(&data);
        assert_eq!(cur.take(3).unwrap(), &[0, 1, 2]);
        assert_eq!(cur.pos(), 3);
        cur.skip(2).unwrap();
        assert_eq!(cur.take(2).unwrap(), &[5, 6]);
        assert_eq!(cur.remaining(), 3);
        assert!(cur.take(4).is_err());
    }

    #[test]
    fn sink_pads_to_block() {
        let mut sink = ByteSink::new();
        sink.extend(b"AB");
        sink.pad_to_block(HEADER_PAD);
        let bytes = sink.into_bytes();
        assert_eq!(bytes.len(), BLOCK_SIZE);
        assert_eq!(&bytes[..2], b"AB");
        assert!(bytes[2..].iter().all(|&b| b == HEADER_PAD));
    }

    #[test]
    fn sink_no_pad_when_aligned() {
        let mut sink = ByteSink::new();
        sink.pad(DATA_PAD, BLOCK_SIZE);
        sink.pad_to_block(DATA_PAD);
        assert_eq!(sink.len(), BLOCK_SIZE);
    }

    #[test]
    fn cursor_metadata() {
        let data = [1u8, 2, 3];
        let mut c = ByteCursor::new(&data);
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
        assert!(!c.at_end());
        assert!(c.take_block().is_err()); // short buffer
        c.skip(3).unwrap();
        assert!(c.at_end());
        assert!(c.skip(1).is_err());

        let empty: [u8; 0] = [];
        assert!(ByteCursor::new(&empty).is_empty());
    }

    #[test]
    fn sink_metadata() {
        let mut s = ByteSink::new();
        assert!(s.is_empty());
        s.extend(b"x");
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
    }
}
