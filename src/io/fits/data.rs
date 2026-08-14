// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The typed data array of an image HDU, decoded per `BITPIX`.
//!
//! [`FitsData`] holds the raw element buffer in its native type; FITS stores
//! elements big-endian. [`FitsData::as_f64`] applies the physical relation
//! `physical = BZERO + BSCALE·array` and maps an integer `BLANK` to `NaN`;
//! [`FitsData::from_f64`] inverts it for a chosen `BITPIX`.

use super::error::FitsError;
use super::block::ByteSink;

/// A `BITPIX`-typed element buffer for one image HDU.
#[derive(Debug, Clone, PartialEq)]
pub enum FitsData {
    /// `BITPIX = 8`: unsigned bytes.
    U8(Vec<u8>),
    /// `BITPIX = 16`: 16-bit two's-complement integers.
    I16(Vec<i16>),
    /// `BITPIX = 32`: 32-bit two's-complement integers.
    I32(Vec<i32>),
    /// `BITPIX = 64`: 64-bit two's-complement integers.
    I64(Vec<i64>),
    /// `BITPIX = -32`: IEEE-754 single-precision floats.
    F32(Vec<f32>),
    /// `BITPIX = -64`: IEEE-754 double-precision floats.
    F64(Vec<f64>),
}

/// The set of valid `BITPIX` values.
pub const VALID_BITPIX: [i32; 6] = [8, 16, 32, 64, -32, -64];

/// Bytes per element for a given `BITPIX`, or an error for an invalid value.
///
/// # Arguments
/// * `bitpix` - the `BITPIX` header value.
pub fn bytes_per_element(bitpix: i32) -> Result<usize, FitsError> {
    match bitpix {
        8 => Ok(1),
        16 => Ok(2),
        32 | -32 => Ok(4),
        64 | -64 => Ok(8),
        other => Err(FitsError::BadBitpix(other as i64)),
    }
}

impl FitsData {
    /// The `BITPIX` value corresponding to this buffer's element type.
    pub fn bitpix(&self) -> i32 {
        match self {
            FitsData::U8(_) => 8,
            FitsData::I16(_) => 16,
            FitsData::I32(_) => 32,
            FitsData::I64(_) => 64,
            FitsData::F32(_) => -32,
            FitsData::F64(_) => -64,
        }
    }

    /// The number of elements in the buffer.
    pub fn len(&self) -> usize {
        match self {
            FitsData::U8(v) => v.len(),
            FitsData::I16(v) => v.len(),
            FitsData::I32(v) => v.len(),
            FitsData::I64(v) => v.len(),
            FitsData::F32(v) => v.len(),
            FitsData::F64(v) => v.len(),
        }
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decode `count` big-endian elements of type `bitpix` from `bytes`.
    ///
    /// # Arguments
    /// * `bitpix` - the element type selector.
    /// * `bytes` - the raw data segment (at least `count · bytes_per_element`).
    /// * `count` - the number of elements to decode.
    ///
    /// Returns the decoded [`FitsData`].
    pub fn decode(bitpix: i32, bytes: &[u8], count: usize) -> Result<Self, FitsError> {
        let esize = bytes_per_element(bitpix)?;
        let need = count * esize;
        if bytes.len() < need {
            return Err(FitsError::structure(format!(
                "data segment too short: need {} bytes, have {}",
                need,
                bytes.len()
            )));
        }
        let chunks = bytes[..need].chunks_exact(esize);
        Ok(match bitpix {
            8 => FitsData::U8(bytes[..need].to_vec()),
            16 => FitsData::I16(
                chunks
                    .map(|c| i16::from_be_bytes([c[0], c[1]]))
                    .collect(),
            ),
            32 => FitsData::I32(
                chunks
                    .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            64 => FitsData::I64(
                chunks
                    .map(|c| {
                        i64::from_be_bytes([
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                        ])
                    })
                    .collect(),
            ),
            -32 => FitsData::F32(
                chunks
                    .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            -64 => FitsData::F64(
                chunks
                    .map(|c| {
                        f64::from_be_bytes([
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                        ])
                    })
                    .collect(),
            ),
            other => return Err(FitsError::BadBitpix(other as i64)),
        })
    }

    /// Encode the buffer big-endian into `sink` and zero-pad to a block.
    ///
    /// # Arguments
    /// * `sink` - the output sink to append to.
    pub fn encode(&self, sink: &mut ByteSink) {
        match self {
            FitsData::U8(v) => sink.extend(v),
            FitsData::I16(v) => {
                for &x in v {
                    sink.extend(&x.to_be_bytes());
                }
            }
            FitsData::I32(v) => {
                for &x in v {
                    sink.extend(&x.to_be_bytes());
                }
            }
            FitsData::I64(v) => {
                for &x in v {
                    sink.extend(&x.to_be_bytes());
                }
            }
            FitsData::F32(v) => {
                for &x in v {
                    sink.extend(&x.to_be_bytes());
                }
            }
            FitsData::F64(v) => {
                for &x in v {
                    sink.extend(&x.to_be_bytes());
                }
            }
        }
        sink.pad_to_block(super::block::DATA_PAD);
    }

    /// Physical `f64` values: `BZERO + BSCALE·array`, integer `BLANK → NaN`.
    ///
    /// # Arguments
    /// * `bscale` - the linear scale factor.
    /// * `bzero` - the additive offset.
    /// * `blank` - the integer sentinel mapped to `NaN` (integer types only).
    pub fn as_f64(&self, bscale: f64, bzero: f64, blank: Option<i64>) -> Vec<f64> {
        let scale = |raw: f64| bzero + bscale * raw;
        let int_scale = |raw: i64| match blank {
            Some(b) if b == raw => f64::NAN,
            _ => scale(raw as f64),
        };
        match self {
            FitsData::U8(v) => v.iter().map(|&x| int_scale(x as i64)).collect(),
            FitsData::I16(v) => v.iter().map(|&x| int_scale(x as i64)).collect(),
            FitsData::I32(v) => v.iter().map(|&x| int_scale(x as i64)).collect(),
            FitsData::I64(v) => v.iter().map(|&x| int_scale(x)).collect(),
            FitsData::F32(v) => v.iter().map(|&x| scale(x as f64)).collect(),
            FitsData::F64(v) => v.iter().map(|&x| scale(x)).collect(),
        }
    }

    /// Build a buffer of `bitpix` from physical `f64` values, inverting scaling.
    ///
    /// For integer `bitpix`, values round to the nearest integer after removing
    /// `bzero`/`bscale`, and `NaN` maps to `blank` (defaulting to the type's
    /// minimum). For float `bitpix`, values are stored directly.
    ///
    /// # Arguments
    /// * `bitpix` - the target element type.
    /// * `values` - the physical values, in element order.
    /// * `bscale` - the scale factor to remove.
    /// * `bzero` - the offset to remove.
    /// * `blank` - the integer sentinel for `NaN` (integer types).
    pub fn from_f64(
        bitpix: i32,
        values: &[f64],
        bscale: f64,
        bzero: f64,
        blank: Option<i64>,
    ) -> Result<Self, FitsError> {
        let to_raw = |x: f64| (x - bzero) / bscale;
        Ok(match bitpix {
            8 => FitsData::U8(
                values
                    .iter()
                    .map(|&x| clamp_int(to_raw(x), blank, u8::MIN as i64, u8::MAX as i64) as u8)
                    .collect(),
            ),
            16 => FitsData::I16(
                values
                    .iter()
                    .map(|&x| {
                        clamp_int(to_raw(x), blank, i16::MIN as i64, i16::MAX as i64) as i16
                    })
                    .collect(),
            ),
            32 => FitsData::I32(
                values
                    .iter()
                    .map(|&x| {
                        clamp_int(to_raw(x), blank, i32::MIN as i64, i32::MAX as i64) as i32
                    })
                    .collect(),
            ),
            64 => FitsData::I64(
                values
                    .iter()
                    .map(|&x| clamp_int(to_raw(x), blank, i64::MIN, i64::MAX))
                    .collect(),
            ),
            -32 => FitsData::F32(values.iter().map(|&x| to_raw(x) as f32).collect()),
            -64 => FitsData::F64(values.iter().map(|&x| to_raw(x)).collect()),
            other => return Err(FitsError::BadBitpix(other as i64)),
        })
    }
}

/// Round a raw value to an integer, sending `NaN` to `blank` (or `lo`) and
/// clamping into `[lo, hi]`.
fn clamp_int(raw: f64, blank: Option<i64>, lo: i64, hi: i64) -> i64 {
    if raw.is_nan() {
        return blank.unwrap_or(lo);
    }
    let r = raw.round();
    if r <= lo as f64 {
        lo
    } else if r >= hi as f64 {
        hi
    } else {
        r as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_encode_roundtrip_all_bitpix() {
        let cases: [FitsData; 6] = [
            FitsData::U8(vec![0, 1, 200, 255]),
            FitsData::I16(vec![-32768, -1, 0, 32767]),
            FitsData::I32(vec![i32::MIN, -1, 0, i32::MAX]),
            FitsData::I64(vec![i64::MIN, -1, 0, i64::MAX]),
            FitsData::F32(vec![-1.5, 0.0, 3.25, 1e30]),
            FitsData::F64(vec![-1.5, 0.0, 3.25, 1e300]),
        ];
        for data in cases {
            let mut sink = ByteSink::new();
            data.encode(&mut sink);
            let bytes = sink.into_bytes();
            assert_eq!(bytes.len() % 2880, 0);
            let decoded = FitsData::decode(data.bitpix(), &bytes, data.len()).unwrap();
            assert_eq!(decoded, data);
        }
    }

    #[test]
    fn as_f64_applies_scaling() {
        // Unsigned-16 convention: BZERO = 32768.
        let d = FitsData::I16(vec![-32768, 0, 32767]);
        let phys = d.as_f64(1.0, 32768.0, None);
        assert_eq!(phys, vec![0.0, 32768.0, 65535.0]);
    }

    #[test]
    fn blank_maps_to_nan() {
        let d = FitsData::I32(vec![10, -1, 20]);
        let phys = d.as_f64(1.0, 0.0, Some(-1));
        assert_eq!(phys[0], 10.0);
        assert!(phys[1].is_nan());
        assert_eq!(phys[2], 20.0);
    }

    #[test]
    fn from_f64_inverts_scaling() {
        let values = [0.0, 32768.0, 65535.0];
        let d = FitsData::from_f64(16, &values, 1.0, 32768.0, None).unwrap();
        assert_eq!(d, FitsData::I16(vec![-32768, 0, 32767]));
    }

    #[test]
    fn from_f64_nan_to_blank() {
        let values = [1.0, f64::NAN, 3.0];
        let d = FitsData::from_f64(32, &values, 1.0, 0.0, Some(-999)).unwrap();
        assert_eq!(d, FitsData::I32(vec![1, -999, 3]));
    }

    #[test]
    fn invalid_bitpix_is_rejected() {
        assert!(bytes_per_element(7).is_err());
        assert!(FitsData::decode(7, &[0; 8], 1).is_err());
        assert!(FitsData::from_f64(7, &[1.0], 1.0, 0.0, None).is_err());
    }

    #[test]
    fn decode_short_segment_errors() {
        // Need 4 bytes for one i32, give 2.
        assert!(FitsData::decode(32, &[0, 0], 1).is_err());
    }

    #[test]
    fn is_empty_reports_element_count() {
        assert!(FitsData::F64(vec![]).is_empty());
        assert!(!FitsData::F64(vec![1.0]).is_empty());
    }

    #[test]
    fn as_f64_and_from_f64_u8_and_i64() {
        let u8d = FitsData::U8(vec![0, 128, 255]);
        assert_eq!(u8d.as_f64(1.0, 0.0, None), vec![0.0, 128.0, 255.0]);
        assert_eq!(
            FitsData::from_f64(8, &[0.0, 128.0, 255.0], 1.0, 0.0, None).unwrap(),
            u8d
        );

        let i64d = FitsData::I64(vec![-5, 0, 42]);
        assert_eq!(i64d.as_f64(2.0, 1.0, None), vec![-9.0, 1.0, 85.0]);
        assert_eq!(
            FitsData::from_f64(64, &[-9.0, 1.0, 85.0], 2.0, 1.0, None).unwrap(),
            i64d
        );
    }

    #[test]
    fn from_f64_clamps_out_of_range() {
        // Values beyond i16 range clamp to its bounds; NaN falls back to lo.
        let values = [-1e9, 1e9, f64::NAN];
        let d = FitsData::from_f64(16, &values, 1.0, 0.0, None).unwrap();
        assert_eq!(
            d,
            FitsData::I16(vec![i16::MIN, i16::MAX, i16::MIN])
        );
    }
}
