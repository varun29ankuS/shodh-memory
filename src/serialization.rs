//! Centralized serialization module for shodh-memory.
//!
//! All RocksDB serialization flows through this module. The canonical format
//! is **postcard** (v1.x, stable wire format). Legacy bincode/msgpack data is
//! readable via [`try_decode`] during the transition window and via the
//! `shodh-memory-server migrate` subcommand.
//!
//! # Format Discrimination
//!
//! Postcard can silently decode bincode data with **wrong values** (the varint
//! encodings are similar enough that postcard "succeeds" but produces garbage).
//! We therefore use explicit format tags to distinguish new data from legacy:
//!
//! ## Memory records (SHO envelope)
//!
//! ```text
//! [S H O] [version] [payload ...] [CRC32-LE]
//!  3 bytes   1 byte    N bytes      4 bytes
//! ```
//!
//! - Version 1: payload is bincode 2.x (legacy)
//! - Version 2: payload is postcard (current)
//!
//! ## Non-Memory records (format tag)
//!
//! ```text
//! [0x50 0x02] [postcard payload ...]
//!   2 bytes          N bytes
//! ```
//!
//! Records without the `[0x50 0x02]` prefix are legacy bincode/msgpack.

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

use crate::memory::storage::{crc32_simple, STORAGE_MAGIC};

/// Current SHO envelope version (postcard payload).
pub const SHO_VERSION_POSTCARD: u8 = 2;

/// Legacy SHO envelope version (bincode 2.x payload).
pub const SHO_VERSION_BINCODE2: u8 = 1;

/// 2-byte format tag prepended to non-Memory postcard records.
/// `[0x50, 0x02]` = 'P', version 2.
const FORMAT_TAG: [u8; 2] = [0x50, 0x02];
const FORMAT_TAG_LEN: usize = FORMAT_TAG.len();

/// SHO envelope overhead constants.
const SHO_HEADER_LEN: usize = 3 + 1; // magic (3) + version (1)
const SHO_TRAILER_LEN: usize = 4; // CRC32-LE

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Serialize a value to postcard bytes with a 2-byte format tag prefix.
///
/// Used for all non-Memory RocksDB records (graph, facts, lineage, etc.).
/// Single allocation: reserves space for the tag, then serializes in-place.
pub fn encode<T: Serialize + ?Sized>(val: &T) -> Result<Vec<u8>> {
    let payload =
        postcard::to_allocvec(val).map_err(|e| anyhow::anyhow!("postcard encode: {e}"))?;
    // Single allocation: prepend 2-byte tag by building final buffer once.
    // Vec::with_capacity + extend is one alloc; the payload Vec is consumed.
    let mut buf = Vec::with_capacity(FORMAT_TAG_LEN + payload.len());
    buf.extend_from_slice(&FORMAT_TAG);
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Serialize a value to raw postcard bytes (no format tag).
///
/// Used internally by `encode_sho` where the SHO envelope provides format info.
pub fn encode_raw<T: Serialize + ?Sized>(val: &T) -> Result<Vec<u8>> {
    postcard::to_allocvec(val).map_err(|e| anyhow::anyhow!("postcard encode: {e}"))
}

/// Serialize a value to postcard bytes wrapped in a SHO v2 envelope.
///
/// Used exclusively for Memory records in `MemoryStorage::store_inner()`.
/// Single allocation: serializes payload, then builds the envelope in one buffer.
pub fn encode_sho<T: Serialize + ?Sized>(val: &T) -> Result<Vec<u8>> {
    let payload = encode_raw(val)?;
    // Build envelope in one allocation: [SHO][v2][payload][CRC32-LE]
    let total = SHO_HEADER_LEN + payload.len() + SHO_TRAILER_LEN;
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(STORAGE_MAGIC);
    buf.push(SHO_VERSION_POSTCARD);
    buf.extend_from_slice(&payload);
    let crc = crc32_simple(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Deserialize raw postcard bytes (no format tag).
///
/// Used for SHO v2 payloads where the envelope provides format discrimination.
pub fn decode_raw<T: DeserializeOwned>(data: &[u8]) -> Result<T> {
    postcard::from_bytes(data).map_err(|e| anyhow::anyhow!("postcard decode: {e}"))
}

/// Deserialize a tagged postcard record. Strips the 2-byte format tag.
///
/// Returns an error if the data doesn't have the format tag or postcard fails.
pub fn decode<T: DeserializeOwned>(data: &[u8]) -> Result<T> {
    if has_format_tag(data) {
        postcard::from_bytes(&data[2..]).map_err(|e| anyhow::anyhow!("postcard decode: {e}"))
    } else {
        Err(anyhow::anyhow!("missing postcard format tag"))
    }
}

/// Decode a non-Memory record: if it has the postcard format tag, decode as
/// postcard; otherwise fall back to bincode 2.x with 10 MB allocation limit.
///
/// Returns `(value, needs_migration)` where `needs_migration = true` means
/// the record was in legacy bincode format.
pub fn try_decode<T: DeserializeOwned>(data: &[u8]) -> Result<(T, bool)> {
    if has_format_tag(data) {
        let val = postcard::from_bytes::<T>(&data[2..])
            .map_err(|e| anyhow::anyhow!("postcard decode (tagged): {e}"))?;
        return Ok((val, false));
    }

    // Legacy: bincode 2.x with safe allocation limit
    let (val, _): (T, _) = bincode::serde::decode_from_slice(data, crate::bincode_safe_config())
        .map_err(|e| anyhow::anyhow!("bincode decode (legacy): {e}"))?;
    Ok((val, true))
}

/// Check if data starts with the postcard format tag.
#[inline]
fn has_format_tag(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == FORMAT_TAG[0] && data[1] == FORMAT_TAG[1]
}

/// Public version of [`has_format_tag`] for use by the migration module.
#[inline]
pub fn has_format_tag_pub(data: &[u8]) -> bool {
    has_format_tag(data)
}

// ---------------------------------------------------------------------------
// SHO Envelope
// ---------------------------------------------------------------------------

/// Wrap a postcard payload in a SHO v2 envelope: `[SHO][2][payload][CRC32-LE]`.
pub fn wrap_sho_v2(payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(SHO_HEADER_LEN + payload.len() + SHO_TRAILER_LEN);
    buf.extend_from_slice(STORAGE_MAGIC);
    buf.push(SHO_VERSION_POSTCARD);
    buf.extend_from_slice(payload);
    let crc = crc32_simple(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

/// Unwrap a SHO envelope, returning `(version, payload)`.
///
/// Returns `None` if the data does not have a valid SHO header
/// (less than 8 bytes or missing magic).
pub fn unwrap_sho(data: &[u8]) -> Option<(u8, &[u8])> {
    if data.len() < 8 || &data[0..3] != STORAGE_MAGIC {
        return None;
    }
    let version = data[3];
    let payload_end = data.len() - 4;
    let stored_crc = u32::from_le_bytes([
        data[payload_end],
        data[payload_end + 1],
        data[payload_end + 2],
        data[payload_end + 3],
    ]);
    let computed_crc = crc32_simple(&data[..payload_end]);
    if stored_crc != computed_crc {
        tracing::warn!(
            stored_crc = format_args!("{stored_crc:08x}"),
            computed_crc = format_args!("{computed_crc:08x}"),
            "SHO envelope checksum mismatch — rejecting corrupted payload"
        );
        return None;
    }
    Some((version, &data[4..payload_end]))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_simple() {
        let val: Vec<u32> = vec![1, 2, 3, 42];
        let bytes = encode(&val).unwrap();
        let decoded: Vec<u32> = decode(&bytes).unwrap();
        assert_eq!(val, decoded);
    }

    #[test]
    fn test_format_tag_present() {
        let bytes = encode(&42u32).unwrap();
        assert!(has_format_tag(&bytes));
        assert_eq!(bytes[0], 0x50);
        assert_eq!(bytes[1], 0x02);
    }

    #[test]
    fn test_decode_rejects_untagged() {
        // Raw postcard without tag should fail decode()
        let raw = postcard::to_allocvec(&42u32).unwrap();
        assert!(decode::<u32>(&raw).is_err());
    }

    #[test]
    fn test_sho_envelope_round_trip() {
        let payload = b"hello postcard";
        let envelope = wrap_sho_v2(payload);
        let (version, extracted) = unwrap_sho(&envelope).unwrap();
        assert_eq!(version, SHO_VERSION_POSTCARD);
        assert_eq!(extracted, payload);
    }

    #[test]
    fn test_sho_envelope_no_magic() {
        assert!(unwrap_sho(b"NOT_SHO").is_none());
        assert!(unwrap_sho(b"short").is_none());
    }

    #[test]
    fn test_try_decode_tagged_postcard() {
        let val: (u32, String) = (42, "test".to_string());
        let bytes = encode(&val).unwrap();
        let (decoded, needs_migration): ((u32, String), bool) = try_decode(&bytes).unwrap();
        assert_eq!(decoded, val);
        assert!(!needs_migration);
    }

    #[test]
    fn test_try_decode_bincode_legacy() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Complex {
            a: u64,
            b: String,
            c: Vec<u32>,
            d: std::collections::HashMap<String, i32>,
        }
        let mut map = std::collections::HashMap::new();
        map.insert("key".to_string(), -7);
        let val = Complex {
            a: 999_999_999,
            b: "hello world".to_string(),
            c: vec![100, 200, 300],
            d: map,
        };
        // Encode with bincode (no format tag)
        let bytes = bincode::serde::encode_to_vec(&val, bincode::config::standard()).unwrap();
        assert!(
            !has_format_tag(&bytes),
            "bincode should not have format tag"
        );
        let (decoded, needs_migration): (Complex, bool) = try_decode(&bytes).unwrap();
        assert_eq!(decoded, val);
        assert!(needs_migration);
    }

    #[test]
    fn test_encode_sho_decode_round_trip() {
        let val: (u64, String) = (12345, "sho test".to_string());
        let envelope = encode_sho(&val).unwrap();
        let (version, payload) = unwrap_sho(&envelope).unwrap();
        assert_eq!(version, SHO_VERSION_POSTCARD);
        let decoded: (u64, String) = postcard::from_bytes(payload).unwrap();
        assert_eq!(decoded, val);
    }
}
