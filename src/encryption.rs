// Apache-2.0 contribution to shodh-memory by Portll (original author).
// Mirrors the implementation in Portll's `veld` project @ dbc9036 (BUSL-1.1);
// contributed here under Apache-2.0 — no BUSL-1.1 terms attach to this file.
// Adapted for shodh-memory SHODH_* environment names.
//!
//! Record-level encryption for memory payloads.
//!
//! Encrypts the serialized `Memory` record before storage and decrypts it on
//! read, using AES-256-GCM authenticated encryption. The whole serialized
//! record — content, summary, tags, entity refs, embeddings, metadata — is
//! opaque at rest **in the primary column family**, without requiring full
//! database encryption.
//!
//! ## Scope — what is NOT covered
//!
//! Secondary indexes are not encrypted. The `memory_index` column family stores
//! derived lookup keys in plaintext (e.g. `tag:<tag>:<id>`, `entity:<name>:<id>`,
//! and date/type/importance/geo/action keys), so the existence and values of
//! tags, entity refs, timestamps, etc. remain observable on disk even when
//! record encryption is enabled — an index scan can confirm "a memory tagged X
//! exists" without ever touching ciphertext. Sibling column families (feedback,
//! files, prospective, todos) and any separately-keyed embedding blobs are
//! likewise plaintext. See `SECURITY.md` for the full threat model. Blinding the
//! index is tracked as separate, deferred work.
//!
//! # Key Management
//!
//! The encryption key is sourced from the `SHODH_ENCRYPTION_KEY` environment variable.
//! Accepted formats:
//! - 64-character hex string (32 bytes decoded)
//! - 44-character base64 string (32 bytes decoded)
//!
//! When the env var is unset or empty, encryption is disabled and records are stored
//! as plaintext (backward compatible).
//!
//! # Wire Format
//!
//! Encrypted records are stored as: `[ENC\0 marker][12-byte nonce][ciphertext+tag]`
//! The nonce is randomly generated per encryption operation for semantic security.

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, Key, Nonce};
use anyhow::{anyhow, Context, Result};
use zeroize::{Zeroize, Zeroizing};

/// AES-256-GCM nonce size in bytes (96 bits per NIST SP 800-38D).
const NONCE_SIZE: usize = 12;

/// Minimum encrypted payload size after the marker: nonce (12) + GCM tag (16).
///
/// AES-GCM permits empty plaintexts, so a valid encrypted blob may contain no
/// ciphertext bytes beyond the authentication tag.
const MIN_ENCRYPTED_SIZE: usize = NONCE_SIZE + 16;

/// Prefix marker for encrypted record blobs.
///
/// The NUL byte makes accidental collision with normal textual/base64 payloads
/// vanishingly unlikely, and the marker lives outside the authenticated
/// ciphertext so the storage layer can route encrypted and legacy plaintext
/// records before decoding.
const ENCRYPTED_MARKER: &[u8; 4] = b"ENC\x00";

/// Record-level encryptor using AES-256-GCM.
///
/// Holds a validated 256-bit key and provides encrypt/decrypt operations.
/// Key material is zeroized on drop: `key_bytes` is wrapped in `Zeroizing`
/// (explicit memset-on-drop), and `Aes256Gcm` zeroizes its expanded key
/// schedule via its own `ZeroizeOnDrop` impl when the `zeroize` feature is
/// enabled (the default in aes-gcm 0.10).
#[derive(Clone)]
pub struct FieldEncryptor {
    cipher: Aes256Gcm,
    /// Raw key bytes retained so they can be zeroized when this struct is dropped.
    key_bytes: Zeroizing<[u8; 32]>,
}

impl Drop for FieldEncryptor {
    fn drop(&mut self) {
        // `Zeroizing` memsets key_bytes to zero on drop automatically.
        // This explicit zeroize call is belt-and-suspenders for the cipher's
        // expanded key schedule in case the aes-gcm `zeroize` feature is disabled.
        self.key_bytes.zeroize();
    }
}

impl FieldEncryptor {
    /// Create a new encryptor from a raw 32-byte key.
    pub fn new(key_bytes: &[u8; 32]) -> Self {
        let key = Key::<Aes256Gcm>::from_slice(key_bytes);
        Self {
            cipher: Aes256Gcm::new(key),
            key_bytes: Zeroizing::new(*key_bytes),
        }
    }

    /// Try to create an encryptor from the `SHODH_ENCRYPTION_KEY` environment variable.
    ///
    /// Returns `Ok(Some(encryptor))` if the key is valid, `Ok(None)` if the env var
    /// is unset or empty (encryption disabled), or `Err` if the key is malformed.
    pub fn from_env() -> Result<Option<Self>> {
        let key_str = match std::env::var("SHODH_ENCRYPTION_KEY") {
            Ok(s) if !s.is_empty() => s,
            _ => return Ok(None),
        };

        let key_bytes = Self::decode_key(&key_str)?;
        tracing::info!("Record-level memory encryption enabled (AES-256-GCM)");
        Ok(Some(Self::new(&key_bytes)))
    }

    /// Stable non-secret key fingerprint used to detect key loss/mismatch.
    ///
    /// This is the first 4 bytes of SHA-256(key). It is not used for
    /// cryptographic authentication; AES-GCM does that. It is only a startup
    /// sentinel so operators get an explicit error instead of silent ciphertext
    /// reads when the env key is absent or wrong.
    pub fn key_fingerprint(&self) -> [u8; 4] {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(self.key_bytes.as_slice());
        [digest[0], digest[1], digest[2], digest[3]]
    }

    /// Decode a key from hex (64 chars) or base64 (44 chars) into a 32-byte array.
    fn decode_key(key_str: &str) -> Result<[u8; 32]> {
        let trimmed = key_str.trim();

        // Try hex first (64 hex chars = 32 bytes)
        if trimmed.len() == 64 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
            let decoded = hex::decode(trimmed).context("Invalid hex in SHODH_ENCRYPTION_KEY")?;
            let mut key = [0u8; 32];
            key.copy_from_slice(&decoded);
            return Ok(key);
        }

        // Try base64 (44 chars with padding = 32 bytes, or 43 without padding)
        use base64::Engine;
        let engine = base64::engine::general_purpose::STANDARD;
        if let Ok(decoded) = engine.decode(trimmed) {
            if decoded.len() == 32 {
                let mut key = [0u8; 32];
                key.copy_from_slice(&decoded);
                return Ok(key);
            }
        }

        // Also try base64 URL-safe variant
        let url_engine = base64::engine::general_purpose::URL_SAFE;
        if let Ok(decoded) = url_engine.decode(trimmed) {
            if decoded.len() == 32 {
                let mut key = [0u8; 32];
                key.copy_from_slice(&decoded);
                return Ok(key);
            }
        }

        Err(anyhow!(
            "SHODH_ENCRYPTION_KEY must be a 64-char hex string or 44-char base64 string (32 bytes). \
             Got {} chars.",
            trimmed.len()
        ))
    }

    /// Encrypt an arbitrary byte payload.
    ///
    /// Returns bytes in the format: `[ENCRYPTED_MARKER (4)][nonce (12)][ciphertext+tag]`.
    /// The nonce is randomly generated for each call.
    pub fn encrypt_bytes(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = self
            .cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| anyhow!("AES-256-GCM encryption failed: {}", e))?;

        let mut output = Vec::with_capacity(ENCRYPTED_MARKER.len() + NONCE_SIZE + ciphertext.len());
        output.extend_from_slice(ENCRYPTED_MARKER);
        output.extend_from_slice(&nonce);
        output.extend_from_slice(&ciphertext);
        Ok(output)
    }

    /// Encrypt a plaintext content string. UTF-8 convenience wrapper for tests
    /// and non-memory secret fields.
    pub fn encrypt_content(&self, plaintext: &str) -> Result<Vec<u8>> {
        self.encrypt_bytes(plaintext.as_bytes())
    }

    /// Decrypt an encrypted blob back to plaintext bytes.
    ///
    /// Expects the format: `[ENCRYPTED_MARKER (4)][nonce (12)][ciphertext+tag]`.
    /// Returns an error if the data is too short, the marker is missing, or
    /// decryption/authentication fails.
    pub fn decrypt_bytes(&self, data: &[u8]) -> Result<Vec<u8>> {
        let min_size = ENCRYPTED_MARKER.len() + MIN_ENCRYPTED_SIZE;
        if data.len() < min_size {
            return Err(anyhow!(
                "Encrypted data too short: {} bytes (minimum {})",
                data.len(),
                min_size
            ));
        }

        if &data[..4] != ENCRYPTED_MARKER {
            return Err(anyhow!(
                "Missing encryption marker — data may not be encrypted"
            ));
        }

        let nonce_bytes = &data[4..4 + NONCE_SIZE];
        let ciphertext = &data[4 + NONCE_SIZE..];

        let nonce = Nonce::from_slice(nonce_bytes);
        self.cipher.decrypt(nonce, ciphertext).map_err(|e| {
            anyhow!(
                "AES-256-GCM decryption failed (wrong key or corrupted data): {}",
                e
            )
        })
    }

    /// Decrypt an encrypted content blob back to a plaintext string. UTF-8
    /// convenience wrapper for tests and non-memory secret fields.
    pub fn decrypt_content(&self, data: &[u8]) -> Result<String> {
        let plaintext_bytes = self.decrypt_bytes(data)?;
        String::from_utf8(plaintext_bytes).context("Decrypted content is not valid UTF-8")
    }

    /// Check whether a byte slice looks like encrypted content (has the marker prefix).
    pub fn is_encrypted(data: &[u8]) -> bool {
        data.len() >= ENCRYPTED_MARKER.len() && &data[..4] == ENCRYPTED_MARKER
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn test_key() -> [u8; 32] {
        // Deterministic test key (NOT for production use)
        let mut key = [0u8; 32];
        for (i, byte) in key.iter_mut().enumerate() {
            *byte = i as u8;
        }
        key
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let enc = FieldEncryptor::new(&test_key());
        let plaintext = "This is sensitive memory content about the user's preferences.";

        let encrypted = enc.encrypt_content(plaintext).unwrap();
        let decrypted = enc.decrypt_content(&encrypted).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_bytes_roundtrip() {
        let enc = FieldEncryptor::new(&test_key());
        let blob: Vec<u8> = (0u8..=255).cycle().take(1000).collect();

        let encrypted = enc.encrypt_bytes(&blob).unwrap();
        assert!(FieldEncryptor::is_encrypted(&encrypted));
        assert_ne!(encrypted, blob);
        assert_eq!(enc.decrypt_bytes(&encrypted).unwrap(), blob);
    }

    #[test]
    fn key_fingerprint_is_stable_and_short() {
        let enc1 = FieldEncryptor::new(&test_key());
        let enc2 = FieldEncryptor::new(&test_key());
        let mut other = test_key();
        other[31] ^= 0xff;
        let enc3 = FieldEncryptor::new(&other);

        assert_eq!(enc1.key_fingerprint(), enc2.key_fingerprint());
        assert_ne!(enc1.key_fingerprint(), enc3.key_fingerprint());
    }

    #[test]
    fn encrypted_content_has_marker() {
        let enc = FieldEncryptor::new(&test_key());
        let encrypted = enc.encrypt_content("test").unwrap();

        assert!(FieldEncryptor::is_encrypted(&encrypted));
        assert!(!FieldEncryptor::is_encrypted(b"plain text content"));
    }

    #[test]
    fn different_nonces_produce_different_ciphertext() {
        let enc = FieldEncryptor::new(&test_key());
        let plaintext = "same content";

        let encrypted1 = enc.encrypt_content(plaintext).unwrap();
        let encrypted2 = enc.encrypt_content(plaintext).unwrap();

        // Ciphertexts should differ due to random nonces (semantic security)
        assert_ne!(encrypted1, encrypted2);

        // But both decrypt to the same plaintext
        assert_eq!(enc.decrypt_content(&encrypted1).unwrap(), plaintext);
        assert_eq!(enc.decrypt_content(&encrypted2).unwrap(), plaintext);
    }

    #[test]
    fn wrong_key_fails_decryption() {
        let enc1 = FieldEncryptor::new(&test_key());
        let mut wrong_key = test_key();
        wrong_key[0] ^= 0xFF; // Flip a byte
        let enc2 = FieldEncryptor::new(&wrong_key);

        let encrypted = enc1.encrypt_content("secret").unwrap();
        assert!(enc2.decrypt_content(&encrypted).is_err());
    }

    #[test]
    fn truncated_data_fails() {
        let enc = FieldEncryptor::new(&test_key());
        let encrypted = enc.encrypt_content("test").unwrap();

        // Truncate to just the marker
        assert!(enc.decrypt_content(&encrypted[..10]).is_err());
    }

    #[test]
    fn non_encrypted_data_fails() {
        let enc = FieldEncryptor::new(&test_key());
        assert!(enc.decrypt_content(b"not encrypted data at all").is_err());
    }

    #[test]
    fn empty_string_roundtrip() {
        let enc = FieldEncryptor::new(&test_key());
        let encrypted = enc.encrypt_content("").unwrap();
        let decrypted = enc.decrypt_content(&encrypted).unwrap();
        assert_eq!(decrypted, "");
    }

    #[test]
    fn unicode_roundtrip() {
        let enc = FieldEncryptor::new(&test_key());
        let plaintext =
            "Memory: user prefers dark mode \u{1f30d} \u{2014} context: \u{e4}\u{fc}\u{f6}\u{df}";
        let encrypted = enc.encrypt_content(plaintext).unwrap();
        let decrypted = enc.decrypt_content(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn decode_hex_key() {
        let hex_key = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
        let decoded = FieldEncryptor::decode_key(hex_key).unwrap();
        assert_eq!(decoded, test_key());
    }

    #[test]
    fn decode_base64_key() {
        use base64::Engine;
        let key = test_key();
        let b64 = base64::engine::general_purpose::STANDARD.encode(key);
        let decoded = FieldEncryptor::decode_key(&b64).unwrap();
        assert_eq!(decoded, key);
    }

    #[test]
    fn invalid_key_length_fails() {
        assert!(FieldEncryptor::decode_key("too_short").is_err());
        assert!(FieldEncryptor::decode_key("aabbccdd").is_err());
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SHODH_ENCRYPTION_KEY should not be set in test env
        std::env::remove_var("SHODH_ENCRYPTION_KEY");
        let result = FieldEncryptor::from_env().unwrap();
        assert!(result.is_none());
    }
}
