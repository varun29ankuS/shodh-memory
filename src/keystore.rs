//! Encryption-at-rest v2 — envelope keystore (crypto core, phase P1).
//!
//! See `docs/encryption-v2-design.md`. This module is the pure crypto core:
//! envelope key hierarchy (a master key / KEK wrapping per-epoch data keys /
//! DEKs), an Argon2id passphrase unseal path, and the serialisable keystore
//! container with a key fingerprint + crypto/schema version gates.
//!
//! P1 deliberately does NOT wire into storage (that is P2) and ships only the
//! passphrase unseal provider (keychain/KMS/recovery are P5). The structure
//! (multi-wrap `kek_wraps`, epoched `deks`) is in place so later phases add
//! providers and rotation without reshaping the on-disk keystore.
//!
//! Threat model and rationale: `docs/encryption-v2-design.md` §1, §11.

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use anyhow::{anyhow, Context, Result};
use argon2::{Algorithm, Argon2, Params, Version};
use base64::Engine;
use hmac::{Hmac, Mac};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

/// Envelope wire/format version (records + keystore). Bumped on format changes.
pub const CRYPTO_VERSION: u8 = 1;
/// Storage schema version gate for migrations (see design §11 B-5).
pub const SCHEMA_VERSION: u8 = 1;

const KEY_LEN: usize = 32;
const SALT_LEN: usize = 16;
const NONCE_SIZE: usize = 12;

/// AAD bound into the KEK wrap (domain separation / substitution resistance).
const KEK_AAD: &[u8] = b"shodh:keystore:kek:v1";

/// A 256-bit secret (KEK or DEK), memory-zeroized on drop. Never serialised —
/// only its *wrapped* form is persisted.
pub type SecretKey = Zeroizing<[u8; KEY_LEN]>;

fn b64e(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn b64d(s: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .context("invalid base64 in keystore")
}

fn random_key() -> SecretKey {
    let mut k = Zeroizing::new([0u8; KEY_LEN]);
    OsRng.fill_bytes(k.as_mut_slice());
    k
}

fn random_salt() -> [u8; SALT_LEN] {
    let mut s = [0u8; SALT_LEN];
    OsRng.fill_bytes(&mut s);
    s
}

/// Non-secret 4-byte fingerprint of a key: `SHA-256(key)[..4]`. A loss/mismatch
/// tripwire, not an authenticator (see design §6 / §11 B-5).
fn fingerprint(key: &[u8]) -> [u8; 4] {
    let d = Sha256::digest(key);
    [d[0], d[1], d[2], d[3]]
}

/// AAD bound into a DEK wrap — pins the wrap to its epoch so a DEK cannot be
/// silently substituted across epochs.
fn dek_aad(epoch: u32) -> Vec<u8> {
    format!("shodh:keystore:dek:v1:epoch:{epoch}").into_bytes()
}

/// Derive the recovery-code wrapping key. The recovery code is high-entropy, so a
/// fast hash (not Argon2) suffices — there is nothing to brute-force.
fn recovery_key_from_code(code: &str) -> [u8; KEY_LEN] {
    let digest = Sha256::digest(code.trim().as_bytes());
    let mut k = [0u8; KEY_LEN];
    k.copy_from_slice(&digest);
    k
}

/// Generate a high-entropy, transcribable recovery code (192 bits → 48 hex chars).
fn generate_recovery_code() -> String {
    let mut raw = [0u8; 24];
    OsRng.fill_bytes(&mut raw);
    hex::encode(raw)
}

/// AES-256-GCM key-wrap: encrypt `key` under `wrapping_key` with `aad` bound in.
fn wrap_key(
    wrapping_key: &[u8; KEY_LEN],
    key: &[u8],
    aad: &[u8],
    provider: &str,
) -> Result<Wrapped> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(wrapping_key));
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, Payload { msg: key, aad })
        .map_err(|e| anyhow!("AES-256-GCM key wrap failed: {e}"))?;
    Ok(Wrapped {
        provider: provider.to_string(),
        nonce: b64e(nonce.as_slice()),
        ciphertext: b64e(&ciphertext),
    })
}

/// Inverse of [`wrap_key`]; errors on wrong key/passphrase, tamper, or AAD mismatch.
fn unwrap_key(wrapping_key: &[u8; KEY_LEN], w: &Wrapped, aad: &[u8]) -> Result<SecretKey> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(wrapping_key));
    let nonce_bytes = b64d(&w.nonce)?;
    if nonce_bytes.len() != 12 {
        return Err(anyhow!("keystore nonce wrong length"));
    }
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = b64d(&w.ciphertext)?;
    let mut plaintext = cipher
        .decrypt(
            nonce,
            Payload {
                msg: &ciphertext,
                aad,
            },
        )
        .map_err(|e| {
            anyhow!("key unwrap failed (wrong key/passphrase or tampered keystore): {e}")
        })?;
    if plaintext.len() != KEY_LEN {
        return Err(anyhow!("unwrapped key has wrong length"));
    }
    let mut k = Zeroizing::new([0u8; KEY_LEN]);
    k.copy_from_slice(&plaintext);
    plaintext.zeroize(); // wipe the intermediate key-material buffer
    Ok(k)
}

/// Decode a 32-byte key from 64-char hex or 44-char base64.
fn decode_32(s: &str) -> Result<[u8; KEY_LEN]> {
    let t = s.trim();
    let bytes = if t.len() == 64 && t.bytes().all(|b| b.is_ascii_hexdigit()) {
        hex::decode(t).context("invalid hex key")?
    } else {
        b64d(t)?
    };
    if bytes.len() != KEY_LEN {
        return Err(anyhow!(
            "key must decode to {KEY_LEN} bytes, got {}",
            bytes.len()
        ));
    }
    let mut k = [0u8; KEY_LEN];
    k.copy_from_slice(&bytes);
    Ok(k)
}

/// A key-management service that wraps/unwraps the master key (P5). Real cloud
/// KMS (AWS/GCP/Vault) implements this with server-side Encrypt/Decrypt calls;
/// the master key never leaves wrapped form on this host. `LocalAeadKms` is the
/// self-hosted concrete impl; an OS-keychain provider implements it too.
pub trait KmsClient {
    fn kms_encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>>;
    fn kms_decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>>;
}

/// Self-hosted KMS: AES-256-GCM under a wrapping key sourced from
/// `SHODH_KMS_WRAP_KEY` (64-hex / 44-base64). Ciphertext is `[nonce][ct+tag]`.
/// Lets a server unseal unattended without a passphrase prompt while keeping the
/// wrapping key off the data disk (env/secret-store). Swap for a cloud KMS by
/// implementing `KmsClient`.
pub struct LocalAeadKms {
    key: SecretKey,
}

impl LocalAeadKms {
    pub fn new(key: [u8; KEY_LEN]) -> Self {
        Self {
            key: Zeroizing::new(key),
        }
    }

    /// Load from `SHODH_KMS_WRAP_KEY`; `Ok(None)` if unset/empty.
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var("SHODH_KMS_WRAP_KEY") {
            Ok(s) if !s.trim().is_empty() => Ok(Some(Self::new(decode_32(&s)?))),
            _ => Ok(None),
        }
    }
}

impl KmsClient for LocalAeadKms {
    fn kms_encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(self.key.as_slice()));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ct = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| anyhow!("KMS encrypt failed: {e}"))?;
        let mut out = Vec::with_capacity(NONCE_SIZE + ct.len());
        out.extend_from_slice(nonce.as_slice());
        out.extend_from_slice(&ct);
        Ok(out)
    }

    fn kms_decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>> {
        if ciphertext.len() < NONCE_SIZE + 16 {
            return Err(anyhow!("KMS ciphertext too short"));
        }
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(self.key.as_slice()));
        let nonce = Nonce::from_slice(&ciphertext[..NONCE_SIZE]);
        cipher
            .decrypt(nonce, &ciphertext[NONCE_SIZE..])
            .map_err(|e| anyhow!("KMS decrypt failed (wrong KMS key or tampered): {e}"))
    }
}

/// A secret wrapped under some unseal provider (passphrase in P1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wrapped {
    /// Unseal provider id: `"passphrase"` (P1); `"keychain"`/`"kms"`/`"recovery"` later.
    pub provider: String,
    /// base64(12-byte GCM nonce).
    pub nonce: String,
    /// base64(ciphertext+tag).
    pub ciphertext: String,
}

/// Argon2id parameters for deriving an unseal key from a passphrase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KdfParams {
    /// Memory cost in KiB.
    pub m_cost: u32,
    /// Iterations.
    pub t_cost: u32,
    /// Parallelism (lanes).
    pub p_cost: u32,
    /// base64(salt).
    pub salt: String,
}

impl KdfParams {
    /// Strong production defaults (~256 MiB, 3 passes). Tune/upgrade over time.
    pub fn production() -> Self {
        Self {
            m_cost: 256 * 1024,
            t_cost: 3,
            p_cost: 1,
            salt: b64e(&random_salt()),
        }
    }

    /// Weak params for fast tests ONLY (never production).
    #[cfg(test)]
    pub fn fast_for_tests() -> Self {
        Self {
            m_cost: 8,
            t_cost: 1,
            p_cost: 1,
            salt: b64e(&random_salt()),
        }
    }

    /// Derive a 256-bit unseal key from the passphrase under these params.
    fn derive(&self, passphrase: &str) -> Result<SecretKey> {
        let params = Params::new(self.m_cost, self.t_cost, self.p_cost, Some(KEY_LEN))
            .map_err(|e| anyhow!("invalid Argon2 params: {e}"))?;
        let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
        let salt = b64d(&self.salt)?;
        let mut out = Zeroizing::new([0u8; KEY_LEN]);
        argon
            .hash_password_into(passphrase.as_bytes(), &salt, out.as_mut_slice())
            .map_err(|e| anyhow!("Argon2id derivation failed: {e}"))?;
        Ok(out)
    }
}

/// A per-epoch data key, wrapped by the master key (KEK).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DekEntry {
    pub epoch: u32,
    /// DEK wrapped by the KEK (provider id `"kek"`).
    pub wrapped: Wrapped,
    /// `"active"` or `"retired"` (rotation, P4).
    pub state: String,
}

/// Serialisable keystore: the master key wrapped per unseal provider, the
/// epoched data keys wrapped by the master key, and version/fingerprint gates.
/// Holds no plaintext key material.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keystore {
    pub crypto_version: u8,
    pub schema_version: u8,
    pub kdf: KdfParams,
    /// KEK wrapped once per enabled unseal provider (P1: passphrase only).
    pub kek_wraps: Vec<Wrapped>,
    pub deks: Vec<DekEntry>,
    pub active_epoch: u32,
    /// base64(4-byte KEK fingerprint) — detects unseal-to-wrong-key.
    pub kek_fingerprint: String,
}

impl Keystore {
    /// Create a fresh keystore: random KEK + an initial active DEK (epoch 0),
    /// with the KEK wrapped under the passphrase. Returns the keystore only; the
    /// caller persists it and re-unseals to obtain key material.
    pub fn create(passphrase: &str, kdf: KdfParams) -> Result<Self> {
        let kek = random_key();
        let dek = random_key();

        let unseal = kdf.derive(passphrase)?;
        let kek_wrap = wrap_key(&unseal, kek.as_slice(), KEK_AAD, "passphrase")?;
        let dek_wrap = wrap_key(&kek, dek.as_slice(), &dek_aad(0), "kek")?;

        Ok(Self {
            crypto_version: CRYPTO_VERSION,
            schema_version: SCHEMA_VERSION,
            kdf,
            kek_wraps: vec![kek_wrap],
            deks: vec![DekEntry {
                epoch: 0,
                wrapped: dek_wrap,
                state: "active".to_string(),
            }],
            active_epoch: 0,
            kek_fingerprint: b64e(&fingerprint(kek.as_slice())),
        })
    }

    /// Unseal the master key (KEK) via the passphrase provider; verifies the
    /// fingerprint so an unexpected key surfaces as an error, not silent garbage.
    pub fn unseal_with_passphrase(&self, passphrase: &str) -> Result<SecretKey> {
        let unseal = self.kdf.derive(passphrase)?;
        let w = self
            .kek_wraps
            .iter()
            .find(|w| w.provider == "passphrase")
            .ok_or_else(|| anyhow!("keystore has no passphrase wrap"))?;
        let kek = unwrap_key(&unseal, w, KEK_AAD)?;
        if b64e(&fingerprint(kek.as_slice())) != self.kek_fingerprint {
            return Err(anyhow!("keystore KEK fingerprint mismatch after unseal"));
        }
        Ok(kek)
    }

    /// Unwrap the active-epoch data key. Returns `(epoch, dek)`.
    pub fn active_data_key(&self, kek: &SecretKey) -> Result<(u32, SecretKey)> {
        let dek = self.data_key_for_epoch(kek, self.active_epoch)?;
        Ok((self.active_epoch, dek))
    }

    /// Unwrap the data key for a specific epoch (older records may be on prior
    /// epochs; rotation in P4).
    pub fn data_key_for_epoch(&self, kek: &SecretKey, epoch: u32) -> Result<SecretKey> {
        let entry = self
            .deks
            .iter()
            .find(|d| d.epoch == epoch)
            .ok_or_else(|| anyhow!("no data key for epoch {epoch}"))?;
        unwrap_key(kek, &entry.wrapped, &dek_aad(epoch))
    }

    /// Rotate the passphrase: re-derive a fresh unseal key (new salt/params) and
    /// re-wrap the SAME master key under it. O(1) — records and indexes are
    /// untouched (they are keyed by the DEK, not the passphrase). Other wraps
    /// (e.g. recovery) are preserved.
    pub fn rotate_passphrase(&mut self, old: &str, new: &str) -> Result<()> {
        let kek = self.unseal_with_passphrase(old)?;
        let new_kdf = KdfParams::production();
        let unseal = new_kdf.derive(new)?;
        let new_wrap = wrap_key(&unseal, kek.as_slice(), KEK_AAD, "passphrase")?;
        self.kek_wraps.retain(|w| w.provider != "passphrase");
        self.kek_wraps.push(new_wrap);
        self.kdf = new_kdf;
        Ok(())
    }

    /// Add (or replace) a recovery wrap of the master key and return the one-time
    /// recovery code. Display it ONCE and never persist it — only the wrap is
    /// stored. Survives passphrase loss as long as the keystore file survives.
    pub fn add_recovery_code(&mut self, kek: &SecretKey) -> Result<String> {
        let code = generate_recovery_code();
        let wrap_key_bytes = recovery_key_from_code(&code);
        let wrap = wrap_key(&wrap_key_bytes, kek.as_slice(), KEK_AAD, "recovery")?;
        self.kek_wraps.retain(|w| w.provider != "recovery");
        self.kek_wraps.push(wrap);
        Ok(code)
    }

    /// Unseal the master key via a recovery code.
    pub fn unseal_with_recovery_code(&self, code: &str) -> Result<SecretKey> {
        let w = self
            .kek_wraps
            .iter()
            .find(|w| w.provider == "recovery")
            .ok_or_else(|| anyhow!("keystore has no recovery wrap"))?;
        let kek = unwrap_key(&recovery_key_from_code(code), w, KEK_AAD)?;
        if b64e(&fingerprint(kek.as_slice())) != self.kek_fingerprint {
            return Err(anyhow!("recovery code unsealed an unexpected key"));
        }
        Ok(kek)
    }

    /// Rotate the active data key: generate a new epoch's DEK, retire the previous
    /// active one (kept for reads of old records), and make the new one active.
    /// Returns the new epoch. Records re-encrypt lazily on next write or via an
    /// offline pass; old epochs stay readable via [`data_key_for_epoch`].
    pub fn rotate_dek(&mut self, kek: &SecretKey) -> Result<u32> {
        let new_epoch = self.deks.iter().map(|d| d.epoch).max().unwrap_or(0) + 1;
        let dek = random_key();
        let wrap = wrap_key(kek, dek.as_slice(), &dek_aad(new_epoch), "kek")?;
        for d in &mut self.deks {
            if d.state == "active" {
                d.state = "retired".to_string();
            }
        }
        self.deks.push(DekEntry {
            epoch: new_epoch,
            wrapped: wrap,
            state: "active".to_string(),
        });
        self.active_epoch = new_epoch;
        Ok(new_epoch)
    }

    /// Add (or replace) a KMS wrap of the master key, enabling unattended unseal
    /// without a passphrase. The KMS holds the wrapping key; only the wrapped KEK
    /// is stored here.
    pub fn add_kms_wrap(&mut self, kek: &SecretKey, kms: &dyn KmsClient) -> Result<()> {
        let blob = kms.kms_encrypt(kek.as_slice())?;
        let wrap = Wrapped {
            provider: "kms".to_string(),
            nonce: String::new(), // KMS ciphertext is self-framed
            ciphertext: b64e(&blob),
        };
        self.kek_wraps.retain(|w| w.provider != "kms");
        self.kek_wraps.push(wrap);
        Ok(())
    }

    /// Unseal the master key via the KMS provider.
    pub fn unseal_with_kms(&self, kms: &dyn KmsClient) -> Result<SecretKey> {
        let w = self
            .kek_wraps
            .iter()
            .find(|w| w.provider == "kms")
            .ok_or_else(|| anyhow!("keystore has no KMS wrap"))?;
        let blob = b64d(&w.ciphertext)?;
        let mut pt = kms.kms_decrypt(&blob)?;
        if pt.len() != KEY_LEN {
            return Err(anyhow!("KMS-unwrapped key has wrong length"));
        }
        let mut kek = Zeroizing::new([0u8; KEY_LEN]);
        kek.copy_from_slice(&pt);
        pt.zeroize(); // wipe the intermediate key-material buffer
        if b64e(&fingerprint(kek.as_slice())) != self.kek_fingerprint {
            return Err(anyhow!("KMS unsealed an unexpected key"));
        }
        Ok(kek)
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("failed to serialize keystore")
    }

    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("failed to parse keystore")
    }
}

type HmacSha256 = Hmac<Sha256>;

/// Label for deriving the (stable) index-blinding key from the master key.
const INDEX_KEY_LABEL: &[u8] = b"shodh:index-blind:v1";

/// Derives deterministic index-blinding tokens: `hex(HMAC-SHA256(index_key, term))`
/// (P3). Equal terms map to equal tokens (so exact-match index lookups still work)
/// while the plaintext term stays off disk. Leaks term equality/frequency only
/// (design §6 / §11 B-4). The key is derived from the KEK, so rotating the KEK
/// requires an index rebuild.
pub struct IndexBlinder {
    key: SecretKey,
}

impl IndexBlinder {
    pub fn new(key: SecretKey) -> Self {
        Self { key }
    }

    /// Derive the stable index-blinding key from the master key (KEK).
    pub fn derive_from_kek(kek: &SecretKey) -> Self {
        let mut mac = <HmacSha256 as Mac>::new_from_slice(kek.as_slice())
            .expect("HMAC accepts any key length");
        Mac::update(&mut mac, INDEX_KEY_LABEL);
        let tag = Mac::finalize(mac).into_bytes();
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        key.copy_from_slice(&tag[..KEY_LEN]);
        Self::new(key)
    }

    /// Blind a single index term to a hex token (safe for ':'-delimited index keys).
    pub fn blind(&self, term: &str) -> String {
        let mut mac = <HmacSha256 as Mac>::new_from_slice(self.key.as_slice())
            .expect("HMAC accepts any key length");
        Mac::update(&mut mac, term.as_bytes());
        hex::encode(Mac::finalize(mac).into_bytes())
    }
}

// ============================================================================
// Record envelope (P2) — encrypt/decrypt a serialized record blob under a DEK.
// Wire format: `ENC\0 | crypto_version(1) | epoch(4 LE) | nonce(12) | ct+tag`.
// AAD binds the epoch (prevents cross-epoch substitution). Record-id binding is
// a documented follow-up: it requires confirming the RocksDB key == memory id
// invariant at every write site first (design §5).
// ============================================================================

/// 4-byte marker distinguishing an encrypted record from a legacy plaintext
/// (bincode) record. bincode of a `Memory` never begins with `ENC\0`.
const RECORD_MARKER: &[u8; 4] = b"ENC\x00";
/// marker(4) + version(1) + epoch(4) + nonce(12)
const RECORD_HEADER_LEN: usize = 4 + 1 + 4 + 12;
/// GCM tag is 16 bytes; a valid ciphertext is at least the tag.
const RECORD_MIN_LEN: usize = RECORD_HEADER_LEN + 16;

fn record_aad(epoch: u32) -> Vec<u8> {
    format!("shodh:record:v1:epoch:{epoch}").into_bytes()
}

/// `true` if `data` carries the encrypted-record marker.
pub fn is_encrypted_record(data: &[u8]) -> bool {
    data.len() >= RECORD_MARKER.len() && &data[..4] == RECORD_MARKER
}

/// Read the epoch from an encrypted record header, so the reader can select the
/// right DEK before decrypting. Returns `None` if not an encrypted record.
pub fn record_epoch(data: &[u8]) -> Option<u32> {
    if !is_encrypted_record(data) || data.len() < RECORD_HEADER_LEN {
        return None;
    }
    Some(u32::from_le_bytes([data[5], data[6], data[7], data[8]]))
}

/// Decrypt an encrypted record blob with the DEK for its epoch. The caller is
/// responsible for selecting `dek` matching [`record_epoch`].
pub fn decrypt_record(dek: &SecretKey, data: &[u8]) -> Result<Vec<u8>> {
    if !is_encrypted_record(data) {
        return Err(anyhow!("not an encrypted record (missing marker)"));
    }
    if data.len() < RECORD_MIN_LEN {
        return Err(anyhow!(
            "encrypted record too short: {} bytes (min {RECORD_MIN_LEN})",
            data.len()
        ));
    }
    let version = data[4];
    if version != CRYPTO_VERSION {
        return Err(anyhow!("unsupported record crypto version {version}"));
    }
    let epoch = u32::from_le_bytes([data[5], data[6], data[7], data[8]]);
    let nonce = Nonce::from_slice(&data[9..RECORD_HEADER_LEN]);
    let ciphertext = &data[RECORD_HEADER_LEN..];
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(dek.as_slice()));
    cipher
        .decrypt(
            nonce,
            Payload {
                msg: ciphertext,
                aad: &record_aad(epoch),
            },
        )
        .map_err(|e| anyhow!("record decrypt failed (wrong key/epoch or tampered): {e}"))
}

/// Encrypts serialized record blobs under a fixed active DEK/epoch. Held by the
/// storage layer (P2 wiring). Reads of older epochs go through [`decrypt_record`]
/// with the DEK fetched from the keystore for that epoch.
pub struct RecordCryptor {
    epoch: u32,
    dek: SecretKey,
}

impl RecordCryptor {
    pub fn new(epoch: u32, dek: SecretKey) -> Self {
        Self { epoch, dek }
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Encrypt a serialized record blob under the active DEK.
    pub fn encrypt_record(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(self.dek.as_slice()));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: plaintext,
                    aad: &record_aad(self.epoch),
                },
            )
            .map_err(|e| anyhow!("record encrypt failed: {e}"))?;
        let mut out = Vec::with_capacity(RECORD_HEADER_LEN + ciphertext.len());
        out.extend_from_slice(RECORD_MARKER);
        out.push(CRYPTO_VERSION);
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(nonce.as_slice());
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt with the active DEK (convenience for same-epoch reads/tests).
    pub fn decrypt_record(&self, data: &[u8]) -> Result<Vec<u8>> {
        decrypt_record(&self.dek, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_unseal_roundtrip() {
        let ks =
            Keystore::create("correct horse battery staple", KdfParams::fast_for_tests()).unwrap();
        let kek = ks
            .unseal_with_passphrase("correct horse battery staple")
            .unwrap();
        let (epoch, dek) = ks.active_data_key(&kek).unwrap();
        assert_eq!(epoch, 0);
        assert_eq!(dek.len(), KEY_LEN);
    }

    #[test]
    fn wrong_passphrase_fails() {
        let ks = Keystore::create("right-passphrase", KdfParams::fast_for_tests()).unwrap();
        assert!(ks.unseal_with_passphrase("wrong-passphrase").is_err());
    }

    #[test]
    fn json_roundtrip_preserves_unseal() {
        let ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let json = ks.to_json().unwrap();
        let ks2 = Keystore::from_json(&json).unwrap();
        let kek = ks2.unseal_with_passphrase("pw").unwrap();
        ks2.active_data_key(&kek).unwrap();
        // The raw KEK must never appear in the serialised keystore (only wrapped).
        assert!(
            !json.contains(&b64e(kek.as_slice())),
            "raw KEK must not appear in serialized keystore"
        );
        assert_eq!(ks.kek_fingerprint, ks2.kek_fingerprint);
        assert_eq!(ks2.crypto_version, CRYPTO_VERSION);
        assert_eq!(ks2.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn argon2_is_deterministic_for_same_params() {
        let kdf = KdfParams::fast_for_tests();
        let a = kdf.derive("pw").unwrap();
        let b = kdf.derive("pw").unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn wrap_unwrap_roundtrip_and_aad_binding() {
        let wk = [7u8; KEY_LEN];
        let secret = [9u8; KEY_LEN];
        let w = wrap_key(&wk, &secret, b"aad-a", "x").unwrap();

        // correct key + AAD recovers the secret
        let got = unwrap_key(&wk, &w, b"aad-a").unwrap();
        assert_eq!(got.as_slice(), &secret);

        // wrong AAD fails (binding holds)
        assert!(unwrap_key(&wk, &w, b"aad-b").is_err());
        // wrong wrapping key fails
        assert!(unwrap_key(&[8u8; KEY_LEN], &w, b"aad-a").is_err());
    }

    #[test]
    fn epoch_aad_prevents_cross_epoch_unwrap() {
        // A DEK wrapped for epoch 0 must not unwrap under epoch 1's AAD.
        let kek = [3u8; KEY_LEN];
        let dek = [4u8; KEY_LEN];
        let w = wrap_key(&kek, &dek, &dek_aad(0), "kek").unwrap();
        assert!(unwrap_key(&kek, &w, &dek_aad(1)).is_err());
        assert!(unwrap_key(&kek, &w, &dek_aad(0)).is_ok());
    }

    #[test]
    fn fingerprint_is_stable_and_distinguishing() {
        assert_eq!(fingerprint(&[1u8; KEY_LEN]), fingerprint(&[1u8; KEY_LEN]));
        assert_ne!(fingerprint(&[1u8; KEY_LEN]), fingerprint(&[2u8; KEY_LEN]));
    }

    #[test]
    fn record_encrypt_decrypt_roundtrip() {
        let cryptor = RecordCryptor::new(0, Zeroizing::new([5u8; KEY_LEN]));
        let plaintext = b"bincode-serialized-memory-record-bytes".to_vec();
        let enc = cryptor.encrypt_record(&plaintext).unwrap();
        assert!(is_encrypted_record(&enc));
        assert_eq!(record_epoch(&enc), Some(0));
        assert_ne!(enc, plaintext);
        assert_eq!(cryptor.decrypt_record(&enc).unwrap(), plaintext);
    }

    #[test]
    fn legacy_plaintext_is_not_marked_encrypted() {
        // A typical bincode blob won't start with ENC\0.
        let plain = vec![0x10, 0x00, 0x00, 0x00, 0x42];
        assert!(!is_encrypted_record(&plain));
        assert_eq!(record_epoch(&plain), None);
    }

    #[test]
    fn record_wrong_dek_fails() {
        let c1 = RecordCryptor::new(0, Zeroizing::new([1u8; KEY_LEN]));
        let enc = c1.encrypt_record(b"secret").unwrap();
        assert!(decrypt_record(&Zeroizing::new([2u8; KEY_LEN]), &enc).is_err());
    }

    #[test]
    fn record_tamper_is_detected() {
        let c = RecordCryptor::new(3, Zeroizing::new([7u8; KEY_LEN]));
        let mut enc = c.encrypt_record(b"payload").unwrap();
        let last = enc.len() - 1;
        enc[last] ^= 0xff; // flip a ciphertext byte
        assert!(c.decrypt_record(&enc).is_err());
    }

    #[test]
    fn record_epoch_in_header_matches() {
        let c = RecordCryptor::new(42, Zeroizing::new([9u8; KEY_LEN]));
        let enc = c.encrypt_record(b"x").unwrap();
        assert_eq!(record_epoch(&enc), Some(42));
        assert_eq!(c.epoch(), 42);
    }

    #[test]
    fn index_blinder_is_deterministic_keyed_and_hides_term() {
        let kek = Zeroizing::new([4u8; KEY_LEN]);
        let b1 = IndexBlinder::derive_from_kek(&kek);
        let b2 = IndexBlinder::derive_from_kek(&kek);
        assert_eq!(b1.blind("alice"), b2.blind("alice")); // deterministic & stable
        assert_ne!(b1.blind("alice"), b1.blind("bob")); // distinct terms differ
        assert!(!b1.blind("alice").contains("alice")); // plaintext term not present
        assert_eq!(b1.blind("alice").len(), 64); // 32-byte HMAC as hex

        let other = IndexBlinder::derive_from_kek(&Zeroizing::new([5u8; KEY_LEN]));
        assert_ne!(b1.blind("alice"), other.blind("alice")); // key-dependent
    }

    #[test]
    fn record_from_keystore_dek_end_to_end() {
        let ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let (epoch, dek) = ks.active_data_key(&kek).unwrap();
        let cryptor = RecordCryptor::new(epoch, dek);
        let enc = cryptor.encrypt_record(b"end-to-end").unwrap();
        assert_eq!(cryptor.decrypt_record(&enc).unwrap(), b"end-to-end");
    }

    #[test]
    fn rotate_passphrase_preserves_master_key() {
        let mut ks = Keystore::create("old-pass", KdfParams::fast_for_tests()).unwrap();
        let kek_before = ks.unseal_with_passphrase("old-pass").unwrap();
        ks.rotate_passphrase("old-pass", "new-pass").unwrap();
        assert!(ks.unseal_with_passphrase("old-pass").is_err());
        let kek_after = ks.unseal_with_passphrase("new-pass").unwrap();
        // Same master key -> records/index untouched by a passphrase change.
        assert_eq!(kek_before.as_slice(), kek_after.as_slice());
    }

    #[test]
    fn rotate_passphrase_wrong_old_is_rejected() {
        let mut ks = Keystore::create("old", KdfParams::fast_for_tests()).unwrap();
        assert!(ks.rotate_passphrase("not-old", "new").is_err());
        assert!(ks.unseal_with_passphrase("old").is_ok()); // unchanged
    }

    #[test]
    fn recovery_code_unseals_same_key() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let code = ks.add_recovery_code(&kek).unwrap();
        assert_eq!(code.len(), 48);
        assert_eq!(
            ks.unseal_with_recovery_code(&code).unwrap().as_slice(),
            kek.as_slice()
        );
        assert!(ks.unseal_with_recovery_code("deadbeef").is_err());
    }

    #[test]
    fn recovery_survives_passphrase_rotation() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let code = ks.add_recovery_code(&kek).unwrap();
        ks.rotate_passphrase("pw", "pw2").unwrap();
        assert_eq!(
            ks.unseal_with_recovery_code(&code).unwrap().as_slice(),
            kek.as_slice()
        );
    }

    #[test]
    fn rotate_dek_adds_epoch_and_keeps_old_readable() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let (e0, dek0) = ks.active_data_key(&kek).unwrap();
        assert_eq!(e0, 0);

        let e1 = ks.rotate_dek(&kek).unwrap();
        assert_eq!(e1, 1);
        let (active_epoch, dek_active) = ks.active_data_key(&kek).unwrap();
        assert_eq!(active_epoch, 1);

        // Old epoch's DEK is still recoverable (for reading old records); new differs.
        assert_eq!(
            ks.data_key_for_epoch(&kek, 0).unwrap().as_slice(),
            dek0.as_slice()
        );
        assert_ne!(dek_active.as_slice(), dek0.as_slice());
    }

    #[test]
    fn local_kms_encrypt_decrypt_roundtrip() {
        let kms = LocalAeadKms::new([3u8; KEY_LEN]);
        let blob = kms.kms_encrypt(b"master-key-bytes").unwrap();
        assert_ne!(blob, b"master-key-bytes");
        assert_eq!(kms.kms_decrypt(&blob).unwrap(), b"master-key-bytes");
        assert!(LocalAeadKms::new([4u8; KEY_LEN])
            .kms_decrypt(&blob)
            .is_err());
    }

    #[test]
    fn kms_wrap_unseals_same_key_and_coexists_with_passphrase() {
        let mut ks = Keystore::create("pw", KdfParams::fast_for_tests()).unwrap();
        let kek = ks.unseal_with_passphrase("pw").unwrap();
        let kms = LocalAeadKms::new([9u8; KEY_LEN]);
        ks.add_kms_wrap(&kek, &kms).unwrap();
        assert_eq!(ks.unseal_with_kms(&kms).unwrap().as_slice(), kek.as_slice());
        assert!(ks.unseal_with_passphrase("pw").is_ok()); // multi-wrap: passphrase still works
        assert!(ks
            .unseal_with_kms(&LocalAeadKms::new([1u8; KEY_LEN]))
            .is_err()); // wrong KMS key
    }

    #[test]
    fn decode_32_accepts_hex_and_base64() {
        assert!(decode_32(&"00".repeat(32)).is_ok());
        assert!(decode_32(&base64::engine::general_purpose::STANDARD.encode([0u8; 32])).is_ok());
        assert!(decode_32("tooshort").is_err());
    }
}
