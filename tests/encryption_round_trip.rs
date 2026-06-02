//! End-to-end encryption round-trip against a real RocksDB store.
//!
//! Sets `SHODH_ENCRYPTION_KEY`, stores a memory through `MemoryStorage`,
//! reads it back, and verifies the plaintext is recovered intact. Then drops
//! the storage and re-opens the underlying RocksDB read-only to assert the
//! raw bytes on disk do NOT contain the plaintext — proof that encryption
//! was actually applied between `store` and `get` rather than the wiring
//! silently no-op'ing.
//!
//! Lives in `tests/` (its own integration-test binary / process) so the
//! storage-side `OnceLock` encryptor initialises from this test's env and
//! is not polluted by other test binaries.

use std::env;
use std::sync::Mutex;

use tempfile::TempDir;
use uuid::Uuid;

use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};

/// 32-byte hex key (the same deterministic key used by the encryption unit tests).
const KEY_HEX: &str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";

/// Distinctive plaintext that is extremely unlikely to appear by chance in
/// any envelope/format/version bytes — keeps the "not present in raw" assertion
/// sharp even if the encoding layer adds incidental ASCII.
const PLAINTEXT: &str = "rT4-encryption-round-trip-distinctive-plaintext-do-not-leak-Z9X-2026";

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn encryption_round_trip_against_real_rocksdb() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("SHODH_ENCRYPTION_KEY", KEY_HEX);

    let temp_dir = TempDir::new().expect("temp dir");
    let storage_path = temp_dir.path().to_path_buf();
    let memory_id = MemoryId(Uuid::new_v4());

    // ── Phase 1: store + get through MemoryStorage; plaintext must recover ──
    {
        let storage = MemoryStorage::new(&storage_path, None).expect("open storage");

        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: PLAINTEXT.to_string(),
            ..Default::default()
        };
        let memory = Memory::new(memory_id.clone(), experience, 0.5, None, None, None, None);

        storage.store(&memory).expect("store");

        let recovered = storage.get(&memory_id).expect("get");
        assert_eq!(
            recovered.experience.content, PLAINTEXT,
            "decrypted content must equal the original plaintext",
        );
    } // storage dropped — releases the RocksDB lock so the raw read below can open it

    // ── Phase 2: raw RocksDB read; plaintext must NOT appear on disk ──
    // MemoryStorage::new joins "storage" to the supplied path internally,
    // so the actual RocksDB lives at storage_path/storage/.
    let db_path = storage_path.join("storage");
    let opts = rocksdb::Options::default();
    let cf_names = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cf_names, false)
        .expect("reopen DB read-only");

    let raw = db
        .get(memory_id.0.as_bytes())
        .expect("rocksdb get")
        .expect("memory key must exist on disk");

    assert!(
        !raw.windows(PLAINTEXT.len())
            .any(|w| w == PLAINTEXT.as_bytes()),
        "plaintext content must NOT appear in raw stored bytes when encryption is enabled \
         (raw len {}, plaintext len {})",
        raw.len(),
        PLAINTEXT.len(),
    );
    assert!(
        raw.starts_with(b"ENC\0"),
        "encrypted memory should be an opaque record-level ciphertext"
    );
    assert!(
        !raw.starts_with(b"SHO"),
        "the SHO serialization envelope should be inside the ciphertext"
    );

    env::remove_var("SHODH_ENCRYPTION_KEY");
}
