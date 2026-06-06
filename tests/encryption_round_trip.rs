//! End-to-end encryption-v2 round-trip against a real RocksDB store.
//!
//! Uses the keystore (SHODH_MASTER_PASSPHRASE) — the v2 scheme — not the old
//! single SHODH_ENCRYPTION_KEY. Verifies: (1) a stored memory round-trips through
//! store()/get(); (2) the on-disk record is an opaque ENC\0 envelope with the
//! plaintext absent; (3) tag/entity index terms are blinded on disk; (4) a wrong
//! passphrase is a hard error.
//!
//! Its own integration-test binary so SHODH_MASTER_PASSPHRASE and the process-
//! global storage crypto initialise from this test's env, isolated from other
//! binaries. ENV_LOCK serialises the env-touching tests within this binary.

use std::env;
use std::sync::Mutex;

use tempfile::TempDir;
use uuid::Uuid;

use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};

static ENV_LOCK: Mutex<()> = Mutex::new(());

const PASSPHRASE: &str = "rT4-shodh-v2-e2e-correct-horse-battery-staple-Z9";
const PLAINTEXT: &str = "rT4-encryption-round-trip-distinctive-plaintext-do-not-leak-Z9X-2026";

fn sample(id: MemoryId, content: &str) -> Memory {
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: content.to_string(),
        ..Default::default()
    };
    Memory::new(id, experience, 0.5, None, None, None, None)
}

#[test]
fn record_round_trip_and_on_disk_opacity() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    let id = MemoryId(Uuid::new_v4());

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        storage
            .store(&sample(id.clone(), PLAINTEXT))
            .expect("store");
        let got = storage.get(&id).expect("get");
        assert_eq!(got.experience.content, PLAINTEXT);
    }

    // Raw RocksDB read (MemoryStorage joins "storage" to the path).
    let db_path = temp.path().join("storage");
    let opts = rocksdb::Options::default();
    let cf_names = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cf_names, false)
        .expect("reopen read-only");
    let raw = db
        .get(id.0.as_bytes())
        .expect("rocksdb get")
        .expect("record on disk");

    assert!(
        raw.starts_with(b"ENC\0"),
        "primary record must be opaque record-level ciphertext"
    );
    assert!(
        !raw.windows(PLAINTEXT.len())
            .any(|w| w == PLAINTEXT.as_bytes()),
        "plaintext must NOT appear in the stored record bytes"
    );

    env::remove_var("SHODH_MASTER_PASSPHRASE");
}

#[test]
fn index_terms_blinded_on_disk() {
    let _guard = ENV_LOCK.lock().unwrap();
    env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    const SECRET_TAG: &str = "secret-tag-do-not-leak-abc123";
    const SECRET_ENTITY: &str = "secret-entity-do-not-leak-def456";

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "indexed content".to_string(),
            tags: vec![SECRET_TAG.to_string()],
            entities: vec![SECRET_ENTITY.to_string()],
            ..Default::default()
        };
        let memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            experience,
            0.5,
            None,
            None,
            None,
            None,
        );
        storage.store(&memory).expect("store");
    }

    let db_path = temp.path().join("storage");
    let opts = rocksdb::Options::default();
    let cf_names = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cf_names, false)
        .expect("reopen read-only");
    let cf = db.cf_handle("memory_index").expect("memory_index cf");

    for item in db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
        let (key, _) = item.expect("iter");
        let ks = String::from_utf8_lossy(&key);
        assert!(!ks.contains(SECRET_TAG), "plaintext tag leaked: {ks}");
        assert!(!ks.contains(SECRET_ENTITY), "plaintext entity leaked: {ks}");
    }

    env::remove_var("SHODH_MASTER_PASSPHRASE");
}

#[test]
fn wrong_passphrase_is_hard_error() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp = TempDir::new().expect("temp dir");

    env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("create keystore");
        storage
            .store(&sample(MemoryId(Uuid::new_v4()), "x"))
            .expect("store");
    }

    env::set_var("SHODH_MASTER_PASSPHRASE", "the-wrong-passphrase");
    assert!(
        MemoryStorage::new(temp.path(), None).is_err(),
        "opening with the wrong passphrase must be a hard error"
    );

    env::remove_var("SHODH_MASTER_PASSPHRASE");
}
