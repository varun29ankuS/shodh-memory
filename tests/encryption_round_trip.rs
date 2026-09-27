//! Contract: with a keystore active, a memory that is stored, recalled (which
//! rewrites the record to bump its access metadata), modified, and read back
//! after the DB is reopened is still an encrypted record on disk, and the raw
//! stored bytes never contain the plaintext.
//!
//! The recall step is the one that matters. `persist_access_updates` runs on
//! the recall hot path and rewrites every recalled record; the branch this was
//! ported from serialized it with `encode_sho` directly, so every memory ever
//! recalled had its ciphertext overwritten with plaintext — and because the
//! read path accepts plaintext, nothing noticed. This test reads the bytes.
//!
//! Own test binary: the keystore crypto is process-global (one keystore per
//! process), so every encryption test file is its own process.

use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-shodh-v2-correct-horse-battery-staple-Z9";
const PLAINTEXT: &str = "rT4-encryption-round-trip-distinctive-plaintext-do-not-leak-Z9X-2026";
const MODIFIED: &str = "rT4-encryption-round-trip-modified-plaintext-do-not-leak-Z9X-2026";

fn raw_record(db_path: &std::path::Path, id: &MemoryId) -> Vec<u8> {
    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, db_path, &cfs, false).expect("reopen");
    db.get(id.0.as_bytes())
        .expect("rocksdb get")
        .expect("record on disk")
}

fn assert_opaque(raw: &[u8], stage: &str) {
    assert!(
        shodh_memory::keystore::is_encrypted_record(raw),
        "{stage}: primary record must be an ENC\\0 envelope, got prefix {:?}",
        &raw[..raw.len().min(8)]
    );
    for needle in [PLAINTEXT, MODIFIED] {
        assert!(
            !raw.windows(needle.len()).any(|w| w == needle.as_bytes()),
            "{stage}: plaintext must NOT appear in the stored record bytes"
        );
    }
}

#[test]
fn record_round_trip_and_on_disk_opacity_survives_recall_and_modify() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let id = MemoryId(Uuid::new_v4());

    // 1. Store, read back, and recall.
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        assert!(shodh_memory::memory::storage::encryption_active());
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: PLAINTEXT.to_string(),
            ..Default::default()
        };
        let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
        storage.store(&memory).expect("store");
        assert_eq!(storage.get(&id).expect("get").experience.content, PLAINTEXT);

        // What recall does: bump access metadata on the in-memory record and
        // persist the batch. This is the write that used to leak plaintext.
        let before = memory.importance();
        memory.update_access();
        storage
            .persist_access_updates(&[(&memory, before)])
            .expect("persist_access_updates");
        let recalled = storage.get(&id).expect("get after access update");
        assert_eq!(recalled.experience.content, PLAINTEXT);
        assert_eq!(recalled.access_count(), 1, "access bump was persisted");
    }
    assert_opaque(&raw_record(&db_path, &id), "after store + recall");

    // 2. Reopen, modify through the read-modify-write path, close.
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen storage");
        let updated = storage
            .modify(&id, |m| m.experience.content = MODIFIED.to_string())
            .expect("modify")
            .expect("record present");
        assert_eq!(updated.experience.content, MODIFIED);
        assert_eq!(storage.get(&id).expect("get").experience.content, MODIFIED);
    }
    let raw = raw_record(&db_path, &id);
    assert_opaque(&raw, "after reopen + modify");
    assert_eq!(
        shodh_memory::keystore::record_epoch(&raw),
        Some(0),
        "a fresh keystore writes under epoch 0"
    );

    // 3. Reopen once more and read the modified content back: the ciphertext
    // on disk is the record, not a stale copy beside a plaintext one.
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen storage");
        assert_eq!(storage.get(&id).expect("get").experience.content, MODIFIED);
    }

    assert_eq!(
        shodh_memory::memory::storage::plaintext_reads_under_keystore(),
        0,
        "no plaintext record was ever read under the keystore"
    );
    assert!(
        temp.path().join("storage").join("keystore.json").exists(),
        "keystore.json was created beside the DB"
    );

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
