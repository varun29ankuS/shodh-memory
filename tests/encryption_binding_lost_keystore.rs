//! Contract: an encrypted database remembers its keystore. With keystore.json
//! gone it refuses to open, with the passphrase or without it, rather than
//! creating a fresh keystore over its records or opening in plaintext; restoring
//! the keystore opens it again with every record intact. Own test binary.

use shodh_memory::keystore::is_encrypted_record;
use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-binding-lost-correct-horse-Z9";

fn sample(content: &str) -> (MemoryId, Memory) {
    let id = MemoryId(Uuid::new_v4());
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: content.to_string(),
        ..Default::default()
    };
    let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
    (id, memory)
}

fn raw_record(db_path: &std::path::Path, id: &MemoryId) -> Vec<u8> {
    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, db_path, &cfs, false).expect("reopen");
    db.get(id.0.as_bytes())
        .expect("rocksdb get")
        .expect("record on disk")
}

fn open_err(path: &std::path::Path) -> String {
    match MemoryStorage::new(path, None) {
        Ok(_) => panic!("a store bound to a missing keystore must not open"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn lost_keystore_is_refused_never_recreated_and_records_survive() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let keystore_path = db_path.join("keystore.json");
    let (id, memory) = sample("bound-record");

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open");
        storage.store(&memory).expect("store");
    }
    assert!(is_encrypted_record(&raw_record(&db_path, &id)));

    let saved = std::fs::read(&keystore_path).expect("keystore on disk");
    std::fs::remove_file(&keystore_path).expect("lose the keystore");

    let err = open_err(temp.path());
    assert!(err.contains("bound to keystore"), "passphrase set: {err}");
    assert!(!keystore_path.exists(), "no fresh keystore was created");

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
    let err = open_err(temp.path());
    assert!(err.contains("bound to keystore"), "passphrase unset: {err}");
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);

    assert!(
        is_encrypted_record(&raw_record(&db_path, &id)),
        "the refused opens left the record as it was"
    );

    std::fs::write(&keystore_path, &saved).expect("restore the keystore");
    let storage = MemoryStorage::new(temp.path(), None).expect("opens with its own keystore");
    assert_eq!(
        storage.get(&id).expect("get").experience.content,
        "bound-record"
    );

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
