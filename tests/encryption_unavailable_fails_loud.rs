//! Contract: encryption requested but unavailable fails loud. A store that has
//! a keystore (encryption was requested when it was created) opened without
//! the passphrase must refuse to open — not open in plaintext mode, serve its
//! ciphertext as corruption, and write new plaintext records beside it.
//!
//! The second half pins the other side of the same contract: even a failed
//! open leaves the bytes on disk exactly as they were. Own test binary.

use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-unavailable-correct-horse-Z9";
const PLAINTEXT: &str = "rT4-fails-loud-distinctive-plaintext-do-not-leak-Z9";

fn raw_record(db_path: &std::path::Path, id: &MemoryId) -> Vec<u8> {
    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, db_path, &cfs, false).expect("reopen");
    db.get(id.0.as_bytes())
        .expect("rocksdb get")
        .expect("record on disk")
}

#[test]
fn keystore_present_without_passphrase_refuses_to_open() {
    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let id = MemoryId(Uuid::new_v4());

    // Encryption requested: create the keystore and write one encrypted record.
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("create keystore");
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: PLAINTEXT.to_string(),
            ..Default::default()
        };
        storage
            .store(&Memory::new(
                id.clone(),
                experience,
                0.5,
                None,
                None,
                None,
                None,
            ))
            .expect("store");
    }
    let before = raw_record(&db_path, &id);
    assert!(shodh_memory::keystore::is_encrypted_record(&before));

    // Encryption unavailable: the passphrase is gone. The open must fail, and
    // fail for the stated reason.
    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
    let err = match MemoryStorage::new(temp.path(), None) {
        Ok(_) => panic!("a store with a keystore must not open without its passphrase"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("keystore.json") && err.contains("SHODH_MASTER_PASSPHRASE"),
        "error must name the keystore and the missing secret, got: {err}"
    );

    // Nothing was written in plaintext, and the ciphertext is byte-identical.
    let after = raw_record(&db_path, &id);
    assert_eq!(after, before, "a refused open must not touch the record");
    assert!(
        !after
            .windows(PLAINTEXT.len())
            .any(|w| w == PLAINTEXT.as_bytes()),
        "plaintext must not be on disk"
    );

    // An empty passphrase is "unset", not a passphrase.
    std::env::set_var("SHODH_MASTER_PASSPHRASE", "");
    assert!(
        MemoryStorage::new(temp.path(), None).is_err(),
        "an empty SHODH_MASTER_PASSPHRASE must not open an encrypted store"
    );
    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
