//! Contract: opt-in, default off. With no keystore and no passphrase the store
//! behaves exactly as before this feature existed: the bytes on disk are the
//! plain SHO envelope `encode_sho` produces, no keystore file is created, no
//! sentinel is written to the index CF, and nothing counts or warns.
//!
//! Own test binary, and the ONLY encryption test that never sets the
//! passphrase: the process-global crypto, once installed, cannot be uninstalled.

use shodh_memory::memory::storage::{
    encryption_active, plaintext_reads_under_keystore, MemoryStorage,
};
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

#[test]
fn no_keystore_means_plain_sho_bytes_and_no_side_files() {
    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
    std::env::remove_var("SHODH_REQUIRE_ENCRYPTED_READS");

    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let id = MemoryId(Uuid::new_v4());
    let experience = Experience {
        experience_type: ExperienceType::Observation,
        content: "default-off-plain-record".to_string(),
        ..Default::default()
    };
    let memory = Memory::new(id.clone(), experience, 0.5, None, None, None, None);
    let expected = shodh_memory::serialization::encode_sho(&memory).expect("encode_sho");

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        assert!(!encryption_active(), "no keystore, no passphrase: off");
        storage.store(&memory).expect("store");
        assert_eq!(
            storage.get(&id).expect("get").experience.content,
            "default-off-plain-record"
        );
        // The recall-path write is plain too.
        let before = memory.importance();
        memory.update_access();
        storage
            .persist_access_updates(&[(&memory, before)])
            .expect("persist_access_updates");
    }

    assert!(
        !db_path.join("keystore.json").exists(),
        "no keystore is created unless a passphrase is given"
    );

    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cfs, false).expect("reopen");
    let raw = db
        .get(id.0.as_bytes())
        .expect("rocksdb get")
        .expect("record on disk");
    assert!(raw.starts_with(b"SHO"), "plain SHO envelope");
    assert!(!shodh_memory::keystore::is_encrypted_record(&raw));
    // Byte-identical to encode_sho of the record as last written (the access
    // bump changed access_count/last_accessed, so re-encode that state).
    let expected_after_access =
        shodh_memory::serialization::encode_sho(&memory).expect("encode_sho");
    assert_ne!(
        expected, expected_after_access,
        "the access bump changed the record"
    );
    assert_eq!(
        raw, expected_after_access,
        "on-disk bytes are exactly encode_sho(memory): no envelope, no marker"
    );

    let idx = db.cf_handle("memory_index").expect("index cf");
    assert!(
        db.get_cf(idx, b"meta:keystore_generation")
            .expect("get sentinel")
            .is_none(),
        "the rollback sentinel is only written while a keystore is active"
    );

    assert_eq!(plaintext_reads_under_keystore(), 0);
}
