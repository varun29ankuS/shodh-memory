//! Contract: with `SHODH_REQUIRE_ENCRYPTED_READS=1`, reading a plaintext
//! record while a keystore is active is an ERROR, not a warning — the record
//! is not served, not rewritten, and the tripwire metric still counts it. Own
//! test binary (the env var and the process-global crypto are both global).

use shodh_memory::memory::storage::{plaintext_reads_under_keystore, MemoryStorage};
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-tripwire-strict-correct-horse-Z9";
const PLAINTEXT: &str = "rT4-tripwire-strict-distinctive-plaintext-Z9";

fn raw_record(db_path: &std::path::Path, id: &MemoryId) -> Vec<u8> {
    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, db_path, &cfs, false).expect("reopen");
    db.get(id.0.as_bytes())
        .expect("rocksdb get")
        .expect("record on disk")
}

#[test]
fn plaintext_read_under_keystore_is_an_error_when_required() {
    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
    std::env::set_var("SHODH_REQUIRE_ENCRYPTED_READS", "1");

    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let id = MemoryId(Uuid::new_v4());

    // Plaintext store (no keystore yet). The strict flag is inert without a
    // keystore: there is nothing to require.
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open plaintext store");
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
        assert_eq!(storage.get(&id).expect("get").experience.content, PLAINTEXT);
        assert_eq!(plaintext_reads_under_keystore(), 0);
    }
    let before = raw_record(&db_path, &id);

    // Keystore on, strict on: the plaintext record is refused.
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open with keystore");
        let err = match storage.get(&id) {
            Ok(m) => panic!(
                "strict mode must refuse a plaintext record, served {:?}",
                m.experience.content
            ),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            err.contains("SHODH_REQUIRE_ENCRYPTED_READS"),
            "error must name the flag, got: {err}"
        );
        assert_eq!(plaintext_reads_under_keystore(), 1, "still counted");
        // `get_opt` is the same path: an error, not `Ok(None)` — and a
        // refused read is still a counted read.
        assert!(storage.get_opt(&id).is_err());
        assert_eq!(plaintext_reads_under_keystore(), 2, "counted again");
    }

    // Refused means untouched: no lazy rewrite ran, the bytes are as they were.
    assert_eq!(raw_record(&db_path, &id), before);

    // Relaxing the flag at runtime (it is read per call) lets the record
    // through and re-encrypts it; strict mode afterwards is satisfied.
    std::env::remove_var("SHODH_REQUIRE_ENCRYPTED_READS");
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen");
        assert_eq!(storage.get(&id).expect("get").experience.content, PLAINTEXT);
        assert_eq!(
            plaintext_reads_under_keystore(),
            3,
            "the relaxed read counts"
        );
    }
    assert!(shodh_memory::keystore::is_encrypted_record(&raw_record(
        &db_path, &id
    )));
    std::env::set_var("SHODH_REQUIRE_ENCRYPTED_READS", "true");
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen strict");
        assert_eq!(storage.get(&id).expect("get").experience.content, PLAINTEXT);
        assert_eq!(
            plaintext_reads_under_keystore(),
            3,
            "encrypted read does not trip"
        );
    }

    std::env::remove_var("SHODH_REQUIRE_ENCRYPTED_READS");
    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
