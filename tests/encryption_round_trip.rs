//! e2e: record round-trip + on-disk opacity (encryption-v2) against a real
//! RocksDB. Own test binary so the process-global keystore crypto is isolated
//! (shodh supports a single keystore per process; index blinding, wrong-
//! passphrase, and the cross-keystore guard live in sibling test files).

use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-shodh-v2-correct-horse-battery-staple-Z9";
const PLAINTEXT: &str = "rT4-encryption-round-trip-distinctive-plaintext-do-not-leak-Z9X-2026";

#[test]
fn record_round_trip_and_on_disk_opacity() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    let id = MemoryId(Uuid::new_v4());
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
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
    }

    let db_path = temp.path().join("storage");
    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cfs, false).expect("reopen");
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

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
