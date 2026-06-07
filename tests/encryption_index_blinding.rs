//! e2e: tag/entity index terms are blinded on disk (encryption-v2). Own test
//! binary so the process-global keystore crypto is isolated (shodh supports a
//! single keystore per process).

use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

#[test]
fn index_terms_blinded_on_disk() {
    std::env::set_var(
        "SHODH_MASTER_PASSPHRASE",
        "rT4-shodh-blind-correct-horse-Z9",
    );
    const SECRET_TAG: &str = "secret-tag-do-not-leak-abc123";
    const SECRET_ENTITY: &str = "secret-entity-do-not-leak-def456";

    let temp = TempDir::new().expect("temp dir");
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "indexed content".to_string(),
            tags: vec![SECRET_TAG.to_string()],
            entities: vec![SECRET_ENTITY.to_string()],
            ..Default::default()
        };
        storage
            .store(&Memory::new(
                MemoryId(Uuid::new_v4()),
                experience,
                0.5,
                None,
                None,
                None,
                None,
            ))
            .expect("store");
    }

    let db_path = temp.path().join("storage");
    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf_for_read_only(&opts, &db_path, &cfs, false).expect("reopen");
    let cf = db.cf_handle("memory_index").expect("memory_index cf");
    for item in db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
        let (key, _) = item.expect("iter");
        let ks = String::from_utf8_lossy(&key);
        assert!(!ks.contains(SECRET_TAG), "plaintext tag leaked: {ks}");
        assert!(!ks.contains(SECRET_ENTITY), "plaintext entity leaked: {ks}");
    }

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
