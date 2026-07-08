//! e2e: a primary record relocated to a different id must fail to decrypt
//! (encryption-v2 identity binding). Own test binary so the process-global
//! keystore crypto is isolated.

use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-shodh-v2-relocation-correct-horse-battery-Z9";

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

#[test]
fn relocated_record_is_rejected_by_identity_binding() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    let (id_a, mem_a) = sample("record-A-distinctive-plaintext-Z9");
    let (id_b, mem_b) = sample("record-B-distinctive-plaintext-Z9");

    // Store both; each round-trips under its own id.
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open storage");
        storage.store(&mem_a).expect("store A");
        storage.store(&mem_b).expect("store B");
        assert_eq!(
            storage.get(&id_a).unwrap().experience.content,
            "record-A-distinctive-plaintext-Z9"
        );
        assert_eq!(
            storage.get(&id_b).unwrap().experience.content,
            "record-B-distinctive-plaintext-Z9"
        );
    } // drop -> release the RocksDB lock

    // Relocate: copy B's ciphertext on top of A's key (an at-rest swap, or a
    // restore mixup). The bytes are a valid ENC\0 envelope, so structural checks
    // pass — only the AAD identity binding (AAD == the record's id) catches it.
    let db_path = temp.path().join("storage");
    {
        let opts = rocksdb::Options::default();
        let cfs = rocksdb::DB::list_cf(&opts, &db_path).expect("list cfs");
        let db = rocksdb::DB::open_cf(&opts, &db_path, &cfs).expect("reopen rw");
        let b_bytes = db
            .get(id_b.0.as_bytes())
            .expect("rocksdb get B")
            .expect("B present");
        assert!(
            b_bytes.starts_with(b"ENC\0"),
            "B must be an ENC\\0 envelope"
        );
        db.put(id_a.0.as_bytes(), &b_bytes)
            .expect("relocate B onto A's key");
    } // drop -> release the lock before reopening through storage

    // Reading A must now be a hard error: B's ciphertext is bound to B's id, so
    // decrypting it under A's id fails. It must NEVER silently serve B's content
    // (or any plaintext) at A's id.
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen");
        let got = storage.get(&id_a);
        assert!(
            got.is_err(),
            "a record relocated to a different id must fail to decrypt, got {:?}",
            got.map(|m| m.experience.content)
        );
    }

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
