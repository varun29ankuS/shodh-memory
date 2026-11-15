//! Contract: under a keystore root (the server's data directory) every tenant's
//! store binds to one keystore, so one process serves them all. A per-store
//! keystore that encrypted records moves to the root; one that encrypted nothing
//! is set aside. Own test binary.

use shodh_memory::keystore::{is_encrypted_record, KdfParams, Keystore};
use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-root-tenants-correct-horse-Z9";

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

#[test]
fn tenants_share_the_root_keystore_and_per_store_keystores_migrate() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);
    std::env::remove_var("SHODH_KEYSTORE_DIR");

    let base = TempDir::new().expect("data root");
    let u1 = base.path().join("u1");
    let u2 = base.path().join("u2");
    let (id1, m1) = sample("tenant-one");
    let (id2, m2) = sample("tenant-two");

    // Tenant one was encrypted under a keystore beside its own data.
    {
        let storage = MemoryStorage::new(&u1, None).expect("u1 on its own");
        storage.store(&m1).expect("store u1");
    }
    assert!(u1.join("storage/keystore.json").exists());

    // Tenant two holds a keystore that encrypted nothing: what a tenant the
    // single-keystore guard refused was left with.
    std::fs::create_dir_all(u2.join("storage")).expect("u2 dir");
    Keystore::create(PASSPHRASE, KdfParams::production())
        .expect("stray keystore")
        .save_to_path(&u2.join("storage/keystore.json"))
        .expect("write stray keystore");

    std::env::set_var("SHODH_KEYSTORE_DIR", base.path());
    let s1 = MemoryStorage::new(&u1, None).expect("u1 under the root");
    assert!(
        base.path().join("keystore.json").exists(),
        "u1's keystore is the root's now"
    );
    assert!(!u1.join("storage/keystore.json").exists());

    let s2 = MemoryStorage::new(&u2, None).expect("u2 under the root, beside u1");
    assert!(
        u2.join("storage/keystore.json.unused").exists(),
        "u2's unused keystore is set aside, not deleted"
    );
    s2.store(&m2).expect("store u2");

    assert_eq!(
        s1.get(&id1).expect("get u1").experience.content,
        "tenant-one"
    );
    assert_eq!(
        s2.get(&id2).expect("get u2").experience.content,
        "tenant-two"
    );
    drop(s1);
    drop(s2);
    assert!(is_encrypted_record(&raw_record(&u2.join("storage"), &id2)));

    std::env::remove_var("SHODH_KEYSTORE_DIR");
    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
