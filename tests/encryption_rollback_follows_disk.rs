//! Contract: an accepted keystore rollback takes effect in the running process.
//! After restoring an older keystore under SHODH_ALLOW_KEYSTORE_ROLLBACK, new
//! records are written under an epoch the keystore on disk holds, so they stay
//! readable after a restart, and the binding is reset to the restored
//! generation. Own test binary.

use shodh_memory::keystore::{record_epoch, Keystore, RecordCryptors};
use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-rollback-disk-correct-horse-Z9";

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
fn accepted_rollback_writes_under_the_restored_keystore() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);
    std::env::remove_var("SHODH_ALLOW_KEYSTORE_ROLLBACK");

    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let keystore_path = db_path.join("keystore.json");
    let (id_first, first) = sample("before-rotation");
    let (id_after, after) = sample("after-rollback");

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open");
        storage.store(&first).expect("store");
    }
    let gen0 = std::fs::read_to_string(&keystore_path).expect("keystore json");

    {
        let mut ks = Keystore::from_json(&gen0).expect("parse");
        let kek = ks.unseal_with_passphrase(PASSPHRASE).expect("unseal");
        assert_eq!(ks.rotate_dek(&kek).expect("rotate"), 1);
        ks.save_to_path(&keystore_path).expect("persist rotation");
    }
    drop(MemoryStorage::new(temp.path(), None).expect("reopen at generation 1"));

    std::fs::write(&keystore_path, &gen0).expect("restore the generation-0 keystore");
    std::env::set_var("SHODH_ALLOW_KEYSTORE_ROLLBACK", "true");
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("rollback accepted");
        storage.store(&after).expect("store after rollback");
        assert_eq!(
            storage.get(&id_first).expect("get").experience.content,
            "before-rotation"
        );
    }
    std::env::remove_var("SHODH_ALLOW_KEYSTORE_ROLLBACK");

    let raw = raw_record(&db_path, &id_after);
    assert_eq!(
        record_epoch(&raw),
        Some(0),
        "written under the epoch the restored keystore holds, not the rotated-away one"
    );

    // What a restart would build: the cryptors from the keystore on disk.
    let ks = Keystore::from_json(&std::fs::read_to_string(&keystore_path).expect("read"))
        .expect("parse");
    let kek = ks.unseal_with_passphrase(PASSPHRASE).expect("unseal");
    let cryptors = RecordCryptors::from_keystore(&ks, &kek).expect("cryptors");
    cryptors
        .for_epoch(0)
        .expect("epoch 0 held on disk")
        .decrypt_record(&raw, id_after.0.as_bytes())
        .expect("readable after a restart");

    MemoryStorage::new(temp.path(), None).expect("the binding was reset: no override needed now");

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
