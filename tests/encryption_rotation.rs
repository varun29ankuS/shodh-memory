//! Contract: data-key rotation is safe end to end. After `rotate_dek` (what
//! `shodh-keyctl rotate-dek` does, persisted before the store reopens) a
//! record written under epoch 0 is still readable, new writes go out under
//! epoch 1, `migrate_legacy` leaves already-encrypted records alone, and a
//! keystore rolled back to an older generation is refused. Own test binary.

use shodh_memory::keystore::{is_encrypted_record, record_epoch, Keystore};
use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-rotation-correct-horse-battery-Z9";

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
fn dek_rotation_keeps_old_epoch_readable_and_moves_new_writes() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);
    std::env::remove_var("SHODH_ALLOW_KEYSTORE_ROLLBACK");

    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let keystore_path = db_path.join("keystore.json");
    let (id_old, mem_old) = sample("written-under-epoch-0");
    let (id_new, mem_new) = sample("written-under-epoch-1");

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open");
        storage.store(&mem_old).expect("store old");
    }
    assert_eq!(record_epoch(&raw_record(&db_path, &id_old)), Some(0));
    let gen0 = std::fs::read_to_string(&keystore_path).expect("keystore json");

    // Offline rotation, as shodh-keyctl does it: unseal, rotate, persist.
    {
        let mut ks = Keystore::from_json(&gen0).expect("parse keystore");
        let kek = ks.unseal_with_passphrase(PASSPHRASE).expect("unseal");
        ks.verify_integrity(&kek).expect("sealed");
        assert_eq!(ks.rotate_dek(&kek).expect("rotate"), 1);
        ks.save_to_path(&keystore_path)
            .expect("persist rotated keystore");
        assert!(
            keystore_path.with_extension("json.bak").exists(),
            "the pre-rotation keystore is kept as a backup"
        );
    }

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen after rotation");
        // The epoch-0 record decrypts under its retired DEK...
        assert_eq!(
            storage.get(&id_old).expect("get old").experience.content,
            "written-under-epoch-0"
        );
        // ...and a new write goes out under the active epoch.
        storage.store(&mem_new).expect("store new");
        assert_eq!(
            storage.get(&id_new).expect("get new").experience.content,
            "written-under-epoch-1"
        );

        // migrate_legacy sees both as current: encrypted and postcard.
        let (migrated, already_current, failed) = storage.migrate_legacy().expect("migrate");
        assert_eq!(
            (migrated, failed),
            (0, 0),
            "nothing to migrate, nothing failed"
        );
        assert!(already_current >= 2);
    }
    let raw_old = raw_record(&db_path, &id_old);
    let raw_new = raw_record(&db_path, &id_new);
    assert!(is_encrypted_record(&raw_old) && is_encrypted_record(&raw_new));
    assert_eq!(record_epoch(&raw_new), Some(1), "new writes use epoch 1");
    // A read does not rewrite an already-encrypted old-epoch record (lazy
    // re-encryption is for plaintext; re-keying old epochs is an explicit
    // migration, not a side effect of a read).
    assert_eq!(record_epoch(&raw_old), Some(0));

    // Rollback guard: restoring the pre-rotation keystore (generation 0 after
    // the DB has seen generation 1) is refused — and that refusal is not
    // reachable through the process-global cryptor set, which still holds
    // epoch 1: the DB-side sentinel is what says no.
    std::fs::write(&keystore_path, &gen0).expect("restore old keystore");
    let err = match MemoryStorage::new(temp.path(), None) {
        Ok(_) => panic!("an older keystore generation must be refused"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("keystore rollback detected"),
        "error must name the rollback guard, got: {err}"
    );

    // The documented override accepts it once and resets the sentinel.
    std::env::set_var("SHODH_ALLOW_KEYSTORE_ROLLBACK", "true");
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("rollback accepted");
        assert_eq!(
            storage.get(&id_old).expect("get old").experience.content,
            "written-under-epoch-0"
        );
    }
    std::env::remove_var("SHODH_ALLOW_KEYSTORE_ROLLBACK");
    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
