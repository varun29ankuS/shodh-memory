//! Contract: cleanup never deletes an encrypted record, because AEAD cannot tell
//! a wrong key from damaged bytes; and a keystore other than the one a database
//! is bound to is refused at open. Own test binary.

use shodh_memory::keystore::{is_encrypted_record, KdfParams, Keystore};
use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-binding-foreign-correct-horse-Z9";

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

fn with_db<T>(db_path: &std::path::Path, f: impl FnOnce(&rocksdb::DB) -> T) -> T {
    let opts = rocksdb::Options::default();
    let cfs = rocksdb::DB::list_cf(&opts, db_path).expect("list cfs");
    let db = rocksdb::DB::open_cf(&opts, db_path, &cfs).expect("open raw db");
    f(&db)
}

#[test]
fn cleanup_keeps_unreadable_ciphertext_and_a_foreign_keystore_is_refused() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);

    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let keystore_path = db_path.join("keystore.json");
    let (id_good, good) = sample("intact");
    let (id_bad, bad) = sample("damaged");

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open");
        storage.store(&good).expect("store good");
        storage.store(&bad).expect("store bad");
    }

    // Flip one byte of the AEAD tag: its epoch's DEK is held, decryption fails.
    with_db(&db_path, |db| {
        let mut value = db.get(id_bad.0.as_bytes()).expect("get").expect("present");
        let last = value.len() - 1;
        value[last] ^= 0x01;
        db.put(id_bad.0.as_bytes(), &value).expect("tamper");
    });

    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen");
        assert!(
            storage.get(&id_bad).is_err(),
            "the damaged record does not decrypt"
        );
        assert_eq!(
            storage.cleanup_corrupted().expect("cleanup"),
            0,
            "cleanup deletes no encrypted record"
        );
        assert_eq!(
            storage.get(&id_good).expect("get").experience.content,
            "intact"
        );
    }
    with_db(&db_path, |db| {
        let value = db
            .get(id_bad.0.as_bytes())
            .expect("get")
            .expect("still present");
        assert!(is_encrypted_record(&value));
    });

    let foreign = Keystore::create(PASSPHRASE, KdfParams::production()).expect("foreign keystore");
    foreign
        .save_to_path(&keystore_path)
        .expect("swap in a different keystore");
    let err = match MemoryStorage::new(temp.path(), None) {
        Ok(_) => panic!("a keystore other than the bound one must be refused"),
        Err(e) => format!("{e:#}"),
    };
    assert!(err.contains("bound to keystore"), "{err}");

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
