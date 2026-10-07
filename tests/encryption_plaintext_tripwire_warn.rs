//! Contract: reading a plaintext record while a keystore is active is a
//! tripwire — counted, logged at WARN, and (via the lazy-migration path in
//! `get`) retired by rewriting the record encrypted. This is the case of a
//! store that predates its keystore; the strict variant lives in
//! `encryption_plaintext_tripwire_strict.rs` (separate process: the env var
//! and the process-global crypto both bleed across tests in one binary).

use shodh_memory::memory::storage::{plaintext_reads_under_keystore, MemoryStorage};
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use std::io::Write;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use uuid::Uuid;

const PASSPHRASE: &str = "rT4-tripwire-warn-correct-horse-Z9";
const PLAINTEXT: &str = "rT4-tripwire-warn-distinctive-plaintext-Z9";

/// Captures everything the tracing subscriber writes.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
fn plaintext_read_under_keystore_warns_and_is_reencrypted() {
    let capture = Capture::default();
    let sink = capture.clone();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .with_writer(move || sink.clone())
        .try_init()
        .expect("install capture subscriber");

    std::env::remove_var("SHODH_REQUIRE_ENCRYPTED_READS");
    std::env::remove_var("SHODH_MASTER_PASSPHRASE");

    let temp = TempDir::new().expect("temp dir");
    let db_path = temp.path().join("storage");
    let id = MemoryId(Uuid::new_v4());

    // A store written before any keystore existed: plaintext on disk.
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open plaintext store");
        assert!(!shodh_memory::memory::storage::encryption_active());
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
    assert!(before.starts_with(b"SHO"), "plaintext SHO envelope on disk");
    assert!(!temp.path().join("storage").join("keystore.json").exists());

    // The operator turns encryption on over the existing data.
    std::env::set_var("SHODH_MASTER_PASSPHRASE", PASSPHRASE);
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("open with new keystore");
        assert!(shodh_memory::memory::storage::encryption_active());
        assert_eq!(plaintext_reads_under_keystore(), 0);

        // The read succeeds (this is the warn mode, not the strict one)...
        let got = storage.get(&id).expect("plaintext record still readable");
        assert_eq!(got.experience.content, PLAINTEXT);

        // ...is counted...
        assert_eq!(plaintext_reads_under_keystore(), 1, "tripwire metric");

        // ...and logged at WARN.
        let log = String::from_utf8_lossy(&capture.0.lock().unwrap()).to_string();
        assert!(
            log.contains("WARN")
                && log.contains("plaintext memory record read while a keystore is active"),
            "expected the tripwire WARN in the captured log, got:\n{log}"
        );
    }

    // The lazy rewrite retired it: the record is now ciphertext on disk, and a
    // second read no longer trips.
    let after = raw_record(&db_path, &id);
    assert!(
        shodh_memory::keystore::is_encrypted_record(&after),
        "record read under a keystore is rewritten encrypted"
    );
    assert!(!after
        .windows(PLAINTEXT.len())
        .any(|w| w == PLAINTEXT.as_bytes()));
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("reopen");
        assert_eq!(storage.get(&id).expect("get").experience.content, PLAINTEXT);
        assert_eq!(
            plaintext_reads_under_keystore(),
            1,
            "an encrypted record does not trip the wire"
        );
    }

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
