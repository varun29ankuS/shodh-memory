//! Contract: the record crypto is process-global (one keystore per process),
//! so a second store opened with a DIFFERENT keystore must fail loudly rather
//! than silently reuse the first store's keys — that would encrypt store B's
//! records under store A's DEK, readable by A's passphrase and unreadable by
//! B's own keystore. Own test binary.

use shodh_memory::memory::storage::MemoryStorage;
use tempfile::TempDir;

#[test]
fn second_store_with_different_keystore_is_rejected() {
    std::env::set_var("SHODH_MASTER_PASSPHRASE", "rT4-guard-correct-horse-Z9");

    let a = TempDir::new().expect("temp a");
    let b = TempDir::new().expect("temp b");

    // First store creates keystore A and installs the process-global crypto.
    let _store_a = MemoryStorage::new(a.path(), None).expect("store A opens");

    // Second store has its own (different-KEK) keystore B → must be refused.
    let err = match MemoryStorage::new(b.path(), None) {
        Ok(_) => panic!("a second store with a different keystore must be rejected"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("different encryption keystore"),
        "error should name the cross-keystore guard, got: {err}"
    );

    // The SAME keystore reopened is fine (a restart, or a second handle).
    drop(_store_a);
    MemoryStorage::new(a.path(), None).expect("store A reopens with its own keystore");

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
