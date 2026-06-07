//! e2e: validates the cross-keystore guard (audit fix). shodh's encryption is
//! process-global = a single keystore per process; a second store opened with a
//! DIFFERENT keystore must fail loudly rather than silently reuse the first
//! store's keys (which would be cross-keystore data confusion). Own test binary.

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
    assert!(
        MemoryStorage::new(b.path(), None).is_err(),
        "a second store with a different keystore must be rejected by the global guard"
    );

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
