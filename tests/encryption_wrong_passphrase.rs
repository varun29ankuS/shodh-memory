//! Contract: a wrong passphrase is a hard error at open — never a plaintext
//! store beside ciphertext, never ciphertext served as corruption. Own test
//! binary (one keystore per process).

use shodh_memory::memory::storage::MemoryStorage;
use shodh_memory::memory::types::{Experience, ExperienceType, Memory, MemoryId};
use tempfile::TempDir;
use uuid::Uuid;

#[test]
fn wrong_passphrase_is_hard_error() {
    let temp = TempDir::new().expect("temp dir");

    std::env::set_var("SHODH_MASTER_PASSPHRASE", "rT4-correct-horse-battery-Z9");
    {
        let storage = MemoryStorage::new(temp.path(), None).expect("create keystore");
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: "x".to_string(),
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

    // Reopen the SAME keystore with the WRONG passphrase: unseal must fail.
    std::env::set_var("SHODH_MASTER_PASSPHRASE", "the-wrong-passphrase");
    let err = match MemoryStorage::new(temp.path(), None) {
        Ok(_) => panic!("opening with the wrong passphrase must be a hard error"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("unseal") || err.contains("passphrase"),
        "error should name the unseal failure, got: {err}"
    );

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
