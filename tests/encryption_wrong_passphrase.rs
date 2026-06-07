//! e2e: a wrong passphrase is a hard error (never silent plaintext). Own test
//! binary (single keystore per process).

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
    assert!(
        MemoryStorage::new(temp.path(), None).is_err(),
        "opening with the wrong passphrase must be a hard error"
    );

    std::env::remove_var("SHODH_MASTER_PASSPHRASE");
}
