//! Memory Consolidation Tests
//!
//! Tests for semantic consolidation and memory compression:
//! - Fact extraction from episodic memories
//! - Compression of old memories
//! - Tier migration (Working -> Session -> LongTerm -> Archive)
//! - Importance-based retention
//! - Auto-compression triggers

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{
    Experience, ExperienceType, Memory, MemoryConfig, MemoryId, MemorySystem, MemoryTier,
};
use shodh_memory::uuid::Uuid;
use tempfile::TempDir;

/// Create fallback NER for testing (rule-based, no ONNX required)
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Create experience with NER entity extraction
fn create_experience_with_ner(content: &str, ner: &NeuralNer) -> Experience {
    let entities = ner.extract(content).unwrap_or_default();
    let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
    Experience {
        experience_type: ExperienceType::Observation,
        content: content.to_string(),
        entities: entity_names,
        ..Default::default()
    }
}

/// Create test memory system
fn setup_memory_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 50,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 200,
        auto_compress: false,
        compression_age_days: 1,
        importance_threshold: 0.7,
    };

    let memory_system = MemorySystem::new(config).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

fn setup_auto_compress_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 10, // Small to trigger eviction
        session_memory_size_mb: 1,
        max_heap_per_user_mb: 10,
        auto_compress: true,
        compression_age_days: 0, // Immediately eligible
        importance_threshold: 0.5,
    };

    let memory_system = MemorySystem::new(config).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

fn create_experience(content: &str) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    }
}

// =============================================================================
// TIER MIGRATION TESTS
// =============================================================================

#[test]
fn test_memory_starts_in_working() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    assert_eq!(memory.tier, MemoryTier::Working);
}

#[test]
fn test_promote_working_to_session() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
}

#[test]
fn test_promote_session_to_longterm() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::Session;
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::LongTerm);
}

#[test]
fn test_promote_longterm_to_archive() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::LongTerm;
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Archive);
}

#[test]
fn test_promote_archive_stays_archive() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::Archive;
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Archive);
}

#[test]
fn test_demote_archive_to_longterm() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::Archive;
    memory.demote();
    assert_eq!(memory.tier, MemoryTier::LongTerm);
}

#[test]
fn test_demote_longterm_to_session() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::LongTerm;
    memory.demote();
    assert_eq!(memory.tier, MemoryTier::Session);
}

#[test]
fn test_demote_session_to_working() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::Session;
    memory.demote();
    assert_eq!(memory.tier, MemoryTier::Working);
}

#[test]
fn test_demote_working_stays_working() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::Working;
    memory.demote();
    assert_eq!(memory.tier, MemoryTier::Working);
}

#[test]
fn test_tier_full_cycle() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    // Promote all the way
    assert_eq!(memory.tier, MemoryTier::Working);
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::LongTerm);
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Archive);

    // Demote all the way
    memory.demote();
    assert_eq!(memory.tier, MemoryTier::LongTerm);
    memory.demote();
    assert_eq!(memory.tier, MemoryTier::Session);
    memory.demote();
    assert_eq!(memory.tier, MemoryTier::Working);
}

// =============================================================================
// TIER PRESERVATION TESTS
// =============================================================================

#[test]
fn test_tier_preserved_on_serialization() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::LongTerm;

    let serialized = bincode::serialize(&memory).unwrap();
    let deserialized: Memory = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.tier, MemoryTier::LongTerm);
}

#[test]
fn test_all_tiers_serialize() {
    for tier in [
        MemoryTier::Working,
        MemoryTier::Session,
        MemoryTier::LongTerm,
        MemoryTier::Archive,
    ] {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
        );
        memory.tier = tier;

        let serialized = bincode::serialize(&memory).unwrap();
        let deserialized: Memory = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.tier, tier);
    }
}

// =============================================================================
// TIER EQUALITY TESTS
// =============================================================================

#[test]
fn test_tier_equality() {
    assert_eq!(MemoryTier::Working, MemoryTier::Working);
    assert_eq!(MemoryTier::Session, MemoryTier::Session);
    assert_eq!(MemoryTier::LongTerm, MemoryTier::LongTerm);
    assert_eq!(MemoryTier::Archive, MemoryTier::Archive);
}

#[test]
fn test_tier_inequality() {
    assert_ne!(MemoryTier::Working, MemoryTier::Session);
    assert_ne!(MemoryTier::Session, MemoryTier::LongTerm);
    assert_ne!(MemoryTier::LongTerm, MemoryTier::Archive);
}

#[test]
fn test_tier_default() {
    assert_eq!(MemoryTier::default(), MemoryTier::Working);
}

// =============================================================================
// IMPORTANCE-BASED CONSOLIDATION TESTS
// =============================================================================

#[test]
fn test_high_importance_memories_preserved() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.9, // High importance
        None,
        None,
        None,
    );

    assert!(memory.importance() >= 0.7);
}

#[test]
fn test_low_importance_eligible_for_forget() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.2, // Low importance
        None,
        None,
        None,
    );

    assert!(memory.importance() < 0.5);
}

#[test]
fn test_importance_affects_retention() {
    let (mut system, _temp) = setup_memory_system();

    // Record high importance memory
    let high_id = system
        .record(Experience {
            content: "Very important observation".to_string(),
            experience_type: ExperienceType::Decision, // Decisions get higher importance
            ..Default::default()
        })
        .unwrap();

    // Record low importance memory
    let low_id = system
        .record(Experience {
            content: "Random observation".to_string(),
            experience_type: ExperienceType::Observation,
            ..Default::default()
        })
        .unwrap();

    // Both should exist
    assert!(system.get_memory(&high_id).is_ok());
    assert!(system.get_memory(&low_id).is_ok());
}

// =============================================================================
// MEMORY SYSTEM CONSOLIDATION TESTS
// =============================================================================

#[test]
fn test_working_memory_capacity() {
    let (mut system, _temp) = setup_auto_compress_system();

    // Record more memories than working memory capacity
    for i in 0..20 {
        system
            .record(create_experience(&format!("Memory {}", i)))
            .unwrap();
    }

    // System should have handled the overflow
    let stats = system.stats();
    assert!(stats.total_memories > 0);
}

#[test]
fn test_session_memory_used() {
    let (mut system, _temp) = setup_auto_compress_system();

    // Fill up working memory
    for i in 0..15 {
        system
            .record(create_experience(&format!("Overflow memory {}", i)))
            .unwrap();
    }

    let stats = system.stats();
    // Should have some memories distributed across tiers
    assert!(stats.total_memories > 0);
}

#[test]
fn test_graph_maintenance_succeeds() {
    let (system, _temp) = setup_memory_system();

    // Record some memories
    for i in 0..5 {
        system
            .record(create_experience(&format!("To consolidate {}", i)))
            .unwrap();
    }

    // Graph maintenance should not panic
    system.graph_maintenance();
}

// =============================================================================
// COMPRESSION STATE TESTS
// =============================================================================

#[test]
fn test_new_memory_not_compressed() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    assert!(!memory.compressed);
}

#[test]
fn test_compressed_flag_serializes() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.compressed = true;

    let serialized = bincode::serialize(&memory).unwrap();
    let deserialized: Memory = bincode::deserialize(&serialized).unwrap();

    assert!(deserialized.compressed);
}

// =============================================================================
// ACCESS PATTERN CONSOLIDATION TESTS
// =============================================================================

#[test]
fn test_frequently_accessed_stays_active() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    // Multiple accesses
    for _ in 0..10 {
        memory.record_access();
    }

    assert_eq!(memory.access_count(), 10);
}

#[test]
fn test_rarely_accessed_eligible_for_archive() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.3,
        None,
        None,
        None,
    );

    // No accesses
    assert_eq!(memory.access_count(), 0);
}

#[test]
fn test_access_updates_timestamp() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    let before = memory.last_accessed();
    std::thread::sleep(std::time::Duration::from_millis(10));
    memory.record_access();
    let after = memory.last_accessed();

    assert!(after >= before);
}

// =============================================================================
// MULTI-MEMORY CONSOLIDATION TESTS
// =============================================================================

#[test]
fn test_many_memories_graph_maintenance() {
    let (system, _temp) = setup_memory_system();

    // Create many memories
    for i in 0..50 {
        system
            .record(create_experience(&format!("Bulk memory {}", i)))
            .unwrap();
    }

    // Graph maintenance should work
    system.graph_maintenance();
}

#[test]
fn test_empty_system_graph_maintenance() {
    let (system, _temp) = setup_memory_system();

    // Graph maintenance on empty system
    system.graph_maintenance();
}

#[test]
fn test_multiple_graph_maintenance_calls() {
    let (system, _temp) = setup_memory_system();

    for i in 0..10 {
        system
            .record(create_experience(&format!("Memory {}", i)))
            .unwrap();
    }

    // Multiple graph maintenance calls should be idempotent
    for _ in 0..5 {
        system.graph_maintenance();
    }
}

// =============================================================================
// TIER AND IMPORTANCE COMBINATION TESTS
// =============================================================================

#[test]
fn test_high_importance_working_memory() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.9,
        None,
        None,
        None,
    );

    assert_eq!(memory.tier, MemoryTier::Working);
    assert!(memory.importance() > 0.8);

    // Promote should work
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
    // Importance preserved
    assert!(memory.importance() > 0.8);
}

#[test]
fn test_low_importance_archive() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.1,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::Archive;

    assert_eq!(memory.tier, MemoryTier::Archive);
    assert!(memory.importance() < 0.2);
}

#[test]
fn test_tier_independent_of_importance() {
    let mut high = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.9,
        None,
        None,
        None,
    );

    let mut low = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.1,
        None,
        None,
        None,
    );

    // Both can be promoted
    high.promote();
    low.promote();

    assert_eq!(high.tier, MemoryTier::Session);
    assert_eq!(low.tier, MemoryTier::Session);
}

// =============================================================================
// BATCH TIER OPERATIONS
// =============================================================================

#[test]
fn test_batch_promote() {
    let mut memories: Vec<Memory> = (0..10)
        .map(|_| {
            Memory::new(
                MemoryId(Uuid::new_v4()),
                Experience::default(),
                0.5,
                None,
                None,
                None,
            )
        })
        .collect();

    for m in &mut memories {
        m.promote();
    }

    for m in &memories {
        assert_eq!(m.tier, MemoryTier::Session);
    }
}

#[test]
fn test_batch_demote() {
    let mut memories: Vec<Memory> = (0..10)
        .map(|_| {
            let mut m = Memory::new(
                MemoryId(Uuid::new_v4()),
                Experience::default(),
                0.5,
                None,
                None,
                None,
            );
            m.tier = MemoryTier::Session;
            m
        })
        .collect();

    for m in &mut memories {
        m.demote();
    }

    for m in &memories {
        assert_eq!(m.tier, MemoryTier::Working);
    }
}

#[test]
fn test_heterogeneous_tier_batch() {
    let mut memories = Vec::new();

    for (i, tier) in [
        MemoryTier::Working,
        MemoryTier::Session,
        MemoryTier::LongTerm,
        MemoryTier::Archive,
    ]
    .iter()
    .enumerate()
    {
        let mut m = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
        );
        m.tier = *tier;
        memories.push((i, m));
    }

    // Promote all
    for (_, m) in &mut memories {
        m.promote();
    }

    // Check each moved up (or stayed at Archive)
    assert_eq!(memories[0].1.tier, MemoryTier::Session);
    assert_eq!(memories[1].1.tier, MemoryTier::LongTerm);
    assert_eq!(memories[2].1.tier, MemoryTier::Archive);
    assert_eq!(memories[3].1.tier, MemoryTier::Archive);
}

// =============================================================================
// EDGE CASES
// =============================================================================

#[test]
fn test_tier_after_many_operations() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    // Many promotions (should cap at Archive)
    for _ in 0..100 {
        memory.promote();
    }
    assert_eq!(memory.tier, MemoryTier::Archive);

    // Many demotions (should cap at Working)
    for _ in 0..100 {
        memory.demote();
    }
    assert_eq!(memory.tier, MemoryTier::Working);
}

#[test]
fn test_tier_with_zero_importance() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.0,
        None,
        None,
        None,
    );

    // Tier operations still work
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
}

#[test]
fn test_tier_with_max_importance() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        1.0,
        None,
        None,
        None,
    );

    // Tier operations still work
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
}

// =============================================================================
// GRAPH STATS TESTS
// =============================================================================

#[test]
fn test_stats_report_accurate() {
    let (system, _temp) = setup_memory_system();

    for i in 0..10 {
        system
            .record(create_experience(&format!("Stats test {}", i)))
            .unwrap();
    }

    let stats = system.stats();
    assert!(stats.total_memories >= 10);
}

#[test]
fn test_empty_system_stats() {
    let (system, _temp) = setup_memory_system();

    let stats = system.stats();
    assert_eq!(stats.total_memories, 0);
}

// =============================================================================
// CONCURRENT TIER OPERATIONS
// =============================================================================

#[test]
fn test_concurrent_tier_reads() {
    use std::sync::Arc;
    use std::thread;

    let memory = Arc::new(Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    ));

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let m = Arc::clone(&memory);
            thread::spawn(move || {
                for _ in 0..100 {
                    let _ = m.tier;
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
