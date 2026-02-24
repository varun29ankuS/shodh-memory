//! Memory Consolidation Tests
//!
//! Tests for semantic consolidation and memory compression:
//! - Fact extraction from episodic memories
//! - Compression of old memories
//! - Tier migration (Working -> Session -> LongTerm -> Archive)
//! - Importance-based retention
//! - Auto-compression triggers

use std::sync::Arc;

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::graph_memory::GraphMemory;
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

/// Create test memory system with knowledge graph
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

    let mut memory_system = MemorySystem::new(config).expect("Failed to create memory system");
    let graph_path = temp_dir.path().join("graph");
    let graph = GraphMemory::new(&graph_path).expect("Failed to create graph memory");
    memory_system.set_graph_memory(Arc::new(shodh_memory::parking_lot::RwLock::new(graph)));
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
    );
    memory.tier = MemoryTier::LongTerm;

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard()).unwrap();

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
            None, // created_at
        );
        memory.tier = tier;

        let serialized =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
        let (deserialized, _): (Memory, _) =
            bincode::serde::decode_from_slice(&serialized, bincode::config::standard()).unwrap();

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
        None, // created_at
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
        None, // created_at
    );

    assert!(memory.importance() < 0.5);
}

#[test]
fn test_importance_affects_retention() {
    let (mut system, _temp) = setup_memory_system();

    // Record high importance memory
    let high_id = system
        .remember(
            Experience {
                content: "Very important observation".to_string(),
                experience_type: ExperienceType::Decision, // Decisions get higher importance
                ..Default::default()
            },
            None,
        )
        .unwrap();

    // Record low importance memory
    let low_id = system
        .remember(
            Experience {
                content: "Random observation".to_string(),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            },
            None,
        )
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
            .remember(create_experience(&format!("Memory {}", i)), None)
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
            .remember(create_experience(&format!("Overflow memory {}", i)), None)
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
            .remember(create_experience(&format!("To consolidate {}", i)), None)
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
        None, // created_at
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
        None, // created_at
    );
    memory.compressed = true;

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard()).unwrap();

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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
            .remember(create_experience(&format!("Bulk memory {}", i)), None)
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
            .remember(create_experience(&format!("Memory {}", i)), None)
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
    );

    let mut low = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.1,
        None,
        None,
        None,
        None, // created_at
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
                None, // created_at
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
                None, // created_at
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
            None, // created_at
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
        None, // created_at
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
        None, // created_at
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
        None, // created_at
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
            .remember(create_experience(&format!("Stats test {}", i)), None)
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
        None, // created_at
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

// =============================================================================
// CONSOLIDATION INTROSPECTION TESTS (SHO-28)
// =============================================================================

use chrono::{Duration, TimeZone, Utc};
use shodh_memory::memory::{ConsolidationEvent, Query};

/// Helper to get a "beginning of time" DateTime for all-time queries
fn epoch() -> chrono::DateTime<chrono::Utc> {
    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
}

#[test]
fn test_consolidation_report_empty_system() {
    let (system, _temp) = setup_memory_system();

    // Empty system should produce valid report with zero counts
    let report = system.get_consolidation_report(epoch(), None);

    assert!(report.strengthened_memories.is_empty());
    assert!(report.decayed_memories.is_empty());
    assert!(report.formed_associations.is_empty());
    assert!(report.pruned_associations.is_empty());
    assert_eq!(report.statistics.maintenance_cycles, 0);
}

#[test]
fn test_consolidation_report_after_retrieval() {
    let (system, _temp) = setup_memory_system();

    // Record some memories
    let _id1 = system
        .remember(create_experience("Paris is the capital of France"), None)
        .unwrap();
    let _id2 = system
        .remember(create_experience("The Eiffel Tower is in Paris"), None)
        .unwrap();

    // Retrieve memories multiple times to trigger strengthening
    // (importance boosts after 5+ accesses)
    for _ in 0..7 {
        let query = Query {
            query_text: Some("Paris".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get report (epoch = all time)
    let report = system.get_consolidation_report(epoch(), None);

    // Should have strengthening events recorded
    // Note: strengthening only occurs when importance actually changes
    // which happens after 5+ accesses
    assert!(
        !report.strengthened_memories.is_empty()
            || !report.formed_associations.is_empty()
            || !report.strengthened_associations.is_empty(),
        "Expected at least one strengthening or association event"
    );
}

#[test]
fn test_consolidation_report_after_maintenance() {
    let (system, _temp) = setup_memory_system();

    // Record memories
    for i in 0..10 {
        system
            .remember(create_experience(&format!("Test memory {}", i)), None)
            .unwrap();
    }

    // Run maintenance to potentially trigger decay events (0.95 = standard decay factor)
    system.run_maintenance(0.95, "test-user", false).unwrap();

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Maintenance should have been recorded
    assert!(
        report.statistics.maintenance_cycles >= 1,
        "Expected at least 1 maintenance cycle"
    );
}

#[test]
fn test_consolidation_report_hebbian_learning() {
    let (system, _temp) = setup_memory_system();

    // Record memories with related content
    let _id1 = system
        .remember(
            create_experience("Rust is a systems programming language"),
            None,
        )
        .unwrap();
    let _id2 = system
        .remember(create_experience("Rust has ownership and borrowing"), None)
        .unwrap();
    let _id3 = system
        .remember(create_experience("Rust prevents memory leaks"), None)
        .unwrap();

    // Retrieve related memories together multiple times
    // This should trigger Hebbian co-activation (fire together, wire together)
    for _ in 0..3 {
        let query = Query {
            query_text: Some("Rust programming".to_string()),
            max_results: 3,
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Should have edge formation events from Hebbian learning
    // Note: edges form when 2+ memories are retrieved together
    assert!(
        !report.formed_associations.is_empty() || !report.strengthened_associations.is_empty(),
        "Expected Hebbian edge formation/strengthening events"
    );
}

#[test]
fn test_consolidation_report_time_filtering() {
    let (system, _temp) = setup_memory_system();

    // Record and retrieve memories
    let _id1 = system
        .remember(create_experience("Time-filtered test memory"), None)
        .unwrap();

    for _ in 0..7 {
        let query = Query {
            query_text: Some("Time-filtered".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get reports for different time periods
    let all_time = system.get_consolidation_report(epoch(), None);
    let one_hour_ago = Utc::now() - Duration::hours(1);
    let last_hour = system.get_consolidation_report(one_hour_ago, None);

    // All events should be within the last hour for this test, so they should be equal
    // (Since we just created them, last_hour should have same events as all_time)
    let all_time_count = all_time.strengthened_memories.len()
        + all_time.formed_associations.len()
        + all_time.statistics.maintenance_cycles;
    let last_hour_count = last_hour.strengthened_memories.len()
        + last_hour.formed_associations.len()
        + last_hour.statistics.maintenance_cycles;

    assert!(
        all_time_count >= last_hour_count,
        "AllTime ({}) should have >= events than LastHour ({})",
        all_time_count,
        last_hour_count
    );
}

#[test]
fn test_consolidation_event_buffer_clear() {
    let (system, _temp) = setup_memory_system();

    // Generate some events (need 2+ memories for coactivation)
    system
        .remember(
            create_experience("Buffer clear test with important data"),
            None,
        )
        .unwrap();
    system
        .remember(
            create_experience("Buffer overflow prevention in systems"),
            None,
        )
        .unwrap();
    let query = Query {
        query_text: Some("Buffer".to_string()),
        ..Default::default()
    };
    for _ in 0..7 {
        let _ = system.recall(&query);
    }

    // Count events before clear
    let events_before = system.get_all_consolidation_events().len();

    // Clear the buffer
    system.clear_consolidation_events();

    // Count events after clear
    let events_after = system.get_all_consolidation_events().len();

    // Second count should be zero or fewer
    assert!(
        events_after < events_before,
        "Events should be cleared: before={}, after={}",
        events_before,
        events_after
    );
    assert_eq!(events_after, 0, "Events should be completely cleared");
}

#[test]
fn test_consolidation_report_stats_consistency() {
    let (system, _temp) = setup_memory_system();

    // Record and interact with memories
    for i in 0..5 {
        system
            .remember(
                create_experience(&format!("Stats consistency test {}", i)),
                None,
            )
            .unwrap();
    }

    // Multiple retrieval rounds
    for _ in 0..3 {
        let query = Query {
            query_text: Some("Stats consistency".to_string()),
            max_results: 5,
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Run maintenance with standard decay factor
    system.run_maintenance(0.95, "test-user", false).unwrap();

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Verify stats are internally consistent
    // statistics counters should match the vector lengths
    assert_eq!(
        report.statistics.memories_strengthened,
        report.strengthened_memories.len(),
        "memories_strengthened stat should match vector length"
    );
    assert_eq!(
        report.statistics.memories_decayed,
        report.decayed_memories.len(),
        "memories_decayed stat should match vector length"
    );
    assert_eq!(
        report.statistics.edges_formed,
        report.formed_associations.len(),
        "edges_formed stat should match vector length"
    );
    assert_eq!(
        report.statistics.edges_strengthened,
        report.strengthened_associations.len(),
        "edges_strengthened stat should match vector length"
    );
}

#[test]
fn test_memory_strengthening_records_before_after() {
    let (system, _temp) = setup_memory_system();

    // Record a memory with moderate initial importance
    let _id = system
        .remember(
            Experience {
                content: "Strengthening before/after test".to_string(),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            },
            None,
        )
        .unwrap();

    // Access the memory many times to trigger importance boost
    for _ in 0..10 {
        let query = Query {
            query_text: Some("before/after test".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Verify strengthening events have valid before/after values
    for change in &report.strengthened_memories {
        assert!(
            change.activation_after >= change.activation_before,
            "activation_after ({}) should be >= activation_before ({})",
            change.activation_after,
            change.activation_before
        );
    }
}

#[test]
fn test_edge_events_have_strength_values() {
    let (system, _temp) = setup_memory_system();

    // Record related memories
    let _id1 = system
        .remember(create_experience("Edge test: topic A related"), None)
        .unwrap();
    let _id2 = system
        .remember(create_experience("Edge test: topic A connected"), None)
        .unwrap();

    // Retrieve together to form edges
    for _ in 0..5 {
        let query = Query {
            query_text: Some("Edge test: topic A".to_string()),
            max_results: 2,
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Verify edge events have valid strength values
    for assoc in &report.formed_associations {
        // strength_after contains the initial_strength for newly formed associations
        assert!(
            assoc.strength_after > 0.0 && assoc.strength_after <= 1.0,
            "strength_after (initial) should be in (0, 1]"
        );
    }

    for assoc in &report.strengthened_associations {
        // strength_before is Option<f32>, so we check if it exists
        if let Some(before) = assoc.strength_before {
            assert!(
                assoc.strength_after >= before,
                "strength_after ({}) should be >= strength_before ({}) for strengthening",
                assoc.strength_after,
                before
            );
        }
    }
}

#[test]
fn test_consolidation_events_list() {
    let (system, _temp) = setup_memory_system();

    // Record multiple memories (coactivation needs 2+)
    system
        .remember(
            create_experience("Test consolidation events list for tracking"),
            None,
        )
        .unwrap();
    system
        .remember(
            create_experience("Consolidation events are important for monitoring"),
            None,
        )
        .unwrap();

    for _ in 0..7 {
        let query = Query {
            query_text: Some("consolidation events".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get all events directly
    let events = system.get_all_consolidation_events();

    // Should have some events recorded
    assert!(
        !events.is_empty(),
        "Expected some consolidation events to be recorded"
    );

    // Verify each event has a valid timestamp
    for event in &events {
        let timestamp = event.timestamp();
        let now = Utc::now();
        // Event should have been created within the last minute
        assert!(
            timestamp <= now,
            "Event timestamp should not be in the future"
        );
    }
}

#[test]
fn test_consolidation_events_since_filter() {
    let (system, _temp) = setup_memory_system();

    // Record a memory and generate some events
    system
        .remember(create_experience("Test events since filter"), None)
        .unwrap();

    let start_time = Utc::now();

    for _ in 0..5 {
        let query = Query {
            query_text: Some("events since".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get events since the start time
    let recent_events = system.get_consolidation_events_since(start_time);

    // All returned events should be >= start_time
    for event in &recent_events {
        assert!(
            event.timestamp() >= start_time,
            "Event timestamp should be >= filter time"
        );
    }
}

#[test]
fn test_consolidation_event_count() {
    let (system, _temp) = setup_memory_system();

    // Initially should have zero events
    let initial_count = system.consolidation_event_count();
    assert_eq!(initial_count, 0, "Initial event count should be zero");

    // Record memories and do some retrievals (coactivation needs 2+)
    system
        .remember(create_experience("Test event count tracking system"), None)
        .unwrap();
    system
        .remember(
            create_experience("Event count should increase with operations"),
            None,
        )
        .unwrap();

    for _ in 0..7 {
        let query = Query {
            query_text: Some("event count".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Should have more events now
    let final_count = system.consolidation_event_count();
    assert!(
        final_count > initial_count,
        "Event count should increase after operations"
    );
}
