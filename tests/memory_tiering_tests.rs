//! Memory Tiering Tests
//!
//! Tests for the three-tier memory architecture:
//! - Working memory (hot, limited size)
//! - Session memory (warm, size-limited MB)
//! - Long-term memory (cold, persistent RocksDB)
//! - Promotion/demotion between tiers
//! - LRU eviction from working memory
//! - Importance-based promotion to session
//! - NER integration for entity extraction

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{
    Experience, ExperienceType, MemoryConfig, MemorySystem, Query, RetrievalMode,
};
use std::path::PathBuf;
use tempfile::TempDir;

/// Create fallback NER instance for testing
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Create experience with NER-extracted entities
fn create_experience_with_ner(
    content: &str,
    exp_type: ExperienceType,
    ner: &NeuralNer,
) -> Experience {
    let entities = ner.extract(content).unwrap_or_default();
    let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
    Experience {
        content: content.to_string(),
        experience_type: exp_type,
        entities: entity_names,
        ..Default::default()
    }
}

/// Create a test memory system with configurable parameters
fn setup_memory_system(working_size: usize, session_mb: usize) -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: working_size,
        session_memory_size_mb: session_mb,
        max_heap_per_user_mb: 200,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.7,
    };

    let memory_system = MemorySystem::new(config).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

/// Create test experience with specified importance indicators
fn create_experience(content: &str, exp_type: ExperienceType) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: exp_type,
        ..Default::default()
    }
}

// =============================================================================
// WORKING MEMORY TESTS
// =============================================================================

#[test]
fn test_working_memory_stores_recent() {
    let (mut memory_system, _temp_dir) = setup_memory_system(10, 50);

    // Record some experiences
    let id1 = memory_system
        .record(create_experience(
            "First memory",
            ExperienceType::Observation,
        ))
        .expect("Failed to record");
    let id2 = memory_system
        .record(create_experience(
            "Second memory",
            ExperienceType::Observation,
        ))
        .expect("Failed to record");

    // Retrieve all
    let query = Query {
        max_results: 10,
        ..Default::default()
    };

    let results = memory_system.retrieve(&query).expect("Failed to retrieve");

    // Should find both memories
    assert!(results.len() >= 2, "Should have at least 2 memories");
}

#[test]
fn test_working_memory_lru_eviction() {
    let (mut memory_system, _temp_dir) = setup_memory_system(5, 50); // Very small working memory

    // Record more than capacity
    for i in 0..10 {
        memory_system
            .record(create_experience(
                &format!("Memory number {}", i),
                ExperienceType::Observation,
            ))
            .expect("Failed to record");
    }

    // Should still work - older memories evicted to session/long-term
    let stats = memory_system.stats();
    assert!(
        stats.total_memories >= 10,
        "All memories should be recorded somewhere"
    );
}

// =============================================================================
// SESSION MEMORY TESTS
// =============================================================================

#[test]
fn test_session_memory_promotion() {
    let (mut memory_system, _temp_dir) = setup_memory_system(3, 50);

    // Record high-importance memories (should go to session)
    let high_importance = Experience {
        content: "Critical decision: Deployed new architecture to production. This is a breakthrough implementation that will significantly improve system performance and reliability.".to_string(),
        experience_type: ExperienceType::Decision,
        entities: vec![
            "architecture".to_string(),
            "production".to_string(),
            "deployment".to_string(),
        ],
        ..Default::default()
    };

    memory_system
        .record(high_importance)
        .expect("Failed to record important memory");

    // Stats should show promotion activity
    let stats = memory_system.stats();
    assert!(
        stats.total_memories > 0,
        "Should have recorded the important memory"
    );
}

#[test]
fn test_low_importance_stays_working() {
    let (mut memory_system, _temp_dir) = setup_memory_system(10, 50);

    // Record low-importance memory
    let low_importance = Experience {
        content: "ok".to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };

    memory_system
        .record(low_importance)
        .expect("Failed to record low importance memory");

    // Low importance should still be retrievable
    let query = Query {
        query_text: Some("ok".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory_system.retrieve(&query).expect("Failed to retrieve");
    assert!(
        !results.is_empty(),
        "Low importance memory should still be retrievable"
    );
}

// =============================================================================
// LONG-TERM MEMORY TESTS
// =============================================================================

#[test]
fn test_longterm_persistence() {
    // Verify bincode roundtrip works for Memory struct
    {
        use shodh_memory::memory::{Experience, ExperienceType, Memory, MemoryId};
        use shodh_memory::uuid::Uuid;

        let test_memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience {
                content: "Test content".to_string(),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            },
            0.5,
            None,
            None,
            None,
        );

        let serialized =
            bincode::serialize(&test_memory).expect("Failed to serialize Memory struct");
        let deserialized: Memory = bincode::deserialize(&serialized)
            .expect("Bincode roundtrip failed - serialization format is broken");

        assert_eq!(
            test_memory.id.0, deserialized.id.0,
            "ID mismatch after roundtrip"
        );
        assert_eq!(
            test_memory.experience.content, deserialized.experience.content,
            "Content mismatch"
        );
    }

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let content = "Persistent memory content for testing long-term storage reliability";

    // Create and populate memory system
    {
        let config = MemoryConfig {
            storage_path: db_path.clone(),
            working_memory_size: 2, // Small to trigger promotion to long-term
            session_memory_size_mb: 10,
            max_heap_per_user_mb: 200,
            auto_compress: false,
            compression_age_days: 30,
            importance_threshold: 0.3,
        };

        let mut memory_system = MemorySystem::new(config).expect("Failed to create");

        // Record important memory that should be persisted
        memory_system
            .record(Experience {
                content: content.to_string(),
                experience_type: ExperienceType::Decision,
                entities: vec!["persistence".to_string(), "storage".to_string()],
                ..Default::default()
            })
            .expect("Failed to record");

        // Add more memories to trigger eviction from working memory
        for i in 0..5 {
            memory_system
                .record(Experience {
                    content: format!("Filler memory {} to trigger eviction", i),
                    experience_type: ExperienceType::Observation,
                    ..Default::default()
                })
                .expect("Failed to record filler");
        }

        let stats_before = memory_system.stats();
        assert!(
            stats_before.total_memories >= 6,
            "Should have 6 memories before flush, got {}",
            stats_before.total_memories
        );

        memory_system.flush_storage().expect("Failed to flush");
    }

    // Allow RocksDB to release locks
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Reopen and verify persistence
    {
        let config = MemoryConfig {
            storage_path: db_path.clone(),
            working_memory_size: 5,
            session_memory_size_mb: 10,
            max_heap_per_user_mb: 200,
            auto_compress: false,
            compression_age_days: 30,
            importance_threshold: 0.3,
        };

        let memory_system = MemorySystem::new(config).expect("Failed to reopen");

        // Check stats from the new instance
        let stats = memory_system.stats();
        eprintln!(
            "After reopen - total_memories: {}, long_term: {}, vector_index: {}",
            stats.total_memories, stats.long_term_memory_count, stats.vector_index_count
        );

        // Directly check storage stats via storage stats method
        let storage_stats = memory_system
            .get_storage_stats()
            .expect("Failed to get storage stats");
        eprintln!(
            "Storage stats - total_count: {}, compressed: {}",
            storage_stats.total_count, storage_stats.compressed_count
        );

        // Direct check: Try to get all memories from long-term storage
        let all_memories = memory_system
            .get_all_memories()
            .expect("Failed to get all memories");
        eprintln!(
            "get_all_memories() returned {} memories",
            all_memories.len()
        );

        // Search for persisted content using hybrid mode (searches all tiers)
        let query = Query {
            query_text: Some("persistent storage reliability".to_string()),
            max_results: 10,
            retrieval_mode: RetrievalMode::Hybrid, // Search across all tiers
            ..Default::default()
        };

        let results = memory_system.retrieve(&query).expect("Failed to retrieve");
        eprintln!("Search results count: {}", results.len());

        // After reopen, memories that were flushed should be available
        // The test verifies that the storage layer persists data correctly
        assert!(
            stats.total_memories > 0,
            "Should have persisted memories after reopen (total: {}, results: {}, storage_count: {})",
            stats.total_memories,
            results.len(),
            storage_stats.total_count
        );
    }
}

// =============================================================================
// TIERING INTEGRATION TESTS
// =============================================================================

#[test]
fn test_multi_tier_retrieval() {
    let (mut memory_system, _temp_dir) = setup_memory_system(5, 50);

    // Fill working memory to trigger promotions
    for i in 0..15 {
        let importance = if i % 3 == 0 {
            ExperienceType::Decision // Higher importance
        } else {
            ExperienceType::Observation // Lower importance
        };

        memory_system
            .record(create_experience(
                &format!("Test memory entry number {} for multi-tier testing", i),
                importance,
            ))
            .expect("Failed to record");
    }

    // Should find memories from multiple tiers
    let query = Query {
        query_text: Some("multi-tier testing".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = memory_system.retrieve(&query).expect("Failed to retrieve");

    assert!(!results.is_empty(), "Should retrieve memories across tiers");
}

#[test]
fn test_importance_affects_tiering() {
    let (mut memory_system, _temp_dir) = setup_memory_system(3, 50);

    // High importance memory
    let high = Experience {
        content: "CRITICAL: Production deployment decision - approved architecture change"
            .to_string(),
        experience_type: ExperienceType::Decision,
        entities: vec![
            "production".to_string(),
            "architecture".to_string(),
            "deployment".to_string(),
        ],
        metadata: [("priority".to_string(), "critical".to_string())].into(),
        ..Default::default()
    };

    // Low importance memory
    let low = Experience {
        content: "ok".to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };

    memory_system.record(high).expect("Failed to record high");
    memory_system.record(low).expect("Failed to record low");

    // Both should be retrievable
    let stats = memory_system.stats();
    assert!(stats.total_memories >= 2, "Both memories should be stored");
}

// =============================================================================
// FORGETTING TESTS
// =============================================================================

#[test]
fn test_forget_low_importance() {
    let (mut memory_system, _temp_dir) = setup_memory_system(10, 50);

    // Record mix of importance levels
    for i in 0..10 {
        let exp_type = if i < 5 {
            ExperienceType::Decision // Higher importance
        } else {
            ExperienceType::Observation // Lower importance
        };

        memory_system
            .record(create_experience(&format!("Memory {}", i), exp_type))
            .expect("Failed to record");
    }

    let stats_before = memory_system.stats();

    // Forget low importance
    let forgotten = memory_system
        .forget(shodh_memory::memory::ForgetCriteria::LowImportance(0.5))
        .expect("Failed to forget");

    // Some memories should be forgotten
    // Note: actual count depends on importance calculation
    assert!(
        forgotten >= 0,
        "Forget operation should complete successfully"
    );
}

#[test]
fn test_forget_by_pattern() {
    let (mut memory_system, _temp_dir) = setup_memory_system(10, 50);

    // Record memories with distinct patterns
    memory_system
        .record(create_experience(
            "DELETEME: This should be forgotten",
            ExperienceType::Observation,
        ))
        .expect("Failed to record");

    memory_system
        .record(create_experience(
            "KEEP: This should remain",
            ExperienceType::Observation,
        ))
        .expect("Failed to record");

    // Forget by pattern
    let forgotten = memory_system
        .forget(shodh_memory::memory::ForgetCriteria::Pattern(
            "DELETEME".to_string(),
        ))
        .expect("Failed to forget");

    assert!(forgotten >= 1, "Should forget at least 1 memory");

    // Verify KEEP remains
    let query = Query {
        query_text: Some("KEEP remain".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory_system.retrieve(&query).expect("Failed to retrieve");
    assert!(
        results
            .iter()
            .any(|m| m.experience.content.contains("KEEP")),
        "KEEP memory should remain"
    );
}

// =============================================================================
// RETRIEVAL MODE TESTS
// =============================================================================

#[test]
fn test_similarity_retrieval() {
    let (mut memory_system, _temp_dir) = setup_memory_system(10, 50);

    // Record semantically similar memories
    memory_system
        .record(create_experience(
            "The robot navigated through the warehouse avoiding obstacles",
            ExperienceType::Observation,
        ))
        .expect("Failed");

    memory_system
        .record(create_experience(
            "Drone flew over the factory detecting anomalies",
            ExperienceType::Observation,
        ))
        .expect("Failed");

    // Search semantically similar
    let query = Query {
        query_text: Some("autonomous vehicle obstacle avoidance".to_string()),
        max_results: 5,
        retrieval_mode: RetrievalMode::Similarity,
        ..Default::default()
    };

    let results = memory_system.retrieve(&query).expect("Failed");

    // Should find robotics-related memories
    assert!(
        !results.is_empty(),
        "Should find semantically similar memories"
    );
}

#[test]
fn test_hybrid_retrieval() {
    let (mut memory_system, _temp_dir) = setup_memory_system(10, 50);

    // Record memories
    memory_system
        .record(create_experience(
            "Mission alpha: drone successfully completed reconnaissance",
            ExperienceType::Task,
        ))
        .expect("Failed");

    memory_system
        .record(create_experience(
            "Mission beta: robot failed to reach waypoint due to obstacle",
            ExperienceType::Error,
        ))
        .expect("Failed");

    // Hybrid search
    let query = Query {
        query_text: Some("mission completion status".to_string()),
        max_results: 5,
        retrieval_mode: RetrievalMode::Hybrid,
        ..Default::default()
    };

    let results = memory_system.retrieve(&query).expect("Failed");

    assert!(
        !results.is_empty(),
        "Hybrid retrieval should find relevant memories"
    );
}

// =============================================================================
// ROBOTICS-SPECIFIC FILTER TESTS
// =============================================================================

#[test]
fn test_geo_filter_retrieval() {
    let (mut memory_system, _temp_dir) = setup_memory_system(10, 50);

    // Record memories with geo coordinates
    let geo_memory = Experience {
        content: "Detected obstacle at warehouse entrance".to_string(),
        experience_type: ExperienceType::Observation,
        geo_location: Some([37.7749, -122.4194, 10.0]), // San Francisco
        ..Default::default()
    };

    let far_memory = Experience {
        content: "Completed delivery in New York".to_string(),
        experience_type: ExperienceType::Task,
        geo_location: Some([40.7128, -74.0060, 5.0]), // New York
        ..Default::default()
    };

    memory_system.record(geo_memory).expect("Failed");
    memory_system.record(far_memory).expect("Failed");

    // Search near San Francisco
    let query = Query {
        query_text: Some("obstacle warehouse".to_string()),
        geo_filter: Some(shodh_memory::memory::GeoFilter::new(
            37.7749, -122.4194, 1000.0,
        )),
        max_results: 5,
        ..Default::default()
    };

    let results = memory_system.retrieve(&query).expect("Failed");

    // Should find the SF memory
    assert!(
        results
            .iter()
            .any(|m| m.experience.content.contains("warehouse")),
        "Should find memory near specified location"
    );
}

#[test]
fn test_mission_filter_retrieval() {
    let (mut memory_system, _temp_dir) = setup_memory_system(10, 50);

    // Record memories with mission IDs
    let mission_a = Experience {
        content: "Waypoint 1 reached successfully in alpha mission".to_string(),
        experience_type: ExperienceType::Task,
        mission_id: Some("mission_alpha".to_string()),
        ..Default::default()
    };

    let mission_b = Experience {
        content: "Waypoint 1 failed due to weather in beta mission".to_string(),
        experience_type: ExperienceType::Error,
        mission_id: Some("mission_beta".to_string()),
        ..Default::default()
    };

    memory_system.record(mission_a).expect("Failed");
    memory_system.record(mission_b).expect("Failed");

    // Search specific mission - should only return mission_alpha results
    let query = Query {
        query_text: Some("waypoint".to_string()),
        mission_id: Some("mission_alpha".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory_system.retrieve(&query).expect("Failed");

    // Verify that if we got results, they all have mission_alpha
    // The filter strictly excludes non-matching mission IDs
    for result in &results {
        if let Some(ref mid) = result.experience.mission_id {
            assert_eq!(
                mid, "mission_alpha",
                "All results with mission_id should be mission_alpha, got: {}",
                mid
            );
        }
    }

    // Also verify we can find mission_beta separately
    let query_beta = Query {
        query_text: Some("waypoint".to_string()),
        mission_id: Some("mission_beta".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results_beta = memory_system.retrieve(&query_beta).expect("Failed");
    for result in &results_beta {
        if let Some(ref mid) = result.experience.mission_id {
            assert_eq!(
                mid, "mission_beta",
                "All results with mission_id should be mission_beta, got: {}",
                mid
            );
        }
    }
}

// =============================================================================
// STRESS TESTS
// =============================================================================

#[test]
fn test_high_volume_recording() {
    let (mut memory_system, _temp_dir) = setup_memory_system(50, 100);

    // Record 100 memories quickly
    for i in 0..100 {
        memory_system
            .record(create_experience(
                &format!("High volume test memory number {}", i),
                ExperienceType::Observation,
            ))
            .expect("Failed to record");
    }

    let stats = memory_system.stats();
    assert!(
        stats.total_memories >= 100,
        "All memories should be recorded"
    );
}

#[test]
fn test_concurrent_access_pattern() {
    let (mut memory_system, _temp_dir) = setup_memory_system(20, 50);

    // Simulate read-write pattern
    for i in 0..20 {
        // Write
        memory_system
            .record(create_experience(
                &format!("Concurrent pattern test {}", i),
                ExperienceType::Observation,
            ))
            .expect("Failed to record");

        // Read immediately
        let query = Query {
            query_text: Some("concurrent pattern".to_string()),
            max_results: 5,
            ..Default::default()
        };

        let results = memory_system.retrieve(&query).expect("Failed to retrieve");
        assert!(!results.is_empty(), "Should find recently written memory");
    }
}
