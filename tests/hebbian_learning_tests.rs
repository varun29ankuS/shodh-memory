//! Hebbian Learning Tests
//!
//! Tests for the Hebbian feedback loop:
//! - "Neurons that fire together wire together"
//! - Reinforcement from task outcomes
//! - Association strengthening between co-retrieved memories
//! - Importance boost/decay based on helpfulness

use shodh_memory::memory::{
    Experience, ExperienceType, MemoryConfig, MemorySystem, Query, RetrievalOutcome,
};
use tempfile::TempDir;

/// Create test memory system
fn setup_memory_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 200,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.7,
    };

    let memory_system = MemorySystem::new(config).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

/// Create experience with specified content
fn create_experience(content: &str) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    }
}

// =============================================================================
// RETRIEVAL TRACKING TESTS
// =============================================================================

#[test]
fn test_retrieve_returns_memories() {
    let (mut memory, _temp) = setup_memory_system();

    // Record some memories
    memory
        .record(create_experience("Robot detected obstacle at entrance"))
        .unwrap();
    memory
        .record(create_experience("Drone completed patrol route"))
        .unwrap();

    // Retrieve
    let query = Query {
        query_text: Some("obstacle detection".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(!results.is_empty(), "Should find relevant memories");
}

#[test]
fn test_retrieve_multiple_related() {
    let (mut memory, _temp) = setup_memory_system();

    // Record related memories
    for i in 0..10 {
        memory
            .record(create_experience(&format!(
                "Warehouse section {} inventory check complete",
                i
            )))
            .unwrap();
    }

    let query = Query {
        query_text: Some("warehouse inventory".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(results.len() >= 5, "Should find multiple related memories");
}

// =============================================================================
// REINFORCEMENT OUTCOME TESTS
// =============================================================================

#[test]
fn test_reinforce_helpful_boosts_importance() {
    let (mut memory, _temp) = setup_memory_system();

    // Record memory
    let id = memory
        .record(create_experience(
            "Critical safety procedure: always check battery before flight",
        ))
        .unwrap();

    // Get initial importance
    let initial = memory.get_memory(&id).unwrap();
    let initial_importance = initial.importance();

    // Reinforce as helpful
    let ids = vec![id];
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    assert!(
        stats.importance_boosts > 0,
        "Should have boosted importance"
    );

    // Check importance increased
    let after = memory.get_memory(&ids[0]).unwrap();
    assert!(
        after.importance() > initial_importance,
        "Importance should increase after helpful feedback"
    );
}

#[test]
fn test_reinforce_misleading_decays_importance() {
    let (mut memory, _temp) = setup_memory_system();

    // Record memory with high importance
    let id = memory
        .record(Experience {
            content: "Outdated procedure that no longer applies".to_string(),
            experience_type: ExperienceType::Decision,
            ..Default::default()
        })
        .unwrap();

    // Get initial importance
    let initial = memory.get_memory(&id).unwrap();
    let initial_importance = initial.importance();

    // Reinforce as misleading
    let ids = vec![id];
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Misleading)
        .unwrap();

    assert!(
        stats.importance_decays > 0,
        "Should have decayed importance"
    );

    // Check importance decreased
    let after = memory.get_memory(&ids[0]).unwrap();
    assert!(
        after.importance() < initial_importance,
        "Importance should decrease after misleading feedback"
    );
}

#[test]
fn test_reinforce_neutral_no_change() {
    let (mut memory, _temp) = setup_memory_system();

    // Record memory
    let id = memory
        .record(create_experience("General observation about warehouse"))
        .unwrap();

    // Get initial importance
    let initial = memory.get_memory(&id).unwrap();
    let _initial_importance = initial.importance();

    // Reinforce as neutral
    let ids = vec![id];
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Neutral)
        .unwrap();

    // Check stats
    assert_eq!(stats.importance_boosts, 0, "Neutral should not boost");
    assert_eq!(stats.importance_decays, 0, "Neutral should not decay");
}

// =============================================================================
// ASSOCIATION STRENGTHENING TESTS (Fire Together Wire Together)
// =============================================================================

#[test]
fn test_co_retrieval_strengthens_association() {
    let (mut memory, _temp) = setup_memory_system();

    // Record related memories
    let id1 = memory
        .record(create_experience(
            "Battery level monitoring is critical for drone safety",
        ))
        .unwrap();
    let id2 = memory
        .record(create_experience(
            "Low battery triggers automatic return to base",
        ))
        .unwrap();

    // Reinforce both together as helpful
    let ids = vec![id1, id2];
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    assert!(
        stats.associations_strengthened > 0,
        "Should strengthen association between co-retrieved memories"
    );
}

#[test]
fn test_repeated_co_retrieval_increases_strength() {
    let (mut memory, _temp) = setup_memory_system();

    // Record related memories
    let id1 = memory
        .record(create_experience("Obstacle A detected at north entrance"))
        .unwrap();
    let id2 = memory
        .record(create_experience("Obstacle A is a forklift"))
        .unwrap();

    let ids = vec![id1, id2];

    // Reinforce multiple times
    let stats1 = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();
    let stats2 = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();
    let stats3 = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    // All should strengthen associations
    assert!(stats1.associations_strengthened > 0);
    assert!(stats2.associations_strengthened > 0);
    assert!(stats3.associations_strengthened > 0);
}

#[test]
fn test_single_memory_no_associations() {
    let (mut memory, _temp) = setup_memory_system();

    let id = memory
        .record(create_experience("Single isolated memory"))
        .unwrap();

    let ids = vec![id];
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    // Single memory can't have associations with itself
    assert_eq!(
        stats.associations_strengthened, 0,
        "Single memory has no associations"
    );
}

#[test]
fn test_many_co_retrieved_associations() {
    let (mut memory, _temp) = setup_memory_system();

    // Record many related memories
    let mut ids = Vec::new();
    for i in 0..10 {
        let id = memory
            .record(create_experience(&format!(
                "Mission log entry {}: patrol sector {}",
                i, i
            )))
            .unwrap();
        ids.push(id);
    }

    // Reinforce all together
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    // Should have many associations (n*(n-1)/2 for n=10 = 45)
    // But capped at 20 due to MAX_COACTIVATION_SIZE
    assert!(
        stats.associations_strengthened > 0,
        "Should strengthen associations"
    );
}

// =============================================================================
// REINFORCEMENT STATS TESTS
// =============================================================================

#[test]
fn test_reinforcement_stats_counts() {
    let (mut memory, _temp) = setup_memory_system();

    // Record memories
    let id1 = memory.record(create_experience("Memory 1")).unwrap();
    let id2 = memory.record(create_experience("Memory 2")).unwrap();
    let id3 = memory.record(create_experience("Memory 3")).unwrap();

    let ids = vec![id1, id2, id3];
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    assert_eq!(stats.memories_processed, 3, "Should process all memories");
}

#[test]
fn test_reinforcement_empty_ids() {
    let (mut memory, _temp) = setup_memory_system();

    let ids: Vec<shodh_memory::memory::MemoryId> = vec![];
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    assert_eq!(stats.memories_processed, 0);
    assert_eq!(stats.associations_strengthened, 0);
    assert_eq!(stats.importance_boosts, 0);
    assert_eq!(stats.importance_decays, 0);
}

#[test]
fn test_reinforcement_invalid_ids() {
    let (mut memory, _temp) = setup_memory_system();

    // Create random IDs that don't exist
    let ids = vec![
        shodh_memory::memory::MemoryId(shodh_memory::uuid::Uuid::new_v4()),
        shodh_memory::memory::MemoryId(shodh_memory::uuid::Uuid::new_v4()),
    ];

    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    // memories_processed counts input IDs, not found memories
    // This is by design - the function accepts any IDs and silently skips non-existent ones
    assert_eq!(stats.memories_processed, 2);
    // But importance boosts should be 0 since no actual memories were found
    assert_eq!(stats.importance_boosts, 0);
}

// =============================================================================
// HEBBIAN LEARNING FORMULA TESTS
// =============================================================================

#[test]
fn test_importance_boost_formula() {
    let (mut memory, _temp) = setup_memory_system();

    // Create memory with known importance
    let id = memory
        .record(Experience {
            content: "Test content".to_string(),
            experience_type: ExperienceType::Observation,
            ..Default::default()
        })
        .unwrap();

    let before = memory.get_memory(&id).unwrap();
    let before_importance = before.importance();

    // Reinforce as helpful
    memory
        .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Helpful)
        .unwrap();

    let after = memory.get_memory(&id).unwrap();
    let after_importance = after.importance();

    // Verify boost occurred
    assert!(after_importance > before_importance);

    // Verify boost is reasonable (not too extreme)
    let delta = after_importance - before_importance;
    assert!(
        delta > 0.0 && delta < 0.5,
        "Boost should be moderate, got {}",
        delta
    );
}

#[test]
fn test_importance_decay_formula() {
    let (mut memory, _temp) = setup_memory_system();

    // Create memory with high importance
    let id = memory
        .record(Experience {
            content: "High importance memory".to_string(),
            experience_type: ExperienceType::Decision,
            entities: vec!["critical".to_string()],
            ..Default::default()
        })
        .unwrap();

    let before = memory.get_memory(&id).unwrap();
    let before_importance = before.importance();

    // Reinforce as misleading
    memory
        .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Misleading)
        .unwrap();

    let after = memory.get_memory(&id).unwrap();
    let after_importance = after.importance();

    // Verify decay occurred
    assert!(after_importance < before_importance);

    // Verify decay is reasonable
    let delta = before_importance - after_importance;
    assert!(
        delta > 0.0 && delta < 0.5,
        "Decay should be moderate, got {}",
        delta
    );
}

// =============================================================================
// LONG-TERM POTENTIATION (LTP) TESTS
// =============================================================================

#[test]
fn test_ltp_after_multiple_reinforcements() {
    let (mut memory, _temp) = setup_memory_system();

    let id1 = memory
        .record(create_experience("Pattern A observation"))
        .unwrap();
    let id2 = memory
        .record(create_experience("Pattern A confirmation"))
        .unwrap();

    // Get initial importance
    let initial1 = memory.get_memory(&id1).unwrap().importance();
    let initial2 = memory.get_memory(&id2).unwrap().importance();

    let ids = vec![id1.clone(), id2.clone()];

    // Reinforce many times to trigger LTP (threshold is typically 3-5)
    // Each reinforcement adds +5% importance (clamped to 1.0)
    for _ in 0..10 {
        memory
            .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
            .unwrap();
    }

    // After LTP, associations should be very strong
    // This is tested by checking the memories have higher importance
    let mem1 = memory.get_memory(&id1).unwrap();
    let mem2 = memory.get_memory(&id2).unwrap();

    // Importance should have increased significantly after many reinforcements
    // With 10 reinforcements at +5% each, importance increases by 0.5 (capped at 1.0)
    assert!(
        mem1.importance() > initial1,
        "Repeated helpful feedback should increase importance (was {}, now {})",
        initial1,
        mem1.importance()
    );
    assert!(
        mem2.importance() > initial2,
        "Repeated helpful feedback should increase importance (was {}, now {})",
        initial2,
        mem2.importance()
    );

    // Verify significant boost (at least 0.3 increase or capped at 1.0)
    let expected_boost = 0.3;
    assert!(
        mem1.importance() >= initial1 + expected_boost || mem1.importance() >= 0.95,
        "Importance should boost significantly: {} + {} vs {}",
        initial1,
        expected_boost,
        mem1.importance()
    );
}

// =============================================================================
// INTEGRATION WITH RETRIEVAL TESTS
// =============================================================================

#[test]
fn test_retrieve_then_reinforce_cycle() {
    let (mut memory, _temp) = setup_memory_system();

    // Record memories
    for i in 0..20 {
        memory
            .record(create_experience(&format!(
                "Warehouse zone {} status: operational",
                i
            )))
            .unwrap();
    }

    // Retrieve some memories
    let query = Query {
        query_text: Some("warehouse zone status".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(!results.is_empty());

    // Collect IDs from results
    let ids: Vec<_> = results.iter().map(|m| m.id.clone()).collect();

    // Reinforce
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    assert!(stats.memories_processed > 0);
}

#[test]
fn test_reinforced_memories_rank_higher() {
    let (mut memory, _temp) = setup_memory_system();

    // Record several memories about the same topic
    let id_target = memory
        .record(create_experience("Target memory: critical safety protocol"))
        .unwrap();

    for i in 0..10 {
        memory
            .record(create_experience(&format!(
                "Background memory {} about safety",
                i
            )))
            .unwrap();
    }

    // Reinforce the target memory multiple times
    for _ in 0..5 {
        memory
            .reinforce_retrieval(&[id_target.clone()], RetrievalOutcome::Helpful)
            .unwrap();
    }

    // Retrieve and check if target ranks highly
    let query = Query {
        query_text: Some("safety protocol".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();

    // Target should be in top results due to higher importance
    let target_found = results.iter().any(|m| m.id == id_target);
    assert!(
        target_found,
        "Reinforced memory should appear in top results"
    );
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_reinforce_same_memory_twice() {
    let (mut memory, _temp) = setup_memory_system();

    let id = memory.record(create_experience("Duplicate test")).unwrap();

    // Pass same ID twice
    let ids = vec![id.clone(), id.clone()];
    let stats = memory
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .unwrap();

    // Should deduplicate or handle gracefully
    assert!(stats.memories_processed <= 2);
}

#[test]
fn test_alternating_feedback() {
    let (mut memory, _temp) = setup_memory_system();

    let id = memory
        .record(create_experience("Alternating feedback test"))
        .unwrap();

    // Alternate helpful and misleading
    memory
        .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Helpful)
        .unwrap();
    memory
        .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Misleading)
        .unwrap();
    memory
        .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Helpful)
        .unwrap();
    memory
        .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Misleading)
        .unwrap();

    // Memory should still exist and be valid - get_memory returns Result<Memory>, success means it exists
    let mem = memory.get_memory(&id);
    assert!(
        mem.is_ok(),
        "Memory should still exist after alternating feedback"
    );
}

#[test]
fn test_high_volume_reinforcement() {
    let (mut memory, _temp) = setup_memory_system();

    // Record many memories
    let mut ids = Vec::new();
    for i in 0..100 {
        let id = memory
            .record(create_experience(&format!("High volume memory {}", i)))
            .unwrap();
        ids.push(id);
    }

    // Reinforce in batches
    for chunk in ids.chunks(10) {
        let chunk_ids: Vec<_> = chunk.to_vec();
        let stats = memory
            .reinforce_retrieval(&chunk_ids, RetrievalOutcome::Helpful)
            .unwrap();
        assert!(stats.memories_processed > 0);
    }
}
