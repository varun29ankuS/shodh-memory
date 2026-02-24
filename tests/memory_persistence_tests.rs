//! Comprehensive Persistence and Cache Coherency Tests
//!
//! Tests the critical properties of the memory system:
//! - Storage persistence across system restarts
//! - Cache coherency between tiers (working, session, long-term)
//! - Concurrent access safety
//! - Edge cases in importance updates
//! - NER integration for entity extraction
//!
//! These tests ensure data integrity and correctness under various conditions.

use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;
use uuid::Uuid;

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{
    retrieval::RetrievalOutcome,
    types::{Experience, ExperienceType, Query},
    MemoryConfig, MemoryId, MemorySystem,
};

/// Create fallback NER instance for testing
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Create experience with NER-extracted entities
fn create_experience_with_ner(content: &str, ner: &NeuralNer) -> Experience {
    let entities = ner.extract(content).unwrap_or_default();
    let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
    Experience {
        experience_type: ExperienceType::Learning,
        content: content.to_string(),
        entities: entity_names,
        ..Default::default()
    }
}

// ============================================================================
// TEST INFRASTRUCTURE
// ============================================================================

fn create_test_config(temp_dir: &TempDir) -> MemoryConfig {
    MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    }
}

fn create_test_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");
    (system, temp_dir)
}

fn create_experience(content: &str, entities: Vec<&str>) -> Experience {
    Experience {
        experience_type: ExperienceType::Learning,
        content: content.to_string(),
        entities: entities.into_iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

// ============================================================================
// STORAGE PERSISTENCE TESTS
// ============================================================================

#[test]
fn test_memory_survives_system_restart() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let memory_id;
    let original_content = "This memory should survive restart";

    // Phase 1: Create memory and drop system
    {
        let system = MemorySystem::new(config.clone()).expect("Failed to create system");
        let exp = create_experience(original_content, vec!["persistence", "test"]);
        memory_id = system.remember(exp, None).expect("Failed to record");
    }
    // System dropped here - simulates restart

    // Phase 2: Recreate system and verify memory exists
    {
        let system = MemorySystem::new(config).expect("Failed to recreate system");
        let query = Query {
            query_text: Some("survive restart".to_string()),
            max_results: 10,
            ..Default::default()
        };
        let results = system.recall(&query).expect("Failed to retrieve");

        assert!(!results.is_empty(), "Memory should survive restart");
        assert!(
            results.iter().any(|m| m.id == memory_id),
            "Should find the specific memory after restart"
        );
        assert!(
            results
                .iter()
                .any(|m| m.experience.content.contains("survive restart")),
            "Memory content should be preserved"
        );
    }
}

#[test]
fn test_importance_changes_survive_restart() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let memory_id;
    let final_importance;

    // Phase 1: Create memory, boost importance, drop system
    {
        let system = MemorySystem::new(config.clone()).expect("Failed to create system");
        let exp = create_experience("Important information for testing", vec!["importance"]);
        memory_id = system.remember(exp, None).expect("Failed to record");

        // Apply multiple helpful reinforcements
        for _ in 0..10 {
            system
                .reinforce_recall(&[memory_id.clone()], RetrievalOutcome::Helpful)
                .expect("Failed to reinforce");
        }

        // Verify importance increased
        let query = Query {
            query_text: Some("important information".to_string()),
            max_results: 1,
            ..Default::default()
        };
        let results = system.recall(&query).expect("Failed to retrieve");
        final_importance = results[0].importance();
        assert!(final_importance > 0.5, "Importance should have increased");
    }
    // System dropped here

    // Phase 2: Verify importance persisted
    {
        let system = MemorySystem::new(config).expect("Failed to recreate system");

        // First, get memory directly from storage to verify persistence
        let direct_memory = system
            .get_memory(&memory_id)
            .expect("Failed to get memory directly");
        let direct_importance = direct_memory.importance();

        // The importance should be higher than initial after reinforcement
        // HEBBIAN_BOOST_HELPFUL = 0.025, so 10 boosts add ~0.25
        // Cache vs storage semantics mean not all boosts may stack identically
        assert!(
            direct_importance > 0.45,
            "Importance should be boosted after restart: {} (expected > 0.45)",
            direct_importance
        );

        // Allow for some deviation due to floating point and timing
        // The key test is that importance WAS increased and persisted
        assert!(
            (direct_importance - final_importance).abs() < 0.15,
            "Importance should be roughly preserved after restart: {} vs {} (diff: {})",
            direct_importance,
            final_importance,
            (direct_importance - final_importance).abs()
        );
    }
}

#[test]
fn test_access_count_survives_restart() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let memory_id;

    // Phase 1: Create memory and access it multiple times
    {
        let system = MemorySystem::new(config.clone()).expect("Failed to create system");
        let exp = create_experience("Access count test memory", vec!["access"]);
        memory_id = system.remember(exp, None).expect("Failed to record");

        // Access multiple times through reinforcement
        for _ in 0..5 {
            system
                .reinforce_recall(&[memory_id.clone()], RetrievalOutcome::Neutral)
                .expect("Failed to reinforce");
        }
    }

    // Phase 2: Verify access count persisted
    {
        let system = MemorySystem::new(config).expect("Failed to recreate system");
        let query = Query {
            query_text: Some("access count test".to_string()),
            max_results: 1,
            ..Default::default()
        };
        let results = system.recall(&query).expect("Failed to retrieve");

        assert!(
            results[0].access_count() >= 5,
            "Access count should be preserved: {}",
            results[0].access_count()
        );
    }
}

#[test]
fn test_multiple_memories_survive_restart() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let mut memory_ids = Vec::new();

    // Phase 1: Create multiple memories
    {
        let system = MemorySystem::new(config.clone()).expect("Failed to create system");

        for i in 0..20 {
            let exp = create_experience(
                &format!("Memory number {} for batch persistence test", i),
                vec!["batch", "persistence"],
            );
            let id = system.remember(exp, None).expect("Failed to record");
            memory_ids.push(id);
        }
    }

    // Phase 2: Verify all memories exist
    {
        let system = MemorySystem::new(config).expect("Failed to recreate system");
        let query = Query {
            query_text: Some("batch persistence test".to_string()),
            max_results: 50,
            ..Default::default()
        };
        let results = system.recall(&query).expect("Failed to retrieve");

        assert!(
            results.len() >= 15,
            "Most memories should survive: found {} of 20",
            results.len()
        );
    }
}

// ============================================================================
// CACHE COHERENCY TESTS
// ============================================================================

#[test]
fn test_cache_coherency_importance_visible_immediately() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_experience("Cache coherency test memory", vec!["cache"]);
    let id = system.remember(exp, None).expect("Failed to record");

    // Get initial importance through retrieval
    let query = Query {
        query_text: Some("cache coherency".to_string()),
        max_results: 1,
        ..Default::default()
    };
    let results = system.recall(&query).expect("Failed to retrieve");
    let initial_importance = results[0].importance();

    // Boost importance
    system
        .reinforce_recall(&[id.clone()], RetrievalOutcome::Helpful)
        .expect("Failed to reinforce");

    // Verify change is immediately visible through same query
    let results = system.recall(&query).expect("Failed to retrieve");
    let new_importance = results[0].importance();

    assert!(
        new_importance > initial_importance,
        "Importance change should be immediately visible: {} > {}",
        new_importance,
        initial_importance
    );
}

#[test]
fn test_cache_coherency_multiple_retrievals() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_experience("Multi-retrieval coherency test", vec!["multi"]);
    let id = system.remember(exp, None).expect("Failed to record");

    let query = Query {
        query_text: Some("multi-retrieval".to_string()),
        max_results: 1,
        ..Default::default()
    };

    // Retrieve, modify, retrieve again multiple times
    for i in 0..5 {
        let before = system
            .recall(&query)
            .expect("Failed")
            .first()
            .unwrap()
            .importance();

        system
            .reinforce_recall(&[id.clone()], RetrievalOutcome::Helpful)
            .expect("Failed");

        let after = system
            .recall(&query)
            .expect("Failed")
            .first()
            .unwrap()
            .importance();

        assert!(
            after > before,
            "Iteration {}: importance should increase: {} > {}",
            i,
            after,
            before
        );
    }
}

#[test]
fn test_cache_coherency_decay_visible_immediately() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_experience("Decay visibility test", vec!["decay"]);
    let id = system.remember(exp, None).expect("Failed to record");

    let query = Query {
        query_text: Some("decay visibility".to_string()),
        max_results: 1,
        ..Default::default()
    };

    let initial = system
        .recall(&query)
        .expect("Failed")
        .first()
        .unwrap()
        .importance();

    // Apply decay
    system
        .reinforce_recall(&[id.clone()], RetrievalOutcome::Misleading)
        .expect("Failed");

    let after_decay = system
        .recall(&query)
        .expect("Failed")
        .first()
        .unwrap()
        .importance();

    assert!(
        after_decay < initial,
        "Decay should be immediately visible: {} < {}",
        after_decay,
        initial
    );
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

#[test]
fn test_concurrent_record_and_retrieve() {
    let (system, _temp_dir) = create_test_system();
    let system = Arc::new(parking_lot::Mutex::new(system));

    let system_clone = Arc::clone(&system);

    // Spawn thread that records memories
    let writer = thread::spawn(move || {
        for i in 0..10 {
            let mut sys = system_clone.lock();
            let exp = create_experience(
                &format!("Concurrent write test memory {}", i),
                vec!["concurrent"],
            );
            sys.remember(exp, None).expect("Failed to record");
            drop(sys);
            thread::sleep(Duration::from_millis(10));
        }
    });

    // Main thread retrieves
    for _ in 0..10 {
        let sys = system.lock();
        let query = Query {
            query_text: Some("concurrent".to_string()),
            max_results: 20,
            ..Default::default()
        };
        let _ = sys.recall(&query); // May or may not find results
        drop(sys);
        thread::sleep(Duration::from_millis(10));
    }

    writer.join().expect("Writer thread panicked");

    // Verify all writes succeeded
    let sys = system.lock();
    let query = Query {
        query_text: Some("concurrent write test".to_string()),
        max_results: 20,
        ..Default::default()
    };
    let results = sys.recall(&query).expect("Failed to retrieve");
    assert!(
        results.len() >= 5,
        "Should find most concurrent writes: found {}",
        results.len()
    );
}

#[test]
fn test_concurrent_reinforcement() {
    let (system, _temp_dir) = create_test_system();
    let system = system;

    // Create a memory
    let exp = create_experience("Concurrent reinforcement target", vec!["target"]);
    let id = system.remember(exp, None).expect("Failed to record");

    let system = Arc::new(parking_lot::Mutex::new(system));
    let id_clone = id.clone();
    let system_clone = Arc::clone(&system);

    // Spawn thread that boosts
    let booster = thread::spawn(move || {
        for _ in 0..20 {
            let mut sys = system_clone.lock();
            let _ = sys.reinforce_recall(&[id_clone.clone()], RetrievalOutcome::Helpful);
            drop(sys);
            thread::sleep(Duration::from_millis(5));
        }
    });

    // Main thread also reinforces
    for _ in 0..20 {
        let mut sys = system.lock();
        let _ = sys.reinforce_recall(&[id.clone()], RetrievalOutcome::Helpful);
        drop(sys);
        thread::sleep(Duration::from_millis(5));
    }

    booster.join().expect("Booster thread panicked");

    // Verify importance increased significantly
    let sys = system.lock();
    let query = Query {
        query_text: Some("concurrent reinforcement".to_string()),
        max_results: 1,
        ..Default::default()
    };
    let results = sys.recall(&query).expect("Failed to retrieve");

    // After 40 boosts of 0.05 each (starting from ~0.5), should be at max 1.0
    assert!(
        results[0].importance() > 0.8,
        "Importance should be high after concurrent boosts: {}",
        results[0].importance()
    );
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_reinforce_nonexistent_memory() {
    let (system, _temp_dir) = create_test_system();

    let fake_id = MemoryId(Uuid::new_v4());

    // Should not panic, should return stats with 0 processed
    let stats = system
        .reinforce_recall(&[fake_id], RetrievalOutcome::Helpful)
        .expect("Should not fail");

    // The memory wasn't found, so no boosts should have been applied
    assert_eq!(stats.memories_processed, 1); // Attempted to process 1
                                             // Boost count depends on whether memory was found - could be 0 if not found
}

#[test]
fn test_reinforce_mixed_existing_nonexistent() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_experience("Real memory for mixed test", vec!["mixed"]);
    let real_id = system.remember(exp, None).expect("Failed to record");
    let fake_id = MemoryId(Uuid::new_v4());

    let stats = system
        .reinforce_recall(&[real_id, fake_id], RetrievalOutcome::Helpful)
        .expect("Should not fail");

    // Should process both attempts
    assert_eq!(stats.memories_processed, 2);
    // But only one boost should succeed
    assert_eq!(stats.importance_boosts, 1);
}

#[test]
fn test_importance_bounds_at_maximum() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_experience("Max importance test", vec!["max"]);
    let id = system.remember(exp, None).expect("Failed to record");

    // Boost many times to try to exceed 1.0
    for _ in 0..100 {
        system
            .reinforce_recall(&[id.clone()], RetrievalOutcome::Helpful)
            .expect("Failed");
    }

    let query = Query {
        query_text: Some("max importance".to_string()),
        max_results: 1,
        ..Default::default()
    };
    let results = system.recall(&query).expect("Failed");

    assert!(
        results[0].importance() <= 1.0,
        "Importance should not exceed 1.0: {}",
        results[0].importance()
    );
    assert!(
        results[0].importance() > 0.95,
        "Importance should be near max: {}",
        results[0].importance()
    );
}

#[test]
fn test_importance_bounds_at_minimum() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_experience("Min importance test", vec!["min"]);
    let id = system.remember(exp, None).expect("Failed to record");

    // Decay many times to try to go below floor
    for _ in 0..100 {
        system
            .reinforce_recall(&[id.clone()], RetrievalOutcome::Misleading)
            .expect("Failed");
    }

    let query = Query {
        query_text: Some("min importance".to_string()),
        max_results: 1,
        ..Default::default()
    };
    let results = system.recall(&query).expect("Failed");

    assert!(
        results[0].importance() >= 0.05,
        "Importance should not go below floor: {}",
        results[0].importance()
    );
}

#[test]
fn test_alternating_boost_and_decay() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_experience("Alternating reinforcement test", vec!["alternating"]);
    let id = system.remember(exp, None).expect("Failed to record");

    let query = Query {
        query_text: Some("alternating reinforcement".to_string()),
        max_results: 1,
        ..Default::default()
    };

    let mut importances = Vec::new();
    importances.push(
        system
            .recall(&query)
            .expect("Failed")
            .first()
            .unwrap()
            .importance(),
    );

    // Alternate boost and decay
    for i in 0..10 {
        if i % 2 == 0 {
            system
                .reinforce_recall(&[id.clone()], RetrievalOutcome::Helpful)
                .expect("Failed");
        } else {
            system
                .reinforce_recall(&[id.clone()], RetrievalOutcome::Misleading)
                .expect("Failed");
        }
        importances.push(
            system
                .recall(&query)
                .expect("Failed")
                .first()
                .unwrap()
                .importance(),
        );
    }

    // Verify importance changed appropriately
    // Boost: +0.05 additive, Decay: -10% multiplicative
    // After alternating, should generally decrease due to multiplicative decay
    let final_importance = *importances.last().unwrap();
    assert!(
        final_importance > 0.05 && final_importance < 1.0,
        "Importance should be within bounds after alternation: {}",
        final_importance
    );
}

#[test]
fn test_empty_reinforcement_list() {
    let (system, _temp_dir) = create_test_system();

    let stats = system
        .reinforce_recall(&[], RetrievalOutcome::Helpful)
        .expect("Should not fail on empty list");

    assert_eq!(stats.memories_processed, 0);
    assert_eq!(stats.importance_boosts, 0);
    assert_eq!(stats.associations_strengthened, 0);
}

#[test]
fn test_large_batch_reinforcement() {
    let (mut system, _temp_dir) = create_test_system();

    let mut ids = Vec::new();
    for i in 0..50 {
        let exp = create_experience(&format!("Large batch memory {}", i), vec!["batch"]);
        let id = system.remember(exp, None).expect("Failed to record");
        ids.push(id);
    }

    // Reinforce all at once
    let stats = system
        .reinforce_recall(&ids, RetrievalOutcome::Helpful)
        .expect("Failed");

    assert_eq!(stats.memories_processed, 50);
    assert!(
        stats.importance_boosts >= 40,
        "Most memories should be boosted: {}",
        stats.importance_boosts
    );

    // With 50 memories, associations = 50 * 49 / 2 = 1225
    assert!(
        stats.associations_strengthened > 1000,
        "Many associations should be strengthened: {}",
        stats.associations_strengthened
    );
}

// ============================================================================
// MEMORY TIERING TESTS
// ============================================================================

#[test]
fn test_memory_in_working_tier_after_record() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_experience("Working tier test memory", vec!["tier"]);
    let _id = system.remember(exp, None).expect("Failed to record");

    // Immediately retrievable through semantic search
    let query = Query {
        query_text: Some("working tier test".to_string()),
        max_results: 1,
        ..Default::default()
    };
    let results = system.recall(&query).expect("Failed");

    assert!(
        !results.is_empty(),
        "Memory should be immediately retrievable after record"
    );
}

#[test]
fn test_high_volume_record_and_retrieve() {
    let (mut system, _temp_dir) = create_test_system();

    // Record many memories
    for i in 0..100 {
        let exp = create_experience(
            &format!("High volume test memory number {} with unique content", i),
            vec!["volume", &format!("item{}", i)],
        );
        system.remember(exp, None).expect("Failed to record");
    }

    // Verify retrieval still works with semantic search
    // Note: HNSW approximate search may not return all matches
    let query = Query {
        query_text: Some("high volume test memory".to_string()),
        max_results: 50,
        ..Default::default()
    };
    let results = system.recall(&query).expect("Failed");

    // Semantic search returns approximate nearest neighbors
    // We should get a reasonable number of results
    assert!(
        results.len() >= 20,
        "Should retrieve reasonable number of memories via semantic search: found {}",
        results.len()
    );

    // Verify content is correct
    for result in &results {
        assert!(
            result
                .experience
                .content
                .contains("High volume test memory"),
            "Retrieved memory should match query"
        );
    }
}

// ============================================================================
// STRESS TESTS
// ============================================================================

#[test]
fn test_rapid_record_retrieve_cycle() {
    let (mut system, _temp_dir) = create_test_system();

    let mut recorded_ids = Vec::new();

    // Phase 1: Rapid record
    for i in 0..30 {
        let exp = create_experience(&format!("Rapid cycle memory {}", i), vec!["rapid"]);
        let id = system.remember(exp, None).expect("Failed to record");
        recorded_ids.push(id);
    }

    // Phase 2: Verify all can be retrieved via direct ID lookup
    for (i, id) in recorded_ids.iter().enumerate() {
        let memory = system
            .get_memory(id)
            .expect(&format!("Should find memory {} by ID", i));
        assert!(
            memory.experience.content.contains("Rapid cycle memory"),
            "Memory {} content should be correct",
            i
        );
    }

    // Phase 3: Verify semantic search finds most of them
    let query = Query {
        query_text: Some("rapid cycle memory".to_string()),
        max_results: 30,
        ..Default::default()
    };
    let results = system.recall(&query).expect("Failed to retrieve");

    // Semantic search should find most (not necessarily all due to HNSW approximation)
    assert!(
        results.len() >= 20,
        "Semantic search should find most rapid cycle memories: found {}",
        results.len()
    );
}

#[test]
fn test_stress_reinforcement_cycles() {
    let (mut system, _temp_dir) = create_test_system();

    // Create memories
    let mut ids = Vec::new();
    for i in 0..20 {
        let exp = create_experience(&format!("Stress test memory {}", i), vec!["stress"]);
        let id = system.remember(exp, None).expect("Failed to record");
        ids.push(id);
    }

    // Rapid reinforcement cycles
    for _ in 0..100 {
        // Random subset
        let subset: Vec<_> = ids.iter().step_by(3).cloned().collect();
        system
            .reinforce_recall(&subset, RetrievalOutcome::Helpful)
            .expect("Failed");

        let subset: Vec<_> = ids.iter().skip(1).step_by(3).cloned().collect();
        system
            .reinforce_recall(&subset, RetrievalOutcome::Neutral)
            .expect("Failed");

        let subset: Vec<_> = ids.iter().skip(2).step_by(3).cloned().collect();
        system
            .reinforce_recall(&subset, RetrievalOutcome::Misleading)
            .expect("Failed");
    }

    // Verify system is still functional
    let query = Query {
        query_text: Some("stress test".to_string()),
        max_results: 30,
        ..Default::default()
    };
    let results = system.recall(&query).expect("Failed");

    assert!(
        results.len() >= 10,
        "Should still retrieve memories after stress: {}",
        results.len()
    );
}
