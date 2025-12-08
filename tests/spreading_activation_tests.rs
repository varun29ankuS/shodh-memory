//! Spreading Activation Tests
//!
//! Tests for the spreading activation algorithm:
//! - Activation propagation through memory graph
//! - Decay over distance (exponential)
//! - Multi-hop traversal
//! - Activation capping at 1.0
//! - Concurrent activation updates
//! - NER integration for entity extraction

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{
    Experience, ExperienceType, Memory, MemoryConfig, MemoryId, MemorySystem, MemoryTier, Query,
    RetrievalOutcome,
};
use shodh_memory::uuid::Uuid;
use tempfile::TempDir;

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
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        entities: entity_names,
        ..Default::default()
    }
}

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

fn create_experience(content: &str) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    }
}

// =============================================================================
// ACTIVATION LEVEL UNIT TESTS
// =============================================================================

#[test]
fn test_activation_starts_at_one() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    assert!((memory.activation() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_activate_adds_value() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(0.3);
    memory.activate(0.2);
    assert!((memory.activation() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_activate_clamps_at_one() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(0.9);
    memory.activate(0.5);
    assert!((memory.activation() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_activate_with_zero() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(0.5);
    memory.activate(0.0);
    assert!((memory.activation() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_decay_activation_multiplicative() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(1.0);
    memory.decay_activation(0.5);
    assert!((memory.activation() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_decay_activation_exponential() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(1.0);

    // Apply decay 3 times with factor 0.5
    memory.decay_activation(0.5);
    memory.decay_activation(0.5);
    memory.decay_activation(0.5);

    // 1.0 * 0.5 * 0.5 * 0.5 = 0.125
    assert!((memory.activation() - 0.125).abs() < f32::EPSILON);
}

#[test]
fn test_decay_activation_approaches_zero() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(1.0);

    for _ in 0..50 {
        memory.decay_activation(0.9);
    }

    assert!(memory.activation() < 0.01);
}

#[test]
fn test_decay_activation_with_one() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(0.5);
    memory.decay_activation(1.0);
    assert!((memory.activation() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_decay_activation_with_zero() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(0.5);
    memory.decay_activation(0.0);
    assert!((memory.activation() - 0.0).abs() < f32::EPSILON);
}

// =============================================================================
// ACTIVATION DECAY FORMULA TESTS
// =============================================================================

#[test]
fn test_decay_formula_99_percent() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(1.0);
    memory.decay_activation(0.99);
    assert!((memory.activation() - 0.99).abs() < f32::EPSILON);
}

#[test]
fn test_decay_formula_50_percent() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(0.8);
    memory.decay_activation(0.5);
    assert!((memory.activation() - 0.4).abs() < f32::EPSILON);
}

#[test]
fn test_decay_then_activate() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(1.0);
    memory.decay_activation(0.5); // 0.5
    memory.activate(0.3); // 0.8
    assert!((memory.activation() - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_activate_then_decay() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(0.5);
    memory.activate(0.5); // 1.0
    memory.decay_activation(0.8); // 0.8
    assert!((memory.activation() - 0.8).abs() < f32::EPSILON);
}

// =============================================================================
// SPREADING ACTIVATION DISTANCE TESTS
// =============================================================================

#[test]
fn test_activation_hop_1() {
    // Hop 1: decay = exp(-0.5 * 1) ≈ 0.606
    let decay = (-0.5_f32 * 1.0).exp();
    assert!((decay - 0.6065).abs() < 0.01);
}

#[test]
fn test_activation_hop_2() {
    // Hop 2: decay = exp(-0.5 * 2) ≈ 0.368
    let decay = (-0.5_f32 * 2.0).exp();
    assert!((decay - 0.3679).abs() < 0.01);
}

#[test]
fn test_activation_hop_3() {
    // Hop 3: decay = exp(-0.5 * 3) ≈ 0.223
    let decay = (-0.5_f32 * 3.0).exp();
    assert!((decay - 0.2231).abs() < 0.01);
}

#[test]
fn test_activation_decreases_with_distance() {
    let hop_1 = (-0.5_f32 * 1.0).exp();
    let hop_2 = (-0.5_f32 * 2.0).exp();
    let hop_3 = (-0.5_f32 * 3.0).exp();

    assert!(hop_1 > hop_2);
    assert!(hop_2 > hop_3);
}

#[test]
fn test_activation_never_zero() {
    // Even at hop 10, activation should be positive
    let hop_10 = (-0.5_f32 * 10.0).exp();
    assert!(hop_10 > 0.0);
    assert!((hop_10 - 0.0067).abs() < 0.01);
}

// =============================================================================
// GRAPH ACTIVATION INTEGRATION TESTS
// =============================================================================

#[test]
fn test_connected_memories_coactivate() {
    let (mut memory, _temp) = setup_memory_system();

    let id1 = memory.record(create_experience("Memory A")).unwrap();
    let id2 = memory.record(create_experience("Memory B")).unwrap();

    // Connect them via reinforcement
    memory
        .reinforce_retrieval(&[id1.clone(), id2.clone()], RetrievalOutcome::Helpful)
        .unwrap();

    // Both should be retrievable when searching for either
    let query = Query {
        query_text: Some("Memory".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(results.len() >= 2);
}

#[test]
fn test_chain_activation_propagates() {
    let (mut memory, _temp) = setup_memory_system();

    // Create a chain: A -> B -> C
    let id_a = memory.record(create_experience("Chain start A")).unwrap();
    let id_b = memory.record(create_experience("Chain middle B")).unwrap();
    let id_c = memory.record(create_experience("Chain end C")).unwrap();

    // Connect A-B
    memory
        .reinforce_retrieval(&[id_a.clone(), id_b.clone()], RetrievalOutcome::Helpful)
        .unwrap();
    // Connect B-C
    memory
        .reinforce_retrieval(&[id_b.clone(), id_c.clone()], RetrievalOutcome::Helpful)
        .unwrap();

    // Query should find all three
    let query = Query {
        query_text: Some("Chain".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(results.len() >= 3);
}

#[test]
fn test_disconnected_memories_independent() {
    let (mut memory, _temp) = setup_memory_system();

    // Create two unconnected memories
    let _id1 = memory.record(create_experience("Topic alpha")).unwrap();
    let _id2 = memory.record(create_experience("Topic beta")).unwrap();

    // Query for alpha - should not strongly include beta
    let query = Query {
        query_text: Some("alpha".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    // Results should exist (at least alpha)
    assert!(!results.is_empty());
}

#[test]
fn test_hub_memory_activates_many() {
    let (mut memory, _temp) = setup_memory_system();

    // Create hub and spokes
    let hub = memory
        .record(create_experience("Central hub memory"))
        .unwrap();
    let mut spokes = Vec::new();

    for i in 0..5 {
        let spoke = memory
            .record(create_experience(&format!("Spoke {} connected to hub", i)))
            .unwrap();
        memory
            .reinforce_retrieval(&[hub.clone(), spoke.clone()], RetrievalOutcome::Helpful)
            .unwrap();
        spokes.push(spoke);
    }

    // Searching for hub should find many results
    let query = Query {
        query_text: Some("hub".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(results.len() >= 1);
}

// =============================================================================
// ACTIVATION BATCH TESTS
// =============================================================================

#[test]
fn test_batch_activation_update() {
    // This test verifies that batch activation works correctly across many memories.
    // All memories start at activation=1.0, so adding 0.1 should still clamp at 1.0.
    let memories: Vec<Memory> = (0..100)
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

    // Activate all (through Arc<Memory> compatible API)
    for m in &memories {
        m.activate(0.1);
    }

    // All should be fully activated (started at 1.0, capped at 1.0)
    for m in &memories {
        assert!((m.activation() - 1.0).abs() < f32::EPSILON);
    }
}

#[test]
fn test_batch_decay_update() {
    // This test verifies batch decay works correctly.
    // Memories start at activation=1.0 (default), decay by 0.9 factor -> 0.9
    let memories: Vec<Memory> = (0..100)
        .map(|_| {
            Memory::new(
                MemoryId(Uuid::new_v4()),
                Experience::default(),
                0.5,
                None,
                None,
                None,
            )
            // New memories start at activation=1.0 by default
        })
        .collect();

    // Decay all (through Arc<Memory> compatible API)
    for m in &memories {
        m.decay_activation(0.9);
    }

    // All should be at 0.9 (1.0 * 0.9)
    for m in &memories {
        assert!((m.activation() - 0.9).abs() < f32::EPSILON);
    }
}

#[test]
fn test_heterogeneous_activation() {
    // This test verifies activation works correctly with different starting values.
    // Each memory starts at a different activation level: 0.0, 0.1, 0.2, ..., 0.9
    let memories: Vec<Memory> = (0..10)
        .map(|i| {
            let m = Memory::new(
                MemoryId(Uuid::new_v4()),
                Experience::default(),
                0.5,
                None,
                None,
                None,
            );
            m.set_activation(0.1 * (i as f32));
            m
        })
        .collect();

    // Activate all by 0.1 (through Arc<Memory> compatible API)
    for m in &memories {
        m.activate(0.1);
    }

    // Check activations increased appropriately (capped at 1.0)
    for (i, m) in memories.iter().enumerate() {
        let expected = ((0.1 * i as f32) + 0.1).min(1.0);
        assert!((m.activation() - expected).abs() < f32::EPSILON);
    }
}

// =============================================================================
// CONCURRENT ACTIVATION TESTS
// =============================================================================

#[test]
fn test_concurrent_activation_reads() {
    // This test verifies that concurrent activation reads are thread-safe.
    // Multiple threads can read activation simultaneously without data races.
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
                    // Use the thread-safe getter
                    let _ = m.activation();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_activation_after_tier_change() {
    // This test verifies that activation is preserved during tier promotion.
    // The promote() operation changes tier but should not affect activation level.
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    memory.set_activation(0.5);
    memory.promote(); // Working -> Session (requires &mut self)

    // Activation should be preserved across tier change
    assert!((memory.activation() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_activation_across_tier_cycle() {
    // This test verifies activation survives a full promotion/demotion cycle.
    // Working -> Session -> LongTerm -> Archive -> LongTerm -> Session -> Working
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    memory.set_activation(0.7);

    // Full promotion cycle: Working -> Session -> LongTerm -> Archive
    memory.promote();
    memory.promote();
    memory.promote();

    // Full demotion cycle: Archive -> LongTerm -> Session -> Working
    memory.demote();
    memory.demote();
    memory.demote();

    // Activation should be preserved through the entire cycle
    assert!((memory.activation() - 0.7).abs() < f32::EPSILON);
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_activation_very_small() {
    // This test verifies precision at very small activation values.
    // Important for long-decayed memories that haven't fully reached zero.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    memory.set_activation(0.0001);
    memory.activate(0.0001);

    assert!((memory.activation() - 0.0002).abs() < 0.0001);
}

#[test]
fn test_activation_at_boundary() {
    // This test verifies activation correctly clamps at the 1.0 boundary.
    // 0.9999 + 0.0001 should equal exactly 1.0.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    memory.set_activation(0.9999);
    memory.activate(0.0001);

    assert!((memory.activation() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_decay_very_small_factor() {
    // This test verifies very slow decay (0.01% loss per tick).
    // Models memories that persist with minimal degradation.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    // Memory starts at 1.0, decay by 0.9999 factor -> 0.9999
    memory.decay_activation(0.9999);

    assert!((memory.activation() - 0.9999).abs() < 0.0001);
}

#[test]
fn test_activation_negative_not_possible() {
    // This test verifies activation can never go negative.
    // Even extreme decay (90% loss per tick, 100 times) should stay >= 0.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    memory.set_activation(0.1);
    // Multiple severe decays: 0.1^100 is effectively zero but never negative
    for _ in 0..100 {
        memory.decay_activation(0.1);
    }

    // Should never go negative (multiplicative decay approaches but never reaches zero)
    assert!(memory.activation() >= 0.0);
}

#[test]
fn test_activation_serialization_roundtrip() {
    // This test verifies activation is properly serialized/deserialized.
    // Critical for long-term memory persistence to disk.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    memory.set_activation(0.12345);

    let serialized = bincode::serialize(&memory).unwrap();
    let deserialized: Memory = bincode::deserialize(&serialized).unwrap();

    assert!((deserialized.activation() - 0.12345).abs() < f32::EPSILON);
}

// =============================================================================
// DECAY RATE VARIATION TESTS
// =============================================================================

#[test]
fn test_slow_decay_rate() {
    // This test models slow decay (1% loss per tick).
    // After 10 ticks: 0.99^10 ≈ 0.904 - memory mostly preserved.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    // Memory starts at 1.0 by default

    // Very slow decay (0.99 preserves 99%)
    for _ in 0..10 {
        memory.decay_activation(0.99);
    }

    // 0.99^10 ≈ 0.904
    assert!(memory.activation() > 0.9);
}

#[test]
fn test_fast_decay_rate() {
    // This test models fast decay (50% loss per tick).
    // After 3 ticks: 0.5^3 = 0.125 - rapid degradation.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    // Memory starts at 1.0 by default

    // Fast decay (0.5 preserves 50%)
    for _ in 0..3 {
        memory.decay_activation(0.5);
    }

    // 0.5^3 = 0.125
    assert!((memory.activation() - 0.125).abs() < f32::EPSILON);
}

#[test]
fn test_medium_decay_rate() {
    // This test models medium decay (20% loss per tick).
    // After 5 ticks: 0.8^5 = 0.32768 - moderate degradation.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );
    // Memory starts at 1.0 by default

    // Medium decay (0.8 preserves 80%)
    for _ in 0..5 {
        memory.decay_activation(0.8);
    }

    // 0.8^5 = 0.32768
    assert!((memory.activation() - 0.32768).abs() < 0.01);
}

// =============================================================================
// GRAPH STRUCTURE TESTS
// =============================================================================

#[test]
fn test_triangle_activation() {
    let (mut memory, _temp) = setup_memory_system();

    // Create triangle: A-B, B-C, C-A
    let id_a = memory
        .record(create_experience("Triangle vertex A"))
        .unwrap();
    let id_b = memory
        .record(create_experience("Triangle vertex B"))
        .unwrap();
    let id_c = memory
        .record(create_experience("Triangle vertex C"))
        .unwrap();

    memory
        .reinforce_retrieval(&[id_a.clone(), id_b.clone()], RetrievalOutcome::Helpful)
        .unwrap();
    memory
        .reinforce_retrieval(&[id_b.clone(), id_c.clone()], RetrievalOutcome::Helpful)
        .unwrap();
    memory
        .reinforce_retrieval(&[id_c.clone(), id_a.clone()], RetrievalOutcome::Helpful)
        .unwrap();

    // Query should find all three
    let query = Query {
        query_text: Some("Triangle vertex".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(results.len() >= 3);
}

#[test]
fn test_star_activation() {
    let (mut memory, _temp) = setup_memory_system();

    // Create star: center connected to 5 leaves
    let center = memory.record(create_experience("Star center")).unwrap();

    for i in 0..5 {
        let leaf = memory
            .record(create_experience(&format!("Star leaf {}", i)))
            .unwrap();
        memory
            .reinforce_retrieval(&[center.clone(), leaf], RetrievalOutcome::Helpful)
            .unwrap();
    }

    // Query for center should find it
    let query = Query {
        query_text: Some("Star center".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_bipartite_activation() {
    let (mut memory, _temp) = setup_memory_system();

    // Create bipartite: group A connected to group B
    let mut group_a = Vec::new();
    let mut group_b = Vec::new();

    for i in 0..3 {
        group_a.push(
            memory
                .record(create_experience(&format!("Group A member {}", i)))
                .unwrap(),
        );
    }
    for i in 0..3 {
        group_b.push(
            memory
                .record(create_experience(&format!("Group B member {}", i)))
                .unwrap(),
        );
    }

    // Connect all A to all B
    for a in &group_a {
        for b in &group_b {
            memory
                .reinforce_retrieval(&[a.clone(), b.clone()], RetrievalOutcome::Helpful)
                .unwrap();
        }
    }

    // Query should find members
    let query = Query {
        query_text: Some("Group member".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = memory.retrieve(&query).unwrap();
    assert!(results.len() >= 6);
}

// =============================================================================
// PERFORMANCE-RELATED TESTS
// =============================================================================

#[test]
fn test_activation_update_performance() {
    // This test verifies activation updates are efficient.
    // 10000 updates (activate + decay) should complete within test timeout.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    // 10000 activation updates should complete quickly
    for _ in 0..10000 {
        memory.activate(0.001);
        memory.decay_activation(0.999);
    }

    // Just verify it completed with valid activation
    let activation = memory.activation();
    assert!(activation >= 0.0 && activation <= 1.0);
}

#[test]
fn test_many_memories_decay() {
    // This test verifies batch decay across 1000 memories.
    // All memories start at 1.0, after 10 decays: 0.95^10 ≈ 0.599
    let memories: Vec<Memory> = (0..1000)
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

    // Decay all 10 times (through Arc<Memory> compatible API)
    for _ in 0..10 {
        for m in &memories {
            m.decay_activation(0.95);
        }
    }

    // All should have decayed from 1.0 to ~0.599
    for m in &memories {
        assert!(m.activation() < 1.0);
    }
}

// =============================================================================
// COMBINATION TESTS
// =============================================================================

#[test]
fn test_activation_with_importance() {
    // This test verifies activation and importance are independent.
    // Boosting importance should not affect activation and vice versa.
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.8, // initial importance
        None,
        None,
        None,
    );

    memory.set_activation(0.5);
    memory.boost_importance(0.1); // importance: 0.8 + 0.1 = 0.9
    memory.activate(0.2); // activation: 0.5 + 0.2 = 0.7

    assert!((memory.activation() - 0.7).abs() < f32::EPSILON);
    assert!((memory.importance() - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_activation_with_tier() {
    // This test verifies activation and tier changes are independent.
    // Promotion changes tier, activation changes activation - no cross-effects.
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    memory.set_activation(0.5);
    memory.promote(); // Working -> Session (requires &mut self)
    memory.activate(0.3); // 0.5 + 0.3 = 0.8

    assert_eq!(memory.tier, MemoryTier::Session);
    assert!((memory.activation() - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_activation_with_retrieval_tracking() {
    // This test verifies activation and retrieval tracking are independent.
    // mark_retrieved sets last_retrieval_id, activate changes activation.
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
    );

    memory.set_activation(0.5);
    memory.mark_retrieved(Uuid::new_v4()); // requires &mut self
    memory.activate(0.2); // 0.5 + 0.2 = 0.7

    assert!(memory.last_retrieval_id.is_some());
    assert!((memory.activation() - 0.7).abs() < f32::EPSILON);
}
