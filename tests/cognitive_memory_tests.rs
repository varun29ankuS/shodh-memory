//! Cognitive Memory Type Tests
//!
//! Comprehensive tests for the unified Memory type with cognitive extensions:
//! - EntityRef bidirectional links
//! - MemoryTier promotion/demotion
//! - Activation levels (spreading activation)
//! - Retrieval tracking (Hebbian feedback)
//! - Serialization roundtrips
//! - NER integration for entity extraction

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{EntityRef, Experience, ExperienceType, Memory, MemoryId, MemoryTier};
use shodh_memory::uuid::Uuid;

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
        experience_type: exp_type,
        content: content.to_string(),
        entities: entity_names,
        ..Default::default()
    }
}

// =============================================================================
// MEMORY CREATION TESTS
// =============================================================================

#[test]
fn test_memory_new_defaults() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    assert_eq!(memory.tier, MemoryTier::Working);
    assert!(memory.entity_refs.is_empty());
    assert!((memory.activation() - 1.0).abs() < f32::EPSILON);
    assert!(memory.last_retrieval_id.is_none());
}

#[test]
fn test_memory_new_with_all_params() {
    let id = Uuid::new_v4();
    let memory = Memory::new(
        MemoryId(id),
        Experience {
            content: "Test content".to_string(),
            experience_type: ExperienceType::Decision,
            ..Default::default()
        },
        0.8,
        Some("agent-1".to_string()),
        Some("run-1".to_string()),
        Some("actor-1".to_string()),
        None, // created_at
    );

    assert_eq!(memory.id.0, id);
    assert_eq!(memory.experience.content, "Test content");
    assert!((memory.importance() - 0.8).abs() < f32::EPSILON);
    assert_eq!(memory.agent_id, Some("agent-1".to_string()));
    assert_eq!(memory.run_id, Some("run-1".to_string()));
    assert_eq!(memory.actor_id, Some("actor-1".to_string()));
}

// =============================================================================
// ENTITY REFERENCE TESTS
// =============================================================================

#[test]
fn test_add_entity_ref() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    let entity_id = Uuid::new_v4();
    memory.add_entity_ref(entity_id, "robot".to_string(), "mentioned".to_string());

    assert_eq!(memory.entity_refs.len(), 1);
    assert_eq!(memory.entity_refs[0].entity_id, entity_id);
    assert_eq!(memory.entity_refs[0].name, "robot");
    assert_eq!(memory.entity_refs[0].relation, "mentioned");
}

#[test]
fn test_add_entity_ref_no_duplicates() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    let entity_id = Uuid::new_v4();
    memory.add_entity_ref(entity_id, "robot".to_string(), "mentioned".to_string());
    memory.add_entity_ref(entity_id, "robot".to_string(), "mentioned".to_string());
    memory.add_entity_ref(entity_id, "robot".to_string(), "subject".to_string());

    assert_eq!(
        memory.entity_refs.len(),
        1,
        "Should not add duplicate entity"
    );
}

#[test]
fn test_add_multiple_entity_refs() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    let entity1 = Uuid::new_v4();
    let entity2 = Uuid::new_v4();
    let entity3 = Uuid::new_v4();

    memory.add_entity_ref(entity1, "robot".to_string(), "subject".to_string());
    memory.add_entity_ref(entity2, "warehouse".to_string(), "location".to_string());
    memory.add_entity_ref(entity3, "obstacle".to_string(), "mentioned".to_string());

    assert_eq!(memory.entity_refs.len(), 3);
}

#[test]
fn test_entity_ids() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    let entity1 = Uuid::new_v4();
    let entity2 = Uuid::new_v4();

    memory.add_entity_ref(entity1, "a".to_string(), "x".to_string());
    memory.add_entity_ref(entity2, "b".to_string(), "y".to_string());

    let ids = memory.entity_ids();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&entity1));
    assert!(ids.contains(&entity2));
}

// =============================================================================
// MEMORY TIER TESTS
// =============================================================================

#[test]
fn test_memory_tier_default() {
    let tier = MemoryTier::default();
    assert_eq!(tier, MemoryTier::Working);
}

#[test]
fn test_memory_tier_promotion() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    assert_eq!(memory.tier, MemoryTier::Working);

    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);

    memory.promote();
    assert_eq!(memory.tier, MemoryTier::LongTerm);

    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Archive);

    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Archive, "Should stay at Archive");
}

#[test]
fn test_memory_tier_demotion() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    // Start at Archive
    memory.tier = MemoryTier::Archive;

    memory.demote();
    assert_eq!(memory.tier, MemoryTier::LongTerm);

    memory.demote();
    assert_eq!(memory.tier, MemoryTier::Session);

    memory.demote();
    assert_eq!(memory.tier, MemoryTier::Working);

    memory.demote();
    assert_eq!(memory.tier, MemoryTier::Working, "Should stay at Working");
}

#[test]
fn test_tier_cycle() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    // Full promotion cycle
    for _ in 0..10 {
        memory.promote();
    }
    assert_eq!(memory.tier, MemoryTier::Archive);

    // Full demotion cycle
    for _ in 0..10 {
        memory.demote();
    }
    assert_eq!(memory.tier, MemoryTier::Working);
}

// =============================================================================
// ACTIVATION LEVEL TESTS (Spreading Activation)
// =============================================================================

#[test]
fn test_activation_default() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    assert!(
        (memory.activation() - 1.0).abs() < f32::EPSILON,
        "New memory should be fully activated"
    );
}

#[test]
fn test_activate_increases() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.set_activation(0.5);
    memory.activate(0.3);

    assert!((memory.activation() - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_activate_capped_at_one() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.set_activation(0.9);
    memory.activate(0.5);

    assert!(
        (memory.activation() - 1.0).abs() < f32::EPSILON,
        "Activation should cap at 1.0"
    );
}

#[test]
fn test_decay_activation() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.set_activation(1.0);
    memory.decay_activation(0.9);

    assert!((memory.activation() - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_decay_activation_multiple() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.set_activation(1.0);

    // Decay 10 times with 0.9 factor
    for _ in 0..10 {
        memory.decay_activation(0.9);
    }

    // 1.0 * 0.9^10 ≈ 0.3487
    assert!(memory.activation() > 0.3 && memory.activation() < 0.4);
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
        None,
    );

    memory.set_activation(1.0);

    // Heavy decay
    for _ in 0..100 {
        memory.decay_activation(0.9);
    }

    assert!(
        memory.activation() < 0.001,
        "Should approach zero with enough decay"
    );
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
        None,
    );

    memory.set_activation(0.5);
    memory.activate(0.3); // Now 0.8
    memory.decay_activation(0.5); // Now 0.4

    assert!((memory.activation() - 0.4).abs() < f32::EPSILON);
}

// =============================================================================
// RETRIEVAL TRACKING TESTS (Hebbian Feedback)
// =============================================================================

#[test]
fn test_mark_retrieved() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    assert!(memory.last_retrieval_id.is_none());

    let retrieval_id = Uuid::new_v4();
    memory.mark_retrieved(retrieval_id);

    assert_eq!(memory.last_retrieval_id, Some(retrieval_id));
}

#[test]
fn test_mark_retrieved_updates_access() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    let initial_access = memory.access_count();

    memory.mark_retrieved(Uuid::new_v4());

    assert_eq!(memory.access_count(), initial_access + 1);
}

#[test]
fn test_mark_retrieved_multiple() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    let retrieval1 = Uuid::new_v4();
    let retrieval2 = Uuid::new_v4();

    memory.mark_retrieved(retrieval1);
    assert_eq!(memory.last_retrieval_id, Some(retrieval1));

    memory.mark_retrieved(retrieval2);
    assert_eq!(
        memory.last_retrieval_id,
        Some(retrieval2),
        "Should update to latest"
    );
}

// =============================================================================
// IMPORTANCE TESTS
// =============================================================================

#[test]
fn test_importance_getter() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.75,
        None,
        None,
        None,
        None,
    );

    assert!((memory.importance() - 0.75).abs() < f32::EPSILON);
}

#[test]
fn test_importance_update() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.set_importance(0.9);
    assert!((memory.importance() - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_importance_clamped() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.set_importance(1.5);
    assert!(
        (memory.importance() - 1.0).abs() < f32::EPSILON,
        "Should clamp at 1.0"
    );

    memory.set_importance(-0.5);
    assert!(
        (memory.importance() - 0.0).abs() < f32::EPSILON,
        "Should clamp at 0.0"
    );
}

#[test]
fn test_boost_importance() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.boost_importance(0.2);
    assert!((memory.importance() - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_decay_importance() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    // decay_importance uses multiplicative decay: importance * (1.0 - decay)
    // 0.5 * (1.0 - 0.1) = 0.5 * 0.9 = 0.45
    memory.decay_importance(0.1);
    assert!((memory.importance() - 0.45).abs() < f32::EPSILON);
}

// =============================================================================
// ACCESS COUNT TESTS
// =============================================================================

#[test]
fn test_access_count_initial() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    assert_eq!(memory.access_count(), 0);
}

#[test]
fn test_record_access() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.record_access();
    assert_eq!(memory.access_count(), 1);

    memory.record_access();
    assert_eq!(memory.access_count(), 2);
}

#[test]
fn test_record_access_many() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    for _ in 0..1000 {
        memory.record_access();
    }

    assert_eq!(memory.access_count(), 1000);
}

// =============================================================================
// SERIALIZATION ROUNDTRIP TESTS
// =============================================================================

#[test]
fn test_bincode_roundtrip_basic() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: "Test content".to_string(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");

    assert_eq!(memory.id.0, deserialized.id.0);
    assert_eq!(memory.experience.content, deserialized.experience.content);
    assert!((memory.importance() - deserialized.importance()).abs() < f32::EPSILON);
}

#[test]
fn test_bincode_roundtrip_with_entity_refs() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    let entity1 = Uuid::new_v4();
    let entity2 = Uuid::new_v4();
    memory.add_entity_ref(entity1, "robot".to_string(), "subject".to_string());
    memory.add_entity_ref(entity2, "warehouse".to_string(), "location".to_string());

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");

    assert_eq!(deserialized.entity_refs.len(), 2);
    assert_eq!(deserialized.entity_refs[0].entity_id, entity1);
    assert_eq!(deserialized.entity_refs[1].entity_id, entity2);
}

#[test]
fn test_bincode_roundtrip_with_tier() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.tier = MemoryTier::LongTerm;

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");

    assert_eq!(deserialized.tier, MemoryTier::LongTerm);
}

#[test]
fn test_bincode_roundtrip_with_activation() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.set_activation(0.42);

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");

    assert!((deserialized.activation() - 0.42).abs() < f32::EPSILON);
}

#[test]
fn test_bincode_roundtrip_with_retrieval_id() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    let retrieval_id = Uuid::new_v4();
    memory.last_retrieval_id = Some(retrieval_id);

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");

    assert_eq!(deserialized.last_retrieval_id, Some(retrieval_id));
}

#[test]
fn test_bincode_roundtrip_full() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: "Full test with all fields".to_string(),
            experience_type: ExperienceType::Decision,
            entities: vec!["entity1".to_string(), "entity2".to_string()],
            ..Default::default()
        },
        0.85,
        Some("agent-1".to_string()),
        Some("run-1".to_string()),
        Some("actor-1".to_string()),
        None, // created_at
    );

    let entity1 = Uuid::new_v4();
    memory.add_entity_ref(entity1, "robot".to_string(), "subject".to_string());
    memory.tier = MemoryTier::Session;
    memory.set_activation(0.75);
    memory.last_retrieval_id = Some(Uuid::new_v4());

    for _ in 0..5 {
        memory.record_access();
    }

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");

    assert_eq!(memory.id.0, deserialized.id.0);
    assert_eq!(memory.experience.content, deserialized.experience.content);
    assert!((memory.importance() - deserialized.importance()).abs() < f32::EPSILON);
    assert_eq!(memory.agent_id, deserialized.agent_id);
    assert_eq!(memory.run_id, deserialized.run_id);
    assert_eq!(memory.actor_id, deserialized.actor_id);
    assert_eq!(memory.entity_refs.len(), deserialized.entity_refs.len());
    assert_eq!(memory.tier, deserialized.tier);
    assert!((memory.activation() - deserialized.activation()).abs() < f32::EPSILON);
    assert_eq!(memory.last_retrieval_id, deserialized.last_retrieval_id);
    assert_eq!(memory.access_count(), deserialized.access_count());
}

// =============================================================================
// JSON SERIALIZATION TESTS
// =============================================================================

#[test]
fn test_json_roundtrip() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: "JSON test".to_string(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );

    let json = serde_json::to_string(&memory).expect("Failed to serialize JSON");
    let deserialized: Memory = serde_json::from_str(&json).expect("Failed to deserialize JSON");

    assert_eq!(memory.id.0, deserialized.id.0);
    assert_eq!(memory.experience.content, deserialized.experience.content);
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_empty_content() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: "".to_string(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );

    assert!(memory.experience.content.is_empty());

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");
    assert!(deserialized.experience.content.is_empty());
}

#[test]
fn test_very_long_content() {
    let long_content = "x".repeat(100_000);
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: long_content.clone(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );

    assert_eq!(memory.experience.content.len(), 100_000);

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");
    assert_eq!(deserialized.experience.content.len(), 100_000);
}

#[test]
fn test_unicode_content() {
    let unicode_content = "こんにちは世界 🤖 émoji 中文 العربية";
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: unicode_content.to_string(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");
    assert_eq!(deserialized.experience.content, unicode_content);
}

#[test]
fn test_many_entity_refs() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    for i in 0..1000 {
        memory.add_entity_ref(
            Uuid::new_v4(),
            format!("entity_{}", i),
            "mentioned".to_string(),
        );
    }

    assert_eq!(memory.entity_refs.len(), 1000);

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
            .expect("Failed to deserialize");
    assert_eq!(deserialized.entity_refs.len(), 1000);
}

#[test]
fn test_zero_importance() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.0,
        None,
        None,
        None,
        None,
    );

    assert!((memory.importance() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_max_importance() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        1.0,
        None,
        None,
        None,
        None,
    );

    assert!((memory.importance() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_zero_activation() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );

    memory.set_activation(0.0);
    memory.activate(0.0);

    assert!((memory.activation() - 0.0).abs() < f32::EPSILON);
}

// =============================================================================
// CONCURRENT ACCESS TESTS
// =============================================================================

#[test]
fn test_concurrent_access_count() {
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

    let mut handles = vec![];

    for _ in 0..10 {
        let mem = Arc::clone(&memory);
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                // Read access count - should not panic
                let _ = mem.access_count();
                let _ = mem.importance();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_clone_independence() {
    let memory1 = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: "Original".to_string(),
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );

    let mut memory2 = memory1.clone();
    memory2.tier = MemoryTier::LongTerm;
    memory2.set_activation(0.1);

    // Original should be unchanged
    assert_eq!(memory1.tier, MemoryTier::Working);
    assert!((memory1.activation() - 1.0).abs() < f32::EPSILON);
}
