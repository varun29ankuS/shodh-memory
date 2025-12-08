//! Storage Edge Case Tests
//!
//! Tests edge cases in storage operations, compression, and embeddings
//! to ensure robust handling of unusual inputs.
//!
//! Run with: cargo test --test storage_edge_case_tests

use std::collections::HashMap;
use tempfile::TempDir;

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::types::{Experience, ExperienceType, GeoFilter, Query};
use shodh_memory::memory::{MemoryConfig, MemorySystem};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

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
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        entities: entity_names,
        ..Default::default()
    }
}

fn create_test_config(temp_dir: &TempDir) -> MemoryConfig {
    MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.3,
    }
}

// ============================================================================
// EMPTY AND WHITESPACE CONTENT TESTS
// ============================================================================

#[test]
fn test_empty_content_handling() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let exp = Experience {
        content: "".to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };

    // Empty content should be handled gracefully
    let result = system.record(exp);
    assert!(
        result.is_ok(),
        "Empty content should be accepted: {:?}",
        result.err()
    );
}

#[test]
fn test_whitespace_only_content() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let exp = Experience {
        content: "   \t\n\r   ".to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Whitespace-only content should be accepted");
}

#[test]
fn test_very_long_content() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // 100KB of content
    let long_content = "x".repeat(100 * 1024);
    let exp = Experience {
        content: long_content,
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Long content should be accepted");
}

// ============================================================================
// UNICODE AND SPECIAL CHARACTER TESTS
// ============================================================================

#[test]
fn test_unicode_content() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let exp = Experience {
        content: "Unicode test: 你好世界 🤖 λ∑∏ العربية עברית".to_string(),
        experience_type: ExperienceType::Observation,
        entities: vec!["unicode".to_string()],
        ..Default::default()
    };

    let memory_id = system
        .record(exp)
        .expect("Failed to record unicode content");

    // Verify retrieval
    let query = Query {
        query_text: Some("Unicode test".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let results = system.retrieve(&query).expect("Failed to retrieve");
    assert!(!results.is_empty(), "Unicode content should be retrievable");
    assert!(
        results[0].experience.content.contains("你好世界"),
        "Unicode should be preserved"
    );
}

#[test]
fn test_special_characters_content() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let exp = Experience {
        content: r#"Special chars: <>&"' \n\t\r `~!@#$%^&*()[]{}|;:,.<>?"#.to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Special characters should be accepted");
}

#[test]
fn test_null_bytes_content() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let exp = Experience {
        content: "Content with\0null\0bytes".to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Null bytes should be handled");
}

// ============================================================================
// ENTITY EDGE CASES
// ============================================================================

#[test]
fn test_many_entities() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // 100 entities
    let entities: Vec<String> = (0..100).map(|i| format!("entity_{}", i)).collect();
    let exp = Experience {
        content: "Memory with many entities".to_string(),
        experience_type: ExperienceType::Observation,
        entities,
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Many entities should be accepted");
}

#[test]
fn test_empty_entity_name() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let exp = Experience {
        content: "Memory with empty entity".to_string(),
        experience_type: ExperienceType::Observation,
        entities: vec!["".to_string(), "valid_entity".to_string()],
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Empty entity names should be handled");
}

#[test]
fn test_unicode_entity_names() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let exp = Experience {
        content: "Memory with unicode entities".to_string(),
        experience_type: ExperienceType::Observation,
        entities: vec!["实体".to_string(), "🤖".to_string(), "λ".to_string()],
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Unicode entity names should be accepted");

    // Search by unicode tag
    let query = Query {
        tags: Some(vec!["实体".to_string()]),
        max_results: 5,
        ..Default::default()
    };

    let results = system.retrieve(&query).expect("Failed to retrieve");
    // May or may not find results depending on tag indexing
}

// ============================================================================
// GEO LOCATION EDGE CASES
// ============================================================================

#[test]
fn test_geo_location_boundary_values() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Test boundary coordinates
    let boundary_locations = [
        ([90.0, 0.0, 0.0], "North pole"),
        ([-90.0, 0.0, 0.0], "South pole"),
        ([0.0, 180.0, 0.0], "Dateline east"),
        ([0.0, -180.0, 0.0], "Dateline west"),
        ([0.0, 0.0, 0.0], "Origin"),
    ];

    for (coords, name) in boundary_locations {
        let exp = Experience {
            content: format!("Location at {}", name),
            experience_type: ExperienceType::Observation,
            geo_location: Some(coords),
            ..Default::default()
        };

        let result = system.record(exp);
        assert!(
            result.is_ok(),
            "Boundary location {} should be accepted",
            name
        );
    }
}

#[test]
fn test_geo_location_invalid_coordinates() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Invalid coordinates (should be clamped/handled)
    let exp = Experience {
        content: "Invalid geo location".to_string(),
        experience_type: ExperienceType::Observation,
        geo_location: Some([100.0, 200.0, 0.0]), // Invalid: lat > 90, lon > 180
        ..Default::default()
    };

    // Should either clamp or reject gracefully
    let result = system.record(exp);
    // Either way, it shouldn't crash
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_geo_filter_zero_radius() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Record a memory with location
    let exp = Experience {
        content: "Zero radius test".to_string(),
        experience_type: ExperienceType::Observation,
        geo_location: Some([37.7749, -122.4194, 0.0]),
        ..Default::default()
    };
    system.record(exp).expect("Failed to record");

    // Query with zero radius
    let query = Query {
        query_text: Some("Zero radius".to_string()),
        geo_filter: Some(GeoFilter::new(37.7749, -122.4194, 0.0)),
        max_results: 5,
        ..Default::default()
    };

    let result = system.retrieve(&query);
    assert!(result.is_ok(), "Zero radius query should not crash");
}

// ============================================================================
// METADATA EDGE CASES
// ============================================================================

#[test]
fn test_large_metadata() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Large metadata map
    let mut metadata = HashMap::new();
    for i in 0..100 {
        metadata.insert(format!("key_{}", i), format!("value_{}", i));
    }

    let exp = Experience {
        content: "Memory with large metadata".to_string(),
        experience_type: ExperienceType::Observation,
        metadata,
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Large metadata should be accepted");
}

#[test]
fn test_unicode_metadata_keys_values() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let mut metadata = HashMap::new();
    metadata.insert("键".to_string(), "值".to_string());
    metadata.insert("🔑".to_string(), "🗝️".to_string());

    let exp = Experience {
        content: "Unicode metadata test".to_string(),
        experience_type: ExperienceType::Observation,
        metadata,
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Unicode metadata should be accepted");
}

// ============================================================================
// SENSOR DATA EDGE CASES
// ============================================================================

#[test]
fn test_sensor_data_extreme_values() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let mut sensor_data = HashMap::new();
    sensor_data.insert("normal".to_string(), 42.0);
    sensor_data.insert("very_small".to_string(), 1e-300);
    sensor_data.insert("very_large".to_string(), 1e300);
    sensor_data.insert("negative".to_string(), -1000.0);
    sensor_data.insert("zero".to_string(), 0.0);

    let exp = Experience {
        content: "Sensor data with extreme values".to_string(),
        experience_type: ExperienceType::Observation,
        sensor_data,
        ..Default::default()
    };

    let result = system.record(exp);
    assert!(result.is_ok(), "Extreme sensor values should be accepted");
}

#[test]
fn test_sensor_data_special_floats() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let mut sensor_data = HashMap::new();
    sensor_data.insert("infinity".to_string(), f64::INFINITY);
    sensor_data.insert("neg_infinity".to_string(), f64::NEG_INFINITY);
    sensor_data.insert("nan".to_string(), f64::NAN);

    let exp = Experience {
        content: "Sensor data with special floats".to_string(),
        experience_type: ExperienceType::Observation,
        sensor_data,
        ..Default::default()
    };

    // Should handle NaN/Infinity gracefully
    let result = system.record(exp);
    // May succeed or fail, but shouldn't panic
    assert!(result.is_ok() || result.is_err());
}

// ============================================================================
// QUERY EDGE CASES
// ============================================================================

#[test]
fn test_query_very_long_text() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Record some memories
    for i in 0..5 {
        let exp = Experience {
            content: format!("Test memory {} for query testing", i),
            experience_type: ExperienceType::Observation,
            ..Default::default()
        };
        system.record(exp).expect("Failed to record");
    }

    // Query with very long text
    let long_query = "word ".repeat(1000);
    let query = Query {
        query_text: Some(long_query),
        max_results: 5,
        ..Default::default()
    };

    let result = system.retrieve(&query);
    assert!(result.is_ok(), "Long query text should not crash");
}

#[test]
fn test_query_max_results_zero() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Record a memory
    let exp = Experience {
        content: "Test memory".to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };
    system.record(exp).expect("Failed to record");

    // Query with max_results = 0
    let query = Query {
        query_text: Some("Test".to_string()),
        max_results: 0,
        ..Default::default()
    };

    // Zero max_results should not crash - behavior may vary
    let result = system.retrieve(&query);
    assert!(result.is_ok(), "Zero max_results should not crash");
}

#[test]
fn test_query_max_results_very_large() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Record a few memories
    for i in 0..5 {
        let exp = Experience {
            content: format!("Test memory {}", i),
            experience_type: ExperienceType::Observation,
            ..Default::default()
        };
        system.record(exp).expect("Failed to record");
    }

    // Query with very large max_results
    let query = Query {
        query_text: Some("Test".to_string()),
        max_results: 1_000_000,
        ..Default::default()
    };

    let result = system.retrieve(&query);
    assert!(result.is_ok(), "Large max_results should not crash");
    // Should only return actual memories (5), not allocate for 1M
}

// ============================================================================
// REWARD AND IMPORTANCE EDGE CASES
// ============================================================================

#[test]
fn test_reward_boundary_values() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let rewards = [
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        -1.1,
        1.1,
        f32::NAN,
        f32::INFINITY,
    ];

    for reward in rewards {
        let exp = Experience {
            content: format!("Reward test: {}", reward),
            experience_type: ExperienceType::Observation,
            reward: Some(reward),
            ..Default::default()
        };

        // Should handle all reward values gracefully
        let _ = system.record(exp);
    }

    // System should still be functional
    let stats = system.stats();
    assert!(stats.total_memories > 0, "Some memories should be recorded");
}

#[test]
fn test_confidence_boundary_values() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let confidences = [0.0, 0.5, 1.0, -0.1, 1.1];

    for confidence in confidences {
        let exp = Experience {
            content: format!("Confidence test: {}", confidence),
            experience_type: ExperienceType::Decision,
            confidence: Some(confidence),
            ..Default::default()
        };

        let _ = system.record(exp);
    }

    let stats = system.stats();
    assert!(stats.total_memories > 0, "Some memories should be recorded");
}

// ============================================================================
// CONCURRENT STRESS TESTS
// ============================================================================

#[test]
fn test_rapid_sequential_operations() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Rapidly alternate between record and retrieve
    for i in 0..100 {
        if i % 2 == 0 {
            let exp = Experience {
                content: format!("Rapid test {}", i),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            };
            let _ = system.record(exp);
        } else {
            let query = Query {
                query_text: Some("Rapid test".to_string()),
                max_results: 5,
                ..Default::default()
            };
            let _ = system.retrieve(&query);
        }
    }

    let stats = system.stats();
    assert_eq!(
        stats.total_memories, 50,
        "50 memories should be recorded from 100 alternating ops"
    );
}

#[test]
fn test_flush_under_load() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Record memories and flush multiple times
    for batch in 0..5 {
        for i in 0..20 {
            let exp = Experience {
                content: format!("Batch {} memory {}", batch, i),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            };
            system.record(exp).expect("Failed to record");
        }

        system.flush_storage().expect("Failed to flush");
    }

    let stats = system.stats();
    assert_eq!(
        stats.total_memories, 100,
        "100 memories should survive multiple flushes"
    );
}

// ============================================================================
// PERSISTENCE TESTS
// ============================================================================

#[test]
fn test_data_survives_reopen() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().to_path_buf();

    // First session: record memories
    {
        let config = MemoryConfig {
            storage_path: path.clone(),
            working_memory_size: 100,
            session_memory_size_mb: 50,
            max_heap_per_user_mb: 500,
            auto_compress: false,
            compression_age_days: 30,
            importance_threshold: 0.3,
        };
        let system = MemorySystem::new(config).expect("Failed to create memory system");

        for i in 0..10 {
            let exp = Experience {
                content: format!("Persistence test memory {}", i),
                experience_type: ExperienceType::Learning,
                entities: vec!["persistence".to_string()],
                ..Default::default()
            };
            system.record(exp).expect("Failed to record");
        }

        system.flush_storage().expect("Failed to flush");
    }

    // Second session: verify data exists
    {
        let config = MemoryConfig {
            storage_path: path,
            working_memory_size: 100,
            session_memory_size_mb: 50,
            max_heap_per_user_mb: 500,
            auto_compress: false,
            compression_age_days: 30,
            importance_threshold: 0.3,
        };
        let system = MemorySystem::new(config).expect("Failed to reopen memory system");

        let stats = system.stats();
        assert_eq!(
            stats.total_memories, 10,
            "10 memories should persist after reopen"
        );

        // Verify retrieval works
        let query = Query {
            query_text: Some("Persistence test".to_string()),
            max_results: 20,
            ..Default::default()
        };

        let results = system.retrieve(&query).expect("Failed to retrieve");
        assert!(
            results.len() >= 5,
            "Should retrieve persisted memories, got {}",
            results.len()
        );
    }
}
