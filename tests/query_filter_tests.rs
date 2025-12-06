//! Comprehensive tests for Query::matches() filter logic
//!
//! Tests the SINGLE source of truth for memory filtering across all tiers.

use chrono::{Duration, Utc};
use std::collections::HashMap;
use std::sync::Arc;

use shodh_memory::memory::types::{
    Experience, ExperienceType, GeoFilter, Memory, MemoryId, Query, RetrievalMode,
};
use shodh_memory::uuid::Uuid;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn create_test_memory(content: &str, importance: f32) -> Memory {
    let experience = Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };
    Memory::new(
        MemoryId(Uuid::new_v4()),
        experience,
        importance,
        None,
        None,
        None,
    )
}

fn create_robotics_memory(
    content: &str,
    robot_id: Option<&str>,
    mission_id: Option<&str>,
    geo_location: Option<[f64; 3]>,
    action_type: Option<&str>,
    reward: Option<f32>,
) -> Memory {
    let experience = Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Task,
        robot_id: robot_id.map(String::from),
        mission_id: mission_id.map(String::from),
        geo_location,
        action_type: action_type.map(String::from),
        reward,
        ..Default::default()
    };
    Memory::new(MemoryId(Uuid::new_v4()), experience, 0.5, None, None, None)
}

fn create_decision_memory(
    content: &str,
    outcome_type: Option<&str>,
    is_anomaly: bool,
    severity: Option<&str>,
    tags: Vec<&str>,
    confidence: Option<f32>,
) -> Memory {
    let experience = Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Decision,
        outcome_type: outcome_type.map(String::from),
        is_anomaly,
        severity: severity.map(String::from),
        tags: tags.into_iter().map(String::from).collect(),
        confidence,
        ..Default::default()
    };
    Memory::new(MemoryId(Uuid::new_v4()), experience, 0.5, None, None, None)
}

// ============================================================================
// IMPORTANCE THRESHOLD TESTS
// ============================================================================

#[test]
fn test_importance_threshold_exact_match() {
    let memory = create_test_memory("test", 0.5);
    let query = Query {
        importance_threshold: Some(0.5),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Memory at threshold should match");
}

#[test]
fn test_importance_threshold_above() {
    let memory = create_test_memory("test", 0.7);
    let query = Query {
        importance_threshold: Some(0.5),
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "Memory above threshold should match"
    );
}

#[test]
fn test_importance_threshold_below() {
    let memory = create_test_memory("test", 0.3);
    let query = Query {
        importance_threshold: Some(0.5),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Memory below threshold should not match"
    );
}

#[test]
fn test_importance_threshold_none() {
    let memory = create_test_memory("test", 0.1);
    let query = Query::default();
    assert!(
        query.matches(&memory),
        "No threshold means all memories match"
    );
}

// ============================================================================
// EXPERIENCE TYPE TESTS
// ============================================================================

#[test]
fn test_experience_type_single_match() {
    let experience = Experience {
        content: "test".to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    };
    let memory = Memory::new(MemoryId(Uuid::new_v4()), experience, 0.5, None, None, None);

    let query = Query {
        experience_types: Some(vec![ExperienceType::Observation]),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Matching type should match");
}

#[test]
fn test_experience_type_no_match() {
    let experience = Experience {
        content: "test".to_string(),
        experience_type: ExperienceType::Decision,
        ..Default::default()
    };
    let memory = Memory::new(MemoryId(Uuid::new_v4()), experience, 0.5, None, None, None);

    let query = Query {
        experience_types: Some(vec![ExperienceType::Observation]),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Non-matching type should not match"
    );
}

#[test]
fn test_experience_type_multiple_allowed() {
    let experience = Experience {
        content: "test".to_string(),
        experience_type: ExperienceType::Task,
        ..Default::default()
    };
    let memory = Memory::new(MemoryId(Uuid::new_v4()), experience, 0.5, None, None, None);

    let query = Query {
        experience_types: Some(vec![
            ExperienceType::Observation,
            ExperienceType::Task,
            ExperienceType::Decision,
        ]),
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "Memory type in allowed list should match"
    );
}

// ============================================================================
// TIME RANGE TESTS
// ============================================================================

#[test]
fn test_time_range_within() {
    let memory = create_test_memory("test", 0.5);
    let now = Utc::now();
    let query = Query {
        time_range: Some((now - Duration::hours(1), now + Duration::hours(1))),
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "Recent memory should be within range"
    );
}

#[test]
fn test_time_range_before_start() {
    let mut memory = create_test_memory("test", 0.5);
    // Manually set created_at to 2 days ago
    memory.created_at = Utc::now() - Duration::days(2);

    let now = Utc::now();
    let query = Query {
        time_range: Some((now - Duration::hours(1), now + Duration::hours(1))),
        ..Default::default()
    };
    assert!(!query.matches(&memory), "Old memory should be before range");
}

// ============================================================================
// ROBOT ID FILTER TESTS
// ============================================================================

#[test]
fn test_robot_id_match() {
    let memory = create_robotics_memory("test", Some("robot_001"), None, None, None, None);
    let query = Query {
        robot_id: Some("robot_001".to_string()),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Matching robot_id should match");
}

#[test]
fn test_robot_id_no_match() {
    let memory = create_robotics_memory("test", Some("robot_001"), None, None, None, None);
    let query = Query {
        robot_id: Some("robot_002".to_string()),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Different robot_id should not match"
    );
}

#[test]
fn test_robot_id_memory_has_none() {
    let memory = create_robotics_memory("test", None, None, None, None, None);
    let query = Query {
        robot_id: Some("robot_001".to_string()),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Memory without robot_id should not match filter"
    );
}

// ============================================================================
// MISSION ID FILTER TESTS
// ============================================================================

#[test]
fn test_mission_id_match() {
    let memory = create_robotics_memory("test", None, Some("mission_alpha"), None, None, None);
    let query = Query {
        mission_id: Some("mission_alpha".to_string()),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Matching mission_id should match");
}

#[test]
fn test_mission_id_no_match() {
    let memory = create_robotics_memory("test", None, Some("mission_alpha"), None, None, None);
    let query = Query {
        mission_id: Some("mission_beta".to_string()),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Different mission_id should not match"
    );
}

// ============================================================================
// GEO FILTER TESTS
// ============================================================================

#[test]
fn test_geo_filter_within_radius() {
    // Memory at exact center
    let memory = create_robotics_memory(
        "test",
        None,
        None,
        Some([37.7749, -122.4194, 0.0]),
        None,
        None,
    );
    // Filter at same location with 1000m radius
    let query = Query {
        geo_filter: Some(GeoFilter::new(37.7749, -122.4194, 1000.0)),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Memory at center should match");
}

#[test]
fn test_geo_filter_outside_radius() {
    // Memory 10km away (roughly 0.1 degrees)
    let memory = create_robotics_memory(
        "test",
        None,
        None,
        Some([37.8749, -122.4194, 0.0]),
        None,
        None,
    );
    // Filter with 1000m radius
    let query = Query {
        geo_filter: Some(GeoFilter::new(37.7749, -122.4194, 1000.0)),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Memory outside radius should not match"
    );
}

#[test]
fn test_geo_filter_memory_no_location() {
    let memory = create_robotics_memory("test", None, None, None, None, None);
    let query = Query {
        geo_filter: Some(GeoFilter::new(37.7749, -122.4194, 1000.0)),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Memory without location should not match geo filter"
    );
}

// ============================================================================
// ACTION TYPE FILTER TESTS
// ============================================================================

#[test]
fn test_action_type_match() {
    let memory = create_robotics_memory("test", None, None, None, Some("navigate"), None);
    let query = Query {
        action_type: Some("navigate".to_string()),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Matching action_type should match");
}

#[test]
fn test_action_type_no_match() {
    let memory = create_robotics_memory("test", None, None, None, Some("navigate"), None);
    let query = Query {
        action_type: Some("grasp".to_string()),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Different action_type should not match"
    );
}

// ============================================================================
// REWARD RANGE FILTER TESTS
// ============================================================================

#[test]
fn test_reward_range_within() {
    let memory = create_robotics_memory("test", None, None, None, None, Some(0.75));
    let query = Query {
        reward_range: Some((0.5, 1.0)),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Reward within range should match");
}

#[test]
fn test_reward_range_below() {
    let memory = create_robotics_memory("test", None, None, None, None, Some(0.3));
    let query = Query {
        reward_range: Some((0.5, 1.0)),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Reward below range should not match"
    );
}

#[test]
fn test_reward_range_above() {
    let memory = create_robotics_memory("test", None, None, None, None, Some(1.5));
    let query = Query {
        reward_range: Some((0.5, 1.0)),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Reward above range should not match"
    );
}

#[test]
fn test_reward_range_no_reward() {
    let memory = create_robotics_memory("test", None, None, None, None, None);
    let query = Query {
        reward_range: Some((0.5, 1.0)),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Memory without reward should not match reward filter"
    );
}

// ============================================================================
// OUTCOME TYPE FILTER TESTS
// ============================================================================

#[test]
fn test_outcome_type_match() {
    let memory = create_decision_memory("test", Some("success"), false, None, vec![], None);
    let query = Query {
        outcome_type: Some("success".to_string()),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Matching outcome should match");
}

#[test]
fn test_outcome_type_no_match() {
    let memory = create_decision_memory("test", Some("success"), false, None, vec![], None);
    let query = Query {
        outcome_type: Some("failure".to_string()),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Different outcome should not match"
    );
}

// ============================================================================
// FAILURES ONLY FILTER TESTS
// ============================================================================

#[test]
fn test_failures_only_matches_failure() {
    let memory = create_decision_memory("test", Some("failure"), false, None, vec![], None);
    let query = Query {
        failures_only: true,
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "Failure outcome should match failures_only"
    );
}

#[test]
fn test_failures_only_matches_failed() {
    let memory = create_decision_memory("test", Some("failed"), false, None, vec![], None);
    let query = Query {
        failures_only: true,
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "'failed' outcome should match failures_only"
    );
}

#[test]
fn test_failures_only_matches_error() {
    let memory = create_decision_memory("test", Some("error"), false, None, vec![], None);
    let query = Query {
        failures_only: true,
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "'error' outcome should match failures_only"
    );
}

#[test]
fn test_failures_only_rejects_success() {
    let memory = create_decision_memory("test", Some("success"), false, None, vec![], None);
    let query = Query {
        failures_only: true,
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Success outcome should not match failures_only"
    );
}

#[test]
fn test_failures_only_rejects_no_outcome() {
    let memory = create_decision_memory("test", None, false, None, vec![], None);
    let query = Query {
        failures_only: true,
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "No outcome should not match failures_only"
    );
}

// ============================================================================
// ANOMALIES ONLY FILTER TESTS
// ============================================================================

#[test]
fn test_anomalies_only_matches_anomaly() {
    let memory = create_decision_memory("test", None, true, None, vec![], None);
    let query = Query {
        anomalies_only: true,
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "Anomaly should match anomalies_only"
    );
}

#[test]
fn test_anomalies_only_rejects_normal() {
    let memory = create_decision_memory("test", None, false, None, vec![], None);
    let query = Query {
        anomalies_only: true,
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Non-anomaly should not match anomalies_only"
    );
}

// ============================================================================
// SEVERITY FILTER TESTS
// ============================================================================

#[test]
fn test_severity_match() {
    let memory = create_decision_memory("test", None, false, Some("critical"), vec![], None);
    let query = Query {
        severity: Some("critical".to_string()),
        ..Default::default()
    };
    assert!(query.matches(&memory), "Matching severity should match");
}

#[test]
fn test_severity_no_match() {
    let memory = create_decision_memory("test", None, false, Some("warning"), vec![], None);
    let query = Query {
        severity: Some("critical".to_string()),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Different severity should not match"
    );
}

// ============================================================================
// TAGS FILTER TESTS
// ============================================================================

#[test]
fn test_tags_any_match() {
    let memory = create_decision_memory(
        "test",
        None,
        false,
        None,
        vec!["robot", "autonomous", "navigation"],
        None,
    );
    let query = Query {
        tags: Some(vec!["navigation".to_string()]),
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "Memory with matching tag should match"
    );
}

#[test]
fn test_tags_multiple_query_one_match() {
    let memory = create_decision_memory("test", None, false, None, vec!["robot", "sensor"], None);
    let query = Query {
        tags: Some(vec![
            "navigation".to_string(),
            "sensor".to_string(),
            "arm".to_string(),
        ]),
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "Any matching tag should be sufficient"
    );
}

#[test]
fn test_tags_no_match() {
    let memory = create_decision_memory("test", None, false, None, vec!["robot", "sensor"], None);
    let query = Query {
        tags: Some(vec!["navigation".to_string(), "arm".to_string()]),
        ..Default::default()
    };
    assert!(!query.matches(&memory), "No matching tags should not match");
}

#[test]
fn test_tags_memory_empty() {
    let memory = create_decision_memory("test", None, false, None, vec![], None);
    let query = Query {
        tags: Some(vec!["robot".to_string()]),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Memory with empty tags should not match tag filter"
    );
}

// ============================================================================
// CONFIDENCE RANGE FILTER TESTS
// ============================================================================

#[test]
fn test_confidence_range_within() {
    let memory = create_decision_memory("test", None, false, None, vec![], Some(0.85));
    let query = Query {
        confidence_range: Some((0.8, 1.0)),
        ..Default::default()
    };
    assert!(
        query.matches(&memory),
        "Confidence within range should match"
    );
}

#[test]
fn test_confidence_range_below() {
    let memory = create_decision_memory("test", None, false, None, vec![], Some(0.7));
    let query = Query {
        confidence_range: Some((0.8, 1.0)),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Confidence below range should not match"
    );
}

#[test]
fn test_confidence_range_no_confidence() {
    let memory = create_decision_memory("test", None, false, None, vec![], None);
    let query = Query {
        confidence_range: Some((0.8, 1.0)),
        ..Default::default()
    };
    assert!(
        !query.matches(&memory),
        "Memory without confidence should not match confidence filter"
    );
}

// ============================================================================
// COMBINED FILTER TESTS
// ============================================================================

#[test]
fn test_combined_all_match() {
    let experience = Experience {
        content: "Robot mission complete".to_string(),
        experience_type: ExperienceType::Task,
        robot_id: Some("robot_001".to_string()),
        mission_id: Some("mission_alpha".to_string()),
        geo_location: Some([37.7749, -122.4194, 0.0]),
        action_type: Some("navigate".to_string()),
        reward: Some(0.9),
        outcome_type: Some("success".to_string()),
        severity: Some("info".to_string()),
        tags: vec!["autonomous".to_string(), "navigation".to_string()],
        confidence: Some(0.95),
        ..Default::default()
    };
    let memory = Memory::new(MemoryId(Uuid::new_v4()), experience, 0.8, None, None, None);

    let query = Query {
        importance_threshold: Some(0.5),
        experience_types: Some(vec![ExperienceType::Task]),
        robot_id: Some("robot_001".to_string()),
        mission_id: Some("mission_alpha".to_string()),
        geo_filter: Some(GeoFilter::new(37.7749, -122.4194, 1000.0)),
        action_type: Some("navigate".to_string()),
        reward_range: Some((0.5, 1.0)),
        outcome_type: Some("success".to_string()),
        severity: Some("info".to_string()),
        tags: Some(vec!["navigation".to_string()]),
        confidence_range: Some((0.9, 1.0)),
        ..Default::default()
    };

    assert!(
        query.matches(&memory),
        "Memory matching all filters should match"
    );
}

#[test]
fn test_combined_one_fails() {
    let experience = Experience {
        content: "Robot mission complete".to_string(),
        experience_type: ExperienceType::Task,
        robot_id: Some("robot_001".to_string()),
        mission_id: Some("mission_alpha".to_string()),
        reward: Some(0.9),
        ..Default::default()
    };
    let memory = Memory::new(MemoryId(Uuid::new_v4()), experience, 0.8, None, None, None);

    // Query expects robot_002 but memory has robot_001
    let query = Query {
        importance_threshold: Some(0.5),
        robot_id: Some("robot_002".to_string()),
        mission_id: Some("mission_alpha".to_string()),
        ..Default::default()
    };

    assert!(
        !query.matches(&memory),
        "One failing filter should reject the memory"
    );
}

// ============================================================================
// QUERY BUILDER TESTS
// ============================================================================

#[test]
fn test_query_builder_basic() {
    let query = Query::builder()
        .query_text("test query")
        .importance_threshold(0.5)
        .max_results(20)
        .build();

    assert_eq!(query.query_text, Some("test query".to_string()));
    assert_eq!(query.importance_threshold, Some(0.5));
    assert_eq!(query.max_results, 20);
}

#[test]
fn test_query_builder_robotics() {
    let query = Query::builder()
        .robot_id("robot_001")
        .mission_id("mission_alpha")
        .failures_only(true)
        .anomalies_only(true)
        .build();

    assert_eq!(query.robot_id, Some("robot_001".to_string()));
    assert_eq!(query.mission_id, Some("mission_alpha".to_string()));
    assert!(query.failures_only);
    assert!(query.anomalies_only);
}

#[test]
fn test_query_builder_with_tags() {
    let query = Query::builder()
        .tags(vec!["robot".to_string(), "autonomous".to_string()])
        .retrieval_mode(RetrievalMode::Hybrid)
        .build();

    assert_eq!(
        query.tags,
        Some(vec!["robot".to_string(), "autonomous".to_string()])
    );
    assert_eq!(query.retrieval_mode, RetrievalMode::Hybrid);
}
