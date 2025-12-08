//! Universe Visualization Tests
//!
//! Tests for the Memory Universe API:
//! - 3D galaxy layout generation
//! - Star positioning based on salience
//! - Connection visualization
//! - Entity type coloring
//! - Bounds calculation

use chrono::Utc;
use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::graph_memory::{
    EntityLabel, EntityNode, GraphMemory, RelationType, RelationshipEdge,
};
use shodh_memory::uuid::Uuid;
use std::collections::HashMap;
use tempfile::TempDir;

/// Create fallback NER for testing (rule-based, no ONNX required)
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Convert NER entity type to EntityLabel
fn ner_type_to_label(ner_type: &str) -> EntityLabel {
    match ner_type {
        "PER" => EntityLabel::Person,
        "ORG" => EntityLabel::Organization,
        "LOC" => EntityLabel::Location,
        _ => EntityLabel::Concept,
    }
}

/// Create entity from NER extraction result
fn create_entity_from_ner(
    name: &str,
    ner_type: &str,
    is_proper: bool,
    salience: f32,
) -> EntityNode {
    EntityNode {
        uuid: Uuid::new_v4(),
        name: name.to_string(),
        labels: vec![ner_type_to_label(ner_type)],
        created_at: Utc::now(),
        last_seen_at: Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: HashMap::new(),
        name_embedding: None,
        salience,
        is_proper_noun: is_proper,
    }
}

/// Create a test graph memory instance
fn setup_graph_memory() -> (GraphMemory, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let graph = GraphMemory::new(temp_dir.path()).expect("Failed to create graph memory");
    (graph, temp_dir)
}

/// Create a new entity node with the given parameters
fn create_entity(
    name: &str,
    label: Option<EntityLabel>,
    is_proper: bool,
    salience: f32,
) -> EntityNode {
    EntityNode {
        uuid: Uuid::new_v4(),
        name: name.to_string(),
        labels: label.map(|l| vec![l]).unwrap_or_default(),
        created_at: Utc::now(),
        last_seen_at: Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: HashMap::new(),
        name_embedding: None,
        salience,
        is_proper_noun: is_proper,
    }
}

/// Create a new relationship edge
fn create_relationship(
    from: Uuid,
    to: Uuid,
    rel_type: RelationType,
    strength: f32,
) -> RelationshipEdge {
    RelationshipEdge {
        uuid: Uuid::new_v4(),
        from_entity: from,
        to_entity: to,
        relation_type: rel_type,
        strength,
        created_at: Utc::now(),
        valid_at: Utc::now(),
        invalidated_at: None,
        source_episode_id: None,
        context: String::new(),
        // Hebbian plasticity fields
        last_activated: Utc::now(),
        activation_count: 0,
        potentiated: false,
    }
}

// =============================================================================
// UNIVERSE STRUCTURE TESTS
// =============================================================================

#[test]
fn test_empty_universe() {
    let (graph, _temp_dir) = setup_graph_memory();

    let universe = graph.get_universe().expect("Failed to get universe");

    assert!(
        universe.stars.is_empty(),
        "Empty graph should have no stars"
    );
    assert!(
        universe.connections.is_empty(),
        "Empty graph should have no connections"
    );
    assert_eq!(universe.total_entities, 0);
    assert_eq!(universe.total_connections, 0);
}

#[test]
fn test_universe_with_entities() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add some entities
    let sun = create_entity("Sun", Some(EntityLabel::Location), true, 0.9);
    let earth = create_entity("Earth", Some(EntityLabel::Location), true, 0.7);
    let mars = create_entity("Mars", Some(EntityLabel::Location), true, 0.6);

    graph.add_entity(sun).expect("Failed to add Sun");
    graph.add_entity(earth).expect("Failed to add Earth");
    graph.add_entity(mars).expect("Failed to add Mars");

    let universe = graph.get_universe().expect("Failed to get universe");

    assert_eq!(universe.stars.len(), 3, "Should have 3 stars");
    assert_eq!(universe.total_entities, 3);
}

#[test]
fn test_universe_with_connections() {
    let (graph, _temp_dir) = setup_graph_memory();

    let sun_entity = create_entity("Sun", Some(EntityLabel::Location), true, 0.9);
    let earth_entity = create_entity("Earth", Some(EntityLabel::Location), true, 0.7);
    let moon_entity = create_entity("Moon", Some(EntityLabel::Location), true, 0.5);

    let sun = graph.add_entity(sun_entity).expect("Failed");
    let earth = graph.add_entity(earth_entity).expect("Failed");
    let moon = graph.add_entity(moon_entity).expect("Failed");

    // Create orbital relationships
    graph
        .add_relationship(create_relationship(
            earth,
            sun,
            RelationType::RelatedTo,
            0.9,
        ))
        .expect("Failed to add Earth->Sun");
    graph
        .add_relationship(create_relationship(
            moon,
            earth,
            RelationType::RelatedTo,
            0.95,
        ))
        .expect("Failed to add Moon->Earth");

    let universe = graph.get_universe().expect("Failed to get universe");

    assert_eq!(universe.stars.len(), 3, "Should have 3 stars");
    assert_eq!(universe.connections.len(), 2, "Should have 2 connections");
    assert_eq!(universe.total_connections, 2);
}

// =============================================================================
// STAR POSITIONING TESTS (SALIENCE-BASED)
// =============================================================================

#[test]
fn test_high_salience_near_center() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add high salience entity (proper noun with high initial salience)
    let high_entity = create_entity("CentralStar", Some(EntityLabel::Person), true, 0.9);
    let high_salience_id = graph
        .add_entity(high_entity)
        .expect("Failed to add high salience entity");

    // Boost salience with mentions
    for _ in 0..50 {
        let boost = create_entity("CentralStar", Some(EntityLabel::Person), true, 0.9);
        graph.add_entity(boost).expect("Failed to increment");
    }

    // Add low salience entity
    let low_entity = create_entity("peripheralstar", None, false, 0.2);
    graph
        .add_entity(low_entity)
        .expect("Failed to add low salience entity");

    let universe = graph.get_universe().expect("Failed to get universe");

    // Find the stars
    let high_star = universe
        .stars
        .iter()
        .find(|s| s.name == "CentralStar")
        .expect("Should find CentralStar");
    let low_star = universe
        .stars
        .iter()
        .find(|s| s.name == "peripheralstar")
        .expect("Should find peripheralstar");

    // High salience should be closer to center (0,0,0)
    let high_distance = (high_star.position.x.powi(2)
        + high_star.position.y.powi(2)
        + high_star.position.z.powi(2))
    .sqrt();
    let low_distance =
        (low_star.position.x.powi(2) + low_star.position.y.powi(2) + low_star.position.z.powi(2))
            .sqrt();

    assert!(
        high_distance < low_distance,
        "High salience star ({:.2}) should be closer to center than low salience ({:.2})",
        high_distance,
        low_distance
    );
}

#[test]
fn test_star_positions_3d() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add multiple entities
    for i in 0..10 {
        let entity = create_entity(&format!("Star_{}", i), None, i % 2 == 0, 0.5);
        graph.add_entity(entity).expect("Failed to add star");
    }

    let universe = graph.get_universe().expect("Failed to get universe");

    // Verify all stars have valid 3D positions
    for star in &universe.stars {
        assert!(
            star.position.x.is_finite(),
            "X position should be finite for {}",
            star.name
        );
        assert!(
            star.position.y.is_finite(),
            "Y position should be finite for {}",
            star.name
        );
        assert!(
            star.position.z.is_finite(),
            "Z position should be finite for {}",
            star.name
        );
    }
}

#[test]
fn test_spiral_galaxy_layout() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add many entities to see spiral pattern
    for i in 0..20 {
        let is_proper = i < 5; // First 5 are proper nouns
        let entity = create_entity(&format!("GalaxyStar_{}", i), None, is_proper, 0.5);
        graph.add_entity(entity).expect("Failed to add star");
    }

    let universe = graph.get_universe().expect("Failed to get universe");

    // Verify we have a spread of positions
    let mut x_coords: Vec<f32> = universe.stars.iter().map(|s| s.position.x).collect();
    let mut y_coords: Vec<f32> = universe.stars.iter().map(|s| s.position.y).collect();

    x_coords.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_coords.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let x_spread = x_coords.last().unwrap() - x_coords.first().unwrap();
    let y_spread = y_coords.last().unwrap() - y_coords.first().unwrap();

    // Should have some spread in both dimensions (not all clustered)
    assert!(
        x_spread > 1.0,
        "Stars should spread across X axis (spread: {})",
        x_spread
    );
    assert!(
        y_spread > 1.0,
        "Stars should spread across Y axis (spread: {})",
        y_spread
    );
}

// =============================================================================
// STAR SIZE TESTS
// =============================================================================

#[test]
fn test_star_size_based_on_salience() {
    let (graph, _temp_dir) = setup_graph_memory();

    // High salience star
    let high_entity = create_entity("BigStar", Some(EntityLabel::Person), true, 0.9);
    let high_id = graph.add_entity(high_entity).expect("Failed");

    for _ in 0..30 {
        let boost = create_entity("BigStar", Some(EntityLabel::Person), true, 0.9);
        graph.add_entity(boost).expect("Failed");
    }

    // Low salience star
    let low_entity = create_entity("smallstar", None, false, 0.2);
    graph.add_entity(low_entity).expect("Failed");

    let universe = graph.get_universe().expect("Failed to get universe");

    let big_star = universe
        .stars
        .iter()
        .find(|s| s.name == "BigStar")
        .expect("Should find BigStar");
    let small_star = universe
        .stars
        .iter()
        .find(|s| s.name == "smallstar")
        .expect("Should find smallstar");

    // Star size formula: 5.0 + salience * 20.0
    // Higher salience = bigger star
    assert!(
        big_star.size > small_star.size,
        "High salience star should be larger ({} > {})",
        big_star.size,
        small_star.size
    );

    // Verify size range (should be between 5 and 25)
    assert!(
        big_star.size >= 5.0 && big_star.size <= 25.0,
        "Star size should be in range [5, 25], got {}",
        big_star.size
    );
    assert!(
        small_star.size >= 5.0 && small_star.size <= 25.0,
        "Star size should be in range [5, 25], got {}",
        small_star.size
    );
}

// =============================================================================
// COLOR TESTS (ENTITY TYPE)
// =============================================================================

#[test]
fn test_person_color() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("PersonEntity", Some(EntityLabel::Person), true, 0.6);
    graph.add_entity(entity).expect("Failed");

    let universe = graph.get_universe().expect("Failed");

    let star = universe
        .stars
        .iter()
        .find(|s| s.name == "PersonEntity")
        .expect("Should find entity");

    assert_eq!(star.color, "#FF6B6B", "Person should have coral red color");
}

#[test]
fn test_organization_color() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("OrgEntity", Some(EntityLabel::Organization), true, 0.6);
    graph.add_entity(entity).expect("Failed");

    let universe = graph.get_universe().expect("Failed");

    let star = universe
        .stars
        .iter()
        .find(|s| s.name == "OrgEntity")
        .expect("Should find entity");

    assert_eq!(star.color, "#4ECDC4", "Organization should have teal color");
}

#[test]
fn test_location_color() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("LocationEntity", Some(EntityLabel::Location), true, 0.6);
    graph.add_entity(entity).expect("Failed");

    let universe = graph.get_universe().expect("Failed");

    let star = universe
        .stars
        .iter()
        .find(|s| s.name == "LocationEntity")
        .expect("Should find entity");

    assert_eq!(star.color, "#45B7D1", "Location should have sky blue color");
}

#[test]
fn test_technology_color() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("TechEntity", Some(EntityLabel::Technology), true, 0.6);
    graph.add_entity(entity).expect("Failed");

    let universe = graph.get_universe().expect("Failed");

    let star = universe
        .stars
        .iter()
        .find(|s| s.name == "TechEntity")
        .expect("Should find entity");

    assert_eq!(
        star.color, "#96CEB4",
        "Technology should have sage green color"
    );
}

#[test]
fn test_unlabeled_entity_color() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("UnlabeledEntity", None, false, 0.3);
    graph.add_entity(entity).expect("Failed");

    let universe = graph.get_universe().expect("Failed");

    let star = universe
        .stars
        .iter()
        .find(|s| s.name == "UnlabeledEntity")
        .expect("Should find entity");

    assert_eq!(
        star.color, "#AEB6BF",
        "Unlabeled entity should have gray color"
    );
}

// =============================================================================
// CONNECTION TESTS
// =============================================================================

#[test]
fn test_connection_positions() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("Entity1", None, false, 0.5);
    let entity2 = create_entity("Entity2", None, false, 0.5);

    let e1 = graph.add_entity(entity1).expect("Failed");
    let e2 = graph.add_entity(entity2).expect("Failed");

    graph
        .add_relationship(create_relationship(e1, e2, RelationType::Knows, 0.8))
        .expect("Failed to add relationship");

    let universe = graph.get_universe().expect("Failed to get universe");

    assert_eq!(universe.connections.len(), 1, "Should have 1 connection");

    let connection = &universe.connections[0];

    // Verify from/to positions match star positions
    let star1 = universe
        .stars
        .iter()
        .find(|s| s.id == connection.from_id)
        .expect("Should find source star");
    let star2 = universe
        .stars
        .iter()
        .find(|s| s.id == connection.to_id)
        .expect("Should find target star");

    assert!(
        (connection.from_position.x - star1.position.x).abs() < 0.01,
        "From position X should match star position"
    );
    assert!(
        (connection.to_position.x - star2.position.x).abs() < 0.01,
        "To position X should match star position"
    );
}

#[test]
fn test_connection_strength() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("A", None, false, 0.5);
    let entity2 = create_entity("B", None, false, 0.5);

    let e1 = graph.add_entity(entity1).expect("Failed");
    let e2 = graph.add_entity(entity2).expect("Failed");

    graph
        .add_relationship(create_relationship(e1, e2, RelationType::Knows, 0.75))
        .expect("Failed");

    let universe = graph.get_universe().expect("Failed");

    assert!(
        (universe.connections[0].strength - 0.75).abs() < 0.01,
        "Connection strength should be 0.75"
    );
}

// =============================================================================
// BOUNDS TESTS
// =============================================================================

#[test]
fn test_universe_bounds() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add entities that will spread out
    for i in 0..15 {
        let entity = create_entity(&format!("BoundedStar_{}", i), None, i % 3 == 0, 0.5);
        graph.add_entity(entity).expect("Failed");
    }

    let universe = graph.get_universe().expect("Failed");

    // Verify bounds contain all stars
    for star in &universe.stars {
        assert!(
            star.position.x >= universe.bounds.min.x && star.position.x <= universe.bounds.max.x,
            "Star X ({}) should be within bounds [{}, {}]",
            star.position.x,
            universe.bounds.min.x,
            universe.bounds.max.x
        );
        assert!(
            star.position.y >= universe.bounds.min.y && star.position.y <= universe.bounds.max.y,
            "Star Y ({}) should be within bounds [{}, {}]",
            star.position.y,
            universe.bounds.min.y,
            universe.bounds.max.y
        );
        assert!(
            star.position.z >= universe.bounds.min.z && star.position.z <= universe.bounds.max.z,
            "Star Z ({}) should be within bounds [{}, {}]",
            star.position.z,
            universe.bounds.min.z,
            universe.bounds.max.z
        );
    }
}

// =============================================================================
// METADATA TESTS
// =============================================================================

#[test]
fn test_star_metadata() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("MetadataStar", Some(EntityLabel::Person), true, 0.6);
    let entity_id = graph.add_entity(entity).expect("Failed");

    // Add some mentions
    for _ in 0..5 {
        let boost = create_entity("MetadataStar", Some(EntityLabel::Person), true, 0.6);
        graph.add_entity(boost).expect("Failed");
    }

    let universe = graph.get_universe().expect("Failed");

    let star = universe
        .stars
        .iter()
        .find(|s| s.name == "MetadataStar")
        .expect("Should find star");

    // Verify metadata is populated
    assert!(!star.id.is_empty(), "Star should have ID");
    assert_eq!(star.name, "MetadataStar");
    assert_eq!(
        star.entity_type,
        Some("Person".to_string()),
        "Entity type should be Person"
    );
    assert!(star.is_proper_noun, "Should be marked as proper noun");
    assert!(star.salience > 0.0, "Should have positive salience");
    assert!(star.mention_count >= 6, "Should have at least 6 mentions");
}

// =============================================================================
// LARGE GRAPH TESTS
// =============================================================================

#[test]
fn test_large_universe_performance() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add 100 entities with relationships
    let mut entity_ids = Vec::new();
    for i in 0..100 {
        let entity = create_entity(&format!("LargeEntity_{}", i), None, i % 5 == 0, 0.5);
        let id = graph.add_entity(entity).expect("Failed to add entity");
        entity_ids.push(id);
    }

    // Add relationships (sparse graph)
    for i in 0..50 {
        let from_idx = i * 2;
        let to_idx = (i * 2 + 1) % 100;
        graph
            .add_relationship(create_relationship(
                entity_ids[from_idx],
                entity_ids[to_idx],
                RelationType::RelatedTo,
                0.5,
            ))
            .expect("Failed to add relationship");
    }

    // Time the universe generation
    let start = std::time::Instant::now();
    let universe = graph.get_universe().expect("Failed to get universe");
    let elapsed = start.elapsed();

    assert_eq!(universe.stars.len(), 100, "Should have 100 stars");
    assert_eq!(universe.connections.len(), 50, "Should have 50 connections");

    // Universe generation should be fast (< 100ms even for 100 entities)
    assert!(
        elapsed.as_millis() < 100,
        "Universe generation should be fast, took {}ms",
        elapsed.as_millis()
    );
}
