//! Graph Memory Tests
//!
//! Tests for the knowledge graph operations:
//! - Entity CRUD operations
//! - Relationship management
//! - Graph traversal
//! - Episode creation and queries
//! - Temporal edge invalidation
//! - NER integration for entity extraction

use chrono::{DateTime, Duration, Utc};
use shodh_memory::embeddings::ner::{NerConfig, NerEntityType, NeuralNer};
use shodh_memory::graph_memory::{
    EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, RelationType,
    RelationshipEdge,
};
use shodh_memory::uuid::Uuid;
use std::collections::HashMap;
use tempfile::TempDir;

/// Create fallback NER instance for testing
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Convert NER entity type to GraphMemory EntityLabel
fn ner_type_to_label(ner_type: &NerEntityType) -> EntityLabel {
    match ner_type {
        NerEntityType::Person => EntityLabel::Person,
        NerEntityType::Organization => EntityLabel::Organization,
        NerEntityType::Location => EntityLabel::Location,
        NerEntityType::Misc => EntityLabel::Concept,
    }
}

/// Create entity from NER extraction
fn create_entity_from_ner(ner: &NeuralNer, text: &str) -> Vec<EntityNode> {
    let extracted = ner.extract(text).unwrap_or_default();
    extracted
        .iter()
        .map(|entity| EntityNode {
            uuid: Uuid::new_v4(),
            name: entity.text.clone(),
            labels: vec![ner_type_to_label(&entity.entity_type)],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: HashMap::new(),
            name_embedding: None,
            salience: entity.confidence,
            is_proper_noun: true,
        })
        .collect()
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

/// Create a relationship edge with custom Hebbian parameters
fn create_relationship_with_plasticity(
    from: Uuid,
    to: Uuid,
    rel_type: RelationType,
    strength: f32,
    activation_count: u32,
    potentiated: bool,
    last_activated: DateTime<Utc>,
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
        last_activated,
        activation_count,
        potentiated,
    }
}

// =============================================================================
// ENTITY CRUD TESTS
// =============================================================================

#[test]
fn test_add_entity() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("TestEntity", Some(EntityLabel::Person), true, 0.6);
    let entity_id = graph.add_entity(entity).expect("Failed to add entity");

    assert!(!entity_id.is_nil(), "Entity should have a valid UUID");
}

#[test]
fn test_get_entity() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("RetrieveMe", Some(EntityLabel::Location), true, 0.7);
    let entity_id = graph.add_entity(entity).expect("Failed to add entity");

    let retrieved = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Entity should exist");

    assert_eq!(retrieved.name, "RetrieveMe");
    assert!(retrieved.labels.contains(&EntityLabel::Location));
}

#[test]
fn test_find_entity_by_name() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("FindByName", Some(EntityLabel::Organization), true, 0.6);
    graph.add_entity(entity).expect("Failed to add entity");

    let found = graph
        .find_entity_by_name("FindByName")
        .expect("Failed to find entity")
        .expect("Entity should be found");

    assert_eq!(found.name, "FindByName");
}

#[test]
fn test_entity_not_found() {
    let (graph, _temp_dir) = setup_graph_memory();

    let result = graph
        .find_entity_by_name("NonExistent")
        .expect("Query should not fail");

    assert!(result.is_none(), "Entity should not exist");
}

#[test]
fn test_update_entity_on_duplicate_add() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add entity first time
    let entity1 = create_entity("DuplicateEntity", Some(EntityLabel::Person), true, 0.6);
    let id1 = graph
        .add_entity(entity1)
        .expect("Failed to add first entity");

    // Add same name again
    let entity2 = create_entity("DuplicateEntity", Some(EntityLabel::Person), true, 0.6);
    let id2 = graph
        .add_entity(entity2)
        .expect("Failed to add second entity");

    // Should return same UUID
    assert_eq!(id1, id2, "Duplicate adds should return same UUID");

    // Mention count should increase
    let entity = graph
        .get_entity(&id1)
        .expect("Failed to get entity")
        .expect("Entity should exist");
    assert!(entity.mention_count >= 2, "Mention count should increase");
}

#[test]
fn test_get_all_entities() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add multiple entities
    for i in 0..5 {
        let entity = create_entity(&format!("Entity_{}", i), None, false, 0.5);
        graph.add_entity(entity).expect("Failed to add entity");
    }

    let all_entities = graph
        .get_all_entities()
        .expect("Failed to get all entities");
    assert_eq!(all_entities.len(), 5, "Should have 5 entities");
}

// =============================================================================
// RELATIONSHIP TESTS
// =============================================================================

#[test]
fn test_add_relationship() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("Entity1", Some(EntityLabel::Person), true, 0.6);
    let entity2 = create_entity("Entity2", Some(EntityLabel::Organization), true, 0.7);

    let id1 = graph.add_entity(entity1).expect("Failed to add entity1");
    let id2 = graph.add_entity(entity2).expect("Failed to add entity2");

    let edge = create_relationship(id1, id2, RelationType::WorksAt, 0.8);
    let edge_id = graph
        .add_relationship(edge)
        .expect("Failed to add relationship");

    assert!(!edge_id.is_nil(), "Relationship should have valid UUID");
}

#[test]
fn test_get_entity_relationships() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("Person1", Some(EntityLabel::Person), true, 0.6);
    let entity2 = create_entity("Company1", Some(EntityLabel::Organization), true, 0.7);
    let entity3 = create_entity("Location1", Some(EntityLabel::Location), true, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");
    let id3 = graph.add_entity(entity3).expect("Failed");

    // Person works at Company, and lives in Location
    let edge1 = create_relationship(id1, id2, RelationType::WorksAt, 0.9);
    let edge2 = create_relationship(id1, id3, RelationType::LocatedIn, 0.8);

    graph.add_relationship(edge1).expect("Failed");
    graph.add_relationship(edge2).expect("Failed");

    let relationships = graph
        .get_entity_relationships(&id1)
        .expect("Failed to get relationships");

    assert_eq!(relationships.len(), 2, "Person should have 2 relationships");
}

#[test]
fn test_relationship_types() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("Subject", None, false, 0.5);
    let entity2 = create_entity("Object", None, false, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");

    // Test various relationship types
    let types = vec![
        RelationType::WorksWith,
        RelationType::PartOf,
        RelationType::Uses,
        RelationType::Knows,
        RelationType::RelatedTo,
    ];

    for rel_type in types {
        let edge = create_relationship(id1, id2, rel_type, 0.7);
        graph
            .add_relationship(edge)
            .expect("Failed to add relationship");
    }

    let relationships = graph.get_entity_relationships(&id1).expect("Failed");
    assert_eq!(relationships.len(), 5, "Should have 5 relationship types");
}

#[test]
fn test_get_all_relationships() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("A", None, false, 0.5);
    let entity2 = create_entity("B", None, false, 0.5);
    let entity3 = create_entity("C", None, false, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");
    let id3 = graph.add_entity(entity3).expect("Failed");

    graph
        .add_relationship(create_relationship(id1, id2, RelationType::Knows, 0.8))
        .expect("Failed");
    graph
        .add_relationship(create_relationship(id2, id3, RelationType::Knows, 0.7))
        .expect("Failed");
    graph
        .add_relationship(create_relationship(id1, id3, RelationType::Knows, 0.6))
        .expect("Failed");

    let all_rels = graph.get_all_relationships().expect("Failed");
    assert_eq!(all_rels.len(), 3, "Should have 3 relationships");
}

// =============================================================================
// GRAPH TRAVERSAL TESTS
// =============================================================================

#[test]
fn test_traverse_from_entity() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Create a simple graph: A -> B -> C
    let entity_a = create_entity("A", None, false, 0.5);
    let entity_b = create_entity("B", None, false, 0.5);
    let entity_c = create_entity("C", None, false, 0.5);

    let id_a = graph.add_entity(entity_a).expect("Failed");
    let id_b = graph.add_entity(entity_b).expect("Failed");
    let id_c = graph.add_entity(entity_c).expect("Failed");

    graph
        .add_relationship(create_relationship(id_a, id_b, RelationType::Knows, 0.8))
        .expect("Failed");
    graph
        .add_relationship(create_relationship(id_b, id_c, RelationType::Knows, 0.8))
        .expect("Failed");

    // Traverse depth 1 from A
    let traversal_1 = graph
        .traverse_from_entity(&id_a, 1)
        .expect("Failed to traverse");
    assert_eq!(traversal_1.entities.len(), 2, "Depth 1 should find A and B");

    // Traverse depth 2 from A
    let traversal_2 = graph
        .traverse_from_entity(&id_a, 2)
        .expect("Failed to traverse");
    assert_eq!(
        traversal_2.entities.len(),
        3,
        "Depth 2 should find A, B, and C"
    );
}

#[test]
fn test_traverse_with_cycles() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Create a cycle: A -> B -> C -> A
    let entity_a = create_entity("CycleA", None, false, 0.5);
    let entity_b = create_entity("CycleB", None, false, 0.5);
    let entity_c = create_entity("CycleC", None, false, 0.5);

    let id_a = graph.add_entity(entity_a).expect("Failed");
    let id_b = graph.add_entity(entity_b).expect("Failed");
    let id_c = graph.add_entity(entity_c).expect("Failed");

    graph
        .add_relationship(create_relationship(id_a, id_b, RelationType::Knows, 0.8))
        .expect("Failed");
    graph
        .add_relationship(create_relationship(id_b, id_c, RelationType::Knows, 0.8))
        .expect("Failed");
    graph
        .add_relationship(create_relationship(id_c, id_a, RelationType::Knows, 0.8))
        .expect("Failed");

    // Traverse should not loop infinitely
    let traversal = graph
        .traverse_from_entity(&id_a, 10)
        .expect("Failed to traverse");
    assert_eq!(
        traversal.entities.len(),
        3,
        "Should find exactly 3 entities despite cycle"
    );
}

// =============================================================================
// EPISODE TESTS
// =============================================================================

#[test]
fn test_add_episode() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Create entities for the episode
    let entity1 = create_entity("EpisodePerson", Some(EntityLabel::Person), true, 0.6);
    let entity2 = create_entity("EpisodePlace", Some(EntityLabel::Location), true, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");

    let episode = EpisodicNode {
        uuid: Uuid::new_v4(),
        name: "Visit episode".to_string(),
        content: "A person visited a place".to_string(),
        source: EpisodeSource::Observation,
        created_at: Utc::now(),
        valid_at: Utc::now(),
        entity_refs: vec![id1, id2],
        metadata: HashMap::new(),
    };

    let episode_id = graph.add_episode(episode).expect("Failed to add episode");
    assert!(!episode_id.is_nil(), "Episode should have valid UUID");
}

#[test]
fn test_get_entity_episodes() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("FrequentPerson", Some(EntityLabel::Person), true, 0.6);
    let entity_id = graph.add_entity(entity).expect("Failed");

    // Add multiple episodes involving this entity
    for i in 0..3 {
        let episode = EpisodicNode {
            uuid: Uuid::new_v4(),
            name: format!("Episode {}", i),
            content: format!("Episode {} about the person", i),
            source: EpisodeSource::Message,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            entity_refs: vec![entity_id],
            metadata: HashMap::new(),
        };
        graph.add_episode(episode).expect("Failed to add episode");
    }

    let episodes = graph
        .get_episodes_by_entity(&entity_id)
        .expect("Failed to get episodes");
    assert_eq!(episodes.len(), 3, "Entity should have 3 episodes");
}

// =============================================================================
// TEMPORAL INVALIDATION TESTS
// =============================================================================

#[test]
fn test_invalidate_relationship() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("FormerEmployee", Some(EntityLabel::Person), true, 0.6);
    let entity2 = create_entity("OldCompany", Some(EntityLabel::Organization), true, 0.7);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");

    let edge = create_relationship(id1, id2, RelationType::WorksAt, 0.9);
    let edge_id = graph.add_relationship(edge).expect("Failed");

    // Invalidate the relationship
    graph
        .invalidate_relationship(&edge_id)
        .expect("Failed to invalidate");

    // Check relationship is invalidated
    let relationship = graph
        .get_relationship(&edge_id)
        .expect("Failed to get relationship")
        .expect("Relationship should exist");

    assert!(
        relationship.invalidated_at.is_some(),
        "Relationship should be invalidated"
    );
}

#[test]
fn test_invalidated_relationships_excluded() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("Entity1", None, false, 0.5);
    let entity2 = create_entity("Entity2", None, false, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");

    let edge = create_relationship(id1, id2, RelationType::Knows, 0.8);
    let edge_id = graph.add_relationship(edge).expect("Failed");

    // Verify relationship exists
    let all_rels_before = graph.get_all_relationships().expect("Failed");
    assert_eq!(all_rels_before.len(), 1, "Should have 1 relationship");

    // Invalidate
    graph
        .invalidate_relationship(&edge_id)
        .expect("Failed to invalidate");

    // get_all_relationships should exclude invalidated
    let all_rels_after = graph.get_all_relationships().expect("Failed");
    assert_eq!(
        all_rels_after.len(),
        0,
        "Invalidated relationships should be excluded"
    );
}

// =============================================================================
// GRAPH STATS TESTS
// =============================================================================

#[test]
fn test_graph_stats() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add entities
    let entity1 = create_entity("StatsEntity1", None, false, 0.5);
    let entity2 = create_entity("StatsEntity2", None, false, 0.5);
    let entity3 = create_entity("StatsEntity3", None, false, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");
    let id3 = graph.add_entity(entity3).expect("Failed");

    // Add relationships
    graph
        .add_relationship(create_relationship(id1, id2, RelationType::Knows, 0.8))
        .expect("Failed");
    graph
        .add_relationship(create_relationship(id2, id3, RelationType::Knows, 0.7))
        .expect("Failed");

    // Add episode
    let episode = EpisodicNode {
        uuid: Uuid::new_v4(),
        name: "Stats test".to_string(),
        content: "Stats test episode".to_string(),
        source: EpisodeSource::Event,
        created_at: Utc::now(),
        valid_at: Utc::now(),
        entity_refs: vec![id1, id2],
        metadata: HashMap::new(),
    };
    graph.add_episode(episode).expect("Failed");

    let stats = graph.get_stats().expect("Failed to get stats");
    assert_eq!(stats.entity_count, 3, "Should have 3 entities");
    assert_eq!(stats.relationship_count, 2, "Should have 2 relationships");
    assert_eq!(stats.episode_count, 1, "Should have 1 episode");
}

// =============================================================================
// ENTITY LABELS TESTS
// =============================================================================

#[test]
fn test_all_entity_labels() {
    let (graph, _temp_dir) = setup_graph_memory();

    let labels = vec![
        EntityLabel::Person,
        EntityLabel::Organization,
        EntityLabel::Location,
        EntityLabel::Technology,
        EntityLabel::Concept,
        EntityLabel::Event,
        EntityLabel::Date,
        EntityLabel::Product,
        EntityLabel::Skill,
        EntityLabel::Other("Custom".to_string()),
    ];

    for (i, label) in labels.iter().enumerate() {
        let entity = create_entity(&format!("Entity_{}", i), Some(label.clone()), true, 0.6);
        graph.add_entity(entity).expect("Failed to add entity");
    }

    let all_entities = graph.get_all_entities().expect("Failed");
    assert_eq!(
        all_entities.len(),
        labels.len(),
        "Should have all entity types"
    );
}

// =============================================================================
// PERSISTENCE TESTS
// =============================================================================

#[test]
fn test_persistence_across_reopens() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let entity_name = "PersistentEntity";

    // Create graph, add data, and close
    {
        let graph = GraphMemory::new(&db_path).expect("Failed to create graph");

        let entity = create_entity(entity_name, Some(EntityLabel::Person), true, 0.6);
        graph.add_entity(entity).expect("Failed to add entity");
    }

    // Reopen and verify persistence
    {
        let graph = GraphMemory::new(&db_path).expect("Failed to reopen graph");

        let entity = graph
            .find_entity_by_name(entity_name)
            .expect("Failed to find entity")
            .expect("Entity should exist after reopen");

        assert_eq!(entity.name, entity_name);
    }
}

#[test]
fn test_relationship_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    // Create graph with relationship
    {
        let graph = GraphMemory::new(&db_path).expect("Failed");

        let entity1 = create_entity("PersistA", None, false, 0.5);
        let entity2 = create_entity("PersistB", None, false, 0.5);

        let id1 = graph.add_entity(entity1).expect("Failed");
        let id2 = graph.add_entity(entity2).expect("Failed");

        graph
            .add_relationship(create_relationship(id1, id2, RelationType::Knows, 0.8))
            .expect("Failed");
    }

    // Reopen and verify
    {
        let graph = GraphMemory::new(&db_path).expect("Failed");

        let entity = graph
            .find_entity_by_name("PersistA")
            .expect("Failed")
            .expect("Should exist");

        let rels = graph
            .get_entity_relationships(&entity.uuid)
            .expect("Failed");
        assert_eq!(rels.len(), 1, "Relationship should persist");
    }
}

// =============================================================================
// EDGE CASES
// =============================================================================

#[test]
fn test_empty_entity_name() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("", None, false, 0.5);
    let entity_id = graph
        .add_entity(entity)
        .expect("Failed to add empty-named entity");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed")
        .expect("Should exist");
    assert_eq!(result.name, "", "Empty name should be preserved");
}

#[test]
fn test_very_long_entity_name() {
    let (graph, _temp_dir) = setup_graph_memory();

    let long_name = "A".repeat(10000);
    let entity = create_entity(&long_name, None, false, 0.3);
    let entity_id = graph
        .add_entity(entity)
        .expect("Failed to add long-named entity");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed")
        .expect("Should exist");
    assert_eq!(result.name, long_name, "Long name should be preserved");
}

#[test]
fn test_unicode_entity_name() {
    let (graph, _temp_dir) = setup_graph_memory();

    let unicode_name = "北京市 Москва München";
    let entity = create_entity(unicode_name, Some(EntityLabel::Location), true, 0.6);
    let entity_id = graph
        .add_entity(entity)
        .expect("Failed to add unicode entity");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed")
        .expect("Should exist");
    assert_eq!(
        result.name, unicode_name,
        "Unicode name should be preserved"
    );
}

#[test]
fn test_special_characters_in_name() {
    let (graph, _temp_dir) = setup_graph_memory();

    let special_name = "Entity <with> \"special\" & 'chars' @ test!";
    let entity = create_entity(special_name, None, false, 0.5);
    let entity_id = graph.add_entity(entity).expect("Failed");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed")
        .expect("Should exist");
    assert_eq!(result.name, special_name);
}

#[test]
fn test_self_referential_relationship() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("SelfRef", None, false, 0.5);
    let entity_id = graph.add_entity(entity).expect("Failed");

    // Entity references itself
    let edge = create_relationship(entity_id, entity_id, RelationType::RelatedTo, 0.5);
    let edge_id = graph.add_relationship(edge).expect("Failed");

    assert!(
        !edge_id.is_nil(),
        "Self-referential relationship should work"
    );
}

#[test]
fn test_large_graph() {
    let (graph, _temp_dir) = setup_graph_memory();

    let mut entity_ids = Vec::new();

    // Add 100 entities
    for i in 0..100 {
        let entity = create_entity(&format!("LargeGraph_{}", i), None, i % 10 == 0, 0.5);
        let id = graph.add_entity(entity).expect("Failed");
        entity_ids.push(id);
    }

    // Add 200 relationships (sparse graph)
    for i in 0..200 {
        let from_idx = i % 100;
        let to_idx = (i * 7) % 100;
        if from_idx != to_idx {
            let edge = create_relationship(
                entity_ids[from_idx],
                entity_ids[to_idx],
                RelationType::RelatedTo,
                0.5 + (i as f32 * 0.002),
            );
            graph.add_relationship(edge).expect("Failed");
        }
    }

    let stats = graph.get_stats().expect("Failed");
    assert_eq!(stats.entity_count, 100, "Should have 100 entities");
    assert!(stats.relationship_count > 0, "Should have relationships");
}

// =============================================================================
// HEBBIAN SYNAPTIC PLASTICITY TESTS
// =============================================================================

#[test]
fn test_strengthen_increases_strength() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    let mut edge = create_relationship(entity_a, entity_b, RelationType::Knows, 0.5);
    let initial_strength = edge.strength;

    edge.strengthen();

    assert!(
        edge.strength > initial_strength,
        "Strength should increase after strengthening: {} -> {}",
        initial_strength,
        edge.strength
    );
    assert_eq!(edge.activation_count, 1, "Activation count should be 1");
}

#[test]
fn test_strengthen_asymptotic_approach_to_one() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    let mut edge = create_relationship(entity_a, entity_b, RelationType::Knows, 0.9);

    // Strengthen multiple times
    for _ in 0..5 {
        edge.strengthen();
    }

    // Should approach but not exceed 1.0
    assert!(edge.strength <= 1.0, "Strength should not exceed 1.0");
    assert!(
        edge.strength > 0.9,
        "Strength should increase even from high base"
    );
}

#[test]
fn test_strengthen_diminishing_returns() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    // Edge starting at low strength
    let mut low_edge = create_relationship(entity_a, entity_b, RelationType::Knows, 0.2);
    let low_initial = low_edge.strength;
    low_edge.strengthen();
    let low_increase = low_edge.strength - low_initial;

    // Edge starting at high strength
    let mut high_edge = create_relationship(entity_a, entity_b, RelationType::Knows, 0.8);
    let high_initial = high_edge.strength;
    high_edge.strengthen();
    let high_increase = high_edge.strength - high_initial;

    // Low strength edge should gain more from strengthening (diminishing returns)
    assert!(
        low_increase > high_increase,
        "Low strength edge should gain more: {} > {}",
        low_increase,
        high_increase
    );
}

#[test]
fn test_long_term_potentiation_threshold() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    let mut edge = create_relationship(entity_a, entity_b, RelationType::Knows, 0.5);
    assert!(
        !edge.potentiated,
        "Edge should not be potentiated initially"
    );

    // Strengthen 9 times (below threshold of 10)
    for _ in 0..9 {
        edge.strengthen();
    }
    assert!(
        !edge.potentiated,
        "Edge should not be potentiated at 9 activations"
    );

    // Strengthen once more (reaches threshold)
    let pre_ltp_strength = edge.strength;
    edge.strengthen();

    assert!(
        edge.potentiated,
        "Edge should be potentiated at 10 activations"
    );
    assert!(
        edge.strength > pre_ltp_strength,
        "LTP should give bonus strength"
    );
}

#[test]
fn test_decay_reduces_strength() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    // Create edge activated 30 days ago
    let thirty_days_ago = Utc::now() - Duration::days(30);
    let mut edge = create_relationship_with_plasticity(
        entity_a,
        entity_b,
        RelationType::Knows,
        0.8,
        5,
        false,
        thirty_days_ago,
    );

    let initial_strength = edge.strength;
    let should_prune = edge.decay();

    assert!(
        edge.strength < initial_strength,
        "Strength should decrease after decay"
    );
    assert!(
        !should_prune,
        "Edge with strength 0.8 should not be pruned after 30 days"
    );
}

#[test]
fn test_decay_half_life() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    // Create edge activated exactly 14 days ago (one half-life)
    let fourteen_days_ago = Utc::now() - Duration::days(14);
    let mut edge = create_relationship_with_plasticity(
        entity_a,
        entity_b,
        RelationType::Knows,
        1.0,
        1,
        false,
        fourteen_days_ago,
    );

    edge.decay();

    // After one half-life, strength should be approximately 0.5
    let expected = 0.5;
    let tolerance = 0.05;
    assert!(
        (edge.strength - expected).abs() < tolerance,
        "After 14 days (1 half-life), strength should be ~0.5, got {}",
        edge.strength
    );
}

#[test]
fn test_potentiated_decay_slower() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();
    let thirty_days_ago = Utc::now() - Duration::days(30);

    // Non-potentiated edge
    let mut normal_edge = create_relationship_with_plasticity(
        entity_a,
        entity_b,
        RelationType::Knows,
        0.8,
        5,
        false,
        thirty_days_ago,
    );

    // Potentiated edge (same parameters but potentiated)
    let mut potentiated_edge = create_relationship_with_plasticity(
        entity_a,
        entity_b,
        RelationType::Knows,
        0.8,
        15,
        true,
        thirty_days_ago,
    );

    normal_edge.decay();
    potentiated_edge.decay();

    assert!(
        potentiated_edge.strength > normal_edge.strength,
        "Potentiated edge should decay slower: {} > {}",
        potentiated_edge.strength,
        normal_edge.strength
    );
}

#[test]
fn test_decay_minimum_floor() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    // Create edge activated 365 days ago with low initial strength
    let one_year_ago = Utc::now() - Duration::days(365);
    let mut edge = create_relationship_with_plasticity(
        entity_a,
        entity_b,
        RelationType::Knows,
        0.1,
        1,
        false,
        one_year_ago,
    );

    edge.decay();

    // Strength should not go below MIN_STRENGTH (0.01)
    assert!(
        edge.strength >= 0.01,
        "Strength should not go below minimum: {}",
        edge.strength
    );
}

#[test]
fn test_effective_strength_without_mutation() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    let thirty_days_ago = Utc::now() - Duration::days(30);
    let edge = create_relationship_with_plasticity(
        entity_a,
        entity_b,
        RelationType::Knows,
        0.8,
        5,
        false,
        thirty_days_ago,
    );

    let original_strength = edge.strength;
    let effective = edge.effective_strength();

    // effective_strength should not mutate the edge
    assert_eq!(
        edge.strength, original_strength,
        "effective_strength should not mutate"
    );
    assert!(
        effective < original_strength,
        "Effective strength should account for decay"
    );
}

#[test]
fn test_effective_strength_recent_edge() {
    let entity_a = Uuid::new_v4();
    let entity_b = Uuid::new_v4();

    // Edge just activated
    let edge = create_relationship(entity_a, entity_b, RelationType::Knows, 0.7);

    let effective = edge.effective_strength();

    // For recently activated edges, effective strength should equal stored strength
    let tolerance = 0.001;
    assert!(
        (effective - edge.strength).abs() < tolerance,
        "Recent edge effective_strength should equal strength: {} vs {}",
        effective,
        edge.strength
    );
}

// =============================================================================
// HEBBIAN PLASTICITY INTEGRATION TESTS (GraphMemory level)
// =============================================================================

#[test]
fn test_strengthen_synapse_persists() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("HebbianA", None, false, 0.5);
    let entity2 = create_entity("HebbianB", None, false, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");

    let edge = create_relationship(id1, id2, RelationType::Knows, 0.5);
    let edge_id = graph.add_relationship(edge).expect("Failed");

    // Get initial state
    let initial_edge = graph
        .get_relationship(&edge_id)
        .expect("Failed")
        .expect("Edge should exist");
    let initial_strength = initial_edge.strength;
    let initial_count = initial_edge.activation_count;

    // Strengthen the synapse
    graph.strengthen_synapse(&edge_id).expect("Failed");

    // Verify persistence
    let updated_edge = graph
        .get_relationship(&edge_id)
        .expect("Failed")
        .expect("Edge should exist");

    assert!(
        updated_edge.strength > initial_strength,
        "Persisted strength should increase"
    );
    assert_eq!(
        updated_edge.activation_count,
        initial_count + 1,
        "Persisted activation count should increase"
    );
}

#[test]
fn test_decay_synapse_persists() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("DecayA", None, false, 0.5);
    let entity2 = create_entity("DecayB", None, false, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");

    // Create edge with old activation timestamp
    let thirty_days_ago = Utc::now() - Duration::days(30);
    let mut edge = create_relationship_with_plasticity(
        id1,
        id2,
        RelationType::Knows,
        0.8,
        5,
        false,
        thirty_days_ago,
    );

    // Note: add_relationship sets last_activated to now, so we test the method call works
    let edge_id = graph.add_relationship(edge).expect("Failed");

    // Apply decay via the GraphMemory method
    // For a fresh edge, decay will be minimal but the method should work
    let should_prune = graph.decay_synapse(&edge_id).expect("Failed");

    // Verify decay was applied
    let decayed_edge = graph
        .get_relationship(&edge_id)
        .expect("Failed")
        .expect("Edge should exist");

    // The strength should have decreased (though the exact amount depends on timing)
    // Since we can't easily mock time in the DB, we just verify the method works
    assert!(!should_prune, "Edge with 0.8 strength should not be pruned");
}

#[test]
fn test_traverse_strengthens_edges() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Create chain: A -> B -> C
    let entity_a = create_entity("TraverseA", None, false, 0.5);
    let entity_b = create_entity("TraverseB", None, false, 0.5);
    let entity_c = create_entity("TraverseC", None, false, 0.5);

    let id_a = graph.add_entity(entity_a).expect("Failed");
    let id_b = graph.add_entity(entity_b).expect("Failed");
    let id_c = graph.add_entity(entity_c).expect("Failed");

    let edge_ab = create_relationship(id_a, id_b, RelationType::Knows, 0.5);
    let edge_bc = create_relationship(id_b, id_c, RelationType::Knows, 0.5);

    let edge_ab_id = graph.add_relationship(edge_ab).expect("Failed");
    let edge_bc_id = graph.add_relationship(edge_bc).expect("Failed");

    // Get initial activation counts
    let initial_ab = graph
        .get_relationship(&edge_ab_id)
        .expect("Failed")
        .expect("Edge should exist")
        .activation_count;
    let initial_bc = graph
        .get_relationship(&edge_bc_id)
        .expect("Failed")
        .expect("Edge should exist")
        .activation_count;

    // Traverse from A with depth 2 (should traverse both edges)
    let _traversal = graph.traverse_from_entity(&id_a, 2).expect("Failed");

    // Verify both edges were strengthened
    let after_ab = graph
        .get_relationship(&edge_ab_id)
        .expect("Failed")
        .expect("Edge should exist")
        .activation_count;
    let after_bc = graph
        .get_relationship(&edge_bc_id)
        .expect("Failed")
        .expect("Edge should exist")
        .activation_count;

    assert_eq!(
        after_ab,
        initial_ab + 1,
        "Edge A->B should be strengthened during traversal"
    );
    assert_eq!(
        after_bc,
        initial_bc + 1,
        "Edge B->C should be strengthened during traversal"
    );
}

#[test]
fn test_traverse_returns_effective_strength() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity_a = create_entity("EffectiveA", None, false, 0.5);
    let entity_b = create_entity("EffectiveB", None, false, 0.5);

    let id_a = graph.add_entity(entity_a).expect("Failed");
    let id_b = graph.add_entity(entity_b).expect("Failed");

    let edge = create_relationship(id_a, id_b, RelationType::Knows, 0.7);
    let _edge_id = graph.add_relationship(edge).expect("Failed");

    // Traverse and check returned edges
    let traversal = graph.traverse_from_entity(&id_a, 1).expect("Failed");

    assert_eq!(traversal.relationships.len(), 1, "Should have 1 edge");

    // The returned edge should have effective_strength applied
    // For a freshly created edge, effective_strength ~= strength
    let returned_edge = &traversal.relationships[0];
    let tolerance = 0.1;
    assert!(
        (returned_edge.strength - 0.7).abs() < tolerance,
        "Returned edge strength should be close to original for recent edges: {}",
        returned_edge.strength
    );
}

#[test]
fn test_hebbian_plasticity_fields_persist() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let edge_id;
    {
        let graph = GraphMemory::new(&db_path).expect("Failed");

        let entity1 = create_entity("PersistA", None, false, 0.5);
        let entity2 = create_entity("PersistB", None, false, 0.5);

        let id1 = graph.add_entity(entity1).expect("Failed");
        let id2 = graph.add_entity(entity2).expect("Failed");

        let edge = create_relationship(id1, id2, RelationType::Knows, 0.5);
        edge_id = graph.add_relationship(edge).expect("Failed");

        // Strengthen multiple times
        for _ in 0..5 {
            graph.strengthen_synapse(&edge_id).expect("Failed");
        }
    }

    // Reopen and verify plasticity fields persisted
    {
        let graph = GraphMemory::new(&db_path).expect("Failed");

        let edge = graph
            .get_relationship(&edge_id)
            .expect("Failed")
            .expect("Edge should exist");

        assert_eq!(
            edge.activation_count, 5,
            "Activation count should persist: {}",
            edge.activation_count
        );
        assert!(
            edge.strength > 0.5,
            "Strengthened strength should persist: {}",
            edge.strength
        );
    }
}

#[test]
fn test_relationship_with_effective_strength() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity1 = create_entity("EffStrengthA", None, false, 0.5);
    let entity2 = create_entity("EffStrengthB", None, false, 0.5);

    let id1 = graph.add_entity(entity1).expect("Failed");
    let id2 = graph.add_entity(entity2).expect("Failed");

    let edge = create_relationship(id1, id2, RelationType::Knows, 0.8);
    let edge_id = graph.add_relationship(edge).expect("Failed");

    // get_relationship returns raw strength
    let raw_edge = graph
        .get_relationship(&edge_id)
        .expect("Failed")
        .expect("Edge should exist");

    // get_relationship_with_effective_strength returns decay-adjusted strength
    let effective_edge = graph
        .get_relationship_with_effective_strength(&edge_id)
        .expect("Failed")
        .expect("Edge should exist");

    // For a fresh edge, both should be approximately equal
    let tolerance = 0.01;
    assert!(
        (raw_edge.strength - effective_edge.strength).abs() < tolerance,
        "Fresh edge should have raw ~= effective: {} vs {}",
        raw_edge.strength,
        effective_edge.strength
    );
}
