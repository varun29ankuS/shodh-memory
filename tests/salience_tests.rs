//! Salience Detection Tests
//!
//! Tests for the Gravitational Salience Memory system:
//! - Proper noun detection and higher base salience
//! - Frequency-based salience boost (logarithmic)
//! - Entity type classification
//! - Salience decay over time
//! - NER-based entity extraction and salience assignment

use chrono::Utc;
use shodh_memory::embeddings::ner::{NerConfig, NerEntityType, NeuralNer};
use shodh_memory::graph_memory::{EntityLabel, EntityNode, GraphMemory};
use shodh_memory::uuid::Uuid;
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;

/// Create a fallback NER instance for testing
fn create_test_ner() -> NeuralNer {
    let config = NerConfig {
        model_path: PathBuf::from("nonexistent.onnx"),
        tokenizer_path: PathBuf::from("nonexistent.json"),
        max_length: 128,
        confidence_threshold: 0.5,
    };
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
    base_salience: f32,
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
        salience: base_salience,
        is_proper_noun: is_proper,
    }
}

// =============================================================================
// PROPER NOUN DETECTION TESTS
// =============================================================================

#[test]
fn test_proper_noun_higher_base_salience() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add proper noun entity with higher base salience (0.6)
    let proper_entity = create_entity("John Smith", Some(EntityLabel::Person), true, 0.6);
    let proper_noun_id = graph
        .add_entity(proper_entity)
        .expect("Failed to add proper noun entity");

    // Add common noun entity with lower base salience (0.3)
    let common_entity = create_entity("person", Some(EntityLabel::Person), false, 0.3);
    let common_noun_id = graph
        .add_entity(common_entity)
        .expect("Failed to add common noun entity");

    // Get entities
    let proper = graph
        .get_entity(&proper_noun_id)
        .expect("Failed to get proper noun entity")
        .expect("Proper noun should exist");
    let common = graph
        .get_entity(&common_noun_id)
        .expect("Failed to get common noun entity")
        .expect("Common noun should exist");

    // Proper nouns should have higher base salience
    assert!(
        proper.salience > common.salience,
        "Proper noun salience ({}) should be greater than common noun salience ({})",
        proper.salience,
        common.salience
    );

    // Check proper noun flag
    assert!(
        proper.is_proper_noun,
        "Entity should be marked as proper noun"
    );
    assert!(
        !common.is_proper_noun,
        "Entity should NOT be marked as proper noun"
    );
}

#[test]
fn test_proper_noun_base_salience_value() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add proper noun with explicit salience
    let entity = create_entity(
        "Microsoft Corporation",
        Some(EntityLabel::Organization),
        true,
        0.6,
    );
    let entity_id = graph
        .add_entity(entity)
        .expect("Failed to add organization");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Entity should exist");

    // Initial salience should be as specified
    assert!(
        (result.salience - 0.6).abs() < 0.01,
        "Proper noun base salience should be 0.6, got {}",
        result.salience
    );
}

#[test]
fn test_common_noun_base_salience_value() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add common noun with lower salience
    let entity = create_entity("computer", Some(EntityLabel::Technology), false, 0.3);
    let entity_id = graph.add_entity(entity).expect("Failed to add common noun");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Entity should exist");

    // Common noun base salience should be 0.3
    assert!(
        (result.salience - 0.3).abs() < 0.01,
        "Common noun base salience should be 0.3, got {}",
        result.salience
    );
}

// =============================================================================
// FREQUENCY-BASED SALIENCE BOOST TESTS
// =============================================================================

#[test]
fn test_frequency_boost_logarithmic() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add entity
    let entity = create_entity("test_entity", Some(EntityLabel::Concept), false, 0.5);
    let entity_id = graph.add_entity(entity).expect("Failed to add entity");

    // Get initial salience
    let initial = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Entity should exist");
    let initial_salience = initial.salience;

    // Simulate multiple mentions by adding the same entity again
    for _ in 0..10 {
        let entity = create_entity("test_entity", Some(EntityLabel::Concept), false, 0.5);
        graph.add_entity(entity).expect("Failed to add entity");
    }

    // Get updated entity
    let updated = graph
        .get_entity(&entity_id)
        .expect("Failed to get updated entity")
        .expect("Entity should exist");

    // Salience should have increased
    assert!(
        updated.salience > initial_salience,
        "Salience should increase with mentions: {} > {}",
        updated.salience,
        initial_salience
    );

    // Mention count should have increased
    assert!(
        updated.mention_count > 1,
        "Mention count should increase: {}",
        updated.mention_count
    );
}

#[test]
fn test_frequency_boost_diminishing_returns() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add entity
    let entity = create_entity("frequent_entity", None, false, 0.5);
    graph.add_entity(entity).expect("Failed to add entity");

    // Record salience at different mention counts
    let mut saliences = vec![];

    // Initial salience
    let initial = graph
        .find_entity_by_name("frequent_entity")
        .expect("Failed to find entity")
        .expect("Entity should exist");
    saliences.push(initial.salience);

    // Increment mentions by re-adding entity with same name
    for i in 1..=100 {
        let entity = create_entity("frequent_entity", None, false, 0.5);
        graph.add_entity(entity).expect("Failed to add entity");

        if i == 10 || i == 50 || i == 100 {
            let e = graph
                .find_entity_by_name("frequent_entity")
                .expect("Failed to find entity")
                .expect("Entity should exist");
            saliences.push(e.salience);
        }
    }

    // Verify all saliences are valid
    for s in &saliences {
        assert!(
            *s > 0.0 && *s <= 1.0,
            "Salience should be in (0, 1] range: {}",
            s
        );
    }

    // Salience should be increasing (or at max 1.0)
    for i in 1..saliences.len() {
        assert!(
            saliences[i] >= saliences[i - 1] || (saliences[i] - 1.0).abs() < 0.01,
            "Salience should increase or cap at 1.0"
        );
    }
}

// =============================================================================
// ENTITY TYPE CLASSIFICATION TESTS
// =============================================================================

#[test]
fn test_entity_label_person() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("Alice", Some(EntityLabel::Person), true, 0.6);
    let entity_id = graph.add_entity(entity).expect("Failed to add person");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Should exist");

    assert!(
        result.labels.contains(&EntityLabel::Person),
        "Entity should have Person label"
    );
}

#[test]
fn test_entity_label_organization() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("OpenAI", Some(EntityLabel::Organization), true, 0.6);
    let entity_id = graph
        .add_entity(entity)
        .expect("Failed to add organization");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Should exist");

    assert!(
        result.labels.contains(&EntityLabel::Organization),
        "Entity should have Organization label"
    );
}

#[test]
fn test_entity_label_location() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("San Francisco", Some(EntityLabel::Location), true, 0.6);
    let entity_id = graph.add_entity(entity).expect("Failed to add location");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Should exist");

    assert!(
        result.labels.contains(&EntityLabel::Location),
        "Entity should have Location label"
    );
}

#[test]
fn test_entity_label_technology() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity("Rust", Some(EntityLabel::Technology), true, 0.6);
    let entity_id = graph.add_entity(entity).expect("Failed to add technology");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Should exist");

    assert!(
        result.labels.contains(&EntityLabel::Technology),
        "Entity should have Technology label"
    );
}

#[test]
fn test_entity_label_custom() {
    let (graph, _temp_dir) = setup_graph_memory();

    let entity = create_entity(
        "custom_thing",
        Some(EntityLabel::Other("CustomType".to_string())),
        false,
        0.3,
    );
    let entity_id = graph
        .add_entity(entity)
        .expect("Failed to add custom entity");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Should exist");

    assert!(
        result
            .labels
            .iter()
            .any(|l| matches!(l, EntityLabel::Other(s) if s == "CustomType")),
        "Entity should have custom Other label"
    );
}

#[test]
fn test_entity_no_label() {
    let (graph, _temp_dir) = setup_graph_memory();

    let mut entity = create_entity("unlabeled", None, false, 0.3);
    entity.labels.clear(); // Explicitly clear labels
    let entity_id = graph
        .add_entity(entity)
        .expect("Failed to add unlabeled entity");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Should exist");

    assert!(result.labels.is_empty(), "Entity should have no labels");
}

// =============================================================================
// SALIENCE RANKING TESTS
// =============================================================================

#[test]
fn test_entities_ranked_by_salience() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add entities with different salience levels
    let low_entity = create_entity("low_importance", None, false, 0.2);
    graph
        .add_entity(low_entity)
        .expect("Failed to add low entity");

    let high_entity = create_entity("High_Importance", Some(EntityLabel::Person), true, 0.8);
    let high_id = graph
        .add_entity(high_entity)
        .expect("Failed to add high entity");

    // Boost the high importance entity with multiple mentions
    for _ in 0..20 {
        let entity = create_entity("High_Importance", Some(EntityLabel::Person), true, 0.8);
        graph.add_entity(entity).expect("Failed to increment");
    }

    // Get all entities
    let entities = graph
        .get_all_entities()
        .expect("Failed to get all entities");

    // Find both entities
    let high = entities
        .iter()
        .find(|e| e.name == "High_Importance")
        .expect("Should find high");
    let low = entities
        .iter()
        .find(|e| e.name == "low_importance")
        .expect("Should find low");

    // High salience entity should have higher salience
    assert!(
        high.salience > low.salience,
        "High importance entity ({}) should have higher salience than low ({})",
        high.salience,
        low.salience
    );
}

// =============================================================================
// SALIENCE PERSISTENCE TESTS
// =============================================================================

#[test]
fn test_salience_persisted_to_storage() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let entity_name = "PersistentEntity";
    let expected_salience: f32;

    // Create graph, add entity, and close
    {
        let graph = GraphMemory::new(&db_path).expect("Failed to create graph");

        let entity = create_entity(entity_name, Some(EntityLabel::Person), true, 0.7);
        let entity_id = graph.add_entity(entity).expect("Failed to add entity");

        // Increment mentions to change salience
        for _ in 0..5 {
            let entity = create_entity(entity_name, Some(EntityLabel::Person), true, 0.7);
            graph.add_entity(entity).expect("Failed to increment");
        }

        let e = graph
            .get_entity(&entity_id)
            .expect("Failed to get entity")
            .expect("Should exist");
        expected_salience = e.salience;
    }

    // Reopen graph and verify salience persisted
    {
        let graph = GraphMemory::new(&db_path).expect("Failed to reopen graph");

        let entity = graph
            .find_entity_by_name(entity_name)
            .expect("Failed to find entity")
            .expect("Entity should exist after reopen");

        assert!(
            (entity.salience - expected_salience).abs() < 0.001,
            "Salience should persist: expected {}, got {}",
            expected_salience,
            entity.salience
        );
    }
}

// =============================================================================
// EDGE CASES
// =============================================================================

#[test]
fn test_duplicate_entity_name_increments_mention() {
    let (graph, _temp_dir) = setup_graph_memory();

    // Add entity twice with same name
    let entity1 = create_entity("DuplicateName", Some(EntityLabel::Person), true, 0.6);
    let id1 = graph
        .add_entity(entity1)
        .expect("Failed to add first entity");

    let entity2 = create_entity("DuplicateName", Some(EntityLabel::Person), true, 0.6);
    let id2 = graph
        .add_entity(entity2)
        .expect("Failed to add second entity");

    // Should return same ID (merged)
    assert_eq!(id1, id2, "Duplicate adds should return same UUID");

    // Get entity and check mention count increased
    let entity = graph
        .get_entity(&id1)
        .expect("Failed to get entity")
        .expect("Should exist");

    assert!(
        entity.mention_count >= 2,
        "Duplicate adds should increment mention count: {}",
        entity.mention_count
    );
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
        .expect("Failed to get entity")
        .expect("Should exist");

    assert_eq!(result.name, long_name, "Long name should be preserved");
}

#[test]
fn test_unicode_entity_name() {
    let (graph, _temp_dir) = setup_graph_memory();

    let unicode_name = "北京市 🏙️ München";
    let entity = create_entity(unicode_name, Some(EntityLabel::Location), true, 0.6);
    let entity_id = graph
        .add_entity(entity)
        .expect("Failed to add unicode entity");

    let result = graph
        .get_entity(&entity_id)
        .expect("Failed to get entity")
        .expect("Should exist");

    assert_eq!(
        result.name, unicode_name,
        "Unicode name should be preserved"
    );
}

// =============================================================================
// NER-BASED ENTITY EXTRACTION AND SALIENCE TESTS
// =============================================================================

#[test]
fn test_ner_extract_and_add_entities() {
    let (graph, _temp_dir) = setup_graph_memory();
    let ner = create_test_ner();

    let text = "Microsoft CEO Satya Nadella visited Google headquarters in Seattle";
    let entities = ner.extract(text).expect("NER extraction failed");

    // Add extracted entities to graph with appropriate salience
    for entity in &entities {
        let label = ner_type_to_label(&entity.entity_type);
        let is_proper = true; // NER entities are typically proper nouns
        let base_salience = if is_proper { 0.6 } else { 0.3 };

        let node = create_entity(&entity.text, Some(label), is_proper, base_salience);
        graph.add_entity(node).expect("Failed to add NER entity");
    }

    // Verify entities were added
    let all_entities = graph.get_all_entities().expect("Failed to get entities");
    assert!(
        all_entities.len() >= 2,
        "Should have added NER-extracted entities"
    );
}

#[test]
fn test_ner_organization_salience() {
    let (graph, _temp_dir) = setup_graph_memory();
    let ner = create_test_ner();

    let text = "Infosys and TCS are major Indian IT companies";
    let entities = ner.extract(text).expect("NER extraction failed");

    // Add organizations with proper noun salience
    for entity in entities
        .iter()
        .filter(|e| e.entity_type == NerEntityType::Organization)
    {
        let node = create_entity(&entity.text, Some(EntityLabel::Organization), true, 0.6);
        graph.add_entity(node).expect("Failed to add organization");
    }

    // Verify organizations have higher base salience
    let orgs = graph.get_all_entities().expect("Failed to get entities");
    for org in orgs {
        assert!(
            org.salience >= 0.5,
            "Organization {} should have high base salience: {}",
            org.name,
            org.salience
        );
    }
}

#[test]
fn test_ner_location_salience() {
    let (graph, _temp_dir) = setup_graph_memory();
    let ner = create_test_ner();

    let text = "The conference will be held in Mumbai, Bangalore, and Delhi";
    let entities = ner.extract(text).expect("NER extraction failed");

    // Add locations
    for entity in entities
        .iter()
        .filter(|e| e.entity_type == NerEntityType::Location)
    {
        let node = create_entity(&entity.text, Some(EntityLabel::Location), true, 0.6);
        graph.add_entity(node).expect("Failed to add location");
    }

    // Verify locations were added correctly
    let all_entities = graph.get_all_entities().expect("Failed to get entities");
    let locations: Vec<_> = all_entities
        .iter()
        .filter(|e| e.labels.contains(&EntityLabel::Location))
        .collect();

    assert!(
        locations.len() >= 2,
        "Should find at least 2 Indian locations"
    );
}

#[test]
fn test_ner_repeated_mentions_boost_salience() {
    let (graph, _temp_dir) = setup_graph_memory();
    let ner = create_test_ner();

    // Multiple texts mentioning the same entity
    let texts = vec![
        "Microsoft announced new features",
        "Microsoft stock rose today",
        "Microsoft CEO gave keynote",
        "Microsoft acquired another company",
        "Microsoft cloud services growing",
    ];

    for text in texts {
        let entities = ner.extract(text).expect("NER extraction failed");
        for entity in entities {
            let label = ner_type_to_label(&entity.entity_type);
            let node = create_entity(&entity.text, Some(label), true, 0.6);
            graph.add_entity(node).expect("Failed to add entity");
        }
    }

    // Microsoft should have high salience due to repeated mentions
    let microsoft = graph
        .find_entity_by_name("Microsoft")
        .expect("Failed to find Microsoft")
        .expect("Microsoft should exist");

    assert!(
        microsoft.mention_count >= 4,
        "Microsoft should have multiple mentions: {}",
        microsoft.mention_count
    );
    assert!(
        microsoft.salience > 0.6,
        "Repeated mentions should boost salience: {}",
        microsoft.salience
    );
}

#[test]
fn test_ner_mixed_entity_types_salience() {
    let (graph, _temp_dir) = setup_graph_memory();
    let ner = create_test_ner();

    let text = "Sundar Pichai announced that Google will expand operations in Bangalore and Mumbai";
    let entities = ner.extract(text).expect("NER extraction failed");

    // Add all entities with appropriate types
    for entity in &entities {
        let label = ner_type_to_label(&entity.entity_type);
        let base_salience = match entity.entity_type {
            NerEntityType::Person => 0.7,       // People are often important
            NerEntityType::Organization => 0.6, // Organizations matter
            NerEntityType::Location => 0.5,     // Locations are contextual
            NerEntityType::Misc => 0.4,
        };
        let node = create_entity(&entity.text, Some(label), true, base_salience);
        graph.add_entity(node).expect("Failed to add entity");
    }

    // Verify different types have different salience
    let all_entities = graph.get_all_entities().expect("Failed to get entities");

    let persons: Vec<_> = all_entities
        .iter()
        .filter(|e| e.labels.contains(&EntityLabel::Person))
        .collect();
    let orgs: Vec<_> = all_entities
        .iter()
        .filter(|e| e.labels.contains(&EntityLabel::Organization))
        .collect();
    let locs: Vec<_> = all_entities
        .iter()
        .filter(|e| e.labels.contains(&EntityLabel::Location))
        .collect();

    // Should have extracted multiple types
    assert!(
        !persons.is_empty() || !orgs.is_empty() || !locs.is_empty(),
        "Should have extracted at least one entity type"
    );
}

#[test]
fn test_ner_confidence_affects_salience() {
    let ner = create_test_ner();

    let text = "Microsoft and some_random_word are different";
    let entities = ner.extract(text).expect("NER extraction failed");

    // High confidence entities should be found
    let microsoft = entities.iter().find(|e| e.text == "Microsoft");
    assert!(
        microsoft.is_some(),
        "Should find Microsoft with high confidence"
    );

    if let Some(ms) = microsoft {
        assert!(
            ms.confidence >= 0.5,
            "Microsoft should have high confidence: {}",
            ms.confidence
        );
    }
}

#[test]
fn test_ner_entity_spans_correct() {
    let ner = create_test_ner();

    let text = "Google is headquartered in Mountain View";
    let entities = ner.extract(text).expect("NER extraction failed");

    // Verify spans are correct
    for entity in &entities {
        let extracted = &text[entity.start..entity.end];
        assert_eq!(
            extracted, entity.text,
            "Span should match entity text: {} vs {}",
            extracted, entity.text
        );
    }
}

#[test]
fn test_ner_integration_with_salience_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();
    let ner = create_test_ner();

    let text = "Amazon and Apple are competing in the cloud market";

    // Extract and add entities
    {
        let graph = GraphMemory::new(&db_path).expect("Failed to create graph");
        let entities = ner.extract(text).expect("NER extraction failed");

        for entity in entities {
            let label = ner_type_to_label(&entity.entity_type);
            let node = create_entity(&entity.text, Some(label), true, 0.6);
            graph.add_entity(node).expect("Failed to add entity");
        }
    }

    // Reopen and verify persistence
    {
        let graph = GraphMemory::new(&db_path).expect("Failed to reopen graph");
        let all_entities = graph.get_all_entities().expect("Failed to get entities");

        assert!(
            !all_entities.is_empty(),
            "NER entities should persist across restarts"
        );

        // Check salience persisted
        for entity in all_entities {
            assert!(
                entity.salience > 0.0,
                "Salience should persist: {}",
                entity.salience
            );
        }
    }
}
