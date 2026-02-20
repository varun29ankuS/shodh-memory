//! Graph Memory Benchmarks
//!
//! Performance benchmarks for knowledge graph operations:
//! - Entity CRUD operations
//! - Relationship operations
//! - Graph traversal
//! - Universe visualization
//! - NER integration for entity extraction
//!
//! Compare against industry standards (Neo4j, Memgraph, etc.)

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use shodh_memory::chrono::Utc;
use shodh_memory::embeddings::ner::{NerConfig, NerEntityType, NeuralNer};
use shodh_memory::graph_memory::{
    EntityLabel, EntityNode, GraphMemory, LtpStatus, RelationType, RelationshipEdge,
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
        ltp_status: LtpStatus::None,
        tier: Default::default(),
        activation_timestamps: None,
        entity_confidence: None,
    }
}

/// Helper: Create test graph memory
fn setup_graph_memory() -> (GraphMemory, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let graph = GraphMemory::new(temp_dir.path()).expect("Failed to create graph memory");
    (graph, temp_dir)
}

/// Helper: Populate graph with test data
fn populate_graph(
    graph: &GraphMemory,
    entity_count: usize,
    relationship_count: usize,
) -> Vec<Uuid> {
    let mut entity_ids = Vec::new();

    for i in 0..entity_count {
        let entity = create_entity(
            &format!("Entity_{}", i),
            Some(EntityLabel::Person),
            i % 2 == 0,
            0.5,
        );
        let id = graph.add_entity(entity).expect("Failed to add entity");
        entity_ids.push(id);
    }

    // Add relationships (sparse graph)
    for i in 0..relationship_count.min(entity_count.saturating_sub(1)) {
        let from_idx = i;
        let to_idx = (i + 1) % entity_count;
        let edge = create_relationship(
            entity_ids[from_idx],
            entity_ids[to_idx],
            RelationType::Knows,
            0.8,
        );
        graph
            .add_relationship(edge)
            .expect("Failed to add relationship");
    }

    entity_ids
}

// =============================================================================
// Entity CRUD Benchmarks
// =============================================================================

fn bench_entity_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_entity_add");

    // Test different entity counts
    for count in [1, 10, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter_batched(
                || setup_graph_memory(),
                |(graph, _temp_dir)| {
                    for i in 0..count {
                        let entity = create_entity(
                            &format!("BenchEntity_{}", i),
                            Some(EntityLabel::Person),
                            true,
                            0.6,
                        );
                        graph.add_entity(entity).expect("Failed to add entity");
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_entity_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_entity_get");

    // Pre-populate graphs of different sizes
    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // Setup: populate graph
            let (graph, _temp_dir) = setup_graph_memory();
            let entity_ids = populate_graph(&graph, size, 0);

            b.iter(|| {
                // Get a random entity
                let idx = rand::random::<usize>() % entity_ids.len();
                graph
                    .get_entity(&entity_ids[idx])
                    .expect("Failed to get entity");
            });
        });
    }

    group.finish();
}

fn bench_entity_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_entity_search");

    for size in [100, 500, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let (graph, _temp_dir) = setup_graph_memory();
            populate_graph(&graph, size, 0);

            b.iter(|| {
                graph
                    .find_entity_by_name("Entity_50")
                    .expect("Failed to search");
            });
        });
    }

    group.finish();
}

// =============================================================================
// Relationship Benchmarks
// =============================================================================

fn bench_relationship_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_relationship_add");

    for count in [10, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter_batched(
                || {
                    let (graph, temp_dir) = setup_graph_memory();
                    let entity_ids = populate_graph(&graph, count + 1, 0);
                    (graph, temp_dir, entity_ids)
                },
                |(graph, _temp_dir, entity_ids)| {
                    for i in 0..count {
                        let edge = create_relationship(
                            entity_ids[i],
                            entity_ids[i + 1],
                            RelationType::Knows,
                            0.8,
                        );
                        graph
                            .add_relationship(edge)
                            .expect("Failed to add relationship");
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_relationship_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_relationship_query");

    for (entities, relationships) in [(100, 50), (500, 200), (1000, 500)] {
        let label = format!("{}e_{}r", entities, relationships);
        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(entities, relationships),
            |b, &(e, r)| {
                let (graph, _temp_dir) = setup_graph_memory();
                let entity_ids = populate_graph(&graph, e, r);

                b.iter(|| {
                    // Query relationships for a random entity
                    let idx = rand::random::<usize>() % entity_ids.len();
                    graph
                        .get_entity_relationships(&entity_ids[idx])
                        .expect("Failed to get relationships");
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Graph Traversal Benchmarks
// =============================================================================

fn bench_connected_entities(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_traversal");
    group.sample_size(50); // Reduced for faster benchmarks

    // Create a densely connected graph
    for depth in [1, 2, 3] {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            let (graph, _temp_dir) = setup_graph_memory();

            // Create a graph with ~100 entities and ~200 relationships
            let entity_ids = populate_graph(&graph, 100, 200);

            b.iter(|| {
                graph
                    .traverse_from_entity(&entity_ids[0], depth)
                    .expect("Failed to traverse");
            });
        });
    }

    group.finish();
}

// =============================================================================
// Salience Update Benchmarks
// =============================================================================

fn bench_mention_increment(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_salience_update");

    for batch_size in [1, 10, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let (graph, _temp_dir) = setup_graph_memory();
                let entity_ids = populate_graph(&graph, batch_size, 0);

                b.iter(|| {
                    // Re-add entities to increment mention counts
                    for i in 0..batch_size {
                        let entity = create_entity(
                            &format!("Entity_{}", i),
                            Some(EntityLabel::Person),
                            true,
                            0.5,
                        );
                        graph.add_entity(entity).expect("Failed to increment");
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Hebbian Synaptic Plasticity Benchmarks
// =============================================================================

fn bench_synapse_strengthen(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_hebbian_strengthen");

    for count in [10, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let (graph, _temp_dir) = setup_graph_memory();
            let _ = populate_graph(&graph, count + 1, count);

            // Get all relationships
            let all_rels = graph.get_all_relationships().expect("Failed");
            let edge_ids: Vec<_> = all_rels.iter().map(|e| e.uuid).collect();

            b.iter(|| {
                // Strengthen a random synapse
                let idx = rand::random::<usize>() % edge_ids.len();
                graph
                    .strengthen_synapse(&edge_ids[idx])
                    .expect("Failed to strengthen");
            });
        });
    }

    group.finish();
}

fn bench_synapse_decay(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_hebbian_decay");

    for count in [10, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let (graph, _temp_dir) = setup_graph_memory();
            let _ = populate_graph(&graph, count + 1, count);

            // Get all relationships
            let all_rels = graph.get_all_relationships().expect("Failed");
            let edge_ids: Vec<_> = all_rels.iter().map(|e| e.uuid).collect();

            b.iter(|| {
                // Decay a random synapse
                let idx = rand::random::<usize>() % edge_ids.len();
                graph
                    .decay_synapse(&edge_ids[idx])
                    .expect("Failed to decay");
            });
        });
    }

    group.finish();
}

fn bench_effective_strength(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_hebbian_effective_strength");

    for count in [10, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let (graph, _temp_dir) = setup_graph_memory();
            let _ = populate_graph(&graph, count + 1, count);

            // Get all relationships
            let all_rels = graph.get_all_relationships().expect("Failed");
            let edge_ids: Vec<_> = all_rels.iter().map(|e| e.uuid).collect();

            b.iter(|| {
                // Get relationship with effective strength (lazy decay calculation)
                let idx = rand::random::<usize>() % edge_ids.len();
                graph
                    .get_relationship_with_effective_strength(&edge_ids[idx])
                    .expect("Failed to get effective strength");
            });
        });
    }

    group.finish();
}

fn bench_traversal_with_hebbian(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_hebbian_traversal");
    group.sample_size(30); // Reduced for complex operations

    // Traverse graph with Hebbian strengthening enabled
    for (entities, relationships, depth) in [(50, 100, 2), (100, 200, 2), (200, 400, 3)] {
        let label = format!("{}e_{}r_d{}", entities, relationships, depth);
        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(entities, relationships, depth),
            |b, &(e, r, d)| {
                let (graph, _temp_dir) = setup_graph_memory();
                let entity_ids = populate_graph(&graph, e, r);

                b.iter(|| {
                    // Traverse from a random starting point
                    let idx = rand::random::<usize>() % entity_ids.len();
                    graph
                        .traverse_from_entity(&entity_ids[idx], d)
                        .expect("Failed to traverse");
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// NER-Driven Entity Extraction Benchmarks
// =============================================================================

fn bench_ner_to_graph_entity(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_ner_extraction");

    // Test entity extraction from text and graph insertion
    let test_texts = vec![
        "Satya Nadella from Microsoft discussed strategy with Sundar Pichai at Google headquarters",
        "OpenAI's Sam Altman met Elon Musk in San Francisco to discuss artificial intelligence",
        "Tim Cook announced Apple's new headquarters in Cupertino California",
    ];

    for (idx, text) in test_texts.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("extract_and_add", idx), text, |b, text| {
            b.iter_batched(
                || {
                    let (graph, temp_dir) = setup_graph_memory();
                    let ner = setup_fallback_ner();
                    (graph, temp_dir, ner)
                },
                |(graph, _temp_dir, ner)| {
                    // Extract entities using NER
                    let entities = ner.extract(text).unwrap_or_default();

                    // Add each extracted entity to graph
                    for entity in entities {
                        let label = ner_type_to_label(&entity.entity_type);
                        let node =
                            create_entity(&entity.text, Some(label), true, entity.confidence);
                        graph.add_entity(node).expect("Failed to add entity");
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_ner_batch_to_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_ner_batch");

    // Batch processing: multiple texts
    for batch_size in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let texts: Vec<String> = (0..size)
                    .map(|i| {
                        format!(
                            "Person_{} from Organization_{} visited Location_{} to discuss Project_{}",
                            i, i, i, i
                        )
                    })
                    .collect();

                b.iter_batched(
                    || {
                        let (graph, temp_dir) = setup_graph_memory();
                        let ner = setup_fallback_ner();
                        (graph, temp_dir, ner, texts.clone())
                    },
                    |(graph, _temp_dir, ner, texts)| {
                        for text in &texts {
                            let entities = ner.extract(text).unwrap_or_default();
                            for entity in entities {
                                let label = ner_type_to_label(&entity.entity_type);
                                let node = create_entity(
                                    &entity.text,
                                    Some(label),
                                    true,
                                    entity.confidence,
                                );
                                graph.add_entity(node).expect("Failed");
                            }
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_ner_entity_relationships(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_ner_relationships");
    group.sample_size(30);

    // Extract entities and create co-occurrence relationships
    group.bench_function("cooccurrence_from_text", |b| {
        let text = "Satya Nadella and Sundar Pichai discussed AI partnerships between Microsoft and Google in Seattle";

        b.iter_batched(
            || {
                let (graph, temp_dir) = setup_graph_memory();
                let ner = setup_fallback_ner();
                (graph, temp_dir, ner)
            },
            |(graph, _temp_dir, ner)| {
                let entities = ner.extract(text).unwrap_or_default();
                let mut entity_ids = Vec::new();

                // Add all entities
                for entity in &entities {
                    let label = ner_type_to_label(&entity.entity_type);
                    let node = create_entity(&entity.text, Some(label), true, entity.confidence);
                    let id = graph.add_entity(node).expect("Failed");
                    entity_ids.push(id);
                }

                // Create co-occurrence relationships
                for i in 0..entity_ids.len() {
                    for j in (i + 1)..entity_ids.len() {
                        let edge = create_relationship(
                            entity_ids[i],
                            entity_ids[j],
                            RelationType::RelatedTo,
                            0.7,
                        );
                        graph.add_relationship(edge).expect("Failed");
                    }
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// Universe Visualization Benchmarks
// =============================================================================

fn bench_universe_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_universe");
    group.sample_size(30); // Reduced for complex operations

    for (entities, relationships) in [(10, 5), (50, 25), (100, 50), (500, 200)] {
        let label = format!("{}e_{}r", entities, relationships);
        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(entities, relationships),
            |b, &(e, r)| {
                let (graph, _temp_dir) = setup_graph_memory();
                populate_graph(&graph, e, r);

                b.iter(|| {
                    graph.get_universe().expect("Failed to get universe");
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Comparison Summary (printed at end)
// =============================================================================

fn bench_print_graph_summary(c: &mut Criterion) {
    c.bench_function("zzz_graph_summary", |b| {
        b.iter(|| std::hint::black_box(1 + 1))
    });

    print_graph_performance_summary();
}

fn print_graph_performance_summary() {
    println!("\n");
    println!("================================================================================");
    println!("                    GRAPH MEMORY PERFORMANCE SUMMARY                            ");
    println!("================================================================================");
    println!();
    println!("  OPERATION                    | EXPECTED P50  | INDUSTRY (NEO4J)              ");
    println!("--------------------------------------------------------------------------------");
    println!("  Entity Add                   |  < 1ms        | ~2-5ms (cold)                 ");
    println!("  Entity Get                   |  < 0.1ms      | ~0.5-1ms                      ");
    println!("  Entity Search                |  < 5ms        | ~10-50ms (depends on index)   ");
    println!("  Relationship Add             |  < 1ms        | ~2-5ms                        ");
    println!("  Relationship Query           |  < 1ms        | ~1-5ms                        ");
    println!("  Graph Traversal (depth=2)    |  < 10ms       | ~10-100ms                     ");
    println!("  Universe Generation (100e)   |  < 50ms       | N/A (custom feature)          ");
    println!("  Salience Update              |  < 0.5ms      | N/A (custom feature)          ");
    println!();
    println!("  HEBBIAN PLASTICITY           | EXPECTED P50  | NOTES                         ");
    println!("--------------------------------------------------------------------------------");
    println!("  Synapse Strengthen           |  < 0.5ms      | Per-edge, with persistence    ");
    println!("  Synapse Decay                |  < 0.5ms      | Per-edge, exponential         ");
    println!("  Effective Strength (lazy)    |  < 0.1ms      | Read-only calculation         ");
    println!("  Traverse w/ Hebbian (100e)   |  < 15ms       | Includes auto-strengthening   ");
    println!();
    println!("================================================================================");
    println!("                           ADVANTAGES                                           ");
    println!("================================================================================");
    println!();
    println!("  - Embedded (no network latency) vs Neo4j's client-server model               ");
    println!("  - Built-in salience (gravitational memory) - unique feature                  ");
    println!("  - Temporal invalidation for evolving relationships                           ");
    println!("  - Universe visualization API for 3D rendering                                ");
    println!("  - Integrated with vector search for hybrid retrieval                         ");
    println!("  - Hebbian synaptic plasticity: edges strengthen with use                     ");
    println!("  - Long-Term Potentiation: frequently used paths become permanent             ");
    println!();
    println!("================================================================================");
    println!("\n");
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_entity_add,
        bench_entity_get,
        bench_entity_search,
        bench_relationship_add,
        bench_relationship_query,
        bench_connected_entities,
        bench_mention_increment,
        bench_synapse_strengthen,
        bench_synapse_decay,
        bench_effective_strength,
        bench_traversal_with_hebbian,
        bench_ner_to_graph_entity,
        bench_ner_batch_to_graph,
        bench_ner_entity_relationships,
        bench_universe_generation,
        bench_print_graph_summary
);

criterion_main!(benches);
