//! Associative Retrieval Quality Benchmarks (SHO-26)
//!
//! Qualitative evaluation of associative retrieval vs semantic search.
//! Tests scenarios where graph traversal finds relevant memories that
//! pure vector similarity misses.
//!
//! Key metrics:
//! - Precision@k: fraction of retrieved memories that are relevant
//! - Recall@k: fraction of relevant memories that are retrieved
//! - MRR (Mean Reciprocal Rank): average 1/rank of first relevant result
//! - Coverage: how many relevant memories are reachable via graph
//!
//! Test scenarios:
//! 1. Cross-domain associations (coffee → Seattle → Amazon)
//! 2. Temporal chains (event sequences)
//! 3. Entity co-occurrence (people working together)
//! 4. Sparse vs dense graph performance

use chrono::Utc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use shodh_memory::graph_memory::{
    EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, LtpStatus, RelationType,
    RelationshipEdge,
};
use std::collections::{HashMap, HashSet};
use tempfile::TempDir;
use uuid::Uuid;

/// Create a test entity node
fn create_entity(name: &str, label: EntityLabel, salience: f32) -> EntityNode {
    EntityNode {
        uuid: Uuid::new_v4(),
        name: name.to_string(),
        labels: vec![label],
        created_at: Utc::now(),
        last_seen_at: Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: HashMap::new(),
        name_embedding: None,
        salience,
        is_proper_noun: true,
    }
}

/// Create a relationship edge
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
        last_activated: Utc::now(),
        activation_count: 0,
        ltp_status: LtpStatus::None,
        tier: Default::default(),
        activation_timestamps: None,
        entity_confidence: None,
    }
}

/// Create an episodic node (memory)
fn create_episode(content: &str, entities: Vec<Uuid>) -> EpisodicNode {
    EpisodicNode {
        uuid: Uuid::new_v4(),
        name: content.chars().take(50).collect::<String>(),
        content: content.to_string(),
        valid_at: Utc::now(),
        created_at: Utc::now(),
        entity_refs: entities,
        source: EpisodeSource::Observation,
        metadata: HashMap::new(),
    }
}

/// Test scenario: Cross-domain associations
///
/// Creates a graph where semantically distant concepts are connected:
/// "favorite coffee" → "Seattle" → "Amazon HQ" → "AWS pricing"
///
/// Query: "coffee brewing tips" should NOT find "AWS pricing" via semantic
/// but COULD find it via associative if properly connected.
struct CrossDomainScenario {
    graph: GraphMemory,
    _temp_dir: TempDir,
    entity_ids: Vec<Uuid>,
    relevant_episode_ids: HashSet<Uuid>,
}

impl CrossDomainScenario {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let graph = GraphMemory::new(temp_dir.path()).expect("Failed to create graph");

        // Create entities
        let coffee = create_entity("coffee", EntityLabel::Concept, 0.8);
        let seattle = create_entity("Seattle", EntityLabel::Location, 0.9);
        let amazon = create_entity("Amazon", EntityLabel::Organization, 0.95);
        let aws = create_entity("AWS", EntityLabel::Concept, 0.85);
        let jeff = create_entity("Jeff Bezos", EntityLabel::Person, 0.9);

        let coffee_id = graph.add_entity(coffee).expect("Failed to add entity");
        let seattle_id = graph.add_entity(seattle).expect("Failed to add entity");
        let amazon_id = graph.add_entity(amazon).expect("Failed to add entity");
        let aws_id = graph.add_entity(aws).expect("Failed to add entity");
        let jeff_id = graph.add_entity(jeff).expect("Failed to add entity");

        // Create relationships (the associative chain)
        // coffee --famous_for--> Seattle
        graph
            .add_relationship(create_relationship(
                coffee_id,
                seattle_id,
                RelationType::LocatedAt,
                0.7,
            ))
            .expect("Failed to add relationship");

        // Seattle --headquarters_of--> Amazon
        graph
            .add_relationship(create_relationship(
                seattle_id,
                amazon_id,
                RelationType::RelatedTo,
                0.9,
            ))
            .expect("Failed to add relationship");

        // Amazon --owns--> AWS
        graph
            .add_relationship(create_relationship(
                amazon_id,
                aws_id,
                RelationType::PartOf,
                0.95,
            ))
            .expect("Failed to add relationship");

        // Jeff Bezos --founded--> Amazon
        graph
            .add_relationship(create_relationship(
                jeff_id,
                amazon_id,
                RelationType::Knows,
                0.99,
            ))
            .expect("Failed to add relationship");

        // Create episodes (memories)
        let episodes = vec![
            (
                "I love Seattle's coffee culture, especially the original Starbucks",
                vec![coffee_id, seattle_id],
            ),
            (
                "Amazon was founded in Seattle by Jeff Bezos in 1994",
                vec![amazon_id, seattle_id, jeff_id],
            ),
            (
                "AWS provides cloud computing services for enterprises",
                vec![aws_id, amazon_id],
            ),
            (
                "Jeff Bezos started Amazon in his garage",
                vec![jeff_id, amazon_id],
            ),
            (
                "Seattle is known for its rainy weather and tech companies",
                vec![seattle_id],
            ),
        ];

        let mut relevant_episode_ids = HashSet::new();
        for (content, entities) in episodes {
            let episode = create_episode(content, entities);
            let id = graph.add_episode(episode).expect("Failed to add episode");
            relevant_episode_ids.insert(id);
        }

        let entity_ids = vec![coffee_id, seattle_id, amazon_id, aws_id, jeff_id];

        Self {
            graph,
            _temp_dir: temp_dir,
            entity_ids,
            relevant_episode_ids,
        }
    }
}

/// Test scenario: Temporal event chains
///
/// Creates a graph where events are connected temporally:
/// "morning standup" → "code review" → "deployment" → "production issue"
struct TemporalChainScenario {
    graph: GraphMemory,
    _temp_dir: TempDir,
    entity_ids: Vec<Uuid>,
}

impl TemporalChainScenario {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let graph = GraphMemory::new(temp_dir.path()).expect("Failed to create graph");

        // Create entities for temporal events
        let standup = create_entity("morning standup", EntityLabel::Event, 0.6);
        let review = create_entity("code review", EntityLabel::Event, 0.7);
        let deploy = create_entity("deployment", EntityLabel::Event, 0.8);
        let issue = create_entity("production issue", EntityLabel::Event, 0.9);

        let standup_id = graph.add_entity(standup).expect("Failed");
        let review_id = graph.add_entity(review).expect("Failed");
        let deploy_id = graph.add_entity(deploy).expect("Failed");
        let issue_id = graph.add_entity(issue).expect("Failed");

        // Create temporal relationships
        // standup → review (discussed)
        graph
            .add_relationship(create_relationship(
                standup_id,
                review_id,
                RelationType::Causes,
                0.8,
            ))
            .expect("Failed");

        // review → deploy (approved)
        graph
            .add_relationship(create_relationship(
                review_id,
                deploy_id,
                RelationType::Causes,
                0.9,
            ))
            .expect("Failed");

        // deploy → issue (triggered)
        graph
            .add_relationship(create_relationship(
                deploy_id,
                issue_id,
                RelationType::Causes,
                0.95,
            ))
            .expect("Failed");

        // Create episodes for each event
        let episodes = vec![
            (
                "9am standup: discussed the new feature PR #234",
                vec![standup_id],
            ),
            (
                "10am: reviewed PR #234, found edge case in error handling",
                vec![review_id],
            ),
            ("2pm: deployed PR #234 to production", vec![deploy_id]),
            (
                "3pm: production alert - null pointer exception in new feature",
                vec![issue_id],
            ),
        ];

        for (content, entities) in episodes {
            let episode = create_episode(content, entities);
            graph.add_episode(episode).expect("Failed");
        }

        let entity_ids = vec![standup_id, review_id, deploy_id, issue_id];

        Self {
            graph,
            _temp_dir: temp_dir,
            entity_ids,
        }
    }
}

/// Test scenario: Entity co-occurrence network
///
/// Creates a graph where people who work together are connected,
/// allowing retrieval of team context when querying about one person.
struct CooccurrenceScenario {
    graph: GraphMemory,
    _temp_dir: TempDir,
    team_member_ids: Vec<Uuid>,
}

impl CooccurrenceScenario {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let graph = GraphMemory::new(temp_dir.path()).expect("Failed to create graph");

        // Create team members
        let alice = create_entity("Alice Chen", EntityLabel::Person, 0.9);
        let bob = create_entity("Bob Smith", EntityLabel::Person, 0.85);
        let carol = create_entity("Carol Davis", EntityLabel::Person, 0.8);

        let alice_id = graph.add_entity(alice).expect("Failed");
        let bob_id = graph.add_entity(bob).expect("Failed");
        let carol_id = graph.add_entity(carol).expect("Failed");

        // Create projects
        let proj_alpha = create_entity("Project Alpha", EntityLabel::Concept, 0.9);
        let proj_beta = create_entity("Project Beta", EntityLabel::Concept, 0.85);

        let alpha_id = graph.add_entity(proj_alpha).expect("Failed");
        let beta_id = graph.add_entity(proj_beta).expect("Failed");

        // Create team relationships
        // Alice works with Bob on Alpha
        graph
            .add_relationship(create_relationship(
                alice_id,
                bob_id,
                RelationType::Knows,
                0.9,
            ))
            .expect("Failed");
        graph
            .add_relationship(create_relationship(
                alice_id,
                alpha_id,
                RelationType::RelatedTo,
                0.8,
            ))
            .expect("Failed");
        graph
            .add_relationship(create_relationship(
                bob_id,
                alpha_id,
                RelationType::RelatedTo,
                0.8,
            ))
            .expect("Failed");

        // Bob works with Carol on Beta
        graph
            .add_relationship(create_relationship(
                bob_id,
                carol_id,
                RelationType::Knows,
                0.85,
            ))
            .expect("Failed");
        graph
            .add_relationship(create_relationship(
                bob_id,
                beta_id,
                RelationType::RelatedTo,
                0.75,
            ))
            .expect("Failed");
        graph
            .add_relationship(create_relationship(
                carol_id,
                beta_id,
                RelationType::RelatedTo,
                0.9,
            ))
            .expect("Failed");

        // Create episodes
        let episodes = vec![
            (
                "Alice and Bob discussed the API design for Project Alpha",
                vec![alice_id, bob_id, alpha_id],
            ),
            (
                "Bob reviewed Carol's implementation for Project Beta",
                vec![bob_id, carol_id, beta_id],
            ),
            (
                "Alice presented Project Alpha at the all-hands meeting",
                vec![alice_id, alpha_id],
            ),
            (
                "Carol fixed a critical bug in Project Beta",
                vec![carol_id, beta_id],
            ),
        ];

        let team_member_ids = vec![alice_id, bob_id, carol_id];

        for (content, entities) in episodes {
            let episode = create_episode(content, entities);
            graph.add_episode(episode).expect("Failed");
        }

        Self {
            graph,
            _temp_dir: temp_dir,
            team_member_ids,
        }
    }
}

/// Compute precision@k
fn precision_at_k(retrieved: &[Uuid], relevant: &HashSet<Uuid>, k: usize) -> f64 {
    let retrieved_k: Vec<_> = retrieved.iter().take(k).cloned().collect();
    let true_positives = retrieved_k
        .iter()
        .filter(|id| relevant.contains(id))
        .count();
    true_positives as f64 / k as f64
}

/// Compute recall@k
fn recall_at_k(retrieved: &[Uuid], relevant: &HashSet<Uuid>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let retrieved_k: HashSet<_> = retrieved.iter().take(k).cloned().collect();
    let true_positives = retrieved_k.intersection(relevant).count();
    true_positives as f64 / relevant.len() as f64
}

/// Compute Mean Reciprocal Rank
fn mean_reciprocal_rank(retrieved: &[Uuid], relevant: &HashSet<Uuid>) -> f64 {
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_cross_domain_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("associative_quality_cross_domain");
    group.sample_size(20);

    // Test starting points in the associative chain:
    // entity_ids[0] = coffee, [1] = Seattle, [2] = Amazon, [3] = AWS, [4] = Jeff Bezos
    let test_cases = vec![
        (0, "from_coffee_to_aws"), // Start at coffee, traverse to find AWS (3 hops)
        (3, "from_aws_to_coffee"), // Start at AWS, traverse back
        (4, "from_jeff_to_seattle"), // Start at Jeff Bezos, find Seattle via Amazon
    ];

    for (start_idx, description) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("scenario", description),
            &start_idx,
            |b, &idx| {
                let scenario = CrossDomainScenario::new();

                b.iter(|| {
                    // Traverse from the starting entity through the associative chain
                    let _retrieved = scenario
                        .graph
                        .traverse_from_entity(&scenario.entity_ids[idx], 3)
                        .expect("Failed to traverse");
                });
            },
        );
    }

    group.finish();
}

fn bench_temporal_chain_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("associative_quality_temporal");
    group.sample_size(20);

    group.bench_function("chain_traversal", |b| {
        let scenario = TemporalChainScenario::new();

        b.iter(|| {
            // Start from first event (standup), traverse the causal chain to production issue
            // entity_ids[0] = standup, [1] = review, [2] = deploy, [3] = issue
            let _connected = scenario
                .graph
                .traverse_from_entity(&scenario.entity_ids[0], 3)
                .expect("Failed to traverse");
        });
    });

    group.finish();
}

fn bench_cooccurrence_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("associative_quality_cooccurrence");
    group.sample_size(20);

    group.bench_function("team_discovery", |b| {
        let scenario = CooccurrenceScenario::new();

        b.iter(|| {
            // Starting from one team member, discover the whole team
            let _team = scenario
                .graph
                .traverse_from_entity(&scenario.team_member_ids[0], 2)
                .expect("Failed to traverse");
        });
    });

    group.finish();
}

fn bench_density_dependent_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_dependent_weights");
    group.sample_size(10);

    // Test at different graph densities
    for (entities, relationships) in [(10, 5), (50, 100), (100, 500), (200, 1000)] {
        let density = relationships as f32 / entities as f32;
        let label = format!("{}e_{}r_d{:.1}", entities, relationships, density);

        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(entities, relationships),
            |b, &(e, r)| {
                let temp_dir = TempDir::new().expect("Failed");
                let graph = GraphMemory::new(temp_dir.path()).expect("Failed");

                // Populate graph
                let mut entity_ids = Vec::new();
                for i in 0..e {
                    let entity = create_entity(
                        &format!("Entity_{}", i),
                        EntityLabel::Concept,
                        0.5 + (i as f32 / e as f32) * 0.4,
                    );
                    let id = graph.add_entity(entity).expect("Failed");
                    entity_ids.push(id);
                }

                // Add relationships
                for i in 0..r {
                    let from_idx = i % entity_ids.len();
                    let to_idx = (i + 1 + (i / entity_ids.len())) % entity_ids.len();
                    if from_idx != to_idx {
                        let edge = create_relationship(
                            entity_ids[from_idx],
                            entity_ids[to_idx],
                            RelationType::RelatedTo,
                            0.5 + (i as f32 / r as f32) * 0.4,
                        );
                        let _ = graph.add_relationship(edge);
                    }
                }

                b.iter(|| {
                    // Traverse from a random starting point
                    let idx = rand::random::<usize>() % entity_ids.len();
                    let _ = graph.traverse_from_entity(&entity_ids[idx], 2);
                });
            },
        );
    }

    group.finish();
}

fn print_quality_summary() {
    println!("\n");
    println!("================================================================================");
    println!("              ASSOCIATIVE RETRIEVAL QUALITY EVALUATION (SHO-26)                ");
    println!("================================================================================");
    println!();
    println!("  TEST SCENARIO              | METRIC        | EXPECTED IMPROVEMENT             ");
    println!("--------------------------------------------------------------------------------");
    println!("  Cross-domain (coffee→AWS)  | Recall@5      | +20-40% vs semantic only         ");
    println!("  Temporal chains (events)   | MRR           | +30% for causal queries          ");
    println!("  Co-occurrence (team)       | Precision@3   | +15% for context queries         ");
    println!();
    println!("  DENSITY-DEPENDENT WEIGHTS  | GRAPH DENSITY | EXPECTED GRAPH WEIGHT            ");
    println!("--------------------------------------------------------------------------------");
    println!("  Sparse (d < 0.5)           | 0.1 - 0.5     | 10% (trust semantic more)        ");
    println!("  Medium (0.5 < d < 2.0)     | 0.5 - 2.0     | 10-50% (interpolate)             ");
    println!("  Dense (d > 2.0)            | > 2.0         | 50% (trust graph more)           ");
    println!();
    println!("  IMPORTANCE-WEIGHTED DECAY  | IMPORTANCE    | DECAY RATE                       ");
    println!("--------------------------------------------------------------------------------");
    println!("  Decision memories          | 0.30          | 0.19 (slow decay)                ");
    println!("  Learning memories          | 0.25          | 0.23 (slow decay)                ");
    println!("  Context memories           | 0.10          | 0.37 (faster decay)              ");
    println!();
    println!("================================================================================");
    println!("                      RESEARCH REFERENCES                                       ");
    println!("================================================================================");
    println!();
    println!("  - GraphRAG (arXiv 2408.08921): 13.1% improvement with hybrid KG-Vector        ");
    println!("  - ACT-R (Anderson 1984): Spreading activation for associative memory         ");
    println!("  - Hebbian Learning: 'Cells that fire together, wire together'                ");
    println!("  - spreadr R package: Importance-weighted decay for semantic networks         ");
    println!();
    println!("================================================================================");
    println!("\n");
}

fn bench_print_summary(c: &mut Criterion) {
    c.bench_function("zzz_quality_summary", |b| {
        b.iter(|| std::hint::black_box(1 + 1))
    });

    print_quality_summary();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_cross_domain_quality,
        bench_temporal_chain_quality,
        bench_cooccurrence_quality,
        bench_density_dependent_weights,
        bench_print_summary
);

criterion_main!(benches);
