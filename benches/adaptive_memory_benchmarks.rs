//! Benchmarks for Adaptive Memory Operations
//!
//! Performance benchmarks for the three core adaptive memory systems:
//! - Outcome Feedback System (Hebbian reinforcement)
//! - Semantic Consolidation (episodic → semantic extraction)
//! - Anticipatory Prefetch (context-aware query generation)
//! - NER integration for entity extraction
//!
//! These benchmarks ensure adaptive operations are fast enough for real-time use.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tempfile::TempDir;

use shodh_memory::uuid::Uuid;

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{
    compression::SemanticConsolidator,
    retrieval::{AnticipatoryPrefetch, PrefetchContext, RetrievalOutcome},
    types::{Experience, ExperienceType, Query},
    Memory, MemoryConfig, MemoryId, MemorySystem,
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create fallback NER instance for testing
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

fn setup_memory_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    };
    let system = MemorySystem::new(config).expect("Failed to create memory system");
    (system, temp_dir)
}

fn create_experience(content: &str, exp_type: ExperienceType, entities: Vec<&str>) -> Experience {
    Experience {
        experience_type: exp_type,
        content: content.to_string(),
        entities: entities.into_iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
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

fn populate_memories(system: &MemorySystem, count: usize) -> Vec<shodh_memory::memory::MemoryId> {
    let mut ids = Vec::with_capacity(count);
    let ner = setup_fallback_ner();

    // Rich content with named entities for NER extraction
    let templates = [
        "Satya Nadella from Microsoft discussed strategy in Seattle",
        "Sundar Pichai announced Google's new AI features at Mountain View",
        "Tim Cook presented Apple's latest products in Cupertino",
        "Mark Zuckerberg talked about Meta's VR developments in Menlo Park",
        "Jensen Huang showcased NVIDIA's GPUs at GTC conference",
    ];

    for i in 0..count {
        let exp_type = match i % 5 {
            0 => ExperienceType::Learning,
            1 => ExperienceType::Decision,
            2 => ExperienceType::Error,
            3 => ExperienceType::Conversation,
            _ => ExperienceType::Observation,
        };

        let content = format!("Benchmark {} - {}", i, templates[i % templates.len()]);

        let exp = create_experience_with_ner(&content, exp_type, &ner);

        if let Ok(id) = system.record(exp) {
            ids.push(id);
        }
    }

    ids
}

// ============================================================================
// OUTCOME FEEDBACK BENCHMARKS
// ============================================================================

fn bench_tracked_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_tracked_retrieval");

    for memory_count in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(memory_count),
            &memory_count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let (system, temp_dir) = setup_memory_system();
                        populate_memories(&system, count);
                        (system, temp_dir)
                    },
                    |(system, _temp_dir)| {
                        let query = Query {
                            query_text: Some("benchmark test performance".to_string()),
                            max_results: 10,
                            ..Default::default()
                        };
                        system.retrieve_tracked(&query).expect("Failed")
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_reinforce_helpful(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_reinforce_helpful");

    for batch_size in [2, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let (system, temp_dir) = setup_memory_system();
                        let ids = populate_memories(&system, size);
                        (system, temp_dir, ids)
                    },
                    |(system, _temp_dir, ids)| {
                        system
                            .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
                            .expect("Failed")
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_reinforce_misleading(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_reinforce_misleading");

    for batch_size in [2, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let (system, temp_dir) = setup_memory_system();
                        let ids = populate_memories(&system, size);
                        (system, temp_dir, ids)
                    },
                    |(system, _temp_dir, ids)| {
                        system
                            .reinforce_retrieval(&ids, RetrievalOutcome::Misleading)
                            .expect("Failed")
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_reinforce_neutral(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_reinforce_neutral");

    for batch_size in [2, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let (system, temp_dir) = setup_memory_system();
                        let ids = populate_memories(&system, size);
                        (system, temp_dir, ids)
                    },
                    |(system, _temp_dir, ids)| {
                        system
                            .reinforce_retrieval(&ids, RetrievalOutcome::Neutral)
                            .expect("Failed")
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_coactivation_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_coactivation_scaling");
    group.sample_size(30);

    // Test how coactivation scales with number of memories
    // Coactivation creates C(n,2) = n*(n-1)/2 edges
    for n in [3, 5, 10, 15, 20] {
        let expected_edges = n * (n - 1) / 2;
        let label = format!("{}mem_{}edges", n, expected_edges);

        group.bench_with_input(BenchmarkId::from_parameter(&label), &n, |b, &size| {
            b.iter_batched(
                || {
                    let (system, temp_dir) = setup_memory_system();
                    let ids = populate_memories(&system, size);
                    (system, temp_dir, ids)
                },
                |(system, _temp_dir, ids)| {
                    system
                        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
                        .expect("Failed")
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_repeated_reinforcement(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_repeated_reinforcement");

    // Test performance of repeated reinforcement on same memories
    for iterations in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(iterations),
            &iterations,
            |b, &iters| {
                b.iter_batched(
                    || {
                        let (system, temp_dir) = setup_memory_system();
                        let ids = populate_memories(&system, 5);
                        (system, temp_dir, ids)
                    },
                    |(system, _temp_dir, ids)| {
                        for _ in 0..iters {
                            system
                                .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
                                .expect("Failed");
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_graph_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_graph_stats");

    for memory_count in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(memory_count),
            &memory_count,
            |b, &count| {
                // Setup: create system and build graph
                let (system, _temp_dir) = setup_memory_system();
                let ids = populate_memories(&system, count);

                // Build some associations
                for chunk in ids.chunks(5) {
                    let _ = system.reinforce_retrieval(chunk, RetrievalOutcome::Helpful);
                }

                b.iter(|| system.graph_stats());
            },
        );
    }

    group.finish();
}

fn bench_graph_maintenance(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_graph_maintenance");

    for memory_count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(memory_count),
            &memory_count,
            |b, &count| {
                // Setup: create system with associations
                let (system, _temp_dir) = setup_memory_system();
                let ids = populate_memories(&system, count);

                // Build associations
                for chunk in ids.chunks(5) {
                    let _ = system.reinforce_retrieval(chunk, RetrievalOutcome::Helpful);
                }

                b.iter(|| system.graph_maintenance());
            },
        );
    }

    group.finish();
}

// ============================================================================
// SEMANTIC CONSOLIDATION BENCHMARKS
// ============================================================================

fn create_test_memories(count: usize) -> Vec<Memory> {
    let mut memories = Vec::with_capacity(count);
    let ner = setup_fallback_ner();

    // Rich templates with named entities for NER extraction
    let templates = [
        (
            "Learning: Satya Nadella shared that Microsoft Azure handles {} concurrent users",
            ExperienceType::Learning,
        ),
        (
            "Decision: Sundar Pichai approved deployment of Google Cloud version {}",
            ExperienceType::Decision,
        ),
        (
            "Error: Amazon AWS system {} failed with database connection in Seattle",
            ExperienceType::Error,
        ),
        (
            "Conversation: Tim Cook discussed Apple's dark mode feature with user {}",
            ExperienceType::Conversation,
        ),
        (
            "Discovery: Jensen Huang found pattern {} in NVIDIA's GPU data",
            ExperienceType::Discovery,
        ),
    ];

    for i in 0..count {
        let (template, exp_type) = &templates[i % templates.len()];
        let content = template.replace("{}", &format!("{}", i));

        let exp = create_experience_with_ner(&content, exp_type.clone(), &ner);
        memories.push(Memory::new(
            MemoryId(Uuid::new_v4()),
            exp,
            0.5,
            None,
            None,
            None,
        ));
    }

    memories
}

fn bench_consolidator_creation(c: &mut Criterion) {
    c.bench_function("adaptive_consolidator_new", |b| {
        b.iter(|| SemanticConsolidator::new());
    });
}

fn bench_consolidator_with_thresholds(c: &mut Criterion) {
    c.bench_function("adaptive_consolidator_with_thresholds", |b| {
        b.iter(|| SemanticConsolidator::with_thresholds(3, 14));
    });
}

fn bench_consolidate_empty(c: &mut Criterion) {
    c.bench_function("adaptive_consolidate_empty", |b| {
        let consolidator = SemanticConsolidator::new();
        b.iter(|| consolidator.consolidate(&[]));
    });
}

fn bench_consolidate_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_consolidate_scaling");

    for count in [10, 50, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &size| {
            let consolidator = SemanticConsolidator::with_thresholds(2, 0); // Allow immediate consolidation
            let memories = create_test_memories(size);

            b.iter(|| consolidator.consolidate(&memories));
        });
    }

    group.finish();
}

fn bench_reinforce_fact(c: &mut Criterion) {
    use shodh_memory::chrono::Utc;
    use shodh_memory::memory::compression::{FactType, SemanticFact};

    c.bench_function("adaptive_reinforce_fact", |b| {
        let consolidator = SemanticConsolidator::new();

        b.iter_batched(
            || {
                let fact = SemanticFact {
                    id: "test_fact".to_string(),
                    fact: "Test fact for benchmarking".to_string(),
                    confidence: 0.5,
                    support_count: 1,
                    source_memories: vec![],
                    related_entities: vec![],
                    created_at: Utc::now(),
                    last_reinforced: Utc::now(),
                    fact_type: FactType::Pattern,
                };

                let exp = create_experience("Evidence", ExperienceType::Learning, vec!["test"]);
                let memory = Memory::new(MemoryId(Uuid::new_v4()), exp, 0.5, None, None, None);

                (fact, memory)
            },
            |(mut fact, memory)| consolidator.reinforce_fact(&mut fact, &memory),
            BatchSize::SmallInput,
        );
    });
}

fn bench_should_decay_fact(c: &mut Criterion) {
    use shodh_memory::chrono::{Duration, Utc};
    use shodh_memory::memory::compression::{FactType, SemanticFact};

    let mut group = c.benchmark_group("adaptive_should_decay_fact");

    // Test with different fact ages
    for age_days in [10, 30, 90, 365] {
        group.bench_with_input(
            BenchmarkId::from_parameter(age_days),
            &age_days,
            |b, &age| {
                let consolidator = SemanticConsolidator::new();
                let fact = SemanticFact {
                    id: "old_fact".to_string(),
                    fact: "Old fact".to_string(),
                    confidence: 0.5,
                    support_count: 3,
                    source_memories: vec![],
                    related_entities: vec![],
                    created_at: Utc::now() - Duration::days(age),
                    last_reinforced: Utc::now() - Duration::days(age),
                    fact_type: FactType::Pattern,
                };

                b.iter(|| consolidator.should_decay_fact(&fact));
            },
        );
    }

    group.finish();
}

// ============================================================================
// ANTICIPATORY PREFETCH BENCHMARKS
// ============================================================================

fn bench_prefetch_creation(c: &mut Criterion) {
    c.bench_function("adaptive_prefetch_new", |b| {
        b.iter(|| AnticipatoryPrefetch::new());
    });
}

fn bench_prefetch_with_limits(c: &mut Criterion) {
    c.bench_function("adaptive_prefetch_with_limits", |b| {
        b.iter(|| AnticipatoryPrefetch::with_limits(50, 0.5, 4));
    });
}

fn bench_prefetch_context_creation(c: &mut Criterion) {
    c.bench_function("adaptive_prefetch_context_default", |b| {
        b.iter(|| PrefetchContext::default());
    });
}

fn bench_prefetch_context_from_time(c: &mut Criterion) {
    c.bench_function("adaptive_prefetch_context_from_time", |b| {
        b.iter(|| PrefetchContext::from_current_time());
    });
}

fn bench_generate_prefetch_query_project(c: &mut Criterion) {
    c.bench_function("adaptive_generate_query_project", |b| {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            project_id: Some("benchmark-project".to_string()),
            ..Default::default()
        };

        b.iter(|| prefetch.generate_prefetch_query(&ctx));
    });
}

fn bench_generate_prefetch_query_entities(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_generate_query_entities");

    for entity_count in [1, 3, 5, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(entity_count),
            &entity_count,
            |b, &count| {
                let prefetch = AnticipatoryPrefetch::new();
                let entities: Vec<String> = (0..count).map(|i| format!("Entity{}", i)).collect();

                let ctx = PrefetchContext {
                    recent_entities: entities,
                    ..Default::default()
                };

                b.iter(|| prefetch.generate_prefetch_query(&ctx));
            },
        );
    }

    group.finish();
}

fn bench_generate_prefetch_query_file(c: &mut Criterion) {
    c.bench_function("adaptive_generate_query_file", |b| {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            current_file: Some("/src/components/auth/LoginForm.tsx".to_string()),
            ..Default::default()
        };

        b.iter(|| prefetch.generate_prefetch_query(&ctx));
    });
}

fn bench_generate_prefetch_query_temporal(c: &mut Criterion) {
    c.bench_function("adaptive_generate_query_temporal", |b| {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            hour_of_day: Some(14),
            day_of_week: Some(1),
            ..Default::default()
        };

        b.iter(|| prefetch.generate_prefetch_query(&ctx));
    });
}

fn bench_generate_prefetch_query_empty(c: &mut Criterion) {
    c.bench_function("adaptive_generate_query_empty", |b| {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext::default();

        b.iter(|| prefetch.generate_prefetch_query(&ctx));
    });
}

fn bench_relevance_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_relevance_score");

    // Test with different context complexities
    let contexts = vec![
        ("minimal", PrefetchContext::default()),
        (
            "project_only",
            PrefetchContext {
                project_id: Some("test-project".to_string()),
                ..Default::default()
            },
        ),
        (
            "entities_only",
            PrefetchContext {
                recent_entities: vec!["User".to_string(), "Auth".to_string()],
                ..Default::default()
            },
        ),
        (
            "full_context",
            PrefetchContext {
                project_id: Some("test-project".to_string()),
                current_file: Some("login.rs".to_string()),
                recent_entities: vec!["User".to_string(), "Auth".to_string()],
                hour_of_day: Some(14),
                day_of_week: Some(1),
                task_type: Some("coding".to_string()),
                ..Default::default()
            },
        ),
    ];

    for (label, ctx) in contexts {
        group.bench_with_input(BenchmarkId::from_parameter(label), &ctx, |b, ctx| {
            let prefetch = AnticipatoryPrefetch::new();
            let exp = create_experience(
                "Memory about login.rs and authentication",
                ExperienceType::Learning,
                vec!["User", "Auth"],
            );
            let memory = Memory::new(MemoryId(Uuid::new_v4()), exp, 0.5, None, None, None);

            b.iter(|| prefetch.relevance_score(&memory, ctx));
        });
    }

    group.finish();
}

fn bench_relevance_score_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_relevance_score_batch");

    for batch_size in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let prefetch = AnticipatoryPrefetch::new();
                let memories = create_test_memories(size);
                let ctx = PrefetchContext {
                    project_id: Some("test".to_string()),
                    recent_entities: vec!["test".to_string()],
                    ..Default::default()
                };

                b.iter(|| {
                    memories
                        .iter()
                        .map(|m| prefetch.relevance_score(m, &ctx))
                        .collect::<Vec<_>>()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// END-TO-END WORKFLOW BENCHMARKS
// ============================================================================

fn bench_full_feedback_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_full_feedback_loop");
    group.sample_size(30);

    for memory_count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(memory_count),
            &memory_count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let (system, temp_dir) = setup_memory_system();
                        populate_memories(&system, count);
                        (system, temp_dir)
                    },
                    |(system, _temp_dir)| {
                        // 1. Query with tracking
                        let query = Query {
                            query_text: Some("benchmark test".to_string()),
                            max_results: 10,
                            ..Default::default()
                        };
                        let tracked = system.retrieve_tracked(&query).expect("Failed");

                        // 2. Provide feedback
                        system
                            .reinforce_tracked(&tracked, RetrievalOutcome::Helpful)
                            .expect("Failed");

                        // 3. Get stats
                        system.graph_stats()
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_iterative_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_iterative_learning");
    group.sample_size(20);

    // Simulate iterative learning over multiple feedback cycles
    for cycles in [5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(cycles),
            &cycles,
            |b, &num_cycles| {
                b.iter_batched(
                    || {
                        let (system, temp_dir) = setup_memory_system();
                        populate_memories(&system, 20);
                        (system, temp_dir)
                    },
                    |(system, _temp_dir)| {
                        for i in 0..num_cycles {
                            let query = Query {
                                query_text: Some(format!("benchmark memory {}", i % 5)),
                                max_results: 5,
                                ..Default::default()
                            };

                            if let Ok(tracked) = system.retrieve_tracked(&query) {
                                let outcome = match i % 3 {
                                    0 => RetrievalOutcome::Helpful,
                                    1 => RetrievalOutcome::Neutral,
                                    _ => RetrievalOutcome::Misleading,
                                };
                                let _ = system.reinforce_tracked(&tracked, outcome);
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

// ============================================================================
// SUMMARY
// ============================================================================

fn bench_print_summary(c: &mut Criterion) {
    c.bench_function("zzz_adaptive_summary", |b| {
        b.iter(|| std::hint::black_box(1 + 1))
    });

    print_performance_summary();
}

fn print_performance_summary() {
    println!("\n");
    println!("================================================================================");
    println!("                 ADAPTIVE MEMORY PERFORMANCE SUMMARY                           ");
    println!("================================================================================");
    println!();
    println!("  OUTCOME FEEDBACK              | TARGET        | NOTES                        ");
    println!("--------------------------------------------------------------------------------");
    println!("  Tracked Retrieval             |  < 50ms       | Query + tracking overhead    ");
    println!("  Reinforce (Helpful, 5 mem)    |  < 1ms        | Graph update + importance    ");
    println!("  Reinforce (Misleading, 5 mem) |  < 0.5ms      | No graph update              ");
    println!("  Coactivation (10 mem, 45 edges)|  < 5ms       | O(n²) edge creation          ");
    println!("  Graph Stats                   |  < 0.1ms      | In-memory traversal          ");
    println!("  Graph Maintenance             |  < 10ms       | Decay calculation            ");
    println!();
    println!("  SEMANTIC CONSOLIDATION        | TARGET        | NOTES                        ");
    println!("--------------------------------------------------------------------------------");
    println!("  Consolidator Creation         |  < 0.01ms     | Simple allocation            ");
    println!("  Consolidate (100 memories)    |  < 10ms       | Pattern extraction           ");
    println!("  Reinforce Fact                |  < 0.1ms      | Update existing fact         ");
    println!("  Should Decay Check            |  < 0.01ms     | Simple calculation           ");
    println!();
    println!("  ANTICIPATORY PREFETCH         | TARGET        | NOTES                        ");
    println!("--------------------------------------------------------------------------------");
    println!("  Context Creation              |  < 0.01ms     | Simple struct                ");
    println!("  Generate Query (project)      |  < 0.1ms      | String formatting            ");
    println!("  Generate Query (entities)     |  < 0.1ms      | Join operation               ");
    println!("  Generate Query (temporal)     |  < 0.5ms      | DateTime calculations        ");
    println!("  Relevance Score               |  < 0.1ms      | Per-memory scoring           ");
    println!("  Relevance Score Batch (100)   |  < 10ms       | Scoring all memories         ");
    println!();
    println!("  END-TO-END WORKFLOWS          | TARGET        | NOTES                        ");
    println!("--------------------------------------------------------------------------------");
    println!("  Full Feedback Loop            |  < 100ms      | Query → Feedback → Stats     ");
    println!("  Iterative Learning (10 cycles)|  < 500ms      | Multiple feedback rounds     ");
    println!();
    println!("================================================================================");
    println!("                         DESIGN IMPLICATIONS                                    ");
    println!("================================================================================");
    println!();
    println!("  - Feedback is cheap enough for every retrieval                               ");
    println!("  - Coactivation scales O(n²) - consider batching for large result sets        ");
    println!("  - Consolidation should run in background (not on hot path)                   ");
    println!("  - Prefetch queries are fast - can run speculatively                          ");
    println!("  - Relevance scoring enables real-time cache prioritization                   ");
    println!();
    println!("================================================================================");
    println!("\n");
}

// ============================================================================
// CRITERION GROUPS
// ============================================================================

criterion_group!(
    name = outcome_feedback_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_tracked_retrieval,
        bench_reinforce_helpful,
        bench_reinforce_misleading,
        bench_reinforce_neutral,
        bench_coactivation_scaling,
        bench_repeated_reinforcement,
        bench_graph_stats,
        bench_graph_maintenance
);

criterion_group!(
    name = semantic_consolidation_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_consolidator_creation,
        bench_consolidator_with_thresholds,
        bench_consolidate_empty,
        bench_consolidate_scaling,
        bench_reinforce_fact,
        bench_should_decay_fact
);

criterion_group!(
    name = anticipatory_prefetch_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_prefetch_creation,
        bench_prefetch_with_limits,
        bench_prefetch_context_creation,
        bench_prefetch_context_from_time,
        bench_generate_prefetch_query_project,
        bench_generate_prefetch_query_entities,
        bench_generate_prefetch_query_file,
        bench_generate_prefetch_query_temporal,
        bench_generate_prefetch_query_empty,
        bench_relevance_score,
        bench_relevance_score_batch
);

criterion_group!(
    name = workflow_benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_full_feedback_loop,
        bench_iterative_learning,
        bench_print_summary
);

criterion_main!(
    outcome_feedback_benches,
    semantic_consolidation_benches,
    anticipatory_prefetch_benches,
    workflow_benches
);
