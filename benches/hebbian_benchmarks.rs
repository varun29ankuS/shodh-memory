//! Hebbian Graph Memory Benchmarks
//!
//! Performance measurements for Hebbian learning features:
//! - Graph stats retrieval
//! - Edge formation during reinforcement
//! - Edge strengthening (LTP pathway)
//! - Graph persistence and reload
//! - Associative retrieval with learned associations

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use shodh_memory::memory::{Experience, MemoryConfig, MemorySystem, Query, RetrievalOutcome};
use tempfile::TempDir;

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

    let memory_system = MemorySystem::new(config, None).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

// =============================================================================
// GRAPH STATS BENCHMARKS
// =============================================================================

fn bench_graph_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("hebbian_graph_stats");
    group.sample_size(30);

    // Empty graph
    group.bench_function("stats_empty", |b| {
        let (memory, _temp) = setup_memory_system();
        b.iter(|| memory.graph_stats());
    });

    // Graph with varying sizes
    for count in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("stats_after_reinforcement", count),
            &count,
            |b, &count| {
                let (mut memory, _temp) = setup_memory_system();

                // Record memories
                let ids: Vec<_> = (0..count)
                    .map(|i| {
                        memory
                            .remember(
                                Experience {
                                    content: format!("Graph stats test memory {}", i),
                                    ..Default::default()
                                },
                                None,
                            )
                            .unwrap()
                    })
                    .collect();

                // Create associations by reinforcing pairs
                for chunk in ids.chunks(5) {
                    if chunk.len() >= 2 {
                        memory
                            .reinforce_recall(&chunk.to_vec(), RetrievalOutcome::Helpful)
                            .unwrap();
                    }
                }

                b.iter(|| memory.graph_stats());
            },
        );
    }

    group.finish();
}

// =============================================================================
// EDGE FORMATION BENCHMARKS
// =============================================================================

fn bench_edge_formation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hebbian_edge_formation");
    group.sample_size(20);

    // Measure edge formation via reinforce_recall
    for pair_count in [2, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("form_edges", pair_count),
            &pair_count,
            |b, &pair_count| {
                b.iter_batched(
                    || {
                        let (mut memory, temp) = setup_memory_system();
                        let ids: Vec<_> = (0..pair_count)
                            .map(|i| {
                                memory
                                    .remember(
                                        Experience {
                                            content: format!("Edge formation memory {}", i),
                                            ..Default::default()
                                        },
                                        None,
                                    )
                                    .unwrap()
                            })
                            .collect();
                        (memory, temp, ids)
                    },
                    |(mut memory, _temp, ids)| {
                        // This forms (n*(n-1))/2 edges between all pairs
                        memory
                            .reinforce_recall(&ids, RetrievalOutcome::Helpful)
                            .unwrap();
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// EDGE STRENGTHENING BENCHMARKS (LTP PATHWAY)
// =============================================================================

fn bench_edge_strengthening(c: &mut Criterion) {
    let mut group = c.benchmark_group("hebbian_edge_strengthening");
    group.sample_size(20);

    // Measure repeated strengthening (simulates LTP pathway)
    for coactivation_count in [1, 3, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new("strengthen_edge", coactivation_count),
            &coactivation_count,
            |b, &coactivation_count| {
                b.iter_batched(
                    || {
                        let (mut memory, temp) = setup_memory_system();
                        // Create two memories to form an edge between
                        let id1 = memory
                            .remember(
                                Experience {
                                    content: "First memory for edge".to_string(),
                                    ..Default::default()
                                },
                                None,
                            )
                            .unwrap();
                        let id2 = memory
                            .remember(
                                Experience {
                                    content: "Second memory for edge".to_string(),
                                    ..Default::default()
                                },
                                None,
                            )
                            .unwrap();

                        // Form initial edge
                        memory
                            .reinforce_recall(
                                &vec![id1.clone(), id2.clone()],
                                RetrievalOutcome::Helpful,
                            )
                            .unwrap();

                        (memory, temp, id1, id2)
                    },
                    |(mut memory, _temp, id1, id2)| {
                        // Strengthen edge multiple times
                        for _ in 0..coactivation_count {
                            memory
                                .reinforce_recall(
                                    &vec![id1.clone(), id2.clone()],
                                    RetrievalOutcome::Helpful,
                                )
                                .unwrap();
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// GRAPH PERSISTENCE BENCHMARKS
// =============================================================================

fn bench_graph_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("hebbian_graph_persistence");
    group.sample_size(10);

    // Measure graph save time (via drop + recreate)
    for edge_count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("persist_and_reload", edge_count),
            &edge_count,
            |b, &edge_count| {
                b.iter_batched(
                    || {
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

                        let mut memory = MemorySystem::new(config.clone(), None)
                            .expect("Failed to create system");

                        // Create edges
                        let memory_count = (edge_count as f64).sqrt().ceil() as usize * 2;
                        let ids: Vec<_> = (0..memory_count)
                            .map(|i| {
                                memory
                                    .remember(
                                        Experience {
                                            content: format!("Persistence test {}", i),
                                            ..Default::default()
                                        },
                                        None,
                                    )
                                    .unwrap()
                            })
                            .collect();

                        // Form edges in batches
                        for chunk in ids.chunks(5) {
                            if chunk.len() >= 2 {
                                memory
                                    .reinforce_recall(&chunk.to_vec(), RetrievalOutcome::Helpful)
                                    .unwrap();
                            }
                        }

                        (memory, config, temp_dir)
                    },
                    |(memory, config, temp_dir)| {
                        // Drop triggers save
                        drop(memory);

                        // Recreate loads persisted graph
                        let _reloaded = MemorySystem::new(config, None).expect("Failed to reload");
                        drop(temp_dir);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// ASSOCIATIVE RETRIEVAL BENCHMARKS
// =============================================================================

fn bench_associative_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("hebbian_associative_retrieval");
    group.sample_size(15);

    // Compare retrieval with and without Hebbian associations
    group.bench_function("retrieval_with_associations", |b| {
        b.iter_batched(
            || {
                let (mut memory, temp) = setup_memory_system();

                // Create semantically related memories
                let contents = vec![
                    "Rust programming language memory safety",
                    "Rust borrow checker prevents data races",
                    "Rust ownership model for memory management",
                    "Python dynamic typing interpreted language",
                    "Python GIL global interpreter lock",
                ];

                let ids: Vec<_> = contents
                    .into_iter()
                    .map(|c| {
                        memory
                            .remember(
                                Experience {
                                    content: c.to_string(),
                                    ..Default::default()
                                },
                                None,
                            )
                            .unwrap()
                    })
                    .collect();

                // Form associations between Rust-related memories
                memory
                    .reinforce_recall(&ids[0..3].to_vec(), RetrievalOutcome::Helpful)
                    .unwrap();

                // Form associations between Python-related memories
                memory
                    .reinforce_recall(&ids[3..5].to_vec(), RetrievalOutcome::Helpful)
                    .unwrap();

                (memory, temp)
            },
            |(memory, _temp)| {
                // Query for Rust - should retrieve associated memories
                let query = Query {
                    query_text: Some("Rust memory".to_string()),
                    max_results: 5,
                    ..Default::default()
                };
                let _results = memory.recall(&query).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// LTP THRESHOLD BENCHMARKS
// =============================================================================

fn bench_ltp_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("hebbian_ltp");
    group.sample_size(10);

    // Measure performance of reaching LTP threshold (5 coactivations)
    group.bench_function("reach_ltp_threshold", |b| {
        b.iter_batched(
            || {
                let (mut memory, temp) = setup_memory_system();
                let id1 = memory
                    .remember(
                        Experience {
                            content: "LTP memory 1".to_string(),
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap();
                let id2 = memory
                    .remember(
                        Experience {
                            content: "LTP memory 2".to_string(),
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap();
                (memory, temp, id1, id2)
            },
            |(mut memory, _temp, id1, id2)| {
                // Reach LTP threshold (5 coactivations)
                for _ in 0..5 {
                    memory
                        .reinforce_recall(
                            &vec![id1.clone(), id2.clone()],
                            RetrievalOutcome::Helpful,
                        )
                        .unwrap();
                }

                // Verify potentiation via stats
                let _stats = memory.graph_stats();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// SCALABILITY BENCHMARKS
// =============================================================================

fn bench_large_graph_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hebbian_scalability");
    group.sample_size(10);

    // Large graph with many edges
    group.bench_function("large_graph_1000_memories", |b| {
        b.iter_batched(
            || {
                let (mut memory, temp) = setup_memory_system();

                // Create 1000 memories
                let ids: Vec<_> = (0..1000)
                    .map(|i| {
                        memory
                            .remember(
                                Experience {
                                    content: format!("Large graph memory {} with some content", i),
                                    ..Default::default()
                                },
                                None,
                            )
                            .unwrap()
                    })
                    .collect();

                // Form associations in groups of 5
                for chunk in ids.chunks(5) {
                    if chunk.len() >= 2 {
                        memory
                            .reinforce_recall(&chunk.to_vec(), RetrievalOutcome::Helpful)
                            .unwrap();
                    }
                }

                (memory, temp)
            },
            |(memory, _temp)| {
                // Query and check stats
                let stats = memory.graph_stats();
                assert!(stats.edge_count > 0);
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

// =============================================================================
// SUMMARY OUTPUT
// =============================================================================

fn bench_print_summary(c: &mut Criterion) {
    c.bench_function("zzz_hebbian_summary", |b| {
        b.iter(|| std::hint::black_box(1 + 1))
    });

    print_hebbian_performance_summary();
}

fn print_hebbian_performance_summary() {
    println!("\n");
    println!("================================================================================");
    println!("                    HEBBIAN GRAPH PERFORMANCE SUMMARY                          ");
    println!("================================================================================");
    println!();
    println!("  OPERATION                    | EXPECTED P50  | NOTES                         ");
    println!("--------------------------------------------------------------------------------");
    println!("  Graph Stats (empty)          |  < 0.1ms      | Constant time O(1)            ");
    println!("  Graph Stats (500 memories)   |  < 1ms        | With ~200 edges               ");
    println!("  Edge Formation (2 memories)  |  < 1ms        | Forms 1 edge                  ");
    println!("  Edge Formation (20 memories) |  < 5ms        | Forms 190 edges               ");
    println!("  Edge Strengthening (1x)      |  < 1ms        | Single coactivation           ");
    println!("  Edge Strengthening (5x)      |  < 3ms        | LTP threshold                 ");
    println!("  Persist + Reload (100 edges) |  < 50ms       | Full graph serialization      ");
    println!("  Associative Retrieval        |  < 10ms       | Semantic + association boost  ");
    println!("  LTP Threshold (5 coact)      |  < 5ms        | Edge becomes permanent        ");
    println!();
    println!("  HEBBIAN LEARNING PROPERTIES                                                  ");
    println!("--------------------------------------------------------------------------------");
    println!("  - Edges form between co-retrieved memories                                   ");
    println!("  - Edge strength increases: w' = w + α(1 - w)                                 ");
    println!("  - LTP threshold: 5 coactivations                                             ");
    println!("  - Potentiated edges persist permanently                                      ");
    println!("  - Graph persists across restarts via binary serialization                    ");
    println!();
    println!("================================================================================");
    println!("\n");
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    name = hebbian_benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_graph_stats,
        bench_edge_formation,
        bench_edge_strengthening,
        bench_graph_persistence,
        bench_associative_retrieval,
        bench_ltp_threshold,
        bench_large_graph_operations,
        bench_print_summary
);

criterion_main!(hebbian_benches);
