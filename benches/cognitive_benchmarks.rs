//! Cognitive System Benchmarks
//!
//! Performance measurements for cognitive memory features:
//! - Hebbian reinforcement (feedback loop latency)
//! - Spreading activation (graph traversal)
//! - Memory tier promotion/demotion
//! - Entity reference operations
//! - Activation decay calculations
//! - NER integration for entity extraction

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{
    EntityRef, Experience, ExperienceType, Memory, MemoryConfig, MemoryId, MemorySystem,
    MemoryTier, Query, RetrievalOutcome,
};
use shodh_memory::uuid::Uuid;
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

    let memory_system = MemorySystem::new(config).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

/// Create fallback NER instance for testing
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Populate memory system with NER-extracted entities
fn populate_memories(memory: &mut MemorySystem, count: usize) {
    let ner = setup_fallback_ner();
    for i in 0..count {
        let content = format!(
            "Memory entry {} - Satya Nadella from Microsoft discussed with Sundar Pichai from Google in Bangalore",
            i
        );
        let entities = ner.extract(&content).unwrap_or_default();
        let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
        let exp = Experience {
            content,
            experience_type: ExperienceType::Observation,
            entities: entity_names,
            ..Default::default()
        };
        memory.remember(exp, None).expect("Failed to record");
    }
}

// =============================================================================
// MEMORY STRUCT OPERATION BENCHMARKS
// =============================================================================

fn bench_memory_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_creation");

    group.bench_function("new_minimal", |b| {
        b.iter(|| {
            Memory::new(
                MemoryId(Uuid::new_v4()),
                Experience::default(),
                0.5,
                None,
                None,
                None,
                None, // created_at
            )
        });
    });

    group.bench_function("new_full", |b| {
        b.iter(|| {
            Memory::new(
                MemoryId(Uuid::new_v4()),
                Experience {
                    content: "Full experience with all fields populated for benchmarking"
                        .to_string(),
                    experience_type: ExperienceType::Decision,
                    entities: vec!["entity1".to_string(), "entity2".to_string()],
                    ..Default::default()
                },
                0.85,
                Some("agent-1".to_string()),
                Some("run-1".to_string()),
                Some("actor-1".to_string()),
                None, // created_at
            )
        });
    });

    group.finish();
}

fn bench_entity_ref_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_ref_ops");

    // Add entity refs
    group.bench_function("add_entity_ref", |b| {
        b.iter_batched(
            || {
                Memory::new(
                    MemoryId(Uuid::new_v4()),
                    Experience::default(),
                    0.5,
                    None,
                    None,
                    None,
                    None, // created_at
                )
            },
            |mut memory| {
                memory.add_entity_ref(
                    Uuid::new_v4(),
                    "entity".to_string(),
                    "mentioned".to_string(),
                );
            },
            BatchSize::SmallInput,
        );
    });

    // Add many entity refs
    for count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("add_many_refs", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        Memory::new(
                            MemoryId(Uuid::new_v4()),
                            Experience::default(),
                            0.5,
                            None,
                            None,
                            None,
                            None, // created_at
                        )
                    },
                    |mut memory| {
                        for i in 0..count {
                            memory.add_entity_ref(
                                Uuid::new_v4(),
                                format!("entity_{}", i),
                                "mentioned".to_string(),
                            );
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    // Get entity IDs
    group.bench_function("get_entity_ids_100", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );
        for i in 0..100 {
            memory.add_entity_ref(Uuid::new_v4(), format!("e{}", i), "x".to_string());
        }

        b.iter(|| memory.entity_ids());
    });

    group.finish();
}

fn bench_tier_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier_ops");

    group.bench_function("promote", |b| {
        b.iter_batched(
            || {
                Memory::new(
                    MemoryId(Uuid::new_v4()),
                    Experience::default(),
                    0.5,
                    None,
                    None,
                    None,
                    None, // created_at
                )
            },
            |mut memory| {
                memory.promote();
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("demote", |b| {
        b.iter_batched(
            || {
                let mut m = Memory::new(
                    MemoryId(Uuid::new_v4()),
                    Experience::default(),
                    0.5,
                    None,
                    None,
                    None,
                    None, // created_at
                );
                m.tier = MemoryTier::Archive;
                m
            },
            |mut memory| {
                memory.demote();
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("full_promotion_cycle", |b| {
        b.iter_batched(
            || {
                Memory::new(
                    MemoryId(Uuid::new_v4()),
                    Experience::default(),
                    0.5,
                    None,
                    None,
                    None,
                    None, // created_at
                )
            },
            |mut memory| {
                memory.promote(); // Working -> Session
                memory.promote(); // Session -> LongTerm
                memory.promote(); // LongTerm -> Archive
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_activation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_ops");

    group.bench_function("activate", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );
        memory.set_activation(0.5);

        b.iter(|| {
            memory.activate(0.1);
        });
    });

    group.bench_function("decay_activation", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );

        b.iter(|| {
            memory.decay_activation(0.99);
        });
    });

    // Simulate spreading activation decay across many memories
    group.bench_function("batch_decay_1000", |b| {
        let mut memories: Vec<_> = (0..1000)
            .map(|_| {
                Memory::new(
                    MemoryId(Uuid::new_v4()),
                    Experience::default(),
                    0.5,
                    None,
                    None,
                    None,
                    None, // created_at
                )
            })
            .collect();

        b.iter(|| {
            for memory in &mut memories {
                memory.decay_activation(0.95);
            }
        });
    });

    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    // Create memory with entity refs
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: "Test content for serialization benchmarks".to_string(),
            experience_type: ExperienceType::Decision,
            ..Default::default()
        },
        0.75,
        Some("agent".to_string()),
        None,
        None,
        None, // created_at
    );
    for i in 0..10 {
        memory.add_entity_ref(
            Uuid::new_v4(),
            format!("entity_{}", i),
            "mentioned".to_string(),
        );
    }
    memory.tier = MemoryTier::Session;
    memory.set_activation(0.8);
    memory.last_retrieval_id = Some(Uuid::new_v4());

    group.bench_function("bincode_serialize", |b| {
        b.iter(|| {
            bincode::serde::encode_to_vec(&memory, bincode::config::standard())
                .expect("Failed to serialize")
        });
    });

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
        .expect("Failed to serialize");

    group.bench_function("bincode_deserialize", |b| {
        b.iter(|| {
            let (_result, _): (Memory, _) =
                bincode::serde::decode_from_slice(&serialized, bincode::config::standard())
                    .expect("Failed to deserialize");
        });
    });

    group.bench_function("bincode_roundtrip", |b| {
        b.iter(|| {
            let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard())
                .expect("Failed to serialize");
            let (_result, _): (Memory, _) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                    .expect("Failed to deserialize");
        });
    });

    group.finish();
}

// =============================================================================
// HEBBIAN LEARNING BENCHMARKS
// =============================================================================

fn bench_hebbian_reinforcement(c: &mut Criterion) {
    let mut group = c.benchmark_group("hebbian");
    group.sample_size(20); // Reduced due to I/O

    // Single memory reinforcement
    group.bench_function("reinforce_single_helpful", |b| {
        b.iter_batched(
            || {
                let (mut memory, temp) = setup_memory_system();
                let id = memory
                    .remember(
                        Experience {
                            content: "Test memory".to_string(),
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap();
                (memory, temp, vec![id])
            },
            |(mut memory, _temp, ids)| {
                memory
                    .reinforce_recall(&ids, RetrievalOutcome::Helpful)
                    .unwrap()
            },
            BatchSize::SmallInput,
        );
    });

    // Multiple memory reinforcement (association strengthening)
    for count in [2, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("reinforce_batch", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let (mut memory, temp) = setup_memory_system();
                        let ids: Vec<_> = (0..count)
                            .map(|i| {
                                memory
                                    .remember(
                                        Experience {
                                            content: format!("Memory {}", i),
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
                        memory
                            .reinforce_recall(&ids, RetrievalOutcome::Helpful)
                            .unwrap()
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_full_feedback_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("feedback_loop");
    group.sample_size(10);

    // Complete retrieve -> reinforce cycle
    group.bench_function("retrieve_and_reinforce", |b| {
        b.iter_batched(
            || {
                let (mut memory, temp) = setup_memory_system();
                populate_memories(&mut memory, 50);
                (memory, temp)
            },
            |(mut memory, _temp)| {
                // Retrieve
                let query = Query {
                    query_text: Some("cognitive benchmark test".to_string()),
                    max_results: 5,
                    ..Default::default()
                };
                let results = memory.recall(&query).unwrap();

                // Reinforce
                let ids: Vec<_> = results.iter().map(|m| m.id.clone()).collect();
                if !ids.is_empty() {
                    memory
                        .reinforce_recall(&ids, RetrievalOutcome::Helpful)
                        .unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// IMPORTANCE CALCULATION BENCHMARKS
// =============================================================================

fn bench_importance_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("importance");

    group.bench_function("get_importance", |b| {
        let memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );

        b.iter(|| memory.importance());
    });

    group.bench_function("set_importance", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );

        b.iter(|| {
            memory.set_importance(0.75);
        });
    });

    group.bench_function("boost_importance", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );

        b.iter(|| {
            memory.boost_importance(0.05);
        });
    });

    group.bench_function("decay_importance", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );

        b.iter(|| {
            memory.decay_importance(0.01);
        });
    });

    group.finish();
}

// =============================================================================
// ACCESS PATTERN BENCHMARKS
// =============================================================================

fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_patterns");

    group.bench_function("record_access", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );

        b.iter(|| {
            memory.record_access();
        });
    });

    group.bench_function("get_access_count", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );
        for _ in 0..100 {
            memory.record_access();
        }

        b.iter(|| memory.access_count());
    });

    group.bench_function("mark_retrieved", |b| {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );

        b.iter(|| {
            memory.mark_retrieved(Uuid::new_v4());
        });
    });

    group.finish();
}

// =============================================================================
// CONCURRENT ACCESS BENCHMARKS
// =============================================================================

fn bench_concurrent_memory_ops(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let mut group = c.benchmark_group("concurrent");
    group.sample_size(20);

    group.bench_function("concurrent_importance_reads_10t", |b| {
        b.iter_batched(
            || {
                Arc::new(Memory::new(
                    MemoryId(Uuid::new_v4()),
                    Experience::default(),
                    0.5,
                    None,
                    None,
                    None,
                    None, // created_at
                ))
            },
            |memory| {
                let handles: Vec<_> = (0..10)
                    .map(|_| {
                        let mem = Arc::clone(&memory);
                        thread::spawn(move || {
                            for _ in 0..100 {
                                let _ = mem.importance();
                                let _ = mem.access_count();
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    name = cognitive_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_memory_creation,
        bench_entity_ref_operations,
        bench_tier_operations,
        bench_activation_operations,
        bench_serialization,
        bench_hebbian_reinforcement,
        bench_full_feedback_loop,
        bench_importance_operations,
        bench_access_patterns,
        bench_concurrent_memory_ops
);

criterion_main!(cognitive_benches);
