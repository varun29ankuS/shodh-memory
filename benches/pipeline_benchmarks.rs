//! Full Pipeline Benchmarks - NER + Embedding + Storage Step-by-Step Breakdown
//!
//! This benchmark measures each individual component of the memory pipeline:
//! 1. NER Extraction (Neural or Fallback)
//! 2. Embedding Generation (MiniLM-L6-v2)
//! 3. Memory Creation (struct allocation)
//! 4. Storage (RocksDB + Vector Index)
//! 5. Retrieval (Query embedding + Vector search + Deserialization)
//!
//! Run with: cargo bench --bench pipeline_benchmarks

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use shodh_memory::embeddings::minilm::{EmbeddingConfig, MiniLMEmbedder};
use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::embeddings::Embedder;
use shodh_memory::memory::{
    Experience, ExperienceType, MemoryConfig, MemorySystem, Query, RetrievalMode,
};
use std::time::Instant;
use tempfile::TempDir;

// ==============================================================================
// Test Data: Realistic AI Agent Memory Content
// ==============================================================================

/// Short text - typical user input or brief observation
const SHORT_TEXT: &str = "Sundar Pichai from Google visited IIT Madras in Chennai";

/// Medium text - typical memory content with entities
const MEDIUM_TEXT: &str = "Satya Nadella announced that Microsoft will partner with OpenAI \
    to bring GPT technology to Azure cloud services. The collaboration was celebrated \
    at their headquarters in Seattle and will expand to offices in Bangalore and London.";

/// Long text - paragraph-level memory with multiple entities
const LONG_TEXT: &str = "In a landmark development for India's technology sector, \
    Mukesh Ambani's Reliance Industries announced a strategic partnership with Google and Facebook. \
    The collaboration, valued at over 500 crore rupees, will see Jio Platforms expanding its \
    5G infrastructure across Mumbai, Delhi, Bangalore, and Chennai. This follows similar investments \
    by Tata Group in Pune and Infosys in Hyderabad. Industry experts including N. R. Narayana Murthy \
    and Nandan Nilekani praised the move, stating it would accelerate India's digital transformation. \
    The Ministry of Electronics and Information Technology in New Delhi welcomed the announcement.";

// ==============================================================================
// Helper Functions
// ==============================================================================

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

/// Create NER instance (neural with fallback)
fn setup_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new(config).expect("Failed to create NER")
}

/// Create fallback NER (rule-based only)
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Create embedder
fn setup_embedder() -> MiniLMEmbedder {
    let config = EmbeddingConfig::default();
    MiniLMEmbedder::new(config).expect("Failed to create embedder")
}

/// Populate memory system with test data
fn populate_memories(memory_system: &mut MemorySystem, count: usize) {
    for i in 0..count {
        let content = format!(
            "Memory entry {i} - This is a test memory containing various information about task execution, \
             decision making, and context tracking. References to entities like Microsoft, Google, Bangalore."
        );
        let experience = Experience {
            content,
            experience_type: ExperienceType::Observation,
            ..Default::default()
        };
        memory_system.record(experience).expect("Failed to record");
    }
}

// ==============================================================================
// PIPELINE STEP 1: NER EXTRACTION BENCHMARKS
// ==============================================================================

fn bench_pipeline_step1_ner(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STEP 1: NER EXTRACTION - Named Entity Recognition          ║");
    eprintln!("║  Model: bert-tiny-NER (~17MB ONNX)                           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let ner = setup_ner();
    let fallback_ner = setup_fallback_ner();

    // Warm up neural model
    let _ = ner.extract("Warmup text");

    let mut group = c.benchmark_group("step1_ner_extraction");

    // Neural NER benchmarks
    group.bench_function("neural_short", |b| {
        b.iter(|| ner.extract(SHORT_TEXT));
    });

    group.bench_function("neural_medium", |b| {
        b.iter(|| ner.extract(MEDIUM_TEXT));
    });

    group.bench_function("neural_long", |b| {
        b.iter(|| ner.extract(LONG_TEXT));
    });

    // Fallback NER benchmarks (rule-based)
    group.bench_function("fallback_short", |b| {
        b.iter(|| fallback_ner.extract(SHORT_TEXT));
    });

    group.bench_function("fallback_medium", |b| {
        b.iter(|| fallback_ner.extract(MEDIUM_TEXT));
    });

    group.bench_function("fallback_long", |b| {
        b.iter(|| fallback_ner.extract(LONG_TEXT));
    });

    group.finish();

    // Print entity extraction summary
    eprintln!("\n📊 NER EXTRACTION SUMMARY:");
    if let Ok(entities) = ner.extract(MEDIUM_TEXT) {
        eprintln!(
            "   Medium text ({} chars): {} entities found",
            MEDIUM_TEXT.len(),
            entities.len()
        );
        for e in &entities {
            eprintln!(
                "     - {} ({:?}, confidence: {:.2})",
                e.text, e.entity_type, e.confidence
            );
        }
    }
}

// ==============================================================================
// PIPELINE STEP 2: EMBEDDING GENERATION BENCHMARKS
// ==============================================================================

fn bench_pipeline_step2_embedding(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STEP 2: EMBEDDING GENERATION - MiniLM-L6-v2                 ║");
    eprintln!("║  Model: all-MiniLM-L6-v2 (~22MB ONNX, 384-dim vectors)       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let embedder = setup_embedder();

    let mut group = c.benchmark_group("step2_embedding_generation");

    group.bench_function("short_text", |b| {
        b.iter(|| embedder.encode(SHORT_TEXT).expect("Failed to encode"));
    });

    group.bench_function("medium_text", |b| {
        b.iter(|| embedder.encode(MEDIUM_TEXT).expect("Failed to encode"));
    });

    group.bench_function("long_text", |b| {
        b.iter(|| embedder.encode(LONG_TEXT).expect("Failed to encode"));
    });

    group.finish();

    // Print embedding info
    eprintln!("\n📊 EMBEDDING GENERATION SUMMARY:");
    if let Ok(embedding) = embedder.encode(MEDIUM_TEXT) {
        eprintln!("   Vector dimensions: {}", embedding.len());
        eprintln!("   First 5 values: {:?}", &embedding[..5]);
    }
}

// ==============================================================================
// PIPELINE STEP 3: COMBINED NER + EMBEDDING (PARALLEL POTENTIAL)
// ==============================================================================

fn bench_pipeline_step3_ner_embedding_combined(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STEP 3: COMBINED NER + EMBEDDING (Both Models)              ║");
    eprintln!("║  Measures total latency for entity + vector extraction       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let ner = setup_ner();
    let embedder = setup_embedder();

    // Warm up
    let _ = ner.extract("Warmup");
    let _ = embedder.encode("Warmup");

    let mut group = c.benchmark_group("step3_ner_embedding_combined");

    // Sequential NER + Embedding (current implementation)
    group.bench_function("sequential_short", |b| {
        b.iter(|| {
            let _ = ner.extract(SHORT_TEXT);
            let _ = embedder.encode(SHORT_TEXT);
        });
    });

    group.bench_function("sequential_medium", |b| {
        b.iter(|| {
            let _ = ner.extract(MEDIUM_TEXT);
            let _ = embedder.encode(MEDIUM_TEXT);
        });
    });

    group.bench_function("sequential_long", |b| {
        b.iter(|| {
            let _ = ner.extract(LONG_TEXT);
            let _ = embedder.encode(LONG_TEXT);
        });
    });

    group.finish();
}

// ==============================================================================
// PIPELINE STEP 4: MEMORY STORAGE BENCHMARKS
// ==============================================================================

fn bench_pipeline_step4_storage(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STEP 4: MEMORY STORAGE - RocksDB + Vector Index             ║");
    eprintln!("║  Measures write latency (embedding + indexing + persistence) ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    // Create shared memory system (model loads once)
    let (mut memory_system, _temp_dir) = setup_memory_system();

    let mut group = c.benchmark_group("step4_storage");

    // Short content storage
    group.bench_function("store_short", |b| {
        b.iter_batched(
            || Experience {
                content: SHORT_TEXT.to_string(),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            },
            |exp| memory_system.record(exp).expect("Failed to store"),
            BatchSize::SmallInput,
        );
    });

    // Medium content storage
    group.bench_function("store_medium", |b| {
        b.iter_batched(
            || Experience {
                content: MEDIUM_TEXT.to_string(),
                experience_type: ExperienceType::Decision,
                ..Default::default()
            },
            |exp| memory_system.record(exp).expect("Failed to store"),
            BatchSize::SmallInput,
        );
    });

    // Long content storage
    group.bench_function("store_long", |b| {
        b.iter_batched(
            || Experience {
                content: LONG_TEXT.to_string(),
                experience_type: ExperienceType::Learning,
                ..Default::default()
            },
            |exp| memory_system.record(exp).expect("Failed to store"),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ==============================================================================
// PIPELINE STEP 5: MEMORY RETRIEVAL BENCHMARKS
// ==============================================================================

fn bench_pipeline_step5_retrieval(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  STEP 5: MEMORY RETRIEVAL - Query Embedding + Vector Search  ║");
    eprintln!("║  Measures end-to-end retrieval latency                       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    // Pre-populate memory system
    let (mut memory_system, _temp_dir) = setup_memory_system();
    populate_memories(&mut memory_system, 100);

    let mut group = c.benchmark_group("step5_retrieval");

    // Different result sizes
    for k in [1, 5, 10, 25] {
        group.bench_with_input(BenchmarkId::new("top_k", k), &k, |b, &k| {
            b.iter(|| {
                let query = Query {
                    query_text: Some("Microsoft Google partnership technology".to_string()),
                    max_results: k,
                    retrieval_mode: RetrievalMode::Similarity,
                    ..Default::default()
                };
                memory_system.retrieve(&query).expect("Failed to retrieve")
            });
        });
    }

    // Different retrieval modes
    group.bench_function("similarity_mode", |b| {
        b.iter(|| {
            let query = Query {
                query_text: Some("task execution debugging".to_string()),
                max_results: 5,
                retrieval_mode: RetrievalMode::Similarity,
                ..Default::default()
            };
            memory_system.retrieve(&query).expect("Failed")
        });
    });

    group.bench_function("hybrid_mode", |b| {
        b.iter(|| {
            let query = Query {
                query_text: Some("task execution debugging".to_string()),
                max_results: 5,
                retrieval_mode: RetrievalMode::Hybrid,
                ..Default::default()
            };
            memory_system.retrieve(&query).expect("Failed")
        });
    });

    group.finish();
}

// ==============================================================================
// FULL PIPELINE END-TO-END BENCHMARK
// ==============================================================================

fn bench_full_pipeline_end_to_end(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  FULL PIPELINE: Record (NER + Embedding) → Retrieve          ║");
    eprintln!("║  Complete user experience latency                            ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");

    let (mut memory_system, _temp_dir) = setup_memory_system();
    populate_memories(&mut memory_system, 50);

    c.bench_function("full_pipeline_record_retrieve", |b| {
        b.iter(|| {
            // Record new memory (NER + Embedding + Storage)
            let experience = Experience {
                content:
                    "User discussed partnership between Infosys and Google in Bangalore office"
                        .to_string(),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            };
            let _id = memory_system.record(experience).expect("Failed to record");

            // Retrieve related memories (Embedding + Vector Search)
            let query = Query {
                query_text: Some("partnership technology company".to_string()),
                max_results: 5,
                retrieval_mode: RetrievalMode::Hybrid,
                ..Default::default()
            };
            memory_system.retrieve(&query).expect("Failed to retrieve")
        });
    });
}

// ==============================================================================
// STEP-BY-STEP TIMING BREAKDOWN (Manual Measurement)
// ==============================================================================

fn bench_pipeline_breakdown_timing(c: &mut Criterion) {
    eprintln!("\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                           PIPELINE BREAKDOWN - DETAILED TIMING                               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    // Initialize components
    let ner = setup_ner();
    let embedder = setup_embedder();
    let (mut memory_system, _temp_dir) = setup_memory_system();

    // Warm up
    let _ = ner.extract("Warmup");
    let _ = embedder.encode("Warmup");

    // Run detailed timing for medium text
    let iterations = 10;
    let mut ner_times = Vec::new();
    let mut embed_times = Vec::new();
    let mut store_times = Vec::new();
    let mut retrieve_times = Vec::new();

    for _ in 0..iterations {
        // NER timing
        let start = Instant::now();
        let entities = ner.extract(MEDIUM_TEXT).unwrap_or_default();
        ner_times.push(start.elapsed());

        // Embedding timing
        let start = Instant::now();
        let _embedding = embedder.encode(MEDIUM_TEXT).expect("Failed");
        embed_times.push(start.elapsed());

        // Storage timing (includes internal embedding + indexing)
        let start = Instant::now();
        let exp = Experience {
            content: format!(
                "Test memory with entities: {:?}",
                entities.iter().map(|e| &e.text).collect::<Vec<_>>()
            ),
            experience_type: ExperienceType::Observation,
            entities: entities.iter().map(|e| e.text.clone()).collect(),
            ..Default::default()
        };
        memory_system.record(exp).expect("Failed to record");
        store_times.push(start.elapsed());

        // Retrieval timing
        let start = Instant::now();
        let query = Query {
            query_text: Some("technology partnership Microsoft".to_string()),
            max_results: 5,
            retrieval_mode: RetrievalMode::Similarity,
            ..Default::default()
        };
        let _ = memory_system.retrieve(&query);
        retrieve_times.push(start.elapsed());
    }

    // Calculate averages
    let avg_ner = ner_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / iterations as f64;
    let avg_embed = embed_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / iterations as f64;
    let avg_store = store_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / iterations as f64;
    let avg_retrieve = retrieve_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / iterations as f64;

    let total = avg_ner + avg_embed + avg_store + avg_retrieve;

    // Print breakdown
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           PIPELINE STEP-BY-STEP BREAKDOWN                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  STEP                        │  AVG TIME     │  % OF TOTAL  │  NOTES                         ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  1. NER Extraction           │  {:>8.2}ms   │   {:>5.1}%     │  bert-tiny-NER (~17MB)        ║", avg_ner, (avg_ner / total) * 100.0);
    println!("║  2. Embedding Generation     │  {:>8.2}ms   │   {:>5.1}%     │  MiniLM-L6-v2 (384-dim)       ║", avg_embed, (avg_embed / total) * 100.0);
    println!("║  3. Memory Storage           │  {:>8.2}ms   │   {:>5.1}%     │  RocksDB + HNSW Index         ║", avg_store, (avg_store / total) * 100.0);
    println!("║  4. Memory Retrieval         │  {:>8.2}ms   │   {:>5.1}%     │  Embed + Vector Search        ║", avg_retrieve, (avg_retrieve / total) * 100.0);
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  TOTAL PIPELINE              │  {:>8.2}ms   │   100.0%     │  Record + Retrieve cycle      ║", total);
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Dummy benchmark to satisfy criterion
    c.bench_function("zzz_breakdown_summary", |b| {
        b.iter(|| std::hint::black_box(1 + 1))
    });

    // Print recommendations
    print_optimization_recommendations(avg_ner, avg_embed, avg_store, avg_retrieve);
}

fn print_optimization_recommendations(ner_ms: f64, embed_ms: f64, store_ms: f64, retrieve_ms: f64) {
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           OPTIMIZATION RECOMMENDATIONS                                        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                                               ║");

    if ner_ms > 20.0 {
        println!("║  ⚠️  NER is slow ({:.1}ms). Consider:                                                         ║", ner_ms);
        println!("║      • Use fallback mode for non-critical text                                              ║");
        println!("║      • Run NER async/background                                                             ║");
    } else {
        println!("║  ✅ NER performance is good ({:.1}ms)                                                         ║", ner_ms);
    }
    println!("║                                                                                               ║");

    if embed_ms > 50.0 {
        println!("║  ⚠️  Embedding is slow ({:.1}ms). Consider:                                                   ║", embed_ms);
        println!("║      • Enable embedding cache for repeated text                                             ║");
        println!("║      • Batch embedding generation                                                           ║");
    } else {
        println!("║  ✅ Embedding performance is good ({:.1}ms)                                                   ║", embed_ms);
    }
    println!("║                                                                                               ║");

    if store_ms > 100.0 {
        println!("║  ⚠️  Storage is slow ({:.1}ms). Consider:                                                     ║", store_ms);
        println!("║      • Tune RocksDB write buffer size                                                       ║");
        println!("║      • Use async writes for non-critical memories                                           ║");
    } else {
        println!("║  ✅ Storage performance is good ({:.1}ms)                                                     ║", store_ms);
    }
    println!("║                                                                                               ║");

    if retrieve_ms > 50.0 {
        println!("║  ⚠️  Retrieval is slow ({:.1}ms). Consider:                                                   ║", retrieve_ms);
        println!("║      • Cache query embeddings                                                               ║");
        println!("║      • Tune HNSW search parameters                                                          ║");
    } else {
        println!("║  ✅ Retrieval performance is good ({:.1}ms)                                                   ║", retrieve_ms);
    }
    println!("║                                                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Print final summary table
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                   EDGE DEVICE TARGETS                                         ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  STEP                    │  RASPBERRY PI 4  │  JETSON NANO  │  INTEL NUC  │  DESKTOP          ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  NER Extraction          │     50-100ms     │    20-50ms    │   10-30ms   │    5-15ms         ║");
    println!("║  Embedding Generation    │     80-150ms     │    30-60ms    │   20-40ms   │   15-35ms         ║");
    println!("║  Memory Storage          │     30-80ms      │    15-40ms    │   10-25ms   │    5-15ms         ║");
    println!("║  Memory Retrieval        │     40-100ms     │    20-50ms    │   10-30ms   │    5-20ms         ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  TOTAL (Record+Retrieve) │    200-430ms     │   85-200ms    │  50-125ms   │  30-85ms          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

// ==============================================================================
// CRITERION CONFIGURATION
// ==============================================================================

criterion_group!(
    name = pipeline_benches;
    config = Criterion::default()
        .sample_size(30)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_pipeline_step1_ner,
        bench_pipeline_step2_embedding,
        bench_pipeline_step3_ner_embedding_combined,
        bench_pipeline_step4_storage,
        bench_pipeline_step5_retrieval,
        bench_full_pipeline_end_to_end,
        bench_pipeline_breakdown_timing
);

criterion_main!(pipeline_benches);
