//! Relevance Engine Benchmarks
//!
//! Performance measurements for proactive memory surfacing:
//! - Entity extraction (<30ms target)
//! - Full relevance pipeline
//! - Configuration impact on latency

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{Experience, ExperienceType, MemoryConfig, MemorySystem};
use shodh_memory::relevance::{RelevanceConfig, RelevanceEngine};
use std::sync::Arc;
use std::time::Duration;
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

/// Create fallback NER instance for testing
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Populate memory system with varied content for entity matching
fn populate_memories_for_relevance(memory: &mut MemorySystem, count: usize) {
    let contents = [
        "Meeting with Satya Nadella from Microsoft about Azure deployment strategies in Seattle",
        "Discussion with Sundar Pichai at Google I/O about AI integration with Gemini",
        "Technical review of Rust codebase with core team in San Francisco office",
        "Planning session for Q4 objectives with product team in New York",
        "Architecture discussion about microservices migration with engineering leads",
        "Customer feedback analysis meeting with UX researchers in London",
        "Sprint retrospective with agile team about velocity improvements",
        "Security audit findings review with compliance team in Singapore",
        "Performance optimization workshop with database engineers",
        "Feature prioritization meeting with stakeholders from marketing",
    ];

    let ner = setup_fallback_ner();
    for i in 0..count {
        let base_content = contents[i % contents.len()];
        let content = format!("{} - iteration {}", base_content, i);
        let entities = ner.extract(&content).unwrap_or_default();
        let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();

        let exp = Experience {
            content,
            experience_type: match i % 5 {
                0 => ExperienceType::Observation,
                1 => ExperienceType::Decision,
                2 => ExperienceType::Learning,
                3 => ExperienceType::Error,
                _ => ExperienceType::Pattern,
            },
            entities: entity_names,
            ..Default::default()
        };
        memory.remember(exp, None).expect("Failed to record");
    }
}

// =============================================================================
// ENTITY EXTRACTION BENCHMARKS
// =============================================================================

fn bench_entity_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_extraction");
    group.measurement_time(Duration::from_secs(10));

    let ner = setup_fallback_ner();

    // Test different context lengths
    let contexts = [
        ("short", "Meeting with Microsoft team about Azure"),
        (
            "medium",
            "Discussion with Satya Nadella from Microsoft about Azure deployment in Seattle next week",
        ),
        (
            "long",
            "We had an extensive meeting with Satya Nadella from Microsoft and Sundar Pichai from Google about cloud infrastructure and AI integration. The discussion covered Azure, GCP, Kubernetes, and various deployment strategies. Team members from San Francisco and New York participated.",
        ),
    ];

    for (name, context) in contexts.iter() {
        group.bench_with_input(BenchmarkId::new("context", name), context, |b, ctx| {
            b.iter(|| ner.extract(ctx));
        });
    }

    group.finish();
}

// =============================================================================
// FULL PIPELINE BENCHMARKS (<30ms target)
// =============================================================================

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_relevance_pipeline");
    group.measurement_time(Duration::from_secs(15));

    // Test realistic scenarios with different memory counts
    let scenarios = [
        (10, "small_db_10_memories"),
        (50, "medium_db_50_memories"),
        (100, "large_db_100_memories"),
        (200, "xlarge_db_200_memories"),
    ];

    for (memory_count, name) in scenarios.iter() {
        let (mut memory_system, _temp_dir) = setup_memory_system();
        populate_memories_for_relevance(&mut memory_system, *memory_count);

        let ner = setup_fallback_ner();
        let engine = RelevanceEngine::new(Arc::new(ner));
        let config = RelevanceConfig::default();

        group.throughput(Throughput::Elements(*memory_count as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                engine.surface_relevant(
                    "Meeting with Microsoft team about Azure deployment",
                    &memory_system,
                    None,
                    &config,
                )
            });
        });
    }

    group.finish();
}

// =============================================================================
// LATENCY VERIFICATION (<30ms)
// =============================================================================

fn bench_latency_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_30ms_target");
    group.measurement_time(Duration::from_secs(10));

    // Target scenario: 100 memories - must complete in <30ms
    let (mut memory_system, _temp_dir) = setup_memory_system();
    populate_memories_for_relevance(&mut memory_system, 100);

    let ner = setup_fallback_ner();
    let engine = RelevanceEngine::new(Arc::new(ner));
    let config = RelevanceConfig::default();

    group.bench_function("target_100_memories", |b| {
        b.iter(|| {
            engine.surface_relevant(
                "Meeting with Microsoft team about Azure deployment",
                &memory_system,
                None,
                &config,
            )
        });
    });

    // Different context lengths
    let contexts = [
        ("short_context", "Azure deployment"),
        (
            "medium_context",
            "Meeting with Microsoft team about Azure deployment strategies",
        ),
        (
            "long_context",
            "We had an extensive meeting with Satya Nadella from Microsoft and Sundar Pichai from Google about cloud infrastructure and AI integration in Seattle",
        ),
    ];

    for (name, context) in contexts.iter() {
        group.bench_function(*name, |b| {
            b.iter(|| engine.surface_relevant(context, &memory_system, None, &config));
        });
    }

    group.finish();
}

// =============================================================================
// RELEVANCE CONFIG IMPACT
// =============================================================================

fn bench_config_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_variations");
    group.measurement_time(Duration::from_secs(10));

    let (mut memory_system, _temp_dir) = setup_memory_system();
    populate_memories_for_relevance(&mut memory_system, 100);

    let ner = setup_fallback_ner();
    let engine = RelevanceEngine::new(Arc::new(ner));

    // Test different max_results values
    for max_results in [3, 5, 10, 20].iter() {
        let config = RelevanceConfig {
            max_results: *max_results,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("max_results", max_results),
            max_results,
            |b, _| {
                b.iter(|| {
                    engine.surface_relevant(
                        "Technical discussion about product features",
                        &memory_system,
                        None,
                        &config,
                    )
                });
            },
        );
    }

    // Test entity matching enabled/disabled
    for enable_entity in [true, false].iter() {
        let config = RelevanceConfig {
            enable_entity_matching: *enable_entity,
            ..Default::default()
        };

        let name = if *enable_entity {
            "entity_matching_enabled"
        } else {
            "entity_matching_disabled"
        };

        group.bench_function(name, |b| {
            b.iter(|| {
                engine.surface_relevant(
                    "Meeting with Microsoft team about Azure",
                    &memory_system,
                    None,
                    &config,
                )
            });
        });
    }

    group.finish();
}

// =============================================================================
// SEMANTIC THRESHOLD IMPACT
// =============================================================================

fn bench_threshold_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_impact");
    group.measurement_time(Duration::from_secs(10));

    let (mut memory_system, _temp_dir) = setup_memory_system();
    populate_memories_for_relevance(&mut memory_system, 100);

    let ner = setup_fallback_ner();
    let engine = RelevanceEngine::new(Arc::new(ner));

    // Test different semantic thresholds
    for threshold in [0.5, 0.65, 0.75, 0.85].iter() {
        let config = RelevanceConfig {
            semantic_threshold: *threshold,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("semantic_threshold", format!("{:.2}", threshold)),
            threshold,
            |b, _| {
                b.iter(|| {
                    engine.surface_relevant(
                        "Cloud infrastructure discussion",
                        &memory_system,
                        None,
                        &config,
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_entity_extraction,
    bench_full_pipeline,
    bench_latency_verification,
    bench_config_variations,
    bench_threshold_impact,
);
criterion_main!(benches);
