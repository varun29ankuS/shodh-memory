//! Timing SLA Tests - Performance Contract Enforcement
//!
//! These tests enforce performance SLAs to prevent latency regressions.
//! They measure P50, P90, and P99 latencies for core operations.
//!
//! Run with: cargo test --test timing_sla_tests -- --test-threads=1
//! Note: Run single-threaded for accurate timing measurements.

use std::time::{Duration, Instant};
use tempfile::TempDir;

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::types::{Experience, ExperienceType, GeoFilter, Query};
use shodh_memory::memory::{MemoryConfig, MemorySystem};

// ============================================================================
// SLA THRESHOLDS (in milliseconds)
// These are the maximum acceptable latencies for each operation.
// Note: These thresholds are set for debug builds. Release builds are ~10x faster.
// ============================================================================

const RECORD_P50_MS: u128 = 500; // 500ms P50 for record (debug mode)
const RECORD_P99_MS: u128 = 3000; // 3s P99 for record (debug mode, first record loads model)
const RETRIEVE_P50_MS: u128 = 200; // 200ms P50 for retrieve (debug mode)
const RETRIEVE_P99_MS: u128 = 1000; // 1s P99 for retrieve (debug mode)
const BATCH_100_MAX_MS: u128 = 60000; // 60s max for 100 records (debug mode)
const INDEX_OP_MAX_MS: u128 = 10; // 10ms max for single index op (debug mode)
const STATS_MAX_MS: u128 = 50; // 50ms max for stats (debug mode)

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create fallback NER for testing (rule-based, no ONNX required)
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Create experience with NER entity extraction
fn create_experience_with_ner(content: &str, ner: &NeuralNer) -> Experience {
    let entities = ner.extract(content).unwrap_or_default();
    let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        entities: entity_names,
        ..Default::default()
    }
}

fn create_test_config(temp_dir: &TempDir) -> MemoryConfig {
    MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 30,
        importance_threshold: 0.3,
    }
}

fn create_experience(content: &str) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        entities: vec!["timing_test".to_string()],
        ..Default::default()
    }
}

fn create_rich_experience(i: usize) -> Experience {
    Experience {
        content: format!(
            "This is a comprehensive test memory {} containing multiple \
             sentences about machine learning, neural networks, and AI. \
             It includes technical terms like transformers, attention mechanisms, \
             and gradient descent optimization algorithms.",
            i
        ),
        experience_type: ExperienceType::Learning,
        entities: vec![
            "machine_learning".to_string(),
            "neural_networks".to_string(),
            format!("topic_{}", i % 5),
        ],
        robot_id: Some(format!("robot_{}", i % 10)),
        ..Default::default()
    }
}

/// Create rich experience with NER entity extraction
fn create_rich_experience_with_ner(i: usize, ner: &NeuralNer) -> Experience {
    let content = format!(
        "CEO Satya Nadella at Microsoft headquarters in Seattle discussed {} \
         with Sundar Pichai from Google about neural networks and AI transformers. \
         Dr. Yann LeCun at Meta Research in New York presented gradient descent algorithms.",
        i
    );
    let entities = ner.extract(&content).unwrap_or_default();
    let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
    Experience {
        content,
        experience_type: ExperienceType::Learning,
        entities: entity_names,
        robot_id: Some(format!("robot_{}", i % 10)),
        ..Default::default()
    }
}

/// Calculate percentile from sorted durations
fn percentile(sorted_durations: &[u128], p: f64) -> u128 {
    if sorted_durations.is_empty() {
        return 0;
    }
    let index = ((p / 100.0) * (sorted_durations.len() - 1) as f64).round() as usize;
    sorted_durations[index]
}

/// Collect timing samples for an operation
fn benchmark_operation<F>(op: F, iterations: usize) -> Vec<u128>
where
    F: Fn() -> (),
{
    let mut durations = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        op();
        durations.push(start.elapsed().as_millis());
    }
    durations.sort();
    durations
}

// ============================================================================
// RECORD OPERATION SLA TESTS
// ============================================================================

#[test]
fn test_sla_record_p50_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Warm up
    for i in 0..5 {
        let exp = create_experience(&format!("Warmup {}", i));
        let _ = system.record(exp);
    }

    // Measure
    let mut durations = Vec::with_capacity(20);
    for i in 0..20 {
        let exp = create_experience(&format!("Timing test {}", i));
        let start = Instant::now();
        let _ = system.record(exp);
        durations.push(start.elapsed().as_millis());
    }

    durations.sort();
    let p50 = percentile(&durations, 50.0);

    assert!(
        p50 <= RECORD_P50_MS,
        "SLA VIOLATION: Record P50 latency {}ms exceeds threshold {}ms",
        p50,
        RECORD_P50_MS
    );

    println!("Record P50: {}ms (threshold: {}ms)", p50, RECORD_P50_MS);
}

#[test]
fn test_sla_record_p99_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Warm up
    for i in 0..5 {
        let exp = create_experience(&format!("Warmup {}", i));
        let _ = system.record(exp);
    }

    // Measure more samples for P99
    let mut durations = Vec::with_capacity(50);
    for i in 0..50 {
        let exp = create_rich_experience(i);
        let start = Instant::now();
        let _ = system.record(exp);
        durations.push(start.elapsed().as_millis());
    }

    durations.sort();
    let p99 = percentile(&durations, 99.0);

    assert!(
        p99 <= RECORD_P99_MS,
        "SLA VIOLATION: Record P99 latency {}ms exceeds threshold {}ms",
        p99,
        RECORD_P99_MS
    );

    println!("Record P99: {}ms (threshold: {}ms)", p99, RECORD_P99_MS);
}

// ============================================================================
// RETRIEVE OPERATION SLA TESTS
// ============================================================================

#[test]
fn test_sla_retrieve_p50_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Populate with test data
    for i in 0..50 {
        let exp = create_rich_experience(i);
        let _ = system.record(exp);
    }

    // Warm up retrieval
    for _ in 0..3 {
        let query = Query {
            query_text: Some("machine learning".to_string()),
            max_results: 10,
            ..Default::default()
        };
        let _ = system.retrieve(&query);
    }

    // Measure
    let queries = [
        "machine learning neural networks",
        "transformers attention",
        "gradient descent optimization",
        "AI algorithms",
    ];

    let mut durations = Vec::with_capacity(20);
    for i in 0..20 {
        let query = Query {
            query_text: Some(queries[i % queries.len()].to_string()),
            max_results: 10,
            ..Default::default()
        };
        let start = Instant::now();
        let _ = system.retrieve(&query);
        durations.push(start.elapsed().as_millis());
    }

    durations.sort();
    let p50 = percentile(&durations, 50.0);

    assert!(
        p50 <= RETRIEVE_P50_MS,
        "SLA VIOLATION: Retrieve P50 latency {}ms exceeds threshold {}ms",
        p50,
        RETRIEVE_P50_MS
    );

    println!("Retrieve P50: {}ms (threshold: {}ms)", p50, RETRIEVE_P50_MS);
}

#[test]
fn test_sla_retrieve_p99_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Populate with more data
    for i in 0..100 {
        let exp = create_rich_experience(i);
        let _ = system.record(exp);
    }

    // Measure with varied queries
    let mut durations = Vec::with_capacity(50);
    for i in 0..50 {
        let query = Query {
            query_text: Some(format!("test query {} machine learning", i)),
            max_results: 20,
            importance_threshold: Some(0.1),
            ..Default::default()
        };
        let start = Instant::now();
        let _ = system.retrieve(&query);
        durations.push(start.elapsed().as_millis());
    }

    durations.sort();
    let p99 = percentile(&durations, 99.0);

    assert!(
        p99 <= RETRIEVE_P99_MS,
        "SLA VIOLATION: Retrieve P99 latency {}ms exceeds threshold {}ms",
        p99,
        RETRIEVE_P99_MS
    );

    println!("Retrieve P99: {}ms (threshold: {}ms)", p99, RETRIEVE_P99_MS);
}

// ============================================================================
// BATCH OPERATION SLA TESTS
// ============================================================================

#[test]
fn test_sla_batch_100_records() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    let start = Instant::now();
    for i in 0..100 {
        let exp = create_rich_experience(i);
        system.record(exp).expect("Failed to record");
    }
    let duration = start.elapsed().as_millis();

    assert!(
        duration <= BATCH_100_MAX_MS,
        "SLA VIOLATION: Batch 100 records took {}ms, threshold {}ms",
        duration,
        BATCH_100_MAX_MS
    );

    let avg = duration / 100;
    println!(
        "Batch 100 records: {}ms total, {}ms avg (threshold: {}ms total)",
        duration, avg, BATCH_100_MAX_MS
    );
}

#[test]
fn test_sla_throughput_sustained() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Measure sustained throughput over 200 records
    let start = Instant::now();
    let mut success_count = 0;

    for i in 0..200 {
        let exp = create_experience(&format!("Throughput test {}", i));
        if system.record(exp).is_ok() {
            success_count += 1;
        }
    }

    let duration_secs = start.elapsed().as_secs_f64();
    let throughput = success_count as f64 / duration_secs;

    // Should sustain at least 5 records/sec (debug mode)
    assert!(
        throughput >= 5.0,
        "SLA VIOLATION: Throughput {:.1} records/sec below 5 records/sec minimum (debug mode)",
        throughput
    );

    println!(
        "Sustained throughput: {:.1} records/sec ({} records in {:.2}s)",
        throughput, success_count, duration_secs
    );
}

// ============================================================================
// STATS OPERATION SLA TEST
// ============================================================================

#[test]
fn test_sla_stats_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Add some data
    for i in 0..50 {
        let exp = create_experience(&format!("Stats test {}", i));
        let _ = system.record(exp);
    }

    // Measure stats latency
    let mut durations = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        let _ = system.stats();
        durations.push(start.elapsed().as_millis());
    }

    durations.sort();
    let p99 = percentile(&durations, 99.0);

    assert!(
        p99 <= STATS_MAX_MS,
        "SLA VIOLATION: Stats P99 latency {}ms exceeds threshold {}ms",
        p99,
        STATS_MAX_MS
    );

    println!("Stats P99: {}ms (threshold: {}ms)", p99, STATS_MAX_MS);
}

// ============================================================================
// COLD START LATENCY TEST
// ============================================================================

#[test]
fn test_sla_cold_start_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    // Measure cold start (system initialization)
    let start = Instant::now();
    let system = MemorySystem::new(config).expect("Failed to create memory system");
    let init_duration = start.elapsed().as_millis();

    // Cold start should be < 2s (debug mode)
    assert!(
        init_duration <= 2000,
        "SLA VIOLATION: Cold start took {}ms, threshold 2000ms",
        init_duration
    );

    // First record after cold start loads the embedding model, so give it extra time
    let exp = create_experience("First record after cold start");
    let start = Instant::now();
    system.record(exp).expect("Failed to record");
    let first_record_duration = start.elapsed().as_millis();

    // First record can take longer due to model loading (5s threshold)
    assert!(
        first_record_duration <= 5000,
        "SLA VIOLATION: First record after cold start took {}ms, threshold 5000ms",
        first_record_duration
    );

    println!(
        "Cold start: {}ms, First record: {}ms",
        init_duration, first_record_duration
    );
}

// ============================================================================
// FLUSH OPERATION SLA TEST
// ============================================================================

#[test]
fn test_sla_flush_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Add data
    for i in 0..100 {
        let exp = create_experience(&format!("Flush test {}", i));
        let _ = system.record(exp);
    }

    // Measure flush latency
    let start = Instant::now();
    system.flush_storage().expect("Failed to flush");
    let duration = start.elapsed().as_millis();

    // Flush of 100 memories should be < 500ms
    assert!(
        duration <= 500,
        "SLA VIOLATION: Flush of 100 memories took {}ms, threshold 500ms",
        duration
    );

    println!("Flush 100 memories: {}ms", duration);
}

// ============================================================================
// CONCURRENT ACCESS LATENCY TEST
// ============================================================================

#[test]
fn test_sla_concurrent_access_latency() {
    use std::sync::Arc;
    use std::thread;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = Arc::new(MemorySystem::new(config).expect("Failed to create memory system"));

    // Pre-populate
    for i in 0..50 {
        let exp = create_experience(&format!("Pre-populate {}", i));
        let _ = system.record(exp);
    }

    // Simulate concurrent access (alternating reads and writes)
    let mut total_duration = 0u128;
    let ops = 40;

    for i in 0..ops {
        if i % 2 == 0 {
            // Write
            let exp = create_experience(&format!("Concurrent write {}", i));
            let start = Instant::now();
            let _ = system.record(exp);
            total_duration += start.elapsed().as_millis();
        } else {
            // Read
            let query = Query {
                query_text: Some("Pre-populate".to_string()),
                max_results: 5,
                ..Default::default()
            };
            let start = Instant::now();
            let _ = system.retrieve(&query);
            total_duration += start.elapsed().as_millis();
        }
    }

    let avg = total_duration / ops as u128;

    // Average concurrent operation should be < 200ms in debug mode under load
    // (When run alone: ~17ms, but parallel test execution adds contention)
    assert!(
        avg <= 200,
        "SLA VIOLATION: Average concurrent operation {}ms exceeds 200ms threshold",
        avg
    );

    println!(
        "Concurrent access: {} ops, {}ms total, {}ms avg",
        ops, total_duration, avg
    );
}

// ============================================================================
// GEO FILTER LATENCY TEST
// ============================================================================

#[test]
fn test_sla_geo_filter_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Add geo-tagged memories
    for i in 0..30 {
        let lat = 37.7749 + (i as f64 * 0.001);
        let lon = -122.4194 + (i as f64 * 0.001);
        let exp = Experience {
            content: format!("Geo location test {}", i),
            experience_type: ExperienceType::Observation,
            geo_location: Some([lat, lon, 0.0]),
            ..Default::default()
        };
        let _ = system.record(exp);
    }

    // Measure geo-filtered retrieval
    let mut durations = Vec::with_capacity(10);
    for i in 0..10 {
        let query = Query {
            query_text: Some("Geo location test".to_string()),
            geo_filter: Some(GeoFilter::new(
                37.7749,
                -122.4194,
                1000.0 * (i as f64 + 1.0),
            )),
            max_results: 10,
            ..Default::default()
        };
        let start = Instant::now();
        let _ = system.retrieve(&query);
        durations.push(start.elapsed().as_millis());
    }

    durations.sort();
    let p50 = percentile(&durations, 50.0);

    // Geo-filtered retrieval should be < 150ms P50
    assert!(
        p50 <= 150,
        "SLA VIOLATION: Geo-filtered retrieval P50 {}ms exceeds 150ms threshold",
        p50
    );

    println!("Geo-filtered retrieval P50: {}ms", p50);
}

// ============================================================================
// MAINTENANCE OPERATION LATENCY TEST
// ============================================================================

#[test]
fn test_sla_maintenance_latency() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    // Add memories
    for i in 0..100 {
        let exp = create_rich_experience(i);
        let _ = system.record(exp);
    }

    // Measure maintenance operation
    let start = Instant::now();
    let processed = system.run_maintenance(0.95).expect("Maintenance failed");
    let duration = start.elapsed().as_millis();

    // Maintenance of 100 memories should be < 1s
    assert!(
        duration <= 1000,
        "SLA VIOLATION: Maintenance of {} memories took {}ms, threshold 1000ms",
        processed,
        duration
    );

    println!(
        "Maintenance: {} memories processed in {}ms",
        processed, duration
    );
}

// ============================================================================
// PERFORMANCE SUMMARY TEST
// ============================================================================

#[test]
fn test_performance_summary() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");

    println!("\n=== Performance Summary ===\n");

    // Record performance
    let mut record_durations = Vec::with_capacity(30);
    for i in 0..30 {
        let exp = create_rich_experience(i);
        let start = Instant::now();
        let _ = system.record(exp);
        record_durations.push(start.elapsed().as_millis());
    }
    record_durations.sort();

    println!("RECORD Operations (30 samples):");
    println!(
        "  P50: {}ms (threshold: {}ms)",
        percentile(&record_durations, 50.0),
        RECORD_P50_MS
    );
    println!("  P90: {}ms", percentile(&record_durations, 90.0));
    println!(
        "  P99: {}ms (threshold: {}ms)",
        percentile(&record_durations, 99.0),
        RECORD_P99_MS
    );

    // Retrieve performance
    let mut retrieve_durations = Vec::with_capacity(30);
    for i in 0..30 {
        let query = Query {
            query_text: Some(format!("test query {}", i)),
            max_results: 10,
            ..Default::default()
        };
        let start = Instant::now();
        let _ = system.retrieve(&query);
        retrieve_durations.push(start.elapsed().as_millis());
    }
    retrieve_durations.sort();

    println!("\nRETRIEVE Operations (30 samples):");
    println!(
        "  P50: {}ms (threshold: {}ms)",
        percentile(&retrieve_durations, 50.0),
        RETRIEVE_P50_MS
    );
    println!("  P90: {}ms", percentile(&retrieve_durations, 90.0));
    println!(
        "  P99: {}ms (threshold: {}ms)",
        percentile(&retrieve_durations, 99.0),
        RETRIEVE_P99_MS
    );

    // Stats
    let stats = system.stats();
    println!("\nMEMORY Stats:");
    println!("  Total: {}", stats.total_memories);
    println!("  Working: {}", stats.working_memory_count);
    println!("  Session: {}", stats.session_memory_count);
    println!("  Long-term: {}", stats.long_term_memory_count);

    println!("\n=== End Summary ===\n");
}
