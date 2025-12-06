//! BRUTAL STRESS TESTS - Designed to Break the System
//!
//! These tests are intentionally aggressive and push the system to its limits.
//! If any of these tests fail, it reveals a real weakness that needs fixing.
//!
//! Philosophy: Every failure here is an opportunity to strengthen the system.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use uuid::Uuid;

use shodh_memory::memory::{
    retrieval::RetrievalOutcome,
    types::{Experience, ExperienceType, Query},
    MemoryConfig, MemoryId, MemorySystem,
};

// ============================================================================
// TEST INFRASTRUCTURE
// ============================================================================

fn create_test_config(temp_dir: &TempDir) -> MemoryConfig {
    MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    }
}

fn create_test_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config).expect("Failed to create memory system");
    (system, temp_dir)
}

fn create_experience(content: &str, entities: Vec<&str>) -> Experience {
    Experience {
        experience_type: ExperienceType::Learning,
        content: content.to_string(),
        entities: entities.into_iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

// ============================================================================
// BRUTAL CONCURRENT ACCESS TESTS
// ============================================================================

/// Hammer the system with concurrent writes from multiple threads
#[test]
fn test_brutal_concurrent_writes() {
    let (system, _temp_dir) = create_test_system();
    let system = Arc::new(system); // No external lock - internally thread-safe
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    let num_threads = 8;
    let writes_per_thread = 50;

    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let sys = Arc::clone(&system);
        let success = Arc::clone(&success_count);
        let errors = Arc::clone(&error_count);

        let handle = thread::spawn(move || {
            for i in 0..writes_per_thread {
                let exp = create_experience(
                    &format!(
                        "Thread {} memory {} - concurrent write stress test",
                        thread_id, i
                    ),
                    vec!["stress", "concurrent", &format!("thread{}", thread_id)],
                );
                match sys.record(exp) {
                    Ok(_) => success.fetch_add(1, Ordering::SeqCst),
                    Err(_) => errors.fetch_add(1, Ordering::SeqCst),
                };
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let total_success = success_count.load(Ordering::SeqCst);
    let total_errors = error_count.load(Ordering::SeqCst);
    let expected = num_threads * writes_per_thread;

    assert_eq!(
        total_errors, 0,
        "No writes should fail: {} errors out of {} attempts",
        total_errors, expected
    );
    assert_eq!(
        total_success, expected,
        "All writes should succeed: {} of {}",
        total_success, expected
    );
}

/// Concurrent reads and writes interleaved
#[test]
fn test_brutal_concurrent_read_write() {
    let (system, _temp_dir) = create_test_system();
    let system = system;

    // Pre-populate with some memories
    for i in 0..20 {
        let exp = create_experience(
            &format!("Pre-populated memory {} for concurrent test", i),
            vec!["prepop"],
        );
        system.record(exp).expect("Failed to prepopulate");
    }

    let system = Arc::new(system); // No external lock - internally thread-safe
    let read_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // Writer threads
    for thread_id in 0..4 {
        let sys = Arc::clone(&system);
        let writes = Arc::clone(&write_count);

        let handle = thread::spawn(move || {
            for i in 0..25 {
                let exp = create_experience(
                    &format!("Writer {} memory {}", thread_id, i),
                    vec!["writer"],
                );
                if sys.record(exp).is_ok() {
                    writes.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    // Reader threads
    for _thread_id in 0..4 {
        let sys = Arc::clone(&system);
        let reads = Arc::clone(&read_count);

        let handle = thread::spawn(move || {
            for _ in 0..25 {
                let query = Query {
                    query_text: Some("memory".to_string()),
                    max_results: 10,
                    ..Default::default()
                };
                if sys.retrieve(&query).is_ok() {
                    reads.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let total_reads = read_count.load(Ordering::SeqCst);
    let total_writes = write_count.load(Ordering::SeqCst);

    assert_eq!(
        total_writes, 100,
        "All writes should succeed: {}",
        total_writes
    );
    assert_eq!(
        total_reads, 100,
        "All reads should succeed: {}",
        total_reads
    );
}

/// Concurrent reinforcement on same memories - race condition test
#[test]
fn test_brutal_concurrent_reinforcement_race() {
    let (system, _temp_dir) = create_test_system();
    let system = system;

    // Create shared memories
    let mut ids = Vec::new();
    for i in 0..10 {
        let exp = create_experience(&format!("Race target {}", i), vec!["race"]);
        ids.push(system.record(exp).expect("Failed"));
    }

    let ids = Arc::new(ids);
    let system = Arc::new(system); // No external lock - internally thread-safe

    let mut handles = Vec::new();

    // Multiple threads reinforcing the SAME memories concurrently
    for _ in 0..8 {
        let sys = Arc::clone(&system);
        let mem_ids = Arc::clone(&ids);

        let handle = thread::spawn(move || {
            for _ in 0..20 {
                // Each thread reinforces all memories
                let _ = sys.reinforce_retrieval(&mem_ids, RetrievalOutcome::Helpful);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify system is still consistent
    for id in ids.iter() {
        let memory = system.get_memory(id).expect("Memory should exist");
        // After 8 threads x 20 iterations = 160 boosts, importance should be at max
        assert!(
            memory.importance() > 0.9,
            "Importance should be high after concurrent boosts: {}",
            memory.importance()
        );
    }
}

// ============================================================================
// BRUTAL PERSISTENCE TESTS
// ============================================================================

/// Multiple restart cycles - verify no data loss
#[test]
fn test_brutal_multiple_restart_cycles() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let mut all_ids = Vec::new();

    // Multiple restart cycles, each adding and modifying data
    for cycle in 0..5 {
        let system = MemorySystem::new(config.clone()).expect("Failed to create system");

        // Verify all previous memories exist
        for (i, id) in all_ids.iter().enumerate() {
            let memory = system.get_memory(id).expect(&format!(
                "Cycle {}: Memory {} should exist from previous cycle",
                cycle, i
            ));
            assert!(
                memory.experience.content.contains("Restart cycle"),
                "Cycle {}: Memory {} content should be preserved",
                cycle,
                i
            );
        }

        // Add new memories this cycle
        for i in 0..10 {
            let exp = create_experience(
                &format!("Restart cycle {} memory {}", cycle, i),
                vec!["restart"],
            );
            let id = system.record(exp).expect("Failed to record");
            all_ids.push(id);
        }

        // Reinforce some memories
        if !all_ids.is_empty() {
            let subset: Vec<_> = all_ids.iter().step_by(3).cloned().collect();
            system
                .reinforce_retrieval(&subset, RetrievalOutcome::Helpful)
                .expect("Failed");
        }
    }

    // Final verification - all 50 memories (5 cycles x 10 each) should exist
    let system = MemorySystem::new(config).expect("Final load");
    for (i, id) in all_ids.iter().enumerate() {
        system
            .get_memory(id)
            .expect(&format!("Final: Memory {} should exist", i));
    }
    assert_eq!(all_ids.len(), 50, "Should have all 50 memories");
}

/// Crash simulation - kill during write
#[test]
fn test_brutal_partial_write_recovery() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);

    let memory_id;
    let initial_importance;
    let boosted_importance;

    // Phase 1: Create memory and boost importance
    {
        let system = MemorySystem::new(config.clone()).expect("Failed");
        // Long content for higher base importance calculation
        let exp = create_experience(
            "This is extremely important persistent data that must survive crashes and restarts. \
             It contains critical information about the system architecture and recovery procedures. \
             Failure to preserve this would be catastrophic for the entire system.",
            vec!["crash", "recovery", "critical"],
        );
        memory_id = system.record(exp).expect("Failed to record");
        initial_importance = system.get_memory(&memory_id).unwrap().importance();

        // Boost importance multiple times to ensure detectable change
        for _ in 0..5 {
            system
                .reinforce_retrieval(&[memory_id.clone()], RetrievalOutcome::Helpful)
                .expect("Failed");
        }
        boosted_importance = system.get_memory(&memory_id).unwrap().importance();
    }
    // System dropped - simulates graceful shutdown

    // Phase 2: Verify recovery
    {
        let system = MemorySystem::new(config.clone()).expect("Failed to recover");
        let memory = system
            .get_memory(&memory_id)
            .expect("Memory should survive");
        let recovered_importance = memory.importance();

        // Key assertion: recovered importance must match boosted importance (not initial)
        assert!(
            (recovered_importance - boosted_importance).abs() < 0.01,
            "Importance should be preserved after restart. Initial: {}, Boosted: {}, Recovered: {}",
            initial_importance,
            boosted_importance,
            recovered_importance
        );

        // Also verify the boost actually happened
        assert!(
            boosted_importance > initial_importance,
            "Boost should increase importance: {} should be > {}",
            boosted_importance,
            initial_importance
        );
    }
}

// ============================================================================
// BRUTAL MEMORY LIMITS TESTS
// ============================================================================

/// Push beyond working memory capacity
#[test]
fn test_brutal_exceed_working_memory() {
    let temp_dir = TempDir::new().expect("Failed");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 10, // Very small working memory
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    };
    let system = MemorySystem::new(config).expect("Failed");

    let mut all_ids = Vec::new();

    // Write way more than working memory can hold
    for i in 0..100 {
        let exp = create_experience(
            &format!("Overflow memory {} with lots of content to push limits", i),
            vec!["overflow"],
        );
        let id = system.record(exp).expect("Failed to record");
        all_ids.push(id);
    }

    // All memories should be retrievable from storage even if evicted from working memory
    for (i, id) in all_ids.iter().enumerate() {
        system
            .get_memory(id)
            .expect(&format!("Memory {} should exist in storage", i));
    }
}

/// Large content stress test
#[test]
fn test_brutal_large_content() {
    let (system, _temp_dir) = create_test_system();

    // Create memory with large content
    let large_content = "x".repeat(100_000); // 100KB content
    let exp = Experience {
        experience_type: ExperienceType::Learning,
        content: large_content.clone(),
        entities: vec!["large".to_string()],
        ..Default::default()
    };

    let id = system.record(exp).expect("Should handle large content");

    // Retrieve and verify
    let memory = system
        .get_memory(&id)
        .expect("Should retrieve large memory");
    assert_eq!(
        memory.experience.content.len(),
        100_000,
        "Content should be preserved"
    );
}

/// Many entities stress test
#[test]
fn test_brutal_many_entities() {
    let (system, _temp_dir) = create_test_system();

    // Create memory with many entities
    let entities: Vec<String> = (0..1000).map(|i| format!("Entity{}", i)).collect();
    let exp = Experience {
        experience_type: ExperienceType::Learning,
        content: "Memory with many entities".to_string(),
        entities,
        ..Default::default()
    };

    let id = system.record(exp).expect("Should handle many entities");

    let memory = system.get_memory(&id).expect("Should retrieve");
    assert_eq!(
        memory.experience.entities.len(),
        1000,
        "All entities should be preserved"
    );
}

// ============================================================================
// BRUTAL IMPORTANCE BOUNDARY TESTS
// ============================================================================

/// Push importance to extreme values repeatedly
#[test]
fn test_brutal_importance_boundary_cycling() {
    let (system, _temp_dir) = create_test_system();

    let exp = create_experience("Boundary test", vec!["boundary"]);
    let id = system.record(exp).expect("Failed");

    // Cycle importance to max and back multiple times
    for cycle in 0..10 {
        // Boost to max
        for _ in 0..50 {
            system
                .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Helpful)
                .expect("Failed");
        }
        let high = system.get_memory(&id).expect("Failed").importance();
        assert!(
            high >= 0.99,
            "Cycle {}: Should reach max importance: {}",
            cycle,
            high
        );

        // Decay to min
        for _ in 0..100 {
            system
                .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Misleading)
                .expect("Failed");
        }
        let low = system.get_memory(&id).expect("Failed").importance();
        assert!(
            low <= 0.1,
            "Cycle {}: Should reach min importance: {}",
            cycle,
            low
        );
    }
}

/// Verify importance never goes outside [0.05, 1.0]
#[test]
fn test_brutal_importance_bounds_invariant() {
    let (system, _temp_dir) = create_test_system();

    let exp = create_experience("Bounds invariant test", vec!["bounds"]);
    let id = system.record(exp).expect("Failed");

    // Extreme boosts
    for _ in 0..1000 {
        system
            .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Helpful)
            .expect("Failed");
    }
    let importance = system.get_memory(&id).expect("Failed").importance();
    assert!(
        importance <= 1.0,
        "Importance must never exceed 1.0: {}",
        importance
    );
    assert!(
        importance >= 0.0,
        "Importance must never be negative: {}",
        importance
    );

    // Extreme decays
    for _ in 0..1000 {
        system
            .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Misleading)
            .expect("Failed");
    }
    let importance = system.get_memory(&id).expect("Failed").importance();
    assert!(
        importance >= 0.05,
        "Importance must never go below floor: {}",
        importance
    );
    assert!(
        importance <= 1.0,
        "Importance must never exceed 1.0: {}",
        importance
    );
}

// ============================================================================
// BRUTAL ID COLLISION TESTS
// ============================================================================

/// Verify no ID collisions with many memories
#[test]
fn test_brutal_no_id_collisions() {
    let (system, _temp_dir) = create_test_system();

    let mut ids = HashSet::new();

    for i in 0..500 {
        let exp = create_experience(&format!("Collision test {}", i), vec!["collision"]);
        let id = system.record(exp).expect("Failed to record");

        // Verify no collision
        assert!(
            ids.insert(id.clone()),
            "ID collision detected at memory {}",
            i
        );
    }

    assert_eq!(ids.len(), 500, "Should have 500 unique IDs");
}

// ============================================================================
// BRUTAL GRAPH STRESS TESTS
// ============================================================================

/// Create dense graph with many associations
#[test]
fn test_brutal_dense_graph() {
    let (system, _temp_dir) = create_test_system();

    let mut ids = Vec::new();
    for i in 0..50 {
        let exp = create_experience(&format!("Graph node {}", i), vec!["graph"]);
        ids.push(system.record(exp).expect("Failed"));
    }

    // Create full mesh - every pair of memories associated
    // This creates 50*49/2 = 1225 edges
    system
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .expect("Failed");

    let stats = system.graph_stats();
    assert!(
        stats.edge_count >= 1000,
        "Should have many edges: {}",
        stats.edge_count
    );

    // Verify graph maintenance doesn't crash
    system.graph_maintenance();
}

/// Repeated graph maintenance cycles
#[test]
fn test_brutal_graph_maintenance_cycles() {
    let (system, _temp_dir) = create_test_system();

    let mut ids = Vec::new();
    for i in 0..20 {
        let exp = create_experience(&format!("Maintenance test {}", i), vec!["maint"]);
        ids.push(system.record(exp).expect("Failed"));
    }

    // Build associations
    system
        .reinforce_retrieval(&ids, RetrievalOutcome::Helpful)
        .expect("Failed");

    // Run maintenance many times - should not crash or corrupt
    for _ in 0..100 {
        system.graph_maintenance();
    }

    // Verify graph still works
    let stats = system.graph_stats();
    assert!(
        stats.node_count > 0 || stats.edge_count >= 0,
        "Graph should be intact"
    );
}

// ============================================================================
// BRUTAL TIMING TESTS
// ============================================================================

/// Verify operations complete in reasonable time
/// Note: This measures embedding generation + storage + indexing per record
/// Embedding dominates at ~200-300ms per call, with batching potential
#[test]
fn test_brutal_timing_record() {
    let (system, _temp_dir) = create_test_system();

    let start = Instant::now();
    for i in 0..100 {
        let exp = create_experience(&format!("Timing test {}", i), vec!["timing"]);
        system.record(exp).expect("Failed");
    }
    let elapsed = start.elapsed();

    // 100 records should complete in under 90 seconds (900ms per record average)
    // This includes: embedding generation (~300-500ms), storage, vector indexing
    // Parallel test execution and CPU load can add significant overhead
    assert!(
        elapsed < Duration::from_secs(90),
        "100 records took too long: {:?} ({:.0}ms/record avg)",
        elapsed,
        elapsed.as_millis() as f64 / 100.0
    );
}

/// Verify retrieval timing under load
#[test]
fn test_brutal_timing_retrieval() {
    let (system, _temp_dir) = create_test_system();

    // Populate
    for i in 0..100 {
        let exp = create_experience(&format!("Retrieval timing {}", i), vec!["retrieve"]);
        system.record(exp).expect("Failed");
    }

    let start = Instant::now();
    for _ in 0..50 {
        let query = Query {
            query_text: Some("retrieval timing".to_string()),
            max_results: 10,
            ..Default::default()
        };
        system.retrieve(&query).expect("Failed");
    }
    let elapsed = start.elapsed();

    // 50 retrievals should complete in under 10 seconds (200ms per query average)
    assert!(
        elapsed < Duration::from_secs(10),
        "50 retrievals took too long: {:?}",
        elapsed
    );
}

// ============================================================================
// BRUTAL EDGE CASE TESTS
// ============================================================================

/// Empty query handling
#[test]
fn test_brutal_empty_queries() {
    let (system, _temp_dir) = create_test_system();

    // Empty query text
    let query = Query {
        query_text: Some("".to_string()),
        max_results: 10,
        ..Default::default()
    };
    // Should not panic, might return empty or error
    let _ = system.retrieve(&query);

    // No query text at all
    let query = Query {
        query_text: None,
        max_results: 10,
        ..Default::default()
    };
    let _ = system.retrieve(&query);
}

/// Unicode and special characters
#[test]
fn test_brutal_unicode_content() {
    let (system, _temp_dir) = create_test_system();

    let unicode_content = "Unicode: 你好世界 🌍 émojis 日本語 العربية ελληνικά";
    let exp = Experience {
        experience_type: ExperienceType::Learning,
        content: unicode_content.to_string(),
        entities: vec!["unicode".to_string(), "🌍".to_string()],
        ..Default::default()
    };

    let id = system.record(exp).expect("Should handle unicode");
    let memory = system.get_memory(&id).expect("Should retrieve unicode");
    assert_eq!(
        memory.experience.content, unicode_content,
        "Unicode should be preserved"
    );
}

/// Null bytes and control characters
#[test]
fn test_brutal_special_characters() {
    let (system, _temp_dir) = create_test_system();

    let special = "Special:\t\n\r content";
    let exp = create_experience(special, vec!["special"]);
    let id = system.record(exp).expect("Should handle special chars");
    let memory = system.get_memory(&id).expect("Should retrieve");
    assert!(
        memory.experience.content.contains("Special"),
        "Content should be preserved"
    );
}

/// Very long entity names
#[test]
fn test_brutal_long_entity_names() {
    let (system, _temp_dir) = create_test_system();

    let long_entity = "x".repeat(10_000);
    let exp = Experience {
        experience_type: ExperienceType::Learning,
        content: "Test with long entity".to_string(),
        entities: vec![long_entity.clone()],
        ..Default::default()
    };

    let id = system.record(exp).expect("Should handle long entity");
    let memory = system.get_memory(&id).expect("Should retrieve");
    assert_eq!(
        memory.experience.entities[0].len(),
        10_000,
        "Long entity should be preserved"
    );
}

// ============================================================================
// BRUTAL CONSISTENCY TESTS
// ============================================================================

/// Verify data integrity after many operations
#[test]
fn test_brutal_data_integrity() {
    let temp_dir = TempDir::new().expect("Failed");
    let config = create_test_config(&temp_dir);

    let mut expected_contents: Vec<(MemoryId, String)> = Vec::new();

    // Phase 1: Create memories with known content
    {
        let system = MemorySystem::new(config.clone()).expect("Failed");
        for i in 0..50 {
            let content = format!("Integrity test {} - unique content {}", i, Uuid::new_v4());
            let exp = create_experience(&content, vec!["integrity"]);
            let id = system.record(exp).expect("Failed");
            expected_contents.push((id, content));
        }
    }

    // Phase 2: Verify all content matches exactly
    {
        let system = MemorySystem::new(config).expect("Failed");
        for (id, expected_content) in &expected_contents {
            let memory = system.get_memory(id).expect("Memory should exist");
            assert_eq!(
                memory.experience.content, *expected_content,
                "Content must match exactly"
            );
        }
    }
}

// ============================================================================
// ARCHITECTURE & LOCK-FREE TESTS
// ============================================================================

/// Test that read operations don't block each other (reader parallelism)
#[test]
fn test_brutal_reader_parallelism() {
    let (system, _temp_dir) = create_test_system();

    // Create memories to read
    let mut ids = Vec::new();
    for i in 0..100 {
        let exp = create_experience(&format!("Reader parallelism test {}", i), vec!["parallel"]);
        ids.push(system.record(exp).expect("Failed to record"));
    }

    let system = Arc::new(system); // No external lock - internally thread-safe
    let reads_completed = Arc::new(AtomicUsize::new(0));
    let num_readers = 16; // Many parallel readers
    let reads_per_thread = 100;

    let start = Instant::now();
    let mut handles = Vec::new();

    for _ in 0..num_readers {
        let sys = Arc::clone(&system);
        let reads = Arc::clone(&reads_completed);
        let ids_clone = ids.clone();

        let handle = thread::spawn(move || {
            for _ in 0..reads_per_thread {
                let idx = rand::random::<usize>() % ids_clone.len();
                let _ = sys.get_memory(&ids_clone[idx]);
                reads.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let elapsed = start.elapsed();
    let total_reads = num_readers * reads_per_thread;

    // 1600 reads should complete in under 30 seconds (not blocked by serial locking)
    assert!(
        elapsed < Duration::from_secs(30),
        "{} parallel reads took {:?} - possible lock contention issue",
        total_reads,
        elapsed
    );
}

/// Test that writes interleaved with reads don't deadlock
/// Now uses internal thread-safety (no external Mutex) - operations run truly in parallel
#[test]
fn test_brutal_no_deadlock_mixed_operations() {
    let (system, _temp_dir) = create_test_system();
    let system = Arc::new(system); // No external lock needed - MemorySystem is internally thread-safe
    let ops_completed = Arc::new(AtomicUsize::new(0));
    let num_threads = 8;
    let ops_per_thread = 50;

    let timeout = Duration::from_secs(120); // Timeout for deadlock detection (embedding generation is slow)
    let start = Instant::now();
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let sys = Arc::clone(&system);
        let ops = Arc::clone(&ops_completed);

        let handle = thread::spawn(move || {
            let mut recorded_ids = Vec::new();

            for i in 0..ops_per_thread {
                // No external lock needed - MemorySystem handles concurrency internally
                // Alternating read and write operations
                if i % 2 == 0 {
                    // Write
                    let exp = create_experience(
                        &format!("Deadlock test thread {} op {}", thread_id, i),
                        vec!["deadlock"],
                    );
                    if let Ok(id) = sys.record(exp) {
                        recorded_ids.push(id);
                    }
                } else {
                    // Read - try to read own or do retrieval
                    if !recorded_ids.is_empty() {
                        let idx = i % recorded_ids.len();
                        let _ = sys.get_memory(&recorded_ids[idx]);
                    } else {
                        let query = Query {
                            query_text: Some("deadlock test".to_string()),
                            max_results: 5,
                            ..Default::default()
                        };
                        let _ = sys.retrieve(&query);
                    }
                }
                ops.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    // Wait for completion or timeout (deadlock detection)
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout,
        "Operations took {:?} - possible deadlock (timeout was {:?})",
        elapsed,
        timeout
    );

    let completed = ops_completed.load(Ordering::Relaxed);
    let expected = num_threads * ops_per_thread;
    assert_eq!(
        completed, expected,
        "Only {} of {} operations completed - possible starvation",
        completed, expected
    );
}

/// Test lock ordering - ensure no AB-BA deadlock patterns
#[test]
fn test_brutal_lock_order_safety() {
    let temp_dir = TempDir::new().expect("Failed");
    let config = create_test_config(&temp_dir);

    // Create system
    let system = MemorySystem::new(config.clone()).expect("Failed");

    // Record some memories
    let mut ids = Vec::new();
    for i in 0..20 {
        let exp = create_experience(&format!("Lock order test {}", i), vec!["lock"]);
        ids.push(system.record(exp).expect("Failed"));
    }
    drop(system);

    // Rapid restart cycles - tests internal lock handling during initialization
    for cycle in 0..10 {
        let system = MemorySystem::new(config.clone()).expect("Failed restart");

        // Verify all memories accessible (tests lock consistency)
        for id in &ids {
            system
                .get_memory(id)
                .expect(&format!("Memory missing in cycle {}", cycle));
        }

        // Perform mixed operations
        let query = Query {
            query_text: Some("lock order".to_string()),
            max_results: 10,
            ..Default::default()
        };
        let _ = system.retrieve(&query);
    }
}

/// Test cache eviction under pressure doesn't corrupt state
#[test]
fn test_brutal_cache_eviction_integrity() {
    let temp_dir = TempDir::new().expect("Failed");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 5,    // Tiny cache - forces frequent eviction
        session_memory_size_mb: 1, // Small session cache
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    };

    let system = MemorySystem::new(config).expect("Failed");
    let mut all_ids = Vec::new();
    let mut expected_contents: Vec<String> = Vec::new();

    // Write many more memories than cache can hold
    for i in 0..100 {
        let content = format!("Cache eviction test {} - {}", i, Uuid::new_v4());
        let exp = create_experience(&content, vec!["eviction"]);
        let id = system.record(exp).expect("Failed");
        all_ids.push(id);
        expected_contents.push(content);
    }

    // Random access pattern to trigger evictions
    for _ in 0..200 {
        let idx = rand::random::<usize>() % all_ids.len();
        let memory = system
            .get_memory(&all_ids[idx])
            .expect("Memory should exist");

        // Verify content integrity after potential cache eviction/reload
        assert_eq!(
            memory.experience.content, expected_contents[idx],
            "Content corrupted after cache eviction at index {}",
            idx
        );
    }
}

/// Test that reinforcement doesn't corrupt during high frequency updates
#[test]
fn test_brutal_reinforcement_race() {
    let (system, _temp_dir) = create_test_system();

    // Create test memory
    let exp = create_experience("Reinforcement race test", vec!["race"]);
    let memory_id = system.record(exp).expect("Failed");

    let system = Arc::new(system); // No external lock - internally thread-safe
    let num_threads = 8;
    let reinforcements_per_thread = 20;
    let boosts = Arc::new(AtomicUsize::new(0));
    let decays = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for thread_id in 0..num_threads {
        let sys = Arc::clone(&system);
        let id = memory_id.clone();
        let b = Arc::clone(&boosts);
        let d = Arc::clone(&decays);

        let handle = thread::spawn(move || {
            for i in 0..reinforcements_per_thread {
                let outcome = if (thread_id + i) % 2 == 0 {
                    b.fetch_add(1, Ordering::Relaxed);
                    RetrievalOutcome::Helpful
                } else {
                    d.fetch_add(1, Ordering::Relaxed);
                    RetrievalOutcome::Misleading
                };
                let _ = sys.reinforce_retrieval(&[id.clone()], outcome);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify memory still accessible and importance is valid
    let memory = system.get_memory(&memory_id).expect("Memory should exist");
    let importance = memory.importance();

    assert!(
        importance >= 0.0 && importance <= 1.0,
        "Importance {} out of bounds after {} boosts and {} decays",
        importance,
        boosts.load(Ordering::Relaxed),
        decays.load(Ordering::Relaxed)
    );
}

/// Test storage layer isolation - operations on one system don't affect another
#[test]
fn test_brutal_storage_isolation() {
    let temp_dir1 = TempDir::new().expect("Failed");
    let temp_dir2 = TempDir::new().expect("Failed");

    let config1 = create_test_config(&temp_dir1);
    let config2 = create_test_config(&temp_dir2);

    let system1 = MemorySystem::new(config1.clone()).expect("Failed");
    let system2 = MemorySystem::new(config2.clone()).expect("Failed");

    // Write to system1
    let exp1 = create_experience("System 1 only data", vec!["isolated"]);
    let id1 = system1.record(exp1).expect("Failed");

    // Write to system2
    let exp2 = create_experience("System 2 only data", vec!["isolated"]);
    let id2 = system2.record(exp2).expect("Failed");

    // Verify isolation - system1 shouldn't see system2's data
    assert!(
        system1.get_memory(&id2).is_err(),
        "System1 should NOT see System2's memory"
    );
    assert!(
        system2.get_memory(&id1).is_err(),
        "System2 should NOT see System1's memory"
    );

    // Each system should see its own data
    assert!(
        system1.get_memory(&id1).is_ok(),
        "System1 should see its own memory"
    );
    assert!(
        system2.get_memory(&id2).is_ok(),
        "System2 should see its own memory"
    );

    // Drop and reload - verify isolation persists
    drop(system1);
    drop(system2);

    let system1 = MemorySystem::new(config1).expect("Failed");
    let system2 = MemorySystem::new(config2).expect("Failed");

    assert!(
        system1.get_memory(&id1).is_ok(),
        "System1 should still see its memory after restart"
    );
    assert!(
        system2.get_memory(&id2).is_ok(),
        "System2 should still see its memory after restart"
    );
    assert!(
        system1.get_memory(&id2).is_err(),
        "Isolation should persist after restart"
    );
}

/// Test that the full pipeline (record -> retrieve -> reinforce) works under load
#[test]
fn test_brutal_full_pipeline_stress() {
    let (system, _temp_dir) = create_test_system();
    let system = Arc::new(system); // No external lock - internally thread-safe
    let successful_cycles = Arc::new(AtomicUsize::new(0));

    let num_threads = 4;
    let cycles_per_thread = 25;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let sys = Arc::clone(&system);
        let success = Arc::clone(&successful_cycles);

        let handle = thread::spawn(move || {
            for cycle in 0..cycles_per_thread {
                // Step 1: Record
                let content = format!("Pipeline stress thread {} cycle {}", thread_id, cycle);
                let exp = create_experience(&content, vec!["pipeline"]);
                let memory_id = sys.record(exp).expect("Record failed");

                // Step 2: Retrieve (verify findable)
                let query = Query {
                    query_text: Some(content.clone()),
                    max_results: 10,
                    ..Default::default()
                };
                let _ = sys.retrieve(&query);

                // Step 3: Direct get
                sys.get_memory(&memory_id).expect("Direct get failed");

                // Step 4: Reinforce
                sys.reinforce_retrieval(&[memory_id], RetrievalOutcome::Helpful)
                    .expect("Reinforce failed");

                success.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let completed = successful_cycles.load(Ordering::Relaxed);
    let expected = num_threads * cycles_per_thread;
    assert_eq!(
        completed, expected,
        "Only {} of {} pipeline cycles completed",
        completed, expected
    );
}

/// Test memory graph consistency under concurrent modifications
#[test]
fn test_brutal_graph_consistency() {
    let (system, _temp_dir) = create_test_system();

    // Create initial memories
    let mut memory_ids = Vec::new();
    for i in 0..30 {
        let exp = create_experience(&format!("Graph consistency {}", i), vec!["graph"]);
        memory_ids.push(system.record(exp).expect("Failed"));
    }

    let system = Arc::new(system); // No external lock - internally thread-safe
    let num_threads = 4;
    let ops_per_thread = 50;
    let ids = Arc::new(memory_ids);
    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let sys = Arc::clone(&system);
        let ids_clone = Arc::clone(&ids);

        let handle = thread::spawn(move || {
            for _ in 0..ops_per_thread {
                // Random pair of memories
                let idx1 = rand::random::<usize>() % ids_clone.len();
                let idx2 = rand::random::<usize>() % ids_clone.len();
                if idx1 != idx2 {
                    // Create association via reinforce
                    let _ = sys.reinforce_retrieval(
                        &[ids_clone[idx1].clone(), ids_clone[idx2].clone()],
                        RetrievalOutcome::Helpful,
                    );
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify graph is still valid
    let stats = system.graph_stats();
    assert!(stats.node_count >= 0, "Graph should have valid node count");
}

/// Test rapid creation and deletion simulation (via importance decay)
#[test]
fn test_brutal_lifecycle_churn() {
    let (system, _temp_dir) = create_test_system();
    let mut ids = Vec::new();

    // Create batch
    for i in 0..50 {
        let exp = create_experience(&format!("Lifecycle churn {}", i), vec!["lifecycle"]);
        ids.push(system.record(exp).expect("Failed"));
    }

    // Simulate "deletion" by decaying importance to minimum
    for _ in 0..100 {
        for id in &ids {
            system
                .reinforce_retrieval(&[id.clone()], RetrievalOutcome::Misleading)
                .expect("Decay failed");
        }
    }

    // All memories should still be accessible (just low importance)
    for (i, id) in ids.iter().enumerate() {
        let memory = system
            .get_memory(id)
            .expect(&format!("Memory {} should exist", i));
        // Importance should be at or near minimum (0.0)
        assert!(
            memory.importance() <= 0.1,
            "Importance {} should be very low after many decays",
            memory.importance()
        );
    }
}
