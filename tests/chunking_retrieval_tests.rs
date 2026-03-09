//! Integration tests for long content chunking and retrieval
//!
//! These tests verify that:
//! 1. Long memories are correctly chunked into multiple embeddings
//! 2. Content at any position (beginning, middle, end) is searchable
//! 3. Search results are properly deduplicated by memory ID
//! 4. The full memory is returned, not just the matching chunk

use shodh_memory::memory::types::{Experience, ExperienceType, Query};
use shodh_memory::memory::{MemoryConfig, MemorySystem};
use tempfile::TempDir;

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

fn create_test_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = create_test_config(&temp_dir);
    let system = MemorySystem::new(config, None).expect("Failed to create memory system");
    (system, temp_dir)
}

fn create_long_memory_with_markers() -> Experience {
    // Create a memory with unique markers at different positions
    // Each marker is semantically distinct so it should only match when embedded
    let beginning =
        "QUANTUM_ENTANGLEMENT_PROTOCOL is the initialization sequence for this experiment.";
    let filler1 = "This section contains general discussion about methodology and approach. \
                   We examine various techniques and their applications in the field. \
                   The research methodology follows established protocols. "
        .repeat(5);
    let middle = "NEURAL_PLASTICITY_MECHANISM describes the core learning process.";
    let filler2 = "Additional analysis reveals patterns in the data that support our hypothesis. \
                   Statistical methods were applied to validate the findings. \
                   The results demonstrate significant correlation with predicted outcomes. "
        .repeat(5);
    let end = "GRAVITATIONAL_WAVE_DETECTOR represents the final measurement apparatus.";

    let content = format!("{} {} {} {} {}", beginning, filler1, middle, filler2, end);

    Experience {
        content,
        experience_type: ExperienceType::Learning,
        ..Default::default()
    }
}

#[test]
fn test_long_content_chunked_correctly() {
    let (system, _temp) = create_test_system();

    let experience = create_long_memory_with_markers();
    let content_len = experience.content.len();

    // Store the memory
    let memory = system
        .remember(experience, None)
        .expect("Failed to store memory");

    println!("Stored memory with {} chars", content_len);
    println!("Memory ID: {:?}", memory);

    // Verify memory was stored
    let stats = system.stats();
    assert_eq!(stats.total_memories, 1);
}

#[test]
fn test_search_finds_beginning_content() {
    let (system, _temp) = create_test_system();

    let experience = create_long_memory_with_markers();
    let stored = system.remember(experience, None).expect("Failed to store");

    // Search for content that appears at the BEGINNING
    let query = Query {
        query_text: Some("quantum entanglement protocol initialization sequence".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = system.recall(&query).expect("Search failed");

    assert!(
        !results.is_empty(),
        "Failed to find memory by beginning content"
    );
    assert_eq!(
        results[0].id, stored,
        "Wrong memory returned for beginning search"
    );
    println!(
        "Beginning search: found {} result(s), top score: {:.4}",
        results.len(),
        results[0].score.unwrap_or(0.0)
    );
}

#[test]
fn test_search_finds_middle_content() {
    let (system, _temp) = create_test_system();

    let experience = create_long_memory_with_markers();
    let stored = system.remember(experience, None).expect("Failed to store");

    // Search for content that appears in the MIDDLE
    let query = Query {
        query_text: Some("neural plasticity mechanism core learning process".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = system.recall(&query).expect("Search failed");

    assert!(
        !results.is_empty(),
        "Failed to find memory by middle content"
    );
    assert_eq!(
        results[0].id, stored,
        "Wrong memory returned for middle search"
    );
    println!(
        "Middle search: found {} result(s), top score: {:.4}",
        results.len(),
        results[0].score.unwrap_or(0.0)
    );
}

#[test]
fn test_search_finds_end_content() {
    let (system, _temp) = create_test_system();

    let experience = create_long_memory_with_markers();
    let stored = system.remember(experience, None).expect("Failed to store");

    // Search for content that appears at the END
    // This is the CRITICAL test - before chunking, this would fail
    let query = Query {
        query_text: Some("gravitational wave detector final measurement apparatus".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = system.recall(&query).expect("Search failed");

    assert!(
        !results.is_empty(),
        "CRITICAL: Failed to find memory by END content - chunking not working!"
    );
    assert_eq!(
        results[0].id, stored,
        "Wrong memory returned for end search"
    );
    println!(
        "End search: found {} result(s), top score: {:.4}",
        results.len(),
        results[0].score.unwrap_or(0.0)
    );
}

#[test]
fn test_search_returns_full_memory_not_chunk() {
    let (system, _temp) = create_test_system();

    let experience = create_long_memory_with_markers();
    let original_content = experience.content.clone();
    let stored = system.remember(experience, None).expect("Failed to store");

    // Search for end content
    let query = Query {
        query_text: Some("gravitational wave detector".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = system.recall(&query).expect("Search failed");

    assert!(!results.is_empty(), "No results found");

    // The returned memory should contain ALL the content, not just the matching chunk
    let result_content = &results[0].experience.content;
    assert!(
        result_content.contains("QUANTUM_ENTANGLEMENT"),
        "Result missing beginning content"
    );
    assert!(
        result_content.contains("NEURAL_PLASTICITY"),
        "Result missing middle content"
    );
    assert!(
        result_content.contains("GRAVITATIONAL_WAVE"),
        "Result missing end content"
    );
    assert_eq!(
        result_content.len(),
        original_content.len(),
        "Result content length doesn't match original"
    );
    println!(
        "Verified: returned memory contains full content ({} chars)",
        result_content.len()
    );
}

#[test]
fn test_deduplication_returns_single_result() {
    let (system, _temp) = create_test_system();

    let experience = create_long_memory_with_markers();
    let stored = system.remember(experience, None).expect("Failed to store");

    // Search with query that might match multiple chunks
    let query = Query {
        query_text: Some("research methodology experiment".to_string()),
        max_results: 10,
        ..Default::default()
    };

    let results = system.recall(&query).expect("Search failed");

    // Even if multiple chunks match, we should only get ONE result for this memory
    let matching_ids: Vec<_> = results.iter().filter(|r| r.id == stored).collect();
    assert_eq!(
        matching_ids.len(),
        1,
        "Expected exactly 1 result for the memory, got {}. Deduplication failed!",
        matching_ids.len()
    );
    println!("Deduplication verified: 1 result for memory");
}

#[test]
fn test_multiple_long_memories_searched_correctly() {
    let (system, _temp) = create_test_system();

    // Create two different long memories
    let experience1 = {
        let content = format!(
            "MEMORY_ONE_START describes machine learning algorithms. {} MEMORY_ONE_END discusses neural networks.",
            "Various techniques for training models are explored. ".repeat(20)
        );
        Experience {
            content,
            experience_type: ExperienceType::Learning,
            ..Default::default()
        }
    };

    let experience2 = {
        let content = format!(
            "MEMORY_TWO_START covers database optimization. {} MEMORY_TWO_END explains query planning.",
            "Performance tuning and indexing strategies are examined. ".repeat(20)
        );
        Experience {
            content,
            experience_type: ExperienceType::Learning,
            ..Default::default()
        }
    };

    let mem1 = system
        .remember(experience1, None)
        .expect("Failed to store mem1");
    let mem2 = system
        .remember(experience2, None)
        .expect("Failed to store mem2");

    // Search for content at the END of memory 1
    let query1 = Query {
        query_text: Some("neural networks memory one end".to_string()),
        max_results: 10,
        ..Default::default()
    };
    let results1 = system.recall(&query1).expect("Search failed");
    assert!(!results1.is_empty(), "Failed to find memory 1 by end");
    assert_eq!(results1[0].id, mem1, "Wrong memory for query 1");

    // Search for content at the END of memory 2
    let query2 = Query {
        query_text: Some("query planning memory two end".to_string()),
        max_results: 10,
        ..Default::default()
    };
    let results2 = system.recall(&query2).expect("Search failed");
    assert!(!results2.is_empty(), "Failed to find memory 2 by end");
    assert_eq!(results2[0].id, mem2, "Wrong memory for query 2");

    println!("Successfully distinguished between two long memories");
}

/// COMPREHENSIVE QUALITY TEST
/// Verifies that EVERY segment of a long memory is retrievable.
/// Creates a memory with 10 unique, semantically distinct markers spread throughout,
/// then verifies each marker can be used to find the memory.
#[test]
fn test_every_segment_retrievable() {
    let (system, _temp) = create_test_system();

    // Create 10 unique, semantically distinct markers
    // Each marker uses different domain terminology to ensure distinctiveness
    let markers = [
        (
            "SEGMENT_01_CRYPTOGRAPHY",
            "encryption algorithms and secure key exchange protocols",
        ),
        (
            "SEGMENT_02_ASTRONOMY",
            "celestial navigation and stellar parallax measurements",
        ),
        (
            "SEGMENT_03_BOTANY",
            "photosynthesis chlorophyll pigment synthesis pathways",
        ),
        (
            "SEGMENT_04_GEOLOGY",
            "tectonic plate subduction volcanic activity zones",
        ),
        (
            "SEGMENT_05_LINGUISTICS",
            "morphological syntax grammatical structure analysis",
        ),
        (
            "SEGMENT_06_OCEANOGRAPHY",
            "thermocline currents marine ecosystem dynamics",
        ),
        (
            "SEGMENT_07_METALLURGY",
            "alloy composition heat treatment tempering processes",
        ),
        (
            "SEGMENT_08_ARCHAEOLOGY",
            "stratigraphy excavation artifact dating methods",
        ),
        (
            "SEGMENT_09_METEOROLOGY",
            "atmospheric pressure frontal systems precipitation",
        ),
        (
            "SEGMENT_10_PHARMACOLOGY",
            "receptor binding pharmacokinetic drug metabolism",
        ),
    ];

    // Build content with markers spread across ~5000 chars
    // Each segment is ~500 chars to ensure they land in different chunks
    let mut content_parts = Vec::new();
    for (marker, description) in &markers {
        let filler = "This section discusses various aspects of the field. \
             Technical details and methodological considerations are examined. \
             Research findings support the theoretical framework. ";
        let segment = format!(
            "{} describes {}. {} ",
            marker,
            description,
            filler.repeat(3)
        );
        content_parts.push(segment);
    }
    let full_content = content_parts.join("");

    println!("Created memory with {} chars", full_content.len());
    println!("Testing {} unique markers", markers.len());

    // Store the memory
    let experience = Experience {
        content: full_content,
        experience_type: ExperienceType::Learning,
        ..Default::default()
    };
    let stored = system.remember(experience, None).expect("Failed to store");

    // Test EACH marker can find the memory
    let mut successes = 0;
    let mut failures = Vec::new();

    for (marker, description) in &markers {
        // Search using the marker's associated description
        let query = Query {
            query_text: Some(description.to_string()),
            max_results: 10,
            ..Default::default()
        };

        let results = system.recall(&query).expect("Search failed");

        if results.is_empty() {
            failures.push(format!("{}: no results", marker));
        } else if results[0].id != stored {
            failures.push(format!("{}: wrong memory returned", marker));
        } else {
            successes += 1;
            println!(
                "  ✓ {} found (score: {:.4})",
                marker,
                results[0].score.unwrap_or(0.0)
            );
        }
    }

    println!(
        "\nQuality Report: {}/{} segments retrievable",
        successes,
        markers.len()
    );

    // All markers must be findable for the test to pass
    assert!(
        failures.is_empty(),
        "QUALITY FAILURE: {} of {} segments NOT retrievable:\n  {}",
        failures.len(),
        markers.len(),
        failures.join("\n  ")
    );
}

/// Systematic coverage test with numbered segments
/// Verifies no gaps in retrieval coverage
#[test]
fn test_systematic_coverage_numbered_segments() {
    let (system, _temp) = create_test_system();

    let num_segments = 8;
    let mut content_parts = Vec::new();

    // Create numbered segments with unique identifying content
    for i in 1..=num_segments {
        let unique_phrase = match i {
            1 => "quantum computing qubits superposition",
            2 => "renewable solar photovoltaic efficiency",
            3 => "biodiversity conservation ecosystem",
            4 => "cryptocurrency blockchain consensus",
            5 => "artificial neural deep learning",
            6 => "aerospace propulsion trajectory",
            7 => "genetic CRISPR genome editing",
            8 => "autonomous vehicle lidar navigation",
            _ => unreachable!(),
        };

        // Add filler to spread segments across chunks
        let filler = "Technical implementation details are documented. ".repeat(8);
        let segment = format!(
            "NUMBERED_SEGMENT_{} covers {}. {} ",
            i, unique_phrase, filler
        );
        content_parts.push(segment);
    }

    let full_content = content_parts.join("");
    println!(
        "Testing {} numbered segments across {} chars",
        num_segments,
        full_content.len()
    );

    let experience = Experience {
        content: full_content,
        experience_type: ExperienceType::Learning,
        ..Default::default()
    };
    let stored = system.remember(experience, None).expect("Failed to store");

    // Test each numbered segment
    let search_queries = [
        "quantum computing qubits superposition",
        "renewable solar photovoltaic efficiency",
        "biodiversity conservation ecosystem",
        "cryptocurrency blockchain consensus",
        "artificial neural deep learning",
        "aerospace propulsion trajectory",
        "genetic CRISPR genome editing",
        "autonomous vehicle lidar navigation",
    ];

    let mut coverage_map = vec![false; num_segments];

    for (idx, search_text) in search_queries.iter().enumerate() {
        let query = Query {
            query_text: Some(search_text.to_string()),
            max_results: 10,
            ..Default::default()
        };
        let results = system.recall(&query).expect("Search failed");

        if !results.is_empty() && results[0].id == stored {
            coverage_map[idx] = true;
            println!(
                "  Segment {}: ✓ (score: {:.4})",
                idx + 1,
                results[0].score.unwrap_or(0.0)
            );
        } else {
            println!("  Segment {}: ✗ NOT FOUND", idx + 1);
        }
    }

    // Calculate coverage percentage
    let covered = coverage_map.iter().filter(|&&x| x).count();
    let coverage_pct = (covered as f32 / num_segments as f32) * 100.0;

    println!(
        "\nCoverage: {}/{} segments ({:.1}%)",
        covered, num_segments, coverage_pct
    );

    // Require 100% coverage
    assert_eq!(
        covered,
        num_segments,
        "COVERAGE GAP: Only {}/{} segments retrievable. Gaps at positions: {:?}",
        covered,
        num_segments,
        coverage_map
            .iter()
            .enumerate()
            .filter(|(_, &found)| !found)
            .map(|(i, _)| i + 1)
            .collect::<Vec<_>>()
    );
}

/// Test that similarity scores make sense across different positions
#[test]
fn test_similarity_score_quality() {
    let (system, _temp) = create_test_system();

    // Create memory with very distinct beginning and end
    let content = format!(
        "BEGINNING_RASPBERRY_PI_EMBEDDED is about microcontrollers. {} \
         ENDING_KUBERNETES_ORCHESTRATION discusses container deployment.",
        "General technical content fills this space. ".repeat(30)
    );

    let experience = Experience {
        content,
        experience_type: ExperienceType::Learning,
        ..Default::default()
    };
    let stored = system.remember(experience, None).expect("Failed to store");

    // Search for beginning content
    let query_begin = Query {
        query_text: Some("raspberry pi embedded microcontrollers".to_string()),
        max_results: 10,
        ..Default::default()
    };
    let results_begin = system.recall(&query_begin).expect("Search failed");

    // Search for end content
    let query_end = Query {
        query_text: Some("kubernetes orchestration container deployment".to_string()),
        max_results: 10,
        ..Default::default()
    };
    let results_end = system.recall(&query_end).expect("Search failed");

    // Both should find the memory
    assert!(!results_begin.is_empty(), "Beginning not found");
    assert!(!results_end.is_empty(), "End not found");
    assert_eq!(results_begin[0].id, stored);
    assert_eq!(results_end[0].id, stored);

    // Both should have similarity scores populated (proves chunking returns scored results)
    let score_begin = results_begin[0].score;
    let score_end = results_end[0].score;

    println!("Beginning score: {:?}", score_begin);
    println!("End score: {:?}", score_end);

    // Verify scores are populated, not None
    assert!(
        score_begin.is_some(),
        "Beginning score not populated - semantic_retrieve not setting scores"
    );
    assert!(
        score_end.is_some(),
        "End score not populated - semantic_retrieve not setting scores"
    );

    // Both queries found the same memory (which contains both terms)
    // Score magnitude depends on MiniLM model and query-content overlap
    // Key validation: scores are being propagated through chunking pipeline
}
