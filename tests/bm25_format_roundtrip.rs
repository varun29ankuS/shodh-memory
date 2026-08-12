//! Cross-version on-disk BM25 index round-trip harness.
//!
//! Driven by two env vars so the SAME corpus and the SAME queries can be run
//! by two different tantivy versions against one directory:
//!
//!   SHODH_BM25_RT_DIR   absolute path to the index directory
//!   SHODH_BM25_RT_MODE  "write" (create + populate + query) or "read" (open + query)
//!
//! Output is a stable, line-oriented dump so the two runs can be diffed byte
//! for byte.

use shodh_memory::memory::hybrid_search::BM25Index;
use shodh_memory::memory::types::MemoryId;
use std::path::PathBuf;

/// Fixed corpus. Content is chosen so BM25 ordering is discriminating:
/// varying term frequency, varying document length, and one document that
/// only matches via the `tags` / `entities` fields.
const CORPUS: &[(u128, &str, &[&str], &[&str])] = &[
    (
        1,
        "The user prefers Rust programming language for systems development",
        &["rust", "programming"],
        &["Rust"],
    ),
    (
        2,
        "Rust rust rust ownership and borrowing make memory safety a compile time property",
        &["rust", "memory"],
        &["Rust"],
    ),
    (
        3,
        "Python is used for the data science pipeline and notebook experiments",
        &["python", "data"],
        &["Python"],
    ),
    (
        4,
        "Memory safety in C requires manual discipline and static analysis tooling",
        &["c", "memory"],
        &["C"],
    ),
    (
        5,
        "A short note about rust",
        &["rust"],
        &["Rust"],
    ),
    (
        6,
        "The deployment runbook covers rollback, health checks, and canary traffic shifting for the production cluster during a release window",
        &["ops", "deployment"],
        &["Kubernetes"],
    ),
    (
        7,
        "Vector search recall improves when the graph leg and the keyword leg disagree",
        &["retrieval", "search"],
        &["BM25"],
    ),
    (
        8,
        "Nothing in this document mentions the target term at all",
        &["rust", "programming"],
        &["Rust"],
    ),
    (
        9,
        "Systems programming languages trade safety against control over memory layout",
        &["systems", "programming"],
        &["Rust", "C"],
    ),
    (
        10,
        "The memory system stores episodic traces and consolidates them during idle periods",
        &["memory", "systems"],
        &["Shodh"],
    ),
];

/// Fixed queries, exercised at a limit large enough to see the full ranking.
const QUERIES: &[&str] = &[
    "rust",
    "memory safety",
    "programming language systems",
    "rust programming",
    "deployment",
    "recall",
];

const LIMIT: usize = 10;

fn dir() -> PathBuf {
    PathBuf::from(
        std::env::var("SHODH_BM25_RT_DIR")
            .expect("SHODH_BM25_RT_DIR must be set for the round-trip harness"),
    )
}

fn dump(index: &BM25Index) {
    // tantivy is a private dependency of the lib, so an integration test cannot
    // name it. The operator labels the run instead; the authoritative version is
    // read back out of Cargo.lock alongside the dump.
    println!(
        "RT label={}",
        std::env::var("SHODH_BM25_RT_LABEL").unwrap_or_else(|_| "unlabelled".to_string())
    );
    println!("RT num_docs={}", index.len());
    for q in QUERIES {
        let hits = index.search(q, LIMIT).expect("search must not fail");
        println!("RT query={q:?} hits={}", hits.len());
        for (rank, (id, score)) in hits.iter().enumerate() {
            // Scores are printed at 6 decimal places: enough to catch a scoring
            // change, coarse enough not to trip on last-bit float noise.
            println!("RT   {rank} {} {score:.6}", id.0);
        }
    }
}

#[test]
#[ignore = "cross-version harness; run explicitly with SHODH_BM25_RT_DIR/MODE set"]
fn bm25_format_roundtrip() {
    let path = dir();
    let mode = std::env::var("SHODH_BM25_RT_MODE").expect("SHODH_BM25_RT_MODE must be set");

    match mode.as_str() {
        "write" => {
            assert!(
                !path.exists() || std::fs::read_dir(&path).unwrap().next().is_none(),
                "write mode wants an empty or absent directory, got a populated {path:?}"
            );
            let index = BM25Index::new(&path).expect("create index");
            for (n, content, tags, entities) in CORPUS {
                let tags: Vec<String> = tags.iter().map(|s| s.to_string()).collect();
                let entities: Vec<String> = entities.iter().map(|s| s.to_string()).collect();
                index
                    .upsert(
                        &MemoryId(uuid::Uuid::from_u128(*n)),
                        content,
                        &tags,
                        &entities,
                    )
                    .expect("upsert");
            }
            index.commit().expect("commit");
            index.reload().expect("reload");
            assert_eq!(
                index.commit_failure_count(),
                0,
                "commit failures mean the index on disk is incomplete; the baseline is void"
            );
            assert_eq!(index.len(), CORPUS.len(), "all docs must be searchable");
            dump(&index);
        }
        "read" => {
            assert!(path.exists(), "read mode needs an existing index at {path:?}");
            let index = BM25Index::new(&path).expect("open pre-existing index");
            index.reload().expect("reload");
            assert_eq!(
                index.len(),
                CORPUS.len(),
                "document count changed across versions"
            );
            dump(&index);
        }
        other => panic!("unknown SHODH_BM25_RT_MODE {other:?}"),
    }
}
