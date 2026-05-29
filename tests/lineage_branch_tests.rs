//! Branch-subsystem wiring tests.
//!
//! A project-pivot signal must do more than create an orphan branch metadata
//! record: it must (a) open a branch and (b) anchor it into the edge graph with
//! a `BranchedFrom` edge tagged by the new `branch_id`, so the branch is
//! reachable via `trace()` / `find_root_cause()`. Before the wiring fix the
//! branch was created but no edge referenced it and every edge stayed on `main`.

use chrono::{DateTime, Duration, Utc};
use tempfile::TempDir;
use uuid::Uuid;

use shodh_memory::memory::lineage::CausalRelation;
use shodh_memory::memory::types::{Experience, ExperienceType};
use shodh_memory::memory::{Memory, MemoryConfig, MemoryId, MemorySystem};

fn create_test_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    };
    let system = MemorySystem::new(config, None).expect("memory system");
    (system, temp_dir)
}

fn mem(content: &str, created_at: DateTime<Utc>) -> Memory {
    let exp = Experience {
        experience_type: ExperienceType::Decision,
        content: content.to_string(),
        ..Default::default()
    };
    Memory::new(
        MemoryId(Uuid::new_v4()),
        exp,
        0.7,
        None,
        None,
        None,
        Some(created_at),
    )
}

#[test]
fn pivot_signal_opens_branch_and_anchors_branchedfrom_edge() {
    let (system, _tmp) = create_test_system();
    let user = "branch-user";
    let now = Utc::now();
    let prior = mem(
        "Working on the Vamana index integration for vector search.",
        now - Duration::days(1),
    );
    let pivot = mem(
        "We need to abandon this and pivot to a different strategy entirely.",
        now,
    );

    let edges = system
        .infer_lineage_for_memory(user, &pivot, std::slice::from_ref(&prior))
        .expect("infer lineage");

    // (a) A pivot branch was created (in addition to main).
    let branches = system
        .lineage_graph()
        .list_branches(user)
        .expect("branches");
    let pivot_branch = branches
        .iter()
        .find(|b| b.id != "main")
        .unwrap_or_else(|| panic!("expected a pivot branch, got {branches:?}"));
    let bid = pivot_branch.id.clone();

    // (b) A BranchedFrom edge connects the pivot memory to the prior work and
    //     carries the new branch id.
    let branched = edges
        .iter()
        .find(|e| e.relation == CausalRelation::BranchedFrom)
        .unwrap_or_else(|| panic!("expected a BranchedFrom edge, got {edges:?}"));
    assert_eq!(branched.from, pivot.id);
    assert_eq!(branched.to, prior.id);
    assert_eq!(branched.branch_id.as_deref(), Some(bid.as_str()));

    // The edge is persisted (not merely returned).
    let stored = system
        .lineage_graph()
        .get_edges_from(user, &pivot.id)
        .expect("edges from");
    assert!(stored
        .iter()
        .any(|e| e.relation == CausalRelation::BranchedFrom && e.branch_id.is_some()));
}

#[test]
fn no_pivot_signal_creates_no_branch() {
    let (system, _tmp) = create_test_system();
    let user = "no-pivot-user";
    let now = Utc::now();
    let a = mem(
        "Implemented the SPANN search dimension guard.",
        now - Duration::days(1),
    );
    let b = mem("Added a unit test for the SPANN guard.", now);

    let _ = system
        .infer_lineage_for_memory(user, &b, std::slice::from_ref(&a))
        .expect("infer lineage");

    // No pivot language → no auto branch (only main may exist, if anything).
    let branches = system
        .lineage_graph()
        .list_branches(user)
        .expect("branches");
    assert!(
        branches.iter().all(|br| br.id == "main"),
        "unexpected non-main branch: {branches:?}"
    );
}
