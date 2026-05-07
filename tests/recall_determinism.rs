//! RH-11 — Determinism gate (Rig 1).
//!
//! Two consecutive runs of the L1 smoke suite, in the same process, against
//! the same fixtures, against fresh storage directories, must produce
//! byte-identical per-case rank lists.
//!
//! If this test ever fails, there is a non-deterministic source somewhere
//! in the retrieval pipeline that the RH-10 tie-break + thread-pinning
//! changes failed to cover. Do NOT relax this assertion — diagnose the
//! source. Examples of what flips ranks if not pinned:
//!
//! * `total_cmp` on raw distance with no secondary key — equal-distance
//!   candidates retain `HashMap` iteration order (random per process).
//! * Multi-threaded ONNX intra-op or rayon par_iter — float reductions
//!   accumulate sums in non-deterministic order, perturbing the
//!   fourth-decimal of the score.
//! * Wall-clock time leaking into a score (e.g. `Utc::now()` per-comparison
//!   inside a sort comparator instead of captured once outside).
//!
//! This test is the canary. Aggregate metrics ALSO have to match; if rank
//! lists are equal but metrics differ, the metric computation itself is
//! non-deterministic.

use std::path::PathBuf;

use shodh_memory::recall_harness::runner::{run_smoke_suite_with_ranks, RunInputs};

/// Build a `RunInputs` rooted at a unique storage path under the system
/// temp dir. Each invocation gets a fresh directory so the harness is not
/// reading state left over from a previous run.
fn run_inputs(tag: &str) -> RunInputs {
    let mut storage = std::env::temp_dir();
    storage.push(format!(
        "shodh-recall-determinism-{}-{}",
        tag,
        std::process::id()
    ));
    // Make sure we start clean even if a prior run crashed.
    let _ = std::fs::remove_dir_all(&storage);

    RunInputs {
        storage_path: storage,
        corpus_path: None,
        cases_path: None,
        suite: "smoke".to_string(),
        git_sha: "determinism-test".to_string(),
        // RH-11 (Rig 1) compares two consecutive single-pass runs. The
        // RH-12 cross-repeat check is exercised separately by the
        // `runner_repeats_*` tests in `src/recall_harness/runner.rs`;
        // keep this test single-pass so it stays under the CI budget.
        repeats: 1,
        // RH-8 (#270): determinism gate runs Full-only — per-mode
        // determinism is exercised by dedicated unit tests so this gate
        // stays focused on the production pipeline path.
        layer_modes: vec![shodh_memory::memory::types::LayerMode::Full],
    }
}

/// Two consecutive runs of the smoke suite must return byte-identical
/// rank lists.
///
/// This is the load-bearing assertion of RH-11. If it ever fires, do not
/// patch the assertion — find the source of non-determinism and fix it.
#[test]
fn smoke_suite_produces_byte_identical_rank_lists_across_runs() {
    let run1 =
        run_smoke_suite_with_ranks(&run_inputs("a")).expect("first smoke run should succeed");
    let run2 =
        run_smoke_suite_with_ranks(&run_inputs("b")).expect("second smoke run should succeed");

    assert_eq!(
        run1.ranks.len(),
        run2.ranks.len(),
        "both runs must execute the same number of cases"
    );

    let mut diverged = Vec::new();
    for (a, b) in run1.ranks.iter().zip(run2.ranks.iter()) {
        assert_eq!(
            a.case_id, b.case_id,
            "runner must walk cases in the same order both times"
        );
        if a.retrieved != b.retrieved {
            diverged.push(format!(
                "case `{}` diverged:\n  run1: {:?}\n  run2: {:?}",
                a.case_id, a.retrieved, b.retrieved
            ));
        }
    }

    assert!(
        diverged.is_empty(),
        "{} of {} cases produced different rank lists across runs:\n\n{}",
        diverged.len(),
        run1.ranks.len(),
        diverged.join("\n\n")
    );

    // Cleanup — only on success so a failure leaves storage on disk for
    // post-mortem inspection.
    let _ = std::fs::remove_dir_all(PathBuf::from(format!(
        "{}/shodh-recall-determinism-a-{}",
        std::env::temp_dir().display(),
        std::process::id()
    )));
    let _ = std::fs::remove_dir_all(PathBuf::from(format!(
        "{}/shodh-recall-determinism-b-{}",
        std::env::temp_dir().display(),
        std::process::id()
    )));
}

/// Aggregate metrics must also match bit-for-bit across runs.
///
/// If rank lists are equal (asserted above) but metrics differ, the metric
/// computation has its own non-determinism — for example, summing in
/// `HashMap` iteration order instead of fixture order.
#[test]
fn smoke_suite_produces_byte_identical_metrics_across_runs() {
    let run1 =
        run_smoke_suite_with_ranks(&run_inputs("m1")).expect("first smoke run should succeed");
    let run2 =
        run_smoke_suite_with_ranks(&run_inputs("m2")).expect("second smoke run should succeed");

    // Compare every gating metric of the `full` layer with bit-exact
    // equality. Latency and timestamp are excluded — they are wall-clock
    // measurements, not deterministic functions of inputs.
    let full1 = run1
        .report
        .layers
        .get("full")
        .expect("run1 must emit a `full` layer");
    let full2 = run2
        .report
        .layers
        .get("full")
        .expect("run2 must emit a `full` layer");

    assert_eq!(
        full1.ndcg_at_10.to_bits(),
        full2.ndcg_at_10.to_bits(),
        "ndcg@10"
    );
    assert_eq!(
        full1.recall_at_10.to_bits(),
        full2.recall_at_10.to_bits(),
        "recall@10"
    );
    assert_eq!(
        full1.precision_at_10.to_bits(),
        full2.precision_at_10.to_bits(),
        "precision@10"
    );
    assert_eq!(full1.mrr.to_bits(), full2.mrr.to_bits(), "mrr");
    assert_eq!(full1.p_at_1.to_bits(), full2.p_at_1.to_bits(), "p@1");
    assert_eq!(full1.map.to_bits(), full2.map.to_bits(), "map");

    // Per-category aggregates must match too.
    assert_eq!(
        run1.report.by_category.len(),
        run2.report.by_category.len(),
        "category set must match"
    );
    for (cat, c1) in &run1.report.by_category {
        let c2 = run2
            .report
            .by_category
            .get(cat)
            .unwrap_or_else(|| panic!("run2 missing category `{}`", cat));
        assert_eq!(
            c1.ndcg_at_10.to_bits(),
            c2.ndcg_at_10.to_bits(),
            "{} ndcg@10",
            cat
        );
        assert_eq!(
            c1.recall_at_10.to_bits(),
            c2.recall_at_10.to_bits(),
            "{} recall@10",
            cat
        );
        assert_eq!(c1.mrr.to_bits(), c2.mrr.to_bits(), "{} mrr", cat);
        assert_eq!(c1.p_at_1.to_bits(), c2.p_at_1.to_bits(), "{} p@1", cat);
        assert_eq!(c1.map.to_bits(), c2.map.to_bits(), "{} map", cat);
        assert_eq!(c1.case_count, c2.case_count, "{} case_count", cat);
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(PathBuf::from(format!(
        "{}/shodh-recall-determinism-m1-{}",
        std::env::temp_dir().display(),
        std::process::id()
    )));
    let _ = std::fs::remove_dir_all(PathBuf::from(format!(
        "{}/shodh-recall-determinism-m2-{}",
        std::env::temp_dir().display(),
        std::process::id()
    )));
}
