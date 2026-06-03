//! E4 — causal-lineage harness.
//!
//! Tests causal-chain retrieval ("why did X happen?"): the answer to a why-query
//! is a ROOT-CAUSE memory two causal hops back that does NOT lexically mention the
//! queried event, so plain BM25/vector cannot reach it — only following the causal
//! chain (event_c ← event_b ← event_a) does. Mirrors E3's structure but with
//! causal relations, and reports whether the cause is retrievable at all.
//!
//! NOTE (fidelity): lineage edges are inferred in the production remember()
//! handler, not necessarily in the harness ingest path — so this also reveals
//! whether causal retrieval works via the graph at all in eval. An inert result
//! here is an honest signal that lineage is unexercised, not that the question is
//! unanswerable.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};

use crate::memory::types::LayerMode;
use crate::recall_harness::fixtures::{CorpusItem, RelevanceJudgement, SmokeCase, SmokeCategory};
use crate::recall_harness::report::{MultiHopLayerRow, MultiHopReport};
use crate::recall_harness::runner::{run_smoke_suite_with_ranks, RunInputs};

pub const DEFAULT_CHAINS: usize = 60;

const EVENTS: &[&str] = &[
    "Vornak", "Meslin", "Caldor", "Brennar", "Torvel", "Quillan", "Davmor", "Fennick", "Gorlan",
    "Halven", "Jennor", "Korvath", "Lannic", "Morwen", "Norvell", "Polnar", "Rinhall", "Selvic",
    "Tavmor", "Wendel",
];

fn event(n: usize) -> String {
    let base = EVENTS[n % EVENTS.len()];
    let block = n / EVENTS.len();
    if block == 0 {
        format!("the {base} incident")
    } else {
        format!("the {base}{block} incident")
    }
}

pub fn generate_lineage_fixtures(chains: usize) -> (Vec<CorpusItem>, Vec<SmokeCase>) {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut corpus: Vec<CorpusItem> = Vec::with_capacity(chains * 2);
    let mut cases: Vec<SmokeCase> = Vec::with_capacity(chains * 2);

    for i in 0..chains {
        let a = event(3 * i); // root cause
        let b = event(3 * i + 1); // intermediate
        let c = event(3 * i + 2); // observed effect

        let root_id = format!("lin-root-{i:04}");
        let mid_id = format!("lin-mid-{i:04}");

        // root: a caused b. (does NOT mention c)
        corpus.push(CorpusItem {
            id: root_id.clone(),
            content: format!("{a} was the root cause that triggered {b}."),
            memory_type: "fact".to_string(),
            tags: vec![a.clone(), b.clone()],
            created_at: base + chrono::Duration::minutes(2 * i as i64),
        });
        // mid: b caused c.
        corpus.push(CorpusItem {
            id: mid_id.clone(),
            content: format!("{b} in turn led directly to {c}."),
            memory_type: "fact".to_string(),
            tags: vec![b.clone(), c.clone()],
            created_at: base + chrono::Duration::minutes(2 * i as i64 + 1),
        });

        // Root-cause case: "what was the root cause of c?" → gold = root memory
        // (a→b), which never mentions c. Reachable only by chaining c←b←a.
        cases.push(SmokeCase {
            id: format!("lin-why-{i:04}"),
            category: SmokeCategory::MultiHop,
            query: format!("What was the root cause behind {c}?"),
            fixture_corpus_id: "lineage".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: root_id,
                grade: 3,
            }],
        });
        // Control: direct cause of c (gold = mid memory, mentions c → BM25-solvable).
        cases.push(SmokeCase {
            id: format!("lin-direct-{i:04}"),
            category: SmokeCategory::SingleHop,
            query: format!("What led directly to {c}?"),
            fixture_corpus_id: "lineage".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: mid_id,
                grade: 3,
            }],
        });
    }
    (corpus, cases)
}

fn write_fixtures(
    dir: &std::path::Path,
    corpus: &[CorpusItem],
    cases: &[SmokeCase],
) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(dir)?;
    let corpus_path = dir.join("lineage_corpus.jsonl");
    let cases_path = dir.join("lineage_cases.jsonl");
    let mut cbuf = String::new();
    for item in corpus {
        cbuf.push_str(&serde_json::to_string(item)?);
        cbuf.push('\n');
    }
    std::fs::write(&corpus_path, cbuf)?;
    let mut qbuf = String::new();
    for case in cases {
        qbuf.push_str(&serde_json::to_string(case)?);
        qbuf.push('\n');
    }
    std::fs::write(&cases_path, qbuf)?;
    Ok((corpus_path, cases_path))
}

/// Run the planted causal-chain corpus through every LayerMode; report per-layer
/// recall on the root-cause cases (2 causal hops) vs the direct-cause control.
/// `multihop_*` = root-cause, `onehop_*` = direct control.
pub fn analyze_lineage(inputs: &RunInputs, chains: usize) -> Result<MultiHopReport> {
    let (corpus, cases) = generate_lineage_fixtures(chains);
    let why_cases = cases
        .iter()
        .filter(|c| c.category == SmokeCategory::MultiHop)
        .count();
    let direct_cases = cases
        .iter()
        .filter(|c| c.category == SmokeCategory::SingleHop)
        .count();
    let fixture_dir = inputs.storage_path.join("lineage_fixtures");
    let (corpus_path, cases_path) = write_fixtures(&fixture_dir, &corpus, &cases)?;

    let run_inputs = RunInputs {
        storage_path: inputs.storage_path.join("lineage_run"),
        corpus_path: Some(corpus_path),
        cases_path: Some(cases_path),
        suite: "lineage".to_string(),
        git_sha: inputs.git_sha.clone(),
        repeats: 1,
        layer_modes: LayerMode::ALL.to_vec(),
        age_days: inputs.age_days,
    };
    let out = run_smoke_suite_with_ranks(&run_inputs).context("lineage suite run failed")?;

    let mean = |records: &[crate::recall_harness::report::PerCaseRecord], cat: &str| -> f64 {
        let vals: Vec<f64> = records
            .iter()
            .filter(|r| r.category == cat)
            .map(|r| r.recall_at_k)
            .collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };
    let mut rows: Vec<MultiHopLayerRow> = Vec::new();
    for mode in LayerMode::ALL {
        let key = mode.report_key().to_string();
        if let Some(records) = out.per_case_by_layer.get(&key) {
            rows.push(MultiHopLayerRow {
                layer: key,
                multihop_recall_at_10: mean(records, "multi_hop"),
                onehop_recall_at_10: mean(records, "single_hop"),
                multihop_mrr: 0.0,
            });
        }
    }
    Ok(MultiHopReport {
        chains,
        multihop_cases: why_cases,
        onehop_cases: direct_cases,
        rows,
    })
}
