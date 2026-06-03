//! Temporal controlled harness — planted time-varying facts.
//!
//! Tests temporal reasoning directly: for each subject, an attribute CHANGES over
//! time (e.g. "in 2019 X was a pilot" → "in 2022 X became a captain"). A query
//! asks for the attribute at an intermediate time ("what was X in 2020?"). The
//! gold is the EARLIER memory (the fact valid at the queried time), NOT the latest
//! — so a system that just returns the most recent / most lexically-similar
//! memory fails, and only the temporal-fact layer (Layer 0.6, now populated by the
//! remember()-path fact storage + the consolidation cycle) gets it right.
//!
//! Deterministic, model-free. Run through every LayerMode so the +facts delta on
//! the temporal cases is the temporal layer's isolated contribution — the thing a
//! plain recall@k Temporal category cannot isolate.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};

use crate::memory::types::LayerMode;
use crate::recall_harness::fixtures::{CorpusItem, RelevanceJudgement, SmokeCase, SmokeCategory};
use crate::recall_harness::report::{MultiHopLayerRow, MultiHopReport};
use crate::recall_harness::runner::{run_smoke_suite_with_ranks, RunInputs};

/// Default number of planted temporal subjects. Each yields 2 memories (early,
/// late) + 1 "valid-at-T" query and 1 "latest" control query.
pub const DEFAULT_SUBJECTS: usize = 60;

const NAMES: &[&str] = &[
    "Vorland", "Mesker", "Calthen", "Brenwick", "Torsten", "Quillon", "Davmont", "Fenwick",
    "Gorlan", "Halbrook", "Jennoll", "Korvath", "Lanyon", "Morwell", "Norcliff", "Polk",
    "Rinhart", "Selby", "Tavish", "Wendell",
];
const EARLY_ROLES: &[&str] = &["pilot", "analyst", "courier", "scout", "clerk"];
const LATE_ROLES: &[&str] = &["captain", "director", "marshal", "warden", "chief"];

fn name(n: usize) -> String {
    let base = NAMES[n % NAMES.len()];
    let block = n / NAMES.len();
    if block == 0 {
        base.to_string()
    } else {
        format!("{base}{block}")
    }
}

/// Generate planted temporal fixtures. Subject `i` is a {pilot in 2018+i%3}…
/// design where early year < query year < late year, so the gold for the
/// valid-at-T query is unambiguously the EARLY memory.
pub fn generate_temporal_fixtures(subjects: usize) -> (Vec<CorpusItem>, Vec<SmokeCase>) {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut corpus: Vec<CorpusItem> = Vec::with_capacity(subjects * 2);
    let mut cases: Vec<SmokeCase> = Vec::with_capacity(subjects * 2);

    for i in 0..subjects {
        let subj = name(i);
        let early_role = EARLY_ROLES[i % EARLY_ROLES.len()];
        let late_role = LATE_ROLES[i % LATE_ROLES.len()];
        // Fixed, well-separated years so "valid at T" is unambiguous.
        let early_year = 2017;
        let query_year = 2019;
        let late_year = 2021;

        let early_id = format!("temp-early-{i:04}");
        let late_id = format!("temp-late-{i:04}");

        corpus.push(CorpusItem {
            id: early_id.clone(),
            content: format!("In {early_year}, {subj} worked as a {early_role}."),
            memory_type: "fact".to_string(),
            tags: vec![subj.clone()],
            created_at: base + chrono::Duration::minutes(2 * i as i64),
        });
        corpus.push(CorpusItem {
            id: late_id.clone(),
            content: format!("In {late_year}, {subj} was promoted to {late_role}."),
            memory_type: "fact".to_string(),
            tags: vec![subj.clone()],
            created_at: base + chrono::Duration::minutes(2 * i as i64 + 1),
        });

        // Valid-at-T: the gold is the EARLY memory (the fact true in query_year).
        // A "return the latest" system returns the late memory and fails.
        cases.push(SmokeCase {
            id: format!("temp-validT-{i:04}"),
            category: SmokeCategory::Temporal,
            query: format!("What did {subj} do in {query_year}?"),
            fixture_corpus_id: "temporal".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: early_id,
                grade: 3,
            }],
        });
        // Latest control: asks for the current/most-recent role; gold = late memory.
        // BM25/recency should solve this — the contrast that isolates temporal reasoning.
        cases.push(SmokeCase {
            id: format!("temp-latest-{i:04}"),
            category: SmokeCategory::SingleHop,
            query: format!("What was {subj} promoted to?"),
            fixture_corpus_id: "temporal".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: late_id,
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
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating temporal fixture dir {}", dir.display()))?;
    let corpus_path = dir.join("temporal_corpus.jsonl");
    let cases_path = dir.join("temporal_cases.jsonl");
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

/// Build the planted temporal corpus, run it through every LayerMode, and report
/// per-layer recall on the valid-at-T cases (temporal) vs the latest-control
/// cases. The +facts delta on valid-at-T = the temporal-fact layer's isolated
/// contribution. Reuses the multi-hop report shape (multihop_* = valid-at-T,
/// onehop_* = latest control).
pub fn analyze_temporal(inputs: &RunInputs, subjects: usize) -> Result<MultiHopReport> {
    let (corpus, cases) = generate_temporal_fixtures(subjects);
    let validt_cases = cases
        .iter()
        .filter(|c| c.category == SmokeCategory::Temporal)
        .count();
    let latest_cases = cases
        .iter()
        .filter(|c| c.category == SmokeCategory::SingleHop)
        .count();

    let fixture_dir = inputs.storage_path.join("temporal_fixtures");
    let (corpus_path, cases_path) = write_fixtures(&fixture_dir, &corpus, &cases)?;

    let run_inputs = RunInputs {
        storage_path: inputs.storage_path.join("temporal_run"),
        corpus_path: Some(corpus_path),
        cases_path: Some(cases_path),
        suite: "temporal".to_string(),
        git_sha: inputs.git_sha.clone(),
        repeats: 1,
        layer_modes: LayerMode::ALL.to_vec(),
        age_days: inputs.age_days,
    };
    let out = run_smoke_suite_with_ranks(&run_inputs).context("temporal suite run failed")?;

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
                multihop_recall_at_10: mean(records, "temporal"),
                onehop_recall_at_10: mean(records, "single_hop"),
                multihop_mrr: 0.0,
            });
        }
    }

    Ok(MultiHopReport {
        chains: subjects,
        multihop_cases: validt_cases,
        onehop_cases: latest_cases,
        rows,
    })
}
