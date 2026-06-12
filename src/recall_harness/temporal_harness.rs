//! Temporal controlled harness — planted time-varying facts (adversarial).
//!
//! Tests temporal reasoning under the adversarial-isolation standard: each subject
//! has THREE equi-lexical time-states of one attribute (identical sentence frame,
//! differing only in {year}/{role}) at well-separated years. The valid-at-T query
//! asks for the state at an INTERMEDIATE year that appears in NO memory, so BM25 /
//! vector see three equally-good candidates (chance ≈ 1/3) and only date-interval
//! logic — "the state in force at T is the most recent state ≤ T" — picks the gold
//! (state[0]). A system that returns the latest, or the most lexically similar,
//! fails. A lexical control names an exact state year (BM25-solvable) to prove the
//! corpus is retrievable; the valid-at-T − control gap isolates the temporal layer.
//!
//! This replaces a v1 generator whose query lexically matched the gold, ceilinged
//! recall at 1.0 across every layer, and therefore measured BM25, not temporality.
//!
//! Deterministic, model-free. Run through every LayerMode so the +facts delta on
//! the valid-at-T cases is the temporal layer's isolated contribution.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};

use crate::memory::types::LayerMode;
use crate::recall_harness::fixtures::{CorpusItem, RelevanceJudgement, SmokeCase, SmokeCategory};
use crate::recall_harness::report::{MultiHopLayerRow, MultiHopReport};
use crate::recall_harness::runner::{run_smoke_suite_with_ranks, RunInputs};

/// Default number of planted temporal subjects. Each yields 3 equi-lexical time-
/// state memories + 1 "valid-at-T" query and 1 lexical control query.
pub const DEFAULT_SUBJECTS: usize = 60;

const NAMES: &[&str] = &[
    "Vorland", "Mesker", "Calthen", "Brenwick", "Torsten", "Quillon", "Davmont", "Fenwick",
    "Gorlan", "Halbrook", "Jennoll", "Korvath", "Lanyon", "Morwell", "Norcliff", "Polk", "Rinhart",
    "Selby", "Tavish", "Wendell",
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

/// Generate ADVERSARIAL temporal fixtures. Each subject has THREE equi-lexical
/// time-states of the same attribute — identical phrasing except {year} and
/// {role} — at well-separated years. The valid-at-T query asks for the role at an
/// intermediate year that matches NONE of the three lexically, so BM25/vector see
/// three equally-good candidates (recall ~1/3 by chance) and only date-interval
/// logic ("the state in force at T is the most recent state ≤ T") can pick the
/// gold. A lexical control names the exact year (BM25-solvable) to prove the
/// corpus is retrievable. This is the standard the v1 generator failed.
pub fn generate_temporal_fixtures(subjects: usize) -> (Vec<CorpusItem>, Vec<SmokeCase>) {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    // Three states at fixed, well-separated years. The valid-at-T query uses a
    // year strictly between state[0] and state[1], so the gold is state[0] and the
    // queried year appears in NO memory (no lexical year match for any candidate).
    const STATE_YEARS: [u32; 3] = [2013, 2017, 2021];
    // T sits CLOSER to the later state (2017) but strictly before it, so a naive
    // nearest-year heuristic answers 2017 (WRONG) and only correct "most-recent
    // state ≤ T" semantics answers 2013 (gold = state[0]). 2015 would be
    // equidistant from 2013/2017 and let nearest-year score 50% by luck.
    const QUERY_YEAR: u32 = 2016; // 2013 < 2016 < 2017 → valid-at-2016 = the 2013 state
    let roles = [EARLY_ROLES, LATE_ROLES].concat();

    let mut corpus: Vec<CorpusItem> = Vec::with_capacity(subjects * 3);
    let mut cases: Vec<SmokeCase> = Vec::with_capacity(subjects * 2);

    for i in 0..subjects {
        let subj = name(i);
        let mut state_ids = Vec::with_capacity(3);
        for (s, &year) in STATE_YEARS.iter().enumerate() {
            // Distinct role per state, but IDENTICAL sentence frame so the three
            // states are equi-lexical except {year}/{role}.
            let role = roles[(i + s) % roles.len()];
            let id = format!("temp-s{s}-{i:04}");
            corpus.push(CorpusItem {
                id: id.clone(),
                content: format!("In {year}, {subj}'s assignment was {role}."),
                memory_type: "fact".to_string(),
                tags: vec![subj.clone()],
                created_at: base + chrono::Duration::minutes(3 * i as i64 + s as i64),
            });
            state_ids.push(id);
        }

        // Valid-at-T: queried year (2015) matches no memory; gold = state[0] (2013,
        // the most recent state ≤ 2015). Only temporal interval logic resolves it.
        cases.push(SmokeCase {
            id: format!("temp-validT-{i:04}"),
            category: SmokeCategory::Temporal,
            query: format!("What was {subj}'s assignment in {QUERY_YEAR}?"),
            fixture_corpus_id: "temporal".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: state_ids[0].clone(),
                grade: 3,
            }],
        });
        // Lexical control: names an exact state year (2017) → BM25-solvable.
        cases.push(SmokeCase {
            id: format!("temp-ctrl-{i:04}"),
            category: SmokeCategory::SingleHop,
            query: format!("In {}, what was {subj}'s assignment?", STATE_YEARS[1]),
            fixture_corpus_id: "temporal".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: state_ids[1].clone(),
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
/// per-layer recall on the valid-at-T cases (temporal) vs the lexical control
/// cases. The +facts delta on valid-at-T = the temporal-fact layer's isolated
/// contribution. Reuses the multi-hop report shape (multihop_* = valid-at-T,
/// onehop_* = lexical control).
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

    let mean = |records: &[crate::recall_harness::report::PerCaseRecord],
                cat: &str,
                sel: fn(&crate::recall_harness::report::PerCaseRecord) -> f64|
     -> f64 {
        let vals: Vec<f64> = records
            .iter()
            .filter(|r| r.category == cat)
            .map(sel)
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
                multihop_recall_at_10: mean(records, "temporal", |r| r.recall_at_k),
                onehop_recall_at_10: mean(records, "single_hop", |r| r.recall_at_k),
                multihop_mrr: mean(records, "temporal", |r| r.mrr),
                multihop_p_at_1: mean(records, "temporal", |r| r.p_at_1),
                onehop_p_at_1: mean(records, "single_hop", |r| r.p_at_1),
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
