//! E5 — ontology / type-disambiguation harness.
//!
//! Tests whether the ontological rerank (Layer 4.9) uses entity TYPE to break
//! ties that lexical/semantic retrieval cannot. For each item we plant a PERSON
//! memory and an ORG memory that share a LOCATION, then ask a type-qualified
//! question ("which person was in {place}?"). Both memories match on {place}, so
//! BM25/vector cannot disambiguate — only the ontology layer, by preferring the
//! Person-typed entity, can put the right one on top.
//!
//! Real person/org/place names are used so the NER pipeline assigns reliable
//! PER/ORG/LOC labels (nonce tokens do not type reliably). The +rerank delta on
//! the type-qualified cases is the ontology layer's isolated contribution; if it
//! is ~0, the ontology rerank is inert and this says so honestly.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};

use crate::memory::types::LayerMode;
use crate::recall_harness::fixtures::{CorpusItem, RelevanceJudgement, SmokeCase, SmokeCategory};
use crate::recall_harness::report::{MultiHopLayerRow, MultiHopReport};
use crate::recall_harness::runner::{run_smoke_suite_with_ranks, RunInputs};

pub const DEFAULT_ITEMS: usize = 40;

const PERSONS: &[&str] = &[
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Iris", "Jack",
    "Karen", "Liam", "Mary", "Nora", "Oscar", "Paula", "Quinn", "Rachel", "Sam", "Tina",
];
const ORGS: &[&str] = &[
    "Acme Corporation", "Globex Corporation", "Initech", "Umbrella Corporation",
    "Stark Industries", "Wayne Enterprises", "Wonka Industries", "Cyberdyne Systems",
    "Soylent Corp", "Hooli", "Pied Piper", "Massive Dynamic", "Tyrell Corporation",
    "Aperture Science", "Black Mesa", "Oscorp", "LexCorp", "Weyland Corp", "Abstergo", "Vault Tech",
];
const PLACES: &[&str] = &[
    "Paris", "Tokyo", "Berlin", "Madrid", "Rome", "Cairo", "Lima", "Oslo", "Vienna", "Dublin",
    "Lisbon", "Athens", "Prague", "Warsaw", "Helsinki", "Bangkok", "Seoul", "Boston", "Denver",
    "Seattle",
];

fn pick(list: &[&str], i: usize) -> String {
    let base = list[i % list.len()];
    let block = i / list.len();
    if block == 0 {
        base.to_string()
    } else {
        format!("{base} {block}")
    }
}

pub fn generate_ontology_fixtures(items: usize) -> (Vec<CorpusItem>, Vec<SmokeCase>) {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut corpus: Vec<CorpusItem> = Vec::with_capacity(items * 2);
    let mut cases: Vec<SmokeCase> = Vec::with_capacity(items * 2);

    for i in 0..items {
        let person = pick(PERSONS, i);
        let org = pick(ORGS, i);
        let place = pick(PLACES, i);

        let person_id = format!("onto-person-{i:04}");
        let org_id = format!("onto-org-{i:04}");

        corpus.push(CorpusItem {
            id: person_id.clone(),
            content: format!("{person} attended the conference in {place}."),
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + chrono::Duration::minutes(2 * i as i64),
        });
        corpus.push(CorpusItem {
            id: org_id.clone(),
            content: format!("{org} opened a new office in {place}."),
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + chrono::Duration::minutes(2 * i as i64 + 1),
        });

        // Type-qualified: both memories share {place}; only TYPE distinguishes the
        // person memory (gold) from the org memory. Uses SmokeCategory::Entity.
        cases.push(SmokeCase {
            id: format!("onto-type-{i:04}"),
            category: SmokeCategory::Entity,
            query: format!("Which person was in {place}?"),
            fixture_corpus_id: "ontology".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: person_id.clone(),
                grade: 3,
            }],
        });
        // Control: a direct lexical query for the person memory (BM25-solvable),
        // so the type-qualified vs control gap isolates the ontology rerank.
        cases.push(SmokeCase {
            id: format!("onto-ctrl-{i:04}"),
            category: SmokeCategory::Code,
            query: format!("Who attended the conference in {place}?"),
            fixture_corpus_id: "ontology".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: person_id,
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
    let corpus_path = dir.join("ontology_corpus.jsonl");
    let cases_path = dir.join("ontology_cases.jsonl");
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

/// Run the planted ontology corpus through every LayerMode; report per-layer
/// recall on the type-qualified cases (entity) vs the lexical control (code). The
/// +rerank delta on the type-qualified column is the ontology layer's isolated
/// contribution. `multihop_*` = type-qualified, `onehop_*` = control.
pub fn analyze_ontology(inputs: &RunInputs, items: usize) -> Result<MultiHopReport> {
    let (corpus, cases) = generate_ontology_fixtures(items);
    let type_cases = cases
        .iter()
        .filter(|c| c.category == SmokeCategory::Entity)
        .count();
    let ctrl_cases = cases
        .iter()
        .filter(|c| c.category == SmokeCategory::Code)
        .count();
    let fixture_dir = inputs.storage_path.join("ontology_fixtures");
    let (corpus_path, cases_path) = write_fixtures(&fixture_dir, &corpus, &cases)?;

    let run_inputs = RunInputs {
        storage_path: inputs.storage_path.join("ontology_run"),
        corpus_path: Some(corpus_path),
        cases_path: Some(cases_path),
        suite: "ontology".to_string(),
        git_sha: inputs.git_sha.clone(),
        repeats: 1,
        layer_modes: LayerMode::ALL.to_vec(),
        age_days: inputs.age_days,
    };
    let out = run_smoke_suite_with_ranks(&run_inputs).context("ontology suite run failed")?;

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
                multihop_recall_at_10: mean(records, "entity"),
                onehop_recall_at_10: mean(records, "code"),
                multihop_mrr: 0.0,
            });
        }
    }
    Ok(MultiHopReport {
        chains: items,
        multihop_cases: type_cases,
        onehop_cases: ctrl_cases,
        rows,
    })
}
