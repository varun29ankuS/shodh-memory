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

// PERSONS and PLACES must be globally unique SINGLE tokens, one per item, so the
// {place} localizer in the type-qualified query (and the {person} localizer in the
// control) cannot lexically collide with another item's memories. They are 60+
// long so `pick` never appends a block suffix for the default 60-item run — a
// space-suffixed "Paris 1" would tokenize to ["paris","1"] and bleed onto item 0's
// "Paris", silently corrupting P@1 (the bug this list size fixes). Persons must
// stay real first names so NER types them PER (the gold the rerank must prefer).
const PERSONS: &[&str] = &[
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Iris", "Jack", "Karen",
    "Liam", "Mary", "Nora", "Oscar", "Paula", "Quinn", "Rachel", "Sam", "Tina", "Ursula", "Victor",
    "Wendy", "Xavier", "Yolanda", "Zachary", "Adam", "Brenda", "Caleb", "Diana", "Ethan", "Fiona",
    "George", "Hannah", "Ian", "Julia", "Kevin", "Laura", "Marcus", "Natalie", "Owen", "Priya",
    "Quincy", "Rosa", "Steven", "Tara", "Umar", "Vera", "Walter", "Ximena", "Yusuf", "Zoe",
    "Aaron", "Bella", "Carlos", "Delia", "Elena", "Felix", "Gloria", "Hugo",
];
const ORGS: &[&str] = &[
    "Acme Corporation",
    "Globex Corporation",
    "Initech",
    "Umbrella Corporation",
    "Stark Industries",
    "Wayne Enterprises",
    "Wonka Industries",
    "Cyberdyne Systems",
    "Soylent Corp",
    "Hooli",
    "Pied Piper",
    "Massive Dynamic",
    "Tyrell Corporation",
    "Aperture Science",
    "Black Mesa",
    "Oscorp",
    "LexCorp",
    "Weyland Corp",
    "Abstergo",
    "Vault Tech",
];
const PLACES: &[&str] = &[
    "Paris",
    "Tokyo",
    "Berlin",
    "Madrid",
    "Rome",
    "Cairo",
    "Lima",
    "Oslo",
    "Vienna",
    "Dublin",
    "Lisbon",
    "Athens",
    "Prague",
    "Warsaw",
    "Helsinki",
    "Bangkok",
    "Seoul",
    "Boston",
    "Denver",
    "Seattle",
    "Toronto",
    "Sydney",
    "Mumbai",
    "Munich",
    "Geneva",
    "Zurich",
    "Brussels",
    "Amsterdam",
    "Copenhagen",
    "Stockholm",
    "Budapest",
    "Bucharest",
    "Belgrade",
    "Naples",
    "Venice",
    "Florence",
    "Porto",
    "Valencia",
    "Seville",
    "Glasgow",
    "Manchester",
    "Liverpool",
    "Bristol",
    "Portland",
    "Austin",
    "Dallas",
    "Houston",
    "Phoenix",
    "Atlanta",
    "Miami",
    "Chicago",
    "Detroit",
    "Cleveland",
    "Pittsburgh",
    "Baltimore",
    "Richmond",
    "Nashville",
    "Memphis",
    "Orlando",
    "Tampa",
];

fn pick(list: &[&str], i: usize) -> String {
    let base = list[i % list.len()];
    let block = i / list.len();
    if block == 0 {
        base.to_string()
    } else {
        // Past the list length tokens would start colliding; the generator clamps
        // `items` to the list length so this branch is unreachable for the localizer
        // lists. No-space suffix keeps any incidental use as a distinct token.
        format!("{base}{block}")
    }
}

/// Number of ORG distractors planted per item. With K orgs sharing the exact same
/// sentence frame + place as the single person, BM25/vector see K+1 equally-good
/// candidates for the type-qualified query, so the shortcut baseline is ~1/(K+1)
/// and ONLY entity-type can lift the person above the orgs.
const DISTRACTORS_PER_ITEM: usize = 3;

pub fn generate_ontology_fixtures(items: usize) -> (Vec<CorpusItem>, Vec<SmokeCase>) {
    // Clamp to the localizer-list length: beyond it, `pick` would reuse a base
    // token across items and the {place}/{person} localizers would collide, so two
    // items' persons would both answer one query (an unlabeled-correct distractor)
    // and silently corrupt P@1. Better a smaller clean sample than a larger dirty
    // one — surface the cap loudly rather than quietly degrade.
    let max_clean = PERSONS.len().min(PLACES.len());
    let items = if items > max_clean {
        eprintln!(
            "ontology harness: clamping items {items} → {max_clean} (localizer lists exhausted; \
             larger N would collide place/person tokens and corrupt P@1)"
        );
        max_clean
    } else {
        items
    };
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut corpus: Vec<CorpusItem> = Vec::with_capacity(items * (DISTRACTORS_PER_ITEM + 1));
    let mut cases: Vec<SmokeCase> = Vec::with_capacity(items * 2);

    for i in 0..items {
        let person = pick(PERSONS, i);
        let place = pick(PLACES, i);
        let person_id = format!("onto-person-{i:04}");

        // EQUI-LEXICAL frame: the person and every org distractor use the IDENTICAL
        // sentence — "<entity> was registered for the <place> trade summit." — so
        // the ONLY thing that distinguishes them is the entity's TYPE. No content
        // word (conference/office) can be used as a lexical shortcut.
        let frame = |entity: &str| format!("{entity} was registered for the {place} trade summit.");

        corpus.push(CorpusItem {
            id: person_id.clone(),
            content: frame(&person),
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + chrono::Duration::minutes(8 * i as i64),
        });
        for d in 0..DISTRACTORS_PER_ITEM {
            // Stride the org list so each item's distractors are distinct orgs.
            let org = pick(ORGS, i + 1 + d * 7);
            corpus.push(CorpusItem {
                id: format!("onto-org-{i:04}-{d}"),
                content: frame(&org),
                memory_type: "fact".to_string(),
                tags: vec![],
                created_at: base + chrono::Duration::minutes(8 * i as i64 + 1 + d as i64),
            });
        }

        // Type-qualified: all K+1 memories share {place}+frame; only the entity TYPE
        // distinguishes the person (gold) from the K orgs. SmokeCategory::Entity.
        cases.push(SmokeCase {
            id: format!("onto-type-{i:04}"),
            category: SmokeCategory::Entity,
            query: format!("Which person was registered for the {place} trade summit?"),
            fixture_corpus_id: "ontology".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: person_id.clone(),
                grade: 3,
            }],
        });
        // Control: name the person directly (BM25-solvable) → proves the corpus is
        // retrievable, so the type-qualified vs control gap isolates the rerank.
        cases.push(SmokeCase {
            id: format!("onto-ctrl-{i:04}"),
            category: SmokeCategory::Code,
            query: format!("What was {person} registered for?"),
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
                multihop_recall_at_10: mean(records, "entity", |r| r.recall_at_k),
                onehop_recall_at_10: mean(records, "code", |r| r.recall_at_k),
                multihop_mrr: mean(records, "entity", |r| r.mrr),
                multihop_p_at_1: mean(records, "entity", |r| r.p_at_1),
                onehop_p_at_1: mean(records, "code", |r| r.p_at_1),
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
