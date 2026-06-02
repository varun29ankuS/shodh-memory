//! E3 — controlled multi-hop harness (synthetic planted 2-hop chains).
//!
//! Motivation: on LoCoMo recall@k the graph leg contributes only ~+0.0185 over
//! vector alone, because most LoCoMo gold is reachable by BM25/vector directly —
//! the corpus does not *stress* multi-hop traversal. That makes LoCoMo recall@k
//! structurally unable to reward graph/relational work: every graph change reads
//! as a null. This harness builds the opposite: a corpus where the gold for a
//! query is reachable ONLY by traversing the entity graph, so the spreading-
//! activation (graph) stage is the deciding factor and improvements to graph
//! construction / relations have a metric that can actually move.
//!
//! Construction (fully deterministic, no model, no fixtures on disk):
//! - For each chain `i` we mint three unique pseudo-name entities a → b → c.
//! - `seed_i`  : "{a} works closely with {b}."        tags = [a, b]
//! - `link_i`  : "{b} manages the … team with {c}."    tags = [b, c]
//!   Putting the chain entities in `tags` guarantees they become graph nodes and
//!   that the pairwise co-occurrence edge (a–b, b–c) is created, independent of
//!   whether NER recognises the nonce names.
//! - 2-hop case : query mentions ONLY `a`, gold = `link_i`. There is no lexical
//!   overlap between the query and `link_i` (a/c never co-occur, and every link
//!   memory shares the verb "manages", so BM25 cannot pick the right one). The
//!   only path to `link_i` is a → b (via `seed_i`) → `link_i` (b's other memory).
//! - 1-hop control : query mentions `b`, gold = `link_i`. BM25 solves this — it
//!   is the contrast that proves the 2-hop cases isolate graph traversal.
//!
//! The same planted corpus + cases are run through every `LayerMode`
//! (`--layer all`), so the per-layer 2-hop recall is the headline: if
//! `+spreading` lifts 2-hop recall far above `vamana_only`/`+bm25`, the graph
//! carries the multi-hop load and relation-quality work can be measured here.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};

use crate::memory::types::LayerMode;
use crate::recall_harness::fixtures::{
    CorpusItem, RelevanceJudgement, SmokeCase, SmokeCategory,
};
use crate::recall_harness::report::{MultiHopLayerRow, MultiHopReport};
use crate::recall_harness::runner::{run_smoke_suite_with_ranks, RunInputs};

/// Default number of planted chains. Each chain yields 2 memories (seed, link)
/// and 2 cases (2-hop, 1-hop control). 60 chains → 120 memories, 60+60 cases,
/// enough mutual distractors that BM25's "manages" match cannot discriminate.
pub const DEFAULT_CHAINS: usize = 60;

/// Syllable pools for deterministic, unique, capitalised pseudo-names. The names
/// only need to be unique strings that read as proper nouns; correctness does
/// not depend on NER tagging them (tags create the nodes regardless).
const PREFIXES: &[&str] = &[
    "Vor", "Mes", "Cal", "Bren", "Tor", "Quil", "Dav", "Fen", "Gor", "Hal", "Jen", "Kor", "Lan",
    "Mor", "Nor", "Pol", "Rin", "Sel", "Tav", "Wen",
];
const SUFFIXES: &[&str] = &[
    "land", "ker", "then", "wick", "ston", "ridge", "mont", "field", "gate", "born", "dale",
    "ford", "grove", "haven", "worth", "holm", "bury", "crest", "mere", "vale",
];

/// Deterministic unique name for entity index `n`. Combines prefix × suffix and,
/// once that 400-name space is exhausted, appends a numeric block so names stay
/// unique for arbitrarily many chains.
fn entity_name(n: usize) -> String {
    let p = PREFIXES[n % PREFIXES.len()];
    let s = SUFFIXES[(n / PREFIXES.len()) % SUFFIXES.len()];
    let block = n / (PREFIXES.len() * SUFFIXES.len());
    if block == 0 {
        format!("{p}{s}")
    } else {
        format!("{p}{s}{block}")
    }
}

/// Generate the planted-chain corpus and cases. Deterministic: identical output
/// for identical `chains`, so the runner's cross-repeat byte-identity check holds.
pub fn generate_multihop_fixtures(chains: usize) -> (Vec<CorpusItem>, Vec<SmokeCase>) {
    // Fixed base timestamp keeps `created_at` deterministic (no Utc::now()).
    let base = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();

    let mut corpus: Vec<CorpusItem> = Vec::with_capacity(chains * 2);
    let mut cases: Vec<SmokeCase> = Vec::with_capacity(chains * 2);

    for i in 0..chains {
        let a = entity_name(3 * i);
        let b = entity_name(3 * i + 1);
        let c = entity_name(3 * i + 2);

        let seed_id = format!("mh-seed-{i:04}");
        let link_id = format!("mh-link-{i:04}");

        // seed_i: a — b (the first hop). Tags guarantee a,b nodes + a–b edge.
        corpus.push(CorpusItem {
            id: seed_id.clone(),
            content: format!("{a} works closely with {b} on the regional account."),
            memory_type: "fact".to_string(),
            tags: vec![a.clone(), b.clone()],
            created_at: base + chrono::Duration::minutes(2 * i as i64),
        });

        // link_i: b — c (the second hop, and the gold for both cases). Every link
        // memory shares the verb "manages" so BM25 from a 2-hop query (which has
        // no b/c term) cannot discriminate the right one.
        corpus.push(CorpusItem {
            id: link_id.clone(),
            content: format!("{b} manages the delivery team together with {c}."),
            memory_type: "fact".to_string(),
            tags: vec![b.clone(), c.clone()],
            created_at: base + chrono::Duration::minutes(2 * i as i64 + 1),
        });

        // 2-hop case: mentions ONLY `a`; gold = link_i. Reachable solely via the
        // graph (a → b → link_i). No lexical bridge to link_i.
        cases.push(SmokeCase {
            id: format!("mh-2hop-{i:04}"),
            category: SmokeCategory::MultiHop,
            query: format!("Who does the colleague that {a} works closely with manage?"),
            fixture_corpus_id: "multihop".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: link_id.clone(),
                grade: 3,
            }],
        });

        // 1-hop control: mentions `b` directly; gold = link_i. BM25-solvable.
        cases.push(SmokeCase {
            id: format!("mh-1hop-{i:04}"),
            category: SmokeCategory::SingleHop,
            query: format!("Who does {b} manage?"),
            fixture_corpus_id: "multihop".to_string(),
            relevant: vec![RelevanceJudgement {
                corpus_item_id: link_id,
                grade: 3,
            }],
        });
    }

    (corpus, cases)
}

/// Write the generated fixtures as JSONL into `dir`, returning the two paths.
fn write_fixtures(
    dir: &std::path::Path,
    corpus: &[CorpusItem],
    cases: &[SmokeCase],
) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating multihop fixture dir {}", dir.display()))?;
    let corpus_path = dir.join("multihop_corpus.jsonl");
    let cases_path = dir.join("multihop_cases.jsonl");

    let mut cbuf = String::new();
    for item in corpus {
        cbuf.push_str(&serde_json::to_string(item)?);
        cbuf.push('\n');
    }
    std::fs::write(&corpus_path, cbuf)
        .with_context(|| format!("writing {}", corpus_path.display()))?;

    let mut qbuf = String::new();
    for case in cases {
        qbuf.push_str(&serde_json::to_string(case)?);
        qbuf.push('\n');
    }
    std::fs::write(&cases_path, qbuf).with_context(|| format!("writing {}", cases_path.display()))?;

    Ok((corpus_path, cases_path))
}

/// Build the planted corpus, run it through every `LayerMode`, and report the
/// per-layer 2-hop vs 1-hop recall. The `+spreading − vamana_only` 2-hop delta is
/// the graph leg's isolated multi-hop contribution.
pub fn analyze_multihop(inputs: &RunInputs, chains: usize) -> Result<MultiHopReport> {
    let (corpus, cases) = generate_multihop_fixtures(chains);
    let multihop_cases = cases
        .iter()
        .filter(|c| c.category == SmokeCategory::MultiHop)
        .count();
    let onehop_cases = cases
        .iter()
        .filter(|c| c.category == SmokeCategory::SingleHop)
        .count();

    let fixture_dir = inputs.storage_path.join("multihop_fixtures");
    let (corpus_path, cases_path) = write_fixtures(&fixture_dir, &corpus, &cases)?;

    // Reuse the production eval machinery: ingest once, run the case set under
    // every cumulative LayerMode. suite != "smoke" → structural validation only.
    let run_inputs = RunInputs {
        storage_path: inputs.storage_path.join("multihop_run"),
        corpus_path: Some(corpus_path),
        cases_path: Some(cases_path),
        suite: "multihop".to_string(),
        git_sha: inputs.git_sha.clone(),
        repeats: 1,
        layer_modes: LayerMode::ALL.to_vec(),
        age_days: inputs.age_days,
    };

    let out = run_smoke_suite_with_ranks(&run_inputs)
        .context("multihop suite run failed")?;

    // Per layer: mean recall@10 split by case category.
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
    let mean_mrr = |records: &[crate::recall_harness::report::PerCaseRecord], cat: &str| -> f64 {
        let vals: Vec<f64> = records
            .iter()
            .filter(|r| r.category == cat)
            .map(|r| r.mrr)
            .collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };

    // Order rows by the cumulative LayerMode ladder.
    let mut rows: Vec<MultiHopLayerRow> = Vec::new();
    for mode in LayerMode::ALL {
        let key = mode.report_key().to_string();
        if let Some(records) = out.per_case_by_layer.get(&key) {
            rows.push(MultiHopLayerRow {
                layer: key,
                multihop_recall_at_10: mean(records, "multi_hop"),
                onehop_recall_at_10: mean(records, "single_hop"),
                multihop_mrr: mean_mrr(records, "multi_hop"),
            });
        }
    }

    Ok(MultiHopReport {
        chains,
        multihop_cases,
        onehop_cases,
        rows,
    })
}
