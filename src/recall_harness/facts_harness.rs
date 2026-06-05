//! E7 — fact-extraction QUALITY harness.
//!
//! The recall suite and the other capability harnesses (E3–E6) measure *retrieval*
//! — does the right memory rank #1. None of them measure whether the **facts the
//! system distills are correct**: the `+facts` recall layer contributes ≈0, so a
//! bad fact extractor reads as a null, not a regression. This harness closes that
//! hole: it plants memories whose ground-truth facts are known, forces the on-demand
//! consolidation path (`distill_facts`, which is otherwise gated off in the eval
//! ingest path), reads the extracted `SemanticFact`s back, and scores them.
//!
//! Design:
//! - Each **gold concept** is asserted by TWO differently-worded memories, so the
//!   consolidator has cross-memory support to cluster on, and so we can probe
//!   **dedup** (the same fact stated twice must yield ONE fact with support≥2, not
//!   two). Wording is aligned to the `FactType` categories the extractor recognises
//!   (preference / relationship / capability / procedure / definition).
//! - **No-fact distractors** assert no subject–attribute fact; they must yield no
//!   fact (a false-positive / hallucination probe).
//! - A gold concept is **recalled** iff some extracted fact contains all of its
//!   required tokens (entity-overlap match — `SemanticFact.fact` is a natural-language
//!   string, not a triple). An extracted fact is **correct** iff it matches ≥1 gold
//!   concept; everything else is a spurious extraction.
//!
//! Metrics: precision, recall, F1, dedup-rate, spurious (distractor) count,
//! per-type recall, and confidence calibration (correct vs spurious mean confidence).

use anyhow::{Context, Result};
use std::collections::BTreeMap;

use crate::recall_harness::fixtures::CorpusItem;
use crate::recall_harness::report::{FactsReport, FactsTypeRow};
use crate::recall_harness::runner::{build_manager, ingest_corpus, RunInputs, EVAL_USER};

/// A planted gold fact. A correct extraction must mention every token in `required`
/// (lowercase substring match against the extracted fact text + related entities).
struct GoldFact {
    concept: usize,
    required: &'static [&'static str],
    fact_type: &'static str,
}

/// (concept, type, two differently-worded assertions of the same fact, required tokens)
const CONCEPTS: &[(&str, [&str; 2], &[&str])] = &[
    ("preference", ["Alice prefers Rust for systems programming.",
                    "Alice's go-to language is Rust."], &["alice", "rust"]),
    ("preference", ["Bob loves hiking on the weekends.",
                    "Bob enjoys hiking in his free time."], &["bob", "hiking"]),
    ("preference", ["Carol's favorite food is sushi.",
                    "Carol almost always orders sushi."], &["carol", "sushi"]),
    ("relationship", ["The auth module depends on the JWT library.",
                      "Authentication in the service is built on JWT."], &["auth", "jwt"]),
    ("relationship", ["Dave moved to Paris last year.",
                      "Dave now lives in Paris."], &["dave", "paris"]),
    ("relationship", ["Shodh uses RocksDB for storage.",
                      "Storage in the engine is backed by RocksDB."], &["rocksdb", "storage"]),
    ("capability", ["The ingest API can handle ten thousand requests per second.",
                    "Our ingest API sustains ten thousand requests per second."], &["api", "requests"]),
    ("capability", ["Erin has played the piano for fifteen years.",
                    "Erin is an accomplished piano player."], &["erin", "piano"]),
    ("procedure", ["To deploy the server, run cargo build release.",
                   "Deploying the server means running cargo build."], &["server", "cargo"]),
    ("definition", ["A MemoryId is a wrapper around a UUID.",
                    "MemoryId wraps a UUID value."], &["memoryid", "uuid"]),
    ("definition", ["The knowledge graph uses Hebbian learning.",
                    "Hebbian learning drives the graph's edge weights."], &["graph", "hebbian"]),
    ("relationship", ["Vamana switches to SPANN above one hundred thousand vectors.",
                      "Above one hundred thousand vectors the index uses SPANN instead of Vamana."],
     &["vamana", "spann"]),
];

/// Sentences that assert no subject–attribute fact: must produce NO extracted fact.
const DISTRACTORS: &[&str] = &[
    "The weather was pleasant that afternoon.",
    "It happened sometime around noon.",
    "Everyone agreed it was a reasonable idea.",
    "Things went smoothly overall, more or less.",
    "There was quite a lot left to discuss.",
];

fn generate_facts_fixtures() -> (Vec<CorpusItem>, Vec<GoldFact>) {
    let base = chrono::DateTime::from_timestamp(1_700_000_000, 0)
        .unwrap_or_else(|| chrono::DateTime::<chrono::Utc>::MIN_UTC);
    let mut corpus = Vec::new();
    let mut gold = Vec::new();
    let mut t = 0i64;
    for (ci, (ftype, sentences, required)) in CONCEPTS.iter().enumerate() {
        gold.push(GoldFact { concept: ci, required, fact_type: ftype });
        for (si, sentence) in sentences.iter().enumerate() {
            corpus.push(CorpusItem {
                id: format!("fact-{ci:02}-{si}"),
                content: sentence.to_string(),
                memory_type: "Observation".to_string(),
                tags: vec![],
                created_at: base + chrono::Duration::minutes(t),
            });
            t += 1;
        }
    }
    for (di, d) in DISTRACTORS.iter().enumerate() {
        corpus.push(CorpusItem {
            id: format!("distractor-{di:02}"),
            content: d.to_string(),
            memory_type: "Observation".to_string(),
            tags: vec![],
            created_at: base + chrono::Duration::minutes(t),
        });
        t += 1;
    }
    (corpus, gold)
}

/// Run E7: ingest the planted corpus, force fact distillation, score the extracted
/// `SemanticFact`s against the gold concepts.
pub fn analyze_facts(inputs: &RunInputs) -> Result<FactsReport> {
    let (corpus, gold) = generate_facts_fixtures();
    let n_concepts = gold.len();

    let manager =
        build_manager(&inputs.storage_path.join("facts_run")).context("facts: build manager")?;
    ingest_corpus(&manager, &corpus).context("facts: ingest corpus")?;

    let user_mem = manager.get_user_memory(EVAL_USER).context("facts: get user memory")?;
    // Force on-demand extraction. min_support=1 (a single supporting memory is enough)
    // and min_age_days=0 to bypass the 7-day consolidation age gate — the harness
    // corpus is freshly ingested, so the production default would discard all of it.
    let consolidation = user_mem
        .read()
        .distill_facts(EVAL_USER, 1, 0)
        .context("facts: distill_facts")?;
    let facts = user_mem
        .read()
        .get_facts(EVAL_USER, 100_000)
        .context("facts: read back facts")?;

    // Lowercase haystack (statement + related entities) per extracted fact.
    let haystacks: Vec<(String, f32)> = facts
        .iter()
        .map(|f| {
            let hay = format!("{} {}", f.fact, f.related_entities.join(" ")).to_lowercase();
            (hay, f.confidence)
        })
        .collect();

    let matches = |req: &[&str], hay: &str| req.iter().all(|tok| hay.contains(tok));

    let mut concept_hits = vec![0usize; n_concepts];
    let mut correct = vec![false; facts.len()];
    for (fi, (hay, _)) in haystacks.iter().enumerate() {
        for g in &gold {
            if matches(g.required, hay) {
                concept_hits[g.concept] += 1;
                correct[fi] = true;
            }
        }
    }

    let recalled = concept_hits.iter().filter(|&&h| h >= 1).count();
    let dedup_ok = concept_hits.iter().filter(|&&h| h == 1).count();
    let correct_extracted = correct.iter().filter(|&&c| c).count();
    let total_extracted = facts.len();
    let spurious = total_extracted.saturating_sub(correct_extracted);

    let precision = if total_extracted == 0 {
        0.0
    } else {
        correct_extracted as f64 / total_extracted as f64
    };
    let recall = if n_concepts == 0 {
        0.0
    } else {
        recalled as f64 / n_concepts as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    // Per-type recall.
    let mut by_type: BTreeMap<String, FactsTypeRow> = BTreeMap::new();
    for g in &gold {
        let row = by_type.entry(g.fact_type.to_string()).or_insert(FactsTypeRow {
            gold: 0,
            recalled: 0,
            recall: 0.0,
        });
        row.gold += 1;
        if concept_hits[g.concept] >= 1 {
            row.recalled += 1;
        }
    }
    for row in by_type.values_mut() {
        row.recall = if row.gold == 0 {
            0.0
        } else {
            row.recalled as f64 / row.gold as f64
        };
    }

    // Confidence calibration: mean confidence of correct vs spurious extractions.
    let mean_conf = |want_correct: bool| -> f64 {
        let vals: Vec<f32> = haystacks
            .iter()
            .enumerate()
            .filter(|(fi, _)| correct[*fi] == want_correct)
            .map(|(_, (_, c))| *c)
            .collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().map(|c| *c as f64).sum::<f64>() / vals.len() as f64
        }
    };

    Ok(FactsReport {
        suite: "facts".to_string(),
        git_sha: inputs.git_sha.clone(),
        timestamp: chrono::DateTime::from_timestamp(1_700_000_000, 0)
            .unwrap_or_else(|| chrono::DateTime::<chrono::Utc>::MIN_UTC),
        gold_concepts: n_concepts,
        distractors: DISTRACTORS.len(),
        facts_extracted: total_extracted,
        facts_extracted_this_cycle: consolidation.facts_extracted,
        correct_extracted,
        recalled_concepts: recalled,
        precision,
        recall,
        f1,
        dedup_ok,
        spurious,
        mean_confidence_correct: mean_conf(true),
        mean_confidence_spurious: mean_conf(false),
        by_type,
    })
}
