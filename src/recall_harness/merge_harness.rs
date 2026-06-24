//! ER merge-quality harness — measures cross-document entity resolution, and in
//! particular the CATASTROPHIC over-merge (false merge) that the linking harness
//! cannot see. Plants known coref clusters (the same entity under surface variants)
//! and distinct entities with tempting near-collisions, then checks pairwise whether
//! the graph merged what it should and — critically — did NOT merge what it must not.
//!
//! `false_merge_rate` is the load-bearing number: a lever that lifts `merge_recall`
//! while raising `false_merge_rate` is a NET LOSS, because a wrong merge is far worse
//! than a missed one.

use std::collections::BTreeSet;

use anyhow::Result;
use chrono::{TimeZone, Utc};

use crate::recall_harness::fixtures::CorpusItem;
use crate::recall_harness::report::MergeReport;
use crate::recall_harness::runner::{build_manager, ingest_corpus, RunInputs, EVAL_USER};

/// (surface form, true entity id). Variants that share an id SHOULD resolve together;
/// different ids must NOT, even when the surfaces collide on a shared token.
fn labels() -> Vec<(&'static str, &'static str)> {
    vec![
        // same entity, surface variants — should merge
        ("Rajesh Kumar", "E1"),
        ("R. Kumar", "E1"),
        ("Kumar Rajesh", "E1"),
        ("Priya Sharma", "E2"),
        ("P. Sharma", "E2"),
        ("Mohammed Ali", "E3"),
        ("Muhammad Ali", "E3"), // transliteration variant
        // distinct entities with tempting collisions — must NOT merge
        ("Rajesh Verma", "E4"),  // shares "Rajesh" with E1
        ("Mohammed Khan", "E5"), // shares "Mohammed" with E3
        ("Anil Gupta", "E6"),
        ("Sunil Gupta", "E7"), // shares "Gupta" with E6
    ]
}

/// Ingest one document per planted mention, then resolve each surface through the
/// graph and score the pairwise merge decisions against the planted truth.
pub fn analyze_merge(inputs: &RunInputs) -> Result<MergeReport> {
    let labels = labels();
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let corpus: Vec<CorpusItem> = labels
        .iter()
        .enumerate()
        .map(|(i, (surf, _))| CorpusItem {
            id: format!("merge-{i:03}"),
            content: format!("{surf} attended the planning meeting in the capital."),
            memory_type: "fact".to_string(),
            tags: vec![surf.to_string()],
            created_at: base + chrono::Duration::minutes(i as i64),
        })
        .collect();

    let storage = inputs.storage_path.join("merge_run");
    let manager = build_manager(&storage)?;
    let _ = ingest_corpus(&manager, &corpus)?;
    let graph = manager.get_user_graph(EVAL_USER)?;

    // Canonical key per surface: the resolved entity uuid, else a unique singleton
    // (an unresolved surface merges with nothing).
    let canon: Vec<String> = {
        let g = graph.read();
        labels
            .iter()
            .map(|(surf, _)| match g.find_entity_by_name(surf) {
                Ok(Some(ent)) => ent.name.to_lowercase(),
                _ => format!("__unresolved__{surf}"),
            })
            .collect()
    };

    let (mut tp, mut fp, mut fn_, mut tn) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            let same_true = labels[i].1 == labels[j].1;
            let same_graph = canon[i] == canon[j];
            match (same_true, same_graph) {
                (true, true) => tp += 1,
                (false, true) => fp += 1, // FALSE MERGE — the catastrophic one
                (true, false) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }
    }
    let merge_precision = if tp + fp == 0 {
        1.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let merge_recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    };
    let false_merge_rate = if fp + tn == 0 {
        0.0
    } else {
        fp as f64 / (fp + tn) as f64
    };
    let true_clusters = labels.iter().map(|(_, e)| *e).collect::<BTreeSet<_>>().len();

    Ok(MergeReport {
        git_sha: inputs.git_sha.clone(),
        mentions: labels.len(),
        true_clusters,
        merge_precision,
        merge_recall,
        false_merge_rate,
        false_merges: fp,
        correct_merges: tp,
        missed_merges: fn_,
    })
}
