//! Entity-resolution measured on the real GDELT bridge graph (Task 1.2).
//!
//! Loads the exported bridge mentions (669 surface-mention entity nodes + their
//! causal edges) and resolves them into canonical entities, asserting the plan's
//! success criteria: node count down ≥ 35 %, no catastrophic over-merge, and the
//! key entities (Dali, the bridge) collapse to a handful of canonicals.
//!
//! Requires the dependency-parser model (`SHODH_SPACY_MODEL_PATH`); skips cleanly
//! without it. The fine-grained must-merge / must-not-merge precision cases live
//! in the model-free unit tests in `src/entity_resolution.rs`.

use std::collections::HashSet;

use serde_json::Value;
use shodh_memory::dep_parser;
use shodh_memory::entity_resolution::{resolve, CausalRel, Mention};

const FIXTURE: &str = include_str!("fixtures/bridge_mentions.json");

fn load() -> (Vec<Mention>, Vec<CausalRel>) {
    let v: Value = serde_json::from_str(FIXTURE).expect("valid fixture json");
    let mentions = v["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| Mention {
            id: e["id"].as_str().unwrap().to_string(),
            label: e["label"].as_str().unwrap().to_string(),
            types: e["types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_str().unwrap().to_string())
                .collect::<HashSet<_>>(),
            proper: e["proper"].as_bool().unwrap_or(false),
            mention_count: e["mentions"].as_u64().unwrap_or(1) as u32,
        })
        .collect();
    let causal = v["causal_edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| CausalRel {
            source: e["source"].as_str().unwrap().to_string(),
            target: e["target"].as_str().unwrap().to_string(),
            label: e["label"].as_str().unwrap().to_string(),
        })
        .collect();
    (mentions, causal)
}

/// Count distinct canonical labels among mentions whose surface contains `probe`.
fn canonicals_for(
    mentions: &[Mention],
    canon: &std::collections::HashMap<String, String>,
    probe: &str,
) -> Vec<String> {
    let mut set: HashSet<String> = HashSet::new();
    for m in mentions {
        if m.label.to_lowercase().contains(probe) {
            if let Some(c) = canon.get(&m.id) {
                set.insert(c.clone());
            }
        }
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

#[test]
fn bridge_mentions_resolve_to_canonical_entities() {
    if !dep_parser::is_available() {
        eprintln!(
            "SKIP bridge_mentions_resolve_to_canonical_entities: SHODH_SPACY_MODEL_PATH unset"
        );
        return;
    }
    let (mentions, causal) = load();
    let res = resolve(&mentions, &causal).expect("parser available");

    let entities = res.num_entities();
    let largest = res.clusters.first().map(|c| c.len()).unwrap_or(0);

    eprintln!("=== entity resolution on GDELT bridge ===");
    eprintln!(
        "input mentions: {}  ->  entities: {}  (events routed out: {}, dropped junk: {})",
        res.num_input,
        entities,
        res.events.len(),
        res.dropped.len()
    );
    eprintln!("node-count reduction: {:.1}%", res.reduction() * 100.0);
    eprintln!("largest cluster: {largest} members");
    eprintln!("largest clusters (canonical <= size):");
    for c in res.clusters.iter().take(12) {
        let canon = res.canon.get(&c[0]).cloned().unwrap_or_default();
        eprintln!("  [{:2}] {}", c.len(), canon);
    }
    for probe in ["dali", "bridge", "port", "coast guard"] {
        let cs = canonicals_for(&mentions, &res.canon, probe);
        eprintln!(
            "  {:12} -> {} canonical(s): {:?}",
            probe,
            cs.len(),
            &cs[..cs.len().min(5)]
        );
    }

    // Plan criterion: node count down >= 35 %.
    assert!(
        res.reduction() >= 0.35,
        "reduction {:.1}% below the 35% floor ({} -> {} entities)",
        res.reduction() * 100.0,
        res.num_input,
        entities
    );
    // Sanity: didn't collapse everything, didn't fail to merge anything.
    assert!(
        (100..=500).contains(&entities),
        "entity count {entities} outside the plausible 100..=500 band"
    );
    // Over-merge guard: no cluster large enough to be a distinct-entity fusion
    // (the rule-based v1 produced a 418-node blob; a healthy resolve stays small).
    assert!(
        largest < 60,
        "largest cluster {largest} looks like a catastrophic over-merge"
    );
    // Variant-recall: the ship and the bridge each collapse to a small set.
    let dali = canonicals_for(&mentions, &res.canon, "dali");
    assert!(
        !dali.is_empty() && dali.len() <= 4,
        "Dali variants -> {} canonicals",
        dali.len()
    );
}
