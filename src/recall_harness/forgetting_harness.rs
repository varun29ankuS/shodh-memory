//! E6 — decay / forgetting / stability harness.
//!
//! Tests the homeostasis claim — "forgets what doesn't matter, but stays stable
//! and does not catastrophically forget what does." Runs the recall suite at
//! increasing simulated ages (the knowledge-graph edges are aged via
//! `simulate_edge_aging` at the production ~6h cadence before querying) and
//! reports recall@k / ndcg / mrr as a function of age.
//!
//! Interpretation: a FLAT curve = stable memory (good homeostasis — aged edges
//! don't erase retrievable gold). A sharply DECLINING curve = catastrophic
//! forgetting (decay erodes recall). A modest decline on a corpus with no
//! reinforcement is expected and healthy; a cliff is the failure mode.
//!
//! Reuses the production recall path end-to-end; the only knob is `age_days`.

use std::collections::HashSet;

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use uuid::Uuid;

use crate::memory::retrieval::RetrievalOutcome;
use crate::memory::types::{LayerMode, MemoryId};
use crate::memory::Query;
use crate::recall_harness::fixtures::CorpusItem;
use crate::recall_harness::report::{
    DecayReport, DecayRow, SelectiveForgettingReport, SelectiveForgettingRow,
};
use crate::recall_harness::runner::{
    build_manager, ingest_corpus, run_smoke_suite_with_ranks, RunInputs, EVAL_USER,
};

/// Default age points (days) for the stability curve.
pub const DEFAULT_AGES: &[f64] = &[0.0, 7.0, 30.0, 90.0, 365.0];

/// Run the configured suite at each age and tabulate recall@k vs age. Each age
/// gets its own ingest (fresh storage subdir) so the aged state is isolated.
pub fn analyze_forgetting(inputs: &RunInputs, ages: &[f64]) -> Result<DecayReport> {
    let mut rows: Vec<DecayRow> = Vec::with_capacity(ages.len());
    for &age in ages {
        let ri = RunInputs {
            storage_path: inputs
                .storage_path
                .join(format!("age_{}", (age * 10.0) as i64)),
            corpus_path: inputs.corpus_path.clone(),
            cases_path: inputs.cases_path.clone(),
            suite: inputs.suite.clone(),
            git_sha: inputs.git_sha.clone(),
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: age,
        };
        let out = run_smoke_suite_with_ranks(&ri)
            .with_context(|| format!("forgetting run at age_days={age}"))?;
        let full = out
            .report
            .layers
            .get("full")
            .or_else(|| out.report.layers.values().next())
            .context("no layer report produced")?;
        rows.push(DecayRow {
            age_days: age,
            recall_at_10: full.recall_at_10,
            ndcg_at_10: full.ndcg_at_10,
            mrr: full.mrr,
        });
    }
    Ok(DecayReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        rows,
    })
}

// ============================ Selective forgetting ============================
//
// The global stability curve above answers "does aging erase recall?" but NOT the
// real cognitive claim: "forgets what doesn't matter, retains what does." This
// second harness isolates SELECTIVITY with a competitive, two-population design:
//
//   - For each group, plant M equi-lexical memories that all match ONE query
//     equally (same frame + group token; the only per-memory difference is a
//     trailing nonce that NEVER appears in the query, so lexical/vector ranking
//     among them is a tie that only activation/edge state can break).
//   - K of the M are "important" — reinforced via the real `reinforce_recall`
//     (Helpful) path for `reinforce_cycles` rounds. The remaining M−K are
//     "trivial" — never reinforced.
//   - Age the graph edges cumulatively and, at each age, measure how many
//     important vs trivial memories survive in the top-`RETENTION_TOP_K` slots.
//
// The signal is DIVERGENCE = important_retention − trivial_retention GROWING with
// age. Equal decay (divergence ~0) means forgetting is indiscriminate.
//
// Measurement caveat (honest): `recall` bumps access on every returned memory. We
// set `max_results` wide enough to return the whole group, so that bump lands on
// important and trivial SYMMETRICALLY and cannot bias the divergence; if anything
// it adds equal access to trivials, making the test conservative.

/// Equi-lexical memories planted per competitive group.
const MEMORIES_PER_GROUP: usize = 12;
/// Reinforced ("important") memories per group; the rest are trivial.
const IMPORTANT_PER_GROUP: usize = 4;
/// Scarce top-k cutoff at which retention is scored (< group size so slots
/// compete).
const RETENTION_TOP_K: usize = 6;
/// Default number of competitive groups.
pub const DEFAULT_GROUPS: usize = 24;
/// Default reinforcement cycles applied to each important memory before aging.
pub const DEFAULT_REINFORCE_CYCLES: usize = 6;

const GROUP_TOKENS: &[&str] = &[
    "Aldrin", "Bractor", "Crelm", "Dynra", "Embex", "Frusk", "Glave", "Hexil", "Ironna", "Jolfen",
    "Kresk", "Luvox", "Marnic", "Nexor", "Oryll", "Plinth", "Qorvex", "Rundle", "Sythe", "Tavolk",
    "Ulmer", "Vandt", "Wexol", "Yarrow",
];

fn group_token(g: usize) -> String {
    let base = GROUP_TOKENS[g % GROUP_TOKENS.len()];
    let block = g / GROUP_TOKENS.len();
    if block == 0 {
        base.to_string()
    } else {
        format!("{base}{block}")
    }
}

/// Generated competitive forgetting fixtures.
pub struct SelectiveFixtures {
    pub corpus: Vec<CorpusItem>,
    /// One natural-language query per group (matches all M memories equally).
    pub queries: Vec<String>,
    /// Per group: corpus ids of the important (reinforced) memories.
    pub important_ids: Vec<Vec<String>>,
    /// Per group: corpus ids of the trivial (never-reinforced) memories.
    pub trivial_ids: Vec<Vec<String>>,
}

pub fn generate_selective_fixtures(groups: usize) -> SelectiveFixtures {
    let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    // Spread important indices evenly across the M memories so importance does NOT
    // correlate with creation recency (otherwise recency, not reinforcement, would
    // explain any divergence). With M=12, K=4 → stride 3 → important = {0,3,6,9}.
    let stride = MEMORIES_PER_GROUP / IMPORTANT_PER_GROUP;

    let mut corpus = Vec::with_capacity(groups * MEMORIES_PER_GROUP);
    let mut queries = Vec::with_capacity(groups);
    let mut important_ids = Vec::with_capacity(groups);
    let mut trivial_ids = Vec::with_capacity(groups);

    for g in 0..groups {
        let token = group_token(g);
        let mut imp = Vec::with_capacity(IMPORTANT_PER_GROUP);
        let mut triv = Vec::with_capacity(MEMORIES_PER_GROUP - IMPORTANT_PER_GROUP);
        for m in 0..MEMORIES_PER_GROUP {
            // Equi-lexical frame: every memory in the group shares
            // "Regarding the {token} initiative ... was filed."; only the trailing
            // entry word differs, and it never appears in the query.
            let entry = ENTRY_WORDS[m % ENTRY_WORDS.len()];
            let id = format!("self-{g:04}-{m:02}");
            corpus.push(CorpusItem {
                id: id.clone(),
                content: format!("Regarding the {token} initiative, the {entry} entry was filed."),
                memory_type: "fact".to_string(),
                tags: vec![token.clone()],
                created_at: base + chrono::Duration::minutes((MEMORIES_PER_GROUP * g + m) as i64),
            });
            if m % stride == 0 && imp.len() < IMPORTANT_PER_GROUP {
                imp.push(id);
            } else {
                triv.push(id);
            }
        }
        queries.push(format!("What was filed regarding the {token} initiative?"));
        important_ids.push(imp);
        trivial_ids.push(triv);
    }

    SelectiveFixtures {
        corpus,
        queries,
        important_ids,
        trivial_ids,
    }
}

/// Distinct, content-neutral entry words so each memory in a group is a separate
/// record while staying equi-lexical w.r.t. the query (none appear in the query).
const ENTRY_WORDS: &[&str] = &[
    "amber", "cobalt", "crimson", "emerald", "indigo", "ivory", "jade", "maroon", "ochre",
    "russet", "saffron", "teal",
];

/// Run the selective-forgetting study: ingest the competitive corpus, reinforce
/// the important memories via the real `reinforce_recall` path, then age the graph
/// cumulatively and report important-vs-trivial retention divergence per age.
pub fn analyze_selective_forgetting(
    inputs: &RunInputs,
    groups: usize,
    ages: &[f64],
    reinforce_cycles: usize,
) -> Result<SelectiveForgettingReport> {
    let fx = generate_selective_fixtures(groups);
    let storage = inputs.storage_path.join("selective_forget");
    let manager = build_manager(&storage)?;
    let id_map = ingest_corpus(&manager, &fx.corpus).context("ingesting selective corpus")?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // Wide enough to return every group member, so measurement access-bumps are
    // symmetric across important/trivial within a group.
    let wide = (MEMORIES_PER_GROUP * 3).max(40);
    let recall_ids = |q: &str| -> Vec<Uuid> {
        let query = Query {
            query_text: Some(q.to_string()),
            max_results: wide,
            layers: LayerMode::Full,
            ..Default::default()
        };
        match system.read().recall(&query) {
            Ok(mems) => mems.iter().map(|m| m.id.0).collect(),
            Err(_) => Vec::new(),
        }
    };

    // Reinforcement phase: recall the group query (co-activates the whole group),
    // then apply Helpful feedback to ONLY the important memories. Repeat.
    for _ in 0..reinforce_cycles.max(1) {
        for (g, q) in fx.queries.iter().enumerate() {
            let _ = recall_ids(q);
            let imp_ids: Vec<MemoryId> = fx.important_ids[g]
                .iter()
                .filter_map(|cid| id_map.get(cid).copied())
                .map(MemoryId)
                .collect();
            if !imp_ids.is_empty() {
                let _ = system
                    .read()
                    .reinforce_recall(&imp_ids, RetrievalOutcome::Helpful);
            }
        }
    }

    // Cumulative aging timeline: sort ages ascending and age by the delta to each.
    let mut sorted_ages = ages.to_vec();
    sorted_ages.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut rows = Vec::with_capacity(sorted_ages.len());
    let mut prev = 0.0f64;
    for age in sorted_ages {
        let delta = age - prev;
        if delta > 0.0 {
            system
                .read()
                .simulate_edge_aging(delta, super::decay_sim::PRODUCTION_CADENCE_HOURS)
                .with_context(|| format!("aging edges to {age} days"))?;
        }
        prev = age;

        let (mut imp_hit, mut imp_tot, mut triv_hit, mut triv_tot) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (g, q) in fx.queries.iter().enumerate() {
            let ranked = recall_ids(q);
            let topk: HashSet<Uuid> = ranked.into_iter().take(RETENTION_TOP_K).collect();
            for cid in &fx.important_ids[g] {
                if let Some(u) = id_map.get(cid) {
                    imp_tot += 1.0;
                    if topk.contains(u) {
                        imp_hit += 1.0;
                    }
                }
            }
            for cid in &fx.trivial_ids[g] {
                if let Some(u) = id_map.get(cid) {
                    triv_tot += 1.0;
                    if topk.contains(u) {
                        triv_hit += 1.0;
                    }
                }
            }
        }
        let important_retention = imp_hit / imp_tot.max(1.0);
        let trivial_retention = triv_hit / triv_tot.max(1.0);
        rows.push(SelectiveForgettingRow {
            age_days: age,
            important_retention,
            trivial_retention,
            divergence: important_retention - trivial_retention,
        });
    }

    Ok(SelectiveForgettingReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        reinforce_cycles: reinforce_cycles.max(1),
        pairs: groups * IMPORTANT_PER_GROUP,
        rows,
    })
}
