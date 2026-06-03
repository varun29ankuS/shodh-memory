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

use anyhow::{Context, Result};

use crate::memory::types::LayerMode;
use crate::recall_harness::report::{DecayReport, DecayRow};
use crate::recall_harness::runner::{run_smoke_suite_with_ranks, RunInputs};

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
