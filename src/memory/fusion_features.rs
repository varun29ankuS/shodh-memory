//! Per-query fusion feature export (approach C, stage 2).
//!
//! Hand-tuned single features for per-query leg trust all re-trade categories
//! (bm25 peakedness: run 27244857747; vector↔BM25 agreement: runs 27255684449 +
//! 27265607332 — the largest multi_hop gain measured, +0.051, but single_hop
//! −0.037 regardless of the spread). Stage 2 therefore FITS a multi-feature
//! model offline. This module exports, per query, the candidate-pool features
//! the fusion can see at rank time PLUS the per-leg gold ranks (which directly
//! label "which leg should have been trusted" for this query).
//!
//! Same thread-local pattern as [`super::gold_funnel`]: the recall harness arms
//! it per query with the gold ids, the FLAT fusion path populates it (a no-op
//! when unarmed — production pays nothing), the harness drains it and writes
//! JSONL for the offline fit.

use super::types::MemoryId;
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    static STATE: RefCell<Option<ExportState>> = const { RefCell::new(None) };
}

struct ExportState {
    gold: HashSet<MemoryId>,
    features: Option<FusionFeatures>,
}

/// Candidate-pool features visible to the fusion at rank time, plus per-leg
/// gold ranks (the supervision signal for the offline trust fit).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct FusionFeatures {
    /// Hybrid pool size (vector ∪ BM25 candidates).
    pub n_hybrid: usize,
    /// Candidates with a positive BM25 / vector component.
    pub n_bm_pos: usize,
    pub n_vec_pos: usize,
    /// max/mean over positive scores (1.0 when degenerate). The stage-1 features.
    pub bm_peak: f32,
    pub vec_peak: f32,
    /// Fraction of the vector top-K also in the BM25 top-K (K=10, capped by pool).
    pub agreement_top10: f32,
    /// Raw best scores per leg.
    pub max_bm: f32,
    pub max_vec: f32,
    /// Graph leg: candidate count and best activation.
    pub n_graph: usize,
    pub graph_max_activation: f32,
    /// Rank (0-based) of the best gold id when candidates are ordered by the
    /// vector / BM25 / graph score alone. None = no gold in that leg's pool.
    pub gold_vec_rank: Option<usize>,
    pub gold_bm_rank: Option<usize>,
    pub gold_graph_rank: Option<usize>,
}

/// Arm the exporter for the next recall with this query's gold ids.
pub fn begin(gold: HashSet<MemoryId>) {
    STATE.with(|c| {
        *c.borrow_mut() = Some(ExportState {
            gold,
            features: None,
        });
    });
}

/// Disarm and return the recorded features (None if never armed or the fusion
/// path that populates them did not run).
pub fn take() -> Option<FusionFeatures> {
    STATE.with(|c| c.borrow_mut().take().and_then(|s| s.features))
}

/// True when armed — lets the fusion skip the (sort-heavy) feature computation.
pub fn is_armed() -> bool {
    STATE.with(|c| c.borrow().is_some())
}

/// Run `f` with the armed gold set to produce the features, and store them.
/// No-op when unarmed.
pub fn record_with(f: impl FnOnce(&HashSet<MemoryId>) -> FusionFeatures) {
    STATE.with(|c| {
        let mut borrow = c.borrow_mut();
        let Some(state) = borrow.as_mut() else {
            return;
        };
        let features = f(&state.gold);
        state.features = Some(features);
    });
}
