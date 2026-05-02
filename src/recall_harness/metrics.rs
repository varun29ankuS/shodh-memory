//! Retrieval-quality metrics for the recall harness.
//!
//! All metrics operate on a ranked list of candidate IDs (`retrieved`) against
//! a relevance judgement. Two judgement shapes are supported:
//! - **Binary** (`HashSet<Uuid>`): item is relevant or not. Used by `recall_at_k`,
//!   `precision_at_k`, `mrr`, `p_at_1`, `map`.
//! - **Graded** (`HashMap<Uuid, f32>`): item carries a non-negative relevance
//!   score. Used by `ndcg_at_k`. Items not present in the map are treated as
//!   non-relevant (gain = 0).
//!
//! `Metrics::compute` returns all six metrics in one pass and is the primary
//! entry point for the harness. Individual functions are exposed for unit
//! testing and for callers that only need a subset.
//!
//! # Conventions
//! - All metrics return `f64` in `[0.0, 1.0]`.
//! - `k = 0` returns `0.0` for top-k metrics (no division by zero).
//! - Empty `retrieved` returns `0.0` for every metric.
//! - Empty relevance set returns `0.0` for `recall_at_k`, `mrr`, `p_at_1`,
//!   `map`, and `ndcg_at_k` (the ideal DCG is `0`, so the ratio is undefined
//!   and conventionally reported as `0.0`).
//! - `precision_at_k` uses `k` as the denominator even when `retrieved.len() < k`,
//!   matching the IR convention in TREC and BEIR.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// All six retrieval-quality metrics for a single query.
///
/// Computed in a single pass via [`Metrics::compute`] to avoid recomputing
/// shared intermediates (intersection sets, hit positions).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Metrics {
    pub ndcg_at_k: f64,
    pub recall_at_k: f64,
    pub precision_at_k: f64,
    pub mrr: f64,
    pub p_at_1: f64,
    pub map: f64,
}

impl Metrics {
    /// Compute all metrics for one query.
    ///
    /// `retrieved` is the ranked candidate list (rank 1 = index 0).
    /// `relevance` carries graded relevance scores; a key's presence implies
    /// binary relevance for the binary-judgement metrics.
    /// `k` is the cutoff for top-k metrics; full-list metrics (MRR, MAP, P@1)
    /// ignore `k`.
    pub fn compute(retrieved: &[Uuid], relevance: &HashMap<Uuid, f32>, k: usize) -> Self {
        let relevant: HashSet<Uuid> = relevance.keys().copied().collect();
        Self {
            ndcg_at_k: ndcg_at_k(retrieved, relevance, k),
            recall_at_k: recall_at_k(retrieved, &relevant, k),
            precision_at_k: precision_at_k(retrieved, &relevant, k),
            mrr: mrr(retrieved, &relevant),
            p_at_1: p_at_1(retrieved, &relevant),
            map: map(retrieved, &relevant),
        }
    }
}

/// Precision at cutoff `k`: fraction of the top-`k` results that are relevant.
///
/// Denominator is `k` (TREC convention), not `min(k, retrieved.len())`. This
/// penalises short result lists, which is the desired behaviour when comparing
/// systems that may return fewer than `k` candidates.
pub fn precision_at_k(retrieved: &[Uuid], relevant: &HashSet<Uuid>, k: usize) -> f64 {
    if k == 0 || retrieved.is_empty() || relevant.is_empty() {
        return 0.0;
    }
    let cap = retrieved.len().min(k);
    let hits = retrieved[..cap]
        .iter()
        .filter(|id| relevant.contains(id))
        .count();
    hits as f64 / k as f64
}

/// Recall at cutoff `k`: fraction of relevant items found in the top-`k`.
pub fn recall_at_k(retrieved: &[Uuid], relevant: &HashSet<Uuid>, k: usize) -> f64 {
    if k == 0 || retrieved.is_empty() || relevant.is_empty() {
        return 0.0;
    }
    let cap = retrieved.len().min(k);
    let hits = retrieved[..cap]
        .iter()
        .filter(|id| relevant.contains(id))
        .count();
    hits as f64 / relevant.len() as f64
}

/// Mean Reciprocal Rank: `1 / rank` of the first relevant item, or `0.0` if
/// no relevant item appears in the list.
pub fn mrr(retrieved: &[Uuid], relevant: &HashSet<Uuid>) -> f64 {
    if retrieved.is_empty() || relevant.is_empty() {
        return 0.0;
    }
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Precision at rank 1: `1.0` if the top-ranked item is relevant, else `0.0`.
pub fn p_at_1(retrieved: &[Uuid], relevant: &HashSet<Uuid>) -> f64 {
    if retrieved.is_empty() || relevant.is_empty() {
        return 0.0;
    }
    if relevant.contains(&retrieved[0]) {
        1.0
    } else {
        0.0
    }
}

/// Mean Average Precision over a single query (a.k.a. AP).
///
/// `AP = (1/|R|) * Σ_{i: retrieved[i] ∈ R} precision_at_(i+1)`
///
/// Documents past the end of `retrieved` are treated as non-relevant (their
/// contribution is zero). This is the standard TREC AP definition.
pub fn map(retrieved: &[Uuid], relevant: &HashSet<Uuid>) -> f64 {
    if retrieved.is_empty() || relevant.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut sum = 0.0f64;
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id) {
            hits += 1;
            sum += hits as f64 / (i as f64 + 1.0);
        }
    }
    sum / relevant.len() as f64
}

/// Normalised Discounted Cumulative Gain at cutoff `k`.
///
/// `DCG@k = Σ_{i=1..=cap} rel_i / log2(i + 1)`, where `cap = min(k, retrieved.len())`
/// and `rel_i` is the graded relevance of the item at rank `i` (rank 1 = index 0).
/// The discount uses base-2 log starting at `log2(2) = 1` for rank 1, matching
/// Järvelin & Kekäläinen (2002) and BEIR.
///
/// `NDCG@k = DCG@k / IDCG@k`, where `IDCG@k` is the DCG of the ideal ranking
/// (top-`k` relevance scores in descending order). Returns `0.0` when
/// `IDCG@k = 0` (no relevant items, or all relevances are zero).
///
/// Negative relevance scores are clamped to `0.0` — they would otherwise let
/// a worse ranking score higher than the ideal, breaking the `[0, 1]` bound.
pub fn ndcg_at_k(retrieved: &[Uuid], relevance: &HashMap<Uuid, f32>, k: usize) -> f64 {
    if k == 0 || retrieved.is_empty() || relevance.is_empty() {
        return 0.0;
    }

    let cap = retrieved.len().min(k);
    let dcg: f64 = retrieved[..cap]
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let rel = relevance.get(id).copied().unwrap_or(0.0).max(0.0) as f64;
            rel / ((i as f64 + 2.0).log2())
        })
        .sum();

    // IDCG: sort relevance values descending, take top-k, apply same discount.
    let mut sorted: Vec<f64> = relevance.values().map(|&v| (v as f64).max(0.0)).collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("relevance scores are finite"));

    let idcg_cap = sorted.len().min(k);
    let idcg: f64 = sorted[..idcg_cap]
        .iter()
        .enumerate()
        .map(|(i, &rel)| rel / ((i as f64 + 2.0).log2()))
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Five fixed UUIDs so test assertions are stable and readable.
    fn ids() -> [Uuid; 5] {
        [
            Uuid::from_u128(1),
            Uuid::from_u128(2),
            Uuid::from_u128(3),
            Uuid::from_u128(4),
            Uuid::from_u128(5),
        ]
    }

    fn binary(items: &[Uuid]) -> HashSet<Uuid> {
        items.iter().copied().collect()
    }

    fn graded(items: &[(Uuid, f32)]) -> HashMap<Uuid, f32> {
        items.iter().copied().collect()
    }

    fn approx(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-9,
            "expected {b}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    // ---- Edge cases (apply uniformly to every metric) -----------------------

    #[test]
    fn empty_retrieved_returns_zero_everywhere() {
        let id = ids();
        let rel = binary(&[id[0]]);
        let grad = graded(&[(id[0], 1.0)]);
        assert_eq!(precision_at_k(&[], &rel, 5), 0.0);
        assert_eq!(recall_at_k(&[], &rel, 5), 0.0);
        assert_eq!(mrr(&[], &rel), 0.0);
        assert_eq!(p_at_1(&[], &rel), 0.0);
        assert_eq!(map(&[], &rel), 0.0);
        assert_eq!(ndcg_at_k(&[], &grad, 5), 0.0);
    }

    #[test]
    fn empty_relevant_returns_zero_without_panic() {
        let id = ids();
        let retrieved = id.to_vec();
        assert_eq!(precision_at_k(&retrieved, &HashSet::new(), 5), 0.0);
        assert_eq!(recall_at_k(&retrieved, &HashSet::new(), 5), 0.0);
        assert_eq!(mrr(&retrieved, &HashSet::new()), 0.0);
        assert_eq!(p_at_1(&retrieved, &HashSet::new()), 0.0);
        assert_eq!(map(&retrieved, &HashSet::new()), 0.0);
        assert_eq!(ndcg_at_k(&retrieved, &HashMap::new(), 5), 0.0);
    }

    #[test]
    fn k_zero_returns_zero_for_top_k_metrics() {
        let id = ids();
        let retrieved = id.to_vec();
        let rel = binary(&[id[0]]);
        let grad = graded(&[(id[0], 1.0)]);
        assert_eq!(precision_at_k(&retrieved, &rel, 0), 0.0);
        assert_eq!(recall_at_k(&retrieved, &rel, 0), 0.0);
        assert_eq!(ndcg_at_k(&retrieved, &grad, 0), 0.0);
    }

    // ---- Precision / Recall -------------------------------------------------

    #[test]
    fn precision_perfect_ranking() {
        let id = ids();
        let retrieved = vec![id[0], id[1], id[2], id[3], id[4]];
        let rel = binary(&[id[0], id[1], id[2]]);
        // 3 of top-5 are relevant.
        approx(precision_at_k(&retrieved, &rel, 5), 3.0 / 5.0);
        approx(precision_at_k(&retrieved, &rel, 3), 1.0);
    }

    #[test]
    fn precision_uses_k_not_actual_length() {
        let id = ids();
        let retrieved = vec![id[0], id[1]]; // only 2 results
        let rel = binary(&[id[0], id[1]]);
        // Both retrieved are relevant, but k=5 → 2/5, not 2/2.
        approx(precision_at_k(&retrieved, &rel, 5), 2.0 / 5.0);
    }

    #[test]
    fn recall_perfect_ranking() {
        let id = ids();
        let retrieved = vec![id[0], id[1], id[2], id[3], id[4]];
        let rel = binary(&[id[0], id[1], id[2]]);
        approx(recall_at_k(&retrieved, &rel, 3), 1.0);
        approx(recall_at_k(&retrieved, &rel, 2), 2.0 / 3.0);
    }

    // ---- MRR / P@1 ----------------------------------------------------------

    #[test]
    fn mrr_first_relevant_at_rank_three() {
        let id = ids();
        let retrieved = vec![id[3], id[4], id[0]];
        let rel = binary(&[id[0]]);
        approx(mrr(&retrieved, &rel), 1.0 / 3.0);
    }

    #[test]
    fn mrr_no_relevant_in_list_returns_zero() {
        let id = ids();
        let retrieved = vec![id[3], id[4]];
        let rel = binary(&[id[0]]);
        approx(mrr(&retrieved, &rel), 0.0);
    }

    #[test]
    fn p_at_1_top_relevant() {
        let id = ids();
        let retrieved = vec![id[0], id[3]];
        let rel = binary(&[id[0]]);
        approx(p_at_1(&retrieved, &rel), 1.0);
    }

    #[test]
    fn p_at_1_top_not_relevant() {
        let id = ids();
        let retrieved = vec![id[3], id[0]];
        let rel = binary(&[id[0]]);
        approx(p_at_1(&retrieved, &rel), 0.0);
    }

    // ---- MAP ----------------------------------------------------------------

    #[test]
    fn map_perfect_ranking_is_one() {
        let id = ids();
        let retrieved = vec![id[0], id[1], id[2]];
        let rel = binary(&[id[0], id[1], id[2]]);
        approx(map(&retrieved, &rel), 1.0);
    }

    #[test]
    fn map_known_case() {
        // Manning, Raghavan, Schütze IR textbook example:
        // retrieved = [R, N, R, N, R, N, N, N, R, R]
        // relevant total = 5
        // hits at ranks 1, 3, 5, 9, 10
        // AP = (1/5) * (1/1 + 2/3 + 3/5 + 4/9 + 5/10)
        //    = (1/5) * (1.0 + 0.6666... + 0.6 + 0.4444... + 0.5)
        //    = (1/5) * 3.2111...
        //    = 0.6422...
        let id = ids();
        // Use 10 distinct ids; reuse the 5-id helper by building extras.
        let extra: Vec<Uuid> = (6..=10).map(|n| Uuid::from_u128(n)).collect();
        let r = id[0];
        let n = id[1];
        let r2 = id[2];
        let n2 = id[3];
        let r3 = id[4];
        let n3 = extra[0];
        let n4 = extra[1];
        let n5 = extra[2];
        let r4 = extra[3];
        let r5 = extra[4];
        let retrieved = vec![r, n, r2, n2, r3, n3, n4, n5, r4, r5];
        let rel = binary(&[r, r2, r3, r4, r5]);
        let expected = (1.0 / 5.0) * (1.0 / 1.0 + 2.0 / 3.0 + 3.0 / 5.0 + 4.0 / 9.0 + 5.0 / 10.0);
        approx(map(&retrieved, &rel), expected);
    }

    // ---- NDCG ---------------------------------------------------------------

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        let id = ids();
        let retrieved = vec![id[0], id[1], id[2]];
        let grad = graded(&[(id[0], 3.0), (id[1], 2.0), (id[2], 1.0)]);
        approx(ndcg_at_k(&retrieved, &grad, 3), 1.0);
    }

    #[test]
    fn ndcg_reversed_ranking_known_value() {
        // Items have graded relevance 3, 2, 1 but are returned in reverse.
        // DCG  = 1/log2(2) + 2/log2(3) + 3/log2(4)
        //      = 1.0 + 2/1.5849625... + 3/2.0
        //      = 1.0 + 1.2618595... + 1.5
        //      = 3.7618595...
        // IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4)
        //      = 3.0 + 2/1.5849625... + 1/2.0
        //      = 3.0 + 1.2618595... + 0.5
        //      = 4.7618595...
        // NDCG = 3.7618595... / 4.7618595... ≈ 0.79002274...
        let id = ids();
        let retrieved = vec![id[2], id[1], id[0]];
        let grad = graded(&[(id[0], 3.0), (id[1], 2.0), (id[2], 1.0)]);

        let dcg = 1.0_f64 / 2.0_f64.log2() + 2.0_f64 / 3.0_f64.log2() + 3.0_f64 / 4.0_f64.log2();
        let idcg = 3.0_f64 / 2.0_f64.log2() + 2.0_f64 / 3.0_f64.log2() + 1.0_f64 / 4.0_f64.log2();
        approx(ndcg_at_k(&retrieved, &grad, 3), dcg / idcg);
    }

    #[test]
    fn ndcg_binary_relevance_perfect_at_top_then_irrelevant() {
        let id = ids();
        let retrieved = vec![id[0], id[3], id[4]]; // only id[0] relevant
        let grad = graded(&[(id[0], 1.0)]);
        // DCG  = 1/log2(2) = 1.0; IDCG = 1.0 → NDCG = 1.0
        approx(ndcg_at_k(&retrieved, &grad, 3), 1.0);
    }

    #[test]
    fn ndcg_irrelevant_at_top_then_relevant() {
        let id = ids();
        let retrieved = vec![id[3], id[0]]; // id[0] relevant at rank 2
        let grad = graded(&[(id[0], 1.0)]);
        // DCG = 0/log2(2) + 1/log2(3) = 1/log2(3); IDCG = 1.0 → NDCG = 1/log2(3)
        approx(ndcg_at_k(&retrieved, &grad, 2), 1.0 / 3.0_f64.log2());
    }

    #[test]
    fn ndcg_negative_relevance_clamped_to_zero() {
        let id = ids();
        let retrieved = vec![id[0], id[1]];
        let grad = graded(&[(id[0], 1.0), (id[1], -5.0)]);
        // id[1]'s gain is clamped to 0; ideal ranking is just id[0].
        // DCG = 1/log2(2) + 0 = 1.0; IDCG = 1.0 → NDCG = 1.0
        approx(ndcg_at_k(&retrieved, &grad, 2), 1.0);
    }

    #[test]
    fn ndcg_cuts_off_at_k() {
        // Relevant item is at rank 5 but k=3; should score 0.
        let id = ids();
        let extra: Vec<Uuid> = (6..=10).map(Uuid::from_u128).collect();
        let retrieved = vec![extra[0], extra[1], extra[2], extra[3], id[0]];
        let grad = graded(&[(id[0], 1.0)]);
        approx(ndcg_at_k(&retrieved, &grad, 3), 0.0);
    }

    // ---- Metrics::compute aggregator ---------------------------------------

    #[test]
    fn compute_returns_all_six_consistent_with_individual_fns() {
        let id = ids();
        let retrieved = vec![id[3], id[0], id[1]]; // miss, hit, hit
        let grad = graded(&[(id[0], 2.0), (id[1], 1.0), (id[2], 3.0)]);
        let rel = binary(&[id[0], id[1], id[2]]);

        let m = Metrics::compute(&retrieved, &grad, 3);

        approx(m.ndcg_at_k, ndcg_at_k(&retrieved, &grad, 3));
        approx(m.recall_at_k, recall_at_k(&retrieved, &rel, 3));
        approx(m.precision_at_k, precision_at_k(&retrieved, &rel, 3));
        approx(m.mrr, mrr(&retrieved, &rel));
        approx(m.p_at_1, p_at_1(&retrieved, &rel));
        approx(m.map, map(&retrieved, &rel));
    }

    #[test]
    fn compute_default_is_all_zero() {
        let m = Metrics::default();
        assert_eq!(
            m,
            Metrics {
                ndcg_at_k: 0.0,
                recall_at_k: 0.0,
                precision_at_k: 0.0,
                mrr: 0.0,
                p_at_1: 0.0,
                map: 0.0,
            }
        );
    }
}
