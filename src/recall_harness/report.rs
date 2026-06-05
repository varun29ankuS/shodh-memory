//! Recall-eval report types and JSON schema.
//!
//! These types are the contract consumed by the CI gate (RH-5) and by
//! humans via baseline diffs. They are deliberately self-contained and
//! `serde`-serialisable so a report file can be diffed across runs and
//! across embedder swaps without touching the engine code.
//!
//! See issue #266 for the schema.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::metrics::Metrics;

/// `k` value used by the L1 smoke suite for top-`k` metrics.
pub const SMOKE_K: usize = 10;

/// Top-level recall-eval report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Report {
    /// Suite identifier — `"smoke"` for the L1 suite.
    pub suite: String,
    /// Embedding model name, e.g. `"minilm-l6-v2"`.
    pub embedder: String,
    /// Git SHA of the working tree the run was produced from.
    pub git_sha: String,
    /// RFC3339 timestamp at the start of the run.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Per-pipeline-layer aggregates. RH-4 emits a single `"full"` layer;
    /// RH-8 will widen this map.
    pub layers: BTreeMap<String, LayerReport>,
    /// Per-category aggregates for the `full` layer.
    pub by_category: BTreeMap<String, CategoryReport>,
    /// Total number of cases run.
    pub case_count: usize,
    /// Number of independent ingest+query passes the report aggregates over.
    /// RH-12 (#272). For each case, latency is the median across this many
    /// samples; quality metrics are taken from the first repeat after a
    /// determinism check confirms identical rank lists across all repeats.
    /// Defaults to `1` when reading reports written before RH-12 landed so
    /// older baselines remain parseable.
    #[serde(default = "default_repeats")]
    pub repeats: usize,
    /// Cases or metrics that failed validation or regression.
    pub failures: Vec<Failure>,
}

fn default_repeats() -> usize {
    1
}

/// Per-case diagnostics for the highest (gated) layer.
///
/// Not part of [`Report`] / `baseline.json` — those stay aggregate-only so the
/// committed baseline is small and diffable. This is emitted to a side artifact
/// (`recall-eval --per-case-output`) so a human can answer "which query is weak
/// and which relevant items did it drop?" instead of only seeing a category
/// average move. `missed` lists the relevant `corpus_item_id`s that fell
/// outside the top-`k`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerCaseRecord {
    pub case_id: String,
    pub category: String,
    pub query: String,
    pub ndcg_at_k: f64,
    pub recall_at_k: f64,
    pub mrr: f64,
    pub p_at_1: f64,
    pub relevant_total: usize,
    pub relevant_found: usize,
    pub missed: Vec<String>,
    /// Recall computed over a wider cutoff of the SAME retrieved list, to split
    /// ranking failures (gold present but ranked >10) from retrieval-reach
    /// failures (gold absent from the candidate funnel). Only informative when
    /// the harness queries with `max_results >= 50/100` (via `RECALL_DIAG_K`);
    /// otherwise the list is shorter than the cutoff and these equal
    /// `recall_at_k`. The gap `recall_at_100 - recall_at_k` is the upper bound
    /// on what a perfect reranker over the top-100 pool could recover.
    #[serde(default)]
    pub recall_at_50: f64,
    #[serde(default)]
    pub recall_at_100: f64,
}

/// One age point in the E6 decay/stability curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecayRow {
    pub age_days: f64,
    #[serde(rename = "recall@10")]
    pub recall_at_10: f64,
    #[serde(rename = "ndcg@10")]
    pub ndcg_at_10: f64,
    pub mrr: f64,
}

/// E6 decay/forgetting report: recall@k vs simulated age. Flat = stable memory
/// (good homeostasis); a cliff = catastrophic forgetting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecayReport {
    pub suite: String,
    pub git_sha: String,
    pub rows: Vec<DecayRow>,
}

/// One age point in the SELECTIVE-forgetting curve. For each age, `important_*`
/// and `trivial_*` are retention rates (recall@10) of the reinforced vs the
/// never-reinforced population under the SAME query; `divergence` =
/// important − trivial. A real cognitive memory keeps `important_retention` high
/// while `trivial_retention` decays, so `divergence` GROWS with age. Equal decay
/// (divergence flat near 0) means forgetting is indiscriminate — the failure the
/// global stability curve cannot see.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectiveForgettingRow {
    pub age_days: f64,
    pub important_retention: f64,
    pub trivial_retention: f64,
    pub divergence: f64,
}

/// Selective-forgetting report: retention divergence (important − trivial) vs age.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectiveForgettingReport {
    pub suite: String,
    pub git_sha: String,
    /// Reinforcement cycles applied to each important memory before aging.
    pub reinforce_cycles: usize,
    /// Number of (important, trivial) competitive pairs.
    pub pairs: usize,
    pub rows: Vec<SelectiveForgettingRow>,
}

/// One row in the unified ablation matrix: a named config (a set of query-time
/// flag overrides) and its aggregate metrics over the suite. The whole point is
/// a single, re-runnable table where each fix/component is a row you can see and
/// compare against the baseline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AblationRow {
    /// Human-readable config name (e.g. `baseline`, `+graph-expand(K5)`).
    pub name: String,
    /// The env overrides applied for this row (key=value), for reproducibility.
    pub flags: Vec<String>,
    #[serde(rename = "recall@10")]
    pub recall_at_10: f64,
    #[serde(rename = "ndcg@10")]
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub p_at_1: f64,
    /// Per-category recall@10 (category name → value), so a config that helps one
    /// capability but hurts another is visible, not hidden in the average.
    pub by_category_recall: std::collections::BTreeMap<String, f64>,
}

/// Unified ablation report: one ingest, N query-time configs, one comparison
/// table. The living artifact for "see and update ablation studies".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AblationReport {
    pub suite: String,
    pub git_sha: String,
    pub case_count: usize,
    pub rows: Vec<AblationRow>,
}

/// One layer's row in the E3 multi-hop ladder: recall@10 split by 2-hop
/// (graph-only-reachable) vs 1-hop (BM25-solvable control) cases.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiHopLayerRow {
    /// `report_key()` of the `LayerMode` (e.g. `vamana_only`, `+spreading`).
    pub layer: String,
    /// Mean recall@10 over the planted 2-hop cases (gold reachable only via
    /// graph traversal). The `+spreading − vamana_only` delta on this column is
    /// the graph leg's isolated multi-hop contribution.
    pub multihop_recall_at_10: f64,
    /// Mean recall@10 over the 1-hop control cases (gold lexically findable).
    pub onehop_recall_at_10: f64,
    /// Mean MRR over the 2-hop cases.
    pub multihop_mrr: f64,
    /// Mean P@1 over the 2-hop / capability cases. For controlled harnesses with
    /// few equi-confusable candidates per query (temporal: 3 states; ontology:
    /// 1 person + K orgs), recall@10 saturates at 1.0 because the whole confusable
    /// set fits in the top-10 window — only P@1 (is the CAPABILITY-correct item
    /// ranked #1 above its distractors?) discriminates. This is the headline
    /// metric for those harnesses.
    #[serde(default)]
    pub multihop_p_at_1: f64,
    /// Mean P@1 over the control cases.
    #[serde(default)]
    pub onehop_p_at_1: f64,
}

/// E3 controlled multi-hop report: per-layer 2-hop vs 1-hop recall over a
/// synthetic planted-chain corpus where 2-hop gold is reachable only by graph
/// traversal. The metric LoCoMo recall@k cannot provide.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiHopReport {
    pub chains: usize,
    pub multihop_cases: usize,
    pub onehop_cases: usize,
    /// One row per `LayerMode`, ordered along the cumulative ladder.
    pub rows: Vec<MultiHopLayerRow>,
}

/// E7 fact-extraction QUALITY report: precision/recall/F1 of distilled facts vs
/// planted gold concepts, plus dedup, spurious-extraction, per-type recall, and
/// confidence calibration. The first harness to score fact CORRECTNESS — the other
/// suites only measure retrieval rank, so a weak fact extractor reads as a null.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactsReport {
    pub suite: String,
    pub git_sha: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub gold_concepts: usize,
    pub distractors: usize,
    /// Facts present in the store after distillation.
    pub facts_extracted: usize,
    /// Facts the distillation cycle reported creating (from ConsolidationResult).
    pub facts_extracted_this_cycle: usize,
    pub correct_extracted: usize,
    pub recalled_concepts: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// Recalled concepts represented by EXACTLY one fact (dedup worked, not split).
    pub dedup_ok: usize,
    /// Extracted facts matching no gold concept (hallucination / distractor leak).
    pub spurious: usize,
    pub mean_confidence_correct: f64,
    pub mean_confidence_spurious: f64,
    pub by_type: std::collections::BTreeMap<String, FactsTypeRow>,
}

/// Per-`FactType` recall row for [`FactsReport`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactsTypeRow {
    pub gold: usize,
    pub recalled: usize,
    pub recall: f64,
}

/// Aggregate metrics for one pipeline layer across all cases.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerReport {
    #[serde(rename = "ndcg@10")]
    pub ndcg_at_10: f64,
    #[serde(rename = "recall@10")]
    pub recall_at_10: f64,
    #[serde(rename = "precision@10")]
    pub precision_at_10: f64,
    pub mrr: f64,
    #[serde(rename = "p@1")]
    pub p_at_1: f64,
    pub map: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    /// Smallest per-case median latency (ms) across the suite.
    /// Defaults to `0.0` for back-compat with pre-RH-12 reports where
    /// the field was absent. RH-12 (#272).
    #[serde(default)]
    pub latency_min_ms: f64,
    /// Largest per-case median latency (ms) across the suite.
    /// Defaults to `0.0` for back-compat. RH-12 (#272).
    #[serde(default)]
    pub latency_max_ms: f64,
    /// Inter-quartile range (p75 − p25) of per-case median latencies.
    /// A small IQR with stable medians is the signal we want; a wide IQR
    /// means our timing is dominated by hardware noise. RH-12 (#272).
    #[serde(default)]
    pub latency_iqr_ms: f64,
}

/// Aggregate metrics for one category — same fields as `LayerReport` minus
/// latency, since the relevant signal at category granularity is quality.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CategoryReport {
    #[serde(rename = "ndcg@10")]
    pub ndcg_at_10: f64,
    #[serde(rename = "recall@10")]
    pub recall_at_10: f64,
    #[serde(rename = "precision@10")]
    pub precision_at_10: f64,
    pub mrr: f64,
    #[serde(rename = "p@1")]
    pub p_at_1: f64,
    pub map: f64,
    pub case_count: usize,
}

/// A failure entry in the report.
///
/// Two flavours are emitted:
/// - `kind = "regression"` when a baseline comparison exceeds the tolerance.
/// - `kind = "case"` when a single case has zero relevant retrievals at all.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Failure {
    pub kind: String,
    pub detail: String,
}

/// Compute (p50, p95, p99) over a vector of latencies in milliseconds.
///
/// Uses nearest-rank percentile on a sorted copy. Returns `(0.0, 0.0, 0.0)`
/// for an empty input. With small N (e.g. 30 smoke cases) p95 and p99 are
/// noisy by construction; the harness reports them anyway because their
/// trend across runs is still informative.
pub fn latency_percentiles(samples: &[f64]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    (
        nearest_rank(&sorted, 0.50),
        nearest_rank(&sorted, 0.95),
        nearest_rank(&sorted, 0.99),
    )
}

/// Compute the median of a slice of f64 samples.
///
/// For odd-length slices returns the middle element. For even-length slices
/// returns the arithmetic mean of the two middle elements (continuous
/// median). Returns `0.0` for empty input — callers that care about
/// emptiness handle it before calling.
pub fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute (min, max, iqr) over per-case median latencies.
///
/// IQR is `p75 − p25` using nearest-rank, matching the rest of the harness.
/// Returns `(0.0, 0.0, 0.0)` for empty input. The trio is reported alongside
/// p50/p95/p99 so reviewers can spot a tight median masking a wide spread —
/// the exact failure mode RH-12 was opened to surface (issue #272).
pub fn latency_distribution_stats(per_case_medians: &[f64]) -> (f64, f64, f64) {
    if per_case_medians.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = per_case_medians.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let iqr = nearest_rank(&sorted, 0.75) - nearest_rank(&sorted, 0.25);
    (min, max, iqr)
}

fn nearest_rank(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty(), "caller checks emptiness");
    debug_assert!((0.0..=1.0).contains(&p), "p must be in [0,1]");
    // Nearest-rank: index = ceil(p * N) - 1, clamped to [0, N-1].
    let n = sorted.len();
    let idx = ((p * n as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

/// Aggregate per-case metrics into a [`LayerReport`] (mean of each metric).
///
/// `latencies_ms` is the per-case representative latency — under RH-12 this
/// is the median across `repeats` independent runs, not a raw sample. The
/// percentile and distribution stats are computed over the same slice.
pub fn aggregate_layer(per_case: &[Metrics], latencies_ms: &[f64]) -> LayerReport {
    let n = per_case.len() as f64;
    let mean = |sel: fn(&Metrics) -> f64| -> f64 {
        if n == 0.0 {
            0.0
        } else {
            per_case.iter().map(sel).sum::<f64>() / n
        }
    };
    let (p50, p95, p99) = latency_percentiles(latencies_ms);
    let (lat_min, lat_max, lat_iqr) = latency_distribution_stats(latencies_ms);
    LayerReport {
        ndcg_at_10: mean(|m| m.ndcg_at_k),
        recall_at_10: mean(|m| m.recall_at_k),
        precision_at_10: mean(|m| m.precision_at_k),
        mrr: mean(|m| m.mrr),
        p_at_1: mean(|m| m.p_at_1),
        map: mean(|m| m.map),
        latency_p50_ms: p50,
        latency_p95_ms: p95,
        latency_p99_ms: p99,
        latency_min_ms: lat_min,
        latency_max_ms: lat_max,
        latency_iqr_ms: lat_iqr,
    }
}

/// Aggregate per-case metrics for one category into a [`CategoryReport`].
pub fn aggregate_category(per_case: &[Metrics]) -> CategoryReport {
    let n = per_case.len();
    let div = n as f64;
    let mean = |sel: fn(&Metrics) -> f64| -> f64 {
        if div == 0.0 {
            0.0
        } else {
            per_case.iter().map(sel).sum::<f64>() / div
        }
    };
    CategoryReport {
        ndcg_at_10: mean(|m| m.ndcg_at_k),
        recall_at_10: mean(|m| m.recall_at_k),
        precision_at_10: mean(|m| m.precision_at_k),
        mrr: mean(|m| m.mrr),
        p_at_1: mean(|m| m.p_at_1),
        map: mean(|m| m.map),
        case_count: n,
    }
}

/// Compare the `full` layer of `current` against `baseline`.
///
/// Returns the list of regressions where `current < baseline - tolerance_pct%
/// of baseline` for any of the gating metrics: `ndcg@10`, `recall@10`,
/// `mrr`, `p@1`. Latency is reported but not gated because per-run hardware
/// noise dominates.
pub fn compare_to_baseline(
    baseline: &Report,
    current: &Report,
    tolerance_pct: f64,
) -> Vec<Failure> {
    let Some(base_full) = baseline.layers.get("full") else {
        return vec![Failure {
            kind: "infrastructure".to_string(),
            detail: "baseline report has no `full` layer".to_string(),
        }];
    };
    let Some(cur_full) = current.layers.get("full") else {
        return vec![Failure {
            kind: "infrastructure".to_string(),
            detail: "current report has no `full` layer".to_string(),
        }];
    };

    let mut failures = Vec::new();
    let frac = tolerance_pct / 100.0;
    let mut check = |name: &str, base: f64, cur: f64| {
        // Treat baselines at zero as a special case: any drop below zero is
        // impossible, any non-zero current is an improvement, and zero current
        // is a no-op. So nothing to gate.
        if base <= 0.0 {
            return;
        }
        let allowed_drop = base * frac;
        if cur + allowed_drop < base {
            failures.push(Failure {
                kind: "regression".to_string(),
                detail: format!(
                    "{name}: baseline {base:.4}, current {cur:.4}, allowed drop {allowed_drop:.4}"
                ),
            });
        }
    };

    check("ndcg@10", base_full.ndcg_at_10, cur_full.ndcg_at_10);
    check("recall@10", base_full.recall_at_10, cur_full.recall_at_10);
    check("mrr", base_full.mrr, cur_full.mrr);
    check("p@1", base_full.p_at_1, cur_full.p_at_1);

    failures
}

/// Graph-reachability diagnostic: for each case, is each gold memory reachable
/// from the query's seed entities within N entity-hops in the *built* knowledge
/// graph — ignoring the (separately-audited) spreading-activation weights and
/// prune threshold. This isolates a pure topology question — "does an
/// associative path EXIST?" — from "does the current activation math surface
/// it?". It answers whether the graph-native fix cluster can lift multi_hop:
/// if the stranded gold is graph-reachable, better activation/ranking will
/// surface it; if it is not, the deficit is entity-extraction/graph-construction
/// or a non-entity-mediated hop (a retrieval-reach problem the graph cannot fix).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReachabilityReport {
    pub suite: String,
    pub git_sha: String,
    pub max_hops: usize,
    pub overall: ReachabilityCategory,
    pub by_category: BTreeMap<String, ReachabilityCategory>,
    /// Degree distribution of the built graph — the direct scoreboard for
    /// anti-hub construction tuning (IDF-at-birth edges, hub-degree cap). The
    /// hub pathology only appears at corpus scale (LoCoMo), so this is where the
    /// tuning is actually measurable.
    #[serde(default)]
    pub graph: GraphStructure,
}

/// Graph degree-distribution summary (for anti-hub construction tuning).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphStructure {
    pub total_entities: usize,
    pub total_edges: usize,
    /// Highest single-entity degree (the worst hub — should fall with tuning).
    pub max_degree: usize,
    pub mean_degree: f64,
    /// Number of entities whose degree exceeds the hub report threshold.
    pub hub_count: usize,
    pub hub_threshold: usize,
    /// Top entity degrees, descending (the hub tail).
    pub top_degrees: Vec<usize>,
}

/// Reachability tallies for one category (cumulative within-N-hops counts).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ReachabilityCategory {
    /// Number of cases in this category.
    pub cases: usize,
    /// Cases where NER found no query entity that resolves to a graph node —
    /// the query has no associative anchor at all (an extraction gap, not a
    /// traversal gap).
    pub cases_no_seed: usize,
    /// Total gold memories summed across the category's cases.
    pub gold_total: usize,
    /// Gold reachable with the seed entity directly mentioning it (1 hop).
    pub reachable_within_1: usize,
    /// Gold reachable within 2 entity-hops (one bridge entity) — the canonical
    /// double-hop path.
    pub reachable_within_2: usize,
    /// Gold reachable within 3 entity-hops.
    pub reachable_within_3: usize,
    /// Gold not reachable from any seed entity within `max_hops`.
    pub unreachable: usize,
    /// Gold episodes directly attached (1 hop) to ≥2 DISTINCT query seeds — the
    /// multi-seed discrimination signal G5 needs. If this is ~0, queries resolve
    /// to too few graph cues for multi-seed graph reasoning to do anything, and
    /// the lever is richer query→graph cue extraction, not episode scoring.
    #[serde(default)]
    pub gold_multi_seed: usize,
    /// Cases whose query resolved to ≥2 distinct graph seed entities at all.
    #[serde(default)]
    pub cases_multi_seed: usize,
}

/// Learning-curve diagnostic: does recall of a memory IMPROVE as the memory is
/// used? The flagship test of the "smarter with use" claim that no single-shot
/// retrieval metric can see. Protocol: for cases whose gold sits at a
/// recallable-but-not-top rank (headroom), repeatedly recall the query while
/// applying `Helpful` feedback to the gold; track the gold's rank and score per
/// cycle. A genuine associative/Hebbian memory shows the rank DECREASE (gold
/// climbs) and score RISE over cycles; a static retriever stays flat.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningCurveReport {
    pub suite: String,
    pub git_sha: String,
    pub cycles: usize,
    /// One arm per reinforcement-outcome (Helpful / Neutral / Misleading), each
    /// run on a FRESH ingest. The reward-gradient test: a genuine reward signal
    /// pushes the gold UP under Helpful, DOWN under Misleading, and leaves it
    /// flat under Neutral. If all three look the same, the reward loop is inert.
    pub arms: Vec<LearningCurveArm>,
}

/// One reinforcement-outcome arm of the learning-curve diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningCurveArm {
    /// "Helpful" | "Neutral" | "Misleading".
    pub outcome: String,
    /// Cases whose cold gold-rank fell in the headroom band (≥2, ≤cap).
    pub tracked_cases: usize,
    /// Mean gold rank at each cycle: index 0 = cold, 1..=cycles after each
    /// reinforcement. DECREASING under Helpful = learning.
    pub mean_rank_by_cycle: Vec<f64>,
    /// Mean gold score at each cycle.
    pub mean_score_by_cycle: Vec<f64>,
    /// Cases where final rank < initial rank (the memory got easier to recall).
    pub improved: usize,
    pub worsened: usize,
    pub unchanged: usize,
    /// Mean (final_rank − initial_rank); NEGATIVE = climbed with use.
    pub mean_rank_delta: f64,
    pub mean_score_delta: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(values: f64) -> Metrics {
        Metrics {
            ndcg_at_k: values,
            recall_at_k: values,
            precision_at_k: values,
            mrr: values,
            p_at_1: values,
            map: values,
        }
    }

    #[test]
    fn percentiles_empty_is_zero() {
        assert_eq!(latency_percentiles(&[]), (0.0, 0.0, 0.0));
    }

    #[test]
    fn percentiles_single_sample() {
        assert_eq!(latency_percentiles(&[42.0]), (42.0, 42.0, 42.0));
    }

    #[test]
    fn percentiles_known_values() {
        // Samples 1..=100, sorted. Nearest-rank: p50→50, p95→95, p99→99.
        let s: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let (p50, p95, p99) = latency_percentiles(&s);
        assert_eq!(p50, 50.0);
        assert_eq!(p95, 95.0);
        assert_eq!(p99, 99.0);
    }

    #[test]
    fn aggregate_layer_means_each_metric() {
        let cases = [metrics(0.0), metrics(0.5), metrics(1.0)];
        let r = aggregate_layer(&cases, &[10.0, 20.0, 30.0]);
        assert!((r.ndcg_at_10 - 0.5).abs() < 1e-12);
        assert!((r.mrr - 0.5).abs() < 1e-12);
        assert_eq!(r.latency_p50_ms, 20.0);
    }

    #[test]
    fn aggregate_layer_handles_empty() {
        let r = aggregate_layer(&[], &[]);
        assert_eq!(r.ndcg_at_10, 0.0);
        assert_eq!(r.latency_p50_ms, 0.0);
    }

    #[test]
    fn aggregate_category_records_count() {
        let cases = [metrics(0.5), metrics(0.5)];
        let r = aggregate_category(&cases);
        assert_eq!(r.case_count, 2);
        assert!((r.recall_at_10 - 0.5).abs() < 1e-12);
    }

    fn report_with_full(ndcg: f64, recall: f64, mrr: f64, p1: f64) -> Report {
        let mut layers = BTreeMap::new();
        layers.insert(
            "full".to_string(),
            LayerReport {
                ndcg_at_10: ndcg,
                recall_at_10: recall,
                precision_at_10: 0.2,
                mrr,
                p_at_1: p1,
                map: 0.5,
                latency_p50_ms: 0.0,
                latency_p95_ms: 0.0,
                latency_p99_ms: 0.0,
                latency_min_ms: 0.0,
                latency_max_ms: 0.0,
                latency_iqr_ms: 0.0,
            },
        );
        Report {
            suite: "smoke".to_string(),
            embedder: "test".to_string(),
            git_sha: "deadbeef".to_string(),
            timestamp: chrono::Utc::now(),
            layers,
            by_category: BTreeMap::new(),
            case_count: 30,
            repeats: 1,
            failures: vec![],
        }
    }

    #[test]
    fn no_regressions_when_metrics_match() {
        let baseline = report_with_full(0.6, 0.7, 0.5, 0.4);
        let current = report_with_full(0.6, 0.7, 0.5, 0.4);
        assert!(compare_to_baseline(&baseline, &current, 2.0).is_empty());
    }

    #[test]
    fn small_drop_within_tolerance_is_accepted() {
        // 1% drop in ndcg with 2% tolerance is fine.
        let baseline = report_with_full(0.60, 0.70, 0.50, 0.40);
        let current = report_with_full(0.594, 0.70, 0.50, 0.40);
        assert!(compare_to_baseline(&baseline, &current, 2.0).is_empty());
    }

    #[test]
    fn drop_beyond_tolerance_is_flagged() {
        // 5% drop with 2% tolerance fails.
        let baseline = report_with_full(0.60, 0.70, 0.50, 0.40);
        let current = report_with_full(0.57, 0.70, 0.50, 0.40);
        let failures = compare_to_baseline(&baseline, &current, 2.0);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].kind, "regression");
        assert!(failures[0].detail.contains("ndcg@10"));
    }

    #[test]
    fn improvements_are_not_flagged() {
        let baseline = report_with_full(0.60, 0.70, 0.50, 0.40);
        let current = report_with_full(0.80, 0.90, 0.70, 0.60);
        assert!(compare_to_baseline(&baseline, &current, 2.0).is_empty());
    }

    #[test]
    fn missing_baseline_full_layer_emits_infra_failure() {
        let mut baseline = report_with_full(0.6, 0.7, 0.5, 0.4);
        baseline.layers.clear();
        let current = report_with_full(0.6, 0.7, 0.5, 0.4);
        let failures = compare_to_baseline(&baseline, &current, 2.0);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].kind, "infrastructure");
    }

    // -------------------- RH-12 (#272) -------------------------------------
    //
    // The harness now reports per-case median latency aggregated over N
    // independent ingest+query repeats. These tests pin the median helper,
    // the distribution-stats helper, and the back-compat parser path so a
    // stale baseline.json keeps loading after RH-12 ships.

    #[test]
    fn median_of_odd_length_picks_middle_element() {
        assert!((median(&[10.0, 20.0, 30.0, 40.0, 50.0]) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn median_of_even_length_averages_middle_pair() {
        assert!((median(&[10.0, 20.0, 30.0, 40.0]) - 25.0).abs() < 1e-12);
    }

    #[test]
    fn median_of_unsorted_input_handles_order() {
        assert!((median(&[40.0, 10.0, 30.0, 20.0, 50.0]) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn median_empty_is_zero() {
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn latency_distribution_stats_known_values() {
        let s: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let (min, max, iqr) = latency_distribution_stats(&s);
        assert_eq!(min, 1.0);
        assert_eq!(max, 100.0);
        // Nearest-rank p75 = 75, p25 = 25 → IQR = 50.
        assert_eq!(iqr, 50.0);
    }

    #[test]
    fn latency_distribution_stats_empty_is_zero_triple() {
        assert_eq!(latency_distribution_stats(&[]), (0.0, 0.0, 0.0));
    }

    #[test]
    fn report_serde_back_compat_old_baseline_has_no_repeats_field() {
        // Mimic a pre-RH-12 baseline.json: no `repeats`, no latency_min/max/iqr.
        // Must parse with default repeats=1 and zeroed latency stats.
        let json = r#"{
            "suite": "smoke",
            "embedder": "minilm-l6-v2",
            "git_sha": "abc123",
            "timestamp": "2026-05-03T14:36:36.058394900Z",
            "layers": {
                "full": {
                    "ndcg@10": 0.8,
                    "recall@10": 0.85,
                    "precision@10": 0.17,
                    "mrr": 0.88,
                    "p@1": 0.83,
                    "map": 0.75,
                    "latency_p50_ms": 130.0,
                    "latency_p95_ms": 143.0,
                    "latency_p99_ms": 145.0
                }
            },
            "by_category": {},
            "case_count": 30,
            "failures": []
        }"#;
        let report: Report =
            serde_json::from_str(json).expect("old baseline.json must still parse");
        assert_eq!(report.repeats, 1);
        let full = report.layers.get("full").expect("full layer present");
        assert_eq!(full.latency_min_ms, 0.0);
        assert_eq!(full.latency_max_ms, 0.0);
        assert_eq!(full.latency_iqr_ms, 0.0);
    }

    #[test]
    fn aggregate_layer_populates_latency_distribution_stats() {
        let cases = [metrics(0.5), metrics(0.5), metrics(0.5), metrics(0.5)];
        // Per-case median latencies: 10, 20, 30, 40 (already sorted).
        let r = aggregate_layer(&cases, &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(r.latency_min_ms, 10.0);
        assert_eq!(r.latency_max_ms, 40.0);
        // p75 = 30 (nearest-rank, ceil(0.75*4)-1 = 2 → 30)
        // p25 = 10 (ceil(0.25*4)-1 = 0 → 10) → IQR = 20.
        assert_eq!(r.latency_iqr_ms, 20.0);
    }
}
