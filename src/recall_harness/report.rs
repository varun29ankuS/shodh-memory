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
    /// Cases or metrics that failed validation or regression.
    pub failures: Vec<Failure>,
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
}
