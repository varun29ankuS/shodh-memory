//! `recall-eval` — run the L1 smoke suite and emit a machine-diffable report.
//!
//! Used by:
//! - **RH-5 CI gate** to fail PRs that regress recall by more than the
//!   tolerance against the checked-in baseline.
//! - **RH-6 baseline capture** to record the current production numbers on
//!   `main`.
//! - **Humans** comparing embedder swaps, scoring tweaks, and pipeline
//!   refactors via JSON diff.
//!
//! See issue #266 for the full schema and acceptance criteria. RH-8 will
//! widen this binary with `--layer` once per-pipeline-layer attribution is
//! plumbed through `MemorySystem`.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use shodh_memory::memory::types::LayerMode;
use shodh_memory::recall_harness::report::{compare_to_baseline, Report};
use shodh_memory::recall_harness::runner::{run_smoke_suite, RunInputs};

/// Exit codes — kept stable so CI scripts can branch on them.
const EXIT_PASS: i32 = 0;
const EXIT_REGRESSION: i32 = 1;
const EXIT_INFRASTRUCTURE: i32 = 2;

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum Suite {
    /// L1 smoke suite — 30 hand-crafted shodh queries (see issue #265).
    Smoke,
}

impl Suite {
    fn as_str(self) -> &'static str {
        match self {
            Suite::Smoke => "smoke",
        }
    }
}

/// Per-pipeline-layer attribution selector. RH-8 (#270).
///
/// `all` runs the full ladder of cumulative modes (vamana-only → full),
/// emitting one entry per mode in the report's `layers` map. CI gating
/// remains keyed on `full` so per-layer numbers are diagnostic, not gated.
///
/// Naming note: the `+rerank` mode in the spec is a misnomer for this
/// codebase — there is no cross-encoder. The gate covers the ontological
/// re-ranker at Layer 4.9 instead. The label is preserved for spec fidelity.
#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum LayerArg {
    /// All six modes, ascending. Use for diagnostic per-layer attribution.
    All,
    /// Layer 3 vector ANN only.
    VamanaOnly,
    /// + Layer 2 graph spreading activation.
    PlusSpreading,
    /// + Layer 4 BM25/RRF fusion.
    PlusBm25,
    /// + Layer 4.9 ontological rerank (spec calls this `+rerank`).
    PlusRerank,
    /// + Layer 0.7/4.8 fact-source boost.
    PlusFacts,
    /// Production pipeline — every stage on. CI gates on this mode only.
    Full,
}

impl LayerArg {
    fn to_modes(self) -> Vec<LayerMode> {
        match self {
            LayerArg::All => LayerMode::ALL.to_vec(),
            LayerArg::VamanaOnly => vec![LayerMode::VamanaOnly],
            LayerArg::PlusSpreading => vec![LayerMode::PlusSpreading],
            LayerArg::PlusBm25 => vec![LayerMode::PlusBm25],
            LayerArg::PlusRerank => vec![LayerMode::PlusRerank],
            LayerArg::PlusFacts => vec![LayerMode::PlusFacts],
            LayerArg::Full => vec![LayerMode::Full],
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "recall-eval",
    about = "Run the recall harness and emit a JSON report."
)]
struct Args {
    /// Suite to run. Only `smoke` is implemented today; `regression` and
    /// `deep` ship with RH-7.
    #[arg(long, value_enum, default_value_t = Suite::Smoke)]
    suite: Suite,

    /// Path to write the JSON report to.
    #[arg(long)]
    output: PathBuf,

    /// Optional baseline report to compare against. When provided, a
    /// regression in any gating metric beyond `--tolerance` exits with
    /// status `1`.
    #[arg(long)]
    baseline: Option<PathBuf>,

    /// Allowed regression in percent of baseline. Default: 2.0%.
    #[arg(long, default_value_t = 2.0)]
    tolerance: f64,

    /// Optional storage directory for the harness's `MemorySystem`. When
    /// omitted, a unique directory under the system temp dir is used and
    /// kept on disk so a failed run can be inspected.
    #[arg(long)]
    storage: Option<PathBuf>,

    /// Number of independent ingest+query repeats. RH-12 (#272). Each
    /// repeat stands up its own `MemorySystem` against a fresh storage
    /// subdirectory, ingests the corpus, and runs every case. Per-case
    /// latency in the report is the median across repeats; quality
    /// metrics (ndcg/recall/mrr/p@1/map) MUST be byte-identical across
    /// repeats — any divergence fails the run with `EXIT_INFRASTRUCTURE`.
    /// Default `5` is the smallest N that survives a single outlier and
    /// still produces a true median for IQR calculation.
    #[arg(long, default_value_t = 5)]
    repeats: usize,

    /// Per-pipeline-layer attribution mode. RH-8 (#270). `full` (default)
    /// runs only the production pipeline and reproduces the pre-RH-8
    /// behavior bit-for-bit. `all` runs every mode and emits one entry
    /// per mode in the report's `layers` map for diagnostic attribution.
    /// CI keys on `full` only — per-layer numbers are not gated.
    #[arg(long, value_enum, default_value_t = LayerArg::Full)]
    layer: LayerArg,
}

fn main() {
    let args = Args::parse();
    let exit = match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("recall-eval: infrastructure failure: {e:#}");
            EXIT_INFRASTRUCTURE
        }
    };
    std::process::exit(exit);
}

fn run(args: &Args) -> Result<i32> {
    let storage_path = args
        .storage
        .clone()
        .unwrap_or_else(|| default_storage_dir(args.suite.as_str()));

    let git_sha = current_git_sha().unwrap_or_else(|_| "unknown".to_string());

    let inputs = RunInputs {
        storage_path: storage_path.clone(),
        corpus_path: None,
        cases_path: None,
        suite: args.suite.as_str().to_string(),
        git_sha,
        repeats: args.repeats,
        layer_modes: args.layer.to_modes(),
    };

    let mut report = run_smoke_suite(&inputs).context("running smoke suite")?;

    if let Some(baseline_path) = &args.baseline {
        let baseline_bytes = std::fs::read(baseline_path)
            .with_context(|| format!("reading baseline {}", baseline_path.display()))?;
        let baseline: Report = serde_json::from_slice(&baseline_bytes)
            .with_context(|| format!("parsing baseline {}", baseline_path.display()))?;
        let mut regressions = compare_to_baseline(&baseline, &report, args.tolerance);
        report.failures.append(&mut regressions);
    }

    write_report(&args.output, &report)?;

    eprintln!(
        "recall-eval: storage retained at {} (delete manually after inspection)",
        storage_path.display()
    );

    let regression_count = report
        .failures
        .iter()
        .filter(|f| f.kind == "regression")
        .count();
    let infra_count = report
        .failures
        .iter()
        .filter(|f| f.kind == "infrastructure")
        .count();

    summarise(&report);

    if infra_count > 0 {
        Ok(EXIT_INFRASTRUCTURE)
    } else if regression_count > 0 {
        Ok(EXIT_REGRESSION)
    } else {
        Ok(EXIT_PASS)
    }
}

fn write_report(path: &std::path::Path, report: &Report) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let json = serde_json::to_vec_pretty(report).context("serialising report to JSON")?;
    std::fs::write(path, &json).with_context(|| format!("writing report to {}", path.display()))?;
    Ok(())
}

fn summarise(report: &Report) {
    eprintln!(
        "recall-eval: suite={} cases={} repeats={} embedder={} sha={}",
        report.suite, report.case_count, report.repeats, report.embedder, report.git_sha
    );
    // Print modes in pipeline order (vamana_only → full), not BTreeMap order.
    // Match against the canonical mode keys so any unknown key just falls
    // through to the BTreeMap iteration at the bottom.
    let mode_order = [
        "vamana_only",
        "+spreading",
        "+bm25",
        "+rerank",
        "+facts",
        "full",
    ];
    for name in mode_order {
        if let Some(layer) = report.layers.get(name) {
            eprintln!(
                "  {:<12} ndcg@10={:.4} recall@10={:.4} mrr={:.4} p@1={:.4} map={:.4}",
                name, layer.ndcg_at_10, layer.recall_at_10, layer.mrr, layer.p_at_1, layer.map
            );
        }
    }
    if let Some(layer) = report.layers.get("full") {
        eprintln!(
            "  latency p50={:.1}ms p95={:.1}ms p99={:.1}ms (per-case median, full mode)",
            layer.latency_p50_ms, layer.latency_p95_ms, layer.latency_p99_ms
        );
        eprintln!(
            "  latency min={:.1}ms max={:.1}ms iqr={:.1}ms",
            layer.latency_min_ms, layer.latency_max_ms, layer.latency_iqr_ms
        );
    }
    for (name, cat) in &report.by_category {
        eprintln!(
            "  {:<10} ndcg@10={:.4} recall@10={:.4} mrr={:.4}",
            name, cat.ndcg_at_10, cat.recall_at_10, cat.mrr
        );
    }
    if !report.failures.is_empty() {
        eprintln!("  failures ({}):", report.failures.len());
        for f in &report.failures {
            eprintln!("    [{}] {}", f.kind, f.detail);
        }
    }
}

/// Resolve the current git SHA by shelling out. Returns an error rather
/// than panicking so the binary can still produce a report when run from a
/// non-git checkout (e.g. a release tarball).
fn current_git_sha() -> Result<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .context("running `git rev-parse HEAD`")?;
    if !out.status.success() {
        anyhow::bail!(
            "`git rev-parse HEAD` failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Build a unique storage path under the system temp dir. Caller is
/// responsible for cleanup; we deliberately do not delete on success so
/// CI logs can preserve the storage state for debugging.
fn default_storage_dir(suite: &str) -> PathBuf {
    let id = uuid::Uuid::new_v4().simple().to_string();
    std::env::temp_dir().join(format!("shodh-recall-eval-{suite}-{id}"))
}
