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
use shodh_memory::recall_harness::multihop::analyze_multihop;
use shodh_memory::recall_harness::report::{
    compare_to_baseline, AblationReport, DecayReport, LearningCurveReport, MultiHopReport,
    ReachabilityReport, Report, SelectiveForgettingReport,
};
use shodh_memory::recall_harness::runner::{
    analyze_ablation, analyze_graph_reachability, analyze_learning_curve,
    run_smoke_suite_with_ranks, ReportWithRanks, RunInputs,
};

/// Exit codes — kept stable so CI scripts can branch on them.
const EXIT_PASS: i32 = 0;
const EXIT_REGRESSION: i32 = 1;
const EXIT_INFRASTRUCTURE: i32 = 2;

#[derive(Debug, Clone, Copy, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum Suite {
    /// L1 smoke suite — hand-crafted shodh queries (see issue #265). The set we
    /// tune against; this is the one CI gates on.
    Smoke,
    /// LoCoMo recall suite — 5,882 dialogue-turn corpus + 1,531 questions with
    /// gold evidence dia-ids (snap-research/locomo). The HELD-OUT set: none of
    /// the pipeline changes were diagnosed against it, so recall@k here tests
    /// generalization, not fit. Not gated — diagnostic only.
    Locomo,
}

impl Suite {
    fn as_str(self) -> &'static str {
        match self {
            Suite::Smoke => "smoke",
            Suite::Locomo => "locomo",
        }
    }

    /// Corpus + cases fixture paths, or `None` to use the runner's smoke
    /// defaults.
    fn fixture_paths(self) -> Option<(PathBuf, PathBuf)> {
        match self {
            Suite::Smoke => None,
            Suite::Locomo => Some((
                shodh_memory::recall_harness::fixtures::manifest_path(
                    shodh_memory::recall_harness::fixtures::LOCOMO_CORPUS_PATH,
                ),
                shodh_memory::recall_harness::fixtures::manifest_path(
                    shodh_memory::recall_harness::fixtures::LOCOMO_CASES_PATH,
                ),
            )),
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

    /// Optional path to write per-case diagnostics (JSON array) for the gated
    /// layer: each case's ndcg/recall/mrr plus the relevant items it dropped
    /// from the top-k. Diagnostic side output — not the gated report, not the
    /// baseline. Use it to see *which* query is weak, not just a category mean.
    #[arg(long)]
    per_case_output: Option<PathBuf>,

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

    /// Graph-reachability diagnostic: when set, skip the recall run and instead
    /// ingest the corpus and walk the built knowledge graph to report, per
    /// category, how much gold is reachable from the query's seed entities
    /// within N entity-hops (pure topology, ignoring activation weights). Writes
    /// a `ReachabilityReport` JSON to this path. Answers whether the graph-native
    /// fix cluster can lift multi_hop, or whether the gold has no associative
    /// path at all.
    #[arg(long)]
    graph_reachability: Option<PathBuf>,

    /// Learning-curve diagnostic ("smarter with use"): when set, skip the recall
    /// run and instead repeatedly recall + reinforce each tracked query, writing
    /// a `LearningCurveReport` (gold rank/score per reinforcement cycle) to this
    /// path. Measures whether memories become easier to recall as they are used.
    #[arg(long)]
    learning_curve: Option<PathBuf>,

    /// Reinforcement cycles for the learning-curve diagnostic.
    #[arg(long, default_value_t = 8)]
    lc_cycles: usize,

    /// E3 controlled multi-hop diagnostic: when set, skip the configured suite
    /// and instead generate a synthetic planted 2-hop-chain corpus (gold
    /// reachable ONLY by graph traversal), run it through every LayerMode, and
    /// write a `MultiHopReport` (per-layer 2-hop vs 1-hop recall) to this path.
    /// The `+spreading − vamana_only` 2-hop delta is the graph leg's isolated
    /// multi-hop contribution — the signal LoCoMo recall@k cannot surface.
    #[arg(long)]
    multi_hop: Option<PathBuf>,

    /// Number of planted chains for the multi-hop diagnostic.
    #[arg(long, default_value_t = shodh_memory::recall_harness::multihop::DEFAULT_CHAINS)]
    mh_chains: usize,

    /// Ablation matrix: when set, ingest the suite once and run it under a set of
    /// named query-time configs (baseline, facts-off, +graph-expand, +spread-fix,
    /// …), writing an `AblationReport` (one comparison table) to this path. The
    /// single place to see + update every component's recall contribution.
    #[arg(long)]
    ablation: Option<PathBuf>,

    /// Temporal controlled diagnostic: generate planted time-varying facts (gold =
    /// the fact valid at the queried time, not the latest), run through every
    /// LayerMode, and write per-layer recall on the valid-at-T cases vs the latest
    /// control. The +facts delta on valid-at-T isolates the temporal layer.
    #[arg(long)]
    temporal: Option<PathBuf>,

    /// E6 decay/forgetting: run the suite at increasing simulated ages and write a
    /// recall@k-vs-age stability curve (flat = stable, cliff = catastrophic
    /// forgetting).
    #[arg(long)]
    forgetting: Option<PathBuf>,

    /// E6b selective forgetting: competitive important-vs-trivial populations under
    /// one query; reports retention divergence (important − trivial) vs age. A real
    /// cognitive memory grows the divergence; indiscriminate decay keeps it ~0.
    #[arg(long)]
    selective_forgetting: Option<PathBuf>,

    /// E5 ontology: planted type-disambiguation (person vs org sharing a place);
    /// the +rerank delta on the type-qualified cases isolates the ontology layer.
    #[arg(long)]
    ontology: Option<PathBuf>,

    /// E4 causal-lineage: planted causal chains; root-cause query reachable only by
    /// chaining. Reports per-layer recall on root-cause vs direct-cause control.
    #[arg(long)]
    lineage: Option<PathBuf>,

    /// E7 fact-extraction QUALITY: plant memories with known gold facts, force
    /// distillation, and score precision/recall/F1/dedup of the extracted facts.
    /// The first harness to measure fact CORRECTNESS, not retrieval rank.
    #[arg(long)]
    facts: Option<PathBuf>,

    /// Simulated edge age in days, applied AFTER ingest and BEFORE queries
    /// (decay study). When `> 0`, the harness ages the knowledge-graph edges via
    /// `simulate_edge_aging` at the production ~6h cadence, so recall quality is
    /// measured as if the edges were `age_days` old. Default `0` = no aging
    /// (pre-existing behavior). Run at 0 / 7 / 30 / 90 and diff the reports to
    /// see how edge decay erodes recall.
    #[arg(long, default_value_t = 0.0)]
    age_days: f64,
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

    let (corpus_path, cases_path) = match args.suite.fixture_paths() {
        Some((c, q)) => (Some(c), Some(q)),
        None => (None, None),
    };
    let inputs = RunInputs {
        storage_path: storage_path.clone(),
        corpus_path,
        cases_path,
        suite: args.suite.as_str().to_string(),
        git_sha,
        repeats: args.repeats,
        layer_modes: args.layer.to_modes(),
        age_days: args.age_days,
    };

    // Graph-reachability diagnostic short-circuits the recall run entirely.
    if let Some(reach_path) = &args.graph_reachability {
        let report = analyze_graph_reachability(&inputs).context("graph-reachability analysis")?;
        write_reachability(reach_path, &report)?;
        summarise_reachability(&report);
        eprintln!(
            "recall-eval: storage retained at {} (delete manually after inspection)",
            storage_path.display()
        );
        return Ok(EXIT_PASS);
    }

    // Ablation matrix short-circuits the recall run entirely.
    if let Some(abl_path) = &args.ablation {
        let report = analyze_ablation(&inputs).context("ablation analysis")?;
        write_ablation(abl_path, &report)?;
        summarise_ablation(&report);
        eprintln!(
            "recall-eval: storage retained at {} (delete manually after inspection)",
            storage_path.display()
        );
        return Ok(EXIT_PASS);
    }

    // Temporal controlled diagnostic short-circuits the recall run entirely.
    if let Some(t_path) = &args.temporal {
        let report =
            shodh_memory::recall_harness::temporal_harness::analyze_temporal(&inputs, args.mh_chains)
                .context("temporal analysis")?;
        write_multihop(t_path, &report)?;
        eprintln!(
            "recall-eval: temporal (subjects={} validT_cases={} latest_cases={})",
            report.chains, report.multihop_cases, report.onehop_cases
        );
        println!("## Temporal controlled (planted time-varying facts)\n");
        println!("| stage | valid-at-T P@1 | Δ vs prev | control P@1 | validT recall@10 |");
        println!("| --- | --- | --- | --- | --- |");
        let mut prev: Option<f64> = None;
        for r in &report.rows {
            let d = match prev {
                Some(p) => format!("{:+.4}", r.multihop_p_at_1 - p),
                None => String::new(),
            };
            println!(
                "| {} | {:.4} | {} | {:.4} | {:.4} |",
                r.layer, r.multihop_p_at_1, d, r.onehop_p_at_1, r.multihop_recall_at_10
            );
            prev = Some(r.multihop_p_at_1);
        }
        println!("\nP@1 is the metric: gold = the state in force at the queried year, which must rank #1 above the 2 equi-lexical distractor states (recall@10 saturates at 1.0 — only 3 candidates fit the window). The +facts/+rerank delta on valid-at-T P@1 is the temporal layer's isolated contribution; baseline ≈ 1/3 chance.");
        eprintln!(
            "recall-eval: storage retained at {} (delete manually after inspection)",
            storage_path.display()
        );
        return Ok(EXIT_PASS);
    }

    // E5 ontology diagnostic.
    if let Some(o_path) = &args.ontology {
        let report = shodh_memory::recall_harness::ontology_harness::analyze_ontology(
            &inputs, args.mh_chains,
        )
        .context("ontology analysis")?;
        write_multihop(o_path, &report)?;
        print_cap_ladder(
            &report,
            "Ontology (type-disambiguation)",
            "type-qualified",
            "lexical-control",
            "+rerank delta on type-qualified P@1 = the ontology layer's isolated contribution (baseline ≈ 1/(1+K orgs)); ~0 ⇒ ontology rerank inert.",
        );
        return Ok(EXIT_PASS);
    }

    // E4 causal-lineage diagnostic.
    if let Some(l_path) = &args.lineage {
        let report = shodh_memory::recall_harness::lineage_harness::analyze_lineage(
            &inputs, args.mh_chains,
        )
        .context("lineage analysis")?;
        write_multihop(l_path, &report)?;
        print_cap_ladder(
            &report,
            "Causal lineage (root-cause chains)",
            "root-cause",
            "direct-cause control",
            "root-cause is reachable only by chaining past the lexical direct-cause distractor; if it stays ~0 across layers, causal/lineage retrieval is not exercised in eval.",
        );
        return Ok(EXIT_PASS);
    }

    // E7 fact-extraction quality.
    if let Some(facts_path) = &args.facts {
        let report = shodh_memory::recall_harness::facts_harness::analyze_facts(&inputs)
            .context("facts analysis")?;
        std::fs::write(facts_path, serde_json::to_string_pretty(&report)?)
            .context("write facts report")?;
        println!("## Fact-extraction quality (E7)\n");
        println!(
            "- gold concepts: {}  distractors: {}  facts in store: {} (cycle created: {})",
            report.gold_concepts,
            report.distractors,
            report.facts_extracted,
            report.facts_extracted_this_cycle
        );
        println!(
            "- **precision: {:.3}  recall: {:.3}  F1: {:.3}**",
            report.precision, report.recall, report.f1
        );
        println!(
            "- correct: {}  spurious: {}  dedup-ok: {}/{}  conf(correct/spurious): {:.2}/{:.2}",
            report.correct_extracted,
            report.spurious,
            report.dedup_ok,
            report.recalled_concepts,
            report.mean_confidence_correct,
            report.mean_confidence_spurious
        );
        println!("\n| fact type | gold | recalled | recall |");
        println!("| --- | --- | --- | --- |");
        for (t, row) in &report.by_type {
            println!("| {} | {} | {} | {:.3} |", t, row.gold, row.recalled, row.recall);
        }
        println!("\nprecision = correct/extracted; recall = gold concepts captured; dedup-ok = concepts represented by exactly one fact; spurious = extracted facts matching no gold (incl. distractor leakage). A weak extractor reads as low recall here — the signal the +facts retrieval layer cannot surface.");
        return Ok(EXIT_PASS);
    }

    // E6 decay/forgetting stability curve.
    if let Some(f_path) = &args.forgetting {
        let report = shodh_memory::recall_harness::forgetting_harness::analyze_forgetting(
            &inputs,
            shodh_memory::recall_harness::forgetting_harness::DEFAULT_AGES,
        )
        .context("forgetting analysis")?;
        write_decay(f_path, &report)?;
        eprintln!("recall-eval: forgetting/stability (suite={})", report.suite);
        println!("## Decay / forgetting stability curve ({})\n", report.suite);
        println!("| age (days) | recall@10 | ndcg@10 | mrr |");
        println!("| --- | --- | --- | --- |");
        for r in &report.rows {
            println!(
                "| {:.0} | {:.4} | {:.4} | {:.4} |",
                r.age_days, r.recall_at_10, r.ndcg_at_10, r.mrr
            );
        }
        println!("\nFlat = stable memory (good homeostasis); a cliff = catastrophic forgetting from edge decay.");
        return Ok(EXIT_PASS);
    }

    // E6b selective forgetting: important-vs-trivial retention divergence vs age.
    if let Some(sf_path) = &args.selective_forgetting {
        use shodh_memory::recall_harness::forgetting_harness as fh;
        let cycles = std::env::var("SHODH_SELECTIVE_REINFORCE_CYCLES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(fh::DEFAULT_REINFORCE_CYCLES);
        let report =
            fh::analyze_selective_forgetting(&inputs, fh::DEFAULT_GROUPS, fh::DEFAULT_AGES, cycles)
                .context("selective forgetting analysis")?;
        write_selective(sf_path, &report)?;
        eprintln!(
            "recall-eval: selective forgetting (suite={}, pairs={}, cycles={})",
            report.suite, report.pairs, report.reinforce_cycles
        );
        println!(
            "## Selective forgetting — important vs trivial retention ({}, {} pairs, {} reinforce cycles)\n",
            report.suite, report.pairs, report.reinforce_cycles
        );
        println!("| age (days) | important@{} | trivial@{} | divergence |", 6, 6);
        println!("| --- | --- | --- | --- |");
        for r in &report.rows {
            println!(
                "| {:.0} | {:.4} | {:.4} | {:+.4} |",
                r.age_days, r.important_retention, r.trivial_retention, r.divergence
            );
        }
        println!("\nDivergence GROWING with age = selective forgetting (retains important, drops trivial). Flat ~0 = indiscriminate decay.");
        return Ok(EXIT_PASS);
    }

    // E3 multi-hop diagnostic short-circuits the recall run entirely.
    if let Some(mh_path) = &args.multi_hop {
        let report = analyze_multihop(&inputs, args.mh_chains).context("multi-hop analysis")?;
        write_multihop(mh_path, &report)?;
        summarise_multihop(&report);
        eprintln!(
            "recall-eval: storage retained at {} (delete manually after inspection)",
            storage_path.display()
        );
        return Ok(EXIT_PASS);
    }

    // Learning-curve diagnostic short-circuits the recall run entirely.
    if let Some(lc_path) = &args.learning_curve {
        let report =
            analyze_learning_curve(&inputs, args.lc_cycles).context("learning-curve analysis")?;
        write_learning_curve(lc_path, &report)?;
        summarise_learning_curve(&report);
        eprintln!(
            "recall-eval: storage retained at {} (delete manually after inspection)",
            storage_path.display()
        );
        return Ok(EXIT_PASS);
    }

    let ReportWithRanks {
        mut report,
        per_case_by_layer,
        ..
    } = run_smoke_suite_with_ranks(&inputs).context("running smoke suite")?;

    // Per-case diagnostics are written before the baseline comparison so they
    // are always captured, even on a regressing run that exits non-zero. The
    // payload is keyed by layer (`"full"`, … `--layer all` gives every stage)
    // so a missed item can be traced to the stage that dropped it.
    if let Some(per_case_path) = &args.per_case_output {
        let json = serde_json::to_vec_pretty(&per_case_by_layer)
            .context("serialising per-case diagnostics to JSON")?;
        std::fs::write(per_case_path, &json).with_context(|| {
            format!(
                "writing per-case diagnostics to {}",
                per_case_path.display()
            )
        })?;
        eprintln!(
            "recall-eval: per-case diagnostics written to {}",
            per_case_path.display()
        );
    }

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

fn write_reachability(path: &std::path::Path, report: &ReachabilityReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let json = serde_json::to_vec_pretty(report).context("serialising reachability report")?;
    std::fs::write(path, &json)
        .with_context(|| format!("writing reachability report to {}", path.display()))?;
    Ok(())
}

fn summarise_reachability(report: &ReachabilityReport) {
    let pct = |n: usize, d: usize| if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 };
    eprintln!(
        "recall-eval: graph reachability (suite={} sha={} max_hops={})",
        report.suite, report.git_sha, report.max_hops
    );
    eprintln!(
        "  {:<12} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "category", "cases", "gold", "≤1hop%", "≤2hop%", "≤3hop%", "unreach%", "no-seed%"
    );
    let overall_label = report_overall_label();
    let mut rows: Vec<(&String, &shodh_memory::recall_harness::report::ReachabilityCategory)> =
        report.by_category.iter().collect();
    rows.push((&overall_label, &report.overall));
    for (name, c) in rows {
        eprintln!(
            "  {:<12} {:>6} {:>9} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>9.1}",
            name,
            c.cases,
            c.gold_total,
            pct(c.reachable_within_1, c.gold_total),
            pct(c.reachable_within_2, c.gold_total),
            pct(c.reachable_within_3, c.gold_total),
            pct(c.unreachable, c.gold_total),
            pct(c.cases_no_seed, c.cases),
        );
    }
    eprintln!("  ≤2hop% is the canonical double-hop signal: high => graph-native fixes can lift multi_hop;");
    eprintln!("  low + high unreach% => gold has no associative path (extraction/construction gap or non-entity hop).");
}

/// Label used for the aggregate reachability row. A function (not a const) so it
/// owns its `String` and slots into the same `(&String, _)` row vector.
fn report_overall_label() -> String {
    "ALL".to_string()
}

fn write_learning_curve(path: &std::path::Path, report: &LearningCurveReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let json = serde_json::to_vec_pretty(report).context("serialising learning-curve report")?;
    std::fs::write(path, &json)
        .with_context(|| format!("writing learning-curve report to {}", path.display()))?;
    Ok(())
}

fn summarise_learning_curve(report: &LearningCurveReport) {
    eprintln!(
        "recall-eval: learning curve (suite={} sha={} cycles={}) — reward gradient",
        report.suite, report.git_sha, report.cycles
    );
    for arm in &report.arms {
        let (first, last) = (
            arm.mean_rank_by_cycle.first().copied().unwrap_or(0.0),
            arm.mean_rank_by_cycle.last().copied().unwrap_or(0.0),
        );
        let (sfirst, slast) = (
            arm.mean_score_by_cycle.first().copied().unwrap_or(0.0),
            arm.mean_score_by_cycle.last().copied().unwrap_or(0.0),
        );
        eprintln!(
            "  [{:<10}] tracked={:<3} rank {:.2}->{:.2} (Δ{:+.3}) score {:.4}->{:.4} (Δ{:+.4})  imp/wor/unc={}/{}/{}",
            arm.outcome,
            arm.tracked_cases,
            first,
            last,
            arm.mean_rank_delta,
            sfirst,
            slast,
            arm.mean_score_delta,
            arm.improved,
            arm.worsened,
            arm.unchanged
        );
    }
    eprintln!("  Helpful rank DOWN, Misleading rank UP => reward steers recall (a real gradient).");
}

fn write_ablation(path: &std::path::Path, report: &AblationReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let json = serde_json::to_vec_pretty(report).context("serialising ablation report")?;
    std::fs::write(path, &json)
        .with_context(|| format!("writing ablation report to {}", path.display()))?;
    Ok(())
}

/// Print the ablation matrix as a markdown table (Δ recall@10 vs the row named
/// `baseline`), so it can be pasted straight into the study doc.
fn summarise_ablation(report: &AblationReport) {
    let base = report
        .rows
        .iter()
        .find(|r| r.name.starts_with("baseline"))
        .map(|r| r.recall_at_10);
    eprintln!(
        "recall-eval: ablation (suite={} cases={} sha={})",
        report.suite, report.case_count, report.git_sha
    );
    println!("## Ablation matrix ({} suite, {} cases)\n", report.suite, report.case_count);
    println!("| config | recall@10 | Δ vs base | ndcg@10 | mrr | p@1 |");
    println!("| --- | --- | --- | --- | --- | --- |");
    for r in &report.rows {
        let delta = match base {
            Some(b) => format!("{:+.4}", r.recall_at_10 - b),
            None => String::new(),
        };
        println!(
            "| {} | {:.4} | {} | {:.4} | {:.4} | {:.4} |",
            r.name, r.recall_at_10, delta, r.ndcg_at_10, r.mrr, r.p_at_1
        );
    }
    // Per-category recall, transposed (category × config), surfaces a config that
    // trades one capability for another.
    let cats: std::collections::BTreeSet<String> = report
        .rows
        .iter()
        .flat_map(|r| r.by_category_recall.keys().cloned())
        .collect();
    if !cats.is_empty() {
        println!("\n### Per-category recall@10\n");
        let header: Vec<String> = report.rows.iter().map(|r| r.name.clone()).collect();
        println!("| category | {} |", header.join(" | "));
        println!("| --- | {} |", vec!["---"; header.len()].join(" | "));
        for cat in &cats {
            let cells: Vec<String> = report
                .rows
                .iter()
                .map(|r| {
                    r.by_category_recall
                        .get(cat)
                        .map(|v| format!("{v:.3}"))
                        .unwrap_or_else(|| "—".to_string())
                })
                .collect();
            println!("| {} | {} |", cat, cells.join(" | "));
        }
    }
}

/// Shared per-layer capability ladder printer (E4/E5 reuse the MultiHopReport
/// shape: `multihop_*` = the capability column, `onehop_*` = the control column).
fn print_cap_ladder(
    report: &MultiHopReport,
    title: &str,
    cap_col: &str,
    ctrl_col: &str,
    note: &str,
) {
    eprintln!(
        "recall-eval: {title} (cases={} control={})",
        report.multihop_cases, report.onehop_cases
    );
    println!("## {title}\n");
    println!("| stage | {cap_col} P@1 | Δ vs prev | {ctrl_col} P@1 | {cap_col} recall@10 |");
    println!("| --- | --- | --- | --- | --- |");
    let mut prev: Option<f64> = None;
    for r in &report.rows {
        let d = match prev {
            Some(p) => format!("{:+.4}", r.multihop_p_at_1 - p),
            None => String::new(),
        };
        println!(
            "| {} | {:.4} | {} | {:.4} | {:.4} |",
            r.layer, r.multihop_p_at_1, d, r.onehop_p_at_1, r.multihop_recall_at_10
        );
        prev = Some(r.multihop_p_at_1);
    }
    println!("\nP@1 is the metric: the capability-correct item must rank #1 above its equi-confusable distractors (recall@10 saturates because the whole confusable set fits the top-10 window). {note}");
}

fn write_decay(path: &std::path::Path, report: &DecayReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let json = serde_json::to_vec_pretty(report).context("serialising decay report")?;
    std::fs::write(path, &json)
        .with_context(|| format!("writing decay report to {}", path.display()))?;
    Ok(())
}

fn write_selective(path: &std::path::Path, report: &SelectiveForgettingReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let json =
        serde_json::to_vec_pretty(report).context("serialising selective-forgetting report")?;
    std::fs::write(path, &json)
        .with_context(|| format!("writing selective-forgetting report to {}", path.display()))?;
    Ok(())
}

fn write_multihop(path: &std::path::Path, report: &MultiHopReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let json = serde_json::to_vec_pretty(report).context("serialising multi-hop report")?;
    std::fs::write(path, &json)
        .with_context(|| format!("writing multi-hop report to {}", path.display()))?;
    Ok(())
}

fn summarise_multihop(report: &MultiHopReport) {
    eprintln!(
        "recall-eval: multi-hop (chains={} 2hop_cases={} 1hop_cases={}) — graph-only-reachable gold",
        report.chains, report.multihop_cases, report.onehop_cases
    );
    eprintln!("  layer         2hop_recall@10  1hop_recall@10  2hop_mrr");
    let mut prev_2hop: Option<f64> = None;
    for row in &report.rows {
        let delta = match prev_2hop {
            Some(p) => format!("(Δ{:+.4})", row.multihop_recall_at_10 - p),
            None => String::new(),
        };
        eprintln!(
            "  {:<12}  {:.4} {:<10}  {:.4}          {:.4}",
            row.layer, row.multihop_recall_at_10, delta, row.onehop_recall_at_10, row.multihop_mrr
        );
        prev_2hop = Some(row.multihop_recall_at_10);
    }
    eprintln!("  +spreading 2hop delta = the GRAPH leg's isolated multi-hop recall; if large while +bm25");
    eprintln!("  adds little to 2hop (but lifts 1hop), the graph carries the multi-hop load as designed.");
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
