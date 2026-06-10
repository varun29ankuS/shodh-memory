//! End-to-end runner for the L1 smoke suite.
//!
//! Loads fixtures via [`crate::recall_harness::fixtures`], stands up a
//! `MemorySystem` against an isolated storage directory, ingests the corpus,
//! runs every smoke case through `MemorySystem::recall`, and emits a
//! [`Report`] aggregating per-case [`Metrics`] into per-layer and
//! per-category sections.
//!
//! See issue #266 for the runner's CLI and acceptance criteria.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use uuid::Uuid;

use crate::config::ServerConfig;
use crate::handlers::MultiUserMemoryManager;
use crate::memory::types::{ExperienceType, LayerMode, NerEntityRecord};
use crate::memory::retrieval::RetrievalOutcome;
use crate::memory::types::MemoryId;
use crate::memory::{Experience, Query};

use super::fixtures::{
    self, CorpusItem, SmokeCase, SmokeCategory, SMOKE_CASES_PATH, SMOKE_CORPUS_PATH,
};
use super::metrics::Metrics;
use super::report::{
    aggregate_category, aggregate_layer, median, AblationReport, AblationRow, CategoryReport,
    Failure, LayerReport, FunnelReport, FunnelStageRow, GraphStructure, LearningCurveArm,
    LearningCurveReport, LinkingReport, LinkingRow, PerCaseRecord, ReachabilityCategory,
    ReachabilityReport, Report, SMOKE_K,
};

/// Embedder identifier emitted in the report. Matches the model wired into
/// `MemorySystem::new` today; bump when the embedder changes.
pub const EMBEDDER_ID: &str = "minilm-l6-v2";

/// Inputs to [`run_smoke_suite`].
#[derive(Debug, Clone)]
pub struct RunInputs {
    /// Storage directory for the harness's `MemorySystem`. The directory
    /// must be writeable; the runner does not delete it on completion so
    /// callers can inspect state after a failed run. With `repeats > 1`
    /// each repeat gets its own subdirectory `<storage_path>/repeat_<i>/`
    /// so storage state is independent across repeats — that is what gives
    /// the median across repeats meaning as a measure of cross-process
    /// noise, not just within-process noise.
    pub storage_path: PathBuf,
    /// Optional path overrides; if `None` the canonical fixture paths are used.
    pub corpus_path: Option<PathBuf>,
    pub cases_path: Option<PathBuf>,
    /// Suite identifier echoed into the report. Currently fixed to `"smoke"`.
    pub suite: String,
    /// Git SHA of the working tree the run was produced from. Caller is
    /// responsible for resolution because the runner does not shell out.
    pub git_sha: String,
    /// Number of independent ingest+query repeats. RH-12 (#272). Each repeat
    /// stands up its own isolated `MemorySystem`, ingests the corpus, and
    /// runs all cases. Per-case latency is the median across repeats; quality
    /// metrics must be byte-identical across repeats (any divergence is an
    /// infrastructure failure that fails the run).
    ///
    /// Default in the CLI is `5`. Setting `0` is treated as `1`.
    pub repeats: usize,

    /// Per-pipeline-layer attribution modes to run. RH-8 (#270). Each mode
    /// is a cumulative subset of the production retrieval pipeline; running
    /// the same query set under multiple modes lets us attribute quality
    /// deltas to specific stages rather than treating the pipeline as a
    /// single number. Within one repeat, ingest runs once and the case loop
    /// runs once per mode (cost ≈ ingest + N_modes × queries).
    ///
    /// CI gating still keys on `LayerMode::Full` only; the other modes are
    /// diagnostic. A single-element vec containing `Full` reproduces the
    /// pre-RH-8 behavior bit-for-bit.
    ///
    /// An empty vec is treated as `[LayerMode::Full]` so callers that don't
    /// care about per-layer attribution don't have to populate this.
    pub layer_modes: Vec<LayerMode>,

    /// Simulated edge age in days, applied AFTER ingest and BEFORE the query
    /// passes (RH decay study). When `> 0`, the harness ages the
    /// knowledge-graph edges via `MemorySystem::simulate_edge_aging` at the
    /// production ~6h cadence, so recall quality is measured as if the edges
    /// were `age_days` old. `0.0` (the default) means no aging — the
    /// pre-existing behavior, bit-for-bit. Run at 0 / 7 / 30 / 90 and diff the
    /// reports to see how edge decay erodes recall.
    pub age_days: f64,
}

/// One case's retrieved rank list, in score-descending order.
///
/// Used by the determinism gate (RH-11) to assert byte-identical
/// rank lists across consecutive runs of the same suite.
///
/// `retrieved` holds **corpus item IDs** (stable string handles from the
/// fixture), not the random UUIDs assigned by `MemorySystem::remember`.
/// UUIDs are freshly generated per ingest and therefore differ across
/// runs even when the rank order is byte-identical; comparing them would
/// guarantee a false negative. Items that were retrieved but do not
/// belong to the corpus (e.g. system-injected memories) are emitted as
/// `"<unknown:...>"` sentinels so divergence in those slots is still
/// observable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseRankList {
    pub case_id: String,
    pub retrieved: Vec<String>,
}

/// Wrapper around [`Report`] plus the per-case rank lists. Returned by
/// [`run_smoke_suite_with_ranks`]; the ranks are stripped by the public
/// [`run_smoke_suite`] wrapper to preserve the simpler caller contract.
#[derive(Debug, Clone)]
pub struct ReportWithRanks {
    pub report: Report,
    pub ranks: Vec<CaseRankList>,
    /// Per-case diagnostics keyed by layer `report_key` (e.g. `"full"`,
    /// `"vamana-only"`), each aligned with the fixture case order. With
    /// `--layer all` this carries every mode, so a missed item can be traced
    /// to the stage that dropped it (e.g. present in `vamana-only`, gone in
    /// `full` ⇒ a later layer demoted it). Side output only — never folded
    /// into `report`.
    pub per_case_by_layer: BTreeMap<String, Vec<PerCaseRecord>>,
}

/// Run the smoke suite end-to-end and return a populated [`Report`].
///
/// On infrastructure errors (corpus parse failure, system init failure,
/// remember failure) the call returns `Err`. Per-case recall errors are
/// recorded as `Failure { kind = "case", .. }` entries so a single broken
/// query does not abort the whole run.
pub fn run_smoke_suite(inputs: &RunInputs) -> Result<Report> {
    run_smoke_suite_with_ranks(inputs).map(|rwr| rwr.report)
}

/// Same as [`run_smoke_suite`] but also returns the per-case rank lists.
///
/// Exists for the RH-11 determinism gate: two consecutive calls with
/// identical inputs must return byte-identical [`CaseRankList`]s. If they
/// don't, there is a non-deterministic source somewhere in the retrieval
/// pipeline that the tie-break + thread pinning failed to cover.
pub fn run_smoke_suite_with_ranks(inputs: &RunInputs) -> Result<ReportWithRanks> {
    // ------------------------------------------------------------------
    // Determinism: pin every parallel runtime to a single thread before
    // any work touches ONNX or rayon. Multi-threaded float reductions
    // accumulate in non-deterministic order, which flips ranks on the
    // recall harness even when no source code has changed.
    //
    // - SHODH_ONNX_THREADS=1 → MiniLM/NER intra-op runs single-threaded
    //   (already plumbed through src/embeddings/{minilm,ner}.rs).
    // - RAYON_NUM_THREADS=1  → any par_iter() in scoring runs serially.
    //
    // We only set these vars if the caller hasn't pinned them already,
    // so production callers (which never invoke the harness) stay
    // unaffected.
    // ------------------------------------------------------------------
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));

    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading smoke corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading smoke cases from {}", cases_path.display()))?;

    // Only the smoke suite is balance-gated (fixed 108 / even categories). Other
    // suites (e.g. LoCoMo) just need structural integrity: unique ids, resolvable
    // evidence. Picking by suite lets a second corpus reuse this runner.
    if inputs.suite == "smoke" {
        fixtures::validate_smoke_suite(&corpus, &cases)
            .context("smoke suite failed structural validation — fix the JSONL fixtures")?;
    } else {
        fixtures::validate_structure(&corpus, &cases)
            .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;
    }

    // Optional deterministic case subsample for fast directional reads
    // (SHODH_MAX_CASES=N). The CORPUS is always ingested in full so every gold
    // item still resolves; only the QUERY set is strided down to ~N cases evenly
    // across the file ordering (which groups by conversation/category), so the
    // category mix is preserved. Unset / 0 / >= len → no subsampling. This is a
    // diagnostic-speed knob, never used by the gated CI baseline.
    let cases = match std::env::var("SHODH_MAX_CASES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n < cases.len())
    {
        Some(n) => {
            let total = cases.len();
            let stride = (total / n).max(1);
            let sampled: Vec<_> = cases.into_iter().step_by(stride).take(n).collect();
            tracing::info!(
                "SHODH_MAX_CASES={n}: subsampled {} of {} cases (stride {})",
                sampled.len(),
                total,
                stride
            );
            sampled
        }
        None => cases,
    };

    // Optional deterministic CORPUS subsample for MAX-speed directional reads
    // (SHODH_MAX_CORPUS=N). Unlike SHODH_MAX_CASES — which keeps the full corpus
    // so each query still faces all distractors and recall stays comparable to
    // full scale — this SHRINKS the distractor pool. Fewer distractors inflate
    // recall and compress the gaps between candidates, so it is a directional
    // speed knob ONLY (never the gated baseline); a winner picked here must be
    // confirmed at full corpus scale. Safety invariant: every gold item
    // referenced by the (already-subsampled) surviving cases is ALWAYS retained,
    // so no case is silently zeroed — only non-gold distractors are strided down
    // to fill the remaining budget. Original corpus order is preserved so
    // conversation grouping (which drives graph co-occurrence edges) is intact.
    let corpus = match std::env::var("SHODH_MAX_CORPUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n < corpus.len())
    {
        Some(n) => {
            let gold: std::collections::HashSet<&str> = cases
                .iter()
                .flat_map(|c| c.relevant.iter().map(|r| r.corpus_item_id.as_str()))
                .collect();
            let total = corpus.len();
            let gold_count = corpus.iter().filter(|it| gold.contains(it.id.as_str())).count();
            let keep_distractors = n.saturating_sub(gold_count);
            let distractor_total = total - gold_count;
            let stride = if keep_distractors == 0 {
                usize::MAX
            } else {
                (distractor_total / keep_distractors).max(1)
            };
            let mut seen_distractor = 0usize;
            let mut kept_distractors = 0usize;
            let reduced: Vec<_> = corpus
                .into_iter()
                .filter(|it| {
                    if gold.contains(it.id.as_str()) {
                        return true;
                    }
                    let take = kept_distractors < keep_distractors
                        && seen_distractor % stride == 0;
                    seen_distractor += 1;
                    if take {
                        kept_distractors += 1;
                    }
                    take
                })
                .collect();
            tracing::info!(
                "SHODH_MAX_CORPUS={n}: reduced corpus {} → {} ({} gold kept + {} distractors). \
                 Directional only — recall inflates vs full scale.",
                total,
                reduced.len(),
                gold_count,
                kept_distractors
            );
            reduced
        }
        None => corpus,
    };

    let repeats = inputs.repeats.max(1);
    // RH-8 (#270): default to `[Full]` when caller passed an empty vec so
    // the existing single-mode contract is preserved by construction.
    let layer_modes: Vec<LayerMode> = if inputs.layer_modes.is_empty() {
        vec![LayerMode::Full]
    } else {
        let mut m = inputs.layer_modes.clone();
        m.sort();
        m.dedup();
        m
    };

    // Run N independent ingest+query passes against fresh storage
    // directories. RH-12 (#272): per-case latency is the median across
    // these passes; rank lists are required to be byte-identical across
    // every pass. The for-loop is intentionally serial — running passes
    // in parallel would defeat the determinism check, since concurrent
    // RocksDB instances under the same root would race on file handles.
    //
    // RH-8: within each pass, ingest happens once and the case loop runs
    // once per `LayerMode`. Determinism check is per-mode.
    let mut passes: Vec<OnePassResult> = Vec::with_capacity(repeats);
    for i in 0..repeats {
        let pass_storage = if repeats == 1 {
            // Preserve the historical layout for single-repeat runs so
            // existing CI artifacts and the integration test stay valid.
            inputs.storage_path.clone()
        } else {
            inputs.storage_path.join(format!("repeat_{i}"))
        };
        let pass = run_one_pass(
            &pass_storage,
            &corpus,
            &cases,
            &layer_modes,
            inputs.age_days,
        )
        .with_context(|| format!("repeat {i} of {repeats}"))?;
        passes.push(pass);
    }

    // ------------------------------------------------------------------
    // Determinism check across repeats. Two passes against the same
    // fixtures, with thread pinning + the RH-10/11 tie-break in place,
    // MUST produce byte-identical rank lists per mode. If they don't,
    // something upstream regressed. Surface every diverging case as an
    // `infrastructure` failure so reviewers see the full picture.
    // ------------------------------------------------------------------
    let mut failures: Vec<Failure> = passes[0].failures.clone();
    if repeats > 1 {
        for mode in &layer_modes {
            let ref_pass = passes[0]
                .per_mode
                .get(mode)
                .expect("repeat 0 must have all modes");
            for (i, pass) in passes.iter().enumerate().skip(1) {
                let cur_pass = pass
                    .per_mode
                    .get(mode)
                    .expect("every repeat must have every mode");
                for (k, ref_rank) in ref_pass.ranks.iter().enumerate() {
                    let cur_rank = &cur_pass.ranks[k];
                    debug_assert_eq!(ref_rank.case_id, cur_rank.case_id);
                    if ref_rank.retrieved != cur_rank.retrieved {
                        failures.push(Failure {
                            kind: "infrastructure".to_string(),
                            detail: format!(
                                "non-determinism [mode={}]: case {} rank list diverged between repeat 0 and repeat {i} \
                                 — repeat 0 = {:?}, repeat {i} = {:?}",
                                mode.report_key(), ref_rank.case_id, ref_rank.retrieved, cur_rank.retrieved
                            ),
                        });
                    }
                }
            }
        }
    }

    // Quality metrics: per-mode aggregation. Take per-case metrics from
    // repeat 0; per-case latency is the median across repeats.
    let mut layers: BTreeMap<String, LayerReport> = BTreeMap::new();
    for mode in &layer_modes {
        let per_case = &passes[0]
            .per_mode
            .get(mode)
            .expect("repeat 0 must have mode")
            .per_case;
        let mut latencies_median_ms: Vec<f64> = Vec::with_capacity(cases.len());
        for k in 0..cases.len() {
            let samples: Vec<f64> = passes
                .iter()
                .map(|p| {
                    p.per_mode
                        .get(mode)
                        .expect("every repeat must have every mode")
                        .latencies_ms[k]
                })
                .collect();
            latencies_median_ms.push(median(&samples));
        }
        layers.insert(
            mode.report_key().to_string(),
            aggregate_layer(per_case, &latencies_median_ms),
        );
    }

    // Per-category breakdown is reported for the highest mode present
    // (typically `Full`). Lower modes' per-category numbers are encoded
    // implicitly via the per-layer ndcg/recall tables; surfacing six
    // category-mode crosstabs would explode the report without adding
    // signal at the current corpus size.
    let highest_mode = layer_modes.last().copied().unwrap_or(LayerMode::Full);
    let by_category_cases = &passes[0]
        .per_mode
        .get(&highest_mode)
        .expect("repeat 0 must have highest mode")
        .by_category_cases;
    // Report whatever categories the suite actually contains (the smoke suite
    // has its six; LoCoMo has single_hop/open_domain/multi_hop/temporal), so a
    // second suite does not need its categories hard-coded into `ALL`.
    let mut by_category: BTreeMap<String, CategoryReport> = BTreeMap::new();
    for (cat, cases_for_cat) in by_category_cases {
        by_category.insert(
            category_name(*cat).to_string(),
            aggregate_category(cases_for_cat),
        );
    }

    let report = Report {
        suite: inputs.suite.clone(),
        embedder: EMBEDDER_ID.to_string(),
        git_sha: inputs.git_sha.clone(),
        timestamp: chrono::Utc::now(),
        layers,
        by_category,
        case_count: cases.len(),
        repeats,
        failures,
    };

    // Per-case diagnostics for every mode, from the same repeat-0 pass the
    // aggregates use and aligned with `cases` by index. With `--layer all`
    // this lets a missed item be traced to the stage that dropped it.
    let per_case_by_layer: BTreeMap<String, Vec<PerCaseRecord>> = layer_modes
        .iter()
        .map(|mode| {
            let mp = passes[0]
                .per_mode
                .get(mode)
                .expect("repeat 0 must have mode");
            (
                mode.report_key().to_string(),
                build_per_case_records(&cases, &mp.per_case, &mp.ranks),
            )
        })
        .collect();

    // The public ranks output is the repeat-0 rank list for the highest
    // mode — that matches existing RH-11 test expectations (which assume
    // a single mode). Cross-repeat divergence is already surfaced via
    // `failures` above on a per-mode basis.
    let ranks = passes
        .into_iter()
        .next()
        .expect("repeats >= 1")
        .per_mode
        .remove(&highest_mode)
        .expect("repeat 0 must have highest mode")
        .ranks;

    Ok(ReportWithRanks {
        report,
        ranks,
        per_case_by_layer,
    })
}

/// Build per-case diagnostics from one mode's aligned outputs.
///
/// `metrics[i]`, `ranks[i]`, and `cases[i]` must describe the same query (the
/// runner guarantees this by pushing all three in lockstep over `cases`).
/// `missed` is the set of relevant `corpus_item_id`s absent from the top-`k`
/// retrieved list, which is what makes a weak case actionable.
fn build_per_case_records(
    cases: &[SmokeCase],
    metrics: &[Metrics],
    ranks: &[CaseRankList],
) -> Vec<PerCaseRecord> {
    cases
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let m = &metrics[i];
            let topk: HashSet<&str> = ranks[i]
                .retrieved
                .iter()
                .take(SMOKE_K)
                .map(|s| s.as_str())
                .collect();
            let missed: Vec<String> = case
                .relevant
                .iter()
                .map(|r| r.corpus_item_id.clone())
                .filter(|id| !topk.contains(id.as_str()))
                .collect();
            let relevant_total = case.relevant.len();
            // Recall over wider cutoffs of the same retrieved list. When the
            // harness queries with a diagnostic `max_results` (RECALL_DIAG_K),
            // these split "gold ranked >10" from "gold never retrieved".
            let gold: HashSet<&str> =
                case.relevant.iter().map(|r| r.corpus_item_id.as_str()).collect();
            let recall_at = |k: usize| -> f64 {
                if gold.is_empty() {
                    return 0.0;
                }
                let topn: HashSet<&str> =
                    ranks[i].retrieved.iter().take(k).map(|s| s.as_str()).collect();
                let hit = gold.iter().filter(|g| topn.contains(*g)).count();
                hit as f64 / gold.len() as f64
            };
            PerCaseRecord {
                case_id: case.id.clone(),
                category: category_name(case.category).to_string(),
                query: case.query.clone(),
                ndcg_at_k: m.ndcg_at_k,
                recall_at_k: m.recall_at_k,
                mrr: m.mrr,
                p_at_1: m.p_at_1,
                relevant_total,
                relevant_found: relevant_total - missed.len(),
                missed,
                recall_at_50: recall_at(50),
                recall_at_100: recall_at(100),
            }
        })
        .collect()
}

/// Per-case results for one `LayerMode` within a single ingest+query pass.
struct ModePassResult {
    per_case: Vec<Metrics>,
    by_category_cases: HashMap<SmokeCategory, Vec<Metrics>>,
    latencies_ms: Vec<f64>,
    ranks: Vec<CaseRankList>,
}

/// Output of one ingest pass plus N mode-keyed query passes over the
/// fixture suite. Ingest happens once per pass; the case loop runs once
/// per `LayerMode` so per-mode quality and latency can be attributed
/// without paying the ingest cost N times. RH-8 (#270), RH-12 (#272).
struct OnePassResult {
    per_mode: BTreeMap<LayerMode, ModePassResult>,
    failures: Vec<Failure>,
}

/// Run a single ingest pass, then run the case loop once per `LayerMode`.
///
/// The caller is responsible for choosing the storage dir so multiple
/// repeats in the same harness invocation get independent state. The
/// corpus and cases are passed in by reference so we don't reload fixtures
/// per pass. The system is built once and reused across modes — the modes
/// themselves are pure read-side gates and do not mutate persisted state
/// in lower modes (Hebbian/access-count/competition writes are gated to
/// `Full`), so cross-mode contamination is impossible.
fn run_one_pass(
    storage_path: &Path,
    corpus: &[CorpusItem],
    cases: &[SmokeCase],
    layer_modes: &[LayerMode],
    age_days: f64,
) -> Result<OnePassResult> {
    // Ingest through the production manager so the graph/lineage/ontology layer
    // is actually built, then query the per-user `MemorySystem` (which now has a
    // populated graph wired in). The manager is kept alive for the whole pass.
    let manager = build_manager(storage_path)?;
    let id_map = ingest_corpus(&manager, corpus)?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // SHODH_VAMANA_QUALITY_REBUILD=1 — A/B lever: bulk alpha-RNG rebuild at the
    // ingest→query boundary. The harness is the only place that KNOWS this
    // boundary; a search-time trigger fires on remember()'s dedup search before
    // any insert and silently no-ops (caught by run 27255269454's gate). Vector
    // composition and the id mapping are preserved — only graph topology changes,
    // so the arm isolates greedy-insert vs alpha-RNG construction.
    if std::env::var("SHODH_VAMANA_QUALITY_REBUILD")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        system
            .read()
            .force_vector_quality_rebuild()
            .context("vamana quality rebuild at the ingest→query boundary")?;
    }

    // RH decay study: optionally age the knowledge-graph edges before querying
    // so recall quality reflects decayed/pruned edges. Driven at the production
    // ~6h cadence (see `MemorySystem::simulate_edge_aging`); a single large jump
    // would skip into the power-law decay phase and misrepresent the curve.
    // `age_days <= 0` is a no-op, preserving pre-existing behavior bit-for-bit.
    if age_days > 0.0 {
        system
            .read()
            .simulate_edge_aging(age_days, super::decay_sim::PRODUCTION_CADENCE_HOURS)
            .context("aging knowledge-graph edges for the decay study")?;
    }
    // Reverse map: random per-run UUIDs → stable corpus item IDs. The
    // determinism gate (RH-11) compares rank lists across runs, so we MUST
    // emit stable handles. Comparing UUIDs would always diverge because
    // each ingest assigns a fresh `Uuid::new_v4()`.
    let uuid_to_corpus_id: HashMap<Uuid, String> = id_map
        .iter()
        .map(|(corpus_id, uuid)| (*uuid, corpus_id.clone()))
        .collect();

    let mut per_mode: BTreeMap<LayerMode, ModePassResult> = BTreeMap::new();
    let mut failures: Vec<Failure> = Vec::new();

    // Diagnostic cutoff: when RECALL_DIAG_K is set, fetch a wider list so the
    // per-case recall@50/@100 fields are meaningful. Headline metrics still cut
    // at SMOKE_K. Defaults to SMOKE_K (no behavior change). Parse once per pass.
    let diag_k = std::env::var("RECALL_DIAG_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|k| *k >= SMOKE_K)
        .unwrap_or(SMOKE_K);

    // Approach C stage 2: per-query fusion-feature export for the offline trust
    // fit (SHODH_FUSION_FEATURE_EXPORT=<jsonl path>). Full mode only — that is
    // the production path the fit targets. Lines accumulate across modes/repeats
    // and append at pass end.
    let feature_export: Option<std::path::PathBuf> =
        std::env::var("SHODH_FUSION_FEATURE_EXPORT")
            .ok()
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from);
    let mut feature_lines: Vec<String> = Vec::new();

    for mode in layer_modes {
        let mut per_case = Vec::with_capacity(cases.len());
        let mut latencies_ms = Vec::with_capacity(cases.len());
        let mut by_category_cases: HashMap<SmokeCategory, Vec<Metrics>> = HashMap::new();
        let mut ranks: Vec<CaseRankList> = Vec::with_capacity(cases.len());

        for case in cases {
            let mut query = Query {
                query_text: Some(case.query.clone()),
                max_results: diag_k,
                layers: *mode,
                ..Default::default()
            };
            // SHODH_QUERY_NER A/B lever: neural-NER graph seeding (no-op when unset).
            manager.annotate_query_ner(&mut query);

            // Pure function of fixtures + ingest (hoisted above recall so the
            // feature exporter can arm with this query's gold ids).
            let relevance = build_relevance_map(case, &id_map);
            if feature_export.is_some() && matches!(*mode, LayerMode::Full) {
                crate::memory::fusion_features::begin(
                    relevance
                        .keys()
                        .map(|u| crate::memory::types::MemoryId(*u))
                        .collect(),
                );
            }

            let started = Instant::now();
            let result = system.read().recall(&query);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            latencies_ms.push(elapsed_ms);

            let memories = match result {
                Ok(m) => m,
                Err(e) => {
                    failures.push(Failure {
                        kind: "case".to_string(),
                        detail: format!(
                            "recall failed for {} [mode={}]: {e:#}",
                            case.id,
                            mode.report_key()
                        ),
                    });
                    // Treat as zero-recall so aggregates keep their denominator.
                    Vec::new()
                }
            };

            let retrieved_uuids: Vec<Uuid> = memories.iter().map(|m| m.id.0).collect();
            let retrieved_corpus_ids: Vec<String> = retrieved_uuids
                .iter()
                .map(|u| {
                    uuid_to_corpus_id
                        .get(u)
                        .cloned()
                        .unwrap_or_else(|| format!("<unknown:{u}>"))
                })
                .collect();
            ranks.push(CaseRankList {
                case_id: case.id.clone(),
                retrieved: retrieved_corpus_ids,
            });

            // Drain the per-query fusion features (None unless armed above AND
            // the FLAT fusion path ran) and pair them with the case outcome.
            if let Some(features) = crate::memory::fusion_features::take() {
                let gold_final_rank = retrieved_uuids
                    .iter()
                    .position(|u| relevance.contains_key(u));
                feature_lines.push(
                    serde_json::json!({
                        "case_id": case.id,
                        "category": format!("{:?}", case.category),
                        "query_len": case.query.len(),
                        "gold_final_rank": gold_final_rank,
                        "features": features,
                    })
                    .to_string(),
                );
            }

            // Only emit the "missing relevance map" failure once across modes
            // — the relevance map is a function of fixtures + ingest, not
            // the mode, so duplicating it would clutter the failure list.
            if relevance.is_empty() && *mode == layer_modes[0] {
                failures.push(Failure {
                    kind: "case".to_string(),
                    detail: format!(
                        "case {}: every relevant corpus item was missing from id_map (ingest skipped them)",
                        case.id
                    ),
                });
            }

            let metrics = Metrics::compute(&retrieved_uuids, &relevance, SMOKE_K);
            by_category_cases
                .entry(case.category)
                .or_default()
                .push(metrics);
            per_case.push(metrics);
        }

        per_mode.insert(
            *mode,
            ModePassResult {
                per_case,
                by_category_cases,
                latencies_ms,
                ranks,
            },
        );
    }

    if let Some(path) = &feature_export {
        if !feature_lines.is_empty() {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("opening feature export {}", path.display()))?;
            for line in &feature_lines {
                writeln!(f, "{line}")?;
            }
        }
    }

    Ok(OnePassResult { per_mode, failures })
}

/// Pin parallel runtimes to a single thread for the harness process.
///
/// Recall harness reproducibility requires that two runs of the same query
/// produce byte-identical rank lists. Multi-threaded float reductions
/// (RAYON par_iter) and ONNX intra-op parallelism both accumulate sums in
/// non-deterministic order, which is enough to flip ranks at the
/// fourth-decimal level.
///
/// This function only sets each variable when it is currently unset, so a
/// caller that explicitly chose a different value (e.g. for a benchmark)
/// keeps their override.
fn pin_harness_threads() {
    // SAFETY: env mutation is process-wide. The harness is the sole entry
    // point that calls this; the production server never invokes the
    // recall harness, so we are not racing other readers in any deployed
    // binary. The recall-eval CLI is single-threaded at startup.
    unsafe {
        if std::env::var_os("SHODH_ONNX_THREADS").is_none() {
            std::env::set_var("SHODH_ONNX_THREADS", "1");
        }
        if std::env::var_os("RAYON_NUM_THREADS").is_none() {
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }
    }
}

/// Construct an isolated `MemorySystem` rooted at `storage_path`.
/// Single tenant the harness ingests under.
pub const EVAL_USER: &str = "recall-eval";

/// Build the production `MultiUserMemoryManager` (NOT a bare `MemorySystem`).
///
/// FIDELITY (critical): a bare `MemorySystem` has `graph_memory = None` and
/// `remember()` deliberately does NOT build the entity graph — production builds
/// it in the HTTP handler via `process_experience_into_graph`. So the previous
/// harness measured the pipeline with the ENTIRE knowledge-graph / spreading-
/// activation / lineage / ontology layer disabled. The manager wires a per-user
/// graph + NER, so the eval exercises the same ingest path production does.
pub(crate) fn build_manager(storage_path: &Path) -> Result<MultiUserMemoryManager> {
    std::fs::create_dir_all(storage_path)
        .with_context(|| format!("creating storage dir {}", storage_path.display()))?;
    MultiUserMemoryManager::new(storage_path.to_path_buf(), ServerConfig::default())
        .context("initialising MultiUserMemoryManager for recall harness")
}

/// Ingest the corpus through the production ingest path (NER → remember →
/// graph), returning the `string handle → Uuid` map used to translate
/// ground-truth references to system memory IDs.
///
/// Mirrors the HTTP `remember` handler: run NER, merge entity names, store the
/// memory, then build the entity graph from it. Without the
/// `process_experience_into_graph` step the graph stays empty and Layer 2
/// spreading activation is a no-op.
pub fn ingest_corpus(
    manager: &MultiUserMemoryManager,
    corpus: &[CorpusItem],
) -> Result<HashMap<String, Uuid>> {
    let mut map = HashMap::with_capacity(corpus.len());
    let ner = manager.get_neural_ner();
    let user_mem = manager.get_user_memory(EVAL_USER)?;
    // Eval-fidelity gate: production embeds entity names so Tier-4 embedding
    // concept-merge runs (cross-memory entity resolution). The eval passed None →
    // Tier-4 OFF → entities under-resolved vs prod (every reach number measured
    // without it). SHODH_EVAL_NAME_EMB=1 populates name embeddings, matching the
    // handler (remember.rs:740). Default off → byte-identical (None).
    let name_emb_on = std::env::var("SHODH_EVAL_NAME_EMB")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    for item in corpus {
        // Pass 1: NER (faithful to the handler's NerEntityRecord shape).
        let ner_entities: Vec<NerEntityRecord> = match ner.extract(&item.content) {
            Ok(entities) => entities
                .into_iter()
                .map(|e| NerEntityRecord {
                    text: e.text,
                    entity_type: e.entity_type.as_str().to_string(),
                    confidence: e.confidence,
                    start_char: Some(e.start),
                    end_char: Some(e.end),
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        // Merge fixture tags + NER entity names (deduped, case-insensitive).
        let mut merged: Vec<String> = item.tags.clone();
        let mut seen: HashSet<String> = merged.iter().map(|t| t.to_lowercase()).collect();
        for r in &ner_entities {
            if seen.insert(r.text.to_lowercase()) {
                merged.push(r.text.clone());
            }
        }

        // Embed entity/tag names so Tier-4 concept-merge runs (gated; see above).
        // Computed before `merged` is moved into the experience.
        let name_embeddings: Option<HashMap<String, Vec<f32>>> =
            if name_emb_on && !merged.is_empty() {
                let refs: Vec<&str> = merged.iter().map(|s| s.as_str()).collect();
                match user_mem.read().get_embedder().encode_batch(&refs) {
                    Ok(vecs) => Some(merged.iter().cloned().zip(vecs).collect()),
                    Err(_) => None,
                }
            } else {
                None
            };

        let experience = Experience {
            experience_type: experience_type_for(&item.memory_type),
            content: item.content.clone(),
            entities: merged.clone(),
            tags: merged,
            ner_entities,
            ..Default::default()
        };

        let memory_id = user_mem
            .read()
            .remember(experience.clone(), Some(item.created_at))
            .with_context(|| format!("remembering corpus item {}", item.id))?;

        // Pass 2: build the entity graph from this memory (the step the bare
        // MemorySystem path skipped entirely).
        manager
            .process_experience_into_graph(
                EVAL_USER,
                &experience,
                &memory_id,
                name_embeddings.as_ref(),
            )
            .with_context(|| format!("graph-processing corpus item {}", item.id))?;

        map.insert(item.id.clone(), memory_id.0);
    }

    // Consolidation fidelity (SHODH_EVAL_CONSOLIDATE): production runs a periodic
    // HEAVY maintenance cycle that the harness previously skipped — so the eval
    // never exercised semantic-fact extraction, episodic replay, or tier
    // promotion (W6/W7), leaving the whole consolidation stack unmeasured. When
    // enabled, run one heavy maintenance pass after ingest so those
    // consolidation-driven features are actually populated. decay_factor = 1.0 →
    // no activation decay (forgetting/aging is the separate --age-days path), so
    // this isolates consolidation from decay. Default off → prior behaviour, so a
    // consolidate-off vs -on A/B measures the consolidation/fact contribution.
    if std::env::var("SHODH_EVAL_CONSOLIDATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        match user_mem.read().run_maintenance(1.0, EVAL_USER, true) {
            Ok(r) => tracing::info!(
                "eval consolidation cycle: facts_extracted~maintenance ran ({:?})",
                r
            ),
            Err(e) => tracing::warn!("eval consolidation cycle failed: {e}"),
        }
    }

    Ok(map)
}

/// Graph-reachability diagnostic — see [`ReachabilityReport`].
///
/// Ingests the corpus through the production graph-building path (identical to
/// the recall run), then for every case performs a pure breadth-first topology
/// walk from the query's seed entities over entity→entity relationships,
/// collecting the episodes attached to each entity at each depth. A gold memory
/// is "reachable within h hops" if its uuid appears among the episodes of any
/// entity within h relationship-hops of a seed entity. Activation weights, tier
/// trust, and the prune threshold are deliberately IGNORED: this measures
/// whether an associative PATH exists, not whether the current scoring would
/// traverse it. `within_2` (one bridge entity) is the canonical double-hop
/// signal for multi_hop.
/// Unified ablation matrix (E1): ingest the suite ONCE, then run the full case
/// set under each named query-time config and tabulate recall@10 / ndcg / mrr /
/// p@1 + per-category recall. The single, re-runnable place to see and update
/// every component's contribution — "don't fly blind".
///
/// Only QUERY-TIME flags are ablatable here (they are read per-recall against the
/// same ingested corpus). Ingest-time flags (concept entities, IDF edges, hub cap)
/// change what is stored and need separate ingests — run those via the workflow's
/// before/after `ref` A/B instead.
pub fn analyze_ablation(inputs: &RunInputs) -> Result<AblationReport> {
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    let manager = build_manager(&inputs.storage_path)?;
    let id_map = ingest_corpus(&manager, &corpus)?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // Query-time ablation configs: (display name, [(env_key, env_val)]). Add a
    // row here whenever a new query-time fix lands — that is how a fix becomes a
    // measured, comparable line in the study.
    let configs: Vec<(&str, Vec<(&str, &str)>)> = vec![
        ("facts-off (lexical+graph)", vec![("SHODH_DISABLE_FACT_LAYERS", "1")]),
        ("baseline (facts on)", vec![]),
        ("graph-off", vec![("SHODH_GRAPH_FUSION_WEIGHT", "0")]),
        ("+spread-fix", vec![("SHODH_SPREAD_FIX", "1")]),
        ("+graph-expand(K5)", vec![("SHODH_GRAPH_EXPAND_K", "5")]),
        (
            "+graph-expand+margin",
            vec![
                ("SHODH_GRAPH_EXPAND_K", "5"),
                ("SHODH_GRAPH_EXPAND_MIN_STRENGTH", "0.3"),
            ],
        ),
        (
            "+expand+spread-fix",
            vec![("SHODH_GRAPH_EXPAND_K", "5"), ("SHODH_SPREAD_FIX", "1")],
        ),
    ];

    let mut rows: Vec<AblationRow> = Vec::with_capacity(configs.len());
    for (name, env) in &configs {
        // SAFETY: the harness is single-threaded (pin_harness_threads) and the
        // production server never invokes it, so process-wide env mutation here
        // races no other reader. Set before the query pass, clear after.
        unsafe {
            for (k, v) in env {
                std::env::set_var(k, v);
            }
        }

        let mut by_cat: HashMap<SmokeCategory, Vec<f64>> = HashMap::new();
        let mut all: Vec<Metrics> = Vec::with_capacity(cases.len());
        for case in &cases {
            let query = Query {
                query_text: Some(case.query.clone()),
                max_results: SMOKE_K,
                layers: LayerMode::Full,
                ..Default::default()
            };
            let memories = system.read().recall(&query).unwrap_or_default();
            let retrieved: Vec<Uuid> = memories.iter().map(|m| m.id.0).collect();
            let relevance = build_relevance_map(case, &id_map);
            let m = Metrics::compute(&retrieved, &relevance, SMOKE_K);
            by_cat.entry(case.category).or_default().push(m.recall_at_k);
            all.push(m);
        }

        unsafe {
            for (k, _) in env {
                std::env::remove_var(k);
            }
        }

        let n = all.len().max(1) as f64;
        let by_category_recall = by_cat
            .iter()
            .map(|(c, rs)| {
                let r = rs.iter().sum::<f64>() / rs.len().max(1) as f64;
                (category_name(*c).to_string(), r)
            })
            .collect();
        rows.push(AblationRow {
            name: name.to_string(),
            flags: env.iter().map(|(k, v)| format!("{k}={v}")).collect(),
            recall_at_10: all.iter().map(|m| m.recall_at_k).sum::<f64>() / n,
            ndcg_at_10: all.iter().map(|m| m.ndcg_at_k).sum::<f64>() / n,
            mrr: all.iter().map(|m| m.mrr).sum::<f64>() / n,
            p_at_1: all.iter().map(|m| m.p_at_1).sum::<f64>() / n,
            by_category_recall,
        });
    }

    Ok(AblationReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        case_count: cases.len(),
        rows,
    })
}

/// Whole-word / phrase containment: does `name` appear in `text_lc` at word boundaries?
/// Both args must already be lowercased. Avoids spurious substring hits (e.g. "al" in "also").
fn phrase_in_text(name: &str, text_lc: &str) -> bool {
    if name.len() < 2 {
        return false;
    }
    let bytes = text_lc.as_bytes();
    let mut start = 0;
    while let Some(rel) = text_lc[start..].find(name) {
        let pos = start + rel;
        let before_ok = pos == 0 || !bytes[pos - 1].is_ascii_alphanumeric();
        let end = pos + name.len();
        let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        start = pos + 1;
        if start >= text_lc.len() {
            break;
        }
    }
    false
}

/// Query→graph entity-linking accuracy diagnostic. For each query: NER → mentions → link via
/// `find_entity_by_name` → linked graph nodes; compared against a lexical ground truth (graph
/// entity names that appear as phrases in the query). Reports link precision/recall, no-seed
/// rate, and extraction counts — turning "is NER the bottleneck?" into a number. Low recall ⇒
/// the traversal/graph starts from wrong-or-missing seeds and NER/linking is the ceiling.
pub fn analyze_linking(inputs: &RunInputs) -> Result<LinkingReport> {
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    let manager = build_manager(&inputs.storage_path)?;
    let _id_map = ingest_corpus(&manager, &corpus)?;
    let ner = manager.get_neural_ner();
    let graph = manager.get_user_graph(EVAL_USER)?;

    // Lexical ground truth: all graph entity names (lowercased) — the entities the corpus built.
    let entity_names_lc: Vec<String> = {
        let g = graph.read();
        g.get_all_entities()
            .unwrap_or_default()
            .into_iter()
            .map(|e| e.name.to_lowercase())
            .filter(|n| n.len() >= 2)
            .collect()
    };

    #[derive(Default)]
    struct Acc {
        cases: usize,
        ner_sum: usize,
        linked_sum: usize,
        no_seed: usize,
        lexical_sum: usize,
        tp_sum: usize,
        linked_total: usize,
        lexical_total: usize,
    }
    let mut overall = Acc::default();
    let mut by_cat: BTreeMap<String, Acc> = BTreeMap::new();

    for case in &cases {
        let cat = category_name(case.category).to_string();
        let query_lc = case.query.to_lowercase();

        let mentions: Vec<String> = match ner.extract(&case.query) {
            Ok(es) => es.into_iter().map(|e| e.text).collect(),
            Err(_) => Vec::new(),
        };
        let mut linked: HashSet<String> = HashSet::new();
        {
            let g = graph.read();
            for m in &mentions {
                if let Ok(Some(ent)) = g.find_entity_by_name(m) {
                    linked.insert(ent.name.to_lowercase());
                }
            }
        }
        let lexical: HashSet<String> = entity_names_lc
            .iter()
            .filter(|n| phrase_in_text(n, &query_lc))
            .cloned()
            .collect();
        let tp = linked.iter().filter(|n| lexical.contains(*n)).count();

        let (m, l, lex, no_seed) = (mentions.len(), linked.len(), lexical.len(), linked.is_empty());
        let bump = |acc: &mut Acc| {
            acc.cases += 1;
            acc.ner_sum += m;
            acc.linked_sum += l;
            if no_seed {
                acc.no_seed += 1;
            }
            acc.lexical_sum += lex;
            acc.tp_sum += tp;
            acc.linked_total += l;
            acc.lexical_total += lex;
        };
        bump(&mut overall);
        bump(by_cat.entry(cat).or_default());
    }

    let to_row = |cat: &str, a: &Acc| -> LinkingRow {
        let n = a.cases.max(1) as f64;
        LinkingRow {
            category: cat.to_string(),
            cases: a.cases,
            ner_mean: a.ner_sum as f64 / n,
            linked_mean: a.linked_sum as f64 / n,
            no_seed_pct: 100.0 * a.no_seed as f64 / n,
            lexical_mean: a.lexical_sum as f64 / n,
            link_precision: if a.linked_total == 0 {
                0.0
            } else {
                a.tp_sum as f64 / a.linked_total as f64
            },
            link_recall: if a.lexical_total == 0 {
                0.0
            } else {
                a.tp_sum as f64 / a.lexical_total as f64
            },
        }
    };

    let by_category: BTreeMap<String, LinkingRow> =
        by_cat.iter().map(|(c, a)| (c.clone(), to_row(c, a))).collect();
    Ok(LinkingReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        overall: to_row("ALL", &overall),
        by_category,
    })
}

/// Per-stage gold-rank funnel diagnostic. Runs the suite under Full mode with the gold ids
/// armed, recording at each pipeline-stage boundary (graph → vector → fusion → final) whether
/// the gold is present and its best rank. Aggregates the per-stage drop-off so the step that
/// loses reachable gold is LOCATED, not inferred (e.g. present after `graph`, absent after
/// `fusion` ⇒ RRF is burying it).
/// Apply the optional fast-read subsamples to a loaded `(cases, corpus)`.
/// `SHODH_MAX_CASES` strides the query set down (corpus stays full → distractor
/// difficulty unchanged); then `SHODH_MAX_CORPUS` shrinks the distractor pool while
/// ALWAYS retaining every gold item referenced by the surviving cases (directional
/// only — recall inflates). Both unset → returned unchanged (full scale). Shared so
/// the recall run and the funnel honor the SAME knobs; without it the funnel ran
/// full-scale (1531 × 5882) and timed out. Mirrors the inline blocks in
/// `run_smoke_suite_with_ranks` (dedupe those into this helper is a follow-up).
fn apply_eval_caps(cases: Vec<SmokeCase>, corpus: Vec<CorpusItem>) -> (Vec<SmokeCase>, Vec<CorpusItem>) {
    let cases = match std::env::var("SHODH_MAX_CASES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n < cases.len())
    {
        Some(n) => {
            let total = cases.len();
            let stride = (total / n).max(1);
            let sampled: Vec<_> = cases.into_iter().step_by(stride).take(n).collect();
            tracing::info!(
                "SHODH_MAX_CASES={n}: subsampled {} of {} cases (stride {})",
                sampled.len(),
                total,
                stride
            );
            sampled
        }
        None => cases,
    };
    let corpus = match std::env::var("SHODH_MAX_CORPUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n < corpus.len())
    {
        Some(n) => {
            let gold: std::collections::HashSet<&str> = cases
                .iter()
                .flat_map(|c| c.relevant.iter().map(|r| r.corpus_item_id.as_str()))
                .collect();
            let total = corpus.len();
            let gold_count = corpus.iter().filter(|it| gold.contains(it.id.as_str())).count();
            let keep_distractors = n.saturating_sub(gold_count);
            let distractor_total = total - gold_count;
            let stride = if keep_distractors == 0 {
                usize::MAX
            } else {
                (distractor_total / keep_distractors).max(1)
            };
            let mut seen_distractor = 0usize;
            let mut kept_distractors = 0usize;
            let reduced: Vec<_> = corpus
                .into_iter()
                .filter(|it| {
                    if gold.contains(it.id.as_str()) {
                        return true;
                    }
                    let take = kept_distractors < keep_distractors && seen_distractor % stride == 0;
                    seen_distractor += 1;
                    if take {
                        kept_distractors += 1;
                    }
                    take
                })
                .collect();
            tracing::info!(
                "SHODH_MAX_CORPUS={n}: reduced corpus {} → {} ({} gold + {} distractors)",
                total,
                reduced.len(),
                gold_count,
                kept_distractors
            );
            reduced
        }
        None => corpus,
    };
    (cases, corpus)
}

pub fn analyze_funnel(inputs: &RunInputs) -> Result<FunnelReport> {
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    // Honor the same fast-read caps as the recall run, or the funnel runs full-scale
    // (1531 cases × 5882 corpus) per arm and times out. Default unset → full scale.
    let (cases, corpus) = apply_eval_caps(cases, corpus);

    let manager = build_manager(&inputs.storage_path)?;
    let id_map = ingest_corpus(&manager, &corpus)?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // Same ingest→query boundary lever as run_one_pass — the funnel is a separate
    // pass and silently skipped the rebuild (run 27262844551 funnel-arm gate).
    if std::env::var("SHODH_VAMANA_QUALITY_REBUILD")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        system
            .read()
            .force_vector_quality_rebuild()
            .context("vamana quality rebuild at the funnel ingest→query boundary")?;
    }

    let diag_k = std::env::var("RECALL_DIAG_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|k| *k >= SMOKE_K)
        .unwrap_or(SMOKE_K);

    const STAGES: [&str; 5] = ["graph", "vector", "hybrid", "fusion", "final"];

    #[derive(Default, Clone)]
    struct Acc {
        seen: usize,
        present: usize,
        top_k: usize,
        rank_sum: usize,
    }
    impl Acc {
        fn record(&mut self, rank: Option<usize>) {
            self.seen += 1;
            if let Some(r) = rank {
                self.present += 1;
                self.rank_sum += r;
                if r < SMOKE_K {
                    self.top_k += 1;
                }
            }
        }
    }

    let mut overall: BTreeMap<String, Acc> = BTreeMap::new();
    let mut by_cat: BTreeMap<String, BTreeMap<String, Acc>> = BTreeMap::new();
    let mut case_count = 0usize;

    for case in &cases {
        let gold: HashSet<MemoryId> = case
            .relevant
            .iter()
            .filter_map(|r| id_map.get(&r.corpus_item_id).copied())
            .map(MemoryId)
            .collect();
        if gold.is_empty() {
            continue;
        }
        case_count += 1;
        let cat = category_name(case.category).to_string();

        let query = Query {
            query_text: Some(case.query.clone()),
            max_results: diag_k,
            layers: crate::memory::types::LayerMode::Full,
            ..Default::default()
        };

        crate::memory::gold_funnel::begin(gold.clone());
        let _ = system.read().recall(&query);
        let funnel = crate::memory::gold_funnel::take().unwrap_or_default();

        for (stage, rank) in &funnel {
            overall.entry(stage.clone()).or_default().record(*rank);
            by_cat
                .entry(cat.clone())
                .or_default()
                .entry(stage.clone())
                .or_default()
                .record(*rank);
        }
    }

    let to_rows = |m: &BTreeMap<String, Acc>| -> Vec<FunnelStageRow> {
        STAGES
            .iter()
            .filter_map(|s| m.get(*s).map(|a| (*s, a)))
            .map(|(s, a)| {
                let seen = a.seen.max(1) as f64;
                FunnelStageRow {
                    stage: s.to_string(),
                    present_pct: 100.0 * a.present as f64 / seen,
                    top10_pct: 100.0 * a.top_k as f64 / seen,
                    mean_rank_when_present: if a.present == 0 {
                        0.0
                    } else {
                        a.rank_sum as f64 / a.present as f64
                    },
                }
            })
            .collect()
    };

    let by_category: BTreeMap<String, Vec<FunnelStageRow>> =
        by_cat.iter().map(|(c, m)| (c.clone(), to_rows(m))).collect();

    Ok(FunnelReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        case_count,
        overall: to_rows(&overall),
        by_category,
    })
}

pub fn analyze_graph_reachability(inputs: &RunInputs) -> Result<ReachabilityReport> {
    pin_harness_threads();
    const MAX_HOPS: usize = 3;
    // Safety valve against hub-entity blowup on dense graphs. Far above any
    // LoCoMo component size, so it does not bias the result in practice.
    const MAX_VISITED_ENTITIES: usize = 50_000;

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    // Respect the same SHODH_MAX_CASES / SHODH_MAX_CORPUS caps as the recall and
    // funnel paths, so the reachability graph matches the graph those runs
    // measured (entity counts comparable to the 51%-present funnel) and ingest
    // time is bounded. Unset env → full corpus, identical to prior behaviour.
    let (cases, corpus) = apply_eval_caps(cases, corpus);

    let manager = build_manager(&inputs.storage_path)?;
    let id_map = ingest_corpus(&manager, &corpus)?;
    let ner = manager.get_neural_ner();
    let graph = manager.get_user_graph(EVAL_USER)?;
    // Mirror the ingest-side concept gate on the query side so the test is
    // symmetric: concept nodes added to the corpus are only seedable if the
    // query also extracts concepts.
    let concept_query_seeding = std::env::var("SHODH_CONCEPT_ENTITIES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let concept_seed_df_max: Option<usize> = std::env::var("SHODH_CONCEPT_SEED_DF_MAX")
        .ok()
        .and_then(|s| s.parse().ok());
    let keyword_extractor = manager.get_keyword_extractor();

    // Degree distribution of the built graph — the direct scoreboard for
    // anti-hub construction tuning (only visible at corpus scale, where the
    // speaker hubs actually form).
    let graph_structure = {
        const HUB_REPORT_THRESHOLD: usize = 50;
        let g = graph.read();
        let entities = g.get_all_entities().unwrap_or_default();
        let mut degrees: Vec<usize> = entities
            .iter()
            .map(|e| g.get_entity_relationships(&e.uuid).map(|r| r.len()).unwrap_or(0))
            .collect();
        degrees.sort_unstable_by(|a, b| b.cmp(a));
        let total_entities = degrees.len();
        let degree_sum: usize = degrees.iter().sum();
        GraphStructure {
            total_entities,
            total_edges: degree_sum / 2,
            max_degree: degrees.first().copied().unwrap_or(0),
            mean_degree: if total_entities == 0 {
                0.0
            } else {
                degree_sum as f64 / total_entities as f64
            },
            hub_count: degrees.iter().filter(|d| **d > HUB_REPORT_THRESHOLD).count(),
            hub_threshold: HUB_REPORT_THRESHOLD,
            top_degrees: degrees.iter().take(8).copied().collect(),
        }
    };

    let mut by_category: BTreeMap<String, ReachabilityCategory> = BTreeMap::new();
    let mut overall = ReachabilityCategory::default();

    for case in &cases {
        let cat = by_category
            .entry(category_name(case.category).to_string())
            .or_default();
        cat.cases += 1;
        overall.cases += 1;

        // Gold memory uuids for this case.
        let gold: HashSet<Uuid> = case
            .relevant
            .iter()
            .filter_map(|r| id_map.get(&r.corpus_item_id).copied())
            .collect();
        cat.gold_total += gold.len();
        overall.gold_total += gold.len();

        // Seed entities: NER over the query text, resolved to graph nodes by
        // name (same name-keyed lookup the ingest path builds them under).
        let mut seed_names: Vec<String> = match ner.extract(&case.query) {
            Ok(es) => es.into_iter().map(|e| e.text).collect(),
            Err(_) => Vec::new(),
        };
        // Symmetric concept seeding: when concept entities are added to the
        // corpus graph (SHODH_CONCEPT_ENTITIES), the QUERY must also resolve to
        // them or those nodes are unreachable as seeds. Extract YAKE keyphrases
        // from the query (mirroring the ingest-side concept extraction) so a
        // concept→concept seed can form. Without this the corpus concept nodes
        // exist but no query ever lands on them.
        if concept_query_seeding {
            for kw in keyword_extractor.extract(&case.query).into_iter().take(8) {
                let name = kw.text.trim().to_string();
                if name.len() >= 3 {
                    seed_names.push(name);
                }
            }
        }
        let g = graph.read();
        let mut seed_uuids: HashSet<Uuid> = HashSet::new();
        for name in &seed_names {
            if let Ok(Some(ent)) = g.find_entity_by_name(name) {
                // Mirror the live IDF gate (mod.rs): skip generic concept seeds
                // whose corpus mention_count exceeds SHODH_CONCEPT_SEED_DF_MAX.
                if let Some(df_max) = concept_seed_df_max {
                    let is_concept = matches!(
                        ent.labels.first(),
                        Some(crate::graph_memory::EntityLabel::Other(_))
                    );
                    if is_concept && ent.mention_count > df_max {
                        continue;
                    }
                }
                seed_uuids.insert(ent.uuid);
            }
        }
        if seed_uuids.is_empty() {
            cat.cases_no_seed += 1;
            overall.cases_no_seed += 1;
            cat.unreachable += gold.len();
            overall.unreachable += gold.len();
            continue;
        }
        if seed_uuids.len() >= 2 {
            cat.cases_multi_seed += 1;
            overall.cases_multi_seed += 1;
        }

        // BFS over entity→entity edges; record the min hop at which each gold
        // episode is first attached to a reached entity.
        let mut visited: HashSet<Uuid> = seed_uuids.clone();
        let mut frontier: Vec<Uuid> = seed_uuids.iter().copied().collect();
        let mut gold_min_hop: HashMap<Uuid, usize> = HashMap::new();
        // Per-gold set of distinct query seeds directly (1-hop) attached — the
        // G5 multi-seed discrimination signal.
        let mut gold_seed_cov: HashMap<Uuid, HashSet<Uuid>> = HashMap::new();
        for hop in 1..=MAX_HOPS {
            for ent in &frontier {
                // At hop 1 the frontier is exactly the query seeds.
                let ent_is_seed = hop == 1;
                if let Ok(eps) = g.get_episodes_by_entity(ent) {
                    for ep in eps {
                        if gold.contains(&ep.uuid) {
                            gold_min_hop.entry(ep.uuid).or_insert(hop);
                            if ent_is_seed {
                                gold_seed_cov.entry(ep.uuid).or_default().insert(*ent);
                            }
                        }
                    }
                }
            }
            if hop == MAX_HOPS || gold_min_hop.len() == gold.len() {
                break;
            }
            let mut next: Vec<Uuid> = Vec::new();
            for ent in &frontier {
                if let Ok(edges) = g.get_entity_relationships(ent) {
                    for e in edges {
                        for nbr in [e.from_entity, e.to_entity] {
                            if nbr != *ent && visited.insert(nbr) {
                                next.push(nbr);
                            }
                        }
                    }
                }
                if visited.len() >= MAX_VISITED_ENTITIES {
                    break;
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        drop(g);

        for gid in &gold {
            match gold_min_hop.get(gid).copied() {
                Some(1) => {
                    cat.reachable_within_1 += 1;
                    cat.reachable_within_2 += 1;
                    cat.reachable_within_3 += 1;
                    overall.reachable_within_1 += 1;
                    overall.reachable_within_2 += 1;
                    overall.reachable_within_3 += 1;
                }
                Some(2) => {
                    cat.reachable_within_2 += 1;
                    cat.reachable_within_3 += 1;
                    overall.reachable_within_2 += 1;
                    overall.reachable_within_3 += 1;
                }
                Some(_) => {
                    cat.reachable_within_3 += 1;
                    overall.reachable_within_3 += 1;
                }
                None => {
                    cat.unreachable += 1;
                    overall.unreachable += 1;
                }
            }
            // G5 signal availability: is this gold directly attached to ≥2
            // distinct query seeds?
            if gold_seed_cov.get(gid).map(|s| s.len()).unwrap_or(0) >= 2 {
                cat.gold_multi_seed += 1;
                overall.gold_multi_seed += 1;
            }
        }
    }

    Ok(ReachabilityReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        max_hops: MAX_HOPS,
        overall,
        by_category,
        graph: graph_structure,
    })
}

/// Learning-curve diagnostic — see [`LearningCurveReport`]. The flagship test of
/// "smarter with use": does a memory become EASIER to recall as it is used?
///
/// Protocol per case (only those whose cold gold-rank lands in the headroom band
/// — already-top-1 has nowhere to climb, very-deep gold won't surface in a few
/// cycles): recall the query, then apply `Helpful` feedback to the gold via
/// `reinforce_recall` (strengthens the gold's graph edges, importance-gated), and
/// re-measure the gold's rank + score. Repeat `cycles` times. `recall` itself
/// also auto-reinforces every returned memory (access-count + co-activation), so
/// distractors are reinforced too — only the *extra* `Helpful` signal on the gold
/// breaks the symmetry. A genuine associative memory shows the gold's rank fall
/// and score rise across cycles; a static retriever stays flat. Full mode only
/// (Hebbian/LTP writes are gated to `Full`).
/// One reinforcement-outcome arm: fresh ingest, then for each headroom case run
/// `cyc` rounds of recall + `outcome` feedback on the gold, tracking the gold's
/// rank/score per cycle.
fn run_learning_arm(
    storage_path: &Path,
    corpus: &[CorpusItem],
    cases: &[SmokeCase],
    cyc: usize,
    outcome: RetrievalOutcome,
    outcome_name: &str,
) -> Result<LearningCurveArm> {
    const TRACK_K: usize = 50;
    const MIN_COLD_RANK: usize = 2;
    const MAX_COLD_RANK: usize = 30;

    let manager = build_manager(storage_path)?;
    let id_map = ingest_corpus(&manager, corpus)?;
    let system = manager.get_user_memory(EVAL_USER)?;

    // Recall the query (Full mode → reinforcement active) and return the ranked
    // (uuid, score) list, wide enough to observe the gold's rank.
    let recall_ranked = |q: &str| -> Vec<(Uuid, f32)> {
        let query = Query {
            query_text: Some(q.to_string()),
            max_results: TRACK_K,
            layers: LayerMode::Full,
            ..Default::default()
        };
        match system.read().recall(&query) {
            Ok(mems) => mems.iter().map(|m| (m.id.0, m.score.unwrap_or(0.0))).collect(),
            Err(_) => Vec::new(),
        }
    };

    let mut rank_sums = vec![0.0f64; cyc + 1];
    let mut score_sums = vec![0.0f64; cyc + 1];
    let mut tracked = 0usize;
    let (mut improved, mut worsened, mut unchanged) = (0usize, 0usize, 0usize);
    let mut rank_delta_sum = 0.0f64;
    let mut score_delta_sum = 0.0f64;

    for case in cases {
        let gold: HashSet<Uuid> = case
            .relevant
            .iter()
            .filter_map(|r| id_map.get(&r.corpus_item_id).copied())
            .collect();
        if gold.is_empty() {
            continue;
        }
        // Best-ranked gold's (rank, score) in a ranked list; 1-based.
        let gold_rank = |ranked: &[(Uuid, f32)]| -> Option<(usize, f32)> {
            ranked
                .iter()
                .enumerate()
                .find(|(_, (u, _))| gold.contains(u))
                .map(|(i, (_, s))| (i + 1, *s))
        };

        let cold = recall_ranked(&case.query);
        let (r0, s0) = match gold_rank(&cold) {
            Some(x) => x,
            None => continue,
        };
        if !(MIN_COLD_RANK..=MAX_COLD_RANK).contains(&r0) {
            continue;
        }

        tracked += 1;
        rank_sums[0] += r0 as f64;
        score_sums[0] += s0 as f64;

        let gold_ids: Vec<MemoryId> = gold.iter().map(|u| MemoryId(*u)).collect();
        let (mut last_rank, mut last_score) = (r0, s0);
        for c in 1..=cyc {
            let _ = system.read().reinforce_recall(&gold_ids, outcome);
            let ranked = recall_ranked(&case.query);
            let (rc, sc) = gold_rank(&ranked).unwrap_or((TRACK_K + 1, 0.0));
            rank_sums[c] += rc as f64;
            score_sums[c] += sc as f64;
            last_rank = rc;
            last_score = sc;
        }

        rank_delta_sum += last_rank as f64 - r0 as f64;
        score_delta_sum += (last_score - s0) as f64;
        match last_rank.cmp(&r0) {
            std::cmp::Ordering::Less => improved += 1,
            std::cmp::Ordering::Greater => worsened += 1,
            std::cmp::Ordering::Equal => unchanged += 1,
        }
    }

    let n = tracked.max(1) as f64;
    Ok(LearningCurveArm {
        outcome: outcome_name.to_string(),
        tracked_cases: tracked,
        mean_rank_by_cycle: rank_sums.iter().map(|s| s / n).collect(),
        mean_score_by_cycle: score_sums.iter().map(|s| s / n).collect(),
        improved,
        worsened,
        unchanged,
        mean_rank_delta: rank_delta_sum / n,
        mean_score_delta: score_delta_sum / n,
    })
}

/// Learning-curve diagnostic — see [`LearningCurveReport`]. Runs three arms
/// (Helpful / Neutral / Misleading), each on a FRESH ingest, to expose the
/// reward GRADIENT: a real reward-modulated memory pushes the gold UP under
/// Helpful, DOWN under Misleading, and flat under Neutral. Amplify the reward
/// learning rate via `SHODH_REWARD_LR_MULT` and re-run to see if a stronger
/// reward moves rank, not just score.
pub fn analyze_learning_curve(inputs: &RunInputs, cycles: usize) -> Result<LearningCurveReport> {
    pin_harness_threads();

    let corpus_path = inputs
        .corpus_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CORPUS_PATH));
    let cases_path = inputs
        .cases_path
        .clone()
        .unwrap_or_else(|| fixtures::manifest_path(SMOKE_CASES_PATH));
    let corpus = fixtures::load_corpus(&corpus_path)
        .with_context(|| format!("loading corpus from {}", corpus_path.display()))?;
    let cases = fixtures::load_smoke_cases(&cases_path)
        .with_context(|| format!("loading cases from {}", cases_path.display()))?;
    fixtures::validate_structure(&corpus, &cases)
        .with_context(|| format!("{} suite failed structural validation", inputs.suite))?;

    let cyc = cycles.max(1);
    let arms_spec = [
        ("Helpful", RetrievalOutcome::Helpful),
        ("Neutral", RetrievalOutcome::Neutral),
        ("Misleading", RetrievalOutcome::Misleading),
    ];
    let mut arms = Vec::with_capacity(arms_spec.len());
    for (name, outcome) in arms_spec {
        // Fresh storage per arm — reinforcement mutates state; arms must not bleed.
        let arm_storage = inputs.storage_path.join(name.to_lowercase());
        let arm = run_learning_arm(&arm_storage, &corpus, &cases, cyc, outcome, name)
            .with_context(|| format!("learning-curve arm {name}"))?;
        arms.push(arm);
    }

    Ok(LearningCurveReport {
        suite: inputs.suite.clone(),
        git_sha: inputs.git_sha.clone(),
        cycles: cyc,
        arms,
    })
}

/// Map a corpus item's `memory_type` string to an `ExperienceType` variant.
///
/// Unknown types fall back to `Observation`, which is the most neutral
/// option in the existing taxonomy. This keeps fixture authoring lenient
/// while still flowing real type information through the pipeline when it
/// is recognised.
fn experience_type_for(s: &str) -> ExperienceType {
    match s.to_ascii_lowercase().as_str() {
        "decision" => ExperienceType::Decision,
        "task" => ExperienceType::Task,
        "conversation" => ExperienceType::Conversation,
        "error" => ExperienceType::Error,
        "learning" => ExperienceType::Learning,
        "discovery" => ExperienceType::Discovery,
        "pattern" => ExperienceType::Pattern,
        "context" => ExperienceType::Context,
        "event" | "reference" | "observation" => ExperienceType::Observation,
        _ => ExperienceType::Observation,
    }
}

/// Build the graded relevance map for a single case using the ingest id map.
///
/// Items that did not make it into `id_map` are silently skipped here; the
/// caller logs a `case` failure if the resulting map is empty.
fn build_relevance_map(case: &SmokeCase, id_map: &HashMap<String, Uuid>) -> HashMap<Uuid, f32> {
    let mut out = HashMap::with_capacity(case.relevant.len());
    for r in &case.relevant {
        if let Some(uuid) = id_map.get(&r.corpus_item_id) {
            out.insert(*uuid, r.grade as f32);
        }
    }
    out
}

fn category_name(c: SmokeCategory) -> &'static str {
    match c {
        SmokeCategory::Decision => "decision",
        SmokeCategory::Code => "code",
        SmokeCategory::Temporal => "temporal",
        SmokeCategory::Entity => "entity",
        SmokeCategory::MultiHop => "multi_hop",
        SmokeCategory::Negation => "negation",
        SmokeCategory::SingleHop => "single_hop",
        SmokeCategory::OpenDomain => "open_domain",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_storage_dir(label: &str) -> PathBuf {
        let id = Uuid::new_v4().simple().to_string();
        std::env::temp_dir().join(format!("shodh-recall-{label}-{id}"))
    }

    /// Lineage repro (substrate diagnosis 2026-06-10): root-cause P@1 has been
    /// 0.0 through every fix, and the instrumented CI run produced ZERO edge
    /// provenance lines and ZERO causal-funnel lines — hypothesis: the chain
    /// entities never become graph structure at all. This test pushes one
    /// lineage chain through the REAL harness ingest (ingest_corpus →
    /// process_experience_into_graph) and asserts stage by stage:
    ///   1. the chain entities exist in the graph by name
    ///   2. edges connect them
    ///   3. at least one connecting edge is typed CAUSAL (the walk's food)
    /// Whichever assert fails localizes the lineage zero.
    #[test]
    fn lineage_chain_builds_causal_graph_structure() {
        let dir = unique_storage_dir("lineage-repro");
        let manager = build_manager(&dir).expect("manager");
        let (corpus, _cases) = crate::recall_harness::lineage_harness::generate_lineage_fixtures(1);
        ingest_corpus(&manager, &corpus).expect("ingest");

        let graph = manager.get_user_graph(EVAL_USER).expect("graph");
        let g = graph.read();

        // Chain 0 entities: "the Vornak incident" → "the Meslin incident" →
        // "the Caldor incident" (event(0..3)).
        let names = ["the Vornak incident", "the Meslin incident", "the Caldor incident"];
        let mut uuids = Vec::new();
        for name in names {
            let found = g.find_entity_by_name(name).expect("lookup");
            assert!(
                found.is_some(),
                "STAGE 1 FAIL: chain entity '{name}' missing from graph — \
                 entity creation is the lineage zero"
            );
            uuids.push(found.unwrap().uuid);
        }

        // Edges between consecutive chain entities?
        for w in uuids.windows(2) {
            let edges = g
                .get_entity_relationships_limited(&w[0], Some(64))
                .expect("edges");
            let connecting: Vec<_> = edges
                .iter()
                .filter(|e| {
                    (e.from_entity == w[0] && e.to_entity == w[1])
                        || (e.from_entity == w[1] && e.to_entity == w[0])
                })
                .collect();
            assert!(
                !connecting.is_empty(),
                "STAGE 2 FAIL: no edge between consecutive chain entities — \
                 pair-loop/edge creation is the lineage zero"
            );
            let causal = connecting.iter().any(|e| e.relation_type.is_causal());
            assert!(
                causal,
                "STAGE 3 FAIL: edges exist but none is causal (types: {:?}) — \
                 the typing chain is the lineage zero",
                connecting
                    .iter()
                    .map(|e| e.relation_type.as_str())
                    .collect::<Vec<_>>()
            );
        }

        // STAGE 4: the backward origin walk itself. From the observed effect
        // (Caldor), trace_causal_origins must return the root (Vornak).
        let origins = g
            .trace_causal_origins(&[uuids[2]], 8)
            .expect("origin walk");
        assert!(
            origins.contains(&uuids[0]),
            "STAGE 4 FAIL: backward walk from the effect did not reach the \
             root cause (origins: {} found) — edge direction or walk is the \
             lineage zero",
            origins.len()
        );
        drop(g);

        // STAGE 5: the full recall path. The root memory (which shares NO token
        // with the query other than what the walk provides) must appear in the
        // results for the root-cause query.
        let system = manager.get_user_memory(EVAL_USER).expect("system");
        let query = crate::memory::types::Query {
            query_text: Some("What was the earliest origin behind the Caldor incident?".into()),
            max_results: 10,
            layers: LayerMode::Full,
            ..Default::default()
        };
        let results = system.read().recall(&query).expect("recall");
        let texts: Vec<String> = results
            .iter()
            .map(|m| m.experience.content.clone())
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("Vornak")),
            "STAGE 5 FAIL: walk works but the root memory is not in the top-10 \
             — the origins-USAGE path (BM25 cue expansion) is the lineage zero. \
             Retrieved: {texts:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Same pipeline at HARNESS scale: with the default chain count every
    /// memory shares the "incident"/"set in motion"/"brought about" phrasing,
    /// so the failure mode (lineage P@1 = 0.0 in CI while the 1-chain repro
    /// passes end-to-end) must be scale-induced — dilution of the walked
    /// origin among cross-chain lexical distractors, or cross-chain entity
    /// resolution bleeding (substring tiers matching "Vornak" ↔ "Vornak2").
    #[test]
    fn lineage_walk_survives_harness_scale() {
        let dir = unique_storage_dir("lineage-scale");
        let manager = build_manager(&dir).expect("manager");
        let chains = crate::recall_harness::multihop::DEFAULT_CHAINS;
        let (corpus, _cases) = crate::recall_harness::lineage_harness::generate_lineage_fixtures(chains);
        ingest_corpus(&manager, &corpus).expect("ingest");

        let graph = manager.get_user_graph(EVAL_USER).expect("graph");
        let g = graph.read();
        // Probe a middle chain (i=5): events 15, 16, 17.
        let a = g
            .find_entity_by_name("the Polnar incident")
            .expect("lookup")
            .map(|e| e.uuid);
        let c = g
            .find_entity_by_name("the Selvic incident")
            .expect("lookup")
            .map(|e| e.uuid);
        let (a, c) = match (a, c) {
            (Some(a), Some(c)) => (a, c),
            other => panic!(
                "SCALE STAGE 1 FAIL: chain-5 entities unresolved at {chains} chains: {other:?} \
                 — cross-chain entity-resolution bleeding"
            ),
        };
        let origins = g.trace_causal_origins(&[c], 8).expect("walk");
        if !origins.contains(&a) {
            // Diagnostics: what does the effect node's neighbourhood look like?
            let c_edges = g
                .get_entity_relationships_limited(&c, Some(64))
                .expect("edges");
            let incoming_causal = c_edges
                .iter()
                .filter(|e| e.to_entity == c && e.relation_type.is_causal())
                .count();
            let type_counts: std::collections::HashMap<&str, usize> =
                c_edges.iter().fold(Default::default(), |mut m, e| {
                    *m.entry(e.relation_type.as_str()).or_default() += 1;
                    m
                });
            panic!(
                "SCALE STAGE 4 FAIL at {chains} chains: {} origins; effect node has \
                 {} edges (incoming causal: {incoming_causal}; types: {type_counts:?}); \
                 graph stats: {:?}",
                origins.len(),
                c_edges.len(),
                g.get_stats()
                    .map(|s| (s.entity_count, s.relationship_count))
                    .ok()
            );
        }
        drop(g);

        let system = manager.get_user_memory(EVAL_USER).expect("system");
        let query = crate::memory::types::Query {
            query_text: Some("What was the earliest origin behind the Selvic incident?".into()),
            max_results: 10,
            layers: LayerMode::Full,
            ..Default::default()
        };
        let results = system.read().recall(&query).expect("recall");
        let texts: Vec<String> = results
            .iter()
            .map(|m| m.experience.content.clone())
            .collect();
        let root_rank = texts.iter().position(|t| t.contains("Polnar"));
        assert!(
            root_rank.is_some(),
            "SCALE STAGE 5 FAIL: root memory absent from top-10 at {chains} chains \
             — walked origins diluted by cross-chain lexical distractors. \
             Retrieved: {texts:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn per_case_records_flag_missed_relevant_items() {
        let cases = vec![
            SmokeCase {
                id: "smoke-001".into(),
                category: SmokeCategory::MultiHop,
                query: "chain query".into(),
                fixture_corpus_id: "shodh-smoke".into(),
                relevant: vec![
                    fixtures::RelevanceJudgement {
                        corpus_item_id: "ssm-001".into(),
                        grade: 3,
                    },
                    fixtures::RelevanceJudgement {
                        corpus_item_id: "ssm-002".into(),
                        grade: 1,
                    },
                ],
            },
            SmokeCase {
                id: "smoke-002".into(),
                category: SmokeCategory::Entity,
                query: "entity query".into(),
                fixture_corpus_id: "shodh-smoke".into(),
                relevant: vec![fixtures::RelevanceJudgement {
                    corpus_item_id: "ssm-010".into(),
                    grade: 3,
                }],
            },
        ];
        let metrics = vec![
            Metrics {
                recall_at_k: 0.5,
                ndcg_at_k: 0.6,
                mrr: 1.0,
                p_at_1: 1.0,
                ..Default::default()
            },
            Metrics {
                recall_at_k: 1.0,
                ndcg_at_k: 1.0,
                mrr: 1.0,
                p_at_1: 1.0,
                ..Default::default()
            },
        ];
        let ranks = vec![
            // Case 1 retrieved ssm-001 (found) but dropped ssm-002 (missed).
            CaseRankList {
                case_id: "smoke-001".into(),
                retrieved: vec!["ssm-001".into(), "ssm-099".into()],
            },
            // Case 2 retrieved its only relevant item.
            CaseRankList {
                case_id: "smoke-002".into(),
                retrieved: vec!["ssm-010".into()],
            },
        ];

        let recs = build_per_case_records(&cases, &metrics, &ranks);
        assert_eq!(recs.len(), 2);

        let r0 = &recs[0];
        assert_eq!(r0.case_id, "smoke-001");
        assert_eq!(r0.category, "multi_hop");
        assert_eq!(r0.relevant_total, 2);
        assert_eq!(r0.relevant_found, 1);
        assert_eq!(r0.missed, vec!["ssm-002".to_string()]);
        assert_eq!(r0.recall_at_k, 0.5);
        assert_eq!(r0.ndcg_at_k, 0.6);
        // Wider cutoffs recompute from the full retrieved list vs gold: case 1
        // found ssm-001 but not ssm-002, so 1/2 at every cutoff.
        assert_eq!(r0.recall_at_50, 0.5);
        assert_eq!(r0.recall_at_100, 0.5);

        let r1 = &recs[1];
        assert_eq!(r1.category, "entity");
        assert_eq!(r1.relevant_found, 1);
        assert!(r1.missed.is_empty());
        assert_eq!(r1.recall_at_50, 1.0);
        assert_eq!(r1.recall_at_100, 1.0);
    }

    /// Smoke test the full runner end-to-end against the canonical fixtures.
    ///
    /// Asserts only that the report is well-formed: 30 cases, six categories
    /// represented, full layer present. Quality numbers themselves are
    /// captured by RH-6 baseline runs, not by unit tests.
    #[test]
    fn runner_executes_smoke_suite_and_produces_well_formed_report() {
        let storage = unique_storage_dir("runner");
        let inputs = RunInputs {
            storage_path: storage.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };
        let report = run_smoke_suite(&inputs).expect("runner must succeed");
        let _ = std::fs::remove_dir_all(&storage);

        assert_eq!(
            report.case_count,
            crate::recall_harness::fixtures::TOTAL_SMOKE_CASES
        );
        assert_eq!(report.repeats, 1);
        assert_eq!(report.suite, "smoke");
        assert_eq!(report.embedder, EMBEDDER_ID);
        assert!(report.layers.contains_key("full"));
        for cat in [
            "decision",
            "code",
            "temporal",
            "entity",
            "multi_hop",
            "negation",
        ] {
            let cr = report
                .by_category
                .get(cat)
                .expect("category must be present");
            assert_eq!(
                cr.case_count,
                crate::recall_harness::fixtures::CASES_PER_CATEGORY,
                "category {cat} must have CASES_PER_CATEGORY cases"
            );
        }
        // No infrastructure-level failures expected on the canonical fixtures.
        assert!(
            report.failures.iter().all(|f| f.kind == "case"),
            "no infrastructure failures expected, got {:?}",
            report.failures
        );
    }

    /// OBSERVATION (G3 debug): the harness shows ACTR_FUSION changing the
    /// `vamana_only` ranking, but reading the code says it cannot (graph leg empty,
    /// overlays gated off, both fusions monotonic in the vector score). Rather than
    /// keep predicting, run the real recall path both ways on a tiny known corpus
    /// and PRINT the orderings, so the divergence is observed, not inferred.
    #[test]
    fn observe_actr_fusion_vamana_only_ranking() {
        use crate::recall_harness::fixtures::CorpusItem;
        use chrono::{Duration, TimeZone, Utc};

        let storage = unique_storage_dir("actr-observe");
        let manager = build_manager(&storage).unwrap();
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mk = |id: &str, content: &str, i: i64| CorpusItem {
            id: id.to_string(),
            content: content.to_string(),
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + Duration::minutes(i),
        };
        // Competing vector matches: c0/c4 both share "brown"; query favours c0.
        let corpus = vec![
            mk("c0", "The brown fox leaps over the lazy dog in the meadow", 0),
            mk("c1", "A red car drove down the busy city highway at night", 1),
            mk("c2", "The chef prepared a delicious pasta with fresh basil", 2),
            mk("c3", "Quantum computers use qubits for parallel computation", 3),
            mk("c4", "The brown bear caught a salmon in the rushing river", 4),
            mk("c5", "She painted a vivid sunset over the calm blue ocean", 5),
        ];
        let id_map = ingest_corpus(&manager, &corpus).unwrap();
        let rev: HashMap<Uuid, String> =
            id_map.iter().map(|(cid, u)| (*u, cid.clone())).collect();
        let system = manager.get_user_memory(EVAL_USER).unwrap();

        let recall_order = |q: &str| -> Vec<(String, f32)> {
            let query = Query {
                query_text: Some(q.to_string()),
                max_results: 10,
                layers: LayerMode::VamanaOnly,
                ..Default::default()
            };
            match system.read().recall(&query) {
                Ok(mems) => mems
                    .iter()
                    .map(|m| {
                        (
                            rev.get(&m.id.0).cloned().unwrap_or_else(|| m.id.0.to_string()),
                            m.score.unwrap_or(0.0),
                        )
                    })
                    .collect(),
                Err(e) => panic!("recall failed: {e}"),
            }
        };

        std::env::remove_var("SHODH_ACTR_FUSION");
        let off = recall_order("brown fox meadow");
        std::env::set_var("SHODH_ACTR_FUSION", "1");
        let on = recall_order("brown fox meadow");
        std::env::remove_var("SHODH_ACTR_FUSION");
        let _ = std::fs::remove_dir_all(&storage);

        eprintln!("VAMANA_ONLY  OFF (RRF)  : {off:?}");
        eprintln!("VAMANA_ONLY  ON  (ACTR) : {on:?}");

        let off_ids: Vec<&String> = off.iter().map(|(id, _)| id).collect();
        let on_ids: Vec<&String> = on.iter().map(|(id, _)| id).collect();
        assert_eq!(
            off_ids, on_ids,
            "ACTR_FUSION changed the vamana_only ranking — the code said it could not.\n OFF={off:?}\n ON ={on:?}"
        );
    }

    /// OBSERVATION (FUSION_V2): does Borda+rescue actually lift a graph-reachable
    /// answer that RRF buries? Planted 2-hop: query entity Alice —(bridge)→ Bob;
    /// gold "Bob discovered ..." is reachable via the graph but the lexical
    /// distractor "Alice discovered ..." matches the query better. Print the order
    /// and the gold's rank under RRF vs V2 — observe, don't assert.
    #[test]
    fn observe_fusion_v2_graph_rescue() {
        use crate::recall_harness::fixtures::CorpusItem;
        use chrono::{Duration, TimeZone, Utc};

        let storage = unique_storage_dir("v2-observe");
        let manager = build_manager(&storage).unwrap();
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let mk = |id: &str, content: &str, i: i64| CorpusItem {
            id: id.to_string(),
            content: content.to_string(),
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + Duration::minutes(i),
        };
        let mut corpus = vec![
            mk("bridge", "Alice collaborates closely with Bob on their research", 0),
            mk("gold", "Bob discovered a rare luminescent mineral inside the cave", 1),
            mk("distract", "Alice discovered an ordinary grey rock out in the field", 2),
        ];
        // Many noise memories so a graph-only answer is genuinely OUTSIDE the
        // hybrid top-`max_results` window (the rescue gate). Without this the
        // rescue can never fire (corpus must exceed max_results).
        for i in 0..24 {
            corpus.push(mk(
                &format!("noise{i:02}"),
                &format!("Unrelated note number {i} about gardening, cooking, and travel plans"),
                10 + i,
            ));
        }
        let id_map = ingest_corpus(&manager, &corpus).unwrap();
        let rev: HashMap<Uuid, String> =
            id_map.iter().map(|(c, u)| (*u, c.clone())).collect();
        let system = manager.get_user_memory(EVAL_USER).unwrap();

        let recall_order = |q: &str| -> Vec<(String, f32)> {
            let query = Query {
                query_text: Some(q.to_string()),
                max_results: 3,
                layers: LayerMode::Full,
                ..Default::default()
            };
            system
                .read()
                .recall(&query)
                .map(|ms| {
                    ms.iter()
                        .map(|m| {
                            (
                                rev.get(&m.id.0).cloned().unwrap_or_else(|| m.id.0.to_string()),
                                m.score.unwrap_or(0.0),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        let q = "What did Alice's collaborator discover?";
        std::env::remove_var("SHODH_FUSION_V2");
        let off = recall_order(q);
        std::env::set_var("SHODH_FUSION_V2", "1");
        let on = recall_order(q);
        std::env::remove_var("SHODH_FUSION_V2");
        let _ = std::fs::remove_dir_all(&storage);

        let rank = |v: &Vec<(String, f32)>, id: &str| v.iter().position(|(i, _)| i == id);
        eprintln!("FULL  OFF (RRF): {off:?}");
        eprintln!("FULL  ON  (V2) : {on:?}");
        eprintln!(
            "gold rank  OFF={:?}  ON={:?}   distract rank OFF={:?} ON={:?}",
            rank(&off, "gold"),
            rank(&on, "gold"),
            rank(&off, "distract"),
            rank(&on, "distract"),
        );
    }

    /// OBSERVATION: the clean A/B showed V2 ontology +facts=0.88 but full=0.083.
    /// Reproduce the harness scenario (short "Alice" vs longer org names, equi-frame)
    /// and check the person's rank at PlusFacts vs Full under V2 — to confirm the
    /// content-length quality gate was the `full`-layer destroyer (and that dropping
    /// the verbosity factor keeps the person #1 at full too).
    #[test]
    fn observe_v2_full_layer_ontology() {
        use crate::recall_harness::fixtures::CorpusItem;
        use chrono::{Duration, TimeZone, Utc};

        let storage = unique_storage_dir("v2-onto");
        let manager = build_manager(&storage).unwrap();
        let base = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let frame = |e: &str| format!("{e} was registered for the Paris trade summit.");
        let mk = |id: &str, content: String, i: i64| CorpusItem {
            id: id.to_string(),
            content,
            memory_type: "fact".to_string(),
            tags: vec![],
            created_at: base + Duration::minutes(i),
        };
        let corpus = vec![
            mk("person", frame("Alice"), 0),
            mk("org1", frame("Acme Corporation"), 1),
            mk("org2", frame("Globex Industries"), 2),
            mk("org3", frame("Initech Systems"), 3),
        ];
        let id_map = ingest_corpus(&manager, &corpus).unwrap();
        let rev: HashMap<Uuid, String> =
            id_map.iter().map(|(c, u)| (*u, c.clone())).collect();
        let system = manager.get_user_memory(EVAL_USER).unwrap();

        let order = |layers: LayerMode| -> Vec<String> {
            let query = Query {
                query_text: Some(
                    "Which person was registered for the Paris trade summit?".to_string(),
                ),
                max_results: 10,
                layers,
                ..Default::default()
            };
            system
                .read()
                .recall(&query)
                .map(|ms| {
                    ms.iter()
                        .map(|m| rev.get(&m.id.0).cloned().unwrap_or_else(|| m.id.0.to_string()))
                        .collect()
                })
                .unwrap_or_default()
        };

        std::env::set_var("SHODH_FUSION_V2", "1");
        let pf = order(LayerMode::PlusFacts);
        let full = order(LayerMode::Full);
        std::env::remove_var("SHODH_FUSION_V2");
        let _ = std::fs::remove_dir_all(&storage);

        let rank = |v: &Vec<String>, id: &str| v.iter().position(|i| i == id);
        eprintln!("V2 +facts: {pf:?}");
        eprintln!("V2 full  : {full:?}");
        eprintln!(
            "person rank  +facts={:?}  full={:?}",
            rank(&pf, "person"),
            rank(&full, "person")
        );
    }

    #[test]
    fn experience_type_recognises_known_strings() {
        assert!(matches!(
            experience_type_for("decision"),
            ExperienceType::Decision
        ));
        assert!(matches!(
            experience_type_for("Reference"),
            ExperienceType::Observation
        ));
        assert!(matches!(
            experience_type_for("event"),
            ExperienceType::Observation
        ));
        assert!(matches!(
            experience_type_for("nonsense"),
            ExperienceType::Observation
        ));
    }

    // -------------------- RH-12 (#272) -------------------------------------
    //
    // Run the full smoke suite with N=2 and verify (a) the report records
    // repeats=2, (b) per-repeat storage subdirs exist, (c) quality metrics
    // are bit-identical between the N=1 baseline run and the N=2 run.
    //
    // This test is the load-bearing assertion that RH-12 changed only the
    // measurement, not the metric — if N=1 and N=2 disagree on quality,
    // either the determinism guarantee broke OR `aggregate_layer` is no
    // longer a pure function of its inputs.
    //
    // Marked `#[ignore]` because it pays the ingest cost twice (~12 min on
    // a cold runner). Run via `cargo test --release -- --ignored
    // runner_repeats_2_produces_same_quality_as_repeats_1` before shipping
    // changes that touch the harness or scoring path.
    #[test]
    #[ignore = "expensive: runs the smoke suite twice (~12min). enable with --ignored before shipping harness changes."]
    fn runner_repeats_2_produces_same_quality_as_repeats_1() {
        let storage1 = unique_storage_dir("repeats1");
        let storage2 = unique_storage_dir("repeats2");

        let inputs1 = RunInputs {
            storage_path: storage1.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            repeats: 1,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };
        let inputs2 = RunInputs {
            storage_path: storage2.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            repeats: 2,
            layer_modes: vec![LayerMode::Full],
            age_days: 0.0,
        };

        let r1 = run_smoke_suite(&inputs1).expect("repeats=1 must succeed");
        let r2 = run_smoke_suite(&inputs2).expect("repeats=2 must succeed");

        assert_eq!(r1.repeats, 1);
        assert_eq!(r2.repeats, 2);

        // Quality metrics — bit-identical (RH-11 determinism).
        let f1 = r1.layers.get("full").expect("r1 full");
        let f2 = r2.layers.get("full").expect("r2 full");
        assert_eq!(f1.ndcg_at_10.to_bits(), f2.ndcg_at_10.to_bits(), "ndcg@10");
        assert_eq!(
            f1.recall_at_10.to_bits(),
            f2.recall_at_10.to_bits(),
            "recall@10"
        );
        assert_eq!(f1.mrr.to_bits(), f2.mrr.to_bits(), "mrr");
        assert_eq!(f1.p_at_1.to_bits(), f2.p_at_1.to_bits(), "p@1");
        assert_eq!(f1.map.to_bits(), f2.map.to_bits(), "map");

        // Per-repeat storage subdirs must have been created.
        assert!(
            storage2.join("repeat_0").exists(),
            "repeat_0 storage missing"
        );
        assert!(
            storage2.join("repeat_1").exists(),
            "repeat_1 storage missing"
        );

        // No infrastructure failures (rank-list divergence) on the
        // canonical fixtures — RH-11 guarantees byte-identical ranks.
        let infra_failures: Vec<_> = r2
            .failures
            .iter()
            .filter(|f| f.kind == "infrastructure")
            .collect();
        assert!(
            infra_failures.is_empty(),
            "no infrastructure failures expected on canonical fixtures, got {:?}",
            infra_failures
        );

        let _ = std::fs::remove_dir_all(&storage1);
        let _ = std::fs::remove_dir_all(&storage2);
    }

    /// RH-8 (#270): running the smoke suite with multiple `LayerMode`s
    /// emits one `layers` entry per mode, in pipeline order, and per-mode
    /// rank lists are byte-identical across repeats.
    ///
    /// Marked `#[ignore]` for the same reason as the other end-to-end
    /// runner tests — it pays the ingest cost once and the query loop six
    /// times. Run via `cargo test --release -- --ignored
    /// runner_layer_all_emits_six_modes_with_per_mode_determinism`
    /// before shipping changes that touch any layer gate.
    #[test]
    #[ignore = "expensive: runs the smoke suite with 6 modes (~6× query time). enable with --ignored before shipping layer-gate changes."]
    fn runner_layer_all_emits_six_modes_with_per_mode_determinism() {
        let storage = unique_storage_dir("layer-all");
        let inputs = RunInputs {
            storage_path: storage.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            // Two repeats so the per-mode determinism check actually runs.
            repeats: 2,
            layer_modes: LayerMode::ALL.to_vec(),
            age_days: 0.0,
        };
        let report = run_smoke_suite(&inputs).expect("layer-all run must succeed");
        let _ = std::fs::remove_dir_all(&storage);

        // Six modes, all present, in their canonical report-key form.
        for mode in LayerMode::ALL {
            assert!(
                report.layers.contains_key(mode.report_key()),
                "missing layer entry for {}",
                mode.report_key()
            );
        }
        assert_eq!(report.layers.len(), LayerMode::ALL.len());

        // Per-mode determinism: any divergence would have been recorded
        // as `infrastructure` failures by the runner.
        let infra: Vec<_> = report
            .failures
            .iter()
            .filter(|f| f.kind == "infrastructure")
            .collect();
        assert!(
            infra.is_empty(),
            "per-mode determinism violated: {:?}",
            infra
        );

        // Sanity: `full` ndcg@10 must be >= `vamana_only` ndcg@10 on the
        // canonical fixtures. This is the #270 acceptance invariant — if a
        // future scoring change ever inverts it, that's a real signal.
        let full = report.layers.get("full").expect("full layer present");
        let vamana = report
            .layers
            .get("vamana_only")
            .expect("vamana_only layer present");
        assert!(
            full.ndcg_at_10 + 1e-6 >= vamana.ndcg_at_10,
            "full ndcg@10 ({}) must be >= vamana_only ndcg@10 ({}) on canonical fixtures",
            full.ndcg_at_10,
            vamana.ndcg_at_10
        );
    }

    /// Direct unit test of the cross-repeat divergence detection logic.
    ///
    /// Builds two synthetic `OnePassResult`s with one diverging case, then
    /// runs the same comparison the public path uses. Cheap (no harness),
    /// runs on every CI build.
    #[test]
    fn synthetic_rank_divergence_surfaces_infrastructure_failure() {
        let pass_a_ranks = [
            CaseRankList {
                case_id: "smoke-001".to_string(),
                retrieved: vec!["doc-a".to_string(), "doc-b".to_string()],
            },
            CaseRankList {
                case_id: "smoke-002".to_string(),
                retrieved: vec!["doc-c".to_string(), "doc-d".to_string()],
            },
        ];
        let pass_b_ranks = [
            CaseRankList {
                case_id: "smoke-001".to_string(),
                retrieved: vec!["doc-a".to_string(), "doc-b".to_string()],
            },
            CaseRankList {
                // Divergence on smoke-002.
                case_id: "smoke-002".to_string(),
                retrieved: vec!["doc-d".to_string(), "doc-c".to_string()],
            },
        ];

        // Re-implement the same comparison loop the public path uses,
        // narrow to the assertion we care about. Keeps the test
        // independent of harness setup cost.
        let mut failures: Vec<Failure> = Vec::new();
        for (k, ref_rank) in pass_a_ranks.iter().enumerate() {
            let cur_rank = &pass_b_ranks[k];
            assert_eq!(ref_rank.case_id, cur_rank.case_id);
            if ref_rank.retrieved != cur_rank.retrieved {
                failures.push(Failure {
                    kind: "infrastructure".to_string(),
                    detail: format!(
                        "non-determinism: case {} rank list diverged",
                        ref_rank.case_id
                    ),
                });
            }
        }

        assert_eq!(failures.len(), 1, "exactly one divergence expected");
        assert_eq!(failures[0].kind, "infrastructure");
        assert!(
            failures[0].detail.contains("smoke-002"),
            "failure must name the diverging case: {}",
            failures[0].detail
        );
    }
}
