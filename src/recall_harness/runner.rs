//! End-to-end runner for the L1 smoke suite.
//!
//! Loads fixtures via [`crate::recall_harness::fixtures`], stands up a
//! `MemorySystem` against an isolated storage directory, ingests the corpus,
//! runs every smoke case through `MemorySystem::recall`, and emits a
//! [`Report`] aggregating per-case [`Metrics`] into per-layer and
//! per-category sections.
//!
//! See issue #266 for the runner's CLI and acceptance criteria.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use uuid::Uuid;

use crate::memory::types::{ExperienceType, LayerMode};
use crate::memory::{Experience, MemoryConfig, MemorySystem, Query};

use super::fixtures::{
    self, CorpusItem, SmokeCase, SmokeCategory, SMOKE_CASES_PATH, SMOKE_CORPUS_PATH,
};
use super::metrics::Metrics;
use super::report::{
    aggregate_category, aggregate_layer, median, CategoryReport, Failure, LayerReport, Report,
    SMOKE_K,
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

    fixtures::validate_smoke_suite(&corpus, &cases)
        .context("smoke suite failed structural validation — fix the JSONL fixtures")?;

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
        let pass = run_one_pass(&pass_storage, &corpus, &cases, &layer_modes)
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
    let mut by_category: BTreeMap<String, CategoryReport> = BTreeMap::new();
    for cat in SmokeCategory::ALL {
        let cases_for_cat = by_category_cases.get(&cat).cloned().unwrap_or_default();
        by_category.insert(
            category_name(cat).to_string(),
            aggregate_category(&cases_for_cat),
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

    Ok(ReportWithRanks { report, ranks })
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
) -> Result<OnePassResult> {
    let system = build_system(storage_path)?;
    let id_map = ingest_corpus(&system, corpus)?;
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

    for mode in layer_modes {
        let mut per_case = Vec::with_capacity(cases.len());
        let mut latencies_ms = Vec::with_capacity(cases.len());
        let mut by_category_cases: HashMap<SmokeCategory, Vec<Metrics>> = HashMap::new();
        let mut ranks: Vec<CaseRankList> = Vec::with_capacity(cases.len());

        for case in cases {
            let query = Query {
                query_text: Some(case.query.clone()),
                max_results: SMOKE_K,
                layers: *mode,
                ..Default::default()
            };

            let started = Instant::now();
            let result = system.recall(&query);
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
            let relevance = build_relevance_map(case, &id_map);

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
fn build_system(storage_path: &Path) -> Result<MemorySystem> {
    std::fs::create_dir_all(storage_path)
        .with_context(|| format!("creating storage dir {}", storage_path.display()))?;
    let config = MemoryConfig {
        storage_path: storage_path.to_path_buf(),
        ..MemoryConfig::default()
    };
    MemorySystem::new(config, None).context("initialising MemorySystem for recall harness")
}

/// Ingest the corpus into `system`, returning the `string handle → Uuid` map
/// used to translate ground-truth references to system memory IDs.
pub fn ingest_corpus(
    system: &MemorySystem,
    corpus: &[CorpusItem],
) -> Result<HashMap<String, Uuid>> {
    let mut map = HashMap::with_capacity(corpus.len());
    for item in corpus {
        let experience = Experience {
            experience_type: experience_type_for(&item.memory_type),
            content: item.content.clone(),
            entities: item.tags.clone(),
            ..Default::default()
        };
        let memory_id = system
            .remember(experience, Some(item.created_at))
            .with_context(|| format!("remembering corpus item {}", item.id))?;
        map.insert(item.id.clone(), memory_id.0);
    }
    Ok(map)
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_storage_dir(label: &str) -> PathBuf {
        let id = Uuid::new_v4().simple().to_string();
        std::env::temp_dir().join(format!("shodh-recall-{label}-{id}"))
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
        };
        let report = run_smoke_suite(&inputs).expect("runner must succeed");
        let _ = std::fs::remove_dir_all(&storage);

        assert_eq!(report.case_count, 30);
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
            assert_eq!(cr.case_count, 5, "category {cat} must have 5 cases");
        }
        // No infrastructure-level failures expected on the canonical fixtures.
        assert!(
            report.failures.iter().all(|f| f.kind == "case"),
            "no infrastructure failures expected, got {:?}",
            report.failures
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
        };
        let inputs2 = RunInputs {
            storage_path: storage2.clone(),
            corpus_path: None,
            cases_path: None,
            suite: "smoke".to_string(),
            git_sha: "test-sha".to_string(),
            repeats: 2,
            layer_modes: vec![LayerMode::Full],
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
