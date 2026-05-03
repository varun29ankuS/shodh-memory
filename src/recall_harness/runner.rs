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

use crate::memory::types::ExperienceType;
use crate::memory::{Experience, MemoryConfig, MemorySystem, Query};

use super::fixtures::{
    self, CorpusItem, SmokeCase, SmokeCategory, SMOKE_CASES_PATH, SMOKE_CORPUS_PATH,
};
use super::metrics::Metrics;
use super::report::{
    aggregate_category, aggregate_layer, CategoryReport, Failure, LayerReport, Report, SMOKE_K,
};

/// Embedder identifier emitted in the report. Matches the model wired into
/// `MemorySystem::new` today; bump when the embedder changes.
pub const EMBEDDER_ID: &str = "minilm-l6-v2";

/// Inputs to [`run_smoke_suite`].
#[derive(Debug, Clone)]
pub struct RunInputs {
    /// Storage directory for the harness's `MemorySystem`. The directory
    /// must be writeable; the runner does not delete it on completion so
    /// callers can inspect state after a failed run.
    pub storage_path: PathBuf,
    /// Optional path overrides; if `None` the canonical fixture paths are used.
    pub corpus_path: Option<PathBuf>,
    pub cases_path: Option<PathBuf>,
    /// Suite identifier echoed into the report. Currently fixed to `"smoke"`.
    pub suite: String,
    /// Git SHA of the working tree the run was produced from. Caller is
    /// responsible for resolution because the runner does not shell out.
    pub git_sha: String,
}

/// Run the smoke suite end-to-end and return a populated [`Report`].
///
/// On infrastructure errors (corpus parse failure, system init failure,
/// remember failure) the call returns `Err`. Per-case recall errors are
/// recorded as `Failure { kind = "case", .. }` entries so a single broken
/// query does not abort the whole run.
pub fn run_smoke_suite(inputs: &RunInputs) -> Result<Report> {
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

    let system = build_system(&inputs.storage_path)?;
    let id_map = ingest_corpus(&system, &corpus)?;

    let mut per_case = Vec::with_capacity(cases.len());
    let mut latencies_ms = Vec::with_capacity(cases.len());
    let mut by_category_cases: HashMap<SmokeCategory, Vec<Metrics>> = HashMap::new();
    let mut failures: Vec<Failure> = Vec::new();

    for case in &cases {
        let query = Query {
            query_text: Some(case.query.clone()),
            max_results: SMOKE_K,
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
                    detail: format!("recall failed for {}: {e:#}", case.id),
                });
                // Treat as zero-recall so aggregates keep their denominator.
                Vec::new()
            }
        };

        let retrieved_uuids: Vec<Uuid> = memories.iter().map(|m| m.id.0).collect();
        let relevance = build_relevance_map(case, &id_map);

        if relevance.is_empty() {
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

    let layer_report = aggregate_layer(&per_case, &latencies_ms);
    let mut layers: BTreeMap<String, LayerReport> = BTreeMap::new();
    layers.insert("full".to_string(), layer_report);

    let mut by_category: BTreeMap<String, CategoryReport> = BTreeMap::new();
    for cat in SmokeCategory::ALL {
        let cases_for_cat = by_category_cases.remove(&cat).unwrap_or_default();
        by_category.insert(
            category_name(cat).to_string(),
            aggregate_category(&cases_for_cat),
        );
    }

    Ok(Report {
        suite: inputs.suite.clone(),
        embedder: EMBEDDER_ID.to_string(),
        git_sha: inputs.git_sha.clone(),
        timestamp: chrono::Utc::now(),
        layers,
        by_category,
        case_count: cases.len(),
        failures,
    })
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
        };
        let report = run_smoke_suite(&inputs).expect("runner must succeed");
        let _ = std::fs::remove_dir_all(&storage);

        assert_eq!(report.case_count, 30);
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
}
