//! HTTP surface for the read-only integrity scrub ([`crate::integrity`]).
//!
//! Deliberately complementary to the existing diagnostics rather than
//! overlapping them:
//!
//! - `POST /api/index/verify` compares storage membership against the vector
//!   index. It loads records through `get_all()`, which silently drops anything
//!   that fails to decode — so the population this scrub exists to find is
//!   invisible to it by construction.
//! - `GET /api/audit/{user_id}` and `GET /api/memory/{id}/history` answer
//!   "who changed this record". This answers "is what we stored still what we
//!   wrote", which no audit trail can tell you: the July breakage changed no
//!   record at all, it changed the schema that reads them.

use std::collections::BTreeSet;

use axum::extract::{Query, State};
use axum::response::Json;
use serde::{Deserialize, Serialize};

use crate::errors::{AppError, ValidationErrorExt};
use crate::handlers::state::MultiUserMemoryManager;
use crate::integrity::{
    self, IntegrityScrubReport, LastScrub, LedgerSummary, ScrubBudget, ScrubSource, Verdict,
};
use crate::validation;

pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Upper bound on a caller-supplied time budget.
///
/// Held below `ServerConfig::request_timeout_secs` (default 60s) on purpose: a
/// budget above the request timeout could never bind, so the scrub would be
/// killed mid-sweep and return nothing at all instead of the honest partial
/// report the budget exists to produce.
const MAX_DURATION_MS: u64 = 55_000;

#[derive(Debug, Deserialize)]
pub struct ScrubRequest {
    pub user_id: String,
    /// Stop after this many records. Omit for a full sweep, which is the
    /// intended mode — see [`crate::integrity::scrub_user`] on why sampling is
    /// the wrong instrument here. A capped run reports `complete: false` and
    /// can never return a `healthy` verdict.
    pub max_records: Option<u64>,
    /// Wall-clock ceiling in milliseconds. Defaults to
    /// [`crate::integrity::DEFAULT_MAX_DURATION_MS`].
    pub max_duration_ms: Option<u64>,
}

/// POST /api/integrity/scrub - read-only wire-level integrity scrub over one
/// user's memories and graph nodes.
///
/// Classifies every record as clean, readable-only-via-a-legacy-generation,
/// undecodable, or decoding-into-implausible-values, and returns a verdict with
/// the numeric rule that produced it.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn scrub(
    State(state): State<AppState>,
    Json(req): Json<ScrubRequest>,
) -> Result<Json<IntegrityScrubReport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    if let Some(max) = req.max_records {
        validation::validate_limit(max as usize, "max_records")
            .map_validation_err("max_records")?;
    }
    if let Some(ms) = req.max_duration_ms {
        if ms == 0 || ms > MAX_DURATION_MS {
            return Err(AppError::InvalidInput {
                field: "max_duration_ms".to_string(),
                reason: format!("must be in 1..={MAX_DURATION_MS}"),
            });
        }
    }

    let budget = ScrubBudget {
        max_records: req.max_records,
        max_duration: Some(std::time::Duration::from_millis(
            req.max_duration_ms
                .unwrap_or(integrity::DEFAULT_MAX_DURATION_MS),
        )),
    };

    let user_id = req.user_id.clone();
    // Measured at ~13s on the largest live profile, dominated by iterating the
    // 660k non-memory keys sharing the memory default column family.
    // `crud::list_memories` is the precedent: full scans go on the blocking
    // pool, never on a runtime worker.
    //
    // The sweep itself lives on the manager rather than here so that the
    // scheduler and this handler run the SAME code and file into the SAME
    // ledger. A second copy of the handle-extraction dance would drift, and the
    // GET below would then be reporting on a scrub that is not quite the one
    // the scheduler runs.
    let manager = std::sync::Arc::clone(&state);
    let report = tokio::task::spawn_blocking(move || {
        manager.scrub_user_and_record(&user_id, budget, ScrubSource::OnDemand)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("integrity scrub task panicked: {e}")))?;

    Ok(Json(report))
}

/// Query for [`last_scrub`].
#[derive(Debug, Deserialize)]
pub struct LastScrubQuery {
    /// Restrict the answer to one profile. Omit for every profile on disk.
    pub user_id: Option<String>,
    /// Include the per-record findings. Off by default: on an unhealthy store
    /// this is up to `FINDINGS_PER_CLASS` records per class per profile, which
    /// is evidence for a human reading one profile, not payload for a poller.
    #[serde(default)]
    pub findings: bool,
}

/// One profile's last known result, flattened for reading.
#[derive(Debug, Serialize)]
pub struct LastScrubEntry {
    pub user_id: String,
    pub source: ScrubSource,
    pub recorded_at: chrono::DateTime<chrono::Utc>,
    /// How long ago the result was filed. The number that says whether the
    /// scheduler is alive; a verdict without it is unfalsifiable.
    pub age_secs: i64,
    /// `true` when the result is older than two scheduler intervals, i.e. at
    /// least one scheduled run did not happen. Always `true` when the scheduler
    /// is disabled and the only results are on-demand ones.
    pub stale: bool,
    pub verdict: Verdict,
    pub is_healthy: bool,
    pub complete: bool,
    pub stop_reason: Option<String>,
    pub skipped: Vec<String>,
    pub duration_ms: u64,
    pub memories: crate::integrity::ClassCounts,
    pub graph_nodes: crate::integrity::ClassCounts,
    pub verdict_rule: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub findings: Option<Vec<crate::integrity::Finding>>,
}

impl LastScrubEntry {
    fn from(entry: LastScrub, stale_after_secs: i64, with_findings: bool) -> Self {
        let age_secs = (chrono::Utc::now() - entry.recorded_at)
            .num_seconds()
            .max(0);
        let r = entry.report;
        Self {
            user_id: r.user_id,
            source: entry.source,
            recorded_at: entry.recorded_at,
            age_secs,
            stale: age_secs > stale_after_secs,
            verdict: r.verdict,
            is_healthy: r.is_healthy,
            complete: r.complete,
            stop_reason: r.stop_reason,
            skipped: r.skipped,
            duration_ms: r.duration_ms,
            memories: r.memories,
            graph_nodes: r.graph_nodes,
            verdict_rule: r.verdict_rule,
            findings: with_findings.then_some(r.findings),
        }
    }
}

/// What the scheduler is configured to do, returned alongside the results so a
/// reader can tell "clean" from "nothing has run".
#[derive(Debug, Serialize)]
pub struct SchedulerStatus {
    pub enabled: bool,
    pub interval_secs: u64,
    pub budget_ms: u64,
}

/// The answer to "what does the system currently believe about its own
/// storage", with no sweep started.
#[derive(Debug, Serialize)]
pub struct LastScrubResponse {
    pub scheduler: SchedulerStatus,
    /// The worst verdict on file. `None` means nothing has been scrubbed —
    /// which is NOT healthy, and is why this is an `Option` rather than
    /// defaulting to `Healthy`.
    pub worst_verdict: Option<Verdict>,
    pub summary: LedgerSummary,
    /// Profiles with a store on disk and no result at all. Absence rendered
    /// explicitly: an all-healthy `results` list next to a non-empty
    /// `never_scrubbed` is not a healthy system, and a reader that only looks
    /// at `results` would never learn that.
    pub never_scrubbed: Vec<String>,
    pub results: Vec<LastScrubEntry>,
}

/// GET /api/integrity/scrub - the last known verdict per profile, without
/// running anything.
///
/// # Why a separate verb rather than a cached POST
///
/// A scrub costs ~14s on the largest live profile. Folding "tell me the last
/// answer" into the endpoint that produces one would mean either paying that
/// on every poll or inventing a cache invalidation rule; a GET that is
/// explicitly a *read of the ledger* has neither problem, and makes the
/// distinction between "the system checked" and "somebody asked" visible in
/// the URL.
///
/// This endpoint is the reason the scheduler is not another write with no
/// reader. The learning ledger wrote for days with nothing rendering it;
/// `CF_AUDIT` had twenty-three call sites and a dead reader; context reminders
/// still have no consumer. A scheduled scrub nobody can query would have been
/// the next one.
#[tracing::instrument(skip(state))]
pub async fn last_scrub(
    State(state): State<AppState>,
    Query(q): Query<LastScrubQuery>,
) -> Result<Json<LastScrubResponse>, AppError> {
    if let Some(user_id) = &q.user_id {
        validation::validate_user_id(user_id).map_validation_err("user_id")?;
    }

    let interval = state.server_config().integrity_scrub_interval_secs;
    // Two intervals: one missed tick is a slow sweep or a restart, two is a
    // scheduler that is not running. With the scheduler off there is no tick to
    // miss, so every result is stale by definition and says so.
    let stale_after = if interval == 0 {
        -1
    } else {
        (interval as i64).saturating_mul(2)
    };

    let ledger = state.scrub_ledger();
    let mut results: Vec<LastScrubEntry> = ledger
        .all()
        .into_iter()
        .filter(|e| q.user_id.as_ref().is_none_or(|u| *u == e.report.user_id))
        .map(|e| LastScrubEntry::from(e, stale_after, q.findings))
        .collect();
    results.sort_by(|a, b| a.user_id.cmp(&b.user_id));

    let have: BTreeSet<String> = results.iter().map(|e| e.user_id.clone()).collect();
    let never_scrubbed: Vec<String> = state
        .profiles_on_disk()
        .into_iter()
        .filter(|u| !have.contains(u))
        .filter(|u| q.user_id.as_ref().is_none_or(|q| q == u))
        .collect();

    let summary = ledger.summary();
    crate::metrics::publish_integrity_scrub_metrics(&summary);
    crate::metrics::INTEGRITY_USERS_NEVER_SCRUBBED.set(never_scrubbed.len() as i64);

    Ok(Json(LastScrubResponse {
        scheduler: SchedulerStatus {
            enabled: interval > 0,
            interval_secs: interval,
            budget_ms: integrity::SCHEDULED_MAX_DURATION_MS,
        },
        worst_verdict: ledger.worst_verdict(),
        summary,
        never_scrubbed,
        results,
    }))
}

#[cfg(test)]
mod tests {
    use crate::handlers::test_helpers::{
        get, post_json, send, send_typed, TestHarness, TEST_API_KEY,
    };
    use crate::integrity::{IntegrityScrubReport, Verdict};
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use serde_json::json;

    /// The route exists, is authenticated, and returns the report shape.
    ///
    /// Route registration compiles whether or not the path is actually
    /// mounted; only a request through the real router proves it.
    #[tokio::test]
    async fn scrub_route_is_reachable_and_returns_a_verdict() {
        let harness = TestHarness::new();
        let (status, report): (StatusCode, IntegrityScrubReport) = send_typed(
            harness.router(),
            post_json("/api/integrity/scrub", &json!({ "user_id": "scrub-test" })),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(report.user_id, "scrub-test");
        // A fresh user has an empty store and a real graph, so the sweep is
        // complete and finds nothing.
        assert!(report.complete, "stop_reason: {:?}", report.stop_reason);
        assert!(report.skipped.is_empty());
        assert_eq!(report.verdict, Verdict::Healthy);
        assert!(report.is_healthy);
        assert!(
            !report.checks_applied.is_empty(),
            "the report must name what it checked"
        );
        assert!(
            !report.verdict_rule.is_empty(),
            "the judgement must travel with the numbers"
        );
    }

    #[tokio::test]
    async fn scrub_requires_authentication() {
        let harness = TestHarness::new();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/api/integrity/scrub")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"user_id":"scrub-test"}"#))
            .unwrap();
        let (status, _) = send(harness.router(), req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        // Guard against the assertion above passing for the wrong reason.
        assert!(!TEST_API_KEY.is_empty());
    }

    #[tokio::test]
    async fn scrub_rejects_an_invalid_user_id() {
        let harness = TestHarness::new();
        let (status, body) = send(
            harness.router(),
            post_json("/api/integrity/scrub", &json!({ "user_id": "../../etc" })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["code"], "INVALID_INPUT");
    }

    #[tokio::test]
    async fn scrub_rejects_an_out_of_range_duration_budget() {
        let harness = TestHarness::new();
        let (status, body) = send(
            harness.router(),
            post_json(
                "/api/integrity/scrub",
                &json!({ "user_id": "scrub-test", "max_duration_ms": 0 }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["code"], "INVALID_INPUT");
    }

    /// A capped sweep must come back over the wire saying it was capped, and
    /// must not claim health.
    #[tokio::test]
    async fn a_capped_sweep_reports_itself_over_http() {
        let harness = TestHarness::new();
        let (status, report): (StatusCode, IntegrityScrubReport) = send_typed(
            harness.router(),
            post_json(
                "/api/integrity/scrub",
                &json!({ "user_id": "scrub-test", "max_duration_ms": 1 }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // Either it finished inside 1ms on an empty store (legitimately
        // healthy), or it was cut short -- in which case it must say so and
        // must not report health.
        if !report.complete {
            assert!(report.stop_reason.is_some());
            assert_ne!(report.verdict, Verdict::Healthy);
            assert!(!report.is_healthy);
        }
    }

    // -----------------------------------------------------------------------
    // THE READER
    //
    // A scheduled scrub nobody can query is the same write-then-never-read
    // failure that runs through this codebase. These pin that it is readable,
    // that reading it starts nothing, and that absence is visible.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn a_scrub_files_its_result_where_the_reader_can_find_it() {
        let harness = TestHarness::new();

        let (status, _): (StatusCode, IntegrityScrubReport) = send_typed(
            harness.router(),
            post_json("/api/integrity/scrub", &json!({ "user_id": "scrub-test" })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (status, body) = send(harness.router(), get("/api/integrity/scrub")).await;
        assert_eq!(status, StatusCode::OK);

        let results = body["results"].as_array().expect("results array");
        let mine = results
            .iter()
            .find(|r| r["user_id"] == "scrub-test")
            .expect("the run that just happened must be readable afterwards");
        assert_eq!(
            mine["source"], "on_demand",
            "the reader must say who asked, so a scheduled verdict is \
             distinguishable from one a human triggered"
        );
        assert!(mine["verdict"].is_string());
        assert!(
            mine["recorded_at"].is_string(),
            "a verdict with no timestamp cannot be told apart from a stale one"
        );
        assert!(mine["age_secs"].is_number());
        assert!(
            body["worst_verdict"].is_string(),
            "with a result on file the worst verdict is a verdict"
        );
    }

    /// Reading must not start a sweep.
    ///
    /// The whole point of a separate GET is that polling it is free. If it ran
    /// a scrub it would be a ~14s request on the largest live profile, and a
    /// dashboard refreshing every 30s would keep one blocking thread busy
    /// forever.
    #[tokio::test]
    async fn reading_the_last_scrub_starts_nothing() {
        let harness = TestHarness::new();

        let (status, body) = send(harness.router(), get("/api/integrity/scrub")).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["results"].as_array().map(|a| a.len()),
            Some(0),
            "a GET before any scrub must report that none has run, not run one"
        );
        assert!(
            body["worst_verdict"].is_null(),
            "nothing scrubbed is NOT healthy; the reader must return null so a \
             consumer cannot mistake absence for a pass"
        );
        assert_eq!(body["summary"]["users_with_a_result"], 0);
    }

    /// The scheduler's configuration travels with the results.
    ///
    /// Without it, "everything healthy, one result, filed four days ago" reads
    /// as a healthy system instead of as a scheduler that stopped.
    #[tokio::test]
    async fn the_reader_states_whether_anything_is_scheduled() {
        let harness = TestHarness::new();
        let (status, body) = send(harness.router(), get("/api/integrity/scrub")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["scheduler"]["enabled"].is_boolean());
        assert!(body["scheduler"]["interval_secs"].is_number());
        assert_eq!(
            body["scheduler"]["budget_ms"],
            crate::integrity::SCHEDULED_MAX_DURATION_MS
        );
    }

    #[tokio::test]
    async fn reading_the_last_scrub_requires_authentication() {
        let harness = TestHarness::new();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/api/integrity/scrub")
            .body(Body::empty())
            .unwrap();
        let (status, _) = send(harness.router(), req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(!TEST_API_KEY.is_empty());
    }

    #[tokio::test]
    async fn reading_rejects_an_invalid_user_id() {
        let harness = TestHarness::new();
        let (status, body) = send(
            harness.router(),
            get("/api/integrity/scrub?user_id=../../etc"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["code"], "INVALID_INPUT");
    }

    /// Findings are opt-in.
    ///
    /// On an unhealthy store they are up to `FINDINGS_PER_CLASS` records per
    /// class per profile — evidence for a human reading one profile, not
    /// payload for a poller.
    #[tokio::test]
    async fn findings_are_off_by_default_and_available_on_request() {
        let harness = TestHarness::new();
        let (status, _): (StatusCode, IntegrityScrubReport) = send_typed(
            harness.router(),
            post_json("/api/integrity/scrub", &json!({ "user_id": "scrub-test" })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, plain) = send(harness.router(), get("/api/integrity/scrub")).await;
        assert!(plain["results"][0]["findings"].is_null());

        let (_, detailed) = send(harness.router(), get("/api/integrity/scrub?findings=true")).await;
        assert!(
            detailed["results"][0]["findings"].is_array(),
            "the evidence must be reachable, or the verdict is unfalsifiable"
        );
    }
}
