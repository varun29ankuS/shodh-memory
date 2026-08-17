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

use axum::extract::State;
use axum::response::Json;
use serde::Deserialize;

use crate::errors::{AppError, ValidationErrorExt};
use crate::handlers::state::MultiUserMemoryManager;
use crate::integrity::{self, IntegrityScrubReport, ScrubBudget};
use crate::validation;

pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Upper bound on a caller-supplied time budget.
///
/// Above the server's own request timeout the budget would never bind, and the
/// scrub would be killed mid-sweep and return nothing at all instead of an
/// honest partial report.
const MAX_DURATION_MS: u64 = 120_000;

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

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    // A user with no graph yet is not an error: the scrub reports the graph
    // column family as skipped, which forbids a `healthy` verdict rather than
    // quietly reporting health for half the data.
    let graph = state.get_user_graph(&req.user_id).ok();

    let user_id = req.user_id.clone();
    // Measured at ~13s on the largest live profile, dominated by iterating the
    // 660k non-memory keys sharing the memory default column family.
    // `crud::list_memories` is the precedent: full scans go on the blocking
    // pool, never on a runtime worker.
    let report = tokio::task::spawn_blocking(move || {
        // Take owned RocksDB handles and release both guards BEFORE sweeping.
        // Holding the graph read guard for thirteen seconds would stall every
        // graph writer (`handlers::graph` and `handlers::mif` take `.write()`),
        // and parking_lot prefers writers, so every reader queued behind them
        // would stall too. RocksDB handles are internally synchronised; the
        // guards are needed only to reach them.
        let memory_db = {
            let memory_guard = memory_sys.read();
            memory_guard.storage().raw_db().clone()
        };
        let graph_db = graph.map(|g| {
            let graph_guard = g.read();
            graph_guard.db_arc()
        });

        match graph_db.as_ref().and_then(|db| {
            db.cf_handle(crate::graph_memory::ENTITIES_CF_NAME)
                .map(|cf| (db, cf))
        }) {
            Some((gdb, cf)) => integrity::scrub_user(&user_id, &memory_db, Some((gdb, cf)), budget),
            None => integrity::scrub_user(&user_id, &memory_db, None, budget),
        }
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("integrity scrub task panicked: {e}")))?;

    Ok(Json(report))
}

#[cfg(test)]
mod tests {
    use crate::handlers::test_helpers::{post_json, send, send_typed, TestHarness, TEST_API_KEY};
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
}
