//! Audit trail read API.
//!
//! The server has audited memory mutations for a long time — roughly two dozen
//! `log_event` call sites across crud, todos, recall, search, mif and
//! consolidation write an [`AuditEvent`] into the dedicated `audit` column
//! family, keyed `{user_id}:{timestamp_nanos:020}`. Until this module existed
//! there was no way to read any of it over HTTP: the trail surfaced only as
//! aggregated `tools_used` counts inside the session digest
//! (`sessions::build_active_digest_with_budget`), which reports how many times
//! each event type fired and discards every individual event. A trail that
//! cannot be read back is not an audit trail.
//!
//! # What this endpoint deliberately does not have
//!
//! **An actor.** Every entry is scoped by `user_id` — the memory namespace —
//! and nothing else. Caller identity does not exist at the `log_event` call
//! sites, and it does not exist anywhere upstream of them either:
//!
//! - `auth::auth_middleware` validates the presented key with
//!   `auth::validate_api_key`, which returns `Result<(), AuthError>`. It
//!   deliberately compares against every configured key without breaking early
//!   (constant-time), so it does not even learn *which* key matched, let alone
//!   who presented it.
//! - The middleware inserts nothing into the request extensions, so no
//!   principal is propagated to any handler.
//! - Configured keys are opaque secrets from a comma-separated
//!   `SHODH_API_KEYS`; they carry no name, owner or role to attribute to.
//! - The dashboard makes this structural rather than incidental: `shodh-front`
//!   injects one shared `SHODH_API_KEY` on behalf of every browser session, so
//!   all UI-originated writes arrive under a single identity no matter who is
//!   at the keyboard.
//!
//! An `actor` field would therefore hold the same constant on every row. A
//! column that always says the same thing is worse than no column — it reads
//! like attribution and provides none. Recording an actor requires giving the
//! auth layer a notion of principal first; that is a change to authentication,
//! not to audit, and it is not made here.
//!
//! **Immutability.** Entries are deleted on a timer. See
//! `MultiUserMemoryManagerRotationHelper::rotate_user_audit_logs` in
//! `handlers::state` and `ServerConfig::audit_retention_days` /
//! `ServerConfig::audit_max_entries_per_user`. That
//! retention policy is unchanged by this module — this is a read API, and what
//! it can return is bounded by what rotation has left behind.

use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};

use super::router::AppState;
use super::types::AuditEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::validation;

/// Default page size when the caller does not ask for one.
const DEFAULT_AUDIT_LIMIT: usize = 100;

/// Query parameters for `GET /api/audit/{user_id}`.
#[derive(Debug, Deserialize)]
pub struct AuditQuery {
    /// Max entries to return. Defaults to [`DEFAULT_AUDIT_LIMIT`]; rejected
    /// above [`validation::MAX_LIMIT`] rather than silently clamped, so a
    /// caller asking for more than the server will give is told so.
    pub limit: Option<usize>,
    /// Entries to skip, counting back from the newest. Defaults to 0.
    pub offset: Option<usize>,
}

/// Response for `GET /api/audit/{user_id}`.
#[derive(Debug, Serialize)]
pub struct AuditResponse {
    /// One page of the trail, ORDERED NEWEST FIRST (`timestamp` descending).
    ///
    /// Each entry is the stored [`AuditEvent`] verbatim — no field is
    /// synthesised for presentation.
    pub events: Vec<AuditEvent>,
    /// Every audit entry currently stored for this user, independent of
    /// `limit`/`offset`. This is a true count, not a value capped by the page
    /// size: compare `events.len() + offset` against it to know whether more
    /// pages remain.
    ///
    /// It counts what survives rotation, which is not everything that was ever
    /// recorded — entries older than `audit_retention_days`, and entries beyond
    /// the newest `audit_max_entries_per_user`, have been deleted.
    pub total: usize,
}

/// GET /api/audit/{user_id} - Read a user's audit trail, newest first.
///
/// Query params: `?limit=100&offset=0`
///
/// `limit` defaults to [`DEFAULT_AUDIT_LIMIT`] and must be in
/// `1..=validation::MAX_LIMIT`; `offset` defaults to 0. Use `total` in the
/// response to page through the trail (offset += limit).
#[tracing::instrument(skip(state), fields(user_id = %user_id))]
pub async fn get_audit_trail(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<AuditQuery>,
) -> Result<Json<AuditResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let limit = query.limit.unwrap_or(DEFAULT_AUDIT_LIMIT);
    validation::validate_limit(limit, "limit").map_validation_err("limit")?;
    let offset = query.offset.unwrap_or(0);

    // The scan touches every key for this user, so it runs off the async
    // runtime — same treatment the other whole-namespace reads get.
    let (events, total) = {
        let state = state.clone();
        let user_id = user_id.clone();
        tokio::task::spawn_blocking(move || state.get_audit_page(&user_id, offset, limit))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    Ok(Json(AuditResponse { events, total }))
}

#[cfg(test)]
mod tests {
    use crate::handlers::test_helpers::{self, TestHarness};
    use axum::http::StatusCode;

    /// Drive real audit events through the same path production uses:
    /// `log_event` is what every mutating handler calls.
    ///
    /// Two things have to settle before the trail can be asserted on, and both
    /// are properties of `log_event` rather than of the test:
    ///
    /// - It persists on `spawn_blocking`, so the RocksDB write has not
    ///   necessarily happened when the call returns. The helper waits for the
    ///   entries to actually land instead of assuming they have.
    /// - The key is `{user_id}:{timestamp_nanos:020}`, so two events sharing a
    ///   nanosecond share a key and the second overwrites the first. The system
    ///   clock's resolution is coarser than a nanosecond (markedly so on
    ///   Windows), so events are spaced far enough apart to guarantee distinct
    ///   keys — otherwise the seed itself would silently lose entries and the
    ///   assertions below would be testing the wrong thing.
    async fn seed(harness: &TestHarness, user_id: &str, n: usize) {
        for i in 0..n {
            harness.manager.log_event(
                user_id,
                "TEST_EVENT",
                &format!("memory-{i}"),
                &format!("detail {i}"),
            );
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }

        // Wait for the background writes to land rather than racing them.
        for _ in 0..500 {
            if harness.manager.get_audit_page(user_id, 0, 0).1 >= n {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            harness.manager.get_audit_page(user_id, 0, 0).1,
            n,
            "seed must produce exactly {n} distinct audit entries"
        );
    }

    /// The defect this module fixes: the trail was written but had no HTTP
    /// route, so `GET /api/audit/{user_id}` returned 404 and the only way to
    /// see any of it was aggregated `tools_used` counts in the session digest.
    #[tokio::test]
    async fn audit_trail_is_reachable_over_http() {
        let harness = TestHarness::new();
        let user_id = "audit-reachable-user";
        seed(&harness, user_id, 3).await;

        let req = test_helpers::get(&format!("/api/audit/{user_id}"));
        let (status, body) = test_helpers::send(harness.router(), req).await;

        assert_eq!(
            status,
            StatusCode::OK,
            "the audit trail must be readable over HTTP"
        );
        let events = body["events"].as_array().expect("events array");
        assert_eq!(events.len(), 3);
        assert_eq!(body["total"], 3u64);
        // The stored shape, verbatim.
        assert_eq!(events[0]["event_type"], "TEST_EVENT");
        assert!(events[0]["memory_id"].is_string());
        assert!(events[0]["details"].is_string());
        assert!(events[0]["timestamp"].is_string());
    }

    /// Newest first, and `total` is the true stored count rather than the page
    /// size — the property that keeps paging honest.
    #[tokio::test]
    async fn audit_page_is_newest_first_with_true_total() {
        let harness = TestHarness::new();
        let user_id = "audit-order-user";
        seed(&harness, user_id, 25).await;

        let req = test_helpers::get(&format!("/api/audit/{user_id}?limit=10"));
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);

        let events = body["events"].as_array().unwrap();
        assert_eq!(events.len(), 10, "limit must be honoured");
        assert_eq!(
            body["total"], 25u64,
            "`total` must be every stored entry, not the page size"
        );

        let stamps: Vec<&str> = events
            .iter()
            .map(|e| e["timestamp"].as_str().unwrap())
            .collect();
        let mut expected = stamps.clone();
        expected.sort_unstable();
        expected.reverse();
        assert_eq!(stamps, expected, "the page must be newest first");

        // The newest page must hold the newest events, not an arbitrary 10.
        assert_eq!(events[0]["memory_id"], "memory-24");
        assert_eq!(events[9]["memory_id"], "memory-15");
    }

    /// `offset` pages backwards through the trail without gaps or repeats.
    #[tokio::test]
    async fn audit_offset_pages_without_gaps() {
        let harness = TestHarness::new();
        let user_id = "audit-offset-user";
        seed(&harness, user_id, 25).await;

        let req = test_helpers::get(&format!("/api/audit/{user_id}?limit=10&offset=10"));
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);

        let events = body["events"].as_array().unwrap();
        assert_eq!(events.len(), 10);
        assert_eq!(body["total"], 25u64, "`total` must not move with `offset`");
        assert_eq!(events[0]["memory_id"], "memory-14");
        assert_eq!(events[9]["memory_id"], "memory-5");

        // The tail is short, and `total` still reports the whole trail.
        let req = test_helpers::get(&format!("/api/audit/{user_id}?limit=10&offset=20"));
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["events"].as_array().unwrap().len(), 5);
        assert_eq!(body["total"], 25u64);
    }

    /// One user's trail must never leak into another's page or count.
    #[tokio::test]
    async fn audit_is_scoped_to_one_user() {
        let harness = TestHarness::new();
        seed(&harness, "audit-tenant-a", 4).await;
        seed(&harness, "audit-tenant-b", 7).await;

        let req = test_helpers::get("/api/audit/audit-tenant-a");
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total"], 4u64);
        assert_eq!(body["events"].as_array().unwrap().len(), 4);
    }

    /// The route is authenticated, like every other memory-scoped read.
    #[tokio::test]
    async fn audit_requires_authentication() {
        let harness = TestHarness::new();
        let req = test_helpers::get_unauthenticated("/api/audit/audit-auth-user");
        let (status, _) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    /// A limit past `MAX_LIMIT` is rejected, not silently clamped.
    #[tokio::test]
    async fn audit_rejects_oversized_limit() {
        let harness = TestHarness::new();
        let req = test_helpers::get(&format!(
            "/api/audit/audit-limit-user?limit={}",
            crate::validation::MAX_LIMIT + 1
        ));
        let (status, _) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let req = test_helpers::get("/api/audit/audit-limit-user?limit=0");
        let (status, _) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// An empty trail is an empty page, not an error.
    #[tokio::test]
    async fn audit_empty_trail_is_ok() {
        let harness = TestHarness::new();
        let req = test_helpers::get("/api/audit/audit-empty-user");
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total"], 0u64);
        assert!(body["events"].as_array().unwrap().is_empty());
    }
}
