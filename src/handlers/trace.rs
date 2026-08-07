//! Witnessed-operation capture middleware (traceability slice 1, Task 3).
//!
//! One choke point wraps every protected route: the middleware inserts an
//! [`OpTrace`] extension, the handler enriches it with body-derived identity
//! and evidence, and after the response is produced exactly one
//! [`OpRecordDraft`] is appended to the per-user operation log
//! (spec `docs/superpowers/specs/2026-07-30-agent-traceability-design.md` §3.1).
//!
//! Capture policy (audit amendments 9–13):
//! - Mounted INSIDE `build_protected_routes` so both transports — HTTP (with
//!   auth) and local IPC (`server.rs`, no auth layer) — are covered
//!   structurally.
//! - Excluded: `/api/trace/*` (capture must not observe itself), `/metrics`
//!   (a scrape is not an agent op), and the four SSE/WS streaming routes
//!   (long-lived connections are not discrete ops; their capture story is a
//!   future slice).
//! - Identity-less requests: some protected routes carry no user identity by
//!   construction (see [`NO_IDENTITY_EXACT`]/[`NO_IDENTITY_SUBTREES`]); for
//!   those, and for any request whose handler never set an identity, the
//!   record is DROPPED and the
//!   [`crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL`] counter records the
//!   reason. We never append under a synthetic user: creating a user store on
//!   miss costs up to ~750 ms of retry sleeps and materializes junk user dirs.
//! - Capture failure never fails the operation and is never silent: counter
//!   increment + best-effort `oplog_mark_incomplete`.

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::handlers::state::MultiUserMemoryManager;
use crate::memory::oplog::{OpRecordDraft, ATTESTATION_WITNESSED};

/// Application state type alias (matches the rest of `handlers/`).
pub type AppState = Arc<MultiUserMemoryManager>;

/// Request extension enriched by handlers with body-derived identity and
/// evidence. The middleware owns the record's envelope (op, timing, outcome);
/// handlers own what only they can see (who asked, what memories were touched).
#[derive(Clone, Default, Debug)]
pub struct OpTrace(pub Arc<parking_lot::Mutex<OpTraceInner>>);

#[derive(Default, Debug)]
pub struct OpTraceInner {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub evidence_refs: Vec<String>,
    pub payload_summary: Option<String>,
}

impl OpTrace {
    /// Handler-side enrichment helper: set identity once, append evidence ids.
    pub fn set_identity(&self, user_id: &str, session_id: Option<&str>) {
        let mut inner = self.0.lock();
        inner.user_id = Some(user_id.to_string());
        if let Some(sid) = session_id {
            inner.session_id = Some(sid.to_string());
        }
    }

    pub fn push_evidence<I: IntoIterator<Item = String>>(&self, ids: I) {
        self.0.lock().evidence_refs.extend(ids);
    }

    pub fn set_summary(&self, summary: String) {
        self.0.lock().payload_summary = Some(summary);
    }
}

/// Never-captured paths. Exact `/api/trace` plus the `/api/trace/` subtree
/// cover the read API and the reported-event ingest (Task 4) — capture
/// observing capture would self-amplify. `/metrics` is matched EXACTLY (a
/// scrape is not an agent op; a future `/metricsz` route would NOT be silently
/// excluded — review M1).
const EXCLUDED_SUBTREES: &[&str] = &["/api/trace/"];
const EXCLUDED_EXACT: &[&str] = &[
    "/api/trace",
    "/metrics",
    // Long-lived streaming routes (SSE/WS) — connections, not discrete ops.
    "/api/context/monitor",
    "/api/events/sse",
    "/api/events",
    "/api/stream",
];

/// Protected routes that carry NO user identity by construction, split by
/// match semantics so the set names EXACTLY what the slice-1 audit verified
/// (review C1 — a prefix match on `/api/users` would also swallow
/// `DELETE /api/users/{user_id}`, the erasure op an audit log most needs to
/// see): `users.rs:71` (`State`-only list), `mif.rs:311` (no arguments), and
/// the `/api/ab/*` family (`ab_testing.rs:81`, `State`-only). Task 5's
/// completeness sweep consumes both constants (audit amendment 18).
pub const NO_IDENTITY_EXACT: &[&str] = &["/api/users", "/api/mif/adapters"];
pub const NO_IDENTITY_SUBTREES: &[&str] = &["/api/ab/"];

fn is_excluded(path: &str) -> bool {
    EXCLUDED_EXACT.iter().any(|e| path == *e)
        || EXCLUDED_SUBTREES.iter().any(|p| path.starts_with(p))
}

fn is_known_no_identity(path: &str) -> bool {
    NO_IDENTITY_EXACT.iter().any(|e| path == *e)
        || NO_IDENTITY_SUBTREES.iter().any(|p| path.starts_with(p))
}

/// Derive the op name from method + ROUTE TEMPLATE (never the raw path).
///
/// Using `MatchedPath` (review C3/I4 fix) gives every op a bounded domain and
/// keeps user-controlled path segments out of the permanent record:
/// `PUT /api/memory/{memory_id}` → `put:memory:{memory_id}`,
/// `DELETE /api/memory/{memory_id}` → `delete:memory:{memory_id}` — verbs are
/// ALWAYS prefixed, so an update and a deletion are never byte-identical in
/// the log. The concrete resource ids belong in `evidence_refs`, not in `op`.
fn op_name(method: &axum::http::Method, route_template: &str) -> String {
    let tail = route_template
        .strip_prefix("/api/")
        .unwrap_or(route_template)
        .replace('/', ":");
    format!("{}:{tail}", method.as_str().to_ascii_lowercase())
}

/// The capture choke point. Mounted inside `build_protected_routes` — see the
/// layer-order note at the mount site.
pub async fn capture_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    if is_excluded(&path) {
        return next.run(req).await;
    }
    let method = req.method().clone();
    // Route template BEFORE the handler runs (review C3/I4): MatchedPath is
    // inserted by the router during matching and is absent for fallback
    // requests — its absence doubles as the 404/405 detector (review I1).
    let route_template = req
        .extensions()
        .get::<axum::extract::MatchedPath>()
        .map(|m| m.as_str().to_string());

    let trace = OpTrace::default();
    let mut req = req;
    req.extensions_mut().insert(trace.clone());

    let response = next.run(req).await;
    let status = response.status();

    // Routing misses are NOT agent ops — an authenticated scanner or a
    // misconfigured client must not mint permanent records or pollute the
    // failure counter (review I1; the dead /api/record hook endpoint was
    // doing exactly this on every fire). A routing miss never carries a
    // route template (it goes through axum's fallback), so the template's
    // absence is the entire detector. A MATCHED route that returns 404
    // (missing memory id, …) stays captured: a failed access attempt is
    // exactly what an audit log must witness (re-review round 1 — a status
    // check here would let callers vanish ops by referencing absent ids).
    let Some(route_template) = route_template else {
        return response;
    };
    // 405: the path matched (template present) but no handler ran, so
    // identity can never be set — skip before it counts as unenriched.
    if status == axum::http::StatusCode::METHOD_NOT_ALLOWED {
        return response;
    }

    // Post-handler: build and append the record. Everything below is
    // best-effort by contract — the response is returned unchanged no matter
    // what happens here.
    let inner = trace.0.lock();
    let Some(user_id) = inner.user_id.clone() else {
        // No identity: known-anonymous routes are an accepted drop; anything
        // else is an un-enriched route — counted under its own reason label
        // and logged at debug (review I2: ~150 routes are not yet enriched;
        // per-request warn spam would bury the alertable signals).
        if !is_known_no_identity(&path) {
            crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL
                .with_label_values(&["unenriched_route"])
                .inc();
            tracing::debug!(path = %path, "trace capture skipped: route not yet enriched");
        }
        drop(inner);
        return response;
    };

    // Session fallback: the store the rest of the codebase already uses —
    // never an invented namespace (audit amendment 12). A handler-supplied
    // session id that fails validation is NOT trusted to address the log
    // (review C2: a malformed id would be rejected at append time, silently
    // suppressing the record while the caller sees 200) — it falls back to
    // the session store, counted under its own reason.
    let session_id = match inner.session_id.clone() {
        Some(sid) if crate::validation::validate_session_id(&sid).is_ok() => sid,
        Some(bad) => {
            crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL
                .with_label_values(&["invalid_session_id"])
                .inc();
            tracing::warn!(user_id = %user_id, session_id = %bad,
                "trace capture: invalid handler-supplied session id — using session-store fallback");
            state
                .session_store()
                .get_or_create_session(&user_id)
                .to_string()
        }
        None => state
            .session_store()
            .get_or_create_session(&user_id)
            .to_string(),
    };
    let evidence_refs = inner.evidence_refs.clone();
    let payload_summary = inner.payload_summary.clone().unwrap_or_default();
    drop(inner);

    // Cache-only lookup (audit amendment 11): the handler that just ran has
    // already materialized the user's MemorySystem on any op that touches
    // memory; if the user is not cached, we DROP rather than construct a
    // store from the capture path.
    let Some(system) = state.cached_user_memory(&user_id) else {
        crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL
            .with_label_values(&["uncached_user"])
            .inc();
        tracing::warn!(user_id = %user_id, path = %path, "trace capture dropped: user not cached");
        return response;
    };

    let draft = OpRecordDraft {
        ts: chrono::Utc::now(),
        session_id: session_id.clone(),
        user_id: user_id.clone(),
        op: op_name(&method, &route_template),
        attestation: ATTESTATION_WITNESSED.to_string(),
        payload_summary,
        evidence_refs,
        outcome: if status.is_success() {
            "ok".to_string()
        } else {
            format!("error:{}", status.as_u16())
        },
        reported_ts: None,
        source: None,
    };

    let append_result = { system.read().storage().oplog_append(draft) };
    if let Err(e) = append_result {
        crate::metrics::TRACE_CAPTURE_FAILURES_TOTAL
            .with_label_values(&["append_error"])
            .inc();
        tracing::error!(user_id = %user_id, session_id = %session_id, error = %e,
            "oplog append failed — session trace marked incomplete");
        let _ = { system.read().storage().oplog_mark_incomplete(&session_id) };
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_name_uses_templates_and_prefixes_every_verb() {
        use axum::http::Method;
        assert_eq!(op_name(&Method::POST, "/api/recall"), "post:recall");
        assert_eq!(op_name(&Method::GET, "/api/memories"), "get:memories");
        // Review C3: an update and a deletion of the same resource must never
        // be byte-identical in the permanent record.
        assert_ne!(
            op_name(&Method::PUT, "/api/memory/{memory_id}"),
            op_name(&Method::DELETE, "/api/memory/{memory_id}"),
        );
        // Review I4: templates keep user-controlled segments out of op names.
        assert_eq!(
            op_name(&Method::GET, "/api/brain/{user_id}"),
            "get:brain:{user_id}"
        );
    }

    #[test]
    fn exclusions_cover_amendment_10() {
        for p in [
            "/api/trace/report",
            "/api/trace/s1",
            "/api/trace",
            "/metrics",
        ] {
            assert!(is_excluded(p), "{p} must be excluded");
        }
        for p in EXCLUDED_EXACT {
            assert!(is_excluded(p), "{p} must be excluded");
        }
        // Review M1: exact-match semantics — sibling routes are NOT excluded.
        assert!(!is_excluded("/metricsz"));
        assert!(!is_excluded("/api/traces"));
        assert!(!is_excluded("/api/recall"));
    }

    #[test]
    fn no_identity_set_is_exact_where_audit_verified_exact() {
        assert!(is_known_no_identity("/api/users"));
        assert!(is_known_no_identity("/api/ab/summary"));
        // Review C1: the erasure and per-user routes MUST be capture-visible —
        // prefix semantics on /api/users would have swallowed them.
        assert!(!is_known_no_identity("/api/users/some-user"));
        assert!(!is_known_no_identity("/api/users/some-user/stats"));
        assert!(!is_known_no_identity("/api/recall"));
    }
}
