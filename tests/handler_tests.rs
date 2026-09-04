//! Smoke tests for all HTTP handler endpoints.
//!
//! Each handler group (health, users, recall, etc.) gets at least one test
//! that verifies:
//! - Valid requests return 2xx on fresh (empty) state.
//! - The auth middleware rejects unauthenticated access to protected routes.
//!
//! Run with: `cargo test --test handler_tests`

use std::sync::{Arc, Once};

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use serde_json::json;
use tempfile::TempDir;
use tower::ServiceExt;

use shodh_memory::{
    config::ServerConfig,
    handlers::{
        build_probe_routes, build_protected_routes, build_public_routes, MultiUserMemoryManager,
    },
    memory::types::{Experience, ExperienceType},
};

// ═══════════════════════════════════════════════════════════════════════
// Test infrastructure
// ═══════════════════════════════════════════════════════════════════════

const TEST_KEY: &str = "handler-smoke-test-key";
static ENV_INIT: Once = Once::new();

fn init_env() {
    ENV_INIT.call_once(|| {
        // SAFETY: called once before any parallel tests start.
        unsafe {
            std::env::set_var("SHODH_API_KEYS", TEST_KEY);
        }
        let _ = shodh_memory::metrics::register_metrics();
    });
}

/// Self-contained test harness with a fresh temp directory and RocksDB.
struct Harness {
    mgr: Arc<MultiUserMemoryManager>,
    _dir: TempDir,
}

impl Harness {
    fn new() -> Self {
        init_env();
        let dir = TempDir::new().expect("create temp dir");
        let cfg = ServerConfig {
            storage_path: dir.path().to_path_buf(),
            backup_enabled: false,
            ..ServerConfig::default()
        };
        let mgr = MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg)
            .expect("create MultiUserMemoryManager");
        Self {
            mgr: Arc::new(mgr),
            _dir: dir,
        }
    }

    fn app(&self) -> Router {
        // Mirror main.rs: auth middleware only wraps protected routes.
        let probe = build_probe_routes(self.mgr.clone());
        let public = build_public_routes(self.mgr.clone());
        let protected = build_protected_routes(self.mgr.clone()).layer(axum::middleware::from_fn(
            shodh_memory::auth::auth_middleware,
        ));
        Router::new().merge(probe).merge(public).merge(protected)
    }

    /// Seed `count` distinct memories for `user_id` directly through the memory
    /// system, bypassing HTTP and the real embedder (a canned 384-dim vector is
    /// supplied so `remember()` skips embedding generation). This is the same
    /// direct-call pattern `brutal_stress_tests.rs` uses for bulk seeding, and is
    /// necessary here to seed 1000+ records without paying ~300-500ms/record of
    /// real embedding-generation cost per the timing measured in
    /// `test_brutal_timing_record`.
    fn seed_memories(&self, user_id: &str, count: usize) {
        let memory = self.mgr.get_user_memory(user_id).expect("get_user_memory");
        let guard = memory.read();
        for i in 0..count {
            let exp = Experience {
                experience_type: ExperienceType::Learning,
                content: format!("pagination-seed-{user_id}-{i:06}"),
                embeddings: Some(vec![0.01_f32; 384]),
                ..Default::default()
            };
            guard.remember(exp, None).expect("seed remember");
        }
    }
}

// ── request helpers ──

fn authed_get(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .header("x-api-key", TEST_KEY)
        .body(Body::empty())
        .unwrap()
}

fn authed_post(uri: &str, body: serde_json::Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(bytes))
        .unwrap()
}

#[allow(dead_code)]
fn authed_put(uri: &str, body: serde_json::Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::PUT)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(bytes))
        .unwrap()
}

#[allow(dead_code)]
fn authed_delete(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::DELETE)
        .uri(uri)
        .header("x-api-key", TEST_KEY)
        .body(Body::empty())
        .unwrap()
}

#[allow(dead_code)]
fn authed_delete_json(uri: &str, body: serde_json::Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::DELETE)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(bytes))
        .unwrap()
}

fn noauth_get(uri: &str) -> Request<Body> {
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn noauth_post(uri: &str, body: serde_json::Value) -> Request<Body> {
    let bytes = serde_json::to_vec(&body).unwrap();
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(bytes))
        .unwrap()
}

// ── response helpers ──

async fn status_of(app: Router, req: Request<Body>) -> StatusCode {
    app.oneshot(req).await.unwrap().status()
}

async fn json_of(app: Router, req: Request<Body>) -> (StatusCode, serde_json::Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let val = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&bytes).to_string())
        })
    };
    (status, val)
}

// ═══════════════════════════════════════════════════════════════════════
// AUTH MIDDLEWARE
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn auth_public_routes_need_no_key() {
    let h = Harness::new();
    // /health is public
    assert_eq!(
        status_of(h.app(), noauth_get("/health")).await,
        StatusCode::OK
    );
    // /health/live is public
    assert_eq!(
        status_of(h.app(), noauth_get("/health/live")).await,
        StatusCode::OK
    );
}

#[tokio::test]
async fn auth_protected_routes_reject_missing_key() {
    let h = Harness::new();
    let status = status_of(
        h.app(),
        noauth_post("/api/recall", json!({"user_id":"u","query":"test"})),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_graph_data_requires_key() {
    // Regression: /api/graph/data/{user_id} was accidentally public.
    let h = Harness::new();
    let status = status_of(h.app(), noauth_get("/api/graph/data/test-user")).await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "/api/graph/data must be behind auth"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// health.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn health_endpoint() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body.get("status").is_some(),
        "health response needs 'status' field"
    );
    let system_memory = body["system_memory"]
        .as_object()
        .expect("health response needs 'system_memory' object");
    for field in [
        "process_rss_bytes",
        "process_peak_rss_bytes",
        "process_virtual_bytes",
        "cgroup_memory_current_bytes",
        "cgroup_memory_peak_bytes",
    ] {
        assert!(
            system_memory.contains_key(field),
            "system_memory missing '{field}'"
        );
    }
}

#[tokio::test]
async fn health_live() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/health/live")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "alive");
}

#[tokio::test]
async fn health_ready() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/health/ready")).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn health_index() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/health/index?user_id=test-user")).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn metrics_endpoint() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/metrics")).await;
    // metrics handler should always return 200 in test harness
    assert!(
        status == StatusCode::OK,
        "unexpected metrics status: {status}"
    );
    let metrics = body.as_str().expect("metrics response should be text");
    assert!(
        metrics.contains("shodh_process_rss_bytes"),
        "metrics response should include process RSS gauge"
    );
    assert!(
        metrics.contains("shodh_cgroup_memory_current_bytes"),
        "metrics response should include cgroup memory gauge"
    );
    assert!(
        metrics.contains("shodh_rocksdb_block_cache_capacity_bytes"),
        "metrics response should include RocksDB block cache capacity gauge"
    );
}

#[tokio::test]
async fn context_status_roundtrip() {
    let h = Harness::new();
    // POST a context status update (public route)
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/context/status",
            json!({
                "session_id": "test-session",
                "tokens_used": 1000,
                "tokens_budget": 100000
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // GET context statuses
    let (status, body) = json_of(h.app(), authed_get("/api/context/status")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

// ═══════════════════════════════════════════════════════════════════════
// users.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_users_empty() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/api/users")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

#[tokio::test]
async fn user_stats_fresh() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/users/test-user/stats")).await;
    // Creates user on demand → should succeed
    assert!(status.is_success(), "user stats returned {status}");
}

#[tokio::test]
async fn stats_query() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/stats?user_id=test-user")).await;
    assert!(status.is_success(), "stats query returned {status}");
}

// ═══════════════════════════════════════════════════════════════════════
// sessions.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_sessions_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/sessions", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn session_stats() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/sessions/stats?user_id=test-user")).await;
    assert!(status.is_success());
}

/// `/api/sessions/stats` must answer about the caller's own sessions and nobody
/// else's.
///
/// Regression guard for a cross-tenant leak: the handler used to take no
/// `user_id` at all and return `SessionStore::stats()`, a sum over every user in
/// the process. The `user_id` query parameter was accepted by axum and silently
/// dropped, so four different tenants received a byte-identical aggregate —
/// including `users_with_sessions`, which disclosed how many other tenants
/// existed to any authenticated key.
///
/// The load-bearing assertion is the inequality: any future refactor that
/// reintroduces a cross-user sum makes the two responses identical again and
/// turns this red.
#[tokio::test]
async fn session_stats_are_scoped_to_the_caller() {
    let h = Harness::new();

    // Tenant A has one completed and one active session. Tenant B has none.
    let store = h.mgr.session_store();
    let finished = store.start_session("tenant-a");
    store.end_session(&finished, "test");
    let _live = store.start_session("tenant-a");

    let (status_a, body_a) =
        json_of(h.app(), authed_get("/api/sessions/stats?user_id=tenant-a")).await;
    assert!(status_a.is_success(), "tenant-a stats returned {status_a}");

    let (status_b, body_b) =
        json_of(h.app(), authed_get("/api/sessions/stats?user_id=tenant-b")).await;
    assert!(status_b.is_success(), "tenant-b stats returned {status_b}");

    assert_ne!(
        body_a["stats"], body_b["stats"],
        "two tenants with different session histories received the same \
         aggregate — the endpoint is not scoped to the caller"
    );

    assert_eq!(body_a["stats"]["user_id"], "tenant-a");
    assert_eq!(body_a["stats"]["active_sessions"], 1);
    assert_eq!(body_a["stats"]["completed_sessions"], 1);

    assert_eq!(body_b["stats"]["user_id"], "tenant-b");
    assert_eq!(body_b["stats"]["active_sessions"], 0);
    assert_eq!(body_b["stats"]["completed_sessions"], 0);

    // The tenant census must not be reachable from a per-user response.
    assert!(
        body_a["stats"]["users_with_sessions"].is_null(),
        "per-user session stats must not disclose how many tenants exist"
    );
}

/// The endpoint must reject a missing `user_id` rather than fall back to an
/// unscoped answer. A silent default is exactly how the original leak read to a
/// caller: a request that looks scoped and is not.
#[tokio::test]
async fn session_stats_require_a_user_id() {
    let h = Harness::new();
    let status = status_of(h.app(), authed_get("/api/sessions/stats")).await;
    assert!(
        status.is_client_error(),
        "unscoped /api/sessions/stats returned {status}, expected a 4xx"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// remember.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn remember_basic() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "test-user",
                "content": "The sky is blue because of Rayleigh scattering."
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "remember failed: {body}");
    // Should return a memory_id
    assert!(
        body.get("id").is_some() || body.get("memory_id").is_some(),
        "remember response should contain an id: {body}"
    );
}

/// END-TO-END NER EMISSION — `POST /api/remember` → `GET /api/memory/{id}`.
///
/// Pins the whole remember path in one assertion chain: GLiNER types the
/// content, the `LOC` records survive into the stored `Experience`, and the
/// gazetteer resolves them into `toponyms`. No layer below this had coverage of
/// the production typer through the HTTP handler — which is why a report of
/// `ner_entities: []` could not be answered by running the suite.
///
/// The GLiNER assets are a hard requirement rather than a skip condition: a
/// test that returns early when the model is absent cannot fail, so it cannot
/// answer the question either. CI provisions `SHODH_GLINER_MODEL_PATH` and
/// already refuses to run without it.
#[tokio::test]
async fn remember_persists_ner_entities_end_to_end() {
    use shodh_memory::embeddings::gliner::GlinerConfig;

    let gliner = GlinerConfig::from_env();
    assert!(
        gliner.assets_present(),
        "GLiNER assets missing at {:?} — this test pins the PRODUCTION remember path and must \
         not silently pass on the rule-based fallback. Set SHODH_GLINER_MODEL_PATH.",
        gliner.model_path
    );

    let h = Harness::new();
    let content = "The annual conference was held in Baltimore before moving to Norfolk.";

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({ "user_id": "ner-emission-user", "content": content }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "remember failed: {body}");

    let memory_id = body
        .get("id")
        .or_else(|| body.get("memory_id"))
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("remember response has no id: {body}"))
        .to_string();

    let (status, fetched) = json_of(
        h.app(),
        authed_get(&format!(
            "/api/memory/{memory_id}?user_id=ner-emission-user"
        )),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "get_memory failed: {fetched}");

    // `MemoryWithHierarchy` flattens the memory, so `experience` sits at the
    // response root. `toponyms` is NOT under `experience`: it is `#[serde(skip)]`
    // there and carried at the `MemoryFlat` tail, so it surfaces at the root too.
    let ner_entities = fetched
        .pointer("/experience/ner_entities")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("no experience.ner_entities in response: {fetched}"));

    assert!(
        !ner_entities.is_empty(),
        "remember stored ZERO ner_entities for {content:?} while GLiNER was loaded — emission \
         is broken. The gazetteer sees no LOC records and every downstream consumer falls back \
         to the untyped keyword path. Full memory: {fetched}"
    );

    let locations: Vec<&str> = ner_entities
        .iter()
        .filter(|e| e.get("entity_type").and_then(|t| t.as_str()) == Some("LOC"))
        .filter_map(|e| e.get("text").and_then(|t| t.as_str()))
        .collect();
    for expected in ["Baltimore", "Norfolk"] {
        assert!(
            locations.iter().any(|t| t.eq_ignore_ascii_case(expected)),
            "expected a LOC record for {expected:?}; got LOC records {locations:?} out of \
             {ner_entities:?}"
        );
    }

    // The gazetteer is the first consumer that reads ONLY `LOC` records, which
    // makes it the sharpest downstream detector of an emission break: zero LOC
    // records and it resolves nothing, indistinguishable from a memory that
    // names no places. Pin the resolution, not just the entity.
    let toponyms = fetched
        .pointer("/toponyms")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("no toponyms in response: {fetched}"));
    let resolved: Vec<&str> = toponyms
        .iter()
        .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
        .collect();
    for expected in ["Baltimore", "Norfolk"] {
        assert!(
            resolved.iter().any(|n| n.eq_ignore_ascii_case(expected)),
            "NER emitted a LOC record for {expected:?} but the gazetteer resolved {resolved:?} — \
             the toponym half of the remember path is broken"
        );
    }
}

#[tokio::test]
async fn remember_with_tags_and_type() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "test-user",
                "content": "Decided to use Rust for the memory engine.",
                "memory_type": "Decision",
                "tags": ["architecture", "rust"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn batch_remember() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember/batch",
            json!({
                "user_id": "test-user",
                "memories": [
                    {"content": "First memory item"},
                    {"content": "Second memory item", "tags": ["batch"]}
                ]
            }),
        ),
    )
    .await;
    assert!(status.is_success(), "batch remember returned {status}");
}

#[tokio::test]
async fn upsert_memory() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/upsert",
            json!({
                "user_id": "test-user",
                "external_id": "ext-001",
                "content": "Upserted memory content."
            }),
        ),
    )
    .await;
    assert!(status.is_success(), "upsert returned {status}");
}

/// Caller-declared write identity must survive the round trip.
///
/// `Memory` carries three multi-tenancy fields — `agent_id`, `run_id`,
/// `actor_id` — and all three read as `null` on every row of a real store. That
/// is two different causes wearing one symptom, and this test separates them:
/// `agent_id`/`run_id` have been accepted and persisted all along (no client
/// sends them), while `actor_id` had no request field and no write path at all,
/// so it was structurally unreachable.
///
/// The `actor_id` assertion is the fail-first one. The other two are a
/// regression guard on the `remember_with_agent_detailed` branch at
/// `remember.rs`, which is easy to lose because nothing in production exercises
/// it.
#[tokio::test]
async fn remember_persists_caller_declared_identity() {
    let h = Harness::new();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "identity-user",
                "content": "Provenance round-trip: agent, run and actor.",
                "agent_id": "agent-7",
                "run_id": "run-42",
                "actor_id": "actor-varun"
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "remember failed: {body}");

    let id = body
        .get("id")
        .or_else(|| body.get("memory_id"))
        .and_then(|v| v.as_str())
        .expect("remember response carries an id")
        .to_string();

    let (status, stored) = json_of(
        h.app(),
        authed_get(&format!("/api/memory/{id}?user_id=identity-user")),
    )
    .await;
    assert!(status.is_success(), "get_memory returned {status}");

    let memory = stored.get("memory").unwrap_or(&stored);
    assert_eq!(memory["agent_id"], "agent-7", "agent_id dropped: {stored}");
    assert_eq!(memory["run_id"], "run-42", "run_id dropped: {stored}");
    assert_eq!(
        memory["actor_id"], "actor-varun",
        "actor_id dropped — the field is persisted by `Memory` but nothing \
         accepts or writes it: {stored}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// audit trail — per-memory history
// ═══════════════════════════════════════════════════════════════════════

/// Every memory mutation is written to `CF_AUDIT` by ~23 `log_event` call
/// sites, and until this route existed the only thing any client could learn
/// from that trail was a bucketed tally of event-type names on
/// `POST /api/sessions/digest`. `MultiUserMemoryManager::get_history` — the
/// per-memory reader written to serve exactly this — had zero callers
/// repo-wide.
///
/// Fail-first: without the route this is a 404.
#[tokio::test]
async fn memory_history_serves_the_audit_trail_for_one_memory() {
    let h = Harness::new();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "audit-user",
                "content": "Original content before the update."
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "remember failed: {body}");
    let id = body
        .get("id")
        .or_else(|| body.get("memory_id"))
        .and_then(|v| v.as_str())
        .expect("remember response carries an id")
        .to_string();

    // A second memory whose audit entries must NOT appear in the first one's
    // history — the filter is the point of a per-memory reader.
    let (_, other) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": "audit-user", "content": "An unrelated memory."}),
        ),
    )
    .await;
    let other_id = other
        .get("id")
        .or_else(|| other.get("memory_id"))
        .and_then(|v| v.as_str())
        .expect("remember response carries an id")
        .to_string();

    // `PUT /api/memory/{id}` is an audited mutation (crud.rs logs "UPDATE").
    for (target, text) in [(&id, "First revision."), (&other_id, "Other revision.")] {
        let (status, resp) = json_of(
            h.app(),
            authed_put(
                &format!("/api/memory/{target}"),
                json!({"user_id": "audit-user", "content": text}),
            ),
        )
        .await;
        assert!(status.is_success(), "update returned {status}: {resp}");
    }

    let (status, history) = json_of(
        h.app(),
        authed_get(&format!("/api/memory/{id}/history?user_id=audit-user")),
    )
    .await;
    assert!(
        status.is_success(),
        "per-memory history returned {status}: {history}"
    );

    let events = history["events"]
        .as_array()
        .unwrap_or_else(|| panic!("history response has an `events` array: {history}"));
    assert!(
        !events.is_empty(),
        "the UPDATE was audited but the reader returned nothing: {history}"
    );
    assert!(
        events
            .iter()
            .all(|e| e["memory_id"].as_str() == Some(id.as_str())),
        "history leaked another memory's audit entries: {history}"
    );
    assert!(
        events
            .iter()
            .any(|e| e["event_type"].as_str() == Some("UPDATE")),
        "the UPDATE event is missing from the memory's history: {history}"
    );
    assert_eq!(history["memory_id"], id);
    assert_eq!(history["count"], events.len());
}

/// The route is user-scoped like every other namespace read: `user_id` is
/// required, and one tenant cannot read another tenant's audit trail.
#[tokio::test]
async fn memory_history_is_user_scoped() {
    let h = Harness::new();

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": "audit-owner", "content": "Owned by audit-owner."}),
        ),
    )
    .await;
    let id = body
        .get("id")
        .or_else(|| body.get("memory_id"))
        .and_then(|v| v.as_str())
        .expect("remember response carries an id")
        .to_string();

    let (status, resp) = json_of(
        h.app(),
        authed_put(
            &format!("/api/memory/{id}"),
            json!({"user_id": "audit-owner", "content": "Revised by the owner."}),
        ),
    )
    .await;
    assert!(status.is_success(), "update returned {status}: {resp}");

    let (status, mine) = json_of(
        h.app(),
        authed_get(&format!("/api/memory/{id}/history?user_id=audit-owner")),
    )
    .await;
    assert!(status.is_success(), "owner history returned {status}");
    assert!(
        !mine["events"].as_array().expect("events array").is_empty(),
        "owner sees no history: {mine}"
    );

    let (status, theirs) = json_of(
        h.app(),
        authed_get(&format!("/api/memory/{id}/history?user_id=audit-stranger")),
    )
    .await;
    assert!(status.is_success(), "stranger history returned {status}");
    assert!(
        theirs["events"]
            .as_array()
            .expect("events array")
            .is_empty(),
        "a different tenant read this memory's audit trail: {theirs}"
    );

    // Missing user_id is a 4xx, not an unscoped answer.
    let status = status_of(h.app(), authed_get(&format!("/api/memory/{id}/history"))).await;
    assert!(
        status.is_client_error(),
        "unscoped history returned {status}, expected a 4xx"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// recall.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn recall_empty_state() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({
                "user_id": "test-user",
                "query": "what do I know?"
            }),
        ),
    )
    .await;
    assert!(status.is_success(), "recall returned {status}: {body}");
}

#[tokio::test]
async fn context_summary_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/context_summary", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn proactive_context() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/proactive_context",
            json!({
                "user_id": "test-user",
                "context": "Working on handler tests."
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn recall_by_tags_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/recall/tags",
            json!({
                "user_id": "test-user",
                "tags": ["nonexistent"]
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn reinforce_feedback() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/reinforce",
            json!({
                "user_id": "test-user",
                "ids": [],
                "outcome": "helpful"
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// crud.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_memories_get_empty() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/list/test-user")).await;
    assert!(status.is_success());
}

#[tokio::test]
async fn list_memories_post_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/memories", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

// ── Issue #407: offset/limit pagination on the list endpoints ──

/// GET /api/list/{user_id} (list_memories): offset actually skips records rather
/// than being silently ignored. Pages of 10, chunked from a known-good full
/// listing, must match the corresponding slice — proving `.skip(offset)` lines
/// up with the same ordering `get_all_memories()` produces.
#[tokio::test]
async fn list_memories_get_offset_paginates_distinct_pages() {
    let h = Harness::new();
    h.seed_memories("offset-user", 30);

    let (status, full_body) = json_of(h.app(), authed_get("/api/list/offset-user?limit=30")).await;
    assert_eq!(status, StatusCode::OK);
    let full_ids: Vec<String> = full_body["memories"]
        .as_array()
        .expect("memories array")
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(full_ids.len(), 30, "expected all 30 seeded memories");

    for (page_index, expected_chunk) in full_ids.chunks(10).enumerate() {
        let offset = page_index * 10;
        let (status, body) = json_of(
            h.app(),
            authed_get(&format!("/api/list/offset-user?limit=10&offset={offset}")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let page_ids: Vec<String> = body["memories"]
            .as_array()
            .expect("memories array")
            .iter()
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            page_ids, expected_chunk,
            "page at offset={offset} should equal slice [{offset}..{offset}+10] of the full list"
        );
    }

    // Past-the-end offset returns an empty page, not an error.
    let (status, body) = json_of(
        h.app(),
        authed_get("/api/list/offset-user?limit=10&offset=1000"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["memories"].as_array().unwrap().is_empty());
}

/// POST /api/memories (list_memories_inner via list_memories_post): offset in the
/// request body composes with limit to walk distinct, non-overlapping pages.
#[tokio::test]
async fn list_memories_post_offset_paginates_distinct_pages() {
    let h = Harness::new();
    h.seed_memories("post-offset-user", 12);

    let (status, page0) = json_of(
        h.app(),
        authed_post(
            "/api/memories",
            json!({"user_id": "post-offset-user", "limit": 5, "offset": 0}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, page1) = json_of(
        h.app(),
        authed_post(
            "/api/memories",
            json!({"user_id": "post-offset-user", "limit": 5, "offset": 5}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let ids0: std::collections::HashSet<String> = page0["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    let ids1: std::collections::HashSet<String> = page1["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids0.len(), 5);
    assert_eq!(ids1.len(), 5);
    assert!(
        ids0.is_disjoint(&ids1),
        "offset=0 and offset=5 pages must not overlap: {ids0:?} vs {ids1:?}"
    );
}

/// GET /api/memories?...&offset=... (list_memories_get -> list_memories_inner):
/// confirms `offset` is threaded from `ListMemoriesQuery` into `ListMemoriesRequest`
/// rather than being dropped on the query-param path.
#[tokio::test]
async fn list_memories_get_query_offset_is_threaded() {
    let h = Harness::new();
    h.seed_memories("get-query-offset-user", 12);

    let (status, page0) = json_of(
        h.app(),
        authed_get("/api/memories?user_id=get-query-offset-user&limit=5&offset=0"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, page1) = json_of(
        h.app(),
        authed_get("/api/memories?user_id=get-query-offset-user&limit=5&offset=5"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let ids0: std::collections::HashSet<String> = page0["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    let ids1: std::collections::HashSet<String> = page1["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert!(
        ids0.is_disjoint(&ids1),
        "offset must be threaded through the GET query-param path, not ignored"
    );
}

/// A `limit` above the old hard-coded 1000 cap must actually return more than
/// 1000 records when the store has more — the core regression from #407.
#[tokio::test]
async fn list_memories_limit_above_1000_returns_more_than_1000() {
    let h = Harness::new();
    let seeded: usize = 1200;
    h.seed_memories("big-user", seeded);

    let (status, body) = json_of(h.app(), authed_get("/api/list/big-user?limit=2000")).await;
    assert_eq!(status, StatusCode::OK);
    let memories = body["memories"].as_array().expect("memories array");
    assert!(
        memories.len() > 1000,
        "limit=2000 over {seeded} stored memories must exceed the old hard cap of 1000, got {}",
        memories.len()
    );
    assert_eq!(
        memories.len(),
        seeded,
        "all {seeded} seeded memories should be returned under the new MAX_LIST_LIMIT ceiling"
    );
    assert_eq!(body["total"].as_u64().unwrap(), seeded as u64);
}

/// `total` must reflect the full filtered count, not the page size — so callers
/// can tell whether they've received everything.
#[tokio::test]
async fn list_memories_total_reflects_full_count_not_page_size() {
    let h = Harness::new();
    let seeded: usize = 15;
    h.seed_memories("total-user", seeded);

    let (status, body) =
        json_of(h.app(), authed_get("/api/list/total-user?limit=3&offset=5")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["memories"].as_array().unwrap().len(),
        3,
        "page size should honor limit=3"
    );
    assert_eq!(
        body["total"].as_u64().unwrap(),
        seeded as u64,
        "total must reflect the full filtered count, independent of limit/offset"
    );
}

/// Callers that don't pass offset/limit keep the pre-#407 defaults: limit=100,
/// offset=0 (i.e. from the start).
#[tokio::test]
async fn list_memories_default_limit_and_offset_unchanged() {
    let h = Harness::new();
    h.seed_memories("defaults-user", 5);

    let (status, body) = json_of(h.app(), authed_get("/api/list/defaults-user")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["memories"].as_array().unwrap().len(),
        5,
        "default limit=100 should return all 5 seeded memories"
    );
    assert_eq!(body["total"].as_u64().unwrap(), 5);
}

#[tokio::test]
async fn get_memory_not_found() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/memory/nonexistent-id")).await;
    // handler may return 404 or 422 for missing memory
    assert!(
        status == StatusCode::NOT_FOUND
            || status == StatusCode::UNPROCESSABLE_ENTITY
            || status == StatusCode::BAD_REQUEST,
        "expected error status for missing memory, got {status}"
    );
}

#[tokio::test]
async fn forget_by_age() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/forget/age",
            json!({"user_id": "test-user", "days_old": 30}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn forget_by_importance() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/forget/importance",
            json!({"user_id": "test-user", "threshold": 0.1}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn forget_by_pattern() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/forget/pattern",
            json!({"user_id": "test-user", "pattern": "nonexistent"}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn forget_by_tags() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/forget/tags",
            json!({"user_id": "test-user", "tags": ["cleanup"]}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn clear_all_memories_requires_confirm() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/memories/clear",
            json!({"user_id": "test-user", "confirm": "CONFIRM"}),
        ),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// search.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn advanced_search_empty() {
    let h = Harness::new();
    // Handler requires at least one criterion; sending none → 4xx.
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/search/advanced", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(
        status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY,
        "expected 4xx for missing criteria, got {status}"
    );

    // With a criterion, should succeed even on empty state.
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/search/advanced",
            json!({"user_id": "test-user", "entity_name": "Rust"}),
        ),
    )
    .await;
    assert!(
        status.is_success(),
        "advanced_search with entity should succeed: {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// facts.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_facts_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/facts/list", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn search_facts_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/facts/search",
            json!({"user_id": "test-user", "query": "anything"}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn facts_by_entity_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/facts/by-entity",
            json!({"user_id": "test-user", "entity": "Rust"}),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn facts_stats() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/facts/stats", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// compression.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn compress_nonexistent_memory() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/memory/compress",
            json!({"user_id": "test-user", "id": "nonexistent"}),
        ),
    )
    .await;
    // Should fail gracefully for nonexistent memory
    assert!(
        status == StatusCode::NOT_FOUND
            || status == StatusCode::BAD_REQUEST
            || status == StatusCode::UNPROCESSABLE_ENTITY,
        "compress returned unexpected: {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// lineage.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn lineage_list_edges_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/lineage/edges", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn lineage_stats_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/lineage/stats", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// graph.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn graph_stats_fresh() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/graph/test-user/stats")).await;
    assert!(status.is_success());
}

#[tokio::test]
async fn get_all_entities_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/graph/entities/all", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn find_entity_nonexistent() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/graph/entity/find",
            json!({"user_id": "test-user", "entity_name": "NoSuchEntity"}),
        ),
    )
    .await;
    // Should return 200 with null/empty or 404
    assert!(
        status.is_success() || status == StatusCode::NOT_FOUND,
        "find_entity returned {status}"
    );
}

#[tokio::test]
async fn traverse_graph_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/graph/traverse",
            json!({
                "user_id": "test-user",
                "entity_name": "NoEntity"
            }),
        ),
    )
    .await;
    assert!(
        status.is_success() || status == StatusCode::NOT_FOUND,
        "traverse returned {status}"
    );
}

#[tokio::test]
async fn graph_data_authenticated() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/graph/data/test-user")).await;
    assert!(
        status.is_success(),
        "graph data with auth returned {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// todos.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn todo_lifecycle() {
    let h = Harness::new();

    // Create a todo
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({
                "user_id": "test-user",
                "content": "Write handler tests",
                "priority": "high",
                "tags": ["testing"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create todo: {body}");
    let todo_id = body["todo"]["id"]
        .as_str()
        .or(body["id"].as_str())
        .or(body["todo_id"].as_str());
    assert!(todo_id.is_some(), "todo response should contain id: {body}");
    let todo_id = todo_id.unwrap().to_string();

    // List todos
    let (status, body) = json_of(
        h.app(),
        authed_post("/api/todos", json!({"user_id": "test-user"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // Should have at least 1 todo
    let todos = body["todos"].as_array().or(body.as_array());
    assert!(
        todos.map(|t| !t.is_empty()).unwrap_or(false),
        "should have todos after create: {body}"
    );

    // Complete the todo (path-style route)
    let (status, _) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/complete"),
            json!({"user_id": "test-user"}),
        ),
    )
    .await;
    assert!(status.is_success(), "complete todo returned {status}");

    // Delete the todo (DELETE with query param)
    let (status, _) = json_of(
        h.app(),
        Request::builder()
            .method(Method::DELETE)
            .uri(format!("/api/todos/{todo_id}?user_id=test-user"))
            .header("x-api-key", TEST_KEY)
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert!(status.is_success(), "delete todo returned {status}");
}

#[tokio::test]
async fn list_todos_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/todos", json!({"user_id": "fresh-user"})),
    )
    .await;
    assert!(status.is_success());
}

/// Regression (capability-map F19): `list_todos` with a `query` that is a
/// literal prefix of an existing todo's content returned nothing. The lexical
/// path of hybrid search must guarantee exact word matches surface.
#[tokio::test]
async fn list_todos_query_finds_exact_word_match() {
    let h = Harness::new();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": "test-user", "content": "Audit-2 parent task"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create todo: {body}");

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos",
            json!({"user_id": "test-user", "query": "Audit-2 parent"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let todos = body["todos"].as_array().expect("todos array");
    assert!(
        todos
            .iter()
            .any(|t| t["content"].as_str() == Some("Audit-2 parent task")),
        "query must find the todo whose content it literally prefixes: {body}"
    );
}

/// Regression (capability-map F15): `reorder_todo` accepted any direction and
/// silently treated it as "down". Invalid directions must be a 400 that names
/// the accepted values.
#[tokio::test]
async fn reorder_todo_rejects_invalid_direction() {
    let h = Harness::new();

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": "test-user", "content": "Reorder me"}),
        ),
    )
    .await;
    let todo_id = body["todo"]["id"].as_str().expect("todo id").to_string();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/reorder"),
            json!({"user_id": "test-user", "direction": "sideways"}),
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "invalid direction must be rejected, got: {body}"
    );
    assert!(
        body.to_string().contains("up"),
        "error should name accepted values: {body}"
    );

    // Valid direction still works
    let (status, _) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/reorder"),
            json!({"user_id": "test-user", "direction": "up"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// Regression (capability-map F7): comment ids were never rendered, making
/// `update_todo_comment` / `delete_todo_comment` unreachable for clients that
/// read formatted output. Both the add confirmation and the list must carry
/// the id, and the id must actually drive update and delete.
#[tokio::test]
async fn todo_comment_ids_are_discoverable_and_usable() {
    let h = Harness::new();

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": "test-user", "content": "Comment target"}),
        ),
    )
    .await;
    let todo_id = body["todo"]["id"].as_str().expect("todo id").to_string();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/comments"),
            json!({"user_id": "test-user", "content": "an audit comment"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "add comment: {body}");
    let comment_id = body["comment"]["id"]
        .as_str()
        .expect("comment id")
        .to_string();
    assert!(
        body["formatted"]
            .as_str()
            .unwrap_or("")
            .contains(&comment_id),
        "add confirmation must render the comment id: {body}"
    );

    let (status, body) = json_of(
        h.app(),
        authed_get(&format!("/api/todos/{todo_id}/comments?user_id=test-user")),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["formatted"]
            .as_str()
            .unwrap_or("")
            .contains(&comment_id),
        "comment list must render ids: {body}"
    );

    // The rendered id drives update...
    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/comments/{comment_id}/update"),
            json!({"user_id": "test-user", "content": "edited"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "update comment: {body}");

    // ...and delete
    let (status, body) = json_of(
        h.app(),
        authed_delete(&format!(
            "/api/todos/{todo_id}/comments/{comment_id}?user_id=test-user"
        )),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "delete comment: {body}");
    assert_eq!(body["success"], json!(true));
}

/// Regression (capability-map F8): the subtasks endpoint returned a header and
/// a count with no rows. The formatted output must render the children.
#[tokio::test]
async fn subtasks_listing_renders_rows() {
    let h = Harness::new();

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": "test-user", "content": "Parent task"}),
        ),
    )
    .await;
    let parent_id = body["todo"]["id"].as_str().expect("parent id").to_string();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({
                "user_id": "test-user",
                "content": "Child task row",
                "parent_id": parent_id
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create subtask: {body}");
    assert_eq!(
        body["todo"]["parent_id"].as_str(),
        Some(parent_id.as_str()),
        "child must be parented: {body}"
    );

    let (status, body) = json_of(
        h.app(),
        authed_get(&format!(
            "/api/todos/{parent_id}/subtasks?user_id=test-user"
        )),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], json!(1), "one subtask expected: {body}");
    assert!(
        body["formatted"]
            .as_str()
            .unwrap_or("")
            .contains("Child task row"),
        "formatted subtask list must render the rows, not just a count: {body}"
    );
}

/// Structured dependencies end to end: create with blocked_by, reject a
/// dependency cycle with 400, and surface the newly unblocked todo on
/// completion of its last blocker.
#[tokio::test]
async fn todo_blocked_by_dependency_flow() {
    let h = Harness::new();

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": "test-user", "content": "Blocker task"}),
        ),
    )
    .await;
    let blocker_id = body["todo"]["id"].as_str().expect("blocker id").to_string();

    // Create a dependent todo referencing the blocker by UUID
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({
                "user_id": "test-user",
                "content": "Dependent task",
                "blocked_by": [blocker_id]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create dependent: {body}");
    let dependent_id = body["todo"]["id"]
        .as_str()
        .expect("dependent id")
        .to_string();
    assert_eq!(
        body["todo"]["blocked_by"].as_array().map(|a| a.len()),
        Some(1),
        "dependency must be stored: {body}"
    );

    // Unknown reference is rejected
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({
                "user_id": "test-user",
                "content": "Bad dep",
                "blocked_by": ["NOPE-999"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "unknown dependency ref");

    // Cycle: blocker cannot depend on dependent (dependent already waits on it)
    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{blocker_id}/update"),
            json!({"user_id": "test-user", "blocked_by": [dependent_id]}),
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "dependency cycle must be rejected: {body}"
    );

    // Completing the blocker surfaces the dependent as unblocked
    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{blocker_id}/complete"),
            json!({"user_id": "test-user"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "complete blocker: {body}");
    let unblocked = body["unblocked"].as_array().expect("unblocked array");
    assert!(
        unblocked
            .iter()
            .any(|t| t["content"].as_str() == Some("Dependent task")),
        "completing the last blocker must surface the dependent as unblocked: {body}"
    );
    assert!(
        body["formatted"]
            .as_str()
            .unwrap_or("")
            .contains("Unblocked"),
        "formatted completion must mention what was unblocked: {body}"
    );
}

#[tokio::test]
async fn todo_stats_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/todos/stats", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn project_lifecycle() {
    let h = Harness::new();

    // Create project
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/projects",
            json!({
                "user_id": "test-user",
                "name": "Test Project",
                "prefix": "TST"
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create project: {body}");

    // List projects
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/projects/list", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

/// `update_todo` with `status=done` assigned the status and stopped: no
/// `completed_at`, and — the expensive part — no recurrence rollover. A daily
/// task "finished" that way silently stopped recurring. Every client but the
/// one routed through `/complete` was exposed; the TUI cycles
/// `in_progress → done` through this exact path (`tui/src/stream.rs`,
/// `next_status`), and the MCP `update_todo` tool lists `done` in its enum.
#[tokio::test]
async fn update_to_done_settles_and_rolls_over_recurrence() {
    let h = Harness::new();
    let user = "settle-user";

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({
                "user_id": user,
                "content": "Water the plants",
                "recurrence": "daily",
                "due_date": "today"
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create todo: {body}");
    let todo_id = body["todo"]["id"].as_str().expect("todo id").to_string();
    assert_eq!(
        body["todo"]["recurrence"]["type"],
        json!("daily"),
        "recurrence must be persisted on create: {body}"
    );

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "status": "done"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "update to done: {body}");
    assert_eq!(body["todo"]["status"], json!("done"), "status: {body}");
    assert!(
        body["todo"]["completed_at"].is_string(),
        "completing through the update path must stamp completed_at: {body}"
    );
    assert_eq!(
        body["next_recurrence"]["content"],
        json!("Water the plants"),
        "the update response must surface the occurrence it spawned: {body}"
    );

    // The decisive assertion: a recurring task completed through /update must
    // leave a live next occurrence behind, exactly as /complete does.
    let (status, body) =
        json_of(h.app(), authed_post("/api/todos", json!({"user_id": user}))).await;
    assert_eq!(status, StatusCode::OK);
    let live: Vec<&serde_json::Value> = body["todos"]
        .as_array()
        .expect("todos array")
        .iter()
        .filter(|t| t["content"] == json!("Water the plants"))
        .collect();
    assert_eq!(
        live.len(),
        1,
        "completing a daily todo through /update must spawn exactly one next occurrence: {body}"
    );
    assert_ne!(
        live[0]["id"].as_str(),
        Some(todo_id.as_str()),
        "the next occurrence must be a new todo, not the completed one: {body}"
    );
    assert!(
        live[0]["completed_at"].is_null(),
        "a freshly spawned occurrence must not carry a completion stamp: {body}"
    );

    // Settlement fires on the transition, not on the value, so repeating the
    // same update spawns nothing further. The MCP tool metadata declares
    // update_todo idempotent; this is what that declaration rests on.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "status": "done"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "repeat update: {body}");
    assert!(
        body["next_recurrence"].is_null(),
        "re-marking a done todo done must not spawn another occurrence: {body}"
    );

    let (_, body) = json_of(h.app(), authed_post("/api/todos", json!({"user_id": user}))).await;
    let live = body["todos"]
        .as_array()
        .expect("todos array")
        .iter()
        .filter(|t| t["content"] == json!("Water the plants"))
        .count();
    assert_eq!(
        live, 1,
        "a repeated done update must leave exactly one live occurrence: {body}"
    );
}

/// The two doors must agree on every transition, not just the common one.
/// `/complete` on a cancelled recurring todo re-completes it and rolls it
/// over, so `/update` with `status=done` has to do the same — otherwise
/// "revive this and mark it done" silently ends the series depending on which
/// endpoint the client happens to use.
#[tokio::test]
async fn reviving_a_cancelled_todo_into_done_rolls_over_like_complete() {
    let h = Harness::new();
    let user = "revive-user";

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({
                "user_id": user,
                "content": "Take out the bins",
                "recurrence": "every 7 days",
                "due_date": "today"
            }),
        ),
    )
    .await;
    let todo_id = body["todo"]["id"].as_str().expect("todo id").to_string();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "status": "cancelled"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "cancel: {body}");
    let cancelled_at = body["todo"]["completed_at"]
        .as_str()
        .expect("cancelling stamps the settlement time")
        .to_string();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "status": "done"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "revive to done: {body}");
    assert_eq!(
        body["next_recurrence"]["content"],
        json!("Take out the bins"),
        "cancelled → done must roll the series over, as /complete does: {body}"
    );
    assert_eq!(
        body["todo"]["completed_at"].as_str(),
        Some(cancelled_at.as_str()),
        "the todo settled when it was cancelled; that time must not move: {body}"
    );
}

/// `completed_at` was written only by `Todo::complete()`, and never cleared.
/// A todo reopened after being done kept its completion stamp, which is how a
/// row in the "To do" column ended up reporting "took 2d". Settlement is the
/// server's job: entering Done/Cancelled stamps it, leaving them clears it.
#[tokio::test]
async fn settlement_stamp_follows_status_in_both_directions() {
    let h = Harness::new();
    let user = "stamp-user";

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": user, "content": "Reopen me later"}),
        ),
    )
    .await;
    let todo_id = body["todo"]["id"].as_str().expect("todo id").to_string();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/complete"),
            json!({"user_id": user}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "complete: {body}");
    let stamped_at = body["todo"]["completed_at"]
        .as_str()
        .expect("complete must stamp completed_at")
        .to_string();

    // Reopen: the stamp must go, or every client has to reconstruct settlement
    // from the status field to avoid showing "took 2d" on an open task.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "status": "todo"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "reopen: {body}");
    assert!(
        body["todo"]["completed_at"].is_null(),
        "reopening must clear completed_at (was {stamped_at}): {body}"
    );

    // Cancelling settles the todo too — it leaves the working set and stops
    // counting as overdue, so it needs a settlement time like Done does.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "status": "cancelled"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "cancel: {body}");
    assert!(
        body["todo"]["completed_at"].is_string(),
        "cancelling must stamp the settlement time: {body}"
    );
    assert!(
        body["next_recurrence"].is_null(),
        "cancelling must not spawn a next occurrence: {body}"
    );
}

/// `parse_recurrence` accepted three words and hardcoded their parameters:
/// weekly was always Mon–Fri, monthly always day 1, and `EveryNDays` was
/// unreachable from any client. All four variants must be expressible, the
/// bare words must keep working, and a zero interval must be rejected rather
/// than stored as a same-day infinite repeat.
#[tokio::test]
async fn recurrence_grammar_reaches_every_variant() {
    let h = Harness::new();
    let user = "recur-user";

    let cases = [
        ("daily", json!({"type": "daily"})),
        ("weekly", json!({"type": "weekly", "days": [1, 2, 3, 4, 5]})),
        ("monthly", json!({"type": "monthly", "day": 1})),
        ("weekly:mon,fri", json!({"type": "weekly", "days": [1, 5]})),
        ("weekly:0,6", json!({"type": "weekly", "days": [0, 6]})),
        // `next_occurrence` scans `days` for the first entry greater than
        // today, so the list has to come out sorted and deduplicated.
        ("Weekly:FRI,mon", json!({"type": "weekly", "days": [1, 5]})),
        ("weekly:mon,mon", json!({"type": "weekly", "days": [1]})),
        ("monthly:15", json!({"type": "monthly", "day": 15})),
        ("every 3 days", json!({"type": "every_n_days", "n": 3})),
        ("every_10_days", json!({"type": "every_n_days", "n": 10})),
    ];

    for (input, expected) in cases {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/todos/add",
                json!({"user_id": user, "content": format!("Recurs {input}"), "recurrence": input}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "create with '{input}': {body}");
        assert_eq!(
            body["todo"]["recurrence"], expected,
            "'{input}' must parse to {expected}: {body}"
        );
    }

    for bad in ["every 0 days", "weekly:funday", "monthly:0", "fortnightly"] {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/todos/add",
                json!({"user_id": user, "content": format!("Bad {bad}"), "recurrence": bad}),
            ),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "'{bad}' must be rejected, got: {body}"
        );
    }
}

/// `UpdateTodoRequest` had no `recurrence` field, so a recurrence could never
/// be changed or removed once the todo was created — the only escape was
/// delete-and-recreate, which loses the id, the comments and the links.
#[tokio::test]
async fn recurrence_is_editable_and_removable() {
    let h = Harness::new();
    let user = "recur-edit-user";

    let (_, body) = json_of(
        h.app(),
        authed_post(
            "/api/todos/add",
            json!({"user_id": user, "content": "Standup", "recurrence": "daily"}),
        ),
    )
    .await;
    let todo_id = body["todo"]["id"].as_str().expect("todo id").to_string();

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "recurrence": "weekly:mon"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "change recurrence: {body}");
    assert_eq!(
        body["todo"]["recurrence"],
        json!({"type": "weekly", "days": [1]}),
        "recurrence must be changeable: {body}"
    );

    // Empty string clears, matching how `parent_id` clears on this same request.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "recurrence": ""}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "clear recurrence: {body}");
    assert!(
        body["todo"]["recurrence"].is_null(),
        "an empty recurrence must remove it: {body}"
    );

    let (status, body) = json_of(
        h.app(),
        authed_post(
            &format!("/api/todos/{todo_id}/update"),
            json!({"user_id": user, "recurrence": "fortnightly"}),
        ),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "an unparseable recurrence must be rejected on update too: {body}"
    );
}

#[tokio::test]
async fn list_reminders_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/reminders", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn create_duration_reminder() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/reminders/set",
            json!({
                "user_id": "test-user",
                "content": "Check test results",
                "trigger": {"type": "duration", "after_seconds": 3600}
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// ab_testing.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_ab_tests_empty() {
    let h = Harness::new();
    let (status, body) = json_of(h.app(), authed_get("/api/ab/tests")).await;
    assert_eq!(status, StatusCode::OK);
    // Response is {"success":true, "tests":[...], "summary":{...}}
    assert!(
        body["tests"].is_array() || body.is_array(),
        "expected tests array: {body}"
    );
}

#[tokio::test]
async fn ab_test_lifecycle() {
    let h = Harness::new();

    // Create an A/B test
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/ab/tests",
            json!({
                "name": "recall_weights_v2",
                "description": "Test new recall weights",
                "traffic_split": 0.5
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create ab test: {body}");

    // List again
    let (status, body) = json_of(h.app(), authed_get("/api/ab/tests")).await;
    assert_eq!(status, StatusCode::OK);
    let tests = body["tests"].as_array().or(body.as_array());
    assert!(
        tests.map(|t| !t.is_empty()).unwrap_or(false),
        "should have one A/B test: {body}"
    );

    // Summary
    let (status, _) = json_of(h.app(), authed_get("/api/ab/summary")).await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// consolidation.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn verify_index_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/index/verify", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

/// Regression (capability-map F17): with no `since`, the consolidation report
/// covered only the last hour while clients document a 24-hour default, so it
/// reported "no activity" on stores that consolidated earlier the same day.
/// The default window must span 24 hours.
#[tokio::test]
async fn consolidation_report_defaults_to_24h_window() {
    let h = Harness::new();
    let (status, body) = json_of(
        h.app(),
        authed_post("/api/consolidation/report", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success(), "consolidation report: {body}");

    let start = body["period"]["start"]
        .as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .expect("period.start");
    let end = body["period"]["end"]
        .as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .expect("period.end");

    let window_mins = (end - start).num_minutes();
    assert!(
        (window_mins - 24 * 60).abs() <= 5,
        "default report window must span ~24h, got {window_mins} minutes"
    );
}

#[tokio::test]
async fn rebuild_index_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/index/rebuild", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn create_backup() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/backup/create", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn list_backups_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/backup/list", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// files.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn file_stats() {
    let h = Harness::new();
    // Handler takes user_id as query param: Query<TodoQuery>
    let (status, _) = json_of(h.app(), authed_get("/api/files/stats?user_id=test-user")).await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// mif.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn add_entity_to_graph() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/graph/entity/add",
            json!({
                "user_id": "test-user",
                "name": "Rust",
                "label": "Technology"
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn add_relationship_to_graph() {
    let h = Harness::new();

    // Add two entities first
    let _ = json_of(
        h.app(),
        authed_post(
            "/api/graph/entity/add",
            json!({"user_id": "test-user", "name": "Rust", "label": "Tech"}),
        ),
    )
    .await;
    let _ = json_of(
        h.app(),
        authed_post(
            "/api/graph/entity/add",
            json!({"user_id": "test-user", "name": "ONNX", "label": "Tech"}),
        ),
    )
    .await;

    // Add relationship
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/graph/relationship/add",
            json!({
                "user_id": "test-user",
                "from_entity": "Rust",
                "to_entity": "ONNX",
                "relation_type": "USES"
            }),
        ),
    )
    .await;
    assert!(status.is_success());
}

#[tokio::test]
async fn export_mif_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/export/mif", json!({"user_id": "test-user"})),
    )
    .await;
    assert!(status.is_success());
}

// ═══════════════════════════════════════════════════════════════════════
// visualization.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn graph_view_html() {
    let h = Harness::new();
    let resp = h.app().oneshot(noauth_get("/graph/view")).await.unwrap();
    // graph/view is public and returns HTML
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn brain_state_fresh() {
    let h = Harness::new();
    let (status, _) = json_of(h.app(), authed_get("/api/brain/test-user")).await;
    assert!(status.is_success());
}

#[tokio::test]
async fn build_visualization_empty() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/visualization/build", json!({"user_id": "test-user"})),
    )
    .await;
    // May return 200 or 404 on fresh state
    assert!(
        status.is_success() || status == StatusCode::NOT_FOUND,
        "build visualization returned {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// integrations.rs
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn linear_sync_no_integration() {
    let h = Harness::new();
    let (status, _) = json_of(
        h.app(),
        authed_post("/api/sync/linear", json!({"user_id": "test-user"})),
    )
    .await;
    // Without Linear API key configured, expect graceful error
    assert!(
        status == StatusCode::OK
            || status == StatusCode::BAD_REQUEST
            || status == StatusCode::UNPROCESSABLE_ENTITY,
        "linear sync returned unexpected {status}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// End-to-end: remember → recall cycle
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn remember_then_list() {
    let h = Harness::new();

    // Store a memory
    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "e2e-user",
                "content": "Melbourne has unpredictable weather.",
                "tags": ["weather", "melbourne"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // List memories — should have at least one
    let (status, body) = json_of(h.app(), authed_get("/api/list/e2e-user")).await;
    assert_eq!(status, StatusCode::OK);
    let memories = body.as_array().or(body["memories"].as_array());
    assert!(
        memories.map(|m| !m.is_empty()).unwrap_or(false),
        "should see stored memory in list: {body}"
    );

    // User stats should show 1 memory
    let (status, body) = json_of(h.app(), authed_get("/api/users/e2e-user/stats")).await;
    assert_eq!(status, StatusCode::OK);
    let count = body["total_memories"]
        .as_u64()
        .or(body["memory_count"].as_u64());
    assert!(
        count.unwrap_or(0) >= 1,
        "stats should show at least 1 memory: {body}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Capability-map defects: recall tag filter, lineage depth, backup reporting
// ═══════════════════════════════════════════════════════════════════════

/// `/api/recall`'s `tags` filter compared tags exactly while the storage tag
/// index (`search_by_tags`) normalises to lowercase, so `recall_by_tags(["X"])`
/// matched a memory tagged "x" and `recall(tags: ["X"])` returned nothing. An
/// agent reading "No memories found" concludes the corpus has no such memory.
#[tokio::test]
async fn recall_tag_filter_matches_regardless_of_case() {
    let h = Harness::new();

    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": "tag-case",
                "content": "The Seagirt terminal gate processed a seasonal high of truck transactions.",
                "tags": ["seagirt", "Terminal Gate"]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Every casing of a tag that exists must find the memory.
    for tag in ["seagirt", "Seagirt", "SEAGIRT"] {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({"user_id": "tag-case", "query": "terminal", "limit": 10, "tags": [tag]}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "recall with tags=[{tag}] failed");
        let count = recall_count(&body);
        assert!(
            count >= 1,
            "tags=[{tag}] should match the seeded memory: {body}"
        );
    }

    // A tag stored with capitals is equally reachable in lower case.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({"user_id": "tag-case", "query": "terminal", "limit": 10, "tags": ["terminal gate"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        recall_count(&body) >= 1,
        "lower-cased query of a capitalised tag should match: {body}"
    );

    // The filter must still exclude: a tag nothing carries returns nothing.
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({"user_id": "tag-case", "query": "terminal", "limit": 10, "tags": ["no-such-tag"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        recall_count(&body),
        0,
        "an absent tag must still filter everything out: {body}"
    );
}

fn recall_count(body: &serde_json::Value) -> u64 {
    body["count"]
        .as_u64()
        .or_else(|| body["memories"].as_array().map(|a| a.len() as u64))
        .unwrap_or(0)
}

/// `/api/lineage/trace` reported a `depth` that was the count of visited nodes,
/// so it always equalled `edges.len()` — a 5-hop request on a wide fan printed
/// "Depth reached: 31 │ Edges: 31". Depth must be the hop distance actually
/// walked, and can never exceed the requested `max_depth`.
#[tokio::test]
async fn lineage_trace_depth_is_hop_distance_not_edge_count() {
    let h = Harness::new();
    let user = "lineage-depth";

    // A fan: one root with three direct children, so edges (3) > depth (1).
    let mut ids = Vec::new();
    for i in 0..4 {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({"user_id": user, "content": format!("lineage depth node {i}")}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let id = body["memory_id"]
            .as_str()
            .or_else(|| body["id"].as_str())
            .unwrap_or_else(|| panic!("no id in remember response: {body}"))
            .to_string();
        ids.push(id);
    }

    for target in ids.iter().skip(1) {
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/link",
                json!({
                    "user_id": user,
                    "from_memory_id": ids[0],
                    "to_memory_id": target,
                    "relation": "Caused"
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "failed to link {target}: {body}");
    }

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/lineage/trace",
            json!({"user_id": user, "memory_id": ids[0], "direction": "forward", "max_depth": 1}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let depth = body["depth"]
        .as_u64()
        .unwrap_or_else(|| panic!("no depth: {body}"));
    let edges = body["edges"].as_array().map(|a| a.len()).unwrap_or(0);

    // The explicit fan is three edges. Automatic extraction may add more, so
    // assert a floor rather than an exact count — the point of the test is the
    // relationship between depth and edges, not the edge total.
    assert!(
        edges >= 3,
        "expected at least the three fan-out edges: {body}"
    );
    assert_eq!(
        depth, 1,
        "one hop was requested and one hop was walked; depth reported {depth} \
         against {edges} edges, which is the edge count, not a depth: {body}"
    );
    assert!(
        depth <= 1,
        "depth must never exceed the requested max_depth of 1: {body}"
    );
}

/// `backup_restore` reported `["graph"]` while its description promised it
/// "replaces all current data". The main memories DB is restored too — by
/// `restore_comprehensive_backup`, which propagates failure — it was simply
/// never named in the response, so an operator reading it would believe the
/// memories had not come back.
#[tokio::test]
async fn backup_restore_reports_every_store_it_restored() {
    let h = Harness::new();
    let user = "backup-report";

    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": user, "content": "a memory worth backing up"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = json_of(
        h.app(),
        authed_post("/api/backup/create", json!({"user_id": user})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "backup create failed: {body}");
    assert_eq!(body["success"], json!(true), "backup create failed: {body}");
    let backup_id = body["backup"]["backup_id"]
        .as_u64()
        .unwrap_or_else(|| panic!("no backup_id: {body}"));

    // The count rendered as "Memories" must be a memory count, not a raw key
    // count over the whole column family (which reported 897 for 88 memories).
    let memory_count = body["backup"]["memory_count"]
        .as_u64()
        .unwrap_or_else(|| panic!("no memory_count: {body}"));
    assert_eq!(
        memory_count, 1,
        "memory_count must count memories, not every RocksDB key: {body}"
    );

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/backup/restore",
            json!({"user_id": user, "backup_id": backup_id}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "restore failed: {body}");

    let restored: Vec<String> = body["restored_stores"]
        .as_array()
        .unwrap_or_else(|| panic!("no restored_stores: {body}"))
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();

    assert!(
        restored.iter().any(|s| s == "memories"),
        "the memories DB is restored and must be reported: {restored:?}"
    );
    // Whatever is reported must be reported once — the vector index used to be
    // pushed once per file found in the backup directory.
    let mut deduped = restored.clone();
    deduped.sort();
    deduped.dedup();
    assert_eq!(
        deduped.len(),
        restored.len(),
        "restored_stores contains duplicates: {restored:?}"
    );
}

/// Restoring a backup id that does not exist is a not-found condition, not a
/// server fault. It surfaced as HTTP 500 `INTERNAL_ERROR: NotFound`, which a
/// client cannot distinguish from a real backend failure.
#[tokio::test]
async fn backup_restore_unknown_id_is_not_found_not_server_error() {
    let h = Harness::new();
    let user = "backup-missing";

    let (status, _) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({"user_id": user, "content": "seed so the user exists"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Create one backup so the user's backup directory exists — the failure
    // under test is "that id is not among them", not "no backups at all".
    let (status, body) = json_of(
        h.app(),
        authed_post("/api/backup/create", json!({"user_id": user})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], json!(true), "backup create failed: {body}");

    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/backup/restore",
            json!({"user_id": user, "backup_id": 424242}),
        ),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "unknown backup id should be 404, got {status}: {body}"
    );
    assert_eq!(
        body["code"].as_str(),
        Some("BACKUP_NOT_FOUND"),
        "a structured code lets a client tell 'no such backup' from 'server broken': {body}"
    );
}
