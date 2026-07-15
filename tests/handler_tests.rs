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
    let (status, _) = json_of(h.app(), authed_get("/api/sessions/stats")).await;
    assert!(status.is_success());
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
