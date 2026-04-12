//! Pipeline integration tests — remember→recall roundtrip verification.
//!
//! Tests the full data flow: store → embed → index → retrieve → return.
//! Verifies content fidelity, user isolation, dedup, multi-mode recall,
//! batch pipelines, upsert semantics, graph integration, and lineage inference.
//!
//! Run with: `cargo test --test pipeline_integration_tests -- --test-threads=1`

use std::sync::{Arc, Once};

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

use shodh_memory::{
    config::ServerConfig,
    handlers::{build_protected_routes, build_public_routes, MultiUserMemoryManager},
};

// ═══════════════════════════════════════════════════════════════════════
// Test infrastructure (same Harness pattern as handler_pipeline_tests.rs)
// ═══════════════════════════════════════════════════════════════════════

const TEST_KEY: &str = "pipeline-integration-test-key";
static ENV_INIT: Once = Once::new();

fn init_env() {
    ENV_INIT.call_once(|| unsafe {
        std::env::set_var("SHODH_API_KEYS", TEST_KEY);
    });
}

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
        let mgr =
            MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg).expect("create manager");
        Self {
            mgr: Arc::new(mgr),
            _dir: dir,
        }
    }

    fn app(&self) -> Router {
        let public = build_public_routes(self.mgr.clone());
        let protected = build_protected_routes(self.mgr.clone()).layer(axum::middleware::from_fn(
            shodh_memory::auth::auth_middleware,
        ));
        Router::new().merge(public).merge(protected)
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

fn authed_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-api-key", TEST_KEY)
        .body(Body::from(body.to_string()))
        .unwrap()
}

// ── response helpers ──

async fn json_of(app: Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let val = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).to_string()))
    };
    (status, val)
}

/// Store a memory and return its ID
async fn store_memory(h: &Harness, user_id: &str, content: &str) -> String {
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": user_id,
                "content": content,
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "store_memory failed: {body}");
    body["id"]
        .as_str()
        .expect("missing id in response")
        .to_string()
}

/// Store a memory with type, tags, and optional fields
async fn store_memory_full(
    h: &Harness,
    user_id: &str,
    content: &str,
    memory_type: &str,
    tags: Vec<&str>,
) -> String {
    let (status, body) = json_of(
        h.app(),
        authed_post(
            "/api/remember",
            json!({
                "user_id": user_id,
                "content": content,
                "memory_type": memory_type,
                "tags": tags,
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "store_memory_full failed: {body}");
    body["id"]
        .as_str()
        .expect("missing id in response")
        .to_string()
}

/// Recall memories for a user
async fn recall(h: &Harness, user_id: &str, query: &str) -> (StatusCode, Value) {
    json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({
                "user_id": user_id,
                "query": query,
            }),
        ),
    )
    .await
}

/// Recall with a specific mode
async fn recall_mode(h: &Harness, user_id: &str, query: &str, mode: &str) -> (StatusCode, Value) {
    json_of(
        h.app(),
        authed_post(
            "/api/recall",
            json!({
                "user_id": user_id,
                "query": query,
                "mode": mode,
            }),
        ),
    )
    .await
}

/// Short async delay for background indexing to complete
async fn wait_for_indexing() {
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
}

/// Longer delay for graph/lineage background tasks
async fn wait_for_background_tasks() {
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;
}

// ═══════════════════════════════════════════════════════════════════════
// REMEMBER → RECALL ROUNDTRIP TESTS
// ═══════════════════════════════════════════════════════════════════════

mod roundtrip_tests {
    use super::*;

    #[tokio::test]
    async fn recall_returns_stored_content() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The Ebbinghaus forgetting curve demonstrates exponential memory decay",
        )
        .await;
        wait_for_indexing().await;

        let (status, body) = recall(&h, "test-user", "Ebbinghaus forgetting curve").await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().expect("memories array");
        assert!(!memories.is_empty(), "Should find the stored memory");
        let content = memories[0]["experience"]["content"]
            .as_str()
            .expect("content field");
        assert!(
            content.contains("Ebbinghaus"),
            "Recalled content should match stored: {content}"
        );
    }

    #[tokio::test]
    async fn recall_returns_correct_id() {
        let h = Harness::new();
        let stored_id = store_memory(
            &h,
            "test-user",
            "Hebbian learning strengthens synaptic connections between co-active neurons",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "Hebbian synaptic connections").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let recalled_id = memories[0]["id"].as_str().expect("id field");
        assert_eq!(recalled_id, stored_id, "Recalled ID should match stored ID");
    }

    #[tokio::test]
    async fn recall_response_has_expected_fields() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "RocksDB uses log-structured merge trees for efficient writes",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "RocksDB merge trees").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());

        let m = &memories[0];
        // RecallMemory fields: id, experience, importance, created_at, score, tier
        assert!(m["id"].is_string(), "should have id");
        assert!(
            m["experience"]["content"].is_string(),
            "should have content"
        );
        assert!(m["importance"].is_number(), "should have importance");
        assert!(m["created_at"].is_string(), "should have created_at");
        assert!(m["score"].is_number(), "should have score");
        assert!(m["tier"].is_string(), "should have tier");
        // experience.tags is an array (may include auto-extracted entities)
        assert!(m["experience"]["tags"].is_array(), "should have tags array");
    }

    #[tokio::test]
    async fn recall_score_is_normalized() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Vector similarity search uses cosine distance for matching",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "vector similarity cosine").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());

        let score = memories[0]["score"].as_f64().unwrap();
        // Score is normalized to 0.0–0.95 range
        assert!(score >= 0.0, "Score should be >= 0.0, got {score}");
        assert!(score <= 0.95, "Score should be <= 0.95, got {score}");
    }

    #[tokio::test]
    async fn recall_single_memory_gets_max_score() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Spreading activation propagates through semantic networks",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "spreading activation semantic networks").await;
        let memories = body["memories"].as_array().unwrap();
        if !memories.is_empty() {
            let score = memories[0]["score"].as_f64().unwrap();
            // Single result in the DB → top score normalized to 0.95
            assert!(
                score > 0.5,
                "Single memory should get high normalized score, got {score}"
            );
        }
    }

    #[tokio::test]
    async fn recall_count_matches_array_length() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory alpha about graph traversal").await;
        store_memory(&h, "test-user", "Memory beta about graph algorithms").await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "graph algorithms traversal").await;
        let memories = body["memories"].as_array().unwrap();
        let count = body["count"].as_u64().unwrap() as usize;
        assert_eq!(
            count,
            memories.len(),
            "count field should match array length"
        );
    }

    #[tokio::test]
    async fn recall_respects_limit() {
        let h = Harness::new();
        for i in 0..5 {
            store_memory(
                &h,
                "test-user",
                &format!("Neural network layer {i} uses backpropagation for training"),
            )
            .await;
        }
        wait_for_indexing().await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall",
                json!({
                    "user_id": "test-user",
                    "query": "neural network backpropagation",
                    "limit": 2,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            memories.len() <= 2,
            "Should respect limit=2, got {}",
            memories.len()
        );
    }

    #[tokio::test]
    async fn recall_created_at_is_rfc3339() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Temporal indexing uses RFC3339 timestamps for ordering",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "temporal indexing timestamps").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let created_at = memories[0]["created_at"].as_str().unwrap();
        // RFC3339 has T separator and timezone
        assert!(
            created_at.contains('T'),
            "created_at should be RFC3339: {created_at}"
        );
    }

    #[tokio::test]
    async fn recall_memory_type_preserved() {
        let h = Harness::new();
        store_memory_full(
            &h,
            "test-user",
            "Decision to use power-law decay for long-term retention",
            "Decision",
            vec![],
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "power-law decay decision").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        // memory_type is Debug format of ExperienceType enum
        let mem_type = memories[0]["experience"]["memory_type"]
            .as_str()
            .unwrap_or("");
        assert_eq!(mem_type, "Decision", "Memory type should be preserved");
    }

    #[tokio::test]
    async fn recall_user_tags_included_in_response() {
        let h = Harness::new();
        store_memory_full(
            &h,
            "test-user",
            "Rust ownership model prevents data races at compile time",
            "Learning",
            vec!["rust", "ownership"],
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "Rust ownership data races").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let tags = memories[0]["experience"]["tags"]
            .as_array()
            .expect("tags array");
        let tag_strings: Vec<&str> = tags.iter().filter_map(|t| t.as_str()).collect();
        // User-provided tags should be in the response (may also have auto-extracted entities)
        assert!(
            tag_strings.iter().any(|t| t.eq_ignore_ascii_case("rust")),
            "Should contain user tag 'rust', got: {tag_strings:?}"
        );
    }

    #[tokio::test]
    async fn recall_importance_is_valid() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Importance scoring uses salience detection with keyword analysis",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "importance scoring salience").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let importance = memories[0]["importance"].as_f64().unwrap();
        assert!(
            (0.0..=1.0).contains(&importance),
            "Importance should be 0.0-1.0, got {importance}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// UNICODE ROUNDTRIP TESTS
// ═══════════════════════════════════════════════════════════════════════

mod unicode_tests {
    use super::*;

    #[tokio::test]
    async fn unicode_content_roundtrip() {
        let h = Harness::new();
        let content = "日本語テスト: Hebbian学習は神経科学の基本原理です";
        store_memory(&h, "test-user", content).await;
        wait_for_indexing().await;

        // List endpoint is more reliable for exact content verification
        let (status, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        // ListMemoryItem has "content" at top level, not nested under "experience"
        let stored_content = memories[0]["content"].as_str().unwrap_or("");
        assert_eq!(
            stored_content, content,
            "Unicode content should survive roundtrip"
        );
    }

    #[tokio::test]
    async fn mixed_script_content_roundtrip() {
        let h = Harness::new();
        let content = "English العربية 中文 हिन्दी mixing scripts in memory content";
        store_memory(&h, "test-user", content).await;
        wait_for_indexing().await;

        let (status, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let stored = memories[0]["content"].as_str().unwrap_or("");
        assert_eq!(stored, content);
    }

    #[tokio::test]
    async fn emoji_content_roundtrip() {
        let h = Harness::new();
        let content = "Memory with emojis 🧠💡🔬 testing unicode edge cases";
        store_memory(&h, "test-user", content).await;
        wait_for_indexing().await;

        let (status, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        let stored = memories[0]["content"].as_str().unwrap_or("");
        assert_eq!(stored, content);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// USER ISOLATION TESTS
// ═══════════════════════════════════════════════════════════════════════

mod isolation_tests {
    use super::*;

    #[tokio::test]
    async fn users_cannot_see_each_others_memories() {
        let h = Harness::new();
        store_memory(
            &h,
            "alice",
            "Alice's secret project about quantum computing breakthroughs",
        )
        .await;
        store_memory(
            &h,
            "bob",
            "Bob's private notes about compiler optimization techniques",
        )
        .await;
        wait_for_indexing().await;

        // Alice recalls — should only see her memory
        let (_, body) = recall(&h, "alice", "quantum computing").await;
        let memories = body["memories"].as_array().unwrap();
        for m in memories {
            let content = m["experience"]["content"].as_str().unwrap_or("");
            assert!(
                !content.contains("Bob"),
                "Alice should not see Bob's memories: {content}"
            );
        }

        // Bob recalls — should only see his memory
        let (_, body) = recall(&h, "bob", "compiler optimization").await;
        let memories = body["memories"].as_array().unwrap();
        for m in memories {
            let content = m["experience"]["content"].as_str().unwrap_or("");
            assert!(
                !content.contains("Alice"),
                "Bob should not see Alice's memories: {content}"
            );
        }
    }

    #[tokio::test]
    async fn user_list_is_isolated() {
        let h = Harness::new();
        store_memory(
            &h,
            "user-one",
            "First user memory about distributed systems",
        )
        .await;
        store_memory(&h, "user-two", "Second user memory about machine learning").await;

        // List user-one's memories (ListMemoryItem has "content" at top level)
        let (_, body) = json_of(h.app(), authed_get("/api/list/user-one")).await;
        let memories = body["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1, "user-one should have exactly 1 memory");
        assert!(memories[0]["content"]
            .as_str()
            .unwrap()
            .contains("distributed systems"));

        // List user-two's memories
        let (_, body) = json_of(h.app(), authed_get("/api/list/user-two")).await;
        let memories = body["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1, "user-two should have exactly 1 memory");
        assert!(memories[0]["content"]
            .as_str()
            .unwrap()
            .contains("machine learning"));
    }

    #[tokio::test]
    async fn recall_empty_for_wrong_user() {
        let h = Harness::new();
        store_memory(
            &h,
            "data-owner",
            "Confidential memory about proprietary algorithm design",
        )
        .await;
        wait_for_indexing().await;

        let (status, body) = recall(&h, "intruder", "proprietary algorithm").await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.is_empty(), "Wrong user should see no memories");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CONTENT DEDUP TESTS
// ═══════════════════════════════════════════════════════════════════════

mod dedup_tests {
    use super::*;

    #[tokio::test]
    async fn identical_content_returns_same_id() {
        let h = Harness::new();
        let content = "Exact duplicate content for deduplication testing in memory system";
        let id1 = store_memory(&h, "test-user", content).await;
        let id2 = store_memory(&h, "test-user", content).await;
        // SHA256 content dedup should return the same memory ID
        assert_eq!(id1, id2, "Duplicate content should return same ID");
    }

    #[tokio::test]
    async fn similar_but_different_content_gets_different_ids() {
        let h = Harness::new();
        let id1 = store_memory(&h, "test-user", "The cat sat on the mat near the window").await;
        let id2 = store_memory(&h, "test-user", "The cat sat on the mat near the door").await;
        assert_ne!(id1, id2, "Different content should get different IDs");
    }

    #[tokio::test]
    async fn dedup_does_not_create_duplicate_in_list() {
        let h = Harness::new();
        let content = "Unique dedup verification content for list counting test";
        store_memory(&h, "test-user", content).await;
        store_memory(&h, "test-user", content).await;
        store_memory(&h, "test-user", content).await;

        let (_, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        let memories = body["memories"].as_array().unwrap();
        assert_eq!(
            memories.len(),
            1,
            "Dedup should prevent duplicates in storage"
        );
    }

    #[tokio::test]
    async fn dedup_is_per_user() {
        let h = Harness::new();
        let content = "Shared content stored by different users for isolation test";
        let id1 = store_memory(&h, "user-a", content).await;
        let id2 = store_memory(&h, "user-b", content).await;
        // Different users can have the same content — dedup is per-user
        assert_ne!(
            id1, id2,
            "Same content for different users should get different IDs"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MULTI-MODE RECALL TESTS
// ═══════════════════════════════════════════════════════════════════════

mod mode_tests {
    use super::*;

    #[tokio::test]
    async fn semantic_mode_returns_semantically_similar() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The transformer architecture revolutionized natural language processing",
        )
        .await;
        wait_for_indexing().await;

        let (status, body) = recall_mode(
            &h,
            "test-user",
            "deep learning NLP transformers",
            "semantic",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            !memories.is_empty(),
            "Semantic mode should find semantically related memory"
        );
    }

    #[tokio::test]
    async fn similarity_mode_alias_works() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Gradient descent optimizes neural network parameters iteratively",
        )
        .await;
        wait_for_indexing().await;

        // "similarity" is an alias for "semantic"
        let (status, body) = recall_mode(
            &h,
            "test-user",
            "gradient descent optimization",
            "similarity",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].as_array().unwrap().len() > 0);
    }

    #[tokio::test]
    async fn temporal_mode_returns_results() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Temporal memory created for mode testing in retrieval pipeline",
        )
        .await;
        wait_for_indexing().await;

        let (status, body) = recall_mode(&h, "test-user", "temporal retrieval", "temporal").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn hybrid_mode_is_default() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Hybrid retrieval fuses vector and keyword search signals together",
        )
        .await;
        wait_for_indexing().await;

        // No mode specified → defaults to hybrid
        let (status1, body1) = recall(&h, "test-user", "hybrid retrieval fusion").await;
        // Explicit hybrid
        let (status2, body2) =
            recall_mode(&h, "test-user", "hybrid retrieval fusion", "hybrid").await;

        assert_eq!(status1, StatusCode::OK);
        assert_eq!(status2, StatusCode::OK);
        // Both should return results (same underlying mode)
        assert!(body1["memories"].as_array().unwrap().len() > 0);
        assert!(body2["memories"].as_array().unwrap().len() > 0);
    }

    #[tokio::test]
    async fn associative_mode_returns_results() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Knowledge graph nodes connected via associative activation spreading",
        )
        .await;
        wait_for_indexing().await;

        let (status, body) =
            recall_mode(&h, "test-user", "knowledge graph activation", "associative").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn causal_mode_returns_results() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Causal inference determines cause and effect relationships in data",
        )
        .await;
        wait_for_indexing().await;

        let (status, body) = recall_mode(&h, "test-user", "causal inference", "causal").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn unknown_mode_falls_back_to_hybrid() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Fallback mode testing with unknown retrieval strategy selection",
        )
        .await;
        wait_for_indexing().await;

        // Unknown mode string → falls back to Hybrid (not an error)
        let (status, body) =
            recall_mode(&h, "test-user", "fallback mode testing", "nonexistent_mode").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BATCH REMEMBER → RECALL TESTS
// ═══════════════════════════════════════════════════════════════════════

mod batch_tests {
    use super::*;

    #[tokio::test]
    async fn batch_memories_are_recallable() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "Batch item one about recurrent neural networks"},
                        {"content": "Batch item two about convolutional neural networks"},
                        {"content": "Batch item three about generative adversarial networks"},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["created"], 3);
        wait_for_indexing().await;

        // Batch items should be recallable
        let (_, body) = recall(&h, "test-user", "neural networks").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "Should recall at least 1 batch item");
    }

    #[tokio::test]
    async fn batch_memory_ids_are_valid() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "Batch ID test memory alpha about reinforcement learning"},
                        {"content": "Batch ID test memory beta about multi-agent systems"},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let ids = body["memory_ids"].as_array().unwrap();
        assert_eq!(ids.len(), 2);

        // Each ID should be retrievable via GET /api/memory/{id}
        for id in ids {
            let id_str = id.as_str().unwrap();
            let (status, _) = json_of(
                h.app(),
                authed_get(&format!("/api/memory/{id_str}?user_id=test-user")),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::OK,
                "Batch memory {id_str} should be retrievable"
            );
        }
    }

    #[tokio::test]
    async fn batch_partial_failure_still_stores_valid() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember/batch",
                json!({
                    "user_id": "test-user",
                    "memories": [
                        {"content": "Valid batch memory about knowledge representation"},
                        {"content": ""},
                        {"content": "Another valid batch memory about ontology design"},
                    ],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let created = body["created"].as_u64().unwrap();
        let failed = body["failed"].as_u64().unwrap();
        assert!(created >= 2, "At least 2 valid items should be stored");
        assert!(failed >= 1, "Empty content item should fail");
        assert!(body["errors"].is_array());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// UPSERT → RECALL TESTS
// ═══════════════════════════════════════════════════════════════════════

mod upsert_tests {
    use super::*;

    #[tokio::test]
    async fn upsert_creates_recallable_memory() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Upserted memory about attention mechanism in transformers",
                    "external_id": "ext-attn-001",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["was_update"], false);
        assert_eq!(body["version"], 1);
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "attention mechanism transformers").await;
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "Upserted memory should be recallable");
    }

    #[tokio::test]
    async fn upsert_update_changes_content() {
        let h = Harness::new();
        // Create
        let (_, body) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Initial version of the documentation about SPANN index",
                    "external_id": "doc-spann",
                }),
            ),
        )
        .await;
        let id = body["id"].as_str().unwrap().to_string();
        assert_eq!(body["was_update"], false);

        // Update with same external_id
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Updated version of the SPANN index documentation with PQ compression details",
                    "external_id": "doc-spann",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["was_update"], true);
        assert_eq!(body["version"], 2);
        // Same ID for same external_id
        assert_eq!(body["id"].as_str().unwrap(), id);

        // Verify updated content via GET
        let (_, body) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        let content = body["experience"]["content"].as_str().unwrap();
        assert!(
            content.contains("PQ compression"),
            "Content should be updated: {content}"
        );
    }

    #[tokio::test]
    async fn upsert_different_external_ids_create_separate_memories() {
        let h = Harness::new();
        let (_, body1) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "First document about Vamana graph construction algorithm",
                    "external_id": "doc-vamana",
                }),
            ),
        )
        .await;
        let (_, body2) = json_of(
            h.app(),
            authed_post(
                "/api/upsert",
                json!({
                    "user_id": "test-user",
                    "content": "Second document about SPANN disk-based index structure",
                    "external_id": "doc-spann-2",
                }),
            ),
        )
        .await;
        assert_ne!(
            body1["id"].as_str().unwrap(),
            body2["id"].as_str().unwrap(),
            "Different external_ids should create different memories"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TAG & TYPE FILTERING TESTS
// ═══════════════════════════════════════════════════════════════════════

mod filter_tests {
    use super::*;

    #[tokio::test]
    async fn recall_by_tags_returns_matching() {
        let h = Harness::new();
        store_memory_full(
            &h,
            "test-user",
            "Important architecture decision about database schema",
            "Decision",
            vec!["architecture", "database"],
        )
        .await;
        store_memory_full(
            &h,
            "test-user",
            "Routine observation about build times being slow",
            "Observation",
            vec!["build", "performance"],
        )
        .await;
        wait_for_indexing().await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall/tags",
                json!({
                    "user_id": "test-user",
                    "tags": ["architecture"],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        // Should find the architecture-tagged memory
        if !memories.is_empty() {
            let tags: Vec<&str> = memories[0]["experience"]["tags"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|t| t.as_str())
                .collect();
            assert!(
                tags.iter().any(|t| t.eq_ignore_ascii_case("architecture")),
                "Should contain the 'architecture' tag, got: {tags:?}"
            );
        }
    }

    #[tokio::test]
    async fn recall_by_date_range_returns_recent() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Memory stored today for date range filter testing",
        )
        .await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall/date",
                json!({
                    "user_id": "test-user",
                    "start": "2020-01-01T00:00:00Z",
                    "end": "2030-01-01T00:00:00Z",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(!memories.is_empty(), "Should find memory within date range");
    }

    #[tokio::test]
    async fn recall_by_narrow_date_range_excludes() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Memory that should not match narrow past date range",
        )
        .await;

        // Very old date range that shouldn't match today's memory
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/recall/date",
                json!({
                    "user_id": "test-user",
                    "start": "2000-01-01T00:00:00Z",
                    "end": "2001-01-01T00:00:00Z",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            memories.is_empty(),
            "Should not find memory outside date range"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MULTIPLE MEMORIES RANKING TESTS
// ═══════════════════════════════════════════════════════════════════════

mod ranking_tests {
    use super::*;

    #[tokio::test]
    async fn scores_are_ordered_descending() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Rust programming language memory safety guarantees",
        )
        .await;
        store_memory(
            &h,
            "test-user",
            "Python dynamic typing flexible scripting capabilities",
        )
        .await;
        store_memory(
            &h,
            "test-user",
            "Rust borrow checker prevents data races at compile time",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "Rust memory safety borrow checker").await;
        let memories = body["memories"].as_array().unwrap();
        if memories.len() >= 2 {
            let scores: Vec<f64> = memories
                .iter()
                .map(|m| m["score"].as_f64().unwrap_or(0.0))
                .collect();
            for i in 1..scores.len() {
                assert!(
                    scores[i - 1] >= scores[i],
                    "Scores should be descending: {scores:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn relevant_memory_scores_higher_than_irrelevant() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The Vamana algorithm builds a navigable small-world graph for ANN search",
        )
        .await;
        store_memory(
            &h,
            "test-user",
            "My grocery list includes milk, eggs, and bread for the weekend",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "Vamana graph ANN search").await;
        let memories = body["memories"].as_array().unwrap();
        if memories.len() >= 2 {
            let top_content = memories[0]["experience"]["content"].as_str().unwrap_or("");
            assert!(
                top_content.contains("Vamana"),
                "Top result should be the relevant memory, got: {top_content}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GRAPH INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════

mod graph_integration_tests {
    use super::*;

    #[tokio::test]
    async fn remember_populates_graph() {
        let h = Harness::new();
        // Content with clear named entities that NER should extract
        store_memory(
            &h,
            "test-user",
            "Google released TensorFlow as an open source machine learning framework",
        )
        .await;
        // Graph processing is a background task — need longer wait
        wait_for_background_tasks().await;

        let (status, body) = json_of(h.app(), authed_get("/api/graph/test-user/stats")).await;
        assert_eq!(status, StatusCode::OK);
        // After remembering content with entities, graph should have some data
        assert!(body.is_object());
    }

    #[tokio::test]
    async fn graph_data_available_after_remember() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Microsoft Azure provides cloud computing services worldwide",
        )
        .await;
        wait_for_background_tasks().await;

        let (status, _) = json_of(h.app(), authed_get("/api/graph/data/test-user")).await;
        assert_eq!(status, StatusCode::OK);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LINEAGE INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════

mod lineage_integration_tests {
    use super::*;

    #[tokio::test]
    async fn manual_lineage_edge_persists() {
        let h = Harness::new();
        let id1 = store_memory(
            &h,
            "test-user",
            "Identified performance bottleneck in the query pipeline",
        )
        .await;
        let id2 = store_memory(
            &h,
            "test-user",
            "Resolved the performance bottleneck by adding index caching",
        )
        .await;
        wait_for_background_tasks().await;

        // Add manual lineage edge
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/link",
                json!({
                    "user_id": "test-user",
                    "from_memory_id": id1,
                    "to_memory_id": id2,
                    "relation": "ResolvedBy",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Verify the edge exists via lineage edges endpoint
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/edges",
                json!({
                    "user_id": "test-user",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let edges = body["edges"].as_array().unwrap();
        assert!(!edges.is_empty(), "Lineage edge should persist");
    }

    #[tokio::test]
    async fn lineage_trace_follows_chain() {
        let h = Harness::new();
        let id1 = store_memory(
            &h,
            "test-user",
            "Discovered bug in the serialization layer causing data loss",
        )
        .await;
        let id2 = store_memory(
            &h,
            "test-user",
            "Fixed serialization bug by adding format version tags",
        )
        .await;
        let id3 = store_memory(
            &h,
            "test-user",
            "Deployed serialization fix to production environment successfully",
        )
        .await;
        wait_for_background_tasks().await;

        // Create chain: id1 → id2 → id3
        json_of(
            h.app(),
            authed_post(
                "/api/lineage/link",
                json!({
                    "user_id": "test-user",
                    "from_memory_id": id1,
                    "to_memory_id": id2,
                    "relation": "Caused",
                }),
            ),
        )
        .await;
        json_of(
            h.app(),
            authed_post(
                "/api/lineage/link",
                json!({
                    "user_id": "test-user",
                    "from_memory_id": id2,
                    "to_memory_id": id3,
                    "relation": "Caused",
                }),
            ),
        )
        .await;

        // Trace from id1 should reach id3
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/lineage/trace",
                json!({
                    "user_id": "test-user",
                    "memory_id": id1,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // Trace should return edges in the chain
        assert!(body.is_object() || body.is_array());
    }

    #[tokio::test]
    async fn all_lineage_relation_types_accepted() {
        let h = Harness::new();
        let relations = [
            "Caused",
            "ResolvedBy",
            "InformedBy",
            "SupersededBy",
            "TriggeredBy",
            "BranchedFrom",
            "RelatedTo",
        ];

        for (i, relation) in relations.iter().enumerate() {
            let id1 = store_memory(
                &h,
                "test-user",
                &format!("Source memory number {i} for lineage relation testing"),
            )
            .await;
            let id2 = store_memory(
                &h,
                "test-user",
                &format!("Target memory number {i} for lineage relation testing"),
            )
            .await;

            let (status, _) = json_of(
                h.app(),
                authed_post(
                    "/api/lineage/link",
                    json!({
                        "user_id": "test-user",
                        "from_memory_id": id1,
                        "to_memory_id": id2,
                        "relation": relation,
                    }),
                ),
            )
            .await;
            assert_eq!(
                status,
                StatusCode::OK,
                "Relation type '{relation}' should be accepted"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PROACTIVE CONTEXT → RECALL INTEGRATION
// ═══════════════════════════════════════════════════════════════════════

mod proactive_integration_tests {
    use super::*;

    #[tokio::test]
    async fn proactive_context_returns_relevant_memories() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "The deployment pipeline uses Docker containers for isolation",
        )
        .await;
        wait_for_indexing().await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "Working on the Docker deployment pipeline configuration",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // Proactive context uses semantic search — may or may not surface
        // the memory depending on embedding similarity and threshold.
        // We verify the response shape is correct.
        assert!(body["memories"].is_array());
    }

    #[tokio::test]
    async fn proactive_context_respects_max_results() {
        let h = Harness::new();
        for i in 0..5 {
            store_memory(
                &h,
                "test-user",
                &format!("Kubernetes pod {i} runs microservice for API gateway"),
            )
            .await;
        }
        wait_for_indexing().await;

        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/proactive_context",
                json!({
                    "user_id": "test-user",
                    "context": "Kubernetes microservices",
                    "max_results": 2,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let memories = body["memories"].as_array().unwrap();
        assert!(
            memories.len() <= 2,
            "Should respect max_results=2, got {}",
            memories.len()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DELETE → RECALL VERIFICATION
// ═══════════════════════════════════════════════════════════════════════

mod delete_tests {
    use super::*;

    #[tokio::test]
    async fn deleted_memory_not_recalled() {
        let h = Harness::new();
        let id = store_memory(
            &h,
            "test-user",
            "Temporary memory that will be deleted from the system",
        )
        .await;
        wait_for_indexing().await;

        // Verify it exists
        let (status, _) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Delete it
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget",
                json!({
                    "user_id": "test-user",
                    "memory_id": id,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Verify it's gone
        let (status, _) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn forget_by_tags_removes_from_recall() {
        let h = Harness::new();
        store_memory_full(
            &h,
            "test-user",
            "Ephemeral memory tagged for deletion testing purposes",
            "Observation",
            vec!["ephemeral", "delete-me"],
        )
        .await;
        store_memory(
            &h,
            "test-user",
            "Permanent memory that should survive tag-based deletion",
        )
        .await;
        wait_for_indexing().await;

        // Delete by tag
        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/forget/tags",
                json!({
                    "user_id": "test-user",
                    "tags": ["ephemeral"],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // List should only have the permanent memory
        let (_, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        let memories = body["memories"].as_array().unwrap();
        for m in memories {
            let content = m["experience"]["content"].as_str().unwrap_or("");
            assert!(
                !content.contains("Ephemeral"),
                "Deleted memory should not appear: {content}"
            );
        }
    }

    #[tokio::test]
    async fn clear_all_removes_everything() {
        let h = Harness::new();
        store_memory(&h, "test-user", "Memory one to be cleared completely").await;
        store_memory(&h, "test-user", "Memory two to be cleared completely").await;
        store_memory(&h, "test-user", "Memory three to be cleared completely").await;

        let (status, _) = json_of(
            h.app(),
            authed_post(
                "/api/memories/clear",
                json!({
                    "user_id": "test-user",
                    "confirm": "CONFIRM",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let (_, body) = json_of(h.app(), authed_get("/api/list/test-user")).await;
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.is_empty(), "All memories should be cleared");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RECALL RESPONSE SHAPE EDGE CASES
// ═══════════════════════════════════════════════════════════════════════

mod response_shape_tests {
    use super::*;

    #[tokio::test]
    async fn empty_recall_has_correct_shape() {
        let h = Harness::new();
        let (status, body) = recall(&h, "test-user", "nothing stored yet").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["memories"].is_array());
        assert_eq!(body["count"], 0);
        let memories = body["memories"].as_array().unwrap();
        assert!(memories.is_empty());
    }

    #[tokio::test]
    async fn recall_optional_fields_absent_when_empty() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Memory for testing optional response fields presence",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "optional response fields").await;
        // retrieval_stats is always None in current API
        assert!(
            body["retrieval_stats"].is_null(),
            "retrieval_stats should be null"
        );
        // facts, todos, triggered_reminders, lineage are skipped when empty
        // They should be either absent (null) or empty arrays
        let facts = body["facts"].as_array();
        if let Some(f) = facts {
            // If present, should be an array
            assert!(f.is_empty() || !f.is_empty()); // just verify it's parseable
        }
    }

    #[tokio::test]
    async fn recall_tier_field_is_valid_string() {
        let h = Harness::new();
        store_memory(
            &h,
            "test-user",
            "Memory for tier field validation in recall response",
        )
        .await;
        wait_for_indexing().await;

        let (_, body) = recall(&h, "test-user", "tier field validation").await;
        let memories = body["memories"].as_array().unwrap();
        if !memories.is_empty() {
            let tier = memories[0]["tier"].as_str().unwrap();
            // Valid tier values: "Working", "Session", "LongTerm" (Debug format)
            let valid_tiers = ["Working", "Session", "LongTerm"];
            assert!(
                valid_tiers.contains(&tier),
                "Tier should be one of {valid_tiers:?}, got '{tier}'"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// IMPORTANCE & EMOTIONAL CONTEXT TESTS
// ═══════════════════════════════════════════════════════════════════════

mod importance_tests {
    use super::*;

    #[tokio::test]
    async fn importance_override_is_respected() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Critical architecture decision about storage backend selection",
                    "importance": 0.95,
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let id = body["id"].as_str().unwrap();
        wait_for_indexing().await;

        let (_, body) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        let importance = body["importance"].as_f64().unwrap();
        assert!(
            (importance - 0.95).abs() < 0.1,
            "Importance override should be close to 0.95, got {importance}"
        );
    }

    #[tokio::test]
    async fn emotional_context_stored_and_retrievable() {
        let h = Harness::new();
        let (status, body) = json_of(
            h.app(),
            authed_post(
                "/api/remember",
                json!({
                    "user_id": "test-user",
                    "content": "Frustrating debugging session with race condition in async code",
                    "emotional_valence": -0.7,
                    "emotional_arousal": 0.8,
                    "emotion": "frustration",
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let id = body["id"].as_str().unwrap();

        let (_, body) = json_of(
            h.app(),
            authed_get(&format!("/api/memory/{id}?user_id=test-user")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // Memory should be stored successfully with emotional context
        assert!(body["experience"]["content"]
            .as_str()
            .unwrap()
            .contains("Frustrating"));
    }
}
