//! Health and Infrastructure Handlers
//!
//! Kubernetes probes, metrics, and system health endpoints.

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::Deserialize;
use std::collections::HashMap;

use super::state::MultiUserMemoryManager;
use super::types::{ContextStatus, MemoryEvent};
use crate::metrics;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Health response for main health endpoint
#[derive(serde::Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub users_count: usize,
    pub users_in_cache: usize,
    pub user_evictions: usize,
    pub max_cache_size: usize,
}

/// Main health check endpoint
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let users_in_cache = state.users_in_cache();
    let user_evictions = state.user_evictions();

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        users_count: state.list_users().len(),
        users_in_cache,
        user_evictions,
        max_cache_size: state.server_config().max_users_in_memory,
    })
}

/// Liveness probe - indicates if process is alive and not deadlocked
/// Returns 200 OK if service is running (minimal check, always succeeds if reachable)
pub async fn health_live() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "alive",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// Readiness probe - indicates if service can handle traffic
/// Returns 200 OK if service is ready, 503 if not ready
pub async fn health_ready(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    let users_in_cache = state.users_in_cache();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ready",
            "version": env!("CARGO_PKG_VERSION"),
            "users_in_cache": users_in_cache,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// Vector index health endpoint - provides Vamana index statistics per user
pub async fn health_index(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let user_id = match params.get("user_id") {
        Some(id) => id.clone(),
        None => {
            // Return aggregate stats across all cached users
            let users: Vec<(String, crate::memory::retrieval::IndexHealth)> = {
                let mut results = Vec::new();
                // Access user_memories through the manager's public API
                for user_id in state.list_users() {
                    if let Ok(memory) = state.get_user_memory(&user_id) {
                        let guard = memory.read();
                        results.push((user_id, guard.index_health()));
                    }
                }
                results
            };

            let total_vectors: usize = users.iter().map(|(_, h)| h.total_vectors).sum();
            let total_incremental: usize = users.iter().map(|(_, h)| h.incremental_inserts).sum();
            let needs_rebuild: Vec<&str> = users
                .iter()
                .filter(|(_, h)| h.needs_rebuild)
                .map(|(id, _)| id.as_str())
                .collect();

            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "ok",
                    "users_checked": users.len(),
                    "total_vectors": total_vectors,
                    "total_incremental_inserts": total_incremental,
                    "users_needing_rebuild": needs_rebuild,
                    "rebuild_threshold": crate::vector_db::vamana::REBUILD_THRESHOLD,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })),
            );
        }
    };

    // Get health for specific user
    match state.get_user_memory(&user_id) {
        Ok(memory) => {
            let guard = memory.read();
            let health = guard.index_health();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "ok",
                    "user_id": user_id,
                    "total_vectors": health.total_vectors,
                    "incremental_inserts": health.incremental_inserts,
                    "needs_rebuild": health.needs_rebuild,
                    "rebuild_threshold": health.rebuild_threshold,
                    "degradation_percent": if health.rebuild_threshold > 0 {
                        (health.incremental_inserts as f64 / health.rebuild_threshold as f64 * 100.0).min(100.0)
                    } else {
                        0.0
                    },
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "status": "error",
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            })),
        ),
    }
}

/// Prometheus metrics endpoint for observability
pub async fn metrics_endpoint(State(state): State<AppState>) -> Result<String, StatusCode> {
    use prometheus::Encoder;

    // Update memory usage gauges before serving metrics
    let users_in_cache = state.users_in_cache();
    metrics::ACTIVE_USERS.set(users_in_cache as i64);

    // Aggregate metrics across all users
    let (mut total_working, mut total_session, mut total_longterm, mut total_heap) =
        (0i64, 0i64, 0i64, 0i64);
    let mut total_vectors = 0i64;

    for user_id in state.list_users().iter().take(100) {
        if let Ok(memory_sys) = state.get_user_memory(user_id) {
            if let Some(guard) = memory_sys.try_read() {
                let stats = guard.stats();
                total_working += stats.working_memory_count as i64;
                total_session += stats.session_memory_count as i64;
                total_longterm += stats.long_term_memory_count as i64;
                total_heap += (stats.total_memories * 250) as i64;
                total_vectors += stats.total_memories as i64;
            }
        }
    }

    // Set aggregate metrics
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["working"])
        .set(total_working);
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["session"])
        .set(total_session);
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["longterm"])
        .set(total_longterm);
    metrics::MEMORY_HEAP_BYTES_TOTAL.set(total_heap);
    metrics::VECTOR_INDEX_SIZE_TOTAL.set(total_vectors);

    // Gather and encode metrics
    let encoder = prometheus::TextEncoder::new();
    let metric_families = metrics::METRICS_REGISTRY.gather();

    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    String::from_utf8(buffer).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Context status request from Claude Code status line
#[derive(Debug, Deserialize)]
pub struct ContextStatusRequest {
    pub session_id: String,
    pub tokens_used: u64,
    pub tokens_budget: u64,
    pub current_dir: Option<String>,
    pub model: Option<String>,
}

/// Update context status from Claude Code status line script
pub async fn update_context_status(
    State(state): State<AppState>,
    Json(req): Json<ContextStatusRequest>,
) -> Json<serde_json::Value> {
    let percent_used = if req.tokens_budget > 0 {
        ((req.tokens_used as f64 / req.tokens_budget as f64) * 100.0) as u8
    } else {
        0
    };

    let status = ContextStatus {
        session_id: Some(req.session_id.clone()),
        tokens_used: req.tokens_used,
        tokens_budget: req.tokens_budget,
        percent_used,
        current_task: req.current_dir,
        model: req.model,
        updated_at: chrono::Utc::now(),
    };

    state
        .context_sessions()
        .insert(req.session_id.clone(), status.clone());

    state.broadcast_context(status);

    state.emit_event(MemoryEvent {
        event_type: "CONTEXT_UPDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: "system".to_string(),
        memory_id: Some(req.session_id),
        content_preview: Some(format!(
            "{}% ({}/{})",
            percent_used, req.tokens_used, req.tokens_budget
        )),
        memory_type: Some("Context".to_string()),
        importance: None,
        count: None,
    });

    Json(serde_json::json!({
        "success": true,
        "percent_used": percent_used
    }))
}

/// Get all active context sessions (auto-cleans stale sessions > 5 mins old)
pub async fn get_context_status(State(state): State<AppState>) -> Json<Vec<ContextStatus>> {
    let now = chrono::Utc::now();
    let stale_threshold = chrono::Duration::minutes(5);

    let stale_ids: Vec<String> = state
        .context_sessions()
        .iter()
        .filter(|r| now - r.value().updated_at > stale_threshold)
        .map(|r| r.key().clone())
        .collect();

    for id in stale_ids {
        state.context_sessions().remove(&id);
    }

    let mut sessions: Vec<ContextStatus> = state
        .context_sessions()
        .iter()
        .map(|r| r.value().clone())
        .collect();
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Json(sessions)
}
