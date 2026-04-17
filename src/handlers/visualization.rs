//! Visualization Handlers
//!
//! Handlers for brain state visualization and memory graph visualization.
//! Includes live browser-based graph visualization with SSE updates.

use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::GraphStats as VisualizationStats;
use crate::validation;
use std::sync::Arc;

/// Response extension carrying a CSP script nonce; read by the security_headers middleware.
#[derive(Debug, Clone)]
pub struct CspNonce(pub String);

type AppState = Arc<MultiUserMemoryManager>;

/// Brain state response with memories organized by tier
#[derive(Debug, Serialize)]
pub struct BrainStateResponse {
    pub working_memory: Vec<MemoryNeuron>,
    pub session_memory: Vec<MemoryNeuron>,
    pub longterm_memory: Vec<MemoryNeuron>,
    pub stats: BrainStats,
}

/// Individual memory neuron for visualization
#[derive(Debug, Serialize)]
pub struct MemoryNeuron {
    pub id: String,
    pub content_preview: String,
    pub activation: f32,
    pub importance: f32,
    pub tier: String,
    pub access_count: u32,
    pub created_at: String,
}

/// Brain statistics
#[derive(Debug, Serialize)]
pub struct BrainStats {
    pub total_memories: usize,
    pub working_count: usize,
    pub session_count: usize,
    pub longterm_count: usize,
    pub avg_activation: f32,
    pub avg_importance: f32,
}

/// GET /api/brain/{user_id} - Get brain state visualization
pub async fn get_brain_state(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<BrainStateResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let mut working_memory = Vec::new();
    let mut session_memory = Vec::new();
    let mut longterm_memory = Vec::new();
    let mut total_activation = 0.0f32;
    let mut total_importance = 0.0f32;

    // Get working memory
    for mem in memory_guard.get_working_memories() {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "working".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        working_memory.push(neuron);
    }

    // Get session memory
    for mem in memory_guard.get_session_memories() {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "session".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        session_memory.push(neuron);
    }

    // Get longterm memory sample
    let longterm_sample = memory_guard.get_longterm_memories(50).unwrap_or_default();
    for mem in longterm_sample {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "longterm".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        longterm_memory.push(neuron);
    }

    let total_count = working_memory.len() + session_memory.len() + longterm_memory.len();
    let stats = BrainStats {
        total_memories: total_count,
        working_count: working_memory.len(),
        session_count: session_memory.len(),
        longterm_count: longterm_memory.len(),
        avg_activation: if total_count > 0 {
            total_activation / total_count as f32
        } else {
            0.0
        },
        avg_importance: if total_count > 0 {
            total_importance / total_count as f32
        } else {
            0.0
        },
    };

    Ok(Json(BrainStateResponse {
        working_memory,
        session_memory,
        longterm_memory,
        stats,
    }))
}

/// GET /api/visualization/{user_id}/stats - Get visualization statistics
pub async fn get_visualization_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<VisualizationStats>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard.get_visualization_stats();

    Ok(Json(stats))
}

/// GET /api/visualization/{user_id}/dot - Export graph as DOT format
pub async fn get_visualization_dot(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<String, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let dot = memory_guard.export_visualization_dot();

    Ok(dot)
}

/// Request to build visualization
#[derive(Debug, Deserialize)]
pub struct BuildVisualizationRequest {
    pub user_id: String,
}

/// POST /api/visualization/build - Build visualization graph
pub async fn build_visualization(
    State(state): State<AppState>,
    Json(req): Json<BuildVisualizationRequest>,
) -> Result<Json<VisualizationStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard
        .build_visualization_graph()
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// Query parameters for graph view
#[derive(Debug, Deserialize)]
pub struct GraphViewParams {
    pub user_id: Option<String>,
}

/// Graph node for d3.js visualization
#[derive(Debug, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String, // "memory", "entity"
    pub tier: String,      // "L1", "L2", "L3" or memory tier
    pub strength: f32,
    pub size: f32,
}

/// Graph edge for d3.js visualization
#[derive(Debug, Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub edge_type: String,
    pub tier: String, // "L1", "L2", "L3"
    pub strength: f32,
}

/// Graph data response for d3.js
#[derive(Debug, Serialize)]
pub struct GraphDataResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub stats: GraphDataStats,
}

/// Graph statistics
#[derive(Debug, Serialize)]
pub struct GraphDataStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub l1_edges: usize,
    pub l2_edges: usize,
    pub l3_edges: usize,
}

const VIEWER_HTML: &str = include_str!("viewer/index.html");

/// GET /graph/view - Serve interactive graph visualization HTML
pub async fn graph_view(Query(params): Query<GraphViewParams>) -> Response {
    let user_id = params.user_id.unwrap_or_else(|| "default".to_string());
    let nonce = generate_nonce();
    let body = generate_graph_html(&user_id, &nonce);
    let mut response = Html(body).into_response();
    response.extensions_mut().insert(CspNonce(nonce));
    response
}

/// GET /graph/view2 - Serve the sigma.js GEXF viewer
pub async fn graph_view2(Query(params): Query<GraphViewParams>) -> Response {
    let user_id = params.user_id.unwrap_or_else(|| "default".to_string());
    let nonce = generate_nonce();

    // Match the security stance of `generate_graph_html`: never leak
    // a server API key into HTML served by a public route in production.
    let api_key = if crate::auth::is_production_mode() {
        String::new()
    } else {
        std::env::var("SHODH_DEV_API_KEY").unwrap_or_default()
    };

    let escaped_user = html_escape(&user_id);
    let escaped_key = html_escape(&api_key);

    let body = VIEWER_HTML
        .replace("{{NONCE}}", &nonce)
        .replace("{{API_KEY}}", &escaped_key)
        .replace("{{USER_ID}}", &escaped_user);

    let mut response = Html(body).into_response();
    response.extensions_mut().insert(CspNonce(nonce));
    response
}

/// GET /graph/assets/{file} - Serve vendored JS libraries (d3, three.js, OrbitControls, sigma, graphology)
pub async fn graph_asset(Path(file): Path<String>) -> Response {
    let bytes: &'static [u8] = match file.as_str() {
        "d3.v7.9.0.min.js" => include_bytes!("assets/d3.v7.9.0.min.js"),
        "graphology.umd.min.js" => include_bytes!("assets/graphology.umd.min.js"),
        "graphology-library.min.js" => include_bytes!("assets/graphology-library.min.js"),
        "sigma.min.js" => include_bytes!("assets/sigma.min.js"),
        "three.module.js" => include_bytes!("assets/three.module.js"),
        "three.core.js" => include_bytes!("assets/three.core.js"),
        "OrbitControls.js" => include_bytes!("assets/OrbitControls.js"),
        _ => return (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
        ],
        bytes,
    )
        .into_response()
}

/// Generate a 128-bit base64-encoded CSP nonce.
fn generate_nonce() -> String {
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// GET /api/graph/data/{user_id} - Get graph data as JSON for d3.js
pub async fn get_graph_data(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<GraphDataResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let graph = memory_guard
        .graph_memory()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Graph memory not initialized")))?;
    let graph_guard = graph.read();

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut l1_count = 0;
    let mut l2_count = 0;
    let mut l3_count = 0;

    // Get entities as nodes
    if let Ok(entities) = graph_guard.get_all_entities() {
        for entity in entities.iter().take(200) {
            let tier_label = entity
                .labels
                .first()
                .map(|l| l.as_str().to_string())
                .unwrap_or_else(|| "entity".to_string());
            nodes.push(GraphNode {
                id: entity.uuid.to_string(),
                label: entity.name.clone(),
                node_type: "entity".to_string(),
                tier: tier_label,
                strength: 1.0,
                size: 10.0,
            });
        }
    }

    // Get relationships as edges - sample from each tier for visibility
    if let Ok(relationships) = graph_guard.get_all_relationships() {
        use crate::graph_memory::EdgeTier;

        // Separate by tier for proportional sampling
        let l1_edges: Vec<_> = relationships
            .iter()
            .filter(|r| matches!(r.tier, EdgeTier::L1Working))
            .take(200)
            .collect();
        let l2_edges: Vec<_> = relationships
            .iter()
            .filter(|r| matches!(r.tier, EdgeTier::L2Episodic))
            .take(200)
            .collect();
        let l3_edges: Vec<_> = relationships
            .iter()
            .filter(|r| matches!(r.tier, EdgeTier::L3Semantic))
            .take(200)
            .collect();

        // Add edges from each tier
        for rel in l1_edges
            .iter()
            .chain(l2_edges.iter())
            .chain(l3_edges.iter())
        {
            let tier_str = match rel.tier {
                EdgeTier::L1Working => {
                    l1_count += 1;
                    "L1"
                }
                EdgeTier::L2Episodic => {
                    l2_count += 1;
                    "L2"
                }
                EdgeTier::L3Semantic => {
                    l3_count += 1;
                    "L3"
                }
            };

            edges.push(GraphEdge {
                source: rel.from_entity.to_string(),
                target: rel.to_entity.to_string(),
                edge_type: rel.relation_type.as_str().to_string(),
                tier: tier_str.to_string(),
                strength: rel.effective_strength(),
            });
        }
    }

    // Add memory nodes and connect them to their entities
    let memories = memory_guard.get_longterm_memories(100).unwrap_or_default();
    let entity_ids: std::collections::HashSet<String> =
        nodes.iter().map(|n| n.id.clone()).collect();

    for mem in memories {
        let mem_id = mem.id.0.to_string();
        nodes.push(GraphNode {
            id: mem_id.clone(),
            label: mem.experience.content.chars().take(30).collect::<String>() + "...",
            node_type: "memory".to_string(),
            tier: "longterm".to_string(),
            strength: mem.importance(),
            size: 6.0 + mem.importance() * 8.0,
        });

        // Connect memory to its entities
        for entity_id in mem.entity_ids() {
            let entity_id_str = entity_id.to_string();
            if entity_ids.contains(&entity_id_str) {
                edges.push(GraphEdge {
                    source: mem_id.clone(),
                    target: entity_id_str,
                    edge_type: "mentions".to_string(),
                    tier: "L2".to_string(),
                    strength: 0.5,
                });
            }
        }
    }

    Ok(Json(GraphDataResponse {
        stats: GraphDataStats {
            total_nodes: nodes.len(),
            total_edges: edges.len(),
            l1_edges: l1_count,
            l2_edges: l2_count,
            l3_edges: l3_count,
        },
        nodes,
        edges,
    }))
}

/// Generate the HTML page for graph visualization (includes 2D/3D toggle)
fn generate_graph_html(user_id: &str, nonce: &str) -> String {
    let html = include_str!("graph_view.html");
    let escaped_user = html_escape(user_id);
    // Only expose the dev API key in non-production mode; production keys
    // must never be embedded in a page served by the public `/graph/view` route.
    let api_key = if crate::auth::is_production_mode() {
        String::new()
    } else {
        std::env::var("SHODH_DEV_API_KEY").unwrap_or_default()
    };
    let escaped_key = html_escape(&api_key);
    html.replace("{{USER_ID}}", &escaped_user)
        .replace("{{NONCE}}", nonce)
        .replace("{{API_KEY}}", &escaped_key)
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request, Router};
    use std::sync::Mutex;
    use tower::ServiceExt;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn asset_router() -> Router {
        Router::new().route("/graph/assets/{file}", axum::routing::get(graph_asset))
    }

    #[tokio::test]
    async fn graph_asset_serves_vendored_sigma() {
        let app = asset_router();

        let req = Request::builder()
            .uri("/graph/assets/sigma.min.js")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/javascript; charset=utf-8"
        );
    }

    #[tokio::test]
    async fn graph_asset_rejects_unlisted_filename() {
        let app = asset_router();

        let req = Request::builder()
            .uri("/graph/assets/../Cargo.toml")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    // Hold ENV_LOCK across awaits so concurrent tests cannot mutate
    // SHODH_DEV_API_KEY between our set_var and the handler's env::var read.
    // Safe: the handler does not recursively acquire ENV_LOCK.
    #[allow(clippy::await_holding_lock)]
    async fn graph_view2_responds_with_html_and_substitutes_placeholders() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("SHODH_DEV_API_KEY", "view2-key");

        let app = axum::Router::new().route(
            "/graph/view2",
            axum::routing::get(graph_view2),
        );

        let req = Request::builder()
            .uri("/graph/view2?user_id=alice")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body.contains("<!DOCTYPE html>"));
        assert!(body.contains("view2-key"), "API key not substituted");
        assert!(body.contains("alice"), "user_id not substituted");
        assert!(!body.contains("{{API_KEY}}"), "template placeholder leaked");
        assert!(!body.contains("{{NONCE}}"), "template placeholder leaked");
        assert!(!body.contains("{{USER_ID}}"), "template placeholder leaked");

        std::env::remove_var("SHODH_DEV_API_KEY");
    }
}
