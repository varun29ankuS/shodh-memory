//! Decision Lineage Graph API Handlers
//!
//! Handlers for tracing decision lineage and causal relationships between memories.

use axum::{
    extract::State,
    response::Json,
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{CausalRelation, LineageBranch, LineageEdge, LineageStats, MemoryId, TraceDirection};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

fn lineage_direction_default() -> String {
    "backward".to_string()
}

fn lineage_max_depth_default() -> usize {
    10
}

fn lineage_list_limit_default() -> usize {
    50
}

/// Request to trace lineage from a memory
#[derive(Debug, Deserialize)]
pub struct LineageTraceRequest {
    pub user_id: String,
    pub memory_id: String,
    #[serde(default = "lineage_direction_default")]
    pub direction: String,
    #[serde(default = "lineage_max_depth_default")]
    pub max_depth: usize,
}

/// Request to confirm/reject an edge
#[derive(Debug, Deserialize)]
pub struct LineageEdgeRequest {
    pub user_id: String,
    pub edge_id: String,
}

/// Request to add an explicit lineage edge
#[derive(Debug, Deserialize)]
pub struct LineageAddEdgeRequest {
    pub user_id: String,
    pub from_memory_id: String,
    pub to_memory_id: String,
    pub relation: String,
}

/// Request to list lineage edges
#[derive(Debug, Deserialize)]
pub struct LineageListRequest {
    pub user_id: String,
    #[serde(default = "lineage_list_limit_default")]
    pub limit: usize,
}

/// Request to create a branch
#[derive(Debug, Deserialize)]
pub struct LineageCreateBranchRequest {
    pub user_id: String,
    pub name: String,
    pub parent_branch: String,
    pub branch_point_memory_id: String,
    pub description: Option<String>,
}

/// Response for lineage trace
#[derive(Debug, Serialize)]
pub struct LineageTraceResponse {
    pub root: String,
    pub direction: String,
    pub edges: Vec<LineageEdge>,
    pub path: Vec<String>,
    pub depth: usize,
}

/// Response for lineage edges list
#[derive(Debug, Serialize)]
pub struct LineageEdgesResponse {
    pub edges: Vec<LineageEdge>,
    pub total: usize,
}

/// Response for branches list
#[derive(Debug, Serialize)]
pub struct LineageBranchesResponse {
    pub branches: Vec<LineageBranch>,
    pub total: usize,
}

/// POST /api/lineage/trace - Trace lineage from a memory
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, memory_id = %req.memory_id))]
pub async fn lineage_trace(
    State(state): State<AppState>,
    Json(req): Json<LineageTraceRequest>,
) -> Result<Json<LineageTraceResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let memory_id_str = req.memory_id.clone();
    let direction = req.direction.clone();
    let max_depth = req.max_depth;

    let trace = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        let memory_id = MemoryId(
            uuid::Uuid::parse_str(&memory_id_str)
                .map_err(|e| anyhow::anyhow!("Invalid memory_id: {}", e))?,
        );
        let dir = match direction.as_str() {
            "forward" => TraceDirection::Forward,
            "both" => TraceDirection::Both,
            _ => TraceDirection::Backward,
        };
        memory_guard.trace_lineage(&user_id, &memory_id, dir, max_depth)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(LineageTraceResponse {
        root: trace.root.0.to_string(),
        direction: format!("{:?}", trace.direction),
        edges: trace.edges,
        path: trace.path.iter().map(|id| id.0.to_string()).collect(),
        depth: trace.depth,
    }))
}

/// POST /api/lineage/edges - List all lineage edges for a user
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn lineage_list_edges(
    State(state): State<AppState>,
    Json(req): Json<LineageListRequest>,
) -> Result<Json<LineageEdgesResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let limit = req.limit;

    let edges = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.lineage_graph().list_edges(&user_id, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = edges.len();
    Ok(Json(LineageEdgesResponse { edges, total }))
}

/// POST /api/lineage/confirm - Confirm an inferred lineage edge
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, edge_id = %req.edge_id))]
pub async fn lineage_confirm_edge(
    State(state): State<AppState>,
    Json(req): Json<LineageEdgeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let edge_id = req.edge_id.clone();

    let confirmed = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard
            .lineage_graph()
            .confirm_edge(&user_id, &edge_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({ "confirmed": confirmed })))
}

/// POST /api/lineage/reject - Reject (delete) an inferred lineage edge
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, edge_id = %req.edge_id))]
pub async fn lineage_reject_edge(
    State(state): State<AppState>,
    Json(req): Json<LineageEdgeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let edge_id = req.edge_id.clone();

    let rejected = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.lineage_graph().reject_edge(&user_id, &edge_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({ "rejected": rejected })))
}

/// POST /api/lineage/link - Add an explicit lineage edge
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn lineage_add_edge(
    State(state): State<AppState>,
    Json(req): Json<LineageAddEdgeRequest>,
) -> Result<Json<LineageEdge>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let from_str = req.from_memory_id.clone();
    let to_str = req.to_memory_id.clone();
    let relation_str = req.relation.clone();

    let edge = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        let from = MemoryId(
            uuid::Uuid::parse_str(&from_str)
                .map_err(|e| anyhow::anyhow!("Invalid from_memory_id: {}", e))?,
        );
        let to = MemoryId(
            uuid::Uuid::parse_str(&to_str)
                .map_err(|e| anyhow::anyhow!("Invalid to_memory_id: {}", e))?,
        );
        let relation = match relation_str.as_str() {
            "Caused" => CausalRelation::Caused,
            "ResolvedBy" => CausalRelation::ResolvedBy,
            "InformedBy" => CausalRelation::InformedBy,
            "SupersededBy" => CausalRelation::SupersededBy,
            "TriggeredBy" => CausalRelation::TriggeredBy,
            "BranchedFrom" => CausalRelation::BranchedFrom,
            _ => CausalRelation::RelatedTo,
        };
        memory_guard
            .lineage_graph()
            .add_explicit_edge(&user_id, from, to, relation)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(edge))
}

/// POST /api/lineage/stats - Get lineage statistics
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn lineage_stats(
    State(state): State<AppState>,
    Json(req): Json<LineageListRequest>,
) -> Result<Json<LineageStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();

    let stats = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.lineage_stats(&user_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// POST /api/lineage/branches - List all branches for a user
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn lineage_list_branches(
    State(state): State<AppState>,
    Json(req): Json<LineageListRequest>,
) -> Result<Json<LineageBranchesResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();

    let branches = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.lineage_graph().list_branches(&user_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = branches.len();
    Ok(Json(LineageBranchesResponse { branches, total }))
}

/// POST /api/lineage/branch - Create a new branch
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, name = %req.name))]
pub async fn lineage_create_branch(
    State(state): State<AppState>,
    Json(req): Json<LineageCreateBranchRequest>,
) -> Result<Json<LineageBranch>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let name = req.name.clone();
    let parent = req.parent_branch.clone();
    let branch_point_str = req.branch_point_memory_id.clone();
    let description = req.description.clone();

    let branch = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        let branch_point = MemoryId(
            uuid::Uuid::parse_str(&branch_point_str)
                .map_err(|e| anyhow::anyhow!("Invalid branch_point_memory_id: {}", e))?,
        );
        memory_guard.lineage_graph().create_branch(
            &user_id,
            &name,
            &parent,
            branch_point,
            description.as_deref(),
        )
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(branch))
}
