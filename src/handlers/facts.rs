//! Facts API Handlers
//!
//! Handlers for semantic facts extracted from episodic memories.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{self, FactCluster, SemanticFact};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

fn facts_default_limit() -> usize {
    50
}

/// Request for listing facts
#[derive(Debug, Deserialize)]
pub struct FactsListRequest {
    pub user_id: String,
    #[serde(default = "facts_default_limit")]
    pub limit: usize,
}

/// Request for searching facts
#[derive(Debug, Deserialize)]
pub struct FactsSearchRequest {
    pub user_id: String,
    pub query: String,
    #[serde(default = "facts_default_limit")]
    pub limit: usize,
}

/// Request for facts by entity
#[derive(Debug, Deserialize)]
pub struct FactsByEntityRequest {
    pub user_id: String,
    pub entity: String,
    #[serde(default = "facts_default_limit")]
    pub limit: usize,
}

/// Response containing facts
#[derive(Debug, Serialize)]
pub struct FactsResponse {
    pub facts: Vec<SemanticFact>,
    pub total: usize,
}

/// POST /api/facts/list - List semantic facts for a user
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn list_facts(
    State(state): State<AppState>,
    Json(req): Json<FactsListRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_facts(&user_id, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/search - Search facts by keyword
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
pub async fn search_facts(
    State(state): State<AppState>,
    Json(req): Json<FactsSearchRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let query = req.query.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.search_facts(&user_id, &query, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/by-entity - Get facts related to an entity
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, entity = %req.entity))]
pub async fn facts_by_entity(
    State(state): State<AppState>,
    Json(req): Json<FactsByEntityRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let entity = req.entity.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_facts_by_entity(&user_id, &entity, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/stats - Get statistics about stored facts
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn get_facts_stats(
    State(state): State<AppState>,
    Json(req): Json<FactsListRequest>,
) -> Result<Json<memory::FactStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();

    let stats = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_fact_stats(&user_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

// =============================================================================
// Fact Purge
// =============================================================================

/// Request for purging garbage facts
#[derive(Debug, Deserialize)]
pub struct FactPurgeRequest {
    pub user_id: String,
    /// Pattern to match against fact content (case-insensitive substring).
    /// Facts containing this pattern will be deleted.
    pub pattern: String,
    /// If true, only count matches without deleting (default: false)
    #[serde(default)]
    pub dry_run: bool,
}

/// Response for fact purge
#[derive(Debug, Serialize)]
pub struct FactPurgeResponse {
    pub success: bool,
    pub deleted: usize,
    pub total_scanned: usize,
    pub dry_run: bool,
}

/// POST /api/facts/purge - Delete facts matching a content pattern
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, pattern = %req.pattern))]
pub async fn purge_facts(
    State(state): State<AppState>,
    Json(req): Json<FactPurgeRequest>,
) -> Result<Json<FactPurgeResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.pattern.len() < 3 {
        return Err(AppError::InvalidInput {
            field: "pattern".into(),
            reason: "Must be at least 3 characters to prevent accidental mass deletion".into(),
        });
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    let user_id = req.user_id.clone();
    let pattern = req.pattern.to_lowercase();
    let dry_run = req.dry_run;

    let (deleted, total_scanned) = tokio::task::spawn_blocking(move || {
        let ms = memory.read();
        if dry_run {
            // Count only — don't delete
            let all_facts = ms.get_facts(&user_id, 10_000)?;
            let total = all_facts.len();
            let matched = all_facts
                .iter()
                .filter(|f| f.fact.to_lowercase().contains(&pattern))
                .count();
            Ok::<_, anyhow::Error>((matched, total))
        } else {
            ms.purge_facts(&user_id, |f| f.fact.to_lowercase().contains(&pattern))
        }
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(FactPurgeResponse {
        success: true,
        deleted,
        total_scanned,
        dry_run,
    }))
}

// =============================================================================
// Fact Narratives
// =============================================================================

fn default_narratives_limit() -> usize {
    20
}

/// Request for fact narratives
#[derive(Debug, Deserialize)]
pub struct FactNarrativesRequest {
    pub user_id: String,
    #[serde(default = "default_narratives_limit")]
    pub limit: usize,
    #[serde(default)]
    pub entity_filter: Option<String>,
}

/// Response for fact narratives
#[derive(Debug, Serialize)]
pub struct FactNarrativesResponse {
    pub success: bool,
    pub clusters: Vec<FactCluster>,
    pub total_facts: usize,
    pub total_clusters: usize,
}

/// POST /api/facts/narratives - Get fact narratives clustered by topic
pub async fn fact_narratives(
    State(state): State<AppState>,
    Json(req): Json<FactNarrativesRequest>,
) -> Result<Json<FactNarrativesResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    let user_id = req.user_id.clone();
    let limit = req.limit.min(50);
    let entity_filter = req.entity_filter.clone();

    let clusters = tokio::task::spawn_blocking(move || {
        let ms = memory.read();
        ms.build_fact_narratives(&user_id, limit, entity_filter.as_deref())
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total_facts: usize = clusters.iter().map(|c| c.facts.len()).sum();
    let total_clusters = clusters.len();
    Ok(Json(FactNarrativesResponse {
        success: true,
        clusters,
        total_facts,
        total_clusters,
    }))
}
