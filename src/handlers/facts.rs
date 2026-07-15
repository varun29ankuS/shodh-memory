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
    validation::validate_limit(req.limit, "limit").map_validation_err("limit")?;

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
    validation::validate_limit(req.limit, "limit").map_validation_err("limit")?;
    validation::validate_query_text(&req.query).map_validation_err("query")?;

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
    validation::validate_limit(req.limit, "limit").map_validation_err("limit")?;
    validation::validate_short_string(&req.entity, "entity").map_validation_err("entity")?;

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
    /// Max clusters to return. Defaults to 20, clamped to [`validation::MAX_LIMIT`].
    #[serde(default = "default_narratives_limit")]
    pub limit: usize,
    /// Clusters to skip before collecting `limit` results. Defaults to 0.
    /// Combine with `total_clusters` in the response to page through results
    /// beyond `validation::MAX_LIMIT` (e.g. offset += limit).
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub entity_filter: Option<String>,
}

/// Response for fact narratives
#[derive(Debug, Serialize)]
pub struct FactNarrativesResponse {
    pub success: bool,
    pub clusters: Vec<FactCluster>,
    /// Total facts across ALL matching clusters (not just the returned page).
    pub total_facts: usize,
    /// Total matching clusters (not just the returned page). Compare against
    /// `offset + clusters.len()` to know whether more pages remain.
    pub total_clusters: usize,
}

/// POST /api/facts/narratives - Get fact narratives clustered by topic
/// `limit` defaults to 20 and is clamped to [`validation::MAX_LIMIT`]; `offset`
/// defaults to 0. Use `total_clusters` in the response to page through results.
pub async fn fact_narratives(
    State(state): State<AppState>,
    Json(req): Json<FactNarrativesRequest>,
) -> Result<Json<FactNarrativesResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    let user_id = req.user_id.clone();
    let limit = req.limit.min(validation::MAX_LIMIT);
    let offset = req.offset.unwrap_or(0);
    let entity_filter = req.entity_filter.clone();

    let clusters = tokio::task::spawn_blocking(move || {
        let ms = memory.read();
        ms.build_fact_narratives(&user_id, entity_filter.as_deref())
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total_clusters = clusters.len();
    let total_facts: usize = clusters.iter().map(|c| c.facts.len()).sum();

    let paged_clusters: Vec<FactCluster> = clusters.into_iter().skip(offset).take(limit).collect();

    Ok(Json(FactNarrativesResponse {
        success: true,
        clusters: paged_clusters,
        total_facts,
        total_clusters,
    }))
}

#[cfg(test)]
mod tests {
    use crate::handlers::test_helpers::{self, TestHarness};
    use crate::memory::{FactType, SemanticFact};
    use axum::http::StatusCode;

    /// Seed `n` independent 2-fact clusters (each pair shares a unique topic entity,
    /// so `build_fact_narratives` groups them into exactly `n` clusters — none of
    /// them hit the hub-entity exclusion since each topic entity only appears twice).
    fn seed_narrative_clusters(harness: &TestHarness, user_id: &str, n: usize) {
        let memory = harness.manager.get_user_memory(user_id).unwrap();
        let guard = memory.read();
        let store = guard.fact_store();
        for i in 0..n {
            for j in 0..2 {
                let fact = SemanticFact {
                    id: format!("fact-{i}-{j}"),
                    fact: format!("Topic {i} statement {j}"),
                    confidence: 0.8,
                    support_count: 3,
                    source_memories: vec![],
                    related_entities: vec![format!("topic{i}")],
                    created_at: chrono::Utc::now(),
                    last_reinforced: chrono::Utc::now(),
                    fact_type: FactType::Pattern,
                };
                store.store(user_id, &fact).unwrap();
            }
        }
    }

    /// Regression test for the silent-cap-at-50 bug: previously `limit` was
    /// hard-clamped to 50 with no way to page past it. Seeds 60 clusters (above
    /// the old cap) and verifies `limit` > 50 is honored and `offset` pages
    /// correctly, while `total_clusters` always reports the true total.
    #[tokio::test]
    async fn narratives_offset_and_limit_above_old_cap() {
        let harness = TestHarness::new();
        let user_id = "narratives-user";
        seed_narrative_clusters(&harness, user_id, 60);

        // limit=55 (> old hard cap of 50), offset=0 -> first 55 of 60 clusters.
        let req = test_helpers::post_json(
            "/api/facts/narratives",
            &serde_json::json!({ "user_id": user_id, "limit": 55, "offset": 0 }),
        );
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["clusters"].as_array().unwrap().len(),
            55,
            "limit above the old 50-cluster cap must be honored"
        );
        assert_eq!(body["total_clusters"], 60);

        // offset=55 pages past what the old cap made permanently unreachable.
        let req = test_helpers::post_json(
            "/api/facts/narratives",
            &serde_json::json!({ "user_id": user_id, "limit": 55, "offset": 55 }),
        );
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["clusters"].as_array().unwrap().len(),
            5,
            "offset must skip past the first page (60 - 55 = 5 remaining)"
        );
        assert_eq!(body["total_clusters"], 60);

        // Existing callers omitting limit/offset keep the old default of 20.
        let req = test_helpers::post_json(
            "/api/facts/narratives",
            &serde_json::json!({ "user_id": user_id }),
        );
        let (status, body) = test_helpers::send(harness.router(), req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["clusters"].as_array().unwrap().len(),
            20,
            "default limit must be unchanged for existing callers"
        );
        assert_eq!(body["total_clusters"], 60);
    }
}
