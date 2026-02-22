//! Search Handlers — Advanced, Multimodal, and Robotics search
//!
//! Handlers for advanced memory search with entity filtering, date ranges,
//! importance thresholds, multi-modal retrieval, and robotics-specific queries.

use axum::{extract::State, response::Json};
use serde::Deserialize;

use super::state::MultiUserMemoryManager;
use super::types::RetrieveResponse;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{self, Memory, Query as MemoryQuery};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// Request for advanced search
#[derive(Debug, Deserialize)]
pub struct AdvancedSearchRequest {
    pub user_id: String,
    pub entity_name: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub min_importance: Option<f32>,
    pub max_importance: Option<f32>,
}

/// POST /api/search/advanced - Advanced memory search with entity filtering
pub async fn advanced_search(
    State(state): State<AppState>,
    Json(req): Json<AdvancedSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    // Build search criteria
    let mut criterias = Vec::new();

    if let Some(entity) = req.entity_name {
        criterias.push(memory::storage::SearchCriteria::ByEntity(entity));
    }

    if let (Some(start), Some(end)) = (req.start_date, req.end_date) {
        let start_dt = chrono::DateTime::parse_from_rfc3339(&start)
            .map_err(|_| AppError::InvalidInput {
                field: "start_date".to_string(),
                reason: "Invalid RFC3339 format".to_string(),
            })?
            .with_timezone(&chrono::Utc);

        let end_dt = chrono::DateTime::parse_from_rfc3339(&end)
            .map_err(|_| AppError::InvalidInput {
                field: "end_date".to_string(),
                reason: "Invalid RFC3339 format".to_string(),
            })?
            .with_timezone(&chrono::Utc);

        criterias.push(memory::storage::SearchCriteria::ByDate {
            start: start_dt,
            end: end_dt,
        });
    }

    if let (Some(min), Some(max)) = (req.min_importance, req.max_importance) {
        criterias.push(memory::storage::SearchCriteria::ByImportance { min, max });
    }

    // Execute combined search
    if criterias.is_empty() {
        return Err(AppError::InvalidInput {
            field: "search".to_string(),
            reason: "At least one search criterion must be provided".to_string(),
        });
    }

    // Combine criteria or use single criterion directly
    let criteria = if criterias.len() == 1 {
        criterias.pop().unwrap() // Safe: just verified len() == 1
    } else {
        memory::storage::SearchCriteria::Combined(criterias)
    };

    let raw_memories = memory_guard
        .advanced_search(criteria)
        .map_err(AppError::Internal)?;

    let count = raw_memories.len();
    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    Ok(Json(RetrieveResponse { memories, count }))
}

// =============================================================================
// MULTIMODAL SEARCH
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct MultiModalSearchRequest {
    pub user_id: String,
    pub query_text: String,
    pub mode: String,
    pub limit: Option<usize>,
}

pub async fn multimodal_search(
    State(state): State<AppState>,
    Json(req): Json<MultiModalSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let retrieval_mode = match req.mode.as_str() {
        "similarity" => memory::RetrievalMode::Similarity,
        "temporal" => memory::RetrievalMode::Temporal,
        "causal" => memory::RetrievalMode::Causal,
        "associative" => memory::RetrievalMode::Associative,
        "hybrid" => memory::RetrievalMode::Hybrid,
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        _ => {
            return Err(AppError::InvalidInput {
                field: "mode".to_string(),
                reason: format!(
                    "Invalid mode: {}. Must be one of: similarity, temporal, causal, associative, hybrid, spatial, mission, action_outcome",
                    req.mode
                ),
            })
        }
    };

    let query = MemoryQuery {
        query_text: Some(req.query_text.clone()),
        max_results: req.limit.unwrap_or(10),
        retrieval_mode,
        ..Default::default()
    };

    let shared_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;
    let raw_memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();
    let count = raw_memories.len();

    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    state.log_event(
        &req.user_id,
        "MULTIMODAL_SEARCH",
        &req.mode,
        &format!("Retrieved {} memories using {} mode", count, req.mode),
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

// =============================================================================
// ROBOTICS SEARCH
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct RoboticsSearchRequest {
    pub user_id: String,
    pub mode: String,
    pub query_text: Option<String>,
    pub robot_id: Option<String>,
    pub mission_id: Option<String>,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub radius_meters: Option<f64>,
    pub action_type: Option<String>,
    pub min_reward: Option<f32>,
    pub max_reward: Option<f32>,
    pub limit: Option<usize>,
}

pub async fn robotics_search(
    State(state): State<AppState>,
    Json(req): Json<RoboticsSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let retrieval_mode = match req.mode.as_str() {
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        "hybrid" => memory::RetrievalMode::Hybrid,
        "similarity" => memory::RetrievalMode::Similarity,
        _ => {
            return Err(AppError::InvalidInput {
                field: "mode".to_string(),
                reason: "Invalid mode. Use: spatial, mission, action_outcome, hybrid, similarity"
                    .to_string(),
            })
        }
    };

    let geo_filter = match (req.lat, req.lon, req.radius_meters) {
        (Some(lat), Some(lon), Some(radius)) => Some(memory::GeoFilter::new(lat, lon, radius)),
        _ => None,
    };

    let reward_range = match (req.min_reward, req.max_reward) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, 1.0)),
        (None, Some(max)) => Some((-1.0, max)),
        _ => None,
    };

    if matches!(retrieval_mode, memory::RetrievalMode::Spatial) && geo_filter.is_none() {
        return Err(AppError::InvalidInput {
            field: "lat/lon/radius_meters".to_string(),
            reason: "Spatial mode requires lat, lon, and radius_meters".to_string(),
        });
    }

    if matches!(retrieval_mode, memory::RetrievalMode::Mission) && req.mission_id.is_none() {
        return Err(AppError::InvalidInput {
            field: "mission_id".to_string(),
            reason: "Mission mode requires mission_id".to_string(),
        });
    }

    let query = MemoryQuery {
        query_text: req.query_text,
        robot_id: req.robot_id.clone(),
        mission_id: req.mission_id.clone(),
        geo_filter,
        action_type: req.action_type.clone(),
        reward_range,
        max_results: req.limit.unwrap_or(10),
        retrieval_mode,
        ..Default::default()
    };

    let shared_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;
    let raw_memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();
    let count = raw_memories.len();

    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    state.log_event(
        &req.user_id,
        "ROBOTICS_SEARCH",
        &req.mode,
        &format!(
            "Retrieved {} robotics memories (robot={:?}, mission={:?})",
            count, req.robot_id, req.mission_id
        ),
    );

    Ok(Json(RetrieveResponse { memories, count }))
}
