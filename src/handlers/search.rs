//! Advanced Search Handlers
//!
//! Handlers for advanced memory search with entity filtering, date ranges, and importance.

use axum::{
    extract::State,
    response::Json,
};
use serde::Deserialize;

use super::state::MultiUserMemoryManager;
use super::types::RetrieveResponse;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory;
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

    let criteria = if criterias.len() == 1 {
        criterias
            .into_iter()
            .next()
            .expect("Criteria list has exactly one element")
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
