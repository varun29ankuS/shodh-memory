//! CRUD Handlers for Memory Operations
//!
//! Create, Read, Update, Delete operations for individual memories
//! and bulk delete operations (forget by age, importance, tags, etc.)

use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use super::state::MultiUserMemoryManager;
use super::types::MemoryEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{self, ExperienceType, Memory, MemoryId, Query as MemoryQuery};
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// LIST MEMORIES TYPES
// =============================================================================

/// Query parameters for listing memories
#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub limit: Option<usize>,
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    /// Text search query - filters by content or tags (case-insensitive)
    pub query: Option<String>,
}

/// List response - simplified memory list
#[derive(Debug, Serialize)]
pub struct ListResponse {
    pub memories: Vec<ListMemoryItem>,
    pub total: usize,
}

/// Request for POST /api/memories - list memories with user_id in body
#[derive(Debug, Deserialize)]
pub struct ListMemoriesRequest {
    pub user_id: String,
    pub limit: Option<usize>,
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    pub query: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListMemoryItem {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub importance: f32,
    pub tags: Vec<String>,
    pub created_at: String,
}

// =============================================================================
// UPDATE/DELETE RESPONSE TYPES
// =============================================================================

/// Request for updating memory content
#[derive(Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    pub user_id: String,
    pub content: String,
    pub embeddings: Option<Vec<f32>>,
}

/// Response for memory update operations
#[derive(Debug, Serialize)]
pub struct UpdateMemoryResponse {
    pub success: bool,
    pub id: String,
    pub message: String,
}

/// Response for memory delete operations
#[derive(Debug, Serialize)]
pub struct DeleteMemoryResponse {
    pub success: bool,
    pub id: String,
    pub message: String,
}

// =============================================================================
// FORGET REQUEST TYPES (local - not in shared types.rs)
// =============================================================================

/// Forget memories by age
#[derive(Debug, Deserialize)]
pub struct ForgetByAgeRequest {
    pub user_id: String,
    pub days_old: u32,
}

/// Forget memories by importance threshold
#[derive(Debug, Deserialize)]
pub struct ForgetByImportanceRequest {
    pub user_id: String,
    pub threshold: f32,
}

/// Forget memories matching a pattern
#[derive(Debug, Deserialize)]
pub struct ForgetByPatternRequest {
    pub user_id: String,
    pub pattern: String,
}

/// Forget memories by tags
#[derive(Debug, Deserialize)]
pub struct ForgetByTagsRequest {
    pub user_id: String,
    /// Tags to match for deletion (deletes memories matching ANY of these tags)
    pub tags: Vec<String>,
}

/// Forget memories by date range
#[derive(Debug, Deserialize)]
pub struct ForgetByDateRequest {
    pub user_id: String,
    /// Start of date range (inclusive) - ISO 8601 format
    pub start: chrono::DateTime<chrono::Utc>,
    /// End of date range (inclusive) - ISO 8601 format
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Bulk delete memories by filters
#[derive(Debug, Deserialize)]
pub struct BulkDeleteRequest {
    pub user_id: String,
    /// Delete memories matching ANY of these tags
    pub tags: Option<Vec<String>>,
    /// Delete memories of this type
    pub memory_type: Option<String>,
    /// Delete memories created after this timestamp
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    /// Delete memories created before this timestamp
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
}

/// Clear ALL memories for a user (GDPR compliance)
#[derive(Debug, Deserialize)]
pub struct ClearAllRequest {
    pub user_id: String,
    /// Safety confirmation - must be "CONFIRM" to proceed
    pub confirm: String,
}

/// PATCH endpoint for partial memory updates
#[derive(Debug, Deserialize)]
pub struct PatchMemoryRequest {
    pub user_id: String,
    /// New content (optional)
    pub content: Option<String>,
    /// New/additional tags (optional)
    pub tags: Option<Vec<String>>,
    /// New memory type (optional)
    pub memory_type: Option<String>,
}

// =============================================================================
// GET MEMORY HANDLER
// =============================================================================

/// GET /api/memories/{memory_id} - Get specific memory by ID
#[tracing::instrument(skip(state))]
pub async fn get_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Memory>, AppError> {
    let user_id = params
        .get("user_id")
        .ok_or_else(|| AppError::InvalidInput {
            field: "user_id".to_string(),
            reason: "user_id required".to_string(),
        })?;

    validation::validate_user_id(user_id).map_validation_err("user_id")?;
    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state.get_user_memory(user_id).map_err(AppError::Internal)?;
    let memory_guard = memory.read();

    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    // Search for memory
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    Ok(Json((*shared_memory).clone()))
}

// =============================================================================
// LIST MEMORIES HANDLER
// =============================================================================

/// GET /api/list/{user_id} - List all memories for a user
/// Query params: ?limit=100&type=Decision
#[tracing::instrument(skip(state), fields(user_id = %user_id))]
pub async fn list_memories(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<Json<ListResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let all_memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_all_memories()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Filter by type if specified
    let mut filtered: Vec<_> = if let Some(ref type_filter) = query.memory_type {
        let type_lower = type_filter.to_lowercase();
        all_memories
            .into_iter()
            .filter(|m| format!("{:?}", m.experience.experience_type).to_lowercase() == type_lower)
            .collect()
    } else {
        all_memories
    };

    // Filter by text query if specified (search in content and tags)
    if let Some(ref text_query) = query.query {
        let query_lower = text_query.to_lowercase();
        filtered = filtered
            .into_iter()
            .filter(|m| {
                // Check content
                if m.experience.content.to_lowercase().contains(&query_lower) {
                    return true;
                }
                // Check tags/entities
                for tag in &m.experience.entities {
                    if tag.to_lowercase().contains(&query_lower) {
                        return true;
                    }
                }
                false
            })
            .collect();
    }

    let total = filtered.len();
    let limit = query.limit.unwrap_or(100).min(1000);

    let memories: Vec<ListMemoryItem> = filtered
        .into_iter()
        .take(limit)
        .map(|m| ListMemoryItem {
            id: m.id.0.to_string(),
            content: m.experience.content.chars().take(500).collect(),
            memory_type: format!("{:?}", m.experience.experience_type),
            importance: m.importance(),
            tags: m.experience.entities.clone(),
            created_at: m.created_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(ListResponse { memories, total }))
}

/// POST /api/memories - List memories (user_id in body)
/// Alternative to GET /api/list/{user_id} for clients that prefer POST
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn list_memories_post(
    State(state): State<AppState>,
    Json(req): Json<ListMemoriesRequest>,
) -> Result<Json<ListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let all_memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_all_memories()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Filter by type if specified
    let mut filtered: Vec<_> = if let Some(ref type_filter) = req.memory_type {
        let type_lower = type_filter.to_lowercase();
        all_memories
            .into_iter()
            .filter(|m| format!("{:?}", m.experience.experience_type).to_lowercase() == type_lower)
            .collect()
    } else {
        all_memories
    };

    // Filter by text query if specified (search in content and tags)
    if let Some(ref text_query) = req.query {
        let query_lower = text_query.to_lowercase();
        filtered = filtered
            .into_iter()
            .filter(|m| {
                // Check content
                if m.experience.content.to_lowercase().contains(&query_lower) {
                    return true;
                }
                // Check tags/entities
                for tag in &m.experience.entities {
                    if tag.to_lowercase().contains(&query_lower) {
                        return true;
                    }
                }
                false
            })
            .collect();
    }

    let total = filtered.len();
    let limit = req.limit.unwrap_or(100).min(1000);

    let memories: Vec<ListMemoryItem> = filtered
        .into_iter()
        .take(limit)
        .map(|m| ListMemoryItem {
            id: m.id.0.to_string(),
            content: m.experience.content.chars().take(500).collect(),
            memory_type: format!("{:?}", m.experience.experience_type),
            importance: m.importance(),
            tags: m.experience.entities.clone(),
            created_at: m.created_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(ListResponse { memories, total }))
}

// =============================================================================
// UPDATE MEMORY HANDLER
// =============================================================================

/// PUT /api/memories/{memory_id} - Update memory content
#[tracing::instrument(skip(state), fields(memory_id = %memory_id))]
pub async fn update_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<Json<UpdateMemoryResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    if let Some(ref emb) = req.embeddings {
        validation::validate_embeddings(emb)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    // Get current memory to preserve metadata
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    let mut current_memory = (*shared_memory).clone();

    let content_preview: String = req.content.chars().take(50).collect();

    current_memory.experience.content = req.content;
    if let Some(emb) = req.embeddings {
        current_memory.experience.embeddings = Some(emb);
    }

    let experience = current_memory.experience.clone();
    memory_guard
        .remember(experience, None)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "UPDATE",
        &memory_id,
        &format!("Updated memory content: {content_preview}"),
    );

    Ok(Json(UpdateMemoryResponse {
        success: true,
        id: memory_id,
        message: "Memory updated successfully".to_string(),
    }))
}

// =============================================================================
// DELETE MEMORY HANDLER
// =============================================================================

/// DELETE /api/memories/{memory_id} - Delete specific memory
#[tracing::instrument(skip(state), fields(memory_id = %memory_id))]
pub async fn delete_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<DeleteMemoryResponse>, AppError> {
    let user_id = params
        .get("user_id")
        .ok_or_else(|| AppError::InvalidInput {
            field: "user_id".to_string(),
            reason: "user_id required".to_string(),
        })?;

    validation::validate_user_id(user_id).map_validation_err("user_id")?;
    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state.get_user_memory(user_id).map_err(AppError::Internal)?;
    let memory_guard = memory.read();

    let uuid =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;
    memory_guard
        .forget(memory::ForgetCriteria::ById(MemoryId(uuid)))
        .map_err(AppError::Internal)?;

    state.log_event(user_id, "DELETE", &memory_id, "Memory deleted");

    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.to_string(),
        memory_id: Some(memory_id.clone()),
        content_preview: None,
        memory_type: None,
        importance: None,
        count: None,
    });

    Ok(Json(DeleteMemoryResponse {
        success: true,
        id: memory_id,
        message: "Memory deleted successfully".to_string(),
    }))
}

// =============================================================================
// PATCH MEMORY HANDLER
// =============================================================================

/// PATCH /api/memories/{memory_id} - Partial memory update
#[tracing::instrument(skip(state), fields(memory_id = %memory_id))]
pub async fn patch_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<PatchMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    let mut current_memory = (*shared_memory).clone();
    let mut changes = Vec::new();

    // Update content if provided
    if let Some(ref new_content) = req.content {
        validation::validate_content(new_content, false).map_validation_err("content")?;
        current_memory.experience.content = new_content.clone();
        current_memory.experience.embeddings = None;
        changes.push("content");
    }

    // Update tags if provided (add to existing entities)
    if let Some(ref new_tags) = req.tags {
        for tag in new_tags {
            if !current_memory.experience.entities.contains(tag) {
                current_memory.experience.entities.push(tag.clone());
            }
        }
        changes.push("tags");
    }

    // Update type if provided
    if let Some(ref type_str) = req.memory_type {
        current_memory.experience.experience_type = parse_experience_type(type_str)?;
        changes.push("type");
    }

    if changes.is_empty() {
        return Err(AppError::InvalidInput {
            field: "body".to_string(),
            reason: "No fields to update provided".to_string(),
        });
    }

    let experience = current_memory.experience.clone();
    memory_guard
        .remember(experience, None)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "PATCH",
        &memory_id,
        &format!("Updated fields: {}", changes.join(", ")),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "id": memory_id,
        "updated_fields": changes
    })))
}

// =============================================================================
// FORGET BY AGE HANDLER
// =============================================================================

/// POST /api/forget/age - Forget memories older than N days
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_age(
    State(state): State<AppState>,
    Json(req): Json<ForgetByAgeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::OlderThan(req.days_old))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_AGE",
        &format!("{} days", req.days_old),
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "criteria": format!("older than {} days", req.days_old)
    })))
}

// =============================================================================
// FORGET BY IMPORTANCE HANDLER
// =============================================================================

/// POST /api/forget/importance - Forget memories below importance threshold
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_importance(
    State(state): State<AppState>,
    Json(req): Json<ForgetByImportanceRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.threshold < 0.0 || req.threshold > 1.0 {
        return Err(AppError::InvalidInput {
            field: "threshold".to_string(),
            reason: "Must be between 0.0 and 1.0".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::LowImportance(req.threshold))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_IMPORTANCE",
        &format!("threshold {}", req.threshold),
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "criteria": format!("importance < {}", req.threshold)
    })))
}

// =============================================================================
// FORGET BY PATTERN HANDLER
// =============================================================================

/// POST /api/forget/pattern - Forget memories matching a pattern
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_pattern(
    State(state): State<AppState>,
    Json(req): Json<ForgetByPatternRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::Pattern(req.pattern.clone()))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_PATTERN",
        &req.pattern,
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "pattern": req.pattern
    })))
}

// =============================================================================
// FORGET BY TAGS HANDLER
// =============================================================================

/// POST /api/forget/tags - Forget memories matching any of the provided tags
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_tags(
    State(state): State<AppState>,
    Json(req): Json<ForgetByTagsRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.tags.is_empty() {
        return Err(AppError::InvalidInput {
            field: "tags".to_string(),
            reason: "At least one tag must be provided".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let deleted_count = memory_guard
        .forget(memory::ForgetCriteria::ByTags(req.tags.clone()))
        .map_err(AppError::Internal)?;

    info!(
        "🏷️ Forget by tags: user={}, tags={:?}, deleted={}",
        req.user_id, req.tags, deleted_count
    );

    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(format!("tags: {:?}", req.tags)),
        memory_type: None,
        importance: None,
        count: Some(deleted_count),
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "tags": req.tags
    })))
}

// =============================================================================
// FORGET BY DATE HANDLER
// =============================================================================

/// POST /api/forget/date - Forget memories within a date range
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn forget_by_date(
    State(state): State<AppState>,
    Json(req): Json<ForgetByDateRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.end < req.start {
        return Err(AppError::InvalidInput {
            field: "end".to_string(),
            reason: "End date must be after start date".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let deleted_count = memory_guard
        .forget(memory::ForgetCriteria::ByDateRange {
            start: req.start,
            end: req.end,
        })
        .map_err(AppError::Internal)?;

    info!(
        "📅 Forget by date: user={}, start={}, end={}, deleted={}",
        req.user_id, req.start, req.end, deleted_count
    );

    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(format!(
            "{} to {}",
            req.start.format("%Y-%m-%d"),
            req.end.format("%Y-%m-%d")
        )),
        memory_type: None,
        importance: None,
        count: Some(deleted_count),
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "start": req.start.to_rfc3339(),
        "end": req.end.to_rfc3339()
    })))
}

// =============================================================================
// BULK DELETE HANDLER
// =============================================================================

/// POST /api/bulk_delete - Bulk delete memories by filters
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn bulk_delete_memories(
    State(state): State<AppState>,
    Json(req): Json<BulkDeleteRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let mut total_count = 0;

    // Delete by tags if specified
    if let Some(ref tags) = req.tags {
        if !tags.is_empty() {
            let count = memory_guard
                .forget(memory::ForgetCriteria::ByTags(tags.clone()))
                .map_err(AppError::Internal)?;
            total_count += count;
        }
    }

    // Delete by type if specified
    if let Some(ref type_str) = req.memory_type {
        let exp_type = parse_experience_type(type_str)?;
        let count = memory_guard
            .forget(memory::ForgetCriteria::ByType(exp_type))
            .map_err(AppError::Internal)?;
        total_count += count;
    }

    // Delete by date range if specified
    if req.created_after.is_some() || req.created_before.is_some() {
        let start = req
            .created_after
            .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC);
        let end = req.created_before.unwrap_or(chrono::Utc::now());
        let count = memory_guard
            .forget(memory::ForgetCriteria::ByDateRange { start, end })
            .map_err(AppError::Internal)?;
        total_count += count;
    }

    state.log_event(
        &req.user_id,
        "BULK_DELETE",
        "multiple",
        &format!("Deleted {total_count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": total_count
    })))
}

// =============================================================================
// CLEAR ALL HANDLER (GDPR)
// =============================================================================

/// POST /api/clear_all - Clear ALL memories for a user (GDPR compliance)
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn clear_all_memories(
    State(state): State<AppState>,
    Json(req): Json<ClearAllRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Safety check - require explicit confirmation
    if req.confirm != "CONFIRM" {
        return Err(AppError::InvalidInput {
            field: "confirm".to_string(),
            reason: "Must provide confirm: \"CONFIRM\" to clear all memories".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::All)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "CLEAR_ALL",
        "GDPR",
        &format!("GDPR erasure: deleted {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": count,
        "message": "All memories have been permanently deleted (GDPR erasure)"
    })))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Parse experience type from string
fn parse_experience_type(type_str: &str) -> Result<ExperienceType, AppError> {
    match type_str.to_lowercase().as_str() {
        "observation" => Ok(ExperienceType::Observation),
        "decision" => Ok(ExperienceType::Decision),
        "learning" => Ok(ExperienceType::Learning),
        "error" => Ok(ExperienceType::Error),
        "discovery" => Ok(ExperienceType::Discovery),
        "pattern" => Ok(ExperienceType::Pattern),
        "context" => Ok(ExperienceType::Context),
        "task" => Ok(ExperienceType::Task),
        "codeedit" => Ok(ExperienceType::CodeEdit),
        "fileaccess" => Ok(ExperienceType::FileAccess),
        "search" => Ok(ExperienceType::Search),
        "command" => Ok(ExperienceType::Command),
        "conversation" => Ok(ExperienceType::Conversation),
        _ => Err(AppError::InvalidInput {
            field: "memory_type".to_string(),
            reason: format!("Invalid memory type: {type_str}"),
        }),
    }
}
