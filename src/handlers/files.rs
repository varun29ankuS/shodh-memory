//! File Memory Handlers
//!
//! Handlers for codebase file indexing and search.

use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use super::state::MultiUserMemoryManager;
use super::todos::TodoQuery;
use super::types::MemoryEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{FileMemoryStats, IndexingResult, ProjectId};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

fn default_search_limit() -> usize {
    10
}

/// Request for listing files
#[derive(Debug, Deserialize)]
pub struct ListFilesRequest {
    pub user_id: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Request to scan/index a codebase
#[derive(Debug, Deserialize)]
pub struct IndexCodebaseRequest {
    pub user_id: String,
    pub codebase_path: String,
    #[serde(default)]
    pub force: bool,
}

/// Request to search files
#[derive(Debug, Deserialize)]
pub struct SearchFilesRequest {
    pub user_id: String,
    pub query: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

/// Response for file list operations
#[derive(Debug, Serialize)]
pub struct FileListResponse {
    pub success: bool,
    pub files: Vec<FileMemorySummary>,
    pub total: usize,
}

/// Summary of a file memory
#[derive(Debug, Serialize)]
pub struct FileMemorySummary {
    pub id: String,
    pub path: String,
    pub absolute_path: String,
    pub file_type: String,
    pub summary: String,
    pub key_items: Vec<String>,
    pub access_count: u32,
    pub last_accessed: String,
    pub heat_score: u8,
    pub size_bytes: u64,
    pub line_count: usize,
}

/// Response for scan operation
#[derive(Debug, Serialize)]
pub struct ScanResponse {
    pub success: bool,
    pub total_files: usize,
    pub eligible_files: usize,
    pub skipped_files: usize,
    pub limit_reached: bool,
    pub message: String,
}

/// Response for index operation
#[derive(Debug, Serialize)]
pub struct IndexResponse {
    pub success: bool,
    pub result: IndexingResult,
    pub message: String,
}

/// Response for file stats
#[derive(Debug, Serialize)]
pub struct FileStatsResponse {
    pub success: bool,
    pub stats: FileMemoryStats,
}

/// POST /api/projects/{project_id}/files - List files for a project
pub async fn list_project_files(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<ListFilesRequest>,
) -> Result<Json<FileListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let files = state
        .file_store
        .list_by_project(&req.user_id, &project.id, req.limit)
        .map_err(AppError::Internal)?;

    let total = files.len();
    let summaries: Vec<FileMemorySummary> = files
        .into_iter()
        .map(|f| {
            let heat_score = f.heat_score();
            FileMemorySummary {
                id: f.id.0.to_string(),
                path: f.path,
                absolute_path: f.absolute_path,
                file_type: format!("{:?}", f.file_type),
                summary: f.summary,
                key_items: f.key_items,
                access_count: f.access_count,
                last_accessed: f.last_accessed.to_rfc3339(),
                heat_score,
                size_bytes: f.size_bytes,
                line_count: f.line_count,
            }
        })
        .collect();

    Ok(Json(FileListResponse {
        success: true,
        files: summaries,
        total,
    }))
}

/// POST /api/projects/{project_id}/scan - Scan codebase
pub async fn scan_project_codebase(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<IndexCodebaseRequest>,
) -> Result<Json<ScanResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let codebase_path = std::path::Path::new(&req.codebase_path);
    if !codebase_path.exists() {
        return Err(AppError::InvalidInput {
            field: "codebase_path".to_string(),
            reason: format!("Path does not exist: {}", req.codebase_path),
        });
    }
    if !codebase_path.is_dir() {
        return Err(AppError::InvalidInput {
            field: "codebase_path".to_string(),
            reason: format!("Path is not a directory: {}", req.codebase_path),
        });
    }

    let scan_result = state
        .file_store
        .scan_codebase(codebase_path, None)
        .map_err(AppError::Internal)?;

    let message = if scan_result.limit_reached {
        format!(
            "Found {} eligible files (limit reached). {} files skipped.",
            scan_result.eligible_files, scan_result.skipped_files
        )
    } else {
        format!(
            "Found {} eligible files. {} files skipped.",
            scan_result.eligible_files, scan_result.skipped_files
        )
    };

    info!(
        user_id = %req.user_id,
        project_id = %project.id.0,
        path = %req.codebase_path,
        eligible = scan_result.eligible_files,
        "Scanned codebase"
    );

    Ok(Json(ScanResponse {
        success: true,
        total_files: scan_result.total_files,
        eligible_files: scan_result.eligible_files,
        skipped_files: scan_result.skipped_files,
        limit_reached: scan_result.limit_reached,
        message,
    }))
}

/// POST /api/projects/{project_id}/index - Index codebase files
pub async fn index_project_codebase(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<IndexCodebaseRequest>,
) -> Result<Json<IndexResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let mut project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    if project.codebase_indexed && !req.force {
        return Err(AppError::InvalidInput {
            field: "force".to_string(),
            reason: "Codebase already indexed. Use force=true to re-index.".to_string(),
        });
    }

    let codebase_path = std::path::Path::new(&req.codebase_path);
    if !codebase_path.exists() {
        return Err(AppError::InvalidInput {
            field: "codebase_path".to_string(),
            reason: format!("Path does not exist: {}", req.codebase_path),
        });
    }

    if req.force && project.codebase_indexed {
        state
            .file_store
            .delete_project_files(&req.user_id, &project.id)
            .map_err(AppError::Internal)?;
    }

    let result = state
        .file_store
        .index_codebase(codebase_path, &project.id, &req.user_id, None)
        .map_err(AppError::Internal)?;

    project.codebase_path = Some(req.codebase_path.clone());
    project.codebase_indexed = true;
    project.codebase_indexed_at = Some(chrono::Utc::now());
    project.codebase_file_count = result.indexed_files;

    state
        .todo_store
        .store_project(&project)
        .map_err(AppError::Internal)?;

    let message = format!(
        "Indexed {} files ({} skipped, {} errors)",
        result.indexed_files,
        result.skipped_files,
        result.errors.len()
    );

    state.emit_event(MemoryEvent {
        event_type: "CODEBASE_INDEXED".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(project.id.0.to_string()),
        content_preview: Some(format!("{} files indexed", result.indexed_files)),
        memory_type: Some("Codebase".to_string()),
        importance: None,
        count: Some(result.indexed_files),
    });

    info!(
        user_id = %req.user_id,
        project_id = %project.id.0,
        indexed = result.indexed_files,
        "Indexed codebase"
    );

    Ok(Json(IndexResponse {
        success: true,
        result,
        message,
    }))
}

/// POST /api/projects/{project_id}/files/search - Search files
pub async fn search_project_files(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<SearchFilesRequest>,
) -> Result<Json<FileListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let all_files = state
        .file_store
        .list_by_project(&req.user_id, &project.id, None)
        .map_err(AppError::Internal)?;

    let query_lower = req.query.to_lowercase();
    let matching_files: Vec<_> = all_files
        .into_iter()
        .filter(|f| {
            f.path.to_lowercase().contains(&query_lower)
                || f.key_items
                    .iter()
                    .any(|k| k.to_lowercase().contains(&query_lower))
                || f.summary.to_lowercase().contains(&query_lower)
        })
        .take(req.limit)
        .collect();

    let total = matching_files.len();
    let summaries: Vec<FileMemorySummary> = matching_files
        .into_iter()
        .map(|f| {
            let heat_score = f.heat_score();
            FileMemorySummary {
                id: f.id.0.to_string(),
                path: f.path,
                absolute_path: f.absolute_path,
                file_type: format!("{:?}", f.file_type),
                summary: f.summary,
                key_items: f.key_items,
                access_count: f.access_count,
                last_accessed: f.last_accessed.to_rfc3339(),
                heat_score,
                size_bytes: f.size_bytes,
                line_count: f.line_count,
            }
        })
        .collect();

    Ok(Json(FileListResponse {
        success: true,
        files: summaries,
        total,
    }))
}

/// GET /api/files/stats - Get file memory statistics
pub async fn get_file_stats(
    State(state): State<AppState>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<FileStatsResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    let stats = state
        .file_store
        .stats(&query.user_id)
        .map_err(AppError::Internal)?;

    Ok(Json(FileStatsResponse {
        success: true,
        stats,
    }))
}
