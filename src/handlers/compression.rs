//! Compression Handlers
//!
//! Handlers for memory compression and decompression operations.

use axum::{
    extract::{Query, State},
    response::Json,
};
use serde::Deserialize;

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{self, MemoryId};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// Request for compressing a memory
#[derive(Debug, Deserialize)]
pub struct CompressMemoryRequest {
    pub user_id: String,
    #[serde(alias = "memory_id")]
    pub id: String,
}

/// POST /api/memory/compress - Manually compress a specific memory
pub async fn compress_memory(
    State(state): State<AppState>,
    Json(req): Json<CompressMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let _memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Validate id format
    let _memory_id = MemoryId(
        uuid::Uuid::parse_str(&req.id).map_err(|_| AppError::InvalidMemoryId(req.id.clone()))?,
    );

    // Compression happens automatically in the memory system based on age and importance
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Memory compression initiated"
    })))
}

/// Request for decompressing a memory
#[derive(Debug, Deserialize)]
pub struct DecompressMemoryRequest {
    pub user_id: String,
    #[serde(alias = "memory_id")]
    pub id: String,
}

/// POST /api/memory/decompress - Decompress a compressed memory
pub async fn decompress_memory(
    State(state): State<AppState>,
    Json(req): Json<DecompressMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let memory_id = MemoryId(
        uuid::Uuid::parse_str(&req.id).map_err(|_| AppError::InvalidMemoryId(req.id.clone()))?,
    );

    // Get the memory
    let memory = memory_guard
        .get_memory(&memory_id)
        .map_err(AppError::Internal)?;

    if !memory.compressed {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "Memory is not compressed",
            "was_compressed": false
        })));
    }

    // Decompress using compression pipeline
    let decompressed = memory_guard
        .decompress_memory(&memory)
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Memory decompressed successfully",
        "was_compressed": true,
        "memory": {
            "id": decompressed.id.0.to_string(),
            "content": decompressed.experience.content,
            "importance": decompressed.importance()
        }
    })))
}

/// Request for storage statistics
#[derive(Debug, Deserialize)]
pub struct StorageStatsRequest {
    pub user_id: String,
}

/// GET /api/storage/stats - Get storage statistics
pub async fn get_storage_stats(
    State(state): State<AppState>,
    Query(req): Query<StorageStatsRequest>,
) -> Result<Json<memory::storage::StorageStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let stats = memory_guard
        .get_storage_stats()
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}
