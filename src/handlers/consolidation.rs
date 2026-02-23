//! Consolidation, Index Maintenance, and Backup Handlers
//!
//! Semantic consolidation for fact extraction, vector index maintenance,
//! and backup/restore operations.

use axum::{extract::State, response::Json};

use super::state::MultiUserMemoryManager;
use super::types::{
    BackupResponse, CleanupCorruptedRequest, CleanupCorruptedResponse, ConsolidateRequest,
    ConsolidateResponse, CreateBackupRequest, ListBackupsRequest, ListBackupsResponse, MemoryEvent,
    MigrateLegacyRequest, MigrateLegacyResponse, PurgeBackupsRequest, PurgeBackupsResponse,
    RebuildIndexRequest, RebuildIndexResponse, RepairIndexRequest, RepairIndexResponse,
    VerifyBackupRequest, VerifyBackupResponse, VerifyIndexRequest,
};
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory;
use crate::metrics;
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// SEMANTIC CONSOLIDATION
// =============================================================================

/// Consolidate memories into semantic facts (SHO-AUD-7)
///
/// Spawns the full pipeline (fact extraction → replay → edge strengthening) as a
/// background task and returns immediately with 202 Accepted. This avoids the 60s
/// HTTP timeout killing the handler mid-flight for large memory stores.
/// Results are logged server-side and visible via `/api/consolidation/report`.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn consolidate_memories(
    State(state): State<AppState>,
    Json(req): Json<ConsolidateRequest>,
) -> Result<Json<ConsolidateResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Validate user exists before spawning background work
    let _ = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let min_support = req.min_support;
    let min_age_days = req.min_age_days;
    let state_clone = state.clone();

    // Spawn the entire pipeline as a detached background task.
    // This survives HTTP timeout cancellation — the work always completes.
    tokio::task::spawn(async move {
        let op_start = std::time::Instant::now();

        let memory = match state_clone.get_user_memory(&user_id) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(user_id = %user_id, "Consolidation: failed to get memory: {e}");
                return;
            }
        };

        // Step 1: Fact extraction
        let result = {
            let memory = memory.clone();
            let uid = user_id.clone();
            match tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                memory_guard.distill_facts(&uid, min_support, min_age_days)
            })
            .await
            {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    tracing::error!(user_id = %user_id, "Consolidation fact extraction failed: {e}");
                    return;
                }
                Err(e) => {
                    tracing::error!(user_id = %user_id, "Consolidation fact extraction panicked: {e}");
                    return;
                }
            }
        };

        // Step 2: Maintenance (replay + tier consolidation + decay)
        let decay_factor = state_clone.server_config().activation_decay_factor;
        let maintenance_result = {
            let memory = memory.clone();
            let uid = user_id.clone();
            match tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                memory_guard.run_maintenance(decay_factor, &uid, true)
            })
            .await
            {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    tracing::error!(user_id = %user_id, "Consolidation maintenance failed: {e}");
                    return;
                }
                Err(e) => {
                    tracing::error!(user_id = %user_id, "Consolidation maintenance panicked: {e}");
                    return;
                }
            }
        };

        // Step 3: Apply graph strengthening from replay results
        let mut edges_strengthened: usize = 0;
        let mut entity_edges_strengthened: usize = 0;

        // Direction 1: Edge strengthening + promotion boost propagation
        if !maintenance_result.edge_boosts.is_empty() {
            if let Ok(graph) = state_clone.get_user_graph(&user_id) {
                let graph_guard = graph.read();
                match graph_guard.strengthen_memory_edges(&maintenance_result.edge_boosts) {
                    Ok((count, promotion_boosts)) => {
                        edges_strengthened += count;
                        if !promotion_boosts.is_empty() {
                            let memory_guard = memory.read();
                            let _ = memory_guard.apply_edge_promotion_boosts(&promotion_boosts);
                        }
                    }
                    Err(e) => {
                        tracing::debug!("On-demand edge boost failed: {e}");
                    }
                }
            }
        }

        // Direction 3: Entity-entity Hebbian reinforcement for replayed memories
        if !maintenance_result.replay_memory_ids.is_empty() {
            if let Ok(graph) = state_clone.get_user_graph(&user_id) {
                let graph_guard = graph.read();
                for mem_id_str in &maintenance_result.replay_memory_ids {
                    if let Ok(uuid) = uuid::Uuid::parse_str(mem_id_str) {
                        match graph_guard.strengthen_episode_entity_edges(&uuid) {
                            Ok(count) => entity_edges_strengthened += count,
                            Err(e) => {
                                tracing::debug!(
                                    "Entity edge strengthening failed for {mem_id_str}: {e}"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Direction 2: Lazy decay — flush opportunistic pruning queue
        if let Ok(graph) = state_clone.get_user_graph(&user_id) {
            let graph_guard = graph.read();
            let _ = graph_guard.flush_pending_maintenance();
        }

        let duration = op_start.elapsed().as_secs_f64();
        metrics::CONSOLIDATE_DURATION.observe(duration);
        metrics::CONSOLIDATE_TOTAL
            .with_label_values(&["success"])
            .inc();

        tracing::info!(
            user_id = %user_id,
            memories_processed = result.memories_processed,
            facts_extracted = result.facts_extracted,
            facts_reinforced = result.facts_reinforced,
            memories_replayed = maintenance_result.replay_memory_ids.len(),
            edges_strengthened,
            entity_edges_strengthened,
            memories_decayed = maintenance_result.decayed_count,
            duration_secs = format!("{:.1}", duration),
            "Consolidation complete (background)"
        );
    });

    // Return immediately — work continues in background
    Ok(Json(ConsolidateResponse {
        memories_analyzed: 0,
        facts_extracted: 0,
        facts_reinforced: 0,
        fact_ids: vec![],
        memories_replayed: 0,
        edges_strengthened: 0,
        entity_edges_strengthened: 0,
        memories_decayed: 0,
        warnings: vec![
            "Consolidation started in background. Check /api/consolidation/report for results."
                .to_string(),
        ],
    }))
}

// =============================================================================
// INDEX MAINTENANCE
// =============================================================================

/// Verify vector index integrity - diagnose orphaned memories
pub async fn verify_index_integrity(
    State(state): State<AppState>,
    Json(req): Json<VerifyIndexRequest>,
) -> Result<Json<memory::IndexIntegrityReport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let report = memory_guard
        .verify_index_integrity()
        .map_err(AppError::Internal)?;

    Ok(Json(report))
}

/// Repair vector index - re-index orphaned memories
pub async fn repair_vector_index(
    State(state): State<AppState>,
    Json(req): Json<RepairIndexRequest>,
) -> Result<Json<RepairIndexResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (total_storage, total_indexed, repaired, failed) = memory_guard
        .repair_vector_index()
        .map_err(AppError::Internal)?;

    Ok(Json(RepairIndexResponse {
        success: failed == 0,
        total_storage,
        total_indexed,
        repaired,
        failed,
        is_healthy: total_storage == total_indexed,
    }))
}

/// Cleanup corrupted memories that fail to deserialize
pub async fn cleanup_corrupted(
    State(state): State<AppState>,
    Json(req): Json<CleanupCorruptedRequest>,
) -> Result<Json<CleanupCorruptedResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let deleted_count = memory_guard
        .cleanup_corrupted()
        .map_err(AppError::Internal)?;

    // Broadcast DELETE event for real-time dashboard so TUI updates its count
    if deleted_count > 0 {
        state.emit_event(MemoryEvent {
            event_type: "DELETE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: req.user_id.clone(),
            memory_id: None,
            content_preview: Some(format!("cleanup: {} corrupted entries", deleted_count)),
            memory_type: None,
            importance: None,
            count: Some(deleted_count),
            results: None,
        });
    }

    Ok(Json(CleanupCorruptedResponse {
        success: true,
        deleted_count,
    }))
}

/// Migrate legacy memories to current format
/// This converts old storage formats to the current schema without data loss
pub async fn migrate_legacy(
    State(state): State<AppState>,
    Json(req): Json<MigrateLegacyRequest>,
) -> Result<Json<MigrateLegacyResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (migrated, already_current, failed) =
        memory_guard.migrate_legacy().map_err(AppError::Internal)?;

    // Broadcast event for real-time dashboard
    if migrated > 0 {
        state.emit_event(MemoryEvent {
            event_type: "MIGRATE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: req.user_id.clone(),
            memory_id: None,
            content_preview: Some(format!(
                "migrated {} memories, {} already current, {} failed",
                migrated, already_current, failed
            )),
            memory_type: None,
            importance: None,
            count: Some(migrated),
            results: None,
        });
    }

    Ok(Json(MigrateLegacyResponse {
        success: true,
        migrated_count: migrated,
        already_current_count: already_current,
        failed_count: failed,
    }))
}

/// Rebuild vector index from storage (removes orphaned index entries)
pub async fn rebuild_index(
    State(state): State<AppState>,
    Json(req): Json<RebuildIndexRequest>,
) -> Result<Json<RebuildIndexResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (storage_count, indexed_count) =
        memory_guard.rebuild_index().map_err(AppError::Internal)?;

    Ok(Json(RebuildIndexResponse {
        success: true,
        storage_count,
        indexed_count,
        is_healthy: storage_count == indexed_count,
    }))
}

// =============================================================================
// BACKUP & RESTORE
// =============================================================================

/// Create a comprehensive backup for a user (memories + secondary stores)
pub async fn create_backup(
    State(state): State<AppState>,
    Json(req): Json<CreateBackupRequest>,
) -> Result<Json<BackupResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let db = memory_guard.get_db();

    // Collect secondary store DB references for comprehensive backup
    let secondary_refs = state.collect_secondary_store_refs();
    let store_refs: Vec<crate::backup::SecondaryStoreRef<'_>> = secondary_refs
        .iter()
        .map(|(name, db)| crate::backup::SecondaryStoreRef { name, db })
        .collect();

    let result = if store_refs.is_empty() {
        state.backup_engine().create_backup(&db, &req.user_id)
    } else {
        state
            .backup_engine()
            .create_comprehensive_backup(&db, &req.user_id, &store_refs)
    };

    match result {
        Ok(metadata) => {
            let secondary_count = metadata.secondary_stores.len();
            state.log_event(
                &req.user_id,
                "BACKUP_CREATED",
                &metadata.backup_id.to_string(),
                &format!(
                    "Backup created: {} bytes + {} secondary stores ({} bytes)",
                    metadata.size_bytes, secondary_count, metadata.secondary_size_bytes
                ),
            );
            Ok(Json(BackupResponse {
                success: true,
                backup: Some(metadata),
                message: "Backup created successfully".to_string(),
            }))
        }
        Err(e) => Ok(Json(BackupResponse {
            success: false,
            backup: None,
            message: format!("Backup failed: {}", e),
        })),
    }
}

/// List all backups for a user
pub async fn list_backups(
    State(state): State<AppState>,
    Json(req): Json<ListBackupsRequest>,
) -> Result<Json<ListBackupsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state.backup_engine().list_backups(&req.user_id) {
        Ok(backups) => {
            let count = backups.len();
            Ok(Json(ListBackupsResponse {
                success: true,
                backups,
                count,
            }))
        }
        Err(e) => Err(AppError::Internal(e)),
    }
}

/// Verify backup integrity
pub async fn verify_backup(
    State(state): State<AppState>,
    Json(req): Json<VerifyBackupRequest>,
) -> Result<Json<VerifyBackupResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .backup_engine()
        .verify_backup(&req.user_id, req.backup_id)
    {
        Ok(is_valid) => Ok(Json(VerifyBackupResponse {
            success: true,
            is_valid,
            message: if is_valid {
                "Backup integrity verified".to_string()
            } else {
                "Backup checksum mismatch - may be corrupted".to_string()
            },
        })),
        Err(e) => Ok(Json(VerifyBackupResponse {
            success: false,
            is_valid: false,
            message: format!("Verification failed: {}", e),
        })),
    }
}

/// Purge old backups
pub async fn purge_backups(
    State(state): State<AppState>,
    Json(req): Json<PurgeBackupsRequest>,
) -> Result<Json<PurgeBackupsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .backup_engine()
        .purge_old_backups(&req.user_id, req.keep_count)
    {
        Ok(purged_count) => {
            if purged_count > 0 {
                state.log_event(
                    &req.user_id,
                    "BACKUP_PURGE",
                    &format!("keep_{}", req.keep_count),
                    &format!("Purged {} old backups", purged_count),
                );
            }
            Ok(Json(PurgeBackupsResponse {
                success: true,
                purged_count,
            }))
        }
        Err(e) => Err(AppError::Internal(e)),
    }
}

// =============================================================================
// CONSOLIDATION INTROSPECTION
// =============================================================================

use serde::Deserialize;

/// Request for consolidation report
#[derive(Debug, Deserialize)]
pub struct ConsolidationReportRequest {
    pub user_id: String,
    #[serde(default)]
    pub since: Option<String>,
    #[serde(default)]
    pub until: Option<String>,
}

/// Request for consolidation events
#[derive(Debug, Deserialize)]
pub struct ConsolidationEventsRequest {
    pub user_id: String,
    #[serde(default)]
    pub since: Option<String>,
}

/// POST /api/consolidation/report - Get consolidation introspection report
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn get_consolidation_report(
    State(state): State<AppState>,
    Json(req): Json<ConsolidationReportRequest>,
) -> Result<Json<memory::ConsolidationReport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let now = chrono::Utc::now();
    let since = if let Some(since_str) = &req.since {
        chrono::DateTime::parse_from_rfc3339(since_str)
            .map_err(|e| AppError::InvalidInput {
                field: "since".to_string(),
                reason: format!("Invalid timestamp: {}", e),
            })?
            .with_timezone(&chrono::Utc)
    } else {
        now - chrono::Duration::hours(1)
    };

    let until = if let Some(until_str) = &req.until {
        Some(
            chrono::DateTime::parse_from_rfc3339(until_str)
                .map_err(|e| AppError::InvalidInput {
                    field: "until".to_string(),
                    reason: format!("Invalid timestamp: {}", e),
                })?
                .with_timezone(&chrono::Utc),
        )
    } else {
        None
    };

    let report = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_consolidation_report(since, until)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    Ok(Json(report))
}

/// GET /api/consolidation/events - Get raw consolidation events since a timestamp
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn get_consolidation_events(
    State(state): State<AppState>,
    axum::extract::Query(req): axum::extract::Query<ConsolidationEventsRequest>,
) -> Result<Json<Vec<memory::ConsolidationEvent>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let now = chrono::Utc::now();
    let since = if let Some(since_str) = &req.since {
        chrono::DateTime::parse_from_rfc3339(since_str)
            .map_err(|e| AppError::InvalidInput {
                field: "since".to_string(),
                reason: format!("Invalid timestamp: {}", e),
            })?
            .with_timezone(&chrono::Utc)
    } else {
        now - chrono::Duration::hours(1)
    };

    let events = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_consolidation_events_since(since)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    Ok(Json(events))
}
