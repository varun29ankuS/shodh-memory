//! Per-memory audit history.
//!
//! Every memory mutation in the product is written to the `audit` column family
//! by ~23 `log_event` call sites, and until this module existed the only path
//! from that data to any client was `POST /api/sessions/digest`, which reduces
//! the whole trail to a `HashMap<String, usize>` of event-type counts. The
//! per-memory reader written to serve it —
//! [`MultiUserMemoryManager::get_history`] — had zero callers repo-wide.
//!
//! This module answers "what happened to THIS memory". A per-user trail dump is
//! a different question over a different volume and is not served here.

use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::state::MultiUserMemoryManager;
use super::types::AuditEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::validation;

type AppState = Arc<MultiUserMemoryManager>;

/// Upper bound on events returned in one response, matching the list handlers.
const MAX_HISTORY_LIMIT: usize = 10_000;

fn default_history_limit() -> usize {
    100
}

/// Query parameters for `GET /api/memory/{memory_id}/history`.
#[derive(Debug, Deserialize)]
pub struct MemoryHistoryQuery {
    /// REQUIRED. Audit keys are `{user_id}:{timestamp}`, so this is what scopes
    /// the read to one tenant. A missing value is a 4xx rather than an
    /// unscoped answer.
    pub user_id: String,
    /// Max events to return, newest first. Defaults to 100, clamped to
    /// [`MAX_HISTORY_LIMIT`].
    #[serde(default = "default_history_limit")]
    pub limit: usize,
}

/// Response for `GET /api/memory/{memory_id}/history`.
#[derive(Debug, Serialize)]
pub struct MemoryHistoryResponse {
    pub success: bool,
    /// The id the events were filtered by. When the supplied value was a hex
    /// prefix of a memory that still exists, this is the resolved full UUID;
    /// otherwise it is the value as supplied.
    pub memory_id: String,
    /// Newest first.
    pub events: Vec<AuditEvent>,
    pub count: usize,
}

/// GET /api/memory/{memory_id}/history?user_id={user}&limit={n}
///
/// The audit trail for one memory, newest first.
///
/// A missing memory is NOT an error. Deletion is itself an audited event, so
/// the history of a memory that no longer exists is exactly the case a per-
/// memory audit reader has to serve — resolving the id is a convenience for
/// prefix input, not an existence check. Callers get an empty `events` array
/// for an id that never existed and for one whose entries have been rotated
/// out; those are indistinguishable here by design, because the alternative is
/// an endpoint that tells an unauthorised caller which ids are real.
pub async fn get_memory_history(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Query(query): Query<MemoryHistoryQuery>,
) -> Result<Json<MemoryHistoryResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;
    validation::validate_memory_id_or_prefix(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    // Audit keys carry the full UUID, so an 8+ char prefix has to be resolved
    // against the store before it can be matched. If it resolves to nothing the
    // supplied value is used verbatim — see the note above on deleted memories.
    let resolved_id = state
        .get_user_memory(&query.user_id)
        .ok()
        .and_then(|memory| {
            let guard = memory.read();
            guard.find_memory_by_prefix(&memory_id).ok().flatten()
        })
        .map(|m| m.id.0.to_string())
        .unwrap_or_else(|| memory_id.clone());

    let limit = query.limit.min(MAX_HISTORY_LIMIT);
    let events = state.get_history(&query.user_id, Some(&resolved_id), limit);
    let count = events.len();

    Ok(Json(MemoryHistoryResponse {
        success: true,
        memory_id: resolved_id,
        events,
        count,
    }))
}
