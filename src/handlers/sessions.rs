//! Session Management Handlers
//!
//! Handlers for user session tracking and management.

use axum::{
    extract::{Path, Query, State},
    response::Json,
};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{Session, SessionId, SessionStatus, SessionStoreStats, SessionSummary};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

fn default_sessions_limit() -> usize {
    10
}

fn default_end_reason() -> String {
    "user_ended".to_string()
}

/// Request for listing sessions
#[derive(Debug, Deserialize)]
pub struct ListSessionsRequest {
    pub user_id: String,
    #[serde(default = "default_sessions_limit")]
    pub limit: usize,
}

/// Response for listing sessions
#[derive(Debug, Serialize)]
pub struct ListSessionsResponse {
    pub success: bool,
    pub sessions: Vec<SessionSummary>,
    pub count: usize,
}

/// Request for getting a specific session
#[derive(Debug, Deserialize)]
pub struct GetSessionRequest {
    pub user_id: String,
}

/// Response for getting a session
#[derive(Debug, Serialize)]
pub struct GetSessionResponse {
    pub success: bool,
    pub session: Option<Session>,
}

/// Request for ending a session
#[derive(Debug, Deserialize)]
pub struct EndSessionRequest {
    pub user_id: String,
    #[serde(default = "default_end_reason")]
    pub reason: String,
}

/// Response for ending a session
#[derive(Debug, Serialize)]
pub struct EndSessionResponse {
    pub success: bool,
    pub session: Option<Session>,
}

/// Response for session store stats
#[derive(Debug, Serialize)]
pub struct SessionStoreStatsResponse {
    pub success: bool,
    pub stats: SessionStoreStats,
}

/// POST /api/sessions - List sessions for a user
pub async fn list_sessions(
    State(state): State<AppState>,
    Json(req): Json<ListSessionsRequest>,
) -> Result<Json<ListSessionsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let sessions = state
        .session_store
        .get_user_sessions(&req.user_id, req.limit);
    let count = sessions.len();

    Ok(Json(ListSessionsResponse {
        success: true,
        sessions,
        count,
    }))
}

/// GET /api/sessions/{session_id} - Get a specific session
pub async fn get_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Query(req): Query<GetSessionRequest>,
) -> Result<Json<GetSessionResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let uuid = uuid::Uuid::parse_str(&session_id).map_err(|e| AppError::InvalidInput {
        field: "session_id".to_string(),
        reason: format!("Invalid UUID: {e}"),
    })?;
    let sid = SessionId(uuid);
    let session = state.session_store.get_session(&sid);

    Ok(Json(GetSessionResponse {
        success: session.is_some(),
        session,
    }))
}

/// POST /api/sessions/end - End the current/active session for a user
pub async fn end_session(
    State(state): State<AppState>,
    Json(req): Json<EndSessionRequest>,
) -> Result<Json<EndSessionResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let sessions = state.session_store.get_user_sessions(&req.user_id, 1);
    let active_session = sessions
        .into_iter()
        .find(|s| matches!(s.status, SessionStatus::Active));

    if let Some(summary) = active_session {
        let session = state.session_store.end_session(&summary.id, &req.reason);
        Ok(Json(EndSessionResponse {
            success: session.is_some(),
            session,
        }))
    } else {
        Ok(Json(EndSessionResponse {
            success: false,
            session: None,
        }))
    }
}

/// GET /api/sessions/stats - Get overall session store statistics
pub async fn get_session_stats(
    State(state): State<AppState>,
) -> Result<Json<SessionStoreStatsResponse>, AppError> {
    let stats = state.session_store.stats();

    Ok(Json(SessionStoreStatsResponse {
        success: true,
        stats,
    }))
}
