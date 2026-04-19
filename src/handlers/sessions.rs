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
use crate::memory::{
    Experience, ExperienceType, Session, SessionDigest, SessionEvent, SessionId, SessionStatus,
    SessionStoreStats, SessionSummary,
};
use crate::validation;
use std::collections::HashMap;
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

// =============================================================================
// Session Digest
// =============================================================================

fn default_token_budget() -> usize {
    100_000
}

/// Request for session digest
#[derive(Debug, Deserialize)]
pub struct DigestRequest {
    pub user_id: String,
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
}

/// Response for session digest
#[derive(Debug, Serialize)]
pub struct DigestResponse {
    pub success: bool,
    pub digest: Option<SessionDigest>,
}

/// POST /api/sessions/digest - Get digest of active session
pub async fn get_session_digest(
    State(state): State<AppState>,
    Json(req): Json<DigestRequest>,
) -> Result<Json<DigestResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let digest = build_active_digest_with_budget(&state, &req.user_id, req.token_budget);

    Ok(Json(DigestResponse {
        success: digest.is_some(),
        digest,
    }))
}

/// Request for context compression signal
#[derive(Debug, Deserialize)]
pub struct ContextCompressedRequest {
    pub user_id: String,
    #[serde(default)]
    pub tokens_before: usize,
    #[serde(default)]
    pub tokens_after: usize,
}

/// Response for context compression signal
#[derive(Debug, Serialize)]
pub struct ContextCompressedResponse {
    pub success: bool,
}

/// POST /api/sessions/context-compressed - Record a context compression event
/// Also persists a session digest snapshot as a Context memory for future recall.
pub async fn context_compressed(
    State(state): State<AppState>,
    Json(req): Json<ContextCompressedRequest>,
) -> Result<Json<ContextCompressedResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let now = chrono::Utc::now();

    state.session_store.add_event_to_user(
        &req.user_id,
        SessionEvent::ContextCompressed {
            timestamp: now,
            tokens_before: req.tokens_before,
            tokens_after: req.tokens_after,
        },
    );

    // Persist a digest snapshot as a Context memory so it survives restarts
    // and can be recalled in future sessions ("what was I working on?")
    if let Some(digest) = build_active_digest(&state, &req.user_id) {
        let reduction_pct = if req.tokens_before > 0 {
            ((req.tokens_before.saturating_sub(req.tokens_after)) * 100) / req.tokens_before
        } else {
            0
        };

        let entities_str = if digest.entities_extracted.is_empty() {
            "none".to_string()
        } else {
            digest.entities_extracted.join(", ")
        };

        let duration_mins = digest.duration_secs / 60;
        let content = format!(
            "Session digest at context compression ({now}):\n\
             Duration: {duration_mins}m | Tokens: {before}→{after} ({reduction_pct}% reduced)\n\
             Memories: {created} created, {surfaced} surfaced, {used} used (hit rate {hit_rate:.0}%)\n\
             Todos: {todos_created} created, {todos_completed} completed\n\
             Entities: {entities_str}\n\
             Topic changes: {topics} | Compressions: {compressions}",
            before = req.tokens_before,
            after = req.tokens_after,
            created = digest.memories_created,
            surfaced = digest.memories_surfaced,
            used = digest.memories_used,
            hit_rate = digest.memory_hit_rate * 100.0,
            todos_created = digest.todos_created,
            todos_completed = digest.todos_completed,
            topics = digest.topic_changes,
            compressions = digest.compressions,
        );

        let mut metadata = HashMap::new();
        metadata.insert("session_digest".to_string(), "true".to_string());
        metadata.insert("session_id".to_string(), digest.session_id.0.to_string());
        metadata.insert("started_at".to_string(), digest.started_at.to_rfc3339());
        metadata.insert(
            "duration_secs".to_string(),
            digest.duration_secs.to_string(),
        );
        metadata.insert(
            "memories_created".to_string(),
            digest.memories_created.to_string(),
        );
        metadata.insert("tokens_before".to_string(), req.tokens_before.to_string());
        metadata.insert("tokens_after".to_string(), req.tokens_after.to_string());

        // Include "session-summary" tag so these show up in session_history queries,
        // alongside "session-digest" to distinguish compression-time snapshots from
        // session-end summaries.
        let mut entities = digest.entities_extracted;
        entities.push("session-summary".to_string());
        entities.push("session-digest".to_string());

        let experience = Experience {
            experience_type: ExperienceType::Context,
            content,
            entities,
            metadata,
            ..Default::default()
        };

        // Best-effort persistence — don't fail the compression signal if memory write fails
        if let Ok(user_memory) = state.get_user_memory(&req.user_id) {
            let ms = user_memory.read();
            let _ = ms.remember(experience, Some(now));
        }
    }

    Ok(Json(ContextCompressedResponse { success: true }))
}

/// Build a digest for the active session, enriched with audit log tool counts.
fn build_active_digest(state: &AppState, user_id: &str) -> Option<SessionDigest> {
    build_active_digest_with_budget(state, user_id, 0)
}

fn build_active_digest_with_budget(
    state: &AppState,
    user_id: &str,
    token_budget: usize,
) -> Option<SessionDigest> {
    let sessions = state.session_store.get_user_sessions(user_id, 1);
    let active_summary = sessions
        .into_iter()
        .find(|s| matches!(s.status, SessionStatus::Active))?;

    let session = state.session_store.get_session(&active_summary.id)?;
    let mut digest = session.digest(token_budget);

    // Enrich with audit log tool counts (cap at 1000 entries to bound scan cost)
    let audit_entries = state.get_audit_logs(user_id, 1000);
    if !audit_entries.is_empty() {
        let mut tool_counts: HashMap<String, usize> = HashMap::new();
        for entry in &audit_entries {
            if entry.timestamp >= session.started_at {
                *tool_counts.entry(entry.event_type.clone()).or_insert(0) += 1;
            }
        }
        digest.tools_used = tool_counts;
    }

    Some(digest)
}

// =============================================================================
// Session History
// =============================================================================

fn default_history_limit() -> usize {
    10
}

/// Request for session history
#[derive(Debug, Deserialize)]
pub struct SessionHistoryRequest {
    pub user_id: String,
    #[serde(default = "default_history_limit")]
    pub limit: usize,
    #[serde(default)]
    pub group_by_project: bool,
}

/// A single session entry in the history
#[derive(Debug, Serialize)]
pub struct SessionHistoryEntry {
    pub session_id: Option<String>,
    pub content: String,
    pub entities: Vec<String>,
    pub started_at: Option<String>,
    pub duration_secs: Option<i64>,
    pub memories_created: Option<usize>,
    pub created_at: String,
}

/// A detected project thread spanning multiple sessions
#[derive(Debug, Serialize)]
pub struct ProjectThread {
    pub name: String,
    pub sessions: Vec<usize>,
    pub shared_entities: Vec<String>,
    pub session_count: usize,
}

/// Response for session history
#[derive(Debug, Serialize)]
pub struct SessionHistoryResponse {
    pub success: bool,
    pub sessions: Vec<SessionHistoryEntry>,
    pub project_threads: Vec<ProjectThread>,
    pub total: usize,
}

/// POST /api/sessions/history - Get persisted session history with project continuity
pub async fn session_history(
    State(state): State<AppState>,
    Json(req): Json<SessionHistoryRequest>,
) -> Result<Json<SessionHistoryResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    let limit = req.limit.min(100);
    let group = req.group_by_project;

    let result = tokio::task::spawn_blocking(move || {
        let ms = memory.read();
        let tags = vec!["session-summary".to_string()];

        // Fetch ALL matching memories (recall_by_tags truncates on random HashSet order,
        // so we over-fetch with a high cap and sort/truncate ourselves).
        let memories = ms.recall_by_tags(&tags, 500)?;

        let mut entries: Vec<SessionHistoryEntry> = memories
            .iter()
            .map(|m| {
                let meta = &m.experience.metadata;
                SessionHistoryEntry {
                    session_id: meta.get("session_id").cloned(),
                    content: m.experience.content.clone(),
                    entities: m.experience.entities.clone(),
                    started_at: meta.get("started_at").cloned(),
                    duration_secs: meta.get("duration_secs").and_then(|s| s.parse().ok()),
                    memories_created: meta.get("memories_created").and_then(|s| s.parse().ok()),
                    created_at: m.created_at.to_rfc3339(),
                }
            })
            .collect();

        // Sort newest-first, then deduplicate by session_id.
        // A single session can produce both a compression digest ("session-digest" tag)
        // and a hook stop summary ("source:hook" tag). Keep only the newest entry per
        // session (hook stop comes after compression, so it's more complete).
        entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        let mut seen_sessions: std::collections::HashSet<String> = std::collections::HashSet::new();
        entries.retain(|e| {
            match &e.session_id {
                Some(sid) => seen_sessions.insert(sid.clone()),
                // Entries without session_id (legacy) are always kept
                None => true,
            }
        });

        entries.truncate(limit);

        let threads = if group {
            compute_project_threads(&entries)
        } else {
            vec![]
        };

        let total = entries.len();
        Ok::<_, anyhow::Error>(SessionHistoryResponse {
            success: true,
            sessions: entries,
            project_threads: threads,
            total,
        })
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))??;

    Ok(Json(result))
}

/// Detect project threads by clustering sessions with overlapping entities.
///
/// Uses union-find: sessions sharing >= 3 entities merge into the same cluster.
/// Clusters with 2+ sessions become ProjectThread.
fn compute_project_threads(entries: &[SessionHistoryEntry]) -> Vec<ProjectThread> {
    use std::collections::HashSet;

    // Jaccard threshold for entity overlap — analogous to the thalamic reticular
    // nucleus gating irrelevant stimuli before cortical processing. Raw count
    // overlap (e.g., 3/30 entities = 10%) causes sessions with many entities to
    // trivially match. Jaccard normalizes by set union, requiring genuine semantic
    // overlap to cluster sessions together.
    const JACCARD_THRESHOLD: f32 = 0.15;

    // Noise entities filtered before clustering — the attentional gate.
    // These appear in every session (system boilerplate, auto-extraction artifacts,
    // generic terms) and would inflate overlap counts if not filtered, causing
    // every session to cluster together. Analogous to how the PFC suppresses
    // habitual/background stimuli (e.g., the hum of a fridge) to focus on signal.
    const NOISE_ENTITIES: &[&str] = &[
        // System tags
        "session-summary",
        "session-digest",
        "source:hook",
        // Auto-extraction artifacts
        "auto-extract",
        "source:transcript",
        // Session summary boilerplate entities
        "completed entities",
        "topics changed",
        "hit rate",
        "created",
        "memories",
        "todos",
        "compressions",
        "context",
        "proactive",
        "ended",
        "tools",
        // Generic terms appearing in most sessions
        "file",
        "source",
        "cargo.toml",
    ];

    if entries.len() < 2 {
        return vec![];
    }

    let n = entries.len();
    let mut parent: Vec<usize> = (0..n).collect();

    // Build entity sets once, filtering out noise entities (attentional gate)
    let entity_sets: Vec<HashSet<&str>> = entries
        .iter()
        .map(|e| {
            e.entities
                .iter()
                .map(|s| s.as_str())
                .filter(|s| !NOISE_ENTITIES.contains(s))
                .collect()
        })
        .collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let overlap = entity_sets[i].intersection(&entity_sets[j]).count();
            let union = entity_sets[i].union(&entity_sets[j]).count();
            let jaccard = if union > 0 {
                overlap as f32 / union as f32
            } else {
                0.0
            };
            if jaccard >= JACCARD_THRESHOLD {
                let ri = uf_find_root(&parent, i);
                let rj = uf_find_root(&parent, j);
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }

    // Group by root
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        groups.entry(uf_find_root(&parent, i)).or_default().push(i);
    }

    // Build threads from groups with 2+ sessions
    groups
        .into_values()
        .filter(|indices| indices.len() >= 2)
        .map(|indices| {
            let mut entity_counts: HashMap<&str, usize> = HashMap::new();
            for &i in &indices {
                for e in &entries[i].entities {
                    if !NOISE_ENTITIES.contains(&e.as_str()) {
                        *entity_counts.entry(e.as_str()).or_default() += 1;
                    }
                }
            }
            // Shared = entities appearing in 2+ sessions of this thread
            let mut shared: Vec<String> = entity_counts
                .iter()
                .filter(|(_, &c)| c >= 2)
                .map(|(&e, _)| e.to_string())
                .collect();
            shared.sort();

            // Thread naming: pick the most-frequent entity that is semantically
            // meaningful (>= 4 chars, not noise). Short fragments like "sho",
            // "dh", "mem" are typically entity extraction artifacts, not topics.
            let name = entity_counts
                .iter()
                .filter(|(&e, _)| e.len() >= 4 && !NOISE_ENTITIES.contains(&e))
                .max_by_key(|(_, c)| *c)
                .map(|(&e, _)| e.to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            let session_count = indices.len();
            ProjectThread {
                name,
                sessions: indices,
                shared_entities: shared,
                session_count,
            }
        })
        .collect()
}

fn uf_find_root(parent: &[usize], mut i: usize) -> usize {
    while parent[i] != i {
        i = parent[i];
    }
    i
}
