//! Ingestion source registry API.
//!
//! # What this replaces
//!
//! The `/sources` screen that exists today is a proxy: it reads session
//! history, the MIF adapter list and memory counts, and *infers* a source by
//! looking for a `source:hook` string in a memory's tag list. Its own module
//! doc says there is no connector subsystem to read. It is not a surface to
//! design to — it is the surface this replaces.
//!
//! Three pieces of its judgement are worth keeping, and are carried into the
//! wire format here so a client cannot get them wrong:
//!
//! 1. **Never print a confident zero.** `last_run` is `null` when a source has
//!    never run, not a zero-filled object. A zero-filled object reads as "this
//!    source delivered nothing" when the truth is "nothing recorded whether it
//!    did".
//! 2. **A floor is distinguishable from a total.** Run listings carry a true
//!    `total` next to the page, and the failure list on a run is explicitly
//!    labelled a sample with its cap stated.
//! 3. **Never infer a writer.** `origin` is server-observed at write time. A
//!    memory this connector did not write is never counted as its output.
//!
//! And one question the SaaS framing misses. "Is my data flowing" and "is it
//! current" are the right two questions for a dashboard; on a machine holding
//! a corpus somebody has to defend, the first question in the room is **"what
//! is this thing reading?"** — answered here by the echoed canonical `root`,
//! `items_seen`, and `items_denied_by_policy`.
//!
//! # Trust boundary
//!
//! `GET /api/sources/{user_id}` echoes a filesystem path to any holder of the
//! shared API key. That is the same boundary `GET /api/list/{user_id}` already
//! sits on — it returns memory *content* to the same holder — and
//! `auth::validate_api_key` returns `Result<(), AuthError>` without learning
//! which key matched, so there is no principal to gate on. Whether
//! `POST /api/sources`, which grants filesystem read, deserves a stronger
//! credential is a change to authentication, not to this subsystem, and is not
//! made here.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};

use super::router::AppState;
use crate::errors::{AppError, ValidationErrorExt};
use crate::ingest::folder;
use crate::memory::sources::{
    ItemCursor, SourceConfig, SourceDefinition, SourceId, SourceKind, SourceRun,
    RUN_FAILURE_SAMPLE,
};
use crate::validation;

const DEFAULT_RUN_LIMIT: usize = 20;
const DEFAULT_ITEM_LIMIT: usize = 100;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct WatchedFolderConfigBody {
    pub root: String,
    #[serde(default)]
    pub include_globs: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_globs: Option<Vec<String>>,
    #[serde(default)]
    pub max_depth: Option<u16>,
    #[serde(default)]
    pub max_files_per_run: Option<u32>,
    #[serde(default)]
    pub max_file_bytes: Option<u64>,
    #[serde(default)]
    pub max_run_bytes: Option<u64>,
    #[serde(default)]
    pub rehash_every_n_runs: Option<u32>,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct CreateSourceRequest {
    pub user_id: String,
    pub name: String,
    /// Wire name of a [`SourceKind`]. Only `watched_folder` exists today; an
    /// unknown value is rejected rather than defaulted, so a client typo does
    /// not silently register something else.
    pub kind: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub config: WatchedFolderConfigBody,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct UpdateSourceRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub include_globs: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_globs: Option<Vec<String>>,
    #[serde(default)]
    pub max_depth: Option<u16>,
    #[serde(default)]
    pub max_files_per_run: Option<u32>,
    #[serde(default)]
    pub max_file_bytes: Option<u64>,
    #[serde(default)]
    pub max_run_bytes: Option<u64>,
    #[serde(default)]
    pub rehash_every_n_runs: Option<u32>,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Default, Deserialize)]
pub struct RunSourceRequest {
    /// Ignore the size/mtime fast path and re-read every item, and give
    /// quarantined items one more chance.
    #[serde(default)]
    pub force: bool,
}

#[derive(Debug, Serialize)]
pub struct RunView {
    pub run_id: String,
    pub trigger: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub finished_at: Option<chrono::DateTime<chrono::Utc>>,
    pub status: String,
    pub items_seen: u32,
    pub items_unchanged: u32,
    pub items_ingested: u32,
    pub items_deduped: u32,
    pub items_skipped: u32,
    pub items_failed: u32,
    pub items_disappeared: u32,
    pub items_denied_by_policy: u32,
    pub memories_written: u32,
    pub bytes_read: u64,
    pub truncated_by: Option<String>,
    pub error: Option<String>,
    pub failures: Vec<FailureView>,
}

#[derive(Debug, Serialize)]
pub struct FailureView {
    pub item: String,
    pub reason: String,
    pub at: chrono::DateTime<chrono::Utc>,
    pub retryable: bool,
}

#[derive(Debug, Serialize)]
pub struct SourceView {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    /// The CANONICAL path actually walked, with the Windows `\\?\` verbatim
    /// prefix stripped for display only. The stored value keeps it. A caller
    /// sees what will be read, not what they typed.
    pub root: String,
    pub include_globs: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub max_depth: u16,
    pub max_files_per_run: u32,
    pub max_file_bytes: u64,
    pub max_run_bytes: u64,
    pub rehash_every_n_runs: u32,
    pub memory_type: String,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// `null` until the source has run once. Never a zero-filled object.
    pub last_run: Option<RunView>,
    pub last_success_at: Option<chrono::DateTime<chrono::Utc>>,
    pub consecutive_failures: u16,
    pub run_count: u64,
    pub memories_written_total: u64,
    /// Item totals as of the last completed run, so this listing does not have
    /// to walk one key per file on every dashboard poll.
    pub items_tracked: u64,
    pub items_failed: u64,
    pub items_quarantined: u64,
    /// A run holds a lease right now. Read from RocksDB, so it survives a
    /// restart honestly: after a crash the lease is swept and this is false.
    pub running: bool,
}

#[derive(Debug, Serialize)]
pub struct ListSourcesResponse {
    pub sources: Vec<SourceView>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct ListRunsResponse {
    pub runs: Vec<RunView>,
    /// True count, not the page length.
    pub total: usize,
    /// The `failures` list on each run is capped. Saying so on the wire is
    /// what stops a client rendering "3 failures" over a run that had 900.
    pub failures_are_a_sample: bool,
    pub failure_sample_size: usize,
}

#[derive(Debug, Serialize)]
pub struct ItemView {
    pub path: String,
    pub state: String,
    pub reason: Option<String>,
    pub size_bytes: u64,
    pub consecutive_failures: u16,
    pub first_ingested_at: chrono::DateTime<chrono::Utc>,
    pub last_ingested_at: chrono::DateTime<chrono::Utc>,
    pub last_seen_at: chrono::DateTime<chrono::Utc>,
    pub memory_ids: Vec<String>,
    /// Memories earlier versions of this item produced. They are still in the
    /// store and still recallable; this is the only link back to the file.
    pub superseded_memory_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ListItemsResponse {
    pub items: Vec<ItemView>,
    /// Number of items matching the filter, across all pages.
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct RunAcceptedResponse {
    pub accepted: bool,
    pub source_id: String,
    /// Where the outcome will appear. The run is durable before the first
    /// directory read, so this is answerable immediately.
    pub runs_url: String,
}

#[derive(Debug, Serialize)]
pub struct DeleteSourceResponse {
    pub deleted: bool,
    pub source_id: String,
    /// Cursors removed. The memories the source produced are **not** deleted:
    /// a memory store records what was observed, and deleting a corpus because
    /// a connector was unregistered is unrecoverable.
    pub cursors_removed: usize,
    pub memories_deleted: usize,
}

#[derive(Debug, Deserialize)]
pub struct PageQuery {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ItemQuery {
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// Strip the Windows verbatim prefix for display. The stored path keeps it —
/// that is the string that is actually walked and compared against the
/// deny-list, and rewriting it would be rewriting the security decision.
fn display_root(root: &str) -> String {
    root.strip_prefix(r"\\?\UNC\")
        .map(|rest| format!(r"\\{rest}"))
        .or_else(|| root.strip_prefix(r"\\?\").map(|r| r.to_string()))
        .unwrap_or_else(|| root.to_string())
}

fn run_view(run: &SourceRun) -> RunView {
    RunView {
        run_id: run.run_id.to_string(),
        trigger: run.trigger.as_str().to_string(),
        started_at: run.started_at,
        finished_at: run.finished_at,
        status: run.status.as_str().to_string(),
        items_seen: run.items_seen,
        items_unchanged: run.items_unchanged,
        items_ingested: run.items_ingested,
        items_deduped: run.items_deduped,
        items_skipped: run.items_skipped,
        items_failed: run.items_failed,
        items_disappeared: run.items_disappeared,
        items_denied_by_policy: run.items_denied_by_policy,
        memories_written: run.memories_written,
        bytes_read: run.bytes_read,
        truncated_by: run.truncated_by.clone(),
        error: run.error.clone(),
        failures: run
            .failures
            .iter()
            .map(|f| FailureView {
                item: f.item.clone(),
                reason: f.reason.clone(),
                at: f.at,
                retryable: f.retryable,
            })
            .collect(),
    }
}

fn item_view(cursor: &ItemCursor) -> ItemView {
    ItemView {
        path: cursor.path.clone(),
        state: cursor.state.as_str().to_string(),
        reason: cursor.state.reason().map(|r| r.to_string()),
        size_bytes: cursor.size_bytes,
        consecutive_failures: cursor.consecutive_failures,
        first_ingested_at: cursor.first_ingested_at,
        last_ingested_at: cursor.last_ingested_at,
        last_seen_at: cursor.last_seen_at,
        memory_ids: cursor.memory_ids.iter().map(|i| i.to_string()).collect(),
        superseded_memory_ids: cursor
            .superseded_memory_ids
            .iter()
            .map(|i| i.to_string())
            .collect(),
    }
}

fn source_view(state: &AppState, def: &SourceDefinition) -> Result<SourceView, AppError> {
    let runtime = state
        .source_store
        .get_runtime(&def.user_id, &def.id)
        .map_err(AppError::Internal)?;

    // The last run is read from the run history rather than reconstructed from
    // the runtime counters, so a source that has never run has no run object
    // at all instead of one full of zeros.
    let (runs, _) = state
        .source_store
        .list_runs(&def.user_id, &def.id, 1, 0)
        .map_err(AppError::Internal)?;

    let cfg = def.config.as_watched_folder();
    Ok(SourceView {
        id: def.id.0.to_string(),
        user_id: def.user_id.clone(),
        name: def.name.clone(),
        kind: def.kind.as_str().to_string(),
        enabled: def.enabled,
        root: display_root(&cfg.root),
        include_globs: cfg.include_globs.clone(),
        exclude_globs: cfg.exclude_globs.clone(),
        max_depth: cfg.max_depth,
        max_files_per_run: cfg.max_files_per_run,
        max_file_bytes: cfg.max_file_bytes,
        max_run_bytes: cfg.max_run_bytes,
        rehash_every_n_runs: cfg.rehash_every_n_runs,
        memory_type: cfg.memory_type.clone(),
        tags: cfg.tags.clone(),
        created_at: def.created_at,
        updated_at: def.updated_at,
        last_run: runs.first().map(run_view),
        last_success_at: runtime.last_success_at,
        consecutive_failures: runtime.consecutive_failures,
        run_count: runtime.run_count,
        memories_written_total: runtime.memories_written_total,
        items_tracked: runtime.items_tracked,
        items_failed: runtime.items_failed,
        items_quarantined: runtime.items_quarantined,
        running: state
            .source_store
            .is_running(&def.user_id, &def.id)
            .map_err(AppError::Internal)?,
    })
}

fn parse_source_id(raw: &str) -> Result<SourceId, AppError> {
    uuid::Uuid::parse_str(raw)
        .map(SourceId)
        .map_err(|_| AppError::SourceNotFound(raw.to_string()))
}

fn load_source(
    state: &AppState,
    user_id: &str,
    source_id: &SourceId,
) -> Result<SourceDefinition, AppError> {
    state
        .source_store
        .get_source(user_id, source_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::SourceNotFound(source_id.0.to_string()))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `POST /api/sources` — register a source.
///
/// The root is canonicalised and vetted here, once. Every later run walks the
/// stored canonical path and never re-resolves the string the caller sent.
pub async fn create_source(
    State(state): State<AppState>,
    Json(req): Json<CreateSourceRequest>,
) -> Result<impl IntoResponse, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_short_string(&req.name, "name").map_validation_err("name")?;

    let kind = SourceKind::parse(&req.kind).ok_or_else(|| AppError::InvalidInput {
        field: "kind".to_string(),
        reason: format!(
            "unknown source kind '{}'. Known kinds: {}",
            req.kind,
            SourceKind::ALL
                .iter()
                .map(|k| k.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    })?;

    let existing = state
        .source_store
        .list_sources(&req.user_id)
        .map_err(AppError::Internal)?;
    if existing.iter().any(|d| d.name == req.name) {
        return Err(AppError::InvalidInput {
            field: "name".to_string(),
            reason: format!("a source named '{}' already exists for this user", req.name),
        });
    }

    let mut cfg = build_folder_config(&req.config)?;
    let base_path = state.base_path.clone();
    let root = folder::validate_root(&req.config.root, &base_path, &existing)?;
    cfg.root = root.to_string_lossy().to_string();

    let now = chrono::Utc::now();
    let def = SourceDefinition {
        id: SourceId(uuid::Uuid::new_v4()),
        user_id: req.user_id.clone(),
        name: req.name.clone(),
        kind,
        config: SourceConfig::WatchedFolder(cfg),
        enabled: req.enabled,
        created_at: now,
        updated_at: now,
        schema_version: 1,
    };
    state
        .source_store
        .put_source(&def)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "SOURCE_CREATE",
        &def.id.0.to_string(),
        &format!("registered source '{}' at {}", def.name, display_root(&def.config.as_watched_folder().root)),
    );

    Ok((StatusCode::CREATED, Json(source_view(&state, &def)?)))
}

fn build_folder_config(
    body: &WatchedFolderConfigBody,
) -> Result<crate::memory::sources::WatchedFolderConfig, AppError> {
    let mut cfg = crate::memory::sources::WatchedFolderConfig::with_root(String::new());
    apply_folder_overrides(
        &mut cfg,
        body.include_globs.as_ref(),
        body.exclude_globs.as_ref(),
        body.max_depth,
        body.max_files_per_run,
        body.max_file_bytes,
        body.max_run_bytes,
        body.rehash_every_n_runs,
        body.memory_type.as_ref(),
        body.tags.as_ref(),
    )?;
    Ok(cfg)
}

/// Shared by create and update so a limit can never be enforced on one path
/// and not the other.
#[allow(clippy::too_many_arguments)]
fn apply_folder_overrides(
    cfg: &mut crate::memory::sources::WatchedFolderConfig,
    include_globs: Option<&Vec<String>>,
    exclude_globs: Option<&Vec<String>>,
    max_depth: Option<u16>,
    max_files_per_run: Option<u32>,
    max_file_bytes: Option<u64>,
    max_run_bytes: Option<u64>,
    rehash_every_n_runs: Option<u32>,
    memory_type: Option<&String>,
    tags: Option<&Vec<String>>,
) -> Result<(), AppError> {
    if let Some(globs) = include_globs {
        if globs.is_empty() {
            return Err(AppError::InvalidInput {
                field: "config.include_globs".to_string(),
                reason: "at least one include glob is required; an empty list would ingest \
                         nothing and read as a broken source"
                    .to_string(),
            });
        }
        validate_globs(globs, "config.include_globs")?;
        cfg.include_globs = globs.clone();
    }
    if let Some(globs) = exclude_globs {
        validate_globs(globs, "config.exclude_globs")?;
        cfg.exclude_globs = globs.clone();
    }
    if let Some(depth) = max_depth {
        if depth == 0 || depth > 64 {
            return Err(AppError::InvalidInput {
                field: "config.max_depth".to_string(),
                reason: "max_depth must be between 1 and 64".to_string(),
            });
        }
        cfg.max_depth = depth;
    }
    if let Some(files) = max_files_per_run {
        if files == 0 || files > 1_000_000 {
            return Err(AppError::InvalidInput {
                field: "config.max_files_per_run".to_string(),
                reason: "max_files_per_run must be between 1 and 1000000".to_string(),
            });
        }
        cfg.max_files_per_run = files;
    }
    if let Some(bytes) = max_file_bytes {
        if bytes == 0 {
            return Err(AppError::InvalidInput {
                field: "config.max_file_bytes".to_string(),
                reason: "max_file_bytes must be greater than 0".to_string(),
            });
        }
        cfg.max_file_bytes = bytes;
    }
    if let Some(bytes) = max_run_bytes {
        if bytes == 0 {
            return Err(AppError::InvalidInput {
                field: "config.max_run_bytes".to_string(),
                reason: "max_run_bytes must be greater than 0".to_string(),
            });
        }
        cfg.max_run_bytes = bytes;
    }
    if let Some(n) = rehash_every_n_runs {
        cfg.rehash_every_n_runs = n;
    }
    if let Some(memory_type) = memory_type {
        // Rejected here rather than at run time: a typo in the type would
        // otherwise fail every scheduled run forever with no request to blame.
        crate::handlers::remember::parse_experience_type(Some(memory_type)).map_err(|e| {
            AppError::InvalidInput {
                field: "config.memory_type".to_string(),
                reason: e.message(),
            }
        })?;
        cfg.memory_type = memory_type.clone();
    }
    if let Some(tags) = tags {
        validation::validate_tags(tags).map_validation_err("config.tags")?;
        cfg.tags = tags.clone();
    }
    Ok(())
}

fn validate_globs(patterns: &[String], field: &str) -> Result<(), AppError> {
    if patterns.len() > validation::MAX_TAGS {
        return Err(AppError::InvalidInput {
            field: field.to_string(),
            reason: format!("too many patterns: {} (max: {})", patterns.len(), validation::MAX_TAGS),
        });
    }
    for pattern in patterns {
        if pattern.len() > validation::MAX_SHORT_STRING_LENGTH {
            return Err(AppError::InvalidInput {
                field: field.to_string(),
                reason: format!("pattern too long: {} chars", pattern.len()),
            });
        }
        glob::Pattern::new(pattern).map_err(|e| AppError::InvalidInput {
            field: field.to_string(),
            reason: format!("invalid glob '{pattern}': {e}"),
        })?;
    }
    Ok(())
}

/// `GET /api/sources/{user_id}`
pub async fn list_sources(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<ListSourcesResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let defs = state
        .source_store
        .list_sources(&user_id)
        .map_err(AppError::Internal)?;
    let mut sources = Vec::with_capacity(defs.len());
    for def in &defs {
        sources.push(source_view(&state, def)?);
    }
    let total = sources.len();
    Ok(Json(ListSourcesResponse { sources, total }))
}

/// `GET /api/sources/{user_id}/{source_id}`
pub async fn get_source(
    State(state): State<AppState>,
    Path((user_id, source_id)): Path<(String, String)>,
) -> Result<Json<SourceView>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let id = parse_source_id(&source_id)?;
    let def = load_source(&state, &user_id, &id)?;
    Ok(Json(source_view(&state, &def)?))
}

/// `POST /api/sources/{user_id}/{source_id}/update`
///
/// The root is deliberately **not** updatable. Every cursor key is derived
/// from a path relative to the root, so moving the root would silently orphan
/// every cursor and re-ingest the whole corpus under new identities. Delete
/// the source and register the new root.
pub async fn update_source(
    State(state): State<AppState>,
    Path((user_id, source_id)): Path<(String, String)>,
    Json(req): Json<UpdateSourceRequest>,
) -> Result<Json<SourceView>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let id = parse_source_id(&source_id)?;
    let mut def = load_source(&state, &user_id, &id)?;

    if let Some(name) = &req.name {
        validation::validate_short_string(name, "name").map_validation_err("name")?;
        let existing = state
            .source_store
            .list_sources(&user_id)
            .map_err(AppError::Internal)?;
        if existing.iter().any(|d| d.name == *name && d.id != def.id) {
            return Err(AppError::InvalidInput {
                field: "name".to_string(),
                reason: format!("a source named '{name}' already exists for this user"),
            });
        }
        def.name = name.clone();
    }
    if let Some(enabled) = req.enabled {
        def.enabled = enabled;
    }

    let SourceConfig::WatchedFolder(mut cfg) = def.config.clone();
    apply_folder_overrides(
        &mut cfg,
        req.include_globs.as_ref(),
        req.exclude_globs.as_ref(),
        req.max_depth,
        req.max_files_per_run,
        req.max_file_bytes,
        req.max_run_bytes,
        req.rehash_every_n_runs,
        req.memory_type.as_ref(),
        req.tags.as_ref(),
    )?;
    def.config = SourceConfig::WatchedFolder(cfg);
    def.updated_at = chrono::Utc::now();

    state
        .source_store
        .put_source(&def)
        .map_err(AppError::Internal)?;
    Ok(Json(source_view(&state, &def)?))
}

/// `DELETE /api/sources/{user_id}/{source_id}`
pub async fn delete_source(
    State(state): State<AppState>,
    Path((user_id, source_id)): Path<(String, String)>,
) -> Result<Json<DeleteSourceResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let id = parse_source_id(&source_id)?;
    load_source(&state, &user_id, &id)?;

    let cursors_removed = state
        .source_store
        .delete_source(&user_id, &id)
        .map_err(AppError::Internal)?;
    state.source_locks.remove(&id);

    state.log_event(
        &user_id,
        "SOURCE_DELETE",
        &id.0.to_string(),
        &format!("unregistered source, {cursors_removed} cursors removed, memories retained"),
    );

    Ok(Json(DeleteSourceResponse {
        deleted: true,
        source_id: id.0.to_string(),
        cursors_removed,
        // Stated explicitly rather than omitted, so no client has to guess
        // whether unregistering a source destroyed a corpus.
        memories_deleted: 0,
    }))
}

/// `POST /api/sources/{user_id}/{source_id}/run` — ingest now.
///
/// Returns 202 and runs on `task_tracker`, because a first run over a large
/// folder outlives any sensible HTTP timeout. The run record and its lease are
/// durable before the first directory read, so the outcome is observable from
/// the moment this returns.
pub async fn trigger_run(
    State(state): State<AppState>,
    Path((user_id, source_id)): Path<(String, String)>,
    body: Option<Json<RunSourceRequest>>,
) -> Result<impl IntoResponse, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let id = parse_source_id(&source_id)?;
    let def = load_source(&state, &user_id, &id)?;

    if !def.enabled {
        return Err(AppError::InvalidInput {
            field: "enabled".to_string(),
            reason: "source is disabled; enable it before running".to_string(),
        });
    }

    // A lease left behind by a process that died is swept at startup, so a
    // lease seen here belongs to a run that really is in flight.
    if state
        .source_store
        .is_running(&user_id, &id)
        .map_err(AppError::Internal)?
    {
        return Err(AppError::SourceRunInProgress(id.0.to_string()));
    }

    let lock = state
        .source_locks
        .entry(id.clone())
        .or_insert_with(|| std::sync::Arc::new(tokio::sync::Mutex::new(())))
        .clone();
    let guard = lock
        .try_lock_owned()
        .map_err(|_| AppError::SourceRunInProgress(id.0.to_string()))?;

    let force = body.map(|Json(b)| b.force).unwrap_or(false);
    let runner_state = state.clone();
    let runner_def = def.clone();
    state.task_tracker.spawn(async move {
        // The guard rides with the run, not with the request, so the lock is
        // held for exactly as long as a run is in flight.
        let _guard = guard;
        match crate::ingest::folder::execute_run(
            &runner_state,
            &runner_def,
            crate::memory::sources::RunTrigger::Manual,
            force,
        )
        .await
        {
            Ok(run) => tracing::info!(
                source = %runner_def.id,
                run = %run.run_id,
                status = run.status.as_str(),
                ingested = run.items_ingested,
                "Source run finished"
            ),
            Err(e) => tracing::error!(
                source = %runner_def.id,
                error = %e.message(),
                "Source run could not be recorded"
            ),
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(RunAcceptedResponse {
            accepted: true,
            source_id: id.0.to_string(),
            runs_url: format!("/api/sources/{user_id}/{}/runs", id.0),
        }),
    ))
}

/// `GET /api/sources/{user_id}/{source_id}/runs`
pub async fn list_runs(
    State(state): State<AppState>,
    Path((user_id, source_id)): Path<(String, String)>,
    Query(page): Query<PageQuery>,
) -> Result<Json<ListRunsResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let id = parse_source_id(&source_id)?;
    load_source(&state, &user_id, &id)?;

    let limit = page.limit.unwrap_or(DEFAULT_RUN_LIMIT);
    validation::validate_limit(limit, "limit").map_validation_err("limit")?;
    let offset = page.offset.unwrap_or(0);

    let (runs, total) = state
        .source_store
        .list_runs(&user_id, &id, limit, offset)
        .map_err(AppError::Internal)?;

    Ok(Json(ListRunsResponse {
        runs: runs.iter().map(run_view).collect(),
        total,
        failures_are_a_sample: true,
        failure_sample_size: RUN_FAILURE_SAMPLE,
    }))
}

/// `GET /api/sources/{user_id}/{source_id}/items`
///
/// This is the dead-letter queue. There is no second store: an item's cursor
/// carries its failure reason and its consecutive-failure count, and filtering
/// on `state=failed|quarantined|skipped` is how a person finds what did not
/// land.
pub async fn list_items(
    State(state): State<AppState>,
    Path((user_id, source_id)): Path<(String, String)>,
    Query(query): Query<ItemQuery>,
) -> Result<Json<ListItemsResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;
    let id = parse_source_id(&source_id)?;
    load_source(&state, &user_id, &id)?;

    let limit = query.limit.unwrap_or(DEFAULT_ITEM_LIMIT);
    validation::validate_limit(limit, "limit").map_validation_err("limit")?;
    let offset = query.offset.unwrap_or(0);

    if let Some(filter) = &query.state {
        const KNOWN: &[&str] = &["ingested", "deduped", "skipped", "failed", "quarantined"];
        if !KNOWN.contains(&filter.to_ascii_lowercase().as_str()) {
            return Err(AppError::InvalidInput {
                field: "state".to_string(),
                reason: format!("unknown state '{filter}'. Known states: {}", KNOWN.join(", ")),
            });
        }
    }

    let cursors = state
        .source_store
        .list_cursors(&id)
        .map_err(AppError::Internal)?;

    let filtered: Vec<&ItemCursor> = match &query.state {
        None => cursors.iter().collect(),
        Some(filter) => {
            let filter = filter.to_ascii_lowercase();
            cursors
                .iter()
                .filter(|c| c.state.as_str() == filter)
                .collect()
        }
    };
    let total = filtered.len();
    let items = filtered
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(item_view)
        .collect();

    Ok(Json(ListItemsResponse { items, total }))
}
