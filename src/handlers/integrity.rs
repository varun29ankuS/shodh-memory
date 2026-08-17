//! HTTP surface for the read-only integrity scrub ([`crate::integrity`]).
//!
//! Deliberately complementary to the existing diagnostics rather than
//! overlapping them:
//!
//! - `POST /api/index/verify` compares storage membership against the vector
//!   index. It loads records through `get_all()`, which silently drops anything
//!   that fails to decode — so the population this scrub exists to find is
//!   invisible to it by construction.
//! - `GET /api/audit/{user_id}` and `GET /api/memory/{id}/history` answer
//!   "who changed this record". This answers "is what we stored still what we
//!   wrote", which no audit trail can tell you: the July breakage changed no
//!   record at all, it changed the schema that reads them.

use axum::extract::State;
use axum::response::Json;
use serde::Deserialize;

use crate::errors::{AppError, ValidationErrorExt};
use crate::handlers::state::MultiUserMemoryManager;
use crate::integrity::{self, IntegrityScrubReport, ScrubBudget};
use crate::validation;

pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

/// Upper bound on a caller-supplied time budget.
///
/// Above the server's own request timeout the budget would never bind, and the
/// scrub would be killed mid-sweep and return nothing at all instead of an
/// honest partial report.
const MAX_DURATION_MS: u64 = 120_000;

#[derive(Debug, Deserialize)]
pub struct ScrubRequest {
    pub user_id: String,
    /// Stop after this many records. Omit for a full sweep, which is the
    /// intended mode — see [`crate::integrity::scrub_user`] on why sampling is
    /// the wrong instrument here. A capped run reports `complete: false` and
    /// can never return a `healthy` verdict.
    pub max_records: Option<u64>,
    /// Wall-clock ceiling in milliseconds. Defaults to
    /// [`crate::integrity::DEFAULT_MAX_DURATION_MS`].
    pub max_duration_ms: Option<u64>,
}

/// POST /api/integrity/scrub - read-only wire-level integrity scrub over one
/// user's memories and graph nodes.
///
/// Classifies every record as clean, readable-only-via-a-legacy-generation,
/// undecodable, or decoding-into-implausible-values, and returns a verdict with
/// the numeric rule that produced it.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn scrub(
    State(state): State<AppState>,
    Json(req): Json<ScrubRequest>,
) -> Result<Json<IntegrityScrubReport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    if let Some(max) = req.max_records {
        validation::validate_limit(max as usize, "max_records").map_validation_err("max_records")?;
    }
    if let Some(ms) = req.max_duration_ms {
        if ms == 0 || ms > MAX_DURATION_MS {
            return Err(AppError::InvalidInput {
                field: "max_duration_ms".to_string(),
                reason: format!("must be in 1..={MAX_DURATION_MS}"),
            });
        }
    }

    let budget = ScrubBudget {
        max_records: req.max_records,
        max_duration: Some(std::time::Duration::from_millis(
            req.max_duration_ms
                .unwrap_or(integrity::DEFAULT_MAX_DURATION_MS),
        )),
    };

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    // A user with no graph yet is not an error: the scrub reports the graph
    // column family as skipped, which forbids a `healthy` verdict rather than
    // quietly reporting health for half the data.
    let graph = state.get_user_graph(&req.user_id).ok();

    let user_id = req.user_id.clone();
    // A full sweep is tens to hundreds of milliseconds of synchronous RocksDB
    // iteration and postcard decoding. `crud::list_memories` is the precedent:
    // full scans go on the blocking pool, never on a runtime worker.
    let report = tokio::task::spawn_blocking(move || {
        let memory_guard = memory_sys.read();
        let memory_db = memory_guard.storage().raw_db();

        match graph {
            Some(g) => {
                let graph_guard = g.read();
                let gdb = graph_guard.get_db();
                match gdb.cf_handle(crate::graph_memory::ENTITIES_CF_NAME) {
                    Some(cf) => {
                        integrity::scrub_user(&user_id, memory_db, Some((gdb, cf)), budget)
                    }
                    None => integrity::scrub_user(&user_id, memory_db, None, budget),
                }
            }
            None => integrity::scrub_user(&user_id, memory_db, None, budget),
        }
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("integrity scrub task panicked: {e}")))?;

    Ok(Json(report))
}
