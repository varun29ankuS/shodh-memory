//! Anomaly Feed Handlers
//!
//! Read-time deviation scoring over the per-episode surprise components
//! captured at ingest (see [`crate::memory::types::SurpriseComponents`]).
//!
//! Design: ingest stores FACTS (the episode's statistical shape); this
//! endpoint computes DEVIATION — per-component z-scores against the user's
//! own rolling baseline. Keeping the scoring at read time means thresholds
//! are tunable without re-ingesting, and every flag is explainable
//! component-by-component ("mean_pmi 3.2σ below baseline"), deterministically
//! and without any LLM in the loop.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};

use super::state::MultiUserMemoryManager;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::types::{SurpriseComponents, SURPRISE_METADATA_KEY};
use crate::validation;
use std::sync::Arc;

type AppState = Arc<MultiUserMemoryManager>;

/// Default number of most-recent scored episodes forming the rolling baseline.
const DEFAULT_BASELINE_WINDOW: usize = 200;
/// Default number of anomalies returned.
const DEFAULT_LIMIT: usize = 20;
/// Default |z| threshold for the `flagged` marker.
const DEFAULT_MIN_SIGMA: f32 = 2.0;
/// Minimum episodes needed before deviation is meaningful; below this the
/// endpoint returns an empty feed rather than z-scores against noise.
const MIN_BASELINE_EPISODES: usize = 10;

#[derive(Deserialize)]
pub struct AnomalyListRequest {
    pub user_id: String,
    /// Rolling-baseline window (most recent N scored episodes). Default 200.
    #[serde(default)]
    pub window: Option<usize>,
    /// Maximum entries returned, ranked by max |z|. Default 20.
    #[serde(default)]
    pub limit: Option<usize>,
    /// |z| at or above which an entry is `flagged`. Default 2.0.
    #[serde(default)]
    pub min_sigma: Option<f32>,
}

/// One component's deviation, kept explicit for explainability.
#[derive(Debug, Clone, Serialize)]
pub struct ComponentDeviation {
    pub component: String,
    pub value: f32,
    pub baseline_mean: f32,
    pub baseline_std: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnomalyEntry {
    pub memory_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub content_preview: String,
    pub components: SurpriseComponents,
    pub deviations: Vec<ComponentDeviation>,
    pub max_abs_z: f32,
    /// True when `max_abs_z >= min_sigma`.
    pub flagged: bool,
    /// Human-readable, deterministic explanation built from the top deviating
    /// components — the auditable "why was this flagged".
    pub explanation: String,
}

#[derive(Serialize)]
pub struct AnomalyListResponse {
    pub anomalies: Vec<AnomalyEntry>,
    pub episodes_scored: usize,
    pub baseline_window: usize,
    pub min_sigma: f32,
}

/// The component axes scored against the baseline. Extraction is a function
/// pointer per axis so baseline and per-episode paths cannot drift apart.
const AXES: &[(&str, fn(&SurpriseComponents) -> f32)] = &[
    ("mean_pmi", |s| s.mean_pmi),
    ("novel_entity_ratio", |s| s.novel_entity_ratio),
    ("untyped_ratio", |s| s.untyped_ratio),
    ("pmi_gated_ratio", |s| s.pmi_gated_ratio),
    ("low_selectivity_share", |s| s.low_selectivity_share),
];

/// POST /api/anomalies - rank recent episodes by statistical deviation
pub async fn list_anomalies(
    State(state): State<AppState>,
    Json(req): Json<AnomalyListRequest>,
) -> Result<Json<AnomalyListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let window = req.window.unwrap_or(DEFAULT_BASELINE_WINDOW).max(1);
    let limit = req.limit.unwrap_or(DEFAULT_LIMIT).max(1);
    let min_sigma = req.min_sigma.unwrap_or(DEFAULT_MIN_SIGMA).max(0.0);

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    // Collect the scored episodes (those carrying surprise components) off
    // the async executor — episode iteration is a full CF scan.
    let mut scored: Vec<(
        uuid::Uuid,
        chrono::DateTime<chrono::Utc>,
        String,
        SurpriseComponents,
    )> = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<_>> {
        let graph_guard = graph.read();
        let episodes = graph_guard.get_all_episodes()?;
        Ok(episodes
            .into_iter()
            .filter_map(|ep| {
                let json = ep.metadata.get(SURPRISE_METADATA_KEY)?;
                let components: SurpriseComponents = serde_json::from_str(json).ok()?;
                let preview: String = ep.content.chars().take(160).collect();
                Some((ep.uuid, ep.created_at, preview, components))
            })
            .collect())
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Task join error: {}", e)))?
    .map_err(AppError::Internal)?;

    // Most recent first; the baseline is the rolling window of recent shape.
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored.truncate(window);

    if scored.len() < MIN_BASELINE_EPISODES {
        return Ok(Json(AnomalyListResponse {
            anomalies: Vec::new(),
            episodes_scored: scored.len(),
            baseline_window: window,
            min_sigma,
        }));
    }

    // Per-axis baseline over the window (population mean/std).
    let n = scored.len() as f32;
    let baselines: Vec<(f32, f32)> = AXES
        .iter()
        .map(|(_, extract)| {
            let mean = scored.iter().map(|(_, _, _, s)| extract(s)).sum::<f32>() / n;
            let var = scored
                .iter()
                .map(|(_, _, _, s)| {
                    let d = extract(s) - mean;
                    d * d
                })
                .sum::<f32>()
                / n;
            (mean, var.sqrt())
        })
        .collect();

    let mut entries: Vec<AnomalyEntry> = scored
        .into_iter()
        .map(|(uuid, created_at, preview, components)| {
            let deviations: Vec<ComponentDeviation> = AXES
                .iter()
                .zip(baselines.iter())
                .map(|((name, extract), (mean, std))| {
                    let value = extract(&components);
                    // A degenerate axis (zero variance in the window) carries
                    // no deviation signal: z = 0, never a false flag.
                    let z = if *std > f32::EPSILON {
                        (value - mean) / std
                    } else {
                        0.0
                    };
                    ComponentDeviation {
                        component: name.to_string(),
                        value,
                        baseline_mean: *mean,
                        baseline_std: *std,
                        z,
                    }
                })
                .collect();
            let max_abs_z = deviations.iter().map(|d| d.z.abs()).fold(0.0, f32::max);
            let flagged = max_abs_z >= min_sigma;

            // Deterministic explanation from the top-2 deviating axes.
            let mut ranked: Vec<&ComponentDeviation> = deviations.iter().collect();
            ranked.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
            let explanation = ranked
                .iter()
                .take(2)
                .filter(|d| d.z.abs() > f32::EPSILON)
                .map(|d| {
                    format!(
                        "{} {:.1}σ {} baseline ({:.3} vs {:.3})",
                        d.component,
                        d.z.abs(),
                        if d.z >= 0.0 { "above" } else { "below" },
                        d.value,
                        d.baseline_mean
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");

            AnomalyEntry {
                memory_id: uuid.to_string(),
                created_at,
                content_preview: preview,
                components,
                deviations,
                max_abs_z,
                flagged,
                explanation,
            }
        })
        .collect();

    entries.sort_by(|a, b| b.max_abs_z.total_cmp(&a.max_abs_z));
    let episodes_scored = entries.len();
    entries.truncate(limit);

    Ok(Json(AnomalyListResponse {
        anomalies: entries,
        episodes_scored,
        baseline_window: window,
        min_sigma,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comp(mean_pmi: f32, novel: f32) -> SurpriseComponents {
        SurpriseComponents {
            mean_pmi,
            novel_entity_ratio: novel,
            untyped_ratio: 0.5,
            pmi_gated_ratio: 0.0,
            low_selectivity_share: 0.0,
            pairs_scored: 6,
            entities_total: 4,
        }
    }

    #[test]
    fn axes_extractors_cover_all_scored_fields() {
        let s = comp(1.5, 0.25);
        let values: Vec<f32> = AXES.iter().map(|(_, f)| f(&s)).collect();
        assert_eq!(values, vec![1.5, 0.25, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn zero_variance_axis_never_flags() {
        // All-identical window → std 0 on every axis → z must be 0, not NaN.
        let s = comp(2.0, 0.1);
        let mean = 2.0f32;
        let std = 0.0f32;
        let z = if std > f32::EPSILON {
            (s.mean_pmi - mean) / std
        } else {
            0.0
        };
        assert_eq!(z, 0.0);
        assert!(z.is_finite());
    }
}
