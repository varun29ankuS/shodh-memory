//! Remember Handlers - Memory Storage Operations
//!
//! Core handlers for storing memories: remember, batch_remember, upsert.

use std::collections::HashSet;

use axum::{extract::State, response::Json};

use super::health::AppState;
use super::types::MemoryEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{
    storage::Modality,
    types::{
        ChangeType, ContextId, EmotionalContext, EpisodeContext, MediaRef, NerEntityRecord,
        RichContext, SourceContext, SourceType,
    },
    Experience, ExperienceType, SessionEvent,
};
use crate::metrics;
use crate::validation;

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// Remember request - store a new memory
#[derive(Debug, serde::Deserialize)]
pub struct RememberRequest {
    pub user_id: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, alias = "type", alias = "experience_type")]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub external_id: Option<String>,
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub emotional_valence: Option<f32>,
    #[serde(default)]
    pub emotional_arousal: Option<f32>,
    #[serde(default)]
    pub emotion: Option<String>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub credibility: Option<f32>,
    #[serde(default)]
    pub episode_id: Option<String>,
    #[serde(default)]
    pub sequence_number: Option<u32>,
    #[serde(default)]
    pub preceding_memory_id: Option<String>,
    // Caller-declared write identity. All three land on `Memory` and are
    // stored verbatim; the server does not verify them.
    //
    // `parent_agent_id` used to sit here between `agent_id` and `run_id`. It
    // was accepted by serde, referenced nowhere else in the repo, and had no
    // destination field on `Memory` — the same accepted-and-silently-dropped
    // shape as the `user_id` that `/api/sessions/stats` ignored. It is removed
    // rather than wired: giving it a home means a new persisted field, and
    // `MemoryFlat` is encoded positionally, so that is a wire-format change
    // that has to be justified by a consumer. There is none. Removing it is
    // transparent to clients — `RememberRequest` does not deny unknown fields,
    // so a request still carrying it is accepted exactly as before.
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub actor_id: Option<String>,
    /// Parent memory ID for hierarchical organization
    /// Use this to create memory trees (e.g., "71-research" -> "algebraic" -> "21×27≡-1")
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Optional importance override (0.0-1.0). When provided, bypasses auto-calculation.
    /// Use for hook-generated memories where importance is known from context:
    /// Decision=0.8, Learning=0.7, Error=0.7, Discovery=0.6, Observation=0.3
    #[serde(default)]
    pub importance: Option<f32>,

    // === Robotics Context ===
    /// Robot/drone identifier for multi-robot systems
    #[serde(default)]
    pub robot_id: Option<String>,
    /// Mission identifier for grouping experiences
    #[serde(default)]
    pub mission_id: Option<String>,
    /// GPS coordinates [latitude, longitude, altitude] in WGS84
    #[serde(default)]
    pub geo_location: Option<[f64; 3]>,
    /// Local position [x, y, z] in meters (robot-local frame)
    #[serde(default)]
    pub local_position: Option<[f32; 3]>,
    /// Heading in degrees (0-360)
    #[serde(default)]
    pub heading: Option<f32>,
    /// Action type name (e.g., "navigate", "grasp", "dock")
    #[serde(default)]
    pub action_type: Option<String>,
    /// Action parameters (e.g., {"speed": "0.5", "target": "shelf_3"})
    #[serde(default)]
    pub action_params: Option<std::collections::HashMap<String, String>>,
    /// Reinforcement learning reward signal (-1.0 to 1.0)
    #[serde(default)]
    pub reward: Option<f32>,
    /// Raw sensor readings (e.g., {"battery": 72.5, "temperature": 23.1})
    #[serde(default)]
    pub sensor_data: Option<std::collections::HashMap<String, f64>>,
    /// Outcome type: success, failure, partial, aborted, timeout
    #[serde(default)]
    pub outcome_type: Option<String>,
    /// Detailed outcome description
    #[serde(default)]
    pub outcome_details: Option<String>,
    /// Terrain type: indoor, outdoor, urban, rural, water, aerial
    #[serde(default)]
    pub terrain_type: Option<String>,
    /// Whether this is a failure/error experience
    #[serde(default)]
    pub is_failure: Option<bool>,
    /// Whether this is an anomaly/unexpected event
    #[serde(default)]
    pub is_anomaly: Option<bool>,
    /// Severity level: info, warning, error, critical
    #[serde(default)]
    pub severity: Option<String>,
    /// When true, require robot_id and geo_location for strict robotics mode
    #[serde(default)]
    pub validate_robotics: Option<bool>,
    /// Arbitrary key-value metadata (e.g., session_id, started_at, duration_secs).
    /// Stored directly on the Experience and queryable via session_history.
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,

    // === Multimodal Embeddings (pre-computed) ===
    /// Pre-computed image embeddings (1024-dim, ImageBind/CLIP).
    /// Pass embeddings generated by your own vision encoder.
    #[serde(default)]
    pub image_embeddings: Option<Vec<f32>>,
    /// Pre-computed audio embeddings (1024-dim, ImageBind/wav2vec).
    #[serde(default)]
    pub audio_embeddings: Option<Vec<f32>>,
    /// Pre-computed video embeddings (1024-dim, ImageBind).
    #[serde(default)]
    pub video_embeddings: Option<Vec<f32>>,
    /// References to attached media files (URIs, not raw bytes).
    #[serde(default)]
    pub media_refs: Vec<MediaRef>,
}

/// Remember response
#[derive(Debug, serde::Serialize)]
pub struct RememberResponse {
    pub id: String,
    pub success: bool,
}

/// Batch remember request
#[derive(Debug, serde::Deserialize)]
pub struct BatchRememberRequest {
    pub user_id: String,
    pub memories: Vec<BatchMemoryItem>,
    #[serde(default)]
    pub options: BatchRememberOptions,
}

/// Options for batch remember
#[derive(Debug, serde::Deserialize, Clone)]
pub struct BatchRememberOptions {
    #[serde(default = "default_true")]
    pub extract_entities: bool,
    #[serde(default = "default_true")]
    pub create_edges: bool,
}

impl Default for BatchRememberOptions {
    fn default() -> Self {
        Self {
            extract_entities: true,
            create_edges: true,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Single item in batch remember
#[derive(Debug, serde::Deserialize, Clone)]
pub struct BatchMemoryItem {
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, alias = "type", alias = "experience_type")]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub emotional_valence: Option<f32>,
    #[serde(default)]
    pub emotional_arousal: Option<f32>,
    #[serde(default)]
    pub emotion: Option<String>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub credibility: Option<f32>,
    #[serde(default)]
    pub episode_id: Option<String>,
    #[serde(default)]
    pub sequence_number: Option<u32>,
    #[serde(default)]
    pub preceding_memory_id: Option<String>,
    /// Parent memory ID for hierarchical organization
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Optional importance override (0.0-1.0)
    #[serde(default)]
    pub importance: Option<f32>,
}

/// Error detail for batch item
#[derive(Debug, serde::Serialize)]
pub struct BatchErrorItem {
    pub index: usize,
    pub error: String,
}

/// Batch remember response
#[derive(Debug, serde::Serialize)]
pub struct BatchRememberResponse {
    pub created: usize,
    pub failed: usize,
    pub memory_ids: Vec<String>,
    pub errors: Vec<BatchErrorItem>,
}

/// Upsert request - create or update memory
#[derive(Debug, serde::Deserialize)]
pub struct UpsertRequest {
    pub user_id: String,
    pub external_id: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, alias = "type", alias = "experience_type")]
    pub memory_type: Option<String>,
    #[serde(default = "default_change_type")]
    pub change_type: String,
    #[serde(default)]
    pub changed_by: Option<String>,
    #[serde(default)]
    pub change_reason: Option<String>,
    /// Optional importance override (0.0-1.0). When provided, bypasses auto-calculation.
    #[serde(default)]
    pub importance: Option<f32>,
}

fn default_change_type() -> String {
    "content_updated".to_string()
}

/// Upsert response
#[derive(Debug, serde::Serialize)]
pub struct UpsertResponse {
    pub id: String,
    pub success: bool,
    pub was_update: bool,
    pub version: u32,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Parse memory type from string.
///
/// Returns `Ok(Observation)` when no type is provided (default).
/// Returns `Err` when an explicit type string doesn't match any known type,
/// preventing silent data corruption from typos.
pub fn parse_experience_type(
    s: Option<&String>,
) -> Result<ExperienceType, crate::errors::AppError> {
    match s {
        None => Ok(ExperienceType::Observation),
        Some(s) => match s.to_lowercase().as_str() {
            "observation" => Ok(ExperienceType::Observation),
            "decision" => Ok(ExperienceType::Decision),
            "learning" => Ok(ExperienceType::Learning),
            "error" => Ok(ExperienceType::Error),
            "discovery" => Ok(ExperienceType::Discovery),
            "pattern" => Ok(ExperienceType::Pattern),
            "context" => Ok(ExperienceType::Context),
            "task" => Ok(ExperienceType::Task),
            "codeedit" | "code_edit" => Ok(ExperienceType::CodeEdit),
            "fileaccess" | "file_access" => Ok(ExperienceType::FileAccess),
            "search" => Ok(ExperienceType::Search),
            "command" => Ok(ExperienceType::Command),
            "conversation" => Ok(ExperienceType::Conversation),
            "intention" => Ok(ExperienceType::Intention),
            unknown => Err(crate::errors::AppError::InvalidInput {
                field: "type".to_string(),
                reason: format!(
                    "Unknown memory type '{}'. Valid types: Observation, Decision, Learning, Error, \
                     Discovery, Pattern, Context, Task, CodeEdit, FileAccess, Search, Command, \
                     Conversation, Intention",
                    unknown
                ),
            }),
        },
    }
}

/// Parse source type from string
pub fn parse_source_type(s: Option<&String>) -> SourceType {
    s.map(|s| match s.to_lowercase().as_str() {
        "user" => SourceType::User,
        "system" => SourceType::System,
        "api" | "external_api" => SourceType::ExternalApi,
        "file" => SourceType::File,
        "web" => SourceType::Web,
        "ai_generated" | "ai" => SourceType::AiGenerated,
        "inferred" => SourceType::Inferred,
        _ => SourceType::Unknown,
    })
    .unwrap_or(SourceType::User)
}

/// Build RichContext from request fields
#[allow(clippy::too_many_arguments)]
pub fn build_rich_context(
    emotional_valence: Option<f32>,
    emotional_arousal: Option<f32>,
    emotion: Option<String>,
    source_type: Option<String>,
    credibility: Option<f32>,
    episode_id: Option<String>,
    sequence_number: Option<u32>,
    preceding_memory_id: Option<String>,
) -> Option<RichContext> {
    let has_context = emotional_valence.is_some()
        || emotional_arousal.is_some()
        || emotion.is_some()
        || source_type.is_some()
        || credibility.is_some()
        || episode_id.is_some()
        || sequence_number.is_some()
        || preceding_memory_id.is_some();

    if !has_context {
        return None;
    }

    let emotional = EmotionalContext {
        valence: emotional_valence.unwrap_or(0.0),
        arousal: emotional_arousal.unwrap_or(0.0),
        dominant_emotion: emotion,
        confidence: if emotional_valence.is_some() || emotional_arousal.is_some() {
            0.8
        } else {
            0.0
        },
        ..Default::default()
    };

    let source = SourceContext {
        source_type: parse_source_type(source_type.as_ref()),
        credibility: credibility.unwrap_or(0.8),
        ..Default::default()
    };

    let episode = EpisodeContext {
        episode_id,
        sequence_number,
        preceding_memory_id,
        ..Default::default()
    };

    let now = chrono::Utc::now();
    Some(RichContext {
        id: ContextId(uuid::Uuid::new_v4()),
        emotional,
        source,
        episode,
        conversation: Default::default(),
        user: Default::default(),
        project: Default::default(),
        temporal: Default::default(),
        semantic: Default::default(),
        code: Default::default(),
        document: Default::default(),
        environment: Default::default(),
        parent: None,
        embeddings: None,
        decay_rate: 1.0,
        created_at: now,
        updated_at: now,
    })
}

// =============================================================================
// HANDLERS
// =============================================================================

/// Remember a single memory
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn remember(
    State(state): State<AppState>,
    trace: Option<axum::Extension<crate::handlers::trace::OpTrace>>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    let experience_type = parse_experience_type(req.memory_type.as_ref())?;

    // PERF: Run NER and YAKE extraction in parallel using spawn_blocking
    // Both are CPU-bound and independent - parallelization reduces latency by ~40%
    let ner = state.get_neural_ner();
    let yake = state.get_keyword_extractor();
    let content_for_ner = req.content.clone();
    let content_for_yake = req.content.clone();

    let (ner_result, yake_result) = tokio::join!(
        // NER extraction (named entities: Person, Org, Location, Misc)
        // Preserve full entity records for downstream graph insertion with proper labels
        tokio::task::spawn_blocking(move || {
            match ner.extract(&content_for_ner) {
                Ok(entities) => entities
                    .into_iter()
                    .map(|e| NerEntityRecord {
                        text: e.text,
                        entity_type: e.entity_type.as_str().to_string(),
                        confidence: e.confidence,
                        start_char: Some(e.start),
                        end_char: Some(e.end),
                        fine_label: e.fine_label,
                    })
                    .collect::<Vec<NerEntityRecord>>(),
                // Reaching here means BOTH the neural typer and the rule-based
                // fallback failed — the memory is about to be stored with no
                // typed entities at all, which starves graph labelling and the
                // toponym gazetteer. That is not a debug-level event.
                Err(e) => {
                    tracing::warn!(
                        "NER extraction failed on remember — storing memory with NO typed \
                         entities: {e}"
                    );
                    Vec::new()
                }
            }
        }),
        // YAKE extraction (keywords: common nouns, verbs, etc.)
        // Captures important terms like "sunrise", "painting", "lake"
        tokio::task::spawn_blocking(move || yake.extract_texts(&content_for_yake))
    );

    let ner_entities = match ner_result {
        Ok(entities) => entities,
        Err(e) => {
            if e.is_panic() {
                tracing::error!("NER extraction task panicked: {:?}", e);
            } else {
                tracing::debug!("NER extraction task cancelled: {:?}", e);
            }
            Vec::new()
        }
    };
    let extracted_keywords = match yake_result {
        Ok(keywords) => keywords,
        Err(e) => {
            if e.is_panic() {
                tracing::error!("YAKE extraction task panicked: {:?}", e);
            } else {
                tracing::debug!("YAKE extraction task cancelled: {:?}", e);
            }
            Vec::new()
        }
    };

    let mut merged_entities: Vec<String> = req.tags.clone();
    let mut seen: HashSet<String> = merged_entities.iter().map(|t| t.to_lowercase()).collect();
    for record in &ner_entities {
        if seen.insert(record.text.to_lowercase()) {
            if validation::validate_entity(&record.text).is_ok() {
                merged_entities.push(record.text.clone());
            } else {
                tracing::debug!(entity = %record.text, "Skipping invalid NER entity (too long or invalid chars)");
            }
        }
    }
    for keyword in extracted_keywords {
        if seen.insert(keyword.to_lowercase()) {
            if validation::validate_entity(&keyword).is_ok() {
                merged_entities.push(keyword);
            } else {
                tracing::debug!(entity = %keyword, "Skipping invalid YAKE keyword (too long or invalid chars)");
            }
        }
    }
    if merged_entities.len() > validation::MAX_ENTITIES_PER_MEMORY {
        tracing::debug!(
            count = merged_entities.len(),
            max = validation::MAX_ENTITIES_PER_MEMORY,
            "Capping entities to maximum allowed"
        );
        merged_entities.truncate(validation::MAX_ENTITIES_PER_MEMORY);
    }

    let experience_type_str = format!("{:?}", experience_type);

    let context = build_rich_context(
        req.emotional_valence,
        req.emotional_arousal,
        req.emotion.clone(),
        req.source_type.clone(),
        req.credibility,
        req.episode_id.clone(),
        req.sequence_number,
        req.preceding_memory_id.clone(),
    );

    // Validate numeric range fields
    if let Some(importance) = req.importance {
        validation::validate_unit_float(importance, "importance")
            .map_validation_err("importance")?;
    }
    if let Some(credibility) = req.credibility {
        validation::validate_unit_float(credibility, "credibility")
            .map_validation_err("credibility")?;
    }
    if let Some(valence) = req.emotional_valence {
        validation::validate_bipolar_float(valence, "emotional_valence")
            .map_validation_err("emotional_valence")?;
    }
    if let Some(arousal) = req.emotional_arousal {
        validation::validate_unit_float(arousal, "emotional_arousal")
            .map_validation_err("emotional_arousal")?;
    }

    // Validate tags
    if !req.tags.is_empty() {
        validation::validate_tags(&req.tags).map_validation_err("tags")?;
    }

    // Validate robotics fields
    if let Some(ref geo) = req.geo_location {
        validation::validate_geo_location(geo).map_validation_err("geo_location")?;
    }
    if let Some(reward) = req.reward {
        validation::validate_reward(reward).map_validation_err("reward")?;
    }
    if let Some(heading) = req.heading {
        validation::validate_heading(heading).map_validation_err("heading")?;
    }
    if let Some(ref sensor_data) = req.sensor_data {
        validation::validate_sensor_data(sensor_data).map_validation_err("sensor_data")?;
    }
    if let Some(ref local_pos) = req.local_position {
        let pos_f64 = [
            local_pos[0] as f64,
            local_pos[1] as f64,
            local_pos[2] as f64,
        ];
        validation::validate_local_position(&pos_f64).map_validation_err("local_position")?;
    }
    if let Some(ref robot_id) = req.robot_id {
        validation::validate_short_string(robot_id, "robot_id").map_validation_err("robot_id")?;
    }
    if let Some(ref mission_id) = req.mission_id {
        validation::validate_short_string(mission_id, "mission_id")
            .map_validation_err("mission_id")?;
    }
    if let Some(ref action_type) = req.action_type {
        validation::validate_short_string(action_type, "action_type")
            .map_validation_err("action_type")?;
    }
    if let Some(ref terrain_type) = req.terrain_type {
        validation::validate_short_string(terrain_type, "terrain_type")
            .map_validation_err("terrain_type")?;
    }

    // Warn on unknown outcome_type/severity (log, don't reject)
    let mut warnings = Vec::new();
    if let Some(ref outcome_type) = req.outcome_type {
        if let Some(warn) = validation::warn_outcome_type(outcome_type) {
            warnings.push(warn);
        }
    }
    if let Some(ref severity) = req.severity {
        if let Some(warn) = validation::warn_severity(severity) {
            warnings.push(warn);
        }
    }
    for warn in &warnings {
        tracing::warn!("remember validation warning: {}", warn);
    }

    // Validate multimodal embedding dimensions
    if let Some(ref emb) = req.image_embeddings {
        let expected = Modality::Image.dimension();
        if emb.len() != expected {
            return Err(AppError::InvalidInput {
                field: "image_embeddings".into(),
                reason: format!("expected {expected}-dim, got {}-dim", emb.len()),
            });
        }
    }
    if let Some(ref emb) = req.audio_embeddings {
        let expected = Modality::Audio.dimension();
        if emb.len() != expected {
            return Err(AppError::InvalidInput {
                field: "audio_embeddings".into(),
                reason: format!("expected {expected}-dim, got {}-dim", emb.len()),
            });
        }
    }
    if let Some(ref emb) = req.video_embeddings {
        let expected = Modality::Video.dimension();
        if emb.len() != expected {
            return Err(AppError::InvalidInput {
                field: "video_embeddings".into(),
                reason: format!("expected {expected}-dim, got {}-dim", emb.len()),
            });
        }
    }

    // Strict robotics mode: require robot_id and geo_location
    if req.validate_robotics.unwrap_or(false) {
        if req.robot_id.is_none() {
            return Err(AppError::InvalidInput {
                field: "robot_id".into(),
                reason: "validate_robotics=true requires robot_id".into(),
            });
        }
        if req.geo_location.is_none() {
            return Err(AppError::InvalidInput {
                field: "geo_location".into(),
                reason: "validate_robotics=true requires geo_location".into(),
            });
        }
    }

    // Resolve place mentions to coordinates. Deliberately NOT written to
    // geo_location: that field means "recorded here" and feeds the geohash
    // radius index, while these are places the content merely talks about.
    let toponyms = crate::gazetteer::resolve_ner_locations(&ner_entities);

    let experience = Experience {
        content: req.content.clone(),
        experience_type,
        entities: merged_entities.clone(),
        tags: merged_entities,
        context,
        ner_entities,
        toponyms,
        importance_override: req.importance.map(|v| v.clamp(0.0, 1.0)),
        metadata: req.metadata,
        robot_id: req.robot_id.clone(),
        mission_id: req.mission_id.clone(),
        geo_location: req.geo_location,
        local_position: req.local_position,
        heading: req.heading,
        action_type: req.action_type.clone(),
        action_params: req.action_params.clone(),
        reward: req.reward,
        sensor_data: req.sensor_data.clone().unwrap_or_default(),
        outcome_type: req.outcome_type.clone(),
        outcome_details: req.outcome_details.clone(),
        terrain_type: req.terrain_type.clone(),
        is_failure: req.is_failure.unwrap_or(false),
        is_anomaly: req.is_anomaly.unwrap_or(false),
        severity: req.severity.clone(),
        image_embeddings: req.image_embeddings.clone(),
        audio_embeddings: req.audio_embeddings.clone(),
        video_embeddings: req.video_embeddings.clone(),
        media_refs: req.media_refs.clone(),
        ..Default::default()
    };

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // `_detailed` so a content-hash dedup hit reports whether it MERGED
    // enrichment into the stored memory. Without that signal the merged entity
    // set would reach RocksDB and BM25 but never the graph, because the
    // episode already exists and the graph pass is idempotent.
    let outcome = {
        let memory = memory.clone();
        let exp_clone = experience.clone();
        let created_at = req.created_at;
        let agent_id = req.agent_id.clone();
        let run_id = req.run_id.clone();
        let actor_id = req.actor_id.clone();

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            if agent_id.is_some() || run_id.is_some() || actor_id.is_some() {
                memory_guard.remember_with_agent_detailed(
                    exp_clone, created_at, agent_id, run_id, actor_id,
                )
            } else {
                memory_guard.remember_detailed(exp_clone, created_at)
            }
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };
    let memory_id = outcome.id.clone();

    // A merge that added entities makes the memory's stored experience RICHER
    // than the one this request carried, so every downstream pass (graph, NER
    // embeddings, temporal facts) must run on the merged copy, not the request.
    let needs_graph_rebuild = outcome.needs_graph_rebuild();
    let experience = outcome.merged_experience.clone().unwrap_or(experience);

    // Record metrics + session + broadcast BEFORE returning response (fast, <1ms)
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    let session_id = state.session_store().get_or_create_session(&req.user_id);
    state.session_store().add_event(
        &session_id,
        SessionEvent::MemoryCreated {
            timestamp: chrono::Utc::now(),
            memory_id: memory_id.0.to_string(),
            memory_type: experience_type_str.clone(),
            content_preview: req.content.chars().take(100).collect(),
            entities: req.tags.clone(),
        },
    );

    state.emit_event(MemoryEvent {
        event_type: "CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(req.content.chars().take(500).collect()),
        memory_type: Some(experience_type_str),
        importance: None,
        count: None,
        entities: if req.tags.is_empty() {
            None
        } else {
            Some(req.tags.clone())
        },
        results: None,
    });

    // IDEMPOTENCY FIX (issue #109): Return response IMMEDIATELY after persist.
    // The 4 post-processing tasks below are all non-fatal (log errors and continue)
    // and their results are never included in the response. Running them synchronously
    // caused 5-15s handler latency, exceeding the MCP client's 10s timeout and
    // triggering retries that created duplicate memories (31% duplication rate).
    // Now fire-and-forget: response returns in <200ms, post-tasks run in background.
    // Use task_tracker.spawn() so shutdown can await in-flight graph writes.
    let response_id = memory_id.0.to_string();
    {
        let tracker = state.task_tracker.clone();
        let state = state.clone();
        let memory = memory.clone();
        let user_id = req.user_id.clone();
        let content = req.content.clone();
        let experience = experience.clone();
        let parent_id = req.parent_id.clone();
        let created_at = req.created_at;

        tracker.spawn(async move {
            // Pre-compute entity name embeddings for Tier 4 concept merge
            let entity_embeddings = {
                let mem = memory.clone();
                let names: Vec<String> = experience
                    .ner_entities
                    .iter()
                    .map(|e| e.text.clone())
                    .chain(experience.tags.iter().cloned())
                    .collect();
                if names.is_empty() {
                    None
                } else {
                    match tokio::task::spawn_blocking(move || {
                        let guard = mem.read();
                        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
                        guard.get_embedder().encode_batch(&refs).map(|vecs| {
                            names
                                .into_iter()
                                .zip(vecs)
                                .collect::<std::collections::HashMap<String, Vec<f32>>>()
                        })
                    })
                    .await
                    {
                        Ok(Ok(map)) => Some(map),
                        Ok(Err(e)) => {
                            tracing::debug!("Entity name embedding failed (non-fatal): {}", e);
                            None
                        }
                        Err(e) => {
                            tracing::debug!("Entity name embedding task panicked: {}", e);
                            None
                        }
                    }
                }
            };

            // Task 1: Build episodic graph (entities + episode + relationships).
            // On success the episode's surprise components come back — emit them
            // on the SSE stream so live consumers (dashboard anomaly feed) see
            // each episode's statistical shape as it lands. Read-time deviation
            // scoring stays in /api/anomalies; this event carries the raw facts.
            let graph_pass = if needs_graph_rebuild {
                // The episode exists and was built from the pre-merge entity
                // set, so the idempotency guard would skip it. Demolish and
                // rebuild from the MERGED experience — see
                // `AppState::rebuild_experience_graph`.
                state.rebuild_experience_graph(
                    &user_id,
                    &experience,
                    &memory_id,
                    entity_embeddings.as_ref(),
                )
            } else {
                state.process_experience_into_graph(
                    &user_id,
                    &experience,
                    &memory_id,
                    entity_embeddings.as_ref(),
                )
            };

            match graph_pass {
                Ok(Some(surprise)) => {
                    state.emit_event(MemoryEvent {
                        event_type: "surprise".to_string(),
                        timestamp: chrono::Utc::now(),
                        user_id: user_id.clone(),
                        memory_id: Some(memory_id.0.to_string()),
                        content_preview: Some(experience.content.chars().take(100).collect()),
                        memory_type: None,
                        importance: None,
                        count: None,
                        entities: None,
                        results: serde_json::to_value(&surprise).ok(),
                    });
                }
                Ok(None) => {}
                Err(e) => tracing::debug!("Graph processing failed (non-fatal): {}", e),
            }

            // Task 2: Set parent_id for hierarchical organization
            if let Some(ref parent_id_str) = parent_id {
                let resolved_parent = if let Ok(parent_uuid) = uuid::Uuid::parse_str(parent_id_str)
                {
                    Some(crate::memory::MemoryId(parent_uuid))
                } else {
                    let mem = memory.clone();
                    let prefix = parent_id_str.clone();
                    match tokio::task::spawn_blocking(move || {
                        let guard = mem.read();
                        guard
                            .find_memory_by_prefix(&prefix)
                            .ok()
                            .flatten()
                            .map(|m| m.id.clone())
                    })
                    .await
                    {
                        Ok(result) => result,
                        Err(e) => {
                            tracing::warn!("Parent resolve panicked (non-fatal): {e}");
                            None
                        }
                    }
                };

                if let Some(resolved) = resolved_parent {
                    let mem = memory.clone();
                    let mid = memory_id.clone();
                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        let guard = mem.read();
                        guard.set_memory_parent(&mid, Some(resolved))
                    })
                    .await
                    {
                        tracing::warn!("Parent set task panicked (non-fatal): {e}");
                    }
                } else {
                    tracing::warn!("Could not resolve parent_id: {}", parent_id_str);
                }
            }

            // Task 3: Extract and store temporal facts
            {
                let mem = memory.clone();
                let uid = user_id.clone();
                let cnt = content.clone();
                let ents = experience.entities.clone();
                let ts = created_at.unwrap_or_else(chrono::Utc::now);
                let mid = memory_id.clone();

                if let Err(e) = tokio::task::spawn_blocking(move || {
                    let guard = mem.read();
                    guard.store_temporal_facts_for_memory(&uid, &mid, &cnt, &ents, ts)
                })
                .await
                {
                    tracing::warn!("Temporal fact extraction panicked (non-fatal): {e}");
                }
            }

            // Task 4: Infer causal lineage (runs after graph processing)
            spawn_lineage_inference(state, user_id, memory_id);
        });
    }

    // Trace enrichment (witnessed-op capture): the stored memory id is this
    // op's evidence. RememberRequest carries no session_id — the middleware's
    // session-store fallback covers it (same store this handler already uses).
    if let Some(axum::Extension(trace)) = &trace {
        trace.set_identity(&req.user_id, None);
        trace.push_evidence([response_id.clone()]);
    }

    Ok(Json(RememberResponse {
        id: response_id,
        success: true,
    }))
}

/// Batch remember - store multiple memories at once
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, batch_size = req.memories.len()))]
pub async fn batch_remember(
    State(state): State<AppState>,
    trace: Option<axum::Extension<crate::handlers::trace::OpTrace>>,
    Json(req): Json<BatchRememberRequest>,
) -> Result<Json<BatchRememberResponse>, AppError> {
    let op_start = std::time::Instant::now();
    let batch_size = req.memories.len();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Trace identity set at the TOP so every success path — including the
    // empty-batch early return below — is enriched (review I3: a 200 with
    // zero records was being counted as a capture failure).
    if let Some(axum::Extension(trace)) = &trace {
        trace.set_identity(&req.user_id, None);
    }

    if req.memories.is_empty() {
        return Ok(Json(BatchRememberResponse {
            created: 0,
            failed: 0,
            memory_ids: vec![],
            errors: vec![],
        }));
    }

    if req.memories.len() > 1000 {
        return Err(AppError::InvalidInput {
            field: "memories".to_string(),
            reason: "Batch size exceeds 1000 limit".to_string(),
        });
    }

    // Pre-validate all items
    let mut validation_errors: Vec<BatchErrorItem> = Vec::new();
    let mut valid_items: Vec<(usize, BatchMemoryItem)> = Vec::new();

    let mut seen_content: HashSet<u64> = HashSet::new();
    for (index, item) in req.memories.into_iter().enumerate() {
        if let Err(e) = validation::validate_content(&item.content, false) {
            validation_errors.push(BatchErrorItem {
                index,
                error: e.to_string(),
            });
            continue;
        }
        // Deduplicate within the batch: skip items with identical content
        let content_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            item.content.hash(&mut hasher);
            hasher.finish()
        };
        if !seen_content.insert(content_hash) {
            tracing::debug!(
                batch_index = index,
                "Skipping duplicate content in batch (same content already queued)"
            );
            validation_errors.push(BatchErrorItem {
                index,
                error: "Duplicate content within batch".to_string(),
            });
            continue;
        }
        valid_items.push((index, item));
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let extract_entities = req.options.extract_entities;
    let neural_ner = state.get_neural_ner();
    let keyword_extractor = state.get_keyword_extractor();

    // Build experiences
    let mut experiences_with_index: Vec<(
        usize,
        Experience,
        Option<chrono::DateTime<chrono::Utc>>,
    )> = Vec::with_capacity(valid_items.len());

    for (index, item) in valid_items {
        let experience_type = parse_experience_type(item.memory_type.as_ref())?;

        let (merged_entities, ner_records) = if extract_entities {
            // NER for named entities (Person, Org, Location, Misc)
            let ner_records: Vec<NerEntityRecord> = match neural_ner.extract(&item.content) {
                Ok(entities) => entities
                    .into_iter()
                    .map(|e| NerEntityRecord {
                        text: e.text,
                        entity_type: e.entity_type.as_str().to_string(),
                        confidence: e.confidence,
                        start_char: Some(e.start),
                        end_char: Some(e.end),
                        fine_label: e.fine_label,
                    })
                    .collect(),
                Err(e) => {
                    tracing::warn!(
                        "NER extraction failed for batch item {index} — storing memory with NO \
                         typed entities: {e}"
                    );
                    Vec::new()
                }
            };

            // YAKE for common nouns, verbs, concepts
            let extracted_keywords: Vec<String> = keyword_extractor.extract_texts(&item.content);

            let mut merged: Vec<String> = item.tags.clone();
            let mut seen: HashSet<String> = merged.iter().map(|t| t.to_lowercase()).collect();
            for record in &ner_records {
                if seen.insert(record.text.to_lowercase()) {
                    merged.push(record.text.clone());
                }
            }
            for keyword in extracted_keywords {
                if seen.insert(keyword.to_lowercase()) {
                    merged.push(keyword);
                }
            }
            if merged.len() > validation::MAX_ENTITIES_PER_MEMORY {
                tracing::debug!(
                    batch_index = index,
                    count = merged.len(),
                    max = validation::MAX_ENTITIES_PER_MEMORY,
                    "Capping entities to maximum allowed in batch item"
                );
                merged.truncate(validation::MAX_ENTITIES_PER_MEMORY);
            }
            (merged, ner_records)
        } else {
            (item.tags.clone(), Vec::new())
        };

        let context = build_rich_context(
            item.emotional_valence,
            item.emotional_arousal,
            item.emotion.clone(),
            item.source_type.clone(),
            item.credibility,
            item.episode_id.clone(),
            item.sequence_number,
            item.preceding_memory_id.clone(),
        );

        let toponyms = crate::gazetteer::resolve_ner_locations(&ner_records);

        let experience = Experience {
            content: item.content,
            experience_type,
            entities: merged_entities.clone(),
            tags: merged_entities,
            context,
            ner_entities: ner_records,
            toponyms,
            importance_override: item.importance.map(|v| v.clamp(0.0, 1.0)),
            ..Default::default()
        };

        experiences_with_index.push((index, experience, item.created_at));
    }

    // Store memories
    let (memory_results, storage_errors) = {
        let memory = memory.clone();
        let experiences = experiences_with_index;
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            // (index, memory_id, experience-to-graph, rebuild-required).
            // The experience carried forward is the MERGED one when a dedup hit
            // enriched the stored memory, so the graph pass below sees the union
            // rather than just this item's own entities.
            let mut results: Vec<(usize, String, Experience, bool)> =
                Vec::with_capacity(experiences.len());
            let mut errors: Vec<BatchErrorItem> = Vec::new();

            for (index, experience, created_at) in experiences {
                match memory_guard.remember_detailed(experience.clone(), created_at) {
                    Ok(outcome) => {
                        let rebuild = outcome.needs_graph_rebuild();
                        let effective = outcome.merged_experience.unwrap_or(experience);
                        results.push((index, outcome.id.0.to_string(), effective, rebuild));
                    }
                    Err(e) => {
                        errors.push(BatchErrorItem {
                            index,
                            error: e.to_string(),
                        });
                    }
                }
            }
            (results, errors)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    let memory_ids: Vec<String> = memory_results
        .iter()
        .map(|(_, id, _, _)| id.clone())
        .collect();
    let created = memory_ids.len();

    let mut all_errors = validation_errors;
    all_errors.extend(storage_errors);
    all_errors.sort_by_key(|e| e.index);
    let failed = all_errors.len();

    // Build episodic graph for each stored memory (enables multi-hop retrieval)
    // Then fire-and-forget lineage inference for each.
    for (_, id_str, experience, rebuild) in &memory_results {
        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
            let memory_id = crate::memory::MemoryId(uuid);
            let graph_pass = if *rebuild {
                state.rebuild_experience_graph(&req.user_id, experience, &memory_id, None)
            } else {
                state.process_experience_into_graph(&req.user_id, experience, &memory_id, None)
            };
            if let Err(e) = graph_pass {
                tracing::debug!("Graph processing failed for {} (non-fatal): {}", id_str, e);
            }
            spawn_lineage_inference(state.clone(), req.user_id.clone(), memory_id);
        }
    }

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::BATCH_STORE_DURATION.observe(duration);
    metrics::BATCH_STORE_SIZE.observe(batch_size as f64);
    for _ in 0..created {
        metrics::MEMORY_STORE_TOTAL
            .with_label_values(&["success"])
            .inc();
    }
    for _ in 0..failed {
        metrics::MEMORY_STORE_TOTAL
            .with_label_values(&["error"])
            .inc();
    }

    // Trace enrichment (witnessed-op capture): all created ids are evidence.
    if let Some(axum::Extension(trace)) = &trace {
        trace.set_identity(&req.user_id, None);
        trace.push_evidence(memory_ids.iter().cloned());
        trace.set_summary(format!("batch: {created} created, {failed} failed"));
    }

    Ok(Json(BatchRememberResponse {
        created,
        failed,
        memory_ids,
        errors: all_errors,
    }))
}

/// Upsert memory - create or update with external ID linking
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, external_id = %req.external_id))]
pub async fn upsert_memory(
    State(state): State<AppState>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    if req.external_id.is_empty() {
        return Err(AppError::InvalidInput {
            field: "external_id".to_string(),
            reason: "external_id is required for upsert".to_string(),
        });
    }

    let experience_type = parse_experience_type(req.memory_type.as_ref())?;

    let change_type = match req.change_type.to_lowercase().as_str() {
        "created" => ChangeType::Created,
        "content_updated" => ChangeType::ContentUpdated,
        "status_changed" => ChangeType::StatusChanged,
        "tags_updated" => ChangeType::TagsUpdated,
        "importance_adjusted" => ChangeType::ImportanceAdjusted,
        _ => ChangeType::ContentUpdated,
    };

    // Extract entities via NER (preserve full records for graph label propagation)
    let ner_entities: Vec<NerEntityRecord> = match state.get_neural_ner().extract(&req.content) {
        Ok(entities) => entities
            .into_iter()
            .map(|e| NerEntityRecord {
                text: e.text,
                entity_type: e.entity_type.as_str().to_string(),
                confidence: e.confidence,
                start_char: Some(e.start),
                end_char: Some(e.end),
                fine_label: e.fine_label,
            })
            .collect(),
        Err(e) => {
            tracing::warn!(
                "NER extraction failed in upsert — storing memory with NO typed entities: {e}"
            );
            Vec::new()
        }
    };

    // Extract keywords via YAKE for common nouns, verbs, concepts
    let extracted_keywords: Vec<String> = state.get_keyword_extractor().extract_texts(&req.content);

    let mut merged_entities: Vec<String> = req.tags.clone();
    let mut seen: HashSet<String> = merged_entities.iter().map(|t| t.to_lowercase()).collect();
    for record in &ner_entities {
        if seen.insert(record.text.to_lowercase()) {
            if validation::validate_entity(&record.text).is_ok() {
                merged_entities.push(record.text.clone());
            } else {
                tracing::debug!(entity = %record.text, "Skipping invalid NER entity in upsert (too long or invalid chars)");
            }
        }
    }
    for keyword in extracted_keywords {
        if seen.insert(keyword.to_lowercase()) {
            if validation::validate_entity(&keyword).is_ok() {
                merged_entities.push(keyword);
            } else {
                tracing::debug!(entity = %keyword, "Skipping invalid YAKE keyword in upsert (too long or invalid chars)");
            }
        }
    }
    if merged_entities.len() > validation::MAX_ENTITIES_PER_MEMORY {
        tracing::debug!(
            count = merged_entities.len(),
            max = validation::MAX_ENTITIES_PER_MEMORY,
            "Capping entities to maximum allowed in upsert"
        );
        merged_entities.truncate(validation::MAX_ENTITIES_PER_MEMORY);
    }

    let toponyms = crate::gazetteer::resolve_ner_locations(&ner_entities);

    let experience = Experience {
        content: req.content.clone(),
        experience_type,
        entities: merged_entities.clone(),
        tags: merged_entities,
        ner_entities,
        toponyms,
        importance_override: req.importance.map(|v| v.clamp(0.0, 1.0)),
        ..Default::default()
    };

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let external_id = req.external_id.clone();
    let changed_by = req.changed_by.clone();
    let change_reason = req.change_reason.clone();

    let (memory_id, was_update) = {
        let memory = memory_system.clone();
        let exp = experience.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.upsert(external_id, exp, change_type, changed_by, change_reason)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    let version = {
        let memory = memory_system.clone();
        let mid = memory_id.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard
                .get_memory(&mid)
                .map(|m| m.version)
                .unwrap_or(1)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // Build episodic graph for multi-hop retrieval.
    // On updates the old episode must be demolished first, or the idempotency
    // guard in process_experience_into_graph would keep the stale entity set.
    // Shared with the dedup-merge path in `remember` — see
    // `AppState::rebuild_experience_graph`.
    let graph_result = if was_update {
        state.rebuild_experience_graph(&req.user_id, &experience, &memory_id, None)
    } else {
        state.process_experience_into_graph(&req.user_id, &experience, &memory_id, None)
    };
    if let Err(e) = graph_result {
        tracing::debug!("Graph processing failed (non-fatal): {}", e);
    }

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&[if was_update {
            "upsert_update"
        } else {
            "upsert_create"
        }])
        .inc();

    // Broadcast event
    state.emit_event(MemoryEvent {
        event_type: if was_update {
            "UPDATE".to_string()
        } else {
            "CREATE".to_string()
        },
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(req.content.chars().take(500).collect()),
        memory_type: req.memory_type.clone(),
        importance: None,
        count: None,
        entities: if req.tags.is_empty() {
            None
        } else {
            Some(req.tags.clone())
        },
        results: None,
    });

    // Fire-and-forget lineage inference (only for new memories, not updates)
    if !was_update {
        spawn_lineage_inference(state.clone(), req.user_id.clone(), memory_id.clone());
    }

    Ok(Json(UpsertResponse {
        id: memory_id.0.to_string(),
        success: true,
        was_update,
        version,
    }))
}

/// Spawn fire-and-forget lineage inference for a newly stored memory.
///
/// Resolves the memory's entity graph, finds temporal candidates, infers causal
/// edges, and strengthens corresponding knowledge graph connections. All failures
/// are logged but never propagate — lineage is best-effort.
fn spawn_lineage_inference(state: AppState, user_id: String, memory_id: crate::memory::MemoryId) {
    let tracker = state.task_tracker.clone();
    tracker.spawn(async move {
        let graph_arc = match state.get_user_graph(&user_id) {
            Ok(g) => g,
            Err(e) => {
                tracing::debug!(
                    user_id = %user_id,
                    memory_id = %memory_id.0,
                    error = %e,
                    "Lineage inference skipped: graph initialization failed"
                );
                return;
            }
        };
        let memory_arc = match state.get_user_memory(&user_id) {
            Ok(m) => m,
            Err(_) => return,
        };

        let uid = user_id;
        let mid = memory_id;

        if let Err(e) = tokio::task::spawn_blocking(move || {
            let graph = graph_arc.read();
            let memory_guard = memory_arc.read();

            let episode = match graph.get_episode(&mid.0) {
                Ok(Some(ep)) => ep,
                Ok(None) => {
                    tracing::debug!(
                        memory_id = %mid.0,
                        "Lineage skipped: no episode found (graph processing may have failed)"
                    );
                    return;
                }
                Err(e) => {
                    tracing::debug!(
                        memory_id = %mid.0,
                        error = %e,
                        "Lineage skipped: episode lookup failed"
                    );
                    return;
                }
            };

            let mut candidate_ids = std::collections::HashSet::new();
            let cutoff =
                chrono::Utc::now() - chrono::Duration::days(crate::constants::LINEAGE_LOOKBACK_DAYS);

            // Phase 1: Entity-graph candidates (highest quality — shared entities)
            if !episode.entity_refs.is_empty() {
                for entity_uuid in &episode.entity_refs {
                    if let Ok(episodes) = graph.get_episodes_by_entity(entity_uuid) {
                        for ep in &episodes {
                            if ep.created_at >= cutoff {
                                candidate_ids.insert(crate::memory::MemoryId(ep.uuid));
                            }
                        }
                    }
                }
            }

            // Phase 1.5: Semantic candidates via vector index.
            // Finds memories with similar content even when entity names differ
            // (e.g. "ORT" vs "ONNX Runtime"). Higher quality than recency fallback
            // because similarity is content-based, not just temporal.
            if candidate_ids.len() < crate::constants::LINEAGE_MAX_CANDIDATES {
                if let Ok(new_memory) = memory_guard.get_memory(&mid) {
                    if let Some(embedding) = &new_memory.experience.embeddings {
                        let remaining =
                            crate::constants::LINEAGE_MAX_CANDIDATES - candidate_ids.len();
                        if let Ok(similar) = memory_guard.search_similar_by_embedding(
                            embedding,
                            remaining,
                            Some(&mid),
                        ) {
                            for (mem_id, _score) in similar {
                                // Filter to lookback window (same as Phase 1)
                                if let Ok(mem) = memory_guard.get_memory(&mem_id) {
                                    if mem.created_at >= cutoff {
                                        candidate_ids.insert(mem_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Phase 2: Recency fallback — fill remaining slots with recent memories.
            // This ensures lineage inference runs even when NER fails (empty entity_refs)
            // or when entity-graph candidates are sparse.
            // Fetch more than needed, sort by newest first, then take `remaining`.
            // recall_by_date returns oldest-first (date index order), so we reverse.
            if candidate_ids.len() < crate::constants::LINEAGE_MAX_CANDIDATES {
                let remaining = crate::constants::LINEAGE_MAX_CANDIDATES - candidate_ids.len();
                // Over-fetch to get a representative sample, then pick most recent
                let fetch_limit = remaining * 3;
                if let Ok(mut recent) = memory_guard.recall_by_date(
                    cutoff,
                    chrono::Utc::now(),
                    fetch_limit,
                ) {
                    recent.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                    for mem in recent.into_iter().take(remaining) {
                        candidate_ids.insert(mem.id.clone());
                    }
                }
            }

            candidate_ids.remove(&mid);
            let candidate_ids: Vec<_> = candidate_ids
                .into_iter()
                .take(crate::constants::LINEAGE_MAX_CANDIDATES)
                .collect();
            if candidate_ids.is_empty() {
                return;
            }

            let candidates: Vec<_> = candidate_ids
                .iter()
                .filter_map(|id| memory_guard.get_memory(id).ok())
                .collect();

            let Ok(new_memory) = memory_guard.get_memory(&mid) else {
                return;
            };

            match memory_guard.infer_lineage_for_memory(&uid, &new_memory, &candidates) {
                Ok(edges) if !edges.is_empty() => {
                    tracing::info!(
                        user_id = %uid,
                        memory_id = %mid.0,
                        edges = edges.len(),
                        relations = ?edges.iter().map(|e| format!("{:?}", e.relation)).collect::<Vec<_>>(),
                        "Lineage inference: {} causal edges detected",
                        edges.len()
                    );

                    // Propagate lineage confidence into graph edge weights
                    // AND create typed causal edges visible to spreading activation
                    let boost_scale = crate::constants::LINEAGE_GRAPH_BOOST_SCALE;
                    let mut total_strengthened = 0usize;
                    let mut total_typed_edges = 0usize;
                    for edge in &edges {
                        let boost = edge.confidence * boost_scale;
                        match graph.strengthen_lineage_connection(
                            &edge.from.0,
                            &edge.to.0,
                            boost,
                        ) {
                            Ok(n) => total_strengthened += n,
                            Err(e) => tracing::debug!(
                                "Lineage→graph strengthening failed (non-fatal): {}", e
                            ),
                        }

                        // Create typed causal edges (Causes, Triggers, SupersededBy, etc.)
                        let graph_rel = edge.relation.to_graph_relation_type();
                        match graph.create_lineage_graph_edges(
                            &edge.from.0,
                            &edge.to.0,
                            graph_rel,
                            edge.confidence,
                        ) {
                            Ok(n) => total_typed_edges += n,
                            Err(e) => tracing::debug!(
                                "Lineage→graph typed edge creation failed (non-fatal): {}", e
                            ),
                        }
                    }
                    if total_strengthened > 0 || total_typed_edges > 0 {
                        tracing::debug!(
                            user_id = %uid,
                            lineage_edges = edges.len(),
                            graph_edges_strengthened = total_strengthened,
                            graph_typed_edges_created = total_typed_edges,
                            "Lineage→graph integration complete"
                        );
                    }
                }
                Ok(_) => {
                    tracing::debug!(
                        "Lineage inference: no causal edges for {} (checked {} candidates)",
                        mid.0,
                        candidates.len()
                    );
                }
                Err(e) => {
                    tracing::debug!("Lineage inference failed (non-fatal): {}", e);
                }
            }
        })
        .await
        {
            tracing::warn!("Lineage inference panicked (non-fatal): {e}");
        }
    });
}
