//! Remember Handlers - Memory Storage Operations
//!
//! Core handlers for storing memories: remember, batch_remember, upsert.

use axum::{extract::State, response::Json};

use super::health::AppState;
use super::types::MemoryEvent;
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::{
    types::{
        ChangeType, ContextId, EmotionalContext, EpisodeContext, RichContext, SourceContext,
        SourceType,
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
    #[serde(default, alias = "experience_type")]
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
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub parent_agent_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
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
#[derive(Debug, serde::Deserialize, Clone, Default)]
pub struct BatchRememberOptions {
    #[serde(default = "default_true")]
    pub extract_entities: bool,
    #[serde(default = "default_true")]
    pub create_edges: bool,
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
    #[serde(default, alias = "experience_type")]
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
    #[serde(default, alias = "experience_type")]
    pub memory_type: Option<String>,
    #[serde(default = "default_change_type")]
    pub change_type: String,
    #[serde(default)]
    pub changed_by: Option<String>,
    #[serde(default)]
    pub change_reason: Option<String>,
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

/// Parse memory type from string
pub fn parse_experience_type(s: Option<&String>) -> ExperienceType {
    s.and_then(|s| match s.to_lowercase().as_str() {
        "task" => Some(ExperienceType::Task),
        "learning" => Some(ExperienceType::Learning),
        "decision" => Some(ExperienceType::Decision),
        "error" => Some(ExperienceType::Error),
        "pattern" => Some(ExperienceType::Pattern),
        "conversation" => Some(ExperienceType::Conversation),
        "discovery" => Some(ExperienceType::Discovery),
        "observation" | "context" => Some(ExperienceType::Context),
        _ => None,
    })
    .unwrap_or(ExperienceType::Context)
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
        is_episode_start: sequence_number == Some(1),
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
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    let experience_type = parse_experience_type(req.memory_type.as_ref());

    // Extract entities via NER and merge with user tags
    let extracted_names: Vec<String> = match state.get_neural_ner().extract(&req.content) {
        Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
        Err(e) => {
            tracing::debug!("NER extraction failed: {}", e);
            Vec::new()
        }
    };

    // Extract keywords via YAKE for common nouns, verbs, etc.
    // NER only extracts named entities (Person, Org, Location, Misc)
    // YAKE captures important terms like "sunrise", "painting", "lake"
    let extracted_keywords: Vec<String> = state.get_keyword_extractor().extract_texts(&req.content);

    let mut merged_entities: Vec<String> = req.tags.clone();
    for entity_name in extracted_names {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&entity_name))
        {
            merged_entities.push(entity_name);
        }
    }
    for keyword in extracted_keywords {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&keyword))
        {
            merged_entities.push(keyword);
        }
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

    let experience = Experience {
        content: req.content.clone(),
        experience_type,
        entities: merged_entities.clone(),
        tags: merged_entities, // Include NER + YAKE keywords in tags for retrieval
        context,
        ..Default::default()
    };

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_id = {
        let memory = memory.clone();
        let exp_clone = experience.clone();
        let created_at = req.created_at;
        let agent_id = req.agent_id.clone();
        let run_id = req.run_id.clone();

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            if agent_id.is_some() || run_id.is_some() {
                memory_guard.remember_with_agent(exp_clone, created_at, agent_id, run_id)
            } else {
                memory_guard.remember(exp_clone, created_at)
            }
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Build episodic graph: entities + episode + relationships for multi-hop retrieval
    if let Err(e) = state.process_experience_into_graph(&req.user_id, &experience, &memory_id) {
        tracing::debug!("Graph processing failed (non-fatal): {}", e);
    }

    // Extract and store temporal facts for multi-hop temporal reasoning
    // E.g., "planning camping next month" → resolves "next month" to absolute date
    {
        let memory = memory.clone();
        let user_id = req.user_id.clone();
        let content = req.content.clone();
        let entities = experience.entities.clone();
        let created_at = req.created_at.unwrap_or_else(chrono::Utc::now);
        let memory_id_clone = memory_id.clone(); // Clone before moving into closure

        // Run temporal fact extraction in background (non-blocking)
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            if let Err(e) = memory_guard.store_temporal_facts_for_memory(
                &user_id,
                &memory_id_clone,
                &content,
                &entities,
                created_at,
            ) {
                tracing::debug!("Temporal fact extraction failed (non-fatal): {}", e);
            }
        });
    }

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    // Track session event
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

    // Broadcast CREATE event
    state.emit_event(MemoryEvent {
        event_type: "CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(req.content.chars().take(100).collect()),
        memory_type: Some(experience_type_str),
        importance: None,
        count: None,
    });

    Ok(Json(RememberResponse {
        id: memory_id.0.to_string(),
        success: true,
    }))
}

/// Batch remember - store multiple memories at once
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, batch_size = req.memories.len()))]
pub async fn batch_remember(
    State(state): State<AppState>,
    Json(req): Json<BatchRememberRequest>,
) -> Result<Json<BatchRememberResponse>, AppError> {
    let op_start = std::time::Instant::now();
    let batch_size = req.memories.len();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

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

    for (index, item) in req.memories.into_iter().enumerate() {
        if let Err(e) = validation::validate_content(&item.content, false) {
            validation_errors.push(BatchErrorItem {
                index,
                error: e.to_string(),
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
        let experience_type = parse_experience_type(item.memory_type.as_ref());

        let merged_entities = if extract_entities {
            // NER for named entities (Person, Org, Location, Misc)
            let extracted_names: Vec<String> = match neural_ner.extract(&item.content) {
                Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
                Err(e) => {
                    tracing::debug!("NER extraction failed for batch item {}: {}", index, e);
                    Vec::new()
                }
            };

            // YAKE for common nouns, verbs, concepts
            let extracted_keywords: Vec<String> = keyword_extractor.extract_texts(&item.content);

            let mut merged: Vec<String> = item.tags.clone();
            for entity_name in extracted_names {
                if !merged.iter().any(|t| t.eq_ignore_ascii_case(&entity_name)) {
                    merged.push(entity_name);
                }
            }
            for keyword in extracted_keywords {
                if !merged.iter().any(|t| t.eq_ignore_ascii_case(&keyword)) {
                    merged.push(keyword);
                }
            }
            merged
        } else {
            item.tags.clone()
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

        let experience = Experience {
            content: item.content,
            experience_type,
            entities: merged_entities.clone(),
            tags: merged_entities, // Include NER + YAKE keywords in tags for retrieval
            context,
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
            let mut results: Vec<(usize, String, Experience)> =
                Vec::with_capacity(experiences.len());
            let mut errors: Vec<BatchErrorItem> = Vec::new();

            for (index, experience, created_at) in experiences {
                match memory_guard.remember(experience.clone(), created_at) {
                    Ok(id) => {
                        results.push((index, id.0.to_string(), experience));
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

    let memory_ids: Vec<String> = memory_results.iter().map(|(_, id, _)| id.clone()).collect();
    let created = memory_ids.len();

    let mut all_errors = validation_errors;
    all_errors.extend(storage_errors);
    all_errors.sort_by_key(|e| e.index);
    let failed = all_errors.len();

    // Build episodic graph for each stored memory (enables multi-hop retrieval)
    for (_, id_str, experience) in &memory_results {
        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
            let memory_id = crate::memory::MemoryId(uuid);
            if let Err(e) =
                state.process_experience_into_graph(&req.user_id, experience, &memory_id)
            {
                tracing::debug!("Graph processing failed for {} (non-fatal): {}", id_str, e);
            }
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

    let experience_type = parse_experience_type(req.memory_type.as_ref());

    let change_type = match req.change_type.to_lowercase().as_str() {
        "created" => ChangeType::Created,
        "content_updated" => ChangeType::ContentUpdated,
        "status_changed" => ChangeType::StatusChanged,
        "tags_updated" => ChangeType::TagsUpdated,
        "importance_adjusted" => ChangeType::ImportanceAdjusted,
        _ => ChangeType::ContentUpdated,
    };

    // Extract entities via NER
    let extracted_names: Vec<String> = match state.get_neural_ner().extract(&req.content) {
        Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
        Err(e) => {
            tracing::debug!("NER extraction failed in upsert: {}", e);
            Vec::new()
        }
    };

    // Extract keywords via YAKE for common nouns, verbs, concepts
    let extracted_keywords: Vec<String> = state.get_keyword_extractor().extract_texts(&req.content);

    let mut merged_entities: Vec<String> = req.tags.clone();
    for entity_name in extracted_names {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&entity_name))
        {
            merged_entities.push(entity_name);
        }
    }
    for keyword in extracted_keywords {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&keyword))
        {
            merged_entities.push(keyword);
        }
    }

    let experience = Experience {
        content: req.content.clone(),
        experience_type,
        entities: merged_entities.clone(),
        tags: merged_entities, // Include NER + YAKE keywords in tags for retrieval
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

    // Build episodic graph for multi-hop retrieval
    if let Err(e) = state.process_experience_into_graph(&req.user_id, &experience, &memory_id) {
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
        content_preview: Some(req.content.chars().take(100).collect()),
        memory_type: req.memory_type.clone(),
        importance: None,
        count: None,
    });

    Ok(Json(UpsertResponse {
        id: memory_id.0.to_string(),
        success: true,
        was_update,
        version,
    }))
}
