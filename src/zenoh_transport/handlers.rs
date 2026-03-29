//! Zenoh Message Handlers
//!
//! Translates Zenoh samples into shodh-memory operations.
//! Each handler mirrors an equivalent HTTP handler in `src/handlers/`,
//! reusing the same core types, validation, and processing pipeline.
//!
//! # Design Principles
//! - All blocking MemorySystem operations use `spawn_blocking`
//! - Post-processing (graph, lineage, facts) is fire-and-forget (async, non-fatal)
//! - Handlers never panic — all errors are logged and the subscriber continues
//! - JSON is the primary wire format (matches existing serde derives)

use std::collections::HashSet;
use std::sync::Arc;

use tracing::{debug, error, info, warn};
use zenoh::bytes::ZBytes;
use zenoh::query::Query;
use zenoh::sample::Sample;

use crate::errors::ValidationErrorExt;
use crate::handlers::remember::{build_rich_context, parse_experience_type};
use crate::handlers::state::MultiUserMemoryManager;
use crate::handlers::types::{
    MemoryEvent, RecallExperience, RecallFact, RecallMemory, RecallResponse, RecallTodo,
};
use crate::memory::feedback::{self, PendingFeedback, SurfacedMemoryInfo, ToolAction};
use crate::memory::types::{ForgetCriteria, GeoFilter, MemoryId};
use crate::memory::{
    Experience, MemorySystem, Query as MemoryQuery, RetrievalMode, SessionEvent, TodoStatus,
};
use crate::metrics;
use crate::streaming::{ExtractionConfig, StreamHandshake, StreamMessage, StreamMode};
use crate::validation;

// =============================================================================
// AUTHENTICATION
// =============================================================================

/// Validate the `"api_key"` field in a Zenoh JSON payload against the configured secret.
///
/// Returns `true` if the request should be processed:
/// - If `expected` is `None` (no key configured), authentication is skipped (backwards compatible).
/// - If `expected` is `Some`, the payload must be valid JSON containing an `"api_key"` field
///   whose value matches the expected key via constant-time comparison.
///
/// Returns `false` and logs a warning if authentication fails.
pub fn authenticate_payload(payload: &ZBytes, expected: Option<&str>) -> bool {
    let expected = match expected {
        Some(key) => key,
        None => return true, // No auth configured — allow all
    };

    let text = match payload.try_to_string() {
        Ok(cow) => cow.into_owned(),
        Err(_) => {
            warn!("Zenoh auth: payload is not valid UTF-8, rejecting");
            return false;
        }
    };

    // Parse just enough to extract the api_key field without deserializing the full payload
    let provided = match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(val) => val
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        Err(_) => {
            warn!("Zenoh auth: payload is not valid JSON, rejecting");
            return false;
        }
    };

    match provided {
        Some(ref key) if constant_time_eq(key.as_bytes(), expected.as_bytes()) => true,
        Some(_) => {
            warn!("Zenoh auth: api_key mismatch, rejecting request");
            false
        }
        None => {
            warn!("Zenoh auth: payload missing 'api_key' field, rejecting request");
            false
        }
    }
}

/// Constant-time byte comparison to prevent timing side-channels on key validation.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// =============================================================================
// PAYLOAD HELPERS
// =============================================================================

/// Extract a UTF-8 string from a Zenoh sample payload.
fn payload_to_string(payload: &ZBytes) -> Result<String, String> {
    payload
        .try_to_string()
        .map(|cow| cow.into_owned())
        .map_err(|e| format!("Invalid UTF-8 payload: {e}"))
}

/// Deserialize a JSON payload from a Zenoh sample.
fn deserialize_json<T: serde::de::DeserializeOwned>(payload: &ZBytes) -> Result<T, String> {
    let text = payload_to_string(payload)?;
    serde_json::from_str(&text).map_err(|e| format!("JSON deserialization failed: {e}"))
}

/// Extract user_id from a key expression.
///
/// Given a key like `shodh/robot-1/remember` and prefix `shodh`,
/// extracts `robot-1`.
///
/// Returns `None` if the key doesn't match the expected structure.
pub fn extract_user_id<'a>(key_expr: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = key_expr.strip_prefix(prefix)?.strip_prefix('/')?;
    let slash = rest.find('/')?;
    let user_id = &rest[..slash];
    if user_id.is_empty() {
        return None;
    }
    Some(user_id)
}

// =============================================================================
// ZENOH-SPECIFIC REQUEST TYPES
// =============================================================================

/// Zenoh remember request — extends the HTTP RememberRequest with full robotics fields.
///
/// Robots publishing via Zenoh can include spatial, sensor, and mission context
/// directly in the remember payload, which flows through to the Experience struct.
#[derive(Debug, serde::Deserialize)]
pub struct ZenohRememberRequest {
    // === Core fields (same as HTTP RememberRequest) ===
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
    pub run_id: Option<String>,
    #[serde(default)]
    pub parent_id: Option<String>,

    // === Robotics fields ===
    /// Robot/drone identifier for multi-agent fleet systems
    #[serde(default)]
    pub robot_id: Option<String>,
    /// Mission identifier this experience belongs to
    #[serde(default)]
    pub mission_id: Option<String>,
    /// GPS coordinates [latitude, longitude, altitude]
    #[serde(default)]
    pub geo_location: Option<[f64; 3]>,
    /// Local frame coordinates [x, y, z] in meters
    #[serde(default)]
    pub local_position: Option<[f32; 3]>,
    /// Heading in degrees (0-360)
    #[serde(default)]
    pub heading: Option<f32>,
    /// Action that was performed (for action-outcome learning)
    #[serde(default)]
    pub action_type: Option<String>,
    /// Reward signal for reinforcement learning (-1.0 to 1.0)
    #[serde(default)]
    pub reward: Option<f32>,
    /// Sensor readings at time of experience: {"battery": 75.5, "temperature": 28.3}
    #[serde(default)]
    pub sensor_data: std::collections::HashMap<String, f64>,
    /// Decision context: {"battery_low": "true", "obstacle_ahead": "true"}
    #[serde(default)]
    pub decision_context: Option<std::collections::HashMap<String, String>>,
    /// Action parameters: {"speed": "0.5", "turn_angle": "45"}
    #[serde(default)]
    pub action_params: Option<std::collections::HashMap<String, String>>,
    /// Outcome type: success, failure, partial, aborted, timeout
    #[serde(default)]
    pub outcome_type: Option<String>,
    /// Outcome details
    #[serde(default)]
    pub outcome_details: Option<String>,
    /// Confidence score (0.0-1.0)
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Terrain type: indoor, outdoor, urban, rural, water, aerial
    #[serde(default)]
    pub terrain_type: Option<String>,
    /// Nearby agents: [{"id": "drone_002", "distance": "50m"}]
    #[serde(default)]
    pub nearby_agents: Vec<std::collections::HashMap<String, String>>,
    /// Is this a failure/error event?
    #[serde(default)]
    pub is_failure: bool,
    /// Is this an anomaly?
    #[serde(default)]
    pub is_anomaly: bool,
    /// Severity level: info, warning, error, critical
    #[serde(default)]
    pub severity: Option<String>,
}

/// Zenoh recall request — extends the HTTP RecallRequest with robotics filters.
///
/// Supports spatial, mission, and action-outcome retrieval modes.
#[derive(Debug, serde::Deserialize)]
pub struct ZenohRecallRequest {
    pub user_id: String,
    pub query: String,
    #[serde(default = "default_recall_limit")]
    pub limit: usize,
    #[serde(default = "default_recall_mode")]
    pub mode: String,

    // === Robotics filters ===
    /// Filter by robot/drone identifier
    #[serde(default)]
    pub robot_id: Option<String>,
    /// Filter by mission identifier
    #[serde(default)]
    pub mission_id: Option<String>,
    /// Spatial filter: center latitude
    #[serde(default)]
    pub lat: Option<f64>,
    /// Spatial filter: center longitude
    #[serde(default)]
    pub lon: Option<f64>,
    /// Spatial filter: search radius in meters
    #[serde(default)]
    pub radius_meters: Option<f64>,
    /// Filter by action type
    #[serde(default)]
    pub action_type: Option<String>,
    /// Filter by minimum reward
    #[serde(default)]
    pub min_reward: Option<f32>,
    /// Filter by maximum reward
    #[serde(default)]
    pub max_reward: Option<f32>,
    /// Filter for failures only
    #[serde(default)]
    pub failures_only: bool,
    /// Filter for anomalies only
    #[serde(default)]
    pub anomalies_only: bool,
    /// Filter by terrain type
    #[serde(default)]
    pub terrain_type: Option<String>,
}

fn default_recall_limit() -> usize {
    5
}

fn default_recall_mode() -> String {
    "hybrid".to_string()
}

// =============================================================================
// REMEMBER HANDLER
// =============================================================================

/// Handle a Zenoh PUT on `{prefix}/{user_id}/remember`.
///
/// Mirrors `src/handlers/remember.rs::remember()` with full robotics field support:
/// 1. Deserialize JSON → ZenohRememberRequest (includes spatial, sensor, mission fields)
/// 2. Validate user_id and content
/// 3. Extract entities (NER + YAKE) in parallel
/// 4. Build Experience with robotics fields, store via MemorySystem
/// 5. Fire-and-forget graph + lineage processing
pub async fn handle_remember(sample: Sample, manager: Arc<MultiUserMemoryManager>) {
    let key = sample.key_expr().as_str().to_string();

    let req: ZenohRememberRequest = match deserialize_json(sample.payload()) {
        Ok(r) => r,
        Err(e) => {
            warn!(key = %key, "Invalid remember payload: {}", e);
            return;
        }
    };

    if let Err(e) = validation::validate_user_id(&req.user_id).map_validation_err("user_id") {
        warn!(user_id = %req.user_id, "Validation failed: {}", e);
        return;
    }
    if let Err(e) = validation::validate_content(&req.content, false).map_validation_err("content")
    {
        warn!(user_id = %req.user_id, "Validation failed: {}", e);
        return;
    }

    let op_start = std::time::Instant::now();
    let experience_type = parse_experience_type(req.memory_type.as_ref());

    // Parallel NER + YAKE extraction (same as HTTP handler)
    let ner = manager.get_neural_ner();
    let yake = manager.get_keyword_extractor();
    let content_for_ner = req.content.clone();
    let content_for_yake = req.content.clone();

    let (ner_result, yake_result) = tokio::join!(
        tokio::task::spawn_blocking(move || {
            match ner.extract(&content_for_ner) {
                Ok(entities) => entities
                    .into_iter()
                    .map(|e| crate::memory::types::NerEntityRecord {
                        text: e.text,
                        entity_type: e.entity_type.as_str().to_string(),
                        confidence: e.confidence,
                        start_char: Some(e.start),
                        end_char: Some(e.end),
                    })
                    .collect::<Vec<_>>(),
                Err(e) => {
                    debug!("NER extraction failed (non-fatal): {}", e);
                    Vec::new()
                }
            }
        }),
        tokio::task::spawn_blocking(move || yake.extract_texts(&content_for_yake))
    );

    let ner_entities = ner_result.unwrap_or_else(|e| {
        error!("NER task panicked: {:?}", e);
        Vec::new()
    });
    let extracted_keywords = yake_result.unwrap_or_else(|e| {
        error!("YAKE task panicked: {:?}", e);
        Vec::new()
    });

    // Merge entities with dedup (same as HTTP handler)
    let mut merged_entities: Vec<String> = req.tags.clone();
    let mut seen: HashSet<String> = merged_entities.iter().map(|t| t.to_lowercase()).collect();
    for record in &ner_entities {
        if seen.insert(record.text.to_lowercase()) {
            merged_entities.push(record.text.clone());
        }
    }
    for keyword in extracted_keywords {
        if seen.insert(keyword.to_lowercase()) {
            merged_entities.push(keyword);
        }
    }
    if merged_entities.len() > validation::MAX_ENTITIES_PER_MEMORY {
        merged_entities.truncate(validation::MAX_ENTITIES_PER_MEMORY);
    }

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
        tags: merged_entities,
        context,
        ner_entities,
        // Robotics fields — flow through from Zenoh publisher to stored Experience
        robot_id: req.robot_id.clone(),
        mission_id: req.mission_id.clone(),
        geo_location: req.geo_location,
        local_position: req.local_position,
        heading: req.heading,
        action_type: req.action_type.clone(),
        reward: req.reward,
        sensor_data: req.sensor_data.clone(),
        decision_context: req.decision_context.clone(),
        action_params: req.action_params.clone(),
        outcome_type: req.outcome_type.clone(),
        outcome_details: req.outcome_details.clone(),
        confidence: req.confidence,
        terrain_type: req.terrain_type.clone(),
        nearby_agents: req.nearby_agents.clone(),
        is_failure: req.is_failure,
        is_anomaly: req.is_anomaly,
        severity: req.severity.clone(),
        ..Default::default()
    };

    let memory = match manager.get_user_memory(&req.user_id) {
        Ok(m) => m,
        Err(e) => {
            error!(user_id = %req.user_id, "Failed to get user memory: {}", e);
            return;
        }
    };

    // Store memory (blocking)
    let exp_clone = experience.clone();
    let memory_clone = memory.clone();
    let agent_id = req.agent_id.clone();
    let run_id = req.run_id.clone();
    let created_at = req.created_at;
    let created_at_for_facts = created_at.unwrap_or_else(chrono::Utc::now);

    let memory_id = match tokio::task::spawn_blocking(move || {
        let guard = memory_clone.read();
        if agent_id.is_some() || run_id.is_some() {
            guard.remember_with_agent(exp_clone, created_at, agent_id, run_id)
        } else {
            guard.remember(exp_clone, created_at)
        }
    })
    .await
    {
        Ok(Ok(id)) => id,
        Ok(Err(e)) => {
            error!(user_id = %req.user_id, "Failed to store memory: {}", e);
            return;
        }
        Err(e) => {
            error!(user_id = %req.user_id, "Memory storage task panicked: {:?}", e);
            return;
        }
    };

    // Tool-aware feedback attribution for robotics:
    // If this remember() carries an action+reward, attribute it to the most recent recall.
    // Pattern: robot recalls memories → acts → remembers outcome with action_type+reward.
    // We take the PendingFeedback from the prior recall, attach the action as a ToolAction,
    // then run the same signal processing + momentum update pipeline as HTTP proactive_context.
    if req.action_type.is_some() && req.reward.is_some() {
        let feedback_store = manager.feedback_store().clone();
        let user_id_for_attr = req.user_id.clone();
        let action_type = req.action_type.clone().unwrap_or_default();
        let action_params = req.action_params.clone().unwrap_or_default();
        let outcome_type = req.outcome_type.clone().unwrap_or_default();
        let outcome_details = req.outcome_details.clone();
        let reward = req.reward;

        tokio::spawn(async move {
            // Build ToolAction from robotics experience fields
            let tool_action = ToolAction {
                tool_name: action_type,
                inputs: action_params,
                success: matches!(outcome_type.as_str(), "success" | "partial"),
                output_snippet: outcome_details.map(|s| s.chars().take(200).collect()),
                reward,
            };

            // Phase 1: Take pending feedback under brief write lock
            let pending = {
                let mut store = feedback_store.write();
                store.take_pending(&user_id_for_attr).map(|mut p| {
                    p.tool_actions = vec![tool_action];
                    p
                })
            };

            if let Some(pending) = pending {
                // Phase 2: Compute signals — no lock held
                // For robotics, there's no "response text" — the tool action IS the response.
                // Pass empty response so entity/semantic signals are minimal; tool signal dominates.
                let signals =
                    feedback::process_implicit_feedback_with_semantics(&pending, "", None, None);

                if !signals.is_empty() {
                    // Phase 3: Apply momentum updates under brief write lock
                    let mut store = feedback_store.write();
                    for (memory_id, signal) in &signals {
                        let momentum = store.get_or_create_momentum(
                            memory_id.clone(),
                            crate::memory::types::ExperienceType::Context,
                        );
                        momentum.update(signal.clone());
                        store.mark_dirty(memory_id);
                    }
                    if let Err(e) = store.flush() {
                        debug!(
                            "Failed to flush feedback store after robot attribution: {}",
                            e
                        );
                    }

                    debug!(
                        user_id = %user_id_for_attr,
                        signals = signals.len(),
                        "Robotics tool-aware feedback attribution completed"
                    );
                }
            }
        });
    }

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    let experience_type_str = format!("{:?}", experience.experience_type);

    // Session tracking
    let session_id = manager.session_store().get_or_create_session(&req.user_id);
    manager.session_store().add_event(
        &session_id,
        SessionEvent::MemoryCreated {
            timestamp: chrono::Utc::now(),
            memory_id: memory_id.0.to_string(),
            memory_type: experience_type_str.clone(),
            content_preview: req.content.chars().take(100).collect(),
            entities: req.tags.clone(),
        },
    );

    // Broadcast event (for SSE/WebSocket dashboards)
    manager.emit_event(MemoryEvent {
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

    // Save ID string for logging before moving into async block
    let response_id = memory_id.0.to_string();

    // Fire-and-forget post-processing (graph + lineage + facts)
    let state = manager.clone();
    let user_id = req.user_id.clone();
    let parent_id = req.parent_id.clone();
    tokio::spawn(async move {
        if let Err(e) = state.process_experience_into_graph(&user_id, &experience, &memory_id) {
            debug!("Graph processing failed (non-fatal): {}", e);
        }

        if let Some(ref parent_id_str) = parent_id {
            if let Ok(parent_uuid) = uuid::Uuid::parse_str(parent_id_str) {
                let parent = MemoryId(parent_uuid);
                let mem = memory.clone();
                let mid = memory_id.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    let guard = mem.read();
                    guard.set_memory_parent(&mid, Some(parent))
                })
                .await;
            }
        }

        // Temporal fact extraction
        {
            let mem = memory.clone();
            let mid = memory_id.clone();
            let uid = user_id.clone();
            let content = experience.content.clone();
            let entities = experience.entities.clone();
            let created_at = created_at_for_facts;
            let _ = tokio::task::spawn_blocking(move || {
                let guard = mem.read();
                guard.store_temporal_facts_for_memory(&uid, &mid, &content, &entities, created_at)
            })
            .await;
        }
    });

    debug!(
        user_id = %req.user_id,
        memory_id = %response_id,
        duration_ms = duration * 1000.0,
        "Zenoh remember completed"
    );
}

// =============================================================================
// RECALL HANDLER
// =============================================================================

/// Handle a Zenoh queryable GET on `{prefix}/{user_id}/recall`.
///
/// Mirrors `src/handlers/recall.rs::recall()` with full robotics filter support:
/// 1. Deserialize query payload → ZenohRecallRequest (includes spatial, mission, reward filters)
/// 2. Build MemoryQuery with robotics filters, run retrieval
/// 3. Augment with todos, facts, reminders, lineage
/// 4. Reply with RecallResponse JSON
/// 5. Publish results to `{prefix}/{user_id}/recall/results` for robot subscribers
pub async fn handle_recall(query: Query, manager: Arc<MultiUserMemoryManager>) {
    let key = query.key_expr().as_str().to_string();

    // The query payload contains the ZenohRecallRequest
    let req: ZenohRecallRequest = match query.payload() {
        Some(payload) => match deserialize_json(payload) {
            Ok(r) => r,
            Err(e) => {
                warn!(key = %key, "Invalid recall payload: {}", e);
                reply_error(&query, &format!("Invalid payload: {e}")).await;
                return;
            }
        },
        None => {
            warn!(key = %key, "Recall query has no payload");
            reply_error(&query, "Missing payload").await;
            return;
        }
    };

    if let Err(e) = validation::validate_user_id(&req.user_id) {
        reply_error(&query, &format!("Invalid user_id: {e}")).await;
        return;
    }

    let memory = match manager.get_user_memory(&req.user_id) {
        Ok(m) => m,
        Err(e) => {
            reply_error(&query, &format!("Failed to get user memory: {e}")).await;
            return;
        }
    };

    let limit = req.limit;
    let retrieval_mode = parse_retrieval_mode(&req.mode);
    let user_id = req.user_id.clone();
    let query_text = req.query.clone();

    // Build geo_filter from individual lat/lon/radius fields
    let geo_filter = match (req.lat, req.lon, req.radius_meters) {
        (Some(lat), Some(lon), Some(radius)) => Some(GeoFilter::new(lat, lon, radius)),
        _ => None,
    };

    // Build reward range
    let reward_range = match (req.min_reward, req.max_reward) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, 1.0)),
        (None, Some(max)) => Some((-1.0, max)),
        _ => None,
    };

    // Execute recall in blocking task
    let memory_for_recall = memory.clone();
    let user_id_for_recall = user_id.clone();
    let query_for_recall = query_text.clone();
    let robot_id = req.robot_id.clone();
    let mission_id = req.mission_id.clone();
    let action_type = req.action_type.clone();
    let failures_only = req.failures_only;
    let anomalies_only = req.anomalies_only;
    let terrain_type = req.terrain_type.clone();

    let memories = match tokio::task::spawn_blocking(move || {
        let guard = memory_for_recall.read();
        let mq = MemoryQuery {
            user_id: Some(user_id_for_recall),
            query_text: Some(query_for_recall),
            max_results: limit,
            retrieval_mode,
            // Robotics filters — enables spatial, mission, and action-outcome queries
            robot_id,
            mission_id,
            geo_filter,
            action_type,
            reward_range,
            failures_only,
            anomalies_only,
            terrain_type,
            ..Default::default()
        };
        guard.recall(&mq).unwrap_or_default()
    })
    .await
    {
        Ok(memories) => memories,
        Err(e) => {
            error!("Recall task panicked: {:?}", e);
            reply_error(&query, "Internal error during recall").await;
            return;
        }
    };

    // Convert to response format
    let total = memories.len();
    let recall_memories: Vec<RecallMemory> = memories
        .iter()
        .enumerate()
        .map(|(rank, m)| {
            let rank_score = 1.0 - (rank as f32 / total.max(1) as f32);
            let salience = m.salience_score_with_access();
            let score = rank_score * 0.7 + salience * 0.3;
            RecallMemory {
                id: m.id.0.to_string(),
                experience: RecallExperience {
                    content: m.experience.content.clone(),
                    memory_type: Some(format!("{:?}", m.experience.experience_type)),
                    tags: m.experience.entities.clone(),
                },
                importance: m.importance(),
                created_at: m.created_at.to_rfc3339(),
                score,
                tier: format!("{:?}", m.tier),
            }
        })
        .collect();

    // Search related todos
    let todos: Vec<RecallTodo> = {
        let query_for_embed = query_text.clone();
        let memory_for_embed = memory.clone();
        let embedding: Option<Vec<f32>> = tokio::task::spawn_blocking(move || {
            let guard = memory_for_embed.read();
            guard.compute_embedding(&query_for_embed).ok()
        })
        .await
        .ok()
        .flatten();

        if let Some(emb) = embedding {
            manager
                .todo_store
                .search_similar(&user_id, &emb, 5)
                .unwrap_or_default()
                .into_iter()
                .filter(|(t, _)| {
                    matches!(
                        t.status,
                        TodoStatus::Todo | TodoStatus::InProgress | TodoStatus::Blocked
                    )
                })
                .map(|(t, score)| {
                    let project_name = t.project_id.as_ref().and_then(|pid| {
                        manager
                            .todo_store
                            .get_project(&user_id, pid)
                            .ok()
                            .flatten()
                            .map(|p| p.name)
                    });
                    RecallTodo {
                        id: t.id.0.to_string(),
                        short_id: t.short_id(),
                        content: t.content.clone(),
                        status: format!("{:?}", t.status).to_lowercase(),
                        priority: t.priority.indicator().to_string(),
                        project: project_name,
                        due_date: t.due_date.map(|d| d.format("%Y-%m-%d").to_string()),
                        score,
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    };

    // Fetch related facts from entities in recalled memories
    let facts: Vec<RecallFact> = {
        let mut all_entities: HashSet<String> = HashSet::new();
        for mem in &recall_memories {
            for tag in &mem.experience.tags {
                all_entities.insert(tag.to_lowercase());
            }
        }
        for word in query_text.split_whitespace() {
            let clean = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if clean.len() > 2 {
                all_entities.insert(clean);
            }
        }

        let entity_list: Vec<String> = all_entities.into_iter().take(10).collect();
        let memory_for_facts = memory.clone();
        let user_for_facts = user_id.clone();
        tokio::task::spawn_blocking(move || {
            let guard = memory_for_facts.read();
            let mut found = Vec::new();
            for entity in &entity_list {
                if let Ok(entity_facts) = guard.get_facts_by_entity(&user_for_facts, entity, 10) {
                    for fact in entity_facts {
                        found.push(RecallFact {
                            id: fact.id.to_string(),
                            fact: fact.fact.clone(),
                            confidence: fact.confidence,
                            support_count: fact.support_count,
                            related_entities: fact.related_entities.clone(),
                        });
                    }
                }
            }
            // Dedup by fact ID
            found.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            found.dedup_by(|a, b| a.id == b.id);
            found.truncate(10);
            found
        })
        .await
        .unwrap_or_default()
    };

    let todo_count = if todos.is_empty() {
        None
    } else {
        Some(todos.len())
    };
    let fact_count = if facts.is_empty() {
        None
    } else {
        Some(facts.len())
    };

    let has_recall_memories = !recall_memories.is_empty();

    let response = RecallResponse {
        memories: recall_memories,
        count: total,
        retrieval_stats: None,
        todos,
        todo_count,
        facts,
        fact_count,
        triggered_reminders: Vec::new(),
        reminder_count: None,
        lineage: Vec::new(),
        lineage_count: None,
    };

    match serde_json::to_vec(&response) {
        Ok(bytes) => {
            if let Err(e) = query.reply(query.key_expr(), bytes).await {
                error!("Failed to reply to recall query: {}", e);
            }
        }
        Err(e) => {
            error!("Failed to serialize recall response: {}", e);
            reply_error(&query, "Serialization error").await;
        }
    }

    // Store PendingFeedback so subsequent remember() calls with action+reward
    // can attribute outcomes to the memories we just surfaced.
    if has_recall_memories {
        let memory_for_embed = memory.clone();
        let query_for_embed = query_text.clone();
        let user_id_for_fb = user_id.clone();
        let feedback_store = manager.feedback_store().clone();

        // Build surfaced memory info from recall results
        let surfaced_infos: Vec<SurfacedMemoryInfo> = memories
            .iter()
            .map(|m| SurfacedMemoryInfo {
                id: m.id.clone(),
                entities: m.experience.entities.iter().cloned().collect(),
                content_preview: m.experience.content.chars().take(200).collect(),
                score: m.salience_score_with_access(),
                embedding: Vec::new(), // Filled below if embedding succeeds
            })
            .collect();

        // Compute context embedding in background for semantic feedback later
        let context_for_pending = query_for_embed.clone();
        tokio::spawn(async move {
            let embedding = tokio::task::spawn_blocking(move || {
                let guard = memory_for_embed.read();
                guard
                    .compute_embedding(&query_for_embed)
                    .unwrap_or_default()
            })
            .await
            .unwrap_or_default();

            let pending = PendingFeedback::new(
                user_id_for_fb,
                context_for_pending,
                embedding,
                surfaced_infos,
            );
            let mut store = feedback_store.write();
            store.set_pending(pending);
        });
    }

    debug!(
        user_id = %user_id,
        results = total,
        "Zenoh recall completed"
    );
}

// =============================================================================
// FORGET HANDLER
// =============================================================================

/// Handle a Zenoh DELETE/PUT on `{prefix}/{user_id}/forget`.
///
/// Payload: `{ "memory_id": "uuid-string" }` or `{ "id": "uuid-string" }`
pub async fn handle_forget(sample: Sample, manager: Arc<MultiUserMemoryManager>, prefix: &str) {
    #[derive(serde::Deserialize)]
    struct ForgetRequest {
        #[serde(default)]
        user_id: Option<String>,
        #[serde(alias = "id")]
        memory_id: String,
    }

    let key = sample.key_expr().as_str().to_string();

    let req: ForgetRequest = match deserialize_json(sample.payload()) {
        Ok(r) => r,
        Err(e) => {
            warn!(key = %key, "Invalid forget payload: {}", e);
            return;
        }
    };

    // Extract user_id from payload, falling back to key expression: {prefix}/{user_id}/forget
    let user_id = req.user_id.unwrap_or_else(|| {
        extract_user_id(&key, prefix)
            .map(|s| s.to_string())
            .unwrap_or_default()
    });

    if user_id.is_empty() {
        warn!(key = %key, "Cannot determine user_id for forget operation");
        return;
    }

    let memory_uuid = match uuid::Uuid::parse_str(&req.memory_id) {
        Ok(u) => u,
        Err(e) => {
            warn!(memory_id = %req.memory_id, "Invalid memory UUID: {}", e);
            return;
        }
    };

    let memory = match manager.get_user_memory(&user_id) {
        Ok(m) => m,
        Err(e) => {
            error!(user_id = %user_id, "Failed to get user memory: {}", e);
            return;
        }
    };

    let mid = MemoryId(memory_uuid);
    match tokio::task::spawn_blocking(move || {
        let guard = memory.read();
        guard.forget(ForgetCriteria::ById(mid))
    })
    .await
    {
        Ok(Ok(count)) => {
            debug!(user_id = %user_id, memory_id = %req.memory_id, deleted = count, "Zenoh forget completed");
        }
        Ok(Err(e)) => {
            error!(user_id = %user_id, memory_id = %req.memory_id, "Forget failed: {}", e);
        }
        Err(e) => {
            error!("Forget task panicked: {:?}", e);
        }
    }
}

// =============================================================================
// STREAMING HANDLER
// =============================================================================

/// Create a streaming session for an auto-topic subscriber.
///
/// Returns the session_id that subsequent `handle_stream_message` calls use.
pub async fn create_stream_session(
    user_id: &str,
    mode: StreamMode,
    extraction_config: ExtractionConfig,
    tags: Vec<String>,
    manager: &MultiUserMemoryManager,
) -> Result<String, String> {
    let handshake = StreamHandshake {
        user_id: user_id.to_string(),
        mode,
        extraction_config,
        session_id: None,
        metadata: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "transport".to_string(),
                serde_json::Value::String("zenoh".to_string()),
            );
            if !tags.is_empty() {
                m.insert(
                    "auto_tags".to_string(),
                    serde_json::Value::Array(
                        tags.into_iter().map(serde_json::Value::String).collect(),
                    ),
                );
            }
            m
        },
    };

    manager
        .streaming_extractor()
        .create_session(handshake)
        .await
}

/// Handle an incoming stream message from a Zenoh subscriber.
///
/// For auto-topics with `passthrough` mode, the caller wraps the raw payload
/// into a `StreamMessage::Content` before calling this.
pub async fn handle_stream_message(
    message: StreamMessage,
    session_id: &str,
    memory: Arc<parking_lot::RwLock<MemorySystem>>,
    manager: &MultiUserMemoryManager,
) {
    let result = manager
        .streaming_extractor()
        .process_message(session_id, message, memory)
        .await;

    match result {
        crate::streaming::ExtractionResult::Extraction {
            memories_created,
            entities_detected,
            ..
        } => {
            if memories_created > 0 {
                debug!(
                    session_id = %session_id,
                    memories_created,
                    entities = entities_detected.len(),
                    "Stream extraction completed"
                );
            }
        }
        crate::streaming::ExtractionResult::Error { message, .. } => {
            warn!(session_id = %session_id, "Stream extraction error: {}", message);
        }
        _ => {} // ContextInjection, Closed — logged internally
    }
}

/// Wrap a raw Zenoh payload string into a `StreamMessage::Content`.
///
/// Used for `passthrough` auto-topics where the payload is not a shodh StreamMessage.
pub fn wrap_passthrough(payload: &str, tags: &[String]) -> StreamMessage {
    StreamMessage::Content {
        content: payload.to_string(),
        source: Some("zenoh".to_string()),
        timestamp: Some(chrono::Utc::now()),
        importance: None,
        tags: tags.to_vec(),
        metadata: std::collections::HashMap::new(),
    }
}

// =============================================================================
// HEALTH HANDLER
// =============================================================================

/// Build a health check response for the Zenoh queryable.
pub fn build_health_response() -> serde_json::Value {
    serde_json::json!({
        "status": "ok",
        "transport": "zenoh",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })
}

// =============================================================================
// MISSION LIFECYCLE HANDLER
// =============================================================================

/// Handle a Zenoh PUT on `{prefix}/{user_id}/mission/start`.
///
/// Marks the beginning of a named mission in the session store.
/// All subsequent memories from this user_id are associated with the mission.
///
/// Payload: `{ "mission_id": "inspection_2024", "metadata": {...} }`
pub async fn handle_mission_start(sample: Sample, manager: Arc<MultiUserMemoryManager>) {
    #[derive(serde::Deserialize)]
    struct MissionStartRequest {
        mission_id: String,
        #[serde(default)]
        robot_id: Option<String>,
        #[serde(default)]
        description: Option<String>,
    }

    let key = sample.key_expr().as_str().to_string();

    let req: MissionStartRequest = match deserialize_json(sample.payload()) {
        Ok(r) => r,
        Err(e) => {
            warn!(key = %key, "Invalid mission start payload: {}", e);
            return;
        }
    };

    let prefix = key.split('/').next().unwrap_or("shodh");
    let user_id = match extract_user_id(&key, prefix) {
        Some(uid) => uid.to_string(),
        None => {
            warn!(key = %key, "Cannot extract user_id from mission start key");
            return;
        }
    };

    // Record mission start as a session event
    let session_id = manager.session_store().get_or_create_session(&user_id);
    manager.session_store().add_event(
        &session_id,
        SessionEvent::MemoryCreated {
            timestamp: chrono::Utc::now(),
            memory_id: req.mission_id.clone(),
            memory_type: "mission_start".to_string(),
            content_preview: req
                .description
                .clone()
                .unwrap_or_else(|| format!("Mission {} started", req.mission_id)),
            entities: vec![req.mission_id.clone()],
        },
    );

    // Store mission start as a memory so it's searchable
    let content = match &req.description {
        Some(desc) => format!("Mission started: {} — {}", req.mission_id, desc),
        None => format!("Mission started: {}", req.mission_id),
    };

    let memory = match manager.get_user_memory(&user_id) {
        Ok(m) => m,
        Err(e) => {
            error!(user_id = %user_id, "Failed to get user memory: {}", e);
            return;
        }
    };

    let mission_id = req.mission_id.clone();
    let robot_id = req.robot_id.clone();
    let _ = tokio::task::spawn_blocking(move || {
        let guard = memory.read();
        let experience = Experience {
            content,
            mission_id: Some(mission_id),
            robot_id,
            tags: vec!["mission".to_string(), "mission_start".to_string()],
            entities: vec!["mission".to_string(), "mission_start".to_string()],
            ..Default::default()
        };
        guard.remember(experience, None)
    })
    .await;

    // Broadcast mission event for SSE/WebSocket dashboards
    manager.emit_event(MemoryEvent {
        event_type: "MISSION_START".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: Some(req.mission_id.clone()),
        content_preview: req.description,
        memory_type: Some("mission".to_string()),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    info!(
        user_id = %user_id,
        mission_id = %req.mission_id,
        robot_id = ?req.robot_id,
        "Mission started via Zenoh"
    );
}

/// Handle a Zenoh PUT on `{prefix}/{user_id}/mission/end`.
///
/// Marks the end of a mission. Stores a summary memory and closes the mission context.
///
/// Payload: `{ "mission_id": "inspection_2024", "summary": "...", "outcome": "success" }`
pub async fn handle_mission_end(sample: Sample, manager: Arc<MultiUserMemoryManager>) {
    #[derive(serde::Deserialize)]
    struct MissionEndRequest {
        mission_id: String,
        #[serde(default)]
        robot_id: Option<String>,
        #[serde(default)]
        summary: Option<String>,
        #[serde(default)]
        outcome: Option<String>,
        #[serde(default)]
        reward: Option<f32>,
    }

    let key = sample.key_expr().as_str().to_string();

    let req: MissionEndRequest = match deserialize_json(sample.payload()) {
        Ok(r) => r,
        Err(e) => {
            warn!(key = %key, "Invalid mission end payload: {}", e);
            return;
        }
    };

    let prefix = key.split('/').next().unwrap_or("shodh");
    let user_id = match extract_user_id(&key, prefix) {
        Some(uid) => uid.to_string(),
        None => {
            warn!(key = %key, "Cannot extract user_id from mission end key");
            return;
        }
    };

    let content = match &req.summary {
        Some(summary) => format!(
            "Mission ended: {} — {} (outcome: {})",
            req.mission_id,
            summary,
            req.outcome.as_deref().unwrap_or("unknown")
        ),
        None => format!(
            "Mission ended: {} (outcome: {})",
            req.mission_id,
            req.outcome.as_deref().unwrap_or("unknown")
        ),
    };

    let memory = match manager.get_user_memory(&user_id) {
        Ok(m) => m,
        Err(e) => {
            error!(user_id = %user_id, "Failed to get user memory: {}", e);
            return;
        }
    };

    let mission_id = req.mission_id.clone();
    let robot_id = req.robot_id.clone();
    let outcome_type = req.outcome.clone();
    let reward = req.reward;
    let _ = tokio::task::spawn_blocking(move || {
        let guard = memory.read();
        let experience = Experience {
            content,
            mission_id: Some(mission_id),
            robot_id,
            outcome_type,
            reward,
            tags: vec!["mission".to_string(), "mission_end".to_string()],
            entities: vec!["mission".to_string(), "mission_end".to_string()],
            ..Default::default()
        };
        guard.remember(experience, None)
    })
    .await;

    manager.emit_event(MemoryEvent {
        event_type: "MISSION_END".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: Some(req.mission_id.clone()),
        content_preview: req.summary,
        memory_type: Some("mission".to_string()),
        importance: None,
        count: None,
        entities: None,
        results: None,
    });

    info!(
        user_id = %user_id,
        mission_id = %req.mission_id,
        outcome = ?req.outcome,
        "Mission ended via Zenoh"
    );
}

// =============================================================================
// FLEET DISCOVERY HANDLER
// =============================================================================

/// Build a fleet status response listing all known robots.
///
/// Called by the fleet queryable (`{prefix}/fleet`). Returns the list of
/// currently tracked liveliness tokens and their metadata.
pub fn build_fleet_response(
    peers: &std::collections::HashMap<String, FleetPeer>,
) -> serde_json::Value {
    let peer_list: Vec<serde_json::Value> = peers
        .iter()
        .map(|(id, peer)| {
            serde_json::json!({
                "robot_id": id,
                "key_expr": peer.key_expr,
                "joined_at": peer.joined_at.to_rfc3339(),
                "status": "online",
            })
        })
        .collect();

    serde_json::json!({
        "fleet_size": peer_list.len(),
        "peers": peer_list,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })
}

/// Metadata for a discovered fleet peer.
#[derive(Debug, Clone)]
pub struct FleetPeer {
    pub key_expr: String,
    pub joined_at: chrono::DateTime<chrono::Utc>,
}

// =============================================================================
// HELPERS
// =============================================================================

/// Map API mode string to RetrievalMode enum.
///
/// Supports all modes including robotics-specific retrieval:
/// - `spatial`: Geo-location based (haversine distance sorting)
/// - `mission`: Filter by mission_id context (chronological order)
/// - `action_outcome`: Reward-based RL learning (sorted by reward)
fn parse_retrieval_mode(mode: &str) -> RetrievalMode {
    match mode {
        "semantic" | "similarity" => RetrievalMode::Similarity,
        "associative" => RetrievalMode::Associative,
        "temporal" => RetrievalMode::Temporal,
        "causal" => RetrievalMode::Causal,
        "spatial" => RetrievalMode::Spatial,
        "mission" => RetrievalMode::Mission,
        "action_outcome" => RetrievalMode::ActionOutcome,
        _ => RetrievalMode::Hybrid,
    }
}

/// Reply to a Zenoh query with an error JSON payload.
async fn reply_error(query: &Query, message: &str) {
    let error_json = serde_json::json!({
        "error": message,
        "success": false,
    });
    if let Ok(bytes) = serde_json::to_vec(&error_json) {
        if let Err(e) = query.reply(query.key_expr(), bytes).await {
            error!("Failed to send error reply: {}", e);
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_user_id_valid() {
        assert_eq!(
            extract_user_id("shodh/robot-1/remember", "shodh"),
            Some("robot-1")
        );
        assert_eq!(
            extract_user_id("shodh/user@example.com/recall", "shodh"),
            Some("user@example.com")
        );
        assert_eq!(
            extract_user_id("custom/drone_alpha/stream/sensor", "custom"),
            Some("drone_alpha")
        );
    }

    #[test]
    fn test_extract_user_id_invalid() {
        // No user_id segment
        assert_eq!(extract_user_id("shodh/remember", "shodh"), None);
        // Wrong prefix
        assert_eq!(extract_user_id("other/robot-1/remember", "shodh"), None);
        // Empty user_id
        assert_eq!(extract_user_id("shodh//remember", "shodh"), None);
        // No trailing operation
        assert_eq!(extract_user_id("shodh/robot-1", "shodh"), None);
    }

    #[test]
    fn test_extract_user_id_nested_prefix() {
        assert_eq!(
            extract_user_id("my/prefix/robot-1/remember", "my/prefix"),
            Some("robot-1")
        );
    }

    #[test]
    fn test_wrap_passthrough() {
        let msg = wrap_passthrough(r#"{"temp": 42.5}"#, &["sensor".to_string()]);
        match msg {
            StreamMessage::Content {
                content,
                source,
                tags,
                ..
            } => {
                assert_eq!(content, r#"{"temp": 42.5}"#);
                assert_eq!(source, Some("zenoh".to_string()));
                assert_eq!(tags, vec!["sensor"]);
            }
            _ => panic!("Expected Content variant"),
        }
    }

    #[test]
    fn test_build_health_response() {
        let resp = build_health_response();
        assert_eq!(resp["status"], "ok");
        assert_eq!(resp["transport"], "zenoh");
    }

    #[test]
    fn test_parse_retrieval_mode() {
        assert!(matches!(
            parse_retrieval_mode("semantic"),
            RetrievalMode::Similarity
        ));
        assert!(matches!(
            parse_retrieval_mode("associative"),
            RetrievalMode::Associative
        ));
        assert!(matches!(
            parse_retrieval_mode("temporal"),
            RetrievalMode::Temporal
        ));
        assert!(matches!(
            parse_retrieval_mode("hybrid"),
            RetrievalMode::Hybrid
        ));
        assert!(matches!(
            parse_retrieval_mode("unknown"),
            RetrievalMode::Hybrid
        ));
        // Robotics modes
        assert!(matches!(
            parse_retrieval_mode("spatial"),
            RetrievalMode::Spatial
        ));
        assert!(matches!(
            parse_retrieval_mode("mission"),
            RetrievalMode::Mission
        ));
        assert!(matches!(
            parse_retrieval_mode("action_outcome"),
            RetrievalMode::ActionOutcome
        ));
    }

    #[test]
    fn test_zenoh_remember_request_deserialization() {
        let json = r#"{
            "user_id": "spot-1",
            "content": "Detected obstacle at waypoint alpha",
            "robot_id": "spot_v2",
            "mission_id": "inspection_2024",
            "geo_location": [37.7749, -122.4194, 10.0],
            "local_position": [1.5, 2.3, 0.0],
            "heading": 90.0,
            "sensor_data": {"battery": 75.5, "temperature": 28.3},
            "action_type": "navigate",
            "reward": 0.8,
            "terrain_type": "indoor",
            "is_failure": false,
            "tags": ["obstacle", "navigation"]
        }"#;
        let req: ZenohRememberRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.user_id, "spot-1");
        assert_eq!(req.robot_id, Some("spot_v2".to_string()));
        assert_eq!(req.mission_id, Some("inspection_2024".to_string()));
        assert_eq!(req.geo_location, Some([37.7749, -122.4194, 10.0]));
        assert_eq!(req.local_position, Some([1.5, 2.3, 0.0]));
        assert_eq!(req.heading, Some(90.0));
        assert_eq!(*req.sensor_data.get("battery").unwrap(), 75.5);
        assert_eq!(req.action_type, Some("navigate".to_string()));
        assert_eq!(req.reward, Some(0.8));
        assert_eq!(req.terrain_type, Some("indoor".to_string()));
        assert!(!req.is_failure);
    }

    #[test]
    fn test_zenoh_recall_request_with_spatial_filter() {
        let json = r#"{
            "user_id": "spot-1",
            "query": "obstacles near entrance",
            "mode": "spatial",
            "lat": 37.7749,
            "lon": -122.4194,
            "radius_meters": 50.0,
            "mission_id": "inspection_2024",
            "robot_id": "spot_v2"
        }"#;
        let req: ZenohRecallRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.mode, "spatial");
        assert_eq!(req.lat, Some(37.7749));
        assert_eq!(req.lon, Some(-122.4194));
        assert_eq!(req.radius_meters, Some(50.0));
        assert_eq!(req.mission_id, Some("inspection_2024".to_string()));
    }

    #[test]
    fn test_zenoh_recall_request_with_reward_filter() {
        let json = r#"{
            "user_id": "drone-1",
            "query": "successful navigation actions",
            "mode": "action_outcome",
            "min_reward": 0.5,
            "action_type": "navigate"
        }"#;
        let req: ZenohRecallRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.mode, "action_outcome");
        assert_eq!(req.min_reward, Some(0.5));
        assert_eq!(req.action_type, Some("navigate".to_string()));
    }

    #[test]
    fn test_build_fleet_response() {
        let mut peers = std::collections::HashMap::new();
        peers.insert(
            "spot-1".to_string(),
            FleetPeer {
                key_expr: "shodh/fleet/spot-1".to_string(),
                joined_at: chrono::Utc::now(),
            },
        );
        let resp = build_fleet_response(&peers);
        assert_eq!(resp["fleet_size"], 1);
        assert_eq!(resp["peers"][0]["robot_id"], "spot-1");
        assert_eq!(resp["peers"][0]["status"], "online");
    }

    #[test]
    fn test_zenoh_remember_minimal_request() {
        // Verify that a minimal request (no robotics fields) still deserializes
        let json = r#"{"user_id": "agent-1", "content": "Hello world"}"#;
        let req: ZenohRememberRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.user_id, "agent-1");
        assert!(req.robot_id.is_none());
        assert!(req.geo_location.is_none());
        assert!(req.sensor_data.is_empty());
    }
}
