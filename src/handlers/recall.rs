//! Recall and Retrieval Handlers
//!
//! Memory retrieval endpoints including:
//! - Semantic recall (vector similarity)
//! - Hybrid recall (semantic + graph)
//! - Proactive context surfacing
//! - Tag-based and date-based recall
//! - Tracked retrieval with Hebbian feedback

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use tracing::info;

use super::state::MultiUserMemoryManager;
use super::types::{
    MemoryEvent, RecallExperience, RecallFact, RecallMemory, RecallRequest, RecallResponse,
    RecallTodo, ReinforceFeedbackRequest, RetrieveResponse, TrackedRetrieveRequest,
    TrackedRetrieveResponse,
};
use super::utils::{is_bare_question, is_boilerplate_response, strip_system_noise};
use crate::errors::{AppError, ValidationErrorExt};
use crate::memory::feedback;
// Note: compute_relevance removed - using unified 5-layer pipeline scoring instead
use crate::memory::segmentation::{InputSource, SegmentationEngine};
use crate::memory::sessions::SessionEvent;
use crate::memory::storage::SearchCriteria;
use crate::memory::types::MemoryId;
use crate::memory::{Experience, ExperienceType, Query as MemoryQuery, SharedMemory};
use crate::memory::{ProspectiveTrigger, TodoStatus};
use crate::metrics;
use crate::relevance;
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// CONTEXT SUMMARY TYPES
// =============================================================================

/// Context summary request
#[derive(Debug, Deserialize)]
pub struct ContextSummaryRequest {
    pub user_id: String,
    #[serde(default = "default_true")]
    pub include_decisions: bool,
    #[serde(default = "default_true")]
    pub include_learnings: bool,
    #[serde(default = "default_true")]
    pub include_context: bool,
    #[serde(default = "default_max_items")]
    pub max_items: usize,
}

fn default_true() -> bool {
    true
}

fn default_max_items() -> usize {
    5
}

/// Summary item - simplified memory for context
#[derive(Debug, Serialize)]
pub struct SummaryItem {
    pub id: String,
    pub content: String,
    pub importance: f32,
    pub created_at: String,
}

/// Context summary response - categorized memories for session bootstrap
#[derive(Debug, Serialize)]
pub struct ContextSummaryResponse {
    pub total_memories: usize,
    pub decisions: Vec<SummaryItem>,
    pub learnings: Vec<SummaryItem>,
    pub context: Vec<SummaryItem>,
    pub patterns: Vec<SummaryItem>,
    pub errors: Vec<SummaryItem>,
}

// =============================================================================
// PROACTIVE CONTEXT TYPES
// =============================================================================

/// Request for proactive context - returns relevant memories + triggered reminders
#[derive(Debug, Deserialize)]
pub struct ProactiveContextRequest {
    pub user_id: String,
    pub context: String,
    #[serde(default = "default_proactive_max_results")]
    pub max_results: usize,
    /// Minimum semantic similarity threshold (0.0-1.0)
    #[serde(default = "default_semantic_threshold")]
    pub semantic_threshold: f32,
    /// Weight for entity matching in relevance scoring
    #[serde(default = "default_entity_weight")]
    pub entity_match_weight: f32,
    /// Weight for recency boost
    #[serde(default = "default_recency_weight")]
    pub recency_weight: f32,
    /// Filter to specific memory types
    #[serde(default)]
    pub memory_types: Vec<String>,
    /// Whether to auto-ingest the context as a Conversation memory
    #[serde(default = "default_true")]
    pub auto_ingest: bool,
    /// Agent's previous response (for implicit feedback extraction)
    #[serde(default)]
    pub previous_response: Option<String>,
    /// User's followup message after agent response (for delayed signals)
    #[serde(default)]
    pub user_followup: Option<String>,
}

fn default_proactive_max_results() -> usize {
    5
}

fn default_semantic_threshold() -> f32 {
    0.45 // Lowered from 0.65 - composite relevance scores blend multiple signals
}

fn default_entity_weight() -> f32 {
    0.4
}

fn default_recency_weight() -> f32 {
    0.2
}

/// Feedback processing results
#[derive(Debug, Serialize)]
pub struct FeedbackProcessed {
    pub memories_evaluated: usize,
    pub reinforced: Vec<String>,
    pub weakened: Vec<String>,
}

/// Surfaced memory in proactive context response
#[derive(Debug, Serialize)]
pub struct ProactiveSurfacedMemory {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub score: f32,
    pub created_at: String,
    pub tags: Vec<String>,
    /// Embedding for semantic feedback (not serialized to response)
    #[serde(skip)]
    pub embedding: Vec<f32>,
}

/// Todo item in proactive context response
#[derive(Debug, Serialize)]
pub struct ProactiveTodoItem {
    pub id: String,
    pub short_id: String,
    pub content: String,
    pub status: String,
    pub priority: String,
    pub project: Option<String>,
    pub due_date: Option<String>,
    pub relevance_reason: String,
    /// Semantic similarity score (0.0 - 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_score: Option<f32>,
}

/// Individual reminder in response
#[derive(Debug, Serialize)]
pub struct ReminderItem {
    pub id: String,
    pub content: String,
    pub trigger_type: String,
    pub status: String,
    pub due_at: Option<chrono::DateTime<chrono::Utc>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub triggered_at: Option<chrono::DateTime<chrono::Utc>>,
    pub dismissed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub priority: u8,
    pub tags: Vec<String>,
    pub overdue_seconds: Option<i64>,
}

/// Response for proactive context
#[derive(Debug, Serialize)]
pub struct ProactiveContextResponse {
    /// Relevant memories based on context
    pub memories: Vec<ProactiveSurfacedMemory>,
    /// Due time-based reminders
    pub due_reminders: Vec<ReminderItem>,
    /// Context-triggered reminders (keyword match)
    pub context_reminders: Vec<ReminderItem>,
    /// Total counts
    pub memory_count: usize,
    pub reminder_count: usize,
    /// ID of auto-ingested memory (if auto_ingest=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ingested_memory_id: Option<String>,
    /// Feedback processing results (if previous_response was provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback_processed: Option<FeedbackProcessed>,
    /// Relevant todos based on context
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant_todos: Vec<ProactiveTodoItem>,
    /// Todo count
    #[serde(default)]
    pub todo_count: usize,
}

// =============================================================================
// REINFORCE FEEDBACK TYPES
// =============================================================================

/// Response from reinforcement
#[derive(Debug, Serialize)]
pub struct ReinforceFeedbackResponse {
    pub memories_processed: usize,
    pub associations_strengthened: usize,
    pub importance_boosts: usize,
    pub importance_decays: usize,
}

// =============================================================================
// RECALL BY TAGS/DATE TYPES (local - not in shared types.rs)
// =============================================================================

/// Recall memories by tags
#[derive(Debug, Deserialize)]
pub struct RecallByTagsRequest {
    pub user_id: String,
    /// Tags to search for (returns memories matching ANY of these tags)
    pub tags: Vec<String>,
    /// Maximum number of results (default: 50)
    pub limit: Option<usize>,
}

/// Recall memories by date range
#[derive(Debug, Deserialize)]
pub struct RecallByDateRequest {
    pub user_id: String,
    /// Start of date range (inclusive) - ISO 8601 format
    pub start: chrono::DateTime<chrono::Utc>,
    /// End of date range (inclusive) - ISO 8601 format
    pub end: chrono::DateTime<chrono::Utc>,
    /// Maximum number of results (default: 50)
    pub limit: Option<usize>,
}

// =============================================================================
// MAIN RECALL HANDLER
// =============================================================================

/// POST /api/recall - Semantic + associative hybrid recall
///
/// Uses a hybrid retrieval strategy:
/// 1. Semantic search via vector similarity
/// 2. Graph traversal via spreading activation
/// 3. Hebbian boosting for frequently co-retrieved memories
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
pub async fn recall(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, AppError> {
    let op_start = std::time::Instant::now();
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let _graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let query_text = req.query.clone();
    let limit = req.limit;
    let mode = req.mode.clone();

    // PROSPECTIVE MEMORY: Check for context-triggered reminders that match the query
    // This enables "future informs present" - pending reminders influence what memories are surfaced
    let (triggered_reminders, prospective_signals) = {
        let prospective = state.prospective_store.clone();
        let user_id = req.user_id.clone();
        let context = req.query.clone();

        // Check for keyword-matched context triggers
        let matched_tasks = prospective
            .check_context_triggers(&user_id, &context)
            .unwrap_or_default();

        // Build triggered_reminders for response and prospective_signals for boosting
        let mut signals: Vec<String> = Vec::new();
        let reminders: Vec<super::types::RecallReminder> = matched_tasks
            .into_iter()
            .map(|task| {
                // Extract keywords from trigger for signals
                let keywords = if let ProspectiveTrigger::OnContext { keywords, .. } = &task.trigger
                {
                    keywords.clone()
                } else {
                    vec![]
                };

                // Add task content and keywords to prospective signals
                signals.push(task.content.clone());
                for kw in &keywords {
                    signals.push(kw.clone());
                }

                super::types::RecallReminder {
                    id: task.id.0.to_string(),
                    content: task.content,
                    keywords,
                    match_type: "keyword_match".to_string(),
                    priority: task.priority,
                    created_at: task.created_at.to_rfc3339(),
                }
            })
            .collect();

        (
            reminders,
            if signals.is_empty() {
                None
            } else {
                Some(signals)
            },
        )
    };

    let triggered_reminder_count = triggered_reminders.len();
    if triggered_reminder_count > 0 {
        tracing::debug!(
            user_id = %req.user_id,
            count = triggered_reminder_count,
            "Context-triggered reminders found - future intentions will boost related memories"
        );
    }

    // Execute recall - the recall() method already does hybrid retrieval
    // (semantic + graph traversal + Hebbian learning)
    // Now also passes prospective_signals to enable future-informed retrieval
    let memories = {
        let memory = memory.clone();
        let signals = prospective_signals.clone();
        let user_id = req.user_id.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();

            // Build query with prospective signals and user_id for temporal fact lookup
            let query = MemoryQuery {
                user_id: Some(user_id),
                query_text: Some(query_text.clone()),
                max_results: limit,
                prospective_signals: signals,
                ..Default::default()
            };

            // recall() internally:
            // 1. Generates query embedding
            // 2. Searches vector index for semantic similarity
            // 3. Traverses knowledge graph for entity connections
            // 4. Applies prospective boost for future intention matches
            // 5. Records coactivation for Hebbian learning
            // 6. Returns ranked results
            memory_guard.recall(&query).unwrap_or_default()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // Convert to response format
    let total = memories.len();
    let recall_memories: Vec<RecallMemory> = memories
        .iter()
        .enumerate()
        .map(|(rank, m)| {
            // Score based on rank position and salience
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
            }
        })
        .collect();

    // Search todos semantically if query provided
    let todos: Vec<RecallTodo> = {
        // Compute embedding for todo search
        let query_for_embed = req.query.clone();
        let memory_for_embed = memory.clone();
        let embedding: Option<Vec<f32>> = tokio::task::spawn_blocking(move || {
            let guard = memory_for_embed.read();
            guard.compute_embedding(&query_for_embed).ok()
        })
        .await
        .ok()
        .flatten();

        if let Some(emb) = embedding {
            state
                .todo_store
                .search_similar(&req.user_id, &emb, 5)
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
                        state
                            .todo_store
                            .get_project(&req.user_id, pid)
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
    let todo_count = if todos.is_empty() {
        None
    } else {
        Some(todos.len())
    };

    // Fetch related semantic facts from entities mentioned in recalled memories
    let facts: Vec<RecallFact> = {
        // Collect all unique entities from recalled memories
        let mut all_entities: std::collections::HashSet<String> = std::collections::HashSet::new();
        for mem in &recall_memories {
            for tag in &mem.experience.tags {
                all_entities.insert(tag.to_lowercase());
            }
        }

        // Also extract simple entities from the query itself
        for word in req.query.split_whitespace() {
            let clean_word = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if clean_word.len() > 2 {
                all_entities.insert(clean_word);
            }
        }

        // Query fact store for facts related to these entities
        let mut found_facts = Vec::new();
        for entity in all_entities.iter().take(10) {
            if let Ok(entity_facts) = state.fact_store.find_by_entity(&req.user_id, entity, 5) {
                for fact in entity_facts {
                    found_facts.push(RecallFact {
                        id: fact.id.clone(),
                        fact: fact.fact.clone(),
                        confidence: fact.confidence,
                        support_count: fact.support_count,
                        related_entities: fact.related_entities.clone(),
                    });
                }
            }
        }

        // Deduplicate by fact ID and take top 5 by confidence
        let mut unique_facts: std::collections::HashMap<String, RecallFact> =
            std::collections::HashMap::new();
        for fact in found_facts {
            unique_facts.entry(fact.id.clone()).or_insert(fact);
        }
        let mut sorted_facts: Vec<RecallFact> = unique_facts.into_values().collect();
        sorted_facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_facts.truncate(5);
        sorted_facts
    };
    let fact_count = if facts.is_empty() {
        None
    } else {
        Some(facts.len())
    };

    let count = recall_memories.len();

    // Note: Coactivation for Hebbian learning is already recorded inside recall()
    // No need for explicit record_memory_coactivation call

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_RETRIEVE_DURATION
        .with_label_values(&[&mode])
        .observe(duration);
    metrics::MEMORY_RETRIEVE_TOTAL
        .with_label_values(&[&mode, &"success".to_string()])
        .inc();
    metrics::MEMORY_RETRIEVE_RESULTS
        .with_label_values(&[&mode])
        .observe(count as f64);

    // Broadcast RETRIEVE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "RETRIEVE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(req.query.chars().take(50).collect()),
        memory_type: Some(mode),
        importance: None,
        count: Some(count),
    });

    // Track session event
    if count > 0 {
        let session_id = state.session_store.get_or_create_session(&req.user_id);
        let memory_ids: Vec<String> = recall_memories.iter().map(|m| m.id.clone()).collect();
        let avg_score = if !recall_memories.is_empty() {
            recall_memories.iter().map(|m| m.score).sum::<f32>() / recall_memories.len() as f32
        } else {
            0.0
        };
        state.session_store.add_event(
            &session_id,
            SessionEvent::MemoriesSurfaced {
                timestamp: chrono::Utc::now(),
                query_preview: req.query.chars().take(100).collect(),
                memory_count: count,
                memory_ids,
                avg_score,
            },
        );
    }

    // Build reminder count for response
    let reminder_count = if triggered_reminders.is_empty() {
        None
    } else {
        Some(triggered_reminders.len())
    };

    Ok(Json(RecallResponse {
        memories: recall_memories,
        count,
        retrieval_stats: None, // Retrieval stats not exposed in new API
        todos,
        todo_count,
        facts,
        fact_count,
        triggered_reminders,
        reminder_count,
    }))
}

// =============================================================================
// CONTEXT SUMMARY HANDLER
// =============================================================================

/// POST /api/context_summary - Get categorized memories for session bootstrap
///
/// Returns memories grouped by type (decisions, learnings, context, patterns, errors)
/// for quick session initialization.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn context_summary(
    State(state): State<AppState>,
    Json(req): Json<ContextSummaryRequest>,
) -> Result<Json<ContextSummaryResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let max_items = req.max_items;
    let include_decisions = req.include_decisions;
    let include_learnings = req.include_learnings;
    let include_context = req.include_context;

    let response = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();

            let stats = memory_guard.stats();
            let total_memories = stats.total_memories;

            // Helper to search by type using advanced_search
            let search_by_type = |exp_type: ExperienceType, limit: usize| -> Vec<SummaryItem> {
                memory_guard
                    .advanced_search(SearchCriteria::ByType(exp_type))
                    .unwrap_or_default()
                    .into_iter()
                    .take(limit)
                    .map(|m| SummaryItem {
                        id: m.id.0.to_string(),
                        content: m.experience.content.clone(),
                        importance: m.importance(),
                        created_at: m.created_at.to_rfc3339(),
                    })
                    .collect()
            };

            // Get memories by type using advanced_search
            let decisions = if include_decisions {
                search_by_type(ExperienceType::Decision, max_items)
            } else {
                Vec::new()
            };

            let learnings = if include_learnings {
                search_by_type(ExperienceType::Learning, max_items)
            } else {
                Vec::new()
            };

            let context = if include_context {
                search_by_type(ExperienceType::Context, max_items)
            } else {
                Vec::new()
            };

            let patterns = search_by_type(ExperienceType::Pattern, max_items);
            let errors = search_by_type(ExperienceType::Error, max_items);

            ContextSummaryResponse {
                total_memories,
                decisions,
                learnings,
                context,
                patterns,
                errors,
            }
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    Ok(Json(response))
}

// =============================================================================
// PROACTIVE CONTEXT HANDLER
// =============================================================================

/// POST /api/proactive_context - Combined recall + reminders for AI agents
///
/// Returns relevant memories based on semantic similarity and entity matching,
/// plus any due or context-triggered reminders. Optionally stores the context
/// as a Conversation memory for future recall.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn proactive_context(
    State(state): State<AppState>,
    Json(req): Json<ProactiveContextRequest>,
) -> Result<Json<ProactiveContextResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    // 0. Process pending feedback if previous_response is provided
    let feedback_processed = if let Some(ref prev_response) = req.previous_response {
        let feedback_store = state.feedback_store.clone();
        let user_id_for_feedback = req.user_id.clone();
        let response_text = prev_response.clone();
        let followup = req.user_followup.clone();
        let memory_for_embed = memory_system.clone();

        // Process feedback and collect memory IDs for reinforcement
        let (result, helpful_ids, misleading_ids) = tokio::task::spawn_blocking(move || {
            let mut store = feedback_store.write();

            // Take pending feedback for this user
            if let Some(pending) = store.take_pending(&user_id_for_feedback) {
                // Compute response embedding for semantic similarity feedback
                let response_embedding: Option<Vec<f32>> = {
                    let memory_guard = memory_for_embed.read();
                    memory_guard.compute_embedding(&response_text).ok()
                };

                // Process the feedback with semantic similarity
                let signals = feedback::process_implicit_feedback_with_semantics(
                    &pending,
                    &response_text,
                    followup.as_deref(),
                    response_embedding.as_deref(),
                );

                let mut reinforced = Vec::new();
                let mut weakened = Vec::new();
                let mut helpful_ids: Vec<MemoryId> = Vec::new();
                let mut misleading_ids: Vec<MemoryId> = Vec::new();

                // Extract entities from context for fingerprinting
                let context_entities: Vec<String> =
                    feedback::extract_entities_simple(&pending.context)
                        .into_iter()
                        .collect();
                let context_embedding = pending.context_embedding.clone();

                for (memory_id, signal) in signals {
                    // Determine if this memory was helpful or misleading
                    let is_helpful = signal.value > 0.3;
                    let is_misleading = signal.value < -0.3;

                    // Get or create momentum for this memory
                    let momentum = store.get_or_create_momentum(
                        memory_id.clone(),
                        crate::memory::types::ExperienceType::Context,
                    );

                    // Track reinforced/weakened
                    let old_ema = momentum.ema;
                    let new_ema = {
                        momentum.update(signal.clone());
                        momentum.ema
                    };

                    // Add context fingerprint for pattern learning
                    if is_helpful || is_misleading {
                        let fingerprint = feedback::ContextFingerprint::new(
                            context_entities.clone(),
                            &context_embedding,
                            is_helpful,
                        );
                        momentum.add_context(fingerprint);
                    }

                    // Determine outcome based on signal and EMA change
                    if is_helpful || new_ema > old_ema + 0.05 {
                        reinforced.push(memory_id.0.to_string());
                        helpful_ids.push(memory_id.clone());
                    } else if is_misleading || new_ema < old_ema - 0.05 {
                        weakened.push(memory_id.0.to_string());
                        misleading_ids.push(memory_id.clone());
                    }

                    // Mark dirty after releasing the mutable borrow
                    store.mark_dirty(&memory_id);
                }

                // Flush dirty entries to disk
                if let Err(e) = store.flush() {
                    tracing::warn!("Failed to flush feedback store: {}", e);
                }

                let result = FeedbackProcessed {
                    memories_evaluated: pending.surfaced_memories.len(),
                    reinforced,
                    weakened,
                };
                (Some(result), helpful_ids, misleading_ids)
            } else {
                (None, Vec::new(), Vec::new())
            }
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Feedback task panicked: {e}")))?;

        // Apply reinforcement to memory system based on feedback
        if !helpful_ids.is_empty() || !misleading_ids.is_empty() {
            let memory_sys_for_reinforce = memory_system.clone();
            tokio::task::spawn_blocking(move || {
                let memory_guard = memory_sys_for_reinforce.read();

                // Reinforce helpful memories
                if !helpful_ids.is_empty() {
                    if let Err(e) = memory_guard
                        .reinforce_recall(&helpful_ids, crate::memory::RetrievalOutcome::Helpful)
                    {
                        tracing::warn!("Failed to reinforce helpful memories: {}", e);
                    }
                }

                // Weaken misleading memories
                if !misleading_ids.is_empty() {
                    if let Err(e) = memory_guard.reinforce_recall(
                        &misleading_ids,
                        crate::memory::RetrievalOutcome::Misleading,
                    ) {
                        tracing::warn!("Failed to weaken misleading memories: {}", e);
                    }
                }
            })
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Reinforce task panicked: {e}")))?;

            // Emit SSE event for feedback processing
            if let Some(ref feedback) = result {
                state.emit_event(MemoryEvent {
                    event_type: "FEEDBACK_PROCESSED".to_string(),
                    timestamp: chrono::Utc::now(),
                    user_id: req.user_id.clone(),
                    memory_id: None,
                    content_preview: Some(format!(
                        "Evaluated {} memories: {} reinforced, {} weakened",
                        feedback.memories_evaluated,
                        feedback.reinforced.len(),
                        feedback.weakened.len()
                    )),
                    memory_type: Some("feedback".to_string()),
                    importance: None,
                    count: Some(feedback.memories_evaluated),
                });
            }
        }

        result
    } else {
        None
    };

    // 1. Compute context embedding first (needed for composite relevance scoring)
    let context_for_embedding = req.context.clone();
    let memory_for_embedding = memory_system.clone();
    let context_embedding: Vec<f32> = tokio::task::spawn_blocking(move || {
        let memory_guard = memory_for_embedding.read();
        memory_guard
            .compute_embedding(&context_for_embedding)
            .unwrap_or_else(|_| vec![0.0; 384])
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding task panicked: {e}")))?;

    // 2. Retrieve memories using unified 5-layer pipeline
    // The pipeline already applies: hebbian boost (10%) + recency decay (10%) + RRF fusion
    // No double-scoring needed - just use the scores from recall() directly
    let context_clone = req.context.clone();
    let max_results = req.max_results;
    let user_id_for_query = req.user_id.clone();
    let feedback_store_for_scoring = state.feedback_store.clone();
    let memories: Vec<ProactiveSurfacedMemory> = {
        let memory = memory_system.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let feedback_guard = feedback_store_for_scoring.read();

            let query = MemoryQuery {
                user_id: Some(user_id_for_query),
                query_text: Some(context_clone),
                max_results: max_results * 2, // Fetch extra for feedback filtering
                ..Default::default()
            };
            let results = memory_guard.recall(&query).unwrap_or_default();

            // Use scores from 5-layer pipeline, apply optional feedback suppression
            let mut candidates: Vec<(SharedMemory, f32)> = results
                .into_iter()
                .filter_map(|m| {
                    // Skip memories without embeddings (can't be meaningfully ranked)
                    if m.experience.embeddings.is_none() {
                        return None;
                    }

                    // Base score from unified 5-layer pipeline (hebbian + recency + RRF)
                    let mut score = m.get_score().unwrap_or(0.0);

                    // Optional: Apply feedback suppression for frequently-ignored memories
                    // Negative momentum = often ignored → penalize
                    if let Some(fm) = feedback_guard.get_momentum(&m.id) {
                        let momentum = fm.ema_with_decay();
                        if momentum < 0.0 {
                            // Suppress by up to 20% for highly negative momentum
                            score *= 1.0 + (momentum * 0.2).max(-0.2);
                        }
                    }

                    Some((m, score))
                })
                .collect();

            // Sort by score (highest first) - already mostly sorted from recall()
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Return top results
            candidates
                .into_iter()
                .take(max_results)
                .map(|(m, score)| ProactiveSurfacedMemory {
                    id: m.id.0.to_string(),
                    content: m.experience.content.clone(),
                    memory_type: format!("{:?}", m.experience.experience_type),
                    score,
                    created_at: m.created_at.to_rfc3339(),
                    tags: m.experience.entities.clone(),
                    embedding: m.experience.embeddings.clone().unwrap_or_default(),
                })
                .collect()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // 3. Check due reminders
    let user_id = req.user_id.clone();
    let due_reminders: Vec<ReminderItem> = {
        let prospective = state.prospective_store.clone();
        tokio::task::spawn_blocking(move || {
            prospective
                .get_due_tasks(&user_id)
                .unwrap_or_default()
                .into_iter()
                .map(|t| {
                    let overdue = t.overdue_seconds();
                    let trigger_type = match &t.trigger {
                        ProspectiveTrigger::AtTime { .. } => "time".to_string(),
                        ProspectiveTrigger::AfterDuration { .. } => "duration".to_string(),
                        ProspectiveTrigger::OnContext { .. } => "context".to_string(),
                    };
                    ReminderItem {
                        id: t.id.0.to_string(),
                        content: t.content,
                        trigger_type,
                        status: format!("{:?}", t.status).to_lowercase(),
                        due_at: t.trigger.due_at(),
                        created_at: t.created_at,
                        triggered_at: t.triggered_at,
                        dismissed_at: t.dismissed_at,
                        priority: t.priority,
                        tags: t.tags,
                        overdue_seconds: overdue,
                    }
                })
                .collect()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // 4. Store pending feedback for next call (with embeddings for semantic feedback)
    {
        let surfaced_infos: Vec<feedback::SurfacedMemoryInfo> = memories
            .iter()
            .map(|m| {
                let id = uuid::Uuid::parse_str(&m.id).unwrap_or_else(|_| uuid::Uuid::new_v4());
                feedback::SurfacedMemoryInfo {
                    id: MemoryId(id),
                    entities: feedback::extract_entities_simple(&m.content),
                    content_preview: m.content.chars().take(100).collect(),
                    score: m.score,
                    embedding: m.embedding.clone(),
                }
            })
            .collect();

        if !surfaced_infos.is_empty() {
            let pending = feedback::PendingFeedback::new(
                req.user_id.clone(),
                req.context.clone(),
                context_embedding.clone(),
                surfaced_infos,
            );
            let feedback_store = state.feedback_store.clone();
            feedback_store.write().set_pending(pending);
        }
    }

    // Now check context triggers with semantic matching
    let user_id = req.user_id.clone();
    let context_for_triggers = req.context.clone();
    let memory_for_task_embed = memory_system.clone();
    let context_emb_for_triggers = context_embedding.clone();
    let context_reminders: Vec<ReminderItem> = {
        let prospective = state.prospective_store.clone();
        tokio::task::spawn_blocking(move || {
            let embed_fn = |text: &str| -> Option<Vec<f32>> {
                let memory_guard = memory_for_task_embed.read();
                memory_guard.compute_embedding(text).ok()
            };

            prospective
                .check_context_triggers_semantic(
                    &user_id,
                    &context_for_triggers,
                    &context_emb_for_triggers,
                    embed_fn,
                )
                .unwrap_or_default()
                .into_iter()
                .map(|(t, score)| {
                    let overdue = t.overdue_seconds();
                    ReminderItem {
                        id: t.id.0.to_string(),
                        content: t.content,
                        trigger_type: format!("context (score: {:.2})", score),
                        status: format!("{:?}", t.status).to_lowercase(),
                        due_at: t.trigger.due_at(),
                        created_at: t.created_at,
                        triggered_at: t.triggered_at,
                        dismissed_at: t.dismissed_at,
                        priority: t.priority,
                        tags: t.tags,
                        overdue_seconds: overdue,
                    }
                })
                .collect()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // 5. Auto-ingest previous assistant response (if provided and meaningful)
    // Uses segmentation engine for Hebbian-optimal atomic memories
    if req.auto_ingest {
        if let Some(ref prev_response) = req.previous_response {
            // Only store meaningful responses (not empty, not just tool calls, not boilerplate)
            let response_text = prev_response.trim();
            let is_meaningful = response_text.len() > 100
                && response_text.len() < 3000  // Skip very long responses (often tool outputs)
                && !response_text.starts_with("```")
                && !is_boilerplate_response(response_text);

            if is_meaningful {
                let response_text_owned = response_text.to_string();
                let memory = memory_system.clone();

                tokio::task::spawn_blocking(move || {
                    let memory_guard = memory.read();
                    let segmenter = SegmentationEngine::new();

                    // Segment assistant response into atomic memories
                    let segments = segmenter.segment(&response_text_owned, InputSource::AutoIngest);

                    for segment in segments {
                        // Format content with type prefix for clarity
                        let content = format!(
                            "[Assistant: {:?}] {}",
                            segment.experience_type, segment.content
                        );
                        let experience = Experience {
                            content,
                            experience_type: segment.experience_type,
                            entities: segment.entities,
                            tags: vec![
                                "assistant-response".to_string(),
                                "auto-captured".to_string(),
                            ],
                            ..Default::default()
                        };
                        let _ = memory_guard.remember(experience, None);
                    }
                })
                .await
                .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?;
            }
        }
    }

    // 6. Auto-ingest user context with segmentation for Hebbian-optimal storage
    // Apply quality filters before storing
    let clean_context = strip_system_noise(&req.context);
    let should_ingest = req.auto_ingest
        && clean_context.len() > 50           // Minimum meaningful length
        && clean_context.len() < 5000         // Allow larger contexts now (segmentation handles splitting)
        && !is_bare_question(&clean_context); // Don't store standalone questions

    let ingested_memory_id = if should_ingest {
        let context = clean_context;
        let memory = memory_system.clone();

        let memory_id = tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let segmenter = SegmentationEngine::new();

            // Segment user context into atomic memories
            let segments = segmenter.segment(&context, InputSource::AutoIngest);

            // Store each segment, return the first memory ID
            let mut first_id = None;
            for segment in segments {
                let experience = Experience {
                    content: segment.content,
                    experience_type: segment.experience_type,
                    entities: segment.entities,
                    tags: vec!["auto-captured".to_string()],
                    ..Default::default()
                };

                if let Ok(id) = memory_guard.remember(experience, None) {
                    if first_id.is_none() {
                        first_id = Some(id);
                    }
                }
            }
            first_id
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?;

        memory_id.map(|id| id.0.to_string())
    } else {
        None
    };

    // 7. Surface relevant todos using semantic search
    let relevant_todos: Vec<ProactiveTodoItem> = {
        // Use semantic search with context embedding for todo retrieval
        let semantic_results = state
            .todo_store
            .search_similar(&req.user_id, &context_embedding, 10)
            .unwrap_or_default();

        // Filter for active statuses and build ProactiveTodoItem list
        let mut todos_with_scores: Vec<ProactiveTodoItem> = semantic_results
            .into_iter()
            .filter(|(t, _score)| {
                matches!(
                    t.status,
                    TodoStatus::Todo | TodoStatus::InProgress | TodoStatus::Blocked
                )
            })
            .map(|(t, score)| {
                // Get project name for display
                let project_name = t.project_id.as_ref().and_then(|pid| {
                    state
                        .todo_store
                        .get_project(&req.user_id, pid)
                        .ok()
                        .flatten()
                        .map(|p| p.name)
                });

                ProactiveTodoItem {
                    id: t.id.0.to_string(),
                    short_id: t.short_id(),
                    content: t.content.clone(),
                    status: format!("{:?}", t.status).to_lowercase(),
                    priority: t.priority.indicator().to_string(),
                    project: project_name,
                    due_date: t.due_date.map(|d| d.format("%Y-%m-%d").to_string()),
                    relevance_reason: format!("semantic: {:.0}%", score * 100.0),
                    similarity_score: Some(score),
                }
            })
            .collect();

        // Also include in_progress todos regardless of semantic score (work continuity)
        let in_progress_candidates = state
            .todo_store
            .list_todos_for_user(&req.user_id, None)
            .unwrap_or_default()
            .into_iter()
            .filter(|t| t.status == TodoStatus::InProgress)
            .collect::<Vec<_>>();

        // Filter out duplicates separately to avoid borrow conflict
        let in_progress_todos: Vec<ProactiveTodoItem> = in_progress_candidates
            .into_iter()
            .filter(|t| {
                // Don't duplicate if already in semantic results
                !todos_with_scores.iter().any(|s| s.id == t.id.0.to_string())
            })
            .map(|t| {
                let project_name = t.project_id.as_ref().and_then(|pid| {
                    state
                        .todo_store
                        .get_project(&req.user_id, pid)
                        .ok()
                        .flatten()
                        .map(|p| p.name)
                });
                ProactiveTodoItem {
                    id: t.id.0.to_string(),
                    short_id: t.short_id(),
                    content: t.content.clone(),
                    status: "in_progress".to_string(),
                    priority: t.priority.indicator().to_string(),
                    project: project_name,
                    due_date: t.due_date.map(|d| d.format("%Y-%m-%d").to_string()),
                    relevance_reason: "active work".to_string(),
                    similarity_score: None,
                }
            })
            .collect();

        todos_with_scores.extend(in_progress_todos);

        // Sort by: in_progress first, then by similarity score
        todos_with_scores.sort_by(|a, b| {
            let a_in_progress = a.status == "in_progress";
            let b_in_progress = b.status == "in_progress";
            match (a_in_progress, b_in_progress) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b
                    .similarity_score
                    .partial_cmp(&a.similarity_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        todos_with_scores.into_iter().take(5).collect()
    };
    let todo_count = relevant_todos.len();

    let memory_count = memories.len();
    let reminder_count = due_reminders.len() + context_reminders.len();

    // Emit event for dashboard
    state.emit_event(MemoryEvent {
        event_type: "PROACTIVE_CONTEXT".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: ingested_memory_id.clone(),
        content_preview: Some(req.context.chars().take(50).collect()),
        memory_type: Some("proactive".to_string()),
        importance: None,
        count: Some(memory_count + reminder_count),
    });

    // Audit log for proactive context operations
    state.log_event(
        &req.user_id,
        "PROACTIVE_CONTEXT",
        ingested_memory_id.as_deref().unwrap_or("none"),
        &format!(
            "Context='{}' surfaced {} memories, {} reminders, {} todos (auto_ingest={})",
            req.context.chars().take(50).collect::<String>(),
            memory_count,
            reminder_count,
            todo_count,
            req.auto_ingest
        ),
    );

    // Track session event for memories surfaced
    if memory_count > 0 {
        let session_id = state.session_store.get_or_create_session(&req.user_id);
        let memory_ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        let avg_score = if !memories.is_empty() {
            memories.iter().map(|m| m.score).sum::<f32>() / memories.len() as f32
        } else {
            0.0
        };
        state.session_store.add_event(
            &session_id,
            SessionEvent::MemoriesSurfaced {
                timestamp: chrono::Utc::now(),
                query_preview: req.context.chars().take(100).collect(),
                memory_count,
                memory_ids,
                avg_score,
            },
        );
    }

    Ok(Json(ProactiveContextResponse {
        memories,
        due_reminders,
        context_reminders,
        memory_count,
        reminder_count,
        ingested_memory_id,
        feedback_processed,
        relevant_todos,
        todo_count,
    }))
}

// =============================================================================
// SURFACE RELEVANT HANDLER
// =============================================================================

/// POST /api/relevant - Proactive memory surfacing
/// Returns relevant memories based on current context using entity matching
/// and semantic similarity. Target latency: <30ms
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn surface_relevant(
    State(state): State<AppState>,
    Json(req): Json<relevance::RelevanceRequest>,
) -> Result<Json<relevance::RelevanceResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let ner = state.get_neural_ner();

    let engine = relevance::RelevanceEngine::new(ner);

    let response = {
        let memory_sys = memory_sys.clone();
        let graph_memory = graph_memory.clone();
        let context = req.context.clone();
        let config = req.config.clone();

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory_sys.read();
            let graph_guard = graph_memory.read();
            engine.surface_relevant(&context, &*memory_guard, Some(&*graph_guard), &config)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Broadcast RETRIEVE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "RETRIEVE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(req.context.chars().take(50).collect()),
        memory_type: Some("proactive".to_string()),
        importance: None,
        count: Some(response.memories.len()),
    });

    Ok(Json(response))
}

// =============================================================================
// TRACKED RECALL HANDLER
// =============================================================================

/// POST /api/recall/tracked - Retrieval with tracking for later feedback
///
/// Use this when you want to provide feedback later on whether memories were helpful.
/// Returns memory_ids that can be passed to /api/reinforce for Hebbian strengthening.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
pub async fn recall_tracked(
    State(state): State<AppState>,
    Json(req): Json<TrackedRetrieveRequest>,
) -> Result<Json<TrackedRetrieveResponse>, AppError> {
    let op_start = std::time::Instant::now();
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let query_text = req.query.clone();
    let limit = req.limit;
    let user_id = req.user_id.clone();

    let memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let query = MemoryQuery {
                user_id: Some(user_id),
                query_text: Some(query_text),
                max_results: limit,
                ..Default::default()
            };
            memory_guard.recall(&query).unwrap_or_default()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // Extract memory IDs for tracking
    let memory_ids: Vec<String> = memories.iter().map(|m| m.id.0.to_string()).collect();

    // Generate tracking ID (could be stored for audit, but for now just a UUID)
    let tracking_id = uuid::Uuid::new_v4().to_string();

    // Convert to response format
    let total = memories.len();
    let recall_memories: Vec<RecallMemory> = memories
        .into_iter()
        .enumerate()
        .map(|(rank, m)| {
            // Score based on rank position and salience
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
            }
        })
        .collect();

    let count = recall_memories.len();

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_RETRIEVE_DURATION
        .with_label_values(&["tracked"])
        .observe(duration);
    metrics::MEMORY_RETRIEVE_TOTAL
        .with_label_values(&["tracked", "success"])
        .inc();
    metrics::MEMORY_RETRIEVE_RESULTS
        .with_label_values(&["tracked"])
        .observe(count as f64);

    Ok(Json(TrackedRetrieveResponse {
        tracking_id,
        ids: memory_ids,
        memories: recall_memories,
    }))
}

// =============================================================================
// REINFORCE FEEDBACK HANDLER
// =============================================================================

/// POST /api/reinforce - Hebbian reinforcement based on task outcome
///
/// Call this after using memories to complete a task:
/// - "helpful": Memories that helped → boost importance, strengthen associations
/// - "misleading": Memories that misled → reduce importance, don't strengthen
/// - "neutral": Just record access, mild strengthening
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, outcome = %req.outcome, count = req.ids.len()))]
pub async fn reinforce_feedback(
    State(state): State<AppState>,
    Json(req): Json<ReinforceFeedbackRequest>,
) -> Result<Json<ReinforceFeedbackResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.ids.is_empty() {
        return Ok(Json(ReinforceFeedbackResponse {
            memories_processed: 0,
            associations_strengthened: 0,
            importance_boosts: 0,
            importance_decays: 0,
        }));
    }

    // Parse outcome
    let outcome_label = req.outcome.to_lowercase();
    let outcome = match outcome_label.as_str() {
        "helpful" => crate::memory::RetrievalOutcome::Helpful,
        "misleading" => crate::memory::RetrievalOutcome::Misleading,
        "neutral" | _ => crate::memory::RetrievalOutcome::Neutral,
    };

    // Convert string IDs to MemoryId
    let memory_ids: Vec<MemoryId> = req
        .ids
        .iter()
        .filter_map(|id| uuid::Uuid::parse_str(id).ok())
        .map(MemoryId)
        .collect();

    if memory_ids.is_empty() {
        return Err(AppError::InvalidInput {
            field: "ids".to_string(),
            reason: "No valid UUIDs provided".to_string(),
        });
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Run reinforcement in blocking task (involves RocksDB writes)
    let stats = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.reinforce_recall(&memory_ids, outcome)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    tracing::info!(
        user_id = %req.user_id,
        processed = stats.memories_processed,
        strengthened = stats.associations_strengthened,
        boosts = stats.importance_boosts,
        decays = stats.importance_decays,
        "Hebbian reinforcement applied"
    );

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::HEBBIAN_REINFORCE_DURATION
        .with_label_values(&[&outcome_label])
        .observe(duration);
    metrics::HEBBIAN_REINFORCE_TOTAL
        .with_label_values(&[&outcome_label, &String::from("success")])
        .inc();

    Ok(Json(ReinforceFeedbackResponse {
        memories_processed: stats.memories_processed,
        associations_strengthened: stats.associations_strengthened,
        importance_boosts: stats.importance_boosts,
        importance_decays: stats.importance_decays,
    }))
}

// =============================================================================
// RECALL BY TAGS HANDLER
// =============================================================================

/// POST /api/recall/tags - Recall memories by tags
///
/// Returns memories matching ANY of the provided tags.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn recall_by_tags(
    State(state): State<AppState>,
    Json(req): Json<RecallByTagsRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.tags.is_empty() {
        return Err(AppError::InvalidInput {
            field: "tags".to_string(),
            reason: "At least one tag must be provided".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let limit = req.limit.unwrap_or(50);

    // Use recall_by_tags which increments the retrieval counter
    let raw_memories = memory_guard
        .recall_by_tags(&req.tags, limit)
        .map_err(AppError::Internal)?;
    let count = raw_memories.len();

    // Serialize memories to JSON for response
    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    info!(
        "📋 Recall by tags: user={}, tags={:?}, found={}",
        req.user_id, req.tags, count
    );

    // Broadcast RETRIEVE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "RETRIEVE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(format!("tags: {}", req.tags.join(", "))),
        memory_type: Some("by_tags".to_string()),
        importance: None,
        count: Some(count),
    });

    Ok(Json(RetrieveResponse { memories, count }))
}

// =============================================================================
// RECALL BY DATE HANDLER
// =============================================================================

/// POST /api/recall/date - Recall memories by date range
///
/// Returns memories created within the specified date range.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn recall_by_date(
    State(state): State<AppState>,
    Json(req): Json<RecallByDateRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.end < req.start {
        return Err(AppError::InvalidInput {
            field: "end".to_string(),
            reason: "End date must be after start date".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let limit = req.limit.unwrap_or(50);

    // Use recall_by_date which increments the retrieval counter
    let raw_memories = memory_guard
        .recall_by_date(req.start, req.end, limit)
        .map_err(AppError::Internal)?;
    let count = raw_memories.len();

    // Serialize memories to JSON for response
    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    info!(
        "📅 Recall by date: user={}, start={}, end={}, found={}",
        req.user_id, req.start, req.end, count
    );

    // Broadcast RETRIEVE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "RETRIEVE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(format!(
            "{} to {}",
            req.start.format("%Y-%m-%d"),
            req.end.format("%Y-%m-%d")
        )),
        memory_type: Some("by_date".to_string()),
        importance: None,
        count: Some(count),
    });

    Ok(Json(RetrieveResponse { memories, count }))
}
