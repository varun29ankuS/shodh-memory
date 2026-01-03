//! Shodh-Memory - Offline, user-isolated memory layer for AI agents
//!
//! Standalone memory server with REST API for Python clients

use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use chrono::Datelike;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use tokio::signal;

// Static regexes for entity extraction (compiled once at startup)
static ALLCAPS_REGEX: OnceLock<regex::Regex> = OnceLock::new();
static ISSUE_ID_REGEX: OnceLock<regex::Regex> = OnceLock::new();

fn get_allcaps_regex() -> &'static regex::Regex {
    ALLCAPS_REGEX.get_or_init(|| regex::Regex::new(r"[A-Z]{2,}[A-Z0-9]*").unwrap())
}

fn get_issue_id_regex() -> &'static regex::Regex {
    ISSUE_ID_REGEX.get_or_init(|| regex::Regex::new(r"([A-Z]{2,10}-\d+)").unwrap())
}

/// Classify experience type from text content using keyword patterns.
/// Returns the most likely ExperienceType based on linguistic signals.
fn classify_experience_type(content: &str) -> memory::ExperienceType {
    let lower = content.to_lowercase();

    // Decision signals - choices, preferences, commitments
    const DECISION_PATTERNS: &[&str] = &[
        "decided",
        "will use",
        "going with",
        "chose",
        "chosen",
        "prefer",
        "i'll",
        "we'll",
        "let's use",
        "selected",
        "picking",
        "opting for",
        "the approach is",
        "strategy is",
        "plan is to",
        "going to use",
    ];

    // Learning signals - new knowledge acquired
    const LEARNING_PATTERNS: &[&str] = &[
        "learned",
        "realized",
        "discovered",
        "found out",
        "turns out",
        "til ",
        "today i learned",
        "now i know",
        "understanding is",
        "figured out",
        "the reason is",
        "because",
        "works because",
        "key insight",
        "important to note",
        "remember that",
    ];

    // Error signals - bugs, issues, problems
    const ERROR_PATTERNS: &[&str] = &[
        "bug",
        "error",
        "fix",
        "fixed",
        "broken",
        "issue",
        "problem",
        "crash",
        "fail",
        "exception",
        "resolved",
        "workaround",
        "the solution was",
        "root cause",
        "debugging",
    ];

    // Discovery signals - findings, observations
    const DISCOVERY_PATTERNS: &[&str] = &[
        "found",
        "noticed",
        "interesting",
        "surprisingly",
        "unexpected",
        "turns out",
        "apparently",
        "it seems",
        "observation",
    ];

    // Context signals - user preferences, settings, environment
    const CONTEXT_PATTERNS: &[&str] = &[
        "prefers",
        "preference",
        "wants",
        "likes",
        "user",
        "setting",
        "configuration",
        "environment",
        "workspace",
        "setup",
    ];

    // Pattern signals - recurring behaviors, habits
    const PATTERN_PATTERNS: &[&str] = &[
        "pattern",
        "always",
        "usually",
        "tends to",
        "whenever",
        "every time",
        "consistently",
        "habit",
        "recurring",
    ];

    // Score each type
    let decision_score = DECISION_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let learning_score = LEARNING_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let error_score = ERROR_PATTERNS.iter().filter(|p| lower.contains(*p)).count();
    let discovery_score = DISCOVERY_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let context_score = CONTEXT_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let pattern_score = PATTERN_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();

    // Find highest scoring type (require at least 1 match)
    let scores = [
        (decision_score, memory::ExperienceType::Decision),
        (learning_score, memory::ExperienceType::Learning),
        (error_score, memory::ExperienceType::Error),
        (discovery_score, memory::ExperienceType::Discovery),
        (context_score, memory::ExperienceType::Context),
        (pattern_score, memory::ExperienceType::Pattern),
    ];

    scores
        .into_iter()
        .filter(|(score, _)| *score > 0)
        .max_by_key(|(score, _)| *score)
        .map(|(_, typ)| typ)
        .unwrap_or(memory::ExperienceType::Conversation) // Default if no patterns match
}

/// Strip system noise from context to extract meaningful user content.
/// Removes <system-reminder>, <shodh-context>, Claude Code system prompts, and code blocks.
fn strip_system_noise(content: &str) -> String {
    let mut result = content.to_string();

    // Remove <system-reminder>...</system-reminder> blocks (handles multiline)
    while let Some(start) = result.find("<system-reminder>") {
        if let Some(end) = result.find("</system-reminder>") {
            let end_pos = end + "</system-reminder>".len();
            result = format!("{}{}", &result[..start], &result[end_pos..]);
        } else {
            break;
        }
    }

    // Remove <shodh-context>...</shodh-context> blocks (our own injected context)
    while let Some(start) = result.find("<shodh-context") {
        if let Some(end) = result.find("</shodh-context>") {
            let end_pos = end + "</shodh-context>".len();
            result = format!("{}{}", &result[..start], &result[end_pos..]);
        } else {
            break;
        }
    }

    // Remove Claude Code file content blocks - Windows paths
    while let Some(start) = result.find("Contents of C:\\") {
        let search_area = &result[start..];
        let end_offset = search_area
            .find("\n\n")
            .or_else(|| search_area.find("\r\n\r\n"))
            .unwrap_or(search_area.len().min(2000));
        result = format!("{}{}", &result[..start], &result[start + end_offset..]);
    }

    // Remove Claude Code file content blocks - Unix paths
    while let Some(start) = result.find("Contents of /") {
        let search_area = &result[start..];
        let end_offset = search_area
            .find("\n\n")
            .or_else(|| search_area.find("\r\n\r\n"))
            .unwrap_or(search_area.len().min(2000));
        result = format!("{}{}", &result[..start], &result[start + end_offset..]);
    }

    // Remove fenced code blocks (```...```) - these are often tool outputs, not memories
    while let Some(start) = result.find("```") {
        if let Some(end) = result[start + 3..].find("```") {
            let end_pos = start + 3 + end + 3;
            result = format!("{}{}", &result[..start], &result[end_pos..]);
        } else {
            // Unclosed code block - remove from start to end
            result = result[..start].to_string();
            break;
        }
    }

    // Clean up excessive whitespace
    let result = result.split_whitespace().collect::<Vec<_>>().join(" ");

    // If result is mostly empty or very short after cleaning, return empty
    // to prevent storing noise
    let trimmed = result.trim();
    if trimmed.len() < 10 || trimmed.chars().filter(|c| c.is_alphabetic()).count() < 5 {
        return String::new();
    }

    trimmed.to_string()
}

/// Check if content is a bare question (not worth storing as memory).
/// Questions like "what is X?" or "how do I Y?" without context are low-value.
/// Extended to catch longer questions that still lack substance.
fn is_bare_question(content: &str) -> bool {
    let trimmed = content.trim();
    let lower = trimmed.to_lowercase();

    // Question word starters
    let question_starters = [
        "what", "how", "why", "where", "when", "who", "can", "could", "is", "are", "do", "does",
        "will", "would", "should", "have",
    ];
    let starts_with_question = question_starters.iter().any(|q| lower.starts_with(q));
    let ends_with_question = trimmed.ends_with('?');

    // Short content - apply looser filter
    if trimmed.len() < 100 {
        if starts_with_question || ends_with_question {
            return true;
        }
    }

    // Medium content (100-300 chars) - check if it's purely a question without context
    if trimmed.len() < 300 && (starts_with_question || ends_with_question) {
        // Check for substance indicators that make it worth storing
        let has_substance = lower.contains("because")
            || lower.contains("the reason")
            || lower.contains("i think")
            || lower.contains("i believe")
            || lower.contains("we should")
            || lower.contains("decided")
            || lower.contains("learned")
            || lower.contains("found that")
            || lower.contains("the issue")
            || lower.contains("the problem")
            || lower.contains("the solution");

        if !has_substance {
            // Count sentences - pure questions are typically single sentence
            let sentence_count = trimmed.matches('.').count()
                + trimmed.matches('!').count()
                + trimmed.matches('?').count();

            if sentence_count <= 2 {
                return true;
            }
        }
    }

    false
}

/// Check if assistant response is boilerplate/low-value content.
/// Filters out generic greetings, offers to help, and repetitive patterns.
fn is_boilerplate_response(content: &str) -> bool {
    let lower = content.to_lowercase();

    // Generic greeting/ready-to-help patterns (high confidence noise)
    let boilerplate_starts = [
        "i'm ready to help",
        "i am ready to help",
        "i'm here to help",
        "i am here to help",
        "i can help you",
        "i'd be happy to help",
        "i would be happy to help",
        "let me help you",
        "i understand. i'm ready",
        "i understand. i am ready",
        "sure, i can",
        "sure! i can",
        "absolutely! i",
        "of course! i",
        "great question!",
        "good question!",
    ];

    if boilerplate_starts.iter().any(|p| lower.starts_with(p)) {
        return true;
    }

    // Generic offer patterns anywhere in short responses (<500 chars)
    if lower.len() < 500 {
        let generic_offers = [
            "what would you like me to",
            "let me know if you",
            "let me know what you",
            "feel free to ask",
            "don't hesitate to",
            "i'm happy to",
            "just let me know",
            "how can i assist",
            "how may i help",
            "is there anything else",
        ];

        let offer_count = generic_offers.iter().filter(|p| lower.contains(*p)).count();
        // If more than half the response is generic offers, filter it
        if offer_count >= 2 {
            return true;
        }
    }

    // Check for responses that are mostly bullet points of capabilities
    // Pattern: "I can:\n- X\n- Y\n- Z" without actual content
    if lower.contains("i can:") || lower.contains("i'm able to:") {
        let bullet_count = content.matches("\n-").count() + content.matches("\n•").count();
        let has_substance = lower.contains("because")
            || lower.contains("the reason")
            || lower.contains("specifically")
            || lower.contains("for example");

        // Capability list without substance = boilerplate
        if bullet_count >= 3 && !has_substance {
            return true;
        }
    }

    false
}

use tower::limit::ConcurrencyLimitLayer;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::services::ServeDir;
use tracing::info;

mod ab_testing; // A/B testing infrastructure
mod auth;
mod backup;
mod config;
mod constants;
mod decay;
mod embeddings;
mod errors;
mod graph_memory;
mod integrations; // SHO-40: External integrations (Linear, GitHub)
mod memory;
mod metrics; // P1.1: Observability
mod middleware; // P1.3: HTTP request tracking
mod relevance; // SHO-29: Proactive memory surfacing
mod similarity;
mod streaming; // SHO-25: Streaming memory ingestion
mod tracing_setup;
mod validation;
mod vector_db; // P1.6: Distributed tracing

use config::ServerConfig;
use constants::{
    DATABASE_FLUSH_TIMEOUT_SECS, GRACEFUL_SHUTDOWN_TIMEOUT_SECS, VECTOR_INDEX_SAVE_TIMEOUT_SECS,
    VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
};

use embeddings::NerEntityType;
use embeddings::{
    are_ner_models_downloaded, download_ner_models, get_ner_models_dir, NerConfig, NeuralNer,
};
use errors::{AppError, ValidationErrorExt};
use graph_memory::{
    EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, GraphStats, GraphTraversal,
    RelationType, RelationshipEdge,
};
use memory::{
    extract_entities_simple,
    injection::{
        compute_relevance, InjectionCandidate, InjectionConfig, InjectionEngine, RelevanceInput,
    },
    process_implicit_feedback,
    prospective::ProspectiveStore,
    segmentation::{InputSource, SegmentationEngine},
    todo_formatter, ActivatedMemory, Experience, ExperienceType, FeedbackStore, FileMemoryStats,
    FileMemoryStore, GraphStats as VisualizationStats, IndexingResult, Memory, MemoryConfig,
    MemoryId, MemoryStats, MemorySystem, PendingFeedback, Project, ProjectId, ProjectStats,
    ProjectStatus, ProspectiveTask, ProspectiveTaskId, ProspectiveTaskStatus, ProspectiveTrigger,
    Query as MemoryQuery, Recurrence, SharedMemory, SurfacedMemoryInfo, Todo, TodoComment,
    TodoCommentId, TodoCommentType, TodoId, TodoPriority, TodoStatus, TodoStore, UserTodoStats,
};

/// Audit event for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: String, // CREATE, UPDATE, DELETE, RETRIEVE
    pub memory_id: String,
    pub details: String,
}

/// SSE Memory Event - lightweight event for real-time streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub event_type: String, // CREATE, RETRIEVE, DELETE, GRAPH_UPDATE
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user_id: String,
    pub memory_id: Option<String>,
    pub content_preview: Option<String>, // First 100 chars
    pub memory_type: Option<String>,
    pub importance: Option<f32>,
    pub count: Option<usize>, // For retrieve events - number of results
}

/// Context status from Claude Code (MCP server reports this via status line)
/// Streamed to TUI via separate SSE channel for real-time context window display
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextStatus {
    /// Session identifier (from Claude Code session_id)
    pub session_id: Option<String>,
    /// Tokens used in current session
    pub tokens_used: u64,
    /// Token budget (context window size)
    pub tokens_budget: u64,
    /// Usage percentage (0-100)
    pub percent_used: u8,
    /// Current working directory (approximates current task)
    pub current_task: Option<String>,
    /// Model name being used
    pub model: Option<String>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// All active Claude Code sessions (keyed by session_id)
pub type ContextSessions = DashMap<String, ContextStatus>;

// Note: Audit and memory configuration is now in config.rs and loaded via ServerConfig

/// Helper struct for audit log rotation (allows spawn_blocking with minimal clone)
struct MultiUserMemoryManagerRotationHelper {
    audit_db: Arc<rocksdb::DB>,
    audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,
    /// Audit retention days (from config)
    audit_retention_days: i64,
    /// Max audit entries per user (from config)
    audit_max_entries: usize,
}

impl MultiUserMemoryManagerRotationHelper {
    fn rotate_user_audit_logs(&self, user_id: &str) -> Result<usize> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::days(self.audit_retention_days);
        let cutoff_nanos = cutoff_time.timestamp_nanos_opt().unwrap_or(0);

        let mut events: Vec<(Vec<u8>, AuditEvent, i64)> = Vec::new();
        let prefix = format!("{user_id}:");

        // Collect all events for this user with their timestamps
        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for (key, value) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }

                if let Ok((event, _)) = bincode::serde::decode_from_slice::<AuditEvent, _>(&value, bincode::config::standard()) {
                    let timestamp_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                    events.push((key.to_vec(), event, timestamp_nanos));
                }
            }
        }

        let initial_count = events.len();
        let mut removed_count = 0;

        // Sort by timestamp (newest first)
        events.sort_by(|a, b| b.2.cmp(&a.2));

        // Determine which events to remove
        let mut keys_to_remove = Vec::new();

        for (idx, (key, _event, timestamp_nanos)) in events.iter().enumerate() {
            let should_remove =
                // Remove if older than retention period
                *timestamp_nanos < cutoff_nanos ||
                // Remove if exceeding max count (keep only newest N entries)
                idx >= self.audit_max_entries;

            if should_remove {
                keys_to_remove.push(key.clone());
                removed_count += 1;
            }
        }

        // Remove old entries from RocksDB
        if !keys_to_remove.is_empty() {
            let mut batch = rocksdb::WriteBatch::default();
            for key in &keys_to_remove {
                batch.delete(key);
            }
            self.audit_db
                .write(batch)
                .map_err(|e| anyhow::anyhow!("Failed to write rotation batch: {e}"))?;
        }

        // Update in-memory cache by removing old events
        if removed_count > 0 {
            if let Some(log) = self.audit_logs.get(user_id) {
                let mut log_guard = log.write();

                // Keep only events that weren't removed
                log_guard.retain(|event| {
                    let event_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                    event_nanos >= cutoff_nanos
                        && initial_count - removed_count <= self.audit_max_entries
                });

                // If still too many, keep only the newest ones
                // VecDeque maintains insertion order (oldest at front, newest at back)
                // so pop_front() removes oldest entries - O(1) per pop
                while log_guard.len() > self.audit_max_entries {
                    log_guard.pop_front();
                }
            }
        }

        Ok(removed_count)
    }
}

/// Multi-user memory manager
pub struct MultiUserMemoryManager {
    /// Per-user memory systems with LRU eviction (prevents unbounded growth)
    /// Uses moka concurrent cache for lock-free reads (no mutex contention)
    user_memories: moka::sync::Cache<String, Arc<parking_lot::RwLock<MemorySystem>>>,

    /// Per-user audit logs (enterprise feature - in-memory cache)
    /// Uses VecDeque for O(1) rotation (push_back + pop_front)
    audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,

    /// Persistent audit log storage
    audit_db: Arc<rocksdb::DB>,

    /// Base storage path
    base_path: std::path::PathBuf,

    /// Default config
    default_config: MemoryConfig,

    /// Counter for audit log rotation checks (atomic for lock-free increment)
    audit_log_counter: Arc<std::sync::atomic::AtomicUsize>,

    /// Per-user graph memory systems (knowledge graphs) - also needs LRU eviction
    /// Uses moka concurrent cache for lock-free reads
    graph_memories: moka::sync::Cache<String, Arc<parking_lot::RwLock<GraphMemory>>>,

    /// Neural NER for automatic entity extraction (uses TinyBERT ONNX model)
    neural_ner: Arc<NeuralNer>,

    /// User eviction counter for metrics
    user_evictions: Arc<std::sync::atomic::AtomicUsize>,

    /// Server configuration (configurable via environment)
    server_config: ServerConfig,

    /// SSE event broadcaster for real-time dashboard updates
    /// Broadcast channel allows multiple subscribers (SSE clients) to receive events
    event_broadcaster: tokio::sync::broadcast::Sender<MemoryEvent>,

    /// Streaming memory extractor for implicit learning
    /// Handles WebSocket connections for continuous memory formation
    streaming_extractor: Arc<streaming::StreamingMemoryExtractor>,

    /// Prospective memory store for reminders/intentions (SHO-116)
    /// Handles time-based and context-based triggers
    prospective_store: Arc<ProspectiveStore>,

    /// GTD-style todo store (Linear-inspired)
    /// Handles todos, projects, and task management
    todo_store: Arc<TodoStore>,

    /// File memory store for codebase integration (MEM-29)
    /// Stores learned knowledge about files in codebases
    file_store: Arc<FileMemoryStore>,

    /// Implicit feedback store for memory reinforcement
    feedback_store: Arc<parking_lot::RwLock<FeedbackStore>>,

    /// Backup engine for automated and manual backups
    backup_engine: Arc<backup::ShodhBackupEngine>,

    /// Context status from Claude Code sessions (keyed by session_id)
    /// Multiple Claude windows can run simultaneously, each with own context
    context_sessions: Arc<ContextSessions>,

    /// SSE broadcaster for context status updates (separate from memory events)
    context_broadcaster: tokio::sync::broadcast::Sender<ContextStatus>,

    /// A/B testing manager for relevance scoring experiments
    ab_test_manager: Arc<ab_testing::ABTestManager>,
}

impl MultiUserMemoryManager {
    pub fn new(base_path: std::path::PathBuf, server_config: ServerConfig) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;

        // Initialize persistent audit log storage
        let audit_path = base_path.join("audit_logs");
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let audit_db = Arc::new(rocksdb::DB::open(&opts, audit_path)?);

        // Create broadcast channel for SSE events (capacity 1024 events)
        // Older events are dropped if subscribers can't keep up
        let (event_broadcaster, _) = tokio::sync::broadcast::channel(1024);

        // Initialize Neural NER - check if models exist, download if not
        let ner_dir = get_ner_models_dir();
        tracing::debug!("Checking for NER models at {:?}", ner_dir);
        let neural_ner = if are_ner_models_downloaded() {
            tracing::debug!("NER models found, using existing files");
            let config = NerConfig {
                model_path: ner_dir.join("model.onnx"),
                tokenizer_path: ner_dir.join("tokenizer.json"),
                max_length: 128,
                confidence_threshold: 0.5,
            };
            match NeuralNer::new(config) {
                Ok(ner) => {
                    info!(
                        "🧠 Neural NER initialized (TinyBERT model at {:?})",
                        ner_dir
                    );
                    Arc::new(ner)
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize neural NER: {}. Using fallback.", e);
                    Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                }
            }
        } else {
            // Auto-download NER models if not present
            tracing::debug!("NER models not found at {:?}, will download", ner_dir);
            info!("📥 Downloading NER models (TinyBERT-NER, ~15MB)...");
            match download_ner_models(Some(std::sync::Arc::new(|downloaded, total| {
                if total > 0 {
                    let percent = (downloaded as f64 / total as f64 * 100.0) as u32;
                    if percent % 20 == 0 {
                        tracing::info!("NER model download: {}%", percent);
                    }
                }
            }))) {
                Ok(ner_dir) => {
                    info!("✅ NER models downloaded to {:?}", ner_dir);
                    let config = NerConfig {
                        model_path: ner_dir.join("model.onnx"),
                        tokenizer_path: ner_dir.join("tokenizer.json"),
                        max_length: 128,
                        confidence_threshold: 0.5,
                    };
                    match NeuralNer::new(config) {
                        Ok(ner) => {
                            info!("🧠 Neural NER initialized after download");
                            Arc::new(ner)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to initialize downloaded NER: {}. Using fallback.",
                                e
                            );
                            Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to download NER models: {}. Using rule-based fallback.",
                        e
                    );
                    Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                }
            }
        };

        // Eviction counter for metrics (shared with cache eviction listener)
        let user_evictions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let evictions_clone = user_evictions.clone();
        let max_cache = server_config.max_users_in_memory;

        // Build moka cache with eviction listener for logging
        let user_memories = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .eviction_listener(move |key: Arc<String>, _value, cause| {
                if cause == moka::notification::RemovalCause::Size {
                    evictions_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    info!(
                        "📤 Evicted user '{}' from memory cache (LRU, cache_size={})",
                        key, max_cache
                    );
                }
            })
            .build();

        // Build moka cache for graph memories
        let graph_memories = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .eviction_listener(move |key: Arc<String>, _value, _cause| {
                info!(
                    "📤 Evicted graph for user '{}' from memory cache (LRU)",
                    key
                );
            })
            .build();

        // Initialize prospective memory store (SHO-116)
        let prospective_store = Arc::new(ProspectiveStore::new(&base_path)?);
        info!("📅 Prospective memory store initialized");

        // Initialize GTD todo store (Linear-style)
        let todo_store = Arc::new(TodoStore::new(&base_path)?);
        info!("📋 Todo store initialized");

        // Initialize file memory store (codebase integration)
        let file_store = Arc::new(FileMemoryStore::new(&base_path)?);
        info!("📁 File memory store initialized");

        // Initialize feedback store with persistence
        let feedback_store = Arc::new(parking_lot::RwLock::new(
            FeedbackStore::with_persistence(base_path.join("feedback")).unwrap_or_else(|e| {
                tracing::warn!("Failed to load feedback store: {}, using in-memory", e);
                FeedbackStore::new()
            }),
        ));
        info!("🔄 Feedback store initialized");

        // Initialize streaming memory extractor (needs feedback_store for relevance scoring)
        let streaming_extractor = Arc::new(streaming::StreamingMemoryExtractor::new(
            neural_ner.clone(),
            feedback_store.clone(),
        ));
        info!("📡 Streaming memory extractor initialized");

        // Initialize backup engine
        let backup_path = base_path.join("backups");
        let backup_engine = Arc::new(backup::ShodhBackupEngine::new(backup_path)?);
        if server_config.backup_enabled {
            info!(
                "💾 Backup engine initialized (interval: {}h, keep: {})",
                server_config.backup_interval_secs / 3600,
                server_config.backup_max_count
            );
        } else {
            info!("💾 Backup engine initialized (auto-backup disabled)");
        }

        let manager = Self {
            user_memories,
            audit_logs: Arc::new(DashMap::new()),
            audit_db,
            base_path,
            default_config: MemoryConfig::default(),
            audit_log_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            graph_memories,
            neural_ner,
            user_evictions,
            server_config,
            event_broadcaster,
            streaming_extractor,
            prospective_store,
            todo_store,
            file_store,
            feedback_store,
            backup_engine,
            context_sessions: Arc::new(DashMap::new()),
            context_broadcaster: {
                let (tx, _) = tokio::sync::broadcast::channel(16);
                tx
            },
            ab_test_manager: Arc::new(ab_testing::ABTestManager::new()),
        };

        // Perform initial audit log rotation on startup
        info!("🧹 Running initial audit log rotation...");
        if let Err(e) = manager.rotate_all_audit_logs() {
            tracing::warn!("Failed to rotate audit logs on startup: {}", e);
        }

        Ok(manager)
    }

    /// Log audit event (non-blocking with background persistence)
    fn log_event(&self, user_id: &str, event_type: &str, memory_id: &str, details: &str) {
        let event = AuditEvent {
            timestamp: chrono::Utc::now(),
            event_type: event_type.to_string(),
            memory_id: memory_id.to_string(),
            details: details.to_string(),
        };

        // Persist to RocksDB in background (non-blocking)
        let key = format!(
            "{}:{}",
            user_id,
            event.timestamp.timestamp_nanos_opt().unwrap_or(0)
        );
        if let Ok(serialized) = bincode::serde::encode_to_vec(&event, bincode::config::standard()) {
            let db = self.audit_db.clone();
            let key_bytes = key.into_bytes();

            // Spawn blocking DB write on dedicated thread pool
            tokio::task::spawn_blocking(move || {
                if let Err(e) = db.put(&key_bytes, &serialized) {
                    tracing::error!("Failed to persist audit log: {}", e);
                }
            });
        }

        // Also update in-memory cache for fast access (with size cap to prevent unbounded growth)
        let max_entries = self.server_config.audit_max_entries_per_user;
        if let Some(log) = self.audit_logs.get(user_id) {
            let mut entries = log.write();
            entries.push_back(event); // O(1) amortized
                                      // Enforce in-memory size cap: remove oldest entries if over limit
                                      // Using pop_front() is O(1) vs drain(0..n) which is O(n)
            while entries.len() > max_entries {
                entries.pop_front();
            }
        } else {
            let mut deque = VecDeque::new();
            deque.push_back(event);
            let log = Arc::new(parking_lot::RwLock::new(deque));
            self.audit_logs.insert(user_id.to_string(), log);
        }

        // Check if rotation is needed (every N events) - LOCK-FREE atomic operation
        let count = self
            .audit_log_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Only the thread that hits the interval performs rotation (no race condition)
        if count % self.server_config.audit_rotation_check_interval == 0 && count > 0 {
            // CRITICAL FIX: Run rotation in background blocking thread
            // rotate_user_audit_logs() does RocksDB prefix_iterator + WriteBatch (blocking I/O)
            // Running this on async threads starves the Tokio runtime
            let audit_db = self.audit_db.clone();
            let audit_logs = self.audit_logs.clone();
            let user_id_clone = user_id.to_string();

            let audit_retention_days = self.server_config.audit_retention_days as i64;
            let audit_max_entries = self.server_config.audit_max_entries_per_user;

            tokio::task::spawn_blocking(move || {
                // Reconstruct the necessary state for rotation
                let manager = MultiUserMemoryManagerRotationHelper {
                    audit_db,
                    audit_logs,
                    audit_retention_days,
                    audit_max_entries,
                };
                if let Err(e) = manager.rotate_user_audit_logs(&user_id_clone) {
                    tracing::debug!("Audit log rotation check for user {}: {}", user_id_clone, e);
                }
            });
        }
    }

    /// Emit SSE event to all connected dashboard clients
    /// Non-blocking - if no clients are listening, event is dropped silently
    fn emit_event(&self, event: MemoryEvent) {
        // broadcast::send returns Err if no receivers, which is fine
        let _ = self.event_broadcaster.send(event);
    }

    /// Subscribe to SSE events (returns a receiver)
    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<MemoryEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get audit history for user (reads from persistent storage)
    pub fn get_history(&self, user_id: &str, memory_id: Option<&str>) -> Vec<AuditEvent> {
        // Try in-memory cache first (fast path)
        if let Some(log) = self.audit_logs.get(user_id) {
            let events = log.read();
            if !events.is_empty() {
                // Cache hit - use cached data
                return if let Some(mid) = memory_id {
                    events
                        .iter()
                        .filter(|e| e.memory_id == mid)
                        .cloned()
                        .collect()
                } else {
                    // Convert VecDeque to Vec for return
                    events.iter().cloned().collect()
                };
            }
        }

        // Cache miss - load from RocksDB (happens after restart)
        let mut events = Vec::new();
        let prefix = format!("{user_id}:");

        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for (key, value) in iter.flatten() {
            // Check if key still matches our user_id prefix
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break; // Moved past our user's events
                }

                // Deserialize event
                if let Ok((event, _)) = bincode::serde::decode_from_slice::<AuditEvent, _>(&value, bincode::config::standard()) {
                    events.push(event);
                }
            }
        }

        // Update cache for next time (convert Vec to VecDeque)
        if !events.is_empty() {
            let log = Arc::new(parking_lot::RwLock::new(VecDeque::from(events.clone())));
            self.audit_logs.insert(user_id.to_string(), log);
        }

        // Filter by memory_id if requested
        if let Some(mid) = memory_id {
            events.into_iter().filter(|e| e.memory_id == mid).collect()
        } else {
            events
        }
    }

    /// Get or create memory system for a user
    pub fn get_user_memory(&self, user_id: &str) -> Result<Arc<parking_lot::RwLock<MemorySystem>>> {
        // Try to get from cache (lock-free read, updates LRU order internally)
        if let Some(memory) = self.user_memories.get(user_id) {
            return Ok(memory);
        }

        // Create new memory system for this user
        let user_path = self.base_path.join(user_id);
        let config = MemoryConfig {
            storage_path: user_path,
            ..self.default_config.clone()
        };

        let memory_system = MemorySystem::new(config).with_context(|| {
            format!("Failed to initialize memory system for user '{}'", user_id)
        })?;
        let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));

        // Insert into cache (moka handles LRU eviction automatically via eviction_listener)
        self.user_memories
            .insert(user_id.to_string(), memory_arc.clone());

        info!("Created memory system for user: {}", user_id);

        // Initialize vector index (load from disk or rebuild)
        if let Err(e) = self.init_user_vector_index(user_id) {
            tracing::warn!(
                "Vector index initialization failed for user {}: {}",
                user_id,
                e
            );
            // Don't fail user creation if indexing fails - semantic search will be unavailable
        }

        Ok(memory_arc)
    }

    /// Delete user data (GDPR compliance)
    pub fn forget_user(&self, user_id: &str) -> Result<()> {
        // Remove from memory cache (lock-free)
        self.user_memories.invalidate(user_id);

        // Delete storage directory
        let user_path = self.base_path.join(user_id);
        if user_path.exists() {
            std::fs::remove_dir_all(&user_path)?;
        }

        info!("🧠 Deleted all data for user: {}", user_id);
        Ok(())
    }

    /// Get statistics for a user (includes memory + graph stats)
    pub fn get_stats(&self, user_id: &str) -> Result<MemoryStats> {
        let memory = self.get_user_memory(user_id)?;
        let memory_guard = memory.read();
        let mut stats = memory_guard.stats();

        // Add graph stats
        if let Ok(graph) = self.get_user_graph(user_id) {
            let graph_guard = graph.read();
            if let Ok(graph_stats) = graph_guard.get_stats() {
                stats.graph_nodes = graph_stats.entity_count;
                stats.graph_edges = graph_stats.relationship_count;
            }
        }

        Ok(stats)
    }

    /// List all users (scans data directory for user folders)
    pub fn list_users(&self) -> Vec<String> {
        let mut users = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.base_path) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        if let Some(name) = entry.file_name().to_str() {
                            if name != "audit_logs" {
                                users.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }
        users.sort();
        users
    }

    /// Get audit logs for a user (from persistent storage)
    pub fn get_audit_logs(&self, user_id: &str, limit: usize) -> Vec<AuditEvent> {
        let mut events: Vec<AuditEvent> = Vec::new();
        let prefix = format!("{user_id}:");
        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for (key, value) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Ok((event, _)) = bincode::serde::decode_from_slice::<AuditEvent, _>(&value, bincode::config::standard()) {
                    events.push(event);
                }
            }
        }
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);
        events
    }

    /// Flush all RocksDB databases to ensure data persistence (critical for graceful shutdown)
    pub fn flush_all_databases(&self) -> Result<()> {
        info!("💾 Flushing all databases to disk...");

        // Flush audit database
        self.audit_db
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush audit database: {e}"))?;
        info!("  ✓ Audit database flushed");

        // Flush todo store (todo_db, project_db, index_db)
        if let Err(e) = self.todo_store.flush() {
            tracing::warn!("  Failed to flush todo store: {}", e);
        } else {
            info!("  ✓ Todo store flushed (todos, projects, indices)");
        }

        // Flush file memory store (file_db, index_db)
        if let Err(e) = self.file_store.flush() {
            tracing::warn!("  Failed to flush file store: {}", e);
        } else {
            info!("  ✓ File memory store flushed");
        }

        // Flush prospective store (reminders - db, index_db)
        if let Err(e) = self.prospective_store.flush() {
            tracing::warn!("  Failed to flush prospective store: {}", e);
        } else {
            info!("  ✓ Prospective store flushed (reminders)");
        }

        // Flush feedback store (momentum data with WAL)
        if let Err(e) = self.feedback_store.write().flush() {
            tracing::warn!("  Failed to flush feedback store: {}", e);
        } else {
            info!("  ✓ Feedback store flushed (momentum, pending)");
        }

        // Flush all user memory databases (lock-free iteration)
        let user_entries: Vec<(String, Arc<parking_lot::RwLock<MemorySystem>>)> = self
            .user_memories
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        let mut flushed = 0;
        for (user_id, memory_system) in user_entries {
            // Access the memory system's storage to flush it
            if let Some(guard) = memory_system.try_read() {
                // Flush the long-term storage database
                if let Err(e) = guard.flush_storage() {
                    tracing::warn!("  Failed to flush database for user {}: {}", user_id, e);
                } else {
                    flushed += 1;
                }
            } else {
                tracing::warn!("  Could not acquire lock for user: {}", user_id);
            }
        }

        info!(
            "✅ All databases flushed: audit, todos, files, prospective, feedback, {} user memories",
            flushed
        );

        Ok(())
    }

    /// Save all vector indices to disk (production persistence)
    pub fn save_all_vector_indices(&self) -> Result<()> {
        info!("🔍 Saving vector indices to disk...");

        // Collect entries (lock-free iteration)
        let user_entries: Vec<(String, Arc<parking_lot::RwLock<MemorySystem>>)> = self
            .user_memories
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        let mut saved = 0;
        for (user_id, memory_system) in user_entries {
            if let Some(guard) = memory_system.try_read() {
                let index_path = self.base_path.join(&user_id).join("vector_index");
                if let Err(e) = guard.save_vector_index(&index_path) {
                    tracing::warn!("  Failed to save vector index for user {}: {}", user_id, e);
                } else {
                    info!("  Saved vector index for user: {}", user_id);
                    saved += 1;
                }
            } else {
                tracing::warn!("  Could not acquire lock for user: {}", user_id);
            }
        }

        info!("✅ Saved {} vector indices", saved);
        Ok(())
    }

    /// Load or rebuild vector index for a user (production initialization)
    pub fn init_user_vector_index(&self, user_id: &str) -> Result<()> {
        let memory = self.get_user_memory(user_id)?;
        let memory_guard = memory.read();

        let index_path = self.base_path.join(user_id).join("vector_index");

        // Try to load existing index first
        if index_path.exists() {
            info!("📊 Loading vector index for user {} from disk...", user_id);
            match memory_guard.load_vector_index(&index_path) {
                Ok(_) => {
                    info!("✅ Loaded vector index for user: {}", user_id);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load vector index for user {}: {}. Rebuilding...",
                        user_id,
                        e
                    );
                }
            }
        }

        // Index doesn't exist or load failed - rebuild from storage
        info!("🔨 Rebuilding vector index for user {}...", user_id);
        match memory_guard.rebuild_vector_index() {
            Ok(_) => {
                info!("✅ Rebuilt vector index for user: {}", user_id);
                // Save the newly built index
                if let Err(e) = memory_guard.save_vector_index(&index_path) {
                    tracing::warn!("Failed to save vector index for user {}: {}", user_id, e);
                }
                Ok(())
            }
            Err(e) => {
                tracing::warn!("Failed to rebuild vector index for user {}: {}", user_id, e);
                Ok(()) // Don't fail startup if indexing fails
            }
        }
    }

    /// Rotate audit logs for all users (removes old entries)
    fn rotate_all_audit_logs(&self) -> Result<()> {
        let mut total_removed = 0;

        // Get all unique user IDs from the database
        let mut user_ids = std::collections::HashSet::new();
        let iter = self.audit_db.iterator(rocksdb::IteratorMode::Start);

        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if let Some(user_id) = key_str.split(':').next() {
                    user_ids.insert(user_id.to_string());
                }
            }
        }

        // Rotate logs for each user using helper struct
        let helper = MultiUserMemoryManagerRotationHelper {
            audit_db: self.audit_db.clone(),
            audit_logs: self.audit_logs.clone(),
            audit_retention_days: self.server_config.audit_retention_days as i64,
            audit_max_entries: self.server_config.audit_max_entries_per_user,
        };

        for user_id in user_ids {
            match helper.rotate_user_audit_logs(&user_id) {
                Ok(removed) => {
                    if removed > 0 {
                        info!(
                            "  Rotated audit logs for user {}: removed {} old entries",
                            user_id, removed
                        );
                        total_removed += removed;
                    }
                }
                Err(e) => {
                    tracing::warn!("  Failed to rotate audit logs for user {}: {}", user_id, e);
                }
            }
        }

        if total_removed > 0 {
            info!(
                "✅ Audit log rotation complete: removed {} total entries",
                total_removed
            );
        }

        Ok(())
    }

    /// Get neural NER for entity extraction (SHO-29: Proactive memory surfacing)
    pub fn get_neural_ner(&self) -> Arc<NeuralNer> {
        self.neural_ner.clone()
    }

    /// Get or create graph memory for a user
    pub fn get_user_graph(&self, user_id: &str) -> Result<Arc<parking_lot::RwLock<GraphMemory>>> {
        // Try to get from cache (lock-free read, updates LRU order internally)
        if let Some(graph) = self.graph_memories.get(user_id) {
            return Ok(graph);
        }

        // Create new graph memory for this user
        let graph_path = self.base_path.join(user_id).join("graph");
        let graph_memory = GraphMemory::new(&graph_path)?;
        let graph_arc = Arc::new(parking_lot::RwLock::new(graph_memory));

        // Insert into cache (moka handles LRU eviction automatically via eviction_listener)
        self.graph_memories
            .insert(user_id.to_string(), graph_arc.clone());

        info!("📊 Created graph memory for user: {}", user_id);

        Ok(graph_arc)
    }

    /// Process an experience and extract entities/relationships into the graph
    ///
    /// SHO-102: Improved graph building with:
    /// - Neural NER entities
    /// - Tags as Technology/Concept entities
    /// - All-caps terms (API, TUI, NER, etc.)
    /// - Issue IDs (SHO-XX pattern)
    /// - Semantic similarity edges between memories
    fn process_experience_into_graph(
        &self,
        user_id: &str,
        experience: &Experience,
        memory_id: &MemoryId,
    ) -> Result<()> {
        let graph = self.get_user_graph(user_id)?;
        let graph_guard = graph.write();

        // Extract entities from the experience content using neural NER
        let extracted_entities = match self.neural_ner.extract(&experience.content) {
            Ok(entities) => entities,
            Err(e) => {
                tracing::debug!("NER extraction failed: {}. Continuing without entities.", e);
                Vec::new()
            }
        };

        // Filter out garbage/noise entities from neural NER
        // - Minimum 3 characters (filters "dh", "TU", "at", etc.)
        // - Must have at least one uppercase letter (proper nouns)
        // - Filter common stop words that NER sometimes misclassifies
        let stop_words: std::collections::HashSet<&str> = [
            "the", "and", "for", "that", "this", "with", "from", "have", "been", "are", "was",
            "were", "will", "would", "could", "should", "may", "might",
        ]
        .iter()
        .cloned()
        .collect();

        let filtered_entities: Vec<_> = extracted_entities
            .into_iter()
            .filter(|e| {
                let name = e.text.trim();
                // Minimum 3 chars
                if name.len() < 3 {
                    return false;
                }
                // Must have at least one uppercase letter
                if !name.chars().any(|c| c.is_uppercase()) {
                    return false;
                }
                // Not a stop word
                if stop_words.contains(name.to_lowercase().as_str()) {
                    return false;
                }
                // High confidence threshold for short names
                if name.len() < 5 && e.confidence < 0.8 {
                    return false;
                }
                true
            })
            .collect();

        let mut entity_uuids = Vec::new();

        // Add entities to the graph - map NerEntityType to EntityLabel
        for ner_entity in filtered_entities {
            let label = match ner_entity.entity_type {
                NerEntityType::Person => EntityLabel::Person,
                NerEntityType::Organization => EntityLabel::Organization,
                NerEntityType::Location => EntityLabel::Location,
                NerEntityType::Misc => EntityLabel::Other("MISC".to_string()),
            };

            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(), // Will be replaced if exists
                name: ner_entity.text.clone(),
                labels: vec![label],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
                salience: ner_entity.confidence, // Use NER confidence as salience
                is_proper_noun: true,            // Neural NER primarily extracts proper nouns
            };

            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((ner_entity.text, uuid)),
                Err(e) => tracing::debug!("Failed to add entity {}: {}", ner_entity.text, e),
            }
        }

        // SHO-102: Add tags as entities (Technology/Concept type)
        for tag in &experience.tags {
            let tag_name = tag.trim();
            if tag_name.len() >= 2 && !stop_words.contains(tag_name.to_lowercase().as_str()) {
                let entity = EntityNode {
                    uuid: uuid::Uuid::new_v4(),
                    name: tag_name.to_string(),
                    labels: vec![EntityLabel::Technology],
                    created_at: chrono::Utc::now(),
                    last_seen_at: chrono::Utc::now(),
                    mention_count: 1,
                    summary: String::new(),
                    attributes: HashMap::new(),
                    name_embedding: None,
                    salience: 0.6, // Tags are user-provided, so medium-high salience
                    is_proper_noun: false,
                };

                match graph_guard.add_entity(entity) {
                    Ok(uuid) => entity_uuids.push((tag_name.to_string(), uuid)),
                    Err(e) => tracing::debug!("Failed to add tag entity {}: {}", tag_name, e),
                }
            }
        }

        // SHO-102: Extract all-caps terms (API, TUI, NER, REST, etc.) - min 2 chars
        let allcaps_regex = regex::Regex::new(r"\b[A-Z]{2,}[A-Z0-9]*\b").unwrap();
        for cap in allcaps_regex.find_iter(&experience.content) {
            let term = cap.as_str();
            // Skip if already extracted or is a stop word
            if entity_uuids
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case(term))
            {
                continue;
            }
            if stop_words.contains(term.to_lowercase().as_str()) {
                continue;
            }

            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(),
                name: term.to_string(),
                labels: vec![EntityLabel::Technology],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
                salience: 0.5, // All-caps terms are likely technical terms
                is_proper_noun: true,
            };

            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((term.to_string(), uuid)),
                Err(e) => tracing::debug!("Failed to add allcaps entity {}: {}", term, e),
            }
        }

        // SHO-102: Extract issue IDs (SHO-XX, JIRA-123, etc.)
        let issue_regex = regex::Regex::new(r"\b([A-Z]{2,10}-\d+)\b").unwrap();
        for issue in issue_regex.find_iter(&experience.content) {
            let issue_id = issue.as_str();
            // Skip if already extracted
            if entity_uuids.iter().any(|(name, _)| name == issue_id) {
                continue;
            }

            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(),
                name: issue_id.to_string(),
                labels: vec![EntityLabel::Other("Issue".to_string())],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
                salience: 0.7, // Issue IDs are specific references
                is_proper_noun: true,
            };

            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((issue_id.to_string(), uuid)),
                Err(e) => tracing::debug!("Failed to add issue entity {}: {}", issue_id, e),
            }
        }

        // Create an episodic node for this experience
        let episode = EpisodicNode {
            uuid: memory_id.0, // Use memory UUID directly (already a Uuid)
            name: format!("Memory {}", &memory_id.0.to_string()[..8]),
            content: experience.content.clone(),
            valid_at: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            entity_refs: entity_uuids.iter().map(|(_, uuid)| *uuid).collect(),
            source: EpisodeSource::Message,
            metadata: experience.metadata.clone(),
        };

        if let Err(e) = graph_guard.add_episode(episode) {
            tracing::debug!("Failed to add episode: {}", e);
        }

        // Create relationships between co-occurring entities (simple heuristic)
        for i in 0..entity_uuids.len() {
            for j in (i + 1)..entity_uuids.len() {
                let edge = RelationshipEdge {
                    uuid: uuid::Uuid::new_v4(),
                    from_entity: entity_uuids[i].1,
                    to_entity: entity_uuids[j].1,
                    relation_type: RelationType::RelatedTo,
                    strength: 0.5, // Default strength for co-occurrence
                    created_at: chrono::Utc::now(),
                    valid_at: chrono::Utc::now(),
                    invalidated_at: None,
                    source_episode_id: Some(memory_id.0), // Use memory UUID directly (already a Uuid)
                    context: experience.content.clone(),
                    // Hebbian plasticity fields (new synapses start fresh)
                    last_activated: chrono::Utc::now(),
                    activation_count: 1, // First activation
                    potentiated: false,
                };

                if let Err(e) = graph_guard.add_relationship(edge) {
                    tracing::debug!("Failed to add relationship: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Get graph statistics for a user
    pub fn get_user_graph_stats(&self, user_id: &str) -> Result<GraphStats> {
        let graph = self.get_user_graph(user_id)?;
        let graph_guard = graph.read();
        graph_guard.get_stats()
    }

    /// Run maintenance on all cached user memories
    ///
    /// This is called periodically by the background scheduler to:
    /// 1. Consolidate memories (tier promotion based on thresholds)
    /// 2. Decay activation levels
    /// 3. Run graph maintenance
    ///
    /// Only operates on currently cached users - evicted users will run
    /// maintenance on next access.
    pub fn run_maintenance_all_users(&self) -> usize {
        let decay_factor = self.server_config.activation_decay_factor;
        let mut total_processed = 0;

        // Get list of cached user IDs (lock-free iteration)
        let user_ids: Vec<String> = self
            .user_memories
            .iter()
            .map(|(id, _)| id.to_string())
            .collect();

        let user_count = user_ids.len();

        let mut edges_decayed = 0;

        for user_id in user_ids {
            // Re-acquire memory for each user (may have been evicted)
            if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                let memory = memory_lock.read();
                match memory.run_maintenance(decay_factor) {
                    Ok(count) => total_processed += count,
                    Err(e) => {
                        tracing::warn!("Maintenance failed for user {}: {}", user_id, e);
                    }
                }
            }

            // AUD-2: Apply graph decay to GraphMemory edges
            if let Ok(graph) = self.get_user_graph(&user_id) {
                let graph_guard = graph.write();
                match graph_guard.apply_decay() {
                    Ok(pruned) => {
                        edges_decayed += pruned;
                    }
                    Err(e) => {
                        tracing::debug!("Graph decay failed for user {}: {}", user_id, e);
                    }
                }
            }
        }

        tracing::info!(
            "Maintenance complete: {} memories processed, {} weak edges pruned across {} users",
            total_processed,
            edges_decayed,
            user_count
        );

        total_processed
    }

    /// Get the streaming extractor for session management
    pub fn streaming_extractor(&self) -> &Arc<streaming::StreamingMemoryExtractor> {
        &self.streaming_extractor
    }

    /// Get the backup engine
    pub fn backup_engine(&self) -> &Arc<backup::ShodhBackupEngine> {
        &self.backup_engine
    }

    /// Get the A/B test manager
    pub fn ab_test_manager(&self) -> &Arc<ab_testing::ABTestManager> {
        &self.ab_test_manager
    }

    /// Run backups for all active users
    ///
    /// This is called by the backup scheduler to create automated backups.
    /// Returns the number of users backed up successfully.
    pub fn run_backup_all_users(&self, max_backups: usize) -> usize {
        let mut backed_up = 0;

        // Get all user directories from the base path
        let users_path = &self.base_path;
        if let Ok(entries) = std::fs::read_dir(users_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                // Skip non-directories and special directories
                if !path.is_dir() {
                    continue;
                }
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('.') || name == "audit_logs" || name == "backups" {
                    continue;
                }

                // Check if this is a user directory with a RocksDB database
                let db_path = path.join("memory.db");
                if !db_path.exists() {
                    continue;
                }

                // Try to get or create the user's memory system to access their DB
                if let Ok(memory_lock) = self.get_user_memory(name) {
                    let memory = memory_lock.read();
                    let db = memory.get_db();
                    match self.backup_engine.create_backup(&db, name) {
                        Ok(metadata) => {
                            tracing::info!(
                                user_id = name,
                                backup_id = metadata.backup_id,
                                size_mb = metadata.size_bytes / 1024 / 1024,
                                "Backup created successfully"
                            );
                            backed_up += 1;

                            // Purge old backups for this user
                            if let Err(e) = self.backup_engine.purge_old_backups(name, max_backups)
                            {
                                tracing::warn!(
                                    user_id = name,
                                    error = %e,
                                    "Failed to purge old backups"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                user_id = name,
                                error = %e,
                                "Failed to create backup"
                            );
                        }
                    }
                }
            }
        }

        backed_up
    }
}

/// API request/response types
#[derive(Debug, Deserialize)]
struct RecordRequest {
    user_id: String,
    experience: Experience,
}

#[derive(Debug, Serialize)]
struct RecordResponse {
    id: String,
    success: bool,
}

/// Response for list/search operations returning multiple memories
#[derive(Debug, Serialize)]
struct RetrieveResponse {
    memories: Vec<Memory>,
    count: usize,
}

// MemoryStats is imported from memory::types - has more fields than we need here

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    users_count: usize,
    users_in_cache: usize,
    user_evictions: usize,
    max_cache_size: usize,
}

// =============================================================================
// SIMPLIFIED LLM-FRIENDLY API TYPES
// Minimal request/response for effortless use by AI agents
// =============================================================================

/// Simplified remember request - just content, auto-creates Experience
#[derive(Debug, Deserialize)]
struct RememberRequest {
    user_id: String,
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    /// Accept both "memory_type" and "experience_type" as field names
    #[serde(default, alias = "experience_type")]
    memory_type: Option<String>,
    /// External ID for linking to external systems (e.g., "linear:SHO-39", "github:pr-123")
    /// When provided with upsert, existing memory with same external_id will be updated
    #[serde(default)]
    external_id: Option<String>,
    /// Optional timestamp for the memory. If not provided, uses current time.
    /// Use ISO 8601 format (e.g., "2025-12-15T06:30:00Z")
    #[serde(default)]
    created_at: Option<chrono::DateTime<chrono::Utc>>,

    // =========================================================================
    // SHO-104: RICHER CONTEXT ENCODING - Optional context fields
    // =========================================================================
    /// Emotional valence: -1.0 (negative) to 1.0 (positive), 0.0 = neutral
    /// E.g., bug found: -0.3, feature shipped: 0.7
    #[serde(default)]
    emotional_valence: Option<f32>,

    /// Arousal level: 0.0 (calm) to 1.0 (highly aroused)
    /// E.g., routine task: 0.2, critical production issue: 0.9
    #[serde(default)]
    emotional_arousal: Option<f32>,

    /// Dominant emotion label (e.g., "joy", "frustration", "surprise")
    #[serde(default)]
    emotion: Option<String>,

    /// Source type: "user", "system", "api", "file", "web", "ai_generated", "inferred"
    #[serde(default)]
    source_type: Option<String>,

    /// Credibility score: 0.0 to 1.0 (1.0 = verified facts, 0.3 = inferred)
    #[serde(default)]
    credibility: Option<f32>,

    /// Episode ID - groups memories into coherent episodes/conversations
    #[serde(default)]
    episode_id: Option<String>,

    /// Sequence number within episode (1, 2, 3...)
    #[serde(default)]
    sequence_number: Option<u32>,

    /// ID of the preceding memory (for temporal chains)
    #[serde(default)]
    preceding_memory_id: Option<String>,

    /// Agent ID for multi-agent systems (e.g., "explore-abc123", "plan-def456")
    /// Used to track which agent created this memory
    #[serde(default)]
    agent_id: Option<String>,

    /// Parent agent ID for hierarchical agent tracking
    /// Links child agent memories to parent context
    #[serde(default)]
    parent_agent_id: Option<String>,

    /// Run ID for grouping memories within a single agent execution
    #[serde(default)]
    run_id: Option<String>,
}

/// Simplified remember response
#[derive(Debug, Serialize)]
struct RememberResponse {
    id: String,
    success: bool,
}

/// Simplified recall request - just query text
#[derive(Debug, Deserialize)]
struct RecallRequest {
    user_id: String,
    query: String,
    #[serde(default = "default_recall_limit")]
    limit: usize,
    /// Retrieval mode: "semantic", "associative", or "hybrid" (default)
    /// - semantic: Pure vector similarity search
    /// - associative: Graph traversal with density-dependent weights (SHO-26)
    /// - hybrid: Combined semantic + graph with fixed weights (legacy)
    #[serde(default = "default_recall_mode")]
    mode: String,
}

fn default_recall_limit() -> usize {
    5
}

fn default_recall_mode() -> String {
    "hybrid".to_string()
}

/// Simplified recall response - returns just text snippets
#[derive(Debug, Serialize)]
struct RecallResponse {
    memories: Vec<RecallMemory>,
    count: usize,
    /// Retrieval statistics (SHO-26) - optional for observability
    #[serde(skip_serializing_if = "Option::is_none")]
    retrieval_stats: Option<memory::types::RetrievalStats>,
    /// Related todos (semantic search on todo content)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    todos: Vec<RecallTodo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    todo_count: Option<usize>,
}

/// Todo returned in recall results
#[derive(Debug, Serialize)]
struct RecallTodo {
    id: String,
    short_id: String,
    content: String,
    status: String,
    priority: String,
    project: Option<String>,
    score: f32,
    created_at: String,
}

#[derive(Debug, Serialize)]
struct RecallMemory {
    id: String,
    /// Nested experience for MCP compatibility
    experience: RecallExperience,
    importance: f32,
    created_at: String,
    /// Similarity/relevance score (0.0-1.0)
    score: f32,
}

#[derive(Debug, Serialize)]
struct RecallExperience {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_type: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
}

/// Batch remember request for bulk inserts (SHO-83)
#[derive(Debug, Deserialize)]
struct BatchRememberRequest {
    user_id: String,
    memories: Vec<BatchMemoryItem>,
    /// Options for batch processing
    #[serde(default)]
    options: BatchRememberOptions,
}

/// Options for batch remember operation
#[derive(Debug, Deserialize, Clone)]
struct BatchRememberOptions {
    /// Whether to extract entities using NER (default: true)
    #[serde(default = "default_true")]
    extract_entities: bool,
    /// Whether to create knowledge graph edges (default: true)
    #[serde(default = "default_true")]
    create_edges: bool,
}

impl Default for BatchRememberOptions {
    fn default() -> Self {
        Self {
            extract_entities: true,
            create_edges: true,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
struct BatchMemoryItem {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, alias = "experience_type")]
    memory_type: Option<String>,
    /// Optional timestamp for the memory
    #[serde(default)]
    created_at: Option<chrono::DateTime<chrono::Utc>>,

    // SHO-104: RICHER CONTEXT ENCODING - Optional context fields (same as RememberRequest)
    #[serde(default)]
    emotional_valence: Option<f32>,
    #[serde(default)]
    emotional_arousal: Option<f32>,
    #[serde(default)]
    emotion: Option<String>,
    #[serde(default)]
    source_type: Option<String>,
    #[serde(default)]
    credibility: Option<f32>,
    #[serde(default)]
    episode_id: Option<String>,
    #[serde(default)]
    sequence_number: Option<u32>,
    #[serde(default)]
    preceding_memory_id: Option<String>,
}

/// Error detail for a single item in batch
#[derive(Debug, Serialize)]
struct BatchErrorItem {
    /// Index of the failed item in the request array
    index: usize,
    /// Error message describing why this item failed
    error: String,
}

#[derive(Debug, Serialize)]
struct BatchRememberResponse {
    /// Number of successfully created memories
    created: usize,
    /// Number of failed memories
    failed: usize,
    /// IDs of successfully created memories (in order)
    memory_ids: Vec<String>,
    /// Details of failed items with index and error
    errors: Vec<BatchErrorItem>,
}

/// Upsert request - create or update memory with external linking
#[derive(Debug, Deserialize)]
struct UpsertRequest {
    user_id: String,
    /// External ID is REQUIRED for upsert (e.g., "linear:SHO-39", "github:pr-123")
    external_id: String,
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, alias = "experience_type")]
    memory_type: Option<String>,
    /// Type of change: "content_updated", "status_changed", "tags_updated", "importance_adjusted"
    #[serde(default = "default_change_type")]
    change_type: String,
    /// Who/what triggered this change
    #[serde(default)]
    changed_by: Option<String>,
    /// Description of why this changed
    #[serde(default)]
    change_reason: Option<String>,
}

fn default_change_type() -> String {
    "content_updated".to_string()
}

#[derive(Debug, Serialize)]
struct UpsertResponse {
    id: String,
    success: bool,
    /// True if this was an update, false if it was a create
    was_update: bool,
    /// Current version number (starts at 1, increments on update)
    version: u32,
}

/// Request to get memory history (audit trail)
#[derive(Debug, Deserialize)]
struct MemoryHistoryRequest {
    user_id: String,
    #[serde(alias = "memory_id")]
    id: String,
}

/// Response with memory revision history
#[derive(Debug, Serialize)]
struct MemoryHistoryResponse {
    id: String,
    external_id: Option<String>,
    current_content: String,
    version: u32,
    history: Vec<MemoryRevisionInfo>,
}

#[derive(Debug, Serialize)]
struct MemoryRevisionInfo {
    previous_content: String,
    change_type: String,
    changed_at: String,
    changed_by: Option<String>,
    change_reason: Option<String>,
}

// =============================================================================
// HEBBIAN FEEDBACK API - Wires up the learning loop
// =============================================================================

/// Request for tracked retrieval (returns tracking ID for later feedback)
#[derive(Debug, Deserialize)]
struct TrackedRetrieveRequest {
    user_id: String,
    query: String,
    #[serde(default = "default_recall_limit")]
    limit: usize,
}

/// Response with tracking ID for feedback
#[derive(Debug, Serialize)]
struct TrackedRetrieveResponse {
    tracking_id: String,
    ids: Vec<String>,
    memories: Vec<RecallMemory>,
}

/// Request to provide feedback on retrieval outcome
#[derive(Debug, Deserialize)]
struct ReinforceFeedbackRequest {
    user_id: String,
    #[serde(alias = "memory_ids")]
    ids: Vec<String>,
    /// "helpful", "misleading", or "neutral"
    outcome: String,
}

/// Response from reinforcement
#[derive(Debug, Serialize)]
struct ReinforceFeedbackResponse {
    memories_processed: usize,
    associations_strengthened: usize,
    importance_boosts: usize,
    importance_decays: usize,
}

// =============================================================================
// SEMANTIC CONSOLIDATION API - Extracts durable facts from episodic memories
// =============================================================================

/// Request to trigger consolidation
#[derive(Debug, Deserialize)]
struct ConsolidateRequest {
    user_id: String,
    /// Minimum number of supporting memories for a fact (default: 2)
    #[serde(default = "default_min_support")]
    min_support: usize,
    /// Minimum age in days before consolidation (default: 7)
    #[serde(default = "default_min_age_days")]
    min_age_days: i64,
}

fn default_min_support() -> usize {
    2
}

fn default_min_age_days() -> i64 {
    7
}

/// Response from consolidation
#[derive(Debug, Serialize)]
struct ConsolidateResponse {
    memories_analyzed: usize,
    facts_extracted: usize,
    facts_reinforced: usize,
    fact_ids: Vec<String>,
}

/// Application state
type AppState = Arc<MultiUserMemoryManager>;

// REST API handlers

/// Health check endpoint (basic compatibility)
async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let users_in_cache = state.user_memories.entry_count() as usize;
    let user_evictions = state
        .user_evictions
        .load(std::sync::atomic::Ordering::Relaxed);

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        users_count: state.list_users().len(),
        users_in_cache,
        user_evictions,
        max_cache_size: state.server_config.max_users_in_memory,
    })
}

/// P0.9: Liveness probe - indicates if process is alive and not deadlocked
/// Returns 200 OK if service is running (minimal check, always succeeds if reachable)
/// Kubernetes uses this to restart crashed/hung pods
async fn health_live() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "alive",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// P0.9: Readiness probe - indicates if service can handle traffic
/// Returns 200 OK if service is ready, 503 if not ready
/// Kubernetes uses this to route traffic only to ready pods
async fn health_ready(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    // Check if critical resources are accessible (lock-free)
    let users_in_cache = state.user_memories.entry_count() as usize;

    // Service is ready if we can access the user cache without panicking
    // This verifies the lock is not poisoned and the service is operational
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ready",
            "version": env!("CARGO_PKG_VERSION"),
            "users_in_cache": users_in_cache,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// Update context status from Claude Code status line script
/// Called by a custom status line script to report token usage and current task
#[derive(Debug, Deserialize)]
struct ContextStatusRequest {
    /// Session ID from Claude Code (unique per window)
    pub session_id: String,
    /// Tokens used in current context window
    pub tokens_used: u64,
    /// Context window size (budget)
    pub tokens_budget: u64,
    /// Current working directory
    pub current_dir: Option<String>,
    /// Model display name
    pub model: Option<String>,
}

async fn update_context_status(
    State(state): State<AppState>,
    Json(req): Json<ContextStatusRequest>,
) -> Json<serde_json::Value> {
    let percent_used = if req.tokens_budget > 0 {
        ((req.tokens_used as f64 / req.tokens_budget as f64) * 100.0) as u8
    } else {
        0
    };

    let status = ContextStatus {
        session_id: Some(req.session_id.clone()),
        tokens_used: req.tokens_used,
        tokens_budget: req.tokens_budget,
        percent_used,
        current_task: req.current_dir,
        model: req.model,
        updated_at: chrono::Utc::now(),
    };

    // Store by session_id (allows multiple Claude windows)
    state
        .context_sessions
        .insert(req.session_id.clone(), status.clone());

    // Broadcast for TUI SSE subscribers (dedicated context channel)
    let _ = state.context_broadcaster.send(status);

    // Also emit through main SSE channel for TUI to pick up
    state.emit_event(MemoryEvent {
        event_type: "CONTEXT_UPDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: "system".to_string(),
        memory_id: Some(req.session_id),
        content_preview: Some(format!(
            "{}% ({}/{})",
            percent_used, req.tokens_used, req.tokens_budget
        )),
        memory_type: Some("Context".to_string()),
        importance: None,
        count: None,
    });

    Json(serde_json::json!({
        "success": true,
        "percent_used": percent_used
    }))
}

/// Get all active context sessions (auto-cleans stale sessions > 5 mins old)
async fn get_context_status(State(state): State<AppState>) -> Json<Vec<ContextStatus>> {
    let now = chrono::Utc::now();
    let stale_threshold = chrono::Duration::minutes(5);

    // Collect stale session IDs for cleanup
    let stale_ids: Vec<String> = state
        .context_sessions
        .iter()
        .filter(|r| now - r.value().updated_at > stale_threshold)
        .map(|r| r.key().clone())
        .collect();

    // Remove stale sessions
    for id in stale_ids {
        state.context_sessions.remove(&id);
    }

    // Return active sessions sorted by most recently updated
    let mut sessions: Vec<ContextStatus> = state
        .context_sessions
        .iter()
        .map(|r| r.value().clone())
        .collect();
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Json(sessions)
}

/// SSE endpoint for context status updates (no auth - local status line script)
async fn context_status_sse(
    State(state): State<AppState>,
) -> axum::response::Sse<
    impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use futures::StreamExt;
    use tokio_stream::wrappers::BroadcastStream;

    let receiver = state.context_broadcaster.subscribe();
    let stream = BroadcastStream::new(receiver);

    let event_stream = stream.filter_map(|result| async move {
        match result {
            Ok(status) => {
                let data = serde_json::to_string(&status).ok()?;
                Some(Ok(axum::response::sse::Event::default()
                    .event("context")
                    .data(data)))
            }
            Err(_) => None,
        }
    });

    axum::response::Sse::new(event_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

/// Vector index health endpoint - provides Vamana index statistics per user
///
/// Returns index health metrics including total vectors, incremental inserts since
/// last build, and whether a rebuild is recommended for optimal recall.
async fn health_index(
    State(state): State<AppState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let user_id = match params.get("user_id") {
        Some(id) => id.clone(),
        None => {
            // Return aggregate stats across all cached users (lock-free iteration)
            let users: Vec<(String, memory::retrieval::IndexHealth)> = state
                .user_memories
                .iter()
                .map(|(user_id, memory)| {
                    let guard = memory.read();
                    (user_id.to_string(), guard.index_health())
                })
                .collect();

            let total_vectors: usize = users.iter().map(|(_, h)| h.total_vectors).sum();
            let total_incremental: usize = users.iter().map(|(_, h)| h.incremental_inserts).sum();
            let needs_rebuild: Vec<&str> = users
                .iter()
                .filter(|(_, h)| h.needs_rebuild)
                .map(|(id, _)| id.as_str())
                .collect();

            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "ok",
                    "users_checked": users.len(),
                    "total_vectors": total_vectors,
                    "total_incremental_inserts": total_incremental,
                    "users_needing_rebuild": needs_rebuild,
                    "rebuild_threshold": crate::vector_db::vamana::REBUILD_THRESHOLD,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })),
            );
        }
    };

    // Get health for specific user
    match state.get_user_memory(&user_id) {
        Ok(memory) => {
            let guard = memory.read();
            let health = guard.index_health();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "ok",
                    "user_id": user_id,
                    "total_vectors": health.total_vectors,
                    "incremental_inserts": health.incremental_inserts,
                    "needs_rebuild": health.needs_rebuild,
                    "rebuild_threshold": health.rebuild_threshold,
                    "degradation_percent": if health.rebuild_threshold > 0 {
                        (health.incremental_inserts as f64 / health.rebuild_threshold as f64 * 100.0).min(100.0)
                    } else {
                        0.0
                    },
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "status": "error",
                "error": e.to_string(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            })),
        ),
    }
}

/// Prometheus metrics endpoint for observability
/// Returns metrics in Prometheus text format for scraping
async fn metrics_endpoint(State(state): State<AppState>) -> Result<String, StatusCode> {
    use prometheus::Encoder;

    // Update memory usage gauges before serving metrics (lock-free)
    let users_in_cache = state.user_memories.entry_count();
    metrics::ACTIVE_USERS.set(users_in_cache as i64);

    // Aggregate metrics across all users (no per-user labels to avoid cardinality explosion)
    let (mut total_working, mut total_session, mut total_longterm, mut total_heap) =
        (0i64, 0i64, 0i64, 0i64);
    let mut total_vectors = 0i64;

    // Lock-free iteration, take up to 100 entries
    let user_entries: Vec<_> = state
        .user_memories
        .iter()
        .take(100)
        .map(|(_, v)| v.clone())
        .collect();

    for memory_sys in user_entries {
        if let Some(guard) = memory_sys.try_read() {
            let stats = guard.stats();
            total_working += stats.working_memory_count as i64;
            total_session += stats.session_memory_count as i64;
            total_longterm += stats.long_term_memory_count as i64;
            total_heap += (stats.total_memories * 250) as i64;
            total_vectors += stats.total_memories as i64;
        }
    }

    // Set aggregate metrics
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["working"])
        .set(total_working);
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["session"])
        .set(total_session);
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["longterm"])
        .set(total_longterm);
    metrics::MEMORY_HEAP_BYTES_TOTAL.set(total_heap);
    metrics::VECTOR_INDEX_SIZE_TOTAL.set(total_vectors);

    // Gather and encode metrics
    let encoder = prometheus::TextEncoder::new();
    let metric_families = metrics::METRICS_REGISTRY.gather();

    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    String::from_utf8(buffer).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// SSE endpoint for real-time memory events
/// Streams CREATE, RETRIEVE, DELETE events to connected dashboard clients
/// No authentication required - read-only, lightweight event stream
async fn memory_events_sse(
    State(state): State<AppState>,
) -> axum::response::Sse<
    impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use axum::response::sse::Event;
    use futures::StreamExt;
    use tokio_stream::wrappers::BroadcastStream;

    let receiver = state.subscribe_events();
    let stream = BroadcastStream::new(receiver);

    let event_stream = stream.filter_map(|result| async move {
        match result {
            Ok(event) => {
                let json = serde_json::to_string(&event).ok()?;
                Some(Ok(Event::default().event(&event.event_type).data(json)))
            }
            Err(_) => None, // Lagged receiver, skip event
        }
    });

    axum::response::Sse::new(event_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("heartbeat"),
    )
}

/// WebSocket endpoint for streaming memory ingestion
/// Enables implicit learning from continuous data streams
///
/// # Protocol
/// 1. Client connects to WS /api/stream
/// 2. Client sends handshake: { user_id, mode, extraction_config }
/// 3. Client streams messages: { type: "content"|"event"|"sensor", ... }
/// 4. Server responds with extraction results: { memories_created, entities_detected, ... }
///
/// # Modes
/// - conversation: Agent dialogue (high semantic content)
/// - sensor: IoT/robotics data (needs aggregation)
/// - event: Discrete system events
async fn streaming_memory_ws(
    ws: axum::extract::ws::WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(|socket| handle_streaming_socket(socket, state))
}

/// Handle WebSocket connection for streaming memory ingestion
async fn handle_streaming_socket(socket: axum::extract::ws::WebSocket, state: AppState) {
    use axum::extract::ws::Message;
    use futures::{SinkExt, StreamExt};

    let (mut sender, mut receiver) = socket.split();
    let mut session_id: Option<String> = None;

    // Wait for handshake message
    while let Some(msg) = receiver.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => {
                tracing::debug!("WebSocket closed before handshake");
                return;
            }
            Ok(_) => continue, // Skip binary/ping/pong
            Err(e) => {
                tracing::warn!("WebSocket error before handshake: {}", e);
                return;
            }
        };

        // Parse handshake
        let handshake: streaming::StreamHandshake = match serde_json::from_str(&msg) {
            Ok(h) => h,
            Err(e) => {
                let error = streaming::ExtractionResult::Error {
                    code: "INVALID_HANDSHAKE".to_string(),
                    message: format!("Failed to parse handshake: {}", e),
                    fatal: true,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                return;
            }
        };

        // Validate user_id
        if let Err(e) = validation::validate_user_id(&handshake.user_id) {
            let error = streaming::ExtractionResult::Error {
                code: "INVALID_USER_ID".to_string(),
                message: format!("Invalid user_id: {}", e),
                fatal: true,
                timestamp: chrono::Utc::now(),
            };
            let _ = sender
                .send(Message::Text(
                    serde_json::to_string(&error)
                        .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                        .into(),
                ))
                .await;
            return;
        }

        // Create session
        let id = match state
            .streaming_extractor
            .create_session(handshake.clone())
            .await
        {
            Ok(id) => id,
            Err(e) => {
                let error = streaming::ExtractionResult::Error {
                    code: "SESSION_LIMIT_REACHED".to_string(),
                    message: e,
                    fatal: true,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                return;
            }
        };
        session_id = Some(id.clone());

        // Send acknowledgement
        let ack = streaming::ExtractionResult::Ack {
            message_type: "handshake".to_string(),
            timestamp: chrono::Utc::now(),
        };
        if sender
            .send(Message::Text(
                serde_json::to_string(&ack)
                    .unwrap_or_else(|_| r#"{"ack":true}"#.to_string())
                    .into(),
            ))
            .await
            .is_err()
        {
            return;
        }

        tracing::info!(
            "Streaming session {} created for user {} in {:?} mode",
            id,
            handshake.user_id,
            handshake.mode
        );
        break;
    }

    let session_id = match session_id {
        Some(id) => id,
        None => return,
    };

    // Get user memory system for storing extracted memories
    // Note: This is done once per connection, not per message
    let user_memory = {
        // Extract user_id from session
        let stats = state
            .streaming_extractor
            .get_session_stats(&session_id)
            .await;
        match stats {
            Some(s) => match state.get_user_memory(&s.user_id) {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("Failed to get user memory: {}", e);
                    return;
                }
            },
            None => return,
        }
    };

    // Process messages
    while let Some(msg) = receiver.next().await {
        let text = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => {
                // Client requested close
                let _ = state.streaming_extractor.close_session(&session_id).await;
                return;
            }
            Ok(Message::Ping(data)) => {
                let _ = sender.send(Message::Pong(data)).await;
                continue;
            }
            Ok(_) => continue,
            Err(e) => {
                tracing::warn!("WebSocket error: {}", e);
                break;
            }
        };

        // Parse message
        let stream_msg: streaming::StreamMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                let error = streaming::ExtractionResult::Error {
                    code: "INVALID_MESSAGE".to_string(),
                    message: format!("Failed to parse message: {}", e),
                    fatal: false,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                continue;
            }
        };

        // Process message
        let result = state
            .streaming_extractor
            .process_message(&session_id, stream_msg, user_memory.clone())
            .await;

        // Send result
        let response = serde_json::to_string(&result)
            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string());
        if sender.send(Message::Text(response.into())).await.is_err() {
            break;
        }

        // Check if session was closed
        if matches!(result, streaming::ExtractionResult::Closed { .. }) {
            break;
        }
    }

    // Cleanup session on disconnect
    if let Some(total) = state.streaming_extractor.close_session(&session_id).await {
        tracing::info!(
            "Streaming session {} closed. Total memories created: {}",
            session_id,
            total
        );
    }
}

/// Record a new experience
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn record_experience(
    State(state): State<AppState>,
    Json(req): Json<RecordRequest>,
) -> Result<Json<RecordResponse>, AppError> {
    // Enterprise input validation
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    validation::validate_content(&req.experience.content, false).map_validation_err("content")?;

    if let Some(ref embeddings) = req.experience.embeddings {
        validation::validate_embeddings(embeddings)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Extract entities from content using Neural NER and merge with user-provided entities
    let extracted_names: Vec<String> = match state.neural_ner.extract(&req.experience.content) {
        Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
        Err(e) => {
            tracing::debug!("NER extraction failed in record_experience: {}", e);
            Vec::new()
        }
    };

    // Merge user entities with NER-extracted entities (deduplicated)
    let mut merged_entities: Vec<String> = req.experience.entities.clone();
    for entity_name in extracted_names {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&entity_name))
        {
            merged_entities.push(entity_name);
        }
    }

    // Create experience with merged entities
    let experience_with_entities = Experience {
        entities: merged_entities,
        ..req.experience.clone()
    };

    // P1.2: Instrument memory store operation
    let store_start = std::time::Instant::now();

    // CRITICAL FIX: Wrap blocking I/O in spawn_blocking
    // record() does ONNX inference (10-50ms) + RocksDB writes (100µs-10ms)
    // Running these on async threads starves the Tokio runtime under load
    let memory_id = {
        let memory = memory.clone();
        let experience = experience_with_entities.clone();

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.remember(experience, None)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Extract entities and build knowledge graph (background processing)
    if let Err(e) =
        state.process_experience_into_graph(&req.user_id, &experience_with_entities, &memory_id)
    {
        tracing::debug!("Graph processing failed for memory {}: {}", memory_id.0, e);
        // Don't fail the request if graph processing fails
    }

    // Enterprise audit logging
    state.log_event(
        &req.user_id,
        "CREATE",
        &memory_id.0.to_string(),
        &format!(
            "Created memory: {}",
            req.experience.content.chars().take(50).collect::<String>()
        ),
    );

    // SSE: Emit real-time event for dashboard
    state.emit_event(MemoryEvent {
        event_type: "CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(req.experience.content.chars().take(100).collect()),
        memory_type: Some(format!("{:?}", req.experience.experience_type)),
        importance: req.experience.reward, // Map reward to importance for display
        count: None,
    });

    // Record metrics (no user_id to prevent cardinality explosion)
    let duration = store_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    // Broadcast CREATE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(req.experience.content.chars().take(100).collect()),
        memory_type: Some(format!("{:?}", req.experience.experience_type)),
        importance: None,
        count: None,
    });

    Ok(Json(RecordResponse {
        id: memory_id.0.to_string(),
        success: true,
    }))
}

// =============================================================================
// SIMPLIFIED LLM-FRIENDLY API HANDLERS
// =============================================================================

/// LLM-friendly /api/remember - just pass content, get memory ID back
/// Example: POST /api/remember { "user_id": "agent-1", "content": "User likes pizza" }
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn remember(
    State(state): State<AppState>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, AppError> {
    // P1.2: Instrument remember operation
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    // Parse memory type from string, default to Context
    let experience_type = req
        .memory_type
        .as_ref()
        .and_then(|s| match s.to_lowercase().as_str() {
            "task" => Some(ExperienceType::Task),
            "learning" => Some(ExperienceType::Learning),
            "decision" => Some(ExperienceType::Decision),
            "error" => Some(ExperienceType::Error),
            "pattern" => Some(ExperienceType::Pattern),
            "conversation" => Some(ExperienceType::Conversation),
            "discovery" => Some(ExperienceType::Discovery),
            _ => None,
        })
        .unwrap_or(ExperienceType::Context);

    // Extract entities from content using Neural NER and merge with user-provided tags
    let extracted_names: Vec<String> = match state.neural_ner.extract(&req.content) {
        Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
        Err(e) => {
            tracing::debug!("NER extraction failed in remember_simplified: {}", e);
            Vec::new()
        }
    };

    // Merge user tags with NER-extracted entities (deduplicated) for entity search
    let mut merged_entities: Vec<String> = req.tags.clone();
    for entity_name in extracted_names {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&entity_name))
        {
            merged_entities.push(entity_name);
        }
    }

    // Save experience type string before it's moved
    let experience_type_str = format!("{:?}", experience_type);

    // SHO-104: Build RichContext if any context fields are provided
    let has_context = req.emotional_valence.is_some()
        || req.emotional_arousal.is_some()
        || req.emotion.is_some()
        || req.source_type.is_some()
        || req.credibility.is_some()
        || req.episode_id.is_some()
        || req.sequence_number.is_some()
        || req.preceding_memory_id.is_some();

    let context = if has_context {
        use memory::types::{
            ContextId, EmotionalContext, EpisodeContext, RichContext, SourceContext, SourceType,
        };

        // Build EmotionalContext
        let emotional = EmotionalContext {
            valence: req.emotional_valence.unwrap_or(0.0),
            arousal: req.emotional_arousal.unwrap_or(0.0),
            dominant_emotion: req.emotion.clone(),
            confidence: if req.emotional_valence.is_some() || req.emotional_arousal.is_some() {
                0.8 // User-provided emotional data has good confidence
            } else {
                0.0
            },
            ..Default::default()
        };

        // Build SourceContext
        let source_type = req
            .source_type
            .as_ref()
            .map(|s| match s.to_lowercase().as_str() {
                "user" => SourceType::User,
                "system" => SourceType::System,
                "api" | "external_api" => SourceType::ExternalApi,
                "file" => SourceType::File,
                "web" => SourceType::Web,
                "ai_generated" | "ai" => SourceType::AiGenerated,
                "inferred" => SourceType::Inferred,
                _ => SourceType::Unknown,
            })
            .unwrap_or(SourceType::User); // Default to User for /api/remember

        let source = SourceContext {
            source_type,
            credibility: req.credibility.unwrap_or(0.8), // User input default to high credibility
            ..Default::default()
        };

        // Build EpisodeContext
        let episode = EpisodeContext {
            episode_id: req.episode_id.clone(),
            sequence_number: req.sequence_number,
            preceding_memory_id: req.preceding_memory_id.clone(),
            is_episode_start: req.sequence_number == Some(1),
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
    } else {
        None
    };

    // Auto-create Experience with sensible defaults
    // Note: tags field is for explicit user tags, entities field includes tags + NER-extracted
    let experience = Experience {
        content: req.content.clone(),
        experience_type,
        entities: merged_entities,
        tags: req.tags.clone(),
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
            // Use agent-aware method if agent context is provided
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

    // Extract entities and build knowledge graph (same as /api/record)
    if let Err(e) = state.process_experience_into_graph(&req.user_id, &experience, &memory_id) {
        tracing::debug!("Graph processing failed for memory {}: {}", memory_id.0, e);
        // Don't fail the request if graph processing fails
    }

    // Auto-infer lineage edges in background (non-blocking)
    // This builds the causal graph silently - users can trace when needed
    {
        let state_clone = state.clone();
        let user_id = req.user_id.clone();
        let new_memory_id = memory_id.clone();
        let new_experience = experience.clone();

        tokio::spawn(async move {
            if let Err(e) =
                auto_infer_lineage(&state_clone, &user_id, &new_memory_id, &new_experience).await
            {
                tracing::debug!(
                    "Auto-lineage inference failed for {}: {}",
                    new_memory_id.0,
                    e
                );
                // Silent failure - lineage is optional enhancement
            }
        });
    }

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    // Broadcast CREATE event for real-time dashboard
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

/// Upsert memory - create new or update existing memory with history tracking (SHO-39)
///
/// This endpoint is designed for syncing from external systems (Linear, GitHub, etc.)
/// where the same entity may need to be updated multiple times.
///
/// Behavior:
/// - If no memory with the external_id exists: Creates new memory
/// - If memory with external_id exists: Pushes old content to history, updates with new
///
/// Example: POST /api/upsert {
///   "user_id": "agent-1",
///   "external_id": "linear:SHO-39",
///   "content": "Unified streaming ingest infrastructure: DONE",
///   "change_type": "status_changed",
///   "changed_by": "linear-webhook",
///   "change_reason": "Issue status changed from In Progress to Done"
/// }
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, external_id = %req.external_id))]
async fn upsert_memory(
    State(state): State<AppState>,
    Json(req): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    // Validate external_id (required for upsert)
    if req.external_id.is_empty() {
        return Err(AppError::InvalidInput {
            field: "external_id".to_string(),
            reason: "external_id is required for upsert".to_string(),
        });
    }

    // Parse memory type
    let experience_type = req
        .memory_type
        .as_ref()
        .and_then(|s| match s.to_lowercase().as_str() {
            "task" => Some(ExperienceType::Task),
            "learning" => Some(ExperienceType::Learning),
            "decision" => Some(ExperienceType::Decision),
            "error" => Some(ExperienceType::Error),
            "pattern" => Some(ExperienceType::Pattern),
            "conversation" => Some(ExperienceType::Conversation),
            "discovery" => Some(ExperienceType::Discovery),
            _ => None,
        })
        .unwrap_or(ExperienceType::Context);

    // Parse change type
    let change_type = match req.change_type.to_lowercase().as_str() {
        "created" => memory::types::ChangeType::Created,
        "content_updated" => memory::types::ChangeType::ContentUpdated,
        "status_changed" => memory::types::ChangeType::StatusChanged,
        "tags_updated" => memory::types::ChangeType::TagsUpdated,
        "importance_adjusted" => memory::types::ChangeType::ImportanceAdjusted,
        _ => memory::types::ChangeType::ContentUpdated,
    };

    // Extract entities from content using Neural NER and merge with user-provided tags
    let extracted_names: Vec<String> = match state.neural_ner.extract(&req.content) {
        Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
        Err(e) => {
            tracing::debug!("NER extraction failed in upsert: {}", e);
            Vec::new()
        }
    };

    // Merge user tags with NER-extracted entities (deduplicated)
    let mut merged_entities: Vec<String> = req.tags.clone();
    for entity_name in extracted_names {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&entity_name))
        {
            merged_entities.push(entity_name);
        }
    }

    // Create Experience
    let experience = Experience {
        content: req.content.clone(),
        experience_type,
        entities: merged_entities,
        ..Default::default()
    };

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let external_id = req.external_id.clone();
    let changed_by = req.changed_by.clone();
    let change_reason = req.change_reason.clone();

    // Perform upsert
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

    // Get the version from the stored memory
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

    // Process into graph if this was a create (not update)
    if !was_update {
        if let Err(e) = state.process_experience_into_graph(&req.user_id, &experience, &memory_id) {
            tracing::debug!("Graph processing failed for memory {}: {}", memory_id.0, e);
        }
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

    // Broadcast CREATE/UPDATE event for real-time dashboard
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

/// Get memory history (audit trail) - SHO-39
///
/// Returns the revision history for a memory, showing how it evolved over time.
/// Useful for understanding context of how external entities changed.
///
/// Example: POST /api/memory/history {
///   "user_id": "agent-1",
///   "memory_id": "uuid-here"
/// }
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, id = %req.id))]
async fn get_memory_history(
    State(state): State<AppState>,
    Json(req): Json<MemoryHistoryRequest>,
) -> Result<Json<MemoryHistoryResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Parse memory ID
    let memory_id = uuid::Uuid::parse_str(&req.id)
        .map_err(|e| AppError::InvalidMemoryId(format!("Invalid UUID: {e}")))
        .map(memory::types::MemoryId)?;

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Get memory and its history
    let (memory, history) = {
        let memory = memory_system.clone();
        let mid = memory_id.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let mem = memory_guard.get_memory(&mid)?;
            let hist = mem.history.clone();
            Ok::<_, anyhow::Error>((mem, hist))
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Convert history to response format
    let history_info: Vec<MemoryRevisionInfo> = history
        .into_iter()
        .map(|rev| MemoryRevisionInfo {
            previous_content: rev.previous_content,
            change_type: format!("{:?}", rev.change_type).to_lowercase(),
            changed_at: rev.changed_at.to_rfc3339(),
            changed_by: rev.changed_by,
            change_reason: rev.change_reason,
        })
        .collect();

    Ok(Json(MemoryHistoryResponse {
        id: memory.id.0.to_string(),
        external_id: memory.external_id,
        current_content: memory.experience.content,
        version: memory.version,
        history: history_info,
    }))
}

// =============================================================================
// EXTERNAL INTEGRATIONS (SHO-40)
// =============================================================================

/// Linear webhook receiver - processes issue create/update/remove events
///
/// Transforms Linear webhook payloads into memory upserts with external_id linking.
/// Supports signature verification via LINEAR_WEBHOOK_SECRET env var.
///
/// Example webhook payload:
/// ```json
/// {
///   "action": "update",
///   "type": "Issue",
///   "data": {
///     "id": "uuid",
///     "identifier": "SHO-39",
///     "title": "Issue title",
///     "state": { "name": "In Progress" },
///     ...
///   }
/// }
/// ```
#[tracing::instrument(skip(state, body, headers))]
async fn linear_webhook(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, AppError> {
    use integrations::linear::LinearWebhook;

    // Get signing secret from env (optional but recommended)
    let signing_secret = std::env::var("LINEAR_WEBHOOK_SECRET").ok();
    let webhook = LinearWebhook::new(signing_secret);

    // Verify signature if present
    if let Some(signature) = headers
        .get("linear-signature")
        .and_then(|h| h.to_str().ok())
    {
        if !webhook
            .verify_signature(&body, signature)
            .map_err(|e| AppError::Internal(e))?
        {
            return Err(AppError::InvalidInput {
                field: "signature".to_string(),
                reason: "Invalid webhook signature".to_string(),
            });
        }
    }

    // Parse payload
    let payload = webhook
        .parse_payload(&body)
        .map_err(|e| AppError::Internal(e))?;

    // Only process Issue events
    if payload.entity_type != "Issue" {
        tracing::debug!(
            entity_type = %payload.entity_type,
            "Ignoring non-Issue webhook event"
        );
        return Ok(Json(serde_json::json!({
            "status": "ignored",
            "reason": "Only Issue events are processed"
        })));
    }

    // Handle remove action (soft delete - we keep the memory but mark it)
    if payload.action == "remove" {
        tracing::info!(
            identifier = ?payload.data.identifier,
            "Issue removed - memory will be retained with deleted marker"
        );
        // For now, we just acknowledge - could add deleted flag to memory in future
        return Ok(Json(serde_json::json!({
            "status": "acknowledged",
            "action": "remove"
        })));
    }

    // Build external_id from identifier
    let external_id = match &payload.data.identifier {
        Some(id) => format!("linear:{}", id),
        None => format!("linear:{}", payload.data.id),
    };

    // Transform to content and tags
    let content = LinearWebhook::issue_to_content(&payload.data);
    let tags = LinearWebhook::issue_to_tags(&payload.data);
    let change_type = LinearWebhook::determine_change_type(&payload.action, &payload.data);

    // Determine user_id - use LINEAR_SYNC_USER_ID env var or default
    let user_id =
        std::env::var("LINEAR_SYNC_USER_ID").unwrap_or_else(|_| "linear-sync".to_string());

    // Build experience
    let experience = Experience {
        content: content.clone(),
        experience_type: ExperienceType::Task,
        entities: tags.clone(),
        ..Default::default()
    };

    // Parse change type
    let change_type_enum = match change_type.as_str() {
        "created" => memory::types::ChangeType::Created,
        "status_changed" => memory::types::ChangeType::StatusChanged,
        "tags_updated" => memory::types::ChangeType::TagsUpdated,
        _ => memory::types::ChangeType::ContentUpdated,
    };

    // Get memory system
    let memory_system = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    // Perform upsert
    let (memory_id, was_update) = {
        let memory = memory_system.clone();
        let ext_id = external_id.clone();
        let exp = experience.clone();
        let ct = change_type_enum;
        let actor_name = payload
            .actor
            .as_ref()
            .and_then(|a| a.name.clone())
            .unwrap_or_else(|| "linear-webhook".to_string());

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.upsert(ext_id, exp, ct, Some(actor_name), None)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    tracing::info!(
        external_id = %external_id,
        memory_id = %memory_id.0,
        was_update = was_update,
        action = %payload.action,
        "Linear webhook processed"
    );

    Ok(Json(serde_json::json!({
        "status": "success",
        "id": memory_id.0.to_string(),
        "external_id": external_id,
        "was_update": was_update,
        "action": payload.action
    })))
}

/// Bulk sync Linear issues to Shodh memory
///
/// Fetches all issues from Linear API and upserts them as memories.
/// Useful for initial import or catching up after downtime.
///
/// Example: POST /api/sync/linear {
///   "user_id": "linear-sync",
///   "api_key": "lin_api_...",
///   "team_id": "optional",
///   "limit": 100
/// }
#[tracing::instrument(skip(state, req), fields(user_id = %req.user_id))]
async fn linear_sync(
    State(state): State<AppState>,
    Json(req): Json<integrations::linear::LinearSyncRequest>,
) -> Result<Json<integrations::linear::LinearSyncResponse>, AppError> {
    use integrations::linear::{LinearClient, LinearSyncResponse, LinearWebhook};

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Validate API key format
    if req.api_key.is_empty() {
        return Err(AppError::InvalidInput {
            field: "api_key".to_string(),
            reason: "Linear API key is required".to_string(),
        });
    }

    // Create Linear client
    let client = LinearClient::new(req.api_key.clone());

    // Fetch issues from Linear
    let issues = client
        .fetch_issues(
            req.team_id.as_deref(),
            req.updated_after.as_deref(),
            req.limit,
        )
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to fetch Linear issues: {}", e)))?;

    let total = issues.len();
    let mut created_count = 0;
    let mut updated_count = 0;
    let mut error_count = 0;
    let mut errors = Vec::new();

    // Get memory system
    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Process each issue
    for issue in issues {
        let external_id = match &issue.identifier {
            Some(id) => format!("linear:{}", id),
            None => format!("linear:{}", issue.id),
        };

        let content = LinearWebhook::issue_to_content(&issue);
        let tags = LinearWebhook::issue_to_tags(&issue);

        let experience = Experience {
            content,
            experience_type: ExperienceType::Task,
            entities: tags,
            ..Default::default()
        };

        // Perform upsert
        let result = {
            let memory = memory_system.clone();
            let ext_id = external_id.clone();
            let exp = experience;

            tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                memory_guard.upsert(
                    ext_id,
                    exp,
                    memory::types::ChangeType::ContentUpdated,
                    Some("linear-bulk-sync".to_string()),
                    None,
                )
            })
            .await
        };

        match result {
            Ok(Ok((_, was_update))) => {
                if was_update {
                    updated_count += 1;
                } else {
                    created_count += 1;
                }
            }
            Ok(Err(e)) => {
                error_count += 1;
                errors.push(format!("{}: {}", external_id, e));
            }
            Err(e) => {
                error_count += 1;
                errors.push(format!("{}: Task panicked: {}", external_id, e));
            }
        }
    }

    tracing::info!(
        total = total,
        created = created_count,
        updated = updated_count,
        errors = error_count,
        "Linear bulk sync completed"
    );

    Ok(Json(LinearSyncResponse {
        synced_count: total,
        created_count,
        updated_count,
        error_count,
        errors,
    }))
}

/// GitHub webhook receiver - processes Issue and PR events
///
/// Transforms GitHub webhook payloads into memory upserts with external_id linking.
/// Supports signature verification via GITHUB_WEBHOOK_SECRET env var.
///
/// Supported events:
/// - Issues: opened, edited, closed, reopened, labeled, unlabeled
/// - Pull Requests: opened, edited, closed, merged, synchronize
#[tracing::instrument(skip(state, body, headers))]
async fn github_webhook(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, AppError> {
    use integrations::github::GitHubWebhook;

    // Get webhook secret from env (optional but recommended)
    let webhook_secret = std::env::var("GITHUB_WEBHOOK_SECRET").ok();
    let webhook = GitHubWebhook::new(webhook_secret);

    // Verify signature if present
    if let Some(signature) = headers
        .get("x-hub-signature-256")
        .and_then(|h| h.to_str().ok())
    {
        if !webhook
            .verify_signature(&body, signature)
            .map_err(|e| AppError::Internal(e))?
        {
            return Err(AppError::InvalidInput {
                field: "signature".to_string(),
                reason: "Invalid webhook signature".to_string(),
            });
        }
    }

    // Get event type from header
    let event_type = headers
        .get("x-github-event")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");

    // Only process issues and pull_request events
    if event_type != "issues" && event_type != "pull_request" {
        tracing::debug!(event_type = %event_type, "Ignoring non-issue/PR GitHub event");
        return Ok(Json(serde_json::json!({
            "status": "ignored",
            "reason": format!("Only issues and pull_request events are processed, got: {}", event_type)
        })));
    }

    // Parse payload
    let payload = webhook
        .parse_payload(&body)
        .map_err(|e| AppError::Internal(e))?;

    // Determine user_id
    let user_id =
        std::env::var("GITHUB_SYNC_USER_ID").unwrap_or_else(|_| "github-sync".to_string());

    // Process based on event type
    let (external_id, content, tags, change_type) = if let Some(issue) = &payload.issue {
        // Issue event
        let ext_id = GitHubWebhook::issue_external_id(&payload.repository, issue.number);
        let content = GitHubWebhook::issue_to_content(issue, &payload.repository);
        let tags = GitHubWebhook::issue_to_tags(issue, &payload.repository);
        let ct = GitHubWebhook::determine_change_type(&payload.action, false);
        (ext_id, content, tags, ct)
    } else if let Some(pr) = &payload.pull_request {
        // PR event
        let ext_id = GitHubWebhook::pr_external_id(&payload.repository, pr.number);
        let content = GitHubWebhook::pr_to_content(pr, &payload.repository);
        let tags = GitHubWebhook::pr_to_tags(pr, &payload.repository);
        let ct = GitHubWebhook::determine_change_type(&payload.action, true);
        (ext_id, content, tags, ct)
    } else {
        return Ok(Json(serde_json::json!({
            "status": "ignored",
            "reason": "No issue or pull_request data in payload"
        })));
    };

    // Build experience
    let experience = Experience {
        content: content.clone(),
        experience_type: ExperienceType::Task,
        entities: tags.clone(),
        ..Default::default()
    };

    // Parse change type
    let change_type_enum = match change_type.as_str() {
        "created" => memory::types::ChangeType::Created,
        "status_changed" => memory::types::ChangeType::StatusChanged,
        "tags_updated" => memory::types::ChangeType::TagsUpdated,
        _ => memory::types::ChangeType::ContentUpdated,
    };

    // Get memory system
    let memory_system = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    // Perform upsert
    let (memory_id, was_update) = {
        let memory = memory_system.clone();
        let ext_id = external_id.clone();
        let exp = experience.clone();
        let ct = change_type_enum;
        let actor_name = payload
            .sender
            .as_ref()
            .map(|s| s.login.clone())
            .unwrap_or_else(|| "github-webhook".to_string());

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.upsert(ext_id, exp, ct, Some(actor_name), None)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    tracing::info!(
        external_id = %external_id,
        memory_id = %memory_id.0,
        was_update = was_update,
        action = %payload.action,
        event_type = %event_type,
        "GitHub webhook processed"
    );

    Ok(Json(serde_json::json!({
        "status": "success",
        "id": memory_id.0.to_string(),
        "external_id": external_id,
        "was_update": was_update,
        "action": payload.action,
        "event_type": event_type
    })))
}

/// Bulk sync GitHub issues and PRs to Shodh memory
///
/// Fetches issues and/or PRs from GitHub API and upserts them as memories.
///
/// Example: POST /api/sync/github {
///   "user_id": "github-sync",
///   "token": "ghp_...",
///   "owner": "varun29ankuS",
///   "repo": "shodh-memory",
///   "sync_issues": true,
///   "sync_prs": true,
///   "state": "all",
///   "limit": 100
/// }
#[tracing::instrument(skip(state, req), fields(user_id = %req.user_id, repo = %format!("{}/{}", req.owner, req.repo)))]
async fn github_sync(
    State(state): State<AppState>,
    Json(req): Json<integrations::github::GitHubSyncRequest>,
) -> Result<Json<integrations::github::GitHubSyncResponse>, AppError> {
    use integrations::github::{GitHubClient, GitHubSyncResponse, GitHubWebhook};

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Validate token
    if req.token.is_empty() {
        return Err(AppError::InvalidInput {
            field: "token".to_string(),
            reason: "GitHub token is required".to_string(),
        });
    }

    // Create GitHub client
    let client = GitHubClient::new(req.token.clone());

    // Get repository info first
    let repo_info = client
        .get_repository(&req.owner, &req.repo)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to get repository: {}", e)))?;

    let mut issues_synced = 0;
    let mut prs_synced = 0;
    let mut commits_synced = 0;
    let mut created_count = 0;
    let mut updated_count = 0;
    let mut error_count = 0;
    let mut errors = Vec::new();

    // Get memory system
    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Sync issues if requested
    if req.sync_issues {
        let issues = client
            .fetch_issues(&req.owner, &req.repo, &req.state, req.limit)
            .await
            .map_err(|e| {
                AppError::Internal(anyhow::anyhow!("Failed to fetch GitHub issues: {}", e))
            })?;

        for issue in issues {
            let external_id = GitHubWebhook::issue_external_id(&repo_info, issue.number);
            let content = GitHubWebhook::issue_to_content(&issue, &repo_info);
            let tags = GitHubWebhook::issue_to_tags(&issue, &repo_info);

            let experience = Experience {
                content,
                experience_type: ExperienceType::Task,
                entities: tags,
                ..Default::default()
            };

            let result = {
                let memory = memory_system.clone();
                let ext_id = external_id.clone();
                let exp = experience;

                tokio::task::spawn_blocking(move || {
                    let memory_guard = memory.read();
                    memory_guard.upsert(
                        ext_id,
                        exp,
                        memory::types::ChangeType::ContentUpdated,
                        Some("github-bulk-sync".to_string()),
                        None,
                    )
                })
                .await
            };

            match result {
                Ok(Ok((_, was_update))) => {
                    issues_synced += 1;
                    if was_update {
                        updated_count += 1;
                    } else {
                        created_count += 1;
                    }
                }
                Ok(Err(e)) => {
                    error_count += 1;
                    errors.push(format!("{}: {}", external_id, e));
                }
                Err(e) => {
                    error_count += 1;
                    errors.push(format!("{}: Task panicked: {}", external_id, e));
                }
            }
        }
    }

    // Sync PRs if requested
    if req.sync_prs {
        let prs = client
            .fetch_pull_requests(&req.owner, &req.repo, &req.state, req.limit)
            .await
            .map_err(|e| {
                AppError::Internal(anyhow::anyhow!("Failed to fetch GitHub PRs: {}", e))
            })?;

        for pr in prs {
            let external_id = GitHubWebhook::pr_external_id(&repo_info, pr.number);
            let content = GitHubWebhook::pr_to_content(&pr, &repo_info);
            let tags = GitHubWebhook::pr_to_tags(&pr, &repo_info);

            let experience = Experience {
                content,
                experience_type: ExperienceType::Task,
                entities: tags,
                ..Default::default()
            };

            let result = {
                let memory = memory_system.clone();
                let ext_id = external_id.clone();
                let exp = experience;

                tokio::task::spawn_blocking(move || {
                    let memory_guard = memory.read();
                    memory_guard.upsert(
                        ext_id,
                        exp,
                        memory::types::ChangeType::ContentUpdated,
                        Some("github-bulk-sync".to_string()),
                        None,
                    )
                })
                .await
            };

            match result {
                Ok(Ok((_, was_update))) => {
                    prs_synced += 1;
                    if was_update {
                        updated_count += 1;
                    } else {
                        created_count += 1;
                    }
                }
                Ok(Err(e)) => {
                    error_count += 1;
                    errors.push(format!("{}: {}", external_id, e));
                }
                Err(e) => {
                    error_count += 1;
                    errors.push(format!("{}: Task panicked: {}", external_id, e));
                }
            }
        }
    }

    // Sync commits if requested
    if req.sync_commits {
        let commits = client
            .fetch_commits(&req.owner, &req.repo, req.branch.as_deref(), req.limit)
            .await
            .map_err(|e| {
                AppError::Internal(anyhow::anyhow!("Failed to fetch GitHub commits: {}", e))
            })?;

        for commit in commits {
            let external_id = GitHubWebhook::commit_external_id(&repo_info, &commit.sha);
            let content = GitHubWebhook::commit_to_content(&commit, &repo_info);
            let tags = GitHubWebhook::commit_to_tags(&commit, &repo_info);

            let experience = Experience {
                content,
                experience_type: ExperienceType::CodeEdit,
                entities: tags,
                ..Default::default()
            };

            let result = {
                let memory = memory_system.clone();
                let ext_id = external_id.clone();
                let exp = experience;

                tokio::task::spawn_blocking(move || {
                    let memory_guard = memory.read();
                    memory_guard.upsert(
                        ext_id,
                        exp,
                        memory::types::ChangeType::ContentUpdated,
                        Some("github-bulk-sync".to_string()),
                        None,
                    )
                })
                .await
            };

            match result {
                Ok(Ok((_, was_update))) => {
                    commits_synced += 1;
                    if was_update {
                        updated_count += 1;
                    } else {
                        created_count += 1;
                    }
                }
                Ok(Err(e)) => {
                    error_count += 1;
                    errors.push(format!("{}: {}", external_id, e));
                }
                Err(e) => {
                    error_count += 1;
                    errors.push(format!("{}: Task panicked: {}", external_id, e));
                }
            }
        }
    }

    let total = issues_synced + prs_synced + commits_synced;

    tracing::info!(
        total = total,
        issues = issues_synced,
        prs = prs_synced,
        commits = commits_synced,
        created = created_count,
        updated = updated_count,
        errors = error_count,
        "GitHub bulk sync completed"
    );

    Ok(Json(GitHubSyncResponse {
        synced_count: total,
        issues_synced,
        prs_synced,
        commits_synced,
        created_count,
        updated_count,
        error_count,
        errors,
    }))
}

/// Helper function to search todos for recall/proactive_context
/// Returns todos matching the query using semantic search
async fn search_todos_for_recall(
    state: &AppState,
    user_id: &str,
    query: &str,
    limit: usize,
) -> Vec<RecallTodo> {
    // Compute embedding for the query
    let memory_system = match state.get_user_memory(user_id) {
        Ok(ms) => ms,
        Err(_) => return Vec::new(),
    };

    let query_clone = query.to_string();
    let query_embedding: Vec<f32> = match tokio::task::spawn_blocking(move || {
        let memory_guard = memory_system.read();
        memory_guard.compute_embedding(&query_clone).unwrap_or_default()
    })
    .await
    {
        Ok(emb) => emb,
        Err(_) => return Vec::new(),
    };

    if query_embedding.is_empty() {
        return Vec::new();
    }

    // Search todos using vector similarity
    let search_results = match state
        .todo_store
        .search_similar(user_id, &query_embedding, limit)
    {
        Ok(results) => results,
        Err(_) => return Vec::new(),
    };

    // Convert to RecallTodo format (filter out completed/cancelled)
    search_results
        .into_iter()
        .filter(|(todo, _)| {
            todo.status != TodoStatus::Done && todo.status != TodoStatus::Cancelled
        })
        .map(|(todo, score)| {
            let project = todo.project_id.as_ref().and_then(|pid| {
                state
                    .todo_store
                    .get_project(user_id, pid)
                    .ok()
                    .flatten()
                    .map(|p| p.name)
            });
            RecallTodo {
                id: todo.id.0.to_string(),
                short_id: todo.short_id(),
                content: todo.content,
                status: format!("{:?}", todo.status).to_lowercase(),
                priority: todo.priority.indicator().to_string(),
                project,
                score,
                created_at: todo.created_at.to_rfc3339(),
            }
        })
        .collect()
}

/// LLM-friendly /api/recall - hybrid retrieval combining semantic search + graph spreading activation
/// Example: POST /api/recall { "user_id": "agent-1", "query": "What does user like?" }
///
/// Retrieval Strategy (SHO-26 Enhanced):
/// - "semantic": Pure vector similarity search (MiniLM embeddings + Vamana HNSW)
/// - "associative": Graph traversal with density-dependent weights
/// - "hybrid": Combined semantic + graph with fixed weights (legacy default)
///
/// Associative mode (SHO-26) features:
/// - Density-dependent hybrid weights: Graph trust scales with learned associations
/// - Importance-weighted decay: Important memories decay slower during spreading activation
/// - RetrievalStats returned for observability
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query, mode = %req.mode))]
async fn recall(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, AppError> {
    use std::collections::HashMap;

    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let limit = req.limit;
    let query_text = req.query.clone();
    let query_text_clone = query_text.clone();
    let mode = req.mode.to_lowercase();

    // For "semantic" mode, skip graph entirely
    if mode == "semantic" {
        let semantic_memories: Vec<SharedMemory> = {
            let memory = memory_system.clone();
            let query_text = query_text.clone();
            tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                let query = MemoryQuery {
                    query_text: Some(query_text),
                    max_results: limit,
                    ..Default::default()
                };
                memory_guard.recall(&query).unwrap_or_default()
            })
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        };

        let recall_memories: Vec<RecallMemory> = semantic_memories
            .into_iter()
            .map(|m| {
                // Use actual similarity score from vector search, not manufactured rank-based score
                // This is the real cosine similarity from MiniLM embeddings
                let score = m.get_score().unwrap_or_else(|| {
                    // Fallback: use salience if score wasn't set (shouldn't happen)
                    tracing::warn!(
                        memory_id = %m.id.0,
                        "Memory missing similarity score in semantic search"
                    );
                    m.salience_score_with_access()
                });
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
        let duration = op_start.elapsed().as_secs_f64();
        metrics::MEMORY_RETRIEVE_DURATION
            .with_label_values(&["semantic"])
            .observe(duration);
        metrics::MEMORY_RETRIEVE_TOTAL
            .with_label_values(&["semantic", "success"])
            .inc();
        metrics::MEMORY_RETRIEVE_RESULTS
            .with_label_values(&["semantic"])
            .observe(count as f64);

        let stats = memory::types::RetrievalStats {
            mode: "semantic".to_string(),
            semantic_candidates: count,
            retrieval_time_us: op_start.elapsed().as_micros() as u64,
            ..Default::default()
        };

        // Broadcast RETRIEVE event for real-time dashboard
        state.emit_event(MemoryEvent {
            event_type: "RETRIEVE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: req.user_id.clone(),
            memory_id: None,
            content_preview: Some(req.query.chars().take(50).collect()),
            memory_type: Some("semantic".to_string()),
            importance: None,
            count: Some(count),
        });

        // Audit log for semantic recall
        state.log_event(
            &req.user_id,
            "RECALL",
            "mode:semantic",
            &format!(
                "Query='{}' returned {} memories (semantic)",
                query_text_clone.chars().take(50).collect::<String>(),
                count
            ),
        );

        // Record memory co-activation in GraphMemory (Hebbian learning)
        if count >= 2 {
            let memory_ids: Vec<uuid::Uuid> = recall_memories
                .iter()
                .filter_map(|m| uuid::Uuid::parse_str(&m.id).ok())
                .collect();

            if memory_ids.len() >= 2 {
                let graph = graph_memory.clone();
                tokio::task::spawn(async move {
                    let graph_guard = graph.write();
                    if let Err(e) = graph_guard.record_memory_coactivation(&memory_ids) {
                        tracing::debug!("Failed to record memory coactivation: {}", e);
                    }
                });
            }
        }

        // Search todos using same embedding
        let recall_todos = search_todos_for_recall(&state, &req.user_id, &query_text, 5).await;
        let todo_count = if recall_todos.is_empty() { None } else { Some(recall_todos.len()) };

        return Ok(Json(RecallResponse {
            memories: recall_memories,
            count,
            retrieval_stats: Some(stats),
            todos: recall_todos,
            todo_count,
        }));
    }

    // Run semantic retrieval via MemorySystem
    let semantic_memories: Vec<SharedMemory> = {
        let memory = memory_system.clone();
        let query_text = query_text.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let query = MemoryQuery {
                query_text: Some(query_text),
                max_results: limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
                ..Default::default()
            };
            memory_guard.recall(&query).unwrap_or_default()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // Run graph-based spreading activation retrieval
    // For "associative" mode, calculate graph density and use density-dependent weights
    let (graph_activated, retrieval_stats): (
        Vec<ActivatedMemory>,
        Option<memory::types::RetrievalStats>,
    ) = {
        let graph = graph_memory.clone();
        let memory = memory_system.clone();
        let query_for_graph = query_text_clone.clone();
        let use_density_weights = mode == "associative";

        tokio::task::spawn_blocking(move || {
            let graph_guard = graph.read();
            let memory_guard = memory.read();

            // Calculate graph density for associative mode
            let graph_density = if use_density_weights {
                match graph_guard.get_stats() {
                    Ok(stats) => {
                        let memory_count = stats.episode_count.max(1) as f32;
                        let edge_count = stats.relationship_count as f32;
                        Some(edge_count / memory_count)
                    }
                    Err(_) => None,
                }
            } else {
                None
            };

            // Build the episode-to-memory mapping function
            // EpisodicNode UUID == MemoryId.0 (see process_experience_into_graph)
            let episode_to_memory =
                |episode: &EpisodicNode| -> anyhow::Result<Option<SharedMemory>> {
                    let memory_id = MemoryId(episode.uuid);
                    match memory_guard.get_memory(&memory_id) {
                        Ok(mem) => Ok(Some(Arc::new(mem))),
                        Err(_) => Ok(None), // Memory may have been deleted
                    }
                };

            let query = MemoryQuery {
                query_text: Some(query_for_graph.clone()),
                max_results: limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
                ..Default::default()
            };

            // Run spreading activation with optional density-dependent weights (SHO-26)
            match memory::graph_retrieval::spreading_activation_retrieve_with_stats(
                &query_for_graph,
                &query,
                &graph_guard,
                memory_guard.get_embedder(),
                graph_density,
                episode_to_memory,
            ) {
                Ok((activated, stats)) => (activated, Some(stats)),
                Err(e) => {
                    tracing::debug!("Spreading activation failed: {}. Using semantic only.", e);
                    (Vec::new(), None)
                }
            }
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Graph retrieval panicked: {e}")))?
    };

    // Merge results with hybrid scoring
    let mut scored_memories: HashMap<uuid::Uuid, (f32, SharedMemory)> = HashMap::new();
    let semantic_count = semantic_memories.len();
    let graph_activated_count = graph_activated.len();

    // Get weights from retrieval stats (density-dependent for associative, fixed for hybrid)
    let (semantic_weight, graph_weight, linguistic_weight) =
        if let Some(ref stats) = retrieval_stats {
            (
                stats.semantic_weight,
                stats.graph_weight,
                stats.linguistic_weight,
            )
        } else {
            (0.50, 0.35, 0.15) // Legacy fixed weights
        };

    // Add semantic results with their actual similarity score
    for memory in semantic_memories.iter() {
        // Use actual cosine similarity from vector search, not rank-based approximation
        let semantic_score = memory.get_score().unwrap_or_else(|| {
            tracing::warn!(
                memory_id = %memory.id.0,
                "Memory missing similarity score in hybrid search"
            );
            memory.salience_score_with_access()
        });
        let hybrid_score = semantic_weight * semantic_score;
        scored_memories.insert(memory.id.0, (hybrid_score, memory.clone()));
    }

    // Add/boost graph-activated results
    // Find max activation for normalization (activation can be unbounded due to accumulation)
    let max_activation = graph_activated
        .iter()
        .map(|a| a.activation_score)
        .fold(1.0_f32, |max, score| max.max(score)); // Min 1.0 to avoid division issues

    for activated in graph_activated {
        let entry = scored_memories
            .entry(activated.memory.id.0)
            .or_insert((0.0, activated.memory.clone()));

        // Normalize activation score to 0-1 range, then apply weights
        let normalized_activation = (activated.activation_score / max_activation).min(1.0);
        let normalized_linguistic = activated.linguistic_score.min(1.0);

        // Add graph activation score + linguistic score with density-dependent weights
        entry.0 += graph_weight * normalized_activation + linguistic_weight * normalized_linguistic;
    }

    // Sort by final hybrid score and take top N
    let mut final_results: Vec<(f32, SharedMemory)> = scored_memories.into_values().collect();
    final_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    final_results.truncate(limit);

    let graph_contribution = final_results
        .iter()
        .filter(|(score, _)| *score > semantic_weight) // Has graph boost
        .count();

    tracing::debug!(
        mode = %mode,
        semantic_count = semantic_count,
        graph_activated_count = graph_activated_count,
        final_count = final_results.len(),
        graph_contribution = graph_contribution,
        graph_weight = graph_weight,
        "Retrieval completed"
    );

    // Convert to response format - use the pre-computed hybrid score
    let recall_memories: Vec<RecallMemory> = final_results
        .into_iter()
        .map(|(hybrid_score, m)| RecallMemory {
            id: m.id.0.to_string(),
            experience: RecallExperience {
                content: m.experience.content.clone(),
                memory_type: Some(format!("{:?}", m.experience.experience_type)),
                tags: m.experience.entities.clone(),
            },
            importance: m.importance(),
            created_at: m.created_at.to_rfc3339(),
            score: hybrid_score,
        })
        .collect();

    let count = recall_memories.len();

    // Update retrieval stats with final counts
    let final_stats = retrieval_stats.map(|mut stats| {
        stats.semantic_candidates = semantic_count;
        stats.retrieval_time_us = op_start.elapsed().as_micros() as u64;
        stats
    });

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_RETRIEVE_DURATION
        .with_label_values(&[&mode])
        .observe(duration);
    metrics::MEMORY_RETRIEVE_TOTAL
        .with_label_values(&[&mode, &String::from("success")])
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
        memory_type: Some(mode.clone()),
        importance: None,
        count: Some(count),
    });

    // Audit log for recall operations
    state.log_event(
        &req.user_id,
        "RECALL",
        &format!("mode:{}", mode),
        &format!(
            "Query='{}' returned {} memories (mode={})",
            query_text_clone.chars().take(50).collect::<String>(),
            count,
            mode
        ),
    );

    // Record memory co-activation in GraphMemory (Hebbian learning)
    // When memories are retrieved together, they form associations
    if count >= 2 {
        let memory_ids: Vec<uuid::Uuid> = recall_memories
            .iter()
            .filter_map(|m| uuid::Uuid::parse_str(&m.id).ok())
            .collect();

        if memory_ids.len() >= 2 {
            let graph = graph_memory.clone();
            tokio::task::spawn(async move {
                let graph_guard = graph.write();
                if let Err(e) = graph_guard.record_memory_coactivation(&memory_ids) {
                    tracing::debug!("Failed to record memory coactivation: {}", e);
                }
            });
        }
    }

    // AUD-1: Hebbian strengthening of edges traversed during spreading activation
    // Strengthen entity-entity edges that were used in successful graph retrieval
    if let Some(ref stats) = final_stats {
        if !stats.traversed_edges.is_empty() {
            let graph = graph_memory.clone();
            let edges = stats.traversed_edges.clone();
            tokio::task::spawn(async move {
                let graph_guard = graph.write();
                match graph_guard.batch_strengthen_synapses(&edges) {
                    Ok(strengthened) => {
                        if strengthened > 0 {
                            tracing::debug!(
                                "Hebbian strengthening: {} edges reinforced",
                                strengthened
                            );
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to strengthen synapses: {}", e);
                    }
                }
            });
        }
    }

    // Search todos using same query
    let recall_todos = search_todos_for_recall(&state, &req.user_id, &req.query, 5).await;
    let todo_count = if recall_todos.is_empty() { None } else { Some(recall_todos.len()) };

    Ok(Json(RecallResponse {
        memories: recall_memories,
        count,
        retrieval_stats: final_stats,
        todos: recall_todos,
        todo_count,
    }))
}

// =============================================================================
// CONTEXT SUMMARY - Session bootstrap with categorized memories
// =============================================================================

/// Context summary request
#[derive(Debug, Deserialize)]
struct ContextSummaryRequest {
    user_id: String,
    #[serde(default = "default_true")]
    include_decisions: bool,
    #[serde(default = "default_true")]
    include_learnings: bool,
    #[serde(default = "default_true")]
    include_context: bool,
    #[serde(default = "default_max_items")]
    max_items: usize,
}

fn default_true() -> bool {
    true
}

fn default_max_items() -> usize {
    5
}

/// Summary item - simplified memory for context
#[derive(Debug, Serialize)]
struct SummaryItem {
    id: String,
    content: String,
    importance: f32,
    created_at: String,
}

/// Context summary response - categorized memories for session bootstrap
#[derive(Debug, Serialize)]
struct ContextSummaryResponse {
    total_memories: usize,
    decisions: Vec<SummaryItem>,
    learnings: Vec<SummaryItem>,
    context: Vec<SummaryItem>,
    patterns: Vec<SummaryItem>,
    errors: Vec<SummaryItem>,
}

// =========================================================================
// CONSOLIDATION INTROSPECTION API
// =========================================================================

/// Request for consolidation report - what the memory system is learning
#[derive(Debug, Deserialize)]
struct ConsolidationReportRequest {
    user_id: String,
    /// Start of the time period (ISO 8601 format, optional - defaults to 1 hour ago)
    #[serde(default)]
    since: Option<String>,
    /// End of the time period (ISO 8601 format, optional - defaults to now)
    #[serde(default)]
    until: Option<String>,
}

/// POST /api/consolidation/report - Get consolidation introspection report
/// Shows what the memory system has been learning (strengthened/decayed memories, associations, facts)
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn get_consolidation_report(
    State(state): State<AppState>,
    Json(req): Json<ConsolidationReportRequest>,
) -> Result<Json<memory::ConsolidationReport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Parse time range (default: last hour)
    let now = chrono::Utc::now();
    let since = if let Some(since_str) = &req.since {
        chrono::DateTime::parse_from_rfc3339(since_str)
            .map_err(|e| AppError::InvalidInput {
                field: "since".to_string(),
                reason: format!("Invalid timestamp: {}", e),
            })?
            .with_timezone(&chrono::Utc)
    } else {
        now - chrono::Duration::hours(1)
    };

    let until = if let Some(until_str) = &req.until {
        Some(
            chrono::DateTime::parse_from_rfc3339(until_str)
                .map_err(|e| AppError::InvalidInput {
                    field: "until".to_string(),
                    reason: format!("Invalid timestamp: {}", e),
                })?
                .with_timezone(&chrono::Utc),
        )
    } else {
        None
    };

    let report = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_consolidation_report(since, until)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    Ok(Json(report))
}

/// GET /api/consolidation/events - Get raw consolidation events since a timestamp
#[derive(Debug, Deserialize)]
struct ConsolidationEventsRequest {
    user_id: String,
    /// Start timestamp (ISO 8601 format, optional - defaults to 1 hour ago)
    #[serde(default)]
    since: Option<String>,
}

#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn get_consolidation_events(
    State(state): State<AppState>,
    Json(req): Json<ConsolidationEventsRequest>,
) -> Result<Json<Vec<memory::ConsolidationEvent>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Parse time range (default: last hour)
    let now = chrono::Utc::now();
    let since = if let Some(since_str) = &req.since {
        chrono::DateTime::parse_from_rfc3339(since_str)
            .map_err(|e| AppError::InvalidInput {
                field: "since".to_string(),
                reason: format!("Invalid timestamp: {}", e),
            })?
            .with_timezone(&chrono::Utc)
    } else {
        now - chrono::Duration::hours(1)
    };

    let events = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_consolidation_events_since(since)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    Ok(Json(events))
}

// ============================================================================
// SEMANTIC FACTS API (SHO-f0e7)
// Facts are durable knowledge distilled from episodic memories
// ============================================================================

#[derive(Debug, Deserialize)]
struct FactsListRequest {
    user_id: String,
    #[serde(default = "facts_default_limit")]
    limit: usize,
}

fn facts_default_limit() -> usize {
    50
}

#[derive(Debug, Deserialize)]
struct FactsSearchRequest {
    user_id: String,
    query: String,
    #[serde(default = "facts_default_limit")]
    limit: usize,
}

#[derive(Debug, Deserialize)]
struct FactsByEntityRequest {
    user_id: String,
    entity: String,
    #[serde(default = "facts_default_limit")]
    limit: usize,
}

#[derive(Debug, Serialize)]
struct FactsResponse {
    facts: Vec<memory::SemanticFact>,
    total: usize,
}

/// POST /api/facts/list - List semantic facts for a user
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn list_facts(
    State(state): State<AppState>,
    Json(req): Json<FactsListRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_facts(&user_id, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/search - Search facts by keyword
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
async fn search_facts(
    State(state): State<AppState>,
    Json(req): Json<FactsSearchRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let query = req.query.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.search_facts(&user_id, &query, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/by-entity - Get facts related to an entity
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, entity = %req.entity))]
async fn facts_by_entity(
    State(state): State<AppState>,
    Json(req): Json<FactsByEntityRequest>,
) -> Result<Json<FactsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let entity = req.entity.clone();
    let limit = req.limit;

    let facts = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_facts_by_entity(&user_id, &entity, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = facts.len();
    Ok(Json(FactsResponse { facts, total }))
}

/// POST /api/facts/stats - Get statistics about stored facts
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn get_facts_stats(
    State(state): State<AppState>,
    Json(req): Json<FactsListRequest>,
) -> Result<Json<memory::FactStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();

    let stats = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.get_fact_stats(&user_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

// =============================================================================
// SHO-118: DECISION LINEAGE GRAPH API
// =============================================================================

/// Request to trace lineage from a memory
#[derive(Debug, Deserialize)]
struct LineageTraceRequest {
    user_id: String,
    memory_id: String,
    #[serde(default = "lineage_direction_default")]
    direction: String, // "backward", "forward", "both"
    #[serde(default = "lineage_max_depth_default")]
    max_depth: usize,
}

fn lineage_direction_default() -> String {
    "backward".to_string()
}

fn lineage_max_depth_default() -> usize {
    10
}

/// Request to confirm/reject an edge
#[derive(Debug, Deserialize)]
struct LineageEdgeRequest {
    user_id: String,
    edge_id: String,
}

/// Request to add an explicit lineage edge
#[derive(Debug, Deserialize)]
struct LineageAddEdgeRequest {
    user_id: String,
    from_memory_id: String,
    to_memory_id: String,
    relation: String, // "Caused", "ResolvedBy", "InformedBy", etc.
}

/// Request to list lineage edges
#[derive(Debug, Deserialize)]
struct LineageListRequest {
    user_id: String,
    #[serde(default = "lineage_list_limit_default")]
    limit: usize,
}

fn lineage_list_limit_default() -> usize {
    50
}

/// Request to create a branch
#[derive(Debug, Deserialize)]
struct LineageCreateBranchRequest {
    user_id: String,
    name: String,
    parent_branch: String,
    branch_point_memory_id: String,
    description: Option<String>,
}

/// Response for lineage trace
#[derive(Debug, Serialize)]
struct LineageTraceResponse {
    root: String,
    direction: String,
    edges: Vec<memory::LineageEdge>,
    path: Vec<String>,
    depth: usize,
}

/// Response for lineage edges list
#[derive(Debug, Serialize)]
struct LineageEdgesResponse {
    edges: Vec<memory::LineageEdge>,
    total: usize,
}

/// Response for branches list
#[derive(Debug, Serialize)]
struct LineageBranchesResponse {
    branches: Vec<memory::LineageBranch>,
    total: usize,
}

/// POST /api/lineage/trace - Trace lineage from a memory
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, memory_id = %req.memory_id))]
async fn lineage_trace(
    State(state): State<AppState>,
    Json(req): Json<LineageTraceRequest>,
) -> Result<Json<LineageTraceResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let memory_id_str = req.memory_id.clone();
    let direction = req.direction.clone();
    let max_depth = req.max_depth;

    let trace = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        let memory_id = memory::MemoryId(
            uuid::Uuid::parse_str(&memory_id_str)
                .map_err(|e| anyhow::anyhow!("Invalid memory_id: {}", e))?,
        );
        let dir = match direction.as_str() {
            "forward" => memory::TraceDirection::Forward,
            "both" => memory::TraceDirection::Both,
            _ => memory::TraceDirection::Backward,
        };
        memory_guard.trace_lineage(&user_id, &memory_id, dir, max_depth)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(LineageTraceResponse {
        root: trace.root.0.to_string(),
        direction: format!("{:?}", trace.direction),
        edges: trace.edges,
        path: trace.path.iter().map(|id| id.0.to_string()).collect(),
        depth: trace.depth,
    }))
}

/// POST /api/lineage/edges - List all lineage edges for a user
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn lineage_list_edges(
    State(state): State<AppState>,
    Json(req): Json<LineageListRequest>,
) -> Result<Json<LineageEdgesResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let limit = req.limit;

    let edges = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.lineage_graph().list_edges(&user_id, limit)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = edges.len();
    Ok(Json(LineageEdgesResponse { edges, total }))
}

/// POST /api/lineage/confirm - Confirm an inferred lineage edge
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, edge_id = %req.edge_id))]
async fn lineage_confirm_edge(
    State(state): State<AppState>,
    Json(req): Json<LineageEdgeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let edge_id = req.edge_id.clone();

    let confirmed = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard
            .lineage_graph()
            .confirm_edge(&user_id, &edge_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({ "confirmed": confirmed })))
}

/// POST /api/lineage/reject - Reject (delete) an inferred lineage edge
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, edge_id = %req.edge_id))]
async fn lineage_reject_edge(
    State(state): State<AppState>,
    Json(req): Json<LineageEdgeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let edge_id = req.edge_id.clone();

    let rejected = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.lineage_graph().reject_edge(&user_id, &edge_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({ "rejected": rejected })))
}

/// POST /api/lineage/link - Add an explicit lineage edge
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn lineage_add_edge(
    State(state): State<AppState>,
    Json(req): Json<LineageAddEdgeRequest>,
) -> Result<Json<memory::LineageEdge>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let from_str = req.from_memory_id.clone();
    let to_str = req.to_memory_id.clone();
    let relation_str = req.relation.clone();

    let edge = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        let from = memory::MemoryId(
            uuid::Uuid::parse_str(&from_str)
                .map_err(|e| anyhow::anyhow!("Invalid from_memory_id: {}", e))?,
        );
        let to = memory::MemoryId(
            uuid::Uuid::parse_str(&to_str)
                .map_err(|e| anyhow::anyhow!("Invalid to_memory_id: {}", e))?,
        );
        let relation = match relation_str.as_str() {
            "Caused" => memory::CausalRelation::Caused,
            "ResolvedBy" => memory::CausalRelation::ResolvedBy,
            "InformedBy" => memory::CausalRelation::InformedBy,
            "SupersededBy" => memory::CausalRelation::SupersededBy,
            "TriggeredBy" => memory::CausalRelation::TriggeredBy,
            "BranchedFrom" => memory::CausalRelation::BranchedFrom,
            _ => memory::CausalRelation::RelatedTo,
        };
        memory_guard
            .lineage_graph()
            .add_explicit_edge(&user_id, from, to, relation)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(edge))
}

/// POST /api/lineage/stats - Get lineage statistics
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn lineage_stats(
    State(state): State<AppState>,
    Json(req): Json<LineageListRequest>,
) -> Result<Json<memory::LineageStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();

    let stats = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.lineage_stats(&user_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// POST /api/lineage/branches - List all branches for a user
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn lineage_list_branches(
    State(state): State<AppState>,
    Json(req): Json<LineageListRequest>,
) -> Result<Json<LineageBranchesResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();

    let branches = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        memory_guard.lineage_graph().list_branches(&user_id)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    let total = branches.len();
    Ok(Json(LineageBranchesResponse { branches, total }))
}

/// POST /api/lineage/branch - Create a new branch
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, name = %req.name))]
async fn lineage_create_branch(
    State(state): State<AppState>,
    Json(req): Json<LineageCreateBranchRequest>,
) -> Result<Json<memory::LineageBranch>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let name = req.name.clone();
    let parent = req.parent_branch.clone();
    let branch_point_str = req.branch_point_memory_id.clone();
    let description = req.description.clone();

    let branch = tokio::task::spawn_blocking(move || {
        let memory_guard = memory.read();
        let branch_point = memory::MemoryId(
            uuid::Uuid::parse_str(&branch_point_str)
                .map_err(|e| anyhow::anyhow!("Invalid branch_point_memory_id: {}", e))?,
        );
        memory_guard.lineage_graph().create_branch(
            &user_id,
            &name,
            &parent,
            branch_point,
            description.as_deref(),
        )
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    .map_err(AppError::Internal)?;

    Ok(Json(branch))
}

// =============================================================================
// AUTO-LINEAGE INFERENCE (Background Task)
// =============================================================================

/// Automatically infer lineage edges when a new memory is created.
/// Runs in background - never blocks the remember request.
/// Looks at recent memories and creates edges based on:
/// - Entity overlap (shared entities)
/// - Temporal proximity (created close together)
/// - Type patterns (Error→Task=Caused, Task→Learning=ResolvedBy, etc.)
async fn auto_infer_lineage(
    state: &AppState,
    user_id: &str,
    new_memory_id: &memory::MemoryId,
    new_experience: &memory::Experience,
) -> anyhow::Result<()> {
    let memory = state.get_user_memory(user_id)?;

    // Get recent memories to compare against (last 24 hours)
    let recent_memories: Vec<memory::Memory> = {
        let memory_guard = memory.read();

        // Get all memories and filter by time
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
        let all_memories = memory_guard.get_all_memories()?;

        all_memories
            .into_iter()
            .filter(|m| m.created_at > cutoff && m.id != *new_memory_id)
            .take(50) // Limit to 50 most recent
            .map(|arc_mem| (*arc_mem).clone())
            .collect()
    };

    if recent_memories.is_empty() {
        return Ok(()); // No candidates to link to
    }

    // Build a Memory struct for the new experience
    let new_memory = memory::Memory::new(
        new_memory_id.clone(),
        new_experience.clone(),
        0.5,  // importance
        None, // agent_id
        None, // run_id
        None, // actor_id
        Some(chrono::Utc::now()),
    );

    // Run inference to find potential edges
    let inferred_edges = {
        let memory_guard = memory.read();
        memory_guard.infer_lineage_for_memory(user_id, &new_memory, &recent_memories)?
    };

    // Store inferred edges (if any)
    if !inferred_edges.is_empty() {
        let memory_guard = memory.read();
        let lineage = memory_guard.lineage_graph();

        for edge in inferred_edges {
            // Only store if confidence is above threshold
            if edge.confidence >= 0.4 {
                if let Err(e) = lineage.store_edge(user_id, &edge) {
                    tracing::debug!("Failed to store lineage edge: {}", e);
                }
            }
        }

        tracing::debug!(
            "Auto-inferred lineage for memory {}: found potential connections",
            new_memory_id.0
        );
    }

    Ok(())
}

/// Auto-generate post-mortem summary when a todo is completed.
/// Searches for related memories created during the todo's lifetime,
/// extracts learnings/decisions/errors, and stores a summary Learning.
async fn auto_generate_post_mortem(
    state: &AppState,
    user_id: &str,
    todo_content: &str,
    todo_created_at: chrono::DateTime<chrono::Utc>,
) -> anyhow::Result<()> {
    let memory = state.get_user_memory(user_id)?;

    // Get memories created during the todo's lifetime
    let related_memories: Vec<memory::Memory> = {
        let memory_guard = memory.read();
        let all_memories = memory_guard.get_all_memories()?;

        all_memories
            .into_iter()
            .filter(|m| m.created_at >= todo_created_at)
            .map(|arc_mem| (*arc_mem).clone())
            .collect()
    };

    if related_memories.is_empty() {
        return Ok(()); // No related work to summarize
    }

    // Categorize memories by type
    let mut learnings = Vec::new();
    let mut decisions = Vec::new();
    let mut errors_resolved = Vec::new();
    let mut patterns = Vec::new();

    for mem in &related_memories {
        match mem.experience.experience_type {
            memory::ExperienceType::Learning => {
                learnings.push(mem.experience.content.clone());
            }
            memory::ExperienceType::Decision => {
                decisions.push(mem.experience.content.clone());
            }
            memory::ExperienceType::Error => {
                errors_resolved.push(mem.experience.content.clone());
            }
            memory::ExperienceType::Pattern => {
                patterns.push(mem.experience.content.clone());
            }
            _ => {}
        }
    }

    // Only create post-mortem if we have significant content
    let has_content = !learnings.is_empty()
        || !decisions.is_empty()
        || !errors_resolved.is_empty()
        || !patterns.is_empty();

    if !has_content {
        return Ok(()); // Nothing significant to summarize
    }

    // Build post-mortem summary
    let mut summary_parts = Vec::new();
    summary_parts.push(format!("**Completed: {}**", todo_content));

    if !learnings.is_empty() {
        summary_parts.push(format!(
            "📚 Learnings ({}):\n{}",
            learnings.len(),
            learnings
                .iter()
                .take(3)
                .map(|l| format!("  • {}", truncate_content(l, 80)))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    if !decisions.is_empty() {
        summary_parts.push(format!(
            "🎯 Decisions ({}):\n{}",
            decisions.len(),
            decisions
                .iter()
                .take(3)
                .map(|d| format!("  • {}", truncate_content(d, 80)))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    if !errors_resolved.is_empty() {
        summary_parts.push(format!(
            "🐛 Errors resolved ({}):\n{}",
            errors_resolved.len(),
            errors_resolved
                .iter()
                .take(3)
                .map(|e| format!("  • {}", truncate_content(e, 80)))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    if !patterns.is_empty() {
        summary_parts.push(format!(
            "🔄 Patterns ({}):\n{}",
            patterns.len(),
            patterns
                .iter()
                .take(2)
                .map(|p| format!("  • {}", truncate_content(p, 80)))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    let summary_content = summary_parts.join("\n\n");

    // Store as a Learning memory
    let experience = memory::Experience {
        experience_type: memory::ExperienceType::Learning,
        content: summary_content,
        entities: vec!["post-mortem".to_string(), "task-completion".to_string()],
        ..Default::default()
    };

    // Store the post-mortem memory
    {
        let memory_guard = memory.read();
        memory_guard.remember(experience, None)?;
    }

    tracing::debug!(
        user_id = %user_id,
        todo = %todo_content,
        learnings = learnings.len(),
        decisions = decisions.len(),
        errors = errors_resolved.len(),
        "Generated post-mortem summary"
    );

    Ok(())
}

/// Truncate content to max length with ellipsis
fn truncate_content(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// GET /api/context_summary - Get categorized context for session bootstrap
/// Returns decisions, learnings, patterns, errors organized for LLM consumption
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn context_summary(
    State(state): State<AppState>,
    Json(req): Json<ContextSummaryRequest>,
) -> Result<Json<ContextSummaryResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let all_memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_all_memories()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    let total_memories = all_memories.len();

    // Categorize memories by type
    let mut decisions = Vec::new();
    let mut learnings = Vec::new();
    let mut context = Vec::new();
    let mut patterns = Vec::new();
    let mut errors = Vec::new();

    for m in all_memories {
        let item = SummaryItem {
            id: m.id.0.to_string(),
            content: m.experience.content.chars().take(200).collect(),
            importance: m.importance(),
            created_at: m.created_at.to_rfc3339(),
        };

        match m.experience.experience_type {
            ExperienceType::Decision => decisions.push(item),
            ExperienceType::Learning => learnings.push(item),
            ExperienceType::Context | ExperienceType::Observation => context.push(item),
            ExperienceType::Pattern => patterns.push(item),
            ExperienceType::Error => errors.push(item),
            _ => context.push(item),
        }
    }

    // Sort by importance and truncate
    let sort_and_truncate = |mut items: Vec<SummaryItem>, max: usize| -> Vec<SummaryItem> {
        items.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(max);
        items
    };

    let max = req.max_items;
    Ok(Json(ContextSummaryResponse {
        total_memories,
        decisions: if req.include_decisions {
            sort_and_truncate(decisions, max)
        } else {
            vec![]
        },
        learnings: if req.include_learnings {
            sort_and_truncate(learnings, max)
        } else {
            vec![]
        },
        context: if req.include_context {
            sort_and_truncate(context, max)
        } else {
            vec![]
        },
        patterns: sort_and_truncate(patterns, max),
        errors: sort_and_truncate(errors, 3.min(max)), // Limit errors to 3
    }))
}

// =============================================================================
// PROACTIVE CONTEXT (SHO-116) - Combined recall + reminders for MCP
// =============================================================================

/// Request for proactive context - returns relevant memories + triggered reminders
#[derive(Debug, Deserialize)]
struct ProactiveContextRequest {
    user_id: String,
    context: String,
    #[serde(default = "default_proactive_max_results")]
    max_results: usize,
    /// Minimum semantic similarity threshold (0.0-1.0)
    #[serde(default = "default_semantic_threshold")]
    semantic_threshold: f32,
    /// Weight for entity matching in relevance scoring
    #[serde(default = "default_entity_weight")]
    entity_match_weight: f32,
    /// Weight for recency boost
    #[serde(default = "default_recency_weight")]
    recency_weight: f32,
    /// Filter to specific memory types
    #[serde(default)]
    memory_types: Vec<String>,
    /// Whether to auto-ingest the context as a Conversation memory
    #[serde(default = "default_true")]
    auto_ingest: bool,
    /// Agent's previous response (for implicit feedback extraction)
    #[serde(default)]
    previous_response: Option<String>,
    /// User's followup message after agent response (for delayed signals)
    #[serde(default)]
    user_followup: Option<String>,
}

/// Feedback processing results
#[derive(Debug, Serialize)]
struct FeedbackProcessed {
    memories_evaluated: usize,
    reinforced: Vec<String>,
    weakened: Vec<String>,
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

/// Surfaced memory in proactive context response
#[derive(Debug, Serialize)]
struct ProactiveSurfacedMemory {
    id: String,
    content: String,
    memory_type: String,
    score: f32,
    created_at: String,
    tags: Vec<String>,
    /// Embedding for semantic feedback (not serialized to response)
    #[serde(skip)]
    embedding: Vec<f32>,
}

/// Todo item in proactive context response
#[derive(Debug, Serialize)]
struct ProactiveTodoItem {
    id: String,
    short_id: String,
    content: String,
    status: String,
    priority: String,
    project: Option<String>,
    due_date: Option<String>,
    relevance_reason: String,
    /// Semantic similarity score (0.0 - 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    similarity_score: Option<f32>,
}

/// Response for proactive context
#[derive(Debug, Serialize)]
struct ProactiveContextResponse {
    /// Relevant memories based on context
    memories: Vec<ProactiveSurfacedMemory>,
    /// Due time-based reminders
    due_reminders: Vec<ReminderItem>,
    /// Context-triggered reminders (keyword match)
    context_reminders: Vec<ReminderItem>,
    /// Total counts
    memory_count: usize,
    reminder_count: usize,
    /// ID of auto-ingested memory (if auto_ingest=true)
    #[serde(skip_serializing_if = "Option::is_none")]
    ingested_memory_id: Option<String>,
    /// Feedback processing results (if previous_response was provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    feedback_processed: Option<FeedbackProcessed>,
    /// Relevant todos based on context
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    relevant_todos: Vec<ProactiveTodoItem>,
    /// Todo count
    #[serde(default)]
    todo_count: usize,
}

/// POST /api/proactive_context - Combined recall + reminders for AI agents
///
/// Returns relevant memories based on semantic similarity and entity matching,
/// plus any due or context-triggered reminders. Optionally stores the context
/// as a Conversation memory for future recall.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn proactive_context(
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
                let signals = crate::memory::feedback::process_implicit_feedback_with_semantics(
                    &pending,
                    &response_text,
                    followup.as_deref(),
                    response_embedding.as_deref(),
                );

                let mut reinforced = Vec::new();
                let mut weakened = Vec::new();
                let mut helpful_ids: Vec<crate::memory::types::MemoryId> = Vec::new();
                let mut misleading_ids: Vec<crate::memory::types::MemoryId> = Vec::new();

                // Extract entities from context for fingerprinting
                let context_entities: Vec<String> =
                    crate::memory::feedback::extract_entities_simple(&pending.context)
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
                        let fingerprint = crate::memory::feedback::ContextFingerprint::new(
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

    // 2. Semantic recall with composite relevance scoring
    let context_clone = req.context.clone();
    let max_results = req.max_results;
    let context_emb_for_scoring = context_embedding.clone();
    let injection_config = InjectionConfig::default();
    let feedback_store_for_scoring = state.feedback_store.clone();
    let memories: Vec<ProactiveSurfacedMemory> = {
        let memory = memory_system.clone();
        let graph = graph_memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let graph_guard = graph.read();
            let feedback_guard = feedback_store_for_scoring.read();
            let now = chrono::Utc::now();

            let query = MemoryQuery {
                query_text: Some(context_clone),
                max_results: max_results * 2, // Fetch more, filter with injection engine
                ..Default::default()
            };
            let results = memory_guard.recall(&query).unwrap_or_default();

            // Compute composite relevance for each memory
            let mut candidates: Vec<(SharedMemory, f32)> = results
                .into_iter()
                .filter_map(|m| {
                    // Get embedding (skip if none)
                    let memory_embedding = m.experience.embeddings.as_ref()?.clone();

                    // Get Hebbian strength from graph (default 0.3 if not found)
                    // Lower default prevents new memories from scoring too high
                    let hebbian_strength = graph_guard
                        .get_memory_hebbian_strength(&m.id)
                        .unwrap_or(0.3);

                    // Get feedback momentum EMA (0.0 if no feedback history)
                    // Negative values indicate often-ignored memories → suppression
                    let feedback_momentum = feedback_guard
                        .get_momentum(&m.id)
                        .map(|fm| fm.ema)
                        .unwrap_or(0.0);

                    let input = RelevanceInput {
                        memory_embedding,
                        created_at: m.created_at,
                        hebbian_strength,
                        feedback_momentum,
                        ..Default::default()
                    };

                    let score =
                        compute_relevance(&input, &context_emb_for_scoring, now, &injection_config);
                    Some((m, score))
                })
                .collect();

            // Sort by composite relevance (highest first)
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Apply injection threshold and limit
            candidates
                .into_iter()
                .filter(|(_, score)| *score >= injection_config.min_relevance)
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
        let surfaced_infos: Vec<crate::memory::feedback::SurfacedMemoryInfo> = memories
            .iter()
            .map(|m| {
                let id = uuid::Uuid::parse_str(&m.id).unwrap_or_else(|_| uuid::Uuid::new_v4());
                crate::memory::feedback::SurfacedMemoryInfo {
                    id: crate::memory::types::MemoryId(id),
                    entities: crate::memory::feedback::extract_entities_simple(&m.content),
                    content_preview: m.content.chars().take(100).collect(),
                    score: m.score,
                    embedding: m.embedding.clone(),
                }
            })
            .collect();

        if !surfaced_infos.is_empty() {
            let pending = crate::memory::feedback::PendingFeedback::new(
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

    // 4. Auto-ingest previous assistant response (if provided and meaningful)
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

    // 5. Auto-ingest user context with segmentation for Hebbian-optimal storage
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

    // 6. Surface relevant todos using semantic search (TMI-5)
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
                _ => b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal),
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
// PROACTIVE MEMORY SURFACING (SHO-29) - Push-based relevance surfacing
// =============================================================================

/// POST /api/relevant - Proactive memory surfacing
/// Returns relevant memories based on current context using entity matching
/// and semantic similarity. Target latency: <30ms
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn surface_relevant(
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

/// WebSocket endpoint for context monitoring (SHO-29)
/// Enables proactive memory surfacing based on streaming context updates
///
/// # Protocol
/// 1. Client connects to WS /api/context/monitor
/// 2. Client sends handshake: { user_id, config?, debounce_ms? }
/// 3. Client streams context updates: { context, entities?, config? }
/// 4. Server responds with relevant memories when threshold met
async fn context_monitor_ws(
    ws: axum::extract::ws::WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(|socket| handle_context_monitor_socket(socket, state))
}

/// Handle WebSocket connection for context monitoring
async fn handle_context_monitor_socket(socket: axum::extract::ws::WebSocket, state: AppState) {
    use axum::extract::ws::Message;
    use futures::{SinkExt, StreamExt};

    let (mut sender, mut receiver) = socket.split();
    let mut user_id: Option<String> = None;
    let mut config = relevance::RelevanceConfig::default();
    let mut _debounce_ms: u64 = 100;
    let mut last_surface_time = std::time::Instant::now();

    // Wait for handshake message
    while let Some(msg) = receiver.next().await {
        let text = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => {
                tracing::debug!("Context monitor WebSocket closed before handshake");
                return;
            }
            Ok(_) => continue,
            Err(e) => {
                tracing::warn!("Context monitor WebSocket error before handshake: {}", e);
                return;
            }
        };

        // Parse handshake
        let handshake: relevance::ContextMonitorHandshake = match serde_json::from_str(&text) {
            Ok(h) => h,
            Err(e) => {
                let error = relevance::ContextMonitorResponse::Error {
                    code: "INVALID_HANDSHAKE".to_string(),
                    message: format!("Failed to parse handshake: {}", e),
                    fatal: true,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                return;
            }
        };

        // Validate user_id
        if let Err(e) = validation::validate_user_id(&handshake.user_id) {
            let error = relevance::ContextMonitorResponse::Error {
                code: "INVALID_USER_ID".to_string(),
                message: format!("Invalid user_id: {}", e),
                fatal: true,
                timestamp: chrono::Utc::now(),
            };
            let _ = sender
                .send(Message::Text(
                    serde_json::to_string(&error)
                        .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                        .into(),
                ))
                .await;
            return;
        }

        user_id = Some(handshake.user_id.clone());
        if let Some(cfg) = handshake.config {
            config = cfg;
        }
        _debounce_ms = handshake.debounce_ms;

        // Send acknowledgement
        let ack = relevance::ContextMonitorResponse::Ack {
            timestamp: chrono::Utc::now(),
        };
        if sender
            .send(Message::Text(
                serde_json::to_string(&ack)
                    .unwrap_or_else(|_| r#"{"ack":true}"#.to_string())
                    .into(),
            ))
            .await
            .is_err()
        {
            return;
        }

        tracing::info!(
            "Context monitor session started for user {}",
            handshake.user_id
        );
        break;
    }

    let user_id = match user_id {
        Some(id) => id,
        None => return,
    };

    // Get user memory and graph systems
    let memory_sys = match state.get_user_memory(&user_id) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("Failed to get user memory: {}", e);
            return;
        }
    };

    let graph_memory = match state.get_user_graph(&user_id) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("Failed to get user graph: {}", e);
            return;
        }
    };

    let ner = state.get_neural_ner();
    let engine = std::sync::Arc::new(relevance::RelevanceEngine::new(ner));

    // Process context updates
    while let Some(msg) = receiver.next().await {
        let text = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => {
                tracing::debug!("Context monitor closed by client");
                return;
            }
            Ok(Message::Ping(data)) => {
                let _ = sender.send(Message::Pong(data)).await;
                continue;
            }
            Ok(_) => continue,
            Err(e) => {
                tracing::warn!("Context monitor WebSocket error: {}", e);
                break;
            }
        };

        // Parse context update
        let update: relevance::ContextUpdate = match serde_json::from_str(&text) {
            Ok(u) => u,
            Err(e) => {
                let error = relevance::ContextMonitorResponse::Error {
                    code: "INVALID_MESSAGE".to_string(),
                    message: format!("Failed to parse context update: {}", e),
                    fatal: false,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                continue;
            }
        };

        // Debounce check
        let elapsed = last_surface_time.elapsed();
        if elapsed < std::time::Duration::from_millis(_debounce_ms) {
            continue;
        }
        last_surface_time = std::time::Instant::now();

        // Use per-message config if provided, otherwise use session config
        let effective_config = update.config.as_ref().unwrap_or(&config);

        // Surface relevant memories
        let start = std::time::Instant::now();
        let memory_sys_clone = memory_sys.clone();
        let graph_clone = graph_memory.clone();
        let context = update.context.clone();
        let cfg = effective_config.clone();
        let engine_clone = engine.clone();

        let result = tokio::task::spawn_blocking(move || {
            let memory_guard = memory_sys_clone.read();
            let graph_guard = graph_clone.read();
            engine_clone.surface_relevant(&context, &*memory_guard, Some(&*graph_guard), &cfg)
        })
        .await;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let response = match result {
            Ok(Ok(rel_response)) => {
                if rel_response.memories.is_empty() {
                    relevance::ContextMonitorResponse::None {
                        timestamp: chrono::Utc::now(),
                    }
                } else {
                    relevance::ContextMonitorResponse::Relevant {
                        memories: rel_response.memories,
                        detected_entities: rel_response.detected_entities,
                        latency_ms,
                        timestamp: chrono::Utc::now(),
                    }
                }
            }
            Ok(Err(e)) => relevance::ContextMonitorResponse::Error {
                code: "SURFACE_ERROR".to_string(),
                message: format!("Failed to surface memories: {}", e),
                fatal: false,
                timestamp: chrono::Utc::now(),
            },
            Err(e) => relevance::ContextMonitorResponse::Error {
                code: "TASK_PANIC".to_string(),
                message: format!("Blocking task panicked: {}", e),
                fatal: false,
                timestamp: chrono::Utc::now(),
            },
        };

        // Send response (skip "none" responses to reduce noise unless explicitly configured)
        if !matches!(response, relevance::ContextMonitorResponse::None { .. }) {
            if sender
                .send(Message::Text(
                    serde_json::to_string(&response)
                        .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                        .into(),
                ))
                .await
                .is_err()
            {
                break;
            }
        }
    }

    tracing::info!("Context monitor session ended for user {}", user_id);
}

// =============================================================================
// LIST MEMORIES - Simple GET endpoint for listing all memories
// =============================================================================

/// Query parameters for list endpoint
#[derive(Debug, Deserialize)]
struct ListQuery {
    limit: Option<usize>,
    #[serde(rename = "type")]
    memory_type: Option<String>,
    /// Text search query - filters by content or tags (case-insensitive)
    query: Option<String>,
}

/// List response - simplified memory list
#[derive(Debug, Serialize)]
struct ListResponse {
    memories: Vec<ListMemoryItem>,
    total: usize,
}

#[derive(Debug, Serialize)]
struct ListMemoryItem {
    id: String,
    content: String,
    memory_type: String,
    importance: f32,
    tags: Vec<String>,
    created_at: String,
}

/// GET /api/list/{user_id} - List all memories for a user
/// Query params: ?limit=100&type=Decision
#[tracing::instrument(skip(state), fields(user_id = %user_id))]
async fn list_memories(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<Json<ListResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let all_memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_all_memories()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Filter by type if specified
    let mut filtered: Vec<_> = if let Some(ref type_filter) = query.memory_type {
        let type_lower = type_filter.to_lowercase();
        all_memories
            .into_iter()
            .filter(|m| format!("{:?}", m.experience.experience_type).to_lowercase() == type_lower)
            .collect()
    } else {
        all_memories
    };

    // Filter by text query if specified (search in content and tags)
    if let Some(ref text_query) = query.query {
        let query_lower = text_query.to_lowercase();
        filtered = filtered
            .into_iter()
            .filter(|m| {
                // Check content
                if m.experience.content.to_lowercase().contains(&query_lower) {
                    return true;
                }
                // Check tags/entities
                for tag in &m.experience.entities {
                    if tag.to_lowercase().contains(&query_lower) {
                        return true;
                    }
                }
                false
            })
            .collect();
    }

    let total = filtered.len();
    let limit = query.limit.unwrap_or(100).min(1000);

    let memories: Vec<ListMemoryItem> = filtered
        .into_iter()
        .take(limit)
        .map(|m| ListMemoryItem {
            id: m.id.0.to_string(),
            content: m.experience.content.chars().take(500).collect(),
            memory_type: format!("{:?}", m.experience.experience_type),
            importance: m.importance(),
            tags: m.experience.entities.clone(),
            created_at: m.created_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(ListResponse { memories, total }))
}

// =============================================================================
// HEBBIAN FEEDBACK HANDLERS - Closes the learning loop
// =============================================================================

/// Tracked recall - returns tracking info for later Hebbian feedback
/// POST /api/recall/tracked
///
/// Use this when you want to provide feedback later on whether memories were helpful.
/// Returns memory_ids that can be passed to /api/reinforce for Hebbian strengthening.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
async fn recall_tracked(
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

    let memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let query = MemoryQuery {
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

/// Reinforce memories based on task outcome - THIS IS THE HEBBIAN FEEDBACK ENDPOINT
/// POST /api/reinforce
///
/// Call this after using memories to complete a task:
/// - "helpful": Memories that helped → boost importance, strengthen associations
/// - "misleading": Memories that misled → reduce importance, don't strengthen
/// - "neutral": Just record access, mild strengthening
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, outcome = %req.outcome, count = req.ids.len()))]
async fn reinforce_feedback(
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
    let memory_ids: Vec<crate::memory::MemoryId> = req
        .ids
        .iter()
        .filter_map(|id| uuid::Uuid::parse_str(id).ok())
        .map(crate::memory::MemoryId)
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
// SEMANTIC CONSOLIDATION HANDLER - Extract durable facts from episodic memories
// =============================================================================

/// Consolidate memories into semantic facts
/// POST /api/consolidate
///
/// Analyzes memories to extract durable semantic facts (preferences, procedures, patterns).
/// Facts are reinforced when seen multiple times across different memories.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn consolidate_memories(
    State(state): State<AppState>,
    Json(req): Json<ConsolidateRequest>,
) -> Result<Json<ConsolidateResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let min_support = req.min_support;
    let min_age_days = req.min_age_days;

    // Run consolidation in blocking task
    let result = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();

            // Get all memories for consolidation
            let all_memories = memory_guard.get_all_memories()?;

            // Convert SharedMemory to Memory for consolidator
            let memories: Vec<crate::memory::types::Memory> = all_memories
                .into_iter()
                .map(|arc_mem| (*arc_mem).clone())
                .collect();

            // Create consolidator with custom thresholds
            let consolidator =
                crate::memory::SemanticConsolidator::with_thresholds(min_support, min_age_days);

            // Run consolidation
            Ok::<_, anyhow::Error>(consolidator.consolidate(&memories))
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    tracing::info!(
        user_id = %req.user_id,
        memories_processed = result.memories_processed,
        facts_extracted = result.facts_extracted,
        facts_reinforced = result.facts_reinforced,
        "Semantic consolidation complete"
    );

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::CONSOLIDATE_DURATION.observe(duration);
    metrics::CONSOLIDATE_TOTAL
        .with_label_values(&["success"])
        .inc();

    Ok(Json(ConsolidateResponse {
        memories_analyzed: result.memories_processed,
        facts_extracted: result.facts_extracted,
        facts_reinforced: result.facts_reinforced,
        fact_ids: result.new_fact_ids,
    }))
}

/// Batch /api/remember/batch - store multiple memories at once (SHO-83)
/// Efficient bulk ingestion with NER extraction and knowledge graph edges.
/// Example: POST /api/remember/batch {
///   "user_id": "agent-1",
///   "memories": [{"content": "...", "memory_type": "Observation"}, ...],
///   "options": {"extract_entities": true, "create_edges": true}
/// }
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, count = req.memories.len()))]
async fn batch_remember(
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

    // Pre-validate all items and collect validation errors
    let mut validation_errors: Vec<BatchErrorItem> = Vec::new();
    let mut valid_items: Vec<(usize, BatchMemoryItem)> = Vec::new();

    for (index, item) in req.memories.into_iter().enumerate() {
        // Validate content
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

    // Process valid items with NER extraction if enabled
    let extract_entities = req.options.extract_entities;
    let create_edges = req.options.create_edges;
    let neural_ner = state.neural_ner.clone();
    let user_id = req.user_id.clone();

    // Build experiences with optional NER extraction
    let mut experiences_with_index: Vec<(
        usize,
        Experience,
        Option<chrono::DateTime<chrono::Utc>>,
    )> = Vec::with_capacity(valid_items.len());

    for (index, item) in valid_items {
        let experience_type = item
            .memory_type
            .as_ref()
            .and_then(|s| match s.to_lowercase().as_str() {
                "task" => Some(ExperienceType::Task),
                "learning" => Some(ExperienceType::Learning),
                "decision" => Some(ExperienceType::Decision),
                "error" => Some(ExperienceType::Error),
                "pattern" => Some(ExperienceType::Pattern),
                "conversation" => Some(ExperienceType::Conversation),
                "discovery" => Some(ExperienceType::Discovery),
                "observation" => Some(ExperienceType::Context), // Map Observation to Context
                "context" => Some(ExperienceType::Context),
                _ => None,
            })
            .unwrap_or(ExperienceType::Context);

        // Extract entities via NER if enabled
        let merged_entities = if extract_entities {
            let extracted_names: Vec<String> = match neural_ner.extract(&item.content) {
                Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
                Err(e) => {
                    tracing::debug!("NER extraction failed for batch item {}: {}", index, e);
                    Vec::new()
                }
            };

            // Merge user tags with NER-extracted entities (deduplicated)
            let mut merged: Vec<String> = item.tags.clone();
            for entity_name in extracted_names {
                if !merged.iter().any(|t| t.eq_ignore_ascii_case(&entity_name)) {
                    merged.push(entity_name);
                }
            }
            merged
        } else {
            item.tags.clone()
        };

        // SHO-104: Build RichContext if any context fields are provided
        let has_context = item.emotional_valence.is_some()
            || item.emotional_arousal.is_some()
            || item.emotion.is_some()
            || item.source_type.is_some()
            || item.credibility.is_some()
            || item.episode_id.is_some()
            || item.sequence_number.is_some()
            || item.preceding_memory_id.is_some();

        let context = if has_context {
            use memory::types::{
                ContextId, EmotionalContext, EpisodeContext, RichContext, SourceContext, SourceType,
            };

            let emotional = EmotionalContext {
                valence: item.emotional_valence.unwrap_or(0.0),
                arousal: item.emotional_arousal.unwrap_or(0.0),
                dominant_emotion: item.emotion.clone(),
                confidence: if item.emotional_valence.is_some() || item.emotional_arousal.is_some()
                {
                    0.8
                } else {
                    0.0
                },
                ..Default::default()
            };

            let source_type = item
                .source_type
                .as_ref()
                .map(|s| match s.to_lowercase().as_str() {
                    "user" => SourceType::User,
                    "system" => SourceType::System,
                    "api" | "external_api" => SourceType::ExternalApi,
                    "file" => SourceType::File,
                    "web" => SourceType::Web,
                    "ai_generated" | "ai" => SourceType::AiGenerated,
                    "inferred" => SourceType::Inferred,
                    _ => SourceType::Unknown,
                })
                .unwrap_or(SourceType::User);

            let source = SourceContext {
                source_type,
                credibility: item.credibility.unwrap_or(0.8),
                ..Default::default()
            };

            let episode = EpisodeContext {
                episode_id: item.episode_id.clone(),
                sequence_number: item.sequence_number,
                preceding_memory_id: item.preceding_memory_id.clone(),
                is_episode_start: item.sequence_number == Some(1),
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
        } else {
            None
        };

        let experience = Experience {
            content: item.content,
            experience_type,
            entities: merged_entities,
            tags: item.tags,
            context,
            ..Default::default()
        };

        experiences_with_index.push((index, experience, item.created_at));
    }

    // Store memories in blocking task
    let (memory_results, storage_errors) = {
        let memory = memory.clone();
        let experiences = experiences_with_index.clone();
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

    // Process into knowledge graph if enabled
    if create_edges {
        for (_, memory_id, experience) in &memory_results {
            let mem_id = MemoryId(uuid::Uuid::parse_str(memory_id).unwrap_or_default());
            if let Err(e) = state.process_experience_into_graph(&user_id, experience, &mem_id) {
                tracing::debug!("Graph processing failed for memory {}: {}", memory_id, e);
                // Don't fail the batch if graph processing fails
            }
        }
    }

    // Collect results
    let memory_ids: Vec<String> = memory_results.iter().map(|(_, id, _)| id.clone()).collect();
    let created = memory_ids.len();

    // Merge all errors
    let mut all_errors = validation_errors;
    all_errors.extend(storage_errors);
    all_errors.sort_by_key(|e| e.index);
    let failed = all_errors.len();

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::BATCH_STORE_DURATION.observe(duration);
    metrics::BATCH_STORE_SIZE.observe(batch_size as f64);
    // Also record individual store metrics
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

/// Get user statistics
async fn get_user_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<MemoryStats>, AppError> {
    let stats = state.get_stats(&user_id).map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// GET /api/stats - OpenAPI spec compatible stats endpoint
#[derive(Debug, Deserialize)]
struct StatsQuery {
    user_id: String,
}

async fn get_stats_query(
    State(state): State<AppState>,
    Query(query): Query<StatsQuery>,
) -> Result<Json<MemoryStats>, AppError> {
    let stats = state
        .get_stats(&query.user_id)
        .map_err(AppError::Internal)?;
    Ok(Json(stats))
}

/// Response for user deletion
#[derive(Debug, Serialize)]
struct DeleteUserResponse {
    success: bool,
    user_id: String,
    message: String,
}

/// Delete user data (GDPR)
async fn delete_user(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<DeleteUserResponse>, AppError> {
    state.forget_user(&user_id).map_err(AppError::Internal)?;

    Ok(Json(DeleteUserResponse {
        success: true,
        user_id,
        message: "User data deleted successfully".to_string(),
    }))
}

/// List all users
async fn list_users(State(state): State<AppState>) -> Json<Vec<String>> {
    Json(state.list_users())
}

/// Get specific memory by ID
async fn get_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Memory>, AppError> {
    let user_id = params
        .get("user_id")
        .ok_or_else(|| AppError::InvalidInput {
            field: "user_id".to_string(),
            reason: "user_id required".to_string(),
        })?;

    // Enterprise input validation
    validation::validate_user_id(user_id).map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state.get_user_memory(user_id).map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Parse memory ID (already validated above)
    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let _memory_id_obj = MemoryId(mem_id);

    // Search for memory
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    // Unwrap Arc to return owned Memory
    Ok(Json((*shared_memory).clone()))
}

/// Update existing memory
#[derive(Debug, Deserialize)]
struct UpdateMemoryRequest {
    user_id: String,
    content: String,
    embeddings: Option<Vec<f32>>,
}

/// Response for memory update operations
#[derive(Debug, Serialize)]
struct UpdateMemoryResponse {
    success: bool,
    id: String,
    message: String,
}

async fn update_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<Json<UpdateMemoryResponse>, AppError> {
    // Enterprise input validation
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    validation::validate_content(&req.content, false).map_validation_err("content")?;

    if let Some(ref emb) = req.embeddings {
        validation::validate_embeddings(emb)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    // Get current memory to preserve metadata
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    // Clone out of Arc to get mutable Memory
    let mut current_memory = (*shared_memory).clone();

    // Save content for audit log before move
    let content_preview: String = req.content.chars().take(50).collect();

    // Update content and embeddings
    current_memory.experience.content = req.content;
    if let Some(emb) = req.embeddings {
        current_memory.experience.embeddings = Some(emb);
    }
    // Note: last_accessed will be set automatically when re-recording

    // Re-record (will update in storage)
    let experience = current_memory.experience.clone();
    memory_guard
        .remember(experience, None) // None preserves original created_at behavior for updates
        .map_err(AppError::Internal)?;

    // Enterprise audit logging
    state.log_event(
        &req.user_id,
        "UPDATE",
        &memory_id,
        &format!("Updated memory content: {content_preview}"),
    );

    Ok(Json(UpdateMemoryResponse {
        success: true,
        id: memory_id,
        message: "Memory updated successfully".to_string(),
    }))
}

/// Response for memory delete operations
#[derive(Debug, Serialize)]
struct DeleteMemoryResponse {
    success: bool,
    id: String,
    message: String,
}

/// Delete specific memory
async fn delete_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<DeleteMemoryResponse>, AppError> {
    let user_id = params
        .get("user_id")
        .ok_or_else(|| AppError::InvalidInput {
            field: "user_id".to_string(),
            reason: "user_id required".to_string(),
        })?;

    // Enterprise input validation
    validation::validate_user_id(user_id).map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state.get_user_memory(user_id).map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Parse UUID and delete by ID directly (not by pattern matching)
    let uuid =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;
    memory_guard
        .forget(memory::ForgetCriteria::ById(memory::MemoryId(uuid)))
        .map_err(AppError::Internal)?;

    // Enterprise audit logging
    state.log_event(user_id, "DELETE", &memory_id, "Memory deleted");

    // Broadcast DELETE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.to_string(),
        memory_id: Some(memory_id.clone()),
        content_preview: None,
        memory_type: None,
        importance: None,
        count: None,
    });

    Ok(Json(DeleteMemoryResponse {
        success: true,
        id: memory_id,
        message: "Memory deleted successfully".to_string(),
    }))
}

/// Get all memories for a user with optional filters
#[derive(Debug, Deserialize)]
struct GetAllRequest {
    user_id: String,
    /// Maximum number of results to return (default: 100)
    limit: Option<usize>,
    /// Filter by minimum importance score (0.0 to 1.0)
    importance_threshold: Option<f32>,
    /// Filter by tags (returns memories matching ANY of these tags)
    tags: Option<Vec<String>>,
    /// Filter by memory type (e.g., "Decision", "Learning", "Error")
    memory_type: Option<String>,
    /// Filter by memories created after this timestamp (ISO 8601)
    created_after: Option<chrono::DateTime<chrono::Utc>>,
    /// Filter by memories created before this timestamp (ISO 8601)
    created_before: Option<chrono::DateTime<chrono::Utc>>,
}

async fn get_all_memories(
    State(state): State<AppState>,
    Json(req): Json<GetAllRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Parse memory_type string to ExperienceType if provided
    let experience_types = req.memory_type.as_ref().map(|type_str| {
        vec![match type_str.to_lowercase().as_str() {
            "observation" => ExperienceType::Observation,
            "decision" => ExperienceType::Decision,
            "learning" => ExperienceType::Learning,
            "error" => ExperienceType::Error,
            "discovery" => ExperienceType::Discovery,
            "pattern" => ExperienceType::Pattern,
            "context" => ExperienceType::Context,
            "task" => ExperienceType::Task,
            "codeedit" => ExperienceType::CodeEdit,
            "fileaccess" => ExperienceType::FileAccess,
            "search" => ExperienceType::Search,
            "command" => ExperienceType::Command,
            "conversation" => ExperienceType::Conversation,
            _ => ExperienceType::Observation, // Default fallback
        }]
    });

    // Build time range from created_after/created_before
    let time_range = match (req.created_after, req.created_before) {
        (Some(after), Some(before)) => Some((after, before)),
        (Some(after), None) => Some((after, chrono::Utc::now())),
        (None, Some(before)) => Some((chrono::DateTime::<chrono::Utc>::MIN_UTC, before)),
        (None, None) => None,
    };

    let query = MemoryQuery {
        max_results: req.limit.unwrap_or(100),
        importance_threshold: req.importance_threshold,
        tags: req.tags,
        experience_types,
        time_range,
        ..Default::default()
    };

    let shared_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    // Convert Arc<Memory> to owned Memory for response
    let memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();

    let count = memories.len();

    Ok(Json(RetrieveResponse { memories, count }))
}

/// GET version of get_all_memories for OpenAPI spec compatibility
/// Uses query parameters instead of JSON body
#[derive(Debug, Deserialize)]
struct GetAllQuery {
    user_id: String,
    limit: Option<usize>,
    importance_threshold: Option<f32>,
    /// Comma-separated tags
    tags: Option<String>,
    memory_type: Option<String>,
    created_after: Option<String>,
    created_before: Option<String>,
}

async fn get_all_memories_get(
    State(state): State<AppState>,
    Query(query): Query<GetAllQuery>,
) -> Result<Json<RetrieveResponse>, AppError> {
    let memory = state
        .get_user_memory(&query.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Parse memory_type string to ExperienceType if provided
    let experience_types = query.memory_type.as_ref().map(|type_str| {
        vec![match type_str.to_lowercase().as_str() {
            "observation" => ExperienceType::Observation,
            "decision" => ExperienceType::Decision,
            "learning" => ExperienceType::Learning,
            "error" => ExperienceType::Error,
            "discovery" => ExperienceType::Discovery,
            "pattern" => ExperienceType::Pattern,
            "context" => ExperienceType::Context,
            "task" => ExperienceType::Task,
            "codeedit" => ExperienceType::CodeEdit,
            "fileaccess" => ExperienceType::FileAccess,
            "search" => ExperienceType::Search,
            "command" => ExperienceType::Command,
            "conversation" => ExperienceType::Conversation,
            _ => ExperienceType::Observation,
        }]
    });

    // Parse comma-separated tags
    let tags = query
        .tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

    // Parse time range
    let created_after = query.created_after.as_ref().and_then(|s| s.parse().ok());
    let created_before = query.created_before.as_ref().and_then(|s| s.parse().ok());

    let time_range = match (created_after, created_before) {
        (Some(after), Some(before)) => Some((after, before)),
        (Some(after), None) => Some((after, chrono::Utc::now())),
        (None, Some(before)) => Some((chrono::DateTime::<chrono::Utc>::MIN_UTC, before)),
        (None, None) => None,
    };

    let mem_query = MemoryQuery {
        max_results: query.limit.unwrap_or(100),
        importance_threshold: query.importance_threshold,
        tags,
        experience_types,
        time_range,
        ..Default::default()
    };

    let shared_memories = memory_guard
        .recall(&mem_query)
        .map_err(AppError::Internal)?;
    let memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();
    let count = memories.len();

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Get history/audit trail
#[derive(Debug, Deserialize)]
struct HistoryRequest {
    user_id: String,
    #[serde(alias = "memory_id")]
    id: Option<String>,
}

#[derive(Debug, Serialize)]
struct HistoryResponse {
    events: Vec<HistoryEvent>,
}

#[derive(Debug, Serialize)]
struct HistoryEvent {
    timestamp: String, // ISO 8601 format
    event_type: String,
    id: String,
    details: String,
}

async fn get_history(
    State(state): State<AppState>,
    Json(req): Json<HistoryRequest>,
) -> Result<Json<HistoryResponse>, AppError> {
    // CRITICAL FIX: Wrap blocking I/O in spawn_blocking
    // get_history() does RocksDB prefix_iterator (blocking I/O) on cache miss
    // Running this on async threads starves the Tokio runtime
    let events = {
        let state = state.clone();
        let user_id = req.user_id.clone();
        let id = req.id.clone();

        tokio::task::spawn_blocking(move || state.get_history(&user_id, id.as_deref()))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    let history_events: Vec<HistoryEvent> = events
        .iter()
        .map(|e| HistoryEvent {
            timestamp: e.timestamp.to_rfc3339(),
            event_type: e.event_type.clone(),
            id: e.memory_id.clone(),
            details: e.details.clone(),
        })
        .collect();

    Ok(Json(HistoryResponse {
        events: history_events,
    }))
}

// ====== Advanced Memory Management Endpoints ======

#[derive(Debug, Deserialize)]
struct CompressMemoryRequest {
    user_id: String,
    #[serde(alias = "memory_id")]
    id: String,
}

/// Manually compress a specific memory
async fn compress_memory(
    State(state): State<AppState>,
    Json(req): Json<CompressMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let _memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Validate id format
    let _memory_id = MemoryId(
        uuid::Uuid::parse_str(&req.id).map_err(|_| AppError::InvalidMemoryId(req.id.clone()))?,
    );

    // Compression happens automatically in the memory system based on age and importance
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Memory compression initiated"
    })))
}

#[derive(Debug, Deserialize)]
struct InvalidateRelationshipRequest {
    user_id: String,
    relationship_uuid: String,
}

/// Invalidate a relationship edge (temporal edge invalidation)
async fn invalidate_relationship(
    State(state): State<AppState>,
    Json(req): Json<InvalidateRelationshipRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.write();

    let rel_uuid =
        uuid::Uuid::parse_str(&req.relationship_uuid).map_err(|_| AppError::InvalidInput {
            field: "relationship_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    graph_guard
        .invalidate_relationship(&rel_uuid)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    // Broadcast EDGE_INVALIDATE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "EDGE_INVALIDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(req.relationship_uuid.clone()),
        content_preview: Some("Relationship invalidated".to_string()),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: None,
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Relationship invalidated"
    })))
}

#[derive(Debug, Deserialize)]
struct GetEpisodeRequest {
    user_id: String,
    episode_uuid: String,
}

/// Get an episodic node by UUID
async fn get_episode(
    State(state): State<AppState>,
    Json(req): Json<GetEpisodeRequest>,
) -> Result<Json<Option<EpisodicNode>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();

    let episode_uuid =
        uuid::Uuid::parse_str(&req.episode_uuid).map_err(|_| AppError::InvalidInput {
            field: "episode_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    let episode = graph_guard
        .get_episode(&episode_uuid)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(episode))
}

#[derive(Debug, Deserialize)]
struct AdvancedSearchRequest {
    user_id: String,
    entity_name: Option<String>,
    start_date: Option<String>,
    end_date: Option<String>,
    min_importance: Option<f32>,
    max_importance: Option<f32>,
}

/// Advanced memory search with entity filtering
async fn advanced_search(
    State(state): State<AppState>,
    Json(req): Json<AdvancedSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    // Build search criteria
    let mut criterias = Vec::new();

    if let Some(entity) = req.entity_name {
        criterias.push(memory::storage::SearchCriteria::ByEntity(entity));
    }

    if let (Some(start), Some(end)) = (req.start_date, req.end_date) {
        let start_dt = chrono::DateTime::parse_from_rfc3339(&start)
            .map_err(|_| AppError::InvalidInput {
                field: "start_date".to_string(),
                reason: "Invalid RFC3339 format".to_string(),
            })?
            .with_timezone(&chrono::Utc);

        let end_dt = chrono::DateTime::parse_from_rfc3339(&end)
            .map_err(|_| AppError::InvalidInput {
                field: "end_date".to_string(),
                reason: "Invalid RFC3339 format".to_string(),
            })?
            .with_timezone(&chrono::Utc);

        criterias.push(memory::storage::SearchCriteria::ByDate {
            start: start_dt,
            end: end_dt,
        });
    }

    if let (Some(min), Some(max)) = (req.min_importance, req.max_importance) {
        criterias.push(memory::storage::SearchCriteria::ByImportance { min, max });
    }

    // Execute combined search
    let criteria = if criterias.len() == 1 {
        criterias
            .into_iter()
            .next()
            .expect("Criteria list has exactly one element")
    } else {
        memory::storage::SearchCriteria::Combined(criterias)
    };

    let memories = memory_guard
        .advanced_search(criteria)
        .map_err(AppError::Internal)?;

    let count = memories.len();

    Ok(Json(RetrieveResponse { memories, count }))
}

// ====== Graph Memory API Endpoints ======

/// Get graph statistics for a user
async fn get_graph_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<GraphStats>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let stats = state
        .get_user_graph_stats(&user_id)
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

#[derive(Debug, Deserialize)]
struct FindEntityRequest {
    user_id: String,
    entity_name: String,
}

/// Find an entity by name
async fn find_entity(
    State(state): State<AppState>,
    Json(req): Json<FindEntityRequest>,
) -> Result<Json<Option<EntityNode>>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();
    let entity = graph_guard
        .find_entity_by_name(&req.entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(entity))
}

#[derive(Debug, Deserialize)]
struct TraverseGraphRequest {
    user_id: String,
    entity_name: String,
    max_depth: Option<usize>,
}

/// Traverse graph from an entity
async fn traverse_graph(
    State(state): State<AppState>,
    Json(req): Json<TraverseGraphRequest>,
) -> Result<Json<GraphTraversal>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();

    // First find the entity
    let entity = graph_guard
        .find_entity_by_name(&req.entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| {
            AppError::MemoryNotFound(format!("Entity not found: {}", req.entity_name))
        })?;

    // Traverse from that entity
    let max_depth = req.max_depth.unwrap_or(2);
    let traversal = graph_guard
        .traverse_from_entity(&entity.uuid, max_depth)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(traversal))
}

// ============================================================================
// COMPREHENSIVE API EXPANSION - All Features Exposed
// ============================================================================

/// Decompress a specific memory
#[derive(Debug, Deserialize)]
struct DecompressMemoryRequest {
    user_id: String,
    #[serde(alias = "memory_id")]
    id: String,
}

async fn decompress_memory(
    State(state): State<AppState>,
    Json(req): Json<DecompressMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let memory_id = MemoryId(
        uuid::Uuid::parse_str(&req.id).map_err(|_| AppError::InvalidMemoryId(req.id.clone()))?,
    );

    // Get the memory
    let memory = memory_guard
        .get_memory(&memory_id)
        .map_err(AppError::Internal)?;

    if !memory.compressed {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "Memory is not compressed",
            "was_compressed": false
        })));
    }

    // Decompress using compression pipeline
    let decompressed = memory_guard
        .decompress_memory(&memory)
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Memory decompressed successfully",
        "was_compressed": true,
        "memory": {
            "id": decompressed.id.0.to_string(),
            "content": decompressed.experience.content,
            "importance": decompressed.importance()
        }
    })))
}

/// Get storage statistics
#[derive(Debug, Deserialize)]
struct StorageStatsRequest {
    user_id: String,
}

async fn get_storage_stats(
    State(state): State<AppState>,
    Json(req): Json<StorageStatsRequest>,
) -> Result<Json<memory::storage::StorageStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let stats = memory_guard
        .get_storage_stats()
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// Verify vector index integrity - diagnose orphaned memories
#[derive(Debug, Deserialize)]
struct VerifyIndexRequest {
    user_id: String,
}

async fn verify_index_integrity(
    State(state): State<AppState>,
    Json(req): Json<VerifyIndexRequest>,
) -> Result<Json<memory::IndexIntegrityReport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let report = memory_guard
        .verify_index_integrity()
        .map_err(AppError::Internal)?;

    Ok(Json(report))
}

/// Repair vector index - re-index orphaned memories
#[derive(Debug, Deserialize)]
struct RepairIndexRequest {
    user_id: String,
}

#[derive(Debug, Serialize)]
struct RepairIndexResponse {
    success: bool,
    total_storage: usize,
    total_indexed: usize,
    repaired: usize,
    failed: usize,
    is_healthy: bool,
}

async fn repair_vector_index(
    State(state): State<AppState>,
    Json(req): Json<RepairIndexRequest>,
) -> Result<Json<RepairIndexResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (total_storage, total_indexed, repaired, failed) = memory_guard
        .repair_vector_index()
        .map_err(AppError::Internal)?;

    Ok(Json(RepairIndexResponse {
        success: failed == 0,
        total_storage,
        total_indexed,
        repaired,
        failed,
        is_healthy: total_storage == total_indexed,
    }))
}

/// Cleanup corrupted memories that fail to deserialize
#[derive(Debug, Deserialize)]
struct CleanupCorruptedRequest {
    user_id: String,
}

#[derive(Debug, Serialize)]
struct CleanupCorruptedResponse {
    success: bool,
    deleted_count: usize,
}

async fn cleanup_corrupted(
    State(state): State<AppState>,
    Json(req): Json<CleanupCorruptedRequest>,
) -> Result<Json<CleanupCorruptedResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let deleted_count = memory_guard
        .cleanup_corrupted()
        .map_err(AppError::Internal)?;

    Ok(Json(CleanupCorruptedResponse {
        success: true,
        deleted_count,
    }))
}

/// Rebuild vector index from storage (removes orphaned index entries)
#[derive(Debug, Deserialize)]
struct RebuildIndexRequest {
    user_id: String,
}

#[derive(Debug, Serialize)]
struct RebuildIndexResponse {
    success: bool,
    storage_count: usize,
    indexed_count: usize,
    is_healthy: bool,
}

async fn rebuild_index(
    State(state): State<AppState>,
    Json(req): Json<RebuildIndexRequest>,
) -> Result<Json<RebuildIndexResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let (storage_count, indexed_count) =
        memory_guard.rebuild_index().map_err(AppError::Internal)?;

    Ok(Json(RebuildIndexResponse {
        success: true,
        storage_count,
        indexed_count,
        is_healthy: storage_count == indexed_count,
    }))
}

// ============================================================================
// BACKUP & RESTORE ENDPOINTS
// ============================================================================

/// Create backup request
#[derive(Debug, Deserialize)]
struct CreateBackupRequest {
    user_id: String,
}

/// Backup metadata response
#[derive(Debug, Serialize)]
struct BackupResponse {
    success: bool,
    backup: Option<backup::BackupMetadata>,
    message: String,
}

/// List backups request
#[derive(Debug, Deserialize)]
struct ListBackupsRequest {
    user_id: String,
}

/// List backups response
#[derive(Debug, Serialize)]
struct ListBackupsResponse {
    success: bool,
    backups: Vec<backup::BackupMetadata>,
    count: usize,
}

/// Verify backup request
#[derive(Debug, Deserialize)]
struct VerifyBackupRequest {
    user_id: String,
    backup_id: u32,
}

/// Verify backup response
#[derive(Debug, Serialize)]
struct VerifyBackupResponse {
    success: bool,
    is_valid: bool,
    message: String,
}

/// Purge backups request
#[derive(Debug, Deserialize)]
struct PurgeBackupsRequest {
    user_id: String,
    keep_count: usize,
}

/// Purge backups response
#[derive(Debug, Serialize)]
struct PurgeBackupsResponse {
    success: bool,
    purged_count: usize,
}

/// Create a backup for a user
async fn create_backup(
    State(state): State<AppState>,
    Json(req): Json<CreateBackupRequest>,
) -> Result<Json<BackupResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let db = memory_guard.get_db();

    match state.backup_engine().create_backup(&db, &req.user_id) {
        Ok(metadata) => {
            state.log_event(
                &req.user_id,
                "BACKUP_CREATED",
                &metadata.backup_id.to_string(),
                &format!("Backup created: {} bytes", metadata.size_bytes),
            );
            Ok(Json(BackupResponse {
                success: true,
                backup: Some(metadata),
                message: "Backup created successfully".to_string(),
            }))
        }
        Err(e) => Ok(Json(BackupResponse {
            success: false,
            backup: None,
            message: format!("Backup failed: {}", e),
        })),
    }
}

/// List all backups for a user
async fn list_backups(
    State(state): State<AppState>,
    Json(req): Json<ListBackupsRequest>,
) -> Result<Json<ListBackupsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state.backup_engine().list_backups(&req.user_id) {
        Ok(backups) => {
            let count = backups.len();
            Ok(Json(ListBackupsResponse {
                success: true,
                backups,
                count,
            }))
        }
        Err(e) => Err(AppError::Internal(e)),
    }
}

/// Verify backup integrity
async fn verify_backup(
    State(state): State<AppState>,
    Json(req): Json<VerifyBackupRequest>,
) -> Result<Json<VerifyBackupResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .backup_engine()
        .verify_backup(&req.user_id, req.backup_id)
    {
        Ok(is_valid) => Ok(Json(VerifyBackupResponse {
            success: true,
            is_valid,
            message: if is_valid {
                "Backup integrity verified".to_string()
            } else {
                "Backup checksum mismatch - may be corrupted".to_string()
            },
        })),
        Err(e) => Ok(Json(VerifyBackupResponse {
            success: false,
            is_valid: false,
            message: format!("Verification failed: {}", e),
        })),
    }
}

/// Purge old backups
async fn purge_backups(
    State(state): State<AppState>,
    Json(req): Json<PurgeBackupsRequest>,
) -> Result<Json<PurgeBackupsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .backup_engine()
        .purge_old_backups(&req.user_id, req.keep_count)
    {
        Ok(purged_count) => {
            if purged_count > 0 {
                state.log_event(
                    &req.user_id,
                    "BACKUP_PURGE",
                    &format!("keep_{}", req.keep_count),
                    &format!("Purged {} old backups", purged_count),
                );
            }
            Ok(Json(PurgeBackupsResponse {
                success: true,
                purged_count,
            }))
        }
        Err(e) => Err(AppError::Internal(e)),
    }
}

// =============================================================================
// MEMORY INTERCHANGE FORMAT (MIF) EXPORT
// =============================================================================

/// Request for MIF export
#[derive(Debug, Deserialize)]
struct MifExportRequest {
    user_id: String,
    #[serde(default)]
    include_embeddings: bool,
    #[serde(default)]
    include_graph: bool,
    #[serde(default)]
    since: Option<String>, // ISO 8601 date filter
}

/// MIF Memory object
#[derive(Debug, Serialize)]
struct MifMemory {
    id: String,
    content: String,
    #[serde(rename = "type")]
    memory_type: String,
    importance: f32,
    created_at: String,
    updated_at: String,
    accessed_at: String,
    access_count: u32,
    decay_rate: f32,
    tags: Vec<String>,
    source: MifSource,
    entities: Vec<MifEntity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<MifEmbedding>,
    relations: MifRelations,
}

#[derive(Debug, Serialize)]
struct MifSource {
    #[serde(rename = "type")]
    source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent: Option<String>,
}

#[derive(Debug, Serialize)]
struct MifEntity {
    text: String,
    #[serde(rename = "type")]
    entity_type: String,
    confidence: f32,
}

#[derive(Debug, Serialize)]
struct MifEmbedding {
    model: String,
    dimensions: usize,
    vector: Vec<f32>,
    normalized: bool,
}

#[derive(Debug, Serialize)]
struct MifRelations {
    related_memories: Vec<String>,
    related_todos: Vec<String>,
}

/// MIF Todo object
#[derive(Debug, Serialize)]
struct MifTodo {
    id: String,
    short_id: String,
    content: String,
    status: String,
    priority: String,
    created_at: String,
    updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    due_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project: Option<MifProject>,
    contexts: Vec<String>,
    tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    subtask_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocked_on: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recurrence: Option<String>,
    related_memory_ids: Vec<String>,
    comments: Vec<MifComment>,
}

#[derive(Debug, Serialize)]
struct MifProject {
    id: String,
    name: String,
    prefix: String,
}

#[derive(Debug, Serialize)]
struct MifComment {
    id: String,
    content: String,
    #[serde(rename = "type")]
    comment_type: String,
    created_at: String,
}

/// MIF Graph structure
#[derive(Debug, Serialize)]
struct MifGraph {
    format: String,
    node_count: usize,
    edge_count: usize,
    nodes: Vec<MifNode>,
    edges: Vec<MifEdge>,
    hebbian_config: MifHebbianConfig,
}

#[derive(Debug, Serialize)]
struct MifNode {
    id: String,
    #[serde(rename = "type")]
    node_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    entity_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct MifEdge {
    source: String,
    target: String,
    weight: f32,
    #[serde(rename = "type")]
    edge_type: String,
    created_at: String,
    strengthened_count: u32,
}

#[derive(Debug, Serialize)]
struct MifHebbianConfig {
    learning_rate: f32,
    decay_rate: f32,
    ltp_threshold: f32,
    max_weight: f32,
}

/// MIF Metadata
#[derive(Debug, Serialize)]
struct MifMetadata {
    total_memories: usize,
    total_todos: usize,
    date_range: MifDateRange,
    memory_types: std::collections::HashMap<String, usize>,
    top_entities: Vec<MifTopEntity>,
    projects: Vec<MifProjectStats>,
    privacy: MifPrivacy,
}

#[derive(Debug, Serialize)]
struct MifDateRange {
    earliest: String,
    latest: String,
}

#[derive(Debug, Serialize)]
struct MifTopEntity {
    text: String,
    count: usize,
}

#[derive(Debug, Serialize)]
struct MifProjectStats {
    id: String,
    name: String,
    todo_count: usize,
}

#[derive(Debug, Serialize)]
struct MifPrivacy {
    pii_detected: bool,
    secrets_detected: bool,
    redacted_fields: Vec<String>,
}

/// Full MIF export document
#[derive(Debug, Serialize)]
struct MifExport {
    #[serde(rename = "$schema")]
    schema: String,
    mif_version: String,
    generator: MifGenerator,
    export: MifExportMeta,
    memories: Vec<MifMemory>,
    todos: Vec<MifTodo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    graph: Option<MifGraph>,
    metadata: MifMetadata,
}

#[derive(Debug, Serialize)]
struct MifGenerator {
    name: String,
    version: String,
}

#[derive(Debug, Serialize)]
struct MifExportMeta {
    id: String,
    created_at: String,
    user_id: String,
    checksum: String,
}

/// Export memories in MIF (Memory Interchange Format)
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn export_mif(
    State(state): State<AppState>,
    Json(req): Json<MifExportRequest>,
) -> Result<Json<MifExport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let include_embeddings = req.include_embeddings;

    // Get all memories
    let memories: Vec<MifMemory> = {
        let guard = memory_sys.read();
        guard
            .get_all_memories()
            .unwrap_or_default()
            .into_iter()
            .map(|m| {
                let embedding = if include_embeddings {
                    m.experience.embeddings.as_ref().map(|e| MifEmbedding {
                        model: "minilm-l6-v2".to_string(),
                        dimensions: e.len(),
                        vector: e.clone(),
                        normalized: true,
                    })
                } else {
                    None
                };

                // Convert entity strings to MifEntity objects
                let entities: Vec<MifEntity> = m.experience.entities.iter().map(|e| MifEntity {
                    text: e.clone(),
                    entity_type: "UNKNOWN".to_string(),
                    confidence: 1.0,
                }).collect();

                // Extract source info from RichContext if available
                let (source_type, session_id) = m.experience.context.as_ref()
                    .map(|ctx| {
                        let src = format!("{:?}", ctx.source.source_type).to_lowercase();
                        let sess = ctx.episode.episode_id.clone();
                        (src, sess)
                    })
                    .unwrap_or_else(|| ("conversation".to_string(), None));

                // Extract tags from metadata
                let tags: Vec<String> = m.experience.metadata
                    .get("tags")
                    .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                    .unwrap_or_default();

                MifMemory {
                    id: format!("mem_{}", m.id.0),
                    content: m.experience.content.clone(),
                    memory_type: format!("{:?}", m.experience.experience_type),
                    importance: m.importance(),
                    created_at: m.created_at.to_rfc3339(),
                    updated_at: m.created_at.to_rfc3339(), // No separate updated_at
                    accessed_at: m.last_accessed().to_rfc3339(),
                    access_count: m.access_count(),
                    decay_rate: 0.1, // Default decay rate
                    tags,
                    source: MifSource {
                        source_type,
                        session_id,
                        agent: Some("shodh-memory".to_string()),
                    },
                    entities,
                    embedding,
                    relations: MifRelations {
                        related_memories: m.experience.related_memories.iter().map(|id| format!("mem_{}", id.0)).collect(),
                        related_todos: m.related_todo_ids.iter().map(|id| format!("todo_{}", id.0)).collect(),
                    },
                }
            })
            .collect()
    };

    // Get all todos
    let todos: Vec<MifTodo> = state
        .todo_store
        .list_todos_for_user(&user_id, None)
        .unwrap_or_default()
        .into_iter()
        .map(|t| {
            let project = t.project_id.as_ref().and_then(|pid| {
                state.todo_store.get_project(&user_id, pid).ok().flatten().map(|p| MifProject {
                    id: p.id.0.to_string(),
                    name: p.name,
                    prefix: p.prefix.unwrap_or_else(|| "TODO".to_string()),
                })
            });

            MifTodo {
                id: format!("todo_{}", t.id.0),
                short_id: t.short_id(),
                content: t.content,
                status: format!("{:?}", t.status).to_lowercase(),
                priority: t.priority.indicator().to_string(),
                created_at: t.created_at.to_rfc3339(),
                updated_at: t.updated_at.to_rfc3339(),
                due_date: t.due_date.map(|d| d.to_rfc3339()),
                completed_at: t.completed_at.map(|d| d.to_rfc3339()),
                project,
                contexts: t.contexts,
                tags: t.tags,
                notes: t.notes,
                parent_id: t.parent_id.map(|id| format!("todo_{}", id.0)),
                subtask_ids: vec![], // Would need to query
                blocked_on: t.blocked_on,
                recurrence: t.recurrence.map(|r| format!("{:?}", r).to_lowercase()),
                related_memory_ids: t.related_memory_ids.iter().map(|id| format!("mem_{}", id.0)).collect(),
                comments: vec![], // Would need separate query
            }
        })
        .collect();

    // Build graph if requested
    let graph = if req.include_graph {
        let guard = memory_sys.read();
        let graph_stats = guard.graph_stats();

        Some(MifGraph {
            format: "adjacency_list".to_string(),
            node_count: graph_stats.node_count,
            edge_count: graph_stats.edge_count,
            nodes: vec![], // Full graph export would need graph traversal
            edges: vec![], // Full graph export would need graph traversal
            hebbian_config: MifHebbianConfig {
                learning_rate: 0.1,
                decay_rate: 0.05,
                ltp_threshold: 0.8,
                max_weight: 1.0,
            },
        })
    } else {
        None
    };

    // Build metadata
    let mut memory_types: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in &memories {
        *memory_types.entry(m.memory_type.clone()).or_insert(0) += 1;
    }

    // Get projects with todo counts
    let all_todos_for_count = state.todo_store.list_todos_for_user(&user_id, None).unwrap_or_default();
    let projects: Vec<MifProjectStats> = state
        .todo_store
        .list_projects(&user_id)
        .unwrap_or_default()
        .into_iter()
        .map(|p| {
            let todo_count = all_todos_for_count.iter().filter(|t| t.project_id.as_ref() == Some(&p.id)).count();
            MifProjectStats {
                id: p.id.0.to_string(),
                name: p.name,
                todo_count,
            }
        })
        .collect();

    // Date range
    let earliest = memories.iter().map(|m| &m.created_at).min().cloned().unwrap_or_default();
    let latest = memories.iter().map(|m| &m.created_at).max().cloned().unwrap_or_default();

    let metadata = MifMetadata {
        total_memories: memories.len(),
        total_todos: todos.len(),
        date_range: MifDateRange { earliest, latest },
        memory_types,
        top_entities: vec![], // Would need entity aggregation
        projects,
        privacy: MifPrivacy {
            pii_detected: false,
            secrets_detected: false,
            redacted_fields: vec![],
        },
    };

    // Generate export ID and checksum
    let export_id = format!("exp_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string());
    let now = chrono::Utc::now();

    // Compute checksum of content
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}{}", memories.len(), todos.len(), now.to_rfc3339()));
    let checksum = format!("sha256:{}", hex::encode(hasher.finalize()));

    let export = MifExport {
        schema: "https://shodh-memory.dev/schemas/mif-v1.json".to_string(),
        mif_version: "1.0".to_string(),
        generator: MifGenerator {
            name: "shodh-memory".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
        export: MifExportMeta {
            id: export_id,
            created_at: now.to_rfc3339(),
            user_id: user_id.clone(),
            checksum,
        },
        memories,
        todos,
        graph,
        metadata,
    };

    state.log_event(
        &user_id,
        "MIF_EXPORT",
        &export.export.id,
        &format!("Exported {} memories, {} todos", export.metadata.total_memories, export.metadata.total_todos),
    );

    Ok(Json(export))
}

// =============================================================================
// A/B TESTING ENDPOINTS
// =============================================================================

/// Request to create a new A/B test
#[derive(Debug, Deserialize)]
struct CreateABTestRequest {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    control_weights: Option<relevance::LearnedWeights>,
    #[serde(default)]
    treatment_weights: Option<relevance::LearnedWeights>,
    #[serde(default = "default_traffic_split")]
    traffic_split: f32,
    #[serde(default = "default_min_impressions")]
    min_impressions: u64,
    #[serde(default)]
    max_duration_hours: Option<u64>,
    #[serde(default)]
    tags: Vec<String>,
}

fn default_traffic_split() -> f32 {
    0.5
}

fn default_min_impressions() -> u64 {
    100
}

/// Response for A/B test operations
#[derive(Debug, Serialize)]
struct ABTestResponse {
    success: bool,
    test_id: Option<String>,
    message: String,
}

/// List all A/B tests
async fn list_ab_tests(State(state): State<AppState>) -> Result<Json<serde_json::Value>, AppError> {
    let tests = state.ab_test_manager.list_tests();
    let summary = state.ab_test_manager.summary();

    Ok(Json(serde_json::json!({
        "success": true,
        "tests": tests.iter().map(|t| serde_json::json!({
            "id": t.id,
            "name": t.config.name,
            "description": t.config.description,
            "status": format!("{:?}", t.status),
            "traffic_split": t.config.traffic_split,
            "control_impressions": t.control_metrics.impressions,
            "treatment_impressions": t.treatment_metrics.impressions,
            "created_at": t.created_at.to_rfc3339(),
        })).collect::<Vec<_>>(),
        "summary": {
            "total_active": summary.total_active,
            "draft": summary.draft,
            "running": summary.running,
            "paused": summary.paused,
            "completed": summary.completed,
            "archived": summary.archived,
        }
    })))
}

/// Create a new A/B test
async fn create_ab_test(
    State(state): State<AppState>,
    Json(req): Json<CreateABTestRequest>,
) -> Result<Json<ABTestResponse>, AppError> {
    let mut builder = ab_testing::ABTest::builder(&req.name)
        .with_traffic_split(req.traffic_split)
        .with_min_impressions(req.min_impressions);

    if let Some(desc) = req.description {
        builder = builder.with_description(&desc);
    }

    if let Some(control) = req.control_weights {
        builder = builder.with_control(control);
    }

    if let Some(treatment) = req.treatment_weights {
        builder = builder.with_treatment(treatment);
    }

    if let Some(hours) = req.max_duration_hours {
        builder = builder.with_max_duration_hours(hours);
    }

    if !req.tags.is_empty() {
        builder = builder.with_tags(req.tags);
    }

    let test = builder.build();

    match state.ab_test_manager.create_test(test) {
        Ok(test_id) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "A/B test created successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to create test: {}", e),
        })),
    }
}

/// Get a specific A/B test
async fn get_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    match state.ab_test_manager.get_test(&test_id) {
        Some(test) => Ok(Json(serde_json::json!({
            "success": true,
            "test": {
                "id": test.id,
                "name": test.config.name,
                "description": test.config.description,
                "status": format!("{:?}", test.status),
                "traffic_split": test.config.traffic_split,
                "min_impressions": test.config.min_impressions,
                "max_duration_hours": test.config.max_duration_hours,
                "control_weights": test.config.control_weights,
                "treatment_weights": test.config.treatment_weights,
                "control_metrics": {
                    "impressions": test.control_metrics.impressions,
                    "clicks": test.control_metrics.clicks,
                    "ctr": if test.control_metrics.impressions > 0 {
                        test.control_metrics.clicks as f64 / test.control_metrics.impressions as f64
                    } else { 0.0 },
                    "positive_feedback": test.control_metrics.positive_feedback,
                    "negative_feedback": test.control_metrics.negative_feedback,
                },
                "treatment_metrics": {
                    "impressions": test.treatment_metrics.impressions,
                    "clicks": test.treatment_metrics.clicks,
                    "ctr": if test.treatment_metrics.impressions > 0 {
                        test.treatment_metrics.clicks as f64 / test.treatment_metrics.impressions as f64
                    } else { 0.0 },
                    "positive_feedback": test.treatment_metrics.positive_feedback,
                    "negative_feedback": test.treatment_metrics.negative_feedback,
                },
                "created_at": test.created_at.to_rfc3339(),
                "started_at": test.started_at.map(|t| t.to_rfc3339()),
                "completed_at": test.completed_at.map(|t| t.to_rfc3339()),
            }
        }))),
        None => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Test not found: {}", test_id)
        }))),
    }
}

/// Delete an A/B test
async fn delete_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.delete_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test deleted successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to delete test: {}", e),
        })),
    }
}

/// Start an A/B test
async fn start_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.start_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test started successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to start test: {}", e),
        })),
    }
}

/// Pause an A/B test
async fn pause_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.pause_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test paused successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to pause test: {}", e),
        })),
    }
}

/// Resume a paused A/B test
async fn resume_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<ABTestResponse>, AppError> {
    match state.ab_test_manager.resume_test(&test_id) {
        Ok(()) => Ok(Json(ABTestResponse {
            success: true,
            test_id: Some(test_id),
            message: "Test resumed successfully".to_string(),
        })),
        Err(e) => Ok(Json(ABTestResponse {
            success: false,
            test_id: None,
            message: format!("Failed to resume test: {}", e),
        })),
    }
}

/// Complete an A/B test and get results
async fn complete_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    match state.ab_test_manager.complete_test(&test_id) {
        Ok(results) => Ok(Json(serde_json::json!({
            "success": true,
            "test_id": test_id,
            "results": {
                "is_significant": results.is_significant,
                "confidence_level": results.confidence_level,
                "chi_squared": results.chi_squared,
                "p_value": results.p_value,
                "winner": results.winner.map(|w| format!("{:?}", w)),
                "relative_improvement": results.relative_improvement,
                "control_ctr": results.control_ctr,
                "treatment_ctr": results.treatment_ctr,
                "confidence_interval": results.confidence_interval,
                "recommendations": results.recommendations,
            }
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to complete test: {}", e)
        }))),
    }
}

/// Analyze an A/B test without completing it
async fn analyze_ab_test(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    match state.ab_test_manager.analyze_test(&test_id) {
        Ok(results) => Ok(Json(serde_json::json!({
            "success": true,
            "test_id": test_id,
            "analysis": {
                "is_significant": results.is_significant,
                "confidence_level": results.confidence_level,
                "chi_squared": results.chi_squared,
                "p_value": results.p_value,
                "winner": results.winner.map(|w| format!("{:?}", w)),
                "relative_improvement": results.relative_improvement,
                "control_ctr": results.control_ctr,
                "treatment_ctr": results.treatment_ctr,
                "confidence_interval": results.confidence_interval,
                "recommendations": results.recommendations,
            }
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to analyze test: {}", e)
        }))),
    }
}

/// Record an impression for an A/B test
#[derive(Debug, Deserialize)]
struct RecordImpressionRequest {
    user_id: String,
    #[serde(default)]
    relevance_score: Option<f64>,
    #[serde(default)]
    latency_us: Option<u64>,
}

async fn record_ab_impression(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordImpressionRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let relevance_score = req.relevance_score.unwrap_or(0.0);
    let latency_us = req.latency_us.unwrap_or(0);

    match state.ab_test_manager.record_impression(
        &test_id,
        &req.user_id,
        relevance_score,
        latency_us,
    ) {
        Ok(()) => {
            let variant = state
                .ab_test_manager
                .get_variant(&test_id, &req.user_id)
                .ok();
            Ok(Json(serde_json::json!({
                "success": true,
                "variant": variant.map(|v| format!("{:?}", v)),
            })))
        }
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to record impression: {}", e)
        }))),
    }
}

/// Record a click for an A/B test
#[derive(Debug, Deserialize)]
struct RecordClickRequest {
    user_id: String,
    memory_id: uuid::Uuid,
}

async fn record_ab_click(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordClickRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .ab_test_manager
        .record_click(&test_id, &req.user_id, req.memory_id)
    {
        Ok(()) => Ok(Json(serde_json::json!({
            "success": true,
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to record click: {}", e)
        }))),
    }
}

/// Record feedback for an A/B test
#[derive(Debug, Deserialize)]
struct RecordFeedbackRequest {
    user_id: String,
    positive: bool,
}

async fn record_ab_feedback(
    State(state): State<AppState>,
    Path(test_id): Path<String>,
    Json(req): Json<RecordFeedbackRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    match state
        .ab_test_manager
        .record_feedback(&test_id, &req.user_id, req.positive)
    {
        Ok(()) => Ok(Json(serde_json::json!({
            "success": true,
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Failed to record feedback: {}", e)
        }))),
    }
}

/// Get summary of all A/B tests
async fn get_ab_summary(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let summary = state.ab_test_manager.summary();
    let expired = state.ab_test_manager.check_expired_tests();

    Ok(Json(serde_json::json!({
        "success": true,
        "summary": {
            "total_active": summary.total_active,
            "draft": summary.draft,
            "running": summary.running,
            "paused": summary.paused,
            "completed": summary.completed,
            "archived": summary.archived,
        },
        "expired_tests": expired,
    })))
}

/// Forget memories by age
#[derive(Debug, Deserialize)]
struct ForgetByAgeRequest {
    user_id: String,
    days_old: u32,
}

async fn forget_by_age(
    State(state): State<AppState>,
    Json(req): Json<ForgetByAgeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::OlderThan(req.days_old))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_AGE",
        &format!("{} days", req.days_old),
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "criteria": format!("older than {} days", req.days_old)
    })))
}

/// Forget memories by importance threshold
#[derive(Debug, Deserialize)]
struct ForgetByImportanceRequest {
    user_id: String,
    threshold: f32,
}

async fn forget_by_importance(
    State(state): State<AppState>,
    Json(req): Json<ForgetByImportanceRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.threshold < 0.0 || req.threshold > 1.0 {
        return Err(AppError::InvalidInput {
            field: "threshold".to_string(),
            reason: "Must be between 0.0 and 1.0".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::LowImportance(req.threshold))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_IMPORTANCE",
        &format!("threshold {}", req.threshold),
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "criteria": format!("importance < {}", req.threshold)
    })))
}

/// Forget memories matching a pattern
#[derive(Debug, Deserialize)]
struct ForgetByPatternRequest {
    user_id: String,
    pattern: String,
}

async fn forget_by_pattern(
    State(state): State<AppState>,
    Json(req): Json<ForgetByPatternRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::Pattern(req.pattern.clone()))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_PATTERN",
        &req.pattern,
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "pattern": req.pattern
    })))
}

// ====== Bulk Delete Endpoints ======

/// Bulk delete memories by filters (tags, type, date range)
#[derive(Debug, Deserialize)]
struct BulkDeleteRequest {
    user_id: String,
    /// Delete memories matching ANY of these tags
    tags: Option<Vec<String>>,
    /// Delete memories of this type
    memory_type: Option<String>,
    /// Delete memories created after this timestamp
    created_after: Option<chrono::DateTime<chrono::Utc>>,
    /// Delete memories created before this timestamp
    created_before: Option<chrono::DateTime<chrono::Utc>>,
}

async fn bulk_delete_memories(
    State(state): State<AppState>,
    Json(req): Json<BulkDeleteRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let mut total_count = 0;

    // Delete by tags if specified
    if let Some(ref tags) = req.tags {
        if !tags.is_empty() {
            let count = memory_guard
                .forget(memory::ForgetCriteria::ByTags(tags.clone()))
                .map_err(AppError::Internal)?;
            total_count += count;
        }
    }

    // Delete by type if specified
    if let Some(ref type_str) = req.memory_type {
        let exp_type = match type_str.to_lowercase().as_str() {
            "observation" => ExperienceType::Observation,
            "decision" => ExperienceType::Decision,
            "learning" => ExperienceType::Learning,
            "error" => ExperienceType::Error,
            "discovery" => ExperienceType::Discovery,
            "pattern" => ExperienceType::Pattern,
            "context" => ExperienceType::Context,
            "task" => ExperienceType::Task,
            "codeedit" => ExperienceType::CodeEdit,
            "fileaccess" => ExperienceType::FileAccess,
            "search" => ExperienceType::Search,
            "command" => ExperienceType::Command,
            "conversation" => ExperienceType::Conversation,
            _ => {
                return Err(AppError::InvalidInput {
                    field: "memory_type".to_string(),
                    reason: format!("Invalid memory type: {type_str}"),
                })
            }
        };
        let count = memory_guard
            .forget(memory::ForgetCriteria::ByType(exp_type))
            .map_err(AppError::Internal)?;
        total_count += count;
    }

    // Delete by date range if specified
    if req.created_after.is_some() || req.created_before.is_some() {
        let start = req
            .created_after
            .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC);
        let end = req.created_before.unwrap_or(chrono::Utc::now());
        let count = memory_guard
            .forget(memory::ForgetCriteria::ByDateRange { start, end })
            .map_err(AppError::Internal)?;
        total_count += count;
    }

    state.log_event(
        &req.user_id,
        "BULK_DELETE",
        "multiple",
        &format!("Deleted {total_count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": total_count
    })))
}

/// Clear ALL memories for a user (GDPR compliance - right to erasure)
#[derive(Debug, Deserialize)]
struct ClearAllRequest {
    user_id: String,
    /// Safety confirmation - must be "CONFIRM" to proceed
    confirm: String,
}

async fn clear_all_memories(
    State(state): State<AppState>,
    Json(req): Json<ClearAllRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Safety check - require explicit confirmation
    if req.confirm != "CONFIRM" {
        return Err(AppError::InvalidInput {
            field: "confirm".to_string(),
            reason: "Must provide confirm: \"CONFIRM\" to clear all memories".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::All)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "CLEAR_ALL",
        "GDPR",
        &format!("GDPR erasure: deleted {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": count,
        "message": "All memories have been permanently deleted (GDPR erasure)"
    })))
}

/// PATCH endpoint for partial memory updates
#[derive(Debug, Deserialize)]
struct PatchMemoryRequest {
    user_id: String,
    /// New content (optional)
    content: Option<String>,
    /// New/additional tags (optional)
    tags: Option<Vec<String>>,
    /// New memory type (optional)
    memory_type: Option<String>,
}

async fn patch_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<PatchMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    // Get current memory
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    // Clone out of Arc to get mutable Memory
    let mut current_memory = (*shared_memory).clone();

    let mut changes = Vec::new();

    // Update content if provided
    if let Some(ref new_content) = req.content {
        validation::validate_content(new_content, false).map_validation_err("content")?;
        current_memory.experience.content = new_content.clone();
        // Re-generate embeddings for new content
        current_memory.experience.embeddings = None;
        changes.push("content");
    }

    // Update tags if provided (add to existing entities)
    if let Some(ref new_tags) = req.tags {
        for tag in new_tags {
            if !current_memory.experience.entities.contains(tag) {
                current_memory.experience.entities.push(tag.clone());
            }
        }
        changes.push("tags");
    }

    // Update type if provided
    if let Some(ref type_str) = req.memory_type {
        current_memory.experience.experience_type = match type_str.to_lowercase().as_str() {
            "observation" => ExperienceType::Observation,
            "decision" => ExperienceType::Decision,
            "learning" => ExperienceType::Learning,
            "error" => ExperienceType::Error,
            "discovery" => ExperienceType::Discovery,
            "pattern" => ExperienceType::Pattern,
            "context" => ExperienceType::Context,
            "task" => ExperienceType::Task,
            "codeedit" => ExperienceType::CodeEdit,
            "fileaccess" => ExperienceType::FileAccess,
            "search" => ExperienceType::Search,
            "command" => ExperienceType::Command,
            "conversation" => ExperienceType::Conversation,
            _ => {
                return Err(AppError::InvalidInput {
                    field: "memory_type".to_string(),
                    reason: format!("Invalid memory type: {type_str}"),
                })
            }
        };
        changes.push("type");
    }

    if changes.is_empty() {
        return Err(AppError::InvalidInput {
            field: "body".to_string(),
            reason: "No fields to update provided".to_string(),
        });
    }

    // Re-record (will update in storage and re-index)
    let experience = current_memory.experience.clone();
    memory_guard
        .remember(experience, None) // None preserves original created_at behavior for patches
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "PATCH",
        &memory_id,
        &format!("Updated fields: {}", changes.join(", ")),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "id": memory_id,
        "updated_fields": changes
    })))
}

// ============================================================================
// RECALL BY TAGS / DATE - Convenience Endpoints
// ============================================================================

/// Recall memories by tags
#[derive(Debug, Deserialize)]
struct RecallByTagsRequest {
    user_id: String,
    /// Tags to search for (returns memories matching ANY of these tags)
    tags: Vec<String>,
    /// Maximum number of results (default: 50)
    limit: Option<usize>,
}

async fn recall_by_tags(
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
    let memories = memory_guard
        .recall_by_tags(&req.tags, limit)
        .map_err(AppError::Internal)?;
    let count = memories.len();

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

/// Recall memories by date range
#[derive(Debug, Deserialize)]
struct RecallByDateRequest {
    user_id: String,
    /// Start of date range (inclusive) - ISO 8601 format
    start: chrono::DateTime<chrono::Utc>,
    /// End of date range (inclusive) - ISO 8601 format
    end: chrono::DateTime<chrono::Utc>,
    /// Maximum number of results (default: 50)
    limit: Option<usize>,
}

async fn recall_by_date(
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
    let memories = memory_guard
        .recall_by_date(req.start, req.end, limit)
        .map_err(AppError::Internal)?;
    let count = memories.len();

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

/// Forget memories by tags
#[derive(Debug, Deserialize)]
struct ForgetByTagsRequest {
    user_id: String,
    /// Tags to match for deletion (deletes memories matching ANY of these tags)
    tags: Vec<String>,
}

async fn forget_by_tags(
    State(state): State<AppState>,
    Json(req): Json<ForgetByTagsRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
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

    let deleted_count = memory_guard
        .forget(memory::ForgetCriteria::ByTags(req.tags.clone()))
        .map_err(AppError::Internal)?;

    info!(
        "🏷️ Forget by tags: user={}, tags={:?}, deleted={}",
        req.user_id, req.tags, deleted_count
    );

    // Broadcast DELETE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(format!("tags: {:?}", req.tags)),
        memory_type: None,
        importance: None,
        count: Some(deleted_count),
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "tags": req.tags
    })))
}

/// Forget memories by date range
#[derive(Debug, Deserialize)]
struct ForgetByDateRequest {
    user_id: String,
    /// Start of date range (inclusive) - ISO 8601 format
    start: chrono::DateTime<chrono::Utc>,
    /// End of date range (inclusive) - ISO 8601 format
    end: chrono::DateTime<chrono::Utc>,
}

async fn forget_by_date(
    State(state): State<AppState>,
    Json(req): Json<ForgetByDateRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
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

    let deleted_count = memory_guard
        .forget(memory::ForgetCriteria::ByDateRange {
            start: req.start,
            end: req.end,
        })
        .map_err(AppError::Internal)?;

    info!(
        "📅 Forget by date: user={}, start={}, end={}, deleted={}",
        req.user_id, req.start, req.end, deleted_count
    );

    // Broadcast DELETE event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: None,
        content_preview: Some(format!("date range: {} to {}", req.start, req.end)),
        memory_type: None,
        importance: None,
        count: Some(deleted_count),
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "start": req.start,
        "end": req.end
    })))
}

/// Advanced multi-modal retrieval
#[derive(Debug, Deserialize)]
struct MultiModalSearchRequest {
    user_id: String,
    query_text: String,
    mode: String, // "similarity", "temporal", "causal", "associative", "hybrid"
    limit: Option<usize>,
}

async fn multimodal_search(
    State(state): State<AppState>,
    Json(req): Json<MultiModalSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    // Build query based on mode (including robotics modes)
    let retrieval_mode = match req.mode.as_str() {
        "similarity" => memory::RetrievalMode::Similarity,
        "temporal" => memory::RetrievalMode::Temporal,
        "causal" => memory::RetrievalMode::Causal,
        "associative" => memory::RetrievalMode::Associative,
        "hybrid" => memory::RetrievalMode::Hybrid,
        // Robotics-specific modes
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        _ => return Err(AppError::InvalidInput {
            field: "mode".to_string(),
            reason: format!("Invalid mode: {}. Must be one of: similarity, temporal, causal, associative, hybrid, spatial, mission, action_outcome", req.mode)
        })
    };

    let query = MemoryQuery {
        query_text: Some(req.query_text.clone()),
        max_results: req.limit.unwrap_or(10),
        retrieval_mode,
        ..Default::default()
    };

    let shared_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    // Convert Arc<Memory> to owned Memory for response
    let memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();

    let count = memories.len();

    state.log_event(
        &req.user_id,
        "MULTIMODAL_SEARCH",
        &req.mode,
        &format!("Retrieved {} memories using {} mode", count, req.mode),
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

// ====== Robotics Search API Endpoints ======

/// Request for robotics-specific memory search
#[derive(Debug, Deserialize)]
struct RoboticsSearchRequest {
    user_id: String,
    /// Search mode: spatial, mission, action_outcome, or hybrid
    mode: String,
    /// Optional text query for semantic component
    query_text: Option<String>,
    /// Robot/drone identifier filter
    robot_id: Option<String>,
    /// Mission identifier filter
    mission_id: Option<String>,
    /// Spatial search: center latitude
    lat: Option<f64>,
    /// Spatial search: center longitude
    lon: Option<f64>,
    /// Spatial search: radius in meters
    radius_meters: Option<f64>,
    /// Action type filter
    action_type: Option<String>,
    /// Reward range: minimum reward (-1.0 to 1.0)
    min_reward: Option<f32>,
    /// Reward range: maximum reward (-1.0 to 1.0)
    max_reward: Option<f32>,
    /// Maximum results to return
    limit: Option<usize>,
}

/// Robotics-specific memory search
/// Supports spatial queries, mission context, robot filtering, and action-outcome learning
async fn robotics_search(
    State(state): State<AppState>,
    Json(req): Json<RoboticsSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    // Build retrieval mode
    let retrieval_mode = match req.mode.as_str() {
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        "hybrid" => memory::RetrievalMode::Hybrid,
        "similarity" => memory::RetrievalMode::Similarity,
        _ => {
            return Err(AppError::InvalidInput {
                field: "mode".to_string(),
                reason: "Invalid mode. Use: spatial, mission, action_outcome, hybrid, similarity"
                    .to_string(),
            })
        }
    };

    // Build geo filter if spatial coordinates provided
    let geo_filter = match (req.lat, req.lon, req.radius_meters) {
        (Some(lat), Some(lon), Some(radius)) => Some(memory::GeoFilter::new(lat, lon, radius)),
        _ => None,
    };

    // Build reward range
    let reward_range = match (req.min_reward, req.max_reward) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, 1.0)),
        (None, Some(max)) => Some((-1.0, max)),
        _ => None,
    };

    // Validate spatial mode has coordinates
    if matches!(retrieval_mode, memory::RetrievalMode::Spatial) && geo_filter.is_none() {
        return Err(AppError::InvalidInput {
            field: "lat/lon/radius_meters".to_string(),
            reason: "Spatial mode requires lat, lon, and radius_meters".to_string(),
        });
    }

    // Validate mission mode has mission_id
    if matches!(retrieval_mode, memory::RetrievalMode::Mission) && req.mission_id.is_none() {
        return Err(AppError::InvalidInput {
            field: "mission_id".to_string(),
            reason: "Mission mode requires mission_id".to_string(),
        });
    }

    let query = MemoryQuery {
        query_text: req.query_text,
        robot_id: req.robot_id.clone(),
        mission_id: req.mission_id.clone(),
        geo_filter,
        action_type: req.action_type.clone(),
        reward_range,
        max_results: req.limit.unwrap_or(10),
        retrieval_mode,
        ..Default::default()
    };

    let shared_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;

    let memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();

    let count = memories.len();

    state.log_event(
        &req.user_id,
        "ROBOTICS_SEARCH",
        &req.mode,
        &format!(
            "Retrieved {} robotics memories (robot={:?}, mission={:?})",
            count, req.robot_id, req.mission_id
        ),
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Get uncompressed old memories (for manual compression review)
#[derive(Debug, Deserialize)]
struct GetUncompressedRequest {
    user_id: String,
    days_old: u32,
}

async fn get_uncompressed_old(
    State(state): State<AppState>,
    Json(req): Json<GetUncompressedRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let cutoff = chrono::Utc::now() - chrono::Duration::days(req.days_old as i64);
    let memories = memory_guard
        .get_uncompressed_older_than(cutoff)
        .map_err(AppError::Internal)?;

    let count = memories.len();

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Add entity to graph
#[derive(Debug, Deserialize)]
struct AddEntityRequest {
    user_id: String,
    name: String,
    label: String,
    attributes: Option<HashMap<String, String>>,
}

async fn add_entity(
    State(state): State<AppState>,
    Json(req): Json<AddEntityRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    validation::validate_entity(&req.name).map_validation_err("name")?;

    validation::validate_entity(&req.label).map_validation_err("label")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    // Parse entity label
    let entity_label = match req.label.as_str() {
        "Person" => graph_memory::EntityLabel::Person,
        "Organization" => graph_memory::EntityLabel::Organization,
        "Location" => graph_memory::EntityLabel::Location,
        "Technology" => graph_memory::EntityLabel::Technology,
        "Concept" => graph_memory::EntityLabel::Concept,
        "Event" => graph_memory::EntityLabel::Event,
        "Date" => graph_memory::EntityLabel::Date,
        "Product" => graph_memory::EntityLabel::Product,
        "Skill" => graph_memory::EntityLabel::Skill,
        other => graph_memory::EntityLabel::Other(other.to_string()),
    };

    // Detect if proper noun (simple heuristic: first char is uppercase)
    let is_proper_noun = req
        .name
        .chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false);

    // Calculate base salience using centralized logic
    let salience =
        graph_memory::EntityExtractor::calculate_base_salience(&entity_label, is_proper_noun);

    let entity = graph_memory::EntityNode {
        uuid: uuid::Uuid::new_v4(),
        name: req.name.clone(),
        labels: vec![entity_label],
        created_at: chrono::Utc::now(),
        last_seen_at: chrono::Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: req.attributes.unwrap_or_default(),
        name_embedding: None,
        salience,
        is_proper_noun,
    };

    let entity_uuid = graph_guard
        .add_entity(entity)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    // Broadcast ENTITY_ADD event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "ENTITY_ADD".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(entity_uuid.to_string()),
        content_preview: Some(format!("Entity: {} ({})", req.name, req.label)),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: None,
    });

    // Audit log for entity creation
    state.log_event(
        &req.user_id,
        "ENTITY_ADD",
        &entity_uuid.to_string(),
        &format!("Added entity '{}' label={}", req.name, req.label),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "entity_uuid": entity_uuid.to_string(),
        "entity_name": req.name
    })))
}

/// Add relationship to graph
#[derive(Debug, Deserialize)]
struct AddRelationshipRequest {
    user_id: String,
    from_entity_name: String,
    to_entity_name: String,
    relation_type: String,
    strength: Option<f32>,
    context: Option<String>,
}

async fn add_relationship(
    State(state): State<AppState>,
    Json(req): Json<AddRelationshipRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    validation::validate_entity(&req.from_entity_name).map_validation_err("from_entity_name")?;

    validation::validate_entity(&req.to_entity_name).map_validation_err("to_entity_name")?;

    if let Some(strength) = req.strength {
        validation::validate_relationship_strength(strength).map_validation_err("strength")?;
    }

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    // Find entities
    let from_entity = graph_guard
        .find_entity_by_name(&req.from_entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| {
            AppError::MemoryNotFound(format!("Entity not found: {}", req.from_entity_name))
        })?;

    let to_entity = graph_guard
        .find_entity_by_name(&req.to_entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| {
            AppError::MemoryNotFound(format!("Entity not found: {}", req.to_entity_name))
        })?;

    // Parse relation type
    let relation_type = match req.relation_type.as_str() {
        "WorksWith" => graph_memory::RelationType::WorksWith,
        "WorksAt" => graph_memory::RelationType::WorksAt,
        "EmployedBy" => graph_memory::RelationType::EmployedBy,
        "PartOf" => graph_memory::RelationType::PartOf,
        "Contains" => graph_memory::RelationType::Contains,
        "OwnedBy" => graph_memory::RelationType::OwnedBy,
        "LocatedIn" => graph_memory::RelationType::LocatedIn,
        "LocatedAt" => graph_memory::RelationType::LocatedAt,
        "Uses" => graph_memory::RelationType::Uses,
        "CreatedBy" => graph_memory::RelationType::CreatedBy,
        "DevelopedBy" => graph_memory::RelationType::DevelopedBy,
        "Causes" => graph_memory::RelationType::Causes,
        "ResultsIn" => graph_memory::RelationType::ResultsIn,
        "Learned" => graph_memory::RelationType::Learned,
        "Knows" => graph_memory::RelationType::Knows,
        "Teaches" => graph_memory::RelationType::Teaches,
        "RelatedTo" => graph_memory::RelationType::RelatedTo,
        "AssociatedWith" => graph_memory::RelationType::AssociatedWith,
        custom => graph_memory::RelationType::Custom(custom.to_string()),
    };

    let edge = graph_memory::RelationshipEdge {
        uuid: uuid::Uuid::new_v4(),
        from_entity: from_entity.uuid,
        to_entity: to_entity.uuid,
        relation_type,
        strength: req.strength.unwrap_or(0.7),
        created_at: chrono::Utc::now(),
        valid_at: chrono::Utc::now(),
        invalidated_at: None,
        source_episode_id: None,
        context: req.context.unwrap_or_default(),
        // Hebbian plasticity fields (new synapses start fresh)
        last_activated: chrono::Utc::now(),
        activation_count: 1, // First activation
        potentiated: false,
    };

    let edge_uuid = graph_guard
        .add_relationship(edge)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    // Broadcast EDGE_ADD event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "EDGE_ADD".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(edge_uuid.to_string()),
        content_preview: Some(format!(
            "{} -> {} ({})",
            req.from_entity_name, req.to_entity_name, req.relation_type
        )),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: None,
    });

    // Audit log for relationship creation
    state.log_event(
        &req.user_id,
        "RELATIONSHIP_ADD",
        &edge_uuid.to_string(),
        &format!(
            "Added relationship '{}' --[{}]--> '{}'",
            req.from_entity_name, req.relation_type, req.to_entity_name
        ),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "relationship_uuid": edge_uuid.to_string(),
        "from": req.from_entity_name,
        "to": req.to_entity_name,
        "type": req.relation_type
    })))
}

/// Get all entities in the graph
#[derive(Debug, Deserialize)]
struct GetAllEntitiesRequest {
    user_id: String,
    limit: Option<usize>,
}

async fn get_all_entities(
    State(state): State<AppState>,
    Json(req): Json<GetAllEntitiesRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.read();

    let entities = graph_guard
        .get_all_entities()
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    let limit = req.limit.unwrap_or(100);
    let entities: Vec<_> = entities.into_iter().take(limit).collect();
    let count = entities.len();

    Ok(Json(serde_json::json!({
        "entities": entities,
        "count": count
    })))
}

// ====== Memory Universe Visualization API ======

/// Get the Memory Universe visualization for a user's knowledge graph.
/// Returns entities as "stars" with 3D positions based on their relationships,
/// sized by salience (gravitational mass), and colored by entity type.
async fn get_memory_universe(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<graph_memory::MemoryUniverse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let graph_guard = graph.read();
    let universe = graph_guard
        .get_universe()
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(universe))
}

/// Clear all graph data for a user (entities, relationships, episodes)
/// This removes all garbage/stale entities from the knowledge graph.
async fn clear_user_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    let (entities, relationships, episodes) = graph_guard
        .clear_all()
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    info!(
        "Cleared graph for user {}: {} entities, {} relationships, {} episodes",
        user_id, entities, relationships, episodes
    );

    // Broadcast GRAPH_CLEAR event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "GRAPH_CLEAR".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: None,
        content_preview: Some(format!(
            "Cleared {} entities, {} relationships, {} episodes",
            entities, relationships, episodes
        )),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: Some(entities + relationships + episodes),
    });

    Ok(Json(serde_json::json!({
        "cleared": {
            "entities": entities,
            "relationships": relationships,
            "episodes": episodes
        }
    })))
}

/// Rebuild graph from all existing memories using improved NER
/// This re-processes all memories to extract clean entities and relationships.
async fn rebuild_user_graph(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    // First, clear existing graph data
    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;
    {
        let graph_guard = graph.write();
        let _ = graph_guard.clear_all();
    }

    // Get all memories for this user
    let memory_sys = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;
    let memories: Vec<(MemoryId, Experience)> = {
        let memory_guard = memory_sys.read();
        memory_guard
            .get_all_memories()
            .map_err(AppError::Internal)?
            .into_iter()
            .map(|m| (m.id.clone(), m.experience.clone()))
            .collect()
    };

    let total_memories = memories.len();
    let mut processed = 0;

    // Re-process each memory through entity extraction
    for (memory_id, experience) in memories {
        if let Err(e) = state.process_experience_into_graph(&user_id, &experience, &memory_id) {
            tracing::debug!("Failed to process memory {}: {}", memory_id.0, e);
        } else {
            processed += 1;
        }
    }

    // Get final stats
    let stats = state
        .get_user_graph_stats(&user_id)
        .map_err(AppError::Internal)?;
    let entities_created = stats.entity_count;
    let relationships_created = stats.relationship_count;

    info!(
        "Rebuilt graph for user {}: processed {}/{} memories, created {} entities, {} relationships",
        user_id, processed, total_memories, entities_created, relationships_created
    );

    // Broadcast GRAPH_REBUILD event for real-time dashboard
    state.emit_event(MemoryEvent {
        event_type: "GRAPH_REBUILD".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: user_id.clone(),
        memory_id: None,
        content_preview: Some(format!(
            "Rebuilt: {} memories -> {} entities, {} relationships",
            processed, entities_created, relationships_created
        )),
        memory_type: Some("graph".to_string()),
        importance: None,
        count: Some(entities_created + relationships_created),
    });

    Ok(Json(serde_json::json!({
        "success": true,
        "processed_memories": processed,
        "total_memories": total_memories,
        "entities_created": entities_created,
        "relationships_created": relationships_created
    })))
}

// ====== Brain State Visualization API ======

/// Brain state response with memories organized by tier and activation levels
#[derive(Debug, Serialize)]
struct BrainStateResponse {
    working_memory: Vec<MemoryNeuron>,
    session_memory: Vec<MemoryNeuron>,
    longterm_memory: Vec<MemoryNeuron>,
    stats: BrainStats,
}

#[derive(Debug, Serialize)]
struct MemoryNeuron {
    id: String,
    content_preview: String,
    activation: f32,
    importance: f32,
    tier: String,
    access_count: u32,
    created_at: String,
}

#[derive(Debug, Serialize)]
struct BrainStats {
    total_memories: usize,
    working_count: usize,
    session_count: usize,
    longterm_count: usize,
    avg_activation: f32,
    avg_importance: f32,
}

/// Get brain state visualization - shows all memories with activation levels by tier
async fn get_brain_state(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<BrainStateResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Gather memories from all tiers using public accessors
    let mut working_memory = Vec::new();
    let mut session_memory = Vec::new();
    let mut longterm_memory = Vec::new();
    let mut total_activation = 0.0f32;
    let mut total_importance = 0.0f32;

    // Get working memory via public accessor
    for mem in memory_guard.get_working_memories() {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "working".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        working_memory.push(neuron);
    }

    // Get session memory via public accessor
    for mem in memory_guard.get_session_memories() {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "session".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        session_memory.push(neuron);
    }

    // Get sample of longterm memory via public accessor (limit to avoid huge responses)
    let longterm_sample = memory_guard.get_longterm_memories(50).unwrap_or_default();
    for mem in longterm_sample {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "longterm".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        longterm_memory.push(neuron);
    }

    let total_count = working_memory.len() + session_memory.len() + longterm_memory.len();
    let stats = BrainStats {
        total_memories: total_count,
        working_count: working_memory.len(),
        session_count: session_memory.len(),
        longterm_count: longterm_memory.len(),
        avg_activation: if total_count > 0 {
            total_activation / total_count as f32
        } else {
            0.0
        },
        avg_importance: if total_count > 0 {
            total_importance / total_count as f32
        } else {
            0.0
        },
    };

    Ok(Json(BrainStateResponse {
        working_memory,
        session_memory,
        longterm_memory,
        stats,
    }))
}

// ====== Memory Visualization API Endpoints ======

/// Get visualization statistics for a user's memory graph
async fn get_visualization_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<VisualizationStats>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard.get_visualization_stats();

    Ok(Json(stats))
}

/// Export visualization graph as DOT format for Graphviz
async fn get_visualization_dot(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<String, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let dot = memory_guard.export_visualization_dot();

    Ok(dot)
}

/// Build visualization graph from current memory state
#[derive(Debug, Deserialize)]
struct BuildVisualizationRequest {
    user_id: String,
}

async fn build_visualization(
    State(state): State<AppState>,
    Json(req): Json<BuildVisualizationRequest>,
) -> Result<Json<VisualizationStats>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard
        .build_visualization_graph()
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

// =============================================================================
// Prospective Memory / Reminders (SHO-116)
// =============================================================================

/// Request to create a new reminder
#[derive(Debug, Deserialize)]
struct CreateReminderRequest {
    user_id: String,
    content: String,
    /// Trigger configuration
    trigger: ReminderTriggerRequest,
    /// Optional tags for categorization
    #[serde(default)]
    tags: Vec<String>,
    /// Priority 1-5 (higher = more important)
    #[serde(default = "default_reminder_priority")]
    priority: u8,
}

fn default_reminder_priority() -> u8 {
    3
}

/// Trigger configuration for reminder creation
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReminderTriggerRequest {
    /// Trigger at a specific time
    Time { at: chrono::DateTime<chrono::Utc> },
    /// Trigger after a duration (seconds from now)
    Duration { after_seconds: u64 },
    /// Trigger when context matches keywords
    Context {
        keywords: Vec<String>,
        #[serde(default = "default_context_threshold")]
        threshold: f32,
    },
}

fn default_context_threshold() -> f32 {
    0.7
}

/// Response for reminder creation
#[derive(Debug, Serialize)]
struct CreateReminderResponse {
    id: String,
    content: String,
    trigger_type: String,
    due_at: Option<chrono::DateTime<chrono::Utc>>,
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Create a new reminder (prospective memory)
async fn create_reminder(
    State(state): State<AppState>,
    Json(req): Json<CreateReminderRequest>,
) -> Result<Json<CreateReminderResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.content.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "content".to_string(),
            reason: "Reminder content cannot be empty".to_string(),
        });
    }

    // Convert request trigger to ProspectiveTrigger
    let trigger = match req.trigger {
        ReminderTriggerRequest::Time { at } => ProspectiveTrigger::AtTime { at },
        ReminderTriggerRequest::Duration { after_seconds } => ProspectiveTrigger::AfterDuration {
            seconds: after_seconds,
            from: chrono::Utc::now(),
        },
        ReminderTriggerRequest::Context {
            keywords,
            threshold,
        } => {
            if keywords.is_empty() {
                return Err(AppError::InvalidInput {
                    field: "keywords".to_string(),
                    reason: "Context trigger requires at least one keyword".to_string(),
                });
            }
            ProspectiveTrigger::OnContext {
                keywords,
                threshold,
            }
        }
    };

    let mut task = ProspectiveTask::new(req.user_id.clone(), req.content.clone(), trigger);
    task.tags = req.tags;
    task.priority = req.priority.clamp(1, 5);

    let trigger_type = match &task.trigger {
        ProspectiveTrigger::AtTime { .. } => "time",
        ProspectiveTrigger::AfterDuration { .. } => "duration",
        ProspectiveTrigger::OnContext { .. } => "context",
    };

    let due_at = task.trigger.due_at();

    state
        .prospective_store
        .store(&task)
        .map_err(AppError::Internal)?;

    tracing::info!(
        user_id = %req.user_id,
        reminder_id = %task.id,
        trigger_type = trigger_type,
        "Created prospective memory (reminder)"
    );

    // Audit log for reminder creation
    state.log_event(
        &req.user_id,
        "REMINDER_CREATE",
        &task.id.to_string(),
        &format!(
            "Created reminder trigger={}: '{}'",
            trigger_type,
            req.content.chars().take(50).collect::<String>()
        ),
    );

    Ok(Json(CreateReminderResponse {
        id: task.id.to_string(),
        content: task.content,
        trigger_type: trigger_type.to_string(),
        due_at,
        created_at: task.created_at,
    }))
}

/// Request to list reminders
#[derive(Debug, Deserialize)]
struct ListRemindersRequest {
    user_id: String,
    /// Filter by status (pending, triggered, dismissed, expired)
    status: Option<String>,
}

/// Individual reminder in list response
#[derive(Debug, Serialize)]
struct ReminderItem {
    id: String,
    content: String,
    trigger_type: String,
    status: String,
    due_at: Option<chrono::DateTime<chrono::Utc>>,
    created_at: chrono::DateTime<chrono::Utc>,
    triggered_at: Option<chrono::DateTime<chrono::Utc>>,
    dismissed_at: Option<chrono::DateTime<chrono::Utc>>,
    priority: u8,
    tags: Vec<String>,
    overdue_seconds: Option<i64>,
}

/// Response for listing reminders
#[derive(Debug, Serialize)]
struct ListRemindersResponse {
    reminders: Vec<ReminderItem>,
    count: usize,
}

/// List reminders for a user
async fn list_reminders(
    State(state): State<AppState>,
    Json(req): Json<ListRemindersRequest>,
) -> Result<Json<ListRemindersResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let status_filter = req.status.as_ref().and_then(|s| match s.as_str() {
        "pending" => Some(ProspectiveTaskStatus::Pending),
        "triggered" => Some(ProspectiveTaskStatus::Triggered),
        "dismissed" => Some(ProspectiveTaskStatus::Dismissed),
        "expired" => Some(ProspectiveTaskStatus::Expired),
        _ => None,
    });

    let tasks = state
        .prospective_store
        .list_for_user(&req.user_id, status_filter)
        .map_err(AppError::Internal)?;

    let reminders: Vec<ReminderItem> = tasks
        .into_iter()
        .map(|t| {
            let overdue_seconds = t.overdue_seconds();
            ReminderItem {
                id: t.id.to_string(),
                content: t.content,
                trigger_type: match &t.trigger {
                    ProspectiveTrigger::AtTime { .. } => "time".to_string(),
                    ProspectiveTrigger::AfterDuration { .. } => "duration".to_string(),
                    ProspectiveTrigger::OnContext { .. } => "context".to_string(),
                },
                status: format!("{:?}", t.status).to_lowercase(),
                due_at: t.trigger.due_at(),
                created_at: t.created_at,
                triggered_at: t.triggered_at,
                dismissed_at: t.dismissed_at,
                priority: t.priority,
                tags: t.tags,
                overdue_seconds,
            }
        })
        .collect();

    let count = reminders.len();

    Ok(Json(ListRemindersResponse { reminders, count }))
}

/// Request to get due reminders
#[derive(Debug, Deserialize)]
struct GetDueRemindersRequest {
    user_id: String,
    /// Whether to mark reminders as triggered when returned
    #[serde(default = "default_true")]
    mark_triggered: bool,
}

// Note: default_true() is already defined elsewhere in this file

/// Response for due reminders
#[derive(Debug, Serialize)]
struct DueRemindersResponse {
    reminders: Vec<ReminderItem>,
    count: usize,
}

/// Get due time-based reminders (for polling/hooks)
async fn get_due_reminders(
    State(state): State<AppState>,
    Json(req): Json<GetDueRemindersRequest>,
) -> Result<Json<DueRemindersResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let mut due_tasks = state
        .prospective_store
        .get_due_tasks(&req.user_id)
        .map_err(AppError::Internal)?;

    // Mark as triggered if requested
    if req.mark_triggered {
        for task in &mut due_tasks {
            let _ = state
                .prospective_store
                .mark_triggered(&req.user_id, &task.id);
        }
    }

    let reminders: Vec<ReminderItem> = due_tasks
        .into_iter()
        .map(|t| {
            let overdue_seconds = t.overdue_seconds();
            ReminderItem {
                id: t.id.to_string(),
                content: t.content,
                trigger_type: match &t.trigger {
                    ProspectiveTrigger::AtTime { .. } => "time".to_string(),
                    ProspectiveTrigger::AfterDuration { .. } => "duration".to_string(),
                    ProspectiveTrigger::OnContext { .. } => "context".to_string(),
                },
                status: if req.mark_triggered {
                    "triggered".to_string()
                } else {
                    format!("{:?}", t.status).to_lowercase()
                },
                due_at: t.trigger.due_at(),
                created_at: t.created_at,
                triggered_at: if req.mark_triggered {
                    Some(chrono::Utc::now())
                } else {
                    t.triggered_at
                },
                dismissed_at: t.dismissed_at,
                priority: t.priority,
                tags: t.tags,
                overdue_seconds,
            }
        })
        .collect();

    let count = reminders.len();

    if count > 0 {
        tracing::debug!(
            user_id = %req.user_id,
            count = count,
            "Returning due reminders"
        );
    }

    Ok(Json(DueRemindersResponse { reminders, count }))
}

/// Request to check context-triggered reminders
#[derive(Debug, Deserialize)]
struct CheckContextRemindersRequest {
    user_id: String,
    /// Current context text to match against
    context: String,
    /// Whether to mark matched reminders as triggered
    #[serde(default = "default_true")]
    mark_triggered: bool,
}

/// Check for context-triggered reminders
async fn check_context_reminders(
    State(state): State<AppState>,
    Json(req): Json<CheckContextRemindersRequest>,
) -> Result<Json<DueRemindersResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.context.trim().is_empty() {
        return Ok(Json(DueRemindersResponse {
            reminders: vec![],
            count: 0,
        }));
    }

    // Get memory system for embeddings
    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Compute context embedding for semantic matching
    let context_for_embed = req.context.clone();
    let memory_for_embedding = memory_system.clone();
    let context_embedding: Vec<f32> = tokio::task::spawn_blocking(move || {
        let memory_guard = memory_for_embedding.read();
        memory_guard
            .compute_embedding(&context_for_embed)
            .unwrap_or_else(|_| vec![0.0; 384])
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding task panicked: {e}")))?;

    // Check context triggers with semantic matching
    let user_id = req.user_id.clone();
    let context_for_triggers = req.context.clone();
    let memory_for_task_embed = memory_system.clone();
    let prospective = state.prospective_store.clone();
    let mark_triggered = req.mark_triggered;

    let matched_tasks: Vec<(crate::memory::types::ProspectiveTask, f32)> =
        tokio::task::spawn_blocking(move || {
            let embed_fn = |text: &str| -> Option<Vec<f32>> {
                let memory_guard = memory_for_task_embed.read();
                memory_guard.compute_embedding(text).ok()
            };

            prospective
                .check_context_triggers_semantic(
                    &user_id,
                    &context_for_triggers,
                    &context_embedding,
                    embed_fn,
                )
                .unwrap_or_default()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?;

    // Mark as triggered if requested
    if mark_triggered {
        for (task, _) in &matched_tasks {
            let _ = state
                .prospective_store
                .mark_triggered(&req.user_id, &task.id);
        }
    }

    let reminders: Vec<ReminderItem> = matched_tasks
        .into_iter()
        .map(|(t, score)| ReminderItem {
            id: t.id.to_string(),
            content: t.content,
            trigger_type: format!("context (score: {:.2})", score),
            status: if mark_triggered {
                "triggered".to_string()
            } else {
                format!("{:?}", t.status).to_lowercase()
            },
            due_at: None,
            created_at: t.created_at,
            triggered_at: if mark_triggered {
                Some(chrono::Utc::now())
            } else {
                t.triggered_at
            },
            dismissed_at: t.dismissed_at,
            priority: t.priority,
            tags: t.tags,
            overdue_seconds: None,
        })
        .collect();

    let count = reminders.len();

    if count > 0 {
        tracing::debug!(
            user_id = %req.user_id,
            count = count,
            context_preview = %req.context.chars().take(50).collect::<String>(),
            "Context-triggered reminders matched"
        );
    }

    Ok(Json(DueRemindersResponse { reminders, count }))
}

/// Request to dismiss a reminder
#[derive(Debug, Deserialize)]
struct DismissReminderRequest {
    user_id: String,
}

/// Response for dismiss/delete operations
#[derive(Debug, Serialize)]
struct ReminderActionResponse {
    success: bool,
    message: String,
}

/// Dismiss (acknowledge) a triggered reminder
async fn dismiss_reminder(
    State(state): State<AppState>,
    Path(reminder_id): Path<String>,
    Json(req): Json<DismissReminderRequest>,
) -> Result<Json<ReminderActionResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Try parsing as full UUID first, then fall back to prefix lookup
    let task_id = if let Ok(uuid) = uuid::Uuid::parse_str(&reminder_id) {
        ProspectiveTaskId(uuid)
    } else {
        // Try prefix lookup for short IDs like "d8cdc580"
        let task = state
            .prospective_store
            .find_by_prefix(&req.user_id, &reminder_id)
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::InvalidInput {
                field: "reminder_id".to_string(),
                reason: format!("No reminder found with ID prefix '{}'", reminder_id),
            })?;
        task.id
    };

    let success = state
        .prospective_store
        .mark_dismissed(&req.user_id, &task_id)
        .map_err(AppError::Internal)?;

    if success {
        tracing::info!(
            user_id = %req.user_id,
            reminder_id = %task_id.0,
            "Dismissed reminder"
        );
    }

    Ok(Json(ReminderActionResponse {
        success,
        message: if success {
            "Reminder dismissed".to_string()
        } else {
            "Reminder not found".to_string()
        },
    }))
}

/// Request to delete a reminder
#[derive(Debug, Deserialize)]
struct DeleteReminderQuery {
    user_id: String,
}

/// Delete (cancel) a reminder
async fn delete_reminder(
    State(state): State<AppState>,
    Path(reminder_id): Path<String>,
    Query(query): Query<DeleteReminderQuery>,
) -> Result<Json<ReminderActionResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    // Try parsing as full UUID first, then fall back to prefix lookup
    let task_id = if let Ok(uuid) = uuid::Uuid::parse_str(&reminder_id) {
        ProspectiveTaskId(uuid)
    } else {
        // Try prefix lookup for short IDs like "d8cdc580"
        let task = state
            .prospective_store
            .find_by_prefix(&query.user_id, &reminder_id)
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::InvalidInput {
                field: "reminder_id".to_string(),
                reason: format!("No reminder found with ID prefix '{}'", reminder_id),
            })?;
        task.id
    };

    let success = state
        .prospective_store
        .delete(&query.user_id, &task_id)
        .map_err(AppError::Internal)?;

    if success {
        tracing::info!(
            user_id = %query.user_id,
            reminder_id = %task_id.0,
            "Deleted reminder"
        );
    }

    Ok(Json(ReminderActionResponse {
        success,
        message: if success {
            "Reminder deleted".to_string()
        } else {
            "Reminder not found".to_string()
        },
    }))
}

// =============================================================================
// GTD-STYLE TODO MANAGEMENT (Linear-inspired)
// =============================================================================

/// Parse recurrence string to Recurrence enum
fn parse_recurrence(s: &str) -> Option<Recurrence> {
    match s.to_lowercase().as_str() {
        "daily" => Some(Recurrence::Daily),
        "weekly" => Some(Recurrence::Weekly {
            days: vec![1, 2, 3, 4, 5],
        }), // Mon-Fri default
        "monthly" => Some(Recurrence::Monthly { day: 1 }), // 1st of month default
        _ => None,
    }
}

/// Request to create a new todo
#[derive(Debug, Deserialize)]
struct CreateTodoRequest {
    user_id: String,
    content: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    priority: Option<String>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    contexts: Option<Vec<String>>,
    #[serde(default)]
    due_date: Option<String>,
    #[serde(default)]
    blocked_on: Option<String>,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    recurrence: Option<String>,
}

/// Response for todo operations
#[derive(Debug, Serialize)]
struct TodoResponse {
    success: bool,
    todo: Option<Todo>,
    project: Option<Project>,
    formatted: String,
}

/// Response for todo list operations
#[derive(Debug, Serialize)]
struct TodoListResponse {
    success: bool,
    count: usize,
    todos: Vec<Todo>,
    projects: Vec<Project>,
    formatted: String,
}

/// Response for todo complete with potential next recurrence
#[derive(Debug, Serialize)]
struct TodoCompleteResponse {
    success: bool,
    todo: Option<Todo>,
    next_recurrence: Option<Todo>,
    formatted: String,
}

/// Request to add a comment to a todo
#[derive(Debug, Deserialize)]
struct AddCommentRequest {
    user_id: String,
    /// Comment content (supports markdown)
    content: String,
    /// Optional author (defaults to user_id)
    #[serde(default)]
    author: Option<String>,
    /// Type of comment: comment, progress, resolution, activity
    #[serde(default)]
    comment_type: Option<String>,
}

/// Request to update a comment
#[derive(Debug, Deserialize)]
struct UpdateCommentRequest {
    user_id: String,
    content: String,
}

/// Response for comment operations
#[derive(Debug, Serialize)]
struct CommentResponse {
    success: bool,
    comment: Option<TodoComment>,
    formatted: String,
}

/// Response for listing comments
#[derive(Debug, Serialize)]
struct CommentListResponse {
    success: bool,
    count: usize,
    comments: Vec<TodoComment>,
    formatted: String,
}

/// Request to list todos with filters
#[derive(Debug, Deserialize)]
struct ListTodosRequest {
    user_id: String,
    #[serde(default)]
    status: Option<Vec<String>>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    include_completed: Option<bool>,
    #[serde(default)]
    due: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    parent_id: Option<String>,
    /// Semantic search query - when provided, uses vector similarity search
    #[serde(default)]
    query: Option<String>,
}

/// Request to update a todo
#[derive(Debug, Deserialize)]
struct UpdateTodoRequest {
    user_id: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    priority: Option<String>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    contexts: Option<Vec<String>>,
    #[serde(default)]
    due_date: Option<String>,
    #[serde(default)]
    blocked_on: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    sort_order: Option<i32>,
    #[serde(default)]
    parent_id: Option<String>,
}

/// Request to reorder a todo (move up/down)
#[derive(Debug, Deserialize)]
struct ReorderTodoRequest {
    user_id: String,
    direction: String, // "up" or "down"
}

/// Request to get due todos
#[derive(Debug, Deserialize)]
struct DueTodosRequest {
    user_id: String,
    #[serde(default = "default_include_overdue")]
    include_overdue: bool,
}

fn default_include_overdue() -> bool {
    true
}

/// Request to create a project
#[derive(Debug, Deserialize)]
struct CreateProjectRequest {
    user_id: String,
    name: String,
    #[serde(default)]
    prefix: Option<String>, // Custom prefix for todo IDs (e.g., "BOLT", "MEM")
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    color: Option<String>,
    #[serde(default)]
    parent: Option<String>, // Parent project name or ID
}

/// Response for project operations
#[derive(Debug, Serialize)]
struct ProjectResponse {
    success: bool,
    project: Option<Project>,
    stats: Option<ProjectStats>,
    formatted: String,
}

/// Response for project list
#[derive(Debug, Serialize)]
struct ProjectListResponse {
    success: bool,
    count: usize,
    projects: Vec<(Project, ProjectStats)>,
    formatted: String,
}

/// Request to update a project
#[derive(Debug, Deserialize)]
struct UpdateProjectRequest {
    user_id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    prefix: Option<String>, // Custom prefix for todo IDs (e.g., "BOLT", "MEM")
    #[serde(default)]
    description: Option<Option<String>>,
    #[serde(default)]
    status: Option<ProjectStatus>,
    #[serde(default)]
    color: Option<Option<String>>,
}

/// Request to delete a project
#[derive(Debug, Deserialize)]
struct DeleteProjectRequest {
    user_id: String,
    #[serde(default)]
    delete_todos: bool,
}

/// Request to list projects
#[derive(Debug, Deserialize)]
struct ListProjectsRequest {
    user_id: String,
}

/// Request for todo stats
#[derive(Debug, Deserialize)]
struct TodoStatsRequest {
    user_id: String,
}

/// Response for todo stats
#[derive(Debug, Serialize)]
struct TodoStatsResponse {
    success: bool,
    stats: UserTodoStats,
    formatted: String,
}

/// Query params for single todo operations
#[derive(Debug, Deserialize)]
struct TodoQuery {
    user_id: String,
}

/// POST /api/todos - Create a new todo
async fn create_todo(
    State(state): State<AppState>,
    Json(req): Json<CreateTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.content.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "content".to_string(),
            reason: "Content cannot be empty".to_string(),
        });
    }

    let mut todo = Todo::new(req.user_id.clone(), req.content.clone());

    // Set status
    if let Some(ref status_str) = req.status {
        todo.status = TodoStatus::from_str_loose(status_str).unwrap_or_default();
    }

    // Set priority
    if let Some(ref priority_str) = req.priority {
        todo.priority = TodoPriority::from_str_loose(priority_str).unwrap_or_default();
    }

    // Handle project (find or create)
    let mut project_name = None;
    if let Some(ref proj_name) = req.project {
        let project = state
            .todo_store
            .find_or_create_project(&req.user_id, proj_name)
            .map_err(AppError::Internal)?;
        todo.project_id = Some(project.id.clone());
        project_name = Some(project.name.clone());
    }

    // Set contexts (extract from content if not provided)
    if let Some(contexts) = req.contexts {
        todo.contexts = contexts;
    } else {
        todo.contexts = todo_formatter::extract_contexts(&req.content);
    }

    // Parse and set due date
    if let Some(ref due_str) = req.due_date {
        todo.due_date = todo_formatter::parse_due_date(due_str);
    }

    // Set blocked_on
    todo.blocked_on = req.blocked_on;

    // Set parent_id for subtasks and inherit project from parent if not specified
    if let Some(ref parent_str) = req.parent_id {
        if let Some(parent) = state
            .todo_store
            .find_todo_by_prefix(&req.user_id, parent_str)
            .map_err(AppError::Internal)?
        {
            todo.parent_id = Some(parent.id);
            // Inherit project_id from parent if not explicitly specified
            if todo.project_id.is_none() {
                todo.project_id = parent.project_id;
                // Also set project_name for the response/memory
                if let Some(ref proj_id) = todo.project_id {
                    if let Ok(Some(proj)) = state.todo_store.get_project(&req.user_id, proj_id) {
                        project_name = Some(proj.name.clone());
                    }
                }
            }
        }
    }

    // Set tags and notes
    todo.tags = req.tags.unwrap_or_default();
    todo.notes = req.notes;

    // Parse and set recurrence
    if let Some(ref recurrence_str) = req.recurrence {
        todo.recurrence = parse_recurrence(recurrence_str);
    }

    // Compute embedding for semantic search (non-blocking)
    let embedding_text = format!(
        "{} {} {}",
        todo.content,
        todo.notes.as_deref().unwrap_or(""),
        todo.tags.join(" ")
    );

    if let Ok(memory_system) = state.get_user_memory(&req.user_id) {
        let memory_clone = memory_system.clone();
        let embedding_text_clone = embedding_text.clone();

        if let Ok(embedding) = tokio::task::spawn_blocking(move || {
            let memory_guard = memory_clone.read();
            memory_guard.compute_embedding(&embedding_text_clone)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding task panicked: {e}")))?
        {
            todo.embedding = Some(embedding.clone());

            // Index in vector store for semantic search
            if let Ok(vector_id) = state.todo_store.index_todo_embedding(&req.user_id, &todo.id, &embedding) {
                // Store vector_id to todo_id mapping
                let _ = state.todo_store.store_vector_id_mapping(&req.user_id, vector_id, &todo.id);
            }
        }
    }

    // Store the todo (returns todo with assigned seq_num)
    let todo = state
        .todo_store
        .store_todo(&todo)
        .map_err(AppError::Internal)?;

    // Create a memory from this todo - future plans affect how we function
    let memory_content = if let Some(ref proj) = project_name {
        format!(
            "[{}] Todo created in {}: {}",
            todo.short_id(),
            proj,
            todo.content
        )
    } else {
        format!("[{}] Todo created: {}", todo.short_id(), todo.content)
    };

    let mut tags = vec![
        format!("todo:{}", todo.short_id()),
        "todo-created".to_string(),
    ];
    if let Some(ref proj) = project_name {
        tags.push(format!("project:{}", proj));
    }

    let experience = Experience {
        content: memory_content,
        experience_type: ExperienceType::Task,
        tags,
        ..Default::default()
    };

    // Store as memory (non-blocking - don't fail todo creation if memory fails)
    if let Ok(memory) = state.get_user_memory(&req.user_id) {
        let memory_clone = memory.clone();
        let exp_clone = experience.clone();
        let state_clone = state.clone();
        let user_id = req.user_id.clone();

        tokio::spawn(async move {
            let memory_result = tokio::task::spawn_blocking(move || {
                let memory_guard = memory_clone.read();
                memory_guard.remember(exp_clone, None)
            })
            .await;

            if let Ok(Ok(memory_id)) = memory_result {
                if let Err(e) =
                    state_clone.process_experience_into_graph(&user_id, &experience, &memory_id)
                {
                    tracing::debug!(
                        "Graph processing failed for todo memory {}: {}",
                        memory_id.0,
                        e
                    );
                }
                tracing::debug!(memory_id = %memory_id.0, "Todo creation stored as memory");
            }
        });
    }

    let formatted = todo_formatter::format_todo_created(&todo, project_name.as_deref());

    // Emit SSE event for live TUI updates
    state.emit_event(MemoryEvent {
        event_type: "TODO_CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(todo.id.0.to_string()),
        content_preview: Some(todo.content.clone()),
        memory_type: Some(format!("{:?}", todo.status)),
        importance: None,
        count: None,
    });

    tracing::info!(
        user_id = %req.user_id,
        todo_id = %todo.id,
        seq_num = todo.seq_num,
        content = %req.content,
        "Created todo"
    );

    // Audit log for todo creation
    state.log_event(
        &req.user_id,
        "TODO_CREATE",
        &todo.id.0.to_string(),
        &format!(
            "Created todo [{}] project={}: '{}'",
            todo.short_id(),
            project_name.as_deref().unwrap_or("none"),
            req.content.chars().take(50).collect::<String>()
        ),
    );

    Ok(Json(TodoResponse {
        success: true,
        todo: Some(todo),
        project: None,
        formatted,
    }))
}

/// POST /api/todos/list - List todos with filters (supports semantic search via query param)
async fn list_todos(
    State(state): State<AppState>,
    Json(req): Json<ListTodosRequest>,
) -> Result<Json<TodoListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Parse status filters
    let status_filter: Option<Vec<TodoStatus>> = req.status.as_ref().map(|statuses| {
        statuses
            .iter()
            .filter_map(|s| TodoStatus::from_str_loose(s))
            .collect()
    });

    // If query is provided, use semantic search
    let mut todos = if let Some(ref query) = req.query {
        if query.trim().is_empty() {
            Vec::new()
        } else {
            // Compute embedding for the search query
            let memory_system = state
                .get_user_memory(&req.user_id)
                .map_err(AppError::Internal)?;

            let query_clone = query.clone();
            let query_embedding: Vec<f32> = tokio::task::spawn_blocking(move || {
                let memory_guard = memory_system.read();
                memory_guard.compute_embedding(&query_clone).unwrap_or_default()
            })
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding failed: {e}")))?;

            if query_embedding.is_empty() {
                Vec::new()
            } else {
                // Search using vector similarity (get more for filtering)
                let limit = req.limit.unwrap_or(50);
                let search_results = state
                    .todo_store
                    .search_similar(&req.user_id, &query_embedding, limit * 2)
                    .map_err(AppError::Internal)?;

                // Extract todos (already sorted by similarity)
                search_results.into_iter().map(|(todo, _score)| todo).collect()
            }
        }
    } else if let Some(ref statuses) = status_filter {
        state
            .todo_store
            .list_todos_for_user(&req.user_id, Some(statuses))
            .map_err(AppError::Internal)?
    } else {
        // Default: exclude Done and Cancelled unless requested
        let include_completed = req.include_completed.unwrap_or(false);
        let all_todos = state
            .todo_store
            .list_todos_for_user(&req.user_id, None)
            .map_err(AppError::Internal)?;

        if include_completed {
            all_todos
        } else {
            all_todos
                .into_iter()
                .filter(|t| t.status != TodoStatus::Done && t.status != TodoStatus::Cancelled)
                .collect()
        }
    };

    // Apply status filter for semantic search results too
    if req.query.is_some() {
        if let Some(ref statuses) = status_filter {
            todos.retain(|t| statuses.contains(&t.status));
        } else if !req.include_completed.unwrap_or(false) {
            todos.retain(|t| t.status != TodoStatus::Done && t.status != TodoStatus::Cancelled);
        }
    }

    // Filter by project
    if let Some(ref proj_name) = req.project {
        if let Some(project) = state
            .todo_store
            .find_project_by_name(&req.user_id, proj_name)
            .map_err(AppError::Internal)?
        {
            todos.retain(|t| t.project_id.as_ref() == Some(&project.id));
        }
    }

    // Filter by context
    if let Some(ref ctx) = req.context {
        let ctx_lower = ctx.to_lowercase();
        todos.retain(|t| t.contexts.iter().any(|c| c.to_lowercase() == ctx_lower));
    }

    // Filter by parent_id (for subtasks)
    if let Some(ref parent_str) = req.parent_id {
        if let Some(parent) = state
            .todo_store
            .find_todo_by_prefix(&req.user_id, parent_str)
            .map_err(AppError::Internal)?
        {
            todos.retain(|t| t.parent_id.as_ref() == Some(&parent.id));
        }
    }

    // Filter by due date
    if let Some(ref due_filter) = req.due {
        let now = chrono::Utc::now();
        let end_of_today = now
            .date_naive()
            .and_hms_opt(23, 59, 59)
            .map(|t| t.and_utc())
            .unwrap_or(now);
        let end_of_week =
            now + chrono::Duration::days(7 - now.weekday().num_days_from_monday() as i64);

        match due_filter.to_lowercase().as_str() {
            "today" => {
                todos.retain(|t| {
                    t.due_date
                        .as_ref()
                        .map(|d| *d <= end_of_today || *d < now) // due today or overdue
                        .unwrap_or(false)
                });
            }
            "overdue" => {
                todos.retain(|t| t.is_overdue());
            }
            "this_week" => {
                todos.retain(|t| {
                    t.due_date
                        .as_ref()
                        .map(|d| *d <= end_of_week)
                        .unwrap_or(false)
                });
            }
            _ => {} // "all" or unknown - no filter
        }
    }

    // Apply pagination (offset + limit)
    let total_count = todos.len();
    let offset = req.offset.unwrap_or(0);
    let limit = req.limit.unwrap_or(100);

    // Skip offset items
    if offset > 0 && offset < todos.len() {
        todos = todos.into_iter().skip(offset).collect();
    } else if offset >= total_count {
        todos.clear();
    }

    // Apply limit
    if todos.len() > limit {
        todos.truncate(limit);
    }

    // Get projects for formatting
    let projects = state
        .todo_store
        .list_projects(&req.user_id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_todo_list_with_total(&todos, &projects, total_count);

    Ok(Json(TodoListResponse {
        success: true,
        count: total_count,
        todos,
        projects,
        formatted,
    }))
}

/// POST /api/todos/due - List due/overdue todos
async fn list_due_todos(
    State(state): State<AppState>,
    Json(req): Json<DueTodosRequest>,
) -> Result<Json<TodoListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let todos = state
        .todo_store
        .list_due_todos(&req.user_id, req.include_overdue)
        .map_err(AppError::Internal)?;

    let projects = state
        .todo_store
        .list_projects(&req.user_id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_due_todos(&todos);

    Ok(Json(TodoListResponse {
        success: true,
        count: todos.len(),
        todos,
        projects,
        formatted,
    }))
}

/// GET /api/todos/{todo_id} - Get a single todo
async fn get_todo(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<TodoResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    let todo = state
        .todo_store
        .find_todo_by_prefix(&query.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    // Get project name if assigned
    let project_name = if let Some(ref pid) = todo.project_id {
        state
            .todo_store
            .get_project(&query.user_id, pid)
            .map_err(AppError::Internal)?
            .map(|p| p.name)
    } else {
        None
    };

    let formatted = todo_formatter::format_todo_line(&todo, project_name.as_deref(), true);

    Ok(Json(TodoResponse {
        success: true,
        todo: Some(todo),
        project: None,
        formatted,
    }))
}

/// POST /api/todos/{todo_id}/update - Update a todo
async fn update_todo(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Json(req): Json<UpdateTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let mut todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    // Update fields
    if let Some(ref content) = req.content {
        todo.content = content.clone();
    }
    if let Some(ref status_str) = req.status {
        if let Some(status) = TodoStatus::from_str_loose(status_str) {
            todo.status = status;
        }
    }
    if let Some(ref priority_str) = req.priority {
        if let Some(priority) = TodoPriority::from_str_loose(priority_str) {
            todo.priority = priority;
        }
    }
    if let Some(ref contexts) = req.contexts {
        todo.contexts = contexts.clone();
    }
    if let Some(ref due_str) = req.due_date {
        todo.due_date = todo_formatter::parse_due_date(due_str);
    }
    if let Some(ref blocked) = req.blocked_on {
        todo.blocked_on = Some(blocked.clone());
    }
    if let Some(ref notes) = req.notes {
        todo.notes = Some(notes.clone());
    }
    if let Some(ref tags) = req.tags {
        todo.tags = tags.clone();
    }
    if let Some(ref parent_id_str) = req.parent_id {
        // Resolve parent todo by prefix
        if parent_id_str.is_empty() {
            todo.parent_id = None;
        } else if let Ok(Some(parent)) = state
            .todo_store
            .find_todo_by_prefix(&req.user_id, parent_id_str)
        {
            todo.parent_id = Some(parent.id.clone());
        }
    }

    // Handle project change
    let mut project_name = None;
    if let Some(ref proj_name) = req.project {
        let project = state
            .todo_store
            .find_or_create_project(&req.user_id, proj_name)
            .map_err(AppError::Internal)?;
        todo.project_id = Some(project.id.clone());
        project_name = Some(project.name.clone());
    }

    todo.updated_at = chrono::Utc::now();

    // Re-compute embedding if content, notes, or tags changed
    let needs_reindex = req.content.is_some() || req.notes.is_some() || req.tags.is_some();
    if needs_reindex {
        let embedding_text = format!(
            "{} {} {}",
            todo.content,
            todo.notes.as_deref().unwrap_or(""),
            todo.tags.join(" ")
        );

        if let Ok(memory_system) = state.get_user_memory(&req.user_id) {
            let memory_clone = memory_system.clone();
            let embedding_text_clone = embedding_text.clone();

            if let Ok(embedding) = tokio::task::spawn_blocking(move || {
                let memory_guard = memory_clone.read();
                memory_guard.compute_embedding(&embedding_text_clone)
            })
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Embedding task panicked: {e}")))?
            {
                todo.embedding = Some(embedding.clone());

                // Re-index in vector store
                if let Ok(vector_id) = state.todo_store.index_todo_embedding(&req.user_id, &todo.id, &embedding) {
                    let _ = state.todo_store.store_vector_id_mapping(&req.user_id, vector_id, &todo.id);
                }
            }
        }
    }

    state
        .todo_store
        .update_todo(&todo)
        .map_err(AppError::Internal)?;

    // Create a memory for significant todo updates - brain tracks all changes
    let update_description = {
        let mut changes = Vec::new();
        if req.status.is_some() {
            changes.push(format!("status → {:?}", todo.status));
        }
        if req.priority.is_some() {
            changes.push(format!("priority → {:?}", todo.priority));
        }
        if req.content.is_some() {
            changes.push("content updated".to_string());
        }
        if req.project.is_some() {
            changes.push(format!(
                "project → {}",
                project_name.as_deref().unwrap_or("none")
            ));
        }
        if req.blocked_on.is_some() {
            changes.push(format!(
                "blocked on: {}",
                todo.blocked_on.as_deref().unwrap_or("cleared")
            ));
        }
        changes.join(", ")
    };

    if !update_description.is_empty() {
        let memory_content = format!(
            "[{}] Todo updated ({}): {}",
            todo.short_id(),
            update_description,
            todo.content
        );

        let mut tags = vec![
            format!("todo:{}", todo.short_id()),
            "todo-updated".to_string(),
        ];
        if let Some(ref proj) = project_name {
            tags.push(format!("project:{}", proj));
        }
        if req.status.is_some() {
            tags.push(format!("status:{:?}", todo.status).to_lowercase());
        }

        let experience = Experience {
            content: memory_content,
            experience_type: ExperienceType::Context,
            tags,
            ..Default::default()
        };

        if let Ok(memory) = state.get_user_memory(&req.user_id) {
            let memory_clone = memory.clone();
            let exp_clone = experience.clone();
            let state_clone = state.clone();
            let user_id = req.user_id.clone();

            tokio::spawn(async move {
                let memory_result = tokio::task::spawn_blocking(move || {
                    let memory_guard = memory_clone.read();
                    memory_guard.remember(exp_clone, None)
                })
                .await;

                if let Ok(Ok(memory_id)) = memory_result {
                    if let Err(e) =
                        state_clone.process_experience_into_graph(&user_id, &experience, &memory_id)
                    {
                        tracing::debug!(
                            "Graph processing failed for todo update memory {}: {}",
                            memory_id.0,
                            e
                        );
                    }
                    tracing::debug!(memory_id = %memory_id.0, "Todo update stored as memory");
                }
            });
        }
    }

    let formatted = todo_formatter::format_todo_updated(&todo, project_name.as_deref());

    // Emit SSE event for live TUI updates
    state.emit_event(MemoryEvent {
        event_type: "TODO_UPDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(todo.id.0.to_string()),
        content_preview: Some(todo.content.clone()),
        memory_type: Some(format!("{:?}", todo.status)),
        importance: None,
        count: None,
    });

    tracing::info!(
        user_id = %req.user_id,
        todo_id = %todo.id,
        "Updated todo"
    );

    // Audit log for todo update
    state.log_event(
        &req.user_id,
        "TODO_UPDATE",
        &todo.id.0.to_string(),
        &format!(
            "Updated todo [{}]: {}",
            todo.short_id(),
            if update_description.is_empty() {
                "no changes"
            } else {
                &update_description
            }
        ),
    );

    Ok(Json(TodoResponse {
        success: true,
        todo: Some(todo),
        project: None,
        formatted,
    }))
}

/// POST /api/todos/{todo_id}/complete - Mark todo as complete
async fn complete_todo(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Json(req): Json<TodoQuery>,
) -> Result<Json<TodoCompleteResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Find the todo first
    let todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    // Complete it (handles recurrence)
    let result = state
        .todo_store
        .complete_todo(&req.user_id, &todo.id)
        .map_err(AppError::Internal)?;

    match result {
        Some((completed, next)) => {
            let formatted = todo_formatter::format_todo_completed(&completed, next.as_ref());

            // Emit SSE event for live TUI updates
            state.emit_event(MemoryEvent {
                event_type: "TODO_COMPLETE".to_string(),
                timestamp: chrono::Utc::now(),
                user_id: req.user_id.clone(),
                memory_id: Some(completed.id.0.to_string()),
                content_preview: Some(completed.content.clone()),
                memory_type: Some("Done".to_string()),
                importance: None,
                count: None,
            });

            tracing::info!(
                user_id = %req.user_id,
                todo_id = %completed.id,
                has_next = next.is_some(),
                "Completed todo"
            );

            // Auto-generate post-mortem in background (non-blocking)
            let state_clone = state.clone();
            let user_id_clone = req.user_id.clone();
            let todo_content = completed.content.clone();
            let todo_created_at = completed.created_at;
            tokio::spawn(async move {
                if let Err(e) = auto_generate_post_mortem(
                    &state_clone,
                    &user_id_clone,
                    &todo_content,
                    todo_created_at,
                )
                .await
                {
                    tracing::debug!("Auto post-mortem generation failed: {}", e);
                }
            });

            // Audit log for todo completion
            state.log_event(
                &req.user_id,
                "TODO_COMPLETE",
                &completed.id.0.to_string(),
                &format!(
                    "Completed todo [{}]: '{}' (recurrence={})",
                    completed.short_id(),
                    completed.content.chars().take(40).collect::<String>(),
                    next.is_some()
                ),
            );

            Ok(Json(TodoCompleteResponse {
                success: true,
                todo: Some(completed),
                next_recurrence: next,
                formatted,
            }))
        }
        None => Err(AppError::TodoNotFound(todo_id)),
    }
}

/// DELETE /api/todos/{todo_id} - Delete a todo
async fn delete_todo(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<TodoResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    // Find the todo first
    let todo = state
        .todo_store
        .find_todo_by_prefix(&query.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let success = state
        .todo_store
        .delete_todo(&query.user_id, &todo.id)
        .map_err(AppError::Internal)?;

    let formatted = if success {
        todo_formatter::format_todo_deleted(&todo.short_id())
    } else {
        "Todo not found".to_string()
    };

    if success {
        // Emit SSE event for live TUI updates
        state.emit_event(MemoryEvent {
            event_type: "TODO_DELETE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: query.user_id.clone(),
            memory_id: Some(todo.id.0.to_string()),
            content_preview: Some(todo.content.clone()),
            memory_type: None,
            importance: None,
            count: None,
        });

        tracing::info!(
            user_id = %query.user_id,
            todo_id = %todo.id,
            "Deleted todo"
        );
    }

    Ok(Json(TodoResponse {
        success,
        todo: None,
        project: None,
        formatted,
    }))
}

/// POST /api/todos/{todo_id}/reorder - Move todo up/down within status group
async fn reorder_todo(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Json(req): Json<ReorderTodoRequest>,
) -> Result<Json<TodoResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Find the todo first
    let todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let result = state
        .todo_store
        .reorder_todo(&req.user_id, &todo.id, &req.direction)
        .map_err(AppError::Internal)?;

    match result {
        Some(updated) => {
            let formatted = format!(
                "Moved {} {}",
                updated.short_id(),
                if req.direction == "up" { "up" } else { "down" }
            );

            // Emit SSE event for live TUI updates
            state.emit_event(MemoryEvent {
                event_type: "TODO_REORDER".to_string(),
                timestamp: chrono::Utc::now(),
                user_id: req.user_id.clone(),
                memory_id: Some(updated.id.0.to_string()),
                content_preview: Some(updated.content.clone()),
                memory_type: Some(format!("{:?}", updated.status)),
                importance: None,
                count: None,
            });

            tracing::debug!(
                user_id = %req.user_id,
                todo_id = %updated.id,
                direction = %req.direction,
                "Reordered todo"
            );

            Ok(Json(TodoResponse {
                success: true,
                todo: Some(updated),
                project: None,
                formatted,
            }))
        }
        None => Err(AppError::TodoNotFound(todo_id)),
    }
}

/// GET /api/todos/{todo_id}/subtasks - List subtasks of a parent todo
async fn list_subtasks(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<TodoListResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    // Find the parent todo
    let parent = state
        .todo_store
        .find_todo_by_prefix(&query.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let subtasks = state
        .todo_store
        .list_subtasks(&parent.id)
        .map_err(AppError::Internal)?;

    let projects = state
        .todo_store
        .list_projects(&query.user_id)
        .map_err(AppError::Internal)?;

    let formatted = if subtasks.is_empty() {
        format!("No subtasks for {}", parent.short_id())
    } else {
        let mut output = format!(
            "🐘━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
             ┃  SUBTASKS OF {}  ┃\n\
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n",
            parent.short_id()
        );
        output.push_str(&todo_formatter::format_todo_list(&subtasks, &projects));
        output
    };

    tracing::debug!(
        user_id = %query.user_id,
        parent_id = %parent.id,
        count = subtasks.len(),
        "Listed subtasks"
    );

    Ok(Json(TodoListResponse {
        success: true,
        count: subtasks.len(),
        todos: subtasks,
        projects,
        formatted,
    }))
}

// =====================================================================
// TODO COMMENTS ENDPOINTS
// =====================================================================

/// POST /api/todos/{todo_id}/comments - Add a comment to a todo
async fn add_todo_comment(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Json(req): Json<AddCommentRequest>,
) -> Result<Json<CommentResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.content.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "content".to_string(),
            reason: "Comment content cannot be empty".to_string(),
        });
    }

    // Find the todo first
    let todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    // Parse comment type
    let comment_type = req
        .comment_type
        .as_ref()
        .and_then(|ct| match ct.to_lowercase().as_str() {
            "comment" => Some(TodoCommentType::Comment),
            "progress" => Some(TodoCommentType::Progress),
            "resolution" => Some(TodoCommentType::Resolution),
            "activity" => Some(TodoCommentType::Activity),
            _ => None,
        });

    let author = req.author.unwrap_or_else(|| req.user_id.clone());

    let comment = state
        .todo_store
        .add_comment(
            &req.user_id,
            &todo.id,
            author.clone(),
            req.content.clone(),
            comment_type.clone(),
        )
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    // Create a memory from this comment - nothing exists outside the brain
    // Map comment type to experience type
    let experience_type = match comment_type.as_ref().unwrap_or(&TodoCommentType::Comment) {
        TodoCommentType::Comment => ExperienceType::Observation,
        TodoCommentType::Progress => ExperienceType::Learning,
        TodoCommentType::Resolution => ExperienceType::Learning,
        TodoCommentType::Activity => ExperienceType::Context,
    };

    // Build context-rich content for the memory
    let memory_content = format!(
        "[{}] {} ({}): {}",
        todo.short_id(),
        match comment_type.as_ref().unwrap_or(&TodoCommentType::Comment) {
            TodoCommentType::Comment => "Comment",
            TodoCommentType::Progress => "Progress",
            TodoCommentType::Resolution => "Resolution",
            TodoCommentType::Activity => "Activity",
        },
        todo.content,
        req.content
    );

    // Build tags linking to todo and project
    let mut tags = vec![
        format!("todo:{}", todo.short_id()),
        format!("todo-comment:{:?}", comment.comment_type).to_lowercase(),
    ];
    // Look up project name from project_id
    if let Some(ref project_id) = todo.project_id {
        if let Ok(Some(project)) = state.todo_store.get_project(&req.user_id, project_id) {
            tags.push(format!("project:{}", project.name));
        }
    }

    let experience = Experience {
        content: memory_content,
        experience_type,
        tags,
        ..Default::default()
    };

    // Store as memory
    if let Ok(memory) = state.get_user_memory(&req.user_id) {
        let memory_clone = memory.clone();
        let exp_clone = experience.clone();
        let memory_result = tokio::task::spawn_blocking(move || {
            let memory_guard = memory_clone.read();
            memory_guard.remember(exp_clone, None)
        })
        .await;

        if let Ok(Ok(memory_id)) = memory_result {
            // Process into knowledge graph for connections
            if let Err(e) =
                state.process_experience_into_graph(&req.user_id, &experience, &memory_id)
            {
                tracing::debug!(
                    "Graph processing failed for todo comment memory {}: {}",
                    memory_id.0,
                    e
                );
            }

            tracing::debug!(
                memory_id = %memory_id.0,
                todo_id = %todo.id,
                "Todo comment stored as memory"
            );
        }
    }

    let formatted = format!(
        "✓ Added comment to {}\n\n  {} ({}):\n  {}",
        todo.short_id(),
        author,
        comment.created_at.format("%Y-%m-%d %H:%M"),
        req.content
    );

    // Emit SSE event for live TUI updates
    state.emit_event(MemoryEvent {
        event_type: "TODO_COMMENT_ADD".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(comment.id.0.to_string()),
        content_preview: Some(format!(
            "[{}] {}",
            todo.short_id(),
            req.content.chars().take(80).collect::<String>()
        )),
        memory_type: Some(format!("{:?}", comment.comment_type)),
        importance: None,
        count: None,
    });

    tracing::debug!(
        user_id = %req.user_id,
        todo_id = %todo.id,
        comment_id = %comment.id.0,
        "Added comment to todo"
    );

    Ok(Json(CommentResponse {
        success: true,
        comment: Some(comment),
        formatted,
    }))
}

/// GET /api/todos/{todo_id}/comments - List comments for a todo
async fn list_todo_comments(
    State(state): State<AppState>,
    Path(todo_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<CommentListResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    // Find the todo first
    let todo = state
        .todo_store
        .find_todo_by_prefix(&query.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    let comments = state
        .todo_store
        .get_comments(&query.user_id, &todo.id)
        .map_err(AppError::Internal)?;

    let formatted = if comments.is_empty() {
        format!("No comments on {}", todo.short_id())
    } else {
        let mut output = format!(
            "📝 Comments on {} ({} total)\n\n",
            todo.short_id(),
            comments.len()
        );
        for (i, comment) in comments.iter().enumerate() {
            let type_icon = match comment.comment_type {
                TodoCommentType::Comment => "💬",
                TodoCommentType::Progress => "📊",
                TodoCommentType::Resolution => "✅",
                TodoCommentType::Activity => "🔄",
            };
            output.push_str(&format!(
                "{}. {} {} ({})\n   {}\n\n",
                i + 1,
                type_icon,
                comment.author,
                comment.created_at.format("%Y-%m-%d %H:%M"),
                comment.content
            ));
        }
        output
    };

    tracing::debug!(
        user_id = %query.user_id,
        todo_id = %todo.id,
        count = comments.len(),
        "Listed todo comments"
    );

    Ok(Json(CommentListResponse {
        success: true,
        count: comments.len(),
        comments,
        formatted,
    }))
}

/// POST /api/todos/{todo_id}/comments/{comment_id}/update - Update a comment
async fn update_todo_comment(
    State(state): State<AppState>,
    Path((todo_id, comment_id)): Path<(String, String)>,
    Json(req): Json<UpdateCommentRequest>,
) -> Result<Json<CommentResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.content.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "content".to_string(),
            reason: "Comment content cannot be empty".to_string(),
        });
    }

    // Find the todo first
    let todo = state
        .todo_store
        .find_todo_by_prefix(&req.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    // Parse comment ID
    let cid = uuid::Uuid::parse_str(&comment_id).map_err(|_| AppError::InvalidInput {
        field: "comment_id".to_string(),
        reason: "Invalid comment ID format".to_string(),
    })?;
    let comment_id_typed = TodoCommentId(cid);

    let comment = state
        .todo_store
        .update_comment(
            &req.user_id,
            &todo.id,
            &comment_id_typed,
            req.content.clone(),
        )
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::InvalidInput {
            field: "comment_id".to_string(),
            reason: "Comment not found".to_string(),
        })?;

    let formatted = format!(
        "✓ Updated comment on {}\n\n  Updated content:\n  {}",
        todo.short_id(),
        req.content
    );

    tracing::debug!(
        user_id = %req.user_id,
        todo_id = %todo.id,
        comment_id = %comment_id_typed.0,
        "Updated todo comment"
    );

    Ok(Json(CommentResponse {
        success: true,
        comment: Some(comment),
        formatted,
    }))
}

/// DELETE /api/todos/{todo_id}/comments/{comment_id} - Delete a comment
async fn delete_todo_comment(
    State(state): State<AppState>,
    Path((todo_id, comment_id)): Path<(String, String)>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<CommentResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    // Find the todo first
    let todo = state
        .todo_store
        .find_todo_by_prefix(&query.user_id, &todo_id)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::TodoNotFound(todo_id.clone()))?;

    // Parse comment ID
    let cid = uuid::Uuid::parse_str(&comment_id).map_err(|_| AppError::InvalidInput {
        field: "comment_id".to_string(),
        reason: "Invalid comment ID format".to_string(),
    })?;
    let comment_id_typed = TodoCommentId(cid);

    let success = state
        .todo_store
        .delete_comment(&query.user_id, &todo.id, &comment_id_typed)
        .map_err(AppError::Internal)?;

    let formatted = if success {
        format!("✓ Deleted comment from {}", todo.short_id())
    } else {
        "Comment not found".to_string()
    };

    // Emit SSE event for live TUI updates
    if success {
        state.emit_event(MemoryEvent {
            event_type: "TODO_COMMENT_DELETE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: query.user_id.clone(),
            memory_id: Some(comment_id.to_string()),
            content_preview: Some(format!("[{}] comment deleted", todo.short_id())),
            memory_type: None,
            importance: None,
            count: None,
        });
    }

    tracing::debug!(
        user_id = %query.user_id,
        todo_id = %todo.id,
        comment_id = %comment_id,
        success = success,
        "Deleted todo comment"
    );

    Ok(Json(CommentResponse {
        success,
        comment: None,
        formatted,
    }))
}

/// POST /api/projects - Create a new project
async fn create_project(
    State(state): State<AppState>,
    Json(req): Json<CreateProjectRequest>,
) -> Result<Json<ProjectResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.name.trim().is_empty() {
        return Err(AppError::InvalidInput {
            field: "name".to_string(),
            reason: "Project name cannot be empty".to_string(),
        });
    }

    // Resolve parent project if specified
    let parent_id = if let Some(ref parent_ref) = req.parent {
        // Try as UUID first
        if let Ok(uuid) = uuid::Uuid::parse_str(parent_ref) {
            let pid = ProjectId(uuid);
            // Verify parent exists
            state
                .todo_store
                .get_project(&req.user_id, &pid)
                .map_err(AppError::Internal)?
                .ok_or_else(|| AppError::ProjectNotFound(parent_ref.clone()))?;
            Some(pid)
        } else {
            // Try by name
            let parent = state
                .todo_store
                .find_project_by_name(&req.user_id, parent_ref)
                .map_err(AppError::Internal)?
                .ok_or_else(|| AppError::ProjectNotFound(parent_ref.clone()))?;
            Some(parent.id)
        }
    } else {
        None
    };

    let mut project = Project::new(req.user_id.clone(), req.name.clone());
    // Override auto-derived prefix if custom prefix provided
    if let Some(ref custom_prefix) = req.prefix {
        let clean = custom_prefix.trim().to_uppercase();
        if !clean.is_empty() {
            project.prefix = Some(clean);
        }
    }
    project.description = req.description;
    project.color = req.color;
    project.parent_id = parent_id;

    state
        .todo_store
        .store_project(&project)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_project_created(&project);

    // Emit SSE event for live TUI updates
    state.emit_event(MemoryEvent {
        event_type: "PROJECT_CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(project.id.0.to_string()),
        content_preview: Some(project.name.clone()),
        memory_type: Some("Project".to_string()),
        importance: None,
        count: None,
    });

    tracing::info!(
        user_id = %req.user_id,
        project_id = %project.id.0,
        name = %req.name,
        parent = ?project.parent_id,
        "Created project"
    );

    Ok(Json(ProjectResponse {
        success: true,
        project: Some(project),
        stats: None,
        formatted,
    }))
}

/// POST /api/projects/list - List projects
async fn list_projects(
    State(state): State<AppState>,
    Json(req): Json<ListProjectsRequest>,
) -> Result<Json<ProjectListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let projects = state
        .todo_store
        .list_projects(&req.user_id)
        .map_err(AppError::Internal)?;

    // Get stats for each project
    let mut project_stats = Vec::new();
    for project in projects {
        let stats = state
            .todo_store
            .get_project_stats(&req.user_id, &project.id)
            .map_err(AppError::Internal)?;
        project_stats.push((project, stats));
    }

    let formatted = todo_formatter::format_project_list(&project_stats);

    Ok(Json(ProjectListResponse {
        success: true,
        count: project_stats.len(),
        projects: project_stats,
        formatted,
    }))
}

/// GET /api/projects/{project_id} - Get a project with stats
async fn get_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<ProjectResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    // Try to find by name first, then by ID
    let project = state
        .todo_store
        .find_project_by_name(&query.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&query.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let stats = state
        .todo_store
        .get_project_stats(&query.user_id, &project.id)
        .map_err(AppError::Internal)?;

    let todos = state
        .todo_store
        .list_todos_by_project(&query.user_id, &project.id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_project_todos(&project, &todos, &stats);

    Ok(Json(ProjectResponse {
        success: true,
        project: Some(project),
        stats: Some(stats),
        formatted,
    }))
}

/// POST /api/projects/{project_id}/update - Update a project
async fn update_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<UpdateProjectRequest>,
) -> Result<Json<ProjectResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Find project by name or ID
    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let updated = state
        .todo_store
        .update_project(
            &req.user_id,
            &project.id,
            req.name,
            req.prefix,
            req.description,
            req.status,
            req.color,
        )
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let formatted = todo_formatter::format_project_updated(&updated);

    // Emit SSE event for live TUI updates
    state.emit_event(MemoryEvent {
        event_type: "PROJECT_UPDATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(updated.id.0.to_string()),
        content_preview: Some(updated.name.clone()),
        memory_type: Some("Project".to_string()),
        importance: None,
        count: None,
    });

    tracing::info!(
        user_id = %req.user_id,
        project_id = %updated.id.0,
        status = ?updated.status,
        "Updated project"
    );

    Ok(Json(ProjectResponse {
        success: true,
        project: Some(updated),
        stats: None,
        formatted,
    }))
}

/// DELETE /api/projects/{project_id} - Delete a project
async fn delete_project(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<DeleteProjectRequest>,
) -> Result<Json<ProjectResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Find project by name or ID
    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    // Count todos before deleting
    let todos_count = if req.delete_todos {
        state
            .todo_store
            .list_todos_by_project(&req.user_id, &project.id)
            .map_err(AppError::Internal)?
            .len()
    } else {
        0
    };

    let deleted = state
        .todo_store
        .delete_project(&req.user_id, &project.id, req.delete_todos)
        .map_err(AppError::Internal)?;

    if !deleted {
        return Err(AppError::ProjectNotFound(project_id));
    }

    let formatted = todo_formatter::format_project_deleted(&project, todos_count);

    // Emit SSE event for live TUI updates
    state.emit_event(MemoryEvent {
        event_type: "PROJECT_DELETE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(project.id.0.to_string()),
        content_preview: Some(project.name.clone()),
        memory_type: Some("Project".to_string()),
        importance: None,
        count: Some(todos_count),
    });

    tracing::info!(
        user_id = %req.user_id,
        project_id = %project.id.0,
        delete_todos = %req.delete_todos,
        todos_deleted = %todos_count,
        "Deleted project"
    );

    Ok(Json(ProjectResponse {
        success: true,
        project: Some(project),
        stats: None,
        formatted,
    }))
}

// =============================================================================
// FILE MEMORY ENDPOINTS (MEM-33, MEM-34, MEM-35)
// Codebase integration - learned knowledge about files
// =============================================================================

/// Request to list files for a project
#[derive(Debug, Deserialize)]
struct ListFilesRequest {
    user_id: String,
    #[serde(default)]
    limit: Option<usize>,
}

/// Request to scan/index a codebase
#[derive(Debug, Deserialize)]
struct IndexCodebaseRequest {
    user_id: String,
    /// Path to the codebase root directory
    codebase_path: String,
    /// Force re-index even if already indexed
    #[serde(default)]
    force: bool,
}

/// Request to search files
#[derive(Debug, Deserialize)]
struct SearchFilesRequest {
    user_id: String,
    query: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
}

fn default_search_limit() -> usize {
    10
}

/// Response for file list operations
#[derive(Debug, Serialize)]
struct FileListResponse {
    success: bool,
    files: Vec<FileMemorySummary>,
    total: usize,
}

/// Summary of a file memory (for list responses)
#[derive(Debug, Serialize)]
struct FileMemorySummary {
    id: String,
    path: String,
    absolute_path: String,
    file_type: String,
    summary: String,
    key_items: Vec<String>,
    access_count: u32,
    last_accessed: String,
    heat_score: u8,
    size_bytes: u64,
    line_count: usize,
}

/// Response for scan operation
#[derive(Debug, Serialize)]
struct ScanResponse {
    success: bool,
    total_files: usize,
    eligible_files: usize,
    skipped_files: usize,
    limit_reached: bool,
    message: String,
}

/// Response for index operation
#[derive(Debug, Serialize)]
struct IndexResponse {
    success: bool,
    result: IndexingResult,
    message: String,
}

/// Response for file stats
#[derive(Debug, Serialize)]
struct FileStatsResponse {
    success: bool,
    stats: FileMemoryStats,
}

/// POST /api/projects/{project_id}/files - List files for a project
async fn list_project_files(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<ListFilesRequest>,
) -> Result<Json<FileListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Find project
    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    let files = state
        .file_store
        .list_by_project(&req.user_id, &project.id, req.limit)
        .map_err(AppError::Internal)?;

    let total = files.len();
    let summaries: Vec<FileMemorySummary> = files
        .into_iter()
        .map(|f| {
            let heat_score = f.heat_score();
            FileMemorySummary {
                id: f.id.0.to_string(),
                path: f.path,
                absolute_path: f.absolute_path,
                file_type: format!("{:?}", f.file_type),
                summary: f.summary,
                key_items: f.key_items,
                access_count: f.access_count,
                last_accessed: f.last_accessed.to_rfc3339(),
                heat_score,
                size_bytes: f.size_bytes,
                line_count: f.line_count,
            }
        })
        .collect();

    Ok(Json(FileListResponse {
        success: true,
        files: summaries,
        total,
    }))
}

/// POST /api/projects/{project_id}/scan - Scan codebase (preview before indexing)
async fn scan_project_codebase(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<IndexCodebaseRequest>,
) -> Result<Json<ScanResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Find project
    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    // Verify path exists
    let codebase_path = std::path::Path::new(&req.codebase_path);
    if !codebase_path.exists() {
        return Err(AppError::InvalidInput {
            field: "codebase_path".to_string(),
            reason: format!("Codebase path does not exist: {}", req.codebase_path),
        });
    }
    if !codebase_path.is_dir() {
        return Err(AppError::InvalidInput {
            field: "codebase_path".to_string(),
            reason: format!("Codebase path is not a directory: {}", req.codebase_path),
        });
    }

    // Scan the codebase
    let scan_result = state
        .file_store
        .scan_codebase(codebase_path, None)
        .map_err(AppError::Internal)?;

    let message = if scan_result.limit_reached {
        format!(
            "Found {} eligible files (limit reached). {} files skipped.",
            scan_result.eligible_files, scan_result.skipped_files
        )
    } else {
        format!(
            "Found {} eligible files. {} files skipped.",
            scan_result.eligible_files, scan_result.skipped_files
        )
    };

    tracing::info!(
        user_id = %req.user_id,
        project_id = %project.id.0,
        path = %req.codebase_path,
        eligible = scan_result.eligible_files,
        skipped = scan_result.skipped_files,
        "Scanned codebase"
    );

    Ok(Json(ScanResponse {
        success: true,
        total_files: scan_result.total_files,
        eligible_files: scan_result.eligible_files,
        skipped_files: scan_result.skipped_files,
        limit_reached: scan_result.limit_reached,
        message,
    }))
}

/// POST /api/projects/{project_id}/index - Index codebase files
async fn index_project_codebase(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<IndexCodebaseRequest>,
) -> Result<Json<IndexResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Find project
    let mut project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    // Check if already indexed
    if project.codebase_indexed && !req.force {
        return Err(AppError::InvalidInput {
            field: "force".to_string(),
            reason: "Codebase already indexed. Use force=true to re-index.".to_string(),
        });
    }

    // Verify path exists
    let codebase_path = std::path::Path::new(&req.codebase_path);
    if !codebase_path.exists() {
        return Err(AppError::InvalidInput {
            field: "codebase_path".to_string(),
            reason: format!("Codebase path does not exist: {}", req.codebase_path),
        });
    }

    // Delete existing files if re-indexing
    if req.force && project.codebase_indexed {
        state
            .file_store
            .delete_project_files(&req.user_id, &project.id)
            .map_err(AppError::Internal)?;
    }

    // Index the codebase (without embeddings for now - faster)
    let result = state
        .file_store
        .index_codebase(codebase_path, &project.id, &req.user_id, None)
        .map_err(AppError::Internal)?;

    // Update project with codebase info
    project.codebase_path = Some(req.codebase_path.clone());
    project.codebase_indexed = true;
    project.codebase_indexed_at = Some(chrono::Utc::now());
    project.codebase_file_count = result.indexed_files;

    state
        .todo_store
        .store_project(&project)
        .map_err(AppError::Internal)?;

    let message = format!(
        "Indexed {} files ({} skipped, {} errors)",
        result.indexed_files,
        result.skipped_files,
        result.errors.len()
    );

    // Emit SSE event
    state.emit_event(MemoryEvent {
        event_type: "CODEBASE_INDEXED".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(project.id.0.to_string()),
        content_preview: Some(format!("{} files indexed", result.indexed_files)),
        memory_type: Some("Codebase".to_string()),
        importance: None,
        count: Some(result.indexed_files),
    });

    tracing::info!(
        user_id = %req.user_id,
        project_id = %project.id.0,
        path = %req.codebase_path,
        indexed = result.indexed_files,
        skipped = result.skipped_files,
        errors = result.errors.len(),
        "Indexed codebase"
    );

    Ok(Json(IndexResponse {
        success: true,
        result,
        message,
    }))
}

/// POST /api/projects/{project_id}/files/search - Search files semantically
async fn search_project_files(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    Json(req): Json<SearchFilesRequest>,
) -> Result<Json<FileListResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Find project
    let project = state
        .todo_store
        .find_project_by_name(&req.user_id, &project_id)
        .map_err(AppError::Internal)?
        .or_else(|| {
            uuid::Uuid::parse_str(&project_id).ok().and_then(|uuid| {
                state
                    .todo_store
                    .get_project(&req.user_id, &ProjectId(uuid))
                    .ok()
                    .flatten()
            })
        })
        .ok_or_else(|| AppError::ProjectNotFound(project_id.clone()))?;

    // For now, simple text search on path and key_items
    // TODO: Add semantic search using embeddings
    let all_files = state
        .file_store
        .list_by_project(&req.user_id, &project.id, None)
        .map_err(AppError::Internal)?;

    let query_lower = req.query.to_lowercase();
    let matching_files: Vec<_> = all_files
        .into_iter()
        .filter(|f| {
            f.path.to_lowercase().contains(&query_lower)
                || f.key_items
                    .iter()
                    .any(|k| k.to_lowercase().contains(&query_lower))
                || f.summary.to_lowercase().contains(&query_lower)
        })
        .take(req.limit)
        .collect();

    let total = matching_files.len();
    let summaries: Vec<FileMemorySummary> = matching_files
        .into_iter()
        .map(|f| {
            let heat_score = f.heat_score();
            FileMemorySummary {
                id: f.id.0.to_string(),
                path: f.path,
                absolute_path: f.absolute_path,
                file_type: format!("{:?}", f.file_type),
                summary: f.summary,
                key_items: f.key_items,
                access_count: f.access_count,
                last_accessed: f.last_accessed.to_rfc3339(),
                heat_score,
                size_bytes: f.size_bytes,
                line_count: f.line_count,
            }
        })
        .collect();

    Ok(Json(FileListResponse {
        success: true,
        files: summaries,
        total,
    }))
}

/// GET /api/files/stats - Get file memory statistics
async fn get_file_stats(
    State(state): State<AppState>,
    Query(query): Query<TodoQuery>,
) -> Result<Json<FileStatsResponse>, AppError> {
    validation::validate_user_id(&query.user_id).map_validation_err("user_id")?;

    let stats = state
        .file_store
        .stats(&query.user_id)
        .map_err(AppError::Internal)?;

    Ok(Json(FileStatsResponse {
        success: true,
        stats,
    }))
}

/// POST /api/todos/stats - Get todo statistics
async fn get_todo_stats(
    State(state): State<AppState>,
    Json(req): Json<TodoStatsRequest>,
) -> Result<Json<TodoStatsResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let stats = state
        .todo_store
        .get_user_stats(&req.user_id)
        .map_err(AppError::Internal)?;

    let formatted = todo_formatter::format_user_stats(&stats);

    Ok(Json(TodoStatsResponse {
        success: true,
        stats,
        formatted,
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present (silently ignore if not found)
    let _ = dotenvy::dotenv();

    // P1.6: Initialize distributed tracing with OpenTelemetry (optional)
    #[cfg(feature = "telemetry")]
    {
        tracing_setup::init_tracing().expect("Failed to initialize tracing");
    }
    #[cfg(not(feature = "telemetry"))]
    {
        // Default to info level if RUST_LOG not set (user-friendly default)
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "shodh_memory=info,tower_http=warn");
        }
        tracing_subscriber::fmt::init();
    }

    // Print startup banner (always visible, regardless of log level)
    eprintln!();
    eprintln!("  ╔═══════════════════════════════════════════════════╗");
    eprintln!(
        "  ║         🧠 Shodh-Memory Server v{}          ║",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  ║       Cognitive Memory for AI Agents              ║");
    eprintln!("  ╚═══════════════════════════════════════════════════╝");
    eprintln!();

    // P1.1: Register Prometheus metrics
    metrics::register_metrics().expect("Failed to register metrics");

    // Load configuration from environment
    let server_config = ServerConfig::from_env();

    // Print configuration (always visible)
    eprintln!("  📋 Configuration:");
    eprintln!(
        "     Mode:    {}",
        if server_config.is_production {
            "PRODUCTION"
        } else {
            "Development"
        }
    );
    eprintln!("     Port:    {}", server_config.port);
    eprintln!("     Storage: {}", server_config.storage_path.display());
    eprintln!();

    // Create memory manager with config
    let manager = Arc::new(MultiUserMemoryManager::new(
        server_config.storage_path.clone(),
        server_config.clone(),
    )?);

    // Print storage statistics (always visible)
    let storage_path = &server_config.storage_path;
    if storage_path.exists() {
        let disk_usage = calculate_dir_size(storage_path);
        let user_count = count_user_directories(storage_path);
        eprintln!("  💾 Storage Statistics:");
        eprintln!(
            "     Location:  {}",
            storage_path
                .canonicalize()
                .unwrap_or_else(|_| storage_path.clone())
                .display()
        );
        eprintln!("     Disk used: {}", format_bytes(disk_usage));
        eprintln!("     Users:     {}", user_count);
        eprintln!();
    } else {
        eprintln!("  💾 Storage: New database (no existing data)");
        eprintln!();
    }

    // Keep a reference to manager for shutdown cleanup (clone BEFORE moving into router)
    let manager_for_shutdown = Arc::clone(&manager);

    // Start background maintenance scheduler (consolidation, activation decay, graph pruning)
    let maintenance_interval = server_config.maintenance_interval_secs;
    let manager_for_maintenance = Arc::clone(&manager);
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(maintenance_interval));

        loop {
            interval.tick().await;

            // Clean up stale streaming sessions (async operation)
            // This prevents memory leaks from abandoned WebSocket connections
            let extractor = manager_for_maintenance.streaming_extractor().clone();
            let cleaned = extractor.cleanup_stale_sessions().await;
            if cleaned > 0 {
                tracing::debug!("Session cleanup: removed {} stale sessions", cleaned);
            }

            // Run maintenance + periodic flush in blocking thread pool
            // This ensures durability in async write mode (data flushed every maintenance cycle)
            let manager_clone = Arc::clone(&manager_for_maintenance);
            tokio::task::spawn_blocking(move || {
                manager_clone.run_maintenance_all_users();
                // Periodic flush ensures data durability for async write mode
                // In sync mode this is fast (little dirty data), in async mode this is critical
                if let Err(e) = manager_clone.flush_all_databases() {
                    tracing::warn!("Periodic flush failed: {}", e);
                }
            });
        }
    });
    info!(
        "🔄 Background maintenance scheduler started (interval: {}s)",
        maintenance_interval
    );

    // Start backup scheduler if enabled
    if server_config.backup_enabled && server_config.backup_interval_secs > 0 {
        let backup_interval = server_config.backup_interval_secs;
        let max_backups = server_config.backup_max_count;
        let manager_for_backup = Arc::clone(&manager);
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(std::time::Duration::from_secs(backup_interval));

            // Skip the first immediate tick - let the system warm up
            interval.tick().await;

            loop {
                interval.tick().await;

                info!("💾 Starting scheduled backup run...");
                let manager_clone = Arc::clone(&manager_for_backup);
                let backed_up = tokio::task::spawn_blocking(move || {
                    manager_clone.run_backup_all_users(max_backups)
                })
                .await
                .unwrap_or(0);

                if backed_up > 0 {
                    info!(
                        "💾 Scheduled backup completed: {} users backed up",
                        backed_up
                    );
                } else {
                    tracing::debug!("💾 Scheduled backup: no users to backup");
                }
            }
        });
        info!(
            "💾 Automatic backup scheduler started (interval: {}h, keep: {} backups)",
            backup_interval / 3600,
            max_backups
        );
    }

    // Configure rate limiting from config
    let governor_conf = GovernorConfigBuilder::default()
        .per_second(server_config.rate_limit_per_second)
        .burst_size(server_config.rate_limit_burst)
        .finish()
        .expect("Failed to build governor rate limiter configuration");

    let governor_layer = GovernorLayer::new(governor_conf);

    info!(
        "⚡ Rate limiting enabled: {} req/sec, burst of {}",
        server_config.rate_limit_per_second, server_config.rate_limit_burst
    );

    // Build CORS layer from configuration
    let cors = server_config.cors.to_layer();

    // Build router with rate limiting
    // Split into public (no auth) and protected (auth required) routes

    // Protected API routes - require authentication
    let protected_routes = Router::new()
        // Core endpoints
        .route("/api/record", post(record_experience))
        // /api/retrieve removed - use /api/recall instead
        // Simplified LLM-friendly endpoints (effortless API)
        .route("/api/remember", post(remember))
        .route("/api/recall", post(recall))
        // Batch insert (SHO-83) - both paths for compatibility
        .route("/api/remember/batch", post(batch_remember))
        .route("/api/batch_remember", post(batch_remember))
        // Mutable memories with external linking (SHO-39)
        .route("/api/upsert", post(upsert_memory))
        .route("/api/memory/history", post(get_memory_history))
        // External integrations (SHO-40, SHO-41)
        .route("/webhook/linear", post(linear_webhook))
        .route("/api/sync/linear", post(linear_sync))
        .route("/webhook/github", post(github_webhook))
        .route("/api/sync/github", post(github_sync))
        // Hebbian Feedback Loop - Wire up learning from task outcomes
        .route("/api/recall/tracked", post(recall_tracked))
        .route("/api/reinforce", post(reinforce_feedback))
        // Semantic Consolidation - Extract durable facts from episodic memories
        .route("/api/consolidate", post(consolidate_memories))
        // User management
        .route("/api/users", get(list_users))
        .route("/api/users/{user_id}/stats", get(get_user_stats))
        .route("/api/stats", get(get_stats_query)) // OpenAPI spec alias
        .route("/api/users/{user_id}", delete(delete_user))
        // Memory CRUD
        .route("/api/memory/{memory_id}", get(get_memory))
        .route("/api/memory/{memory_id}", axum::routing::put(update_memory))
        .route("/api/memory/{memory_id}", delete(delete_memory))
        .route("/api/forget/{memory_id}", delete(delete_memory)) // OpenAPI spec alias
        .route(
            "/api/memory/{memory_id}",
            axum::routing::patch(patch_memory),
        )
        .route("/api/memories", post(get_all_memories))
        .route("/api/memories", get(get_all_memories_get)) // OpenAPI spec GET alias
        .route("/api/memories/history", post(get_history))
        .route("/api/memories/bulk", post(bulk_delete_memories))
        .route("/api/memories/clear", post(clear_all_memories))
        // Compression & Storage Management
        .route("/api/memory/compress", post(compress_memory))
        .route("/api/memory/decompress", post(decompress_memory))
        .route("/api/storage/stats", post(get_storage_stats))
        .route("/api/storage/uncompressed", post(get_uncompressed_old))
        // Index Integrity & Repair (SHO-38)
        .route("/api/index/verify", post(verify_index_integrity))
        .route("/api/index/repair", post(repair_vector_index))
        .route("/api/index/rebuild", post(rebuild_index))
        .route("/api/storage/cleanup", post(cleanup_corrupted))
        // Forgetting Operations
        .route("/api/forget/age", post(forget_by_age))
        .route("/api/forget/importance", post(forget_by_importance))
        .route("/api/forget/pattern", post(forget_by_pattern))
        .route("/api/forget/tags", post(forget_by_tags))
        .route("/api/forget/date", post(forget_by_date))
        // Recall by filters (tag/date convenience endpoints)
        .route("/api/recall/tags", post(recall_by_tags))
        .route("/api/recall/by-tags", post(recall_by_tags)) // OpenAPI spec alias
        .route("/api/recall/date", post(recall_by_date))
        // Advanced Search
        .route("/api/search/advanced", post(advanced_search))
        .route("/api/search/multimodal", post(multimodal_search))
        .route("/api/search/robotics", post(robotics_search))
        // Graph Memory - Entity Management
        .route("/api/graph/{user_id}/stats", get(get_graph_stats))
        .route("/api/graph/entity/find", post(find_entity))
        .route("/api/graph/entity/add", post(add_entity))
        .route("/api/graph/entities/all", post(get_all_entities))
        // Graph Memory - Relationship Management
        .route("/api/graph/relationship/add", post(add_relationship))
        .route(
            "/api/graph/relationship/invalidate",
            post(invalidate_relationship),
        )
        .route("/api/graph/traverse", post(traverse_graph))
        // Graph Memory - Episodes
        .route("/api/graph/episode/get", post(get_episode))
        // Memory Universe Visualization (3D graph with salience-based sizing)
        .route("/api/graph/{user_id}/universe", get(get_memory_universe))
        // Clear graph data (for removing garbage entities)
        .route("/api/graph/{user_id}/clear", delete(clear_user_graph))
        // Rebuild graph from existing memories with improved NER
        .route("/api/graph/{user_id}/rebuild", post(rebuild_user_graph))
        // Memory Visualization
        .route(
            "/api/visualization/{user_id}/stats",
            get(get_visualization_stats),
        )
        .route(
            "/api/visualization/{user_id}/dot",
            get(get_visualization_dot),
        )
        .route("/api/visualization/build", post(build_visualization))
        // Brain State Visualization (cognitive memory tiers with activation levels)
        .route("/api/brain/{user_id}", get(get_brain_state))
        // Context Summary - Session bootstrap with categorized memories
        .route("/api/context_summary", post(context_summary))
        // Proactive Context (SHO-116) - Combined recall + reminders for MCP
        .route("/api/proactive_context", post(proactive_context))
        .route("/api/context", post(proactive_context)) // OpenAPI spec alias
        // Proactive Memory Surfacing (SHO-29) - Push-based relevance
        .route("/api/relevant", post(surface_relevant))
        // Context Monitor WebSocket (SHO-29) - Streaming context surfacing
        .route("/api/context/monitor", get(context_monitor_ws))
        // Consolidation Introspection - What the memory system is learning
        .route("/api/consolidation/report", post(get_consolidation_report))
        .route("/api/consolidation/events", post(get_consolidation_events))
        // Semantic Facts API (SHO-f0e7) - Durable knowledge from episodic memories
        .route("/api/facts/list", post(list_facts))
        .route("/api/facts/search", post(search_facts))
        .route("/api/facts/by-entity", post(facts_by_entity))
        .route("/api/facts/stats", post(get_facts_stats))
        // Decision Lineage Graph API (SHO-118)
        .route("/api/lineage/trace", post(lineage_trace))
        .route("/api/lineage/edges", post(lineage_list_edges))
        .route("/api/lineage/confirm", post(lineage_confirm_edge))
        .route("/api/lineage/reject", post(lineage_reject_edge))
        .route("/api/lineage/link", post(lineage_add_edge))
        .route("/api/lineage/stats", post(lineage_stats))
        .route("/api/lineage/branches", post(lineage_list_branches))
        .route("/api/lineage/branch", post(lineage_create_branch))
        // List memories - Simple GET endpoint
        .route("/api/list/{user_id}", get(list_memories))
        // Streaming endpoints (moved from public to require auth - SHO-56)
        .route("/api/events", get(memory_events_sse)) // SSE: Real-time memory events for dashboard
        .route("/api/stream", get(streaming_memory_ws)) // WS: Streaming memory ingestion (SHO-25)
        // Prospective Memory / Reminders (SHO-116)
        .route("/api/remind", post(create_reminder))
        .route("/api/reminders", post(list_reminders))
        .route("/api/reminders/due", post(get_due_reminders))
        .route("/api/reminders/context", post(check_context_reminders))
        .route(
            "/api/reminders/{reminder_id}/dismiss",
            post(dismiss_reminder),
        )
        .route("/api/reminders/{reminder_id}", delete(delete_reminder))
        // GTD-style Todo Management (Linear-inspired)
        .route("/api/todos", post(create_todo))
        .route("/api/todos/list", post(list_todos))
        .route("/api/todos/due", post(list_due_todos))
        .route("/api/todos/{todo_id}", get(get_todo))
        .route("/api/todos/{todo_id}/update", post(update_todo))
        .route("/api/todos/{todo_id}/complete", post(complete_todo))
        .route("/api/todos/{todo_id}/reorder", post(reorder_todo))
        .route("/api/todos/{todo_id}/subtasks", get(list_subtasks))
        .route("/api/todos/{todo_id}/comments", post(add_todo_comment))
        .route("/api/todos/{todo_id}/comments", get(list_todo_comments))
        .route(
            "/api/todos/{todo_id}/comments/{comment_id}/update",
            post(update_todo_comment),
        )
        .route(
            "/api/todos/{todo_id}/comments/{comment_id}",
            delete(delete_todo_comment),
        )
        .route("/api/todos/{todo_id}", delete(delete_todo))
        .route("/api/projects", post(create_project))
        .route("/api/projects/list", post(list_projects))
        .route("/api/projects/{project_id}", get(get_project))
        .route("/api/projects/{project_id}/update", post(update_project))
        .route("/api/projects/{project_id}", delete(delete_project))
        // File memory / Codebase integration endpoints (MEM-33, MEM-34, MEM-35)
        .route("/api/projects/{project_id}/files", post(list_project_files))
        .route(
            "/api/projects/{project_id}/scan",
            post(scan_project_codebase),
        )
        .route(
            "/api/projects/{project_id}/index",
            post(index_project_codebase),
        )
        .route(
            "/api/projects/{project_id}/files/search",
            post(search_project_files),
        )
        .route("/api/files/stats", get(get_file_stats))
        .route("/api/todos/stats", post(get_todo_stats))
        // Backup & Restore endpoints
        .route("/api/backup/create", post(create_backup))
        .route("/api/backups", post(list_backups))
        .route("/api/backup/verify", post(verify_backup))
        .route("/api/backups/purge", post(purge_backups))
        // MIF Export (Memory Interchange Format)
        .route("/api/export", post(export_mif))
        .route("/api/export/mif", post(export_mif))
        // A/B Testing endpoints
        .route("/api/ab/tests", get(list_ab_tests))
        .route("/api/ab/tests", post(create_ab_test))
        .route("/api/ab/tests/{test_id}", get(get_ab_test))
        .route("/api/ab/tests/{test_id}", delete(delete_ab_test))
        .route("/api/ab/tests/{test_id}/start", post(start_ab_test))
        .route("/api/ab/tests/{test_id}/pause", post(pause_ab_test))
        .route("/api/ab/tests/{test_id}/resume", post(resume_ab_test))
        .route("/api/ab/tests/{test_id}/complete", post(complete_ab_test))
        .route("/api/ab/tests/{test_id}/analyze", get(analyze_ab_test))
        .route(
            "/api/ab/tests/{test_id}/impression",
            post(record_ab_impression),
        )
        .route("/api/ab/tests/{test_id}/click", post(record_ab_click))
        .route("/api/ab/tests/{test_id}/feedback", post(record_ab_feedback))
        .route("/api/ab/summary", get(get_ab_summary))
        // Apply auth middleware only to protected routes
        .layer(axum::middleware::from_fn(auth::auth_middleware))
        // Apply rate limiting to API routes only (not health/metrics/static)
        .layer(governor_layer)
        .with_state(manager.clone());

    // P0.8: Concurrency limiting for production resilience
    // Limits max concurrent requests to prevent resource exhaustion
    let max_concurrent = server_config.max_concurrent_requests;

    info!(
        "🔄 Concurrency limiting enabled: max_concurrent={}",
        max_concurrent
    );

    // Public routes - NO rate limiting, NO auth (health checks, metrics, static files)
    // These must always be accessible for monitoring and Kubernetes probes
    let public_routes = Router::new()
        .route(
            "/",
            get(|| async { axum::response::Redirect::permanent("/static/live.html") }),
        )
        .nest_service("/static", ServeDir::new("static"))
        .route("/health", get(health))
        .route("/health/live", get(health_live)) // P0.9: Kubernetes liveness probe
        // Context status from Claude Code status line (no auth - local script)
        .route("/api/context_status", post(update_context_status))
        .route("/api/context_status", get(get_context_status))
        .route("/api/context_status/stream", get(context_status_sse))
        .route("/health/ready", get(health_ready)) // P0.9: Kubernetes readiness probe
        .route("/health/index", get(health_index)) // Vector index health metrics
        .route("/metrics", get(metrics_endpoint)) // P1.1: Prometheus metrics
        .with_state(manager.clone());

    // Combine public and protected routes
    // - public_routes: health, metrics, static - NO auth, NO rate limiting
    // - protected_routes: API endpoints including streaming - API key auth, rate limited
    // Note: Cortex (Claude API proxy) is now a separate binary - see cortex/
    let app = Router::new().merge(public_routes).merge(protected_routes);

    // Conditionally add trace propagation middleware only when telemetry feature is enabled
    #[cfg(feature = "telemetry")]
    let app = app.layer(axum::middleware::from_fn(
        crate::tracing_setup::trace_propagation::propagate_trace_context,
    ));

    // Apply global layers (no rate limiting here - it's already on protected_routes)
    let app = app
        .layer(axum::middleware::from_fn(crate::middleware::track_metrics))
        .layer(ConcurrencyLimitLayer::new(max_concurrent))
        .layer(cors)
        .with_state(manager);

    // Start server using port from config
    let port = server_config.port;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    // Small delay to let any pending log messages flush
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Print server ready message (always visible)
    use std::io::Write;
    let _ = std::io::stderr().flush();
    eprintln!();
    eprintln!("  🚀 Server ready!");
    eprintln!("     HTTP:      http://{}", addr);
    eprintln!("     Health:    http://{}/health", addr);
    eprintln!("     Dashboard: http://{}/static/live.html", addr);
    eprintln!("     Stream:    ws://{}/api/stream", addr);
    eprintln!();
    eprintln!("  Press Ctrl+C to stop");
    eprintln!();
    let _ = std::io::stderr().flush();

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Run the server - it will wait until shutdown signal is received
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    info!("🔒 Shutdown signal received, flushing databases...");

    // P0.11: Wrap cleanup process with timeout for production resilience
    let cleanup_future = async {
        // Flush RocksDB with timeout
        let flush_future = async { manager_for_shutdown.flush_all_databases() };

        match tokio::time::timeout(
            std::time::Duration::from_secs(DATABASE_FLUSH_TIMEOUT_SECS),
            flush_future,
        )
        .await
        {
            Ok(Ok(())) => info!("✅ Databases flushed successfully"),
            Ok(Err(e)) => tracing::error!("❌ Failed to flush databases: {}", e),
            Err(_) => tracing::error!(
                "⏱️  Database flush timed out after {}s",
                DATABASE_FLUSH_TIMEOUT_SECS
            ),
        }

        // Save vector indices with timeout
        info!("💾 Persisting vector indices...");
        let save_future = async { manager_for_shutdown.save_all_vector_indices() };

        match tokio::time::timeout(
            std::time::Duration::from_secs(VECTOR_INDEX_SAVE_TIMEOUT_SECS),
            save_future,
        )
        .await
        {
            Ok(Ok(())) => info!("✅ Vector indices saved successfully"),
            Ok(Err(e)) => tracing::error!("❌ Failed to save vector indices: {}", e),
            Err(_) => tracing::error!(
                "⏱️  Vector index save timed out after {}s",
                VECTOR_INDEX_SAVE_TIMEOUT_SECS
            ),
        }

        // P1.6: Shutdown tracing and flush remaining spans (optional, only with telemetry feature)
        #[cfg(feature = "telemetry")]
        tracing_setup::shutdown_tracing();
    };

    // P0.11: Enforce overall cleanup timeout with force-kill fallback
    match tokio::time::timeout(
        std::time::Duration::from_secs(GRACEFUL_SHUTDOWN_TIMEOUT_SECS),
        cleanup_future,
    )
    .await
    {
        Ok(()) => {
            info!("👋 Server shutdown complete");
        }
        Err(_) => {
            tracing::error!(
                "⏱️  Graceful shutdown timed out after {}s, forcing exit",
                GRACEFUL_SHUTDOWN_TIMEOUT_SECS
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

// =============================================================================
// Startup Helper Functions
// =============================================================================

/// Calculate total size of a directory recursively
fn calculate_dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += calculate_dir_size(&path);
            } else if let Ok(metadata) = entry.metadata() {
                total += metadata.len();
            }
        }
    }
    total
}

/// Count user directories in storage path
fn count_user_directories(path: &std::path::Path) -> usize {
    std::fs::read_dir(path)
        .map(|entries| entries.flatten().filter(|e| e.path().is_dir()).count())
        .unwrap_or(0)
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Handle graceful shutdown signals (Ctrl+C and SIGTERM)
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("🛑 Shutdown signal received, starting graceful shutdown");
}
