//! Streaming Memory Ingestion for Implicit Learning
//!
//! Enables continuous memory formation from streaming data without explicit `remember()` calls.
//! Designed for LLM agents, robotics, drones, and other autonomous systems that need
//! to learn implicitly from their environment.
//!
//! # Architecture
//! ```text
//! WebSocket /api/stream
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  StreamingMemoryExtractor                                       │
//! │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
//! │  │ Buffer       │→ │ Triggers     │→ │ Extraction Pipeline    │ │
//! │  │ (messages)   │  │ (time/event) │  │ (NER + dedup + store)  │ │
//! │  └──────────────┘  └──────────────┘  └────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//!     │
//!     ▼
//! Response stream: { memories_created, entities_detected, dedupe_skipped }
//! ```
//!
//! # Supported Stream Modes
//! - **Conversation**: Agent dialogue with user (high semantic content)
//! - **Sensor**: IoT/robotics sensor readings (continuous, needs aggregation)
//! - **Event**: Discrete system events (logs, errors, state changes)
//!
//! # Extraction Triggers
//! - **Time-based**: Flush every N milliseconds (configurable checkpoint_interval_ms)
//! - **Event-based**: Flush on important events (errors, decisions, discoveries)
//! - **Content-based**: Flush when buffer semantic density exceeds threshold
//! - **Manual**: Explicit flush via `{ type: "flush" }` message

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::embeddings::{NerEntity, NeuralNer};
use crate::graph_memory::GraphMemory;
use crate::memory::{Experience, ExperienceType, MemorySystem, Query as MemoryQuery};
use crate::similarity::cosine_similarity;

/// Case-insensitive substring search without allocation.
/// Returns true if `haystack` contains `needle` (ASCII case-insensitive).
/// `needle` MUST be lowercase ASCII for correct results.
#[inline]
pub fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    let needle_bytes = needle.as_bytes();
    let needle_len = needle_bytes.len();
    if needle_len == 0 {
        return true;
    }
    let haystack_bytes = haystack.as_bytes();
    if haystack_bytes.len() < needle_len {
        return false;
    }
    // Sliding window search
    'outer: for start in 0..=(haystack_bytes.len() - needle_len) {
        for (i, &needle_byte) in needle_bytes.iter().enumerate() {
            if haystack_bytes[start + i].to_ascii_lowercase() != needle_byte {
                continue 'outer;
            }
        }
        return true;
    }
    false
}

/// Quick hash for exact deduplication (public for benchmarking).
/// Uses case-insensitive byte-level hashing with batched writes to minimize hasher call overhead.
#[inline]
pub fn content_hash(content: &str) -> u64 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let trimmed = content.trim();
    let bytes = trimmed.as_bytes();

    // Process in 64-byte chunks to minimize hasher.write() calls
    // while keeping the buffer on the stack
    const CHUNK_SIZE: usize = 64;
    let mut buffer: [u8; CHUNK_SIZE] = [0u8; CHUNK_SIZE];

    let mut i = 0;
    while i + CHUNK_SIZE <= bytes.len() {
        // Fill buffer with lowercase bytes
        for j in 0..CHUNK_SIZE {
            buffer[j] = bytes[i + j].to_ascii_lowercase();
        }
        hasher.write(&buffer);
        i += CHUNK_SIZE;
    }

    // Handle remaining bytes
    let remaining = bytes.len() - i;
    if remaining > 0 {
        for j in 0..remaining {
            buffer[j] = bytes[i + j].to_ascii_lowercase();
        }
        hasher.write(&buffer[..remaining]);
    }

    hasher.finish()
}

/// Stream processing modes - determines extraction behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamMode {
    /// Agent-user dialogue - extract semantic concepts, entities, decisions
    Conversation,
    /// IoT/robotics sensor data - aggregate readings, detect anomalies
    Sensor,
    /// Discrete system events - logs, errors, state changes
    Event,
}

impl Default for StreamMode {
    fn default() -> Self {
        StreamMode::Conversation
    }
}

/// Configuration for automatic memory extraction from streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Minimum importance threshold for memory creation (0.0 - 1.0)
    /// Lower values = more memories created, higher = only important content
    #[serde(default = "default_min_importance")]
    pub min_importance: f32,

    /// Automatically deduplicate similar memories
    #[serde(default = "default_true")]
    pub auto_dedupe: bool,

    /// Similarity threshold for deduplication (0.0 - 1.0)
    /// Higher = stricter dedup, lower = more aggressive dedup
    #[serde(default = "default_dedupe_threshold")]
    pub dedupe_threshold: f32,

    /// Checkpoint interval in milliseconds (time-based trigger)
    /// Set to 0 to disable time-based extraction
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval_ms: u64,

    /// Maximum messages to buffer before forced flush
    #[serde(default = "default_max_buffer_size")]
    pub max_buffer_size: usize,

    /// Extract entities using NER
    #[serde(default = "default_true")]
    pub extract_entities: bool,

    /// Create graph relationships between extracted entities
    #[serde(default = "default_true")]
    pub create_relationships: bool,

    /// Merge consecutive messages from same source
    #[serde(default = "default_true")]
    pub merge_consecutive: bool,

    /// Event types that trigger immediate extraction
    #[serde(default = "default_trigger_events")]
    pub trigger_events: Vec<String>,

    // === Context Injection Configuration ===
    /// Enable proactive context injection (surface relevant memories on each message)
    #[serde(default = "default_true")]
    pub enable_context_injection: bool,

    /// Minimum relevance score for context injection (0.0 - 1.0)
    /// Higher = stricter, surfaces only highly relevant memories
    #[serde(default = "default_injection_min_relevance")]
    pub injection_min_relevance: f32,

    /// Maximum memories to inject per message
    #[serde(default = "default_injection_max_memories")]
    pub injection_max_memories: usize,

    /// Cooldown in seconds before re-injecting same memory
    #[serde(default = "default_injection_cooldown")]
    pub injection_cooldown_secs: u64,
}

fn default_min_importance() -> f32 {
    0.3
}
fn default_true() -> bool {
    true
}
fn default_dedupe_threshold() -> f32 {
    0.85
}
fn default_checkpoint_interval() -> u64 {
    5000 // 5 seconds
}
fn default_max_buffer_size() -> usize {
    50
}
fn default_trigger_events() -> Vec<String> {
    vec![
        "error".to_string(),
        "decision".to_string(),
        "discovery".to_string(),
        "learning".to_string(),
    ]
}
fn default_injection_min_relevance() -> f32 {
    0.70 // Same default as InjectionConfig
}
fn default_injection_max_memories() -> usize {
    3 // Same default as InjectionConfig
}
fn default_injection_cooldown() -> u64 {
    180 // 3 minutes
}

// Validation bounds to prevent DoS via malformed configs
const MIN_CHECKPOINT_INTERVAL_MS: u64 = 100; // Minimum 100ms to prevent tight loops
const MAX_CHECKPOINT_INTERVAL_MS: u64 = 3_600_000; // Maximum 1 hour
const MAX_BUFFER_SIZE: usize = 10_000; // Maximum 10k messages to prevent memory exhaustion
const MAX_TRIGGER_EVENTS: usize = 100; // Maximum trigger event types
const MAX_SEEN_HASHES: usize = 10_000; // Maximum dedup hashes per session

impl ExtractionConfig {
    /// Validate and clamp config values to sane ranges
    /// This prevents DoS attacks via malformed configurations
    pub fn validate_and_clamp(&mut self) {
        // Clamp importance to [0.0, 1.0]
        self.min_importance = self.min_importance.clamp(0.0, 1.0);

        // Clamp dedupe threshold to [0.0, 1.0]
        self.dedupe_threshold = self.dedupe_threshold.clamp(0.0, 1.0);

        // Clamp checkpoint interval (0 disables, otherwise min 100ms, max 1 hour)
        if self.checkpoint_interval_ms > 0 {
            self.checkpoint_interval_ms = self
                .checkpoint_interval_ms
                .clamp(MIN_CHECKPOINT_INTERVAL_MS, MAX_CHECKPOINT_INTERVAL_MS);
        }

        // Clamp buffer size
        if self.max_buffer_size == 0 {
            self.max_buffer_size = default_max_buffer_size();
        }
        self.max_buffer_size = self.max_buffer_size.min(MAX_BUFFER_SIZE);

        // Limit trigger events
        if self.trigger_events.len() > MAX_TRIGGER_EVENTS {
            self.trigger_events.truncate(MAX_TRIGGER_EVENTS);
        }

        // Clamp injection config
        self.injection_min_relevance = self.injection_min_relevance.clamp(0.0, 1.0);
        self.injection_max_memories = self.injection_max_memories.clamp(1, 10);
        self.injection_cooldown_secs = self.injection_cooldown_secs.clamp(0, 3600);
    }
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_importance: default_min_importance(),
            auto_dedupe: true,
            dedupe_threshold: default_dedupe_threshold(),
            checkpoint_interval_ms: default_checkpoint_interval(),
            max_buffer_size: default_max_buffer_size(),
            extract_entities: true,
            create_relationships: true,
            merge_consecutive: true,
            trigger_events: default_trigger_events(),
            // Context injection enabled by default for bidirectional streaming
            enable_context_injection: true,
            injection_min_relevance: default_injection_min_relevance(),
            injection_max_memories: default_injection_max_memories(),
            injection_cooldown_secs: default_injection_cooldown(),
        }
    }
}

/// WebSocket handshake message - sent by client to initialize stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamHandshake {
    /// User ID for memory isolation
    pub user_id: String,

    /// Stream processing mode
    #[serde(default)]
    pub mode: StreamMode,

    /// Extraction configuration
    #[serde(default)]
    pub extraction_config: ExtractionConfig,

    /// Optional session ID for grouping related memories
    pub session_id: Option<String>,

    /// Optional metadata attached to all extracted memories
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Incoming stream message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamMessage {
    /// Text content for memory extraction
    Content {
        /// The actual content
        content: String,

        /// Optional message source (e.g., "user", "assistant", "system")
        #[serde(default)]
        source: Option<String>,

        /// Optional timestamp (defaults to now)
        #[serde(default)]
        timestamp: Option<DateTime<Utc>>,

        /// Optional explicit importance override
        #[serde(default)]
        importance: Option<f32>,

        /// Optional tags
        #[serde(default)]
        tags: Vec<String>,

        /// Additional metadata
        #[serde(default)]
        metadata: HashMap<String, serde_json::Value>,
    },

    /// Sensor reading (for robotics/IoT)
    Sensor {
        /// Sensor identifier
        sensor_id: String,

        /// Sensor value(s)
        values: HashMap<String, f64>,

        /// Optional timestamp
        #[serde(default)]
        timestamp: Option<DateTime<Utc>>,

        /// Optional unit labels
        #[serde(default)]
        units: HashMap<String, String>,
    },

    /// Discrete event
    Event {
        /// Event name/type
        event: String,

        /// Event description/details
        description: String,

        /// Optional timestamp
        #[serde(default)]
        timestamp: Option<DateTime<Utc>>,

        /// Event severity (info, warning, error)
        #[serde(default)]
        severity: Option<String>,

        /// Additional event data
        #[serde(default)]
        data: HashMap<String, serde_json::Value>,
    },

    /// Manual flush trigger
    Flush,

    /// Ping/keepalive
    Ping,

    /// Close stream gracefully
    Close,
}

/// Result of memory extraction - sent back to client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExtractionResult {
    /// Extraction completed successfully
    Extraction {
        /// Number of memories created
        memories_created: usize,

        /// IDs of created memories
        memory_ids: Vec<String>,

        /// Entities detected by NER
        entities_detected: Vec<DetectedEntity>,

        /// Number of messages skipped due to deduplication
        dedupe_skipped: usize,

        /// Processing time in milliseconds
        processing_time_ms: u64,

        /// Timestamp of extraction
        timestamp: DateTime<Utc>,
    },

    /// Context injection - proactively surfaced memories based on streaming content
    /// This enables bidirectional streaming: client sends context, server surfaces memories
    ContextInjection {
        /// Surfaced memories with relevance scores
        memories: Vec<SurfacedStreamMemory>,

        /// Context signature (hash of triggering content for dedup)
        context_hash: u64,

        /// Processing time in milliseconds
        processing_time_ms: u64,

        /// Timestamp
        timestamp: DateTime<Utc>,
    },

    /// Acknowledgement for non-extraction messages
    Ack {
        /// Original message type acknowledged
        message_type: String,
        timestamp: DateTime<Utc>,
    },

    /// Error during processing
    Error {
        /// Error code
        code: String,
        /// Error message
        message: String,
        /// Whether the stream should be closed
        fatal: bool,
        timestamp: DateTime<Utc>,
    },

    /// Stream closed
    Closed {
        /// Reason for closure
        reason: String,
        /// Total memories created during session
        total_memories_created: usize,
        timestamp: DateTime<Utc>,
    },
}

/// Memory surfaced during streaming context injection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacedStreamMemory {
    /// Memory ID
    pub id: String,

    /// Memory content
    pub content: String,

    /// Memory type (Observation, Decision, Learning, etc.)
    pub memory_type: String,

    /// Composite relevance score (0.0 - 1.0)
    pub relevance: f32,

    /// Breakdown of relevance components
    pub relevance_breakdown: RelevanceBreakdown,

    /// When the memory was created
    pub created_at: DateTime<Utc>,

    /// Tags/entities from the memory
    pub tags: Vec<String>,
}

/// Breakdown of composite relevance score components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceBreakdown {
    /// Semantic similarity component (cosine similarity with context)
    pub semantic: f32,

    /// Recency component (exponential decay based on age)
    pub recency: f32,

    /// Hebbian strength component (graph-based association strength)
    pub strength: f32,
}

/// Entity detected during extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEntity {
    /// Entity text
    pub text: String,
    /// Entity type (PER, ORG, LOC, MISC)
    pub entity_type: String,
    /// Confidence score
    pub confidence: f32,
    /// Whether this entity was already known
    pub existing: bool,
}

impl From<&NerEntity> for DetectedEntity {
    fn from(ner: &NerEntity) -> Self {
        Self {
            text: ner.text.clone(),
            entity_type: ner.entity_type.as_str().to_string(),
            confidence: ner.confidence,
            existing: false,
        }
    }
}

/// Buffered message awaiting extraction
#[derive(Debug, Clone)]
pub struct BufferedMessage {
    pub content: String,
    pub source: Option<String>,
    #[allow(dead_code)]
    pub timestamp: DateTime<Utc>,
    pub importance: Option<f32>,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Maximum concurrent sessions per instance (prevents memory exhaustion)
const MAX_CONCURRENT_SESSIONS: usize = 1000;

/// Session timeout in seconds (1 hour of inactivity)
const SESSION_TIMEOUT_SECS: i64 = 3600;

/// Per-session streaming state
pub struct StreamSession {
    /// Session identifier
    pub session_id: String,

    /// User ID
    pub user_id: String,

    /// Stream mode
    pub mode: StreamMode,

    /// Extraction configuration
    pub config: ExtractionConfig,

    /// Session metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Message buffer
    buffer: VecDeque<BufferedMessage>,

    /// Last extraction timestamp
    last_extraction: DateTime<Utc>,

    /// Last activity timestamp (for stale session cleanup)
    last_activity: DateTime<Utc>,

    /// Total memories created this session
    total_memories_created: usize,

    /// Seen content hashes for deduplication
    seen_hashes: HashSet<u64>,

    /// Recent embeddings for similarity-based dedup (ring buffer)
    /// Reserved for future semantic deduplication
    #[allow(dead_code)]
    recent_embeddings: VecDeque<(String, Vec<f32>)>,

    /// Injection cooldowns: memory_id -> last injection timestamp
    /// Prevents re-surfacing the same memory too frequently
    injection_cooldowns: HashMap<String, DateTime<Utc>>,

    /// Recent context hashes to avoid redundant injections on similar content
    recent_context_hashes: VecDeque<u64>,
}

impl StreamSession {
    pub fn new(handshake: StreamHandshake) -> Self {
        let session_id = handshake
            .session_id
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        // Validate and clamp config to prevent DoS via malformed values
        let mut config = handshake.extraction_config;
        config.validate_and_clamp();

        let now = Utc::now();
        Self {
            session_id,
            user_id: handshake.user_id,
            mode: handshake.mode,
            config,
            metadata: handshake.metadata,
            buffer: VecDeque::with_capacity(64),
            last_extraction: now,
            last_activity: now,
            total_memories_created: 0,
            seen_hashes: HashSet::with_capacity(1024),
            recent_embeddings: VecDeque::with_capacity(100),
            injection_cooldowns: HashMap::new(),
            recent_context_hashes: VecDeque::with_capacity(20),
        }
    }

    /// Check if a memory is on injection cooldown
    fn is_on_injection_cooldown(&self, memory_id: &str) -> bool {
        if let Some(last_injected) = self.injection_cooldowns.get(memory_id) {
            let elapsed = Utc::now()
                .signed_duration_since(*last_injected)
                .num_seconds() as u64;
            elapsed < self.config.injection_cooldown_secs
        } else {
            false
        }
    }

    /// Mark a memory as recently injected
    fn mark_injected(&mut self, memory_id: &str) {
        self.injection_cooldowns
            .insert(memory_id.to_string(), Utc::now());
    }

    /// Cleanup old injection cooldowns to prevent memory leak
    fn cleanup_injection_cooldowns(&mut self) {
        let threshold = self.config.injection_cooldown_secs as i64 * 2;
        let cutoff = Utc::now() - chrono::Duration::seconds(threshold);
        self.injection_cooldowns.retain(|_, ts| *ts > cutoff);
    }

    /// Check if time-based extraction should trigger
    fn should_extract_by_time(&self) -> bool {
        if self.config.checkpoint_interval_ms == 0 {
            return false;
        }

        let elapsed = Utc::now()
            .signed_duration_since(self.last_extraction)
            .num_milliseconds() as u64;

        elapsed >= self.config.checkpoint_interval_ms
    }

    /// Check if buffer size triggers extraction
    fn should_extract_by_size(&self) -> bool {
        self.buffer.len() >= self.config.max_buffer_size
    }

    /// Quick hash for exact deduplication - delegates to public helper
    #[inline]
    fn hash_content(content: &str) -> u64 {
        content_hash(content)
    }

    /// Check if content is duplicate (exact match)
    fn is_exact_duplicate(&self, content: &str) -> bool {
        let hash = Self::hash_content(content);
        self.seen_hashes.contains(&hash)
    }

    /// Add content hash to seen set
    /// Evicts oldest half at capacity to prevent unbounded growth while
    /// retaining recent dedup history (full clear would allow recent duplicates)
    fn mark_seen(&mut self, content: &str) {
        if self.seen_hashes.len() >= MAX_SEEN_HASHES {
            let target = MAX_SEEN_HASHES / 2;
            let mut kept = 0usize;
            self.seen_hashes.retain(|_| {
                kept += 1;
                kept <= target
            });
        }
        let hash = Self::hash_content(content);
        self.seen_hashes.insert(hash);
    }

    /// Add message to buffer
    pub fn buffer_message(&mut self, msg: BufferedMessage) -> bool {
        // Check exact duplicate
        if self.config.auto_dedupe && self.is_exact_duplicate(&msg.content) {
            return false;
        }

        // Merge consecutive messages from same source
        if self.config.merge_consecutive && !self.buffer.is_empty() {
            if let Some(last) = self.buffer.back_mut() {
                if last.source == msg.source {
                    last.content.push('\n');
                    last.content.push_str(&msg.content);
                    last.tags.extend(msg.tags);
                    for (k, v) in msg.metadata {
                        last.metadata.insert(k, v);
                    }
                    return true;
                }
            }
        }

        self.mark_seen(&msg.content);
        self.buffer.push_back(msg);
        true
    }

    /// Drain buffer for extraction
    fn drain_buffer(&mut self) -> Vec<BufferedMessage> {
        self.last_extraction = Utc::now();
        self.buffer.drain(..).collect()
    }

    /// Update last activity timestamp
    fn touch(&mut self) {
        self.last_activity = Utc::now();
    }

    /// Check if session is stale (no activity for SESSION_TIMEOUT_SECS)
    fn is_stale(&self) -> bool {
        let elapsed = Utc::now()
            .signed_duration_since(self.last_activity)
            .num_seconds();
        elapsed > SESSION_TIMEOUT_SECS
    }
}

/// Streaming Memory Extractor - core processing engine
pub struct StreamingMemoryExtractor {
    /// Neural NER for entity extraction
    neural_ner: Arc<NeuralNer>,

    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, StreamSession>>>,
    // PIPE-9: feedback_store removed - feedback momentum now applied in MemorySystem pipeline
}

impl StreamingMemoryExtractor {
    pub fn new(neural_ner: Arc<NeuralNer>) -> Self {
        Self {
            neural_ner,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new streaming session
    pub async fn create_session(&self, handshake: StreamHandshake) -> Result<String, String> {
        // Cleanup stale sessions first
        self.cleanup_stale_sessions().await;

        let mut sessions = self.sessions.write().await;

        // Check max concurrent sessions limit
        if sessions.len() >= MAX_CONCURRENT_SESSIONS {
            return Err(format!(
                "Maximum concurrent sessions ({}) reached. Try again later.",
                MAX_CONCURRENT_SESSIONS
            ));
        }

        let session = StreamSession::new(handshake);
        let session_id = session.session_id.clone();
        sessions.insert(session_id.clone(), session);

        Ok(session_id)
    }

    /// Remove stale sessions that have been inactive for too long
    pub async fn cleanup_stale_sessions(&self) -> usize {
        let mut sessions = self.sessions.write().await;
        let before_count = sessions.len();

        sessions.retain(|_id, session| !session.is_stale());

        let removed = before_count - sessions.len();
        if removed > 0 {
            tracing::info!("Cleaned up {} stale streaming sessions", removed);
        }
        removed
    }

    /// Get current session count
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// Process incoming message
    pub async fn process_message(
        &self,
        session_id: &str,
        message: StreamMessage,
        memory_system: Arc<parking_lot::RwLock<MemorySystem>>,
    ) -> ExtractionResult {
        let mut sessions = self.sessions.write().await;

        let session = match sessions.get_mut(session_id) {
            Some(s) => s,
            None => {
                return ExtractionResult::Error {
                    code: "SESSION_NOT_FOUND".to_string(),
                    message: format!("Session {} not found", session_id),
                    fatal: true,
                    timestamp: Utc::now(),
                }
            }
        };

        // Update activity timestamp to prevent stale session cleanup
        session.touch();

        match message {
            StreamMessage::Content {
                content,
                source,
                timestamp,
                importance,
                tags,
                metadata,
            } => {
                let msg = BufferedMessage {
                    content,
                    source,
                    timestamp: timestamp.unwrap_or_else(Utc::now),
                    importance,
                    tags,
                    metadata,
                };

                let buffered = session.buffer_message(msg);

                // Check triggers
                let should_extract = session.should_extract_by_time()
                    || session.should_extract_by_size()
                    || !buffered; // Force extract if couldn't buffer (duplicate)

                if should_extract {
                    drop(sessions);
                    return self.extract_memories(session_id, memory_system).await;
                }

                ExtractionResult::Ack {
                    message_type: "content".to_string(),
                    timestamp: Utc::now(),
                }
            }

            StreamMessage::Event {
                event,
                description,
                timestamp,
                severity,
                data,
            } => {
                // Check if this event type triggers immediate extraction
                let is_trigger = {
                    let sessions = self.sessions.read().await;
                    sessions
                        .get(session_id)
                        .map(|s| {
                            s.config
                                .trigger_events
                                .iter()
                                .any(|t| t.eq_ignore_ascii_case(&event))
                        })
                        .unwrap_or(false)
                };

                let content = format!(
                    "[{}] {}: {}",
                    severity.unwrap_or_default(),
                    event,
                    description
                );
                let mut metadata: HashMap<String, serde_json::Value> = data;
                metadata.insert("event_type".to_string(), serde_json::json!(event));

                let msg = BufferedMessage {
                    content,
                    source: Some("event".to_string()),
                    timestamp: timestamp.unwrap_or_else(Utc::now),
                    importance: if is_trigger { Some(0.8) } else { None },
                    tags: vec![event.clone()],
                    metadata,
                };

                {
                    let mut sessions = self.sessions.write().await;
                    if let Some(session) = sessions.get_mut(session_id) {
                        session.buffer_message(msg);
                    }
                }

                if is_trigger {
                    return self.extract_memories(session_id, memory_system).await;
                }

                ExtractionResult::Ack {
                    message_type: "event".to_string(),
                    timestamp: Utc::now(),
                }
            }

            StreamMessage::Sensor {
                sensor_id,
                values,
                timestamp,
                units,
            } => {
                // Format sensor reading as content
                let mut parts: Vec<String> = Vec::new();
                for (key, value) in &values {
                    let unit = units.get(key).map(|u| u.as_str()).unwrap_or("");
                    parts.push(format!("{}={}{}", key, value, unit));
                }
                let content = format!("[{}] {}", sensor_id, parts.join(", "));

                let msg = BufferedMessage {
                    content,
                    source: Some(format!("sensor:{}", sensor_id)),
                    timestamp: timestamp.unwrap_or_else(Utc::now),
                    importance: None,
                    tags: vec!["sensor".to_string(), sensor_id],
                    metadata: HashMap::new(),
                };

                let should_extract = {
                    let mut sessions = self.sessions.write().await;
                    if let Some(session) = sessions.get_mut(session_id) {
                        session.buffer_message(msg);
                        session.should_extract_by_time() || session.should_extract_by_size()
                    } else {
                        return ExtractionResult::Error {
                            code: "SESSION_NOT_FOUND".to_string(),
                            message: format!("Session '{}' not found", session_id),
                            fatal: true,
                            timestamp: Utc::now(),
                        };
                    }
                };

                // Sensors use time-based extraction primarily
                if should_extract {
                    return self.extract_memories(session_id, memory_system).await;
                }

                ExtractionResult::Ack {
                    message_type: "sensor".to_string(),
                    timestamp: Utc::now(),
                }
            }

            StreamMessage::Flush => {
                drop(sessions);
                self.extract_memories(session_id, memory_system).await
            }

            StreamMessage::Ping => ExtractionResult::Ack {
                message_type: "ping".to_string(),
                timestamp: Utc::now(),
            },

            StreamMessage::Close => {
                // Final extraction before closing
                drop(sessions);
                let final_result = self.extract_memories(session_id, memory_system).await;

                // Get total count and remove session
                let mut sessions = self.sessions.write().await;
                let total = sessions
                    .get(session_id)
                    .map(|s| s.total_memories_created)
                    .unwrap_or(0);
                sessions.remove(session_id);

                ExtractionResult::Closed {
                    reason: "client_requested".to_string(),
                    total_memories_created: total
                        + match &final_result {
                            ExtractionResult::Extraction {
                                memories_created, ..
                            } => *memories_created,
                            _ => 0,
                        },
                    timestamp: Utc::now(),
                }
            }
        }
    }

    /// Extract memories from buffered messages
    async fn extract_memories(
        &self,
        session_id: &str,
        memory_system: Arc<parking_lot::RwLock<MemorySystem>>,
    ) -> ExtractionResult {
        let start = std::time::Instant::now();

        // Get session and drain buffer
        let (messages, config, user_metadata, mode) = {
            let mut sessions = self.sessions.write().await;
            let session = match sessions.get_mut(session_id) {
                Some(s) => s,
                None => {
                    return ExtractionResult::Error {
                        code: "SESSION_NOT_FOUND".to_string(),
                        message: format!("Session {} not found", session_id),
                        fatal: true,
                        timestamp: Utc::now(),
                    }
                }
            };

            let messages = session.drain_buffer();
            let config = session.config.clone();
            let metadata = session.metadata.clone();
            let mode = session.mode;

            (messages, config, metadata, mode)
        };

        if messages.is_empty() {
            return ExtractionResult::Extraction {
                memories_created: 0,
                memory_ids: vec![],
                entities_detected: vec![],
                dedupe_skipped: 0,
                processing_time_ms: start.elapsed().as_millis() as u64,
                timestamp: Utc::now(),
            };
        }

        let mut memory_ids = Vec::new();
        let mut all_entities = Vec::new();
        let mut dedupe_skipped = 0;

        // Process each buffered message
        for msg in messages {
            // Calculate importance
            let importance = msg
                .importance
                .unwrap_or_else(|| Self::calculate_importance(&msg.content, mode, &config));

            // Skip low importance content
            if importance < config.min_importance {
                dedupe_skipped += 1;
                continue;
            }

            // Extract entities if enabled
            let entities: Vec<NerEntity> = if config.extract_entities {
                match self.neural_ner.extract(&msg.content) {
                    Ok(ents) => ents,
                    Err(e) => {
                        tracing::debug!("NER extraction failed: {}", e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };

            // Add to detected entities list
            for ent in &entities {
                all_entities.push(DetectedEntity::from(ent));
            }

            // Determine experience type based on mode and content
            let experience_type = Self::determine_experience_type(mode, &msg);

            // Merge metadata: Convert serde_json::Value to String for Experience.metadata
            let mut string_metadata: HashMap<String, String> = HashMap::new();
            for (k, v) in user_metadata.iter() {
                string_metadata.insert(k.clone(), v.to_string());
            }
            for (k, v) in msg.metadata {
                string_metadata.insert(k, v.to_string());
            }
            // Clone tags before they're consumed (needed for Experience.tags)
            let tags: Vec<String> = msg.tags.clone();

            // Add tags as metadata
            for tag in &tags {
                string_metadata.insert(format!("tag:{}", tag), "true".to_string());
            }

            // Merge entities from NER with tags (tags can serve as entity hints)
            let mut all_entity_names: Vec<String> =
                entities.iter().map(|e| e.text.clone()).collect();
            for tag in &tags {
                if !all_entity_names.iter().any(|e| e.eq_ignore_ascii_case(tag)) {
                    all_entity_names.push(tag.clone());
                }
            }

            // Create experience with proper fields including tags
            let experience = Experience {
                content: msg.content,
                experience_type,
                entities: all_entity_names,
                metadata: string_metadata,
                embeddings: None, // Will be computed by MemorySystem
                tags,
                ..Default::default()
            };

            // Store memory using remember() method
            let memory_sys = memory_system.read();
            match memory_sys.remember(experience, Some(msg.timestamp)) {
                // Use message timestamp
                Ok(memory_id) => {
                    // MemoryId is a newtype around Uuid - access inner uuid for string
                    memory_ids.push(memory_id.0.to_string());
                }
                Err(e) => {
                    tracing::warn!("Failed to store streaming memory: {}", e);
                }
            }
        }

        // Update session stats
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                session.total_memories_created += memory_ids.len();
            }
        }

        ExtractionResult::Extraction {
            memories_created: memory_ids.len(),
            memory_ids,
            entities_detected: all_entities,
            dedupe_skipped,
            processing_time_ms: start.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        }
    }

    /// Calculate importance based on content and mode
    fn calculate_importance(content: &str, mode: StreamMode, _config: &ExtractionConfig) -> f32 {
        let mut importance: f32 = 0.5;

        // Length factor (longer = more detailed = more important)
        let word_count = content.split_whitespace().count();
        if word_count > 50 {
            importance += 0.1;
        } else if word_count < 10 {
            importance -= 0.1;
        }

        // Mode-specific adjustments
        match mode {
            StreamMode::Conversation => {
                // Questions are important
                if content.contains('?') {
                    importance += 0.15;
                }
                // Code blocks are important
                if content.contains("```") || content.contains("fn ") || content.contains("def ") {
                    importance += 0.2;
                }
                // Error mentions (case-insensitive without allocation)
                if contains_ignore_ascii_case(content, "error")
                    || contains_ignore_ascii_case(content, "failed")
                {
                    importance += 0.2;
                }
            }
            StreamMode::Sensor => {
                // Anomaly detection would go here
                // For now, all sensor data gets baseline importance
                importance = 0.4;
            }
            StreamMode::Event => {
                // Error events are most important (case-insensitive without allocation)
                if contains_ignore_ascii_case(content, "error") {
                    importance += 0.3;
                } else if contains_ignore_ascii_case(content, "warning") {
                    importance += 0.15;
                }
            }
        }

        importance.clamp(0.0, 1.0)
    }

    /// Determine experience type from mode and message
    fn determine_experience_type(mode: StreamMode, msg: &BufferedMessage) -> ExperienceType {
        // Check tags first (case-insensitive without allocation)
        for tag in &msg.tags {
            if contains_ignore_ascii_case(tag, "error") {
                return ExperienceType::Error;
            }
            if contains_ignore_ascii_case(tag, "decision") {
                return ExperienceType::Decision;
            }
            if contains_ignore_ascii_case(tag, "learning") {
                return ExperienceType::Learning;
            }
            if contains_ignore_ascii_case(tag, "discovery") {
                return ExperienceType::Discovery;
            }
        }

        // Fall back to mode-based defaults
        match mode {
            StreamMode::Conversation => ExperienceType::Conversation,
            StreamMode::Sensor => ExperienceType::Observation,
            StreamMode::Event => ExperienceType::Observation,
        }
    }

    /// Close session and cleanup
    pub async fn close_session(&self, session_id: &str) -> Option<usize> {
        let mut sessions = self.sessions.write().await;
        sessions
            .remove(session_id)
            .map(|s| s.total_memories_created)
    }

    /// Get session stats
    /// Surface relevant memories for streaming content (context injection)
    ///
    /// This is the core of bidirectional streaming: as content flows in,
    /// relevant memories are surfaced back to the client.
    pub async fn inject_context(
        &self,
        session_id: &str,
        content: &str,
        memory_system: Arc<parking_lot::RwLock<MemorySystem>>,
        graph_memory: Arc<parking_lot::RwLock<GraphMemory>>,
    ) -> Option<ExtractionResult> {
        let start = std::time::Instant::now();

        // Get session config and check if injection is enabled
        let (config, _user_id) = {
            let sessions = self.sessions.read().await;
            let session = sessions.get(session_id)?;
            if !session.config.enable_context_injection {
                return None;
            }
            (session.config.clone(), session.user_id.clone())
        };

        // Hash content for dedup tracking
        let context_hash = content_hash(content);

        // Check if we recently injected for similar context
        {
            let sessions = self.sessions.read().await;
            if let Some(session) = sessions.get(session_id) {
                if session.recent_context_hashes.contains(&context_hash) {
                    return None; // Skip - already injected for this context
                }
            }
        }

        // Compute context embedding
        let content_for_embed = content.to_string();
        let memory_for_embed = memory_system.clone();
        let context_embedding: Vec<f32> = tokio::task::spawn_blocking(move || {
            let guard = memory_for_embed.read();
            guard
                .compute_embedding(&content_for_embed)
                .unwrap_or_else(|_| vec![0.0; 384])
        })
        .await
        .ok()?;

        // Create injection config from extraction config
        // Streaming injection config (simple values, no InjectionConfig struct needed)
        let min_relevance = config.injection_min_relevance;
        let max_per_message = config.injection_max_memories;
        let cooldown_seconds = config.injection_cooldown_secs;

        // Query memories semantically
        let content_for_query = content.to_string();
        let max_results = max_per_message * 2; // Fetch more for filtering
        let context_emb = context_embedding.clone();

        // Snapshot injection cooldowns BEFORE spawn_blocking to avoid
        // nested block_on() inside blocking thread (thread pool starvation risk)
        let cooldown_snapshot: HashSet<String> = {
            let sessions_guard = self.sessions.read().await;
            if let Some(session) = sessions_guard.get(session_id) {
                session
                    .injection_cooldowns
                    .iter()
                    .filter(|(_, ts)| {
                        let elapsed = Utc::now().signed_duration_since(**ts).num_seconds() as u64;
                        elapsed < cooldown_seconds
                    })
                    .map(|(id, _)| id.clone())
                    .collect()
            } else {
                HashSet::new()
            }
        };

        let surfaced: Vec<SurfacedStreamMemory> = {
            let memory = memory_system.clone();
            let graph = graph_memory.clone();

            tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                let graph_guard = graph.read();
                let now = Utc::now();

                // Semantic query
                let query = MemoryQuery {
                    query_text: Some(content_for_query),
                    max_results,
                    ..Default::default()
                };
                let results = memory_guard.recall(&query).unwrap_or_default();

                // Use scores from unified 5-layer pipeline (PIPE-9: feedback now in pipeline)
                // Recall already applies: RRF fusion + hebbian + recency + feedback
                const RECENCY_DECAY_RATE: f32 = 0.01; // For breakdown display only

                let mut candidates: Vec<(_, f32, f32, f32, f32)> = results
                    .into_iter()
                    .filter_map(|m| {
                        let memory_embedding = m.experience.embeddings.as_ref()?.clone();

                        // Use score from 5-layer pipeline directly (includes feedback momentum)
                        let score = m.get_score().unwrap_or(0.0);

                        // Compute breakdown components for display
                        let semantic = cosine_similarity(&memory_embedding, &context_emb);
                        let hours_old = (now - m.created_at).num_hours().max(0) as f32;
                        let recency = (-RECENCY_DECAY_RATE * hours_old).exp();
                        let hebbian_strength = graph_guard
                            .get_memory_hebbian_strength(&m.id)
                            .unwrap_or(0.0);

                        Some((m, score, semantic, recency, hebbian_strength))
                    })
                    .collect();

                // Sort by score descending
                candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

                // Filter by threshold, cooldown snapshot, and limit
                candidates
                    .into_iter()
                    .filter(|(m, score, _, _, _)| {
                        if *score < min_relevance {
                            return false;
                        }
                        // Check cooldown from pre-captured snapshot
                        if cooldown_snapshot.contains(&m.id.0.to_string()) {
                            return false;
                        }
                        true
                    })
                    .take(max_per_message)
                    .map(
                        |(m, score, semantic, recency, strength)| SurfacedStreamMemory {
                            id: m.id.0.to_string(),
                            content: m.experience.content.clone(),
                            memory_type: format!("{:?}", m.experience.experience_type),
                            relevance: score,
                            relevance_breakdown: RelevanceBreakdown {
                                semantic,
                                recency,
                                strength,
                            },
                            created_at: m.created_at,
                            tags: m.experience.entities.clone(),
                        },
                    )
                    .collect()
            })
            .await
            .ok()?
        };

        if surfaced.is_empty() {
            return None;
        }

        // Update session state: mark memories as injected and track context hash
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                for mem in &surfaced {
                    session.mark_injected(&mem.id);
                }
                session.recent_context_hashes.push_back(context_hash);
                if session.recent_context_hashes.len() > 20 {
                    session.recent_context_hashes.pop_front();
                }
                session.cleanup_injection_cooldowns();
            }
        }

        Some(ExtractionResult::ContextInjection {
            memories: surfaced,
            context_hash,
            processing_time_ms: start.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        })
    }

    pub async fn get_session_stats(&self, session_id: &str) -> Option<SessionStats> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).map(|s| SessionStats {
            session_id: s.session_id.clone(),
            user_id: s.user_id.clone(),
            mode: s.mode,
            buffer_size: s.buffer.len(),
            total_memories_created: s.total_memories_created,
            last_extraction: s.last_extraction,
        })
    }
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    pub session_id: String,
    pub user_id: String,
    pub mode: StreamMode,
    pub buffer_size: usize,
    pub total_memories_created: usize,
    pub last_extraction: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::NerEntityType;

    #[test]
    fn test_extraction_config_defaults() {
        let config = ExtractionConfig::default();
        assert_eq!(config.min_importance, 0.3);
        assert!(config.auto_dedupe);
        assert_eq!(config.checkpoint_interval_ms, 5000);
        assert_eq!(config.max_buffer_size, 50);

        // Context injection defaults
        assert!(config.enable_context_injection);
        assert_eq!(config.injection_min_relevance, 0.70);
        assert_eq!(config.injection_max_memories, 3);
        assert_eq!(config.injection_cooldown_secs, 180);
    }

    #[test]
    fn test_stream_mode_default() {
        let mode = StreamMode::default();
        assert_eq!(mode, StreamMode::Conversation);
    }

    #[test]
    fn test_content_hash_consistency() {
        let h1 = content_hash("Hello World");
        let h2 = content_hash("hello world");
        let h3 = content_hash("  hello world  ");

        // Should be case-insensitive and trim-aware
        assert_eq!(h1, h2);
        assert_eq!(h2, h3);
    }

    #[test]
    fn test_calculate_importance_conversation() {
        let config = ExtractionConfig::default();

        // Short content = lower importance
        let short =
            StreamingMemoryExtractor::calculate_importance("ok", StreamMode::Conversation, &config);
        assert!(short < 0.5);

        // Question = higher importance
        let question = StreamingMemoryExtractor::calculate_importance(
            "How do I implement streaming in Rust?",
            StreamMode::Conversation,
            &config,
        );
        assert!(question > 0.5);

        // Error mention = higher importance
        let error = StreamingMemoryExtractor::calculate_importance(
            "Error: connection failed to database server unexpectedly while processing request",
            StreamMode::Conversation,
            &config,
        );
        assert!(error > 0.6);
    }

    #[test]
    fn test_determine_experience_type() {
        let msg_error = BufferedMessage {
            content: "test".to_string(),
            source: None,
            timestamp: Utc::now(),
            importance: None,
            tags: vec!["error".to_string()],
            metadata: HashMap::new(),
        };
        assert_eq!(
            StreamingMemoryExtractor::determine_experience_type(
                StreamMode::Conversation,
                &msg_error
            ),
            ExperienceType::Error
        );

        let msg_default = BufferedMessage {
            content: "test".to_string(),
            source: None,
            timestamp: Utc::now(),
            importance: None,
            tags: vec![],
            metadata: HashMap::new(),
        };
        assert_eq!(
            StreamingMemoryExtractor::determine_experience_type(
                StreamMode::Conversation,
                &msg_default
            ),
            ExperienceType::Conversation
        );
        assert_eq!(
            StreamingMemoryExtractor::determine_experience_type(StreamMode::Sensor, &msg_default),
            ExperienceType::Observation
        );
    }

    #[test]
    fn test_stream_handshake_deserialization() {
        let json = r#"{
            "user_id": "test-user",
            "mode": "conversation",
            "extraction_config": {
                "min_importance": 0.5,
                "checkpoint_interval_ms": 10000
            }
        }"#;

        let handshake: StreamHandshake = serde_json::from_str(json).unwrap();
        assert_eq!(handshake.user_id, "test-user");
        assert_eq!(handshake.mode, StreamMode::Conversation);
        assert_eq!(handshake.extraction_config.min_importance, 0.5);
        assert_eq!(handshake.extraction_config.checkpoint_interval_ms, 10000);
        // Defaults should be applied for missing fields
        assert!(handshake.extraction_config.auto_dedupe);
    }

    #[test]
    fn test_stream_message_variants() {
        // Content message
        let content_json = r#"{
            "type": "content",
            "content": "Hello world",
            "source": "user",
            "tags": ["greeting"]
        }"#;
        let msg: StreamMessage = serde_json::from_str(content_json).unwrap();
        matches!(msg, StreamMessage::Content { .. });

        // Event message
        let event_json = r#"{
            "type": "event",
            "event": "error",
            "description": "Database connection failed",
            "severity": "error"
        }"#;
        let msg: StreamMessage = serde_json::from_str(event_json).unwrap();
        matches!(msg, StreamMessage::Event { .. });

        // Flush message
        let flush_json = r#"{"type": "flush"}"#;
        let msg: StreamMessage = serde_json::from_str(flush_json).unwrap();
        matches!(msg, StreamMessage::Flush);
    }

    #[test]
    fn test_detected_entity_from_ner() {
        let ner_entity = NerEntity {
            text: "Microsoft".to_string(),
            entity_type: NerEntityType::Organization,
            confidence: 0.95,
            start: 0,
            end: 9,
        };

        let detected = DetectedEntity::from(&ner_entity);
        assert_eq!(detected.text, "Microsoft");
        assert_eq!(detected.entity_type, "ORG");
        assert_eq!(detected.confidence, 0.95);
        assert!(!detected.existing);
    }
}
