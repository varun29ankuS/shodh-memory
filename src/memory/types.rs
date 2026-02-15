//! Type definitions for the memory system

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::constants::{
    DEFAULT_MAX_RESULTS, IMPORTANCE_FLOOR, RECENCY_FULL_DAYS, RECENCY_HIGH_DAYS,
    RECENCY_HIGH_WEIGHT, RECENCY_LOW_WEIGHT, RECENCY_MEDIUM_DAYS, RECENCY_MEDIUM_WEIGHT,
    SALIENCE_RECENCY_WEIGHT,
};

/// Unique identifier for memories
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)] // Serialize as plain UUID string, not array
pub struct MemoryId(pub Uuid);

/// Shared memory reference for zero-copy operations
///
/// Using Arc<Memory> instead of Memory eliminates expensive cloning
/// of large embedding vectors (384-1536 floats = 1.5-6KB each).
///
/// Performance impact: 10-100x reduction in allocations on hot paths.
pub type SharedMemory = Arc<Memory>;

/// Unique identifier for contexts
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)] // Serialize as plain UUID string, not array
pub struct ContextId(pub Uuid);

/// Experience types that can be recorded
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExperienceType {
    Conversation,
    Decision,
    Error,
    Learning,
    Discovery,
    Pattern,
    Context,
    Task,
    CodeEdit,
    FileAccess,
    Search,
    Command,
    Observation,
    /// Prospective memory - future intention/reminder (SHO-116)
    /// Filtered from normal recall, surfaces via dedicated reminder queries
    /// or when context triggers via spreading activation
    Intention,
}

/// Default experience type for minimal API calls
fn default_experience_type() -> ExperienceType {
    ExperienceType::Observation
}

// =============================================================================
// MULTIMODAL SUPPORT - Images, Audio, Video embeddings
// =============================================================================

/// Media type for multimodal memories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MediaType {
    /// Image file (JPEG, PNG, WebP, etc.)
    Image,
    /// Audio file (WAV, MP3, FLAC, etc.)
    Audio,
    /// Video file (MP4, WebM, etc.)
    Video,
    /// Document file (PDF with images/diagrams)
    Document,
}

/// Reference to attached media with metadata
///
/// Media files are stored externally (filesystem or blob storage).
/// This struct holds the reference and pre-computed embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaRef {
    /// Media type
    pub media_type: MediaType,

    /// External reference (file path, blob hash, or URL)
    /// Not the actual bytes - keeps Memory struct lean
    pub uri: String,

    /// Original filename (for display)
    pub filename: Option<String>,

    /// MIME type (e.g., "image/jpeg", "audio/wav")
    pub mime_type: Option<String>,

    /// File size in bytes
    pub size_bytes: Option<u64>,

    /// Media-specific metadata
    /// Images: {"width": "1920", "height": "1080", "format": "jpeg"}
    /// Audio: {"duration_ms": "30000", "sample_rate": "16000", "channels": "1"}
    /// Video: {"duration_ms": "60000", "width": "1920", "height": "1080", "fps": "30"}
    #[serde(default)]
    pub media_metadata: HashMap<String, String>,

    /// Timestamp within media (for video/audio segments)
    /// Allows referencing a specific moment, not just the whole file
    pub timestamp_ms: Option<u64>,

    /// Duration of the segment (for audio/video clips)
    pub duration_ms: Option<u64>,
}

/// Rich multi-dimensional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichContext {
    pub id: ContextId,

    /// Conversation context - what's being discussed
    pub conversation: ConversationContext,

    /// User context - who the user is, their patterns
    pub user: UserContext,

    /// Project context - what they're working on
    pub project: ProjectContext,

    /// Temporal context - when and patterns over time
    pub temporal: TemporalContext,

    /// Semantic context - relationships and meaning
    pub semantic: SemanticContext,

    /// Code context - related code elements
    pub code: CodeContext,

    /// Document context - related documents
    pub document: DocumentContext,

    /// Environment context - system state, location, etc
    pub environment: EnvironmentContext,

    // =========================================================================
    // SHO-104: RICHER CONTEXT ENCODING
    // =========================================================================
    /// Emotional context - affective state during memory formation (SHO-104)
    /// Captures valence, arousal, and dominant emotion for emotional memory enhancement
    #[serde(default)]
    pub emotional: EmotionalContext,

    /// Source context - tracks where information came from (SHO-104)
    /// Enables source monitoring for memory accuracy and credibility weighting
    #[serde(default)]
    pub source: SourceContext,

    /// Episode context - groups memories into coherent episodes (SHO-104)
    /// Enables temporal ordering and event segmentation within conversations
    #[serde(default)]
    pub episode: EpisodeContext,

    /// Parent context (for hierarchical context)
    pub parent: Option<Box<RichContext>>,

    /// Context embeddings for similarity search
    pub embeddings: Option<Vec<f32>>,

    /// Context decay factor (how relevant this context is over time)
    pub decay_rate: f32,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Last updated
    pub updated_at: DateTime<Utc>,
}

/// Conversation-specific context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationContext {
    /// Current conversation ID
    pub conversation_id: Option<String>,

    /// Topic being discussed
    pub topic: Option<String>,

    /// Recent messages (last N turns)
    pub recent_messages: Vec<String>,

    /// Key entities mentioned
    pub mentioned_entities: Vec<String>,

    /// Active questions/intents
    pub active_intents: Vec<String>,

    /// Conversation mood/tone
    pub tone: Option<String>,
}

/// User-specific context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserContext {
    /// User ID
    pub user_id: Option<String>,

    /// User name
    pub name: Option<String>,

    /// User preferences
    pub preferences: HashMap<String, String>,

    /// User's typical working hours
    pub work_patterns: Vec<TimePattern>,

    /// User's expertise areas
    pub expertise: Vec<String>,

    /// User's goals/objectives
    pub goals: Vec<String>,

    /// User's learning style
    pub learning_style: Option<String>,
}

/// Project-specific context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectContext {
    /// Project ID
    pub project_id: Option<String>,

    /// Project name
    pub name: Option<String>,

    /// Project type (web, mobile, ML, etc)
    pub project_type: Option<String>,

    /// Tech stack
    pub technologies: Vec<String>,

    /// Current sprint/milestone
    pub current_phase: Option<String>,

    /// Related files being worked on
    pub active_files: Vec<String>,

    /// Current task/feature
    pub current_task: Option<String>,

    /// Project dependencies
    pub dependencies: Vec<String>,
}

/// Temporal context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemporalContext {
    /// Time of day
    pub time_of_day: Option<String>,

    /// Day of week
    pub day_of_week: Option<String>,

    /// Session duration
    pub session_duration_minutes: Option<u32>,

    /// Time since last interaction
    pub time_since_last_interaction: Option<i64>,

    /// Recurring patterns detected
    pub patterns: Vec<TimePattern>,

    /// Historical trends
    pub trends: Vec<String>,
}

/// Semantic context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SemanticContext {
    /// Main concepts/topics
    pub concepts: Vec<String>,

    /// Related concepts
    pub related_concepts: Vec<String>,

    /// Concept relationships
    pub relationships: Vec<ConceptRelationship>,

    /// Domain/field
    pub domain: Option<String>,

    /// Abstraction level (high-level vs detailed)
    pub abstraction_level: Option<String>,

    /// Semantic tags
    pub tags: Vec<String>,
}

/// Code-specific context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodeContext {
    /// Current file being edited
    pub current_file: Option<String>,

    /// Current function/class
    pub current_scope: Option<String>,

    /// Related files (imports, dependencies)
    pub related_files: Vec<String>,

    /// Recently edited functions
    pub recent_edits: Vec<String>,

    /// Call stack context
    pub call_stack: Vec<String>,

    /// Git branch
    pub git_branch: Option<String>,

    /// Recent commits
    pub recent_commits: Vec<String>,

    /// Code patterns detected
    pub patterns: Vec<String>,
}

/// Document-specific context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentContext {
    /// Current document ID
    pub document_id: Option<String>,

    /// Document type
    pub document_type: Option<String>,

    /// Section/chapter being read
    pub current_section: Option<String>,

    /// Related documents
    pub related_documents: Vec<String>,

    /// Citations/references
    pub citations: Vec<String>,

    /// Document tags/categories
    pub categories: Vec<String>,
}

/// Environment context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentContext {
    /// Operating system
    pub os: Option<String>,

    /// Device type
    pub device: Option<String>,

    /// Screen resolution/size
    pub screen_size: Option<String>,

    /// Location (if available)
    pub location: Option<String>,

    /// Network status
    pub network: Option<String>,

    /// System resource usage
    pub resources: HashMap<String, String>,
}

// =============================================================================
// SHO-104: RICHER CONTEXT ENCODING
// Based on neuroscience research on multi-dimensional memory encoding
// =============================================================================

/// Emotional context - captures affective state during memory formation
///
/// Research: Emotional memories are encoded differently and retrieved more easily.
/// The amygdala modulates hippocampal encoding based on emotional arousal.
///
/// Reference: LaBar & Cabeza (2006) "Cognitive neuroscience of emotional memory"
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalContext {
    /// Emotional valence: -1.0 (very negative) to 1.0 (very positive)
    /// 0.0 = neutral
    ///
    /// Examples:
    /// - Bug found: -0.3 (mildly negative)
    /// - Feature shipped: 0.7 (positive)
    /// - Critical error: -0.8 (strongly negative)
    #[serde(default)]
    pub valence: f32,

    /// Arousal level: 0.0 (calm) to 1.0 (highly aroused/excited)
    ///
    /// High arousal memories (both positive and negative) are better retained.
    /// Examples:
    /// - Routine task: 0.2 (low arousal)
    /// - Important deadline: 0.8 (high arousal)
    /// - Critical production issue: 0.9 (very high arousal)
    #[serde(default)]
    pub arousal: f32,

    /// Dominant emotion label (optional, for categorical access)
    /// E.g., "joy", "frustration", "surprise", "satisfaction", "anxiety"
    #[serde(default)]
    pub dominant_emotion: Option<String>,

    /// Sentiment of the content itself (not the user's reaction)
    /// Useful for distinguishing "user felt frustrated" vs "content describes frustration"
    #[serde(default)]
    pub content_sentiment: Option<f32>,

    /// Confidence in the emotional assessment (0.0 to 1.0)
    /// Lower if inferred from text, higher if explicitly stated
    #[serde(default)]
    pub confidence: f32,
}

/// Source context - tracks where information came from
///
/// Research: Source monitoring is crucial for memory accuracy.
/// Knowing WHO told you something affects how you weight and retrieve it.
///
/// Reference: Johnson et al. (1993) "Source monitoring"
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceContext {
    /// Type of source
    #[serde(default)]
    pub source_type: SourceType,

    /// Specific source identifier
    /// E.g., "user:alice", "api:openai", "file:readme.md", "url:https://..."
    #[serde(default)]
    pub source_id: Option<String>,

    /// Human-readable source name
    #[serde(default)]
    pub source_name: Option<String>,

    /// Credibility score (0.0 to 1.0)
    /// How reliable is this source? Affects retrieval ranking.
    /// - 1.0: Direct user input, verified facts
    /// - 0.8: Trusted documentation, official sources
    /// - 0.5: General web content, unverified
    /// - 0.3: Inferred/generated content
    #[serde(default = "default_credibility")]
    pub credibility: f32,

    /// Was this information verified/confirmed?
    #[serde(default)]
    pub verified: bool,

    /// Chain of sources (for information that was relayed)
    /// E.g., ["api:openai", "doc:react-docs", "user:alice"]
    #[serde(default)]
    pub source_chain: Vec<String>,
}

fn default_credibility() -> f32 {
    0.7 // Default moderate credibility
}

/// Types of information sources
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum SourceType {
    /// Direct user input (highest credibility default)
    User,
    /// System-generated (logs, errors, metrics)
    System,
    /// External API response
    ExternalApi,
    /// File content
    File,
    /// Web/URL content
    Web,
    /// AI/LLM generated content
    AiGenerated,
    /// Inferred by the system
    Inferred,
    /// Unknown source
    #[default]
    Unknown,
}

/// Episode context - groups memories into coherent episodes
///
/// Research: Episodic memory organizes experiences into bounded events.
/// The hippocampus creates "event boundaries" that segment continuous experience.
///
/// Reference: Zacks et al. (2007) "Event perception and memory"
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EpisodeContext {
    /// Unique episode identifier
    /// All memories in the same conversation/session share this ID
    #[serde(default)]
    pub episode_id: Option<String>,

    /// Sequence number within the episode (1, 2, 3, ...)
    /// Enables temporal ordering within an episode
    #[serde(default)]
    pub sequence_number: Option<u32>,

    /// ID of the immediately preceding memory (temporal chain)
    #[serde(default)]
    pub preceding_memory_id: Option<String>,

    /// Episode type (conversation, task, session, etc.)
    #[serde(default)]
    pub episode_type: Option<String>,

    /// Episode start time
    #[serde(default)]
    pub episode_start: Option<DateTime<Utc>>,

    /// Is this the first memory in the episode?
    #[serde(default)]
    pub is_episode_start: bool,

    /// Is this the last memory in the episode?
    #[serde(default)]
    pub is_episode_end: bool,

    /// Parent episode (for hierarchical episodes)
    /// E.g., a conversation within a larger task session
    #[serde(default)]
    pub parent_episode_id: Option<String>,
}

/// Time pattern for recurring behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePattern {
    pub pattern_type: String,
    pub frequency: String,
    pub time_range: Option<(String, String)>,
    pub confidence: f32,
}

/// Relationship between concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptRelationship {
    pub from: String,
    pub to: String,
    pub relationship_type: RelationshipType,
    pub strength: f32,
}

/// Types of relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    IsA,       // Inheritance
    HasA,      // Composition
    Uses,      // Dependency
    RelatedTo, // General association
    Causes,    // Causation
    PartOf,    // Part-whole
    Similar,   // Similarity
    Opposite,  // Antonym/opposite
}

/// Raw experience data to be stored (ENHANCED with smart defaults)
///
/// Only `content` is required. All other fields have intelligent defaults:
/// - experience_type: Defaults to Observation
/// - context: Optional (null by default)
/// - entities: Empty vector (auto-extracted if empty)
/// - metadata: Empty HashMap
/// - embeddings: Optional (auto-generated)
/// - related_memories: Empty vector
/// - causal_chain: Empty vector
/// - outcomes: Empty vector
/// Structured NER entity record preserving type classification and confidence.
/// Used to carry NER results from handler through to graph insertion
/// without losing type information (Person, Organization, Location, Misc).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NerEntityRecord {
    pub text: String,
    /// NER type: "PER", "ORG", "LOC", "MISC"
    pub entity_type: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Type of experience (defaults to Observation)
    #[serde(default = "default_experience_type")]
    pub experience_type: ExperienceType,

    /// Content of the experience (REQUIRED)
    pub content: String,

    /// RICH CONTEXT instead of simple string (optional, null by default)
    #[serde(default)]
    pub context: Option<RichContext>,

    /// Extracted entities (empty by default, auto-extracted if empty)
    #[serde(default)]
    pub entities: Vec<String>,

    /// Additional metadata (empty by default)
    #[serde(default)]
    pub metadata: HashMap<String, String>,

    /// Content embeddings (optional, auto-generated if null)
    #[serde(default)]
    pub embeddings: Option<Vec<f32>>,

    // =========================================================================
    // MULTIMODAL EMBEDDINGS (optional, for images/audio/video)
    // =========================================================================
    /// Image embeddings from CLIP/SigLIP (512-768 dims)
    /// Generated from attached images via vision encoder
    #[serde(default)]
    pub image_embeddings: Option<Vec<f32>>,

    /// Audio embeddings from Whisper/wav2vec (768-1024 dims)
    /// Generated from attached audio via speech encoder
    #[serde(default)]
    pub audio_embeddings: Option<Vec<f32>>,

    /// Video embeddings (768+ dims)
    /// Can be single embedding (averaged frames) or keyframe embedding
    #[serde(default)]
    pub video_embeddings: Option<Vec<f32>>,

    /// References to attached media files
    /// Stores metadata and URIs, not raw bytes
    #[serde(default)]
    pub media_refs: Vec<MediaRef>,

    /// Related memories (empty by default)
    #[serde(default)]
    pub related_memories: Vec<MemoryId>,

    /// Causality chain - what led to this (empty by default)
    #[serde(default)]
    pub causal_chain: Vec<MemoryId>,

    /// Outcome/result - what happened after (empty by default)
    #[serde(default)]
    pub outcomes: Vec<String>,

    // =========================================================================
    // ROBOTICS FIELDS (optional, backward compatible)
    // =========================================================================
    // NOTE: skip_serializing_if REMOVED - breaks bincode binary format
    // Binary formats like bincode are positional and require all fields to be present
    /// Robot/drone identifier for multi-agent systems
    #[serde(default)]
    pub robot_id: Option<String>,

    /// Mission identifier this experience belongs to
    #[serde(default)]
    pub mission_id: Option<String>,

    /// GPS coordinates (latitude, longitude, altitude)
    #[serde(default)]
    pub geo_location: Option<[f64; 3]>,

    /// Local coordinates (x, y, z in meters)
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

    /// Sensor readings at time of experience
    #[serde(default)]
    pub sensor_data: HashMap<String, f64>,

    // =========================================================================
    // DECISION & LEARNING FIELDS (for action-outcome learning)
    // =========================================================================
    /// Decision context: What state/conditions led to this decision?
    /// E.g., "battery_low=true, obstacle_ahead=true, weather=windy"
    #[serde(default)]
    pub decision_context: Option<HashMap<String, String>>,

    /// Action parameters: Specific parameters of the action taken
    /// E.g., {"speed": "0.5", "turn_angle": "45", "altitude_change": "-10"}
    #[serde(default)]
    pub action_params: Option<HashMap<String, String>>,

    /// Outcome type: success, failure, partial, aborted, timeout
    #[serde(default)]
    pub outcome_type: Option<String>,

    /// Outcome details: What specifically happened?
    #[serde(default)]
    pub outcome_details: Option<String>,

    /// Confidence score for this decision (0.0-1.0)
    /// How confident was the system when making this decision?
    #[serde(default)]
    pub confidence: Option<f32>,

    /// Alternative actions considered but not taken
    /// For learning "what else could have been done"
    #[serde(default)]
    pub alternatives_considered: Vec<String>,

    // =========================================================================
    // ENVIRONMENTAL CONTEXT
    // =========================================================================
    /// Weather conditions: {"wind_speed": "15", "visibility": "good", "precipitation": "none"}
    #[serde(default)]
    pub weather: Option<HashMap<String, String>>,

    /// Terrain type: indoor, outdoor, urban, rural, water, aerial
    #[serde(default)]
    pub terrain_type: Option<String>,

    /// Lighting conditions: bright, dim, dark, variable
    #[serde(default)]
    pub lighting: Option<String>,

    /// Other agents detected: [{"id": "drone_002", "distance": "50m", "type": "friendly"}]
    #[serde(default)]
    pub nearby_agents: Vec<HashMap<String, String>>,

    // =========================================================================
    // FAILURE & ANOMALY TRACKING
    // =========================================================================
    /// Is this a failure/error event?
    #[serde(default)]
    pub is_failure: bool,

    /// Is this an anomaly (unexpected sensor reading, behavior, etc.)?
    #[serde(default)]
    pub is_anomaly: bool,

    /// Severity level: info, warning, error, critical
    #[serde(default)]
    pub severity: Option<String>,

    /// Recovery action taken (if this was a failure)
    #[serde(default)]
    pub recovery_action: Option<String>,

    /// Root cause (if known)
    #[serde(default)]
    pub root_cause: Option<String>,

    // =========================================================================
    // LEARNED PATTERNS & PREDICTIONS
    // =========================================================================
    /// Pattern ID this experience matches (if recognized)
    #[serde(default)]
    pub pattern_id: Option<String>,

    /// Predicted outcome before action was taken
    #[serde(default)]
    pub predicted_outcome: Option<String>,

    /// Was the prediction correct?
    #[serde(default)]
    pub prediction_accurate: Option<bool>,

    /// Tags for quick filtering: ["obstacle", "battery", "navigation", "emergency"]
    #[serde(default)]
    pub tags: Vec<String>,

    // =========================================================================
    // TEMPORAL EXTRACTION (v4 - TEMPR approach for LoCoMo)
    // =========================================================================
    /// Extracted temporal references from content (dates mentioned in text)
    /// Used for temporal filtering in retrieval (key for multi-hop accuracy)
    /// E.g., ["2023-05-07", "2023-06-15"] for memories mentioning specific dates
    #[serde(default)]
    pub temporal_refs: Vec<String>,

    // =========================================================================
    // STRUCTURED NER & CO-OCCURRENCE (pre-extracted by handler)
    // =========================================================================
    /// Structured NER entities with type and confidence (preserves NER classification)
    /// Populated by handler during remember/upsert. Consumed by graph insertion
    /// to create entities with proper labels (Person, Organization, Location, etc.)
    #[serde(default)]
    pub ner_entities: Vec<NerEntityRecord>,

    /// Co-occurrence pairs extracted from content (sentence-level proximity)
    /// Pre-computed by handler to avoid redundant content parsing in downstream passes
    #[serde(default)]
    pub cooccurrence_pairs: Vec<(String, String)>,
}

impl Default for Experience {
    fn default() -> Self {
        Self {
            experience_type: ExperienceType::Observation,
            content: String::new(),
            context: None,
            entities: Vec::new(),
            metadata: HashMap::new(),
            embeddings: None,
            image_embeddings: None,
            audio_embeddings: None,
            video_embeddings: None,
            media_refs: Vec::new(),
            related_memories: Vec::new(),
            causal_chain: Vec::new(),
            outcomes: Vec::new(),
            robot_id: None,
            mission_id: None,
            geo_location: None,
            local_position: None,
            heading: None,
            action_type: None,
            reward: None,
            sensor_data: HashMap::new(),
            decision_context: None,
            action_params: None,
            outcome_type: None,
            outcome_details: None,
            confidence: None,
            alternatives_considered: Vec::new(),
            weather: None,
            terrain_type: None,
            lighting: None,
            nearby_agents: Vec::new(),
            is_failure: false,
            is_anomaly: false,
            severity: None,
            recovery_action: None,
            root_cause: None,
            pattern_id: None,
            predicted_outcome: None,
            prediction_accurate: None,
            tags: Vec::new(),
            temporal_refs: Vec::new(),
            ner_entities: Vec::new(),
            cooccurrence_pairs: Vec::new(),
        }
    }
}

/// Mutable metadata for memory (interior mutability)
/// Separated from immutable core data to enable zero-copy updates via Arc<Memory>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub importance: f32,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
    pub temporal_relevance: f32,
    /// Activation level (spreading activation algorithm)
    /// Decays over time, boosted by access and co-activation
    /// Range: 0.0 (dormant) to 1.0 (highly active)
    pub activation: f32,
}

impl MemoryMetadata {
    /// Boost importance based on access patterns (enterprise feature)
    pub fn boost_importance(&mut self) {
        if self.access_count > 5 {
            self.importance = (self.importance * 1.1).min(1.0);
        }
    }
}

/// Entity reference - lightweight link from Memory to GraphMemory entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRef {
    /// UUID of the entity in GraphMemory
    pub entity_id: Uuid,
    /// Entity name for quick access without graph lookup
    pub name: String,
    /// Relationship type (e.g., "mentioned", "subject", "location")
    pub relation: String,
}

/// Memory tier in the cognitive hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTier {
    /// Active, immediate context (Cowan's focus of attention)
    Working,
    /// Current task/session context
    Session,
    /// Consolidated durable memories
    LongTerm,
    /// Compressed archival storage
    Archive,
}

impl Default for MemoryTier {
    fn default() -> Self {
        MemoryTier::Working
    }
}

/// Type of change made to a memory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChangeType {
    /// Memory was created
    Created,
    /// Content was updated
    ContentUpdated,
    /// Status/state changed (e.g., Linear issue status)
    StatusChanged,
    /// Tags/entities were modified
    TagsUpdated,
    /// Importance was adjusted
    ImportanceAdjusted,
}

/// A revision in memory history - tracks what changed and when
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRevision {
    /// Previous content before this change
    pub previous_content: String,
    /// What type of change occurred
    pub change_type: ChangeType,
    /// When the change happened
    pub changed_at: DateTime<Utc>,
    /// Optional: who/what triggered the change
    pub changed_by: Option<String>,
    /// Optional: brief description of change
    pub change_reason: Option<String>,
}

/// Stored memory with metadata
///
/// This is the UNIFIED memory kernel - the single source of truth.
/// All indices (vector, graph, temporal) are projections of this structure.
///
/// Uses Arc<Mutex<MemoryMetadata>> for interior mutability, enabling updates
/// through Arc<Memory> without cloning large embedding vectors (1.5-6KB each).
/// This eliminates 10-100x allocation overhead on hot paths (record, retrieve).
///
/// Note: Clone is manually implemented to deep-copy metadata (creating a new Arc).
/// This ensures cloned memories are fully independent.
#[derive(Debug)]
pub struct Memory {
    pub id: MemoryId,
    pub experience: Experience,

    // Mutable metadata protected by Mutex for zero-copy updates
    metadata: Arc<parking_lot::Mutex<MemoryMetadata>>,

    pub created_at: DateTime<Utc>,
    pub compressed: bool,

    // ==========================================================================
    // COGNITIVE EXTENSIONS - Unified memory with graph awareness
    // ==========================================================================
    /// Current tier in the memory hierarchy
    /// Memories flow: Working → Session → LongTerm → Archive
    pub tier: MemoryTier,

    /// Entity references - bidirectional links to GraphMemory
    /// Populated during record() via entity extraction
    /// Enables spreading activation without graph lookup
    pub entity_refs: Vec<EntityRef>,

    /// Retrieval tracking ID - set when memory is retrieved
    /// Used for Hebbian feedback loop (reinforce_recall)
    pub last_retrieval_id: Option<Uuid>,

    // ==========================================================================
    // Multi-tenancy support for enterprise deployments
    // ==========================================================================
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub actor_id: Option<String>,

    // Similarity score (only populated in search results, not stored)
    pub score: Option<f32>,

    // ==========================================================================
    // External linking - enables upsert from external sources (Linear, GitHub, etc.)
    // ==========================================================================
    /// External identifier for linking to external systems
    /// Format: "{source}:{id}" e.g. "linear:SHO-39", "github:pr-123"
    /// When set, enables upsert semantics: same external_id = update existing memory
    pub external_id: Option<String>,

    /// Version counter - incremented on each update (starts at 1)
    pub version: u32,

    /// Audit history - tracks all changes to this memory
    /// Only populated for memories with external_id (mutable memories)
    /// Empty for regular immutable memories
    pub history: Vec<MemoryRevision>,

    /// Related todo IDs for bidirectional linking with todo system
    /// Populated when a todo references this memory or vice versa
    pub related_todo_ids: Vec<TodoId>,

    // ==========================================================================
    // HIERARCHY - Memory trees for structured knowledge
    // ==========================================================================
    /// Parent memory ID for hierarchical organization
    /// Enables tree structures: parent -> children
    /// Example: "71-research" -> "algebraic relationships" -> "21 × 27 ≡ -1"
    pub parent_id: Option<MemoryId>,
}

impl Clone for Memory {
    /// Deep clone that creates independent metadata.
    /// This ensures cloned memories don't share mutable state.
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            experience: self.experience.clone(),
            // Deep copy: create new Arc with cloned inner data
            metadata: Arc::new(parking_lot::Mutex::new(self.metadata.lock().clone())),
            created_at: self.created_at,
            compressed: self.compressed,
            tier: self.tier,
            entity_refs: self.entity_refs.clone(),
            last_retrieval_id: self.last_retrieval_id,
            agent_id: self.agent_id.clone(),
            run_id: self.run_id.clone(),
            actor_id: self.actor_id.clone(),
            score: self.score,
            external_id: self.external_id.clone(),
            version: self.version,
            history: self.history.clone(),
            related_todo_ids: self.related_todo_ids.clone(),
            parent_id: self.parent_id.clone(),
        }
    }
}

impl Memory {
    /// Create new memory with given parameters
    /// If `created_at` is None, uses current time (Utc::now())
    pub fn new(
        id: MemoryId,
        experience: Experience,
        importance: f32,
        agent_id: Option<String>,
        run_id: Option<String>,
        actor_id: Option<String>,
        created_at: Option<DateTime<Utc>>,
    ) -> Self {
        let now = created_at.unwrap_or_else(Utc::now);
        Self {
            id,
            experience,
            metadata: Arc::new(parking_lot::Mutex::new(MemoryMetadata {
                importance,
                access_count: 0,
                last_accessed: now,
                temporal_relevance: 1.0,
                activation: 1.0, // Start fully activated (just created)
            })),
            created_at: now,
            compressed: false,
            // Cognitive extensions - initialize to defaults
            tier: MemoryTier::Working,
            entity_refs: Vec::new(),
            last_retrieval_id: None,
            // Multi-tenancy
            agent_id,
            run_id,
            actor_id,
            score: None,
            // External linking - defaults to None (immutable memory)
            external_id: None,
            version: 1,
            history: Vec::new(),
            // Todo system linking - empty by default
            related_todo_ids: Vec::new(),
            // Hierarchy - no parent by default (root memory)
            parent_id: None,
        }
    }

    /// Create a new memory linked to an external system (enables upsert)
    pub fn new_with_external_id(
        id: MemoryId,
        experience: Experience,
        importance: f32,
        external_id: String,
        agent_id: Option<String>,
        run_id: Option<String>,
        actor_id: Option<String>,
        created_at: Option<DateTime<Utc>>,
    ) -> Self {
        let mut memory = Self::new(
            id, experience, importance, agent_id, run_id, actor_id, created_at,
        );
        memory.external_id = Some(external_id);
        memory
    }

    /// Create memory from legacy storage format during migration
    /// Preserves all original field values without modification
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_legacy(
        id: MemoryId,
        experience: Experience,
        importance: f32,
        access_count: u32,
        created_at: DateTime<Utc>,
        last_accessed: DateTime<Utc>,
        compressed: bool,
        tier: MemoryTier,
        entity_refs: Vec<EntityRef>,
        activation: f32,
        last_retrieval_id: Option<uuid::Uuid>,
        agent_id: Option<String>,
        run_id: Option<String>,
        actor_id: Option<String>,
        temporal_relevance: f32,
        score: Option<f32>,
        external_id: Option<String>,
        version: u32,
        history: Vec<MemoryRevision>,
        related_todo_ids: Vec<TodoId>,
    ) -> Self {
        Self {
            id,
            experience,
            metadata: Arc::new(parking_lot::Mutex::new(MemoryMetadata {
                importance,
                access_count,
                last_accessed,
                temporal_relevance,
                activation,
            })),
            created_at,
            compressed,
            tier,
            entity_refs,
            last_retrieval_id,
            agent_id,
            run_id,
            actor_id,
            score,
            external_id,
            version,
            history,
            related_todo_ids,
            // Legacy memories don't have hierarchy - default to root
            parent_id: None,
        }
    }

    /// Check if this memory has been soft-deleted (marked as forgotten)
    pub fn is_forgotten(&self) -> bool {
        self.experience
            .metadata
            .get("forgotten")
            .map(|v| v == "true")
            .unwrap_or(false)
    }

    /// Update this memory's content, pushing old content to history
    /// Returns the new version number
    pub fn update_content(
        &mut self,
        new_content: String,
        change_type: ChangeType,
        changed_by: Option<String>,
        change_reason: Option<String>,
    ) -> u32 {
        // Push current content to history
        self.history.push(MemoryRevision {
            previous_content: self.experience.content.clone(),
            change_type,
            changed_at: Utc::now(),
            changed_by,
            change_reason,
        });

        // Update content
        self.experience.content = new_content;
        self.version += 1;

        self.version
    }

    /// Get the full history of this memory
    pub fn get_history(&self) -> &[MemoryRevision] {
        &self.history
    }

    /// Check if this memory has been updated (version > 1)
    pub fn has_history(&self) -> bool {
        self.version > 1
    }

    /// Add entity reference (bidirectional link to graph)
    pub fn add_entity_ref(&mut self, entity_id: Uuid, name: String, relation: String) {
        // Avoid duplicates
        if !self.entity_refs.iter().any(|r| r.entity_id == entity_id) {
            self.entity_refs.push(EntityRef {
                entity_id,
                name,
                relation,
            });
        }
    }

    /// Add a related todo ID (bidirectional link to todo system)
    pub fn add_related_todo(&mut self, todo_id: TodoId) {
        if !self.related_todo_ids.contains(&todo_id) {
            self.related_todo_ids.push(todo_id);
        }
    }

    /// Remove a related todo ID
    pub fn remove_related_todo(&mut self, todo_id: &TodoId) {
        self.related_todo_ids.retain(|id| id != todo_id);
    }

    /// Check if this memory is linked to a specific todo
    pub fn has_related_todo(&self, todo_id: &TodoId) -> bool {
        self.related_todo_ids.contains(todo_id)
    }

    /// Get entity IDs for graph operations
    pub fn entity_ids(&self) -> Vec<Uuid> {
        self.entity_refs.iter().map(|r| r.entity_id).collect()
    }

    /// Boost activation (spreading activation algorithm, thread-safe)
    pub fn activate(&self, amount: f32) {
        let mut meta = self.metadata.lock();
        meta.activation = (meta.activation + amount).min(1.0);
    }

    /// Decay activation over time (thread-safe)
    pub fn decay_activation(&self, decay_factor: f32) {
        self.metadata.lock().activation *= decay_factor;
    }

    /// Set retrieval tracking ID for Hebbian feedback
    pub fn mark_retrieved(&mut self, retrieval_id: Uuid) {
        self.last_retrieval_id = Some(retrieval_id);
        // Also record access
        self.record_access();
    }

    /// Promote to next tier
    pub fn promote(&mut self) {
        self.tier = match self.tier {
            MemoryTier::Working => MemoryTier::Session,
            MemoryTier::Session => MemoryTier::LongTerm,
            MemoryTier::LongTerm => MemoryTier::Archive,
            MemoryTier::Archive => MemoryTier::Archive, // Already at max
        };
    }

    /// Set the similarity score (used in search results)
    pub fn set_score(&mut self, score: f32) {
        self.score = Some(score);
    }

    /// Get the similarity score (only populated in search results)
    pub fn get_score(&self) -> Option<f32> {
        self.score
    }

    /// Demote to previous tier (for decay)
    pub fn demote(&mut self) {
        self.tier = match self.tier {
            MemoryTier::Working => MemoryTier::Working, // Can't go lower
            MemoryTier::Session => MemoryTier::Working,
            MemoryTier::LongTerm => MemoryTier::Session,
            MemoryTier::Archive => MemoryTier::LongTerm,
        };
    }

    /// Get current importance (thread-safe)
    pub fn importance(&self) -> f32 {
        self.metadata.lock().importance
    }

    /// Get access count (thread-safe)
    pub fn access_count(&self) -> u32 {
        self.metadata.lock().access_count
    }

    /// Get last accessed time (thread-safe)
    pub fn last_accessed(&self) -> DateTime<Utc> {
        self.metadata.lock().last_accessed
    }

    /// Get temporal relevance (thread-safe)
    pub fn temporal_relevance(&self) -> f32 {
        self.metadata.lock().temporal_relevance
    }

    /// Get activation level (thread-safe)
    pub fn activation(&self) -> f32 {
        self.metadata.lock().activation
    }

    /// Set activation level directly (thread-safe, clamped to [0.0, 1.0])
    ///
    /// Use cases:
    /// - Data restoration from backups
    /// - Migration from older data formats
    /// - Testing with specific activation states
    ///
    /// For normal operation, prefer `activate()` (adds) and `decay_activation()` (multiplies).
    pub fn set_activation(&self, activation: f32) {
        self.metadata.lock().activation = activation.clamp(0.0, 1.0);
    }

    /// Update access metadata (zero-copy through Arc)
    pub fn update_access(&self) {
        let mut meta = self.metadata.lock();
        meta.last_accessed = Utc::now();
        meta.access_count += 1;
        meta.boost_importance();
    }

    /// Set importance (thread-safe, clamped to [0.0, 1.0])
    pub fn set_importance(&self, importance: f32) {
        self.metadata.lock().importance = importance.clamp(0.0, 1.0);
    }

    /// Set temporal relevance (thread-safe)
    pub fn set_temporal_relevance(&self, relevance: f32) {
        self.metadata.lock().temporal_relevance = relevance;
    }

    // =========================================================================
    // ADAPTIVE LEARNING METHODS - For outcome feedback loop
    // =========================================================================

    /// Record that this memory was accessed (updates count and timestamp)
    ///
    /// Call this whenever a memory is retrieved, even if just viewed.
    /// For stronger reinforcement when memory helped a task, use `boost_importance`.
    pub fn record_access(&self) {
        let mut meta = self.metadata.lock();
        meta.last_accessed = Utc::now();
        meta.access_count += 1;
    }

    /// Boost importance by a factor (for helpful memories)
    ///
    /// Uses additive boost clamped to [0.0, 1.0]:
    /// - boost of 0.05 = +5% (typical for helpful retrieval)
    /// - boost of 0.10 = +10% (very helpful, task completed successfully)
    ///
    /// Example: memory with importance 0.6 + boost 0.05 -> 0.65
    pub fn boost_importance(&self, boost: f32) {
        let mut meta = self.metadata.lock();
        meta.importance = (meta.importance + boost).clamp(0.0, 1.0);
    }

    /// Decay importance by a factor (for misleading memories)
    ///
    /// Uses multiplicative decay clamped to [IMPORTANCE_FLOOR, 1.0]:
    /// - decay of 0.10 = -10% (memory was misleading)
    /// - Never drops below IMPORTANCE_FLOOR (0.05) to allow recovery
    ///
    /// The floor prevents complete forgetting, mimicking the "savings effect"
    /// in human memory - relearning is faster than initial learning.
    ///
    /// Example: memory with importance 0.6 - decay 0.10 -> 0.54
    pub fn decay_importance(&self, decay: f32) {
        let mut meta = self.metadata.lock();
        meta.importance = (meta.importance * (1.0 - decay)).max(IMPORTANCE_FLOOR);
    }

    /// Get all metadata snapshot (for debugging/stats)
    pub fn metadata_snapshot(&self) -> MemoryMetadata {
        self.metadata.lock().clone()
    }

    // =========================================================================
    // SALIENCE SCORING - Ebbinghaus Forgetting Curve Implementation
    // =========================================================================

    /// Calculate salience score based on Ebbinghaus forgetting curve
    ///
    /// This implements a time-based relevance decay that mimics human memory:
    /// - Memories < 7 days: Full relevance (1.0)
    /// - Memories 8-30 days: High relevance (0.7)
    /// - Memories 31-90 days: Medium relevance (0.4)
    /// - Memories 90+ days: Low relevance (0.1)
    ///
    /// The score is weighted by SALIENCE_RECENCY_WEIGHT (default 1.0) and combined
    /// with importance to produce a final salience score.
    ///
    /// Reference: Ebbinghaus (1885) "Memory: A Contribution to Experimental Psychology"
    ///
    /// # Returns
    /// A salience score between 0.0 and 1.0, where higher = more salient
    pub fn salience_score(&self) -> f32 {
        let age_days = (Utc::now() - self.created_at).num_days();

        // Calculate recency factor based on Ebbinghaus forgetting curve
        let recency_factor = if age_days <= RECENCY_FULL_DAYS {
            1.0 // Full relevance for recent memories
        } else if age_days <= RECENCY_HIGH_DAYS {
            RECENCY_HIGH_WEIGHT // High relevance (0.7)
        } else if age_days <= RECENCY_MEDIUM_DAYS {
            RECENCY_MEDIUM_WEIGHT // Medium relevance (0.4)
        } else {
            RECENCY_LOW_WEIGHT // Low relevance (0.1)
        };

        // Combine recency with importance for final salience
        // Formula: salience = (recency_weight * recency_factor + importance) / 2
        // This balances time-based decay with inherent memory importance
        let importance = self.importance();
        let weighted_recency = SALIENCE_RECENCY_WEIGHT * recency_factor;

        // Weighted average: recency contributes 60%, importance contributes 40%
        // This prioritizes recent memories but preserves important old ones
        (weighted_recency * 0.6 + importance * 0.4).clamp(0.0, 1.0)
    }

    /// Calculate salience score with access-based boost
    ///
    /// Similar to `salience_score()` but also factors in access frequency.
    /// Frequently accessed memories resist forgetting (spacing effect).
    ///
    /// # Returns
    /// A salience score between 0.0 and 1.0
    pub fn salience_score_with_access(&self) -> f32 {
        let base_salience = self.salience_score();
        let access_count = self.access_count();

        // Access boost: logarithmic growth to prevent runaway scores
        // Each access adds a diminishing boost (log2 scale)
        // 1 access: +0, 2: +0.05, 4: +0.1, 8: +0.15, 16: +0.2
        let access_boost = if access_count > 0 {
            ((access_count as f32).log2() * 0.05).min(0.3)
        } else {
            0.0
        };

        (base_salience + access_boost).clamp(0.0, 1.0)
    }

    // =========================================================================
    // HIERARCHY METHODS - For memory trees
    // =========================================================================

    /// Set the parent memory ID
    pub fn set_parent(&mut self, parent_id: Option<MemoryId>) {
        self.parent_id = parent_id;
    }

    /// Get the parent memory ID
    pub fn get_parent(&self) -> Option<&MemoryId> {
        self.parent_id.as_ref()
    }

    /// Check if this memory has a parent
    pub fn has_parent(&self) -> bool {
        self.parent_id.is_some()
    }

    /// Check if this is a root memory (no parent)
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }
}

/// Tree node for memory hierarchy traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTreeNode {
    /// The memory at this node
    pub memory: Memory,
    /// Child nodes
    pub children: Vec<MemoryTreeNode>,
    /// Depth in tree (0 = root)
    pub depth: usize,
}

impl MemoryTreeNode {
    /// Create a new tree node with no children
    pub fn new(memory: Memory, depth: usize) -> Self {
        Self {
            memory,
            children: Vec::new(),
            depth,
        }
    }

    /// Add a child node
    pub fn add_child(&mut self, child: MemoryTreeNode) {
        self.children.push(child);
    }

    /// Get total count of nodes in this subtree (including self)
    pub fn total_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_count()).sum::<usize>()
    }

    /// Format as ASCII tree
    pub fn format_tree(&self) -> String {
        let mut output = String::new();
        self.format_tree_recursive(&mut output, "", true);
        output
    }

    fn format_tree_recursive(&self, output: &mut String, prefix: &str, is_last: bool) {
        let connector = if self.depth == 0 {
            ""
        } else if is_last {
            "└── "
        } else {
            "├── "
        };

        // Truncate content for display
        let content = &self.memory.experience.content;
        let display_content = if content.len() > 60 {
            format!("{}...", &content[..57])
        } else {
            content.clone()
        };

        output.push_str(&format!("{}{}{}\n", prefix, connector, display_content));

        let child_prefix = if self.depth == 0 {
            String::new()
        } else if is_last {
            format!("{}    ", prefix)
        } else {
            format!("{}│   ", prefix)
        };

        for (i, child) in self.children.iter().enumerate() {
            let is_last_child = i == self.children.len() - 1;
            child.format_tree_recursive(output, &child_prefix, is_last_child);
        }
    }
}

/// Summary of a memory tree (for listing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTreeSummary {
    /// Root memory ID
    pub root_id: MemoryId,
    /// Root memory content preview
    pub root_content: String,
    /// Direct child count
    pub child_count: usize,
    /// Total descendant count (all levels)
    pub total_count: usize,
    /// Created timestamp of root
    pub created_at: DateTime<Utc>,
}

// Custom serialization for Memory to flatten the Arc<Mutex<>> field
/// Flat representation for serialization - MUST match exact field order for binary formats
/// This struct ensures symmetric serialize/deserialize with bincode
#[derive(Serialize, Deserialize)]
struct MemoryFlat {
    id: MemoryId,
    experience: Experience,
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    // Cognitive extensions
    tier: MemoryTier,
    entity_refs: Vec<EntityRef>,
    activation: f32,
    last_retrieval_id: Option<Uuid>,
    // Multi-tenancy
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
    // External linking (mutable memories)
    external_id: Option<String>,
    version: u32,
    history: Vec<MemoryRevision>,
    // Todo system linking
    #[serde(default)]
    related_todo_ids: Vec<TodoId>,
    // Hierarchy
    #[serde(default)]
    parent_id: Option<MemoryId>,
}

impl Serialize for Memory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let meta = self.metadata.lock();
        // Convert to flat struct for consistent binary serialization
        let flat = MemoryFlat {
            id: self.id.clone(),
            experience: self.experience.clone(),
            importance: meta.importance,
            access_count: meta.access_count,
            created_at: self.created_at,
            last_accessed: meta.last_accessed,
            compressed: self.compressed,
            // Cognitive extensions
            tier: self.tier,
            entity_refs: self.entity_refs.clone(),
            activation: meta.activation,
            last_retrieval_id: self.last_retrieval_id,
            // Multi-tenancy
            agent_id: self.agent_id.clone(),
            run_id: self.run_id.clone(),
            actor_id: self.actor_id.clone(),
            temporal_relevance: meta.temporal_relevance,
            score: self.score,
            // External linking (mutable memories)
            external_id: self.external_id.clone(),
            version: self.version,
            history: self.history.clone(),
            // Todo system linking
            related_todo_ids: self.related_todo_ids.clone(),
            // Hierarchy
            parent_id: self.parent_id.clone(),
        };
        flat.serialize(serializer)
    }
}

// Custom deserialization for Memory to reconstruct Arc<Mutex<>>
impl<'de> Deserialize<'de> for Memory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let flat = MemoryFlat::deserialize(deserializer)?;
        Ok(Memory {
            id: flat.id,
            experience: flat.experience,
            metadata: Arc::new(parking_lot::Mutex::new(MemoryMetadata {
                importance: flat.importance,
                access_count: flat.access_count,
                last_accessed: flat.last_accessed,
                temporal_relevance: flat.temporal_relevance,
                activation: flat.activation,
            })),
            created_at: flat.created_at,
            compressed: flat.compressed,
            // Cognitive extensions
            tier: flat.tier,
            entity_refs: flat.entity_refs,
            last_retrieval_id: flat.last_retrieval_id,
            // Multi-tenancy
            agent_id: flat.agent_id,
            run_id: flat.run_id,
            actor_id: flat.actor_id,
            score: flat.score,
            // External linking (mutable memories)
            external_id: flat.external_id,
            version: flat.version,
            history: flat.history,
            // Todo system linking
            related_todo_ids: flat.related_todo_ids,
            // Hierarchy
            parent_id: flat.parent_id,
        })
    }
}

/// Spatial filter for geo-based queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoFilter {
    /// Center latitude
    pub lat: f64,
    /// Center longitude
    pub lon: f64,
    /// Search radius in meters
    pub radius_meters: f64,
}

impl GeoFilter {
    pub fn new(lat: f64, lon: f64, radius_meters: f64) -> Self {
        Self {
            lat,
            lon,
            radius_meters,
        }
    }

    /// Calculate haversine distance between two points in meters
    pub fn haversine_distance(&self, other_lat: f64, other_lon: f64) -> f64 {
        const EARTH_RADIUS_METERS: f64 = 6_371_000.0;

        let lat1_rad = self.lat.to_radians();
        let lat2_rad = other_lat.to_radians();
        let delta_lat = (other_lat - self.lat).to_radians();
        let delta_lon = (other_lon - self.lon).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();

        EARTH_RADIUS_METERS * c
    }

    /// Check if a point is within the radius
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        self.haversine_distance(lat, lon) <= self.radius_meters
    }
}

// ============================================================================
// Geohash utilities for efficient spatial indexing
// ============================================================================

/// Base32 character set for geohash encoding
const GEOHASH_CHARS: &[u8] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// Precision reference table (approximate cell dimensions at equator):
/// - 1 char: 5000km x 5000km
/// - 2 chars: 1250km x 625km
/// - 3 chars: 156km x 156km
/// - 4 chars: 39km x 20km
/// - 5 chars: 5km x 5km
/// - 6 chars: 1.2km x 600m
/// - 7 chars: 150m x 150m
/// - 8 chars: 38m x 19m
/// - 9 chars: 5m x 5m (warehouse aisle)
/// - 10 chars: 1.2m x 60cm (shelf location)
/// - 11 chars: 15cm x 15cm (sub-meter)
/// - 12 chars: 4cm x 2cm (high precision)
///
/// Encode latitude/longitude to geohash string
pub fn geohash_encode(lat: f64, lon: f64, precision: usize) -> String {
    let mut lat_range = (-90.0, 90.0);
    let mut lon_range = (-180.0, 180.0);
    let mut hash = String::with_capacity(precision);
    let mut bits = 0u8;
    let mut bit_count = 0;
    let mut is_lon = true;

    while hash.len() < precision {
        if is_lon {
            let mid = (lon_range.0 + lon_range.1) / 2.0;
            if lon >= mid {
                bits = (bits << 1) | 1;
                lon_range.0 = mid;
            } else {
                bits <<= 1;
                lon_range.1 = mid;
            }
        } else {
            let mid = (lat_range.0 + lat_range.1) / 2.0;
            if lat >= mid {
                bits = (bits << 1) | 1;
                lat_range.0 = mid;
            } else {
                bits <<= 1;
                lat_range.1 = mid;
            }
        }
        is_lon = !is_lon;
        bit_count += 1;

        if bit_count == 5 {
            hash.push(GEOHASH_CHARS[bits as usize] as char);
            bits = 0;
            bit_count = 0;
        }
    }

    hash
}

/// Decode geohash to bounding box (min_lat, min_lon, max_lat, max_lon)
/// Invalid characters are skipped to prevent silent data corruption
pub fn geohash_decode(hash: &str) -> (f64, f64, f64, f64) {
    let mut lat_range = (-90.0, 90.0);
    let mut lon_range = (-180.0, 180.0);
    let mut is_lon = true;

    for c in hash.chars() {
        // Skip invalid characters instead of silently using index 0
        let idx = match GEOHASH_CHARS.iter().position(|&x| x == c as u8) {
            Some(i) => i,
            None => continue, // Skip invalid characters
        };
        for i in (0..5).rev() {
            let bit = (idx >> i) & 1;
            if is_lon {
                let mid = (lon_range.0 + lon_range.1) / 2.0;
                if bit == 1 {
                    lon_range.0 = mid;
                } else {
                    lon_range.1 = mid;
                }
            } else {
                let mid = (lat_range.0 + lat_range.1) / 2.0;
                if bit == 1 {
                    lat_range.0 = mid;
                } else {
                    lat_range.1 = mid;
                }
            }
            is_lon = !is_lon;
        }
    }

    (lat_range.0, lon_range.0, lat_range.1, lon_range.1)
}

/// Get 8 neighboring geohashes (N, NE, E, SE, S, SW, W, NW)
pub fn geohash_neighbors(hash: &str) -> Vec<String> {
    if hash.is_empty() {
        return vec![];
    }

    let (min_lat, min_lon, max_lat, max_lon) = geohash_decode(hash);
    let lat_delta = max_lat - min_lat;
    let lon_delta = max_lon - min_lon;
    let center_lat = (min_lat + max_lat) / 2.0;
    let center_lon = (min_lon + max_lon) / 2.0;
    let precision = hash.len();

    let directions: [(f64, f64); 8] = [
        (1.0, 0.0),   // N
        (1.0, 1.0),   // NE
        (0.0, 1.0),   // E
        (-1.0, 1.0),  // SE
        (-1.0, 0.0),  // S
        (-1.0, -1.0), // SW
        (0.0, -1.0),  // W
        (1.0, -1.0),  // NW
    ];

    directions
        .iter()
        .map(|(lat_dir, lon_dir)| {
            let neighbor_lat = center_lat + lat_dir * lat_delta;
            let neighbor_lon = center_lon + lon_dir * lon_delta;
            geohash_encode(
                neighbor_lat.clamp(-90.0, 90.0),
                wrap_longitude(neighbor_lon),
                precision,
            )
        })
        .collect()
}

/// Wrap longitude to [-180, 180]
fn wrap_longitude(lon: f64) -> f64 {
    if lon > 180.0 {
        lon - 360.0
    } else if lon < -180.0 {
        lon + 360.0
    } else {
        lon
    }
}

/// Get optimal geohash precision for a given search radius
/// Returns precision that gives cells roughly matching the radius
pub fn geohash_precision_for_radius(radius_meters: f64) -> usize {
    // Validate input - clamp to reasonable bounds
    let radius = if !radius_meters.is_finite() || radius_meters <= 0.0 {
        1.0 // Default to 1 meter if invalid
    } else if radius_meters > 20_000_000.0 {
        20_000_000.0 // Cap at half Earth circumference
    } else {
        radius_meters
    };

    // Approximate cell sizes at equator (width in meters)
    const CELL_SIZES: [(usize, f64); 12] = [
        (1, 5_000_000.0),
        (2, 1_250_000.0),
        (3, 156_000.0),
        (4, 39_000.0),
        (5, 5_000.0),
        (6, 1_200.0),
        (7, 150.0),
        (8, 38.0),
        (9, 5.0),
        (10, 1.2),
        (11, 0.15),
        (12, 0.04),
    ];

    for (precision, cell_size) in CELL_SIZES.iter() {
        if *cell_size <= radius * 2.0 {
            return *precision;
        }
    }
    12 // Maximum precision
}

/// Get all geohash prefixes to scan for a radius search
/// Returns the center hash and its neighbors at appropriate precision
pub fn geohash_search_prefixes(lat: f64, lon: f64, radius_meters: f64) -> Vec<String> {
    // Clamp coordinates to valid ranges
    let lat = lat.clamp(-90.0, 90.0);
    // Normalize longitude to [-180, 180]
    let lon = if lon > 180.0 {
        lon - 360.0
    } else if lon < -180.0 {
        lon + 360.0
    } else {
        lon
    };

    let precision = geohash_precision_for_radius(radius_meters);
    let center = geohash_encode(lat, lon, precision);
    let mut prefixes = geohash_neighbors(&center);
    prefixes.push(center);
    prefixes
}

/// Query for retrieving memories
#[derive(Debug, Clone)]
pub struct Query {
    // === User Context ===
    /// User ID for per-user temporal fact lookup and personalized retrieval
    pub user_id: Option<String>,

    // === Semantic Search ===
    pub query_text: Option<String>,
    pub query_embedding: Option<Vec<f32>>,

    // === Temporal Filters ===
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,

    // === Content Filters ===
    pub experience_types: Option<Vec<ExperienceType>>,
    pub importance_threshold: Option<f32>,

    // === Robotics Filters ===
    /// Filter by robot/drone identifier
    pub robot_id: Option<String>,
    /// Filter by mission identifier
    pub mission_id: Option<String>,
    /// Spatial filter (geo_location within radius)
    pub geo_filter: Option<GeoFilter>,
    /// Filter by action type
    pub action_type: Option<String>,
    /// Filter by reward range (min, max) for RL-style queries
    pub reward_range: Option<(f32, f32)>,

    // === Decision & Learning Filters ===
    /// Filter by outcome type: success, failure, partial, aborted, timeout
    pub outcome_type: Option<String>,
    /// Filter for failures only
    pub failures_only: bool,
    /// Filter for anomalies only
    pub anomalies_only: bool,
    /// Filter by severity level: info, warning, error, critical
    pub severity: Option<String>,
    /// Filter by tags (any match)
    pub tags: Option<Vec<String>>,
    /// Filter by pattern_id (for finding similar situations)
    pub pattern_id: Option<String>,
    /// Filter by terrain type
    pub terrain_type: Option<String>,
    /// Filter by confidence range (min, max)
    pub confidence_range: Option<(f32, f32)>,

    // === Prospective Memory Signals ===
    /// Future intention keywords/content from pending prospective tasks
    /// When set, memories matching these signals get boosted in retrieval
    /// This enables "future informs present" - pending reminders influence recall
    pub prospective_signals: Option<Vec<String>>,

    // === Episode Context (SHO-temporal) ===
    /// Episode ID for context-aware retrieval
    /// When set, memories from the same episode get a coherence boost
    /// This prevents episode bleeding where unrelated memories mix in results
    pub episode_id: Option<String>,

    // === Result Control ===
    pub max_results: usize,
    pub retrieval_mode: RetrievalMode,

    // === Pagination (SHO-69) ===
    /// Offset for pagination (skip first N results)
    pub offset: usize,
}

/// Paginated search results with metadata for "load more" patterns (SHO-69)
#[derive(Debug, Clone)]
pub struct PaginatedResults<T> {
    /// The results for this page
    pub results: Vec<T>,
    /// Whether there are more results beyond this page
    pub has_more: bool,
    /// Total count of matching results (if available, may be expensive to compute)
    pub total_count: Option<usize>,
    /// The offset used for this query
    pub offset: usize,
    /// The limit used for this query
    pub limit: usize,
}

impl<T> PaginatedResults<T> {
    /// Create a new paginated result from a full result set
    pub fn from_results(all_results: Vec<T>, offset: usize, limit: usize) -> Self {
        let total_count = all_results.len();
        let end = (offset + limit).min(total_count);
        let results: Vec<T> = all_results.into_iter().skip(offset).take(limit).collect();
        let has_more = end < total_count;

        Self {
            results,
            has_more,
            total_count: Some(total_count),
            offset,
            limit,
        }
    }

    /// Create a paginated result when total count is unknown
    /// Uses limit+1 trick: request limit+1, return limit, has_more if got limit+1
    pub fn from_limited_results(mut results: Vec<T>, limit: usize, offset: usize) -> Self {
        let has_more = results.len() > limit;
        if has_more {
            results.pop(); // Remove the extra result we fetched
        }

        Self {
            results,
            has_more,
            total_count: None,
            offset,
            limit,
        }
    }
}

impl Default for Query {
    fn default() -> Self {
        Self {
            user_id: None,
            query_text: None,
            query_embedding: None,
            time_range: None,
            experience_types: None,
            importance_threshold: None,
            robot_id: None,
            mission_id: None,
            geo_filter: None,
            action_type: None,
            reward_range: None,
            outcome_type: None,
            failures_only: false,
            anomalies_only: false,
            severity: None,
            tags: None,
            pattern_id: None,
            terrain_type: None,
            confidence_range: None,
            prospective_signals: None,
            episode_id: None,
            max_results: DEFAULT_MAX_RESULTS,
            retrieval_mode: RetrievalMode::Hybrid,
            offset: 0,
        }
    }
}

impl Query {
    /// Check if a memory matches all query filters
    ///
    /// This is the SINGLE source of truth for filtering.
    /// All memory tiers (working, session, long-term) should use this
    /// instead of implementing their own filter logic.
    ///
    /// # Arguments
    /// * `memory` - The memory to check
    ///
    /// # Returns
    /// * `true` if memory passes all filters, `false` otherwise
    pub fn matches(&self, memory: &Memory) -> bool {
        // Skip soft-deleted (forgotten) memories
        if memory.is_forgotten() {
            return false;
        }

        // Importance threshold
        if let Some(threshold) = self.importance_threshold {
            if memory.importance() < threshold {
                return false;
            }
        }

        // Experience type filter
        // By default, exclude Intention type from normal queries (prospective memory)
        // This makes reminders invisible to regular recall, surfacing only via dedicated APIs
        if let Some(types) = &self.experience_types {
            // Explicit filter: only include specified types
            if !types.iter().any(|t| {
                std::mem::discriminant(&memory.experience.experience_type)
                    == std::mem::discriminant(t)
            }) {
                return false;
            }
        } else {
            // Default filter: exclude Intention (prospective memory)
            if matches!(memory.experience.experience_type, ExperienceType::Intention) {
                return false;
            }
        }

        // Time range filter
        if let Some((start, end)) = &self.time_range {
            if memory.created_at < *start || memory.created_at > *end {
                return false;
            }
        }

        // === Robotics Filters ===

        // Robot ID filter
        if let Some(robot_id) = &self.robot_id {
            if memory.experience.robot_id.as_ref() != Some(robot_id) {
                return false;
            }
        }

        // Mission ID filter
        if let Some(mission_id) = &self.mission_id {
            if memory.experience.mission_id.as_ref() != Some(mission_id) {
                return false;
            }
        }

        // Geo filter (spatial)
        // geo_location is [lat, lon, alt] array
        if let Some(geo_filter) = &self.geo_filter {
            if let Some(geo) = &memory.experience.geo_location {
                let lat = geo[0];
                let lon = geo[1];
                if !geo_filter.contains(lat, lon) {
                    return false;
                }
            } else {
                // No geo_location on memory, and we have a geo filter = no match
                return false;
            }
        }

        // Action type filter
        if let Some(action_type) = &self.action_type {
            if memory.experience.action_type.as_ref() != Some(action_type) {
                return false;
            }
        }

        // Reward range filter
        if let Some((min_reward, max_reward)) = &self.reward_range {
            if let Some(reward) = memory.experience.reward {
                if reward < *min_reward || reward > *max_reward {
                    return false;
                }
            } else {
                // No reward on memory = no match
                return false;
            }
        }

        // === Decision & Learning Filters ===

        // Outcome type filter
        if let Some(outcome_type) = &self.outcome_type {
            if memory.experience.outcome_type.as_ref() != Some(outcome_type) {
                return false;
            }
        }

        // Failures only filter
        if self.failures_only {
            let is_failure = memory
                .experience
                .outcome_type
                .as_ref()
                .map(|o| o == "failure" || o == "failed" || o == "error")
                .unwrap_or(false);
            if !is_failure {
                return false;
            }
        }

        // Anomalies only filter
        if self.anomalies_only {
            if !memory.experience.is_anomaly {
                return false;
            }
        }

        // Severity filter
        if let Some(severity) = &self.severity {
            if memory.experience.severity.as_ref() != Some(severity) {
                return false;
            }
        }

        // Tags filter (any match)
        if let Some(query_tags) = &self.tags {
            let memory_tags = &memory.experience.tags;
            if memory_tags.is_empty() || !query_tags.iter().any(|qt| memory_tags.contains(qt)) {
                return false;
            }
        }

        // Pattern ID filter
        if let Some(pattern_id) = &self.pattern_id {
            if memory.experience.pattern_id.as_ref() != Some(pattern_id) {
                return false;
            }
        }

        // Terrain type filter
        if let Some(terrain_type) = &self.terrain_type {
            if memory.experience.terrain_type.as_ref() != Some(terrain_type) {
                return false;
            }
        }

        // Confidence range filter
        if let Some((min_conf, max_conf)) = &self.confidence_range {
            if let Some(confidence) = memory.experience.confidence {
                if confidence < *min_conf || confidence > *max_conf {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Create a builder for Query
    pub fn builder() -> QueryBuilder {
        QueryBuilder::default()
    }
}

/// Builder for Query to make construction cleaner
#[derive(Default)]
pub struct QueryBuilder {
    query: Query,
}

impl QueryBuilder {
    /// Set user ID for per-user temporal fact lookup and personalized retrieval
    pub fn user_id(mut self, id: impl Into<String>) -> Self {
        self.query.user_id = Some(id.into());
        self
    }

    pub fn query_text(mut self, text: impl Into<String>) -> Self {
        self.query.query_text = Some(text.into());
        self
    }

    pub fn importance_threshold(mut self, threshold: f32) -> Self {
        self.query.importance_threshold = Some(threshold);
        self
    }

    pub fn experience_types(mut self, types: Vec<ExperienceType>) -> Self {
        self.query.experience_types = Some(types);
        self
    }

    pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.query.time_range = Some((start, end));
        self
    }

    pub fn max_results(mut self, max: usize) -> Self {
        self.query.max_results = max;
        self
    }

    pub fn robot_id(mut self, id: impl Into<String>) -> Self {
        self.query.robot_id = Some(id.into());
        self
    }

    pub fn mission_id(mut self, id: impl Into<String>) -> Self {
        self.query.mission_id = Some(id.into());
        self
    }

    pub fn failures_only(mut self, only: bool) -> Self {
        self.query.failures_only = only;
        self
    }

    pub fn anomalies_only(mut self, only: bool) -> Self {
        self.query.anomalies_only = only;
        self
    }

    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.query.tags = Some(tags);
        self
    }

    pub fn retrieval_mode(mut self, mode: RetrievalMode) -> Self {
        self.query.retrieval_mode = mode;
        self
    }

    /// Set offset for pagination (skip first N results)
    pub fn offset(mut self, offset: usize) -> Self {
        self.query.offset = offset;
        self
    }

    /// Set prospective memory signals (future intentions that boost related memories)
    pub fn prospective_signals(mut self, signals: Vec<String>) -> Self {
        self.query.prospective_signals = Some(signals);
        self
    }

    /// Set episode ID for context-aware retrieval
    /// Memories from the same episode get a coherence boost (prevents episode bleeding)
    pub fn episode_id(mut self, id: impl Into<String>) -> Self {
        self.query.episode_id = Some(id.into());
        self
    }

    pub fn build(self) -> Query {
        self.query
    }
}

/// Retrieval modes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalMode {
    Similarity,  // Vector similarity search
    Temporal,    // Time-based retrieval
    Causal,      // Cause-effect chains
    Associative, // Related memories
    Hybrid,      // Combination of methods
    // === Robotics-Specific Modes ===
    Spatial,       // Geo-location based retrieval
    Mission,       // Mission context retrieval
    ActionOutcome, // Reward-based learning retrieval
}

/// Criteria for forgetting memories
#[derive(Debug, Clone)]
pub enum ForgetCriteria {
    /// Delete a single memory by its ID
    ById(MemoryId),
    OlderThan(u32),     // Days
    LowImportance(f32), // Threshold
    Pattern(String),    // Regex pattern
    /// Delete memories matching ANY of these tags
    ByTags(Vec<String>),
    /// Delete memories within a date range (inclusive)
    ByDateRange {
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    },
    /// Delete memories of a specific type
    ByType(ExperienceType),
    /// Delete ALL memories for a user (GDPR compliance - right to erasure)
    All,
}

/// Working memory - fast access, limited size
///
/// Now uses Arc<Memory> for zero-copy shared ownership.
/// Performance improvement: ~10x fewer allocations.
pub struct WorkingMemory {
    memories: HashMap<MemoryId, SharedMemory>,
    capacity: usize,
    access_order: Vec<MemoryId>,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            memories: HashMap::new(),
            capacity,
            access_order: Vec::new(),
        }
    }

    /// Add memory (convenience wrapper - use add_shared for zero-copy)
    #[allow(unused)] // Public API convenience method
    pub fn add(&mut self, memory: Memory) -> anyhow::Result<()> {
        self.add_shared(Arc::new(memory))
    }

    /// Add shared memory (zero-copy)
    pub fn add_shared(&mut self, memory: SharedMemory) -> anyhow::Result<()> {
        // Evict LRU if at capacity
        if self.memories.len() >= self.capacity {
            if let Some(oldest) = self.access_order.first().cloned() {
                self.memories.remove(&oldest);
                self.access_order.remove(0);
            }
        }

        let id = memory.id.clone();
        self.memories.insert(id.clone(), memory);
        self.access_order.push(id);
        Ok(())
    }

    /// Search memories (returns Arc<Memory> for zero-copy)
    ///
    /// Uses Query::matches() for filtering - SINGLE source of truth for all filter logic
    pub fn search(&self, query: &Query, limit: usize) -> anyhow::Result<Vec<SharedMemory>> {
        let mut results: Vec<SharedMemory> = self
            .memories
            .values()
            .filter(|m| query.matches(m))
            .cloned() // Arc::clone is cheap (just increments ref count)
            .collect();

        // Sort by importance and recency
        results.sort_by(|a, b| {
            b.importance()
                .partial_cmp(&a.importance())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.last_accessed().cmp(&a.last_accessed()))
        });

        results.truncate(limit);
        Ok(results)
    }

    pub fn size(&self) -> usize {
        self.memories.len()
    }

    /// Get the number of memories (alias for size())
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Check if the working memory is empty
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Clear all memories from working memory
    pub fn clear(&mut self) {
        self.memories.clear();
        self.access_order.clear();
    }

    pub fn contains(&self, id: &MemoryId) -> bool {
        self.memories.contains_key(id)
    }

    /// Get memory by ID (zero-copy Arc clone)
    pub fn get(&self, id: &MemoryId) -> Option<SharedMemory> {
        self.memories.get(id).map(Arc::clone)
    }

    pub fn update_access(&mut self, id: &MemoryId) -> anyhow::Result<()> {
        if let Some(shared_memory) = self.memories.get(id) {
            // ZERO-COPY: Update metadata through Arc without cloning Experience/embeddings
            shared_memory.update_access();

            // Update access order for LRU tracking
            if let Some(pos) = self.access_order.iter().position(|x| x == id) {
                self.access_order.remove(pos);
                self.access_order.push(id.clone());
            }
        }
        Ok(())
    }

    /// Get least recently used memories (zero-copy with Arc)
    pub fn get_lru(&self, count: usize) -> anyhow::Result<Vec<SharedMemory>> {
        let mut result = Vec::new();
        for id in self.access_order.iter().take(count) {
            if let Some(memory) = self.memories.get(id) {
                result.push(Arc::clone(memory)); // Cheap: just ref count increment
            }
        }
        Ok(result)
    }

    pub fn remove(&mut self, id: &MemoryId) -> anyhow::Result<()> {
        self.memories.remove(id);
        self.access_order.retain(|x| x != id);
        Ok(())
    }

    pub fn remove_older_than(&mut self, cutoff: DateTime<Utc>) -> anyhow::Result<usize> {
        let to_remove: Vec<MemoryId> = self
            .memories
            .iter()
            .filter(|(_, m)| m.created_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(count)
    }

    pub fn remove_below_importance(&mut self, threshold: f32) -> anyhow::Result<usize> {
        let to_remove: Vec<MemoryId> = self
            .memories
            .iter()
            .filter(|(_, m)| m.importance() < threshold)
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(count)
    }

    pub fn remove_matching(&mut self, regex: &regex::Regex) -> anyhow::Result<usize> {
        let to_remove: Vec<MemoryId> = self
            .memories
            .iter()
            .filter(|(_, m)| regex.is_match(&m.experience.content))
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(count)
    }

    /// Get all memories (for semantic search across all tiers)
    pub fn all_memories(&self) -> Vec<SharedMemory> {
        self.memories.values().cloned().collect()
    }
}

/// Entry in session memory - tracks size at insertion time
struct SessionMemoryEntry {
    memory: SharedMemory,
    /// Size in bytes when inserted (used for accurate size tracking)
    insertion_size: usize,
}

/// Session memory - medium-term storage
///
/// Now uses Arc<Memory> for zero-copy shared ownership.
/// Tracks insertion size separately to avoid overflow when memory is modified after insertion.
pub struct SessionMemory {
    memories: HashMap<MemoryId, SessionMemoryEntry>,
    max_size_mb: usize,
    current_size_bytes: usize,
}

impl SessionMemory {
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            memories: HashMap::new(),
            max_size_mb,
            current_size_bytes: 0,
        }
    }

    /// Add memory (convenience wrapper - use add_shared for zero-copy)
    pub fn add(&mut self, memory: Memory) -> anyhow::Result<()> {
        self.add_shared(Arc::new(memory))
    }

    /// Add shared memory (zero-copy)
    pub fn add_shared(&mut self, memory: SharedMemory) -> anyhow::Result<()> {
        let memory_size =
            bincode::serde::encode_to_vec(&*memory, bincode::config::standard())?.len();

        // Check if adding would exceed limit
        if self.current_size_bytes + memory_size > self.max_size_mb * 1024 * 1024 {
            // Evict lowest importance memories until there's space
            self.evict_to_make_space(memory_size)?;
        }

        let id = memory.id.clone();
        self.memories.insert(
            id,
            SessionMemoryEntry {
                memory,
                insertion_size: memory_size,
            },
        );
        self.current_size_bytes += memory_size;
        Ok(())
    }

    fn evict_to_make_space(&mut self, needed_bytes: usize) -> anyhow::Result<()> {
        let mut sorted: Vec<(MemoryId, f32)> = self
            .memories
            .iter()
            .map(|(id, entry)| (id.clone(), entry.memory.importance()))
            .collect();

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (id, _) in sorted {
            if self.current_size_bytes + needed_bytes <= self.max_size_mb * 1024 * 1024 {
                break;
            }
            if let Some(entry) = self.memories.remove(&id) {
                // Use stored insertion_size for accurate tracking (not re-serialized size)
                self.current_size_bytes =
                    self.current_size_bytes.saturating_sub(entry.insertion_size);
            }
        }
        Ok(())
    }

    /// Search memories (returns Arc<Memory> for zero-copy)
    ///
    /// Uses Query::matches() for filtering - SINGLE source of truth for all filter logic
    pub fn search(&self, query: &Query, limit: usize) -> anyhow::Result<Vec<SharedMemory>> {
        let mut results: Vec<SharedMemory> = self
            .memories
            .values()
            .map(|entry| &entry.memory)
            .filter(|m| query.matches(m))
            .cloned() // Arc::clone is cheap
            .collect();

        results.sort_by(|a, b| {
            b.importance()
                .partial_cmp(&a.importance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    pub fn size_mb(&self) -> usize {
        self.current_size_bytes / (1024 * 1024)
    }

    /// Get the number of memories
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Check if the session memory is empty
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Clear all memories from session memory
    pub fn clear(&mut self) {
        self.memories.clear();
        self.current_size_bytes = 0;
    }

    pub fn contains(&self, id: &MemoryId) -> bool {
        self.memories.contains_key(id)
    }

    /// Get memory by ID (zero-copy Arc clone)
    pub fn get(&self, id: &MemoryId) -> Option<SharedMemory> {
        self.memories.get(id).map(|entry| Arc::clone(&entry.memory))
    }

    pub fn update_access(&mut self, id: &MemoryId) -> anyhow::Result<()> {
        if let Some(entry) = self.memories.get(id) {
            // ZERO-COPY: Update metadata through Arc without cloning Experience/embeddings
            entry.memory.update_access();
        }
        Ok(())
    }

    /// Get important memories (zero-copy with Arc)
    pub fn get_important(&self, threshold: f32) -> anyhow::Result<Vec<SharedMemory>> {
        Ok(self
            .memories
            .values()
            .map(|entry| &entry.memory)
            .filter(|m| m.importance() >= threshold)
            .cloned() // Arc::clone is cheap
            .collect())
    }

    pub fn remove(&mut self, id: &MemoryId) -> anyhow::Result<()> {
        if let Some(entry) = self.memories.remove(id) {
            // Use stored insertion_size for accurate tracking (avoids overflow)
            self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.insertion_size);
        }
        Ok(())
    }

    pub fn remove_older_than(&mut self, cutoff: DateTime<Utc>) -> anyhow::Result<usize> {
        let to_remove: Vec<MemoryId> = self
            .memories
            .iter()
            .filter(|(_, entry)| entry.memory.created_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(count)
    }

    pub fn remove_below_importance(&mut self, threshold: f32) -> anyhow::Result<usize> {
        let to_remove: Vec<MemoryId> = self
            .memories
            .iter()
            .filter(|(_, entry)| entry.memory.importance() < threshold)
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(count)
    }

    pub fn remove_matching(&mut self, regex: &regex::Regex) -> anyhow::Result<usize> {
        let to_remove: Vec<MemoryId> = self
            .memories
            .iter()
            .filter(|(_, entry)| regex.is_match(&entry.memory.experience.content))
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(count)
    }

    /// Iterate over all memories for access statistics
    pub fn iter(&self) -> impl Iterator<Item = (&MemoryId, &SharedMemory)> {
        self.memories.iter().map(|(id, entry)| (id, &entry.memory))
    }

    /// Get all memories (for semantic search across all tiers)
    pub fn all_memories(&self) -> Vec<SharedMemory> {
        self.memories
            .values()
            .map(|entry| entry.memory.clone())
            .collect()
    }
}

/// Memory statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub working_memory_count: usize,
    pub session_memory_count: usize,
    pub long_term_memory_count: usize,
    pub vector_index_count: usize,
    pub compressed_count: usize,
    pub promotions_to_session: usize,
    pub promotions_to_longterm: usize,
    pub total_retrievals: usize,
    pub average_importance: f32,
    /// Knowledge graph entity count
    #[serde(default)]
    pub graph_nodes: usize,
    /// Knowledge graph relationship count
    #[serde(default)]
    pub graph_edges: usize,
}

/// Report from index integrity verification
///
/// Used to diagnose vector index gaps where memories are stored in RocksDB
/// but missing from the vector index (orphaned memories).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexIntegrityReport {
    /// Total memories in RocksDB storage
    pub total_storage: usize,
    /// Total memories in vector index
    pub total_indexed: usize,
    /// Number of orphaned memories (storage - indexed)
    pub orphaned_count: usize,
    /// First 100 orphaned memory IDs for debugging
    pub orphaned_ids: Vec<MemoryId>,
    /// Whether the index is healthy (no orphans)
    pub is_healthy: bool,
    /// Compat alias for `is_healthy` (Cloudflare Worker clients use `healthy`)
    #[serde(default, skip_deserializing)]
    pub healthy: bool,
}

/// Retrieval statistics for SHO-26 associative retrieval (debugging/observability)
///
/// Returned with recall responses to expose the hybrid scoring internals.
/// Helps users understand why certain memories were retrieved and tune parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalStats {
    /// Mode used for this retrieval (semantic, associative, hybrid)
    pub mode: String,

    /// Number of memories retrieved via semantic similarity
    pub semantic_candidates: usize,

    /// Number of memories retrieved via graph traversal
    pub graph_candidates: usize,

    /// Graph density (edges per memory) - determines graph weight
    pub graph_density: f32,

    /// Actual graph weight used (density-dependent: 0.1 to 0.5)
    pub graph_weight: f32,

    /// Actual semantic weight used (1.0 - graph_weight - linguistic_weight)
    pub semantic_weight: f32,

    /// Linguistic weight used (fixed at 0.15)
    pub linguistic_weight: f32,

    /// Number of graph hops performed in spreading activation
    pub graph_hops: usize,

    /// Number of unique entities activated during graph traversal
    pub entities_activated: usize,

    /// Average salience boost applied to initial activations (0.0 if no entities)
    /// Tracks ACT-R inspired salience weighting effectiveness
    pub avg_salience_boost: f32,

    /// Total time spent on retrieval (microseconds)
    pub retrieval_time_us: u64,

    /// Time spent on embedding generation (microseconds)
    pub embedding_time_us: u64,

    /// Time spent on graph traversal (microseconds)
    pub graph_time_us: u64,

    /// Edge UUIDs traversed during spreading activation (for Hebbian strengthening)
    /// Not serialized - internal use only for wiring strengthening calls
    #[serde(skip)]
    pub traversed_edges: Vec<uuid::Uuid>,
}

// =============================================================================
// PROSPECTIVE MEMORY - Future intentions and reminders (SHO-116)
// =============================================================================

/// Unique identifier for prospective tasks (reminders)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProspectiveTaskId(pub Uuid);

impl ProspectiveTaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ProspectiveTaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ProspectiveTaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Trigger condition for prospective tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProspectiveTrigger {
    /// Trigger at a specific time
    AtTime { at: DateTime<Utc> },
    /// Trigger after a duration from creation
    AfterDuration {
        seconds: u64,
        #[serde(default = "Utc::now")]
        from: DateTime<Utc>,
    },
    /// Trigger when context matches keywords or semantic similarity
    OnContext {
        keywords: Vec<String>,
        #[serde(default = "default_context_threshold")]
        threshold: f32,
    },
}

fn default_context_threshold() -> f32 {
    0.7
}

impl ProspectiveTrigger {
    /// Check if a time-based trigger is due
    pub fn is_due(&self) -> bool {
        let now = Utc::now();
        match self {
            ProspectiveTrigger::AtTime { at } => now >= *at,
            ProspectiveTrigger::AfterDuration { seconds, from } => {
                let due_at = *from + chrono::Duration::seconds(*seconds as i64);
                now >= due_at
            }
            ProspectiveTrigger::OnContext { .. } => false, // Context triggers are checked separately
        }
    }

    /// Get the due time for time-based triggers
    pub fn due_at(&self) -> Option<DateTime<Utc>> {
        match self {
            ProspectiveTrigger::AtTime { at } => Some(*at),
            ProspectiveTrigger::AfterDuration { seconds, from } => {
                Some(*from + chrono::Duration::seconds(*seconds as i64))
            }
            ProspectiveTrigger::OnContext { .. } => None,
        }
    }

    /// Check if this is a context-based trigger
    pub fn is_context_trigger(&self) -> bool {
        matches!(self, ProspectiveTrigger::OnContext { .. })
    }

    /// Check if context matches this trigger's keywords
    pub fn matches_context(&self, context: &str) -> bool {
        match self {
            ProspectiveTrigger::OnContext { keywords, .. } => {
                let context_lower = context.to_lowercase();
                keywords
                    .iter()
                    .any(|kw| context_lower.contains(&kw.to_lowercase()))
            }
            _ => false,
        }
    }
}

/// Status of a prospective task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProspectiveTaskStatus {
    /// Waiting for trigger condition
    Pending,
    /// Trigger condition met, shown to user
    Triggered,
    /// User acknowledged/dismissed
    Dismissed,
    /// Task expired without being triggered (optional cleanup)
    Expired,
}

impl Default for ProspectiveTaskStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// A prospective memory task (reminder/intention)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectiveTask {
    /// Unique identifier
    pub id: ProspectiveTaskId,

    /// User who created this task
    pub user_id: String,

    /// What to remember/remind about
    pub content: String,

    /// When/how to trigger this reminder
    pub trigger: ProspectiveTrigger,

    /// Current status
    #[serde(default)]
    pub status: ProspectiveTaskStatus,

    /// When the task was created
    pub created_at: DateTime<Utc>,

    /// When the trigger condition was met (if triggered)
    pub triggered_at: Option<DateTime<Utc>>,

    /// When the user dismissed the reminder
    pub dismissed_at: Option<DateTime<Utc>>,

    /// Optional tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Optional priority (1-5, higher = more important)
    #[serde(default = "default_priority")]
    pub priority: u8,

    /// Vector embedding for semantic search and context matching (MiniLM-L6-v2, 384 dimensions)
    /// Used for semantic similarity matching on context triggers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,

    /// Related memory IDs for bidirectional linking
    #[serde(default)]
    pub related_memory_ids: Vec<MemoryId>,
}

fn default_priority() -> u8 {
    3
}

impl ProspectiveTask {
    /// Create a new prospective task
    pub fn new(user_id: String, content: String, trigger: ProspectiveTrigger) -> Self {
        Self {
            id: ProspectiveTaskId::new(),
            user_id,
            content,
            trigger,
            status: ProspectiveTaskStatus::Pending,
            created_at: Utc::now(),
            triggered_at: None,
            dismissed_at: None,
            tags: Vec::new(),
            priority: default_priority(),
            embedding: None,
            related_memory_ids: Vec::new(),
        }
    }

    /// Check if this task is due (for time-based triggers)
    pub fn is_due(&self) -> bool {
        self.status == ProspectiveTaskStatus::Pending && self.trigger.is_due()
    }

    /// Mark as triggered
    pub fn mark_triggered(&mut self) {
        self.status = ProspectiveTaskStatus::Triggered;
        self.triggered_at = Some(Utc::now());
    }

    /// Mark as dismissed
    pub fn mark_dismissed(&mut self) {
        self.status = ProspectiveTaskStatus::Dismissed;
        self.dismissed_at = Some(Utc::now());
    }

    /// Get overdue duration in seconds (if time-based and overdue)
    pub fn overdue_seconds(&self) -> Option<i64> {
        if let Some(due_at) = self.trigger.due_at() {
            let now = Utc::now();
            if now > due_at {
                return Some((now - due_at).num_seconds());
            }
        }
        None
    }
}

// =============================================================================
// TODO/GTD SYSTEM TYPES (Linear-style)
// =============================================================================

/// Unique identifier for todos
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TodoId(pub Uuid);

impl TodoId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get short ID format (SHO-xxxx)
    pub fn short(&self) -> String {
        format!("SHO-{}", &self.0.to_string()[..4])
    }
}

impl Default for TodoId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TodoId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short())
    }
}

/// Unique identifier for projects
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProjectId(pub Uuid);

impl ProjectId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ProjectId {
    fn default() -> Self {
        Self::new()
    }
}

/// Todo status (Linear-style workflow)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// ◌ Not started, someday/maybe
    Backlog,
    /// ○ Ready to do
    #[default]
    Todo,
    /// ◐ Actively working on
    InProgress,
    /// ⊘ Waiting for someone/something
    Blocked,
    /// ● Completed
    Done,
    /// ⊗ Won't do
    Cancelled,
}

impl TodoStatus {
    /// Get the status icon (Linear-style)
    pub fn icon(&self) -> &'static str {
        match self {
            TodoStatus::Backlog => "◌",
            TodoStatus::Todo => "○",
            TodoStatus::InProgress => "◐",
            TodoStatus::Blocked => "⊘",
            TodoStatus::Done => "●",
            TodoStatus::Cancelled => "⊗",
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "backlog" | "someday" | "maybe" => Some(TodoStatus::Backlog),
            "todo" | "next" | "ready" => Some(TodoStatus::Todo),
            "in_progress" | "inprogress" | "active" | "doing" => Some(TodoStatus::InProgress),
            "blocked" | "waiting" | "waiting_for" => Some(TodoStatus::Blocked),
            "done" | "completed" | "complete" => Some(TodoStatus::Done),
            "cancelled" | "canceled" | "wont_do" => Some(TodoStatus::Cancelled),
            _ => None,
        }
    }
}

/// Todo priority (Linear-style)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TodoPriority {
    /// !!! Urgent (P1)
    Urgent,
    /// !! High (P2)
    High,
    /// ! Medium (P3) - default
    #[default]
    Medium,
    /// (none) Low (P4)
    Low,
    /// No priority set
    None,
}

impl TodoPriority {
    /// Get priority indicator (Linear-style)
    pub fn indicator(&self) -> &'static str {
        match self {
            TodoPriority::Urgent => "!!!",
            TodoPriority::High => "!!",
            TodoPriority::Medium => "!",
            TodoPriority::Low => "",
            TodoPriority::None => "",
        }
    }

    /// Get numeric value (1=urgent, 4=low, 5=none)
    pub fn value(&self) -> u8 {
        match self {
            TodoPriority::Urgent => 1,
            TodoPriority::High => 2,
            TodoPriority::Medium => 3,
            TodoPriority::Low => 4,
            TodoPriority::None => 5,
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "urgent" | "p1" | "1" | "!!!" => Some(TodoPriority::Urgent),
            "high" | "p2" | "2" | "!!" => Some(TodoPriority::High),
            "medium" | "p3" | "3" | "!" => Some(TodoPriority::Medium),
            "low" | "p4" | "4" => Some(TodoPriority::Low),
            "none" | "no_priority" | "" => Some(TodoPriority::None),
            _ => None,
        }
    }
}

/// Recurrence pattern for repeating todos
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Recurrence {
    /// Every day
    Daily,
    /// Specific days of week (0=Sun, 6=Sat)
    Weekly { days: Vec<u8> },
    /// Specific day of month (1-31)
    Monthly { day: u8 },
    /// Every N days
    EveryNDays { n: u32 },
}

impl Recurrence {
    /// Calculate the next due date from a given date
    pub fn next_occurrence(&self, from: DateTime<Utc>) -> DateTime<Utc> {
        use chrono::{Datelike, Duration};

        match self {
            Recurrence::Daily => from + Duration::days(1),
            Recurrence::Weekly { days } => {
                if days.is_empty() {
                    return from + Duration::weeks(1);
                }
                let current_dow = from.weekday().num_days_from_sunday() as u8;
                // Find next day in the list
                let next_day = days
                    .iter()
                    .find(|&&d| d > current_dow)
                    .copied()
                    .unwrap_or(days[0]);

                let days_until = if next_day > current_dow {
                    (next_day - current_dow) as i64
                } else {
                    (7 - current_dow + next_day) as i64
                };
                from + Duration::days(days_until)
            }
            Recurrence::Monthly { day } => {
                let target_day = (*day).min(28) as u32; // Cap at 28 to avoid month overflow
                let mut next = from;
                // Move to next month if we're past the target day
                if from.day() >= target_day {
                    next = from + Duration::days(32); // Jump to next month
                }
                // Set to target day
                next.with_day(target_day).unwrap_or(next)
            }
            Recurrence::EveryNDays { n } => from + Duration::days(*n as i64),
        }
    }
}

/// Unique identifier for todo comments
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TodoCommentId(pub Uuid);

impl TodoCommentId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TodoCommentId {
    fn default() -> Self {
        Self::new()
    }
}

/// A comment/activity on a todo item
/// Used to track progress, notes, and actions taken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoComment {
    /// Unique identifier
    pub id: TodoCommentId,

    /// The todo this comment belongs to
    pub todo_id: TodoId,

    /// Author of the comment (user_id or system)
    pub author: String,

    /// Comment content (supports markdown)
    pub content: String,

    /// Type of activity/comment
    #[serde(default)]
    pub comment_type: TodoCommentType,

    /// When the comment was created
    pub created_at: DateTime<Utc>,

    /// When the comment was last edited (if ever)
    pub updated_at: Option<DateTime<Utc>>,
}

impl TodoComment {
    /// Create a new comment
    pub fn new(todo_id: TodoId, author: String, content: String) -> Self {
        Self {
            id: TodoCommentId::new(),
            todo_id,
            author,
            content,
            comment_type: TodoCommentType::Comment,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    /// Create a system activity comment
    pub fn system_activity(todo_id: TodoId, content: String) -> Self {
        Self {
            id: TodoCommentId::new(),
            todo_id,
            author: "system".to_string(),
            content,
            comment_type: TodoCommentType::Activity,
            created_at: Utc::now(),
            updated_at: None,
        }
    }
}

/// Type of todo comment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TodoCommentType {
    /// User comment
    #[default]
    Comment,
    /// System-generated activity (status change, assignment, etc.)
    Activity,
    /// Progress update
    Progress,
    /// Resolution/fix description
    Resolution,
}

/// A GTD-style todo item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Todo {
    /// Unique identifier
    pub id: TodoId,

    /// Sequential number within project (BOLT-1, MEM-2, etc.)
    #[serde(default)]
    pub seq_num: u32,

    /// Cached project prefix for display (e.g., "BOLT", "MEM")
    /// Set when todo is created with a project
    #[serde(default)]
    pub project_prefix: Option<String>,

    /// Compat alias for `project_prefix` (Cloudflare Worker clients use `project`)
    #[serde(default, skip_deserializing)]
    pub project: Option<String>,

    /// User who owns this todo
    pub user_id: String,

    /// What needs to be done
    pub content: String,

    /// Current status (Linear-style workflow)
    #[serde(default)]
    pub status: TodoStatus,

    /// Priority level
    #[serde(default)]
    pub priority: TodoPriority,

    /// Associated project (optional)
    pub project_id: Option<ProjectId>,

    /// Parent todo for subtasks
    pub parent_id: Option<TodoId>,

    /// GTD contexts (@computer, @phone, @errands, etc.)
    #[serde(default)]
    pub contexts: Vec<String>,

    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// External ID for linking to external systems (e.g., "todoist:123", "linear:SHO-39")
    /// Used for two-way sync with external todo/task management systems
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,

    /// Due date/time (optional)
    pub due_date: Option<DateTime<Utc>>,

    /// Recurrence pattern (optional)
    pub recurrence: Option<Recurrence>,

    /// Who/what this is blocked on (when status=Blocked)
    pub blocked_on: Option<String>,

    /// Additional notes
    pub notes: Option<String>,

    /// When created
    pub created_at: DateTime<Utc>,

    /// When last modified
    pub updated_at: DateTime<Utc>,

    /// When completed (if Done)
    pub completed_at: Option<DateTime<Utc>>,

    /// Manual sort order within status group (lower = higher in list)
    #[serde(default)]
    pub sort_order: i32,

    /// Comments and activity history for this todo
    #[serde(default)]
    pub comments: Vec<TodoComment>,

    /// Vector embedding for semantic search (MiniLM-L6-v2, 384 dimensions)
    /// Computed from content + notes + tags for similarity matching
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,

    /// Related memory IDs for bidirectional linking
    /// Memories that are semantically or explicitly linked to this todo
    #[serde(default)]
    pub related_memory_ids: Vec<MemoryId>,
}

impl Todo {
    /// Create a new todo
    pub fn new(user_id: String, content: String) -> Self {
        let now = Utc::now();
        Self {
            id: TodoId::new(),
            seq_num: 0,           // Will be assigned by TodoStore on creation
            project_prefix: None, // Will be set by TodoStore based on project
            project: None,        // Synced from project_prefix
            user_id,
            content,
            status: TodoStatus::Todo,
            priority: TodoPriority::Medium,
            project_id: None,
            parent_id: None,
            contexts: Vec::new(),
            tags: Vec::new(),
            external_id: None,
            due_date: None,
            recurrence: None,
            blocked_on: None,
            notes: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
            sort_order: 0,
            comments: Vec::new(),
            embedding: None,
            related_memory_ids: Vec::new(),
        }
    }

    /// Sync compat alias fields from their canonical counterparts.
    /// Call after construction or deserialization to keep aliases in sync.
    pub fn sync_compat_fields(&mut self) {
        self.project = self.project_prefix.clone();
    }

    /// Get the user-facing short ID (BOLT-1, MEM-2, SHO-3, etc.)
    /// Uses project prefix if available, otherwise "SHO" for standalone todos
    pub fn short_id(&self) -> String {
        if self.seq_num > 0 {
            let prefix = self.project_prefix.as_deref().unwrap_or("SHO");
            format!("{}-{}", prefix, self.seq_num)
        } else {
            // Fallback for legacy todos without seq_num
            self.id.short()
        }
    }

    /// Check if todo is overdue
    pub fn is_overdue(&self) -> bool {
        if let Some(due) = &self.due_date {
            Utc::now() > *due
                && self.status != TodoStatus::Done
                && self.status != TodoStatus::Cancelled
        } else {
            false
        }
    }

    /// Get overdue duration in seconds
    pub fn overdue_seconds(&self) -> Option<i64> {
        if let Some(due) = &self.due_date {
            let now = Utc::now();
            if now > *due && self.status != TodoStatus::Done && self.status != TodoStatus::Cancelled
            {
                return Some((now - *due).num_seconds());
            }
        }
        None
    }

    /// Check if due today
    pub fn is_due_today(&self) -> bool {
        if let Some(due) = &self.due_date {
            let now = Utc::now();
            due.date_naive() == now.date_naive()
        } else {
            false
        }
    }

    /// Mark as completed
    pub fn complete(&mut self) {
        self.status = TodoStatus::Done;
        self.completed_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    /// Create next recurrence if applicable
    pub fn create_next_recurrence(&self) -> Option<Todo> {
        self.recurrence.as_ref().map(|r| {
            let base_date = self.due_date.unwrap_or_else(Utc::now);
            let next_due = r.next_occurrence(base_date);

            let mut next = self.clone();
            next.id = TodoId::new();
            next.status = TodoStatus::Todo;
            next.due_date = Some(next_due);
            next.completed_at = None;
            next.created_at = Utc::now();
            next.updated_at = Utc::now();
            next.comments = Vec::new(); // Fresh comments for new recurrence
            next
        })
    }

    /// Add a comment to this todo
    pub fn add_comment(&mut self, author: String, content: String) -> &TodoComment {
        let comment = TodoComment::new(self.id.clone(), author, content);
        self.comments.push(comment);
        self.updated_at = Utc::now();
        self.comments.last().unwrap()
    }

    /// Add a progress update
    pub fn add_progress(&mut self, author: String, content: String) -> &TodoComment {
        let mut comment = TodoComment::new(self.id.clone(), author, content);
        comment.comment_type = TodoCommentType::Progress;
        self.comments.push(comment);
        self.updated_at = Utc::now();
        self.comments.last().unwrap()
    }

    /// Add a resolution comment
    pub fn add_resolution(&mut self, author: String, content: String) -> &TodoComment {
        let mut comment = TodoComment::new(self.id.clone(), author, content);
        comment.comment_type = TodoCommentType::Resolution;
        self.comments.push(comment);
        self.updated_at = Utc::now();
        self.comments.last().unwrap()
    }

    /// Add a system activity entry
    pub fn add_activity(&mut self, content: String) {
        let comment = TodoComment::system_activity(self.id.clone(), content);
        self.comments.push(comment);
        self.updated_at = Utc::now();
    }

    /// Add a related memory ID (bidirectional link to memory system)
    pub fn add_related_memory(&mut self, memory_id: MemoryId) {
        if !self.related_memory_ids.contains(&memory_id) {
            self.related_memory_ids.push(memory_id);
            self.updated_at = Utc::now();
        }
    }

    /// Remove a related memory ID
    pub fn remove_related_memory(&mut self, memory_id: &MemoryId) {
        self.related_memory_ids.retain(|id| id != memory_id);
        self.updated_at = Utc::now();
    }

    /// Check if this todo is linked to a specific memory
    pub fn has_related_memory(&self, memory_id: &MemoryId) -> bool {
        self.related_memory_ids.contains(memory_id)
    }
}

/// Project status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProjectStatus {
    #[default]
    Active,
    OnHold,
    Completed,
    Archived,
}

/// A project that groups related todos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Unique identifier
    pub id: ProjectId,

    /// User who owns this project
    pub user_id: String,

    /// Project name
    pub name: String,

    /// Short prefix for todo IDs (e.g., "BOLT", "MEM")
    /// If not set, derived from first letters of project name
    #[serde(default)]
    pub prefix: Option<String>,

    /// Optional description
    pub description: Option<String>,

    /// Project status
    #[serde(default)]
    pub status: ProjectStatus,

    /// Optional color (hex)
    pub color: Option<String>,

    /// Parent project ID (for sub-projects)
    #[serde(default)]
    pub parent_id: Option<ProjectId>,

    /// When created
    pub created_at: DateTime<Utc>,

    /// When completed
    pub completed_at: Option<DateTime<Utc>>,

    // =========================================================================
    // CODEBASE INTEGRATION (MEM-30)
    // =========================================================================
    /// Absolute path to the codebase root directory
    #[serde(default)]
    pub codebase_path: Option<String>,

    /// Whether the codebase has been indexed
    #[serde(default)]
    pub codebase_indexed: bool,

    /// When the codebase was last indexed
    #[serde(default)]
    pub codebase_indexed_at: Option<DateTime<Utc>>,

    /// Number of files indexed in the codebase
    #[serde(default)]
    pub codebase_file_count: usize,

    /// Vector embedding for semantic search (MiniLM-L6-v2, 384 dimensions)
    /// Computed from name + description for similarity matching
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,

    /// Related memory IDs for project-level context
    #[serde(default)]
    pub related_memory_ids: Vec<MemoryId>,

    /// Aggregated todo count by status (cached for quick display)
    #[serde(default)]
    pub todo_counts: ProjectTodoCounts,
}

/// Cached todo counts for quick project display
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProjectTodoCounts {
    pub total: usize,
    pub backlog: usize,
    pub todo: usize,
    pub in_progress: usize,
    pub blocked: usize,
    pub done: usize,
}

impl Project {
    /// Create a new project
    pub fn new(user_id: String, name: String) -> Self {
        let prefix = Self::derive_prefix(&name);
        Self {
            id: ProjectId::new(),
            user_id,
            name,
            prefix: Some(prefix),
            description: None,
            status: ProjectStatus::Active,
            color: None,
            parent_id: None,
            created_at: Utc::now(),
            completed_at: None,
            codebase_path: None,
            codebase_indexed: false,
            codebase_indexed_at: None,
            codebase_file_count: 0,
            embedding: None,
            related_memory_ids: Vec::new(),
            todo_counts: ProjectTodoCounts::default(),
        }
    }

    /// Create a new sub-project under a parent
    pub fn new_subproject(user_id: String, name: String, parent_id: ProjectId) -> Self {
        let prefix = Self::derive_prefix(&name);
        Self {
            id: ProjectId::new(),
            user_id,
            name,
            prefix: Some(prefix),
            description: None,
            status: ProjectStatus::Active,
            color: None,
            parent_id: Some(parent_id),
            created_at: Utc::now(),
            completed_at: None,
            codebase_path: None,
            codebase_indexed: false,
            codebase_indexed_at: None,
            codebase_file_count: 0,
            embedding: None,
            related_memory_ids: Vec::new(),
            todo_counts: ProjectTodoCounts::default(),
        }
    }

    /// Derive a short prefix from project name
    /// Examples: "bolt-parser" -> "BOLT", "Shodh-memory" -> "MEM", "My Project" -> "MYP"
    pub fn derive_prefix(name: &str) -> String {
        let name_clean = name.trim().to_uppercase();

        // If name has a hyphen, use first part
        if let Some(first_part) = name_clean.split('-').next() {
            let first_part = first_part.trim();
            if first_part.len() >= 2 {
                // Take up to 4 chars from first part
                return first_part.chars().take(4).collect();
            }
        }

        // If name has spaces, use initials
        let words: Vec<&str> = name_clean.split_whitespace().collect();
        if words.len() > 1 {
            return words
                .iter()
                .filter_map(|w| w.chars().next())
                .take(4)
                .collect();
        }

        // Single word: take first 3-4 chars
        name_clean.chars().take(4).collect()
    }

    /// Get the effective prefix (derived if not explicitly set)
    pub fn effective_prefix(&self) -> String {
        self.prefix
            .clone()
            .unwrap_or_else(|| Self::derive_prefix(&self.name))
    }
}

// =============================================================================
// FILE MEMORY TYPES (MEM-29)
// Codebase integration - learned knowledge about files
// =============================================================================

/// Unique identifier for file memories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FileMemoryId(pub Uuid);

impl FileMemoryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for FileMemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for FileMemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Programming language / file type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FileType {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
    CSharp,
    Cpp,
    C,
    Ruby,
    Markdown,
    Json,
    Yaml,
    Toml,
    Html,
    Css,
    Sql,
    Shell,
    Other(String),
}

impl FileType {
    /// Detect file type from extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => FileType::Rust,
            "ts" | "tsx" => FileType::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => FileType::JavaScript,
            "py" | "pyi" => FileType::Python,
            "go" => FileType::Go,
            "java" => FileType::Java,
            "cs" => FileType::CSharp,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => FileType::Cpp,
            "c" | "h" => FileType::C,
            "rb" => FileType::Ruby,
            "md" | "mdx" => FileType::Markdown,
            "json" => FileType::Json,
            "yaml" | "yml" => FileType::Yaml,
            "toml" => FileType::Toml,
            "html" | "htm" => FileType::Html,
            "css" | "scss" | "sass" | "less" => FileType::Css,
            "sql" => FileType::Sql,
            "sh" | "bash" | "zsh" | "fish" | "ps1" => FileType::Shell,
            other => FileType::Other(other.to_string()),
        }
    }

    /// Check if this is a code file (vs config/doc)
    pub fn is_code(&self) -> bool {
        matches!(
            self,
            FileType::Rust
                | FileType::TypeScript
                | FileType::JavaScript
                | FileType::Python
                | FileType::Go
                | FileType::Java
                | FileType::CSharp
                | FileType::Cpp
                | FileType::C
                | FileType::Ruby
                | FileType::Shell
        )
    }
}

impl Default for FileType {
    fn default() -> Self {
        FileType::Other("unknown".to_string())
    }
}

/// How we learned about this file
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LearnedFrom {
    /// User triggered batch indexing
    ManualIndex,
    /// AI read the file content
    ReadAccess,
    /// AI edited the file
    EditAccess,
    /// File was mentioned in conversation
    Mentioned,
}

impl Default for LearnedFrom {
    fn default() -> Self {
        LearnedFrom::ManualIndex
    }
}

/// Learned knowledge about a file in a codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMemory {
    /// Unique identifier
    pub id: FileMemoryId,

    /// Project this file belongs to
    pub project_id: ProjectId,

    /// User who owns this file memory
    pub user_id: String,

    // =========================================================================
    // FILE IDENTIFICATION
    // =========================================================================
    /// Relative path from codebase root (e.g., "src/main.rs")
    pub path: String,

    /// Absolute path for file access
    pub absolute_path: String,

    /// SHA256 hash of content (for change detection)
    pub file_hash: String,

    // =========================================================================
    // LEARNED CONTENT
    // =========================================================================
    /// AI-generated summary of what this file does
    #[serde(default)]
    pub summary: String,

    /// Key items in the file (functions, classes, exports, constants)
    #[serde(default)]
    pub key_items: Vec<String>,

    /// High-level purpose of this file
    #[serde(default)]
    pub purpose: Option<String>,

    /// Related files (imports, dependencies)
    #[serde(default)]
    pub connections: Vec<String>,

    /// Embedding vector for semantic search
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,

    // =========================================================================
    // METADATA
    // =========================================================================
    /// Detected file type
    pub file_type: FileType,

    /// Number of lines in the file
    pub line_count: usize,

    /// File size in bytes
    pub size_bytes: u64,

    // =========================================================================
    // USAGE TRACKING
    // =========================================================================
    /// Number of times this file was accessed by AI
    #[serde(default)]
    pub access_count: u32,

    /// When this file was last accessed
    pub last_accessed: DateTime<Utc>,

    /// When this FileMemory was created
    pub created_at: DateTime<Utc>,

    /// When this FileMemory was last updated
    pub updated_at: DateTime<Utc>,

    /// How we learned about this file
    #[serde(default)]
    pub learned_from: LearnedFrom,
}

impl FileMemory {
    /// Create a new FileMemory from a file path
    pub fn new(
        project_id: ProjectId,
        user_id: String,
        path: String,
        absolute_path: String,
        file_hash: String,
        file_type: FileType,
        line_count: usize,
        size_bytes: u64,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: FileMemoryId::new(),
            project_id,
            user_id,
            path,
            absolute_path,
            file_hash,
            summary: String::new(),
            key_items: Vec::new(),
            purpose: None,
            connections: Vec::new(),
            embedding: None,
            file_type,
            line_count,
            size_bytes,
            access_count: 1,
            last_accessed: now,
            created_at: now,
            updated_at: now,
            learned_from: LearnedFrom::ManualIndex,
        }
    }

    /// Record an access to this file
    pub fn record_access(&mut self, learned_from: LearnedFrom) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
        self.updated_at = Utc::now();
        // Upgrade the learned_from if more meaningful
        // EditAccess > ReadAccess > Mentioned > ManualIndex
        let should_upgrade = match (&self.learned_from, &learned_from) {
            (LearnedFrom::ManualIndex, _) => true,
            (LearnedFrom::Mentioned, LearnedFrom::ReadAccess | LearnedFrom::EditAccess) => true,
            (LearnedFrom::ReadAccess, LearnedFrom::EditAccess) => true,
            _ => false,
        };
        if should_upgrade {
            self.learned_from = learned_from;
        }
    }

    /// Check if file content has changed (different hash)
    pub fn has_changed(&self, new_hash: &str) -> bool {
        self.file_hash != new_hash
    }

    /// Get a heat score (0-3) based on access count
    /// Used for TUI display: ● (1), ●● (2), ●●● (3+)
    pub fn heat_score(&self) -> u8 {
        match self.access_count {
            0..=2 => 1,
            3..=10 => 2,
            _ => 3,
        }
    }
}

/// Configuration for codebase indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseConfig {
    /// Maximum files to index per project
    #[serde(default = "default_max_files")]
    pub max_files_per_project: usize,

    /// Maximum file size to embed (bytes)
    #[serde(default = "default_max_file_size")]
    pub max_file_size_for_embedding: usize,

    /// Patterns to exclude from indexing
    #[serde(default = "default_exclude_patterns")]
    pub exclude_patterns: Vec<String>,

    /// Whether to skip binary files
    #[serde(default = "default_true")]
    pub skip_binary: bool,
}

fn default_max_files() -> usize {
    1000
}

fn default_max_file_size() -> usize {
    524288 // 500KB
}

fn default_exclude_patterns() -> Vec<String> {
    vec![
        "target/".to_string(),
        "node_modules/".to_string(),
        ".git/".to_string(),
        "__pycache__/".to_string(),
        "dist/".to_string(),
        "build/".to_string(),
        "*.lock".to_string(),
        "*.min.js".to_string(),
        "*.min.css".to_string(),
        "*.map".to_string(),
    ]
}

fn default_true() -> bool {
    true
}

impl Default for CodebaseConfig {
    fn default() -> Self {
        Self {
            max_files_per_project: default_max_files(),
            max_file_size_for_embedding: default_max_file_size(),
            exclude_patterns: default_exclude_patterns(),
            skip_binary: true,
        }
    }
}

/// Result of scanning a codebase directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseScanResult {
    /// Total files found (before filtering)
    pub total_files: usize,
    /// Files that passed filtering
    pub eligible_files: usize,
    /// Files skipped (excluded patterns, binary, etc.)
    pub skipped_files: usize,
    /// Reasons for skipping (pattern -> count)
    pub skip_reasons: HashMap<String, usize>,
    /// Whether limit was reached
    pub limit_reached: bool,
    /// List of eligible file paths
    pub file_paths: Vec<String>,
}

/// Progress update during indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingProgress {
    /// Total files to process
    pub total: usize,
    /// Files processed so far
    pub processed: usize,
    /// Current file being processed
    pub current_file: Option<String>,
    /// Files that errored
    pub errors: Vec<String>,
    /// Whether indexing is complete
    pub complete: bool,
}

impl IndexingProgress {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            processed: 0,
            current_file: None,
            errors: Vec::new(),
            complete: false,
        }
    }

    pub fn percentage(&self) -> f32 {
        if self.total == 0 {
            100.0
        } else {
            (self.processed as f32 / self.total as f32) * 100.0
        }
    }
}

/// Result of maintenance cycle including edge boosts for Hebbian learning
#[derive(Debug, Clone, Default)]
pub struct MaintenanceResult {
    /// Number of memories that had decay applied
    pub decayed_count: usize,
    /// Edge boosts from replay: (from_memory_id, to_memory_id, boost_value)
    /// These should be applied via GraphMemory.strengthen_memory_edge()
    pub edge_boosts: Vec<(String, String, f32)>,
    /// Number of memories replayed during consolidation
    pub memories_replayed: usize,
    /// Total priority score of replayed memories
    pub total_priority_score: f32,
    /// Number of new semantic facts extracted during consolidation
    pub facts_extracted: usize,
    /// Number of existing facts reinforced (dedup hit) during consolidation
    pub facts_reinforced: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_filter_haversine_distance() {
        // San Francisco to Oakland (~13km)
        let sf = GeoFilter::new(37.7749, -122.4194, 100.0);
        let oakland_lat = 37.8044;
        let oakland_lon = -122.2712;

        let distance = sf.haversine_distance(oakland_lat, oakland_lon);
        // Should be approximately 13km (13000m)
        assert!(
            distance > 12000.0 && distance < 14000.0,
            "SF to Oakland should be ~13km, got {distance}m"
        );
    }

    #[test]
    fn test_geo_filter_same_point() {
        let filter = GeoFilter::new(37.7749, -122.4194, 100.0);
        let distance = filter.haversine_distance(37.7749, -122.4194);
        assert!(
            distance < 1.0,
            "Same point should have ~0 distance, got {distance}"
        );
    }

    #[test]
    fn test_geo_filter_contains() {
        // Center at SF with 100m radius
        let filter = GeoFilter::new(37.7749, -122.4194, 100.0);

        // Point within 100m should be contained
        // ~0.001 degrees latitude ≈ 111m
        let nearby_lat = 37.7750;
        let nearby_lon = -122.4194;
        assert!(
            filter.contains(nearby_lat, nearby_lon),
            "Point ~11m away should be within 100m radius"
        );

        // Point far away should NOT be contained
        let oakland_lat = 37.8044;
        let oakland_lon = -122.2712;
        assert!(
            !filter.contains(oakland_lat, oakland_lon),
            "Oakland (~13km) should NOT be within 100m radius"
        );
    }

    #[test]
    fn test_geo_filter_equator_distance() {
        // Test at equator where 1 degree longitude = 111km
        let equator = GeoFilter::new(0.0, 0.0, 1000.0);
        let distance = equator.haversine_distance(0.0, 0.01);
        // 0.01 degrees at equator ≈ 1.11km
        assert!(
            distance > 1000.0 && distance < 1200.0,
            "0.01 degrees at equator should be ~1.1km, got {distance}m"
        );
    }

    #[test]
    fn test_query_default() {
        let query = Query::default();
        assert!(query.query_text.is_none());
        assert!(query.robot_id.is_none());
        assert!(query.mission_id.is_none());
        assert!(query.geo_filter.is_none());
        assert!(query.action_type.is_none());
        assert!(query.reward_range.is_none());
        assert_eq!(query.max_results, DEFAULT_MAX_RESULTS);
    }

    #[test]
    fn test_query_with_robotics_filters() {
        let query = Query {
            robot_id: Some("drone_001".to_string()),
            mission_id: Some("recon_alpha".to_string()),
            geo_filter: Some(GeoFilter::new(37.7749, -122.4194, 500.0)),
            action_type: Some("landing".to_string()),
            reward_range: Some((0.5, 1.0)),
            ..Default::default()
        };

        assert_eq!(query.robot_id, Some("drone_001".to_string()));
        assert_eq!(query.mission_id, Some("recon_alpha".to_string()));
        assert!(query.geo_filter.is_some());
        assert_eq!(query.action_type, Some("landing".to_string()));
        assert_eq!(query.reward_range, Some((0.5, 1.0)));
    }
}
