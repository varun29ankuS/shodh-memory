//! Type definitions for the memory system

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unique identifier for memories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
}

/// Default experience type for minimal API calls
fn default_experience_type() -> ExperienceType {
    ExperienceType::Observation
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
    IsA,           // Inheritance
    HasA,          // Composition
    Uses,          // Dependency
    RelatedTo,     // General association
    Causes,        // Causation
    PartOf,        // Part-whole
    Similar,       // Similarity
    Opposite,      // Antonym/opposite
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

    /// Robot/drone identifier for multi-agent systems
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_id: Option<String>,

    /// Mission identifier this experience belongs to
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mission_id: Option<String>,

    /// GPS coordinates (latitude, longitude, altitude)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geo_location: Option<[f64; 3]>,

    /// Local coordinates (x, y, z in meters)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_position: Option<[f32; 3]>,

    /// Heading in degrees (0-360)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading: Option<f32>,

    /// Action that was performed (for action-outcome learning)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_type: Option<String>,

    /// Reward signal for reinforcement learning (-1.0 to 1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,

    /// Sensor readings at time of experience
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub sensor_data: HashMap<String, f64>,

    // =========================================================================
    // DECISION & LEARNING FIELDS (for action-outcome learning)
    // =========================================================================

    /// Decision context: What state/conditions led to this decision?
    /// E.g., "battery_low=true, obstacle_ahead=true, weather=windy"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_context: Option<HashMap<String, String>>,

    /// Action parameters: Specific parameters of the action taken
    /// E.g., {"speed": "0.5", "turn_angle": "45", "altitude_change": "-10"}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_params: Option<HashMap<String, String>>,

    /// Outcome type: success, failure, partial, aborted, timeout
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_type: Option<String>,

    /// Outcome details: What specifically happened?
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_details: Option<String>,

    /// Confidence score for this decision (0.0-1.0)
    /// How confident was the system when making this decision?
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,

    /// Alternative actions considered but not taken
    /// For learning "what else could have been done"
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives_considered: Vec<String>,

    // =========================================================================
    // ENVIRONMENTAL CONTEXT
    // =========================================================================

    /// Weather conditions: {"wind_speed": "15", "visibility": "good", "precipitation": "none"}
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weather: Option<HashMap<String, String>>,

    /// Terrain type: indoor, outdoor, urban, rural, water, aerial
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terrain_type: Option<String>,

    /// Lighting conditions: bright, dim, dark, variable
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lighting: Option<String>,

    /// Other agents detected: [{"id": "drone_002", "distance": "50m", "type": "friendly"}]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity: Option<String>,

    /// Recovery action taken (if this was a failure)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_action: Option<String>,

    /// Root cause (if known)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_cause: Option<String>,

    // =========================================================================
    // LEARNED PATTERNS & PREDICTIONS
    // =========================================================================

    /// Pattern ID this experience matches (if recognized)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern_id: Option<String>,

    /// Predicted outcome before action was taken
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_outcome: Option<String>,

    /// Was the prediction correct?
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction_accurate: Option<bool>,

    /// Tags for quick filtering: ["obstacle", "battery", "navigation", "emergency"]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

/// Mutable metadata for memory (interior mutability)
/// Separated from immutable core data to enable zero-copy updates via Arc<Memory>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub importance: f32,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
    pub temporal_relevance: f32,
}

impl MemoryMetadata {
    /// Boost importance based on access patterns (enterprise feature)
    pub fn boost_importance(&mut self) {
        if self.access_count > 5 {
            self.importance = (self.importance * 1.1).min(1.0);
        }
    }
}

/// Stored memory with metadata
///
/// Uses Arc<Mutex<MemoryMetadata>> for interior mutability, enabling updates
/// through Arc<Memory> without cloning large embedding vectors (1.5-6KB each).
/// This eliminates 10-100x allocation overhead on hot paths (record, retrieve).
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: MemoryId,
    pub experience: Experience,

    // Mutable metadata protected by Mutex for zero-copy updates
    metadata: Arc<parking_lot::Mutex<MemoryMetadata>>,

    pub created_at: DateTime<Utc>,
    pub compressed: bool,

    // Multi-tenancy support for enterprise deployments
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub actor_id: Option<String>,

    // Similarity score (only populated in search results, not stored)
    pub score: Option<f32>,
}

impl Memory {
    /// Create new memory with given parameters
    pub fn new(
        id: MemoryId,
        experience: Experience,
        importance: f32,
        agent_id: Option<String>,
        run_id: Option<String>,
        actor_id: Option<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            experience,
            metadata: Arc::new(parking_lot::Mutex::new(MemoryMetadata {
                importance,
                access_count: 0,
                last_accessed: now,
                temporal_relevance: 1.0,
            })),
            created_at: now,
            compressed: false,
            agent_id,
            run_id,
            actor_id,
            score: None,
        }
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

    /// Update access metadata (zero-copy through Arc)
    pub fn update_access(&self) {
        let mut meta = self.metadata.lock();
        meta.last_accessed = Utc::now();
        meta.access_count += 1;
        meta.boost_importance();
    }

    /// Set importance (thread-safe)
    pub fn set_importance(&self, importance: f32) {
        self.metadata.lock().importance = importance;
    }

    /// Set temporal relevance (thread-safe)
    pub fn set_temporal_relevance(&self, relevance: f32) {
        self.metadata.lock().temporal_relevance = relevance;
    }
}

// Custom serialization for Memory to flatten the Arc<Mutex<>> field
impl Serialize for Memory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let meta = self.metadata.lock();
        // Always serialize all 12 fields for binary format consistency
        let mut state = serializer.serialize_struct("Memory", 12)?;
        state.serialize_field("memory_id", &self.id)?;
        state.serialize_field("experience", &self.experience)?;
        state.serialize_field("importance", &meta.importance)?;
        state.serialize_field("access_count", &meta.access_count)?;
        state.serialize_field("created_at", &self.created_at)?;
        state.serialize_field("last_accessed", &meta.last_accessed)?;
        state.serialize_field("compressed", &self.compressed)?;
        state.serialize_field("agent_id", &self.agent_id)?;
        state.serialize_field("run_id", &self.run_id)?;
        state.serialize_field("actor_id", &self.actor_id)?;
        state.serialize_field("temporal_relevance", &meta.temporal_relevance)?;
        state.serialize_field("score", &self.score)?;  // Always serialize, Option handles None
        state.end()
    }
}

// Custom deserialization for Memory to reconstruct Arc<Mutex<>>
impl<'de> Deserialize<'de> for Memory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MemoryFlat {
            #[serde(rename = "memory_id")]
            id: MemoryId,
            experience: Experience,
            importance: f32,
            access_count: u32,
            created_at: DateTime<Utc>,
            last_accessed: DateTime<Utc>,
            compressed: bool,
            agent_id: Option<String>,
            run_id: Option<String>,
            actor_id: Option<String>,
            temporal_relevance: f32,
            score: Option<f32>,
        }

        let flat = MemoryFlat::deserialize(deserializer)?;
        Ok(Memory {
            id: flat.id,
            experience: flat.experience,
            metadata: Arc::new(parking_lot::Mutex::new(MemoryMetadata {
                importance: flat.importance,
                access_count: flat.access_count,
                last_accessed: flat.last_accessed,
                temporal_relevance: flat.temporal_relevance,
            })),
            created_at: flat.created_at,
            compressed: flat.compressed,
            agent_id: flat.agent_id,
            run_id: flat.run_id,
            actor_id: flat.actor_id,
            score: flat.score,
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
        Self { lat, lon, radius_meters }
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

/// Query for retrieving memories
#[derive(Debug, Clone)]
pub struct Query {
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

    // === Result Control ===
    pub max_results: usize,
    pub retrieval_mode: RetrievalMode,
}

impl Default for Query {
    fn default() -> Self {
        Self {
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
            max_results: 10,
            retrieval_mode: RetrievalMode::Hybrid,
        }
    }
}

/// Retrieval modes
#[derive(Debug, Clone)]
pub enum RetrievalMode {
    Similarity,     // Vector similarity search
    Temporal,       // Time-based retrieval
    Causal,         // Cause-effect chains
    Associative,    // Related memories
    Hybrid,         // Combination of methods
    // === Robotics-Specific Modes ===
    Spatial,        // Geo-location based retrieval
    Mission,        // Mission context retrieval
    ActionOutcome,  // Reward-based learning retrieval
}

/// Criteria for forgetting memories
#[derive(Debug, Clone)]
pub enum ForgetCriteria {
    OlderThan(u32),           // Days
    LowImportance(f32),       // Threshold
    Pattern(String),          // Regex pattern
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
    #[allow(unused)]  // Public API convenience method
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
    pub fn search(&self, query: &Query, limit: usize) -> anyhow::Result<Vec<SharedMemory>> {
        let mut results: Vec<SharedMemory> = self.memories.values()
            .filter(|m| {
                // Apply filters
                if let Some(threshold) = query.importance_threshold {
                    if m.importance() < threshold {
                        return false;
                    }
                }
                if let Some(types) = &query.experience_types {
                    if !types.iter().any(|t| std::mem::discriminant(&m.experience.experience_type) == std::mem::discriminant(t)) {
                        return false;
                    }
                }
                if let Some((start, end)) = &query.time_range {
                    if m.created_at < *start || m.created_at > *end {
                        return false;
                    }
                }
                true
            })
            .cloned()  // Arc::clone is cheap (just increments ref count)
            .collect();

        // Sort by importance and recency
        results.sort_by(|a, b| {
            b.importance().partial_cmp(&a.importance())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.last_accessed().cmp(&a.last_accessed()))
        });

        results.truncate(limit);
        Ok(results)
    }

    pub fn size(&self) -> usize {
        self.memories.len()
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
                result.push(Arc::clone(memory));  // Cheap: just ref count increment
            }
        }
        Ok(result)
    }

    pub fn remove(&mut self, id: &MemoryId) -> anyhow::Result<()> {
        self.memories.remove(id);
        self.access_order.retain(|x| x != id);
        Ok(())
    }

    pub fn remove_older_than(&mut self, cutoff: DateTime<Utc>) -> anyhow::Result<()> {
        let to_remove: Vec<MemoryId> = self.memories
            .iter()
            .filter(|(_, m)| m.created_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(())
    }

    pub fn remove_below_importance(&mut self, threshold: f32) -> anyhow::Result<()> {
        let to_remove: Vec<MemoryId> = self.memories
            .iter()
            .filter(|(_, m)| m.importance() < threshold)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(())
    }

    pub fn remove_matching(&mut self, regex: &regex::Regex) -> anyhow::Result<usize> {
        let to_remove: Vec<MemoryId> = self.memories
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

/// Session memory - medium-term storage
///
/// Now uses Arc<Memory> for zero-copy shared ownership.
pub struct SessionMemory {
    memories: HashMap<MemoryId, SharedMemory>,
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
        let memory_size = bincode::serialize(&*memory)?.len();

        // Check if adding would exceed limit
        if self.current_size_bytes + memory_size > self.max_size_mb * 1024 * 1024 {
            // Evict lowest importance memories until there's space
            self.evict_to_make_space(memory_size)?;
        }

        self.memories.insert(memory.id.clone(), memory);
        self.current_size_bytes += memory_size;
        Ok(())
    }

    fn evict_to_make_space(&mut self, needed_bytes: usize) -> anyhow::Result<()> {
        let mut sorted: Vec<(MemoryId, f32)> = self.memories
            .iter()
            .map(|(id, m)| (id.clone(), m.importance()))
            .collect();

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (id, _) in sorted {
            if self.current_size_bytes + needed_bytes <= self.max_size_mb * 1024 * 1024 {
                break;
            }
            if let Some(memory) = self.memories.remove(&id) {
                let size = bincode::serialize(&*memory)?.len();
                self.current_size_bytes -= size;
            }
        }
        Ok(())
    }

    /// Search memories (returns Arc<Memory> for zero-copy)
    pub fn search(&self, query: &Query, limit: usize) -> anyhow::Result<Vec<SharedMemory>> {
        let mut results: Vec<SharedMemory> = self.memories.values()
            .filter(|m| {
                if let Some(threshold) = query.importance_threshold {
                    if m.importance() < threshold {
                        return false;
                    }
                }
                true
            })
            .cloned()  // Arc::clone is cheap
            .collect();

        results.sort_by(|a, b| b.importance().partial_cmp(&a.importance()).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    pub fn size_mb(&self) -> usize {
        self.current_size_bytes / (1024 * 1024)
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
        }
        Ok(())
    }

    /// Get important memories (zero-copy with Arc)
    pub fn get_important(&self, threshold: f32) -> anyhow::Result<Vec<SharedMemory>> {
        Ok(self.memories.values()
            .filter(|m| m.importance() >= threshold)
            .cloned()  // Arc::clone is cheap
            .collect())
    }

    pub fn remove(&mut self, id: &MemoryId) -> anyhow::Result<()> {
        if let Some(memory) = self.memories.remove(id) {
            let size = bincode::serialize(&*memory)?.len();
            self.current_size_bytes -= size;
        }
        Ok(())
    }

    pub fn remove_older_than(&mut self, cutoff: DateTime<Utc>) -> anyhow::Result<()> {
        let to_remove: Vec<MemoryId> = self.memories
            .iter()
            .filter(|(_, m)| m.created_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(())
    }

    pub fn remove_below_importance(&mut self, threshold: f32) -> anyhow::Result<()> {
        let to_remove: Vec<MemoryId> = self.memories
            .iter()
            .filter(|(_, m)| m.importance() < threshold)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.remove(&id)?;
        }
        Ok(())
    }

    pub fn remove_matching(&mut self, regex: &regex::Regex) -> anyhow::Result<usize> {
        let to_remove: Vec<MemoryId> = self.memories
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

/// Memory statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub working_memory_count: usize,
    pub session_memory_count: usize,
    pub long_term_memory_count: usize,
    pub compressed_count: usize,
    pub promotions_to_session: usize,
    pub promotions_to_longterm: usize,
    pub total_retrievals: usize,
    pub average_importance: f32,
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
        assert!(distance > 12000.0 && distance < 14000.0,
                "SF to Oakland should be ~13km, got {distance}m");
    }

    #[test]
    fn test_geo_filter_same_point() {
        let filter = GeoFilter::new(37.7749, -122.4194, 100.0);
        let distance = filter.haversine_distance(37.7749, -122.4194);
        assert!(distance < 1.0, "Same point should have ~0 distance, got {distance}");
    }

    #[test]
    fn test_geo_filter_contains() {
        // Center at SF with 100m radius
        let filter = GeoFilter::new(37.7749, -122.4194, 100.0);

        // Point within 100m should be contained
        // ~0.001 degrees latitude ≈ 111m
        let nearby_lat = 37.7750;
        let nearby_lon = -122.4194;
        assert!(filter.contains(nearby_lat, nearby_lon),
                "Point ~11m away should be within 100m radius");

        // Point far away should NOT be contained
        let oakland_lat = 37.8044;
        let oakland_lon = -122.2712;
        assert!(!filter.contains(oakland_lat, oakland_lon),
                "Oakland (~13km) should NOT be within 100m radius");
    }

    #[test]
    fn test_geo_filter_equator_distance() {
        // Test at equator where 1 degree longitude = 111km
        let equator = GeoFilter::new(0.0, 0.0, 1000.0);
        let distance = equator.haversine_distance(0.0, 0.01);
        // 0.01 degrees at equator ≈ 1.11km
        assert!(distance > 1000.0 && distance < 1200.0,
                "0.01 degrees at equator should be ~1.1km, got {distance}m");
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
        assert_eq!(query.max_results, 10);
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