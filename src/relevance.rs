//! Proactive Memory Surfacing (SHO-29)
//!
//! Shifts from pull-based to push-based memory model. Instead of agents explicitly
//! querying for memories, this module proactively surfaces relevant context based on
//! current conversation/context.
//!
//! # Key Features
//! - Entity matching triggers: Surface memories mentioning detected entities
//! - Semantic similarity scoring: Find memories similar to current context
//! - Configurable relevance thresholds
//! - Sub-30ms latency requirement for real-time use
//!
//! # Performance Architecture
//! - Pre-indexed entity lookups (O(1) hash lookup + O(k) retrieval)
//! - LRU-cached context embeddings to avoid re-computation
//! - Entity index for fast entity-to-memory lookups
//! - Rank-based semantic similarity scoring
//! - Parallel entity + semantic search
//! - Early termination when threshold memories found

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use regex::Regex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::embeddings::NeuralNer;
use crate::graph_memory::GraphMemory;
use crate::memory::feedback::FeedbackStore;
use crate::memory::{Memory, MemorySystem};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Check if a word appears as a complete word in text (not as substring).
/// Uses word boundary matching to avoid false positives like "MIT" matching "submit".
fn contains_word(text: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    // Build regex pattern with word boundaries
    // Escape special regex characters in the word
    let escaped = regex::escape(word);
    let pattern = format!(r"(?i)\b{}\b", escaped);
    match Regex::new(&pattern) {
        Ok(re) => re.is_match(text),
        Err(_) => text.contains(word), // Fallback to substring if regex fails
    }
}

// =============================================================================
// LEARNED WEIGHT CONSTANTS
// Default starting points, adapted via feedback
// =============================================================================

/// Default weight for semantic similarity in score fusion
/// CTX-3: Reduced to prioritize proven-value signals over looks-good signals
const DEFAULT_SEMANTIC_WEIGHT: f32 = 0.18;

/// Default weight for entity matching in score fusion
/// CTX-3: Reduced to prioritize proven-value signals
const DEFAULT_ENTITY_WEIGHT: f32 = 0.17;

/// Default weight for tag matching in score fusion
const DEFAULT_TAG_WEIGHT: f32 = 0.05;

/// Default weight for importance in score fusion
const DEFAULT_IMPORTANCE_WEIGHT: f32 = 0.05;

/// Default weight for feedback momentum EMA in score fusion (FBK-5, CTX-3)
/// Significantly increased to prioritize proven-helpful memories
/// High momentum = consistently useful in past interactions
/// This is the PRIMARY signal for memory quality
const DEFAULT_MOMENTUM_WEIGHT: f32 = 0.28;

/// Default weight for access count in score fusion (CTX-3)
/// Memories accessed more frequently have proven value
const DEFAULT_ACCESS_COUNT_WEIGHT: f32 = 0.14;

/// Default weight for graph Hebbian strength in score fusion (CTX-3)
/// Memories with stronger entity relationship edges have proven associations
const DEFAULT_GRAPH_STRENGTH_WEIGHT: f32 = 0.13;

/// Learning rate for weight updates from feedback
const WEIGHT_LEARNING_RATE: f32 = 0.05;

/// Minimum weight (prevents any component from being zeroed out)
const MIN_WEIGHT: f32 = 0.05;

/// Sigmoid calibration steepness (higher = sharper cutoff)
const SIGMOID_STEEPNESS: f32 = 10.0;

/// Sigmoid calibration midpoint (scores below this are penalized)
const SIGMOID_MIDPOINT: f32 = 0.5;

/// Configuration for relevance triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceConfig {
    /// Minimum semantic similarity score (0.0-1.0) to consider a memory relevant
    #[serde(default = "default_semantic_threshold")]
    pub semantic_threshold: f32,

    /// Minimum entity match score (0.0-1.0) to trigger entity-based surfacing
    #[serde(default = "default_entity_threshold")]
    pub entity_threshold: f32,

    /// Maximum number of memories to surface
    #[serde(default = "default_max_results")]
    pub max_results: usize,

    /// Memory types to include (empty = all types)
    #[serde(default)]
    pub memory_types: Vec<String>,

    /// Whether to include entity-based matching
    #[serde(default = "default_true")]
    pub enable_entity_matching: bool,

    /// Whether to include semantic similarity matching
    #[serde(default = "default_true")]
    pub enable_semantic_matching: bool,

    /// Minimum importance score for memories to be surfaced
    #[serde(default = "default_min_importance")]
    pub min_importance: f32,

    /// Time window for recency boost (memories within this window get boosted)
    #[serde(default = "default_recency_hours")]
    pub recency_boost_hours: u64,

    /// Recency boost multiplier (1.0 = no boost)
    #[serde(default = "default_recency_multiplier")]
    pub recency_boost_multiplier: f32,

    /// Graph connectivity boost multiplier (FBK-7)
    /// Applied to memories that have graph relationships with detected entities
    #[serde(default = "default_graph_boost_multiplier")]
    pub graph_boost_multiplier: f32,
}

fn default_graph_boost_multiplier() -> f32 {
    1.15 // 15% boost for graph-connected memories
}

fn default_semantic_threshold() -> f32 {
    0.45 // Lowered from 0.65 - composite relevance scores blend semantic, entity, recency
}

fn default_entity_threshold() -> f32 {
    0.5
}

fn default_max_results() -> usize {
    5
}

fn default_true() -> bool {
    true
}

fn default_min_importance() -> f32 {
    0.3
}

fn default_recency_hours() -> u64 {
    24
}

fn default_recency_multiplier() -> f32 {
    1.2
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        Self {
            semantic_threshold: default_semantic_threshold(),
            entity_threshold: default_entity_threshold(),
            max_results: default_max_results(),
            memory_types: Vec::new(),
            enable_entity_matching: true,
            enable_semantic_matching: true,
            min_importance: default_min_importance(),
            recency_boost_hours: default_recency_hours(),
            recency_boost_multiplier: default_recency_multiplier(),
            graph_boost_multiplier: default_graph_boost_multiplier(),
        }
    }
}

/// A surfaced memory with relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacedMemory {
    /// Memory ID
    pub id: String,

    /// Memory content
    pub content: String,

    /// Memory type (Decision, Learning, etc.)
    pub memory_type: String,

    /// Base importance score
    pub importance: f32,

    /// Relevance score for this context (0.0-1.0)
    pub relevance_score: f32,

    /// Why this memory was surfaced
    pub relevance_reason: RelevanceReason,

    /// Matched entities (if entity-based)
    pub matched_entities: Vec<String>,

    /// Semantic similarity score (if semantic-based)
    pub semantic_similarity: Option<f32>,

    /// When the memory was created
    pub created_at: DateTime<Utc>,

    /// Tags associated with the memory
    pub tags: Vec<String>,
}

/// Reason why a memory was surfaced
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelevanceReason {
    /// Matched one or more entities in the context
    EntityMatch,
    /// High semantic similarity to current context
    SemanticSimilarity,
    /// Both entity match and semantic similarity
    Combined,
    /// Recent and important (fallback when no strong matches)
    RecentImportant,
}

/// Request for proactive memory surfacing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceRequest {
    /// User ID for memory lookup
    pub user_id: String,

    /// Current context text (conversation turn, user message, etc.)
    pub context: String,

    /// Optional pre-extracted entities (skip NER if provided)
    #[serde(default)]
    pub entities: Vec<String>,

    /// Configuration overrides
    #[serde(default)]
    pub config: RelevanceConfig,
}

/// Response from proactive memory surfacing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceResponse {
    /// Surfaced memories ordered by relevance
    pub memories: Vec<SurfacedMemory>,

    /// Entities detected in the context
    pub detected_entities: Vec<DetectedEntity>,

    /// Processing time in milliseconds
    pub latency_ms: f64,

    /// Whether latency target (<30ms) was met
    pub latency_target_met: bool,

    /// Debug info (only in debug builds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<RelevanceDebug>,
}

/// Detected entity with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEntity {
    /// Entity name
    pub name: String,

    /// Entity type (Person, Organization, etc.)
    pub entity_type: String,

    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
}

/// Debug information for relevance calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceDebug {
    /// Time spent on NER
    pub ner_ms: f64,

    /// Time spent on entity matching
    pub entity_match_ms: f64,

    /// Time spent on semantic search
    pub semantic_search_ms: f64,

    /// Time spent on scoring and ranking
    pub ranking_ms: f64,

    /// Number of memories scanned
    pub memories_scanned: usize,

    /// Number of entity matches found
    pub entity_matches: usize,

    /// Number of semantic matches found
    pub semantic_matches: usize,
}

/// Entity index entry: maps entity name to memory IDs containing that entity
#[derive(Debug, Clone, Default)]
struct EntityIndexEntry {
    /// Memory IDs that mention this entity
    memory_ids: HashSet<Uuid>,
    /// Last time this entry was updated
    #[allow(dead_code)]
    last_updated: Option<DateTime<Utc>>,
}

/// Learned weights for score fusion, adapted via feedback
///
/// These weights determine how different scoring signals are combined:
/// - semantic: Cosine similarity from vector search
/// - entity: Entity name overlap between query and memory
/// - tag: Tag overlap between query context and memory tags
/// - importance: Memory's stored importance score
/// - momentum: Feedback momentum EMA (FBK-5) - learned from past interactions
/// - access_count: How often the memory has been retrieved (CTX-3)
/// - graph_strength: Hebbian edge strength from knowledge graph (CTX-3)
///
/// Weights are normalized to sum to 1.0 and updated via gradient descent
/// when user provides feedback on surfaced memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedWeights {
    /// Weight for semantic similarity score
    pub semantic: f32,
    /// Weight for entity matching score
    pub entity: f32,
    /// Weight for tag matching score
    pub tag: f32,
    /// Weight for importance score
    pub importance: f32,
    /// Weight for feedback momentum EMA (FBK-5)
    /// Higher momentum = memory has been consistently helpful
    /// Lower/negative momentum = memory has been consistently misleading
    #[serde(default = "default_momentum_weight")]
    pub momentum: f32,
    /// Weight for access count (CTX-3)
    /// Higher access count = memory has proven retrieval value
    #[serde(default = "default_access_count_weight")]
    pub access_count: f32,
    /// Weight for graph Hebbian strength (CTX-3)
    /// Higher strength = stronger entity relationships in knowledge graph
    #[serde(default = "default_graph_strength_weight")]
    pub graph_strength: f32,
    /// Number of feedback updates applied
    pub update_count: u32,
    /// Last time weights were updated
    pub last_updated: Option<DateTime<Utc>>,
}

fn default_momentum_weight() -> f32 {
    DEFAULT_MOMENTUM_WEIGHT
}

fn default_access_count_weight() -> f32 {
    DEFAULT_ACCESS_COUNT_WEIGHT
}

fn default_graph_strength_weight() -> f32 {
    DEFAULT_GRAPH_STRENGTH_WEIGHT
}

impl Default for LearnedWeights {
    fn default() -> Self {
        Self {
            semantic: DEFAULT_SEMANTIC_WEIGHT,
            entity: DEFAULT_ENTITY_WEIGHT,
            tag: DEFAULT_TAG_WEIGHT,
            importance: DEFAULT_IMPORTANCE_WEIGHT,
            momentum: DEFAULT_MOMENTUM_WEIGHT,
            access_count: DEFAULT_ACCESS_COUNT_WEIGHT,
            graph_strength: DEFAULT_GRAPH_STRENGTH_WEIGHT,
            update_count: 0,
            last_updated: None,
        }
    }
}

impl LearnedWeights {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.semantic
            + self.entity
            + self.tag
            + self.importance
            + self.momentum
            + self.access_count
            + self.graph_strength;
        if sum > 0.0 {
            self.semantic /= sum;
            self.entity /= sum;
            self.tag /= sum;
            self.importance /= sum;
            self.momentum /= sum;
            self.access_count /= sum;
            self.graph_strength /= sum;
        }
    }

    /// Apply feedback to update weights
    ///
    /// - helpful: Increase weights for components that contributed to this memory
    /// - not_helpful: Decrease weights for components that led to this memory
    ///
    /// Uses gradient descent with learning rate WEIGHT_LEARNING_RATE
    pub fn apply_feedback(
        &mut self,
        semantic_contributed: bool,
        entity_contributed: bool,
        tag_contributed: bool,
        helpful: bool,
    ) {
        let direction = if helpful { 1.0 } else { -1.0 };
        let delta = WEIGHT_LEARNING_RATE * direction;

        // Update weights based on which components contributed
        if semantic_contributed {
            self.semantic = (self.semantic + delta).max(MIN_WEIGHT);
        }
        if entity_contributed {
            self.entity = (self.entity + delta).max(MIN_WEIGHT);
        }
        if tag_contributed {
            self.tag = (self.tag + delta).max(MIN_WEIGHT);
        }

        // Importance is always a factor, so adjust inversely to others
        if helpful && !semantic_contributed && !entity_contributed && !tag_contributed {
            // Memory was helpful but no clear signal - boost importance
            self.importance = (self.importance + delta).max(MIN_WEIGHT);
        }

        self.normalize();
        self.update_count += 1;
        self.last_updated = Some(Utc::now());
    }

    /// Calculate fused score from component scores
    pub fn fuse_scores(
        &self,
        semantic_score: f32,
        entity_score: f32,
        tag_score: f32,
        importance_score: f32,
    ) -> f32 {
        // Use the full method with neutral defaults for backwards compatibility
        // momentum = 0 (neutral), access_count = 0, graph_strength = 0.5 (neutral)
        self.fuse_scores_full(
            semantic_score,
            entity_score,
            tag_score,
            importance_score,
            0.0,
            0,
            0.5,
        )
    }

    /// Calculate fused score from component scores including momentum EMA (FBK-5)
    ///
    /// # Arguments
    /// - `semantic_score`: Cosine similarity from vector search (0.0 to 1.0)
    /// - `entity_score`: Entity name overlap score (0.0 to 1.0)
    /// - `tag_score`: Tag overlap score (0.0 to 1.0)
    /// - `importance_score`: Memory's importance score (0.0 to 1.0)
    /// - `momentum_ema`: Feedback momentum EMA (-1.0 to 1.0, 0.0 = neutral)
    ///   - Positive: Memory has been consistently helpful
    ///   - Negative: Memory has been consistently misleading
    pub fn fuse_scores_with_momentum(
        &self,
        semantic_score: f32,
        entity_score: f32,
        tag_score: f32,
        importance_score: f32,
        momentum_ema: f32,
    ) -> f32 {
        // access_count = 0, graph_strength = 0.5 (neutral)
        self.fuse_scores_full(
            semantic_score,
            entity_score,
            tag_score,
            importance_score,
            momentum_ema,
            0,
            0.5,
        )
    }

    /// Calculate fused score from all component scores (CTX-3)
    ///
    /// # Arguments
    /// - `semantic_score`: Cosine similarity from vector search (0.0 to 1.0)
    /// - `entity_score`: Entity name overlap score (0.0 to 1.0)
    /// - `tag_score`: Tag overlap score (0.0 to 1.0)
    /// - `importance_score`: Memory's importance score (0.0 to 1.0)
    /// - `momentum_ema`: Feedback momentum EMA (-1.0 to 1.0, 0.0 = neutral)
    /// - `access_count`: Number of times this memory has been retrieved
    /// - `graph_strength`: Hebbian edge strength from knowledge graph (0.0 to 1.0)
    pub fn fuse_scores_full(
        &self,
        semantic_score: f32,
        entity_score: f32,
        tag_score: f32,
        importance_score: f32,
        momentum_ema: f32,
        access_count: u32,
        graph_strength: f32,
    ) -> f32 {
        // Calibrate each score using sigmoid to normalize different scales
        let calibrated_semantic = calibrate_score(semantic_score);
        let calibrated_entity = calibrate_score(entity_score);
        let calibrated_tag = calibrate_score(tag_score);
        let calibrated_importance = calibrate_score(importance_score);

        // Transform momentum from [-1, 1] to [0, 1] range for calibration
        // EMA of -1.0 (always misleading) -> 0.0
        // EMA of  0.0 (neutral)           -> 0.5
        // EMA of  1.0 (always helpful)    -> 1.0
        let normalized_momentum = (momentum_ema + 1.0) / 2.0;

        // CTX-3: Apply momentum amplification for high-momentum memories
        // Aggressive thresholds to strongly differentiate proven vs unproven memories
        let amplified_momentum = if normalized_momentum > 0.65 {
            // Positive momentum: 1.5x amplification (threshold lowered from 0.75)
            (normalized_momentum * 1.5).min(1.0)
        } else if normalized_momentum < 0.40 {
            // Negative/neutral momentum: 0.3x penalty (threshold raised, penalty increased)
            // This aggressively demotes memories that haven't proven helpful
            (normalized_momentum * 0.3).max(0.0)
        } else {
            normalized_momentum
        };
        let calibrated_momentum = calibrate_score(amplified_momentum);

        // CTX-3: Transform access count to 0-1 scale with diminishing returns
        // 0 accesses -> 0.0, 1 -> 0.25, 2 -> 0.40, 5 -> 0.65, 16+ -> 1.0
        let access_score = if access_count == 0 {
            0.0
        } else {
            // log2(n+1) ensures access_count=1 produces a non-zero score (log2(2)=1.0)
            // unlike log2(1)=0 which made first-access indistinguishable from never-accessed
            let log_access = (access_count as f32 + 1.0).log2();
            (log_access / 4.0).min(1.0)
        };
        let calibrated_access = calibrate_score(access_score);

        // CTX-3: Graph strength already in 0-1 range, calibrate directly
        let calibrated_graph = calibrate_score(graph_strength);

        // Weighted sum
        self.semantic * calibrated_semantic
            + self.entity * calibrated_entity
            + self.tag * calibrated_tag
            + self.importance * calibrated_importance
            + self.momentum * calibrated_momentum
            + self.access_count * calibrated_access
            + self.graph_strength * calibrated_graph
    }
}

/// Calibrate a score using sigmoid function
///
/// Maps scores to a 0-1 range with smooth cutoff around SIGMOID_MIDPOINT.
/// Scores near 1.0 stay near 1.0, scores near 0 are penalized more.
fn calibrate_score(score: f32) -> f32 {
    1.0 / (1.0 + (-SIGMOID_STEEPNESS * (score - SIGMOID_MIDPOINT)).exp())
}

/// Proactive relevance engine for memory surfacing
pub struct RelevanceEngine {
    /// Neural NER for entity extraction
    ner: Arc<NeuralNer>,

    /// Entity name index for O(1) lookup (entity_name_lower -> memory_ids)
    /// Populated from GraphMemory on first use or via refresh_entity_index()
    entity_index: Arc<RwLock<HashMap<String, EntityIndexEntry>>>,

    /// Tracks when entity index was last fully refreshed
    entity_index_timestamp: Arc<RwLock<Option<DateTime<Utc>>>>,

    /// Learned weights for score fusion, adapted via feedback
    learned_weights: Arc<RwLock<LearnedWeights>>,

    /// Active A/B test ID (if any) for this engine
    active_ab_test: Arc<RwLock<Option<String>>>,
}

impl RelevanceEngine {
    /// Create a new relevance engine
    pub fn new(ner: Arc<NeuralNer>) -> Self {
        Self {
            ner,
            entity_index: Arc::new(RwLock::new(HashMap::new())),
            entity_index_timestamp: Arc::new(RwLock::new(None)),
            learned_weights: Arc::new(RwLock::new(LearnedWeights::default())),
            active_ab_test: Arc::new(RwLock::new(None)),
        }
    }

    /// Set active A/B test for this engine
    ///
    /// When set, the engine will use weights from the A/B test based on user assignment.
    pub fn set_active_ab_test(&self, test_id: Option<String>) {
        *self.active_ab_test.write() = test_id;
    }

    /// Get active A/B test ID
    pub fn get_active_ab_test(&self) -> Option<String> {
        self.active_ab_test.read().clone()
    }

    /// Get current learned weights
    pub fn get_weights(&self) -> LearnedWeights {
        self.learned_weights.read().clone()
    }

    /// Set learned weights (e.g., loaded from storage)
    pub fn set_weights(&self, weights: LearnedWeights) {
        *self.learned_weights.write() = weights;
    }

    /// Apply feedback to update learned weights
    ///
    /// Call this when user indicates a surfaced memory was helpful or not.
    pub fn apply_feedback(
        &self,
        semantic_contributed: bool,
        entity_contributed: bool,
        tag_contributed: bool,
        helpful: bool,
    ) {
        self.learned_weights.write().apply_feedback(
            semantic_contributed,
            entity_contributed,
            tag_contributed,
            helpful,
        );
    }

    /// Calculate tag overlap score between context and memory tags
    fn calculate_tag_score(&self, context: &str, tags: &[String]) -> f32 {
        if tags.is_empty() {
            return 0.0;
        }

        let context_lower = context.to_lowercase();
        let mut matches = 0;

        for tag in tags {
            let tag_lower = tag.to_lowercase();
            // Check if tag appears in context (or context words match tag)
            if context_lower.contains(&tag_lower) {
                matches += 1;
            } else {
                // Check if any context word starts with or equals the tag
                for word in context_lower.split_whitespace() {
                    if word.starts_with(&tag_lower) || tag_lower.starts_with(word) {
                        matches += 1;
                        break;
                    }
                }
            }
        }

        matches as f32 / tags.len() as f32
    }

    /// Surface relevant memories for the given context
    ///
    /// This is the main entry point for proactive memory surfacing.
    /// Target latency: <30ms
    pub fn surface_relevant(
        &self,
        context: &str,
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: &RelevanceConfig,
        feedback_store: Option<&RwLock<FeedbackStore>>,
    ) -> Result<RelevanceResponse> {
        self.surface_relevant_inner(
            context,
            memory_system,
            graph_memory,
            config,
            feedback_store,
            None,
        )
    }

    fn surface_relevant_inner(
        &self,
        context: &str,
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: &RelevanceConfig,
        feedback_store: Option<&RwLock<FeedbackStore>>,
        weights_override: Option<LearnedWeights>,
    ) -> Result<RelevanceResponse> {
        let start = Instant::now();
        let mut debug = RelevanceDebug {
            ner_ms: 0.0,
            entity_match_ms: 0.0,
            semantic_search_ms: 0.0,
            ranking_ms: 0.0,
            memories_scanned: 0,
            entity_matches: 0,
            semantic_matches: 0,
        };

        // Phase 1: Entity extraction (if enabled)
        let ner_start = Instant::now();
        let detected_entities = if config.enable_entity_matching {
            self.extract_entities(context)
        } else {
            Vec::new()
        };
        debug.ner_ms = ner_start.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Parallel entity + semantic search
        // Track individual component scores for learned weight fusion
        // Structure: (Memory, semantic_score, entity_score, matched_entities)
        let mut candidate_memories: HashMap<Uuid, (Memory, f32, f32, Vec<String>)> = HashMap::new();

        // 2a: Entity-based matching
        if config.enable_entity_matching && !detected_entities.is_empty() {
            let entity_start = Instant::now();
            let entity_matches =
                self.match_by_entities(&detected_entities, memory_system, graph_memory, config)?;
            debug.entity_match_ms = entity_start.elapsed().as_secs_f64() * 1000.0;
            debug.entity_matches = entity_matches.len();

            for (memory, score, matched) in entity_matches {
                let id = memory.id.0;
                // semantic=0, entity=score
                candidate_memories.insert(id, (memory, 0.0, score, matched));
            }
        }

        // 2b: Semantic similarity matching
        if config.enable_semantic_matching {
            let semantic_start = Instant::now();
            let semantic_matches = self.match_by_semantic(context, memory_system, config)?;
            debug.semantic_search_ms = semantic_start.elapsed().as_secs_f64() * 1000.0;
            debug.semantic_matches = semantic_matches.len();

            for (memory, score) in semantic_matches {
                let id = memory.id.0;
                if let Some((_, semantic_score, _entity_score, _matched)) =
                    candidate_memories.get_mut(&id)
                {
                    // Already found via entity match - add semantic score
                    *semantic_score = score;
                } else {
                    // New candidate from semantic search only
                    candidate_memories.insert(id, (memory, score, 0.0, Vec::new()));
                }
            }
        }

        debug.memories_scanned = candidate_memories.len();

        // Phase 3: Rank and select top results using learned weights
        let ranking_start = Instant::now();
        let weights = weights_override.unwrap_or_else(|| self.learned_weights.read().clone());

        let mut results: Vec<SurfacedMemory> = candidate_memories
            .into_iter()
            .filter_map(
                |(_, (memory, semantic_score, entity_score, matched_entities))| {
                    // Apply minimum importance filter
                    let importance = memory.importance();
                    if importance < config.min_importance {
                        return None;
                    }

                    // Apply memory type filter
                    if !config.memory_types.is_empty() {
                        let mem_type = format!("{:?}", memory.experience.experience_type);
                        if !config
                            .memory_types
                            .iter()
                            .any(|t| t.eq_ignore_ascii_case(&mem_type))
                        {
                            return None;
                        }
                    }

                    // Calculate tag score
                    let tag_score = self.calculate_tag_score(context, &memory.experience.tags);

                    // CTX-3: Get access count for momentum scoring
                    let access_count = memory.access_count();

                    // CTX-3: Get graph Hebbian strength for this memory
                    let graph_strength = graph_memory
                        .and_then(|g| g.get_memory_hebbian_strength(&memory.id))
                        .unwrap_or(0.5); // Neutral default if no graph or no edges

                    // Look up momentum EMA from feedback store when available
                    let momentum_ema = feedback_store
                        .and_then(|fs| {
                            let store = fs.read();
                            store.get_momentum(&memory.id).map(|m| m.ema_with_decay())
                        })
                        .unwrap_or(0.0);

                    // Fuse scores using learned weights with all signals including momentum
                    let fused_score = weights.fuse_scores_full(
                        semantic_score,
                        entity_score,
                        tag_score,
                        importance,
                        momentum_ema,
                        access_count,
                        graph_strength,
                    );

                    // Determine relevance reason
                    let reason = if semantic_score > 0.0 && entity_score > 0.0 {
                        RelevanceReason::Combined
                    } else if entity_score > 0.0 {
                        RelevanceReason::EntityMatch
                    } else if semantic_score > 0.0 {
                        RelevanceReason::SemanticSimilarity
                    } else {
                        RelevanceReason::RecentImportant
                    };

                    // Apply recency boost
                    let recency_boosted = self.apply_recency_boost(
                        fused_score,
                        memory.created_at,
                        config.recency_boost_hours,
                        config.recency_boost_multiplier,
                    );

                    // FBK-7: Apply graph boost for memories found via entity matching
                    let final_score = if entity_score > 0.0 {
                        (recency_boosted * config.graph_boost_multiplier).min(1.0)
                    } else {
                        recency_boosted
                    };

                    Some(SurfacedMemory {
                        id: memory.id.0.to_string(),
                        content: memory.experience.content.clone(),
                        memory_type: format!("{:?}", memory.experience.experience_type),
                        importance,
                        relevance_score: final_score,
                        relevance_reason: reason.clone(),
                        matched_entities,
                        semantic_similarity: if semantic_score > 0.0 {
                            Some(semantic_score)
                        } else {
                            None
                        },
                        created_at: memory.created_at,
                        tags: memory.experience.tags.clone(),
                    })
                },
            )
            .collect();

        // Sort by relevance score (descending)
        results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));

        // Quality-based filtering: only return memories above minimum relevance
        // This prevents returning low-quality matches just to fill max_results
        const MIN_RELEVANCE_SCORE: f32 = 0.25;
        results.retain(|r| r.relevance_score >= MIN_RELEVANCE_SCORE);

        // Cap at max_results (but may return fewer if quality threshold filters them)
        results.truncate(config.max_results);

        debug.ranking_ms = ranking_start.elapsed().as_secs_f64() * 1000.0;

        let total_latency = start.elapsed().as_secs_f64() * 1000.0;
        let latency_target_met = total_latency < 30.0;

        Ok(RelevanceResponse {
            memories: results,
            detected_entities,
            latency_ms: total_latency,
            latency_target_met,
            debug: if cfg!(debug_assertions) {
                Some(debug)
            } else {
                None
            },
        })
    }

    /// Surface relevant memories with feedback momentum integration (FBK-5)
    ///
    /// This variant incorporates momentum EMA from past feedback into scoring.
    /// Memories with positive momentum (consistently helpful) get boosted.
    /// Memories with negative momentum (consistently misleading) get penalized.
    ///
    /// # Arguments
    /// - `context`: Current context/query text
    /// - `memory_system`: Memory storage
    /// - `graph_memory`: Optional graph memory for entity lookup
    /// - `config`: Relevance configuration
    /// - `momentum_lookup`: Map of memory_id -> momentum EMA (-1.0 to 1.0)
    pub fn surface_relevant_with_momentum(
        &self,
        context: &str,
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: &RelevanceConfig,
        momentum_lookup: &HashMap<Uuid, f32>,
    ) -> Result<RelevanceResponse> {
        let start = Instant::now();
        let mut debug = RelevanceDebug {
            ner_ms: 0.0,
            entity_match_ms: 0.0,
            semantic_search_ms: 0.0,
            ranking_ms: 0.0,
            memories_scanned: 0,
            entity_matches: 0,
            semantic_matches: 0,
        };

        // Phase 1: Entity extraction (if enabled)
        let ner_start = Instant::now();
        let detected_entities = if config.enable_entity_matching {
            self.extract_entities(context)
        } else {
            Vec::new()
        };
        debug.ner_ms = ner_start.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Parallel entity + semantic search
        let mut candidate_memories: HashMap<Uuid, (Memory, f32, f32, Vec<String>)> = HashMap::new();

        // 2a: Entity-based matching
        if config.enable_entity_matching && !detected_entities.is_empty() {
            let entity_start = Instant::now();
            let entity_matches =
                self.match_by_entities(&detected_entities, memory_system, graph_memory, config)?;
            debug.entity_match_ms = entity_start.elapsed().as_secs_f64() * 1000.0;
            debug.entity_matches = entity_matches.len();

            for (memory, score, matched) in entity_matches {
                let id = memory.id.0;
                candidate_memories.insert(id, (memory, 0.0, score, matched));
            }
        }

        // 2b: Semantic similarity matching
        if config.enable_semantic_matching {
            let semantic_start = Instant::now();
            let semantic_matches = self.match_by_semantic(context, memory_system, config)?;
            debug.semantic_search_ms = semantic_start.elapsed().as_secs_f64() * 1000.0;
            debug.semantic_matches = semantic_matches.len();

            for (memory, score) in semantic_matches {
                let id = memory.id.0;
                if let Some((_, semantic_score, _entity_score, _matched)) =
                    candidate_memories.get_mut(&id)
                {
                    *semantic_score = score;
                } else {
                    candidate_memories.insert(id, (memory, score, 0.0, Vec::new()));
                }
            }
        }

        debug.memories_scanned = candidate_memories.len();

        // Phase 3: Rank with momentum-aware scoring
        let ranking_start = Instant::now();
        let weights = self.learned_weights.read().clone();

        let mut results: Vec<SurfacedMemory> = candidate_memories
            .into_iter()
            .filter_map(
                |(id, (memory, semantic_score, entity_score, matched_entities))| {
                    let importance = memory.importance();
                    if importance < config.min_importance {
                        return None;
                    }

                    if !config.memory_types.is_empty() {
                        let mem_type = format!("{:?}", memory.experience.experience_type);
                        if !config
                            .memory_types
                            .iter()
                            .any(|t| t.eq_ignore_ascii_case(&mem_type))
                        {
                            return None;
                        }
                    }

                    let tag_score = self.calculate_tag_score(context, &memory.experience.tags);

                    // FBK-5: Look up momentum EMA for this memory
                    let momentum_ema = momentum_lookup.get(&id).copied().unwrap_or(0.0);

                    // CTX-3: Get access count for scoring
                    let access_count = memory.access_count();

                    // CTX-3: Get graph Hebbian strength for this memory
                    let graph_strength = graph_memory
                        .and_then(|g| g.get_memory_hebbian_strength(&memory.id))
                        .unwrap_or(0.5); // Neutral default if no graph or no edges

                    // Use full score fusion with momentum, access count, and graph strength (CTX-3)
                    let fused_score = weights.fuse_scores_full(
                        semantic_score,
                        entity_score,
                        tag_score,
                        importance,
                        momentum_ema,
                        access_count,
                        graph_strength,
                    );

                    let reason = if semantic_score > 0.0 && entity_score > 0.0 {
                        RelevanceReason::Combined
                    } else if entity_score > 0.0 {
                        RelevanceReason::EntityMatch
                    } else if semantic_score > 0.0 {
                        RelevanceReason::SemanticSimilarity
                    } else {
                        RelevanceReason::RecentImportant
                    };

                    // Apply recency boost
                    let recency_boosted = self.apply_recency_boost(
                        fused_score,
                        memory.created_at,
                        config.recency_boost_hours,
                        config.recency_boost_multiplier,
                    );

                    // FBK-7: Apply graph boost for memories found via entity matching
                    // Entity matching uses the knowledge graph, so entity_score > 0 indicates graph relevance
                    let final_score = if entity_score > 0.0 {
                        (recency_boosted * config.graph_boost_multiplier).min(1.0)
                    } else {
                        recency_boosted
                    };

                    Some(SurfacedMemory {
                        id: memory.id.0.to_string(),
                        content: memory.experience.content.clone(),
                        memory_type: format!("{:?}", memory.experience.experience_type),
                        importance,
                        relevance_score: final_score,
                        relevance_reason: reason.clone(),
                        matched_entities,
                        semantic_similarity: if semantic_score > 0.0 {
                            Some(semantic_score)
                        } else {
                            None
                        },
                        created_at: memory.created_at,
                        tags: memory.experience.tags.clone(),
                    })
                },
            )
            .collect();

        results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));

        // Quality-based filtering: only return memories above minimum relevance
        const MIN_RELEVANCE_SCORE: f32 = 0.25;
        results.retain(|r| r.relevance_score >= MIN_RELEVANCE_SCORE);

        // Cap at max_results (but may return fewer if quality threshold filters them)
        results.truncate(config.max_results);

        debug.ranking_ms = ranking_start.elapsed().as_secs_f64() * 1000.0;

        let total_latency = start.elapsed().as_secs_f64() * 1000.0;
        let latency_target_met = total_latency < 30.0;

        Ok(RelevanceResponse {
            memories: results,
            detected_entities,
            latency_ms: total_latency,
            latency_target_met,
            debug: if cfg!(debug_assertions) {
                Some(debug)
            } else {
                None
            },
        })
    }

    /// Surface relevant memories with A/B test integration
    ///
    /// When an A/B test manager is provided, this method:
    /// 1. Assigns the user to a variant (control or treatment)
    /// 2. Uses weights from the assigned variant
    /// 3. Records impressions for the test
    ///
    /// Returns the variant used along with the response for tracking.
    pub fn surface_relevant_with_ab_test(
        &self,
        context: &str,
        user_id: &str,
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: &RelevanceConfig,
        ab_manager: &crate::ab_testing::ABTestManager,
    ) -> Result<(RelevanceResponse, Option<crate::ab_testing::ABTestVariant>)> {
        let start = Instant::now();

        // Check if we have an active A/B test
        let active_test = self.get_active_ab_test();

        let (weights, variant) = if let Some(ref test_id) = active_test {
            // Get weights from A/B test based on user assignment
            match ab_manager.get_weights_for_user(test_id, user_id) {
                Ok(w) => {
                    let v = ab_manager.get_variant(test_id, user_id).ok();
                    (w, v)
                }
                Err(_) => {
                    // Fall back to learned weights if test error
                    (self.get_weights(), None)
                }
            }
        } else {
            // No active test, use learned weights
            (self.get_weights(), None)
        };

        // Use per-request weights override (thread-safe: no global state mutation)
        let response = self.surface_relevant_inner(
            context,
            memory_system,
            graph_memory,
            config,
            None,
            Some(weights),
        )?;

        // Record impression for A/B test if active
        if let (Some(ref test_id), Some(ref _v)) = (&active_test, &variant) {
            let latency_us = start.elapsed().as_micros() as u64;
            let avg_score = if response.memories.is_empty() {
                0.0
            } else {
                response
                    .memories
                    .iter()
                    .map(|m| m.relevance_score as f64)
                    .sum::<f64>()
                    / response.memories.len() as f64
            };

            let _ = ab_manager.record_impression(test_id, user_id, avg_score, latency_us);
        }

        Ok((response, variant))
    }

    /// Record a click for A/B testing
    ///
    /// Call this when a user interacts with a surfaced memory.
    pub fn record_ab_click(
        &self,
        user_id: &str,
        memory_id: Uuid,
        ab_manager: &crate::ab_testing::ABTestManager,
    ) -> Result<()> {
        if let Some(test_id) = self.get_active_ab_test() {
            ab_manager
                .record_click(&test_id, user_id, memory_id)
                .map_err(|e| anyhow::anyhow!("Failed to record A/B click: {}", e))?;
        }
        Ok(())
    }

    /// Record feedback for A/B testing
    ///
    /// Call this when a user provides explicit feedback on a surfaced memory.
    pub fn record_ab_feedback(
        &self,
        user_id: &str,
        positive: bool,
        ab_manager: &crate::ab_testing::ABTestManager,
    ) -> Result<()> {
        if let Some(test_id) = self.get_active_ab_test() {
            ab_manager
                .record_feedback(&test_id, user_id, positive)
                .map_err(|e| anyhow::anyhow!("Failed to record A/B feedback: {}", e))?;
        }
        Ok(())
    }

    /// Extract entities from context using NER
    fn extract_entities(&self, context: &str) -> Vec<DetectedEntity> {
        match self.ner.extract(context) {
            Ok(entities) => entities
                .into_iter()
                .map(|e| DetectedEntity {
                    name: e.text,
                    entity_type: format!("{:?}", e.entity_type),
                    confidence: e.confidence,
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Match memories by entity overlap
    ///
    /// Optimized for <30ms latency with two strategies:
    /// 1. **Cache path (O(k))**: Use entity_index for direct entity->memory ID lookup
    /// 2. **Scan path (O(n))**: Fall back to content scanning if cache miss
    ///
    /// The cache path is ~10-100x faster for repeated entity lookups.
    fn match_by_entities(
        &self,
        entities: &[DetectedEntity],
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: &RelevanceConfig,
    ) -> Result<Vec<(Memory, f32, Vec<String>)>> {
        // Pre-compute lowercase entity names and weights for O(1) lookup
        let entity_lookup: Vec<(String, &DetectedEntity, f32)> = entities
            .iter()
            .map(|e| {
                let weight = self.entity_type_weight(&e.entity_type);
                (e.name.to_lowercase(), e, weight)
            })
            .collect();

        let max_candidates = config.max_results * 3;
        let mut results: Vec<(Memory, f32, Vec<String>)> = Vec::with_capacity(max_candidates);
        let mut found_ids: HashSet<Uuid> = HashSet::new();

        // =====================================================================
        // FAST PATH: Use cached entity index for O(1) lookups
        // =====================================================================
        {
            let index = self.entity_index.read();
            if !index.is_empty() {
                // Collect candidate memory IDs from index
                let mut candidate_ids: HashMap<Uuid, (f32, Vec<String>)> = HashMap::new();

                for (name_lower, entity, weight) in &entity_lookup {
                    if let Some(entry) = index.get(name_lower) {
                        for &memory_id in &entry.memory_ids {
                            let score = entity.confidence * weight;
                            candidate_ids
                                .entry(memory_id)
                                .and_modify(|(existing_score, matched)| {
                                    *existing_score += score;
                                    if !matched.contains(&entity.name) {
                                        matched.push(entity.name.clone());
                                    }
                                })
                                .or_insert((score, vec![entity.name.clone()]));
                        }
                    }
                }

                // Fetch memories for candidate IDs
                if !candidate_ids.is_empty() {
                    // Sort by score descending and take top candidates
                    let mut sorted_candidates: Vec<_> = candidate_ids.into_iter().collect();
                    sorted_candidates.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));

                    for (memory_id, (score, matched)) in
                        sorted_candidates.into_iter().take(max_candidates)
                    {
                        let normalized_score = (score / matched.len() as f32).min(1.0);
                        if normalized_score >= config.entity_threshold {
                            // Try to get the memory
                            let mem_id = crate::memory::MemoryId(memory_id);
                            if let Ok(memory) = memory_system.get_memory(&mem_id) {
                                found_ids.insert(memory_id);
                                results.push((memory, normalized_score, matched));
                            }
                        }
                    }
                }

                // If cache hit was successful, return early
                if !results.is_empty() {
                    return Ok(results);
                }
            }
        }

        // =====================================================================
        // SLOW PATH: Full memory scan (when cache empty or miss)
        // =====================================================================
        let all_memories = memory_system.get_all_memories()?;

        if all_memories.is_empty() {
            return Ok(results);
        }

        for shared_memory in &all_memories {
            // Early termination: stop when we have enough high-quality candidates
            if results.len() >= max_candidates {
                break;
            }

            // Skip if already found via cache
            if found_ids.contains(&shared_memory.id.0) {
                continue;
            }

            let content_lower = shared_memory.experience.content.to_lowercase();
            let mut matched: Vec<String> = Vec::new();
            let mut match_score = 0.0f32;

            // Check direct text matches with word boundaries
            for (name_lower, entity, weight) in &entity_lookup {
                if contains_word(&content_lower, name_lower) {
                    matched.push(entity.name.clone());
                    match_score += entity.confidence * weight;
                }
            }

            // Normalize score and filter
            if !matched.is_empty() {
                let normalized_score = (match_score / matched.len() as f32).min(1.0);
                if normalized_score >= config.entity_threshold {
                    results.push(((**shared_memory).clone(), normalized_score, matched));
                }
            }
        }

        // Skip graph memory lookup if we already have enough results (fast path)
        // This avoids expensive graph traversal when text matching is sufficient
        if results.len() >= config.max_results * 2 || graph_memory.is_none() {
            return Ok(results);
        }

        // Graph memory lookup with TRAVERSAL for additional entity relationships
        let mut graph_results = results;
        if let Some(graph) = graph_memory {
            // Build set of already-found memory IDs to avoid duplicates
            let found_ids: HashSet<Uuid> = graph_results.iter().map(|(m, _, _)| m.id.0).collect();

            // Limit graph lookups to avoid latency spikes
            let max_graph_lookups = 5;
            for (idx, entity) in entities.iter().enumerate() {
                if idx >= max_graph_lookups {
                    break;
                }

                if let Ok(Some(entity_node)) = graph.find_entity_by_name(&entity.name) {
                    // GRAPH TRAVERSAL: Traverse to connected entities (5 hops for deep multi-hop reasoning)
                    // Decay factor (0.86^hop) naturally down-weights distant connections
                    if let Ok(traversal) = graph.traverse_from_entity(&entity_node.uuid, 5) {
                        for traversed in &traversal.entities {
                            // Get episodes (memories) connected to this entity
                            if let Ok(episodes) =
                                graph.get_episodes_by_entity(&traversed.entity.uuid)
                            {
                                for episode in episodes.iter().take(10) {
                                    // Episode UUID = Memory ID (we store it this way in remember())
                                    let memory_id = crate::memory::MemoryId(episode.uuid);

                                    // Skip if already found
                                    if found_ids.contains(&episode.uuid) {
                                        continue;
                                    }

                                    // Score = confidence * salience * decay_factor
                                    // decay_factor accounts for hop distance (0.7^hop)
                                    let score = entity.confidence
                                        * traversed.entity.salience
                                        * traversed.decay_factor;
                                    if score >= config.entity_threshold {
                                        // Direct memory lookup by ID (fast!)
                                        if let Ok(memory) = memory_system.get_memory(&memory_id) {
                                            graph_results.push((
                                                memory,
                                                score,
                                                vec![
                                                    entity.name.clone(),
                                                    traversed.entity.name.clone(),
                                                ],
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(graph_results)
    }

    /// Get weight for entity type (used in scoring)
    fn entity_type_weight(&self, entity_type: &str) -> f32 {
        match entity_type.to_lowercase().as_str() {
            "person" => 1.0,
            "organization" => 0.9,
            "location" => 0.8,
            "technology" => 0.85,
            "product" => 0.9,
            "event" => 0.7,
            "date" => 0.5,
            _ => 0.6,
        }
    }

    /// Match memories by semantic similarity
    fn match_by_semantic(
        &self,
        context: &str,
        memory_system: &MemorySystem,
        config: &RelevanceConfig,
    ) -> Result<Vec<(Memory, f32)>> {
        // Build query for semantic search using memory system's retrieve()
        let query = crate::memory::Query {
            query_text: Some(context.to_string()),
            max_results: config.max_results * 2, // Get more candidates for filtering
            importance_threshold: Some(config.min_importance),
            ..Default::default()
        };

        let mut results: Vec<(Memory, f32)> = Vec::new();

        // Use memory system's semantic recall (uses vector index)
        match memory_system.recall(&query) {
            Ok(shared_memories) => {
                // Use the unified 5-layer pipeline score when available.
                // Falls back to reciprocal rank scoring when pipeline score is absent.
                for (rank, shared_memory) in shared_memories.into_iter().enumerate() {
                    let memory = (*shared_memory).clone();
                    let score = shared_memory
                        .get_score()
                        .unwrap_or(1.0 / (rank as f32 + 1.0));
                    if score >= config.semantic_threshold {
                        results.push((memory, score));
                    }
                }
            }
            Err(_) => {
                // Fallback: simple keyword matching (less accurate but fast)
                let context_words: HashSet<&str> =
                    context.split_whitespace().filter(|w| w.len() > 3).collect();

                let all_memories = memory_system.get_all_memories()?;
                for shared_memory in all_memories {
                    let content_words: HashSet<&str> = shared_memory
                        .experience
                        .content
                        .split_whitespace()
                        .filter(|w| w.len() > 3)
                        .collect();

                    let overlap = context_words.intersection(&content_words).count();
                    if overlap > 0 {
                        let score = overlap as f32
                            / (context_words.len() + content_words.len()) as f32
                            * 2.0;
                        if score >= config.semantic_threshold {
                            let memory = (*shared_memory).clone();
                            results.push((memory, score.min(1.0)));
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Apply recency boost to relevance score
    fn apply_recency_boost(
        &self,
        base_score: f32,
        created_at: DateTime<Utc>,
        boost_hours: u64,
        multiplier: f32,
    ) -> f32 {
        let now = Utc::now();
        let age = now.signed_duration_since(created_at);
        let age_hours = age.num_hours() as u64;

        if age_hours <= boost_hours {
            // Linear decay of boost based on age
            let decay = 1.0 - (age_hours as f32 / boost_hours as f32);
            let boost = 1.0 + (multiplier - 1.0) * decay;
            (base_score * boost).min(1.0)
        } else {
            base_score
        }
    }

    // =========================================================================
    // Entity Index Caching Methods
    // =========================================================================

    /// Refresh the entity index from GraphMemory
    ///
    /// This builds an O(1) lookup table from entity names to memory IDs.
    /// Call this periodically or when the graph changes significantly.
    ///
    /// Performance: O(E * M) where E = entities, M = avg memories per entity
    /// Typically completes in <10ms for 1000 entities.
    pub fn refresh_entity_index(&self, graph_memory: &GraphMemory) -> Result<()> {
        let mut index = self.entity_index.write();
        index.clear();

        let now = Utc::now();
        let entities = graph_memory.get_all_entities()?;

        for entity in entities {
            let episodes = graph_memory.get_episodes_by_entity(&entity.uuid)?;
            let episode_ids: HashSet<Uuid> = episodes.iter().map(|e| e.uuid).collect();

            let name_lower = entity.name.to_lowercase();
            index.insert(
                name_lower,
                EntityIndexEntry {
                    memory_ids: episode_ids,
                    last_updated: Some(now),
                },
            );
        }

        // Update timestamp
        *self.entity_index_timestamp.write() = Some(now);

        Ok(())
    }

    /// Get memory IDs for an entity name using the cached index
    ///
    /// Returns None if entity not in index, empty set if entity known but has no memories.
    /// Falls back to direct GraphMemory lookup if index is stale or missing.
    pub fn get_memories_for_entity(
        &self,
        entity_name: &str,
        graph_memory: Option<&GraphMemory>,
    ) -> Option<HashSet<Uuid>> {
        let name_lower = entity_name.to_lowercase();

        // Try cache first
        {
            let index = self.entity_index.read();
            if let Some(entry) = index.get(&name_lower) {
                return Some(entry.memory_ids.clone());
            }
        }

        // Cache miss - try direct lookup if GraphMemory available
        if let Some(graph) = graph_memory {
            if let Ok(Some(entity)) = graph.find_entity_by_name(&name_lower) {
                if let Ok(episodes) = graph.get_episodes_by_entity(&entity.uuid) {
                    let memory_ids: HashSet<Uuid> = episodes.iter().map(|e| e.uuid).collect();

                    // Update cache with this entry
                    let mut index = self.entity_index.write();
                    index.insert(
                        name_lower,
                        EntityIndexEntry {
                            memory_ids: memory_ids.clone(),
                            last_updated: Some(Utc::now()),
                        },
                    );

                    return Some(memory_ids);
                }
            }
        }

        None
    }

    /// Check if entity index needs refresh (older than max_age_hours)
    pub fn entity_index_needs_refresh(&self, max_age_hours: i64) -> bool {
        let timestamp = self.entity_index_timestamp.read();
        match *timestamp {
            None => true,
            Some(ts) => {
                let age = Utc::now().signed_duration_since(ts);
                age.num_hours() > max_age_hours
            }
        }
    }

    /// Get entity index statistics
    pub fn entity_index_stats(&self) -> (usize, Option<DateTime<Utc>>) {
        let index = self.entity_index.read();
        let timestamp = *self.entity_index_timestamp.read();
        (index.len(), timestamp)
    }

    /// Clear the entity index (useful for testing or memory pressure)
    pub fn clear_entity_index(&self) {
        self.entity_index.write().clear();
        *self.entity_index_timestamp.write() = None;
    }
}

// =============================================================================
// WebSocket Protocol Types for /api/context/monitor
// =============================================================================

/// WebSocket handshake message for context monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMonitorHandshake {
    /// User ID to monitor context for
    pub user_id: String,

    /// Optional configuration override
    #[serde(default)]
    pub config: Option<RelevanceConfig>,

    /// Debounce interval in milliseconds (default: 100ms)
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
}

fn default_debounce_ms() -> u64 {
    100
}

/// Context update message sent by client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextUpdate {
    /// The current context/conversation text
    pub context: String,

    /// Optional entities pre-extracted by client
    #[serde(default)]
    pub entities: Vec<String>,

    /// Optional configuration override for this update
    #[serde(default)]
    pub config: Option<RelevanceConfig>,
}

/// Server response for context monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContextMonitorResponse {
    /// Handshake acknowledgement
    #[serde(rename = "ack")]
    Ack { timestamp: DateTime<Utc> },

    /// Relevant memories surfaced
    #[serde(rename = "relevant")]
    Relevant {
        memories: Vec<SurfacedMemory>,
        detected_entities: Vec<DetectedEntity>,
        latency_ms: f64,
        timestamp: DateTime<Utc>,
    },

    /// No relevant memories found (optional, can be suppressed)
    #[serde(rename = "none")]
    None { timestamp: DateTime<Utc> },

    /// Error occurred
    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
        fatal: bool,
        timestamp: DateTime<Utc>,
    },
}

/// Context monitor for WebSocket-based proactive surfacing
pub struct ContextMonitor {
    /// Relevance engine for surfacing
    engine: Arc<RelevanceEngine>,

    /// Default configuration
    default_config: RelevanceConfig,

    /// Minimum time between surfacing checks (prevents spam)
    debounce_ms: u64,
}

impl ContextMonitor {
    /// Create a new context monitor
    pub fn new(engine: Arc<RelevanceEngine>, debounce_ms: u64) -> Self {
        Self {
            engine,
            default_config: RelevanceConfig::default(),
            debounce_ms,
        }
    }

    /// Get the debounce interval
    pub fn debounce_ms(&self) -> u64 {
        self.debounce_ms
    }

    /// Set default configuration
    pub fn set_config(&mut self, config: RelevanceConfig) {
        self.default_config = config;
    }

    /// Get the underlying engine
    pub fn engine(&self) -> &Arc<RelevanceEngine> {
        &self.engine
    }

    /// Process a context update and return relevant memories (if any)
    pub fn process_context(
        &self,
        context: &str,
        memory_system: &MemorySystem,
        graph_memory: Option<&GraphMemory>,
        config: Option<&RelevanceConfig>,
    ) -> Result<Option<RelevanceResponse>> {
        let cfg = config.unwrap_or(&self.default_config);

        // Skip very short contexts
        if context.len() < 10 {
            return Ok(None);
        }

        let response =
            self.engine
                .surface_relevant(context, memory_system, graph_memory, cfg, None)?;

        // Only return if we found relevant memories
        if response.memories.is_empty() {
            Ok(None)
        } else {
            Ok(Some(response))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_config_defaults() {
        let config = RelevanceConfig::default();
        assert_eq!(config.semantic_threshold, 0.45);
        assert_eq!(config.entity_threshold, 0.5);
        assert_eq!(config.max_results, 5);
        assert!(config.enable_entity_matching);
        assert!(config.enable_semantic_matching);
    }

    #[test]
    fn test_recency_boost() {
        let engine = RelevanceEngine::new(Arc::new(crate::embeddings::NeuralNer::new_fallback(
            crate::embeddings::NerConfig::default(),
        )));

        // Recent memory should get boost
        let recent = Utc::now();
        let boosted = engine.apply_recency_boost(0.5, recent, 24, 1.2);
        assert!(boosted > 0.5);

        // Old memory should not get boost
        let old = Utc::now() - chrono::Duration::hours(48);
        let not_boosted = engine.apply_recency_boost(0.5, old, 24, 1.2);
        assert!((not_boosted - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_entity_type_weight() {
        let engine = RelevanceEngine::new(Arc::new(crate::embeddings::NeuralNer::new_fallback(
            crate::embeddings::NerConfig::default(),
        )));

        assert_eq!(engine.entity_type_weight("Person"), 1.0);
        assert_eq!(engine.entity_type_weight("organization"), 0.9);
        assert!(engine.entity_type_weight("unknown") < 1.0);
    }

    #[test]
    fn test_detected_entity_serialization() {
        let entity = DetectedEntity {
            name: "Rust".to_string(),
            entity_type: "Technology".to_string(),
            confidence: 0.95,
        };

        let json = serde_json::to_string(&entity).unwrap();
        assert!(json.contains("Rust"));
        assert!(json.contains("Technology"));
    }

    #[test]
    fn test_learned_weights_default() {
        let weights = LearnedWeights::default();

        // Should sum to 1.0 (includes momentum, access_count, and graph_strength)
        let sum = weights.semantic
            + weights.entity
            + weights.tag
            + weights.importance
            + weights.momentum
            + weights.access_count
            + weights.graph_strength;
        assert!((sum - 1.0).abs() < 0.001);

        // Check default values
        assert_eq!(weights.semantic, DEFAULT_SEMANTIC_WEIGHT);
        assert_eq!(weights.entity, DEFAULT_ENTITY_WEIGHT);
        assert_eq!(weights.tag, DEFAULT_TAG_WEIGHT);
        assert_eq!(weights.importance, DEFAULT_IMPORTANCE_WEIGHT);
        assert_eq!(weights.momentum, DEFAULT_MOMENTUM_WEIGHT);
        assert_eq!(weights.access_count, DEFAULT_ACCESS_COUNT_WEIGHT);
        assert_eq!(weights.graph_strength, DEFAULT_GRAPH_STRENGTH_WEIGHT);
    }

    #[test]
    fn test_learned_weights_normalize() {
        let mut weights = LearnedWeights {
            semantic: 0.5,
            entity: 0.5,
            tag: 0.5,
            importance: 0.5,
            momentum: 0.5,
            access_count: 0.5,
            graph_strength: 0.5, // CTX-3: added graph_strength
            update_count: 0,
            last_updated: None,
        };

        weights.normalize();

        let sum = weights.semantic
            + weights.entity
            + weights.tag
            + weights.importance
            + weights.momentum
            + weights.access_count
            + weights.graph_strength;
        assert!((sum - 1.0).abs() < 0.001);
        // 0.5/3.5 ≈ 0.143 (7 weights at 0.5 each = 3.5, each normalized = 0.5/3.5 = 1/7)
        assert!((weights.semantic - 1.0 / 7.0).abs() < 0.001);
    }

    #[test]
    fn test_learned_weights_feedback_helpful() {
        let mut weights = LearnedWeights::default();
        let initial_semantic = weights.semantic;
        let initial_entity = weights.entity;

        // Positive feedback on semantic and entity
        weights.apply_feedback(true, true, false, true);

        // Semantic and entity should increase (relative to tag/importance)
        // After normalization, they may not be strictly higher, but update_count should increase
        assert_eq!(weights.update_count, 1);
        assert!(weights.last_updated.is_some());

        // Weights should still sum to 1.0 (includes momentum, access_count, and graph_strength)
        let sum = weights.semantic
            + weights.entity
            + weights.tag
            + weights.importance
            + weights.momentum
            + weights.access_count
            + weights.graph_strength;
        assert!((sum - 1.0).abs() < 0.001);

        // Semantic + entity together should have gained relative weight
        let new_se = weights.semantic + weights.entity;
        let old_se = initial_semantic + initial_entity;
        assert!(new_se >= old_se - 0.1); // Allow for normalization effects
    }

    #[test]
    fn test_learned_weights_feedback_not_helpful() {
        let mut weights = LearnedWeights::default();

        // Negative feedback - semantic was the main signal
        weights.apply_feedback(true, false, false, false);

        // Weights should still sum to 1.0 (includes momentum, access_count, and graph_strength)
        let sum = weights.semantic
            + weights.entity
            + weights.tag
            + weights.importance
            + weights.momentum
            + weights.access_count
            + weights.graph_strength;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calibrate_score() {
        // High score should stay high
        let high = calibrate_score(0.9);
        assert!(high > 0.9);

        // Low score should be reduced more
        let low = calibrate_score(0.1);
        assert!(low < 0.1);

        // Mid-point score
        let mid = calibrate_score(SIGMOID_MIDPOINT);
        assert!((mid - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_score_fusion() {
        let weights = LearnedWeights::default();

        // All high scores with high momentum, access count, and graph strength
        let high = weights.fuse_scores_full(0.9, 0.9, 0.9, 0.9, 0.9, 16, 0.9);
        assert!(high > 0.8, "high score was {}", high);

        // All low scores with negative momentum, no access, and low graph strength
        let low = weights.fuse_scores_full(0.1, 0.1, 0.1, 0.1, -0.9, 0, 0.1);
        assert!(low < 0.3, "low score was {}", low);

        // Mixed scores - result should be between
        let mixed = weights.fuse_scores_full(0.9, 0.1, 0.5, 0.7, 0.0, 2, 0.5);
        assert!(mixed > 0.2 && mixed < 0.8, "mixed score was {}", mixed);

        // Test backwards compatibility - legacy fuse_scores uses neutral momentum, 0 access, and neutral graph
        let legacy = weights.fuse_scores(0.9, 0.9, 0.9, 0.9);
        assert!(legacy > 0.5, "legacy score was {}", legacy); // Lower threshold due to neutral momentum/access/graph
    }

    #[test]
    fn test_tag_score_calculation() {
        let engine = RelevanceEngine::new(Arc::new(crate::embeddings::NeuralNer::new_fallback(
            crate::embeddings::NerConfig::default(),
        )));

        // Exact match
        let score = engine.calculate_tag_score("I love Rust programming", &["rust".to_string()]);
        assert_eq!(score, 1.0);

        // Partial match
        let score = engine
            .calculate_tag_score("Learning Rust", &["rust".to_string(), "python".to_string()]);
        assert_eq!(score, 0.5);

        // No match
        let score = engine.calculate_tag_score("Hello world", &["rust".to_string()]);
        assert_eq!(score, 0.0);

        // Empty tags
        let score = engine.calculate_tag_score("Test", &[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_min_weight_enforcement() {
        let mut weights = LearnedWeights {
            semantic: 0.1,
            entity: MIN_WEIGHT + 0.01, // 0.06
            tag: 0.3,                  // Reduced from 0.5 so sum < 1.0
            importance: 0.1,
            momentum: 0.1,
            access_count: 0.1,
            graph_strength: 0.1,
            update_count: 0,
            last_updated: None,
        };

        // Apply negative feedback on entity (already near minimum)
        weights.apply_feedback(false, true, false, false);

        // Entity should not go below MIN_WEIGHT after feedback + normalization
        // Before normalize: entity = 0.05 (clamped), sum = 0.85
        // After normalize: entity = 0.05 / 0.85 ≈ 0.059 >= 0.05 ✓
        assert!(
            weights.entity >= MIN_WEIGHT,
            "entity {} < MIN_WEIGHT {}",
            weights.entity,
            MIN_WEIGHT
        );

        // Still normalized
        let sum = weights.semantic
            + weights.entity
            + weights.tag
            + weights.importance
            + weights.momentum
            + weights.access_count
            + weights.graph_strength;
        assert!((sum - 1.0).abs() < 0.001);
    }
}
