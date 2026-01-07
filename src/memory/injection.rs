//! Proactive Context Injection System
//!
//! Implements truly proactive memory injection - surfacing relevant memories
//! without explicit agent action. Based on multi-signal relevance scoring
//! with user-adaptive thresholds.
//!
//! # Enhanced Relevance Model (MEMO-1)
//!
//! ```text
//! R(m, c) = α·semantic + β·recency + γ·strength + δ·entity_overlap + ε·type_boost + ζ·file_match - η·suppression
//! ```
//!
//! Where:
//! - semantic: cosine similarity between memory and context embeddings
//! - recency: exponential decay based on memory age
//! - strength: Hebbian edge weight from knowledge graph
//! - entity_overlap: Jaccard similarity of entities between memory and context
//! - type_boost: Weight bonus based on memory type (Decision > Learning > Context)
//! - file_match: Boost when memory mentions files in current context
//! - suppression: Penalty for memories with negative feedback momentum
//!
//! # Feedback Loop
//!
//! The system learns from implicit feedback:
//! - Positive: injected memory referenced in next turn
//! - Negative: user indicates irrelevance
//! - Neutral: memory ignored (no adjustment)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::types::{ExperienceType, MemoryId};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Weights for composite relevance scoring
///
/// Enhanced with entity_overlap, type_boost, file_match, and suppression (MEMO-1).
/// New fields default to 0.0 for backwards compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceWeights {
    /// Weight for semantic similarity (cosine distance)
    pub semantic: f32,
    /// Weight for recency (exponential decay)
    pub recency: f32,
    /// Weight for Hebbian strength from graph
    pub strength: f32,
    /// Weight for entity overlap between memory and context (MEMO-1)
    #[serde(default)]
    pub entity_overlap: f32,
    /// Weight for memory type boost (Decision > Learning > Context) (MEMO-1)
    #[serde(default)]
    pub type_boost: f32,
    /// Weight for file path matching (MEMO-1)
    #[serde(default)]
    pub file_match: f32,
    /// Weight for negative feedback suppression (MEMO-1)
    #[serde(default)]
    pub suppression: f32,
    /// Weight for episode coherence boost (SHO-temporal)
    /// Memories from the same episode as the query get boosted
    #[serde(default)]
    pub episode_coherence: f32,
    /// Weight for graph activation from spreading activation traversal
    /// Higher activation = stronger association in knowledge graph
    #[serde(default)]
    pub graph_activation: f32,
    /// Weight for linguistic score from query analysis
    /// Focal entity matches, modifier matches, etc.
    #[serde(default)]
    pub linguistic_score: f32,
}

impl Default for RelevanceWeights {
    fn default() -> Self {
        Self {
            semantic: 0.40,           // Primary signal - semantic similarity
            recency: 0.08,            // Recent memories get boost
            strength: 0.08,           // Hebbian edge weight (from graph)
            entity_overlap: 0.08,     // Entity Jaccard similarity
            type_boost: 0.06,         // Decision/Learning type boost
            file_match: 0.04,         // File path matching
            suppression: 0.02,        // Negative feedback penalty
            episode_coherence: 0.06,  // Same-episode boost (prevents bleeding)
            graph_activation: 0.10,   // Spreading activation from graph traversal
            linguistic_score: 0.08,   // Query analysis (focal entities, modifiers)
        }
    }
}

impl RelevanceWeights {
    /// Legacy weights for backwards compatibility (original 3-signal model)
    pub fn legacy() -> Self {
        Self {
            semantic: 0.5,
            recency: 0.3,
            strength: 0.2,
            entity_overlap: 0.0,
            type_boost: 0.0,
            file_match: 0.0,
            suppression: 0.0,
            episode_coherence: 0.0,
            graph_activation: 0.0,
            linguistic_score: 0.0,
        }
    }
}

/// Configuration for proactive injection behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionConfig {
    /// Minimum relevance score to trigger injection (0.0 - 1.0)
    pub min_relevance: f32,

    /// Maximum memories to inject per message
    pub max_per_message: usize,

    /// Cooldown in seconds before re-injecting same memory
    pub cooldown_seconds: u64,

    /// Weights for relevance score components
    pub weights: RelevanceWeights,

    /// Decay rate for recency calculation (λ in e^(-λt))
    /// Higher = faster decay. Default 0.01 means ~50% at 70 hours
    pub recency_decay_rate: f32,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            min_relevance: 0.50, // Raised from 0.35 - require stronger semantic match
            max_per_message: 3,
            cooldown_seconds: 180,
            weights: RelevanceWeights::default(),
            recency_decay_rate: 0.01,
        }
    }
}

impl InjectionConfig {
    /// Legacy config for backwards compatibility
    pub fn legacy() -> Self {
        Self {
            min_relevance: 0.70,
            max_per_message: 3,
            cooldown_seconds: 180,
            weights: RelevanceWeights::legacy(),
            recency_decay_rate: 0.01,
        }
    }
}

// =============================================================================
// RELEVANCE SCORING
// =============================================================================

/// Input data for computing relevance of a memory
///
/// Enhanced with optional fields for entity overlap, memory type, file paths,
/// and feedback momentum (MEMO-1). Old callers can omit new fields.
#[derive(Debug, Clone, Default)]
pub struct RelevanceInput {
    /// Memory's embedding vector
    pub memory_embedding: Vec<f32>,
    /// Memory creation timestamp
    pub created_at: DateTime<Utc>,
    /// Hebbian strength from knowledge graph (0.0 - 1.0)
    pub hebbian_strength: f32,
    /// Memory entities for overlap calculation (MEMO-1)
    pub memory_entities: Vec<String>,
    /// Context entities for overlap calculation (MEMO-1)
    pub context_entities: Vec<String>,
    /// Memory type for type boost (MEMO-1)
    pub memory_type: Option<ExperienceType>,
    /// File paths mentioned in memory content (MEMO-1)
    pub memory_files: Vec<String>,
    /// File paths active in current context (MEMO-1)
    pub context_files: Vec<String>,
    /// Feedback momentum EMA - negative values indicate often-ignored (MEMO-1)
    pub feedback_momentum: f32,
    /// Memory's episode ID (SHO-temporal) - for episode coherence scoring
    pub episode_id: Option<String>,
    /// Query's episode ID (SHO-temporal) - for episode coherence scoring
    pub query_episode_id: Option<String>,
    /// Memory's sequence number within episode (SHO-temporal) - for temporal ordering
    pub sequence_number: Option<u32>,
    /// Graph activation level from spreading activation traversal (0.0 - 1.0)
    /// Higher values mean stronger association via knowledge graph
    pub graph_activation: f32,
    /// Linguistic score from query analysis (0.0 - 1.0)
    /// Measures focal entity matches, modifier matches, etc.
    pub linguistic_score: f32,
}

impl RelevanceInput {
    /// Create a basic input with only the original 3 fields (backwards compat)
    pub fn basic(
        memory_embedding: Vec<f32>,
        created_at: DateTime<Utc>,
        hebbian_strength: f32,
    ) -> Self {
        Self {
            memory_embedding,
            created_at,
            hebbian_strength,
            ..Default::default()
        }
    }
}

/// Compute composite relevance score for a memory
///
/// Uses all available signals from RelevanceInput. Missing optional data
/// contributes 0 to that signal component.
///
/// # Arguments
/// * `input` - Memory data for scoring
/// * `context_embedding` - Current context embedding
/// * `now` - Current timestamp
/// * `config` - Injection configuration with weights
///
/// # Returns
/// Relevance score in range [0.0, 1.0]
/// Minimum semantic similarity required for any relevance score
/// This prevents high-scoring memories with zero semantic relevance
const SEMANTIC_FLOOR: f32 = 0.30;

pub fn compute_relevance(
    input: &RelevanceInput,
    context_embedding: &[f32],
    now: DateTime<Utc>,
    config: &InjectionConfig,
) -> f32 {
    let w = &config.weights;

    // Original signals
    let semantic = cosine_similarity(&input.memory_embedding, context_embedding);

    // Semantic floor gate: if semantic similarity is below threshold,
    // return 0 regardless of other signals. This prevents irrelevant
    // memories from being surfaced due to high recency or strength alone.
    if semantic < SEMANTIC_FLOOR {
        return 0.0;
    }

    let hours_old = (now - input.created_at).num_hours().max(0) as f32;
    let recency = (-config.recency_decay_rate * hours_old).exp();
    let strength = input.hebbian_strength;

    // Enhanced signals (MEMO-1)
    let entity_overlap = compute_entity_overlap(&input.memory_entities, &input.context_entities);
    let type_boost = compute_type_boost(input.memory_type.as_ref());
    let file_match = compute_file_match(&input.memory_files, &input.context_files);
    let suppression = compute_suppression(input.feedback_momentum);

    // Episode coherence (SHO-temporal) - prevents episode bleeding
    let episode_coherence = compute_episode_coherence(
        input.episode_id.as_ref(),
        input.query_episode_id.as_ref(),
    );

    // Graph and linguistic signals (unified scoring)
    let graph_activation = input.graph_activation;
    let linguistic_score = input.linguistic_score;

    // Weighted sum - all signals contribute to final relevance
    let score = w.semantic * semantic
        + w.recency * recency
        + w.strength * strength
        + w.entity_overlap * entity_overlap
        + w.type_boost * type_boost
        + w.file_match * file_match
        + w.episode_coherence * episode_coherence
        + w.graph_activation * graph_activation
        + w.linguistic_score * linguistic_score
        - w.suppression * suppression;

    score.clamp(0.0, 1.0)
}

/// Compute Jaccard similarity between two entity sets
fn compute_entity_overlap(memory_entities: &[String], context_entities: &[String]) -> f32 {
    if memory_entities.is_empty() || context_entities.is_empty() {
        return 0.0;
    }

    let memory_set: HashSet<&str> = memory_entities.iter().map(|s| s.as_str()).collect();
    let context_set: HashSet<&str> = context_entities.iter().map(|s| s.as_str()).collect();

    let intersection = memory_set.intersection(&context_set).count();
    let union = memory_set.union(&context_set).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Compute type boost based on memory type
/// Decision memories are most valuable, then Learning, then others
fn compute_type_boost(memory_type: Option<&ExperienceType>) -> f32 {
    match memory_type {
        Some(ExperienceType::Decision) => 1.0,
        Some(ExperienceType::Learning) => 0.8,
        Some(ExperienceType::Error) => 0.7,
        Some(ExperienceType::Pattern) => 0.6,
        Some(ExperienceType::Discovery) => 0.5,
        Some(ExperienceType::Context) => 0.4,
        Some(ExperienceType::Task) => 0.3,
        Some(ExperienceType::CodeEdit) => 0.3,
        Some(ExperienceType::Conversation) => 0.2,
        Some(ExperienceType::Observation) => 0.1,
        _ => 0.0,
    }
}

/// Compute file match score - 1.0 if any memory file matches context files
fn compute_file_match(memory_files: &[String], context_files: &[String]) -> f32 {
    if memory_files.is_empty() || context_files.is_empty() {
        return 0.0;
    }

    // Normalize file paths for comparison (extract filename)
    let memory_names: HashSet<&str> = memory_files
        .iter()
        .filter_map(|p| p.rsplit(['/', '\\']).next())
        .collect();
    let context_names: HashSet<&str> = context_files
        .iter()
        .filter_map(|p| p.rsplit(['/', '\\']).next())
        .collect();

    let matches = memory_names.intersection(&context_names).count();
    if matches > 0 {
        1.0
    } else {
        // Partial credit for directory match
        let memory_dirs: HashSet<&str> = memory_files
            .iter()
            .filter_map(|p| {
                let parts: Vec<&str> = p.split(['/', '\\']).collect();
                if parts.len() > 1 {
                    Some(parts[parts.len() - 2])
                } else {
                    None
                }
            })
            .collect();
        let context_dirs: HashSet<&str> = context_files
            .iter()
            .filter_map(|p| {
                let parts: Vec<&str> = p.split(['/', '\\']).collect();
                if parts.len() > 1 {
                    Some(parts[parts.len() - 2])
                } else {
                    None
                }
            })
            .collect();

        if memory_dirs.intersection(&context_dirs).count() > 0 {
            0.3
        } else {
            0.0
        }
    }
}

/// Compute suppression penalty from feedback momentum
/// Negative momentum = often ignored → higher suppression
fn compute_suppression(feedback_momentum: f32) -> f32 {
    if feedback_momentum >= 0.0 {
        0.0 // No suppression for positive/neutral momentum
    } else {
        // Map negative momentum to suppression: -1.0 → 1.0 suppression
        (-feedback_momentum).clamp(0.0, 1.0)
    }
}

/// Compute episode coherence score (SHO-temporal)
/// Returns 1.0 if memory and query share the same episode_id, 0.0 otherwise
/// This prevents episode bleeding where unrelated memories mix into results
fn compute_episode_coherence(
    memory_episode_id: Option<&String>,
    query_episode_id: Option<&String>,
) -> f32 {
    match (memory_episode_id, query_episode_id) {
        (Some(mem), Some(query)) if mem == query => 1.0, // Same episode: full coherence
        (Some(_), None) | (None, Some(_)) => 0.0,        // Mismatched: no boost
        (None, None) => 0.0,                              // Both missing: no boost
        _ => 0.0,                                         // Different episodes: no boost
    }
}

/// Cosine similarity between two vectors (public for use by streaming module)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Extract file paths from text content
pub fn extract_file_paths(content: &str) -> Vec<String> {
    let mut paths = Vec::new();

    // Common file path patterns
    let patterns = [
        r"(?:^|\s)([A-Za-z]:)?(?:[/\\][\w.-]+)+\.[a-zA-Z]{1,10}(?:\s|$|:|\))",
        r"src/[\w/.-]+\.\w+",
        r"[\w.-]+\.(?:rs|ts|tsx|js|jsx|py|go|java|cpp|c|h|md|json|yaml|yml|toml)",
    ];

    for pattern in patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            for cap in re.find_iter(content) {
                let path = cap.as_str().trim();
                if !path.is_empty() && path.len() < 200 {
                    paths.push(path.to_string());
                }
            }
        }
    }

    paths
}

// =============================================================================
// INJECTION ENGINE
// =============================================================================

/// Candidate memory for injection with computed relevance
#[derive(Debug, Clone)]
pub struct InjectionCandidate {
    pub memory_id: MemoryId,
    pub relevance_score: f32,
}

/// Engine that decides which memories to inject
pub struct InjectionEngine {
    config: InjectionConfig,
    /// Tracks last injection time per memory for cooldown
    cooldowns: HashMap<MemoryId, Instant>,
}

impl InjectionEngine {
    pub fn new(config: InjectionConfig) -> Self {
        Self {
            config,
            cooldowns: HashMap::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(InjectionConfig::default())
    }

    /// Check if a memory is on cooldown
    fn on_cooldown(&self, memory_id: &MemoryId) -> bool {
        if let Some(last) = self.cooldowns.get(memory_id) {
            last.elapsed().as_secs() < self.config.cooldown_seconds
        } else {
            false
        }
    }

    /// Select memories for injection from candidates
    ///
    /// Filters by:
    /// 1. Minimum relevance threshold
    /// 2. Cooldown (recently injected memories excluded)
    /// 3. Max count limit
    ///
    /// Returns memory IDs sorted by relevance (highest first)
    pub fn select_for_injection(
        &mut self,
        mut candidates: Vec<InjectionCandidate>,
    ) -> Vec<MemoryId> {
        // Sort by relevance descending
        candidates.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected: Vec<MemoryId> = candidates
            .into_iter()
            .filter(|c| {
                c.relevance_score >= self.config.min_relevance && !self.on_cooldown(&c.memory_id)
            })
            .take(self.config.max_per_message)
            .map(|c| c.memory_id)
            .collect();

        // Record injection time for cooldown
        let now = Instant::now();
        for id in &selected {
            self.cooldowns.insert(id.clone(), now);
        }

        selected
    }

    /// Clear expired cooldowns to prevent memory leak
    pub fn cleanup_cooldowns(&mut self) {
        let threshold = self.config.cooldown_seconds;
        self.cooldowns
            .retain(|_, last| last.elapsed().as_secs() < threshold * 2);
    }

    /// Get current configuration
    pub fn config(&self) -> &InjectionConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: InjectionConfig) {
        self.config = config;
    }
}

// =============================================================================
// INJECTION TRACKING (for feedback loop)
// =============================================================================

/// Record of an injection for feedback tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionRecord {
    pub memory_id: MemoryId,
    pub injected_at: DateTime<Utc>,
    pub relevance_score: f32,
    pub context_signature: u64,
}

/// Tracks injections for feedback learning
#[derive(Debug, Default)]
pub struct InjectionTracker {
    /// Recent injections awaiting feedback
    pending: Vec<InjectionRecord>,
    /// Max pending records to keep
    max_pending: usize,
}

impl InjectionTracker {
    pub fn new(max_pending: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_pending,
        }
    }

    /// Record a new injection
    pub fn record_injection(
        &mut self,
        memory_id: MemoryId,
        relevance_score: f32,
        context_signature: u64,
    ) {
        let record = InjectionRecord {
            memory_id,
            injected_at: Utc::now(),
            relevance_score,
            context_signature,
        };

        self.pending.push(record);

        // Trim old records
        if self.pending.len() > self.max_pending {
            self.pending.remove(0);
        }
    }

    /// Get pending injections for feedback analysis
    pub fn pending_injections(&self) -> &[InjectionRecord] {
        &self.pending
    }

    /// Clear injections older than given duration
    pub fn clear_old(&mut self, max_age_seconds: i64) {
        let cutoff = Utc::now() - chrono::Duration::seconds(max_age_seconds);
        self.pending.retain(|r| r.injected_at > cutoff);
    }

    /// Remove specific injection after feedback processed
    pub fn mark_processed(&mut self, memory_id: &MemoryId) {
        self.pending.retain(|r| &r.memory_id != memory_id);
    }
}

// =============================================================================
// USER PROFILE (adaptive thresholds)
// =============================================================================

/// Feedback signal type for learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackSignal {
    /// Memory was referenced/used - lower threshold
    Positive,
    /// Memory was explicitly rejected - raise threshold
    Negative,
    /// Memory was ignored - no change
    Neutral,
}

/// Per-user adaptive injection profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInjectionProfile {
    pub user_id: String,
    /// Effective threshold (starts at default, adapts over time)
    pub effective_threshold: f32,
    /// Count of positive signals received
    pub positive_signals: u32,
    /// Count of negative signals received
    pub negative_signals: u32,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl UserInjectionProfile {
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            effective_threshold: InjectionConfig::default().min_relevance,
            positive_signals: 0,
            negative_signals: 0,
            updated_at: Utc::now(),
        }
    }

    /// Adjust threshold based on feedback signal
    ///
    /// - Positive: lower threshold by 0.01 (min 0.50)
    /// - Negative: raise threshold by 0.02 (max 0.90)
    /// - Neutral: no change
    ///
    /// Asymmetric adjustment: we're more cautious about noise
    pub fn adjust(&mut self, signal: FeedbackSignal) {
        match signal {
            FeedbackSignal::Positive => {
                self.positive_signals += 1;
                self.effective_threshold = (self.effective_threshold - 0.01).max(0.50);
            }
            FeedbackSignal::Negative => {
                self.negative_signals += 1;
                self.effective_threshold = (self.effective_threshold + 0.02).min(0.90);
            }
            FeedbackSignal::Neutral => {}
        }
        self.updated_at = Utc::now();
    }

    /// Get signal ratio (positive / total)
    pub fn signal_ratio(&self) -> f32 {
        let total = self.positive_signals + self.negative_signals;
        if total == 0 {
            0.5 // No data yet
        } else {
            self.positive_signals as f32 / total as f32
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_compute_relevance_legacy() {
        let config = InjectionConfig::legacy();
        let now = Utc::now();

        let input = RelevanceInput::basic(vec![1.0, 0.0, 0.0], now, 0.8);

        let context = vec![1.0, 0.0, 0.0]; // Perfect match
        let score = compute_relevance(&input, &context, now, &config);

        // semantic=1.0, recency=1.0 (just created), strength=0.8
        // 0.5*1.0 + 0.3*1.0 + 0.2*0.8 = 0.5 + 0.3 + 0.16 = 0.96
        assert!(score > 0.9);
    }

    #[test]
    fn test_compute_relevance_enhanced() {
        let config = InjectionConfig::default();
        let now = Utc::now();

        let input = RelevanceInput {
            memory_embedding: vec![1.0, 0.0, 0.0],
            created_at: now,
            hebbian_strength: 0.8,
            memory_entities: vec!["shodh".to_string(), "memory".to_string()],
            context_entities: vec!["shodh".to_string(), "api".to_string()],
            memory_type: Some(ExperienceType::Decision),
            memory_files: vec!["src/main.rs".to_string()],
            context_files: vec!["src/main.rs".to_string()],
            feedback_momentum: 0.2,
            episode_id: Some("test-episode".to_string()),
            query_episode_id: Some("test-episode".to_string()),
            sequence_number: Some(1),
            graph_activation: 0.5,
            linguistic_score: 0.3,
        };

        let context = vec![1.0, 0.0, 0.0];
        let score = compute_relevance(&input, &context, now, &config);

        // Should score higher than legacy due to entity overlap, type boost, file match
        assert!(score > 0.8);
    }

    #[test]
    fn test_entity_overlap() {
        let memory = vec!["shodh".to_string(), "memory".to_string(), "api".to_string()];
        let context = vec!["shodh".to_string(), "api".to_string()];

        let overlap = compute_entity_overlap(&memory, &context);
        // Intersection: {shodh, api} = 2, Union: {shodh, memory, api} = 3
        assert!((overlap - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_type_boost() {
        assert!((compute_type_boost(Some(&ExperienceType::Decision)) - 1.0).abs() < 0.01);
        assert!((compute_type_boost(Some(&ExperienceType::Learning)) - 0.8).abs() < 0.01);
        assert!((compute_type_boost(Some(&ExperienceType::Observation)) - 0.1).abs() < 0.01);
        assert!((compute_type_boost(None) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_file_match() {
        let memory_files = vec!["src/main.rs".to_string()];
        let context_files = vec!["src/main.rs".to_string()];
        assert!((compute_file_match(&memory_files, &context_files) - 1.0).abs() < 0.01);

        let context_files2 = vec!["src/lib.rs".to_string()];
        assert!((compute_file_match(&memory_files, &context_files2) - 0.3).abs() < 0.01); // Same dir

        let context_files3 = vec!["tests/test.rs".to_string()];
        assert!((compute_file_match(&memory_files, &context_files3) - 0.0).abs() < 0.01);
        // Different dir
    }

    #[test]
    fn test_suppression() {
        assert!((compute_suppression(0.5) - 0.0).abs() < 0.01);
        assert!((compute_suppression(-0.5) - 0.5).abs() < 0.01);
        assert!((compute_suppression(-1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_injection_engine_filtering() {
        let mut engine = InjectionEngine::with_default_config();

        let candidates = vec![
            InjectionCandidate {
                memory_id: MemoryId(Uuid::new_v4()),
                relevance_score: 0.85,
            },
            InjectionCandidate {
                memory_id: MemoryId(Uuid::new_v4()),
                relevance_score: 0.45, // Below threshold (0.50)
            },
            InjectionCandidate {
                memory_id: MemoryId(Uuid::new_v4()),
                relevance_score: 0.75,
            },
        ];

        let selected = engine.select_for_injection(candidates);

        assert_eq!(selected.len(), 2); // Only 0.85 and 0.75 pass threshold (0.50)
    }

    #[test]
    fn test_user_profile_adjustment() {
        let mut profile = UserInjectionProfile::new("test-user".to_string());

        assert_eq!(profile.effective_threshold, 0.50);

        profile.adjust(FeedbackSignal::Positive);
        assert!((profile.effective_threshold - 0.49).abs() < 0.01);

        profile.adjust(FeedbackSignal::Negative);
        assert!((profile.effective_threshold - 0.51).abs() < 0.01);

        // Many negatives should cap at 0.90
        for _ in 0..20 {
            profile.adjust(FeedbackSignal::Negative);
        }
        assert_eq!(profile.effective_threshold, 0.90);
    }
}
