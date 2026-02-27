//! Implicit Feedback System for Memory Reinforcement
//!
//! Extracts feedback signals from agent behavior without explicit ratings.
//! Uses entity overlap, semantic similarity, and user corrections to
//! determine memory usefulness. Implements momentum-based updates with
//! type-dependent inertia to prevent noise from destabilizing useful memories.

use chrono::{DateTime, Duration, Utc};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, IteratorMode, Options, WriteBatch, DB};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use crate::memory::types::{ExperienceType, MemoryId};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Column family name for feedback data in the shared RocksDB instance
pub(crate) const CF_FEEDBACK: &str = "feedback";

/// Maximum number of recent signals to keep for trend detection
const MAX_RECENT_SIGNALS: usize = 20;

/// Maximum context fingerprints per memory
const MAX_CONTEXT_FINGERPRINTS: usize = 100;

/// Entity overlap thresholds
/// FBK-4: Lowered thresholds so weak signals (0.06-0.15 range) actually affect learning
const OVERLAP_STRONG_THRESHOLD: f32 = 0.4;
const OVERLAP_WEAK_THRESHOLD: f32 = 0.1;

/// Semantic similarity thresholds
/// FBK-4: Lowered to catch more meaningful signals
const SEMANTIC_STRONG_THRESHOLD: f32 = 0.6;
const SEMANTIC_WEAK_THRESHOLD: f32 = 0.3;

/// Signal value multipliers (ACT-R inspired)
const SIGNAL_STRONG_MULTIPLIER: f32 = 0.8;
const SIGNAL_WEAK_MULTIPLIER: f32 = 0.3;
const SIGNAL_NO_OVERLAP_PENALTY: f32 = -0.2; // Strengthened: was -0.1 (FBK-3)
const SIGNAL_NEGATIVE_KEYWORD_PENALTY: f32 = -0.5;

/// Action-based signals (FBK-1, FBK-2)
const SIGNAL_REPETITION_PENALTY: f32 = -0.4; // User asked again = memories failed
const SIGNAL_TOPIC_CHANGE_BOOST: f32 = 0.2; // User moved on = task might be complete
const SIGNAL_IGNORED_PENALTY: f32 = -0.2; // Memory shown but completely unused

/// Weights for combining entity and semantic signals
const ENTITY_WEIGHT: f32 = 0.4;
const SEMANTIC_WEIGHT: f32 = 0.6;

/// Stability adjustment rates
const STABILITY_INCREMENT: f32 = 0.05;
const STABILITY_DECREMENT_MULTIPLIER: f32 = 0.1;

/// Trend detection thresholds
const TREND_IMPROVING_THRESHOLD: f32 = 0.1;
const TREND_DECLINING_THRESHOLD: f32 = -0.1;

/// Time decay constants for momentum (AUD-6)
/// Momentum should decay towards 0 when not reinforced
const DECAY_HALF_LIFE_DAYS: f32 = 14.0; // Half-life of 14 days

/// Negative keywords indicating correction/failure
/// Multi-word phrases checked first (contains match on lowercased text)
const NEGATIVE_KEYWORDS: &[&str] = &[
    // Direct negation / correction
    "wrong",
    "incorrect",
    "not correct",
    "nope",
    // Frustration / repetition
    "not what i meant",
    "that's not right",
    "that's wrong",
    "i already said",
    "i told you",
    "i already told",
    "already mentioned",
    // Irrelevance / unhelpfulness
    "not helpful",
    "not relevant",
    "not useful",
    "irrelevant",
    "useless",
    "doesn't help",
    "didn't help",
    "not related",
    // Failure / broken
    "doesn't work",
    "didn't work",
    "broken",
    "still broken",
    "that failed",
    // Explicit rejection
    "forget that",
    "ignore that",
    "disregard",
    "stop suggesting",
    "don't show",
];

// =============================================================================
// SIGNAL TYPES
// =============================================================================

/// What triggered a feedback signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalTrigger {
    /// Entity overlap between memory and agent response
    EntityOverlap { overlap_ratio: f32 },

    /// Semantic similarity between memory and response
    SemanticSimilarity { similarity: f32 },

    /// Negative keywords detected in user's followup
    NegativeKeywords { keywords: Vec<String> },

    /// User repeated the same question (retrieval failed)
    /// Action: user asked again → memories didn't help
    UserRepetition { similarity: f32 },

    /// Topic changed successfully (task completed)
    /// Action: user moved on → memories may have helped
    TopicChange { similarity: f32 },

    /// Memory was surfaced but completely ignored
    /// Action: response has no relation to memory
    Ignored { overlap_ratio: f32 },

    /// FBK-8: Entity flow tracking
    /// Measures how response builds on memory entities
    /// - derived_ratio: proportion of response entities that came from memory
    /// - novel_ratio: proportion of response entities that are new
    EntityFlow {
        derived_ratio: f32,
        novel_ratio: f32,
        memory_entities_used: usize,
        response_entities_total: usize,
    },
}

/// A single feedback signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRecord {
    /// When the signal was recorded
    pub timestamp: DateTime<Utc>,

    /// Signal value: -1.0 (misleading) to +1.0 (helpful)
    pub value: f32,

    /// Confidence in this signal (0.0 to 1.0)
    pub confidence: f32,

    /// What triggered this signal
    pub trigger: SignalTrigger,
}

impl SignalRecord {
    pub fn new(value: f32, confidence: f32, trigger: SignalTrigger) -> Self {
        Self {
            timestamp: Utc::now(),
            value: value.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            trigger,
        }
    }

    /// Create signal from entity overlap ratio
    pub fn from_entity_overlap(overlap_ratio: f32) -> Self {
        let (value, confidence) = if overlap_ratio >= OVERLAP_STRONG_THRESHOLD {
            (SIGNAL_STRONG_MULTIPLIER * overlap_ratio, 0.9)
        } else if overlap_ratio >= OVERLAP_WEAK_THRESHOLD {
            (SIGNAL_WEAK_MULTIPLIER * overlap_ratio, 0.6)
        } else {
            (SIGNAL_NO_OVERLAP_PENALTY, 0.4)
        };

        Self::new(
            value,
            confidence,
            SignalTrigger::EntityOverlap { overlap_ratio },
        )
    }

    /// Create signal from negative keyword detection
    pub fn from_negative_keywords(keywords: Vec<String>) -> Self {
        Self::new(
            SIGNAL_NEGATIVE_KEYWORD_PENALTY,
            0.95, // High confidence - explicit correction
            SignalTrigger::NegativeKeywords { keywords },
        )
    }
}

// =============================================================================
// TREND DETECTION
// =============================================================================

/// Trend direction for a memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    /// Memory is becoming more useful over time
    Improving,
    /// Memory usefulness is stable
    Stable,
    /// Memory is becoming less useful (possibly outdated)
    Declining,
    /// Not enough data to determine trend
    Insufficient,
}

impl Trend {
    /// Calculate trend from recent signals using linear regression
    pub fn from_signals(signals: &VecDeque<SignalRecord>) -> Self {
        if signals.len() < 3 {
            return Trend::Insufficient;
        }

        let n = signals.len() as f32;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, signal) in signals.iter().enumerate() {
            let x = i as f32;
            let y = signal.value;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        // Linear regression slope: (n*Σxy - Σx*Σy) / (n*Σxx - Σx²)
        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < f32::EPSILON {
            return Trend::Stable;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        if slope > TREND_IMPROVING_THRESHOLD {
            Trend::Improving
        } else if slope < TREND_DECLINING_THRESHOLD {
            Trend::Declining
        } else {
            Trend::Stable
        }
    }
}

// =============================================================================
// CONTEXT FINGERPRINT
// =============================================================================

/// Fingerprint of a context for pattern detection
/// Tracks which contexts a memory was helpful vs misleading in
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFingerprint {
    /// Top entities in the context
    pub entities: Vec<String>,

    /// Compressed embedding signature (top 16 components)
    pub embedding_signature: [f32; 16],

    /// When this context occurred
    pub timestamp: DateTime<Utc>,

    /// Was the memory helpful in this context?
    pub was_helpful: bool,
}

impl ContextFingerprint {
    pub fn new(entities: Vec<String>, embedding: &[f32], was_helpful: bool) -> Self {
        // Compress embedding to 16 components by taking evenly spaced samples
        let mut signature = [0.0f32; 16];
        if !embedding.is_empty() {
            let step = embedding.len() / 16;
            for (i, sig) in signature.iter_mut().enumerate() {
                let idx = (i * step).min(embedding.len() - 1);
                *sig = embedding[idx];
            }
        }

        Self {
            entities,
            embedding_signature: signature,
            timestamp: Utc::now(),
            was_helpful,
        }
    }

    /// Calculate similarity to another fingerprint
    pub fn similarity(&self, other: &ContextFingerprint) -> f32 {
        // Entity Jaccard similarity
        let self_set: HashSet<_> = self.entities.iter().collect();
        let other_set: HashSet<_> = other.entities.iter().collect();
        let intersection = self_set.intersection(&other_set).count() as f32;
        let union = self_set.union(&other_set).count() as f32;
        let entity_sim = if union > 0.0 {
            intersection / union
        } else {
            0.0
        };

        // Embedding cosine similarity
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..16 {
            dot += self.embedding_signature[i] * other.embedding_signature[i];
            norm_a += self.embedding_signature[i] * self.embedding_signature[i];
            norm_b += other.embedding_signature[i] * other.embedding_signature[i];
        }
        let embed_sim = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        } else {
            0.0
        };

        // Weighted combination
        entity_sim * 0.6 + embed_sim * 0.4
    }
}

// =============================================================================
// FEEDBACK MOMENTUM
// =============================================================================

/// Tracks feedback history for a single memory
/// Implements momentum-based updates with type-dependent inertia
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackMomentum {
    /// Memory this momentum belongs to
    pub memory_id: MemoryId,

    /// Memory type (for inertia calculation)
    pub memory_type: ExperienceType,

    /// Exponential moving average of feedback signals
    /// Range: -1.0 (always misleading) to +1.0 (always helpful)
    pub ema: f32,

    /// How many feedback signals have we received?
    pub signal_count: u32,

    /// Stability score: how consistent is the feedback?
    /// High stability = resistant to change
    pub stability: f32,

    /// When did we first evaluate this memory?
    pub first_signal_at: Option<DateTime<Utc>>,

    /// When was the last signal?
    pub last_signal_at: Option<DateTime<Utc>>,

    /// Recent signals for trend detection
    pub recent_signals: VecDeque<SignalRecord>,

    /// Contexts where this memory was helpful
    pub helpful_contexts: Vec<ContextFingerprint>,

    /// Contexts where this memory was misleading
    pub misleading_contexts: Vec<ContextFingerprint>,
}

impl FeedbackMomentum {
    pub fn new(memory_id: MemoryId, memory_type: ExperienceType) -> Self {
        Self {
            memory_id,
            memory_type,
            ema: 0.0,
            signal_count: 0,
            stability: 0.5, // Start neutral
            first_signal_at: None,
            last_signal_at: None,
            recent_signals: VecDeque::with_capacity(MAX_RECENT_SIGNALS),
            helpful_contexts: Vec::new(),
            misleading_contexts: Vec::new(),
        }
    }

    /// Get base inertia for memory type
    /// Higher inertia = more resistant to change
    pub fn base_inertia(&self) -> f32 {
        match self.memory_type {
            ExperienceType::Learning => 0.95,
            ExperienceType::Decision => 0.90,
            ExperienceType::Pattern => 0.85,
            ExperienceType::Discovery => 0.75,
            ExperienceType::Context => 0.60,
            ExperienceType::Task => 0.50,
            ExperienceType::Observation => 0.40,
            ExperienceType::Conversation => 0.30,
            ExperienceType::Error => 0.20,
            // Others default to medium
            ExperienceType::CodeEdit => 0.50,
            ExperienceType::FileAccess => 0.40,
            ExperienceType::Search => 0.35,
            ExperienceType::Command => 0.35,
            ExperienceType::Intention => 0.60,
        }
    }

    /// Calculate age factor for inertia
    /// Older memories are more stable
    pub fn age_factor(&self) -> f32 {
        let age_days = self
            .first_signal_at
            .map(|first| {
                let duration = Utc::now() - first;
                duration.num_days() as f32
            })
            .unwrap_or(0.0);

        if age_days < 1.0 {
            0.8 // New, still malleable
        } else if age_days < 7.0 {
            0.9 // Consolidating
        } else if age_days < 30.0 {
            1.0 // Consolidated
        } else {
            1.1 // Deeply encoded
        }
    }

    /// Calculate history factor for inertia
    /// More evaluations = more confidence = more inertia
    pub fn history_factor(&self) -> f32 {
        match self.signal_count {
            0..=2 => 0.7,   // Not enough data
            3..=9 => 0.9,   // Some history
            10..=49 => 1.0, // Good history
            _ => 1.1,       // Very well tested
        }
    }

    /// Calculate stability factor for inertia
    /// Consistent history = resist change
    pub fn stability_factor(&self) -> f32 {
        // Map stability 0.0-1.0 to factor 0.8-1.2
        0.8 + (self.stability * 0.4)
    }

    /// Calculate effective inertia combining all factors
    pub fn effective_inertia(&self) -> f32 {
        let inertia = self.base_inertia()
            * self.age_factor()
            * self.history_factor()
            * self.stability_factor();

        // Clamp to valid range - never fully frozen, never fully fluid
        inertia.clamp(0.5, 0.99)
    }

    /// Calculate recency weight for a signal
    pub fn recency_weight(&self, signal_time: DateTime<Utc>) -> f32 {
        let time_since_last = self
            .last_signal_at
            .map(|last| signal_time - last)
            .unwrap_or_else(Duration::zero);

        if time_since_last < Duration::hours(1) {
            1.0
        } else if time_since_last < Duration::days(1) {
            0.9
        } else if time_since_last < Duration::days(7) {
            0.7
        } else {
            0.5
        }
    }

    /// Update momentum with a new signal
    pub fn update(&mut self, signal: SignalRecord) {
        let now = signal.timestamp;

        // Initialize first signal time if needed
        if self.first_signal_at.is_none() {
            self.first_signal_at = Some(now);
        }

        // Calculate effective inertia before update
        let effective_inertia = self.effective_inertia();
        let recency = self.recency_weight(now);

        // Alpha = how much new signal affects EMA
        // High inertia = low alpha = resistant to change
        let alpha = (1.0 - effective_inertia) * recency * signal.confidence;

        // Store old EMA for stability calculation
        let old_ema = self.ema;

        // Update EMA
        self.ema = old_ema * (1.0 - alpha) + signal.value * alpha;

        // Update stability
        let direction_matches =
            (signal.value > 0.0) == (old_ema > 0.0) || old_ema.abs() < f32::EPSILON;

        if direction_matches {
            // Consistent feedback: increase stability
            self.stability = (self.stability + STABILITY_INCREMENT).min(1.0);
        } else {
            // Contradictory feedback: decrease stability
            let contradiction_strength = (signal.value - old_ema).abs();
            self.stability =
                (self.stability - STABILITY_DECREMENT_MULTIPLIER * contradiction_strength).max(0.0);
        }

        // Record signal
        self.recent_signals.push_back(signal);
        if self.recent_signals.len() > MAX_RECENT_SIGNALS {
            self.recent_signals.pop_front();
        }

        self.signal_count += 1;
        self.last_signal_at = Some(now);
    }

    /// Get current trend
    pub fn trend(&self) -> Trend {
        Trend::from_signals(&self.recent_signals)
    }

    /// Add context fingerprint
    pub fn add_context(&mut self, fingerprint: ContextFingerprint) {
        let target = if fingerprint.was_helpful {
            &mut self.helpful_contexts
        } else {
            &mut self.misleading_contexts
        };

        target.push(fingerprint);

        // Trim to max size, keeping most recent
        if target.len() > MAX_CONTEXT_FINGERPRINTS {
            target.remove(0);
        }
    }

    /// Check if current context matches helpful pattern
    pub fn matches_helpful_pattern(&self, current: &ContextFingerprint) -> Option<f32> {
        self.helpful_contexts
            .iter()
            .map(|fp| fp.similarity(current))
            .max_by(|a, b| a.total_cmp(b))
    }

    /// Check if current context matches misleading pattern
    pub fn matches_misleading_pattern(&self, current: &ContextFingerprint) -> Option<f32> {
        self.misleading_contexts
            .iter()
            .map(|fp| fp.similarity(current))
            .max_by(|a, b| a.total_cmp(b))
    }

    /// Apply time-based decay to momentum (AUD-6)
    /// Returns the decayed EMA value without mutating the struct.
    /// Momentum decays towards 0 when not reinforced by feedback.
    pub fn ema_with_decay(&self) -> f32 {
        let days_since_last = self
            .last_signal_at
            .map(|last| {
                let duration = Utc::now() - last;
                duration.num_hours() as f32 / 24.0
            })
            .unwrap_or(0.0);

        if days_since_last < 0.1 {
            // Very recent signal, no decay
            return self.ema;
        }

        // Exponential decay with half-life
        // decay_factor = 0.5^(days / half_life)
        let decay_factor = 0.5_f32.powf(days_since_last / DECAY_HALF_LIFE_DAYS);

        // Decay towards 0
        self.ema * decay_factor
    }
}

// =============================================================================
// PENDING FEEDBACK
// =============================================================================

/// Information about a surfaced memory awaiting feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacedMemoryInfo {
    pub id: MemoryId,
    pub entities: HashSet<String>,
    pub content_preview: String,
    pub score: f32,
    /// Memory embedding for semantic similarity feedback
    #[serde(default)]
    pub embedding: Vec<f32>,
}

/// Pending feedback for a user - tracks what was surfaced, awaiting response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingFeedback {
    pub user_id: String,
    pub surfaced_at: DateTime<Utc>,
    pub surfaced_memories: Vec<SurfacedMemoryInfo>,
    pub context: String,
    pub context_embedding: Vec<f32>,
}

impl PendingFeedback {
    pub fn new(
        user_id: String,
        context: String,
        context_embedding: Vec<f32>,
        memories: Vec<SurfacedMemoryInfo>,
    ) -> Self {
        Self {
            user_id,
            surfaced_at: Utc::now(),
            surfaced_memories: memories,
            context,
            context_embedding,
        }
    }

    /// Check if this pending feedback has expired (older than 1 hour)
    pub fn is_expired(&self) -> bool {
        Utc::now() - self.surfaced_at > Duration::hours(1)
    }
}

// =============================================================================
// SIGNAL EXTRACTION
// =============================================================================

/// Extract entities from text using simple word extraction
/// TODO: Use NER model for better extraction
pub fn extract_entities_simple(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|word| word.len() > 2)
        .map(|s| s.to_string())
        .collect()
}

/// Calculate entity overlap between memory entities and response entities
pub fn calculate_entity_overlap(
    memory_entities: &HashSet<String>,
    response_entities: &HashSet<String>,
) -> f32 {
    if memory_entities.is_empty() {
        return 0.0;
    }

    let intersection = memory_entities.intersection(response_entities).count() as f32;
    intersection / memory_entities.len() as f32
}

/// Calculate cosine similarity between two embedding vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// Create signal from semantic similarity
fn signal_from_semantic_similarity(similarity: f32) -> (f32, f32) {
    if similarity >= SEMANTIC_STRONG_THRESHOLD {
        (SIGNAL_STRONG_MULTIPLIER * similarity, 0.9)
    } else if similarity >= SEMANTIC_WEAK_THRESHOLD {
        (SIGNAL_WEAK_MULTIPLIER * similarity, 0.6)
    } else {
        (SIGNAL_NO_OVERLAP_PENALTY * 0.5, 0.3) // Lighter penalty for semantic
    }
}

/// Detect negative keywords in user's followup message
pub fn detect_negative_keywords(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    NEGATIVE_KEYWORDS
        .iter()
        .filter(|&&kw| lower.contains(kw))
        .map(|&s| s.to_string())
        .collect()
}

/// FBK-8: Calculate entity flow between memory and response
///
/// Tracks how the response builds on memory entities:
/// - derived_ratio: How many response entities came from the memory (0.0 to 1.0)
/// - novel_ratio: How many response entities are new/not from memory (0.0 to 1.0)
///
/// High derived_ratio = response uses memory knowledge = positive signal
/// High novel_ratio with low derived = memory might not have been relevant
pub fn calculate_entity_flow(
    memory_entities: &HashSet<String>,
    response_entities: &HashSet<String>,
) -> (f32, f32, usize, usize) {
    if response_entities.is_empty() {
        return (0.0, 0.0, 0, 0);
    }

    // Count how many response entities came from the memory
    let derived: HashSet<_> = response_entities
        .intersection(memory_entities)
        .cloned()
        .collect();
    let derived_count = derived.len();

    // Count novel entities (in response but not in memory)
    let novel_count = response_entities.len() - derived_count;

    let derived_ratio = derived_count as f32 / response_entities.len() as f32;
    let novel_ratio = novel_count as f32 / response_entities.len() as f32;

    (
        derived_ratio,
        novel_ratio,
        derived_count,
        response_entities.len(),
    )
}

/// FBK-8: Create signal from entity flow analysis
pub fn signal_from_entity_flow(
    derived_ratio: f32,
    novel_ratio: f32,
    memory_entities_used: usize,
    response_entities_total: usize,
) -> SignalRecord {
    // Signal value based on how much the response builds on memory
    // High derived ratio = memory was useful
    // Low derived ratio with high novel = memory might be irrelevant
    let value = if derived_ratio >= 0.5 {
        // Response heavily uses memory entities - strong positive
        0.6 + (derived_ratio - 0.5) * 0.4
    } else if derived_ratio >= 0.2 {
        // Response somewhat uses memory entities - weak positive
        derived_ratio * 1.5
    } else if novel_ratio >= 0.8 {
        // Response mostly novel, memory barely used - slight negative
        -0.1
    } else {
        // Mixed - neutral
        0.0
    };

    let confidence = if response_entities_total >= 3 {
        0.8 // Good sample size
    } else {
        0.5 // Small sample, lower confidence
    };

    SignalRecord::new(
        value,
        confidence,
        SignalTrigger::EntityFlow {
            derived_ratio,
            novel_ratio,
            memory_entities_used,
            response_entities_total,
        },
    )
}

/// Process feedback for surfaced memories based on agent response
/// Uses both entity overlap and semantic similarity for more accurate signals
pub fn process_implicit_feedback(
    pending: &PendingFeedback,
    response_text: &str,
    user_followup: Option<&str>,
) -> Vec<(MemoryId, SignalRecord)> {
    // For backwards compatibility, call enhanced version with no response embedding
    process_implicit_feedback_with_semantics(pending, response_text, user_followup, None)
}

/// Enhanced feedback processing using both entity overlap and semantic similarity
///
/// When response_embedding is provided, combines entity overlap (40%) with
/// semantic similarity (60%) for a more robust feedback signal. This helps
/// detect when a memory was genuinely useful vs just sharing some words.
pub fn process_implicit_feedback_with_semantics(
    pending: &PendingFeedback,
    response_text: &str,
    user_followup: Option<&str>,
    response_embedding: Option<&[f32]>,
) -> Vec<(MemoryId, SignalRecord)> {
    let response_entities = extract_entities_simple(response_text);
    let mut signals = Vec::new();

    // Calculate combined signals for each memory
    for memory in &pending.surfaced_memories {
        // Entity overlap signal
        let entity_overlap = calculate_entity_overlap(&memory.entities, &response_entities);
        let (entity_value, entity_conf) = if entity_overlap >= OVERLAP_STRONG_THRESHOLD {
            (SIGNAL_STRONG_MULTIPLIER * entity_overlap, 0.9)
        } else if entity_overlap >= OVERLAP_WEAK_THRESHOLD {
            (SIGNAL_WEAK_MULTIPLIER * entity_overlap, 0.6)
        } else {
            (SIGNAL_NO_OVERLAP_PENALTY, 0.4)
        };

        // Semantic similarity signal (if embeddings available)
        let (semantic_value, semantic_conf, has_semantic) =
            if let Some(resp_emb) = response_embedding {
                if !memory.embedding.is_empty() {
                    let similarity = cosine_similarity(&memory.embedding, resp_emb);
                    let (val, conf) = signal_from_semantic_similarity(similarity);
                    (val, conf, true)
                } else {
                    (0.0, 0.0, false)
                }
            } else {
                (0.0, 0.0, false)
            };

        // Combine signals with weights
        let (combined_value, combined_confidence, trigger) = if has_semantic {
            let value = (ENTITY_WEIGHT * entity_value) + (SEMANTIC_WEIGHT * semantic_value);
            let confidence = (ENTITY_WEIGHT * entity_conf) + (SEMANTIC_WEIGHT * semantic_conf);

            // Use semantic similarity as trigger since it's the primary signal
            let similarity = if let Some(resp_emb) = response_embedding {
                cosine_similarity(&memory.embedding, resp_emb)
            } else {
                0.0
            };
            (
                value,
                confidence,
                SignalTrigger::SemanticSimilarity { similarity },
            )
        } else {
            // Fallback to entity-only signal
            (
                entity_value,
                entity_conf,
                SignalTrigger::EntityOverlap {
                    overlap_ratio: entity_overlap,
                },
            )
        };

        let mut signal = SignalRecord::new(combined_value, combined_confidence, trigger);

        // Apply negative keyword penalty if detected in followup
        if let Some(followup) = user_followup {
            let negative = detect_negative_keywords(followup);
            if !negative.is_empty() {
                signal.value += SIGNAL_NEGATIVE_KEYWORD_PENALTY;
                signal.value = signal.value.clamp(-1.0, 1.0);
                signal.confidence = 0.95; // High confidence on explicit correction
            }
        }

        signals.push((memory.id.clone(), signal));
    }

    signals
}

/// Apply context pattern signals (repetition/topic change) to existing signals
///
/// This function modifies signal values based on detected user actions:
/// - Repetition (user asked same thing again): negative signal (memories failed)
/// - Topic change (user moved on): positive signal (task might be complete)
/// - Ignored (memory shown but no overlap): negative signal
///
/// # Arguments
/// - `signals`: Existing signals from process_implicit_feedback
/// - `is_repetition`: User is asking the same question again
/// - `is_topic_change`: User has moved to a different topic
/// - `context_similarity`: Similarity between current and previous context
pub fn apply_context_pattern_signals(
    signals: &mut [(MemoryId, SignalRecord)],
    is_repetition: bool,
    is_topic_change: bool,
    _context_similarity: f32,
) {
    for (memory_id, signal) in signals.iter_mut() {
        if is_repetition {
            // User asked the same thing again - memories didn't help
            // Apply penalty proportional to how irrelevant the memory was
            // FBK-4: Lowered threshold from 0.3 to 0.15 so more signals affect learning
            if signal.value < 0.15 {
                // Memory wasn't used in response AND user is re-asking
                signal.value += SIGNAL_REPETITION_PENALTY;
                signal.value = signal.value.clamp(-1.0, 1.0);
                signal.trigger = SignalTrigger::UserRepetition {
                    similarity: _context_similarity,
                };
                signal.confidence = 0.85; // High confidence - clear action signal
                tracing::debug!(
                    "Repetition detected for memory {:?}: applied penalty",
                    memory_id
                );
            }
        } else if is_topic_change {
            // User moved on to different topic - task might be complete
            // Apply boost to memories that were used in the response
            // FBK-4: Lowered threshold from 0.1 to 0.05 so more signals affect learning
            if signal.value > 0.05 {
                // Memory was somewhat used - boost it
                signal.value += SIGNAL_TOPIC_CHANGE_BOOST;
                signal.value = signal.value.clamp(-1.0, 1.0);
                signal.trigger = SignalTrigger::TopicChange {
                    similarity: _context_similarity,
                };
                signal.confidence = 0.7; // Moderate confidence
                tracing::debug!(
                    "Topic change detected for memory {:?}: applied boost",
                    memory_id
                );
            }
        }

        // Apply ignored penalty for memories with very low overlap
        // regardless of repetition/topic change
        if signal.value < -0.05 && signal.value > -0.3 {
            // Memory was surfaced but not used - strengthen the penalty
            signal.value = SIGNAL_IGNORED_PENALTY.min(signal.value);
            if !matches!(signal.trigger, SignalTrigger::UserRepetition { .. }) {
                signal.trigger = SignalTrigger::Ignored {
                    overlap_ratio: match &signal.trigger {
                        SignalTrigger::EntityOverlap { overlap_ratio } => *overlap_ratio,
                        _ => 0.0,
                    },
                };
            }
        }
    }
}

// =============================================================================
// FEEDBACK STORE
// =============================================================================

/// Previous context for a user - used for repetition/topic change detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviousContext {
    /// The query/context text
    pub context: String,
    /// Embedding of the context for similarity comparison
    pub embedding: Vec<f32>,
    /// When this context was recorded
    pub timestamp: DateTime<Utc>,
    /// Memory IDs that were surfaced for this context
    pub surfaced_memory_ids: Vec<MemoryId>,
}

/// Persistent store for feedback momentum with in-memory cache
pub struct FeedbackStore {
    /// In-memory cache: memory_id -> FeedbackMomentum
    pub momentum: HashMap<MemoryId, FeedbackMomentum>,

    /// Pending feedback per user: user_id -> PendingFeedback (in-memory only)
    pending: HashMap<String, PendingFeedback>,

    /// Previous context per user: for repetition/topic change detection
    /// Tracks what the user asked last time to detect patterns
    previous_context: HashMap<String, PreviousContext>,

    /// Persistent storage for momentum data
    db: Option<Arc<DB>>,

    /// Track dirty entries that need persistence
    dirty: HashSet<MemoryId>,
}

impl std::fmt::Debug for FeedbackStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedbackStore")
            .field("momentum_count", &self.momentum.len())
            .field("pending_count", &self.pending.len())
            .field("previous_context_count", &self.previous_context.len())
            .field("has_db", &self.db.is_some())
            .field("dirty_count", &self.dirty.len())
            .finish()
    }
}

impl Default for FeedbackStore {
    fn default() -> Self {
        Self {
            momentum: HashMap::new(),
            pending: HashMap::new(),
            previous_context: HashMap::new(),
            db: None,
            dirty: HashSet::new(),
        }
    }
}

impl FeedbackStore {
    /// Create in-memory only store (no persistence)
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a reference to the feedback column family handle.
    /// Returns `None` when running in-memory only or when the CF is missing.
    fn feedback_cf(&self) -> Option<&ColumnFamily> {
        self.db.as_ref().and_then(|db| db.cf_handle(CF_FEEDBACK))
    }

    /// Create persistent store backed by a shared RocksDB instance.
    ///
    /// The caller is responsible for opening the DB with the `CF_FEEDBACK` column
    /// family already declared. On first use this constructor migrates data from
    /// the legacy standalone `feedback/` DB directory into the shared CF.
    pub fn with_shared_db(db: Arc<DB>, base_path: &Path) -> anyhow::Result<Self> {
        Self::migrate_from_separate_db(base_path, &db)?;

        let cf = db.cf_handle(CF_FEEDBACK).expect("feedback CF must exist");

        // Load all momentum entries from the feedback CF
        let mut momentum = HashMap::new();
        let iter = db.prefix_iterator_cf(cf, b"momentum:");
        for item in iter {
            if let Ok((key, value)) = item {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with("momentum:") {
                        break;
                    }
                    if let Ok(m) = serde_json::from_slice::<FeedbackMomentum>(&value) {
                        momentum.insert(m.memory_id.clone(), m);
                    }
                }
            }
        }

        let mut pending = HashMap::new();
        let iter = db.prefix_iterator_cf(cf, b"pending:");
        for item in iter {
            if let Ok((key, value)) = item {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with("pending:") {
                        break;
                    }
                    if let Ok(p) = serde_json::from_slice::<PendingFeedback>(&value) {
                        if !p.is_expired() {
                            pending.insert(p.user_id.clone(), p);
                        } else {
                            let _ = db.delete_cf(cf, key_str.as_bytes());
                        }
                    }
                }
            }
        }

        let mut previous_context = HashMap::new();
        let iter = db.prefix_iterator_cf(cf, b"prev_ctx:");
        for item in iter {
            if let Ok((key, value)) = item {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with("prev_ctx:") {
                        break;
                    }
                    if let Ok(ctx) = serde_json::from_slice::<PreviousContext>(&value) {
                        let user_id = key_str.strip_prefix("prev_ctx:").unwrap_or("");
                        previous_context.insert(user_id.to_string(), ctx);
                    }
                }
            }
        }

        tracing::info!(
            "Loaded {} momentum, {} pending, {} previous context from shared feedback CF",
            momentum.len(),
            pending.len(),
            previous_context.len()
        );

        Ok(Self {
            momentum,
            pending,
            previous_context,
            db: Some(db),
            dirty: HashSet::new(),
        })
    }

    /// Migrate data from the legacy standalone `feedback/` RocksDB directory
    /// into the `CF_FEEDBACK` column family of the shared DB.
    ///
    /// The old directory is renamed to `feedback.pre_cf_migration` so it can be
    /// restored manually if needed.
    fn migrate_from_separate_db(base_path: &Path, db: &DB) -> anyhow::Result<()> {
        let old_dir = base_path.join("feedback");
        if !old_dir.is_dir() {
            return Ok(());
        }

        let cf = db.cf_handle(CF_FEEDBACK).expect("feedback CF must exist");
        let old_opts = Options::default();
        match DB::open_for_read_only(&old_opts, &old_dir, false) {
            Ok(old_db) => {
                let mut batch = WriteBatch::default();
                let mut count = 0usize;
                for item in old_db.iterator(IteratorMode::Start) {
                    if let Ok((key, value)) = item {
                        batch.put_cf(cf, &key, &value);
                        count += 1;
                        if count % 10_000 == 0 {
                            db.write(std::mem::take(&mut batch))?;
                        }
                    }
                }
                if !batch.is_empty() {
                    db.write(batch)?;
                }
                drop(old_db);
                tracing::info!("  feedback: migrated {count} entries to {CF_FEEDBACK} CF");

                let backup = base_path.join("feedback.pre_cf_migration");
                if let Err(e) = std::fs::rename(&old_dir, &backup) {
                    tracing::warn!("Could not rename old feedback dir: {e}");
                }
            }
            Err(e) => tracing::warn!("Could not open old feedback DB for migration: {e}"),
        }
        Ok(())
    }

    /// Create persistent store with its own standalone RocksDB instance.
    ///
    /// Primarily useful for tests and standalone operation. In production, prefer
    /// [`with_shared_db`](Self::with_shared_db) to share a single DB instance.
    pub fn with_persistence<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        let cfs = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
            ColumnFamilyDescriptor::new(CF_FEEDBACK, {
                let mut cf_opts = Options::default();
                cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                cf_opts
            }),
        ];
        let db = DB::open_cf_descriptors(&opts, path.as_ref(), cfs)?;
        let db = Arc::new(db);

        let cf = db.cf_handle(CF_FEEDBACK).expect("feedback CF must exist");

        // Load all momentum entries from the feedback CF
        let mut momentum = HashMap::new();
        let iter = db.prefix_iterator_cf(cf, b"momentum:");
        for item in iter {
            if let Ok((key, value)) = item {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with("momentum:") {
                        break;
                    }
                    if let Ok(m) = serde_json::from_slice::<FeedbackMomentum>(&value) {
                        momentum.insert(m.memory_id.clone(), m);
                    }
                }
            }
        }

        // Also load pending feedback entries (filter expired ones)
        let mut pending = HashMap::new();
        let iter = db.prefix_iterator_cf(cf, b"pending:");
        for item in iter {
            if let Ok((key, value)) = item {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with("pending:") {
                        break;
                    }
                    if let Ok(p) = serde_json::from_slice::<PendingFeedback>(&value) {
                        if !p.is_expired() {
                            pending.insert(p.user_id.clone(), p);
                        } else {
                            // Clean up expired pending feedback from disk
                            let _ = db.delete_cf(cf, key_str.as_bytes());
                        }
                    }
                }
            }
        }

        // Load previous context entries
        let mut previous_context = HashMap::new();
        let iter = db.prefix_iterator_cf(cf, b"prev_ctx:");
        for item in iter {
            if let Ok((key, value)) = item {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with("prev_ctx:") {
                        break;
                    }
                    if let Ok(ctx) = serde_json::from_slice::<PreviousContext>(&value) {
                        let user_id = key_str.strip_prefix("prev_ctx:").unwrap_or("");
                        previous_context.insert(user_id.to_string(), ctx);
                    }
                }
            }
        }

        tracing::info!(
            "Loaded {} momentum, {} pending, {} previous context from feedback CF",
            momentum.len(),
            pending.len(),
            previous_context.len()
        );

        Ok(Self {
            momentum,
            pending,
            previous_context,
            db: Some(db),
            dirty: HashSet::new(),
        })
    }

    /// Get or create momentum for a memory
    pub fn get_or_create_momentum(
        &mut self,
        memory_id: MemoryId,
        memory_type: ExperienceType,
    ) -> &mut FeedbackMomentum {
        // Check if we need to load from disk
        if !self.momentum.contains_key(&memory_id) {
            if let (Some(db), Some(cf)) = (&self.db, self.feedback_cf()) {
                let key = format!("momentum:{}", memory_id.0);
                if let Ok(Some(data)) = db.get_cf(cf, key.as_bytes()) {
                    if let Ok(m) = serde_json::from_slice::<FeedbackMomentum>(&data) {
                        self.momentum.insert(memory_id.clone(), m);
                    }
                }
            }
        }

        self.momentum.entry(memory_id.clone()).or_insert_with(|| {
            self.dirty.insert(memory_id.clone());
            FeedbackMomentum::new(memory_id, memory_type)
        })
    }

    /// Get momentum for a memory (if exists in-memory), with disk fallback.
    /// Checks the in-memory HashMap first, then falls back to RocksDB.
    pub fn get_momentum(&self, memory_id: &MemoryId) -> Option<FeedbackMomentum> {
        if let Some(m) = self.momentum.get(memory_id) {
            return Some(m.clone());
        }
        // Fall back to disk lookup
        if let (Some(db), Some(cf)) = (&self.db, self.feedback_cf()) {
            let key = format!("momentum:{}", memory_id.0);
            if let Ok(Some(data)) = db.get_cf(cf, key.as_bytes()) {
                if let Ok(m) = serde_json::from_slice::<FeedbackMomentum>(&data) {
                    return Some(m);
                }
            }
        }
        None
    }

    /// Mark a memory as dirty (needs persistence)
    pub fn mark_dirty(&mut self, memory_id: &MemoryId) {
        self.dirty.insert(memory_id.clone());
    }

    /// Set pending feedback for a user (also persists to disk)
    pub fn set_pending(&mut self, pending: PendingFeedback) {
        let user_id = pending.user_id.clone();
        self.pending.insert(user_id.clone(), pending.clone());

        // Persist to disk
        if let (Some(db), Some(cf)) = (&self.db, self.feedback_cf()) {
            let key = format!("pending:{}", user_id);
            if let Ok(value) = serde_json::to_vec(&pending) {
                if let Err(e) = db.put_cf(cf, key.as_bytes(), &value) {
                    tracing::warn!("Failed to persist pending feedback: {}", e);
                }
            }
        }
    }

    /// Take pending feedback for a user (removes from store and disk)
    pub fn take_pending(&mut self, user_id: &str) -> Option<PendingFeedback> {
        let result = self.pending.remove(user_id);

        // Remove from disk
        if let (Some(db), Some(cf)) = (&self.db, self.feedback_cf()) {
            let key = format!("pending:{}", user_id);
            let _ = db.delete_cf(cf, key.as_bytes());
        }

        result
    }

    /// Get pending feedback for a user (without removing)
    pub fn get_pending(&self, user_id: &str) -> Option<&PendingFeedback> {
        self.pending.get(user_id)
    }

    /// Clean up expired pending feedback
    pub fn cleanup_expired(&mut self) {
        self.pending.retain(|_, p| !p.is_expired());
    }

    /// Set previous context for a user (for repetition/topic change detection)
    /// Called when memories are surfaced to track what the user asked
    pub fn set_previous_context(
        &mut self,
        user_id: &str,
        context: String,
        embedding: Vec<f32>,
        surfaced_memory_ids: Vec<MemoryId>,
    ) {
        let prev_ctx = PreviousContext {
            context,
            embedding,
            timestamp: Utc::now(),
            surfaced_memory_ids,
        };

        self.previous_context
            .insert(user_id.to_string(), prev_ctx.clone());

        // Persist to disk
        if let (Some(db), Some(cf)) = (&self.db, self.feedback_cf()) {
            let key = format!("prev_ctx:{}", user_id);
            if let Ok(value) = serde_json::to_vec(&prev_ctx) {
                if let Err(e) = db.put_cf(cf, key.as_bytes(), &value) {
                    tracing::warn!("Failed to persist previous context: {}", e);
                }
            }
        }
    }

    /// Get previous context for a user
    pub fn get_previous_context(&self, user_id: &str) -> Option<&PreviousContext> {
        self.previous_context.get(user_id)
    }

    /// Compare current context to previous and detect action patterns
    /// Returns: (is_repetition, is_topic_change, similarity)
    /// - Repetition: similarity > 0.8 means user is asking same thing again (memories failed)
    /// - Topic change: similarity < 0.3 means user moved on (task might be complete)
    pub fn detect_context_pattern(
        &self,
        user_id: &str,
        current_embedding: &[f32],
    ) -> Option<(bool, bool, f32)> {
        let prev = self.previous_context.get(user_id)?;

        if prev.embedding.is_empty() || current_embedding.is_empty() {
            return None;
        }

        let similarity = cosine_similarity(&prev.embedding, current_embedding);

        // ACT-R inspired thresholds
        let is_repetition = similarity > 0.8; // High similarity = re-asking
        let is_topic_change = similarity < 0.3; // Low similarity = moved on

        Some((is_repetition, is_topic_change, similarity))
    }

    /// Flush dirty entries to disk and ensure WAL is persisted
    pub fn flush(&mut self) -> anyhow::Result<usize> {
        let Some(ref db) = self.db else {
            return Ok(0);
        };
        let Some(cf) = db.cf_handle(CF_FEEDBACK) else {
            return Ok(0);
        };

        // Drain dirty set first so the mutable borrow is released before we
        // take shared references to self.momentum / self.pending below.
        let dirty: Vec<MemoryId> = self.dirty.drain().collect();

        let mut flushed = 0;
        for memory_id in &dirty {
            if let Some(momentum) = self.momentum.get(memory_id) {
                let key = format!("momentum:{}", memory_id.0);
                let value = serde_json::to_vec(momentum)?;
                db.put_cf(cf, key.as_bytes(), &value)?;
                flushed += 1;
            }
        }

        // Also persist any pending feedback entries
        for (user_id, pending) in &self.pending {
            let key = format!("pending:{}", user_id);
            let value = serde_json::to_vec(pending)?;
            db.put_cf(cf, key.as_bytes(), &value)?;
        }

        // Flush the feedback CF to ensure data persistence (critical for graceful shutdown)
        use rocksdb::FlushOptions;
        let mut flush_opts = FlushOptions::default();
        flush_opts.set_wait(true);
        db.flush_cf_opt(cf, &flush_opts)
            .map_err(|e| anyhow::anyhow!("Failed to flush feedback CF: {e}"))?;

        if flushed > 0 {
            tracing::debug!("Flushed {} feedback momentum entries to disk", flushed);
        }

        Ok(flushed)
    }

    /// Get reference to the RocksDB database for backup (if available)
    pub fn database(&self) -> Option<&Arc<DB>> {
        self.db.as_ref()
    }

    /// Get statistics
    pub fn stats(&self) -> FeedbackStoreStats {
        FeedbackStoreStats {
            total_momentum_entries: self.momentum.len(),
            total_pending: self.pending.len(),
            avg_ema: if self.momentum.is_empty() {
                0.0
            } else {
                self.momentum.values().map(|m| m.ema).sum::<f32>() / self.momentum.len() as f32
            },
            avg_stability: if self.momentum.is_empty() {
                0.0
            } else {
                self.momentum.values().map(|m| m.stability).sum::<f32>()
                    / self.momentum.len() as f32
            },
        }
    }
}

/// Statistics about the feedback store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackStoreStats {
    pub total_momentum_entries: usize,
    pub total_pending: usize,
    pub avg_ema: f32,
    pub avg_stability: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_signal_from_entity_overlap() {
        // Strong overlap (>= 0.4 after FBK-4 threshold adjustment)
        let signal = SignalRecord::from_entity_overlap(0.7);
        assert!(signal.value > 0.5);
        assert!(signal.confidence > 0.8);

        // Weak overlap (>= 0.1 after FBK-4 threshold adjustment)
        let signal = SignalRecord::from_entity_overlap(0.3);
        assert!(signal.value > 0.0);
        assert!(signal.value < 0.5);

        // No overlap (< 0.1 after FBK-4 threshold adjustment)
        let signal = SignalRecord::from_entity_overlap(0.05);
        assert!(signal.value < 0.0);
    }

    #[test]
    fn test_momentum_inertia_by_type() {
        let learning = FeedbackMomentum::new(MemoryId(Uuid::new_v4()), ExperienceType::Learning);
        let conversation =
            FeedbackMomentum::new(MemoryId(Uuid::new_v4()), ExperienceType::Conversation);

        assert!(learning.base_inertia() > conversation.base_inertia());
        assert!(learning.base_inertia() >= 0.9);
        assert!(conversation.base_inertia() <= 0.4);
    }

    #[test]
    fn test_momentum_update_with_inertia() {
        let mut momentum = FeedbackMomentum::new(
            MemoryId(Uuid::new_v4()),
            ExperienceType::Learning, // High inertia
        );

        // Apply positive signal
        momentum.update(SignalRecord::new(
            1.0,
            1.0,
            SignalTrigger::EntityOverlap { overlap_ratio: 1.0 },
        ));

        // EMA should move slowly due to high inertia
        assert!(momentum.ema > 0.0);
        assert!(momentum.ema < 0.5); // Not too fast

        // Apply many positive signals
        for _ in 0..20 {
            momentum.update(SignalRecord::new(
                1.0,
                1.0,
                SignalTrigger::EntityOverlap { overlap_ratio: 1.0 },
            ));
        }

        // Now EMA should be higher
        assert!(momentum.ema > 0.5);
        // Stability should be high after consistent signals
        assert!(momentum.stability > 0.7);
    }

    #[test]
    fn test_trend_detection() {
        let mut signals = VecDeque::new();

        // Not enough data
        assert_eq!(Trend::from_signals(&signals), Trend::Insufficient);

        // Add improving signals (steeper slope > 0.1 threshold)
        for i in 0..10 {
            signals.push_back(SignalRecord::new(
                i as f32 * 0.15, // 0, 0.15, 0.3, ... gives slope ~0.15
                1.0,
                SignalTrigger::TopicChange { similarity: 0.2 },
            ));
        }
        assert_eq!(Trend::from_signals(&signals), Trend::Improving);

        // Add declining signals (steeper slope < -0.1 threshold)
        signals.clear();
        for i in (0..10).rev() {
            signals.push_back(SignalRecord::new(
                i as f32 * 0.15, // 1.35, 1.2, ... 0 gives slope ~-0.15
                1.0,
                SignalTrigger::TopicChange { similarity: 0.2 },
            ));
        }
        assert_eq!(Trend::from_signals(&signals), Trend::Declining);
    }

    #[test]
    fn test_entity_overlap() {
        let memory: HashSet<String> = ["rust", "async", "tokio"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let response: HashSet<String> = ["rust", "tokio", "spawn"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let overlap = calculate_entity_overlap(&memory, &response);
        assert!((overlap - 0.666).abs() < 0.01); // 2/3
    }

    #[test]
    fn test_negative_keyword_detection() {
        // Multi-word phrase detection
        let text = "No, that's not what I meant";
        let keywords = detect_negative_keywords(text);
        assert!(keywords.contains(&"not what i meant".to_string()));

        // Irrelevance signals
        let text2 = "That's not helpful at all, it's irrelevant";
        let keywords2 = detect_negative_keywords(text2);
        assert!(keywords2.contains(&"not helpful".to_string()));
        assert!(keywords2.contains(&"irrelevant".to_string()));

        // Explicit rejection
        let text3 = "Please forget that, it doesn't work";
        let keywords3 = detect_negative_keywords(text3);
        assert!(keywords3.contains(&"forget that".to_string()));
        assert!(keywords3.contains(&"doesn't work".to_string()));

        // No false positives on neutral text
        let text4 = "Can you help me debug this function?";
        let keywords4 = detect_negative_keywords(text4);
        assert!(keywords4.is_empty());
    }

    #[test]
    fn test_feedback_store_pending() {
        let mut store = FeedbackStore::new();
        let user_id = "test-user";

        // Initially no pending
        assert!(store.get_pending(user_id).is_none());

        // Set pending feedback
        let pending = PendingFeedback::new(
            user_id.to_string(),
            "test context".to_string(),
            vec![0.1; 384],
            vec![SurfacedMemoryInfo {
                id: MemoryId(Uuid::new_v4()),
                entities: ["rust", "memory"].iter().map(|s| s.to_string()).collect(),
                content_preview: "Test memory".to_string(),
                score: 0.8,
                embedding: Vec::new(),
            }],
        );
        store.set_pending(pending);

        // Should have pending now
        assert!(store.get_pending(user_id).is_some());
        assert_eq!(
            store.get_pending(user_id).unwrap().surfaced_memories.len(),
            1
        );

        // Take should remove it
        let taken = store.take_pending(user_id);
        assert!(taken.is_some());
        assert!(store.get_pending(user_id).is_none());
    }

    #[test]
    fn test_feedback_store_momentum() {
        let mut store = FeedbackStore::new();
        let memory_id = MemoryId(Uuid::new_v4());

        // Get or create momentum
        let momentum = store.get_or_create_momentum(memory_id.clone(), ExperienceType::Context);
        assert_eq!(momentum.signal_count, 0);
        assert_eq!(momentum.ema, 0.0);

        // Update it
        momentum.update(SignalRecord::new(
            0.8,
            1.0,
            SignalTrigger::EntityOverlap { overlap_ratio: 0.8 },
        ));
        assert!(momentum.ema > 0.0);
        assert_eq!(momentum.signal_count, 1);

        // Get should return existing
        let momentum2 = store.get_momentum(&memory_id);
        assert!(momentum2.is_some());
        assert_eq!(momentum2.unwrap().signal_count, 1);
    }

    #[test]
    fn test_process_implicit_feedback_full() {
        let memory_id1 = MemoryId(Uuid::new_v4());
        let memory_id2 = MemoryId(Uuid::new_v4());

        let pending = PendingFeedback::new(
            "user1".to_string(),
            "How do I use async in Rust?".to_string(),
            vec![0.1; 384],
            vec![
                SurfacedMemoryInfo {
                    id: memory_id1.clone(),
                    entities: ["rust", "async", "tokio"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    content_preview: "Rust async with tokio".to_string(),
                    score: 0.9,
                    embedding: Vec::new(),
                },
                SurfacedMemoryInfo {
                    id: memory_id2.clone(),
                    entities: ["python", "django"].iter().map(|s| s.to_string()).collect(),
                    content_preview: "Python Django web".to_string(),
                    score: 0.3,
                    embedding: Vec::new(),
                },
            ],
        );

        // Response that uses Rust async terminology
        let response =
            "To use async in Rust, you can use tokio runtime. Here is an example with async await.";
        let signals = process_implicit_feedback(&pending, response, None);

        assert_eq!(signals.len(), 2);

        // First memory should have positive signal (high entity overlap)
        let (id1, sig1) = &signals[0];
        assert_eq!(id1, &memory_id1);
        assert!(sig1.value > 0.0);

        // Second memory should have negative/low signal (no overlap)
        let (id2, sig2) = &signals[1];
        assert_eq!(id2, &memory_id2);
        assert!(sig2.value <= 0.0);
    }

    #[test]
    fn test_process_implicit_feedback_with_negative_keywords() {
        let memory_id = MemoryId(Uuid::new_v4());

        let pending = PendingFeedback::new(
            "user1".to_string(),
            "How do I use async?".to_string(),
            vec![0.1; 384],
            vec![SurfacedMemoryInfo {
                id: memory_id.clone(),
                entities: ["async", "code"].iter().map(|s| s.to_string()).collect(),
                content_preview: "Async code".to_string(),
                score: 0.9,
                embedding: Vec::new(),
            }],
        );

        // Response uses entities
        let response = "Here is the async code pattern";

        // Process without negative keywords
        let signals1 = process_implicit_feedback(&pending, response, None);
        let value_without = signals1[0].1.value;

        // Process with negative keywords in followup
        let signals2 = process_implicit_feedback(&pending, response, Some("No, that is wrong!"));
        let value_with = signals2[0].1.value;

        // Negative keywords should decrease the signal
        assert!(value_with < value_without);
    }

    #[test]
    fn test_context_fingerprint_similarity() {
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let fp1 = ContextFingerprint::new(
            vec!["rust".to_string(), "memory".to_string()],
            &embedding,
            true,
        );
        let fp2 = ContextFingerprint::new(
            vec!["rust".to_string(), "async".to_string()],
            &embedding,
            false,
        );
        let different_embedding: Vec<f32> = (0..384).map(|i| 1.0 - (i as f32) * 0.01).collect();
        let fp3 = ContextFingerprint::new(
            vec!["python".to_string(), "django".to_string()],
            &different_embedding,
            true,
        );

        // fp1 and fp2 share "rust" entity and same embedding
        let sim12 = fp1.similarity(&fp2);
        // fp1 and fp3 have no entity overlap and different embedding
        let sim13 = fp1.similarity(&fp3);

        assert!(sim12 > sim13);
    }

    #[test]
    fn test_feedback_store_stats() {
        let mut store = FeedbackStore::new();

        // Empty stats
        let stats = store.stats();
        assert_eq!(stats.total_momentum_entries, 0);
        assert_eq!(stats.total_pending, 0);

        // Add some momentum entries
        for i in 0..5 {
            let mut momentum =
                FeedbackMomentum::new(MemoryId(Uuid::new_v4()), ExperienceType::Context);
            momentum.ema = i as f32 * 0.2; // 0, 0.2, 0.4, 0.6, 0.8
            store.momentum.insert(momentum.memory_id.clone(), momentum);
        }

        let stats = store.stats();
        assert_eq!(stats.total_momentum_entries, 5);
        assert!((stats.avg_ema - 0.4).abs() < 0.01); // (0+0.2+0.4+0.6+0.8)/5 = 0.4
    }

    #[test]
    fn test_process_feedback_with_semantic_similarity() {
        let memory_id1 = MemoryId(Uuid::new_v4());
        let memory_id2 = MemoryId(Uuid::new_v4());

        // Create embeddings: similar embeddings for related content
        let rust_embedding: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let python_embedding: Vec<f32> = (0..384).map(|i| 1.0 - (i as f32) * 0.01).collect();

        let pending = PendingFeedback::new(
            "user1".to_string(),
            "How do I use async in Rust?".to_string(),
            vec![0.1; 384],
            vec![
                SurfacedMemoryInfo {
                    id: memory_id1.clone(),
                    entities: ["rust", "async", "tokio"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    content_preview: "Rust async with tokio".to_string(),
                    score: 0.9,
                    embedding: rust_embedding.clone(),
                },
                SurfacedMemoryInfo {
                    id: memory_id2.clone(),
                    entities: ["python", "django"].iter().map(|s| s.to_string()).collect(),
                    content_preview: "Python Django web".to_string(),
                    score: 0.3,
                    embedding: python_embedding.clone(),
                },
            ],
        );

        // Response embedding similar to rust_embedding
        let response = "Here is how to use async/await in Rust with tokio runtime.";
        let response_embedding = rust_embedding; // Similar to memory 1

        // Process without semantic (backwards compat)
        let signals_entity_only = process_implicit_feedback(&pending, response, None);

        // Process with semantic similarity
        let signals_with_semantic = process_implicit_feedback_with_semantics(
            &pending,
            response,
            None,
            Some(&response_embedding),
        );

        // First memory should score higher with semantic (response embedding matches memory embedding)
        let (id1, _sig1_entity) = &signals_entity_only[0];
        let (_, sig1_semantic) = &signals_with_semantic[0];
        assert_eq!(id1, &memory_id1);

        // Semantic signal should use SemanticSimilarity trigger
        match &sig1_semantic.trigger {
            SignalTrigger::SemanticSimilarity { similarity } => {
                assert!(*similarity > 0.9); // High similarity since embeddings are same
            }
            _ => panic!("Expected SemanticSimilarity trigger"),
        }

        // Second memory (python) should have low semantic score since embedding is different
        let (id2, sig2_semantic) = &signals_with_semantic[1];
        assert_eq!(id2, &memory_id2);
        match &sig2_semantic.trigger {
            SignalTrigger::SemanticSimilarity { similarity } => {
                assert!(*similarity < 0.5); // Low similarity - different embeddings
            }
            _ => panic!("Expected SemanticSimilarity trigger"),
        }
    }

    #[test]
    fn test_cosine_similarity_basic() {
        // Identical vectors = 1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors = 0.0
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        // Opposite vectors = -1.0
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);

        // Empty vectors = 0.0
        assert!((cosine_similarity(&[], &[]) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_entity_flow() {
        use std::collections::HashSet;

        // Case 1: Response heavily derived from memory
        let memory_entities: HashSet<String> = ["rust", "async", "tokio", "futures"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let response_entities: HashSet<String> = ["rust", "async", "tokio", "runtime"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let (derived_ratio, novel_ratio, derived_count, total) =
            calculate_entity_flow(&memory_entities, &response_entities);

        assert_eq!(derived_count, 3); // rust, async, tokio
        assert_eq!(total, 4);
        assert!((derived_ratio - 0.75).abs() < 0.01);
        assert!((novel_ratio - 0.25).abs() < 0.01);

        // Case 2: Response mostly novel (memory not used)
        let response_novel: HashSet<String> = ["python", "django", "flask", "web"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let (derived_ratio2, novel_ratio2, derived_count2, _) =
            calculate_entity_flow(&memory_entities, &response_novel);

        assert_eq!(derived_count2, 0);
        assert!((derived_ratio2 - 0.0).abs() < 0.01);
        assert!((novel_ratio2 - 1.0).abs() < 0.01);

        // Case 3: Empty response
        let empty: HashSet<String> = HashSet::new();
        let (dr, nr, dc, total) = calculate_entity_flow(&memory_entities, &empty);
        assert_eq!(dc, 0);
        assert_eq!(total, 0);
        assert!((dr - 0.0).abs() < 0.01);
        assert!((nr - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_signal_from_entity_flow() {
        // Case 1: High derived ratio (>=0.5) = strong positive
        let sig1 = signal_from_entity_flow(0.75, 0.25, 3, 4);
        assert!(sig1.value > 0.5); // Strong positive
        assert!((sig1.confidence - 0.8).abs() < 0.01); // Good sample size

        // Case 2: Medium derived ratio (0.2 to 0.5) = weak positive
        let sig2 = signal_from_entity_flow(0.3, 0.7, 1, 4);
        assert!(sig2.value > 0.0 && sig2.value <= 0.5); // Weak positive
        assert!((sig2.confidence - 0.8).abs() < 0.01);

        // Case 3: Low derived, high novel = slight negative
        let sig3 = signal_from_entity_flow(0.1, 0.9, 0, 4);
        assert!(sig3.value < 0.0); // Negative
        assert!((sig3.value - (-0.1)).abs() < 0.01);

        // Case 4: Small sample size = lower confidence
        let sig4 = signal_from_entity_flow(0.5, 0.5, 1, 2);
        assert!((sig4.confidence - 0.5).abs() < 0.01);

        // Verify trigger variant
        match sig1.trigger {
            SignalTrigger::EntityFlow {
                derived_ratio,
                novel_ratio,
                memory_entities_used,
                response_entities_total,
            } => {
                assert!((derived_ratio - 0.75).abs() < 0.01);
                assert!((novel_ratio - 0.25).abs() < 0.01);
                assert_eq!(memory_entities_used, 3);
                assert_eq!(response_entities_total, 4);
            }
            _ => panic!("Expected EntityFlow trigger"),
        }
    }
}
