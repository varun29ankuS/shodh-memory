//! Graph Memory System - Inspired by Graphiti
//!
//! Temporal knowledge graph for tracking entities, relationships, and episodic memories.
//! Implements bi-temporal tracking and hybrid retrieval (semantic + graph traversal).

use anyhow::Result;
use chrono::{DateTime, Utc};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, WriteBatch, DB};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use crate::constants::{ENTITY_CONCEPT_MERGE_THRESHOLD, LTP_MIN_STRENGTH, LTP_PRUNE_FLOOR};

// Column family names for the unified graph database
const CF_ENTITIES: &str = "entities";
const CF_RELATIONSHIPS: &str = "relationships";
const CF_EPISODES: &str = "episodes";
const CF_ENTITY_EDGES: &str = "entity_edges";
const CF_ENTITY_PAIR_INDEX: &str = "entity_pair_index";
const CF_ENTITY_EPISODES: &str = "entity_episodes";
const CF_NAME_INDEX: &str = "name_index";
const CF_LOWERCASE_INDEX: &str = "lowercase_index";
const CF_STEMMED_INDEX: &str = "stemmed_index";

const GRAPH_CF_NAMES: &[&str] = &[
    CF_ENTITIES,
    CF_RELATIONSHIPS,
    CF_EPISODES,
    CF_ENTITY_EDGES,
    CF_ENTITY_PAIR_INDEX,
    CF_ENTITY_EPISODES,
    CF_NAME_INDEX,
    CF_LOWERCASE_INDEX,
    CF_STEMMED_INDEX,
];

/// Entity node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    /// Unique identifier
    pub uuid: Uuid,

    /// Entity name (e.g., "John", "Paris", "Rust programming")
    pub name: String,

    /// Entity labels/types (e.g., ["Person"], ["Location", "City"])
    pub labels: Vec<EntityLabel>,

    /// When this entity was first created in the graph
    pub created_at: DateTime<Utc>,

    /// When this entity was last observed
    pub last_seen_at: DateTime<Utc>,

    /// How many times this entity has been mentioned
    pub mention_count: usize,

    /// Summary of this entity's context (built from surrounding edges)
    pub summary: String,

    /// Additional attributes based on entity type
    pub attributes: HashMap<String, String>,

    /// Semantic embedding of the entity name (for similarity search)
    pub name_embedding: Option<Vec<f32>>,

    /// Salience score (0.0 - 1.0): How important is this entity?
    /// Higher salience = larger gravitational well in the memory universe
    /// Factors: proper noun status, mention frequency, recency, user-defined importance
    #[serde(default = "default_salience")]
    pub salience: f32,

    /// Whether this is a proper noun (names, places, products)
    /// Proper nouns have higher base salience than common nouns
    #[serde(default)]
    pub is_proper_noun: bool,
}

fn default_salience() -> f32 {
    0.5 // Default middle salience
}

/// Entity labels following Graphiti's categorization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityLabel {
    Person,
    Organization,
    Location,
    Technology,
    Concept,
    Event,
    Date,
    Product,
    Skill,
    /// YAKE-extracted discriminative keyword (not a named entity)
    /// Used for graph-based retrieval of rare/important terms like "sunrise"
    Keyword,
    Other(String),
}

impl EntityLabel {
    /// Get string representation of the entity label
    #[allow(unused)] // Public API for serialization/display
    pub fn as_str(&self) -> &str {
        match self {
            Self::Person => "Person",
            Self::Organization => "Organization",
            Self::Location => "Location",
            Self::Technology => "Technology",
            Self::Concept => "Concept",
            Self::Event => "Event",
            Self::Date => "Date",
            Self::Product => "Product",
            Self::Skill => "Skill",
            Self::Keyword => "Keyword",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// Memory tier for edge consolidation
///
/// Based on hippocampal-cortical memory consolidation research:
/// - L1 (Working): Dense, fast encoding, aggressive pruning (Dentate Gyrus-like)
/// - L2 (Episodic): Moderate density, Hebbian selection (CA1/CA3-like)
/// - L3 (Semantic): Sparse, near-permanent (Neocortex-like)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeTier {
    /// Working memory tier: new edges, dense, aggressive decay
    #[default]
    L1Working,
    /// Episodic memory tier: proven edges, moderate decay
    L2Episodic,
    /// Semantic memory tier: consolidated edges, near-permanent
    L3Semantic,
}

impl EdgeTier {
    /// Get the initial weight for edges in this tier
    pub fn initial_weight(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::L1Working => L1_INITIAL_WEIGHT,
            Self::L2Episodic => L2_PROMOTION_WEIGHT,
            Self::L3Semantic => L3_PROMOTION_WEIGHT,
        }
    }

    /// Get the prune threshold for this tier
    pub fn prune_threshold(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::L1Working => L1_PRUNE_THRESHOLD,
            Self::L2Episodic => L2_PRUNE_THRESHOLD,
            Self::L3Semantic => L3_PRUNE_THRESHOLD,
        }
    }

    /// Get the promotion threshold to move to next tier
    pub fn promotion_threshold(&self) -> Option<f32> {
        use crate::constants::*;
        match self {
            Self::L1Working => Some(L1_PROMOTION_THRESHOLD),
            Self::L2Episodic => Some(L2_PROMOTION_THRESHOLD),
            Self::L3Semantic => None, // Already at highest tier
        }
    }

    /// Get the next tier (for promotion)
    pub fn next_tier(&self) -> Option<Self> {
        match self {
            Self::L1Working => Some(Self::L2Episodic),
            Self::L2Episodic => Some(Self::L3Semantic),
            Self::L3Semantic => None,
        }
    }

    /// Get target density for this tier
    pub fn target_density(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::L1Working => L1_TARGET_DENSITY,
            Self::L2Episodic => L2_TARGET_DENSITY,
            Self::L3Semantic => L3_TARGET_DENSITY,
        }
    }
}

/// Long-Term Potentiation status for edges (PIPE-4)
///
/// Multi-scale LTP based on neuroscience research:
/// - Burst: Temporary protection from high-frequency activation (E-LTP)
/// - Weekly: Moderate protection from consistent routine use (L-LTP)
/// - Full: Maximum protection from sustained long-term use (systems consolidation)
///
/// Reference: Frey & Morris (1997) "Synaptic tagging and long-term potentiation"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LtpStatus {
    /// Not potentiated - normal decay applies
    #[default]
    None,

    /// Burst potentiated: 5+ activations in 24 hours
    /// Temporary protection (2x slower decay) that expires after 48h
    /// Represents early-phase LTP (protein synthesis independent)
    Burst {
        /// When burst was detected (for expiration check)
        #[serde(with = "chrono::serde::ts_seconds")]
        detected_at: DateTime<Utc>,
    },

    /// Weekly potentiated: 3+/week for 2+ weeks
    /// Moderate protection (3x slower decay)
    /// Represents late-phase LTP (habit formation)
    Weekly,

    /// Fully potentiated: 10+ activations OR 5+ over 30 days
    /// Maximum protection (10x slower decay)
    /// Represents systems consolidation (semantic memory)
    Full,
}

impl LtpStatus {
    /// Get the decay factor for this LTP status
    pub fn decay_factor(&self) -> f32 {
        use crate::constants::*;
        match self {
            Self::None => 1.0,
            Self::Burst { detected_at } => {
                // Check if burst has expired
                let hours_since = (Utc::now() - *detected_at).num_hours();
                if hours_since > LTP_BURST_DURATION_HOURS {
                    1.0 // Expired, normal decay
                } else {
                    LTP_BURST_DECAY_FACTOR
                }
            }
            Self::Weekly => LTP_WEEKLY_DECAY_FACTOR,
            Self::Full => LTP_DECAY_FACTOR,
        }
    }

    /// Check if this status provides any protection
    pub fn is_potentiated(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Check if burst protection has expired
    pub fn is_burst_expired(&self) -> bool {
        use crate::constants::LTP_BURST_DURATION_HOURS;
        match self {
            Self::Burst { detected_at } => {
                (Utc::now() - *detected_at).num_hours() > LTP_BURST_DURATION_HOURS
            }
            _ => false,
        }
    }

    /// Get priority for LTP upgrades (higher = stronger protection)
    pub fn priority(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::Burst { .. } => 1,
            Self::Weekly => 2,
            Self::Full => 3,
        }
    }
}

/// Relationship edge between entities
///
/// Implements Hebbian synaptic plasticity: "Neurons that fire together, wire together"
/// - Strength increases with co-activation (strengthen method)
/// - Strength decays over time without use (decay method)
/// - Long-Term Potentiation (LTP): After threshold activations, becomes permanent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEdge {
    /// Unique identifier for this edge
    pub uuid: Uuid,

    /// Source entity UUID
    pub from_entity: Uuid,

    /// Target entity UUID
    pub to_entity: Uuid,

    /// Type of relationship
    pub relation_type: RelationType,

    /// Confidence/strength of this relationship (0.0 to 1.0)
    /// Dynamic: increases with co-activation, decays without use
    pub strength: f32,

    /// When this relationship was created
    pub created_at: DateTime<Utc>,

    /// When this relationship was last observed (temporal tracking)
    pub valid_at: DateTime<Utc>,

    /// Whether this relationship has been invalidated (temporal edge invalidation)
    pub invalidated_at: Option<DateTime<Utc>>,

    /// Source episode that created this relationship
    pub source_episode_id: Option<Uuid>,

    /// Additional context about the relationship
    pub context: String,

    // === Hebbian Synaptic Plasticity Fields ===
    /// When this synapse was last activated (used in retrieval/traversal)
    /// Used to calculate time-based decay
    #[serde(default = "default_last_activated")]
    pub last_activated: DateTime<Utc>,

    /// Number of times both entities were co-accessed (Hebbian co-activation)
    /// Higher count = stronger learned association
    #[serde(default)]
    pub activation_count: u32,

    /// Long-Term Potentiation status (PIPE-4: multi-scale LTP)
    /// Replaces simple bool with tiered protection levels:
    /// - None: Normal decay
    /// - Burst: Temporary 2x protection (5+ activations in 24h)
    /// - Weekly: Moderate 3x protection (3+/week for 2 weeks)
    /// - Full: Maximum 10x protection (10+ activations or 5+ over 30 days)
    #[serde(default)]
    pub ltp_status: LtpStatus,

    /// Memory tier for consolidation (L1→L2→L3)
    /// Edges start in L1 (working memory) and promote based on Hebbian strength
    #[serde(default)]
    pub tier: EdgeTier,

    /// Activation timestamp history for temporal pattern detection (PIPE-4)
    /// Only populated for L2+ edges (L1 edges die too quickly to need history)
    /// Capacity: L2 = 20 timestamps, L3 = 50 timestamps
    /// Enables: burst detection, weekly patterns, temporal query relevance
    #[serde(default)]
    pub activation_timestamps: Option<VecDeque<DateTime<Utc>>>,

    /// Entity extraction confidence (PIPE-5: Unified LTP Readiness)
    /// Average confidence of the entities connected by this edge.
    /// Affects LTP threshold: high confidence → faster LTP (7 activations)
    /// Low confidence → slower LTP (13 activations).
    /// Based on synaptic tagging: behaviorally relevant synapses consolidate faster.
    #[serde(default)]
    pub entity_confidence: Option<f32>,
}

fn default_last_activated() -> DateTime<Utc> {
    Utc::now()
}

// Hebbian learning constants now imported from crate::constants:
// - LTP_LEARNING_RATE (0.1): η for strength increase per co-activation
// - LTP_DECAY_HALF_LIFE_DAYS (14.0): λ for time-based decay
// - LTP_THRESHOLD (10): Activations needed for Full LTP
// - LTP_DECAY_FACTOR (0.1): Fully potentiated synapses decay 10x slower
// - LTP_MIN_STRENGTH (0.01): Floor to prevent complete forgetting
// PIPE-4 additions:
// - LTP_BURST_THRESHOLD (5): Activations in 24h for burst LTP
// - LTP_BURST_WINDOW_HOURS (24): Window for burst detection
// - LTP_WEEKLY_THRESHOLD (3): Activations per week for weekly LTP
// - LTP_WEEKLY_MIN_WEEKS (2): Minimum weeks of consistent activation

impl RelationshipEdge {
    /// Strengthen this synapse (Hebbian learning)
    ///
    /// Called when both connected entities are accessed together.
    /// Formula: w_new = w_old + η × (1 - w_old) × co_activation_boost
    ///
    /// PIPE-4: Multi-scale LTP detection
    /// - Records activation timestamps for L2+ edges
    /// - Detects burst patterns (5+ in 24h) → temporary protection
    /// - Detects weekly patterns (3+/week for 2 weeks) → moderate protection
    /// - Detects sustained patterns (10+ total or 5+ over 30 days) → full protection
    ///
    /// Also handles tier promotion (L1→L2→L3) when strength exceeds tier threshold.
    ///
    /// Returns `Some((old_tier_name, new_tier_name))` if a tier promotion occurred,
    /// `None` otherwise. This enables the memory-edge coupling: edge promotions
    /// can signal the memory layer to boost the associated memory's importance.
    pub fn strengthen(&mut self) -> Option<(String, String)> {
        use crate::constants::*;

        let now = Utc::now();
        self.activation_count += 1;
        self.last_activated = now;

        // PIPE-4: Record activation timestamp for L2+ edges
        self.record_activation_timestamp(now);

        // Hebbian strengthening with tier-specific boost
        let tier_boost = match self.tier {
            EdgeTier::L1Working => TIER_CO_ACCESS_BOOST,
            EdgeTier::L2Episodic => TIER_CO_ACCESS_BOOST * 0.8,
            EdgeTier::L3Semantic => TIER_CO_ACCESS_BOOST * 0.5,
        };
        let boost = (LTP_LEARNING_RATE + tier_boost) * (1.0 - self.strength);
        self.strength = (self.strength + boost).min(1.0);

        // PIPE-4: Multi-scale LTP detection (only upgrade, never downgrade)
        let new_ltp_status = self.detect_ltp_status(now);
        if new_ltp_status.priority() > self.ltp_status.priority() {
            let old_status = self.ltp_status;
            self.ltp_status = new_ltp_status;

            // LTP bonus: immediate strength boost on upgrade
            let bonus = match new_ltp_status {
                LtpStatus::Burst { .. } => 0.05,
                LtpStatus::Weekly => 0.1,
                LtpStatus::Full => 0.2,
                LtpStatus::None => 0.0,
            };
            self.strength = (self.strength + bonus).min(1.0);

            tracing::debug!(
                "Edge {} LTP upgrade: {:?} → {:?} (activations: {}, age: {} days)",
                self.uuid,
                old_status,
                self.ltp_status,
                self.activation_count,
                (now - self.created_at).num_days()
            );
        }

        // Check for burst expiration and potential downgrade
        if self.ltp_status.is_burst_expired() {
            // Burst expired - check if weekly pattern has emerged
            let weekly_check = self.detect_weekly_pattern();
            if weekly_check {
                self.ltp_status = LtpStatus::Weekly;
            } else {
                self.ltp_status = LtpStatus::None;
            }
        }

        // Tier promotion: check if strength exceeds current tier's promotion threshold
        let mut promotion_result = None;
        if let Some(threshold) = self.tier.promotion_threshold() {
            if self.strength >= threshold {
                if let Some(next_tier) = self.tier.next_tier() {
                    let old_tier = self.tier;
                    self.tier = next_tier;
                    // Preserve strength if already above next tier's initial weight
                    self.strength = self.strength.max(next_tier.initial_weight());

                    // PIPE-4: Initialize activation_timestamps on L1→L2 promotion
                    if old_tier == EdgeTier::L1Working {
                        self.activation_timestamps =
                            Some(VecDeque::with_capacity(ACTIVATION_HISTORY_L2_CAPACITY));
                        // Seed with current timestamp
                        if let Some(ref mut ts) = self.activation_timestamps {
                            ts.push_back(now);
                        }
                    }

                    // Expand capacity on L2→L3 promotion
                    if old_tier == EdgeTier::L2Episodic {
                        if let Some(ref mut ts) = self.activation_timestamps {
                            let current = ts.capacity();
                            if current < ACTIVATION_HISTORY_L3_CAPACITY {
                                ts.reserve(ACTIVATION_HISTORY_L3_CAPACITY - current);
                            }
                        }
                    }

                    tracing::debug!(
                        "Edge {} promoted: {:?} → {:?}",
                        self.uuid,
                        old_tier,
                        self.tier
                    );

                    promotion_result =
                        Some((format!("{:?}", old_tier), format!("{:?}", self.tier)));
                }
            }
        }

        // PIPE-5: L3 auto-LTP removed - now handled by unified ltp_readiness()
        // The readiness formula combines strength + activation count + entity confidence,
        // ensuring both intensity and repetition evidence are required for Full LTP.

        promotion_result
    }

    /// Record an activation timestamp (PIPE-4)
    ///
    /// Only records for L2+ edges. Maintains capacity limits.
    fn record_activation_timestamp(&mut self, timestamp: DateTime<Utc>) {
        use crate::constants::*;

        // L1 edges don't track history (too transient)
        if matches!(self.tier, EdgeTier::L1Working) {
            return;
        }

        // Initialize if needed
        if self.activation_timestamps.is_none() {
            let capacity = match self.tier {
                EdgeTier::L1Working => return,
                EdgeTier::L2Episodic => ACTIVATION_HISTORY_L2_CAPACITY,
                EdgeTier::L3Semantic => ACTIVATION_HISTORY_L3_CAPACITY,
            };
            self.activation_timestamps = Some(VecDeque::with_capacity(capacity));
        }

        if let Some(ref mut timestamps) = self.activation_timestamps {
            let capacity = match self.tier {
                EdgeTier::L1Working => return,
                EdgeTier::L2Episodic => ACTIVATION_HISTORY_L2_CAPACITY,
                EdgeTier::L3Semantic => ACTIVATION_HISTORY_L3_CAPACITY,
            };

            // Maintain capacity limit (ring buffer behavior)
            while timestamps.len() >= capacity {
                timestamps.pop_front();
            }
            timestamps.push_back(timestamp);
        }
    }

    /// Detect LTP status based on unified readiness model (PIPE-4 + PIPE-5)
    ///
    /// PIPE-5 unifies LTP detection into a single readiness score that combines:
    /// - Activation count (repetition path)
    /// - Strength (intensity/durability path)
    /// - Entity confidence (synaptic tagging bonus)
    ///
    /// Multiple paths can lead to Full LTP:
    /// - High repetition alone (15+ activations)
    /// - High intensity alone (0.95+ strength at L3)
    /// - Balanced contribution from both
    /// - High-confidence edges reach threshold ~30% faster
    ///
    /// Temporal patterns (Burst, Weekly) remain separate as they represent
    /// different consolidation mechanisms (E-LTP vs habit formation).
    fn detect_ltp_status(&self, now: DateTime<Utc>) -> LtpStatus {
        use crate::constants::*;

        // PIPE-5: Unified LTP readiness for Full LTP
        // Combines activation count, strength, and entity confidence
        if self.ltp_readiness() >= LTP_READINESS_THRESHOLD {
            return LtpStatus::Full;
        }

        // Legacy time-aware path: 5+ activations over 30+ days
        // Kept for backward compatibility and edges that survived long decay
        let edge_age_days = (now - self.created_at).num_days();
        if edge_age_days >= LTP_TIME_AWARE_DAYS && self.activation_count >= LTP_TIME_AWARE_THRESHOLD
        {
            return LtpStatus::Full;
        }

        // Check for Weekly LTP (requires timestamp history)
        // Temporal pattern: 3+/week for 2+ weeks indicates habit
        if self.detect_weekly_pattern() {
            return LtpStatus::Weekly;
        }

        // Check for Burst LTP (requires timestamp history)
        // Temporal pattern: 5+ in 24h indicates high immediate interest
        if self.detect_burst_pattern(now) {
            return LtpStatus::Burst { detected_at: now };
        }

        LtpStatus::None
    }

    /// Detect burst pattern: 5+ activations in 24 hours (PIPE-4)
    fn detect_burst_pattern(&self, now: DateTime<Utc>) -> bool {
        use crate::constants::*;
        use chrono::Duration;

        let timestamps = match &self.activation_timestamps {
            Some(ts) => ts,
            None => return false,
        };

        let window_start = now - Duration::hours(LTP_BURST_WINDOW_HOURS);
        let count_in_window = timestamps.iter().filter(|&&ts| ts >= window_start).count();

        count_in_window >= LTP_BURST_THRESHOLD as usize
    }

    /// Detect weekly pattern: 3+/week for 2+ weeks (PIPE-4)
    fn detect_weekly_pattern(&self) -> bool {
        use crate::constants::*;
        use chrono::Duration;

        let timestamps = match &self.activation_timestamps {
            Some(ts) => ts,
            None => return false,
        };

        if timestamps.is_empty() {
            return false;
        }

        let now = Utc::now();
        let mut weeks_meeting_threshold = 0u32;

        // Check each of the last LTP_WEEKLY_MIN_WEEKS weeks
        for week_offset in 0..LTP_WEEKLY_MIN_WEEKS {
            let week_end = now - Duration::weeks(week_offset as i64);
            let week_start = week_end - Duration::weeks(1);

            let count_in_week = timestamps
                .iter()
                .filter(|&&ts| ts >= week_start && ts < week_end)
                .count();

            if count_in_week >= LTP_WEEKLY_THRESHOLD as usize {
                weeks_meeting_threshold += 1;
            }
        }

        weeks_meeting_threshold >= LTP_WEEKLY_MIN_WEEKS
    }

    /// Get activation count within a time window (for temporal retrieval scoring)
    pub fn activations_in_window(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> usize {
        match &self.activation_timestamps {
            Some(ts) => ts.iter().filter(|&&t| t >= start && t <= end).count(),
            None => 0,
        }
    }

    /// Check if edge was active at similar time of day (for temporal retrieval)
    pub fn time_of_day_match(&self, target_hour: u32, window_hours: u32) -> usize {
        use chrono::Timelike;

        match &self.activation_timestamps {
            Some(ts) => ts
                .iter()
                .filter(|t| {
                    let hour = t.hour();
                    let diff = if hour > target_hour {
                        (hour - target_hour).min(24 + target_hour - hour)
                    } else {
                        (target_hour - hour).min(24 + hour - target_hour)
                    };
                    diff <= window_hours
                })
                .count(),
            None => 0,
        }
    }

    /// Apply time-based decay to this synapse
    ///
    /// Uses tier-aware decay model (3-tier memory consolidation):
    /// - L1 (Working): 15%/hour decay, max 4 hours before prune
    /// - L2 (Episodic): 10%/day decay, max 14 days before prune
    /// - L3 (Semantic): 2%/month decay, near-permanent
    ///
    /// PIPE-4: Multi-scale LTP protection
    /// - Burst: 2x slower decay (temporary, 48h)
    /// - Weekly: 3x slower decay (habit protection)
    /// - Full: 10x slower decay (permanent protection)
    ///
    /// **Important:** Updates `last_activated` to prevent double-decay on
    /// repeated calls.
    ///
    /// Returns true if synapse should be pruned (below tier's threshold)
    pub fn decay(&mut self) -> bool {
        use crate::decay::tier_decay_factor;

        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.last_activated);
        let hours_elapsed = elapsed.num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return false;
        }

        // Cap max decay to protect against clock jumps (max 1 year = 8760 hours)
        let hours_elapsed = hours_elapsed.min(8760.0);

        // Tier-aware decay with PIPE-4 multi-scale LTP
        let tier_num = match self.tier {
            EdgeTier::L1Working => 0,
            EdgeTier::L2Episodic => 1,
            EdgeTier::L3Semantic => 2,
        };
        let ltp_factor = self.ltp_status.decay_factor();
        let (decay_factor, exceeded_max_age) =
            tier_decay_factor(hours_elapsed, tier_num, ltp_factor);
        self.strength *= decay_factor;

        // Update last_activated to prevent double-decay on repeated calls
        self.last_activated = now;

        // Apply floor to prevent complete forgetting
        let prune_threshold = self.tier.prune_threshold();
        if self.strength < LTP_MIN_STRENGTH {
            self.strength = LTP_MIN_STRENGTH;
        }

        // Downgrade expired burst LTP before prune decision
        // decay_factor() already returns 1.0 for expired bursts (correct rate),
        // but is_potentiated() still returns true — preventing pruning
        if self.ltp_status.is_burst_expired() {
            if self.detect_weekly_pattern() {
                self.ltp_status = LtpStatus::Weekly;
            } else {
                self.ltp_status = LtpStatus::None;
            }
        }

        // Strip LTP protection from near-zero edges (zombie edge cleanup)
        // Prevents immortal edges that retain LTP despite negligible strength
        if self.ltp_status.is_potentiated() && self.strength <= LTP_PRUNE_FLOOR {
            self.ltp_status = LtpStatus::None;
        }

        // Return whether this synapse should be pruned
        // Prune if: exceeded max age OR below prune threshold (unless any LTP protection)
        if self.ltp_status.is_potentiated() {
            false
        } else {
            exceeded_max_age || self.strength <= prune_threshold
        }
    }

    /// Get the effective strength considering recency
    ///
    /// This is a read-only version that calculates what the strength
    /// would be after decay, without modifying the edge.
    /// Uses tier-aware decay (L1/L2/L3 have different decay rates).
    pub fn effective_strength(&self) -> f32 {
        use crate::decay::tier_decay_factor;

        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.last_activated);
        let hours_elapsed = elapsed.num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return self.strength;
        }

        let tier_num = match self.tier {
            EdgeTier::L1Working => 0,
            EdgeTier::L2Episodic => 1,
            EdgeTier::L3Semantic => 2,
        };
        let ltp_factor = self.ltp_status.decay_factor();
        let (decay_factor, _) = tier_decay_factor(hours_elapsed, tier_num, ltp_factor);
        (self.strength * decay_factor).max(LTP_MIN_STRENGTH)
    }

    /// Check if this edge has any LTP protection (for backward compatibility)
    pub fn is_potentiated(&self) -> bool {
        self.ltp_status.is_potentiated()
    }

    // =========================================================================
    // PIPE-5: Unified LTP Readiness Model
    // =========================================================================

    /// Get confidence-adjusted LTP threshold (PIPE-5)
    ///
    /// High-confidence edges (strong entity extraction) need fewer activations.
    /// Low-confidence edges need more activations to prove value.
    ///
    /// Returns: threshold in range [LTP_THRESHOLD_MIN, LTP_THRESHOLD_MAX]
    pub fn adjusted_threshold(&self) -> u32 {
        use crate::constants::*;

        let confidence = self.entity_confidence.unwrap_or(0.5);

        // Linear interpolation: high confidence → low threshold
        // confidence 0.0 → threshold_max (13)
        // confidence 1.0 → threshold_min (7)
        let range = LTP_THRESHOLD_MAX - LTP_THRESHOLD_MIN;
        let threshold = LTP_THRESHOLD_MAX as f32 - (confidence * range as f32);
        threshold.round() as u32
    }

    /// Get tier-specific strength floor for Full LTP (PIPE-5)
    ///
    /// L2 edges have lower floor (still proving themselves).
    /// L3 edges have higher floor (must demonstrate durability).
    /// L1 edges return 1.0 (effectively impossible to reach Full LTP).
    pub fn strength_floor(&self) -> f32 {
        use crate::constants::*;

        match self.tier {
            EdgeTier::L1Working => 1.0, // L1 can't reach Full LTP via readiness
            EdgeTier::L2Episodic => LTP_STRENGTH_FLOOR_L2,
            EdgeTier::L3Semantic => LTP_STRENGTH_FLOOR_L3,
        }
    }

    /// Calculate LTP readiness score (PIPE-5)
    ///
    /// Unified formula combining activation count, strength, and entity confidence:
    /// - count_score = activation_count / adjusted_threshold
    /// - strength_score = strength / strength_floor
    /// - tag_bonus = entity_confidence * TAG_WEIGHT
    ///
    /// readiness = count_score * COUNT_WEIGHT + strength_score * STRENGTH_WEIGHT + tag_bonus
    ///
    /// Full LTP when readiness >= 1.0
    ///
    /// This allows multiple paths to LTP:
    /// - Repetition-dominant: 15 activations can compensate for lower strength
    /// - Intensity-dominant: 0.95 strength can compensate for fewer activations
    /// - Balanced: 10 activations + 0.75 strength + moderate confidence
    /// - Tagged boost: high-confidence edges reach LTP ~30% faster
    pub fn ltp_readiness(&self) -> f32 {
        use crate::constants::*;

        // L1 edges can't reach Full LTP via readiness (too transient)
        if matches!(self.tier, EdgeTier::L1Working) {
            return 0.0;
        }

        let threshold = self.adjusted_threshold() as f32;
        let floor = self.strength_floor();

        // Count score: how close to activation threshold
        let count_score = self.activation_count as f32 / threshold;

        // Strength score: how close to strength floor
        let strength_score = self.strength / floor;

        // Tag bonus: entity confidence provides synaptic tagging advantage
        let confidence = self.entity_confidence.unwrap_or(0.5);
        let tag_bonus = confidence * LTP_READINESS_TAG_WEIGHT;

        // Weighted combination
        count_score * LTP_READINESS_COUNT_WEIGHT
            + strength_score * LTP_READINESS_STRENGTH_WEIGHT
            + tag_bonus
    }
}

/// Relationship types following Graphiti's semantic model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationType {
    /// Work relationships
    WorksWith,
    WorksAt,
    EmployedBy,

    /// Structural relationships
    PartOf,
    Contains,
    OwnedBy,

    /// Location relationships
    LocatedIn,
    LocatedAt,

    /// Usage relationships
    Uses,
    CreatedBy,
    DevelopedBy,

    /// Causal relationships
    Causes,
    ResultsIn,

    /// Learning relationships
    Learned,
    Knows,
    Teaches,

    /// Generic relationships
    RelatedTo,
    AssociatedWith,

    /// Memory co-retrieval (Hebbian association between memories)
    CoRetrieved,

    /// Sentence co-occurrence (entities appearing in same sentence)
    /// Key for multi-hop: "Melanie" <-> "sunrise" when "Melanie painted a sunrise"
    CoOccurs,

    /// Custom relationship
    Custom(String),
}

impl RelationType {
    /// Get string representation of the relation type
    #[allow(unused)] // Public API for serialization/display
    pub fn as_str(&self) -> &str {
        match self {
            Self::WorksWith => "WorksWith",
            Self::WorksAt => "WorksAt",
            Self::EmployedBy => "EmployedBy",
            Self::PartOf => "PartOf",
            Self::Contains => "Contains",
            Self::OwnedBy => "OwnedBy",
            Self::LocatedIn => "LocatedIn",
            Self::LocatedAt => "LocatedAt",
            Self::Uses => "Uses",
            Self::CreatedBy => "CreatedBy",
            Self::DevelopedBy => "DevelopedBy",
            Self::Causes => "Causes",
            Self::ResultsIn => "ResultsIn",
            Self::Learned => "Learned",
            Self::Knows => "Knows",
            Self::Teaches => "Teaches",
            Self::RelatedTo => "RelatedTo",
            Self::AssociatedWith => "AssociatedWith",
            Self::CoRetrieved => "CoRetrieved",
            Self::CoOccurs => "CoOccurs",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// Episodic node representing a discrete experience/memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicNode {
    /// Unique identifier
    pub uuid: Uuid,

    /// Human-readable name/title
    pub name: String,

    /// Episode content (the actual experience data)
    pub content: String,

    /// When the original event occurred (event time)
    pub valid_at: DateTime<Utc>,

    /// When this was ingested into the system (ingestion time)
    pub created_at: DateTime<Utc>,

    /// Entities extracted from this episode
    pub entity_refs: Vec<Uuid>,

    /// Source type
    pub source: EpisodeSource,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Episode source types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EpisodeSource {
    Message,
    Document,
    Event,
    Observation,
}

/// Graph memory storage and operations
///
/// Uses a single RocksDB instance with 9 column families for all graph data.
/// This reduces file descriptor usage from 9 separate DBs to 1 (sharing WAL, MANIFEST, LOCK).
pub struct GraphMemory {
    /// Unified RocksDB database with column families for entities, relationships,
    /// episodes, and all index tables
    db: Arc<DB>,

    /// In-memory entity name index for fast lookups (loaded from name_index CF)
    entity_name_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory lowercase name index for O(1) case-insensitive lookups
    entity_lowercase_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    /// In-memory stemmed name index for O(1) linguistic lookups
    /// Key: Porter-stemmed lowercase name, Value: Entity UUID
    entity_stemmed_index: Arc<parking_lot::RwLock<HashMap<String, Uuid>>>,

    // === Atomic counters for O(1) stats (P1 fix) ===
    /// Entity count - initialized from entity_name_index.len(), updated on add
    entity_count: Arc<AtomicUsize>,

    /// Relationship count - initialized on startup, updated on add
    relationship_count: Arc<AtomicUsize>,

    /// Episode count - initialized on startup, updated on add
    episode_count: Arc<AtomicUsize>,

    /// Mutex for serializing synapse updates to prevent race conditions (SHO-64)
    /// Uses parking_lot::Mutex for better performance than std::sync::Mutex
    synapse_update_lock: Arc<parking_lot::Mutex<()>>,

    /// In-memory cache of entity name embeddings for concept merging.
    /// Maps entity UUID → embedding vector. Loaded on startup, updated on add.
    /// Used when string-based dedup (exact/case/stemmed) fails — catches synonyms
    /// like "authentication" ↔ "auth" via cosine similarity.
    entity_embedding_cache: Arc<parking_lot::RwLock<Vec<(Uuid, Vec<f32>)>>>,

    /// Edges found below prune threshold during lazy-decay reads.
    /// Flushed as batch deletes on each maintenance cycle (no full scan needed).
    pending_prune: parking_lot::Mutex<Vec<Uuid>>,

    /// Entities that may have become orphaned from pruned edges.
    /// Checked during flush_pending_maintenance().
    pending_orphan_checks: parking_lot::Mutex<Vec<Uuid>>,
}

impl GraphMemory {
    // Column family accessors — cheap HashMap lookups on DB internals
    fn entities_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_ENTITIES)
            .expect("entities CF must exist")
    }
    fn relationships_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_RELATIONSHIPS)
            .expect("relationships CF must exist")
    }
    fn episodes_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_EPISODES)
            .expect("episodes CF must exist")
    }
    fn entity_edges_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_ENTITY_EDGES)
            .expect("entity_edges CF must exist")
    }
    fn entity_pair_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_ENTITY_PAIR_INDEX)
            .expect("entity_pair_index CF must exist")
    }
    fn entity_episodes_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_ENTITY_EPISODES)
            .expect("entity_episodes CF must exist")
    }
    fn name_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_NAME_INDEX)
            .expect("name_index CF must exist")
    }
    fn lowercase_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_LOWERCASE_INDEX)
            .expect("lowercase_index CF must exist")
    }
    fn stemmed_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_STEMMED_INDEX)
            .expect("stemmed_index CF must exist")
    }

    /// Create a new graph memory system
    pub fn new(path: &Path) -> Result<Self> {
        let graph_path = path.join("graph");
        std::fs::create_dir_all(&graph_path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB — graph entries are small KV pairs
        opts.set_max_write_buffer_number(2);

        // Bounded block cache prevents unbounded C++ heap growth during full scans.
        // 32MB is sufficient for the graph DB (entities + edges are small KV pairs).
        use rocksdb::{BlockBasedOptions, Cache};
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(&Cache::new_lru_cache(32 * 1024 * 1024));
        block_opts.set_cache_index_and_filter_blocks(true);
        opts.set_block_based_table_factory(&block_opts);

        // Build column family descriptors — all CFs share the same options
        let cf_descriptors: Vec<ColumnFamilyDescriptor> = GRAPH_CF_NAMES
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, opts.clone()))
            .collect();

        let db = Arc::new(DB::open_cf_descriptors(&opts, &graph_path, cf_descriptors)?);

        // Migrate data from old separate-DB layout if needed
        let migrated = Self::migrate_from_separate_dbs(path, &db)?;
        if migrated > 0 {
            tracing::info!(
                "Migrated {} entries from separate graph DBs to column families",
                migrated
            );
        }

        // Load entity name index from name_index CF (O(n) but faster than deserializing entities)
        // If empty, migrate from entities CF (one-time migration for existing data)
        let entity_name_index = Self::load_or_migrate_name_index(&db)?;

        // Load/migrate lowercase index for O(1) case-insensitive lookup
        let entity_lowercase_index =
            Self::load_or_migrate_lowercase_index(&db, &entity_name_index)?;

        // Load/migrate stemmed index for O(1) linguistic lookup
        let entity_stemmed_index = Self::load_or_migrate_stemmed_index(&db, &entity_name_index)?;

        let entity_count = entity_name_index.len();

        // Count relationships and episodes during startup (one-time cost)
        // This is O(n) at startup, but get_stats() will be O(1) at runtime
        let relationships_cf = db.cf_handle(CF_RELATIONSHIPS).unwrap();
        let episodes_cf = db.cf_handle(CF_EPISODES).unwrap();
        let relationship_count = Self::count_cf_entries(&db, relationships_cf);
        let episode_count = Self::count_cf_entries(&db, episodes_cf);

        // Load entity embedding cache for concept merging
        // Only entities with pre-computed name_embeddings are cached
        let entities_cf = db.cf_handle(CF_ENTITIES).unwrap();
        let entity_embedding_cache =
            Self::load_entity_embedding_cache(&db, entities_cf, &entity_name_index);
        let embedding_cache_size = entity_embedding_cache.len();

        let graph = Self {
            db,
            entity_name_index: Arc::new(parking_lot::RwLock::new(entity_name_index)),
            entity_lowercase_index: Arc::new(parking_lot::RwLock::new(entity_lowercase_index)),
            entity_stemmed_index: Arc::new(parking_lot::RwLock::new(entity_stemmed_index)),
            entity_count: Arc::new(AtomicUsize::new(entity_count)),
            relationship_count: Arc::new(AtomicUsize::new(relationship_count)),
            episode_count: Arc::new(AtomicUsize::new(episode_count)),
            synapse_update_lock: Arc::new(parking_lot::Mutex::new(())),
            entity_embedding_cache: Arc::new(parking_lot::RwLock::new(entity_embedding_cache)),
            pending_prune: parking_lot::Mutex::new(Vec::new()),
            pending_orphan_checks: parking_lot::Mutex::new(Vec::new()),
        };

        if entity_count > 0 || relationship_count > 0 || episode_count > 0 {
            tracing::info!(
                "Loaded graph with {} entities ({} with embeddings), {} relationships, {} episodes",
                entity_count,
                embedding_cache_size,
                relationship_count,
                episode_count
            );
        }

        Ok(graph)
    }

    /// Migrate data from the old separate-DB layout (pre-CF) into column families.
    ///
    /// Detects old `graph_*` subdirectories, opens them read-only, copies all KV
    /// pairs into the corresponding CF, then renames the old directory for rollback safety.
    fn migrate_from_separate_dbs(base_path: &Path, db: &DB) -> Result<usize> {
        let old_dirs: &[(&str, &str)] = &[
            ("graph_entities", CF_ENTITIES),
            ("graph_relationships", CF_RELATIONSHIPS),
            ("graph_episodes", CF_EPISODES),
            ("graph_entity_edges", CF_ENTITY_EDGES),
            ("graph_entity_pair_index", CF_ENTITY_PAIR_INDEX),
            ("graph_entity_episodes", CF_ENTITY_EPISODES),
            ("graph_entity_name_index", CF_NAME_INDEX),
            ("graph_entity_lowercase_index", CF_LOWERCASE_INDEX),
            ("graph_entity_stemmed_index", CF_STEMMED_INDEX),
        ];

        let mut total_migrated = 0usize;

        for (old_name, cf_name) in old_dirs {
            let old_path = base_path.join(old_name);
            if !old_path.exists() {
                continue;
            }

            let cf = db.cf_handle(cf_name).unwrap();

            // Only migrate if the CF is empty (avoid double migration)
            if db
                .iterator_cf(cf, rocksdb::IteratorMode::Start)
                .next()
                .is_some()
            {
                // CF already has data — just rename the old dir
                let renamed = base_path.join(format!("{}.pre_cf_migration", old_name));
                if !renamed.exists() {
                    let _ = std::fs::rename(&old_path, &renamed);
                }
                continue;
            }

            // Open old DB read-only and copy all entries
            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_path, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;

                    for item in old_db.iterator(rocksdb::IteratorMode::Start) {
                        match item {
                            Ok((key, value)) => {
                                batch.put_cf(cf, &key, &value);
                                count += 1;
                                // Flush in chunks to limit memory usage
                                if count % 10_000 == 0 {
                                    db.write(std::mem::take(&mut batch))?;
                                    batch = WriteBatch::default();
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Error reading from old {}: {}", old_name, e);
                                break;
                            }
                        }
                    }

                    if count > 0 {
                        db.write(batch)?;
                    }

                    drop(old_db);

                    // Rename old directory for rollback safety
                    let renamed = base_path.join(format!("{}.pre_cf_migration", old_name));
                    if let Err(e) = std::fs::rename(&old_path, &renamed) {
                        tracing::warn!(
                            "Migrated {} entries from {} but failed to rename: {}",
                            count,
                            old_name,
                            e
                        );
                    } else {
                        tracing::info!(
                            "Migrated {} entries from {} to CF '{}'",
                            count,
                            old_name,
                            cf_name
                        );
                    }

                    total_migrated += count;
                }
                Err(e) => {
                    tracing::warn!("Failed to open old DB {} for migration: {}", old_name, e);
                }
            }
        }

        Ok(total_migrated)
    }

    /// Load entity name->UUID index from name_index CF, or migrate from entities CF if empty
    fn load_or_migrate_name_index(db: &DB) -> Result<HashMap<String, Uuid>> {
        let name_index_cf = db.cf_handle(CF_NAME_INDEX).unwrap();
        let entities_cf = db.cf_handle(CF_ENTITIES).unwrap();
        let mut index = HashMap::new();

        // Try to load from name_index CF first
        let iter = db.iterator_cf(name_index_cf, rocksdb::IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If name_index CF is empty but entities exist, migrate (one-time operation)
        if index.is_empty() {
            let entity_iter = db.iterator_cf(entities_cf, rocksdb::IteratorMode::Start);
            let mut migrated_count = 0;
            for (_, value) in entity_iter.flatten() {
                if let Ok(entity) = bincode::serde::decode_from_slice::<EntityNode, _>(
                    &value,
                    bincode::config::standard(),
                )
                .map(|(v, _)| v)
                {
                    // Store in name_index CF: name -> UUID bytes
                    db.put_cf(
                        name_index_cf,
                        entity.name.as_bytes(),
                        entity.uuid.as_bytes(),
                    )?;
                    index.insert(entity.name.clone(), entity.uuid);
                    migrated_count += 1;
                }
            }
            if migrated_count > 0 {
                tracing::info!("Migrated {} entities to name index CF", migrated_count);
            }
        }

        Ok(index)
    }

    /// Load lowercase name->UUID index, or migrate from name_index if empty
    ///
    /// This enables O(1) case-insensitive entity lookup instead of O(n) linear search.
    fn load_or_migrate_lowercase_index(
        db: &DB,
        name_index: &HashMap<String, Uuid>,
    ) -> Result<HashMap<String, Uuid>> {
        let lowercase_cf = db.cf_handle(CF_LOWERCASE_INDEX).unwrap();
        let mut index = HashMap::new();

        // Try to load from lowercase_index CF
        let iter = db.iterator_cf(lowercase_cf, rocksdb::IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If empty but name_index has data, migrate (one-time operation)
        if index.is_empty() && !name_index.is_empty() {
            for (name, uuid) in name_index {
                let lowercase_name = name.to_lowercase();
                db.put_cf(lowercase_cf, lowercase_name.as_bytes(), uuid.as_bytes())?;
                index.insert(lowercase_name, *uuid);
            }
            tracing::info!(
                "Migrated {} entities to lowercase index CF",
                name_index.len()
            );
        }

        Ok(index)
    }

    /// Load stemmed name->UUID index, or migrate from name_index if empty
    ///
    /// This enables O(1) linguistic entity lookup: "running" matches "run"
    /// Uses Porter2 stemmer for English language stemming.
    fn load_or_migrate_stemmed_index(
        db: &DB,
        name_index: &HashMap<String, Uuid>,
    ) -> Result<HashMap<String, Uuid>> {
        let stemmed_cf = db.cf_handle(CF_STEMMED_INDEX).unwrap();
        let mut index = HashMap::new();

        // Try to load from stemmed_index CF
        let iter = db.iterator_cf(stemmed_cf, rocksdb::IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if let (Ok(name), Ok(uuid_bytes)) = (
                std::str::from_utf8(&key),
                <[u8; 16]>::try_from(value.as_ref()),
            ) {
                index.insert(name.to_string(), Uuid::from_bytes(uuid_bytes));
            }
        }

        // If empty but name_index has data, migrate (one-time operation)
        if index.is_empty() && !name_index.is_empty() {
            let stemmer = Stemmer::create(Algorithm::English);
            for (name, uuid) in name_index {
                let stemmed_name = Self::stem_entity_name(&stemmer, name);
                db.put_cf(stemmed_cf, stemmed_name.as_bytes(), uuid.as_bytes())?;
                index.insert(stemmed_name, *uuid);
            }
            tracing::info!("Migrated {} entities to stemmed index CF", name_index.len());
        }

        Ok(index)
    }

    /// Stem an entity name for linguistic matching
    ///
    /// For multi-word names (e.g., "New York City"), stems each word and joins.
    /// Returns lowercase stemmed version for consistent matching.
    fn stem_entity_name(stemmer: &Stemmer, name: &str) -> String {
        name.split_whitespace()
            .map(|word| stemmer.stem(&word.to_lowercase()).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Count entries in a column family (one-time startup cost)
    fn count_cf_entries(db: &DB, cf: &ColumnFamily) -> usize {
        db.iterator_cf(cf, rocksdb::IteratorMode::Start).count()
    }

    /// Load entity embedding cache from persisted entities.
    ///
    /// Scans entities referenced by the name index and collects those with
    /// pre-computed name_embeddings into an in-memory cache for O(n) concept
    /// merging during `add_entity()`. Entities without embeddings (pre-upgrade
    /// data) are skipped and will gain embeddings on their next mention.
    fn load_entity_embedding_cache(
        db: &DB,
        entities_cf: &ColumnFamily,
        name_index: &HashMap<String, Uuid>,
    ) -> Vec<(Uuid, Vec<f32>)> {
        let mut cache = Vec::new();
        for uuid in name_index.values() {
            let key = uuid.as_bytes();
            if let Ok(Some(value)) = db.get_cf(entities_cf, key) {
                if let Ok((entity, _)) = bincode::serde::decode_from_slice::<EntityNode, _>(
                    &value,
                    bincode::config::standard(),
                ) {
                    if let Some(emb) = entity.name_embedding {
                        cache.push((*uuid, emb));
                    }
                }
            }
        }
        cache
    }

    /// Add or update an entity node
    /// Salience is updated using the formula: salience = base_salience * (1 + 0.1 * ln(mention_count))
    /// This means frequently mentioned entities grow in salience (gravitational wells get heavier)
    ///
    /// BUG-002 FIX: Handles crash recovery for orphaned entities/stale indices
    pub fn add_entity(&self, mut entity: EntityNode) -> Result<Uuid> {
        // Multi-tier dedup pipeline: exact → case-insensitive → stemmed → embedding
        // Each tier is faster than the next; short-circuits on first match.

        // Tier 1: Exact name match (O(1))
        let mut existing_uuid = {
            let index = self.entity_name_index.read();
            index.get(&entity.name).cloned()
        };

        // Tier 2: Case-insensitive match (O(1))
        if existing_uuid.is_none() {
            let lowercase_name = entity.name.to_lowercase();
            let index = self.entity_lowercase_index.read();
            existing_uuid = index.get(&lowercase_name).cloned();
        }

        // Tier 3: Stemmed match (O(1)) — "running" matches "run"
        // Skip for proper nouns to prevent "Paris" → "pari" merging with "Parison"
        if existing_uuid.is_none() && !entity.is_proper_noun {
            let stemmer = Stemmer::create(Algorithm::English);
            let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);
            let index = self.entity_stemmed_index.read();
            existing_uuid = index.get(&stemmed_name).cloned();
        }

        // Tier 4: Embedding-based concept merge (O(n) over cache)
        // Catches synonyms like "authentication" ↔ "auth" that string matching misses.
        // Only runs when the entity carries a name_embedding (populated by caller).
        if existing_uuid.is_none() {
            if let Some(ref new_emb) = entity.name_embedding {
                let cache = self.entity_embedding_cache.read();
                let mut best_match: Option<(Uuid, f32)> = None;
                for (uuid, existing_emb) in cache.iter() {
                    let sim = crate::similarity::cosine_similarity(new_emb, existing_emb);
                    if sim >= ENTITY_CONCEPT_MERGE_THRESHOLD {
                        if best_match.map_or(true, |(_, best_sim)| sim > best_sim) {
                            best_match = Some((*uuid, sim));
                        }
                    }
                }
                if let Some((matched_uuid, sim)) = best_match {
                    tracing::debug!(
                        "Concept merge: '{}' matched existing entity {} (cosine={:.3})",
                        entity.name,
                        matched_uuid,
                        sim
                    );
                    existing_uuid = Some(matched_uuid);
                }
            }
        }

        let is_new_entity;
        if let Some(uuid) = existing_uuid {
            // BUG-002 FIX: Verify entity actually exists in DB (handles stale index)
            if let Some(existing) = self.get_entity(&uuid)? {
                // Update existing entity — merge into canonical node
                entity.uuid = uuid;
                entity.mention_count = existing.mention_count + 1;
                entity.last_seen_at = Utc::now();
                entity.created_at = existing.created_at;
                entity.is_proper_noun = existing.is_proper_noun || entity.is_proper_noun;

                // Preserve the canonical name (first-seen name wins)
                entity.name = existing.name.clone();

                // Preserve existing embedding if the incoming one is None
                if entity.name_embedding.is_none() {
                    entity.name_embedding = existing.name_embedding;
                }

                // Update salience with frequency boost
                // Formula: salience = base_salience * (1 + 0.1 * ln(mention_count))
                // This caps at about 1.3x boost at 20 mentions
                let frequency_boost = 1.0 + 0.1 * (entity.mention_count as f32).ln();
                entity.salience = (existing.salience * frequency_boost).min(1.0);
                is_new_entity = false;
            } else {
                // BUG-002 FIX: Stale index entry - entity in index but not in DB
                tracing::warn!(
                    "Stale index entry for entity '{}' (uuid={}), recreating",
                    entity.name,
                    uuid
                );
                entity.uuid = Uuid::new_v4();
                entity.created_at = Utc::now();
                entity.last_seen_at = entity.created_at;
                entity.mention_count = 1;
                is_new_entity = true;
            }
        } else {
            // Genuinely new entity — no match at any tier
            entity.uuid = Uuid::new_v4();
            entity.created_at = Utc::now();
            entity.last_seen_at = entity.created_at;
            entity.mention_count = 1;
            is_new_entity = true;
        }

        // BUG-002 FIX: Write index FIRST, then entity
        let lowercase_name = entity.name.to_lowercase();
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);

        // Update in-memory indices
        {
            let mut index = self.entity_name_index.write();
            index.insert(entity.name.clone(), entity.uuid);
        }
        {
            let mut lowercase_index = self.entity_lowercase_index.write();
            lowercase_index.insert(lowercase_name.clone(), entity.uuid);
        }
        // Skip stemmed index for proper nouns to prevent "Paris" → "pari" collisions
        if !entity.is_proper_noun {
            let mut stemmed_index = self.entity_stemmed_index.write();
            stemmed_index.insert(stemmed_name.clone(), entity.uuid);
        }

        // Update entity embedding cache for future concept merges
        if let Some(ref emb) = entity.name_embedding {
            let mut cache = self.entity_embedding_cache.write();
            if is_new_entity {
                cache.push((entity.uuid, emb.clone()));
            } else {
                // Update existing entry in cache (embedding may have changed)
                if let Some(entry) = cache.iter_mut().find(|(uuid, _)| *uuid == entity.uuid) {
                    entry.1 = emb.clone();
                }
            }
        }

        // Persist name->UUID mappings
        self.db.put_cf(
            self.name_index_cf(),
            entity.name.as_bytes(),
            entity.uuid.as_bytes(),
        )?;
        self.db.put_cf(
            self.lowercase_index_cf(),
            lowercase_name.as_bytes(),
            entity.uuid.as_bytes(),
        )?;
        if !entity.is_proper_noun {
            self.db.put_cf(
                self.stemmed_index_cf(),
                stemmed_name.as_bytes(),
                entity.uuid.as_bytes(),
            )?;
        }

        // Store entity in database
        let key = entity.uuid.as_bytes();
        let value = bincode::serde::encode_to_vec(&entity, bincode::config::standard())?;
        self.db.put_cf(self.entities_cf(), key, value)?;

        // Increment counter only for truly new entities
        if is_new_entity {
            self.entity_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(entity.uuid)
    }

    /// Get entity by UUID
    pub fn get_entity(&self, uuid: &Uuid) -> Result<Option<EntityNode>> {
        let key = uuid.as_bytes();
        match self.db.get_cf(self.entities_cf(), key)? {
            Some(value) => {
                let (entity, _): (EntityNode, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Delete an entity and all its index entries.
    ///
    /// Removes the entity from:
    /// 1. `entities` CF (primary storage)
    /// 2. `entity_name_index` (exact name → UUID)
    /// 3. `entity_lowercase_index` (lowercase name → UUID)
    /// 4. `entity_stemmed_index` (stemmed name → UUID)
    /// 5. `entity_embedding_cache` (in-memory embedding vector)
    /// 6. `entity_pair_index` CF (co-occurrence pair entries)
    /// 7. Decrements `entity_count`
    ///
    /// Returns true if the entity existed and was deleted.
    pub fn delete_entity(&self, uuid: &Uuid) -> Result<bool> {
        let entity = match self.get_entity(uuid)? {
            Some(e) => e,
            None => return Ok(false),
        };

        // 1. Remove from entities CF
        self.db.delete_cf(self.entities_cf(), uuid.as_bytes())?;

        // 2-3-4. Remove from name indices (in-memory + persisted)
        let lowercase_name = entity.name.to_lowercase();
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, &entity.name);

        {
            let mut index = self.entity_name_index.write();
            index.remove(&entity.name);
        }
        self.db
            .delete_cf(self.name_index_cf(), entity.name.as_bytes())?;

        {
            let mut index = self.entity_lowercase_index.write();
            index.remove(&lowercase_name);
        }
        self.db
            .delete_cf(self.lowercase_index_cf(), lowercase_name.as_bytes())?;

        {
            let mut index = self.entity_stemmed_index.write();
            index.remove(&stemmed_name);
        }
        self.db
            .delete_cf(self.stemmed_index_cf(), stemmed_name.as_bytes())?;

        // 5. Remove from embedding cache
        {
            let mut cache = self.entity_embedding_cache.write();
            cache.retain(|(id, _)| id != uuid);
        }

        // 6. Remove entity_pair_index entries (prefix scan)
        let prefix = format!("{}:", uuid);
        let mut pairs_to_delete = Vec::new();
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_pair_index_cf(), prefix.as_bytes());
        for item in iter {
            match item {
                Ok((key, _)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if key_str.starts_with(&prefix) {
                        pairs_to_delete.push(key.to_vec());
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        // Also scan for reverse direction (other_uuid:this_uuid)
        let suffix = format!(":{}", uuid);
        let iter = self
            .db
            .iterator_cf(self.entity_pair_index_cf(), rocksdb::IteratorMode::Start);
        for item in iter {
            match item {
                Ok((key, _)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if key_str.ends_with(&suffix) {
                        pairs_to_delete.push(key.to_vec());
                    }
                }
                Err(_) => break,
            }
        }
        for key in &pairs_to_delete {
            self.db.delete_cf(self.entity_pair_index_cf(), key)?;
        }

        // 7. Decrement counter
        self.entity_count.fetch_sub(1, Ordering::Relaxed);

        tracing::debug!("Deleted orphaned entity '{}' (uuid={})", entity.name, uuid);
        Ok(true)
    }

    /// Find entity by name (case-insensitive, O(1) lookup)
    ///
    /// Uses a multi-tier matching strategy:
    /// 1. Exact match (O(1)) - fastest
    /// 2. Case-insensitive match (O(1)) - common case
    /// 3. Stemmed match (O(1)) - "running" matches "run"
    /// 4. Substring match - "York" matches "New York City"
    /// 5. Word-level match - "York" matches "New York"
    pub fn find_entity_by_name(&self, name: &str) -> Result<Option<EntityNode>> {
        // Tier 1: Exact match (O(1))
        let uuid = {
            let index = self.entity_name_index.read();
            index.get(name).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 2: Case-insensitive match (O(1))
        let name_lower = name.to_lowercase();
        let uuid = {
            let lowercase_index = self.entity_lowercase_index.read();
            lowercase_index.get(&name_lower).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 3: Stemmed match (O(1)) - "running" matches "run", "conversations" matches "conversation"
        let stemmer = Stemmer::create(Algorithm::English);
        let stemmed_name = Self::stem_entity_name(&stemmer, name);
        let uuid = {
            let stemmed_index = self.entity_stemmed_index.read();
            stemmed_index.get(&stemmed_name).copied()
        };

        if let Some(uuid) = uuid {
            return self.get_entity(&uuid);
        }

        // Tier 4 & 5: Fuzzy matching (O(n) but bounded)
        // Only do fuzzy matching for names >= 3 chars to avoid noise
        // Deterministic: collect ALL matches, pick highest salience (break ties by shortest name)
        if name.len() >= 3 {
            let lowercase_index = self.entity_lowercase_index.read();
            let mut candidates: Vec<(Uuid, String)> = Vec::new();

            // Tier 4: Substring match - query is substring of entity
            // e.g., "York" matches "New York City"
            for (entity_name, uuid) in lowercase_index.iter() {
                if entity_name.contains(&name_lower) {
                    candidates.push((*uuid, entity_name.clone()));
                }
            }

            // Tier 5: Word-level match (only if Tier 4 found nothing)
            if candidates.is_empty() {
                let query_words: Vec<&str> = name_lower.split_whitespace().collect();
                for (entity_name, uuid) in lowercase_index.iter() {
                    let entity_words: Vec<&str> = entity_name.split_whitespace().collect();
                    for qw in &query_words {
                        if entity_words.iter().any(|ew| ew == qw || ew.starts_with(qw)) {
                            candidates.push((*uuid, entity_name.clone()));
                            break;
                        }
                    }
                }
            }

            // Pick best candidate: highest salience, then shortest name for ties
            if !candidates.is_empty() {
                let mut best: Option<(Uuid, f32, usize)> = None; // (uuid, salience, name_len)
                for (uuid, name) in &candidates {
                    let salience = self.get_entity(uuid)?.map(|e| e.salience).unwrap_or(0.0);
                    match &best {
                        Some((_, best_sal, best_len))
                            if salience > *best_sal
                                || (salience == *best_sal && name.len() < *best_len) =>
                        {
                            best = Some((*uuid, salience, name.len()));
                        }
                        None => {
                            best = Some((*uuid, salience, name.len()));
                        }
                        _ => {}
                    }
                }
                if let Some((uuid, _, _)) = best {
                    return self.get_entity(&uuid);
                }
            }
        }

        Ok(None)
    }

    /// Find all entities matching a name with fuzzy matching
    ///
    /// Returns multiple matches ranked by match quality.
    /// Useful for spreading activation across related entities.
    pub fn find_entities_fuzzy(&self, name: &str, max_results: usize) -> Result<Vec<EntityNode>> {
        let mut results = Vec::new();
        let name_lower = name.to_lowercase();

        // Skip very short queries
        if name.len() < 2 {
            return Ok(results);
        }

        let lowercase_index = self.entity_lowercase_index.read();

        // Score and collect matches
        let mut scored: Vec<(Uuid, f32)> = Vec::new();

        for (entity_name, uuid) in lowercase_index.iter() {
            let score = if entity_name == &name_lower {
                1.0 // Exact match
            } else if entity_name.starts_with(&name_lower) {
                0.9 // Prefix match
            } else if entity_name.contains(&name_lower) {
                0.7 // Substring match
            } else {
                // Word-level match
                let entity_words: Vec<&str> = entity_name.split_whitespace().collect();
                let query_words: Vec<&str> = name_lower.split_whitespace().collect();

                let mut word_score: f32 = 0.0;
                for qw in &query_words {
                    for ew in &entity_words {
                        if ew == qw {
                            word_score += 0.5;
                        } else if ew.starts_with(qw) {
                            word_score += 0.3;
                        }
                    }
                }
                word_score.min(0.6) // Cap word-level score
            };

            if score > 0.0 {
                scored.push((*uuid, score));
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Take top results
        for (uuid, _score) in scored.into_iter().take(max_results) {
            if let Some(entity) = self.get_entity(&uuid)? {
                results.push(entity);
            }
        }

        Ok(results)
    }

    /// Canonical pair key for the entity-pair index.
    /// Uses min/max UUID ordering so A→B and B→A produce the same key.
    fn pair_key(entity_a: &Uuid, entity_b: &Uuid) -> String {
        if entity_a < entity_b {
            format!("{entity_a}:{entity_b}")
        } else {
            format!("{entity_b}:{entity_a}")
        }
    }

    /// Index an entity pair → edge UUID for O(1) dedup lookups
    fn index_entity_pair(&self, entity_a: &Uuid, entity_b: &Uuid, edge_uuid: &Uuid) -> Result<()> {
        let key = Self::pair_key(entity_a, entity_b);
        self.db.put_cf(
            self.entity_pair_index_cf(),
            key.as_bytes(),
            edge_uuid.as_bytes(),
        )?;
        Ok(())
    }

    /// Remove entity pair from the pair index
    fn remove_entity_pair_index(&self, entity_a: &Uuid, entity_b: &Uuid) -> Result<()> {
        let key = Self::pair_key(entity_a, entity_b);
        self.db
            .delete_cf(self.entity_pair_index_cf(), key.as_bytes())?;
        Ok(())
    }

    /// Find existing relationship between two entities (either direction)
    ///
    /// O(1) lookup via entity-pair index, with fallback to linear scan
    /// for edges created before the pair index existed (migration path).
    pub fn find_relationship_between(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        // Fast path: O(1) pair index lookup
        let key = Self::pair_key(entity_a, entity_b);
        if let Some(edge_uuid_bytes) = self
            .db
            .get_cf(self.entity_pair_index_cf(), key.as_bytes())?
        {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    return Ok(Some(edge));
                }
                // Edge was deleted but pair index is stale — clean up and fall through
                let _ = self
                    .db
                    .delete_cf(self.entity_pair_index_cf(), key.as_bytes());
            }
        }

        // Slow path: linear scan for pre-index edges (backward compatibility)
        // This path is only hit for edges created before the pair index existed.
        // Once all old edges are either strengthened (which updates the index) or
        // pruned, this path becomes dead code.
        let edges_a = self.get_entity_relationships(entity_a)?;
        for edge in edges_a {
            if (edge.from_entity == *entity_a && edge.to_entity == *entity_b)
                || (edge.from_entity == *entity_b && edge.to_entity == *entity_a)
            {
                // Backfill pair index for this legacy edge
                let _ = self.index_entity_pair(entity_a, entity_b, &edge.uuid);
                return Ok(Some(edge));
            }
        }
        Ok(None)
    }

    /// Find existing relationship between two entities with a specific relation type.
    ///
    /// Unlike `find_relationship_between` which returns any edge between the pair,
    /// this method only matches edges with the same `RelationType`. This allows
    /// multiple semantically distinct edges (e.g. WorksWith + PartOf) between
    /// the same entity pair.
    pub fn find_relationship_between_typed(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
        relation_type: &RelationType,
    ) -> Result<Option<RelationshipEdge>> {
        let edges = self.get_entity_relationships(entity_a)?;
        for edge in edges {
            if edge.relation_type == *relation_type
                && ((edge.from_entity == *entity_a && edge.to_entity == *entity_b)
                    || (edge.from_entity == *entity_b && edge.to_entity == *entity_a))
            {
                return Ok(Some(edge));
            }
        }
        Ok(None)
    }

    /// Add a relationship edge (or strengthen existing one)
    ///
    /// If an edge already exists between the two entities, strengthens it
    /// instead of creating a duplicate. This implements proper Hebbian learning:
    /// "neurons that fire together, wire together" - repeated co-occurrence
    /// strengthens the same synapse rather than creating parallel connections.
    pub fn add_relationship(&self, mut edge: RelationshipEdge) -> Result<Uuid> {
        // Check for existing relationship between these entities WITH SAME TYPE
        // Different relation types (e.g. WorksWith vs PartOf) are distinct edges
        if let Some(mut existing) = self.find_relationship_between_typed(
            &edge.from_entity,
            &edge.to_entity,
            &edge.relation_type,
        )? {
            // Strengthen existing edge instead of creating duplicate
            let _ = existing.strengthen();
            existing.last_activated = Utc::now();

            // Update context if new context is more informative
            if edge.context.len() > existing.context.len() {
                existing.context = edge.context;
            }

            // Persist the strengthened edge
            let key = existing.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&existing, bincode::config::standard())?;
            self.db.put_cf(self.relationships_cf(), key, value)?;

            return Ok(existing.uuid);
        }

        // No existing edge - create new one
        edge.uuid = Uuid::new_v4();
        edge.created_at = Utc::now();

        // Store relationship
        let key = edge.uuid.as_bytes();
        let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
        self.db.put_cf(self.relationships_cf(), key, value)?;

        // Increment relationship counter
        self.relationship_count.fetch_add(1, Ordering::Relaxed);

        // Update entity->edges index for both entities
        self.index_entity_edge(&edge.from_entity, &edge.uuid)?;
        self.index_entity_edge(&edge.to_entity, &edge.uuid)?;

        // Update entity-pair index for O(1) dedup lookups
        self.index_entity_pair(&edge.from_entity, &edge.to_entity, &edge.uuid)?;

        // Insert-time degree pruning: cap edges per entity to prevent O(n²) explosion.
        // If either entity exceeds MAX_ENTITY_DEGREE, prune the weakest edges.
        // This is the primary defense against graph bloat (132MB for 600KB of content).
        self.prune_entity_if_over_degree(&edge.from_entity)?;
        self.prune_entity_if_over_degree(&edge.to_entity)?;

        Ok(edge.uuid)
    }

    /// Index an edge for an entity
    fn index_entity_edge(&self, entity_uuid: &Uuid, edge_uuid: &Uuid) -> Result<()> {
        let key = format!("{entity_uuid}:{edge_uuid}");
        self.db
            .put_cf(self.entity_edges_cf(), key.as_bytes(), b"1")?;
        Ok(())
    }

    /// Prune an entity's edges if degree exceeds MAX_ENTITY_DEGREE
    ///
    /// Loads all edges for the entity, sorts by effective strength, and deletes
    /// the weakest edges that exceed the cap. LTP-protected edges are preserved
    /// preferentially (sorted last, so they survive pruning).
    ///
    /// This is called at insert time to prevent unbounded edge growth.
    /// Amortized cost: O(1) for most insertions (only triggers when over cap),
    /// O(d log d) when pruning is needed (d = entity degree).
    fn prune_entity_if_over_degree(&self, entity_uuid: &Uuid) -> Result<()> {
        use crate::constants::MAX_ENTITY_DEGREE;

        // Fast path: count edges without loading them
        let prefix = format!("{entity_uuid}:");
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_edges_cf(), prefix.as_bytes());

        let mut edge_count = 0usize;
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                edge_count += 1;
            }
        }

        if edge_count <= MAX_ENTITY_DEGREE {
            return Ok(());
        }

        // Over cap — load all edges, sort, prune weakest
        let all_edges = self.get_entity_relationships(entity_uuid)?;
        if all_edges.len() <= MAX_ENTITY_DEGREE {
            return Ok(()); // Race condition guard
        }

        // Sort by pruning priority: LTP-protected edges last (survive pruning),
        // then by effective strength descending (strongest survive)
        let mut scored: Vec<(Uuid, f32, bool)> = all_edges
            .iter()
            .map(|e| {
                let is_protected = e.is_potentiated();
                (e.uuid, e.effective_strength(), is_protected)
            })
            .collect();

        // Sort: unprotected+weak first (pruning candidates), protected+strong last (survivors)
        scored.sort_by(|a, b| {
            // Protected edges sort after unprotected
            match a.2.cmp(&b.2) {
                CmpOrdering::Equal => {
                    // Within same protection class, weaker edges first (prune candidates)
                    a.1.total_cmp(&b.1)
                }
                other => other,
            }
        });

        // Prune excess: first N edges in sorted order are weakest/unprotected
        let prune_count = scored.len() - MAX_ENTITY_DEGREE;
        let to_prune: Vec<Uuid> = scored.iter().take(prune_count).map(|s| s.0).collect();

        for edge_uuid in &to_prune {
            if let Err(e) = self.delete_relationship(edge_uuid) {
                tracing::warn!(
                    edge = %edge_uuid,
                    entity = %entity_uuid,
                    "Failed to prune edge during degree cap: {}",
                    e
                );
            }
        }

        if !to_prune.is_empty() {
            tracing::info!(
                entity = %entity_uuid,
                pruned = to_prune.len(),
                remaining = MAX_ENTITY_DEGREE,
                "Pruned edges exceeding degree cap"
            );
        }

        Ok(())
    }

    /// Get relationships for an entity with optional limit
    ///
    /// Uses batch reading (multi_get) to eliminate N+1 query pattern.
    /// If limit is None, returns all edges (use sparingly for large graphs).
    pub fn get_entity_relationships(&self, entity_uuid: &Uuid) -> Result<Vec<RelationshipEdge>> {
        self.get_entity_relationships_limited(entity_uuid, None)
    }

    /// Get relationships for an entity with limit, ordered by effective strength
    ///
    /// Collects ALL edge UUIDs first, batch-reads them, sorts by effective_strength
    /// descending, then returns the top `limit` strongest edges. This ensures
    /// traversal and queries always use the most valuable connections.
    ///
    /// When no limit is specified, returns all edges sorted by strength.
    pub fn get_entity_relationships_limited(
        &self,
        entity_uuid: &Uuid,
        limit: Option<usize>,
    ) -> Result<Vec<RelationshipEdge>> {
        let prefix = format!("{entity_uuid}:");

        // Phase 1: Collect ALL edge UUIDs from index (fast prefix scan)
        // We must read all to sort by strength — storage order is arbitrary
        let mut edge_uuids: Vec<Uuid> = Vec::with_capacity(256);
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_edges_cf(), prefix.as_bytes());

        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }

                if let Some(edge_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(edge_uuid) = Uuid::parse_str(edge_uuid_str) {
                        edge_uuids.push(edge_uuid);
                    }
                }
            }
        }

        if edge_uuids.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Batch read all edges using multi_get (single RocksDB call)
        let keys: Vec<[u8; 16]> = edge_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();

        let results = self
            .db
            .batched_multi_get_cf(self.relationships_cf(), &key_refs, false);

        let mut edges = Vec::with_capacity(edge_uuids.len());
        for value in results.into_iter().flatten().flatten() {
            if let Ok((edge, _)) = bincode::serde::decode_from_slice::<RelationshipEdge, _>(
                &value,
                bincode::config::standard(),
            ) {
                edges.push(edge);
            }
        }

        // Phase 3: Sort by effective strength descending (strongest first)
        edges.sort_by(|a, b| b.effective_strength().total_cmp(&a.effective_strength()));

        // Phase 3.5: Opportunistic pruning — queue edges that have decayed below
        // their tier's threshold for batch deletion on next maintenance cycle.
        // This replaces the eager full-scan apply_decay() with lazy on-read pruning.
        let mut has_prunable = false;
        for edge in &edges {
            if edge.effective_strength() < edge.tier.prune_threshold()
                && !edge.ltp_status.is_potentiated()
            {
                has_prunable = true;
                break;
            }
        }
        if has_prunable {
            let mut prune_queue = self.pending_prune.lock();
            let mut orphan_queue = self.pending_orphan_checks.lock();
            edges.retain(|edge| {
                if edge.effective_strength() < edge.tier.prune_threshold()
                    && !edge.ltp_status.is_potentiated()
                {
                    prune_queue.push(edge.uuid);
                    orphan_queue.push(edge.from_entity);
                    orphan_queue.push(edge.to_entity);
                    false // remove from results
                } else {
                    true
                }
            });
        }

        // Phase 4: Truncate to limit if specified
        if let Some(max) = limit {
            edges.truncate(max);
        }

        Ok(edges)
    }

    /// Calculate edge density for a specific entity (SHO-D5)
    ///
    /// Returns the number of edges connected to this entity.
    /// Used for per-entity density calculation: dense entities use vector search,
    /// sparse entities use graph search.
    ///
    /// This is an O(1) prefix count operation.
    pub fn entity_edge_count(&self, entity_uuid: &Uuid) -> Result<usize> {
        let prefix = format!("{entity_uuid}:");
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_edges_cf(), prefix.as_bytes());

        let mut count = 0;
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                count += 1;
            }
        }

        Ok(count)
    }

    /// Calculate average edge density for a set of entities (SHO-D5)
    ///
    /// Returns the mean number of edges per entity for the given UUIDs.
    /// Used to determine optimal retrieval strategy:
    /// - Low density (<5 edges): Trust graph search (sparse, high-signal)
    /// - High density (>20 edges): Trust vector search (dense, noisy)
    ///
    /// Returns None if no entities provided.
    pub fn entities_average_density(&self, entity_uuids: &[Uuid]) -> Result<Option<f32>> {
        if entity_uuids.is_empty() {
            return Ok(None);
        }

        let mut total_edges = 0usize;
        for uuid in entity_uuids {
            total_edges += self.entity_edge_count(uuid)?;
        }

        Ok(Some(total_edges as f32 / entity_uuids.len() as f32))
    }

    /// Calculate edge density per tier for a specific entity (SHO-D5)
    ///
    /// Returns counts of edges by tier: (L1_count, L2_count, L3_count, LTP_count)
    /// Useful for understanding if an entity's graph is consolidated (mostly L3/LTP)
    /// or still noisy (mostly L1).
    pub fn entity_density_by_tier(
        &self,
        entity_uuid: &Uuid,
    ) -> Result<(usize, usize, usize, usize)> {
        let edges = self.get_entity_relationships(entity_uuid)?;

        let mut l1_count = 0;
        let mut l2_count = 0;
        let mut l3_count = 0;
        let mut ltp_count = 0;

        for edge in edges {
            if edge.is_potentiated() {
                ltp_count += 1;
            } else {
                match edge.tier {
                    EdgeTier::L1Working => l1_count += 1,
                    EdgeTier::L2Episodic => l2_count += 1,
                    EdgeTier::L3Semantic => l3_count += 1,
                }
            }
        }

        Ok((l1_count, l2_count, l3_count, ltp_count))
    }

    /// Calculate consolidated ratio for an entity (SHO-D5)
    ///
    /// Returns the ratio of consolidated edges (L2 + L3 + LTP) to total edges.
    /// High ratio (>0.7) = trust graph search, Low ratio (<0.3) = trust vector search.
    ///
    /// Returns None if entity has no edges.
    pub fn entity_consolidation_ratio(&self, entity_uuid: &Uuid) -> Result<Option<f32>> {
        let (l1, l2, l3, ltp) = self.entity_density_by_tier(entity_uuid)?;
        let total = l1 + l2 + l3 + ltp;

        if total == 0 {
            return Ok(None);
        }

        let consolidated = l2 + l3 + ltp;
        Ok(Some(consolidated as f32 / total as f32))
    }

    /// Get relationship by UUID (raw, without decay applied)
    pub fn get_relationship(&self, uuid: &Uuid) -> Result<Option<RelationshipEdge>> {
        let key = uuid.as_bytes();
        match self.db.get_cf(self.relationships_cf(), key)? {
            Some(value) => {
                let (edge, _): (RelationshipEdge, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Get relationship by UUID with effective strength (lazy decay calculation)
    ///
    /// Returns the edge with strength reflecting time-based decay.
    /// This doesn't persist the decay - just calculates what the strength would be.
    /// Use this for API responses to show accurate current strength.
    pub fn get_relationship_with_effective_strength(
        &self,
        uuid: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        let key = uuid.as_bytes();
        match self.db.get_cf(self.relationships_cf(), key)? {
            Some(value) => {
                let (mut edge, _): (RelationshipEdge, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                // Apply effective strength calculation (doesn't persist)
                edge.strength = edge.effective_strength();
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Delete a relationship by UUID
    ///
    /// Removes the relationship from storage and decrements the counter.
    /// Returns true if the relationship was found and deleted.
    pub fn delete_relationship(&self, uuid: &Uuid) -> Result<bool> {
        let key = uuid.as_bytes();

        // Get the edge first to find both entities for index cleanup
        let edge = match self.get_relationship(uuid)? {
            Some(e) => e,
            None => return Ok(false),
        };

        // Delete from main storage
        self.db.delete_cf(self.relationships_cf(), key)?;
        self.relationship_count.fetch_sub(1, Ordering::Relaxed);

        // Remove from entity_edges index for BOTH entities
        // (add_relationship indexes both from_entity and to_entity)
        let from_key = format!("{}:{}", edge.from_entity, uuid);
        if let Err(e) = self
            .db
            .delete_cf(self.entity_edges_cf(), from_key.as_bytes())
        {
            tracing::warn!(edge = %uuid, key = %from_key, error = %e, "Failed to delete from entity_edges index");
        }
        let to_key = format!("{}:{}", edge.to_entity, uuid);
        if let Err(e) = self.db.delete_cf(self.entity_edges_cf(), to_key.as_bytes()) {
            tracing::warn!(edge = %uuid, key = %to_key, error = %e, "Failed to delete from entity_edges index");
        }

        // Remove from entity-pair index
        if let Err(e) = self.remove_entity_pair_index(&edge.from_entity, &edge.to_entity) {
            tracing::warn!(edge = %uuid, "Failed to delete from entity_pair index: {}", e);
        }

        Ok(true)
    }

    /// Delete an episode and clean up associated indices and orphan edges
    ///
    /// When a memory is deleted, its corresponding episode should also be removed.
    /// This method:
    /// 1. Removes the episode from the episodes DB
    /// 2. Removes entity_episodes index entries
    /// 3. Deletes relationship edges that were sourced from this episode
    pub fn delete_episode(&self, episode_uuid: &Uuid) -> Result<bool> {
        // Fetch episode to get entity_refs for index cleanup
        let episode = match self.get_episode(episode_uuid)? {
            Some(ep) => ep,
            None => return Ok(false),
        };

        // Delete episode from main storage
        self.db
            .delete_cf(self.episodes_cf(), episode_uuid.as_bytes())?;
        self.episode_count.fetch_sub(1, Ordering::Relaxed);

        // Remove from entity_episodes inverted index
        for entity_uuid in &episode.entity_refs {
            let key = format!("{entity_uuid}:{episode_uuid}");
            if let Err(e) = self.db.delete_cf(self.entity_episodes_cf(), key.as_bytes()) {
                tracing::warn!(episode = %episode_uuid, key = %key, error = %e, "Failed to delete from entity_episodes index");
            }
        }

        // Delete edges sourced from this episode
        // Scan all relationships for matching source_episode_id
        let iter = self
            .db
            .iterator_cf(self.relationships_cf(), rocksdb::IteratorMode::Start);
        let mut edges_to_delete = Vec::new();
        for (_, value) in iter.flatten() {
            if let Ok((edge, _)) = bincode::serde::decode_from_slice::<RelationshipEdge, _>(
                &value,
                bincode::config::standard(),
            ) {
                if edge.source_episode_id == Some(*episode_uuid) {
                    edges_to_delete.push(edge.uuid);
                }
            }
        }

        for edge_uuid in &edges_to_delete {
            if let Err(e) = self.delete_relationship(edge_uuid) {
                tracing::debug!("Failed to delete orphan edge {}: {}", edge_uuid, e);
            }
        }

        tracing::debug!(
            "Deleted episode {} with {} entity_refs and {} sourced edges",
            &episode_uuid.to_string()[..8],
            episode.entity_refs.len(),
            edges_to_delete.len()
        );

        Ok(true)
    }

    /// Clear all graph data (GDPR full erasure)
    ///
    /// Wipes all entities, relationships, episodes, and all indices.
    /// Resets all counters to zero.
    /// Returns (entity_count, relationship_count, episode_count) that were cleared.
    pub fn clear_all(&self) -> Result<(usize, usize, usize)> {
        let entity_count = self.entity_count.load(Ordering::Relaxed);
        let relationship_count = self.relationship_count.load(Ordering::Relaxed);
        let episode_count = self.episode_count.load(Ordering::Relaxed);

        // Clear each column family by iterating and batch-deleting
        for cf_name in GRAPH_CF_NAMES {
            let cf = self.db.cf_handle(cf_name).unwrap();
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for (key, _) in iter.flatten() {
                batch.delete_cf(cf, &key);
            }
            self.db.write(batch)?;
        }

        // Clear in-memory indices
        self.entity_name_index.write().clear();
        self.entity_lowercase_index.write().clear();
        self.entity_stemmed_index.write().clear();

        // Reset counters
        self.entity_count.store(0, Ordering::Relaxed);
        self.relationship_count.store(0, Ordering::Relaxed);
        self.episode_count.store(0, Ordering::Relaxed);

        // Drain pending maintenance queues — they reference now-deleted entities/edges
        let _ = std::mem::take(&mut *self.pending_prune.lock());
        let _ = std::mem::take(&mut *self.pending_orphan_checks.lock());

        tracing::info!(
            "Graph data cleared (GDPR erasure): {} entities, {} relationships, {} episodes",
            entity_count,
            relationship_count,
            episode_count
        );
        Ok((entity_count, relationship_count, episode_count))
    }

    /// Add an episodic node
    pub fn add_episode(&self, episode: EpisodicNode) -> Result<Uuid> {
        let key = episode.uuid.as_bytes();
        let entity_count = episode.entity_refs.len();
        tracing::debug!(
            "add_episode: {} with {} entity_refs",
            &episode.uuid.to_string()[..8],
            entity_count
        );

        let value = bincode::serde::encode_to_vec(&episode, bincode::config::standard())?;
        self.db.put_cf(self.episodes_cf(), key, value)?;

        // Increment episode counter
        let prev = self.episode_count.fetch_add(1, Ordering::Relaxed);
        tracing::debug!("add_episode: count {} -> {}", prev, prev + 1);

        // Update inverted index: entity_uuid -> episode_uuid
        for entity_uuid in &episode.entity_refs {
            self.index_entity_episode(entity_uuid, &episode.uuid)?;
        }

        Ok(episode.uuid)
    }

    /// Index an episode for an entity (inverted index)
    fn index_entity_episode(&self, entity_uuid: &Uuid, episode_uuid: &Uuid) -> Result<()> {
        let key = format!("{entity_uuid}:{episode_uuid}");
        self.db
            .put_cf(self.entity_episodes_cf(), key.as_bytes(), b"1")?;
        Ok(())
    }

    /// Get episode by UUID
    pub fn get_episode(&self, uuid: &Uuid) -> Result<Option<EpisodicNode>> {
        let key = uuid.as_bytes();
        match self.db.get_cf(self.episodes_cf(), key)? {
            Some(value) => {
                let (episode, _): (EpisodicNode, _) =
                    bincode::serde::decode_from_slice(&value, bincode::config::standard())?;
                Ok(Some(episode))
            }
            None => Ok(None),
        }
    }

    /// Get all episodes that contain a specific entity
    ///
    /// Uses inverted index for O(k) lookup instead of O(n) full scan.
    /// Collects episode UUIDs first, then batch-reads them using multi_get.
    /// Crucial for spreading activation algorithm.
    pub fn get_episodes_by_entity(&self, entity_uuid: &Uuid) -> Result<Vec<EpisodicNode>> {
        let prefix = format!("{entity_uuid}:");
        tracing::debug!("get_episodes_by_entity: prefix {}", &prefix[..12]);

        // Phase 1: Collect episode UUIDs from index (fast prefix scan, no data transfer)
        let mut episode_uuids: Vec<Uuid> = Vec::new();
        let iter = self
            .db
            .prefix_iterator_cf(self.entity_episodes_cf(), prefix.as_bytes());
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(episode_uuid_str) = key_str.split(':').nth(1) {
                    if let Ok(episode_uuid) = Uuid::parse_str(episode_uuid_str) {
                        episode_uuids.push(episode_uuid);
                    }
                }
            }
        }

        if episode_uuids.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Batch read all episodes using multi_get (single RocksDB call)
        let keys: Vec<[u8; 16]> = episode_uuids.iter().map(|u| *u.as_bytes()).collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();

        let results = self
            .db
            .batched_multi_get_cf(self.episodes_cf(), &key_refs, false);

        let mut episodes = Vec::with_capacity(episode_uuids.len());
        for value in results.into_iter().flatten().flatten() {
            if let Ok((episode, _)) = bincode::serde::decode_from_slice::<EpisodicNode, _>(
                &value,
                bincode::config::standard(),
            ) {
                episodes.push(episode);
            }
        }

        tracing::debug!("get_episodes_by_entity: found {} episodes", episodes.len());
        Ok(episodes)
    }

    /// Traverse graph starting from an entity (breadth-first)
    ///
    /// Implements Hebbian learning: edges traversed during retrieval are strengthened.
    /// This means frequently accessed pathways become stronger over time.
    ///
    /// Returns `TraversedEntity` with hop distance and decay factor for proper scoring:
    /// - hop 0 (start entity): decay = 1.0
    /// - hop 1: decay = 0.7
    /// - hop 2: decay = 0.49
    /// - etc.
    ///
    /// Performance: Uses batch edge reading and limits to handle large graphs.
    pub fn traverse_from_entity(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
    ) -> Result<GraphTraversal> {
        self.traverse_from_entity_filtered(start_uuid, max_depth, None)
    }

    /// BFS graph traversal with optional minimum edge strength filter.
    ///
    /// When `min_strength` is Some, edges below the threshold are skipped
    /// during traversal and NOT Hebbianly strengthened (prevents ghost edge revival).
    pub fn traverse_from_entity_filtered(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
        min_strength: Option<f32>,
    ) -> Result<GraphTraversal> {
        // Performance limits
        const MAX_ENTITIES: usize = 200;
        const MAX_EDGES_PER_NODE: usize = 100;

        // Use tuned decay from constants (0.15 max decay → ~86% retention per hop)
        // This enables deeper traversal than the old 0.7 factor
        use crate::constants::IMPORTANCE_DECAY_MAX;
        let hop_decay_factor: f32 = (-IMPORTANCE_DECAY_MAX).exp(); // e^(-0.15) ≈ 0.86

        let mut visited_entities = HashSet::new();
        let mut visited_edges = HashSet::new();
        let mut current_level: Vec<(Uuid, usize)> = vec![(*start_uuid, 0)]; // (uuid, hop_distance)
        let mut all_entities: Vec<TraversedEntity> = Vec::new();
        let mut all_edges = Vec::new();
        let mut edges_to_strengthen = Vec::new();

        visited_entities.insert(*start_uuid);
        if let Some(entity) = self.get_entity(start_uuid)? {
            all_entities.push(TraversedEntity {
                entity,
                hop_distance: 0,
                decay_factor: 1.0,
            });
        }

        for depth in 0..max_depth {
            // Early termination if we have enough entities
            if all_entities.len() >= MAX_ENTITIES {
                break;
            }

            let mut next_level = Vec::new();

            for (entity_uuid, _hop) in &current_level {
                // Use limited edge reading
                let edges =
                    self.get_entity_relationships_limited(entity_uuid, Some(MAX_EDGES_PER_NODE))?;

                for edge in edges {
                    if visited_edges.contains(&edge.uuid) {
                        continue;
                    }

                    visited_edges.insert(edge.uuid);

                    // Only traverse non-invalidated edges
                    if edge.invalidated_at.is_some() {
                        continue;
                    }

                    // Compute effective strength (lazy decay calculation)
                    let effective = edge.effective_strength();

                    // Skip weak edges if min_strength filter is set
                    if let Some(threshold) = min_strength {
                        if effective < threshold {
                            continue;
                        }
                    }

                    // Collect edge UUID for Hebbian strengthening (only for traversed edges)
                    edges_to_strengthen.push(edge.uuid);

                    // Return edge with effective strength
                    let mut edge_with_decay = edge.clone();
                    edge_with_decay.strength = effective;
                    all_edges.push(edge_with_decay);

                    // Add connected entity
                    let connected_uuid = if edge.from_entity == *entity_uuid {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };

                    if !visited_entities.contains(&connected_uuid) {
                        visited_entities.insert(connected_uuid);
                        let next_hop = depth + 1;
                        let decay = hop_decay_factor.powi(next_hop as i32);

                        if let Some(entity) = self.get_entity(&connected_uuid)? {
                            all_entities.push(TraversedEntity {
                                entity,
                                hop_distance: next_hop,
                                decay_factor: decay,
                            });
                        }
                        next_level.push((connected_uuid, next_hop));
                    }
                }
            }

            if next_level.is_empty() {
                break;
            }

            current_level = next_level;
        }

        // Apply Hebbian strengthening to all traversed edges atomically (SHO-65)
        // "Neurons that fire together, wire together"
        // Uses batch update for efficiency instead of individual writes
        if !edges_to_strengthen.is_empty() {
            match self.batch_strengthen_synapses(&edges_to_strengthen) {
                Ok(count) => {
                    if count > 0 {
                        tracing::trace!("Strengthened {} synapses during traversal", count);
                    }
                }
                Err(e) => {
                    tracing::debug!("Failed to batch strengthen synapses: {}", e);
                }
            }
        }

        Ok(GraphTraversal {
            entities: all_entities,
            relationships: all_edges,
        })
    }

    /// Weighted graph traversal with filtering (Dijkstra-style best-first)
    ///
    /// Unlike BFS traverse_from_entity, this uses edge weights to prioritize
    /// stronger connections. Enables Cypher-like pattern matching:
    /// - Filter by relationship types
    /// - Filter by minimum edge strength
    /// - Score paths by cumulative weight
    ///
    /// Returns entities sorted by path score (strongest connections first).
    ///
    /// Performance: Uses batch edge reading and early termination to handle
    /// large graphs (10000+ edges) efficiently.
    pub fn traverse_weighted(
        &self,
        start_uuid: &Uuid,
        max_depth: usize,
        relation_types: Option<&[RelationType]>,
        min_strength: f32,
    ) -> Result<GraphTraversal> {
        // Performance limits - prevents exponential blowup on dense graphs
        const MAX_ENTITIES: usize = 200; // Stop after finding this many entities
        const MAX_EDGES_PER_NODE: usize = 100; // Limit edges loaded per node
        const MAX_ITERATIONS: usize = 500; // Prevent infinite loops

        // Priority queue entry: (negative score for max-heap, uuid, depth, path_score)
        #[derive(Clone)]
        struct PQEntry {
            score: f32,
            uuid: Uuid,
            depth: usize,
        }
        impl PartialEq for PQEntry {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }
        impl Eq for PQEntry {}
        impl PartialOrd for PQEntry {
            fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for PQEntry {
            fn cmp(&self, other: &Self) -> CmpOrdering {
                // BinaryHeap is a max-heap: higher score = popped first = explored first
                self.score.total_cmp(&other.score)
            }
        }

        let mut visited: HashMap<Uuid, f32> = HashMap::new(); // uuid -> best path score
        let mut heap: BinaryHeap<PQEntry> = BinaryHeap::new();
        let mut all_entities: Vec<TraversedEntity> = Vec::new();
        let mut all_edges: Vec<RelationshipEdge> = Vec::new();
        let mut edges_to_strengthen: Vec<Uuid> = Vec::new();
        let mut iterations = 0;

        // Start node
        heap.push(PQEntry {
            score: 1.0,
            uuid: *start_uuid,
            depth: 0,
        });
        visited.insert(*start_uuid, 1.0);

        if let Some(entity) = self.get_entity(start_uuid)? {
            all_entities.push(TraversedEntity {
                entity,
                hop_distance: 0,
                decay_factor: 1.0,
            });
        }

        while let Some(PQEntry { score, uuid, depth }) = heap.pop() {
            iterations += 1;

            // Early termination: entity/iteration limits reached, stop draining heap
            if all_entities.len() >= MAX_ENTITIES || iterations >= MAX_ITERATIONS {
                break;
            }

            // Depth limit: skip this node's children but keep processing others
            if depth >= max_depth {
                continue;
            }

            // Skip if we've found a better path to this node
            if let Some(&best) = visited.get(&uuid) {
                if score < best * 0.99 {
                    continue;
                }
            }

            // Use limited edge reading to avoid loading 10000+ edges
            let edges = self.get_entity_relationships_limited(&uuid, Some(MAX_EDGES_PER_NODE))?;

            for edge in edges {
                // Skip invalidated edges
                if edge.invalidated_at.is_some() {
                    continue;
                }

                // Filter by relationship type if specified
                if let Some(types) = relation_types {
                    if !types.contains(&edge.relation_type) {
                        continue;
                    }
                }

                // Filter by minimum strength
                let effective = edge.effective_strength();
                if effective < min_strength {
                    continue;
                }

                // Track edge for Hebbian strengthening
                edges_to_strengthen.push(edge.uuid);

                let connected_uuid = if edge.from_entity == uuid {
                    edge.to_entity
                } else {
                    edge.from_entity
                };

                // Path score = parent_score * edge_strength (multiplicative decay)
                let new_score = score * effective;

                // Only visit if this is a better path
                let dominated = visited
                    .get(&connected_uuid)
                    .is_some_and(|&best| new_score <= best);
                if dominated {
                    continue;
                }

                visited.insert(connected_uuid, new_score);

                // Add edge to results
                let mut edge_with_strength = edge.clone();
                edge_with_strength.strength = effective;
                all_edges.push(edge_with_strength);

                // Add entity to results
                if let Some(entity) = self.get_entity(&connected_uuid)? {
                    all_entities.push(TraversedEntity {
                        entity,
                        hop_distance: depth + 1,
                        decay_factor: new_score,
                    });
                }

                heap.push(PQEntry {
                    score: new_score,
                    uuid: connected_uuid,
                    depth: depth + 1,
                });
            }
        }

        // Sort entities by path score (decay_factor) descending
        all_entities.sort_by(|a, b| b.decay_factor.total_cmp(&a.decay_factor));

        // Hebbian strengthening
        if !edges_to_strengthen.is_empty() {
            if let Err(e) = self.batch_strengthen_synapses(&edges_to_strengthen) {
                tracing::debug!("Failed to strengthen synapses: {}", e);
            }
        }

        tracing::debug!(
            "traverse_weighted: {} entities, {} edges (min_strength={:.2})",
            all_entities.len(),
            all_edges.len(),
            min_strength
        );

        Ok(GraphTraversal {
            entities: all_entities,
            relationships: all_edges,
        })
    }

    /// Bidirectional search between two entities (meet-in-middle)
    ///
    /// Complexity: O(b^(d/2)) instead of O(b^d) for unidirectional search.
    /// Finds the shortest weighted path between start and goal.
    /// Returns entities along the path with their path scores.
    ///
    /// Performance: Uses batch edge reading with limits.
    pub fn traverse_bidirectional(
        &self,
        start_uuid: &Uuid,
        goal_uuid: &Uuid,
        max_depth: usize,
        min_strength: f32,
    ) -> Result<GraphTraversal> {
        const MAX_EDGES_PER_NODE: usize = 100;

        // Track forward search from start
        let mut forward_visited: HashMap<Uuid, (f32, usize)> = HashMap::new(); // uuid -> (score, depth)
        let mut forward_parents: HashMap<Uuid, (Uuid, Uuid)> = HashMap::new(); // child -> (parent, edge_uuid)
        let mut forward_frontier: Vec<(Uuid, f32, usize)> = vec![(*start_uuid, 1.0, 0)];
        forward_visited.insert(*start_uuid, (1.0, 0));

        // Track backward search from goal
        let mut backward_visited: HashMap<Uuid, (f32, usize)> = HashMap::new();
        let mut backward_parents: HashMap<Uuid, (Uuid, Uuid)> = HashMap::new();
        let mut backward_frontier: Vec<(Uuid, f32, usize)> = vec![(*goal_uuid, 1.0, 0)];
        backward_visited.insert(*goal_uuid, (1.0, 0));

        let mut meeting_node: Option<Uuid> = None;
        let mut best_path_score: f32 = 0.0;
        let half_depth = max_depth / 2 + 1;

        // Alternate forward and backward expansion
        for _round in 0..half_depth {
            // Expand forward frontier
            let mut new_forward: Vec<(Uuid, f32, usize)> = Vec::new();
            for (uuid, score, depth) in forward_frontier.drain(..) {
                if depth >= half_depth {
                    continue;
                }

                let edges =
                    self.get_entity_relationships_limited(&uuid, Some(MAX_EDGES_PER_NODE))?;
                for edge in edges {
                    if edge.invalidated_at.is_some() {
                        continue;
                    }
                    let effective = edge.effective_strength();
                    if effective < min_strength {
                        continue;
                    }

                    let connected = if edge.from_entity == uuid {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };
                    let new_score = score * effective;

                    // Check if we meet backward search
                    if let Some(&(back_score, _)) = backward_visited.get(&connected) {
                        let combined = new_score * back_score;
                        if combined > best_path_score {
                            best_path_score = combined;
                            meeting_node = Some(connected);
                        }
                    }

                    // Update forward frontier
                    let dominated = forward_visited
                        .get(&connected)
                        .is_some_and(|&(best, _)| new_score <= best);
                    if !dominated {
                        forward_visited.insert(connected, (new_score, depth + 1));
                        forward_parents.insert(connected, (uuid, edge.uuid));
                        new_forward.push((connected, new_score, depth + 1));
                    }
                }
            }
            forward_frontier = new_forward;

            // Expand backward frontier
            let mut new_backward: Vec<(Uuid, f32, usize)> = Vec::new();
            for (uuid, score, depth) in backward_frontier.drain(..) {
                if depth >= half_depth {
                    continue;
                }

                let edges =
                    self.get_entity_relationships_limited(&uuid, Some(MAX_EDGES_PER_NODE))?;
                for edge in edges {
                    if edge.invalidated_at.is_some() {
                        continue;
                    }
                    let effective = edge.effective_strength();
                    if effective < min_strength {
                        continue;
                    }

                    let connected = if edge.from_entity == uuid {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };
                    let new_score = score * effective;

                    // Check if we meet forward search
                    if let Some(&(fwd_score, _)) = forward_visited.get(&connected) {
                        let combined = fwd_score * new_score;
                        if combined > best_path_score {
                            best_path_score = combined;
                            meeting_node = Some(connected);
                        }
                    }

                    // Update backward frontier
                    let dominated = backward_visited
                        .get(&connected)
                        .is_some_and(|&(best, _)| new_score <= best);
                    if !dominated {
                        backward_visited.insert(connected, (new_score, depth + 1));
                        backward_parents.insert(connected, (uuid, edge.uuid));
                        new_backward.push((connected, new_score, depth + 1));
                    }
                }
            }
            backward_frontier = new_backward;

            // Early termination if we found a meeting point
            if meeting_node.is_some() {
                break;
            }
        }

        // Reconstruct path from meeting node
        let mut all_entities: Vec<TraversedEntity> = Vec::new();
        let mut all_edges: Vec<RelationshipEdge> = Vec::new();
        let mut edges_to_strengthen: Vec<Uuid> = Vec::new();

        if let Some(meeting) = meeting_node {
            // Forward path: start -> meeting
            let mut path_forward: Vec<Uuid> = vec![meeting];
            let mut current = meeting;
            while let Some(&(parent, edge_uuid)) = forward_parents.get(&current) {
                path_forward.push(parent);
                edges_to_strengthen.push(edge_uuid);
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    all_edges.push(edge);
                }
                current = parent;
            }
            path_forward.reverse();

            // Backward path: meeting -> goal
            let mut path_backward: Vec<Uuid> = Vec::new();
            current = meeting;
            while let Some(&(parent, edge_uuid)) = backward_parents.get(&current) {
                path_backward.push(parent);
                edges_to_strengthen.push(edge_uuid);
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    all_edges.push(edge);
                }
                current = parent;
            }

            // Combine paths
            let full_path: Vec<Uuid> = path_forward.into_iter().chain(path_backward).collect();

            // Build entities with scores
            for (i, uuid) in full_path.iter().enumerate() {
                if let Some(entity) = self.get_entity(uuid)? {
                    let score = forward_visited
                        .get(uuid)
                        .map(|&(s, _)| s)
                        .or_else(|| backward_visited.get(uuid).map(|&(s, _)| s))
                        .unwrap_or(0.5);
                    all_entities.push(TraversedEntity {
                        entity,
                        hop_distance: i,
                        decay_factor: score,
                    });
                }
            }
        } else {
            // No path found - return empty traversal
            tracing::debug!(
                "traverse_bidirectional: no path between {:?} and {:?}",
                start_uuid,
                goal_uuid
            );
        }

        // Hebbian strengthening for traversed edges
        if !edges_to_strengthen.is_empty() {
            if let Err(e) = self.batch_strengthen_synapses(&edges_to_strengthen) {
                tracing::debug!("Failed to strengthen synapses: {}", e);
            }
        }

        tracing::debug!(
            "traverse_bidirectional: {} entities, {} edges, path_score={:.4}",
            all_entities.len(),
            all_edges.len(),
            best_path_score
        );

        Ok(GraphTraversal {
            entities: all_entities,
            relationships: all_edges,
        })
    }

    /// Subgraph pattern matching (Cypher-like MATCH patterns)
    ///
    /// Pattern format: Vec of (relation_type, direction) tuples
    /// Direction: true = outgoing (a->b), false = incoming (a<-b)
    ///
    /// Example: MATCH (a)-[:WORKS_AT]->(b)-[:LOCATED_IN]->(c)
    /// Pattern: vec![(WorksAt, true), (LocatedIn, true)]
    ///
    /// Returns all entities that match the pattern starting from start_uuid.
    pub fn match_pattern(
        &self,
        start_uuid: &Uuid,
        pattern: &[(RelationType, bool)], // (relation_type, is_outgoing)
        min_strength: f32,
    ) -> Result<Vec<Vec<TraversedEntity>>> {
        let mut matches: Vec<Vec<TraversedEntity>> = Vec::new();

        // Start entity
        let start_entity = match self.get_entity(start_uuid)? {
            Some(e) => e,
            None => return Ok(matches),
        };

        // DFS backtracking search
        let mut path: Vec<TraversedEntity> = vec![TraversedEntity {
            entity: start_entity,
            hop_distance: 0,
            decay_factor: 1.0,
        }];

        self.match_pattern_recursive(
            *start_uuid,
            pattern,
            0,
            min_strength,
            1.0,
            &mut path,
            &mut matches,
        )?;

        tracing::debug!(
            "match_pattern: found {} matches for {}-step pattern",
            matches.len(),
            pattern.len()
        );

        Ok(matches)
    }

    #[allow(clippy::too_many_arguments)]
    fn match_pattern_recursive(
        &self,
        current_uuid: Uuid,
        pattern: &[(RelationType, bool)],
        pattern_idx: usize,
        min_strength: f32,
        path_score: f32,
        path: &mut Vec<TraversedEntity>,
        matches: &mut Vec<Vec<TraversedEntity>>,
    ) -> Result<()> {
        // Base case: completed the pattern
        if pattern_idx >= pattern.len() {
            matches.push(path.clone());
            return Ok(());
        }

        const MAX_EDGES_PER_NODE: usize = 100;
        let (required_type, is_outgoing) = &pattern[pattern_idx];
        let edges =
            self.get_entity_relationships_limited(&current_uuid, Some(MAX_EDGES_PER_NODE))?;

        for edge in edges {
            if edge.invalidated_at.is_some() {
                continue;
            }

            // Check relationship type
            if edge.relation_type != *required_type {
                continue;
            }

            // Check direction
            let (next_uuid, direction_matches) = if *is_outgoing {
                // Looking for current -> next
                if edge.from_entity == current_uuid {
                    (edge.to_entity, true)
                } else {
                    (edge.from_entity, false) // Wrong direction
                }
            } else {
                // Looking for current <- next (incoming)
                if edge.to_entity == current_uuid {
                    (edge.from_entity, true)
                } else {
                    (edge.to_entity, false) // Wrong direction
                }
            };

            if !direction_matches {
                continue;
            }

            // Check strength
            let effective = edge.effective_strength();
            if effective < min_strength {
                continue;
            }

            // Avoid cycles in pattern
            if path.iter().any(|te| te.entity.uuid == next_uuid) {
                continue;
            }

            // Add to path and recurse
            if let Some(entity) = self.get_entity(&next_uuid)? {
                let new_score = path_score * effective;
                path.push(TraversedEntity {
                    entity,
                    hop_distance: pattern_idx + 1,
                    decay_factor: new_score,
                });

                self.match_pattern_recursive(
                    next_uuid,
                    pattern,
                    pattern_idx + 1,
                    min_strength,
                    new_score,
                    path,
                    matches,
                )?;

                path.pop();
            }
        }

        Ok(())
    }

    /// Find entities matching a pattern from any starting point
    ///
    /// Scans all entities and finds those that match the given pattern.
    /// More expensive than match_pattern but doesn't require a known start.
    ///
    /// Pattern: Vec of (relation_type, is_outgoing) tuples
    /// Returns: All complete pattern matches with their paths.
    pub fn find_pattern_matches(
        &self,
        pattern: &[(RelationType, bool)],
        min_strength: f32,
        limit: usize,
    ) -> Result<Vec<Vec<TraversedEntity>>> {
        let mut all_matches: Vec<Vec<TraversedEntity>> = Vec::new();

        // Iterate through all entities as potential starting points
        let iter = self
            .db
            .iterator_cf(self.entities_cf(), rocksdb::IteratorMode::Start);
        for result in iter {
            if all_matches.len() >= limit {
                break;
            }

            let (_, value) = result?;
            let (entity, _): (EntityNode, _) =
                bincode::serde::decode_from_slice(&value, bincode::config::standard())?;

            let entity_matches = self.match_pattern(&entity.uuid, pattern, min_strength)?;
            for m in entity_matches {
                if all_matches.len() >= limit {
                    break;
                }
                all_matches.push(m);
            }
        }

        tracing::debug!(
            "find_pattern_matches: {} total matches (limit={})",
            all_matches.len(),
            limit
        );

        Ok(all_matches)
    }

    /// Invalidate a relationship (temporal edge invalidation)
    pub fn invalidate_relationship(&self, edge_uuid: &Uuid) -> Result<()> {
        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            edge.invalidated_at = Some(Utc::now());

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.db.put_cf(self.relationships_cf(), key, value)?;
        }

        Ok(())
    }

    /// Strengthen a synapse (Hebbian learning)
    ///
    /// Called when an edge is traversed during memory retrieval.
    /// Implements "neurons that fire together, wire together".
    ///
    /// Uses a mutex to prevent race conditions during concurrent updates (SHO-64).
    pub fn strengthen_synapse(&self, edge_uuid: &Uuid) -> Result<()> {
        // Lock to prevent concurrent read-modify-write race conditions
        let _guard = self.synapse_update_lock.lock();

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            let _ = edge.strengthen();

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.db.put_cf(self.relationships_cf(), key, value)?;
        }

        Ok(())
    }

    /// Batch strengthen multiple synapses atomically (SHO-65)
    ///
    /// More efficient than calling strengthen_synapse individually for each edge.
    /// Uses RocksDB WriteBatch for atomic multi-write and a single lock acquisition.
    ///
    /// Returns the number of synapses successfully strengthened.
    pub fn batch_strengthen_synapses(&self, edge_uuids: &[Uuid]) -> Result<usize> {
        if edge_uuids.is_empty() {
            return Ok(0);
        }

        // Single lock acquisition for entire batch
        let _guard = self.synapse_update_lock.lock();

        let mut batch = WriteBatch::default();
        let mut strengthened = 0;

        for edge_uuid in edge_uuids {
            if let Some(mut edge) = self.get_relationship(edge_uuid)? {
                let _ = edge.strengthen();

                let key = edge.uuid.as_bytes();
                match bincode::serde::encode_to_vec(&edge, bincode::config::standard()) {
                    Ok(value) => {
                        batch.put_cf(self.relationships_cf(), key, value);
                        strengthened += 1;
                    }
                    Err(e) => {
                        tracing::debug!("Failed to serialize edge {}: {}", edge_uuid, e);
                    }
                }
            }
        }

        // Atomic write of all updates
        if strengthened > 0 {
            self.db.write(batch)?;
        }

        Ok(strengthened)
    }

    /// Record co-retrieval of memories (Hebbian learning between memories)
    ///
    /// When memories are retrieved together, they form associations.
    /// This creates or strengthens CoRetrieved edges between all pairs of memories.
    ///
    /// Note: Limits to top N memories to avoid O(n²) explosion on large retrievals.
    /// Returns the number of edges created/strengthened.
    pub fn record_memory_coactivation(&self, memory_ids: &[Uuid]) -> Result<usize> {
        const MAX_COACTIVATION_SIZE: usize = 20;

        // Limit to top N to bound worst-case complexity
        let memories_to_process = if memory_ids.len() > MAX_COACTIVATION_SIZE {
            &memory_ids[..MAX_COACTIVATION_SIZE]
        } else {
            memory_ids
        };

        if memories_to_process.len() < 2 {
            return Ok(0);
        }

        let _guard = self.synapse_update_lock.lock();
        let mut batch = WriteBatch::default();
        let mut edges_updated = 0;
        let mut new_edges = 0;

        // Process all pairs
        for i in 0..memories_to_process.len() {
            for j in (i + 1)..memories_to_process.len() {
                let mem_a = memories_to_process[i];
                let mem_b = memories_to_process[j];

                // Try to find existing edge between these memories
                let existing_edge = self.find_edge_between_entities(&mem_a, &mem_b)?;

                if let Some(mut edge) = existing_edge {
                    // Strengthen existing edge
                    let _ = edge.strengthen();
                    let key = edge.uuid.as_bytes();
                    if let Ok(value) =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                    {
                        batch.put_cf(self.relationships_cf(), key, value);
                        edges_updated += 1;
                    }
                } else {
                    // Create new CoRetrieved edge (bidirectional represented as single edge)
                    // Starts in L1 (working memory) with tier-specific initial weight
                    let edge = RelationshipEdge {
                        uuid: Uuid::new_v4(),
                        from_entity: mem_a,
                        to_entity: mem_b,
                        relation_type: RelationType::CoRetrieved,
                        strength: EdgeTier::L1Working.initial_weight(),
                        created_at: Utc::now(),
                        valid_at: Utc::now(),
                        invalidated_at: None,
                        source_episode_id: None,
                        context: String::new(),
                        last_activated: Utc::now(),
                        activation_count: 1,
                        ltp_status: LtpStatus::None,
                        activation_timestamps: None,
                        tier: EdgeTier::L1Working,
                        // PIPE-5: Memory-to-memory edges use default confidence
                        entity_confidence: None,
                    };

                    let key = edge.uuid.as_bytes();
                    if let Ok(value) =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                    {
                        batch.put_cf(self.relationships_cf(), key, value);

                        // Also index in the reverse direction for lookup
                        let idx_key_fwd = format!("mem_edge:{mem_a}:{mem_b}");
                        let idx_key_rev = format!("mem_edge:{mem_b}:{mem_a}");
                        batch.put_cf(
                            self.relationships_cf(),
                            idx_key_fwd.as_bytes(),
                            edge.uuid.as_bytes(),
                        );
                        batch.put_cf(
                            self.relationships_cf(),
                            idx_key_rev.as_bytes(),
                            edge.uuid.as_bytes(),
                        );

                        edges_updated += 1;
                        new_edges += 1;
                    }
                }
            }
        }

        if edges_updated > 0 {
            self.db.write(batch)?;
            // Update relationship counter for newly created edges
            if new_edges > 0 {
                self.relationship_count
                    .fetch_add(new_edges, Ordering::Relaxed);
            }
        }

        Ok(edges_updated)
    }

    /// Find an edge between two entities/memories (in either direction)
    fn find_edge_between_entities(
        &self,
        entity_a: &Uuid,
        entity_b: &Uuid,
    ) -> Result<Option<RelationshipEdge>> {
        // Check forward index
        let idx_key = format!("mem_edge:{entity_a}:{entity_b}");
        if let Some(edge_uuid_bytes) = self
            .db
            .get_cf(self.relationships_cf(), idx_key.as_bytes())?
        {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                return self.get_relationship(&edge_uuid);
            }
        }

        // Check reverse index
        let idx_key_rev = format!("mem_edge:{entity_b}:{entity_a}");
        if let Some(edge_uuid_bytes) = self
            .db
            .get_cf(self.relationships_cf(), idx_key_rev.as_bytes())?
        {
            if edge_uuid_bytes.len() == 16 {
                let edge_uuid = Uuid::from_slice(&edge_uuid_bytes)?;
                return self.get_relationship(&edge_uuid);
            }
        }

        Ok(None)
    }

    /// Batch strengthen edges between memory pairs from replay consolidation
    ///
    /// Takes edge boosts from memory replay and applies Hebbian strengthening.
    /// Creates edges if they don't exist, strengthens if they do.
    ///
    /// Returns (count_strengthened, promotion_boosts) where promotion_boosts contains
    /// signals for any edge tier promotions that occurred (Direction 1 coupling).
    pub fn strengthen_memory_edges(
        &self,
        edge_boosts: &[(String, String, f32)],
    ) -> Result<(usize, Vec<crate::memory::types::EdgePromotionBoost>)> {
        use crate::constants::{EDGE_PROMOTION_MEMORY_BOOST_L2, EDGE_PROMOTION_MEMORY_BOOST_L3};

        if edge_boosts.is_empty() {
            return Ok((0, Vec::new()));
        }

        let _guard = self.synapse_update_lock.lock();
        let mut batch = WriteBatch::default();
        let mut strengthened = 0;
        let mut promotion_boosts = Vec::new();

        for (from_id_str, to_id_str, _boost) in edge_boosts {
            // Parse UUIDs
            let from_uuid = match Uuid::parse_str(from_id_str) {
                Ok(u) => u,
                Err(_) => {
                    tracing::debug!("Invalid from_id UUID: {}", from_id_str);
                    continue;
                }
            };
            let to_uuid = match Uuid::parse_str(to_id_str) {
                Ok(u) => u,
                Err(_) => {
                    tracing::debug!("Invalid to_id UUID: {}", to_id_str);
                    continue;
                }
            };

            // Find or create edge
            let existing_edge = self.find_edge_between_entities(&from_uuid, &to_uuid)?;

            if let Some(mut edge) = existing_edge {
                // Strengthen existing edge — capture tier promotion if it occurs
                let promotion = edge.strengthen();
                let key = edge.uuid.as_bytes();
                if let Ok(value) = bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                {
                    batch.put_cf(self.relationships_cf(), key, value);
                    strengthened += 1;

                    // If a tier promotion occurred, emit boost signals for both memories
                    if let Some((old_tier, new_tier)) = promotion {
                        let boost = if new_tier.contains("L2") {
                            EDGE_PROMOTION_MEMORY_BOOST_L2
                        } else {
                            EDGE_PROMOTION_MEMORY_BOOST_L3
                        };
                        let entity_name = format!(
                            "{}↔{}",
                            &from_id_str[..8.min(from_id_str.len())],
                            &to_id_str[..8.min(to_id_str.len())]
                        );
                        // Boost both memories involved in the promoted edge
                        promotion_boosts.push(crate::memory::types::EdgePromotionBoost {
                            memory_id: from_id_str.clone(),
                            entity_name: entity_name.clone(),
                            old_tier: old_tier.clone(),
                            new_tier: new_tier.clone(),
                            boost,
                        });
                        promotion_boosts.push(crate::memory::types::EdgePromotionBoost {
                            memory_id: to_id_str.clone(),
                            entity_name,
                            old_tier,
                            new_tier,
                            boost,
                        });
                    }
                }
            } else {
                // Create new ReplayStrengthened edge
                // Replay edges start in L2 (episodic) since they represent consolidated associations
                let edge = RelationshipEdge {
                    uuid: Uuid::new_v4(),
                    from_entity: from_uuid,
                    to_entity: to_uuid,
                    relation_type: RelationType::CoRetrieved,
                    strength: EdgeTier::L2Episodic.initial_weight(),
                    created_at: Utc::now(),
                    valid_at: Utc::now(),
                    invalidated_at: None,
                    source_episode_id: None,
                    context: "replay_strengthened".to_string(),
                    last_activated: Utc::now(),
                    activation_count: 1,
                    ltp_status: LtpStatus::None,
                    activation_timestamps: None,
                    tier: EdgeTier::L2Episodic,
                    // PIPE-5: Replay edges use default confidence
                    entity_confidence: None,
                };

                let key = edge.uuid.as_bytes();
                if let Ok(value) = bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                {
                    batch.put_cf(self.relationships_cf(), key, value);

                    // Index both directions
                    let idx_key_fwd = format!("mem_edge:{from_uuid}:{to_uuid}");
                    let idx_key_rev = format!("mem_edge:{to_uuid}:{from_uuid}");
                    batch.put_cf(
                        self.relationships_cf(),
                        idx_key_fwd.as_bytes(),
                        edge.uuid.as_bytes(),
                    );
                    batch.put_cf(
                        self.relationships_cf(),
                        idx_key_rev.as_bytes(),
                        edge.uuid.as_bytes(),
                    );

                    strengthened += 1;
                }
            }
        }

        if strengthened > 0 {
            self.db.write(batch)?;

            // Index new replay edges in entity_edges CF so they're visible to
            // traversal and degree-cap enforcement (GQ-11 fix)
            let mut entities_to_prune = Vec::new();
            for (from_id_str, to_id_str, _boost) in edge_boosts {
                let from_uuid = match Uuid::parse_str(from_id_str) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                let to_uuid = match Uuid::parse_str(to_id_str) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                // Only index edges that we actually wrote (find_edge_between_entities returns
                // the edge if it existed before, so new edges are the ones that didn't exist)
                if let Ok(Some(edge)) = self.find_edge_between_entities(&from_uuid, &to_uuid) {
                    if edge.context == "replay_strengthened" && edge.activation_count <= 1 {
                        if let Err(e) = self.index_entity_edge(&from_uuid, &edge.uuid) {
                            tracing::debug!("Failed to index replay edge for entity: {}", e);
                        }
                        if let Err(e) = self.index_entity_edge(&to_uuid, &edge.uuid) {
                            tracing::debug!("Failed to index replay edge for entity: {}", e);
                        }
                        entities_to_prune.push(from_uuid);
                        entities_to_prune.push(to_uuid);
                    }
                }
            }

            // Enforce degree cap on affected entities
            for entity_uuid in &entities_to_prune {
                let _ = self.prune_entity_if_over_degree(entity_uuid);
            }

            tracing::debug!(
                "Applied {} edge boosts from replay consolidation ({} tier promotions)",
                strengthened,
                promotion_boosts.len()
            );
        }

        Ok((strengthened, promotion_boosts))
    }

    /// Find memories associated with a given memory through co-retrieval
    ///
    /// Uses weighted graph traversal prioritizing stronger associations.
    /// Returns memory UUIDs sorted by association strength.
    pub fn find_memory_associations(
        &self,
        memory_id: &Uuid,
        max_results: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        let mut associations: Vec<(Uuid, f32)> = Vec::new();

        // Scan for edges involving this memory
        let prefix_fwd = format!("mem_edge:{memory_id}:");

        let iter = self
            .db
            .prefix_iterator_cf(self.relationships_cf(), prefix_fwd.as_bytes());
        for item in iter {
            let (key, value) = item?;

            // Check if this is our prefix (RocksDB prefix_iterator may return extra)
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix_fwd) {
                break;
            }

            // Get edge UUID from value and look up edge
            if value.len() == 16 {
                let edge_uuid = Uuid::from_slice(&value)?;
                if let Some(edge) = self.get_relationship(&edge_uuid)? {
                    // Get the other memory in this edge
                    let other_id = if edge.from_entity == *memory_id {
                        edge.to_entity
                    } else {
                        edge.from_entity
                    };

                    // Get effective strength with decay
                    let effective_strength = edge.effective_strength();
                    if effective_strength > LTP_MIN_STRENGTH {
                        associations.push((other_id, effective_strength));
                    }
                }
            }
        }

        // Sort by strength descending and limit
        associations.sort_by(|a, b| b.1.total_cmp(&a.1));
        associations.truncate(max_results);

        Ok(associations)
    }

    /// Strengthen entity-entity edges for a replayed memory's episode.
    ///
    /// During consolidation replay, this reinforces the entity relationships that
    /// were involved in the replayed memory. This is "Direction 3" of the Hebbian
    /// maintenance system — entity-entity edges get strengthened alongside
    /// memory-to-memory edges (Direction 1) and lazy pruning (Direction 2).
    ///
    /// Algorithm:
    /// 1. Look up EpisodicNode for episode_id → get entity_refs
    /// 2. For each pair of entities, find their RelationshipEdge
    /// 3. Call strengthen() on each edge (Hebbian boost + LTP detection + tier promotion)
    /// 4. Batch write all updates
    pub fn strengthen_episode_entity_edges(&self, episode_id: &Uuid) -> Result<usize> {
        let episode = match self.get_episode(episode_id) {
            Ok(Some(ep)) => ep,
            Ok(None) => return Ok(0),
            Err(_) => return Ok(0),
        };

        if episode.entity_refs.len() < 2 {
            return Ok(0);
        }

        let _guard = self.synapse_update_lock.lock();
        let mut batch = WriteBatch::default();
        let mut strengthened = 0;

        // Iterate over unique entity pairs
        let refs = &episode.entity_refs;
        let max_pairs = refs.len().min(20); // Cap to avoid O(n²) on large episodes
        for i in 0..max_pairs {
            for j in (i + 1)..max_pairs {
                let entity_a = &refs[i];
                let entity_b = &refs[j];

                // Find existing edge between this entity pair
                if let Ok(Some(mut edge)) = self.find_edge_between_entities(entity_a, entity_b) {
                    if edge.invalidated_at.is_some() {
                        continue;
                    }
                    let _ = edge.strengthen();
                    let key = edge.uuid.as_bytes();
                    if let Ok(value) =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())
                    {
                        batch.put_cf(self.relationships_cf(), key, value);
                        strengthened += 1;
                    }
                }
                // Don't create new edges — only strengthen existing ones from NER
            }
        }

        if strengthened > 0 {
            self.db.write(batch)?;
            tracing::debug!(
                "Strengthened {} entity-entity edges for episode {}",
                strengthened,
                &episode_id.to_string()[..8]
            );
        }

        Ok(strengthened)
    }

    /// Get average Hebbian strength for a memory based on its entity relationships
    ///
    /// This looks up the entities referenced by the memory and averages their
    /// relationship strengths in the graph. Used for composite relevance scoring.
    ///
    /// The algorithm:
    /// 1. Look up memory's EpisodicNode (memory_id.0 == episode UUID)
    /// 2. Get entity_refs from the episode
    /// 3. For each entity, get relationships using get_entity_relationships
    /// 4. Filter to edges where both endpoints are in the memory's entity set
    /// 5. Return average effective_strength of these intra-memory edges
    ///
    /// Returns 0.5 (neutral) if no entities or relationships found.
    pub fn get_memory_hebbian_strength(&self, memory_id: &crate::memory::MemoryId) -> Option<f32> {
        // 1. Look up EpisodicNode for this memory (memory_id.0 == episode UUID)
        let episode = match self.get_episode(&memory_id.0) {
            Ok(Some(ep)) => ep,
            Ok(None) => return Some(0.5), // No episode found - neutral
            Err(_) => return Some(0.5),   // Error - neutral fallback
        };

        // 2. Get entity references from the episode
        if episode.entity_refs.is_empty() {
            return Some(0.5); // No entities - neutral
        }

        // Build a set of entity UUIDs for fast lookup
        let entity_set: std::collections::HashSet<Uuid> =
            episode.entity_refs.iter().cloned().collect();

        // 3. Collect all intra-memory relationship strengths
        let mut strengths: Vec<f32> = Vec::new();
        let mut seen_edges: std::collections::HashSet<Uuid> = std::collections::HashSet::new();

        const MAX_EDGES_PER_ENTITY: usize = 50; // Limit per entity for Hebbian lookup
        for entity_uuid in &episode.entity_refs {
            if let Ok(edges) =
                self.get_entity_relationships_limited(entity_uuid, Some(MAX_EDGES_PER_ENTITY))
            {
                for edge in edges {
                    // Skip if already processed (edges are bidirectional in lookup)
                    if seen_edges.contains(&edge.uuid) {
                        continue;
                    }
                    seen_edges.insert(edge.uuid);

                    // 4. Only count edges where BOTH endpoints are in this memory's entities
                    if entity_set.contains(&edge.from_entity)
                        && entity_set.contains(&edge.to_entity)
                    {
                        // Skip invalidated edges
                        if edge.invalidated_at.is_some() {
                            continue;
                        }
                        // Use effective_strength which applies time-based decay
                        strengths.push(edge.effective_strength());
                    }
                }
            }
        }

        // 5. Return average strength, or neutral if no intra-memory edges
        if strengths.is_empty() {
            Some(0.5)
        } else {
            let avg = strengths.iter().sum::<f32>() / strengths.len() as f32;
            Some(avg)
        }
    }

    /// Apply decay to a synapse
    ///
    /// Returns true if the synapse should be pruned (non-potentiated and below threshold)
    ///
    /// Uses a mutex to prevent race conditions during concurrent updates (SHO-64).
    pub fn decay_synapse(&self, edge_uuid: &Uuid) -> Result<bool> {
        // Lock to prevent concurrent read-modify-write race conditions
        let _guard = self.synapse_update_lock.lock();

        if let Some(mut edge) = self.get_relationship(edge_uuid)? {
            let should_prune = edge.decay();

            let key = edge.uuid.as_bytes();
            let value = bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
            self.db.put_cf(self.relationships_cf(), key, value)?;

            return Ok(should_prune);
        }

        Ok(false)
    }

    /// Batch decay multiple synapses atomically
    ///
    /// Returns a vector of edge UUIDs that should be pruned.
    pub fn batch_decay_synapses(&self, edge_uuids: &[Uuid]) -> Result<Vec<Uuid>> {
        if edge_uuids.is_empty() {
            return Ok(Vec::new());
        }

        // Single lock acquisition for entire batch
        let _guard = self.synapse_update_lock.lock();

        let mut batch = WriteBatch::default();
        let mut to_prune = Vec::new();

        for edge_uuid in edge_uuids {
            if let Some(mut edge) = self.get_relationship(edge_uuid)? {
                let should_prune = edge.decay();

                let key = edge.uuid.as_bytes();
                match bincode::serde::encode_to_vec(&edge, bincode::config::standard()) {
                    Ok(value) => {
                        batch.put_cf(self.relationships_cf(), key, value);
                        if should_prune {
                            to_prune.push(*edge_uuid);
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to serialize edge {}: {}", edge_uuid, e);
                    }
                }
            }
        }

        // Atomic write of all updates
        self.db.write(batch)?;

        Ok(to_prune)
    }

    /// Apply decay to already-loaded edges in-place, avoiding double deserialization.
    ///
    /// Mutates edges directly, serializes results into a WriteBatch, and returns
    /// the UUIDs of edges that should be pruned. Used by `apply_decay()` which
    /// already has the full edge list from `get_all_relationships()`.
    fn batch_decay_edges_in_place(&self, edges: &mut [RelationshipEdge]) -> Result<Vec<Uuid>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }

        let _guard = self.synapse_update_lock.lock();
        let mut batch = WriteBatch::default();
        let mut to_prune = Vec::new();

        for edge in edges.iter_mut() {
            let strength_before = edge.strength;
            let should_prune = edge.decay();

            // Only write back edges whose strength actually changed (or need pruning).
            // With 300s maintenance intervals, most edges won't have meaningful decay,
            // so this reduces the WriteBatch from ~12MB (all 34k edges) to ~150KB.
            if should_prune || (edge.strength - strength_before).abs() > f32::EPSILON {
                let key = edge.uuid.as_bytes();
                match bincode::serde::encode_to_vec(&*edge, bincode::config::standard()) {
                    Ok(value) => {
                        batch.put_cf(self.relationships_cf(), key, value);
                        if should_prune {
                            to_prune.push(edge.uuid);
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Failed to serialize edge {}: {}", edge.uuid, e);
                    }
                }
            }
        }

        self.db.write(batch)?;
        Ok(to_prune)
    }

    /// Apply decay to all synapses and prune weak edges (AUD-2)
    ///
    /// Called during maintenance cycle to:
    /// 1. Apply time-based decay to all edge strengths
    /// 2. Remove edges that have decayed below threshold
    /// 3. Detect orphaned entities (entities that lost all their edges)
    ///
    /// Returns a `GraphDecayResult` with pruned count and orphaned entity/memory IDs
    /// for Direction 2 coupling (edge pruning → orphan detection).
    pub fn apply_decay(&self) -> Result<crate::memory::types::GraphDecayResult> {
        // Get all edges (need full data for orphan tracking)
        let mut all_edges = self.get_all_relationships()?;

        if all_edges.is_empty() {
            return Ok(crate::memory::types::GraphDecayResult::default());
        }

        // Apply decay in-place on already-deserialized edges (avoids double deserialization)
        let to_prune = self.batch_decay_edges_in_place(&mut all_edges)?;

        if to_prune.is_empty() {
            return Ok(crate::memory::types::GraphDecayResult::default());
        }

        // Collect entity UUIDs from edges being pruned (candidates for orphan status)
        let pruned_set: std::collections::HashSet<Uuid> = to_prune.iter().copied().collect();
        let mut orphan_candidates: std::collections::HashSet<Uuid> =
            std::collections::HashSet::new();
        for edge in &all_edges {
            if pruned_set.contains(&edge.uuid) {
                orphan_candidates.insert(edge.from_entity);
                orphan_candidates.insert(edge.to_entity);
            }
        }

        // Delete pruned edges
        let mut pruned_count = 0;
        for edge_uuid in &to_prune {
            if self.delete_relationship(edge_uuid)? {
                pruned_count += 1;
            }
        }

        // Check which candidate entities became orphaned (lost ALL edges)
        // Delete orphaned entities to prevent stale index pollution
        let mut orphaned_entity_ids = Vec::new();
        for entity_uuid in &orphan_candidates {
            let remaining = self.get_entity_relationships(entity_uuid)?;
            if remaining.is_empty() {
                orphaned_entity_ids.push(entity_uuid.to_string());
                if let Err(e) = self.delete_entity(entity_uuid) {
                    tracing::warn!("Failed to delete orphaned entity {}: {}", entity_uuid, e);
                }
            }
        }

        if pruned_count > 0 {
            tracing::debug!(
                "Graph decay: {} edges pruned (of {} total), {} entities orphaned",
                pruned_count,
                all_edges.len(),
                orphaned_entity_ids.len()
            );
        }

        Ok(crate::memory::types::GraphDecayResult {
            pruned_count,
            orphaned_entity_ids,
            orphaned_memory_ids: Vec::new(), // Populated by memory layer via entity→memory lookup
        })
    }

    /// Flush pending maintenance from opportunistic pruning queues.
    ///
    /// Called every maintenance cycle (5 min). Instead of scanning all 34k+ edges,
    /// this only processes edges that were found below prune threshold during normal
    /// reads (via `get_entity_relationships_limited`). Typical cost: 0-50 targeted
    /// deletes per cycle vs a full CF iterator scan.
    pub fn flush_pending_maintenance(&self) -> Result<crate::memory::types::GraphDecayResult> {
        // 1. Drain queues (fast — just swaps empty Vecs)
        let to_prune: Vec<Uuid> = std::mem::take(&mut *self.pending_prune.lock());
        let orphan_candidates: Vec<Uuid> = std::mem::take(&mut *self.pending_orphan_checks.lock());

        if to_prune.is_empty() {
            return Ok(crate::memory::types::GraphDecayResult::default());
        }

        // 2. Dedup UUIDs
        let to_prune: std::collections::HashSet<Uuid> = to_prune.into_iter().collect();
        let orphan_candidates: std::collections::HashSet<Uuid> =
            orphan_candidates.into_iter().collect();

        // 3. Batch delete pruned edges
        let mut pruned_count = 0;
        for edge_uuid in &to_prune {
            if self.delete_relationship(edge_uuid)? {
                pruned_count += 1;
            }
        }

        // 4. Check which candidate entities became orphaned (lost ALL edges)
        let mut orphaned_entity_ids = Vec::new();
        for entity_uuid in &orphan_candidates {
            let remaining = self.get_entity_relationships(entity_uuid)?;
            if remaining.is_empty() {
                orphaned_entity_ids.push(entity_uuid.to_string());
                if let Err(e) = self.delete_entity(entity_uuid) {
                    tracing::warn!("Failed to delete orphaned entity {}: {}", entity_uuid, e);
                }
            }
        }

        if pruned_count > 0 {
            tracing::debug!(
                "Lazy pruning: {} edges deleted, {} entities orphaned",
                pruned_count,
                orphaned_entity_ids.len()
            );
        }

        Ok(crate::memory::types::GraphDecayResult {
            pruned_count,
            orphaned_entity_ids,
            orphaned_memory_ids: Vec::new(),
        })
    }

    /// Get graph statistics - O(1) using atomic counters
    pub fn get_stats(&self) -> Result<GraphStats> {
        Ok(GraphStats {
            entity_count: self.entity_count.load(Ordering::Relaxed),
            relationship_count: self.relationship_count.load(Ordering::Relaxed),
            episode_count: self.episode_count.load(Ordering::Relaxed),
        })
    }

    /// Get all entities in the graph
    pub fn get_all_entities(&self) -> Result<Vec<EntityNode>> {
        let mut entities = Vec::new();

        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter =
            self.db
                .iterator_cf_opt(self.entities_cf(), read_opts, rocksdb::IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok(entity) = bincode::serde::decode_from_slice::<EntityNode, _>(
                &value,
                bincode::config::standard(),
            )
            .map(|(v, _)| v)
            {
                entities.push(entity);
            }
        }

        // Sort by mention count (most mentioned first)
        entities.sort_by(|a, b| b.mention_count.cmp(&a.mention_count));

        Ok(entities)
    }

    /// Get all relationships in the graph
    pub fn get_all_relationships(&self) -> Result<Vec<RelationshipEdge>> {
        let mut relationships = Vec::new();

        // fill_cache(false) prevents this full scan from evicting hot data from
        // the block cache. Decompressed blocks are used transiently and freed
        // after the iterator advances, reducing peak C++ heap usage.
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_cf_opt(
            self.relationships_cf(),
            read_opts,
            rocksdb::IteratorMode::Start,
        );
        for (_, value) in iter.flatten() {
            if let Ok(edge) = bincode::serde::decode_from_slice::<RelationshipEdge, _>(
                &value,
                bincode::config::standard(),
            )
            .map(|(v, _)| v)
            {
                // Only include non-invalidated relationships
                if edge.invalidated_at.is_none() {
                    relationships.push(edge);
                }
            }
        }

        // Sort by strength (strongest first)
        relationships.sort_by(|a, b| b.strength.total_cmp(&a.strength));

        Ok(relationships)
    }

    /// Get the Memory Universe visualization data
    /// Returns entities as "stars" with positions based on their relationships,
    /// sized by salience, and colored by entity type.
    pub fn get_universe(&self) -> Result<MemoryUniverse> {
        let entities = self.get_all_entities()?;
        let relationships = self.get_all_relationships()?;

        // Create entity UUID to index mapping for position calculation
        let entity_indices: HashMap<Uuid, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.uuid, i))
            .collect();

        // Calculate 3D positions using a force-directed layout approximation
        // High-salience entities are positioned more centrally
        let mut stars: Vec<UniverseStar> = entities
            .iter()
            .enumerate()
            .map(|(i, entity)| {
                // Use a spiral galaxy layout with salience affecting radius
                // Higher salience = closer to center
                let angle = (i as f32) * 2.4; // Golden angle for even distribution
                let base_radius = 1.0 - entity.salience; // High salience = small radius
                let radius = base_radius * 100.0 + 10.0; // 10-110 range

                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let z = ((i as f32) * 0.1).sin() * 20.0; // Slight z variation

                UniverseStar {
                    id: entity.uuid.to_string(),
                    name: entity.name.clone(),
                    entity_type: entity.labels.first().map(|l| l.as_str().to_string()),
                    salience: entity.salience,
                    mention_count: entity.mention_count,
                    is_proper_noun: entity.is_proper_noun,
                    position: Position3D { x, y, z },
                    color: entity_type_color(entity.labels.first()),
                    size: 5.0 + entity.salience * 20.0, // Size 5-25 based on salience
                }
            })
            .collect();

        // Apply gravitational forces FIRST, before creating connections
        // This ensures connection positions match final star positions
        for rel in &relationships {
            if let (Some(from_idx), Some(to_idx)) = (
                entity_indices.get(&rel.from_entity),
                entity_indices.get(&rel.to_entity),
            ) {
                // Apply small gravitational pull based on effective (decay-aware) strength
                let pull_factor = rel.effective_strength() * 0.05;

                let from_pos = stars[*from_idx].position.clone();
                let to_pos = stars[*to_idx].position.clone();

                let dx = (to_pos.x - from_pos.x) * pull_factor;
                let dy = (to_pos.y - from_pos.y) * pull_factor;
                let dz = (to_pos.z - from_pos.z) * pull_factor;

                stars[*from_idx].position.x += dx;
                stars[*from_idx].position.y += dy;
                stars[*from_idx].position.z += dz;

                stars[*to_idx].position.x -= dx;
                stars[*to_idx].position.y -= dy;
                stars[*to_idx].position.z -= dz;
            }
        }

        // Create gravitational connections AFTER star positions are finalized
        // This ensures from_position/to_position match current star positions
        let connections: Vec<GravitationalConnection> = relationships
            .iter()
            .filter_map(|rel| {
                let from_idx = entity_indices.get(&rel.from_entity)?;
                let to_idx = entity_indices.get(&rel.to_entity)?;

                Some(GravitationalConnection {
                    id: rel.uuid.to_string(),
                    from_id: rel.from_entity.to_string(),
                    to_id: rel.to_entity.to_string(),
                    relation_type: rel.relation_type.as_str().to_string(),
                    strength: rel.effective_strength(),
                    from_position: stars[*from_idx].position.clone(),
                    to_position: stars[*to_idx].position.clone(),
                })
            })
            .collect();

        // Calculate universe bounds
        let (min_x, max_x, min_y, max_y, min_z, max_z) = stars.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(min_x, max_x, min_y, max_y, min_z, max_z), star| {
                (
                    min_x.min(star.position.x),
                    max_x.max(star.position.x),
                    min_y.min(star.position.y),
                    max_y.max(star.position.y),
                    min_z.min(star.position.z),
                    max_z.max(star.position.z),
                )
            },
        );

        Ok(MemoryUniverse {
            stars,
            connections,
            total_entities: entities.len(),
            total_connections: relationships.len(),
            bounds: UniverseBounds {
                min: Position3D {
                    x: min_x,
                    y: min_y,
                    z: min_z,
                },
                max: Position3D {
                    x: max_x,
                    y: max_y,
                    z: max_z,
                },
            },
        })
    }
}

/// Helper function to get color for entity type
fn entity_type_color(label: Option<&EntityLabel>) -> String {
    match label {
        Some(EntityLabel::Person) => "#FF6B6B".to_string(), // Coral red
        Some(EntityLabel::Organization) => "#4ECDC4".to_string(), // Teal
        Some(EntityLabel::Location) => "#45B7D1".to_string(), // Sky blue
        Some(EntityLabel::Technology) => "#96CEB4".to_string(), // Sage green
        Some(EntityLabel::Product) => "#FFEAA7".to_string(), // Soft yellow
        Some(EntityLabel::Event) => "#DDA0DD".to_string(),  // Plum
        Some(EntityLabel::Skill) => "#98D8C8".to_string(),  // Mint
        Some(EntityLabel::Concept) => "#F7DC6F".to_string(), // Gold
        Some(EntityLabel::Date) => "#BB8FCE".to_string(),   // Light purple
        Some(EntityLabel::Keyword) => "#FF9F43".to_string(), // Orange for YAKE keywords
        Some(EntityLabel::Other(_)) => "#AEB6BF".to_string(), // Gray
        None => "#AEB6BF".to_string(),                      // Gray default
    }
}

/// 3D position in the memory universe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A star in the memory universe (represents an entity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseStar {
    pub id: String,
    pub name: String,
    pub entity_type: Option<String>,
    pub salience: f32,
    pub mention_count: usize,
    pub is_proper_noun: bool,
    pub position: Position3D,
    pub color: String,
    pub size: f32,
}

/// A gravitational connection between stars (represents a relationship)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravitationalConnection {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub relation_type: String,
    pub strength: f32,
    pub from_position: Position3D,
    pub to_position: Position3D,
}

/// Bounds of the memory universe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseBounds {
    pub min: Position3D,
    pub max: Position3D,
}

/// The complete memory universe visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUniverse {
    pub stars: Vec<UniverseStar>,
    pub connections: Vec<GravitationalConnection>,
    pub total_entities: usize,
    pub total_connections: usize,
    pub bounds: UniverseBounds,
}

/// Entity with hop distance from traversal origin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversedEntity {
    pub entity: EntityNode,
    /// Number of hops from the starting entity (0 = start entity)
    pub hop_distance: usize,
    /// Decay factor based on hop distance: 1.0 at hop 0, decays with each hop
    pub decay_factor: f32,
}

/// Result of graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTraversal {
    /// Entities found during traversal with hop distance info
    pub entities: Vec<TraversedEntity>,
    pub relationships: Vec<RelationshipEdge>,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub episode_count: usize,
}

/// Extracted entity with salience information
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub name: String,
    pub label: EntityLabel,
    pub base_salience: f32,
}

/// Simple entity extraction (rule-based NER) with salience detection
pub struct EntityExtractor {
    /// Common person name indicators
    person_indicators: HashSet<String>,

    /// Common organization indicators (suffixes like Inc, Corp)
    org_indicators: HashSet<String>,

    /// Known organization names (direct matches)
    org_keywords: HashSet<String>,

    /// Known location names (cities, countries, regions)
    location_keywords: HashSet<String>,

    /// Common technology keywords
    tech_keywords: HashSet<String>,

    /// Common words that should NOT be extracted as entities
    /// (stop words that start with capitals at sentence beginning)
    stop_words: HashSet<String>,
}

impl EntityExtractor {
    pub fn new() -> Self {
        let person_indicators: HashSet<String> =
            vec!["mr", "mrs", "ms", "dr", "prof", "sir", "madam"]
                .into_iter()
                .map(String::from)
                .collect();

        let org_indicators: HashSet<String> = vec![
            "inc",
            "corp",
            "ltd",
            "llc",
            "company",
            "corporation",
            "university",
            "institute",
            "foundation",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let tech_keywords: HashSet<String> = vec![
            "rust",
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "vue",
            "angular",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "gcp",
            "sql",
            "nosql",
            "mongodb",
            "postgresql",
            "redis",
            "kafka",
            "api",
            "rest",
            "graphql",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Known organization names (global - India-first, then worldwide)
        let org_keywords: HashSet<String> = vec![
            // Indian Companies - IT/Tech
            "tcs",
            "infosys",
            "wipro",
            "hcl",
            "tech mahindra",
            "cognizant",
            "mindtree",
            "mphasis",
            "ltimindtree",
            "persistent",
            "zensar",
            "cyient",
            "hexaware",
            "coforge",
            "birlasoft",
            "sonata software",
            "mastek",
            "newgen",
            // Indian Companies - Startups/Unicorns
            "flipkart",
            "paytm",
            "zomato",
            "swiggy",
            "ola",
            "oyo",
            "byju's",
            "byjus",
            "razorpay",
            "phonepe",
            "cred",
            "zerodha",
            "groww",
            "upstox",
            "policybazaar",
            "nykaa",
            "meesho",
            "udaan",
            "delhivery",
            "freshworks",
            "zoho",
            "postman",
            "browserstack",
            "chargebee",
            "clevertap",
            "druva",
            "hasura",
            "innovaccer",
            "lenskart",
            "mamaearth",
            "unacademy",
            "vedantu",
            "physicswallah",
            "dream11",
            "mpl",
            "winzo",
            "slice",
            "jupiter",
            "fi",
            "niyo",
            "smallcase",
            "koo",
            "sharechat",
            "dailyhunt",
            "pratilipi",
            "inshorts",
            "rapido",
            "urban company",
            "dunzo",
            "bigbasket",
            "grofers",
            "blinkit",
            "jiomart",
            "tata neu",
            // Indian Conglomerates
            "tata",
            "reliance",
            "jio",
            "adani",
            "birla",
            "mahindra",
            "godrej",
            "bajaj",
            "hdfc",
            "icici",
            "kotak",
            "axis",
            "sbi",
            "bharti",
            "airtel",
            "vodafone",
            "idea",
            "hero",
            "tvs",
            "maruti",
            "suzuki",
            "hyundai",
            "kia",
            "mg",
            "tata motors",
            "larsen",
            "toubro",
            "l&t",
            "itc",
            "hindustan unilever",
            "hul",
            "nestle",
            "britannia",
            "parle",
            "amul",
            "dabur",
            "patanjali",
            "emami",
            "marico",
            // Indian Banks & Finance
            "rbi",
            "sebi",
            "nse",
            "bse",
            "npci",
            "upi",
            "bhim",
            "paisa",
            "mswipe",
            "pine labs",
            "billdesk",
            "ccavenue",
            "instamojo",
            "cashfree",
            // Indian Institutions
            "iit",
            "iim",
            "iisc",
            "nit",
            "bits",
            "isro",
            "drdo",
            "barc",
            "tifr",
            "aiims",
            "iiser",
            "iiit",
            "srm",
            "vit",
            "manipal",
            "amity",
            "lovely",
            // Global Tech Giants
            "microsoft",
            "google",
            "apple",
            "amazon",
            "meta",
            "facebook",
            "netflix",
            "alphabet",
            "youtube",
            "instagram",
            "whatsapp",
            "tiktok",
            "snapchat",
            "twitter",
            "x",
            "linkedin",
            "pinterest",
            "reddit",
            "discord",
            "telegram",
            // Global Enterprise Tech
            "salesforce",
            "oracle",
            "ibm",
            "sap",
            "vmware",
            "dell",
            "hp",
            "hpe",
            "cisco",
            "juniper",
            "palo alto",
            "crowdstrike",
            "fortinet",
            "splunk",
            "servicenow",
            "workday",
            "atlassian",
            "jira",
            "confluence",
            "trello",
            "asana",
            "monday",
            "notion",
            "airtable",
            "figma",
            "canva",
            "miro",
            // Global Cloud & Infrastructure
            "aws",
            "azure",
            "gcp",
            "digitalocean",
            "linode",
            "vultr",
            "cloudflare",
            "akamai",
            "fastly",
            "vercel",
            "netlify",
            "heroku",
            "render",
            "railway",
            // Global Hardware/Chip
            "intel",
            "amd",
            "nvidia",
            "qualcomm",
            "broadcom",
            "arm",
            "tsmc",
            "samsung",
            "mediatek",
            "apple silicon",
            "marvell",
            "micron",
            "sk hynix",
            "western digital",
            // Global AI/ML Companies
            "openai",
            "anthropic",
            "deepmind",
            "cohere",
            "stability",
            "midjourney",
            "hugging face",
            "databricks",
            "snowflake",
            "palantir",
            "c3ai",
            "datarobot",
            // Global Fintech
            "stripe",
            "square",
            "block",
            "paypal",
            "venmo",
            "wise",
            "revolut",
            "robinhood",
            "coinbase",
            "binance",
            "kraken",
            "gemini",
            "ftx",
            "blockchain",
            "ripple",
            // Global Dev Tools
            "github",
            "gitlab",
            "bitbucket",
            "jetbrains",
            "vscode",
            "sublime",
            "vim",
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "puppet",
            "chef",
            // Global Consulting
            "accenture",
            "deloitte",
            "pwc",
            "kpmg",
            "ey",
            "mckinsey",
            "bcg",
            "bain",
            // Global Auto/EV
            "tesla",
            "rivian",
            "lucid",
            "nio",
            "byd",
            "xpeng",
            "volkswagen",
            "bmw",
            "mercedes",
            "audi",
            "porsche",
            "toyota",
            "honda",
            "nissan",
            "ford",
            "gm",
            // Global Aerospace
            "spacex",
            "boeing",
            "airbus",
            "lockheed",
            "northrop",
            "raytheon",
            "nasa",
            "esa",
            "jaxa",
            "isro",
            "blue origin",
            "virgin galactic",
            // Universities - India
            "delhi university",
            "jnu",
            "bhu",
            "amu",
            "jadavpur",
            "presidency",
            "st stephens",
            "loyola",
            "xavier",
            "symbiosis",
            "nmims",
            "sp jain",
            "xlri",
            "fms",
            "iift",
            "mdi",
            "great lakes",
            "ism dhanbad",
            // Universities - Global
            "mit",
            "stanford",
            "harvard",
            "yale",
            "princeton",
            "caltech",
            "berkeley",
            "oxford",
            "cambridge",
            "imperial",
            "eth zurich",
            "epfl",
            "tsinghua",
            "peking",
            "nus",
            "nanyang",
            "kaist",
            "university of tokyo",
            "melbourne",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Known location names (global - India-first, then worldwide)
        let location_keywords: HashSet<String> = vec![
            // Indian Metro Cities
            "mumbai",
            "delhi",
            "bangalore",
            "bengaluru",
            "hyderabad",
            "chennai",
            "kolkata",
            "pune",
            "ahmedabad",
            "surat",
            "jaipur",
            "lucknow",
            // Indian Tier-1 Cities
            "kochi",
            "cochin",
            "thiruvananthapuram",
            "trivandrum",
            "coimbatore",
            "madurai",
            "visakhapatnam",
            "vizag",
            "vijayawada",
            "nagpur",
            "indore",
            "bhopal",
            "chandigarh",
            "mohali",
            "panchkula",
            "noida",
            "gurgaon",
            "gurugram",
            "faridabad",
            "ghaziabad",
            "greater noida",
            "dwarka",
            // Indian Tier-2 Cities
            "mysore",
            "mangalore",
            "hubli",
            "belgaum",
            "nashik",
            "aurangabad",
            "rajkot",
            "vadodara",
            "baroda",
            "gandhinagar",
            "kanpur",
            "varanasi",
            "allahabad",
            "prayagraj",
            "agra",
            "meerut",
            "dehradun",
            "rishikesh",
            "haridwar",
            "amritsar",
            "jalandhar",
            "ludhiana",
            "shimla",
            "manali",
            "dharamshala",
            "jammu",
            "srinagar",
            "ranchi",
            "jamshedpur",
            "patna",
            "guwahati",
            "shillong",
            "imphal",
            "kohima",
            "gangtok",
            "darjeeling",
            "bhubaneswar",
            "cuttack",
            "rourkela",
            "raipur",
            "bilaspur",
            // Indian States & UTs
            "maharashtra",
            "karnataka",
            "tamil nadu",
            "telangana",
            "andhra pradesh",
            "kerala",
            "gujarat",
            "rajasthan",
            "uttar pradesh",
            "madhya pradesh",
            "west bengal",
            "bihar",
            "odisha",
            "jharkhand",
            "chhattisgarh",
            "punjab",
            "haryana",
            "himachal pradesh",
            "uttarakhand",
            "goa",
            "assam",
            "meghalaya",
            "manipur",
            "nagaland",
            "tripura",
            "mizoram",
            "arunachal pradesh",
            "sikkim",
            "jammu and kashmir",
            "ladakh",
            // Indian Regions
            "silicon valley of india",
            "electronic city",
            "whitefield",
            "marathahalli",
            "koramangala",
            "indiranagar",
            "hsr layout",
            "jayanagar",
            "malleshwaram",
            "bandra",
            "andheri",
            "powai",
            "lower parel",
            "bkc",
            "navi mumbai",
            "thane",
            "connaught place",
            "nehru place",
            "saket",
            "cyber city",
            "dlf",
            "hitech city",
            "madhapur",
            "gachibowli",
            "ecr",
            "omr",
            "it corridor",
            // Asian Cities
            "singapore",
            "hong kong",
            "tokyo",
            "osaka",
            "seoul",
            "busan",
            "beijing",
            "shanghai",
            "shenzhen",
            "guangzhou",
            "hangzhou",
            "taipei",
            "bangkok",
            "kuala lumpur",
            "jakarta",
            "manila",
            "ho chi minh",
            "hanoi",
            "dubai",
            "abu dhabi",
            "doha",
            "riyadh",
            "tel aviv",
            "istanbul",
            // European Cities
            "london",
            "paris",
            "berlin",
            "munich",
            "frankfurt",
            "amsterdam",
            "rotterdam",
            "brussels",
            "zurich",
            "geneva",
            "vienna",
            "prague",
            "warsaw",
            "budapest",
            "barcelona",
            "madrid",
            "milan",
            "rome",
            "lisbon",
            "dublin",
            "edinburgh",
            "manchester",
            "stockholm",
            "oslo",
            "helsinki",
            "copenhagen",
            "athens",
            "moscow",
            "st petersburg",
            // North American Cities
            "new york",
            "los angeles",
            "san francisco",
            "seattle",
            "boston",
            "chicago",
            "austin",
            "denver",
            "portland",
            "miami",
            "atlanta",
            "dallas",
            "houston",
            "phoenix",
            "san diego",
            "san jose",
            "oakland",
            "palo alto",
            "mountain view",
            "cupertino",
            "menlo park",
            "redwood city",
            "washington dc",
            "philadelphia",
            "detroit",
            "toronto",
            "vancouver",
            "montreal",
            "calgary",
            "ottawa",
            "mexico city",
            "guadalajara",
            // South American Cities
            "sao paulo",
            "rio de janeiro",
            "buenos aires",
            "santiago",
            "bogota",
            "lima",
            "medellin",
            "cartagena",
            // African Cities
            "johannesburg",
            "cape town",
            "lagos",
            "nairobi",
            "cairo",
            "casablanca",
            "accra",
            "addis ababa",
            "kigali",
            // Australian/NZ Cities
            "sydney",
            "melbourne",
            "brisbane",
            "perth",
            "auckland",
            "wellington",
            // Countries - Asia
            "india",
            "china",
            "japan",
            "south korea",
            "korea",
            "taiwan",
            "singapore",
            "malaysia",
            "thailand",
            "vietnam",
            "indonesia",
            "philippines",
            "bangladesh",
            "pakistan",
            "sri lanka",
            "nepal",
            "bhutan",
            "myanmar",
            "cambodia",
            "laos",
            // Countries - Middle East
            "uae",
            "emirates",
            "saudi arabia",
            "qatar",
            "bahrain",
            "kuwait",
            "oman",
            "israel",
            "turkey",
            "iran",
            "iraq",
            "jordan",
            "lebanon",
            "egypt",
            // Countries - Europe
            "uk",
            "united kingdom",
            "britain",
            "england",
            "scotland",
            "wales",
            "ireland",
            "france",
            "germany",
            "italy",
            "spain",
            "portugal",
            "netherlands",
            "belgium",
            "switzerland",
            "austria",
            "poland",
            "czech",
            "hungary",
            "romania",
            "bulgaria",
            "greece",
            "sweden",
            "norway",
            "finland",
            "denmark",
            "russia",
            "ukraine",
            // Countries - Americas
            "usa",
            "united states",
            "america",
            "canada",
            "mexico",
            "brazil",
            "argentina",
            "chile",
            "colombia",
            "peru",
            "venezuela",
            // Countries - Africa/Oceania
            "south africa",
            "nigeria",
            "kenya",
            "ghana",
            "ethiopia",
            "rwanda",
            "australia",
            "new zealand",
            // Famous Tech Hubs
            "silicon valley",
            "bay area",
            "wall street",
            "tech city",
            "shoreditch",
            "station f",
            "blockchain island",
            "crypto valley",
            "startup nation",
            "innovation district",
            "tech park",
            "it park",
            "sez",
            "special economic zone",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        // Stop words: common words that appear capitalized at sentence start
        // These aren't named entities even when capitalized
        let stop_words: HashSet<String> = vec![
            // Articles & pronouns
            "the", "a", "an", "this", "that", "these", "those", "i", "we", "you", "he", "she", "it",
            "they", // Common verbs (appear at sentence start)
            "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", // Question words
            "if", "when", "where", "what", "why", "how",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            person_indicators,
            org_indicators,
            org_keywords,
            location_keywords,
            tech_keywords,
            stop_words,
        }
    }

    /// Calculate base salience for an entity based on its type and detection confidence
    ///
    /// Salience values by entity type:
    /// - Person: 0.8 (highest - people are key context)
    /// - Organization/Product: 0.7
    /// - Location/Technology/Event: 0.6
    /// - Skill: 0.5
    /// - Concept: 0.4
    /// - Date/Other: 0.3
    ///
    /// Proper nouns receive a 20% boost (capped at 1.0).
    pub fn calculate_base_salience(label: &EntityLabel, is_proper_noun: bool) -> f32 {
        let type_salience = match label {
            EntityLabel::Person => 0.8,       // People are highly salient
            EntityLabel::Organization => 0.7, // Organizations are important
            EntityLabel::Location => 0.6,     // Locations matter for context
            EntityLabel::Technology => 0.6,   // Tech keywords matter for dev context
            EntityLabel::Product => 0.7,      // Products are specific entities
            EntityLabel::Event => 0.6,        // Events are temporal anchors
            EntityLabel::Skill => 0.5,        // Skills are somewhat important
            EntityLabel::Keyword => 0.55,     // YAKE keywords - discriminative terms
            EntityLabel::Concept => 0.4,      // Concepts are more generic
            EntityLabel::Date => 0.3,         // Dates are structural, not salient
            EntityLabel::Other(_) => 0.3,     // Unknown types get low salience
        };

        // Proper nouns get a 20% boost
        if is_proper_noun {
            (type_salience * 1.2_f32).min(1.0_f32)
        } else {
            type_salience
        }
    }

    /// Check if a word is likely a proper noun (not just capitalized at sentence start)
    fn is_likely_proper_noun(&self, word: &str, position: usize, prev_char: Option<char>) -> bool {
        // If it's not at position 0 and is capitalized, it's likely a proper noun
        if position > 0 {
            return true;
        }

        // At position 0, check if previous character was punctuation (sentence start)
        // If previous char was '.', '!', '?' then this might just be sentence capitalization
        if let Some(c) = prev_char {
            if c == '.' || c == '!' || c == '?' {
                // It's at sentence start - could be either
                // Check if it's a common word
                let lower = word.to_lowercase();
                return !self.stop_words.contains(&lower);
            }
        }

        // Default to proper noun for capitalized words
        true
    }

    /// Extract entities from text with salience information
    pub fn extract_with_salience(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let mut seen = HashSet::new();
        let mut skip_until_index = 0; // For skipping sub-spans of multi-word entities

        // Split into words and detect capitalized sequences
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            // Skip if this word is part of a multi-word entity we already extracted
            if i < skip_until_index {
                continue;
            }

            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());

            if clean_word.is_empty() {
                continue;
            }

            let lower = clean_word.to_lowercase();

            // Skip common stop words
            if self.stop_words.contains(&lower) {
                continue;
            }

            // Check for known organization keywords (direct match, min 2 chars to filter "x" noise)
            if lower.len() >= 2 && self.org_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Organization,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Organization, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for known location keywords (direct match)
            if self.location_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Location,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Location, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for technology keywords (always proper nouns in tech context)
            if self.tech_keywords.contains(&lower) && !seen.contains(&lower) {
                let entity = ExtractedEntity {
                    name: clean_word.to_string(),
                    label: EntityLabel::Technology,
                    base_salience: Self::calculate_base_salience(&EntityLabel::Technology, true),
                };
                entities.push(entity);
                seen.insert(lower.clone());
                continue;
            }

            // Check for capitalized words (potential entities)
            if clean_word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                let mut entity_name = clean_word.to_string();
                let mut entity_label = EntityLabel::Other("Unknown".to_string());

                // Determine previous character for proper noun detection
                let prev_char = if i > 0 {
                    words[i - 1].chars().last()
                } else {
                    None
                };

                let is_proper = self.is_likely_proper_noun(clean_word, i, prev_char);

                // Check for person indicators
                if i > 0
                    && self
                        .person_indicators
                        .contains(&words[i - 1].to_lowercase())
                {
                    entity_label = EntityLabel::Person;
                }

                // Check for multi-word capitalized sequences
                let mut j = i + 1;
                while j < words.len()
                    && words[j]
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    let next_word = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                    // Skip stop words in multi-word sequences
                    if !self.stop_words.contains(&next_word.to_lowercase()) {
                        entity_name.push(' ');
                        entity_name.push_str(next_word);
                    }
                    j += 1;
                }

                // Set skip_until_index to avoid extracting sub-spans
                // e.g., if we extracted "John Smith", skip "Smith" on next iteration
                if j > i + 1 {
                    skip_until_index = j;
                }

                let entity_name_lower = entity_name.to_lowercase();

                // Check multi-word entity against known lists
                if self.org_keywords.contains(&entity_name_lower) {
                    entity_label = EntityLabel::Organization;
                } else if self.location_keywords.contains(&entity_name_lower) {
                    entity_label = EntityLabel::Location;
                }

                // Check for organization indicators (suffixes)
                if matches!(entity_label, EntityLabel::Other(_)) {
                    for word in entity_name.split_whitespace() {
                        if self.org_indicators.contains(&word.to_lowercase()) {
                            entity_label = EntityLabel::Organization;
                            break;
                        }
                    }
                }

                // Only extract entities we have evidence for
                // Don't guess on single unknown capitalized words - they're often noise
                if matches!(entity_label, EntityLabel::Other(_)) {
                    if entity_name.contains(' ') {
                        // Multi-word capitalized sequences (like "John Smith", "New York")
                        // are likely proper names — use Concept as safe default
                        // Concept(0.4) + proper noun boost(1.2x) = 0.48 salience
                        // Hebbian strengthening will promote genuinely important entities
                        entity_label = EntityLabel::Concept;
                    } else {
                        // Single capitalized word not in any keyword list
                        // Skip it - we don't have enough evidence it's a real entity
                        // The neural NER model handles these cases properly
                        continue;
                    }
                }

                let entity_key = entity_name_lower;
                if !seen.contains(&entity_key) {
                    let base_salience = Self::calculate_base_salience(&entity_label, is_proper);
                    let entity = ExtractedEntity {
                        name: entity_name,
                        label: entity_label,
                        base_salience,
                    };
                    entities.push(entity);
                    seen.insert(entity_key);
                }
            }
        }

        // HYBRID APPROACH: POS-based extraction + YAKE importance scoring
        //
        // 1. POS extraction ensures ALL content words are captured (no frequency bias)
        // 2. YAKE provides discriminativeness scores for boosting rare/important terms
        //
        // This solves the "sunrise problem": YAKE alone buries rare words at position 41,
        // but POS ensures "sunrise" is extracted, and YAKE boosts its salience.
        use crate::embeddings::keywords::{KeywordConfig, KeywordExtractor};
        use crate::memory::query_parser::{extract_chunks, PosTag};

        // Get YAKE importance scores for discriminative weighting
        let kw_config = KeywordConfig {
            max_keywords: 100, // Get many keywords for lookup
            ngrams: 1,
            min_length: 3,
            ..Default::default()
        };
        let kw_extractor = KeywordExtractor::with_config(kw_config);
        let keywords = kw_extractor.extract(text);

        // Build a lookup map: term -> importance (0.0-1.0)
        let yake_importance: std::collections::HashMap<String, f32> = keywords
            .into_iter()
            .map(|kw| (kw.text.to_lowercase(), kw.importance))
            .collect();

        // POS-based extraction for comprehensive coverage
        let chunk_extraction = extract_chunks(text);

        // Add all proper nouns (these are likely named entities we might have missed)
        for proper_noun in &chunk_extraction.proper_nouns {
            let term_lower = proper_noun.to_lowercase();
            if !seen.contains(&term_lower) && term_lower.len() >= 3 {
                // Boost salience if YAKE identified this as discriminative
                let yake_boost = yake_importance.get(&term_lower).copied().unwrap_or(0.0);
                let entity = ExtractedEntity {
                    name: proper_noun.clone(),
                    label: EntityLabel::Person,
                    base_salience: 0.7 + (yake_boost * 0.2), // 0.7-0.9
                };
                entities.push(entity);
                seen.insert(term_lower);
            }
        }

        // Add all content words as Keyword entities
        // POS ensures comprehensive extraction, YAKE boosts discriminative terms
        for chunk in &chunk_extraction.chunks {
            for word in &chunk.words {
                let term_lower = word.text.to_lowercase();

                // Skip if already extracted or too short
                if seen.contains(&term_lower) || term_lower.len() < 4 {
                    continue;
                }

                // Skip stop words
                if self.stop_words.contains(&term_lower) {
                    continue;
                }

                // Base salience by POS, boosted by YAKE importance
                let yake_boost = yake_importance.get(&term_lower).copied().unwrap_or(0.0);

                let (label, base_salience) = match word.pos {
                    PosTag::Noun | PosTag::ProperNoun => {
                        // Nouns are most important, start at 0.5
                        (EntityLabel::Keyword, 0.5)
                    }
                    PosTag::Verb => {
                        // Verbs connect entities, start at 0.4
                        (EntityLabel::Keyword, 0.4)
                    }
                    PosTag::Adjective => {
                        // Adjectives are modifiers, start at 0.35
                        (EntityLabel::Keyword, 0.35)
                    }
                    _ => continue,
                };

                // Boost by YAKE importance (0.0-0.3 boost based on discriminativeness)
                let final_salience = base_salience + (yake_boost * 0.3);

                let entity = ExtractedEntity {
                    name: word.text.clone(),
                    label,
                    base_salience: final_salience,
                };
                entities.push(entity);
                seen.insert(term_lower);
            }
        }

        entities
    }

    /// Extract co-occurrence pairs from text for graph edge creation
    ///
    /// Returns pairs of (entity1, entity2) that appear in the same sentence.
    /// This enables creating edges between words that co-occur, which is critical
    /// for multi-hop retrieval (e.g., connecting "Melanie" to "sunrise" when
    /// they appear in the same sentence about painting).
    pub fn extract_cooccurrence_pairs(&self, text: &str) -> Vec<(String, String)> {
        use crate::memory::query_parser::extract_chunks;

        let chunk_extraction = extract_chunks(text);
        let mut pairs = Vec::new();

        // Get all co-occurrence pairs from chunks (same sentence)
        for chunk in &chunk_extraction.chunks {
            let content_words = chunk.content_words();

            // Create pairs between all content words in the same sentence
            for i in 0..content_words.len() {
                for j in (i + 1)..content_words.len() {
                    let w1 = content_words[i].text.to_lowercase();
                    let w2 = content_words[j].text.to_lowercase();

                    // Skip very short words and stop words
                    if w1.len() >= 3
                        && w2.len() >= 3
                        && !self.stop_words.contains(&w1)
                        && !self.stop_words.contains(&w2)
                    {
                        pairs.push((w1, w2));
                    }
                }
            }
        }

        pairs
    }
}

impl Default for EntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    /// Create a test relationship edge with specified strength and last_activated (L1 tier)
    fn create_test_edge(strength: f32, days_since_activated: i64) -> RelationshipEdge {
        create_test_edge_with_tier(strength, days_since_activated, EdgeTier::L1Working)
    }

    /// Create a test relationship edge with specified strength, last_activated, and tier
    fn create_test_edge_with_tier(
        strength: f32,
        days_since_activated: i64,
        tier: EdgeTier,
    ) -> RelationshipEdge {
        RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: Uuid::new_v4(),
            to_entity: Uuid::new_v4(),
            relation_type: RelationType::RelatedTo,
            strength,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: None,
            context: String::new(),
            last_activated: Utc::now() - Duration::days(days_since_activated),
            activation_count: 0,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier,
            entity_confidence: None, // PIPE-5: Default for tests
        }
    }

    #[test]
    fn test_hebbian_strengthen_increases_strength() {
        use crate::constants::*;
        // Use L2 tier to avoid L1 promotion resetting strength
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);
        let initial_strength = edge.strength;

        let _ = edge.strengthen();

        // With tier boost (L2 gets 80% of TIER_CO_ACCESS_BOOST), strength should increase
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.8;
        let expected_boost = (LTP_LEARNING_RATE + tier_boost) * (1.0 - initial_strength);
        assert!(
            edge.strength > initial_strength,
            "Strengthen should increase strength (expected boost {expected_boost})"
        );
        assert_eq!(edge.activation_count, 1);
    }

    #[test]
    fn test_hebbian_strengthen_asymptotic() {
        use crate::constants::*;
        // Use L3 tier (no promotion) with high initial strength
        let mut edge = create_test_edge_with_tier(0.95, 0, EdgeTier::L3Semantic);

        let _ = edge.strengthen();

        // High strength should still increase but slowly (asymptotic to 1.0)
        // L3 tier boost = TIER_CO_ACCESS_BOOST * 0.5 = 0.075
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.5;
        let expected_min = 0.95 + (LTP_LEARNING_RATE + tier_boost) * 0.05 - 0.01;
        assert!(
            edge.strength > expected_min,
            "Expected > {expected_min}, got {}",
            edge.strength
        );
        assert!(edge.strength <= 1.0);
    }

    #[test]
    fn test_hebbian_strengthen_formula() {
        use crate::constants::*;
        // Test: w_new = w_old + (η + tier_boost) × (1 - w_old)
        // Use L2 tier (tier_boost = TIER_CO_ACCESS_BOOST * 0.8) at 0.3 to avoid promotion
        let mut edge = create_test_edge_with_tier(0.3, 0, EdgeTier::L2Episodic);

        let _ = edge.strengthen();

        // L2 tier boost = 0.15 * 0.8 = 0.12
        // Expected: 0.3 + (0.1 + 0.12) * (1 - 0.3) = 0.3 + 0.22 * 0.7 = 0.454
        let tier_boost = TIER_CO_ACCESS_BOOST * 0.8;
        let expected = 0.3 + (LTP_LEARNING_RATE + tier_boost) * 0.7;
        assert!(
            (edge.strength - expected).abs() < 0.001,
            "Expected {expected}, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_ltp_threshold_potentiation() {
        let mut edge = create_test_edge(0.5, 0);
        assert!(!edge.is_potentiated());

        // Strengthen 10 times (LTP_THRESHOLD = 10)
        for _ in 0..10 {
            let _ = edge.strengthen();
        }

        assert!(
            edge.is_potentiated(),
            "Should be potentiated after 10 activations"
        );
        assert!(
            matches!(edge.ltp_status, LtpStatus::Full),
            "Should have Full LTP status after 10 activations"
        );
        assert!(
            edge.strength > 0.7,
            "Potentiated edge should have bonus strength"
        );
    }

    #[test]
    fn test_pipe4_burst_ltp_detection() {
        // Create an L2 edge with low strength to avoid early tier promotion
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Strengthen 5 times (LTP_BURST_THRESHOLD = 5) within 24 hours
        for _ in 0..5 {
            let _ = edge.strengthen();
        }

        // Should have burst LTP (5+ activations in 24h)
        // Edge may promote to L3 during strengthening, but should keep Burst status
        assert!(
            matches!(edge.ltp_status, LtpStatus::Burst { .. }),
            "Should have Burst LTP after 5 rapid activations, got {:?}",
            edge.ltp_status
        );
    }

    #[test]
    fn test_pipe4_activation_timestamps_recorded() {
        // L2 edges should record activation timestamps
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Strengthen a few times
        for _ in 0..3 {
            let _ = edge.strengthen();
        }

        // Should have recorded timestamps (edge may have promoted to L3, but still tracks)
        assert!(
            edge.activation_timestamps.is_some(),
            "L2+ edge should have activation timestamps"
        );
        assert_eq!(
            edge.activation_timestamps.as_ref().unwrap().len(),
            3,
            "Should have 3 recorded timestamps"
        );
    }

    #[test]
    fn test_pipe4_fresh_l1_no_timestamps() {
        // Fresh L1 edges should NOT have activation timestamps
        let edge = create_test_edge(0.3, 0);
        assert!(matches!(edge.tier, EdgeTier::L1Working));
        assert!(
            edge.activation_timestamps.is_none(),
            "Fresh L1 edges should not have timestamps"
        );
    }

    #[test]
    fn test_pipe4_l1_promotes_and_tracks() {
        // L1 edges that promote to L2 should start tracking timestamps
        let mut edge = create_test_edge(0.3, 0);
        assert!(matches!(edge.tier, EdgeTier::L1Working));

        // Strengthen until it promotes to L2 (L1_PROMOTION_THRESHOLD = 0.5)
        while matches!(edge.tier, EdgeTier::L1Working) {
            let _ = edge.strengthen();
        }

        // After promotion to L2, should start tracking
        assert!(
            matches!(edge.tier, EdgeTier::L2Episodic),
            "Should have promoted to L2"
        );
        // Timestamps are initialized on promotion
        assert!(
            edge.activation_timestamps.is_some(),
            "L2 edges should track timestamps after promotion"
        );
    }

    #[test]
    fn test_pipe4_ltp_status_decay_factors() {
        // Test that each LTP status has correct decay factor
        use crate::constants::*;

        assert_eq!(LtpStatus::None.decay_factor(), 1.0);
        assert_eq!(LtpStatus::Weekly.decay_factor(), LTP_WEEKLY_DECAY_FACTOR);
        assert_eq!(LtpStatus::Full.decay_factor(), LTP_DECAY_FACTOR);

        // Burst factor depends on expiration
        let burst = LtpStatus::Burst {
            detected_at: Utc::now(),
        };
        assert_eq!(burst.decay_factor(), LTP_BURST_DECAY_FACTOR);
    }

    #[test]
    fn test_pipe4_burst_to_full_upgrade() {
        // LTP should upgrade from Burst to Full after 10 activations
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Get to burst LTP (5 activations)
        for _ in 0..5 {
            let _ = edge.strengthen();
        }
        assert!(
            matches!(edge.ltp_status, LtpStatus::Burst { .. }),
            "Should have Burst after 5 activations, got {:?}",
            edge.ltp_status
        );

        // Continue strengthening to Full LTP (10 total)
        for _ in 0..5 {
            let _ = edge.strengthen();
        }

        // Should now be Full (upgraded from Burst via 10 activations)
        assert!(
            matches!(edge.ltp_status, LtpStatus::Full),
            "Should have upgraded to Full LTP after 10 activations"
        );
    }

    #[test]
    fn test_pipe4_activations_in_window() {
        let mut edge = create_test_edge_with_tier(0.22, 0, EdgeTier::L2Episodic);

        // Record some activations
        for _ in 0..5 {
            let _ = edge.strengthen();
        }

        let now = Utc::now();
        let hour_ago = now - chrono::Duration::hours(1);
        let day_ago = now - chrono::Duration::days(1);

        // All activations are recent (within last second really)
        let in_hour = edge.activations_in_window(hour_ago, now);
        let in_day = edge.activations_in_window(day_ago, now);
        assert!(in_hour >= 5, "Expected 5+ in hour window, got {in_hour}");
        assert!(in_day >= 5, "Expected 5+ in day window, got {in_day}");
    }

    // =========================================================================
    // PIPE-5: Unified LTP Readiness Model Tests
    // =========================================================================

    #[test]
    fn test_pipe5_adjusted_threshold_default() {
        // Default confidence (None → 0.5) should give default threshold (10)
        let edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        assert!(edge.entity_confidence.is_none());

        let threshold = edge.adjusted_threshold();
        // confidence 0.5 → threshold = 13 - (0.5 * 6) = 10
        assert_eq!(threshold, 10, "Default confidence should give threshold 10");
    }

    #[test]
    fn test_pipe5_adjusted_threshold_high_confidence() {
        // High confidence (0.9) should give lower threshold (7-8)
        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        edge.entity_confidence = Some(0.9);

        let threshold = edge.adjusted_threshold();
        // confidence 0.9 → threshold = 13 - (0.9 * 6) = 7.6 → 8
        assert!(
            threshold <= 8,
            "High confidence should give threshold <= 8, got {threshold}"
        );
    }

    #[test]
    fn test_pipe5_adjusted_threshold_low_confidence() {
        // Low confidence (0.2) should give higher threshold (12-13)
        let mut edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        edge.entity_confidence = Some(0.2);

        let threshold = edge.adjusted_threshold();
        // confidence 0.2 → threshold = 13 - (0.2 * 6) = 11.8 → 12
        assert!(
            threshold >= 11,
            "Low confidence should give threshold >= 11, got {threshold}"
        );
    }

    #[test]
    fn test_pipe5_strength_floor_by_tier() {
        use crate::constants::*;

        let l1_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L1Working);
        let l2_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L2Episodic);
        let l3_edge = create_test_edge_with_tier(0.5, 0, EdgeTier::L3Semantic);

        assert_eq!(
            l1_edge.strength_floor(),
            1.0,
            "L1 should have floor 1.0 (impossible)"
        );
        assert_eq!(
            l2_edge.strength_floor(),
            LTP_STRENGTH_FLOOR_L2,
            "L2 floor mismatch"
        );
        assert_eq!(
            l3_edge.strength_floor(),
            LTP_STRENGTH_FLOOR_L3,
            "L3 floor mismatch"
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_l1_returns_zero() {
        // L1 edges should always return 0 readiness (can't reach Full LTP)
        let mut edge = create_test_edge_with_tier(0.99, 0, EdgeTier::L1Working);
        edge.activation_count = 100;
        edge.entity_confidence = Some(1.0);

        assert_eq!(
            edge.ltp_readiness(),
            0.0,
            "L1 edges should always return 0 readiness"
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_balanced_path() {
        use crate::constants::*;

        // Balanced: 10 activations + 0.75 strength + 0.5 confidence
        // count_score = 10 / 10 = 1.0
        // strength_score = 0.75 / 0.65 = 1.15
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 1.0 * 0.5 + 1.15 * 0.5 + 0.15 = 0.5 + 0.575 + 0.15 = 1.225
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.activation_count = 10;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Balanced path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_repetition_dominant() {
        use crate::constants::*;

        // Repetition dominant: 15 activations + 0.50 strength (below floor)
        // count_score = 15 / 10 = 1.5
        // strength_score = 0.50 / 0.65 = 0.77
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 1.5 * 0.5 + 0.77 * 0.5 + 0.15 = 0.75 + 0.385 + 0.15 = 1.285
        let mut edge = create_test_edge_with_tier(0.50, 0, EdgeTier::L2Episodic);
        edge.activation_count = 15;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Repetition-dominant path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_intensity_dominant() {
        use crate::constants::*;

        // Intensity dominant: 5 activations + 0.95 strength (L3)
        // count_score = 5 / 10 = 0.5
        // strength_score = 0.95 / 0.80 = 1.1875
        // tag_bonus = 0.5 * 0.3 = 0.15
        // readiness = 0.5 * 0.5 + 1.1875 * 0.5 + 0.15 = 0.25 + 0.59 + 0.15 = 0.99
        // Need more strength or count for intensity-only path on L3
        let mut edge = create_test_edge_with_tier(0.99, 0, EdgeTier::L3Semantic);
        edge.activation_count = 6;
        edge.entity_confidence = Some(0.5);

        let readiness = edge.ltp_readiness();
        // count_score = 6/10 = 0.6, strength_score = 0.99/0.80 = 1.24
        // readiness = 0.6*0.5 + 1.24*0.5 + 0.15 = 0.3 + 0.62 + 0.15 = 1.07
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "Intensity-dominant path should reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_ltp_readiness_high_confidence_boost() {
        use crate::constants::*;

        // High confidence edge reaches LTP faster
        // 7 activations + 0.65 strength + 0.9 confidence
        // threshold = 13 - 0.9*6 = 7.6 → 8
        // count_score = 7 / 8 = 0.875
        // strength_score = 0.65 / 0.65 = 1.0
        // tag_bonus = 0.9 * 0.3 = 0.27
        // readiness = 0.875 * 0.5 + 1.0 * 0.5 + 0.27 = 0.44 + 0.5 + 0.27 = 1.21
        let mut edge = create_test_edge_with_tier(0.65, 0, EdgeTier::L2Episodic);
        edge.activation_count = 7;
        edge.entity_confidence = Some(0.9);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness >= LTP_READINESS_THRESHOLD,
            "High-confidence should boost to LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_weak_edge_no_ltp() {
        use crate::constants::*;

        // Weak edge: 4 activations + 0.40 strength + 0.3 confidence
        // threshold = 13 - 0.3*6 = 11.2 → 11
        // count_score = 4 / 11 = 0.36
        // strength_score = 0.40 / 0.65 = 0.62
        // tag_bonus = 0.3 * 0.3 = 0.09
        // readiness = 0.36 * 0.5 + 0.62 * 0.5 + 0.09 = 0.18 + 0.31 + 0.09 = 0.58
        let mut edge = create_test_edge_with_tier(0.40, 0, EdgeTier::L2Episodic);
        edge.activation_count = 4;
        edge.entity_confidence = Some(0.3);

        let readiness = edge.ltp_readiness();
        assert!(
            readiness < LTP_READINESS_THRESHOLD,
            "Weak edge should NOT reach LTP, readiness = {}",
            readiness
        );
    }

    #[test]
    fn test_pipe5_unified_detect_ltp_status() {
        // Test that detect_ltp_status uses the unified readiness formula
        let mut edge = create_test_edge_with_tier(0.75, 0, EdgeTier::L2Episodic);
        edge.activation_count = 10;
        edge.entity_confidence = Some(0.5);
        edge.activation_timestamps = Some(std::collections::VecDeque::new());

        let status = edge.detect_ltp_status(Utc::now());
        assert_eq!(
            status,
            LtpStatus::Full,
            "Balanced path should grant Full LTP via readiness"
        );
    }

    #[test]
    fn test_pipe5_l3_no_auto_ltp_without_activations() {
        // L3 with high strength but low activation count should NOT auto-LTP
        // This tests that the old auto-LTP behavior is removed
        let mut edge = create_test_edge_with_tier(0.85, 0, EdgeTier::L3Semantic);
        edge.activation_count = 2; // Low count
        edge.entity_confidence = Some(0.5);
        edge.activation_timestamps = Some(std::collections::VecDeque::new());

        // count_score = 2/10 = 0.2, strength_score = 0.85/0.80 = 1.06
        // readiness = 0.2*0.5 + 1.06*0.5 + 0.15 = 0.1 + 0.53 + 0.15 = 0.78
        let status = edge.detect_ltp_status(Utc::now());
        assert_eq!(
            status,
            LtpStatus::None,
            "L3 high strength alone should NOT grant Full LTP, needs activations too"
        );
    }

    #[test]
    fn test_decay_reduces_strength() {
        // Use L2 tier for multi-day decay testing (L1 max age is only 4 hours)
        let mut edge = create_test_edge_with_tier(0.5, 7, EdgeTier::L2Episodic);

        let initial_strength = edge.strength;
        edge.decay();

        assert!(
            edge.strength < initial_strength,
            "Decay should reduce strength (initial: {}, after: {})",
            initial_strength,
            edge.strength
        );
    }

    #[test]
    fn test_decay_tier_aware() {
        // Test tier-aware decay: L2 episodic with exponential decay (λ=0.031/day) over 7 days
        // Expected: e^(-0.031 * 7) ≈ 0.805, so strength decays to ~80%
        let mut edge = create_test_edge_with_tier(1.0, 7, EdgeTier::L2Episodic);

        edge.decay();

        // After 7 days with L2 exponential decay, expect moderate reduction (~80% retained)
        // but well above floor since within max age
        assert!(
            edge.strength < 0.85,
            "After 7 days with L2 decay, strength should be below 0.85, got {}",
            edge.strength
        );
        assert!(
            edge.strength > 0.75,
            "After 7 days with L2 decay, strength should be above 0.75, got {}",
            edge.strength
        );
        assert!(
            edge.strength > LTP_MIN_STRENGTH,
            "Strength should still be above floor, got {}",
            edge.strength
        );
    }

    #[test]
    fn test_decay_minimum_floor() {
        // Use L3 tier for very old edge testing (L3 has 10 year max age)
        let mut edge = create_test_edge_with_tier(0.02, 100, EdgeTier::L3Semantic);

        edge.decay();

        assert!(
            edge.strength >= LTP_MIN_STRENGTH,
            "Strength should not go below minimum floor"
        );
    }

    #[test]
    fn test_potentiated_decay_slower() {
        // Use L2 tier for multi-day decay comparison
        let mut edge1 = create_test_edge_with_tier(0.8, 7, EdgeTier::L2Episodic);
        let mut edge2 = create_test_edge_with_tier(0.8, 7, EdgeTier::L2Episodic);
        edge2.ltp_status = LtpStatus::Full; // Full LTP = 10x slower decay

        edge1.decay();
        edge2.decay();

        assert!(
            edge2.strength > edge1.strength,
            "Potentiated edge should decay slower (normal: {}, potentiated: {})",
            edge1.strength,
            edge2.strength
        );
    }

    #[test]
    fn test_effective_strength_read_only() {
        // Use L2 tier for multi-day testing
        let edge = create_test_edge_with_tier(0.5, 7, EdgeTier::L2Episodic);
        let initial_strength = edge.strength;

        let effective = edge.effective_strength();

        // effective_strength should not modify the edge
        assert_eq!(edge.strength, initial_strength);
        assert!(effective < initial_strength);
    }

    #[test]
    fn test_decay_prune_threshold() {
        // Use L2 tier for decay testing beyond its max age (14 days)
        let mut weak_edge = create_test_edge_with_tier(LTP_MIN_STRENGTH, 30, EdgeTier::L2Episodic);
        // No LTP status = normal decay
        assert!(matches!(weak_edge.ltp_status, LtpStatus::None));

        let should_prune = weak_edge.decay();

        // Non-potentiated edge at minimum strength past max age should be prunable
        assert!(
            should_prune,
            "Weak non-potentiated edge past max age should be marked for pruning"
        );
    }

    #[test]
    fn test_potentiated_above_floor_never_pruned() {
        // Potentiated edge above LTP_PRUNE_FLOOR should be protected
        let mut edge = create_test_edge_with_tier(0.1, 30, EdgeTier::L2Episodic);
        edge.ltp_status = LtpStatus::Full;

        let should_prune = edge.decay();

        assert!(
            !should_prune,
            "Potentiated edges above LTP_PRUNE_FLOOR should not be pruned"
        );
    }

    #[test]
    fn test_potentiated_at_floor_stripped_and_prunable() {
        // Potentiated edge at/below LTP_PRUNE_FLOOR should have LTP stripped
        let mut edge = create_test_edge_with_tier(LTP_MIN_STRENGTH, 30, EdgeTier::L2Episodic);
        edge.ltp_status = LtpStatus::Full;

        let should_prune = edge.decay();

        // LTP gets stripped because strength <= LTP_PRUNE_FLOOR,
        // then normal prune logic applies (strength at floor, past max age)
        assert!(
            should_prune,
            "Zombie potentiated edges at floor strength should be prunable"
        );
        assert!(
            matches!(edge.ltp_status, LtpStatus::None),
            "LTP status should be stripped when strength at floor"
        );
    }

    #[test]
    fn test_salience_calculation() {
        let person_salience = EntityExtractor::calculate_base_salience(&EntityLabel::Person, false);
        let person_proper_salience =
            EntityExtractor::calculate_base_salience(&EntityLabel::Person, true);

        assert_eq!(person_salience, 0.8);
        assert!((person_proper_salience - 0.96).abs() < 0.01); // 0.8 * 1.2 = 0.96
    }

    #[test]
    fn test_salience_caps_at_one() {
        // Person (0.8) * 1.2 = 0.96, should not exceed 1.0
        let salience = EntityExtractor::calculate_base_salience(&EntityLabel::Person, true);
        assert!(salience <= 1.0);
    }

    #[test]
    fn test_hebbian_strength_no_episode() {
        // Create a temporary graph memory for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path()).unwrap();

        // Random memory ID with no associated episode should return 0.5 (neutral)
        let fake_memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let strength = graph.get_memory_hebbian_strength(&fake_memory_id);
        assert_eq!(strength, Some(0.5), "No episode should return neutral 0.5");
    }

    #[test]
    fn test_hebbian_strength_with_episode_no_edges() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path()).unwrap();

        // Create entities
        let entity1 = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Entity1".to_string(),
            labels: vec![EntityLabel::Person],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
        };
        let entity2 = EntityNode {
            uuid: Uuid::new_v4(),
            name: "Entity2".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
        };

        let entity1_uuid = graph.add_entity(entity1.clone()).unwrap();
        let entity2_uuid = graph.add_entity(entity2.clone()).unwrap();

        // Create episode with entities but no edges
        let memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let episode = EpisodicNode {
            uuid: memory_id.0,
            name: "Test Episode".to_string(),
            content: "Test content".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![entity1_uuid, entity2_uuid],
            source: EpisodeSource::Message,
            metadata: std::collections::HashMap::new(),
        };
        graph.add_episode(episode).unwrap();

        // Episode with entities but no edges should return 0.5
        let strength = graph.get_memory_hebbian_strength(&memory_id);
        assert_eq!(
            strength,
            Some(0.5),
            "Episode without edges should return neutral 0.5"
        );
    }

    #[test]
    fn test_hebbian_strength_with_edges() {
        let temp_dir = tempfile::tempdir().unwrap();
        let graph = GraphMemory::new(temp_dir.path()).unwrap();

        // Create entities
        let entity1_uuid = Uuid::new_v4();
        let entity2_uuid = Uuid::new_v4();

        let entity1 = EntityNode {
            uuid: entity1_uuid,
            name: "Entity1".to_string(),
            labels: vec![EntityLabel::Person],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
        };
        let entity2 = EntityNode {
            uuid: entity2_uuid,
            name: "Entity2".to_string(),
            labels: vec![EntityLabel::Organization],
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: false,
        };

        graph.add_entity(entity1).unwrap();
        graph.add_entity(entity2).unwrap();

        // Create episode
        let memory_id = crate::memory::MemoryId(Uuid::new_v4());
        let episode = EpisodicNode {
            uuid: memory_id.0,
            name: "Test Episode".to_string(),
            content: "Test content".to_string(),
            valid_at: Utc::now(),
            created_at: Utc::now(),
            entity_refs: vec![entity1_uuid, entity2_uuid],
            source: EpisodeSource::Message,
            metadata: std::collections::HashMap::new(),
        };
        graph.add_episode(episode).unwrap();

        // Create edge between entities with known strength
        let edge = RelationshipEdge {
            uuid: Uuid::new_v4(),
            from_entity: entity1_uuid,
            to_entity: entity2_uuid,
            relation_type: RelationType::RelatedTo,
            strength: 0.8,
            created_at: Utc::now(),
            valid_at: Utc::now(),
            invalidated_at: None,
            source_episode_id: Some(memory_id.0),
            context: "Test context".to_string(),
            last_activated: Utc::now(), // Just activated - no decay
            activation_count: 5,
            ltp_status: LtpStatus::None,
            activation_timestamps: None,
            tier: EdgeTier::L2Episodic, // Use L2 since it has activation count
            entity_confidence: None,    // PIPE-5: Default for tests
        };
        graph.add_relationship(edge).unwrap();

        // Should return the edge strength (0.8, with minimal decay since just activated)
        let strength = graph.get_memory_hebbian_strength(&memory_id);
        assert!(strength.is_some());
        let s = strength.unwrap();
        assert!(s > 0.75 && s <= 0.8, "Strength should be ~0.8, got {}", s);
    }
}
