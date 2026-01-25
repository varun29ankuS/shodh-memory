//! Pattern-Triggered Replay Detection (PIPE-2)
//!
//! Implements biologically-inspired pattern detection for memory consolidation.
//! Instead of fixed-interval replay, this module detects meaningful patterns
//! that should trigger immediate consolidation.
//!
//! ## Neuroscience Basis
//! Based on hippocampal sharp-wave ripple research (Rasch & Born 2013):
//! - Consolidation is triggered by coherent neural activity patterns
//! - High-value memories and semantically related clusters get priority
//! - Emotional significance (salience spikes) triggers immediate replay
//!
//! ## Pattern Types
//! 1. **Entity Co-occurrence**: Memories sharing multiple named entities
//! 2. **Semantic Clustering**: Dense groups of semantically similar memories
//! 3. **Temporal Clustering**: Memories from the same session/timeframe
//! 4. **Salience Spikes**: High importance/arousal memories
//! 5. **Behavioral Changes**: Topic switches, user corrections

use crate::constants::{
    BEHAVIORAL_PATTERN_WINDOW_HOURS, ENTITY_COOCCURRENCE_THRESHOLD, ENTITY_PATTERN_CONFIDENCE,
    HIGH_AROUSAL_THRESHOLD, HIGH_IMPORTANCE_THRESHOLD, MIN_CLUSTER_SIZE, MIN_MEMORIES_PER_PATTERN,
    MIN_MEMORIES_PER_SESSION, SEMANTIC_CLUSTER_THRESHOLD, SURPRISE_THRESHOLD,
    TEMPORAL_CLUSTER_WINDOW_SECS,
};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Maximum recent memories to track for pattern detection
const MAX_RECENT_MEMORIES: usize = 500;

/// Maximum salience spikes to track
const MAX_SALIENCE_SPIKES: usize = 100;

/// Memory data for pattern analysis
#[derive(Debug, Clone)]
pub struct PatternMemory {
    pub id: String,
    pub content_preview: String,
    pub entities: Vec<String>,
    pub importance: f32,
    pub arousal: f32,
    pub created_at: DateTime<Utc>,
    pub embedding_hash: Option<u64>,
    pub session_id: Option<String>,
    pub memory_type: String,
}

/// Statistics for an entity group pattern
#[derive(Debug, Clone, Default)]
pub struct EntityPatternStats {
    pub memory_ids: Vec<String>,
    pub first_seen: Option<DateTime<Utc>>,
    pub last_seen: Option<DateTime<Utc>>,
    pub total_occurrences: usize,
    /// Whether this pattern has triggered a replay (consumed)
    pub triggered: bool,
}

/// A detected semantic cluster
#[derive(Debug, Clone)]
pub struct SemanticCluster {
    pub memory_ids: Vec<String>,
    pub centroid_memory_id: String,
    pub avg_similarity: f32,
    pub formed_at: DateTime<Utc>,
    /// Whether this cluster has triggered a replay (consumed)
    pub triggered: bool,
}

/// A detected temporal cluster (session)
#[derive(Debug, Clone)]
pub struct TemporalCluster {
    pub memory_ids: Vec<String>,
    pub session_start: DateTime<Utc>,
    pub session_end: DateTime<Utc>,
    pub session_id: Option<String>,
}

/// A salience spike event
#[derive(Debug, Clone)]
pub struct SalienceEvent {
    pub memory_id: String,
    pub importance: f32,
    pub arousal: f32,
    pub surprise_factor: f32,
    pub detected_at: DateTime<Utc>,
    /// Whether this spike has triggered a replay (consumed)
    pub triggered: bool,
}

/// Type of behavioral change detected
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BehaviorChangeType {
    /// User switched topics
    TopicSwitch { from: String, to: String },
    /// User corrected/negated something
    UserCorrection { correction_keywords: Vec<String> },
    /// User switched projects
    ProjectSwitch {
        from: Option<String>,
        to: Option<String>,
    },
    /// User is repeatedly querying the same topic
    QueryRepetition { repeated_topic: String },
}

/// Behavioral context tracking
#[derive(Debug, Clone, Default)]
pub struct BehavioralContext {
    pub current_topic: Option<String>,
    pub current_project: Option<String>,
    pub recent_queries: VecDeque<String>,
    pub last_correction: Option<DateTime<Utc>>,
}

/// Trigger types for pattern-based replay
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "trigger_type", rename_all = "snake_case")]
pub enum ReplayTrigger {
    /// Multiple memories share the same entity group
    EntityCoOccurrence {
        entities: Vec<String>,
        memory_ids: Vec<String>,
        overlap_score: f32,
        confidence: f32,
    },

    /// Dense cluster of semantically similar memories
    SemanticCluster {
        memory_ids: Vec<String>,
        centroid_id: String,
        avg_similarity: f32,
        cluster_size: usize,
    },

    /// Memories created within the same time window
    TemporalCluster {
        memory_ids: Vec<String>,
        window_secs: i64,
        session_id: Option<String>,
    },

    /// High importance/arousal memory detected
    SalienceSpike {
        memory_id: String,
        content_preview: String,
        importance: f32,
        arousal: f32,
        surprise_factor: f32,
    },

    /// User behavior changed (topic switch, correction)
    BehavioralPatternChange {
        change_type: BehaviorChangeType,
        affected_memory_ids: Vec<String>,
        context: String,
    },

    /// Fallback: fixed time interval (legacy behavior)
    TimerInterval,
}

impl ReplayTrigger {
    /// Get a human-readable description of the trigger
    pub fn description(&self) -> String {
        match self {
            ReplayTrigger::EntityCoOccurrence { entities, .. } => {
                format!("Entity co-occurrence: {}", entities.join(", "))
            }
            ReplayTrigger::SemanticCluster { cluster_size, .. } => {
                format!("Semantic cluster of {} memories", cluster_size)
            }
            ReplayTrigger::TemporalCluster { memory_ids, .. } => {
                format!("Temporal cluster: {} memories in session", memory_ids.len())
            }
            ReplayTrigger::SalienceSpike {
                content_preview, ..
            } => {
                format!("Salience spike: {}", content_preview)
            }
            ReplayTrigger::BehavioralPatternChange { change_type, .. } => match change_type {
                BehaviorChangeType::TopicSwitch { from, to } => {
                    format!("Topic switch: {} -> {}", from, to)
                }
                BehaviorChangeType::UserCorrection { .. } => "User correction detected".to_string(),
                BehaviorChangeType::ProjectSwitch { from, to } => {
                    format!(
                        "Project switch: {} -> {}",
                        from.as_deref().unwrap_or("none"),
                        to.as_deref().unwrap_or("none")
                    )
                }
                BehaviorChangeType::QueryRepetition { repeated_topic } => {
                    format!("Repeated queries about: {}", repeated_topic)
                }
            },
            ReplayTrigger::TimerInterval => "Fixed interval timer".to_string(),
        }
    }

    /// Get the memory IDs affected by this trigger
    pub fn memory_ids(&self) -> Vec<String> {
        match self {
            ReplayTrigger::EntityCoOccurrence { memory_ids, .. } => memory_ids.clone(),
            ReplayTrigger::SemanticCluster { memory_ids, .. } => memory_ids.clone(),
            ReplayTrigger::TemporalCluster { memory_ids, .. } => memory_ids.clone(),
            ReplayTrigger::SalienceSpike { memory_id, .. } => vec![memory_id.clone()],
            ReplayTrigger::BehavioralPatternChange {
                affected_memory_ids,
                ..
            } => affected_memory_ids.clone(),
            ReplayTrigger::TimerInterval => vec![],
        }
    }

    /// Get trigger type name for metrics/logging
    pub fn trigger_type_name(&self) -> &'static str {
        match self {
            ReplayTrigger::EntityCoOccurrence { .. } => "entity_cooccurrence",
            ReplayTrigger::SemanticCluster { .. } => "semantic_cluster",
            ReplayTrigger::TemporalCluster { .. } => "temporal_cluster",
            ReplayTrigger::SalienceSpike { .. } => "salience_spike",
            ReplayTrigger::BehavioralPatternChange { .. } => "behavioral_change",
            ReplayTrigger::TimerInterval => "timer_interval",
        }
    }
}

/// Result of pattern detection cycle
#[derive(Debug, Clone, Default)]
pub struct PatternDetectionResult {
    pub triggers: Vec<ReplayTrigger>,
    pub entity_patterns_found: usize,
    pub semantic_clusters_found: usize,
    pub temporal_clusters_found: usize,
    pub salience_spikes_found: usize,
    pub behavioral_changes_found: usize,
}

/// Pattern detector for triggering memory replay
///
/// Monitors incoming memories and detects patterns that should
/// trigger consolidation rather than waiting for fixed intervals.
pub struct PatternDetector {
    /// Recent memories for pattern analysis (bounded)
    recent_memories: VecDeque<PatternMemory>,

    /// Entity co-occurrence patterns: entity_group_key -> stats
    entity_patterns: HashMap<String, EntityPatternStats>,

    /// Detected semantic clusters (updated periodically)
    semantic_clusters: Vec<SemanticCluster>,

    /// Current temporal cluster (session-based)
    current_temporal_cluster: Option<TemporalCluster>,

    /// Recent salience spikes
    salience_spikes: VecDeque<SalienceEvent>,

    /// Behavioral context tracking
    behavioral_context: BehavioralContext,

    /// Running average of recent memory importance (for surprise detection)
    importance_moving_avg: f32,

    /// Count of memories seen (for moving average)
    memories_seen: usize,

    /// Last pattern detection timestamp
    last_detection: DateTime<Utc>,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector {
    pub fn new() -> Self {
        Self {
            recent_memories: VecDeque::with_capacity(MAX_RECENT_MEMORIES),
            entity_patterns: HashMap::new(),
            semantic_clusters: Vec::new(),
            current_temporal_cluster: None,
            salience_spikes: VecDeque::with_capacity(MAX_SALIENCE_SPIKES),
            behavioral_context: BehavioralContext::default(),
            importance_moving_avg: 0.5,
            memories_seen: 0,
            last_detection: Utc::now() - Duration::hours(1),
        }
    }

    /// Register a new memory for pattern tracking
    ///
    /// Call this when a memory is created to enable pattern detection.
    pub fn register_memory(&mut self, memory: PatternMemory) {
        // Update moving average of importance
        self.memories_seen += 1;
        let alpha = 0.1_f32.min(1.0 / self.memories_seen as f32);
        self.importance_moving_avg =
            alpha * memory.importance + (1.0 - alpha) * self.importance_moving_avg;

        // Track entity co-occurrence
        if memory.entities.len() >= 2 {
            let key = self.entity_group_key(&memory.entities);
            let stats = self.entity_patterns.entry(key).or_default();
            stats.memory_ids.push(memory.id.clone());
            stats.total_occurrences += 1;
            let now = Utc::now();
            if stats.first_seen.is_none() {
                stats.first_seen = Some(now);
            }
            stats.last_seen = Some(now);
        }

        // Update temporal cluster
        self.update_temporal_cluster(&memory);

        // Add to recent memories (with bounds)
        if self.recent_memories.len() >= MAX_RECENT_MEMORIES {
            self.recent_memories.pop_front();
        }
        self.recent_memories.push_back(memory);
    }

    /// Detect all patterns and return triggers
    ///
    /// This is the main entry point for pattern detection.
    /// Call periodically (e.g., every 5 minutes) or after significant events.
    pub fn detect_patterns(&mut self) -> PatternDetectionResult {
        let mut result = PatternDetectionResult::default();
        let now = Utc::now();

        // 1. Check for entity co-occurrence patterns
        let entity_triggers = self.detect_entity_patterns();
        result.entity_patterns_found = entity_triggers.len();
        result.triggers.extend(entity_triggers);

        // 2. Check for temporal clusters
        if let Some(trigger) = self.detect_temporal_cluster() {
            result.temporal_clusters_found = 1;
            result.triggers.push(trigger);
        }

        // 3. Check for salience spikes (process pending)
        let salience_triggers = self.process_salience_spikes();
        result.salience_spikes_found = salience_triggers.len();
        result.triggers.extend(salience_triggers);

        // Note: Semantic clustering requires embeddings and is done separately
        // via detect_semantic_clusters() which takes similarity data as input

        self.last_detection = now;
        result
    }

    /// Check for salience spike (call when storing a new memory)
    ///
    /// Returns Some(trigger) if the memory is surprising enough to warrant
    /// immediate replay.
    pub fn check_salience_spike(&mut self, memory: &PatternMemory) -> Option<ReplayTrigger> {
        // Calculate surprise factor
        let surprise_factor = (memory.importance - self.importance_moving_avg).abs();

        // Check thresholds
        let is_spike = memory.importance > HIGH_IMPORTANCE_THRESHOLD
            || memory.arousal > HIGH_AROUSAL_THRESHOLD
            || surprise_factor > SURPRISE_THRESHOLD;

        if is_spike {
            let event = SalienceEvent {
                memory_id: memory.id.clone(),
                importance: memory.importance,
                arousal: memory.arousal,
                surprise_factor,
                detected_at: Utc::now(),
                triggered: false,
            };

            // Track spike
            if self.salience_spikes.len() >= MAX_SALIENCE_SPIKES {
                self.salience_spikes.pop_front();
            }
            self.salience_spikes.push_back(event.clone());

            // Return immediate trigger for high-surprise events
            if surprise_factor > SURPRISE_THRESHOLD * 1.5 || memory.arousal > 0.8 {
                return Some(ReplayTrigger::SalienceSpike {
                    memory_id: memory.id.clone(),
                    content_preview: memory.content_preview.clone(),
                    importance: memory.importance,
                    arousal: memory.arousal,
                    surprise_factor,
                });
            }
        }

        None
    }

    /// Detect semantic clusters from similarity data
    ///
    /// Call this with pre-computed similarity data (from vector search).
    /// Returns cluster triggers if dense clusters are found.
    pub fn detect_semantic_clusters(
        &mut self,
        similarities: &[(String, String, f32)], // (memory_id_1, memory_id_2, similarity)
    ) -> Vec<ReplayTrigger> {
        let mut triggers = Vec::new();

        // Build adjacency list of similar memories
        let mut adjacency: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        for (id1, id2, sim) in similarities {
            if *sim >= SEMANTIC_CLUSTER_THRESHOLD {
                adjacency
                    .entry(id1.clone())
                    .or_default()
                    .push((id2.clone(), *sim));
                adjacency
                    .entry(id2.clone())
                    .or_default()
                    .push((id1.clone(), *sim));
            }
        }

        // Find connected components (clusters)
        let mut visited: HashSet<String> = HashSet::new();
        let mut clusters: Vec<(Vec<String>, f32)> = Vec::new();

        for start_id in adjacency.keys() {
            if visited.contains(start_id) {
                continue;
            }

            // BFS to find cluster
            let mut cluster = Vec::new();
            let mut queue = VecDeque::new();
            let mut total_sim = 0.0_f32;
            let mut sim_count = 0;

            queue.push_back(start_id.clone());
            visited.insert(start_id.clone());

            while let Some(current) = queue.pop_front() {
                cluster.push(current.clone());

                if let Some(neighbors) = adjacency.get(&current) {
                    for (neighbor, sim) in neighbors {
                        if !visited.contains(neighbor) {
                            visited.insert(neighbor.clone());
                            queue.push_back(neighbor.clone());
                            total_sim += sim;
                            sim_count += 1;
                        }
                    }
                }
            }

            if cluster.len() >= MIN_CLUSTER_SIZE {
                let avg_sim = if sim_count > 0 {
                    total_sim / sim_count as f32
                } else {
                    SEMANTIC_CLUSTER_THRESHOLD
                };
                clusters.push((cluster, avg_sim));
            }
        }

        // Create triggers for significant clusters
        for (memory_ids, avg_similarity) in clusters {
            let centroid_id = memory_ids.first().cloned().unwrap_or_default();
            let cluster_size = memory_ids.len();

            // Store cluster for tracking (marked as triggered since we're returning it)
            self.semantic_clusters.push(SemanticCluster {
                memory_ids: memory_ids.clone(),
                centroid_memory_id: centroid_id.clone(),
                avg_similarity,
                formed_at: Utc::now(),
                triggered: true, // Already consumed - trigger returned immediately
            });

            triggers.push(ReplayTrigger::SemanticCluster {
                memory_ids,
                centroid_id,
                avg_similarity,
                cluster_size,
            });
        }

        triggers
    }

    /// Check for behavioral pattern change
    ///
    /// Call when user behavior signals are detected (from feedback system).
    pub fn check_behavioral_change(
        &mut self,
        new_topic: Option<&str>,
        new_project: Option<&str>,
        correction_detected: bool,
        correction_keywords: &[String],
    ) -> Option<ReplayTrigger> {
        let now = Utc::now();

        // Check for topic switch
        if let Some(topic) = new_topic {
            let old_topic = self.behavioral_context.current_topic.clone();
            if let Some(ref old) = old_topic {
                if old != topic {
                    let affected = self.find_memories_by_topic(old);
                    self.behavioral_context.current_topic = Some(topic.to_string());

                    if !affected.is_empty() {
                        return Some(ReplayTrigger::BehavioralPatternChange {
                            change_type: BehaviorChangeType::TopicSwitch {
                                from: old.clone(),
                                to: topic.to_string(),
                            },
                            affected_memory_ids: affected,
                            context: format!("Switched from {} to {}", old, topic),
                        });
                    }
                }
            } else {
                self.behavioral_context.current_topic = Some(topic.to_string());
            }
        }

        // Check for project switch
        if let Some(project) = new_project {
            if self.behavioral_context.current_project.as_deref() != Some(project) {
                let old_project = self.behavioral_context.current_project.clone();
                self.behavioral_context.current_project = Some(project.to_string());

                if old_project.is_some() {
                    let affected = self.find_memories_by_project(old_project.as_deref());
                    if !affected.is_empty() {
                        return Some(ReplayTrigger::BehavioralPatternChange {
                            change_type: BehaviorChangeType::ProjectSwitch {
                                from: old_project,
                                to: Some(project.to_string()),
                            },
                            affected_memory_ids: affected,
                            context: format!("Switched to project: {}", project),
                        });
                    }
                }
            }
        }

        // Check for user correction
        if correction_detected && !correction_keywords.is_empty() {
            // Only trigger if not too recent
            let should_trigger = self
                .behavioral_context
                .last_correction
                .map(|t| (now - t).num_hours() >= BEHAVIORAL_PATTERN_WINDOW_HOURS)
                .unwrap_or(true);

            if should_trigger {
                self.behavioral_context.last_correction = Some(now);
                let affected = self.find_memories_by_keywords(correction_keywords);

                if !affected.is_empty() {
                    return Some(ReplayTrigger::BehavioralPatternChange {
                        change_type: BehaviorChangeType::UserCorrection {
                            correction_keywords: correction_keywords.to_vec(),
                        },
                        affected_memory_ids: affected,
                        context: format!("User correction: {}", correction_keywords.join(", ")),
                    });
                }
            }
        }

        None
    }

    /// Get recent pattern detection statistics
    pub fn stats(&self) -> PatternDetectorStats {
        PatternDetectorStats {
            recent_memories_tracked: self.recent_memories.len(),
            entity_patterns_tracked: self.entity_patterns.len(),
            semantic_clusters_tracked: self.semantic_clusters.len(),
            salience_spikes_tracked: self.salience_spikes.len(),
            importance_moving_avg: self.importance_moving_avg,
            last_detection: self.last_detection,
        }
    }

    /// Clear processed patterns to prevent memory growth
    ///
    /// Removes patterns that have triggered replay (consumed).
    /// This is event-based cleanup, not time-based, aligning with the
    /// neuroscience-inspired philosophy: patterns persist until processed.
    pub fn cleanup(&mut self) {
        // Remove entity patterns that have triggered
        self.entity_patterns.retain(|_, stats| !stats.triggered);

        // Remove semantic clusters that have triggered
        self.semantic_clusters.retain(|c| !c.triggered);

        // Remove salience spikes that have triggered
        self.salience_spikes.retain(|s| !s.triggered);
    }

    // =========================================================================
    // Private Helper Methods
    // =========================================================================

    /// Create a canonical key for an entity group
    fn entity_group_key(&self, entities: &[String]) -> String {
        let mut sorted: Vec<_> = entities.iter().map(|e| e.to_lowercase()).collect();
        sorted.sort();
        sorted.join("|")
    }

    /// Update temporal cluster tracking
    fn update_temporal_cluster(&mut self, memory: &PatternMemory) {
        let now = memory.created_at;
        let window = Duration::seconds(TEMPORAL_CLUSTER_WINDOW_SECS);

        match &mut self.current_temporal_cluster {
            Some(cluster) => {
                // Check if memory falls within current cluster window
                if now - cluster.session_end <= window {
                    cluster.memory_ids.push(memory.id.clone());
                    cluster.session_end = now;
                    if memory.session_id.is_some() && cluster.session_id.is_none() {
                        cluster.session_id = memory.session_id.clone();
                    }
                } else {
                    // Start new cluster
                    self.current_temporal_cluster = Some(TemporalCluster {
                        memory_ids: vec![memory.id.clone()],
                        session_start: now,
                        session_end: now,
                        session_id: memory.session_id.clone(),
                    });
                }
            }
            None => {
                self.current_temporal_cluster = Some(TemporalCluster {
                    memory_ids: vec![memory.id.clone()],
                    session_start: now,
                    session_end: now,
                    session_id: memory.session_id.clone(),
                });
            }
        }
    }

    /// Detect entity co-occurrence patterns
    fn detect_entity_patterns(&mut self) -> Vec<ReplayTrigger> {
        let mut triggers = Vec::new();
        let mut triggered_keys = Vec::new();

        for (key, stats) in &self.entity_patterns {
            if stats.memory_ids.len() >= MIN_MEMORIES_PER_PATTERN && !stats.triggered {
                // Calculate confidence based on recency and frequency
                let recency_factor = stats
                    .last_seen
                    .map(|t| {
                        let age_hours = (Utc::now() - t).num_hours() as f32;
                        (24.0 - age_hours.min(24.0)) / 24.0
                    })
                    .unwrap_or(0.5);

                let frequency_factor =
                    (stats.total_occurrences as f32 / stats.memory_ids.len() as f32).min(1.0);

                let confidence = recency_factor * 0.6 + frequency_factor * 0.4;

                if confidence >= ENTITY_PATTERN_CONFIDENCE {
                    let entities: Vec<String> = key.split('|').map(String::from).collect();

                    triggers.push(ReplayTrigger::EntityCoOccurrence {
                        entities,
                        memory_ids: stats.memory_ids.clone(),
                        overlap_score: ENTITY_COOCCURRENCE_THRESHOLD,
                        confidence,
                    });

                    triggered_keys.push(key.clone());
                }
            }
        }

        // Mark patterns as triggered (consumed)
        for key in triggered_keys {
            if let Some(stats) = self.entity_patterns.get_mut(&key) {
                stats.triggered = true;
            }
        }

        triggers
    }

    /// Detect temporal cluster trigger
    fn detect_temporal_cluster(&self) -> Option<ReplayTrigger> {
        self.current_temporal_cluster.as_ref().and_then(|cluster| {
            if cluster.memory_ids.len() >= MIN_MEMORIES_PER_SESSION {
                Some(ReplayTrigger::TemporalCluster {
                    memory_ids: cluster.memory_ids.clone(),
                    window_secs: TEMPORAL_CLUSTER_WINDOW_SECS,
                    session_id: cluster.session_id.clone(),
                })
            } else {
                None
            }
        })
    }

    /// Process pending salience spikes into triggers
    fn process_salience_spikes(&mut self) -> Vec<ReplayTrigger> {
        let mut triggers = Vec::new();
        let mut triggered_indices = Vec::new();

        for (idx, spike) in self.salience_spikes.iter().enumerate() {
            // Only process spikes not yet triggered and above threshold
            if !spike.triggered && spike.surprise_factor > SURPRISE_THRESHOLD {
                triggers.push(ReplayTrigger::SalienceSpike {
                    memory_id: spike.memory_id.clone(),
                    content_preview: String::new(), // Would need to look up
                    importance: spike.importance,
                    arousal: spike.arousal,
                    surprise_factor: spike.surprise_factor,
                });
                triggered_indices.push(idx);
            }
        }

        // Mark spikes as triggered (consumed)
        for idx in triggered_indices {
            if let Some(spike) = self.salience_spikes.get_mut(idx) {
                spike.triggered = true;
            }
        }

        triggers
    }

    /// Find memories related to a topic (simple text match)
    fn find_memories_by_topic(&self, topic: &str) -> Vec<String> {
        let topic_lower = topic.to_lowercase();
        self.recent_memories
            .iter()
            .filter(|m| {
                m.content_preview.to_lowercase().contains(&topic_lower)
                    || m.entities
                        .iter()
                        .any(|e| e.to_lowercase().contains(&topic_lower))
            })
            .map(|m| m.id.clone())
            .collect()
    }

    /// Find memories by project
    fn find_memories_by_project(&self, project: Option<&str>) -> Vec<String> {
        match project {
            Some(proj) => self
                .recent_memories
                .iter()
                .filter(|m| m.session_id.as_deref() == Some(proj))
                .map(|m| m.id.clone())
                .collect(),
            None => Vec::new(),
        }
    }

    /// Find memories by keywords
    fn find_memories_by_keywords(&self, keywords: &[String]) -> Vec<String> {
        let keywords_lower: Vec<_> = keywords.iter().map(|k| k.to_lowercase()).collect();
        self.recent_memories
            .iter()
            .filter(|m| {
                let content_lower = m.content_preview.to_lowercase();
                keywords_lower.iter().any(|kw| content_lower.contains(kw))
            })
            .map(|m| m.id.clone())
            .collect()
    }
}

/// Statistics about pattern detection
#[derive(Debug, Clone)]
pub struct PatternDetectorStats {
    pub recent_memories_tracked: usize,
    pub entity_patterns_tracked: usize,
    pub semantic_clusters_tracked: usize,
    pub salience_spikes_tracked: usize,
    pub importance_moving_avg: f32,
    pub last_detection: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_memory(id: &str, entities: Vec<&str>, importance: f32) -> PatternMemory {
        PatternMemory {
            id: id.to_string(),
            content_preview: format!("Test memory {}", id),
            entities: entities.into_iter().map(String::from).collect(),
            importance,
            arousal: 0.5,
            created_at: Utc::now(),
            embedding_hash: None,
            session_id: None,
            memory_type: "Observation".to_string(),
        }
    }

    #[test]
    fn test_entity_pattern_detection() {
        let mut detector = PatternDetector::new();

        // Register memories with shared entities
        detector.register_memory(make_test_memory("m1", vec!["Rust", "HNSW"], 0.7));
        detector.register_memory(make_test_memory("m2", vec!["Rust", "HNSW"], 0.6));
        detector.register_memory(make_test_memory("m3", vec!["Rust", "HNSW"], 0.8));

        let result = detector.detect_patterns();

        // Should detect entity co-occurrence pattern
        assert!(
            result.entity_patterns_found > 0,
            "Should detect entity pattern"
        );
    }

    #[test]
    fn test_salience_spike_detection() {
        let mut detector = PatternDetector::new();

        // Register some baseline memories
        for i in 0..10 {
            detector.register_memory(make_test_memory(&format!("m{}", i), vec![], 0.5));
        }

        // Register a high-importance memory
        let spike_memory = PatternMemory {
            id: "spike".to_string(),
            content_preview: "Critical error detected".to_string(),
            entities: vec![],
            importance: 0.95,
            arousal: 0.9,
            created_at: Utc::now(),
            embedding_hash: None,
            session_id: None,
            memory_type: "Error".to_string(),
        };

        let trigger = detector.check_salience_spike(&spike_memory);
        assert!(trigger.is_some(), "Should detect salience spike");
    }

    #[test]
    fn test_temporal_cluster() {
        let mut detector = PatternDetector::new();

        // Register memories in quick succession
        for i in 0..5 {
            detector.register_memory(make_test_memory(&format!("m{}", i), vec![], 0.5));
        }

        let result = detector.detect_patterns();

        // Should detect temporal cluster
        assert!(
            result
                .triggers
                .iter()
                .any(|t| matches!(t, ReplayTrigger::TemporalCluster { .. })),
            "Should detect temporal cluster"
        );
    }

    #[test]
    fn test_semantic_cluster_detection() {
        let mut detector = PatternDetector::new();

        // Provide similarity data forming a cluster
        let similarities = vec![
            ("m1".to_string(), "m2".to_string(), 0.85),
            ("m2".to_string(), "m3".to_string(), 0.82),
            ("m1".to_string(), "m3".to_string(), 0.80),
        ];

        let triggers = detector.detect_semantic_clusters(&similarities);

        // Should detect semantic cluster
        assert!(!triggers.is_empty(), "Should detect semantic cluster");
    }
}
