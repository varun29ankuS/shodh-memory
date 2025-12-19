//! Memory Replay and Interference Module (SHO-105, SHO-106)
//!
//! This module implements biologically-inspired memory consolidation mechanisms:
//!
//! ## Memory Replay (SHO-105)
//! Based on Rasch & Born (2013) - sleep consolidation research:
//! - Hippocampus replays recent experiences during rest/sleep
//! - Co-activation strengthens related memories and their associations
//! - High-value memories (important + recent + emotional) get priority
//!
//! ## Memory Interference (SHO-106)
//! Based on Anderson & Neely (1996) - retrieval competition:
//! - Retroactive interference: new learning disrupts old memories
//! - Proactive interference: old memories interfere with new learning
//! - Similar memories compete during retrieval

use crate::constants::{
    INTERFERENCE_COMPETITION_FACTOR, INTERFERENCE_MAX_TRACKED, INTERFERENCE_PROACTIVE_DECAY,
    INTERFERENCE_PROACTIVE_THRESHOLD, INTERFERENCE_RETROACTIVE_DECAY,
    INTERFERENCE_SEVERE_THRESHOLD, INTERFERENCE_SIMILARITY_THRESHOLD,
    INTERFERENCE_VULNERABILITY_HOURS, REPLAY_AROUSAL_THRESHOLD, REPLAY_BATCH_SIZE,
    REPLAY_EDGE_BOOST, REPLAY_IMPORTANCE_THRESHOLD, REPLAY_MAX_AGE_DAYS, REPLAY_MIN_CONNECTIONS,
    REPLAY_STRENGTH_BOOST,
};
use crate::memory::introspection::{ConsolidationEvent, InterferenceType};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Candidate memory for replay, scored by priority
#[derive(Debug, Clone)]
pub struct ReplayCandidate {
    pub memory_id: String,
    pub content_preview: String,
    pub importance: f32,
    pub arousal: f32,
    pub age_days: f64,
    pub connection_count: usize,
    pub priority_score: f32,
    pub connected_memory_ids: Vec<String>,
}

/// Result of a replay cycle
#[derive(Debug, Clone, Default)]
pub struct ReplayCycleResult {
    pub memories_replayed: usize,
    pub edges_strengthened: usize,
    pub total_priority_score: f32,
    pub events: Vec<ConsolidationEvent>,
}

/// Manager for memory replay during consolidation
///
/// Implements sleep-like consolidation by:
/// 1. Identifying high-value memories for replay
/// 2. Simulating co-activation during replay
/// 3. Strengthening both memories and their associations
pub struct ReplayManager {
    /// Last replay cycle timestamp
    last_replay: DateTime<Utc>,
    /// Minimum interval between replay cycles (hours)
    replay_interval_hours: i64,
    /// Replay statistics
    total_replays: usize,
}

impl Default for ReplayManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayManager {
    pub fn new() -> Self {
        Self {
            last_replay: Utc::now() - Duration::hours(24), // Allow immediate first replay
            replay_interval_hours: 1,                      // Replay every hour during active use
            total_replays: 0,
        }
    }

    /// Check if replay cycle should run
    pub fn should_replay(&self) -> bool {
        let elapsed = Utc::now() - self.last_replay;
        elapsed.num_hours() >= self.replay_interval_hours
    }

    /// Identify memories eligible for replay
    ///
    /// Selection criteria:
    /// - Recent (within REPLAY_MAX_AGE_DAYS)
    /// - Important (above REPLAY_IMPORTANCE_THRESHOLD)
    /// - Connected (at least REPLAY_MIN_CONNECTIONS)
    /// - Optionally: high emotional arousal for priority
    pub fn identify_replay_candidates(
        &self,
        memories: &[(String, f32, f32, DateTime<Utc>, Vec<String>, String)], // (id, importance, arousal, created_at, connections, content_preview)
    ) -> Vec<ReplayCandidate> {
        let now = Utc::now();
        let mut candidates: Vec<ReplayCandidate> = memories
            .iter()
            .filter_map(
                |(id, importance, arousal, created_at, connections, preview)| {
                    let age = now - *created_at;
                    let age_days = age.num_hours() as f64 / 24.0;

                    // Check eligibility
                    if age_days > REPLAY_MAX_AGE_DAYS as f64 {
                        return None;
                    }
                    if *importance < REPLAY_IMPORTANCE_THRESHOLD {
                        return None;
                    }
                    if connections.len() < REPLAY_MIN_CONNECTIONS {
                        return None;
                    }

                    // Calculate priority score
                    // Priority = importance × recency_factor × (1 + arousal_boost) × connectivity_factor
                    let recency_factor = 1.0 - (age_days / REPLAY_MAX_AGE_DAYS as f64) as f32;
                    let arousal_boost = if *arousal > REPLAY_AROUSAL_THRESHOLD {
                        (*arousal - REPLAY_AROUSAL_THRESHOLD) * 0.5
                    } else {
                        0.0
                    };
                    let connectivity_factor = 1.0 + (connections.len() as f32 / 10.0).min(0.5); // Max 50% boost

                    let priority =
                        importance * recency_factor * (1.0 + arousal_boost) * connectivity_factor;

                    Some(ReplayCandidate {
                        memory_id: id.clone(),
                        content_preview: preview.clone(),
                        importance: *importance,
                        arousal: *arousal,
                        age_days,
                        connection_count: connections.len(),
                        priority_score: priority,
                        connected_memory_ids: connections.clone(),
                    })
                },
            )
            .collect();

        // Sort by priority (highest first)
        candidates.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top REPLAY_BATCH_SIZE candidates
        candidates.truncate(REPLAY_BATCH_SIZE);
        candidates
    }

    /// Execute replay for a batch of candidates
    ///
    /// Returns strength boosts to apply to memories and edges
    pub fn execute_replay(
        &mut self,
        candidates: &[ReplayCandidate],
    ) -> (
        Vec<(String, f32)>,
        Vec<(String, String, f32)>,
        Vec<ConsolidationEvent>,
    ) {
        // (memory_id, boost), (from_id, to_id, boost), events
        let mut memory_boosts: Vec<(String, f32)> = Vec::new();
        let mut edge_boosts: Vec<(String, String, f32)> = Vec::new();
        let mut events: Vec<ConsolidationEvent> = Vec::new();
        let now = Utc::now();

        // Track replayed memories to avoid duplicate boosts
        let mut replayed: HashSet<String> = HashSet::new();

        for candidate in candidates {
            if replayed.contains(&candidate.memory_id) {
                continue;
            }

            // Boost the primary memory
            memory_boosts.push((candidate.memory_id.clone(), REPLAY_STRENGTH_BOOST));
            replayed.insert(candidate.memory_id.clone());

            // Co-activate connected memories
            let mut connected_replayed = 0;
            for connected_id in &candidate.connected_memory_ids {
                if !replayed.contains(connected_id) {
                    // Boost connected memory (slightly less than primary)
                    memory_boosts.push((connected_id.clone(), REPLAY_STRENGTH_BOOST * 0.5));
                    replayed.insert(connected_id.clone());
                }

                // Strengthen the edge between them
                edge_boosts.push((
                    candidate.memory_id.clone(),
                    connected_id.clone(),
                    REPLAY_EDGE_BOOST,
                ));
                connected_replayed += 1;
            }

            // Create replay event
            events.push(ConsolidationEvent::MemoryReplayed {
                memory_id: candidate.memory_id.clone(),
                content_preview: candidate.content_preview.clone(),
                activation_before: candidate.importance,
                activation_after: (candidate.importance + REPLAY_STRENGTH_BOOST).min(1.0),
                replay_priority: candidate.priority_score,
                connected_memories_replayed: connected_replayed,
                timestamp: now,
            });
        }

        self.last_replay = now;
        self.total_replays += candidates.len();

        (memory_boosts, edge_boosts, events)
    }

    /// Get replay statistics
    pub fn stats(&self) -> (usize, DateTime<Utc>) {
        (self.total_replays, self.last_replay)
    }
}

// =============================================================================
// MEMORY INTERFERENCE (SHO-106)
// =============================================================================

/// Record of an interference event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceRecord {
    pub interfering_memory_id: String,
    pub similarity: f32,
    pub interference_type: InterferenceType,
    pub strength_change: f32,
    pub timestamp: DateTime<Utc>,
}

/// Result of interference check during memory storage
#[derive(Debug, Clone, Default)]
pub struct InterferenceCheckResult {
    /// Retroactive interference: old memories to weaken
    pub retroactive_targets: Vec<(String, f32, f32)>, // (memory_id, similarity, decay_amount)
    /// Proactive interference: strength reduction for new memory
    pub proactive_decay: f32,
    /// Whether memories are duplicates (should merge instead of interfere)
    pub is_duplicate: bool,
    /// Events generated
    pub events: Vec<ConsolidationEvent>,
}

/// Result of retrieval competition
#[derive(Debug, Clone)]
pub struct CompetitionResult {
    /// Memory IDs that won (survive suppression)
    pub winners: Vec<(String, f32)>, // (memory_id, final_score)
    /// Memory IDs that were suppressed
    pub suppressed: Vec<String>,
    /// Competition factor applied
    pub competition_factor: f32,
    /// Event generated
    pub event: Option<ConsolidationEvent>,
}

/// Detector for memory interference effects
pub struct InterferenceDetector {
    /// Tracked interference records per memory
    interference_history: HashMap<String, Vec<InterferenceRecord>>,
    /// Total interference events
    total_interference_events: usize,
}

impl Default for InterferenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl InterferenceDetector {
    pub fn new() -> Self {
        Self {
            interference_history: HashMap::new(),
            total_interference_events: 0,
        }
    }

    /// Check for interference when storing a new memory
    ///
    /// Compares the new memory's embedding against existing memories
    /// and determines interference effects.
    pub fn check_interference(
        &mut self,
        new_memory_id: &str,
        new_memory_importance: f32,
        _new_memory_created: DateTime<Utc>,
        similar_memories: &[(String, f32, f32, DateTime<Utc>, String)], // (id, similarity, importance, created_at, content_preview)
    ) -> InterferenceCheckResult {
        let mut result = InterferenceCheckResult::default();
        let now = Utc::now();

        for (old_id, similarity, old_importance, old_created, old_preview) in similar_memories {
            // Skip self
            if old_id == new_memory_id {
                continue;
            }

            // Check if similarity exceeds threshold
            if *similarity < INTERFERENCE_SIMILARITY_THRESHOLD {
                continue;
            }

            // Check for duplicates (very high similarity)
            if *similarity >= INTERFERENCE_SEVERE_THRESHOLD {
                result.is_duplicate = true;
                // Return early - should merge, not interfere
                return result;
            }

            // Calculate interference effects
            let age_hours = (now - *old_created).num_hours();
            let is_vulnerable = age_hours < INTERFERENCE_VULNERABILITY_HOURS;

            // Retroactive interference: new memory weakens old
            if is_vulnerable || *old_importance < new_memory_importance {
                // Stronger interference for more similar memories
                let interference_strength = (*similarity - INTERFERENCE_SIMILARITY_THRESHOLD)
                    / (1.0 - INTERFERENCE_SIMILARITY_THRESHOLD);

                let decay = INTERFERENCE_RETROACTIVE_DECAY * interference_strength;
                result
                    .retroactive_targets
                    .push((old_id.clone(), *similarity, decay));

                // Record event
                result
                    .events
                    .push(ConsolidationEvent::InterferenceDetected {
                        new_memory_id: new_memory_id.to_string(),
                        old_memory_id: old_id.clone(),
                        similarity: *similarity,
                        interference_type: InterferenceType::Retroactive,
                        timestamp: now,
                    });

                result.events.push(ConsolidationEvent::MemoryWeakened {
                    memory_id: old_id.clone(),
                    content_preview: old_preview.clone(),
                    activation_before: *old_importance,
                    activation_after: (*old_importance - decay).max(0.05),
                    interfering_memory_id: new_memory_id.to_string(),
                    interference_type: InterferenceType::Retroactive,
                    timestamp: now,
                });

                // Track in history
                self.record_interference(
                    old_id,
                    new_memory_id,
                    *similarity,
                    InterferenceType::Retroactive,
                    decay,
                );
            }

            // Proactive interference: strong old memory suppresses new
            if *old_importance > INTERFERENCE_PROACTIVE_THRESHOLD {
                let interference_strength = (*similarity - INTERFERENCE_SIMILARITY_THRESHOLD)
                    / (1.0 - INTERFERENCE_SIMILARITY_THRESHOLD);

                let decay = INTERFERENCE_PROACTIVE_DECAY
                    * interference_strength
                    * (*old_importance - INTERFERENCE_PROACTIVE_THRESHOLD);

                result.proactive_decay += decay;

                result
                    .events
                    .push(ConsolidationEvent::InterferenceDetected {
                        new_memory_id: new_memory_id.to_string(),
                        old_memory_id: old_id.clone(),
                        similarity: *similarity,
                        interference_type: InterferenceType::Proactive,
                        timestamp: now,
                    });

                self.record_interference(
                    new_memory_id,
                    old_id,
                    *similarity,
                    InterferenceType::Proactive,
                    decay,
                );
            }
        }

        self.total_interference_events += result.events.len();
        result
    }

    /// Apply retrieval competition between similar memories
    ///
    /// When multiple similar memories are retrieved, they compete
    /// for activation. Stronger memories suppress weaker ones.
    pub fn apply_retrieval_competition(
        &mut self,
        candidates: &[(String, f32, f32)], // (memory_id, relevance_score, similarity_to_query)
        query_preview: &str,
    ) -> CompetitionResult {
        if candidates.len() <= 1 {
            return CompetitionResult {
                winners: candidates
                    .iter()
                    .map(|(id, score, _)| (id.clone(), *score))
                    .collect(),
                suppressed: Vec::new(),
                competition_factor: 0.0,
                event: None,
            };
        }

        // Find groups of competing memories (high similarity to each other)
        let mut scores: Vec<(String, f32)> = candidates
            .iter()
            .map(|(id, score, _)| (id.clone(), *score))
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut winners: Vec<(String, f32)> = Vec::new();
        let mut suppressed: Vec<String> = Vec::new();

        if let Some((winner_id, winner_score)) = scores.first() {
            winners.push((winner_id.clone(), *winner_score));

            // Apply competition suppression to lower-ranked memories
            for (id, score) in scores.iter().skip(1) {
                let score_ratio = score / winner_score;

                // Strong suppression for very close competitors
                if score_ratio > 0.9 {
                    let suppression = INTERFERENCE_COMPETITION_FACTOR * (1.0 - score_ratio) * 10.0;
                    let new_score = (score - suppression).max(0.0);

                    if new_score > 0.1 {
                        winners.push((id.clone(), new_score));
                    } else {
                        suppressed.push(id.clone());
                    }
                } else {
                    winners.push((id.clone(), *score));
                }
            }
        }

        let event = if !suppressed.is_empty() {
            Some(ConsolidationEvent::RetrievalCompetition {
                query_preview: query_preview.to_string(),
                winner_memory_id: winners
                    .first()
                    .map(|(id, _)| id.clone())
                    .unwrap_or_default(),
                suppressed_memory_ids: suppressed.clone(),
                competition_factor: INTERFERENCE_COMPETITION_FACTOR,
                timestamp: Utc::now(),
            })
        } else {
            None
        };

        CompetitionResult {
            winners,
            suppressed,
            competition_factor: INTERFERENCE_COMPETITION_FACTOR,
            event,
        }
    }

    /// Record an interference event
    fn record_interference(
        &mut self,
        affected_memory_id: &str,
        interfering_memory_id: &str,
        similarity: f32,
        interference_type: InterferenceType,
        strength_change: f32,
    ) {
        let record = InterferenceRecord {
            interfering_memory_id: interfering_memory_id.to_string(),
            similarity,
            interference_type,
            strength_change,
            timestamp: Utc::now(),
        };

        let history = self
            .interference_history
            .entry(affected_memory_id.to_string())
            .or_default();

        history.push(record);

        // Limit history size
        if history.len() > INTERFERENCE_MAX_TRACKED {
            history.remove(0);
        }
    }

    /// Get interference history for a memory
    pub fn get_history(&self, memory_id: &str) -> Option<&Vec<InterferenceRecord>> {
        self.interference_history.get(memory_id)
    }

    /// Get statistics
    pub fn stats(&self) -> (usize, usize) {
        (
            self.total_interference_events,
            self.interference_history.len(),
        )
    }

    /// Clear history for a deleted memory
    pub fn clear_memory(&mut self, memory_id: &str) {
        self.interference_history.remove(memory_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_candidate_identification() {
        let manager = ReplayManager::new();
        let now = Utc::now();

        // Create test memories
        let memories = vec![
            (
                "mem-1".to_string(),
                0.8,                                            // High importance
                0.7,                                            // High arousal
                now - Duration::hours(12),                      // Recent
                vec!["mem-2".to_string(), "mem-3".to_string()], // Connected
                "Important memory".to_string(),
            ),
            (
                "mem-2".to_string(),
                0.2, // Low importance - should be excluded
                0.3,
                now - Duration::hours(6),
                vec!["mem-1".to_string()],
                "Unimportant memory".to_string(),
            ),
            (
                "mem-3".to_string(),
                0.6,
                0.4,
                now - Duration::days(10), // Too old - should be excluded
                vec!["mem-1".to_string(), "mem-4".to_string()],
                "Old memory".to_string(),
            ),
        ];

        let candidates = manager.identify_replay_candidates(&memories);

        // Only mem-1 should be eligible
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].memory_id, "mem-1");
        assert!(candidates[0].priority_score > 0.0);
    }

    #[test]
    fn test_replay_execution() {
        let mut manager = ReplayManager::new();
        let now = Utc::now();

        let candidates = vec![ReplayCandidate {
            memory_id: "mem-1".to_string(),
            content_preview: "Test memory".to_string(),
            importance: 0.7,
            arousal: 0.6,
            age_days: 1.0,
            connection_count: 2,
            priority_score: 0.8,
            connected_memory_ids: vec!["mem-2".to_string(), "mem-3".to_string()],
        }];

        let (memory_boosts, edge_boosts, events) = manager.execute_replay(&candidates);

        // Primary memory should get a boost
        assert!(memory_boosts.iter().any(|(id, _)| id == "mem-1"));

        // Connected memories should get boosts
        assert!(memory_boosts.iter().any(|(id, _)| id == "mem-2"));
        assert!(memory_boosts.iter().any(|(id, _)| id == "mem-3"));

        // Edges should be strengthened
        assert_eq!(edge_boosts.len(), 2);

        // Event should be generated
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_interference_detection() {
        let mut detector = InterferenceDetector::new();
        let now = Utc::now();

        // Test retroactive interference
        let similar_memories = vec![(
            "old-mem".to_string(),
            0.90,                      // High similarity
            0.5,                       // Moderate importance
            now - Duration::hours(12), // Recent, vulnerable
            "Old memory content".to_string(),
        )];

        let result = detector.check_interference(
            "new-mem",
            0.7, // Higher importance than old
            now,
            &similar_memories,
        );

        // Should detect retroactive interference
        assert!(!result.retroactive_targets.is_empty());
        assert!(!result.events.is_empty());
    }

    #[test]
    fn test_duplicate_detection() {
        let mut detector = InterferenceDetector::new();
        let now = Utc::now();

        // Very similar memory (near duplicate)
        let similar_memories = vec![(
            "existing-mem".to_string(),
            0.98, // Very high similarity - duplicate
            0.5,
            now - Duration::hours(1),
            "Existing content".to_string(),
        )];

        let result = detector.check_interference("new-mem", 0.6, now, &similar_memories);

        // Should detect as duplicate
        assert!(result.is_duplicate);
        // No interference events for duplicates
        assert!(result.events.is_empty());
    }

    #[test]
    fn test_retrieval_competition() {
        let mut detector = InterferenceDetector::new();

        let candidates = vec![
            ("mem-1".to_string(), 0.9, 0.85),  // Winner
            ("mem-2".to_string(), 0.88, 0.82), // Close competitor
            ("mem-3".to_string(), 0.5, 0.70),  // Lower, should survive
        ];

        let result = detector.apply_retrieval_competition(&candidates, "test query");

        // Winner should be first
        assert_eq!(result.winners[0].0, "mem-1");
        // Close competitor may be suppressed depending on competition factor
        assert!(!result.winners.is_empty());
    }
}
