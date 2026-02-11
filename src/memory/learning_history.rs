//! Learning History Storage
//!
//! Persistent storage for significant learning events. This enables:
//! - Recency-weighted retrieval (recently learned = more relevant)
//! - Learning velocity tracking (rapidly reinforced = more relevant)
//! - Session coherence (current session learning prioritized)
//! - Audit trail (what did the system learn and when)
//!
//! Storage schema:
//! - `learning:{user_id}:{timestamp_nanos}` - Primary event storage (time-ordered)
//! - `learning_by_memory:{user_id}:{memory_id}:{timestamp}` - Index by memory
//! - `learning_by_type:{user_id}:{event_type}:{timestamp}` - Index by event type
//! - `learning_stats:{user_id}` - Aggregated learning statistics

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use rocksdb::{IteratorMode, DB};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::introspection::ConsolidationEvent;

/// Stored learning event with full event fidelity
///
/// Stores the complete ConsolidationEvent (including its serde tag) using
/// MessagePack (rmp-serde), which is a binary format that supports tagged enums.
/// This preserves all event metadata for accurate retrieval and reporting.
///
/// Additional index fields enable efficient queries without deserializing
/// the full event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredLearningEvent {
    /// The full consolidation event (stored with rmp-serde, supports tagged enums)
    pub event: ConsolidationEvent,
    /// Event type for indexing and queries (denormalized for fast filtering)
    pub event_type: LearningEventType,
    /// Primary memory ID involved (if any) - for memory index
    pub memory_id: Option<String>,
    /// Secondary memory ID (for edges/interference) - for memory index
    pub related_memory_id: Option<String>,
    /// Fact ID (if fact-related) - for fact index
    pub fact_id: Option<String>,
}

/// Categorized learning event types for indexing and querying
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    bincode::Encode,
    bincode::Decode,
)]
#[serde(rename_all = "snake_case")]
pub enum LearningEventType {
    /// Edge became permanent (LTP)
    EdgePotentiated,
    /// New semantic fact extracted
    FactExtracted,
    /// Fact was deleted (knowledge lost)
    FactDeleted,
    /// Fact was reinforced
    FactReinforced,
    /// Memory interference detected
    InterferenceDetected,
    /// Memory was replayed during consolidation
    MemoryReplayed,
    /// Memory promoted to higher tier
    MemoryPromoted,
    /// Replay cycle completed (summary)
    ReplayCycleCompleted,
    /// Maintenance cycle completed (summary)
    MaintenanceCycleCompleted,
}

impl LearningEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EdgePotentiated => "edge_potentiated",
            Self::FactExtracted => "fact_extracted",
            Self::FactDeleted => "fact_deleted",
            Self::FactReinforced => "fact_reinforced",
            Self::InterferenceDetected => "interference_detected",
            Self::MemoryReplayed => "memory_replayed",
            Self::MemoryPromoted => "memory_promoted",
            Self::ReplayCycleCompleted => "replay_cycle_completed",
            Self::MaintenanceCycleCompleted => "maintenance_cycle_completed",
        }
    }
}

/// Learning velocity for a memory or fact
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningVelocity {
    /// Number of learning events in the window
    pub event_count: usize,
    /// Types of events that occurred
    pub event_types: Vec<LearningEventType>,
    /// Most recent event timestamp
    pub last_event: Option<DateTime<Utc>>,
    /// Velocity score (events per day, weighted by recency)
    pub velocity_score: f32,
}

/// Aggregated learning statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningStats {
    pub total_events: usize,
    pub events_by_type: HashMap<String, usize>,
    pub events_last_24h: usize,
    pub events_last_7d: usize,
    pub most_active_memories: Vec<(String, usize)>,
    pub potentiation_count: usize,
    pub interference_count: usize,
}

/// Persistent storage for learning history
pub struct LearningHistoryStore {
    db: Arc<DB>,
}

impl LearningHistoryStore {
    /// Create a new learning history store
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Record a significant learning event
    pub fn record(&self, user_id: &str, event: &ConsolidationEvent) -> Result<()> {
        let (event_type, memory_id, related_memory_id, fact_id) = Self::classify_event(event);

        let stored = StoredLearningEvent {
            event: event.clone(),
            event_type,
            memory_id: memory_id.clone(),
            related_memory_id: related_memory_id.clone(),
            fact_id: fact_id.clone(),
        };

        let timestamp = event.timestamp();
        let timestamp_micros = timestamp.timestamp_micros();

        // Primary storage (time-ordered)
        // Use MessagePack (rmp-serde) - binary format that supports serde tagged enums
        let key = format!("learning:{}:{:020}", user_id, timestamp_micros);
        let value = rmp_serde::to_vec(&stored)?;
        self.db.put(key.as_bytes(), &value)?;

        // Index by memory ID
        if let Some(ref mem_id) = memory_id {
            let mem_key = format!(
                "learning_by_memory:{}:{}:{:020}",
                user_id, mem_id, timestamp_micros
            );
            self.db.put(mem_key.as_bytes(), key.as_bytes())?;
        }

        // Index related memory too (for edges/interference)
        if let Some(ref related_id) = related_memory_id {
            let related_key = format!(
                "learning_by_memory:{}:{}:{:020}",
                user_id, related_id, timestamp_micros
            );
            self.db.put(related_key.as_bytes(), key.as_bytes())?;
        }

        // Index by fact ID
        if let Some(ref f_id) = fact_id {
            let fact_key = format!(
                "learning_by_fact:{}:{}:{:020}",
                user_id, f_id, timestamp_micros
            );
            self.db.put(fact_key.as_bytes(), key.as_bytes())?;
        }

        // Index by event type
        let type_key = format!(
            "learning_by_type:{}:{}:{:020}",
            user_id,
            event_type.as_str(),
            timestamp_micros
        );
        self.db.put(type_key.as_bytes(), key.as_bytes())?;

        Ok(())
    }

    /// Classify a consolidation event for indexing
    fn classify_event(
        event: &ConsolidationEvent,
    ) -> (
        LearningEventType,
        Option<String>,
        Option<String>,
        Option<String>,
    ) {
        match event {
            ConsolidationEvent::EdgePotentiated {
                from_memory_id,
                to_memory_id,
                ..
            } => (
                LearningEventType::EdgePotentiated,
                Some(from_memory_id.clone()),
                Some(to_memory_id.clone()),
                None,
            ),
            ConsolidationEvent::FactExtracted { fact_id, .. } => (
                LearningEventType::FactExtracted,
                None,
                None,
                Some(fact_id.clone()),
            ),
            ConsolidationEvent::FactDeleted { fact_id, .. } => (
                LearningEventType::FactDeleted,
                None,
                None,
                Some(fact_id.clone()),
            ),
            ConsolidationEvent::FactReinforced { fact_id, .. } => (
                LearningEventType::FactReinforced,
                None,
                None,
                Some(fact_id.clone()),
            ),
            ConsolidationEvent::InterferenceDetected {
                new_memory_id,
                old_memory_id,
                ..
            } => (
                LearningEventType::InterferenceDetected,
                Some(new_memory_id.clone()),
                Some(old_memory_id.clone()),
                None,
            ),
            ConsolidationEvent::MemoryReplayed { memory_id, .. } => (
                LearningEventType::MemoryReplayed,
                Some(memory_id.clone()),
                None,
                None,
            ),
            ConsolidationEvent::MemoryPromoted { memory_id, .. } => (
                LearningEventType::MemoryPromoted,
                Some(memory_id.clone()),
                None,
                None,
            ),
            ConsolidationEvent::ReplayCycleCompleted { .. } => {
                (LearningEventType::ReplayCycleCompleted, None, None, None)
            }
            ConsolidationEvent::MaintenanceCycleCompleted { .. } => (
                LearningEventType::MaintenanceCycleCompleted,
                None,
                None,
                None,
            ),
            // These are not significant - shouldn't be stored
            _ => (
                LearningEventType::MaintenanceCycleCompleted,
                None,
                None,
                None,
            ),
        }
    }

    /// Query learning events since a timestamp
    pub fn events_since(
        &self,
        user_id: &str,
        since: DateTime<Utc>,
    ) -> Result<Vec<StoredLearningEvent>> {
        let since_micros = since.timestamp_micros();
        let prefix = format!("learning:{}:", user_id);
        let start_key = format!("learning:{}:{:020}", user_id, since_micros);

        let mut events = Vec::new();
        let iter = self.db.iterator(IteratorMode::From(
            start_key.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            // Stop if we've moved past this user's events
            if !key_str.starts_with(&prefix) {
                break;
            }

            if let Ok(event) = rmp_serde::from_slice::<StoredLearningEvent>(&value) {
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Query learning events in a time range
    pub fn events_in_range(
        &self,
        user_id: &str,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> Result<Vec<StoredLearningEvent>> {
        let events = self.events_since(user_id, since)?;
        Ok(events
            .into_iter()
            .filter(|e| e.event.timestamp() <= until)
            .collect())
    }

    /// Get learning events for a specific memory
    pub fn events_for_memory(
        &self,
        user_id: &str,
        memory_id: &str,
        limit: usize,
    ) -> Result<Vec<StoredLearningEvent>> {
        let prefix = format!("learning_by_memory:{}:{}:", user_id, memory_id);
        let mut events = Vec::new();

        let iter = self.db.prefix_iterator(prefix.as_bytes());
        for item in iter.take(limit) {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            // Value is the primary key - fetch the actual event
            let primary_key = String::from_utf8_lossy(&value);
            if let Some(event_data) = self.db.get(primary_key.as_bytes())? {
                if let Ok(event) = rmp_serde::from_slice::<StoredLearningEvent>(&event_data) {
                    events.push(event);
                }
            }
        }

        Ok(events)
    }

    /// Get learning events for a specific fact
    pub fn events_for_fact(
        &self,
        user_id: &str,
        fact_id: &str,
        limit: usize,
    ) -> Result<Vec<StoredLearningEvent>> {
        let prefix = format!("learning_by_fact:{}:{}:", user_id, fact_id);
        let mut events = Vec::new();

        let iter = self.db.prefix_iterator(prefix.as_bytes());
        for item in iter.take(limit) {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let primary_key = String::from_utf8_lossy(&value);
            if let Some(event_data) = self.db.get(primary_key.as_bytes())? {
                if let Ok(event) = rmp_serde::from_slice::<StoredLearningEvent>(&event_data) {
                    events.push(event);
                }
            }
        }

        Ok(events)
    }

    /// Calculate learning velocity for a memory
    ///
    /// Returns a velocity score based on recent learning activity.
    /// Higher velocity = more actively being learned = more relevant.
    pub fn memory_learning_velocity(
        &self,
        user_id: &str,
        memory_id: &str,
        window_hours: i64,
    ) -> Result<LearningVelocity> {
        let since = Utc::now() - Duration::hours(window_hours);
        let events = self.events_for_memory(user_id, memory_id, 100)?;

        let recent_events: Vec<_> = events
            .into_iter()
            .filter(|e| e.event.timestamp() >= since)
            .collect();

        if recent_events.is_empty() {
            return Ok(LearningVelocity::default());
        }

        let event_count = recent_events.len();
        let event_types: Vec<_> = recent_events.iter().map(|e| e.event_type).collect();
        let last_event = recent_events.iter().map(|e| e.event.timestamp()).max();

        // Calculate velocity score with recency weighting
        let now = Utc::now();
        let mut weighted_sum = 0.0;
        for event in &recent_events {
            let age_hours = (now - event.event.timestamp()).num_hours() as f32;
            // Exponential decay: recent events weighted more heavily
            let weight = (-age_hours / window_hours as f32).exp();
            // Potentiation and interference events are more significant
            let type_weight = match event.event_type {
                LearningEventType::EdgePotentiated => 2.0,
                LearningEventType::InterferenceDetected => 1.5,
                LearningEventType::FactExtracted => 1.5,
                _ => 1.0,
            };
            weighted_sum += weight * type_weight;
        }

        // Normalize to events per day
        let velocity_score = weighted_sum * (24.0 / window_hours as f32);

        Ok(LearningVelocity {
            event_count,
            event_types,
            last_event,
            velocity_score,
        })
    }

    /// Calculate learning velocity for a fact
    pub fn fact_learning_velocity(
        &self,
        user_id: &str,
        fact_id: &str,
        window_days: i64,
    ) -> Result<LearningVelocity> {
        let since = Utc::now() - Duration::days(window_days);
        let events = self.events_for_fact(user_id, fact_id, 100)?;

        let recent_events: Vec<_> = events
            .into_iter()
            .filter(|e| e.event.timestamp() >= since)
            .collect();

        if recent_events.is_empty() {
            return Ok(LearningVelocity::default());
        }

        let event_count = recent_events.len();
        let event_types: Vec<_> = recent_events.iter().map(|e| e.event_type).collect();
        let last_event = recent_events.iter().map(|e| e.event.timestamp()).max();

        // Reinforcement events are the key signal for facts
        let reinforcement_count = recent_events
            .iter()
            .filter(|e| e.event_type == LearningEventType::FactReinforced)
            .count();

        // Velocity = reinforcements per day
        let velocity_score = reinforcement_count as f32 / window_days as f32;

        Ok(LearningVelocity {
            event_count,
            event_types,
            last_event,
            velocity_score,
        })
    }

    /// Get aggregated learning statistics
    pub fn stats(&self, user_id: &str) -> Result<LearningStats> {
        let now = Utc::now();
        let day_ago = now - Duration::hours(24);
        let week_ago = now - Duration::days(7);

        // Get all events (could be optimized with counters)
        let all_events = self.events_since(user_id, now - Duration::days(365))?;

        let mut stats = LearningStats::default();
        stats.total_events = all_events.len();

        let mut memory_event_counts: HashMap<String, usize> = HashMap::new();

        for event in &all_events {
            // Count by type
            *stats
                .events_by_type
                .entry(event.event_type.as_str().to_string())
                .or_insert(0) += 1;

            // Count recent
            let ts = event.event.timestamp();
            if ts >= day_ago {
                stats.events_last_24h += 1;
            }
            if ts >= week_ago {
                stats.events_last_7d += 1;
            }

            // Track memory activity
            if let Some(ref mem_id) = event.memory_id {
                *memory_event_counts.entry(mem_id.clone()).or_insert(0) += 1;
            }

            // Count specific types
            match event.event_type {
                LearningEventType::EdgePotentiated => stats.potentiation_count += 1,
                LearningEventType::InterferenceDetected => stats.interference_count += 1,
                _ => {}
            }
        }

        // Find most active memories
        let mut memory_counts: Vec<_> = memory_event_counts.into_iter().collect();
        memory_counts.sort_by(|a, b| b.1.cmp(&a.1));
        stats.most_active_memories = memory_counts.into_iter().take(10).collect();

        Ok(stats)
    }

    /// Check if a memory has recent learning activity
    ///
    /// Used for retrieval boosting - recently learned memories are more relevant
    pub fn has_recent_learning(&self, user_id: &str, memory_id: &str, hours: i64) -> Result<bool> {
        let velocity = self.memory_learning_velocity(user_id, memory_id, hours)?;
        Ok(velocity.event_count > 0)
    }

    /// Get the recency boost factor for a memory
    ///
    /// Returns a multiplier (1.0 = no boost, up to ~1.3 for very active learning)
    pub fn recency_boost(&self, user_id: &str, memory_id: &str) -> Result<f32> {
        // Check 24-hour window
        let velocity = self.memory_learning_velocity(user_id, memory_id, 24)?;

        // Base boost from having any recent learning
        let mut boost = 1.0;

        if velocity.event_count > 0 {
            // Base boost for any recent activity
            boost += 0.05;

            // Additional boost based on velocity (capped)
            boost += (velocity.velocity_score * 0.05).min(0.15);

            // Extra boost for potentiation (permanent learning)
            if velocity
                .event_types
                .contains(&LearningEventType::EdgePotentiated)
            {
                boost += 0.1;
            }
        }

        Ok(boost.min(1.3)) // Cap at 30% boost
    }

    /// Get recency boost for a fact
    pub fn fact_recency_boost(&self, user_id: &str, fact_id: &str) -> Result<f32> {
        let velocity = self.fact_learning_velocity(user_id, fact_id, 7)?;

        let mut boost = 1.0;

        if velocity.event_count > 0 {
            boost += 0.05;
            // Reinforcement velocity is key for facts
            boost += (velocity.velocity_score * 0.1).min(0.2);
        }

        Ok(boost.min(1.25))
    }

    /// Prune old learning history (keep last N days)
    pub fn prune_old_events(&self, user_id: &str, keep_days: i64) -> Result<usize> {
        let cutoff = Utc::now() - Duration::days(keep_days);
        let cutoff_nanos = cutoff.timestamp_nanos_opt().unwrap_or(0);

        let prefix = format!("learning:{}:", user_id);
        let mut deleted = 0;

        let iter = self.db.prefix_iterator(prefix.as_bytes());
        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            // Extract timestamp from key
            if let Some(ts_str) = key_str.strip_prefix(&prefix) {
                if let Ok(ts) = ts_str.parse::<i64>() {
                    if ts < cutoff_nanos {
                        self.db.delete(&key)?;
                        deleted += 1;
                    }
                }
            }
        }

        // Note: This doesn't clean up secondary indexes - could add that if needed

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store() -> (LearningHistoryStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());
        (LearningHistoryStore::new(db), temp_dir)
    }

    #[test]
    fn test_record_and_query() {
        let (store, _dir) = create_test_store();

        let event = ConsolidationEvent::EdgePotentiated {
            from_memory_id: "mem-1".to_string(),
            to_memory_id: "mem-2".to_string(),
            final_strength: 0.95,
            total_co_activations: 15,
            timestamp: Utc::now(),
        };

        store.record("user-1", &event).unwrap();

        let events = store
            .events_since("user-1", Utc::now() - Duration::hours(1))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, LearningEventType::EdgePotentiated);
    }

    #[test]
    fn test_memory_velocity() {
        let (store, _dir) = create_test_store();

        // Record several events for a memory
        for i in 0..5 {
            let event = ConsolidationEvent::MemoryReplayed {
                memory_id: "mem-1".to_string(),
                content_preview: "test".to_string(),
                activation_before: 0.5,
                activation_after: 0.55,
                replay_priority: 0.8,
                connected_memories_replayed: 2,
                timestamp: Utc::now() - Duration::hours(i),
            };
            store.record("user-1", &event).unwrap();
        }

        let velocity = store
            .memory_learning_velocity("user-1", "mem-1", 24)
            .unwrap();
        assert_eq!(velocity.event_count, 5);
        assert!(velocity.velocity_score > 0.0);
    }

    #[test]
    fn test_recency_boost() {
        let (store, _dir) = create_test_store();

        // No events = no boost
        let boost = store.recency_boost("user-1", "mem-1").unwrap();
        assert_eq!(boost, 1.0);

        // Add an event
        let event = ConsolidationEvent::MemoryReplayed {
            memory_id: "mem-1".to_string(),
            content_preview: "test".to_string(),
            activation_before: 0.5,
            activation_after: 0.55,
            replay_priority: 0.8,
            connected_memories_replayed: 2,
            timestamp: Utc::now(),
        };
        store.record("user-1", &event).unwrap();

        let boost = store.recency_boost("user-1", "mem-1").unwrap();
        assert!(boost > 1.0);
    }
}
