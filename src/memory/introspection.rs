//! Memory Consolidation Introspection
//!
//! Provides visibility into what the memory system is learning:
//! - Which memories are strengthening/decaying
//! - What associations are forming
//! - When consolidation events occur
//!
//! This makes the "brain" transparent rather than a black box.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;


/// Maximum number of events to keep in the event buffer
const MAX_EVENT_BUFFER_SIZE: usize = 1000;

/// Types of consolidation events that can occur
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConsolidationEvent {
    /// Memory activation was strengthened (accessed/boosted)
    MemoryStrengthened {
        memory_id: String,
        content_preview: String,
        activation_before: f32,
        activation_after: f32,
        reason: StrengtheningReason,
        timestamp: DateTime<Utc>,
    },

    /// Memory activation decayed during maintenance
    MemoryDecayed {
        memory_id: String,
        content_preview: String,
        activation_before: f32,
        activation_after: f32,
        at_risk: bool, // Below threshold soon
        timestamp: DateTime<Utc>,
    },

    /// Hebbian edge was formed between memories
    EdgeFormed {
        from_memory_id: String,
        to_memory_id: String,
        initial_strength: f32,
        reason: EdgeFormationReason,
        timestamp: DateTime<Utc>,
    },

    /// Hebbian edge was strengthened (co-activation)
    EdgeStrengthened {
        from_memory_id: String,
        to_memory_id: String,
        strength_before: f32,
        strength_after: f32,
        co_activations: u32,
        timestamp: DateTime<Utc>,
    },

    /// Edge became potentiated (permanent through LTP)
    EdgePotentiated {
        from_memory_id: String,
        to_memory_id: String,
        final_strength: f32,
        total_co_activations: u32,
        timestamp: DateTime<Utc>,
    },

    /// Edge was pruned (decayed below threshold)
    EdgePruned {
        from_memory_id: String,
        to_memory_id: String,
        final_strength: f32,
        reason: PruningReason,
        timestamp: DateTime<Utc>,
    },

    /// Semantic fact was extracted from episodic memories
    FactExtracted {
        fact_id: String,
        fact_content: String,
        confidence: f32,
        source_memory_count: usize,
        fact_type: String,
        timestamp: DateTime<Utc>,
    },

    /// Existing fact was reinforced with new evidence
    FactReinforced {
        fact_id: String,
        fact_content: String,
        confidence_before: f32,
        confidence_after: f32,
        new_support_count: usize,
        timestamp: DateTime<Utc>,
    },

    /// Memory was promoted to a higher tier
    MemoryPromoted {
        memory_id: String,
        content_preview: String,
        from_tier: String,
        to_tier: String,
        timestamp: DateTime<Utc>,
    },

    /// Maintenance cycle completed
    MaintenanceCycleCompleted {
        memories_processed: usize,
        memories_decayed: usize,
        edges_pruned: usize,
        duration_ms: u64,
        timestamp: DateTime<Utc>,
    },
}

/// Reasons for memory strengthening
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrengtheningReason {
    /// Accessed during recall
    Recalled,
    /// Part of spreading activation
    SpreadingActivation,
    /// Explicitly boosted by user
    ExplicitBoost,
    /// Co-retrieved with another memory
    CoRetrieval,
}

/// Reasons for edge formation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeFormationReason {
    /// Co-retrieved during recall
    CoRetrieval,
    /// Shared entities
    SharedEntities,
    /// Semantic similarity above threshold
    SemanticSimilarity,
    /// Temporal proximity
    TemporalProximity,
}

/// Reasons for edge pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PruningReason {
    /// Decayed below minimum threshold
    DecayedBelowThreshold,
    /// Not accessed for too long
    Inactivity,
    /// Explicitly invalidated
    Invalidated,
}

/// Aggregated consolidation report for a time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Time period covered
    pub period: ReportPeriod,

    /// Memories that got stronger
    pub strengthened_memories: Vec<MemoryChange>,

    /// Memories that decayed
    pub decayed_memories: Vec<MemoryChange>,

    /// New associations formed
    pub formed_associations: Vec<AssociationChange>,

    /// Associations that got stronger
    pub strengthened_associations: Vec<AssociationChange>,

    /// Associations that became permanent (LTP)
    pub potentiated_associations: Vec<AssociationChange>,

    /// Associations that were pruned
    pub pruned_associations: Vec<AssociationChange>,

    /// Facts extracted from episodic memories
    pub extracted_facts: Vec<FactChange>,

    /// Facts that were reinforced
    pub reinforced_facts: Vec<FactChange>,

    /// Aggregate statistics
    pub statistics: ConsolidationStats,
}

/// Time period for a report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportPeriod {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Change in a memory's state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChange {
    pub memory_id: String,
    pub content_preview: String,
    pub activation_before: f32,
    pub activation_after: f32,
    pub change_reason: String,
    pub at_risk: bool,
    pub timestamp: DateTime<Utc>,
}

/// Change in an association/edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationChange {
    pub from_memory_id: String,
    pub to_memory_id: String,
    pub strength_before: Option<f32>,
    pub strength_after: f32,
    pub co_activations: Option<u32>,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Change in a semantic fact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactChange {
    pub fact_id: String,
    pub fact_content: String,
    pub confidence: f32,
    pub support_count: usize,
    pub fact_type: String,
    pub timestamp: DateTime<Utc>,
}

/// Aggregate statistics for consolidation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationStats {
    pub total_memories: usize,
    pub memories_strengthened: usize,
    pub memories_decayed: usize,
    pub memories_at_risk: usize,
    pub edges_formed: usize,
    pub edges_strengthened: usize,
    pub edges_potentiated: usize,
    pub edges_pruned: usize,
    pub facts_extracted: usize,
    pub facts_reinforced: usize,
    pub maintenance_cycles: usize,
    pub total_maintenance_duration_ms: u64,
}

/// Buffer for storing consolidation events
#[derive(Debug, Default)]
pub struct ConsolidationEventBuffer {
    events: VecDeque<ConsolidationEvent>,
    max_size: usize,
}

impl ConsolidationEventBuffer {
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            max_size: MAX_EVENT_BUFFER_SIZE,
        }
    }

    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Push a new event, evicting oldest if at capacity
    pub fn push(&mut self, event: ConsolidationEvent) {
        if self.events.len() >= self.max_size {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Get all events since a given timestamp
    pub fn events_since(&self, since: DateTime<Utc>) -> Vec<ConsolidationEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp() >= since)
            .cloned()
            .collect()
    }

    /// Get all events
    pub fn all_events(&self) -> Vec<ConsolidationEvent> {
        self.events.iter().cloned().collect()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Number of events in buffer
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Generate a report from events in a time period
    pub fn generate_report(&self, since: DateTime<Utc>, until: DateTime<Utc>) -> ConsolidationReport {
        let events: Vec<_> = self
            .events
            .iter()
            .filter(|e| {
                let ts = e.timestamp();
                ts >= since && ts <= until
            })
            .collect();

        let mut report = ConsolidationReport {
            period: ReportPeriod {
                start: since,
                end: until,
            },
            strengthened_memories: Vec::new(),
            decayed_memories: Vec::new(),
            formed_associations: Vec::new(),
            strengthened_associations: Vec::new(),
            potentiated_associations: Vec::new(),
            pruned_associations: Vec::new(),
            extracted_facts: Vec::new(),
            reinforced_facts: Vec::new(),
            statistics: ConsolidationStats::default(),
        };

        for event in events {
            match event {
                ConsolidationEvent::MemoryStrengthened {
                    memory_id,
                    content_preview,
                    activation_before,
                    activation_after,
                    reason,
                    timestamp,
                } => {
                    report.strengthened_memories.push(MemoryChange {
                        memory_id: memory_id.clone(),
                        content_preview: content_preview.clone(),
                        activation_before: *activation_before,
                        activation_after: *activation_after,
                        change_reason: format!("{:?}", reason),
                        at_risk: false,
                        timestamp: *timestamp,
                    });
                    report.statistics.memories_strengthened += 1;
                }

                ConsolidationEvent::MemoryDecayed {
                    memory_id,
                    content_preview,
                    activation_before,
                    activation_after,
                    at_risk,
                    timestamp,
                } => {
                    report.decayed_memories.push(MemoryChange {
                        memory_id: memory_id.clone(),
                        content_preview: content_preview.clone(),
                        activation_before: *activation_before,
                        activation_after: *activation_after,
                        change_reason: "decay".to_string(),
                        at_risk: *at_risk,
                        timestamp: *timestamp,
                    });
                    report.statistics.memories_decayed += 1;
                    if *at_risk {
                        report.statistics.memories_at_risk += 1;
                    }
                }

                ConsolidationEvent::EdgeFormed {
                    from_memory_id,
                    to_memory_id,
                    initial_strength,
                    reason,
                    timestamp,
                } => {
                    report.formed_associations.push(AssociationChange {
                        from_memory_id: from_memory_id.clone(),
                        to_memory_id: to_memory_id.clone(),
                        strength_before: None,
                        strength_after: *initial_strength,
                        co_activations: Some(1),
                        reason: format!("{:?}", reason),
                        timestamp: *timestamp,
                    });
                    report.statistics.edges_formed += 1;
                }

                ConsolidationEvent::EdgeStrengthened {
                    from_memory_id,
                    to_memory_id,
                    strength_before,
                    strength_after,
                    co_activations,
                    timestamp,
                } => {
                    report.strengthened_associations.push(AssociationChange {
                        from_memory_id: from_memory_id.clone(),
                        to_memory_id: to_memory_id.clone(),
                        strength_before: Some(*strength_before),
                        strength_after: *strength_after,
                        co_activations: Some(*co_activations),
                        reason: "co_activation".to_string(),
                        timestamp: *timestamp,
                    });
                    report.statistics.edges_strengthened += 1;
                }

                ConsolidationEvent::EdgePotentiated {
                    from_memory_id,
                    to_memory_id,
                    final_strength,
                    total_co_activations,
                    timestamp,
                } => {
                    report.potentiated_associations.push(AssociationChange {
                        from_memory_id: from_memory_id.clone(),
                        to_memory_id: to_memory_id.clone(),
                        strength_before: None,
                        strength_after: *final_strength,
                        co_activations: Some(*total_co_activations),
                        reason: "long_term_potentiation".to_string(),
                        timestamp: *timestamp,
                    });
                    report.statistics.edges_potentiated += 1;
                }

                ConsolidationEvent::EdgePruned {
                    from_memory_id,
                    to_memory_id,
                    final_strength,
                    reason,
                    timestamp,
                } => {
                    report.pruned_associations.push(AssociationChange {
                        from_memory_id: from_memory_id.clone(),
                        to_memory_id: to_memory_id.clone(),
                        strength_before: Some(*final_strength),
                        strength_after: 0.0,
                        co_activations: None,
                        reason: format!("{:?}", reason),
                        timestamp: *timestamp,
                    });
                    report.statistics.edges_pruned += 1;
                }

                ConsolidationEvent::FactExtracted {
                    fact_id,
                    fact_content,
                    confidence,
                    source_memory_count,
                    fact_type,
                    timestamp,
                } => {
                    report.extracted_facts.push(FactChange {
                        fact_id: fact_id.clone(),
                        fact_content: fact_content.clone(),
                        confidence: *confidence,
                        support_count: *source_memory_count,
                        fact_type: fact_type.clone(),
                        timestamp: *timestamp,
                    });
                    report.statistics.facts_extracted += 1;
                }

                ConsolidationEvent::FactReinforced {
                    fact_id,
                    fact_content,
                    confidence_after,
                    new_support_count,
                    timestamp,
                    ..
                } => {
                    report.reinforced_facts.push(FactChange {
                        fact_id: fact_id.clone(),
                        fact_content: fact_content.clone(),
                        confidence: *confidence_after,
                        support_count: *new_support_count,
                        fact_type: "reinforced".to_string(),
                        timestamp: *timestamp,
                    });
                    report.statistics.facts_reinforced += 1;
                }

                ConsolidationEvent::MemoryPromoted { .. } => {
                    // Track promotions if needed
                }

                ConsolidationEvent::MaintenanceCycleCompleted {
                    duration_ms,
                    ..
                } => {
                    report.statistics.maintenance_cycles += 1;
                    report.statistics.total_maintenance_duration_ms += duration_ms;
                }
            }
        }

        report
    }
}

impl ConsolidationEvent {
    /// Get the timestamp of this event
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            ConsolidationEvent::MemoryStrengthened { timestamp, .. } => *timestamp,
            ConsolidationEvent::MemoryDecayed { timestamp, .. } => *timestamp,
            ConsolidationEvent::EdgeFormed { timestamp, .. } => *timestamp,
            ConsolidationEvent::EdgeStrengthened { timestamp, .. } => *timestamp,
            ConsolidationEvent::EdgePotentiated { timestamp, .. } => *timestamp,
            ConsolidationEvent::EdgePruned { timestamp, .. } => *timestamp,
            ConsolidationEvent::FactExtracted { timestamp, .. } => *timestamp,
            ConsolidationEvent::FactReinforced { timestamp, .. } => *timestamp,
            ConsolidationEvent::MemoryPromoted { timestamp, .. } => *timestamp,
            ConsolidationEvent::MaintenanceCycleCompleted { timestamp, .. } => *timestamp,
        }
    }
}

impl Default for ConsolidationReport {
    fn default() -> Self {
        Self {
            period: ReportPeriod {
                start: Utc::now(),
                end: Utc::now(),
            },
            strengthened_memories: Vec::new(),
            decayed_memories: Vec::new(),
            formed_associations: Vec::new(),
            strengthened_associations: Vec::new(),
            potentiated_associations: Vec::new(),
            pruned_associations: Vec::new(),
            extracted_facts: Vec::new(),
            reinforced_facts: Vec::new(),
            statistics: ConsolidationStats::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_buffer_push() {
        let mut buffer = ConsolidationEventBuffer::with_capacity(3);

        for i in 0..5 {
            buffer.push(ConsolidationEvent::MemoryDecayed {
                memory_id: format!("mem-{}", i),
                content_preview: format!("Memory {}", i),
                activation_before: 0.5,
                activation_after: 0.4,
                at_risk: false,
                timestamp: Utc::now(),
            });
        }

        // Should only keep last 3
        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_generate_report() {
        let mut buffer = ConsolidationEventBuffer::new();
        let now = Utc::now();

        buffer.push(ConsolidationEvent::MemoryStrengthened {
            memory_id: "mem-1".to_string(),
            content_preview: "Test memory".to_string(),
            activation_before: 0.5,
            activation_after: 0.7,
            reason: StrengtheningReason::Recalled,
            timestamp: now,
        });

        buffer.push(ConsolidationEvent::EdgeFormed {
            from_memory_id: "mem-1".to_string(),
            to_memory_id: "mem-2".to_string(),
            initial_strength: 0.5,
            reason: EdgeFormationReason::CoRetrieval,
            timestamp: now,
        });

        let report = buffer.generate_report(
            now - chrono::Duration::hours(1),
            now + chrono::Duration::hours(1),
        );

        assert_eq!(report.statistics.memories_strengthened, 1);
        assert_eq!(report.statistics.edges_formed, 1);
    }
}
