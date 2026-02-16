//! Memory System Visualization using Petgraph
//! Creates a real-time graph of memory connections like a neural network

#![allow(dead_code)]

use crate::memory::{ExperienceType, Memory, MemoryId};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use tracing::{debug, info, trace};

/// Node type in the memory graph
#[derive(Debug, Clone)]
pub enum MemoryNode {
    WorkingMemory {
        id: MemoryId,
        importance: f32,
    },
    SessionMemory {
        id: MemoryId,
        importance: f32,
    },
    LongTermMemory {
        id: MemoryId,
        importance: f32,
        compressed: bool,
    },
    Experience {
        exp_type: ExperienceType,
        content: String,
    },
    Context {
        context_id: String,
        decay: f32,
    },
}

impl fmt::Display for MemoryNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MemoryNode::WorkingMemory { id: _, importance } => {
                write!(f, "WM\\n{importance:.2}")
            }
            MemoryNode::SessionMemory { id: _, importance } => {
                write!(f, "SM\\n{importance:.2}")
            }
            MemoryNode::LongTermMemory {
                id: _,
                importance,
                compressed,
            } => {
                write!(
                    f,
                    "LTM\\n{:.2}{}",
                    importance,
                    if *compressed { "🗜️" } else { "" }
                )
            }
            MemoryNode::Experience { exp_type, .. } => {
                write!(f, "{exp_type:?}")
            }
            MemoryNode::Context {
                context_id: _,
                decay,
            } => {
                write!(f, "CTX\\n{decay:.2}")
            }
        }
    }
}

/// Edge type in the memory graph
#[derive(Debug, Clone)]
pub enum MemoryEdge {
    Promotion,               // Working -> Session -> LongTerm
    SemanticSimilarity(f32), // Similarity score
    TemporalSuccession,      // A happened after B
    CausalLink,              // A caused B
    ContextRelation,         // Related through context
}

impl fmt::Display for MemoryEdge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MemoryEdge::Promotion => write!(f, "→"),
            MemoryEdge::SemanticSimilarity(score) => write!(f, "~{score:.2}"),
            MemoryEdge::TemporalSuccession => write!(f, "⏭"),
            MemoryEdge::CausalLink => write!(f, "⚡"),
            MemoryEdge::ContextRelation => write!(f, "⊕"),
        }
    }
}

/// Memory visualization graph
pub struct MemoryGraph {
    graph: DiGraph<MemoryNode, MemoryEdge>,
    node_map: HashMap<String, NodeIndex>,
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Add a memory to the graph
    pub fn add_memory(&mut self, memory: &Memory, tier: &str) -> NodeIndex {
        let key = format!("{}_{}", tier, memory.id.0);

        if let Some(&idx) = self.node_map.get(&key) {
            return idx;
        }

        let node = match tier {
            "working" => MemoryNode::WorkingMemory {
                id: memory.id.clone(),
                importance: memory.importance(),
            },
            "session" => MemoryNode::SessionMemory {
                id: memory.id.clone(),
                importance: memory.importance(),
            },
            "longterm" => MemoryNode::LongTermMemory {
                id: memory.id.clone(),
                importance: memory.importance(),
                compressed: memory.compressed,
            },
            _ => {
                tracing::error!(
                    "Invalid tier '{}' passed to add_memory for memory {}, defaulting to WorkingMemory",
                    tier,
                    memory.id.0
                );
                MemoryNode::WorkingMemory {
                    id: memory.id.clone(),
                    importance: memory.importance(),
                }
            }
        };

        let idx = self.graph.add_node(node);
        self.node_map.insert(key, idx);
        idx
    }

    /// Add an experience node
    pub fn add_experience(&mut self, exp_type: ExperienceType, content: &str) -> NodeIndex {
        let node = MemoryNode::Experience {
            exp_type,
            content: content.chars().take(50).collect(),
        };
        self.graph.add_node(node)
    }

    /// Add a context node
    pub fn add_context(&mut self, context_id: &str, decay: f32) -> NodeIndex {
        let node = MemoryNode::Context {
            context_id: context_id.to_string(),
            decay,
        };
        self.graph.add_node(node)
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge_type: MemoryEdge) {
        self.graph.add_edge(from, to, edge_type);
    }

    /// Visualize memory promotion (working -> session -> longterm)
    pub fn log_promotion(&mut self, from_tier: &str, to_tier: &str, memory_id: &MemoryId) {
        let from_key = format!("{}_{}", from_tier, memory_id.0);
        let to_key = format!("{}_{}", to_tier, memory_id.0);

        if let (Some(&from_idx), Some(&to_idx)) =
            (self.node_map.get(&from_key), self.node_map.get(&to_key))
        {
            self.add_edge(from_idx, to_idx, MemoryEdge::Promotion);
            debug!(
                from = from_tier.to_uppercase().as_str(),
                to = to_tier.to_uppercase().as_str(),
                "Graph tier promotion"
            );
        }
    }

    /// Export graph as DOT format for Graphviz
    pub fn to_dot(&self) -> String {
        format!(
            "{:?}",
            Dot::with_config(&self.graph, &[Config::EdgeNoLabel])
        )
    }

    /// Get statistics about the graph
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            total_nodes: self.graph.node_count(),
            total_edges: self.graph.edge_count(),
            working_memory_count: self.count_tier("working"),
            session_memory_count: self.count_tier("session"),
            longterm_memory_count: self.count_tier("longterm"),
        }
    }

    fn count_tier(&self, tier: &str) -> usize {
        self.node_map
            .keys()
            .filter(|k| k.starts_with(&format!("{tier}_")))
            .count()
    }

    /// Print ASCII visualization of current memory state
    pub fn print_ascii_visualization(&self) {
        let stats = self.stats();

        info!(
            working = stats.working_memory_count,
            session = stats.session_memory_count,
            longterm = stats.longterm_memory_count,
            nodes = stats.total_nodes,
            edges = stats.total_edges,
            "Memory system visualization: working={}, session={}, longterm={}, nodes={}, edges={}",
            stats.working_memory_count,
            stats.session_memory_count,
            stats.longterm_memory_count,
            stats.total_nodes,
            stats.total_edges,
        );
    }
}

/// Graph statistics
#[derive(Debug, Clone, Serialize)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub working_memory_count: usize,
    pub session_memory_count: usize,
    pub longterm_memory_count: usize,
}

/// Logger for memory operations
pub struct MemoryLogger {
    pub graph: MemoryGraph,
    enabled: bool,
}

impl MemoryLogger {
    pub fn new(enabled: bool) -> Self {
        Self {
            graph: MemoryGraph::new(),
            enabled,
        }
    }

    /// Log memory creation
    pub fn log_created(&mut self, memory: &Memory, tier: &str) {
        if !self.enabled {
            return;
        }

        debug!(
            tier = tier.to_uppercase().as_str(),
            importance = memory.importance(),
            experience_type = ?memory.experience.experience_type,
            "Memory created"
        );

        self.graph.add_memory(memory, tier);
    }

    /// Log memory access
    pub fn log_accessed(&self, memory_id: &MemoryId, tier: &str) {
        if !self.enabled {
            return;
        }

        trace!(
            tier = tier.to_uppercase().as_str(),
            memory_id = %memory_id.0,
            "Memory accessed"
        );
    }

    /// Log memory promotion
    pub fn log_promoted(&mut self, memory_id: &MemoryId, from: &str, to: &str, count: usize) {
        if !self.enabled {
            return;
        }

        debug!(
            from = from.to_uppercase().as_str(),
            to = to.to_uppercase().as_str(),
            count,
            "Memory tier promotion"
        );

        self.graph.log_promotion(from, to, memory_id);
    }

    /// Log compression
    pub fn log_compressed(
        &self,
        _memory_id: &MemoryId,
        original_size: usize,
        compressed_size: usize,
    ) {
        if !self.enabled {
            return;
        }

        let ratio = (compressed_size as f32 / original_size as f32 * 100.0) as usize;
        debug!(original_size, compressed_size, ratio, "Memory compressed");
    }

    /// Log retrieval
    pub fn log_retrieved(&self, query: &str, result_count: usize, sources: &[&str]) {
        if !self.enabled {
            return;
        }

        debug!(
            query = %query.chars().take(50).collect::<String>(),
            result_count,
            sources = %sources.join(", "),
            "Memory retrieved"
        );
    }

    /// Show visualization
    pub fn show_visualization(&self) {
        if !self.enabled {
            return;
        }

        self.graph.print_ascii_visualization();
    }

    /// Export graph
    pub fn export_dot(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let dot = self.graph.to_dot();
        std::fs::write(path, dot)?;
        info!(path = %path.display(), "Graph exported");
        Ok(())
    }

    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        self.graph.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Experience, ExperienceType, Memory, MemoryId};

    fn create_test_memory() -> Memory {
        use uuid::Uuid;

        let experience = Experience {
            experience_type: ExperienceType::Conversation,
            content: "test content".to_string(),
            ..Default::default()
        };

        Memory::new(
            MemoryId(Uuid::new_v4()),
            experience,
            0.5,  // importance
            None, // agent_id
            None, // run_id
            None, // actor_id
            None, // created_at
        )
    }

    #[test]
    fn test_add_memory_with_valid_tiers() {
        let mut graph = MemoryGraph::new();
        let memory = create_test_memory();

        // Test all valid tier names
        let idx1 = graph.add_memory(&memory, "working");
        let idx2 = graph.add_memory(&memory, "session");
        let idx3 = graph.add_memory(&memory, "longterm");

        // Verify nodes were created
        assert_eq!(graph.graph.node_count(), 3);
        assert!(idx1 != idx2 && idx2 != idx3 && idx1 != idx3);
    }

    #[test]
    fn test_add_memory_with_invalid_tier_does_not_panic() {
        let mut graph = MemoryGraph::new();
        let memory = create_test_memory();

        // This should NOT panic - it should log error and default to WorkingMemory
        let idx = graph.add_memory(&memory, "invalid_tier_name");

        // Verify node was created despite invalid tier
        assert_eq!(graph.graph.node_count(), 1);

        // Verify the node exists
        assert!(graph.graph.node_weight(idx).is_some());

        // Verify it was added as WorkingMemory (default fallback)
        let node = graph.graph.node_weight(idx).unwrap();
        match node {
            MemoryNode::WorkingMemory { .. } => {
                // Success - defaulted to WorkingMemory
            }
            _ => panic!("Expected WorkingMemory for invalid tier, got {node:?}"),
        }
    }

    #[test]
    fn test_add_memory_with_various_invalid_tiers() {
        let mut graph = MemoryGraph::new();
        let memory = create_test_memory();

        // Test various invalid tier names - none should panic
        let invalid_tiers = vec![
            "",
            "Working",
            "WORKING",
            "long-term",
            "unknown",
            "123",
            "session_memory",
        ];

        for tier in invalid_tiers {
            let _ = graph.add_memory(&memory, tier);
        }

        // All should have been created as WorkingMemory
        assert_eq!(graph.graph.node_count(), 7);
    }
}
