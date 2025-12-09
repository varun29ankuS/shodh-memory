//! Graph-Aware Retrieval using Spreading Activation
//!
//! Based on:
//! - Anderson & Pirolli (1984): "Spread of Activation"
//! - Xiong et al. (2017): "Explicit Semantic Ranking via Knowledge Graph Embedding"
//!
//! Implements spreading activation algorithm for memory retrieval:
//! 1. Extract entities from query (using linguistic analysis)
//! 2. Activate entities in knowledge graph
//! 3. Spread activation through graph relationships
//! 4. Retrieve episodic memories connected to activated entities
//! 5. Score using hybrid method (graph + semantic + linguistic)

use anyhow::Result;
use std::collections::HashMap;
use uuid::Uuid;

use crate::constants::{
    HYBRID_GRAPH_WEIGHT, HYBRID_LINGUISTIC_WEIGHT, HYBRID_SEMANTIC_WEIGHT,
    SPREADING_ACTIVATION_THRESHOLD, SPREADING_DECAY_RATE, SPREADING_MAX_HOPS,
};
use crate::embeddings::Embedder;
use crate::graph_memory::{EpisodicNode, GraphMemory};
use crate::memory::query_parser::{analyze_query, QueryAnalysis};
use crate::memory::types::{Memory, Query, SharedMemory};

/// Memory with activation score
#[derive(Debug, Clone)]
pub struct ActivatedMemory {
    pub memory: SharedMemory,
    #[allow(dead_code)] // Useful for debugging score breakdown
    pub activation_score: f32,
    #[allow(dead_code)] // Useful for debugging score breakdown
    pub semantic_score: f32,
    #[allow(dead_code)] // Useful for debugging score breakdown
    pub linguistic_score: f32,
    pub final_score: f32,
}

/// Spreading activation retrieval
///
/// This is the core algorithm implementing Anderson & Pirolli (1984)
/// spreading activation model adapted for episodic memory retrieval.
pub fn spreading_activation_retrieve(
    query_text: &str,
    query: &Query,
    graph: &GraphMemory,
    embedder: &dyn Embedder,
    episode_to_memory_fn: impl Fn(&EpisodicNode) -> Result<Option<SharedMemory>>,
) -> Result<Vec<ActivatedMemory>> {
    // Step 1: Linguistic query analysis (Lioma & Ounis 2006)
    let analysis = analyze_query(query_text);

    tracing::info!("🔍 Query Analysis:");
    tracing::info!(
        "  Focal Entities: {:?}",
        analysis
            .focal_entities
            .iter()
            .map(|e| &e.text)
            .collect::<Vec<_>>()
    );
    tracing::info!(
        "  Modifiers: {:?}",
        analysis
            .discriminative_modifiers
            .iter()
            .map(|m| &m.text)
            .collect::<Vec<_>>()
    );
    tracing::info!(
        "  Relations: {:?}",
        analysis
            .relational_context
            .iter()
            .map(|r| &r.text)
            .collect::<Vec<_>>()
    );

    // Step 2: Initialize activation map from focal entities (nouns)
    let mut activation_map: HashMap<Uuid, f32> = HashMap::new();

    for entity in &analysis.focal_entities {
        // Find entity in graph by name
        if let Some(entity_node) = graph.find_entity_by_name(&entity.text)? {
            // Initial activation = IC weight (2.3 for nouns)
            activation_map.insert(entity_node.uuid, entity.ic_weight);

            tracing::debug!(
                "  ✓ Activated entity '{}' (UUID: {}, IC: {})",
                entity.text,
                entity_node.uuid,
                entity.ic_weight
            );
        } else {
            tracing::debug!("  ✗ Entity '{}' not found in graph", entity.text);
        }
    }

    if activation_map.is_empty() {
        tracing::warn!("No entities found in graph, falling back to semantic search");
        return Ok(Vec::new()); // Caller should fall back to semantic search
    }

    // Step 3: Spread activation through graph (Anderson & Pirolli 1984)
    // Formula: A(d) = A₀ × e^(-λd)
    for hop in 1..=SPREADING_MAX_HOPS {
        let decay = (-SPREADING_DECAY_RATE * hop as f32).exp();

        tracing::debug!(
            "📊 Spreading activation (hop {}/{}), decay factor: {:.3}",
            hop,
            SPREADING_MAX_HOPS,
            decay
        );

        // Clone to avoid borrow issues
        let current_activated: Vec<(Uuid, f32)> =
            activation_map.iter().map(|(id, act)| (*id, *act)).collect();

        for (entity_uuid, source_activation) in current_activated {
            // Only spread from entities with sufficient activation
            if source_activation < SPREADING_ACTIVATION_THRESHOLD {
                continue;
            }

            // Get relationships from this entity
            let edges = graph.get_entity_relationships(&entity_uuid)?;

            for edge in edges {
                // Spread activation to connected entity
                let target_uuid = edge.to_entity;
                let spread_amount = source_activation * decay * edge.strength;

                // Accumulate activation
                *activation_map.entry(target_uuid).or_insert(0.0) += spread_amount;
            }
        }

        // Prune weak activations (ACT-R model)
        activation_map.retain(|_, activation| *activation > SPREADING_ACTIVATION_THRESHOLD);

        tracing::debug!("  Activated entities: {}", activation_map.len());
    }

    tracing::info!("📊 Final activated entities: {}", activation_map.len());

    // Step 4: Retrieve episodic memories connected to activated entities
    let mut activated_memories: HashMap<Uuid, (f32, EpisodicNode)> = HashMap::new();

    for (entity_uuid, entity_activation) in &activation_map {
        let episodes = graph.get_episodes_by_entity(entity_uuid)?;

        for episode in episodes {
            // Accumulate activation for each episode (might be connected to multiple entities)
            let current = activated_memories
                .entry(episode.uuid)
                .or_insert((0.0, episode.clone()));

            current.0 += entity_activation;
        }
    }

    tracing::info!(
        "📊 Retrieved {} episodic memories",
        activated_memories.len()
    );

    // Step 5: Convert episodes to memories and calculate scores
    let mut scored_memories = Vec::new();

    // Generate query embedding once (for semantic scoring)
    let query_embedding = embedder.encode(query_text)?;

    for (_episode_uuid, (graph_activation, episode)) in activated_memories {
        // Convert episode to memory
        if let Some(memory) = episode_to_memory_fn(&episode)? {
            // Calculate semantic similarity
            let semantic_score = if let Some(mem_emb) = &memory.experience.embeddings {
                cosine_similarity(&query_embedding, mem_emb)
            } else {
                0.0
            };

            // Calculate linguistic match score
            let linguistic_score = calculate_linguistic_match(&memory, &analysis);

            // Hybrid scoring (adjusted for semantic-first retrieval)
            // Weights from constants: Semantic 50%, Graph 35%, Linguistic 15%
            // Semantic similarity is primary for content matching
            // Graph activation helps with context-related memories
            let final_score = HYBRID_GRAPH_WEIGHT * graph_activation
                + HYBRID_SEMANTIC_WEIGHT * semantic_score
                + HYBRID_LINGUISTIC_WEIGHT * linguistic_score;

            scored_memories.push(ActivatedMemory {
                memory,
                activation_score: graph_activation,
                semantic_score,
                linguistic_score,
                final_score,
            });
        }
    }

    // Step 6: Sort by final score (descending)
    scored_memories.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 7: Apply limit
    scored_memories.truncate(query.max_results);

    tracing::info!(
        "🎯 Returning {} memories (top scores: {:?})",
        scored_memories.len(),
        scored_memories
            .iter()
            .take(3)
            .map(|m| m.final_score)
            .collect::<Vec<_>>()
    );

    Ok(scored_memories)
}

/// Calculate cosine similarity between two embeddings
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    (dot_product / (magnitude_a * magnitude_b)).clamp(0.0, 1.0)
}

/// Calculate linguistic feature match score
///
/// Based on IC-weighted matching:
/// - Focal entities (nouns): 1.0 point each
/// - Modifiers (adjectives): 0.5 points each
/// - Relations (verbs): 0.2 points each
fn calculate_linguistic_match(memory: &Memory, analysis: &QueryAnalysis) -> f32 {
    let content_lower = memory.experience.content.to_lowercase();
    let mut score = 0.0;

    // Entity matches (nouns) - highest weight
    for entity in &analysis.focal_entities {
        if content_lower.contains(&entity.text.to_lowercase()) {
            score += 1.0;
        }
    }

    // Modifier matches (adjectives) - medium weight
    for modifier in &analysis.discriminative_modifiers {
        if content_lower.contains(&modifier.text.to_lowercase()) {
            score += 0.5;
        }
    }

    // Relation matches (verbs) - low weight (they're "bus stops")
    for relation in &analysis.relational_context {
        if content_lower.contains(&relation.text.to_lowercase()) {
            score += 0.2;
        }
    }

    // Normalize by total possible score
    let max_possible = analysis.focal_entities.len() as f32 * 1.0
        + analysis.discriminative_modifiers.len() as f32 * 0.5
        + analysis.relational_context.len() as f32 * 0.2;

    if max_possible > 0.0 {
        score / max_possible
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_activation_decay() {
        use crate::constants::{
            SPREADING_ACTIVATION_THRESHOLD, SPREADING_DECAY_RATE, SPREADING_MAX_HOPS,
        };

        // Test decay formula: A(d) = A₀ × e^(-λd)
        let initial_activation = 1.0;

        for hop in 1..=SPREADING_MAX_HOPS {
            let decay = (-SPREADING_DECAY_RATE * hop as f32).exp();
            let activation = initial_activation * decay;

            // Activation should decrease with each hop
            assert!(activation < initial_activation);
            assert!(activation > SPREADING_ACTIVATION_THRESHOLD);
        }
    }
}
