//! Graph-Aware Retrieval using Spreading Activation
//!
//! Based on:
//! - Anderson & Pirolli (1984): "Spread of Activation"
//! - Xiong et al. (2017): "Explicit Semantic Ranking via Knowledge Graph Embedding"
//! - GraphRAG Survey (arXiv 2408.08921): Hybrid KG-Vector improves 13.1%
//! - spreadr R package (Siew, 2019): Importance-weighted decay
//!
//! Implements spreading activation algorithm for memory retrieval:
//! 1. Extract entities from query (using linguistic analysis)
//! 2. Activate entities in knowledge graph
//! 3. Spread activation through graph relationships (importance-weighted decay)
//! 4. Retrieve episodic memories connected to activated entities
//! 5. Score using hybrid method (density-dependent graph + semantic + linguistic)
//!
//! SHO-26 Enhancements:
//! - Density-dependent hybrid weights: Graph trust scales with learned associations
//! - Importance-weighted decay: Important memories decay slower, preserve signal

use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

use crate::constants::{
    DENSITY_GRAPH_WEIGHT_MAX, DENSITY_GRAPH_WEIGHT_MIN, DENSITY_LINGUISTIC_WEIGHT,
    DENSITY_THRESHOLD_MAX, DENSITY_THRESHOLD_MIN, HYBRID_GRAPH_WEIGHT, HYBRID_LINGUISTIC_WEIGHT,
    HYBRID_SEMANTIC_WEIGHT, IMPORTANCE_DECAY_MAX, IMPORTANCE_DECAY_MIN,
    SPREADING_ACTIVATION_THRESHOLD, SPREADING_MAX_HOPS,
};
use crate::embeddings::Embedder;
use crate::graph_memory::{EpisodicNode, GraphMemory};
use crate::memory::query_parser::{analyze_query, QueryAnalysis};
use crate::memory::types::{Memory, Query, RetrievalStats, SharedMemory};

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

/// Calculate density-dependent graph weight (SHO-26)
///
/// Graph weight scales linearly with density from MIN to MAX:
/// - density < 0.5: weight = 0.1 (sparse graph, don't trust associations)
/// - density > 2.0: weight = 0.5 (dense graph, trust learned associations)
/// - in between: linear interpolation
///
/// Formula: weight = MIN + (density - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN) * (MAX - MIN)
///
/// Reference: GraphRAG Survey (arXiv 2408.08921)
pub fn calculate_density_weights(graph_density: f32) -> (f32, f32, f32) {
    let graph_weight = if graph_density <= DENSITY_THRESHOLD_MIN {
        DENSITY_GRAPH_WEIGHT_MIN
    } else if graph_density >= DENSITY_THRESHOLD_MAX {
        DENSITY_GRAPH_WEIGHT_MAX
    } else {
        // Linear interpolation between MIN and MAX
        let ratio =
            (graph_density - DENSITY_THRESHOLD_MIN) / (DENSITY_THRESHOLD_MAX - DENSITY_THRESHOLD_MIN);
        DENSITY_GRAPH_WEIGHT_MIN + ratio * (DENSITY_GRAPH_WEIGHT_MAX - DENSITY_GRAPH_WEIGHT_MIN)
    };

    let linguistic_weight = DENSITY_LINGUISTIC_WEIGHT;
    let semantic_weight = 1.0 - graph_weight - linguistic_weight;

    (semantic_weight, graph_weight, linguistic_weight)
}

/// Calculate importance-weighted decay for spreading activation (SHO-26)
///
/// Important memories (decisions, learnings) decay slower to preserve signal.
/// Weak memories (observations, context) decay faster for exploration.
///
/// Formula: decay = DECAY_MIN + (1.0 - importance) * (DECAY_MAX - DECAY_MIN)
/// - importance = 1.0: decay = 0.1 (high-value, preserve)
/// - importance = 0.0: decay = 0.4 (low-value, explore)
///
/// Reference: spreadr R package (Siew, 2019)
pub fn calculate_importance_weighted_decay(importance: f32) -> f32 {
    let clamped_importance = importance.clamp(0.0, 1.0);
    IMPORTANCE_DECAY_MIN + (1.0 - clamped_importance) * (IMPORTANCE_DECAY_MAX - IMPORTANCE_DECAY_MIN)
}

/// Spreading activation retrieval (legacy - uses fixed weights)
///
/// This is the core algorithm implementing Anderson & Pirolli (1984)
/// spreading activation model adapted for episodic memory retrieval.
///
/// For SHO-26 density-dependent weights, use `spreading_activation_retrieve_with_stats`
pub fn spreading_activation_retrieve(
    query_text: &str,
    query: &Query,
    graph: &GraphMemory,
    embedder: &dyn Embedder,
    episode_to_memory_fn: impl Fn(&EpisodicNode) -> Result<Option<SharedMemory>>,
) -> Result<Vec<ActivatedMemory>> {
    // Delegate to enhanced version with default density (uses legacy weights)
    let (memories, _stats) = spreading_activation_retrieve_with_stats(
        query_text,
        query,
        graph,
        embedder,
        None, // No density = use legacy fixed weights
        episode_to_memory_fn,
    )?;
    Ok(memories)
}

/// Spreading activation retrieval with density-dependent weights (SHO-26)
///
/// Enhanced version that:
/// - Uses density-dependent hybrid weights (graph trust scales with associations)
/// - Applies importance-weighted decay (important memories decay slower)
/// - Returns RetrievalStats for observability
///
/// # Arguments
/// - `query_text`: The search query
/// - `query`: Query parameters including max_results
/// - `graph`: The knowledge graph for spreading activation
/// - `embedder`: Embedding model for semantic scoring
/// - `graph_density`: Optional density (edges/memories). If None, uses fixed weights.
/// - `episode_to_memory_fn`: Function to convert episodes to memories
///
/// # Returns
/// (Vec<ActivatedMemory>, RetrievalStats)
pub fn spreading_activation_retrieve_with_stats(
    query_text: &str,
    query: &Query,
    graph: &GraphMemory,
    embedder: &dyn Embedder,
    graph_density: Option<f32>,
    episode_to_memory_fn: impl Fn(&EpisodicNode) -> Result<Option<SharedMemory>>,
) -> Result<(Vec<ActivatedMemory>, RetrievalStats)> {
    let start_time = Instant::now();
    let mut stats = RetrievalStats::default();

    // Determine weights based on density
    let (semantic_weight, graph_weight, linguistic_weight) = if let Some(density) = graph_density {
        stats.mode = "associative".to_string();
        stats.graph_density = density;
        calculate_density_weights(density)
    } else {
        stats.mode = "hybrid".to_string();
        stats.graph_density = 0.0;
        (
            HYBRID_SEMANTIC_WEIGHT,
            HYBRID_GRAPH_WEIGHT,
            HYBRID_LINGUISTIC_WEIGHT,
        )
    };

    stats.semantic_weight = semantic_weight;
    stats.graph_weight = graph_weight;
    stats.linguistic_weight = linguistic_weight;

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
    tracing::info!(
        "  Weights: semantic={:.2}, graph={:.2}, linguistic={:.2}",
        semantic_weight,
        graph_weight,
        linguistic_weight
    );

    // Step 2: Initialize activation map from focal entities (nouns)
    let mut activation_map: HashMap<Uuid, f32> = HashMap::new();

    for entity in &analysis.focal_entities {
        // Find entity in graph by name
        if let Some(entity_node) = graph.find_entity_by_name(&entity.text)? {
            // Initial activation = IC weight (2.3 for nouns)
            activation_map.insert(entity_node.uuid, entity.ic_weight);
            stats.entities_activated += 1;

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
        stats.retrieval_time_us = start_time.elapsed().as_micros() as u64;
        return Ok((Vec::new(), stats)); // Caller should fall back to semantic search
    }

    // Step 3: Spread activation through graph (Anderson & Pirolli 1984)
    // With importance-weighted decay (SHO-26)
    let graph_start = Instant::now();

    for hop in 1..=SPREADING_MAX_HOPS {
        stats.graph_hops = hop;

        tracing::debug!(
            "📊 Spreading activation (hop {}/{})",
            hop,
            SPREADING_MAX_HOPS
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

                // SHO-26: Importance-weighted decay
                // Use edge strength as proxy for importance (stronger edges = more important)
                let importance = edge.strength;
                let decay_rate = calculate_importance_weighted_decay(importance);
                let decay = (-decay_rate * hop as f32).exp();

                let spread_amount = source_activation * decay * edge.strength;

                // Accumulate activation
                let new_activation = activation_map.entry(target_uuid).or_insert(0.0);
                *new_activation += spread_amount;

                // Track newly activated entities
                if *new_activation >= SPREADING_ACTIVATION_THRESHOLD
                    && *new_activation - spread_amount < SPREADING_ACTIVATION_THRESHOLD
                {
                    stats.entities_activated += 1;
                }
            }
        }

        // Prune weak activations (ACT-R model)
        activation_map.retain(|_, activation| *activation > SPREADING_ACTIVATION_THRESHOLD);

        tracing::debug!("  Activated entities: {}", activation_map.len());
    }

    stats.graph_time_us = graph_start.elapsed().as_micros() as u64;
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

    stats.graph_candidates = activated_memories.len();
    tracing::info!(
        "📊 Retrieved {} episodic memories via graph",
        activated_memories.len()
    );

    // Step 5: Convert episodes to memories and calculate scores
    let mut scored_memories = Vec::new();

    // Generate query embedding once (for semantic scoring)
    let embedding_start = Instant::now();
    let query_embedding = embedder.encode(query_text)?;
    stats.embedding_time_us = embedding_start.elapsed().as_micros() as u64;

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

            // SHO-26: Density-dependent hybrid scoring
            let final_score = graph_weight * graph_activation
                + semantic_weight * semantic_score
                + linguistic_weight * linguistic_score;

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

    stats.retrieval_time_us = start_time.elapsed().as_micros() as u64;

    tracing::info!(
        "🎯 Returning {} memories (top scores: {:?})",
        scored_memories.len(),
        scored_memories
            .iter()
            .take(3)
            .map(|m| m.final_score)
            .collect::<Vec<_>>()
    );

    Ok((scored_memories, stats))
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
    fn test_density_weights_sparse() {
        // Sparse graph: density < 0.5 -> min graph weight
        let (semantic, graph, linguistic) = calculate_density_weights(0.3);
        assert!((graph - DENSITY_GRAPH_WEIGHT_MIN).abs() < 0.001);
        assert!((linguistic - DENSITY_LINGUISTIC_WEIGHT).abs() < 0.001);
        assert!((semantic + graph + linguistic - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_density_weights_dense() {
        // Dense graph: density > 2.0 -> max graph weight
        let (semantic, graph, linguistic) = calculate_density_weights(2.5);
        assert!((graph - DENSITY_GRAPH_WEIGHT_MAX).abs() < 0.001);
        assert!((linguistic - DENSITY_LINGUISTIC_WEIGHT).abs() < 0.001);
        assert!((semantic + graph + linguistic - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_density_weights_interpolation() {
        // Medium density: should interpolate
        let (semantic, graph, linguistic) = calculate_density_weights(1.25);
        assert!(graph > DENSITY_GRAPH_WEIGHT_MIN);
        assert!(graph < DENSITY_GRAPH_WEIGHT_MAX);
        assert!((linguistic - DENSITY_LINGUISTIC_WEIGHT).abs() < 0.001);
        assert!((semantic + graph + linguistic - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_importance_weighted_decay_high() {
        // High importance -> low decay (preserve signal)
        let decay = calculate_importance_weighted_decay(1.0);
        assert!((decay - IMPORTANCE_DECAY_MIN).abs() < 0.001);
    }

    #[test]
    fn test_importance_weighted_decay_low() {
        // Low importance -> high decay (explore)
        let decay = calculate_importance_weighted_decay(0.0);
        assert!((decay - IMPORTANCE_DECAY_MAX).abs() < 0.001);
    }

    #[test]
    fn test_importance_weighted_decay_mid() {
        // Medium importance -> intermediate decay
        let decay = calculate_importance_weighted_decay(0.5);
        let expected = IMPORTANCE_DECAY_MIN + 0.5 * (IMPORTANCE_DECAY_MAX - IMPORTANCE_DECAY_MIN);
        assert!((decay - expected).abs() < 0.001);
    }

    #[test]
    fn test_activation_decay() {
        // Test that importance-weighted decay varies correctly
        let initial_activation = 1.0;

        // High importance = slow decay
        let high_importance_decay = calculate_importance_weighted_decay(0.9);
        let high_importance_final = initial_activation * (-high_importance_decay).exp();

        // Low importance = fast decay
        let low_importance_decay = calculate_importance_weighted_decay(0.1);
        let low_importance_final = initial_activation * (-low_importance_decay).exp();

        // High importance should retain more activation
        assert!(high_importance_final > low_importance_final);
    }
}
