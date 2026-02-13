//! Graph-Aware Retrieval using Spreading Activation
//!
//! Based on:
//! - Anderson & Pirolli (1984): "Spread of Activation"
//! - Collins & Loftus (1975): "Spreading-activation theory of semantic processing"
//! - Xiong et al. (2017): "Explicit Semantic Ranking via Knowledge Graph Embedding"
//! - GraphRAG Survey (arXiv 2408.08921): Hybrid KG-Vector improves 13.1%
//! - spreadr R package (Siew, 2019): Importance-weighted decay
//! - ACT-R cognitive architecture: Multi-source activation with intersection boost
//!
//! Implements spreading activation algorithm for memory retrieval:
//! 1. Extract entities from query (using linguistic analysis)
//! 2. Activate entities in knowledge graph (salience-weighted, ACT-R inspired)
//! 3. Spread activation through graph relationships (importance-weighted decay)
//!    - PIPE-7: Bidirectional spreading for 2+ entities (meet-in-middle)
//! 4. Retrieve episodic memories connected to activated entities
//! 5. Score using hybrid method (density-dependent graph + semantic + linguistic)
//!
//! SHO-26 Enhancements:
//! - Density-dependent hybrid weights: Graph trust scales with learned associations
//! - Importance-weighted decay: Important memories decay slower, preserve signal
//!
//! PIPE-7 Enhancements:
//! - Bidirectional spreading: When 2+ focal entities, split and spread from both ends
//! - Intersection boost: Entities reached from both directions get 1.5× activation
//! - Complexity reduction: O(b^d) → O(2 × b^(d/2)) for multi-entity queries

use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

use crate::constants::{
    BIDIRECTIONAL_DENSITY_DENSE, BIDIRECTIONAL_DENSITY_SPARSE, BIDIRECTIONAL_HOPS_DENSE,
    BIDIRECTIONAL_HOPS_MEDIUM, BIDIRECTIONAL_HOPS_SPARSE, BIDIRECTIONAL_INTERSECTION_BOOST,
    BIDIRECTIONAL_INTERSECTION_MIN, BIDIRECTIONAL_MIN_ENTITIES, DENSITY_GRAPH_WEIGHT_MAX,
    DENSITY_GRAPH_WEIGHT_MIN, DENSITY_LINGUISTIC_WEIGHT, DENSITY_THRESHOLD_MAX,
    DENSITY_THRESHOLD_MIN, EDGE_TIER_TRUST_L1, EDGE_TIER_TRUST_L2, EDGE_TIER_TRUST_L3,
    EDGE_TIER_TRUST_LTP, HYBRID_GRAPH_WEIGHT, HYBRID_LINGUISTIC_WEIGHT, HYBRID_SEMANTIC_WEIGHT,
    IMPORTANCE_DECAY_MAX, IMPORTANCE_DECAY_MIN, MEMORY_TIER_GRAPH_MULT_ARCHIVE,
    MEMORY_TIER_GRAPH_MULT_LONGTERM, MEMORY_TIER_GRAPH_MULT_SESSION,
    MEMORY_TIER_GRAPH_MULT_WORKING, SALIENCE_BOOST_FACTOR, SPREADING_ACTIVATION_THRESHOLD,
    SPREADING_DEGREE_NORMALIZATION, SPREADING_EARLY_TERMINATION_CANDIDATES,
    SPREADING_EARLY_TERMINATION_RATIO, SPREADING_MAX_HOPS, SPREADING_MIN_CANDIDATES,
    SPREADING_MIN_HOPS, SPREADING_NORMALIZATION_FACTOR, SPREADING_RELAXED_THRESHOLD,
};
use crate::embeddings::Embedder;
use crate::graph_memory::{EdgeTier, EpisodicNode, GraphMemory};
use crate::memory::types::MemoryTier;
// Note: compute_relevance removed - using unified density-weighted scoring directly
use crate::memory::query_parser::{analyze_query, QueryAnalysis};
use crate::memory::types::{Memory, Query, RetrievalStats, SharedMemory};
use crate::similarity::cosine_similarity;

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

/// Calculate density-dependent graph weight (SHO-26, corrected)
///
/// SPARSE = TRUST GRAPH (edges that survived Hebbian decay are high-quality)
/// DENSE = TRUST VECTOR (too many noisy edges, use semantic similarity)
///
/// Graph weight scales INVERSELY with density:
/// - density < 0.5: weight = 0.5 (sparse graph, trust high-signal associations)
/// - density > 2.0: weight = 0.1 (dense graph, noisy, trust vector search)
/// - in between: linear interpolation
///
/// Reference: GraphRAG Survey (arXiv 2408.08921) + Hebbian learning insight
pub fn calculate_density_weights(graph_density: f32) -> (f32, f32, f32) {
    // INVERTED logic: sparse graphs have higher graph weight
    let graph_weight = if graph_density <= DENSITY_THRESHOLD_MIN {
        DENSITY_GRAPH_WEIGHT_MAX // Sparse = trust graph more
    } else if graph_density >= DENSITY_THRESHOLD_MAX {
        DENSITY_GRAPH_WEIGHT_MIN // Dense = trust vector more
    } else {
        // Linear interpolation: higher density = lower graph weight
        let ratio = (graph_density - DENSITY_THRESHOLD_MIN)
            / (DENSITY_THRESHOLD_MAX - DENSITY_THRESHOLD_MIN);
        DENSITY_GRAPH_WEIGHT_MAX - ratio * (DENSITY_GRAPH_WEIGHT_MAX - DENSITY_GRAPH_WEIGHT_MIN)
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
    IMPORTANCE_DECAY_MIN
        + (1.0 - clamped_importance) * (IMPORTANCE_DECAY_MAX - IMPORTANCE_DECAY_MIN)
}

// =============================================================================
// PIPE-7: BIDIRECTIONAL SPREADING ACTIVATION
// =============================================================================
//
// Based on:
// - Collins & Loftus (1975): Spreading activation in semantic memory
// - ACT-R cognitive architecture: Multi-source activation with intersection boost
// - Meet-in-middle search: O(b^d) → O(2 × b^(d/2)) complexity reduction
//
// Key insight: When query has multiple focal entities (e.g., "Rust" and "database"),
// spreading from both ends and boosting intersection entities finds the "bridge"
// concepts that connect the query terms - these are often the most relevant.
//
// Density-adaptive hop count:
// - Dense (fresh) graphs: fewer hops to avoid noise from L1 edges
// - Sparse (mature) graphs: more hops to find connections through curated L2/L3 edges

/// Calculate density-adaptive hop count for bidirectional spreading
///
/// Graph lifecycle: Dense → Sparse as decay prunes weak edges over time.
/// - Fresh system (dense): many noisy L1 edges → fewer hops (2)
/// - Mature system (sparse): curated L2/L3 edges → more hops (4)
///
/// Returns: number of hops per direction
pub fn calculate_adaptive_hops(graph_density: Option<f32>) -> usize {
    match graph_density {
        Some(density) if density > BIDIRECTIONAL_DENSITY_DENSE => {
            // Dense graph: limit exploration to avoid noise
            BIDIRECTIONAL_HOPS_DENSE
        }
        Some(density) if density < BIDIRECTIONAL_DENSITY_SPARSE => {
            // Sparse graph: explore deeper through quality edges
            BIDIRECTIONAL_HOPS_SPARSE
        }
        Some(_) => {
            // Medium density: balanced exploration
            BIDIRECTIONAL_HOPS_MEDIUM
        }
        None => {
            // No density info: use medium as safe default
            BIDIRECTIONAL_HOPS_MEDIUM
        }
    }
}

/// Spread activation from a set of seed entities for a fixed number of hops
///
/// This is a single-direction spread used by bidirectional algorithm.
/// Returns the activation map after spreading.
fn spread_single_direction(
    seeds: &[(Uuid, f32)],
    graph: &GraphMemory,
    max_hops: usize,
    threshold: f32,
) -> Result<(HashMap<Uuid, f32>, Vec<Uuid>)> {
    let mut activation_map: HashMap<Uuid, f32> = seeds.iter().cloned().collect();
    let mut traversed_edges: Vec<Uuid> = Vec::new();

    for hop in 1..=max_hops {
        let current_activated: Vec<(Uuid, f32)> =
            activation_map.iter().map(|(id, act)| (*id, *act)).collect();

        for (entity_uuid, source_activation) in current_activated {
            if source_activation < threshold {
                continue;
            }

            const MAX_EDGES_PER_SPREAD: usize = 100;
            let edges =
                graph.get_entity_relationships_limited(&entity_uuid, Some(MAX_EDGES_PER_SPREAD))?;

            // Degree normalization: prevent hub nodes from flooding the network.
            // Divides outgoing activation by sqrt(1 + degree), matching the fan effect
            // in ACT-R spreading activation (Anderson & Reder 1999).
            let degree_norm = if SPREADING_DEGREE_NORMALIZATION {
                1.0 / (1.0 + edges.len() as f32).sqrt()
            } else {
                1.0
            };

            for edge in edges {
                let target_uuid = edge.to_entity;

                // Edge-tier trust weight
                let tier_trust = if edge.is_potentiated() {
                    EDGE_TIER_TRUST_LTP
                } else {
                    match edge.tier {
                        EdgeTier::L3Semantic => EDGE_TIER_TRUST_L3,
                        EdgeTier::L2Episodic => EDGE_TIER_TRUST_L2,
                        EdgeTier::L1Working => EDGE_TIER_TRUST_L1,
                    }
                };

                // Importance-weighted decay
                let importance = edge.strength;
                let decay_rate = calculate_importance_weighted_decay(importance);
                let decay = (-decay_rate * hop as f32).exp();

                let spread_amount =
                    source_activation * decay * edge.strength * tier_trust * degree_norm;

                let new_activation = activation_map.entry(target_uuid).or_insert(0.0);
                *new_activation += spread_amount;

                if spread_amount > 0.01 {
                    traversed_edges.push(edge.uuid);
                }
            }
        }

        // Normalize to prevent unbounded growth
        let max_activation = activation_map
            .values()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        if max_activation > SPREADING_NORMALIZATION_FACTOR {
            let scale = SPREADING_NORMALIZATION_FACTOR / max_activation;
            for activation in activation_map.values_mut() {
                *activation *= scale;
            }
        }

        // Prune weak activations
        activation_map.retain(|_, activation| *activation > threshold);
    }

    Ok((activation_map, traversed_edges))
}

/// Bidirectional spreading activation with intersection boost (PIPE-7)
///
/// Splits focal entities into forward/backward sets, spreads from each,
/// and boosts entities found at the intersection.
///
/// # Arguments
/// - `entity_data`: Focal entities with (uuid, name, ic_weight, salience)
/// - `graph`: The knowledge graph
/// - `total_salience`: Sum of entity saliences for normalization
/// - `hops_per_direction`: Density-adaptive hop count (2-4)
///
/// Returns: (combined_activation_map, traversed_edges, intersection_count)
fn bidirectional_spread(
    entity_data: &[(Uuid, String, f32, f32)], // (uuid, name, ic_weight, salience)
    graph: &GraphMemory,
    total_salience: f32,
    hops_per_direction: usize,
) -> Result<(HashMap<Uuid, f32>, Vec<Uuid>, usize)> {
    // Split entities into forward/backward sets (alternating assignment)
    // This distributes entities evenly regardless of count
    let mut forward_seeds: Vec<(Uuid, f32)> = Vec::new();
    let mut backward_seeds: Vec<(Uuid, f32)> = Vec::new();

    for (i, (uuid, _name, ic_weight, salience)) in entity_data.iter().enumerate() {
        let normalized_salience = salience / total_salience;
        let salience_boost = SALIENCE_BOOST_FACTOR * normalized_salience;
        let initial_activation = ic_weight * (1.0 + salience_boost);

        if i % 2 == 0 {
            forward_seeds.push((*uuid, initial_activation));
        } else {
            backward_seeds.push((*uuid, initial_activation));
        }
    }

    // If odd number of entities, backward might be empty - add last entity to both
    if backward_seeds.is_empty() && !forward_seeds.is_empty() {
        backward_seeds.push(forward_seeds[forward_seeds.len() - 1]);
    }

    tracing::debug!(
        "🔀 Bidirectional spread: {} forward seeds, {} backward seeds",
        forward_seeds.len(),
        backward_seeds.len()
    );

    // Spread from both directions with density-adaptive hops
    let threshold = SPREADING_ACTIVATION_THRESHOLD;
    let (forward_map, forward_edges) =
        spread_single_direction(&forward_seeds, graph, hops_per_direction, threshold)?;

    let (backward_map, backward_edges) =
        spread_single_direction(&backward_seeds, graph, hops_per_direction, threshold)?;

    // Combine maps with intersection boost
    let mut combined_map: HashMap<Uuid, f32> = HashMap::new();
    let mut intersection_count = 0;

    // Collect all entities from both directions
    let all_entities: std::collections::HashSet<Uuid> = forward_map
        .keys()
        .chain(backward_map.keys())
        .cloned()
        .collect();

    for entity_uuid in all_entities {
        let forward_activation = forward_map.get(&entity_uuid).cloned().unwrap_or(0.0);
        let backward_activation = backward_map.get(&entity_uuid).cloned().unwrap_or(0.0);

        // Check if this is an intersection entity (meaningful activation from both)
        let is_intersection = forward_activation >= BIDIRECTIONAL_INTERSECTION_MIN
            && backward_activation >= BIDIRECTIONAL_INTERSECTION_MIN;

        let combined_activation = if is_intersection {
            intersection_count += 1;
            // Intersection boost: these are "bridge" concepts
            (forward_activation + backward_activation) * BIDIRECTIONAL_INTERSECTION_BOOST
        } else {
            // Non-intersection: just sum the activations
            forward_activation + backward_activation
        };

        combined_map.insert(entity_uuid, combined_activation);
    }

    // Combine traversed edges
    let mut all_edges = forward_edges;
    all_edges.extend(backward_edges);

    tracing::debug!(
        "🔀 Bidirectional result: {} entities ({} intersections), {} edges",
        combined_map.len(),
        intersection_count,
        all_edges.len()
    );

    Ok((combined_map, all_edges, intersection_count))
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
    // ACT-R inspired: weight initial activation by entity salience (attention budget)
    let mut activation_map: HashMap<Uuid, f32> = HashMap::new();

    // First pass: collect entities with their salience values
    let mut entity_data: Vec<(Uuid, String, f32, f32)> = Vec::new(); // (uuid, name, ic_weight, salience)

    for entity in &analysis.focal_entities {
        if let Some(entity_node) = graph.find_entity_by_name(&entity.text)? {
            entity_data.push((
                entity_node.uuid,
                entity.text.clone(),
                entity.ic_weight,
                entity_node.salience,
            ));
        } else {
            tracing::debug!("  ✗ Entity '{}' not found in graph", entity.text);
        }
    }

    // Calculate total salience for normalization (attention budget)
    let total_salience: f32 = entity_data.iter().map(|(_, _, _, s)| s).sum();
    let total_salience = total_salience.max(0.1); // Prevent division by zero

    // Second pass: apply salience-weighted activation
    let mut total_boost = 0.0_f32;
    for (uuid, name, ic_weight, salience) in &entity_data {
        // Normalize salience across query entities (ACT-R attention budget)
        let normalized_salience = salience / total_salience;

        // ACT-R inspired: activation = IC_weight × (1 + boost_factor × normalized_salience)
        let salience_boost = SALIENCE_BOOST_FACTOR * normalized_salience;
        let initial_activation = ic_weight * (1.0 + salience_boost);

        activation_map.insert(*uuid, initial_activation);
        stats.entities_activated += 1;
        total_boost += salience_boost;

        tracing::debug!(
            "  ✓ Activated '{}' (IC={:.2}, salience={:.2}, norm={:.2}, boost={:.2}, activation={:.2})",
            name,
            ic_weight,
            salience,
            normalized_salience,
            salience_boost,
            initial_activation
        );
    }

    // Track average salience boost for observability
    stats.avg_salience_boost = if !entity_data.is_empty() {
        total_boost / entity_data.len() as f32
    } else {
        0.0
    };

    if activation_map.is_empty() {
        tracing::warn!("No entities found in graph, falling back to semantic search");
        stats.retrieval_time_us = start_time.elapsed().as_micros() as u64;
        return Ok((Vec::new(), stats)); // Caller should fall back to semantic search
    }

    // Step 3: Spread activation through graph
    // PIPE-7: Use bidirectional spreading when 2+ focal entities, else unidirectional
    let graph_start = Instant::now();
    let mut traversed_edges: Vec<Uuid>;

    if entity_data.len() >= BIDIRECTIONAL_MIN_ENTITIES {
        // PIPE-7: Bidirectional spreading activation
        // Split entities into forward/backward sets, spread from each, boost intersections
        // Density-adaptive hops: dense (fresh) graphs use fewer hops, sparse (mature) use more
        let adaptive_hops = calculate_adaptive_hops(graph_density);

        tracing::info!(
            "🔀 Using bidirectional spreading ({} focal entities, {} hops/direction, density={:.2})",
            entity_data.len(),
            adaptive_hops,
            graph_density.unwrap_or(0.0)
        );

        let (bidirectional_map, edges, intersection_count) =
            bidirectional_spread(&entity_data, graph, total_salience, adaptive_hops)?;

        activation_map = bidirectional_map;
        traversed_edges = edges;
        stats.entities_activated = activation_map.len();
        stats.graph_hops = adaptive_hops * 2; // Effective depth (forward + backward)

        tracing::info!(
            "🔀 Bidirectional complete: {} entities, {} intersections",
            activation_map.len(),
            intersection_count
        );
    } else {
        // Standard unidirectional spreading (Anderson & Pirolli 1984)
        // With importance-weighted decay (SHO-26) and adaptive limits
        tracing::info!(
            "📊 Using unidirectional spreading ({} focal entity)",
            entity_data.len()
        );

        let mut edges_collected: Vec<Uuid> = Vec::new();
        let mut current_threshold = SPREADING_ACTIVATION_THRESHOLD;

        for hop in 1..=SPREADING_MAX_HOPS {
            stats.graph_hops = hop;
            let count_before = activation_map.len();

            tracing::debug!(
                "📊 Spreading activation (hop {}/{}, threshold={:.4})",
                hop,
                SPREADING_MAX_HOPS,
                current_threshold
            );

            // Clone to avoid borrow issues
            let current_activated: Vec<(Uuid, f32)> =
                activation_map.iter().map(|(id, act)| (*id, *act)).collect();

            for (entity_uuid, source_activation) in current_activated {
                // Only spread from entities with sufficient activation
                if source_activation < current_threshold {
                    continue;
                }

                // Get relationships from this entity (limited to prevent blowup)
                const MAX_EDGES_PER_SPREAD: usize = 100;
                let edges = graph
                    .get_entity_relationships_limited(&entity_uuid, Some(MAX_EDGES_PER_SPREAD))?;

                for edge in edges {
                    // Spread activation to connected entity
                    let target_uuid = edge.to_entity;

                    // Edge-tier trust weight (SHO-D1, PIPE-4)
                    let tier_trust = if edge.is_potentiated() {
                        EDGE_TIER_TRUST_LTP
                    } else {
                        match edge.tier {
                            EdgeTier::L3Semantic => EDGE_TIER_TRUST_L3,
                            EdgeTier::L2Episodic => EDGE_TIER_TRUST_L2,
                            EdgeTier::L1Working => EDGE_TIER_TRUST_L1,
                        }
                    };

                    // SHO-26: Importance-weighted decay
                    let importance = edge.strength;
                    let decay_rate = calculate_importance_weighted_decay(importance);
                    let decay = (-decay_rate * hop as f32).exp();

                    let spread_amount = source_activation * decay * edge.strength * tier_trust;

                    let new_activation = activation_map.entry(target_uuid).or_insert(0.0);
                    *new_activation += spread_amount;

                    if spread_amount > 0.01 {
                        edges_collected.push(edge.uuid);
                    }

                    if *new_activation >= current_threshold
                        && *new_activation - spread_amount < current_threshold
                    {
                        stats.entities_activated += 1;
                    }
                }
            }

            // Normalize activations to prevent unbounded growth
            let max_activation = activation_map
                .values()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);

            if max_activation > SPREADING_NORMALIZATION_FACTOR {
                let scale = SPREADING_NORMALIZATION_FACTOR / max_activation;
                for activation in activation_map.values_mut() {
                    *activation *= scale;
                }
            }

            // Prune weak activations (ACT-R model)
            activation_map.retain(|_, activation| *activation > current_threshold);

            let count_after = activation_map.len();
            let new_activations = count_after.saturating_sub(count_before);

            tracing::debug!(
                "  Activated entities: {} (+{} new)",
                count_after,
                new_activations
            );

            // Adaptive threshold relaxation
            if count_after < SPREADING_MIN_CANDIDATES
                && current_threshold > SPREADING_RELAXED_THRESHOLD
            {
                current_threshold = SPREADING_RELAXED_THRESHOLD;
                tracing::debug!(
                    "  Relaxing threshold to {:.4} (only {} candidates)",
                    current_threshold,
                    count_after
                );
            }

            // Early termination checks (only after minimum hops)
            if hop >= SPREADING_MIN_HOPS {
                let new_ratio = if count_after > 0 {
                    new_activations as f32 / count_after as f32
                } else {
                    0.0
                };

                if new_ratio < SPREADING_EARLY_TERMINATION_RATIO && count_after > 0 {
                    tracing::debug!(
                        "  Early termination: activation saturated ({:.1}% new)",
                        new_ratio * 100.0
                    );
                    break;
                }

                if count_after >= SPREADING_EARLY_TERMINATION_CANDIDATES {
                    tracing::debug!(
                        "  Early termination: sufficient coverage ({} candidates)",
                        count_after
                    );
                    break;
                }
            }
        }

        traversed_edges = edges_collected;
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

    // Step 5: Convert episodes to memories and calculate scores using UNIFIED scoring
    let mut scored_memories = Vec::new();

    // Generate query embedding once (for semantic scoring)
    let embedding_start = Instant::now();
    let query_embedding = embedder.encode(query_text)?;
    stats.embedding_time_us = embedding_start.elapsed().as_micros() as u64;

    let now = chrono::Utc::now();

    for (_episode_uuid, (graph_activation, episode)) in activated_memories {
        // Convert episode to memory
        if let Some(memory) = episode_to_memory_fn(&episode)? {
            // Calculate semantic similarity (still needed for ActivatedMemory debug fields)
            let semantic_score = if let Some(mem_emb) = &memory.experience.embeddings {
                cosine_similarity(&query_embedding, mem_emb)
            } else {
                0.0
            };

            // Calculate linguistic match score (normalized to 0.0-1.0)
            let linguistic_raw = calculate_linguistic_match(&memory, &analysis);
            let linguistic_score = linguistic_raw; // Already normalized in calculate_linguistic_match

            // Memory-tier graph weight multiplier (SHO-D2)
            // Working memories are dense/noisy → lower graph trust
            // LongTerm memories are sparse/proven → full graph trust
            let tier_graph_mult = match memory.tier {
                MemoryTier::Working => MEMORY_TIER_GRAPH_MULT_WORKING, // 0.3
                MemoryTier::Session => MEMORY_TIER_GRAPH_MULT_SESSION, // 0.6
                MemoryTier::LongTerm => MEMORY_TIER_GRAPH_MULT_LONGTERM, // 1.0
                MemoryTier::Archive => MEMORY_TIER_GRAPH_MULT_ARCHIVE, // 1.2
            };

            // Unified scoring using density-dependent weights (calculated at function start)
            // Graph weight is further adjusted by memory tier
            // Weights are: semantic_weight, graph_weight * tier_mult, linguistic_weight
            let tier_adjusted_graph_weight = graph_weight * tier_graph_mult;
            // Renormalize weights to sum to 1.0
            let weight_sum = semantic_weight + tier_adjusted_graph_weight + linguistic_weight;
            let norm_semantic = semantic_weight / weight_sum;
            let norm_graph = tier_adjusted_graph_weight / weight_sum;
            let norm_linguistic = linguistic_weight / weight_sum;

            let hybrid_score = semantic_score * norm_semantic
                + graph_activation * norm_graph
                + linguistic_score * norm_linguistic;

            // Recency decay (10% contribution) - recent memories get boost
            // λ = 0.01 means ~50% at 70 hours, ~25% at 140 hours
            const RECENCY_DECAY_RATE: f32 = 0.01;
            let hours_old = (now - memory.created_at).num_hours().max(0) as f32;
            let recency_boost = (-RECENCY_DECAY_RATE * hours_old).exp() * 0.1;

            // Emotional arousal boost: high arousal = more salient (5% contribution)
            let arousal_boost = memory
                .experience
                .context
                .as_ref()
                .map(|c| c.emotional.arousal * 0.05)
                .unwrap_or(0.0);

            // Source credibility boost (5% contribution)
            let credibility_boost = memory
                .experience
                .context
                .as_ref()
                .map(|c| (c.source.credibility - 0.5).max(0.0) * 0.1)
                .unwrap_or(0.0);

            let final_score = hybrid_score + recency_boost + arousal_boost + credibility_boost;

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

    // Deduplicate traversed edges (same edge may be traversed multiple times across hops)
    traversed_edges.sort();
    traversed_edges.dedup();
    stats.traversed_edges = traversed_edges;

    tracing::info!(
        "🎯 Returning {} memories (top scores: {:?}), {} edges traversed",
        scored_memories.len(),
        scored_memories
            .iter()
            .take(3)
            .map(|m| m.final_score)
            .collect::<Vec<_>>(),
        stats.traversed_edges.len()
    );

    Ok((scored_memories, stats))
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
        // Sparse graph: density < 0.5 -> MAX graph weight (trust graph)
        let (semantic, graph, linguistic) = calculate_density_weights(0.3);
        assert!((graph - DENSITY_GRAPH_WEIGHT_MAX).abs() < 0.001);
        assert!((linguistic - DENSITY_LINGUISTIC_WEIGHT).abs() < 0.001);
        assert!((semantic + graph + linguistic - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_density_weights_dense() {
        // Dense graph: density > 2.0 -> MIN graph weight (trust vector)
        let (semantic, graph, linguistic) = calculate_density_weights(2.5);
        assert!((graph - DENSITY_GRAPH_WEIGHT_MIN).abs() < 0.001);
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

    #[test]
    fn test_adaptive_constants_valid() {
        use crate::constants::*;

        // Relaxed threshold must be lower than strict threshold
        assert!(SPREADING_RELAXED_THRESHOLD < SPREADING_ACTIVATION_THRESHOLD);

        // Min hops must be <= max hops
        assert!(SPREADING_MIN_HOPS <= SPREADING_MAX_HOPS);

        // Early termination ratio must be in (0, 1)
        assert!(SPREADING_EARLY_TERMINATION_RATIO > 0.0);
        assert!(SPREADING_EARLY_TERMINATION_RATIO < 1.0);

        // Normalization factor must be positive
        assert!(SPREADING_NORMALIZATION_FACTOR > 0.0);

        // Min candidates for relaxation should be reasonable
        assert!(SPREADING_MIN_CANDIDATES > 0);
        assert!(SPREADING_MIN_CANDIDATES < SPREADING_EARLY_TERMINATION_CANDIDATES);
    }

    #[test]
    fn test_normalization_prevents_explosion() {
        use crate::constants::SPREADING_NORMALIZATION_FACTOR;

        // Simulate activation accumulation over hops
        let mut activations = vec![1.0, 0.8, 0.5, 0.3];

        // Simulate 5 hops of accumulation (each hop adds to existing)
        for _ in 0..5 {
            for activation in &mut activations {
                *activation += *activation * 0.5; // 50% growth per hop
            }

            // Normalize
            let max_activation = activations
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);

            if max_activation > SPREADING_NORMALIZATION_FACTOR {
                let scale = SPREADING_NORMALIZATION_FACTOR / max_activation;
                for activation in &mut activations {
                    *activation *= scale;
                }
            }
        }

        // After normalization, max should never exceed factor
        let final_max = activations
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        assert!(final_max <= SPREADING_NORMALIZATION_FACTOR + 0.001);
    }

    #[test]
    fn test_early_termination_ratio() {
        use crate::constants::SPREADING_EARLY_TERMINATION_RATIO;

        // Simulate saturation detection
        let total_before = 50;
        let total_after = 52; // Only 2 new activations

        let new_activations = total_after - total_before;
        let new_ratio = new_activations as f32 / total_after as f32;

        // Should trigger early termination (only ~4% new)
        assert!(new_ratio < SPREADING_EARLY_TERMINATION_RATIO);

        // Simulate rapid growth - should not terminate
        let growing_before = 10;
        let growing_after = 25; // 15 new activations

        let growing_new = growing_after - growing_before;
        let growing_ratio = growing_new as f32 / growing_after as f32;

        // Should NOT trigger early termination (60% new)
        assert!(growing_ratio >= SPREADING_EARLY_TERMINATION_RATIO);
    }

    // =========================================================================
    // PIPE-7: Bidirectional Spreading Activation Tests
    // =========================================================================

    #[test]
    fn test_bidirectional_constants_valid() {
        // Minimum entities must be at least 2 for bidirectional to make sense
        assert!(BIDIRECTIONAL_MIN_ENTITIES >= 2);

        // Intersection boost must be positive
        assert!(BIDIRECTIONAL_INTERSECTION_BOOST > 1.0);

        // Intersection minimum must be below activation threshold
        assert!(BIDIRECTIONAL_INTERSECTION_MIN < SPREADING_ACTIVATION_THRESHOLD);

        // Density thresholds must be ordered correctly
        assert!(BIDIRECTIONAL_DENSITY_SPARSE < BIDIRECTIONAL_DENSITY_DENSE);

        // Hop counts must be ordered: dense < medium < sparse
        assert!(BIDIRECTIONAL_HOPS_DENSE < BIDIRECTIONAL_HOPS_MEDIUM);
        assert!(BIDIRECTIONAL_HOPS_MEDIUM < BIDIRECTIONAL_HOPS_SPARSE);

        // Medium hops × 2 should approximate max hops for unidirectional
        assert!(BIDIRECTIONAL_HOPS_MEDIUM * 2 >= SPREADING_MAX_HOPS);
    }

    #[test]
    fn test_adaptive_hops_dense_graph() {
        // Dense graph (fresh system): fewer hops to avoid noise
        let hops = calculate_adaptive_hops(Some(3.0)); // Above DENSE threshold
        assert_eq!(hops, BIDIRECTIONAL_HOPS_DENSE);
        assert_eq!(hops, 2);
    }

    #[test]
    fn test_adaptive_hops_sparse_graph() {
        // Sparse graph (mature system): more hops for quality edges
        let hops = calculate_adaptive_hops(Some(0.3)); // Below SPARSE threshold
        assert_eq!(hops, BIDIRECTIONAL_HOPS_SPARSE);
        assert_eq!(hops, 4);
    }

    #[test]
    fn test_adaptive_hops_medium_graph() {
        // Medium density: balanced exploration
        let hops = calculate_adaptive_hops(Some(1.0)); // Between thresholds
        assert_eq!(hops, BIDIRECTIONAL_HOPS_MEDIUM);
        assert_eq!(hops, 3);
    }

    #[test]
    fn test_adaptive_hops_no_density() {
        // No density info: use medium as safe default
        let hops = calculate_adaptive_hops(None);
        assert_eq!(hops, BIDIRECTIONAL_HOPS_MEDIUM);
    }

    #[test]
    fn test_adaptive_hops_lifecycle() {
        // Simulate graph lifecycle: dense → medium → sparse
        let fresh_hops = calculate_adaptive_hops(Some(2.5)); // Fresh system
        let mid_hops = calculate_adaptive_hops(Some(1.0)); // 6 months in
        let mature_hops = calculate_adaptive_hops(Some(0.3)); // Mature system

        // Hops should increase as graph matures (gets sparser)
        assert!(fresh_hops <= mid_hops);
        assert!(mid_hops <= mature_hops);

        // Verify actual values
        assert_eq!(fresh_hops, 2);
        assert_eq!(mid_hops, 3);
        assert_eq!(mature_hops, 4);
    }

    #[test]
    fn test_intersection_boost_calculation() {
        // Test that intersection entities get boosted
        let forward_activation = 0.5;
        let backward_activation = 0.3;

        // Both exceed minimum threshold
        assert!(forward_activation >= BIDIRECTIONAL_INTERSECTION_MIN);
        assert!(backward_activation >= BIDIRECTIONAL_INTERSECTION_MIN);

        // Intersection boost calculation
        let boosted = (forward_activation + backward_activation) * BIDIRECTIONAL_INTERSECTION_BOOST;
        let unboosted = forward_activation + backward_activation;

        // Boosted should be higher
        assert!(boosted > unboosted);

        // Boosted should be exactly 1.5× unboosted (with default constants)
        let expected_ratio = BIDIRECTIONAL_INTERSECTION_BOOST;
        assert!((boosted / unboosted - expected_ratio).abs() < 0.001);
    }

    #[test]
    fn test_non_intersection_no_boost() {
        // Test that non-intersection entities don't get boosted
        let forward_activation = 0.5;
        let backward_activation = 0.0; // Below threshold

        // Backward doesn't meet threshold
        assert!(backward_activation < BIDIRECTIONAL_INTERSECTION_MIN);

        // Non-intersection: just sum
        let combined = forward_activation + backward_activation;

        // Should equal just forward activation
        assert!((combined - forward_activation).abs() < 0.001);
    }

    #[test]
    fn test_bidirectional_entity_split() {
        // Test alternating assignment distributes evenly
        let entities = vec![
            (Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity2".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity3".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity4".to_string(), 1.0, 0.5),
        ];

        // With 4 entities, split should be 2-2
        let mut forward_count = 0;
        let mut backward_count = 0;

        for (i, _) in entities.iter().enumerate() {
            if i % 2 == 0 {
                forward_count += 1;
            } else {
                backward_count += 1;
            }
        }

        assert_eq!(forward_count, 2);
        assert_eq!(backward_count, 2);
    }

    #[test]
    fn test_bidirectional_odd_entities() {
        // Test odd number of entities doesn't leave backward empty
        let entities = vec![
            (Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity2".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity3".to_string(), 1.0, 0.5),
        ];

        // With 3 entities: indices 0,2 go forward, index 1 goes backward
        let mut forward_seeds = Vec::new();
        let mut backward_seeds = Vec::new();

        for (i, entity) in entities.iter().enumerate() {
            if i % 2 == 0 {
                forward_seeds.push(entity.0);
            } else {
                backward_seeds.push(entity.0);
            }
        }

        // 2 forward, 1 backward
        assert_eq!(forward_seeds.len(), 2);
        assert_eq!(backward_seeds.len(), 1);

        // Both sets are non-empty (the actual function duplicates if backward would be empty)
        assert!(!forward_seeds.is_empty());
        assert!(!backward_seeds.is_empty());
    }

    #[test]
    fn test_bidirectional_threshold_triggers() {
        // Test when bidirectional is triggered vs unidirectional

        // 1 entity: unidirectional
        let single_entity = vec![(Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5)];
        assert!(single_entity.len() < BIDIRECTIONAL_MIN_ENTITIES);

        // 2 entities: bidirectional
        let two_entities = vec![
            (Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity2".to_string(), 1.0, 0.5),
        ];
        assert!(two_entities.len() >= BIDIRECTIONAL_MIN_ENTITIES);

        // 5 entities: bidirectional
        let many_entities = vec![
            (Uuid::new_v4(), "entity1".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity2".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity3".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity4".to_string(), 1.0, 0.5),
            (Uuid::new_v4(), "entity5".to_string(), 1.0, 0.5),
        ];
        assert!(many_entities.len() >= BIDIRECTIONAL_MIN_ENTITIES);
    }

    #[test]
    fn test_complexity_improvement() {
        // Theoretical complexity comparison
        // Let b = branching factor, d = depth

        // Unidirectional: O(b^d)
        // Bidirectional: O(2 × b^(d/2))

        // For b=10, d=6:
        // Unidirectional: 10^6 = 1,000,000
        // Bidirectional: 2 × 10^3 = 2,000

        let b: f64 = 10.0;
        let d: f64 = 6.0;

        let unidirectional = b.powf(d);
        let bidirectional = 2.0 * b.powf(d / 2.0);

        // Bidirectional should be significantly smaller
        assert!(bidirectional < unidirectional);

        // Ratio should be ~500× improvement
        let improvement = unidirectional / bidirectional;
        assert!(improvement > 100.0);
    }

    #[test]
    fn test_intersection_detection_threshold() {
        // Test the minimum threshold for intersection detection
        let min_threshold = BIDIRECTIONAL_INTERSECTION_MIN;

        // Should be half of activation threshold
        let expected = SPREADING_ACTIVATION_THRESHOLD / 2.0;
        assert!((min_threshold - expected).abs() < 0.001);

        // Should be positive
        assert!(min_threshold > 0.0);

        // Should be less than typical initial activation
        // (IC_NOUN * (1 + SALIENCE_BOOST_FACTOR) = 2.3 * 2 = 4.6 max)
        assert!(min_threshold < 1.0);
    }
}
