//! Production-grade retrieval engine for memory search
//! Integrated with Vamana HNSW and MiniLM embeddings for robotics/drones
//!
//! Features Hebbian-inspired adaptive learning:
//! - Outcome feedback: Memories that help complete tasks get reinforced
//! - Co-activation strengthening: Memories retrieved together form associations
//! - Time-based decay: Unused associations naturally weaken

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

use super::introspection::{
    ConsolidationEvent, ConsolidationEventBuffer, EdgeFormationReason, PruningReason,
};
use super::storage::{MemoryStorage, SearchCriteria};
use super::types::*;
use crate::constants::{
    EDGE_HALF_LIFE_HOURS, EDGE_INITIAL_STRENGTH, EDGE_MIN_STRENGTH, PREFETCH_RECENCY_FULL_BOOST,
    PREFETCH_RECENCY_FULL_HOURS, PREFETCH_RECENCY_PARTIAL_BOOST, PREFETCH_RECENCY_PARTIAL_HOURS,
    VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
};
use crate::embeddings::{minilm::MiniLMEmbedder, Embedder};
use crate::vector_db::vamana::{VamanaConfig, VamanaIndex};

/// Multi-modal retrieval engine with production vector search
pub struct RetrievalEngine {
    storage: Arc<MemoryStorage>,
    embedder: Arc<MiniLMEmbedder>,
    vector_index: Arc<RwLock<VamanaIndex>>,
    id_mapping: Arc<RwLock<IdMapping>>,
    graph: RwLock<MemoryGraph>, // Interior mutability for graph updates
    /// Storage path for persisting vector index and ID mapping
    storage_path: PathBuf,
    /// Shared consolidation event buffer for introspection
    /// Records edge formation, strengthening, and pruning events
    consolidation_events: Option<Arc<RwLock<ConsolidationEventBuffer>>>,
}

/// Bidirectional mapping between memory IDs and vector IDs
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct IdMapping {
    memory_to_vector: HashMap<MemoryId, u32>,
    vector_to_memory: HashMap<u32, MemoryId>,
}

impl IdMapping {
    fn new() -> Self {
        Self {
            memory_to_vector: HashMap::new(),
            vector_to_memory: HashMap::new(),
        }
    }

    fn insert(&mut self, memory_id: MemoryId, vector_id: u32) {
        self.memory_to_vector.insert(memory_id.clone(), vector_id);
        self.vector_to_memory.insert(vector_id, memory_id);
    }

    fn get_memory_id(&self, vector_id: u32) -> Option<&MemoryId> {
        self.vector_to_memory.get(&vector_id)
    }

    fn len(&self) -> usize {
        self.memory_to_vector.len()
    }
}

impl RetrievalEngine {
    /// Create new retrieval engine with shared embedder (CRITICAL: embedder loaded only once)
    ///
    /// Automatically loads persisted vector index and ID mapping if they exist.
    pub fn new(storage: Arc<MemoryStorage>, embedder: Arc<MiniLMEmbedder>) -> Result<Self> {
        Self::with_event_buffer(storage, embedder, None)
    }

    /// Create retrieval engine with event buffer for consolidation introspection
    ///
    /// The event buffer is used to record Hebbian learning events:
    /// - Edge formation (new associations)
    /// - Edge strengthening (co-activation)
    /// - Edge potentiation (LTP)
    /// - Edge pruning (decay below threshold)
    pub fn with_event_buffer(
        storage: Arc<MemoryStorage>,
        embedder: Arc<MiniLMEmbedder>,
        consolidation_events: Option<Arc<RwLock<ConsolidationEventBuffer>>>,
    ) -> Result<Self> {
        // Get storage path from MemoryStorage for persistence
        let storage_path = storage.path().to_path_buf();
        let index_path = storage_path.join("vector_index");

        // Initialize Vamana index optimized for 10M+ memories per user
        // Balanced between recall quality and memory efficiency
        let vamana_config = VamanaConfig {
            dimension: 384,        // MiniLM dimension
            max_degree: 32,        // Increased for better recall at scale
            search_list_size: 100, // 2x for better accuracy with 10M vectors
            alpha: 1.2,
            use_mmap: false, // Keep in memory for low-latency robotics
        };

        let mut vector_index = VamanaIndex::new(vamana_config)?;
        let mut id_mapping = IdMapping::new();

        // Try to load persisted index and ID mapping
        let index_file = index_path.join("vamana_index.bin");
        let mapping_file = index_path.join("id_mapping.bin");

        if index_file.exists() && mapping_file.exists() {
            match vector_index.load(&index_path) {
                Ok(_) => {
                    // Load ID mapping
                    match Self::load_id_mapping(&mapping_file) {
                        Ok(mapping) => {
                            id_mapping = mapping;
                            info!(
                                "Loaded persisted vector index with {} vectors and {} ID mappings",
                                vector_index.len(),
                                id_mapping.len()
                            );
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load ID mapping, starting fresh: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load vector index, starting fresh: {}", e);
                }
            }
        } else {
            info!("No persisted vector index found, starting with empty index");
        }

        Ok(Self {
            storage,
            embedder,
            vector_index: Arc::new(RwLock::new(vector_index)),
            id_mapping: Arc::new(RwLock::new(id_mapping)),
            graph: RwLock::new(MemoryGraph::new()),
            storage_path,
            consolidation_events,
        })
    }

    /// Set the consolidation event buffer (for late binding after construction)
    pub fn set_consolidation_events(&mut self, events: Arc<RwLock<ConsolidationEventBuffer>>) {
        self.consolidation_events = Some(events);
    }

    /// Record a consolidation event (helper method)
    fn record_event(&self, event: ConsolidationEvent) {
        if let Some(ref buffer) = self.consolidation_events {
            buffer.write().push(event);
        }
    }

    /// Load ID mapping from file
    fn load_id_mapping(path: &Path) -> Result<IdMapping> {
        let file = File::open(path).context("Failed to open ID mapping file")?;
        let reader = BufReader::new(file);
        bincode::deserialize_from(reader).context("Failed to deserialize ID mapping")
    }

    /// Save vector index and ID mapping to disk
    ///
    /// Called during flush_storage to persist the index for restart recovery.
    pub fn save(&self) -> Result<()> {
        let index_path = self.storage_path.join("vector_index");
        fs::create_dir_all(&index_path)?;

        // Save Vamana index
        let index = self.vector_index.read();
        index.save(&index_path)?;

        // Save ID mapping
        let mapping_file = index_path.join("id_mapping.bin");
        let id_mapping = self.id_mapping.read();
        let file = File::create(&mapping_file).context("Failed to create ID mapping file")?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &*id_mapping).context("Failed to serialize ID mapping")?;

        info!(
            "Saved vector index with {} vectors and {} ID mappings",
            index.len(),
            id_mapping.len()
        );

        Ok(())
    }

    /// Get number of vectors in the index
    pub fn len(&self) -> usize {
        self.id_mapping.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add memory to vector index (call when storing new memory)
    pub fn index_memory(&self, memory: &Memory) -> Result<()> {
        // Use pre-computed embedding if available, otherwise generate
        let embedding = if let Some(emb) = &memory.experience.embeddings {
            emb.clone()
        } else {
            // Generate embedding if not provided
            let text = Self::extract_searchable_text(memory);
            self.embedder
                .encode(&text)
                .context("Failed to generate embedding")?
        };

        // Add to Vamana index
        let mut index = self.vector_index.write();
        let vector_id = index
            .add_vector(embedding)
            .context("Failed to add vector to index")?;

        // Map memory ID to vector ID
        self.id_mapping.write().insert(memory.id.clone(), vector_id);

        Ok(())
    }

    /// Extract searchable text from memory
    fn extract_searchable_text(memory: &Memory) -> String {
        // Start with main content
        let mut text = memory.experience.content.clone();

        // Add entities
        if !memory.experience.entities.is_empty() {
            text.push(' ');
            text.push_str(&memory.experience.entities.join(" "));
        }

        // Add rich context if available
        if let Some(context) = &memory.experience.context {
            // Add conversation topic
            if let Some(topic) = &context.conversation.topic {
                text.push(' ');
                text.push_str(topic);
            }
            // Add recent conversation messages
            if !context.conversation.recent_messages.is_empty() {
                text.push(' ');
                text.push_str(&context.conversation.recent_messages.join(" "));
            }
            // Add project name
            if let Some(name) = &context.project.name {
                text.push(' ');
                text.push_str(name);
            }
        }

        // Add outcomes
        if !memory.experience.outcomes.is_empty() {
            text.push(' ');
            text.push_str(&memory.experience.outcomes.join(" "));
        }

        text
    }

    /// Search for memory IDs only (for cache-aware retrieval)
    /// Returns (MemoryId, similarity_score) pairs
    pub fn search_ids(&self, query: &Query, limit: usize) -> Result<Vec<(MemoryId, f32)>> {
        // BUG-006 FIX: Log warning for empty queries
        let query_embedding = if let Some(embedding) = &query.query_embedding {
            embedding.clone()
        } else if let Some(query_text) = &query.query_text {
            self.embedder
                .encode(query_text)
                .context("Failed to generate query embedding")?
        } else {
            tracing::warn!("Empty query in search_ids: no query_text or query_embedding provided");
            return Ok(Vec::new());
        };

        // Search vector index
        // Use VECTOR_SEARCH_CANDIDATE_MULTIPLIER to retrieve extra candidates for filtering
        // This accounts for ~50% filter rejection rate in typical queries
        let index = self.vector_index.read();
        let results = index
            .search(&query_embedding, limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER)
            .context("Vector search failed")?;

        // Map vector IDs to memory IDs
        let id_mapping = self.id_mapping.read();
        let memory_ids: Vec<(MemoryId, f32)> = results
            .into_iter()
            .filter_map(|(vector_id, distance)| {
                id_mapping
                    .get_memory_id(vector_id)
                    .map(|id| (id.clone(), distance))
            })
            .collect();

        Ok(memory_ids)
    }

    /// Get memory from storage by ID
    pub fn get_from_storage(&self, id: &MemoryId) -> Result<Memory> {
        self.storage.get(id)
    }

    /// Search for memories using multiple retrieval modes (zero-copy with Arc)
    pub fn search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let results = match query.retrieval_mode {
            // Standard modes
            RetrievalMode::Similarity => self.similarity_search(query, limit)?,
            RetrievalMode::Temporal => self.temporal_search(query, limit)?,
            RetrievalMode::Causal => self.causal_search(query, limit)?,
            RetrievalMode::Associative => self.associative_search(query, limit)?,
            RetrievalMode::Hybrid => self.hybrid_search(query, limit)?,
            // Robotics-specific modes
            RetrievalMode::Spatial => self.spatial_search(query, limit)?,
            RetrievalMode::Mission => self.mission_search(query, limit)?,
            RetrievalMode::ActionOutcome => self.action_outcome_search(query, limit)?,
        };

        Ok(results)
    }

    /// PRODUCTION: Similarity search using Vamana HNSW (sub-millisecond, zero-copy)
    fn similarity_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        // BUG-006 FIX: Log warning for empty queries
        let query_embedding = if let Some(embedding) = &query.query_embedding {
            embedding.clone()
        } else if let Some(query_text) = &query.query_text {
            self.embedder
                .encode(query_text)
                .context("Failed to generate query embedding")?
        } else {
            tracing::warn!(
                "Empty query in similarity_search: no query_text or query_embedding provided"
            );
            return Ok(Vec::new());
        };

        // Search Vamana index with candidate multiplier for filtering headroom
        let index = self.vector_index.read();
        let results = index
            .search(&query_embedding, limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER)
            .context("Vector search failed")?;

        // Map vector IDs to memory IDs and fetch memories
        let id_mapping = self.id_mapping.read();
        let mut memories = Vec::new();

        for (vector_id, _distance) in results {
            if let Some(memory_id) = id_mapping.get_memory_id(vector_id) {
                if let Ok(memory) = self.storage.get(memory_id) {
                    let shared_memory = Arc::new(memory);
                    if self.matches_filters(&shared_memory, query) {
                        memories.push(shared_memory);
                        if memories.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        Ok(memories)
    }

    /// Check if memory matches query filters
    ///
    /// Delegates to Query::matches() which is the SINGLE source of truth for all filter logic.
    /// This ensures consistent filtering across all memory tiers and retrieval modes.
    #[inline]
    pub fn matches_filters(&self, memory: &Memory, query: &Query) -> bool {
        query.matches(memory)
    }

    fn temporal_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let criteria = if let Some((start, end)) = &query.time_range {
            SearchCriteria::ByDate {
                start: *start,
                end: *end,
            }
        } else {
            let end = chrono::Utc::now();
            let start = end - chrono::Duration::days(7);
            SearchCriteria::ByDate { start, end }
        };

        let mut memories: Vec<SharedMemory> = self
            .storage
            .search(criteria)?
            .into_iter()
            .map(Arc::new)
            .collect();

        memories.retain(|m| self.matches_filters(m, query));
        memories.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        memories.truncate(limit);
        Ok(memories)
    }

    fn causal_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let seeds = self.similarity_search(query, 3)?;

        let mut results = HashSet::new();
        let mut to_explore = Vec::new();

        for seed in &seeds {
            to_explore.push(seed.id.clone());
            results.insert(seed.id.clone());
        }

        while !to_explore.is_empty() && results.len() < limit {
            if let Some(current_id) = to_explore.pop() {
                if let Ok(memory) = self.storage.get(&current_id) {
                    for related_id in &memory.experience.related_memories {
                        if !results.contains(related_id) {
                            results.insert(related_id.clone());
                            to_explore.push(related_id.clone());
                        }
                    }
                }
            }
        }

        let mut memories = Vec::new();
        for id in results.into_iter().take(limit) {
            if let Ok(memory) = self.storage.get(&id) {
                memories.push(Arc::new(memory));
            }
        }

        Ok(memories)
    }

    fn associative_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let seeds = self.similarity_search(query, 2)?;

        let mut associated = Vec::new();
        let graph = self.graph.read();

        for seed in &seeds {
            let associations = graph.find_associations(&seed.id, 5)?;

            for assoc_id in associations {
                if let Ok(memory) = self.storage.get(&assoc_id) {
                    associated.push(Arc::new(memory));
                }
            }
        }

        let mut seen = HashSet::new();
        let mut unique = Vec::new();

        for memory in associated {
            if seen.insert(memory.id.clone()) {
                unique.push(memory);
                if unique.len() >= limit {
                    break;
                }
            }
        }

        Ok(unique)
    }

    fn hybrid_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let mut all_results: HashMap<MemoryId, SharedMemory> = HashMap::new();
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        // Weight for each retrieval mode (tuned for robotics)
        let weights = [
            (RetrievalMode::Similarity, 0.5),  // Higher weight for semantic
            (RetrievalMode::Temporal, 0.2),    // Recent memories important
            (RetrievalMode::Causal, 0.2),      // Context chains
            (RetrievalMode::Associative, 0.1), // Associations
        ];

        for (mode, weight) in weights.iter() {
            let mut mode_query = query.clone();
            mode_query.retrieval_mode = mode.clone();

            let results = match mode {
                RetrievalMode::Similarity => self.similarity_search(&mode_query, limit),
                RetrievalMode::Temporal => self.temporal_search(&mode_query, limit),
                RetrievalMode::Causal => self.causal_search(&mode_query, limit),
                RetrievalMode::Associative => self.associative_search(&mode_query, limit),
                _ => continue,
            };

            if let Ok(memories) = results {
                for (rank, memory) in memories.into_iter().enumerate() {
                    // Rank score: higher rank = higher score
                    let score = weight * (1.0 / (rank as f32 + 1.0));

                    // Clone ID before moving memory into HashMap to avoid double clone
                    let memory_id = memory.id.clone();
                    *scores.entry(memory_id.clone()).or_insert(0.0) += score;
                    all_results.insert(memory_id, memory);
                }
            }
        }

        // Apply Ebbinghaus salience scoring: combines retrieval score with time-based relevance
        // This ensures older, less-accessed memories naturally fade in ranking
        let mut sorted: Vec<(f32, SharedMemory)> = all_results
            .into_iter()
            .map(|(id, memory)| {
                let retrieval_score = scores.get(&id).copied().unwrap_or(0.0);
                // Salience score factors in recency (Ebbinghaus curve) and access frequency
                let salience = memory.salience_score_with_access();
                // Final score: 70% retrieval relevance, 30% salience (time-based decay)
                let final_score = retrieval_score * 0.7 + salience * 0.3;
                (final_score, memory)
            })
            .collect();

        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(sorted.into_iter().take(limit).map(|(_, m)| m).collect())
    }

    // ========================================================================
    // ROBOTICS-SPECIFIC RETRIEVAL MODES
    // ========================================================================

    /// Spatial search: Find memories within geographic radius
    /// Uses haversine distance for accurate earth-surface calculations
    fn spatial_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let geo_filter = query
            .geo_filter
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Spatial search requires geo_filter"))?;

        let criteria = SearchCriteria::ByLocation {
            lat: geo_filter.lat,
            lon: geo_filter.lon,
            radius_meters: geo_filter.radius_meters,
        };

        let mut memories: Vec<SharedMemory> = self
            .storage
            .search(criteria)?
            .into_iter()
            .map(Arc::new)
            .collect();

        // Apply additional filters
        memories.retain(|m| self.matches_filters(m, query));

        // Sort by distance (closest first)
        memories.sort_by(|a, b| {
            let dist_a = match a.experience.geo_location {
                Some(geo) => geo_filter.haversine_distance(geo[0], geo[1]),
                None => f64::MAX,
            };
            let dist_b = match b.experience.geo_location {
                Some(geo) => geo_filter.haversine_distance(geo[0], geo[1]),
                None => f64::MAX,
            };
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        memories.truncate(limit);
        Ok(memories)
    }

    /// Mission search: Retrieve all memories from a specific mission
    /// Useful for mission replay, analysis, and learning
    fn mission_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        let mission_id = query
            .mission_id
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Mission search requires mission_id"))?;

        let criteria = SearchCriteria::ByMission(mission_id.clone());

        let mut memories: Vec<SharedMemory> = self
            .storage
            .search(criteria)?
            .into_iter()
            .map(Arc::new)
            .collect();

        // Apply additional filters
        memories.retain(|m| self.matches_filters(m, query));

        // Sort by timestamp (chronological order for mission replay)
        memories.sort_by(|a, b| a.created_at.cmp(&b.created_at));

        memories.truncate(limit);
        Ok(memories)
    }

    /// Action-outcome search: Find memories with specific reward outcomes
    /// For reinforcement learning: "What actions led to positive rewards?"
    fn action_outcome_search(&self, query: &Query, limit: usize) -> Result<Vec<SharedMemory>> {
        // Get reward range or default to positive rewards
        let (min_reward, max_reward) = query.reward_range.unwrap_or((0.0, 1.0));

        let criteria = SearchCriteria::ByReward {
            min: min_reward,
            max: max_reward,
        };

        let mut memories: Vec<SharedMemory> = self
            .storage
            .search(criteria)?
            .into_iter()
            .map(Arc::new)
            .collect();

        // Apply additional filters (action_type, robot_id, etc.)
        memories.retain(|m| self.matches_filters(m, query));

        // Sort by reward (highest first for learning from best outcomes)
        memories.sort_by(|a, b| {
            let reward_a = a.experience.reward.unwrap_or(0.0);
            let reward_b = b.experience.reward.unwrap_or(0.0);
            reward_b
                .partial_cmp(&reward_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        memories.truncate(limit);
        Ok(memories)
    }

    /// Build vector index from existing memories (one-time initialization)
    pub fn rebuild_index(&self) -> Result<()> {
        // Get all memories from storage
        let end = chrono::Utc::now();
        let start = chrono::DateTime::from_timestamp(0, 0).unwrap();
        let memories = self.storage.search(SearchCriteria::ByDate { start, end })?;

        // Batch add to Vamana
        let mut vectors = Vec::new();
        let mut memory_ids = Vec::new();

        for memory in &memories {
            let text = Self::extract_searchable_text(memory);
            let embedding = self.embedder.encode(&text)?;
            vectors.push(embedding);
            memory_ids.push(memory.id.clone());
        }

        // Build index
        let mut index = self.vector_index.write();
        index.build(vectors)?;

        // Build ID mapping
        let mut id_mapping = self.id_mapping.write();
        for (vector_id, memory_id) in memory_ids.into_iter().enumerate() {
            id_mapping.insert(memory_id, vector_id as u32);
        }

        Ok(())
    }

    /// Save vector index to disk (for persistence)
    pub fn save_index(&self, path: &Path) -> Result<()> {
        let index = self.vector_index.read();
        index.save(path).context("Failed to save vector index")?;
        Ok(())
    }

    /// Load vector index from disk
    pub fn load_index(&self, path: &Path) -> Result<()> {
        let mut index = self.vector_index.write();
        index.load(path).context("Failed to load vector index")?;
        Ok(())
    }

    /// Add memory to knowledge graph (for associative/causal retrieval)
    pub fn add_to_graph(&self, memory: &Memory) {
        self.graph.write().add_memory(memory);
    }

    /// Record co-activation of memories (Hebbian learning)
    ///
    /// Call this when multiple memories are accessed together in a retrieval result.
    /// Strengthens the associations between them.
    /// Records consolidation events for introspection.
    pub fn record_coactivation(&self, memory_ids: &[MemoryId]) {
        if memory_ids.len() >= 2 {
            let results = self.graph.write().record_coactivation(memory_ids);
            let now = chrono::Utc::now();

            // Record consolidation events for each edge update
            for result in results {
                let res = &result.forward_result;
                if res.is_new {
                    // New edge formed
                    self.record_event(ConsolidationEvent::EdgeFormed {
                        from_memory_id: result.from_id.0.to_string(),
                        to_memory_id: result.to_id.0.to_string(),
                        initial_strength: res.strength_after,
                        reason: EdgeFormationReason::CoRetrieval,
                        timestamp: now,
                    });
                } else {
                    // Existing edge strengthened
                    self.record_event(ConsolidationEvent::EdgeStrengthened {
                        from_memory_id: result.from_id.0.to_string(),
                        to_memory_id: result.to_id.0.to_string(),
                        strength_before: res.strength_before,
                        strength_after: res.strength_after,
                        co_activations: res.activation_count,
                        timestamp: now,
                    });
                }

                // Record LTP event if triggered
                if res.ltp_triggered {
                    self.record_event(ConsolidationEvent::EdgePotentiated {
                        from_memory_id: result.from_id.0.to_string(),
                        to_memory_id: result.to_id.0.to_string(),
                        final_strength: res.strength_after,
                        total_co_activations: res.activation_count,
                        timestamp: now,
                    });
                }
            }
        }
    }

    /// Perform graph maintenance (decay old edges, prune weak ones)
    ///
    /// Call this periodically (e.g., every hour or on user logout)
    /// Records consolidation events for edge pruning.
    /// Returns the number of edges pruned.
    pub fn graph_maintenance(&self) -> usize {
        let prune_results = self.graph.write().maintenance();
        let now = chrono::Utc::now();
        let pruned_count = prune_results.len();

        // Record pruning events
        for result in prune_results {
            self.record_event(ConsolidationEvent::EdgePruned {
                from_memory_id: result.from_id.0.to_string(),
                to_memory_id: result.to_id.0.to_string(),
                final_strength: result.final_strength,
                reason: PruningReason::DecayedBelowThreshold,
                timestamp: now,
            });
        }

        pruned_count
    }

    /// Get memory graph statistics
    pub fn graph_stats(&self) -> MemoryGraphStats {
        self.graph.read().stats()
    }

    /// Get mutable access to memory graph (for Hebbian updates)
    ///
    /// Returns a write guard to the memory graph for recording coactivations
    /// and adding edges. The guard is automatically released when dropped.
    pub fn graph_mut(&self) -> parking_lot::RwLockWriteGuard<'_, MemoryGraph> {
        self.graph.write()
    }

    /// Check if vector index needs rebuild and rebuild if necessary
    ///
    /// Returns true if rebuild was performed
    pub fn auto_rebuild_index_if_needed(&self) -> Result<bool> {
        let mut index = self.vector_index.write();
        index.auto_rebuild_if_needed()
    }

    /// Get vector index degradation info
    pub fn index_health(&self) -> IndexHealth {
        let index = self.vector_index.read();
        IndexHealth {
            total_vectors: index.len(),
            incremental_inserts: index.incremental_insert_count(),
            needs_rebuild: index.needs_rebuild(),
            rebuild_threshold: crate::vector_db::vamana::REBUILD_THRESHOLD,
        }
    }
}

/// Health information about the vector index
#[derive(Debug, Clone)]
pub struct IndexHealth {
    pub total_vectors: usize,
    pub incremental_inserts: usize,
    pub needs_rebuild: bool,
    pub rebuild_threshold: usize,
}

// ============================================================================
// OUTCOME FEEDBACK SYSTEM - Hebbian "Fire Together, Wire Together"
// ============================================================================

/// Outcome of a retrieval operation - used to reinforce or weaken memories
///
/// When memories are retrieved and used to complete a task, this feedback
/// tells the system whether they were helpful, enabling adaptive learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalOutcome {
    /// Memory helped complete the task successfully
    /// Triggers: +importance boost, +association strength, +access count
    Helpful,
    /// Memory was misleading or caused errors
    /// Triggers: -importance penalty, relationship weakening
    Misleading,
    /// Memory was retrieved but not actionably useful
    /// Triggers: +access count only (neutral)
    Neutral,
}

/// Result of a retrieval with tracking for feedback
#[derive(Debug, Clone)]
pub struct TrackedRetrieval {
    /// The memories that were retrieved
    pub memories: Vec<SharedMemory>,
    /// Unique ID for this retrieval (for later feedback)
    pub retrieval_id: String,
    /// Query that produced these results
    pub query_fingerprint: u64,
    /// Timestamp of retrieval
    pub retrieved_at: chrono::DateTime<chrono::Utc>,
}

impl TrackedRetrieval {
    fn new(memories: Vec<SharedMemory>, query: &Query) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        if let Some(text) = &query.query_text {
            text.hash(&mut hasher);
        }

        Self {
            memories,
            retrieval_id: uuid::Uuid::new_v4().to_string(),
            query_fingerprint: hasher.finish(),
            retrieved_at: chrono::Utc::now(),
        }
    }

    /// Get memory IDs for feedback
    pub fn memory_ids(&self) -> Vec<MemoryId> {
        self.memories.iter().map(|m| m.id.clone()).collect()
    }
}

/// Feedback record for a retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalFeedback {
    /// Which retrieval this feedback is for
    pub retrieval_id: String,
    /// The outcome
    pub outcome: RetrievalOutcome,
    /// Optional task context (what was the user trying to do)
    pub task_context: Option<String>,
    /// When feedback was provided
    pub feedback_at: chrono::DateTime<chrono::Utc>,
}

impl RetrievalEngine {
    // ========================================================================
    // OUTCOME FEEDBACK METHODS
    // ========================================================================

    /// Search with tracking for later feedback
    ///
    /// Use this when you want to provide feedback on retrieval quality.
    /// Returns a TrackedRetrieval that can be used with `reinforce_retrieval`.
    pub fn search_tracked(&self, query: &Query, limit: usize) -> Result<TrackedRetrieval> {
        let memories = self.search(query, limit)?;
        Ok(TrackedRetrieval::new(memories, query))
    }

    /// Reinforce memories based on task outcome (core feedback loop)
    ///
    /// This is THE key method that closes the Hebbian loop:
    /// - If outcome is Helpful: strengthen associations, boost importance
    /// - If outcome is Misleading: weaken associations, reduce importance
    /// - If outcome is Neutral: just record access (mild reinforcement)
    ///
    /// Call this after a task completes to indicate which memories helped.
    pub fn reinforce_retrieval(
        &self,
        memory_ids: &[MemoryId],
        outcome: RetrievalOutcome,
    ) -> Result<ReinforcementStats> {
        if memory_ids.is_empty() {
            return Ok(ReinforcementStats::default());
        }

        let mut stats = ReinforcementStats {
            memories_processed: memory_ids.len(),
            ..Default::default()
        };

        match outcome {
            RetrievalOutcome::Helpful => {
                // 1. Strengthen associations between all retrieved memories
                //    "Fire together, wire together"
                if memory_ids.len() >= 2 {
                    self.graph.write().record_coactivation(memory_ids);
                    stats.associations_strengthened = memory_ids.len() * (memory_ids.len() - 1) / 2;
                }

                // 2. Boost importance of helpful memories and PERSIST to storage
                for id in memory_ids {
                    if let Ok(memory) = self.storage.get(id) {
                        // Increment access and apply importance boost
                        memory.record_access();
                        memory.boost_importance(0.05); // +5% importance

                        // PERSIST: Write updated memory back to durable storage
                        if self.storage.update(&memory).is_ok() {
                            stats.importance_boosts += 1;
                        }
                    }
                }
            }
            RetrievalOutcome::Misleading => {
                // Reduce importance of misleading memories and PERSIST to storage
                for id in memory_ids {
                    if let Ok(memory) = self.storage.get(id) {
                        memory.record_access();
                        memory.decay_importance(0.10); // -10% importance

                        // PERSIST: Write updated memory back to durable storage
                        if self.storage.update(&memory).is_ok() {
                            stats.importance_decays += 1;
                        }
                    }
                }
                // Don't strengthen associations for misleading memories
            }
            RetrievalOutcome::Neutral => {
                // Just record access, mild reinforcement - PERSIST to storage
                for id in memory_ids {
                    if let Ok(memory) = self.storage.get(id) {
                        memory.record_access();

                        // PERSIST: Write access update to storage
                        if let Err(e) = self.storage.update(&memory) {
                            tracing::warn!(
                                "Failed to persist access update for memory {}: {}",
                                id.0,
                                e
                            );
                        }
                    }
                }
                // Mild association strengthening for neutral
                if memory_ids.len() >= 2 {
                    let mut graph = self.graph.write();
                    // Only strengthen adjacent pairs (not all pairs)
                    for window in memory_ids.windows(2) {
                        graph.add_edge(&window[0], &window[1]);
                    }
                    stats.associations_strengthened = memory_ids.len() - 1;
                }
            }
        }

        stats.outcome = outcome;
        Ok(stats)
    }

    /// Reinforce using a tracked retrieval (convenience wrapper)
    pub fn reinforce_tracked(
        &self,
        tracked: &TrackedRetrieval,
        outcome: RetrievalOutcome,
    ) -> Result<ReinforcementStats> {
        let ids = tracked.memory_ids();
        self.reinforce_retrieval(&ids, outcome)
    }

    /// Batch reinforce multiple retrievals (for async feedback processing)
    pub fn reinforce_batch(
        &self,
        feedbacks: &[RetrievalFeedback],
        retrieval_memories: &HashMap<String, Vec<MemoryId>>,
    ) -> Result<Vec<ReinforcementStats>> {
        let mut results = Vec::with_capacity(feedbacks.len());

        for feedback in feedbacks {
            if let Some(memory_ids) = retrieval_memories.get(&feedback.retrieval_id) {
                let stats = self.reinforce_retrieval(memory_ids, feedback.outcome)?;
                results.push(stats);
            }
        }

        Ok(results)
    }
}

/// Statistics from a reinforcement operation
#[derive(Debug, Clone, Default)]
pub struct ReinforcementStats {
    /// How many memories were processed
    pub memories_processed: usize,
    /// How many association edges were strengthened
    pub associations_strengthened: usize,
    /// How many importance boosts were applied
    pub importance_boosts: usize,
    /// How many importance decays were applied
    pub importance_decays: usize,
    /// The outcome that triggered this reinforcement
    pub outcome: RetrievalOutcome,
    /// How many persistence operations failed (non-zero indicates data loss risk)
    pub persist_failures: usize,
}

impl Default for RetrievalOutcome {
    fn default() -> Self {
        Self::Neutral
    }
}

/// Memory graph for associative retrieval with Hebbian learning
///
/// Implements simplified synaptic plasticity:
/// - Edge strength increases with co-access (Hebbian strengthening)
/// - Edges decay over time without use
/// - Strong edges are prioritized in traversal
pub(crate) struct MemoryGraph {
    /// Adjacency with edge weights (strength 0.0-1.0)
    adjacency: HashMap<MemoryId, HashMap<MemoryId, EdgeWeight>>,
}

/// Edge weight with Hebbian learning properties
#[derive(Clone)]
struct EdgeWeight {
    /// Current strength (0.0 to 1.0)
    strength: f32,
    /// Number of co-activations
    activation_count: u32,
    /// Last activation timestamp (Unix millis)
    last_activated: i64,
}

/// Result of strengthening an edge (for introspection)
#[derive(Debug)]
pub(crate) struct StrengthenResult {
    /// Was this a new edge?
    pub is_new: bool,
    /// Strength before this operation
    pub strength_before: f32,
    /// Strength after this operation
    pub strength_after: f32,
    /// Total co-activations
    pub activation_count: u32,
    /// Did this trigger long-term potentiation?
    pub ltp_triggered: bool,
}

impl Default for EdgeWeight {
    fn default() -> Self {
        Self {
            strength: EDGE_INITIAL_STRENGTH, // From constants.rs (0.5)
            activation_count: 0, // Start at 0, strengthen() will increment to 1
            last_activated: chrono::Utc::now().timestamp_millis(),
        }
    }
}

impl EdgeWeight {
    /// Hebbian learning constants (local, kept for learning rate and LTP)
    const LEARNING_RATE: f32 = 0.15;
    const LTP_THRESHOLD: u32 = 5; // Lower threshold for memory associations

    /// Strengthen the edge (called when both memories are accessed together)
    /// Returns info about what happened for introspection
    fn strengthen(&mut self) -> StrengthenResult {
        let strength_before = self.strength;
        let was_new = self.activation_count == 0;

        self.activation_count += 1;
        self.last_activated = chrono::Utc::now().timestamp_millis();

        // Hebbian: w_new = w_old + η × (1 - w_old)
        let boost = Self::LEARNING_RATE * (1.0 - self.strength);
        self.strength = (self.strength + boost).min(1.0);

        // Long-term potentiation bonus
        let ltp_triggered = self.activation_count == Self::LTP_THRESHOLD;
        if ltp_triggered {
            self.strength = (self.strength + 0.15).min(1.0);
        }

        StrengthenResult {
            is_new: was_new,
            strength_before,
            strength_after: self.strength,
            activation_count: self.activation_count,
            ltp_triggered,
        }
    }

    /// Apply time-based decay, returns true if edge should be pruned
    fn decay(&mut self) -> bool {
        let now = chrono::Utc::now().timestamp_millis();
        let hours_elapsed = (now - self.last_activated) as f64 / 3_600_000.0;

        if hours_elapsed <= 0.0 {
            return false;
        }

        // Potentiated edges decay slower
        let effective_half_life = if self.activation_count >= Self::LTP_THRESHOLD {
            EDGE_HALF_LIFE_HOURS * 5.0 // 5x slower decay (from constants.rs)
        } else {
            EDGE_HALF_LIFE_HOURS // From constants.rs (24.0 hours)
        };

        let decay_rate = (0.5_f64).ln() / effective_half_life;
        let decay_factor = (decay_rate * hours_elapsed).exp() as f32;
        self.strength *= decay_factor;

        self.strength < EDGE_MIN_STRENGTH // From constants.rs (0.05)
    }
}

/// Result of an edge update (for introspection)
#[derive(Debug)]
pub(crate) struct EdgeUpdateResult {
    pub from_id: MemoryId,
    pub to_id: MemoryId,
    pub forward_result: StrengthenResult,
}

/// Result of a pruning operation (for introspection)
#[derive(Debug)]
pub(crate) struct PruneResult {
    pub from_id: MemoryId,
    pub to_id: MemoryId,
    pub final_strength: f32,
}

impl MemoryGraph {
    fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Add edge between memories (bidirectional) with initial weight
    /// Returns the result of the forward edge update for introspection
    pub(crate) fn add_edge(&mut self, from: &MemoryId, to: &MemoryId) -> EdgeUpdateResult {
        // Forward edge
        let forward_result = self
            .adjacency
            .entry(from.clone())
            .or_default()
            .entry(to.clone())
            .or_default()
            .strengthen();

        // Backward edge (we don't need to return this, symmetric)
        self.adjacency
            .entry(to.clone())
            .or_default()
            .entry(from.clone())
            .or_default()
            .strengthen();

        EdgeUpdateResult {
            from_id: from.clone(),
            to_id: to.clone(),
            forward_result,
        }
    }

    /// Add a memory to the graph (creates edges based on related_memories)
    fn add_memory(&mut self, memory: &Memory) {
        // Add edges to explicitly related memories
        for related_id in &memory.experience.related_memories {
            self.add_edge(&memory.id, related_id);
        }

        // Add edges to causal chain
        for causal_id in &memory.experience.causal_chain {
            self.add_edge(&memory.id, causal_id);
        }
    }

    /// Record co-access of memories (strengthens their connection)
    ///
    /// Note: For large retrievals, we limit to top N memories to avoid O(n²) explosion.
    /// This is a deliberate tradeoff: we still get good association learning while
    /// keeping worst-case complexity bounded to O(MAX_COACTIVATION_SIZE²).
    /// Returns the edge update results for introspection.
    pub(crate) fn record_coactivation(&mut self, memories: &[MemoryId]) -> Vec<EdgeUpdateResult> {
        const MAX_COACTIVATION_SIZE: usize = 20;

        // Limit to top N memories to bound worst-case O(n²) to O(400)
        let memories_to_process = if memories.len() > MAX_COACTIVATION_SIZE {
            &memories[..MAX_COACTIVATION_SIZE]
        } else {
            memories
        };

        let mut results = Vec::new();

        // Strengthen edges between all pairs of co-accessed memories
        for i in 0..memories_to_process.len() {
            for j in (i + 1)..memories_to_process.len() {
                let result = self.add_edge(&memories_to_process[i], &memories_to_process[j]);
                results.push(result);
            }
        }

        results
    }

    /// Find associated memories using weighted graph traversal
    ///
    /// Prioritizes stronger edges (higher activation count, more recent)
    fn find_associations(&self, start: &MemoryId, max_depth: usize) -> Result<Vec<MemoryId>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();

        // Priority queue: (negative_strength for max-heap behavior, depth, id)
        let mut heap = std::collections::BinaryHeap::new();
        heap.push((ordered_float::OrderedFloat(1.0_f32), 0_usize, start.clone()));

        while let Some((_, depth, current)) = heap.pop() {
            if depth > max_depth {
                continue;
            }

            if !visited.insert(current.clone()) {
                continue;
            }

            if current != *start {
                result.push(current.clone());
            }

            if let Some(neighbors) = self.adjacency.get(&current) {
                for (neighbor, weight) in neighbors {
                    if !visited.contains(neighbor) {
                        // Apply decay inline (non-mutating check)
                        let effective_strength = weight.strength;
                        if effective_strength >= EDGE_MIN_STRENGTH {
                            heap.push((
                                ordered_float::OrderedFloat(effective_strength),
                                depth + 1,
                                neighbor.clone(),
                            ));
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Periodic maintenance: decay edges and prune weak ones
    /// Returns list of pruned edges for introspection
    fn maintenance(&mut self) -> Vec<PruneResult> {
        let mut to_remove: Vec<(MemoryId, MemoryId, f32)> = Vec::new();

        for (from_id, neighbors) in &mut self.adjacency {
            for (to_id, weight) in neighbors.iter_mut() {
                let strength_before = weight.strength;
                if weight.decay() {
                    to_remove.push((from_id.clone(), to_id.clone(), strength_before));
                }
            }
        }

        // Remove pruned edges and collect results
        let mut prune_results = Vec::new();
        for (from_id, to_id, final_strength) in to_remove {
            if let Some(neighbors) = self.adjacency.get_mut(&from_id) {
                neighbors.remove(&to_id);
            }
            // Only record one direction (edges are bidirectional)
            if from_id.0 < to_id.0 {
                prune_results.push(PruneResult {
                    from_id,
                    to_id,
                    final_strength,
                });
            }
        }

        prune_results
    }

    /// Get graph statistics
    fn stats(&self) -> MemoryGraphStats {
        let mut edge_count = 0;
        let mut total_strength = 0.0;
        let mut potentiated_count = 0;

        for neighbors in self.adjacency.values() {
            for weight in neighbors.values() {
                edge_count += 1;
                total_strength += weight.strength;
                if weight.activation_count >= EdgeWeight::LTP_THRESHOLD {
                    potentiated_count += 1;
                }
            }
        }

        MemoryGraphStats {
            node_count: self.adjacency.len(),
            edge_count: edge_count / 2, // Bidirectional edges counted once
            avg_strength: if edge_count > 0 {
                total_strength / edge_count as f32
            } else {
                0.0
            },
            potentiated_edges: potentiated_count / 2,
        }
    }
}

/// Statistics about the memory graph
#[derive(Debug, Clone, Default)]
pub struct MemoryGraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_strength: f32,
    pub potentiated_edges: usize,
}

// ============================================================================
// ANTICIPATORY PREFETCH - Context-aware cache warming
// ============================================================================

/// Context signals used to anticipate which memories will be needed
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefetchContext {
    /// Current project/workspace being worked on
    pub project_id: Option<String>,
    /// Current file path being edited
    pub current_file: Option<String>,
    /// Recent entities mentioned
    pub recent_entities: Vec<String>,
    /// Current time of day (for temporal patterns)
    pub hour_of_day: Option<u32>,
    /// Day of week (0=Sunday)
    pub day_of_week: Option<u32>,
    /// Recent query patterns (for predictive prefetch)
    pub recent_queries: Vec<String>,
    /// Current task type (coding, debugging, reviewing)
    pub task_type: Option<String>,
}

impl PrefetchContext {
    /// Create context from RichContext
    pub fn from_rich_context(ctx: &super::types::RichContext) -> Self {
        Self {
            project_id: ctx.project.project_id.clone(),
            current_file: ctx.code.current_file.clone(),
            recent_entities: ctx.conversation.mentioned_entities.clone(),
            hour_of_day: ctx
                .temporal
                .time_of_day
                .as_ref()
                .and_then(|t| t.parse().ok()),
            day_of_week: ctx
                .temporal
                .day_of_week
                .as_ref()
                .and_then(|d| match d.as_str() {
                    "Sunday" => Some(0),
                    "Monday" => Some(1),
                    "Tuesday" => Some(2),
                    "Wednesday" => Some(3),
                    "Thursday" => Some(4),
                    "Friday" => Some(5),
                    "Saturday" => Some(6),
                    _ => None,
                }),
            recent_queries: Vec::new(),
            task_type: ctx.project.current_task.clone(),
        }
    }

    /// Create context from current system state
    pub fn from_current_time() -> Self {
        let now = chrono::Utc::now();
        Self {
            hour_of_day: Some(now.hour()),
            day_of_week: Some(now.weekday().num_days_from_sunday()),
            ..Default::default()
        }
    }
}

/// Result of a prefetch operation
#[derive(Debug, Clone, Default)]
pub struct PrefetchResult {
    /// Memory IDs that were prefetched
    pub prefetched_ids: Vec<MemoryId>,
    /// Why these memories were selected
    pub reason: PrefetchReason,
    /// How many were already cached
    pub cache_hits: usize,
    /// How many were fetched from storage
    pub fetches: usize,
}

/// Reason for prefetching specific memories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum PrefetchReason {
    /// Project-based: memories from same project
    Project(String),
    /// File-based: memories about related files
    RelatedFiles,
    /// Entity-based: memories mentioning same entities
    SharedEntities,
    /// Temporal: memories from similar time patterns
    TemporalPattern,
    /// Association: strongly associated with recent memories
    AssociatedMemories,
    /// Predicted: predicted from query patterns
    QueryPrediction,
    #[default]
    /// Unknown or multiple reasons
    Mixed,
}

/// Anticipatory prefetch engine
///
/// Pre-warms the memory cache based on contextual signals:
/// - Project: "I'm working on auth module" → prefetch auth-related memories
/// - File: "I opened user.rs" → prefetch memories about user.rs and imports
/// - Temporal: "It's Monday morning" → prefetch Monday morning patterns
/// - Association: "I just accessed memory A" → prefetch A's strong associations
pub struct AnticipatoryPrefetch {
    /// Maximum memories to prefetch at once
    max_prefetch: usize,
    /// Minimum association strength for association-based prefetch
    min_association_strength: f32,
    /// Temporal window for pattern matching (hours)
    temporal_window_hours: i64,
}

impl Default for AnticipatoryPrefetch {
    fn default() -> Self {
        Self::new()
    }
}

impl AnticipatoryPrefetch {
    pub fn new() -> Self {
        Self {
            max_prefetch: 20,
            min_association_strength: 0.3,
            temporal_window_hours: 2,
        }
    }

    /// Create with custom limits
    pub fn with_limits(max_prefetch: usize, min_strength: f32, temporal_hours: i64) -> Self {
        Self {
            max_prefetch,
            min_association_strength: min_strength,
            temporal_window_hours: temporal_hours,
        }
    }

    /// Generate prefetch query based on context
    ///
    /// This is the main entry point - given a context, determine what memories
    /// are likely to be needed soon and return a query to fetch them.
    pub fn generate_prefetch_query(&self, context: &PrefetchContext) -> Option<Query> {
        // Priority 1: Project-based prefetch (strongest signal)
        if let Some(project_id) = &context.project_id {
            return Some(self.project_query(project_id));
        }

        // Priority 2: Entity-based prefetch
        if !context.recent_entities.is_empty() {
            return Some(self.entity_query(&context.recent_entities));
        }

        // Priority 3: File-based prefetch
        if let Some(file_path) = &context.current_file {
            return Some(self.file_query(file_path));
        }

        // Priority 4: Temporal pattern prefetch
        if let (Some(hour), Some(day)) = (context.hour_of_day, context.day_of_week) {
            return Some(self.temporal_query(hour, day));
        }

        None
    }

    /// Generate query for project-related memories
    fn project_query(&self, project_id: &str) -> Query {
        Query {
            query_text: Some(format!("project:{}", project_id)),
            max_results: self.max_prefetch,
            retrieval_mode: super::types::RetrievalMode::Similarity,
            ..Default::default()
        }
    }

    /// Generate query for entity-related memories
    fn entity_query(&self, entities: &[String]) -> Query {
        let query_text = entities.join(" ");
        Query {
            query_text: Some(query_text),
            max_results: self.max_prefetch,
            retrieval_mode: super::types::RetrievalMode::Similarity,
            ..Default::default()
        }
    }

    /// Generate query for file-related memories
    fn file_query(&self, file_path: &str) -> Query {
        // Extract filename for broader search
        let filename = std::path::Path::new(file_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(file_path);

        Query {
            query_text: Some(format!("file {} code", filename)),
            max_results: self.max_prefetch,
            retrieval_mode: super::types::RetrievalMode::Similarity,
            ..Default::default()
        }
    }

    /// Generate query for temporal pattern matching
    fn temporal_query(&self, hour: u32, _day: u32) -> Query {
        let now = chrono::Utc::now();

        // Find similar time window
        let start_hour = if hour >= self.temporal_window_hours as u32 {
            hour - self.temporal_window_hours as u32
        } else {
            0
        };
        let end_hour = (hour + self.temporal_window_hours as u32).min(23);

        // Calculate time range for today at similar hours
        let start = now
            .with_hour(start_hour)
            .unwrap_or(now)
            .with_minute(0)
            .unwrap_or(now);
        let end = now
            .with_hour(end_hour)
            .unwrap_or(now)
            .with_minute(59)
            .unwrap_or(now);

        Query {
            time_range: Some((start, end)),
            max_results: self.max_prefetch,
            retrieval_mode: super::types::RetrievalMode::Temporal,
            ..Default::default()
        }
    }

    /// Get memory IDs that should be prefetched based on associations
    ///
    /// Given a set of recently accessed memories, find their strong associations.
    /// Note: This is pub(crate) because MemoryGraph is internal.
    pub(crate) fn association_prefetch_ids(
        &self,
        recent_ids: &[MemoryId],
        graph: &MemoryGraph,
    ) -> Vec<MemoryId> {
        let mut candidates: HashMap<MemoryId, f32> = HashMap::new();
        let recent_set: HashSet<_> = recent_ids.iter().collect();

        // Collect strong associations for each recent memory
        for id in recent_ids {
            if let Some(neighbors) = graph.adjacency.get(id) {
                for (neighbor_id, weight) in neighbors {
                    // Skip if already in recent set
                    if recent_set.contains(neighbor_id) {
                        continue;
                    }

                    // Only include strong associations
                    if weight.strength >= self.min_association_strength {
                        *candidates.entry(neighbor_id.clone()).or_default() += weight.strength;
                    }
                }
            }
        }

        // Sort by total association strength and take top N
        let mut sorted: Vec<_> = candidates.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        sorted
            .into_iter()
            .take(self.max_prefetch)
            .map(|(id, _)| id)
            .collect()
    }

    /// Score how relevant a memory is to the current context
    ///
    /// Higher scores mean more likely to be needed soon.
    pub fn relevance_score(&self, memory: &Memory, context: &PrefetchContext) -> f32 {
        let mut score = 0.0;

        // Project match (strong signal)
        if let Some(project_id) = &context.project_id {
            if let Some(ctx) = &memory.experience.context {
                if ctx.project.project_id.as_ref() == Some(project_id) {
                    score += 0.4;
                }
            }
        }

        // Entity overlap
        let memory_entities: HashSet<_> = memory.experience.entities.iter().collect();
        let context_entities: HashSet<_> = context.recent_entities.iter().collect();
        let overlap = memory_entities.intersection(&context_entities).count();
        if overlap > 0 {
            score += 0.2 * (overlap as f32 / context_entities.len().max(1) as f32);
        }

        // File relevance
        if let Some(current_file) = &context.current_file {
            if memory.experience.content.contains(current_file) {
                score += 0.2;
            }
            // Also check related files in context
            if let Some(ctx) = &memory.experience.context {
                if ctx.code.related_files.iter().any(|f| f == current_file) {
                    score += 0.1;
                }
            }
        }

        // Temporal relevance (same hour of day)
        if let Some(hour) = context.hour_of_day {
            let memory_hour = memory.created_at.hour();
            if (memory_hour as i32 - hour as i32).abs() <= self.temporal_window_hours as i32 {
                score += 0.1;
            }
        }

        // Recency boost (using centralized constants)
        let age_hours = (chrono::Utc::now() - memory.created_at).num_hours();
        if age_hours < PREFETCH_RECENCY_FULL_HOURS {
            score += PREFETCH_RECENCY_FULL_BOOST;
        } else if age_hours < PREFETCH_RECENCY_PARTIAL_HOURS {
            score += PREFETCH_RECENCY_PARTIAL_BOOST;
        }

        score.min(1.0)
    }
}

use chrono::{Datelike, Timelike};
