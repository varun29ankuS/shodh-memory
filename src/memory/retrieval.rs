//! Production-grade retrieval engine for memory search
//! Integrated with Vamana HNSW and MiniLM embeddings for robotics/drones

use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

use super::storage::{MemoryStorage, SearchCriteria};
use super::types::*;
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
        })
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
        // Get or generate query embedding
        let query_embedding = if let Some(embedding) = &query.query_embedding {
            embedding.clone()
        } else if let Some(query_text) = &query.query_text {
            self.embedder
                .encode(query_text)
                .context("Failed to generate query embedding")?
        } else {
            return Ok(Vec::new());
        };

        // Search vector index
        let index = self.vector_index.read();
        let results = index
            .search(&query_embedding, limit * 2) // Get 2x for filtering
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
        // Get or generate query embedding
        let query_embedding = if let Some(embedding) = &query.query_embedding {
            embedding.clone()
        } else if let Some(query_text) = &query.query_text {
            self.embedder
                .encode(query_text)
                .context("Failed to generate query embedding")?
        } else {
            return Ok(Vec::new());
        };

        // Search Vamana index
        let index = self.vector_index.read();
        let results = index
            .search(&query_embedding, limit * 2)
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
    /// This is the CANONICAL filter implementation. All retrieval paths MUST use this
    /// to ensure consistent filtering behavior (mission_id, robot_id, geo, etc.)
    pub fn matches_filters(&self, memory: &Memory, query: &Query) -> bool {
        // === Standard Filters ===

        // Importance filter
        if let Some(threshold) = query.importance_threshold {
            if memory.importance() < threshold {
                return false;
            }
        }

        // Experience type filter
        if let Some(types) = &query.experience_types {
            let matches_type = types.iter().any(|t| {
                std::mem::discriminant(&memory.experience.experience_type)
                    == std::mem::discriminant(t)
            });
            if !matches_type {
                return false;
            }
        }

        // Time range filter
        if let Some((start, end)) = &query.time_range {
            if memory.created_at < *start || memory.created_at > *end {
                return false;
            }
        }

        // === Robotics Filters ===

        // Robot ID filter
        if let Some(ref robot_id) = query.robot_id {
            match &memory.experience.robot_id {
                Some(mem_robot_id) if mem_robot_id == robot_id => {}
                _ => return false,
            }
        }

        // Mission ID filter
        if let Some(ref mission_id) = query.mission_id {
            match &memory.experience.mission_id {
                Some(mem_mission_id) if mem_mission_id == mission_id => {}
                _ => return false,
            }
        }

        // Geo filter (spatial radius)
        if let Some(ref geo_filter) = query.geo_filter {
            match memory.experience.geo_location {
                Some(geo) => {
                    let lat = geo[0];
                    let lon = geo[1];
                    if !geo_filter.contains(lat, lon) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Action type filter
        if let Some(ref action_type) = query.action_type {
            match &memory.experience.action_type {
                Some(mem_action) if mem_action == action_type => {}
                _ => return false,
            }
        }

        // Reward range filter
        if let Some((min_reward, max_reward)) = query.reward_range {
            match memory.experience.reward {
                Some(reward) if reward >= min_reward && reward <= max_reward => {}
                _ => return false,
            }
        }

        true
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

                    let memory_id = memory.id.clone();
                    all_results.insert(memory_id.clone(), memory);
                    *scores.entry(memory_id).or_insert(0.0) += score;
                }
            }
        }

        let mut sorted: Vec<(f32, SharedMemory)> = all_results
            .into_iter()
            .map(|(id, memory)| {
                let score = scores.get(&id).copied().unwrap_or(0.0);
                (score, memory)
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
}

/// Memory graph for associative retrieval
struct MemoryGraph {
    adjacency: HashMap<MemoryId, HashSet<MemoryId>>,
}

impl MemoryGraph {
    fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Add edge between memories (bidirectional)
    fn add_edge(&mut self, from: &MemoryId, to: &MemoryId) {
        self.adjacency
            .entry(from.clone())
            .or_default()
            .insert(to.clone());

        self.adjacency
            .entry(to.clone())
            .or_default()
            .insert(from.clone());
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

        // Note: outcomes are text descriptions, not memory IDs, so we don't add them to graph
    }

    /// Find associated memories using graph traversal
    fn find_associations(&self, start: &MemoryId, max_depth: usize) -> Result<Vec<MemoryId>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = vec![(start.clone(), 0)];

        while let Some((current, depth)) = queue.pop() {
            if depth > max_depth {
                continue;
            }

            if visited.insert(current.clone()) {
                if current != *start {
                    result.push(current.clone());
                }

                if let Some(neighbors) = self.adjacency.get(&current) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            queue.push((neighbor.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}
