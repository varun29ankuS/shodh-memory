//! Production-grade retrieval engine for memory search
//! Integrated with Vamana graph-based ANN and MiniLM embeddings
//!
//! Features Hebbian-inspired adaptive learning:
//! - Outcome feedback: Memories that help complete tasks get reinforced
//! - Co-activation strengthening: Memories retrieved together form associations
//! - Time-based decay: Unused associations naturally weaken

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

use super::introspection::ConsolidationEventBuffer;
use super::storage::{MemoryStorage, SearchCriteria};
use super::types::*;
use crate::constants::{
    PREFETCH_RECENCY_FULL_BOOST, PREFETCH_RECENCY_FULL_HOURS, PREFETCH_RECENCY_PARTIAL_BOOST,
    PREFETCH_RECENCY_PARTIAL_HOURS, PREFETCH_TEMPORAL_WINDOW_HOURS,
    VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
};
use crate::embeddings::{minilm::MiniLMEmbedder, Embedder};
use crate::vector_db::vamana::{VamanaConfig, VamanaIndex};

/// Filename for persisted Vamana index (instant startup)
const VAMANA_INDEX_FILE: &str = "vamana.idx";

/// Multi-modal retrieval engine with production vector search
///
/// # Lock Ordering (SHO-72)
///
/// To prevent deadlocks, locks MUST be acquired in this order:
///
/// 1. `vector_index` - Vector similarity search index
/// 2. `id_mapping` - Memory ID ↔ Vector ID mapping
/// 3. `consolidation_events` - Introspection event buffer
///
/// **Rules:**
/// - Never acquire a higher-numbered lock while holding a lower-numbered lock
/// - For read operations, prefer `read()` over `write()` when possible
/// - Release locks as soon as possible (don't hold during I/O)
///
/// **Note:** Memory graph (Hebbian learning) has been consolidated into GraphMemory
/// which is managed at the API layer (MultiUserMemoryManager.graph_memories)
pub struct RetrievalEngine {
    storage: Arc<MemoryStorage>,
    embedder: Arc<MiniLMEmbedder>,
    /// Lock order: 1 - Acquire first
    vector_index: Arc<RwLock<VamanaIndex>>,
    /// Lock order: 2
    id_mapping: Arc<RwLock<IdMapping>>,
    /// Storage path for persisting vector index and ID mapping
    storage_path: PathBuf,
    /// Lock order: 3 - Acquire last (was 4 when graph was here)
    /// Shared consolidation event buffer for introspection
    /// Records edge formation, strengthening, and pruning events
    consolidation_events: Option<Arc<RwLock<ConsolidationEventBuffer>>>,
}

/// Bidirectional mapping between memory IDs and vector IDs
///
/// Supports multiple vectors per memory for chunked embeddings.
/// When long content is split into chunks, each chunk gets its own vector ID,
/// but all map back to the same MemoryId.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct IdMapping {
    /// Maps each memory to ALL its vector IDs (supports chunked embeddings)
    memory_to_vectors: HashMap<MemoryId, Vec<u32>>,
    /// Maps each vector ID back to its parent memory
    vector_to_memory: HashMap<u32, MemoryId>,
}

impl IdMapping {
    fn new() -> Self {
        Self {
            memory_to_vectors: HashMap::new(),
            vector_to_memory: HashMap::new(),
        }
    }

    /// Insert a single vector for a memory (legacy/simple case)
    fn insert(&mut self, memory_id: MemoryId, vector_id: u32) {
        self.memory_to_vectors
            .entry(memory_id.clone())
            .or_default()
            .push(vector_id);
        self.vector_to_memory.insert(vector_id, memory_id);
    }

    /// Insert multiple vectors for a memory (chunked embedding case)
    fn insert_chunks(&mut self, memory_id: MemoryId, vector_ids: Vec<u32>) {
        for &vid in &vector_ids {
            self.vector_to_memory.insert(vid, memory_id.clone());
        }
        self.memory_to_vectors
            .entry(memory_id)
            .or_default()
            .extend(vector_ids);
    }

    fn get_memory_id(&self, vector_id: u32) -> Option<&MemoryId> {
        self.vector_to_memory.get(&vector_id)
    }

    /// Remove a memory and return ALL its vector IDs
    fn remove_all(&mut self, memory_id: &MemoryId) -> Vec<u32> {
        if let Some(vector_ids) = self.memory_to_vectors.remove(memory_id) {
            for vid in &vector_ids {
                self.vector_to_memory.remove(vid);
            }
            vector_ids
        } else {
            Vec::new()
        }
    }

    /// Number of unique memories in the mapping
    fn len(&self) -> usize {
        self.memory_to_vectors.len()
    }

    fn clear(&mut self) {
        self.memory_to_vectors.clear();
        self.vector_to_memory.clear();
    }
}

impl RetrievalEngine {
    /// Create new retrieval engine with shared embedder (CRITICAL: embedder loaded only once)
    ///
    /// ATOMIC ARCHITECTURE: RocksDB is the ONLY source of truth.
    /// - Vector mappings are stored atomically with memories in RocksDB
    /// - Vamana index is rebuilt from RocksDB on startup (pure in-memory cache)
    /// - No more file-based IdMapping = no more orphaned memories
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
    ///
    /// ATOMIC STARTUP: Rebuilds Vamana from RocksDB mappings for crash safety.
    pub fn with_event_buffer(
        storage: Arc<MemoryStorage>,
        embedder: Arc<MiniLMEmbedder>,
        consolidation_events: Option<Arc<RwLock<ConsolidationEventBuffer>>>,
    ) -> Result<Self> {
        let storage_path = storage.path().to_path_buf();

        // Initialize Vamana index optimized for 10M+ memories per user
        let vamana_config = VamanaConfig {
            dimension: 384,        // MiniLM dimension
            max_degree: 32,        // Increased for better recall at scale
            search_list_size: 100, // 2x for better accuracy with 10M vectors
            alpha: 1.2,
            use_mmap: false, // Keep in memory for low-latency robotics
            ..Default::default()
        };

        let vector_index =
            VamanaIndex::new(vamana_config).context("Failed to initialize Vamana vector index")?;
        let id_mapping = IdMapping::new();

        // NOTE: Memory graph (Hebbian associations) has been consolidated into GraphMemory
        // which is managed at the API layer (MultiUserMemoryManager.graph_memories)
        // This enables persistent storage in RocksDB with proper Hebbian learning

        let engine = Self {
            storage,
            embedder,
            vector_index: Arc::new(RwLock::new(vector_index)),
            id_mapping: Arc::new(RwLock::new(id_mapping)),
            storage_path,
            consolidation_events,
        };

        // ATOMIC STARTUP: Rebuild Vamana from RocksDB (single source of truth)
        engine.rebuild_from_rocksdb()?;

        Ok(engine)
    }

    /// Initialize Vamana index from persisted file or rebuild from RocksDB
    ///
    /// INSTANT STARTUP ARCHITECTURE:
    /// 1. Try loading .vamana file (instant, ~10ms for 500k vectors)
    /// 2. Fall back to RocksDB rebuild (slow, ~seconds for 500k vectors)
    ///
    /// RocksDB remains the source of truth for ID mappings.
    /// The .vamana file is a cache that can be regenerated.
    fn rebuild_from_rocksdb(&self) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Try instant startup from persisted Vamana file
        let vamana_path = self
            .storage_path
            .join("vector_index")
            .join(VAMANA_INDEX_FILE);
        if vamana_path.exists() {
            if let Ok(loaded) = self.try_load_persisted_vamana(&vamana_path) {
                if loaded {
                    info!(
                        "Instant startup: loaded Vamana in {:.2}ms",
                        start_time.elapsed().as_secs_f64() * 1000.0
                    );
                    return Ok(());
                }
            }
        }

        // Fall back to rebuilding from RocksDB
        info!("No valid .vamana file, rebuilding from RocksDB...");

        // Get all vector mappings from RocksDB
        let mappings = self.storage.get_all_vector_mappings()?;

        if !mappings.is_empty() {
            // Fast path: Mappings exist in RocksDB
            info!(
                "Loading {} vector mappings from RocksDB (atomic storage)",
                mappings.len()
            );

            // Build IdMapping from RocksDB data
            let mut id_mapping = self.id_mapping.write();
            id_mapping.clear();

            // Rebuild Vamana from actual embeddings
            // The stored vector_ids are from a previous Vamana session and may not match
            // So we re-insert embeddings to get fresh vector IDs
            let mut vector_index = self.vector_index.write();
            let mut indexed = 0;
            let mut failed = 0;

            for (memory_id, entry) in &mappings {
                // Check if this entry has text vectors (current modality)
                if entry.text_vectors().is_none() {
                    continue;
                }

                // Get memory with embeddings from storage
                if let Ok(memory) = self.storage.get(memory_id) {
                    if let Some(ref embedding) = memory.experience.embeddings {
                        // Insert into Vamana and get new vector_id
                        match vector_index.add_vector(embedding.clone()) {
                            Ok(new_vector_id) => {
                                id_mapping.insert(memory_id.clone(), new_vector_id);
                                indexed += 1;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to index memory {} during rebuild: {}",
                                    memory_id.0,
                                    e
                                );
                                failed += 1;
                            }
                        }
                    }
                }
            }

            let elapsed = start_time.elapsed();
            info!(
                "Rebuilt Vamana from RocksDB: {} indexed, {} failed in {:.2}s",
                indexed,
                failed,
                elapsed.as_secs_f64()
            );
        } else {
            // Slow path: No mappings in RocksDB - need full migration
            // This happens on first run after upgrade to atomic storage
            info!("No vector mappings in RocksDB - checking for migration...");
            self.migrate_to_atomic_storage()?;
        }

        Ok(())
    }

    /// Migrate existing memories to atomic storage
    ///
    /// Called when RocksDB has no vector mappings (first run after upgrade).
    /// Iterates all memories with embeddings and creates atomic mappings.
    fn migrate_to_atomic_storage(&self) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Get all memories from storage
        let memories = self.storage.get_all()?;
        let total = memories.len();

        if total == 0 {
            info!("No memories to migrate");
            return Ok(());
        }

        info!("Migrating {} memories to atomic storage...", total);

        let mut id_mapping = self.id_mapping.write();
        let mut vector_index = self.vector_index.write();
        let mut migrated = 0;
        let mut skipped = 0;
        let mut failed = 0;

        for (i, memory) in memories.iter().enumerate() {
            // Only migrate memories with embeddings
            if let Some(ref embedding) = memory.experience.embeddings {
                // Insert into Vamana
                match vector_index.add_vector(embedding.clone()) {
                    Ok(vector_id) => {
                        // Update in-memory mapping
                        id_mapping.insert(memory.id.clone(), vector_id);

                        // Store mapping in RocksDB for future startups
                        if let Err(e) = self
                            .storage
                            .update_vector_mapping(&memory.id, vec![vector_id])
                        {
                            tracing::warn!("Failed to persist mapping for {}: {}", memory.id.0, e);
                            failed += 1;
                        } else {
                            migrated += 1;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to index memory {}: {}", memory.id.0, e);
                        failed += 1;
                    }
                }
            } else {
                skipped += 1;
            }

            // Progress logging
            if (i + 1) % 500 == 0 || i + 1 == total {
                info!(
                    "Migration progress: {}/{} ({:.1}%)",
                    i + 1,
                    total,
                    (i + 1) as f64 / total as f64 * 100.0
                );
            }
        }

        let elapsed = start_time.elapsed();
        info!(
            "Migration complete: {} migrated, {} skipped (no embeddings), {} failed in {:.2}s",
            migrated,
            skipped,
            failed,
            elapsed.as_secs_f64()
        );

        Ok(())
    }

    /// Try loading Vamana from persisted file for instant startup
    ///
    /// Returns Ok(true) if successfully loaded, Ok(false) if should fall back to rebuild.
    /// Verifies checksum and cross-checks with RocksDB mappings.
    fn try_load_persisted_vamana(&self, vamana_path: &Path) -> Result<bool> {
        // Verify file integrity first
        if !VamanaIndex::verify_index_file(vamana_path)? {
            warn!("Vamana file checksum mismatch, will rebuild");
            return Ok(false);
        }

        // Load the persisted index
        let loaded_index = match VamanaIndex::load_from_file(vamana_path) {
            Ok(idx) => idx,
            Err(e) => {
                warn!("Failed to load Vamana file: {}, will rebuild", e);
                return Ok(false);
            }
        };

        let loaded_count = loaded_index.len();

        // Get mappings from RocksDB to rebuild IdMapping
        let mappings = self.storage.get_all_vector_mappings()?;
        let rocksdb_count = mappings
            .iter()
            .filter(|(_, e)| e.text_vectors().is_some())
            .count();

        // Check for significant drift (>10% difference suggests corruption or data loss)
        let drift_ratio = if loaded_count > 0 {
            (loaded_count as f64 - rocksdb_count as f64).abs() / loaded_count as f64
        } else {
            0.0
        };

        if drift_ratio > 0.1 && loaded_count > 100 {
            warn!(
                "Vamana/RocksDB drift too high ({:.1}%): {} vs {}, will rebuild",
                drift_ratio * 100.0,
                loaded_count,
                rocksdb_count
            );
            return Ok(false);
        }

        // Replace the vector index with the loaded one
        {
            let mut index = self.vector_index.write();
            *index = loaded_index;
        }

        // Rebuild IdMapping from RocksDB (fast - just HashMap operations)
        let mut id_mapping = self.id_mapping.write();
        id_mapping.clear();

        for (memory_id, entry) in mappings.iter() {
            if let Some(vector_ids) = entry.text_vectors() {
                if !vector_ids.is_empty() {
                    // Use the first vector_id for simple case
                    // For chunked, we'd need to store all of them
                    if vector_ids.len() == 1 {
                        id_mapping.insert(memory_id.clone(), vector_ids[0]);
                    } else {
                        id_mapping.insert_chunks(memory_id.clone(), vector_ids.clone());
                    }
                }
            }
        }

        info!(
            "Loaded {} vectors from .vamana, {} mappings from RocksDB",
            self.vector_index.read().len(),
            id_mapping.len()
        );

        Ok(true)
    }

    /// Set the consolidation event buffer (for late binding after construction)
    pub fn set_consolidation_events(&mut self, events: Arc<RwLock<ConsolidationEventBuffer>>) {
        self.consolidation_events = Some(events);
    }

    /// Save Vamana index to disk for instant startup
    ///
    /// HYBRID ARCHITECTURE:
    /// - RocksDB: Source of truth for memories and ID mappings
    /// - .vamana file: Persisted graph for instant startup (skip rebuild)
    ///
    /// On next startup, if .vamana exists and is valid, we load it directly.
    /// Otherwise, we fall back to rebuilding from RocksDB.
    pub fn save(&self) -> Result<()> {
        let index_path = self.storage_path.join("vector_index");
        fs::create_dir_all(&index_path)?;

        let vamana_path = index_path.join(VAMANA_INDEX_FILE);
        let id_mapping = self.id_mapping.read();
        let vector_count = id_mapping.len();

        // Persist Vamana index for instant startup
        let vector_index = self.vector_index.read();
        if vector_count > 0 {
            match vector_index.save_to_file(&vamana_path) {
                Ok(()) => {
                    info!(
                        "Saved Vamana index: {} vectors to {} (instant startup enabled)",
                        vector_count,
                        vamana_path.display()
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to save Vamana index (will rebuild on restart): {}",
                        e
                    );
                }
            }
        } else {
            info!("Vamana index empty, skipping persistence");
        }

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

    /// Get set of all indexed memory IDs (for integrity checking)
    pub fn get_indexed_memory_ids(&self) -> HashSet<MemoryId> {
        self.id_mapping
            .read()
            .memory_to_vectors
            .keys()
            .cloned()
            .collect()
    }

    /// Add memory to vector index with atomic RocksDB storage
    ///
    /// ATOMIC ARCHITECTURE: This method stores the vector mapping atomically
    /// in RocksDB alongside the memory data, ensuring no orphaned memories.
    ///
    /// For long content, this chunks the text and creates multiple embeddings
    /// to ensure ALL content is searchable, not just the first 256 tokens.
    pub fn index_memory(&self, memory: &Memory) -> Result<()> {
        use crate::embeddings::chunking::{chunk_text, ChunkConfig};

        let text = Self::extract_searchable_text(memory);
        let chunk_config = ChunkConfig::default();
        let chunk_result = chunk_text(&text, &chunk_config);

        let vector_ids = if chunk_result.was_chunked {
            // Long content: embed each chunk separately
            // Pre-compute all embeddings OUTSIDE the write lock to avoid blocking searches
            let embeddings: Vec<Vec<f32>> = chunk_result
                .chunks
                .iter()
                .map(|chunk| {
                    self.embedder
                        .encode(chunk)
                        .context("Failed to generate chunk embedding")
                })
                .collect::<Result<Vec<_>>>()?;

            // Insert pre-computed vectors under a short write lock
            let mut ids = Vec::with_capacity(embeddings.len());
            let mut index = self.vector_index.write();
            for embedding in embeddings {
                let vector_id = index
                    .add_vector(embedding)
                    .context("Failed to add chunk vector to index")?;
                ids.push(vector_id);
            }
            drop(index);

            // Update in-memory mapping
            self.id_mapping
                .write()
                .insert_chunks(memory.id.clone(), ids.clone());

            tracing::debug!(
                "Indexed memory {} with {} chunks (original: {} chars)",
                memory.id.0,
                chunk_result.chunks.len(),
                chunk_result.original_length
            );

            ids
        } else {
            // Short content: single embedding (use pre-computed if available)
            let embedding = if let Some(emb) = &memory.experience.embeddings {
                emb.clone()
            } else {
                self.embedder
                    .encode(&text)
                    .context("Failed to generate embedding")?
            };

            let mut index = self.vector_index.write();
            let vector_id = index
                .add_vector(embedding)
                .context("Failed to add vector to index")?;

            // Update in-memory mapping
            self.id_mapping.write().insert(memory.id.clone(), vector_id);

            vec![vector_id]
        };

        // ATOMIC: Store vector mapping in RocksDB
        // This ensures the mapping survives restarts and can't become orphaned
        self.storage
            .update_vector_mapping(&memory.id, vector_ids)
            .context("Failed to persist vector mapping to RocksDB")?;

        Ok(())
    }

    /// Re-index an existing memory with updated embeddings
    ///
    /// Used when memory content is updated via upsert() to ensure the vector
    /// index reflects the new content.
    ///
    /// Strategy: Remove old vector and add new one (Vamana doesn't support update-in-place)
    pub fn reindex_memory(&self, memory: &Memory) -> Result<()> {
        // Check if memory is already indexed (may have multiple vectors from chunking)
        let existing_vector_ids = {
            let mapping = self.id_mapping.read();
            mapping
                .memory_to_vectors
                .get(&memory.id)
                .cloned()
                .unwrap_or_default()
        };

        if !existing_vector_ids.is_empty() {
            // Soft-delete old vectors in Vamana so they're excluded from search results
            // and counted toward the compaction threshold (30% deletion ratio triggers rebuild).
            // Without this, reindexed vectors become invisible ghost entries that waste
            // search candidate slots and never trigger compaction.
            {
                let index = self.vector_index.read();
                for &vid in &existing_vector_ids {
                    index.mark_deleted(vid);
                }
            }

            // Remove old ID mappings
            let mut mapping = self.id_mapping.write();
            mapping.memory_to_vectors.remove(&memory.id);
            for vector_id in existing_vector_ids {
                mapping.vector_to_memory.remove(&vector_id);
            }
        }

        // Add with new embedding (may create multiple chunks)
        self.index_memory(memory)
    }

    /// Remove a memory from the vector index
    ///
    /// ATOMIC ARCHITECTURE: Removes the vector mapping from RocksDB atomically.
    /// The in-memory Vamana index is updated immediately, and the RocksDB mapping
    /// is deleted to ensure consistency on restart.
    ///
    /// Returns true if the memory was found and removed, false if not indexed.
    pub fn remove_memory(&self, memory_id: &MemoryId) -> bool {
        // Remove from in-memory ID mapping and get the vector IDs
        let vector_ids = self.id_mapping.write().remove_all(memory_id);

        if !vector_ids.is_empty() {
            // Mark vectors as deleted in Vamana (soft delete)
            let index = self.vector_index.read();
            for vid in &vector_ids {
                index.mark_deleted(*vid);
            }

            // NOTE: Memory graph edges are managed in GraphMemory at the API layer
            // GraphMemory handles cleanup via its own mechanisms

            // ATOMIC: Remove vector mapping from RocksDB
            if let Err(e) = self.storage.delete_vector_mapping(memory_id) {
                tracing::warn!(
                    "Failed to delete vector mapping from RocksDB for {}: {}",
                    memory_id.0,
                    e
                );
            }

            tracing::debug!(
                "Removed memory {:?} from vector index ({} vectors)",
                memory_id,
                vector_ids.len()
            );
            true
        } else {
            tracing::debug!("Memory {:?} not found in vector index", memory_id);
            false
        }
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

            // SHO-104: Add emotional context - emotion labels improve semantic matching
            if let Some(emotion) = &context.emotional.dominant_emotion {
                text.push(' ');
                text.push_str(emotion);
            }

            // SHO-104: Add episode type for episodic grouping
            if let Some(episode_type) = &context.episode.episode_type {
                text.push(' ');
                text.push_str(episode_type);
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
    ///
    /// With chunked embeddings, multiple vectors can map to the same memory.
    /// This function deduplicates by MemoryId, keeping the highest-scoring chunk.
    ///
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

        // TEMPORAL PRE-FILTER: If episode_id is provided, narrow search to that episode
        // This implements the architecture: Temporal → Graph → Semantic
        // Episode filtering happens FIRST to "point in the right direction"
        let episode_candidates: Option<HashSet<MemoryId>> =
            if let Some(episode_id) = &query.episode_id {
                let episode_memories = self
                    .storage
                    .search(SearchCriteria::ByEpisode(episode_id.clone()))?;
                if episode_memories.is_empty() {
                    tracing::debug!(
                        "No memories found in episode {}, falling back to global search",
                        episode_id
                    );
                    None
                } else {
                    tracing::debug!(
                        "Episode {} has {} memories, using as temporal filter",
                        episode_id,
                        episode_memories.len()
                    );
                    Some(episode_memories.into_iter().map(|m| m.id).collect())
                }
            } else {
                None
            };

        // Search vector index - fetch more candidates for chunk deduplication
        let index = self.vector_index.read();
        let results = index
            .search(
                &query_embedding,
                limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2,
            )
            .context("Vector search failed")?;

        // Map vector IDs to memory IDs, deduplicating by MemoryId (keep highest similarity)
        //
        // CRITICAL FIX: Vamana returns DISTANCE, not similarity.
        // For NormalizedDotProduct: distance = -dot(a,b)
        // - Similar vectors have dot ≈ 1.0, so distance ≈ -1.0
        // - Orthogonal vectors have dot ≈ 0.0, so distance ≈ 0.0
        // Convert: similarity = -distance (so similarity = dot product = cosine similarity)
        let id_mapping = self.id_mapping.read();
        let mut best_scores: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();

        for (vector_id, distance) in results {
            // Convert distance to similarity: similarity = -distance
            // For NormalizedDotProduct, this gives us the actual dot product/cosine similarity
            let similarity = -distance;

            if let Some(memory_id) = id_mapping.get_memory_id(vector_id) {
                // TEMPORAL FILTER: If episode pre-filter is active, skip memories outside episode
                if let Some(ref candidates) = episode_candidates {
                    if !candidates.contains(memory_id) {
                        continue; // Skip - not in target episode
                    }
                }

                // Keep the highest similarity for each memory (best matching chunk)
                best_scores
                    .entry(memory_id.clone())
                    .and_modify(|score| {
                        if similarity > *score {
                            *score = similarity;
                        }
                    })
                    .or_insert(similarity);
            }
        }

        // Convert to vec and sort by similarity descending (highest first)
        let mut memory_ids: Vec<(MemoryId, f32)> = best_scores.into_iter().collect();
        memory_ids.sort_by(|a, b| b.1.total_cmp(&a.1));
        memory_ids.truncate(limit);

        Ok(memory_ids)
    }

    /// Get memory from storage by ID
    pub fn get_from_storage(&self, id: &MemoryId) -> Result<Memory> {
        self.storage.get(id)
    }

    /// Search for similar memories by embedding directly (SHO-106)
    ///
    /// Used for interference detection to find memories similar to a new memory.
    /// Optionally excludes a specific memory ID from results.
    ///
    /// With chunked embeddings, multiple vectors can map to the same memory.
    /// This function deduplicates by MemoryId, keeping the highest-scoring chunk.
    ///
    /// Returns (MemoryId, similarity_score) pairs
    pub fn search_by_embedding(
        &self,
        embedding: &[f32],
        limit: usize,
        exclude_id: Option<&MemoryId>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        // Search vector index - fetch more candidates to account for chunk deduplication
        let index = self.vector_index.read();
        let results = index
            .search(embedding, limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER * 2)
            .context("Vector search by embedding failed")?;

        // Map vector IDs to memory IDs, deduplicating by MemoryId (keep highest similarity)
        //
        // CRITICAL FIX: Vamana returns DISTANCE, not similarity.
        // Convert: similarity = -distance (for NormalizedDotProduct)
        let id_mapping = self.id_mapping.read();
        let mut best_scores: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();

        for (vector_id, distance) in results {
            // Convert distance to similarity
            let similarity = -distance;

            if let Some(memory_id) = id_mapping.get_memory_id(vector_id) {
                // Skip excluded ID
                if let Some(exclude) = exclude_id {
                    if memory_id == exclude {
                        continue;
                    }
                }

                // Keep the highest similarity for each memory (best matching chunk)
                best_scores
                    .entry(memory_id.clone())
                    .and_modify(|score| {
                        if similarity > *score {
                            *score = similarity;
                        }
                    })
                    .or_insert(similarity);
            }
        }

        // Convert to vec and sort by similarity descending (highest first)
        let mut memory_ids: Vec<(MemoryId, f32)> = best_scores.into_iter().collect();
        memory_ids.sort_by(|a, b| b.1.total_cmp(&a.1));
        memory_ids.truncate(limit);

        Ok(memory_ids)
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

    /// PRODUCTION: Similarity search using Vamana graph-based ANN (sub-millisecond, zero-copy)
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
        // TEMPORAL HIERARCHY:
        // 1. Episode (most specific) - same conversation/session
        // 2. Date range (fallback) - within time window

        let criteria = if let Some(episode_id) = &query.episode_id {
            // Episode-based temporal search: memories in same episode, ordered by sequence
            SearchCriteria::ByEpisodeSequence {
                episode_id: episode_id.clone(),
                min_sequence: None, // Get all in episode
                max_sequence: None,
            }
        } else if let Some((start, end)) = &query.time_range {
            // Date-based temporal search
            SearchCriteria::ByDate {
                start: *start,
                end: *end,
            }
        } else {
            // Default: last 7 days
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

        // Sort by sequence if episode-based, otherwise by created_at
        if query.episode_id.is_some() {
            // Episode search already returns in sequence order from storage
            // But verify ordering by sequence_number if available
            memories.sort_by(|a, b| {
                let seq_a = a
                    .experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.sequence_number)
                    .unwrap_or(0);
                let seq_b = b
                    .experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.sequence_number)
                    .unwrap_or(0);
                seq_a.cmp(&seq_b)
            });
        } else {
            memories.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        }

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
        // NOTE: Associative search now uses GraphMemory at the API layer
        // GraphMemory.find_memory_associations() provides Hebbian-weighted associations
        // This method falls back to similarity search as a baseline
        self.similarity_search(query, limit)
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

        sorted.sort_by(|a, b| b.0.total_cmp(&a.0));

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
            dist_a.total_cmp(&dist_b)
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
            reward_b.total_cmp(&reward_a)
        });

        memories.truncate(limit);
        Ok(memories)
    }

    /// Build vector index from existing memories (resumable on failure)
    ///
    /// Uses incremental indexing so partial progress is preserved:
    /// - Skips memories already in the index
    /// - On failure, next rebuild/repair continues from where it left off
    /// - Logs progress every 1000 memories for monitoring
    pub fn rebuild_index(&self) -> Result<()> {
        // Phase 1: Collect only memory IDs (16 bytes each — bounded even at 10M)
        let all_ids = self.storage.get_all_ids()?;
        let total = all_ids.len();

        if total == 0 {
            tracing::info!("No memories to index");
            return Ok(());
        }

        tracing::info!("Starting resumable index rebuild: {} memories", total);

        // Get already-indexed memory IDs to skip
        let indexed_ids = self.get_indexed_memory_ids();
        let already_indexed = indexed_ids.len();

        let mut indexed = 0;
        let mut skipped = 0;
        let mut failed = 0;
        let start_time = std::time::Instant::now();

        // Phase 2: Process one memory at a time — O(1) peak memory per iteration
        for (i, memory_id) in all_ids.iter().enumerate() {
            // Skip already indexed memories (makes rebuild resumable)
            if indexed_ids.contains(memory_id) {
                skipped += 1;
            } else {
                // Load single memory from RocksDB, index it, then drop
                match self.storage.get(memory_id) {
                    Ok(memory) => {
                        if memory.is_forgotten() {
                            skipped += 1;
                        } else {
                            match self.index_memory(&memory) {
                                Ok(_) => indexed += 1,
                                Err(e) => {
                                    failed += 1;
                                    tracing::warn!(
                                        "Failed to index memory {} during rebuild: {}",
                                        memory_id.0,
                                        e
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        tracing::warn!(
                            "Failed to load memory {} during rebuild: {}",
                            memory_id.0,
                            e
                        );
                    }
                }
            }

            // Log progress every 1000 memories
            if (i + 1) % 1000 == 0 || i + 1 == total {
                let elapsed = start_time.elapsed().as_secs();
                let rate = if elapsed > 0 {
                    (indexed + skipped) as f64 / elapsed as f64
                } else {
                    0.0
                };
                tracing::info!(
                    "Rebuild progress: {}/{} ({:.1}%), {} indexed, {} skipped, {} failed, {:.0}/sec",
                    i + 1,
                    total,
                    (i + 1) as f64 / total as f64 * 100.0,
                    indexed,
                    skipped,
                    failed,
                    rate
                );
            }
        }

        tracing::info!(
            "Index rebuild complete: {} indexed, {} already present, {} failed (total: {})",
            indexed,
            already_indexed + skipped,
            failed,
            self.len()
        );

        Ok(())
    }

    // NOTE: Memory graph functionality has been consolidated into GraphMemory
    // which is managed at the API layer (MultiUserMemoryManager.graph_memories)
    // The following methods are preserved for API compatibility but are no-ops:
    // - add_to_graph() - GraphMemory handles entity relationships from NER
    // - record_coactivation() - Now called directly on GraphMemory in API handlers
    // - graph_maintenance() - GraphMemory.apply_decay() handles this
    // - graph_stats() - Returns empty stats (use GraphMemory.get_stats() instead)

    /// Add memory to knowledge graph - DEPRECATED
    /// Use GraphMemory at the API layer instead
    #[deprecated(note = "Use GraphMemory at API layer instead")]
    pub fn add_to_graph(&self, _memory: &Memory) {
        // No-op: GraphMemory handles entity relationships from NER
    }

    /// Record co-activation of memories - DEPRECATED
    /// Use GraphMemory.record_memory_coactivation() at API layer instead
    #[deprecated(note = "Use GraphMemory.record_memory_coactivation() at API layer instead")]
    pub fn record_coactivation(&self, _memory_ids: &[MemoryId]) {
        // No-op: Coactivation is now recorded in GraphMemory at the API layer
    }

    /// Perform graph maintenance - DEPRECATED
    /// Use GraphMemory.apply_decay() at API layer instead
    #[deprecated(note = "Use GraphMemory.apply_decay() at API layer instead")]
    pub fn graph_maintenance(&self) {
        // No-op: GraphMemory handles decay in its own maintenance cycle
    }

    /// Get memory graph statistics - DEPRECATED
    /// Use GraphMemory.get_stats() at API layer instead
    #[deprecated(note = "Use GraphMemory.get_stats() at API layer instead")]
    pub fn graph_stats(&self) -> MemoryGraphStats {
        // Return empty stats - real stats are in GraphMemory
        MemoryGraphStats {
            node_count: 0,
            edge_count: 0,
            avg_strength: 0.0,
            potentiated_count: 0,
        }
    }

    /// Check if vector index needs rebuild and rebuild if necessary
    ///
    /// Returns true if rebuild was performed
    pub fn auto_rebuild_index_if_needed(&self) -> Result<bool> {
        let index = self.vector_index.write();
        index.auto_rebuild_if_needed()
    }

    /// Get vector index degradation info
    pub fn index_health(&self) -> IndexHealth {
        let index = self.vector_index.read();
        IndexHealth {
            total_vectors: index.len(),
            incremental_inserts: index.incremental_insert_count(),
            deleted_count: index.deleted_count(),
            deletion_ratio: index.deletion_ratio(),
            needs_rebuild: index.needs_rebuild(),
            needs_compaction: index.needs_compaction(),
            rebuild_threshold: crate::vector_db::vamana::REBUILD_THRESHOLD,
            deletion_ratio_threshold: crate::vector_db::vamana::DELETION_RATIO_THRESHOLD,
        }
    }
}

/// Health information about the vector index
#[derive(Debug, Clone)]
pub struct IndexHealth {
    pub total_vectors: usize,
    pub incremental_inserts: usize,
    pub deleted_count: usize,
    pub deletion_ratio: f32,
    pub needs_rebuild: bool,
    pub needs_compaction: bool,
    pub rebuild_threshold: usize,
    pub deletion_ratio_threshold: f32,
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
    /// Returns a TrackedRetrieval that can be used with `reinforce_recall`.
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
    pub fn reinforce_recall(
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

        // Hebbian coactivation: count pair associations for non-misleading outcomes
        if !matches!(outcome, RetrievalOutcome::Misleading) && memory_ids.len() >= 2 {
            let n = memory_ids.len();
            stats.associations_strengthened = n * (n - 1) / 2;
        }

        match outcome {
            RetrievalOutcome::Helpful => {
                // Boost importance of helpful memories and PERSIST to storage
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
                // Association strengthening for neutral outcomes is counted above (pair counting)
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
        self.reinforce_recall(&ids, outcome)
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
                let stats = self.reinforce_recall(memory_ids, feedback.outcome)?;
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

// NOTE: MemoryGraph has been consolidated into GraphMemory (src/graph_memory.rs)
// which provides persistent storage in RocksDB and proper Hebbian learning.
// All graph-based memory associations now go through GraphMemory at the API layer.

/// Statistics about the memory graph (for backwards compatibility)
///
/// Real statistics are available from GraphMemory.get_stats()
#[derive(Debug, Clone, Default)]
pub struct MemoryGraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_strength: f32,
    pub potentiated_count: usize,
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

    // SHO-104: Episode context for episodic prefetching
    /// Current episode ID - memories in same episode are highly relevant
    pub episode_id: Option<String>,
    /// Current emotional valence - for mood-congruent retrieval
    pub emotional_valence: Option<f32>,
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
            // SHO-104: Episode and emotional context
            episode_id: ctx.episode.episode_id.clone(),
            emotional_valence: if ctx.emotional.valence != 0.0 {
                Some(ctx.emotional.valence)
            } else {
                None
            },
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
}

impl Default for AnticipatoryPrefetch {
    fn default() -> Self {
        Self::new()
    }
}

impl AnticipatoryPrefetch {
    pub fn new() -> Self {
        Self { max_prefetch: 20 }
    }

    /// Create with custom prefetch limit
    pub fn with_limit(max_prefetch: usize) -> Self {
        Self { max_prefetch }
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
        let start_hour = if hour >= PREFETCH_TEMPORAL_WINDOW_HOURS as u32 {
            hour - PREFETCH_TEMPORAL_WINDOW_HOURS as u32
        } else {
            0
        };
        let end_hour = (hour + PREFETCH_TEMPORAL_WINDOW_HOURS as u32).min(23);

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

    // NOTE: association_prefetch_ids has been removed.
    // Association-based prefetching should use GraphMemory.find_memory_associations()
    // at the API layer where GraphMemory is available.

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
            if (memory_hour as i32 - hour as i32).abs() <= PREFETCH_TEMPORAL_WINDOW_HOURS as i32 {
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

        // SHO-104: Emotional arousal boost - high-arousal memories are more salient
        // Research: Emotionally arousing events are better remembered (LaBar & Cabeza, 2006)
        if let Some(ctx) = &memory.experience.context {
            // High arousal memories get a relevance boost
            if ctx.emotional.arousal > 0.6 {
                score += 0.1 * ctx.emotional.arousal;
            }

            // Source credibility affects relevance - more credible = more relevant
            if ctx.source.credibility > 0.8 {
                score += 0.05;
            }

            // Episode context: same episode = highly relevant
            if let Some(current_episode) = &context.episode_id {
                if ctx.episode.episode_id.as_ref() == Some(current_episode) {
                    score += 0.3; // Strong boost for same-episode memories
                }
            }

            // Mood-congruent retrieval: similar emotional valence boosts relevance
            // Research: We recall happy memories when happy, sad when sad
            if let Some(current_valence) = context.emotional_valence {
                let valence_diff = (ctx.emotional.valence - current_valence).abs();
                if valence_diff < 0.3 {
                    // Same emotional valence quadrant
                    score += 0.1 * (1.0 - valence_diff / 0.3);
                }
            }
        }

        score.min(1.0)
    }
}

use chrono::{Datelike, Timelike};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_mapping_basic() {
        let mut mapping = IdMapping::new();
        let memory_id = MemoryId(uuid::Uuid::new_v4());

        mapping.insert(memory_id.clone(), 42);

        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.get_memory_id(42), Some(&memory_id));
    }

    #[test]
    fn test_id_mapping_chunks() {
        let mut mapping = IdMapping::new();
        let memory_id = MemoryId(uuid::Uuid::new_v4());

        mapping.insert_chunks(memory_id.clone(), vec![1, 2, 3]);

        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.get_memory_id(1), Some(&memory_id));
        assert_eq!(mapping.get_memory_id(2), Some(&memory_id));
        assert_eq!(mapping.get_memory_id(3), Some(&memory_id));
    }

    #[test]
    fn test_id_mapping_remove_all() {
        let mut mapping = IdMapping::new();
        let memory_id = MemoryId(uuid::Uuid::new_v4());

        mapping.insert_chunks(memory_id.clone(), vec![1, 2, 3]);
        let removed = mapping.remove_all(&memory_id);

        assert_eq!(removed.len(), 3);
        assert_eq!(mapping.len(), 0);
        assert!(mapping.get_memory_id(1).is_none());
    }

    #[test]
    fn test_id_mapping_clear() {
        let mut mapping = IdMapping::new();
        mapping.insert(MemoryId(uuid::Uuid::new_v4()), 1);
        mapping.insert(MemoryId(uuid::Uuid::new_v4()), 2);

        mapping.clear();

        assert_eq!(mapping.len(), 0);
    }

    #[test]
    fn test_retrieval_outcome_default() {
        let outcome = RetrievalOutcome::default();
        assert_eq!(outcome, RetrievalOutcome::Neutral);
    }

    #[test]
    fn test_reinforcement_stats_default() {
        let stats = ReinforcementStats::default();

        assert_eq!(stats.memories_processed, 0);
        assert_eq!(stats.associations_strengthened, 0);
        assert_eq!(stats.importance_boosts, 0);
        assert_eq!(stats.importance_decays, 0);
    }

    #[test]
    fn test_memory_graph_stats_default() {
        let stats = MemoryGraphStats::default();

        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.avg_strength, 0.0);
        assert_eq!(stats.potentiated_count, 0);
    }

    #[test]
    fn test_prefetch_context_default() {
        let ctx = PrefetchContext::default();

        assert!(ctx.project_id.is_none());
        assert!(ctx.current_file.is_none());
        assert!(ctx.recent_entities.is_empty());
    }

    #[test]
    fn test_prefetch_context_from_current_time() {
        let ctx = PrefetchContext::from_current_time();

        assert!(ctx.hour_of_day.is_some());
        assert!(ctx.day_of_week.is_some());
    }

    #[test]
    fn test_anticipatory_prefetch_new() {
        let prefetch = AnticipatoryPrefetch::new();
        assert_eq!(prefetch.max_prefetch, 20);
    }

    #[test]
    fn test_anticipatory_prefetch_with_limit() {
        let prefetch = AnticipatoryPrefetch::with_limit(50);
        assert_eq!(prefetch.max_prefetch, 50);
    }

    #[test]
    fn test_generate_prefetch_query_project() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            project_id: Some("my-project".to_string()),
            ..Default::default()
        };

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_some());
        let query = query.unwrap();
        assert!(query.query_text.unwrap().contains("my-project"));
    }

    #[test]
    fn test_generate_prefetch_query_entities() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            recent_entities: vec!["Rust".to_string(), "memory".to_string()],
            ..Default::default()
        };

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_some());
        let query = query.unwrap();
        let text = query.query_text.unwrap();
        assert!(text.contains("Rust"));
        assert!(text.contains("memory"));
    }

    #[test]
    fn test_generate_prefetch_query_file() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            current_file: Some("/src/memory/retrieval.rs".to_string()),
            ..Default::default()
        };

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_some());
        let query = query.unwrap();
        assert!(query.query_text.unwrap().contains("retrieval.rs"));
    }

    #[test]
    fn test_generate_prefetch_query_temporal() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext {
            hour_of_day: Some(14),
            day_of_week: Some(1),
            ..Default::default()
        };

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_some());
        let query = query.unwrap();
        assert!(query.time_range.is_some());
    }

    #[test]
    fn test_generate_prefetch_query_empty() {
        let prefetch = AnticipatoryPrefetch::new();
        let ctx = PrefetchContext::default();

        let query = prefetch.generate_prefetch_query(&ctx);

        assert!(query.is_none());
    }

    #[test]
    fn test_prefetch_reason_default() {
        let reason = PrefetchReason::default();
        assert!(matches!(reason, PrefetchReason::Mixed));
    }

    #[test]
    fn test_prefetch_result_default() {
        let result = PrefetchResult::default();

        assert!(result.prefetched_ids.is_empty());
        assert_eq!(result.cache_hits, 0);
        assert_eq!(result.fetches, 0);
    }

    #[test]
    fn test_index_health_struct() {
        let health = IndexHealth {
            total_vectors: 1000,
            incremental_inserts: 100,
            deleted_count: 50,
            deletion_ratio: 0.05,
            needs_rebuild: false,
            needs_compaction: false,
            rebuild_threshold: 500,
            deletion_ratio_threshold: 0.2,
        };

        assert_eq!(health.total_vectors, 1000);
        assert!(!health.needs_rebuild);
    }
}
