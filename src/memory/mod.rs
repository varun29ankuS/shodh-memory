//! Memory System for LLM Context Management
//!
//! A medium-complexity memory system that provides:
//! - Hierarchical memory storage (working → session → long-term)
//! - Smart compression based on age and importance
//! - Multi-modal retrieval (similarity, temporal, causal)
//! - Automatic memory consolidation

pub mod compression;
pub mod context;
pub mod introspection;
pub mod retrieval;
pub mod storage;
pub mod types;
// pub mod vector_storage;  // Disabled - requires crate::rag::vamana from parent project
pub mod graph_retrieval;
pub mod query_parser;
pub mod visualization;

use anyhow::{Context, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::debug;
use uuid::Uuid;

use crate::constants::{
    DEFAULT_COMPRESSION_AGE_DAYS, DEFAULT_IMPORTANCE_THRESHOLD, DEFAULT_MAX_HEAP_PER_USER_MB,
    DEFAULT_SESSION_MEMORY_SIZE_MB, DEFAULT_WORKING_MEMORY_SIZE, ESTIMATED_BYTES_PER_MEMORY,
    HEBBIAN_BOOST_HELPFUL, HEBBIAN_DECAY_MISLEADING,
};

use crate::memory::storage::MemoryStorage;
pub use crate::memory::types::*;
// pub use crate::memory::vector_storage::{VectorIndexedMemoryStorage, StorageStats};  // Disabled
use crate::embeddings::Embedder;
use crate::memory::compression::CompressionPipeline;
pub use crate::memory::compression::{
    ConsolidationResult, FactType, SemanticConsolidator, SemanticFact,
};
pub use crate::memory::graph_retrieval::{spreading_activation_retrieve, ActivatedMemory};
pub use crate::memory::introspection::{
    AssociationChange, ConsolidationEvent, ConsolidationEventBuffer, ConsolidationReport,
    ConsolidationStats, EdgeFormationReason, FactChange, MemoryChange, PruningReason, ReportPeriod,
    StrengtheningReason,
};
use crate::memory::retrieval::RetrievalEngine;
pub use crate::memory::retrieval::{
    AnticipatoryPrefetch, IndexHealth, MemoryGraphStats, PrefetchContext, PrefetchReason,
    PrefetchResult, ReinforcementStats, RetrievalFeedback, RetrievalOutcome, TrackedRetrieval,
};
pub use crate::memory::visualization::{GraphStats, MemoryLogger};

/// Configuration for the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Base directory for memory storage
    pub storage_path: PathBuf,

    /// Maximum size of working memory (in entries)
    pub working_memory_size: usize,

    /// Maximum size of session memory (in MB)
    pub session_memory_size_mb: usize,

    /// Maximum heap memory per user (in MB) - prevents OOM from single user
    pub max_heap_per_user_mb: usize,

    /// Enable auto-compression of old memories
    pub auto_compress: bool,

    /// Compression threshold (days)
    pub compression_age_days: u32,

    /// Importance threshold for long-term storage
    pub importance_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./memory_store"),
            working_memory_size: DEFAULT_WORKING_MEMORY_SIZE,
            session_memory_size_mb: DEFAULT_SESSION_MEMORY_SIZE_MB,
            max_heap_per_user_mb: DEFAULT_MAX_HEAP_PER_USER_MB,
            auto_compress: true,
            compression_age_days: DEFAULT_COMPRESSION_AGE_DAYS,
            importance_threshold: DEFAULT_IMPORTANCE_THRESHOLD,
        }
    }
}

/// Main memory system
pub struct MemorySystem {
    config: MemoryConfig,

    /// Three-tier memory hierarchy
    working_memory: Arc<RwLock<WorkingMemory>>,
    session_memory: Arc<RwLock<SessionMemory>>,
    long_term_memory: Arc<MemoryStorage>,

    /// Compression pipeline
    compressor: CompressionPipeline,

    /// Retrieval engine
    retriever: RetrievalEngine,

    /// Embedder for semantic search
    embedder: Arc<crate::embeddings::minilm::MiniLMEmbedder>,

    /// Query embedding cache - SHA256(query_text) → embedding
    /// Uses SHA256 for stable hashing across restarts (unlike DefaultHasher)
    /// MASSIVE PERF WIN: 80ms → <1ms for cached queries
    query_cache: Arc<DashMap<[u8; 32], Vec<f32>>>,

    /// Content embedding cache - SHA256(content) → embedding
    /// Uses SHA256 for stable hashing across restarts (unlike DefaultHasher)
    /// MASSIVE PERF WIN: 80ms → <1ms for repeated content
    content_cache: Arc<DashMap<[u8; 32], Vec<f32>>>,

    /// Memory statistics
    stats: Arc<RwLock<MemoryStats>>,

    /// Visualization logger
    logger: Arc<RwLock<MemoryLogger>>,

    /// Consolidation event buffer for introspection
    /// Tracks what the memory system is learning (strengthening, decay, edges, facts)
    consolidation_events: Arc<RwLock<ConsolidationEventBuffer>>,
}

impl MemorySystem {
    /// Create a new memory system
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let storage = Arc::new(MemoryStorage::new(&config.storage_path)?);

        // CRITICAL: Initialize embedder ONCE and share between MemorySystem and RetrievalEngine
        // This prevents loading the ONNX model multiple times (50-200ms overhead per load)
        let embedding_config = crate::embeddings::minilm::EmbeddingConfig::default();
        let embedder = Arc::new(
            crate::embeddings::minilm::MiniLMEmbedder::new(embedding_config)
                .context("Failed to initialize embedder")?,
        );

        // Create consolidation event buffer first so we can share it with retriever
        let consolidation_events = Arc::new(RwLock::new(ConsolidationEventBuffer::new()));

        // Pass shared embedder and event buffer to retrieval engine (no duplicate model load)
        // Event buffer allows retriever to record Hebbian edge events for introspection
        let retriever = RetrievalEngine::with_event_buffer(
            storage.clone(),
            embedder.clone(),
            Some(consolidation_events.clone()),
        )?;

        // Disable visualization logging for production performance
        let logger = Arc::new(RwLock::new(MemoryLogger::new(false)));

        // Load stats from storage to recover state after restart
        let initial_stats = {
            let storage_stats = storage.get_stats().unwrap_or_default();
            let vector_count = retriever.len();
            MemoryStats {
                total_memories: storage_stats.total_count,
                long_term_memory_count: storage_stats.total_count,
                vector_index_count: vector_count,
                ..Default::default()
            }
        };

        Ok(Self {
            config: config.clone(),
            working_memory: Arc::new(RwLock::new(WorkingMemory::new(config.working_memory_size))),
            session_memory: Arc::new(RwLock::new(SessionMemory::new(
                config.session_memory_size_mb,
            ))),
            long_term_memory: storage,
            compressor: CompressionPipeline::new(),
            retriever,
            embedder,
            query_cache: Arc::new(DashMap::new()),
            content_cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(initial_stats)),
            logger,
            consolidation_events, // Use the shared buffer created earlier
        })
    }

    /// Record a new experience (takes ownership to avoid clones)
    /// Thread-safe: uses interior mutability for all internal state
    /// If `created_at` is None, uses current time (Utc::now())
    pub fn record(
        &self,
        mut experience: Experience,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<MemoryId> {
        // CRITICAL: Check resource limits before recording to prevent OOM
        self.check_resource_limits()?;

        let memory_id = MemoryId(Uuid::new_v4());

        // Calculate importance
        let importance = self.calculate_importance(&experience);

        // PERFORMANCE: Content embedding cache (80ms → <1μs for repeated content)
        // If experience doesn't have embeddings, check cache or generate
        if experience.embeddings.is_none() {
            // SHA256 hash for stable cache keys (survives restarts, unlike DefaultHasher)
            let content_hash = Self::sha256_hash(&experience.content);

            // Check cache first
            if let Some(cached_embedding) = self.content_cache.get(&content_hash) {
                experience.embeddings = Some(cached_embedding.clone());
                tracing::debug!("Content embedding cache HIT");
            } else {
                // Cache miss - generate embedding
                match self.embedder.encode(&experience.content) {
                    Ok(embedding) => {
                        // Store in cache for future use
                        self.content_cache.insert(content_hash, embedding.clone());
                        experience.embeddings = Some(embedding);
                        tracing::debug!("Content embedding cache MISS - generated and cached");
                    }
                    Err(e) => {
                        tracing::warn!("Failed to generate embedding: {}", e);
                        // Continue without embedding - will be generated on-demand if needed
                    }
                }
            }
        }

        // Create memory entry (zero-copy with Arc)
        // CRITICAL: Move experience instead of clone to avoid 2-10KB allocation
        let memory = Arc::new(Memory::new(
            memory_id.clone(),
            experience, // Move ownership (zero-cost)
            importance,
            None,       // agent_id
            None,       // run_id
            None,       // actor_id
            created_at, // Use provided timestamp or Utc::now() if None
        ));

        // CRITICAL: Persist to RocksDB storage FIRST (before indexing/in-memory tiers)
        // This ensures retrieval can always fetch the memory from persistent storage
        self.long_term_memory.store(&memory)?;

        // Log creation
        self.logger.write().log_created(&memory, "working");

        // Add to working memory (cheap Arc clone, not full Memory clone)
        self.working_memory
            .write()
            .add_shared(Arc::clone(&memory))?;

        // CRITICAL: Index memory immediately for semantic search (don't wait for long-term promotion)
        // This ensures new memories are searchable right away, not only after consolidation
        let indexed = if let Err(e) = self.retriever.index_memory(&memory) {
            tracing::warn!("Failed to index memory {} in vector DB: {}", memory.id.0, e);
            // Don't fail the record operation if indexing fails - memory is still stored
            false
        } else {
            true
        };

        // Add to knowledge graph for associative/causal retrieval
        self.retriever.add_to_graph(&memory);

        // If important enough, prepare for session storage
        let added_to_session = if importance > self.config.importance_threshold {
            self.session_memory
                .write()
                .add_shared(Arc::clone(&memory))?;
            self.logger.write().log_created(&memory, "session");
            true
        } else {
            false
        };

        // Update stats - track all tier counts accurately
        {
            let mut stats = self.stats.write();
            stats.total_memories += 1;
            stats.long_term_memory_count += 1; // Always stored to long-term first
            stats.working_memory_count += 1;
            if added_to_session {
                stats.session_memory_count += 1;
            }
            if indexed {
                stats.vector_index_count += 1;
            }
        }

        // Trigger background consolidation if needed
        self.consolidate_if_needed()?;

        Ok(memory_id)
    }

    /// Retrieve relevant memories for current context (zero-copy with Arc<Memory>)
    ///
    /// PRODUCTION IMPLEMENTATION:
    /// - Semantic search: Uses embeddings + vector similarity across ALL tiers
    /// - Non-semantic search: Uses importance * temporal decay
    /// - Zero shortcuts, no TODOs, enterprise-grade
    pub fn retrieve(&self, query: &Query) -> Result<Vec<SharedMemory>> {
        // Semantic search requires special handling
        if let Some(query_text) = &query.query_text {
            return self.semantic_retrieve(query_text, query);
        }

        // Non-semantic search: filter-based retrieval
        let mut memories = Vec::new();
        let mut seen_ids: HashSet<MemoryId> = HashSet::new();
        let mut sources = Vec::new();

        // Collect from all tiers with deduplication (priority: working > session > long_term)
        {
            let working = self.working_memory.read();
            let working_results = working.search(query, query.max_results)?;
            if !working_results.is_empty() {
                sources.push("working");
            }
            for memory in working_results {
                if seen_ids.insert(memory.id.clone()) {
                    memories.push(memory);
                }
            }
        }

        {
            let session = self.session_memory.read();
            let session_results = session.search(query, query.max_results)?;
            if !session_results.is_empty() {
                sources.push("session");
            }
            for memory in session_results {
                if seen_ids.insert(memory.id.clone()) {
                    memories.push(memory);
                }
            }
        }

        {
            let long_term_results = self.retriever.search(query, query.max_results)?;
            if !long_term_results.is_empty() {
                sources.push("longterm");
            }
            for memory in long_term_results {
                if seen_ids.insert(memory.id.clone()) {
                    memories.push(memory);
                }
            }
        }

        // Rank by importance * temporal relevance
        let now = chrono::Utc::now();
        memories.sort_by(|a, b| {
            let age_days_a = (now - a.created_at).num_days();
            let temporal_a = Self::calculate_temporal_relevance(age_days_a);
            let score_a = a.importance() * temporal_a;

            let age_days_b = (now - b.created_at).num_days();
            let temporal_b = Self::calculate_temporal_relevance(age_days_b);
            let score_b = b.importance() * temporal_b;

            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        memories.truncate(query.max_results);

        // Log retrieval
        self.logger
            .read()
            .log_retrieved("", memories.len(), &sources);

        // Update access counts with instrumentation for consolidation events
        for memory in &memories {
            self.update_access_count_instrumented(memory, StrengtheningReason::Recalled);
        }

        // Hebbian learning: co-activation strengthens associations between memories
        // When memories are retrieved together, they form/strengthen edges in the memory graph
        if memories.len() >= 2 {
            let memory_ids: Vec<MemoryId> = memories.iter().map(|m| m.id.clone()).collect();
            self.retriever.record_coactivation(&memory_ids);
        }

        Ok(memories)
    }

    /// CACHE-AWARE semantic retrieval: Check working → session → storage
    ///
    /// Implementation:
    /// 1. Generate query embedding and search vector index for memory IDs
    /// 2. For each ID, check working memory (instant Arc clone)
    /// 3. If not found, check session memory (instant Arc clone)
    /// 4. Only fetch from RocksDB storage as last resort
    /// 5. This eliminates deserialization overhead for cached memories
    fn semantic_retrieve(&self, query_text: &str, query: &Query) -> Result<Vec<SharedMemory>> {
        // PERFORMANCE: Query embedding cache (80ms → <1μs for repeated queries)
        // SHA256 hash for stable cache keys (survives restarts, unlike DefaultHasher)
        let query_hash = Self::sha256_hash(query_text);

        // Check cache first
        let query_embedding = if let Some(cached_embedding) = self.query_cache.get(&query_hash) {
            tracing::debug!("Query embedding cache HIT for: {}", query_text);
            cached_embedding.clone()
        } else {
            // Cache miss - generate embedding
            tracing::debug!(
                "Query embedding cache MISS - generating for: {}",
                query_text
            );
            let embedding = self
                .embedder
                .as_ref()
                .encode(query_text)
                .context("Failed to generate query embedding")?;

            // Store in cache for future use
            self.query_cache.insert(query_hash, embedding.clone());
            embedding
        };

        // Create a modified query with the embedding for vector search
        let vector_query = Query {
            query_text: None, // Don't re-generate embedding
            query_embedding: Some(query_embedding),
            time_range: query.time_range,
            experience_types: query.experience_types.clone(),
            importance_threshold: query.importance_threshold,
            max_results: query.max_results,
            retrieval_mode: query.retrieval_mode.clone(),
            // Robotics filters (carry over from original query)
            robot_id: query.robot_id.clone(),
            mission_id: query.mission_id.clone(),
            geo_filter: query.geo_filter.clone(),
            action_type: query.action_type.clone(),
            reward_range: query.reward_range,
            // Decision & Learning filters (carry over from original query)
            outcome_type: query.outcome_type.clone(),
            failures_only: query.failures_only,
            anomalies_only: query.anomalies_only,
            severity: query.severity.clone(),
            tags: query.tags.clone(),
            pattern_id: query.pattern_id.clone(),
            terrain_type: query.terrain_type.clone(),
            confidence_range: query.confidence_range,
        };

        // Get memory IDs from vector search (fast HNSW search)
        let memory_ids = self
            .retriever
            .search_ids(&vector_query, query.max_results)?;

        // Fetch memories with cache-aware strategy
        // CRITICAL: Apply filters after fetching to ensure mission_id, robot_id etc. are respected
        let mut memories = Vec::new();
        let mut sources = Vec::new();
        let mut cache_hits = 0;
        let mut storage_fetches = 0;
        let mut filtered_out = 0;

        for (memory_id, _score) in memory_ids {
            // Try working memory first (hot cache, zero-copy Arc clone)
            if let Some(memory) = self.working_memory.read().get(&memory_id) {
                // CRITICAL FIX: Apply filters before adding to results
                if self.retriever.matches_filters(&memory, &vector_query) {
                    memories.push(memory);
                    if !sources.contains(&"working") {
                        sources.push("working");
                    }
                    cache_hits += 1;
                } else {
                    filtered_out += 1;
                }
                continue;
            }

            // Try session memory second (warm cache, zero-copy Arc clone)
            if let Some(memory) = self.session_memory.read().get(&memory_id) {
                // CRITICAL FIX: Apply filters before adding to results
                if self.retriever.matches_filters(&memory, &vector_query) {
                    memories.push(memory);
                    if !sources.contains(&"session") {
                        sources.push("session");
                    }
                    cache_hits += 1;
                } else {
                    filtered_out += 1;
                }
                continue;
            }

            // Cold path: Fetch from RocksDB storage (expensive deserialization)
            match self.retriever.get_from_storage(&memory_id) {
                Ok(memory) => {
                    // CRITICAL FIX: Apply filters before adding to results
                    if self.retriever.matches_filters(&memory, &vector_query) {
                        memories.push(Arc::new(memory));
                        if !sources.contains(&"longterm") {
                            sources.push("longterm");
                        }
                        storage_fetches += 1;
                    } else {
                        filtered_out += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!(memory_id = %memory_id.0, error = %e, "Failed to fetch from storage");
                }
            }

            if memories.len() >= query.max_results {
                break;
            }
        }

        tracing::debug!(filtered_out = filtered_out, "Filter pass completed");

        // Log cache efficiency
        tracing::debug!(
            cache_hits = cache_hits,
            storage_fetches = storage_fetches,
            hit_rate = if cache_hits + storage_fetches > 0 {
                (cache_hits as f32 / (cache_hits + storage_fetches) as f32) * 100.0
            } else {
                0.0
            },
            "Cache-aware retrieval completed"
        );

        // Re-rank using linguistic analysis (focal entities boost)
        let analysis = query_parser::analyze_query(query_text);
        if !analysis.focal_entities.is_empty() {
            memories.sort_by(|a, b| {
                let score_a = Self::linguistic_boost(&a.experience.content, &analysis);
                let score_b = Self::linguistic_boost(&b.experience.content, &analysis);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        self.logger
            .read()
            .log_retrieved(query_text, memories.len(), &sources);

        // Update access counts with instrumentation for consolidation events
        for memory in &memories {
            self.update_access_count_instrumented(memory, StrengtheningReason::Recalled);
        }

        // Hebbian learning: co-activation strengthens associations between memories
        // When memories are retrieved together, they form/strengthen edges in the memory graph
        if memories.len() >= 2 {
            let memory_ids: Vec<MemoryId> = memories.iter().map(|m| m.id.clone()).collect();
            self.retriever.record_coactivation(&memory_ids);
        }

        Ok(memories)
    }

    /// Calculate linguistic boost based on focal entity matches
    fn linguistic_boost(content: &str, analysis: &query_parser::QueryAnalysis) -> f32 {
        let content_lower = content.to_lowercase();
        let mut boost = 0.0;

        for entity in &analysis.focal_entities {
            if content_lower.contains(&entity.text) {
                boost += entity.ic_weight;
            }
        }

        for modifier in &analysis.discriminative_modifiers {
            if content_lower.contains(&modifier.text) {
                boost += 1.7; // IC_ADJECTIVE
            }
        }

        boost
    }

    /// Compute SHA256 hash of text for stable cache keys
    ///
    /// Unlike std::hash::DefaultHasher, SHA256 produces deterministic hashes
    /// across process restarts and Rust versions. This is critical for:
    /// - Embedding cache persistence (future feature)
    /// - Consistent behavior across restarts
    /// - Avoiding cache key collisions
    #[inline]
    fn sha256_hash(text: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hasher.finalize().into()
    }

    /// Forget memories based on criteria
    /// Thread-safe: uses interior mutability for all internal state
    pub fn forget(&self, criteria: ForgetCriteria) -> Result<usize> {
        let forgotten_count = match criteria {
            ForgetCriteria::ById(memory_id) => {
                // Delete a single memory by ID from all tiers, tracking which tiers had it
                let mut deleted_from_any = false;
                let mut was_in_working = false;
                let mut was_in_session = false;
                let mut was_in_longterm = false;

                // Remove from working memory
                if self.working_memory.write().remove(&memory_id).is_ok() {
                    deleted_from_any = true;
                    was_in_working = true;
                }

                // Remove from session memory
                if self.session_memory.write().remove(&memory_id).is_ok() {
                    deleted_from_any = true;
                    was_in_session = true;
                }

                // Remove from long-term storage
                if self.long_term_memory.delete(&memory_id).is_ok() {
                    deleted_from_any = true;
                    was_in_longterm = true;
                }

                // Remove from vector index (soft delete) - CRITICAL for semantic search
                // This marks the vector as deleted so it won't appear in search results
                let was_indexed = self.retriever.remove_memory(&memory_id);

                // Update stats - decrement each tier count that had this memory
                if deleted_from_any {
                    let mut stats = self.stats.write();
                    stats.total_memories = stats.total_memories.saturating_sub(1);
                    if was_in_working {
                        stats.working_memory_count = stats.working_memory_count.saturating_sub(1);
                    }
                    if was_in_session {
                        stats.session_memory_count = stats.session_memory_count.saturating_sub(1);
                    }
                    if was_in_longterm {
                        stats.long_term_memory_count =
                            stats.long_term_memory_count.saturating_sub(1);
                    }
                    if was_indexed {
                        stats.vector_index_count = stats.vector_index_count.saturating_sub(1);
                    }
                    1
                } else {
                    0
                }
            }
            ForgetCriteria::OlderThan(days) => {
                let cutoff = chrono::Utc::now() - chrono::Duration::days(days as i64);

                // Remove from working memory
                self.working_memory.write().remove_older_than(cutoff)?;

                // Remove from session memory
                self.session_memory.write().remove_older_than(cutoff)?;

                // Mark as forgotten in long-term (don't delete, just flag)
                self.long_term_memory.mark_forgotten_by_age(cutoff)?
            }
            ForgetCriteria::LowImportance(threshold) => {
                self.working_memory
                    .write()
                    .remove_below_importance(threshold)?;
                self.session_memory
                    .write()
                    .remove_below_importance(threshold)?;
                self.long_term_memory
                    .mark_forgotten_by_importance(threshold)?
            }
            ForgetCriteria::Pattern(pattern) => {
                // Remove memories matching pattern
                self.forget_by_pattern(&pattern)?
            }
            ForgetCriteria::ByTags(tags) => {
                // Remove memories matching ANY of the specified tags
                self.forget_by_tags(&tags)?
            }
            ForgetCriteria::ByDateRange { start, end } => {
                // Remove memories within the date range
                self.forget_by_date_range(start, end)?
            }
            ForgetCriteria::ByType(exp_type) => {
                // Remove memories of a specific type
                self.forget_by_type(exp_type)?
            }
            ForgetCriteria::All => {
                // GDPR: Clear ALL memories for the user
                self.forget_all()?
            }
        };

        Ok(forgotten_count)
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.read().clone()
    }

    /// Export visualization graph as DOT format for Graphviz
    pub fn export_visualization_dot(&self) -> String {
        self.logger.read().graph.to_dot()
    }

    /// Build visualization graph from current memory state
    /// Call this to populate the visualization graph with all current memories
    pub fn build_visualization_graph(&self) -> Result<visualization::GraphStats> {
        let mut logger = self.logger.write();

        // Add working memory entries directly to the graph (bypasses enabled check)
        for memory in self.working_memory.read().all_memories() {
            logger.graph.add_memory(&memory, "working");
        }

        // Add session memory entries
        for memory in self.session_memory.read().all_memories() {
            logger.graph.add_memory(&memory, "session");
        }

        // Add long-term memory entries
        for memory in self.long_term_memory.get_all()? {
            logger.graph.add_memory(&memory, "longterm");
        }

        Ok(logger.get_stats())
    }

    /// Get reference to embedder for graph-aware retrieval
    pub fn get_embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    /// Get all memories across all tiers for graph-aware retrieval
    /// Deduplicates by memory ID, preferring working > session > long-term
    pub fn get_all_memories(&self) -> Result<Vec<SharedMemory>> {
        use std::collections::HashSet;
        let mut seen_ids: HashSet<MemoryId> = HashSet::new();
        let mut all_memories = Vec::new();

        // Collect from working memory (highest priority - most recent/active)
        {
            let working = self.working_memory.read();
            for mem in working.all_memories() {
                if seen_ids.insert(mem.id.clone()) {
                    all_memories.push(mem);
                }
            }
        }

        // Collect from session memory (medium priority)
        {
            let session = self.session_memory.read();
            for mem in session.all_memories() {
                if seen_ids.insert(mem.id.clone()) {
                    all_memories.push(mem);
                }
            }
        }

        // Collect from long-term memory (lowest priority - wrap in Arc)
        {
            let longterm_mems = self.long_term_memory.get_all()?;
            for mem in longterm_mems {
                if seen_ids.insert(mem.id.clone()) {
                    all_memories.push(Arc::new(mem));
                }
            }
        }

        Ok(all_memories)
    }

    /// Get memories from working memory tier (highest activation, most recent)
    pub fn get_working_memories(&self) -> Vec<SharedMemory> {
        let working = self.working_memory.read();
        working.all_memories()
    }

    /// Get memories from session memory tier (medium-term, consolidated)
    pub fn get_session_memories(&self) -> Vec<SharedMemory> {
        let session = self.session_memory.read();
        session.all_memories()
    }

    /// Get memories from long-term memory tier (persistent, lower activation)
    /// Returns up to `limit` memories to avoid overwhelming responses
    pub fn get_longterm_memories(&self, limit: usize) -> Result<Vec<Memory>> {
        let all = self.long_term_memory.get_all()?;
        Ok(all.into_iter().take(limit).collect())
    }

    /// Calculate temporal relevance based on memory age (ENTERPRISE FEATURE)
    ///
    /// Implements exponential decay curve for time-aware memory retrieval:
    /// - 0-7 days: Full relevance (1.0) - recent memories
    /// - 8-30 days: High relevance (0.7) - medium-term memories
    /// - 31-90 days: Moderate relevance (0.4) - older memories
    /// - 90+ days: Low relevance (0.2) - ancient memories
    ///
    /// This ensures recent experiences are prioritized while maintaining
    /// access to historical context when needed.
    fn calculate_temporal_relevance(age_days: i64) -> f32 {
        match age_days {
            0..=7 => 1.0,   // Recent: Full weight
            8..=30 => 0.7,  // Medium-term: 70% weight
            31..=90 => 0.4, // Old: 40% weight
            _ => 0.2,       // Ancient: 20% weight (never completely forgotten)
        }
    }

    /// Calculate importance of an experience using multi-factor analysis
    fn calculate_importance(&self, experience: &Experience) -> f32 {
        let mut factors = Vec::new();

        // Factor 1: Experience type base score (0.0 - 0.3)
        let type_score = match experience.experience_type {
            ExperienceType::Decision => 0.3,
            ExperienceType::Error => 0.25,
            ExperienceType::Learning => 0.25,
            ExperienceType::Discovery => 0.2,
            ExperienceType::Pattern => 0.2,
            ExperienceType::Task => 0.15,
            ExperienceType::Conversation => 0.1,
            ExperienceType::Context => 0.1,
            _ => 0.05,
        };
        factors.push(("type", type_score));

        // Factor 2: Content richness (0.0 - 0.25)
        let _content_length = experience.content.len();
        let word_count = experience.content.split_whitespace().count();
        let richness_score = if word_count > 50 {
            0.25
        } else if word_count > 20 {
            0.15
        } else if word_count > 5 {
            0.08
        } else {
            0.02
        };
        factors.push(("richness", richness_score));

        // Factor 3: Entity density (0.0 - 0.2)
        let entity_score = if experience.entities.len() > 10 {
            0.2
        } else if experience.entities.len() > 5 {
            0.15
        } else if experience.entities.len() > 2 {
            0.1
        } else if !experience.entities.is_empty() {
            0.05
        } else {
            0.0
        };
        factors.push(("entities", entity_score));

        // Factor 4: Context depth (0.0 - 0.2)
        let context_score = if let Some(ctx) = &experience.context {
            let mut score: f32 = 0.0;

            // Rich semantic context
            if !ctx.semantic.concepts.is_empty() {
                score += 0.05;
            }
            if !ctx.semantic.tags.is_empty() {
                score += 0.03;
            }
            if !ctx.semantic.related_concepts.is_empty() {
                score += 0.04;
            }

            // Project/workspace context
            if ctx.project.project_id.is_some() {
                score += 0.03;
            }

            // Code context
            if ctx.code.current_file.is_some() {
                score += 0.03;
            }

            // Document citations
            if !ctx.document.citations.is_empty() {
                score += 0.02;
            }

            score.min(0.2)
        } else {
            0.0
        };
        factors.push(("context", context_score));

        // Factor 5: Metadata signals (0.0 - 0.15)
        let mut metadata_score: f32 = 0.0;

        if experience.metadata.contains_key("priority") {
            if let Some(priority) = experience.metadata.get("priority") {
                metadata_score += match priority.as_str() {
                    "critical" => 0.15,
                    "high" => 0.10,
                    "medium" => 0.05,
                    _ => 0.0,
                };
            }
        }

        if experience.metadata.contains_key("unexpected") {
            metadata_score += 0.08;
        }

        if experience.metadata.contains_key("breakthrough") {
            metadata_score += 0.12;
        }

        if experience.metadata.get("role") == Some(&"user".to_string()) {
            metadata_score += 0.02; // User messages slightly more important
        }

        factors.push(("metadata", metadata_score.min(0.15)));

        // Factor 6: Embeddings quality (0.0 - 0.1)
        let embedding_score = if let Some(emb) = &experience.embeddings {
            if emb.len() >= 384 {
                // Full embedding vector
                0.1
            } else {
                0.05
            }
        } else {
            0.0
        };
        factors.push(("embeddings", embedding_score));

        // Factor 7: Content quality indicators (0.0 - 0.1)
        let content_lower = experience.content.to_lowercase();
        let mut quality_score: f32 = 0.0;

        // Technical terms indicate higher quality
        let technical_terms = [
            "algorithm",
            "architecture",
            "implementation",
            "optimization",
            "performance",
            "security",
            "database",
            "api",
            "framework",
        ];
        for term in &technical_terms {
            if content_lower.contains(term) {
                quality_score += 0.015;
            }
        }

        // Questions indicate learning/discovery
        if content_lower.contains('?') {
            quality_score += 0.02;
        }

        // Code snippets indicate actionable content
        if experience.content.contains("```")
            || experience.content.contains("fn ")
            || experience.content.contains("function ")
            || experience.content.contains("class ")
        {
            quality_score += 0.03;
        }

        factors.push(("quality", quality_score.min(0.1)));

        // Aggregate all factors
        let importance: f32 = factors.iter().map(|(_, score)| score).sum();

        // Ensure importance is in valid range [0.0, 1.0]
        let importance = importance.clamp(0.0, 1.0);

        // Log importance calculation for transparency
        if importance > 0.7 {
            debug!("High importance memory: {:.2} (type={:.2}, richness={:.2}, entities={:.2}, context={:.2})",
                importance,
                factors.iter().find(|(k, _)| *k == "type").map(|(_, v)| v).unwrap_or(&0.0),
                factors.iter().find(|(k, _)| *k == "richness").map(|(_, v)| v).unwrap_or(&0.0),
                factors.iter().find(|(k, _)| *k == "entities").map(|(_, v)| v).unwrap_or(&0.0),
                factors.iter().find(|(k, _)| *k == "context").map(|(_, v)| v).unwrap_or(&0.0)
            );
        }

        importance
    }

    /// Consolidate memories when thresholds are reached
    fn consolidate_if_needed(&self) -> Result<()> {
        let working_size = self.working_memory.read().size();

        // If working memory is full, move to session
        if working_size >= self.config.working_memory_size {
            self.promote_working_to_session()?;
        }

        // If session memory is large, compress and move to long-term
        let session_size = self.session_memory.read().size_mb();
        if session_size >= self.config.session_memory_size_mb {
            self.promote_session_to_longterm()?;
        }

        // Compress old memories if auto-compress is enabled
        if self.config.auto_compress {
            self.compress_old_memories()?;
        }

        Ok(())
    }

    /// Move memories from working to session memory
    fn promote_working_to_session(&self) -> Result<()> {
        let mut working = self.working_memory.write();
        let mut session = self.session_memory.write();

        // Get least recently used memories
        let to_promote = working.get_lru(self.config.working_memory_size / 2)?;
        let count = to_promote.len();

        for memory in &to_promote {
            // Log promotion
            self.logger
                .write()
                .log_promoted(&memory.id, "working", "session", count);

            // Clone out of Arc and update tier before session storage
            let mut promoted_memory = (**memory).clone();
            promoted_memory.promote(); // Working -> Session
            session.add(promoted_memory)?;
            working.remove(&memory.id)?;
        }

        self.stats.write().promotions_to_session += 1;
        Ok(())
    }

    /// Move memories from session to long-term storage
    fn promote_session_to_longterm(&self) -> Result<()> {
        let mut session = self.session_memory.write();

        // Get memories to promote (important ones)
        let to_promote = session.get_important(self.config.importance_threshold)?;
        let count = to_promote.len();

        for memory in &to_promote {
            // Log promotion
            self.logger
                .write()
                .log_promoted(&memory.id, "session", "longterm", count);

            // Clone out of Arc and update tier before long-term storage
            let mut owned_memory = (**memory).clone();
            owned_memory.promote(); // Session -> LongTerm

            // Compress if old enough
            let compressed_memory = if self.should_compress(&owned_memory) {
                self.compressor.compress(&owned_memory)?
            } else {
                owned_memory
            };

            // Store in long-term
            self.long_term_memory.store(&compressed_memory)?;

            // PRODUCTION: Index memory in Vamana vector DB for semantic search
            if let Err(e) = self.retriever.index_memory(&compressed_memory) {
                tracing::warn!(
                    "Failed to index memory {} in vector DB: {}",
                    compressed_memory.id.0,
                    e
                );
                // Don't fail promotion if indexing fails - memory is still stored
            }

            // Remove from session
            session.remove(&memory.id)?;
        }

        self.stats.write().promotions_to_longterm += 1;
        Ok(())
    }

    /// Compress old memories to save space
    fn compress_old_memories(&self) -> Result<()> {
        let cutoff =
            chrono::Utc::now() - chrono::Duration::days(self.config.compression_age_days as i64);

        // Get uncompressed old memories
        let to_compress = self.long_term_memory.get_uncompressed_older_than(cutoff)?;

        for memory in to_compress {
            let compressed = self.compressor.compress(&memory)?;
            self.long_term_memory.update(&compressed)?;
            self.stats.write().compressed_count += 1;
        }

        Ok(())
    }

    /// Check if a memory should be compressed
    fn should_compress(&self, memory: &Memory) -> bool {
        let age = chrono::Utc::now() - memory.created_at;
        age.num_days() > self.config.compression_age_days as i64 && !memory.compressed
    }

    /// Update access count for a memory (handles concurrency properly)
    fn update_access_count(&self, memory_id: &MemoryId) -> Result<()> {
        // Try updating in working memory first (most common case)
        // Use write lock directly to avoid TOCTOU race condition
        {
            let mut wm = self.working_memory.write();

            if wm.contains(memory_id) {
                // Memory found in working memory - update and return
                return wm
                    .update_access(memory_id)
                    .map_err(|e| anyhow::anyhow!("Failed to update working memory access: {e}"));
            }
        } // Release write lock

        // Try session memory
        {
            let mut sm = self.session_memory.write();

            if sm.contains(memory_id) {
                return sm
                    .update_access(memory_id)
                    .map_err(|e| anyhow::anyhow!("Failed to update session memory access: {e}"));
            }
        } // Release write lock

        // Try long-term memory (has its own internal locking)
        self.long_term_memory
            .update_access(memory_id)
            .map_err(|e| anyhow::anyhow!("Failed to update long-term memory access: {e}"))
    }

    /// Update access count with instrumentation for consolidation events
    ///
    /// Records MemoryStrengthened events when memories are accessed during retrieval,
    /// capturing activation changes for introspection.
    fn update_access_count_instrumented(&self, memory: &SharedMemory, reason: StrengtheningReason) {
        // Capture activation before update
        let activation_before = memory.importance();

        // Perform the actual access update
        memory.update_access();

        // Capture activation after update
        let activation_after = memory.importance();

        // Only record event if activation actually changed
        if (activation_after - activation_before).abs() > f32::EPSILON {
            let content_preview = if memory.experience.content.chars().count() > 50 {
                let truncated: String = memory.experience.content.chars().take(50).collect();
                format!("{}...", truncated)
            } else {
                memory.experience.content.clone()
            };

            let event = ConsolidationEvent::MemoryStrengthened {
                memory_id: memory.id.0.to_string(),
                content_preview,
                activation_before,
                activation_after,
                reason,
                timestamp: chrono::Utc::now(),
            };

            self.consolidation_events.write().push(event);
        }
    }

    /// Forget memories matching a pattern
    ///
    /// Uses validated regex compilation with ReDoS protection
    fn forget_by_pattern(&self, pattern: &str) -> Result<usize> {
        // Use validated pattern compilation with ReDoS protection
        let regex = crate::validation::validate_and_compile_pattern(pattern)
            .map_err(|e| anyhow::anyhow!("Invalid pattern: {e}"))?;
        let mut count = 0;
        let mut working_removed = 0;
        let mut session_removed = 0;
        let mut long_term_removed = 0;

        // Collect IDs from working memory that match
        let working_ids: Vec<MemoryId> = {
            let working = self.working_memory.read();
            working
                .all_memories()
                .iter()
                .filter(|m| regex.is_match(&m.experience.content))
                .map(|m| m.id.clone())
                .collect()
        };
        // Remove from working memory and vector index
        {
            let mut working = self.working_memory.write();
            for id in &working_ids {
                if working.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    working_removed += 1;
                    count += 1;
                }
            }
        }

        // Collect IDs from session memory that match
        let session_ids: Vec<MemoryId> = {
            let session = self.session_memory.read();
            session
                .all_memories()
                .iter()
                .filter(|m| regex.is_match(&m.experience.content))
                .map(|m| m.id.clone())
                .collect()
        };
        // Remove from session memory and vector index
        {
            let mut session = self.session_memory.write();
            for id in &session_ids {
                if session.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    session_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from long-term memory
        let all_lt = self.long_term_memory.get_all()?;
        for memory in all_lt {
            if regex.is_match(&memory.experience.content) {
                self.retriever.remove_memory(&memory.id);
                self.long_term_memory.delete(&memory.id)?;
                long_term_removed += 1;
                count += 1;
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories = stats.total_memories.saturating_sub(count);
            stats.working_memory_count = stats.working_memory_count.saturating_sub(working_removed);
            stats.session_memory_count = stats.session_memory_count.saturating_sub(session_removed);
            stats.long_term_memory_count = stats
                .long_term_memory_count
                .saturating_sub(long_term_removed);
            stats.vector_index_count = stats.vector_index_count.saturating_sub(count);
        }

        Ok(count)
    }

    /// Forget memories matching ANY of the specified tags
    fn forget_by_tags(&self, tags: &[String]) -> Result<usize> {
        let mut count = 0;
        let mut working_removed = 0;
        let mut session_removed = 0;
        let mut long_term_removed = 0;

        // Remove from working memory
        {
            let mut working = self.working_memory.write();
            let ids_to_remove: Vec<MemoryId> = working
                .all_memories()
                .iter()
                .filter(|m| m.experience.tags.iter().any(|t| tags.contains(t)))
                .map(|m| m.id.clone())
                .collect();
            for id in &ids_to_remove {
                if working.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    working_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from session memory
        {
            let mut session = self.session_memory.write();
            let ids_to_remove: Vec<MemoryId> = session
                .all_memories()
                .iter()
                .filter(|m| m.experience.tags.iter().any(|t| tags.contains(t)))
                .map(|m| m.id.clone())
                .collect();
            for id in &ids_to_remove {
                if session.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    session_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from long-term memory (hard delete for tag-based)
        let all_lt = self.long_term_memory.get_all()?;
        for memory in all_lt {
            if memory.experience.tags.iter().any(|t| tags.contains(t)) {
                self.retriever.remove_memory(&memory.id);
                self.long_term_memory.delete(&memory.id)?;
                long_term_removed += 1;
                count += 1;
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories = stats.total_memories.saturating_sub(count);
            stats.working_memory_count = stats.working_memory_count.saturating_sub(working_removed);
            stats.session_memory_count = stats.session_memory_count.saturating_sub(session_removed);
            stats.long_term_memory_count = stats
                .long_term_memory_count
                .saturating_sub(long_term_removed);
            stats.vector_index_count = stats.vector_index_count.saturating_sub(count);
        }

        Ok(count)
    }

    /// Forget memories within a date range (inclusive)
    fn forget_by_date_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<usize> {
        let mut count = 0;
        let mut working_removed = 0;
        let mut session_removed = 0;
        let mut long_term_removed = 0;

        // Remove from working memory
        {
            let mut working = self.working_memory.write();
            let ids_to_remove: Vec<MemoryId> = working
                .all_memories()
                .iter()
                .filter(|m| m.created_at >= start && m.created_at <= end)
                .map(|m| m.id.clone())
                .collect();
            for id in &ids_to_remove {
                if working.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    working_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from session memory
        {
            let mut session = self.session_memory.write();
            let ids_to_remove: Vec<MemoryId> = session
                .all_memories()
                .iter()
                .filter(|m| m.created_at >= start && m.created_at <= end)
                .map(|m| m.id.clone())
                .collect();
            for id in &ids_to_remove {
                if session.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    session_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from long-term memory using storage search
        let memories = self
            .long_term_memory
            .search(storage::SearchCriteria::ByDate { start, end })?;
        for memory in memories {
            self.retriever.remove_memory(&memory.id);
            self.long_term_memory.delete(&memory.id)?;
            long_term_removed += 1;
            count += 1;
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories = stats.total_memories.saturating_sub(count);
            stats.working_memory_count = stats.working_memory_count.saturating_sub(working_removed);
            stats.session_memory_count = stats.session_memory_count.saturating_sub(session_removed);
            stats.long_term_memory_count = stats
                .long_term_memory_count
                .saturating_sub(long_term_removed);
            stats.vector_index_count = stats.vector_index_count.saturating_sub(count);
        }

        Ok(count)
    }

    /// Forget memories of a specific type
    fn forget_by_type(&self, exp_type: ExperienceType) -> Result<usize> {
        let mut count = 0;
        let mut working_removed = 0;
        let mut session_removed = 0;
        let mut long_term_removed = 0;

        // Remove from working memory
        {
            let mut working = self.working_memory.write();
            let ids_to_remove: Vec<MemoryId> = working
                .all_memories()
                .iter()
                .filter(|m| m.experience.experience_type == exp_type)
                .map(|m| m.id.clone())
                .collect();
            for id in &ids_to_remove {
                if working.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    working_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from session memory
        {
            let mut session = self.session_memory.write();
            let ids_to_remove: Vec<MemoryId> = session
                .all_memories()
                .iter()
                .filter(|m| m.experience.experience_type == exp_type)
                .map(|m| m.id.clone())
                .collect();
            for id in &ids_to_remove {
                if session.remove(id).is_ok() {
                    self.retriever.remove_memory(id);
                    session_removed += 1;
                    count += 1;
                }
            }
        }

        // Remove from long-term memory using storage search
        let memories = self
            .long_term_memory
            .search(storage::SearchCriteria::ByType(exp_type))?;
        for memory in memories {
            self.retriever.remove_memory(&memory.id);
            self.long_term_memory.delete(&memory.id)?;
            long_term_removed += 1;
            count += 1;
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_memories = stats.total_memories.saturating_sub(count);
            stats.working_memory_count = stats.working_memory_count.saturating_sub(working_removed);
            stats.session_memory_count = stats.session_memory_count.saturating_sub(session_removed);
            stats.long_term_memory_count = stats
                .long_term_memory_count
                .saturating_sub(long_term_removed);
            stats.vector_index_count = stats.vector_index_count.saturating_sub(count);
        }

        Ok(count)
    }

    /// Forget ALL memories for a user (GDPR compliance - right to erasure)
    ///
    /// WARNING: This is a destructive operation. All memories across all tiers
    /// will be permanently deleted. This cannot be undone.
    fn forget_all(&self) -> Result<usize> {
        let mut count = 0;

        // Collect all IDs from working memory and clear
        let working_ids: Vec<MemoryId> = {
            let working = self.working_memory.read();
            working
                .all_memories()
                .iter()
                .map(|m| m.id.clone())
                .collect()
        };
        let working_count = working_ids.len();
        for id in &working_ids {
            self.retriever.remove_memory(id);
        }
        {
            let mut working = self.working_memory.write();
            working.clear();
        }
        count += working_count;

        // Collect all IDs from session memory and clear
        let session_ids: Vec<MemoryId> = {
            let session = self.session_memory.read();
            session
                .all_memories()
                .iter()
                .map(|m| m.id.clone())
                .collect()
        };
        let session_count = session_ids.len();
        for id in &session_ids {
            self.retriever.remove_memory(id);
        }
        {
            let mut session = self.session_memory.write();
            session.clear();
        }
        count += session_count;

        // Clear all from long-term memory (hard delete)
        let all_lt = self.long_term_memory.get_all()?;
        let long_term_count = all_lt.len();
        for memory in all_lt {
            self.retriever.remove_memory(&memory.id);
            self.long_term_memory.delete(&memory.id)?;
        }
        count += long_term_count;

        // Reset all stats to zero
        {
            let mut stats = self.stats.write();
            stats.total_memories = 0;
            stats.working_memory_count = 0;
            stats.session_memory_count = 0;
            stats.long_term_memory_count = 0;
            stats.vector_index_count = 0;
        }

        Ok(count)
    }

    /// Show memory visualization (ASCII art of memory graph)
    pub fn show_visualization(&self) {
        self.logger.read().show_visualization();
    }

    /// Export memory graph as DOT file for Graphviz
    pub fn export_graph(&self, path: &std::path::Path) -> Result<()> {
        self.logger.read().export_dot(path)
    }

    /// Get visualization statistics
    pub fn get_visualization_stats(&self) -> GraphStats {
        self.logger.read().get_stats()
    }

    /// Flush long-term storage to ensure data persistence (critical for graceful shutdown)
    pub fn flush_storage(&self) -> Result<()> {
        // Flush RocksDB storage
        self.long_term_memory.flush()?;

        // Persist vector index and ID mapping for restart recovery
        self.retriever.save()?;

        Ok(())
    }

    /// Advanced search using storage criteria
    pub fn advanced_search(&self, criteria: storage::SearchCriteria) -> Result<Vec<Memory>> {
        self.long_term_memory.search(criteria)
    }

    /// Get memory by ID from long-term storage
    pub fn get_memory(&self, id: &MemoryId) -> Result<Memory> {
        self.long_term_memory.get(id)
    }

    /// Decompress a memory
    pub fn decompress_memory(&self, memory: &Memory) -> Result<Memory> {
        self.compressor.decompress(memory)
    }

    /// Get storage statistics
    pub fn get_storage_stats(&self) -> Result<storage::StorageStats> {
        self.long_term_memory.get_stats()
    }

    /// Get uncompressed old memories
    pub fn get_uncompressed_older_than(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<Memory>> {
        self.long_term_memory.get_uncompressed_older_than(cutoff)
    }

    /// Rebuild vector index from all existing long-term memories (startup initialization)
    pub fn rebuild_vector_index(&self) -> Result<()> {
        self.retriever.rebuild_index()
    }

    /// Repair vector index by finding and re-indexing orphaned memories
    ///
    /// Orphaned memories are those stored in RocksDB but missing from the vector index.
    /// This can happen if embedding generation fails during record().
    ///
    /// Returns: (total_storage, indexed, repaired, failed)
    pub fn repair_vector_index(&self) -> Result<(usize, usize, usize, usize)> {
        let all_memories = self.long_term_memory.get_all()?;
        let total_storage = all_memories.len();
        let indexed_before = self.retriever.len();

        let mut repaired = 0;
        let mut failed = 0;

        // Get set of indexed memory IDs
        let indexed_ids = self.retriever.get_indexed_memory_ids();

        for memory in all_memories {
            // Check if memory is already indexed
            if indexed_ids.contains(&memory.id) {
                continue;
            }

            // Memory is orphaned - try to index it
            tracing::info!(
                memory_id = %memory.id.0,
                content_preview = %memory.experience.content.chars().take(50).collect::<String>(),
                "Repairing orphaned memory"
            );

            match self.retriever.index_memory(&memory) {
                Ok(_) => {
                    repaired += 1;
                    tracing::info!(memory_id = %memory.id.0, "Successfully repaired orphaned memory");
                }
                Err(e) => {
                    failed += 1;
                    tracing::error!(
                        memory_id = %memory.id.0,
                        error = %e,
                        "Failed to repair orphaned memory - embedding generation failed"
                    );
                }
            }
        }

        let indexed_after = self.retriever.len();

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.vector_index_count = indexed_after;
        }

        tracing::info!(
            total_storage = total_storage,
            indexed_before = indexed_before,
            indexed_after = indexed_after,
            repaired = repaired,
            failed = failed,
            "Vector index repair completed"
        );

        Ok((total_storage, indexed_after, repaired, failed))
    }

    /// Verify index integrity and return diagnostic information
    ///
    /// Returns a struct with:
    /// - total_storage: memories in RocksDB
    /// - total_indexed: memories in vector index
    /// - orphaned_count: memories missing from index
    /// - orphaned_ids: list of orphaned memory IDs (first 100)
    pub fn verify_index_integrity(&self) -> Result<IndexIntegrityReport> {
        let all_memories = self.long_term_memory.get_all()?;
        let total_storage = all_memories.len();
        let indexed_ids = self.retriever.get_indexed_memory_ids();
        let total_indexed = indexed_ids.len();

        let mut orphaned_ids = Vec::new();
        for memory in &all_memories {
            if !indexed_ids.contains(&memory.id) {
                if orphaned_ids.len() < 100 {
                    orphaned_ids.push(memory.id.clone());
                }
            }
        }

        let orphaned_count = total_storage.saturating_sub(total_indexed);

        Ok(IndexIntegrityReport {
            total_storage,
            total_indexed,
            orphaned_count,
            orphaned_ids,
            is_healthy: orphaned_count == 0,
        })
    }

    /// Cleanup corrupted memories that fail to deserialize
    /// Returns the number of entries deleted
    pub fn cleanup_corrupted(&self) -> Result<usize> {
        self.long_term_memory.cleanup_corrupted()
    }

    /// Rebuild vector index from scratch using only valid memories in storage
    /// This removes orphaned index entries and rebuilds with proper ID mappings
    /// Returns (total_memories, total_indexed)
    pub fn rebuild_index(&self) -> Result<(usize, usize)> {
        tracing::info!("Starting full index rebuild from storage");
        self.retriever.rebuild_index()?;
        let indexed = self.retriever.len();
        let storage_count = self.long_term_memory.get_stats()?.total_count;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.vector_index_count = indexed;
        }

        tracing::info!(
            storage_count = storage_count,
            indexed = indexed,
            "Index rebuild complete"
        );

        Ok((storage_count, indexed))
    }

    /// Save vector index to disk (shutdown persistence)
    pub fn save_vector_index(&self, path: &Path) -> Result<()> {
        self.retriever.save_index(path)
    }

    /// Load vector index from disk (startup restoration)
    pub fn load_vector_index(&self, path: &Path) -> Result<()> {
        self.retriever.load_index(path)
    }

    /// Get vector index health information
    ///
    /// Returns metrics about the Vamana index including total vectors,
    /// incremental inserts since last build, and whether rebuild is recommended.
    pub fn index_health(&self) -> retrieval::IndexHealth {
        self.retriever.index_health()
    }

    /// Auto-rebuild vector index if degradation threshold is exceeded
    ///
    /// Returns `Ok(true)` if rebuild was performed, `Ok(false)` if not needed.
    /// Thread-safe: concurrent calls are no-ops while rebuild is in progress.
    pub fn auto_rebuild_index_if_needed(&self) -> Result<bool> {
        self.retriever.auto_rebuild_index_if_needed()
    }

    /// Auto-repair index integrity and compact if needed
    ///
    /// Called during maintenance to ensure storage↔index consistency:
    /// 1. Checks index health (fast O(1) operation)
    /// 2. If needs compaction (>30% deleted), triggers auto-rebuild
    /// 3. If orphaned memories detected, repairs them
    ///
    /// This provides eventual consistency between storage and index.
    fn auto_repair_and_compact(&self) {
        // Check index health first (fast operation)
        let health = self.index_health();

        // Auto-compact if deletion ratio exceeds threshold
        if health.needs_compaction {
            tracing::info!(
                "Index compaction triggered: {:.1}% deleted ({} of {} vectors)",
                health.deletion_ratio * 100.0,
                health.deleted_count,
                health.total_vectors
            );
            if let Err(e) = self.auto_rebuild_index_if_needed() {
                tracing::warn!("Index compaction failed: {}", e);
            }
        }

        // Check for orphaned memories (stored but not indexed)
        // Only do full scan if we suspect issues (cheap heuristic: counts differ)
        let storage_count = self
            .long_term_memory
            .get_stats()
            .map(|s| s.total_count)
            .unwrap_or(0);
        let index_count = health.total_vectors.saturating_sub(health.deleted_count);

        if storage_count > index_count {
            // Potential orphans detected - run repair
            let orphan_estimate = storage_count - index_count;
            if orphan_estimate > 0 {
                tracing::info!(
                    "Potential orphaned memories detected: ~{} (storage={}, indexed={})",
                    orphan_estimate,
                    storage_count,
                    index_count
                );
                match self.repair_vector_index() {
                    Ok((_, _, repaired, failed)) => {
                        if repaired > 0 || failed > 0 {
                            tracing::info!(
                                "Index repair complete: {} repaired, {} failed",
                                repaired,
                                failed
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Index repair failed: {}", e);
                    }
                }
            }
        }
    }

    /// Check resource limits to prevent OOM from single user
    ///
    /// Uses ESTIMATED_BYTES_PER_MEMORY constant for size estimation.
    /// See constants.rs for justification of the estimate.
    pub fn check_resource_limits(&self) -> Result<(), crate::errors::AppError> {
        // Get current memory counts from stats
        let stats = self.stats.read();
        let total_memories = stats.working_memory_count + stats.session_memory_count;

        // Estimate size using documented constant (see constants.rs for breakdown)
        let estimated_size_bytes = total_memories * ESTIMATED_BYTES_PER_MEMORY;
        let estimated_size_mb = estimated_size_bytes / (1024 * 1024);

        if estimated_size_mb > self.config.max_heap_per_user_mb {
            return Err(crate::errors::AppError::ResourceLimit {
                resource: "user_memory".to_string(),
                current: estimated_size_mb,
                limit: self.config.max_heap_per_user_mb,
            });
        }

        Ok(())
    }

    // =========================================================================
    // OUTCOME FEEDBACK SYSTEM - Hebbian "Fire Together, Wire Together"
    // =========================================================================

    /// Retrieve memories with tracking for later feedback
    ///
    /// Use this when you want to provide feedback on retrieval quality.
    /// Returns a TrackedRetrieval that can be used with `reinforce_retrieval`.
    ///
    /// # Example
    /// ```ignore
    /// let tracked = memory_system.retrieve_tracked(&query)?;
    /// // Use memories...
    /// // Later, after task completion:
    /// memory_system.reinforce_retrieval(&tracked.memory_ids(), RetrievalOutcome::Helpful)?;
    /// ```
    pub fn retrieve_tracked(&self, query: &Query) -> Result<TrackedRetrieval> {
        self.retriever.search_tracked(query, query.max_results)
    }

    /// Reinforce memories based on task outcome (core feedback loop)
    ///
    /// This is THE key method that closes the Hebbian loop:
    /// - If outcome is Helpful: strengthen associations, boost importance
    /// - If outcome is Misleading: weaken associations, reduce importance
    /// - If outcome is Neutral: just record access (mild reinforcement)
    ///
    /// CACHE COHERENCY: This method updates BOTH the in-memory caches AND
    /// persistent storage to ensure importance changes are visible immediately
    /// through cached references (via Arc interior mutability) AND survive restarts.
    ///
    /// # Arguments
    /// * `memory_ids` - IDs of memories that were used in the task
    /// * `outcome` - Whether the memories were helpful, misleading, or neutral
    ///
    /// # Returns
    /// Statistics about what was reinforced
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
            outcome: outcome.clone(),
            ..Default::default()
        };

        // Handle graph updates via retriever (Hebbian associations)
        match &outcome {
            RetrievalOutcome::Helpful => {
                if memory_ids.len() >= 2 {
                    self.retriever.graph_mut().record_coactivation(memory_ids);
                    stats.associations_strengthened = memory_ids.len() * (memory_ids.len() - 1) / 2;
                }
            }
            RetrievalOutcome::Neutral => {
                if memory_ids.len() >= 2 {
                    let mut graph = self.retriever.graph_mut();
                    for window in memory_ids.windows(2) {
                        graph.add_edge(&window[0], &window[1]);
                    }
                    stats.associations_strengthened = memory_ids.len() - 1;
                }
            }
            RetrievalOutcome::Misleading => {
                // Don't strengthen associations for misleading memories
            }
        }

        // CACHE COHERENT IMPORTANCE UPDATES:
        // 1. First try to find memory in caches (working, session)
        // 2. If found in cache, modify through the cached Arc (interior mutability)
        //    This updates ALL holders of this Arc reference
        // 3. Then persist to storage for durability
        // 4. If not in cache, get from storage, modify, and persist
        let mut persist_failures: Vec<(MemoryId, String)> = Vec::new();

        for id in memory_ids {
            // Try working memory cache first
            let cached_memory = {
                let working = self.working_memory.read();
                working.get(id)
            };

            // Try session memory cache if not in working
            let cached_memory = cached_memory.or_else(|| {
                let session = self.session_memory.read();
                session.get(id)
            });

            if let Some(memory) = cached_memory {
                // CACHE HIT: Modify through cached Arc (updates all references)
                memory.record_access();
                match &outcome {
                    RetrievalOutcome::Helpful => {
                        memory.boost_importance(HEBBIAN_BOOST_HELPFUL);
                        stats.importance_boosts += 1;
                    }
                    RetrievalOutcome::Misleading => {
                        memory.decay_importance(HEBBIAN_DECAY_MISLEADING);
                        stats.importance_decays += 1;
                    }
                    RetrievalOutcome::Neutral => {
                        // Just access recorded
                    }
                }
                // PERSIST: Write updated memory to durable storage
                // Track failures instead of silently ignoring
                if let Err(e) = self.long_term_memory.update(&memory) {
                    persist_failures.push((id.clone(), e.to_string()));
                    tracing::warn!(
                        memory_id = %id.0,
                        error = %e,
                        "Failed to persist reinforcement update - Hebbian feedback may be lost on restart"
                    );
                }
            } else {
                // CACHE MISS: Get from storage, modify, and persist
                match self.long_term_memory.get(id) {
                    Ok(memory) => {
                        memory.record_access();
                        match &outcome {
                            RetrievalOutcome::Helpful => {
                                memory.boost_importance(HEBBIAN_BOOST_HELPFUL);
                                stats.importance_boosts += 1;
                            }
                            RetrievalOutcome::Misleading => {
                                memory.decay_importance(HEBBIAN_DECAY_MISLEADING);
                                stats.importance_decays += 1;
                            }
                            RetrievalOutcome::Neutral => {
                                // Just access recorded
                            }
                        }
                        // PERSIST: Write to durable storage
                        if let Err(e) = self.long_term_memory.update(&memory) {
                            persist_failures.push((id.clone(), e.to_string()));
                            tracing::warn!(
                                memory_id = %id.0,
                                error = %e,
                                "Failed to persist reinforcement update - Hebbian feedback may be lost on restart"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::debug!(
                            memory_id = %id.0,
                            error = %e,
                            "Memory not found during reinforcement - may have been deleted"
                        );
                    }
                }
            }
        }

        // Report aggregate persistence failures
        if !persist_failures.is_empty() {
            stats.persist_failures = persist_failures.len();
            tracing::error!(
                failure_count = persist_failures.len(),
                "Hebbian reinforcement had persistence failures - learning feedback partially lost"
            );
        }

        Ok(stats)
    }

    /// Reinforce using a tracked retrieval (convenience wrapper)
    pub fn reinforce_tracked(
        &self,
        tracked: &TrackedRetrieval,
        outcome: RetrievalOutcome,
    ) -> Result<ReinforcementStats> {
        self.retriever.reinforce_tracked(tracked, outcome)
    }

    /// Perform graph maintenance (decay old edges, prune weak ones)
    ///
    /// Call this periodically (e.g., every hour or on user logout)
    /// to let unused associations naturally fade.
    pub fn graph_maintenance(&self) {
        self.retriever.graph_maintenance();
    }

    /// Get memory graph statistics
    pub fn graph_stats(&self) -> MemoryGraphStats {
        self.retriever.graph_stats()
    }

    // =========================================================================
    // UPSERT: Mutable memories with external linking and audit history
    // =========================================================================

    /// Upsert a memory: create if new, update with history tracking if exists
    ///
    /// When a memory with the same external_id exists:
    /// 1. Old content is pushed to history (audit trail)
    /// 2. Content is updated with new content
    /// 3. Version is incremented
    /// 4. Embeddings are regenerated for new content
    /// 5. Vector index is updated
    ///
    /// # Arguments
    /// * `external_id` - External system identifier (e.g., "linear:SHO-39", "github:pr-123")
    /// * `experience` - The experience data to store
    /// * `change_type` - Type of change (ContentUpdated, StatusChanged, etc.)
    /// * `changed_by` - Optional: who/what triggered the change
    /// * `change_reason` - Optional: description of why this changed
    ///
    /// # Returns
    /// * `(MemoryId, bool)` - Memory ID and whether it was an update (true) or create (false)
    pub fn upsert(
        &self,
        external_id: String,
        mut experience: Experience,
        change_type: ChangeType,
        changed_by: Option<String>,
        change_reason: Option<String>,
    ) -> Result<(MemoryId, bool)> {
        // Check resource limits
        self.check_resource_limits()?;

        // Try to find existing memory with this external_id
        if let Some(mut existing) = self.long_term_memory.find_by_external_id(&external_id)? {
            // === UPDATE PATH ===
            let memory_id = existing.id.clone();

            // Push old content to history and update
            existing.update_content(
                experience.content.clone(),
                change_type,
                changed_by,
                change_reason,
            );

            // Update entities if provided
            if !experience.entities.is_empty() {
                existing.experience.entities = experience.entities;
            }

            // Update tags if provided
            if !experience.tags.is_empty() {
                existing.experience.tags = experience.tags;
            }

            // Regenerate embeddings for new content
            let content_hash = Self::sha256_hash(&existing.experience.content);
            if let Some(cached_embedding) = self.content_cache.get(&content_hash) {
                existing.experience.embeddings = Some(cached_embedding.clone());
            } else {
                match self.embedder.encode(&existing.experience.content) {
                    Ok(embedding) => {
                        self.content_cache.insert(content_hash, embedding.clone());
                        existing.experience.embeddings = Some(embedding);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to regenerate embedding during upsert: {}", e);
                    }
                }
            }

            // Persist updated memory
            self.long_term_memory.update(&existing)?;

            // Re-index in vector DB with new embeddings
            if let Err(e) = self.retriever.reindex_memory(&existing) {
                tracing::warn!(
                    "Failed to reindex memory {} in vector DB: {}",
                    memory_id.0,
                    e
                );
            }

            // Update in working/session memory if cached
            {
                let mut working = self.working_memory.write();
                if working.contains(&memory_id) {
                    working.remove(&memory_id)?;
                    working.add_shared(Arc::new(existing.clone()))?;
                }
            }
            {
                let mut session = self.session_memory.write();
                if session.contains(&memory_id) {
                    session.remove(&memory_id)?;
                    session.add_shared(Arc::new(existing.clone()))?;
                }
            }

            tracing::info!(
                external_id = %external_id,
                memory_id = %memory_id.0,
                version = existing.version,
                "Memory upserted (update)"
            );

            Ok((memory_id, true))
        } else {
            // === CREATE PATH ===
            let memory_id = MemoryId(Uuid::new_v4());
            let importance = self.calculate_importance(&experience);

            // Generate embeddings if not provided
            if experience.embeddings.is_none() {
                let content_hash = Self::sha256_hash(&experience.content);
                if let Some(cached_embedding) = self.content_cache.get(&content_hash) {
                    experience.embeddings = Some(cached_embedding.clone());
                } else {
                    match self.embedder.encode(&experience.content) {
                        Ok(embedding) => {
                            self.content_cache.insert(content_hash, embedding.clone());
                            experience.embeddings = Some(embedding);
                        }
                        Err(e) => {
                            tracing::warn!("Failed to generate embedding: {}", e);
                        }
                    }
                }
            }

            // Create memory with external_id
            let memory = Arc::new(Memory::new_with_external_id(
                memory_id.clone(),
                experience,
                importance,
                external_id.clone(),
                None, // agent_id
                None, // run_id
                None, // actor_id
                None, // created_at
            ));

            // Persist to storage
            self.long_term_memory.store(&memory)?;

            // Log creation
            self.logger.write().log_created(&memory, "working");

            // Add to working memory
            self.working_memory
                .write()
                .add_shared(Arc::clone(&memory))?;

            // Index in vector DB
            if let Err(e) = self.retriever.index_memory(&memory) {
                tracing::warn!("Failed to index memory {} in vector DB: {}", memory.id.0, e);
            }

            // Add to knowledge graph
            self.retriever.add_to_graph(&memory);

            // Add to session if important
            if importance > self.config.importance_threshold {
                self.session_memory
                    .write()
                    .add_shared(Arc::clone(&memory))?;
            }

            // Update stats
            self.stats.write().total_memories += 1;

            tracing::info!(
                external_id = %external_id,
                memory_id = %memory_id.0,
                "Memory upserted (create)"
            );

            Ok((memory_id, false))
        }
    }

    /// Get the history of a memory (audit trail of changes)
    ///
    /// Returns the full revision history for memories with external linking.
    /// Returns empty vec for regular (non-mutable) memories.
    pub fn get_memory_history(&self, memory_id: &MemoryId) -> Result<Vec<MemoryRevision>> {
        let memory = self.long_term_memory.get(memory_id)?;
        Ok(memory.history.clone())
    }

    /// Find a memory by external ID
    ///
    /// Used to check if a memory already exists for an external entity
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<Memory>> {
        self.long_term_memory.find_by_external_id(external_id)
    }

    /// Run periodic maintenance (consolidation, activation decay, graph maintenance)
    ///
    /// Call this periodically (e.g., every 5 minutes) to:
    /// 1. Promote memories between tiers based on thresholds
    /// 2. Decay activation levels on all memories
    /// 3. Run graph maintenance (prune weak edges)
    ///
    /// Returns the number of memories processed for activation decay.
    /// Also records consolidation events for introspection.
    pub fn run_maintenance(&self, decay_factor: f32) -> Result<usize> {
        let start_time = std::time::Instant::now();
        let now = chrono::Utc::now();

        // 1. Consolidation: promote memories between tiers
        self.consolidate_if_needed()?;

        // 2. Decay activation on all in-memory memories (working + session)
        let mut decayed_count = 0;
        let mut at_risk_count = 0;
        const AT_RISK_THRESHOLD: f32 = 0.2; // Memories below this are at risk of being forgotten

        // Decay working memory activations with event tracking
        {
            let working = self.working_memory.read();
            for memory in working.all_memories() {
                let activation_before = memory.activation();
                memory.decay_activation(decay_factor);
                let activation_after = memory.activation();
                decayed_count += 1;

                // Only record event if there was actual decay
                if activation_before != activation_after {
                    let at_risk = activation_after < AT_RISK_THRESHOLD;
                    if at_risk {
                        at_risk_count += 1;
                    }

                    // Record decay event
                    self.record_consolidation_event(ConsolidationEvent::MemoryDecayed {
                        memory_id: memory.id.0.to_string(),
                        content_preview: memory.experience.content.chars().take(50).collect(),
                        activation_before,
                        activation_after,
                        at_risk,
                        timestamp: now,
                    });
                }
            }
        }

        // Decay session memory activations with event tracking
        {
            let session = self.session_memory.read();
            for memory in session.all_memories() {
                let activation_before = memory.activation();
                memory.decay_activation(decay_factor);
                let activation_after = memory.activation();
                decayed_count += 1;

                // Only record event if there was actual decay
                if activation_before != activation_after {
                    let at_risk = activation_after < AT_RISK_THRESHOLD;
                    if at_risk {
                        at_risk_count += 1;
                    }

                    // Record decay event
                    self.record_consolidation_event(ConsolidationEvent::MemoryDecayed {
                        memory_id: memory.id.0.to_string(),
                        content_preview: memory.experience.content.chars().take(50).collect(),
                        activation_before,
                        activation_after,
                        at_risk,
                        timestamp: now,
                    });
                }
            }
        }

        // 3. Graph maintenance: prune weak edges
        // Note: Graph maintenance doesn't currently report pruned edges,
        // but the retriever could be modified to return pruning stats
        self.graph_maintenance();

        // 4. Auto-repair index integrity and compact if needed
        // This ensures storage↔index sync and prevents memory leaks from soft-deleted vectors
        self.auto_repair_and_compact();

        let duration_ms = start_time.elapsed().as_millis() as u64;

        // Record maintenance cycle completion event
        self.record_consolidation_event(ConsolidationEvent::MaintenanceCycleCompleted {
            memories_processed: decayed_count,
            memories_decayed: decayed_count, // All memories get decay applied
            edges_pruned: 0,                 // Graph maintenance doesn't report this yet
            duration_ms,
            timestamp: now,
        });

        tracing::debug!(
            "Maintenance complete: {} memories decayed (factor={}), {} at risk, took {}ms",
            decayed_count,
            decay_factor,
            at_risk_count,
            duration_ms
        );

        Ok(decayed_count)
    }

    // =========================================================================
    // CONSOLIDATION INTROSPECTION API
    // =========================================================================

    /// Get a consolidation report for a time period
    ///
    /// Shows what the memory system has been learning:
    /// - Which memories strengthened or decayed
    /// - What associations formed or were pruned
    /// - What facts were extracted or reinforced
    ///
    /// # Arguments
    /// * `since` - Start of the time period
    /// * `until` - End of the time period (default: now)
    pub fn get_consolidation_report(
        &self,
        since: chrono::DateTime<chrono::Utc>,
        until: Option<chrono::DateTime<chrono::Utc>>,
    ) -> ConsolidationReport {
        let until = until.unwrap_or_else(chrono::Utc::now);
        let events = self.consolidation_events.read();
        events.generate_report(since, until)
    }

    /// Get all consolidation events since a timestamp
    ///
    /// Returns raw events for detailed analysis
    pub fn get_consolidation_events_since(
        &self,
        since: chrono::DateTime<chrono::Utc>,
    ) -> Vec<ConsolidationEvent> {
        let events = self.consolidation_events.read();
        events.events_since(since)
    }

    /// Get all consolidation events in the buffer
    pub fn get_all_consolidation_events(&self) -> Vec<ConsolidationEvent> {
        let events = self.consolidation_events.read();
        events.all_events()
    }

    /// Record a consolidation event
    ///
    /// Used internally by the memory system to log learning events.
    /// Also available for external callers that want to track custom events.
    pub fn record_consolidation_event(&self, event: ConsolidationEvent) {
        let mut events = self.consolidation_events.write();
        events.push(event);
    }

    /// Clear all consolidation events
    pub fn clear_consolidation_events(&self) {
        let mut events = self.consolidation_events.write();
        events.clear();
    }

    /// Get the number of consolidation events in the buffer
    pub fn consolidation_event_count(&self) -> usize {
        let events = self.consolidation_events.read();
        events.len()
    }
}

/// Automatic persistence on drop - ensures vector index and ID mappings survive restarts
///
/// This is CRITICAL for local memory: when the system shuts down (gracefully or via drop),
/// all in-memory state (vector index, ID mappings) must be persisted to disk.
impl Drop for MemorySystem {
    fn drop(&mut self) {
        // Persist vector index and ID mapping for restart recovery
        if let Err(e) = self.retriever.save() {
            tracing::error!("Failed to persist vector index on shutdown: {}", e);
        } else {
            tracing::info!("Vector index persisted successfully on shutdown");
        }

        // Flush RocksDB WAL to ensure all writes are durable
        if let Err(e) = self.long_term_memory.flush() {
            tracing::error!("Failed to flush storage on shutdown: {}", e);
        }
    }
}
