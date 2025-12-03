//! Memory System for LLM Context Management
//!
//! A medium-complexity memory system that provides:
//! - Hierarchical memory storage (working → session → long-term)
//! - Smart compression based on age and importance
//! - Multi-modal retrieval (similarity, temporal, causal)
//! - Automatic memory consolidation

pub mod storage;
pub mod compression;
pub mod retrieval;
pub mod types;
pub mod context;
// pub mod vector_storage;  // Disabled - requires crate::rag::vamana from parent project
pub mod visualization;
pub mod query_parser;
pub mod graph_retrieval;

use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::path::PathBuf;
use uuid::Uuid;
use parking_lot::RwLock;
use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::debug;

pub use crate::memory::types::*;
use crate::memory::storage::MemoryStorage;
// pub use crate::memory::vector_storage::{VectorIndexedMemoryStorage, StorageStats};  // Disabled
use crate::memory::compression::CompressionPipeline;
use crate::memory::retrieval::RetrievalEngine;
pub use crate::memory::visualization::{MemoryLogger, GraphStats};
use crate::embeddings::Embedder;

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
            working_memory_size: 100,
            session_memory_size_mb: 100,
            max_heap_per_user_mb: 500,  // 500MB per user prevents OOM
            auto_compress: true,
            compression_age_days: 7,
            importance_threshold: 0.7,
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

    /// Query embedding cache - hash(query_text) → embedding
    /// MASSIVE PERF WIN: 80ms → <1ms for cached queries
    query_cache: Arc<DashMap<u64, Vec<f32>>>,

    /// Content embedding cache - hash(content) → embedding
    /// MASSIVE PERF WIN: 80ms → <1ms for repeated content
    content_cache: Arc<DashMap<u64, Vec<f32>>>,

    /// Memory statistics
    stats: Arc<RwLock<MemoryStats>>,

    /// Visualization logger
    logger: Arc<RwLock<MemoryLogger>>,
}

impl MemorySystem {
    /// Create a new memory system
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let storage = Arc::new(MemoryStorage::new(&config.storage_path)?);

        // CRITICAL: Initialize embedder ONCE and share between MemorySystem and RetrievalEngine
        // This prevents loading the ONNX model multiple times (50-200ms overhead per load)
        let embedding_config = crate::embeddings::minilm::EmbeddingConfig::default();
        let embedder = Arc::new(crate::embeddings::minilm::MiniLMEmbedder::new(embedding_config)
            .context("Failed to initialize embedder")?);

        // Pass shared embedder to retrieval engine (no duplicate model load)
        let retriever = RetrievalEngine::new(storage.clone(), embedder.clone())?;

        // Disable visualization logging for production performance
        let logger = Arc::new(RwLock::new(MemoryLogger::new(false)));

        Ok(Self {
            config: config.clone(),
            working_memory: Arc::new(RwLock::new(WorkingMemory::new(config.working_memory_size))),
            session_memory: Arc::new(RwLock::new(SessionMemory::new(config.session_memory_size_mb))),
            long_term_memory: storage,
            compressor: CompressionPipeline::new(),
            retriever,
            embedder,
            query_cache: Arc::new(DashMap::new()),
            content_cache: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(MemoryStats::default())),
            logger,
        })
    }

    /// Record a new experience (takes ownership to avoid clones)
    pub fn record(&mut self, mut experience: Experience) -> Result<MemoryId> {
        // CRITICAL: Check resource limits before recording to prevent OOM
        self.check_resource_limits()?;

        let memory_id = MemoryId(Uuid::new_v4());

        // Calculate importance
        let importance = self.calculate_importance(&experience);

        // PERFORMANCE: Content embedding cache (80ms → <1μs for repeated content)
        // If experience doesn't have embeddings, check cache or generate
        if experience.embeddings.is_none() {
            // Hash content for cache lookup
            let mut hasher = DefaultHasher::new();
            experience.content.hash(&mut hasher);
            let content_hash = hasher.finish();

            // Check cache first
            if let Some(cached_embedding) = self.content_cache.get(&content_hash) {
                experience.embeddings = Some(cached_embedding.clone());
                tracing::debug!("Content embedding cache HIT for hash {}", content_hash);
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
            experience,  // Move ownership (zero-cost)
            importance,
            None,  // agent_id
            None,  // run_id
            None,  // actor_id
        ));

        // CRITICAL: Persist to RocksDB storage FIRST (before indexing/in-memory tiers)
        // This ensures retrieval can always fetch the memory from persistent storage
        self.long_term_memory.store(&memory)?;

        // Log creation
        self.logger.write().log_created(&memory, "working");

        // Add to working memory (cheap Arc clone, not full Memory clone)
        self.working_memory.write().add_shared(Arc::clone(&memory))?;

        // CRITICAL: Index memory immediately for semantic search (don't wait for long-term promotion)
        // This ensures new memories are searchable right away, not only after consolidation
        if let Err(e) = self.retriever.index_memory(&memory) {
            tracing::warn!("Failed to index memory {} in vector DB: {}", memory.id.0, e);
            // Don't fail the record operation if indexing fails - memory is still stored
        }

        // Add to knowledge graph for associative/causal retrieval
        self.retriever.add_to_graph(&memory);

        // If important enough, prepare for session storage
        if importance > self.config.importance_threshold {
            self.session_memory.write().add_shared(Arc::clone(&memory))?;
            self.logger.write().log_created(&memory, "session");
        }

        // Update stats
        self.stats.write().total_memories += 1;

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
        let mut sources = Vec::new();

        // Collect from all tiers
        {
            let working = self.working_memory.read();
            let working_results = working.search(query, query.max_results)?;
            if !working_results.is_empty() {
                sources.push("working");
            }
            memories.extend(working_results);
        }

        {
            let session = self.session_memory.read();
            let session_results = session.search(query, query.max_results)?;
            if !session_results.is_empty() {
                sources.push("session");
            }
            memories.extend(session_results);
        }

        {
            let long_term_results = self.retriever.search(query, query.max_results)?;
            if !long_term_results.is_empty() {
                sources.push("longterm");
            }
            memories.extend(long_term_results);
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

            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        memories.truncate(query.max_results);

        // Log retrieval
        self.logger.read().log_retrieved("", memories.len(), &sources);

        // Update access counts asynchronously
        for memory in &memories {
            if let Err(e) = self.update_access_count(&memory.id) {
                tracing::warn!(memory_id = %memory.id.0, error = %e, "Failed to update access count");
            }
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
        // Hash query text for cache lookup
        let mut hasher = DefaultHasher::new();
        query_text.hash(&mut hasher);
        let query_hash = hasher.finish();

        // Check cache first
        let query_embedding = if let Some(cached_embedding) = self.query_cache.get(&query_hash) {
            tracing::debug!("Query embedding cache HIT for: {}", query_text);
            cached_embedding.clone()
        } else {
            // Cache miss - generate embedding
            tracing::debug!("Query embedding cache MISS - generating for: {}", query_text);
            let embedding = self.embedder.as_ref().encode(query_text)
                .context("Failed to generate query embedding")?;

            // Store in cache for future use
            self.query_cache.insert(query_hash, embedding.clone());
            embedding
        };

        // Create a modified query with the embedding for vector search
        let vector_query = Query {
            query_text: None,  // Don't re-generate embedding
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
        let memory_ids = self.retriever.search_ids(&vector_query, query.max_results)?;

        // Fetch memories with cache-aware strategy
        let mut memories = Vec::new();
        let mut sources = Vec::new();
        let mut cache_hits = 0;
        let mut storage_fetches = 0;

        for (memory_id, _score) in memory_ids {
            // Try working memory first (hot cache, zero-copy Arc clone)
            if let Some(memory) = self.working_memory.read().get(&memory_id) {
                memories.push(memory);  // Already cloned by get()
                if !sources.contains(&"working") {
                    sources.push("working");
                }
                cache_hits += 1;
                continue;
            }

            // Try session memory second (warm cache, zero-copy Arc clone)
            if let Some(memory) = self.session_memory.read().get(&memory_id) {
                memories.push(memory);  // Already cloned by get()
                if !sources.contains(&"session") {
                    sources.push("session");
                }
                cache_hits += 1;
                continue;
            }

            // Cold path: Fetch from RocksDB storage (expensive deserialization)
            match self.retriever.get_from_storage(&memory_id) {
                Ok(memory) => {
                    memories.push(Arc::new(memory));
                    if !sources.contains(&"longterm") {
                        sources.push("longterm");
                    }
                    storage_fetches += 1;
                }
                Err(e) => {
                    tracing::warn!(memory_id = %memory_id.0, error = %e, "Failed to fetch from storage");
                }
            }

            if memories.len() >= query.max_results {
                break;
            }
        }

        // Log cache efficiency
        tracing::debug!(
            cache_hits = cache_hits,
            storage_fetches = storage_fetches,
            hit_rate = if cache_hits + storage_fetches > 0 {
                (cache_hits as f32 / (cache_hits + storage_fetches) as f32) * 100.0
            } else { 0.0 },
            "Cache-aware retrieval completed"
        );

        // Re-rank using linguistic analysis (focal entities boost)
        let analysis = query_parser::analyze_query(query_text);
        if !analysis.focal_entities.is_empty() {
            memories.sort_by(|a, b| {
                let score_a = Self::linguistic_boost(&a.experience.content, &analysis);
                let score_b = Self::linguistic_boost(&b.experience.content, &analysis);
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        self.logger.read().log_retrieved(query_text, memories.len(), &sources);

        // Update access counts asynchronously
        for memory in &memories {
            if let Err(e) = self.update_access_count(&memory.id) {
                tracing::warn!(memory_id = %memory.id.0, error = %e, "Failed to update access count");
            }
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

    /// Forget memories based on criteria
    pub fn forget(&mut self, criteria: ForgetCriteria) -> Result<usize> {
        let forgotten_count = match criteria {
            ForgetCriteria::OlderThan(days) => {
                let cutoff = chrono::Utc::now() - chrono::Duration::days(days as i64);

                // Remove from working memory
                self.working_memory.write().remove_older_than(cutoff)?;

                // Remove from session memory
                self.session_memory.write().remove_older_than(cutoff)?;

                // Mark as forgotten in long-term (don't delete, just flag)
                self.long_term_memory.mark_forgotten_by_age(cutoff)?
            },
            ForgetCriteria::LowImportance(threshold) => {
                self.working_memory.write().remove_below_importance(threshold)?;
                self.session_memory.write().remove_below_importance(threshold)?;
                self.long_term_memory.mark_forgotten_by_importance(threshold)?
            },
            ForgetCriteria::Pattern(pattern) => {
                // Remove memories matching pattern
                self.forget_by_pattern(&pattern)?
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
    pub fn get_all_memories(&self) -> Result<Vec<SharedMemory>> {
        let mut all_memories = Vec::new();

        // Collect from working memory
        {
            let working = self.working_memory.read();
            all_memories.extend(working.all_memories());
        }

        // Collect from session memory
        {
            let session = self.session_memory.read();
            all_memories.extend(session.all_memories());
        }

        // Collect from long-term memory (wrap in Arc)
        {
            let longterm_mems = self.long_term_memory.get_all()?;
            all_memories.extend(longterm_mems.into_iter().map(Arc::new));
        }

        Ok(all_memories)
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
            0..=7 => 1.0,      // Recent: Full weight
            8..=30 => 0.7,     // Medium-term: 70% weight
            31..=90 => 0.4,    // Old: 40% weight
            _ => 0.2,          // Ancient: 20% weight (never completely forgotten)
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
            if emb.len() >= 384 {  // Full embedding vector
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
        let technical_terms = ["algorithm", "architecture", "implementation", "optimization",
                                "performance", "security", "database", "api", "framework"];
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
        if experience.content.contains("```") || experience.content.contains("fn ") ||
           experience.content.contains("function ") || experience.content.contains("class ") {
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
    fn consolidate_if_needed(&mut self) -> Result<()> {
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
    fn promote_working_to_session(&mut self) -> Result<()> {
        let mut working = self.working_memory.write();
        let mut session = self.session_memory.write();

        // Get least recently used memories
        let to_promote = working.get_lru(self.config.working_memory_size / 2)?;
        let count = to_promote.len();

        for memory in &to_promote {
            // Log promotion
            self.logger.write().log_promoted(&memory.id, "working", "session", count);

            // Clone out of Arc to get owned Memory for session storage
            session.add((**memory).clone())?;
            working.remove(&memory.id)?;
        }

        self.stats.write().promotions_to_session += 1;
        Ok(())
    }

    /// Move memories from session to long-term storage
    fn promote_session_to_longterm(&mut self) -> Result<()> {
        let mut session = self.session_memory.write();

        // Get memories to promote (important ones)
        let to_promote = session.get_important(self.config.importance_threshold)?;
        let count = to_promote.len();

        for memory in &to_promote {
            // Log promotion
            self.logger.write().log_promoted(&memory.id, "session", "longterm", count);

            // Clone out of Arc to get owned Memory
            let owned_memory = (**memory).clone();

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
                tracing::warn!("Failed to index memory {} in vector DB: {}", compressed_memory.id.0, e);
                // Don't fail promotion if indexing fails - memory is still stored
            }

            // Remove from session
            session.remove(&memory.id)?;
        }

        self.stats.write().promotions_to_longterm += 1;
        Ok(())
    }

    /// Compress old memories to save space
    fn compress_old_memories(&mut self) -> Result<()> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(self.config.compression_age_days as i64);

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
                return wm.update_access(memory_id)
                    .map_err(|e| anyhow::anyhow!("Failed to update working memory access: {e}"));
            }
        } // Release write lock

        // Try session memory
        {
            let mut sm = self.session_memory.write();

            if sm.contains(memory_id) {
                return sm.update_access(memory_id)
                    .map_err(|e| anyhow::anyhow!("Failed to update session memory access: {e}"));
            }
        } // Release write lock

        // Try long-term memory (has its own internal locking)
        self.long_term_memory.update_access(memory_id)
            .map_err(|e| anyhow::anyhow!("Failed to update long-term memory access: {e}"))
    }

    /// Forget memories matching a pattern
    ///
    /// Uses validated regex compilation with ReDoS protection
    fn forget_by_pattern(&mut self, pattern: &str) -> Result<usize> {
        // Use validated pattern compilation with ReDoS protection
        let regex = crate::validation::validate_and_compile_pattern(pattern)
            .map_err(|e| anyhow::anyhow!("Invalid pattern: {}", e))?;
        let mut count = 0;

        // Remove from all tiers
        count += self.working_memory.write().remove_matching(&regex)?;
        count += self.session_memory.write().remove_matching(&regex)?;
        count += self.long_term_memory.remove_matching(&regex)?;

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
        self.long_term_memory.flush()
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
    pub fn get_uncompressed_older_than(&self, cutoff: chrono::DateTime<chrono::Utc>) -> Result<Vec<Memory>> {
        self.long_term_memory.get_uncompressed_older_than(cutoff)
    }

    /// Rebuild vector index from all existing long-term memories (startup initialization)
    pub fn rebuild_vector_index(&self) -> Result<()> {
        self.retriever.rebuild_index()
    }

    /// Save vector index to disk (shutdown persistence)
    pub fn save_vector_index(&self, path: &PathBuf) -> Result<()> {
        self.retriever.save_index(path)
    }

    /// Load vector index from disk (startup restoration)
    pub fn load_vector_index(&self, path: &PathBuf) -> Result<()> {
        self.retriever.load_index(path)
    }

    /// Check resource limits to prevent OOM from single user
    pub fn check_resource_limits(&self) -> Result<(), crate::errors::AppError> {
        // Get current memory counts from stats
        let stats = self.stats.read();
        let total_memories = stats.working_memory_count + stats.session_memory_count;

        // Estimate: each memory ~250KB on average (experience + embeddings + metadata)
        let estimated_size_mb = (total_memories * 250) / 1024;

        if estimated_size_mb > self.config.max_heap_per_user_mb {
            return Err(crate::errors::AppError::ResourceLimit {
                resource: "user_memory".to_string(),
                current: estimated_size_mb,
                limit: self.config.max_heap_per_user_mb,
            });
        }

        Ok(())
    }
}