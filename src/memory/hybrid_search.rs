//! Hybrid Search Pipeline (BM25 + Vector + Cognitive)
//!
//! Production-grade retrieval combining:
//! 1. BM25 full-text search (tantivy) - keyword matching
//! 2. Vector search (Vamana) - semantic similarity
//! 3. Reciprocal Rank Fusion (RRF) - signal combination
//! 4. Cross-encoder reranking - accurate top-k scoring
//! 5. Cognitive signals - Hebbian strength, decay, feedback momentum
//!
//! Architecture:
//! ```text
//! Query → [BM25] ──┐
//!                  ├─→ [RRF Fusion] → [Cross-Encoder] → [Cognitive Boost] → Results
//! Query → [Vector] ┘
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{Index, IndexReader, IndexWriter, TantivyDocument};
use tracing::{debug, info};

use super::types::MemoryId;
use crate::embeddings::minilm::MiniLMEmbedder;

/// Configuration for hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Weight for BM25 scores in RRF (0.0-1.0)
    #[serde(default = "default_bm25_weight")]
    pub bm25_weight: f32,

    /// Weight for vector scores in RRF (0.0-1.0)
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,

    /// DEPRECATED: Graph fusion happens in mod.rs Layer 4 (not in hybrid_search).
    /// This field is kept for deserialization compatibility but is not used.
    /// Actual graph weight is computed dynamically by graph_retrieval.rs based on density.
    #[serde(default = "default_graph_weight")]
    pub graph_weight: f32,

    /// RRF constant k (higher = more equal weighting)
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,

    /// Number of candidates to fetch from each retriever
    #[serde(default = "default_candidate_count")]
    pub candidate_count: usize,

    /// Minimum BM25 score to consider (filters noise)
    #[serde(default = "default_min_bm25_score")]
    pub min_bm25_score: f32,

    /// Minimum graph activation score to consider (SHO-D4)
    #[serde(default = "default_min_graph_score")]
    pub min_graph_score: f32,
}

fn default_bm25_weight() -> f32 {
    0.35 // Reduced: BM25 over-matches common terms (names), diluting rare discriminative terms
}
fn default_vector_weight() -> f32 {
    0.40 // Vector similarity handles semantic relationships
}
fn default_graph_weight() -> f32 {
    0.25 // Graph spreading activation for associative retrieval (SHO-D4)
}
fn default_rrf_k() -> f32 {
    crate::constants::RRF_K_HYBRID_FUSION
}
fn default_candidate_count() -> usize {
    100 // Increased for better recall; slight latency tradeoff acceptable
}
fn default_min_bm25_score() -> f32 {
    0.01 // Lower threshold to capture more keyword matches
}
fn default_min_graph_score() -> f32 {
    0.01 // Lower threshold to capture graph-based associations (SHO-D4)
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            bm25_weight: default_bm25_weight(),
            vector_weight: default_vector_weight(),
            graph_weight: default_graph_weight(),
            rrf_k: default_rrf_k(),
            candidate_count: default_candidate_count(),
            min_bm25_score: default_min_bm25_score(),
            min_graph_score: default_min_graph_score(),
        }
    }
}

/// Result from hybrid search with component scores
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// Memory ID
    pub memory_id: MemoryId,

    /// Final combined score (0.0-1.0)
    pub score: f32,

    /// BM25 score (if matched)
    pub bm25_score: Option<f32>,

    /// Vector similarity score (if matched)
    pub vector_score: Option<f32>,

    /// Graph activation score from spreading activation (if matched) (SHO-D4)
    pub graph_score: Option<f32>,

    /// RRF score before post-processing
    pub rrf_score: f32,

    /// Rank from BM25 (if matched)
    pub bm25_rank: Option<usize>,

    /// Rank from vector search (if matched)
    pub vector_rank: Option<usize>,

    /// Rank from graph spreading activation (if matched) (SHO-D4)
    pub graph_rank: Option<usize>,
}

/// BM25 Index using Tantivy
pub struct BM25Index {
    index: Index,
    reader: IndexReader,
    writer: Arc<RwLock<IndexWriter>>,
    id_field: Field,
    content_field: Field,
    tags_field: Field,
    entities_field: Field,
}

impl BM25Index {
    /// Create or open a BM25 index at the given path
    pub fn new(path: &Path) -> Result<Self> {
        let mut schema_builder = Schema::builder();

        // Memory ID (stored, not tokenized)
        schema_builder.add_text_field("id", STRING | STORED);

        // Main content (tokenized for BM25)
        schema_builder.add_text_field("content", TEXT | STORED);

        // Tags (tokenized)
        schema_builder.add_text_field("tags", TEXT);

        // Entities (tokenized)
        schema_builder.add_text_field("entities", TEXT);

        let schema = schema_builder.build();

        // Create or open index
        std::fs::create_dir_all(path)?;
        let dir = tantivy::directory::MmapDirectory::open(path)
            .context("Failed to open tantivy directory")?;

        let index = if Index::exists(&dir)? {
            Index::open(dir).context("Failed to open existing BM25 index")?
        } else {
            Index::create_in_dir(path, schema).context("Failed to create BM25 index")?
        };

        // Resolve field handles from the index's actual schema (which may have
        // been loaded from disk). Using builder-created handles would be wrong
        // if the on-disk schema has different field IDs due to schema evolution.
        let actual_schema = index.schema();
        let id_field = actual_schema
            .get_field("id")
            .context("BM25 schema missing 'id' field")?;
        let content_field = actual_schema
            .get_field("content")
            .context("BM25 schema missing 'content' field")?;
        let tags_field = actual_schema
            .get_field("tags")
            .context("BM25 schema missing 'tags' field")?;
        let entities_field = actual_schema
            .get_field("entities")
            .context("BM25 schema missing 'entities' field")?;

        // 15MB writer heap — sufficient for edge workloads
        let writer = index
            .writer(15_000_000)
            .context("Failed to create index writer")?;

        let reader = index
            .reader_builder()
            .reload_policy(tantivy::ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create index reader")?;

        info!("BM25 index initialized at {:?}", path);

        Ok(Self {
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            id_field,
            content_field,
            tags_field,
            entities_field,
        })
    }

    /// Add or update a document in the index
    pub fn upsert(
        &self,
        memory_id: &MemoryId,
        content: &str,
        tags: &[String],
        entities: &[String],
    ) -> Result<()> {
        let writer = self.writer.write();

        // Delete existing document with this ID
        let id_term = tantivy::Term::from_field_text(self.id_field, &memory_id.0.to_string());
        writer.delete_term(id_term);

        // Create new document
        let mut doc = TantivyDocument::new();
        doc.add_text(self.id_field, memory_id.0.to_string());
        doc.add_text(self.content_field, content);
        doc.add_text(self.tags_field, tags.join(" "));
        doc.add_text(self.entities_field, entities.join(" "));

        writer.add_document(doc)?;

        Ok(())
    }

    /// Remove a document from the index
    pub fn delete(&self, memory_id: &MemoryId) -> Result<()> {
        let writer = self.writer.write();
        let id_term = tantivy::Term::from_field_text(self.id_field, &memory_id.0.to_string());
        writer.delete_term(id_term);
        Ok(())
    }

    /// Commit pending changes to disk
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.write();
        writer.commit().context("Failed to commit BM25 index")?;
        Ok(())
    }

    /// Search using BM25
    ///
    /// Returns (memory_id, score) pairs sorted by score descending
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(MemoryId, f32)>> {
        self.search_with_term_weights(query, limit, None)
    }

    /// Search using BM25 with IC-weighted term boosting
    ///
    /// Term weights are derived from linguistic Information Content (IC):
    /// - Nouns: IC_NOUN = 1.5 (focal entities, highest weight)
    /// - Adjectives: IC_ADJECTIVE = 0.9 (discriminative modifiers)
    /// - Verbs: IC_VERB = 0.7 (relational context)
    ///
    /// The weights are applied as Tantivy boost operators (term^weight)
    pub fn search_with_term_weights(
        &self,
        query: &str,
        limit: usize,
        term_weights: Option<&HashMap<String, f32>>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        self.search_with_term_and_phrase_weights(query, limit, term_weights, None)
    }

    /// Search with IC-weighted term boosting AND phrase matching
    ///
    /// Phrase boosts significantly improve retrieval for multi-word concepts:
    /// - "support group" matches exact phrase, not just "support" OR "group"
    /// - Reduces false positives from partial term matches
    pub fn search_with_term_and_phrase_weights(
        &self,
        query: &str,
        limit: usize,
        term_weights: Option<&HashMap<String, f32>>,
        phrase_boosts: Option<&[(String, f32)]>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();

        // Parse query across content, tags, and entities fields
        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.content_field, self.tags_field, self.entities_field],
        );

        // Build boosted query with term weights
        let mut query_parts: Vec<String> = Vec::new();

        // Add individual terms with IC weights.
        // Strip ALL non-alphanumeric characters (not just at boundaries) to prevent
        // Tantivy query syntax injection (+, -, ^, ~, etc. have special meaning).
        if let Some(weights) = term_weights {
            for word in query.split_whitespace() {
                let clean_word: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                if clean_word.is_empty() {
                    continue;
                }
                if let Some(&weight) = weights.get(&clean_word) {
                    // Apply boost (Tantivy uses ^ for boost, like Lucene)
                    query_parts.push(format!("{}^{:.1}", clean_word, weight));
                } else {
                    query_parts.push(clean_word);
                }
            }
        } else {
            // No term weights - add words as-is
            for word in query.split_whitespace() {
                let clean_word: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                if !clean_word.is_empty() {
                    query_parts.push(clean_word);
                }
            }
        }

        // Add phrase queries with boosts (e.g., "support group"^2.0)
        // Phrase queries provide significant boost when exact phrase is found
        if let Some(phrases) = phrase_boosts {
            for (phrase, boost) in phrases {
                // Tantivy phrase query syntax: "word1 word2"^boost
                // Only add if phrase has multiple words and doesn't contain special chars
                if phrase.contains(' ') && !phrase.contains('"') {
                    query_parts.push(format!("\"{}\"^{:.1}", phrase, boost));
                }
            }
        }

        let boosted_query = query_parts.join(" ");

        // Handle query parsing errors gracefully
        let parsed_query = match query_parser.parse_query(&boosted_query) {
            Ok(q) => q,
            Err(e) => {
                debug!("BM25 query parse error for '{}': {}", boosted_query, e);
                // Fall back to simple term query without boosts
                let escaped = query.replace(
                    [
                        ':', '^', '~', '*', '?', '[', ']', '{', '}', '(', ')', '"', '\\', '/', '+',
                        '-', '!', '&', '|',
                    ],
                    " ",
                );
                match query_parser.parse_query(&escaped) {
                    Ok(q) => q,
                    Err(_) => return Ok(Vec::new()),
                }
            }
        };

        let top_docs = searcher
            .search(&parsed_query, &TopDocs::with_limit(limit))
            .context("BM25 search failed")?;

        let mut results = Vec::with_capacity(top_docs.len());

        for (score, doc_address) in top_docs {
            if let Ok(doc) = searcher.doc::<TantivyDocument>(doc_address) {
                if let Some(id_value) = doc.get_first(self.id_field) {
                    if let Some(id_str) = id_value.as_str() {
                        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                            results.push((MemoryId(uuid), score));
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get document count
    pub fn len(&self) -> usize {
        let searcher = self.reader.searcher();
        searcher.num_docs() as usize
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reload the reader to see committed changes
    pub fn reload(&self) -> Result<()> {
        self.reader.reload()?;
        Ok(())
    }

    /// Merge all segments into a single segment.
    ///
    /// Tantivy accumulates segments from commits. Deleted documents (from upserts)
    /// remain as tombstones until segments are merged. Without periodic merging:
    /// - Ghost state from overwrites pollutes BM25 scoring
    /// - Disk usage grows unboundedly
    /// - Search latency increases with segment count
    ///
    /// This is expensive (rewrites entire index) so should only run on heavy
    /// maintenance cycles, not per-request.
    pub fn optimize(&self) -> Result<usize> {
        let segment_count = {
            let searcher = self.reader.searcher();
            searcher.segment_readers().len()
        };

        if segment_count <= 1 {
            return Ok(0);
        }

        let mut writer = self.writer.write();

        // Collect all segment IDs for merging
        let segment_ids: Vec<_> = self
            .index
            .searchable_segment_ids()
            .context("Failed to get segment IDs")?;

        if segment_ids.len() <= 1 {
            return Ok(0);
        }

        let merged = segment_ids.len();
        writer
            .merge(&segment_ids)
            .wait()
            .context("Failed to merge BM25 segments")?;
        writer
            .commit()
            .context("Failed to commit after BM25 merge")?;
        drop(writer);

        self.reader.reload()?;

        tracing::info!("BM25 index optimized: merged {} segments into 1", merged);

        Ok(merged)
    }

    /// Get the current number of segments (for health/metrics)
    pub fn segment_count(&self) -> usize {
        let searcher = self.reader.searcher();
        searcher.segment_readers().len()
    }
}

/// Reciprocal Rank Fusion (RRF) implementation
///
/// Combines rankings from multiple retrievers using:
/// RRF(d) = Σ 1/(k + rank_i(d))
///
/// Where k is a constant (typically 60) that controls how much
/// weight is given to documents ranked lower.
pub struct RRFusion {
    /// RRF constant k
    k: f32,
    /// Weight for each retriever (normalized)
    weights: Vec<f32>,
}

impl RRFusion {
    /// Create new RRF with given k and weights
    pub fn new(k: f32, weights: Vec<f32>) -> Self {
        // Normalize weights
        let sum: f32 = weights.iter().sum();
        let normalized = if sum > 0.0 {
            weights.iter().map(|w| w / sum).collect()
        } else {
            vec![1.0 / weights.len() as f32; weights.len()]
        };

        Self {
            k,
            weights: normalized,
        }
    }

    /// Fuse multiple ranked lists into a single ranking
    ///
    /// Each input is a Vec of (MemoryId, score) sorted by score descending.
    /// Returns fused (MemoryId, rrf_score) sorted by rrf_score descending.
    pub fn fuse(&self, ranked_lists: Vec<Vec<(MemoryId, f32)>>) -> Vec<(MemoryId, f32)> {
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();
        let mut original_scores: HashMap<MemoryId, Vec<Option<f32>>> = HashMap::new();

        for (list_idx, ranked_list) in ranked_lists.iter().enumerate() {
            let weight = self.weights.get(list_idx).copied().unwrap_or(1.0);

            for (rank, (memory_id, score)) in ranked_list.iter().enumerate() {
                // RRF formula: weight * 1/(k + rank)
                // rank is 0-indexed, so rank+1 for 1-indexed
                let rrf_contribution = weight / (self.k + (rank + 1) as f32);

                *scores.entry(memory_id.clone()).or_insert(0.0) += rrf_contribution;

                // Track original scores for debugging
                let orig = original_scores
                    .entry(memory_id.clone())
                    .or_insert_with(|| vec![None; ranked_lists.len()]);
                if list_idx < orig.len() {
                    orig[list_idx] = Some(*score);
                }
            }
        }

        // Sort by RRF score descending
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1));

        results
    }
}

/// Unified hybrid search engine
///
/// Combines BM25 + Vector + RRF fusion + cognitive post-processing
pub struct HybridSearchEngine {
    bm25_index: BM25Index,
    config: HybridSearchConfig,
}

impl HybridSearchEngine {
    /// Create hybrid search engine
    pub fn new(
        bm25_path: &Path,
        _embedder: Arc<MiniLMEmbedder>,
        config: HybridSearchConfig,
    ) -> Result<Self> {
        let bm25_index = BM25Index::new(bm25_path)?;

        Ok(Self { bm25_index, config })
    }

    /// Index a memory for BM25 search
    pub fn index_memory(
        &self,
        memory_id: &MemoryId,
        content: &str,
        tags: &[String],
        entities: &[String],
    ) -> Result<()> {
        self.bm25_index.upsert(memory_id, content, tags, entities)
    }

    /// Remove a memory from the BM25 index
    pub fn remove_memory(&self, memory_id: &MemoryId) -> Result<()> {
        self.bm25_index.delete(memory_id)
    }

    /// Commit BM25 index changes
    pub fn commit(&self) -> Result<()> {
        self.bm25_index.commit()
    }

    /// Reload BM25 reader to see committed changes immediately
    pub fn reload(&self) -> Result<()> {
        self.bm25_index.reload()
    }

    /// Commit and reload in one call for immediate searchability
    pub fn commit_and_reload(&self) -> Result<()> {
        self.bm25_index.commit()?;
        self.bm25_index.reload()
    }

    /// Merge BM25 segments to remove ghost state and reclaim space.
    /// Returns the number of segments merged (0 if already optimal).
    pub fn optimize_bm25(&self) -> Result<usize> {
        self.bm25_index.optimize()
    }

    /// Get BM25 segment count for health metrics
    pub fn bm25_segment_count(&self) -> usize {
        self.bm25_index.segment_count()
    }

    /// Get BM25 index reference for direct searches
    pub fn bm25_index(&self) -> &BM25Index {
        &self.bm25_index
    }

    /// Perform hybrid search combining BM25 and vector results
    ///
    /// # Arguments
    /// * `query` - Search query text
    /// * `vector_results` - Pre-computed vector search results (memory_id, similarity)
    /// * `get_content` - Closure to fetch content for reranking
    ///
    /// # Returns
    /// Hybrid search results with component scores
    pub fn search<F>(
        &self,
        query: &str,
        vector_results: Vec<(MemoryId, f32)>,
        get_content: F,
    ) -> Result<Vec<HybridSearchResult>>
    where
        F: Fn(&MemoryId) -> Option<String>,
    {
        self.search_with_ic_weights(query, vector_results, get_content, None)
    }

    /// Perform hybrid search with IC-weighted BM25 term boosting
    ///
    /// IC weights from linguistic analysis boost important terms:
    /// - Nouns (focal entities): IC=1.5
    /// - Adjectives (modifiers): IC=0.9
    /// - Verbs (relations): IC=0.7
    ///
    /// This improves retrieval by prioritizing semantically important query terms.
    pub fn search_with_ic_weights<F>(
        &self,
        query: &str,
        vector_results: Vec<(MemoryId, f32)>,
        get_content: F,
        term_weights: Option<&HashMap<String, f32>>,
    ) -> Result<Vec<HybridSearchResult>>
    where
        F: Fn(&MemoryId) -> Option<String>,
    {
        self.search_with_ic_weights_and_phrases(
            query,
            vector_results,
            get_content,
            term_weights,
            None,
        )
    }

    /// Perform hybrid search with IC-weighted BM25 term boosting AND phrase matching
    ///
    /// IC weights from linguistic analysis boost important terms.
    /// Phrase boosts enable exact multi-word phrase matching:
    /// - "support group" matches the exact phrase, not just "support" OR "group"
    /// - Compound nouns get 2.0x boost, adjacent nouns get 1.5x boost
    pub fn search_with_ic_weights_and_phrases<F>(
        &self,
        query: &str,
        vector_results: Vec<(MemoryId, f32)>,
        get_content: F,
        term_weights: Option<&HashMap<String, f32>>,
        phrase_boosts: Option<&[(String, f32)]>,
    ) -> Result<Vec<HybridSearchResult>>
    where
        F: Fn(&MemoryId) -> Option<String>,
    {
        // Use default discriminativeness (no dynamic weight adjustment)
        self.search_with_dynamic_weights(
            query,
            vector_results,
            get_content,
            term_weights,
            phrase_boosts,
            None,
        )
    }

    /// Perform hybrid search with dynamic BM25/vector weight adjustment
    ///
    /// When `keyword_discriminativeness` is provided and high (>0.5), BM25 weight
    /// is boosted to ensure discriminative keywords are properly matched.
    ///
    /// This solves the multi-hop retrieval problem where queries like
    /// "When did Melanie paint a sunrise?" fail because common terms ("Melanie", "paint")
    /// dominate, while the discriminative term ("sunrise") gets diluted in vector search.
    ///
    /// Dynamic weight adjustment:
    /// - discriminativeness 0.0-0.4: use default weights (BM25=0.4, Vector=0.6)
    /// - discriminativeness 0.5-0.7: boost BM25 (BM25=0.55, Vector=0.45)
    /// - discriminativeness 0.8-1.0: strong BM25 (BM25=0.7, Vector=0.3)
    pub fn search_with_dynamic_weights<F>(
        &self,
        query: &str,
        vector_results: Vec<(MemoryId, f32)>,
        _get_content: F,
        term_weights: Option<&HashMap<String, f32>>,
        phrase_boosts: Option<&[(String, f32)]>,
        keyword_discriminativeness: Option<f32>,
    ) -> Result<Vec<HybridSearchResult>>
    where
        F: Fn(&MemoryId) -> Option<String>,
    {
        // 1. BM25 search with IC-weighted term boosting AND phrase matching
        let bm25_results = self.bm25_index.search_with_term_and_phrase_weights(
            query,
            self.config.candidate_count,
            term_weights,
            phrase_boosts,
        )?;

        // Filter low BM25 scores
        let bm25_results: Vec<_> = bm25_results
            .into_iter()
            .filter(|(_, score)| *score >= self.config.min_bm25_score)
            .collect();

        // Calculate dynamic weights based on keyword discriminativeness
        // When YAKE identifies discriminative keywords, trust BM25 more
        // YAKE importance = 1/(1+score), so 0.9+ means very discriminative keywords
        let (bm25_weight, vector_weight) = if let Some(disc) = keyword_discriminativeness {
            if disc >= 0.8 {
                // Highly discriminative keywords - strong BM25 preference
                (0.75, 0.25)
            } else if disc >= 0.5 {
                // Moderately discriminative - BM25 dominant
                (0.6, 0.4)
            } else {
                // Low discriminativeness - use default weights
                (self.config.bm25_weight, self.config.vector_weight)
            }
        } else {
            (self.config.bm25_weight, self.config.vector_weight)
        };

        // Log counts and weights for debugging
        if bm25_results.is_empty() {
            tracing::warn!(
                "Hybrid search: BM25 returned 0 results for query '{}', using {} vector results only",
                query,
                vector_results.len()
            );
        } else {
            debug!(
                "Hybrid search: {} BM25 (top: {:.3}), {} vector, weights: BM25={:.2}/Vec={:.2}, disc={:?} for '{}'",
                bm25_results.len(),
                bm25_results.first().map(|(_, s)| *s).unwrap_or(0.0),
                vector_results.len(),
                bm25_weight,
                vector_weight,
                keyword_discriminativeness,
                &query[..query.len().min(50)]
            );
        }

        // 2. RRF Fusion with dynamic weights
        let rrf = RRFusion::new(self.config.rrf_k, vec![bm25_weight, vector_weight]);

        let fused = rrf.fuse(vec![bm25_results.clone(), vector_results.clone()]);

        // Build lookup maps for component scores
        let bm25_map: HashMap<MemoryId, (f32, usize)> = bm25_results
            .iter()
            .enumerate()
            .map(|(rank, (id, score))| (id.clone(), (*score, rank)))
            .collect();

        let vector_map: HashMap<MemoryId, (f32, usize)> = vector_results
            .iter()
            .enumerate()
            .map(|(rank, (id, score))| (id.clone(), (*score, rank)))
            .collect();

        // 3. Build final results from RRF-fused scores
        let final_results: Vec<HybridSearchResult> = fused
            .into_iter()
            .map(|(memory_id, rrf_score)| {
                let bm25_info = bm25_map.get(&memory_id);
                let vector_info = vector_map.get(&memory_id);

                HybridSearchResult {
                    memory_id,
                    score: rrf_score,
                    bm25_score: bm25_info.map(|(s, _)| *s),
                    vector_score: vector_info.map(|(s, _)| *s),
                    graph_score: None,
                    rrf_score,
                    bm25_rank: bm25_info.map(|(_, r)| *r),
                    vector_rank: vector_info.map(|(_, r)| *r),
                    graph_rank: None,
                }
            })
            .collect();

        Ok(final_results)
    }

    /// Get BM25 document count
    pub fn bm25_doc_count(&self) -> usize {
        self.bm25_index.len()
    }

    /// Check if BM25 index is empty (needs backfill)
    pub fn needs_backfill(&self) -> bool {
        self.bm25_index.is_empty()
    }

    /// Backfill BM25 index from existing memories
    ///
    /// Call this on startup if the BM25 index is empty but memories exist.
    /// This indexes all memories into BM25 for hybrid search.
    ///
    /// # Arguments
    /// * `memories` - Iterator of (memory_id, content, tags, entities)
    ///
    /// # Returns
    /// Number of memories indexed
    pub fn backfill<I>(&self, memories: I) -> Result<usize>
    where
        I: Iterator<Item = (MemoryId, String, Vec<String>, Vec<String>)>,
    {
        let mut count = 0;
        let mut batch_count = 0;
        const BATCH_SIZE: usize = 100;

        for (memory_id, content, tags, entities) in memories {
            self.bm25_index
                .upsert(&memory_id, &content, &tags, &entities)?;
            count += 1;
            batch_count += 1;

            // Commit in batches to avoid holding locks too long
            if batch_count >= BATCH_SIZE {
                self.bm25_index.commit()?;
                batch_count = 0;
                debug!("BM25 backfill: indexed {} memories", count);
            }
        }

        // Final commit
        if batch_count > 0 {
            self.bm25_index.commit()?;
        }

        // Reload reader to see new documents
        self.bm25_index.reload()?;

        info!("BM25 backfill complete: indexed {} memories", count);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion_basic() {
        let rrf = RRFusion::new(60.0, vec![0.5, 0.5]);

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());
        let id3 = MemoryId(uuid::Uuid::new_v4());

        // List 1: id1 > id2 > id3
        let list1 = vec![(id1.clone(), 0.9), (id2.clone(), 0.7), (id3.clone(), 0.5)];

        // List 2: id2 > id1 > id3
        let list2 = vec![(id2.clone(), 0.95), (id1.clone(), 0.6), (id3.clone(), 0.4)];

        let fused = rrf.fuse(vec![list1, list2]);

        // id1 and id2 have symmetric ranks (1,2) and (2,1), so they should have equal RRF scores
        // The ordering between them is implementation-defined, but both should be above id3
        // id3 is rank 3 in both lists, so it should be last
        assert_eq!(fused.len(), 3);

        // Both id1 and id2 should have higher scores than id3
        let id1_score = fused.iter().find(|(id, _)| *id == id1).unwrap().1;
        let id2_score = fused.iter().find(|(id, _)| *id == id2).unwrap().1;
        let id3_score = fused.iter().find(|(id, _)| *id == id3).unwrap().1;

        // id1 and id2 should have equal scores (symmetric ranking)
        assert!(
            (id1_score - id2_score).abs() < 0.0001,
            "id1 and id2 should have equal RRF scores"
        );

        // id3 should be last (lowest score)
        assert!(
            id3_score < id1_score,
            "id3 should have lower score than id1"
        );
        assert!(
            id3_score < id2_score,
            "id3 should have lower score than id2"
        );
        assert_eq!(fused[2].0, id3, "id3 should be ranked last");
    }

    #[test]
    fn test_rrf_fusion_disjoint() {
        let rrf = RRFusion::new(60.0, vec![0.5, 0.5]);

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());

        // Disjoint lists
        let list1 = vec![(id1.clone(), 0.9)];
        let list2 = vec![(id2.clone(), 0.8)];

        let fused = rrf.fuse(vec![list1, list2]);

        assert_eq!(fused.len(), 2);
        // Both should have same RRF score (rank 1 in their respective list)
        assert!((fused[0].1 - fused[1].1).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_config_defaults() {
        let config = HybridSearchConfig::default();
        assert_eq!(config.bm25_weight, 0.35); // BM25 for keyword matching
        assert_eq!(config.vector_weight, 0.40); // Vector for semantic relationships
        assert_eq!(config.graph_weight, 0.25); // Graph for associative retrieval (SHO-D4)
        assert_eq!(config.rrf_k, 45.0); // Lower k for top-rank emphasis
        assert_eq!(config.candidate_count, 100); // Increased for better recall
        assert_eq!(config.min_graph_score, 0.01); // Graph score threshold (SHO-D4)
    }

    #[test]
    fn test_bm25_index_and_search() {
        let temp_dir = tempfile::tempdir().unwrap();
        let index = BM25Index::new(temp_dir.path()).unwrap();

        // Create test memories
        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());
        let id3 = MemoryId(uuid::Uuid::new_v4());

        // Index documents with different content
        index
            .upsert(
                &id1,
                "The user prefers Rust programming language for systems development",
                &["rust".to_string(), "programming".to_string()],
                &["Rust".to_string()],
            )
            .unwrap();

        index
            .upsert(
                &id2,
                "Python is great for machine learning and data science projects",
                &["python".to_string(), "ml".to_string()],
                &["Python".to_string()],
            )
            .unwrap();

        index
            .upsert(
                &id3,
                "The authentication system uses JWT tokens for security",
                &["auth".to_string(), "security".to_string()],
                &["JWT".to_string()],
            )
            .unwrap();

        index.commit().unwrap();
        index.reload().unwrap();

        // Test: Search for "Rust" should find id1
        let results = index.search("Rust programming", 10).unwrap();
        assert!(!results.is_empty(), "Should find Rust document");
        assert_eq!(results[0].0, id1, "Rust doc should be first");

        // Test: Search for "Python" should find id2
        let results = index.search("Python machine learning", 10).unwrap();
        assert!(!results.is_empty(), "Should find Python document");
        assert_eq!(results[0].0, id2, "Python doc should be first");

        // Test: Search for "JWT" should find id3
        let results = index.search("JWT authentication", 10).unwrap();
        assert!(!results.is_empty(), "Should find auth document");
        assert_eq!(results[0].0, id3, "Auth doc should be first");

        // Test: Search for unrelated term should return empty or low scores
        let results = index.search("quantum physics", 10).unwrap();
        assert!(
            results.is_empty() || results[0].1 < 0.5,
            "Unrelated search should have low/no results"
        );
    }

    #[test]
    fn test_bm25_keyword_vs_semantic_gap() {
        // This test demonstrates why BM25 is needed alongside vector search
        let temp_dir = tempfile::tempdir().unwrap();
        let index = BM25Index::new(temp_dir.path()).unwrap();

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());

        // Document with specific technical term "SIGHUP"
        index
            .upsert(
                &id1,
                "The server reloads configuration when it receives SIGHUP signal",
                &["linux".to_string(), "signals".to_string()],
                &[],
            )
            .unwrap();

        // Document about reloading (semantically similar but different keyword)
        index
            .upsert(
                &id2,
                "Configuration refresh happens automatically every hour",
                &["config".to_string()],
                &[],
            )
            .unwrap();

        index.commit().unwrap();
        index.reload().unwrap();

        // BM25 should find exact match for "SIGHUP" even if vector search might not
        let results = index.search("SIGHUP", 10).unwrap();
        assert!(!results.is_empty(), "BM25 should find SIGHUP");
        assert_eq!(results[0].0, id1, "Exact keyword match should win");
    }

    #[test]
    fn test_rrf_weighted_fusion() {
        // Test that weights affect fusion correctly
        let rrf_bm25_heavy = RRFusion::new(60.0, vec![0.8, 0.2]); // BM25 weighted higher
        let rrf_vector_heavy = RRFusion::new(60.0, vec![0.2, 0.8]); // Vector weighted higher

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());

        // id1 ranks #1 in BM25, #2 in vector
        // id2 ranks #2 in BM25, #1 in vector
        let bm25_list = vec![(id1.clone(), 0.9), (id2.clone(), 0.7)];
        let vector_list = vec![(id2.clone(), 0.95), (id1.clone(), 0.6)];

        // With BM25 weighted higher, id1 should win
        let fused_bm25 = rrf_bm25_heavy.fuse(vec![bm25_list.clone(), vector_list.clone()]);
        assert_eq!(fused_bm25[0].0, id1, "BM25-heavy should favor BM25 winner");

        // With vector weighted higher, id2 should win
        let fused_vector = rrf_vector_heavy.fuse(vec![bm25_list, vector_list]);
        assert_eq!(
            fused_vector[0].0, id2,
            "Vector-heavy should favor vector winner"
        );
    }

    #[test]
    fn test_rrf_k_parameter_effect() {
        // Higher k = more equal weighting across ranks
        // Lower k = more emphasis on top ranks
        let rrf_low_k = RRFusion::new(1.0, vec![0.5, 0.5]); // Low k
        let rrf_high_k = RRFusion::new(100.0, vec![0.5, 0.5]); // High k

        let id1 = MemoryId(uuid::Uuid::new_v4());
        let id2 = MemoryId(uuid::Uuid::new_v4());
        let id3 = MemoryId(uuid::Uuid::new_v4());

        // id1 is #1 in list1, #3 in list2
        // id3 is #3 in list1, #1 in list2
        // id2 is #2 in both lists
        let list1 = vec![(id1.clone(), 0.9), (id2.clone(), 0.7), (id3.clone(), 0.5)];
        let list2 = vec![(id3.clone(), 0.9), (id2.clone(), 0.7), (id1.clone(), 0.5)];

        let fused_low_k = rrf_low_k.fuse(vec![list1.clone(), list2.clone()]);
        let fused_high_k = rrf_high_k.fuse(vec![list1, list2]);

        // With low k, rank differences matter more
        // With high k, id2 (consistent #2) should do relatively better
        // id2's score should be relatively higher with high k
        let id2_score_low = fused_low_k.iter().find(|(id, _)| *id == id2).unwrap().1;
        let id2_score_high = fused_high_k.iter().find(|(id, _)| *id == id2).unwrap().1;

        // Normalize by max score to compare relative positions
        let max_low = fused_low_k[0].1;
        let max_high = fused_high_k[0].1;

        let id2_relative_low = id2_score_low / max_low;
        let id2_relative_high = id2_score_high / max_high;

        // id2 should have higher relative score with high k (more forgiving of rank differences)
        assert!(
            id2_relative_high >= id2_relative_low - 0.01,
            "High k should be more forgiving of rank variation"
        );
    }
}
