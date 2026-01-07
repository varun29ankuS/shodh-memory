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
use crate::embeddings::Embedder;

/// Configuration for hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Weight for BM25 scores in RRF (0.0-1.0)
    #[serde(default = "default_bm25_weight")]
    pub bm25_weight: f32,

    /// Weight for vector scores in RRF (0.0-1.0)
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,

    /// RRF constant k (higher = more equal weighting)
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,

    /// Number of candidates to fetch from each retriever
    #[serde(default = "default_candidate_count")]
    pub candidate_count: usize,

    /// Number of top results to rerank with cross-encoder
    #[serde(default = "default_rerank_count")]
    pub rerank_count: usize,

    /// Whether to use cross-encoder reranking
    #[serde(default = "default_use_reranking")]
    pub use_reranking: bool,

    /// Minimum BM25 score to consider (filters noise)
    #[serde(default = "default_min_bm25_score")]
    pub min_bm25_score: f32,
}

fn default_bm25_weight() -> f32 {
    0.7 // Prioritize keyword matching for better recall accuracy
}
fn default_vector_weight() -> f32 {
    0.3 // Semantic similarity as secondary signal
}
fn default_rrf_k() -> f32 {
    45.0 // Lower k = more emphasis on top-ranked results
}
fn default_candidate_count() -> usize {
    50 // Reduced from 100 for faster search; still sufficient for recall
}
fn default_rerank_count() -> usize {
    20
}
fn default_use_reranking() -> bool {
    false // Disabled: current implementation is bi-encoder, not true cross-encoder
}
fn default_min_bm25_score() -> f32 {
    0.01 // Lower threshold to capture more keyword matches
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            bm25_weight: default_bm25_weight(),
            vector_weight: default_vector_weight(),
            rrf_k: default_rrf_k(),
            candidate_count: default_candidate_count(),
            rerank_count: default_rerank_count(),
            use_reranking: default_use_reranking(),
            min_bm25_score: default_min_bm25_score(),
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

    /// RRF score before reranking
    pub rrf_score: f32,

    /// Cross-encoder score (if reranked)
    pub rerank_score: Option<f32>,

    /// Rank from BM25 (if matched)
    pub bm25_rank: Option<usize>,

    /// Rank from vector search (if matched)
    pub vector_rank: Option<usize>,
}

/// BM25 Index using Tantivy
pub struct BM25Index {
    index: Index,
    reader: IndexReader,
    writer: Arc<RwLock<IndexWriter>>,
    schema: Schema,
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
        let id_field = schema_builder.add_text_field("id", STRING | STORED);

        // Main content (tokenized for BM25)
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);

        // Tags (tokenized)
        let tags_field = schema_builder.add_text_field("tags", TEXT);

        // Entities (tokenized)
        let entities_field = schema_builder.add_text_field("entities", TEXT);

        let schema = schema_builder.build();

        // Create or open index
        std::fs::create_dir_all(path)?;
        let dir = tantivy::directory::MmapDirectory::open(path)
            .context("Failed to open tantivy directory")?;

        let index = if Index::exists(&dir)? {
            Index::open(dir).context("Failed to open existing BM25 index")?
        } else {
            Index::create_in_dir(path, schema.clone()).context("Failed to create BM25 index")?
        };

        // 50MB writer heap
        let writer = index
            .writer(50_000_000)
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
            schema,
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
        doc.add_text(self.id_field, &memory_id.0.to_string());
        doc.add_text(self.content_field, content);
        doc.add_text(self.tags_field, &tags.join(" "));
        doc.add_text(self.entities_field, &entities.join(" "));

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
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();

        // Parse query across content, tags, and entities fields
        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.content_field, self.tags_field, self.entities_field],
        );

        // Build boosted query if term weights provided
        let boosted_query = if let Some(weights) = term_weights {
            let mut boosted_terms: Vec<String> = Vec::new();
            for word in query.split_whitespace() {
                let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if clean_word.is_empty() {
                    continue;
                }
                if let Some(&weight) = weights.get(&clean_word) {
                    // Apply boost (Tantivy uses ^ for boost, like Lucene)
                    boosted_terms.push(format!("{}^{:.1}", clean_word, weight));
                } else {
                    boosted_terms.push(clean_word);
                }
            }
            boosted_terms.join(" ")
        } else {
            query.to_string()
        };

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
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
    }
}

/// Cross-encoder reranker using the same MiniLM model
///
/// For true cross-encoder reranking, you'd use a model trained for
/// query-document scoring (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2).
/// This implementation uses cosine similarity as a proxy, which works
/// but isn't as accurate as a dedicated cross-encoder.
///
/// Future: Replace with actual cross-encoder model for better accuracy.
pub struct CrossEncoderReranker {
    embedder: Arc<MiniLMEmbedder>,
}

impl CrossEncoderReranker {
    /// Create reranker with shared embedder
    pub fn new(embedder: Arc<MiniLMEmbedder>) -> Self {
        Self { embedder }
    }

    /// Rerank candidates based on query-document similarity
    ///
    /// Takes (memory_id, content, current_score) and returns reranked scores.
    ///
    /// Note: This uses bi-encoder (separate query/doc embeddings) not true
    /// cross-encoder (joint query+doc encoding). True cross-encoders are
    /// more accurate but slower. This is a reasonable approximation.
    pub fn rerank(
        &self,
        query: &str,
        candidates: Vec<(MemoryId, String, f32)>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Encode query
        let query_embedding = self.embedder.encode(query)?;

        let mut results: Vec<(MemoryId, f32)> = Vec::with_capacity(candidates.len());

        for (memory_id, content, _original_score) in candidates {
            // Encode document
            let doc_embedding = self.embedder.encode(&content)?;

            // Cosine similarity
            let similarity = cosine_similarity(&query_embedding, &doc_embedding);

            results.push((memory_id, similarity));
        }

        // Sort by reranked score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Unified hybrid search engine
///
/// Combines BM25 + Vector + RRF + Cross-encoder + Cognitive signals
pub struct HybridSearchEngine {
    bm25_index: BM25Index,
    config: HybridSearchConfig,
    reranker: Option<CrossEncoderReranker>,
}

impl HybridSearchEngine {
    /// Create hybrid search engine
    pub fn new(
        bm25_path: &Path,
        embedder: Arc<MiniLMEmbedder>,
        config: HybridSearchConfig,
    ) -> Result<Self> {
        let bm25_index = BM25Index::new(bm25_path)?;

        let reranker = if config.use_reranking {
            Some(CrossEncoderReranker::new(embedder))
        } else {
            None
        };

        Ok(Self {
            bm25_index,
            config,
            reranker,
        })
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
        // 1. BM25 search with IC-weighted term boosting
        let bm25_results = self.bm25_index.search_with_term_weights(
            query,
            self.config.candidate_count,
            term_weights,
        )?;

        // Filter low BM25 scores
        let bm25_results: Vec<_> = bm25_results
            .into_iter()
            .filter(|(_, score)| *score >= self.config.min_bm25_score)
            .collect();

        // Log counts at info level for debugging search quality
        if bm25_results.is_empty() {
            tracing::warn!(
                "Hybrid search: BM25 returned 0 results for query '{}', using {} vector results only",
                query,
                vector_results.len()
            );
        } else {
            debug!(
                "Hybrid search: {} BM25 results (top score: {:.3}), {} vector results for '{}'",
                bm25_results.len(),
                bm25_results.first().map(|(_, s)| *s).unwrap_or(0.0),
                vector_results.len(),
                &query[..query.len().min(50)]
            );
        }

        // 2. RRF Fusion
        let rrf = RRFusion::new(
            self.config.rrf_k,
            vec![self.config.bm25_weight, self.config.vector_weight],
        );

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

        // 3. Optional cross-encoder reranking
        let final_results = if let Some(ref reranker) = self.reranker {
            // Take top-k for reranking
            let to_rerank: Vec<_> = fused
                .iter()
                .take(self.config.rerank_count)
                .filter_map(|(id, _score)| {
                    get_content(id).map(|content| (id.clone(), content, *_score))
                })
                .collect();

            if !to_rerank.is_empty() {
                let reranked = reranker.rerank(query, to_rerank)?;

                // Build rerank map
                let rerank_map: HashMap<MemoryId, f32> = reranked.into_iter().collect();

                // Combine reranked results with non-reranked
                let mut results: Vec<HybridSearchResult> = Vec::new();

                for (memory_id, rrf_score) in fused {
                    let bm25_info = bm25_map.get(&memory_id);
                    let vector_info = vector_map.get(&memory_id);
                    let rerank_score = rerank_map.get(&memory_id).copied();

                    // Final score: if reranked, use rerank score, else use RRF
                    let final_score = rerank_score.unwrap_or(rrf_score);

                    results.push(HybridSearchResult {
                        memory_id,
                        score: final_score,
                        bm25_score: bm25_info.map(|(s, _)| *s),
                        vector_score: vector_info.map(|(s, _)| *s),
                        rrf_score,
                        rerank_score,
                        bm25_rank: bm25_info.map(|(_, r)| *r),
                        vector_rank: vector_info.map(|(_, r)| *r),
                    });
                }

                // Re-sort by final score
                results.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                results
            } else {
                // No content available for reranking, use RRF scores
                fused
                    .into_iter()
                    .map(|(memory_id, rrf_score)| {
                        let bm25_info = bm25_map.get(&memory_id);
                        let vector_info = vector_map.get(&memory_id);

                        HybridSearchResult {
                            memory_id,
                            score: rrf_score,
                            bm25_score: bm25_info.map(|(s, _)| *s),
                            vector_score: vector_info.map(|(s, _)| *s),
                            rrf_score,
                            rerank_score: None,
                            bm25_rank: bm25_info.map(|(_, r)| *r),
                            vector_rank: vector_info.map(|(_, r)| *r),
                        }
                    })
                    .collect()
            }
        } else {
            // No reranking, use RRF scores directly
            fused
                .into_iter()
                .map(|(memory_id, rrf_score)| {
                    let bm25_info = bm25_map.get(&memory_id);
                    let vector_info = vector_map.get(&memory_id);

                    HybridSearchResult {
                        memory_id,
                        score: rrf_score,
                        bm25_score: bm25_info.map(|(s, _)| *s),
                        vector_score: vector_info.map(|(s, _)| *s),
                        rrf_score,
                        rerank_score: None,
                        bm25_rank: bm25_info.map(|(_, r)| *r),
                        vector_rank: vector_info.map(|(_, r)| *r),
                    }
                })
                .collect()
        };

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
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_config_defaults() {
        let config = HybridSearchConfig::default();
        assert_eq!(config.bm25_weight, 0.7); // BM25 prioritized for recall accuracy
        assert_eq!(config.vector_weight, 0.3); // Semantic as secondary signal
        assert_eq!(config.rrf_k, 45.0); // Lower k for top-rank emphasis
        assert_eq!(config.candidate_count, 50); // Reduced for speed
        assert_eq!(config.rerank_count, 20);
        assert!(!config.use_reranking); // Disabled: bi-encoder, not cross-encoder
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
