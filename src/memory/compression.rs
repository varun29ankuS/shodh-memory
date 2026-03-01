//! Compression pipeline for memory optimization

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use lz4;
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::types::*;
use crate::constants::{
    COMPRESSION_ACCESS_THRESHOLD, COMPRESSION_AGE_DAYS, COMPRESSION_IMPORTANCE_HIGH,
    COMPRESSION_IMPORTANCE_LOW, CONSOLIDATION_JACCARD_THRESHOLD,
    CONSOLIDATION_MAX_CANDIDATES_PER_MEMORY, CONSOLIDATION_MIN_AGE_DAYS, CONSOLIDATION_MIN_SUPPORT,
    MAX_COMPRESSION_RATIO, MAX_DECOMPRESSED_SIZE,
};

/// Compression strategy for memories
#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    None,
    Lz4,           // Fast compression
    Summarization, // Semantic compression
    Hybrid,        // Combination of methods
}

/// Compressed memory representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMemory {
    pub id: MemoryId,
    pub summary: String,
    pub keywords: Vec<String>,
    pub importance: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub compression_ratio: f32,
    pub original_size: usize,
    pub compressed_data: Vec<u8>,
    pub strategy: String,
}

/// Compression pipeline for optimizing memory storage
pub struct CompressionPipeline {
    keyword_extractor: KeywordExtractor,
}

impl Default for CompressionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionPipeline {
    pub fn new() -> Self {
        Self {
            keyword_extractor: KeywordExtractor::new(),
        }
    }

    /// Compress a memory based on its characteristics
    pub fn compress(&self, memory: &Memory) -> Result<Memory> {
        // Don't compress if already compressed or very recent
        if memory.compressed {
            return Ok(memory.clone());
        }

        let strategy = self.select_strategy(memory);

        match strategy {
            CompressionStrategy::None => Ok(memory.clone()),
            CompressionStrategy::Lz4 => self.compress_lz4(memory),
            CompressionStrategy::Summarization => self.compress_semantic(memory),
            CompressionStrategy::Hybrid => self.compress_hybrid(memory),
        }
    }

    /// Select compression strategy based on memory characteristics
    fn select_strategy(&self, memory: &Memory) -> CompressionStrategy {
        // High importance memories get lighter compression (lossless LZ4)
        if memory.importance() > COMPRESSION_IMPORTANCE_HIGH {
            return CompressionStrategy::Lz4;
        }

        // Frequently accessed memories stay uncompressed
        if memory.access_count() > COMPRESSION_ACCESS_THRESHOLD {
            return CompressionStrategy::None;
        }

        // Old, low-importance memories get aggressive compression (lossy semantic)
        let age = chrono::Utc::now() - memory.created_at;
        if age.num_days() > COMPRESSION_AGE_DAYS && memory.importance() < COMPRESSION_IMPORTANCE_LOW
        {
            return CompressionStrategy::Summarization;
        }

        // Default to hybrid approach
        CompressionStrategy::Hybrid
    }

    /// LZ4 compression - preserves all data
    fn compress_lz4(&self, memory: &Memory) -> Result<Memory> {
        let original =
            bincode::serde::encode_to_vec(&memory.experience, bincode::config::standard())?;
        let compressed = lz4::block::compress(&original, None, false)?;

        let compression_ratio = compressed.len() as f32 / original.len() as f32;

        // Create compressed version
        let mut compressed_memory = memory.clone();
        compressed_memory.compressed = true;

        // Store compressed data in metadata
        let compressed_b64 = general_purpose::STANDARD.encode(&compressed);
        compressed_memory
            .experience
            .metadata
            .insert("compressed_data".to_string(), compressed_b64);
        compressed_memory.experience.metadata.insert(
            "compression_ratio".to_string(),
            compression_ratio.to_string(),
        );
        compressed_memory
            .experience
            .metadata
            .insert("compression_strategy".to_string(), "lz4".to_string());

        Ok(compressed_memory)
    }

    /// Semantic compression - extract essence
    fn compress_semantic(&self, memory: &Memory) -> Result<Memory> {
        let mut compressed_memory = memory.clone();

        // Extract keywords
        let keywords = self.keyword_extractor.extract(&memory.experience.content);

        // Create summary (simplified - in production would use LLM)
        let summary = self.create_summary(&memory.experience.content, 50);

        // Store only summary and keywords
        compressed_memory.experience.content = summary;
        compressed_memory
            .experience
            .metadata
            .insert("keywords".to_string(), keywords.join(","));
        compressed_memory
            .experience
            .metadata
            .insert("compression_strategy".to_string(), "semantic".to_string());
        compressed_memory.compressed = true;

        Ok(compressed_memory)
    }

    /// Hybrid compression - combine strategies
    fn compress_hybrid(&self, memory: &Memory) -> Result<Memory> {
        // First apply semantic compression
        let semantic = self.compress_semantic(memory)?;

        // Then apply LZ4 on the result
        self.compress_lz4(&semantic)
    }

    /// Decompress a memory
    ///
    /// # Returns
    /// - `Ok(Memory)` - Decompressed memory with original content restored
    /// - `Err` - If decompression fails or compression is lossy (semantic)
    ///
    /// # Errors
    /// - Returns error for semantic compression (lossy - original data not recoverable)
    /// - Returns error if compressed data is missing or corrupted
    pub fn decompress(&self, memory: &Memory) -> Result<Memory> {
        if !memory.compressed {
            return Ok(memory.clone());
        }

        let strategy = memory
            .experience
            .metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        match strategy {
            "lz4" => self.decompress_lz4(memory),
            "semantic" => {
                // Semantic compression is LOSSY - original content is NOT recoverable
                // This is intentional: we extracted keywords and summary, discarded original
                // Callers must handle this error appropriately
                Err(anyhow!(
                    "Cannot decompress semantically compressed memory '{}': \
                     semantic compression is lossy. Original content was replaced with \
                     summary and keywords. Use memory.experience.content for the summary \
                     and metadata['keywords'] for extracted keywords.",
                    memory.id.0
                ))
            }
            "hybrid" => {
                // Hybrid = semantic + lz4. The lz4 layer can be decompressed,
                // but the underlying content is still the semantic summary
                let lz4_decompressed = self.decompress_lz4(memory)?;
                // Mark that this is still semantically compressed (lossy)
                Err(anyhow!(
                    "Cannot fully decompress hybrid-compressed memory '{}': \
                     underlying semantic compression is lossy. LZ4 layer decompressed, \
                     but original content is not recoverable.",
                    lz4_decompressed.id.0
                ))
            }
            unknown => Err(anyhow!(
                "Unknown compression strategy '{}' for memory '{}'. \
                 Cannot decompress.",
                unknown,
                memory.id.0
            )),
        }
    }

    /// Check if a memory's compression is lossless (can be fully decompressed)
    pub fn is_lossless(&self, memory: &Memory) -> bool {
        if !memory.compressed {
            return true;
        }
        let strategy = memory
            .experience
            .metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        strategy == "lz4"
    }

    /// Get the compression strategy used for a memory
    pub fn get_strategy<'a>(&self, memory: &'a Memory) -> Option<&'a str> {
        if !memory.compressed {
            return None;
        }
        memory
            .experience
            .metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
    }

    /// Decompress LZ4 compressed memory
    fn decompress_lz4(&self, memory: &Memory) -> Result<Memory> {
        if let Some(compressed_b64) = memory.experience.metadata.get("compressed_data") {
            let compressed = general_purpose::STANDARD.decode(compressed_b64)?;

            // Zip bomb protection: Check compression ratio before decompressing
            // A small payload claiming to decompress to MAX_DECOMPRESSED_SIZE is suspicious
            let compressed_size = compressed.len();
            let max_expected_decompressed = compressed_size.saturating_mul(MAX_COMPRESSION_RATIO);

            if max_expected_decompressed > MAX_DECOMPRESSED_SIZE as usize {
                // The compressed size is so small that even at MAX_COMPRESSION_RATIO
                // it would exceed our limit - this is suspicious
                return Err(anyhow!(
                    "Suspicious compression ratio: compressed size {} bytes with max ratio {} \
                     would allow {} bytes decompressed, which exceeds limit of {} bytes. \
                     This may indicate a zip bomb attack.",
                    compressed_size,
                    MAX_COMPRESSION_RATIO,
                    max_expected_decompressed,
                    MAX_DECOMPRESSED_SIZE
                ));
            }

            // Limit decompression size to prevent DoS attacks
            let decompressed = lz4::block::decompress(&compressed, Some(MAX_DECOMPRESSED_SIZE))?;

            // Post-decompression ratio check for additional safety
            let actual_ratio = if compressed_size > 0 {
                decompressed.len() / compressed_size
            } else {
                0
            };
            if actual_ratio > MAX_COMPRESSION_RATIO {
                return Err(anyhow!(
                    "Decompression ratio {} exceeds maximum allowed ratio of {}. \
                     Compressed: {} bytes, Decompressed: {} bytes. \
                     This may indicate a zip bomb attack.",
                    actual_ratio,
                    MAX_COMPRESSION_RATIO,
                    compressed_size,
                    decompressed.len()
                ));
            }

            let (experience, _): (Experience, _) =
                bincode::serde::decode_from_slice(&decompressed, bincode::config::standard())?;

            // Restore the memory
            let mut restored = memory.clone();
            restored.experience = experience;
            restored.compressed = false;
            restored.experience.metadata.remove("compressed_data");
            restored.experience.metadata.remove("compression_ratio");
            restored.experience.metadata.remove("compression_strategy");

            Ok(restored)
        } else {
            Err(anyhow!("No compressed data found"))
        }
    }

    /// Create a summary of content (extractive - takes first N words)
    fn create_summary(&self, content: &str, max_words: usize) -> String {
        // Simple extractive summary - take first N words
        // In production, this would use NLP/LLM
        let words: Vec<&str> = content.split_whitespace().collect();
        let summary_words = &words[..words.len().min(max_words)];
        format!("{}...", summary_words.join(" "))
    }
}

/// Keyword extraction for semantic compression
struct KeywordExtractor {
    stop_words: HashSet<String>,
}

impl KeywordExtractor {
    fn new() -> Self {
        let stop_words = Self::load_stop_words();
        Self { stop_words }
    }

    fn extract(&self, text: &str) -> Vec<String> {
        // Simple TF-IDF style extraction
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        for word in text.split_whitespace() {
            let clean_word = word
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>();

            if clean_word.len() >= 2 && !self.stop_words.contains(&clean_word) {
                *word_freq.entry(clean_word).or_insert(0) += 1;
            }
        }

        // Sort by frequency and take top keywords
        let mut keywords: Vec<(String, usize)> = word_freq.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));

        keywords
            .into_iter()
            .take(10)
            .map(|(word, _)| word)
            .collect()
    }

    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(word)
    }

    fn load_stop_words() -> HashSet<String> {
        let words = vec![
            "the",
            "is",
            "at",
            "which",
            "on",
            "and",
            "a",
            "an",
            "as",
            "are",
            "was",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "there",
            "these",
            "those",
            "this",
            "that",
            "it",
            "its",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "where",
            "when",
            "why",
            "how",
            "all",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "but",
            "or",
            "if",
            // Pronouns and possessives
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "our",
            "their",
            // Common adverbs and adjectives that aren't meaningful entities
            "also",
            "always",
            "never",
            "often",
            "sometimes",
            "usually",
            "really",
            "actually",
            "basically",
            "currently",
            "recently",
            "already",
            "still",
            "yet",
            "now",
            "here",
            "carefully",
            "quickly",
            "easily",
            "well",
            "much",
            "many",
            "new",
            "old",
            "good",
            "great",
            "best",
            "like",
            "even",
            "also",
            "get",
            "got",
            "set",
            "use",
            "using",
            "used",
            "make",
            "made",
            "being",
        ];

        words.into_iter().map(String::from).collect()
    }
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub total_compressed: usize,
    pub total_original_size: usize,
    pub total_compressed_size: usize,
    pub average_compression_ratio: f32,
    pub strategies_used: HashMap<String, usize>,
}

impl Default for CompressionStats {
    fn default() -> Self {
        Self {
            total_compressed: 0,
            total_original_size: 0,
            total_compressed_size: 0,
            average_compression_ratio: 1.0,
            strategies_used: HashMap::new(),
        }
    }
}

// ============================================================================
// SEMANTIC CONSOLIDATION - Extract durable facts from episodic memories
// ============================================================================

/// A semantic fact extracted from episodic memories
///
/// As memories age, specific episodes ("yesterday I debugged the auth module")
/// consolidate into semantic knowledge ("the auth module uses JWT tokens").
/// This mimics how human memory transitions from episodic to semantic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    /// Unique identifier
    pub id: String,
    /// The factual statement
    pub fact: String,
    /// Confidence in this fact (0.0 - 1.0)
    pub confidence: f32,
    /// How many episodic memories support this fact
    pub support_count: usize,
    /// Source memory IDs that contributed to this fact
    pub source_memories: Vec<MemoryId>,
    /// Keywords/entities this fact relates to
    pub related_entities: Vec<String>,
    /// When this fact was first extracted
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// When this fact was last reinforced
    pub last_reinforced: chrono::DateTime<chrono::Utc>,
    /// Category of fact (preference, capability, relationship, procedure)
    pub fact_type: FactType,
}

/// Types of semantic facts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FactType {
    /// User preference: "prefers concise code"
    Preference,
    /// System capability: "can handle 10k requests/sec"
    Capability,
    /// Relationship: "auth module depends on JWT library"
    Relationship,
    /// Procedure: "to deploy, run cargo build --release"
    Procedure,
    /// Definition: "MemoryId is a UUID wrapper"
    Definition,
    /// Pattern: "errors often occur after deployment"
    Pattern,
}

impl Default for FactType {
    fn default() -> Self {
        Self::Pattern
    }
}

/// Result of consolidation operation
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Number of memories processed
    pub memories_processed: usize,
    /// Number of new facts extracted
    pub facts_extracted: usize,
    /// Number of existing facts reinforced
    pub facts_reinforced: usize,
    /// IDs of newly created facts
    pub new_fact_ids: Vec<String>,
    /// Newly extracted semantic facts (ready for storage)
    pub new_facts: Vec<SemanticFact>,
}

/// Semantic consolidation engine
///
/// Extracts durable semantic facts from episodic memories using:
/// - Multi-extractor pipeline: all extractors run on every memory
/// - Stemmed-token Jaccard clustering: groups semantically similar patterns
/// - Sentence-level extraction: facts are real content, not synthetic strings
pub struct SemanticConsolidator {
    keyword_extractor: KeywordExtractor,
    /// Minimum times a pattern must appear to become a fact
    min_support: usize,
    /// Minimum age in days before consolidation
    min_age_days: i64,
    stemmer: Stemmer,
}

/// A cluster of semantically similar pattern candidates
struct PatternCluster {
    /// Stemmed token set representing this cluster (union of all members)
    stem_set: HashSet<String>,
    /// All candidate entries: (pattern_text, memory_id, confidence)
    members: Vec<(String, MemoryId, f32)>,
}

impl Default for SemanticConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticConsolidator {
    pub fn new() -> Self {
        Self {
            keyword_extractor: KeywordExtractor::new(),
            min_support: CONSOLIDATION_MIN_SUPPORT,
            min_age_days: CONSOLIDATION_MIN_AGE_DAYS,
            stemmer: Stemmer::create(Algorithm::English),
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(min_support: usize, min_age_days: i64) -> Self {
        Self {
            keyword_extractor: KeywordExtractor::new(),
            min_support,
            min_age_days,
            stemmer: Stemmer::create(Algorithm::English),
        }
    }

    /// Extract semantic facts from a set of memories
    ///
    /// Pipeline:
    /// 1. Filter memories by age threshold
    /// 2. Run multi-extractor on each eligible memory
    /// 3. Cluster candidates by stemmed-token Jaccard similarity
    /// 4. Convert qualifying clusters (>= min_support) into facts
    pub fn consolidate(&self, memories: &[Memory]) -> ConsolidationResult {
        let mut result = ConsolidationResult {
            memories_processed: memories.len(),
            ..Default::default()
        };

        if memories.is_empty() {
            return result;
        }

        let now = chrono::Utc::now();
        let eligible: Vec<&Memory> = memories
            .iter()
            .filter(|m| (now - m.created_at).num_days() >= self.min_age_days)
            .collect();

        if eligible.is_empty() {
            return result;
        }

        // Phase 1: Extract candidates using multi-extractor pipeline
        let mut all_candidates: Vec<(String, MemoryId, f32)> = Vec::new();
        for memory in &eligible {
            let extracted = self.extract_fact_candidates(memory);
            for (pattern, confidence) in extracted {
                all_candidates.push((pattern, memory.id.clone(), confidence));
            }
        }

        if all_candidates.is_empty() {
            return result;
        }

        // Phase 2: Group by stemmed-token Jaccard similarity
        let clusters =
            self.group_candidates_by_similarity(&all_candidates, CONSOLIDATION_JACCARD_THRESHOLD);

        // Phase 3: Convert qualifying clusters to facts
        for cluster in clusters {
            if cluster.members.len() >= self.min_support {
                let representative = Self::select_representative(&cluster.members);
                let avg_confidence = cluster.members.iter().map(|(_, _, c)| c).sum::<f32>()
                    / cluster.members.len() as f32;

                let source_ids: Vec<MemoryId> = cluster
                    .members
                    .iter()
                    .map(|(_, id, _)| id.clone())
                    .collect();
                let entities = self.keyword_extractor.extract(representative);
                let fact_type = self.classify_fact(representative);

                let fact = SemanticFact {
                    id: uuid::Uuid::new_v4().to_string(),
                    fact: representative.to_string(),
                    confidence: avg_confidence.min(1.0),
                    support_count: cluster.members.len(),
                    source_memories: source_ids,
                    related_entities: entities,
                    created_at: now,
                    last_reinforced: now,
                    fact_type,
                };

                result.new_fact_ids.push(fact.id.clone());
                result.new_facts.push(fact);
                result.facts_extracted += 1;
            }
        }

        result
    }

    // ── Clustering ──────────────────────────────────────────────────────────

    /// Tokenize text into stemmed tokens, removing stop words and punctuation
    fn stemmed_tokens(&self, text: &str) -> HashSet<String> {
        text.split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|w| w.len() >= 2 && !self.keyword_extractor.is_stop_word(w))
            .map(|w| self.stemmer.stem(&w).to_string())
            .collect()
    }

    /// Jaccard similarity between two token sets: |A ∩ B| / |A ∪ B|
    fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count();
        let union = a.union(b).count();
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Group candidates into clusters using greedy single-pass Jaccard matching
    fn group_candidates_by_similarity(
        &self,
        candidates: &[(String, MemoryId, f32)],
        threshold: f32,
    ) -> Vec<PatternCluster> {
        let mut clusters: Vec<PatternCluster> = Vec::new();

        for (pattern, memory_id, confidence) in candidates {
            let tokens = self.stemmed_tokens(pattern);
            if tokens.is_empty() {
                continue;
            }

            // Find best matching cluster
            let mut best_idx = None;
            let mut best_sim = 0.0f32;
            for (i, cluster) in clusters.iter().enumerate() {
                let sim = Self::jaccard_similarity(&tokens, &cluster.stem_set);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(i);
                }
            }

            if best_sim >= threshold {
                // Merge into existing cluster — expand the stem set
                let idx = best_idx.unwrap();
                clusters[idx].stem_set = clusters[idx].stem_set.union(&tokens).cloned().collect();
                clusters[idx]
                    .members
                    .push((pattern.clone(), memory_id.clone(), *confidence));
            } else {
                // Create new cluster
                clusters.push(PatternCluster {
                    stem_set: tokens,
                    members: vec![(pattern.clone(), memory_id.clone(), *confidence)],
                });
            }
        }

        clusters
    }

    /// Select the representative fact text from a cluster (highest confidence)
    fn select_representative(members: &[(String, MemoryId, f32)]) -> &str {
        members
            .iter()
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(text, _, _)| text.as_str())
            .unwrap_or("")
    }

    // ── Multi-Extractor Pipeline ────────────────────────────────────────────

    /// Extract fact candidates from a single memory using all applicable extractors
    fn extract_fact_candidates(&self, memory: &Memory) -> Vec<(String, f32)> {
        let mut candidates = Vec::new();
        let content = &memory.experience.content;
        let importance = memory.importance();
        let exp_type = &memory.experience.experience_type;

        // Run all extractors with type-based confidence multipliers
        if let Some(fact) = self.extract_procedure(content) {
            let mult = if *exp_type == ExperienceType::Decision {
                1.0
            } else {
                0.7
            };
            candidates.push((fact, importance * mult));
        }

        if let Some(fact) = self.extract_definition(content) {
            let mult = if *exp_type == ExperienceType::Learning
                || *exp_type == ExperienceType::Discovery
            {
                1.2
            } else {
                0.8
            };
            candidates.push((fact, importance * mult));
        }

        if let Some(fact) = self.extract_pattern(content) {
            let mult = if *exp_type == ExperienceType::Error {
                1.1
            } else {
                0.7
            };
            candidates.push((fact, importance * mult));
        }

        if let Some(fact) = self.extract_preference(content) {
            let mult = if *exp_type == ExperienceType::Conversation {
                1.0
            } else {
                0.6
            };
            candidates.push((fact, importance * mult));
        }

        // Universal fallback: extract most salient sentence
        if let Some(fact) = self.extract_salient_statement(content, &memory.experience.entities) {
            candidates.push((fact, importance * 0.6));
        }

        // Deduplicate overlapping extractions from the same memory
        candidates = self.dedup_within_memory(candidates);

        // Cap per-memory candidates
        candidates.truncate(CONSOLIDATION_MAX_CANDIDATES_PER_MEMORY);

        // Entity pair relationships (sorted for determinism)
        // Filter: min 3 chars, no stop words, remove substring-redundant entities
        if memory.experience.entities.len() >= 2 {
            let mut sorted_entities: Vec<String> = memory
                .experience
                .entities
                .iter()
                .map(|e| e.to_lowercase())
                .filter(|e| e.len() >= 3 && !self.keyword_extractor.is_stop_word(e))
                .collect();
            sorted_entities.sort();
            sorted_entities.dedup();

            // Remove entities that are substrings of another entity in the set
            // e.g. "chose" is redundant when "chose rust" exists
            let before_dedup = sorted_entities.clone();
            sorted_entities.retain(|entity| {
                !before_dedup
                    .iter()
                    .any(|other| other != entity && other.contains(entity.as_str()))
            });

            let max_entities = sorted_entities.len().min(4);
            for i in 0..max_entities {
                for j in (i + 1)..max_entities {
                    let relationship =
                        format!("{} relates to {}", sorted_entities[i], sorted_entities[j]);
                    candidates.push((relationship, importance * 0.8));
                }
            }
        }

        candidates
    }

    /// Remove overlapping extractions from the same memory (keep highest confidence)
    fn dedup_within_memory(&self, mut candidates: Vec<(String, f32)>) -> Vec<(String, f32)> {
        if candidates.len() <= 1 {
            return candidates;
        }

        // Sort by confidence descending so we keep higher-confidence versions
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut kept: Vec<(String, f32, HashSet<String>)> = Vec::new();
        for (text, conf) in candidates {
            let tokens = self.stemmed_tokens(&text);
            let overlaps = kept
                .iter()
                .any(|(_, _, t)| Self::jaccard_similarity(&tokens, t) > 0.8);
            if !overlaps {
                kept.push((text, conf, tokens));
            }
        }

        kept.into_iter()
            .map(|(text, conf, _)| (text, conf))
            .collect()
    }

    // ── Individual Extractors ───────────────────────────────────────────────

    /// Extract a procedure from content (looks for action words)
    fn extract_procedure(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        // to_lowercase() can change byte lengths for non-ASCII chars (e.g. İ→i̇),
        // making byte offsets from `lower.find()` invalid for indexing into `content`.
        let lower = content.to_ascii_lowercase();
        let action_markers = [
            "to ",
            "run ",
            "execute ",
            "use ",
            "call ",
            "invoke ",
            "create ",
            "build ",
            "deploy ",
            "install ",
            "configure ",
            "start ",
            "stop ",
            "restart ",
            "set up ",
            "update ",
            "remove ",
            "delete ",
            "add ",
            "import ",
            "export ",
            "migrate ",
        ];

        for marker in action_markers {
            if let Some(pos) = lower.find(marker) {
                if let Some(sentence) = Self::extract_sentence(content, pos) {
                    return Some(sentence);
                }
            }
        }
        None
    }

    /// Extract a definition from content
    fn extract_definition(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        let lower = content.to_ascii_lowercase();
        let def_markers = [
            " is ",
            " are ",
            " means ",
            " refers to ",
            " represents ",
            " denotes ",
            " describes ",
            " consists of ",
            " defined as ",
            " known as ",
            " stands for ",
            " equivalent to ",
        ];

        for marker in def_markers {
            if let Some(pos) = lower.find(marker) {
                // Extract subject and definition
                let subject_start =
                    content[..pos].rfind(|c: char| !c.is_alphanumeric() && c != '_');
                let subject_start = subject_start.map(|i| i + 1).unwrap_or(0);
                let subject = &content[subject_start..pos];

                if subject.len() >= 2 {
                    let def_end = content[pos + marker.len()..]
                        .find(|c| c == '.' || c == '!' || c == '?' || c == ',');
                    let def_end = def_end
                        .map(|i| pos + marker.len() + i)
                        .unwrap_or(content.len().min(pos + marker.len() + 100));

                    let definition = &content[pos + marker.len()..def_end];
                    if definition.len() > 5 {
                        return Some(format!("{}{}{}", subject, marker, definition.trim()));
                    }
                }
            }
        }
        None
    }

    /// Extract a pattern from error content (returns the actual sentence)
    fn extract_pattern(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        let lower = content.to_ascii_lowercase();
        let pattern_markers = [
            "error",
            "failed",
            "crash",
            "bug",
            "issue",
            "problem",
            "exception",
            "warning",
            "panic",
            "timeout",
            "overflow",
            "deadlock",
            "leak",
            "corrupt",
        ];

        for marker in pattern_markers {
            if let Some(pos) = lower.find(marker) {
                if let Some(sentence) = Self::extract_sentence(content, pos) {
                    return Some(sentence);
                }
            }
        }
        None
    }

    /// Extract a preference from conversation content (returns the actual sentence)
    fn extract_preference(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        let lower = content.to_ascii_lowercase();
        let pref_markers = [
            "prefer",
            "like",
            "want",
            "better",
            "should",
            "always",
            "never",
            "dislike",
            "avoid",
            "recommend",
            "favorite",
            "rather",
            "instead of",
            "opt for",
        ];

        for marker in pref_markers {
            if let Some(pos) = lower.find(marker) {
                if let Some(sentence) = Self::extract_sentence(content, pos) {
                    return Some(sentence);
                }
            }
        }
        None
    }

    /// Extract the most information-dense sentence from content
    fn extract_salient_statement(&self, content: &str, entities: &[String]) -> Option<String> {
        let sentences = Self::split_sentences(content);
        let entity_lower: Vec<String> = entities.iter().map(|e| e.to_lowercase()).collect();

        let mut best: Option<(String, f32)> = None;

        for sentence in sentences {
            let trimmed = sentence.trim();
            if trimmed.len() < 20 || trimmed.len() > 200 {
                continue;
            }

            let lower = trimmed.to_lowercase();

            // Score: count of non-stop-words (content density)
            let content_words: usize = lower
                .split_whitespace()
                .map(|w| {
                    w.chars()
                        .filter(|c| c.is_alphanumeric())
                        .collect::<String>()
                })
                .filter(|w| !w.is_empty() && !self.keyword_extractor.is_stop_word(w))
                .count();

            // Require at least 3 content words for a meaningful statement
            if content_words < 3 {
                continue;
            }

            // Bonus for entity mentions
            let entity_bonus: f32 = entity_lower
                .iter()
                .filter(|e| lower.contains(e.as_str()))
                .count() as f32
                * 2.0;

            let score = content_words as f32 + entity_bonus;

            if best.as_ref().map_or(true, |(_, s)| score > *s) {
                best = Some((trimmed.to_string(), score));
            }
        }

        best.map(|(s, _)| s)
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Extract the sentence containing a character position
    fn extract_sentence(content: &str, pos: usize) -> Option<String> {
        let start = content[..pos].rfind(|c| c == '.' || c == '!' || c == '?');
        let start = start.map(|i| i + 1).unwrap_or(0);

        let end = content[pos..].find(|c| c == '.' || c == '!' || c == '?');
        let end = end.map(|i| pos + i).unwrap_or(content.len());

        let sentence = content[start..end].trim();
        if sentence.len() >= 20 && sentence.len() < 200 {
            Some(sentence.to_string())
        } else {
            None
        }
    }

    /// Split content into sentences by sentence-ending punctuation
    fn split_sentences(content: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in content.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let sentence = &content[start..i];
                if !sentence.trim().is_empty() {
                    sentences.push(sentence.trim());
                }
                start = i + c.len_utf8();
            }
        }

        // Trailing content without sentence-ending punctuation
        let remaining = content[start..].trim();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }

        sentences
    }

    /// Classify what type of fact this is
    fn classify_fact(&self, pattern: &str) -> FactType {
        let lower = pattern.to_lowercase();

        if lower.contains("prefer")
            || lower.contains("like")
            || lower.contains("always")
            || lower.contains("never")
            || lower.contains("favorite")
        {
            FactType::Preference
        } else if lower.contains("can ") || lower.contains("able to") || lower.contains("supports")
        {
            FactType::Capability
        } else if lower.contains("relates to")
            || lower.contains("depends on")
            || lower.contains("connects")
        {
            FactType::Relationship
        } else if lower.contains("to ")
            || lower.contains("run ")
            || lower.contains("execute")
            || lower.contains("deploy")
        {
            FactType::Procedure
        } else if lower.contains(" is ") || lower.contains(" are ") || lower.contains("means") {
            FactType::Definition
        } else {
            FactType::Pattern
        }
    }

    /// Reinforce an existing fact with new evidence
    ///
    /// Called when a memory matches an existing fact, strengthening confidence.
    pub fn reinforce_fact(&self, fact: &mut SemanticFact, memory: &Memory) {
        fact.support_count += 1;
        fact.last_reinforced = chrono::Utc::now();

        // Increase confidence with diminishing returns
        let boost = 0.1 * (1.0 - fact.confidence);
        fact.confidence = (fact.confidence + boost).min(1.0);

        // Add source if not already present
        if !fact.source_memories.contains(&memory.id) {
            fact.source_memories.push(memory.id.clone());
        }

        // Add any new entities (filter noise: short tokens, stop words)
        for entity in &memory.experience.entities {
            let lower = entity.to_lowercase();
            if lower.len() >= 3
                && !self.keyword_extractor.is_stop_word(&lower)
                && !fact.related_entities.contains(entity)
            {
                fact.related_entities.push(entity.clone());
            }
        }
    }

    /// Check if a fact would decay below deletion threshold (0.1 confidence)
    ///
    /// Uses exponential half-life model matching `decay_facts_for_all_users()`:
    /// 90-day grace period, then half-life = 180 + (30 × support_count) days.
    pub fn should_decay_fact(&self, fact: &SemanticFact) -> bool {
        use crate::constants::{
            FACT_DECAY_GRACE_DAYS, FACT_DECAY_HALF_LIFE_BASE_DAYS,
            FACT_DECAY_HALF_LIFE_PER_SUPPORT_DAYS,
        };
        let days_since = (chrono::Utc::now() - fact.last_reinforced).num_days();
        if days_since <= FACT_DECAY_GRACE_DAYS {
            return false;
        }
        let elapsed = (days_since - FACT_DECAY_GRACE_DAYS) as f64;
        let half_life = FACT_DECAY_HALF_LIFE_BASE_DAYS
            + (fact.support_count as f64 * FACT_DECAY_HALF_LIFE_PER_SUPPORT_DAYS);
        let projected = fact.confidence * (0.5_f64).powf(elapsed / half_life) as f32;
        projected < 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn create_test_memory(content: &str, importance: f32) -> Memory {
        let experience = Experience {
            content: content.to_string(),
            experience_type: ExperienceType::Observation,
            entities: vec!["test".to_string()],
            ..Default::default()
        };

        let created_at = Some(chrono::Utc::now() - chrono::Duration::days(60));

        Memory::new(
            MemoryId(Uuid::new_v4()),
            experience,
            importance,
            None, // agent_id
            None, // run_id
            None, // actor_id
            created_at,
        )
    }

    #[test]
    fn test_compression_pipeline_default() {
        let pipeline = CompressionPipeline::default();
        assert!(pipeline.keyword_extractor.stop_words.contains("the"));
    }

    #[test]
    fn test_lz4_compress_decompress() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory("This is a test memory content for compression", 0.9);

        let compressed = pipeline.compress(&memory).unwrap();
        assert!(compressed.compressed);
        assert_eq!(
            compressed
                .experience
                .metadata
                .get("compression_strategy")
                .unwrap(),
            "lz4"
        );

        let decompressed = pipeline.decompress(&compressed).unwrap();
        assert!(!decompressed.compressed);
        assert_eq!(decompressed.experience.content, memory.experience.content);
    }

    #[test]
    fn test_already_compressed_memory() {
        let pipeline = CompressionPipeline::new();
        let mut memory = create_test_memory("Test content", 0.9);
        memory.compressed = true;

        let result = pipeline.compress(&memory).unwrap();
        assert!(result.compressed);
    }

    #[test]
    fn test_semantic_compression_lossy() {
        let pipeline = CompressionPipeline::new();
        let mut memory = create_test_memory(
            "This is a long test memory with many words for semantic compression testing purposes",
            0.1,
        );
        memory.created_at = chrono::Utc::now() - chrono::Duration::days(100);

        let compressed = pipeline.compress_semantic(&memory).unwrap();
        assert!(compressed.compressed);
        assert!(compressed.experience.metadata.contains_key("keywords"));

        let result = pipeline.decompress(&compressed);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("lossy"));
    }

    #[test]
    fn test_is_lossless() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory("Test content", 0.9);

        assert!(pipeline.is_lossless(&memory));

        let compressed_lz4 = pipeline.compress_lz4(&memory).unwrap();
        assert!(pipeline.is_lossless(&compressed_lz4));

        let compressed_semantic = pipeline.compress_semantic(&memory).unwrap();
        assert!(!pipeline.is_lossless(&compressed_semantic));
    }

    #[test]
    fn test_get_strategy() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory("Test content", 0.9);

        assert!(pipeline.get_strategy(&memory).is_none());

        let compressed = pipeline.compress_lz4(&memory).unwrap();
        assert_eq!(pipeline.get_strategy(&compressed), Some("lz4"));
    }

    #[test]
    fn test_keyword_extraction() {
        let extractor = KeywordExtractor::new();
        let text = "Rust programming language memory management ownership borrowing";
        let keywords = extractor.extract(text);

        assert!(!keywords.is_empty());
        assert!(keywords.contains(&"rust".to_string()));
        assert!(keywords.contains(&"memory".to_string()));
        assert!(!keywords.contains(&"the".to_string()));
    }

    #[test]
    fn test_stop_words_filtered() {
        let extractor = KeywordExtractor::new();
        let text = "the is at which on and a an as are was were";
        let keywords = extractor.extract(text);

        assert!(keywords.is_empty());
    }

    #[test]
    fn test_semantic_consolidator_empty() {
        let consolidator = SemanticConsolidator::new();
        let result = consolidator.consolidate(&[]);

        assert_eq!(result.memories_processed, 0);
        assert_eq!(result.facts_extracted, 0);
    }

    #[test]
    fn test_semantic_consolidator_with_thresholds() {
        let consolidator = SemanticConsolidator::with_thresholds(2, 7);
        assert_eq!(consolidator.min_support, 2);
        assert_eq!(consolidator.min_age_days, 7);
    }

    #[test]
    fn test_fact_type_classification() {
        let consolidator = SemanticConsolidator::new();

        assert_eq!(
            consolidator.classify_fact("preference: concise code"),
            FactType::Preference
        );
        assert_eq!(
            consolidator.classify_fact("system can handle 10k requests"),
            FactType::Capability
        );
        assert_eq!(
            consolidator.classify_fact("auth relates to jwt"),
            FactType::Relationship
        );
        assert_eq!(
            consolidator.classify_fact("to deploy, run cargo build"),
            FactType::Procedure
        );
        assert_eq!(
            consolidator.classify_fact("MemoryId is a UUID wrapper"),
            FactType::Definition
        );
    }

    #[test]
    fn test_reinforce_fact() {
        let consolidator = SemanticConsolidator::new();
        let mut fact = SemanticFact {
            id: "test-fact".to_string(),
            fact: "test fact content".to_string(),
            confidence: 0.5,
            support_count: 1,
            source_memories: vec![],
            related_entities: vec![],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now() - chrono::Duration::days(10),
            fact_type: FactType::Pattern,
        };
        let memory = create_test_memory("reinforcing memory", 0.7);

        let old_confidence = fact.confidence;
        consolidator.reinforce_fact(&mut fact, &memory);

        assert!(fact.confidence > old_confidence);
        assert_eq!(fact.support_count, 2);
        assert!(fact.source_memories.contains(&memory.id));
    }

    #[test]
    fn test_fact_decay_threshold() {
        let consolidator = SemanticConsolidator::new();

        let recent_fact = SemanticFact {
            id: "recent".to_string(),
            fact: "recent fact".to_string(),
            confidence: 0.8,
            support_count: 5,
            source_memories: vec![],
            related_entities: vec![],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now(),
            fact_type: FactType::Pattern,
        };
        assert!(!consolidator.should_decay_fact(&recent_fact));

        let old_fact = SemanticFact {
            id: "old".to_string(),
            fact: "old fact".to_string(),
            confidence: 0.1,
            support_count: 1,
            source_memories: vec![],
            related_entities: vec![],
            created_at: chrono::Utc::now() - chrono::Duration::days(365),
            last_reinforced: chrono::Utc::now() - chrono::Duration::days(100),
            fact_type: FactType::Pattern,
        };
        assert!(consolidator.should_decay_fact(&old_fact));
    }

    #[test]
    fn test_compression_stats_default() {
        let stats = CompressionStats::default();

        assert_eq!(stats.total_compressed, 0);
        assert_eq!(stats.average_compression_ratio, 1.0);
        assert!(stats.strategies_used.is_empty());
    }

    #[test]
    fn test_create_summary() {
        let pipeline = CompressionPipeline::new();
        let content = "This is a long piece of content that should be summarized into fewer words";
        let summary = pipeline.create_summary(content, 5);

        assert!(summary.ends_with("..."));
        assert!(summary.len() < content.len());
    }

    #[test]
    fn test_consolidation_result_default() {
        let result = ConsolidationResult::default();

        assert_eq!(result.memories_processed, 0);
        assert_eq!(result.facts_extracted, 0);
        assert!(result.new_facts.is_empty());
    }

    #[test]
    fn test_fact_type_default() {
        let fact_type = FactType::default();
        assert_eq!(fact_type, FactType::Pattern);
    }

    fn create_typed_memory(
        content: &str,
        importance: f32,
        exp_type: ExperienceType,
        entities: Vec<String>,
    ) -> Memory {
        let experience = Experience {
            content: content.to_string(),
            experience_type: exp_type,
            entities,
            ..Default::default()
        };

        let created_at = Some(chrono::Utc::now() - chrono::Duration::days(60));

        Memory::new(
            MemoryId(Uuid::new_v4()),
            experience,
            importance,
            None,
            None,
            None,
            created_at,
        )
    }

    #[test]
    fn test_jaccard_similarity_helper() {
        let a: HashSet<String> = ["rust", "memory", "safety"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let b: HashSet<String> = ["rust", "memory", "performance"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let c: HashSet<String> = ["python", "web", "django"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let ab = SemanticConsolidator::jaccard_similarity(&a, &b);
        assert!(ab > 0.4, "rust+memory overlap should give ~0.5, got {ab}");
        assert!(ab < 0.6);

        let ac = SemanticConsolidator::jaccard_similarity(&a, &c);
        assert!(ac < 0.01, "disjoint sets should give 0.0, got {ac}");

        let aa = SemanticConsolidator::jaccard_similarity(&a, &a);
        assert!((aa - 1.0).abs() < 0.001, "identical sets should give 1.0");

        let empty: HashSet<String> = HashSet::new();
        assert_eq!(
            SemanticConsolidator::jaccard_similarity(&empty, &empty),
            0.0
        );
    }

    #[test]
    fn test_stemmed_tokens_removes_stop_words() {
        let consolidator = SemanticConsolidator::new();
        let tokens = consolidator.stemmed_tokens("The Rust programming language is very fast");

        assert!(!tokens.is_empty());
        // "the", "is", "very" should be filtered as stop words
        assert!(!tokens.contains("the"));
        assert!(!tokens.contains("is"));
        assert!(!tokens.contains("very"));
        // "rust", "programming", "language", "fast" should survive (stemmed)
        assert!(tokens.contains("rust"));
        assert!(tokens.contains("fast"));
    }

    #[test]
    fn test_similarity_grouping_clusters_similar_patterns() {
        let consolidator = SemanticConsolidator::with_thresholds(2, 0);

        let m1 = create_test_memory(
            "Rust provides memory safety and performance guarantees",
            0.8,
        );
        let m2 = create_test_memory("Rust gives memory safety with great performance", 0.7);
        let m3 = create_test_memory("Python is great for data science and machine learning", 0.6);

        let result = consolidator.consolidate(&[m1, m2, m3]);

        // The two Rust/memory/safety memories should cluster and produce a fact
        assert!(
            result.facts_extracted >= 1,
            "Similar memories about Rust should cluster into at least 1 fact, got {}",
            result.facts_extracted
        );

        // Verify the fact contains Rust-related content
        let has_rust_fact = result
            .new_facts
            .iter()
            .any(|f| f.fact.to_lowercase().contains("rust"));
        assert!(has_rust_fact, "Should have a fact about Rust");
    }

    #[test]
    fn test_multi_extractor_produces_multiple_candidates() {
        let consolidator = SemanticConsolidator::new();

        // This memory has both a definition ("is") and an action word ("use")
        let memory = create_typed_memory(
            "RocksDB is a high-performance embedded database. We should use it for storage.",
            0.8,
            ExperienceType::Decision,
            vec!["RocksDB".to_string()],
        );

        let candidates = consolidator.extract_fact_candidates(&memory);

        // Should produce multiple candidates from different extractors
        assert!(
            candidates.len() >= 2,
            "Multi-extractor should produce >=2 candidates, got {}",
            candidates.len()
        );
    }

    #[test]
    fn test_generic_fallback_produces_real_sentence() {
        let consolidator = SemanticConsolidator::new();

        let memory = create_typed_memory(
            "The deployment pipeline uses Docker containers for isolation. Each service runs independently.",
            0.7,
            ExperienceType::Observation,
            vec!["Docker".to_string()],
        );

        let candidates = consolidator.extract_fact_candidates(&memory);

        // Should NOT produce "involves: ..." synthetic patterns
        let has_involves = candidates
            .iter()
            .any(|(text, _)| text.starts_with("involves:"));
        assert!(
            !has_involves,
            "Should not produce synthetic 'involves:' patterns"
        );

        // Should have at least one real sentence
        assert!(
            !candidates.is_empty(),
            "Should extract at least one candidate"
        );
    }

    #[test]
    fn test_entity_relationships_sorted_deterministic() {
        let consolidator = SemanticConsolidator::new();

        let m1 = create_typed_memory(
            "Testing entity ordering",
            0.7,
            ExperienceType::Observation,
            vec!["JWT".to_string(), "Auth".to_string(), "Token".to_string()],
        );

        let m2 = create_typed_memory(
            "Testing entity ordering",
            0.7,
            ExperienceType::Observation,
            vec!["Token".to_string(), "Auth".to_string(), "JWT".to_string()],
        );

        let c1 = consolidator.extract_fact_candidates(&m1);
        let c2 = consolidator.extract_fact_candidates(&m2);

        // Entity relationships should be identical regardless of input order
        let relations1: Vec<&str> = c1
            .iter()
            .filter(|(t, _)| t.contains("relates to"))
            .map(|(t, _)| t.as_str())
            .collect();
        let relations2: Vec<&str> = c2
            .iter()
            .filter(|(t, _)| t.contains("relates to"))
            .map(|(t, _)| t.as_str())
            .collect();

        assert_eq!(
            relations1, relations2,
            "Entity relationships should be deterministic regardless of input order"
        );
    }
}
