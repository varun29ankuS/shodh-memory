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
    COMPRESSION_IMPORTANCE_LOW, CONSOLIDATION_CLUSTER_SIZE_CAP, CONSOLIDATION_JACCARD_THRESHOLD,
    CONSOLIDATION_MAX_CANDIDATES_PER_MEMORY, CONSOLIDATION_MIN_AGE_DAYS, CONSOLIDATION_MIN_SUPPORT,
    CONSOLIDATION_MIN_SUPPORT_LARGE, CONSOLIDATION_MIN_SUPPORT_MEDIUM,
    CONSOLIDATION_MIN_SUPPORT_SMALL, CONSOLIDATION_SALIENT_MIN_CONTENT_WORDS,
    MAX_COMPRESSION_RATIO, MAX_DECOMPRESSED_SIZE,
};
use crate::embeddings::keywords::KeywordExtractor;

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
        let original = crate::serialization::encode_raw(&memory.experience)?;
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

    /// Semantic compression - attach an additive summary WITHOUT discarding content.
    ///
    /// The full `experience.content` is preserved byte-for-byte. An extractive
    /// summary and keywords are stored as ADDITIONAL metadata (`summary`,
    /// `keywords`) to serve as a lightweight retrieval-layer view — never as a
    /// replacement for the body. Semantic compression is therefore lossless;
    /// actual byte savings come from the LZ4 layer (`compress_lz4` /
    /// `compress_hybrid`).
    ///
    /// The presence of the `summary` metadata key is also the on-disk marker
    /// that distinguishes new-format (content-preserving) records from legacy
    /// pre-fix records whose bodies were destructively truncated.
    fn compress_semantic(&self, memory: &Memory) -> Result<Memory> {
        let mut compressed_memory = memory.clone();

        // Extract keywords and an extractive summary as retrieval-layer views.
        let keywords = self
            .keyword_extractor
            .extract_texts(&memory.experience.content);
        let summary = self.create_summary(&memory.experience.content, 50);

        // Full content stays intact — the summary is additive, not a replacement.
        compressed_memory
            .experience
            .metadata
            .insert("summary".to_string(), summary);
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

    /// Hybrid compression - additive semantic summary + lossless LZ4 of the FULL body.
    ///
    /// Runs the (lossless) semantic step to attach summary/keywords, then LZ4-
    /// compresses the full experience. All byte savings come from LZ4; the
    /// original content is fully recoverable via `decompress`. The record is
    /// tagged `"hybrid"` so decompression restores the full body via the LZ4
    /// blob and callers can distinguish it from a plain LZ4 record.
    fn compress_hybrid(&self, memory: &Memory) -> Result<Memory> {
        // First attach the additive semantic view (content preserved).
        let semantic = self.compress_semantic(memory)?;

        // Then LZ4-compress the full experience.
        let mut compressed = self.compress_lz4(&semantic)?;

        // compress_lz4 tags the record "lz4"; relabel it as the true strategy.
        compressed
            .experience
            .metadata
            .insert("compression_strategy".to_string(), "hybrid".to_string());

        Ok(compressed)
    }

    /// Decompress a memory, restoring its full original content.
    ///
    /// All strategies produced by the current pipeline are lossless:
    /// - `"lz4"` / `"hybrid"` decompress the full experience from the LZ4 blob.
    /// - `"semantic"` records already hold the full body in `experience.content`
    ///   (the summary is additive metadata), so decompression is a no-op restore.
    ///
    /// # Returns
    /// - `Ok(Memory)` - Uncompressed memory with the full original content.
    /// - `Err` - If the LZ4 blob is missing/corrupt, the strategy is unknown, or
    ///   the record is a legacy pre-fix semantic record whose body was truncated
    ///   at storage time and is genuinely unrecoverable.
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
            // Both LZ4 and hybrid store the full experience in the LZ4 blob.
            "lz4" | "hybrid" => self.decompress_lz4(memory),
            "semantic" => {
                // Legacy pre-fix records truncated the body at storage time and
                // are unrecoverable. Fail honestly rather than returning a
                // truncated summary as if it were the original content.
                if Self::is_legacy_lossy(memory) {
                    return Err(anyhow!(
                        "Memory '{}' was written by the pre-fix lossy semantic compressor: \
                         its full content was truncated at storage time and cannot be \
                         recovered. Only the surviving summary/keywords remain in place.",
                        memory.id.0
                    ));
                }
                // New-format semantic records preserve the full body — restore
                // the uncompressed state in place.
                let mut restored = memory.clone();
                restored.compressed = false;
                restored.experience.metadata.remove("compression_strategy");
                Ok(restored)
            }
            unknown => Err(anyhow!(
                "Unknown compression strategy '{}' for memory '{}'. Cannot decompress.",
                unknown,
                memory.id.0
            )),
        }
    }

    /// Detect legacy pre-fix records whose content was destructively truncated.
    ///
    /// The pre-fix semantic compressor overwrote `experience.content` with a
    /// truncated summary ending in `...` and never wrote a `summary` metadata
    /// key. New-format compression always writes `summary` while preserving the
    /// full body, so a `semantic` record missing that key with a `...`-terminated
    /// body is an unrecoverable legacy record.
    fn is_legacy_lossy(memory: &Memory) -> bool {
        let strategy = memory
            .experience
            .metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
            .unwrap_or("");
        strategy == "semantic"
            && !memory.experience.metadata.contains_key("summary")
            && memory.experience.content.trim_end().ends_with("...")
    }

    /// Check if a memory's compression is lossless (can be fully restored).
    ///
    /// Everything the current pipeline produces is lossless; the only lossy
    /// case that can still exist on disk is a legacy pre-fix semantic record.
    pub fn is_lossless(&self, memory: &Memory) -> bool {
        if !memory.compressed {
            return true;
        }
        if Self::is_legacy_lossy(memory) {
            return false;
        }
        let strategy = memory
            .experience
            .metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        matches!(strategy, "lz4" | "hybrid" | "semantic")
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

            // The blob is a standalone postcard `Experience`, written at
            // compression time — so it carries whatever `NerEntityRecord`
            // layout was current then. A blob written before `fine_label`
            // existed (2026-07-12) desynchronises today's decoder by one byte
            // per NER record, exactly as an uncompressed record does, and on a
            // store where nearly every memory is compressed this is the
            // majority path rather than an edge case. Same recovery as
            // `deserialize_memory`: current layout first, older layout only on
            // failure, and the current-layout error is the one reported.
            let mut experience: Experience =
                match crate::serialization::decode_raw::<Experience>(&decompressed) {
                    Ok(experience) => experience,
                    Err(current_err) => {
                        let _generation =
                            crate::memory::types::NerWireGeneration::PreFineLabel.enter();
                        crate::serialization::decode_raw::<Experience>(&decompressed)
                            .map_err(|_| current_err)?
                    }
                };

            // `Experience::toponyms` is `#[serde(skip)]` — it rides at the tail
            // of `MemoryFlat` rather than inside the `Experience` encoding, so
            // it is NOT in this blob. The compressed memory kept it on its outer
            // experience (compress_lz4 clones the memory and only adds metadata
            // keys), so carry it across; otherwise decompression would silently
            // drop resolved places and LZ4 compression would stop being
            // lossless, contrary to `is_lossless`.
            experience.toponyms = memory.experience.toponyms.clone();

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

    // =========================================================================
    // Invalidation / contradiction (trailing fields — postcard-positional).
    //
    // Before these, a fact could never be CORRECTED, only out-waited.
    // `RelationshipEdge` has carried `invalidated_at` + `invalidate_relationship`
    // + traversal that honours it for a long time; none of that machinery
    // extended to facts. Worse, the polarity check in `find_similar` is a DEDUP
    // guard: a claim and its negation did not merge, so they coexisted as two
    // rows, unlinked, each ratcheting its own confidence and each extending its
    // own half-life. The better supported a wrong fact was, the more durable it
    // became.
    //
    // These mirror the edge pattern. `#[serde(default)]` plus
    // `FACT_DEFAULT_SUFFIX` let facts written before they existed decode.
    // =========================================================================
    /// When this fact was superseded. `Some` means it must not influence
    /// retrieval, must not accrue further support, and must not extend its
    /// half-life — but it is RETAINED, not deleted, so the correction has an
    /// auditable "what it replaced".
    #[serde(default)]
    pub invalidated_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Provenance of the invalidation: the id of the fact that superseded this
    /// one. `None` alongside `Some(invalidated_at)` means it was invalidated
    /// directly rather than by a competing fact.
    #[serde(default)]
    pub invalidated_by: Option<String>,

    /// Ids of facts this one is in direct contradiction with, in both
    /// directions — the surviving fact records what it superseded, and the
    /// superseded one records its victor. This is the LINK the polarity guard
    /// never created.
    #[serde(default)]
    pub contradicts: Vec<String>,
}

impl SemanticFact {
    /// True when this fact still counts as knowledge.
    ///
    /// The single predicate every consumer must go through — scoring, half-life
    /// decay, support accrual and narrative building. An invalidated fact stays
    /// in the store for audit but stops being evidence.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.invalidated_at.is_none()
    }

    /// Mark this fact as superseded by `winner_id` at `now`.
    ///
    /// Idempotent: re-invalidating keeps the FIRST invalidation timestamp, so
    /// the audit trail records when the fact stopped being believed, not the
    /// last time something noticed.
    pub fn invalidate(&mut self, winner_id: Option<&str>, now: chrono::DateTime<chrono::Utc>) {
        if self.invalidated_at.is_none() {
            self.invalidated_at = Some(now);
            self.invalidated_by = winner_id.map(|s| s.to_string());
        }
        if let Some(id) = winner_id {
            if !self.contradicts.iter().any(|c| c == id) {
                self.contradicts.push(id.to_string());
            }
        }
    }

    /// Record a contradiction link without invalidating (used on the winning
    /// side, which stays active but must remember what it displaced).
    pub fn link_contradiction(&mut self, other_id: &str) {
        if !self.contradicts.iter().any(|c| c == other_id) {
            self.contradicts.push(other_id.to_string());
        }
    }
}

/// Types of semantic facts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
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
    #[default]
    Pattern,
}

/// A cluster of related facts grouped by shared entities, with a template-generated
/// narrative and heuristic causal chain detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCluster {
    /// Most common entity in the cluster (used as topic heading)
    pub topic: String,
    /// All entities appearing in the cluster's facts
    pub entities: Vec<String>,
    /// Facts in this cluster, sorted by created_at
    pub facts: Vec<SemanticFact>,
    /// Template-generated narrative summary (no LLM required)
    pub narrative: String,
    /// Average confidence across facts in the cluster
    pub avg_confidence: f32,
    /// Sum of support_count across all facts
    pub total_support: usize,
    /// Detected causal relationships between facts in this cluster
    pub causal_chains: Vec<CausalFactLink>,
}

/// A heuristic-detected causal relationship between two facts.
///
/// Detected via keyword analysis on temporally ordered facts sharing entities.
/// Relations: "led_to" (default), "superseded_by" (replaced/deprecated),
/// "resolved_by" (fixed/resolved).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalFactLink {
    pub from_fact_id: String,
    pub to_fact_id: String,
    /// The source fact statement
    pub from_fact: String,
    /// The target fact statement
    pub to_fact: String,
    /// Relationship type: "led_to", "superseded_by", "resolved_by"
    pub relation: String,
}

/// Result of consolidation operation
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Number of memories processed
    pub memories_processed: usize,
    /// Number of memories that passed the age gate (`min_age_days`) and were
    /// actually run through extraction. Callers use this to detect that part
    /// of the corpus is still aging in: when it is smaller than the input,
    /// memories exist that will only become consolidatable on a LATER cycle,
    /// so extraction must run again even if nothing new is written.
    pub memories_eligible: usize,
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
    /// Negation polarity of the centroid (see `facts::detect_polarity`).
    ///
    /// A cluster means "the same claim restated" — the definition
    /// `SemanticFactStore::find_similar` enforces at dedup time, where
    /// polarity is a hard gate. Clustering must enforce the same invariant:
    /// negation is often a one-word difference ("no", "not"), so a claim and
    /// its correction usually share enough stems to clear the Jaccard
    /// threshold. Without this gate they merge into ONE cluster, a single
    /// representative is minted, and the contradiction machinery downstream
    /// never sees the losing side — the correction is silently swallowed at
    /// extraction, before arbitration could record it.
    polarity: bool,
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
        result.memories_eligible = eligible.len();

        if eligible.is_empty() {
            return result;
        }

        // Memory creation times, needed twice: to order candidates before
        // clustering and to order minted facts before ingest (see below).
        let created_by_id: HashMap<&MemoryId, chrono::DateTime<chrono::Utc>> =
            eligible.iter().map(|m| (&m.id, m.created_at)).collect();

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

        // Deterministic clustering: greedy cluster assignment depends on
        // candidate order, and the input arrives in storage iteration order —
        // random UUIDs, so two ingests of the same corpus clustered (and
        // minted) differently. Anchor on (evidence age, text): the same
        // corpus always clusters the same way, and cluster centroids freeze
        // at the EARLIEST phrasing of a claim, which is also the natural
        // anchor for "later restatements corroborate the original".
        all_candidates.sort_by(|a, b| {
            let ta = created_by_id
                .get(&a.1)
                .copied()
                .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC);
            let tb = created_by_id
                .get(&b.1)
                .copied()
                .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC);
            ta.cmp(&tb).then_with(|| a.0.cmp(&b.0))
        });

        // Phase 2: Group by stemmed-token Jaccard similarity
        let clusters =
            self.group_candidates_by_similarity(&all_candidates, CONSOLIDATION_JACCARD_THRESHOLD);

        // Phase 3: Convert qualifying clusters to facts.
        // Adaptive min_support scales with corpus size (LTP induction threshold).
        let effective_min_support = if eligible.len() <= 100 {
            CONSOLIDATION_MIN_SUPPORT_SMALL
        } else if eligible.len() <= 1000 {
            CONSOLIDATION_MIN_SUPPORT_MEDIUM
        } else {
            CONSOLIDATION_MIN_SUPPORT_LARGE
        }
        .max(self.min_support);

        for cluster in clusters {
            if cluster.members.len() >= effective_min_support {
                let representative = Self::select_representative(&cluster.members);
                let avg_confidence = cluster.members.iter().map(|(_, _, c)| c).sum::<f32>()
                    / cluster.members.len() as f32;

                let source_ids: Vec<MemoryId> = cluster
                    .members
                    .iter()
                    .map(|(_, id, _)| id.clone())
                    .collect();
                let entities = self.keyword_extractor.extract_texts(representative);
                let fact_type = self.classify_fact(representative);

                let fact = SemanticFact {
                    id: uuid::Uuid::new_v4().to_string(),
                    fact: representative.to_string(),
                    confidence: avg_confidence.min(1.0),
                    // A newly extracted fact has been seen once (this extraction).
                    // Cluster size informs confidence, not support_count.
                    // support_count increments by 1 per independent consolidation
                    // cycle that re-confirms the pattern (synaptic normalization).
                    support_count: 1,
                    source_memories: source_ids,
                    related_entities: entities,
                    created_at: now,
                    last_reinforced: now,
                    fact_type,
                    invalidated_at: None,
                    invalidated_by: None,
                    contradicts: Vec::new(),
                };

                result.new_fact_ids.push(fact.id.clone());
                result.new_facts.push(fact);
                result.facts_extracted += 1;
            }
        }

        // Order minted facts by the recency of their newest supporting memory,
        // OLDEST CLAIM FIRST. Callers ingest facts in this order, and
        // `SemanticFactStore::ingest_candidate` resolves a contradiction tie
        // in favour of the incoming candidate — so ingest order is the
        // recency signal arbitration acts on. Without this sort the order was
        // storage iteration order (random UUIDs): a batch containing both a
        // claim and its later correction — exactly what a bulk-seeded store
        // produces — settled on whichever side happened to be ingested last,
        // a coin flip per store. With it, the side supported by the newest
        // evidence is ingested last and wins, which is the documented
        // "recency is the default because a later contradiction is usually a
        // correction" policy applied to batches.
        let newest_evidence = |fact: &SemanticFact| {
            fact.source_memories
                .iter()
                .filter_map(|id| created_by_id.get(id).copied())
                .max()
                .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC)
        };
        result.new_facts.sort_by(|a, b| {
            newest_evidence(a)
                .cmp(&newest_evidence(b))
                // Text tie-break: same-instant cohorts (bulk seeds) must
                // still ingest in a repeatable order.
                .then_with(|| a.fact.cmp(&b.fact))
        });
        result.new_fact_ids = result.new_facts.iter().map(|f| f.id.clone()).collect();

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

    /// Group candidates into clusters using centroid-anchored Jaccard matching.
    ///
    /// Models dentate gyrus pattern separation: each cluster's stem set is
    /// frozen at its first member (the centroid). New candidates must match
    /// the centroid directly — no transitive expansion via stem-set union.
    /// This prevents semantic drift where "Rust compilation" absorbs
    /// "code formatting" through intermediate matches.
    ///
    /// Clusters are also size-capped at `CONSOLIDATION_CLUSTER_SIZE_CAP` to
    /// prevent giant absorb-everything clusters from forming.
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
            let polarity = crate::memory::facts::detect_polarity(&pattern.to_lowercase());

            // Find best matching cluster (that hasn't hit the size cap)
            let mut best_idx = None;
            let mut best_sim = 0.0f32;
            for (i, cluster) in clusters.iter().enumerate() {
                // Skip full clusters — they no longer accept members
                if cluster.members.len() >= CONSOLIDATION_CLUSTER_SIZE_CAP {
                    continue;
                }
                // A negated statement is never a restatement of the claim it
                // negates — see the polarity note on `PatternCluster`.
                if cluster.polarity != polarity {
                    continue;
                }
                let sim = Self::jaccard_similarity(&tokens, &cluster.stem_set);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(i);
                }
            }

            if best_sim >= threshold {
                // Merge into existing cluster — do NOT expand the stem set.
                // The centroid stays fixed as the first member's tokens.
                let idx = best_idx.unwrap();
                clusters[idx]
                    .members
                    .push((pattern.clone(), memory_id.clone(), *confidence));
            } else {
                // Create new cluster with this candidate as centroid
                clusters.push(PatternCluster {
                    stem_set: tokens,
                    polarity,
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

    /// Extract fact candidates from a single memory.
    ///
    /// Models prefrontal selective attention: operational traces (Context,
    /// Command, CodeEdit, FileAccess, Search) are execution logs rather than
    /// knowledge and produce zero candidates outright.
    ///
    /// Everything past that reject runs two layers. Four SPECIALIST extractors
    /// stay type-gated, because each looks for a distinct rhetorical shape and
    /// only certain experience types plausibly carry it: procedures in
    /// Decision/Learning/Task, definitions in Learning/Discovery/Pattern,
    /// failure patterns in Error/…, stated preferences in
    /// Decision/Conversation. One GENERAL declarative extractor then runs on
    /// every surviving type, because a plain factual sentence has no rhetorical
    /// marker to gate on and belongs to no single experience type. Without that
    /// second layer the pipeline was four keyword banks and nothing else, so
    /// ordinary declarative prose — the bulk of what users actually store —
    /// produced no candidates at all.
    ///
    /// Low-importance memories are no longer suppressed here: importance scales
    /// candidate CONFIDENCE, and `is_knowledge_worthy` (which enumerates the
    /// actual auto-ingest noise: session lifecycle, tool logs, todo chatter,
    /// protocol fragments) is what rejects noise, by naming it rather than by
    /// proxying it through a score the default experience type cannot reach.
    fn extract_fact_candidates(&self, memory: &Memory) -> Vec<(String, f32)> {
        let mut candidates = Vec::new();
        let content = &memory.experience.content;
        let importance = memory.importance();
        let exp_type = &memory.experience.experience_type;

        // Operational memory types produce no fact candidates.
        // These are execution traces, not knowledge worth encoding.
        match exp_type {
            ExperienceType::Context
            | ExperienceType::Command
            | ExperienceType::CodeEdit
            | ExperienceType::FileAccess
            | ExperienceType::Search => return candidates,
            _ => {}
        }

        // Type-gated extraction: each extractor only runs on eligible types.
        // is_fact_shaped() validates structural integrity before accepting.

        // Procedure extractor: Decision, Learning, Task
        if matches!(
            exp_type,
            ExperienceType::Decision | ExperienceType::Learning | ExperienceType::Task
        ) {
            if let Some(fact) = self.extract_procedure(content) {
                if Self::is_fact_shaped(&fact) {
                    let mult = if *exp_type == ExperienceType::Decision {
                        1.0
                    } else {
                        0.7
                    };
                    candidates.push((fact, importance * mult));
                }
            }
        }

        // Definition extractor: Learning, Discovery, Pattern
        if matches!(
            exp_type,
            ExperienceType::Learning | ExperienceType::Discovery | ExperienceType::Pattern
        ) {
            if let Some(fact) = self.extract_definition(content) {
                if Self::is_fact_shaped(&fact) {
                    let mult = if *exp_type == ExperienceType::Learning
                        || *exp_type == ExperienceType::Discovery
                    {
                        1.2
                    } else {
                        0.8
                    };
                    candidates.push((fact, importance * mult));
                }
            }
        }

        // Pattern extractor: Error, Learning, Discovery, Pattern
        if matches!(
            exp_type,
            ExperienceType::Error
                | ExperienceType::Learning
                | ExperienceType::Discovery
                | ExperienceType::Pattern
        ) {
            if let Some(fact) = self.extract_pattern(content) {
                if Self::is_fact_shaped(&fact) {
                    let mult = if *exp_type == ExperienceType::Error {
                        1.1
                    } else {
                        0.7
                    };
                    candidates.push((fact, importance * mult));
                }
            }
        }

        // Preference extractor: Decision, Conversation (importance-gated)
        if *exp_type == ExperienceType::Decision
            || (*exp_type == ExperienceType::Conversation && importance >= 0.5)
        {
            if let Some(fact) = self.extract_preference(content) {
                if Self::is_fact_shaped(&fact) {
                    let mult = if *exp_type == ExperienceType::Conversation {
                        1.0
                    } else {
                        0.6
                    };
                    candidates.push((fact, importance * mult));
                }
            }
        }

        // Declarative extractor: the general path, and the ONLY route an ordinary
        // factual sentence has. It runs on every experience type that survived the
        // operational reject above.
        //
        // It used to read `*exp_type == ExperienceType::Observation && importance
        // >= 0.5`, and `extract_salient_statement` additionally refused any
        // sentence that did not literally contain one of `experience.entities`.
        // Three filters in series on the only general route, every one of them
        // failing CLOSED, which is why ordinary corpora minted zero facts and the
        // whole downstream semantic layer — clustering, reinforcement, and the
        // contradiction/invalidation machinery — ran on an empty input:
        //
        //   * TYPE. `remember` defaults `memory_type` to `Observation`
        //     (`handlers::remember::parse_experience_type`), so the default type
        //     was the only one served. `Conversation`, `Task` and `Intention`
        //     reached ZERO extractors for plain prose, because the other four are
        //     keyword banks — copula markers, error nouns, sentence-initial
        //     imperative verbs, first-person preference markers — that a sentence
        //     like "Initial reports said four crew members were injured in the
        //     collapse" matches none of.
        //   * IMPORTANCE. `calculate_importance` scores `Observation` with the
        //     0.05 `_` catch-all type weight, so a rich 15-word observation with
        //     three entities lands near 0.23. The DEFAULT experience type could
        //     not clear a 0.5 bar — the gate was unreachable for the population it
        //     was written to serve.
        //   * ENTITIES. The entity requirement depended on NER emission, an
        //     upstream signal that is empty for whole classes of records; when it
        //     emits nothing, the semantic layer silently produces nothing.
        //
        // Both surviving signals are now WEIGHTS rather than gates: `importance`
        // still scales this candidate's confidence, and entity mentions still rank
        // which sentence is chosen. What decides whether a candidate becomes a
        // FACT is corroboration — `CONSOLIDATION_MIN_SUPPORT` requires the same
        // pattern from at least two DISTINCT memories. That is the filter the
        // architecture documents (constants.rs, the BCM sliding threshold), it is
        // strictly stronger evidence than "an optional upstream field was
        // non-empty", and it cannot be zeroed by that field going missing.
        if let Some(fact) = self.extract_salient_statement(content, &memory.experience.entities) {
            if Self::is_fact_shaped(&fact) {
                candidates.push((fact, importance * 0.6));
            }
        }

        // Filter operational noise before dedup/truncation
        candidates.retain(|(text, _)| Self::is_knowledge_worthy(text));

        // Deduplicate overlapping extractions from the same memory
        candidates = self.dedup_within_memory(candidates);

        // Cap per-memory candidates
        candidates.truncate(CONSOLIDATION_MAX_CANDIDATES_PER_MEMORY);

        // NOTE: Entity pair relationships ("X relates to Y") were removed here.
        // The knowledge graph already captures entity co-occurrence via O(n²) all-pairs
        // RelationshipEdge creation (state.rs process_experience_into_graph) with Hebbian
        // strengthening and edge tier promotion. The template-based pairs were redundant,
        // lower-fidelity, and constituted ~86% of all extracted facts as noise.

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

    /// Extract a procedure from content (looks for action words).
    ///
    /// Hardened: removed `"to "` (matches every infinitive in English).
    /// Remaining markers require the action word to appear at sentence start
    /// or after instructional punctuation (`:`, `—`, `-`).
    fn extract_procedure(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        // to_lowercase() can change byte lengths for non-ASCII chars (e.g. İ→i̇),
        // making byte offsets from `lower.find()` invalid for indexing into `content`.
        let lower = content.to_ascii_lowercase();
        let action_markers = [
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
                // Require marker at sentence start or after instructional punctuation.
                // This filters "we use Rust" but keeps "Step 1: use cargo build".
                if pos > 0 {
                    let before = &content[..pos];
                    let prev_char = before.trim_end().chars().last().unwrap_or(' ');
                    if !matches!(prev_char, '.' | ':' | '—' | '-' | '\n' | '!' | '•' | '*') {
                        continue;
                    }
                }
                if let Some(sentence) = Self::extract_sentence(content, pos) {
                    return Some(sentence);
                }
            }
        }
        None
    }

    /// Extract a definition from content.
    ///
    /// Hardened: rejects pronoun subjects ("it is", "this is"), requires
    /// subject >= 3 chars, definition >= 10 chars, and subject must start
    /// with uppercase or contain a technical separator (_/.).
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

        // Pronouns that produce vacuous definitions ("it is a...", "this is the...")
        const PRONOUN_SUBJECTS: &[&str] = &[
            "it", "this", "that", "there", "here", "which", "what", "they", "these", "those",
        ];

        for marker in def_markers {
            if let Some(pos) = lower.find(marker) {
                // Extract subject and definition
                let subject_start =
                    content[..pos].rfind(|c: char| !c.is_alphanumeric() && c != '_');
                let subject_start = subject_start.map(|i| i + 1).unwrap_or(0);
                let subject = &content[subject_start..pos];

                // Subject must be >= 3 chars (filter single-char and 2-char noise)
                if subject.len() < 3 {
                    continue;
                }

                // Reject pronoun subjects
                if PRONOUN_SUBJECTS
                    .iter()
                    .any(|p| subject.eq_ignore_ascii_case(p))
                {
                    continue;
                }

                // Subject must start with uppercase or contain technical separator
                let first_char = subject.chars().next().unwrap_or(' ');
                if !first_char.is_uppercase() && !subject.contains('_') && !subject.contains('.') {
                    continue;
                }

                let def_end = content[pos + marker.len()..].find(['.', '!', '?', ',']);
                let def_end = def_end
                    .map(|i| pos + marker.len() + i)
                    .unwrap_or(content.len().min(pos + marker.len() + 100));

                let definition = &content[pos + marker.len()..def_end];
                // Definition must be >= 10 chars for meaningful content
                if definition.trim().len() >= 10 {
                    return Some(format!("{}{}{}", subject, marker, definition.trim()));
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

    /// Extract a preference from conversation content (returns the actual sentence).
    ///
    /// Hardened: requires first-person subject near the marker ("I prefer",
    /// "we want") or imperative markers ("always", "never"). Rejects simile
    /// usage of "like" ("looks like", "seems like").
    fn extract_preference(&self, content: &str) -> Option<String> {
        // Use to_ascii_lowercase() to preserve byte alignment with `content`.
        let lower = content.to_ascii_lowercase();

        // Imperative markers valid without subject check
        const IMPERATIVE_MARKERS: &[&str] = &["always", "never"];

        // Subject-requiring markers
        const SUBJECT_MARKERS: &[&str] = &[
            "prefer",
            "like",
            "want",
            "better",
            "should",
            "dislike",
            "avoid",
            "recommend",
            "favorite",
            "rather",
            "instead of",
            "opt for",
        ];

        // First-person subjects that validate a preference
        const FIRST_PERSON: &[&str] = &["i ", "we ", "my ", "our ", "user "];

        // Simile prefixes that invalidate "like" as a preference marker
        const SIMILE_PREFIXES: &[&str] = &["looks ", "seems ", "feels ", "sounds ", "acts "];

        // Try imperative markers first (no subject check needed)
        for marker in IMPERATIVE_MARKERS {
            if let Some(pos) = lower.find(marker) {
                if let Some(sentence) = Self::extract_sentence(content, pos) {
                    return Some(sentence);
                }
            }
        }

        // Subject-requiring markers: need first-person subject nearby
        for marker in SUBJECT_MARKERS {
            if let Some(pos) = lower.find(marker) {
                // Reject simile usage of "like"
                if *marker == "like" {
                    let before = &lower[..pos];
                    if SIMILE_PREFIXES.iter().any(|p| before.ends_with(p)) {
                        continue;
                    }
                }

                // Check for first-person subject within ~30 chars before the marker
                let lookback_start = pos.saturating_sub(30);
                let before_marker = &lower[lookback_start..pos];
                if FIRST_PERSON.iter().any(|s| before_marker.contains(s)) {
                    if let Some(sentence) = Self::extract_sentence(content, pos) {
                        return Some(sentence);
                    }
                }
            }
        }
        None
    }

    /// Extract the most information-dense declarative sentence from `content`.
    ///
    /// Entity mentions RANK sentences here; they no longer gate them. The
    /// previous contract returned `None` unless the winning sentence literally
    /// contained one of `entities`, on the reasoning that a sentence with no
    /// entity mention is generic prose. The reasoning is sound and the mechanism
    /// is not: `entities` is populated by NER upstream, so the rule reads "reject
    /// all knowledge whenever an optional enrichment step produced nothing". It
    /// fails closed, silently, for the entire corpus at once — which is exactly
    /// what happened, and it took the clustering, reinforcement and
    /// contradiction/invalidation layers down with it, since none of them can act
    /// on an input that is always empty.
    ///
    /// What replaces it is not a weaker bar but a differently placed one. The
    /// structural filters are unchanged and self-contained — the 20..=200 char
    /// window, the content-word floor below, `is_fact_shaped` and
    /// `is_knowledge_worthy` at the call site — and above them sits
    /// `CONSOLIDATION_MIN_SUPPORT`, which mints nothing until the same pattern
    /// arrives from at least two DISTINCT memories. "Two independent memories say
    /// this" is stronger evidence of domain relevance than "an NER pass found a
    /// span", and unlike the entity gate it degrades gracefully: a corpus with no
    /// entities yields fewer facts, not zero.
    ///
    /// When entities ARE present they still do real work — `entity_bonus` biases
    /// selection toward the sentence that actually names the domain objects, so a
    /// working NER pass improves WHICH sentence is chosen without ever deciding
    /// WHETHER one is.
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

            // Content-word floor. Raised 3 -> 4 as the deliberate offset for the
            // removed entity gate: a proposition worth storing names at least a
            // subject, a predicate and something predicated of it, and at three
            // non-stop-words a 20-char fragment can still be a caption or a
            // heading. Four is the smallest floor that requires an actual clause
            // and it costs nothing on real prose (`CONSOLIDATION_MIN_SUPPORT`
            // remains the filter that decides what is minted).
            if content_words < CONSOLIDATION_SALIENT_MIN_CONTENT_WORDS {
                continue;
            }

            // Entity mentions are a RANKING signal, never a gate — see the fn doc.
            // A sentence that names a known entity outranks one that does not, but
            // an empty `entities` list (NER unavailable or silent) leaves ranking
            // to content density instead of suppressing the sentence entirely.
            let entity_match_count = entity_lower
                .iter()
                .filter(|e| lower.contains(e.as_str()))
                .count();

            let entity_bonus = entity_match_count as f32 * 2.0;
            let score = content_words as f32 + entity_bonus;

            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((trimmed.to_string(), score));
            }
        }

        best.map(|(s, _)| s)
    }

    // ── Structural Validators (Thalamic Gating) ────────────────────────

    /// Reject text that is structurally not a fact, regardless of content.
    ///
    /// Models thalamic sensory gating: filters stimulus structure before
    /// higher cortical processing (extractors) evaluates meaning.
    /// Questions, truncated fragments, code/config, log lines, and
    /// path-heavy text are rejected before any extractor runs.
    fn is_fact_shaped(text: &str) -> bool {
        let trimmed = text.trim();

        // 1. Question detection — questions are not declarative facts
        if trimmed.ends_with('?') {
            return false;
        }

        // 2. Truncation detection — incomplete fragments
        if trimmed.ends_with("...") || trimmed.ends_with("..") {
            return false;
        }

        // 3. Mid-sentence fragment: starts with lowercase and isn't a common
        //    sentence-starting word (articles, prepositions, conjunctions)
        if let Some(first) = trimmed.chars().next() {
            if first.is_lowercase() {
                const LOWERCASE_STARTERS: &[&str] = &[
                    "the ", "a ", "an ", "in ", "on ", "at ", "for ", "to ", "if ", "when ",
                    "but ", "and ", "or ", "so ", "as ", "by ", "with ",
                ];
                let lower_start = trimmed.to_ascii_lowercase();
                if !LOWERCASE_STARTERS
                    .iter()
                    .any(|s| lower_start.starts_with(s))
                {
                    return false;
                }
            }
        }

        // 4. Code/config fragment: high density of special characters
        let special_count = trimmed
            .chars()
            .filter(|c| {
                matches!(
                    c,
                    '{' | '}'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | ';'
                        | '='
                        | '<'
                        | '>'
                        | '|'
                        | '&'
                        | '#'
                        | '$'
                        | '`'
                        | '\\'
                )
            })
            .count();
        let alpha_count = trimmed.chars().filter(|c| c.is_alphanumeric()).count();
        if alpha_count > 0 && special_count as f32 / alpha_count as f32 > 0.25 {
            return false;
        }

        // 5. Log line detection — timestamps, log levels, structured output
        const LOG_PATTERNS: &[&str] = &[
            "[info]",
            "[warn]",
            "[error]",
            "[debug]",
            "[trace]",
            "info:",
            "warn:",
            "error:",
            "debug:",
            "trace:",
            " at 0x",
            "thread '",
            "stack trace",
        ];
        let lower = trimmed.to_ascii_lowercase();
        if LOG_PATTERNS.iter().any(|p| lower.contains(p)) {
            return false;
        }

        // 6. Path-heavy text (file listings, not knowledge)
        let slash_count = trimmed.chars().filter(|c| *c == '/' || *c == '\\').count();
        let word_count = trimmed.split_whitespace().count();
        if slash_count >= 2 && word_count < 8 {
            return false;
        }

        true
    }

    // ── Quality Gate ──────────────────────────────────────────────────────

    /// Reject operational noise that shouldn't become semantic facts.
    /// Facts should capture domain knowledge, decisions, and patterns —
    /// not session lifecycle events, tool invocations, or status updates.
    pub(crate) fn is_knowledge_worthy(text: &str) -> bool {
        let lower = text.to_lowercase();
        let len = text.trim().len();

        // Too short to carry meaningful knowledge
        if len < 25 {
            return false;
        }

        // Session lifecycle noise — analogous to hippocampal filtering during
        // consolidation: routine maintenance signals are pruned before entering
        // long-term cortical storage (Dudai 2004, systems consolidation).
        const SESSION_NOISE: &[&str] = &[
            "session started",
            "session ended",
            "session ended:",
            "session summary",
            "session in ",
            "context compressed",
            "context window",
            "token budget",
            "tokens used",
            "proactive_context",
            "auto_ingest",
            "memory surfaced",
            "memories surfaced",
            "memory stored",
            "memories stored",
            "hit rate",
            "topics changed",
            "entities extracted",
            "compressions ran",
        ];
        if SESSION_NOISE.iter().any(|s| lower.contains(s)) {
            return false;
        }

        // Tool/MCP invocation logs
        const TOOL_NOISE: &[&str] = &[
            "tool call",
            "tool_name",
            "mcp tool",
            "called remember",
            "called recall",
            "called forget",
            "hook triggered",
            "hook fired",
        ];
        if TOOL_NOISE.iter().any(|s| lower.contains(s)) {
            return false;
        }

        // Todo/task status chatter — task state transitions are operational
        // metadata, not semantic knowledge worth encoding into long-term memory.
        const TODO_NOISE: &[&str] = &[
            "todo created",
            "todo completed",
            "todo updated",
            "todo updated (status",
            "task created",
            "task completed",
            "task updated",
            "marked as done",
            "marked as complete",
            "moved to backlog",
            "moved to in_progress",
            "status → done",
            "status → inprogress",
            "status → in_progress",
            "status → cancelled",
            "status → blocked",
        ];
        if TODO_NOISE.iter().any(|s| lower.contains(s)) {
            return false;
        }

        // System output / structured markup that leaked into fact extraction.
        // XML tags, agent output fragments, and assistant-prefixed entries are
        // not domain knowledge — they're internal protocol noise.
        const SYSTEM_OUTPUT_NOISE: &[&str] = &[
            "<task-notification>",
            "</output-file>",
            "<status>killed</status>",
            "<status>completed</status>",
            "[assistant: codeedit]",
            "[assistant: observation]",
            "[assistant: search]",
            "[assistant: decision]",
            "[assistant: pattern]",
            "reconnected after",
            "oom crash",
            "surfaced 5 memories",
            "surfaced 3 memories",
            "agent \"",
        ];
        if SYSTEM_OUTPUT_NOISE.iter().any(|s| lower.contains(s)) {
            return false;
        }

        // Bare file paths as standalone "facts" (e.g. "src/memory/mod.rs")
        // Heuristic: short text with path separators and few words is not knowledge
        let path_chars = lower.chars().filter(|c| *c == '/' || *c == '\\').count();
        let word_count = text.split_whitespace().count();
        if path_chars >= 1 && word_count < 6 {
            return false;
        }

        true
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Extract the sentence containing a character position
    fn extract_sentence(content: &str, pos: usize) -> Option<String> {
        let start = content[..pos].rfind(['.', '!', '?']);
        let start = start.map(|i| i + 1).unwrap_or(0);

        let end = content[pos..].find(['.', '!', '?']);
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
        assert!(pipeline.keyword_extractor.is_stop_word("the"));
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

    /// Build a deterministic multi-word body of the requested length.
    fn make_body(words: usize) -> String {
        (0..words)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn semantic_compression_preserves_full_content() {
        let pipeline = CompressionPipeline::new();
        let body = make_body(300);
        let memory = create_test_memory(&body, 0.1);

        let compressed = pipeline.compress_semantic(&memory).unwrap();
        assert!(compressed.compressed);
        // Full body must survive byte-for-byte — the summary is additive.
        assert_eq!(compressed.experience.content, body);
        // The extractive summary is stored alongside keywords as metadata.
        assert!(compressed.experience.metadata.contains_key("summary"));
        assert!(compressed.experience.metadata.contains_key("keywords"));

        // Decompression restores the uncompressed state with content intact.
        let restored = pipeline.decompress(&compressed).unwrap();
        assert!(!restored.compressed);
        assert_eq!(restored.experience.content, body);
    }

    #[test]
    fn hybrid_compression_round_trips_full_content() {
        let pipeline = CompressionPipeline::new();
        let body = make_body(300);
        let memory = create_test_memory(&body, 0.6);

        let compressed = pipeline.compress_hybrid(&memory).unwrap();
        assert!(compressed.compressed);
        assert_eq!(pipeline.get_strategy(&compressed), Some("hybrid"));

        let restored = pipeline.decompress(&compressed).unwrap();
        assert!(!restored.compressed);
        assert_eq!(restored.experience.content, body);
    }

    #[test]
    fn promotion_to_longterm_never_truncates() {
        // Exercises the exact entrypoint the tier-promotion path invokes:
        // MemorySystem::promote_session_to_longterm → compressor.compress().
        // An old promoted memory is routed by select_strategy to either
        // Summarization (low importance) or Hybrid (mid importance). Neither
        // may truncate content. Drive compress() so the strategy selection is
        // exercised end-to-end, not the private per-strategy functions.
        let pipeline = CompressionPipeline::new();
        let body = make_body(300);

        // Old + low importance (<0.5) → Summarization strategy.
        let mut low = create_test_memory(&body, 0.1);
        low.created_at = chrono::Utc::now() - chrono::Duration::days(100);
        let c_low = pipeline.compress(&low).unwrap();
        assert_eq!(pipeline.get_strategy(&c_low), Some("semantic"));
        assert_eq!(c_low.experience.content, body);
        assert_eq!(
            pipeline.decompress(&c_low).unwrap().experience.content,
            body
        );

        // Old + mid importance (0.5..=0.8) → Hybrid strategy.
        let mut mid = create_test_memory(&body, 0.6);
        mid.created_at = chrono::Utc::now() - chrono::Duration::days(100);
        let c_mid = pipeline.compress(&mid).unwrap();
        assert_eq!(pipeline.get_strategy(&c_mid), Some("hybrid"));
        assert_eq!(
            pipeline.decompress(&c_mid).unwrap().experience.content,
            body
        );
    }

    #[test]
    fn test_is_lossless() {
        let pipeline = CompressionPipeline::new();
        let memory = create_test_memory("Test content", 0.9);

        assert!(pipeline.is_lossless(&memory));

        let compressed_lz4 = pipeline.compress_lz4(&memory).unwrap();
        assert!(pipeline.is_lossless(&compressed_lz4));

        // New-format semantic compression is lossless: the full content is
        // preserved and the summary is additive.
        let compressed_semantic = pipeline.compress_semantic(&memory).unwrap();
        assert!(pipeline.is_lossless(&compressed_semantic));
    }

    #[test]
    fn legacy_truncated_semantic_is_reported_unrecoverable() {
        // Simulate a pre-fix record: `semantic` strategy, truncated body ending
        // in "...", and no additive `summary` key. It must be reported as
        // unrecoverable rather than silently returning the summary as content.
        let pipeline = CompressionPipeline::new();
        let mut legacy = create_test_memory("first few words of the original body ...", 0.1);
        legacy.compressed = true;
        legacy
            .experience
            .metadata
            .insert("compression_strategy".to_string(), "semantic".to_string());
        legacy
            .experience
            .metadata
            .insert("keywords".to_string(), "first,words,original".to_string());

        assert!(!pipeline.is_lossless(&legacy));
        let result = pipeline.decompress(&legacy);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("cannot be recovered"));
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
        let keywords = extractor.extract_texts(text);

        assert!(!keywords.is_empty());
        // YAKE extracts n-grams; check that meaningful terms appear
        let joined = keywords.join(" ");
        assert!(
            joined.contains("rust") || joined.contains("memory") || joined.contains("programming"),
            "Expected meaningful keywords, got: {keywords:?}"
        );
    }

    #[test]
    fn test_stop_words_filtered() {
        let extractor = KeywordExtractor::new();
        assert!(extractor.is_stop_word("the"));
        assert!(extractor.is_stop_word("is"));
        assert!(!extractor.is_stop_word("rust"));
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
            invalidated_at: None,
            invalidated_by: None,
            contradicts: Vec::new(),
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
            invalidated_at: None,
            invalidated_by: None,
            contradicts: Vec::new(),
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
            invalidated_at: None,
            invalidated_by: None,
            contradicts: Vec::new(),
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

        // Use Learning type (eligible for definition/pattern/procedure extractors).
        // Content uses definition markers (" is ") with proper subjects to trigger
        // the extract_definition extractor after hardening.
        let m1 = create_typed_memory(
            "Rust is a systems programming language with memory safety and performance guarantees",
            0.8,
            ExperienceType::Learning,
            vec!["Rust".to_string(), "memory safety".to_string()],
        );
        let m2 = create_typed_memory(
            "Rust is a compiled language providing memory safety with great performance",
            0.7,
            ExperienceType::Learning,
            vec!["Rust".to_string(), "memory safety".to_string()],
        );
        let m3 = create_typed_memory(
            "Python is a dynamic language great for data science and machine learning workflows",
            0.6,
            ExperienceType::Learning,
            vec!["Python".to_string(), "data science".to_string()],
        );

        let result = consolidator.consolidate(&[m1, m2, m3]);

        // The two Rust memories should cluster and produce a fact
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

        // Learning type is eligible for definition + pattern + procedure.
        // Content has a definition ("RocksDB is ...") and a pattern ("error")
        // to trigger multiple extractors.
        let memory = create_typed_memory(
            "RocksDB is a high-performance embedded key-value database engine. If you see a corruption error, run the repair tool.",
            0.8,
            ExperienceType::Learning,
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
    fn test_no_relates_to_patterns() {
        let consolidator = SemanticConsolidator::new();

        let memory = create_typed_memory(
            "Testing entity extraction without template patterns",
            0.7,
            ExperienceType::Observation,
            vec!["JWT".to_string(), "Auth".to_string(), "Token".to_string()],
        );

        let candidates = consolidator.extract_fact_candidates(&memory);

        // Entity pair "relates to" patterns were removed (redundant with knowledge graph)
        let has_relates_to = candidates.iter().any(|(t, _)| t.contains("relates to"));
        assert!(
            !has_relates_to,
            "Should not produce synthetic 'X relates to Y' patterns"
        );
    }

    #[test]
    fn test_quality_gate_rejects_session_noise() {
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "Session started at 2:15 PM with token budget of 200k"
        ));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "Context compressed after reaching 80% of token budget"
        ));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "3 memories surfaced via proactive_context for the current query"
        ));
    }

    #[test]
    fn test_quality_gate_rejects_todo_noise() {
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "Todo created for implementing the new authentication module"
        ));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "Task completed after fixing the database connection pooling issue"
        ));
    }

    #[test]
    fn test_quality_gate_rejects_short_text() {
        assert!(!SemanticConsolidator::is_knowledge_worthy("short"));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "too short to be fact"
        ));
    }

    #[test]
    fn test_quality_gate_rejects_bare_file_paths() {
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "src/memory/compression.rs:919"
        ));
        assert!(!SemanticConsolidator::is_knowledge_worthy(
            "hooks/memory-hook.ts file"
        ));
    }

    #[test]
    fn test_quality_gate_accepts_genuine_knowledge() {
        assert!(SemanticConsolidator::is_knowledge_worthy(
            "Rust provides memory safety without garbage collection through its ownership system"
        ));
        assert!(SemanticConsolidator::is_knowledge_worthy(
            "The Vamana index automatically switches to SPANN when the dataset exceeds 100k vectors"
        ));
        assert!(SemanticConsolidator::is_knowledge_worthy(
            "Hebbian learning uses additive boost and multiplicative decay for asymmetric strengthening"
        ));
        assert!(SemanticConsolidator::is_knowledge_worthy(
            "RocksDB column families reduce file descriptor usage by consolidating multiple stores"
        ));
    }

    // ── Declarative candidate extraction ────────────────────────────────────
    //
    // The regression these pin: the candidate layer produced NOTHING for plain
    // declarative sentences, so consolidation never had input to cluster, no
    // facts were minted on ordinary corpora, and the contradiction/invalidation
    // machinery downstream sat idle. Each test below isolates one of the gates
    // that caused it.

    /// The demo-corpus sentence, with NO NER entities at all.
    ///
    /// This is the exact shape that produced zero candidates: the general
    /// extractor refused any sentence that did not literally contain one of
    /// `experience.entities`, so an empty entity list suppressed the entire
    /// semantic layer. Entities are now a ranking signal, so extraction is
    /// independent of whether NER emitted anything.
    #[test]
    fn declarative_sentence_yields_a_candidate_with_no_ner_entities() {
        let consolidator = SemanticConsolidator::new();
        let memory = create_typed_memory(
            "Initial reports said four crew members were injured in the bridge collapse.",
            0.3,
            ExperienceType::Observation,
            vec![], // NER emitted nothing — the live production condition
        );

        let candidates = consolidator.extract_fact_candidates(&memory);

        assert!(
            !candidates.is_empty(),
            "a plain declarative sentence must mint a candidate without NER entities"
        );
        assert!(
            candidates
                .iter()
                .any(|(t, _)| t.contains("four crew members were injured")),
            "the candidate must be the real sentence, got {candidates:?}"
        );
    }

    /// The type gate was the second failure in series: `remember` defaults to
    /// `Observation`, and the declarative extractor ran on `Observation` alone,
    /// so `Conversation`, `Task` and `Intention` reached ZERO extractors for
    /// ordinary prose — the other four extractors are keyword banks this
    /// sentence matches none of.
    #[test]
    fn declarative_extractor_runs_on_every_non_operational_type() {
        let consolidator = SemanticConsolidator::new();
        let sentence =
            "Initial reports said four crew members were injured in the bridge collapse.";

        for exp_type in [
            ExperienceType::Observation,
            ExperienceType::Conversation,
            ExperienceType::Task,
            ExperienceType::Intention,
            ExperienceType::Decision,
            ExperienceType::Learning,
            ExperienceType::Discovery,
            ExperienceType::Pattern,
            ExperienceType::Error,
        ] {
            let memory = create_typed_memory(sentence, 0.3, exp_type.clone(), vec![]);
            let candidates = consolidator.extract_fact_candidates(&memory);
            assert!(
                !candidates.is_empty(),
                "{exp_type:?} must produce a declarative candidate; it produced none"
            );
        }
    }

    /// The general extractor must NOT reopen the operational types. Execution
    /// traces are logs, not knowledge, and they are rejected before any
    /// extractor runs.
    #[test]
    fn operational_types_still_produce_no_candidates() {
        let consolidator = SemanticConsolidator::new();
        let sentence =
            "Initial reports said four crew members were injured in the bridge collapse.";

        for exp_type in [
            ExperienceType::Context,
            ExperienceType::Command,
            ExperienceType::CodeEdit,
            ExperienceType::FileAccess,
            ExperienceType::Search,
        ] {
            let memory = create_typed_memory(sentence, 0.9, exp_type.clone(), vec![]);
            assert!(
                consolidator.extract_fact_candidates(&memory).is_empty(),
                "{exp_type:?} is an execution trace and must stay silent"
            );
        }
    }

    /// The third failure in series: the extractor demanded `importance >= 0.5`,
    /// but `calculate_importance` gives `Observation` — the DEFAULT experience
    /// type — the 0.05 catch-all type weight, so a realistic observation lands
    /// near 0.23 and could never clear the bar. Importance now scales the
    /// candidate's confidence instead of deciding whether it exists.
    #[test]
    fn low_importance_no_longer_suppresses_declarative_extraction() {
        let consolidator = SemanticConsolidator::new();
        let memory = create_typed_memory(
            "Initial reports said four crew members were injured in the bridge collapse.",
            0.23, // what a real default-typed observation actually scores
            ExperienceType::Observation,
            vec![],
        );

        let candidates = consolidator.extract_fact_candidates(&memory);
        assert!(
            !candidates.is_empty(),
            "a below-0.5 importance must lower confidence, not erase the candidate"
        );

        // ...and importance still does its real job: it weights confidence.
        let low = candidates[0].1;
        let high_memory = create_typed_memory(
            "Initial reports said four crew members were injured in the bridge collapse.",
            0.9,
            ExperienceType::Observation,
            vec![],
        );
        let high = consolidator.extract_fact_candidates(&high_memory)[0].1;
        assert!(
            high > low,
            "importance must still rank candidates: {high} should exceed {low}"
        );
    }

    /// Entities rank, they do not gate. With entities present the sentence that
    /// mentions one is selected over a denser one that does not; with entities
    /// absent, selection falls back to content density instead of returning
    /// nothing.
    #[test]
    fn entity_mentions_rank_sentences_rather_than_gating_them() {
        let consolidator = SemanticConsolidator::new();
        // The two sentences share an identical tail, so sentence 2 carries
        // exactly ONE more content word than sentence 1 whatever the stop-word
        // list contains. Density alone therefore picks sentence 2; a single
        // entity hit (+2.0) is enough to flip the choice to sentence 1.
        let content = "The Dali blocked the shipping channel for weeks. \
                       The salvage barge blocked the shipping channel for weeks.";

        let with_entity = create_typed_memory(
            content,
            0.6,
            ExperienceType::Observation,
            vec!["Dali".to_string()],
        );
        let picked = consolidator.extract_fact_candidates(&with_entity);
        assert!(
            picked.iter().any(|(t, _)| t.contains("Dali")),
            "an entity mention must outrank raw density, got {picked:?}"
        );

        let without_entity = create_typed_memory(content, 0.6, ExperienceType::Observation, vec![]);
        let fallback = consolidator.extract_fact_candidates(&without_entity);
        assert!(
            !fallback.is_empty(),
            "with no entities the densest sentence must still be extracted"
        );
        assert!(
            fallback.iter().any(|(t, _)| t.contains("salvage barge")),
            "without an entity signal, density decides; got {fallback:?}"
        );
    }

    /// Pins `CONSOLIDATION_SALIENT_MIN_CONTENT_WORDS`, the structural half of
    /// what replaced the entity gate. Three content words still admits a
    /// caption; four requires a clause.
    #[test]
    fn content_word_floor_separates_a_caption_from_a_clause() {
        assert_eq!(
            CONSOLIDATION_SALIENT_MIN_CONTENT_WORDS, 4,
            "this test pins the floor; update both together"
        );

        let consolidator = SemanticConsolidator::new();
        // State the stop-word assumptions the fixtures rely on, so a change to
        // the stop-word list fails HERE with a readable reason.
        for w in ["the", "and", "in"] {
            assert!(
                consolidator.keyword_extractor.is_stop_word(w),
                "fixture assumes '{w}' is a stop word"
            );
        }
        for w in ["barge", "tugboat", "harbor", "sank"] {
            assert!(
                !consolidator.keyword_extractor.is_stop_word(w),
                "fixture assumes '{w}' is a content word"
            );
        }

        // 3 content words (barge, tugboat, harbor) — long enough for every other
        // gate, so the floor is the only thing that can reject it.
        let caption = "The barge and the tugboat in the harbor";
        assert!(
            caption.len() >= 25,
            "must clear is_knowledge_worthy on length"
        );
        assert!(consolidator
            .extract_salient_statement(caption, &[])
            .is_none());

        // 4 content words — the same fixture plus a predicate.
        let clause = "The barge and the tugboat in the harbor sank";
        assert_eq!(
            consolidator.extract_salient_statement(clause, &[]),
            Some(clause.to_string()),
            "four content words is a clause and must be extracted"
        );
    }

    /// The whole point of extracting candidates: repeated mentions must reach
    /// the support threshold and mint a fact, while a single mention must not.
    /// `CONSOLIDATION_MIN_SUPPORT` is the corroboration filter that replaced the
    /// entity gate, so this is the test that shows it carrying the load.
    #[test]
    fn repeated_declarative_mentions_reach_the_support_threshold() {
        let consolidator = SemanticConsolidator::with_thresholds(2, 0);

        let m1 = create_typed_memory(
            "Initial reports said four crew members were injured in the bridge collapse.",
            0.3,
            ExperienceType::Observation,
            vec![],
        );
        let m2 = create_typed_memory(
            "Early reports said four crew members were injured in the bridge collapse.",
            0.3,
            ExperienceType::Observation,
            vec![],
        );

        // One mention is an anecdote — no fact.
        let single = consolidator.consolidate(std::slice::from_ref(&m1));
        assert_eq!(
            single.facts_extracted, 0,
            "a single mention must not mint a fact: {:?}",
            single.new_facts
        );

        // Two independent mentions are corroboration — a fact.
        let paired = consolidator.consolidate(&[m1, m2]);
        assert!(
            paired.facts_extracted >= 1,
            "two corroborating memories must mint a fact, got {}",
            paired.facts_extracted
        );
        assert!(
            paired
                .new_facts
                .iter()
                .any(|f| f.fact.contains("crew members were injured")),
            "the minted fact must be the real sentence, got {:?}",
            paired.new_facts
        );
    }

    #[test]
    fn test_quality_gate_accepts_file_paths_with_prose() {
        // File paths in context of explanation are fine
        assert!(SemanticConsolidator::is_knowledge_worthy(
            "The router configuration in src/handlers/router.rs defines all API endpoint routes for the server"
        ));
    }

    /// The 74-memory demonstration corpus from `seat/eval/seed-demo-corpus.mjs`,
    /// verbatim (content, experience type). This is the "realistic corpus" that
    /// held zero facts in production; the stage-count test below pins where in
    /// the pipeline candidates exist and what the corroboration policy keeps.
    const DEMO_CORPUS: &[(&str, ExperienceType)] = &[
        ("Container ship Dali lost propulsion at 01:24 local time because an electrical breaker tripped during departure from the Port of Baltimore.", ExperienceType::Observation),
        ("The loss of propulsion led to the Dali drifting off the channel heading despite the crew dropping anchor.", ExperienceType::Observation),
        ("The drifting vessel struck a support pier of the Francis Scott Key Bridge, which triggered the collapse of the main truss spans into the Patapsco River.", ExperienceType::Observation),
        ("Because the wreckage blocked the shipping channel, the Port of Baltimore suspended all vessel traffic.", ExperienceType::Decision),
        ("The port suspension halted roll-on/roll-off automobile shipments, so Maersk rerouted its services to the Port of New York and New Jersey.", ExperienceType::Decision),
        ("Overflow automobile volume was diverted south to the Port of Virginia at Norfolk as a result of the closure.", ExperienceType::Decision),
        ("Coal exports from the CSX Curtis Bay terminal stopped for six weeks because bulk carriers could not transit the blocked channel.", ExperienceType::Observation),
        ("NTSB chair Jennifer Homendy stated the investigation would examine the Dali's electrical system, after inspectors recovered the voyage data recorder.", ExperienceType::Observation),
        ("Synergy Marine Group, the ship manager, confirmed the crew reported electrical failures during port inspections in the days before departure.", ExperienceType::Observation),
        ("Captain Maynard of the pilots association credited the pilot's mayday call with stopping bridge traffic, which prevented further casualties.", ExperienceType::Learning),
        ("The bridge lacked pier protection dolphins that current design standards require, a factor engineers said contributed to the total collapse.", ExperienceType::Learning),
        ("Salvage crews used the floating crane Chesapeake 1000 to cut the collapsed truss sections for channel clearance.", ExperienceType::Task),
        ("Initial reports said four crew members were injured in the collapse.", ExperienceType::Observation),
        ("Corrected report: no crew members aboard the Dali were injured; the casualties were road workers on the bridge deck.", ExperienceType::Observation),
        ("The Vamana index rebuild cut recall latency from 340ms to 92ms on the evaluation corpus.", ExperienceType::Learning),
        ("Routine port-state inspection of the Dali at Seagirt noted an intermittent low-voltage alarm on the main switchboard; the crew reset it and the inspector logged it as resolved.", ExperienceType::Observation),
        ("Electrician's shift note: reefer bank on the Dali's forward feeder showed voltage sag twice during cargo operations, within tolerance but recurring.", ExperienceType::Observation),
        ("Synergy Marine deferred the Dali's switchboard breaker overhaul to the next scheduled dry dock to avoid a berth overstay.", ExperienceType::Decision),
        ("Gate scale check: container MSKU-4471820 declared 4,200 kg on the manifest but weighed 28,650 kg at the Seagirt in-gate; flagged for VGM re-verification.", ExperienceType::Observation),
        ("Drayage truck T-118 carrying an export reefer pinged 40 km off its assigned corridor near Annapolis during the evening run.", ExperienceType::Observation),
        ("MSC Brianna completed cargo operations at Seagirt berth 3 and departed on the evening tide.", ExperienceType::Observation),
        ("Crane STS-07 at Seagirt completed its 4,000-hour preventive maintenance; hoist brake pads replaced.", ExperienceType::Observation),
        ("Morning shift moved 1,847 containers at Seagirt, slightly above the 30-day average.", ExperienceType::Observation),
        ("Evergreen Ever Focus assigned to Seagirt berth 2 for Thursday arrival, draft 12.8 metres.", ExperienceType::Observation),
        ("Fog delayed pilot boardings in the Craighill Channel for two hours before conditions lifted.", ExperienceType::Observation),
        ("Customs placed a documentation hold on 12 containers from the CMA CGM Argentina pending broker corrections.", ExperienceType::Observation),
        ("Dundalk Marine Terminal handled 3,200 imported vehicles this week, in line with forecast.", ExperienceType::Observation),
        ("Longshore gang 14 set a terminal record with 42 crane moves per hour on the night shift.", ExperienceType::Observation),
        ("The Wallenius Wilhelmsen Tosca discharged heavy machinery at Dundalk without incident.", ExperienceType::Observation),
        ("Berth 1 at Seagirt scheduled for fender replacement next month; no vessel impact expected.", ExperienceType::Task),
        ("Reefer monitoring rounds found all 240 plugged units within temperature spec.", ExperienceType::Observation),
        ("The harbor tug Bridget McAllister returned to service after routine engine overhaul.", ExperienceType::Observation),
        ("Chesapeake Bay pilots reported normal transit conditions; visibility eight nautical miles.", ExperienceType::Observation),
        ("Weekly safety briefing covered updated lashing procedures for high-cube containers.", ExperienceType::Learning),
        ("CSX intermodal ramp turned 96% of import boxes within 48 hours this week.", ExperienceType::Observation),
        ("Empty container yard at Fairfield reached 78% utilization; repositioning plan drafted.", ExperienceType::Observation),
        ("Maersk Kensington arrived at Seagirt berth 4 with 2,900 import containers.", ExperienceType::Observation),
        ("Gate cameras at Dundalk upgraded to read damaged container door labels.", ExperienceType::Observation),
        ("Thunderstorm forecast prompted crane wind-speed monitoring; operations continued below limits.", ExperienceType::Observation),
        ("Hapag-Lloyd Atlanta Express completed bunkering at anchorage before berthing.", ExperienceType::Observation),
        ("Terminal operating system patched overnight; gate transactions resumed at 05:00 without backlog.", ExperienceType::Observation),
        ("Crane STS-04 flagged a slew-drive vibration reading at the high end of normal; monitoring continued.", ExperienceType::Observation),
        ("Quarterly emissions report filed: terminal equipment idle time down 9% year over year.", ExperienceType::Observation),
        ("Two export soybean trains unloaded at the CNX Marine Terminal on schedule.", ExperienceType::Observation),
        ("Vessel traffic service logged 31 deep-draft transits through the main channel this week.", ExperienceType::Observation),
        ("Warehouse C at Point Breeze passed its annual fire-suppression inspection.", ExperienceType::Observation),
        ("ZIM Baltimore sailed for Norfolk after a routine eight-hour port call.", ExperienceType::Observation),
        ("Chassis pool availability tightened to 91%; provider notified per service agreement.", ExperienceType::Observation),
        ("Pilot association scheduled semi-annual bridge-team simulator training for member pilots.", ExperienceType::Learning),
        ("Seagirt yard block E re-striped; reefer rows gained four additional plug positions.", ExperienceType::Observation),
        ("The Atlantic Container Line Atlantic Sun loaded project cargo bound for Antwerp.", ExperienceType::Observation),
        ("Random cargo exam rate held at 3.1% for the month, unchanged from prior period.", ExperienceType::Observation),
        ("Tug assist requirements reviewed for vessels over 300 metres; no changes recommended.", ExperienceType::Learning),
        ("Marine terminal lighting audit found six fixtures below lux spec in the Dundalk rail yard.", ExperienceType::Observation),
        ("COSCO Development windowed for Saturday arrival; berth 3 turn time projected at 22 hours.", ExperienceType::Observation),
        ("Monthly draft survey of the Curtis Bay coal pier showed silting within dredge tolerance.", ExperienceType::Observation),
        ("Breakbulk crew discharged wind turbine blades at North Locust Point over two shifts.", ExperienceType::Observation),
        ("Ship chandler deliveries consolidated to a single gate window to reduce congestion.", ExperienceType::Decision),
        ("Crane operator recertification completed for 28 operators; two scheduled for retest.", ExperienceType::Observation),
        ("The Grimaldi Grande Baltimora loaded 1,100 export vehicles at Dundalk.", ExperienceType::Observation),
        ("Water taxi service resumed its harbor route after seasonal maintenance.", ExperienceType::Observation),
        ("Line handlers reported a parted stern line on the bulk carrier Ocean Prosperity; replaced without delay to sailing.", ExperienceType::Observation),
        ("Port administration approved the fiscal-year dredging budget for the access channels.", ExperienceType::Decision),
        ("Refrigerated cargo volumes rose 14% month over month, led by poultry exports.", ExperienceType::Observation),
        ("Security drill simulated an unauthorized gate entry; response time met the standard.", ExperienceType::Observation),
        ("The container freight station cleared its LCL backlog ahead of the holiday weekend.", ExperienceType::Observation),
        ("Rail dwell for export coal at Curtis Bay averaged 2.1 days, best quarter in two years.", ExperienceType::Observation),
        ("Evergreen Ever Focus departed berth 2 after a 19-hour turn, one hour ahead of window.", ExperienceType::Observation),
        ("Stevedore payroll system migration completed; no missed shifts reported.", ExperienceType::Observation),
        ("Harbor survey vessel completed side-scan mapping of anchorage B.", ExperienceType::Observation),
        ("Seagirt gate processed 4,102 truck transactions, a seasonal high, with average turn under 40 minutes.", ExperienceType::Observation),
    ];

    fn demo_corpus_memories(age_days: i64) -> Vec<Memory> {
        DEMO_CORPUS
            .iter()
            .map(|(content, exp_type)| {
                let experience = Experience {
                    content: content.to_string(),
                    experience_type: exp_type.clone(),
                    entities: Vec::new(),
                    ..Default::default()
                };
                Memory::new(
                    MemoryId(Uuid::new_v4()),
                    experience,
                    0.3,
                    None,
                    None,
                    None,
                    Some(chrono::Utc::now() - chrono::Duration::days(age_days)),
                )
            })
            .collect()
    }

    /// Stage-by-stage instrumentation of the consolidation pipeline on the
    /// demo corpus. Prints the counts (run with --nocapture) and pins the
    /// two facts about this corpus that the zero-facts investigation
    /// established:
    ///  1. The extractor DOES propose candidates from realistic prose
    ///     (world 3, "extraction gap", is false).
    ///  2. Corroboration (min_support >= 2) is what decides how many facts a
    ///     unique-statement corpus mints — few, and that is the documented
    ///     policy, not a defect.
    #[test]
    fn demo_corpus_stage_counts() {
        let consolidator = SemanticConsolidator::new();
        let memories = demo_corpus_memories(8); // older than CONSOLIDATION_MIN_AGE_DAYS

        // Stage 1: candidate extraction per memory.
        let mut all_candidates: Vec<(String, MemoryId, f32)> = Vec::new();
        let mut memories_with_candidates = 0usize;
        for m in &memories {
            let extracted = consolidator.extract_fact_candidates(m);
            if !extracted.is_empty() {
                memories_with_candidates += 1;
            }
            for (pattern, confidence) in extracted {
                all_candidates.push((pattern, m.id.clone(), confidence));
            }
        }

        // Stage 2: clustering.
        let clusters = consolidator
            .group_candidates_by_similarity(&all_candidates, CONSOLIDATION_JACCARD_THRESHOLD);
        let corroborated = clusters
            .iter()
            .filter(|c| c.members.len() >= CONSOLIDATION_MIN_SUPPORT_SMALL)
            .count();

        // Stage 3: full pipeline.
        let result = consolidator.consolidate(&memories);

        println!("── demo corpus stage counts ──");
        println!("memories:                  {}", memories.len());
        println!("memories with candidates:  {memories_with_candidates}");
        println!("candidates proposed:       {}", all_candidates.len());
        println!("clusters:                  {}", clusters.len());
        println!("clusters >= min_support:   {corroborated}");
        println!("facts minted:              {}", result.facts_extracted);
        for f in &result.new_facts {
            println!(
                "  fact [{:?}] ({} sources): {}",
                f.fact_type,
                f.source_memories.len(),
                f.fact
            );
        }

        // The extractor proposes from the majority of the corpus: zero facts
        // can never again be blamed on "nothing is proposed".
        assert!(
            memories_with_candidates >= DEMO_CORPUS.len() / 2,
            "extractor should propose candidates from most of the demo corpus, \
             got {memories_with_candidates}/{}",
            DEMO_CORPUS.len()
        );
        assert!(
            all_candidates.len() >= memories_with_candidates,
            "at least one candidate per proposing memory"
        );
        // Facts minted equals corroborated clusters — corroboration is the
        // deciding filter, nothing downstream of it silently drops facts.
        assert_eq!(
            result.facts_extracted, corroborated,
            "every corroborated cluster must mint exactly one fact"
        );
    }

    /// A claim and its negation share most of their content stems — negation
    /// is often a one-word difference — so without a polarity gate they clear
    /// the Jaccard threshold and merge into ONE cluster, minting a single
    /// representative fact. The losing side never reaches the store, so the
    /// contradiction/invalidation machinery has nothing to arbitrate: the
    /// correction is silently swallowed at extraction time. Clustering must
    /// keep opposite-polarity candidates apart so BOTH sides mint and
    /// `ingest_candidate` can record the supersession.
    #[test]
    fn contradicting_statements_never_share_a_cluster() {
        let consolidator = SemanticConsolidator::new();
        let mems: Vec<Memory> = [
            "Initial reports said four crew members were injured in the bridge collapse.",
            "Early reports said four crew members were injured in the bridge collapse.",
            "Corrected reports confirm no crew members were injured in the bridge collapse.",
            "Later reports confirm no crew members were injured in the bridge collapse.",
        ]
        .iter()
        .map(|content| {
            let experience = Experience {
                content: content.to_string(),
                experience_type: ExperienceType::Observation,
                entities: Vec::new(),
                ..Default::default()
            };
            Memory::new(
                MemoryId(Uuid::new_v4()),
                experience,
                0.8,
                None,
                None,
                None,
                Some(chrono::Utc::now() - chrono::Duration::days(8)),
            )
        })
        .collect();

        let result = consolidator.consolidate(&mems);
        assert_eq!(
            result.facts_extracted, 2,
            "claim and correction must mint as SEPARATE facts, got {:?}",
            result.new_facts
        );
        assert!(
            result
                .new_facts
                .iter()
                .any(|f| f.fact.contains("four crew members were injured")),
            "the claim side must mint: {:?}",
            result.new_facts
        );
        assert!(
            result
                .new_facts
                .iter()
                .any(|f| f.fact.contains("no crew members were injured")),
            "the correction side must mint: {:?}",
            result.new_facts
        );
    }

    /// Minted facts are ordered by the recency of their newest supporting
    /// memory, oldest claim first: callers ingest in this order, and
    /// contradiction arbitration favours the incoming candidate on equal
    /// support, so the LAST-ingested side — the one backed by the newest
    /// evidence — wins. Without the sort, batch order was storage iteration
    /// order (random UUIDs) and a bulk-seeded claim/correction pair settled
    /// on a coin flip.
    #[test]
    fn minted_facts_are_ordered_by_evidence_recency() {
        let consolidator = SemanticConsolidator::new();
        let mem = |content: &str, days: i64| {
            let experience = Experience {
                content: content.to_string(),
                experience_type: ExperienceType::Observation,
                entities: Vec::new(),
                ..Default::default()
            };
            Memory::new(
                MemoryId(Uuid::new_v4()),
                experience,
                0.8,
                None,
                None,
                None,
                Some(chrono::Utc::now() - chrono::Duration::days(days)),
            )
        };
        // Deliberately interleave: newest pair first in input order.
        let mems = vec![
            mem(
                "Corrected reports confirm no crew members were injured in the bridge collapse.",
                20,
            ),
            mem(
                "Initial reports said four crew members were injured in the bridge collapse.",
                40,
            ),
            mem(
                "Later reports confirm no crew members were injured in the bridge collapse.",
                20,
            ),
            mem(
                "Early reports said four crew members were injured in the bridge collapse.",
                40,
            ),
        ];

        let result = consolidator.consolidate(&mems);
        assert_eq!(result.facts_extracted, 2);
        assert!(
            result.new_facts[0]
                .fact
                .contains("four crew members were injured"),
            "older claim must be first (ingested first, displaced last): {:?}",
            result.new_facts
        );
        assert!(
            result.new_facts[1]
                .fact
                .contains("no crew members were injured"),
            "newest-evidence side must be last (wins the arbitration tie): {:?}",
            result.new_facts
        );
        assert_eq!(
            result.new_fact_ids,
            result
                .new_facts
                .iter()
                .map(|f| f.id.clone())
                .collect::<Vec<_>>(),
            "new_fact_ids must track the sorted order"
        );
    }

    /// The demo corpus produces two DIFFERENT zeros, and `memories_eligible`
    /// is what tells them apart:
    ///
    ///  * Fresh corpus, default thresholds: the age gate
    ///    (CONSOLIDATION_MIN_AGE_DAYS) filters everything out —
    ///    `memories_eligible == 0`. Zero facts because nothing was LOOKED AT.
    ///    (The defect the investigation found was one layer up: the
    ///    extraction watermark advanced past these age-ineligible memories,
    ///    so they stayed unconsolidatable even after aging past the gate.
    ///    See tests/fact_distillation_tests.rs.)
    ///
    ///  * Age gate lifted: every memory IS processed
    ///    (`memories_eligible == corpus`), candidates are proposed from all
    ///    of them, and the corpus STILL mints zero facts — measured above in
    ///    `demo_corpus_stage_counts`, its 71 candidates form 71 singleton
    ///    clusters. Every statement in this corpus is unique, so
    ///    corroboration (min_support >= 2 DISTINCT memories) is never
    ///    satisfied. That zero is the documented policy working as designed:
    ///    minting singletons instead would turn all ~71 routine operational
    ///    statements ("MSC Brianna completed cargo operations…") into
    ///    permanent semantic "facts", which is noise, not knowledge.
    #[test]
    fn fresh_corpus_is_age_gated_not_broken() {
        let consolidator = SemanticConsolidator::new();
        let fresh = demo_corpus_memories(0);
        let result = consolidator.consolidate(&fresh);
        assert_eq!(
            result.memories_eligible, 0,
            "a fresh corpus must be entirely age-ineligible"
        );
        assert_eq!(
            result.facts_extracted, 0,
            "age gate must exclude a fresh corpus from consolidation"
        );

        // Age gate lifted: everything is processed; the zero that remains is
        // the corroboration policy, not an extraction failure.
        let eager = SemanticConsolidator::with_thresholds(CONSOLIDATION_MIN_SUPPORT, 0);
        let eager_result = eager.consolidate(&fresh);
        assert_eq!(
            eager_result.memories_eligible,
            fresh.len(),
            "with the age gate lifted every memory must be processed"
        );
        assert_eq!(
            eager_result.facts_extracted, 0,
            "a corpus of unique statements has no corroborated claims: zero \
             facts is the min_support policy, not a defect — if this ever \
             mints, either the corpus gained restatements or the support \
             policy changed, and both deserve a deliberate look"
        );
    }
}
