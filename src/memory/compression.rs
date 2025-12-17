//! Compression pipeline for memory optimization

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use lz4;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::*;
use crate::constants::{
    COMPRESSION_ACCESS_THRESHOLD, COMPRESSION_AGE_DAYS, COMPRESSION_IMPORTANCE_HIGH,
    COMPRESSION_IMPORTANCE_LOW, CONSOLIDATION_MIN_AGE_DAYS, CONSOLIDATION_MIN_SUPPORT,
    FACT_DECAY_BASE_DAYS, FACT_DECAY_PER_SUPPORT_DAYS, MAX_COMPRESSION_RATIO,
    MAX_DECOMPRESSED_SIZE,
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
        let original = bincode::serialize(&memory.experience)?;
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

            let experience: Experience = bincode::deserialize(&decompressed)?;

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

            if !clean_word.is_empty() && !self.stop_words.contains(&clean_word) {
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

    fn load_stop_words() -> HashSet<String> {
        // Common English stop words
        let words = vec![
            "the", "is", "at", "which", "on", "and", "a", "an", "as", "are", "was", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "to", "of",
            "in", "for", "with", "by", "from", "about", "into", "through", "during", "before",
            "after", "above", "below", "up", "down", "out", "off", "over", "under", "again",
            "further", "then", "once", "there", "these", "those", "this", "that", "it", "its",
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "but", "or", "if",
        ];

        words.into_iter().map(String::from).collect()
    }
}

use std::collections::HashSet;

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
}

/// Semantic consolidation engine
///
/// Extracts durable semantic facts from episodic memories based on:
/// - Repetition: Same information appearing across multiple episodes
/// - Importance: High-importance memories get extracted faster
/// - Age: Older memories are consolidated more aggressively
pub struct SemanticConsolidator {
    keyword_extractor: KeywordExtractor,
    /// Minimum times a pattern must appear to become a fact
    min_support: usize,
    /// Minimum age in days before consolidation
    min_age_days: i64,
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
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(min_support: usize, min_age_days: i64) -> Self {
        Self {
            keyword_extractor: KeywordExtractor::new(),
            min_support,
            min_age_days,
        }
    }

    /// Extract semantic facts from a set of memories
    ///
    /// This identifies recurring patterns and converts them to durable facts.
    /// Returns new facts and IDs of facts that were reinforced.
    pub fn consolidate(&self, memories: &[Memory]) -> ConsolidationResult {
        let mut result = ConsolidationResult {
            memories_processed: memories.len(),
            ..Default::default()
        };

        if memories.is_empty() {
            return result;
        }

        // Filter to memories old enough for consolidation
        let now = chrono::Utc::now();
        let eligible: Vec<&Memory> = memories
            .iter()
            .filter(|m| (now - m.created_at).num_days() >= self.min_age_days)
            .collect();

        if eligible.is_empty() {
            return result;
        }

        // Extract candidate facts from each memory
        let mut candidates: HashMap<String, Vec<(&Memory, f32)>> = HashMap::new();

        for memory in &eligible {
            let extracted = self.extract_fact_candidates(memory);
            for (pattern, confidence) in extracted {
                candidates
                    .entry(pattern)
                    .or_default()
                    .push((memory, confidence));
            }
        }

        // Convert patterns with sufficient support into facts
        for (pattern, sources) in candidates {
            if sources.len() >= self.min_support {
                let avg_confidence: f32 =
                    sources.iter().map(|(_, c)| c).sum::<f32>() / sources.len() as f32;

                let source_ids: Vec<MemoryId> = sources.iter().map(|(m, _)| m.id.clone()).collect();

                // Extract entities from the pattern
                let entities = self.keyword_extractor.extract(&pattern);

                let fact_type = self.classify_fact(&pattern);

                let fact = SemanticFact {
                    id: uuid::Uuid::new_v4().to_string(),
                    fact: pattern,
                    confidence: avg_confidence.min(1.0),
                    support_count: sources.len(),
                    source_memories: source_ids,
                    related_entities: entities,
                    created_at: now,
                    last_reinforced: now,
                    fact_type,
                };

                result.new_fact_ids.push(fact.id.clone());
                result.facts_extracted += 1;
            }
        }

        result
    }

    /// Extract fact candidates from a single memory
    fn extract_fact_candidates(&self, memory: &Memory) -> Vec<(String, f32)> {
        let mut candidates = Vec::new();
        let content = &memory.experience.content;

        // Extract based on experience type
        match memory.experience.experience_type {
            ExperienceType::Decision => {
                // Decision memories often contain procedures
                if let Some(fact) = self.extract_procedure(content) {
                    candidates.push((fact, memory.importance()));
                }
            }
            ExperienceType::Learning | ExperienceType::Discovery => {
                // Learning/Discovery often contain definitions or capabilities
                if let Some(fact) = self.extract_definition(content) {
                    candidates.push((fact, memory.importance() * 1.2));
                }
            }
            ExperienceType::Error => {
                // Errors often reveal patterns
                if let Some(fact) = self.extract_pattern(content) {
                    candidates.push((fact, memory.importance() * 1.1));
                }
            }
            ExperienceType::Conversation => {
                // Conversations may contain preferences
                if let Some(fact) = self.extract_preference(content) {
                    candidates.push((fact, memory.importance()));
                }
            }
            _ => {
                // Generic extraction for other types
                let keywords = self.keyword_extractor.extract(content);
                if keywords.len() >= 3 {
                    let fact = format!("involves: {}", keywords.join(", "));
                    candidates.push((fact, memory.importance() * 0.5));
                }
            }
        }

        // Also extract entity relationships
        if memory.experience.entities.len() >= 2 {
            let relationship = format!(
                "{} relates to {}",
                memory.experience.entities[0],
                memory.experience.entities[1..].join(", ")
            );
            candidates.push((relationship, memory.importance() * 0.8));
        }

        candidates
    }

    /// Extract a procedure from content (looks for action words)
    fn extract_procedure(&self, content: &str) -> Option<String> {
        let lower = content.to_lowercase();
        let action_markers = [
            "to ", "run ", "execute ", "use ", "call ", "invoke ", "create ", "build ", "deploy ",
        ];

        for marker in action_markers {
            if let Some(pos) = lower.find(marker) {
                // Extract the sentence containing this marker
                let start = content[..pos].rfind(|c| c == '.' || c == '!' || c == '?');
                let start = start.map(|i| i + 1).unwrap_or(0);

                let end = content[pos..].find(|c| c == '.' || c == '!' || c == '?');
                let end = end.map(|i| pos + i).unwrap_or(content.len());

                let sentence = content[start..end].trim();
                if sentence.len() > 10 && sentence.len() < 200 {
                    return Some(sentence.to_string());
                }
            }
        }
        None
    }

    /// Extract a definition from content
    fn extract_definition(&self, content: &str) -> Option<String> {
        let lower = content.to_lowercase();
        let def_markers = [" is ", " are ", " means ", " refers to ", " represents "];

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

    /// Extract a pattern from error content
    fn extract_pattern(&self, content: &str) -> Option<String> {
        let lower = content.to_lowercase();
        let pattern_markers = [
            "error",
            "failed",
            "crash",
            "bug",
            "issue",
            "problem",
            "exception",
        ];

        for marker in pattern_markers {
            if lower.contains(marker) {
                // Extract the key phrase around the error
                let keywords = self.keyword_extractor.extract(content);
                if keywords.len() >= 2 {
                    return Some(format!(
                        "{} can cause issues with {}",
                        keywords[0],
                        keywords[1..].join(", ")
                    ));
                }
            }
        }
        None
    }

    /// Extract a preference from conversation content
    fn extract_preference(&self, content: &str) -> Option<String> {
        let lower = content.to_lowercase();
        let pref_markers = [
            "prefer", "like", "want", "better", "should", "always", "never",
        ];

        for marker in pref_markers {
            if lower.contains(marker) {
                // Extract the preference statement
                let keywords = self.keyword_extractor.extract(content);
                if !keywords.is_empty() {
                    return Some(format!("preference: {}", keywords.join(", ")));
                }
            }
        }
        None
    }

    /// Classify what type of fact this is
    fn classify_fact(&self, pattern: &str) -> FactType {
        let lower = pattern.to_lowercase();

        if lower.starts_with("preference:") || lower.contains("prefer") || lower.contains("like") {
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

        // Add any new entities
        for entity in &memory.experience.entities {
            if !fact.related_entities.contains(entity) {
                fact.related_entities.push(entity.clone());
            }
        }
    }

    /// Check if a fact should decay (no reinforcement for too long)
    ///
    /// Returns true if the fact should be removed.
    /// Uses FACT_DECAY_BASE_DAYS (30) as base, extended by confidence and support count.
    pub fn should_decay_fact(&self, fact: &SemanticFact) -> bool {
        let now = chrono::Utc::now();
        let days_since_reinforcement = (now - fact.last_reinforced).num_days();

        // Facts with high confidence and support decay slower
        // Base: 30 days, + 7 days per support count, + confidence bonus
        let decay_threshold = FACT_DECAY_BASE_DAYS
            + (fact.confidence * 30.0) as i64
            + fact.support_count as i64 * FACT_DECAY_PER_SUPPORT_DAYS;

        days_since_reinforcement > decay_threshold
    }
}
