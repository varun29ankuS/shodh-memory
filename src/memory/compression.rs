//! Compression pipeline for memory optimization

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use lz4;
use base64::{Engine as _, engine::general_purpose};

use super::types::*;

/// Maximum decompressed size to prevent DoS via malicious payloads (10MB)
const MAX_DECOMPRESSED_SIZE: i32 = 10 * 1024 * 1024;

/// Compression strategy for memories
#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    None,
    Lz4,              // Fast compression
    Summarization,    // Semantic compression
    Hybrid,           // Combination of methods
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
        // High importance memories get lighter compression
        if memory.importance() > 0.8 {
            return CompressionStrategy::Lz4;
        }

        // Frequently accessed memories stay uncompressed
        if memory.access_count() > 10 {
            return CompressionStrategy::None;
        }

        // Old, low-importance memories get aggressive compression
        let age = chrono::Utc::now() - memory.created_at;
        if age.num_days() > 30 && memory.importance() < 0.5 {
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
        compressed_memory.experience.metadata.insert(
            "compressed_data".to_string(),
            compressed_b64
        );
        compressed_memory.experience.metadata.insert(
            "compression_ratio".to_string(),
            compression_ratio.to_string()
        );
        compressed_memory.experience.metadata.insert(
            "compression_strategy".to_string(),
            "lz4".to_string()
        );

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
        compressed_memory.experience.metadata.insert(
            "keywords".to_string(),
            keywords.join(",")
        );
        compressed_memory.experience.metadata.insert(
            "compression_strategy".to_string(),
            "semantic".to_string()
        );
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
    pub fn decompress(&self, memory: &Memory) -> Result<Memory> {
        if !memory.compressed {
            return Ok(memory.clone());
        }

        let strategy = memory.experience.metadata
            .get("compression_strategy")
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        match strategy {
            "lz4" => self.decompress_lz4(memory),
            "semantic" => {
                // Semantic compression is lossy, can't fully decompress
                Ok(memory.clone())
            },
            _ => Ok(memory.clone())
        }
    }

    /// Decompress LZ4 compressed memory
    fn decompress_lz4(&self, memory: &Memory) -> Result<Memory> {
        if let Some(compressed_b64) = memory.experience.metadata.get("compressed_data") {
            let compressed = general_purpose::STANDARD.decode(compressed_b64)?;

            // Limit decompression size to prevent DoS attacks
            let decompressed = lz4::block::decompress(&compressed, Some(MAX_DECOMPRESSED_SIZE))?;

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
            let clean_word = word.to_lowercase()
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

        keywords.into_iter()
            .take(10)
            .map(|(word, _)| word)
            .collect()
    }

    fn load_stop_words() -> HashSet<String> {
        // Common English stop words
        let words = vec![
            "the", "is", "at", "which", "on", "and", "a", "an", "as", "are",
            "was", "were", "been", "be", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "with", "by", "from", "about", "into", "through", "during",
            "before", "after", "above", "below", "up", "down", "out", "off",
            "over", "under", "again", "further", "then", "once", "there",
            "these", "those", "this", "that", "it", "its", "what", "which",
            "who", "whom", "whose", "where", "when", "why", "how", "all",
            "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "but", "or", "if"
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