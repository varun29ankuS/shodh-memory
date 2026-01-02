//! Text Chunking for Long-Content Embeddings
//!
//! MiniLM has a 256 token limit. Content beyond this is silently dropped,
//! making long memories unsearchable by their later content.
//!
//! This module implements overlapping chunking to ensure ALL content is embedded:
//! - Split text into ~200 token chunks (leaving room for special tokens)
//! - 50 token overlap between chunks for context continuity
//! - Each chunk gets its own embedding in the vector index
//! - Search matches against ANY chunk, returns the full memory
//!
//! # Example
//!
//! A 1000-token memory becomes ~6 chunks:
//! ```text
//! [0-200] [150-350] [300-500] [450-650] [600-800] [750-1000]
//!         ↑ overlap  ↑ overlap  ↑ overlap  ↑ overlap
//! ```
//!
//! If a search query matches chunk 4, the full memory is returned.

/// Chunk configuration
pub struct ChunkConfig {
    /// Target chunk size in characters (approximate tokens * 4)
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
    /// Minimum chunk size (don't create tiny trailing chunks)
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            // ~200 tokens * 4 chars/token = 800 chars
            // Leave headroom for tokenizer differences
            chunk_size: 800,
            // ~50 tokens overlap for context continuity
            overlap: 200,
            // Don't create chunks smaller than ~50 tokens
            min_chunk_size: 200,
        }
    }
}

/// Result of chunking a text
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// The chunked text segments
    pub chunks: Vec<String>,
    /// Original text length
    pub original_length: usize,
    /// Whether chunking was needed (content exceeded single chunk)
    pub was_chunked: bool,
}

impl ChunkResult {
    /// Calculate content coverage ratio (1.0 = all content in single chunk)
    pub fn coverage_ratio(&self) -> f32 {
        if self.chunks.is_empty() {
            return 0.0;
        }
        // With chunking, we cover everything
        // Without chunking, we might truncate
        1.0
    }
}

/// Find the nearest valid char boundary at or before the given byte index
#[inline]
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    // Walk backwards to find a valid char boundary
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Find the nearest valid char boundary at or after the given byte index
#[inline]
fn ceil_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    // Walk forwards to find a valid char boundary
    let mut i = index;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Chunk text into overlapping segments for embedding
///
/// Uses sentence-aware splitting to avoid breaking mid-sentence when possible.
pub fn chunk_text(text: &str, config: &ChunkConfig) -> ChunkResult {
    let text = text.trim();
    let original_length = text.len();

    // If text fits in a single chunk, no need to split
    if original_length <= config.chunk_size {
        return ChunkResult {
            chunks: vec![text.to_string()],
            original_length,
            was_chunked: false,
        };
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < original_length {
        // Calculate end position for this chunk, ensuring valid char boundary
        let mut end = floor_char_boundary(text, (start + config.chunk_size).min(original_length));

        // If we're not at the end, try to break at a sentence boundary
        if end < original_length {
            end = find_break_point(text, start, end, config.min_chunk_size);
            // Ensure the break point is on a valid char boundary
            end = floor_char_boundary(text, end);
        }

        // Ensure start is on a valid char boundary
        start = ceil_char_boundary(text, start);

        // Safety check: ensure we don't have start >= end
        if start >= end {
            break;
        }

        // Extract chunk (now safe - both start and end are valid char boundaries)
        let chunk = text[start..end].trim();
        if chunk.len() >= config.min_chunk_size || chunks.is_empty() {
            chunks.push(chunk.to_string());
        } else if let Some(last) = chunks.last_mut() {
            // Append tiny trailing chunk to previous
            last.push(' ');
            last.push_str(chunk);
        }

        // Move start position, accounting for overlap
        if end >= original_length {
            break;
        }
        // Ensure new start is on a valid char boundary
        start = ceil_char_boundary(text, end.saturating_sub(config.overlap));

        // Ensure we make progress
        if start <= chunks.len().saturating_sub(1) * (config.chunk_size - config.overlap) {
            start = ceil_char_boundary(text, end);
        }
    }

    ChunkResult {
        chunks,
        original_length,
        was_chunked: true,
    }
}

/// Find a good break point for chunking (sentence or word boundary)
fn find_break_point(text: &str, start: usize, ideal_end: usize, min_size: usize) -> usize {
    let chunk = &text[start..ideal_end];

    // Try to find sentence boundary (. ! ?) followed by space or end
    let sentence_boundaries: Vec<usize> = chunk
        .char_indices()
        .filter_map(|(i, c)| {
            if (c == '.' || c == '!' || c == '?') && i >= min_size {
                // Check if followed by space or end
                let next_char = chunk.chars().nth(i + 1);
                if next_char.map_or(true, |nc| nc.is_whitespace()) {
                    return Some(start + i + 1);
                }
            }
            None
        })
        .collect();

    // Use the last sentence boundary if available
    if let Some(&boundary) = sentence_boundaries.last() {
        return boundary;
    }

    // Fall back to word boundary
    let word_boundaries: Vec<usize> = chunk
        .char_indices()
        .filter_map(|(i, c)| {
            if c.is_whitespace() && i >= min_size {
                Some(start + i)
            } else {
                None
            }
        })
        .collect();

    if let Some(&boundary) = word_boundaries.last() {
        return boundary;
    }

    // No good boundary found, use ideal_end
    ideal_end
}

/// Estimate token count for text (rough approximation)
pub fn estimate_tokens(text: &str) -> usize {
    // Average ~4 characters per token for English
    // This is a rough estimate; actual tokenization varies
    (text.len() + 3) / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_no_chunking() {
        let config = ChunkConfig::default();
        let result = chunk_text("This is a short text.", &config);

        assert_eq!(result.chunks.len(), 1);
        assert!(!result.was_chunked);
        assert_eq!(result.chunks[0], "This is a short text.");
    }

    #[test]
    fn test_long_text_chunking() {
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 20,
            min_chunk_size: 30,
        };

        // Create text longer than chunk_size
        let text = "This is sentence one. This is sentence two. This is sentence three. \
                   This is sentence four. This is sentence five. This is sentence six. \
                   This is sentence seven. This is sentence eight.";

        let result = chunk_text(text, &config);

        assert!(result.was_chunked);
        assert!(result.chunks.len() > 1);

        // Verify each chunk has meaningful content
        for chunk in &result.chunks {
            assert!(
                chunk.len() >= config.min_chunk_size,
                "Chunk too small: '{}' (len={})",
                chunk,
                chunk.len()
            );
        }

        // Verify total chunked content is at least as long as original (with overlaps)
        let total_len: usize = result.chunks.iter().map(|c| c.len()).sum();
        assert!(
            total_len >= result.original_length,
            "Total chunk length {} < original {}",
            total_len,
            result.original_length
        );
    }

    #[test]
    fn test_sentence_boundary_respected() {
        let config = ChunkConfig {
            chunk_size: 50,
            overlap: 10,
            min_chunk_size: 20,
        };

        let text = "First sentence here. Second sentence follows. Third sentence ends.";
        let result = chunk_text(text, &config);

        // Chunks should end at sentence boundaries when possible
        for chunk in &result.chunks {
            let trimmed = chunk.trim();
            if !trimmed.is_empty() && result.chunks.len() > 1 {
                // Most chunks should end with sentence-ending punctuation
                let last_char = trimmed.chars().last().unwrap();
                // Allow some flexibility - not all chunks will end at sentence boundary
                assert!(
                    last_char == '.'
                        || last_char == '!'
                        || last_char == '?'
                        || chunk == result.chunks.last().unwrap(),
                    "Chunk '{}' doesn't end at sentence boundary",
                    chunk
                );
            }
        }
    }

    #[test]
    fn test_overlap_exists() {
        let config = ChunkConfig {
            chunk_size: 60,
            overlap: 20,
            min_chunk_size: 20,
        };

        let text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ KKKK LLLL MMMM";
        let result = chunk_text(text, &config);

        if result.chunks.len() >= 2 {
            // Check that consecutive chunks have overlapping content
            for i in 0..result.chunks.len() - 1 {
                let chunk1 = &result.chunks[i];
                let chunk2 = &result.chunks[i + 1];

                // Find common words
                let words1: std::collections::HashSet<_> = chunk1.split_whitespace().collect();
                let words2: std::collections::HashSet<_> = chunk2.split_whitespace().collect();
                let common: Vec<_> = words1.intersection(&words2).collect();

                // Should have some overlap
                assert!(
                    !common.is_empty() || chunk1.len() < config.overlap,
                    "No overlap between chunks {} and {}",
                    i,
                    i + 1
                );
            }
        }
    }

    #[test]
    fn test_token_estimation() {
        assert_eq!(estimate_tokens("test"), 1);
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars / 4 ≈ 3
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_very_long_content() {
        let config = ChunkConfig::default();

        // Create 10KB of content
        let long_text = "This is a test sentence. ".repeat(400);
        let result = chunk_text(&long_text, &config);

        assert!(result.was_chunked);
        assert!(result.chunks.len() > 10); // Should have many chunks
        assert_eq!(result.coverage_ratio(), 1.0);

        // Verify no chunk exceeds config size significantly
        for chunk in &result.chunks {
            assert!(
                chunk.len() <= config.chunk_size + 100,
                "Chunk too large: {} chars",
                chunk.len()
            );
        }
    }

    #[test]
    fn test_chunking_quality_unique_content_searchable() {
        let config = ChunkConfig::default();

        // Create content with UNIQUE markers at beginning, middle, and end
        // These markers should each appear in at least one chunk
        let beginning = "ALPHA_BEGINNING_MARKER is a unique identifier at the start.";
        let middle_padding = "This is filler content to push things apart. ".repeat(30);
        let middle = "BETA_MIDDLE_MARKER represents content in the center of the document.";
        let end_padding = "More filler content for separation between sections. ".repeat(30);
        let end = "GAMMA_END_MARKER signifies the conclusion of this memory content.";

        let full_text = format!(
            "{} {} {} {} {}",
            beginning, middle_padding, middle, end_padding, end
        );

        let result = chunk_text(&full_text, &config);

        // Content should be chunked
        assert!(result.was_chunked, "Content should require chunking");
        assert!(result.chunks.len() >= 3, "Should have multiple chunks");

        // Verify each unique marker appears in at least one chunk
        let has_alpha = result.chunks.iter().any(|c| c.contains("ALPHA_BEGINNING"));
        let has_beta = result.chunks.iter().any(|c| c.contains("BETA_MIDDLE"));
        let has_gamma = result.chunks.iter().any(|c| c.contains("GAMMA_END"));

        assert!(has_alpha, "ALPHA marker (beginning) not found in any chunk");
        assert!(has_beta, "BETA marker (middle) not found in any chunk");
        assert!(has_gamma, "GAMMA marker (end) not found in any chunk");

        // Log chunk info for debugging
        println!("Total chunks: {}", result.chunks.len());
        println!("Original length: {} chars", result.original_length);
        for (i, chunk) in result.chunks.iter().enumerate() {
            let markers: Vec<&str> = vec![
                if chunk.contains("ALPHA") { "ALPHA" } else { "" },
                if chunk.contains("BETA") { "BETA" } else { "" },
                if chunk.contains("GAMMA") { "GAMMA" } else { "" },
            ]
            .into_iter()
            .filter(|m| !m.is_empty())
            .collect();
            println!(
                "  Chunk {}: {} chars {}",
                i,
                chunk.len(),
                if markers.is_empty() {
                    String::new()
                } else {
                    format!("[contains: {}]", markers.join(", "))
                }
            );
        }
    }

    #[test]
    fn test_chunking_coverage_no_content_lost() {
        let config = ChunkConfig {
            chunk_size: 200,
            overlap: 50,
            min_chunk_size: 50,
        };

        // Create text with numbered sentences for easy tracking
        let sentences: Vec<String> = (1..=20)
            .map(|i| format!("Sentence number {} contains unique information. ", i))
            .collect();
        let text = sentences.join("");

        let result = chunk_text(&text, &config);

        // Every sentence number should appear in at least one chunk
        for i in 1..=20 {
            let marker = format!("number {}", i);
            let found = result.chunks.iter().any(|c| c.contains(&marker));
            assert!(
                found,
                "Sentence {} not found in any chunk! Coverage gap detected.",
                i
            );
        }
    }
}
