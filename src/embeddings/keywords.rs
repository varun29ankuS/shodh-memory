//! Statistical Keyword Extraction using YAKE
//!
//! Extracts salient keywords from text using statistical features:
//! - Position in text (earlier = more important)
//! - Word frequency and distribution across sentences
//! - Capitalization patterns
//! - Word length and structure
//!
//! Unlike NER which only extracts named entities (Person, Org, Location, Misc),
//! keyword extraction captures any semantically important terms including:
//! - Common nouns: "sunrise", "painting", "lake"
//! - Verbs/actions: "painted", "visited", "bought"
//! - Adjectives: "beautiful", "expensive", "favorite"
//!
//! This is critical for graph traversal in multi-hop reasoning where
//! query terms like "sunrise" need to match graph nodes but aren't
//! named entities detectable by NER.

use std::collections::HashSet;
use yake_rust::{get_n_best, Config, StopWords};

/// Configuration for keyword extraction
#[derive(Debug, Clone)]
pub struct KeywordConfig {
    /// Maximum number of keywords to extract
    pub max_keywords: usize,
    /// Maximum n-gram size (1=unigrams, 2=bigrams, 3=trigrams)
    pub ngrams: usize,
    /// Minimum keyword length
    pub min_length: usize,
    /// Language for stopwords
    pub language: String,
    /// Deduplication threshold (0.0-1.0, higher = stricter dedup)
    pub dedup_threshold: f64,
}

impl Default for KeywordConfig {
    fn default() -> Self {
        Self {
            max_keywords: 10,
            ngrams: 2,
            min_length: 3,
            language: "en".to_string(),
            dedup_threshold: 0.9,
        }
    }
}

/// A keyword extracted from text with its importance score
#[derive(Debug, Clone)]
pub struct Keyword {
    /// The keyword text (normalized)
    pub text: String,
    /// YAKE score (lower = more important, typically 0.0-1.0)
    pub score: f64,
    /// Normalized importance (0.0-1.0, higher = more important)
    pub importance: f32,
}

/// Keyword extractor using YAKE algorithm
pub struct KeywordExtractor {
    config: KeywordConfig,
    stopwords: StopWords,
}

impl KeywordExtractor {
    /// Create a new keyword extractor with default config
    pub fn new() -> Self {
        Self::with_config(KeywordConfig::default())
    }

    /// Create a new keyword extractor with custom config
    pub fn with_config(config: KeywordConfig) -> Self {
        // StopWords::predefined returns Option, fallback to empty set if not found
        let stopwords = StopWords::predefined(&config.language)
            .or_else(|| StopWords::predefined("en"))
            .unwrap_or_else(|| StopWords::custom(HashSet::new()));
        Self { config, stopwords }
    }

    /// Extract keywords from text
    pub fn extract(&self, text: &str) -> Vec<Keyword> {
        if text.trim().is_empty() {
            return Vec::new();
        }

        // Standard punctuation set
        let punctuation: HashSet<char> = [
            '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';',
            '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
        ]
        .into_iter()
        .collect();

        let yake_config = Config {
            ngrams: self.config.ngrams,
            punctuation,
            remove_duplicates: true,
            deduplication_threshold: self.config.dedup_threshold,
            minimum_chars: self.config.min_length,
            ..Config::default()
        };

        let results = get_n_best(
            self.config.max_keywords,
            text,
            &self.stopwords,
            &yake_config,
        );

        // Convert YAKE results to Keywords
        // YAKE score: lower = better, typically 0.0-0.5 for important keywords
        // We invert this to importance: higher = better
        let mut keywords: Vec<Keyword> = results
            .into_iter()
            .map(|item| {
                // Convert YAKE score (lower=better) to importance (higher=better)
                // Use sigmoid-like transformation: importance = 1 / (1 + score)
                let importance = (1.0 / (1.0 + item.score)) as f32;
                Keyword {
                    text: item.keyword, // Already lowercased
                    score: item.score,
                    importance,
                }
            })
            .collect();

        // Sort by importance descending
        keywords.sort_by(|a, b| b.importance.total_cmp(&a.importance));
        keywords
    }

    /// Extract keyword texts only (for graph node creation)
    pub fn extract_texts(&self, text: &str) -> Vec<String> {
        self.extract(text).into_iter().map(|k| k.text).collect()
    }

    /// Check if a word is a stop word (delegates to YAKE's language-aware stop word list)
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stopwords.contains(word)
    }

    /// Extract keywords with minimum importance threshold
    pub fn extract_filtered(&self, text: &str, min_importance: f32) -> Vec<Keyword> {
        self.extract(text)
            .into_iter()
            .filter(|k| k.importance >= min_importance)
            .collect()
    }
}

impl Default for KeywordExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_basic() {
        let extractor = KeywordExtractor::new();
        let text = "Caroline painted a beautiful sunrise over the lake yesterday morning.";
        let keywords = extractor.extract(text);

        // Should extract key terms
        assert!(!keywords.is_empty());

        // Check that we got some expected keywords
        let texts: Vec<&str> = keywords.iter().map(|k| k.text.as_str()).collect();
        assert!(
            texts.contains(&"sunrise") || texts.contains(&"beautiful sunrise"),
            "Should extract 'sunrise': {texts:?}"
        );
    }

    #[test]
    fn test_extract_texts() {
        let extractor = KeywordExtractor::new();
        let text = "The quick brown fox jumps over the lazy dog near the river.";
        let texts = extractor.extract_texts(text);

        assert!(!texts.is_empty());
        // All should be lowercase
        for t in &texts {
            assert_eq!(t.to_lowercase(), *t);
        }
    }

    #[test]
    fn test_empty_text() {
        let extractor = KeywordExtractor::new();
        let keywords = extractor.extract("");
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_importance_ordering() {
        let extractor = KeywordExtractor::new();
        let text =
            "Machine learning and artificial intelligence are transforming computer science.";
        let keywords = extractor.extract(text);

        // Should be sorted by importance (descending)
        for i in 1..keywords.len() {
            assert!(keywords[i - 1].importance >= keywords[i].importance);
        }
    }

    #[test]
    fn test_filter_by_importance() {
        let extractor = KeywordExtractor::new();
        let text = "The conference discussed various topics including climate change and renewable energy.";
        let filtered = extractor.extract_filtered(text, 0.5);

        for k in filtered {
            assert!(k.importance >= 0.5);
        }
    }

    #[test]
    fn test_custom_config() {
        let config = KeywordConfig {
            max_keywords: 5,
            ngrams: 3,
            min_length: 4,
            ..Default::default()
        };
        let extractor = KeywordExtractor::with_config(config);
        let text = "Natural language processing enables computers to understand human language.";
        let keywords = extractor.extract(text);

        assert!(keywords.len() <= 5);
        for k in &keywords {
            assert!(k.text.chars().count() >= 4);
        }
    }
}
