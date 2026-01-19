//! Modular Query Parsing System
//!
//! Provides a trait-based abstraction for query parsing, allowing easy swapping
//! between rule-based and LLM-based implementations.
//!
//! # Architecture
//! ```text
//! Query → QueryParser (trait) → ParsedQuery
//!              ↓
//!     ┌────────┴────────┐
//!     │                 │
//! RuleBasedParser   LlmParser
//! (YAKE/regex)      (Qwen 1.5B)
//! ```
//!
//! # Usage
//! ```rust,ignore
//! let parser = create_parser(ParserConfig::default());
//! let parsed = parser.parse("When did Melanie paint a sunrise?", Some(conv_date))?;
//! ```

mod parser_trait;
mod rule_based;
mod llm_parser;

pub use parser_trait::*;
pub use rule_based::RuleBasedParser;
pub use llm_parser::{ApiType, LlmParser};

use std::sync::Arc;

/// Parser implementation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParserType {
    /// Rule-based parsing using YAKE, regex, and heuristics (default)
    #[default]
    RuleBased,
    /// LLM-based parsing using Qwen 1.5B or similar
    Llm,
}

/// Configuration for the query parser
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Which parser implementation to use
    pub parser_type: ParserType,
    /// Path to LLM model (only used if parser_type is Llm)
    pub llm_model_path: Option<String>,
    /// Number of threads for LLM inference
    pub llm_threads: usize,
    /// Context size for LLM
    pub llm_context_size: usize,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            parser_type: ParserType::RuleBased,
            llm_model_path: None,
            llm_threads: 4,
            llm_context_size: 2048,
        }
    }
}

impl ParserConfig {
    /// Create config for rule-based parser
    pub fn rule_based() -> Self {
        Self::default()
    }

    /// Create config for LLM parser
    pub fn llm(model_path: impl Into<String>) -> Self {
        Self {
            parser_type: ParserType::Llm,
            llm_model_path: Some(model_path.into()),
            ..Default::default()
        }
    }
}

/// Create a parser based on configuration
pub fn create_parser(config: ParserConfig) -> Arc<dyn QueryParser> {
    match config.parser_type {
        ParserType::RuleBased => Arc::new(RuleBasedParser::new()),
        #[cfg(feature = "llm-parser")]
        ParserType::Llm => {
            let model_path = config
                .llm_model_path
                .expect("LLM model path required for LLM parser");
            Arc::new(
                LlmParser::new(&model_path, config.llm_threads, config.llm_context_size)
                    .expect("Failed to load LLM model"),
            )
        }
        #[cfg(not(feature = "llm-parser"))]
        ParserType::Llm => {
            tracing::warn!("LLM parser requested but 'llm-parser' feature not enabled, falling back to rule-based");
            Arc::new(RuleBasedParser::new())
        }
    }
}
