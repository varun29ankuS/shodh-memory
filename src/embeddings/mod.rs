//! Embedding generation module
//!
//! Provides semantic embedding generation for memory retrieval.
//! Uses ONNX Runtime with MiniLM-L6-v2 for 384-dimensional embeddings.
//!
//! # Features
//! - **Auto-download**: Model files downloaded on first use to ~/.cache/shodh-memory/
//! - **Circuit breaker**: Automatic fallback when ONNX service is degraded
//! - **Lazy loading**: Model loaded on first embed() call, not at startup
//!
//! # Configuration
//! - `SHODH_OFFLINE=true` - Disable auto-download
//! - `SHODH_LAZY_LOAD=false` - Load model at startup
//! - `SHODH_ONNX_THREADS=N` - Set ONNX thread count

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod circuit_breaker;
pub mod downloader;
pub mod minilm;
pub mod ner;

use anyhow::Result;

// Re-export downloader functions for convenience
pub use downloader::{
    are_models_downloaded, are_ner_models_downloaded, download_ner_models, ensure_downloaded,
    get_cache_dir, get_models_dir, get_ner_models_dir, get_onnx_runtime_path,
    is_onnx_runtime_downloaded, print_status,
};

// Re-export NER types
pub use ner::{NerConfig, NerEntity, NerEntityType, NeuralNer};

// Re-export circuit breaker types
pub use circuit_breaker::{
    CircuitBreakerConfig, CircuitBreakerMetrics, CircuitState, ResilientEmbedder,
};

/// Trait for embedding generation
pub trait Embedder: Send + Sync {
    /// Generate embedding for text
    fn encode(&self, text: &str) -> Result<Vec<f32>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Batch encode multiple texts (default: sequential)
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
}
