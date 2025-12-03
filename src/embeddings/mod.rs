//! Embedding generation module
//!
//! Provides semantic embedding generation for memory retrieval.
//! Uses ONNX Runtime with MiniLM-L6-v2 for 384-dimensional embeddings.
//!
//! Auto-download feature:
//! - Model files are downloaded on first use to ~/.cache/shodh-memory/
//! - ONNX Runtime is downloaded if ORT_DYLIB_PATH is not set
//! - Set SHODH_OFFLINE=true to disable auto-download

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod minilm;
pub mod downloader;

use anyhow::Result;

// Re-export downloader functions for convenience
pub use downloader::{
    ensure_downloaded,
    are_models_downloaded,
    is_onnx_runtime_downloaded,
    get_cache_dir,
    get_models_dir,
    get_onnx_runtime_path,
    print_status,
};

/// Trait for embedding generation
pub trait Embedder: Send + Sync {
    /// Generate embedding for text
    fn encode(&self, text: &str) -> Result<Vec<f32>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Batch encode multiple texts
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter()
            .map(|text| self.encode(text))
            .collect()
    }
}
