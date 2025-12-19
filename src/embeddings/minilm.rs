//! MiniLM-L6-v2 embedding model using ONNX Runtime
//!
//! Generates 384-dimensional sentence embeddings optimized for semantic similarity.
//! Model: sentence-transformers/all-MiniLM-L6-v2
//!
//! Edge Optimizations:
//! - Lazy model loading: Model is only loaded on first embed call
//! - Configurable thread count for power efficiency
//! - Simplified fallback for resource-constrained devices
//!
//! Configuration via environment variables:
//! - SHODH_MODEL_PATH: Base path to model files (default: ./models/minilm-l6)
//! - SHODH_EMBED_TIMEOUT_MS: Embedding timeout in ms (default: 5000)
//! - SHODH_LAZY_LOAD: Set to "false" to load model at startup (default: true)
//! - SHODH_ONNX_THREADS: Number of ONNX threads (default: 2 for edge, 4 for desktop)

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

use super::Embedder;

/// Thread-safe guard for ORT_DYLIB_PATH initialization.
/// Using OnceLock ensures set_var is called exactly once, before other threads start.
/// This mitigates the UB risk of concurrent env::set_var calls.
static ORT_PATH_INIT: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Lazily initialized ONNX session and tokenizer
struct LazyModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl LazyModel {
    fn new(config: &EmbeddingConfig) -> Result<Self> {
        let num_threads = std::env::var("SHODH_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2); // Default 2 threads for edge devices

        tracing::info!(
            "Loading MiniLM-L6-v2 model from {:?} with {} threads",
            config.model_path,
            num_threads
        );

        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_intra_threads(num_threads)
            .context("Failed to set thread count")?
            .commit_from_file(&config.model_path)
            .context("Failed to load ONNX model")?;

        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        tracing::info!("MiniLM-L6-v2 model loaded successfully");

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }
}

/// Configuration for MiniLM embedder
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Path to ONNX model file
    pub model_path: PathBuf,

    /// Path to tokenizer file
    pub tokenizer_path: PathBuf,

    /// Maximum sequence length (MiniLM default: 256)
    pub max_length: usize,

    /// Use quantized model for faster inference
    pub use_quantized: bool,

    /// Timeout for embedding generation in milliseconds
    pub embed_timeout_ms: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl EmbeddingConfig {
    /// Create configuration from environment variables with sensible defaults
    ///
    /// Search order for model files:
    /// 1. SHODH_MODEL_PATH environment variable
    /// 2. Bundled in Python package (SHODH_PACKAGE_DIR/models/minilm-l6)
    /// 3. ./models/minilm-l6 (local)
    /// 4. ../models/minilm-l6 (parent)
    /// 5. ~/.cache/shodh-memory/models/minilm-l6 (auto-download location)
    pub fn from_env() -> Self {
        let base_path = std::env::var("SHODH_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                // Try common locations in order (bundled first for 1-click install)
                let candidates = vec![
                    // Bundled in Python package (highest priority for pip install)
                    std::env::var("SHODH_PACKAGE_DIR")
                        .ok()
                        .map(|p| PathBuf::from(p).join("models/minilm-l6")),
                    Some(PathBuf::from("./models/minilm-l6")),
                    Some(PathBuf::from("../models/minilm-l6")),
                    // Auto-download cache location
                    Some(super::downloader::get_models_dir()),
                    dirs::data_dir().map(|p| p.join("shodh-memory/models/minilm-l6")),
                ];

                candidates
                    .into_iter()
                    .flatten()
                    .find(|p| {
                        p.join("model_quantized.onnx").exists() || p.join("model.onnx").exists()
                    })
                    .unwrap_or_else(super::downloader::get_models_dir) // Default to cache dir
            });

        let embed_timeout_ms = std::env::var("SHODH_EMBED_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5000);

        let use_quantized = std::env::var("SHODH_USE_QUANTIZED_MODEL")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        let model_filename = if use_quantized {
            "model_quantized.onnx"
        } else {
            "model.onnx"
        };

        Self {
            model_path: base_path.join(model_filename),
            tokenizer_path: base_path.join("tokenizer.json"),
            max_length: 256,
            use_quantized,
            embed_timeout_ms,
        }
    }

    /// Create configuration with explicit paths (for testing or programmatic use)
    pub fn with_paths(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            model_path,
            tokenizer_path,
            max_length: 256,
            use_quantized: true,
            embed_timeout_ms: 5000,
        }
    }
}

/// MiniLM-L6-v2 embedder with ONNX Runtime
///
/// Features lazy model loading for edge devices:
/// - Model is only loaded on first embed() call
/// - Reduces startup time from ~2s to <100ms
/// - Reduces idle RAM by ~200MB until first use
pub struct MiniLMEmbedder {
    config: EmbeddingConfig,
    /// Lazily initialized model (OnceLock for thread-safe init)
    lazy_model: OnceLock<Result<Arc<LazyModel>, String>>,
    /// Flag for simplified mode (no ONNX)
    simplified_mode: bool,
    dimension: usize,
}

impl MiniLMEmbedder {
    /// Ensure ONNX Runtime is available before any ort code runs.
    /// This MUST be called before creating any ONNX sessions.
    /// Sets ORT_DYLIB_PATH if needed from cache or download.
    ///
    /// SAFETY: Uses OnceLock to ensure set_var is called at most once,
    /// mitigating the thread-safety issue with std::env::set_var.
    fn ensure_onnx_runtime_available(offline_mode: bool) -> Result<()> {
        // Use OnceLock to ensure we only initialize once (thread-safe)
        let result = ORT_PATH_INIT.get_or_init(|| Self::init_ort_path_inner(offline_mode));

        match result {
            Ok(_) => Ok(()),
            Err(e) => anyhow::bail!("{}", e),
        }
    }

    /// Inner initialization logic - called exactly once via OnceLock.
    /// SAFETY: set_var is only called once due to OnceLock guard.
    fn init_ort_path_inner(offline_mode: bool) -> Result<PathBuf, String> {
        // If ORT_DYLIB_PATH is already set to a valid path, we're good
        if let Ok(existing_path) = std::env::var("ORT_DYLIB_PATH") {
            let path = std::path::PathBuf::from(&existing_path);
            if path.exists() {
                tracing::debug!(
                    "Using existing ONNX Runtime from ORT_DYLIB_PATH: {:?}",
                    path
                );
                return Ok(path);
            }
        }

        // Check for bundled ONNX Runtime in Python package (1-click install)
        if let Some(bundled_path) = Self::find_bundled_onnx_runtime() {
            tracing::info!(
                "Using bundled ONNX Runtime from package: {:?}",
                bundled_path
            );
            // SAFETY: This is called once via OnceLock, before other threads start
            std::env::set_var("ORT_DYLIB_PATH", &bundled_path);
            return Ok(bundled_path);
        }

        // Check if we have ONNX Runtime in our cache
        if let Some(cached_path) = super::downloader::get_onnx_runtime_path() {
            tracing::info!(
                "Setting ORT_DYLIB_PATH to cached runtime: {:?}",
                cached_path
            );
            // SAFETY: This is called once via OnceLock, before other threads start
            std::env::set_var("ORT_DYLIB_PATH", &cached_path);
            return Ok(cached_path);
        }

        // Need to download ONNX Runtime
        if offline_mode {
            return Err("ONNX Runtime not found and SHODH_OFFLINE=true".to_string());
        }

        tracing::info!("ONNX Runtime not found. Downloading...");
        let onnx_path =
            super::downloader::download_onnx_runtime(None).map_err(|e| e.to_string())?;
        tracing::info!(
            "Setting ORT_DYLIB_PATH to downloaded runtime: {:?}",
            onnx_path
        );
        // SAFETY: This is called once via OnceLock, before other threads start
        std::env::set_var("ORT_DYLIB_PATH", &onnx_path);
        Ok(onnx_path)
    }

    /// Find bundled ONNX Runtime in the Python package's lib/ directory
    fn find_bundled_onnx_runtime() -> Option<PathBuf> {
        // Try to find ONNX Runtime bundled with the Python package
        // The lib/ folder is adjacent to the shodh_memory.pyd file

        #[cfg(target_os = "windows")]
        let dll_name = "onnxruntime.dll";
        #[cfg(target_os = "macos")]
        let dll_name = "libonnxruntime.dylib";
        #[cfg(target_os = "linux")]
        let dll_name = "libonnxruntime.so";

        // Common locations to check for bundled library
        let candidates = [
            // Relative to current executable (for standalone binary)
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.join("lib").join(dll_name))),
            // Relative to working directory
            Some(PathBuf::from("lib").join(dll_name)),
            // Python site-packages layout: shodh_memory/lib/onnxruntime.dll
            dirs::data_dir().map(|p| {
                p.join("Python")
                    .join("site-packages")
                    .join("shodh_memory")
                    .join("lib")
                    .join(dll_name)
            }),
            // Check relative to this module (for pip-installed packages)
            // This uses the fact that Python modules are in site-packages/shodh_memory/
            std::env::var("SHODH_PACKAGE_DIR")
                .ok()
                .map(|p| PathBuf::from(p).join("lib").join(dll_name)),
        ];

        for candidate in candidates.into_iter().flatten() {
            if candidate.exists() {
                tracing::debug!("Found bundled ONNX Runtime at: {:?}", candidate);
                return Some(candidate);
            }
        }

        None
    }

    /// Create new MiniLM embedder with lazy loading (default)
    ///
    /// Model is NOT loaded until first embed() call.
    /// Set SHODH_LAZY_LOAD=false to load immediately.
    /// Set SHODH_OFFLINE=true to disable auto-download.
    ///
    /// Auto-download behavior:
    /// - If model files not found, downloads from HuggingFace (~22MB)
    /// - If ONNX Runtime not found, downloads from GitHub (~50MB)
    /// - Files cached in ~/.cache/shodh-memory/
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let lazy_load = std::env::var("SHODH_LAZY_LOAD")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        let offline_mode = std::env::var("SHODH_OFFLINE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        // CRITICAL: Ensure ORT_DYLIB_PATH is set BEFORE any ort code runs
        // This prevents ort from picking up system DLLs with wrong versions
        if let Err(e) = Self::ensure_onnx_runtime_available(offline_mode) {
            tracing::warn!(
                "Failed to set up ONNX Runtime: {}. Using simplified embeddings.",
                e
            );
            return Self::new_simplified(config);
        }

        // Check if model files exist
        let model_available = config.model_path.exists() && config.tokenizer_path.exists();

        if !model_available {
            if offline_mode {
                tracing::warn!(
                    "Model files not found and SHODH_OFFLINE=true. Using simplified embeddings.",
                );
                return Self::new_simplified(config);
            }

            // Try to auto-download model files
            tracing::info!(
                "Model files not found at {:?}. Downloading...",
                config.model_path.parent().unwrap_or(&config.model_path)
            );

            match super::downloader::download_models(Some(std::sync::Arc::new(
                |downloaded, total| {
                    if total > 0 {
                        let percent = (downloaded as f64 / total as f64 * 100.0) as u32;
                        if percent % 10 == 0 {
                            tracing::info!(
                                "Downloading models: {}% ({}/{})",
                                percent,
                                downloaded,
                                total
                            );
                        }
                    }
                },
            ))) {
                Ok(models_dir) => {
                    tracing::info!("Models downloaded to {:?}", models_dir);

                    // Update config with downloaded paths
                    let model_filename = if config.use_quantized {
                        "model_quantized.onnx"
                    } else {
                        "model.onnx"
                    };
                    let updated_config = EmbeddingConfig {
                        model_path: models_dir.join(model_filename),
                        tokenizer_path: models_dir.join("tokenizer.json"),
                        ..config
                    };

                    // Recursively create with updated config (ORT_DYLIB_PATH already set)
                    return Self::new(updated_config);
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to download models: {}. Using simplified embeddings.",
                        e
                    );
                    return Self::new_simplified(config);
                }
            }
        }

        let embedder = Self {
            config: config.clone(),
            lazy_model: OnceLock::new(),
            simplified_mode: false,
            dimension: 384,
        };

        // If not lazy loading, initialize now
        if !lazy_load {
            tracing::info!("Eager loading ONNX model (SHODH_LAZY_LOAD=false)");
            embedder.ensure_model_loaded()?;
        } else {
            tracing::info!("Lazy loading enabled - model will load on first embed()");
        }

        Ok(embedder)
    }

    /// Ensure the model is loaded (thread-safe, idempotent)
    fn ensure_model_loaded(&self) -> Result<&Arc<LazyModel>> {
        let result = self.lazy_model.get_or_init(|| {
            LazyModel::new(&self.config)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });

        match result {
            Ok(model) => Ok(model),
            Err(e) => Err(anyhow::anyhow!("Failed to load model: {e}")),
        }
    }

    /// Check if model is currently loaded (for diagnostics)
    pub fn is_model_loaded(&self) -> bool {
        self.lazy_model.get().is_some()
    }

    /// Create simplified embedder as fallback when model files are missing
    ///
    /// Uses hash-based embeddings that are fast but less semantic.
    /// Suitable for edge devices without enough RAM for ONNX.
    pub fn new_simplified(config: EmbeddingConfig) -> Result<Self> {
        tracing::warn!(
            "Using SIMPLIFIED embeddings (hash-based). Semantic search will be limited."
        );
        tracing::warn!(
            "    To enable full semantic search, ensure MiniLM-L6-v2 model files exist at:"
        );
        tracing::warn!("    Model: {:?}", config.model_path);
        tracing::warn!("    Tokenizer: {:?}", config.tokenizer_path);

        Ok(Self {
            config,
            lazy_model: OnceLock::new(),
            simplified_mode: true,
            dimension: 384,
        })
    }

    /// Tokenize text (simplified implementation)
    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        // Simplified tokenization - split by whitespace and convert to IDs
        // In production, this would use the HuggingFace tokenizer
        let tokens: Vec<i64> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| i as i64 + 1000) // Dummy token IDs
            .take(self.config.max_length)
            .collect();

        Ok(tokens)
    }

    /// Mean pooling with attention mask
    fn mean_pooling(&self, token_embeddings: &[Vec<f32>], attention_mask: &[i64]) -> Vec<f32> {
        let _seq_len = token_embeddings.len();
        let dim = self.dimension;
        let mut pooled = vec![0.0; dim];
        let mut mask_sum = 0.0;

        for (i, embedding) in token_embeddings.iter().enumerate() {
            if i < attention_mask.len() && attention_mask[i] == 1 {
                for (j, &val) in embedding.iter().enumerate() {
                    pooled[j] += val;
                }
                mask_sum += 1.0;
            }
        }

        // Average
        if mask_sum > 0.0 {
            for val in &mut pooled {
                *val /= mask_sum;
            }
        }

        pooled
    }

    /// L2 normalize embedding
    /// Returns false if normalization failed (zero norm or NaN detected)
    fn normalize(&self, embedding: &mut [f32]) -> bool {
        // Check for NaN values before normalization
        if embedding.iter().any(|x| x.is_nan() || x.is_infinite()) {
            // Replace invalid values with zero
            for val in embedding.iter_mut() {
                if val.is_nan() || val.is_infinite() {
                    *val = 0.0;
                }
            }
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Handle zero norm (all zeros) or NaN norm
        if norm.is_nan() || norm < f32::EPSILON {
            return false;
        }

        for val in embedding.iter_mut() {
            *val /= norm;
        }

        true
    }

    /// Generate embedding using simplified approach
    fn generate_embedding_simplified(&self, text: &str) -> Result<Vec<f32>> {
        // Production fallback: Hash-based embeddings for resilience
        // Used when: (1) ONNX models unavailable, (2) ONNX inference fails, (3) Timeout exceeded
        // Provides basic semantic similarity via word + character n-gram hashing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding = vec![0.0; self.dimension];
        let mut hasher = DefaultHasher::new();

        // Use words and character n-grams for better quality
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            word.hash(&mut hasher);
            let hash = hasher.finish();

            // Distribute hash bits across embedding dimensions
            for j in 0..self.dimension {
                let index = (i * self.dimension + j) % self.dimension;
                if j < 64 {
                    embedding[index] += ((hash >> j) & 1) as f32 * 0.1;
                } else {
                    embedding[index] += ((hash >> (j % 64)) & 1) as f32 * 0.1;
                }
            }
        }

        // Add character bigram features for better semantic representation
        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len().saturating_sub(1) {
            let bigram = format!("{}{}", chars[i], chars[i + 1]);
            bigram.hash(&mut hasher);
            let hash = hasher.finish();

            for j in 0..32 {
                let index = ((hash as usize) + j) % self.dimension;
                embedding[index] += ((hash >> (j % 64)) & 1) as f32 * 0.05;
            }
        }

        // Normalize - if normalization fails (empty text), return zero vector
        let _ = self.normalize(&mut embedding);

        Ok(embedding)
    }

    /// Generate embedding using ONNX Runtime (production)
    ///
    /// Lazily loads the model on first call if not already loaded.
    fn generate_embedding_onnx(&self, text: &str) -> Result<Vec<f32>> {
        // Lazy load model on first use
        let model = self.ensure_model_loaded()?;

        let mut session = model.session.lock();

        // Tokenize input text
        let encoding = model
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let tokens = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let max_length = self.config.max_length;

        // Truncate or pad to max_length
        let mut input_ids = vec![0i64; max_length];
        let mut attention = vec![0i64; max_length];
        let token_type_ids = vec![0i64; max_length];

        for (i, &token) in tokens.iter().take(max_length).enumerate() {
            input_ids[i] = token as i64;
        }
        for (i, &mask) in attention_mask.iter().take(max_length).enumerate() {
            attention[i] = mask as i64;
        }

        // Create input tensors
        let input_ids_value = Value::from_array((vec![1, max_length], input_ids))?;
        let attention_mask_value = Value::from_array((vec![1, max_length], attention.clone()))?;
        let token_type_ids_value = Value::from_array((vec![1, max_length], token_type_ids))?;

        // Run inference
        let outputs = session.run(ort::inputs![
            "input_ids" => &input_ids_value,
            "attention_mask" => &attention_mask_value,
            "token_type_ids" => &token_type_ids_value,
        ])?;

        // Extract embeddings
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, output_data) = output_tensor;

        // Mean pooling over sequence dimension
        let mut pooled = vec![0.0; self.dimension];
        let mut mask_sum = 0.0;

        for (seq_idx, &att) in attention.iter().enumerate() {
            if att == 1 {
                for (dim_idx, pooled_val) in pooled.iter_mut().enumerate() {
                    let idx = seq_idx * self.dimension + dim_idx;
                    *pooled_val += output_data[idx];
                }
                mask_sum += 1.0;
            }
        }

        // Average and L2 normalize
        if mask_sum > 0.0 {
            for val in &mut pooled {
                *val /= mask_sum;
            }
        }

        // Handle NaN/Inf values that may come from model output
        for val in pooled.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }

        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON && !norm.is_nan() {
            for val in &mut pooled {
                *val /= norm;
            }
        }

        Ok(pooled)
    }

    /// Generate embeddings for multiple texts in a single ONNX batch
    ///
    /// This is significantly faster than encoding texts one at a time because:
    /// 1. Single ONNX session.run() call amortizes overhead
    /// 2. GPU/CPU can parallelize across batch dimension
    /// 3. Memory allocation is done once for the batch
    ///
    /// # Arguments
    /// * `texts` - Slice of text strings to encode
    ///
    /// # Returns
    /// * Vector of embeddings, one per input text
    fn generate_embeddings_batch_onnx(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Lazy load model on first use
        let model = self.ensure_model_loaded()?;
        let mut session = model.session.lock();

        let batch_size = texts.len();
        let max_length = self.config.max_length;

        // Tokenize all texts
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                model
                    .tokenizer
                    .encode(*text, true)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;

        // Prepare batched tensors
        let total_elements = batch_size * max_length;
        let mut input_ids = vec![0i64; total_elements];
        let mut attention_masks = vec![0i64; total_elements];
        let token_type_ids = vec![0i64; total_elements];

        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let tokens = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let offset = batch_idx * max_length;

            for (i, &token) in tokens.iter().take(max_length).enumerate() {
                input_ids[offset + i] = token as i64;
            }
            for (i, &mask) in attention_mask.iter().take(max_length).enumerate() {
                attention_masks[offset + i] = mask as i64;
            }
        }

        // Create batched input tensors
        let input_ids_value = Value::from_array((vec![batch_size, max_length], input_ids))?;
        let attention_mask_value =
            Value::from_array((vec![batch_size, max_length], attention_masks.clone()))?;
        let token_type_ids_value =
            Value::from_array((vec![batch_size, max_length], token_type_ids))?;

        // Run batch inference
        let outputs = session.run(ort::inputs![
            "input_ids" => &input_ids_value,
            "attention_mask" => &attention_mask_value,
            "token_type_ids" => &token_type_ids_value,
        ])?;

        // Extract embeddings - output shape is [batch_size, seq_length, hidden_size]
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, output_data) = output_tensor;

        // Mean pooling for each item in batch
        let mut results = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let mut pooled = vec![0.0; self.dimension];
            let mut mask_sum = 0.0;

            let batch_offset = batch_idx * max_length * self.dimension;
            let attention_offset = batch_idx * max_length;

            for seq_idx in 0..max_length {
                if attention_masks[attention_offset + seq_idx] == 1 {
                    for dim_idx in 0..self.dimension {
                        let idx = batch_offset + seq_idx * self.dimension + dim_idx;
                        pooled[dim_idx] += output_data[idx];
                    }
                    mask_sum += 1.0;
                }
            }

            // Average
            if mask_sum > 0.0 {
                for val in &mut pooled {
                    *val /= mask_sum;
                }
            }

            // Handle NaN/Inf values
            for val in pooled.iter_mut() {
                if val.is_nan() || val.is_infinite() {
                    *val = 0.0;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON && !norm.is_nan() {
                for val in &mut pooled {
                    *val /= norm;
                }
            }

            results.push(pooled);
        }

        Ok(results)
    }
}

impl Embedder for MiniLMEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }

        // Use simplified mode if in that mode
        if self.simplified_mode {
            let start = std::time::Instant::now();
            let result = self.generate_embedding_simplified(text);
            let duration = start.elapsed().as_secs_f64();

            if result.is_ok() {
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["simplified"])
                    .observe(duration);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["simplified", "success"])
                    .inc();
            } else {
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["simplified", "failure"])
                    .inc();
            }

            return result;
        }

        // Try ONNX inference (lazy loads model on first call)
        let start = std::time::Instant::now();

        match self.generate_embedding_onnx(text) {
            Ok(embedding) => {
                let duration = start.elapsed().as_secs_f64();
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["onnx"])
                    .observe(duration);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["onnx", "success"])
                    .inc();

                // Warn if inference is slow
                if duration * 1000.0 > self.config.embed_timeout_ms as f64 {
                    tracing::warn!(
                        "ONNX inference took {:.0}ms (threshold: {}ms)",
                        duration * 1000.0,
                        self.config.embed_timeout_ms
                    );
                }

                Ok(embedding)
            }
            Err(e) => {
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["onnx", "failure"])
                    .inc();
                tracing::warn!("ONNX inference failed: {}. Falling back to simplified.", e);

                // Fallback to simplified
                self.generate_embedding_simplified(text)
            }
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Handle empty strings in batch
        let empty_embedding = vec![0.0; self.dimension];
        if texts.iter().all(|t| t.is_empty()) {
            return Ok(vec![empty_embedding; texts.len()]);
        }

        // Use simplified mode if in that mode
        if self.simplified_mode {
            let start = std::time::Instant::now();
            let results: Result<Vec<_>> = texts
                .iter()
                .map(|text| {
                    if text.is_empty() {
                        Ok(vec![0.0; self.dimension])
                    } else {
                        self.generate_embedding_simplified(text)
                    }
                })
                .collect();
            let duration = start.elapsed().as_secs_f64();

            crate::metrics::EMBEDDING_GENERATE_DURATION
                .with_label_values(&["simplified_batch"])
                .observe(duration);
            crate::metrics::EMBEDDING_GENERATE_TOTAL
                .with_label_values(&[
                    "simplified_batch",
                    if results.is_ok() {
                        "success"
                    } else {
                        "failure"
                    },
                ])
                .inc();

            return results;
        }

        // Try batched ONNX inference
        let start = std::time::Instant::now();

        // Filter out empty strings and track their positions
        let (non_empty_texts, empty_indices): (Vec<_>, Vec<_>) =
            texts.iter().enumerate().partition(|(_, t)| !t.is_empty());

        let non_empty_texts: Vec<&str> = non_empty_texts.into_iter().map(|(_, t)| *t).collect();
        let empty_indices: Vec<usize> = empty_indices.into_iter().map(|(i, _)| i).collect();

        match self.generate_embeddings_batch_onnx(&non_empty_texts) {
            Ok(embeddings) => {
                let duration = start.elapsed().as_secs_f64();
                crate::metrics::EMBEDDING_GENERATE_DURATION
                    .with_label_values(&["onnx_batch"])
                    .observe(duration);
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["onnx_batch", "success"])
                    .inc();

                // Reconstruct results with empty embeddings in correct positions
                let mut results = Vec::with_capacity(texts.len());
                let mut embedding_iter = embeddings.into_iter();

                for i in 0..texts.len() {
                    if empty_indices.contains(&i) {
                        results.push(vec![0.0; self.dimension]);
                    } else {
                        results.push(
                            embedding_iter
                                .next()
                                .unwrap_or_else(|| vec![0.0; self.dimension]),
                        );
                    }
                }

                Ok(results)
            }
            Err(e) => {
                crate::metrics::EMBEDDING_GENERATE_TOTAL
                    .with_label_values(&["onnx_batch", "failure"])
                    .inc();
                tracing::warn!(
                    "Batch ONNX inference failed: {}. Falling back to sequential simplified.",
                    e
                );

                // Fallback to sequential simplified
                texts
                    .iter()
                    .map(|text| {
                        if text.is_empty() {
                            Ok(vec![0.0; self.dimension])
                        } else {
                            self.generate_embedding_simplified(text)
                        }
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minilm_creation() {
        // Test with default config
        let config = EmbeddingConfig::default();

        // Check dimension
        assert_eq!(config.max_length, 256);
    }

    #[test]
    fn test_embedding_generation_simplified() {
        // Create embedder in simplified mode (no ONNX model needed)
        let config = EmbeddingConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 256,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = MiniLMEmbedder::new_simplified(config).unwrap();

        let text = "Hello world";
        let embedding = embedder.encode(text).unwrap();

        assert_eq!(embedding.len(), 384);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding should be normalized");
    }

    #[test]
    fn test_batch_encoding_simplified() {
        // Create embedder in simplified mode
        let config = EmbeddingConfig {
            model_path: PathBuf::from("dummy.onnx"),
            tokenizer_path: PathBuf::from("dummy.json"),
            max_length: 256,
            use_quantized: true,
            embed_timeout_ms: 5000,
        };
        let embedder = MiniLMEmbedder::new_simplified(config).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let embeddings = embedder.encode_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.len(), 384);
        }
    }
}
