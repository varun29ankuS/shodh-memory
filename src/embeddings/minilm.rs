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
//! - SHODH_ONNX_THREADS: Number of ONNX threads (default: 1 on macOS ARM64, 2 elsewhere)

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

use super::Embedder;

/// Thread-safe guard for ORT_DYLIB_PATH initialization.
/// Using OnceLock ensures set_var is called exactly once.
static ORT_PATH_INIT: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Pre-initialize the ONNX Runtime path before any async work begins.
///
/// # Safety
/// This function calls `std::env::set_var` which is unsound in multi-threaded
/// contexts (Rust 1.66+). It MUST be called before `tokio::main` spawns worker
/// threads — i.e., very early in `async fn main()` before any `.await` or
/// `tokio::spawn` calls. The OnceLock ensures it only runs once.
pub fn pre_init_ort_runtime(offline_mode: bool) {
    let _ = ORT_PATH_INIT.get_or_init(|| MiniLMEmbedder::init_ort_path_inner(offline_mode));
}

/// Lazily initialized ONNX session and tokenizer
struct LazyModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl LazyModel {
    fn new(config: &EmbeddingConfig) -> Result<Self> {
        // macOS ARM64 (M1/M2/M3): default to 1 thread to avoid Eigen thread pool
        // spin-to-block deadlock on heterogeneous P/E cores.
        // See: https://github.com/microsoft/onnxruntime/issues/10270
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let default_threads = 1;
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let default_threads = 2;

        let num_threads = std::env::var("SHODH_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_threads);

        tracing::info!(
            "Loading MiniLM-L6-v2 model from {:?} with {} threads",
            config.model_path,
            num_threads
        );

        let builder = Session::builder()
            .context("Failed to create session builder")?
            .with_intra_threads(num_threads)
            .context("Failed to set intra thread count")?
            .with_inter_threads(1)
            .context("Failed to set inter thread count")?;

        // Disable thread pool spinning to prevent Eigen spin-to-block deadlock
        // on macOS ARM64 heterogeneous cores (P-core/E-core architecture).
        // See: microsoft/onnxruntime#10270, pykeio/ort#516
        let builder = builder
            .with_intra_op_spinning(false)
            .context("Failed to disable intra-op spinning")?
            .with_inter_op_spinning(false)
            .context("Failed to disable inter-op spinning")?;

        let session = builder
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
            max_length: configured_max_length(),
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
    /// Final, stored embedding dimension (always the 384-dim edge envelope).
    dimension: usize,
    /// Model's native hidden/output size used as the mean-pool stride. Equals
    /// `dimension` for native-384 models; larger (768) for nomic, which is
    /// pooled wide then (optionally) truncated to `dimension`.
    native_hidden: usize,
    /// Apply nomic's parameter-free LayerNorm over the full `native_hidden`
    /// width before any truncation. True for nomic (its reference recipe applies
    /// it at EVERY output dim, 768 or truncated); false for all other models.
    apply_prenorm: bool,
    /// Asymmetric instruction prefix prepended to QUERY text before encoding.
    /// Empty for symmetric models (MiniLM). Retrieval-tuned models like
    /// e5-small-v2 require `"query: "` / `"passage: "` to separate the query and
    /// document manifolds — this is the biggest free recall lever within the
    /// edge envelope (same 384-dim, same BERT arch + mean-pool). Set from the
    /// SHODH_EMBEDDER env var.
    query_prefix: String,
    /// Asymmetric instruction prefix prepended to DOCUMENT text before encoding.
    doc_prefix: String,
    /// Token-pooling strategy (Mean/CLS/LastToken), selected per model from
    /// `SHODH_EMBEDDER`. MiniLM = Mean (default, byte-identical to before).
    pooling: Pooling,
}

/// Resolve the (query, document) instruction prefixes from `SHODH_EMBEDDER`.
/// `e5` → e5-style `query:` / `passage:`; anything else → no prefixes
/// (symmetric MiniLM behavior, byte-identical to before).
fn embedder_prefixes() -> (String, String) {
    match std::env::var("SHODH_EMBEDDER")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "e5" | "e5-small" | "e5-small-v2" => ("query: ".to_string(), "passage: ".to_string()),
        // bge-v1.5 / mxbai use a query-side retrieval instruction, no doc prefix.
        "bge" | "bge-small" | "bge-small-en-v1.5" | "mxbai" | "mxbai-xsmall" => (
            "Represent this sentence for searching relevant passages: ".to_string(),
            String::new(),
        ),
        // gte is symmetric — no instruction prefix (same as default, made explicit).
        "gte" | "gte-small" => (String::new(), String::new()),
        // nomic-embed-text-v1.5 is asymmetric with task-instruction prefixes.
        "nomic" | "nomic-embed-text" | "nomic-embed-text-v1.5" => (
            "search_query: ".to_string(),
            "search_document: ".to_string(),
        ),
        // arctic-embed-m-v2.0: query instruction, no doc prefix (CLS-pooled).
        "arctic" | "arctic-embed" | "arctic-embed-m" => ("query: ".to_string(), String::new()),
        // granite-r2 (CLS) and harrier (last-token) are used symmetrically here —
        // no instruction prefix → fall through to the default empty pair.
        _ => (String::new(), String::new()),
    }
}

/// The configured text-embedding dimension — the SINGLE SOURCE OF TRUTH that the
/// embedder output, the Vamana index (`retrieval.rs`), and the stored vector
/// metadata (`storage.rs`) must all agree on. A mismatch makes vector search
/// stride over misaligned memory (silent corruption), so every consumer reads
/// this one value.
///
/// Default 384 (MiniLM/bge/gte/mxbai, and nomic-Matryoshka truncated to 384).
/// Set `SHODH_TEXT_DIM=768` to run nomic at its NATIVE 768 dim with no
/// truncation — the experiment that decouples "is the embedder weak?" from "is
/// 384 the binding constraint?". Only nomic supports 768; native-384 models
/// ignore it (they cannot emit 768).
pub fn configured_text_dim() -> usize {
    static DIM: OnceLock<usize> = OnceLock::new();
    *DIM.get_or_init(|| {
        std::env::var("SHODH_TEXT_DIM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|d| [128, 256, 384, 512, 768, 1024].contains(d))
            .unwrap_or(384)
    })
}

/// The tokenizer truncation length. Default 256 (MiniLM's limit — content beyond
/// is dropped). Long-context models (granite/harrier, 32K) can read whole sessions
/// when raised via `SHODH_MAX_LENGTH`. Clamped to [16, 32768]; a 256-token model
/// given a larger value will still error at inference, so only raise it for a model
/// whose positions support it.
pub fn configured_max_length() -> usize {
    static ML: OnceLock<usize> = OnceLock::new();
    *ML.get_or_init(|| {
        std::env::var("SHODH_MAX_LENGTH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| (16..=32768).contains(&n))
            .unwrap_or(256)
    })
}

/// True when `SHODH_EMBEDDER` selects nomic (needs the parameter-free LayerNorm).
fn embedder_is_nomic() -> bool {
    matches!(
        std::env::var("SHODH_EMBEDDER")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "nomic" | "nomic-embed-text" | "nomic-embed-text-v1.5"
    )
}

/// Resolve `(native_hidden, output_dim)` from `SHODH_EMBEDDER` + `SHODH_TEXT_DIM`.
///
/// - nomic: native_hidden = 768 (its true hidden size, the mean-pool stride);
///   output_dim = `configured_text_dim()` (768 = native/no-truncation, or 384 =
///   Matryoshka-truncated to the edge envelope).
/// - all others: native-384, `(384, 384)` — byte-identical to before.
fn embedder_dims() -> (usize, usize) {
    if embedder_native768() {
        (768, configured_text_dim())
    } else {
        (384, 384)
    }
}

/// True when `SHODH_EMBEDDER` selects a native-768 model (pooled wide, then
/// optionally Matryoshka-truncated to `configured_text_dim()`). nomic + the
/// 768-dim retrieval upgrades (granite-r2, arctic-embed, harrier). All others
/// are native-384 and emit `(384, 384)`.
fn embedder_native768() -> bool {
    matches!(
        std::env::var("SHODH_EMBEDDER")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "nomic"
            | "nomic-embed-text"
            | "nomic-embed-text-v1.5"
            | "granite"
            | "granite-r2"
            | "granite-embedding-r2"
            | "arctic"
            | "arctic-embed"
            | "arctic-embed-m"
            | "harrier"
            | "harrier-270m"
            | "harrier-oss-v1"
    )
}

/// Sentence-vector pooling strategy over the model's per-token output. Different
/// architectures need different reductions — getting this wrong silently corrupts
/// EVERY embedding (the recall number becomes meaningless), so it is selected
/// explicitly per model rather than assumed.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Pooling {
    /// Mean over non-padding tokens. BERT encoders trained with mean pooling
    /// (MiniLM, e5, bge, gte, nomic).
    Mean,
    /// First token ([CLS], position 0). Granite-r2 and Arctic-Embed use the CLS
    /// hidden state as the sentence vector.
    Cls,
    /// Last non-padding token. Decoder-only embedders (Harrier) read the final
    /// token's hidden state.
    LastToken,
}

/// Resolve the pooling strategy from `SHODH_EMBEDDER`. Default `Mean` keeps every
/// existing embedder byte-identical.
fn embedder_pooling() -> Pooling {
    match std::env::var("SHODH_EMBEDDER")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        // granite-r2 (768-native 311m) AND granite-small (384-native 97m) both pool
        // on CLS; they differ only in hidden size (see embedder_native768 — "granite"
        // is 768, "granite-small" is NOT listed there, so it stays 384-native).
        "granite"
        | "granite-r2"
        | "granite-embedding-r2"
        | "granite-small"
        | "granite-97m"
        | "arctic"
        | "arctic-embed"
        | "arctic-embed-m" => Pooling::Cls,
        "harrier" | "harrier-270m" | "harrier-oss-v1" => Pooling::LastToken,
        _ => Pooling::Mean,
    }
}

/// Reduce one item's `[seq_len, hidden]` slice of the model output into a single
/// `hidden`-wide vector per `mode`. `base` is the flat offset where this item's
/// block starts (0 for a single input; `batch_idx * max_length * hidden` in a
/// batch). `attended(seq_idx)` is true for non-padding positions. Pure function
/// of its inputs so the index arithmetic is unit-testable without a live model.
fn pool_tokens(
    mode: Pooling,
    output_data: &[f32],
    base: usize,
    max_length: usize,
    hidden: usize,
    attended: impl Fn(usize) -> bool,
) -> Vec<f32> {
    let mut pooled = vec![0.0f32; hidden];
    match mode {
        Pooling::Cls => {
            // [CLS] is always position 0.
            pooled.copy_from_slice(&output_data[base..base + hidden]);
        }
        Pooling::LastToken => {
            // Robust to either padding side: the last seq position with mask==1.
            let mut last = 0usize;
            for seq_idx in 0..max_length {
                if attended(seq_idx) {
                    last = seq_idx;
                }
            }
            let s = base + last * hidden;
            pooled.copy_from_slice(&output_data[s..s + hidden]);
        }
        Pooling::Mean => {
            let mut mask_sum = 0.0f32;
            for seq_idx in 0..max_length {
                if attended(seq_idx) {
                    let s = base + seq_idx * hidden;
                    for (d, pv) in pooled.iter_mut().enumerate() {
                        *pv += output_data[s + d];
                    }
                    mask_sum += 1.0;
                }
            }
            if mask_sum > 0.0 {
                for v in &mut pooled {
                    *v /= mask_sum;
                }
            }
        }
    }
    pooled
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
            Err(e) => anyhow::bail!("{e}"),
        }
    }

    /// Inner initialization logic - called exactly once via OnceLock.
    /// SAFETY: set_var is only called once due to OnceLock guard.
    fn init_ort_path_inner(offline_mode: bool) -> Result<PathBuf, String> {
        // If ORT_DYLIB_PATH is already set to a valid path, we're good
        if let Ok(existing_path) = std::env::var("ORT_DYLIB_PATH") {
            let path = std::path::PathBuf::from(&existing_path);
            if path.exists() {
                // eprintln because tracing subscriber may not be initialized yet
                // (pre_init_ort_runtime runs before tokio/tracing setup)
                eprintln!("[shodh] Using ONNX Runtime from ORT_DYLIB_PATH: {:?}", path);
                return Ok(path);
            }
        }

        // Check for bundled ONNX Runtime in Python package (1-click install)
        if let Some(bundled_path) = Self::find_bundled_onnx_runtime() {
            eprintln!(
                "[shodh] Using bundled ONNX Runtime from package: {:?}",
                bundled_path
            );
            // SAFETY: This is called once via OnceLock, before other threads start
            std::env::set_var("ORT_DYLIB_PATH", &bundled_path);
            return Ok(bundled_path);
        }

        // Check if we have ONNX Runtime in our cache
        if let Some(cached_path) = super::downloader::get_onnx_runtime_path() {
            eprintln!("[shodh] Using cached ONNX Runtime: {:?}", cached_path);
            // SAFETY: This is called once via OnceLock, before other threads start
            std::env::set_var("ORT_DYLIB_PATH", &cached_path);
            return Ok(cached_path);
        }

        // Need to download ONNX Runtime
        if offline_mode {
            return Err("ONNX Runtime not found and SHODH_OFFLINE=true".to_string());
        }

        eprintln!();
        eprintln!("  \u{1F4E6} Downloading runtime (first run only)...");
        let progress = super::downloader::make_stderr_progress("ONNX Runtime".to_string());
        let onnx_path =
            super::downloader::download_onnx_runtime(Some(progress)).map_err(|e| e.to_string())?;
        eprintln!();
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
            return Self::simplified_or_fail(config, format!("Failed to set up ONNX Runtime: {e}"));
        }

        // Check if model files exist
        let model_available = config.model_path.exists() && config.tokenizer_path.exists();

        if !model_available {
            if offline_mode {
                return Self::simplified_or_fail(
                    config,
                    "Model files not found and SHODH_OFFLINE=true".to_string(),
                );
            }

            // Try to auto-download model files
            tracing::info!(
                "Model files not found at {:?}. Downloading...",
                config.model_path.parent().unwrap_or(&config.model_path)
            );

            eprintln!();
            eprintln!("  \u{1F4E6} Downloading models (first run only)...");
            match super::downloader::download_models(Some(super::downloader::make_stderr_progress(
                "MiniLM-L6 (23 MB)".to_string(),
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
                    return Self::simplified_or_fail(
                        config,
                        format!("Failed to download models: {e}"),
                    );
                }
            }
        }

        let (query_prefix, doc_prefix) = embedder_prefixes();
        let (native_hidden, dimension) = embedder_dims();
        let embedder = Self {
            config: config.clone(),
            lazy_model: OnceLock::new(),
            simplified_mode: false,
            dimension,
            native_hidden,
            apply_prenorm: embedder_is_nomic(),
            query_prefix,
            doc_prefix,
            pooling: embedder_pooling(),
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

    /// Either fall back to simplified (hash) embeddings — but ONLY when the caller has
    /// explicitly opted in via `SHODH_ALLOW_SIMPLIFIED_EMBEDDINGS` — or hard-fail.
    ///
    /// Hash-based embeddings are non-semantic: vector search becomes meaningless and any
    /// downstream benchmark silently degrades into a plausible-looking but garbage "success"
    /// (this is exactly what voided the E3 substrate A/B — a model-download failure on
    /// concurrent CI runners silently produced all-zero vamana recall that still reported
    /// `conclusion=success`). Default behaviour is therefore to FAIL LOUDLY so corrupt runs
    /// are caught, not hidden. Genuine resource-constrained / offline edge deployments that
    /// accept degraded recall opt in with `SHODH_ALLOW_SIMPLIFIED_EMBEDDINGS=1`.
    fn simplified_or_fail(config: EmbeddingConfig, reason: String) -> Result<Self> {
        let allow = std::env::var("SHODH_ALLOW_SIMPLIFIED_EMBEDDINGS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if allow {
            tracing::warn!(
                "{reason}. SHODH_ALLOW_SIMPLIFIED_EMBEDDINGS is set — falling back to \
                 hash-based embeddings (DEGRADED, non-semantic; recall quality will be poor).",
            );
            Self::new_simplified(config)
        } else {
            anyhow::bail!(
                "{reason}. Refusing to silently fall back to hash-based (non-semantic) \
                 embeddings — vector search would be meaningless and any benchmark a false \
                 success. Fix the model path / network, or set \
                 SHODH_ALLOW_SIMPLIFIED_EMBEDDINGS=1 to explicitly accept degraded embeddings \
                 (edge/offline only)."
            )
        }
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

        let (query_prefix, doc_prefix) = embedder_prefixes();
        let (native_hidden, dimension) = embedder_dims();
        Ok(Self {
            config,
            lazy_model: OnceLock::new(),
            simplified_mode: true,
            dimension,
            native_hidden,
            apply_prenorm: embedder_is_nomic(),
            query_prefix,
            doc_prefix,
            pooling: embedder_pooling(),
        })
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

        // Use words and character n-grams for better quality
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();

            // Distribute hash bits across embedding dimensions with positional offset
            for j in 0..self.dimension {
                let index = (i.wrapping_mul(7) + j) % self.dimension;
                // For j >= 64, use a scattered bit index that varies with both word
                // position (i) and dimension (j), avoiding reuse of the same bit pattern.
                let bit_index = if j < 64 {
                    j
                } else {
                    (i.wrapping_mul(7).wrapping_add(j)) % 64
                };
                embedding[index] += ((hash >> bit_index) & 1) as f32 * 0.1;
            }
        }

        // Add character bigram features for better semantic representation
        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len().saturating_sub(1) {
            let mut hasher = DefaultHasher::new();
            let bigram = format!("{}{}", chars[i], chars[i + 1]);
            bigram.hash(&mut hasher);
            let hash = hasher.finish();

            for j in 0..32 {
                let index = ((hash as usize) + j) % self.dimension;
                embedding[index] += ((hash >> (j % 64)) & 1) as f32 * 0.05;
            }
        }

        // Normalize - if normalization fails (empty text / NaN), return zero vector
        if !self.normalize(&mut embedding) {
            tracing::warn!(
                "Embedding normalization failed (zero norm or NaN), returning zero vector"
            );
            embedding.iter_mut().for_each(|v| *v = 0.0);
        }

        Ok(embedding)
    }

    /// Finalize a wide mean-pooled vector into the stored embedding.
    ///
    /// Native-384 models (MiniLM/bge/gte/mxbai): scrub NaN/Inf → L2-normalize,
    /// byte-identical to the prior path (no LayerNorm, no truncation).
    ///
    /// nomic (`apply_prenorm`): scrub → parameter-free LayerNorm over the FULL
    /// `native_hidden` (768) width → truncate to the leading `dimension` dims (a
    /// no-op when running native 768) → L2-normalize. This is nomic's exact
    /// reference recipe (`F.layer_norm(x, (768,))` no learned affine, then
    /// `x[:, :d]`, then `F.normalize`). The LayerNorm is gated on the MODEL, not
    /// on truncation: nomic applies it at every output dim (768 or truncated).
    /// Order is load-bearing — LayerNorm sees all 768 dims, L2 is the last step on
    /// the (possibly truncated) prefix, or the embedding is not a valid unit vector.
    fn finalize_pooled(&self, mut pooled: Vec<f32>) -> Vec<f32> {
        for val in pooled.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }

        if self.apply_prenorm {
            // Parameter-free LayerNorm across the full native width (nomic recipe).
            let n = pooled.len() as f32;
            let mean = pooled.iter().sum::<f32>() / n;
            let var = pooled.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
            let denom = (var + 1e-5).sqrt();
            if denom > f32::EPSILON {
                for val in &mut pooled {
                    *val = (*val - mean) / denom;
                }
            }
        }

        // Matryoshka truncation to the configured output dim (no-op at native 768).
        if pooled.len() > self.dimension {
            pooled.truncate(self.dimension);
        }

        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON && !norm.is_nan() {
            for val in &mut pooled {
                *val /= norm;
            }
        }
        pooled
    }

    /// Generate embedding using ONNX Runtime (production)
    ///
    /// Lazily loads the model on first call if not already loaded.
    fn generate_embedding_onnx(&self, text: &str) -> Result<Vec<f32>> {
        // Lazy load model on first use
        tracing::debug!("ONNX: ensuring model loaded...");
        let model = self.ensure_model_loaded()?;
        tracing::debug!("ONNX: model ready, acquiring session lock...");

        let lock_timeout = std::time::Duration::from_secs(30);
        let mut session = model.session.try_lock_for(lock_timeout).ok_or_else(|| {
            tracing::error!(
                "ONNX session lock acquisition timed out after {}s — a previous inference \
                     call is likely stuck. Falling back to simplified embeddings.",
                lock_timeout.as_secs()
            );
            anyhow::anyhow!("ONNX session lock timeout ({}s)", lock_timeout.as_secs())
        })?;
        tracing::debug!("ONNX: session lock acquired, tokenizing...");

        // Tokenize input text
        let encoding = model
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let tokens = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let max_length = self.config.max_length;
        tracing::debug!("ONNX: tokenized {} tokens", tokens.len());

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

        // token_type_ids is optional: standard BERT exports (MiniLM/bge/gte/mxbai)
        // declare it, but nomic-bert and other rotary-position models do not — and
        // ORT rejects an input the graph never declared. Only bind it when present.
        let wants_token_type = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        // Run inference
        tracing::debug!("ONNX: running inference...");
        let outputs = if wants_token_type {
            session.run(ort::inputs![
                "input_ids" => &input_ids_value,
                "attention_mask" => &attention_mask_value,
                "token_type_ids" => &token_type_ids_value,
            ])?
        } else {
            session.run(ort::inputs![
                "input_ids" => &input_ids_value,
                "attention_mask" => &attention_mask_value,
            ])?
        };
        tracing::debug!("ONNX: inference complete");

        // Extract embeddings
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, output_data) = output_tensor;

        // Pool per the model's strategy (Mean/CLS/LastToken) at the native hidden
        // size (the output stride; wider than `dimension` for Matryoshka models).
        let pooled = pool_tokens(
            self.pooling,
            output_data,
            0,
            attention.len(),
            self.native_hidden,
            |seq_idx| attention[seq_idx] == 1,
        );

        // Scrub NaN/Inf, Matryoshka-truncate to `dimension`, then L2 normalize.
        Ok(self.finalize_pooled(pooled))
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
        let lock_timeout = std::time::Duration::from_secs(30);
        let mut session = model.session.try_lock_for(lock_timeout).ok_or_else(|| {
            tracing::error!(
                "ONNX session lock timed out after {}s in batch embed — \
                 a previous inference call is likely stuck.",
                lock_timeout.as_secs()
            );
            anyhow::anyhow!(
                "ONNX session lock timeout ({}s) in batch embed",
                lock_timeout.as_secs()
            )
        })?;

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

        // token_type_ids is optional — see generate_embedding_onnx. nomic-bert and
        // other rotary-position exports omit it; ORT rejects undeclared inputs.
        let wants_token_type = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        // Run batch inference
        let outputs = if wants_token_type {
            session.run(ort::inputs![
                "input_ids" => &input_ids_value,
                "attention_mask" => &attention_mask_value,
                "token_type_ids" => &token_type_ids_value,
            ])?
        } else {
            session.run(ort::inputs![
                "input_ids" => &input_ids_value,
                "attention_mask" => &attention_mask_value,
            ])?
        };

        // Extract embeddings - output shape is [batch_size, seq_length, hidden_size]
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let (_shape, output_data) = output_tensor;

        // Pool each batch item per the model's strategy at the native hidden size
        // (the output stride; wider than `dimension` for Matryoshka models).
        let hidden = self.native_hidden;
        let mut results = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let base = batch_idx * max_length * hidden;
            let attention_offset = batch_idx * max_length;
            let pooled = pool_tokens(
                self.pooling,
                output_data,
                base,
                max_length,
                hidden,
                |seq_idx| attention_masks[attention_offset + seq_idx] == 1,
            );

            // Scrub NaN/Inf, Matryoshka-truncate to `dimension`, then L2 normalize.
            results.push(self.finalize_pooled(pooled));
        }

        Ok(results)
    }
}

impl MiniLMEmbedder {
    /// Core encode path with an optional asymmetric instruction prefix.
    /// `prefix` is empty for symmetric models (byte-identical to the prior
    /// behavior) and `"query: "` / `"passage: "` for e5-style models.
    fn encode_prefixed(&self, text: &str, prefix: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }

        // Prepend the instruction prefix (if any) before tokenization.
        let owned;
        let text: &str = if prefix.is_empty() {
            text
        } else {
            owned = format!("{prefix}{text}");
            &owned
        };

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
}

impl Embedder for MiniLMEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        // Documents/passages get the doc prefix (empty for symmetric MiniLM).
        self.encode_prefixed(text, &self.doc_prefix)
    }

    fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        // Queries get the query prefix (empty for symmetric MiniLM).
        self.encode_prefixed(text, &self.query_prefix)
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
    fn test_pool_tokens_modes() {
        // One item: 3 tokens x hidden=2 -> [1,2 | 3,4 | 5,6]; token 2 is padding.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let attended = |s: usize| [true, true, false][s];

        // CLS = first token (position 0).
        assert_eq!(
            pool_tokens(Pooling::Cls, &data, 0, 3, 2, attended),
            vec![1.0, 2.0]
        );
        // LastToken = last NON-padding token (seq 1), not the padded seq 2.
        assert_eq!(
            pool_tokens(Pooling::LastToken, &data, 0, 3, 2, attended),
            vec![3.0, 4.0]
        );
        // Mean over non-padding tokens (seq 0,1) -> [(1+3)/2, (2+4)/2].
        assert_eq!(
            pool_tokens(Pooling::Mean, &data, 0, 3, 2, attended),
            vec![2.0, 3.0]
        );
    }

    #[test]
    fn test_pool_tokens_batch_offset() {
        // Two items: item0 [1..6], item1 [7..12]; hidden=2, len=3, all attended.
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let all = |_s: usize| true;
        // Item 1 starts at base = 1 * 3 * 2 = 6 — the batch-offset arithmetic.
        assert_eq!(
            pool_tokens(Pooling::Cls, &data, 6, 3, 2, all),
            vec![7.0, 8.0]
        );
        assert_eq!(
            pool_tokens(Pooling::LastToken, &data, 6, 3, 2, all),
            vec![11.0, 12.0]
        );
        assert_eq!(
            pool_tokens(Pooling::Mean, &data, 6, 3, 2, all),
            vec![9.0, 10.0]
        );
    }

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
