//! Vector database module with pluggable index backends
//!
//! High-performance vector similarity search for the memory system.
//! Supports two backends:
//!
//! - **Vamana** (default): Graph-based ANN, optimal for <100k vectors, lowest latency
//! - **SPANN**: Disk-based IVF+PQ, optimal for >100k vectors, billion-scale
//!
//! # Distance Metrics
//!
//! Both backends support three distance metrics via [`DistanceMetric`]:
//!
//! - **NormalizedDotProduct** (default): Best for normalized embeddings (MiniLM, etc.)
//! - **Euclidean**: L2 squared distance for general use
//! - **Cosine**: Cosine distance (1 - similarity) for unnormalized vectors
//!
//! # Auto-Selection
//!
//! Use `VectorIndexBackend::auto()` to automatically select the best backend
//! based on expected dataset size.
//!
//! # Example
//!
//! ```ignore
//! use shodh_memory::vector_db::{VectorIndexBackend, BackendConfig};
//!
//! // Auto-select based on expected size
//! let backend = VectorIndexBackend::auto(BackendConfig::default(), 50_000)?;
//!
//! // Add vectors
//! backend.add_vector(embedding)?;
//!
//! // Search
//! let results = backend.search(&query, 10)?;
//! ```

pub mod distance_inline;
pub mod pq;
pub mod spann;
pub mod vamana;
pub mod vamana_persist;

// Re-export key types for convenient access
pub use pq::{CompressedVectorStore, PQConfig, ProductQuantizer};
pub use spann::{SpannConfig, SpannIndex};
pub use vamana::{DistanceMetric, VamanaConfig, VamanaIndex, REBUILD_THRESHOLD};

use anyhow::Result;
use std::path::Path;

/// Threshold for auto-selecting SPANN over Vamana
/// SPANN is better for large datasets due to disk-based storage
pub const SPANN_AUTO_THRESHOLD: usize = 100_000;

/// Configuration for vector index backend
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Vector dimension (must match embedding model)
    pub dimension: usize,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Force specific backend (None = auto-select)
    pub force_backend: Option<BackendType>,
    /// Enable PQ compression for SPANN (saves 32x storage)
    pub use_pq: bool,
    /// Number of partitions to probe in SPANN search
    pub spann_probes: usize,
    /// Max degree for Vamana graph
    pub vamana_max_degree: usize,
    /// Search list size for Vamana
    pub vamana_search_list_size: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            dimension: 384, // MiniLM
            distance_metric: DistanceMetric::NormalizedDotProduct,
            force_backend: None,
            use_pq: true,
            spann_probes: 20,
            vamana_max_degree: 32,
            vamana_search_list_size: 100,
        }
    }
}

/// Backend type for vector index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Graph-based ANN - fast, in-memory, best for <100k vectors
    Vamana,
    /// Disk-based IVF+PQ - scalable, best for >100k vectors
    Spann,
}

/// Unified vector index backend supporting Vamana and SPANN
pub enum VectorIndexBackend {
    Vamana(VamanaIndex),
    Spann(SpannIndex),
}

impl VectorIndexBackend {
    /// Create backend with auto-selection based on expected vector count
    pub fn auto(config: BackendConfig, expected_vectors: usize) -> Result<Self> {
        #[allow(clippy::unnecessary_lazy_evaluations)]
        // Intentionally lazy: default depends on expected_vectors
        let backend_type = config.force_backend.unwrap_or_else(|| {
            if expected_vectors >= SPANN_AUTO_THRESHOLD {
                BackendType::Spann
            } else {
                BackendType::Vamana
            }
        });

        match backend_type {
            BackendType::Vamana => Self::new_vamana(config),
            BackendType::Spann => Self::new_spann(config),
        }
    }

    /// Create Vamana backend explicitly
    pub fn new_vamana(config: BackendConfig) -> Result<Self> {
        let vamana_config = VamanaConfig {
            dimension: config.dimension,
            max_degree: config.vamana_max_degree,
            search_list_size: config.vamana_search_list_size,
            distance_metric: config.distance_metric,
            ..Default::default()
        };
        Ok(Self::Vamana(VamanaIndex::new(vamana_config)?))
    }

    /// Create SPANN backend explicitly
    pub fn new_spann(config: BackendConfig) -> Result<Self> {
        let spann_config = SpannConfig {
            dimension: config.dimension,
            use_pq: config.use_pq,
            num_probes: config.spann_probes,
            distance_metric: config.distance_metric,
            ..Default::default()
        };
        Ok(Self::Spann(SpannIndex::new(spann_config)))
    }

    /// Get backend type
    pub fn backend_type(&self) -> BackendType {
        match self {
            Self::Vamana(_) => BackendType::Vamana,
            Self::Spann(_) => BackendType::Spann,
        }
    }

    /// Add a vector to the index, returns vector ID
    pub fn add_vector(&mut self, vector: Vec<f32>) -> Result<u32> {
        match self {
            Self::Vamana(idx) => idx.add_vector(vector),
            Self::Spann(idx) => {
                let id = idx.len() as u32;
                idx.insert(id, &vector)?;
                Ok(id)
            }
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        match self {
            Self::Vamana(idx) => idx.search(query, k),
            Self::Spann(idx) => idx.search(query, k),
        }
    }

    /// Number of vectors in the index
    pub fn len(&self) -> usize {
        match self {
            Self::Vamana(idx) => idx.len(),
            Self::Spann(idx) => idx.len(),
        }
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Save index to file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        match self {
            Self::Vamana(idx) => idx.save_to_file(path),
            Self::Spann(idx) => idx.save_to_file(path),
        }
    }

    /// Load index from file
    pub fn load_from_file(path: &Path, backend_type: BackendType) -> Result<Self> {
        match backend_type {
            BackendType::Vamana => Ok(Self::Vamana(VamanaIndex::load_from_file(path)?)),
            BackendType::Spann => Ok(Self::Spann(SpannIndex::load_from_file(path)?)),
        }
    }

    /// Build index from vectors (for SPANN, Vamana builds incrementally)
    pub fn build(&mut self, vectors: Vec<Vec<f32>>) -> Result<()> {
        match self {
            Self::Vamana(idx) => idx.build(vectors),
            Self::Spann(idx) => idx.build(vectors),
        }
    }

    /// Check if index needs rebuild (Vamana only)
    pub fn needs_rebuild(&self) -> bool {
        match self {
            Self::Vamana(idx) => idx.needs_rebuild(),
            Self::Spann(_) => false, // SPANN doesn't need rebuild
        }
    }

    /// Auto-rebuild if needed (Vamana only)
    pub fn auto_rebuild_if_needed(&mut self) -> Result<bool> {
        match self {
            Self::Vamana(idx) => idx.auto_rebuild_if_needed(),
            Self::Spann(_) => Ok(false),
        }
    }

    /// Get incremental insert count (Vamana only)
    pub fn incremental_insert_count(&self) -> usize {
        match self {
            Self::Vamana(idx) => idx.incremental_insert_count(),
            Self::Spann(_) => 0,
        }
    }

    /// Get deleted count (Vamana only)
    pub fn deleted_count(&self) -> usize {
        match self {
            Self::Vamana(idx) => idx.deleted_count(),
            Self::Spann(_) => 0,
        }
    }

    /// Get deletion ratio (Vamana only)
    pub fn deletion_ratio(&self) -> f32 {
        match self {
            Self::Vamana(idx) => idx.deletion_ratio(),
            Self::Spann(_) => 0.0,
        }
    }

    /// Check if needs compaction (Vamana only)
    pub fn needs_compaction(&self) -> bool {
        match self {
            Self::Vamana(idx) => idx.needs_compaction(),
            Self::Spann(_) => false,
        }
    }

    /// Verify index file integrity
    pub fn verify_index_file(path: &Path, backend_type: BackendType) -> Result<bool> {
        match backend_type {
            BackendType::Vamana => VamanaIndex::verify_index_file(path),
            BackendType::Spann => SpannIndex::verify_index_file(path),
        }
    }
}
