//! Vector database module using Vamana graph-based index
//!
//! High-performance vector similarity search for the memory system.
//! Uses Microsoft Research's Vamana algorithm with SIMD-optimized distance calculations.
//!
//! # Distance Metrics
//!
//! The index supports three distance metrics configurable via [`DistanceMetric`]:
//!
//! - **NormalizedDotProduct** (default): Best for normalized embeddings (MiniLM, etc.)
//! - **Euclidean**: L2 squared distance for general use
//! - **Cosine**: Cosine distance (1 - similarity) for unnormalized vectors
//!
//! # Example
//!
//! ```ignore
//! use shodh_memory::vector_db::{VamanaConfig, VamanaIndex, DistanceMetric};
//!
//! let config = VamanaConfig {
//!     dimension: 384,
//!     distance_metric: DistanceMetric::NormalizedDotProduct,
//!     ..Default::default()
//! };
//! let index = VamanaIndex::new(config)?;
//! ```

pub mod distance_inline;
pub mod vamana;

// Re-export key types for convenient access
pub use vamana::{DistanceMetric, VamanaConfig, VamanaIndex, REBUILD_THRESHOLD};
