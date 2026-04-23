//! Shodh-Memory Library
//!
//! Edge-native AI memory system for autonomous agents.
//! Optimized for deployment on resource-constrained devices.
//!
//! # Key Features
//! - Tiered memory (working/session/long-term) based on cognitive science
//! - Local vector search (Vamana/DiskANN)
//! - Local embeddings (MiniLM-L6 via ONNX)
//! - Knowledge graph for entity relationships
//!
//! # Edge Optimizations
//! - Lazy model loading (reduces startup RAM by ~200MB)
//! - Configurable thread count for power efficiency
//! - RocksDB embedded storage (no external database)
//! - Full offline operation

pub mod ab_testing;
pub mod auth;
pub mod backup;
pub mod config;
pub mod constants;
pub mod decay;
pub mod embeddings;
pub mod errors;
pub mod graph_memory;
pub mod handlers;
pub mod integrations;
pub mod memory;
pub mod metrics;
pub mod middleware;
pub mod mif;
pub mod migration;
pub mod query_parsing;
pub mod relevance;
pub mod serialization;
pub mod server;
pub mod similarity;
pub mod streaming;
pub mod telemetry;
pub mod token_estimation;
pub mod tracing_setup;
pub mod validation;
pub mod vector_db;

/// Bincode 2.x decode with a 10 MB allocation limit.
///
/// All RocksDB deserialization MUST use this instead of `bincode::config::standard()`
/// to prevent OOM crashes from corrupted varint length prefixes. Without a limit,
/// a corrupted varint can declare multi-exabyte allocations that abort the process.
///
/// 10 MB is generous: the largest legitimate record (EntityNode with 384-dim
/// embedding + metadata) is ~2 KB. Even a Memory with full Experience is under 100 KB.
pub fn bincode_safe_config() -> impl bincode::config::Config {
    bincode::config::standard().with_limit::<{ 10 * 1024 * 1024 }>()
}

// Re-export dependencies to ensure tests/benchmarks use the same version
pub use chrono;
pub use parking_lot;
pub use uuid;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "zenoh")]
pub mod zenoh_transport;
