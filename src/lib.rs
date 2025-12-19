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

pub mod auth;
pub mod backup;
pub mod constants;
pub mod decay;
pub mod embeddings;
pub mod errors;
pub mod graph_memory;
pub mod integrations;
pub mod memory;
pub mod metrics;
pub mod middleware;
pub mod relevance;
pub mod similarity;
pub mod streaming;
pub mod tracing_setup;
pub mod validation;
pub mod vector_db;

// Re-export dependencies to ensure tests/benchmarks use the same version
pub use chrono;
pub use parking_lot;
pub use uuid;

#[cfg(feature = "python")]
pub mod python;
