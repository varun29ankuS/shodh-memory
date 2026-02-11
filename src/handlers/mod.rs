//! HTTP API Handlers - Modular organization of the REST API
//!
//! This module contains all HTTP handlers extracted from the monolithic main.rs.
//! Each submodule handles a specific domain of functionality.

// Core modules
pub mod router;
pub mod state;
pub mod types;

// Health and utilities
pub mod health;
pub mod utils;

// Memory core operations
pub mod crud;
pub mod recall;
pub mod remember;

// Advanced memory operations
pub mod compression;
pub mod facts;
pub mod lineage;
pub mod search;

// Knowledge graph
pub mod graph;
pub mod visualization;

// Task management
pub mod todos;

// MCP and webhooks
pub mod mif;
pub mod webhooks;

// External integrations
pub mod integrations;

// Session and user management
pub mod sessions;
pub mod users;

// File and codebase memory
pub mod files;

// Background processing
pub mod consolidation;

// A/B testing
pub mod ab_testing;

// Test utilities (compiled only in test builds)
#[cfg(test)]
pub mod test_helpers;

// Re-export commonly used items
pub use router::{build_protected_routes, build_public_routes, build_router, AppState};
pub use state::MultiUserMemoryManager;
pub use types::*;
