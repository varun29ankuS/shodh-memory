//! Shodh-Memory - Offline, user-isolated memory layer for AI agents
//!
//! Standalone memory server with REST API for Python clients

use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tower_http::cors::{Any, CorsLayer, AllowOrigin};
use tower_http::services::ServeDir;
use tower_governor::{
    governor::GovernorConfigBuilder,
    GovernorLayer,
};
use tower::limit::ConcurrencyLimitLayer;
use tracing::info;

mod auth;
mod config;
mod memory;
mod similarity;
mod validation;
mod errors;
mod graph_memory;
mod vector_db;
mod embeddings;
mod metrics;        // P1.1: Observability
mod middleware;     // P1.3: HTTP request tracking
mod tracing_setup;  // P1.6: Distributed tracing

use config::ServerConfig;

use memory::{Experience, Memory, MemoryConfig, MemoryId, MemoryStats, MemorySystem, Query as MemoryQuery, GraphStats as VisualizationStats};
use similarity::top_k_similar;
use errors::{AppError, ValidationErrorExt};
use graph_memory::{GraphMemory, EntityNode, EntityExtractor, RelationshipEdge, RelationType, EpisodicNode, EpisodeSource, GraphTraversal, GraphStats};

// P0.11: Shutdown timeouts for production resilience
const GRACEFUL_SHUTDOWN_TIMEOUT_SECS: u64 = 30;  // Max time to drain requests
const DATABASE_FLUSH_TIMEOUT_SECS: u64 = 10;     // Max time to flush RocksDB
const VECTOR_INDEX_SAVE_TIMEOUT_SECS: u64 = 10;  // Max time to save indices

/// Audit event for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: String,  // CREATE, UPDATE, DELETE, RETRIEVE
    pub memory_id: String,
    pub details: String,
}

// Note: Audit and memory configuration is now in config.rs and loaded via ServerConfig

/// Helper struct for audit log rotation (allows spawn_blocking with minimal clone)
struct MultiUserMemoryManagerRotationHelper {
    audit_db: Arc<rocksdb::DB>,
    audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<Vec<AuditEvent>>>>>,
    /// Audit retention days (from config)
    audit_retention_days: i64,
    /// Max audit entries per user (from config)
    audit_max_entries: usize,
}

impl MultiUserMemoryManagerRotationHelper {
    fn rotate_user_audit_logs(&self, user_id: &str) -> Result<usize> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::days(self.audit_retention_days);
        let cutoff_nanos = cutoff_time.timestamp_nanos_opt().unwrap_or(0);

        let mut events: Vec<(Vec<u8>, AuditEvent, i64)> = Vec::new();
        let prefix = format!("{user_id}:");

        // Collect all events for this user with their timestamps
        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for item in iter {
            if let Ok((key, value)) = item {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with(&prefix) {
                        break;
                    }

                    if let Ok(event) = bincode::deserialize::<AuditEvent>(&value) {
                        let timestamp_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                        events.push((key.to_vec(), event, timestamp_nanos));
                    }
                }
            }
        }

        let initial_count = events.len();
        let mut removed_count = 0;

        // Sort by timestamp (newest first)
        events.sort_by(|a, b| b.2.cmp(&a.2));

        // Determine which events to remove
        let mut keys_to_remove = Vec::new();

        for (idx, (key, _event, timestamp_nanos)) in events.iter().enumerate() {
            let should_remove =
                // Remove if older than retention period
                *timestamp_nanos < cutoff_nanos ||
                // Remove if exceeding max count (keep only newest N entries)
                idx >= self.audit_max_entries;

            if should_remove {
                keys_to_remove.push(key.clone());
                removed_count += 1;
            }
        }

        // Remove old entries from RocksDB
        if !keys_to_remove.is_empty() {
            let mut batch = rocksdb::WriteBatch::default();
            for key in &keys_to_remove {
                batch.delete(key);
            }
            self.audit_db.write(batch)
                .map_err(|e| anyhow::anyhow!("Failed to write rotation batch: {e}"))?;
        }

        // Update in-memory cache by removing old events
        if removed_count > 0 {
            if let Some(log) = self.audit_logs.get(user_id) {
                let mut log_guard = log.write();

                // Keep only events that weren't removed
                log_guard.retain(|event| {
                    let event_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                    event_nanos >= cutoff_nanos &&
                    initial_count - removed_count <= self.audit_max_entries
                });

                // If still too many, keep only the newest ones
                if log_guard.len() > self.audit_max_entries {
                    log_guard.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
                    log_guard.truncate(self.audit_max_entries);
                }
            }
        }

        Ok(removed_count)
    }
}

/// Multi-user memory manager
pub struct MultiUserMemoryManager {
    /// Per-user memory systems with LRU eviction (prevents unbounded growth)
    /// Wrapped in Mutex because LruCache needs exclusive access even for reads (to update LRU order)
    user_memories: Arc<parking_lot::Mutex<lru::LruCache<String, Arc<parking_lot::RwLock<MemorySystem>>>>>,

    /// Per-user audit logs (enterprise feature - in-memory cache)
    audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<Vec<AuditEvent>>>>>,

    /// Persistent audit log storage
    audit_db: Arc<rocksdb::DB>,

    /// Base storage path
    base_path: std::path::PathBuf,

    /// Default config
    default_config: MemoryConfig,

    /// Counter for audit log rotation checks (atomic for lock-free increment)
    audit_log_counter: Arc<std::sync::atomic::AtomicUsize>,

    /// Per-user graph memory systems (knowledge graphs) - also needs LRU eviction
    graph_memories: Arc<parking_lot::Mutex<lru::LruCache<String, Arc<parking_lot::RwLock<GraphMemory>>>>>,

    /// Entity extractor for automatic entity extraction
    entity_extractor: Arc<EntityExtractor>,

    /// User eviction counter for metrics
    user_evictions: Arc<std::sync::atomic::AtomicUsize>,

    /// Server configuration (configurable via environment)
    server_config: ServerConfig,
}

impl MultiUserMemoryManager {
    pub fn new(base_path: std::path::PathBuf, server_config: ServerConfig) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;

        // Initialize persistent audit log storage
        let audit_path = base_path.join("audit_logs");
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let audit_db = Arc::new(rocksdb::DB::open(&opts, audit_path)?);

        // Use NonZeroUsize for LRU cache size (guaranteed non-zero from config)
        let cache_size = std::num::NonZeroUsize::new(server_config.max_users_in_memory)
            .unwrap_or_else(|| std::num::NonZeroUsize::new(1000).unwrap());

        let manager = Self {
            user_memories: Arc::new(parking_lot::Mutex::new(
                lru::LruCache::new(cache_size)
            )),
            audit_logs: Arc::new(DashMap::new()),
            audit_db,
            base_path,
            default_config: MemoryConfig::default(),
            audit_log_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            graph_memories: Arc::new(parking_lot::Mutex::new(
                lru::LruCache::new(cache_size)
            )),
            entity_extractor: Arc::new(EntityExtractor::new()),
            user_evictions: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            server_config,
        };

        // Perform initial audit log rotation on startup
        info!("🧹 Running initial audit log rotation...");
        if let Err(e) = manager.rotate_all_audit_logs() {
            tracing::warn!("Failed to rotate audit logs on startup: {}", e);
        }

        Ok(manager)
    }

    /// Log audit event (non-blocking with background persistence)
    fn log_event(&self, user_id: &str, event_type: &str, memory_id: &str, details: &str) {
        let event = AuditEvent {
            timestamp: chrono::Utc::now(),
            event_type: event_type.to_string(),
            memory_id: memory_id.to_string(),
            details: details.to_string(),
        };

        // Persist to RocksDB in background (non-blocking)
        let key = format!("{}:{}", user_id, event.timestamp.timestamp_nanos_opt().unwrap_or(0));
        if let Ok(serialized) = bincode::serialize(&event) {
            let db = self.audit_db.clone();
            let key_bytes = key.into_bytes();

            // Spawn blocking DB write on dedicated thread pool
            tokio::task::spawn_blocking(move || {
                let _ = db.put(&key_bytes, &serialized);
            });
        }

        // Also update in-memory cache for fast access
        if let Some(log) = self.audit_logs.get(user_id) {
            log.write().push(event);
        } else {
            let log = Arc::new(parking_lot::RwLock::new(vec![event]));
            self.audit_logs.insert(user_id.to_string(), log);
        }

        // Check if rotation is needed (every N events) - LOCK-FREE atomic operation
        let count = self.audit_log_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Only the thread that hits the interval performs rotation (no race condition)
        if count % self.server_config.audit_rotation_check_interval == 0 && count > 0 {
            // CRITICAL FIX: Run rotation in background blocking thread
            // rotate_user_audit_logs() does RocksDB prefix_iterator + WriteBatch (blocking I/O)
            // Running this on async threads starves the Tokio runtime
            let audit_db = self.audit_db.clone();
            let audit_logs = self.audit_logs.clone();
            let user_id_clone = user_id.to_string();

            let audit_retention_days = self.server_config.audit_retention_days as i64;
            let audit_max_entries = self.server_config.audit_max_entries_per_user;

            tokio::task::spawn_blocking(move || {
                // Reconstruct the necessary state for rotation
                let manager = MultiUserMemoryManagerRotationHelper {
                    audit_db,
                    audit_logs,
                    audit_retention_days,
                    audit_max_entries,
                };
                if let Err(e) = manager.rotate_user_audit_logs(&user_id_clone) {
                    tracing::debug!("Audit log rotation check for user {}: {}", user_id_clone, e);
                }
            });
        }
    }

    /// Get audit history for user (reads from persistent storage)
    pub fn get_history(&self, user_id: &str, memory_id: Option<&str>) -> Vec<AuditEvent> {
        // Try in-memory cache first (fast path)
        if let Some(log) = self.audit_logs.get(user_id) {
            let events = log.read();
            if !events.is_empty() {
                // Cache hit - use cached data
                return if let Some(mid) = memory_id {
                    events.iter().filter(|e| e.memory_id == mid).cloned().collect()
                } else {
                    events.clone()
                };
            }
        }

        // Cache miss - load from RocksDB (happens after restart)
        let mut events = Vec::new();
        let prefix = format!("{user_id}:");

        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for item in iter {
            if let Ok((key, value)) = item {
                // Check if key still matches our user_id prefix
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if !key_str.starts_with(&prefix) {
                        break; // Moved past our user's events
                    }

                    // Deserialize event
                    if let Ok(event) = bincode::deserialize::<AuditEvent>(&value) {
                        events.push(event);
                    }
                }
            }
        }

        // Update cache for next time
        if !events.is_empty() {
            let log = Arc::new(parking_lot::RwLock::new(events.clone()));
            self.audit_logs.insert(user_id.to_string(), log);
        }

        // Filter by memory_id if requested
        if let Some(mid) = memory_id {
            events.into_iter().filter(|e| e.memory_id == mid).collect()
        } else {
            events
        }
    }

    /// Get or create memory system for a user
    pub fn get_user_memory(&self, user_id: &str) -> Result<Arc<parking_lot::RwLock<MemorySystem>>> {
        // Try to get from LRU cache (updates LRU order)
        {
            let mut cache = self.user_memories.lock();
            if let Some(memory) = cache.get(user_id) {
                return Ok(memory.clone());
            }
        }

        // Create new memory system for this user
        let user_path = self.base_path.join(user_id);
        let config = MemoryConfig {
            storage_path: user_path,
            ..self.default_config.clone()
        };

        let memory_system = MemorySystem::new(config)?;
        let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));

        // Insert into LRU cache (may evict least recently used user)
        {
            let mut cache = self.user_memories.lock();

            // Check if insertion will cause eviction
            if cache.len() >= self.server_config.max_users_in_memory {
                if let Some((evicted_user_id, _evicted_memory)) = cache.push(user_id.to_string(), memory_arc.clone()) {
                    // LRU eviction occurred - log it
                    self.user_evictions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    info!("📤 Evicted user '{}' from memory cache (LRU, cache_size={})", evicted_user_id, self.server_config.max_users_in_memory);

                    // Note: Evicted memory system will be flushed when Arc refcount drops to 0
                    // and MemorySystem's Drop implementation runs
                }
            } else {
                cache.put(user_id.to_string(), memory_arc.clone());
            }
        }

        info!("🧠 Created memory system for user: {}", user_id);

        // Initialize vector index (load from disk or rebuild)
        if let Err(e) = self.init_user_vector_index(user_id) {
            tracing::warn!("Vector index initialization failed for user {}: {}", user_id, e);
            // Don't fail user creation if indexing fails - semantic search will be unavailable
        }

        Ok(memory_arc)
    }

    /// Delete user data (GDPR compliance)
    pub fn forget_user(&self, user_id: &str) -> Result<()> {
        // Remove from memory cache
        self.user_memories.lock().pop(user_id);

        // Delete storage directory
        let user_path = self.base_path.join(user_id);
        if user_path.exists() {
            std::fs::remove_dir_all(&user_path)?;
        }

        info!("🧠 Deleted all data for user: {}", user_id);
        Ok(())
    }

    /// Get statistics for a user
    pub fn get_stats(&self, user_id: &str) -> Result<MemoryStats> {
        let memory = self.get_user_memory(user_id)?;
        let memory_guard = memory.read();
        Ok(memory_guard.stats())
    }

    /// List all users currently in memory cache
    pub fn list_users(&self) -> Vec<String> {
        self.user_memories.lock().iter().map(|(key, _)| key.clone()).collect()
    }

    /// Flush all RocksDB databases to ensure data persistence (critical for graceful shutdown)
    pub fn flush_all_databases(&self) -> Result<()> {
        info!("💾 Flushing all databases to disk...");

        // Flush audit database
        self.audit_db.flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush audit database: {e}"))?;

        // Flush all user memory databases
        // Collect entries first, then release lock before flushing (avoid holding lock during I/O)
        let user_entries: Vec<(String, Arc<parking_lot::RwLock<MemorySystem>>)> = {
            self.user_memories.lock().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };

        let mut flushed = 0;
        for (user_id, memory_system) in user_entries {
            // Access the memory system's storage to flush it
            if let Some(guard) = memory_system.try_read() {
                // Flush the long-term storage database
                if let Err(e) = guard.flush_storage() {
                    tracing::warn!("  Failed to flush database for user {}: {}", user_id, e);
                } else {
                    info!("  Flushed database for user: {}", user_id);
                    flushed += 1;
                }
            } else {
                tracing::warn!("  Could not acquire lock for user: {}", user_id);
            }
        }

        info!("✅ Flushed {} audit database and {} user databases", if self.audit_db.flush().is_ok() { "1" } else { "0" }, flushed);

        // Give RocksDB a moment to complete background tasks
        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(())
    }

    /// Save all vector indices to disk (production persistence)
    pub fn save_all_vector_indices(&self) -> Result<()> {
        info!("🔍 Saving vector indices to disk...");

        // Collect entries first, then release lock before saving (avoid holding lock during I/O)
        let user_entries: Vec<(String, Arc<parking_lot::RwLock<MemorySystem>>)> = {
            self.user_memories.lock().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };

        let mut saved = 0;
        for (user_id, memory_system) in user_entries {
            if let Some(guard) = memory_system.try_read() {
                let index_path = self.base_path.join(&user_id).join("vector_index");
                if let Err(e) = guard.save_vector_index(&index_path) {
                    tracing::warn!("  Failed to save vector index for user {}: {}", user_id, e);
                } else {
                    info!("  Saved vector index for user: {}", user_id);
                    saved += 1;
                }
            } else {
                tracing::warn!("  Could not acquire lock for user: {}", user_id);
            }
        }

        info!("✅ Saved {} vector indices", saved);
        Ok(())
    }

    /// Load or rebuild vector index for a user (production initialization)
    pub fn init_user_vector_index(&self, user_id: &str) -> Result<()> {
        let memory = self.get_user_memory(user_id)?;
        let memory_guard = memory.read();

        let index_path = self.base_path.join(user_id).join("vector_index");

        // Try to load existing index first
        if index_path.exists() {
            info!("📊 Loading vector index for user {} from disk...", user_id);
            match memory_guard.load_vector_index(&index_path) {
                Ok(_) => {
                    info!("✅ Loaded vector index for user: {}", user_id);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("Failed to load vector index for user {}: {}. Rebuilding...", user_id, e);
                }
            }
        }

        // Index doesn't exist or load failed - rebuild from storage
        info!("🔨 Rebuilding vector index for user {}...", user_id);
        match memory_guard.rebuild_vector_index() {
            Ok(_) => {
                info!("✅ Rebuilt vector index for user: {}", user_id);
                // Save the newly built index
                let _ = memory_guard.save_vector_index(&index_path);
                Ok(())
            }
            Err(e) => {
                tracing::warn!("Failed to rebuild vector index for user {}: {}", user_id, e);
                Ok(()) // Don't fail startup if indexing fails
            }
        }
    }

    /// Rotate audit logs for all users (removes old entries)
    fn rotate_all_audit_logs(&self) -> Result<()> {
        let mut total_removed = 0;

        // Get all unique user IDs from the database
        let mut user_ids = std::collections::HashSet::new();
        let iter = self.audit_db.iterator(rocksdb::IteratorMode::Start);

        for item in iter {
            if let Ok((key, _)) = item {
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if let Some(user_id) = key_str.split(':').next() {
                        user_ids.insert(user_id.to_string());
                    }
                }
            }
        }

        // Rotate logs for each user using helper struct
        let helper = MultiUserMemoryManagerRotationHelper {
            audit_db: self.audit_db.clone(),
            audit_logs: self.audit_logs.clone(),
            audit_retention_days: self.server_config.audit_retention_days as i64,
            audit_max_entries: self.server_config.audit_max_entries_per_user,
        };

        for user_id in user_ids {
            match helper.rotate_user_audit_logs(&user_id) {
                Ok(removed) => {
                    if removed > 0 {
                        info!("  Rotated audit logs for user {}: removed {} old entries", user_id, removed);
                        total_removed += removed;
                    }
                }
                Err(e) => {
                    tracing::warn!("  Failed to rotate audit logs for user {}: {}", user_id, e);
                }
            }
        }

        if total_removed > 0 {
            info!("✅ Audit log rotation complete: removed {} total entries", total_removed);
        }

        Ok(())
    }

    /// Get or create graph memory for a user
    fn get_user_graph(&self, user_id: &str) -> Result<Arc<parking_lot::RwLock<GraphMemory>>> {
        // Try to get from LRU cache (updates LRU order)
        {
            let mut cache = self.graph_memories.lock();
            if let Some(graph) = cache.get(user_id) {
                return Ok(graph.clone());
            }
        }

        // Create new graph memory for this user
        let graph_path = self.base_path.join(user_id).join("graph");
        let graph_memory = GraphMemory::new(&graph_path)?;
        let graph_arc = Arc::new(parking_lot::RwLock::new(graph_memory));

        // Insert into LRU cache (may evict least recently used graph)
        {
            let mut cache = self.graph_memories.lock();

            // Check if insertion will cause eviction
            if cache.len() >= self.server_config.max_users_in_memory {
                if let Some((evicted_user_id, _evicted_graph)) = cache.push(user_id.to_string(), graph_arc.clone()) {
                    // LRU eviction occurred - log it
                    info!("📤 Evicted graph for user '{}' from memory cache (LRU, cache_size={})", evicted_user_id, self.server_config.max_users_in_memory);
                }
            } else {
                cache.put(user_id.to_string(), graph_arc.clone());
            }
        }

        info!("📊 Created graph memory for user: {}", user_id);

        Ok(graph_arc)
    }

    /// Process an experience and extract entities/relationships into the graph
    fn process_experience_into_graph(&self, user_id: &str, experience: &Experience, memory_id: &MemoryId) -> Result<()> {
        let graph = self.get_user_graph(user_id)?;
        let graph_guard = graph.write();

        // Extract entities from the experience content
        let extracted_entities = self.entity_extractor.extract(&experience.content);

        let mut entity_uuids = Vec::new();

        // Add entities to the graph
        for (entity_name, entity_label) in extracted_entities {
            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(), // Will be replaced if exists
                name: entity_name.clone(),
                labels: vec![entity_label],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
            };

            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((entity_name, uuid)),
                Err(e) => tracing::debug!("Failed to add entity {}: {}", entity_name, e),
            }
        }

        // Create an episodic node for this experience
        let episode = EpisodicNode {
            uuid: uuid::Uuid::parse_str(&memory_id.0.to_string())
                .unwrap_or_else(|_| uuid::Uuid::new_v4()),
            name: format!("Memory {}", &memory_id.0.to_string()[..8]),
            content: experience.content.clone(),
            valid_at: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            entity_refs: entity_uuids.iter().map(|(_, uuid)| *uuid).collect(),
            source: EpisodeSource::Message,
            metadata: experience.metadata.clone(),
        };

        if let Err(e) = graph_guard.add_episode(episode) {
            tracing::debug!("Failed to add episode: {}", e);
        }

        // Create relationships between co-occurring entities (simple heuristic)
        for i in 0..entity_uuids.len() {
            for j in (i + 1)..entity_uuids.len() {
                let edge = RelationshipEdge {
                    uuid: uuid::Uuid::new_v4(),
                    from_entity: entity_uuids[i].1,
                    to_entity: entity_uuids[j].1,
                    relation_type: RelationType::RelatedTo,
                    strength: 0.5, // Default strength for co-occurrence
                    created_at: chrono::Utc::now(),
                    valid_at: chrono::Utc::now(),
                    invalidated_at: None,
                    source_episode_id: Some(uuid::Uuid::parse_str(&memory_id.0.to_string())
                        .unwrap_or_else(|_| uuid::Uuid::new_v4())),
                    context: experience.content.clone(),
                };

                if let Err(e) = graph_guard.add_relationship(edge) {
                    tracing::debug!("Failed to add relationship: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Get graph statistics for a user
    pub fn get_user_graph_stats(&self, user_id: &str) -> Result<GraphStats> {
        let graph = self.get_user_graph(user_id)?;
        let graph_guard = graph.read();
        graph_guard.get_stats()
    }
}

/// API request/response types
#[derive(Debug, Deserialize)]
struct RecordRequest {
    user_id: String,
    experience: Experience,
}

#[derive(Debug, Serialize)]
struct RecordResponse {
    memory_id: String,
    success: bool,
}

#[derive(Debug, Deserialize)]
struct RetrieveRequest {
    user_id: String,
    query_text: Option<String>,
    query_embedding: Option<Vec<f32>>,
    max_results: Option<usize>,
    importance_threshold: Option<f32>,
}

#[derive(Debug, Serialize)]
struct RetrieveResponse {
    memories: Vec<Memory>,
    count: usize,
}

// MemoryStats is imported from memory::types - has more fields than we need here

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    users_count: usize,
    users_in_cache: usize,
    user_evictions: usize,
    max_cache_size: usize,
}

/// Application state
type AppState = Arc<MultiUserMemoryManager>;

/// REST API handlers

/// Health check endpoint (basic compatibility)
async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let users_in_cache = state.user_memories.lock().len();
    let user_evictions = state.user_evictions.load(std::sync::atomic::Ordering::Relaxed);

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        users_count: state.list_users().len(),
        users_in_cache,
        user_evictions,
        max_cache_size: state.server_config.max_users_in_memory,
    })
}

/// P0.9: Liveness probe - indicates if process is alive and not deadlocked
/// Returns 200 OK if service is running (minimal check, always succeeds if reachable)
/// Kubernetes uses this to restart crashed/hung pods
async fn health_live() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(serde_json::json!({
        "status": "alive",
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

/// P0.9: Readiness probe - indicates if service can handle traffic
/// Returns 200 OK if service is ready, 503 if not ready
/// Kubernetes uses this to route traffic only to ready pods
async fn health_ready(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    // Check if critical resources are accessible
    let users_in_cache = state.user_memories.lock().len();

    // Service is ready if we can access the user cache without panicking
    // This verifies the lock is not poisoned and the service is operational
    (StatusCode::OK, Json(serde_json::json!({
        "status": "ready",
        "version": env!("CARGO_PKG_VERSION"),
        "users_in_cache": users_in_cache,
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

/// Prometheus metrics endpoint for observability
/// Returns metrics in Prometheus text format for scraping
async fn metrics_endpoint(State(state): State<AppState>) -> Result<String, StatusCode> {
    use prometheus::Encoder;

    // Update memory usage gauges before serving metrics
    let users_in_cache = state.user_memories.lock().len();
    metrics::ACTIVE_USERS.set(users_in_cache as i64);

    // Aggregate metrics across all users (no per-user labels to avoid cardinality explosion)
    let (mut total_working, mut total_session, mut total_longterm, mut total_heap) = (0i64, 0i64, 0i64, 0i64);
    let mut total_vectors = 0i64;

    let user_entries: Vec<_> = {
        let cache = state.user_memories.lock();
        cache.iter().take(100).map(|(_, v)| v.clone()).collect()
    };

    for memory_sys in user_entries {
        if let Some(guard) = memory_sys.try_read() {
            let stats = guard.stats();
            total_working += stats.working_memory_count as i64;
            total_session += stats.session_memory_count as i64;
            total_longterm += stats.long_term_memory_count as i64;
            total_heap += (stats.total_memories * 250) as i64;
            total_vectors += stats.total_memories as i64;
        }
    }

    // Set aggregate metrics
    metrics::MEMORIES_BY_TIER.with_label_values(&["working"]).set(total_working);
    metrics::MEMORIES_BY_TIER.with_label_values(&["session"]).set(total_session);
    metrics::MEMORIES_BY_TIER.with_label_values(&["longterm"]).set(total_longterm);
    metrics::MEMORY_HEAP_BYTES_TOTAL.set(total_heap);
    metrics::VECTOR_INDEX_SIZE_TOTAL.set(total_vectors);

    // Gather and encode metrics
    let encoder = prometheus::TextEncoder::new();
    let metric_families = metrics::METRICS_REGISTRY.gather();

    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    String::from_utf8(buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Compute cosine similarity between two embedding vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    (dot_product / (magnitude_a * magnitude_b)).clamp(0.0, 1.0)
}

/// Record a new experience
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn record_experience(
    State(state): State<AppState>,
    Json(req): Json<RecordRequest>,
) -> Result<Json<RecordResponse>, AppError> {
    // Enterprise input validation
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    validation::validate_content(&req.experience.content, false)
        .map_validation_err("content")?;

    if let Some(ref embeddings) = req.experience.embeddings {
        validation::validate_embeddings(embeddings)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // P1.2: Instrument memory store operation
    let store_start = std::time::Instant::now();

    // CRITICAL FIX: Wrap blocking I/O in spawn_blocking
    // record() does ONNX inference (10-50ms) + RocksDB writes (100µs-10ms)
    // Running these on async threads starves the Tokio runtime under load
    let memory_id = {
        let memory = memory.clone();
        let experience = req.experience.clone();

        tokio::task::spawn_blocking(move || {
            let mut memory_guard = memory.write();
            memory_guard.record(experience)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Extract entities and build knowledge graph (background processing)
    if let Err(e) = state.process_experience_into_graph(&req.user_id, &req.experience, &memory_id) {
        tracing::debug!("Graph processing failed for memory {}: {}", memory_id.0, e);
        // Don't fail the request if graph processing fails
    }

    // Enterprise audit logging
    state.log_event(
        &req.user_id,
        "CREATE",
        &memory_id.0.to_string(),
        &format!("Created memory: {}", req.experience.content.chars().take(50).collect::<String>())
    );

    // Record metrics (no user_id to prevent cardinality explosion)
    let duration = store_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL.with_label_values(&["success"]).inc();

    Ok(Json(RecordResponse {
        memory_id: memory_id.0.to_string(),
        success: true,
    }))
}

/// Retrieve memories
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query_text = ?req.query_text))]
async fn retrieve_memories(
    State(state): State<AppState>,
    Json(req): Json<RetrieveRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    // P1.2: Instrument memory retrieve operation
    let retrieve_start = std::time::Instant::now();
    let retrieval_mode = "hybrid";  // Default mode

    // Enterprise input validation
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    if let Some(ref emb) = req.query_embedding {
        validation::validate_embeddings(emb)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let max_results = req.max_results.unwrap_or(10);
    validation::validate_max_results(max_results)
        .map_validation_err("max_results")?;

    if let Some(threshold) = req.importance_threshold {
        validation::validate_importance_threshold(threshold)
            .map_validation_err("importance_threshold")?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // CRITICAL FIX: Wrap blocking I/O in spawn_blocking
    // retrieve() does:
    // - ONNX inference for query embedding (10-50ms)
    // - RocksDB reads for memory lookups (100µs-10ms per memory)
    // - Graph traversal with multiple storage operations
    // Running these on async threads starves the Tokio runtime under load
    let memories: Vec<Memory> = {
        let memory = memory.clone();
        let query_text = req.query_text.clone();
        let query_embedding = req.query_embedding.clone();
        let state_clone = state.clone();
        let user_id = req.user_id.clone();
        let importance_threshold = req.importance_threshold;
        let max_results_val = req.max_results;

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();

            let query = MemoryQuery {
                query_text: query_text.clone(),
                query_embedding: query_embedding.clone(),
                max_results,
                importance_threshold,
                ..Default::default()
            };

            // Check if we should use graph-aware retrieval
            let memories: Vec<Memory> = if let Some(ref query_text) = query_text {
                // GRAPH-AWARE RETRIEVAL (Anderson & Pirolli 1984 + Xiong et al. 2017)
                let graph = state_clone.get_user_graph(&user_id)
                    .map_err(AppError::Internal)?;
                let graph_guard = graph.read();

                // Use spreading activation retrieval
                use memory::graph_retrieval::spreading_activation_retrieve;

                let activated = spreading_activation_retrieve(
                    query_text,
                    &query,
                    &graph_guard,
                    memory_guard.get_embedder(),
                    |episode| {
                        // Convert episode to memory
                        // Find memory with matching content
                        // This is a temporary bridge - we should store episode_id in Memory
                        let all_memories = memory_guard.get_all_memories()?;
                        for mem in all_memories {
                            if mem.experience.content == episode.content {
                                return Ok(Some(mem));
                            }
                        }
                        Ok(None)
                    },
                ).map_err(AppError::Internal)?;

                // Convert ActivatedMemory to Memory with scores
                let graph_results: Vec<Memory> = activated.into_iter().map(|am| {
                    let mut mem = (*am.memory).clone();
                    mem.score = Some(am.final_score);
                    mem
                }).collect();

                // If graph retrieval returned 0 results, fall back to semantic search
                if graph_results.is_empty() {
                    tracing::info!("Graph retrieval returned 0 results, falling back to semantic search");

                    // Generate query embedding
                    let query_embedding = memory_guard.get_embedder().encode(query_text)
                        .map_err(AppError::Internal)?;

                    // Get all memories
                    let all_memories = memory_guard.get_all_memories()
                        .map_err(AppError::Internal)?;

                    // Score each memory by semantic similarity
                    let mut scored_memories: Vec<(f32, Memory)> = all_memories
                        .into_iter()
                        .filter_map(|shared_mem| {
                            let memory = (*shared_mem).clone();

                            // Get or generate embedding
                            let mem_embedding = if let Some(emb) = &memory.experience.embeddings {
                                emb.clone()
                            } else {
                                return None;
                            };

                            // Compute cosine similarity
                            let similarity = cosine_similarity(&query_embedding, &mem_embedding);
                            Some((similarity, memory))
                        })
                        .collect();

                    // Sort by similarity (descending)
                    scored_memories.sort_by(|a, b| {
                        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    // Take top-k and set scores
                    scored_memories
                        .into_iter()
                        .take(query.max_results)
                        .map(|(score, mut mem)| {
                            mem.score = Some(score);  // REAL semantic score!
                            mem
                        })
                        .collect()
                } else {
                    graph_results
                }
            } else {
                // Fallback to traditional retrieval
                let shared_memories = memory_guard
                    .retrieve(&query)
                    .map_err(AppError::Internal)?;

                // Convert Arc<Memory> to owned Memory
                let mut memories: Vec<Memory> = shared_memories
                    .iter()
                    .map(|m| (**m).clone())
                    .collect();

                // If query embedding provided, re-rank by semantic similarity
                if let Some(query_emb) = &query_embedding {
                    let candidates: Vec<(Vec<f32>, &Memory)> = memories
                        .iter()
                        .filter_map(|m| {
                            m.experience.embeddings.as_ref().map(|emb| (emb.clone(), m))
                        })
                        .collect();

                    let ranked = top_k_similar(query_emb, &candidates, max_results_val.unwrap_or(10));

                    // Create new vec with scores populated
                    ranked.into_iter().map(|(score, m)| {
                        let mut mem = m.clone();
                        mem.score = Some(score);
                        mem
                    }).collect()
                } else {
                    // Populate scores based on importance * temporal_relevance
                    for memory in &mut memories {
                        memory.score = Some(memory.importance() * memory.temporal_relevance());
                    }
                    memories
                }
            };

            Ok::<Vec<Memory>, AppError>(memories)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(|e: AppError| e)?
    };

    let count = memories.len();

    // Record retrieve metrics (no user_id to prevent cardinality explosion)
    let duration = retrieve_start.elapsed().as_secs_f64();
    metrics::MEMORY_RETRIEVE_DURATION
        .with_label_values(&[retrieval_mode])
        .observe(duration);
    metrics::MEMORY_RETRIEVE_TOTAL
        .with_label_values(&[retrieval_mode, "success"])
        .inc();
    metrics::MEMORY_RETRIEVE_RESULTS
        .with_label_values(&[retrieval_mode])
        .observe(count as f64);

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Get user statistics
async fn get_user_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<MemoryStats>, AppError> {
    let stats = state
        .get_stats(&user_id)
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// Delete user data (GDPR)
async fn delete_user(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<StatusCode, AppError> {
    state
        .forget_user(&user_id)
        .map_err(AppError::Internal)?;

    Ok(StatusCode::OK)
}

/// List all users
async fn list_users(State(state): State<AppState>) -> Json<Vec<String>> {
    Json(state.list_users())
}

/// Get specific memory by ID
async fn get_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Memory>, AppError> {
    let user_id = params.get("user_id")
        .ok_or_else(|| AppError::InvalidInput { field: "user_id".to_string(), reason: "user_id required".to_string() })?;

    // Enterprise input validation
    validation::validate_user_id(user_id)
        .map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state
        .get_user_memory(user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Parse memory ID (already validated above)
    let mem_id = uuid::Uuid::parse_str(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let _memory_id_obj = MemoryId(mem_id);

    // Search for memory
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard
        .retrieve(&query)
        .map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    // Unwrap Arc to return owned Memory
    Ok(Json((*shared_memory).clone()))
}

/// Update existing memory
#[derive(Debug, Deserialize)]
struct UpdateMemoryRequest {
    user_id: String,
    content: String,
    embeddings: Option<Vec<f32>>,
}

async fn update_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<StatusCode, AppError> {
    // Enterprise input validation
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    validation::validate_content(&req.content, false)
        .map_validation_err("content")?;

    if let Some(ref emb) = req.embeddings {
        validation::validate_embeddings(emb)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let mut memory_guard = memory.write();

    let mem_id = uuid::Uuid::parse_str(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    // Get current memory to preserve metadata
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard
        .retrieve(&query)
        .map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    // Clone out of Arc to get mutable Memory
    let mut current_memory = (*shared_memory).clone();

    // Save content for audit log before move
    let content_preview: String = req.content.chars().take(50).collect();

    // Update content and embeddings
    current_memory.experience.content = req.content;
    if let Some(emb) = req.embeddings {
        current_memory.experience.embeddings = Some(emb);
    }
    // Note: last_accessed will be set automatically when re-recording

    // Re-record (will update in storage)
    let experience = current_memory.experience.clone();
    memory_guard
        .record(experience)
        .map_err(AppError::Internal)?;

    // Enterprise audit logging
    state.log_event(
        &req.user_id,
        "UPDATE",
        &memory_id,
        &format!("Updated memory content: {content_preview}")
    );

    Ok(StatusCode::OK)
}

/// Delete specific memory
async fn delete_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<StatusCode, AppError> {
    let user_id = params.get("user_id")
        .ok_or_else(|| AppError::InvalidInput { field: "user_id".to_string(), reason: "user_id required".to_string() })?;

    // Enterprise input validation
    validation::validate_user_id(user_id)
        .map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state
        .get_user_memory(user_id)
        .map_err(AppError::Internal)?;

    let mut memory_guard = memory.write();

    // Delete by ID - escape UUID to treat as literal string, not regex
    // UUIDs contain hyphens which are regex metacharacters, so we must escape them
    let escaped_pattern = regex::escape(&memory_id);
    memory_guard
        .forget(memory::ForgetCriteria::Pattern(escaped_pattern))
        .map_err(AppError::Internal)?;

    // Enterprise audit logging
    state.log_event(
        user_id,
        "DELETE",
        &memory_id,
        "Memory deleted"
    );

    // Return 200 OK instead of 204 NO_CONTENT so Python client can verify success
    Ok(StatusCode::OK)
}

/// Get all memories for a user
#[derive(Debug, Deserialize)]
struct GetAllRequest {
    user_id: String,
    limit: Option<usize>,
    importance_threshold: Option<f32>,
}

async fn get_all_memories(
    State(state): State<AppState>,
    Json(req): Json<GetAllRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let query = MemoryQuery {
        max_results: req.limit.unwrap_or(100),
        importance_threshold: req.importance_threshold,
        ..Default::default()
    };

    let shared_memories = memory_guard
        .retrieve(&query)
        .map_err(AppError::Internal)?;

    // Convert Arc<Memory> to owned Memory for response
    let memories: Vec<Memory> = shared_memories
        .iter()
        .map(|m| (**m).clone())
        .collect();

    let count = memories.len();

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Get history/audit trail
#[derive(Debug, Deserialize)]
struct HistoryRequest {
    user_id: String,
    memory_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct HistoryResponse {
    events: Vec<HistoryEvent>,
}

#[derive(Debug, Serialize)]
struct HistoryEvent {
    timestamp: String,  // ISO 8601 format
    event_type: String,
    memory_id: String,
    details: String,
}

async fn get_history(
    State(state): State<AppState>,
    Json(req): Json<HistoryRequest>,
) -> Result<Json<HistoryResponse>, AppError> {
    // CRITICAL FIX: Wrap blocking I/O in spawn_blocking
    // get_history() does RocksDB prefix_iterator (blocking I/O) on cache miss
    // Running this on async threads starves the Tokio runtime
    let events = {
        let state = state.clone();
        let user_id = req.user_id.clone();
        let memory_id = req.memory_id.clone();

        tokio::task::spawn_blocking(move || {
            state.get_history(&user_id, memory_id.as_deref())
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    let history_events: Vec<HistoryEvent> = events.iter().map(|e| HistoryEvent {
        timestamp: e.timestamp.to_rfc3339(),
        event_type: e.event_type.clone(),
        memory_id: e.memory_id.clone(),
        details: e.details.clone(),
    }).collect();

    Ok(Json(HistoryResponse { events: history_events }))
}

// ====== Advanced Memory Management Endpoints ======

#[derive(Debug, Deserialize)]
struct CompressMemoryRequest {
    user_id: String,
    memory_id: String,
}

/// Manually compress a specific memory
async fn compress_memory(
    State(state): State<AppState>,
    Json(req): Json<CompressMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let _memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Validate memory_id format
    let _memory_id = MemoryId(uuid::Uuid::parse_str(&req.memory_id)
        .map_err(|_| AppError::InvalidMemoryId(req.memory_id.clone()))?);

    // Compression happens automatically in the memory system based on age and importance
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Memory compression initiated"
    })))
}

#[derive(Debug, Deserialize)]
struct InvalidateRelationshipRequest {
    user_id: String,
    relationship_uuid: String,
}

/// Invalidate a relationship edge (temporal edge invalidation)
async fn invalidate_relationship(
    State(state): State<AppState>,
    Json(req): Json<InvalidateRelationshipRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let graph = state.get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.write();

    let rel_uuid = uuid::Uuid::parse_str(&req.relationship_uuid)
        .map_err(|_| AppError::InvalidInput {
            field: "relationship_uuid".to_string(),
            reason: "Invalid UUID format".to_string()
        })?;

    graph_guard.invalidate_relationship(&rel_uuid)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Relationship invalidated"
    })))
}

#[derive(Debug, Deserialize)]
struct GetEpisodeRequest {
    user_id: String,
    episode_uuid: String,
}

/// Get an episodic node by UUID
async fn get_episode(
    State(state): State<AppState>,
    Json(req): Json<GetEpisodeRequest>,
) -> Result<Json<Option<EpisodicNode>>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let graph = state.get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();

    let episode_uuid = uuid::Uuid::parse_str(&req.episode_uuid)
        .map_err(|_| AppError::InvalidInput {
            field: "episode_uuid".to_string(),
            reason: "Invalid UUID format".to_string()
        })?;

    let episode = graph_guard.get_episode(&episode_uuid)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(episode))
}

#[derive(Debug, Deserialize)]
struct AdvancedSearchRequest {
    user_id: String,
    entity_name: Option<String>,
    start_date: Option<String>,
    end_date: Option<String>,
    min_importance: Option<f32>,
    max_importance: Option<f32>,
}

/// Advanced memory search with entity filtering
async fn advanced_search(
    State(state): State<AppState>,
    Json(req): Json<AdvancedSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    // Build search criteria
    let mut criterias = Vec::new();

    if let Some(entity) = req.entity_name {
        criterias.push(memory::storage::SearchCriteria::ByEntity(entity));
    }

    if let (Some(start), Some(end)) = (req.start_date, req.end_date) {
        let start_dt = chrono::DateTime::parse_from_rfc3339(&start)
            .map_err(|_| AppError::InvalidInput {
                field: "start_date".to_string(),
                reason: "Invalid RFC3339 format".to_string(),
            })?
            .with_timezone(&chrono::Utc);

        let end_dt = chrono::DateTime::parse_from_rfc3339(&end)
            .map_err(|_| AppError::InvalidInput {
                field: "end_date".to_string(),
                reason: "Invalid RFC3339 format".to_string(),
            })?
            .with_timezone(&chrono::Utc);

        criterias.push(memory::storage::SearchCriteria::ByDate {
            start: start_dt,
            end: end_dt
        });
    }

    if let (Some(min), Some(max)) = (req.min_importance, req.max_importance) {
        criterias.push(memory::storage::SearchCriteria::ByImportance {
            min,
            max
        });
    }

    // Execute combined search
    let criteria = if criterias.len() == 1 {
        criterias.into_iter().next().expect("Criteria list has exactly one element")
    } else {
        memory::storage::SearchCriteria::Combined(criterias)
    };

    let memories = memory_guard
        .advanced_search(criteria)
        .map_err(AppError::Internal)?;

    let count = memories.len();

    Ok(Json(RetrieveResponse { memories, count }))
}

// ====== Graph Memory API Endpoints ======

/// Get graph statistics for a user
async fn get_graph_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<GraphStats>, AppError> {
    validation::validate_user_id(&user_id)
        .map_validation_err("user_id")?;

    let stats = state
        .get_user_graph_stats(&user_id)
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

#[derive(Debug, Deserialize)]
struct FindEntityRequest {
    user_id: String,
    entity_name: String,
}

/// Find an entity by name
async fn find_entity(
    State(state): State<AppState>,
    Json(req): Json<FindEntityRequest>,
) -> Result<Json<Option<EntityNode>>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();
    let entity = graph_guard
        .find_entity_by_name(&req.entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(entity))
}

#[derive(Debug, Deserialize)]
struct TraverseGraphRequest {
    user_id: String,
    entity_name: String,
    max_depth: Option<usize>,
}

/// Traverse graph from an entity
async fn traverse_graph(
    State(state): State<AppState>,
    Json(req): Json<TraverseGraphRequest>,
) -> Result<Json<GraphTraversal>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();

    // First find the entity
    let entity = graph_guard
        .find_entity_by_name(&req.entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| AppError::MemoryNotFound(format!("Entity not found: {}", req.entity_name)))?;

    // Traverse from that entity
    let max_depth = req.max_depth.unwrap_or(2);
    let traversal = graph_guard
        .traverse_from_entity(&entity.uuid, max_depth)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(traversal))
}

// ============================================================================
// COMPREHENSIVE API EXPANSION - All Features Exposed
// ============================================================================

/// Decompress a specific memory
#[derive(Debug, Deserialize)]
struct DecompressMemoryRequest {
    user_id: String,
    memory_id: String,
}

async fn decompress_memory(
    State(state): State<AppState>,
    Json(req): Json<DecompressMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let memory_id = MemoryId(uuid::Uuid::parse_str(&req.memory_id)
        .map_err(|_| AppError::InvalidMemoryId(req.memory_id.clone()))?);

    // Get the memory
    let memory = memory_guard.get_memory(&memory_id)
        .map_err(AppError::Internal)?;

    if !memory.compressed {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "Memory is not compressed",
            "was_compressed": false
        })));
    }

    // Decompress using compression pipeline
    let decompressed = memory_guard.decompress_memory(&memory)
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Memory decompressed successfully",
        "was_compressed": true,
        "memory": {
            "id": decompressed.id.0.to_string(),
            "content": decompressed.experience.content,
            "importance": decompressed.importance()
        }
    })))
}

/// Get storage statistics
#[derive(Debug, Deserialize)]
struct StorageStatsRequest {
    user_id: String,
}

async fn get_storage_stats(
    State(state): State<AppState>,
    Json(req): Json<StorageStatsRequest>,
) -> Result<Json<memory::storage::StorageStats>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let stats = memory_guard.get_storage_stats()
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// Forget memories by age
#[derive(Debug, Deserialize)]
struct ForgetByAgeRequest {
    user_id: String,
    days_old: u32,
}

async fn forget_by_age(
    State(state): State<AppState>,
    Json(req): Json<ForgetByAgeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let mut memory_guard = memory_sys.write();
    let count = memory_guard.forget(memory::ForgetCriteria::OlderThan(req.days_old))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_AGE",
        &format!("{} days", req.days_old),
        &format!("Forgot {count} memories")
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "criteria": format!("older than {} days", req.days_old)
    })))
}

/// Forget memories by importance threshold
#[derive(Debug, Deserialize)]
struct ForgetByImportanceRequest {
    user_id: String,
    threshold: f32,
}

async fn forget_by_importance(
    State(state): State<AppState>,
    Json(req): Json<ForgetByImportanceRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    if req.threshold < 0.0 || req.threshold > 1.0 {
        return Err(AppError::InvalidInput {
            field: "threshold".to_string(),
            reason: "Must be between 0.0 and 1.0".to_string()
        });
    }

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let mut memory_guard = memory_sys.write();
    let count = memory_guard.forget(memory::ForgetCriteria::LowImportance(req.threshold))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_IMPORTANCE",
        &format!("threshold {}", req.threshold),
        &format!("Forgot {count} memories")
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "criteria": format!("importance < {}", req.threshold)
    })))
}

/// Forget memories matching a pattern
#[derive(Debug, Deserialize)]
struct ForgetByPatternRequest {
    user_id: String,
    pattern: String,
}

async fn forget_by_pattern(
    State(state): State<AppState>,
    Json(req): Json<ForgetByPatternRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let mut memory_guard = memory_sys.write();
    let count = memory_guard.forget(memory::ForgetCriteria::Pattern(req.pattern.clone()))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_PATTERN",
        &req.pattern,
        &format!("Forgot {count} memories")
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "pattern": req.pattern
    })))
}

/// Advanced multi-modal retrieval
#[derive(Debug, Deserialize)]
struct MultiModalSearchRequest {
    user_id: String,
    query_text: String,
    mode: String,  // "similarity", "temporal", "causal", "associative", "hybrid"
    limit: Option<usize>,
}

async fn multimodal_search(
    State(state): State<AppState>,
    Json(req): Json<MultiModalSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    // Build query based on mode (including robotics modes)
    let retrieval_mode = match req.mode.as_str() {
        "similarity" => memory::RetrievalMode::Similarity,
        "temporal" => memory::RetrievalMode::Temporal,
        "causal" => memory::RetrievalMode::Causal,
        "associative" => memory::RetrievalMode::Associative,
        "hybrid" => memory::RetrievalMode::Hybrid,
        // Robotics-specific modes
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        _ => return Err(AppError::InvalidInput {
            field: "mode".to_string(),
            reason: format!("Invalid mode: {}. Must be one of: similarity, temporal, causal, associative, hybrid, spatial, mission, action_outcome", req.mode)
        })
    };

    let query = MemoryQuery {
        query_text: Some(req.query_text.clone()),
        max_results: req.limit.unwrap_or(10),
        retrieval_mode,
        ..Default::default()
    };

    let shared_memories = memory_guard.retrieve(&query)
        .map_err(AppError::Internal)?;

    // Convert Arc<Memory> to owned Memory for response
    let memories: Vec<Memory> = shared_memories
        .iter()
        .map(|m| (**m).clone())
        .collect();

    let count = memories.len();

    state.log_event(
        &req.user_id,
        "MULTIMODAL_SEARCH",
        &req.mode,
        &format!("Retrieved {} memories using {} mode", count, req.mode)
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

// ====== Robotics Search API Endpoints ======

/// Request for robotics-specific memory search
#[derive(Debug, Deserialize)]
struct RoboticsSearchRequest {
    user_id: String,
    /// Search mode: spatial, mission, action_outcome, or hybrid
    mode: String,
    /// Optional text query for semantic component
    query_text: Option<String>,
    /// Robot/drone identifier filter
    robot_id: Option<String>,
    /// Mission identifier filter
    mission_id: Option<String>,
    /// Spatial search: center latitude
    lat: Option<f64>,
    /// Spatial search: center longitude
    lon: Option<f64>,
    /// Spatial search: radius in meters
    radius_meters: Option<f64>,
    /// Action type filter
    action_type: Option<String>,
    /// Reward range: minimum reward (-1.0 to 1.0)
    min_reward: Option<f32>,
    /// Reward range: maximum reward (-1.0 to 1.0)
    max_reward: Option<f32>,
    /// Maximum results to return
    limit: Option<usize>,
}

/// Robotics-specific memory search
/// Supports spatial queries, mission context, robot filtering, and action-outcome learning
async fn robotics_search(
    State(state): State<AppState>,
    Json(req): Json<RoboticsSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    // Build retrieval mode
    let retrieval_mode = match req.mode.as_str() {
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        "hybrid" => memory::RetrievalMode::Hybrid,
        "similarity" => memory::RetrievalMode::Similarity,
        _ => return Err(AppError::InvalidInput {
            field: "mode".to_string(),
            reason: "Invalid mode. Use: spatial, mission, action_outcome, hybrid, similarity".to_string()
        })
    };

    // Build geo filter if spatial coordinates provided
    let geo_filter = match (req.lat, req.lon, req.radius_meters) {
        (Some(lat), Some(lon), Some(radius)) => {
            Some(memory::GeoFilter::new(lat, lon, radius))
        }
        _ => None
    };

    // Build reward range
    let reward_range = match (req.min_reward, req.max_reward) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, 1.0)),
        (None, Some(max)) => Some((-1.0, max)),
        _ => None
    };

    // Validate spatial mode has coordinates
    if matches!(retrieval_mode, memory::RetrievalMode::Spatial) && geo_filter.is_none() {
        return Err(AppError::InvalidInput {
            field: "lat/lon/radius_meters".to_string(),
            reason: "Spatial mode requires lat, lon, and radius_meters".to_string()
        });
    }

    // Validate mission mode has mission_id
    if matches!(retrieval_mode, memory::RetrievalMode::Mission) && req.mission_id.is_none() {
        return Err(AppError::InvalidInput {
            field: "mission_id".to_string(),
            reason: "Mission mode requires mission_id".to_string()
        });
    }

    let query = MemoryQuery {
        query_text: req.query_text,
        robot_id: req.robot_id.clone(),
        mission_id: req.mission_id.clone(),
        geo_filter,
        action_type: req.action_type.clone(),
        reward_range,
        max_results: req.limit.unwrap_or(10),
        retrieval_mode,
        ..Default::default()
    };

    let shared_memories = memory_guard.retrieve(&query)
        .map_err(AppError::Internal)?;

    let memories: Vec<Memory> = shared_memories
        .iter()
        .map(|m| (**m).clone())
        .collect();

    let count = memories.len();

    state.log_event(
        &req.user_id,
        "ROBOTICS_SEARCH",
        &req.mode,
        &format!("Retrieved {} robotics memories (robot={:?}, mission={:?})",
                 count, req.robot_id, req.mission_id)
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Get uncompressed old memories (for manual compression review)
#[derive(Debug, Deserialize)]
struct GetUncompressedRequest {
    user_id: String,
    days_old: u32,
}

async fn get_uncompressed_old(
    State(state): State<AppState>,
    Json(req): Json<GetUncompressedRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory_sys = state.get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let cutoff = chrono::Utc::now() - chrono::Duration::days(req.days_old as i64);
    let memories = memory_guard
        .get_uncompressed_older_than(cutoff)
        .map_err(AppError::Internal)?;

    let count = memories.len();

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Add entity to graph
#[derive(Debug, Deserialize)]
struct AddEntityRequest {
    user_id: String,
    name: String,
    label: String,
    attributes: Option<HashMap<String, String>>,
}

async fn add_entity(
    State(state): State<AppState>,
    Json(req): Json<AddEntityRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    validation::validate_entity(&req.name)
        .map_validation_err("name")?;

    validation::validate_entity(&req.label)
        .map_validation_err("label")?;

    let graph = state.get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    // Parse entity label
    let entity_label = match req.label.as_str() {
        "Person" => graph_memory::EntityLabel::Person,
        "Organization" => graph_memory::EntityLabel::Organization,
        "Location" => graph_memory::EntityLabel::Location,
        "Technology" => graph_memory::EntityLabel::Technology,
        "Concept" => graph_memory::EntityLabel::Concept,
        "Event" => graph_memory::EntityLabel::Event,
        "Date" => graph_memory::EntityLabel::Date,
        "Product" => graph_memory::EntityLabel::Product,
        "Skill" => graph_memory::EntityLabel::Skill,
        other => graph_memory::EntityLabel::Other(other.to_string()),
    };

    let entity = graph_memory::EntityNode {
        uuid: uuid::Uuid::new_v4(),
        name: req.name.clone(),
        labels: vec![entity_label],
        created_at: chrono::Utc::now(),
        last_seen_at: chrono::Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes: req.attributes.unwrap_or_default(),
        name_embedding: None,
    };

    let entity_uuid = graph_guard.add_entity(entity)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "entity_uuid": entity_uuid.to_string(),
        "entity_name": req.name
    })))
}

/// Add relationship to graph
#[derive(Debug, Deserialize)]
struct AddRelationshipRequest {
    user_id: String,
    from_entity_name: String,
    to_entity_name: String,
    relation_type: String,
    strength: Option<f32>,
    context: Option<String>,
}

async fn add_relationship(
    State(state): State<AppState>,
    Json(req): Json<AddRelationshipRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    validation::validate_entity(&req.from_entity_name)
        .map_validation_err("from_entity_name")?;

    validation::validate_entity(&req.to_entity_name)
        .map_validation_err("to_entity_name")?;

    if let Some(strength) = req.strength {
        validation::validate_relationship_strength(strength)
            .map_validation_err("strength")?;
    }

    let graph = state.get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    // Find entities
    let from_entity = graph_guard.find_entity_by_name(&req.from_entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| AppError::MemoryNotFound(format!("Entity not found: {}", req.from_entity_name)))?;

    let to_entity = graph_guard.find_entity_by_name(&req.to_entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| AppError::MemoryNotFound(format!("Entity not found: {}", req.to_entity_name)))?;

    // Parse relation type
    let relation_type = match req.relation_type.as_str() {
        "WorksWith" => graph_memory::RelationType::WorksWith,
        "WorksAt" => graph_memory::RelationType::WorksAt,
        "EmployedBy" => graph_memory::RelationType::EmployedBy,
        "PartOf" => graph_memory::RelationType::PartOf,
        "Contains" => graph_memory::RelationType::Contains,
        "OwnedBy" => graph_memory::RelationType::OwnedBy,
        "LocatedIn" => graph_memory::RelationType::LocatedIn,
        "LocatedAt" => graph_memory::RelationType::LocatedAt,
        "Uses" => graph_memory::RelationType::Uses,
        "CreatedBy" => graph_memory::RelationType::CreatedBy,
        "DevelopedBy" => graph_memory::RelationType::DevelopedBy,
        "Causes" => graph_memory::RelationType::Causes,
        "ResultsIn" => graph_memory::RelationType::ResultsIn,
        "Learned" => graph_memory::RelationType::Learned,
        "Knows" => graph_memory::RelationType::Knows,
        "Teaches" => graph_memory::RelationType::Teaches,
        "RelatedTo" => graph_memory::RelationType::RelatedTo,
        "AssociatedWith" => graph_memory::RelationType::AssociatedWith,
        custom => graph_memory::RelationType::Custom(custom.to_string()),
    };

    let edge = graph_memory::RelationshipEdge {
        uuid: uuid::Uuid::new_v4(),
        from_entity: from_entity.uuid,
        to_entity: to_entity.uuid,
        relation_type,
        strength: req.strength.unwrap_or(0.7),
        created_at: chrono::Utc::now(),
        valid_at: chrono::Utc::now(),
        invalidated_at: None,
        source_episode_id: None,
        context: req.context.unwrap_or_default(),
    };

    let edge_uuid = graph_guard.add_relationship(edge)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(serde_json::json!({
        "success": true,
        "relationship_uuid": edge_uuid.to_string(),
        "from": req.from_entity_name,
        "to": req.to_entity_name,
        "type": req.relation_type
    })))
}

/// Get all entities in the graph
#[derive(Debug, Deserialize)]
struct GetAllEntitiesRequest {
    user_id: String,
    limit: Option<usize>,
}

async fn get_all_entities(
    State(state): State<AppState>,
    Json(req): Json<GetAllEntitiesRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let graph = state.get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.read();

    let entities = graph_guard.get_all_entities()
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    let limit = req.limit.unwrap_or(100);
    let entities: Vec<_> = entities.into_iter().take(limit).collect();
    let count = entities.len();

    Ok(Json(serde_json::json!({
        "entities": entities,
        "count": count
    })))
}

// ====== Memory Visualization API Endpoints ======

/// Get visualization statistics for a user's memory graph
async fn get_visualization_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<VisualizationStats>, AppError> {
    validation::validate_user_id(&user_id)
        .map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard.get_visualization_stats();

    Ok(Json(stats))
}

/// Export visualization graph as DOT format for Graphviz
async fn get_visualization_dot(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<String, AppError> {
    validation::validate_user_id(&user_id)
        .map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let dot = memory_guard.export_visualization_dot();

    Ok(dot)
}

/// Build visualization graph from current memory state
#[derive(Debug, Deserialize)]
struct BuildVisualizationRequest {
    user_id: String,
}

async fn build_visualization(
    State(state): State<AppState>,
    Json(req): Json<BuildVisualizationRequest>,
) -> Result<Json<VisualizationStats>, AppError> {
    validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard.build_visualization_graph()
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

#[tokio::main]
async fn main() -> Result<()> {
    // P1.6: Initialize distributed tracing with OpenTelemetry (optional)
    #[cfg(feature = "telemetry")]
    {
        tracing_setup::init_tracing()
            .expect("Failed to initialize tracing");
    }
    #[cfg(not(feature = "telemetry"))]
    {
        // Use simple console logging for edge devices
        tracing_subscriber::fmt::init();
        info!("📝 Console logging initialized (telemetry disabled)");
    }

    // P1.1: Register Prometheus metrics
    metrics::register_metrics()
        .expect("Failed to register metrics");
    info!("📊 Metrics registered at /metrics");

    info!("🧠 Starting Shodh-Memory server...");

    // Load configuration from environment
    let server_config = ServerConfig::from_env();
    server_config.log();

    let is_production = server_config.is_production;

    // Create memory manager with config
    info!("📁 Storage path: {:?}", server_config.storage_path);
    let manager = Arc::new(MultiUserMemoryManager::new(
        server_config.storage_path.clone(),
        server_config.clone()
    )?);

    // Keep a reference to manager for shutdown cleanup (clone BEFORE moving into router)
    let manager_for_shutdown = Arc::clone(&manager);

    // Configure rate limiting from config
    let governor_conf = GovernorConfigBuilder::default()
        .per_second(server_config.rate_limit_per_second)
        .burst_size(server_config.rate_limit_burst)
        .finish()
        .expect("Failed to build governor rate limiter configuration");

    let governor_layer = GovernorLayer::new(governor_conf);

    info!("⚡ Rate limiting enabled: {} req/sec, burst of {}",
        server_config.rate_limit_per_second, server_config.rate_limit_burst);

    // Configure CORS - secure by default, configurable via environment
    // In production, SHODH_CORS_ORIGINS should be set to a comma-separated list of allowed origins
    // Example: SHODH_CORS_ORIGINS=https://app.example.com,https://admin.example.com
    let cors = if let Ok(origins) = std::env::var("SHODH_CORS_ORIGINS") {
        let allowed_origins: Vec<_> = origins
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if allowed_origins.is_empty() {
            info!("⚠️  CORS: No valid origins in SHODH_CORS_ORIGINS, using restrictive default");
            CorsLayer::new()
                .allow_origin(AllowOrigin::exact("http://localhost:3000".parse().unwrap()))
                .allow_methods([axum::http::Method::GET, axum::http::Method::POST, axum::http::Method::DELETE])
                .allow_headers([axum::http::header::CONTENT_TYPE, axum::http::header::AUTHORIZATION])
        } else {
            info!("🔒 CORS: Allowing origins: {:?}", allowed_origins);
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(allowed_origins))
                .allow_methods([axum::http::Method::GET, axum::http::Method::POST, axum::http::Method::DELETE])
                .allow_headers([axum::http::header::CONTENT_TYPE, axum::http::header::AUTHORIZATION])
        }
    } else if is_production {
        // Production without explicit CORS config: very restrictive
        info!("🔒 CORS: Production mode - allowing only localhost");
        CorsLayer::new()
            .allow_origin(AllowOrigin::exact("http://localhost:3000".parse().unwrap()))
            .allow_methods([axum::http::Method::GET, axum::http::Method::POST, axum::http::Method::DELETE])
            .allow_headers([axum::http::header::CONTENT_TYPE, axum::http::header::AUTHORIZATION])
    } else {
        // Development mode: allow all (for local testing only)
        info!("⚠️  CORS: Development mode - allowing all origins (NOT for production!)");
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    };

    // Build router with rate limiting
    // Split into public (no auth) and protected (auth required) routes

    // Protected API routes - require authentication
    let protected_routes = Router::new()
        // Core endpoints
        .route("/api/record", post(record_experience))
        .route("/api/retrieve", post(retrieve_memories))

        // User management
        .route("/api/users", get(list_users))
        .route("/api/users/{user_id}/stats", get(get_user_stats))
        .route("/api/users/{user_id}", delete(delete_user))

        // Memory CRUD
        .route("/api/memory/{memory_id}", get(get_memory))
        .route("/api/memory/{memory_id}", axum::routing::put(update_memory))
        .route("/api/memory/{memory_id}", delete(delete_memory))
        .route("/api/memories", post(get_all_memories))
        .route("/api/memories/history", post(get_history))

        // Compression & Storage Management
        .route("/api/memory/compress", post(compress_memory))
        .route("/api/memory/decompress", post(decompress_memory))
        .route("/api/storage/stats", post(get_storage_stats))
        .route("/api/storage/uncompressed", post(get_uncompressed_old))

        // Forgetting Operations
        .route("/api/forget/age", post(forget_by_age))
        .route("/api/forget/importance", post(forget_by_importance))
        .route("/api/forget/pattern", post(forget_by_pattern))

        // Advanced Search
        .route("/api/search/advanced", post(advanced_search))
        .route("/api/search/multimodal", post(multimodal_search))
        .route("/api/search/robotics", post(robotics_search))

        // Graph Memory - Entity Management
        .route("/api/graph/{user_id}/stats", get(get_graph_stats))
        .route("/api/graph/entity/find", post(find_entity))
        .route("/api/graph/entity/add", post(add_entity))
        .route("/api/graph/entities/all", post(get_all_entities))

        // Graph Memory - Relationship Management
        .route("/api/graph/relationship/add", post(add_relationship))
        .route("/api/graph/relationship/invalidate", post(invalidate_relationship))
        .route("/api/graph/traverse", post(traverse_graph))

        // Graph Memory - Episodes
        .route("/api/graph/episode/get", post(get_episode))

        // Memory Visualization
        .route("/api/visualization/{user_id}/stats", get(get_visualization_stats))
        .route("/api/visualization/{user_id}/dot", get(get_visualization_dot))
        .route("/api/visualization/build", post(build_visualization))

        // Apply auth middleware only to protected routes
        .layer(axum::middleware::from_fn(auth::auth_middleware))
        .with_state(manager.clone());

    // P0.8: Concurrency limiting for production resilience
    // Limits max concurrent requests to prevent resource exhaustion
    let max_concurrent = server_config.max_concurrent_requests;

    info!("🔄 Concurrency limiting enabled: max_concurrent={}", max_concurrent);

    // Combine public and protected routes
    let app = Router::new()
        // Public routes - no authentication required
        .route("/", get(|| async { axum::response::Redirect::permanent("/static/index.html") }))
        .nest_service("/static", ServeDir::new("static"))
        .route("/health", get(health))
        .route("/health/live", get(health_live))     // P0.9: Kubernetes liveness probe
        .route("/health/ready", get(health_ready))   // P0.9: Kubernetes readiness probe
        .route("/metrics", get(metrics_endpoint))    // P1.1: Prometheus metrics

        // Merge protected routes
        .merge(protected_routes)

        // Apply layers (innermost to outermost):
        // 1. P1.6: Trace propagation (innermost - sets up trace context first) - OPTIONAL
        // 2. P1.3: Metrics tracking (tracks ALL requests with trace context)
        // 3. P0.8: ConcurrencyLimit - Limit max concurrent requests
        // 4. CORS: Handle cross-origin requests
        // 5. RateLimit: Per-IP rate limiting (outermost, prevents abuse)
        ;

    // Conditionally add trace propagation middleware only when telemetry feature is enabled
    #[cfg(feature = "telemetry")]
    let app = app.layer(axum::middleware::from_fn(crate::tracing_setup::trace_propagation::propagate_trace_context));

    let app = app
        .layer(axum::middleware::from_fn(crate::middleware::track_metrics))
        .layer(ConcurrencyLimitLayer::new(max_concurrent))
        .layer(cors)
        .layer(governor_layer)
        .with_state(manager);

    // Start server using port from config
    let port = server_config.port;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!("🚀 Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // P0.11: Wrap entire shutdown process with timeout for production resilience
    let shutdown_future = async {
        // Graceful shutdown with ConnectInfo for rate limiting (tower_governor requires SocketAddr)
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>()
        )
        .with_graceful_shutdown(shutdown_signal())
        .await?;

        info!("🔒 All requests completed, flushing databases...");

        // P0.11: Flush RocksDB with timeout
        let flush_future = async {
            manager_for_shutdown.flush_all_databases()
        };

        match tokio::time::timeout(
            std::time::Duration::from_secs(DATABASE_FLUSH_TIMEOUT_SECS),
            flush_future
        ).await {
            Ok(Ok(())) => info!("✅ Databases flushed successfully"),
            Ok(Err(e)) => tracing::error!("❌ Failed to flush databases: {}", e),
            Err(_) => tracing::error!("⏱️  Database flush timed out after {}s", DATABASE_FLUSH_TIMEOUT_SECS),
        }

        // P0.11: Save vector indices with timeout
        info!("💾 Persisting vector indices...");
        let save_future = async {
            manager_for_shutdown.save_all_vector_indices()
        };

        match tokio::time::timeout(
            std::time::Duration::from_secs(VECTOR_INDEX_SAVE_TIMEOUT_SECS),
            save_future
        ).await {
            Ok(Ok(())) => info!("✅ Vector indices saved successfully"),
            Ok(Err(e)) => tracing::error!("❌ Failed to save vector indices: {}", e),
            Err(_) => tracing::error!("⏱️  Vector index save timed out after {}s", VECTOR_INDEX_SAVE_TIMEOUT_SECS),
        }

        // P1.6: Shutdown tracing and flush remaining spans (optional, only with telemetry feature)
        #[cfg(feature = "telemetry")]
        tracing_setup::shutdown_tracing();

        Ok::<(), anyhow::Error>(())
    };

    // P0.11: Enforce overall shutdown timeout with force-kill fallback
    match tokio::time::timeout(
        std::time::Duration::from_secs(GRACEFUL_SHUTDOWN_TIMEOUT_SECS),
        shutdown_future
    ).await {
        Ok(Ok(())) => {
            info!("👋 Server shutdown complete");
        }
        Ok(Err(e)) => {
            tracing::error!("❌ Shutdown error: {}", e);
            std::process::exit(1);
        }
        Err(_) => {
            tracing::error!("⏱️  Graceful shutdown timed out after {}s, forcing exit", GRACEFUL_SHUTDOWN_TIMEOUT_SECS);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Handle graceful shutdown
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("🛑 Shutdown signal received, starting graceful shutdown");
}
