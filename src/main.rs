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
use tower::limit::ConcurrencyLimitLayer;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::services::ServeDir;
use tracing::info;

mod auth;
mod config;
mod constants;
mod embeddings;
mod errors;
mod graph_memory;
mod memory;
mod metrics; // P1.1: Observability
mod middleware; // P1.3: HTTP request tracking
mod similarity;
mod tracing_setup;
mod validation;
mod vector_db; // P1.6: Distributed tracing

use config::ServerConfig;
use constants::{
    DATABASE_FLUSH_TIMEOUT_SECS, GRACEFUL_SHUTDOWN_TIMEOUT_SECS, VECTOR_INDEX_SAVE_TIMEOUT_SECS,
    VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
};

use embeddings::NerEntityType;
use embeddings::{are_ner_models_downloaded, get_ner_models_dir, NerConfig, NeuralNer};
use errors::{AppError, ValidationErrorExt};
use graph_memory::{
    EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, GraphStats, GraphTraversal,
    RelationType, RelationshipEdge,
};
use memory::{
    spreading_activation_retrieve, ActivatedMemory, Experience, ExperienceType,
    GraphStats as VisualizationStats, Memory, MemoryConfig, MemoryId, MemoryStats, MemorySystem,
    Query as MemoryQuery, SharedMemory,
};
use similarity::top_k_similar;

/// Audit event for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: String, // CREATE, UPDATE, DELETE, RETRIEVE
    pub memory_id: String,
    pub details: String,
}

/// SSE Memory Event - lightweight event for real-time streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub event_type: String, // CREATE, RETRIEVE, DELETE, GRAPH_UPDATE
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user_id: String,
    pub memory_id: Option<String>,
    pub content_preview: Option<String>, // First 100 chars
    pub experience_type: Option<String>,
    pub importance: Option<f32>,
    pub count: Option<usize>, // For retrieve events - number of results
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
        for (key, value) in iter.flatten() {
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
            self.audit_db
                .write(batch)
                .map_err(|e| anyhow::anyhow!("Failed to write rotation batch: {e}"))?;
        }

        // Update in-memory cache by removing old events
        if removed_count > 0 {
            if let Some(log) = self.audit_logs.get(user_id) {
                let mut log_guard = log.write();

                // Keep only events that weren't removed
                log_guard.retain(|event| {
                    let event_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                    event_nanos >= cutoff_nanos
                        && initial_count - removed_count <= self.audit_max_entries
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
    user_memories:
        Arc<parking_lot::Mutex<lru::LruCache<String, Arc<parking_lot::RwLock<MemorySystem>>>>>,

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
    graph_memories:
        Arc<parking_lot::Mutex<lru::LruCache<String, Arc<parking_lot::RwLock<GraphMemory>>>>>,

    /// Neural NER for automatic entity extraction (uses TinyBERT ONNX model)
    neural_ner: Arc<NeuralNer>,

    /// User eviction counter for metrics
    user_evictions: Arc<std::sync::atomic::AtomicUsize>,

    /// Server configuration (configurable via environment)
    server_config: ServerConfig,

    /// SSE event broadcaster for real-time dashboard updates
    /// Broadcast channel allows multiple subscribers (SSE clients) to receive events
    event_broadcaster: tokio::sync::broadcast::Sender<MemoryEvent>,
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

        // Create broadcast channel for SSE events (capacity 1024 events)
        // Older events are dropped if subscribers can't keep up
        let (event_broadcaster, _) = tokio::sync::broadcast::channel(1024);

        // Initialize Neural NER - auto-detects downloaded models
        let neural_ner = if are_ner_models_downloaded() {
            let ner_dir = get_ner_models_dir();
            let config = NerConfig {
                model_path: ner_dir.join("model.onnx"),
                tokenizer_path: ner_dir.join("tokenizer.json"),
                max_length: 128,
                confidence_threshold: 0.5,
            };
            match NeuralNer::new(config) {
                Ok(ner) => {
                    info!(
                        "🧠 Neural NER initialized (TinyBERT model at {:?})",
                        ner_dir
                    );
                    Arc::new(ner)
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize neural NER: {}. Using fallback.", e);
                    Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                }
            }
        } else {
            info!("📋 NER models not downloaded. Using rule-based fallback.");
            Arc::new(NeuralNer::new_fallback(NerConfig::default()))
        };

        let manager = Self {
            user_memories: Arc::new(parking_lot::Mutex::new(lru::LruCache::new(cache_size))),
            audit_logs: Arc::new(DashMap::new()),
            audit_db,
            base_path,
            default_config: MemoryConfig::default(),
            audit_log_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            graph_memories: Arc::new(parking_lot::Mutex::new(lru::LruCache::new(cache_size))),
            neural_ner,
            user_evictions: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            server_config,
            event_broadcaster,
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
        let key = format!(
            "{}:{}",
            user_id,
            event.timestamp.timestamp_nanos_opt().unwrap_or(0)
        );
        if let Ok(serialized) = bincode::serialize(&event) {
            let db = self.audit_db.clone();
            let key_bytes = key.into_bytes();

            // Spawn blocking DB write on dedicated thread pool
            tokio::task::spawn_blocking(move || {
                if let Err(e) = db.put(&key_bytes, &serialized) {
                    tracing::error!("Failed to persist audit log: {}", e);
                }
            });
        }

        // Also update in-memory cache for fast access (with size cap to prevent unbounded growth)
        let max_entries = self.server_config.audit_max_entries_per_user;
        if let Some(log) = self.audit_logs.get(user_id) {
            let mut entries = log.write();
            entries.push(event);
            // Enforce in-memory size cap: remove oldest entries if over limit
            if entries.len() > max_entries {
                let excess = entries.len() - max_entries;
                entries.drain(0..excess);
            }
        } else {
            let log = Arc::new(parking_lot::RwLock::new(vec![event]));
            self.audit_logs.insert(user_id.to_string(), log);
        }

        // Check if rotation is needed (every N events) - LOCK-FREE atomic operation
        let count = self
            .audit_log_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

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

    /// Emit SSE event to all connected dashboard clients
    /// Non-blocking - if no clients are listening, event is dropped silently
    fn emit_event(&self, event: MemoryEvent) {
        // broadcast::send returns Err if no receivers, which is fine
        let _ = self.event_broadcaster.send(event);
    }

    /// Subscribe to SSE events (returns a receiver)
    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<MemoryEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get audit history for user (reads from persistent storage)
    pub fn get_history(&self, user_id: &str, memory_id: Option<&str>) -> Vec<AuditEvent> {
        // Try in-memory cache first (fast path)
        if let Some(log) = self.audit_logs.get(user_id) {
            let events = log.read();
            if !events.is_empty() {
                // Cache hit - use cached data
                return if let Some(mid) = memory_id {
                    events
                        .iter()
                        .filter(|e| e.memory_id == mid)
                        .cloned()
                        .collect()
                } else {
                    events.clone()
                };
            }
        }

        // Cache miss - load from RocksDB (happens after restart)
        let mut events = Vec::new();
        let prefix = format!("{user_id}:");

        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for (key, value) in iter.flatten() {
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
                if let Some((evicted_user_id, _evicted_memory)) =
                    cache.push(user_id.to_string(), memory_arc.clone())
                {
                    // LRU eviction occurred - log it
                    self.user_evictions
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    info!(
                        "📤 Evicted user '{}' from memory cache (LRU, cache_size={})",
                        evicted_user_id, self.server_config.max_users_in_memory
                    );

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
            tracing::warn!(
                "Vector index initialization failed for user {}: {}",
                user_id,
                e
            );
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
        self.user_memories
            .lock()
            .iter()
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Flush all RocksDB databases to ensure data persistence (critical for graceful shutdown)
    pub fn flush_all_databases(&self) -> Result<()> {
        info!("💾 Flushing all databases to disk...");

        // Flush audit database
        self.audit_db
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush audit database: {e}"))?;

        // Flush all user memory databases
        // Collect entries first, then release lock before flushing (avoid holding lock during I/O)
        let user_entries: Vec<(String, Arc<parking_lot::RwLock<MemorySystem>>)> = {
            self.user_memories
                .lock()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
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

        info!("✅ Flushed 1 audit database and {} user databases", flushed);

        Ok(())
    }

    /// Save all vector indices to disk (production persistence)
    pub fn save_all_vector_indices(&self) -> Result<()> {
        info!("🔍 Saving vector indices to disk...");

        // Collect entries first, then release lock before saving (avoid holding lock during I/O)
        let user_entries: Vec<(String, Arc<parking_lot::RwLock<MemorySystem>>)> = {
            self.user_memories
                .lock()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
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
                    tracing::warn!(
                        "Failed to load vector index for user {}: {}. Rebuilding...",
                        user_id,
                        e
                    );
                }
            }
        }

        // Index doesn't exist or load failed - rebuild from storage
        info!("🔨 Rebuilding vector index for user {}...", user_id);
        match memory_guard.rebuild_vector_index() {
            Ok(_) => {
                info!("✅ Rebuilt vector index for user: {}", user_id);
                // Save the newly built index
                if let Err(e) = memory_guard.save_vector_index(&index_path) {
                    tracing::warn!("Failed to save vector index for user {}: {}", user_id, e);
                }
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

        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if let Some(user_id) = key_str.split(':').next() {
                    user_ids.insert(user_id.to_string());
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
                        info!(
                            "  Rotated audit logs for user {}: removed {} old entries",
                            user_id, removed
                        );
                        total_removed += removed;
                    }
                }
                Err(e) => {
                    tracing::warn!("  Failed to rotate audit logs for user {}: {}", user_id, e);
                }
            }
        }

        if total_removed > 0 {
            info!(
                "✅ Audit log rotation complete: removed {} total entries",
                total_removed
            );
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
                if let Some((evicted_user_id, _evicted_graph)) =
                    cache.push(user_id.to_string(), graph_arc.clone())
                {
                    // LRU eviction occurred - log it
                    info!(
                        "📤 Evicted graph for user '{}' from memory cache (LRU, cache_size={})",
                        evicted_user_id, self.server_config.max_users_in_memory
                    );
                }
            } else {
                cache.put(user_id.to_string(), graph_arc.clone());
            }
        }

        info!("📊 Created graph memory for user: {}", user_id);

        Ok(graph_arc)
    }

    /// Process an experience and extract entities/relationships into the graph
    fn process_experience_into_graph(
        &self,
        user_id: &str,
        experience: &Experience,
        memory_id: &MemoryId,
    ) -> Result<()> {
        let graph = self.get_user_graph(user_id)?;
        let graph_guard = graph.write();

        // Extract entities from the experience content using neural NER
        let extracted_entities = match self.neural_ner.extract(&experience.content) {
            Ok(entities) => entities,
            Err(e) => {
                tracing::debug!("NER extraction failed: {}. Continuing without entities.", e);
                Vec::new()
            }
        };

        let mut entity_uuids = Vec::new();

        // Add entities to the graph - map NerEntityType to EntityLabel
        for ner_entity in extracted_entities {
            let label = match ner_entity.entity_type {
                NerEntityType::Person => EntityLabel::Person,
                NerEntityType::Organization => EntityLabel::Organization,
                NerEntityType::Location => EntityLabel::Location,
                NerEntityType::Misc => EntityLabel::Other("MISC".to_string()),
            };

            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(), // Will be replaced if exists
                name: ner_entity.text.clone(),
                labels: vec![label],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
                salience: ner_entity.confidence, // Use NER confidence as salience
                is_proper_noun: true,            // Neural NER primarily extracts proper nouns
            };

            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((ner_entity.text, uuid)),
                Err(e) => tracing::debug!("Failed to add entity {}: {}", ner_entity.text, e),
            }
        }

        // Create an episodic node for this experience
        let episode = EpisodicNode {
            uuid: memory_id.0, // Use memory UUID directly (already a Uuid)
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
                    source_episode_id: Some(memory_id.0), // Use memory UUID directly (already a Uuid)
                    context: experience.content.clone(),
                    // Hebbian plasticity fields (new synapses start fresh)
                    last_activated: chrono::Utc::now(),
                    activation_count: 1, // First activation
                    potentiated: false,
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

    /// Run maintenance on all cached user memories
    ///
    /// This is called periodically by the background scheduler to:
    /// 1. Consolidate memories (tier promotion based on thresholds)
    /// 2. Decay activation levels
    /// 3. Run graph maintenance
    ///
    /// Only operates on currently cached users - evicted users will run
    /// maintenance on next access.
    pub fn run_maintenance_all_users(&self) -> usize {
        let decay_factor = self.server_config.activation_decay_factor;
        let mut total_processed = 0;

        // Get list of cached user IDs (hold lock briefly)
        let user_ids: Vec<String> = {
            let cache = self.user_memories.lock();
            cache.iter().map(|(id, _)| id.clone()).collect()
        };

        let user_count = user_ids.len();

        for user_id in user_ids {
            // Re-acquire memory for each user (may have been evicted)
            if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                let memory = memory_lock.read();
                match memory.run_maintenance(decay_factor) {
                    Ok(count) => total_processed += count,
                    Err(e) => {
                        tracing::warn!("Maintenance failed for user {}: {}", user_id, e);
                    }
                }
            }
        }

        tracing::info!(
            "Maintenance complete: {} memories processed across {} users",
            total_processed,
            user_count
        );

        total_processed
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
    #[serde(alias = "query")]
    query_text: Option<String>,
    query_embedding: Option<Vec<f32>>,
    #[serde(alias = "limit")]
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

// =============================================================================
// SIMPLIFIED LLM-FRIENDLY API TYPES
// Minimal request/response for effortless use by AI agents
// =============================================================================

/// Simplified remember request - just content, auto-creates Experience
#[derive(Debug, Deserialize)]
struct RememberRequest {
    user_id: String,
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    /// Accept both "experience_type" and "memory_type" as field names
    #[serde(default, alias = "memory_type")]
    experience_type: Option<String>,
}

/// Simplified remember response
#[derive(Debug, Serialize)]
struct RememberResponse {
    id: String,
    success: bool,
}

/// Simplified recall request - just query text
#[derive(Debug, Deserialize)]
struct RecallRequest {
    user_id: String,
    query: String,
    #[serde(default = "default_recall_limit")]
    limit: usize,
}

fn default_recall_limit() -> usize {
    5
}

/// Simplified recall response - returns just text snippets
#[derive(Debug, Serialize)]
struct RecallResponse {
    memories: Vec<RecallMemory>,
    count: usize,
}

#[derive(Debug, Serialize)]
struct RecallMemory {
    id: String,
    content: String,
    importance: f32,
    created_at: String,
}

/// Batch remember request for bulk inserts
#[derive(Debug, Deserialize)]
struct BatchRememberRequest {
    user_id: String,
    memories: Vec<BatchMemoryItem>,
}

#[derive(Debug, Deserialize)]
struct BatchMemoryItem {
    content: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    experience_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct BatchRememberResponse {
    ids: Vec<String>,
    success_count: usize,
    error_count: usize,
}

// =============================================================================
// HEBBIAN FEEDBACK API - Wires up the learning loop
// =============================================================================

/// Request for tracked retrieval (returns tracking ID for later feedback)
#[derive(Debug, Deserialize)]
struct TrackedRetrieveRequest {
    user_id: String,
    query: String,
    #[serde(default = "default_recall_limit")]
    limit: usize,
}

/// Response with tracking ID for feedback
#[derive(Debug, Serialize)]
struct TrackedRetrieveResponse {
    tracking_id: String,
    memory_ids: Vec<String>,
    memories: Vec<RecallMemory>,
}

/// Request to provide feedback on retrieval outcome
#[derive(Debug, Deserialize)]
struct ReinforceFeedbackRequest {
    user_id: String,
    memory_ids: Vec<String>,
    /// "helpful", "misleading", or "neutral"
    outcome: String,
}

/// Response from reinforcement
#[derive(Debug, Serialize)]
struct ReinforceFeedbackResponse {
    memories_processed: usize,
    associations_strengthened: usize,
    importance_boosts: usize,
    importance_decays: usize,
}

// =============================================================================
// SEMANTIC CONSOLIDATION API - Extracts durable facts from episodic memories
// =============================================================================

/// Request to trigger consolidation
#[derive(Debug, Deserialize)]
struct ConsolidateRequest {
    user_id: String,
    /// Minimum number of supporting memories for a fact (default: 2)
    #[serde(default = "default_min_support")]
    min_support: usize,
    /// Minimum age in days before consolidation (default: 7)
    #[serde(default = "default_min_age_days")]
    min_age_days: i64,
}

fn default_min_support() -> usize {
    2
}

fn default_min_age_days() -> i64 {
    7
}

/// Response from consolidation
#[derive(Debug, Serialize)]
struct ConsolidateResponse {
    memories_analyzed: usize,
    facts_extracted: usize,
    facts_reinforced: usize,
    fact_ids: Vec<String>,
}

/// Application state
type AppState = Arc<MultiUserMemoryManager>;

// REST API handlers

/// Health check endpoint (basic compatibility)
async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let users_in_cache = state.user_memories.lock().len();
    let user_evictions = state
        .user_evictions
        .load(std::sync::atomic::Ordering::Relaxed);

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
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "alive",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// P0.9: Readiness probe - indicates if service can handle traffic
/// Returns 200 OK if service is ready, 503 if not ready
/// Kubernetes uses this to route traffic only to ready pods
async fn health_ready(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    // Check if critical resources are accessible
    let users_in_cache = state.user_memories.lock().len();

    // Service is ready if we can access the user cache without panicking
    // This verifies the lock is not poisoned and the service is operational
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ready",
            "version": env!("CARGO_PKG_VERSION"),
            "users_in_cache": users_in_cache,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// Prometheus metrics endpoint for observability
/// Returns metrics in Prometheus text format for scraping
async fn metrics_endpoint(State(state): State<AppState>) -> Result<String, StatusCode> {
    use prometheus::Encoder;

    // Update memory usage gauges before serving metrics
    let users_in_cache = state.user_memories.lock().len();
    metrics::ACTIVE_USERS.set(users_in_cache as i64);

    // Aggregate metrics across all users (no per-user labels to avoid cardinality explosion)
    let (mut total_working, mut total_session, mut total_longterm, mut total_heap) =
        (0i64, 0i64, 0i64, 0i64);
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
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["working"])
        .set(total_working);
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["session"])
        .set(total_session);
    metrics::MEMORIES_BY_TIER
        .with_label_values(&["longterm"])
        .set(total_longterm);
    metrics::MEMORY_HEAP_BYTES_TOTAL.set(total_heap);
    metrics::VECTOR_INDEX_SIZE_TOTAL.set(total_vectors);

    // Gather and encode metrics
    let encoder = prometheus::TextEncoder::new();
    let metric_families = metrics::METRICS_REGISTRY.gather();

    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    String::from_utf8(buffer).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// SSE endpoint for real-time memory events
/// Streams CREATE, RETRIEVE, DELETE events to connected dashboard clients
/// No authentication required - read-only, lightweight event stream
async fn memory_events_sse(
    State(state): State<AppState>,
) -> axum::response::Sse<
    impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    use axum::response::sse::Event;
    use futures::StreamExt;
    use tokio_stream::wrappers::BroadcastStream;

    let receiver = state.subscribe_events();
    let stream = BroadcastStream::new(receiver);

    let event_stream = stream.filter_map(|result| async move {
        match result {
            Ok(event) => {
                let json = serde_json::to_string(&event).ok()?;
                Some(Ok(Event::default().event(&event.event_type).data(json)))
            }
            Err(_) => None, // Lagged receiver, skip event
        }
    });

    axum::response::Sse::new(event_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("heartbeat"),
    )
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    validation::validate_content(&req.experience.content, false).map_validation_err("content")?;

    if let Some(ref embeddings) = req.experience.embeddings {
        validation::validate_embeddings(embeddings)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Extract entities from content using Neural NER and merge with user-provided entities
    let extracted_names: Vec<String> = match state.neural_ner.extract(&req.experience.content) {
        Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
        Err(e) => {
            tracing::debug!("NER extraction failed in record_experience: {}", e);
            Vec::new()
        }
    };

    // Merge user entities with NER-extracted entities (deduplicated)
    let mut merged_entities: Vec<String> = req.experience.entities.clone();
    for entity_name in extracted_names {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&entity_name))
        {
            merged_entities.push(entity_name);
        }
    }

    // Create experience with merged entities
    let experience_with_entities = Experience {
        entities: merged_entities,
        ..req.experience.clone()
    };

    // P1.2: Instrument memory store operation
    let store_start = std::time::Instant::now();

    // CRITICAL FIX: Wrap blocking I/O in spawn_blocking
    // record() does ONNX inference (10-50ms) + RocksDB writes (100µs-10ms)
    // Running these on async threads starves the Tokio runtime under load
    let memory_id = {
        let memory = memory.clone();
        let experience = experience_with_entities.clone();

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.record(experience)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Extract entities and build knowledge graph (background processing)
    if let Err(e) =
        state.process_experience_into_graph(&req.user_id, &experience_with_entities, &memory_id)
    {
        tracing::debug!("Graph processing failed for memory {}: {}", memory_id.0, e);
        // Don't fail the request if graph processing fails
    }

    // Enterprise audit logging
    state.log_event(
        &req.user_id,
        "CREATE",
        &memory_id.0.to_string(),
        &format!(
            "Created memory: {}",
            req.experience.content.chars().take(50).collect::<String>()
        ),
    );

    // SSE: Emit real-time event for dashboard
    state.emit_event(MemoryEvent {
        event_type: "CREATE".to_string(),
        timestamp: chrono::Utc::now(),
        user_id: req.user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(req.experience.content.chars().take(100).collect()),
        experience_type: Some(format!("{:?}", req.experience.experience_type)),
        importance: req.experience.reward, // Map reward to importance for display
        count: None,
    });

    // Record metrics (no user_id to prevent cardinality explosion)
    let duration = store_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    Ok(Json(RecordResponse {
        memory_id: memory_id.0.to_string(),
        success: true,
    }))
}

// =============================================================================
// SIMPLIFIED LLM-FRIENDLY API HANDLERS
// =============================================================================

/// LLM-friendly /api/remember - just pass content, get memory ID back
/// Example: POST /api/remember { "user_id": "agent-1", "content": "User likes pizza" }
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn remember(
    State(state): State<AppState>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, AppError> {
    // P1.2: Instrument remember operation
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_content(&req.content, false).map_validation_err("content")?;

    // Parse experience type from string, default to Context
    let experience_type = req
        .experience_type
        .as_ref()
        .and_then(|s| match s.to_lowercase().as_str() {
            "task" => Some(ExperienceType::Task),
            "learning" => Some(ExperienceType::Learning),
            "decision" => Some(ExperienceType::Decision),
            "error" => Some(ExperienceType::Error),
            "pattern" => Some(ExperienceType::Pattern),
            "conversation" => Some(ExperienceType::Conversation),
            "discovery" => Some(ExperienceType::Discovery),
            _ => None,
        })
        .unwrap_or(ExperienceType::Context);

    // Extract entities from content using Neural NER and merge with user-provided tags
    let extracted_names: Vec<String> = match state.neural_ner.extract(&req.content) {
        Ok(entities) => entities.into_iter().map(|e| e.text).collect(),
        Err(e) => {
            tracing::debug!("NER extraction failed in remember_simplified: {}", e);
            Vec::new()
        }
    };

    // Merge user tags with NER-extracted entities (deduplicated)
    let mut merged_entities: Vec<String> = req.tags.clone();
    for entity_name in extracted_names {
        if !merged_entities
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&entity_name))
        {
            merged_entities.push(entity_name);
        }
    }

    // Auto-create Experience with sensible defaults
    let experience = Experience {
        content: req.content.clone(),
        experience_type,
        entities: merged_entities,
        ..Default::default()
    };

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_id = {
        let memory = memory.clone();
        let exp_clone = experience.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.record(exp_clone)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Extract entities and build knowledge graph (same as /api/record)
    if let Err(e) = state.process_experience_into_graph(&req.user_id, &experience, &memory_id) {
        tracing::debug!("Graph processing failed for memory {}: {}", memory_id.0, e);
        // Don't fail the request if graph processing fails
    }

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    Ok(Json(RememberResponse {
        id: memory_id.0.to_string(),
        success: true,
    }))
}

/// LLM-friendly /api/recall - hybrid retrieval combining semantic search + graph spreading activation
/// Example: POST /api/recall { "user_id": "agent-1", "query": "What does user like?" }
///
/// Retrieval Strategy:
/// 1. Semantic search: MiniLM embeddings + Vamana HNSW (50% weight)
/// 2. Graph spreading activation: Entity-based context retrieval (35% weight)
/// 3. Linguistic matching: Focal entity boost (15% weight)
/// Results are deduplicated and ranked by hybrid score.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
async fn recall(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, AppError> {
    use std::collections::HashMap;

    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_system = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_memory = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let limit = req.limit;
    let query_text = req.query.clone();
    let query_text_clone = query_text.clone();

    // Run semantic retrieval via MemorySystem
    let semantic_memories: Vec<SharedMemory> = {
        let memory = memory_system.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let query = MemoryQuery {
                query_text: Some(query_text),
                max_results: limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
                ..Default::default()
            };
            memory_guard.retrieve(&query).unwrap_or_default()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // Run graph-based spreading activation retrieval
    let graph_activated: Vec<ActivatedMemory> = {
        let graph = graph_memory.clone();
        let memory = memory_system.clone();
        let query_for_graph = query_text_clone.clone();

        tokio::task::spawn_blocking(move || {
            let graph_guard = graph.read();
            let memory_guard = memory.read();

            // Build the episode-to-memory mapping function
            // EpisodicNode UUID == MemoryId.0 (see process_experience_into_graph)
            let episode_to_memory = |episode: &EpisodicNode| -> anyhow::Result<Option<SharedMemory>> {
                let memory_id = MemoryId(episode.uuid);
                match memory_guard.get_memory(&memory_id) {
                    Ok(mem) => Ok(Some(Arc::new(mem))),
                    Err(_) => Ok(None), // Memory may have been deleted
                }
            };

            let query = MemoryQuery {
                query_text: Some(query_for_graph.clone()),
                max_results: limit * VECTOR_SEARCH_CANDIDATE_MULTIPLIER,
                ..Default::default()
            };

            // Run spreading activation
            match spreading_activation_retrieve(
                &query_for_graph,
                &query,
                &graph_guard,
                memory_guard.get_embedder(),
                episode_to_memory,
            ) {
                Ok(activated) => activated,
                Err(e) => {
                    tracing::debug!("Spreading activation failed: {}. Using semantic only.", e);
                    Vec::new()
                }
            }
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Graph retrieval panicked: {e}")))?
    };

    // Merge results with hybrid scoring
    // Scoring weights: semantic 50%, graph 35%, linguistic 15% (from constants)
    let mut scored_memories: HashMap<uuid::Uuid, (f32, SharedMemory)> = HashMap::new();
    let semantic_count = semantic_memories.len();
    let graph_activated_count = graph_activated.len();

    // Add semantic results with their position-based score
    for (rank, memory) in semantic_memories.iter().enumerate() {
        let semantic_score = 1.0 / (rank as f32 + 1.0); // Reciprocal rank
        let hybrid_score = 0.50 * semantic_score; // 50% weight for semantic
        scored_memories.insert(memory.id.0, (hybrid_score, memory.clone()));
    }

    // Add/boost graph-activated results
    for activated in graph_activated {
        let entry = scored_memories
            .entry(activated.memory.id.0)
            .or_insert((0.0, activated.memory.clone()));

        // Add graph activation score (35% weight) + linguistic (15% weight)
        entry.0 += 0.35 * activated.activation_score + 0.15 * activated.linguistic_score;
    }

    // Sort by final hybrid score and take top N
    let mut final_results: Vec<(f32, SharedMemory)> = scored_memories.into_values().collect();
    final_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    final_results.truncate(limit);

    let graph_contribution = final_results
        .iter()
        .filter(|(score, _)| *score > 0.50) // Has graph boost
        .count();

    tracing::debug!(
        semantic_count = semantic_count,
        graph_activated_count = graph_activated_count,
        final_count = final_results.len(),
        graph_contribution = graph_contribution,
        "Hybrid retrieval completed"
    );

    // Convert to response format
    let recall_memories: Vec<RecallMemory> = final_results
        .into_iter()
        .map(|(_, m)| RecallMemory {
            id: m.id.0.to_string(),
            content: m.experience.content.clone(),
            importance: m.importance(),
            created_at: m.created_at.to_rfc3339(),
        })
        .collect();

    let count = recall_memories.len();

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_RETRIEVE_DURATION
        .with_label_values(&["hybrid"])
        .observe(duration);
    metrics::MEMORY_RETRIEVE_TOTAL
        .with_label_values(&["hybrid", "success"])
        .inc();
    metrics::MEMORY_RETRIEVE_RESULTS
        .with_label_values(&["hybrid"])
        .observe(count as f64);

    Ok(Json(RecallResponse {
        memories: recall_memories,
        count,
    }))
}

// =============================================================================
// CONTEXT SUMMARY - Session bootstrap with categorized memories
// =============================================================================

/// Context summary request
#[derive(Debug, Deserialize)]
struct ContextSummaryRequest {
    user_id: String,
    #[serde(default = "default_true")]
    include_decisions: bool,
    #[serde(default = "default_true")]
    include_learnings: bool,
    #[serde(default = "default_true")]
    include_context: bool,
    #[serde(default = "default_max_items")]
    max_items: usize,
}

fn default_true() -> bool {
    true
}

fn default_max_items() -> usize {
    5
}

/// Summary item - simplified memory for context
#[derive(Debug, Serialize)]
struct SummaryItem {
    id: String,
    content: String,
    importance: f32,
    created_at: String,
}

/// Context summary response - categorized memories for session bootstrap
#[derive(Debug, Serialize)]
struct ContextSummaryResponse {
    total_memories: usize,
    decisions: Vec<SummaryItem>,
    learnings: Vec<SummaryItem>,
    context: Vec<SummaryItem>,
    patterns: Vec<SummaryItem>,
    errors: Vec<SummaryItem>,
}

/// GET /api/context_summary - Get categorized context for session bootstrap
/// Returns decisions, learnings, patterns, errors organized for LLM consumption
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn context_summary(
    State(state): State<AppState>,
    Json(req): Json<ContextSummaryRequest>,
) -> Result<Json<ContextSummaryResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let all_memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_all_memories()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    let total_memories = all_memories.len();

    // Categorize memories by type
    let mut decisions = Vec::new();
    let mut learnings = Vec::new();
    let mut context = Vec::new();
    let mut patterns = Vec::new();
    let mut errors = Vec::new();

    for m in all_memories {
        let item = SummaryItem {
            id: m.id.0.to_string(),
            content: m.experience.content.chars().take(200).collect(),
            importance: m.importance(),
            created_at: m.created_at.to_rfc3339(),
        };

        match m.experience.experience_type {
            ExperienceType::Decision => decisions.push(item),
            ExperienceType::Learning => learnings.push(item),
            ExperienceType::Context | ExperienceType::Observation => context.push(item),
            ExperienceType::Pattern => patterns.push(item),
            ExperienceType::Error => errors.push(item),
            _ => context.push(item),
        }
    }

    // Sort by importance and truncate
    let sort_and_truncate = |mut items: Vec<SummaryItem>, max: usize| -> Vec<SummaryItem> {
        items.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(max);
        items
    };

    let max = req.max_items;
    Ok(Json(ContextSummaryResponse {
        total_memories,
        decisions: if req.include_decisions {
            sort_and_truncate(decisions, max)
        } else {
            vec![]
        },
        learnings: if req.include_learnings {
            sort_and_truncate(learnings, max)
        } else {
            vec![]
        },
        context: if req.include_context {
            sort_and_truncate(context, max)
        } else {
            vec![]
        },
        patterns: sort_and_truncate(patterns, max),
        errors: sort_and_truncate(errors, 3.min(max)), // Limit errors to 3
    }))
}

// =============================================================================
// LIST MEMORIES - Simple GET endpoint for listing all memories
// =============================================================================

/// Query parameters for list endpoint
#[derive(Debug, Deserialize)]
struct ListQuery {
    limit: Option<usize>,
    #[serde(rename = "type")]
    memory_type: Option<String>,
}

/// List response - simplified memory list
#[derive(Debug, Serialize)]
struct ListResponse {
    memories: Vec<ListMemoryItem>,
    total: usize,
}

#[derive(Debug, Serialize)]
struct ListMemoryItem {
    id: String,
    content: String,
    memory_type: String,
    importance: f32,
    tags: Vec<String>,
    created_at: String,
}

/// GET /api/list/{user_id} - List all memories for a user
/// Query params: ?limit=100&type=Decision
#[tracing::instrument(skip(state), fields(user_id = %user_id))]
async fn list_memories(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<Json<ListResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let all_memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.get_all_memories()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    // Filter by type if specified
    let filtered: Vec<_> = if let Some(ref type_filter) = query.memory_type {
        let type_lower = type_filter.to_lowercase();
        all_memories
            .into_iter()
            .filter(|m| format!("{:?}", m.experience.experience_type).to_lowercase() == type_lower)
            .collect()
    } else {
        all_memories
    };

    let total = filtered.len();
    let limit = query.limit.unwrap_or(100).min(1000);

    let memories: Vec<ListMemoryItem> = filtered
        .into_iter()
        .take(limit)
        .map(|m| ListMemoryItem {
            id: m.id.0.to_string(),
            content: m.experience.content.chars().take(500).collect(),
            memory_type: format!("{:?}", m.experience.experience_type),
            importance: m.importance(),
            tags: m.experience.entities.clone(),
            created_at: m.created_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(ListResponse { memories, total }))
}

// =============================================================================
// HEBBIAN FEEDBACK HANDLERS - Closes the learning loop
// =============================================================================

/// Tracked retrieval - returns tracking info for later feedback
/// POST /api/retrieve/tracked
///
/// Use this when you want to provide feedback later on whether memories were helpful.
/// Returns memory_ids that can be passed to /api/reinforce.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, query = %req.query))]
async fn retrieve_tracked(
    State(state): State<AppState>,
    Json(req): Json<TrackedRetrieveRequest>,
) -> Result<Json<TrackedRetrieveResponse>, AppError> {
    let op_start = std::time::Instant::now();
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let query_text = req.query.clone();
    let limit = req.limit;

    let memories = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let query = MemoryQuery {
                query_text: Some(query_text),
                max_results: limit,
                ..Default::default()
            };
            memory_guard.retrieve(&query).unwrap_or_default()
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    // Extract memory IDs for tracking
    let memory_ids: Vec<String> = memories.iter().map(|m| m.id.0.to_string()).collect();

    // Generate tracking ID (could be stored for audit, but for now just a UUID)
    let tracking_id = uuid::Uuid::new_v4().to_string();

    // Convert to response format
    let recall_memories: Vec<RecallMemory> = memories
        .into_iter()
        .map(|m| RecallMemory {
            id: m.id.0.to_string(),
            content: m.experience.content.clone(),
            importance: m.importance(),
            created_at: m.created_at.to_rfc3339(),
        })
        .collect();

    let count = recall_memories.len();

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_RETRIEVE_DURATION
        .with_label_values(&["tracked"])
        .observe(duration);
    metrics::MEMORY_RETRIEVE_TOTAL
        .with_label_values(&["tracked", "success"])
        .inc();
    metrics::MEMORY_RETRIEVE_RESULTS
        .with_label_values(&["tracked"])
        .observe(count as f64);

    Ok(Json(TrackedRetrieveResponse {
        tracking_id,
        memory_ids,
        memories: recall_memories,
    }))
}

/// Reinforce memories based on task outcome - THIS IS THE HEBBIAN FEEDBACK ENDPOINT
/// POST /api/reinforce
///
/// Call this after using memories to complete a task:
/// - "helpful": Memories that helped → boost importance, strengthen associations
/// - "misleading": Memories that misled → reduce importance, don't strengthen
/// - "neutral": Just record access, mild strengthening
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, outcome = %req.outcome, count = req.memory_ids.len()))]
async fn reinforce_feedback(
    State(state): State<AppState>,
    Json(req): Json<ReinforceFeedbackRequest>,
) -> Result<Json<ReinforceFeedbackResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.memory_ids.is_empty() {
        return Ok(Json(ReinforceFeedbackResponse {
            memories_processed: 0,
            associations_strengthened: 0,
            importance_boosts: 0,
            importance_decays: 0,
        }));
    }

    // Parse outcome
    let outcome_label = req.outcome.to_lowercase();
    let outcome = match outcome_label.as_str() {
        "helpful" => crate::memory::RetrievalOutcome::Helpful,
        "misleading" => crate::memory::RetrievalOutcome::Misleading,
        "neutral" | _ => crate::memory::RetrievalOutcome::Neutral,
    };

    // Convert string IDs to MemoryId
    let memory_ids: Vec<crate::memory::MemoryId> = req
        .memory_ids
        .iter()
        .filter_map(|id| uuid::Uuid::parse_str(id).ok())
        .map(crate::memory::MemoryId)
        .collect();

    if memory_ids.is_empty() {
        return Err(AppError::InvalidInput {
            field: "memory_ids".to_string(),
            reason: "No valid UUIDs provided".to_string(),
        });
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Run reinforcement in blocking task (involves RocksDB writes)
    let stats = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            memory_guard.reinforce_retrieval(&memory_ids, outcome)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    tracing::info!(
        user_id = %req.user_id,
        processed = stats.memories_processed,
        strengthened = stats.associations_strengthened,
        boosts = stats.importance_boosts,
        decays = stats.importance_decays,
        "Hebbian reinforcement applied"
    );

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::HEBBIAN_REINFORCE_DURATION
        .with_label_values(&[&outcome_label])
        .observe(duration);
    metrics::HEBBIAN_REINFORCE_TOTAL
        .with_label_values(&[&outcome_label, "success"])
        .inc();

    Ok(Json(ReinforceFeedbackResponse {
        memories_processed: stats.memories_processed,
        associations_strengthened: stats.associations_strengthened,
        importance_boosts: stats.importance_boosts,
        importance_decays: stats.importance_decays,
    }))
}

// =============================================================================
// SEMANTIC CONSOLIDATION HANDLER - Extract durable facts from episodic memories
// =============================================================================

/// Consolidate memories into semantic facts
/// POST /api/consolidate
///
/// Analyzes memories to extract durable semantic facts (preferences, procedures, patterns).
/// Facts are reinforced when seen multiple times across different memories.
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
async fn consolidate_memories(
    State(state): State<AppState>,
    Json(req): Json<ConsolidateRequest>,
) -> Result<Json<ConsolidateResponse>, AppError> {
    let op_start = std::time::Instant::now();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let min_support = req.min_support;
    let min_age_days = req.min_age_days;

    // Run consolidation in blocking task
    let result = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();

            // Get all memories for consolidation
            let all_memories = memory_guard.get_all_memories()?;

            // Convert SharedMemory to Memory for consolidator
            let memories: Vec<crate::memory::types::Memory> = all_memories
                .into_iter()
                .map(|arc_mem| (*arc_mem).clone())
                .collect();

            // Create consolidator with custom thresholds
            let consolidator =
                crate::memory::SemanticConsolidator::with_thresholds(min_support, min_age_days);

            // Run consolidation
            Ok::<_, anyhow::Error>(consolidator.consolidate(&memories))
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    tracing::info!(
        user_id = %req.user_id,
        memories_processed = result.memories_processed,
        facts_extracted = result.facts_extracted,
        facts_reinforced = result.facts_reinforced,
        "Semantic consolidation complete"
    );

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::CONSOLIDATE_DURATION.observe(duration);
    metrics::CONSOLIDATE_TOTAL
        .with_label_values(&["success"])
        .inc();

    Ok(Json(ConsolidateResponse {
        memories_analyzed: result.memories_processed,
        facts_extracted: result.facts_extracted,
        facts_reinforced: result.facts_reinforced,
        fact_ids: result.new_fact_ids,
    }))
}

/// Batch /api/batch_remember - store multiple memories at once (efficient for bulk)
/// Example: POST /api/batch_remember { "user_id": "agent-1", "memories": [{"content": "..."}, ...] }
#[tracing::instrument(skip(state), fields(user_id = %req.user_id, count = req.memories.len()))]
async fn batch_remember(
    State(state): State<AppState>,
    Json(req): Json<BatchRememberRequest>,
) -> Result<Json<BatchRememberResponse>, AppError> {
    let op_start = std::time::Instant::now();
    let batch_size = req.memories.len();

    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.memories.is_empty() {
        return Ok(Json(BatchRememberResponse {
            ids: vec![],
            success_count: 0,
            error_count: 0,
        }));
    }

    if req.memories.len() > 1000 {
        return Err(AppError::InvalidInput {
            field: "memories".to_string(),
            reason: "Batch size exceeds 1000 limit".to_string(),
        });
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let items = req.memories;
    let (ids, error_count) = {
        let memory = memory.clone();
        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            let mut ids = Vec::with_capacity(items.len());
            let mut errors = 0usize;

            for item in items {
                let experience_type = item
                    .experience_type
                    .as_ref()
                    .and_then(|s| match s.to_lowercase().as_str() {
                        "task" => Some(ExperienceType::Task),
                        "learning" => Some(ExperienceType::Learning),
                        "decision" => Some(ExperienceType::Decision),
                        "error" => Some(ExperienceType::Error),
                        "pattern" => Some(ExperienceType::Pattern),
                        "conversation" => Some(ExperienceType::Conversation),
                        "discovery" => Some(ExperienceType::Discovery),
                        _ => None,
                    })
                    .unwrap_or(ExperienceType::Context);

                let experience = Experience {
                    content: item.content,
                    experience_type,
                    entities: item.tags,
                    ..Default::default()
                };

                match memory_guard.record(experience) {
                    Ok(id) => ids.push(id.0.to_string()),
                    Err(_) => errors += 1,
                }
            }
            (ids, errors)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    let success_count = ids.len();

    // Record metrics
    let duration = op_start.elapsed().as_secs_f64();
    metrics::BATCH_STORE_DURATION.observe(duration);
    metrics::BATCH_STORE_SIZE.observe(batch_size as f64);
    // Also record individual store metrics
    for _ in 0..success_count {
        metrics::MEMORY_STORE_TOTAL
            .with_label_values(&["success"])
            .inc();
    }
    for _ in 0..error_count {
        metrics::MEMORY_STORE_TOTAL
            .with_label_values(&["error"])
            .inc();
    }

    Ok(Json(BatchRememberResponse {
        ids,
        success_count,
        error_count,
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
    let retrieval_mode = "hybrid"; // Default mode

    // Enterprise input validation
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if let Some(ref emb) = req.query_embedding {
        validation::validate_embeddings(emb)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let max_results = req.max_results.unwrap_or(10);
    validation::validate_max_results(max_results).map_validation_err("max_results")?;

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

            // HYBRID RETRIEVAL: Semantic + Graph Boost
            // 1. Semantic similarity is the base score (content matching)
            // 2. Graph activation provides boost for entity-related memories
            // Formula: final_score = semantic_score * (1.0 + graph_boost)
            let memories: Vec<Memory> = if let Some(ref query_text) = query_text {
                // Step 1: Generate query embedding
                let query_embedding = memory_guard
                    .get_embedder()
                    .encode(query_text)
                    .map_err(AppError::Internal)?;

                // Step 2: Extract entities from query for graph lookup using Neural NER
                let query_entities: Vec<(String, NerEntityType)> =
                    match state_clone.neural_ner.extract(query_text) {
                        Ok(entities) => entities
                            .into_iter()
                            .map(|e| (e.text, e.entity_type))
                            .collect(),
                        Err(e) => {
                            tracing::debug!("NER extraction failed in hybrid recall: {}", e);
                            Vec::new()
                        }
                    };

                // Step 3: Build entity activation map from graph
                // This gives us activated memory IDs and their graph scores
                let mut memory_graph_boosts: std::collections::HashMap<uuid::Uuid, f32> =
                    std::collections::HashMap::new();

                // =================================================================
                // SPREADING ACTIVATION (Anderson & Pirolli 1984)
                // Multi-hop activation with exponential decay
                // =================================================================
                const DECAY_RATE: f32 = 0.5; // λ: decay rate per hop
                const MAX_HOPS: usize = 3; // Maximum traversal depth

                if !query_entities.is_empty() {
                    if let Ok(graph) = state_clone.get_user_graph(&user_id) {
                        let graph_guard = graph.read();

                        // Track visited entities to avoid cycles
                        let mut visited_entities: std::collections::HashSet<uuid::Uuid> =
                            std::collections::HashSet::new();

                        // Initialize activation from query entities (hop 0)
                        let mut current_level: Vec<(uuid::Uuid, f32)> = Vec::new();

                        for (entity_name, _) in &query_entities {
                            if let Ok(Some(entity_node)) =
                                graph_guard.find_entity_by_name(entity_name)
                            {
                                // Initial activation = entity salience (IC weight)
                                let initial_activation = entity_node.salience;
                                current_level.push((entity_node.uuid, initial_activation));
                                visited_entities.insert(entity_node.uuid);

                                // Hop 0: Direct episodes get full activation
                                if let Ok(episodes) =
                                    graph_guard.get_episodes_by_entity(&entity_node.uuid)
                                {
                                    for episode in episodes {
                                        let boost = initial_activation * 0.5;
                                        *memory_graph_boosts.entry(episode.uuid).or_insert(0.0) +=
                                            boost;
                                    }
                                }
                            }
                        }

                        // Spread activation through hops 1 to MAX_HOPS
                        for hop in 1..=MAX_HOPS {
                            if current_level.is_empty() {
                                break;
                            }

                            // Decay factor: A(d) = A₀ × e^(-λd)
                            let decay = (-DECAY_RATE * hop as f32).exp();
                            let mut next_level: Vec<(uuid::Uuid, f32)> = Vec::new();

                            for (entity_uuid, activation) in &current_level {
                                // Spread to connected entities via edges
                                if let Ok(edges) = graph_guard.get_entity_relationships(entity_uuid)
                                {
                                    for edge in edges {
                                        let connected_uuid = if edge.from_entity == *entity_uuid {
                                            edge.to_entity
                                        } else {
                                            edge.from_entity
                                        };

                                        // Skip already visited entities
                                        if visited_entities.contains(&connected_uuid) {
                                            continue;
                                        }
                                        visited_entities.insert(connected_uuid);

                                        // Spread activation = parent × edge_strength × decay
                                        let spread_activation = activation * edge.strength * decay;

                                        // Get connected entity's salience for weighting
                                        let connected_salience = graph_guard
                                            .get_entity(&connected_uuid)
                                            .ok()
                                            .flatten()
                                            .map(|e| e.salience)
                                            .unwrap_or(0.3);

                                        // Boost connected episodes with spread activation
                                        if let Ok(connected_episodes) =
                                            graph_guard.get_episodes_by_entity(&connected_uuid)
                                        {
                                            for episode in connected_episodes {
                                                // Boost = spread_activation × connected_salience
                                                let boost =
                                                    spread_activation * connected_salience * 0.5;
                                                *memory_graph_boosts
                                                    .entry(episode.uuid)
                                                    .or_insert(0.0) += boost;
                                            }
                                        }

                                        // Add to next level for further spreading
                                        next_level.push((
                                            connected_uuid,
                                            spread_activation * connected_salience,
                                        ));
                                    }
                                }
                            }

                            // Move to next level
                            current_level = next_level;
                        }
                    }
                }

                // Step 4: Get all memories and compute hybrid scores
                let all_memories = memory_guard
                    .get_all_memories()
                    .map_err(AppError::Internal)?;

                // Step 5: Score each memory by semantic + graph boost
                let mut scored_memories: Vec<(f32, Memory)> = all_memories
                    .into_iter()
                    .filter_map(|shared_mem| {
                        let memory = (*shared_mem).clone();

                        // Get embedding
                        let mem_embedding = if let Some(emb) = &memory.experience.embeddings {
                            emb.clone()
                        } else {
                            return None;
                        };

                        // Compute cosine similarity (base score)
                        let semantic_score = cosine_similarity(&query_embedding, &mem_embedding);

                        // Get graph boost (if any) - use UUID directly (memory.id.0 is already Uuid)
                        let graph_boost = memory_graph_boosts
                            .get(&memory.id.0)
                            .copied()
                            .unwrap_or(0.0)
                            .min(1.0); // Cap boost at 100%

                        // Hybrid score: semantic * (1 + graph_boost)
                        // This ensures semantic match is primary, graph only boosts
                        let final_score = semantic_score * (1.0 + graph_boost);

                        Some((final_score, memory))
                    })
                    .collect();

                // Step 6: Sort by final hybrid score (descending)
                scored_memories
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Step 7: Take top-k and set scores
                scored_memories
                    .into_iter()
                    .take(query.max_results)
                    .map(|(score, mut mem)| {
                        mem.score = Some(score);
                        mem
                    })
                    .collect()
            } else {
                // Fallback to traditional retrieval
                let shared_memories = memory_guard.retrieve(&query).map_err(AppError::Internal)?;

                // Convert Arc<Memory> to owned Memory
                let mut memories: Vec<Memory> =
                    shared_memories.iter().map(|m| (**m).clone()).collect();

                // If query embedding provided, re-rank by semantic similarity
                if let Some(query_emb) = &query_embedding {
                    let candidates: Vec<(Vec<f32>, &Memory)> = memories
                        .iter()
                        .filter_map(|m| {
                            m.experience.embeddings.as_ref().map(|emb| (emb.clone(), m))
                        })
                        .collect();

                    let ranked =
                        top_k_similar(query_emb, &candidates, max_results_val.unwrap_or(10));

                    // Create new vec with scores populated
                    ranked
                        .into_iter()
                        .map(|(score, m)| {
                            let mut mem = m.clone();
                            mem.score = Some(score);
                            mem
                        })
                        .collect()
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

    // SSE: Emit real-time event for dashboard (only if results found)
    if count > 0 {
        state.emit_event(MemoryEvent {
            event_type: "RETRIEVE".to_string(),
            timestamp: chrono::Utc::now(),
            user_id: req.user_id.clone(),
            memory_id: None,
            content_preview: req
                .query_text
                .as_ref()
                .map(|q| q.chars().take(100).collect()),
            experience_type: None,
            importance: None,
            count: Some(count),
        });
    }

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Get user statistics
async fn get_user_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<MemoryStats>, AppError> {
    let stats = state.get_stats(&user_id).map_err(AppError::Internal)?;

    Ok(Json(stats))
}

/// Delete user data (GDPR)
async fn delete_user(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<StatusCode, AppError> {
    state.forget_user(&user_id).map_err(AppError::Internal)?;

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
    let user_id = params
        .get("user_id")
        .ok_or_else(|| AppError::InvalidInput {
            field: "user_id".to_string(),
            reason: "user_id required".to_string(),
        })?;

    // Enterprise input validation
    validation::validate_user_id(user_id).map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state.get_user_memory(user_id).map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Parse memory ID (already validated above)
    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let _memory_id_obj = MemoryId(mem_id);

    // Search for memory
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.retrieve(&query).map_err(AppError::Internal)?;

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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    validation::validate_content(&req.content, false).map_validation_err("content")?;

    if let Some(ref emb) = req.embeddings {
        validation::validate_embeddings(emb)
            .map_err(|e| AppError::InvalidEmbeddings(e.to_string()))?;
    }

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    // Get current memory to preserve metadata
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.retrieve(&query).map_err(AppError::Internal)?;

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
        &format!("Updated memory content: {content_preview}"),
    );

    Ok(StatusCode::OK)
}

/// Delete specific memory
async fn delete_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<StatusCode, AppError> {
    let user_id = params
        .get("user_id")
        .ok_or_else(|| AppError::InvalidInput {
            field: "user_id".to_string(),
            reason: "user_id required".to_string(),
        })?;

    // Enterprise input validation
    validation::validate_user_id(user_id).map_validation_err("user_id")?;

    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state.get_user_memory(user_id).map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Delete by ID - escape UUID to treat as literal string, not regex
    // UUIDs contain hyphens which are regex metacharacters, so we must escape them
    let escaped_pattern = regex::escape(&memory_id);
    memory_guard
        .forget(memory::ForgetCriteria::Pattern(escaped_pattern))
        .map_err(AppError::Internal)?;

    // Enterprise audit logging
    state.log_event(user_id, "DELETE", &memory_id, "Memory deleted");

    // Return 200 OK instead of 204 NO_CONTENT so Python client can verify success
    Ok(StatusCode::OK)
}

/// Get all memories for a user with optional filters
#[derive(Debug, Deserialize)]
struct GetAllRequest {
    user_id: String,
    /// Maximum number of results to return (default: 100)
    limit: Option<usize>,
    /// Filter by minimum importance score (0.0 to 1.0)
    importance_threshold: Option<f32>,
    /// Filter by tags (returns memories matching ANY of these tags)
    tags: Option<Vec<String>>,
    /// Filter by memory type (e.g., "Decision", "Learning", "Error")
    memory_type: Option<String>,
    /// Filter by memories created after this timestamp (ISO 8601)
    created_after: Option<chrono::DateTime<chrono::Utc>>,
    /// Filter by memories created before this timestamp (ISO 8601)
    created_before: Option<chrono::DateTime<chrono::Utc>>,
}

async fn get_all_memories(
    State(state): State<AppState>,
    Json(req): Json<GetAllRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Parse memory_type string to ExperienceType if provided
    let experience_types = req.memory_type.as_ref().map(|type_str| {
        vec![match type_str.to_lowercase().as_str() {
            "observation" => ExperienceType::Observation,
            "decision" => ExperienceType::Decision,
            "learning" => ExperienceType::Learning,
            "error" => ExperienceType::Error,
            "discovery" => ExperienceType::Discovery,
            "pattern" => ExperienceType::Pattern,
            "context" => ExperienceType::Context,
            "task" => ExperienceType::Task,
            "codeedit" => ExperienceType::CodeEdit,
            "fileaccess" => ExperienceType::FileAccess,
            "search" => ExperienceType::Search,
            "command" => ExperienceType::Command,
            "conversation" => ExperienceType::Conversation,
            _ => ExperienceType::Observation, // Default fallback
        }]
    });

    // Build time range from created_after/created_before
    let time_range = match (req.created_after, req.created_before) {
        (Some(after), Some(before)) => Some((after, before)),
        (Some(after), None) => Some((after, chrono::Utc::now())),
        (None, Some(before)) => Some((chrono::DateTime::<chrono::Utc>::MIN_UTC, before)),
        (None, None) => None,
    };

    let query = MemoryQuery {
        max_results: req.limit.unwrap_or(100),
        importance_threshold: req.importance_threshold,
        tags: req.tags,
        experience_types,
        time_range,
        ..Default::default()
    };

    let shared_memories = memory_guard.retrieve(&query).map_err(AppError::Internal)?;

    // Convert Arc<Memory> to owned Memory for response
    let memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();

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
    timestamp: String, // ISO 8601 format
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

        tokio::task::spawn_blocking(move || state.get_history(&user_id, memory_id.as_deref()))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
    };

    let history_events: Vec<HistoryEvent> = events
        .iter()
        .map(|e| HistoryEvent {
            timestamp: e.timestamp.to_rfc3339(),
            event_type: e.event_type.clone(),
            memory_id: e.memory_id.clone(),
            details: e.details.clone(),
        })
        .collect();

    Ok(Json(HistoryResponse {
        events: history_events,
    }))
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let _memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Validate memory_id format
    let _memory_id = MemoryId(
        uuid::Uuid::parse_str(&req.memory_id)
            .map_err(|_| AppError::InvalidMemoryId(req.memory_id.clone()))?,
    );

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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.write();

    let rel_uuid =
        uuid::Uuid::parse_str(&req.relationship_uuid).map_err(|_| AppError::InvalidInput {
            field: "relationship_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    graph_guard
        .invalidate_relationship(&rel_uuid)
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();

    let episode_uuid =
        uuid::Uuid::parse_str(&req.episode_uuid).map_err(|_| AppError::InvalidInput {
            field: "episode_uuid".to_string(),
            reason: "Invalid UUID format".to_string(),
        })?;

    let episode = graph_guard
        .get_episode(&episode_uuid)
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
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
            end: end_dt,
        });
    }

    if let (Some(min), Some(max)) = (req.min_importance, req.max_importance) {
        criterias.push(memory::storage::SearchCriteria::ByImportance { min, max });
    }

    // Execute combined search
    let criteria = if criterias.len() == 1 {
        criterias
            .into_iter()
            .next()
            .expect("Criteria list has exactly one element")
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
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;

    let graph_guard = graph.read();

    // First find the entity
    let entity = graph_guard
        .find_entity_by_name(&req.entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| {
            AppError::MemoryNotFound(format!("Entity not found: {}", req.entity_name))
        })?;

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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let memory_id = MemoryId(
        uuid::Uuid::parse_str(&req.memory_id)
            .map_err(|_| AppError::InvalidMemoryId(req.memory_id.clone()))?,
    );

    // Get the memory
    let memory = memory_guard
        .get_memory(&memory_id)
        .map_err(AppError::Internal)?;

    if !memory.compressed {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "Memory is not compressed",
            "was_compressed": false
        })));
    }

    // Decompress using compression pipeline
    let decompressed = memory_guard
        .decompress_memory(&memory)
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let stats = memory_guard
        .get_storage_stats()
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::OlderThan(req.days_old))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_AGE",
        &format!("{} days", req.days_old),
        &format!("Forgot {count} memories"),
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.threshold < 0.0 || req.threshold > 1.0 {
        return Err(AppError::InvalidInput {
            field: "threshold".to_string(),
            reason: "Must be between 0.0 and 1.0".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::LowImportance(req.threshold))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_IMPORTANCE",
        &format!("threshold {}", req.threshold),
        &format!("Forgot {count} memories"),
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::Pattern(req.pattern.clone()))
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "FORGET_BY_PATTERN",
        &req.pattern,
        &format!("Forgot {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "forgotten_count": count,
        "pattern": req.pattern
    })))
}

// ====== Bulk Delete Endpoints ======

/// Bulk delete memories by filters (tags, type, date range)
#[derive(Debug, Deserialize)]
struct BulkDeleteRequest {
    user_id: String,
    /// Delete memories matching ANY of these tags
    tags: Option<Vec<String>>,
    /// Delete memories of this type
    memory_type: Option<String>,
    /// Delete memories created after this timestamp
    created_after: Option<chrono::DateTime<chrono::Utc>>,
    /// Delete memories created before this timestamp
    created_before: Option<chrono::DateTime<chrono::Utc>>,
}

async fn bulk_delete_memories(
    State(state): State<AppState>,
    Json(req): Json<BulkDeleteRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let mut total_count = 0;

    // Delete by tags if specified
    if let Some(ref tags) = req.tags {
        if !tags.is_empty() {
            let count = memory_guard
                .forget(memory::ForgetCriteria::ByTags(tags.clone()))
                .map_err(AppError::Internal)?;
            total_count += count;
        }
    }

    // Delete by type if specified
    if let Some(ref type_str) = req.memory_type {
        let exp_type = match type_str.to_lowercase().as_str() {
            "observation" => ExperienceType::Observation,
            "decision" => ExperienceType::Decision,
            "learning" => ExperienceType::Learning,
            "error" => ExperienceType::Error,
            "discovery" => ExperienceType::Discovery,
            "pattern" => ExperienceType::Pattern,
            "context" => ExperienceType::Context,
            "task" => ExperienceType::Task,
            "codeedit" => ExperienceType::CodeEdit,
            "fileaccess" => ExperienceType::FileAccess,
            "search" => ExperienceType::Search,
            "command" => ExperienceType::Command,
            "conversation" => ExperienceType::Conversation,
            _ => {
                return Err(AppError::InvalidInput {
                    field: "memory_type".to_string(),
                    reason: format!("Invalid memory type: {type_str}"),
                })
            }
        };
        let count = memory_guard
            .forget(memory::ForgetCriteria::ByType(exp_type))
            .map_err(AppError::Internal)?;
        total_count += count;
    }

    // Delete by date range if specified
    if req.created_after.is_some() || req.created_before.is_some() {
        let start = req
            .created_after
            .unwrap_or(chrono::DateTime::<chrono::Utc>::MIN_UTC);
        let end = req.created_before.unwrap_or(chrono::Utc::now());
        let count = memory_guard
            .forget(memory::ForgetCriteria::ByDateRange { start, end })
            .map_err(AppError::Internal)?;
        total_count += count;
    }

    state.log_event(
        &req.user_id,
        "BULK_DELETE",
        "multiple",
        &format!("Deleted {total_count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": total_count
    })))
}

/// Clear ALL memories for a user (GDPR compliance - right to erasure)
#[derive(Debug, Deserialize)]
struct ClearAllRequest {
    user_id: String,
    /// Safety confirmation - must be "CONFIRM" to proceed
    confirm: String,
}

async fn clear_all_memories(
    State(state): State<AppState>,
    Json(req): Json<ClearAllRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    // Safety check - require explicit confirmation
    if req.confirm != "CONFIRM" {
        return Err(AppError::InvalidInput {
            field: "confirm".to_string(),
            reason: "Must provide confirm: \"CONFIRM\" to clear all memories".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let count = memory_guard
        .forget(memory::ForgetCriteria::All)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "CLEAR_ALL",
        "GDPR",
        &format!("GDPR erasure: deleted {count} memories"),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": count,
        "message": "All memories have been permanently deleted (GDPR erasure)"
    })))
}

/// PATCH endpoint for partial memory updates
#[derive(Debug, Deserialize)]
struct PatchMemoryRequest {
    user_id: String,
    /// New content (optional)
    content: Option<String>,
    /// New/additional tags (optional)
    tags: Option<Vec<String>>,
    /// New memory type (optional)
    memory_type: Option<String>,
}

async fn patch_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(req): Json<PatchMemoryRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_memory_id(&memory_id)
        .map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    let mem_id =
        uuid::Uuid::parse_str(&memory_id).map_err(|e| AppError::InvalidMemoryId(e.to_string()))?;

    // Get current memory
    let query = MemoryQuery {
        max_results: 1000,
        ..Default::default()
    };

    let all_memories = memory_guard.retrieve(&query).map_err(AppError::Internal)?;

    let shared_memory = all_memories
        .into_iter()
        .find(|m| m.id.0 == mem_id)
        .ok_or_else(|| AppError::MemoryNotFound(memory_id.clone()))?;

    // Clone out of Arc to get mutable Memory
    let mut current_memory = (*shared_memory).clone();

    let mut changes = Vec::new();

    // Update content if provided
    if let Some(ref new_content) = req.content {
        validation::validate_content(new_content, false).map_validation_err("content")?;
        current_memory.experience.content = new_content.clone();
        // Re-generate embeddings for new content
        current_memory.experience.embeddings = None;
        changes.push("content");
    }

    // Update tags if provided (add to existing entities)
    if let Some(ref new_tags) = req.tags {
        for tag in new_tags {
            if !current_memory.experience.entities.contains(tag) {
                current_memory.experience.entities.push(tag.clone());
            }
        }
        changes.push("tags");
    }

    // Update type if provided
    if let Some(ref type_str) = req.memory_type {
        current_memory.experience.experience_type = match type_str.to_lowercase().as_str() {
            "observation" => ExperienceType::Observation,
            "decision" => ExperienceType::Decision,
            "learning" => ExperienceType::Learning,
            "error" => ExperienceType::Error,
            "discovery" => ExperienceType::Discovery,
            "pattern" => ExperienceType::Pattern,
            "context" => ExperienceType::Context,
            "task" => ExperienceType::Task,
            "codeedit" => ExperienceType::CodeEdit,
            "fileaccess" => ExperienceType::FileAccess,
            "search" => ExperienceType::Search,
            "command" => ExperienceType::Command,
            "conversation" => ExperienceType::Conversation,
            _ => {
                return Err(AppError::InvalidInput {
                    field: "memory_type".to_string(),
                    reason: format!("Invalid memory type: {type_str}"),
                })
            }
        };
        changes.push("type");
    }

    if changes.is_empty() {
        return Err(AppError::InvalidInput {
            field: "body".to_string(),
            reason: "No fields to update provided".to_string(),
        });
    }

    // Re-record (will update in storage and re-index)
    let experience = current_memory.experience.clone();
    memory_guard
        .record(experience)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "PATCH",
        &memory_id,
        &format!("Updated fields: {}", changes.join(", ")),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "memory_id": memory_id,
        "updated_fields": changes
    })))
}

// ============================================================================
// RECALL BY TAGS / DATE - Convenience Endpoints
// ============================================================================

/// Recall memories by tags
#[derive(Debug, Deserialize)]
struct RecallByTagsRequest {
    user_id: String,
    /// Tags to search for (returns memories matching ANY of these tags)
    tags: Vec<String>,
    /// Maximum number of results (default: 50)
    limit: Option<usize>,
}

async fn recall_by_tags(
    State(state): State<AppState>,
    Json(req): Json<RecallByTagsRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.tags.is_empty() {
        return Err(AppError::InvalidInput {
            field: "tags".to_string(),
            reason: "At least one tag must be provided".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let limit = req.limit.unwrap_or(50);

    // Search using tag criteria
    let criteria = memory::storage::SearchCriteria::ByTags(req.tags.clone());
    let memories = memory_guard
        .advanced_search(criteria)
        .map_err(AppError::Internal)?;

    // Apply limit
    let memories: Vec<_> = memories.into_iter().take(limit).collect();
    let count = memories.len();

    info!(
        "📋 Recall by tags: user={}, tags={:?}, found={}",
        req.user_id, req.tags, count
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Recall memories by date range
#[derive(Debug, Deserialize)]
struct RecallByDateRequest {
    user_id: String,
    /// Start of date range (inclusive) - ISO 8601 format
    start: chrono::DateTime<chrono::Utc>,
    /// End of date range (inclusive) - ISO 8601 format
    end: chrono::DateTime<chrono::Utc>,
    /// Maximum number of results (default: 50)
    limit: Option<usize>,
}

async fn recall_by_date(
    State(state): State<AppState>,
    Json(req): Json<RecallByDateRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.end < req.start {
        return Err(AppError::InvalidInput {
            field: "end".to_string(),
            reason: "End date must be after start date".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();
    let limit = req.limit.unwrap_or(50);

    // Search using date range criteria
    let criteria = memory::storage::SearchCriteria::ByDate {
        start: req.start,
        end: req.end,
    };
    let memories = memory_guard
        .advanced_search(criteria)
        .map_err(AppError::Internal)?;

    // Apply limit
    let memories: Vec<_> = memories.into_iter().take(limit).collect();
    let count = memories.len();

    info!(
        "📅 Recall by date: user={}, start={}, end={}, found={}",
        req.user_id, req.start, req.end, count
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

/// Forget memories by tags
#[derive(Debug, Deserialize)]
struct ForgetByTagsRequest {
    user_id: String,
    /// Tags to match for deletion (deletes memories matching ANY of these tags)
    tags: Vec<String>,
}

async fn forget_by_tags(
    State(state): State<AppState>,
    Json(req): Json<ForgetByTagsRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.tags.is_empty() {
        return Err(AppError::InvalidInput {
            field: "tags".to_string(),
            reason: "At least one tag must be provided".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let deleted_count = memory_guard
        .forget(memory::ForgetCriteria::ByTags(req.tags.clone()))
        .map_err(AppError::Internal)?;

    info!(
        "🏷️ Forget by tags: user={}, tags={:?}, deleted={}",
        req.user_id, req.tags, deleted_count
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "tags": req.tags
    })))
}

/// Forget memories by date range
#[derive(Debug, Deserialize)]
struct ForgetByDateRequest {
    user_id: String,
    /// Start of date range (inclusive) - ISO 8601 format
    start: chrono::DateTime<chrono::Utc>,
    /// End of date range (inclusive) - ISO 8601 format
    end: chrono::DateTime<chrono::Utc>,
}

async fn forget_by_date(
    State(state): State<AppState>,
    Json(req): Json<ForgetByDateRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if req.end < req.start {
        return Err(AppError::InvalidInput {
            field: "end".to_string(),
            reason: "End date must be after start date".to_string(),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let deleted_count = memory_guard
        .forget(memory::ForgetCriteria::ByDateRange {
            start: req.start,
            end: req.end,
        })
        .map_err(AppError::Internal)?;

    info!(
        "📅 Forget by date: user={}, start={}, end={}, deleted={}",
        req.user_id, req.start, req.end, deleted_count
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted_count,
        "start": req.start,
        "end": req.end
    })))
}

/// Advanced multi-modal retrieval
#[derive(Debug, Deserialize)]
struct MultiModalSearchRequest {
    user_id: String,
    query_text: String,
    mode: String, // "similarity", "temporal", "causal", "associative", "hybrid"
    limit: Option<usize>,
}

async fn multimodal_search(
    State(state): State<AppState>,
    Json(req): Json<MultiModalSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
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

    let shared_memories = memory_guard.retrieve(&query).map_err(AppError::Internal)?;

    // Convert Arc<Memory> to owned Memory for response
    let memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();

    let count = memories.len();

    state.log_event(
        &req.user_id,
        "MULTIMODAL_SEARCH",
        &req.mode,
        &format!("Retrieved {} memories using {} mode", count, req.mode),
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    // Build retrieval mode
    let retrieval_mode = match req.mode.as_str() {
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        "hybrid" => memory::RetrievalMode::Hybrid,
        "similarity" => memory::RetrievalMode::Similarity,
        _ => {
            return Err(AppError::InvalidInput {
                field: "mode".to_string(),
                reason: "Invalid mode. Use: spatial, mission, action_outcome, hybrid, similarity"
                    .to_string(),
            })
        }
    };

    // Build geo filter if spatial coordinates provided
    let geo_filter = match (req.lat, req.lon, req.radius_meters) {
        (Some(lat), Some(lon), Some(radius)) => Some(memory::GeoFilter::new(lat, lon, radius)),
        _ => None,
    };

    // Build reward range
    let reward_range = match (req.min_reward, req.max_reward) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, 1.0)),
        (None, Some(max)) => Some((-1.0, max)),
        _ => None,
    };

    // Validate spatial mode has coordinates
    if matches!(retrieval_mode, memory::RetrievalMode::Spatial) && geo_filter.is_none() {
        return Err(AppError::InvalidInput {
            field: "lat/lon/radius_meters".to_string(),
            reason: "Spatial mode requires lat, lon, and radius_meters".to_string(),
        });
    }

    // Validate mission mode has mission_id
    if matches!(retrieval_mode, memory::RetrievalMode::Mission) && req.mission_id.is_none() {
        return Err(AppError::InvalidInput {
            field: "mission_id".to_string(),
            reason: "Mission mode requires mission_id".to_string(),
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

    let shared_memories = memory_guard.retrieve(&query).map_err(AppError::Internal)?;

    let memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();

    let count = memories.len();

    state.log_event(
        &req.user_id,
        "ROBOTICS_SEARCH",
        &req.mode,
        &format!(
            "Retrieved {} robotics memories (robot={:?}, mission={:?})",
            count, req.robot_id, req.mission_id
        ),
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    validation::validate_entity(&req.name).map_validation_err("name")?;

    validation::validate_entity(&req.label).map_validation_err("label")?;

    let graph = state
        .get_user_graph(&req.user_id)
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

    // Detect if proper noun (simple heuristic: first char is uppercase)
    let is_proper_noun = req
        .name
        .chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false);

    // Calculate base salience based on entity type and proper noun status
    let type_salience = match &entity_label {
        graph_memory::EntityLabel::Person => 0.8,
        graph_memory::EntityLabel::Organization => 0.7,
        graph_memory::EntityLabel::Location => 0.6,
        graph_memory::EntityLabel::Technology => 0.6,
        graph_memory::EntityLabel::Product => 0.7,
        graph_memory::EntityLabel::Event => 0.6,
        graph_memory::EntityLabel::Skill => 0.5,
        graph_memory::EntityLabel::Concept => 0.4,
        graph_memory::EntityLabel::Date => 0.3,
        graph_memory::EntityLabel::Other(_) => 0.3,
    };
    let salience = if is_proper_noun {
        (type_salience * 1.2_f32).min(1.0_f32)
    } else {
        type_salience
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
        salience,
        is_proper_noun,
    };

    let entity_uuid = graph_guard
        .add_entity(entity)
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    validation::validate_entity(&req.from_entity_name).map_validation_err("from_entity_name")?;

    validation::validate_entity(&req.to_entity_name).map_validation_err("to_entity_name")?;

    if let Some(strength) = req.strength {
        validation::validate_relationship_strength(strength).map_validation_err("strength")?;
    }

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    // Find entities
    let from_entity = graph_guard
        .find_entity_by_name(&req.from_entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| {
            AppError::MemoryNotFound(format!("Entity not found: {}", req.from_entity_name))
        })?;

    let to_entity = graph_guard
        .find_entity_by_name(&req.to_entity_name)
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        .ok_or_else(|| {
            AppError::MemoryNotFound(format!("Entity not found: {}", req.to_entity_name))
        })?;

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
        // Hebbian plasticity fields (new synapses start fresh)
        last_activated: chrono::Utc::now(),
        activation_count: 1, // First activation
        potentiated: false,
    };

    let edge_uuid = graph_guard
        .add_relationship(edge)
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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.read();

    let entities = graph_guard
        .get_all_entities()
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    let limit = req.limit.unwrap_or(100);
    let entities: Vec<_> = entities.into_iter().take(limit).collect();
    let count = entities.len();

    Ok(Json(serde_json::json!({
        "entities": entities,
        "count": count
    })))
}

// ====== Memory Universe Visualization API ======

/// Get the Memory Universe visualization for a user's knowledge graph.
/// Returns entities as "stars" with 3D positions based on their relationships,
/// sized by salience (gravitational mass), and colored by entity type.
async fn get_memory_universe(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<graph_memory::MemoryUniverse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let graph = state.get_user_graph(&user_id).map_err(AppError::Internal)?;

    let graph_guard = graph.read();
    let universe = graph_guard
        .get_universe()
        .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?;

    Ok(Json(universe))
}

// ====== Brain State Visualization API ======

/// Brain state response with memories organized by tier and activation levels
#[derive(Debug, Serialize)]
struct BrainStateResponse {
    working_memory: Vec<MemoryNeuron>,
    session_memory: Vec<MemoryNeuron>,
    longterm_memory: Vec<MemoryNeuron>,
    stats: BrainStats,
}

#[derive(Debug, Serialize)]
struct MemoryNeuron {
    id: String,
    content_preview: String,
    activation: f32,
    importance: f32,
    tier: String,
    access_count: u32,
    created_at: String,
}

#[derive(Debug, Serialize)]
struct BrainStats {
    total_memories: usize,
    working_count: usize,
    session_count: usize,
    longterm_count: usize,
    avg_activation: f32,
    avg_importance: f32,
}

/// Get brain state visualization - shows all memories with activation levels by tier
async fn get_brain_state(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<BrainStateResponse>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();

    // Gather memories from all tiers using public accessors
    let mut working_memory = Vec::new();
    let mut session_memory = Vec::new();
    let mut longterm_memory = Vec::new();
    let mut total_activation = 0.0f32;
    let mut total_importance = 0.0f32;

    // Get working memory via public accessor
    for mem in memory_guard.get_working_memories() {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "working".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        working_memory.push(neuron);
    }

    // Get session memory via public accessor
    for mem in memory_guard.get_session_memories() {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "session".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        session_memory.push(neuron);
    }

    // Get sample of longterm memory via public accessor (limit to avoid huge responses)
    let longterm_sample = memory_guard.get_longterm_memories(50).unwrap_or_default();
    for mem in longterm_sample {
        let neuron = MemoryNeuron {
            id: mem.id.0.to_string(),
            content_preview: mem.experience.content.chars().take(100).collect(),
            activation: mem.activation(),
            importance: mem.importance(),
            tier: "longterm".to_string(),
            access_count: mem.metadata_snapshot().access_count,
            created_at: mem.created_at.to_rfc3339(),
        };
        total_activation += neuron.activation;
        total_importance += neuron.importance;
        longterm_memory.push(neuron);
    }

    let total_count = working_memory.len() + session_memory.len() + longterm_memory.len();
    let stats = BrainStats {
        total_memories: total_count,
        working_count: working_memory.len(),
        session_count: session_memory.len(),
        longterm_count: longterm_memory.len(),
        avg_activation: if total_count > 0 {
            total_activation / total_count as f32
        } else {
            0.0
        },
        avg_importance: if total_count > 0 {
            total_importance / total_count as f32
        } else {
            0.0
        },
    };

    Ok(Json(BrainStateResponse {
        working_memory,
        session_memory,
        longterm_memory,
        stats,
    }))
}

// ====== Memory Visualization API Endpoints ======

/// Get visualization statistics for a user's memory graph
async fn get_visualization_stats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<VisualizationStats>, AppError> {
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

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
    validation::validate_user_id(&user_id).map_validation_err("user_id")?;

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
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory.read();
    let stats = memory_guard
        .build_visualization_graph()
        .map_err(AppError::Internal)?;

    Ok(Json(stats))
}

#[tokio::main]
async fn main() -> Result<()> {
    // P1.6: Initialize distributed tracing with OpenTelemetry (optional)
    #[cfg(feature = "telemetry")]
    {
        tracing_setup::init_tracing().expect("Failed to initialize tracing");
    }
    #[cfg(not(feature = "telemetry"))]
    {
        // Default to info level if RUST_LOG not set (user-friendly default)
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "shodh_memory=info,tower_http=warn");
        }
        tracing_subscriber::fmt::init();
    }

    // Print startup banner (always visible, regardless of log level)
    eprintln!();
    eprintln!("  ╔═══════════════════════════════════════════════════╗");
    eprintln!(
        "  ║         🧠 Shodh-Memory Server v{}          ║",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  ║       Cognitive Memory for AI Agents              ║");
    eprintln!("  ╚═══════════════════════════════════════════════════╝");
    eprintln!();

    // P1.1: Register Prometheus metrics
    metrics::register_metrics().expect("Failed to register metrics");

    // Load configuration from environment
    let server_config = ServerConfig::from_env();

    // Print configuration (always visible)
    eprintln!("  📋 Configuration:");
    eprintln!(
        "     Mode:    {}",
        if server_config.is_production {
            "PRODUCTION"
        } else {
            "Development"
        }
    );
    eprintln!("     Port:    {}", server_config.port);
    eprintln!("     Storage: {}", server_config.storage_path.display());
    eprintln!();

    // Create memory manager with config
    let manager = Arc::new(MultiUserMemoryManager::new(
        server_config.storage_path.clone(),
        server_config.clone(),
    )?);

    // Print storage statistics (always visible)
    let storage_path = &server_config.storage_path;
    if storage_path.exists() {
        let disk_usage = calculate_dir_size(storage_path);
        let user_count = count_user_directories(storage_path);
        eprintln!("  💾 Storage Statistics:");
        eprintln!(
            "     Location:  {}",
            storage_path
                .canonicalize()
                .unwrap_or_else(|_| storage_path.clone())
                .display()
        );
        eprintln!("     Disk used: {}", format_bytes(disk_usage));
        eprintln!("     Users:     {}", user_count);
        eprintln!();
    } else {
        eprintln!("  💾 Storage: New database (no existing data)");
        eprintln!();
    }

    // Keep a reference to manager for shutdown cleanup (clone BEFORE moving into router)
    let manager_for_shutdown = Arc::clone(&manager);

    // Start background maintenance scheduler (consolidation, activation decay, graph pruning)
    let maintenance_interval = server_config.maintenance_interval_secs;
    let manager_for_maintenance = Arc::clone(&manager);
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(maintenance_interval));

        loop {
            interval.tick().await;

            // Run maintenance + periodic flush in blocking thread pool
            // This ensures durability in async write mode (data flushed every maintenance cycle)
            let manager_clone = Arc::clone(&manager_for_maintenance);
            tokio::task::spawn_blocking(move || {
                manager_clone.run_maintenance_all_users();
                // Periodic flush ensures data durability for async write mode
                // In sync mode this is fast (little dirty data), in async mode this is critical
                if let Err(e) = manager_clone.flush_all_databases() {
                    tracing::warn!("Periodic flush failed: {}", e);
                }
            });
        }
    });
    info!(
        "🔄 Background maintenance scheduler started (interval: {}s)",
        maintenance_interval
    );

    // Configure rate limiting from config
    let governor_conf = GovernorConfigBuilder::default()
        .per_second(server_config.rate_limit_per_second)
        .burst_size(server_config.rate_limit_burst)
        .finish()
        .expect("Failed to build governor rate limiter configuration");

    let governor_layer = GovernorLayer::new(governor_conf);

    info!(
        "⚡ Rate limiting enabled: {} req/sec, burst of {}",
        server_config.rate_limit_per_second, server_config.rate_limit_burst
    );

    // Build CORS layer from configuration
    let cors = server_config.cors.to_layer();

    // Build router with rate limiting
    // Split into public (no auth) and protected (auth required) routes

    // Protected API routes - require authentication
    let protected_routes = Router::new()
        // Core endpoints
        .route("/api/record", post(record_experience))
        .route("/api/retrieve", post(retrieve_memories))
        // Simplified LLM-friendly endpoints (effortless API)
        .route("/api/remember", post(remember))
        .route("/api/recall", post(recall))
        .route("/api/batch_remember", post(batch_remember))
        // Hebbian Feedback Loop - Wire up learning from task outcomes
        .route("/api/retrieve/tracked", post(retrieve_tracked))
        .route("/api/reinforce", post(reinforce_feedback))
        // Semantic Consolidation - Extract durable facts from episodic memories
        .route("/api/consolidate", post(consolidate_memories))
        // User management
        .route("/api/users", get(list_users))
        .route("/api/users/{user_id}/stats", get(get_user_stats))
        .route("/api/users/{user_id}", delete(delete_user))
        // Memory CRUD
        .route("/api/memory/{memory_id}", get(get_memory))
        .route("/api/memory/{memory_id}", axum::routing::put(update_memory))
        .route("/api/memory/{memory_id}", delete(delete_memory))
        .route(
            "/api/memory/{memory_id}",
            axum::routing::patch(patch_memory),
        )
        .route("/api/memories", post(get_all_memories))
        .route("/api/memories/history", post(get_history))
        .route("/api/memories/bulk", post(bulk_delete_memories))
        .route("/api/memories/clear", post(clear_all_memories))
        // Compression & Storage Management
        .route("/api/memory/compress", post(compress_memory))
        .route("/api/memory/decompress", post(decompress_memory))
        .route("/api/storage/stats", post(get_storage_stats))
        .route("/api/storage/uncompressed", post(get_uncompressed_old))
        // Forgetting Operations
        .route("/api/forget/age", post(forget_by_age))
        .route("/api/forget/importance", post(forget_by_importance))
        .route("/api/forget/pattern", post(forget_by_pattern))
        .route("/api/forget/tags", post(forget_by_tags))
        .route("/api/forget/date", post(forget_by_date))
        // Recall by filters (tag/date convenience endpoints)
        .route("/api/recall/tags", post(recall_by_tags))
        .route("/api/recall/date", post(recall_by_date))
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
        .route(
            "/api/graph/relationship/invalidate",
            post(invalidate_relationship),
        )
        .route("/api/graph/traverse", post(traverse_graph))
        // Graph Memory - Episodes
        .route("/api/graph/episode/get", post(get_episode))
        // Memory Universe Visualization (3D graph with salience-based sizing)
        .route("/api/graph/{user_id}/universe", get(get_memory_universe))
        // Memory Visualization
        .route(
            "/api/visualization/{user_id}/stats",
            get(get_visualization_stats),
        )
        .route(
            "/api/visualization/{user_id}/dot",
            get(get_visualization_dot),
        )
        .route("/api/visualization/build", post(build_visualization))
        // Brain State Visualization (cognitive memory tiers with activation levels)
        .route("/api/brain/{user_id}", get(get_brain_state))
        // Context Summary - Session bootstrap with categorized memories
        .route("/api/context_summary", post(context_summary))
        // List memories - Simple GET endpoint
        .route("/api/list/{user_id}", get(list_memories))
        // Apply auth middleware only to protected routes
        .layer(axum::middleware::from_fn(auth::auth_middleware))
        // Apply rate limiting to API routes only (not health/metrics/static)
        .layer(governor_layer)
        .with_state(manager.clone());

    // P0.8: Concurrency limiting for production resilience
    // Limits max concurrent requests to prevent resource exhaustion
    let max_concurrent = server_config.max_concurrent_requests;

    info!(
        "🔄 Concurrency limiting enabled: max_concurrent={}",
        max_concurrent
    );

    // Public routes - NO rate limiting (health checks, metrics, static files)
    // These must always be accessible for monitoring and Kubernetes probes
    let public_routes = Router::new()
        .route(
            "/",
            get(|| async { axum::response::Redirect::permanent("/static/live.html") }),
        )
        .nest_service("/static", ServeDir::new("static"))
        .route("/health", get(health))
        .route("/health/live", get(health_live)) // P0.9: Kubernetes liveness probe
        .route("/health/ready", get(health_ready)) // P0.9: Kubernetes readiness probe
        .route("/metrics", get(metrics_endpoint)) // P1.1: Prometheus metrics
        .route("/api/events", get(memory_events_sse)) // SSE: Real-time memory events for dashboard
        .with_state(manager.clone());

    // Combine public and protected routes
    // Rate limiting is applied only to protected_routes (API endpoints)
    // Public routes (health, metrics, static) are NOT rate limited
    let app = Router::new().merge(public_routes).merge(protected_routes);

    // Conditionally add trace propagation middleware only when telemetry feature is enabled
    #[cfg(feature = "telemetry")]
    let app = app.layer(axum::middleware::from_fn(
        crate::tracing_setup::trace_propagation::propagate_trace_context,
    ));

    // Apply global layers (no rate limiting here - it's already on protected_routes)
    let app = app
        .layer(axum::middleware::from_fn(crate::middleware::track_metrics))
        .layer(ConcurrencyLimitLayer::new(max_concurrent))
        .layer(cors)
        .with_state(manager);

    // Start server using port from config
    let port = server_config.port;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    // Small delay to let any pending log messages flush
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Print server ready message (always visible)
    use std::io::Write;
    let _ = std::io::stderr().flush();
    eprintln!();
    eprintln!("  🚀 Server ready!");
    eprintln!("     HTTP:      http://{}", addr);
    eprintln!("     Health:    http://{}/health", addr);
    eprintln!("     Dashboard: http://{}/static/live.html", addr);
    eprintln!();
    eprintln!("  Press Ctrl+C to stop");
    eprintln!();
    let _ = std::io::stderr().flush();

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Run the server - it will wait until shutdown signal is received
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    info!("🔒 Shutdown signal received, flushing databases...");

    // P0.11: Wrap cleanup process with timeout for production resilience
    let cleanup_future = async {
        // Flush RocksDB with timeout
        let flush_future = async { manager_for_shutdown.flush_all_databases() };

        match tokio::time::timeout(
            std::time::Duration::from_secs(DATABASE_FLUSH_TIMEOUT_SECS),
            flush_future,
        )
        .await
        {
            Ok(Ok(())) => info!("✅ Databases flushed successfully"),
            Ok(Err(e)) => tracing::error!("❌ Failed to flush databases: {}", e),
            Err(_) => tracing::error!(
                "⏱️  Database flush timed out after {}s",
                DATABASE_FLUSH_TIMEOUT_SECS
            ),
        }

        // Save vector indices with timeout
        info!("💾 Persisting vector indices...");
        let save_future = async { manager_for_shutdown.save_all_vector_indices() };

        match tokio::time::timeout(
            std::time::Duration::from_secs(VECTOR_INDEX_SAVE_TIMEOUT_SECS),
            save_future,
        )
        .await
        {
            Ok(Ok(())) => info!("✅ Vector indices saved successfully"),
            Ok(Err(e)) => tracing::error!("❌ Failed to save vector indices: {}", e),
            Err(_) => tracing::error!(
                "⏱️  Vector index save timed out after {}s",
                VECTOR_INDEX_SAVE_TIMEOUT_SECS
            ),
        }

        // P1.6: Shutdown tracing and flush remaining spans (optional, only with telemetry feature)
        #[cfg(feature = "telemetry")]
        tracing_setup::shutdown_tracing();
    };

    // P0.11: Enforce overall cleanup timeout with force-kill fallback
    match tokio::time::timeout(
        std::time::Duration::from_secs(GRACEFUL_SHUTDOWN_TIMEOUT_SECS),
        cleanup_future,
    )
    .await
    {
        Ok(()) => {
            info!("👋 Server shutdown complete");
        }
        Err(_) => {
            tracing::error!(
                "⏱️  Graceful shutdown timed out after {}s, forcing exit",
                GRACEFUL_SHUTDOWN_TIMEOUT_SECS
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

// =============================================================================
// Startup Helper Functions
// =============================================================================

/// Calculate total size of a directory recursively
fn calculate_dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += calculate_dir_size(&path);
            } else if let Ok(metadata) = entry.metadata() {
                total += metadata.len();
            }
        }
    }
    total
}

/// Count user directories in storage path
fn count_user_directories(path: &std::path::Path) -> usize {
    std::fs::read_dir(path)
        .map(|entries| entries.flatten().filter(|e| e.path().is_dir()).count())
        .unwrap_or(0)
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Handle graceful shutdown signals (Ctrl+C and SIGTERM)
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
