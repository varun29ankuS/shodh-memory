//! Multi-User Memory Manager - Core State Management
//!
//! This module contains the central state manager for the shodh-memory server.
//! It handles per-user memory systems, graph memories, audit logs, and all
//! subsidiary stores (todos, reminders, files, etc.).

use anyhow::{Context, Result};
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tracing::info;

use crate::ab_testing;
use crate::backup;
use crate::config::ServerConfig;
use crate::embeddings::{
    are_ner_models_downloaded, download_ner_models, get_ner_models_dir, ner::NerEntityType,
    KeywordExtractor, NerConfig, NeuralNer,
};
use crate::graph_memory::{
    EdgeTier, EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, GraphStats,
    RelationType, RelationshipEdge,
};
use crate::memory::{
    facts::SemanticFactStore, query_parser, Experience, FeedbackStore, FileMemoryStore,
    MemoryConfig, MemoryId, MemoryStats, MemorySystem, ProspectiveStore, SessionStore, TodoStore,
};
use crate::streaming;

use super::types::{AuditEvent, ContextStatus, MemoryEvent};

/// Type alias for context sessions map
pub type ContextSessions = DashMap<String, ContextStatus>;

/// Helper struct for audit log rotation (allows spawn_blocking with minimal clone)
struct MultiUserMemoryManagerRotationHelper {
    audit_db: Arc<rocksdb::DB>,
    audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,
    audit_retention_days: i64,
    audit_max_entries: usize,
}

impl MultiUserMemoryManagerRotationHelper {
    fn rotate_user_audit_logs(&self, user_id: &str) -> Result<usize> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::days(self.audit_retention_days);
        let cutoff_nanos = cutoff_time.timestamp_nanos_opt().unwrap_or(0);

        let mut events: Vec<(Vec<u8>, AuditEvent, i64)> = Vec::new();
        let prefix = format!("{user_id}:");

        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for (key, value) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }

                if let Ok((event, _)) = bincode::serde::decode_from_slice::<AuditEvent, _>(
                    &value,
                    bincode::config::standard(),
                ) {
                    let timestamp_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                    events.push((key.to_vec(), event, timestamp_nanos));
                }
            }
        }

        let initial_count = events.len();
        let mut removed_count = 0;

        events.sort_by(|a, b| b.2.cmp(&a.2));

        let mut keys_to_remove = Vec::new();

        for (idx, (key, _event, timestamp_nanos)) in events.iter().enumerate() {
            let should_remove = *timestamp_nanos < cutoff_nanos || idx >= self.audit_max_entries;

            if should_remove {
                keys_to_remove.push(key.clone());
                removed_count += 1;
            }
        }

        if !keys_to_remove.is_empty() {
            let mut batch = rocksdb::WriteBatch::default();
            for key in &keys_to_remove {
                batch.delete(key);
            }
            self.audit_db
                .write(batch)
                .map_err(|e| anyhow::anyhow!("Failed to write rotation batch: {e}"))?;
        }

        if removed_count > 0 {
            if let Some(log) = self.audit_logs.get(user_id) {
                let mut log_guard = log.write();

                log_guard.retain(|event| {
                    let event_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                    event_nanos >= cutoff_nanos
                        && initial_count - removed_count <= self.audit_max_entries
                });

                while log_guard.len() > self.audit_max_entries {
                    log_guard.pop_front();
                }
            }
        }

        Ok(removed_count)
    }
}

/// Multi-user memory manager - central state for the server
pub struct MultiUserMemoryManager {
    /// Per-user memory systems with LRU eviction
    pub user_memories: moka::sync::Cache<String, Arc<parking_lot::RwLock<MemorySystem>>>,

    /// Per-user audit logs (in-memory cache)
    pub audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,

    /// Persistent audit log storage
    pub audit_db: Arc<rocksdb::DB>,

    /// Base storage path
    pub base_path: std::path::PathBuf,

    /// Default config
    pub default_config: MemoryConfig,

    /// Counter for audit log rotation checks
    pub audit_log_counter: Arc<std::sync::atomic::AtomicUsize>,

    /// Per-user graph memory systems
    pub graph_memories: moka::sync::Cache<String, Arc<parking_lot::RwLock<GraphMemory>>>,

    /// Neural NER for automatic entity extraction
    pub neural_ner: Arc<NeuralNer>,

    /// Statistical keyword extraction for graph population
    pub keyword_extractor: Arc<KeywordExtractor>,

    /// User eviction counter for metrics
    pub user_evictions: Arc<std::sync::atomic::AtomicUsize>,

    /// Server configuration
    pub server_config: ServerConfig,

    /// SSE event broadcaster for real-time dashboard updates
    pub event_broadcaster: tokio::sync::broadcast::Sender<MemoryEvent>,

    /// Streaming memory extractor for implicit learning
    pub streaming_extractor: Arc<streaming::StreamingMemoryExtractor>,

    /// Prospective memory store for reminders/intentions
    pub prospective_store: Arc<ProspectiveStore>,

    /// GTD-style todo store
    pub todo_store: Arc<TodoStore>,

    /// File memory store for codebase integration
    pub file_store: Arc<FileMemoryStore>,

    /// Implicit feedback store for memory reinforcement
    pub feedback_store: Arc<parking_lot::RwLock<FeedbackStore>>,

    /// Backup engine for automated and manual backups
    pub backup_engine: Arc<backup::ShodhBackupEngine>,

    /// Context status from Claude Code sessions
    pub context_sessions: Arc<ContextSessions>,

    /// SSE broadcaster for context status updates
    pub context_broadcaster: tokio::sync::broadcast::Sender<ContextStatus>,

    /// A/B testing manager for relevance scoring experiments
    pub ab_test_manager: Arc<ab_testing::ABTestManager>,

    /// Semantic fact store for durable knowledge
    pub fact_store: Arc<SemanticFactStore>,

    /// Session tracking store
    pub session_store: Arc<SessionStore>,
}

impl MultiUserMemoryManager {
    pub fn new(base_path: std::path::PathBuf, server_config: ServerConfig) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;

        let audit_path = base_path.join("audit_logs");
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let audit_db = Arc::new(rocksdb::DB::open(&opts, audit_path)?);

        let (event_broadcaster, _) = tokio::sync::broadcast::channel(1024);

        let ner_dir = get_ner_models_dir();
        tracing::debug!("Checking for NER models at {:?}", ner_dir);
        let neural_ner = if are_ner_models_downloaded() {
            tracing::debug!("NER models found, using existing files");
            let config = NerConfig {
                model_path: ner_dir.join("model.onnx"),
                tokenizer_path: ner_dir.join("tokenizer.json"),
                max_length: 128,
                confidence_threshold: 0.5,
            };
            match NeuralNer::new(config) {
                Ok(ner) => {
                    info!("Neural NER initialized (TinyBERT model at {:?})", ner_dir);
                    Arc::new(ner)
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize neural NER: {}. Using fallback.", e);
                    Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                }
            }
        } else {
            tracing::debug!("NER models not found at {:?}, will download", ner_dir);
            info!("Downloading NER models (TinyBERT-NER, ~15MB)...");
            match download_ner_models(Some(std::sync::Arc::new(|downloaded, total| {
                if total > 0 {
                    let percent = (downloaded as f64 / total as f64 * 100.0) as u32;
                    if percent % 20 == 0 {
                        tracing::info!("NER model download: {}%", percent);
                    }
                }
            }))) {
                Ok(ner_dir) => {
                    info!("NER models downloaded to {:?}", ner_dir);
                    let config = NerConfig {
                        model_path: ner_dir.join("model.onnx"),
                        tokenizer_path: ner_dir.join("tokenizer.json"),
                        max_length: 128,
                        confidence_threshold: 0.5,
                    };
                    match NeuralNer::new(config) {
                        Ok(ner) => {
                            info!("Neural NER initialized after download");
                            Arc::new(ner)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to initialize downloaded NER: {}. Using fallback.",
                                e
                            );
                            Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to download NER models: {}. Using rule-based fallback.",
                        e
                    );
                    Arc::new(NeuralNer::new_fallback(NerConfig::default()))
                }
            }
        };

        let user_evictions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let evictions_clone = user_evictions.clone();
        let max_cache = server_config.max_users_in_memory;

        let user_memories = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .eviction_listener(move |key: Arc<String>, _value, cause| {
                if cause == moka::notification::RemovalCause::Size {
                    evictions_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    info!(
                        "Evicted user '{}' from memory cache (LRU, cache_size={})",
                        key, max_cache
                    );
                }
            })
            .build();

        let graph_memories = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .eviction_listener(move |key: Arc<String>, _value, _cause| {
                info!("Evicted graph for user '{}' from memory cache (LRU)", key);
            })
            .build();

        let prospective_store = Arc::new(ProspectiveStore::new(&base_path)?);
        info!("Prospective memory store initialized");

        let todo_store = Arc::new(TodoStore::new(&base_path)?);
        info!("Todo store initialized");

        let file_store = Arc::new(FileMemoryStore::new(&base_path)?);
        info!("File memory store initialized");

        let feedback_store = Arc::new(parking_lot::RwLock::new(
            FeedbackStore::with_persistence(base_path.join("feedback")).unwrap_or_else(|e| {
                tracing::warn!("Failed to load feedback store: {}, using in-memory", e);
                FeedbackStore::new()
            }),
        ));
        info!("Feedback store initialized");

        let streaming_extractor = Arc::new(streaming::StreamingMemoryExtractor::new(
            neural_ner.clone(),
            feedback_store.clone(),
        ));
        info!("Streaming memory extractor initialized");

        let keyword_extractor = Arc::new(KeywordExtractor::new());
        info!("Keyword extractor initialized (YAKE)");

        let backup_path = base_path.join("backups");
        let backup_engine = Arc::new(backup::ShodhBackupEngine::new(backup_path)?);
        if server_config.backup_enabled {
            info!(
                "Backup engine initialized (interval: {}h, keep: {})",
                server_config.backup_interval_secs / 3600,
                server_config.backup_max_count
            );
        } else {
            info!("Backup engine initialized (auto-backup disabled)");
        }

        let facts_path = base_path.join("semantic_facts");
        let mut facts_opts = rocksdb::Options::default();
        facts_opts.create_if_missing(true);
        facts_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let facts_db = Arc::new(rocksdb::DB::open(&facts_opts, &facts_path)?);
        let fact_store = Arc::new(SemanticFactStore::new(facts_db));
        info!("Semantic fact store initialized");

        let manager = Self {
            user_memories,
            audit_logs: Arc::new(DashMap::new()),
            audit_db,
            base_path,
            default_config: MemoryConfig::default(),
            audit_log_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            graph_memories,
            neural_ner,
            keyword_extractor,
            user_evictions,
            server_config,
            event_broadcaster,
            streaming_extractor,
            prospective_store,
            todo_store,
            file_store,
            feedback_store,
            backup_engine,
            context_sessions: Arc::new(DashMap::new()),
            context_broadcaster: {
                let (tx, _) = tokio::sync::broadcast::channel(16);
                tx
            },
            ab_test_manager: Arc::new(ab_testing::ABTestManager::new()),
            fact_store,
            session_store: Arc::new(SessionStore::new()),
        };

        info!("Running initial audit log rotation...");
        if let Err(e) = manager.rotate_all_audit_logs() {
            tracing::warn!("Failed to rotate audit logs on startup: {}", e);
        }

        Ok(manager)
    }

    /// Log audit event (non-blocking with background persistence)
    pub fn log_event(&self, user_id: &str, event_type: &str, memory_id: &str, details: &str) {
        let event = AuditEvent {
            timestamp: chrono::Utc::now(),
            event_type: event_type.to_string(),
            memory_id: memory_id.to_string(),
            details: details.to_string(),
        };

        let key = format!(
            "{}:{}",
            user_id,
            event.timestamp.timestamp_nanos_opt().unwrap_or(0)
        );
        if let Ok(serialized) = bincode::serde::encode_to_vec(&event, bincode::config::standard()) {
            let db = self.audit_db.clone();
            let key_bytes = key.into_bytes();

            tokio::task::spawn_blocking(move || {
                if let Err(e) = db.put(&key_bytes, &serialized) {
                    tracing::error!("Failed to persist audit log: {}", e);
                }
            });
        }

        let max_entries = self.server_config.audit_max_entries_per_user;
        if let Some(log) = self.audit_logs.get(user_id) {
            let mut entries = log.write();
            entries.push_back(event);
            while entries.len() > max_entries {
                entries.pop_front();
            }
        } else {
            let mut deque = VecDeque::new();
            deque.push_back(event);
            let log = Arc::new(parking_lot::RwLock::new(deque));
            self.audit_logs.insert(user_id.to_string(), log);
        }

        let count = self
            .audit_log_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if count % self.server_config.audit_rotation_check_interval == 0 && count > 0 {
            let audit_db = self.audit_db.clone();
            let audit_logs = self.audit_logs.clone();
            let user_id_clone = user_id.to_string();

            let audit_retention_days = self.server_config.audit_retention_days as i64;
            let audit_max_entries = self.server_config.audit_max_entries_per_user;

            tokio::task::spawn_blocking(move || {
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
    pub fn emit_event(&self, event: MemoryEvent) {
        let _ = self.event_broadcaster.send(event);
    }

    /// Subscribe to SSE events
    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<MemoryEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Get audit history for user
    pub fn get_history(&self, user_id: &str, memory_id: Option<&str>) -> Vec<AuditEvent> {
        if let Some(log) = self.audit_logs.get(user_id) {
            let events = log.read();
            if !events.is_empty() {
                return if let Some(mid) = memory_id {
                    events
                        .iter()
                        .filter(|e| e.memory_id == mid)
                        .cloned()
                        .collect()
                } else {
                    events.iter().cloned().collect()
                };
            }
        }

        let mut events = Vec::new();
        let prefix = format!("{user_id}:");

        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for (key, value) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }

                if let Ok((event, _)) = bincode::serde::decode_from_slice::<AuditEvent, _>(
                    &value,
                    bincode::config::standard(),
                ) {
                    events.push(event);
                }
            }
        }

        if !events.is_empty() {
            let log = Arc::new(parking_lot::RwLock::new(VecDeque::from(events.clone())));
            self.audit_logs.insert(user_id.to_string(), log);
        }

        if let Some(mid) = memory_id {
            events.into_iter().filter(|e| e.memory_id == mid).collect()
        } else {
            events
        }
    }

    /// Get or create memory system for a user
    pub fn get_user_memory(&self, user_id: &str) -> Result<Arc<parking_lot::RwLock<MemorySystem>>> {
        if let Some(memory) = self.user_memories.get(user_id) {
            return Ok(memory);
        }

        let user_path = self.base_path.join(user_id);
        let config = MemoryConfig {
            storage_path: user_path,
            ..self.default_config.clone()
        };

        let mut memory_system = MemorySystem::new(config).with_context(|| {
            format!("Failed to initialize memory system for user '{}'", user_id)
        })?;
        // Wire up GraphMemory for Layer 2 (spreading activation) and Layer 5 (Hebbian learning)
        let graph = self.get_user_graph(user_id)?;
        memory_system.set_graph_memory(graph);

        let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));

        self.user_memories
            .insert(user_id.to_string(), memory_arc.clone());

        info!("Created memory system for user: {}", user_id);

        Ok(memory_arc)
    }

    /// Delete user data (GDPR compliance)
    pub fn forget_user(&self, user_id: &str) -> Result<()> {
        self.user_memories.invalidate(user_id);
        self.graph_memories.invalidate(user_id);

        self.user_memories.run_pending_tasks();
        self.graph_memories.run_pending_tasks();

        #[cfg(target_os = "windows")]
        {
            std::thread::sleep(std::time::Duration::from_millis(200));
            self.user_memories.run_pending_tasks();
            self.graph_memories.run_pending_tasks();
        }

        let user_path = self.base_path.join(user_id);
        if user_path.exists() {
            let mut attempts = 0;
            let max_attempts = 10;
            while attempts < max_attempts {
                match std::fs::remove_dir_all(&user_path) {
                    Ok(_) => break,
                    Err(e) if attempts < max_attempts - 1 => {
                        let delay = 100 * (1 << attempts.min(4));
                        tracing::debug!(
                            "Delete retry {} for {} (waiting {}ms): {}",
                            attempts + 1,
                            user_id,
                            delay,
                            e
                        );
                        std::thread::sleep(std::time::Duration::from_millis(delay));
                        attempts += 1;
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Failed to delete user data after {} retries: {}",
                            max_attempts,
                            e
                        ))
                    }
                }
            }
        }

        info!("Deleted all data for user: {}", user_id);
        Ok(())
    }

    /// Get statistics for a user
    pub fn get_stats(&self, user_id: &str) -> Result<MemoryStats> {
        let memory = self.get_user_memory(user_id)?;
        let memory_guard = memory.read();
        let mut stats = memory_guard.stats();

        if let Ok(graph) = self.get_user_graph(user_id) {
            let graph_guard = graph.read();
            if let Ok(graph_stats) = graph_guard.get_stats() {
                stats.graph_nodes = graph_stats.entity_count;
                stats.graph_edges = graph_stats.relationship_count;
            }
        }

        Ok(stats)
    }

    /// List all users
    pub fn list_users(&self) -> Vec<String> {
        let mut users = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.base_path) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        if let Some(name) = entry.file_name().to_str() {
                            // Filter out system directories
                            if name != "audit_logs"
                                && name != "backups"
                                && name != "feedback"
                                && name != "semantic_facts"
                                && name != "files"
                                && name != "prospective"
                                && name != "todos"
                            {
                                users.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }
        users.sort();
        users
    }

    /// Get audit logs for a user
    pub fn get_audit_logs(&self, user_id: &str, limit: usize) -> Vec<AuditEvent> {
        let mut events: Vec<AuditEvent> = Vec::new();
        let prefix = format!("{user_id}:");
        let iter = self.audit_db.prefix_iterator(prefix.as_bytes());
        for (key, value) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Ok((event, _)) = bincode::serde::decode_from_slice::<AuditEvent, _>(
                    &value,
                    bincode::config::standard(),
                ) {
                    events.push(event);
                }
            }
        }
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);
        events
    }

    /// Flush all RocksDB databases
    pub fn flush_all_databases(&self) -> Result<()> {
        info!("Flushing all databases to disk...");

        self.audit_db
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush audit database: {e}"))?;
        info!("  Audit database flushed");

        if let Err(e) = self.todo_store.flush() {
            tracing::warn!("  Failed to flush todo store: {}", e);
        } else {
            info!("  Todo store flushed");
        }

        if let Err(e) = self.file_store.flush() {
            tracing::warn!("  Failed to flush file store: {}", e);
        } else {
            info!("  File memory store flushed");
        }

        if let Err(e) = self.prospective_store.flush() {
            tracing::warn!("  Failed to flush prospective store: {}", e);
        } else {
            info!("  Prospective store flushed");
        }

        if let Err(e) = self.feedback_store.write().flush() {
            tracing::warn!("  Failed to flush feedback store: {}", e);
        } else {
            info!("  Feedback store flushed");
        }

        let user_entries: Vec<(String, Arc<parking_lot::RwLock<MemorySystem>>)> = self
            .user_memories
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        let mut flushed = 0;
        for (user_id, memory_system) in user_entries {
            if let Some(guard) = memory_system.try_read() {
                if let Err(e) = guard.flush_storage() {
                    tracing::warn!("  Failed to flush database for user {}: {}", user_id, e);
                } else {
                    flushed += 1;
                }
            } else {
                tracing::warn!("  Could not acquire lock for user: {}", user_id);
            }
        }

        info!(
            "All databases flushed: audit, todos, files, prospective, feedback, {} user memories",
            flushed
        );

        Ok(())
    }

    /// Save all vector indices to disk
    pub fn save_all_vector_indices(&self) -> Result<()> {
        info!("Saving vector indices to disk...");

        let user_entries: Vec<(String, Arc<parking_lot::RwLock<MemorySystem>>)> = self
            .user_memories
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

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

        info!("Saved {} vector indices", saved);
        Ok(())
    }

    /// Rotate audit logs for all users
    fn rotate_all_audit_logs(&self) -> Result<()> {
        let mut total_removed = 0;

        let mut user_ids = std::collections::HashSet::new();
        let iter = self.audit_db.iterator(rocksdb::IteratorMode::Start);

        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if let Some(user_id) = key_str.split(':').next() {
                    user_ids.insert(user_id.to_string());
                }
            }
        }

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
                "Audit log rotation complete: removed {} total entries",
                total_removed
            );
        }

        Ok(())
    }

    /// Get neural NER for entity extraction
    pub fn get_neural_ner(&self) -> Arc<NeuralNer> {
        self.neural_ner.clone()
    }

    /// Get keyword extractor for statistical term extraction
    pub fn get_keyword_extractor(&self) -> Arc<KeywordExtractor> {
        self.keyword_extractor.clone()
    }

    /// Get or create graph memory for a user
    pub fn get_user_graph(&self, user_id: &str) -> Result<Arc<parking_lot::RwLock<GraphMemory>>> {
        if let Some(graph) = self.graph_memories.get(user_id) {
            return Ok(graph);
        }

        let graph_path = self.base_path.join(user_id).join("graph");
        let graph_memory = GraphMemory::new(&graph_path)?;
        let graph_arc = Arc::new(parking_lot::RwLock::new(graph_memory));

        self.graph_memories
            .insert(user_id.to_string(), graph_arc.clone());

        info!("Created graph memory for user: {}", user_id);

        Ok(graph_arc)
    }

    /// Get graph statistics for a user
    pub fn get_user_graph_stats(&self, user_id: &str) -> Result<GraphStats> {
        let graph = self.get_user_graph(user_id)?;
        let graph_guard = graph.read();
        graph_guard.get_stats()
    }

    /// Run maintenance on all cached user memories
    pub fn run_maintenance_all_users(&self) -> usize {
        let decay_factor = self.server_config.activation_decay_factor;
        let mut total_processed = 0;

        let user_ids: Vec<String> = self
            .user_memories
            .iter()
            .map(|(id, _)| id.to_string())
            .collect();

        let user_count = user_ids.len();
        let mut edges_decayed = 0;
        let mut edges_strengthened = 0;

        for user_id in user_ids {
            let maintenance_result = if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                let memory = memory_lock.read();
                match memory.run_maintenance(decay_factor) {
                    Ok(result) => {
                        total_processed += result.decayed_count;
                        Some(result)
                    }
                    Err(e) => {
                        tracing::warn!("Maintenance failed for user {}: {}", user_id, e);
                        None
                    }
                }
            } else {
                None
            };

            if let Some(ref result) = maintenance_result {
                if !result.edge_boosts.is_empty() {
                    if let Ok(graph) = self.get_user_graph(&user_id) {
                        let graph_guard = graph.write();
                        match graph_guard.strengthen_memory_edges(&result.edge_boosts) {
                            Ok(count) => {
                                edges_strengthened += count;
                            }
                            Err(e) => {
                                tracing::debug!(
                                    "Edge boost application failed for user {}: {}",
                                    user_id,
                                    e
                                );
                            }
                        }
                    }
                }
            }

            if let Ok(graph) = self.get_user_graph(&user_id) {
                let graph_guard = graph.write();
                match graph_guard.apply_decay() {
                    Ok(pruned) => {
                        edges_decayed += pruned;
                    }
                    Err(e) => {
                        tracing::debug!("Graph decay failed for user {}: {}", user_id, e);
                    }
                }
            }
        }

        tracing::info!(
            "Maintenance complete: {} memories processed, {} edges strengthened, {} weak edges pruned across {} users",
            total_processed,
            edges_strengthened,
            edges_decayed,
            user_count
        );

        total_processed
    }

    /// Get the streaming extractor
    pub fn streaming_extractor(&self) -> &Arc<streaming::StreamingMemoryExtractor> {
        &self.streaming_extractor
    }

    /// Get the backup engine
    pub fn backup_engine(&self) -> &Arc<backup::ShodhBackupEngine> {
        &self.backup_engine
    }

    /// Get the A/B test manager
    pub fn ab_test_manager(&self) -> &Arc<ab_testing::ABTestManager> {
        &self.ab_test_manager
    }

    /// Get the todo store
    pub fn todo_store(&self) -> &Arc<TodoStore> {
        &self.todo_store
    }

    /// Get the prospective store
    pub fn prospective_store(&self) -> &Arc<ProspectiveStore> {
        &self.prospective_store
    }

    /// Get the file store
    pub fn file_store(&self) -> &Arc<FileMemoryStore> {
        &self.file_store
    }

    /// Get the feedback store
    pub fn feedback_store(&self) -> &Arc<parking_lot::RwLock<FeedbackStore>> {
        &self.feedback_store
    }

    /// Get the fact store
    pub fn fact_store(&self) -> &Arc<SemanticFactStore> {
        &self.fact_store
    }

    /// Get the session store
    pub fn session_store(&self) -> &Arc<SessionStore> {
        &self.session_store
    }

    /// Get context sessions
    pub fn context_sessions(&self) -> &Arc<ContextSessions> {
        &self.context_sessions
    }

    /// Subscribe to context status updates
    pub fn subscribe_context(&self) -> tokio::sync::broadcast::Receiver<ContextStatus> {
        self.context_broadcaster.subscribe()
    }

    /// Broadcast context status update
    pub fn broadcast_context(&self, status: ContextStatus) {
        let _ = self.context_broadcaster.send(status);
    }

    /// Get server config
    pub fn server_config(&self) -> &ServerConfig {
        &self.server_config
    }

    /// Get base path
    pub fn base_path(&self) -> &std::path::Path {
        &self.base_path
    }

    /// Get user evictions count
    pub fn user_evictions(&self) -> usize {
        self.user_evictions
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get users in cache count
    pub fn users_in_cache(&self) -> usize {
        self.user_memories.entry_count() as usize
    }

    /// Run backups for all active users
    pub fn run_backup_all_users(&self, max_backups: usize) -> usize {
        let mut backed_up = 0;

        let users_path = &self.base_path;
        if let Ok(entries) = std::fs::read_dir(users_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('.') || name == "audit_logs" || name == "backups" {
                    continue;
                }

                let db_path = path.join("memory.db");
                if !db_path.exists() {
                    continue;
                }

                if let Ok(memory_lock) = self.get_user_memory(name) {
                    let memory = memory_lock.read();
                    let db = memory.get_db();
                    match self.backup_engine.create_backup(&db, name) {
                        Ok(metadata) => {
                            tracing::info!(
                                user_id = name,
                                backup_id = metadata.backup_id,
                                size_mb = metadata.size_bytes / 1024 / 1024,
                                "Backup created successfully"
                            );
                            backed_up += 1;

                            if let Err(e) = self.backup_engine.purge_old_backups(name, max_backups)
                            {
                                tracing::warn!(
                                    user_id = name,
                                    error = %e,
                                    "Failed to purge old backups"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                user_id = name,
                                error = %e,
                                "Failed to create backup"
                            );
                        }
                    }
                }
            }
        }

        backed_up
    }

    /// Process an experience and extract entities/relationships into the graph
    ///
    /// SHO-102: Improved graph building with:
    /// - Neural NER entities
    /// - Tags as Technology/Concept entities
    /// - All-caps terms (API, TUI, NER, etc.)
    /// - Issue IDs (SHO-XX pattern)
    /// - Semantic similarity edges between memories
    pub fn process_experience_into_graph(
        &self,
        user_id: &str,
        experience: &Experience,
        memory_id: &MemoryId,
    ) -> Result<()> {
        let graph = self.get_user_graph(user_id)?;
        let graph_guard = graph.write();

        // Use pre-extracted entities from experience.entities if available
        // Only run NER if no entities were pre-extracted
        let extracted_entities = if !experience.entities.is_empty() {
            tracing::debug!(
                "Using {} pre-extracted entities: {:?}",
                experience.entities.len(),
                experience.entities
            );
            experience
                .entities
                .iter()
                .map(|name| crate::embeddings::ner::NerEntity {
                    text: name.clone(),
                    entity_type: NerEntityType::Misc,
                    confidence: 0.8,
                    start: 0,
                    end: name.len(),
                })
                .collect()
        } else {
            match self.neural_ner.extract(&experience.content) {
                Ok(entities) => {
                    tracing::debug!(
                        "NER extracted {} entities: {:?}",
                        entities.len(),
                        entities.iter().map(|e| e.text.as_str()).collect::<Vec<_>>()
                    );
                    entities
                }
                Err(e) => {
                    tracing::debug!("NER extraction failed: {}. Continuing without entities.", e);
                    Vec::new()
                }
            }
        };

        // Filter out garbage/noise entities from neural NER
        let stop_words: std::collections::HashSet<&str> = [
            "the", "and", "for", "that", "this", "with", "from", "have", "been", "are", "was",
            "were", "will", "would", "could", "should", "may", "might",
        ]
        .iter()
        .cloned()
        .collect();

        let filtered_entities: Vec<_> = extracted_entities
            .into_iter()
            .filter(|e| {
                let name = e.text.trim();
                if name.len() < 3 {
                    return false;
                }
                // Relaxed: Allow entities without uppercase if NER confidence is high
                // This enables capturing "sarah", "lgbtq" etc. when NER is confident
                if !name.chars().any(|c| c.is_uppercase()) && e.confidence < 0.7 {
                    return false;
                }
                if stop_words.contains(name.to_lowercase().as_str()) {
                    return false;
                }
                if name.len() < 5 && e.confidence < 0.8 {
                    return false;
                }
                true
            })
            .collect();

        tracing::debug!(
            "After filtering: {} entities: {:?}",
            filtered_entities.len(),
            filtered_entities
                .iter()
                .map(|e| e.text.as_str())
                .collect::<Vec<_>>()
        );

        let mut entity_uuids = Vec::new();

        // Add entities to the graph
        for ner_entity in filtered_entities {
            let label = match ner_entity.entity_type {
                NerEntityType::Person => EntityLabel::Person,
                NerEntityType::Organization => EntityLabel::Organization,
                NerEntityType::Location => EntityLabel::Location,
                NerEntityType::Misc => EntityLabel::Other("MISC".to_string()),
            };

            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(),
                name: ner_entity.text.clone(),
                labels: vec![label],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
                salience: ner_entity.confidence,
                is_proper_noun: true,
            };

            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((ner_entity.text, uuid)),
                Err(e) => tracing::debug!("Failed to add entity {}: {}", ner_entity.text, e),
            }
        }

        // Add tags as entities
        for tag in &experience.tags {
            let tag_name = tag.trim();
            if tag_name.len() >= 2 && !stop_words.contains(tag_name.to_lowercase().as_str()) {
                let entity = EntityNode {
                    uuid: uuid::Uuid::new_v4(),
                    name: tag_name.to_string(),
                    labels: vec![EntityLabel::Technology],
                    created_at: chrono::Utc::now(),
                    last_seen_at: chrono::Utc::now(),
                    mention_count: 1,
                    summary: String::new(),
                    attributes: HashMap::new(),
                    name_embedding: None,
                    salience: 0.6,
                    is_proper_noun: false,
                };

                match graph_guard.add_entity(entity) {
                    Ok(uuid) => entity_uuids.push((tag_name.to_string(), uuid)),
                    Err(e) => tracing::debug!("Failed to add tag entity {}: {}", tag_name, e),
                }
            }
        }

        // Extract all-caps terms (API, TUI, NER, REST, etc.)
        let allcaps_regex = regex::Regex::new(r"\b[A-Z]{2,}[A-Z0-9]*\b").unwrap();
        for cap in allcaps_regex.find_iter(&experience.content) {
            let term = cap.as_str();
            if entity_uuids
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case(term))
            {
                continue;
            }
            if stop_words.contains(term.to_lowercase().as_str()) {
                continue;
            }

            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(),
                name: term.to_string(),
                labels: vec![EntityLabel::Technology],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
                salience: 0.5,
                is_proper_noun: true,
            };

            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((term.to_string(), uuid)),
                Err(e) => tracing::debug!("Failed to add allcaps entity {}: {}", term, e),
            }
        }

        // Extract issue IDs (SHO-XX, JIRA-123, etc.)
        let issue_regex = regex::Regex::new(r"\b([A-Z]{2,10}-\d+)\b").unwrap();
        for issue in issue_regex.find_iter(&experience.content) {
            let issue_id = issue.as_str();
            if entity_uuids.iter().any(|(name, _)| name == issue_id) {
                continue;
            }

            let entity = EntityNode {
                uuid: uuid::Uuid::new_v4(),
                name: issue_id.to_string(),
                labels: vec![EntityLabel::Other("Issue".to_string())],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                mention_count: 1,
                summary: String::new(),
                attributes: HashMap::new(),
                name_embedding: None,
                salience: 0.7,
                is_proper_noun: true,
            };

            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((issue_id.to_string(), uuid)),
                Err(e) => tracing::debug!("Failed to add issue entity {}: {}", issue_id, e),
            }
        }

        // Extract verbs from content for multi-hop reasoning
        // Verbs like "painted", "bought", "visited" connect entities across memories
        let analysis = query_parser::analyze_query(&experience.content);
        for verb in &analysis.relational_context {
            let verb_text = verb.text.as_str();
            let verb_stem = verb.stem.as_str();

            // Skip if already added (as noun or other entity)
            if entity_uuids
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case(verb_text))
            {
                continue;
            }
            if stop_words.contains(verb_text.to_lowercase().as_str()) {
                continue;
            }
            if verb_text.len() < 3 {
                continue;
            }

            // Add verb as entity (both text form and stem for matching)
            for name in [verb_text, verb_stem] {
                if name.len() < 3 {
                    continue;
                }
                if entity_uuids
                    .iter()
                    .any(|(n, _)| n.eq_ignore_ascii_case(name))
                {
                    continue;
                }

                let entity = EntityNode {
                    uuid: uuid::Uuid::new_v4(),
                    name: name.to_string(),
                    labels: vec![EntityLabel::Other("Verb".to_string())],
                    created_at: chrono::Utc::now(),
                    last_seen_at: chrono::Utc::now(),
                    mention_count: 1,
                    summary: String::new(),
                    attributes: HashMap::new(),
                    name_embedding: None,
                    salience: 0.4, // Lower salience for verbs
                    is_proper_noun: false,
                };

                match graph_guard.add_entity(entity) {
                    Ok(uuid) => entity_uuids.push((name.to_string(), uuid)),
                    Err(e) => tracing::debug!("Failed to add verb entity {}: {}", name, e),
                }
            }
        }

        // Create an episodic node for this experience
        tracing::debug!(
            "Creating episode for memory {} with {} entities: {:?}",
            &memory_id.0.to_string()[..8],
            entity_uuids.len(),
            entity_uuids
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
        );

        let episode = EpisodicNode {
            uuid: memory_id.0,
            name: format!("Memory {}", &memory_id.0.to_string()[..8]),
            content: experience.content.clone(),
            valid_at: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            entity_refs: entity_uuids.iter().map(|(_, uuid)| *uuid).collect(),
            source: EpisodeSource::Message,
            metadata: experience.metadata.clone(),
        };

        match graph_guard.add_episode(episode) {
            Ok(uuid) => {
                tracing::debug!(
                    "Episode {} added with {} entity refs",
                    &uuid.to_string()[..8],
                    entity_uuids.len()
                );
            }
            Err(e) => {
                tracing::warn!("Failed to add episode: {}", e);
            }
        }

        // Create relationships between co-occurring entities
        for i in 0..entity_uuids.len() {
            for j in (i + 1)..entity_uuids.len() {
                let edge = RelationshipEdge {
                    uuid: uuid::Uuid::new_v4(),
                    from_entity: entity_uuids[i].1,
                    to_entity: entity_uuids[j].1,
                    relation_type: RelationType::RelatedTo,
                    strength: EdgeTier::L1Working.initial_weight(),
                    created_at: chrono::Utc::now(),
                    valid_at: chrono::Utc::now(),
                    invalidated_at: None,
                    source_episode_id: Some(memory_id.0),
                    context: experience.content.clone(),
                    last_activated: chrono::Utc::now(),
                    activation_count: 1,
                    potentiated: false,
                    tier: EdgeTier::L1Working,
                };

                if let Err(e) = graph_guard.add_relationship(edge) {
                    tracing::debug!("Failed to add relationship: {}", e);
                }
            }
        }

        Ok(())
    }
}
