//! Multi-User Memory Manager - Core State Management
//!
//! This module contains the central state manager for the shodh-memory server.
//! It handles per-user memory systems, graph memories, audit logs, and all
//! subsidiary stores (todos, reminders, files, etc.).

use anyhow::{Context, Result};
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use tracing::info;

/// Static regex for extracting all-caps terms (API, TUI, NER, REST, etc.)
fn allcaps_regex() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\b[A-Z]{2,}[A-Z0-9]*\b").unwrap())
}

/// Static regex for extracting issue IDs (SHO-XX, JIRA-123, etc.)
fn issue_regex() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\b([A-Z]{2,10}-\d+)\b").unwrap())
}

use crate::ab_testing;
use crate::backup;
use crate::config::ServerConfig;
use crate::embeddings::{
    are_ner_models_downloaded, download_ner_models, get_ner_models_dir, ner::NerEntityType,
    KeywordExtractor, NerConfig, NeuralNer,
};
use crate::graph_memory::{
    EdgeTier, EntityLabel, EntityNode, EpisodeSource, EpisodicNode, GraphMemory, GraphStats,
    LtpStatus, RelationType, RelationshipEdge,
};
use crate::memory::{
    query_parser, Experience, FeedbackStore, FileMemoryStore, MemoryConfig, MemoryId, MemoryStats,
    MemorySystem, ProspectiveStore, SessionStore, TodoStore,
};
use crate::relevance::RelevanceEngine;
use crate::streaming;

use super::types::{AuditEvent, ContextStatus, MemoryEvent};

/// Type alias for context sessions map
pub type ContextSessions = DashMap<String, ContextStatus>;

/// Helper struct for audit log rotation (allows spawn_blocking with minimal clone)
struct MultiUserMemoryManagerRotationHelper {
    shared_db: Arc<rocksdb::DB>,
    audit_logs: Arc<DashMap<String, Arc<parking_lot::RwLock<VecDeque<AuditEvent>>>>>,
    audit_retention_days: i64,
    audit_max_entries: usize,
}

const CF_AUDIT: &str = "audit";

impl MultiUserMemoryManagerRotationHelper {
    fn audit_cf(&self) -> &rocksdb::ColumnFamily {
        self.shared_db
            .cf_handle(CF_AUDIT)
            .expect("audit CF must exist")
    }

    /// Rotate audit logs for a user - delete old entries and enforce max count.
    ///
    /// Keys are `{user_id}:{timestamp_nanos:020}` so RocksDB returns them in
    /// ascending timestamp order. Two strategies depending on scale:
    /// - ≤100K keys: collect all, compute excess, batch delete
    /// - >100K keys: streaming 2-pass (count, then delete) to avoid OOM
    fn rotate_user_audit_logs(&self, user_id: &str) -> Result<usize> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::days(self.audit_retention_days);
        let cutoff_nanos = cutoff_time.timestamp_nanos_opt().unwrap_or_else(|| {
            tracing::warn!("audit cutoff timestamp outside i64 nanos range, using 0");
            0
        });
        let prefix = format!("{user_id}:");
        let audit = self.audit_cf();

        // Pass 1: count total entries to determine excess
        let mut total_count = 0usize;
        let iter = self.shared_db.prefix_iterator_cf(audit, prefix.as_bytes());
        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if !key_str.starts_with(&prefix) {
                    break;
                }
                total_count += 1;
            }
        }

        if total_count == 0 {
            return Ok(0);
        }

        let excess_count = total_count.saturating_sub(self.audit_max_entries);

        // Pass 2: stream through keys, deleting those that are too old or excess.
        // Flush WriteBatch every 10K deletes to bound memory.
        const BATCH_FLUSH_SIZE: usize = 10_000;
        let mut batch = rocksdb::WriteBatch::default();
        let mut removed_count = 0usize;
        let mut position = 0usize;

        let iter = self.shared_db.prefix_iterator_cf(audit, prefix.as_bytes());
        for (key, _) in iter.flatten() {
            let key_str = match std::str::from_utf8(&key) {
                Ok(s) => s,
                Err(_) => {
                    position += 1;
                    continue;
                }
            };
            if !key_str.starts_with(&prefix) {
                break;
            }

            let ts = key_str
                .strip_prefix(&prefix)
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0); // Malformed keys sort first → get deleted

            if ts < cutoff_nanos || position < excess_count {
                batch.delete_cf(audit, &key);
                removed_count += 1;

                if removed_count % BATCH_FLUSH_SIZE == 0 {
                    self.shared_db
                        .write(std::mem::take(&mut batch))
                        .map_err(|e| anyhow::anyhow!("Failed to write rotation batch: {e}"))?;
                    batch = rocksdb::WriteBatch::default();
                }
            }

            position += 1;
        }

        // Flush remaining
        if removed_count % BATCH_FLUSH_SIZE != 0 {
            self.shared_db
                .write(batch)
                .map_err(|e| anyhow::anyhow!("Failed to write rotation batch: {e}"))?;
        }

        // Sync in-memory cache
        if removed_count > 0 {
            if let Some(log) = self.audit_logs.get(user_id) {
                let mut log_guard = log.write();

                log_guard.retain(|event| {
                    let event_nanos = event.timestamp.timestamp_nanos_opt().unwrap_or(0);
                    event_nanos >= cutoff_nanos
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

    /// Shared DB for all global stores (todos, reminders, files, feedback, audit)
    pub shared_db: Arc<rocksdb::DB>,

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

    /// Session tracking store
    pub session_store: Arc<SessionStore>,

    /// Shared relevance engine for proactive memory surfacing (entity cache + learned weights persist)
    pub relevance_engine: Arc<RelevanceEngine>,

    /// Maintenance cycle counter: cycles 0..5 are lightweight (in-memory only),
    /// cycle 0 (mod 6) is heavyweight (graph decay, fact extraction, flush).
    /// At 300s intervals, heavy cycles fire every 30 minutes.
    maintenance_cycle: std::sync::atomic::AtomicU64,
}

impl MultiUserMemoryManager {
    pub fn new(base_path: std::path::PathBuf, server_config: ServerConfig) -> Result<Self> {
        std::fs::create_dir_all(&base_path)?;

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
        let eviction_base_path = base_path.clone();

        let user_memories = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .eviction_listener(move |key: Arc<String>, value: Arc<parking_lot::RwLock<MemorySystem>>, cause| {
                if cause == moka::notification::RemovalCause::Size {
                    evictions_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    // Spawn blocking task to persist vector index without holding the lock
                    // during I/O. The eviction listener runs synchronously inside moka,
                    // so we must not block here for disk writes.
                    let index_path = eviction_base_path.join(key.as_str()).join("vector_index");
                    let user_key = key.clone();
                    std::thread::spawn(move || {
                        if let Some(guard) = value.try_read() {
                            match guard.save_vector_index(&index_path) {
                                Ok(()) => {
                                    info!(
                                        "Evicted user '{}' from memory cache (LRU, cache_size={}) - vector index saved",
                                        user_key, max_cache
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Evicted user '{}' from memory cache (LRU) - failed to save vector index: {}",
                                        user_key, e
                                    );
                                }
                            }
                        } else {
                            tracing::warn!(
                                "Evicted user '{}' from memory cache (LRU) - could not acquire lock to save index",
                                user_key
                            );
                        }
                    });
                }
            })
            .build();

        let graph_memories = moka::sync::Cache::builder()
            .max_capacity(server_config.max_users_in_memory as u64)
            .eviction_listener(move |key: Arc<String>, _value, _cause| {
                info!("Evicted graph for user '{}' from memory cache (LRU)", key);
            })
            .build();

        // Open a single shared DB for all global stores (todos, reminders, files, feedback, audit).
        // This dramatically reduces file descriptor usage compared to separate DBs per store.
        let shared_db = {
            use rocksdb::{ColumnFamilyDescriptor, Options as RocksOptions};
            let shared_db_path = base_path.join("shared");
            std::fs::create_dir_all(&shared_db_path)?;

            let mut db_opts = RocksOptions::default();
            db_opts.create_if_missing(true);
            db_opts.create_missing_column_families(true);
            db_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            db_opts.set_max_write_buffer_number(2);
            db_opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB

            // Collect CF descriptors from all stores + audit
            let mut cfs = vec![ColumnFamilyDescriptor::new("default", {
                let mut o = RocksOptions::default();
                o.create_if_missing(true);
                o
            })];
            cfs.extend(TodoStore::cf_descriptors());
            cfs.extend(ProspectiveStore::column_family_descriptors());
            cfs.extend(FileMemoryStore::cf_descriptors());
            // Feedback CF
            cfs.push(ColumnFamilyDescriptor::new(
                crate::memory::feedback::CF_FEEDBACK,
                {
                    let mut o = RocksOptions::default();
                    o.create_if_missing(true);
                    o.set_compression_type(rocksdb::DBCompressionType::Lz4);
                    o
                },
            ));
            // Audit CF
            cfs.push(ColumnFamilyDescriptor::new("audit", {
                let mut o = RocksOptions::default();
                o.create_if_missing(true);
                o.set_compression_type(rocksdb::DBCompressionType::Lz4);
                o
            }));

            Arc::new(
                rocksdb::DB::open_cf_descriptors(&db_opts, &shared_db_path, cfs)
                    .context("Failed to open shared DB with column families")?,
            )
        };

        // Migrate old audit_logs DB into shared DB audit CF
        Self::migrate_audit_db(&base_path, &shared_db)?;

        let prospective_store = Arc::new(ProspectiveStore::new(shared_db.clone(), &base_path)?);
        info!("Prospective memory store initialized");

        let todo_store = Arc::new(TodoStore::new(shared_db.clone(), &base_path)?);
        if let Err(e) = todo_store.load_vector_indices() {
            tracing::warn!("Failed to load todo vector indices: {}, semantic todo search will rebuild on first use", e);
        }
        info!("Todo store initialized");

        let file_store = Arc::new(FileMemoryStore::new(shared_db.clone(), &base_path)?);
        info!("File memory store initialized");

        let feedback_store = Arc::new(parking_lot::RwLock::new(
            FeedbackStore::with_shared_db(shared_db.clone(), &base_path).unwrap_or_else(|e| {
                tracing::warn!("Failed to load feedback store: {}, using in-memory", e);
                FeedbackStore::new()
            }),
        ));
        info!("Feedback store initialized");

        // PIPE-9: StreamingMemoryExtractor no longer needs FeedbackStore
        // Feedback momentum is now applied in the MemorySystem pipeline
        let streaming_extractor =
            Arc::new(streaming::StreamingMemoryExtractor::new(neural_ner.clone()));
        info!("Streaming memory extractor initialized");

        let keyword_extractor = Arc::new(KeywordExtractor::new());
        info!("Keyword extractor initialized (YAKE)");

        let relevance_engine = Arc::new(RelevanceEngine::new(neural_ner.clone()));
        info!("Relevance engine initialized (entity cache + learned weights)");

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

        let broadcast_capacity = (server_config.max_users_in_memory * 4).max(64);

        let manager = Self {
            user_memories,
            audit_logs: Arc::new(DashMap::new()),
            shared_db,
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
                let (tx, _) = tokio::sync::broadcast::channel(broadcast_capacity);
                tx
            },
            ab_test_manager: Arc::new(ab_testing::ABTestManager::new()),
            session_store: Arc::new(SessionStore::new()),
            relevance_engine,
            maintenance_cycle: std::sync::atomic::AtomicU64::new(0),
        };

        info!("Running initial audit log rotation...");
        if let Err(e) = manager.rotate_all_audit_logs() {
            tracing::warn!("Failed to rotate audit logs on startup: {}", e);
        }

        Ok(manager)
    }

    /// Get the audit column family handle from the shared DB
    fn audit_cf(&self) -> &rocksdb::ColumnFamily {
        self.shared_db
            .cf_handle(CF_AUDIT)
            .expect("audit CF must exist in shared DB")
    }

    /// Migrate old standalone audit_logs DB into the shared DB's audit CF.
    /// Old directory is renamed to `audit_logs.pre_cf_migration` for rollback safety.
    fn migrate_audit_db(base_path: &std::path::Path, shared_db: &rocksdb::DB) -> Result<()> {
        let old_dir = base_path.join("audit_logs");
        if !old_dir.exists() {
            return Ok(());
        }

        let audit_cf = shared_db
            .cf_handle(CF_AUDIT)
            .expect("audit CF must exist in shared DB");

        // Check if CF already has data (migration already done)
        let mut has_data = false;
        let mut iter = shared_db.raw_iterator_cf(audit_cf);
        iter.seek_to_first();
        if iter.valid() {
            has_data = true;
        }
        if has_data {
            tracing::info!(
                "Audit CF already has data, skipping migration from {:?}",
                old_dir
            );
            return Ok(());
        }

        tracing::info!("Migrating audit_logs from standalone DB to shared DB audit CF...");

        let old_opts = rocksdb::Options::default();
        let old_db = rocksdb::DB::open_for_read_only(&old_opts, &old_dir, false)
            .context("Failed to open old audit_logs DB for migration")?;

        let mut batch = rocksdb::WriteBatch::default();
        let mut count = 0usize;
        const BATCH_SIZE: usize = 10_000;

        let iter = old_db.iterator(rocksdb::IteratorMode::Start);
        for item in iter {
            let (key, value) =
                item.map_err(|e| anyhow::anyhow!("audit migration iter error: {e}"))?;
            batch.put_cf(audit_cf, &key, &value);
            count += 1;

            if count % BATCH_SIZE == 0 {
                shared_db
                    .write(std::mem::take(&mut batch))
                    .map_err(|e| anyhow::anyhow!("audit migration batch write error: {e}"))?;
                batch = rocksdb::WriteBatch::default();
            }
        }

        if count % BATCH_SIZE != 0 {
            shared_db
                .write(batch)
                .map_err(|e| anyhow::anyhow!("audit migration final batch error: {e}"))?;
        }

        drop(old_db);

        let renamed = old_dir.with_file_name("audit_logs.pre_cf_migration");
        std::fs::rename(&old_dir, &renamed)
            .context("Failed to rename old audit_logs dir after migration")?;

        tracing::info!(
            "Migrated {} audit entries from standalone DB to shared CF, old dir renamed to {:?}",
            count,
            renamed
        );

        Ok(())
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
            "{}:{:020}",
            user_id,
            event.timestamp.timestamp_nanos_opt().unwrap_or_else(|| {
                tracing::warn!("audit event timestamp outside i64 nanos range, using 0");
                0
            })
        );
        if let Ok(serialized) = bincode::serde::encode_to_vec(&event, bincode::config::standard()) {
            let db = self.shared_db.clone();
            let key_bytes = key.into_bytes();

            tokio::task::spawn_blocking(move || {
                let audit = db.cf_handle(CF_AUDIT).expect("audit CF must exist");
                if let Err(e) = db.put_cf(&audit, &key_bytes, &serialized) {
                    tracing::error!("Failed to persist audit log: {}", e);
                }
            });
        }

        let max_entries = self.server_config.audit_max_entries_per_user;
        let log = self
            .audit_logs
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(parking_lot::RwLock::new(VecDeque::new())))
            .clone();
        {
            let mut entries = log.write();
            entries.push_back(event);
            while entries.len() > max_entries {
                entries.pop_front();
            }
        }

        let count = self
            .audit_log_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if count % self.server_config.audit_rotation_check_interval == 0 && count > 0 {
            let shared_db = self.shared_db.clone();
            let audit_logs = self.audit_logs.clone();
            let user_id_clone = user_id.to_string();

            let audit_retention_days = self.server_config.audit_retention_days as i64;
            let audit_max_entries = self.server_config.audit_max_entries_per_user;

            tokio::task::spawn_blocking(move || {
                let manager = MultiUserMemoryManagerRotationHelper {
                    shared_db,
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

        let audit = self.audit_cf();
        let iter = self.shared_db.prefix_iterator_cf(audit, prefix.as_bytes());
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
            self.audit_logs
                .entry(user_id.to_string())
                .or_insert_with(|| {
                    Arc::new(parking_lot::RwLock::new(VecDeque::from(events.clone())))
                });
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

        let mut memory_system = MemorySystem::new(config)
            .with_context(|| format!("Failed to initialize memory system for user '{user_id}'"))?;
        // Wire up GraphMemory for Layer 2 (spreading activation) and Layer 5 (Hebbian learning)
        let graph = self.get_user_graph(user_id)?;
        memory_system.set_graph_memory(graph);
        // Wire up FeedbackStore for PIPE-9 (feedback momentum in all retrieval paths)
        memory_system.set_feedback_store(self.feedback_store.clone());

        let memory_arc = Arc::new(parking_lot::RwLock::new(memory_system));

        self.user_memories
            .insert(user_id.to_string(), memory_arc.clone());

        info!("Created memory system for user: {}", user_id);

        Ok(memory_arc)
    }

    /// Delete user data (GDPR compliance)
    ///
    /// Cleans up:
    /// 1. In-memory caches (user_memories, graph_memories)
    /// 2. Shared RocksDB: todos, projects, todo indices, reminders, files, feedback, audit
    /// 3. Per-user filesystem: per-user RocksDB, graph DB, vector indices
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

        // Clean up all user data from shared RocksDB column families
        self.purge_user_from_shared_db(user_id)?;

        // Clean up todo vector indices
        self.todo_store.purge_user_vectors(user_id);

        // Clean up in-memory feedback state
        {
            let mut fb = self.feedback_store.write();
            fb.take_pending(user_id);
        }

        // Delete per-user filesystem (memories DB, graph DB, vector index files)
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
                            "Failed to delete user data after {max_attempts} retries: {e}"
                        ))
                    }
                }
            }
        }

        info!("Deleted all data for user: {}", user_id);
        Ok(())
    }

    /// Prefix-scan and batch-delete all keys starting with `{user_id}:` from a column family
    fn delete_by_prefix(db: &rocksdb::DB, cf: &rocksdb::ColumnFamily, prefix: &[u8]) -> usize {
        let mut batch = rocksdb::WriteBatch::default();
        let mut count = 0;
        let iter = db.prefix_iterator_cf(cf, prefix);
        for item in iter.flatten() {
            let (key, _) = item;
            if !key.starts_with(prefix) {
                break;
            }
            batch.delete_cf(cf, &key);
            count += 1;
        }
        if count > 0 {
            let _ = db.write(batch);
        }
        count
    }

    /// Purge all user data from shared RocksDB (todos, reminders, files, feedback, audit)
    fn purge_user_from_shared_db(&self, user_id: &str) -> Result<()> {
        let prefix = format!("{user_id}:");
        let prefix_bytes = prefix.as_bytes();

        // Shared CF names that use `{user_id}:` as key prefix
        let cf_names = ["todos", "projects", "prospective"];
        for name in &cf_names {
            if let Some(cf) = self.shared_db.cf_handle(name) {
                let n = Self::delete_by_prefix(&self.shared_db, cf, prefix_bytes);
                if n > 0 {
                    tracing::debug!("GDPR: purged {n} entries from {name} CF for {user_id}");
                }
            }
        }

        // Index CFs use varied key prefixes — scan all relevant patterns
        if let Some(cf) = self.shared_db.cf_handle("todo_index") {
            let prefixes = [
                format!("user:{user_id}:"),
                format!("status:Backlog:{user_id}:"),
                format!("status:Todo:{user_id}:"),
                format!("status:InProgress:{user_id}:"),
                format!("status:Blocked:{user_id}:"),
                format!("status:Done:{user_id}:"),
                format!("status:Cancelled:{user_id}:"),
                format!("vector_id:{user_id}:"),
                format!("todo_vector:{user_id}:"),
            ];
            for p in &prefixes {
                Self::delete_by_prefix(&self.shared_db, cf, p.as_bytes());
            }
            // Priority and due/context keys also contain user_id but at varying positions.
            // Full scan of index CF to catch them all.
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.shared_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter.flatten() {
                let (key, _) = item;
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if key_str.contains(&prefix) {
                        batch.delete_cf(cf, &key);
                    }
                }
            }
            let _ = self.shared_db.write(batch);
        }

        if let Some(cf) = self.shared_db.cf_handle("prospective_index") {
            let prefixes = [
                format!("user:{user_id}:"),
                format!("status:Pending:{user_id}:"),
                format!("status:Triggered:{user_id}:"),
                format!("status:Dismissed:{user_id}:"),
            ];
            for p in &prefixes {
                Self::delete_by_prefix(&self.shared_db, cf, p.as_bytes());
            }
            // Context keyword indices: `context:{keyword}:{user_id}:{id}`
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.shared_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter.flatten() {
                let (key, _) = item;
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if key_str.contains(&prefix) {
                        batch.delete_cf(cf, &key);
                    }
                }
            }
            let _ = self.shared_db.write(batch);
        }

        // Files
        if let Some(cf) = self.shared_db.cf_handle("files") {
            Self::delete_by_prefix(&self.shared_db, cf, prefix_bytes);
        }
        if let Some(cf) = self.shared_db.cf_handle("file_index") {
            let idx_prefix = format!("file_idx:{user_id}:");
            Self::delete_by_prefix(&self.shared_db, cf, idx_prefix.as_bytes());
            // Also catch other patterns
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.shared_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter.flatten() {
                let (key, _) = item;
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if key_str.contains(&prefix) {
                        batch.delete_cf(cf, &key);
                    }
                }
            }
            let _ = self.shared_db.write(batch);
        }

        // Feedback: `pending:{user_id}`
        if let Some(cf) = self.shared_db.cf_handle("feedback") {
            let pending_key = format!("pending:{user_id}");
            let _ = self.shared_db.delete_cf(cf, pending_key.as_bytes());
        }

        // Audit logs
        if let Some(cf) = self.shared_db.cf_handle("audit") {
            Self::delete_by_prefix(&self.shared_db, cf, prefix_bytes);
        }

        // Clear in-memory audit log cache
        self.audit_logs.remove(user_id);

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
                                && name != "audit_logs.pre_cf_migration"
                                && name != "backups"
                                && name != "feedback"
                                && name != "feedback.pre_cf_migration"
                                && name != "semantic_facts"
                                && name != "files"
                                && name != "files.pre_cf_migration"
                                && name != "prospective"
                                && name != "prospective.pre_cf_migration"
                                && name != "todos"
                                && name != "todos.pre_cf_migration"
                                && name != "shared"
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
        let audit = self.audit_cf();
        let iter = self.shared_db.prefix_iterator_cf(audit, prefix.as_bytes());
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
        events.reverse();
        events.truncate(limit);
        events
    }

    /// Flush all RocksDB databases
    pub fn flush_all_databases(&self) -> Result<()> {
        info!("Flushing all databases to disk...");

        // Single flush covers all shared stores (todos, prospective, files, feedback, audit)
        self.shared_db
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush shared database: {e}"))?;
        info!("  Shared database flushed (todos, prospective, files, feedback, audit)");

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
            "All databases flushed: shared (5 stores), {} user memories",
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
        let audit = self.audit_cf();
        let iter = self
            .shared_db
            .iterator_cf(audit, rocksdb::IteratorMode::Start);

        for (key, _) in iter.flatten() {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if let Some(user_id) = key_str.split(':').next() {
                    user_ids.insert(user_id.to_string());
                }
            }
        }

        let helper = MultiUserMemoryManagerRotationHelper {
            shared_db: self.shared_db.clone(),
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
        let cycle = self
            .maintenance_cycle
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Heavy cycle every 6th iteration (6 hours at 3600s intervals).
        // Heavy cycles run replay, entity-entity strengthening, fact extraction (full memory scan),
        // and flush databases (triggers compaction). Light cycles only touch in-memory data.
        let is_heavy = cycle % 6 == 0;

        if is_heavy {
            tracing::info!(
                "Maintenance cycle {} (HEAVY — graph decay + fact extraction + flush)",
                cycle
            );
        } else {
            tracing::debug!("Maintenance cycle {} (light — in-memory only)", cycle);
        }

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
        let mut entity_edges_strengthened = 0;
        let mut total_facts_extracted = 0;
        let mut total_facts_reinforced = 0;

        for user_id in user_ids {
            let maintenance_result = if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                let memory = memory_lock.read();
                match memory.run_maintenance(decay_factor, &user_id, is_heavy) {
                    Ok(result) => {
                        total_processed += result.decayed_count;
                        total_facts_extracted += result.facts_extracted;
                        total_facts_reinforced += result.facts_reinforced;
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

            // Direction 1: Edge strengthening + promotion boost propagation
            if let Some(ref result) = maintenance_result {
                if !result.edge_boosts.is_empty() {
                    if let Ok(graph) = self.get_user_graph(&user_id) {
                        let graph_guard = graph.read();
                        match graph_guard.strengthen_memory_edges(&result.edge_boosts) {
                            Ok((count, promotion_boosts)) => {
                                edges_strengthened += count;

                                // Direction 1: Apply edge promotion boosts to memory importance
                                if !promotion_boosts.is_empty() {
                                    if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                                        let memory = memory_lock.read();
                                        match memory.apply_edge_promotion_boosts(&promotion_boosts)
                                        {
                                            Ok(boosted) => {
                                                tracing::debug!(
                                                    user_id = %user_id,
                                                    boosted,
                                                    promotions = promotion_boosts.len(),
                                                    "Applied edge promotion boosts"
                                                );
                                            }
                                            Err(e) => {
                                                tracing::debug!(
                                                    "Edge promotion boost failed for user {}: {}",
                                                    user_id,
                                                    e
                                                );
                                            }
                                        }
                                    }
                                }
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

            // Direction 3: Entity-entity Hebbian reinforcement for replayed memories
            // During replay, memories are re-activated — strengthen edges between entities
            // that co-occur in the same episode, reinforcing semantic associations.
            if let Some(ref result) = maintenance_result {
                if !result.replay_memory_ids.is_empty() {
                    if let Ok(graph) = self.get_user_graph(&user_id) {
                        let graph_guard = graph.read();
                        for mem_id_str in &result.replay_memory_ids {
                            if let Ok(uuid) = uuid::Uuid::parse_str(mem_id_str) {
                                match graph_guard.strengthen_episode_entity_edges(&uuid) {
                                    Ok(count) => entity_edges_strengthened += count,
                                    Err(e) => {
                                        tracing::debug!(
                                            "Entity edge strengthening failed for memory {}: {}",
                                            mem_id_str,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Direction 2: Lazy decay — flush opportunistic pruning queue
            // Instead of scanning all 34k+ edges (apply_decay), we queue edges found
            // below threshold during normal reads and batch-delete them here.
            // Runs every cycle since it's just targeted deletes, not a full scan.
            if let Ok(graph) = self.get_user_graph(&user_id) {
                let graph_guard = graph.read();
                match graph_guard.flush_pending_maintenance() {
                    Ok(decay_result) => {
                        edges_decayed += decay_result.pruned_count;

                        // Direction 2: Compensate memories that lost all graph edges
                        if !decay_result.orphaned_entity_ids.is_empty() {
                            if let Ok(memory_lock) = self.get_user_memory(&user_id) {
                                let memory = memory_lock.read();
                                match memory
                                    .compensate_orphaned_memories(&decay_result.orphaned_entity_ids)
                                {
                                    Ok(compensated) => {
                                        tracing::debug!(
                                            user_id = %user_id,
                                            compensated,
                                            orphaned = decay_result.orphaned_entity_ids.len(),
                                            "Compensated orphaned memories"
                                        );
                                    }
                                    Err(e) => {
                                        tracing::debug!(
                                            "Orphan compensation failed for user {}: {}",
                                            user_id,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!("Graph lazy pruning failed for user {}: {}", user_id, e);
                    }
                }
            }
        }

        // Flush databases only on heavy cycles — flush triggers RocksDB compaction
        // which allocates significant C++ memory through Windows CRT
        if is_heavy {
            if let Err(e) = self.flush_all_databases() {
                tracing::warn!("Periodic flush failed: {}", e);
            }
        }

        tracing::info!(
            "Maintenance complete (cycle {}, {}): {} memories processed, {} edges strengthened, {} entity edges strengthened, {} weak edges pruned, {} facts extracted, {} facts reinforced across {} users",
            cycle,
            if is_heavy { "heavy" } else { "light" },
            total_processed,
            edges_strengthened,
            entity_edges_strengthened,
            edges_decayed,
            total_facts_extracted,
            total_facts_reinforced,
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

    /// Active reminder check: scan all users for due reminders, mark them triggered,
    /// and emit `REMINDER_DUE` events to the broadcast channel.
    ///
    /// Called by the dedicated 60-second reminder scheduler in main.rs.
    /// Returns the number of reminders triggered.
    pub fn check_and_emit_due_reminders(&self) -> usize {
        let due_tasks = match self.prospective_store.get_all_due_tasks() {
            Ok(tasks) => tasks,
            Err(e) => {
                tracing::debug!("Active reminder check failed: {}", e);
                return 0;
            }
        };

        let mut triggered = 0;
        for (user_id, task) in &due_tasks {
            let _ = self.prospective_store.mark_triggered(user_id, &task.id);

            self.emit_event(MemoryEvent {
                event_type: "REMINDER_DUE".to_string(),
                timestamp: chrono::Utc::now(),
                user_id: user_id.clone(),
                memory_id: Some(task.id.0.to_string()),
                content_preview: Some(task.content.chars().take(100).collect()),
                memory_type: Some("reminder".to_string()),
                importance: Some(task.priority as f32 / 5.0),
                count: None,
                results: None,
            });

            tracing::info!(
                user_id = %user_id,
                reminder_id = %task.id.0,
                content = %task.content.chars().take(50).collect::<String>(),
                "Reminder triggered (active)"
            );

            triggered += 1;
        }

        triggered
    }

    /// Collect references to all secondary store databases for comprehensive backup.
    /// All shared stores (todos, prospective, files, feedback, audit) share a single DB,
    /// so we return one reference. BackupEngine handles all CFs automatically.
    pub fn collect_secondary_store_refs(&self) -> Vec<(String, std::sync::Arc<rocksdb::DB>)> {
        vec![("shared".to_string(), std::sync::Arc::clone(&self.shared_db))]
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
                    let secondary_refs = self.collect_secondary_store_refs();
                    let store_refs: Vec<crate::backup::SecondaryStoreRef<'_>> = secondary_refs
                        .iter()
                        .map(|(n, d)| crate::backup::SecondaryStoreRef { name: n, db: d })
                        .collect();
                    match self
                        .backup_engine
                        .create_comprehensive_backup(&db, name, &store_refs)
                    {
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

        // =====================================================================
        // PHASE 1: CPU-INTENSIVE WORK (NO LOCK)
        // All NER, regex, query parsing happens here to minimize lock hold time.
        // Was 100-400ms under lock, now only fast I/O under lock (~10-30ms).
        // =====================================================================

        let now = chrono::Utc::now();

        // Stop words for filtering
        let stop_words: std::collections::HashSet<&str> = [
            "the", "and", "for", "that", "this", "with", "from", "have", "been", "are", "was",
            "were", "will", "would", "could", "should", "may", "might",
        ]
        .iter()
        .cloned()
        .collect();

        // Use pre-extracted NER records for proper entity labels when available
        // This avoids redundant NER inference — the handler already ran NER in Pass 1
        let extracted_entities = if !experience.ner_entities.is_empty() {
            tracing::debug!(
                "Using {} pre-extracted NER entities from handler",
                experience.ner_entities.len()
            );
            experience
                .ner_entities
                .iter()
                .map(|record| crate::embeddings::ner::NerEntity {
                    text: record.text.clone(),
                    entity_type: match record.entity_type.as_str() {
                        "PER" => NerEntityType::Person,
                        "ORG" => NerEntityType::Organization,
                        "LOC" => NerEntityType::Location,
                        _ => NerEntityType::Misc,
                    },
                    confidence: record.confidence,
                    start: 0,
                    end: record.text.len(),
                })
                .collect()
        } else if !experience.entities.is_empty() {
            tracing::debug!(
                "Using {} pre-extracted entity names (no NER types available)",
                experience.entities.len()
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

        // Filter noise entities
        let filtered_entities: Vec<_> = extracted_entities
            .into_iter()
            .filter(|e| {
                let name = e.text.trim();
                if name.len() < 3 {
                    return false;
                }
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

        // Build NER entity nodes
        let ner_entities: Vec<(String, EntityNode)> = filtered_entities
            .into_iter()
            .map(|ner_entity| {
                let label = match ner_entity.entity_type {
                    NerEntityType::Person => EntityLabel::Person,
                    NerEntityType::Organization => EntityLabel::Organization,
                    NerEntityType::Location => EntityLabel::Location,
                    NerEntityType::Misc => EntityLabel::Other("MISC".to_string()),
                };
                let node = EntityNode {
                    uuid: uuid::Uuid::new_v4(),
                    name: ner_entity.text.clone(),
                    labels: vec![label],
                    created_at: now,
                    last_seen_at: now,
                    mention_count: 1,
                    summary: String::new(),
                    attributes: HashMap::new(),
                    name_embedding: None,
                    salience: ner_entity.confidence,
                    is_proper_noun: true,
                };
                (ner_entity.text, node)
            })
            .collect();

        // Build tag entity nodes
        let tag_entities: Vec<(String, EntityNode)> = experience
            .tags
            .iter()
            .filter_map(|tag| {
                let tag_name = tag.trim();
                if tag_name.len() >= 2 && !stop_words.contains(tag_name.to_lowercase().as_str()) {
                    Some((
                        tag_name.to_string(),
                        EntityNode {
                            uuid: uuid::Uuid::new_v4(),
                            name: tag_name.to_string(),
                            labels: vec![EntityLabel::Technology],
                            created_at: now,
                            last_seen_at: now,
                            mention_count: 1,
                            summary: String::new(),
                            attributes: HashMap::new(),
                            name_embedding: None,
                            salience: 0.6,
                            is_proper_noun: false,
                        },
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Collect names already covered (for dedup in regex/verb phases)
        let mut known_names: Vec<String> = ner_entities
            .iter()
            .map(|(name, _)| name.clone())
            .chain(tag_entities.iter().map(|(name, _)| name.clone()))
            .collect();

        // Extract all-caps terms (API, TUI, NER, REST, etc.)
        let allcaps_entities: Vec<(String, EntityNode)> = allcaps_regex()
            .find_iter(&experience.content)
            .filter_map(|cap| {
                let term = cap.as_str();
                if known_names
                    .iter()
                    .any(|name| name.eq_ignore_ascii_case(term))
                {
                    return None;
                }
                if stop_words.contains(term.to_lowercase().as_str()) {
                    return None;
                }
                known_names.push(term.to_string());
                Some((
                    term.to_string(),
                    EntityNode {
                        uuid: uuid::Uuid::new_v4(),
                        name: term.to_string(),
                        labels: vec![EntityLabel::Technology],
                        created_at: now,
                        last_seen_at: now,
                        mention_count: 1,
                        summary: String::new(),
                        attributes: HashMap::new(),
                        name_embedding: None,
                        salience: 0.5,
                        is_proper_noun: true,
                    },
                ))
            })
            .collect();

        // Extract issue IDs (SHO-XX, JIRA-123, etc.)
        let issue_entities: Vec<(String, EntityNode)> = issue_regex()
            .find_iter(&experience.content)
            .filter_map(|issue| {
                let issue_id = issue.as_str();
                if known_names.iter().any(|name| name == issue_id) {
                    return None;
                }
                known_names.push(issue_id.to_string());
                Some((
                    issue_id.to_string(),
                    EntityNode {
                        uuid: uuid::Uuid::new_v4(),
                        name: issue_id.to_string(),
                        labels: vec![EntityLabel::Other("Issue".to_string())],
                        created_at: now,
                        last_seen_at: now,
                        mention_count: 1,
                        summary: String::new(),
                        attributes: HashMap::new(),
                        name_embedding: None,
                        salience: 0.7,
                        is_proper_noun: true,
                    },
                ))
            })
            .collect();

        // Extract verbs for multi-hop reasoning
        let analysis = query_parser::analyze_query(&experience.content);
        let mut verb_entities: Vec<(String, EntityNode)> = Vec::new();
        for verb in &analysis.relational_context {
            let verb_text = verb.text.as_str();
            let verb_stem = verb.stem.as_str();

            if known_names
                .iter()
                .any(|name| name.eq_ignore_ascii_case(verb_text))
            {
                continue;
            }
            if stop_words.contains(verb_text.to_lowercase().as_str()) {
                continue;
            }
            if verb_text.len() < 3 {
                continue;
            }

            for name in [verb_text, verb_stem] {
                if name.len() < 3 {
                    continue;
                }
                if known_names.iter().any(|n| n.eq_ignore_ascii_case(name)) {
                    continue;
                }
                known_names.push(name.to_string());
                verb_entities.push((
                    name.to_string(),
                    EntityNode {
                        uuid: uuid::Uuid::new_v4(),
                        name: name.to_string(),
                        labels: vec![EntityLabel::Other("Verb".to_string())],
                        created_at: now,
                        last_seen_at: now,
                        mention_count: 1,
                        summary: String::new(),
                        attributes: HashMap::new(),
                        name_embedding: None,
                        salience: 0.4,
                        is_proper_noun: false,
                    },
                ));
            }
        }

        // Combine all entity groups for insertion, capped at 10 to prevent
        // O(n²) edge explosion (10 entities → max 45 edges)
        let mut all_entities: Vec<(String, EntityNode)> = ner_entities
            .into_iter()
            .chain(tag_entities)
            .chain(allcaps_entities)
            .chain(issue_entities)
            .chain(verb_entities)
            .collect();
        all_entities.sort_by(|a, b| b.1.salience.total_cmp(&a.1.salience));
        let entity_cap = self.server_config.max_entities_per_memory;
        all_entities.truncate(entity_cap);

        // =====================================================================
        // PHASE 2: GRAPH INSERTION (WITH LOCK)
        // Only fast I/O operations happen here.
        // =====================================================================

        let graph_guard = graph.read();

        let mut entity_uuids = Vec::new();

        // Insert all pre-built entities
        for (name, entity) in all_entities {
            match graph_guard.add_entity(entity) {
                Ok(uuid) => entity_uuids.push((name, uuid)),
                Err(e) => tracing::debug!("Failed to add entity {}: {}", name, e),
            }
        }

        // Create episodic node
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
            valid_at: now,
            created_at: now,
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
        // Pre-compute truncated context once (avoids re-allocating per edge)
        let truncated_context: String = experience.content.chars().take(150).collect();
        for i in 0..entity_uuids.len() {
            for j in (i + 1)..entity_uuids.len() {
                let edge = RelationshipEdge {
                    uuid: uuid::Uuid::new_v4(),
                    from_entity: entity_uuids[i].1,
                    to_entity: entity_uuids[j].1,
                    relation_type: RelationType::RelatedTo,
                    strength: EdgeTier::L1Working.initial_weight(),
                    created_at: now,
                    valid_at: now,
                    invalidated_at: None,
                    source_episode_id: Some(memory_id.0),
                    context: truncated_context.clone(),
                    last_activated: now,
                    activation_count: 1,
                    ltp_status: LtpStatus::None,
                    tier: EdgeTier::L1Working,
                    activation_timestamps: None,
                    entity_confidence: None,
                };

                if let Err(e) = graph_guard.add_relationship(edge) {
                    tracing::debug!("Failed to add relationship: {}", e);
                }
            }
        }
        // Lock released here

        Ok(())
    }
}
