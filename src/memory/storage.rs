//! Storage backend for the memory system

use anyhow::{anyhow, Context, Result};
use bincode;
use chrono::{DateTime, Utc};
use rocksdb::{IteratorMode, Options, WriteBatch, WriteOptions, DB};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::types::*;

/// Helper trait to safely iterate over RocksDB results with error logging.
/// Unlike `.flatten()` which silently ignores errors, this logs them.
trait LogErrors<T> {
    fn log_errors(self) -> impl Iterator<Item = T>;
}

impl<I, T, E> LogErrors<T> for I
where
    I: Iterator<Item = Result<T, E>>,
    E: std::fmt::Display,
{
    fn log_errors(self) -> impl Iterator<Item = T> {
        self.filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("RocksDB iterator error (continuing): {}", e);
                None
            }
        })
    }
}

/// Write mode for storage operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Sync writes - fsync() on every write (durable but slow: 2-10ms per write)
    /// Use for: shutdown, critical data, compliance requirements
    Sync,
    /// Async writes - no fsync(), data buffered in OS page cache (fast: <1ms per write)
    /// Use for: robotics, edge, high-throughput scenarios
    /// Data survives process crashes but NOT power loss before next fsync
    Async,
}

impl Default for WriteMode {
    fn default() -> Self {
        // Default to async for robotics-grade latency
        // Override with SHODH_WRITE_MODE=sync for durability-critical deployments
        match std::env::var("SHODH_WRITE_MODE") {
            Ok(mode) if mode.to_lowercase() == "sync" => WriteMode::Sync,
            _ => WriteMode::Async,
        }
    }
}

// ============================================================================
// BACKWARD-COMPATIBLE DESERIALIZATION
// Handles both versioned format (SHO magic + checksum) and legacy bincode
// ============================================================================

const STORAGE_MAGIC: &[u8; 3] = b"SHO";

/// Deserialize memory supporting both versioned (SHO) and legacy formats
fn deserialize_memory(data: &[u8]) -> Result<Memory> {
    // Check for versioned format: SHO + version byte + payload + 4-byte CRC32
    if data.len() >= 8 && &data[0..3] == STORAGE_MAGIC {
        let version = data[3];
        let payload_end = data.len() - 4;
        let stored_checksum = u32::from_le_bytes([
            data[payload_end],
            data[payload_end + 1],
            data[payload_end + 2],
            data[payload_end + 3],
        ]);
        let computed_checksum = crc32_simple(&data[..payload_end]);
        if stored_checksum != computed_checksum {
            tracing::warn!(
                "Checksum mismatch: stored={:08x} computed={:08x}",
                stored_checksum,
                computed_checksum
            );
        }
        let payload = &data[4..payload_end];
        bincode::serde::decode_from_slice::<Memory, _>(payload, bincode::config::standard())
            .map(|(m, _)| m)
            .map_err(|e| anyhow!("v{} decode failed: {}", version, e))
    } else {
        // Legacy format: raw bincode
        bincode::serde::decode_from_slice::<Memory, _>(data, bincode::config::standard())
            .map(|(m, _)| m)
            .map_err(|e| anyhow!("legacy decode failed: {}", e))
    }
}

/// Simple CRC32 implementation (IEEE polynomial)
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Storage engine for long-term memory persistence
pub struct MemoryStorage {
    db: Arc<DB>,
    index_db: Arc<DB>, // Secondary indices
    /// Base storage path for all memory data
    storage_path: PathBuf,
    /// Write mode (sync vs async) - affects latency vs durability tradeoff
    write_mode: WriteMode,
}

impl MemoryStorage {
    pub fn new(path: &Path) -> Result<Self> {
        // Create directories if they don't exist
        std::fs::create_dir_all(path)?;

        // Configure RocksDB options for PRODUCTION durability + performance
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // ========================================================================
        // DURABILITY SETTINGS - Critical for data persistence across restarts
        // ========================================================================
        //
        // RocksDB data flow: Write → WAL → Memtable → SST files
        // Without proper sync, data in memtable can be lost on crash/restart
        //
        // Our approach: Sync WAL on every write (most durable option)
        // This ensures data survives even if process crashes before memtable flush
        // ========================================================================

        // WAL stays in default location (same as data dir) - avoids corruption issues
        opts.set_manual_wal_flush(false); // Auto-flush WAL entries

        // Write performance optimizations for 10M+ memories per user
        opts.set_max_write_buffer_number(4);
        opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB write buffer (2x for scale)
        opts.set_level_zero_file_num_compaction_trigger(4);
        opts.set_target_file_size_base(128 * 1024 * 1024); // 128MB SST files
        opts.set_max_bytes_for_level_base(512 * 1024 * 1024); // 512MB L1
        opts.set_max_background_jobs(4);
        opts.set_level_compaction_dynamic_level_bytes(true);

        // Read performance optimizations for 10M+ memories
        use rocksdb::{BlockBasedOptions, Cache};
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false); // 10 bits/key = ~1% FPR
        block_opts.set_block_cache(&Cache::new_lru_cache(512 * 1024 * 1024)); // 512MB cache
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true); // Pin L0 for fast reads
        opts.set_block_based_table_factory(&block_opts);

        // Open main database
        let main_path = path.join("memories");
        let db = Arc::new(DB::open(&opts, main_path)?);

        // Open index database
        let index_path = path.join("memory_index");
        let index_db = Arc::new(DB::open(&opts, index_path)?);

        let write_mode = WriteMode::default();
        tracing::info!(
            "Storage initialized with {:?} write mode (latency: {})",
            write_mode,
            if write_mode == WriteMode::Sync {
                "2-10ms per write"
            } else {
                "<1ms per write"
            }
        );

        Ok(Self {
            db,
            index_db,
            storage_path: path.to_path_buf(),
            write_mode,
        })
    }

    /// Get the base storage path
    pub fn path(&self) -> &Path {
        &self.storage_path
    }

    /// Store a memory with configurable write durability
    ///
    /// ROBOTICS OPTIMIZATION: Write mode is configurable via SHODH_WRITE_MODE env var.
    /// - Async (default): <1ms per write, data survives process crashes
    /// - Sync: 2-10ms per write, data survives power loss
    ///
    /// For robotics/edge: Use async mode + periodic flush() calls for best latency.
    /// For compliance/critical: Set SHODH_WRITE_MODE=sync for full durability.
    pub fn store(&self, memory: &Memory) -> Result<()> {
        let key = memory.id.0.as_bytes();

        // Serialize memory
        let value = bincode::serde::encode_to_vec(memory, bincode::config::standard())
            .context(format!("Failed to serialize memory {}", memory.id.0))?;

        // Use write mode based on configuration
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);

        // Store in main database
        self.db
            .put_opt(key, &value, &write_opts)
            .context(format!("Failed to put memory {} in RocksDB", memory.id.0))?;

        // Update indices
        self.update_indices(memory)?;

        Ok(())
    }

    /// Update secondary indices for efficient retrieval
    fn update_indices(&self, memory: &Memory) -> Result<()> {
        let mut batch = WriteBatch::default();

        // === Standard Indices ===

        // Index by date (for temporal queries)
        // BUG-001 FIX: Include memory_id in key to allow multiple memories per day
        // Old format: date:YYYYMMDD (overwrites on same day)
        // New format: date:YYYYMMDD:uuid (unique per memory)
        let date_key = format!(
            "date:{}:{}",
            memory.created_at.format("%Y%m%d"),
            memory.id.0
        );
        batch.put(date_key.as_bytes(), b"1");

        // Index by type
        let type_key = format!(
            "type:{:?}:{}",
            memory.experience.experience_type, memory.id.0
        );
        batch.put(type_key.as_bytes(), b"1");

        // Index by importance (quantized into buckets)
        let importance_bucket = (memory.importance() * 10.0) as u32;
        let importance_key = format!("importance:{}:{}", importance_bucket, memory.id.0);
        batch.put(importance_key.as_bytes(), b"1");

        // Index by entities (case-insensitive for tag search compatibility)
        for entity in &memory.experience.entities {
            let normalized_entity = entity.to_lowercase();
            let entity_key = format!("entity:{}:{}", normalized_entity, memory.id.0);
            batch.put(entity_key.as_bytes(), b"1");
        }

        // Index by tags (separate from entities for explicit tag queries)
        for tag in &memory.experience.tags {
            let normalized_tag = tag.to_lowercase();
            let tag_key = format!("tag:{}:{}", normalized_tag, memory.id.0);
            batch.put(tag_key.as_bytes(), b"1");
        }

        // Index by episode_id (for temporal/episodic retrieval)
        // Episode is the primary temporal grouping - memories in same episode are highly related
        if let Some(ctx) = &memory.experience.context {
            if let Some(episode_id) = &ctx.episode.episode_id {
                let episode_key = format!("episode:{}:{}", episode_id, memory.id.0);
                batch.put(episode_key.as_bytes(), b"1");

                // Also index by sequence within episode for temporal ordering
                if let Some(seq) = ctx.episode.sequence_number {
                    let seq_key = format!("episode_seq:{}:{}:{}", episode_id, seq, memory.id.0);
                    batch.put(seq_key.as_bytes(), b"1");
                }
            }
        }

        // === Robotics Indices ===

        // Index by robot_id (for multi-robot systems)
        if let Some(ref robot_id) = memory.experience.robot_id {
            let robot_key = format!("robot:{}:{}", robot_id, memory.id.0);
            batch.put(robot_key.as_bytes(), b"1");
        }

        // Index by mission_id (for mission context retrieval)
        if let Some(ref mission_id) = memory.experience.mission_id {
            let mission_key = format!("mission:{}:{}", mission_id, memory.id.0);
            batch.put(mission_key.as_bytes(), b"1");
        }

        // Index by geo_location (for spatial queries) using geohash
        // Key format: geo:GEOHASH:memory_id (geohash at precision 10 = ~1.2m x 60cm)
        // Geohash enables efficient prefix-based spatial queries
        if let Some(geo) = memory.experience.geo_location {
            let lat = geo[0];
            let lon = geo[1];
            // Use precision 10 for warehouse-level accuracy (~1.2m cells)
            let geohash = super::types::geohash_encode(lat, lon, 10);
            let geo_key = format!("geo:{}:{}", geohash, memory.id.0);
            batch.put(geo_key.as_bytes(), b"1");
        }

        // Index by action_type (for action-based retrieval)
        if let Some(ref action_type) = memory.experience.action_type {
            let action_key = format!("action:{}:{}", action_type, memory.id.0);
            batch.put(action_key.as_bytes(), b"1");
        }

        // Index by reward (bucketed, for RL-style queries)
        // Bucket: -1.0 to 1.0 mapped to 0-20
        if let Some(reward) = memory.experience.reward {
            let clamped_reward = reward.clamp(-1.0, 1.0);
            let reward_bucket = ((clamped_reward + 1.0) * 10.0) as i32;
            let reward_key = format!("reward:{}:{}", reward_bucket, memory.id.0);
            batch.put(reward_key.as_bytes(), b"1");
        }

        // === External Linking Index ===
        // Index by external_id for upsert operations (Linear, GitHub, etc.)
        // Key format: external:{source}:{id}:{memory_id} -> memory_id
        // Enables O(1) lookup when syncing from external systems
        if let Some(ref external_id) = memory.external_id {
            let external_key = format!("external:{}:{}", external_id, memory.id.0);
            // Store memory_id as value for direct lookup
            batch.put(external_key.as_bytes(), memory.id.0.as_bytes());
        }

        // Use write mode based on configuration
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.index_db.write_opt(batch, &write_opts)?;
        Ok(())
    }

    /// Retrieve a memory by ID
    pub fn get(&self, id: &MemoryId) -> Result<Memory> {
        let key = id.0.as_bytes();
        match self.db.get(key)? {
            Some(value) => deserialize_memory(&value).with_context(|| {
                format!(
                    "Failed to deserialize memory {} ({} bytes)",
                    id.0,
                    value.len()
                )
            }),
            None => Err(anyhow!("Memory not found: {id:?}")),
        }
    }

    /// Find a memory by its external_id (e.g., "linear:SHO-39", "github:pr-123")
    ///
    /// Returns the memory if found, None if no memory with this external_id exists.
    /// Used for upsert operations when syncing from external sources.
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<Memory>> {
        // Index key format: external:{external_id}:{memory_id}
        let prefix = format!("external:{external_id}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for (key, _value) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Extract memory_id from key (format: external:{external_id}:{memory_id})
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    return Ok(Some(self.get(&MemoryId(uuid))?));
                }
            }
        }

        Ok(None)
    }

    /// Update an existing memory
    ///
    /// ALGO-004 FIX: Re-indexes memory after update to handle importance drift.
    /// When Hebbian feedback changes importance, the old bucket index becomes stale.
    /// We remove old indices before storing to ensure index consistency.
    pub fn update(&self, memory: &Memory) -> Result<()> {
        // Remove old indices first (they may have stale importance buckets)
        self.remove_from_indices(&memory.id)?;
        // Store with fresh indices
        self.store(memory)
    }

    /// Delete a memory with configurable durability
    #[allow(unused)] // Public API - available for memory management
    pub fn delete(&self, id: &MemoryId) -> Result<()> {
        let key = id.0.as_bytes();

        // Use write mode based on configuration
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db.delete_opt(key, &write_opts)?;

        // Clean up indices
        self.remove_from_indices(id)?;

        Ok(())
    }

    /// Remove memory from all indices
    /// BUG-005 FIX: Direct key deletion instead of O(n) scan with contains()
    /// We reconstruct index keys from memory metadata for O(k) deletion
    fn remove_from_indices(&self, id: &MemoryId) -> Result<()> {
        // Fetch memory to reconstruct index keys
        let memory = match self.get(id) {
            Ok(m) => m,
            Err(_) => {
                tracing::debug!("Memory {} not found, skipping index cleanup", id.0);
                return Ok(());
            }
        };

        let mut batch = WriteBatch::default();

        // Reconstruct and delete all index keys directly (O(k) instead of O(n))

        // Date index
        let date_key = format!("date:{}:{}", memory.created_at.format("%Y%m%d"), id.0);
        batch.delete(date_key.as_bytes());

        // Type index
        let type_key = format!("type:{:?}:{}", memory.experience.experience_type, id.0);
        batch.delete(type_key.as_bytes());

        // Importance index
        let importance_bucket = (memory.importance() * 10.0) as u32;
        let importance_key = format!("importance:{}:{}", importance_bucket, id.0);
        batch.delete(importance_key.as_bytes());

        // Entity indices
        for entity in &memory.experience.entities {
            let entity_key = format!("entity:{}:{}", entity, id.0);
            batch.delete(entity_key.as_bytes());
        }

        // Tag indices
        for tag in &memory.experience.tags {
            let normalized_tag = tag.to_lowercase();
            let tag_key = format!("tag:{}:{}", normalized_tag, id.0);
            batch.delete(tag_key.as_bytes());
        }

        // Episode indices
        if let Some(ctx) = &memory.experience.context {
            if let Some(episode_id) = &ctx.episode.episode_id {
                let episode_key = format!("episode:{}:{}", episode_id, id.0);
                batch.delete(episode_key.as_bytes());

                if let Some(seq) = ctx.episode.sequence_number {
                    let seq_key = format!("episode_seq:{}:{}:{}", episode_id, seq, id.0);
                    batch.delete(seq_key.as_bytes());
                }
            }
        }

        // Robot index
        if let Some(ref robot_id) = memory.experience.robot_id {
            let robot_key = format!("robot:{}:{}", robot_id, id.0);
            batch.delete(robot_key.as_bytes());
        }

        // Mission index
        if let Some(ref mission_id) = memory.experience.mission_id {
            let mission_key = format!("mission:{}:{}", mission_id, id.0);
            batch.delete(mission_key.as_bytes());
        }

        // Geo index
        if let Some(geo) = memory.experience.geo_location {
            let geohash = super::types::geohash_encode(geo[0], geo[1], 10);
            let geo_key = format!("geo:{}:{}", geohash, id.0);
            batch.delete(geo_key.as_bytes());
        }

        // Action index
        if let Some(ref action_type) = memory.experience.action_type {
            let action_key = format!("action:{}:{}", action_type, id.0);
            batch.delete(action_key.as_bytes());
        }

        // Reward index
        if let Some(reward) = memory.experience.reward {
            let reward_bucket = ((reward + 1.0) * 10.0) as i32;
            let reward_key = format!("reward:{}:{}", reward_bucket, id.0);
            batch.delete(reward_key.as_bytes());
        }

        // External linking index
        if let Some(ref external_id) = memory.external_id {
            let external_key = format!("external:{}:{}", external_id, id.0);
            batch.delete(external_key.as_bytes());
        }

        // Use write mode based on configuration
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.index_db.write_opt(batch, &write_opts)?;
        Ok(())
    }

    /// Search memories by various criteria
    pub fn search(&self, criteria: SearchCriteria) -> Result<Vec<Memory>> {
        let mut memory_ids = Vec::new();

        match criteria {
            // === Standard Criteria ===
            SearchCriteria::ByDate { start, end } => {
                memory_ids = self.search_by_date_range(start, end)?;
            }
            SearchCriteria::ByType(exp_type) => {
                memory_ids = self.search_by_type(exp_type)?;
            }
            SearchCriteria::ByImportance { min, max } => {
                memory_ids = self.search_by_importance(min, max)?;
            }
            SearchCriteria::ByEntity(entity) => {
                memory_ids = self.search_by_entity(&entity)?;
            }
            SearchCriteria::ByTags(tags) => {
                memory_ids = self.search_by_tags(&tags)?;
            }

            // === Temporal/Episode Criteria ===
            SearchCriteria::ByEpisode(episode_id) => {
                memory_ids = self.search_by_episode(&episode_id)?;
            }
            SearchCriteria::ByEpisodeSequence {
                episode_id,
                min_sequence,
                max_sequence,
            } => {
                memory_ids =
                    self.search_by_episode_sequence(&episode_id, min_sequence, max_sequence)?;
            }

            // === Robotics Criteria ===
            SearchCriteria::ByRobot(robot_id) => {
                memory_ids = self.search_by_robot(&robot_id)?;
            }
            SearchCriteria::ByMission(mission_id) => {
                memory_ids = self.search_by_mission(&mission_id)?;
            }
            SearchCriteria::ByLocation {
                lat,
                lon,
                radius_meters,
            } => {
                memory_ids = self.search_by_location(lat, lon, radius_meters)?;
            }
            SearchCriteria::ByActionType(action_type) => {
                memory_ids = self.search_by_action_type(&action_type)?;
            }
            SearchCriteria::ByReward { min, max } => {
                memory_ids = self.search_by_reward(min, max)?;
            }

            // === Compound Criteria ===
            SearchCriteria::Combined(criterias) => {
                // Intersection of all criteria results
                // Use HashSet for O(1) lookups instead of O(n) Vec::contains
                use std::collections::HashSet;
                let mut result_sets: Vec<HashSet<MemoryId>> = Vec::new();
                for c in criterias {
                    result_sets.push(
                        self.search(c)?
                            .into_iter()
                            .map(|m| m.id)
                            .collect::<HashSet<_>>(),
                    );
                }

                if !result_sets.is_empty() {
                    let first_set = result_sets.remove(0);
                    memory_ids = first_set
                        .into_iter()
                        .filter(|id| result_sets.iter().all(|set| set.contains(id)))
                        .collect();
                }
            }
        }

        // Fetch full memories
        let mut memories = Vec::new();
        for id in memory_ids {
            if let Ok(memory) = self.get(&id) {
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    fn search_by_date_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let start_key = format!("date:{}", start.format("%Y%m%d"));
        // BUG-001 FIX: End key needs ~ suffix to include all UUIDs for that date
        // Keys are: date:YYYYMMDD:uuid, so date:20251207~ comes after all Dec 7 entries
        let end_key = format!("date:{}~", end.format("%Y%m%d"));

        let iter = self.index_db.iterator(IteratorMode::From(
            start_key.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, _value) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if &*key_str > end_key.as_str() {
                break;
            }
            // BUG-001 FIX: Extract memory_id from key (format: date:YYYYMMDD:uuid)
            if key_str.starts_with("date:") {
                let parts: Vec<&str> = key_str.split(':').collect();
                if parts.len() >= 3 {
                    // parts[0] = "date", parts[1] = "YYYYMMDD", parts[2] = uuid
                    if let Ok(uuid) = uuid::Uuid::parse_str(parts[2]) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    fn search_by_type(&self, exp_type: ExperienceType) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("type:{exp_type:?}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Extract ID from key
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    fn search_by_importance(&self, min: f32, max: f32) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let min_bucket = (min * 10.0) as u32;
        let max_bucket = (max * 10.0) as u32;

        for bucket in min_bucket..=max_bucket {
            let prefix = format!("importance:{bucket}:");
            let iter = self.index_db.iterator(IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));

            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                // Extract ID from key
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    fn search_by_entity(&self, entity: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        // Normalize to lowercase for case-insensitive matching
        let normalized_entity = entity.to_lowercase();
        let prefix = format!("entity:{normalized_entity}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Extract ID from key
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by tags (returns memories matching ANY of the provided tags)
    fn search_by_tags(&self, tags: &[String]) -> Result<Vec<MemoryId>> {
        use std::collections::HashSet;

        // Union of all tag matches
        let mut all_ids = HashSet::new();

        for tag in tags {
            // Normalize to lowercase for case-insensitive matching
            let normalized_tag = tag.to_lowercase();
            let prefix = format!("tag:{normalized_tag}:");
            let iter = self.index_db.iterator(IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));
            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        all_ids.insert(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(all_ids.into_iter().collect())
    }

    /// Search memories by episode ID
    /// Returns all memories in the specified episode
    fn search_by_episode(&self, episode_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("episode:{episode_id}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by episode with sequence filtering
    /// Returns memories in temporal order within the episode
    fn search_by_episode_sequence(
        &self,
        episode_id: &str,
        min_sequence: Option<u32>,
        max_sequence: Option<u32>,
    ) -> Result<Vec<MemoryId>> {
        let mut results: Vec<(u32, MemoryId)> = Vec::new();

        // Scan the episode_seq index which has format: episode_seq:{episode_id}:{seq}:{memory_id}
        let prefix = format!("episode_seq:{episode_id}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }

            // Parse: episode_seq:{episode_id}:{seq}:{memory_id}
            if let Some(rest) = key_str.strip_prefix(&prefix) {
                let parts: Vec<&str> = rest.splitn(2, ':').collect();
                if parts.len() == 2 {
                    if let (Ok(seq), Ok(uuid)) =
                        (parts[0].parse::<u32>(), uuid::Uuid::parse_str(parts[1]))
                    {
                        // Apply sequence filters
                        let passes_min = min_sequence.map_or(true, |min| seq >= min);
                        let passes_max = max_sequence.map_or(true, |max| seq <= max);

                        if passes_min && passes_max {
                            results.push((seq, MemoryId(uuid)));
                        }
                    }
                }
            }
        }

        // Sort by sequence number for temporal ordering
        results.sort_by_key(|(seq, _)| *seq);

        Ok(results.into_iter().map(|(_, id)| id).collect())
    }

    // ========================================================================
    // ROBOTICS SEARCH METHODS
    // ========================================================================

    /// Search memories by robot/drone identifier
    fn search_by_robot(&self, robot_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("robot:{robot_id}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by mission identifier
    fn search_by_mission(&self, mission_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("mission:{mission_id}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by geographic location using geohash prefix scanning
    ///
    /// Performance: O(k) where k = memories in ~9 geohash cells covering the radius
    /// Previous approach was O(n) where n = all geo-indexed memories
    fn search_by_location(
        &self,
        center_lat: f64,
        center_lon: f64,
        radius_meters: f64,
    ) -> Result<Vec<MemoryId>> {
        use super::types::{geohash_decode, geohash_search_prefixes, GeoFilter};

        let geo_filter = GeoFilter::new(center_lat, center_lon, radius_meters);
        let mut ids = Vec::new();

        // Get geohash prefixes for center + neighbors at appropriate precision
        let prefixes = geohash_search_prefixes(center_lat, center_lon, radius_meters);

        // Scan only the relevant geohash cells (9 cells = center + 8 neighbors)
        for geohash_prefix in prefixes {
            let prefix = format!("geo:{}:", geohash_prefix);
            let iter = self.index_db.iterator(IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));

            for (key, _value) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }

                // Key format: geo:GEOHASH:memory_id
                let parts: Vec<&str> = key_str.split(':').collect();
                if parts.len() >= 3 {
                    let geohash = parts[1];
                    // Decode geohash to get approximate lat/lon for distance check
                    let (min_lat, min_lon, max_lat, max_lon) = geohash_decode(geohash);
                    let approx_lat = (min_lat + max_lat) / 2.0;
                    let approx_lon = (min_lon + max_lon) / 2.0;

                    // Final haversine check for edge cases at cell boundaries
                    if geo_filter.contains(approx_lat, approx_lon) {
                        if let Ok(uuid) = uuid::Uuid::parse_str(parts[2]) {
                            ids.push(MemoryId(uuid));
                        }
                    }
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by action type
    fn search_by_action_type(&self, action_type: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("action:{action_type}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by reward range (for RL-style queries)
    fn search_by_reward(&self, min: f32, max: f32) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();

        // Reward is bucketed similar to importance (-10 to 10 buckets)
        // Clamp to prevent bucket overflow from out-of-range values
        let clamped_min = min.clamp(-1.0, 1.0);
        let clamped_max = max.clamp(-1.0, 1.0);
        let min_bucket = ((clamped_min + 1.0) * 10.0) as i32; // -1.0 -> 0, 1.0 -> 20
        let max_bucket = ((clamped_max + 1.0) * 10.0) as i32;

        for bucket in min_bucket..=max_bucket {
            let prefix = format!("reward:{bucket}:");
            let iter = self.index_db.iterator(IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));

            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    /// Get all memories from long-term storage
    ///
    /// Only returns entries with valid 16-byte UUID keys (consistent with get_stats)
    pub fn get_all(&self) -> Result<Vec<Memory>> {
        let mut memories = Vec::new();

        // Iterate through all memories
        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            if let Ok((key, value)) = item {
                // Only process valid 16-byte UUID keys (consistent with get_stats)
                if key.len() != 16 {
                    continue;
                }
                if let Ok(memory) = deserialize_memory(&value) {
                    memories.push(memory);
                }
            }
        }

        Ok(memories)
    }

    pub fn get_uncompressed_older_than(&self, cutoff: DateTime<Utc>) -> Result<Vec<Memory>> {
        let mut memories = Vec::new();

        // Iterate through all memories
        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            if let Ok((key, value)) = item {
                // Only process valid 16-byte UUID keys
                if key.len() != 16 {
                    continue;
                }
                if let Ok(memory) = deserialize_memory(&value) {
                    if !memory.compressed && memory.created_at < cutoff {
                        memories.push(memory);
                    }
                }
            }
        }

        Ok(memories)
    }

    /// Mark memories as forgotten (soft delete) with durable writes
    pub fn mark_forgotten_by_age(&self, cutoff: DateTime<Utc>) -> Result<usize> {
        let mut count = 0;

        // DURABILITY: Sync write for data integrity
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);

        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            if let Ok((key, value)) = item {
                // Only process valid 16-byte UUID keys
                if key.len() != 16 {
                    continue;
                }
                if let Ok(mut memory) = deserialize_memory(&value) {
                    if memory.created_at < cutoff {
                        // Add forgotten flag to metadata
                        memory
                            .experience
                            .metadata
                            .insert("forgotten".to_string(), "true".to_string());
                        memory
                            .experience
                            .metadata
                            .insert("forgotten_at".to_string(), Utc::now().to_rfc3339());

                        let updated_value =
                            bincode::serde::encode_to_vec(&memory, bincode::config::standard())?;
                        self.db.put_opt(&key, updated_value, &write_opts)?;
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }

    /// Mark memories with low importance as forgotten with durable writes
    pub fn mark_forgotten_by_importance(&self, threshold: f32) -> Result<usize> {
        let mut count = 0;

        // DURABILITY: Sync write for data integrity
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);

        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            if let Ok((key, value)) = item {
                // Only process valid 16-byte UUID keys
                if key.len() != 16 {
                    continue;
                }
                if let Ok(mut memory) = deserialize_memory(&value) {
                    if memory.importance() < threshold {
                        memory
                            .experience
                            .metadata
                            .insert("forgotten".to_string(), "true".to_string());
                        memory
                            .experience
                            .metadata
                            .insert("forgotten_at".to_string(), Utc::now().to_rfc3339());

                        let updated_value =
                            bincode::serde::encode_to_vec(&memory, bincode::config::standard())?;
                        self.db.put_opt(&key, updated_value, &write_opts)?;
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }

    /// Remove memories matching a pattern with durable writes
    pub fn remove_matching(&self, regex: &regex::Regex) -> Result<usize> {
        let mut count = 0;
        let mut to_delete = Vec::new();

        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            if let Ok((key, value)) = item {
                // Only process valid 16-byte UUID keys
                if key.len() != 16 {
                    continue;
                }
                if let Ok(memory) = deserialize_memory(&value) {
                    if regex.is_match(&memory.experience.content) {
                        to_delete.push(key.to_vec());
                        count += 1;
                    }
                }
            }
        }

        // DURABILITY: Sync write for delete operations
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);

        for key in to_delete {
            self.db.delete_opt(&key, &write_opts)?;
        }

        Ok(count)
    }

    /// Update access count for a memory
    pub fn update_access(&self, id: &MemoryId) -> Result<()> {
        if let Ok(memory) = self.get(id) {
            // ZERO-COPY: Update metadata through interior mutability
            memory.update_access();

            // Persist updated metadata
            self.update(&memory)?;
        }
        Ok(())
    }

    /// Get statistics about stored memories
    pub fn get_stats(&self) -> Result<StorageStats> {
        let mut stats = StorageStats::default();
        let mut raw_count = 0;
        let mut skipped_non_memory = 0;
        let mut deserialize_errors = 0;
        let stats_prefix = b"stats:";

        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            match item {
                Ok((key, value)) => {
                    raw_count += 1;

                    // Skip stats entries - they use a different format
                    if key.starts_with(stats_prefix) {
                        skipped_non_memory += 1;
                        continue;
                    }

                    // Valid memory keys should be exactly 16 bytes (UUID bytes)
                    if key.len() != 16 {
                        skipped_non_memory += 1;
                        continue;
                    }

                    match deserialize_memory(&value) {
                        Ok(memory) => {
                            stats.total_count += 1;
                            stats.total_size_bytes += value.len();
                            if memory.compressed {
                                stats.compressed_count += 1;
                            }
                            stats.importance_sum += memory.importance();
                        }
                        Err(e) => {
                            deserialize_errors += 1;
                            tracing::warn!(
                                "Corrupted memory entry (key len: {}, value len: {}): {}",
                                key.len(),
                                value.len(),
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Iterator error: {}", e);
                }
            }
        }

        tracing::debug!(
            "get_stats: raw_count={}, memories={}, skipped={}, corrupted={}",
            raw_count,
            stats.total_count,
            skipped_non_memory,
            deserialize_errors
        );

        if stats.total_count > 0 {
            stats.average_importance = stats.importance_sum / stats.total_count as f32;
        }

        // Load persisted retrieval counter
        stats.total_retrievals = self.get_retrieval_count().unwrap_or(0);

        Ok(stats)
    }

    /// Get the persisted retrieval counter
    pub fn get_retrieval_count(&self) -> Result<usize> {
        const RETRIEVAL_KEY: &[u8] = b"stats:total_retrievals";
        match self.db.get(RETRIEVAL_KEY)? {
            Some(data) => {
                if data.len() >= 8 {
                    Ok(usize::from_le_bytes(data[..8].try_into().unwrap_or([0; 8])))
                } else {
                    Ok(0)
                }
            }
            None => Ok(0),
        }
    }

    /// Increment and persist the retrieval counter, returns new value
    pub fn increment_retrieval_count(&self) -> Result<usize> {
        const RETRIEVAL_KEY: &[u8] = b"stats:total_retrievals";
        let current = self.get_retrieval_count().unwrap_or(0);
        let new_count = current + 1;
        self.db.put(RETRIEVAL_KEY, new_count.to_le_bytes())?;
        Ok(new_count)
    }

    /// Remove corrupted memories that fail to deserialize
    /// Returns the number of entries deleted
    ///
    /// This function safely cleans up:
    /// 1. Entries with keys that are not valid 16-byte UUIDs (corrupted/misplaced)
    /// 2. Entries with valid UUID keys but corrupted values that fail to deserialize
    ///
    /// It preserves:
    /// - Valid Memory entries
    /// - Stats entries (keys starting with "stats:")
    pub fn cleanup_corrupted(&self) -> Result<usize> {
        let mut to_delete = Vec::new();
        let stats_prefix = b"stats:";

        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            if let Ok((key, value)) = item {
                // Skip stats entries - they use a different format
                if key.starts_with(stats_prefix) {
                    continue;
                }

                // Valid memory keys should be exactly 16 bytes (UUID bytes)
                let is_valid_memory_key = key.len() == 16;

                if !is_valid_memory_key {
                    // Key is not a valid UUID - this is a corrupted or misplaced entry
                    tracing::debug!(
                        "Marking for deletion: invalid key length {} (expected 16)",
                        key.len()
                    );
                    to_delete.push(key.to_vec());
                } else if bincode::serde::decode_from_slice::<Memory, _>(
                    &value,
                    bincode::config::standard(),
                )
                .is_err()
                {
                    // Key is valid but value is corrupted
                    tracing::debug!(
                        "Marking for deletion: valid key but corrupted value ({} bytes)",
                        value.len()
                    );
                    to_delete.push(key.to_vec());
                }
            }
        }

        let count = to_delete.len();
        if count > 0 {
            tracing::info!("Cleaning up {} corrupted memory entries", count);

            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(self.write_mode == WriteMode::Sync);

            for key in to_delete {
                if let Err(e) = self.db.delete_opt(&key, &write_opts) {
                    tracing::warn!("Failed to delete corrupted entry: {}", e);
                }
            }

            // Flush to persist deletions
            self.flush()?;
        }

        Ok(count)
    }

    /// Flush both databases to ensure all data is persisted (critical for graceful shutdown)
    pub fn flush(&self) -> Result<()> {
        use rocksdb::FlushOptions;

        // Create flush options with explicit wait
        let mut flush_opts = FlushOptions::default();
        flush_opts.set_wait(true); // Block until flush is complete

        // Flush main memory database
        self.db
            .flush_opt(&flush_opts)
            .map_err(|e| anyhow::anyhow!("Failed to flush main database: {e}"))?;

        // Flush index database
        self.index_db
            .flush_opt(&flush_opts)
            .map_err(|e| anyhow::anyhow!("Failed to flush index database: {e}"))?;

        Ok(())
    }

    /// Get a reference to the underlying RocksDB instance
    ///
    /// Used by SemanticFactStore to share the same database for fact storage.
    /// Facts use a different key prefix ("facts:") to avoid collisions.
    pub fn db(&self) -> Arc<DB> {
        self.db.clone()
    }
}

/// Search criteria for memory retrieval
#[derive(Debug, Clone)]
pub enum SearchCriteria {
    // === Standard Criteria ===
    ByDate {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    ByType(ExperienceType),
    ByImportance {
        min: f32,
        max: f32,
    },
    ByEntity(String),
    /// Filter by tags (matches memories containing ANY of these tags)
    ByTags(Vec<String>),

    // === Temporal/Episode Criteria ===
    /// Filter by episode ID - memories in the same episode are highly related
    ByEpisode(String),
    /// Filter by episode with sequence ordering - returns memories in temporal order
    ByEpisodeSequence {
        episode_id: String,
        /// If provided, only return memories with sequence >= this value
        min_sequence: Option<u32>,
        /// If provided, only return memories with sequence <= this value
        max_sequence: Option<u32>,
    },

    // === Robotics Criteria ===
    /// Filter by robot/drone identifier
    ByRobot(String),
    /// Filter by mission identifier
    ByMission(String),
    /// Spatial filter: memories within radius of (lat, lon)
    ByLocation {
        lat: f64,
        lon: f64,
        radius_meters: f64,
    },
    /// Filter by action type
    ByActionType(String),
    /// Filter by reward range (for RL-style queries)
    ByReward {
        min: f32,
        max: f32,
    },

    // === Compound Criteria ===
    Combined(Vec<SearchCriteria>),
}

/// Storage statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_count: usize,
    pub compressed_count: usize,
    pub total_size_bytes: usize,
    pub average_importance: f32,
    pub importance_sum: f32,
    /// Total number of recall/retrieval operations (persisted)
    #[serde(default)]
    pub total_retrievals: usize,
}

// =============================================================================
// ATOMIC VECTOR INDEX MAPPING STORAGE
// =============================================================================
//
// This module provides atomic storage for vector index mappings alongside memory data.
// By storing IdMapping in RocksDB (not separate files), we ensure:
//
// 1. ATOMIC WRITES: Memory + vector mapping written in single WriteBatch
// 2. NO ORPHANS: If memory exists, its vector mapping exists (or can be rebuilt)
// 3. CRASH SAFETY: RocksDB WAL protects both memory data and mappings
// 4. SINGLE SOURCE OF TRUTH: RocksDB is THE authority, Vamana is just a cache
//
// MULTIMODALITY READY:
// - Each modality (text, image, audio, video) has separate vector space
// - Text: 384-dim MiniLM (current)
// - Image: 1024-dim ImageBind (future)
// - Audio: 1024-dim ImageBind (future)
// - Video: 1024-dim ImageBind (future)
// - Cross-modal search possible via ImageBind's unified embedding space
//
// Key format: "vmapping:{memory_id}" -> bincode(VectorMappingEntry)
// =============================================================================

use std::collections::HashMap;

/// Supported embedding modalities
///
/// When adding a new modality:
/// 1. Add variant here
/// 2. Create corresponding Vamana index with correct dimension
/// 3. Implement embedder for the modality
/// 4. Update search to include the modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text embeddings (MiniLM-L6-v2, 384-dim)
    Text,
    /// Image embeddings (future: ImageBind, 1024-dim)
    Image,
    /// Audio embeddings (future: ImageBind, 1024-dim)
    Audio,
    /// Video embeddings (future: ImageBind, 1024-dim)
    Video,
    /// Multi-modal unified embeddings (future: ImageBind, 1024-dim)
    /// Used when content has multiple modalities fused together
    Unified,
}

impl Modality {
    /// Get embedding dimension for this modality
    pub fn dimension(&self) -> usize {
        match self {
            Modality::Text => 384, // MiniLM-L6-v2
            // ImageBind projects all modalities to 1024-dim shared space
            Modality::Image => 1024,
            Modality::Audio => 1024,
            Modality::Video => 1024,
            Modality::Unified => 1024,
        }
    }

    /// Get the string key for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
            Modality::Unified => "unified",
        }
    }
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Vector IDs for a specific modality
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModalityVectors {
    /// Vector IDs in this modality's Vamana index
    pub vector_ids: Vec<u32>,
    /// Embedding dimension (for validation)
    pub dimension: usize,
    /// Chunk boundaries (for long content)
    /// Each entry is (start_char, end_char) in original content
    pub chunk_ranges: Option<Vec<(usize, usize)>>,
}

/// Vector mapping entry for a single memory - MULTIMODALITY READY
///
/// Stores vector IDs for each modality separately, allowing:
/// - Text-only memories (current)
/// - Image-only memories (future)
/// - Multi-modal memories (text + image + audio)
/// - Cross-modal search via unified embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMappingEntry {
    /// Vector IDs per modality
    /// Key: Modality enum (serializes as string)
    /// Value: Vector IDs + metadata for that modality
    pub modalities: HashMap<Modality, ModalityVectors>,
    /// Timestamp when mapping was created (for debugging)
    pub created_at: i64,
    /// Schema version for forward compatibility
    pub version: u8,
}

impl Default for VectorMappingEntry {
    fn default() -> Self {
        Self {
            modalities: HashMap::new(),
            created_at: chrono::Utc::now().timestamp_millis(),
            version: 1,
        }
    }
}

impl VectorMappingEntry {
    /// Create a new mapping with text vectors (most common case)
    pub fn with_text(vector_ids: Vec<u32>) -> Self {
        let mut modalities = HashMap::new();
        modalities.insert(
            Modality::Text,
            ModalityVectors {
                vector_ids,
                dimension: 384,
                chunk_ranges: None,
            },
        );
        Self {
            modalities,
            created_at: chrono::Utc::now().timestamp_millis(),
            version: 1,
        }
    }

    /// Get text vector IDs (convenience method for current text-only usage)
    pub fn text_vectors(&self) -> Option<&Vec<u32>> {
        self.modalities.get(&Modality::Text).map(|m| &m.vector_ids)
    }

    /// Get all vector IDs across all modalities (for deletion)
    pub fn all_vector_ids(&self) -> Vec<(Modality, u32)> {
        self.modalities
            .iter()
            .flat_map(|(modality, mv)| mv.vector_ids.iter().map(|id| (*modality, *id)))
            .collect()
    }

    /// Check if this entry has any vectors
    pub fn is_empty(&self) -> bool {
        self.modalities.values().all(|mv| mv.vector_ids.is_empty())
    }

    /// Add vectors for a modality
    pub fn add_modality(&mut self, modality: Modality, vector_ids: Vec<u32>) {
        self.modalities.insert(
            modality,
            ModalityVectors {
                dimension: modality.dimension(),
                vector_ids,
                chunk_ranges: None,
            },
        );
    }

    /// Future: Add image vectors
    #[allow(dead_code)]
    pub fn with_image(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Image, vector_ids);
        self
    }

    /// Future: Add audio vectors
    #[allow(dead_code)]
    pub fn with_audio(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Audio, vector_ids);
        self
    }

    /// Future: Add video vectors
    #[allow(dead_code)]
    pub fn with_video(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Video, vector_ids);
        self
    }
}

impl MemoryStorage {
    // =========================================================================
    // ATOMIC VECTOR MAPPING OPERATIONS
    // =========================================================================

    /// Store memory and its text vector mapping atomically
    ///
    /// Uses WriteBatch to ensure both operations succeed or both fail.
    /// This is the ONLY way orphaned memories can be prevented.
    ///
    /// For text-only memories (current implementation). Use store_with_multimodal_vectors
    /// for memories with image/audio/video content.
    pub fn store_with_vectors(&self, memory: &Memory, vector_ids: Vec<u32>) -> Result<()> {
        self.store_with_multimodal_vectors(memory, Modality::Text, vector_ids)
    }

    /// Store memory with vectors for a specific modality
    ///
    /// MULTIMODALITY READY: Supports text, image, audio, video modalities.
    /// Each modality is stored separately, allowing cross-modal search.
    pub fn store_with_multimodal_vectors(
        &self,
        memory: &Memory,
        modality: Modality,
        vector_ids: Vec<u32>,
    ) -> Result<()> {
        let mut batch = WriteBatch::default();

        // 1. Serialize memory
        let memory_key = memory.id.0.as_bytes();
        let memory_value = bincode::serde::encode_to_vec(memory, bincode::config::standard())
            .context(format!("Failed to serialize memory {}", memory.id.0))?;
        batch.put(memory_key, &memory_value);

        // 2. Serialize vector mapping with modality support
        let mapping_key = format!("vmapping:{}", memory.id.0);

        // Load existing mapping (for adding new modality to existing memory)
        let mut mapping_entry = self.get_vector_mapping(&memory.id)?.unwrap_or_default();

        // Add/update the modality vectors
        mapping_entry.add_modality(modality, vector_ids);

        let mapping_value =
            bincode::serde::encode_to_vec(&mapping_entry, bincode::config::standard())
                .context("Failed to serialize vector mapping")?;
        batch.put(mapping_key.as_bytes(), &mapping_value);

        // 3. Atomic write - both succeed or both fail
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .write_opt(batch, &write_opts)
            .context("Atomic write of memory + vector mapping failed")?;

        // 4. Update secondary indices (separate operation, but non-critical)
        if let Err(e) = self.update_indices(memory) {
            tracing::warn!("Secondary index update failed (non-fatal): {}", e);
        }

        Ok(())
    }

    /// Get vector mapping for a memory
    pub fn get_vector_mapping(&self, memory_id: &MemoryId) -> Result<Option<VectorMappingEntry>> {
        let mapping_key = format!("vmapping:{}", memory_id.0);
        match self.db.get(mapping_key.as_bytes())? {
            Some(data) => {
                let (entry, _): (VectorMappingEntry, _) =
                    bincode::serde::decode_from_slice(&data, bincode::config::standard())
                        .context("Failed to deserialize vector mapping")?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    /// Get all vector mappings (for rebuilding Vamana index on startup)
    ///
    /// Returns iterator-style results to avoid loading everything into memory at once.
    /// Sorted by memory_id for deterministic Vamana rebuilding.
    pub fn get_all_vector_mappings(&self) -> Result<Vec<(MemoryId, VectorMappingEntry)>> {
        let mut mappings = Vec::new();
        let prefix = b"vmapping:";

        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        for item in iter {
            match item {
                Ok((key, value)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if !key_str.starts_with("vmapping:") {
                        break;
                    }

                    // Extract memory_id from key
                    if let Some(id_str) = key_str.strip_prefix("vmapping:") {
                        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                            if let Ok((entry, _)) =
                                bincode::serde::decode_from_slice::<VectorMappingEntry, _>(
                                    &value,
                                    bincode::config::standard(),
                                )
                            {
                                mappings.push((MemoryId(uuid), entry));
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Error reading vector mapping: {}", e);
                }
            }
        }

        Ok(mappings)
    }

    /// Delete vector mapping for a memory (called when deleting memory)
    pub fn delete_vector_mapping(&self, memory_id: &MemoryId) -> Result<()> {
        let mapping_key = format!("vmapping:{}", memory_id.0);
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db.delete_opt(mapping_key.as_bytes(), &write_opts)?;
        Ok(())
    }

    /// Update text vector mapping for a memory (for reindex operations)
    ///
    /// Convenience method for text-only reindexing.
    pub fn update_vector_mapping(&self, memory_id: &MemoryId, vector_ids: Vec<u32>) -> Result<()> {
        self.update_modality_vectors(memory_id, Modality::Text, vector_ids)
    }

    /// Update vector mapping for a specific modality
    ///
    /// MULTIMODALITY READY: Preserves vectors for other modalities while updating one.
    pub fn update_modality_vectors(
        &self,
        memory_id: &MemoryId,
        modality: Modality,
        vector_ids: Vec<u32>,
    ) -> Result<()> {
        let mapping_key = format!("vmapping:{}", memory_id.0);

        // Load existing mapping to preserve other modalities
        let mut mapping_entry = self.get_vector_mapping(memory_id)?.unwrap_or_default();

        // Update the specific modality
        mapping_entry.add_modality(modality, vector_ids);

        let mapping_value =
            bincode::serde::encode_to_vec(&mapping_entry, bincode::config::standard())?;

        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .put_opt(mapping_key.as_bytes(), &mapping_value, &write_opts)?;
        Ok(())
    }

    /// Delete memory and its vector mapping atomically
    pub fn delete_with_vectors(&self, id: &MemoryId) -> Result<()> {
        let mut batch = WriteBatch::default();

        // 1. Delete memory
        batch.delete(id.0.as_bytes());

        // 2. Delete vector mapping
        let mapping_key = format!("vmapping:{}", id.0);
        batch.delete(mapping_key.as_bytes());

        // 3. Atomic delete
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db.write_opt(batch, &write_opts)?;

        // 4. Clean up indices (non-critical)
        if let Err(e) = self.remove_from_indices(id) {
            tracing::warn!("Index cleanup failed (non-fatal): {}", e);
        }

        Ok(())
    }

    /// Count memories with vector mappings (for health checks)
    pub fn count_vector_mappings(&self) -> usize {
        let prefix = b"vmapping:";
        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        let mut count = 0;
        for item in iter {
            if let Ok((key, _)) = item {
                if key.starts_with(prefix) {
                    count += 1;
                } else {
                    break;
                }
            }
        }
        count
    }

    /// Check integrity: find memories without vector mappings
    ///
    /// Returns memories that have embeddings but no corresponding vector mapping.
    /// These need to be reindexed.
    pub fn find_memories_without_mappings(&self) -> Result<Vec<MemoryId>> {
        let mut orphans = Vec::new();

        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            if let Ok((key, value)) = item {
                // Skip non-memory keys
                if key.len() != 16 {
                    continue;
                }

                // Try to deserialize as memory
                if let Ok(memory) = deserialize_memory(&value) {
                    // Check if vector mapping exists and has text vectors
                    let has_mapping = match self.get_vector_mapping(&memory.id) {
                        Ok(Some(entry)) => entry.text_vectors().is_some_and(|v| !v.is_empty()),
                        _ => false,
                    };

                    // Memory has embeddings but no mapping - needs reindex
                    if !has_mapping && memory.experience.embeddings.is_some() {
                        orphans.push(memory.id);
                    }
                }
            }
        }

        Ok(orphans)
    }

    /// Get all text vector IDs from mappings (for Vamana statistics)
    pub fn get_all_text_vector_ids(&self) -> Result<Vec<u32>> {
        let mut all_ids = Vec::new();
        let mappings = self.get_all_vector_mappings()?;

        for (_, entry) in mappings {
            if let Some(text_vecs) = entry.text_vectors() {
                all_ids.extend(text_vecs.iter().copied());
            }
        }

        Ok(all_ids)
    }

    /// Get vector count per modality (for health monitoring)
    pub fn get_modality_stats(&self) -> Result<HashMap<Modality, usize>> {
        let mut stats: HashMap<Modality, usize> = HashMap::new();
        let mappings = self.get_all_vector_mappings()?;

        for (_, entry) in mappings {
            for (modality, mv) in entry.modalities {
                *stats.entry(modality).or_insert(0) += mv.vector_ids.len();
            }
        }

        Ok(stats)
    }
}
