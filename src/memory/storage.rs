//! Storage backend for the memory system

use anyhow::{anyhow, Context, Result};
use bincode;
use chrono::{DateTime, Utc};
use rocksdb::{IteratorMode, Options, WriteBatch, WriteOptions, DB};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::types::*;

/// Storage engine for long-term memory persistence
pub struct MemoryStorage {
    db: Arc<DB>,
    index_db: Arc<DB>, // Secondary indices
    /// Base storage path for all memory data
    storage_path: PathBuf,
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

        Ok(Self {
            db,
            index_db,
            storage_path: path.to_path_buf(),
        })
    }

    /// Get the base storage path
    pub fn path(&self) -> &Path {
        &self.storage_path
    }

    /// Store a memory with durable write (WAL sync)
    ///
    /// PRODUCTION: Uses sync writes to ensure data survives crashes/restarts.
    /// The sync flag causes RocksDB to fsync() the WAL before returning,
    /// guaranteeing the write is on stable storage.
    pub fn store(&self, memory: &Memory) -> Result<()> {
        let key = memory.id.0.as_bytes();

        // Serialize memory
        let value = bincode::serialize(memory)
            .context(format!("Failed to serialize memory {}", memory.id.0))?;

        // DURABILITY: Use sync writes to ensure data persists across restarts
        // This is the fundamental fix - without sync, data stays in OS page cache
        // and can be lost if process exits before fsync
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true); // fsync() WAL before returning

        // Store in main database with sync
        self.db
            .put_opt(key, &value, &write_opts)
            .context(format!("Failed to put memory {} in RocksDB", memory.id.0))?;

        // Update indices (also with sync for consistency)
        self.update_indices(memory)?;

        Ok(())
    }

    /// Update secondary indices for efficient retrieval
    fn update_indices(&self, memory: &Memory) -> Result<()> {
        let mut batch = WriteBatch::default();
        let memory_id_str = memory.id.0.to_string();

        // === Standard Indices ===

        // Index by date (for temporal queries)
        let date_key = format!("date:{}", memory.created_at.format("%Y%m%d"));
        batch.put(date_key.as_bytes(), memory_id_str.as_bytes());

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

        // Index by entities
        for entity in &memory.experience.entities {
            let entity_key = format!("entity:{}:{}", entity, memory.id.0);
            batch.put(entity_key.as_bytes(), b"1");
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

        // Index by geo_location (for spatial queries)
        // Key format: geo:lat:lon:memory_id, Value: memory_id
        if let Some(geo) = memory.experience.geo_location {
            let lat = geo[0];
            let lon = geo[1];
            // Round to 4 decimal places (~11m precision) for indexing
            let lat_str = format!("{lat:.4}");
            let lon_str = format!("{lon:.4}");
            let geo_key = format!("geo:{}:{}:{}", lat_str, lon_str, memory.id.0);
            batch.put(geo_key.as_bytes(), memory_id_str.as_bytes());
        }

        // Index by action_type (for action-based retrieval)
        if let Some(ref action_type) = memory.experience.action_type {
            let action_key = format!("action:{}:{}", action_type, memory.id.0);
            batch.put(action_key.as_bytes(), b"1");
        }

        // Index by reward (bucketed, for RL-style queries)
        // Bucket: -1.0 to 1.0 mapped to 0-20
        if let Some(reward) = memory.experience.reward {
            let reward_bucket = ((reward + 1.0) * 10.0) as i32;
            let reward_key = format!("reward:{}:{}", reward_bucket, memory.id.0);
            batch.put(reward_key.as_bytes(), b"1");
        }

        // DURABILITY: Sync write for index consistency
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        self.index_db.write_opt(batch, &write_opts)?;
        Ok(())
    }

    /// Retrieve a memory by ID
    pub fn get(&self, id: &MemoryId) -> Result<Memory> {
        let key = id.0.as_bytes();
        match self.db.get(key)? {
            Some(value) => bincode::deserialize::<Memory>(&value).with_context(|| {
                format!(
                    "Failed to deserialize memory {} ({} bytes)",
                    id.0,
                    value.len()
                )
            }),
            None => Err(anyhow!("Memory not found: {id:?}")),
        }
    }

    /// Update an existing memory
    pub fn update(&self, memory: &Memory) -> Result<()> {
        self.store(memory)
    }

    /// Delete a memory with durable write
    #[allow(unused)] // Public API - available for memory management
    pub fn delete(&self, id: &MemoryId) -> Result<()> {
        let key = id.0.as_bytes();

        // DURABILITY: Sync write for delete operations
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        self.db.delete_opt(key, &write_opts)?;

        // Clean up indices
        self.remove_from_indices(id)?;

        Ok(())
    }

    /// Remove memory from all indices
    fn remove_from_indices(&self, id: &MemoryId) -> Result<()> {
        let mut batch = WriteBatch::default();

        // Iterate through all index entries and remove those matching this ID
        let prefix_patterns = vec![
            format!("date:"),
            format!("type:"),
            format!("importance:"),
            format!("entity:"),
        ];

        for prefix in prefix_patterns {
            let iter = self.index_db.iterator(IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));
            for (key, _) in iter.flatten() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if key_str.contains(&id.0.to_string()) {
                    batch.delete(&key);
                }
            }
        }

        // DURABILITY: Sync write for index consistency
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
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
                let mut result_sets: Vec<Vec<MemoryId>> = Vec::new();
                for c in criterias {
                    result_sets.push(self.search(c)?.into_iter().map(|m| m.id).collect());
                }

                if !result_sets.is_empty() {
                    memory_ids = result_sets[0].clone();
                    for set in result_sets.iter().skip(1) {
                        memory_ids.retain(|id| set.contains(id));
                    }
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
        let end_key = format!("date:{}", end.format("%Y%m%d"));

        let iter = self.index_db.iterator(IteratorMode::From(
            start_key.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, value) in iter.flatten() {
            let key_str = String::from_utf8_lossy(&key);
            if key_str.as_ref() > end_key.as_str() {
                break;
            }
            if key_str.starts_with("date:") {
                let id_str = String::from_utf8_lossy(&value);
                if let Ok(uuid) = uuid::Uuid::parse_str(&id_str) {
                    ids.push(MemoryId(uuid));
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
        for (key, _) in iter.flatten() {
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

            for (key, _) in iter.flatten() {
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
        let prefix = format!("entity:{entity}:");

        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        for (key, _) in iter.flatten() {
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
        for (key, _) in iter.flatten() {
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
        for (key, _) in iter.flatten() {
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

    /// Search memories by geographic location (haversine distance)
    fn search_by_location(
        &self,
        center_lat: f64,
        center_lon: f64,
        radius_meters: f64,
    ) -> Result<Vec<MemoryId>> {
        use super::types::GeoFilter;

        let geo_filter = GeoFilter::new(center_lat, center_lon, radius_meters);
        let mut ids = Vec::new();

        // Scan all memories with geo_location (prefix scan on geo index)
        let prefix = "geo:";
        let iter = self.index_db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for (key, value) in iter.flatten() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(prefix) {
                break;
            }

            // Key format: geo:lat:lon:memory_id
            // Value contains the memory ID
            let parts: Vec<&str> = key_str.split(':').collect();
            if parts.len() >= 4 {
                if let (Ok(lat), Ok(lon)) = (parts[1].parse::<f64>(), parts[2].parse::<f64>()) {
                    if geo_filter.contains(lat, lon) {
                        let id_str = String::from_utf8_lossy(&value);
                        if let Ok(uuid) = uuid::Uuid::parse_str(&id_str) {
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
        for (key, _) in iter.flatten() {
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
        let min_bucket = ((min + 1.0) * 10.0) as i32; // -1.0 -> 0, 1.0 -> 20
        let max_bucket = ((max + 1.0) * 10.0) as i32;

        for bucket in min_bucket..=max_bucket {
            let prefix = format!("reward:{bucket}:");
            let iter = self.index_db.iterator(IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));

            for (key, _) in iter.flatten() {
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
    pub fn get_all(&self) -> Result<Vec<Memory>> {
        let mut memories = Vec::new();

        // Iterate through all memories
        let iter = self.db.iterator(IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok(memory) = bincode::deserialize::<Memory>(&value) {
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    pub fn get_uncompressed_older_than(&self, cutoff: DateTime<Utc>) -> Result<Vec<Memory>> {
        let mut memories = Vec::new();

        // Iterate through all memories
        let iter = self.db.iterator(IteratorMode::Start);
        for (_, value) in iter.flatten() {
            if let Ok(memory) = bincode::deserialize::<Memory>(&value) {
                if !memory.compressed && memory.created_at < cutoff {
                    memories.push(memory);
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
        for (key, value) in iter.flatten() {
            if let Ok(mut memory) = bincode::deserialize::<Memory>(&value) {
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

                    let updated_value = bincode::serialize(&memory)?;
                    self.db.put_opt(&key, updated_value, &write_opts)?;
                    count += 1;
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
        for (key, value) in iter.flatten() {
            if let Ok(mut memory) = bincode::deserialize::<Memory>(&value) {
                if memory.importance() < threshold {
                    memory
                        .experience
                        .metadata
                        .insert("forgotten".to_string(), "true".to_string());
                    memory
                        .experience
                        .metadata
                        .insert("forgotten_at".to_string(), Utc::now().to_rfc3339());

                    let updated_value = bincode::serialize(&memory)?;
                    self.db.put_opt(&key, updated_value, &write_opts)?;
                    count += 1;
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
        for (key, value) in iter.flatten() {
            if let Ok(memory) = bincode::deserialize::<Memory>(&value) {
                if regex.is_match(&memory.experience.content) {
                    to_delete.push(key.to_vec());
                    count += 1;
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
        let mut deserialize_errors = 0;

        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            match item {
                Ok((key, value)) => {
                    raw_count += 1;
                    match bincode::deserialize::<Memory>(&value) {
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
                                "Failed to deserialize memory (key len: {}, value len: {}): {}",
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
            "get_stats: raw_count={}, deserialized={}, errors={}",
            raw_count,
            stats.total_count,
            deserialize_errors
        );

        if stats.total_count > 0 {
            stats.average_importance = stats.importance_sum / stats.total_count as f32;
        }

        Ok(stats)
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
}
