//! P2: Backup & Restore System
//!
//! Provides production-grade backup and restore capabilities:
//! - Incremental backups using RocksDB checkpoints
//! - Point-in-time recovery (PITR)
//! - Export to JSON/Parquet formats
//! - Backup verification and integrity checks
//! - Automated scheduling support

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rocksdb::{
    backup::{BackupEngine, BackupEngineOptions},
    checkpoint::Checkpoint,
    Env, DB,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Backup metadata for tracking and verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Unique backup ID
    pub backup_id: u32,
    /// Timestamp when backup was created
    pub created_at: DateTime<Utc>,
    /// User ID (if single-user backup) or "all" for full backup
    pub user_id: String,
    /// Backup type: "full" or "incremental"
    pub backup_type: BackupType,
    /// Size in bytes (compressed)
    pub size_bytes: u64,
    /// SHA-256 checksum for integrity verification
    pub checksum: String,
    /// Number of memories included in backup
    pub memory_count: usize,
    /// RocksDB sequence number (for PITR)
    pub sequence_number: u64,
    /// Secondary stores included in this backup
    #[serde(default)]
    pub secondary_stores: Vec<String>,
    /// Total size of secondary store backups in bytes
    #[serde(default)]
    pub secondary_size_bytes: u64,
}

/// Named reference to a RocksDB database for backup
pub struct SecondaryStoreRef<'a> {
    pub name: &'a str,
    pub db: &'a Arc<DB>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackupType {
    Full,
    Incremental,
}

/// Backup engine for creating and managing backups
pub struct ShodhBackupEngine {
    backup_path: PathBuf,
}

impl ShodhBackupEngine {
    /// Create a new backup engine
    ///
    /// # Arguments
    /// * `backup_path` - Directory to store backups
    pub fn new(backup_path: PathBuf) -> Result<Self> {
        fs::create_dir_all(&backup_path)?;
        Ok(Self { backup_path })
    }

    /// Get the backup storage path.
    pub fn backup_path(&self) -> &Path {
        &self.backup_path
    }

    /// Create a full backup of a RocksDB database
    ///
    /// # Arguments
    /// * `db` - Reference to the RocksDB database
    /// * `user_id` - User ID for the backup (or "all" for full system backup)
    ///
    /// # Returns
    /// BackupMetadata with backup details
    pub fn create_backup(&self, db: &DB, user_id: &str) -> Result<BackupMetadata> {
        let backup_dir = self.backup_path.join(user_id);
        fs::create_dir_all(&backup_dir)?;

        // Create RocksDB backup engine
        let backup_opts = BackupEngineOptions::new(&backup_dir)?;
        let env = Env::new()?;
        let mut backup_engine = BackupEngine::open(&backup_opts, &env)?;

        // Create backup
        let before_count = backup_engine.get_backup_info().len();
        backup_engine.create_new_backup(db)?;

        let backup_info = backup_engine.get_backup_info();
        let latest_backup = backup_info
            .last()
            .ok_or_else(|| anyhow!("No backup created"))?;

        let backup_id = latest_backup.backup_id;
        let size_bytes = latest_backup.size;

        // Get latest sequence number from DB
        let sequence_number = db.latest_sequence_number();

        // Count memories (estimate from DB size)
        let memory_count = self.estimate_memory_count(db)?;

        // Calculate checksum of backup directory
        let checksum = self.calculate_backup_checksum(&backup_dir, backup_id)?;

        // Determine backup type
        let backup_type = if before_count == 0 {
            BackupType::Full
        } else {
            BackupType::Incremental
        };

        let metadata = BackupMetadata {
            backup_id,
            created_at: Utc::now(),
            user_id: user_id.to_string(),
            backup_type,
            size_bytes,
            checksum,
            memory_count,
            sequence_number,
            secondary_stores: Vec::new(),
            secondary_size_bytes: 0,
        };

        // Save metadata
        self.save_metadata(&metadata)?;

        tracing::info!(
            backup_id = backup_id,
            user_id = user_id,
            size_mb = size_bytes / 1024 / 1024,
            "Backup created successfully"
        );

        Ok(metadata)
    }

    /// Create a comprehensive backup of the main database, secondary stores, and graph.
    ///
    /// Uses RocksDB BackupEngine for the main memories DB and Checkpoint API
    /// for secondary stores (todos, reminders, facts, files, feedback, audit)
    /// and the knowledge graph database.
    pub fn create_comprehensive_backup(
        &self,
        db: &DB,
        user_id: &str,
        secondary_stores: &[SecondaryStoreRef<'_>],
    ) -> Result<BackupMetadata> {
        self.create_comprehensive_backup_with_graph(db, user_id, secondary_stores, None)
    }

    /// Create a comprehensive backup including the knowledge graph DB.
    pub fn create_comprehensive_backup_with_graph(
        &self,
        db: &DB,
        user_id: &str,
        secondary_stores: &[SecondaryStoreRef<'_>],
        graph_db: Option<&DB>,
    ) -> Result<BackupMetadata> {
        // Step 1: Create main memories backup (existing logic)
        let mut metadata = self.create_backup(db, user_id)?;

        // Step 2: Checkpoint each secondary store alongside the backup
        let secondary_dir = self
            .backup_path
            .join(user_id)
            .join(format!("secondary_{}", metadata.backup_id));
        fs::create_dir_all(&secondary_dir)?;

        // Step 2a: Checkpoint graph DB if provided
        if let Some(graph) = graph_db {
            let graph_checkpoint_dir = secondary_dir.join("graph");
            let checkpoint = Checkpoint::new(graph)
                .map_err(|e| anyhow!("Failed to create checkpoint handle for graph DB: {}", e))?;
            checkpoint
                .create_checkpoint(&graph_checkpoint_dir)
                .map_err(|e| {
                    let _ = fs::remove_dir_all(&graph_checkpoint_dir);
                    anyhow!("Failed to checkpoint graph DB: {}", e)
                })?;
            let graph_size = dir_size(&graph_checkpoint_dir).unwrap_or(0);
            tracing::debug!(size_kb = graph_size / 1024, "Graph DB checkpointed");
        }

        let mut backed_up_stores = Vec::new();
        let mut total_secondary_bytes: u64 = 0;

        for store_ref in secondary_stores {
            let store_checkpoint_dir = secondary_dir.join(store_ref.name);

            // Skip if checkpoint directory already exists (shouldn't happen, but be safe)
            if store_checkpoint_dir.exists() {
                tracing::warn!(
                    store = store_ref.name,
                    "Checkpoint directory already exists, skipping"
                );
                continue;
            }

            let checkpoint = Checkpoint::new(store_ref.db).map_err(|e| {
                anyhow!(
                    "Failed to create checkpoint handle for secondary store '{}': {}",
                    store_ref.name,
                    e
                )
            })?;

            if let Err(e) = checkpoint.create_checkpoint(&store_checkpoint_dir) {
                // Clean up partial checkpoint before returning error
                let _ = fs::remove_dir_all(&store_checkpoint_dir);
                return Err(anyhow!(
                    "Failed to checkpoint secondary store '{}': {}",
                    store_ref.name,
                    e
                ));
            }

            let store_size = dir_size(&store_checkpoint_dir).unwrap_or(0);
            total_secondary_bytes += store_size;
            backed_up_stores.push(store_ref.name.to_string());

            tracing::debug!(
                store = store_ref.name,
                size_kb = store_size / 1024,
                "Secondary store checkpointed"
            );
        }

        // Track graph in metadata if it was checkpointed
        if graph_db.is_some() {
            backed_up_stores.push("graph".to_string());
        }

        // Step 3: Update metadata with secondary store info
        metadata.secondary_stores = backed_up_stores;
        metadata.secondary_size_bytes = total_secondary_bytes;

        // Recompute checksum now that secondary stores are included.
        // The initial checksum from create_backup() only covered the main DB.
        let backup_dir = self.backup_path.join(user_id);
        metadata.checksum = self.calculate_backup_checksum(&backup_dir, metadata.backup_id)?;
        self.save_metadata(&metadata)?;

        tracing::info!(
            backup_id = metadata.backup_id,
            user_id = user_id,
            secondary_stores = metadata.secondary_stores.len(),
            secondary_size_kb = total_secondary_bytes / 1024,
            "Comprehensive backup created"
        );

        Ok(metadata)
    }

    /// Restore from a specific backup
    ///
    /// # Arguments
    /// * `user_id` - User ID to restore
    /// * `backup_id` - Backup ID to restore from (None = latest)
    /// * `restore_path` - Path to restore the database to
    pub fn restore_backup(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
        restore_path: &Path,
    ) -> Result<()> {
        let backup_dir = self.backup_path.join(user_id);

        if !backup_dir.exists() {
            return Err(anyhow!("No backups found for user: {user_id}"));
        }

        let backup_opts = BackupEngineOptions::new(&backup_dir)?;
        let env = Env::new()?;
        let mut backup_engine = BackupEngine::open(&backup_opts, &env)?;

        // Restore from specific backup or latest
        match backup_id {
            Some(id) => {
                tracing::info!(backup_id = id, "Restoring from specific backup");
                backup_engine.restore_from_backup(
                    restore_path,
                    restore_path,
                    &rocksdb::backup::RestoreOptions::default(),
                    id,
                )?;
            }
            None => {
                tracing::info!("Restoring from latest backup");
                backup_engine.restore_from_latest_backup(
                    restore_path,
                    restore_path,
                    &rocksdb::backup::RestoreOptions::default(),
                )?;
            }
        }

        tracing::info!(
            user_id = user_id,
            restore_path = ?restore_path,
            "Restore completed successfully"
        );

        Ok(())
    }

    /// List all available backups for a user
    pub fn list_backups(&self, user_id: &str) -> Result<Vec<BackupMetadata>> {
        let backup_dir = self.backup_path.join(user_id);

        if !backup_dir.exists() {
            return Ok(Vec::new());
        }

        let backup_opts = BackupEngineOptions::new(&backup_dir)?;
        let env = Env::new()?;
        let backup_engine = BackupEngine::open(&backup_opts, &env)?;

        let backup_info = backup_engine.get_backup_info();
        let mut metadata_list = Vec::new();

        for info in backup_info {
            if let Ok(metadata) = self.load_metadata(user_id, info.backup_id) {
                metadata_list.push(metadata);
            }
        }

        Ok(metadata_list)
    }

    /// Restore from a comprehensive backup, including secondary stores.
    ///
    /// The `secondary_restore_paths` map store names to their target restore directories.
    /// Secondary stores are restored by copying the checkpoint directory to the target path.
    pub fn restore_comprehensive_backup(
        &self,
        user_id: &str,
        backup_id: Option<u32>,
        restore_path: &Path,
        secondary_restore_paths: &[(&str, &Path)],
    ) -> Result<Vec<String>> {
        // Step 1: Restore main memories DB
        self.restore_backup(user_id, backup_id, restore_path)?;

        // Step 2: Determine which backup_id was restored
        let resolved_backup_id = match backup_id {
            Some(id) => id,
            None => {
                let backup_dir = self.backup_path.join(user_id);
                let backup_opts = BackupEngineOptions::new(&backup_dir)?;
                let env = Env::new()?;
                let backup_engine = BackupEngine::open(&backup_opts, &env)?;
                let info = backup_engine.get_backup_info();
                info.last()
                    .map(|i| i.backup_id)
                    .ok_or_else(|| anyhow!("No backups available"))?
            }
        };

        // Step 3: Restore secondary stores from checkpoints
        let secondary_dir = self
            .backup_path
            .join(user_id)
            .join(format!("secondary_{resolved_backup_id}"));

        let mut restored_stores = Vec::new();

        if secondary_dir.exists() {
            for (store_name, target_path) in secondary_restore_paths {
                let checkpoint_dir = secondary_dir.join(store_name);
                if !checkpoint_dir.exists() {
                    tracing::debug!(
                        store = *store_name,
                        "No checkpoint found in backup, skipping"
                    );
                    continue;
                }

                // Safe restore: copy to temp dir first, then atomic swap.
                // This prevents data loss if copy fails midway.
                let temp_path = target_path.with_extension("restore_tmp");
                if temp_path.exists() {
                    fs::remove_dir_all(&temp_path).map_err(|e| {
                        anyhow!(
                            "Failed to clean up stale temp dir for {}: {}",
                            store_name,
                            e
                        )
                    })?;
                }

                if let Err(e) = copy_dir_recursive(&checkpoint_dir, &temp_path) {
                    // Copy failed — clean up temp, leave original intact
                    let _ = fs::remove_dir_all(&temp_path);
                    tracing::warn!(
                        store = *store_name,
                        error = %e,
                        "Failed to copy checkpoint for restore, skipping (original data preserved)"
                    );
                    continue;
                }

                // Copy succeeded — now swap: remove original, rename temp to target
                if target_path.exists() {
                    if let Err(e) = fs::remove_dir_all(target_path) {
                        // Can't remove original — roll back by removing temp
                        let _ = fs::remove_dir_all(&temp_path);
                        return Err(anyhow!(
                            "Failed to remove existing {} directory at {:?}: {}",
                            store_name,
                            target_path,
                            e
                        ));
                    }
                }

                if let Err(e) = fs::rename(&temp_path, target_path) {
                    // Rename failed (cross-device?), fall back to copy + remove temp
                    if let Err(copy_err) = copy_dir_recursive(&temp_path, target_path) {
                        let _ = fs::remove_dir_all(&temp_path);
                        return Err(anyhow!(
                            "Failed to finalize restore for {}: rename={}, copy={}",
                            store_name,
                            e,
                            copy_err
                        ));
                    }
                    let _ = fs::remove_dir_all(&temp_path);
                }

                restored_stores.push(store_name.to_string());
                tracing::info!(
                    store = *store_name,
                    target = ?target_path,
                    "Secondary store restored from checkpoint"
                );
            }
        }

        tracing::info!(
            user_id = user_id,
            backup_id = resolved_backup_id,
            restored_secondary = restored_stores.len(),
            "Comprehensive restore completed"
        );

        Ok(restored_stores)
    }

    /// Delete old backups, keeping only the most recent N backups.
    /// `keep_count` must be >= 1 to prevent accidental deletion of all backups.
    pub fn purge_old_backups(&self, user_id: &str, keep_count: usize) -> Result<usize> {
        if keep_count == 0 {
            return Err(anyhow!(
                "keep_count must be >= 1 to prevent deleting all backups"
            ));
        }

        let backup_dir = self.backup_path.join(user_id);

        if !backup_dir.exists() {
            return Ok(0);
        }

        let backup_opts = BackupEngineOptions::new(&backup_dir)?;
        let env = Env::new()?;
        let mut backup_engine = BackupEngine::open(&backup_opts, &env)?;

        let backup_info = backup_engine.get_backup_info();
        let total_backups = backup_info.len();

        if total_backups <= keep_count {
            return Ok(0);
        }

        let to_delete = total_backups - keep_count;

        // Collect IDs of backups that will be purged (oldest ones)
        let mut purge_ids: Vec<u32> = backup_info.iter().map(|b| b.backup_id).collect();
        purge_ids.sort();
        let purge_ids: Vec<u32> = purge_ids.into_iter().take(to_delete).collect();

        // Delete oldest backups (purge keeps the most recent N backups)
        backup_engine.purge_old_backups(keep_count)?;

        // Clean up secondary store checkpoints for purged backups
        for purged_id in &purge_ids {
            let secondary_dir = backup_dir.join(format!("secondary_{purged_id}"));
            if secondary_dir.exists() {
                if let Err(e) = fs::remove_dir_all(&secondary_dir) {
                    tracing::warn!(
                        backup_id = purged_id,
                        error = %e,
                        "Failed to clean up secondary store checkpoint"
                    );
                }
            }
            // Clean up metadata file
            let metadata_path = backup_dir.join(format!("backup_{purged_id}.json"));
            if let Err(e) = fs::remove_file(&metadata_path) {
                tracing::warn!(
                    backup_id = purged_id,
                    error = %e,
                    "Failed to remove backup metadata file"
                );
            }
        }

        tracing::info!(
            purged_count = to_delete,
            kept_count = keep_count,
            user_id = user_id,
            "Purged old backups"
        );

        Ok(to_delete)
    }

    /// Verify backup integrity using checksum
    pub fn verify_backup(&self, user_id: &str, backup_id: u32) -> Result<bool> {
        let metadata = self.load_metadata(user_id, backup_id)?;
        let backup_dir = self.backup_path.join(user_id);

        let current_checksum = self.calculate_backup_checksum(&backup_dir, backup_id)?;

        Ok(current_checksum == metadata.checksum)
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    fn save_metadata(&self, metadata: &BackupMetadata) -> Result<()> {
        let metadata_path = self
            .backup_path
            .join(&metadata.user_id)
            .join(format!("backup_{}.json", metadata.backup_id));

        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(metadata_path, json)?;

        Ok(())
    }

    fn load_metadata(&self, user_id: &str, backup_id: u32) -> Result<BackupMetadata> {
        let metadata_path = self
            .backup_path
            .join(user_id)
            .join(format!("backup_{backup_id}.json"));

        let json = fs::read_to_string(metadata_path)?;
        let metadata = serde_json::from_str(&json)?;

        Ok(metadata)
    }

    fn calculate_backup_checksum(&self, backup_dir: &Path, backup_id: u32) -> Result<String> {
        let mut hasher = Sha256::new();

        // Hash main backup directory (sorted by filename for deterministic ordering)
        let backup_path = backup_dir.join(format!("private/{backup_id}"));
        self.hash_directory_sorted(&backup_path, &mut hasher)?;

        // Hash secondary store directory (B5: was previously excluded)
        let secondary_path = backup_dir.join(format!("secondary_{backup_id}"));
        self.hash_directory_sorted(&secondary_path, &mut hasher)?;

        let result = hasher.finalize();
        Ok(format!("{result:x}"))
    }

    fn hash_directory_sorted(&self, dir: &Path, hasher: &mut Sha256) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        let mut entries: Vec<_> = fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            // Hash filename for rename detection
            hasher.update(entry.file_name().to_string_lossy().as_bytes());
            if path.is_dir() {
                // Recurse into subdirectories (secondary stores have nested structure)
                self.hash_directory_sorted(&path, hasher)?;
            } else {
                let file_contents = fs::read(&path)?;
                hasher.update(&file_contents);
            }
        }
        Ok(())
    }

    fn estimate_memory_count(&self, db: &DB) -> Result<usize> {
        // Estimate by counting keys (this is a rough estimate)
        let mut count = 0;
        let iter = db.iterator(rocksdb::IteratorMode::Start);

        for _ in iter {
            count += 1;
        }

        Ok(count)
    }
}

/// Public wrapper for copy_dir_recursive (used by restore handler).
pub fn copy_dir_recursive_pub(src: &Path, dst: &Path) -> Result<()> {
    copy_dir_recursive(src, dst)
}

/// Calculate total size of a directory recursively
fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                total += dir_size(&entry_path)?;
            } else {
                total += entry.metadata()?.len();
            }
        }
    }
    Ok(total)
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_backup_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let backup_engine = ShodhBackupEngine::new(temp_dir.path().to_path_buf());
        assert!(backup_engine.is_ok());
    }

    #[test]
    fn test_backup_metadata_serialization() {
        let metadata = BackupMetadata {
            backup_id: 1,
            created_at: Utc::now(),
            user_id: "test_user".to_string(),
            backup_type: BackupType::Full,
            size_bytes: 1024,
            checksum: "abc123".to_string(),
            memory_count: 100,
            sequence_number: 42,
            secondary_stores: vec!["todo_items".to_string(), "prospective_tasks".to_string()],
            secondary_size_bytes: 2048,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: BackupMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(metadata.backup_id, deserialized.backup_id);
        assert_eq!(metadata.user_id, deserialized.user_id);
    }
}
