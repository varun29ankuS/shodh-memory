//! File Memory Storage for Codebase Integration
//!
//! Stores learned knowledge about files in a codebase.
//! Separate from regular memories to avoid search pollution.
//!
//! Features:
//! - CRUD operations for FileMemory
//! - Indexing by project, path, and file type
//! - Semantic search via embeddings
//! - Access tracking for heat maps

use anyhow::{Context, Result};
use glob::Pattern;
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, WriteBatch, DB};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use super::types::{
    CodebaseConfig, CodebaseScanResult, FileMemory, FileMemoryId, FileType, IndexingProgress,
    LearnedFrom, ProjectId,
};

const CF_FILES: &str = "files";
const CF_FILE_INDEX: &str = "file_index";

/// Storage and query engine for file memories
pub struct FileMemoryStore {
    db: Arc<DB>,
    /// Default configuration
    config: CodebaseConfig,
}

impl FileMemoryStore {
    /// CF accessor for the files column family
    fn files_cf(&self) -> &ColumnFamily {
        self.db.cf_handle(CF_FILES).expect("files CF must exist")
    }

    /// CF accessor for the file_index column family
    fn file_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_FILE_INDEX)
            .expect("file_index CF must exist")
    }

    /// Return CF descriptors needed by this store. The caller must include
    /// these when opening the shared RocksDB instance.
    pub fn cf_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut cf_opts = Options::default();
        cf_opts.create_if_missing(true);
        vec![
            ColumnFamilyDescriptor::new(CF_FILES, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_FILE_INDEX, cf_opts),
        ]
    }

    /// Create a new file memory store backed by a shared DB that already
    /// contains the required column families (`files`, `file_index`).
    pub fn new(db: Arc<DB>, storage_path: &Path) -> Result<Self> {
        let files_path = storage_path.join("files");
        std::fs::create_dir_all(&files_path)?;

        Self::migrate_from_separate_dbs(&files_path, &db)?;

        tracing::info!("File memory store initialized");
        Ok(Self {
            db,
            config: CodebaseConfig::default(),
        })
    }

    /// One-time migration: copy data from the old separate-DB layout
    /// (`files/memories` and `files/index`) into the column families of the
    /// shared DB, then rename the old directories so the migration is
    /// not repeated.
    fn migrate_from_separate_dbs(files_path: &Path, db: &DB) -> Result<()> {
        let old_dirs: &[(&str, &str)] = &[("memories", CF_FILES), ("index", CF_FILE_INDEX)];

        for (old_name, cf_name) in old_dirs {
            let old_dir = files_path.join(old_name);
            if !old_dir.is_dir() {
                continue;
            }

            let cf = db
                .cf_handle(cf_name)
                .unwrap_or_else(|| panic!("{cf_name} CF must exist"));
            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_dir, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;
                    for item in old_db.iterator(rocksdb::IteratorMode::Start) {
                        let (key, value) = item.map_err(|e| {
                            anyhow::anyhow!("RocksDB iterator error during files migration: {e}")
                        })?;
                        batch.put_cf(cf, &key, &value);
                        count += 1;
                        if count.is_multiple_of(10_000) {
                            db.write(std::mem::take(&mut batch))?;
                        }
                    }
                    if !batch.is_empty() {
                        db.write(batch)?;
                    }
                    drop(old_db);
                    tracing::info!("  files/{old_name}: migrated {count} entries to {cf_name} CF");

                    let backup = files_path.join(format!("{old_name}.pre_cf_migration"));
                    if backup.exists() {
                        let _ = std::fs::remove_dir_all(&backup);
                    }
                    if let Err(e) = std::fs::rename(&old_dir, &backup) {
                        tracing::warn!("Could not rename old {old_name} dir: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Could not open old {old_name} DB for migration: {e}");
                }
            }
        }
        Ok(())
    }

    /// Set custom configuration
    pub fn with_config(mut self, config: CodebaseConfig) -> Self {
        self.config = config;
        self
    }

    /// Flush all column families to disk (critical for graceful shutdown)
    pub fn flush(&self) -> Result<()> {
        use rocksdb::FlushOptions;
        let mut flush_opts = FlushOptions::default();
        flush_opts.set_wait(true);
        for cf_name in &[CF_FILES, CF_FILE_INDEX] {
            if let Some(cf) = self.db.cf_handle(cf_name) {
                self.db
                    .flush_cf_opt(cf, &flush_opts)
                    .map_err(|e| anyhow::anyhow!("Failed to flush {cf_name}: {e}"))?;
            }
        }
        Ok(())
    }

    /// Get references to all RocksDB databases for backup
    pub fn databases(&self) -> Vec<(&str, &Arc<DB>)> {
        vec![("files_shared", &self.db)]
    }

    // =========================================================================
    // CRUD OPERATIONS
    // =========================================================================

    /// Store a new file memory
    pub fn store(&self, file_memory: &FileMemory) -> Result<()> {
        let key = format!("{}:{}", file_memory.user_id, file_memory.id.0);
        let value = serde_json::to_vec(file_memory).context("Failed to serialize file memory")?;

        self.db
            .put_cf(self.files_cf(), key.as_bytes(), &value)
            .context("Failed to store file memory")?;

        self.update_indices(file_memory)?;

        tracing::debug!(
            file_id = %file_memory.id,
            path = %file_memory.path,
            user_id = %file_memory.user_id,
            "Stored file memory"
        );

        Ok(())
    }

    /// Get a file memory by ID
    pub fn get(&self, user_id: &str, file_id: &FileMemoryId) -> Result<Option<FileMemory>> {
        let key = format!("{}:{}", user_id, file_id.0);

        match self.db.get_cf(self.files_cf(), key.as_bytes())? {
            Some(value) => {
                let file_memory: FileMemory =
                    serde_json::from_slice(&value).context("Failed to deserialize file memory")?;
                Ok(Some(file_memory))
            }
            None => Ok(None),
        }
    }

    /// Get a file memory by path (relative path within project)
    pub fn get_by_path(
        &self,
        user_id: &str,
        project_id: &ProjectId,
        path: &str,
    ) -> Result<Option<FileMemory>> {
        // Look up in path index
        let path_key = format!(
            "path:{}:{}:{}",
            user_id,
            project_id.0,
            Self::hash_path(path)
        );

        match self.db.get_cf(self.file_index_cf(), path_key.as_bytes())? {
            Some(file_id_bytes) => {
                let file_id_str =
                    String::from_utf8(file_id_bytes.to_vec()).context("Invalid file ID")?;
                let file_id = FileMemoryId(
                    uuid::Uuid::parse_str(&file_id_str).context("Invalid file ID UUID")?,
                );
                self.get(user_id, &file_id)
            }
            None => Ok(None),
        }
    }

    /// Update a file memory
    pub fn update(&self, file_memory: &FileMemory) -> Result<()> {
        // Remove old indices first (in case path changed)
        if let Some(existing) = self.get(&file_memory.user_id, &file_memory.id)? {
            self.remove_indices(&existing)?;
        }

        // Store updated version
        self.store(file_memory)
    }

    /// Delete a file memory
    pub fn delete(&self, user_id: &str, file_id: &FileMemoryId) -> Result<bool> {
        if let Some(file_memory) = self.get(user_id, file_id)? {
            let key = format!("{}:{}", user_id, file_id.0);
            self.db.delete_cf(self.files_cf(), key.as_bytes())?;
            self.remove_indices(&file_memory)?;

            tracing::debug!(
                file_id = %file_id,
                path = %file_memory.path,
                "Deleted file memory"
            );

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete all file memories for a project
    pub fn delete_project_files(&self, user_id: &str, project_id: &ProjectId) -> Result<usize> {
        let files = self.list_by_project(user_id, project_id, None)?;
        let count = files.len();

        for file in files {
            self.delete(user_id, &file.id)?;
        }

        tracing::info!(
            project_id = %project_id.0,
            count = count,
            "Deleted all file memories for project"
        );

        Ok(count)
    }

    // =========================================================================
    // LISTING & QUERYING
    // =========================================================================

    /// List all file memories for a user
    pub fn list_by_user(&self, user_id: &str, limit: Option<usize>) -> Result<Vec<FileMemory>> {
        let prefix = format!("user:{}:", user_id);
        let mut files = Vec::new();

        let iter = self
            .db
            .prefix_iterator_cf(self.file_index_cf(), prefix.as_bytes());
        for item in iter {
            let (key, file_id_bytes) = item?;
            let key_str = String::from_utf8_lossy(&key);

            // Stop if we've left our prefix
            if !key_str.starts_with(&prefix) {
                break;
            }

            let file_id_str = String::from_utf8(file_id_bytes.to_vec())?;
            let file_id =
                FileMemoryId(uuid::Uuid::parse_str(&file_id_str).context("Invalid file ID")?);

            if let Some(file) = self.get(user_id, &file_id)? {
                files.push(file);

                if let Some(lim) = limit {
                    if files.len() >= lim {
                        break;
                    }
                }
            }
        }

        // Sort by access count descending (most accessed first)
        files.sort_by(|a, b| b.access_count.cmp(&a.access_count));

        Ok(files)
    }

    /// List all file memories for a project
    pub fn list_by_project(
        &self,
        user_id: &str,
        project_id: &ProjectId,
        limit: Option<usize>,
    ) -> Result<Vec<FileMemory>> {
        let prefix = format!("project:{}:{}:", user_id, project_id.0);
        let mut files = Vec::new();

        let iter = self
            .db
            .prefix_iterator_cf(self.file_index_cf(), prefix.as_bytes());
        for item in iter {
            let (key, file_id_bytes) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let file_id_str = String::from_utf8(file_id_bytes.to_vec())?;
            let file_id =
                FileMemoryId(uuid::Uuid::parse_str(&file_id_str).context("Invalid file ID")?);

            if let Some(file) = self.get(user_id, &file_id)? {
                files.push(file);

                if let Some(lim) = limit {
                    if files.len() >= lim {
                        break;
                    }
                }
            }
        }

        // Sort by path for consistent ordering
        files.sort_by(|a, b| a.path.cmp(&b.path));

        Ok(files)
    }

    /// List file memories by type
    pub fn list_by_type(
        &self,
        user_id: &str,
        project_id: &ProjectId,
        file_type: &FileType,
        limit: Option<usize>,
    ) -> Result<Vec<FileMemory>> {
        let type_str = format!("{:?}", file_type);
        let prefix = format!("type:{}:{}:{}:", user_id, project_id.0, type_str);
        let mut files = Vec::new();

        let iter = self
            .db
            .prefix_iterator_cf(self.file_index_cf(), prefix.as_bytes());
        for item in iter {
            let (key, file_id_bytes) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let file_id_str = String::from_utf8(file_id_bytes.to_vec())?;
            let file_id =
                FileMemoryId(uuid::Uuid::parse_str(&file_id_str).context("Invalid file ID")?);

            if let Some(file) = self.get(user_id, &file_id)? {
                files.push(file);

                if let Some(lim) = limit {
                    if files.len() >= lim {
                        break;
                    }
                }
            }
        }

        Ok(files)
    }

    /// Get file count for a project
    pub fn count_by_project(&self, user_id: &str, project_id: &ProjectId) -> Result<usize> {
        let prefix = format!("project:{}:{}:", user_id, project_id.0);
        let mut count = 0;

        let iter = self
            .db
            .prefix_iterator_cf(self.file_index_cf(), prefix.as_bytes());
        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            count += 1;
        }

        Ok(count)
    }

    // =========================================================================
    // ACCESS TRACKING
    // =========================================================================

    /// Record an access to a file (increments counter, updates timestamp)
    pub fn record_access(
        &self,
        user_id: &str,
        project_id: &ProjectId,
        path: &str,
        learned_from: LearnedFrom,
    ) -> Result<Option<FileMemory>> {
        if let Some(mut file) = self.get_by_path(user_id, project_id, path)? {
            file.record_access(learned_from);
            self.update(&file)?;
            Ok(Some(file))
        } else {
            Ok(None)
        }
    }

    // =========================================================================
    // CODEBASE SCANNING
    // =========================================================================

    /// Scan a directory and return eligible files for indexing
    pub fn scan_codebase(
        &self,
        codebase_path: &Path,
        config: Option<&CodebaseConfig>,
    ) -> Result<CodebaseScanResult> {
        let config = config.unwrap_or(&self.config);
        let mut result = CodebaseScanResult {
            total_files: 0,
            eligible_files: 0,
            skipped_files: 0,
            skip_reasons: HashMap::new(),
            limit_reached: false,
            file_paths: Vec::new(),
        };

        // Compile exclude patterns
        let exclude_patterns: Vec<Pattern> = config
            .exclude_patterns
            .iter()
            .filter_map(|p| Pattern::new(p).ok())
            .collect();

        self.scan_directory_recursive(
            codebase_path,
            codebase_path,
            &exclude_patterns,
            config,
            &mut result,
        )?;

        tracing::info!(
            path = %codebase_path.display(),
            total = result.total_files,
            eligible = result.eligible_files,
            skipped = result.skipped_files,
            limit_reached = result.limit_reached,
            "Scanned codebase"
        );

        Ok(result)
    }

    fn scan_directory_recursive(
        &self,
        root: &Path,
        current: &Path,
        exclude_patterns: &[Pattern],
        config: &CodebaseConfig,
        result: &mut CodebaseScanResult,
    ) -> Result<()> {
        if result.limit_reached {
            return Ok(());
        }

        let entries = match fs::read_dir(current) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(path = %current.display(), error = %e, "Failed to read directory");
                return Ok(());
            }
        };

        // Commonly excluded directory names (checked explicitly for performance and reliability)
        const EXCLUDED_DIR_NAMES: &[&str] = &[
            ".git",
            ".svn",
            ".hg",
            ".bzr", // VCS
            "node_modules",
            "__pycache__",
            ".venv", // Dependencies
            "venv",
            "env",
            ".env",
            "virtualenv", // More Python venvs
            "site-packages",
            "Lib",
            "Scripts", // Python internals
            "target",
            "dist",
            "build",
            "out",
            "bin", // Build outputs
            ".idea",
            ".vscode", // IDE
            ".cache",
            ".tmp",
            "tmp", // Temp
            "data",
            "logs",
            "coverage", // Runtime data
            "release-test",
            "test-wheel", // Test artifacts
        ];

        // Directory name patterns to skip (suffix matching)
        const EXCLUDED_DIR_SUFFIXES: &[&str] = &[
            "_data",    // Any *_data directories (e.g., shodh_memory_data)
            "_cache",   // Any *_cache directories
            "_output",  // Any *_output directories
            "_venv",    // Any *_venv directories
            "_env",     // Any *_env directories
            "_install", // Any *_install directories (test installs)
        ];

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            // Quick check: skip commonly excluded directories by name
            if path.is_dir() {
                // Exact match exclusion
                if EXCLUDED_DIR_NAMES.iter().any(|&name| file_name_str == name) {
                    *result
                        .skip_reasons
                        .entry(format!("{}/", file_name_str))
                        .or_insert(0) += 1;
                    result.skipped_files += 1;
                    continue;
                }
                // Suffix pattern exclusion (e.g., *_data, *_cache)
                if EXCLUDED_DIR_SUFFIXES
                    .iter()
                    .any(|&suffix| file_name_str.ends_with(suffix))
                {
                    *result
                        .skip_reasons
                        .entry(format!(
                            "*{}/",
                            file_name_str
                                .rsplit_once('_')
                                .map_or(&file_name_str[..], |(_, s)| s)
                        ))
                        .or_insert(0) += 1;
                    result.skipped_files += 1;
                    continue;
                }
            }

            let relative_path = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");

            // Check exclude patterns (for custom patterns and file patterns like *.lock)
            let mut excluded = false;
            for pattern in exclude_patterns {
                // For directory patterns (ending with /), check if relative path starts with it
                let pattern_str = pattern.as_str();
                if pattern_str.ends_with('/') {
                    let dir_name = pattern_str.trim_end_matches('/');
                    if relative_path == dir_name
                        || relative_path.starts_with(&format!("{}/", dir_name))
                    {
                        *result.skip_reasons.entry(pattern.to_string()).or_insert(0) += 1;
                        excluded = true;
                        break;
                    }
                } else if pattern.matches(&relative_path) || pattern.matches(&file_name_str) {
                    *result.skip_reasons.entry(pattern.to_string()).or_insert(0) += 1;
                    excluded = true;
                    break;
                }
            }

            if excluded {
                result.skipped_files += 1;
                continue;
            }

            if path.is_dir() {
                // Recurse into directory
                self.scan_directory_recursive(root, &path, exclude_patterns, config, result)?;
            } else if path.is_file() {
                result.total_files += 1;

                // Check if binary
                if config.skip_binary && Self::is_likely_binary(&path) {
                    *result.skip_reasons.entry("binary".to_string()).or_insert(0) += 1;
                    result.skipped_files += 1;
                    continue;
                }

                // Check file size
                if let Ok(metadata) = path.metadata() {
                    if metadata.len() > config.max_file_size_for_embedding as u64 {
                        *result
                            .skip_reasons
                            .entry("too_large".to_string())
                            .or_insert(0) += 1;
                        result.skipped_files += 1;
                        continue;
                    }
                }

                // File is eligible
                result.eligible_files += 1;
                result.file_paths.push(relative_path);

                // Check limit
                if result.eligible_files >= config.max_files_per_project {
                    result.limit_reached = true;
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Check if a file is likely binary (non-text)
    fn is_likely_binary(path: &Path) -> bool {
        let binary_extensions = [
            "exe", "dll", "so", "dylib", "bin", "obj", "o", "a", "lib", "png", "jpg", "jpeg",
            "gif", "bmp", "ico", "webp", "mp3", "mp4", "avi", "mov", "mkv", "wav", "flac", "zip",
            "tar", "gz", "rar", "7z", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "woff",
            "woff2", "ttf", "otf", "eot", "class", "pyc", "pyo", "wasm",
        ];

        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| binary_extensions.contains(&e.to_lowercase().as_str()))
            .unwrap_or(false)
    }

    // =========================================================================
    // FILE HASHING
    // =========================================================================

    /// Compute SHA256 hash of file content
    pub fn hash_file_content(content: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content);
        format!("{:x}", hasher.finalize())
    }

    /// Compute SHA256 hash of a path (for indexing)
    fn hash_path(path: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(path.as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
    }

    // =========================================================================
    // FILE INDEXING
    // =========================================================================

    /// Index a single file: read, hash, count lines, detect type, extract key items
    pub fn index_file(
        &self,
        codebase_root: &Path,
        relative_path: &str,
        project_id: &ProjectId,
        user_id: &str,
    ) -> Result<FileMemory> {
        let absolute_path = codebase_root.join(relative_path);
        let content = fs::read(&absolute_path)
            .with_context(|| format!("Failed to read file: {}", absolute_path.display()))?;

        let file_hash = Self::hash_file_content(&content);
        let size_bytes = content.len() as u64;

        // Count lines (for text files)
        let content_str = String::from_utf8_lossy(&content);
        let line_count = content_str.lines().count();

        // Detect file type from extension
        let file_type = absolute_path
            .extension()
            .and_then(|e| e.to_str())
            .map(FileType::from_extension)
            .unwrap_or_default();

        // Extract key items (functions, classes, etc.)
        let key_items = Self::extract_key_items(&content_str, &file_type);

        // Create FileMemory
        let mut file_memory = FileMemory::new(
            project_id.clone(),
            user_id.to_string(),
            relative_path.to_string(),
            absolute_path.to_string_lossy().to_string(),
            file_hash,
            file_type,
            line_count,
            size_bytes,
        );

        file_memory.key_items = key_items;

        // Store it
        self.store(&file_memory)?;

        Ok(file_memory)
    }

    /// Index a single file and generate embedding
    pub fn index_file_with_embedding<E: crate::embeddings::Embedder>(
        &self,
        codebase_root: &Path,
        relative_path: &str,
        project_id: &ProjectId,
        user_id: &str,
        embedder: &E,
    ) -> Result<FileMemory> {
        let mut file_memory = self.index_file(codebase_root, relative_path, project_id, user_id)?;

        // Generate embedding from summary content
        let embed_content = Self::prepare_embed_content(&file_memory);
        if !embed_content.is_empty() {
            match embedder.encode(&embed_content) {
                Ok(embedding) => {
                    file_memory.embedding = Some(embedding);
                    self.update(&file_memory)?;
                }
                Err(e) => {
                    tracing::warn!(
                        path = %file_memory.path,
                        error = %e,
                        "Failed to generate embedding for file"
                    );
                }
            }
        }

        Ok(file_memory)
    }

    /// Prepare content for embedding (path + key items + summary)
    fn prepare_embed_content(file: &FileMemory) -> String {
        let mut parts = Vec::new();

        // Include relative path (helps with file discovery)
        parts.push(file.path.clone());

        // Include key items
        if !file.key_items.is_empty() {
            parts.push(file.key_items.join(" "));
        }

        // Include summary if available
        if !file.summary.is_empty() {
            parts.push(file.summary.clone());
        }

        // Include purpose if available
        if let Some(ref purpose) = file.purpose {
            parts.push(purpose.clone());
        }

        parts.join(" | ")
    }

    /// Extract key items from file content based on file type
    fn extract_key_items(content: &str, file_type: &FileType) -> Vec<String> {
        let mut items = Vec::new();

        match file_type {
            FileType::Rust => {
                // Extract pub fn, pub struct, pub enum, pub trait, impl
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub async fn ")
                        || trimmed.starts_with("pub struct ")
                        || trimmed.starts_with("pub enum ")
                        || trimmed.starts_with("pub trait ")
                        || trimmed.starts_with("impl ")
                    {
                        // Extract the name
                        if let Some(name) = Self::extract_rust_name(trimmed) {
                            if !items.contains(&name) {
                                items.push(name);
                            }
                        }
                    }
                }
            }
            FileType::TypeScript | FileType::JavaScript => {
                // Extract export function, export class, export const, export interface
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("export ")
                        || trimmed.starts_with("function ")
                        || trimmed.starts_with("class ")
                        || trimmed.starts_with("interface ")
                        || trimmed.starts_with("const ")
                    {
                        if let Some(name) = Self::extract_js_name(trimmed) {
                            if !items.contains(&name) {
                                items.push(name);
                            }
                        }
                    }
                }
            }
            FileType::Python => {
                // Extract def, class, async def
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("def ")
                        || trimmed.starts_with("async def ")
                        || trimmed.starts_with("class ")
                    {
                        if let Some(name) = Self::extract_python_name(trimmed) {
                            if !items.contains(&name) {
                                items.push(name);
                            }
                        }
                    }
                }
            }
            FileType::Go => {
                // Extract func, type struct, type interface
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("func ") || trimmed.starts_with("type ") {
                        if let Some(name) = Self::extract_go_name(trimmed) {
                            if !items.contains(&name) {
                                items.push(name);
                            }
                        }
                    }
                }
            }
            _ => {
                // For other types, just count significant lines
                // Could add more extractors later
            }
        }

        // Limit to prevent bloat
        items.truncate(50);
        items
    }

    fn extract_rust_name(line: &str) -> Option<String> {
        // "pub fn foo(" -> "foo"
        // "pub struct Bar {" -> "Bar"
        // "impl Foo for Bar {" -> "Foo for Bar"
        let line = line.trim_start_matches("pub ").trim_start_matches("async ");

        if line.starts_with("fn ") {
            let rest = line.strip_prefix("fn ")?;
            let name = rest.split(['(', '<']).next()?;
            Some(name.trim().to_string())
        } else if line.starts_with("struct ") {
            let rest = line.strip_prefix("struct ")?;
            let name = rest.split(['{', '<', '(']).next()?;
            Some(name.trim().to_string())
        } else if line.starts_with("enum ") {
            let rest = line.strip_prefix("enum ")?;
            let name = rest.split(['{', '<']).next()?;
            Some(name.trim().to_string())
        } else if line.starts_with("trait ") {
            let rest = line.strip_prefix("trait ")?;
            let name = rest.split(['{', '<', ':']).next()?;
            Some(name.trim().to_string())
        } else if line.starts_with("impl ") {
            let rest = line.strip_prefix("impl ")?;
            let sig = rest.split('{').next()?;
            Some(sig.trim().to_string())
        } else {
            None
        }
    }

    fn extract_js_name(line: &str) -> Option<String> {
        // "export function foo(" -> "foo"
        // "export class Bar {" -> "Bar"
        let line = line
            .trim_start_matches("export ")
            .trim_start_matches("default ")
            .trim_start_matches("async ");

        if line.starts_with("function ") {
            let rest = line.strip_prefix("function ")?;
            let name = rest.split('(').next()?;
            Some(name.trim().to_string())
        } else if line.starts_with("class ") {
            let rest = line.strip_prefix("class ")?;
            let name = rest.split(['{', ' ']).next()?;
            Some(name.trim().to_string())
        } else if line.starts_with("interface ") {
            let rest = line.strip_prefix("interface ")?;
            let name = rest.split(['{', ' ', '<']).next()?;
            Some(name.trim().to_string())
        } else if line.starts_with("const ") {
            let rest = line.strip_prefix("const ")?;
            let name = rest.split(['=', ':']).next()?;
            Some(name.trim().to_string())
        } else {
            None
        }
    }

    fn extract_python_name(line: &str) -> Option<String> {
        // "def foo(" -> "foo"
        // "class Bar:" -> "Bar"
        let line = line.trim_start_matches("async ");

        if line.starts_with("def ") {
            let rest = line.strip_prefix("def ")?;
            let name = rest.split('(').next()?;
            Some(name.trim().to_string())
        } else if line.starts_with("class ") {
            let rest = line.strip_prefix("class ")?;
            let name = rest.split(['(', ':']).next()?;
            Some(name.trim().to_string())
        } else {
            None
        }
    }

    fn extract_go_name(line: &str) -> Option<String> {
        // "func Foo(" -> "Foo"
        // "func (r *Receiver) Method(" -> "Method"
        // "type Bar struct" -> "Bar"
        if line.starts_with("func ") {
            let rest = line.strip_prefix("func ")?;
            // Handle method receivers: func (r *Type) Name(
            if rest.starts_with('(') {
                // Skip receiver, find method name
                let after_receiver = rest.split(')').nth(1)?;
                let name = after_receiver.trim().split('(').next()?;
                Some(name.trim().to_string())
            } else {
                let name = rest.split('(').next()?;
                Some(name.trim().to_string())
            }
        } else if line.starts_with("type ") {
            let rest = line.strip_prefix("type ")?;
            let name = rest.split_whitespace().next()?;
            Some(name.trim().to_string())
        } else {
            None
        }
    }

    /// Index all files in a codebase (blocking version)
    pub fn index_codebase(
        &self,
        codebase_root: &Path,
        project_id: &ProjectId,
        user_id: &str,
        config: Option<&CodebaseConfig>,
    ) -> Result<IndexingResult> {
        // First scan to get eligible files
        let scan_result = self.scan_codebase(codebase_root, config)?;

        let mut result = IndexingResult {
            total_files: scan_result.eligible_files,
            indexed_files: 0,
            skipped_files: 0,
            errors: Vec::new(),
        };

        for relative_path in &scan_result.file_paths {
            match self.index_file(codebase_root, relative_path, project_id, user_id) {
                Ok(_) => {
                    result.indexed_files += 1;
                }
                Err(e) => {
                    result.errors.push(format!("{}: {}", relative_path, e));
                    result.skipped_files += 1;
                }
            }
        }

        tracing::info!(
            path = %codebase_root.display(),
            total = result.total_files,
            indexed = result.indexed_files,
            skipped = result.skipped_files,
            errors = result.errors.len(),
            "Indexed codebase"
        );

        Ok(result)
    }

    /// Index codebase with embeddings (requires embedder)
    pub fn index_codebase_with_embeddings<E: crate::embeddings::Embedder>(
        &self,
        codebase_root: &Path,
        project_id: &ProjectId,
        user_id: &str,
        embedder: &E,
        config: Option<&CodebaseConfig>,
        progress_callback: Option<&dyn Fn(IndexingProgress)>,
    ) -> Result<IndexingResult> {
        let scan_result = self.scan_codebase(codebase_root, config)?;

        let mut result = IndexingResult {
            total_files: scan_result.eligible_files,
            indexed_files: 0,
            skipped_files: 0,
            errors: Vec::new(),
        };

        let mut progress = IndexingProgress::new(scan_result.eligible_files);

        for relative_path in &scan_result.file_paths {
            progress.current_file = Some(relative_path.clone());

            match self.index_file_with_embedding(
                codebase_root,
                relative_path,
                project_id,
                user_id,
                embedder,
            ) {
                Ok(_) => {
                    result.indexed_files += 1;
                }
                Err(e) => {
                    let error_msg = format!("{}: {}", relative_path, e);
                    result.errors.push(error_msg.clone());
                    progress.errors.push(error_msg);
                    result.skipped_files += 1;
                }
            }

            progress.processed += 1;

            if let Some(cb) = progress_callback {
                cb(progress.clone());
            }
        }

        progress.complete = true;
        if let Some(cb) = progress_callback {
            cb(progress);
        }

        tracing::info!(
            path = %codebase_root.display(),
            total = result.total_files,
            indexed = result.indexed_files,
            skipped = result.skipped_files,
            errors = result.errors.len(),
            "Indexed codebase with embeddings"
        );

        Ok(result)
    }

    // =========================================================================
    // INDEX MANAGEMENT
    // =========================================================================

    fn update_indices(&self, file: &FileMemory) -> Result<()> {
        let mut batch = WriteBatch::default();
        let id_str = file.id.0.to_string();
        let idx_cf = self.file_index_cf();

        // Index by user
        let user_key = format!("user:{}:{}", file.user_id, id_str);
        batch.put_cf(idx_cf, user_key.as_bytes(), id_str.as_bytes());

        // Index by project
        let project_key = format!("project:{}:{}:{}", file.user_id, file.project_id.0, id_str);
        batch.put_cf(idx_cf, project_key.as_bytes(), id_str.as_bytes());

        // Index by path (for fast lookup)
        let path_key = format!(
            "path:{}:{}:{}",
            file.user_id,
            file.project_id.0,
            Self::hash_path(&file.path)
        );
        batch.put_cf(idx_cf, path_key.as_bytes(), id_str.as_bytes());

        // Index by file type
        let type_str = format!("{:?}", file.file_type);
        let type_key = format!(
            "type:{}:{}:{}:{}",
            file.user_id, file.project_id.0, type_str, id_str
        );
        batch.put_cf(idx_cf, type_key.as_bytes(), id_str.as_bytes());

        self.db
            .write(batch)
            .context("Failed to update file memory indices")?;

        Ok(())
    }

    fn remove_indices(&self, file: &FileMemory) -> Result<()> {
        let mut batch = WriteBatch::default();
        let id_str = file.id.0.to_string();
        let idx_cf = self.file_index_cf();

        let user_key = format!("user:{}:{}", file.user_id, id_str);
        batch.delete_cf(idx_cf, user_key.as_bytes());

        let project_key = format!("project:{}:{}:{}", file.user_id, file.project_id.0, id_str);
        batch.delete_cf(idx_cf, project_key.as_bytes());

        let path_key = format!(
            "path:{}:{}:{}",
            file.user_id,
            file.project_id.0,
            Self::hash_path(&file.path)
        );
        batch.delete_cf(idx_cf, path_key.as_bytes());

        let type_str = format!("{:?}", file.file_type);
        let type_key = format!(
            "type:{}:{}:{}:{}",
            file.user_id, file.project_id.0, type_str, id_str
        );
        batch.delete_cf(idx_cf, type_key.as_bytes());

        self.db.write(batch)?;
        Ok(())
    }

    // =========================================================================
    // STATS
    // =========================================================================

    /// Get statistics for file memories
    pub fn stats(&self, user_id: &str) -> Result<FileMemoryStats> {
        let files = self.list_by_user(user_id, None)?;

        let total_files = files.len();
        let total_size: u64 = files.iter().map(|f| f.size_bytes).sum();
        let total_lines: usize = files.iter().map(|f| f.line_count).sum();
        let total_accesses: u32 = files.iter().map(|f| f.access_count).sum();

        // Count by type
        let mut by_type: HashMap<String, usize> = HashMap::new();
        for file in &files {
            let type_str = format!("{:?}", file.file_type);
            *by_type.entry(type_str).or_insert(0) += 1;
        }

        // Count by learned_from
        let mut by_source: HashMap<String, usize> = HashMap::new();
        for file in &files {
            let source_str = format!("{:?}", file.learned_from);
            *by_source.entry(source_str).or_insert(0) += 1;
        }

        Ok(FileMemoryStats {
            total_files,
            total_size_bytes: total_size,
            total_lines,
            total_accesses,
            by_type,
            by_source,
        })
    }
}

/// Statistics about file memories
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileMemoryStats {
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub total_lines: usize,
    pub total_accesses: u32,
    pub by_type: HashMap<String, usize>,
    pub by_source: HashMap<String, usize>,
}

/// Result of indexing a codebase
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexingResult {
    /// Total files attempted
    pub total_files: usize,
    /// Files successfully indexed
    pub indexed_files: usize,
    /// Files skipped due to errors
    pub skipped_files: usize,
    /// Error messages for failed files
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store() -> (FileMemoryStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("files_db");

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let mut cfs = vec![ColumnFamilyDescriptor::new("default", {
            let mut o = Options::default();
            o.create_if_missing(true);
            o
        })];
        cfs.extend(FileMemoryStore::cf_descriptors());
        let db = Arc::new(DB::open_cf_descriptors(&opts, &db_path, cfs).unwrap());
        let store = FileMemoryStore::new(db, temp_dir.path()).unwrap();
        (store, temp_dir)
    }

    #[test]
    fn test_store_and_retrieve() {
        let (store, _dir) = create_test_store();

        let project_id = ProjectId::new();
        let file = FileMemory::new(
            project_id.clone(),
            "test-user".to_string(),
            "src/main.rs".to_string(),
            "/home/user/project/src/main.rs".to_string(),
            "abc123".to_string(),
            FileType::Rust,
            100,
            5000,
        );

        // Store
        store.store(&file).unwrap();

        // Retrieve by ID
        let retrieved = store.get("test-user", &file.id).unwrap().unwrap();
        assert_eq!(retrieved.path, "src/main.rs");
        assert_eq!(retrieved.file_type, FileType::Rust);

        // Retrieve by path
        let by_path = store
            .get_by_path("test-user", &project_id, "src/main.rs")
            .unwrap()
            .unwrap();
        assert_eq!(by_path.id, file.id);
    }

    #[test]
    fn test_list_by_project() {
        let (store, _dir) = create_test_store();

        let project_id = ProjectId::new();

        // Create multiple files
        for i in 0..5 {
            let file = FileMemory::new(
                project_id.clone(),
                "test-user".to_string(),
                format!("src/file{}.rs", i),
                format!("/home/user/project/src/file{}.rs", i),
                format!("hash{}", i),
                FileType::Rust,
                100,
                5000,
            );
            store.store(&file).unwrap();
        }

        let files = store
            .list_by_project("test-user", &project_id, None)
            .unwrap();
        assert_eq!(files.len(), 5);
    }

    #[test]
    fn test_record_access() {
        let (store, _dir) = create_test_store();

        let project_id = ProjectId::new();
        let file = FileMemory::new(
            project_id.clone(),
            "test-user".to_string(),
            "src/main.rs".to_string(),
            "/home/user/project/src/main.rs".to_string(),
            "abc123".to_string(),
            FileType::Rust,
            100,
            5000,
        );

        store.store(&file).unwrap();

        // Record access
        let updated = store
            .record_access(
                "test-user",
                &project_id,
                "src/main.rs",
                LearnedFrom::ReadAccess,
            )
            .unwrap()
            .unwrap();

        assert_eq!(updated.access_count, 2); // 1 initial + 1 access
        assert_eq!(updated.learned_from, LearnedFrom::ReadAccess);
    }

    #[test]
    fn test_delete() {
        let (store, _dir) = create_test_store();

        let project_id = ProjectId::new();
        let file = FileMemory::new(
            project_id.clone(),
            "test-user".to_string(),
            "src/main.rs".to_string(),
            "/home/user/project/src/main.rs".to_string(),
            "abc123".to_string(),
            FileType::Rust,
            100,
            5000,
        );

        store.store(&file).unwrap();

        // Delete
        let deleted = store.delete("test-user", &file.id).unwrap();
        assert!(deleted);

        // Verify gone
        let retrieved = store.get("test-user", &file.id).unwrap();
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_file_type_detection() {
        assert_eq!(FileType::from_extension("rs"), FileType::Rust);
        assert_eq!(FileType::from_extension("ts"), FileType::TypeScript);
        assert_eq!(FileType::from_extension("tsx"), FileType::TypeScript);
        assert_eq!(FileType::from_extension("py"), FileType::Python);
        assert_eq!(FileType::from_extension("go"), FileType::Go);
        assert_eq!(FileType::from_extension("md"), FileType::Markdown);
        assert!(matches!(
            FileType::from_extension("unknown"),
            FileType::Other(_)
        ));
    }
}
