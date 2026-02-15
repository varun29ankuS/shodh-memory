//! GTD-style Todo Management (Linear-inspired)
//!
//! Features:
//! - CRUD operations for todos and projects
//! - Status-based workflow (Backlog -> Todo -> InProgress -> Done)
//! - Priority levels (Urgent, High, Medium, Low)
//! - GTD contexts (@computer, @phone, @errands, etc.)
//! - Project grouping
//! - Recurring tasks with automatic next instance creation
//! - Due date tracking with overdue detection
//! - Vector embeddings for semantic search (MiniLM-L6-v2)
//! - Vamana HNSW index for fast similarity search

use anyhow::{Context, Result};
use chrono::Utc;
use parking_lot::RwLock;
use rocksdb::{Options, WriteBatch, DB};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use super::types::{
    Project, ProjectId, ProjectStatus, Todo, TodoComment, TodoCommentId, TodoCommentType, TodoId,
    TodoStatus,
};
use crate::vector_db::{VamanaConfig, VamanaIndex};

/// Embedding dimension (MiniLM-L6-v2)
const EMBEDDING_DIM: usize = 384;

/// Migrate unpadded `due:{ts}:{uid}:{id}` keys to zero-padded `due:{:020}:{uid}:{id}` format.
///
/// Prior versions wrote bare timestamps (e.g. `due:1739404800:user:uuid`), which break
/// lexicographic ordering (`"9" > "10"`). Zero-padding to 20 digits ensures
/// lex order = chronological order, enabling ordered range scans.
fn migrate_due_key_padding(index_db: &DB) -> Result<usize> {
    let mut batch = WriteBatch::default();
    let mut count = 0;

    for item in index_db.prefix_iterator(b"due:") {
        let (key, value) = item.context("Failed to read due index during migration")?;
        let key_str = std::str::from_utf8(&key).context("Non-UTF8 key in todo due index")?;

        // Key format: due:{timestamp}:{user_id}:{todo_id}
        let parts: Vec<&str> = key_str.splitn(4, ':').collect();
        if parts.len() != 4 {
            continue;
        }

        // Already padded — nothing to do
        if parts[1].len() >= 20 {
            continue;
        }

        if let Ok(ts) = parts[1].parse::<i64>() {
            let new_key = format!("due:{:020}:{}:{}", ts, parts[2], parts[3]);
            batch.delete(&*key);
            batch.put(new_key.as_bytes(), &*value);
            count += 1;
        }
    }

    if count > 0 {
        index_db
            .write(batch)
            .context("Failed to write migrated todo due keys")?;
        tracing::info!(count, "Migrated todo due keys to zero-padded format");
    }

    Ok(count)
}

/// Storage and query engine for todos and projects
pub struct TodoStore {
    /// Main todo storage: key = {user_id}:{todo_id}
    todo_db: Arc<DB>,
    /// Project storage: key = {user_id}:{project_id}
    project_db: Arc<DB>,
    /// Index database for efficient queries
    index_db: Arc<DB>,
    /// Vector index for semantic search (per-user indices)
    vector_indices: RwLock<HashMap<String, VamanaIndex>>,
    /// Storage path for persisting vector indices
    storage_path: std::path::PathBuf,
    /// Mutex for atomic sequence number allocation (prevents TOCTOU race)
    seq_mutex: parking_lot::Mutex<()>,
}

impl TodoStore {
    /// Create a new todo store at the given path
    pub fn new(storage_path: &Path) -> Result<Self> {
        let todos_path = storage_path.join("todos");
        std::fs::create_dir_all(&todos_path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_max_write_buffer_number(2);
        opts.set_write_buffer_size(32 * 1024 * 1024); // 32MB

        let todo_db =
            Arc::new(DB::open(&opts, todos_path.join("items")).context("Failed to open todos DB")?);

        let project_db = Arc::new(
            DB::open(&opts, todos_path.join("projects")).context("Failed to open projects DB")?,
        );

        let index_db = Arc::new(
            DB::open(&opts, todos_path.join("index")).context("Failed to open todos index DB")?,
        );

        // Migrate any unpadded due keys from prior versions
        migrate_due_key_padding(&index_db)?;

        tracing::info!("Todo store initialized");

        Ok(Self {
            todo_db,
            project_db,
            index_db,
            vector_indices: RwLock::new(HashMap::new()),
            storage_path: todos_path,
            seq_mutex: parking_lot::Mutex::new(()),
        })
    }

    /// Get or create a Vamana vector index for a user
    fn get_or_create_index(&self, user_id: &str) -> Result<()> {
        let mut indices = self.vector_indices.write();
        if !indices.contains_key(user_id) {
            let config = VamanaConfig {
                dimension: EMBEDDING_DIM,
                max_degree: 32,
                search_list_size: 75,
                alpha: 1.2,
                ..Default::default()
            };
            let index = VamanaIndex::new(config)?;
            indices.insert(user_id.to_string(), index);
        }
        Ok(())
    }

    /// Add or update a todo in the vector index
    /// Returns the vector ID assigned to this todo
    pub fn index_todo_embedding(
        &self,
        user_id: &str,
        _todo_id: &TodoId,
        embedding: &[f32],
    ) -> Result<u32> {
        self.get_or_create_index(user_id)?;

        let mut indices = self.vector_indices.write();
        if let Some(index) = indices.get_mut(user_id) {
            // Add vector and get assigned ID
            let vector_id = index.add_vector(embedding.to_vec())?;
            return Ok(vector_id);
        }
        anyhow::bail!("Failed to get vector index for user: {}", user_id)
    }

    /// Search for similar todos by embedding
    pub fn search_similar(
        &self,
        user_id: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(Todo, f32)>> {
        let indices = self.vector_indices.read();
        if let Some(index) = indices.get(user_id) {
            let results = index.search(query_embedding, limit)?;

            // Find todos by vector_id (stored in index_db)
            let mut todo_results = Vec::new();
            for (vector_id, score) in results {
                if let Some(todo) = self.get_todo_by_vector_id(user_id, vector_id)? {
                    todo_results.push((todo, score));
                }
            }
            Ok(todo_results)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get a todo by its vector index ID (stored in index_db)
    fn get_todo_by_vector_id(&self, user_id: &str, vector_id: u32) -> Result<Option<Todo>> {
        let key = format!("vector_id:{}:{}", user_id, vector_id);
        if let Some(data) = self.index_db.get(key.as_bytes())? {
            let todo_id_str = String::from_utf8_lossy(&data);
            if let Ok(uuid) = Uuid::parse_str(&todo_id_str) {
                return self.get_todo(user_id, &TodoId(uuid));
            }
        }
        Ok(None)
    }

    /// Store the mapping from vector_id to todo_id
    pub fn store_vector_id_mapping(
        &self,
        user_id: &str,
        vector_id: u32,
        todo_id: &TodoId,
    ) -> Result<()> {
        let key = format!("vector_id:{}:{}", user_id, vector_id);
        self.index_db
            .put(key.as_bytes(), todo_id.0.to_string().as_bytes())?;
        Ok(())
    }

    /// Save vector indices to disk
    pub fn save_vector_indices(&self) -> Result<()> {
        let indices = self.vector_indices.read();
        for (user_id, index) in indices.iter() {
            let index_path = self.storage_path.join("vectors").join(user_id);
            std::fs::create_dir_all(&index_path)?;
            index.save(&index_path)?;
        }
        Ok(())
    }

    /// Load vector indices from disk
    pub fn load_vector_indices(&self) -> Result<()> {
        let vectors_path = self.storage_path.join("vectors");
        if !vectors_path.exists() {
            return Ok(());
        }

        let mut indices = self.vector_indices.write();
        for entry in std::fs::read_dir(&vectors_path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let user_id = entry.file_name().to_string_lossy().to_string();
                let index_path = entry.path();

                // Create a new index and load from disk
                let config = VamanaConfig {
                    dimension: EMBEDDING_DIM,
                    ..Default::default()
                };
                let mut index = VamanaIndex::new(config)?;
                if index.load(&index_path).is_ok() {
                    indices.insert(user_id.clone(), index);
                    tracing::debug!("Loaded todo vector index for user: {}", user_id);
                }
            }
        }
        Ok(())
    }

    // =========================================================================
    // SEQUENCE NUMBER MANAGEMENT
    // =========================================================================

    /// Get the next sequence number for a project (or user if no project) and increment the counter
    /// Key format: "seq:{user_id}:{project_id}" or "seq:{user_id}:_standalone_" for todos without project
    fn next_seq_num(&self, user_id: &str, project_id: Option<&ProjectId>) -> Result<u32> {
        // Hold mutex to prevent TOCTOU race on concurrent seq_num allocation
        let _lock = self.seq_mutex.lock();
        let key = match project_id {
            Some(pid) => format!("seq:{}:{}", user_id, pid.0),
            None => format!("seq:{}:_standalone_", user_id),
        };
        let current = match self.index_db.get(key.as_bytes())? {
            Some(data) => {
                if data.len() >= 4 {
                    let bytes: [u8; 4] = [data[0], data[1], data[2], data[3]];
                    u32::from_le_bytes(bytes)
                } else {
                    0
                }
            }
            None => 0,
        };
        let next = current + 1;
        self.index_db.put(key.as_bytes(), next.to_le_bytes())?;
        Ok(next)
    }

    /// Assign a sequence number and project prefix to a todo if it doesn't have one
    pub fn assign_seq_num(&self, todo: &mut Todo) -> Result<()> {
        if todo.seq_num == 0 {
            // Set project prefix if todo has a project
            if let Some(ref project_id) = todo.project_id {
                if let Some(project) = self.get_project(&todo.user_id, project_id)? {
                    todo.project_prefix = Some(project.effective_prefix());
                }
            }
            todo.seq_num = self.next_seq_num(&todo.user_id, todo.project_id.as_ref())?;
            todo.sync_compat_fields();
        }
        Ok(())
    }

    // =========================================================================
    // TODO CRUD OPERATIONS
    // =========================================================================

    /// Store a new todo (assigns seq_num and project_prefix if needed, returns stored todo)
    pub fn store_todo(&self, todo: &Todo) -> Result<Todo> {
        // If seq_num is 0, assign one (for new todos)
        let mut todo_to_store = todo.clone();
        if todo_to_store.seq_num == 0 {
            // Set project prefix if todo has a project
            if let Some(ref project_id) = todo_to_store.project_id {
                if todo_to_store.project_prefix.is_none() {
                    if let Some(project) = self.get_project(&todo_to_store.user_id, project_id)? {
                        todo_to_store.project_prefix = Some(project.effective_prefix());
                    }
                }
            }
            todo_to_store.seq_num =
                self.next_seq_num(&todo_to_store.user_id, todo_to_store.project_id.as_ref())?;
        }
        todo_to_store.sync_compat_fields();

        let key = format!("{}:{}", todo_to_store.user_id, todo_to_store.id.0);
        let value = serde_json::to_vec(&todo_to_store).context("Failed to serialize todo")?;

        self.todo_db
            .put(key.as_bytes(), &value)
            .context("Failed to store todo")?;

        self.update_todo_indices(&todo_to_store)?;

        tracing::debug!(
            todo_id = %todo_to_store.id,
            short_id = %todo_to_store.short_id(),
            user_id = %todo_to_store.user_id,
            status = ?todo_to_store.status,
            "Stored todo"
        );

        Ok(todo_to_store)
    }

    /// Update todo indices
    fn update_todo_indices(&self, todo: &Todo) -> Result<()> {
        let mut batch = WriteBatch::default();
        let id_str = todo.id.0.to_string();

        // Index by user (for listing)
        let user_key = format!("user:{}:{}", todo.user_id, id_str);
        batch.put(user_key.as_bytes(), b"1");

        // Index by status
        let status_key = format!("status:{:?}:{}:{}", todo.status, todo.user_id, id_str);
        batch.put(status_key.as_bytes(), b"1");

        // Index by priority
        let priority_key = format!(
            "priority:{}:{}:{}",
            todo.priority.value(),
            todo.user_id,
            id_str
        );
        batch.put(priority_key.as_bytes(), b"1");

        // Index by project
        if let Some(ref project_id) = todo.project_id {
            let project_key = format!("project:{}:{}:{}", project_id.0, todo.user_id, id_str);
            batch.put(project_key.as_bytes(), b"1");
        }

        // Index by due date (zero-padded for correct lexicographic ordering)
        if let Some(ref due) = todo.due_date {
            let due_key = format!("due:{:020}:{}:{}", due.timestamp(), todo.user_id, id_str);
            batch.put(due_key.as_bytes(), b"1");
        }

        // Index by context
        for ctx in &todo.contexts {
            let ctx_key = format!("context:{}:{}:{}", ctx.to_lowercase(), todo.user_id, id_str);
            batch.put(ctx_key.as_bytes(), b"1");
        }

        // Index by parent (for subtasks)
        if let Some(ref parent_id) = todo.parent_id {
            let parent_key = format!("parent:{}:{}", parent_id.0, id_str);
            batch.put(parent_key.as_bytes(), todo.user_id.as_bytes());
        }

        self.index_db
            .write(batch)
            .context("Failed to update todo indices")?;

        Ok(())
    }

    /// Remove todo indices
    fn remove_todo_indices(&self, todo: &Todo) -> Result<()> {
        let mut batch = WriteBatch::default();
        let id_str = todo.id.0.to_string();

        let user_key = format!("user:{}:{}", todo.user_id, id_str);
        batch.delete(user_key.as_bytes());

        let status_key = format!("status:{:?}:{}:{}", todo.status, todo.user_id, id_str);
        batch.delete(status_key.as_bytes());

        let priority_key = format!(
            "priority:{}:{}:{}",
            todo.priority.value(),
            todo.user_id,
            id_str
        );
        batch.delete(priority_key.as_bytes());

        if let Some(ref project_id) = todo.project_id {
            let project_key = format!("project:{}:{}:{}", project_id.0, todo.user_id, id_str);
            batch.delete(project_key.as_bytes());
        }

        if let Some(ref due) = todo.due_date {
            let due_key = format!("due:{:020}:{}:{}", due.timestamp(), todo.user_id, id_str);
            batch.delete(due_key.as_bytes());
        }

        for ctx in &todo.contexts {
            let ctx_key = format!("context:{}:{}:{}", ctx.to_lowercase(), todo.user_id, id_str);
            batch.delete(ctx_key.as_bytes());
        }

        if let Some(ref parent_id) = todo.parent_id {
            let parent_key = format!("parent:{}:{}", parent_id.0, id_str);
            batch.delete(parent_key.as_bytes());
        }

        self.index_db.write(batch)?;
        Ok(())
    }

    /// Get a todo by ID
    pub fn get_todo(&self, user_id: &str, todo_id: &TodoId) -> Result<Option<Todo>> {
        let key = format!("{}:{}", user_id, todo_id.0);

        match self.todo_db.get(key.as_bytes())? {
            Some(value) => {
                let mut todo: Todo =
                    serde_json::from_slice(&value).context("Failed to deserialize todo")?;
                todo.sync_compat_fields();
                Ok(Some(todo))
            }
            None => Ok(None),
        }
    }

    /// Find todo by short ID prefix (e.g., "BOLT-1", "MEM-2", "SHO-3", or just "1")
    pub fn find_todo_by_prefix(&self, user_id: &str, prefix: &str) -> Result<Option<Todo>> {
        let todos = self.list_todos_for_user(user_id, None)?;

        // Parse prefix in format "PREFIX-NUMBER" or just "NUMBER"
        let prefix_upper = prefix.trim().to_uppercase();

        // Try to extract project prefix and sequence number
        if let Some((project_prefix, seq_str)) = prefix_upper.rsplit_once('-') {
            // Format: "BOLT-1", "MEM-2", "SHO-3"
            if let Ok(seq_num) = seq_str.parse::<u32>() {
                // Find todo matching both project prefix and seq_num
                if let Some(todo) = todos.iter().find(|t| {
                    t.seq_num == seq_num
                        && t.project_prefix
                            .as_ref()
                            .map(|p| p.to_uppercase() == project_prefix)
                            .unwrap_or(project_prefix == "SHO")
                }) {
                    return Ok(Some(todo.clone()));
                }
            }
        }

        // Try parsing as just a number (e.g., "1", "2")
        if let Ok(seq_num) = prefix_upper.parse::<u32>() {
            // Exact match on sequential number (any project)
            if let Some(todo) = todos.iter().find(|t| t.seq_num == seq_num) {
                return Ok(Some(todo.clone()));
            }
        }

        // Fall back to UUID prefix matching (for legacy todos)
        let clean_prefix_lower = prefix.to_lowercase();
        let matches: Vec<_> = todos
            .into_iter()
            .filter(|t| {
                t.id.0
                    .to_string()
                    .to_lowercase()
                    .starts_with(&clean_prefix_lower)
            })
            .collect();

        match matches.len() {
            0 => Ok(None),
            1 => Ok(Some(matches.into_iter().next().unwrap())),
            _ => {
                tracing::warn!(
                    user_id = %user_id,
                    prefix = %prefix,
                    matches = matches.len(),
                    "Multiple todos match prefix, using first"
                );
                Ok(Some(matches.into_iter().next().unwrap()))
            }
        }
    }

    /// Find todo by external ID (e.g., "todoist:123", "linear:SHO-39")
    /// Used for two-way sync with external todo/task management systems
    pub fn find_by_external_id(&self, user_id: &str, external_id: &str) -> Result<Option<Todo>> {
        let todos = self.list_todos_for_user(user_id, None)?;
        Ok(todos
            .into_iter()
            .find(|t| t.external_id.as_deref() == Some(external_id)))
    }

    /// Update a todo
    pub fn update_todo(&self, todo: &Todo) -> Result<()> {
        // Get old todo to remove old indices
        if let Some(old_todo) = self.get_todo(&todo.user_id, &todo.id)? {
            self.remove_todo_indices(&old_todo)?;
        }

        self.store_todo(todo).map(|_| ())
    }

    /// Delete a todo
    pub fn delete_todo(&self, user_id: &str, todo_id: &TodoId) -> Result<bool> {
        let key = format!("{}:{}", user_id, todo_id.0);

        if let Some(todo) = self.get_todo(user_id, todo_id)? {
            // Cascade delete subtasks to prevent orphans
            let subtasks = self.list_subtasks(todo_id)?;
            for subtask in &subtasks {
                self.remove_todo_indices(&subtask)?;
                let subtask_key = format!("{}:{}", subtask.user_id, subtask.id.0);
                self.todo_db.delete(subtask_key.as_bytes())?;
                tracing::debug!(
                    todo_id = %subtask.id,
                    parent_id = %todo_id,
                    "Cascade deleted subtask"
                );
            }

            self.remove_todo_indices(&todo)?;
            self.todo_db.delete(key.as_bytes())?;
            tracing::debug!(
                todo_id = %todo_id,
                subtasks_deleted = subtasks.len(),
                "Deleted todo"
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Complete a todo (marks as Done, handles recurrence)
    pub fn complete_todo(
        &self,
        user_id: &str,
        todo_id: &TodoId,
    ) -> Result<Option<(Todo, Option<Todo>)>> {
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            // Remove old indices
            self.remove_todo_indices(&todo)?;

            // Mark as complete
            todo.complete();

            // Store updated todo
            let stored_todo = self.store_todo(&todo)?;

            // Create next recurrence if applicable
            let next_todo = if let Some(next) = stored_todo.create_next_recurrence() {
                Some(self.store_todo(&next)?)
            } else {
                None
            };

            Ok(Some((stored_todo, next_todo)))
        } else {
            Ok(None)
        }
    }

    // =========================================================================
    // TODO COMMENTS
    // =========================================================================

    /// Add a comment to a todo
    pub fn add_comment(
        &self,
        user_id: &str,
        todo_id: &TodoId,
        author: String,
        content: String,
        comment_type: Option<TodoCommentType>,
    ) -> Result<Option<TodoComment>> {
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            let mut comment = TodoComment::new(todo_id.clone(), author, content);
            if let Some(ct) = comment_type {
                comment.comment_type = ct;
            }
            let comment_clone = comment.clone();
            todo.comments.push(comment);
            self.update_todo(&todo)?;

            tracing::debug!(
                todo_id = %todo_id,
                comment_id = %comment_clone.id.0,
                "Added comment to todo"
            );

            Ok(Some(comment_clone))
        } else {
            Ok(None)
        }
    }

    /// Add a system activity entry to a todo
    pub fn add_activity(&self, user_id: &str, todo_id: &TodoId, content: String) -> Result<bool> {
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            todo.add_activity(content);
            self.update_todo(&todo)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Update a comment on a todo
    pub fn update_comment(
        &self,
        user_id: &str,
        todo_id: &TodoId,
        comment_id: &TodoCommentId,
        content: String,
    ) -> Result<Option<TodoComment>> {
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            if let Some(comment) = todo.comments.iter_mut().find(|c| c.id == *comment_id) {
                comment.content = content;
                comment.updated_at = Some(chrono::Utc::now());
                let comment_clone = comment.clone();
                self.update_todo(&todo)?;
                Ok(Some(comment_clone))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Delete a comment from a todo
    pub fn delete_comment(
        &self,
        user_id: &str,
        todo_id: &TodoId,
        comment_id: &TodoCommentId,
    ) -> Result<bool> {
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            let initial_len = todo.comments.len();
            todo.comments.retain(|c| c.id != *comment_id);
            if todo.comments.len() < initial_len {
                self.update_todo(&todo)?;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    /// Get all comments for a todo
    pub fn get_comments(&self, user_id: &str, todo_id: &TodoId) -> Result<Vec<TodoComment>> {
        if let Some(todo) = self.get_todo(user_id, todo_id)? {
            Ok(todo.comments)
        } else {
            Ok(Vec::new())
        }
    }

    /// Reorder a todo within its status group
    /// direction: "up" moves earlier in list (lower sort_order), "down" moves later
    pub fn reorder_todo(
        &self,
        user_id: &str,
        todo_id: &TodoId,
        direction: &str,
    ) -> Result<Option<Todo>> {
        let todo = match self.get_todo(user_id, todo_id)? {
            Some(t) => t,
            None => return Ok(None),
        };

        // Get all todos with the same status
        let mut same_status_todos: Vec<Todo> = self
            .list_todos_for_user(user_id, Some(&[todo.status.clone()]))?
            .into_iter()
            .collect();

        // Sort by sort_order to get current ordering
        same_status_todos.sort_by_key(|t| t.sort_order);

        // Find current position
        let pos = same_status_todos
            .iter()
            .position(|t| t.id == *todo_id)
            .unwrap_or(0);

        let swap_pos = match direction {
            "up" => {
                if pos == 0 {
                    return Ok(Some(todo)); // Already at top
                }
                pos - 1
            }
            "down" => {
                if pos >= same_status_todos.len() - 1 {
                    return Ok(Some(todo)); // Already at bottom
                }
                pos + 1
            }
            _ => return Ok(Some(todo)), // Invalid direction
        };

        // Swap sort_order values with adjacent todo
        let mut current = same_status_todos[pos].clone();
        let mut adjacent = same_status_todos[swap_pos].clone();

        std::mem::swap(&mut current.sort_order, &mut adjacent.sort_order);

        // Update both todos
        current.updated_at = Utc::now();
        adjacent.updated_at = Utc::now();

        self.update_todo(&current)?;
        self.update_todo(&adjacent)?;

        Ok(Some(current))
    }

    // =========================================================================
    // TODO QUERIES
    // =========================================================================

    /// List todos for a user with optional status filter
    pub fn list_todos_for_user(
        &self,
        user_id: &str,
        status_filter: Option<&[TodoStatus]>,
    ) -> Result<Vec<Todo>> {
        let prefix = format!("user:{}:", user_id);
        let mut todos = Vec::new();

        let iter = self.index_db.prefix_iterator(prefix.as_bytes());

        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            // Extract todo_id from key "user:{user_id}:{todo_id}"
            let todo_id_str = key_str.strip_prefix(&prefix).unwrap_or("");
            if let Ok(uuid) = Uuid::parse_str(todo_id_str) {
                let todo_id = TodoId(uuid);
                if let Some(todo) = self.get_todo(user_id, &todo_id)? {
                    // Apply status filter
                    if let Some(statuses) = status_filter {
                        if statuses.contains(&todo.status) {
                            todos.push(todo);
                        }
                    } else {
                        todos.push(todo);
                    }
                }
            }
        }

        // Sort by: sort_order (manual), then priority, then due date
        todos.sort_by(|a, b| {
            // First by sort_order (lower = higher in list)
            let order_cmp = a.sort_order.cmp(&b.sort_order);
            if order_cmp != std::cmp::Ordering::Equal {
                return order_cmp;
            }
            // Then by priority
            let priority_cmp = a.priority.value().cmp(&b.priority.value());
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }
            // Finally by due date
            match (&a.due_date, &b.due_date) {
                (Some(a_due), Some(b_due)) => a_due.cmp(b_due),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });

        Ok(todos)
    }

    /// List todos by project
    pub fn list_todos_by_project(
        &self,
        user_id: &str,
        project_id: &ProjectId,
    ) -> Result<Vec<Todo>> {
        let prefix = format!("project:{}:{}:", project_id.0, user_id);
        let mut todos = Vec::new();

        let iter = self.index_db.prefix_iterator(prefix.as_bytes());

        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let todo_id_str = key_str.strip_prefix(&prefix).unwrap_or("");
            if let Ok(uuid) = Uuid::parse_str(todo_id_str) {
                if let Some(todo) = self.get_todo(user_id, &TodoId(uuid))? {
                    todos.push(todo);
                }
            }
        }

        Ok(todos)
    }

    /// List todos by context (e.g., @computer)
    pub fn list_todos_by_context(&self, user_id: &str, context: &str) -> Result<Vec<Todo>> {
        let ctx_lower = context.to_lowercase();
        let prefix = format!("context:{}:{}:", ctx_lower, user_id);
        let mut todos = Vec::new();

        let iter = self.index_db.prefix_iterator(prefix.as_bytes());

        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let todo_id_str = key_str.strip_prefix(&prefix).unwrap_or("");
            if let Ok(uuid) = Uuid::parse_str(todo_id_str) {
                if let Some(todo) = self.get_todo(user_id, &TodoId(uuid))? {
                    todos.push(todo);
                }
            }
        }

        Ok(todos)
    }

    /// List due/overdue todos
    pub fn list_due_todos(&self, user_id: &str, include_overdue: bool) -> Result<Vec<Todo>> {
        let now = Utc::now();
        let end_of_today = now
            .date_naive()
            .and_hms_opt(23, 59, 59)
            .map(|t| t.and_utc())
            .unwrap_or(now);

        let todos = self.list_todos_for_user(user_id, None)?;

        let due_todos: Vec<_> = todos
            .into_iter()
            .filter(|t| {
                if t.status == TodoStatus::Done || t.status == TodoStatus::Cancelled {
                    return false;
                }
                if let Some(due) = &t.due_date {
                    if include_overdue && *due < now {
                        return true;
                    }
                    *due <= end_of_today
                } else {
                    false
                }
            })
            .collect();

        Ok(due_todos)
    }

    /// List subtasks of a parent todo
    pub fn list_subtasks(&self, parent_id: &TodoId) -> Result<Vec<Todo>> {
        let prefix = format!("parent:{}:", parent_id.0);
        let mut todos = Vec::new();

        let iter = self.index_db.prefix_iterator(prefix.as_bytes());

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let todo_id_str = key_str.strip_prefix(&prefix).unwrap_or("");
            let user_id = String::from_utf8_lossy(&value);

            if let Ok(uuid) = Uuid::parse_str(todo_id_str) {
                if let Some(todo) = self.get_todo(&user_id, &TodoId(uuid))? {
                    todos.push(todo);
                }
            }
        }

        Ok(todos)
    }

    // =========================================================================
    // PROJECT CRUD OPERATIONS
    // =========================================================================

    /// Store a project
    pub fn store_project(&self, project: &Project) -> Result<()> {
        let key = format!("{}:{}", project.user_id, project.id.0);
        let value = serde_json::to_vec(project).context("Failed to serialize project")?;

        self.project_db.put(key.as_bytes(), &value)?;

        // Index by user
        let user_key = format!("user:{}:{}", project.user_id, project.id.0);
        self.index_db.put(user_key.as_bytes(), b"p")?; // 'p' for project

        // Index by name (for lookup) - store as string for easy parsing
        let name_key = format!(
            "project_name:{}:{}",
            project.name.to_lowercase(),
            project.user_id
        );
        self.index_db
            .put(name_key.as_bytes(), project.id.0.to_string().as_bytes())?;

        // Index by parent (for sub-projects)
        if let Some(ref parent_id) = project.parent_id {
            let parent_key = format!(
                "project_parent:{}:{}:{}",
                project.user_id, parent_id.0, project.id.0
            );
            self.index_db.put(parent_key.as_bytes(), b"1")?;
        }

        tracing::debug!(project_id = %project.id.0, name = %project.name, parent = ?project.parent_id, "Stored project");

        Ok(())
    }

    /// Get a project by ID
    pub fn get_project(&self, user_id: &str, project_id: &ProjectId) -> Result<Option<Project>> {
        let key = format!("{}:{}", user_id, project_id.0);

        match self.project_db.get(key.as_bytes())? {
            Some(value) => {
                let project: Project =
                    serde_json::from_slice(&value).context("Failed to deserialize project")?;
                Ok(Some(project))
            }
            None => Ok(None),
        }
    }

    /// Find project by name
    pub fn find_project_by_name(&self, user_id: &str, name: &str) -> Result<Option<Project>> {
        let name_key = format!("project_name:{}:{}", name.to_lowercase(), user_id);

        if let Some(value) = self.index_db.get(name_key.as_bytes())? {
            if let Ok(uuid) = Uuid::parse_str(&String::from_utf8_lossy(&value)) {
                return self.get_project(user_id, &ProjectId(uuid));
            }
        }

        Ok(None)
    }

    /// Find or create project by name
    pub fn find_or_create_project(&self, user_id: &str, name: &str) -> Result<Project> {
        if let Some(project) = self.find_project_by_name(user_id, name)? {
            return Ok(project);
        }

        let project = Project::new(user_id.to_string(), name.to_string());
        self.store_project(&project)?;
        Ok(project)
    }

    /// List projects for a user
    pub fn list_projects(&self, user_id: &str) -> Result<Vec<Project>> {
        let mut projects = Vec::new();

        let iter = self
            .project_db
            .prefix_iterator(format!("{}:", user_id).as_bytes());

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&format!("{}:", user_id)) {
                break;
            }

            let project: Project = serde_json::from_slice(&value)?;
            projects.push(project);
        }

        // Sort by name
        projects.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

        Ok(projects)
    }

    /// List sub-projects of a parent project
    pub fn list_subprojects(&self, user_id: &str, parent_id: &ProjectId) -> Result<Vec<Project>> {
        let mut subprojects = Vec::new();

        let prefix = format!("project_parent:{}:{}:", user_id, parent_id.0);
        let iter = self.index_db.prefix_iterator(prefix.as_bytes());

        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            // Extract project ID from key
            let parts: Vec<&str> = key_str.split(':').collect();
            if parts.len() >= 4 {
                if let Ok(uuid) = Uuid::parse_str(parts[3]) {
                    if let Some(project) = self.get_project(user_id, &ProjectId(uuid))? {
                        subprojects.push(project);
                    }
                }
            }
        }

        // Sort by name
        subprojects.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

        Ok(subprojects)
    }

    /// Get project with todo counts
    pub fn get_project_stats(&self, user_id: &str, project_id: &ProjectId) -> Result<ProjectStats> {
        let todos = self.list_todos_by_project(user_id, project_id)?;

        let mut stats = ProjectStats::default();
        for todo in &todos {
            match todo.status {
                TodoStatus::Backlog => stats.backlog += 1,
                TodoStatus::Todo => stats.todo += 1,
                TodoStatus::InProgress => stats.in_progress += 1,
                TodoStatus::Blocked => stats.blocked += 1,
                TodoStatus::Done => stats.done += 1,
                TodoStatus::Cancelled => stats.cancelled += 1,
            }
        }
        stats.total = todos.len();

        Ok(stats)
    }

    /// Update a project's properties
    pub fn update_project(
        &self,
        user_id: &str,
        project_id: &ProjectId,
        name: Option<String>,
        prefix: Option<String>,
        description: Option<Option<String>>,
        status: Option<ProjectStatus>,
        color: Option<Option<String>>,
    ) -> Result<Option<Project>> {
        if let Some(mut project) = self.get_project(user_id, project_id)? {
            let old_name = project.name.clone();
            let mut changed = false;

            if let Some(new_name) = name {
                if !new_name.trim().is_empty() && new_name != project.name {
                    project.name = new_name;
                    changed = true;
                }
            }

            if let Some(new_prefix) = prefix {
                let clean = new_prefix.trim().to_uppercase();
                if !clean.is_empty() {
                    project.prefix = Some(clean);
                    changed = true;
                }
            }

            if let Some(new_description) = description {
                project.description = new_description;
                changed = true;
            }

            if let Some(new_status) = status {
                if new_status != project.status {
                    project.status = new_status.clone();
                    changed = true;

                    // Set completed_at when archiving or completing
                    if new_status == ProjectStatus::Completed
                        || new_status == ProjectStatus::Archived
                    {
                        project.completed_at = Some(Utc::now());
                    } else {
                        project.completed_at = None;
                    }
                }
            }

            if let Some(new_color) = color {
                project.color = new_color;
                changed = true;
            }

            if changed {
                // Update name index if name changed
                if project.name != old_name {
                    let old_name_key =
                        format!("project_name:{}:{}", old_name.to_lowercase(), user_id);
                    self.index_db.delete(old_name_key.as_bytes())?;
                }

                self.store_project(&project)?;
            }

            Ok(Some(project))
        } else {
            Ok(None)
        }
    }

    /// Delete a project (and optionally its todos)
    pub fn delete_project(
        &self,
        user_id: &str,
        project_id: &ProjectId,
        delete_todos: bool,
    ) -> Result<bool> {
        if let Some(project) = self.get_project(user_id, project_id)? {
            // Delete todos if requested
            if delete_todos {
                let todos = self.list_todos_by_project(user_id, project_id)?;
                for todo in todos {
                    self.delete_todo(user_id, &todo.id)?;
                }
            }

            // Delete sub-projects recursively
            let subprojects = self.list_subprojects(user_id, project_id)?;
            for subproject in subprojects {
                self.delete_project(user_id, &subproject.id, delete_todos)?;
            }

            // Delete project
            let key = format!("{}:{}", user_id, project_id.0);
            self.project_db.delete(key.as_bytes())?;

            // Delete indices
            let user_key = format!("user:{}:{}", user_id, project_id.0);
            self.index_db.delete(user_key.as_bytes())?;

            let name_key = format!("project_name:{}:{}", project.name.to_lowercase(), user_id);
            self.index_db.delete(name_key.as_bytes())?;

            // Delete parent index (if this was a sub-project)
            if let Some(ref parent_id) = project.parent_id {
                let parent_key = format!(
                    "project_parent:{}:{}:{}",
                    user_id, parent_id.0, project_id.0
                );
                self.index_db.delete(parent_key.as_bytes())?;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    // =========================================================================
    // STATS
    // =========================================================================

    /// Flush all RocksDB databases and save vector indices to disk (critical for graceful shutdown)
    pub fn flush(&self) -> Result<()> {
        // Flush RocksDB databases
        self.todo_db
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush todo_db: {e}"))?;
        self.project_db
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush project_db: {e}"))?;
        self.index_db
            .flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush index_db: {e}"))?;

        // Save vector indices
        self.save_vector_indices()
            .map_err(|e| anyhow::anyhow!("Failed to save todo vector indices: {e}"))?;

        Ok(())
    }

    /// Get references to all RocksDB databases for backup
    pub fn databases(&self) -> Vec<(&str, &Arc<DB>)> {
        vec![
            ("todo_items", &self.todo_db),
            ("todo_projects", &self.project_db),
            ("todo_index", &self.index_db),
        ]
    }

    /// Get overall todo stats for a user
    pub fn get_user_stats(&self, user_id: &str) -> Result<UserTodoStats> {
        let todos = self.list_todos_for_user(user_id, None)?;

        let mut stats = UserTodoStats::default();

        for todo in &todos {
            stats.total += 1;
            match todo.status {
                TodoStatus::Backlog => stats.backlog += 1,
                TodoStatus::Todo => stats.todo += 1,
                TodoStatus::InProgress => stats.in_progress += 1,
                TodoStatus::Blocked => stats.blocked += 1,
                TodoStatus::Done => stats.done += 1,
                TodoStatus::Cancelled => stats.cancelled += 1,
            }

            if todo.is_overdue() {
                stats.overdue += 1;
            }
            if todo.is_due_today() {
                stats.due_today += 1;
            }
        }

        stats.projects = self.list_projects(user_id)?.len();

        Ok(stats)
    }
}

/// Stats for a single project
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProjectStats {
    pub total: usize,
    pub backlog: usize,
    pub todo: usize,
    pub in_progress: usize,
    pub blocked: usize,
    pub done: usize,
    pub cancelled: usize,
}

/// Overall todo stats for a user
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct UserTodoStats {
    pub total: usize,
    pub backlog: usize,
    pub todo: usize,
    pub in_progress: usize,
    pub blocked: usize,
    pub done: usize,
    pub cancelled: usize,
    pub overdue: usize,
    pub due_today: usize,
    pub projects: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::Recurrence;
    use tempfile::TempDir;

    fn setup_store() -> (TodoStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let store = TodoStore::new(temp_dir.path()).unwrap();
        (store, temp_dir)
    }

    #[test]
    fn test_create_and_get_todo() {
        let (store, _temp) = setup_store();

        let todo = Todo::new("test_user".to_string(), "Test task".to_string());
        store.store_todo(&todo).unwrap();

        let retrieved = store.get_todo("test_user", &todo.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test task");
    }

    #[test]
    fn test_find_by_prefix() {
        let (store, _temp) = setup_store();

        let todo = Todo::new("test_user".to_string(), "Test task".to_string());
        // store_todo assigns seq_num and returns the updated todo
        let stored = store.store_todo(&todo).unwrap();

        // Use short_id() which returns "SHO-1" format (sequence-based)
        let short_id = stored.short_id();

        // Find by full SHO-N format
        let found = store.find_todo_by_prefix("test_user", &short_id).unwrap();
        assert!(
            found.is_some(),
            "Should find by full short_id: {}",
            short_id
        );

        // Find by just the sequence number
        let seq_str = stored.seq_num.to_string();
        let found2 = store.find_todo_by_prefix("test_user", &seq_str).unwrap();
        assert!(found2.is_some(), "Should find by seq_num: {}", seq_str);

        // Also test UUID prefix fallback for legacy compatibility
        let uuid_prefix = &stored.id.0.to_string()[..8];
        let found3 = store.find_todo_by_prefix("test_user", uuid_prefix).unwrap();
        assert!(
            found3.is_some(),
            "Should find by UUID prefix: {}",
            uuid_prefix
        );
    }

    #[test]
    fn test_due_key_migration_and_ordering() {
        let temp_dir = TempDir::new().unwrap();
        let todos_path = temp_dir.path().join("todos");
        std::fs::create_dir_all(&todos_path).unwrap();

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let index_db = DB::open(&opts, todos_path.join("index")).unwrap();

        let todo_id_a = Uuid::new_v4();
        let todo_id_b = Uuid::new_v4();
        // ts_a = 9 (1 digit), ts_b = 10 (2 digits)
        // Without padding: "due:9:..." > "due:10:..." lexicographically (wrong)
        index_db
            .put(format!("due:9:user1:{}", todo_id_a).as_bytes(), b"1")
            .unwrap();
        index_db
            .put(format!("due:10:user1:{}", todo_id_b).as_bytes(), b"1")
            .unwrap();

        // Run migration
        let migrated = migrate_due_key_padding(&index_db).unwrap();
        assert_eq!(migrated, 2);

        // Verify old keys are gone
        assert!(index_db
            .get(format!("due:9:user1:{}", todo_id_a).as_bytes())
            .unwrap()
            .is_none());
        assert!(index_db
            .get(format!("due:10:user1:{}", todo_id_b).as_bytes())
            .unwrap()
            .is_none());

        // Verify new padded keys exist
        let key_a = format!("due:{:020}:user1:{}", 9_i64, todo_id_a);
        let key_b = format!("due:{:020}:user1:{}", 10_i64, todo_id_b);
        assert!(index_db.get(key_a.as_bytes()).unwrap().is_some());
        assert!(index_db.get(key_b.as_bytes()).unwrap().is_some());

        // Verify lexicographic order is now correct: 9 < 10
        assert!(
            key_a < key_b,
            "Padded key for ts=9 should sort before ts=10"
        );

        // Re-running migration should be a no-op
        let migrated_again = migrate_due_key_padding(&index_db).unwrap();
        assert_eq!(migrated_again, 0);
    }

    #[test]
    fn test_complete_todo() {
        let (store, _temp) = setup_store();

        let todo = Todo::new("test_user".to_string(), "Test task".to_string());
        store.store_todo(&todo).unwrap();

        let result = store.complete_todo("test_user", &todo.id).unwrap();
        assert!(result.is_some());

        let (completed, _next) = result.unwrap();
        assert_eq!(completed.status, TodoStatus::Done);
        assert!(completed.completed_at.is_some());
    }

    #[test]
    fn test_recurring_todo() {
        let (store, _temp) = setup_store();

        let mut todo = Todo::new("test_user".to_string(), "Daily task".to_string());
        todo.recurrence = Some(Recurrence::Daily);
        todo.due_date = Some(Utc::now());
        store.store_todo(&todo).unwrap();

        let result = store.complete_todo("test_user", &todo.id).unwrap();
        assert!(result.is_some());

        let (completed, next) = result.unwrap();
        assert_eq!(completed.status, TodoStatus::Done);
        assert!(next.is_some());

        let next_todo = next.unwrap();
        assert_eq!(next_todo.status, TodoStatus::Todo);
        assert!(next_todo.due_date.unwrap() > completed.due_date.unwrap());
    }

    #[test]
    fn test_project_crud() {
        let (store, _temp) = setup_store();

        let project = Project::new("test_user".to_string(), "Test Project".to_string());
        store.store_project(&project).unwrap();

        let found = store
            .find_project_by_name("test_user", "test project")
            .unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Test Project");
    }

    #[test]
    fn test_list_by_status() {
        let (store, _temp) = setup_store();

        let mut todo1 = Todo::new("test_user".to_string(), "Task 1".to_string());
        todo1.status = TodoStatus::InProgress;

        let mut todo2 = Todo::new("test_user".to_string(), "Task 2".to_string());
        todo2.status = TodoStatus::Backlog;

        store.store_todo(&todo1).unwrap();
        store.store_todo(&todo2).unwrap();

        let in_progress = store
            .list_todos_for_user("test_user", Some(&[TodoStatus::InProgress]))
            .unwrap();
        assert_eq!(in_progress.len(), 1);
        assert_eq!(in_progress[0].content, "Task 1");
    }
}
