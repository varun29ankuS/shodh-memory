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
//! - Semantic search via exact cosine scan over embeddings persisted on each
//!   todo (MiniLM-L6-v2, 384-dim), plus a lexical substring path that guarantees
//!   exact word matches even for todos without embeddings.
//! - Structured dependencies (`blocked_by`) with cycle rejection
//! - Bidirectional memory links (`related_memory_ids`)

use anyhow::{bail, Context, Result};
use chrono::Utc;
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, WriteBatch, DB};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use super::types::{
    MemoryId, Project, ProjectId, ProjectStatus, Todo, TodoComment, TodoCommentId, TodoCommentType,
    TodoId, TodoStatus,
};

/// Minimum cosine similarity for a todo to count as a semantic match.
/// Below this, results are noise: an exact scan with no floor would return
/// `limit` todos for ANY query, which misleads callers as badly as returning
/// none. Lexical substring matches are exempt from this floor.
pub const MIN_SEMANTIC_SIMILARITY: f32 = 0.3;

const CF_TODOS: &str = "todos";
const CF_PROJECTS: &str = "projects";
const CF_TODO_INDEX: &str = "todo_index";

/// Migrate unpadded `due:{ts}:{uid}:{id}` keys to zero-padded `due:{:020}:{uid}:{id}` format.
///
/// Prior versions wrote bare timestamps (e.g. `due:1739404800:user:uuid`), which break
/// lexicographic ordering (`"9" > "10"`). Zero-padding to 20 digits ensures
/// lex order = chronological order, enabling ordered range scans.
fn migrate_due_key_padding(db: &DB, index_cf: &ColumnFamily) -> Result<usize> {
    let mut batch = WriteBatch::default();
    let mut count = 0;

    for item in db.prefix_iterator_cf(index_cf, b"due:") {
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
            batch.delete_cf(index_cf, &*key);
            batch.put_cf(index_cf, new_key.as_bytes(), &*value);
            count += 1;
        }
    }

    if count > 0 {
        db.write(batch)
            .context("Failed to write migrated todo due keys")?;
        tracing::info!(count, "Migrated todo due keys to zero-padded format");
    }

    Ok(count)
}

/// Storage and query engine for todos and projects
pub struct TodoStore {
    /// Shared RocksDB instance with column families for todos, projects, and indices
    db: Arc<DB>,
    /// Storage path (used for legacy vector-index cleanup during user purge)
    storage_path: std::path::PathBuf,
    /// Mutex for atomic sequence number allocation (prevents TOCTOU race)
    seq_mutex: parking_lot::Mutex<()>,
    /// Serializes every read-modify-write of a todo record.
    ///
    /// Handlers hold no per-todo lock, so an async memory link-back after
    /// `remember()`, a comment and an activity entry can all land on the same
    /// todo at once; each one reads the whole record, changes a field and
    /// writes the whole record back, so whichever read first is erased.
    ///
    /// It must be taken by ALL of them. It previously guarded only
    /// `add_related_memory`, which serialized that method against itself and
    /// against nothing else — a lock one writer takes and the others ignore
    /// prevents no interleaving at all.
    ///
    /// Process-local is sufficient rather than a shortcut: RocksDB holds an
    /// exclusive file lock, so one process has this DB open and every todo
    /// writer goes through this struct. NOT reentrant, so each method that both
    /// takes the lock and is reachable from a method that already holds it is
    /// split in two: a `pub` wrapper that locks, and a private `_locked` inner
    /// the lock-holders call (`update_todo`/`update_todo_locked`,
    /// `settle_todo`/`settle_todo_locked`). `get_todo`, `store_todo` and the
    /// index helpers never take it and are safe from either side.
    ///
    /// # What this lock does NOT cover
    ///
    /// `update_todo`, `settle_todo` and `store_todo` are handed a whole record
    /// the CALLER composed. Locking them serializes the write and keeps a
    /// record and its indices consistent, but the caller's own read happened
    /// outside the lock, so a concurrent commit made after that read is still
    /// overwritten. `PUT /api/todos/{id}/update` is exactly this shape — it
    /// reads the todo, awaits an embedding, then writes — and closing it needs
    /// the mutation expressed as a closure applied under the lock
    /// (`MemoryStorage::modify`'s shape), not a lock added here. The eight
    /// methods above are immune because their read is INSIDE the lock.
    mutation_mutex: parking_lot::Mutex<()>,
}

impl TodoStore {
    /// Column family descriptors required by the TodoStore.
    /// The caller must include these (plus `"default"`) when opening the shared DB.
    pub fn cf_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut cf_opts = Options::default();
        cf_opts.create_if_missing(true);
        cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        vec![
            ColumnFamilyDescriptor::new(CF_TODOS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_PROJECTS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_TODO_INDEX, cf_opts),
        ]
    }

    fn todos_cf(&self) -> &ColumnFamily {
        self.db.cf_handle(CF_TODOS).expect("todos CF must exist")
    }
    fn projects_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_PROJECTS)
            .expect("projects CF must exist")
    }
    fn todo_index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_TODO_INDEX)
            .expect("todo_index CF must exist")
    }

    /// Create a new todo store backed by the given shared DB
    pub fn new(db: Arc<DB>, storage_path: &Path) -> Result<Self> {
        let todos_path = storage_path.join("todos");
        std::fs::create_dir_all(&todos_path)?;

        // Migrate from old separate-DB layout if needed
        Self::migrate_from_separate_dbs(&todos_path, &db)?;

        // Migrate any unpadded due keys from prior versions
        let index_cf = db
            .cf_handle(CF_TODO_INDEX)
            .expect("todo_index CF must exist");
        migrate_due_key_padding(&db, index_cf)?;

        tracing::info!("Todo store initialized");

        Ok(Self {
            db,
            storage_path: todos_path,
            seq_mutex: parking_lot::Mutex::new(()),
            mutation_mutex: parking_lot::Mutex::new(()),
        })
    }

    /// Migrate data from the old separate-DB layout (items/, projects/, index/ sub-dirs)
    /// into the unified column-family DB. After migration, old dirs are renamed to
    /// `{name}.pre_cf_migration` so the migration is idempotent.
    fn migrate_from_separate_dbs(todos_path: &Path, db: &DB) -> Result<()> {
        let old_dirs: &[(&str, &str)] = &[
            ("items", CF_TODOS),
            ("projects", CF_PROJECTS),
            ("index", CF_TODO_INDEX),
        ];

        for (old_name, cf_name) in old_dirs {
            let old_dir = todos_path.join(old_name);
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
                    for (key, value) in old_db.iterator(rocksdb::IteratorMode::Start).flatten() {
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
                    tracing::info!("  todos/{old_name}: migrated {count} entries to {cf_name} CF");

                    let backup = todos_path.join(format!("{old_name}.pre_cf_migration"));
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

    /// Cosine similarity between two vectors. Returns None on dimension
    /// mismatch or zero-norm inputs (treated as "no semantic signal").
    fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() || a.is_empty() {
            return None;
        }
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
            return None;
        }
        Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
    }

    /// Search for similar todos by embedding.
    ///
    /// Exact cosine scan over the embeddings persisted on each todo record.
    /// This deliberately replaces the previous in-memory Vamana side-index,
    /// which was never persisted in production (its save path had no callers),
    /// so every restart silently emptied it and semantic todo search returned
    /// nothing. The embedding on the todo record is the single source of truth;
    /// an exact scan over it cannot go stale. Todo counts are human-scale, and
    /// other hot paths (`find_todo_by_prefix`) already scan all of a user's
    /// todos, so this matches the store's existing performance profile.
    ///
    /// Results are clamped to [0, 1] and filtered by [`MIN_SEMANTIC_SIMILARITY`].
    pub fn search_similar(
        &self,
        user_id: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(Todo, f32)>> {
        let todos = self.list_todos_for_user(user_id, None)?;
        let mut scored: Vec<(Todo, f32)> = todos
            .into_iter()
            .filter_map(|todo| {
                let emb = todo.embedding.as_deref()?;
                let sim = Self::cosine_similarity(query_embedding, emb)?;
                let sim = sim.clamp(0.0, 1.0);
                if sim >= MIN_SEMANTIC_SIMILARITY {
                    Some((todo, sim))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    /// Hybrid todo search: semantic (cosine over persisted embeddings) unioned
    /// with lexical substring matching over content, notes, and tags.
    ///
    /// The lexical path guarantees that exact word matches ALWAYS surface —
    /// including todos that have no embedding (created while the embedding
    /// model was unavailable) and queries whose embedding could not be
    /// computed (`query_embedding = None`).
    ///
    /// Ordering: lexical matches first (a literal hit is the strongest signal
    /// for a task lookup), then by semantic score descending.
    pub fn search_todos(
        &self,
        user_id: &str,
        query: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
    ) -> Result<Vec<(Todo, f32)>> {
        let query_lower = query.trim().to_lowercase();
        if query_lower.is_empty() {
            return Ok(Vec::new());
        }
        let query_tokens: Vec<&str> = query_lower.split_whitespace().collect();

        let todos = self.list_todos_for_user(user_id, None)?;
        let mut results: Vec<(Todo, f32, bool)> = Vec::new();

        for todo in todos {
            // Searchable text: content + notes + tags (mirrors what the
            // embedding is computed from)
            let haystack = format!(
                "{} {} {}",
                todo.content,
                todo.notes.as_deref().unwrap_or(""),
                todo.tags.join(" ")
            )
            .to_lowercase();

            // A lexical hit is either the full query as a substring, or every
            // query word present somewhere (word-order independent).
            let lexical_hit = haystack.contains(&query_lower)
                || query_tokens.iter().all(|tok| haystack.contains(tok));

            let semantic_score = query_embedding
                .and_then(|q| {
                    todo.embedding
                        .as_deref()
                        .and_then(|e| Self::cosine_similarity(q, e))
                })
                .map(|s| s.clamp(0.0, 1.0));

            let semantic_hit = semantic_score.is_some_and(|s| s >= MIN_SEMANTIC_SIMILARITY);

            if lexical_hit || semantic_hit {
                results.push((todo, semantic_score.unwrap_or(0.0), lexical_hit));
            }
        }

        // Lexical hits first, then semantic score descending
        results.sort_by(|a, b| {
            b.2.cmp(&a.2)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        results.truncate(limit);

        Ok(results
            .into_iter()
            .map(|(todo, score, _)| (todo, score))
            .collect())
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
        let current = match self.db.get_cf(self.todo_index_cf(), key.as_bytes())? {
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
        self.db
            .put_cf(self.todo_index_cf(), key.as_bytes(), next.to_le_bytes())?;
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

        self.db
            .put_cf(self.todos_cf(), key.as_bytes(), &value)
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
        let index_cf = self.todo_index_cf();

        // Index by user (for listing)
        let user_key = format!("user:{}:{}", todo.user_id, id_str);
        batch.put_cf(index_cf, user_key.as_bytes(), b"1");

        // Index by status
        let status_key = format!("status:{:?}:{}:{}", todo.status, todo.user_id, id_str);
        batch.put_cf(index_cf, status_key.as_bytes(), b"1");

        // Index by priority
        let priority_key = format!(
            "priority:{}:{}:{}",
            todo.priority.value(),
            todo.user_id,
            id_str
        );
        batch.put_cf(index_cf, priority_key.as_bytes(), b"1");

        // Index by project
        if let Some(ref project_id) = todo.project_id {
            let project_key = format!("project:{}:{}:{}", project_id.0, todo.user_id, id_str);
            batch.put_cf(index_cf, project_key.as_bytes(), b"1");
        }

        // Index by due date (zero-padded for correct lexicographic ordering)
        if let Some(ref due) = todo.due_date {
            let due_key = format!("due:{:020}:{}:{}", due.timestamp(), todo.user_id, id_str);
            batch.put_cf(index_cf, due_key.as_bytes(), b"1");
        }

        // Index by context
        for ctx in &todo.contexts {
            let ctx_key = format!("context:{}:{}:{}", ctx.to_lowercase(), todo.user_id, id_str);
            batch.put_cf(index_cf, ctx_key.as_bytes(), b"1");
        }

        // Index by parent (for subtasks)
        if let Some(ref parent_id) = todo.parent_id {
            let parent_key = format!("parent:{}:{}", parent_id.0, id_str);
            batch.put_cf(index_cf, parent_key.as_bytes(), todo.user_id.as_bytes());
        }

        self.db
            .write(batch)
            .context("Failed to update todo indices")?;

        Ok(())
    }

    /// Remove todo indices and clean up vector embeddings
    fn remove_todo_indices(&self, todo: &Todo) -> Result<()> {
        let mut batch = WriteBatch::default();
        let id_str = todo.id.0.to_string();
        let index_cf = self.todo_index_cf();

        let user_key = format!("user:{}:{}", todo.user_id, id_str);
        batch.delete_cf(index_cf, user_key.as_bytes());

        let status_key = format!("status:{:?}:{}:{}", todo.status, todo.user_id, id_str);
        batch.delete_cf(index_cf, status_key.as_bytes());

        let priority_key = format!(
            "priority:{}:{}:{}",
            todo.priority.value(),
            todo.user_id,
            id_str
        );
        batch.delete_cf(index_cf, priority_key.as_bytes());

        if let Some(ref project_id) = todo.project_id {
            let project_key = format!("project:{}:{}:{}", project_id.0, todo.user_id, id_str);
            batch.delete_cf(index_cf, project_key.as_bytes());
        }

        if let Some(ref due) = todo.due_date {
            let due_key = format!("due:{:020}:{}:{}", due.timestamp(), todo.user_id, id_str);
            batch.delete_cf(index_cf, due_key.as_bytes());
        }

        for ctx in &todo.contexts {
            let ctx_key = format!("context:{}:{}:{}", ctx.to_lowercase(), todo.user_id, id_str);
            batch.delete_cf(index_cf, ctx_key.as_bytes());
        }

        if let Some(ref parent_id) = todo.parent_id {
            let parent_key = format!("parent:{}:{}", parent_id.0, id_str);
            batch.delete_cf(index_cf, parent_key.as_bytes());
        }

        // Legacy hygiene: earlier versions kept a Vamana side-index with
        // vector-id mappings in the index CF. Remove any leftover mapping keys
        // for this todo so old stores stay clean. (Orphaned `vector_id:*` keys
        // from crashes are harmless and are swept by user purge.)
        let rev_key = format!("todo_vector:{}:{}", todo.user_id, id_str);
        if let Some(vid_bytes) = self.db.get_cf(index_cf, rev_key.as_bytes())? {
            if vid_bytes.len() >= 4 {
                let vector_id =
                    u32::from_le_bytes([vid_bytes[0], vid_bytes[1], vid_bytes[2], vid_bytes[3]]);
                let fwd_key = format!("vector_id:{}:{}", todo.user_id, vector_id);
                batch.delete_cf(index_cf, fwd_key.as_bytes());
            }
        }
        batch.delete_cf(index_cf, rev_key.as_bytes());

        self.db.write(batch)?;
        Ok(())
    }

    /// Get a todo by ID
    pub fn get_todo(&self, user_id: &str, todo_id: &TodoId) -> Result<Option<Todo>> {
        let key = format!("{}:{}", user_id, todo_id.0);

        match self.db.get_cf(self.todos_cf(), key.as_bytes())? {
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

    /// Update a todo.
    ///
    /// Takes `mutation_mutex`: the body reads the stored record to drop its old
    /// indices and then writes, so without the lock that pair interleaves with
    /// the eight locked writers and leaves the indices describing a record that
    /// was never stored.
    ///
    /// This serializes the WRITE, not the caller's read. A caller that read a
    /// todo earlier, changed a field and passes the whole record here still
    /// overwrites anything committed in between — the snapshot is stale before
    /// this function is entered. The lock cannot fix that; only reading inside
    /// the lock can, which is what the eight methods above do.
    pub fn update_todo(&self, todo: &Todo) -> Result<()> {
        let _lock = self.mutation_mutex.lock();
        self.update_todo_locked(todo)
    }

    /// [`Self::update_todo`] for callers already holding `mutation_mutex`.
    /// The mutex is NOT reentrant, so taking it again here would deadlock.
    fn update_todo_locked(&self, todo: &Todo) -> Result<()> {
        // Get old todo to remove old indices
        if let Some(old_todo) = self.get_todo(&todo.user_id, &todo.id)? {
            self.remove_todo_indices(&old_todo)?;
        }

        self.store_todo(todo).map(|_| ())
    }

    /// Delete a todo.
    ///
    /// Takes `mutation_mutex`. The read and the deletes have to be one step:
    /// otherwise a comment writer that read this todo just before the delete
    /// re-stores it afterwards, resurrecting both the record and the indices
    /// this call removed. Its only in-crate caller, `delete_project`, holds no
    /// lock, so there is no reentrancy here.
    pub fn delete_todo(&self, user_id: &str, todo_id: &TodoId) -> Result<bool> {
        let _lock = self.mutation_mutex.lock();
        let key = format!("{}:{}", user_id, todo_id.0);

        if let Some(todo) = self.get_todo(user_id, todo_id)? {
            // Cascade delete subtasks to prevent orphans
            let subtasks = self.list_subtasks(todo_id)?;
            for subtask in &subtasks {
                self.remove_todo_indices(subtask)?;
                let subtask_key = format!("{}:{}", subtask.user_id, subtask.id.0);
                self.db.delete_cf(self.todos_cf(), subtask_key.as_bytes())?;
                tracing::debug!(
                    todo_id = %subtask.id,
                    parent_id = %todo_id,
                    "Cascade deleted subtask"
                );
            }

            self.remove_todo_indices(&todo)?;
            self.db.delete_cf(self.todos_cf(), key.as_bytes())?;
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
    ///
    /// Read-modify-write, so it takes `mutation_mutex` like every other one:
    /// a completion racing the async memory link-back would otherwise write
    /// back a record that predates the link and drop it.
    pub fn complete_todo(
        &self,
        user_id: &str,
        todo_id: &TodoId,
    ) -> Result<Option<(Todo, Option<Todo>)>> {
        let _lock = self.mutation_mutex.lock();
        match self.get_todo(user_id, todo_id)? {
            Some(mut todo) => {
                todo.complete();
                Ok(Some(self.settle_todo_locked(&todo)?))
            }
            None => Ok(None),
        }
    }

    /// Persist a todo that has just entered a settled state and spawn its next
    /// occurrence if it recurs.
    ///
    /// This is the single settlement write path: `complete_todo` and the
    /// update handler both come through here, so a completion cannot lose its
    /// recurrence rollover depending on which door the client used.
    ///
    /// Only Done rolls over. Cancelled means "not doing this", not "done with
    /// this one, see you next time".
    ///
    /// `todo` must already carry its settled status and stamp — use
    /// [`Todo::apply_status`] or [`Todo::complete`] before calling.
    ///
    /// Takes `mutation_mutex` for the same reason [`Self::update_todo`] does:
    /// the update door reaches settlement through here directly, so without the
    /// lock the read-then-write below would interleave with the locked writers
    /// while `/complete` — which arrives through `complete_todo` — is
    /// serialized. Same caveat too: this serializes the write, not the caller's
    /// earlier read of `todo`.
    pub fn settle_todo(&self, todo: &Todo) -> Result<(Todo, Option<Todo>)> {
        let _lock = self.mutation_mutex.lock();
        self.settle_todo_locked(todo)
    }

    /// [`Self::settle_todo`] for callers already holding `mutation_mutex`.
    /// The mutex is NOT reentrant, so taking it again here would deadlock.
    fn settle_todo_locked(&self, todo: &Todo) -> Result<(Todo, Option<Todo>)> {
        if let Some(previous) = self.get_todo(&todo.user_id, &todo.id)? {
            self.remove_todo_indices(&previous)?;
        }

        let settled = self.store_todo(todo)?;

        let next = if settled.status == TodoStatus::Done {
            match settled.create_next_recurrence() {
                Some(next) => Some(self.store_todo(&next)?),
                None => None,
            }
        } else {
            None
        };

        Ok((settled, next))
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
        let _lock = self.mutation_mutex.lock();
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            let mut comment = TodoComment::new(todo_id.clone(), author, content);
            if let Some(ct) = comment_type {
                comment.comment_type = ct;
            }
            let comment_clone = comment.clone();
            todo.comments.push(comment);
            self.update_todo_locked(&todo)?;

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
        let _lock = self.mutation_mutex.lock();
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            todo.add_activity(content);
            self.update_todo_locked(&todo)?;
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
        let _lock = self.mutation_mutex.lock();
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            if let Some(comment) = todo.comments.iter_mut().find(|c| c.id == *comment_id) {
                comment.content = content;
                comment.updated_at = Some(chrono::Utc::now());
                let comment_clone = comment.clone();
                self.update_todo_locked(&todo)?;
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
        let _lock = self.mutation_mutex.lock();
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            let initial_len = todo.comments.len();
            todo.comments.retain(|c| c.id != *comment_id);
            if todo.comments.len() < initial_len {
                self.update_todo_locked(&todo)?;
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

    // =========================================================================
    // MEMORY LINKS ("why does this task exist")
    // =========================================================================

    /// Link a memory to a todo (idempotent). Serialized via `mutation_mutex`
    /// against every other read-modify-write on the same todo, so the async
    /// link-back after `remember()` cannot lose one or be lost.
    /// Returns true if the todo exists (whether or not the link was new).
    pub fn add_related_memory(
        &self,
        user_id: &str,
        todo_id: &TodoId,
        memory_id: MemoryId,
    ) -> Result<bool> {
        let _lock = self.mutation_mutex.lock();
        if let Some(mut todo) = self.get_todo(user_id, todo_id)? {
            if !todo.has_related_memory(&memory_id) {
                todo.add_related_memory(memory_id);
                self.update_todo_locked(&todo)?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Drop every reference to `memory_id` from this user's todos.
    ///
    /// Called when a memory is deleted. Without it the link is one-way
    /// durable: `verify_memory_ids` proves a memory exists when the link is
    /// written and deleting a todo takes its links with it, but deleting the
    /// MEMORY left `related_memory_ids` pointing at nothing and `get_todo`
    /// went on rendering "Linked memories: <uuid>" for a memory that is gone.
    ///
    /// Returns the number of todos changed. Idempotent, and scoped to one
    /// user — memory ids are per-user, so another user's identical link is
    /// none of this call's business.
    pub fn remove_memory_links(&self, user_id: &str, memory_id: &MemoryId) -> Result<usize> {
        let _lock = self.mutation_mutex.lock();
        let todos = self.list_todos_for_user(user_id, None)?;
        let mut changed = 0;
        for mut todo in todos {
            if !todo.has_related_memory(memory_id) {
                continue;
            }
            todo.remove_related_memory(memory_id);
            self.update_todo_locked(&todo)?;
            changed += 1;
        }
        if changed > 0 {
            tracing::debug!(
                user_id = %user_id,
                memory_id = %memory_id.0,
                todos_updated = changed,
                "Cleared todo links to a deleted memory"
            );
        }
        Ok(changed)
    }

    // =========================================================================
    // STRUCTURED DEPENDENCIES (blocked_by)
    // =========================================================================

    /// Check whether setting `new_blockers` on `todo_id` would create a
    /// dependency cycle. Walks the `blocked_by` graph from each proposed
    /// blocker; if any chain reaches back to `todo_id`, the edge set is cyclic.
    pub fn would_create_dependency_cycle(
        &self,
        user_id: &str,
        todo_id: &TodoId,
        new_blockers: &[TodoId],
    ) -> Result<bool> {
        use std::collections::{HashSet, VecDeque};

        if new_blockers.iter().any(|b| b == todo_id) {
            return Ok(true); // Self-dependency
        }

        let mut visited: HashSet<Uuid> = HashSet::new();
        let mut queue: VecDeque<TodoId> = new_blockers.iter().cloned().collect();

        while let Some(current) = queue.pop_front() {
            if current == *todo_id {
                return Ok(true);
            }
            if !visited.insert(current.0) {
                continue;
            }
            if let Some(todo) = self.get_todo(user_id, &current)? {
                for blocker in &todo.blocked_by {
                    queue.push_back(blocker.clone());
                }
            }
        }
        Ok(false)
    }

    /// Walk the full blocking chain for a todo: everything it transitively
    /// waits on, in BFS order (direct blockers first). Cycle-safe.
    pub fn blocking_chain(&self, user_id: &str, todo_id: &TodoId) -> Result<Vec<Todo>> {
        use std::collections::{HashSet, VecDeque};

        let mut visited: HashSet<Uuid> = HashSet::new();
        visited.insert(todo_id.0);
        let mut chain = Vec::new();
        let mut queue: VecDeque<TodoId> = match self.get_todo(user_id, todo_id)? {
            Some(t) => t.blocked_by.iter().cloned().collect(),
            None => return Ok(Vec::new()),
        };

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.0) {
                continue;
            }
            if let Some(todo) = self.get_todo(user_id, &current)? {
                for blocker in &todo.blocked_by {
                    queue.push_back(blocker.clone());
                }
                chain.push(todo);
            }
        }
        Ok(chain)
    }

    /// Todos that become unblocked by completing `completed_id`: they list it
    /// in `blocked_by` and every OTHER blocker is already Done/Cancelled (or
    /// deleted). Used to surface "you can now start X" on completion.
    pub fn unblocked_by_completion(
        &self,
        user_id: &str,
        completed_id: &TodoId,
    ) -> Result<Vec<Todo>> {
        let todos = self.list_todos_for_user(user_id, None)?;
        let mut unblocked = Vec::new();

        'outer: for todo in todos {
            if todo.status == TodoStatus::Done || todo.status == TodoStatus::Cancelled {
                continue;
            }
            if !todo.blocked_by.contains(completed_id) {
                continue;
            }
            for blocker_id in &todo.blocked_by {
                if blocker_id == completed_id {
                    continue;
                }
                if let Some(blocker) = self.get_todo(user_id, blocker_id)? {
                    if blocker.status != TodoStatus::Done && blocker.status != TodoStatus::Cancelled
                    {
                        continue 'outer; // Still blocked by something else
                    }
                }
            }
            unblocked.push(todo);
        }
        Ok(unblocked)
    }

    /// Reorder a todo within its status group
    /// direction: "up" moves earlier in list (lower sort_order), "down" moves later
    /// Read-modify-write across TWO todos (the pair whose `sort_order` is
    /// swapped), so it takes `mutation_mutex` for the same reason as the
    /// single-record mutators — and additionally so the two writes cannot be
    /// interleaved with a third party's, which would leave duplicate or
    /// skipped sort orders.
    pub fn reorder_todo(
        &self,
        user_id: &str,
        todo_id: &TodoId,
        direction: &str,
    ) -> Result<Option<Todo>> {
        let _lock = self.mutation_mutex.lock();
        let todo = match self.get_todo(user_id, todo_id)? {
            Some(t) => t,
            None => return Ok(None),
        };

        // Get all todos with the same status
        let mut same_status_todos: Vec<Todo> = self
            .list_todos_for_user(user_id, Some(std::slice::from_ref(&todo.status)))?
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
            other => bail!(
                "Invalid reorder direction '{}'. Valid values: up, down",
                other
            ),
        };

        // Swap sort_order values with adjacent todo
        let mut current = same_status_todos[pos].clone();
        let mut adjacent = same_status_todos[swap_pos].clone();

        std::mem::swap(&mut current.sort_order, &mut adjacent.sort_order);

        // Update both todos
        current.updated_at = Utc::now();
        adjacent.updated_at = Utc::now();

        self.update_todo_locked(&current)?;
        self.update_todo_locked(&adjacent)?;

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

        let iter = self
            .db
            .prefix_iterator_cf(self.todo_index_cf(), prefix.as_bytes());

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

        let iter = self
            .db
            .prefix_iterator_cf(self.todo_index_cf(), prefix.as_bytes());

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

        let iter = self
            .db
            .prefix_iterator_cf(self.todo_index_cf(), prefix.as_bytes());

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

        let iter = self
            .db
            .prefix_iterator_cf(self.todo_index_cf(), prefix.as_bytes());

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

        self.db.put_cf(self.projects_cf(), key.as_bytes(), &value)?;

        // Index by user
        let user_key = format!("user:{}:{}", project.user_id, project.id.0);
        self.db
            .put_cf(self.todo_index_cf(), user_key.as_bytes(), b"p")?; // 'p' for project

        // Index by name (for lookup) - store as string for easy parsing
        let name_key = format!(
            "project_name:{}:{}",
            project.name.to_lowercase(),
            project.user_id
        );
        self.db.put_cf(
            self.todo_index_cf(),
            name_key.as_bytes(),
            project.id.0.to_string().as_bytes(),
        )?;

        // Index by parent (for sub-projects)
        if let Some(ref parent_id) = project.parent_id {
            let parent_key = format!(
                "project_parent:{}:{}:{}",
                project.user_id, parent_id.0, project.id.0
            );
            self.db
                .put_cf(self.todo_index_cf(), parent_key.as_bytes(), b"1")?;
        }

        tracing::debug!(project_id = %project.id.0, name = %project.name, parent = ?project.parent_id, "Stored project");

        Ok(())
    }

    /// Get a project by ID
    pub fn get_project(&self, user_id: &str, project_id: &ProjectId) -> Result<Option<Project>> {
        let key = format!("{}:{}", user_id, project_id.0);

        match self.db.get_cf(self.projects_cf(), key.as_bytes())? {
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

        if let Some(value) = self.db.get_cf(self.todo_index_cf(), name_key.as_bytes())? {
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
            .db
            .prefix_iterator_cf(self.projects_cf(), format!("{}:", user_id).as_bytes());

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
        let iter = self
            .db
            .prefix_iterator_cf(self.todo_index_cf(), prefix.as_bytes());

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
    #[allow(clippy::too_many_arguments)]
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
                    self.db
                        .delete_cf(self.todo_index_cf(), old_name_key.as_bytes())?;
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
            self.db.delete_cf(self.projects_cf(), key.as_bytes())?;

            // Delete indices
            let user_key = format!("user:{}:{}", user_id, project_id.0);
            self.db
                .delete_cf(self.todo_index_cf(), user_key.as_bytes())?;

            let name_key = format!("project_name:{}:{}", project.name.to_lowercase(), user_id);
            self.db
                .delete_cf(self.todo_index_cf(), name_key.as_bytes())?;

            // Delete parent index (if this was a sub-project)
            if let Some(ref parent_id) = project.parent_id {
                let parent_key = format!(
                    "project_parent:{}:{}:{}",
                    user_id, parent_id.0, project_id.0
                );
                self.db
                    .delete_cf(self.todo_index_cf(), parent_key.as_bytes())?;
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Purge a user's legacy on-disk vector index files (GDPR user deletion).
    /// Embeddings now live only on the todo records themselves, which the
    /// shared-DB purge deletes; this removes index files written by earlier
    /// versions that kept a separate Vamana side-index per user.
    pub fn purge_user_vectors(&self, user_id: &str) {
        let legacy_dir = self.storage_path.join("vectors").join(user_id);
        if legacy_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&legacy_dir) {
                tracing::warn!(
                    user_id = %user_id,
                    error = %e,
                    "Failed to remove legacy todo vector index dir during purge"
                );
            }
        }
    }

    // =========================================================================
    // STATS
    // =========================================================================

    /// Flush all RocksDB column families to disk (critical for graceful shutdown)
    pub fn flush(&self) -> Result<()> {
        use rocksdb::FlushOptions;
        let mut flush_opts = FlushOptions::default();
        flush_opts.set_wait(true);

        for cf_name in &[CF_TODOS, CF_PROJECTS, CF_TODO_INDEX] {
            if let Some(cf) = self.db.cf_handle(cf_name) {
                self.db
                    .flush_cf_opt(cf, &flush_opts)
                    .map_err(|e| anyhow::anyhow!("Failed to flush {cf_name}: {e}"))?;
            }
        }

        Ok(())
    }

    /// Get reference to the shared RocksDB database for backup
    pub fn databases(&self) -> Vec<(&str, &Arc<DB>)> {
        vec![("todos_shared", &self.db)]
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

    fn open_test_shared_db(path: &Path) -> Arc<DB> {
        let shared_path = path.join("shared");
        std::fs::create_dir_all(&shared_path).unwrap();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        let cfs = vec![
            ColumnFamilyDescriptor::new("default", opts.clone()),
            ColumnFamilyDescriptor::new(CF_TODOS, opts.clone()),
            ColumnFamilyDescriptor::new(CF_PROJECTS, opts.clone()),
            ColumnFamilyDescriptor::new(CF_TODO_INDEX, opts.clone()),
        ];
        Arc::new(DB::open_cf_descriptors(&opts, &shared_path, cfs).unwrap())
    }

    fn setup_store() -> (TodoStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db = open_test_shared_db(temp_dir.path());
        let store = TodoStore::new(db, temp_dir.path()).unwrap();
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
        let db = open_test_shared_db(temp_dir.path());
        let index_cf = db.cf_handle(CF_TODO_INDEX).unwrap();

        let todo_id_a = Uuid::new_v4();
        let todo_id_b = Uuid::new_v4();
        // ts_a = 9 (1 digit), ts_b = 10 (2 digits)
        // Without padding: "due:9:..." > "due:10:..." lexicographically (wrong)
        db.put_cf(
            index_cf,
            format!("due:9:user1:{}", todo_id_a).as_bytes(),
            b"1",
        )
        .unwrap();
        db.put_cf(
            index_cf,
            format!("due:10:user1:{}", todo_id_b).as_bytes(),
            b"1",
        )
        .unwrap();

        // Run migration
        let migrated = migrate_due_key_padding(&db, index_cf).unwrap();
        assert_eq!(migrated, 2);

        // Verify old keys are gone
        assert!(db
            .get_cf(index_cf, format!("due:9:user1:{}", todo_id_a).as_bytes())
            .unwrap()
            .is_none());
        assert!(db
            .get_cf(index_cf, format!("due:10:user1:{}", todo_id_b).as_bytes())
            .unwrap()
            .is_none());

        // Verify new padded keys exist
        let key_a = format!("due:{:020}:user1:{}", 9_i64, todo_id_a);
        let key_b = format!("due:{:020}:user1:{}", 10_i64, todo_id_b);
        assert!(db.get_cf(index_cf, key_a.as_bytes()).unwrap().is_some());
        assert!(db.get_cf(index_cf, key_b.as_bytes()).unwrap().is_some());

        // Verify lexicographic order is now correct: 9 < 10
        assert!(
            key_a < key_b,
            "Padded key for ts=9 should sort before ts=10"
        );

        // Re-running migration should be a no-op
        let migrated_again = migrate_due_key_padding(&db, index_cf).unwrap();
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

    // =========================================================================
    // SEMANTIC + LEXICAL SEARCH
    // =========================================================================

    /// Regression for the dead `list_todos({query})` path: semantic search must
    /// read the embeddings persisted on the todo records — including through a
    /// process restart. The previous Vamana side-index was never saved in
    /// production, so every restart silently emptied it and search returned
    /// nothing.
    #[test]
    fn test_search_similar_survives_store_reopen() {
        let temp_dir = TempDir::new().unwrap();
        let db = open_test_shared_db(temp_dir.path());

        let mut todo_a = Todo::new("test_user".to_string(), "Deploy the API server".to_string());
        todo_a.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
        let mut todo_b = Todo::new("test_user".to_string(), "Buy groceries".to_string());
        todo_b.embedding = Some(vec![0.0, 1.0, 0.0, 0.0]);

        {
            let store = TodoStore::new(db.clone(), temp_dir.path()).unwrap();
            store.store_todo(&todo_a).unwrap();
            store.store_todo(&todo_b).unwrap();

            let results = store
                .search_similar("test_user", &[0.9, 0.1, 0.0, 0.0], 10)
                .unwrap();
            // todo_a: cosine ≈ 0.99 (match); todo_b: cosine ≈ 0.11 (below floor)
            assert_eq!(results.len(), 1, "only the todo above the floor matches");
            assert_eq!(results[0].0.content, "Deploy the API server");
            assert!(results[0].1 > MIN_SEMANTIC_SIMILARITY);
        }

        // Simulate a process restart: a brand-new TodoStore over the same DB.
        // No in-memory state may be required for search to work.
        let reopened = TodoStore::new(db, temp_dir.path()).unwrap();
        let results = reopened
            .search_similar("test_user", &[0.9, 0.1, 0.0, 0.0], 10)
            .unwrap();
        assert!(
            !results.is_empty(),
            "semantic search must survive a restart (embeddings are persisted on the todos)"
        );
        assert_eq!(results[0].0.content, "Deploy the API server");
    }

    #[test]
    fn test_search_similar_applies_similarity_floor() {
        let (store, _temp) = setup_store();

        let mut todo = Todo::new("test_user".to_string(), "Unrelated task".to_string());
        todo.embedding = Some(vec![0.0, 0.0, 1.0, 0.0]);
        store.store_todo(&todo).unwrap();

        // Orthogonal query: cosine 0.0 < MIN_SEMANTIC_SIMILARITY
        let results = store
            .search_similar("test_user", &[1.0, 0.0, 0.0, 0.0], 10)
            .unwrap();
        assert!(
            results.is_empty(),
            "matches below the similarity floor must not surface"
        );
    }

    #[test]
    fn test_search_todos_lexical_guarantees_exact_match() {
        let (store, _temp) = setup_store();

        // No embedding at all — created while the embedding model was down
        let todo = Todo::new("test_user".to_string(), "Audit-2 parent task".to_string());
        store.store_todo(&todo).unwrap();

        // No query embedding either — lexical path must still find it
        let results = store
            .search_todos("test_user", "Audit-2 parent", None, 10)
            .unwrap();
        assert_eq!(results.len(), 1, "exact word match must always surface");
        assert_eq!(results[0].0.content, "Audit-2 parent task");

        // Case-insensitive, and matches notes and tags too
        let mut tagged = Todo::new("test_user".to_string(), "Second task".to_string());
        tagged.tags = vec!["release-blocker".to_string()];
        store.store_todo(&tagged).unwrap();

        let by_tag = store
            .search_todos("test_user", "RELEASE-BLOCKER", None, 10)
            .unwrap();
        assert_eq!(by_tag.len(), 1);
        assert_eq!(by_tag[0].0.content, "Second task");

        // Word-order independent: all query tokens present counts as a hit
        let reordered = store
            .search_todos("test_user", "parent Audit-2", None, 10)
            .unwrap();
        assert_eq!(reordered.len(), 1, "token match must be order-independent");
        assert_eq!(reordered[0].0.content, "Audit-2 parent task");
    }

    #[test]
    fn test_search_todos_lexical_ranks_before_semantic() {
        let (store, _temp) = setup_store();

        let mut semantic_only = Todo::new("test_user".to_string(), "Related work".to_string());
        semantic_only.embedding = Some(vec![1.0, 0.0]);
        store.store_todo(&semantic_only).unwrap();

        let mut lexical_hit = Todo::new("test_user".to_string(), "Fix login bug".to_string());
        lexical_hit.embedding = Some(vec![0.8, 0.6]);
        store.store_todo(&lexical_hit).unwrap();

        let results = store
            .search_todos("test_user", "login", Some(&[1.0, 0.0]), 10)
            .unwrap();
        assert_eq!(results[0].0.content, "Fix login bug", "literal hit first");
    }

    // =========================================================================
    // REORDER VALIDATION
    // =========================================================================

    #[test]
    fn test_reorder_invalid_direction_rejected() {
        let (store, _temp) = setup_store();

        let todo = Todo::new("test_user".to_string(), "Task".to_string());
        store.store_todo(&todo).unwrap();

        let err = store
            .reorder_todo("test_user", &todo.id, "sideways")
            .unwrap_err();
        assert!(
            err.to_string().contains("Valid values: up, down"),
            "invalid direction must be an error, not a silent no-op: {err}"
        );
        // Valid directions still work
        assert!(store
            .reorder_todo("test_user", &todo.id, "up")
            .unwrap()
            .is_some());
        assert!(store
            .reorder_todo("test_user", &todo.id, "down")
            .unwrap()
            .is_some());
    }

    // =========================================================================
    // MEMORY LINKS
    // =========================================================================

    #[test]
    fn test_add_related_memory_idempotent() {
        let (store, _temp) = setup_store();

        let todo = Todo::new("test_user".to_string(), "Linked task".to_string());
        store.store_todo(&todo).unwrap();

        let mem_id = MemoryId(Uuid::new_v4());
        assert!(store
            .add_related_memory("test_user", &todo.id, mem_id.clone())
            .unwrap());
        assert!(store
            .add_related_memory("test_user", &todo.id, mem_id.clone())
            .unwrap());

        let stored = store.get_todo("test_user", &todo.id).unwrap().unwrap();
        assert_eq!(stored.related_memory_ids, vec![mem_id]);

        // Unknown todo → false, not an error
        assert!(!store
            .add_related_memory("test_user", &TodoId::new(), MemoryId(Uuid::new_v4()))
            .unwrap());
    }

    // =========================================================================
    // STRUCTURED DEPENDENCIES
    // =========================================================================

    #[test]
    fn test_dependency_cycle_detection() {
        let (store, _temp) = setup_store();

        let mut a = Todo::new("test_user".to_string(), "A".to_string());
        let b = Todo::new("test_user".to_string(), "B".to_string());
        let c = Todo::new("test_user".to_string(), "C".to_string());

        // A depends on B
        a.blocked_by = vec![b.id.clone()];
        store.store_todo(&a).unwrap();
        store.store_todo(&b).unwrap();
        store.store_todo(&c).unwrap();

        // Self-dependency is a cycle
        assert!(store
            .would_create_dependency_cycle("test_user", &a.id, &[a.id.clone()])
            .unwrap());
        // B → A would close the loop (A already waits on B)
        assert!(store
            .would_create_dependency_cycle("test_user", &b.id, &[a.id.clone()])
            .unwrap());
        // C → A is fine (no path from A back to C)
        assert!(!store
            .would_create_dependency_cycle("test_user", &c.id, &[a.id.clone()])
            .unwrap());
    }

    #[test]
    fn test_blocking_chain_walk() {
        let (store, _temp) = setup_store();

        let mut a = Todo::new("test_user".to_string(), "A".to_string());
        let mut b = Todo::new("test_user".to_string(), "B".to_string());
        let c = Todo::new("test_user".to_string(), "C".to_string());

        b.blocked_by = vec![c.id.clone()];
        a.blocked_by = vec![b.id.clone()];
        store.store_todo(&a).unwrap();
        store.store_todo(&b).unwrap();
        store.store_todo(&c).unwrap();

        let chain = store.blocking_chain("test_user", &a.id).unwrap();
        let contents: Vec<&str> = chain.iter().map(|t| t.content.as_str()).collect();
        assert_eq!(contents, vec!["B", "C"], "BFS: direct blocker first");
    }

    #[test]
    fn test_unblocked_by_completion() {
        let (store, _temp) = setup_store();

        let mut a = Todo::new("test_user".to_string(), "A".to_string());
        let b = Todo::new("test_user".to_string(), "B".to_string());
        let c = Todo::new("test_user".to_string(), "C".to_string());

        a.blocked_by = vec![b.id.clone(), c.id.clone()];
        store.store_todo(&a).unwrap();
        store.store_todo(&b).unwrap();
        store.store_todo(&c).unwrap();

        // Completing B alone does not unblock A (C is still open)
        store.complete_todo("test_user", &b.id).unwrap();
        let after_b = store.unblocked_by_completion("test_user", &b.id).unwrap();
        assert!(after_b.is_empty(), "A still waits on C");

        // Completing C releases A
        store.complete_todo("test_user", &c.id).unwrap();
        let after_c = store.unblocked_by_completion("test_user", &c.id).unwrap();
        assert_eq!(after_c.len(), 1);
        assert_eq!(after_c[0].content, "A");
    }

    // =========================================================================
    // PERSISTED-SHAPE COMPATIBILITY
    // =========================================================================

    /// A todo serialized by a pre-`blocked_by` build must deserialize cleanly
    /// (new fields default) and survive a write-back round trip. Todos are
    /// stored as JSON, so unknown/missing fields are tolerated — this test
    /// pins that contract against the exact bytes an old build produced.
    #[test]
    fn test_old_bytes_round_trip() {
        let old_json = r#"{
            "id": "5f2b7a86-3e64-4f0e-9c86-2f9f9a3d1b11",
            "seq_num": 7,
            "project_prefix": "MEM",
            "project": "MEM",
            "user_id": "test_user",
            "content": "Legacy todo from an old build",
            "status": "in_progress",
            "priority": "high",
            "project_id": "0e6f4a92-8f4b-4f0a-b0d9-6a3c62f5a001",
            "parent_id": null,
            "contexts": ["@computer"],
            "tags": ["legacy"],
            "due_date": "2026-01-15T23:59:59Z",
            "recurrence": null,
            "blocked_on": "vendor response",
            "notes": "created before blocked_by existed",
            "created_at": "2026-01-01T10:00:00Z",
            "updated_at": "2026-01-02T11:00:00Z",
            "completed_at": null,
            "sort_order": 3,
            "comments": [],
            "related_memory_ids": []
        }"#;

        let mut todo: Todo = serde_json::from_str(old_json).expect("old bytes must deserialize");
        assert_eq!(todo.content, "Legacy todo from an old build");
        assert_eq!(todo.seq_num, 7);
        assert!(todo.blocked_by.is_empty(), "new field defaults to empty");
        assert!(todo.embedding.is_none());
        assert_eq!(todo.blocked_on.as_deref(), Some("vendor response"));

        // Write-back through the store and read again
        let (store, _temp) = setup_store();
        todo.sync_compat_fields();
        store.store_todo(&todo).unwrap();
        let reread = store.get_todo("test_user", &todo.id).unwrap().unwrap();
        assert_eq!(reread.content, todo.content);
        assert_eq!(reread.seq_num, 7);
        assert!(reread.blocked_by.is_empty());

        // And the new serialized form carries the new field explicitly
        let new_json = serde_json::to_string(&reread).unwrap();
        assert!(new_json.contains("\"blocked_by\":[]"));
    }

    // =========================================================================
    // CONCURRENT READ-MODIFY-WRITE
    // =========================================================================

    /// `link_mutex` claimed to stop the async link-back after `remember()`
    /// losing a concurrent update, but it was one-sided: only
    /// `add_related_memory` ever took it. Every other read-modify-write on a
    /// todo — comments, activity — read and wrote outside it, so a link and a
    /// comment landing at the same time still lost one of the two. A mutex one
    /// writer takes and the other ignores serializes nothing.
    ///
    /// Every writer here mutates an append-only collection, so nothing may be
    /// lost: four memory links, four comments and four activity entries must
    /// all survive (activity is stored as a system comment, so twelve writes
    /// leave four links and eight comments).
    #[test]
    fn concurrent_link_and_comment_writes_all_survive() {
        let temp_dir = TempDir::new().unwrap();
        let db = open_test_shared_db(temp_dir.path());
        let store = Arc::new(TodoStore::new(db, temp_dir.path()).unwrap());

        let todo = Todo::new("test_user".to_string(), "Contended todo".to_string());
        store.store_todo(&todo).unwrap();

        let mut handles = Vec::new();
        for _ in 0..4 {
            let store = Arc::clone(&store);
            let id = todo.id.clone();
            handles.push(std::thread::spawn(move || {
                store
                    .add_related_memory("test_user", &id, MemoryId(Uuid::new_v4()))
                    .expect("add_related_memory");
            }));
        }
        for i in 0..4 {
            let store = Arc::clone(&store);
            let id = todo.id.clone();
            handles.push(std::thread::spawn(move || {
                store
                    .add_activity("test_user", &id, format!("activity {i}"))
                    .expect("add_activity");
            }));
        }
        for i in 0..4 {
            let store = Arc::clone(&store);
            let id = todo.id.clone();
            handles.push(std::thread::spawn(move || {
                store
                    .add_comment(
                        "test_user",
                        &id,
                        "tester".to_string(),
                        format!("comment {i}"),
                        None,
                    )
                    .expect("add_comment");
            }));
        }
        for h in handles {
            h.join().expect("writer thread");
        }

        let stored = store.get_todo("test_user", &todo.id).unwrap().unwrap();
        assert_eq!(
            stored.related_memory_ids.len(),
            4,
            "a memory link was lost to a concurrent todo write"
        );
        // Activity entries are stored as system comments, so all eight
        // appends land in the same collection.
        assert_eq!(
            stored.comments.len(),
            8,
            "a comment or activity entry was lost to a concurrent todo write"
        );
    }

    // =========================================================================
    // MEMORY DELETION LEAVES NO DANGLING LINKS
    // =========================================================================

    /// Deleting a memory used to leave every `todo.related_memory_ids` entry
    /// pointing at nothing: the todo -> memory direction is verified on write
    /// (`verify_memory_ids` in the handler) and repaired when a todo is
    /// deleted, but nothing ran in the other direction, so `get_todo` kept
    /// rendering "Linked memories: <uuid>" for a memory that no longer exists.
    #[test]
    fn scrubbing_a_deleted_memory_clears_dangling_links() {
        let (store, _temp) = setup_store();

        let doomed = MemoryId(Uuid::new_v4());
        let kept = MemoryId(Uuid::new_v4());

        let linked = Todo::new("test_user".to_string(), "Has both links".to_string());
        store.store_todo(&linked).unwrap();
        store
            .add_related_memory("test_user", &linked.id, doomed.clone())
            .unwrap();
        store
            .add_related_memory("test_user", &linked.id, kept.clone())
            .unwrap();

        let untouched = Todo::new("test_user".to_string(), "No links".to_string());
        store.store_todo(&untouched).unwrap();

        // A different user's todo linked to the same id must not be touched:
        // the scrub is scoped to the user whose memory was deleted.
        let other_user = Todo::new("other_user".to_string(), "Other user".to_string());
        store.store_todo(&other_user).unwrap();
        store
            .add_related_memory("other_user", &other_user.id, doomed.clone())
            .unwrap();

        let scrubbed = store
            .remove_memory_links("test_user", &doomed)
            .expect("remove_memory_links");
        assert_eq!(
            scrubbed, 1,
            "exactly one todo referenced the deleted memory"
        );

        let after = store.get_todo("test_user", &linked.id).unwrap().unwrap();
        assert_eq!(
            after.related_memory_ids,
            vec![kept],
            "the scrub must drop only the deleted memory's link"
        );
        assert!(store
            .get_todo("other_user", &other_user.id)
            .unwrap()
            .unwrap()
            .related_memory_ids
            .contains(&doomed));

        // Idempotent: a second pass finds nothing left to do.
        assert_eq!(store.remove_memory_links("test_user", &doomed).unwrap(), 0);
    }
}
