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

use anyhow::{Context, Result};
use chrono::Utc;
use rocksdb::{Options, WriteBatch, DB};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use super::types::{Project, ProjectId, ProjectStatus, Todo, TodoId, TodoStatus};

/// Storage and query engine for todos and projects
pub struct TodoStore {
    /// Main todo storage: key = {user_id}:{todo_id}
    todo_db: Arc<DB>,
    /// Project storage: key = {user_id}:{project_id}
    project_db: Arc<DB>,
    /// Index database for efficient queries
    index_db: Arc<DB>,
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

        tracing::info!("Todo store initialized");

        Ok(Self {
            todo_db,
            project_db,
            index_db,
        })
    }

    // =========================================================================
    // TODO CRUD OPERATIONS
    // =========================================================================

    /// Store a new todo
    pub fn store_todo(&self, todo: &Todo) -> Result<()> {
        let key = format!("{}:{}", todo.user_id, todo.id.0);
        let value = serde_json::to_vec(todo).context("Failed to serialize todo")?;

        self.todo_db
            .put(key.as_bytes(), &value)
            .context("Failed to store todo")?;

        self.update_todo_indices(todo)?;

        tracing::debug!(
            todo_id = %todo.id,
            user_id = %todo.user_id,
            status = ?todo.status,
            "Stored todo"
        );

        Ok(())
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

        // Index by due date
        if let Some(ref due) = todo.due_date {
            let due_key = format!("due:{}:{}:{}", due.timestamp(), todo.user_id, id_str);
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
            let due_key = format!("due:{}:{}:{}", due.timestamp(), todo.user_id, id_str);
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
                let todo: Todo =
                    serde_json::from_slice(&value).context("Failed to deserialize todo")?;
                Ok(Some(todo))
            }
            None => Ok(None),
        }
    }

    /// Find todo by short ID prefix
    pub fn find_todo_by_prefix(&self, user_id: &str, prefix: &str) -> Result<Option<Todo>> {
        // Remove "SHO-" prefix if present
        let clean_prefix = prefix
            .strip_prefix("SHO-")
            .or_else(|| prefix.strip_prefix("sho-"))
            .unwrap_or(prefix)
            .to_lowercase();

        let todos = self.list_todos_for_user(user_id, None)?;

        let matches: Vec<_> = todos
            .into_iter()
            .filter(|t| t.id.0.to_string().to_lowercase().starts_with(&clean_prefix))
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

    /// Update a todo
    pub fn update_todo(&self, todo: &Todo) -> Result<()> {
        // Get old todo to remove old indices
        if let Some(old_todo) = self.get_todo(&todo.user_id, &todo.id)? {
            self.remove_todo_indices(&old_todo)?;
        }

        self.store_todo(todo)
    }

    /// Delete a todo
    pub fn delete_todo(&self, user_id: &str, todo_id: &TodoId) -> Result<bool> {
        let key = format!("{}:{}", user_id, todo_id.0);

        if let Some(todo) = self.get_todo(user_id, todo_id)? {
            self.remove_todo_indices(&todo)?;
            self.todo_db.delete(key.as_bytes())?;
            tracing::debug!(todo_id = %todo_id, "Deleted todo");
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
            self.store_todo(&todo)?;

            // Create next recurrence if applicable
            let next_todo = if let Some(next) = todo.create_next_recurrence() {
                self.store_todo(&next)?;
                Some(next)
            } else {
                None
            };

            Ok(Some((todo, next_todo)))
        } else {
            Ok(None)
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

        tracing::debug!(project_id = %project.id.0, name = %project.name, "Stored project");

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

            if let Some(new_description) = description {
                project.description = new_description;
                changed = true;
            }

            if let Some(new_status) = status {
                if new_status != project.status {
                    project.status = new_status.clone();
                    changed = true;

                    // Set completed_at when archiving or completing
                    if new_status == ProjectStatus::Completed || new_status == ProjectStatus::Archived
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

            // Delete project
            let key = format!("{}:{}", user_id, project_id.0);
            self.project_db.delete(key.as_bytes())?;

            // Delete indices
            let user_key = format!("user:{}:{}", user_id, project_id.0);
            self.index_db.delete(user_key.as_bytes())?;

            let name_key = format!("project_name:{}:{}", project.name.to_lowercase(), user_id);
            self.index_db.delete(name_key.as_bytes())?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    // =========================================================================
    // STATS
    // =========================================================================

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
        let short_id = todo.id.short();
        store.store_todo(&todo).unwrap();

        // Find by full SHO-xxxx
        let found = store.find_todo_by_prefix("test_user", &short_id).unwrap();
        assert!(found.is_some());

        // Find by just the hex part
        let hex_part = short_id.strip_prefix("SHO-").unwrap();
        let found2 = store.find_todo_by_prefix("test_user", hex_part).unwrap();
        assert!(found2.is_some());
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
