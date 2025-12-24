//! Prospective Memory - Future intentions and reminders (SHO-116)
//!
//! Implements the "remembering to remember" capability:
//! - Time-based triggers (at specific time, after duration)
//! - Context-based triggers (keyword match, semantic similarity)
//!
//! Architecture:
//! - ProspectiveTask stored in dedicated RocksDB for operational tracking
//! - Memory with ExperienceType::Intention created for semantic integration
//! - Uses Hebbian learning for decay (same as regular memories)

use anyhow::{Context, Result};
use chrono::Utc;
use rocksdb::{Options, WriteBatch, DB};
use std::path::Path;
use std::sync::Arc;

use super::types::{ProspectiveTask, ProspectiveTaskId, ProspectiveTaskStatus, ProspectiveTrigger};

/// Compute cosine similarity between two embedding vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Storage and query engine for prospective memory (reminders)
pub struct ProspectiveStore {
    /// Main task storage: key = {user_id}:{task_id}
    db: Arc<DB>,
    /// Index database for efficient queries
    index_db: Arc<DB>,
}

impl ProspectiveStore {
    /// Create a new prospective store at the given path
    pub fn new(storage_path: &Path) -> Result<Self> {
        let prospective_path = storage_path.join("prospective");
        std::fs::create_dir_all(&prospective_path)?;

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Smaller buffers - prospective tasks are low volume
        opts.set_max_write_buffer_number(2);
        opts.set_write_buffer_size(16 * 1024 * 1024); // 16MB

        let db = Arc::new(
            DB::open(&opts, prospective_path.join("tasks"))
                .context("Failed to open prospective tasks DB")?,
        );

        let index_db = Arc::new(
            DB::open(&opts, prospective_path.join("index"))
                .context("Failed to open prospective index DB")?,
        );

        tracing::info!("Prospective memory store initialized");

        Ok(Self { db, index_db })
    }

    /// Store a new prospective task
    pub fn store(&self, task: &ProspectiveTask) -> Result<()> {
        let key = format!("{}:{}", task.user_id, task.id);
        // Use JSON instead of bincode - handles tagged enums properly and is human-readable
        let value =
            serde_json::to_vec(task).context("Failed to serialize prospective task to JSON")?;

        self.db
            .put(key.as_bytes(), &value)
            .context("Failed to store prospective task")?;

        // Update indices
        self.update_indices(task)?;

        tracing::debug!(
            task_id = %task.id,
            user_id = %task.user_id,
            trigger = ?task.trigger,
            "Stored prospective task"
        );

        Ok(())
    }

    /// Update secondary indices for efficient queries
    fn update_indices(&self, task: &ProspectiveTask) -> Result<()> {
        let mut batch = WriteBatch::default();

        // Index by user (for listing user's reminders)
        let user_key = format!("user:{}:{}", task.user_id, task.id);
        batch.put(user_key.as_bytes(), b"1");

        // Index by status
        let status_key = format!("status:{:?}:{}:{}", task.status, task.user_id, task.id);
        batch.put(status_key.as_bytes(), b"1");

        // Index by due time (for time-based trigger queries)
        if let Some(due_at) = task.trigger.due_at() {
            let due_key = format!("due:{}:{}", due_at.timestamp(), task.id);
            batch.put(due_key.as_bytes(), task.user_id.as_bytes());
        }

        // Index context triggers by keywords
        if let ProspectiveTrigger::OnContext { ref keywords, .. } = task.trigger {
            for keyword in keywords {
                let kw_key = format!(
                    "context:{}:{}:{}",
                    keyword.to_lowercase(),
                    task.user_id,
                    task.id
                );
                batch.put(kw_key.as_bytes(), b"1");
            }
        }

        self.index_db
            .write(batch)
            .context("Failed to update prospective indices")?;

        Ok(())
    }

    /// Get a task by ID
    pub fn get(
        &self,
        user_id: &str,
        task_id: &ProspectiveTaskId,
    ) -> Result<Option<ProspectiveTask>> {
        let key = format!("{}:{}", user_id, task_id);

        match self.db.get(key.as_bytes())? {
            Some(value) => {
                let task: ProspectiveTask = serde_json::from_slice(&value)
                    .context("Failed to deserialize prospective task from JSON")?;
                Ok(Some(task))
            }
            None => Ok(None),
        }
    }

    /// Update a task (e.g., mark as triggered/dismissed)
    pub fn update(&self, task: &ProspectiveTask) -> Result<()> {
        // Remove old indices first
        self.remove_indices(&task.user_id, &task.id)?;

        // Store updated task
        self.store(task)
    }

    /// Remove indices for a task (used during update/delete)
    fn remove_indices(&self, user_id: &str, task_id: &ProspectiveTaskId) -> Result<()> {
        // Get the task to know what indices to remove
        if let Some(task) = self.get(user_id, task_id)? {
            let mut batch = WriteBatch::default();

            let user_key = format!("user:{}:{}", user_id, task_id);
            batch.delete(user_key.as_bytes());

            let status_key = format!("status:{:?}:{}:{}", task.status, user_id, task_id);
            batch.delete(status_key.as_bytes());

            if let Some(due_at) = task.trigger.due_at() {
                let due_key = format!("due:{}:{}", due_at.timestamp(), task_id);
                batch.delete(due_key.as_bytes());
            }

            if let ProspectiveTrigger::OnContext { ref keywords, .. } = task.trigger {
                for keyword in keywords {
                    let kw_key =
                        format!("context:{}:{}:{}", keyword.to_lowercase(), user_id, task_id);
                    batch.delete(kw_key.as_bytes());
                }
            }

            self.index_db.write(batch)?;
        }

        Ok(())
    }

    /// Delete a task
    pub fn delete(&self, user_id: &str, task_id: &ProspectiveTaskId) -> Result<bool> {
        let key = format!("{}:{}", user_id, task_id);

        // Check if exists
        if self.db.get(key.as_bytes())?.is_none() {
            return Ok(false);
        }

        // Remove indices
        self.remove_indices(user_id, task_id)?;

        // Delete task
        self.db.delete(key.as_bytes())?;

        tracing::debug!(task_id = %task_id, user_id = %user_id, "Deleted prospective task");

        Ok(true)
    }

    /// List all tasks for a user, optionally filtered by status
    pub fn list_for_user(
        &self,
        user_id: &str,
        status_filter: Option<ProspectiveTaskStatus>,
    ) -> Result<Vec<ProspectiveTask>> {
        let prefix = format!("user:{}:", user_id);
        let mut tasks = Vec::new();

        for item in self.index_db.prefix_iterator(prefix.as_bytes()) {
            let (key, _) = item.context("Failed to read index entry")?;
            let key_str = String::from_utf8_lossy(&key);

            // Extract task_id from key: user:{user_id}:{task_id}
            if let Some(task_id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(task_id_str) {
                    let task_id = ProspectiveTaskId(uuid);
                    if let Some(task) = self.get(user_id, &task_id)? {
                        // Apply status filter if specified
                        if let Some(filter) = status_filter {
                            if task.status == filter {
                                tasks.push(task);
                            }
                        } else {
                            tasks.push(task);
                        }
                    }
                }
            }
        }

        // Sort by created_at descending (newest first)
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(tasks)
    }

    /// Get all due time-based tasks for a user
    ///
    /// Returns tasks where:
    /// - Trigger is time-based (AtTime or AfterDuration)
    /// - Trigger time <= now
    /// - Status is Pending
    pub fn get_due_tasks(&self, user_id: &str) -> Result<Vec<ProspectiveTask>> {
        let now = Utc::now();
        let now_ts = now.timestamp();

        let mut due_tasks = Vec::new();

        // Scan due index for tasks with due_time <= now
        // Key format: due:{timestamp}:{task_id}
        for item in self.index_db.prefix_iterator(b"due:") {
            let (key, value) = item.context("Failed to read due index")?;
            let key_str = String::from_utf8_lossy(&key);

            // Parse key: due:{timestamp}:{task_id}
            let parts: Vec<&str> = key_str.splitn(3, ':').collect();
            if parts.len() != 3 {
                continue;
            }

            let task_ts: i64 = match parts[1].parse() {
                Ok(ts) => ts,
                Err(_) => continue,
            };

            // Skip if not yet due
            if task_ts > now_ts {
                continue;
            }

            // Check user matches
            let stored_user_id = String::from_utf8_lossy(&value);
            if stored_user_id != user_id {
                continue;
            }

            // Get task and check status
            if let Ok(uuid) = uuid::Uuid::parse_str(parts[2]) {
                let task_id = ProspectiveTaskId(uuid);
                if let Some(task) = self.get(user_id, &task_id)? {
                    if task.status == ProspectiveTaskStatus::Pending {
                        due_tasks.push(task);
                    }
                }
            }
        }

        // Sort by priority (higher first) then by due time (earliest first)
        due_tasks.sort_by(|a, b| {
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }
            let a_due = a.trigger.due_at().unwrap_or(a.created_at);
            let b_due = b.trigger.due_at().unwrap_or(b.created_at);
            a_due.cmp(&b_due)
        });

        Ok(due_tasks)
    }

    /// Check for context-triggered reminders based on text content (keyword match only)
    ///
    /// Returns tasks where:
    /// - Trigger is OnContext
    /// - Any keyword matches the context text
    /// - Status is Pending
    ///
    /// For semantic matching, use `check_context_triggers_semantic` instead.
    pub fn check_context_triggers(
        &self,
        user_id: &str,
        context: &str,
    ) -> Result<Vec<ProspectiveTask>> {
        let context_lower = context.to_lowercase();
        let mut matches = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        // Get all pending context-based tasks for user
        let pending_tasks = self.list_for_user(user_id, Some(ProspectiveTaskStatus::Pending))?;

        for task in pending_tasks {
            if seen_ids.contains(&task.id.0) {
                continue;
            }

            if let ProspectiveTrigger::OnContext { ref keywords, .. } = task.trigger {
                // Check if any keyword matches
                let matched = keywords
                    .iter()
                    .any(|kw| context_lower.contains(&kw.to_lowercase()));
                if matched {
                    seen_ids.insert(task.id.0);
                    matches.push(task);
                }
            }
        }

        // Sort by priority
        matches.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(matches)
    }

    /// Check for context-triggered reminders using both keyword AND semantic matching
    ///
    /// Returns tasks that either:
    /// - Have keyword matches in the context (score = 1.0), OR
    /// - Have semantic similarity above their threshold
    ///
    /// # Arguments
    /// * `user_id` - User to check reminders for
    /// * `context` - Current context text (for keyword matching)
    /// * `context_embedding` - Precomputed embedding of the context
    /// * `embed_fn` - Closure to compute embedding for task content
    ///
    /// # Returns
    /// Vector of (task, score) tuples sorted by score (highest first)
    pub fn check_context_triggers_semantic<F>(
        &self,
        user_id: &str,
        context: &str,
        context_embedding: &[f32],
        embed_fn: F,
    ) -> Result<Vec<(ProspectiveTask, f32)>>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let context_lower = context.to_lowercase();
        let mut matches: Vec<(ProspectiveTask, f32)> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        // Get all pending context-based tasks for user
        let pending_tasks = self.list_for_user(user_id, Some(ProspectiveTaskStatus::Pending))?;

        for task in pending_tasks {
            if seen_ids.contains(&task.id.0) {
                continue;
            }

            if let ProspectiveTrigger::OnContext {
                ref keywords,
                threshold,
            } = task.trigger
            {
                // 1. Check keyword matches first (fast path)
                let keyword_match = keywords
                    .iter()
                    .any(|kw| context_lower.contains(&kw.to_lowercase()));

                if keyword_match {
                    seen_ids.insert(task.id.0);
                    matches.push((task, 1.0)); // Perfect score for keyword match
                    continue;
                }

                // 2. Try semantic matching
                if let Some(task_embedding) = embed_fn(&task.content) {
                    let similarity = cosine_similarity(context_embedding, &task_embedding);
                    if similarity >= threshold {
                        seen_ids.insert(task.id.0);
                        matches.push((task, similarity));
                    }
                }
            }
        }

        // Sort by score (highest first), then by priority
        matches.sort_by(|a, b| {
            let score_cmp = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }
            b.0.priority.cmp(&a.0.priority)
        });

        Ok(matches)
    }

    /// Mark a task as triggered
    pub fn mark_triggered(&self, user_id: &str, task_id: &ProspectiveTaskId) -> Result<bool> {
        if let Some(mut task) = self.get(user_id, task_id)? {
            if task.status == ProspectiveTaskStatus::Pending {
                task.mark_triggered();
                self.update(&task)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Mark a task as dismissed
    pub fn mark_dismissed(&self, user_id: &str, task_id: &ProspectiveTaskId) -> Result<bool> {
        if let Some(mut task) = self.get(user_id, task_id)? {
            task.mark_dismissed();
            self.update(&task)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Get count of pending tasks for a user
    pub fn pending_count(&self, user_id: &str) -> Result<usize> {
        let tasks = self.list_for_user(user_id, Some(ProspectiveTaskStatus::Pending))?;
        Ok(tasks.len())
    }

    /// Find a task by ID prefix (for short ID lookups)
    ///
    /// Allows users to dismiss reminders using short IDs like "d8cdc580"
    /// instead of full UUIDs like "d8cdc580-bf96-403a-85c5-57098c7b1786"
    pub fn find_by_prefix(
        &self,
        user_id: &str,
        id_prefix: &str,
    ) -> Result<Option<ProspectiveTask>> {
        let prefix_lower = id_prefix.to_lowercase();
        let tasks = self.list_for_user(user_id, None)?;

        let matches: Vec<_> = tasks
            .into_iter()
            .filter(|t| t.id.0.to_string().to_lowercase().starts_with(&prefix_lower))
            .collect();

        match matches.len() {
            0 => Ok(None),
            1 => Ok(Some(matches.into_iter().next().unwrap())),
            _ => {
                // Multiple matches - return the first one but log warning
                tracing::warn!(
                    user_id = %user_id,
                    prefix = %id_prefix,
                    matches = matches.len(),
                    "Multiple reminders match prefix, using first"
                );
                Ok(Some(matches.into_iter().next().unwrap()))
            }
        }
    }

    /// Cleanup expired/dismissed tasks older than given days
    pub fn cleanup_old_tasks(&self, user_id: &str, older_than_days: i64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(older_than_days);
        let mut deleted = 0;

        let tasks = self.list_for_user(user_id, None)?;
        for task in tasks {
            // Only cleanup dismissed or expired tasks
            if matches!(
                task.status,
                ProspectiveTaskStatus::Dismissed | ProspectiveTaskStatus::Expired
            ) {
                let check_time = task.dismissed_at.unwrap_or(task.created_at);
                if check_time < cutoff {
                    self.delete(user_id, &task.id)?;
                    deleted += 1;
                }
            }
        }

        if deleted > 0 {
            tracing::info!(user_id = %user_id, deleted = deleted, "Cleaned up old prospective tasks");
        }

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_store() -> (TempDir, ProspectiveStore) {
        let temp_dir = TempDir::new().unwrap();
        let store = ProspectiveStore::new(temp_dir.path()).unwrap();
        (temp_dir, store)
    }

    #[test]
    fn test_store_and_get() {
        let (_temp, store) = setup_store();

        let task = ProspectiveTask::new(
            "test-user".to_string(),
            "Remember to push code".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 3600,
                from: Utc::now(),
            },
        );

        store.store(&task).unwrap();

        let retrieved = store.get("test-user", &task.id).unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.content, "Remember to push code");
        assert_eq!(retrieved.status, ProspectiveTaskStatus::Pending);
    }

    #[test]
    fn test_list_for_user() {
        let (_temp, store) = setup_store();

        // Create tasks for two users
        let task1 = ProspectiveTask::new(
            "user-a".to_string(),
            "Task 1".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 3600,
                from: Utc::now(),
            },
        );

        let task2 = ProspectiveTask::new(
            "user-a".to_string(),
            "Task 2".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 7200,
                from: Utc::now(),
            },
        );

        let task3 = ProspectiveTask::new(
            "user-b".to_string(),
            "Task 3".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 3600,
                from: Utc::now(),
            },
        );

        store.store(&task1).unwrap();
        store.store(&task2).unwrap();
        store.store(&task3).unwrap();

        let user_a_tasks = store.list_for_user("user-a", None).unwrap();
        assert_eq!(user_a_tasks.len(), 2);

        let user_b_tasks = store.list_for_user("user-b", None).unwrap();
        assert_eq!(user_b_tasks.len(), 1);
    }

    #[test]
    fn test_due_tasks() {
        let (_temp, store) = setup_store();

        // Task that's already due (0 seconds from past)
        let past = Utc::now() - chrono::Duration::seconds(100);
        let due_task = ProspectiveTask::new(
            "test-user".to_string(),
            "Due task".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 0,
                from: past,
            },
        );

        // Task that's not due yet
        let future_task = ProspectiveTask::new(
            "test-user".to_string(),
            "Future task".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 999999,
                from: Utc::now(),
            },
        );

        store.store(&due_task).unwrap();
        store.store(&future_task).unwrap();

        let due = store.get_due_tasks("test-user").unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].content, "Due task");
    }

    #[test]
    fn test_context_trigger() {
        let (_temp, store) = setup_store();

        let task = ProspectiveTask::new(
            "test-user".to_string(),
            "Check auth token".to_string(),
            ProspectiveTrigger::OnContext {
                keywords: vec![
                    "authentication".to_string(),
                    "token".to_string(),
                    "jwt".to_string(),
                ],
                threshold: 0.7,
            },
        );

        store.store(&task).unwrap();

        // Should match on keyword
        let matches = store
            .check_context_triggers("test-user", "I need to fix the JWT token expiry")
            .unwrap();
        assert_eq!(matches.len(), 1);

        // Should not match
        let no_matches = store
            .check_context_triggers("test-user", "Let's update the database schema")
            .unwrap();
        assert_eq!(no_matches.len(), 0);
    }

    #[test]
    fn test_mark_triggered_and_dismissed() {
        let (_temp, store) = setup_store();

        let task = ProspectiveTask::new(
            "test-user".to_string(),
            "Test task".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 0,
                from: Utc::now(),
            },
        );

        store.store(&task).unwrap();

        // Mark as triggered
        store.mark_triggered("test-user", &task.id).unwrap();
        let updated = store.get("test-user", &task.id).unwrap().unwrap();
        assert_eq!(updated.status, ProspectiveTaskStatus::Triggered);
        assert!(updated.triggered_at.is_some());

        // Mark as dismissed
        store.mark_dismissed("test-user", &task.id).unwrap();
        let dismissed = store.get("test-user", &task.id).unwrap().unwrap();
        assert_eq!(dismissed.status, ProspectiveTaskStatus::Dismissed);
        assert!(dismissed.dismissed_at.is_some());
    }

    #[test]
    fn test_delete() {
        let (_temp, store) = setup_store();

        let task = ProspectiveTask::new(
            "test-user".to_string(),
            "To delete".to_string(),
            ProspectiveTrigger::AfterDuration {
                seconds: 3600,
                from: Utc::now(),
            },
        );

        store.store(&task).unwrap();
        assert!(store.get("test-user", &task.id).unwrap().is_some());

        let deleted = store.delete("test-user", &task.id).unwrap();
        assert!(deleted);
        assert!(store.get("test-user", &task.id).unwrap().is_none());
    }
}
