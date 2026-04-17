//! Session Tracking Module
//!
//! Tracks user sessions with timeline, metrics, and analytics.
//! Each session represents a conversation/work period with the AI.

use chrono::{DateTime, Datelike, Duration, Local, Timelike, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Time of day classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeOfDay {
    EarlyMorning, // 5-8
    Morning,      // 8-12
    Afternoon,    // 12-17
    Evening,      // 17-21
    Night,        // 21-5
}

impl TimeOfDay {
    /// Classify hour into time of day
    pub fn from_hour(hour: u32) -> Self {
        match hour {
            5..=7 => Self::EarlyMorning,
            8..=11 => Self::Morning,
            12..=16 => Self::Afternoon,
            17..=20 => Self::Evening,
            _ => Self::Night,
        }
    }

    /// Human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            Self::EarlyMorning => "Early morning",
            Self::Morning => "Morning",
            Self::Afternoon => "Afternoon",
            Self::Evening => "Evening",
            Self::Night => "Night",
        }
    }

    /// Short label for compact display
    pub fn short_label(&self) -> &'static str {
        match self {
            Self::EarlyMorning => "early AM",
            Self::Morning => "AM",
            Self::Afternoon => "PM",
            Self::Evening => "evening",
            Self::Night => "night",
        }
    }
}

/// Temporal context for a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Time of day when session started
    pub time_of_day: TimeOfDay,
    /// Day of week (Monday=0, Sunday=6)
    pub day_of_week: u32,
    /// Day of week name
    pub day_name: String,
    /// Month name
    pub month_name: String,
    /// Day of month
    pub day: u32,
    /// Year
    pub year: i32,
    /// Human-readable label like "Morning session of Dec 20th"
    pub label: String,
    /// Relative time label like "Today", "Yesterday", "Last week"
    pub relative: String,
}

impl TemporalContext {
    /// Create temporal context from a datetime
    pub fn from_datetime(dt: DateTime<Utc>) -> Self {
        let local = dt.with_timezone(&Local);
        let now = Local::now();

        let time_of_day = TimeOfDay::from_hour(local.hour());
        let day_of_week = local.weekday().num_days_from_monday();

        let day_name = match local.weekday() {
            chrono::Weekday::Mon => "Monday",
            chrono::Weekday::Tue => "Tuesday",
            chrono::Weekday::Wed => "Wednesday",
            chrono::Weekday::Thu => "Thursday",
            chrono::Weekday::Fri => "Friday",
            chrono::Weekday::Sat => "Saturday",
            chrono::Weekday::Sun => "Sunday",
        }
        .to_string();

        let month_name = match local.month() {
            1 => "Jan",
            2 => "Feb",
            3 => "Mar",
            4 => "Apr",
            5 => "May",
            6 => "Jun",
            7 => "Jul",
            8 => "Aug",
            9 => "Sep",
            10 => "Oct",
            11 => "Nov",
            12 => "Dec",
            _ => "???",
        }
        .to_string();

        let day = local.day();
        let year = local.year();

        // Ordinal suffix
        let ordinal = match day {
            1 | 21 | 31 => "st",
            2 | 22 => "nd",
            3 | 23 => "rd",
            _ => "th",
        };

        // Relative time
        let days_ago = (now.date_naive() - local.date_naive()).num_days();
        let relative = match days_ago {
            0 => "Today".to_string(),
            1 => "Yesterday".to_string(),
            2..=6 => format!("This {}", day_name),
            7..=13 => "Last week".to_string(),
            14..=30 => format!("{} weeks ago", days_ago / 7),
            _ => format!("{} {} {}", month_name, day, year),
        };

        // Full label: "Morning session of Dec 20th" or "Today's morning session"
        let label = if days_ago == 0 {
            format!("Today's {} session", time_of_day.label().to_lowercase())
        } else if days_ago == 1 {
            format!("Yesterday's {} session", time_of_day.label().to_lowercase())
        } else {
            format!(
                "{} session of {} {}{}",
                time_of_day.label(),
                month_name,
                day,
                ordinal
            )
        };

        Self {
            time_of_day,
            day_of_week,
            day_name,
            month_name,
            day,
            year,
            label,
            relative,
        }
    }

    /// Short label like "Dec 20 AM"
    pub fn short_label(&self) -> String {
        format!(
            "{} {} {}",
            self.month_name,
            self.day,
            self.time_of_day.short_label()
        )
    }
}

/// Unique session identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn short(&self) -> String {
        self.0.to_string()[..8].to_string()
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Session status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Currently active session
    Active,
    /// Session completed normally
    Completed,
    /// Session timed out or abandoned
    Abandoned,
}

/// Event types that can occur during a session
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    /// Session started
    SessionStart { timestamp: DateTime<Utc> },

    /// Memory was created
    MemoryCreated {
        timestamp: DateTime<Utc>,
        memory_id: String,
        memory_type: String,
        content_preview: String,
        entities: Vec<String>,
    },

    /// Memories were surfaced for a query
    MemoriesSurfaced {
        timestamp: DateTime<Utc>,
        query_preview: String,
        memory_count: usize,
        memory_ids: Vec<String>,
        avg_score: f32,
    },

    /// Memory was used in response (entity flow detected usage)
    MemoryUsed {
        timestamp: DateTime<Utc>,
        memory_id: String,
        derived_ratio: f32,
    },

    /// Todo was created
    TodoCreated {
        timestamp: DateTime<Utc>,
        todo_id: String,
        content: String,
        project: Option<String>,
    },

    /// Todo was completed
    TodoCompleted {
        timestamp: DateTime<Utc>,
        todo_id: String,
    },

    /// Topic changed (detected via low similarity)
    TopicChange {
        timestamp: DateTime<Utc>,
        similarity: f32,
    },

    /// Query was processed
    QueryProcessed {
        timestamp: DateTime<Utc>,
        query_preview: String,
        tokens_estimated: usize,
    },

    /// Context window was compressed (triggered by client)
    ContextCompressed {
        timestamp: DateTime<Utc>,
        tokens_before: usize,
        tokens_after: usize,
    },

    /// Session ended
    SessionEnd {
        timestamp: DateTime<Utc>,
        reason: String,
    },
}

impl SessionEvent {
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::SessionStart { timestamp } => *timestamp,
            Self::MemoryCreated { timestamp, .. } => *timestamp,
            Self::MemoriesSurfaced { timestamp, .. } => *timestamp,
            Self::MemoryUsed { timestamp, .. } => *timestamp,
            Self::TodoCreated { timestamp, .. } => *timestamp,
            Self::TodoCompleted { timestamp, .. } => *timestamp,
            Self::TopicChange { timestamp, .. } => *timestamp,
            Self::QueryProcessed { timestamp, .. } => *timestamp,
            Self::ContextCompressed { timestamp, .. } => *timestamp,
            Self::SessionEnd { timestamp, .. } => *timestamp,
        }
    }

    pub fn event_type(&self) -> &'static str {
        match self {
            Self::SessionStart { .. } => "session_start",
            Self::MemoryCreated { .. } => "memory_created",
            Self::MemoriesSurfaced { .. } => "memories_surfaced",
            Self::MemoryUsed { .. } => "memory_used",
            Self::TodoCreated { .. } => "todo_created",
            Self::TodoCompleted { .. } => "todo_completed",
            Self::TopicChange { .. } => "topic_change",
            Self::QueryProcessed { .. } => "query_processed",
            Self::ContextCompressed { .. } => "context_compressed",
            Self::SessionEnd { .. } => "session_end",
        }
    }
}

/// Session statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStats {
    /// Number of memories created
    pub memories_created: usize,
    /// Number of memories surfaced
    pub memories_surfaced: usize,
    /// Number of memories actually used (derived_ratio > 0)
    pub memories_used: usize,
    /// Memory hit rate (used / surfaced)
    pub memory_hit_rate: f32,
    /// Number of todos created
    pub todos_created: usize,
    /// Number of todos completed
    pub todos_completed: usize,
    /// Todo completion rate
    pub todo_completion_rate: f32,
    /// Number of queries processed
    pub queries_count: usize,
    /// Estimated tokens used
    pub tokens_estimated: usize,
    /// Number of topic changes
    pub topic_changes: usize,
}

impl SessionStats {
    pub fn compute_rates(&mut self) {
        self.memory_hit_rate = if self.memories_surfaced > 0 {
            self.memories_used as f32 / self.memories_surfaced as f32
        } else {
            0.0
        };

        self.todo_completion_rate = if self.todos_created > 0 {
            self.todos_completed as f32 / self.todos_created as f32
        } else {
            0.0
        };
    }
}

/// A user session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session ID
    pub id: SessionId,
    /// User ID
    pub user_id: String,
    /// Session status
    pub status: SessionStatus,
    /// When session started
    pub started_at: DateTime<Utc>,
    /// When session ended (if completed/abandoned)
    pub ended_at: Option<DateTime<Utc>>,
    /// Session duration in seconds
    pub duration_secs: Option<i64>,
    /// Temporal context (time of day, relative date, etc.)
    pub temporal: TemporalContext,
    /// Session statistics
    pub stats: SessionStats,
    /// Timeline of events
    pub timeline: Vec<SessionEvent>,
    /// Optional session name/label (user-provided)
    pub label: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Session {
    pub fn new(user_id: String) -> Self {
        let now = Utc::now();
        Self {
            id: SessionId::new(),
            user_id,
            status: SessionStatus::Active,
            started_at: now,
            ended_at: None,
            duration_secs: None,
            temporal: TemporalContext::from_datetime(now),
            stats: SessionStats::default(),
            timeline: vec![SessionEvent::SessionStart { timestamp: now }],
            label: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_id(user_id: String, session_id: SessionId) -> Self {
        let now = Utc::now();
        Self {
            id: session_id,
            user_id,
            status: SessionStatus::Active,
            started_at: now,
            ended_at: None,
            duration_secs: None,
            temporal: TemporalContext::from_datetime(now),
            stats: SessionStats::default(),
            timeline: vec![SessionEvent::SessionStart { timestamp: now }],
            label: None,
            metadata: HashMap::new(),
        }
    }

    /// Get human-readable temporal label like "Morning session of Dec 20th"
    pub fn temporal_label(&self) -> &str {
        &self.temporal.label
    }

    /// Get short temporal label like "Dec 20 AM"
    pub fn short_temporal_label(&self) -> String {
        self.temporal.short_label()
    }

    /// Add an event to the timeline
    pub fn add_event(&mut self, event: SessionEvent) {
        // Update stats based on event type
        match &event {
            SessionEvent::MemoryCreated { .. } => {
                self.stats.memories_created += 1;
            }
            SessionEvent::MemoriesSurfaced { memory_count, .. } => {
                self.stats.memories_surfaced += memory_count;
            }
            SessionEvent::MemoryUsed { .. } => {
                self.stats.memories_used += 1;
            }
            SessionEvent::TodoCreated { .. } => {
                self.stats.todos_created += 1;
            }
            SessionEvent::TodoCompleted { .. } => {
                self.stats.todos_completed += 1;
            }
            SessionEvent::TopicChange { .. } => {
                self.stats.topic_changes += 1;
            }
            SessionEvent::QueryProcessed {
                tokens_estimated, ..
            } => {
                self.stats.queries_count += 1;
                self.stats.tokens_estimated += tokens_estimated;
            }
            SessionEvent::ContextCompressed { .. } => {
                // No stat update — compressions counted from timeline in digest()
            }
            _ => {}
        }

        self.stats.compute_rates();
        self.timeline.push(event);
    }

    /// End the session
    pub fn end(&mut self, reason: &str) {
        let now = Utc::now();
        self.status = if reason == "timeout" || reason == "abandoned" {
            SessionStatus::Abandoned
        } else {
            SessionStatus::Completed
        };
        self.ended_at = Some(now);
        self.duration_secs = Some((now - self.started_at).num_seconds());
        self.timeline.push(SessionEvent::SessionEnd {
            timestamp: now,
            reason: reason.to_string(),
        });
        self.stats.compute_rates();
    }

    /// Check if session is active
    pub fn is_active(&self) -> bool {
        self.status == SessionStatus::Active
    }

    /// Get session duration
    pub fn duration(&self) -> Duration {
        let end = self.ended_at.unwrap_or_else(Utc::now);
        end - self.started_at
    }

    /// Get summary for display
    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            id: self.id.clone(),
            user_id: self.user_id.clone(),
            status: self.status.clone(),
            started_at: self.started_at,
            ended_at: self.ended_at,
            duration_secs: self.duration().num_seconds(),
            temporal: self.temporal.clone(),
            label: self.label.clone(),
            stats: self.stats.clone(),
        }
    }

    /// Generate a session digest by aggregating timeline events and stats.
    ///
    /// `tools_used` and `consolidation_events` are left at defaults (0/empty) —
    /// the API handler enriches them from audit logs and learning history.
    pub fn digest(&self, token_budget: usize) -> SessionDigest {
        let now = Utc::now();
        let mut entities: Vec<String> = Vec::new();
        let mut compressions: usize = 0;

        for event in &self.timeline {
            match event {
                SessionEvent::MemoryCreated {
                    entities: ents, ..
                } => {
                    entities.extend(ents.iter().cloned());
                }
                SessionEvent::ContextCompressed { .. } => {
                    compressions += 1;
                }
                _ => {}
            }
        }

        entities.sort();
        entities.dedup();
        let entity_count = entities.len();

        let token_percent = if token_budget > 0 {
            self.stats.tokens_estimated as f32 / token_budget as f32
        } else {
            0.0
        };

        SessionDigest {
            session_id: self.id.clone(),
            started_at: self.started_at,
            digest_at: now,
            duration_secs: (now - self.started_at).num_seconds(),
            tokens_estimated: self.stats.tokens_estimated,
            token_budget,
            token_percent,
            memories_created: self.stats.memories_created,
            memories_surfaced: self.stats.memories_surfaced,
            memories_used: self.stats.memories_used,
            memory_hit_rate: self.stats.memory_hit_rate,
            todos_created: self.stats.todos_created,
            todos_completed: self.stats.todos_completed,
            entities_extracted: entities,
            entity_count,
            tools_used: HashMap::new(),
            topic_changes: self.stats.topic_changes,
            compressions,
            consolidation_events: 0,
        }
    }
}

/// Aggregated session digest — computed on the fly from session timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDigest {
    /// Session ID
    pub session_id: SessionId,
    /// When session started
    pub started_at: DateTime<Utc>,
    /// When this digest was generated
    pub digest_at: DateTime<Utc>,
    /// Session duration in seconds
    pub duration_secs: i64,
    /// Estimated tokens used (from proactive_context calls)
    pub tokens_estimated: usize,
    /// Token budget (client-configured)
    pub token_budget: usize,
    /// Token usage as fraction (0.0 - 1.0+)
    pub token_percent: f32,
    /// Memories created this session
    pub memories_created: usize,
    /// Memories surfaced (recalled) this session
    pub memories_surfaced: usize,
    /// Memories actually used in responses
    pub memories_used: usize,
    /// Memory hit rate (used / surfaced)
    pub memory_hit_rate: f32,
    /// Todos created
    pub todos_created: usize,
    /// Todos completed
    pub todos_completed: usize,
    /// Unique entities extracted from created memories
    pub entities_extracted: Vec<String>,
    /// Count of unique entities
    pub entity_count: usize,
    /// Tool usage counts (filled by handler from audit logs)
    pub tools_used: HashMap<String, usize>,
    /// Topic changes detected
    pub topic_changes: usize,
    /// Number of context compressions
    pub compressions: usize,
    /// Consolidation events during this session window
    pub consolidation_events: usize,
}

/// Lightweight session summary (without full timeline)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: SessionId,
    pub user_id: String,
    pub status: SessionStatus,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub duration_secs: i64,
    /// Temporal context with human-readable labels
    pub temporal: TemporalContext,
    /// User-provided label
    pub label: Option<String>,
    pub stats: SessionStats,
}

impl SessionSummary {
    /// Get the display title for this session
    /// Uses user label if set, otherwise temporal label
    pub fn display_title(&self) -> &str {
        self.label.as_deref().unwrap_or(&self.temporal.label)
    }
}

/// Session store - manages all sessions for users
pub struct SessionStore {
    /// Active sessions by session ID
    active: RwLock<HashMap<SessionId, Session>>,
    /// Completed sessions (ring buffer per user, keeps last N)
    completed: RwLock<HashMap<String, Vec<Session>>>,
    /// Maximum completed sessions to keep per user
    max_completed_per_user: usize,
    /// Session timeout in seconds
    timeout_secs: i64,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            active: RwLock::new(HashMap::new()),
            completed: RwLock::new(HashMap::new()),
            max_completed_per_user: 50,
            timeout_secs: 3600, // 1 hour
        }
    }

    pub fn with_config(max_completed_per_user: usize, timeout_secs: i64) -> Self {
        Self {
            active: RwLock::new(HashMap::new()),
            completed: RwLock::new(HashMap::new()),
            max_completed_per_user,
            timeout_secs,
        }
    }

    /// Start a new session for a user
    pub fn start_session(&self, user_id: &str) -> SessionId {
        let session = Session::new(user_id.to_string());
        let id = session.id.clone();
        self.active.write().insert(id.clone(), session);
        id
    }

    /// Start a session with a specific ID (for resumption)
    pub fn start_session_with_id(&self, user_id: &str, session_id: SessionId) -> SessionId {
        let session = Session::with_id(user_id.to_string(), session_id.clone());
        self.active.write().insert(session_id.clone(), session);
        session_id
    }

    /// Get or create active session for user
    pub fn get_or_create_session(&self, user_id: &str) -> SessionId {
        // Check if user has an active session
        {
            let active = self.active.read();
            for (id, session) in active.iter() {
                if session.user_id == user_id && session.is_active() {
                    return id.clone();
                }
            }
        }
        // No active session, create one
        self.start_session(user_id)
    }

    /// Add event to a session
    pub fn add_event(&self, session_id: &SessionId, event: SessionEvent) -> bool {
        let mut active = self.active.write();
        if let Some(session) = active.get_mut(session_id) {
            session.add_event(event);
            true
        } else {
            false
        }
    }

    /// Add event to user's active session
    pub fn add_event_to_user(&self, user_id: &str, event: SessionEvent) -> Option<SessionId> {
        let mut active = self.active.write();
        for (id, session) in active.iter_mut() {
            if session.user_id == user_id && session.is_active() {
                session.add_event(event);
                return Some(id.clone());
            }
        }
        None
    }

    /// End a session
    pub fn end_session(&self, session_id: &SessionId, reason: &str) -> Option<Session> {
        let mut active = self.active.write();
        if let Some(mut session) = active.remove(session_id) {
            session.end(reason);

            // Move to completed
            let mut completed = self.completed.write();
            let user_sessions = completed
                .entry(session.user_id.clone())
                .or_default();
            user_sessions.push(session.clone());

            // Trim to max
            if user_sessions.len() > self.max_completed_per_user {
                let excess = user_sessions.len() - self.max_completed_per_user;
                user_sessions.drain(0..excess);
            }

            Some(session)
        } else {
            None
        }
    }

    /// Get session time range (start, end) for date-based retrieval.
    ///
    /// For active sessions, `end` is the current time.
    /// For completed sessions, `end` is the session's `ended_at` timestamp.
    pub fn get_session_time_range(
        &self,
        session_id: &SessionId,
    ) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        self.get_session(session_id).map(|session| {
            let end = session.ended_at.unwrap_or_else(Utc::now);
            (session.started_at, end)
        })
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: &SessionId) -> Option<Session> {
        // Check active first
        if let Some(session) = self.active.read().get(session_id) {
            return Some(session.clone());
        }
        // Check completed
        let completed = self.completed.read();
        for sessions in completed.values() {
            if let Some(session) = sessions.iter().find(|s| &s.id == session_id) {
                return Some(session.clone());
            }
        }
        None
    }

    /// Get all sessions for a user
    pub fn get_user_sessions(&self, user_id: &str, limit: usize) -> Vec<SessionSummary> {
        let mut result = Vec::new();

        // Add active sessions
        {
            let active = self.active.read();
            for session in active.values() {
                if session.user_id == user_id {
                    result.push(session.summary());
                }
            }
        }

        // Add completed sessions
        {
            let completed = self.completed.read();
            if let Some(sessions) = completed.get(user_id) {
                for session in sessions
                    .iter()
                    .rev()
                    .take(limit.saturating_sub(result.len()))
                {
                    result.push(session.summary());
                }
            }
        }

        // Sort by start time (newest first)
        result.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        result.truncate(limit);
        result
    }

    /// Get active session for user
    pub fn get_active_session(&self, user_id: &str) -> Option<Session> {
        let active = self.active.read();
        for session in active.values() {
            if session.user_id == user_id && session.is_active() {
                return Some(session.clone());
            }
        }
        None
    }

    /// Cleanup stale sessions
    pub fn cleanup_stale_sessions(&self) -> usize {
        let now = Utc::now();
        let timeout = Duration::seconds(self.timeout_secs);

        let stale_ids: Vec<SessionId> = {
            let active = self.active.read();
            active
                .iter()
                .filter(|(_, s)| now - s.started_at > timeout)
                .map(|(id, _)| id.clone())
                .collect()
        };

        let count = stale_ids.len();
        for id in stale_ids {
            self.end_session(&id, "timeout");
        }
        count
    }

    /// Get store statistics
    pub fn stats(&self) -> SessionStoreStats {
        let active = self.active.read();
        let completed = self.completed.read();

        let total_completed: usize = completed.values().map(|v| v.len()).sum();

        SessionStoreStats {
            active_sessions: active.len(),
            completed_sessions: total_completed,
            users_with_sessions: completed.len(),
        }
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Session store statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStoreStats {
    pub active_sessions: usize,
    pub completed_sessions: usize,
    pub users_with_sessions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_lifecycle() {
        let store = SessionStore::new();

        // Start session
        let session_id = store.start_session("test-user");
        assert!(store.get_session(&session_id).is_some());

        // Add events
        store.add_event(
            &session_id,
            SessionEvent::MemoryCreated {
                timestamp: Utc::now(),
                memory_id: "mem-1".to_string(),
                memory_type: "Learning".to_string(),
                content_preview: "Test memory".to_string(),
                entities: vec!["rust".to_string()],
            },
        );

        store.add_event(
            &session_id,
            SessionEvent::TodoCreated {
                timestamp: Utc::now(),
                todo_id: "todo-1".to_string(),
                content: "Test todo".to_string(),
                project: None,
            },
        );

        // Check stats
        let session = store.get_session(&session_id).unwrap();
        assert_eq!(session.stats.memories_created, 1);
        assert_eq!(session.stats.todos_created, 1);
        assert_eq!(session.timeline.len(), 3); // start + 2 events

        // End session
        let ended = store.end_session(&session_id, "completed").unwrap();
        assert_eq!(ended.status, SessionStatus::Completed);
        assert!(ended.ended_at.is_some());

        // Should be in completed now
        let sessions = store.get_user_sessions("test-user", 10);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].status, SessionStatus::Completed);
    }

    #[test]
    fn test_memory_hit_rate() {
        let store = SessionStore::new();
        let session_id = store.start_session("test-user");

        // Surface 10 memories
        store.add_event(
            &session_id,
            SessionEvent::MemoriesSurfaced {
                timestamp: Utc::now(),
                query_preview: "test query".to_string(),
                memory_count: 10,
                memory_ids: (0..10).map(|i| format!("mem-{}", i)).collect(),
                avg_score: 0.8,
            },
        );

        // Use 3 of them
        for i in 0..3 {
            store.add_event(
                &session_id,
                SessionEvent::MemoryUsed {
                    timestamp: Utc::now(),
                    memory_id: format!("mem-{}", i),
                    derived_ratio: 0.5,
                },
            );
        }

        let session = store.get_session(&session_id).unwrap();
        assert_eq!(session.stats.memories_surfaced, 10);
        assert_eq!(session.stats.memories_used, 3);
        assert!((session.stats.memory_hit_rate - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_get_or_create() {
        let store = SessionStore::new();

        // First call creates session
        let id1 = store.get_or_create_session("user-1");

        // Second call returns same session
        let id2 = store.get_or_create_session("user-1");
        assert_eq!(id1, id2);

        // Different user gets different session
        let id3 = store.get_or_create_session("user-2");
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_temporal_context() {
        // Test time of day classification
        assert_eq!(TimeOfDay::from_hour(6), TimeOfDay::EarlyMorning);
        assert_eq!(TimeOfDay::from_hour(10), TimeOfDay::Morning);
        assert_eq!(TimeOfDay::from_hour(14), TimeOfDay::Afternoon);
        assert_eq!(TimeOfDay::from_hour(19), TimeOfDay::Evening);
        assert_eq!(TimeOfDay::from_hour(23), TimeOfDay::Night);
        assert_eq!(TimeOfDay::from_hour(3), TimeOfDay::Night);

        // Test labels
        assert_eq!(TimeOfDay::Morning.label(), "Morning");
        assert_eq!(TimeOfDay::Afternoon.short_label(), "PM");

        // Test temporal context creation
        let now = Utc::now();
        let ctx = TemporalContext::from_datetime(now);

        // Should have valid data
        assert!(!ctx.day_name.is_empty());
        assert!(!ctx.month_name.is_empty());
        assert!(ctx.day >= 1 && ctx.day <= 31);
        assert!(!ctx.label.is_empty());
        assert!(!ctx.relative.is_empty());

        // Today's session should say "Today's"
        assert!(ctx.label.contains("Today's") || ctx.relative == "Today");
    }

    #[test]
    fn test_session_temporal_label() {
        let store = SessionStore::new();
        let session_id = store.start_session("test-user");

        let session = store.get_session(&session_id).unwrap();

        // Should have temporal context
        assert!(!session.temporal.label.is_empty());
        assert!(session.temporal_label().contains("Today's"));

        // Summary should include temporal
        let summary = session.summary();
        assert!(!summary.temporal.label.is_empty());
        assert_eq!(summary.display_title(), session.temporal_label());
    }
}
