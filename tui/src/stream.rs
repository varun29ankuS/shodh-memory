use crate::types::{
    AppState, GraphEdge, GraphNode, MemoryEvent, TodoStats, TuiFileMemory, TuiPriority,
    TuiProject, TuiTodo, TuiTodoComment, TuiTodoCommentType, TuiTodoStatus,
};
use chrono::Utc;
use futures_util::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

#[derive(Debug, Deserialize)]
struct MemoryStats {
    total_memories: usize,
    working_memory_count: usize,
    session_memory_count: usize,
    long_term_memory_count: usize,
    #[serde(default)]
    vector_index_count: usize,
    #[serde(default)]
    total_retrievals: usize,
}

#[derive(Debug, Deserialize)]
struct ListMemoryItem {
    id: String,
    content: String,
    memory_type: String,
    #[serde(default)]
    tags: Vec<String>,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
struct ListResponse {
    memories: Vec<ListMemoryItem>,
}

#[derive(Debug, Deserialize)]
struct GraphStatsResponse {
    #[serde(default)]
    entity_count: usize,
    #[serde(default)]
    relationship_count: usize,
}

#[derive(Debug, Deserialize)]
struct UniverseStar {
    id: String,
    name: String,
    #[serde(default)]
    entity_type: String,
    #[serde(default)]
    mention_count: usize,
    #[serde(default)]
    position: UniversePosition,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct UniversePosition {
    #[serde(default)]
    x: f32,
    #[serde(default)]
    y: f32,
    #[serde(default)]
    z: f32,
}

#[derive(Debug, Deserialize)]
struct UniverseConnection {
    from_id: String,
    to_id: String,
    #[serde(default)]
    strength: f32,
}

#[derive(Debug, Deserialize)]
struct MemoryUniverse {
    #[serde(default)]
    stars: Vec<UniverseStar>,
    #[serde(default)]
    connections: Vec<UniverseConnection>,
}

// Todo API response types
#[derive(Debug, Deserialize)]
struct TodoListResponse {
    #[serde(default)]
    todos: Vec<TodoApiItem>,
    #[serde(default)]
    projects: Vec<ProjectApiItem>,
}

#[derive(Debug, Deserialize, Clone)]
struct CommentApiItem {
    id: String,
    author: String,
    content: String,
    #[serde(default)]
    comment_type: String,
    created_at: String,
}

#[derive(Debug, Deserialize, Clone)]
struct TodoApiItem {
    id: String,
    content: String,
    status: String,
    priority: String,
    project_id: Option<String>,
    #[serde(default)]
    contexts: Vec<String>,
    due_date: Option<String>,
    blocked_on: Option<String>,
    created_at: String,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    seq_num: u32,
    #[serde(default)]
    project_prefix: Option<String>,
    #[serde(default)]
    comments: Vec<CommentApiItem>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct ProjectApiItem {
    id: String,
    name: String,
    description: Option<String>,
    status: String,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    prefix: Option<String>,
    #[serde(default)]
    codebase_file_count: usize,
}

#[derive(Debug, Deserialize)]
struct TodoStatsResponse {
    stats: TodoStatsApi,
}

#[derive(Debug, Deserialize)]
struct TodoStatsApi {
    #[serde(default)]
    total: u32,
    #[serde(default)]
    backlog: u32,
    #[serde(default)]
    todo: u32,
    #[serde(default)]
    in_progress: u32,
    #[serde(default)]
    blocked: u32,
    #[serde(default)]
    done: u32,
    #[serde(default)]
    overdue: u32,
}

/// Context session from Claude Code (via status line)
#[derive(Debug, Deserialize, Default)]
struct ContextSessionApi {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    tokens_used: u64,
    #[serde(default)]
    tokens_budget: u64,
    #[serde(default)]
    percent_used: u8,
    #[serde(default)]
    current_task: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    updated_at: Option<String>,
}

pub struct MemoryStream {
    url: String,
    base_url: String,
    api_key: String,
    user_id: String,
    state: Arc<Mutex<AppState>>,
    client: Client,
}

impl MemoryStream {
    pub fn new(base_url: &str, api_key: &str, user_id: &str, state: Arc<Mutex<AppState>>) -> Self {
        let base = base_url.trim_end_matches('/').to_string();
        Self {
            url: format!("{}/api/events?user_id={}", base, user_id),
            base_url: base,
            api_key: api_key.to_string(),
            user_id: user_id.to_string(),
            state,
            client: Client::new(),
        }
    }

    pub async fn run(&self) {
        self.fetch_initial_data().await;

        // SSE connection loop - todo updates are handled via TODO_* events
        loop {
            match self.connect().await {
                Ok(()) => {}
                Err(e) => {
                    let mut state = self.state.lock().await;
                    state.set_error(format!("Connection error: {}", e));
                }
            }
            {
                let mut state = self.state.lock().await;
                state.connected = false;
            }
            sleep(Duration::from_secs(3)).await;
        }
    }

    /// Static poll function for background task
    async fn poll_todos(
        client: &Client,
        base_url: &str,
        api_key: &str,
        user_id: &str,
    ) -> Result<(Vec<TuiTodo>, Vec<TuiProject>, TodoStats), Box<dyn std::error::Error + Send + Sync>>
    {
        let resp = client
            .post(format!("{}/api/todos/list", base_url))
            .header("X-API-Key", api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "user_id": user_id,
                "include_completed": true
            }))
            .send()
            .await?;

        let body = resp.text().await?;
        let todos_resp: TodoListResponse = serde_json::from_str(&body)?;

        let projects_clone = todos_resp.projects.clone();
        let todos: Vec<TuiTodo> = todos_resp
            .todos
            .into_iter()
            .map(|t| {
                let status = match t.status.as_str() {
                    "backlog" => TuiTodoStatus::Backlog,
                    "todo" => TuiTodoStatus::Todo,
                    "in_progress" => TuiTodoStatus::InProgress,
                    "blocked" => TuiTodoStatus::Blocked,
                    "done" => TuiTodoStatus::Done,
                    "cancelled" => TuiTodoStatus::Cancelled,
                    _ => TuiTodoStatus::Todo,
                };
                let priority = match t.priority.as_str() {
                    "urgent" => TuiPriority::Urgent,
                    "high" => TuiPriority::High,
                    "medium" => TuiPriority::Medium,
                    "low" => TuiPriority::Low,
                    _ => TuiPriority::Medium,
                };
                let project_name = t.project_id.as_ref().and_then(|pid| {
                    projects_clone
                        .iter()
                        .find(|p| &p.id == pid)
                        .map(|p| p.name.clone())
                });
                let comments: Vec<TuiTodoComment> = t
                    .comments
                    .into_iter()
                    .map(|c| TuiTodoComment {
                        id: c.id,
                        author: c.author,
                        content: c.content,
                        comment_type: match c.comment_type.as_str() {
                            "progress" => TuiTodoCommentType::Progress,
                            "resolution" => TuiTodoCommentType::Resolution,
                            "activity" => TuiTodoCommentType::Activity,
                            _ => TuiTodoCommentType::Comment,
                        },
                        created_at: c.created_at.parse().unwrap_or_else(|_| Utc::now()),
                    })
                    .collect();

                TuiTodo {
                    id: t.id,
                    content: t.content,
                    status,
                    priority,
                    project_id: t.project_id,
                    project_name,
                    contexts: t.contexts,
                    due_date: t.due_date.and_then(|d| d.parse().ok()),
                    blocked_on: t.blocked_on,
                    created_at: t.created_at.parse().unwrap_or_else(|_| Utc::now()),
                    parent_id: t.parent_id,
                    seq_num: t.seq_num,
                    project_prefix: t.project_prefix,
                    comments,
                    notes: t.notes,
                }
            })
            .collect();

        let projects: Vec<TuiProject> = todos_resp
            .projects
            .into_iter()
            .map(|p| TuiProject {
                id: p.id,
                name: p.name,
                description: p.description,
                status: p.status,
                todo_count: 0,
                completed_count: 0,
                parent_id: p.parent_id,
                prefix: p.prefix,
                codebase_file_count: p.codebase_file_count,
            })
            .collect();

        let stats_resp = client
            .post(format!("{}/api/todos/stats", base_url))
            .header("X-API-Key", api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ "user_id": user_id }))
            .send()
            .await?;

        let stats_body = stats_resp.text().await?;
        let stats_resp: TodoStatsResponse = serde_json::from_str(&stats_body)?;

        let todo_stats = TodoStats {
            total: stats_resp.stats.total,
            backlog: stats_resp.stats.backlog,
            todo: stats_resp.stats.todo,
            in_progress: stats_resp.stats.in_progress,
            blocked: stats_resp.stats.blocked,
            done: stats_resp.stats.done,
            overdue: stats_resp.stats.overdue,
        };

        Ok((todos, projects, todo_stats))
    }

    fn debug_log(msg: &str) {
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("tui_debug.log")
            .and_then(|mut f| {
                use std::io::Write;
                writeln!(f, "[{}] {}", chrono::Local::now().format("%H:%M:%S%.3f"), msg)
            });
    }

    async fn fetch_initial_data(&self) {
        let user_id = &self.user_id;
        Self::debug_log(&format!("fetch_initial_data start user_id={} base_url={}", user_id, self.base_url));
        match self.fetch_user_stats(user_id).await {
            Ok(stats) => {
                Self::debug_log(&format!("stats OK: total={} working={} session={} lt={}",
                    stats.total_memories, stats.working_memory_count,
                    stats.session_memory_count, stats.long_term_memory_count));
                let mut state = self.state.lock().await;
                state.total_memories += stats.total_memories as u64;
                state.total_recalls += stats.total_retrievals as u64;
                state.tier_stats.working += stats.working_memory_count as u32;
                state.tier_stats.session += stats.session_memory_count as u32;
                state.tier_stats.long_term += stats.long_term_memory_count as u32;
                state.index_healthy = stats.vector_index_count >= stats.total_memories;
            }
            Err(e) => {
                Self::debug_log(&format!("stats FAILED: {}", e));
                let mut state = self.state.lock().await;
                state.set_error(format!("Failed to load stats: {}", e));
            }
        }
        Self::debug_log("fetching memory list...");
        match self.fetch_memory_list(user_id).await {
            Ok(list) => {
                Self::debug_log(&format!("memory list OK: {} memories", list.memories.len()));
                let mut state = self.state.lock().await;
                let mut memories = list.memories;
                memories.sort_by(|a, b| a.created_at.cmp(&b.created_at));
                for mem in memories {
                    state.type_stats.increment(&mem.memory_type);
                    for tag in &mem.tags {
                        state.entity_stats.total += 1;
                        if let Some(pos) = state
                            .entity_stats
                            .top_entities
                            .iter()
                            .position(|(e, _)| e == tag)
                        {
                            state.entity_stats.top_entities[pos].1 += 1;
                        } else {
                            state.entity_stats.top_entities.push((tag.clone(), 1));
                        }
                    }
                    state
                        .entity_stats
                        .top_entities
                        .sort_by(|a, b| b.1.cmp(&a.1));
                    state.entity_stats.top_entities.truncate(10);
                    let short_id = if mem.id.len() > 8 {
                        mem.id[..8].to_string()
                    } else {
                        mem.id.clone()
                    };
                    let n = state.graph_data.nodes.len() as f32;
                    let (x, y, z) = (
                        (n * 0.618).sin() * 0.35 + 0.5,
                        (n * 0.618).cos() * 0.35 + 0.5,
                        ((n * 0.3).sin() * 0.2 + 0.5).clamp(0.1, 0.9),
                    );
                    let content_preview = mem.content.clone();
                    let content = if mem.content.chars().count() > 40 {
                        let truncated: String = mem.content.chars().take(37).collect();
                        format!("{}...", truncated)
                    } else {
                        mem.content.clone()
                    };
                    state.add_event(MemoryEvent {
                        event_type: "HISTORY".to_string(),
                        timestamp: mem.created_at,
                        user_id: user_id.to_string(),
                        memory_id: Some(mem.id.clone()),
                        content_preview: Some(content_preview),
                        memory_type: Some(mem.memory_type.clone()),
                        importance: None,
                        count: None,
                        retrieval_mode: None,
                        latency_ms: None,
                        entities: if mem.tags.is_empty() {
                            None
                        } else {
                            Some(mem.tags.clone())
                        },
                        edge_weight: None,
                        from_id: None,
                        to_id: None,
                        results: None,
                    });
                    state.graph_data.nodes.push(GraphNode {
                        id: mem.id,
                        short_id,
                        content,
                        memory_type: mem.memory_type,
                        connections: 0,
                        x,
                        y,
                        z,
                    });
                }
                state.graph_stats.nodes = state.graph_data.nodes.len() as u32;
            }
            Err(e) => {
                Self::debug_log(&format!("memory list FAILED: {}", e));
            }
        }
        Self::debug_log("fetching graph stats...");
        match self.fetch_graph_stats(user_id).await {
            Ok(gs) => {
                Self::debug_log(&format!("graph stats OK: entities={} relationships={}", gs.entity_count, gs.relationship_count));
                let mut state = self.state.lock().await;
                state.total_edges += gs.relationship_count as u64;
                state.graph_stats.edges += gs.relationship_count as u32;
                state.total_entities += gs.entity_count as u64;
            }
            Err(e) => {
                Self::debug_log(&format!("graph stats FAILED: {}", e));
            }
        }
        Self::debug_log("fetching universe...");
        if let Ok(universe) = self.fetch_universe(user_id).await {
            let mut state = self.state.lock().await;
            let valid_stars: Vec<_> = universe
                .stars
                .iter()
                .filter(|s| s.name.len() >= 3 && !s.name.chars().all(|c| c.is_lowercase()))
                .collect();
            if !valid_stars.is_empty() {
                state.graph_data.nodes.clear();
                state.graph_data.edges.clear();
            }
            for (i, star) in valid_stars.iter().enumerate() {
                let short_id = if star.id.len() > 8 {
                    star.id[..8].to_string()
                } else {
                    star.id.clone()
                };
                let x = (star.position.x / 200.0 + 0.5).clamp(0.1, 0.9);
                let y = (star.position.y / 200.0 + 0.5).clamp(0.1, 0.9);
                let z = (star.position.z / 200.0 + 0.5).clamp(0.1, 0.9);
                let (px, py, pz) = if (x - 0.5).abs() < 0.05 && (y - 0.5).abs() < 0.05 {
                    let n = i as f32;
                    (
                        (n * 0.618).sin() * 0.35 + 0.5,
                        (n * 0.618).cos() * 0.35 + 0.5,
                        ((n * 0.3).sin() * 0.2 + 0.5).clamp(0.1, 0.9),
                    )
                } else {
                    (x, y, z)
                };
                state.graph_data.nodes.push(GraphNode {
                    id: star.id.clone(),
                    short_id,
                    content: star.name.clone(),
                    memory_type: star.entity_type.clone(),
                    connections: star.mention_count as u32,
                    x: px,
                    y: py,
                    z: pz,
                });
            }
            state.graph_stats.nodes = state.graph_data.nodes.len() as u32;
            for conn in universe.connections {
                let from_idx = state
                    .graph_data
                    .nodes
                    .iter()
                    .position(|n| n.id == conn.from_id);
                let to_idx = state
                    .graph_data
                    .nodes
                    .iter()
                    .position(|n| n.id == conn.to_id);
                if let (Some(fi), Some(ti)) = (from_idx, to_idx) {
                    state.graph_data.nodes[fi].connections += 1;
                    state.graph_data.nodes[ti].connections += 1;
                    let strength = conn.strength;
                    if strength > 0.7 {
                        state.graph_stats.strong_edges += 1;
                    } else if strength > 0.3 {
                        state.graph_stats.medium_edges += 1;
                    } else {
                        state.graph_stats.weak_edges += 1;
                    }
                    state.graph_data.edges.push(GraphEdge {
                        from_id: conn.from_id.clone(),
                        to_id: conn.to_id.clone(),
                        weight: strength,
                    });
                }
            }
            state.graph_stats.edges = state.graph_data.edges.len() as u32;
            let n = state.graph_data.nodes.len() as f64;
            if n > 1.0 {
                let max_edges = n * (n - 1.0) / 2.0;
                state.graph_stats.density =
                    (state.graph_data.edges.len() as f64 / max_edges) as f32;
            }
            state.graph_data.apply_force_layout(50);
        }
        // Fetch todos and projects
        Self::debug_log("fetching todos...");
        match self.fetch_todos(user_id).await {
            Ok((todos, projects, stats)) => {
                Self::debug_log(&format!("todos OK: {} todos, {} projects", todos.len(), projects.len()));
                let mut state = self.state.lock().await;
                state.todos = todos;
                // Mark projects with indexed files
                for p in &projects {
                    if p.codebase_file_count > 0 {
                        state.indexed_projects.insert(p.id.clone());
                    }
                }
                state.projects = projects;
                state.todo_stats = stats;
            }
            Err(e) => {
                Self::debug_log(&format!("todos FAILED: {}", e));
                let mut state = self.state.lock().await;
                state.set_error(format!("Failed to load todos: {}", e));
            }
        }
        // Fetch Claude Code context sessions (no auth required)
        Self::debug_log("fetching context sessions...");
        match self.fetch_context_sessions().await {
            Ok(sessions) => {
                Self::debug_log(&format!("context sessions OK: {} sessions", sessions.len()));
                let mut state = self.state.lock().await;
                state.context_sessions = sessions;
            }
            Err(e) => {
                Self::debug_log(&format!("context sessions FAILED: {}", e));
            }
        }
        Self::debug_log("fetch_initial_data complete, starting SSE connect...");
    }

    /// Fetch context sessions from Claude Code status line updates
    async fn fetch_context_sessions(
        &self,
    ) -> Result<Vec<crate::types::ContextSession>, Box<dyn std::error::Error + Send + Sync>> {
        let resp = self
            .client
            .get(format!("{}/api/context_status", self.base_url))
            .send()
            .await?;

        if !resp.status().is_success() {
            return Ok(Vec::new()); // Silently return empty if endpoint not available
        }

        let sessions: Vec<ContextSessionApi> = resp.json().await.unwrap_or_default();
        Ok(sessions
            .into_iter()
            .map(|s| crate::types::ContextSession {
                session_id: s.session_id.unwrap_or_default(),
                tokens_used: s.tokens_used,
                tokens_budget: s.tokens_budget,
                percent_used: s.percent_used,
                current_task: s.current_task,
                model: s.model,
                updated_at: s.updated_at.and_then(|d| d.parse().ok()),
            })
            .collect())
    }

    async fn fetch_todos(
        &self,
        user_id: &str,
    ) -> Result<(Vec<TuiTodo>, Vec<TuiProject>, TodoStats), Box<dyn std::error::Error + Send + Sync>>
    {
        let resp = self
            .client
            .post(format!("{}/api/todos/list", self.base_url))
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "user_id": user_id,
                "include_completed": true
            }))
            .send()
            .await?;

        let status = resp.status();
        let body = resp.text().await?;
        let todos_resp: TodoListResponse = serde_json::from_str(&body)
            .map_err(|e| format!("Parse todos: {} (status: {})", e, status))?;

        let projects_clone = todos_resp.projects.clone();
        let todos: Vec<TuiTodo> = todos_resp
            .todos
            .into_iter()
            .map(|t| {
                let status = match t.status.as_str() {
                    "backlog" => TuiTodoStatus::Backlog,
                    "todo" => TuiTodoStatus::Todo,
                    "in_progress" => TuiTodoStatus::InProgress,
                    "blocked" => TuiTodoStatus::Blocked,
                    "done" => TuiTodoStatus::Done,
                    "cancelled" => TuiTodoStatus::Cancelled,
                    _ => TuiTodoStatus::Todo,
                };
                let priority = match t.priority.as_str() {
                    "urgent" => TuiPriority::Urgent,
                    "high" => TuiPriority::High,
                    "medium" => TuiPriority::Medium,
                    "low" => TuiPriority::Low,
                    _ => TuiPriority::Medium,
                };
                let project_name = t.project_id.as_ref().and_then(|pid| {
                    projects_clone
                        .iter()
                        .find(|p| &p.id == pid)
                        .map(|p| p.name.clone())
                });
                let comments: Vec<TuiTodoComment> = t
                    .comments
                    .into_iter()
                    .map(|c| TuiTodoComment {
                        id: c.id,
                        author: c.author,
                        content: c.content,
                        comment_type: match c.comment_type.as_str() {
                            "progress" => TuiTodoCommentType::Progress,
                            "resolution" => TuiTodoCommentType::Resolution,
                            "activity" => TuiTodoCommentType::Activity,
                            _ => TuiTodoCommentType::Comment,
                        },
                        created_at: c.created_at.parse().unwrap_or_else(|_| Utc::now()),
                    })
                    .collect();

                TuiTodo {
                    id: t.id,
                    content: t.content,
                    status,
                    priority,
                    project_id: t.project_id,
                    project_name,
                    contexts: t.contexts,
                    due_date: t.due_date.and_then(|d| d.parse().ok()),
                    blocked_on: t.blocked_on,
                    created_at: t.created_at.parse().unwrap_or_else(|_| Utc::now()),
                    parent_id: t.parent_id,
                    seq_num: t.seq_num,
                    project_prefix: t.project_prefix,
                    comments,
                    notes: t.notes,
                }
            })
            .collect();

        let projects: Vec<TuiProject> = todos_resp
            .projects
            .into_iter()
            .map(|p| TuiProject {
                id: p.id,
                name: p.name,
                description: p.description,
                status: p.status,
                todo_count: 0,
                completed_count: 0,
                parent_id: p.parent_id,
                prefix: p.prefix,
                codebase_file_count: p.codebase_file_count,
            })
            .collect();

        let stats_resp_raw = self
            .client
            .post(format!("{}/api/todos/stats", self.base_url))
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ "user_id": user_id }))
            .send()
            .await?;
        let stats_status = stats_resp_raw.status();
        let stats_body = stats_resp_raw.text().await?;
        let stats_resp: TodoStatsResponse = serde_json::from_str(&stats_body)
            .map_err(|e| format!("Parse stats: {} (status: {})", e, stats_status))?;

        let todo_stats = TodoStats {
            total: stats_resp.stats.total,
            backlog: stats_resp.stats.backlog,
            todo: stats_resp.stats.todo,
            in_progress: stats_resp.stats.in_progress,
            blocked: stats_resp.stats.blocked,
            done: stats_resp.stats.done,
            overdue: stats_resp.stats.overdue,
        };

        Ok((todos, projects, todo_stats))
    }

    async fn fetch_user_stats(&self, user_id: &str) -> Result<MemoryStats, reqwest::Error> {
        self.client
            .get(format!("{}/api/users/{}/stats", self.base_url, user_id))
            .header("X-API-Key", &self.api_key)
            .send()
            .await?
            .json()
            .await
    }

    async fn fetch_memory_list(&self, user_id: &str) -> Result<ListResponse, reqwest::Error> {
        self.client
            .get(format!("{}/api/list/{}?limit=500", self.base_url, user_id))
            .header("X-API-Key", &self.api_key)
            .send()
            .await?
            .json()
            .await
    }

    async fn fetch_graph_stats(&self, user_id: &str) -> Result<GraphStatsResponse, reqwest::Error> {
        self.client
            .get(format!("{}/api/graph/{}/stats", self.base_url, user_id))
            .header("X-API-Key", &self.api_key)
            .send()
            .await?
            .json()
            .await
    }

    async fn fetch_universe(&self, user_id: &str) -> Result<MemoryUniverse, reqwest::Error> {
        self.client
            .get(format!("{}/api/graph/{}/universe", self.base_url, user_id))
            .header("X-API-Key", &self.api_key)
            .send()
            .await?
            .json()
            .await
    }

    async fn connect(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Self::debug_log(&format!("SSE connecting to: {}", self.url));
        let mut es = EventSource::new(
            self.client
                .get(&self.url)
                .header("X-API-Key", &self.api_key),
        )?;
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {
                    Self::debug_log("SSE connection opened");
                    self.state.lock().await.set_connected(true);
                }
                Ok(Event::Message(msg)) => {
                    if let Ok(e) = serde_json::from_str::<MemoryEvent>(&msg.data) {
                        let is_todo_event = e.event_type.starts_with("TODO_");
                        let is_project_event = e.event_type.starts_with("PROJECT_");

                        // Add event to activity feed
                        self.state.lock().await.add_event(e);

                        // Refetch todos on todo or project events for live updates
                        if is_todo_event || is_project_event {
                            if let Ok((todos, projects, stats)) = Self::poll_todos(
                                &self.client,
                                &self.base_url,
                                &self.api_key,
                                &self.user_id,
                            )
                            .await
                            {
                                let mut state = self.state.lock().await;
                                state.todos = todos;
                                // Mark projects with indexed files
                                for p in &projects {
                                    if p.codebase_file_count > 0 {
                                        state.indexed_projects.insert(p.id.clone());
                                    }
                                }
                                state.projects = projects;
                                state.todo_stats = stats;
                            }
                        }

                        // Also refresh context sessions on any event
                        if let Ok(sessions) = self.fetch_context_sessions().await {
                            let mut state = self.state.lock().await;
                            state.context_sessions = sessions;
                        }
                    }
                }
                Err(e) => {
                    Self::debug_log(&format!("SSE error: {:?}", e));
                    break;
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// PUBLIC API FUNCTIONS - Called from keyboard handlers in main.rs
// ============================================================================

/// Complete a todo by ID
pub async fn complete_todo(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    todo_id: &str,
) -> Result<(), String> {
    let client = Client::new();
    let resp = client
        .post(format!("{}/api/todos/{}/complete", base_url, todo_id))
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({ "user_id": user_id }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("Failed to complete todo: {}", resp.status()))
    }
}

/// Update todo status
pub async fn update_todo_status(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    todo_id: &str,
    new_status: &str,
) -> Result<(), String> {
    let client = Client::new();
    let resp = client
        .post(format!("{}/api/todos/{}/update", base_url, todo_id))
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "user_id": user_id,
            "status": new_status
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("Failed to update todo: {}", resp.status()))
    }
}

/// Delete a todo by ID
pub async fn delete_todo(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    todo_id: &str,
) -> Result<(), String> {
    let client = Client::new();
    let resp = client
        .delete(format!("{}/api/todos/{}", base_url, todo_id))
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({ "user_id": user_id }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("Failed to delete todo: {}", resp.status()))
    }
}

/// Cycle todo status: Todo -> InProgress -> Done
pub fn next_status(current: &str) -> &'static str {
    match current {
        "backlog" => "todo",
        "todo" => "in_progress",
        "in_progress" => "done",
        "blocked" => "in_progress",
        _ => "todo",
    }
}

/// Update todo priority
pub async fn update_todo_priority(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    todo_id: &str,
    priority: &str,
) -> Result<(), String> {
    let client = Client::new();
    let resp = client
        .post(format!("{}/api/todos/{}/update", base_url, todo_id))
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "user_id": user_id,
            "priority": priority
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("Failed to update priority: {}", resp.status()))
    }
}

/// Reorder todo (move up/down within status group)
pub async fn reorder_todo(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    todo_id: &str,
    direction: &str,
) -> Result<(), String> {
    let client = Client::new();
    let resp = client
        .post(format!("{}/api/todos/{}/reorder", base_url, todo_id))
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "user_id": user_id,
            "direction": direction
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("Failed to reorder todo: {}", resp.status()))
    }
}

/// Manual refresh of todos and projects - called on F5
pub async fn refresh_todos(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    state: &std::sync::Arc<tokio::sync::Mutex<crate::types::AppState>>,
) -> Result<(), String> {
    // Reuse the existing poll_todos function
    let client = Client::new();
    match MemoryStream::poll_todos(&client, base_url, api_key, user_id).await {
        Ok((todos, projects, stats)) => {
            let mut s = state.lock().await;
            s.todos = todos;
            // Mark projects with indexed files
            for p in &projects {
                if p.codebase_file_count > 0 {
                    s.indexed_projects.insert(p.id.clone());
                }
            }
            s.projects = projects;
            s.todo_stats = stats;
            Ok(())
        }
        Err(e) => Err(format!("Failed to refresh: {}", e)),
    }
}

// =============================================================================
// LINEAGE API FUNCTIONS
// =============================================================================

use crate::types::{LineageEdge, LineageNode, LineageTrace};
use std::collections::HashMap;

/// Fetch lineage trace for a memory/todo
pub async fn fetch_lineage_trace(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    memory_id: &str,
    direction: &str,
    state: &std::sync::Arc<tokio::sync::Mutex<crate::types::AppState>>,
) -> Result<(), String> {
    let client = Client::new();
    let url = format!("{}/api/lineage/trace", base_url);

    #[derive(serde::Serialize)]
    struct TraceRequest {
        user_id: String,
        memory_id: String,
        direction: String,
        max_depth: u32,
    }

    #[derive(serde::Deserialize)]
    struct TraceResponse {
        root: String,
        direction: String,
        edges: Vec<EdgeInfo>,
        path: Vec<String>,
        depth: usize,
    }

    #[derive(serde::Deserialize)]
    struct EdgeInfo {
        id: String,
        from: String,
        to: String,
        relation: String,
        confidence: f32,
        source: String,
    }

    let request = TraceRequest {
        user_id: user_id.to_string(),
        memory_id: memory_id.to_string(),
        direction: direction.to_string(),
        max_depth: 10,
    };

    let response = client
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, body));
    }

    let trace_resp: TraceResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;

    // Convert to LineageTrace
    let edges: Vec<LineageEdge> = trace_resp
        .edges
        .into_iter()
        .map(|e| LineageEdge {
            id: e.id,
            from_id: e.from,
            to_id: e.to,
            relation: e.relation,
            confidence: e.confidence,
            source: e.source,
        })
        .collect();

    // Build nodes map from path (we'll need to fetch memory details separately or use placeholders)
    let mut nodes: HashMap<String, LineageNode> = HashMap::new();
    for node_id in &trace_resp.path {
        nodes.insert(
            node_id.clone(),
            LineageNode {
                id: node_id.clone(),
                short_id: node_id.chars().take(8).collect(),
                content_preview: format!("Memory {}", &node_id[..8.min(node_id.len())]),
                memory_type: "Unknown".to_string(),
            },
        );
    }

    // Also add root if not in path
    if !nodes.contains_key(&trace_resp.root) {
        nodes.insert(
            trace_resp.root.clone(),
            LineageNode {
                id: trace_resp.root.clone(),
                short_id: trace_resp.root.chars().take(8).collect(),
                content_preview: format!(
                    "Root {}",
                    &trace_resp.root[..8.min(trace_resp.root.len())]
                ),
                memory_type: "Unknown".to_string(),
            },
        );
    }

    let trace = LineageTrace {
        root_id: trace_resp.root,
        direction: trace_resp.direction,
        edges,
        nodes,
        path: trace_resp.path,
        depth: trace_resp.depth,
    };

    let mut s = state.lock().await;
    s.set_lineage_trace(trace);
    Ok(())
}

/// Fetch lineage trace with node details (fetches memory info for each node)
pub async fn fetch_lineage_with_details(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    memory_id: &str,
    direction: &str,
    state: &std::sync::Arc<tokio::sync::Mutex<crate::types::AppState>>,
) -> Result<(), String> {
    let client = Client::new();
    let url = format!("{}/api/lineage/trace", base_url);

    #[derive(serde::Serialize)]
    struct TraceRequest {
        user_id: String,
        memory_id: String,
        direction: String,
        max_depth: u32,
    }

    #[derive(serde::Deserialize)]
    struct TraceResponse {
        root: String,
        direction: String,
        edges: Vec<EdgeInfo>,
        path: Vec<String>,
        depth: usize,
    }

    #[derive(serde::Deserialize)]
    struct EdgeInfo {
        id: String,
        from: String,
        to: String,
        relation: String,
        confidence: f32,
        source: String,
    }

    let request = TraceRequest {
        user_id: user_id.to_string(),
        memory_id: memory_id.to_string(),
        direction: direction.to_string(),
        max_depth: 10,
    };

    let response = client
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, body));
    }

    let trace_resp: TraceResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;

    // Collect all unique node IDs
    let mut node_ids: Vec<String> = trace_resp.path.clone();
    if !node_ids.contains(&trace_resp.root) {
        node_ids.push(trace_resp.root.clone());
    }
    for edge in &trace_resp.edges {
        if !node_ids.contains(&edge.from) {
            node_ids.push(edge.from.clone());
        }
        if !node_ids.contains(&edge.to) {
            node_ids.push(edge.to.clone());
        }
    }

    // Fetch memory details for each node
    let mut nodes: HashMap<String, LineageNode> = HashMap::new();
    for node_id in &node_ids {
        // Try to get memory info
        if let Ok(Some(node_info)) =
            fetch_memory_info(&client, base_url, api_key, user_id, node_id).await
        {
            nodes.insert(node_id.clone(), node_info);
        } else {
            // Fallback to placeholder
            nodes.insert(
                node_id.clone(),
                LineageNode {
                    id: node_id.clone(),
                    short_id: node_id.chars().take(8).collect(),
                    content_preview: format!("Memory {}", &node_id[..8.min(node_id.len())]),
                    memory_type: "Unknown".to_string(),
                },
            );
        }
    }

    let edges: Vec<LineageEdge> = trace_resp
        .edges
        .into_iter()
        .map(|e| LineageEdge {
            id: e.id,
            from_id: e.from,
            to_id: e.to,
            relation: e.relation,
            confidence: e.confidence,
            source: e.source,
        })
        .collect();

    let trace = LineageTrace {
        root_id: trace_resp.root,
        direction: trace_resp.direction,
        edges,
        nodes,
        path: trace_resp.path,
        depth: trace_resp.depth,
    };

    let mut s = state.lock().await;
    s.set_lineage_trace(trace);
    Ok(())
}

/// Fetch memory info for a single memory ID
async fn fetch_memory_info(
    client: &Client,
    base_url: &str,
    api_key: &str,
    user_id: &str,
    memory_id: &str,
) -> Result<Option<LineageNode>, String> {
    // Try the recall endpoint with the specific ID
    let url = format!("{}/api/recall", base_url);

    #[derive(serde::Serialize)]
    struct RecallRequest {
        user_id: String,
        query: String,
        limit: u32,
    }

    #[derive(serde::Deserialize)]
    struct RecallResponse {
        memories: Vec<MemoryInfo>,
    }

    #[derive(serde::Deserialize)]
    struct MemoryInfo {
        id: String,
        content: String,
        memory_type: String,
    }

    // Search by ID prefix
    let request = RecallRequest {
        user_id: user_id.to_string(),
        query: memory_id.chars().take(8).collect(),
        limit: 5,
    };

    let response = client
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        return Ok(None);
    }

    let recall_resp: RecallResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;

    // Find matching memory
    for mem in recall_resp.memories {
        if mem.id == memory_id || mem.id.starts_with(&memory_id[..8.min(memory_id.len())]) {
            return Ok(Some(LineageNode {
                id: mem.id.clone(),
                short_id: mem.id.chars().take(8).collect(),
                content_preview: mem.content.chars().take(30).collect(),
                memory_type: mem.memory_type,
            }));
        }
    }

    Ok(None)
}

/// Confirm a lineage edge
pub async fn confirm_lineage_edge(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    edge_id: &str,
) -> Result<String, String> {
    let client = Client::new();
    let url = format!("{}/api/lineage/confirm", base_url);

    #[derive(serde::Serialize)]
    struct ConfirmRequest {
        user_id: String,
        edge_id: String,
    }

    #[derive(serde::Deserialize)]
    struct ConfirmResponse {
        message: String,
    }

    let request = ConfirmRequest {
        user_id: user_id.to_string(),
        edge_id: edge_id.to_string(),
    };

    let response = client
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, body));
    }

    let resp: ConfirmResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;

    Ok(resp.message)
}

/// Reject a lineage edge
pub async fn reject_lineage_edge(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    edge_id: &str,
) -> Result<String, String> {
    let client = Client::new();
    let url = format!("{}/api/lineage/reject", base_url);

    #[derive(serde::Serialize)]
    struct RejectRequest {
        user_id: String,
        edge_id: String,
    }

    #[derive(serde::Deserialize)]
    struct RejectResponse {
        message: String,
    }

    let request = RejectRequest {
        user_id: user_id.to_string(),
        edge_id: edge_id.to_string(),
    };

    let response = client
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, body));
    }

    let resp: RejectResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;

    Ok(resp.message)
}

// ═══════════════════════════════════════════════════════════════════════════
// FILE MEMORY / CODEBASE INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════

/// Fetch files for a project
pub async fn fetch_project_files(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    project_id: &str,
) -> Result<Vec<TuiFileMemory>, String> {
    let client = Client::new();
    let url = format!("{}/api/projects/{}/files", base_url, project_id);

    #[derive(serde::Serialize)]
    struct FilesRequest {
        user_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<usize>,
    }

    #[derive(serde::Deserialize)]
    struct FilesResponse {
        files: Vec<FileApiItem>,
    }

    #[derive(serde::Deserialize)]
    struct FileApiItem {
        id: String,
        path: String,
        #[serde(default)]
        absolute_path: String,
        file_type: String,
        #[serde(default)]
        summary: String,
        #[serde(default)]
        key_items: Vec<String>,
        #[serde(default)]
        access_count: u32,
        #[serde(default)]
        last_accessed: String,
        #[serde(default)]
        heat_score: u8,
        #[serde(default)]
        size_bytes: u64,
        #[serde(default)]
        line_count: usize,
    }

    let request = FilesRequest {
        user_id: user_id.to_string(),
        limit: Some(1000), // Increased limit for tree view
    };

    let response = client
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, body));
    }

    let resp: FilesResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;

    Ok(resp
        .files
        .into_iter()
        .map(|f| TuiFileMemory {
            id: f.id,
            path: f.path,
            absolute_path: f.absolute_path,
            file_type: f.file_type,
            summary: f.summary,
            key_items: f.key_items,
            access_count: f.access_count,
            last_accessed: f.last_accessed,
            heat_score: f.heat_score,
            size_bytes: f.size_bytes,
            line_count: f.line_count,
        })
        .collect())
}

/// Scan project codebase (discover files)
pub async fn scan_project_codebase(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    project_id: &str,
    root_path: &str,
) -> Result<usize, String> {
    let client = Client::new();
    let url = format!("{}/api/projects/{}/scan", base_url, project_id);

    #[derive(serde::Serialize)]
    struct ScanRequest {
        user_id: String,
        codebase_path: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        force: Option<bool>,
    }

    #[derive(serde::Deserialize)]
    struct ScanResponse {
        #[serde(default)]
        eligible_files: usize,
    }

    let request = ScanRequest {
        user_id: user_id.to_string(),
        codebase_path: root_path.to_string(),
        force: Some(false),
    };

    let response = client
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, body));
    }

    let resp: ScanResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;

    Ok(resp.eligible_files)
}

/// Index project codebase (extract summaries and key items)
pub async fn index_project_codebase(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    project_id: &str,
    root_path: &str,
) -> Result<usize, String> {
    let client = Client::new();
    let url = format!("{}/api/projects/{}/index", base_url, project_id);

    #[derive(serde::Serialize)]
    struct IndexRequest {
        user_id: String,
        codebase_path: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        force: Option<bool>,
    }

    #[derive(serde::Deserialize)]
    struct IndexResponse {
        #[serde(default)]
        files_indexed: usize,
    }

    let request = IndexRequest {
        user_id: user_id.to_string(),
        codebase_path: root_path.to_string(),
        force: Some(true), // Always force re-index from TUI
    };

    let response = client
        .post(&url)
        .header("X-API-Key", api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("API error {}: {}", status, body));
    }

    let resp: IndexResponse = response
        .json()
        .await
        .map_err(|e| format!("Parse error: {}", e))?;

    Ok(resp.files_indexed)
}

/// Read file content from disk for preview
/// Returns lines with a reasonable limit to avoid memory issues
pub fn read_file_content(path: &str, max_lines: usize) -> Result<Vec<String>, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path).map_err(|e| format!("Cannot open file: {}", e))?;
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader
        .lines()
        .take(max_lines)
        .filter_map(|l| l.ok())
        .map(|l| {
            // Truncate very long lines for display
            if l.chars().count() > 200 {
                let truncated: String = l.chars().take(197).collect();
                format!("{}...", truncated)
            } else {
                l
            }
        })
        .collect();

    Ok(lines)
}
