use crate::types::{
    AppState, GraphEdge, GraphNode, MemoryEvent, TodoStats, TuiPriority, TuiProject, TuiTodo,
    TuiTodoStatus,
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
}

#[derive(Debug, Deserialize, Clone)]
struct ProjectApiItem {
    id: String,
    name: String,
    description: Option<String>,
    status: String,
    #[serde(default)]
    parent_id: Option<String>,
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
    pub fn new(url: &str, api_key: &str, user_id: &str, state: Arc<Mutex<AppState>>) -> Self {
        let base_url = url
            .trim_end_matches("/api/events")
            .trim_end_matches("/events")
            .to_string();
        Self {
            url: url.to_string(),
            base_url,
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

    async fn fetch_initial_data(&self) {
        let user_id = &self.user_id;
        match self.fetch_user_stats(user_id).await {
            Ok(stats) => {
                let mut state = self.state.lock().await;
                state.total_memories += stats.total_memories as u64;
                state.total_recalls += stats.total_retrievals as u64;
                state.tier_stats.working += stats.working_memory_count as u32;
                state.tier_stats.session += stats.session_memory_count as u32;
                state.tier_stats.long_term += stats.long_term_memory_count as u32;
                state.index_healthy = stats.vector_index_count >= stats.total_memories;
            }
            Err(e) => {
                let mut state = self.state.lock().await;
                state.set_error(format!("Failed to load stats: {}", e));
            }
        }
        if let Ok(list) = self.fetch_memory_list(user_id).await {
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
                state.entity_stats.top_entities.sort_by(|a, b| b.1.cmp(&a.1));
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
                let content = if mem.content.len() > 40 {
                    format!("{}...", &mem.content[..37.min(mem.content.len())])
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
        if let Ok(gs) = self.fetch_graph_stats(user_id).await {
            let mut state = self.state.lock().await;
            state.total_edges += gs.relationship_count as u64;
            state.graph_stats.edges += gs.relationship_count as u32;
            state.total_entities += gs.entity_count as u64;
        }
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
        match self.fetch_todos(user_id).await {
            Ok((todos, projects, stats)) => {
                let mut state = self.state.lock().await;
                state.todos = todos;
                state.projects = projects;
                state.todo_stats = stats;
            }
            Err(e) => {
                let mut state = self.state.lock().await;
                state.set_error(format!("Failed to load todos: {}", e));
            }
        }
        // Fetch Claude Code context sessions (no auth required)
        if let Ok(sessions) = self.fetch_context_sessions().await {
            let mut state = self.state.lock().await;
            state.context_sessions = sessions;
        }
    }

    /// Fetch context sessions from Claude Code status line updates
    async fn fetch_context_sessions(&self) -> Result<Vec<crate::types::ContextSession>, Box<dyn std::error::Error + Send + Sync>> {
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
    ) -> Result<(Vec<TuiTodo>, Vec<TuiProject>, TodoStats), Box<dyn std::error::Error + Send + Sync>> {
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
        let mut es = EventSource::new(
            self.client
                .get(&self.url)
                .header("X-API-Key", &self.api_key),
        )?;
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {
                    self.state.lock().await.set_connected(true);
                }
                Ok(Event::Message(msg)) => {
                    if let Ok(e) = serde_json::from_str::<MemoryEvent>(&msg.data) {
                        let is_todo_event = e.event_type.starts_with("TODO_");

                        // Add event to activity feed
                        self.state.lock().await.add_event(e);

                        // Refetch todos on todo events for live updates
                        if is_todo_event {
                            if let Ok((todos, projects, stats)) =
                                Self::poll_todos(&self.client, &self.base_url, &self.api_key, &self.user_id).await
                            {
                                let mut state = self.state.lock().await;
                                state.todos = todos;
                                state.projects = projects;
                                state.todo_stats = stats;
                            }
                        }
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) | Err(_) => break,
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
