use crate::types::{AppState, MemoryEvent, GraphNode};
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
        let base_url = url.trim_end_matches("/api/events").trim_end_matches("/events").to_string();
        Self { url: url.to_string(), base_url, api_key: api_key.to_string(), user_id: user_id.to_string(), state, client: Client::new() }
    }

    pub async fn run(&self) {
        self.fetch_initial_data().await;
        loop {
            match self.connect().await { Ok(()) => {} Err(_) => {} }
            { let mut state = self.state.lock().await; state.connected = false; }
            sleep(Duration::from_secs(3)).await;
        }
    }

    async fn fetch_initial_data(&self) {
        let user_id = &self.user_id;
        if let Ok(stats) = self.fetch_user_stats(user_id).await {
            let mut state = self.state.lock().await;
            state.total_memories += stats.total_memories as u64;
            state.total_recalls += stats.total_retrievals as u64;
            state.tier_stats.working += stats.working_memory_count as u32;
            state.tier_stats.session += stats.session_memory_count as u32;
            state.tier_stats.long_term += stats.long_term_memory_count as u32;
            state.index_healthy = stats.vector_index_count >= stats.total_memories;
        }
        if let Ok(list) = self.fetch_memory_list(user_id).await {
            let mut state = self.state.lock().await;
            for mem in list.memories {
                state.type_stats.increment(&mem.memory_type);
                for tag in &mem.tags {
                    state.entity_stats.total += 1;
                    if let Some(pos) = state.entity_stats.top_entities.iter().position(|(e, _)| e == tag) {
                        state.entity_stats.top_entities[pos].1 += 1;
                    } else {
                        state.entity_stats.top_entities.push((tag.clone(), 1));
                    }
                }
                state.entity_stats.top_entities.sort_by(|a, b| b.1.cmp(&a.1));
                state.entity_stats.top_entities.truncate(10);
                let short_id = if mem.id.len() > 8 { mem.id[..8].to_string() } else { mem.id.clone() };
                let n = state.graph_data.nodes.len() as f32;
                let (x, y) = ((n * 0.618).sin() * 0.35 + 0.5, (n * 0.618).cos() * 0.35 + 0.5);
                let content_preview = if mem.content.len() > 100 { format!("{}...", &mem.content[..97.min(mem.content.len())]) } else { mem.content.clone() };
                let content = if mem.content.len() > 40 { format!("{}...", &mem.content[..37.min(mem.content.len())]) } else { mem.content.clone() };
                // Add as event for Activity feed
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
                    entities: if mem.tags.is_empty() { None } else { Some(mem.tags.clone()) },
                    edge_weight: None,
                    from_id: None,
                    to_id: None,
                });
                state.graph_data.nodes.push(GraphNode { id: mem.id, short_id, content, memory_type: mem.memory_type, connections: 0, x, y });
            }
            state.graph_stats.nodes = state.graph_data.nodes.len() as u32;
        }
        if let Ok(gs) = self.fetch_graph_stats(user_id).await {
            let mut state = self.state.lock().await;
            state.total_edges += gs.relationship_count as u64;
            state.graph_stats.edges += gs.relationship_count as u32;
            state.total_entities += gs.entity_count as u64;
        }
    }

    async fn fetch_user_stats(&self, user_id: &str) -> Result<MemoryStats, reqwest::Error> {
        self.client.get(format!("{}/api/users/{}/stats", self.base_url, user_id)).header("X-API-Key", &self.api_key).send().await?.json().await
    }

    async fn fetch_memory_list(&self, user_id: &str) -> Result<ListResponse, reqwest::Error> {
        self.client.get(format!("{}/api/list/{}?limit=500", self.base_url, user_id)).header("X-API-Key", &self.api_key).send().await?.json().await
    }

    async fn fetch_graph_stats(&self, user_id: &str) -> Result<GraphStatsResponse, reqwest::Error> {
        self.client.get(format!("{}/api/graph/{}/stats", self.base_url, user_id)).header("X-API-Key", &self.api_key).send().await?.json().await
    }

    async fn connect(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut es = EventSource::new(self.client.get(&self.url).header("X-API-Key", &self.api_key))?;
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => { self.state.lock().await.connected = true; }
                Ok(Event::Message(msg)) => { if let Ok(e) = serde_json::from_str::<MemoryEvent>(&msg.data) { self.state.lock().await.add_event(e); } }
                Err(reqwest_eventsource::Error::StreamEnded) | Err(_) => break,
            }
        }
        Ok(())
    }
}
