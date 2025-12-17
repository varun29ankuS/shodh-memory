use crate::types::{AppState, GraphEdge, GraphNode, MemoryEvent};
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

#[derive(Debug, Deserialize, Default)]
struct UniversePosition {
    #[serde(default)]
    x: f32,
    #[serde(default)]
    y: f32,
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
        loop {
            match self.connect().await {
                Ok(()) => {}
                Err(_) => {}
            }
            {
                let mut state = self.state.lock().await;
                state.connected = false;
            }
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
                let (x, y) = (
                    (n * 0.618).sin() * 0.35 + 0.5,
                    (n * 0.618).cos() * 0.35 + 0.5,
                );
                let content_preview = if mem.content.len() > 100 {
                    format!("{}...", &mem.content[..97.min(mem.content.len())])
                } else {
                    mem.content.clone()
                };
                let content = if mem.content.len() > 40 {
                    format!("{}...", &mem.content[..37.min(mem.content.len())])
                } else {
                    mem.content.clone()
                };
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
        // Fetch universe - entities as nodes, relationships as edges
        if let Ok(universe) = self.fetch_universe(user_id).await {
            let mut state = self.state.lock().await;

            // Add entity nodes from stars
            for (i, star) in universe.stars.iter().enumerate() {
                let short_id = if star.id.len() > 8 {
                    star.id[..8].to_string()
                } else {
                    star.id.clone()
                };
                // Normalize position to 0-1 range
                let x = (star.position.x / 100.0).clamp(0.1, 0.9);
                let y = (star.position.y / 100.0).clamp(0.1, 0.9);
                // Use golden angle for better distribution if position is zero
                let (px, py) = if x.abs() < 0.01 && y.abs() < 0.01 {
                    let n = i as f32;
                    (
                        (n * 0.618).sin() * 0.35 + 0.5,
                        (n * 0.618).cos() * 0.35 + 0.5,
                    )
                } else {
                    (x.abs().min(0.9).max(0.1), y.abs().min(0.9).max(0.1))
                };
                state.graph_data.nodes.push(GraphNode {
                    id: star.id.clone(),
                    short_id,
                    content: star.name.clone(),
                    memory_type: star.entity_type.clone(),
                    connections: star.mention_count as u32,
                    x: px,
                    y: py,
                });
            }
            state.graph_stats.nodes = state.graph_data.nodes.len() as u32;

            // Add edges from connections
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
        }
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
                    self.state.lock().await.connected = true;
                }
                Ok(Event::Message(msg)) => {
                    if let Ok(e) = serde_json::from_str::<MemoryEvent>(&msg.data) {
                        self.state.lock().await.add_event(e);
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) | Err(_) => break,
            }
        }
        Ok(())
    }
}
