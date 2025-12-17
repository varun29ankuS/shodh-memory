use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use ratatui::style::Color;
use std::time::Instant;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ViewMode {
    #[default]
    Dashboard,
    ActivityLogs,
    GraphList,
    GraphMap,
}

#[derive(Debug, Clone, Default)]
pub struct TierStats {
    pub working: u32,
    pub session: u32,
    pub long_term: u32,
}

impl TierStats {
    pub fn total(&self) -> u32 {
        self.working + self.session + self.long_term
    }
}

#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub nodes: u32,
    pub edges: u32,
    pub density: f32,
    pub avg_weight: f32,
    pub strong_edges: u32,
    pub medium_edges: u32,
    pub weak_edges: u32,
}

#[derive(Debug, Clone, Default)]
pub struct RetrievalStats {
    pub semantic: u32,
    pub associative: u32,
    pub hybrid: u32,
    pub avg_latency_ms: f32,
}

impl RetrievalStats {
    pub fn total(&self) -> u32 {
        self.semantic + self.associative + self.hybrid
    }
}

#[derive(Debug, Clone, Default)]
pub struct EntityStats {
    pub total: u32,
    pub top_entities: Vec<(String, u32)>,
}

#[derive(Debug, Clone, Default)]
pub struct ConsolidationStats {
    pub strengthened: u32,
    pub decayed: u32,
    pub promoted: u32,
    pub cycles: u32,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub short_id: String,
    pub content: String,
    pub memory_type: String,
    pub connections: u32,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from_id: String,
    pub to_id: String,
    pub weight: f32,
}

#[derive(Debug, Clone, Default)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub selected_node: usize,
    pub clusters: HashMap<String, Vec<String>>,
}

impl GraphData {
    pub fn selected(&self) -> Option<&GraphNode> {
        self.nodes.get(self.selected_node)
    }

    pub fn edges_from_selected(&self) -> Vec<(&GraphEdge, Option<&GraphNode>)> {
        if let Some(node) = self.selected() {
            self.edges.iter()
                .filter(|e| e.from_id == node.id)
                .map(|e| (e, self.nodes.iter().find(|n| n.id == e.to_id)))
                .collect()
        } else {
            vec![]
        }
    }

    pub fn select_next(&mut self) {
        if !self.nodes.is_empty() {
            self.selected_node = (self.selected_node + 1) % self.nodes.len();
        }
    }

    pub fn select_prev(&mut self) {
        if !self.nodes.is_empty() {
            self.selected_node = self.selected_node.checked_sub(1).unwrap_or(self.nodes.len() - 1);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    #[serde(default)]
    pub memory_id: Option<String>,
    #[serde(default)]
    pub content_preview: Option<String>,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
    #[serde(default)]
    pub count: Option<usize>,
    #[serde(default)]
    pub retrieval_mode: Option<String>,
    #[serde(default)]
    pub latency_ms: Option<f32>,
    #[serde(default)]
    pub entities: Option<Vec<String>>,
    #[serde(default)]
    pub edge_weight: Option<f32>,
    #[serde(default)]
    pub from_id: Option<String>,
    #[serde(default)]
    pub to_id: Option<String>,
}

impl MemoryEvent {
    pub fn event_color(&self) -> Color {
        match self.event_type.as_str() {
            "CREATE" => Color::Green,
            "RETRIEVE" => Color::Cyan,
            "DELETE" => Color::Red,
            "UPDATE" => Color::Yellow,
            "GRAPH_UPDATE" => Color::Magenta,
            "CONSOLIDATE" => Color::LightBlue,
            "STRENGTHEN" => Color::LightGreen,
            "DECAY" => Color::Gray,
            "PROMOTE" => Color::LightYellow,
            "HISTORY" => Color::DarkGray,
            _ => Color::White,
        }
    }

    pub fn event_icon(&self) -> &'static str {
        match self.event_type.as_str() {
            "CREATE" => "●",
            "RETRIEVE" => "◎",
            "DELETE" => "○",
            "UPDATE" => "◐",
            "GRAPH_UPDATE" => "◆",
            "CONSOLIDATE" => "⟳",
            "STRENGTHEN" => "↑",
            "DECAY" => "↓",
            "PROMOTE" => "⇧",
            "HISTORY" => "◌",
            _ => "•",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationState {
    SlideIn(f32),
    Visible,
    Dissolve(f32),
    Done,
}

impl AnimationState {
    pub fn is_done(&self) -> bool {
        matches!(self, AnimationState::Done)
    }

    pub fn opacity(&self) -> f32 {
        match self {
            AnimationState::SlideIn(p) => *p,
            AnimationState::Visible => 1.0,
            AnimationState::Dissolve(p) => *p,
            AnimationState::Done => 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisplayEvent {
    pub event: MemoryEvent,
    pub received_at: Instant,
    pub animation: AnimationState,
}

impl DisplayEvent {
    pub fn new(event: MemoryEvent) -> Self {
        Self {
            event,
            received_at: Instant::now(),
            animation: AnimationState::SlideIn(0.0),
        }
    }

    pub fn time_ago(&self) -> String {
        let now = chrono::Utc::now();
        let elapsed = (now - self.event.timestamp).num_seconds();
        if elapsed < 0 {
            self.event.timestamp.format("%H:%M").to_string()
        } else if elapsed < 60 {
            format!("{}s", elapsed)
        } else if elapsed < 3600 {
            format!("{}m", elapsed / 60)
        } else if elapsed < 86400 {
            format!("{}h", elapsed / 3600)
        } else {
            self.event.timestamp.format("%m/%d %H:%M").to_string()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TypeStats {
    pub context: u32,
    pub learning: u32,
    pub decision: u32,
    pub error: u32,
    pub pattern: u32,
    pub discovery: u32,
    pub task: u32,
    pub conversation: u32,
}

impl TypeStats {
    pub fn increment(&mut self, type_name: &str) {
        match type_name.to_lowercase().as_str() {
            "context" => self.context += 1,
            "learning" => self.learning += 1,
            "decision" => self.decision += 1,
            "error" => self.error += 1,
            "pattern" => self.pattern += 1,
            "discovery" => self.discovery += 1,
            "task" => self.task += 1,
            "conversation" => self.conversation += 1,
            _ => {}
        }
    }

    pub fn total(&self) -> u32 {
        self.context + self.learning + self.decision + self.error
            + self.pattern + self.discovery + self.task + self.conversation
    }

    pub fn as_vec(&self) -> Vec<(&'static str, u32, Color)> {
        vec![
            ("Context", self.context, Color::Cyan),
            ("Learning", self.learning, Color::Green),
            ("Decision", self.decision, Color::Yellow),
            ("Discovery", self.discovery, Color::Magenta),
            ("Task", self.task, Color::Blue),
            ("Pattern", self.pattern, Color::LightCyan),
            ("Error", self.error, Color::Red),
            ("Conversation", self.conversation, Color::White),
        ]
    }
}

pub struct AppState {
    pub events: VecDeque<DisplayEvent>,
    pub view_mode: ViewMode,
    pub type_stats: TypeStats,
    pub tier_stats: TierStats,
    pub graph_stats: GraphStats,
    pub retrieval_stats: RetrievalStats,
    pub entity_stats: EntityStats,
    pub consolidation_stats: ConsolidationStats,
    pub graph_data: GraphData,
    pub total_memories: u64,
    pub total_edges: u64,
    pub total_recalls: u64,
    pub total_entities: u64,
    pub index_healthy: bool,
    pub connected: bool,
    pub scroll_offset: usize,
    pub animation_tick: u64,
    pub max_events: usize,
    pub session_start: Instant,
    pub current_user: String,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            view_mode: ViewMode::Dashboard,
            type_stats: TypeStats::default(),
            tier_stats: TierStats::default(),
            graph_stats: GraphStats::default(),
            retrieval_stats: RetrievalStats::default(),
            entity_stats: EntityStats::default(),
            consolidation_stats: ConsolidationStats::default(),
            graph_data: GraphData::default(),
            total_memories: 0,
            total_edges: 0,
            total_recalls: 0,
            total_entities: 0,
            index_healthy: true,
            connected: false,
            scroll_offset: 0,
            animation_tick: 0,
            max_events: 100,
            session_start: Instant::now(),
            current_user: String::new(),
        }
    }

    pub fn set_view(&mut self, mode: ViewMode) {
        self.view_mode = mode;
        self.scroll_offset = 0;
    }

    pub fn cycle_view(&mut self) {
        self.view_mode = match self.view_mode {
            ViewMode::Dashboard => ViewMode::ActivityLogs,
            ViewMode::ActivityLogs => ViewMode::GraphList,
            ViewMode::GraphList => ViewMode::GraphMap,
            ViewMode::GraphMap => ViewMode::Dashboard,
        };
        self.scroll_offset = 0;
    }

    pub fn session_duration(&self) -> String {
        let secs = self.session_start.elapsed().as_secs();
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        let secs = secs % 60;
        format!("{:02}:{:02}:{:02}", hours, mins, secs)
    }

    pub fn add_event(&mut self, event: MemoryEvent) {
        match event.event_type.as_str() {
            "CREATE" => {
                self.total_memories += 1;
                self.tier_stats.working += 1;
                self.graph_stats.nodes += 1;
                if let Some(ref t) = event.memory_type { self.type_stats.increment(t); }
                if let Some(ref entities) = event.entities {
                    self.total_entities += entities.len() as u64;
                    for e in entities { self.add_entity(e.clone()); }
                }
                if let Some(ref id) = event.memory_id {
                    let content = event.content_preview.clone().unwrap_or_default();
                    let mem_type = event.memory_type.clone().unwrap_or_else(|| "Unknown".to_string());
                    self.add_graph_node(id.clone(), content, mem_type);
                }
            }
            "RETRIEVE" => {
                self.total_recalls += 1;
                if let Some(ref mode) = event.retrieval_mode {
                    match mode.to_lowercase().as_str() {
                        "semantic" => self.retrieval_stats.semantic += 1,
                        "associative" => self.retrieval_stats.associative += 1,
                        "hybrid" => self.retrieval_stats.hybrid += 1,
                        _ => {}
                    }
                }
                if let Some(latency) = event.latency_ms {
                    let total = self.retrieval_stats.total();
                    if total > 0 {
                        self.retrieval_stats.avg_latency_ms = (self.retrieval_stats.avg_latency_ms * (total - 1) as f32 + latency) / total as f32;
                    }
                }
            }
            "DELETE" => {
                let count = event.count.unwrap_or(1);
                self.total_memories = self.total_memories.saturating_sub(count as u64);
                self.graph_stats.nodes = self.graph_stats.nodes.saturating_sub(count as u32);
            }
            "GRAPH_UPDATE" => {
                self.total_edges += 1;
                self.graph_stats.edges += 1;
                if let (Some(from), Some(to), Some(weight)) = (&event.from_id, &event.to_id, event.edge_weight) {
                    self.add_graph_edge(from.clone(), to.clone(), weight);
                }
                self.update_graph_stats();
            }
            "STRENGTHEN" => self.consolidation_stats.strengthened += 1,
            "DECAY" => self.consolidation_stats.decayed += 1,
            "PROMOTE" => {
                self.consolidation_stats.promoted += 1;
                self.tier_stats.working = self.tier_stats.working.saturating_sub(1);
                self.tier_stats.session += 1;
            }
            "CONSOLIDATE" => self.consolidation_stats.cycles += 1,
            _ => {}
        }
        self.events.push_front(DisplayEvent::new(event));
        while self.events.len() > self.max_events { self.events.pop_back(); }
    }

    fn add_entity(&mut self, entity: String) {
        if let Some(pos) = self.entity_stats.top_entities.iter().position(|(e, _)| e == &entity) {
            self.entity_stats.top_entities[pos].1 += 1;
        } else {
            self.entity_stats.top_entities.push((entity, 1));
        }
        self.entity_stats.top_entities.sort_by(|a, b| b.1.cmp(&a.1));
        self.entity_stats.top_entities.truncate(10);
        self.entity_stats.total += 1;
    }

    fn add_graph_node(&mut self, id: String, content: String, memory_type: String) {
        let short_id = if id.len() > 8 { id[..8].to_string() } else { id.clone() };
        let n = self.graph_data.nodes.len() as f32;
        let x = (n * 0.618).sin() * 0.35 + 0.5;
        let y = (n * 0.618).cos() * 0.35 + 0.5;
        self.graph_data.nodes.push(GraphNode {
            id, short_id,
            content: if content.len() > 40 { format!("{}...", &content[..37]) } else { content },
            memory_type, connections: 0, x, y,
        });
    }

    fn add_graph_edge(&mut self, from_id: String, to_id: String, weight: f32) {
        self.graph_data.edges.push(GraphEdge { from_id: from_id.clone(), to_id: to_id.clone(), weight });
        for node in &mut self.graph_data.nodes {
            if node.id == from_id || node.id == to_id { node.connections += 1; }
        }
        self.graph_data.nodes.sort_by(|a, b| b.connections.cmp(&a.connections));
    }

    fn update_graph_stats(&mut self) {
        let n = self.graph_stats.nodes as f32;
        if n > 1.0 { self.graph_stats.density = self.graph_stats.edges as f32 / (n * (n - 1.0) / 2.0); }
        let (s, m, w) = self.graph_data.edges.iter().fold((0, 0, 0), |(s, m, w), e| {
            if e.weight >= 0.7 { (s + 1, m, w) } else if e.weight >= 0.4 { (s, m + 1, w) } else { (s, m, w + 1) }
        });
        self.graph_stats.strong_edges = s;
        self.graph_stats.medium_edges = m;
        self.graph_stats.weak_edges = w;
        if !self.graph_data.edges.is_empty() {
            self.graph_stats.avg_weight = self.graph_data.edges.iter().map(|e| e.weight).sum::<f32>() / self.graph_data.edges.len() as f32;
        }
    }

    pub fn scroll_up(&mut self) {
        match self.view_mode {
            ViewMode::GraphList | ViewMode::GraphMap => self.graph_data.select_prev(),
            _ => self.scroll_offset = self.scroll_offset.saturating_sub(1),
        }
    }

    pub fn scroll_down(&mut self) {
        match self.view_mode {
            ViewMode::GraphList | ViewMode::GraphMap => self.graph_data.select_next(),
            _ => {
                let max = self.events.len().saturating_sub(1);
                if self.scroll_offset < max { self.scroll_offset += 1; }
            }
        }
    }

    pub fn tick(&mut self) {
        self.animation_tick = self.animation_tick.wrapping_add(1);
        for event in &mut self.events {
            if let AnimationState::SlideIn(ref mut p) = event.animation {
                *p += 0.15;
                if *p >= 1.0 { event.animation = AnimationState::Visible; }
            }
        }
    }
}
