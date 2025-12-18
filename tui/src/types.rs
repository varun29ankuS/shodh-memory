use chrono::{DateTime, Utc};
use ratatui::style::Color;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ═══════════════════════════════════════════════════════════════════════════
// ANIMATION SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// Easing functions for smooth animations
#[derive(Debug, Clone, Copy, Default)]
pub enum Easing {
    #[default]
    Linear,
    EaseOut,
    EaseIn,
    EaseInOut,
    Bounce,
    Elastic,
}

impl Easing {
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Easing::Linear => t,
            Easing::EaseOut => 1.0 - (1.0 - t).powi(3),
            Easing::EaseIn => t.powi(3),
            Easing::EaseInOut => {
                if t < 0.5 {
                    4.0 * t.powi(3)
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
            Easing::Bounce => {
                let n1 = 7.5625;
                let d1 = 2.75;
                if t < 1.0 / d1 {
                    n1 * t * t
                } else if t < 2.0 / d1 {
                    let t = t - 1.5 / d1;
                    n1 * t * t + 0.75
                } else if t < 2.5 / d1 {
                    let t = t - 2.25 / d1;
                    n1 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / d1;
                    n1 * t * t + 0.984375
                }
            }
            Easing::Elastic => {
                if t == 0.0 || t == 1.0 {
                    t
                } else {
                    let c4 = (2.0 * std::f32::consts::PI) / 3.0;
                    2.0_f32.powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0
                }
            }
        }
    }
}

/// Animation type defines how an element animates
#[derive(Debug, Clone)]
pub enum AnimationType {
    /// Fade from transparent to opaque
    FadeIn { duration_ms: u64 },
    /// Fade from opaque to transparent
    FadeOut { duration_ms: u64 },
    /// Slide in from a direction with fade
    SlideIn { direction: SlideDirection, distance: f32, duration_ms: u64 },
    /// Slide out to a direction with fade
    SlideOut { direction: SlideDirection, distance: f32, duration_ms: u64 },
    /// Scale from small to full size
    ScaleIn { from: f32, duration_ms: u64 },
    /// Pulsing glow effect (continuous)
    Pulse { min_intensity: f32, max_intensity: f32, period_ms: u64 },
    /// Attention-grabbing highlight flash
    Flash { color: Color, duration_ms: u64 },
    /// Color transition
    ColorTransition { from: Color, to: Color, duration_ms: u64 },
    /// Combined slide + fade + scale for new items
    Entrance { delay_ms: u64, duration_ms: u64 },
}

#[derive(Debug, Clone, Copy, Default)]
pub enum SlideDirection {
    #[default]
    Left,
    Right,
    Top,
    Bottom,
}

/// Animation instance tracking progress
#[derive(Debug, Clone)]
pub struct Animation {
    pub animation_type: AnimationType,
    pub easing: Easing,
    pub started_at: Instant,
    pub delay_elapsed: bool,
}

impl Animation {
    pub fn new(animation_type: AnimationType, easing: Easing) -> Self {
        Self {
            animation_type,
            easing,
            started_at: Instant::now(),
            delay_elapsed: false,
        }
    }

    pub fn entrance(index: usize) -> Self {
        Self::new(
            AnimationType::Entrance {
                delay_ms: (index as u64) * 30, // Quick stagger
                duration_ms: 400, // Snappy 400ms animation
            },
            Easing::EaseOut,
        )
    }

    pub fn slide_in_right() -> Self {
        Self::new(
            AnimationType::SlideIn {
                direction: SlideDirection::Right,
                distance: 20.0,
                duration_ms: 250,
            },
            Easing::EaseOut,
        )
    }

    pub fn fade_in() -> Self {
        Self::new(
            AnimationType::FadeIn { duration_ms: 200 },
            Easing::EaseOut,
        )
    }

    pub fn pulse() -> Self {
        Self::new(
            AnimationType::Pulse {
                min_intensity: 0.6,
                max_intensity: 1.0,
                period_ms: 1000,
            },
            Easing::EaseInOut,
        )
    }

    /// Get animation progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        let elapsed = self.started_at.elapsed().as_millis() as u64;

        let (delay_ms, duration_ms) = match &self.animation_type {
            AnimationType::FadeIn { duration_ms } => (0, *duration_ms),
            AnimationType::FadeOut { duration_ms } => (0, *duration_ms),
            AnimationType::SlideIn { duration_ms, .. } => (0, *duration_ms),
            AnimationType::SlideOut { duration_ms, .. } => (0, *duration_ms),
            AnimationType::ScaleIn { duration_ms, .. } => (0, *duration_ms),
            AnimationType::Flash { duration_ms, .. } => (0, *duration_ms),
            AnimationType::ColorTransition { duration_ms, .. } => (0, *duration_ms),
            AnimationType::Entrance { delay_ms, duration_ms } => (*delay_ms, *duration_ms),
            AnimationType::Pulse { period_ms, .. } => (0, *period_ms),
        };

        if elapsed < delay_ms {
            return 0.0;
        }

        let active_elapsed = elapsed - delay_ms;

        // Pulse animation loops
        if matches!(self.animation_type, AnimationType::Pulse { .. }) {
            let cycle_progress = (active_elapsed % duration_ms) as f32 / duration_ms as f32;
            // Sine wave for smooth pulse
            return (cycle_progress * std::f32::consts::PI * 2.0).sin() * 0.5 + 0.5;
        }

        (active_elapsed as f32 / duration_ms as f32).clamp(0.0, 1.0)
    }

    /// Check if animation is complete
    pub fn is_complete(&self) -> bool {
        if matches!(self.animation_type, AnimationType::Pulse { .. }) {
            return false; // Pulse never completes
        }
        self.progress() >= 1.0
    }

    /// Get current opacity (0.0 to 1.0)
    pub fn opacity(&self) -> f32 {
        let p = self.easing.apply(self.progress());
        match &self.animation_type {
            AnimationType::FadeIn { .. } => p,
            AnimationType::FadeOut { .. } => 1.0 - p,
            AnimationType::SlideIn { .. } => p,
            AnimationType::SlideOut { .. } => 1.0 - p,
            AnimationType::ScaleIn { .. } => p,
            AnimationType::Entrance { .. } => p,
            AnimationType::Pulse { min_intensity, max_intensity, .. } => {
                min_intensity + (max_intensity - min_intensity) * p
            }
            _ => 1.0,
        }
    }

    /// Get X offset for slide animations
    pub fn offset_x(&self) -> i16 {
        let p = self.easing.apply(self.progress());
        match &self.animation_type {
            AnimationType::SlideIn { direction, distance, .. } => {
                let remaining = 1.0 - p;
                match direction {
                    SlideDirection::Left => -(distance * remaining) as i16,
                    SlideDirection::Right => (distance * remaining) as i16,
                    _ => 0,
                }
            }
            AnimationType::SlideOut { direction, distance, .. } => {
                match direction {
                    SlideDirection::Left => -(distance * p) as i16,
                    SlideDirection::Right => (distance * p) as i16,
                    _ => 0,
                }
            }
            AnimationType::Entrance { .. } => {
                let remaining = 1.0 - p;
                (15.0 * remaining) as i16 // Slide from right
            }
            _ => 0,
        }
    }

    /// Get Y offset for slide animations
    pub fn offset_y(&self) -> i16 {
        let p = self.easing.apply(self.progress());
        match &self.animation_type {
            AnimationType::SlideIn { direction, distance, .. } => {
                let remaining = 1.0 - p;
                match direction {
                    SlideDirection::Top => -(distance * remaining) as i16,
                    SlideDirection::Bottom => (distance * remaining) as i16,
                    _ => 0,
                }
            }
            AnimationType::SlideOut { direction, distance, .. } => {
                match direction {
                    SlideDirection::Top => -(distance * p) as i16,
                    SlideDirection::Bottom => (distance * p) as i16,
                    _ => 0,
                }
            }
            AnimationType::Entrance { .. } => {
                let remaining = 1.0 - p;
                (3.0 * remaining) as i16 // Slight vertical offset
            }
            _ => 0,
        }
    }

    /// Get scale factor for scale animations
    pub fn scale(&self) -> f32 {
        let p = self.easing.apply(self.progress());
        match &self.animation_type {
            AnimationType::ScaleIn { from, .. } => from + (1.0 - from) * p,
            AnimationType::Entrance { .. } => 0.95 + 0.05 * p,
            _ => 1.0,
        }
    }

    /// Get glow/highlight intensity
    pub fn glow_intensity(&self) -> f32 {
        match &self.animation_type {
            AnimationType::Flash { .. } => {
                let p = self.progress();
                if p < 0.5 {
                    p * 2.0
                } else {
                    (1.0 - p) * 2.0
                }
            }
            AnimationType::Pulse { min_intensity, max_intensity, .. } => {
                let p = self.progress();
                min_intensity + (max_intensity - min_intensity) * p
            }
            _ => 0.0,
        }
    }
}

/// View transition state for smooth view changes
#[derive(Debug, Clone)]
pub struct ViewTransition {
    pub from_view: ViewMode,
    pub to_view: ViewMode,
    pub started_at: Instant,
    pub duration_ms: u64,
}

impl ViewTransition {
    pub fn new(from: ViewMode, to: ViewMode) -> Self {
        Self {
            from_view: from,
            to_view: to,
            started_at: Instant::now(),
            duration_ms: 250, // Quick snappy transition
        }
    }

    pub fn progress(&self) -> f32 {
        let elapsed = self.started_at.elapsed().as_millis() as f32;
        (elapsed / self.duration_ms as f32).clamp(0.0, 1.0)
    }

    pub fn is_complete(&self) -> bool {
        self.progress() >= 1.0
    }

    /// Get fade-out progress for old view
    pub fn fade_out(&self) -> f32 {
        let p = self.progress();
        if p < 0.5 {
            1.0 - (p * 2.0)
        } else {
            0.0
        }
    }

    /// Get fade-in progress for new view
    pub fn fade_in(&self) -> f32 {
        let p = self.progress();
        if p > 0.5 {
            (p - 0.5) * 2.0
        } else {
            0.0
        }
    }
}

/// Smooth scroll state for animated scrolling
#[derive(Debug, Clone)]
pub struct SmoothScroll {
    pub current: f32,
    pub target: f32,
    pub velocity: f32,
}

impl Default for SmoothScroll {
    fn default() -> Self {
        Self {
            current: 0.0,
            target: 0.0,
            velocity: 0.0,
        }
    }
}

impl SmoothScroll {
    pub fn set_target(&mut self, target: usize) {
        self.target = target as f32;
    }

    pub fn update(&mut self, dt: f32) {
        let diff = self.target - self.current;
        let spring_force = diff * 15.0; // Spring constant
        let damping = self.velocity * 8.0; // Damping factor
        let acceleration = spring_force - damping;
        self.velocity += acceleration * dt;
        self.current += self.velocity * dt;

        // Snap to target when close enough
        if diff.abs() < 0.01 && self.velocity.abs() < 0.01 {
            self.current = self.target;
            self.velocity = 0.0;
        }
    }

    pub fn current_offset(&self) -> usize {
        self.current.round().max(0.0) as usize
    }

    pub fn fractional_offset(&self) -> f32 {
        self.current - self.current.floor()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ViewMode {
    #[default]
    Dashboard,
    ActivityLogs,
    GraphList,
    GraphMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Theme {
    #[default]
    Dark,
    Light,
}

impl Theme {
    pub fn toggle(&self) -> Self {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }

    /// Background color
    pub fn bg(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(15, 15, 20),
            Theme::Light => Color::Rgb(250, 248, 245),
        }
    }

    /// Primary text color
    pub fn fg(&self) -> Color {
        match self {
            Theme::Dark => Color::White,
            Theme::Light => Color::Rgb(30, 30, 35),
        }
    }

    /// Secondary/dimmed text
    pub fn fg_dim(&self) -> Color {
        match self {
            Theme::Dark => Color::DarkGray,
            Theme::Light => Color::Rgb(120, 115, 110),
        }
    }

    /// Border color
    pub fn border(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(60, 60, 80),
            Theme::Light => Color::Rgb(200, 195, 190),
        }
    }

    /// Accent color (orange)
    pub fn accent(&self) -> Color {
        Color::Rgb(255, 140, 50)
    }

    /// Selection highlight background
    pub fn selection_bg(&self) -> Color {
        match self {
            Theme::Dark => Color::Rgb(40, 40, 55),
            Theme::Light => Color::Rgb(255, 240, 220),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SearchMode {
    #[default]
    Keyword,
    Semantic,
    Date,
}

impl SearchMode {
    pub fn label(&self) -> &'static str {
        match self {
            SearchMode::Keyword => "keyword",
            SearchMode::Semantic => "semantic",
            SearchMode::Date => "date",
        }
    }

    pub fn cycle(&self) -> SearchMode {
        match self {
            SearchMode::Keyword => SearchMode::Semantic,
            SearchMode::Semantic => SearchMode::Date,
            SearchMode::Date => SearchMode::Keyword,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub score: f32,
    pub created_at: DateTime<Utc>,
    pub tags: Vec<String>,
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
    pub z: f32,
}

impl GraphNode {
    /// Project 3D coordinates to 2D using perspective projection with rotation and tilt
    /// Returns (screen_x, screen_y, depth) where depth is used for z-sorting and brightness
    pub fn project_2d(&self, rotation: f32) -> (f32, f32, f32) {
        self.project_3d(rotation, 0.3) // Default tilt
    }

    /// Full 3D projection with rotation around Y-axis and tilt around X-axis
    pub fn project_3d(&self, rotation: f32, tilt: f32) -> (f32, f32, f32) {
        // Center coordinates around origin (-0.5 to 0.5)
        let cx = self.x - 0.5;
        let cy = self.y - 0.5;
        let cz = self.z - 0.5;

        // Rotate around Y axis (horizontal rotation)
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();
        let rx = cx * cos_r - cz * sin_r;
        let rz = cx * sin_r + cz * cos_r;
        let ry = cy;

        // Tilt around X axis (vertical tilt)
        let cos_t = tilt.cos();
        let sin_t = tilt.sin();
        let ty = ry * cos_t - rz * sin_t;
        let tz = ry * sin_t + rz * cos_t;
        let tx = rx;

        // Perspective projection (camera at z = -2)
        let camera_dist = 2.0;
        let perspective = camera_dist / (camera_dist + tz + 1.0);

        let screen_x = tx * perspective + 0.5;
        let screen_y = ty * perspective + 0.5;
        let depth = tz; // Positive = further away

        (
            screen_x.clamp(0.02, 0.98),
            screen_y.clamp(0.02, 0.98),
            depth,
        )
    }

    /// Get node brightness based on depth (closer = brighter)
    pub fn depth_brightness(&self, depth: f32) -> f32 {
        // Depth ranges roughly from -1 to 1, map to brightness 0.4 to 1.0
        let normalized = ((depth + 1.0) / 2.0).clamp(0.0, 1.0);
        0.4 + (1.0 - normalized) * 0.6
    }

    /// Get node size based on depth (closer = larger)
    pub fn depth_size(&self, depth: f32) -> f32 {
        let normalized = ((depth + 1.0) / 2.0).clamp(0.0, 1.0);
        0.6 + (1.0 - normalized) * 0.4
    }
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
            self.edges
                .iter()
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
            self.selected_node = self
                .selected_node
                .checked_sub(1)
                .unwrap_or(self.nodes.len() - 1);
        }
    }

    /// Get sorted indices by connection count (descending)
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.nodes.len()).collect();
        indices.sort_by(|&a, &b| self.nodes[b].connections.cmp(&self.nodes[a].connections));
        indices
    }

    /// Select next node in sorted order (by connections)
    pub fn select_next_sorted(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let sorted = self.sorted_indices();
        let current_pos = sorted
            .iter()
            .position(|&idx| idx == self.selected_node)
            .unwrap_or(0);
        let next_pos = (current_pos + 1) % sorted.len();
        self.selected_node = sorted[next_pos];
    }

    /// Select previous node in sorted order (by connections)
    pub fn select_prev_sorted(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let sorted = self.sorted_indices();
        let current_pos = sorted
            .iter()
            .position(|&idx| idx == self.selected_node)
            .unwrap_or(0);
        let prev_pos = current_pos
            .checked_sub(1)
            .unwrap_or(sorted.len().saturating_sub(1));
        self.selected_node = sorted[prev_pos];
    }

    /// Apply Fruchterman-Reingold force-directed layout
    /// Makes connected nodes cluster together, unconnected nodes spread out
    pub fn apply_force_layout(&mut self, iterations: usize) {
        if self.nodes.len() < 2 {
            return;
        }

        let n = self.nodes.len() as f32;
        let area = 1.0_f32;
        let k = (area / n).sqrt() * 0.5; // Optimal distance between nodes

        for iteration in 0..iterations {
            // Temperature decreases over iterations (simulated annealing)
            let temp = 0.1 * (1.0 - iteration as f32 / iterations as f32).max(0.01);

            // Calculate displacements
            let mut dx_vec: Vec<f32> = vec![0.0; self.nodes.len()];
            let mut dy_vec: Vec<f32> = vec![0.0; self.nodes.len()];

            // Repulsive forces between all pairs
            for i in 0..self.nodes.len() {
                for j in (i + 1)..self.nodes.len() {
                    let dx = self.nodes[i].x - self.nodes[j].x;
                    let dy = self.nodes[i].y - self.nodes[j].y;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.001);

                    // Repulsive force: k^2 / d
                    let repulsion = (k * k) / dist;
                    let fx = (dx / dist) * repulsion;
                    let fy = (dy / dist) * repulsion;

                    dx_vec[i] += fx;
                    dy_vec[i] += fy;
                    dx_vec[j] -= fx;
                    dy_vec[j] -= fy;
                }
            }

            // Attractive forces along edges (stronger attraction to cluster connected nodes)
            for edge in &self.edges {
                let i_opt = self.nodes.iter().position(|n| n.id == edge.from_id);
                let j_opt = self.nodes.iter().position(|n| n.id == edge.to_id);

                if let (Some(i), Some(j)) = (i_opt, j_opt) {
                    let dx = self.nodes[i].x - self.nodes[j].x;
                    let dy = self.nodes[i].y - self.nodes[j].y;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.001);

                    // Stronger attractive force: d^2 / k * 3, scaled by edge weight
                    let attraction = (dist * dist) / k * edge.weight * 3.0;
                    let fx = (dx / dist) * attraction;
                    let fy = (dy / dist) * attraction;

                    dx_vec[i] -= fx;
                    dy_vec[i] -= fy;
                    dx_vec[j] += fx;
                    dy_vec[j] += fy;
                }
            }

            // Apply displacements with temperature limiting
            for (i, node) in self.nodes.iter_mut().enumerate() {
                let disp_len = (dx_vec[i] * dx_vec[i] + dy_vec[i] * dy_vec[i])
                    .sqrt()
                    .max(0.001);
                let scale = temp.min(disp_len) / disp_len;

                node.x = (node.x + dx_vec[i] * scale).clamp(0.05, 0.95);
                node.y = (node.y + dy_vec[i] * scale).clamp(0.05, 0.95);
            }
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

#[derive(Debug, Clone)]
pub struct DisplayEvent {
    pub event: MemoryEvent,
    pub received_at: Instant,
    pub animation: Animation,
    pub index: usize,
}

impl DisplayEvent {
    pub fn new(event: MemoryEvent, index: usize) -> Self {
        Self {
            event,
            received_at: Instant::now(),
            animation: Animation::entrance(0), // New events always at index 0
            index,
        }
    }

    pub fn is_animating(&self) -> bool {
        // Use time-based check - flash for first 1 second
        self.received_at.elapsed().as_millis() < 1000
    }

    /// Get visual opacity based on animation state
    pub fn visual_opacity(&self) -> f32 {
        self.animation.opacity()
    }

    /// Get horizontal offset for slide animation
    pub fn offset_x(&self) -> i16 {
        self.animation.offset_x()
    }

    /// Check if this is a "new" event (within last 2 seconds)
    pub fn is_new(&self) -> bool {
        self.received_at.elapsed().as_secs() < 2
    }

    /// Get glow intensity for new events
    pub fn glow(&self) -> f32 {
        if self.is_new() {
            let elapsed = self.received_at.elapsed().as_millis() as f32 / 2000.0;
            1.0 - elapsed.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn time_ago(&self) -> String {
        let now = chrono::Utc::now();
        let elapsed = (now - self.event.timestamp).num_seconds();
        if elapsed < 0 {
            // Convert to local time for display
            let local_time = self.event.timestamp.with_timezone(&chrono::Local);
            local_time.format("%H:%M").to_string()
        } else if elapsed < 60 {
            format!("{}s", elapsed)
        } else if elapsed < 3600 {
            format!("{}m", elapsed / 60)
        } else if elapsed < 86400 {
            format!("{}h", elapsed / 3600)
        } else {
            // Convert to local time for display
            let local_time = self.event.timestamp.with_timezone(&chrono::Local);
            local_time.format("%m/%d %H:%M").to_string()
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
        self.context
            + self.learning
            + self.decision
            + self.error
            + self.pattern
            + self.discovery
            + self.task
            + self.conversation
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
    pub theme: Theme,
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
    pub selected_event: Option<usize>,
    /// Error message to display in footer (message, timestamp)
    pub error_message: Option<(String, Instant)>,
    /// Graph rotation angle in radians (for 3D view)
    pub graph_rotation: f32,
    /// Graph tilt angle for 3D perspective
    pub graph_tilt: f32,
    /// Auto-rotate graph flag
    pub graph_auto_rotate: bool,
    /// Search mode active flag
    pub search_active: bool,
    /// Search query input
    pub search_query: String,
    /// Current search mode (keyword/semantic/date)
    pub search_mode: SearchMode,
    /// Search results
    pub search_results: Vec<SearchResult>,
    /// Selected search result index
    pub search_selected: usize,
    /// Whether search results are being displayed
    pub search_results_visible: bool,
    /// Search loading state
    pub search_loading: bool,
    /// UI zoom level (affects content density: 0=compact, 1=normal, 2=expanded)
    pub zoom_level: u8,
    /// Debounce timestamp for search-as-you-type (triggers search after delay)
    pub search_debounce_at: Option<Instant>,
    /// Last query that was searched (to avoid duplicate searches)
    pub search_last_query: String,
    /// Whether viewing detail of selected search result
    pub search_detail_visible: bool,
    /// Smooth scroll state for animated scrolling
    pub smooth_scroll: SmoothScroll,
    /// View transition animation
    pub view_transition: Option<ViewTransition>,
    /// Last tick timestamp for delta time calculations
    pub last_tick: Instant,
    /// Connection animation (pulse when first connected)
    pub connection_animation: Option<Animation>,
    /// Graph display: max nodes to show (0 = all)
    pub graph_max_nodes: usize,
    /// Graph display: minimum edge weight to show (0.0-1.0)
    pub graph_min_edge_weight: f32,
    /// Graph display: show only connected nodes (hide isolated)
    pub graph_hide_isolated: bool,
    /// Graph display: focus mode (show only selected + neighbors)
    pub graph_focus_mode: bool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            view_mode: ViewMode::Dashboard,
            theme: Theme::default(),
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
            max_events: 500,
            session_start: Instant::now(),
            current_user: String::new(),
            selected_event: None,
            error_message: None,
            graph_rotation: 0.0,
            graph_tilt: 0.3,
            graph_auto_rotate: false,
            search_active: false,
            search_query: String::new(),
            search_mode: SearchMode::default(),
            search_results: Vec::new(),
            search_selected: 0,
            search_results_visible: false,
            search_loading: false,
            zoom_level: 1,
            search_debounce_at: None,
            search_last_query: String::new(),
            search_detail_visible: false,
            smooth_scroll: SmoothScroll::default(),
            view_transition: None,
            last_tick: Instant::now(),
            connection_animation: None,
            graph_max_nodes: 50,
            graph_min_edge_weight: 0.3,
            graph_hide_isolated: true,
            graph_focus_mode: false,
        }
    }

    /// Set connected status with animation
    pub fn set_connected(&mut self, connected: bool) {
        if connected && !self.connected {
            // Trigger connection animation
            self.connection_animation = Some(Animation::new(
                AnimationType::Flash {
                    color: Color::Green,
                    duration_ms: 1000, // Longer flash for visibility
                },
                Easing::EaseOut,
            ));
        }
        self.connected = connected;
    }

    /// Toggle auto-rotate for 3D graph
    pub fn toggle_graph_auto_rotate(&mut self) {
        self.graph_auto_rotate = !self.graph_auto_rotate;
    }

    /// Adjust graph tilt angle
    pub fn tilt_graph(&mut self, delta: f32) {
        self.graph_tilt = (self.graph_tilt + delta).clamp(-0.8, 0.8);
    }

    /// Rotate graph left (counter-clockwise)
    pub fn rotate_graph_left(&mut self) {
        self.graph_rotation -= 0.3; // ~17 degrees per press
        if self.graph_rotation < 0.0 {
            self.graph_rotation += std::f32::consts::PI * 2.0;
        }
    }

    /// Rotate graph right (clockwise)
    pub fn rotate_graph_right(&mut self) {
        self.graph_rotation += 0.3; // ~17 degrees per press
        if self.graph_rotation > std::f32::consts::PI * 2.0 {
            self.graph_rotation -= std::f32::consts::PI * 2.0;
        }
    }

    /// Schedule a debounced search (will execute after 250ms)
    /// Note: Date mode doesn't support search-as-you-type (requires Enter)
    pub fn schedule_search(&mut self) {
        // Skip search-as-you-type for date mode - requires full date input
        if self.search_mode == SearchMode::Date {
            return;
        }
        if self.search_query.len() >= 2 {
            self.search_debounce_at = Some(Instant::now() + std::time::Duration::from_millis(250));
        }
    }

    /// Check if debounced search should execute now
    pub fn should_execute_search(&self) -> bool {
        if let Some(debounce_at) = self.search_debounce_at {
            if Instant::now() >= debounce_at
                && self.search_query != self.search_last_query
                && self.search_query.len() >= 2
                && !self.search_loading
            {
                return true;
            }
        }
        false
    }

    /// Mark search as started
    pub fn mark_search_started(&mut self) {
        self.search_last_query = self.search_query.clone();
        self.search_loading = true;
        self.search_debounce_at = None;
    }

    pub fn zoom_in(&mut self) {
        if self.zoom_level < 2 {
            self.zoom_level += 1;
        }
    }

    pub fn zoom_out(&mut self) {
        if self.zoom_level > 0 {
            self.zoom_level -= 1;
        }
    }

    pub fn toggle_theme(&mut self) {
        self.theme = self.theme.toggle();
    }

    pub fn zoom_label(&self) -> &'static str {
        match self.zoom_level {
            0 => "compact",
            1 => "normal",
            _ => "expanded",
        }
    }

    pub fn start_search(&mut self) {
        self.search_active = true;
        self.search_query.clear();
        self.search_results.clear();
        self.search_results_visible = false;
        self.search_loading = false;
    }

    pub fn cancel_search(&mut self) {
        self.search_active = false;
        self.search_query.clear();
        self.search_results.clear();
        self.search_results_visible = false;
        self.search_loading = false;
    }

    pub fn cycle_search_mode(&mut self) {
        self.search_mode = self.search_mode.cycle();
    }

    pub fn set_search_results(&mut self, results: Vec<SearchResult>) {
        self.search_results = results;
        self.search_selected = 0;
        self.search_results_visible = true;
        self.search_loading = false;
    }

    pub fn search_select_next(&mut self) {
        if !self.search_results.is_empty() {
            self.search_selected = (self.search_selected + 1) % self.search_results.len();
        }
    }

    pub fn search_select_prev(&mut self) {
        if !self.search_results.is_empty() {
            self.search_selected = self
                .search_selected
                .checked_sub(1)
                .unwrap_or(self.search_results.len() - 1);
        }
    }

    pub fn selected_search_result(&self) -> Option<&SearchResult> {
        self.search_results.get(self.search_selected)
    }

    pub fn select_event_prev(&mut self) {
        if self.events.is_empty() {
            return;
        }
        match self.selected_event {
            None => self.selected_event = Some(0),
            Some(i) => {
                let new_idx = i.saturating_sub(1);
                self.selected_event = Some(new_idx);
                if new_idx < self.scroll_offset {
                    self.scroll_offset = new_idx;
                }
            }
        }
    }

    pub fn select_event_next(&mut self) {
        if self.events.is_empty() {
            return;
        }
        let max = self.events.len().saturating_sub(1);
        match self.selected_event {
            None => self.selected_event = Some(0),
            Some(i) => {
                let new_idx = (i + 1).min(max);
                self.selected_event = Some(new_idx);
            }
        }
    }

    pub fn clear_event_selection(&mut self) {
        self.selected_event = None;
    }

    pub fn set_view(&mut self, mode: ViewMode) {
        if self.view_mode != mode {
            self.view_transition = Some(ViewTransition::new(self.view_mode, mode));
            self.view_mode = mode;
        }
        self.scroll_offset = 0;
        self.smooth_scroll.current = 0.0;
        self.smooth_scroll.target = 0.0;
        self.smooth_scroll.velocity = 0.0;
    }

    pub fn cycle_view(&mut self) {
        let new_mode = match self.view_mode {
            ViewMode::Dashboard => ViewMode::ActivityLogs,
            ViewMode::ActivityLogs => ViewMode::GraphList,
            ViewMode::GraphList => ViewMode::GraphMap,
            ViewMode::GraphMap => ViewMode::Dashboard,
        };
        self.set_view(new_mode);
    }

    /// Check if view is transitioning
    pub fn is_transitioning(&self) -> bool {
        self.view_transition.as_ref().map(|t| !t.is_complete()).unwrap_or(false)
    }

    /// Get view transition progress (for rendering)
    pub fn transition_progress(&self) -> f32 {
        self.view_transition.as_ref().map(|t| t.progress()).unwrap_or(1.0)
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
                if let Some(ref t) = event.memory_type {
                    self.type_stats.increment(t);
                }
                if let Some(ref entities) = event.entities {
                    self.total_entities += entities.len() as u64;
                    for e in entities {
                        self.add_entity(e.clone());
                    }
                }
                if let Some(ref id) = event.memory_id {
                    let content = event.content_preview.clone().unwrap_or_default();
                    let mem_type = event
                        .memory_type
                        .clone()
                        .unwrap_or_else(|| "Unknown".to_string());
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
                        self.retrieval_stats.avg_latency_ms =
                            (self.retrieval_stats.avg_latency_ms * (total - 1) as f32 + latency)
                                / total as f32;
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
                if let (Some(from), Some(to), Some(weight)) =
                    (&event.from_id, &event.to_id, event.edge_weight)
                {
                    self.add_graph_edge(from.clone(), to.clone(), weight);
                    self.update_graph_stats();
                }
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
        // Update indices for existing events (they're all shifting down)
        for (i, e) in self.events.iter_mut().enumerate() {
            e.index = i + 1;
        }
        self.events.push_front(DisplayEvent::new(event, 0));
        while self.events.len() > self.max_events {
            self.events.pop_back();
        }
    }

    fn add_entity(&mut self, entity: String) {
        if let Some(pos) = self
            .entity_stats
            .top_entities
            .iter()
            .position(|(e, _)| e == &entity)
        {
            self.entity_stats.top_entities[pos].1 += 1;
        } else {
            self.entity_stats.top_entities.push((entity, 1));
        }
        self.entity_stats.top_entities.sort_by(|a, b| b.1.cmp(&a.1));
        self.entity_stats.top_entities.truncate(10);
        self.entity_stats.total += 1;
    }

    fn add_graph_node(&mut self, id: String, content: String, memory_type: String) {
        let short_id = if id.len() > 8 {
            id[..8].to_string()
        } else {
            id.clone()
        };
        let n = self.graph_data.nodes.len() as f32;
        let x = (n * 0.618).sin() * 0.35 + 0.5;
        let y = (n * 0.618).cos() * 0.35 + 0.5;
        let z = ((n * 0.3).sin() * 0.2 + 0.5).clamp(0.1, 0.9);
        self.graph_data.nodes.push(GraphNode {
            id,
            short_id,
            content: if content.len() > 40 {
                format!("{}...", &content[..37])
            } else {
                content
            },
            memory_type,
            connections: 0,
            x,
            y,
            z,
        });
    }

    fn add_graph_edge(&mut self, from_id: String, to_id: String, weight: f32) {
        self.graph_data.edges.push(GraphEdge {
            from_id: from_id.clone(),
            to_id: to_id.clone(),
            weight,
        });
        for node in &mut self.graph_data.nodes {
            if node.id == from_id || node.id == to_id {
                node.connections += 1;
            }
        }
        self.graph_data
            .nodes
            .sort_by(|a, b| b.connections.cmp(&a.connections));
    }

    fn update_graph_stats(&mut self) {
        // Recalculate all stats from actual data to prevent drift
        self.graph_stats.nodes = self.graph_data.nodes.len() as u32;
        self.graph_stats.edges = self.graph_data.edges.len() as u32;

        let n = self.graph_stats.nodes as f32;
        if n > 1.0 {
            self.graph_stats.density = self.graph_stats.edges as f32 / (n * (n - 1.0) / 2.0);
        }

        // Use consistent thresholds: >= 0.7 strong, >= 0.4 medium, < 0.4 weak
        let (s, m, w) = self
            .graph_data
            .edges
            .iter()
            .fold((0, 0, 0), |(s, m, w), e| {
                if e.weight >= 0.7 {
                    (s + 1, m, w)
                } else if e.weight >= 0.4 {
                    (s, m + 1, w)
                } else {
                    (s, m, w + 1)
                }
            });
        self.graph_stats.strong_edges = s;
        self.graph_stats.medium_edges = m;
        self.graph_stats.weak_edges = w;

        if !self.graph_data.edges.is_empty() {
            self.graph_stats.avg_weight =
                self.graph_data.edges.iter().map(|e| e.weight).sum::<f32>()
                    / self.graph_data.edges.len() as f32;
        }
    }

    pub fn scroll_up(&mut self) {
        match self.view_mode {
            ViewMode::GraphList => self.graph_data.select_prev(),
            ViewMode::GraphMap => self.graph_data.select_prev_sorted(),
            _ => {
                let new_offset = self.scroll_offset.saturating_sub(1);
                self.scroll_offset = new_offset;
                self.smooth_scroll.set_target(new_offset);
            }
        }
    }

    pub fn scroll_down(&mut self) {
        match self.view_mode {
            ViewMode::GraphList => self.graph_data.select_next(),
            ViewMode::GraphMap => self.graph_data.select_next_sorted(),
            _ => {
                let max = self.events.len().saturating_sub(1);
                if self.scroll_offset < max {
                    self.scroll_offset += 1;
                    self.smooth_scroll.set_target(self.scroll_offset);
                }
            }
        }
    }

    /// Set an error message to display in footer (auto-clears after 5s)
    pub fn set_error(&mut self, message: String) {
        self.error_message = Some((message, Instant::now()));
    }

    /// Clear error if older than 5 seconds
    pub fn clear_stale_error(&mut self) {
        if let Some((_, timestamp)) = &self.error_message {
            if timestamp.elapsed().as_secs() >= 5 {
                self.error_message = None;
            }
        }
    }

    pub fn tick(&mut self) {
        // Calculate delta time
        let now = Instant::now();
        let dt = (now - self.last_tick).as_secs_f32().min(0.1); // Cap at 100ms
        self.last_tick = now;

        self.animation_tick = self.animation_tick.wrapping_add(1);
        self.clear_stale_error();

        // Update smooth scroll
        self.smooth_scroll.update(dt);

        // Clear completed view transitions
        if let Some(ref transition) = self.view_transition {
            if transition.is_complete() {
                self.view_transition = None;
            }
        }

        // Clear completed connection animation
        if let Some(ref anim) = self.connection_animation {
            if anim.is_complete() {
                self.connection_animation = None;
            }
        }

        // Auto-rotate graph if enabled
        if self.graph_auto_rotate {
            self.graph_rotation += dt * 0.5; // Slow rotation
            if self.graph_rotation > std::f32::consts::PI * 2.0 {
                self.graph_rotation -= std::f32::consts::PI * 2.0;
            }
        }
    }

    /// Get effective scroll offset (from smooth scroll)
    pub fn effective_scroll(&self) -> usize {
        self.smooth_scroll.current_offset()
    }

    /// Get fractional scroll offset for sub-pixel rendering hints
    pub fn scroll_fraction(&self) -> f32 {
        self.smooth_scroll.fractional_offset()
    }
}
