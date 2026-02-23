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
    SlideIn {
        direction: SlideDirection,
        distance: f32,
        duration_ms: u64,
    },
    /// Slide out to a direction with fade
    SlideOut {
        direction: SlideDirection,
        distance: f32,
        duration_ms: u64,
    },
    /// Scale from small to full size
    ScaleIn { from: f32, duration_ms: u64 },
    /// Pulsing glow effect (continuous)
    Pulse {
        min_intensity: f32,
        max_intensity: f32,
        period_ms: u64,
    },
    /// Attention-grabbing highlight flash
    Flash { color: Color, duration_ms: u64 },
    /// Color transition
    ColorTransition {
        from: Color,
        to: Color,
        duration_ms: u64,
    },
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
                duration_ms: 400,              // Snappy 400ms animation
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
        Self::new(AnimationType::FadeIn { duration_ms: 200 }, Easing::EaseOut)
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
            AnimationType::Entrance {
                delay_ms,
                duration_ms,
            } => (*delay_ms, *duration_ms),
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
            AnimationType::Pulse {
                min_intensity,
                max_intensity,
                ..
            } => min_intensity + (max_intensity - min_intensity) * p,
            _ => 1.0,
        }
    }

    /// Get X offset for slide animations
    pub fn offset_x(&self) -> i16 {
        let p = self.easing.apply(self.progress());
        match &self.animation_type {
            AnimationType::SlideIn {
                direction,
                distance,
                ..
            } => {
                let remaining = 1.0 - p;
                match direction {
                    SlideDirection::Left => -(distance * remaining) as i16,
                    SlideDirection::Right => (distance * remaining) as i16,
                    _ => 0,
                }
            }
            AnimationType::SlideOut {
                direction,
                distance,
                ..
            } => match direction {
                SlideDirection::Left => -(distance * p) as i16,
                SlideDirection::Right => (distance * p) as i16,
                _ => 0,
            },
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
            AnimationType::SlideIn {
                direction,
                distance,
                ..
            } => {
                let remaining = 1.0 - p;
                match direction {
                    SlideDirection::Top => -(distance * remaining) as i16,
                    SlideDirection::Bottom => (distance * remaining) as i16,
                    _ => 0,
                }
            }
            AnimationType::SlideOut {
                direction,
                distance,
                ..
            } => match direction {
                SlideDirection::Top => -(distance * p) as i16,
                SlideDirection::Bottom => (distance * p) as i16,
                _ => 0,
            },
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
            AnimationType::Pulse {
                min_intensity,
                max_intensity,
                ..
            } => {
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
    Projects,
    ActivityLogs,
    GraphMap,
}

/// Which panel is focused in Projects view
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FocusPanel {
    #[default]
    Left,
    Right,
    Detail, // Detail panel at bottom (notes/activity)
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

    /// Get sorted indices by name (alphabetically, case-insensitive)
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.nodes.len()).collect();
        indices.sort_by(|&a, &b| {
            self.nodes[a]
                .content
                .to_lowercase()
                .cmp(&self.nodes[b].content.to_lowercase())
        });
        indices
    }

    /// Select next node in sorted order (alphabetically by name)
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

    /// Select previous node in sorted order (alphabetically by name)
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
    /// Full command results for rich display (recall, proactive_context)
    #[serde(default)]
    pub results: Option<serde_json::Value>,
}

impl MemoryEvent {
    pub fn event_color(&self) -> Color {
        match self.event_type.as_str() {
            "CREATE" => Color::Rgb(180, 230, 180),   // Pastel green
            "RETRIEVE" => Color::Rgb(255, 200, 150), // Pastel orange
            "DELETE" => Color::Red,
            "UPDATE" => Color::Yellow,
            "GRAPH_UPDATE" => Color::Magenta,
            "CONSOLIDATE" => Color::Rgb(180, 200, 255), // Pastel blue
            "STRENGTHEN" => Color::Rgb(200, 255, 200),  // Light pastel green
            "DECAY" => Color::Gray,
            "PROMOTE" => Color::LightYellow,
            "HISTORY" => Color::DarkGray,
            "TODO_CREATE" => Color::Rgb(180, 230, 180), // Pastel green
            "TODO_UPDATE" => Color::Yellow,
            "TODO_COMPLETE" => Color::Rgb(200, 255, 200), // Light pastel green
            "TODO_DELETE" => Color::Red,
            "FEEDBACK_PROCESSED" => Color::Rgb(200, 180, 255), // Pastel purple
            "PROACTIVE_CONTEXT" => Color::Rgb(180, 220, 255),  // Light blue
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
            "TODO_CREATE" => "□",
            "TODO_UPDATE" => "◧",
            "TODO_COMPLETE" => "☑",
            "TODO_DELETE" => "☒",
            "FEEDBACK_PROCESSED" => "⟲",
            "PROACTIVE_CONTEXT" => "◉",
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

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY OPERATION TRACKING (for ribbon display)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryOperationType {
    Storing,
    Recalling,
    Consolidating,
    Strengthening,
    Decaying,
    GraphUpdate,
}

impl MemoryOperationType {
    pub fn label(&self) -> &'static str {
        match self {
            MemoryOperationType::Storing => "STORING",
            MemoryOperationType::Recalling => "RECALLING",
            MemoryOperationType::Consolidating => "CONSOLIDATING",
            MemoryOperationType::Strengthening => "STRENGTHENING",
            MemoryOperationType::Decaying => "DECAYING",
            MemoryOperationType::GraphUpdate => "UPDATING",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            MemoryOperationType::Storing => "●",
            MemoryOperationType::Recalling => "◎",
            MemoryOperationType::Consolidating => "⟳",
            MemoryOperationType::Strengthening => "↑",
            MemoryOperationType::Decaying => "↓",
            MemoryOperationType::GraphUpdate => "◆",
        }
    }

    pub fn color(&self) -> Color {
        match self {
            MemoryOperationType::Storing => Color::Rgb(180, 230, 180), // Pastel green
            MemoryOperationType::Recalling => Color::Rgb(255, 200, 150), // Pastel orange
            MemoryOperationType::Consolidating => Color::Rgb(180, 200, 255), // Pastel blue
            MemoryOperationType::Strengthening => Color::Rgb(200, 255, 200), // Light green
            MemoryOperationType::Decaying => Color::Gray,
            MemoryOperationType::GraphUpdate => Color::Magenta,
        }
    }
}

/// Tracks the current/last memory operation for ribbon display
#[derive(Debug, Clone)]
pub struct CurrentOperation {
    pub op_type: MemoryOperationType,
    pub content_preview: String,
    pub memory_type: Option<String>,
    pub timestamp: Instant,
    pub latency_ms: Option<f32>,
    pub count: Option<usize>,
}

impl CurrentOperation {
    pub fn new(op_type: MemoryOperationType, content: &str, memory_type: Option<String>) -> Self {
        Self {
            op_type,
            content_preview: content.to_string(),
            memory_type,
            timestamp: Instant::now(),
            latency_ms: None,
            count: None,
        }
    }

    pub fn with_latency(mut self, ms: f32) -> Self {
        self.latency_ms = Some(ms);
        self
    }

    pub fn with_count(mut self, count: usize) -> Self {
        self.count = Some(count);
        self
    }

    pub fn age_secs(&self) -> u64 {
        self.timestamp.elapsed().as_secs()
    }

    /// Returns true if this operation is "fresh" (within last 10 seconds)
    pub fn is_fresh(&self) -> bool {
        self.timestamp.elapsed().as_secs() < 10
    }
}

/// Tracks the last memory being used/referenced
#[derive(Debug, Clone)]
pub struct LastUsedMemory {
    pub content_preview: String,
    pub memory_type: String,
    pub memory_id: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub accessed_at: Instant,
}

impl LastUsedMemory {
    pub fn new(content: &str, memory_type: &str) -> Self {
        Self {
            content_preview: content.to_string(),
            memory_type: memory_type.to_string(),
            memory_id: None,
            created_at: None,
            accessed_at: Instant::now(),
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.memory_id = Some(id.to_string());
        self
    }

    pub fn with_created_at(mut self, created: DateTime<Utc>) -> Self {
        self.created_at = Some(created);
        self
    }

    pub fn age_display(&self) -> String {
        if let Some(created) = self.created_at {
            let now = chrono::Utc::now();
            let elapsed = (now - created).num_seconds();
            if elapsed < 60 {
                format!("{}s ago", elapsed)
            } else if elapsed < 3600 {
                format!("{}m ago", elapsed / 60)
            } else if elapsed < 86400 {
                format!("{}h ago", elapsed / 3600)
            } else {
                let local_time = created.with_timezone(&chrono::Local);
                local_time.format("%b %d").to_string()
            }
        } else {
            "just now".to_string()
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
            ("Context", self.context, Color::Rgb(255, 200, 150)), // Pastel orange
            ("Learning", self.learning, Color::Rgb(180, 230, 180)), // Pastel green
            ("Decision", self.decision, Color::Yellow),
            ("Discovery", self.discovery, Color::Magenta),
            ("Task", self.task, Color::Rgb(180, 200, 255)), // Pastel blue
            ("Pattern", self.pattern, Color::Rgb(255, 220, 180)), // Light pastel orange
            ("Error", self.error, Color::Red),
            ("Conversation", self.conversation, Color::White),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TODO/PROJECT TYPES (GTD System)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TuiTodoStatus {
    Backlog,
    Todo,
    InProgress,
    Blocked,
    Done,
    Cancelled,
}

impl TuiTodoStatus {
    pub fn icon(&self) -> &'static str {
        match self {
            TuiTodoStatus::Backlog => "◌",
            TuiTodoStatus::Todo => "○",
            TuiTodoStatus::InProgress => "◐",
            TuiTodoStatus::Blocked => "⊘",
            TuiTodoStatus::Done => "●",
            TuiTodoStatus::Cancelled => "⊗",
        }
    }

    pub fn color(&self) -> Color {
        match self {
            TuiTodoStatus::Backlog => Color::DarkGray,
            TuiTodoStatus::Todo => Color::White,
            TuiTodoStatus::InProgress => Color::Rgb(255, 215, 0),
            TuiTodoStatus::Blocked => Color::Red,
            TuiTodoStatus::Done => Color::Rgb(180, 230, 180), // Pastel green
            TuiTodoStatus::Cancelled => Color::DarkGray,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            TuiTodoStatus::Backlog => "backlog",
            TuiTodoStatus::Todo => "todo",
            TuiTodoStatus::InProgress => "in_progress",
            TuiTodoStatus::Blocked => "blocked",
            TuiTodoStatus::Done => "done",
            TuiTodoStatus::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TuiPriority {
    Urgent,
    High,
    Medium,
    Low,
}

impl TuiPriority {
    pub fn indicator(&self) -> &'static str {
        match self {
            TuiPriority::Urgent => "!!!",
            TuiPriority::High => "!! ",
            TuiPriority::Medium => "!  ",
            TuiPriority::Low => "   ",
        }
    }

    pub fn color(&self) -> Color {
        match self {
            TuiPriority::Urgent => Color::Red,
            TuiPriority::High => Color::LightRed,
            TuiPriority::Medium => Color::Yellow,
            TuiPriority::Low => Color::DarkGray,
        }
    }
}

/// Type of todo comment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TuiTodoCommentType {
    #[default]
    Comment,
    Activity,
    Progress,
    Resolution,
}

impl TuiTodoCommentType {
    pub fn icon(&self) -> &'static str {
        match self {
            TuiTodoCommentType::Comment => "▹",
            TuiTodoCommentType::Progress => "▸",
            TuiTodoCommentType::Resolution => "✓",
            TuiTodoCommentType::Activity => "▸",
        }
    }
}

/// A comment/activity on a todo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiTodoComment {
    pub id: String,
    pub author: String,
    pub content: String,
    #[serde(default)]
    pub comment_type: TuiTodoCommentType,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiTodo {
    pub id: String,
    pub content: String,
    pub status: TuiTodoStatus,
    pub priority: TuiPriority,
    pub project_id: Option<String>,
    pub project_name: Option<String>,
    pub contexts: Vec<String>,
    pub due_date: Option<DateTime<Utc>>,
    pub blocked_on: Option<String>,
    pub created_at: DateTime<Utc>,
    /// Parent todo ID (for subtasks)
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Sequence number within project (for display as BOLT-1, MEM-2, etc.)
    #[serde(default)]
    pub seq_num: u32,
    /// Project prefix (cached from project, e.g., "BOLT", "MEM")
    #[serde(default)]
    pub project_prefix: Option<String>,
    /// Comments and activity history
    #[serde(default)]
    pub comments: Vec<TuiTodoComment>,
    /// Additional notes
    #[serde(default)]
    pub notes: Option<String>,
}

impl TuiTodo {
    /// Check if this is a subtask
    pub fn is_subtask(&self) -> bool {
        self.parent_id.is_some()
    }

    pub fn short_id(&self) -> String {
        if self.seq_num > 0 {
            let prefix = self.project_prefix.as_deref().unwrap_or("SHO");
            format!("{}-{}", prefix, self.seq_num)
        } else {
            // Fallback to old style for legacy todos
            format!("SHO-{}", &self.id[..4.min(self.id.len())])
        }
    }

    pub fn is_overdue(&self) -> bool {
        if let Some(due) = self.due_date {
            due < Utc::now() && self.status != TuiTodoStatus::Done
        } else {
            false
        }
    }

    pub fn due_label(&self) -> Option<String> {
        self.due_date.map(|due| {
            let now = Utc::now();
            let diff = due.signed_duration_since(now);
            if diff.num_hours() < 0 {
                let hours = (-diff.num_hours()) as i32;
                if hours < 24 {
                    format!("⚠ {}h ago", hours)
                } else {
                    format!("⚠ {}d ago", hours / 24)
                }
            } else if diff.num_hours() < 24 {
                format!("Due {}", due.format("%I:%M %p"))
            } else if diff.num_days() < 7 {
                format!("Due {}", due.format("%a"))
            } else {
                format!("Due {}", due.format("%b %d"))
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiProject {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub status: String,
    pub todo_count: usize,
    pub completed_count: usize,
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Project prefix for todo IDs (e.g., "BOLT", "MEM")
    #[serde(default)]
    pub prefix: Option<String>,
    /// Number of indexed codebase files
    #[serde(default)]
    pub codebase_file_count: usize,
}

impl TuiProject {
    pub fn progress_percent(&self) -> u8 {
        if self.todo_count == 0 {
            0
        } else {
            ((self.completed_count as f32 / self.todo_count as f32) * 100.0) as u8
        }
    }

    /// Check if this is a sub-project
    pub fn is_subproject(&self) -> bool {
        self.parent_id.is_some()
    }
}

/// File memory summary for TUI display (codebase integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiFileMemory {
    pub id: String,
    pub path: String,
    #[serde(default)]
    pub absolute_path: String,
    pub file_type: String,
    pub summary: String,
    pub key_items: Vec<String>,
    pub access_count: u32,
    pub last_accessed: String,
    pub heat_score: u8,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub line_count: usize,
}

impl TuiFileMemory {
    /// Get icon for file type
    pub fn type_icon(&self) -> &'static str {
        match self.file_type.to_lowercase().as_str() {
            "rust" => "🦀",
            "typescript" | "javascript" => "📜",
            "python" => "🐍",
            "go" => "🐹",
            "java" => "☕",
            "c" | "cpp" | "c++" => "⚙️",
            "markdown" => "📝",
            "json" | "yaml" | "toml" => "📋",
            _ => "📄",
        }
    }

    /// Get heat indicator (🔥 for hot files)
    pub fn heat_indicator(&self) -> &'static str {
        match self.heat_score {
            3 => "🔥",
            2 => "🌡️",
            _ => "",
        }
    }

    /// Format file size in human readable format
    pub fn format_size(&self) -> String {
        let bytes = self.size_bytes;
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        }
    }

    /// Get short path (last 2 components)
    pub fn short_path(&self) -> String {
        let parts: Vec<&str> = self.path.split(['/', '\\']).collect();
        if parts.len() <= 2 {
            self.path.clone()
        } else {
            parts[parts.len() - 2..].join("/")
        }
    }
}

/// Codebase scan result for TUI
#[derive(Debug, Clone, Default)]
pub struct TuiCodebaseScan {
    pub project_id: String,
    pub file_count: usize,
    pub scanning: bool,
    pub indexing: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TodoStats {
    pub total: u32,
    pub backlog: u32,
    pub todo: u32,
    pub in_progress: u32,
    pub blocked: u32,
    pub done: u32,
    pub overdue: u32,
}

pub struct AppState {
    pub events: VecDeque<DisplayEvent>,
    pub view_mode: ViewMode,
    /// GTD Todos
    pub todos: Vec<TuiTodo>,
    /// Projects
    pub projects: Vec<TuiProject>,
    /// Todo statistics
    pub todo_stats: TodoStats,
    /// Selected todo index for navigation
    pub selected_todo: usize,
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
    /// Activity sparkline - event counts per second for last 20 seconds
    pub activity_sparkline: VecDeque<u8>,
    /// Last second for activity tracking
    pub activity_last_second: u64,
    /// Events in current second
    pub activity_current_count: u8,
    /// Last event timestamp for "last activity" display
    pub last_event_time: Option<Instant>,
    /// Expanded projects in Projects view (project IDs)
    pub expanded_projects: std::collections::HashSet<String>,
    /// Expand all sections to show full todo lists (toggle with 'e')
    pub expand_sections: bool,
    /// Selected item index in Projects view (flat list index)
    pub projects_selected: usize,
    /// Scroll offset for Projects view
    pub projects_scroll: usize,
    /// Which panel is focused in Projects view (Left = projects, Right = todos)
    pub focus_panel: FocusPanel,
    /// Selected todo index in right panel for Projects view
    pub todos_selected: usize,
    /// Which panel is focused in Graph Map view (Left = entities, Right = connections)
    pub graph_map_focus: FocusPanel,
    /// Selected connection index in Graph Map view
    pub selected_connection: usize,
    /// Current/last memory operation (for ribbon display)
    pub current_operation: Option<CurrentOperation>,
    /// Last memory being used/referenced (for ribbon display)
    pub last_used_memory: Option<LastUsedMemory>,
    /// Claude Code context sessions (multiple windows supported)
    pub context_sessions: Vec<ContextSession>,
    /// Current lineage trace (for selected todo/memory)
    pub lineage_trace: Option<LineageTrace>,
    /// Lineage horizontal scroll offset
    pub lineage_scroll: usize,
    /// Notes scroll offset (for detail panel)
    pub notes_scroll: usize,
    /// Whether notes section is focused (for keyboard navigation)
    pub notes_focused: bool,
    /// Activity scroll offset (for detail panel)
    pub activity_scroll: usize,
    /// Whether activity section is focused (for keyboard navigation)
    pub activity_focused: bool,
    /// Which detail column is focused (0 = notes/details, 1 = activity)
    pub detail_focus_column: u8,
    /// Project files cache (project_id -> files)
    pub project_files: HashMap<String, Vec<TuiFileMemory>>,
    /// Projects with files section expanded (show codebase files)
    pub files_expanded_projects: std::collections::HashSet<String>,
    /// Current codebase scan state
    pub codebase_scan: Option<TuiCodebaseScan>,
    /// Selected file index in expanded project
    pub selected_file: usize,
    /// Files loading state (project_id being loaded)
    pub files_loading: Option<String>,
    /// Projects that have indexed codebase (for status indicator)
    pub indexed_projects: std::collections::HashSet<String>,
    /// Codebase scan in progress (project_id)
    pub scanning_project: Option<String>,
    /// File popup visible
    pub file_popup_visible: bool,
    /// File popup scroll offset
    pub file_popup_scroll: usize,
    /// Expanded folders in file tree (folder paths)
    pub expanded_folders: std::collections::HashSet<String>,
    /// File preview visible
    pub file_preview_visible: bool,
    /// File preview content (lines)
    pub file_preview_content: Vec<String>,
    /// File preview path
    pub file_preview_path: String,
    /// File preview scroll offset
    pub file_preview_scroll: usize,
    /// File preview key items
    pub file_preview_key_items: Vec<String>,
    /// File preview line count
    pub file_preview_line_count: usize,
    /// File preview file type
    pub file_preview_file_type: String,
    /// Codebase path input active
    pub codebase_input_active: bool,
    /// Codebase path being entered
    pub codebase_input_path: String,
    /// Project ID for codebase input
    pub codebase_input_project_id: Option<String>,
    /// Scroll offset for rich event detail panel (recall/proactive_context results)
    pub event_detail_scroll: usize,
}

/// Claude Code context session status
#[derive(Debug, Clone, Default)]
pub struct ContextSession {
    /// Session ID (unique per Claude window)
    pub session_id: String,
    /// Tokens used
    pub tokens_used: u64,
    /// Token budget (context window size)
    pub tokens_budget: u64,
    /// Usage percentage
    pub percent_used: u8,
    /// Current directory/task
    pub current_task: Option<String>,
    /// Model name
    pub model: Option<String>,
    /// Last update time
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            view_mode: ViewMode::Dashboard,
            todos: Vec::new(),
            projects: Vec::new(),
            todo_stats: TodoStats::default(),
            selected_todo: 0,
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
            activity_sparkline: VecDeque::with_capacity(20),
            activity_last_second: 0,
            activity_current_count: 0,
            last_event_time: None,
            expanded_projects: std::collections::HashSet::new(),
            expand_sections: false,
            projects_selected: 0,
            projects_scroll: 0,
            focus_panel: FocusPanel::default(),
            todos_selected: 0,
            graph_map_focus: FocusPanel::default(),
            selected_connection: 0,
            current_operation: None,
            last_used_memory: None,
            context_sessions: Vec::new(),
            lineage_trace: None,
            lineage_scroll: 0,
            notes_scroll: 0,
            notes_focused: false,
            activity_scroll: 0,
            activity_focused: false,
            detail_focus_column: 0,
            project_files: HashMap::new(),
            files_expanded_projects: std::collections::HashSet::new(),
            codebase_scan: None,
            selected_file: 0,
            files_loading: None,
            indexed_projects: std::collections::HashSet::new(),
            scanning_project: None,
            file_popup_visible: false,
            file_popup_scroll: 0,
            expanded_folders: std::collections::HashSet::new(),
            file_preview_visible: false,
            file_preview_content: Vec::new(),
            file_preview_path: String::new(),
            file_preview_scroll: 0,
            file_preview_key_items: Vec::new(),
            file_preview_line_count: 0,
            file_preview_file_type: String::new(),
            codebase_input_active: false,
            codebase_input_path: String::new(),
            codebase_input_project_id: None,
            event_detail_scroll: 0,
        }
    }

    /// Set connected status with animation
    pub fn set_connected(&mut self, connected: bool) {
        if connected && !self.connected {
            // Trigger connection animation
            self.connection_animation = Some(Animation::new(
                AnimationType::Flash {
                    color: Color::Rgb(180, 230, 180), // Pastel green
                    duration_ms: 1000,                // Longer flash for visibility
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
        self.event_detail_scroll = 0;
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
        self.event_detail_scroll = 0;
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
        self.event_detail_scroll = 0;
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
            ViewMode::Dashboard => ViewMode::Projects,
            ViewMode::Projects => ViewMode::ActivityLogs,
            ViewMode::ActivityLogs => ViewMode::GraphMap,
            ViewMode::GraphMap => ViewMode::Dashboard,
        };
        self.set_view(new_mode);
    }

    /// Check if view is transitioning
    pub fn is_transitioning(&self) -> bool {
        self.view_transition
            .as_ref()
            .map(|t| !t.is_complete())
            .unwrap_or(false)
    }

    /// Get view transition progress (for rendering)
    pub fn transition_progress(&self) -> f32 {
        self.view_transition
            .as_ref()
            .map(|t| t.progress())
            .unwrap_or(1.0)
    }

    pub fn session_duration(&self) -> String {
        let secs = self.session_start.elapsed().as_secs();
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        let secs = secs % 60;
        format!("{:02}:{:02}:{:02}", hours, mins, secs)
    }

    pub fn add_event(&mut self, event: MemoryEvent) {
        // Skip CONTEXT_UPDATE - it's not a memory event, just status ping
        // Context refresh is handled separately in stream.rs
        if event.event_type == "CONTEXT_UPDATE" {
            return;
        }

        // Track current operation for ribbon display
        let content_preview = event.content_preview.clone().unwrap_or_default();
        let mem_type = event.memory_type.clone();

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

                // Update ribbon: STORING operation
                let mut op = CurrentOperation::new(
                    MemoryOperationType::Storing,
                    &content_preview,
                    mem_type.clone(),
                );
                if let Some(latency) = event.latency_ms {
                    op = op.with_latency(latency);
                }
                self.current_operation = Some(op);
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

                // Update ribbon: RECALLING operation
                let mut op = CurrentOperation::new(
                    MemoryOperationType::Recalling,
                    &content_preview,
                    mem_type.clone(),
                );
                if let Some(latency) = event.latency_ms {
                    op = op.with_latency(latency);
                }
                if let Some(count) = event.count {
                    op = op.with_count(count);
                }
                self.current_operation = Some(op);

                // Update last used memory context
                if !content_preview.is_empty() {
                    let mut last_mem = LastUsedMemory::new(
                        &content_preview,
                        mem_type.as_deref().unwrap_or("Unknown"),
                    );
                    if let Some(ref id) = event.memory_id {
                        last_mem = last_mem.with_id(id);
                    }
                    self.last_used_memory = Some(last_mem);
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

                // Update ribbon: GRAPH UPDATE operation
                let op = CurrentOperation::new(
                    MemoryOperationType::GraphUpdate,
                    &format!(
                        "{} → {}",
                        event.from_id.as_deref().unwrap_or("?")
                            [..8.min(event.from_id.as_ref().map(|s| s.len()).unwrap_or(1))]
                            .to_string(),
                        event.to_id.as_deref().unwrap_or("?")
                            [..8.min(event.to_id.as_ref().map(|s| s.len()).unwrap_or(1))]
                            .to_string()
                    ),
                    None,
                );
                self.current_operation = Some(op);
            }
            "STRENGTHEN" => {
                self.consolidation_stats.strengthened += 1;

                // Update ribbon: STRENGTHENING operation
                let op = CurrentOperation::new(
                    MemoryOperationType::Strengthening,
                    &content_preview,
                    mem_type.clone(),
                );
                self.current_operation = Some(op);
            }
            "DECAY" => {
                self.consolidation_stats.decayed += 1;

                // Update ribbon: DECAYING operation
                let op = CurrentOperation::new(
                    MemoryOperationType::Decaying,
                    &content_preview,
                    mem_type.clone(),
                );
                self.current_operation = Some(op);
            }
            "PROMOTE" => {
                self.consolidation_stats.promoted += 1;
                self.tier_stats.working = self.tier_stats.working.saturating_sub(1);
                self.tier_stats.session += 1;
            }
            "CONSOLIDATE" => {
                self.consolidation_stats.cycles += 1;

                // Update ribbon: CONSOLIDATING operation
                let op = CurrentOperation::new(
                    MemoryOperationType::Consolidating,
                    "Memory consolidation cycle",
                    None,
                );
                self.current_operation = Some(op);
            }
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

        // Track activity for sparkline
        self.last_event_time = Some(Instant::now());
        self.activity_current_count = self.activity_current_count.saturating_add(1);
    }

    /// Update activity sparkline - call this every tick
    pub fn update_activity_sparkline(&mut self) {
        let current_second = self.session_start.elapsed().as_secs();
        if current_second > self.activity_last_second {
            // New second - push the count and reset
            self.activity_sparkline
                .push_back(self.activity_current_count);
            while self.activity_sparkline.len() > 20 {
                self.activity_sparkline.pop_front();
            }
            self.activity_current_count = 0;
            self.activity_last_second = current_second;
        }
    }

    /// Get sparkline data for rendering
    pub fn get_sparkline_data(&self) -> Vec<u8> {
        let mut data: Vec<u8> = self.activity_sparkline.iter().copied().collect();
        // Add current partial second
        data.push(self.activity_current_count);
        data
    }

    /// Get time since last event for "heartbeat" display
    pub fn heartbeat_intensity(&self) -> f32 {
        match self.last_event_time {
            Some(t) => {
                let elapsed = t.elapsed().as_secs_f32();
                // Decays from 1.0 to 0.0 over 3 seconds
                (1.0 - elapsed / 3.0).max(0.0)
            }
            None => 0.0,
        }
    }

    /// Get events per minute (based on last 20 seconds, extrapolated)
    pub fn events_per_minute(&self) -> u32 {
        let total: u32 = self.activity_sparkline.iter().map(|&x| x as u32).sum();
        let total = total + self.activity_current_count as u32;
        let seconds = self.activity_sparkline.len() as u32 + 1;
        if seconds > 0 {
            (total * 60) / seconds.max(1)
        } else {
            0
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
            ViewMode::GraphMap => {
                match self.graph_map_focus {
                    FocusPanel::Left => self.graph_data.select_prev_sorted(),
                    FocusPanel::Right => {
                        // Navigate connections
                        self.selected_connection = self.selected_connection.saturating_sub(1);
                    }
                    FocusPanel::Detail => {} // Not used in GraphMap
                }
            }
            _ => {
                let new_offset = self.scroll_offset.saturating_sub(1);
                self.scroll_offset = new_offset;
                self.smooth_scroll.set_target(new_offset);
            }
        }
    }

    pub fn scroll_down(&mut self) {
        match self.view_mode {
            ViewMode::GraphMap => {
                match self.graph_map_focus {
                    FocusPanel::Left => self.graph_data.select_next_sorted(),
                    FocusPanel::Right => {
                        // Navigate connections - need to check max
                        let max_connections = self.graph_connection_count();
                        if max_connections > 0 && self.selected_connection < max_connections - 1 {
                            self.selected_connection += 1;
                        }
                    }
                    FocusPanel::Detail => {} // Not used in GraphMap
                }
            }
            _ => {
                let max = self.events.len().saturating_sub(1);
                if self.scroll_offset < max {
                    self.scroll_offset += 1;
                    self.smooth_scroll.set_target(self.scroll_offset);
                }
            }
        }
    }

    /// Get count of connections for currently selected entity
    pub fn graph_connection_count(&self) -> usize {
        let Some(selected) = self.graph_data.selected() else {
            return 0;
        };
        self.graph_data
            .edges
            .iter()
            .filter(|e| e.from_id == selected.id || e.to_id == selected.id)
            .count()
    }

    /// Toggle focus between entities and connections in Graph Map view
    pub fn toggle_graph_map_focus(&mut self) {
        self.graph_map_focus = match self.graph_map_focus {
            FocusPanel::Left => FocusPanel::Right,
            FocusPanel::Right => FocusPanel::Left,
            FocusPanel::Detail => FocusPanel::Left, // Not used in GraphMap
        };
        // Reset connection selection when switching to connections
        if self.graph_map_focus == FocusPanel::Right {
            self.selected_connection = 0;
        }
    }

    /// Set an error message to display in footer (auto-clears after 5s)
    pub fn set_error(&mut self, message: String) {
        self.error_message = Some((message, Instant::now()));
    }

    /// Clear error immediately
    pub fn clear_error(&mut self) {
        self.error_message = None;
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

        // Update activity sparkline
        self.update_activity_sparkline();
    }

    /// Get effective scroll offset (from smooth scroll)
    pub fn effective_scroll(&self) -> usize {
        self.smooth_scroll.current_offset()
    }

    /// Get fractional scroll offset for sub-pixel rendering hints
    pub fn scroll_fraction(&self) -> f32 {
        self.smooth_scroll.fractional_offset()
    }

    /// Toggle project expansion in Projects view
    pub fn toggle_project_expansion(&mut self, project_id: &str) {
        if self.expanded_projects.contains(project_id) {
            self.expanded_projects.remove(project_id);
        } else {
            self.expanded_projects.insert(project_id.to_string());
        }
    }

    /// Check if a project is expanded
    pub fn is_project_expanded(&self, project_id: &str) -> bool {
        self.expanded_projects.contains(project_id)
    }

    /// Toggle expand_sections mode to show all todos in sections
    pub fn toggle_expand_sections(&mut self) {
        self.expand_sections = !self.expand_sections;
    }

    /// Toggle files display for a project
    pub fn toggle_files_expansion(&mut self, project_id: &str) {
        if self.files_expanded_projects.contains(project_id) {
            self.files_expanded_projects.remove(project_id);
        } else {
            self.files_expanded_projects.insert(project_id.to_string());
        }
    }

    /// Check if files are expanded for a project
    pub fn is_files_expanded(&self, project_id: &str) -> bool {
        self.files_expanded_projects.contains(project_id)
    }

    /// Get files for a project
    pub fn get_project_files(&self, project_id: &str) -> Option<&Vec<TuiFileMemory>> {
        self.project_files.get(project_id)
    }

    /// Set files for a project
    pub fn set_project_files(&mut self, project_id: &str, files: Vec<TuiFileMemory>) {
        self.project_files.insert(project_id.to_string(), files);
        self.files_loading = None;
    }

    /// Check if files are loading for a project
    pub fn is_files_loading(&self, project_id: &str) -> bool {
        self.files_loading.as_ref().map(|id| id == project_id).unwrap_or(false)
    }

    /// Start loading files for a project
    pub fn start_files_loading(&mut self, project_id: &str) {
        self.files_loading = Some(project_id.to_string());
    }

    /// Check if project has indexed codebase
    pub fn is_project_indexed(&self, project_id: &str) -> bool {
        self.indexed_projects.contains(project_id)
    }

    /// Mark project as indexed
    pub fn mark_project_indexed(&mut self, project_id: &str) {
        self.indexed_projects.insert(project_id.to_string());
    }

    /// Check if scanning is in progress for a project
    pub fn is_scanning(&self, project_id: &str) -> bool {
        self.scanning_project.as_ref().map(|id| id == project_id).unwrap_or(false)
    }

    /// Start scanning a project
    pub fn start_scanning(&mut self, project_id: &str) {
        self.scanning_project = Some(project_id.to_string());
    }

    /// Stop scanning
    pub fn stop_scanning(&mut self) {
        self.scanning_project = None;
    }

    /// Toggle file popup
    pub fn toggle_file_popup(&mut self) {
        self.file_popup_visible = !self.file_popup_visible;
        if self.file_popup_visible {
            self.file_popup_scroll = 0;
        }
    }

    /// Toggle folder expansion in file tree
    pub fn toggle_folder(&mut self, folder_path: &str) {
        if self.expanded_folders.contains(folder_path) {
            self.expanded_folders.remove(folder_path);
        } else {
            self.expanded_folders.insert(folder_path.to_string());
        }
    }

    /// Check if folder is expanded
    pub fn is_folder_expanded(&self, folder_path: &str) -> bool {
        self.expanded_folders.contains(folder_path)
    }

    /// Open file preview with content from TuiFileMemory
    pub fn open_file_preview(&mut self, file: &TuiFileMemory, content: Vec<String>) {
        self.file_preview_visible = true;
        self.file_preview_path = file.path.clone();
        self.file_preview_content = content;
        self.file_preview_scroll = 0;
        self.file_preview_key_items = file.key_items.clone();
        self.file_preview_line_count = file.line_count;
        self.file_preview_file_type = file.file_type.clone();
    }

    /// Close file preview
    pub fn close_file_preview(&mut self) {
        self.file_preview_visible = false;
        self.file_preview_content.clear();
    }

    /// Scroll file preview up
    pub fn file_preview_scroll_up(&mut self) {
        if self.file_preview_scroll > 0 {
            self.file_preview_scroll -= 1;
        }
    }

    /// Scroll file preview down
    pub fn file_preview_scroll_down(&mut self, visible_lines: usize) {
        let max_scroll = self.file_preview_content.len().saturating_sub(visible_lines);
        if self.file_preview_scroll < max_scroll {
            self.file_preview_scroll += 1;
        }
    }

    /// Get in-progress todos for NOW bar
    pub fn in_progress_todos(&self) -> Vec<&TuiTodo> {
        self.todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::InProgress)
            .collect()
    }

    /// Get overdue/due today/blocked counts for attention bar
    pub fn attention_counts(&self) -> (usize, usize, usize) {
        let overdue = self.todos.iter().filter(|t| t.is_overdue()).count();
        let due_today = self
            .todos
            .iter()
            .filter(|t| {
                if let Some(due) = &t.due_date {
                    let now = chrono::Utc::now();
                    due.date_naive() == now.date_naive() && !t.is_overdue()
                } else {
                    false
                }
            })
            .count();
        let blocked = self
            .todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::Blocked)
            .count();
        (overdue, due_today, blocked)
    }

    /// Get todos for a specific project
    pub fn todos_for_project(&self, project_id: &str) -> Vec<&TuiTodo> {
        self.todos
            .iter()
            .filter(|t| t.project_id.as_deref() == Some(project_id))
            .collect()
    }

    /// Get standalone todos (no project)
    pub fn standalone_todos(&self) -> Vec<&TuiTodo> {
        self.todos
            .iter()
            .filter(|t| t.project_id.is_none())
            .collect()
    }

    /// Toggle focus between panels (Left -> Right -> Detail -> Left)
    pub fn toggle_focus_panel(&mut self) {
        self.focus_panel = match self.focus_panel {
            FocusPanel::Left => FocusPanel::Right,
            FocusPanel::Right => {
                // Only go to Detail panel in Dashboard mode where it exists
                if matches!(self.view_mode, ViewMode::Dashboard) {
                    FocusPanel::Detail
                } else {
                    FocusPanel::Left
                }
            }
            FocusPanel::Detail => FocusPanel::Left,
        };
        // Reset right panel selection when switching to it
        if self.focus_panel == FocusPanel::Right {
            self.todos_selected = 0;
        }
        // Reset detail panel state when entering it
        if self.focus_panel == FocusPanel::Detail {
            self.notes_scroll = 0;
            self.activity_scroll = 0;
            self.detail_focus_column = 0;
        }
    }

    /// Toggle focus between Notes (0) and Activity (1) in detail panel
    pub fn toggle_detail_column(&mut self) {
        if self.focus_panel == FocusPanel::Detail {
            self.detail_focus_column = if self.detail_focus_column == 0 { 1 } else { 0 };
            // Reset scroll when switching columns
            if self.detail_focus_column == 0 {
                self.notes_scroll = 0;
            } else {
                self.activity_scroll = 0;
            }
        }
    }

    /// Scroll up in the detail panel (notes or activity)
    pub fn detail_scroll_up(&mut self) {
        if self.focus_panel == FocusPanel::Detail {
            if self.detail_focus_column == 0 {
                self.notes_scroll = self.notes_scroll.saturating_sub(1);
            } else {
                self.activity_scroll = self.activity_scroll.saturating_sub(1);
            }
        }
    }

    /// Scroll down in the detail panel (notes or activity)
    pub fn detail_scroll_down(&mut self, max_notes_lines: usize, max_activity_items: usize) {
        if self.focus_panel == FocusPanel::Detail {
            if self.detail_focus_column == 0 {
                if self.notes_scroll < max_notes_lines.saturating_sub(1) {
                    self.notes_scroll += 1;
                }
            } else {
                if self.activity_scroll < max_activity_items.saturating_sub(1) {
                    self.activity_scroll += 1;
                }
            }
        }
    }

    /// Get todos visible in the right panel (for current project selection)
    /// Returns todos in the SAME order as rendered in right panel:
    /// In Progress → Todo → Blocked → Done, with subtasks after each parent
    pub fn visible_todos_right_panel(&self) -> Vec<&TuiTodo> {
        // Build visual order list (same order as sidebar: root projects, then sub-projects under each)
        let mut visual_order: Vec<&TuiProject> = Vec::new();
        let root_projects: Vec<_> = self
            .projects
            .iter()
            .filter(|p| p.parent_id.is_none())
            .collect();
        let sub_projects: Vec<_> = self
            .projects
            .iter()
            .filter(|p| p.parent_id.is_some())
            .collect();
        for project in root_projects.iter() {
            visual_order.push(project);
            for subproject in sub_projects
                .iter()
                .filter(|sp| sp.parent_id.as_ref() == Some(&project.id))
            {
                visual_order.push(subproject);
            }
        }

        let all_todos: Vec<&TuiTodo> = if self.projects_selected < visual_order.len() {
            let project = visual_order[self.projects_selected];
            let mut todos = self.todos_for_project(&project.id);

            // If this is a parent project, also include todos from sub-projects
            if project.parent_id.is_none() {
                for subproject in sub_projects
                    .iter()
                    .filter(|sp| sp.parent_id.as_ref() == Some(&project.id))
                {
                    todos.extend(self.todos_for_project(&subproject.id));
                }
            }
            todos
        } else {
            self.standalone_todos()
        };

        // Now order by status groups with subtasks after parents (same as render order)
        let mut result: Vec<&TuiTodo> = Vec::new();

        // In Progress (non-subtasks first, then their subtasks)
        for todo in all_todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::InProgress && t.parent_id.is_none())
        {
            result.push(todo);
            for subtask in all_todos
                .iter()
                .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
            {
                result.push(subtask);
            }
        }

        // Todo (non-subtasks first, then their subtasks)
        for todo in all_todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::Todo && t.parent_id.is_none())
        {
            result.push(todo);
            for subtask in all_todos
                .iter()
                .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
            {
                result.push(subtask);
            }
        }

        // Blocked (non-subtasks first, then their subtasks)
        for todo in all_todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::Blocked && t.parent_id.is_none())
        {
            result.push(todo);
            for subtask in all_todos
                .iter()
                .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
            {
                result.push(subtask);
            }
        }

        // Done (non-subtasks first, then their subtasks)
        for todo in all_todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::Done && t.parent_id.is_none())
        {
            result.push(todo);
            for subtask in all_todos
                .iter()
                .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
            {
                result.push(subtask);
            }
        }

        // Backlog (non-subtasks first, then their subtasks)
        for todo in all_todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::Backlog && t.parent_id.is_none())
        {
            result.push(todo);
            for subtask in all_todos
                .iter()
                .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
            {
                result.push(subtask);
            }
        }

        result
    }

    /// Get count of todos visible in the right panel
    pub fn visible_todos_count(&self) -> usize {
        self.visible_todos_right_panel().len()
    }

    /// Navigate up in right panel
    pub fn right_panel_up(&mut self) {
        if self.todos_selected > 0 {
            self.todos_selected -= 1;
        }
    }

    /// Navigate down in right panel
    pub fn right_panel_down(&mut self) {
        let max = self.visible_todos_count().saturating_sub(1);
        if self.todos_selected < max {
            self.todos_selected += 1;
        }
    }

    /// Get count of todos for Dashboard navigation (all non-done todos)
    pub fn dashboard_todo_count(&self) -> usize {
        self.todos
            .iter()
            .filter(|t| t.status != TuiTodoStatus::Done && t.status != TuiTodoStatus::Cancelled)
            .count()
    }

    /// Navigate up in Dashboard todos
    pub fn dashboard_todo_up(&mut self) {
        if self.selected_todo > 0 {
            self.selected_todo -= 1;
        }
    }

    /// Navigate down in Dashboard todos
    pub fn dashboard_todo_down(&mut self) {
        let max = self.dashboard_todo_count().saturating_sub(1);
        if self.selected_todo < max {
            self.selected_todo += 1;
        }
    }

    /// Get the currently selected todo in Dashboard or Projects view
    pub fn get_selected_dashboard_todo(&self) -> Option<&TuiTodo> {
        match self.view_mode {
            ViewMode::Dashboard => {
                let active_todos: Vec<_> = self
                    .todos
                    .iter()
                    .filter(|t| {
                        t.status != TuiTodoStatus::Done && t.status != TuiTodoStatus::Cancelled
                    })
                    .collect();
                active_todos.get(self.selected_todo).copied()
            }
            ViewMode::Projects => {
                // In Projects view, get todo from right panel (or Detail panel which uses right panel's selection)
                if self.focus_panel == FocusPanel::Right || self.focus_panel == FocusPanel::Detail {
                    // Right panel shows project todos - also used when in Detail panel
                    let todos = self.visible_todos_right_panel();
                    todos.get(self.todos_selected).copied()
                } else {
                    // Left panel - check if a todo is selected within expanded project
                    self.get_left_panel_selected_todo()
                }
            }
            _ => None,
        }
    }

    /// Get selected todo from the left panel in Projects view
    fn get_left_panel_selected_todo(&self) -> Option<&TuiTodo> {
        let mut flat_idx = 0;
        for project in &self.projects {
            if self.projects_selected == flat_idx {
                return None; // Project header selected, not a todo
            }
            flat_idx += 1;

            if self.is_project_expanded(&project.id) {
                let todos = self.todos_for_project(&project.id);
                for todo in todos.iter().take(5) {
                    if self.projects_selected == flat_idx {
                        return Some(todo);
                    }
                    flat_idx += 1;
                }
            }
        }
        // Check inbox todos
        for todo in self.standalone_todos().iter().take(5) {
            if self.projects_selected == flat_idx {
                return Some(todo);
            }
            flat_idx += 1;
        }
        None
    }

    /// Get flat item count for left panel (projects + expanded todos + inbox todos)
    pub fn left_panel_flat_count(&self) -> usize {
        let mut count = 0;
        for project in &self.projects {
            count += 1; // The project itself
            if self.is_project_expanded(&project.id) {
                // Add up to 5 expanded todos
                let todos = self.todos_for_project(&project.id);
                count += todos.len().min(5);
            }
        }
        // Add inbox todos (up to 5)
        count += self.standalone_todos().len().min(5);
        count
    }

    /// Check if the current left panel selection is on a project (not a todo)
    pub fn is_project_selected(&self) -> bool {
        let mut flat_idx = 0;
        for project in &self.projects {
            if self.projects_selected == flat_idx {
                return true;
            }
            flat_idx += 1;
            if self.is_project_expanded(&project.id) {
                let todos = self.todos_for_project(&project.id);
                flat_idx += todos.len().min(5);
            }
        }
        false // Must be an inbox todo
    }

    /// Get the project ID if the current selection is on a project
    /// Uses visual order: root projects first, then sub-projects under each parent
    pub fn selected_project_id(&self) -> Option<String> {
        let mut flat_idx = 0;

        // Build visual order: root projects, then sub-projects under each
        let root_projects: Vec<_> = self.projects.iter().filter(|p| p.parent_id.is_none()).collect();
        let sub_projects: Vec<_> = self.projects.iter().filter(|p| p.parent_id.is_some()).collect();

        for project in &root_projects {
            if self.projects_selected == flat_idx {
                return Some(project.id.clone());
            }
            flat_idx += 1;
            if self.is_project_expanded(&project.id) {
                let todos = self.todos_for_project(&project.id);
                flat_idx += todos.len().min(5);
            }

            // Check sub-projects of this root project
            for subproject in sub_projects.iter().filter(|sp| sp.parent_id.as_ref() == Some(&project.id)) {
                if self.projects_selected == flat_idx {
                    return Some(subproject.id.clone());
                }
                flat_idx += 1;
                if self.is_project_expanded(&subproject.id) {
                    let todos = self.todos_for_project(&subproject.id);
                    flat_idx += todos.len().min(5);
                }
            }
        }
        None
    }

    /// Set lineage trace data
    pub fn set_lineage_trace(&mut self, trace: LineageTrace) {
        self.lineage_trace = Some(trace);
        self.lineage_scroll = 0;
    }

    /// Clear lineage trace
    pub fn clear_lineage_trace(&mut self) {
        self.lineage_trace = None;
        self.lineage_scroll = 0;
    }

    /// Scroll lineage left
    pub fn lineage_scroll_left(&mut self) {
        if self.lineage_scroll > 0 {
            self.lineage_scroll -= 1;
        }
    }

    /// Scroll lineage right
    pub fn lineage_scroll_right(&mut self) {
        if let Some(ref trace) = self.lineage_trace {
            let max_scroll = trace.edges.len().saturating_sub(3);
            if self.lineage_scroll < max_scroll {
                self.lineage_scroll += 1;
            }
        }
    }
}

// =============================================================================
// LINEAGE TYPES (Decision Lineage Graph)
// =============================================================================

/// Lineage edge representing a causal relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEdge {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub relation: String,
    pub confidence: f32,
    pub source: String,
}

impl LineageEdge {
    /// Get display icon for relation type
    pub fn relation_icon(&self) -> &'static str {
        match self.relation.as_str() {
            "Caused" => "→",
            "ResolvedBy" => "✓",
            "InformedBy" => "◈",
            "SupersededBy" => "⇢",
            "TriggeredBy" => "⚡",
            "BranchedFrom" => "⑂",
            "RelatedTo" => "~",
            _ => "→",
        }
    }

    /// Get color for relation type
    pub fn relation_color(&self) -> Color {
        match self.relation.as_str() {
            "Caused" => Color::Rgb(255, 100, 100),       // Bright red
            "ResolvedBy" => Color::Rgb(100, 255, 150),   // Bright green
            "InformedBy" => Color::Rgb(100, 180, 255),   // Bright blue
            "SupersededBy" => Color::Rgb(180, 180, 180), // Light gray
            "TriggeredBy" => Color::Rgb(255, 220, 80),   // Bright yellow
            "BranchedFrom" => Color::Rgb(255, 130, 255), // Bright magenta
            "RelatedTo" => Color::Rgb(150, 150, 150),    // Medium gray
            _ => Color::White,
        }
    }

    /// Get source indicator
    pub fn source_indicator(&self) -> &'static str {
        match self.source.as_str() {
            "Confirmed" => "✓",
            "Explicit" => "⚡",
            "Inferred" => "?",
            _ => "·",
        }
    }
}

/// Lineage node info (minimal info for chain display)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    pub id: String,
    pub short_id: String,
    pub content_preview: String,
    pub memory_type: String,
}

impl LineageNode {
    /// Get icon for memory type
    pub fn type_icon(&self) -> &'static str {
        match self.memory_type.to_lowercase().as_str() {
            "error" => "⚠",
            "task" => "□",
            "learning" => "◆",
            "decision" => "◇",
            "discovery" => "★",
            "pattern" => "◎",
            "context" => "●",
            "conversation" => "○",
            _ => "•",
        }
    }

    /// Get color for memory type
    pub fn type_color(&self) -> Color {
        match self.memory_type.to_lowercase().as_str() {
            "error" => Color::Rgb(255, 80, 80),          // Bright red
            "task" => Color::Rgb(100, 180, 255),         // Bright blue
            "learning" => Color::Rgb(80, 255, 130),      // Bright green
            "decision" => Color::Rgb(255, 230, 80),      // Bright yellow
            "discovery" => Color::Rgb(255, 120, 255),    // Bright magenta
            "pattern" => Color::Rgb(255, 180, 80),       // Bright orange
            "context" => Color::Rgb(255, 160, 100),      // Bright coral
            "conversation" => Color::Rgb(220, 220, 220), // Bright white
            _ => Color::Rgb(180, 180, 180),              // Light gray
        }
    }
}

/// Complete lineage trace result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineageTrace {
    pub root_id: String,
    pub direction: String,
    pub edges: Vec<LineageEdge>,
    pub nodes: HashMap<String, LineageNode>,
    pub path: Vec<String>,
    pub depth: usize,
}
