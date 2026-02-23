use crate::logo::{ELEPHANT, ELEPHANT_GRADIENT, SHODH_GRADIENT, SHODH_TEXT};
use crate::types::{
    AppState, DisplayEvent, FocusPanel, LineageEdge, LineageNode, LineageTrace, SearchMode,
    SearchResult, TuiFileMemory, TuiPriority, TuiProject, TuiTodo, TuiTodoComment,
    TuiTodoCommentType, TuiTodoStatus, ViewMode, VERSION,
};
use ratatui::{prelude::*, widgets::*};

// ============================================================================
// SHANTI THEME - Hindu-inspired minimal aesthetic with pastel accents
// ============================================================================

/// Pastel Saffron - Active/In-Progress (soft, warm)
const SAFFRON: Color = Color::Rgb(255, 183, 130);
/// Pastel Gold - Success/Completed (soft golden)
const GOLD: Color = Color::Rgb(255, 214, 130);
/// Pastel Turmeric - Highlights/Due today
const TURMERIC: Color = Color::Rgb(255, 200, 120);
/// Pastel Rose - Blocked/Overdue (soft attention)
const MAROON: Color = Color::Rgb(255, 140, 140);
/// Deep Blue - Links/Connections
const DEEP_BLUE: Color = Color::Rgb(130, 160, 220);
/// Primary text - almost white
const TEXT_PRIMARY: Color = Color::Rgb(240, 240, 240);
/// Secondary text - muted grey
const TEXT_SECONDARY: Color = Color::Rgb(160, 160, 160);
/// Disabled/inactive text
const TEXT_DISABLED: Color = Color::Rgb(100, 100, 100);
/// Subtle borders
const BORDER_SUBTLE: Color = Color::Rgb(50, 50, 50);
/// Section dividers
const BORDER_DIVIDER: Color = Color::Rgb(60, 60, 60);
/// Surface/card background
const SURFACE: Color = Color::Rgb(35, 35, 35);
/// Live/connected indicator
const LIVE_GREEN: Color = Color::Rgb(150, 230, 170);
/// Selection background - pale yellow/cream
const SELECTION_BG: Color = Color::Rgb(75, 70, 50);

// ============================================================================
// CONNECTION STRENGTH COLORS (unified across all panels)
// ============================================================================
/// Strong connection (weight >= 0.7) - bright green
const CONN_STRONG: Color = Color::Rgb(120, 200, 120);
/// Medium connection (weight >= 0.4) - golden yellow
const CONN_MEDIUM: Color = Color::Rgb(230, 200, 100);
/// Weak connection (weight < 0.4) - muted gray
const CONN_WEAK: Color = Color::Rgb(120, 120, 120);

/// Apply opacity to a color (blend with black)
fn apply_opacity(color: Color, opacity: f32) -> Color {
    let opacity = opacity.clamp(0.0, 1.0);
    match color {
        Color::Rgb(r, g, b) => Color::Rgb(
            (r as f32 * opacity) as u8,
            (g as f32 * opacity) as u8,
            (b as f32 * opacity) as u8,
        ),
        Color::Green => Color::Rgb(0, (128.0 * opacity) as u8, 0),
        Color::Cyan => Color::Rgb(0, (255.0 * opacity) as u8, (255.0 * opacity) as u8),
        Color::Yellow => Color::Rgb((255.0 * opacity) as u8, (255.0 * opacity) as u8, 0),
        Color::Red => Color::Rgb((255.0 * opacity) as u8, 0, 0),
        Color::Magenta => Color::Rgb((255.0 * opacity) as u8, 0, (255.0 * opacity) as u8),
        Color::Blue => Color::Rgb(0, 0, (255.0 * opacity) as u8),
        Color::White => Color::Rgb(
            (255.0 * opacity) as u8,
            (255.0 * opacity) as u8,
            (255.0 * opacity) as u8,
        ),
        Color::Gray => Color::Rgb(
            (128.0 * opacity) as u8,
            (128.0 * opacity) as u8,
            (128.0 * opacity) as u8,
        ),
        Color::DarkGray => Color::Rgb(
            (64.0 * opacity) as u8,
            (64.0 * opacity) as u8,
            (64.0 * opacity) as u8,
        ),
        _ => color,
    }
}

/// Interpolate between two colors
fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (r1, g1, b1) = color_to_rgb(from);
    let (r2, g2, b2) = color_to_rgb(to);
    Color::Rgb(
        (r1 as f32 + (r2 as f32 - r1 as f32) * t) as u8,
        (g1 as f32 + (g2 as f32 - g1 as f32) * t) as u8,
        (b1 as f32 + (b2 as f32 - b1 as f32) * t) as u8,
    )
}

fn color_to_rgb(color: Color) -> (u8, u8, u8) {
    match color {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::Green => (0, 128, 0),
        Color::Cyan => (0, 255, 255),
        Color::Yellow => (255, 255, 0),
        Color::Red => (255, 0, 0),
        Color::Magenta => (255, 0, 255),
        Color::Blue => (0, 0, 255),
        Color::White => (255, 255, 255),
        Color::Gray => (128, 128, 128),
        Color::DarkGray => (64, 64, 64),
        Color::Black => (0, 0, 0),
        Color::LightGreen => (144, 238, 144),
        Color::LightBlue => (173, 216, 230),
        Color::LightCyan => (224, 255, 255),
        Color::LightYellow => (255, 255, 224),
        _ => (128, 128, 128),
    }
}

/// Get glow color with intensity
fn glow_color(base_color: Color, intensity: f32) -> Color {
    let (r, g, b) = color_to_rgb(base_color);
    let boost = (intensity * 127.0) as u8;
    Color::Rgb(
        r.saturating_add(boost),
        g.saturating_add(boost),
        b.saturating_add(boost),
    )
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        format!(
            "{}...",
            s.chars()
                .take(max_len.saturating_sub(3))
                .collect::<String>()
        )
    }
}

fn progress_bar(value: u32, max: u32, width: usize) -> String {
    if max == 0 {
        return " ".repeat(width);
    }
    let filled = ((value as f32 / max as f32) * width as f32) as usize;
    format!(
        "{}{}",
        "█".repeat(filled.min(width)),
        "░".repeat(width.saturating_sub(filled))
    )
}

pub fn render_header(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(22),
            Constraint::Min(20),
            Constraint::Length(45), // Wider for context tracker
        ])
        .split(inner);

    // Elephant logo with breathing animation when connected
    let logo_lines: Vec<Line> = ELEPHANT
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let (r, g, b) = if state.connected {
                let (base_r, base_g, base_b) = ELEPHANT_GRADIENT[i % ELEPHANT_GRADIENT.len()];

                // Breathing effect: subtle brightness oscillation
                let breath_phase = (state.animation_tick as f32 * 0.05 + i as f32 * 0.1).sin();
                let breath_intensity = 0.85 + breath_phase * 0.15; // 0.7 to 1.0

                // Activity boost: brighter when events are happening
                let activity_boost = state.heartbeat_intensity() * 0.2;
                let intensity = (breath_intensity + activity_boost).min(1.2);

                (
                    (base_r as f32 * intensity).min(255.0) as u8,
                    (base_g as f32 * intensity).min(255.0) as u8,
                    (base_b as f32 * intensity).min(255.0) as u8,
                )
            } else {
                // Disconnected: dim gray with slow pulse
                let pulse = (state.animation_tick as f32 * 0.03).sin() * 0.2 + 0.5;
                let gray = (60.0 + pulse * 40.0) as u8;
                (gray, gray, gray)
            };
            Line::from(Span::styled(*l, Style::default().fg(Color::Rgb(r, g, b))))
        })
        .collect();
    f.render_widget(Paragraph::new(logo_lines), chunks[0]);

    // Title and stats
    let title_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(chunks[1]);

    let shodh_lines: Vec<Line> = SHODH_TEXT
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let (r, g, b) = SHODH_GRADIENT[i % SHODH_GRADIENT.len()];
            Line::from(Span::styled(
                *l,
                Style::default()
                    .fg(Color::Rgb(r, g, b))
                    .add_modifier(Modifier::BOLD),
            ))
        })
        .collect();
    f.render_widget(Paragraph::new(shodh_lines), title_chunks[0]);

    // Stats bar
    let stats_line = Line::from(vec![
        Span::styled(
            format!("{} ", state.total_memories),
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("memories ", Style::default().fg(Color::DarkGray)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{} ", state.total_edges),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("edges ", Style::default().fg(Color::DarkGray)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{} ", state.total_recalls),
            Style::default()
                .fg(Color::Rgb(180, 230, 180))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("recalls", Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(Paragraph::new(stats_line), title_chunks[1]);

    // Right side: version, status with heartbeat, sparkline, context, session
    // Dynamic context height based on active sessions (min 1, max 4)
    let context_height = state.context_sessions.len().max(1).min(4) as u16;
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),           // Version
            Constraint::Length(2),           // Status with heartbeat
            Constraint::Length(2),           // Sparkline/activity
            Constraint::Min(context_height), // Context window status (multiple sessions)
            Constraint::Length(1),           // Session
        ])
        .split(chunks[2]);

    // Current date and time with version
    let now = chrono::Local::now();
    let time_str = now.format("%b %d, %Y  %I:%M %p").to_string();
    let version_line = Line::from(vec![
        Span::styled(time_str, Style::default().fg(TEXT_SECONDARY)),
        Span::styled("  │  ", Style::default().fg(BORDER_DIVIDER)),
        Span::styled(
            format!("v{}", VERSION),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    f.render_widget(
        Paragraph::new(version_line).alignment(Alignment::Right),
        right_chunks[0],
    );

    // Pulsing LIVE indicator with heartbeat effect
    let status = if state.connected {
        let heartbeat = state.heartbeat_intensity();
        let pulse_phase = (state.animation_tick as f32 * 0.15).sin() * 0.5 + 0.5;

        // Combine heartbeat (event-triggered) with ambient pulse
        let intensity = if heartbeat > 0.1 {
            heartbeat // Use heartbeat when events are happening
        } else {
            pulse_phase * 0.3 + 0.7 // Subtle ambient pulse when idle
        };

        // Color intensity based on activity
        let green_val = (100.0 + intensity * 155.0) as u8;
        let dot_color = Color::Rgb(0, green_val, 0);

        // Choose dot character based on heartbeat
        let dot = if heartbeat > 0.5 {
            "◉"
        } else if heartbeat > 0.1 {
            "●"
        } else {
            "○"
        };

        Line::from(vec![
            Span::styled(
                format!("{} ", dot),
                Style::default().fg(dot_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "LIVE",
                Style::default()
                    .fg(Color::Rgb(0, green_val, 0))
                    .add_modifier(Modifier::BOLD),
            ),
        ])
    } else {
        // Pulsing reconnect indicator
        let pulse = (state.animation_tick as f32 * 0.1).sin() * 0.5 + 0.5;
        let yellow_val = (150.0 + pulse * 105.0) as u8;
        Line::from(Span::styled(
            "○ ...",
            Style::default().fg(Color::Rgb(yellow_val, yellow_val, 0)),
        ))
    };
    f.render_widget(
        Paragraph::new(status).alignment(Alignment::Right),
        right_chunks[1],
    );

    // Activity counter - events per minute
    let events_per_min = state.events_per_minute();
    let counter_color = if events_per_min > 10 {
        Color::Rgb(180, 230, 180) // High activity
    } else if events_per_min > 0 {
        Color::Rgb(100, 180, 220) // Some activity
    } else {
        Color::DarkGray // Idle
    };
    let arrow = if events_per_min > 0 { "↑" } else { "·" };
    let counter_line = Line::from(vec![
        Span::styled(format!("{} ", arrow), Style::default().fg(counter_color)),
        Span::styled(
            format!("{}", events_per_min),
            Style::default()
                .fg(counter_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" events/min", Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(
        Paragraph::new(counter_line).alignment(Alignment::Right),
        right_chunks[2],
    );

    // Context window status from Claude Code (supports multiple sessions)
    // Pastel colors for different sessions
    const SESSION_COLORS: [Color; 4] = [
        Color::Rgb(255, 220, 130), // Pastel yellow
        Color::Rgb(255, 160, 140), // Pastel red/coral
        Color::Rgb(160, 220, 255), // Pastel blue
        Color::Rgb(180, 255, 180), // Pastel green
    ];
    let context_lines: Vec<Line> = if state.context_sessions.is_empty() {
        vec![Line::from(Span::styled(
            "⬡ --",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        state
            .context_sessions
            .iter()
            .enumerate()
            .map(|(idx, session)| {
                let percent = session.percent_used;
                let color = if percent < 50 {
                    Color::Rgb(100, 200, 100) // Green
                } else if percent < 80 {
                    Color::Rgb(220, 180, 80) // Yellow
                } else {
                    Color::Rgb(220, 100, 100) // Red
                };
                let tokens_k = session.tokens_used / 1000;
                let budget_k = session.tokens_budget / 1000;
                let model = session.model.as_deref().unwrap_or("Claude");
                // Extract directory name from full path
                let dir_name = session
                    .current_task
                    .as_ref()
                    .and_then(|p| p.split(['/', '\\']).last())
                    .unwrap_or("");
                // Use different pastel color for each session's directory name
                let session_color = SESSION_COLORS[idx % SESSION_COLORS.len()];
                Line::from(vec![
                    Span::styled(
                        format!("{}%", percent),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" {}k/{}k", tokens_k, budget_k),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(model, Style::default().fg(Color::Rgb(150, 180, 220))),
                    Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(dir_name, Style::default().fg(session_color)),
                ])
            })
            .collect()
    };
    f.render_widget(
        Paragraph::new(context_lines).alignment(Alignment::Right),
        right_chunks[3],
    );

    let session = Line::from(Span::styled(
        state.session_duration(),
        Style::default().fg(Color::White),
    ));
    f.render_widget(
        Paragraph::new(session).alignment(Alignment::Right),
        right_chunks[4],
    );
}

/// Render a mini sparkline using Unicode block characters
fn render_sparkline(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }

    // Find max for scaling
    let max_val = *data.iter().max().unwrap_or(&1) as f32;
    let max_val = max_val.max(1.0); // Avoid division by zero

    // Unicode bar characters for 8 levels
    const BARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    data.iter()
        .map(|&v| {
            let normalized = (v as f32 / max_val * 8.0) as usize;
            BARS[normalized.min(8)]
        })
        .collect()
}

pub fn render_main(f: &mut Frame, area: Rect, state: &AppState) {
    // If viewing search result detail, render detail view
    if state.search_detail_visible {
        render_search_detail(f, area, state);
        return;
    }

    // If search results are visible, render them instead of normal view
    if state.search_results_visible {
        render_search_results(f, area, state);
        return;
    }

    // If search is loading, show loading indicator overlay
    if state.search_loading {
        render_search_loading(f, area, state);
        return;
    }

    // Render the view first, then overlay transition effect on top
    match state.view_mode {
        ViewMode::Dashboard => render_dashboard(f, area, state),
        ViewMode::Projects => render_projects_view(f, area, state),
        ViewMode::ActivityLogs => render_activity_logs(f, area, state),
        ViewMode::GraphMap => render_graph_map(f, area, state),
    }

    // View transition overlay - renders on top of content
    if state.is_transitioning() {
        render_view_transition(f, area, state);
    }
}

/// Render view transition overlay effect
fn render_view_transition(f: &mut Frame, area: Rect, state: &AppState) {
    if let Some(ref transition) = state.view_transition {
        let progress = transition.progress();

        // Simple border flash during transition - works on both themes
        // Use orange/accent color which is visible on both dark and light backgrounds
        let accent = state.theme.accent();

        if progress < 0.5 {
            // First half: orange border flash out
            let fade_intensity = progress * 2.0; // 0 to 1 during first half
            let border_intensity = ((1.0 - fade_intensity) * 200.0 + 55.0) as u8;
            let border_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(255, border_intensity, 0)));
            f.render_widget(border_block, area);
        } else {
            // Second half: accent border fades in
            let fade_intensity = (progress - 0.5) * 2.0; // 0 to 1 during second half

            if fade_intensity < 0.95 {
                // Border that fades to theme accent
                let intensity = (fade_intensity * 255.0) as u8;
                let border_block = Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Rgb(255, 140 + (intensity / 4), 50)));
                f.render_widget(border_block, area);
            }
        }
        let _ = accent; // Suppress unused warning
    }
}

fn render_search_loading(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            " SEARCHING... ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let idx = (state.animation_tick as usize / 2) % spinner.len();
    let loading_text = vec![
        Line::from(""),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                spinner[idx],
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" Searching for \"{}\"...", state.search_query),
                Style::default().fg(Color::White),
            ),
        ]),
    ];
    f.render_widget(
        Paragraph::new(loading_text).alignment(Alignment::Center),
        inner,
    );
}

fn render_search_results(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(180, 230, 180)))
        .title(Span::styled(
            format!(
                " SEARCH RESULTS: \"{}\" ({}) ",
                truncate(&state.search_query, 30),
                state.search_results.len()
            ),
            Style::default()
                .fg(Color::Rgb(180, 230, 180))
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" [{}] ", state.search_mode.label()),
                Style::default().fg(Color::Rgb(255, 200, 150)),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.search_results.is_empty() {
        let no_results = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No results found.",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Press / to search again or Esc to close.",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        f.render_widget(Paragraph::new(no_results), inner);
        return;
    }

    // Calculate lines per result based on zoom level
    let lines_per_result = match state.zoom_level {
        0 => 2u16, // compact: 2 lines
        1 => 4u16, // normal: 4 lines
        _ => 6u16, // expanded: 6 lines
    };
    let max_visible = (inner.height / lines_per_result).max(1) as usize;

    // Calculate scroll to keep selected in view
    let scroll_start = if state.search_selected >= max_visible {
        state.search_selected - max_visible + 1
    } else {
        0
    };

    let mut y_offset = 0u16;
    for (idx, result) in state
        .search_results
        .iter()
        .enumerate()
        .skip(scroll_start)
        .take(max_visible)
    {
        let result_area = Rect {
            x: inner.x,
            y: inner.y + y_offset,
            width: inner.width,
            height: lines_per_result,
        };
        if result_area.y + result_area.height <= inner.y + inner.height {
            let is_selected = idx == state.search_selected;
            render_search_result_item(f, result_area, state, result, is_selected, state.zoom_level);
        }
        y_offset += lines_per_result;
    }

    // Scrollbar
    if state.search_results.len() > max_visible {
        let scrollbar = Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state =
            ScrollbarState::new(state.search_results.len()).position(state.search_selected);
        f.render_stateful_widget(
            scrollbar,
            area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut scrollbar_state,
        );
    }
}

fn render_search_result_item(
    f: &mut Frame,
    area: Rect,
    state: &AppState,
    result: &SearchResult,
    is_selected: bool,
    zoom_level: u8,
) {
    let bg = if is_selected {
        state.theme.selection_bg()
    } else {
        state.theme.bg()
    };
    let content_width = area.width.saturating_sub(4) as usize;

    // Background for selected
    if is_selected {
        f.render_widget(Block::default().style(Style::default().bg(bg)), area);
    }

    let mut lines = Vec::new();

    // Line 1: Type + Score + ID + Time
    let short_id = if result.id.len() > 8 {
        &result.id[..8]
    } else {
        &result.id
    };
    let time_ago = {
        let now = chrono::Utc::now();
        let elapsed = (now - result.created_at).num_seconds();
        if elapsed < 60 {
            format!("{}s", elapsed)
        } else if elapsed < 3600 {
            format!("{}m", elapsed / 60)
        } else if elapsed < 86400 {
            format!("{}h", elapsed / 3600)
        } else {
            let local_time = result.created_at.with_timezone(&chrono::Local);
            local_time.format("%m/%d").to_string()
        }
    };

    let prefix = if is_selected { "▶ " } else { "  " };
    lines.push(Line::from(vec![
        Span::styled(
            prefix,
            Style::default()
                .fg(if is_selected {
                    Color::Rgb(255, 200, 150)
                } else {
                    Color::DarkGray
                })
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        ),
        Span::styled(
            format!("[{}]", result.memory_type),
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        ),
        Span::styled(
            format!(" {:.2} ", result.score),
            Style::default().fg(Color::Yellow).bg(bg),
        ),
        Span::styled(short_id, Style::default().fg(Color::DarkGray).bg(bg)),
        Span::styled(
            format!(" {}", time_ago),
            Style::default().fg(Color::DarkGray).bg(bg),
        ),
    ]));

    // Line 2: Content (truncated based on zoom)
    let content_len = match zoom_level {
        0 => content_width.saturating_sub(4),
        1 => content_width * 2,
        _ => content_width * 4,
    };
    let content_preview = truncate(&result.content, content_len);

    // Content color: use theme-aware colors for readability on both dark and light backgrounds
    let content_fg = if is_selected {
        state.theme.fg()
    } else {
        state.theme.fg_dim()
    };

    if zoom_level == 0 {
        // Compact: single line content
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default().bg(bg)),
            Span::styled(content_preview, Style::default().fg(content_fg).bg(bg)),
        ]));
    } else {
        // Normal/Expanded: word-wrap content
        let mut remaining = content_preview.as_str();
        let line_width = content_width.saturating_sub(2);
        let max_lines = if zoom_level == 1 { 2 } else { 4 };
        let mut line_count = 0;
        while !remaining.is_empty() && line_count < max_lines {
            let take = remaining.chars().take(line_width).collect::<String>();
            let actual_len = take.len();
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default().bg(bg)),
                Span::styled(take, Style::default().fg(content_fg).bg(bg)),
            ]));
            remaining = &remaining[actual_len.min(remaining.len())..];
            line_count += 1;
        }
    }

    // Line 3 (normal+): Tags
    if zoom_level >= 1 && !result.tags.is_empty() {
        let tags_str = result
            .tags
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default().bg(bg)),
            Span::styled("tags: ", Style::default().fg(Color::DarkGray).bg(bg)),
            Span::styled(
                truncate(&tags_str, content_width.saturating_sub(8)),
                Style::default().fg(Color::Rgb(180, 230, 180)).bg(bg),
            ),
        ]));
    }

    f.render_widget(Paragraph::new(lines), area);
}

fn render_search_detail(f: &mut Frame, area: Rect, state: &AppState) {
    let result = match state.search_results.get(state.search_selected) {
        Some(r) => r,
        None => return,
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " MEMORY DETAIL ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                " Esc=back ",
                Style::default().fg(Color::DarkGray),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let content_width = inner.width.saturating_sub(4) as usize;

    // Format timestamp
    let local_time = result.created_at.with_timezone(&chrono::Local);
    let time_str = local_time.format("%Y-%m-%d %H:%M:%S").to_string();

    let mut lines = vec![
        // Header with type and ID
        Line::from(vec![
            Span::styled(
                format!(" [{}] ", result.memory_type),
                Style::default()
                    .fg(Color::Rgb(255, 200, 150))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(&result.id, Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
        // Score and timestamp
        Line::from(vec![
            Span::styled(" Score: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.4}", result.score),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled("  Created: ", Style::default().fg(Color::DarkGray)),
            Span::styled(time_str, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        // Tags
        Line::from(vec![
            Span::styled(" Tags: ", Style::default().fg(Color::DarkGray)),
            if result.tags.is_empty() {
                Span::styled("(none)", Style::default().fg(Color::DarkGray))
            } else {
                Span::styled(
                    result.tags.join(", "),
                    Style::default().fg(Color::Rgb(180, 230, 180)),
                )
            },
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " ─── Content ───",
            Style::default().fg(Color::Magenta),
        )),
        Line::from(""),
    ];

    // Word-wrap content
    let content = &result.content;
    let mut remaining = content.as_str();
    while !remaining.is_empty() {
        let take_len = remaining
            .char_indices()
            .take(content_width)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(remaining.len());
        let line_content = &remaining[..take_len];
        lines.push(Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(line_content, Style::default().fg(state.theme.fg())),
        ]));
        remaining = &remaining[take_len..];
    }

    f.render_widget(Paragraph::new(lines).scroll((0, 0)), inner);
}

// ═══════════════════════════════════════════════════════════════════════════
// VIEW 1: DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════

pub fn render_dashboard(f: &mut Frame, area: Rect, state: &AppState) {
    let content_area = with_ribbon_layout(f, area, state);

    // Split into main content (top) + detail panel (bottom)
    let main_split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),     // Main content takes most space
            Constraint::Length(12), // Detail panel (fixed 12 lines)
        ])
        .split(content_area);

    // 50/50 split: Todos on left, Activity on right
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Todos (full height)
            Constraint::Percentage(50), // Activity
        ])
        .split(main_split[0]);

    render_todos_panel(f, columns[0], state);
    render_activity_feed(f, columns[1], state);

    // Detail panel at bottom (full width, single column)
    render_dashboard_detail_panel(f, main_split[1], state);
}

/// Render dashboard detail panel (single column, full width)
fn render_dashboard_detail_panel(f: &mut Frame, area: Rect, state: &AppState) {
    // Top border
    let border = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(BORDER_SUBTLE));
    f.render_widget(border, area);

    // Adjust area for content (after border)
    let content_area = Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: area.height.saturating_sub(1),
    };

    let selected_todo = state.get_selected_dashboard_todo();
    let is_focused = state.focus_panel == FocusPanel::Detail;

    match selected_todo {
        Some(todo) => {
            let mut lines: Vec<Line> = Vec::new();
            let width = content_area.width as usize;

            // Header: ID + Title + focus indicator
            let short_id = todo.short_id();
            let focus_indicator = if is_focused { "▶ " } else { "  " };
            let focus_style = if is_focused {
                Style::default().fg(SAFFRON)
            } else {
                Style::default()
            };

            lines.push(Line::from(vec![
                Span::styled(focus_indicator, focus_style),
                Span::styled(
                    short_id,
                    Style::default().fg(DEEP_BLUE).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(
                    truncate(&todo.content, width.saturating_sub(20)),
                    Style::default()
                        .fg(TEXT_PRIMARY)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {} {:?}", todo.status.icon(), todo.status),
                    Style::default().fg(todo.status.color()),
                ),
                Span::styled(
                    if is_focused {
                        "  (↑↓ scroll, ←→ section, Esc exit)"
                    } else {
                        "  (Enter to focus)"
                    },
                    Style::default().fg(TEXT_DISABLED),
                ),
            ]));

            lines.push(Line::from(""));

            // Split remaining space: Notes (left half) | Activity (right half)
            let half_width = width / 2;
            let notes_scroll = state.notes_scroll;
            let activity_scroll = state.activity_scroll;
            let notes_focused = is_focused && state.detail_focus_column == 0;
            let activity_focused = is_focused && state.detail_focus_column == 1;

            // Build notes lines
            let mut note_lines: Vec<String> = Vec::new();
            if let Some(ref notes) = todo.notes {
                let line_width = half_width.saturating_sub(4);
                let mut chars = notes.chars().peekable();
                while chars.peek().is_some() {
                    let line_text: String = chars.by_ref().take(line_width).collect();
                    if line_text.is_empty() {
                        break;
                    }
                    note_lines.push(line_text);
                }
            }

            // Build activity lines
            let activity_items: Vec<_> = todo.comments.iter().rev().collect();

            // Render side by side (up to available_lines rows)
            let available_lines = (content_area.height as usize).saturating_sub(3);

            // Headers
            let notes_header = if notes_focused {
                "▶ NOTES"
            } else {
                "  NOTES"
            };
            let activity_header = if activity_focused {
                "▶ ACTIVITY"
            } else {
                "  ACTIVITY"
            };
            let notes_header_style = if notes_focused {
                Style::default().fg(SAFFRON)
            } else {
                Style::default().fg(TEXT_DISABLED)
            };
            let activity_header_style = if activity_focused {
                Style::default().fg(SAFFRON)
            } else {
                Style::default().fg(TEXT_DISABLED)
            };

            let scroll_info_notes = if note_lines.len() > available_lines {
                format!(" [{}/{}]", notes_scroll + 1, note_lines.len())
            } else {
                String::new()
            };
            let scroll_info_activity = if activity_items.len() > available_lines {
                format!(" [{}/{}]", activity_scroll + 1, activity_items.len())
            } else {
                String::new()
            };

            // Calculate left column content length
            let left_header_content = format!("{}{}", notes_header, scroll_info_notes);
            let left_header_len = left_header_content.chars().count();
            let left_pad = half_width.saturating_sub(left_header_len);

            lines.push(Line::from(vec![
                Span::styled(notes_header, notes_header_style),
                Span::styled(&scroll_info_notes, Style::default().fg(TEXT_DISABLED)),
                Span::styled(" ".repeat(left_pad), Style::default()),
                Span::styled("│", Style::default().fg(SAFFRON)),
                Span::styled(activity_header, activity_header_style),
                Span::styled(&scroll_info_activity, Style::default().fg(TEXT_DISABLED)),
            ]));

            // Content rows
            for i in 0..available_lines {
                let mut spans: Vec<Span> = Vec::new();

                // Notes column - fixed width (half_width chars total)
                let note_idx = notes_scroll + i;
                let mut left_content = String::new();
                if note_idx < note_lines.len() {
                    let is_cursor = notes_focused && i == 0;
                    let cursor = if is_cursor { "▸" } else { " " };
                    let style = if is_cursor {
                        Style::default()
                            .fg(TEXT_PRIMARY)
                            .add_modifier(Modifier::ITALIC)
                    } else {
                        Style::default()
                            .fg(TEXT_SECONDARY)
                            .add_modifier(Modifier::ITALIC)
                    };
                    let note_text = truncate(&note_lines[note_idx], half_width.saturating_sub(4));
                    left_content = format!(" {} {}", cursor, note_text);
                    spans.push(Span::styled(
                        format!(" {}", cursor),
                        Style::default().fg(SAFFRON),
                    ));
                    spans.push(Span::styled(note_text.clone(), style));
                }
                // Pad to exact half_width
                let left_len = left_content.chars().count();
                if left_len < half_width {
                    spans.push(Span::raw(" ".repeat(half_width - left_len)));
                }

                // Separator (single char)
                spans.push(Span::styled("│", Style::default().fg(SAFFRON)));

                // Activity column
                let activity_idx = activity_scroll + i;
                if activity_idx < activity_items.len() {
                    let comment = activity_items[activity_idx];
                    let is_cursor = activity_focused && i == 0;
                    let cursor = if is_cursor { "▸" } else { " " };
                    let icon = comment.comment_type.icon();
                    let style = if is_cursor {
                        Style::default().fg(TEXT_PRIMARY)
                    } else {
                        Style::default().fg(TEXT_SECONDARY)
                    };
                    let time_ago = format_duration_since(&comment.created_at);
                    let content_width = half_width.saturating_sub(time_ago.len() + 6);
                    let content_text = truncate(&comment.content, content_width);

                    spans.push(Span::styled(cursor, Style::default().fg(SAFFRON)));
                    spans.push(Span::styled(format!("{} ", icon), style));
                    spans.push(Span::styled(content_text, style));
                    spans.push(Span::styled(
                        format!(" {}", time_ago),
                        Style::default().fg(TEXT_DISABLED),
                    ));
                }

                lines.push(Line::from(spans));
            }

            let content = Paragraph::new(lines);
            f.render_widget(content, content_area);
        }
        None => {
            // No todo selected - show placeholder
            let placeholder = Paragraph::new(Line::from(vec![
                Span::styled("  ◇ ", Style::default().fg(TEXT_DISABLED)),
                Span::styled(
                    "Select a todo to view details",
                    Style::default().fg(TEXT_DISABLED),
                ),
            ]));
            f.render_widget(placeholder, content_area);
        }
    }
}

// ============================================================================
// PROJECTS VIEW - Full-width layout with proper spacing
// ============================================================================

/// Main Projects view - full-width ribbon + two-column layout + detail panel + lineage
fn render_projects_view(f: &mut Frame, area: Rect, state: &AppState) {
    let content_area = with_ribbon_layout(f, area, state);

    // Split into main content (top) + detail panel + lineage chain (bottom)
    let main_split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),    // Main content takes most space
            Constraint::Length(15), // Todo detail panel (fixed 15 lines for notes + activity)
            Constraint::Length(5),  // Lineage chain (fixed 5 lines)
        ])
        .split(content_area);

    // 50/50 columns for main content
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_split[0]);

    render_projects_sidebar(f, columns[0], state);
    render_todos_panel_right(f, columns[1], state);

    // Render todo detail panel above lineage
    render_todo_detail_panel(f, main_split[1], state);

    // Render lineage chain at bottom
    render_lineage_chain(f, main_split[2], state);

    // Render file popup if visible
    if state.file_popup_visible {
        render_file_popup(f, area, state);
    }

    // Render file preview if visible (on top of file popup)
    if state.file_preview_visible {
        render_file_preview(f, area, state);
    }

    // Render codebase path input if active
    if state.codebase_input_active {
        render_codebase_input(f, area, state);
    }
}

/// Render codebase path input popup
fn render_codebase_input(f: &mut Frame, area: Rect, state: &AppState) {
    // Use 90% of screen width
    let popup_width = ((area.width as f32) * 0.9) as u16;
    let popup_height = 7;
    let popup_x = (area.width - popup_width) / 2;
    let popup_y = (area.height - popup_height) / 2;
    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear background
    f.render_widget(Clear, popup_area);

    // Calculate visible path width (popup width minus borders and prefix)
    let visible_width = popup_width.saturating_sub(8) as usize;
    let path = &state.codebase_input_path;

    // Show end of path if too long (so user sees what they're typing)
    let display_path = if path.len() > visible_width {
        format!("...{}", &path[path.len() - visible_width + 3..])
    } else {
        path.clone()
    };

    let lines = vec![
        Line::from(Span::styled(
            " 📂 Enter codebase directory path: ",
            Style::default().fg(SAFFRON).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "    (edit path or press Enter to use default)",
            Style::default().fg(TEXT_DISABLED),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(" > ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(display_path, Style::default().fg(TEXT_PRIMARY)),
            Span::styled("█", Style::default().fg(SAFFRON)), // Cursor
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Enter", Style::default().fg(SAFFRON).add_modifier(Modifier::BOLD)),
            Span::styled("=scan  ", Style::default().fg(TEXT_DISABLED)),
            Span::styled("Esc", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
            Span::styled("=cancel  ", Style::default().fg(TEXT_DISABLED)),
            Span::styled("Backspace", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
            Span::styled("=delete", Style::default().fg(TEXT_DISABLED)),
        ]),
    ];

    let popup = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(SAFFRON))
            .style(Style::default().bg(Color::Rgb(25, 25, 30))),
    );
    f.render_widget(popup, popup_area);
}

/// Get tree node info for navigation (is_dir, folder_path, absolute_path)
pub fn get_tree_node_info(files: &[TuiFileMemory], expanded: &std::collections::HashSet<String>) -> Vec<(bool, String, String)> {
    build_file_tree(files, expanded)
        .into_iter()
        .map(|node| (node.is_dir, node.folder_path, node.absolute_path))
        .collect()
}

/// Build tree structure from flat file list with collapse/expand support
fn build_file_tree(files: &[TuiFileMemory], expanded: &std::collections::HashSet<String>) -> Vec<FileTreeNode> {
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct DirEntry {
        files: Vec<(String, TuiFileMemory)>, // (filename, file)
        dirs: BTreeMap<String, DirEntry>,
        total_size: u64,
    }

    // Build directory tree
    let mut root = DirEntry::default();
    for file in files {
        let parts: Vec<&str> = file.path.split(['/', '\\']).collect();
        let mut current = &mut root;

        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                // It's a file
                current.files.push((part.to_string(), file.clone()));
                current.total_size += file.size_bytes;
            } else {
                // It's a directory
                current.total_size += file.size_bytes;
                current = current.dirs.entry(part.to_string()).or_default();
            }
        }
    }

    // Flatten to tree nodes with proper indentation
    fn flatten(
        entry: &DirEntry,
        current_path: &str,
        is_last_stack: &[bool],
        depth: usize,
        expanded: &std::collections::HashSet<String>,
        nodes: &mut Vec<FileTreeNode>,
    ) {
        let dirs: Vec<_> = entry.dirs.iter().collect();
        let total_items = dirs.len() + entry.files.len();
        let mut item_idx = 0;

        // Directories first
        for (name, dir) in &dirs {
            let is_last = item_idx == total_items - 1;
            let connector = if is_last { "└── " } else { "├── " };

            // Build prefix from stack
            let mut line_prefix = String::new();
            for &was_last in is_last_stack.iter() {
                line_prefix.push_str(if was_last { "    " } else { "│   " });
            }

            // Full path for this folder
            let folder_path = if current_path.is_empty() {
                name.to_string()
            } else {
                format!("{}/{}", current_path, name)
            };

            let is_expanded = expanded.contains(&folder_path);
            let folder_icon = if is_expanded { "▼ 📁" } else { "▶ 📁" };

            nodes.push(FileTreeNode {
                display: format!("{}{}{} {}/", line_prefix, connector, folder_icon, name),
                is_dir: true,
                name: name.to_string(),
                folder_path: folder_path.clone(),
                absolute_path: String::new(),
                size: dir.total_size,
                file_type: String::new(),
                depth,
            });

            // Only recurse if expanded
            if is_expanded {
                let mut new_stack = is_last_stack.to_vec();
                new_stack.push(is_last);
                flatten(dir, &folder_path, &new_stack, depth + 1, expanded, nodes);
            }

            item_idx += 1;
        }

        // Then files (sorted by name)
        let mut sorted_files: Vec<_> = entry.files.iter().collect();
        sorted_files.sort_by(|a, b| a.0.cmp(&b.0));

        for (name, file) in sorted_files {
            let is_last = item_idx == total_items - 1;
            let connector = if is_last { "└── " } else { "├── " };

            let mut line_prefix = String::new();
            for &was_last in is_last_stack.iter() {
                line_prefix.push_str(if was_last { "    " } else { "│   " });
            }

            // For files, folder_path is the FULL file path (used for lookup)
            // Use the original path from the file struct to ensure exact match
            nodes.push(FileTreeNode {
                display: format!("{}{}   {} {}", line_prefix, connector, file.type_icon(), name),
                is_dir: false,
                name: name.clone(),
                folder_path: file.path.clone(), // Full file path for lookup
                absolute_path: file.absolute_path.clone(),
                size: file.size_bytes,
                file_type: file.file_type.clone(),
                depth,
            });

            item_idx += 1;
        }
    }

    let mut nodes = Vec::new();
    flatten(&root, "", &[], 0, expanded, &mut nodes);
    nodes
}

/// A node in the file tree display
struct FileTreeNode {
    display: String,
    is_dir: bool,
    name: String,
    folder_path: String, // Full path for folders, relative path for files
    absolute_path: String, // Absolute path for files (empty for folders)
    size: u64,
    file_type: String,
    #[allow(dead_code)]
    depth: usize,
}

impl FileTreeNode {
    fn format_size(&self) -> String {
        let bytes = self.size;
        if bytes == 0 {
            String::new()
        } else if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        }
    }

    fn type_color(&self) -> Color {
        match self.file_type.to_lowercase().as_str() {
            "rust" => Color::Rgb(220, 160, 120),
            "typescript" | "javascript" => Color::Rgb(100, 180, 220),
            "python" => Color::Rgb(160, 200, 100),
            "go" => Color::Rgb(100, 200, 220),
            "markdown" => Color::Rgb(180, 180, 220),
            "json" | "yaml" | "toml" => Color::Rgb(200, 180, 140),
            _ => TEXT_SECONDARY,
        }
    }
}

/// Render file explorer popup (centered overlay) - broot/nnn style tree view
fn render_file_popup(f: &mut Frame, area: Rect, state: &AppState) {
    // Create centered popup area (85% width, 85% height)
    let popup_width = (area.width as f32 * 0.85) as u16;
    let popup_height = (area.height as f32 * 0.85) as u16;
    let popup_x = (area.width - popup_width) / 2;
    let popup_y = (area.height - popup_height) / 2;

    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear background
    f.render_widget(Clear, popup_area);

    // Get project name and files
    let project_name = state
        .selected_project_id()
        .and_then(|pid| state.projects.iter().find(|p| p.id == pid))
        .map(|p| p.name.clone())
        .unwrap_or_else(|| "Project".to_string());

    let files = state
        .selected_project_id()
        .and_then(|pid| state.get_project_files(&pid));

    let mut lines: Vec<Line> = Vec::new();

    // Calculate content width for size alignment
    let content_width = popup_width.saturating_sub(4) as usize;

    if let Some(files) = files {
        if files.is_empty() {
            lines.push(Line::from(Span::styled(
                " No files indexed. Press S to scan codebase.",
                Style::default().fg(TEXT_DISABLED),
            )));
        } else {
            // Build tree structure with collapse/expand
            let tree_nodes = build_file_tree(files, &state.expanded_folders);

            // Stats
            let total_size: u64 = files.iter().map(|f| f.size_bytes).sum();
            let total_size_str = if total_size < 1024 * 1024 {
                format!("{:.1} KB", total_size as f64 / 1024.0)
            } else {
                format!("{:.1} MB", total_size as f64 / (1024.0 * 1024.0))
            };

            lines.push(Line::from(vec![
                Span::styled(" ", Style::default()),
                Span::styled(
                    format!("{} files", files.len()),
                    Style::default().fg(TEXT_SECONDARY),
                ),
                Span::styled(" │ ", Style::default().fg(Color::Rgb(60, 60, 60))),
                Span::styled(total_size_str, Style::default().fg(SAFFRON)),
                Span::styled(" total", Style::default().fg(TEXT_DISABLED)),
            ]));
            lines.push(Line::from(""));

            // Tree view with scroll
            let visible_height = popup_height.saturating_sub(8) as usize;
            let scroll = state.file_popup_scroll;

            for (i, node) in tree_nodes.iter().skip(scroll).take(visible_height).enumerate() {
                let absolute_idx = scroll + i;
                let is_selected = absolute_idx == state.selected_file;
                let bg = if is_selected {
                    Color::Rgb(45, 45, 55)
                } else {
                    Color::Reset
                };

                let size_str = node.format_size();
                let display_len = node.display.chars().count();
                let padding = content_width.saturating_sub(display_len + size_str.len() + 4);

                // Selection indicator
                let selector = if is_selected { "▸ " } else { "  " };
                let mut spans = vec![
                    Span::styled(
                        selector,
                        Style::default().fg(SAFFRON).bg(bg),
                    ),
                ];

                // Split display into tree chars and name for coloring
                if node.is_dir {
                    spans.push(Span::styled(
                        node.display.clone(),
                        Style::default().fg(SAFFRON).bg(bg),
                    ));
                } else {
                    spans.push(Span::styled(
                        node.display.clone(),
                        Style::default().fg(node.type_color()).bg(bg),
                    ));
                }

                // Padding dots or spaces
                spans.push(Span::styled(
                    " ".repeat(padding),
                    Style::default().fg(Color::Rgb(40, 40, 40)).bg(bg),
                ));

                // Size right-aligned
                spans.push(Span::styled(
                    format!("{:>8}", size_str),
                    Style::default().fg(TEXT_DISABLED).bg(bg),
                ));

                lines.push(Line::from(spans));
            }

            // Scroll indicator
            if tree_nodes.len() > visible_height {
                lines.push(Line::from(""));
                let scroll_pct = (scroll as f32 / (tree_nodes.len() - visible_height) as f32 * 100.0) as usize;
                lines.push(Line::from(Span::styled(
                    format!(" ↑↓ scroll │ {}/{} │ {}%", scroll + 1, tree_nodes.len(), scroll_pct.min(100)),
                    Style::default().fg(TEXT_DISABLED),
                )));
            }
        }
    } else {
        lines.push(Line::from(Span::styled(
            " Loading files...",
            Style::default().fg(TEXT_DISABLED),
        )));
    }

    // Build block with title
    let title = format!(" 📁 {} ", project_name);
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(SAFFRON).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
        .style(Style::default().bg(Color::Rgb(18, 18, 22)));

    // Footer keybinds
    let footer = Line::from(vec![
        Span::styled(" ↑↓", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(" nav  ", Style::default().fg(Color::Rgb(60, 60, 60))),
        Span::styled("Enter", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(" open/expand  ", Style::default().fg(Color::Rgb(60, 60, 60))),
        Span::styled("←", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(" collapse  ", Style::default().fg(Color::Rgb(60, 60, 60))),
        Span::styled("Esc", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(" close", Style::default().fg(Color::Rgb(60, 60, 60))),
    ]);
    lines.push(Line::from(""));
    lines.push(footer);

    let popup = Paragraph::new(lines).block(block);
    f.render_widget(popup, popup_area);
}

/// Render file preview popup (shows file content with line numbers and key items)
fn render_file_preview(f: &mut Frame, area: Rect, state: &AppState) {
    // Create centered popup area (90% width, 90% height)
    let popup_width = (area.width as f32 * 0.90) as u16;
    let popup_height = (area.height as f32 * 0.90) as u16;
    let popup_x = (area.width - popup_width) / 2;
    let popup_y = (area.height - popup_height) / 2;

    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear background
    f.render_widget(Clear, popup_area);

    let mut lines: Vec<Line> = Vec::new();

    // File type icon
    let type_icon = match state.file_preview_file_type.to_lowercase().as_str() {
        "rust" => "🦀",
        "typescript" | "javascript" => "📜",
        "python" => "🐍",
        "go" => "🐹",
        "java" => "☕",
        "c" | "cpp" | "c++" => "⚙️",
        "markdown" => "📝",
        "json" | "yaml" | "toml" => "📋",
        _ => "📄",
    };

    // Header with key items
    if !state.file_preview_key_items.is_empty() {
        let key_items_str = state.file_preview_key_items.iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(" │ ");
        lines.push(Line::from(vec![
            Span::styled(" Key: ", Style::default().fg(TEXT_DISABLED)),
            Span::styled(key_items_str, Style::default().fg(SAFFRON)),
        ]));
        lines.push(Line::from(""));
    }

    // Content with line numbers
    let visible_height = popup_height.saturating_sub(10) as usize;
    let scroll = state.file_preview_scroll;
    let total_lines = state.file_preview_content.len();
    let line_num_width = total_lines.to_string().len();

    for (i, line) in state.file_preview_content.iter()
        .skip(scroll)
        .take(visible_height)
        .enumerate()
    {
        let line_num = scroll + i + 1;
        let line_num_str = format!("{:>width$} ", line_num, width = line_num_width);

        // Truncate long lines for display
        let display_line = if line.len() > (popup_width as usize - line_num_width - 8) {
            format!("{}...", &line[..popup_width as usize - line_num_width - 11])
        } else {
            line.clone()
        };

        lines.push(Line::from(vec![
            Span::styled(line_num_str, Style::default().fg(Color::Rgb(80, 80, 100))),
            Span::styled("│ ", Style::default().fg(Color::Rgb(50, 50, 60))),
            Span::styled(display_line, Style::default().fg(TEXT_PRIMARY)),
        ]));
    }

    // Scroll info
    if total_lines > visible_height {
        lines.push(Line::from(""));
        let scroll_pct = if total_lines > visible_height {
            (scroll as f32 / (total_lines - visible_height) as f32 * 100.0) as usize
        } else {
            0
        };
        lines.push(Line::from(Span::styled(
            format!(" Line {}-{} of {} │ {}%",
                scroll + 1,
                (scroll + visible_height).min(total_lines),
                total_lines,
                scroll_pct.min(100)),
            Style::default().fg(TEXT_DISABLED),
        )));
    }

    // Build block with title
    let filename = std::path::Path::new(&state.file_preview_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&state.file_preview_path);
    let title = format!(" {} {} ({} lines) ", type_icon, filename, state.file_preview_line_count);
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(SAFFRON).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
        .style(Style::default().bg(Color::Rgb(18, 18, 22)));

    // Footer keybinds
    let footer = Line::from(vec![
        Span::styled(" ↑↓/jk", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(" scroll  ", Style::default().fg(Color::Rgb(60, 60, 60))),
        Span::styled("PgUp/PgDn", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(" page  ", Style::default().fg(Color::Rgb(60, 60, 60))),
        Span::styled("Home/End", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(" jump  ", Style::default().fg(Color::Rgb(60, 60, 60))),
        Span::styled("Esc", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(" close", Style::default().fg(Color::Rgb(60, 60, 60))),
    ]);
    lines.push(Line::from(""));
    lines.push(footer);

    let popup = Paragraph::new(lines).block(block);
    f.render_widget(popup, popup_area);
}

/// Single status ribbon — clean, readable
fn render_status_ribbon(f: &mut Frame, area: Rect, state: &AppState) {
    let width = area.width as usize;
    let ribbon_bg = Color::Rgb(40, 38, 35);

    let in_progress = state.in_progress_todos();
    let mut spans: Vec<Span> = Vec::new();

    let pending_count = state
        .todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::Todo)
        .count();
    let done_count = state
        .todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::Done)
        .count();

    if let Some(current) = in_progress.first() {
        let duration = format_duration_since(&current.created_at);

        spans.push(Span::styled(
            " WORKING ",
            Style::default()
                .fg(Color::Black)
                .bg(SAFFRON)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(" ", Style::default().bg(ribbon_bg)));
        spans.push(Span::styled(
            truncate(&current.content, (width / 2).min(60)),
            Style::default()
                .fg(TEXT_PRIMARY)
                .bg(ribbon_bg)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!("  {}", duration),
            Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
        ));
    } else {
        spans.push(Span::styled(
            " IDLE ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Rgb(80, 78, 75))
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(" ", Style::default().bg(ribbon_bg)));

        // Show last operation if fresh
        if let Some(ref op) = state.current_operation {
            if op.is_fresh() {
                spans.push(Span::styled(
                    format!("{} ", op.op_type.label()),
                    Style::default().fg(SAFFRON).bg(ribbon_bg),
                ));
                spans.push(Span::styled(
                    truncate(&op.content_preview, (width / 3).min(40)),
                    Style::default().fg(TEXT_PRIMARY).bg(ribbon_bg),
                ));
                if let Some(count) = op.count {
                    spans.push(Span::styled(
                        format!(" ({} found)", count),
                        Style::default().fg(GOLD).bg(ribbon_bg),
                    ));
                }
            }
        }
    }

    // Right-aligned counts
    let right = format!(" {} done  {} pending ", done_count, pending_count);
    let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    let gap = width.saturating_sub(used).saturating_sub(right.chars().count());
    if gap > 0 {
        spans.push(Span::styled(
            " ".repeat(gap),
            Style::default().bg(ribbon_bg),
        ));
    }
    spans.push(Span::styled(
        right,
        Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
    ));

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Standard view layout with single ribbon at top
/// Returns the content area below ribbon and spacer
fn with_ribbon_layout(f: &mut Frame, area: Rect, state: &AppState) -> Rect {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Status ribbon
            Constraint::Length(1), // Spacer for breathing room
            Constraint::Min(5),    // Main content
        ])
        .split(area);

    render_status_ribbon(f, rows[0], state);
    rows[2]
}

/// Memory operation ribbon - shows current operation and context being used
fn render_memory_ribbon(f: &mut Frame, area: Rect, state: &AppState) {
    let width = area.width as usize;
    let ribbon_bg = Color::Rgb(40, 38, 35); // Same as status ribbon

    let mut spans: Vec<Span> = Vec::new();

    // Left side: Current operation
    if let Some(ref op) = state.current_operation {
        if op.is_fresh() {
            let op_color = op.op_type.color();

            // Operation badge (same format as " WORKING ")
            spans.push(Span::styled(
                format!(" {} ", op.op_type.label()),
                Style::default()
                    .fg(Color::Black)
                    .bg(op_color)
                    .add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled(" ", Style::default().bg(ribbon_bg)));

            // Content preview
            spans.push(Span::styled(
                truncate(&op.content_preview, 40),
                Style::default().fg(TEXT_PRIMARY).bg(ribbon_bg),
            ));

            // Memory type if available
            if let Some(ref mem_type) = op.memory_type {
                spans.push(Span::styled(
                    format!("  [{}]", mem_type),
                    Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
                ));
            }

            // Latency if available
            if let Some(latency) = op.latency_ms {
                spans.push(Span::styled(
                    format!("  {}ms", latency as u32),
                    Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
                ));
            }

            // Count if available (for recalls)
            if let Some(count) = op.count {
                spans.push(Span::styled(
                    format!(" ({} found)", count),
                    Style::default().fg(GOLD).bg(ribbon_bg),
                ));
            }
        } else {
            // Stale operation - show dimmed
            spans.push(Span::styled(
                format!(" {} ", op.op_type.icon()),
                Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
            ));
            spans.push(Span::styled(" ", Style::default().bg(ribbon_bg)));
            spans.push(Span::styled(
                truncate(&op.content_preview, 40),
                Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
            ));
        }
    } else {
        // No badge when idle - just subtle text
        spans.push(Span::styled(
            " ○ Awaiting memory activity",
            Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
        ));
    }

    // Separator (same as status ribbon)
    spans.push(Span::styled(
        "  │  ",
        Style::default().fg(Color::Rgb(60, 55, 50)).bg(ribbon_bg),
    ));

    // Right side: Last used memory (context)
    if let Some(ref mem) = state.last_used_memory {
        spans.push(Span::styled(
            "CONTEXT ",
            Style::default()
                .fg(SAFFRON)
                .bg(ribbon_bg)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            truncate(&mem.content_preview, 30),
            Style::default().fg(TEXT_PRIMARY).bg(ribbon_bg),
        ));
        spans.push(Span::styled(
            format!("  [{}]", mem.memory_type),
            Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
        ));
        spans.push(Span::styled(
            format!("  {}", mem.age_display()),
            Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
        ));
    } else {
        spans.push(Span::styled(
            "No context",
            Style::default().fg(TEXT_DISABLED).bg(ribbon_bg),
        ));
    }

    // Pad to full width (same method as status ribbon)
    let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    if used < width {
        spans.push(Span::styled(
            " ".repeat(width.saturating_sub(used)),
            Style::default().bg(ribbon_bg),
        ));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Left sidebar - projects list with flat navigation (projects + todos)
fn render_projects_sidebar(f: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),    // Projects list
            Constraint::Length(1), // Footer
        ])
        .split(area);

    let inner = chunks[0];
    let width = inner.width as usize;
    let mut lines: Vec<Line> = Vec::new();
    let is_left_focused = state.focus_panel == FocusPanel::Left;

    // Header with breathing room
    let proj_count = state.projects.len();
    let header_line = "─".repeat(width.saturating_sub(14));
    lines.push(Line::from(vec![
        Span::styled(
            format!(" PROJECTS {} ", proj_count),
            Style::default().fg(TEXT_SECONDARY),
        ),
        Span::styled(header_line, Style::default().fg(Color::Rgb(40, 40, 40))),
    ]));
    lines.push(Line::from("")); // breathing room

    // Flat index for navigation - includes projects and their expanded todos
    let mut flat_idx = 0;

    // Separate root projects and sub-projects
    let root_projects: Vec<_> = state
        .projects
        .iter()
        .filter(|p| p.parent_id.is_none())
        .collect();
    let sub_projects: Vec<_> = state
        .projects
        .iter()
        .filter(|p| p.parent_id.is_some())
        .collect();

    // Render projects with folder icons - root projects first, then sub-projects under them
    for project in root_projects.iter() {
        if lines.len() >= inner.height as usize - 1 {
            break;
        }

        // Render the root project
        flat_idx = render_project_line(
            &mut lines,
            project,
            state,
            flat_idx,
            width,
            is_left_focused,
            0,
        );

        // Find and render sub-projects of this project
        for subproject in sub_projects
            .iter()
            .filter(|sp| sp.parent_id.as_ref() == Some(&project.id))
        {
            if lines.len() >= inner.height as usize - 1 {
                break;
            }
            flat_idx = render_project_line(
                &mut lines,
                subproject,
                state,
                flat_idx,
                width,
                is_left_focused,
                1,
            );
        }
    }

    // INBOX section with spacing
    let standalone = state.standalone_todos();
    if !standalone.is_empty() && lines.len() < inner.height as usize - 2 {
        lines.push(Line::from("")); // breathing room
        let inbox_line = "─".repeat(width.saturating_sub(12));
        lines.push(Line::from(vec![
            Span::styled(
                format!(" INBOX {} ", standalone.len()),
                Style::default().fg(TEXT_SECONDARY),
            ),
            Span::styled(inbox_line, Style::default().fg(Color::Rgb(40, 40, 40))),
        ]));
        lines.push(Line::from("")); // breathing room
        let max_inbox = if state.expand_sections {
            standalone.len()
        } else {
            5
        };
        for todo in standalone.iter().take(max_inbox) {
            if lines.len() >= inner.height as usize {
                break;
            }
            let todo_selected = state.projects_selected == flat_idx;
            let is_selected_and_focused = todo_selected && is_left_focused;
            lines.push(render_sidebar_todo(
                todo,
                width,
                todo_selected,
                is_left_focused,
            ));
            if is_selected_and_focused && lines.len() < inner.height as usize {
                lines.push(render_action_bar(todo));
            }
            flat_idx += 1;
        }
    }

    // FILES section - show files for selected project if expanded
    if let Some(project_id) = state.selected_project_id() {
        if state.is_files_expanded(&project_id) && lines.len() < inner.height as usize - 2 {
            lines.push(Line::from("")); // breathing room

            let files = state.get_project_files(&project_id);
            let file_count = files.map(|f| f.len()).unwrap_or(0);
            let loading = state.is_files_loading(&project_id);

            let files_line = "─".repeat(width.saturating_sub(12));
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" FILES {} ", file_count),
                    Style::default().fg(TEXT_SECONDARY),
                ),
                Span::styled(files_line, Style::default().fg(Color::Rgb(40, 40, 40))),
            ]));
            lines.push(Line::from("")); // breathing room

            if loading {
                lines.push(Line::from(Span::styled(
                    "  ◐ Loading files...",
                    Style::default().fg(TEXT_DISABLED),
                )));
            } else if let Some(files) = files {
                let max_files = if state.expand_sections { files.len() } else { 8 };
                for (i, file) in files.iter().take(max_files).enumerate() {
                    if lines.len() >= inner.height as usize {
                        break;
                    }
                    let is_file_selected = i == state.selected_file;
                    lines.push(render_file_line(file, width, is_file_selected, is_left_focused));
                }
                if files.len() > max_files {
                    lines.push(Line::from(Span::styled(
                        format!("       +{} more (press e to expand)", files.len() - max_files),
                        Style::default().fg(TEXT_DISABLED),
                    )));
                }
            } else {
                lines.push(Line::from(Span::styled(
                    "  No files indexed. Press S to scan.",
                    Style::default().fg(TEXT_DISABLED),
                )));
            }
        }
    }

    f.render_widget(Paragraph::new(lines), inner);

    // Footer shows panel focus state with codebase actions
    let footer = if is_left_focused {
        // Check if selected project has codebase indexed
        let codebase_hint = if let Some(pid) = state.selected_project_id() {
            if state.is_scanning(&pid) {
                Span::styled("◐ scanning...", Style::default().fg(SAFFRON))
            } else if state.is_project_indexed(&pid) {
                Span::styled("f=files ", Style::default().fg(Color::Rgb(100, 200, 100)))
            } else {
                Span::styled("S=scan ", Style::default().fg(Color::Rgb(255, 180, 100)))
            }
        } else {
            Span::styled("", Style::default())
        };

        Line::from(vec![
            Span::styled(" ▸ ", Style::default().fg(SAFFRON)),
            Span::styled("↑↓", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(" nav ", Style::default().fg(Color::Rgb(60, 60, 60))),
            Span::styled("→", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(" todos ", Style::default().fg(Color::Rgb(60, 60, 60))),
            Span::styled("Enter", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(" expand ", Style::default().fg(Color::Rgb(60, 60, 60))),
            Span::styled("│ ", Style::default().fg(Color::Rgb(50, 50, 50))),
            codebase_hint,
        ])
    } else {
        Line::from(vec![
            Span::styled("   ", Style::default()),
            Span::styled("←", Style::default().fg(TEXT_DISABLED)),
            Span::styled(" return", Style::default().fg(Color::Rgb(50, 50, 50))),
        ])
    };
    f.render_widget(Paragraph::new(footer), chunks[1]);
}

/// Format duration since a timestamp
fn format_duration_since(created_at: &chrono::DateTime<chrono::Utc>) -> String {
    let now = chrono::Utc::now();
    let duration = now.signed_duration_since(*created_at);

    let hours = duration.num_hours();
    let mins = duration.num_minutes() % 60;

    if hours > 0 {
        format!("{}h {:02}m", hours, mins)
    } else {
        format!("{}m", mins)
    }
}

/// Right panel - todos for selected project
fn render_todos_panel_right(f: &mut Frame, area: Rect, state: &AppState) {
    let width = area.width as usize;
    let is_focused = state.focus_panel == FocusPanel::Right;

    // Split area for header, content, and footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),    // Content
            Constraint::Length(1), // Footer
        ])
        .split(area);

    let content_area = chunks[0];
    let content_height = content_area.height as usize;
    let mut lines: Vec<Line> = Vec::new();

    // Build visual order list (same order as sidebar: root projects, then sub-projects under each)
    // This is needed because projects_selected is a flat index in visual order, not array index
    let mut visual_order: Vec<&TuiProject> = Vec::new();
    let root_projects: Vec<_> = state
        .projects
        .iter()
        .filter(|p| p.parent_id.is_none())
        .collect();
    let sub_projects: Vec<_> = state
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

    // Get todos for selected project or inbox
    // For parent projects, also include todos from sub-projects
    let (title, todos): (String, Vec<&TuiTodo>) = if state.projects_selected < visual_order.len() {
        let project = visual_order[state.projects_selected];
        let mut all_todos = state.todos_for_project(&project.id);

        // If this is a parent project (not a sub-project), also include todos from its sub-projects
        if project.parent_id.is_none() {
            for subproject in sub_projects
                .iter()
                .filter(|sp| sp.parent_id.as_ref() == Some(&project.id))
            {
                all_todos.extend(state.todos_for_project(&subproject.id));
            }
        }

        (project.name.clone(), all_todos)
    } else {
        ("Inbox".to_string(), state.standalone_todos())
    };

    // Header with focus indicator
    let header_line = "─".repeat(width.saturating_sub(title.len() + 6));
    let focus_indicator = if is_focused { "▸ " } else { "  " };
    lines.push(Line::from(vec![
        Span::styled(focus_indicator, Style::default().fg(SAFFRON)),
        Span::styled(
            format!("{} ", title),
            Style::default()
                .fg(TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(header_line, Style::default().fg(Color::Rgb(40, 40, 40))),
    ]));
    lines.push(Line::from("")); // breathing room

    if todos.is_empty() {
        lines.push(Line::from(Span::styled(
            "   No tasks yet",
            Style::default().fg(TEXT_DISABLED),
        )));
    } else {
        // Build flat list of todos for selection tracking
        let mut flat_todos: Vec<&TuiTodo> = Vec::new();

        // Collect parent todos only (exclude subtasks) in order: in_progress, todo, blocked, done
        let in_progress: Vec<_> = todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::InProgress && t.parent_id.is_none())
            .cloned()
            .collect();
        let todo_items: Vec<_> = todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::Todo && t.parent_id.is_none())
            .cloned()
            .collect();
        let blocked: Vec<_> = todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::Blocked && t.parent_id.is_none())
            .cloned()
            .collect();
        let done: Vec<_> = todos
            .iter()
            .filter(|t| t.status == TuiTodoStatus::Done && t.parent_id.is_none())
            .cloned()
            .collect();
        // Keep all todos for subtask lookup
        let all_todos: Vec<_> = todos.iter().cloned().collect();

        flat_todos.extend(in_progress.iter());
        flat_todos.extend(todo_items.iter());
        flat_todos.extend(blocked.iter());
        flat_todos.extend(done.iter());

        let mut todo_idx = 0;

        // In Progress section
        if !in_progress.is_empty() {
            lines.push(Line::from(Span::styled(
                format!(" ◐ In Progress ({})", in_progress.len()),
                Style::default().fg(SAFFRON),
            )));
            let max_in_progress = if state.expand_sections {
                in_progress.len()
            } else {
                4
            };
            for todo in in_progress.iter().take(max_in_progress) {
                if lines.len() >= content_height - 2 {
                    break;
                }
                let is_selected = state.todos_selected == todo_idx && is_focused;
                lines.push(render_todo_row_with_selection(
                    todo,
                    width,
                    is_selected,
                    is_focused,
                ));
                if is_selected {
                    lines.push(render_action_bar(todo));
                }
                todo_idx += 1;
                // Render subtasks of this todo
                for subtask in all_todos
                    .iter()
                    .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
                {
                    if lines.len() >= content_height - 2 {
                        break;
                    }
                    let sub_selected = state.todos_selected == todo_idx && is_focused;
                    lines.push(render_todo_row_with_indent(
                        subtask,
                        width,
                        sub_selected,
                        is_focused,
                        1,
                    ));
                    if sub_selected {
                        lines.push(render_action_bar(subtask));
                    }
                    todo_idx += 1;
                }
            }
        }

        // Todo section
        if !todo_items.is_empty() && lines.len() < content_height - 2 {
            lines.push(Line::from(Span::styled(
                format!(" ○ Todo ({})", todo_items.len()),
                Style::default().fg(TEXT_SECONDARY),
            )));
            let max_todos = if state.expand_sections {
                content_height.saturating_sub(lines.len() + 4)
            } else {
                (content_height.saturating_sub(lines.len() + 4)).min(8)
            };
            for todo in todo_items.iter().take(max_todos) {
                if lines.len() >= content_height - 2 {
                    break;
                }
                let is_selected = state.todos_selected == todo_idx && is_focused;
                lines.push(render_todo_row_with_selection(
                    todo,
                    width,
                    is_selected,
                    is_focused,
                ));
                if is_selected {
                    lines.push(render_action_bar(todo));
                }
                todo_idx += 1;
                // Render subtasks of this todo
                for subtask in all_todos
                    .iter()
                    .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
                {
                    if lines.len() >= content_height - 2 {
                        break;
                    }
                    let sub_selected = state.todos_selected == todo_idx && is_focused;
                    lines.push(render_todo_row_with_indent(
                        subtask,
                        width,
                        sub_selected,
                        is_focused,
                        1,
                    ));
                    if sub_selected {
                        lines.push(render_action_bar(subtask));
                    }
                    todo_idx += 1;
                }
            }
            if todo_items.len() > max_todos {
                lines.push(Line::from(Span::styled(
                    format!(
                        "      +{} more (press e to expand)",
                        todo_items.len() - max_todos
                    ),
                    Style::default().fg(TEXT_DISABLED),
                )));
            }
        }

        // Blocked section
        if !blocked.is_empty() && lines.len() < content_height - 2 {
            lines.push(Line::from(Span::styled(
                format!(" ⊘ Blocked ({})", blocked.len()),
                Style::default().fg(MAROON),
            )));
            let max_blocked = if state.expand_sections {
                blocked.len()
            } else {
                3
            };
            for todo in blocked.iter().take(max_blocked) {
                if lines.len() >= content_height - 2 {
                    break;
                }
                let is_selected = state.todos_selected == todo_idx && is_focused;
                lines.push(render_todo_row_with_selection(
                    todo,
                    width,
                    is_selected,
                    is_focused,
                ));
                if is_selected {
                    lines.push(render_action_bar(todo));
                }
                todo_idx += 1;
                // Render subtasks of this todo
                for subtask in all_todos
                    .iter()
                    .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
                {
                    if lines.len() >= content_height - 2 {
                        break;
                    }
                    let sub_selected = state.todos_selected == todo_idx && is_focused;
                    lines.push(render_todo_row_with_indent(
                        subtask,
                        width,
                        sub_selected,
                        is_focused,
                        1,
                    ));
                    if sub_selected {
                        lines.push(render_action_bar(subtask));
                    }
                    todo_idx += 1;
                }
            }
        }

        // Done section (shows completed items) - always show if there are done items
        if !done.is_empty() {
            lines.push(Line::from(Span::styled(
                format!(" ● Completed ({})", done.len()),
                Style::default().fg(GOLD),
            )));
            let remaining_height = content_height.saturating_sub(lines.len() + 2);
            let show_count = if state.expand_sections {
                remaining_height.min(done.len())
            } else {
                remaining_height.min(done.len()).min(5)
            };
            if show_count > 0 {
                for todo in done.iter().take(show_count) {
                    let is_selected = state.todos_selected == todo_idx && is_focused;
                    lines.push(render_todo_row_with_selection(
                        todo,
                        width,
                        is_selected,
                        is_focused,
                    ));
                    if is_selected {
                        lines.push(render_action_bar(todo));
                    }
                    todo_idx += 1;
                    // Render subtasks of this todo
                    for subtask in all_todos
                        .iter()
                        .filter(|t| t.parent_id.as_ref() == Some(&todo.id))
                    {
                        if lines.len() >= content_height - 2 {
                            break;
                        }
                        let sub_selected = state.todos_selected == todo_idx && is_focused;
                        lines.push(render_todo_row_with_indent(
                            subtask,
                            width,
                            sub_selected,
                            is_focused,
                            1,
                        ));
                        if sub_selected {
                            lines.push(render_action_bar(subtask));
                        }
                        todo_idx += 1;
                    }
                }
            }
            if done.len() > show_count && show_count > 0 {
                lines.push(Line::from(Span::styled(
                    format!(
                        "      +{} more (press e to expand)",
                        done.len() - show_count
                    ),
                    Style::default().fg(TEXT_DISABLED),
                )));
            }
        }
    }

    f.render_widget(Paragraph::new(lines), content_area);

    // Footer shows panel state
    let footer = if is_focused {
        Line::from(vec![
            Span::styled(" ▸ ", Style::default().fg(SAFFRON)),
            Span::styled("↑↓", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(" navigate  ", Style::default().fg(Color::Rgb(60, 60, 60))),
            Span::styled("←", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(" projects", Style::default().fg(Color::Rgb(60, 60, 60))),
        ])
    } else {
        Line::from(vec![
            Span::styled("   ", Style::default()),
            Span::styled("→", Style::default().fg(TEXT_DISABLED)),
            Span::styled(
                " navigate tasks",
                Style::default().fg(Color::Rgb(50, 50, 50)),
            ),
        ])
    };
    f.render_widget(Paragraph::new(footer), chunks[1]);
}

/// Render a todo row with full width
fn render_todo_row(todo: &TuiTodo, width: usize) -> Line<'static> {
    let (icon, color) = match todo.status {
        TuiTodoStatus::Backlog => ("◌", TEXT_DISABLED),
        TuiTodoStatus::Todo => ("○", TEXT_SECONDARY),
        TuiTodoStatus::InProgress => ("◐", SAFFRON),
        TuiTodoStatus::Blocked => ("⊘", MAROON),
        TuiTodoStatus::Done => ("●", GOLD),
        TuiTodoStatus::Cancelled => ("⊗", TEXT_DISABLED),
    };

    let priority = match todo.priority {
        TuiPriority::Urgent => ("!!!", MAROON),
        TuiPriority::High => ("!! ", SAFFRON),
        TuiPriority::Medium => ("!  ", TEXT_DISABLED),
        TuiPriority::Low => ("   ", TEXT_DISABLED),
    };

    // Short ID (BOLT-1, MEM-2, etc.)
    let short_id = format!("{:<9}", todo.short_id());

    let content_width = width.saturating_sub(24); // 15 + 9 for short_id
    let content = truncate(&todo.content, content_width);

    let mut spans = vec![
        Span::styled("   ", Style::default()),
        Span::styled(format!("{} ", icon), Style::default().fg(color)),
        Span::styled(priority.0, Style::default().fg(priority.1)),
        Span::styled(short_id, Style::default().fg(TEXT_SECONDARY)),
        Span::styled(
            content,
            Style::default().fg(if todo.status == TuiTodoStatus::Done {
                TEXT_DISABLED
            } else {
                TEXT_PRIMARY
            }),
        ),
    ];

    if todo.is_overdue() {
        if let Some(label) = todo.due_label() {
            spans.push(Span::styled(
                format!(" {}", label),
                Style::default().fg(MAROON),
            ));
        }
    }

    Line::from(spans)
}

/// Render contextual action bar below selected todo - changes based on status
fn render_action_bar(todo: &TuiTodo) -> Line<'static> {
    // Very pale yellow/cream background - high readability
    let bar_bg = Color::Rgb(250, 245, 200); // Cream/pale yellow
    let bar_style = Style::default().bg(bar_bg);

    let mut spans: Vec<Span> = Vec::new();
    // Left border indicator
    spans.push(Span::styled(
        "   ┃ ",
        bar_style.fg(Color::Rgb(180, 160, 80)),
    ));

    // Very dark text for cream background - maximum contrast
    let label_color = Color::Rgb(20, 15, 5); // Near black
    let key_color = Color::Rgb(50, 20, 70); // Dark purple for keys
    let desc_color = Color::Rgb(40, 35, 20); // Very dark brown

    // Status-specific actions - all very dark for cream bg
    let dark_green = Color::Rgb(15, 50, 15);
    let dark_red = Color::Rgb(80, 20, 20);
    let dark_orange = Color::Rgb(90, 50, 10);

    match todo.status {
        TuiTodoStatus::Done => {
            spans.push(Span::styled("● Done ", bar_style.fg(dark_green)));
            spans.push(Span::styled(
                "Spc",
                bar_style.fg(key_color).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=reopen  ", bar_style.fg(desc_color)));
        }
        TuiTodoStatus::Cancelled => {
            spans.push(Span::styled("⊗ Cancelled ", bar_style.fg(label_color)));
            spans.push(Span::styled(
                "Spc",
                bar_style.fg(key_color).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=restore  ", bar_style.fg(desc_color)));
        }
        TuiTodoStatus::Blocked => {
            spans.push(Span::styled("⊘ Blocked ", bar_style.fg(dark_red)));
            spans.push(Span::styled(
                "x",
                bar_style.fg(dark_green).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=done  ", bar_style.fg(desc_color)));
            spans.push(Span::styled(
                "Spc",
                bar_style.fg(key_color).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=unblock  ", bar_style.fg(desc_color)));
        }
        TuiTodoStatus::InProgress => {
            spans.push(Span::styled("◐ Working ", bar_style.fg(dark_orange)));
            spans.push(Span::styled(
                "x",
                bar_style.fg(dark_green).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=done  ", bar_style.fg(desc_color)));
            spans.push(Span::styled(
                "Spc",
                bar_style.fg(key_color).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=pause  ", bar_style.fg(desc_color)));
        }
        TuiTodoStatus::Todo => {
            spans.push(Span::styled("○ Ready ", bar_style.fg(label_color)));
            spans.push(Span::styled(
                "x",
                bar_style.fg(dark_green).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=done  ", bar_style.fg(desc_color)));
            spans.push(Span::styled(
                "Spc",
                bar_style.fg(key_color).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=start  ", bar_style.fg(desc_color)));
        }
        TuiTodoStatus::Backlog => {
            spans.push(Span::styled("◌ Backlog ", bar_style.fg(label_color)));
            spans.push(Span::styled(
                "Spc",
                bar_style.fg(key_color).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled("=activate  ", bar_style.fg(desc_color)));
        }
    }

    // Common actions for active items only
    if todo.status != TuiTodoStatus::Done && todo.status != TuiTodoStatus::Cancelled {
        spans.push(Span::styled(
            "[]",
            bar_style.fg(key_color).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled("=move  ", bar_style.fg(desc_color)));

        // Priority shortcuts - very dark colors
        let dim = bar_style.fg(Color::Rgb(70, 60, 40));
        let (urg_style, hi_style, med_style, low_style) = match todo.priority {
            TuiPriority::Urgent => (
                bar_style
                    .fg(Color::Rgb(120, 20, 20))
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                dim,
                dim,
                dim,
            ),
            TuiPriority::High => (
                dim,
                bar_style
                    .fg(Color::Rgb(120, 60, 10))
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                dim,
                dim,
            ),
            TuiPriority::Medium => (
                dim,
                dim,
                bar_style
                    .fg(Color::Rgb(80, 70, 10))
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                dim,
            ),
            TuiPriority::Low => (
                dim,
                dim,
                dim,
                bar_style
                    .fg(Color::Rgb(20, 40, 80))
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            ),
        };
        spans.push(Span::styled("!", urg_style));
        spans.push(Span::styled("@", hi_style));
        spans.push(Span::styled("#", med_style));
        spans.push(Span::styled("$", low_style));
    }
    // Fill rest of line with background color for full-width bar effect
    spans.push(Span::styled(format!("{:>80}", " "), bar_style));

    Line::from(spans)
}

/// Render file line in sidebar (codebase integration)
fn render_file_line(
    file: &TuiFileMemory,
    width: usize,
    is_selected: bool,
    is_panel_focused: bool,
) -> Line<'static> {
    // File type icon based on language
    let icon = file.type_icon();
    let heat = file.heat_indicator();

    // Selection indicator
    let sel = if is_selected { "  ▸ " } else { "    " };
    let sel_color = if is_selected && is_panel_focused {
        SAFFRON
    } else if is_selected {
        TEXT_DISABLED
    } else {
        Color::Reset
    };
    let bg = if is_selected {
        SELECTION_BG
    } else {
        Color::Reset
    };

    // Get short path for display
    let short_path = file.short_path();
    let path_width = width.saturating_sub(20);

    // File type color
    let type_color = match file.file_type.to_lowercase().as_str() {
        "rust" => Color::Rgb(220, 160, 120), // Rust orange
        "typescript" | "javascript" => Color::Rgb(100, 180, 220), // JS blue
        "python" => Color::Rgb(160, 200, 100), // Python green
        "go" => Color::Rgb(100, 200, 220), // Go cyan
        _ => TEXT_SECONDARY,
    };

    Line::from(vec![
        Span::styled(sel, Style::default().fg(sel_color).bg(bg)),
        Span::styled(format!("{} ", icon), Style::default().fg(type_color).bg(bg)),
        Span::styled(
            truncate(&short_path, path_width),
            Style::default().fg(TEXT_SECONDARY).bg(bg),
        ),
        Span::styled(
            format!(" {}", heat),
            Style::default().fg(Color::Rgb(255, 140, 50)).bg(bg),
        ),
    ])
}

/// Empty spacer line for visual breathing room between todos
fn spacer_line() -> Line<'static> {
    Line::from(Span::raw(""))
}

// ============================================================================
// TODO DETAIL PANEL
// ============================================================================

/// Render todo detail panel with 2-column layout (Linear-inspired)
/// Left: Details | Right: Activity
fn render_todo_detail_panel(f: &mut Frame, area: Rect, state: &AppState) {
    // Top border
    let border = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(BORDER_SUBTLE));
    f.render_widget(border, area);

    // Adjust area for content (after border)
    let content_area = Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: area.height.saturating_sub(1),
    };

    let selected_todo = state.get_selected_dashboard_todo();

    match selected_todo {
        Some(todo) => {
            // Split into 2 columns: Details (50%) | Separator (1) | Activity (50%)
            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(50),
                    Constraint::Length(1), // Vertical separator
                    Constraint::Percentage(50),
                ])
                .split(content_area);

            // === LEFT COLUMN: Details ===
            render_detail_left_column(f, columns[0], &todo, state);

            // === VERTICAL SEPARATOR ===
            let separator_lines: Vec<Line> = (0..content_area.height)
                .map(|_| Line::from(Span::styled("│", Style::default().fg(BORDER_SUBTLE))))
                .collect();
            let separator = Paragraph::new(separator_lines);
            f.render_widget(separator, columns[1]);

            // === RIGHT COLUMN: Activity ===
            render_detail_right_column(f, columns[2], &todo, state);
        }
        None => {
            // No todo selected - show placeholder
            let placeholder = Paragraph::new(Line::from(vec![
                Span::styled("  ◇ ", Style::default().fg(TEXT_DISABLED)),
                Span::styled(
                    "Select a todo to view details",
                    Style::default().fg(TEXT_DISABLED),
                ),
            ]));
            f.render_widget(placeholder, content_area);
        }
    }
}

/// Render left column of detail panel (metadata + scrollable notes)
fn render_detail_left_column(f: &mut Frame, area: Rect, todo: &TuiTodo, state: &AppState) {
    let mut lines: Vec<Line> = Vec::new();
    let is_focused = state.detail_focus_column == 0;
    let line_width = (area.width as usize).saturating_sub(3);

    // Header: ID + Title
    let short_id = todo.short_id();
    let content_max = (area.width as usize).saturating_sub(short_id.len() + 5);
    let content_truncated: String = todo.content.chars().take(content_max).collect();

    lines.push(Line::from(vec![
        Span::styled(" ", Style::default()),
        Span::styled(
            short_id,
            Style::default().fg(DEEP_BLUE).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default()),
        Span::styled(
            content_truncated,
            Style::default()
                .fg(TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    // Empty line for spacing
    lines.push(Line::from(""));

    // Metadata rows with labels
    let label_width = 10;

    // Status row
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {:label_width$}", "Status"),
            Style::default().fg(TEXT_DISABLED),
        ),
        Span::styled(
            format!("{} ", todo.status.icon()),
            Style::default().fg(todo.status.color()),
        ),
        Span::styled(
            format!("{:?}", todo.status),
            Style::default().fg(TEXT_SECONDARY),
        ),
    ]));

    // Priority row
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {:label_width$}", "Priority"),
            Style::default().fg(TEXT_DISABLED),
        ),
        Span::styled(
            todo.priority.indicator(),
            Style::default().fg(todo.priority.color()),
        ),
    ]));

    // Project row
    let project_display = todo.project_name.as_deref().unwrap_or("—");
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {:label_width$}", "Project"),
            Style::default().fg(TEXT_DISABLED),
        ),
        Span::styled(project_display, Style::default().fg(SAFFRON)),
    ]));

    // Due date row
    let due_display = todo.due_label().unwrap_or_else(|| "—".to_string());
    let due_color = if todo.is_overdue() {
        MAROON
    } else {
        TEXT_SECONDARY
    };
    lines.push(Line::from(vec![
        Span::styled(
            format!(" {:label_width$}", "Due"),
            Style::default().fg(TEXT_DISABLED),
        ),
        Span::styled(due_display, Style::default().fg(due_color)),
    ]));

    // Contexts row (if any)
    if !todo.contexts.is_empty() {
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:label_width$}", "Contexts"),
                Style::default().fg(TEXT_DISABLED),
            ),
            Span::styled(todo.contexts.join(" "), Style::default().fg(DEEP_BLUE)),
        ]));
    }

    // Blocked row (if blocked)
    if let Some(ref blocked) = todo.blocked_on {
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:label_width$}", "Blocked"),
                Style::default().fg(TEXT_DISABLED),
            ),
            Span::styled(format!("⊘ {}", blocked), Style::default().fg(MAROON)),
        ]));
    }

    // Calculate available space for notes
    let metadata_lines = lines.len();
    let available_for_notes = (area.height as usize).saturating_sub(metadata_lines + 2);

    // Notes section (scrollable)
    if let Some(ref notes) = todo.notes {
        lines.push(Line::from(""));

        // Notes header with focus indicator
        let focus_indicator = if is_focused { "▶" } else { " " };
        let header_style = if is_focused {
            Style::default().fg(SAFFRON)
        } else {
            Style::default().fg(TEXT_DISABLED)
        };

        lines.push(Line::from(vec![
            Span::styled(focus_indicator, header_style),
            Span::styled("─── ", Style::default().fg(BORDER_SUBTLE)),
            Span::styled("Notes", header_style),
            Span::styled(
                if is_focused { " (↑↓ scroll)" } else { "" },
                Style::default().fg(TEXT_DISABLED),
            ),
        ]));

        // Split notes into lines for scrolling
        let mut note_lines: Vec<String> = Vec::new();
        let mut chars = notes.chars().peekable();

        while chars.peek().is_some() {
            let line_text: String = chars.by_ref().take(line_width).collect();
            if line_text.is_empty() {
                break;
            }
            note_lines.push(line_text);
        }

        let total_note_lines = note_lines.len();
        let scroll = state.notes_scroll.min(total_note_lines.saturating_sub(1));
        let visible_notes = available_for_notes.saturating_sub(1); // -1 for scroll indicator

        // Show scroll position if needed
        if total_note_lines > visible_notes {
            let scroll_info = format!(" [{}/{}]", scroll + 1, total_note_lines);
            if let Some(last) = lines.last_mut() {
                last.spans.push(Span::styled(
                    scroll_info,
                    Style::default().fg(TEXT_DISABLED),
                ));
            }
        }

        // Render visible note lines
        for (i, line_text) in note_lines
            .iter()
            .enumerate()
            .skip(scroll)
            .take(visible_notes)
        {
            let is_cursor_line = is_focused && i == scroll;
            let line_style = if is_cursor_line {
                Style::default()
                    .fg(TEXT_PRIMARY)
                    .add_modifier(Modifier::ITALIC)
            } else {
                Style::default()
                    .fg(TEXT_SECONDARY)
                    .add_modifier(Modifier::ITALIC)
            };

            lines.push(Line::from(vec![
                Span::styled(
                    if is_cursor_line { "▸" } else { " " },
                    Style::default().fg(SAFFRON),
                ),
                Span::styled(line_text.clone(), line_style),
            ]));
        }

        // Show more indicator
        if scroll + visible_notes < total_note_lines {
            lines.push(Line::from(vec![
                Span::styled(" ", Style::default()),
                Span::styled(
                    format!(
                        "  ↓ {} more lines",
                        total_note_lines - scroll - visible_notes
                    ),
                    Style::default().fg(TEXT_DISABLED),
                ),
            ]));
        }
    }

    let content = Paragraph::new(lines);
    f.render_widget(content, area);
}

/// Render right column of detail panel (scrollable activity feed)
fn render_detail_right_column(f: &mut Frame, area: Rect, todo: &TuiTodo, state: &AppState) {
    let mut lines: Vec<Line> = Vec::new();
    let is_focused = state.detail_focus_column == 1;

    // Header: ACTIVITY with focus indicator
    let focus_indicator = if is_focused { "▶" } else { " " };
    let header_style = if is_focused {
        Style::default().fg(SAFFRON)
    } else {
        Style::default().fg(TEXT_DISABLED)
    };

    lines.push(Line::from(vec![
        Span::styled(focus_indicator, header_style),
        Span::styled("ACTIVITY", header_style),
        Span::styled(
            format!(" ({})", todo.comments.len()),
            Style::default().fg(TEXT_DISABLED),
        ),
        Span::styled(
            if is_focused { " ↑↓ scroll" } else { "" },
            Style::default().fg(TEXT_DISABLED),
        ),
        Span::styled("  ", Style::default()),
        Span::styled("◉", Style::default().fg(LIVE_GREEN)),
    ]));

    lines.push(Line::from(""));

    if !todo.comments.is_empty() {
        let total_comments = todo.comments.len();
        let scroll = state.activity_scroll.min(total_comments.saturating_sub(1));

        // Calculate lines per comment (2 lines each: content + timestamp)
        let lines_per_comment = 2;
        let available_lines = (area.height as usize).saturating_sub(4);
        let comments_visible = available_lines / lines_per_comment;

        // Show scroll position if needed
        if total_comments > comments_visible {
            let scroll_info = format!(" [{}/{}]", scroll + 1, total_comments);
            if let Some(last) = lines.last_mut() {
                last.spans.push(Span::styled(
                    scroll_info,
                    Style::default().fg(TEXT_DISABLED),
                ));
            }
        }

        // Show comments from scroll position (most recent first)
        let recent_comments: Vec<_> = todo.comments.iter().rev().collect();

        for (i, comment) in recent_comments
            .iter()
            .enumerate()
            .skip(scroll)
            .take(comments_visible)
        {
            let icon = comment.comment_type.icon();
            let time_ago = format_duration_since(&comment.created_at);
            let content_max = (area.width as usize).saturating_sub(6);
            let content_preview: String = comment.content.chars().take(content_max).collect();

            let is_cursor = is_focused && i == scroll;
            let cursor_indicator = if is_cursor { "▸" } else { " " };
            let text_style = if is_cursor {
                Style::default().fg(TEXT_PRIMARY)
            } else {
                Style::default().fg(TEXT_SECONDARY)
            };

            lines.push(Line::from(vec![
                Span::styled(cursor_indicator, Style::default().fg(SAFFRON)),
                Span::styled(format!("{} ", icon), text_style),
                Span::styled(content_preview, text_style),
            ]));

            // Time on separate line
            lines.push(Line::from(vec![Span::styled(
                format!("   {}", time_ago),
                Style::default().fg(TEXT_DISABLED),
            )]));
        }

        // Show more indicator
        if scroll + comments_visible < total_comments {
            lines.push(Line::from(vec![Span::styled(
                format!("  ↓ {} more", total_comments - scroll - comments_visible),
                Style::default().fg(TEXT_DISABLED),
            )]));
        }

        // Footer: sync status
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            format!(" {} entries synced to memory", todo.comments.len()),
            Style::default()
                .fg(TEXT_DISABLED)
                .add_modifier(Modifier::DIM),
        )]));
    } else {
        lines.push(Line::from(vec![Span::styled(
            " No activity yet",
            Style::default().fg(TEXT_DISABLED),
        )]));
    }

    let content = Paragraph::new(lines);
    f.render_widget(content, area);
}

// ============================================================================
// LINEAGE CHAIN VISUALIZATION
// ============================================================================

/// Render horizontal lineage chain at bottom of Projects view
fn render_lineage_chain(f: &mut Frame, area: Rect, state: &AppState) {
    let width = area.width as usize;

    // Border line separating from main content (orange tint)
    let border_line = "─".repeat(width);

    let mut lines: Vec<Line> = Vec::new();

    // Header with border (orange/saffron tint for visibility)
    lines.push(Line::from(Span::styled(
        border_line,
        Style::default().fg(Color::Rgb(120, 80, 40)),
    )));

    if let Some(ref trace) = state.lineage_trace {
        if trace.edges.is_empty() {
            // No lineage - show placeholder (orange tint)
            lines.push(Line::from(Span::styled(
                " ◇ No causal chain detected",
                Style::default().fg(SAFFRON),
            )));
            lines.push(Line::from(Span::styled(
                "   Select a todo or memory to see its lineage",
                Style::default().fg(Color::Rgb(150, 100, 60)),
            )));
        } else {
            // Build horizontal chain visualization
            // Format: [Node] ──relation──▸ [Node] ──relation──▸ [Node]

            // Header showing direction
            let direction_label = match trace.direction.as_str() {
                "backward" => "◀── Tracing causes (why this happened)",
                "forward" => "──▸ Tracing effects (what this led to)",
                "both" => "◀── causes │ effects ──▸",
                _ => "──▸ Lineage chain",
            };
            lines.push(Line::from(vec![
                Span::styled(
                    " ⑂ LINEAGE ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(SAFFRON)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {} ", direction_label),
                    Style::default().fg(SAFFRON),
                ),
                Span::styled(
                    format!("(depth: {}, {} edges)", trace.depth, trace.edges.len()),
                    Style::default().fg(Color::Rgb(180, 140, 100)),
                ),
            ]));

            // Build the chain line
            let mut chain_spans: Vec<Span> = vec![Span::raw(" ")];

            // Calculate visible nodes based on scroll
            let visible_start = state.lineage_scroll;
            let max_visible = (width / 25).max(2); // Each node+edge takes ~25 chars

            // Get ordered path (or use edges if path is empty)
            let path: Vec<&str> = if trace.path.is_empty() {
                // Build path from edges
                let mut p = vec![trace.root_id.as_str()];
                for edge in &trace.edges {
                    if !p.contains(&edge.to_id.as_str()) {
                        p.push(&edge.to_id);
                    }
                }
                p
            } else {
                trace.path.iter().map(|s| s.as_str()).collect()
            };

            // Show scroll indicator if needed
            if visible_start > 0 {
                chain_spans.push(Span::styled("◀ ", Style::default().fg(TEXT_DISABLED)));
            }

            // Render visible portion of chain
            for (idx, node_id) in path
                .iter()
                .enumerate()
                .skip(visible_start)
                .take(max_visible)
            {
                // Get node info
                let node_info = trace.nodes.get(*node_id);
                let (type_icon, type_color, preview) = if let Some(node) = node_info {
                    (node.type_icon(), node.type_color(), &node.content_preview)
                } else {
                    ("•", Color::Gray, &node_id.to_string())
                };

                // Render node box
                let node_preview = truncate(preview, 12);
                chain_spans.push(Span::styled(
                    format!("{}", type_icon),
                    Style::default().fg(type_color).add_modifier(Modifier::BOLD),
                ));
                chain_spans.push(Span::styled(
                    format!(" {} ", node_preview),
                    Style::default().fg(Color::Rgb(240, 240, 240)),
                ));

                // Render edge to next node (if not last)
                if idx < path.len() - 1 {
                    // Find edge between this node and next
                    let next_id = path.get(idx + 1).unwrap_or(node_id);
                    let edge = trace.edges.iter().find(|e| {
                        (e.from_id == *node_id && e.to_id == *next_id)
                            || (e.to_id == *node_id && e.from_id == *next_id)
                    });

                    if let Some(e) = edge {
                        let conf_pct = (e.confidence * 100.0) as u8;
                        let source_ind = e.source_indicator();
                        chain_spans.push(Span::styled(
                            format!("─{}{}{}─▸", e.relation_icon(), conf_pct, source_ind),
                            Style::default()
                                .fg(e.relation_color())
                                .add_modifier(Modifier::BOLD),
                        ));
                    } else {
                        chain_spans.push(Span::styled(
                            "───▸",
                            Style::default().fg(Color::Rgb(120, 120, 120)),
                        ));
                    }
                }
            }

            // Show scroll indicator if more nodes
            if visible_start + max_visible < path.len() {
                chain_spans.push(Span::styled(" ▶", Style::default().fg(TEXT_DISABLED)));
            }

            lines.push(Line::from(chain_spans));

            // Footer with navigation hints
            lines.push(Line::from(vec![
                Span::styled(" ", Style::default()),
                Span::styled("<>", Style::default().fg(Color::Rgb(180, 180, 180))),
                Span::styled(" scroll  ", Style::default().fg(Color::Rgb(100, 100, 100))),
                Span::styled("L", Style::default().fg(Color::Rgb(180, 180, 180))),
                Span::styled(" trace  ", Style::default().fg(Color::Rgb(100, 100, 100))),
                Span::styled("C", Style::default().fg(Color::Rgb(100, 255, 150))),
                Span::styled(" confirm  ", Style::default().fg(Color::Rgb(100, 100, 100))),
                Span::styled("X", Style::default().fg(Color::Rgb(255, 100, 100))),
                Span::styled(" reject", Style::default().fg(Color::Rgb(100, 100, 100))),
            ]));
        }
    } else {
        // No trace loaded
        lines.push(Line::from(Span::styled(
            " ⑂ LINEAGE ",
            Style::default().fg(Color::Black).bg(Color::Rgb(60, 60, 70)),
        )));
        lines.push(Line::from(Span::styled(
            "   Select a todo and press Shift+L to trace its causal chain",
            Style::default().fg(TEXT_DISABLED),
        )));
        lines.push(Line::from(Span::styled(
            "   Discover: why did this happen? what did it lead to?",
            Style::default().fg(Color::Rgb(50, 50, 50)),
        )));
    }

    f.render_widget(Paragraph::new(lines), area);
}

/// Render a todo row with selection highlighting
fn render_todo_row_with_selection(
    todo: &TuiTodo,
    width: usize,
    is_selected: bool,
    is_panel_focused: bool,
) -> Line<'static> {
    // Fixed column widths for uniform layout
    // | sel(3) | status(2) | pri(3) | id(9) | content(flex) | project(14) | due(10) |
    const ID_COL_WIDTH: usize = 9;
    const PROJECT_COL_WIDTH: usize = 14;
    const DUE_COL_WIDTH: usize = 10;
    const FIXED_LEFT: usize = 8 + ID_COL_WIDTH; // sel(3) + status(2) + pri(3) + id(9)

    let (icon, color) = match todo.status {
        TuiTodoStatus::Backlog => ("◌", TEXT_DISABLED),
        TuiTodoStatus::Todo => ("○", TEXT_SECONDARY),
        TuiTodoStatus::InProgress => ("◐", SAFFRON),
        TuiTodoStatus::Blocked => ("⊘", MAROON),
        TuiTodoStatus::Done => ("●", GOLD),
        TuiTodoStatus::Cancelled => ("⊗", TEXT_DISABLED),
    };

    let priority = match todo.priority {
        TuiPriority::Urgent => ("!!!", MAROON),
        TuiPriority::High => ("!! ", SAFFRON),
        TuiPriority::Medium => ("!  ", TEXT_DISABLED),
        TuiPriority::Low => ("   ", TEXT_DISABLED),
    };

    // Selection indicator and background - always visible, brighter when focused
    let sel_marker = if is_selected { "▸ " } else { "   " };
    let sel_color = if is_selected && is_panel_focused {
        SAFFRON
    } else if is_selected {
        TEXT_DISABLED
    } else {
        Color::Reset
    };
    let bg = if is_selected {
        SELECTION_BG
    } else {
        Color::Reset
    };
    let text_color = if todo.status == TuiTodoStatus::Done {
        TEXT_DISABLED
    } else {
        TEXT_PRIMARY
    };

    // Short ID column (BOLT-1, MEM-2, etc.)
    let short_id = todo.short_id();
    let id_col = format!("{:<width$}", short_id, width = ID_COL_WIDTH);

    // Calculate content width (flexible column)
    let content_width = width.saturating_sub(FIXED_LEFT + PROJECT_COL_WIDTH + DUE_COL_WIDTH + 1);
    let content = if todo.content.chars().count() <= content_width {
        format!("{:<width$}", todo.content, width = content_width)
    } else {
        format!(
            "{:.<width$}",
            todo.content
                .chars()
                .take(content_width.saturating_sub(2))
                .collect::<String>(),
            width = content_width
        )
    };

    // Project column - fixed width with folder icon
    let project_col = if let Some(project) = &todo.project_name {
        let max_name = PROJECT_COL_WIDTH - 2; // folder icon + space
        let name = if project.chars().count() <= max_name {
            project.clone()
        } else {
            format!(
                "{}..",
                project.chars().take(max_name - 2).collect::<String>()
            )
        };
        format!("📁 {:<width$}", name, width = max_name)
    } else {
        " ".repeat(PROJECT_COL_WIDTH)
    };

    // Due column - fixed width
    let due_col = if todo.is_overdue() {
        if let Some(label) = todo.due_label() {
            format!("{:>width$}", label, width = DUE_COL_WIDTH)
        } else {
            " ".repeat(DUE_COL_WIDTH)
        }
    } else {
        " ".repeat(DUE_COL_WIDTH)
    };

    let spans = vec![
        Span::styled(sel_marker, Style::default().fg(sel_color).bg(bg)),
        Span::styled(format!("{} ", icon), Style::default().fg(color).bg(bg)),
        Span::styled(priority.0, Style::default().fg(priority.1).bg(bg)),
        Span::styled(id_col, Style::default().fg(TEXT_SECONDARY).bg(bg)),
        Span::styled(content, Style::default().fg(text_color).bg(bg)),
        Span::styled(project_col, Style::default().fg(GOLD).bg(bg)),
        Span::styled(due_col, Style::default().fg(MAROON).bg(bg)),
    ];

    Line::from(spans)
}

/// Render a todo row with indentation (for subtasks)
fn render_todo_row_with_indent(
    todo: &TuiTodo,
    width: usize,
    is_selected: bool,
    is_panel_focused: bool,
    indent: usize,
) -> Line<'static> {
    const ID_COL_WIDTH: usize = 9;
    const PROJECT_COL_WIDTH: usize = 14;
    const DUE_COL_WIDTH: usize = 10;
    let indent_chars = indent * 2;
    let fixed_left: usize = 8 + ID_COL_WIDTH + indent_chars;

    let (icon, color) = match todo.status {
        TuiTodoStatus::Backlog => ("◌", TEXT_DISABLED),
        TuiTodoStatus::Todo => ("○", TEXT_SECONDARY),
        TuiTodoStatus::InProgress => ("◐", SAFFRON),
        TuiTodoStatus::Blocked => ("⊘", MAROON),
        TuiTodoStatus::Done => ("●", GOLD),
        TuiTodoStatus::Cancelled => ("⊗", TEXT_DISABLED),
    };

    let priority = match todo.priority {
        TuiPriority::Urgent => ("!!!", MAROON),
        TuiPriority::High => ("!! ", SAFFRON),
        TuiPriority::Medium => ("!  ", TEXT_DISABLED),
        TuiPriority::Low => ("   ", TEXT_DISABLED),
    };

    let indent_str = "  ".repeat(indent);
    let sel_marker = if is_selected {
        format!("{}▸ ", indent_str)
    } else {
        format!("{}  ", indent_str)
    };
    let sel_color = if is_selected && is_panel_focused {
        SAFFRON
    } else if is_selected {
        TEXT_DISABLED
    } else {
        Color::Reset
    };
    let bg = if is_selected {
        SELECTION_BG
    } else {
        Color::Reset
    };
    let text_color = if todo.status == TuiTodoStatus::Done {
        TEXT_DISABLED
    } else {
        TEXT_PRIMARY
    };

    // Short ID column (BOLT-1, MEM-2, etc.)
    let short_id = todo.short_id();
    let id_col = format!("{:<width$}", short_id, width = ID_COL_WIDTH);

    let content_width = width.saturating_sub(fixed_left + PROJECT_COL_WIDTH + DUE_COL_WIDTH + 1);
    let content = if todo.content.chars().count() <= content_width {
        format!("{:<width$}", todo.content, width = content_width)
    } else {
        format!(
            "{:.<width$}",
            todo.content
                .chars()
                .take(content_width.saturating_sub(2))
                .collect::<String>(),
            width = content_width
        )
    };

    let project_col = if let Some(project) = &todo.project_name {
        let max_name = PROJECT_COL_WIDTH - 2;
        let name = if project.chars().count() <= max_name {
            project.clone()
        } else {
            format!(
                "{}..",
                project.chars().take(max_name - 2).collect::<String>()
            )
        };
        format!("📁 {:<width$}", name, width = max_name)
    } else {
        " ".repeat(PROJECT_COL_WIDTH)
    };

    let due_col = if todo.is_overdue() {
        if let Some(label) = todo.due_label() {
            format!("{:>width$}", label, width = DUE_COL_WIDTH)
        } else {
            " ".repeat(DUE_COL_WIDTH)
        }
    } else {
        " ".repeat(DUE_COL_WIDTH)
    };

    let spans = vec![
        Span::styled(sel_marker, Style::default().fg(sel_color).bg(bg)),
        Span::styled(format!("{} ", icon), Style::default().fg(color).bg(bg)),
        Span::styled(priority.0, Style::default().fg(priority.1).bg(bg)),
        Span::styled(id_col, Style::default().fg(TEXT_SECONDARY).bg(bg)),
        Span::styled(content, Style::default().fg(text_color).bg(bg)),
        Span::styled(project_col, Style::default().fg(GOLD).bg(bg)),
        Span::styled(due_col, Style::default().fg(MAROON).bg(bg)),
    ];

    Line::from(spans)
}

/// Render a project line with indentation level (0 = root, 1 = sub-project)
fn render_project_line(
    lines: &mut Vec<Line<'static>>,
    project: &TuiProject,
    state: &AppState,
    mut flat_idx: usize,
    width: usize,
    is_left_focused: bool,
    indent_level: usize,
) -> usize {
    let is_selected = state.projects_selected == flat_idx;
    let is_expanded = state.is_project_expanded(&project.id);
    let todos = state.todos_for_project(&project.id);
    let done = todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::Done)
        .count();
    let active = todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::InProgress)
        .count();
    let remaining = todos
        .iter()
        .filter(|t| t.status != TuiTodoStatus::Done && t.status != TuiTodoStatus::Cancelled)
        .count();
    let total = todos.len();

    // Folder icon: 📂 open, 📁 closed; sub-projects use different icons
    let folder = if indent_level > 0 {
        if is_expanded {
            "📂"
        } else {
            "📁"
        }
    } else {
        if is_expanded {
            "📂"
        } else {
            "📁"
        }
    };

    // Codebase status indicator with animation for scanning
    let spinner_frames = ["◜", "◠", "◝", "◞", "◡", "◟"];
    let (codebase_icon, codebase_color, codebase_hint) = if indent_level > 0 {
        ("", Color::Reset, "") // Sub-projects cannot be indexed
    } else if state.is_scanning(&project.id) {
        let frame = spinner_frames[(state.animation_tick as usize / 2) % spinner_frames.len()];
        (frame, SAFFRON, " scanning...")
    } else if state.is_project_indexed(&project.id) {
        ("●", Color::Rgb(100, 200, 100), " [f]") // Green - indexed, hint to press f
    } else {
        ("●", Color::Rgb(200, 80, 80), " [S]") // Red - not indexed, hint to press S
    };

    // Indentation for sub-projects
    let indent_str = "   ".repeat(indent_level);

    // Cursor always visible - brighter when focused
    let sel = if is_selected { "▸ " } else { "  " };
    let sel_color = if is_selected && is_left_focused {
        SAFFRON
    } else if is_selected {
        TEXT_DISABLED
    } else {
        Color::Reset
    };
    let name_width = width.saturating_sub(32 + indent_level * 3); // Extra space for codebase icon
    let name = truncate(&project.name, name_width);

    // Progress percentage
    let pct = if total > 0 { (done * 100) / total } else { 0 };
    let progress_color = if pct == 100 {
        GOLD
    } else if active > 0 {
        SAFFRON
    } else {
        TEXT_DISABLED
    };
    let bg = if is_selected {
        SELECTION_BG
    } else {
        Color::Reset
    };

    // Format: "3 left · 75%"  or "✓ done" if complete
    let status_str = if total == 0 {
        "empty".to_string()
    } else if pct == 100 {
        "✓ done".to_string()
    } else {
        format!("{} left · {}%", remaining, pct)
    };

    lines.push(Line::from(vec![
        Span::styled(sel, Style::default().fg(sel_color).bg(bg)),
        Span::styled(indent_str.clone(), Style::default().bg(bg)),
        Span::styled(format!("{} ", folder), Style::default().bg(bg)),
        Span::styled(
            format!("{:<w$}", name, w = name_width),
            Style::default().fg(TEXT_PRIMARY).bg(bg),
        ),
        Span::styled(
            format!(" {}", codebase_icon),
            Style::default().fg(codebase_color).bg(bg),
        ),
        Span::styled(
            codebase_hint,
            Style::default().fg(TEXT_DISABLED).bg(bg),
        ),
        Span::styled(" ", Style::default().bg(bg)),
        Span::styled(
            format!("{:<10}", status_str),
            Style::default().fg(progress_color).bg(bg),
        ),
    ]));
    flat_idx += 1;

    // Expanded todos with indentation - each one is navigable
    if is_expanded {
        let max_todos = if state.expand_sections {
            todos.len()
        } else {
            5
        };
        for todo in todos.iter().take(max_todos) {
            let todo_selected = state.projects_selected == flat_idx;
            let is_selected_and_focused = todo_selected && is_left_focused;
            lines.push(render_sidebar_todo(
                todo,
                width,
                todo_selected,
                is_left_focused,
            ));
            if is_selected_and_focused {
                lines.push(render_action_bar(todo));
            }
            flat_idx += 1;
        }
        if todos.len() > max_todos {
            lines.push(Line::from(Span::styled(
                format!(
                    "       +{} more (press e to expand)",
                    todos.len() - max_todos
                ),
                Style::default().fg(TEXT_DISABLED),
            )));
        }
        lines.push(Line::from("")); // space after expanded project
    }

    flat_idx
}

/// Render todo under expanded project in sidebar (with selection support)
fn render_sidebar_todo(
    todo: &TuiTodo,
    width: usize,
    is_selected: bool,
    is_panel_focused: bool,
) -> Line<'static> {
    let (icon, color) = match todo.status {
        TuiTodoStatus::InProgress => ("◐", SAFFRON),
        TuiTodoStatus::Todo => ("○", TEXT_SECONDARY),
        TuiTodoStatus::Done => ("●", GOLD),
        TuiTodoStatus::Blocked => ("⊘", MAROON),
        _ => ("○", TEXT_DISABLED),
    };

    // Selection indicator - always visible, brighter when focused
    let sel = if is_selected { "  ▸ " } else { "    " };
    let sel_color = if is_selected && is_panel_focused {
        SAFFRON
    } else if is_selected {
        TEXT_DISABLED
    } else {
        Color::Reset
    };
    let bg = if is_selected {
        SELECTION_BG
    } else {
        Color::Reset
    };

    // Short ID (BOLT-1, MEM-2, etc.)
    let short_id = format!("{:<8}", todo.short_id());

    let content_width = width.saturating_sub(20); // 12 + 8 for short_id

    Line::from(vec![
        Span::styled(sel, Style::default().fg(sel_color).bg(bg)),
        Span::styled(format!("{} ", icon), Style::default().fg(color).bg(bg)),
        Span::styled(short_id, Style::default().fg(TEXT_DISABLED).bg(bg)),
        Span::styled(
            truncate(&todo.content, content_width),
            Style::default().fg(TEXT_SECONDARY).bg(bg),
        ),
    ])
}

/// Compact stats panel for dashboard
fn render_compact_stats(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " Stats ",
            Style::default().fg(Color::Rgb(255, 215, 0)),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let stats_line1 = Line::from(vec![
        Span::styled("Memories: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", state.total_memories),
            Style::default().fg(Color::White),
        ),
        Span::styled("  Recalls: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", state.total_recalls),
            Style::default().fg(Color::White),
        ),
    ]);

    let stats_line2 = Line::from(vec![
        Span::styled("W:", Style::default().fg(Color::Yellow)),
        Span::styled(
            format!("{} ", state.tier_stats.working),
            Style::default().fg(Color::White),
        ),
        Span::styled("S:", Style::default().fg(Color::Rgb(255, 200, 150))),
        Span::styled(
            format!("{} ", state.tier_stats.session),
            Style::default().fg(Color::White),
        ),
        Span::styled("L:", Style::default().fg(Color::Rgb(180, 230, 180))),
        Span::styled(
            format!("{}", state.tier_stats.long_term),
            Style::default().fg(Color::White),
        ),
        Span::styled("  Nodes:", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", state.graph_stats.nodes),
            Style::default().fg(Color::Magenta),
        ),
    ]);

    let text = vec![stats_line1, stats_line2];
    let paragraph = Paragraph::new(text);
    f.render_widget(paragraph, inner);
}

fn render_stats_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // Tiers
            Constraint::Length(9), // Types
            Constraint::Length(6), // Graph
            Constraint::Length(5), // Retrieval
            Constraint::Min(0),    // Entities
        ])
        .split(inner);

    // TIERS
    let tier_total = state.tier_stats.total().max(1);
    let tier_lines = vec![
        Line::from(Span::styled(
            " TIERS",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("  Working   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar(state.tier_stats.working, tier_total, 8),
                Style::default().fg(Color::Rgb(180, 230, 180)),
            ),
            Span::styled(
                format!(" {}", state.tier_stats.working),
                Style::default().fg(Color::Rgb(180, 230, 180)),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Session   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar(state.tier_stats.session, tier_total, 8),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(
                format!(" {}", state.tier_stats.session),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Long-term ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar(state.tier_stats.long_term, tier_total, 8),
                Style::default().fg(Color::Magenta),
            ),
            Span::styled(
                format!(" {}", state.tier_stats.long_term),
                Style::default().fg(Color::Magenta),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(tier_lines), chunks[0]);

    // TYPES
    let type_total = state.type_stats.total().max(1);
    let mut type_lines = vec![Line::from(Span::styled(
        " TYPES",
        Style::default()
            .fg(Color::Rgb(255, 200, 150))
            .add_modifier(Modifier::BOLD),
    ))];
    for (name, count, color) in state.type_stats.as_vec() {
        if count > 0 {
            type_lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:11}", name),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    progress_bar(count, type_total, 6),
                    Style::default().fg(color),
                ),
                Span::styled(format!(" {}", count), Style::default().fg(color)),
            ]));
        }
    }
    f.render_widget(Paragraph::new(type_lines), chunks[1]);

    // GRAPH
    let graph_lines = vec![
        Line::from(Span::styled(
            " GRAPH",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("  Nodes: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", state.graph_stats.nodes),
                Style::default().fg(Color::White),
            ),
            Span::styled("  Edges: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", state.graph_stats.edges),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Density: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar((state.graph_stats.density * 100.0) as u32, 100, 8),
                Style::default().fg(Color::Rgb(255, 200, 150)),
            ),
            Span::styled(
                format!(" {:.2}", state.graph_stats.density),
                Style::default().fg(Color::Rgb(255, 200, 150)),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Avg wt: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.2}", state.graph_stats.avg_weight),
                Style::default().fg(Color::White),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(graph_lines), chunks[2]);

    // RETRIEVAL
    let ret_total = state.retrieval_stats.total().max(1);
    let ret_lines = vec![
        Line::from(Span::styled(
            " RETRIEVAL",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("  semantic ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar(state.retrieval_stats.semantic, ret_total, 5),
                Style::default().fg(Color::Rgb(255, 200, 150)),
            ),
            Span::styled(
                format!(" {}", state.retrieval_stats.semantic),
                Style::default().fg(Color::Rgb(255, 200, 150)),
            ),
        ]),
        Line::from(vec![
            Span::styled("  assoc.   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar(state.retrieval_stats.associative, ret_total, 5),
                Style::default().fg(Color::Magenta),
            ),
            Span::styled(
                format!(" {}", state.retrieval_stats.associative),
                Style::default().fg(Color::Magenta),
            ),
        ]),
        Line::from(vec![
            Span::styled("  hybrid   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar(state.retrieval_stats.hybrid, ret_total, 5),
                Style::default().fg(Color::Rgb(180, 230, 180)),
            ),
            Span::styled(
                format!(" {}", state.retrieval_stats.hybrid),
                Style::default().fg(Color::Rgb(180, 230, 180)),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(ret_lines), chunks[3]);

    // TOP ENTITIES
    let mut entity_lines = vec![Line::from(Span::styled(
        " ENTITIES",
        Style::default()
            .fg(Color::Rgb(255, 200, 150))
            .add_modifier(Modifier::BOLD),
    ))];
    for (name, count) in state.entity_stats.top_entities.iter().take(5) {
        entity_lines.push(Line::from(vec![
            Span::styled(
                format!("  {} ", truncate(name, 12)),
                Style::default().fg(Color::White),
            ),
            Span::styled(format!("({})", count), Style::default().fg(Color::DarkGray)),
        ]));
    }
    if state.entity_stats.top_entities.is_empty() {
        entity_lines.push(Line::from(Span::styled(
            "  (none)",
            Style::default().fg(Color::DarkGray),
        )));
    }
    f.render_widget(Paragraph::new(entity_lines), chunks[4]);
}

fn render_todos_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let width = area.width as usize;
    let is_focused = state.focus_panel == FocusPanel::Left;

    // Active todos count (non-done, non-cancelled)
    let active_count = state
        .todos
        .iter()
        .filter(|t| t.status != TuiTodoStatus::Done && t.status != TuiTodoStatus::Cancelled)
        .count();

    // Focus indicator in title
    let focus_indicator = if is_focused { "▸ " } else { "  " };
    let title_color = if is_focused {
        SAFFRON
    } else {
        Color::Rgb(255, 215, 0)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if is_focused { SAFFRON } else { Color::Black }))
        .title(Span::styled(
            format!("{} TODO ({}) ", focus_indicator, active_count),
            Style::default()
                .fg(title_color)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.todos.is_empty() {
        let empty_msg = Paragraph::new("No active todos")
            .style(Style::default().fg(TEXT_DISABLED))
            .alignment(Alignment::Center);
        f.render_widget(empty_msg, inner);
        return;
    }

    // Group todos by status for Linear-style display
    let mut lines: Vec<Line> = Vec::new();

    // Calculate available height for todos (leave room for stats summary)
    let available_lines = inner.height.saturating_sub(3) as usize;
    let mut used_lines = 0;
    let mut flat_idx: usize = 0;

    // In Progress section (priority - show more)
    let in_progress: Vec<_> = state
        .todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::InProgress)
        .collect();
    if !in_progress.is_empty() && used_lines < available_lines {
        lines.push(Line::from(Span::styled(
            format!(" ◐ In Progress ({})", in_progress.len()),
            Style::default().fg(SAFFRON).add_modifier(Modifier::BOLD),
        )));
        used_lines += 1;
        let show_count = if state.expand_sections {
            (available_lines - used_lines).min(in_progress.len())
        } else {
            (available_lines - used_lines).min(in_progress.len()).min(4)
        };
        for todo in in_progress.iter().take(show_count) {
            let is_selected = state.selected_todo == flat_idx && is_focused;
            lines.push(render_dashboard_todo_line(
                todo,
                width,
                is_selected,
                is_focused,
            ));
            used_lines += 1;
            if is_selected && used_lines < available_lines {
                lines.push(render_action_bar(todo));
                used_lines += 1;
            }
            flat_idx += 1;
        }
        if in_progress.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!(
                    "    +{} more (press e to expand)",
                    in_progress.len() - show_count
                ),
                Style::default().fg(TEXT_DISABLED),
            )));
            used_lines += 1;
        }
    }

    // Todo section
    let todos: Vec<_> = state
        .todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::Todo)
        .collect();
    if !todos.is_empty() && used_lines < available_lines {
        lines.push(Line::from(Span::styled(
            format!(" ○ Todo ({})", todos.len()),
            Style::default()
                .fg(TEXT_SECONDARY)
                .add_modifier(Modifier::BOLD),
        )));
        used_lines += 1;
        let show_count = if state.expand_sections {
            (available_lines - used_lines).min(todos.len())
        } else {
            (available_lines - used_lines).min(todos.len()).min(4)
        };
        for todo in todos.iter().take(show_count) {
            let is_selected = state.selected_todo == flat_idx && is_focused;
            lines.push(render_dashboard_todo_line(
                todo,
                width,
                is_selected,
                is_focused,
            ));
            used_lines += 1;
            if is_selected && used_lines < available_lines {
                lines.push(render_action_bar(todo));
                used_lines += 1;
            }
            flat_idx += 1;
        }
        if todos.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!("    +{} more (press e to expand)", todos.len() - show_count),
                Style::default().fg(TEXT_DISABLED),
            )));
            used_lines += 1;
        }
    }

    // Blocked section
    let blocked: Vec<_> = state
        .todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::Blocked)
        .collect();
    if !blocked.is_empty() && used_lines < available_lines {
        lines.push(Line::from(Span::styled(
            format!(" ⊘ Blocked ({})", blocked.len()),
            Style::default().fg(MAROON).add_modifier(Modifier::BOLD),
        )));
        used_lines += 1;
        let show_count = if state.expand_sections {
            (available_lines - used_lines).min(blocked.len())
        } else {
            (available_lines - used_lines).min(blocked.len()).min(3)
        };
        for todo in blocked.iter().take(show_count) {
            let is_selected = state.selected_todo == flat_idx && is_focused;
            lines.push(render_dashboard_todo_line(
                todo,
                width,
                is_selected,
                is_focused,
            ));
            used_lines += 1;
            if is_selected && used_lines < available_lines {
                lines.push(render_action_bar(todo));
                used_lines += 1;
            }
            flat_idx += 1;
        }
        if blocked.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!(
                    "    +{} more (press e to expand)",
                    blocked.len() - show_count
                ),
                Style::default().fg(TEXT_DISABLED),
            )));
            used_lines += 1;
        }
    }

    // Backlog section
    let backlog: Vec<_> = state
        .todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::Backlog)
        .collect();
    if !backlog.is_empty() && used_lines < available_lines {
        lines.push(Line::from(Span::styled(
            format!(" ◌ Backlog ({})", backlog.len()),
            Style::default()
                .fg(TEXT_DISABLED)
                .add_modifier(Modifier::BOLD),
        )));
        used_lines += 1;
        let show_count = if state.expand_sections {
            (available_lines - used_lines).min(backlog.len())
        } else {
            (available_lines - used_lines).min(backlog.len()).min(2)
        };
        for todo in backlog.iter().take(show_count) {
            let is_selected = state.selected_todo == flat_idx && is_focused;
            lines.push(render_dashboard_todo_line(
                todo,
                width,
                is_selected,
                is_focused,
            ));
            used_lines += 1;
            if is_selected && used_lines < available_lines {
                lines.push(render_action_bar(todo));
                used_lines += 1;
            }
            flat_idx += 1;
        }
        if backlog.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!(
                    "    +{} more (press e to expand)",
                    backlog.len() - show_count
                ),
                Style::default().fg(TEXT_DISABLED),
            )));
            used_lines += 1;
        }
    }

    // Done section (show recent completions) - always show if there are done items
    let done: Vec<_> = state
        .todos
        .iter()
        .filter(|t| t.status == TuiTodoStatus::Done)
        .collect();
    if !done.is_empty() {
        lines.push(Line::from(Span::styled(
            format!(" ● Done ({})", done.len()),
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
        )));
        // Show up to 5 done items, using remaining space
        let remaining = available_lines.saturating_sub(used_lines + 3);
        let show_count = if state.expand_sections {
            remaining.max(2).min(done.len())
        } else {
            remaining.max(2).min(done.len()).min(5)
        };
        for todo in done.iter().take(show_count) {
            let is_selected = state.selected_todo == flat_idx && is_focused;
            lines.push(render_dashboard_todo_line(
                todo,
                width,
                is_selected,
                is_focused,
            ));
            if is_selected {
                lines.push(render_action_bar(todo));
            }
            flat_idx += 1;
        }
        if done.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!("    +{} more (press e to expand)", done.len() - show_count),
                Style::default().fg(TEXT_DISABLED),
            )));
        }
    }

    // Stats summary at bottom
    lines.push(Line::from(vec![Span::styled(
        "─".repeat(inner.width as usize - 2),
        Style::default().fg(BORDER_DIVIDER),
    )]));
    lines.push(Line::from(vec![
        Span::styled(
            format!(" ◐ {} ", state.todo_stats.in_progress),
            Style::default().fg(SAFFRON),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            format!("○ {} ", state.todo_stats.todo),
            Style::default().fg(TEXT_PRIMARY),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            format!("⊘ {} ", state.todo_stats.blocked),
            Style::default().fg(MAROON),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            format!("● {}", state.todo_stats.done),
            Style::default().fg(GOLD),
        ),
        if state.todo_stats.overdue > 0 {
            Span::styled(
                format!(" ⚠ {}", state.todo_stats.overdue),
                Style::default().fg(MAROON),
            )
        } else {
            Span::raw("")
        },
    ]));

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, inner);
}

/// Render a todo line for Dashboard with selection support
fn render_dashboard_todo_line(
    todo: &TuiTodo,
    width: usize,
    is_selected: bool,
    is_panel_focused: bool,
) -> Line<'static> {
    // Fixed column widths for uniform layout
    // | sel(3) | status(2) | pri(3) | id(9) | content(flex) | project(14) | due(10) |
    const ID_COL_WIDTH: usize = 9;
    const PROJECT_COL_WIDTH: usize = 14;
    const DUE_COL_WIDTH: usize = 10;
    const FIXED_LEFT: usize = 8 + ID_COL_WIDTH; // sel(3) + status(2) + pri(3) + id(9)

    let (icon, color) = match todo.status {
        TuiTodoStatus::Backlog => ("◌", TEXT_DISABLED),
        TuiTodoStatus::Todo => ("○", TEXT_SECONDARY),
        TuiTodoStatus::InProgress => ("◐", SAFFRON),
        TuiTodoStatus::Blocked => ("⊘", MAROON),
        TuiTodoStatus::Done => ("●", GOLD),
        TuiTodoStatus::Cancelled => ("⊗", TEXT_DISABLED),
    };

    let priority = match todo.priority {
        TuiPriority::Urgent => ("!!!", MAROON),
        TuiPriority::High => ("!! ", SAFFRON),
        TuiPriority::Medium => ("!  ", TEXT_DISABLED),
        TuiPriority::Low => ("   ", TEXT_DISABLED),
    };

    // Selection styling
    let sel_marker = if is_selected { "▸ " } else { "   " };
    let sel_color = if is_selected && is_panel_focused {
        SAFFRON
    } else if is_selected {
        TEXT_DISABLED
    } else {
        Color::Reset
    };
    let bg = if is_selected {
        SELECTION_BG
    } else {
        Color::Reset
    };
    let text_color = if todo.status == TuiTodoStatus::Done {
        TEXT_DISABLED
    } else {
        TEXT_PRIMARY
    };

    // Short ID column (BOLT-1, MEM-2, etc.)
    let short_id = todo.short_id();
    let id_col = format!("{:<width$}", short_id, width = ID_COL_WIDTH);

    // Calculate content width (flexible column)
    let content_width = width.saturating_sub(FIXED_LEFT + PROJECT_COL_WIDTH + DUE_COL_WIDTH + 1);
    let content = if todo.content.chars().count() <= content_width {
        // Pad content to fill the column
        format!("{:<width$}", todo.content, width = content_width)
    } else {
        // Truncate with ellipsis
        format!(
            "{:.<width$}",
            todo.content
                .chars()
                .take(content_width.saturating_sub(2))
                .collect::<String>(),
            width = content_width
        )
    };

    // Project column - fixed width with folder icon
    let project_col = if let Some(project) = &todo.project_name {
        let max_name = PROJECT_COL_WIDTH - 2; // folder icon + space
        let name = if project.chars().count() <= max_name {
            project.clone()
        } else {
            format!(
                "{}..",
                project.chars().take(max_name - 2).collect::<String>()
            )
        };
        format!("📁 {:<width$}", name, width = max_name)
    } else {
        " ".repeat(PROJECT_COL_WIDTH)
    };

    // Due column - fixed width
    let due_col = if todo.is_overdue() {
        if let Some(label) = todo.due_label() {
            format!("{:>width$}", label, width = DUE_COL_WIDTH)
        } else {
            " ".repeat(DUE_COL_WIDTH)
        }
    } else {
        " ".repeat(DUE_COL_WIDTH)
    };

    let spans = vec![
        Span::styled(sel_marker, Style::default().fg(sel_color).bg(bg)),
        Span::styled(format!("{} ", icon), Style::default().fg(color).bg(bg)),
        Span::styled(priority.0, Style::default().fg(priority.1).bg(bg)),
        Span::styled(id_col, Style::default().fg(TEXT_SECONDARY).bg(bg)),
        Span::styled(content, Style::default().fg(text_color).bg(bg)),
        Span::styled(project_col, Style::default().fg(GOLD).bg(bg)),
        Span::styled(due_col, Style::default().fg(MAROON).bg(bg)),
    ];

    Line::from(spans)
}
fn render_todo_line(todo: &TuiTodo) -> Line<'static> {
    let mut spans = vec![
        Span::styled("  ", Style::default()),
        Span::styled(todo.status.icon(), Style::default().fg(todo.status.color())),
        Span::styled(" ", Style::default()),
        Span::styled(
            todo.priority.indicator(),
            Style::default().fg(todo.priority.color()),
        ),
    ];

    // Short ID (BOLT-1, MEM-2, etc.)
    spans.push(Span::styled(
        format!("{:<9}", todo.short_id()),
        Style::default().fg(TEXT_SECONDARY),
    ));

    // Content (truncated based on whether we have project name)
    let has_project = todo.project_name.is_some();
    let max_content_len = if has_project { 14 } else { 24 }; // Reduced to make room for ID
    let content = if todo.content.len() > max_content_len {
        format!("{}…", &todo.content[..max_content_len])
    } else {
        todo.content.clone()
    };
    spans.push(Span::styled(content, Style::default().fg(Color::White)));

    // Project name (if any)
    if let Some(project) = &todo.project_name {
        let short_project = if project.len() > 12 {
            format!("{}…", &project[..12])
        } else {
            project.clone()
        };
        spans.push(Span::styled(
            format!("  {}", short_project),
            Style::default().fg(Color::Rgb(180, 180, 180)),
        ));
    }

    // Due date or blocked indicator
    if todo.is_overdue() {
        if let Some(label) = todo.due_label() {
            spans.push(Span::styled(
                format!(" {}", label),
                Style::default().fg(Color::Red),
            ));
        }
    } else if let Some(blocked) = &todo.blocked_on {
        let short_blocked = if blocked.len() > 12 {
            format!("{}…", &blocked[..12])
        } else {
            blocked.clone()
        };
        spans.push(Span::styled(
            format!(" @{}", short_blocked),
            Style::default().fg(Color::DarkGray),
        ));
    }

    Line::from(spans)
}

fn render_activity_feed(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " Activity ",
            Style::default()
                .fg(Color::Rgb(255, 215, 0))
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                " j/k Enter ",
                Style::default().fg(Color::DarkGray),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.events.is_empty() {
        let msg = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Waiting for events...",
                Style::default().fg(Color::DarkGray),
            )),
        ]);
        f.render_widget(msg, inner);
        return;
    }

    let has_rich_selected = state.selected_event.and_then(|i| state.events.get(i)).map_or(false, |e| e.event.results.is_some());
    let detail_height = if state.selected_event.is_some() {
        if has_rich_selected {
            (inner.height * 3 / 4).max(15)
        } else {
            12u16
        }
    } else {
        0u16
    };
    let list_height = inner.height.saturating_sub(detail_height);

    // Timeline: 3 lines per event
    let event_height = 3u16;
    let max_events = (list_height / event_height).max(1) as usize;

    // Calculate scroll position based on selection
    let scroll_start = if let Some(sel) = state.selected_event {
        if sel < state.scroll_offset {
            sel
        } else if sel >= state.scroll_offset + max_events {
            sel.saturating_sub(max_events.saturating_sub(1))
        } else {
            state.scroll_offset
        }
    } else {
        state.scroll_offset
    };

    let end = (scroll_start + max_events).min(state.events.len());
    let content_width = inner.width.saturating_sub(4) as usize; // Leave room for timeline

    let mut y_offset = 0u16;
    for (global_idx, event) in state
        .events
        .iter()
        .enumerate()
        .skip(scroll_start)
        .take(end.saturating_sub(scroll_start))
    {
        if y_offset + event_height > list_height {
            break;
        }

        let is_selected = state.selected_event == Some(global_idx);
        let is_newest = global_idx == 0;

        // Glow effect for new events - smooth fade over 2 seconds
        let glow = event.glow(); // 0.0 to 1.0, fades over 2 seconds
        let is_history = event.event.event_type == "HISTORY";
        let is_glowing = !is_history && glow > 0.0;

        // Dynamic color based on glow intensity
        let base_color = event.event.event_color();
        let color = if is_glowing {
            // Blend towards bright yellow/white based on glow
            let (r, g, b) = color_to_rgb(base_color);
            let glow_boost = glow * 0.5;
            Color::Rgb(
                (r as f32 + (255.0 - r as f32) * glow_boost).min(255.0) as u8,
                (g as f32 + (255.0 - g as f32) * glow_boost).min(255.0) as u8,
                (b as f32 + (100.0 - b as f32) * glow_boost).min(255.0) as u8,
            )
        } else {
            base_color
        };
        let icon = event.event.event_icon();

        // Timeline node: animated for glowing events
        let node = if is_glowing && glow > 0.7 {
            "◉" // Bright filled
        } else if is_glowing && glow > 0.3 {
            "●" // Filled
        } else if is_newest {
            "★" // Star for newest
        } else {
            "○" // Empty
        };

        let node_color = if is_glowing {
            // Pulsing glow color
            let pulse = (state.animation_tick as f32 * 0.3).sin() * 0.2 + 0.8;
            let intensity = glow * pulse;
            Color::Rgb(
                (255.0 * intensity).min(255.0) as u8,
                (200.0 * intensity).min(255.0) as u8,
                (50.0 * intensity).min(255.0) as u8,
            )
        } else if is_newest {
            Color::Rgb(180, 230, 180) // Pastel green
        } else if is_selected {
            Color::Rgb(255, 200, 150) // Pastel orange
        } else {
            Color::DarkGray
        };

        let bg = if is_selected {
            state.theme.selection_bg()
        } else {
            state.theme.bg()
        };

        // Line 1: Timeline node + event type + time
        let time_str = event.time_ago();
        let type_width = 10;
        let padding = content_width.saturating_sub(type_width + time_str.len() + 2);
        let line1 = Line::from(vec![
            Span::styled(
                node,
                Style::default().fg(node_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled("─", Style::default().fg(Color::DarkGray)),
            Span::styled(
                icon,
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {:8}", event.event.event_type),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:>width$}", time_str, width = padding + time_str.len()),
                Style::default().fg(Color::DarkGray),
            ),
        ]);

        // Line 2: Content with glow effect
        let preview = event
            .event
            .content_preview
            .as_ref()
            .map(|s| truncate(s, content_width))
            .unwrap_or_default();
        // Content brightness based on glow
        let content_color = if is_glowing {
            let brightness = 180.0 + glow * 75.0; // 180 to 255
            Color::Rgb(brightness as u8, brightness as u8, (brightness * 0.9) as u8)
        } else if is_selected {
            state.theme.fg()
        } else {
            state.theme.fg_dim()
        };
        let connector_color = if is_glowing {
            Color::Rgb(
                (255.0 * glow) as u8,
                (200.0 * glow) as u8,
                (50.0 * glow) as u8,
            )
        } else {
            Color::DarkGray
        };
        let line2 = Line::from(vec![
            Span::styled("│ ", Style::default().fg(connector_color)),
            Span::styled(preview, Style::default().fg(content_color).bg(bg)),
        ]);

        // Line 3: Metadata + spacing
        let mut meta_spans = vec![Span::styled("│ ", Style::default().fg(Color::DarkGray))];
        if let Some(t) = &event.event.memory_type {
            meta_spans.push(Span::styled(
                format!("[{}]", t),
                Style::default().fg(Color::Rgb(255, 200, 150)), // Pastel orange
            ));
            meta_spans.push(Span::raw(" "));
        }
        if let Some(entities) = &event.event.entities {
            if !entities.is_empty() {
                let tags = entities
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                meta_spans.push(Span::styled(
                    tags,
                    Style::default().fg(Color::Rgb(180, 230, 180)),
                )); // Pastel green
            }
        }
        let line3 = Line::from(meta_spans);

        // Render all 3 lines
        let event_area = Rect {
            x: inner.x,
            y: inner.y + y_offset,
            width: inner.width,
            height: event_height,
        };

        // Background for selected
        if is_selected {
            f.render_widget(Block::default().style(Style::default().bg(bg)), event_area);
        }

        f.render_widget(Paragraph::new(vec![line1, line2, line3]), event_area);

        y_offset += event_height;
    }

    if let Some(sel_idx) = state.selected_event {
        if let Some(event) = state.events.get(sel_idx) {
            let detail_area = Rect {
                x: inner.x,
                y: inner.y + list_height,
                width: inner.width,
                height: detail_height,
            };
            render_event_detail(f, detail_area, event, state);
        }
    }

    if state.events.len() > max_events {
        let scrollbar = Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state = ScrollbarState::new(state.events.len()).position(scroll_start);
        f.render_stateful_widget(
            scrollbar,
            area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut scrollbar_state,
        );
    }
}

fn render_event_card(f: &mut Frame, area: Rect, event: &DisplayEvent, _index: usize, tick: u64) {
    let color = event.event.event_color();
    let icon = event.event.event_icon();
    let label = &event.event.event_type;

    // Calculate glow intensity for smooth fade effect
    let glow = event.glow(); // 0.0 to 1.0, fades over 2 seconds
    let is_new = glow > 0.0;

    // Dynamic border color with glow effect
    let border_color = if is_new {
        // Glow from bright event color to dim
        let (r, g, b) = color_to_rgb(color);
        let intensity = 0.4 + glow * 0.6; // 0.4 to 1.0
        Color::Rgb(
            (r as f32 * intensity) as u8,
            (g as f32 * intensity) as u8,
            (b as f32 * intensity) as u8,
        )
    } else {
        Color::Rgb(60, 60, 70)
    };

    // Title style with subtle pulse for very new events
    let title_style = if glow > 0.5 {
        // Very new: pulsing brightness
        let pulse = (tick as f32 * 0.3).sin() * 0.2 + 0.8;
        let (r, g, b) = color_to_rgb(color);
        Style::default()
            .fg(Color::Rgb(
                (r as f32 * pulse).min(255.0) as u8,
                (g as f32 * pulse).min(255.0) as u8,
                (b as f32 * pulse).min(255.0) as u8,
            ))
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(color).add_modifier(Modifier::BOLD)
    };

    // New event indicator - brief flash marker
    let new_indicator = if glow > 0.7 {
        "★ "
    } else if glow > 0.3 {
        "● "
    } else {
        ""
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(
            format!(" {}{} {} ", new_indicator, icon, label),
            title_style,
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" {} ", event.time_ago()),
                Style::default().fg(if is_new {
                    Color::White
                } else {
                    Color::DarkGray
                }),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Content with glow effect on text
    let content_color = if is_new {
        let base = 180.0 + glow * 75.0; // 180 to 255
        Color::Rgb(base as u8, base as u8, base as u8)
    } else {
        Color::Rgb(180, 180, 180)
    };

    let mut lines: Vec<Line> = Vec::new();
    if let Some(preview) = &event.event.content_preview {
        lines.push(Line::from(Span::styled(
            truncate(preview, inner.width as usize - 2),
            Style::default().fg(content_color),
        )));
    }
    let mut info_spans = Vec::new();
    if let Some(mem_type) = &event.event.memory_type {
        info_spans.push(Span::styled(
            mem_type,
            Style::default().fg(Color::Rgb(255, 200, 150)),
        ));
        info_spans.push(Span::raw(" "));
    }
    if let Some(mode) = &event.event.retrieval_mode {
        info_spans.push(Span::styled(mode, Style::default().fg(Color::Magenta)));
        info_spans.push(Span::raw(" "));
    }
    if let Some(count) = event.event.count {
        info_spans.push(Span::styled(
            format!("x{}", count),
            Style::default().fg(Color::Yellow),
        ));
    }
    if !info_spans.is_empty() {
        lines.push(Line::from(info_spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn render_event_detail(f: &mut Frame, area: Rect, event: &DisplayEvent, state: &AppState) {
    let has_rich = event.event.results.is_some();
    let right_hint = if has_rich {
        " j/k scroll  Backspace close "
    } else {
        " Backspace to close "
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            if has_rich {
                " ▼ RESULTS "
            } else {
                " ▼ MEMORY DETAILS "
            },
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                right_hint,
                Style::default().fg(Color::DarkGray),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    if has_rich {
        render_rich_results(f, inner, event, state);
        return;
    }

    let color = event.event.event_color();
    let mut lines = Vec::new();

    // Header line with type and metadata
    let mut header = vec![Span::styled(
        format!("{} ", event.event.event_type),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )];
    if let Some(t) = &event.event.memory_type {
        header.push(Span::styled(
            format!("[{}] ", t),
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ));
    }
    if let Some(l) = event.event.latency_ms {
        header.push(Span::styled(
            format!("{:.0}ms ", l),
            Style::default().fg(Color::Yellow),
        ));
    }
    lines.push(Line::from(header));

    // Full content - word wrapped across multiple lines
    if let Some(content) = &event.event.content_preview {
        lines.push(Line::from(""));
        let max_width = inner.width.saturating_sub(2) as usize;
        let mut remaining = content.as_str();
        let mut content_lines = 0;
        while !remaining.is_empty() && content_lines < 6 {
            let take = remaining.chars().take(max_width).collect::<String>();
            let actual_len = take.len();
            lines.push(Line::from(Span::styled(
                take,
                Style::default().fg(state.theme.fg()),
            )));
            remaining = &remaining[actual_len.min(remaining.len())..];
            content_lines += 1;
        }
        if !remaining.is_empty() {
            lines.push(Line::from(Span::styled(
                "...(more)",
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    // ID and timestamp
    lines.push(Line::from(""));
    let mut id_line = Vec::new();
    if let Some(id) = &event.event.memory_id {
        id_line.push(Span::styled("ID: ", Style::default().fg(Color::DarkGray)));
        id_line.push(Span::styled(
            id.clone(),
            Style::default().fg(Color::Rgb(255, 200, 150)),
        ));
        id_line.push(Span::raw("  "));
    }
    let local_time = event.event.timestamp.with_timezone(&chrono::Local);
    id_line.push(Span::styled(
        format!("@ {}", local_time.format("%Y-%m-%d %H:%M:%S")),
        Style::default().fg(Color::DarkGray),
    ));
    lines.push(Line::from(id_line));

    // Entities
    if let Some(entities) = &event.event.entities {
        if !entities.is_empty() {
            let es = entities.join(", ");
            lines.push(Line::from(vec![
                Span::styled("Tags: ", Style::default().fg(Color::DarkGray)),
                Span::styled(es, Style::default().fg(Color::Rgb(180, 230, 180))),
            ]));
        }
    }

    f.render_widget(Paragraph::new(lines), inner);
}

/// Render rich results panel for RETRIEVE / PROACTIVE_CONTEXT events
fn render_rich_results(f: &mut Frame, area: Rect, event: &DisplayEvent, state: &AppState) {
    let results = match &event.event.results {
        Some(v) => v,
        None => return,
    };
    let max_w = area.width.saturating_sub(1) as usize;
    let mut lines: Vec<Line> = Vec::new();

    // Header: event type, mode/context, latency, count
    let is_retrieve = event.event.event_type == "RETRIEVE";
    let color = event.event.event_color();

    let mut hdr = vec![Span::styled(
        if is_retrieve { "RECALL" } else { "PROACTIVE" },
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )];

    if let Some(mode) = results.get("mode").and_then(|v| v.as_str()) {
        hdr.push(Span::styled(
            format!(" [{}]", mode),
            Style::default().fg(Color::Rgb(255, 200, 150)),
        ));
    }
    if let Some(lat) = results.get("latency_ms").and_then(|v| v.as_f64()) {
        hdr.push(Span::styled(
            format!(" {:.0}ms", lat),
            Style::default().fg(Color::Yellow),
        ));
    }
    let query_or_ctx = if is_retrieve { "query" } else { "context" };
    if let Some(q) = results.get(query_or_ctx).and_then(|v| v.as_str()) {
        let display: String = q.chars().take(max_w.saturating_sub(30)).collect();
        hdr.push(Span::styled(
            format!("  \"{}\"", display),
            Style::default().fg(Color::DarkGray),
        ));
    }
    lines.push(Line::from(hdr));
    lines.push(Line::from(""));

    // Memories section
    if let Some(mems) = results.get("memories").and_then(|v| v.as_array()) {
        if !mems.is_empty() {
            lines.push(Line::from(Span::styled(
                format!(" Memories ({})", mems.len()),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            for mem in mems {
                let score = mem.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let pct = (score * 100.0).round() as u16;
                let bar_filled = (score * 8.0).round() as usize;
                let bar: String = "█".repeat(bar_filled)
                    + &"░".repeat(8usize.saturating_sub(bar_filled));
                let mtype = mem
                    .get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let bar_color = if pct >= 80 {
                    Color::Green
                } else if pct >= 50 {
                    Color::Yellow
                } else {
                    Color::Red
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {} {:>3}%", bar, pct),
                        Style::default().fg(bar_color),
                    ),
                    Span::styled(
                        format!(" | {}", mtype),
                        Style::default().fg(Color::Rgb(255, 200, 150)),
                    ),
                ]));

                // Content (wrapped to max 2 lines)
                if let Some(content) = mem.get("content").and_then(|v| v.as_str()) {
                    let content_w = max_w.saturating_sub(4);
                    let first: String = content.chars().take(content_w).collect();
                    lines.push(Line::from(Span::styled(
                        format!("    {}", first),
                        Style::default().fg(state.theme.fg()),
                    )));
                    if content.chars().count() > content_w {
                        let second: String =
                            content.chars().skip(content_w).take(content_w).collect();
                        lines.push(Line::from(Span::styled(
                            format!("    {}", second),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                }

                // Tags
                if let Some(tags) = mem.get("tags").and_then(|v| v.as_array()) {
                    if !tags.is_empty() {
                        let tag_strs: Vec<&str> =
                            tags.iter().filter_map(|t| t.as_str()).collect();
                        if !tag_strs.is_empty() {
                            lines.push(Line::from(Span::styled(
                                format!("    #{}", tag_strs.join(" #")),
                                Style::default().fg(Color::Rgb(180, 230, 180)),
                            )));
                        }
                    }
                }

                // Separator between memories
                lines.push(Line::from(Span::styled(
                    format!("  {}", "─".repeat(max_w.saturating_sub(4))),
                    Style::default().fg(Color::Rgb(60, 60, 60)),
                )));
            }
        }
    }

    // Facts section
    if let Some(facts) = results.get("facts").and_then(|v| v.as_array()) {
        if !facts.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!(" Facts ({})", facts.len()),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )));
            for fact in facts {
                let conf = fact
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let text = fact.get("fact").and_then(|v| v.as_str()).unwrap_or("?");
                let display: String = text.chars().take(max_w.saturating_sub(12)).collect();
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  ({:>2}%) ", (conf * 100.0).round() as u16),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled(display, Style::default().fg(state.theme.fg())),
                ]));
            }
        }
    }

    // Todos section
    if let Some(todos) = results.get("todos").and_then(|v| v.as_array()) {
        if !todos.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!(" Todos ({})", todos.len()),
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            )));
            for todo in todos {
                let sid = todo
                    .get("short_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let content = todo
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let pri = todo
                    .get("priority")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let status = todo
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let pri_color = match pri {
                    "urgent" => Color::Red,
                    "high" => Color::Rgb(255, 165, 0),
                    "medium" => Color::Yellow,
                    _ => Color::DarkGray,
                };
                let display: String = content.chars().take(max_w.saturating_sub(25)).collect();
                lines.push(Line::from(vec![
                    Span::styled(format!("  {}: ", sid), Style::default().fg(Color::Cyan)),
                    Span::styled(display, Style::default().fg(state.theme.fg())),
                    Span::styled(
                        format!(" [{}]", pri),
                        Style::default().fg(pri_color),
                    ),
                    Span::styled(
                        format!(" {}", status),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
        }
    }

    // Reminders section (due + context combined)
    let mut all_reminders: Vec<&serde_json::Value> = Vec::new();
    if let Some(r) = results.get("reminders").and_then(|v| v.as_array()) {
        all_reminders.extend(r.iter());
    }
    if let Some(r) = results.get("due_reminders").and_then(|v| v.as_array()) {
        all_reminders.extend(r.iter());
    }
    if let Some(r) = results.get("context_reminders").and_then(|v| v.as_array()) {
        all_reminders.extend(r.iter());
    }
    if !all_reminders.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!(" Reminders ({})", all_reminders.len()),
            Style::default()
                .fg(Color::Rgb(255, 100, 100))
                .add_modifier(Modifier::BOLD),
        )));
        for rem in &all_reminders {
            let content = rem
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let display: String = content.chars().take(max_w.saturating_sub(6)).collect();
            lines.push(Line::from(vec![
                Span::styled("  ! ", Style::default().fg(Color::Red)),
                Span::styled(display, Style::default().fg(state.theme.fg())),
            ]));
        }
    }

    // Detected entities (proactive_context)
    if let Some(entities) = results.get("detected_entities").and_then(|v| v.as_array()) {
        if !entities.is_empty() {
            lines.push(Line::from(""));
            let names: Vec<&str> = entities
                .iter()
                .filter_map(|e| e.get("name").and_then(|v| v.as_str()))
                .collect();
            lines.push(Line::from(vec![
                Span::styled(
                    " Entities: ",
                    Style::default()
                        .fg(Color::Rgb(180, 230, 180))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    names.join(", "),
                    Style::default().fg(Color::Rgb(180, 230, 180)),
                ),
            ]));
        }
    }

    // Apply scroll offset and render visible lines
    let visible_height = area.height as usize;
    let total = lines.len();
    let scroll = state.event_detail_scroll.min(total.saturating_sub(visible_height));
    let visible: Vec<Line> = lines.into_iter().skip(scroll).take(visible_height).collect();

    f.render_widget(Paragraph::new(visible), area);

    // Scrollbar hint if content exceeds visible area
    if total > visible_height {
        let indicator = format!(" {}/{} ", scroll + visible_height.min(total), total);
        let ind_area = Rect {
            x: area.x + area.width.saturating_sub(indicator.len() as u16 + 1),
            y: area.y + area.height.saturating_sub(1),
            width: indicator.len() as u16,
            height: 1,
        };
        f.render_widget(
            Paragraph::new(Span::styled(
                indicator,
                Style::default().fg(Color::DarkGray),
            )),
            ind_area,
        );
    }
}

fn render_activity_logs(f: &mut Frame, area: Rect, state: &AppState) {
    let content_area = with_ribbon_layout(f, area, state);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " ACTIVITY LOGS ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" {} events ", state.events.len()),
                Style::default().fg(Color::Magenta),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(content_area);
    f.render_widget(block, content_area);

    if state.events.is_empty() {
        let msg = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Waiting for memory events...",
                Style::default().fg(Color::DarkGray),
            )),
        ]);
        f.render_widget(msg, inner);
        return;
    }

    let has_rich_selected_logs = state.selected_event.and_then(|i| state.events.get(i)).map_or(false, |e| e.event.results.is_some());
    let detail_height = if state.selected_event.is_some() {
        if has_rich_selected_logs {
            (inner.height * 3 / 4).max(15)
        } else {
            12u16
        }
    } else {
        0u16
    };
    let list_height = inner.height.saturating_sub(detail_height);
    let event_height = 5u16;
    let max_events = (list_height / event_height) as usize;

    let scroll_start = if let Some(sel) = state.selected_event {
        if sel < state.scroll_offset {
            sel
        } else if max_events > 0 && sel >= state.scroll_offset + max_events {
            sel.saturating_sub(max_events.saturating_sub(1))
        } else {
            state.scroll_offset
        }
    } else {
        state.scroll_offset
    };

    let end = (scroll_start + max_events).min(state.events.len());
    let mut y_offset = 0;

    for (global_idx, event) in state
        .events
        .iter()
        .enumerate()
        .skip(scroll_start)
        .take(end.saturating_sub(scroll_start))
    {
        let event_area = Rect {
            x: inner.x,
            y: inner.y + y_offset,
            width: inner.width,
            height: event_height,
        };
        if event_area.y + event_area.height <= inner.y + list_height {
            let is_selected = state.selected_event == Some(global_idx);
            let is_newest = global_idx == 0;
            render_river_event_selectable(f, event_area, state, event, is_selected, is_newest);
        }
        y_offset += event_height;
    }

    if let Some(sel_idx) = state.selected_event {
        if let Some(event) = state.events.get(sel_idx) {
            let detail_area = Rect {
                x: inner.x,
                y: inner.y + list_height,
                width: inner.width,
                height: detail_height,
            };
            render_event_detail(f, detail_area, event, state);
        }
    }

    if state.events.len() > max_events {
        let scrollbar = Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state = ScrollbarState::new(state.events.len()).position(scroll_start);
        f.render_stateful_widget(
            scrollbar,
            area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut scrollbar_state,
        );
    }
}

fn render_river_event_selectable(
    f: &mut Frame,
    area: Rect,
    state: &AppState,
    event: &DisplayEvent,
    is_selected: bool,
    _is_newest: bool,
) {
    let color = event.event.event_color();
    let icon = event.event.event_icon();
    let sep = "-".repeat(area.width.saturating_sub(27) as usize);

    // Animation effects
    let opacity = event.visual_opacity();
    let glow = event.glow();
    let is_animating = event.is_animating();

    // Skip yellow highlight for historical events (loaded on startup)
    let is_history = event.event.event_type == "HISTORY";

    // Yellow highlight for NEW events only - fades over 30 seconds
    let flash_age = event.received_at.elapsed().as_millis() as f32;
    let flash_intensity = if !is_history && flash_age < 30000.0 {
        1.0 - (flash_age / 30000.0) // 1.0 at start, fades to 0.0 over 30 seconds
    } else {
        0.0
    };

    let bg = if is_selected {
        state.theme.selection_bg()
    } else if glow > 0.05 {
        // Bright green glow that fades over 2 seconds
        Color::Rgb(
            (20.0 + glow * 60.0) as u8,
            (50.0 + glow * 150.0) as u8,
            (30.0 + glow * 80.0) as u8,
        )
    } else {
        state.theme.bg()
    };

    // Text colors - Yellow for live new events, not historical ones
    // Use theme-aware colors for readability on both dark and light backgrounds
    let is_fresh = !is_history && flash_intensity > 0.05;
    let text_color = if is_selected {
        state.theme.fg()
    } else if is_fresh {
        Color::Yellow
    } else {
        state.theme.fg_dim()
    };

    let event_color = if is_fresh { Color::Yellow } else { color };

    // Prefix: bright indicator for fresh live events only
    let (prefix, prefix_color) = if is_fresh {
        ("★ ", Color::Yellow) // Yellow star for fresh live events
    } else if is_selected {
        ("▶ ", Color::Rgb(255, 200, 150))
    } else {
        ("  ", Color::DarkGray)
    };

    // Border - yellow for fresh live events only
    let border_color = if is_fresh {
        Color::Yellow
    } else if glow > 0.0 {
        lerp_color(Color::DarkGray, Color::Rgb(180, 230, 180), glow * 0.5)
    } else {
        Color::DarkGray
    };

    let mut lines = vec![Line::from(vec![
        Span::styled(
            prefix,
            Style::default()
                .fg(prefix_color)
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        ),
        Span::styled(
            format!("{:>5} ", event.time_ago()),
            Style::default().fg(border_color).bg(bg),
        ),
        Span::styled("-+- ", Style::default().fg(border_color).bg(bg)),
        Span::styled(
            icon,
            Style::default()
                .fg(event_color)
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        ),
        Span::styled(
            format!(" {} ", event.event.event_type),
            Style::default()
                .fg(event_color)
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        ),
        Span::styled(sep, Style::default().fg(border_color).bg(bg)),
    ])];

    if let Some(preview) = &event.event.content_preview {
        lines.push(Line::from(vec![
            Span::styled("      ", Style::default().bg(bg)),
            Span::styled("| ", Style::default().fg(border_color).bg(bg)),
            Span::styled(
                truncate(preview, area.width.saturating_sub(10) as usize),
                Style::default().fg(text_color).bg(bg),
            ),
        ]));
    }

    let meta_color = if is_animating {
        apply_opacity(Color::Rgb(255, 200, 150), opacity)
    } else {
        Color::Rgb(255, 200, 150)
    };
    let mut meta = vec![
        Span::styled("      ", Style::default().bg(bg)),
        Span::styled("| ", Style::default().fg(border_color).bg(bg)),
    ];
    if let Some(t) = &event.event.memory_type {
        meta.push(Span::styled(
            format!("type:{} ", t),
            Style::default().fg(meta_color).bg(bg),
        ));
    }
    if let Some(m) = &event.event.retrieval_mode {
        meta.push(Span::styled(
            format!("mode:{} ", m),
            Style::default()
                .fg(if is_animating {
                    apply_opacity(Color::Magenta, opacity)
                } else {
                    Color::Magenta
                })
                .bg(bg),
        ));
    }
    if let Some(l) = event.event.latency_ms {
        meta.push(Span::styled(
            format!("{:.0}ms ", l),
            Style::default()
                .fg(if is_animating {
                    apply_opacity(Color::Yellow, opacity)
                } else {
                    Color::Yellow
                })
                .bg(bg),
        ));
    }
    if let Some(id) = &event.event.memory_id {
        meta.push(Span::styled(
            format!("id:{}", &id[..8.min(id.len())]),
            Style::default().fg(border_color).bg(bg),
        ));
    }
    lines.push(Line::from(meta));

    if let Some(entities) = &event.event.entities {
        if !entities.is_empty() {
            let es = entities
                .iter()
                .take(5)
                .map(|e| e.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let tag_color = if is_animating {
                apply_opacity(Color::Rgb(180, 230, 180), opacity)
            } else if glow > 0.0 {
                glow_color(Color::Rgb(180, 230, 180), glow * 0.3)
            } else {
                Color::Rgb(180, 230, 180)
            };
            lines.push(Line::from(vec![
                Span::styled("      ", Style::default().bg(bg)),
                Span::styled("| ", Style::default().fg(border_color).bg(bg)),
                Span::styled("entities: ", Style::default().fg(border_color).bg(bg)),
                Span::styled(truncate(&es, 40), Style::default().fg(tag_color).bg(bg)),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines), area);
}

fn render_river_event(f: &mut Frame, area: Rect, event: &DisplayEvent) {
    let color = event.event.event_color();
    let icon = event.event.event_icon();
    let sep = "-".repeat(area.width.saturating_sub(25) as usize);

    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("{:>5} ", event.time_ago()),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled("-+- ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            icon,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {} ", event.event.event_type),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(sep, Style::default().fg(Color::DarkGray)),
    ])];

    if let Some(preview) = &event.event.content_preview {
        lines.push(Line::from(vec![
            Span::raw("      "),
            Span::styled("| ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                truncate(preview, area.width.saturating_sub(10) as usize),
                Style::default().fg(Color::White),
            ),
        ]));
    }

    let mut meta = vec![
        Span::raw("      "),
        Span::styled("| ", Style::default().fg(Color::DarkGray)),
    ];
    if let Some(t) = &event.event.memory_type {
        meta.push(Span::styled(
            format!("type:{} ", t),
            Style::default().fg(Color::Rgb(255, 200, 150)),
        ));
    }
    if let Some(m) = &event.event.retrieval_mode {
        meta.push(Span::styled(
            format!("mode:{} ", m),
            Style::default().fg(Color::Magenta),
        ));
    }
    if let Some(l) = event.event.latency_ms {
        meta.push(Span::styled(
            format!("{:.0}ms ", l),
            Style::default().fg(Color::Yellow),
        ));
    }
    if let Some(id) = &event.event.memory_id {
        meta.push(Span::styled(
            format!("id:{}", &id[..8.min(id.len())]),
            Style::default().fg(Color::DarkGray),
        ));
    }
    lines.push(Line::from(meta));

    if let Some(entities) = &event.event.entities {
        if !entities.is_empty() {
            let es = entities
                .iter()
                .take(5)
                .map(|e| e.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(Line::from(vec![
                Span::raw("      "),
                Span::styled("| ", Style::default().fg(Color::DarkGray)),
                Span::styled("entities: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    truncate(&es, 40),
                    Style::default().fg(Color::Rgb(180, 230, 180)),
                ),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines), area);
}

/// Get emoji for entity type
fn entity_type_emoji(entity_type: &str) -> &'static str {
    match entity_type.to_lowercase().as_str() {
        "person" | "per" => "👤",
        "organization" | "org" => "🏢",
        "location" | "loc" | "gpe" => "📍",
        "technology" | "tech" | "product" => "💻",
        "project" | "work_of_art" => "📦",
        "language" | "norp" => "🗣️",
        "event" => "📅",
        "money" | "cardinal" | "quantity" | "percent" => "💰",
        "date" | "time" => "⏰",
        "law" | "fac" => "📜",
        _ => "◆",
    }
}

/// Format connection strength as visual indicator
fn connection_strength_indicator(weight: f32) -> (&'static str, Color) {
    if weight >= 0.7 {
        ("━━━", CONN_STRONG) // Strong - green
    } else if weight >= 0.4 {
        ("━━╍", CONN_MEDIUM) // Medium - yellow
    } else {
        ("╍╍╍", CONN_WEAK) // Weak - gray
    }
}

/// Get connection color based on weight (for graph visualization)
fn connection_color(weight: f32) -> Color {
    if weight >= 0.7 {
        CONN_STRONG
    } else if weight >= 0.4 {
        CONN_MEDIUM
    } else {
        CONN_WEAK
    }
}

fn render_graph_map(f: &mut Frame, area: Rect, state: &AppState) {
    let content_area = with_ribbon_layout(f, area, state);

    // Three-column layout: Entities (20%) | Connections (20%) | Visualization (60%)
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20), // Left: Entity list
            Constraint::Percentage(20), // Middle: Connections
            Constraint::Percentage(60), // Right: Visualization (hero)
        ])
        .split(content_area);

    render_graph_entity_selector(f, main_chunks[0], state);
    render_graph_connections_panel(f, main_chunks[1], state);
    render_graph_visualization(f, main_chunks[2], state);
}

/// Compact entity selector for graph map view - sorted by name for stable ordering
fn render_graph_entity_selector(f: &mut Frame, area: Rect, state: &AppState) {
    use crate::types::FocusPanel;

    let entity_count = state.graph_data.nodes.len();
    let selected_idx = state.graph_data.selected_node;
    let is_focused = state.graph_map_focus == FocusPanel::Left;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if is_focused { GOLD } else { BORDER_DIVIDER }))
        .title(Span::styled(
            " Entities ",
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" {} ", entity_count),
                Style::default().fg(TEXT_DISABLED),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.graph_data.nodes.is_empty() {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  No entities yet",
                    Style::default().fg(TEXT_DISABLED),
                )),
            ]),
            inner,
        );
        return;
    }

    // Create sorted indices for stable display order (by name alphabetically)
    let mut sorted_indices: Vec<usize> = (0..state.graph_data.nodes.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        state.graph_data.nodes[a]
            .content
            .to_lowercase()
            .cmp(&state.graph_data.nodes[b].content.to_lowercase())
    });

    // Find where the selected node appears in the sorted list
    let selected_sort_pos = sorted_indices
        .iter()
        .position(|&idx| idx == selected_idx)
        .unwrap_or(0);

    let max_visible = inner.height as usize;
    let scroll_offset = if selected_sort_pos >= max_visible {
        selected_sort_pos - max_visible + 1
    } else {
        0
    };

    let mut lines = Vec::new();
    let max_name_width = inner.width.saturating_sub(10) as usize;

    for &node_idx in sorted_indices.iter().skip(scroll_offset).take(max_visible) {
        let node = &state.graph_data.nodes[node_idx];
        let is_selected = node_idx == selected_idx;
        let emoji = entity_type_emoji(&node.memory_type);
        let bg = if is_selected {
            SELECTION_BG
        } else {
            Color::Reset
        };

        let name = truncate(&node.content, max_name_width);
        let count_str = format!("{:>3}", node.connections);

        // Pad to fill width for background
        let padding = inner.width.saturating_sub(name.len() as u16 + 8) as usize;

        lines.push(Line::from(vec![
            Span::styled(
                if is_selected { " ▸ " } else { "   " },
                Style::default()
                    .fg(if is_selected { GOLD } else { TEXT_DISABLED })
                    .bg(bg),
            ),
            Span::styled(format!("{} ", emoji), Style::default().bg(bg)),
            Span::styled(
                name,
                Style::default()
                    .fg(if is_selected {
                        Color::White
                    } else {
                        TEXT_PRIMARY
                    })
                    .bg(bg)
                    .add_modifier(if is_selected {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
            Span::styled(" ".repeat(padding), Style::default().bg(bg)),
            Span::styled(count_str, Style::default().fg(TEXT_DISABLED).bg(bg)),
        ]));
    }

    f.render_widget(Paragraph::new(lines), inner);

    // Scrollbar
    if entity_count > max_visible {
        let scrollbar = Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state = ScrollbarState::new(entity_count).position(selected_sort_pos);
        f.render_stateful_widget(
            scrollbar,
            area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut scrollbar_state,
        );
    }
}

/// Middle panel: Connections list for selected entity
fn render_graph_connections_panel(f: &mut Frame, area: Rect, state: &AppState) {
    use crate::types::FocusPanel;

    let is_focused = state.graph_map_focus == FocusPanel::Right;
    let selected_conn = state.selected_connection;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if is_focused { GOLD } else { BORDER_DIVIDER }))
        .title(Span::styled(
            " Connections ",
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(" ←→ ", Style::default().fg(TEXT_DISABLED)))
                .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(selected) = state.graph_data.selected() else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  Select an entity",
                    Style::default().fg(TEXT_DISABLED),
                )),
            ]),
            inner,
        );
        return;
    };

    // Subtitle showing selected entity
    let emoji = entity_type_emoji(&selected.memory_type);
    let subtitle = format!(
        " {} {}",
        emoji,
        truncate(&selected.content, (inner.width - 4) as usize)
    );

    // Get connected entities
    let mut connections: Vec<(&str, &str, f32, bool)> = Vec::new();

    for edge in &state.graph_data.edges {
        if edge.from_id == selected.id {
            if let Some(target) = state.graph_data.nodes.iter().find(|n| n.id == edge.to_id) {
                connections.push((&target.content, &target.memory_type, edge.weight, true));
            }
        } else if edge.to_id == selected.id {
            if let Some(source) = state.graph_data.nodes.iter().find(|n| n.id == edge.from_id) {
                connections.push((&source.content, &source.memory_type, edge.weight, false));
            }
        }
    }

    connections.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Start with subtitle showing selected entity
    let mut lines = vec![
        Line::from(Span::styled(
            subtitle,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""), // Spacer
    ];

    if connections.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No connections",
            Style::default().fg(TEXT_DISABLED),
        )));
        f.render_widget(Paragraph::new(lines), inner);
        return;
    }

    // Account for subtitle + spacer (2 lines)
    let max_visible = inner.height.saturating_sub(3) as usize;
    let max_name_width = inner.width.saturating_sub(12) as usize;

    // Calculate scroll offset for connections
    let scroll_offset = if selected_conn >= max_visible {
        selected_conn - max_visible + 1
    } else {
        0
    };

    for (idx, (name, entity_type, weight, is_outgoing)) in connections
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(max_visible)
    {
        let is_selected = is_focused && idx == selected_conn;
        let conn_emoji = entity_type_emoji(entity_type);
        let arrow = if *is_outgoing { "→" } else { "←" };
        let (strength_bar, strength_color) = connection_strength_indicator(*weight);
        let bg = if is_selected {
            SELECTION_BG
        } else {
            Color::Reset
        };

        let prefix = if is_selected { "▸" } else { " " };

        lines.push(Line::from(vec![
            Span::styled(
                format!("{} {} ", prefix, arrow),
                Style::default()
                    .fg(if is_selected { GOLD } else { TEXT_DISABLED })
                    .bg(bg),
            ),
            Span::styled(format!("{} ", conn_emoji), Style::default().bg(bg)),
            Span::styled(
                truncate(name, max_name_width),
                Style::default()
                    .fg(if is_selected {
                        Color::White
                    } else {
                        TEXT_PRIMARY
                    })
                    .bg(bg)
                    .add_modifier(if is_selected {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
            Span::styled(" ", Style::default().bg(bg)),
            Span::styled(strength_bar, Style::default().fg(strength_color).bg(bg)),
        ]));
    }

    if connections.len() > max_visible + scroll_offset {
        lines.push(Line::from(Span::styled(
            format!(
                "  +{} more",
                connections.len() - max_visible - scroll_offset
            ),
            Style::default().fg(TEXT_DISABLED),
        )));
    }

    f.render_widget(Paragraph::new(lines), inner);
}

/// Right panel: Large radial visualization (50%)
fn render_graph_visualization(f: &mut Frame, area: Rect, state: &AppState) {
    use crate::types::FocusPanel;

    let is_connections_focused = state.graph_map_focus == FocusPanel::Right;
    let highlighted_conn = if is_connections_focused {
        Some(state.selected_connection)
    } else {
        None
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER_DIVIDER))
        .title(Span::styled(
            " Knowledge Graph ",
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                " ↑↓ navigate ",
                Style::default().fg(TEXT_DISABLED),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(selected) = state.graph_data.selected() else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  Select an entity to visualize",
                    Style::default().fg(TEXT_DISABLED),
                )),
                Line::from(Span::styled(
                    "  its connections",
                    Style::default().fg(TEXT_DISABLED),
                )),
            ]),
            inner,
        );
        return;
    };

    let width = inner.width as usize;
    let height = inner.height as usize;

    if height < 5 || width < 20 {
        return;
    }

    // Get connections (same order as connections panel)
    let mut connected: Vec<(&str, &str, f32)> = Vec::new(); // (name, type, weight)
    for edge in &state.graph_data.edges {
        if edge.from_id == selected.id {
            if let Some(target) = state.graph_data.nodes.iter().find(|n| n.id == edge.to_id) {
                connected.push((&target.content, &target.memory_type, edge.weight));
            }
        } else if edge.to_id == selected.id {
            if let Some(source) = state.graph_data.nodes.iter().find(|n| n.id == edge.from_id) {
                connected.push((&source.content, &source.memory_type, edge.weight));
            }
        }
    }
    connected.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Build character grid
    let mut grid: Vec<Vec<(char, Color)>> = vec![vec![(' ', Color::DarkGray); width]; height];

    let center_x = width / 2;
    let center_y = height / 2;

    // Draw center node with box
    let center_label = truncate(&selected.content, 16);
    let box_half_width = (center_label.len() + 2) / 2;
    let label_start = center_x.saturating_sub(box_half_width);
    let box_left = label_start.saturating_sub(1);
    let box_right = (label_start + center_label.len() + 1).min(width - 1);

    // Draw box around center
    if center_y > 0 && center_y + 1 < height {
        // Top border
        grid[center_y - 1][box_left] = ('╭', GOLD);
        for x in (box_left + 1)..box_right {
            if x < width {
                grid[center_y - 1][x] = ('─', GOLD);
            }
        }
        if box_right < width {
            grid[center_y - 1][box_right] = ('╮', GOLD);
        }
        // Bottom border
        grid[center_y + 1][box_left] = ('╰', GOLD);
        for x in (box_left + 1)..box_right {
            if x < width {
                grid[center_y + 1][x] = ('─', GOLD);
            }
        }
        if box_right < width {
            grid[center_y + 1][box_right] = ('╯', GOLD);
        }
        // Side borders
        grid[center_y][box_left] = ('│', GOLD);
        if box_right < width {
            grid[center_y][box_right] = ('│', GOLD);
        }
    }

    // Draw center label
    for (i, ch) in center_label.chars().enumerate() {
        let x = label_start + i;
        if x < width && x > box_left && x < box_right {
            grid[center_y][x] = (ch, Color::White);
        }
    }

    // Draw connections radially
    let max_connections = 8.min(connected.len());
    let radius_x = (width as f32 * 0.38) as usize;
    let radius_y = (height as f32 * 0.40) as usize;

    // Highlight color for selected connection
    let highlight_color = Color::Rgb(255, 100, 100); // Red highlight

    for (i, (name, entity_type, weight)) in connected.iter().take(max_connections).enumerate() {
        let is_highlighted = highlighted_conn == Some(i);
        let angle = (i as f32 / max_connections as f32) * std::f32::consts::PI * 2.0
            - std::f32::consts::PI / 2.0;
        let target_x = (center_x as f32 + angle.cos() * radius_x as f32) as usize;
        let target_y = (center_y as f32 + angle.sin() * radius_y as f32) as usize;

        // Edge color - highlight if selected, otherwise use unified weight-based colors
        let edge_color = if is_highlighted {
            highlight_color
        } else {
            connection_color(*weight)
        };

        // Draw line from center to target
        let dx = target_x as i32 - center_x as i32;
        let dy = target_y as i32 - center_y as i32;
        let steps = dx.abs().max(dy.abs()) as usize;

        if steps > 3 {
            for step in 3..(steps - 1) {
                let t = step as f32 / steps as f32;
                let x = (center_x as f32 + dx as f32 * t) as usize;
                let y = (center_y as f32 + dy as f32 * t) as usize;
                if y < height && x < width && grid[y][x].0 == ' ' {
                    let ch = if is_highlighted {
                        '━' // Thicker line for highlighted
                    } else if dx.abs() > dy.abs() * 2 {
                        '─'
                    } else if dy.abs() > dx.abs() * 2 {
                        '│'
                    } else {
                        '·'
                    };
                    grid[y][x] = (ch, edge_color);
                }
            }
        }

        // Draw target node with emoji - highlight if selected
        let node_emoji = entity_type_emoji(entity_type);
        let label = format!("{} {}", node_emoji, truncate(name, 10));
        let label_x = if target_x > center_x {
            target_x.min(width.saturating_sub(label.len()))
        } else {
            target_x.saturating_sub(label.len())
        };

        // Use highlight color for selected connection's target node
        let label_color = if is_highlighted {
            highlight_color
        } else {
            TEXT_PRIMARY
        };

        if target_y < height {
            for (j, ch) in label.chars().enumerate() {
                if label_x + j < width {
                    grid[target_y][label_x + j] = (ch, label_color);
                }
            }
        }
    }

    // Draw legend at bottom (before overflow to avoid collision)
    if !connected.is_empty() && height > 3 {
        let legend_y = height - 2; // One line above the very bottom
        let legend_parts = [
            ("━━━", CONN_STRONG),
            (" Strong  ", TEXT_DISABLED),
            ("━━╍", CONN_MEDIUM),
            (" Medium  ", TEXT_DISABLED),
            ("╍╍╍", CONN_WEAK),
            (" Weak", TEXT_DISABLED),
        ];

        let mut x = 1;
        for (text, color) in legend_parts {
            for ch in text.chars() {
                if x < width {
                    grid[legend_y][x] = (ch, color);
                    x += 1;
                }
            }
        }
    }

    // Show overflow indicator at bottom right
    if connected.len() > max_connections {
        let overflow_text = format!("+{} more", connected.len() - max_connections);
        if height > 1 {
            for (i, ch) in overflow_text.chars().enumerate() {
                let x = width.saturating_sub(overflow_text.len()) + i;
                if x < width {
                    grid[height - 1][x] = (ch, TEXT_DISABLED);
                }
            }
        }
    }

    // Convert grid to lines
    let lines: Vec<Line> = grid
        .iter()
        .map(|row| {
            Line::from(
                row.iter()
                    .map(|(ch, color)| Span::styled(ch.to_string(), Style::default().fg(*color)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    f.render_widget(Paragraph::new(lines), inner);
}

/// Left panel: Top entities sorted by connections (legacy - used by old map view)
fn render_graph_top_entities(f: &mut Frame, area: Rect, state: &AppState) {
    let node_count = state.graph_data.nodes.len();
    let selected_idx = state.graph_data.selected_node;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " TOP ENTITIES ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" [{}/{}] ", selected_idx + 1, node_count),
                Style::default().fg(Color::Yellow),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.graph_data.nodes.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  No entities",
                Style::default().fg(Color::DarkGray),
            )),
            inner,
        );
        return;
    }

    // Sort nodes by connections (descending)
    let mut sorted_indices: Vec<usize> = (0..state.graph_data.nodes.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        state.graph_data.nodes[b]
            .connections
            .cmp(&state.graph_data.nodes[a].connections)
    });

    let max_display = inner.height as usize;

    // Find where selected node is in the sorted list
    let selected_sort_position = sorted_indices
        .iter()
        .position(|&idx| idx == state.graph_data.selected_node)
        .unwrap_or(0);

    // Calculate scroll offset to keep selected in view
    let scroll_offset = if selected_sort_position >= max_display {
        selected_sort_position - max_display + 1
    } else {
        0
    };

    let mut lines = Vec::new();

    for (visible_idx, &node_idx) in sorted_indices
        .iter()
        .skip(scroll_offset)
        .take(max_display)
        .enumerate()
    {
        let node = &state.graph_data.nodes[node_idx];
        let is_selected = node_idx == state.graph_data.selected_node;
        let rank = scroll_offset + visible_idx;

        // Rank medal for top 3
        let rank_str = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        let prefix = if is_selected { "▶" } else { " " };
        let emoji = entity_type_emoji(&node.memory_type);
        let name = truncate(&node.content, 12);

        let style = if is_selected {
            Style::default()
                .fg(state.theme.accent())
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(state.theme.fg())
        };

        let conn_color = if node.connections >= 10 {
            Color::Rgb(180, 230, 180)
        } else if node.connections >= 5 {
            Color::Yellow
        } else {
            state.theme.fg_dim()
        };

        lines.push(Line::from(vec![
            Span::styled(
                prefix,
                Style::default().fg(if is_selected {
                    state.theme.accent()
                } else {
                    state.theme.fg_dim()
                }),
            ),
            Span::raw(rank_str),
            Span::raw(emoji),
            Span::styled(name, style),
            Span::styled(
                format!(" {:>2}", node.connections),
                Style::default().fg(conn_color).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    f.render_widget(Paragraph::new(lines), inner);

    // Show scrollbar if there are more nodes than can display
    if node_count > max_display {
        let scrollbar = Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state = ScrollbarState::new(node_count).position(selected_sort_position);
        f.render_stateful_widget(
            scrollbar,
            area.inner(Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut scrollbar_state,
        );
    }
}

/// Center panel: Selected entity with radial connections visualization
fn render_graph_focus_view(f: &mut Frame, area: Rect, state: &AppState) {
    let selected_info = state.graph_data.selected().map(|n| {
        let emoji = entity_type_emoji(&n.memory_type);
        format!(" {} {} ", emoji, truncate(&n.content, 18))
    });

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER_DIVIDER))
        .title(Span::styled(
            selected_info
                .clone()
                .unwrap_or_else(|| " Connections ".to_string()),
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                " ↑↓ navigate ",
                Style::default().fg(TEXT_DISABLED),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(selected) = state.graph_data.selected() else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  Select an entity to visualize",
                    Style::default().fg(TEXT_DISABLED),
                )),
                Line::from(Span::styled(
                    "  its connections",
                    Style::default().fg(TEXT_DISABLED),
                )),
            ]),
            inner,
        );
        return;
    };

    // Get connected nodes
    let mut connected: Vec<(&str, &str, f32, bool)> = Vec::new(); // (id, name, weight, is_outgoing)

    for edge in &state.graph_data.edges {
        if edge.from_id == selected.id {
            if let Some(target) = state.graph_data.nodes.iter().find(|n| n.id == edge.to_id) {
                connected.push((&target.id, &target.content, edge.weight, true));
            }
        } else if edge.to_id == selected.id {
            if let Some(source) = state.graph_data.nodes.iter().find(|n| n.id == edge.from_id) {
                connected.push((&source.id, &source.content, edge.weight, false));
            }
        }
    }

    // Sort by weight descending
    connected.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let width = inner.width as usize;
    let height = inner.height as usize;

    if height < 3 {
        return;
    }

    // Render as a star/radial pattern with selected in center
    let center_x = width / 2;
    let center_y = height / 2;

    let mut lines: Vec<Line> = vec![Line::from(""); height];

    // Build the visualization as a character grid
    let mut grid: Vec<Vec<(char, Color)>> = vec![vec![(' ', Color::DarkGray); width]; height];

    // Draw center node (selected entity)
    let center_label = truncate(&selected.content, 16);
    let label_start = center_x.saturating_sub(center_label.len() / 2);
    for (i, ch) in center_label.chars().enumerate() {
        if label_start + i < width {
            grid[center_y][label_start + i] = (ch, Color::Yellow);
        }
    }

    // Draw box around center
    if center_y > 0 && center_y + 1 < height {
        let box_left = label_start.saturating_sub(1);
        let box_right = (label_start + center_label.len()).min(width - 1);
        // Top border
        if center_y > 0 {
            grid[center_y - 1][box_left] = ('╭', Color::Yellow);
            for x in (box_left + 1)..box_right {
                grid[center_y - 1][x] = ('─', Color::Yellow);
            }
            if box_right < width {
                grid[center_y - 1][box_right] = ('╮', Color::Yellow);
            }
        }
        // Bottom border
        if center_y + 1 < height {
            grid[center_y + 1][box_left] = ('╰', Color::Yellow);
            for x in (box_left + 1)..box_right {
                grid[center_y + 1][x] = ('─', Color::Yellow);
            }
            if box_right < width {
                grid[center_y + 1][box_right] = ('╯', Color::Yellow);
            }
        }
        // Side borders
        grid[center_y][box_left] = ('│', Color::Yellow);
        if box_right < width {
            grid[center_y][box_right] = ('│', Color::Yellow);
        }
    }

    // Position connected nodes around the center
    let max_connections = 8.min(connected.len());
    let radius_x = (width as f32 * 0.35) as usize;
    let radius_y = (height as f32 * 0.35) as usize;

    for (i, (_, name, weight, is_outgoing)) in connected.iter().take(max_connections).enumerate() {
        let angle = (i as f32 / max_connections as f32) * std::f32::consts::PI * 2.0
            - std::f32::consts::PI / 2.0;
        let target_x = (center_x as f32 + angle.cos() * radius_x as f32) as usize;
        let target_y = (center_y as f32 + angle.sin() * radius_y as f32) as usize;

        // Draw connection line
        let edge_color = if *weight >= 0.7 {
            Color::Rgb(180, 230, 180)
        } else if *weight >= 0.4 {
            Color::Yellow
        } else {
            Color::DarkGray
        };

        let arrow = if *is_outgoing { '→' } else { '←' };

        // Simple line from center to target
        let dx = target_x as i32 - center_x as i32;
        let dy = target_y as i32 - center_y as i32;
        let steps = dx.abs().max(dy.abs()) as usize;

        if steps > 2 {
            for step in 2..(steps - 1) {
                let t = step as f32 / steps as f32;
                let x = (center_x as f32 + dx as f32 * t) as usize;
                let y = (center_y as f32 + dy as f32 * t) as usize;
                if y < height && x < width && grid[y][x].0 == ' ' {
                    let ch = if dx.abs() > dy.abs() * 2 {
                        '─'
                    } else if dy.abs() > dx.abs() * 2 {
                        '│'
                    } else {
                        '·'
                    };
                    grid[y][x] = (ch, edge_color);
                }
            }
        }

        // Draw arrow near target
        if target_y < height && target_x < width {
            grid[target_y][target_x] = (arrow, edge_color);
        }

        // Draw target label
        let label = truncate(name, 10);
        let label_x = if target_x > center_x {
            (target_x + 1).min(width.saturating_sub(label.len()))
        } else {
            target_x.saturating_sub(label.len() + 1)
        };

        // Use theme-aware color for connected node labels
        let label_color = state.theme.fg();
        for (j, ch) in label.chars().enumerate() {
            if label_x + j < width && target_y < height {
                grid[target_y][label_x + j] = (ch, label_color);
            }
        }
    }

    // Show "... and N more" if truncated
    if connected.len() > max_connections {
        let more_text = format!("... +{} more", connected.len() - max_connections);
        if height > 1 {
            for (i, ch) in more_text.chars().enumerate() {
                if i < width {
                    grid[height - 1][i] = (ch, Color::DarkGray);
                }
            }
        }
    }

    // Convert grid to lines
    for (y, row) in grid.iter().enumerate() {
        lines[y] = Line::from(
            row.iter()
                .map(|(ch, color)| Span::styled(ch.to_string(), Style::default().fg(*color)))
                .collect::<Vec<_>>(),
        );
    }

    f.render_widget(Paragraph::new(lines), inner);
}

/// Right panel: Entity type breakdown and stats
fn render_graph_type_summary(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " TYPES ",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" {}e ", state.graph_data.edges.len()),
                Style::default().fg(Color::Yellow),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Count entities by type
    let mut type_counts: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for node in &state.graph_data.nodes {
        let type_name = if node.memory_type.is_empty() {
            "Unknown"
        } else {
            &node.memory_type
        };
        *type_counts.entry(type_name).or_insert(0) += 1;
    }

    // Sort by count
    let mut type_vec: Vec<_> = type_counts.into_iter().collect();
    type_vec.sort_by(|a, b| b.1.cmp(&a.1));

    let mut lines = Vec::new();

    // Type breakdown
    lines.push(Line::from(Span::styled(
        " By Type:",
        Style::default().fg(Color::DarkGray),
    )));

    let total = state.graph_data.nodes.len() as f32;
    for (type_name, count) in type_vec.iter().take(6) {
        let pct = (*count as f32 / total * 100.0) as u32;
        let bar_width = ((pct as f32 / 100.0) * 8.0) as usize;
        let bar = "█".repeat(bar_width) + &"░".repeat(8 - bar_width);

        let type_color = match *type_name {
            "Person" => Color::Rgb(255, 200, 150),
            "Organization" => Color::Rgb(180, 200, 255), // Pastel blue
            "Location" => Color::Rgb(180, 230, 180),
            "Technology" => Color::Magenta,
            "Issue" => Color::Yellow,
            _ => state.theme.fg(),
        };

        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:8}", truncate(type_name, 8)),
                Style::default().fg(type_color),
            ),
            Span::styled(bar, Style::default().fg(type_color)),
            Span::styled(format!(" {}", count), Style::default().fg(Color::DarkGray)),
        ]));
    }

    lines.push(Line::from(""));

    // Edge stats
    lines.push(Line::from(Span::styled(
        " Connections:",
        Style::default().fg(Color::DarkGray),
    )));

    let strong = state.graph_stats.strong_edges;
    let medium = state.graph_stats.medium_edges;
    let weak = state.graph_stats.weak_edges;

    lines.push(Line::from(vec![
        Span::styled(" Strong ", Style::default().fg(Color::Rgb(180, 230, 180))),
        Span::styled(
            format!("{}", strong),
            Style::default()
                .fg(Color::Rgb(180, 230, 180))
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" Medium ", Style::default().fg(Color::Yellow)),
        Span::styled(
            format!("{}", medium),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" Weak   ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", weak),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
    ]));

    lines.push(Line::from(""));

    // Density
    lines.push(Line::from(vec![
        Span::styled(" Density: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.1}%", state.graph_stats.density * 100.0),
            Style::default().fg(Color::Rgb(255, 200, 150)),
        ),
    ]));

    // Avg weight
    lines.push(Line::from(vec![
        Span::styled(" Avg wt:  ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.2}", state.graph_stats.avg_weight),
            Style::default().fg(Color::Rgb(255, 200, 150)),
        ),
    ]));

    f.render_widget(Paragraph::new(lines), inner);
}

pub fn render_footer(f: &mut Frame, area: Rect, state: &AppState) {
    let view_name = match state.view_mode {
        ViewMode::Dashboard => "Dashboard",
        ViewMode::Projects => "Projects",
        ViewMode::ActivityLogs => "Activity",
        ViewMode::GraphMap => "Graph",
    };

    // Check for error message
    if let Some((error_msg, timestamp)) = &state.error_message {
        let secs_remaining = 5u64.saturating_sub(timestamp.elapsed().as_secs());
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Red));
        let error_line = Line::from(vec![
            Span::styled(
                " ⚠ ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                truncate(error_msg, area.width.saturating_sub(20) as usize),
                Style::default().fg(Color::Red),
            ),
            Span::styled(
                format!(" ({}s)", secs_remaining),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
        f.render_widget(Paragraph::new(error_line).block(block), area);
        return;
    }

    // Search input mode
    if state.search_active {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(Span::styled(
                " SEARCH ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
        let cursor_char = if state.animation_tick % 10 < 5 {
            "█"
        } else {
            " "
        };
        // Show format hint for date mode
        let hint = match state.search_mode {
            SearchMode::Date => " (7d, 2w, 1m, YYYY-MM-DD)",
            _ => "",
        };
        let search_line = Line::from(vec![
            Span::styled(
                format!(" [{}] ", state.search_mode.label()),
                Style::default()
                    .fg(Color::Rgb(255, 200, 150))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(&state.search_query, Style::default().fg(Color::White)),
            Span::styled(cursor_char, Style::default().fg(Color::Yellow)),
            Span::styled(hint, Style::default().fg(Color::DarkGray)),
            Span::raw("  "),
            Span::styled("Tab", Style::default().fg(Color::DarkGray)),
            Span::styled("=mode ", Style::default().fg(Color::DarkGray)),
            Span::styled("Enter", Style::default().fg(Color::DarkGray)),
            Span::styled("=search ", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc", Style::default().fg(Color::DarkGray)),
            Span::styled("=cancel", Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(search_line).block(block), area);
        return;
    }

    // Search results visible - show navigation hints
    if state.search_results_visible {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(180, 230, 180)))
            .title(Span::styled(
                format!(" {} RESULTS ", state.search_results.len()),
                Style::default()
                    .fg(Color::Rgb(180, 230, 180))
                    .add_modifier(Modifier::BOLD),
            ));
        let result_line = Line::from(vec![
            Span::styled(
                " j/k ",
                Style::default()
                    .fg(Color::Rgb(255, 200, 150))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("navigate ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " / ",
                Style::default()
                    .fg(Color::Rgb(255, 200, 150))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("new search ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " Esc ",
                Style::default()
                    .fg(Color::Rgb(255, 200, 150))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("close ", Style::default().fg(Color::DarkGray)),
            Span::raw("  "),
            Span::styled(
                format!(
                    "[{}/{}]",
                    state.search_selected + 1,
                    state.search_results.len()
                ),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]);
        f.render_widget(Paragraph::new(result_line).block(block), area);
        return;
    }

    // Normal footer - context-sensitive based on view
    let is_graph_view = matches!(state.view_mode, ViewMode::GraphMap);

    let mut keys = vec![
        Span::styled(
            " / ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("search ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "q ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("quit ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "d ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("dash ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "p ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("proj ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "a ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("activity ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "j/k ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("↑↓ ", Style::default().fg(Color::DarkGray)),
    ];

    // Add view-specific controls
    if matches!(state.view_mode, ViewMode::ActivityLogs) {
        keys.extend([
            Span::styled(
                "+/- ",
                Style::default()
                    .fg(Color::Rgb(255, 200, 150))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("zoom ", Style::default().fg(Color::DarkGray)),
        ]);
    }

    keys.extend([
        Span::styled(
            "^U ",
            Style::default()
                .fg(Color::Rgb(255, 200, 150))
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("switch ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("[{}]", view_name),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            format!("[{}]", state.zoom_label()),
            Style::default().fg(state.theme.fg_dim()),
        ),
    ]);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(state.theme.border()));
    f.render_widget(Paragraph::new(Line::from(keys)).block(block), area);
}
