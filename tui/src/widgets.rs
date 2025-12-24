use crate::logo::{ELEPHANT, ELEPHANT_GRADIENT, SHODH_GRADIENT, SHODH_TEXT};
use crate::types::{
    AppState, DisplayEvent, FocusPanel, SearchMode, SearchResult, TuiPriority, TuiTodo,
    TuiTodoStatus, ViewMode, VERSION,
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
            Constraint::Length(20),
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
                .fg(Color::Cyan)
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
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("recalls", Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(Paragraph::new(stats_line), title_chunks[1]);

    // Right side: version, status with heartbeat, sparkline, session
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Version
            Constraint::Length(2), // Status with heartbeat
            Constraint::Length(2), // Sparkline
            Constraint::Min(0),    // Session
        ])
        .split(chunks[2]);

    let version_line = Line::from(vec![Span::styled(
        format!("v{}", VERSION),
        Style::default().fg(Color::DarkGray),
    )]);
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
        Color::Green // High activity
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

    let session = Line::from(Span::styled(
        state.session_duration(),
        Style::default().fg(Color::White),
    ));
    f.render_widget(
        Paragraph::new(session).alignment(Alignment::Right),
        right_chunks[3],
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
        ViewMode::GraphList => render_graph_list(f, area, state),
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
        .border_style(Style::default().fg(Color::Green))
        .title(Span::styled(
            format!(
                " SEARCH RESULTS: \"{}\" ({}) ",
                truncate(&state.search_query, 30),
                state.search_results.len()
            ),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" [{}] ", state.search_mode.label()),
                Style::default().fg(Color::Cyan),
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
                    Color::Cyan
                } else {
                    Color::DarkGray
                })
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        ),
        Span::styled(
            format!("[{}]", result.memory_type),
            Style::default()
                .fg(Color::Cyan)
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
                Style::default().fg(Color::Green).bg(bg),
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
                .fg(Color::Cyan)
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
                    .fg(Color::Cyan)
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
                Span::styled(result.tags.join(", "), Style::default().fg(Color::Green))
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
    // Vertical split: ribbon at top, spacer, content below
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),   // Status ribbon
            Constraint::Length(2),   // Spacer for breathing room
            Constraint::Min(5),      // Main content
        ])
        .split(area);

    render_status_ribbon(f, rows[0], state);
    // rows[1] is empty spacer

    // 40/60 split: Todos+Stats on left, Activity on right
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),  // Todos + Stats
            Constraint::Percentage(60),  // Activity
        ])
        .split(rows[2]);

    // Left panel: split vertically for todos (top) and stats (bottom)
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),      // Todos (expandable)
            Constraint::Length(8),    // Stats summary (compact)
        ])
        .split(columns[0]);

    render_todos_panel(f, left_chunks[0], state);
    render_compact_stats(f, left_chunks[1], state);
    render_activity_feed(f, columns[1], state);
}

// ============================================================================
// PROJECTS VIEW - Full-width layout with proper spacing
// ============================================================================

/// Main Projects view - full-width ribbon + two-column layout
fn render_projects_view(f: &mut Frame, area: Rect, state: &AppState) {
    // Vertical split: ribbon at top, spacer, columns below
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),   // Status ribbon
            Constraint::Length(2),   // Spacer for breathing room
            Constraint::Min(5),      // Main content
        ])
        .split(area);

    render_status_ribbon(f, rows[0], state);
    // rows[1] is empty spacer

    // 50/50 columns for main content
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(rows[2]);

    render_projects_sidebar(f, columns[0], state);
    render_todos_panel_right(f, columns[1], state);
}

/// Full-width status ribbon showing current work context
fn render_status_ribbon(f: &mut Frame, area: Rect, state: &AppState) {
    let width = area.width as usize;
    let ribbon_bg = Color::Rgb(50, 45, 40);

    let in_progress = state.in_progress_todos();
    let mut spans: Vec<Span> = Vec::new();

    // Current task info
    if let Some(current) = in_progress.first() {
        let duration = format_duration_since(&current.created_at);

        // Project name
        let project_name = if let Some(ref pid) = current.project_id {
            state.projects.iter()
                .find(|p| &p.id == pid)
                .map(|p| p.name.as_str())
                .unwrap_or("—")
        } else {
            "Inbox"
        };

        // Priority indicator
        let (pri_icon, pri_color) = match current.priority {
            TuiPriority::Urgent => ("▲▲▲", MAROON),
            TuiPriority::High => ("▲▲", SAFFRON),
            TuiPriority::Medium => ("▲", TEXT_DISABLED),
            TuiPriority::Low => ("", TEXT_DISABLED),
        };

        spans.push(Span::styled(" ◐ ", Style::default().fg(SAFFRON).bg(ribbon_bg)));
        spans.push(Span::styled(
            truncate(project_name, 15),
            Style::default().fg(TEXT_SECONDARY).bg(ribbon_bg)
        ));
        spans.push(Span::styled(" › ", Style::default().fg(TEXT_DISABLED).bg(ribbon_bg)));
        spans.push(Span::styled(
            truncate(&current.content, 30),
            Style::default().fg(TEXT_PRIMARY).bg(ribbon_bg)
        ));
        if !pri_icon.is_empty() {
            spans.push(Span::styled(
                format!("  {} ", pri_icon),
                Style::default().fg(pri_color).bg(ribbon_bg)
            ));
        }
        spans.push(Span::styled(
            format!("  ⏱ {}", duration),
            Style::default().fg(TEXT_DISABLED).bg(ribbon_bg)
        ));

        // Next up (if there's another todo)
        let todos: Vec<_> = state.todos.iter()
            .filter(|t| t.status == TuiTodoStatus::Todo)
            .collect();
        if let Some(next) = todos.first() {
            spans.push(Span::styled("   │   ", Style::default().fg(TEXT_DISABLED).bg(ribbon_bg)));
            spans.push(Span::styled("next: ", Style::default().fg(TEXT_DISABLED).bg(ribbon_bg)));
            spans.push(Span::styled(
                truncate(&next.content, 20),
                Style::default().fg(TEXT_SECONDARY).bg(ribbon_bg)
            ));
        }
    } else {
        // Idle state
        spans.push(Span::styled(" ○ ", Style::default().fg(TEXT_DISABLED).bg(ribbon_bg)));
        spans.push(Span::styled("idle", Style::default().fg(TEXT_DISABLED).bg(ribbon_bg)));

        // Show next todo if any
        let todos: Vec<_> = state.todos.iter()
            .filter(|t| t.status == TuiTodoStatus::Todo)
            .collect();
        if let Some(next) = todos.first() {
            spans.push(Span::styled("   │   ", Style::default().fg(TEXT_DISABLED).bg(ribbon_bg)));
            spans.push(Span::styled("next: ", Style::default().fg(TEXT_DISABLED).bg(ribbon_bg)));
            spans.push(Span::styled(
                truncate(&next.content, 25),
                Style::default().fg(TEXT_SECONDARY).bg(ribbon_bg)
            ));
        }
    }

    // Pad to full width
    let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    if used < width {
        spans.push(Span::styled(
            " ".repeat(width.saturating_sub(used)),
            Style::default().bg(ribbon_bg)
        ));
    }

    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Left sidebar - projects list with flat navigation (projects + todos)
fn render_projects_sidebar(f: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),      // Projects list
            Constraint::Length(1),   // Footer
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
        Span::styled(format!(" PROJECTS {} ", proj_count), Style::default().fg(TEXT_SECONDARY)),
        Span::styled(header_line, Style::default().fg(Color::Rgb(40, 40, 40))),
    ]));
    lines.push(Line::from("")); // breathing room

    // Flat index for navigation - includes projects and their expanded todos
    let mut flat_idx = 0;

    // Render projects with folder icons
    for project in state.projects.iter() {
        if lines.len() >= inner.height as usize - 1 {
            break;
        }

        let is_selected = state.projects_selected == flat_idx;
        let is_expanded = state.is_project_expanded(&project.id);
        let todos = state.todos_for_project(&project.id);
        let done = todos.iter().filter(|t| t.status == TuiTodoStatus::Done).count();
        let active = todos.iter().filter(|t| t.status == TuiTodoStatus::InProgress).count();
        let remaining = todos.iter().filter(|t| t.status != TuiTodoStatus::Done && t.status != TuiTodoStatus::Cancelled).count();
        let total = todos.len();

        // Folder icon: 📂 open, 📁 closed
        let folder = if is_expanded { "📂" } else { "📁" };
        // Cursor always visible - brighter when focused
        let sel = if is_selected { "▸ " } else { "  " };
        let sel_color = if is_selected && is_left_focused { SAFFRON } else if is_selected { TEXT_DISABLED } else { Color::Reset };
        let name_width = width.saturating_sub(28);
        let name = truncate(&project.name, name_width);

        // Progress percentage
        let pct = if total > 0 { (done * 100) / total } else { 0 };
        let progress_color = if pct == 100 { GOLD } else if active > 0 { SAFFRON } else { TEXT_DISABLED };
        let bg = if is_selected { Color::Rgb(40, 40, 50) } else { Color::Reset };

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
            Span::styled(format!("{} ", folder), Style::default().bg(bg)),
            Span::styled(format!("{:<w$}", name, w = name_width), Style::default().fg(TEXT_PRIMARY).bg(bg)),
            Span::styled(format!(" {:<12}", status_str), Style::default().fg(progress_color).bg(bg)),
        ]));
        flat_idx += 1;

        // Expanded todos with indentation - each one is navigable
        if is_expanded {
            for todo in todos.iter().take(5) {
                if lines.len() >= inner.height as usize - 1 {
                    break;
                }
                let todo_selected = state.projects_selected == flat_idx;
                lines.push(render_sidebar_todo(todo, width, todo_selected, is_left_focused));
                flat_idx += 1;
            }
            if todos.len() > 5 {
                lines.push(Line::from(Span::styled(
                    format!("       +{} more", todos.len() - 5),
                    Style::default().fg(TEXT_DISABLED),
                )));
            }
            lines.push(Line::from("")); // space after expanded project
        }
    }

    // INBOX section with spacing
    let standalone = state.standalone_todos();
    if !standalone.is_empty() && lines.len() < inner.height as usize - 2 {
        lines.push(Line::from("")); // breathing room
        let inbox_line = "─".repeat(width.saturating_sub(12));
        lines.push(Line::from(vec![
            Span::styled(format!(" INBOX {} ", standalone.len()), Style::default().fg(TEXT_SECONDARY)),
            Span::styled(inbox_line, Style::default().fg(Color::Rgb(40, 40, 40))),
        ]));
        lines.push(Line::from("")); // breathing room
        for todo in standalone.iter().take(5) {
            if lines.len() >= inner.height as usize {
                break;
            }
            let todo_selected = state.projects_selected == flat_idx;
            lines.push(render_sidebar_todo(todo, width, todo_selected, is_left_focused));
            flat_idx += 1;
        }
    }

    f.render_widget(Paragraph::new(lines), inner);

    // Footer shows panel focus state
    let footer = if is_left_focused {
        Line::from(vec![
            Span::styled(" ▸ ", Style::default().fg(SAFFRON)),
            Span::styled("↑↓", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(" navigate  ", Style::default().fg(Color::Rgb(60, 60, 60))),
            Span::styled("→", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(" todos  ", Style::default().fg(Color::Rgb(60, 60, 60))),
            Span::styled("Enter", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(" expand", Style::default().fg(Color::Rgb(60, 60, 60))),
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
            Constraint::Min(1),      // Content
            Constraint::Length(1),   // Footer
        ])
        .split(area);

    let content_area = chunks[0];
    let content_height = content_area.height as usize;
    let mut lines: Vec<Line> = Vec::new();

    // Get todos for selected project or inbox
    let (title, todos): (String, Vec<&TuiTodo>) = if state.projects_selected < state.projects.len() {
        let project = &state.projects[state.projects_selected];
        (project.name.clone(), state.todos_for_project(&project.id))
    } else {
        ("Inbox".to_string(), state.standalone_todos())
    };

    // Header with focus indicator
    let header_line = "─".repeat(width.saturating_sub(title.len() + 6));
    let focus_indicator = if is_focused { "▸ " } else { "  " };
    lines.push(Line::from(vec![
        Span::styled(focus_indicator, Style::default().fg(SAFFRON)),
        Span::styled(format!("{} ", title), Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
        Span::styled(header_line, Style::default().fg(Color::Rgb(40, 40, 40))),
    ]));
    lines.push(Line::from("")); // breathing room

    if todos.is_empty() {
        lines.push(Line::from(Span::styled("   No tasks yet", Style::default().fg(TEXT_DISABLED))));
    } else {
        // Build flat list of todos for selection tracking
        let mut flat_todos: Vec<&TuiTodo> = Vec::new();

        // Collect in order: in_progress, todo, blocked, done
        let in_progress: Vec<_> = todos.iter().filter(|t| t.status == TuiTodoStatus::InProgress).cloned().collect();
        let todo_items: Vec<_> = todos.iter().filter(|t| t.status == TuiTodoStatus::Todo).cloned().collect();
        let blocked: Vec<_> = todos.iter().filter(|t| t.status == TuiTodoStatus::Blocked).cloned().collect();
        let done: Vec<_> = todos.iter().filter(|t| t.status == TuiTodoStatus::Done).cloned().collect();

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
            lines.push(Line::from("")); // space after header
            for todo in in_progress.iter().take(4) {
                if lines.len() >= content_height - 1 { break; }
                let is_selected = state.todos_selected == todo_idx;
                lines.push(render_todo_row_with_selection(todo, width, is_selected, is_focused));
                todo_idx += 1;
            }
            lines.push(Line::from("")); // section separator
        }

        // Todo section
        if !todo_items.is_empty() && lines.len() < content_height - 2 {
            lines.push(Line::from(Span::styled(
                format!(" ○ Todo ({})", todo_items.len()),
                Style::default().fg(TEXT_SECONDARY),
            )));
            lines.push(Line::from("")); // space after header
            let max_todos = (content_height.saturating_sub(lines.len() + 4)).min(8);
            for todo in todo_items.iter().take(max_todos) {
                if lines.len() >= content_height - 1 { break; }
                let is_selected = state.todos_selected == todo_idx;
                lines.push(render_todo_row_with_selection(todo, width, is_selected, is_focused));
                todo_idx += 1;
            }
            if todo_items.len() > max_todos {
                lines.push(Line::from(Span::styled(
                    format!("      +{} more", todo_items.len() - max_todos),
                    Style::default().fg(TEXT_DISABLED),
                )));
            }
            lines.push(Line::from("")); // section separator
        }

        // Blocked section
        if !blocked.is_empty() && lines.len() < content_height - 2 {
            lines.push(Line::from(Span::styled(
                format!(" ⊘ Blocked ({})", blocked.len()),
                Style::default().fg(MAROON),
            )));
            lines.push(Line::from("")); // space after header
            for todo in blocked.iter().take(3) {
                if lines.len() >= content_height - 1 { break; }
                let is_selected = state.todos_selected == todo_idx;
                lines.push(render_todo_row_with_selection(todo, width, is_selected, is_focused));
                todo_idx += 1;
            }
            lines.push(Line::from("")); // section separator
        }

        // Done section (collapsed, not navigable)
        if !done.is_empty() && lines.len() < content_height {
            lines.push(Line::from(Span::styled(
                format!(" ● Completed ({})", done.len()),
                Style::default().fg(TEXT_DISABLED),
            )));
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
            Span::styled(" navigate tasks", Style::default().fg(Color::Rgb(50, 50, 50))),
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

    let content_width = width.saturating_sub(15);
    let content = truncate(&todo.content, content_width);

    let mut spans = vec![
        Span::styled("   ", Style::default()),
        Span::styled(format!("{} ", icon), Style::default().fg(color)),
        Span::styled(priority.0, Style::default().fg(priority.1)),
        Span::styled(content, Style::default().fg(if todo.status == TuiTodoStatus::Done { TEXT_DISABLED } else { TEXT_PRIMARY })),
    ];

    if todo.is_overdue() {
        if let Some(label) = todo.due_label() {
            spans.push(Span::styled(format!(" {}", label), Style::default().fg(MAROON)));
        }
    }

    Line::from(spans)
}

/// Render a todo row with selection highlighting
fn render_todo_row_with_selection(todo: &TuiTodo, width: usize, is_selected: bool, is_panel_focused: bool) -> Line<'static> {
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

    let content_width = width.saturating_sub(18);
    let content = truncate(&todo.content, content_width);

    // Selection indicator and background - always visible, brighter when focused
    let sel_marker = if is_selected { "▸ " } else { "   " };
    let sel_color = if is_selected && is_panel_focused { SAFFRON } else if is_selected { TEXT_DISABLED } else { Color::Reset };
    let bg = if is_selected { Color::Rgb(40, 40, 50) } else { Color::Reset };
    let text_color = if todo.status == TuiTodoStatus::Done { TEXT_DISABLED } else { TEXT_PRIMARY };

    let mut spans = vec![
        Span::styled(sel_marker, Style::default().fg(sel_color).bg(bg)),
        Span::styled(format!("{} ", icon), Style::default().fg(color).bg(bg)),
        Span::styled(priority.0, Style::default().fg(priority.1).bg(bg)),
        Span::styled(content, Style::default().fg(text_color).bg(bg)),
    ];

    if todo.is_overdue() {
        if let Some(label) = todo.due_label() {
            spans.push(Span::styled(format!(" {}", label), Style::default().fg(MAROON).bg(bg)));
        }
    }

    // Pad to full width for consistent background
    let used_width: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    if used_width < width && is_selected {
        spans.push(Span::styled(" ".repeat(width.saturating_sub(used_width)), Style::default().bg(bg)));
    }

    Line::from(spans)
}

/// Render todo under expanded project in sidebar (with selection support)
fn render_sidebar_todo(todo: &TuiTodo, width: usize, is_selected: bool, is_panel_focused: bool) -> Line<'static> {
    let (icon, color) = match todo.status {
        TuiTodoStatus::InProgress => ("◐", SAFFRON),
        TuiTodoStatus::Todo => ("○", TEXT_SECONDARY),
        TuiTodoStatus::Done => ("●", GOLD),
        TuiTodoStatus::Blocked => ("⊘", MAROON),
        _ => ("○", TEXT_DISABLED),
    };

    // Selection indicator - always visible, brighter when focused
    let sel = if is_selected { "  ▸ " } else { "    " };
    let sel_color = if is_selected && is_panel_focused { SAFFRON } else if is_selected { TEXT_DISABLED } else { Color::Reset };
    let bg = if is_selected { Color::Rgb(40, 40, 50) } else { Color::Reset };
    let content_width = width.saturating_sub(12);

    Line::from(vec![
        Span::styled(sel, Style::default().fg(sel_color).bg(bg)),
        Span::styled(format!("{} ", icon), Style::default().fg(color).bg(bg)),
        Span::styled(truncate(&todo.content, content_width), Style::default().fg(TEXT_SECONDARY).bg(bg)),
    ])
}

/// Compact stats panel for dashboard
fn render_compact_stats(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(" Stats ", Style::default().fg(Color::Rgb(255, 215, 0))));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let stats_line1 = Line::from(vec![
        Span::styled("Memories: ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", state.total_memories), Style::default().fg(Color::White)),
        Span::styled("  Recalls: ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", state.total_recalls), Style::default().fg(Color::White)),
    ]);

    let stats_line2 = Line::from(vec![
        Span::styled("W:", Style::default().fg(Color::Yellow)),
        Span::styled(format!("{} ", state.tier_stats.working), Style::default().fg(Color::White)),
        Span::styled("S:", Style::default().fg(Color::Cyan)),
        Span::styled(format!("{} ", state.tier_stats.session), Style::default().fg(Color::White)),
        Span::styled("L:", Style::default().fg(Color::Green)),
        Span::styled(format!("{}", state.tier_stats.long_term), Style::default().fg(Color::White)),
        Span::styled("  Nodes:", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", state.graph_stats.nodes), Style::default().fg(Color::Magenta)),
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
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("  Working   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar(state.tier_stats.working, tier_total, 8),
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!(" {}", state.tier_stats.working),
                Style::default().fg(Color::Green),
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
            .fg(Color::Cyan)
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
                .fg(Color::Cyan)
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
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(
                format!(" {:.2}", state.graph_stats.density),
                Style::default().fg(Color::Cyan),
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
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            Span::styled("  semantic ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                progress_bar(state.retrieval_stats.semantic, ret_total, 5),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(
                format!(" {}", state.retrieval_stats.semantic),
                Style::default().fg(Color::Cyan),
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
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                format!(" {}", state.retrieval_stats.hybrid),
                Style::default().fg(Color::Green),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(ret_lines), chunks[3]);

    // TOP ENTITIES
    let mut entity_lines = vec![Line::from(Span::styled(
        " ENTITIES",
        Style::default()
            .fg(Color::Cyan)
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
    // Show both vec len and stats total for debugging
    let active_count = state.todos.len();
    let project_count = state.projects.len();
    let with_project = state.todos.iter().filter(|t| t.project_name.is_some()).count();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            format!(" TODO ({}) [P:{}/{}] ", active_count, with_project, project_count),
            Style::default().fg(Color::Rgb(255, 215, 0)).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.todos.is_empty() {
        let empty_msg = Paragraph::new("No active todos")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        f.render_widget(empty_msg, inner);
        return;
    }

    // Group todos by status for Linear-style display
    let mut lines: Vec<Line> = Vec::new();

    // Calculate available height for todos (leave room for stats summary)
    let available_lines = inner.height.saturating_sub(3) as usize;
    let mut used_lines = 0;

    // In Progress section (priority - show more)
    let in_progress: Vec<_> = state.todos.iter()
        .filter(|t| t.status == TuiTodoStatus::InProgress)
        .collect();
    if !in_progress.is_empty() && used_lines < available_lines {
        lines.push(Line::from(Span::styled(
            format!("◐ In Progress ({})", in_progress.len()),
            Style::default().fg(Color::Rgb(255, 215, 0)).add_modifier(Modifier::BOLD),
        )));
        used_lines += 1;
        let show_count = (available_lines - used_lines).min(in_progress.len()).min(4);
        for todo in in_progress.iter().take(show_count) {
            lines.push(render_todo_line(todo));
            used_lines += 1;
        }
        if in_progress.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!("  +{} more", in_progress.len() - show_count),
                Style::default().fg(Color::DarkGray),
            )));
            used_lines += 1;
        }
    }

    // Todo section
    let todos: Vec<_> = state.todos.iter()
        .filter(|t| t.status == TuiTodoStatus::Todo)
        .collect();
    if !todos.is_empty() && used_lines < available_lines {
        lines.push(Line::from(Span::styled(
            format!("○ Todo ({})", todos.len()),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));
        used_lines += 1;
        let show_count = (available_lines - used_lines).min(todos.len()).min(4);
        for todo in todos.iter().take(show_count) {
            lines.push(render_todo_line(todo));
            used_lines += 1;
        }
        if todos.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!("  +{} more", todos.len() - show_count),
                Style::default().fg(Color::DarkGray),
            )));
            used_lines += 1;
        }
    }

    // Blocked section
    let blocked: Vec<_> = state.todos.iter()
        .filter(|t| t.status == TuiTodoStatus::Blocked)
        .collect();
    if !blocked.is_empty() && used_lines < available_lines {
        lines.push(Line::from(Span::styled(
            format!("⊘ Blocked ({})", blocked.len()),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )));
        used_lines += 1;
        let show_count = (available_lines - used_lines).min(blocked.len()).min(3);
        for todo in blocked.iter().take(show_count) {
            lines.push(render_todo_line(todo));
            used_lines += 1;
        }
        if blocked.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!("  +{} more", blocked.len() - show_count),
                Style::default().fg(Color::DarkGray),
            )));
            used_lines += 1;
        }
    }

    // Backlog section
    let backlog: Vec<_> = state.todos.iter()
        .filter(|t| t.status == TuiTodoStatus::Backlog)
        .collect();
    if !backlog.is_empty() && used_lines < available_lines {
        lines.push(Line::from(Span::styled(
            format!("◌ Backlog ({})", backlog.len()),
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
        )));
        used_lines += 1;
        let show_count = (available_lines - used_lines).min(backlog.len()).min(2);
        for todo in backlog.iter().take(show_count) {
            lines.push(render_todo_line(todo));
        }
        if backlog.len() > show_count {
            lines.push(Line::from(Span::styled(
                format!("  +{} more", backlog.len() - show_count),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    // Stats summary at bottom
    lines.push(Line::from(vec![
        Span::styled("─".repeat(inner.width as usize - 2), Style::default().fg(Color::DarkGray)),
    ]));
    lines.push(Line::from(vec![
        Span::styled(format!("◐{} ", state.todo_stats.in_progress), Style::default().fg(Color::Rgb(255, 215, 0))),
        Span::styled(format!("○{} ", state.todo_stats.todo), Style::default().fg(Color::White)),
        Span::styled(format!("⊘{} ", state.todo_stats.blocked), Style::default().fg(Color::Red)),
        Span::styled(format!("●{}", state.todo_stats.done), Style::default().fg(Color::Green)),
        if state.todo_stats.overdue > 0 {
            Span::styled(format!(" ⚠{}", state.todo_stats.overdue), Style::default().fg(Color::LightRed))
        } else {
            Span::raw("")
        },
    ]));

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, inner);
}

fn render_todo_line(todo: &TuiTodo) -> Line<'static> {
    let mut spans = vec![
        Span::styled("  ", Style::default()),
        Span::styled(
            todo.status.icon(),
            Style::default().fg(todo.status.color()),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(
            todo.priority.indicator(),
            Style::default().fg(todo.priority.color()),
        ),
    ];

    // Content (truncated based on whether we have project name)
    let has_project = todo.project_name.is_some();
    let max_content_len = if has_project { 18 } else { 28 };
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

    let detail_height = if state.selected_event.is_some() {
        12u16 // Expanded view for full content
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
            Color::Green
        } else if is_selected {
            Color::Cyan
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
                Style::default().fg(Color::Cyan),
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
                meta_spans.push(Span::styled(tags, Style::default().fg(Color::Green)));
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
        info_spans.push(Span::styled(mem_type, Style::default().fg(Color::Cyan)));
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
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            " ▼ MEMORY DETAILS ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                " Backspace to close ",
                Style::default().fg(Color::DarkGray),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

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
                .fg(Color::Cyan)
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
        // Split content into lines that fit the width
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
        id_line.push(Span::styled(id.clone(), Style::default().fg(Color::Cyan)));
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
                Span::styled(es, Style::default().fg(Color::Green)),
            ]));
        }
    }

    f.render_widget(Paragraph::new(lines), inner);
}

fn render_activity_logs(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " ACTIVITY LOGS ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" {} events ", state.events.len()),
                Style::default().fg(Color::Magenta),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

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

    let detail_height = if state.selected_event.is_some() {
        12u16 // Expanded view for full content
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
        ("▶ ", Color::Cyan)
    } else {
        ("  ", Color::DarkGray)
    };

    // Border - yellow for fresh live events only
    let border_color = if is_fresh {
        Color::Yellow
    } else if glow > 0.0 {
        lerp_color(Color::DarkGray, Color::Green, glow * 0.5)
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
        apply_opacity(Color::Cyan, opacity)
    } else {
        Color::Cyan
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
                apply_opacity(Color::Green, opacity)
            } else if glow > 0.0 {
                glow_color(Color::Green, glow * 0.3)
            } else {
                Color::Green
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
            Style::default().fg(Color::Cyan),
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
                Span::styled(truncate(&es, 40), Style::default().fg(Color::Green)),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines), area);
}

fn render_graph_list(f: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);

    // LEFT: Node list
    let node_count = state.graph_data.nodes.len();
    let selected_idx = state.graph_data.selected_node;
    let left_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " NODES ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(
                    " [{}/{}] ",
                    if node_count > 0 { selected_idx + 1 } else { 0 },
                    node_count
                ),
                Style::default().fg(Color::Yellow),
            ))
            .alignment(Alignment::Right),
        );
    let left_inner = left_block.inner(chunks[0]);
    f.render_widget(left_block, chunks[0]);

    if state.graph_data.nodes.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  No nodes yet",
                Style::default().fg(Color::DarkGray),
            )),
            left_inner,
        );
    } else {
        let lines_per_node = 2_usize;
        let max_visible_nodes = (left_inner.height as usize) / lines_per_node;

        // Calculate scroll offset to keep selected node in view
        let scroll_offset = if selected_idx >= max_visible_nodes {
            selected_idx - max_visible_nodes + 1
        } else {
            0
        };

        let mut lines = Vec::new();
        for (i, node) in state
            .graph_data
            .nodes
            .iter()
            .enumerate()
            .skip(scroll_offset)
            .take(max_visible_nodes)
        {
            let is_selected = i == selected_idx;
            let prefix = if is_selected { "> " } else { "  " };
            let style = if is_selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            let bar = progress_bar(
                node.connections,
                state.graph_data.nodes.len().max(1) as u32,
                4,
            );
            lines.push(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(
                    &node.short_id,
                    Style::default().fg(if is_selected {
                        Color::Yellow
                    } else {
                        Color::DarkGray
                    }),
                ),
                Span::raw(" "),
                Span::styled(
                    &node.memory_type[..3.min(node.memory_type.len())],
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" "),
                Span::styled(bar, Style::default().fg(Color::Green)),
                Span::styled(
                    format!(" {}", node.connections),
                    Style::default().fg(Color::Green),
                ),
            ]));
            // Use theme-aware colors for node content readability on both dark and light backgrounds
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    truncate(&node.content, left_inner.width.saturating_sub(4) as usize),
                    Style::default().fg(if is_selected {
                        state.theme.fg()
                    } else {
                        state.theme.fg_dim()
                    }),
                ),
            ]));
        }
        f.render_widget(Paragraph::new(lines), left_inner);

        // Show scrollbar if there are more nodes than can display
        if node_count > max_visible_nodes {
            let scrollbar = Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight);
            let mut scrollbar_state = ScrollbarState::new(node_count).position(selected_idx);
            f.render_stateful_widget(
                scrollbar,
                chunks[0].inner(Margin {
                    vertical: 1,
                    horizontal: 0,
                }),
                &mut scrollbar_state,
            );
        }
    }

    // RIGHT: Edges from selected node
    let right_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " EDGES ",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" {} ", state.graph_stats.edges),
                Style::default().fg(Color::Yellow),
            ))
            .alignment(Alignment::Right),
        );
    let right_inner = right_block.inner(chunks[1]);
    f.render_widget(right_block, chunks[1]);

    if let Some(selected) = state.graph_data.selected() {
        let mut lines = vec![
            Line::from(vec![
                Span::styled(" Selected: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    &selected.short_id,
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" ({})", selected.memory_type),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(Span::styled(
                format!(" {}", selected.content),
                Style::default().fg(state.theme.fg()),
            )),
            Line::from(""),
            Line::from(Span::styled(
                " Connections:",
                Style::default().fg(Color::DarkGray),
            )),
        ];

        let edges = state.graph_data.edges_from_selected();
        if edges.is_empty() {
            lines.push(Line::from(Span::styled(
                "   (no outgoing edges)",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            for (edge, target) in edges.iter().take(10) {
                let weight_color = if edge.weight >= 0.7 {
                    Color::Green
                } else if edge.weight >= 0.4 {
                    Color::Yellow
                } else {
                    Color::Red
                };
                let target_info = target
                    .map(|t| format!(" {} \"{}\"", t.short_id, truncate(&t.content, 25)))
                    .unwrap_or_else(|| " ???".to_string());
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("   --{:.2}-->", edge.weight),
                        Style::default().fg(weight_color),
                    ),
                    Span::styled(target_info, Style::default().fg(state.theme.fg())),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(" strong ", Style::default().fg(Color::Green)),
            Span::styled(
                format!("{} ", state.graph_stats.strong_edges),
                Style::default().fg(Color::Green),
            ),
            Span::styled(" medium ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{} ", state.graph_stats.medium_edges),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled(" weak ", Style::default().fg(Color::Red)),
            Span::styled(
                format!("{}", state.graph_stats.weak_edges),
                Style::default().fg(Color::Red),
            ),
        ]));

        f.render_widget(Paragraph::new(lines), right_inner);
    }
}

fn render_graph_map(f: &mut Frame, area: Rect, state: &AppState) {
    // Three-column layout: Top Entities | Selected Focus | Type Summary
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(28), // Left: Top entities
            Constraint::Min(40),    // Center: Focus view
            Constraint::Length(24), // Right: Type summary
        ])
        .split(area);

    render_graph_top_entities(f, main_chunks[0], state);
    render_graph_focus_view(f, main_chunks[1], state);
    render_graph_type_summary(f, main_chunks[2], state);
}

/// Left panel: Top entities sorted by connections
fn render_graph_top_entities(f: &mut Frame, area: Rect, state: &AppState) {
    let node_count = state.graph_data.nodes.len();
    let selected_idx = state.graph_data.selected_node;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            " TOP ENTITIES ",
            Style::default()
                .fg(Color::Cyan)
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
        let name = truncate(&node.content, 15);

        let style = if is_selected {
            Style::default()
                .fg(state.theme.accent())
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(state.theme.fg())
        };

        let conn_color = if node.connections >= 10 {
            Color::Green
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

/// Center panel: Selected entity with radial connections
fn render_graph_focus_view(f: &mut Frame, area: Rect, state: &AppState) {
    let selected_name = state
        .graph_data
        .selected()
        .map(|n| truncate(&n.content, 20))
        .unwrap_or_else(|| "None".to_string());

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Black))
        .title(Span::styled(
            format!(" {} ", selected_name),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                " j/k=select Enter=expand ",
                Style::default().fg(Color::DarkGray),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(selected) = state.graph_data.selected() else {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  Select an entity from the list",
                Style::default().fg(Color::DarkGray),
            )),
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
            Color::Green
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
            "Person" => Color::Cyan,
            "Organization" => Color::Blue,
            "Location" => Color::Green,
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
        Span::styled(" Strong ", Style::default().fg(Color::Green)),
        Span::styled(
            format!("{}", strong),
            Style::default()
                .fg(Color::Green)
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
            Style::default().fg(Color::Cyan),
        ),
    ]));

    // Avg weight
    lines.push(Line::from(vec![
        Span::styled(" Avg wt:  ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.2}", state.graph_stats.avg_weight),
            Style::default().fg(Color::Cyan),
        ),
    ]));

    f.render_widget(Paragraph::new(lines), inner);
}

pub fn render_footer(f: &mut Frame, area: Rect, state: &AppState) {
    let view_name = match state.view_mode {
        ViewMode::Dashboard => "Dashboard",
        ViewMode::Projects => "Projects",
        ViewMode::ActivityLogs => "Activity",
        ViewMode::GraphList => "Graph",
        ViewMode::GraphMap => "Map",
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
                    .fg(Color::Cyan)
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
            .border_style(Style::default().fg(Color::Green))
            .title(Span::styled(
                format!(" {} RESULTS ", state.search_results.len()),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ));
        let result_line = Line::from(vec![
            Span::styled(
                " j/k ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("navigate ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " / ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("new search ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " Esc ",
                Style::default()
                    .fg(Color::Cyan)
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
    let is_graph_view = matches!(state.view_mode, ViewMode::GraphList | ViewMode::GraphMap);

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
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("quit ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "d ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("dash ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "a ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("activity ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "g ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("graph ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "m ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("map ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "j/k ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("↑↓ ", Style::default().fg(Color::DarkGray)),
    ];

    // Add graph-specific controls
    if is_graph_view {
        keys.extend([
            Span::styled(
                "r ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("rebuild ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "R ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("refresh ", Style::default().fg(Color::DarkGray)),
        ]);
    } else {
        keys.extend([
            Span::styled(
                "+/- ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("zoom ", Style::default().fg(Color::DarkGray)),
        ]);
    }

    // Theme toggle
    let theme_label = match state.theme {
        crate::types::Theme::Dark => "◐",
        crate::types::Theme::Light => "◑",
    };
    keys.extend([
        Span::styled(
            "t ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(theme_label, Style::default().fg(state.theme.accent())),
        Span::raw(" "),
    ]);

    keys.extend([
        Span::raw(" "),
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
