use crate::logo::{ELEPHANT, ELEPHANT_GRADIENT, SHODH_GRADIENT, SHODH_TEXT};
use crate::types::{AppState, DisplayEvent, ViewMode, VERSION};
use ratatui::{prelude::*, widgets::*};

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
        .border_style(Style::default().fg(Color::DarkGray));
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

    // Elephant logo with gradient
    let logo_lines: Vec<Line> = ELEPHANT
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let (r, g, b) = if state.connected {
                ELEPHANT_GRADIENT[i % ELEPHANT_GRADIENT.len()]
            } else {
                (80, 80, 80)
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

    // Right side: version, status, session
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Min(0),
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

    let status = if state.connected {
        Line::from(Span::styled(
            "● LIVE",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ))
    } else {
        Line::from(Span::styled(
            "○ ...",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::SLOW_BLINK),
        ))
    };
    f.render_widget(
        Paragraph::new(status).alignment(Alignment::Right),
        right_chunks[1],
    );

    let session = Line::from(Span::styled(
        state.session_duration(),
        Style::default().fg(Color::White),
    ));
    f.render_widget(
        Paragraph::new(session).alignment(Alignment::Right),
        right_chunks[2],
    );
}

pub fn render_main(f: &mut Frame, area: Rect, state: &AppState) {
    match state.view_mode {
        ViewMode::Dashboard => render_dashboard(f, area, state),
        ViewMode::ActivityLogs => render_activity_logs(f, area, state),
        ViewMode::GraphList => render_graph_list(f, area, state),
        ViewMode::GraphMap => render_graph_map(f, area, state),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VIEW 1: DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════

pub fn render_dashboard(f: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(34), Constraint::Min(40)])
        .split(area);
    render_stats_panel(f, chunks[0], state);
    render_activity_feed(f, chunks[1], state);
}

fn render_stats_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
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

fn render_activity_feed(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Activity ",
            Style::default()
                .fg(Color::Cyan)
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
        6u16
    } else {
        0u16
    };
    let list_height = inner.height.saturating_sub(detail_height);
    let max_visible = list_height as usize;

    let scroll_start = if let Some(sel) = state.selected_event {
        if sel < state.scroll_offset {
            sel
        } else if max_visible > 0 && sel >= state.scroll_offset + max_visible {
            sel.saturating_sub(max_visible.saturating_sub(1))
        } else {
            state.scroll_offset
        }
    } else {
        state.scroll_offset
    };

    let end = (scroll_start + max_visible).min(state.events.len());

    for (line_idx, (global_idx, event)) in state
        .events
        .iter()
        .enumerate()
        .skip(scroll_start)
        .take(end.saturating_sub(scroll_start))
        .enumerate()
    {
        let y = inner.y + line_idx as u16;
        if y >= inner.y + list_height {
            break;
        }

        let is_selected = state.selected_event == Some(global_idx);
        let is_newest = global_idx == 0;
        let color = event.event.event_color();
        let icon = event.event.event_icon();
        // Green circle for newest, arrow for selected, space otherwise
        let prefix = if is_newest {
            "●"
        } else if is_selected {
            "▶"
        } else {
            " "
        };
        let prefix_color = if is_newest { Color::Green } else { Color::Cyan };
        // Blue-tinted background for better contrast
        let bg = if is_selected {
            Color::Rgb(25, 40, 60)
        } else {
            Color::Reset
        };

        let preview = event
            .event
            .content_preview
            .as_ref()
            .map(|s| truncate(s, 35))
            .unwrap_or_default();
        let line = Line::from(vec![
            Span::styled(
                prefix,
                Style::default()
                    .fg(prefix_color)
                    .add_modifier(Modifier::BOLD)
                    .bg(bg),
            ),
            Span::styled(icon, Style::default().fg(color).bg(bg)),
            Span::styled(
                format!(" {:8} ", event.event.event_type),
                Style::default()
                    .fg(color)
                    .add_modifier(Modifier::BOLD)
                    .bg(bg),
            ),
            Span::styled(
                preview,
                Style::default()
                    .fg(if is_selected {
                        Color::White
                    } else {
                        Color::Gray
                    })
                    .bg(bg),
            ),
            Span::styled(
                format!(" {}", event.time_ago()),
                Style::default()
                    .fg(if is_selected {
                        Color::White
                    } else {
                        Color::DarkGray
                    })
                    .bg(bg),
            ),
        ]);
        f.render_widget(
            Paragraph::new(line),
            Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            },
        );
    }

    if let Some(sel_idx) = state.selected_event {
        if let Some(event) = state.events.get(sel_idx) {
            let detail_area = Rect {
                x: inner.x,
                y: inner.y + list_height,
                width: inner.width,
                height: detail_height,
            };
            render_event_detail(f, detail_area, event);
        }
    }

    if state.events.len() > max_visible {
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
    let is_new = event.received_at.elapsed().as_secs() < 3;
    let border_color = if is_new { color } else { Color::DarkGray };
    let title_style = if is_new && tick % 10 < 5 {
        Style::default()
            .fg(color)
            .add_modifier(Modifier::BOLD | Modifier::SLOW_BLINK)
    } else {
        Style::default().fg(color).add_modifier(Modifier::BOLD)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(format!(" {} {} ", icon, label), title_style))
        .title(
            block::Title::from(Span::styled(
                format!(" {} ", event.time_ago()),
                Style::default().fg(Color::DarkGray),
            ))
            .alignment(Alignment::Right),
        );
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    if let Some(preview) = &event.event.content_preview {
        lines.push(Line::from(Span::styled(
            truncate(preview, inner.width as usize - 2),
            Style::default().fg(Color::White),
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

fn render_event_detail(f: &mut Frame, area: Rect, event: &DisplayEvent) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Details ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let color = event.event.event_color();
    let mut lines = Vec::new();
    if let Some(content) = &event.event.content_preview {
        lines.push(Line::from(vec![
            Span::styled("Content: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                truncate(content, inner.width.saturating_sub(10) as usize),
                Style::default().fg(Color::White),
            ),
        ]));
    }
    let mut meta = Vec::new();
    meta.push(Span::styled(
        format!("{} ", event.event.event_type),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
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
    if let Some(c) = event.event.count {
        meta.push(Span::styled(
            format!("x{} ", c),
            Style::default().fg(Color::Yellow),
        ));
    }
    lines.push(Line::from(meta));
    let mut id_line = Vec::new();
    if let Some(id) = &event.event.memory_id {
        id_line.push(Span::styled(
            format!("id:{} ", id),
            Style::default().fg(Color::DarkGray),
        ));
    }
    id_line.push(Span::styled(
        format!("@ {}", event.event.timestamp.format("%H:%M:%S")),
        Style::default().fg(Color::DarkGray),
    ));
    lines.push(Line::from(id_line));
    if let Some(entities) = &event.event.entities {
        if !entities.is_empty() {
            let es = entities
                .iter()
                .take(8)
                .map(|e| e.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(Line::from(vec![
                Span::styled("entities: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    truncate(&es, inner.width.saturating_sub(12) as usize),
                    Style::default().fg(Color::Green),
                ),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn render_activity_logs(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
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
        6u16
    } else {
        0u16
    };
    let list_height = inner.height.saturating_sub(detail_height);
    let event_height = 5u16;
    let max_visible = (list_height / event_height) as usize;

    let scroll_start = if let Some(sel) = state.selected_event {
        if sel < state.scroll_offset {
            sel
        } else if max_visible > 0 && sel >= state.scroll_offset + max_visible {
            sel.saturating_sub(max_visible.saturating_sub(1))
        } else {
            state.scroll_offset
        }
    } else {
        state.scroll_offset
    };

    let end = (scroll_start + max_visible).min(state.events.len());
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
            render_river_event_selectable(f, event_area, event, is_selected, is_newest);
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
            render_event_detail(f, detail_area, event);
        }
    }

    if state.events.len() > max_visible {
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
    event: &DisplayEvent,
    is_selected: bool,
    is_newest: bool,
) {
    let color = event.event.event_color();
    let icon = event.event.event_icon();
    let sep = "-".repeat(area.width.saturating_sub(27) as usize);
    // Blue-tinted background for better contrast
    let bg = if is_selected {
        Color::Rgb(25, 40, 60)
    } else {
        Color::Reset
    };
    // Green circle for newest, arrow for selected
    let (prefix, prefix_color) = if is_newest {
        ("● ", Color::Green)
    } else if is_selected {
        ("▶ ", Color::Cyan)
    } else {
        ("  ", Color::Cyan)
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
            Style::default().fg(Color::DarkGray).bg(bg),
        ),
        Span::styled("-+- ", Style::default().fg(Color::DarkGray).bg(bg)),
        Span::styled(
            icon,
            Style::default()
                .fg(color)
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        ),
        Span::styled(
            format!(" {} ", event.event.event_type),
            Style::default()
                .fg(color)
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        ),
        Span::styled(sep, Style::default().fg(Color::DarkGray).bg(bg)),
    ])];

    if let Some(preview) = &event.event.content_preview {
        lines.push(Line::from(vec![
            Span::styled("      ", Style::default().bg(bg)),
            Span::styled("| ", Style::default().fg(Color::DarkGray).bg(bg)),
            Span::styled(
                truncate(preview, area.width.saturating_sub(10) as usize),
                Style::default()
                    .fg(if is_selected {
                        Color::White
                    } else {
                        Color::Gray
                    })
                    .bg(bg),
            ),
        ]));
    }

    let mut meta = vec![
        Span::styled("      ", Style::default().bg(bg)),
        Span::styled("| ", Style::default().fg(Color::DarkGray).bg(bg)),
    ];
    if let Some(t) = &event.event.memory_type {
        meta.push(Span::styled(
            format!("type:{} ", t),
            Style::default().fg(Color::Cyan).bg(bg),
        ));
    }
    if let Some(m) = &event.event.retrieval_mode {
        meta.push(Span::styled(
            format!("mode:{} ", m),
            Style::default().fg(Color::Magenta).bg(bg),
        ));
    }
    if let Some(l) = event.event.latency_ms {
        meta.push(Span::styled(
            format!("{:.0}ms ", l),
            Style::default().fg(Color::Yellow).bg(bg),
        ));
    }
    if let Some(id) = &event.event.memory_id {
        meta.push(Span::styled(
            format!("id:{}", &id[..8.min(id.len())]),
            Style::default().fg(Color::DarkGray).bg(bg),
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
                Span::styled("      ", Style::default().bg(bg)),
                Span::styled("| ", Style::default().fg(Color::DarkGray).bg(bg)),
                Span::styled("entities: ", Style::default().fg(Color::DarkGray).bg(bg)),
                Span::styled(truncate(&es, 40), Style::default().fg(Color::Green).bg(bg)),
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
    let left_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " NODES ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" {} ", state.graph_stats.nodes),
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
        let mut lines = Vec::new();
        for (i, node) in state.graph_data.nodes.iter().enumerate() {
            let is_selected = i == state.graph_data.selected_node;
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
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    truncate(&node.content, left_inner.width.saturating_sub(4) as usize),
                    Style::default().fg(if is_selected {
                        Color::White
                    } else {
                        Color::Gray
                    }),
                ),
            ]));
        }
        f.render_widget(Paragraph::new(lines), left_inner);
    }

    // RIGHT: Edges from selected node
    let right_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
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
                Style::default().fg(Color::White),
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
                    Span::styled(target_info, Style::default().fg(Color::White)),
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
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(6)])
        .split(area);

    // TOP: 2D Graph visualization
    let map_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " GRAPH MAP ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            block::Title::from(Span::styled(
                format!(" density:{:.2} ", state.graph_stats.density),
                Style::default().fg(Color::Yellow),
            ))
            .alignment(Alignment::Right),
        );
    let map_inner = map_block.inner(chunks[0]);
    f.render_widget(map_block, chunks[0]);

    if state.graph_data.nodes.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  No nodes to display",
                Style::default().fg(Color::DarkGray),
            )),
            map_inner,
        );
    } else {
        // Create a simple 2D grid representation
        let width = map_inner.width as usize;
        let height = map_inner.height as usize;
        let mut grid: Vec<Vec<(char, Color)>> = vec![vec![(' ', Color::DarkGray); width]; height];

        // Place nodes on grid based on their x,y positions
        for (i, node) in state.graph_data.nodes.iter().enumerate() {
            let x = ((node.x * (width - 4) as f32) as usize + 2).min(width - 3);
            let y = ((node.y * (height - 2) as f32) as usize + 1).min(height - 2);
            let is_selected = i == state.graph_data.selected_node;
            let symbol = if is_selected { '^' } else { 'o' };
            let color = match node.memory_type.to_lowercase().as_str() {
                "learning" => Color::Green,
                "context" => Color::Cyan,
                "decision" => Color::Yellow,
                "error" => Color::Red,
                "task" => Color::Blue,
                _ => Color::White,
            };
            if y < height && x < width {
                grid[y][x] = (symbol, if is_selected { Color::Yellow } else { color });
            }
        }

        // Draw edges as lines between connected nodes
        for edge in &state.graph_data.edges {
            if let (Some(from), Some(to)) = (
                state.graph_data.nodes.iter().find(|n| n.id == edge.from_id),
                state.graph_data.nodes.iter().find(|n| n.id == edge.to_id),
            ) {
                let x1 = ((from.x * (width - 4) as f32) as usize + 2).min(width - 3);
                let y1 = ((from.y * (height - 2) as f32) as usize + 1).min(height - 2);
                let x2 = ((to.x * (width - 4) as f32) as usize + 2).min(width - 3);
                let y2 = ((to.y * (height - 2) as f32) as usize + 1).min(height - 2);

                // Simple line drawing (just midpoint for now)
                let mx = (x1 + x2) / 2;
                let my = (y1 + y2) / 2;
                if my < height && mx < width && grid[my][mx].0 == ' ' {
                    let edge_color = if edge.weight >= 0.7 {
                        Color::Green
                    } else if edge.weight >= 0.4 {
                        Color::Yellow
                    } else {
                        Color::DarkGray
                    };
                    grid[my][mx] = ('.', edge_color);
                }
            }
        }

        // Render grid
        let lines: Vec<Line> = grid
            .iter()
            .map(|row| {
                Line::from(
                    row.iter()
                        .map(|(c, color)| Span::styled(c.to_string(), Style::default().fg(*color)))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        f.render_widget(Paragraph::new(lines), map_inner);
    }

    // BOTTOM: Selected node info
    let info_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Selected ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    let info_inner = info_block.inner(chunks[1]);
    f.render_widget(info_block, chunks[1]);

    if let Some(node) = state.graph_data.selected() {
        let lines = vec![
            Line::from(vec![
                Span::styled(
                    &node.short_id,
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" | {} | {} connections", node.memory_type, node.connections),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(Span::styled(
                &node.content,
                Style::default().fg(Color::White),
            )),
            Line::from(vec![
                Span::styled("pos: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("({:.2}, {:.2})", node.x, node.y),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
        ];
        f.render_widget(Paragraph::new(lines), info_inner);
    }
}

pub fn render_footer(f: &mut Frame, area: Rect, state: &AppState) {
    let view_name = match state.view_mode {
        ViewMode::Dashboard => "Dashboard",
        ViewMode::ActivityLogs => "Logs",
        ViewMode::GraphList => "Graph List",
        ViewMode::GraphMap => "Graph Map",
    };

    let keys = vec![
        Span::styled(
            " q ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("quit ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            " d ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("dashboard ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            " a ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("activity ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            " g ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("graph ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            " m ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("map ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            " j/k ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("scroll ", Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(
            format!("[{}]", view_name),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    f.render_widget(Paragraph::new(Line::from(keys)).block(block), area);
}
