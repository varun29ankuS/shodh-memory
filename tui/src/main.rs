use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, SetTitle,
    },
};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::{
    env,
    io::{self, Read as IoRead, Write},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

mod logo;
mod stream;
mod types;
mod widgets;

use logo::{ELEPHANT, ELEPHANT_FRAMES, SHODH_GRADIENT, SHODH_TEXT};
use stream::{
    complete_todo, next_status, refresh_todos, reorder_todo, update_todo_priority,
    update_todo_status, MemoryStream,
};
use types::{AppState, FocusPanel, SearchMode, ViewMode};
use widgets::{render_footer, render_header, render_main};

enum TuiExitAction {
    Quit,
    SwitchUser,
}

/// Set the terminal window title
fn set_terminal_title(title: &str) {
    let _ = execute!(io::stdout(), SetTitle(title));
}

/// Generate dynamic title based on current view
fn generate_title(state: &AppState) -> String {
    let view_name = match state.view_mode {
        ViewMode::Dashboard => "Dashboard",
        ViewMode::Projects => "Projects",
        ViewMode::ActivityLogs => "Activity",
        ViewMode::GraphMap => "Graph",
    };

    format!("🦣 Shodh - {}", view_name)
}

struct UserSelector {
    users: Vec<String>,
    selected: usize,
    loading: bool,
    error: Option<String>,
}

impl UserSelector {
    fn new() -> Self {
        Self {
            users: vec![],
            selected: 0,
            loading: true,
            error: None,
        }
    }
    fn select_next(&mut self) {
        if !self.users.is_empty() {
            self.selected = (self.selected + 1) % self.users.len();
        }
    }
    fn select_prev(&mut self) {
        if !self.users.is_empty() {
            self.selected = self.selected.checked_sub(1).unwrap_or(self.users.len() - 1);
        }
    }
    fn selected_user(&self) -> Option<&String> {
        self.users.get(self.selected)
    }
}

async fn fetch_users(base_url: &str, api_key: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/users", base_url);
    match client.get(&url).header("X-API-Key", api_key).send().await {
        Ok(resp) => {
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(format!(
                    "Server error {}: {}",
                    status.as_u16(),
                    body.chars().take(100).collect::<String>()
                ));
            }
            resp.json::<Vec<String>>()
                .await
                .map_err(|e| format!("Parse error: {}", e))
        }
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("Connection refused") || err_str.contains("connection refused") {
                Err(format!(
                    "Server not running at {}. Start with: shodh-memory-server",
                    base_url
                ))
            } else if err_str.contains("timed out") || err_str.contains("timeout") {
                Err(format!("Server at {} not responding (timeout)", base_url))
            } else {
                Err(format!("Cannot connect to {}: {}", base_url, err_str))
            }
        }
    }
}

fn render_user_selector(f: &mut Frame, selector: &UserSelector) {
    let area = f.area();

    // Dark background
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Rgb(15, 15, 20))),
        area,
    );

    // Center everything with generous margins
    let outer = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(15),
            Constraint::Percentage(70),
            Constraint::Percentage(15),
        ])
        .split(area);

    let center = outer[1];

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Top padding
            Constraint::Length(6), // SHODH text logo
            Constraint::Length(1), // Spacing
            Constraint::Length(6), // Elephant logo
            Constraint::Length(1), // Spacing
            Constraint::Length(2), // Tagline
            Constraint::Length(2), // Spacing
            Constraint::Length(1), // "Select Profile" header
            Constraint::Length(1), // Spacing
            Constraint::Min(6),    // User list
            Constraint::Length(1), // Spacing
            Constraint::Length(3), // Footer
        ])
        .split(center);

    // SHODH text logo with gradient
    let shodh_lines: Vec<Line> = SHODH_TEXT
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let (r, g, b) = SHODH_GRADIENT[i % SHODH_GRADIENT.len()];
            Line::from(Span::styled(*l, Style::default().fg(Color::Rgb(r, g, b))))
        })
        .collect();
    f.render_widget(
        Paragraph::new(shodh_lines).alignment(Alignment::Center),
        chunks[1],
    );

    // Elephant logo with orange gradient
    let logo_lines: Vec<Line> = ELEPHANT
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let g = [
                (255, 140, 50),
                (255, 120, 40),
                (255, 100, 30),
                (255, 85, 20),
                (255, 70, 10),
                (200, 55, 5),
            ];
            Line::from(Span::styled(
                *l,
                Style::default().fg(Color::Rgb(g[i % 6].0, g[i % 6].1, g[i % 6].2)),
            ))
        })
        .collect();
    f.render_widget(
        Paragraph::new(logo_lines).alignment(Alignment::Center),
        chunks[3],
    );

    // Tagline
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "Cognitive Memory for ",
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                "AI Agents",
                Style::default()
                    .fg(Color::Rgb(255, 140, 50))
                    .add_modifier(Modifier::BOLD),
            ),
        ]))
        .alignment(Alignment::Center),
        chunks[5],
    );

    // Select Profile header
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "─── Select Profile ───",
            Style::default().fg(Color::Rgb(100, 100, 120)),
        )))
        .alignment(Alignment::Center),
        chunks[7],
    );

    // User list
    let list_area = chunks[9];
    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(60, 60, 80)))
        .border_type(ratatui::widgets::BorderType::Rounded);

    let inner = list_block.inner(list_area);
    f.render_widget(list_block, list_area);

    if selector.loading {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("◐ ", Style::default().fg(Color::Rgb(255, 140, 50))),
                Span::styled("Connecting...", Style::default().fg(Color::DarkGray)),
            ]))
            .alignment(Alignment::Center),
            inner,
        );
    } else if let Some(ref e) = selector.error {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("✗ ", Style::default().fg(Color::Red)),
                Span::styled(e.as_str(), Style::default().fg(Color::Red)),
            ]))
            .alignment(Alignment::Center),
            inner,
        );
    } else if selector.users.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No profiles found",
                Style::default().fg(Color::DarkGray),
            )))
            .alignment(Alignment::Center),
            inner,
        );
    } else {
        let items: Vec<ListItem> = selector
            .users
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let is_selected = i == selector.selected;
                if is_selected {
                    ListItem::new(Line::from(vec![
                        Span::styled("  ▶ ", Style::default().fg(Color::Rgb(255, 140, 50))),
                        Span::styled("🦣 ", Style::default()),
                        Span::styled(
                            u,
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]))
                } else {
                    ListItem::new(Line::from(vec![
                        Span::raw("    "),
                        Span::styled("   ", Style::default()),
                        Span::styled(u, Style::default().fg(Color::Rgb(120, 120, 140))),
                    ]))
                }
            })
            .collect();
        f.render_widget(List::new(items), inner);
    }

    // Footer with keybindings
    let footer_block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::Rgb(50, 50, 60)));
    let footer_inner = footer_block.inner(chunks[11]);
    f.render_widget(footer_block, chunks[11]);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " ↑↓ ",
                Style::default()
                    .fg(Color::Rgb(255, 140, 50))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("navigate  ", Style::default().fg(Color::Rgb(80, 80, 100))),
            Span::styled(
                " Enter ",
                Style::default()
                    .fg(Color::Rgb(255, 140, 50))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("select  ", Style::default().fg(Color::Rgb(80, 80, 100))),
            Span::styled(
                " q ",
                Style::default()
                    .fg(Color::Rgb(255, 140, 50))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("quit", Style::default().fg(Color::Rgb(80, 80, 100))),
        ]))
        .alignment(Alignment::Center),
        footer_inner,
    );
}

fn render_splash(f: &mut Frame, progress: f32, tick: u64) {
    let area = f.area();

    // Dark background
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Rgb(15, 15, 20))),
        area,
    );

    // Center everything
    let outer = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(60),
            Constraint::Percentage(20),
        ])
        .split(area);

    let center = outer[1];

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Length(6), // SHODH text
            Constraint::Length(1),
            Constraint::Length(6), // Elephant
            Constraint::Length(2),
            Constraint::Length(1), // Tagline
            Constraint::Length(3),
            Constraint::Length(1), // Loading bar
            Constraint::Length(3),
            Constraint::Length(1), // Made in India
            Constraint::Percentage(15),
        ])
        .split(center);

    // SHODH text with animated gradient
    let shodh_lines: Vec<Line> = SHODH_TEXT
        .iter()
        .enumerate()
        .map(|(i, l)| {
            // Animate brightness based on tick
            let phase = ((tick as f32 * 0.1) + i as f32 * 0.5).sin() * 0.3 + 0.7;
            let (r, g, b) = SHODH_GRADIENT[i % SHODH_GRADIENT.len()];
            let r = (r as f32 * phase) as u8;
            let g = (g as f32 * phase) as u8;
            let b = (b as f32 * phase) as u8;
            Line::from(Span::styled(*l, Style::default().fg(Color::Rgb(r, g, b))))
        })
        .collect();
    f.render_widget(
        Paragraph::new(shodh_lines).alignment(Alignment::Center),
        chunks[1],
    );

    // Animated elephant logo with trunk waving
    let frame_idx = (tick as usize / 8) % ELEPHANT_FRAMES.len(); // Change frame every 8 ticks
    let current_frame = ELEPHANT_FRAMES[frame_idx];

    let logo_lines: Vec<Line> = current_frame
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let phase = ((tick as f32 * 0.15) + i as f32 * 0.3).sin() * 0.2 + 0.8;
            let base = [
                (255, 140, 50),
                (255, 120, 40),
                (255, 100, 30),
                (255, 85, 20),
                (255, 70, 10),
                (200, 55, 5),
            ];
            let (r, g, b) = base[i % 6];
            let r = (r as f32 * phase) as u8;
            let g = (g as f32 * phase) as u8;
            let b = (b as f32 * phase) as u8;
            Line::from(Span::styled(*l, Style::default().fg(Color::Rgb(r, g, b))))
        })
        .collect();
    f.render_widget(
        Paragraph::new(logo_lines).alignment(Alignment::Center),
        chunks[3],
    );

    // Tagline
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "Cognitive Memory for ",
                Style::default().fg(Color::Rgb(100, 100, 110)),
            ),
            Span::styled(
                "AI Agents",
                Style::default()
                    .fg(Color::Rgb(255, 140, 50))
                    .add_modifier(Modifier::BOLD),
            ),
        ]))
        .alignment(Alignment::Center),
        chunks[5],
    );

    // Progress bar - full width with gradient colors
    let bar_width = chunks[7].width.saturating_sub(12) as usize;
    let filled = (bar_width as f32 * progress) as usize;
    let empty = bar_width.saturating_sub(filled);

    let pct = (progress * 100.0) as u32;

    // Build progress bar with gradient (dark red -> orange -> yellow)
    let mut progress_spans = vec![Span::styled(
        format!(" {:>3}% ", pct),
        Style::default()
            .fg(Color::Rgb(255, 200, 100))
            .add_modifier(Modifier::BOLD),
    )];

    // Gradient colors for filled portion
    for i in 0..filled {
        let t = if bar_width > 0 {
            i as f32 / bar_width as f32
        } else {
            0.0
        };
        // Gradient: dark red (180,50,20) -> orange (255,140,50) -> yellow (255,220,100)
        let (r, g, b) = if t < 0.5 {
            let t2 = t * 2.0;
            (
                (180.0 + 75.0 * t2) as u8,
                (50.0 + 90.0 * t2) as u8,
                (20.0 + 30.0 * t2) as u8,
            )
        } else {
            let t2 = (t - 0.5) * 2.0;
            (255, (140.0 + 80.0 * t2) as u8, (50.0 + 50.0 * t2) as u8)
        };
        progress_spans.push(Span::styled("█", Style::default().fg(Color::Rgb(r, g, b))));
    }

    if filled < bar_width {
        progress_spans.push(Span::styled(
            "▓",
            Style::default().fg(Color::Rgb(120, 60, 20)),
        ));
        progress_spans.push(Span::styled(
            "░".repeat(empty.saturating_sub(1)),
            Style::default().fg(Color::Rgb(40, 40, 50)),
        ));
    }

    f.render_widget(
        Paragraph::new(Line::from(progress_spans)).alignment(Alignment::Center),
        chunks[7],
    );

    // Made in India with proper horizontal tricolor flag
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("Made in ", Style::default().fg(Color::Rgb(80, 80, 90))),
            Span::styled(
                "I",
                Style::default()
                    .fg(Color::Rgb(255, 153, 51))
                    .add_modifier(Modifier::BOLD),
            ), // Saffron
            Span::styled(
                "n",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ), // White
            Span::styled(
                "d",
                Style::default()
                    .fg(Color::Rgb(19, 136, 8))
                    .add_modifier(Modifier::BOLD),
            ), // Green
            Span::styled(
                "i",
                Style::default()
                    .fg(Color::Rgb(255, 153, 51))
                    .add_modifier(Modifier::BOLD),
            ), // Saffron
            Span::styled(
                "a",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ), // White
        ]))
        .alignment(Alignment::Center),
        chunks[9],
    );
}

async fn run_splash(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    execute!(io::stdout(), SetTitle("🦣 Shodh - Loading..."))?;

    let start = Instant::now();
    let duration = Duration::from_secs(5);
    let mut tick: u64 = 0;

    // Clear and draw first frame immediately
    terminal.clear()?;

    loop {
        // Check if duration elapsed
        if start.elapsed() >= duration {
            break;
        }

        let progress = start.elapsed().as_secs_f32() / duration.as_secs_f32();
        terminal.draw(|f| render_splash(f, progress, tick))?;
        tick += 1;

        // Check for early exit with keyboard only (ignore mouse events)
        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => {
                    // Only exit on key press, not release
                    if key.kind == KeyEventKind::Press {
                        break;
                    }
                }
                Event::Mouse(_) => {
                    // Ignore mouse events - don't exit
                }
                _ => {}
            }
        }
    }

    Ok(())
}

async fn run_user_selector(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    base_url: &str,
    api_key: &str,
) -> Result<Option<String>> {
    execute!(io::stdout(), SetTitle("🦣 Shodh - Select Profile"))?;
    let mut selector = UserSelector::new();
    match fetch_users(base_url, api_key).await {
        Ok(u) => {
            selector.users = u;
            selector.loading = false;
        }
        Err(e) => {
            selector.error = Some(e);
            selector.loading = false;
        }
    }
    let result = loop {
        terminal.draw(|f| render_user_selector(f, &selector))?;
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break None,
                        KeyCode::Up | KeyCode::Char('k') => selector.select_prev(),
                        KeyCode::Down | KeyCode::Char('j') => selector.select_next(),
                        KeyCode::Enter => {
                            if let Some(u) = selector.selected_user() {
                                break Some(u.clone());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    };
    Ok(result)
}

/// Status line mode: reads JSON from stdin, posts to backend, outputs formatted status
/// This is called by Claude Code's statusLine configuration
async fn run_statusline() -> Result<()> {
    let base_url = env::var("SHODH_API_URL")
        .or_else(|_| env::var("SHODH_SERVER_URL"))
        .unwrap_or_else(|_| "http://127.0.0.1:3030".to_string());

    // Read JSON from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    // Parse JSON
    let json: serde_json::Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(_) => {
            // Fallback output if JSON parsing fails
            println!("Shodh Memory");
            return Ok(());
        }
    };

    // Extract fields
    let session_id = json
        .pointer("/session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let model = json
        .pointer("/model/display_name")
        .and_then(|v| v.as_str())
        .unwrap_or("Claude");
    let cwd = json
        .pointer("/workspace/current_dir")
        .or_else(|| json.pointer("/cwd"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let context_size = json
        .pointer("/context_window/context_window_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(200000);

    // Calculate token usage
    let (current_tokens, percent) =
        if let Some(usage) = json.pointer("/context_window/current_usage") {
            let input_tokens = usage
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cache_create = usage
                .get("cache_creation_input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cache_read = usage
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let total = input_tokens + cache_create + cache_read;
            let pct = if context_size > 0 {
                (total * 100 / context_size) as u8
            } else {
                0
            };
            (total, pct)
        } else {
            (0, 0)
        };

    // POST to shodh-memory backend (fire and forget)
    let client = reqwest::Client::new();
    let post_body = serde_json::json!({
        "session_id": session_id,
        "tokens_used": current_tokens,
        "tokens_budget": context_size,
        "current_dir": cwd,
        "model": model
    });

    // Spawn background task to POST (don't block status line output)
    let url = format!("{}/api/context_status", base_url);
    tokio::spawn(async move {
        let _ = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&post_body)
            .send()
            .await;
    });

    // Format token counts
    let tokens_fmt = if current_tokens > 1000 {
        format!("{}k", current_tokens / 1000)
    } else {
        current_tokens.to_string()
    };
    let size_fmt = if context_size > 1000 {
        format!("{}k", context_size / 1000)
    } else {
        context_size.to_string()
    };

    // Extract directory name
    let dir_name = std::path::Path::new(cwd)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // ANSI color codes based on usage
    let (color, reset) = if percent < 50 {
        ("\x1b[32m", "\x1b[0m") // Green
    } else if percent < 80 {
        ("\x1b[33m", "\x1b[0m") // Yellow
    } else {
        ("\x1b[31m", "\x1b[0m") // Red
    };

    // Output status line
    println!(
        "{}{}%{} {}/{} | {} | {}",
        color, percent, reset, tokens_fmt, size_fmt, model, dir_name
    );
    io::stdout().flush()?;

    // Give the background POST a moment to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Check for statusline subcommand
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "statusline" {
        return run_statusline().await;
    }

    let base_url = std::env::var("SHODH_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:3030".to_string())
        .trim_end_matches("/api/events")
        .to_string();
    let api_key = std::env::var("SHODH_API_KEY")
        .unwrap_or_else(|_| "sk-shodh-dev-local-testing-key".to_string());

    // Initialize terminal for splash and user selector
    // Note: Mouse capture disabled to allow native terminal text selection/copy
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    // Show splash screen
    run_splash(&mut terminal).await?;

    let mut show_splash = false;
    loop {
        // Run user selector (skip splash on switch-user re-entry)
        if show_splash {
            // Terminal already in alternate screen from previous run_tui cleanup
        }
        let user = match run_user_selector(&mut terminal, &base_url, &api_key).await? {
            Some(u) => u,
            None => {
                // Clean up terminal before exit
                disable_raw_mode()?;
                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                terminal.show_cursor()?;
                return Ok(());
            }
        };

        // Clean up terminal - run_tui will create its own session
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        drop(terminal);

        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut s = state.lock().await;
            s.current_user = user.clone();
        }

        let stream = MemoryStream::new(
            &base_url,
            &api_key,
            &user,
            Arc::clone(&state),
        );
        let h = tokio::spawn(async move {
            stream.run().await;
        });
        let action = run_tui(state).await?;
        h.abort();

        match action {
            TuiExitAction::Quit => return Ok(()),
            TuiExitAction::SwitchUser => {
                // Re-create terminal for user selector
                enable_raw_mode()?;
                execute!(io::stdout(), EnterAlternateScreen)?;
                terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
                show_splash = true;
            }
        }
    }
}

async fn run_tui(state: Arc<Mutex<AppState>>) -> Result<TuiExitAction> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();
    // Clone state for search API calls
    let search_state = Arc::clone(&state);
    let base_url = std::env::var("SHODH_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:3030".to_string())
        .trim_end_matches("/api/events")
        .to_string();
    let api_key = std::env::var("SHODH_API_KEY")
        .unwrap_or_else(|_| "sk-shodh-dev-local-testing-key".to_string());

    // Set initial title
    let mut last_title = {
        let g = state.lock().await;
        let title = generate_title(&g);
        set_terminal_title(&title);
        title
    };

    let mut exit_action = TuiExitAction::Quit;
    loop {
        {
            let g = state.lock().await;
            terminal.draw(|f| ui(f, &g))?;

            // Update title if changed
            let new_title = generate_title(&g);
            if new_title != last_title {
                set_terminal_title(&new_title);
                last_title = new_title;
            }
        }
        if crossterm::event::poll(tick_rate.saturating_sub(last_tick.elapsed()))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    let mut g = state.lock().await;

                    // Handle search mode input
                    if g.search_active {
                        match key.code {
                            KeyCode::Esc => {
                                g.cancel_search();
                            }
                            KeyCode::Enter => {
                                if !g.search_query.is_empty() {
                                    let query = g.search_query.clone();
                                    let mode = g.search_mode;
                                    let user_id = g.current_user.clone();
                                    g.search_loading = true;
                                    drop(g);

                                    // Execute search API call
                                    let results =
                                        execute_search(&base_url, &api_key, &user_id, &query, mode)
                                            .await;

                                    let mut g = search_state.lock().await;
                                    match results {
                                        Ok(r) => g.set_search_results(r),
                                        Err(e) => {
                                            g.set_error(format!("Search failed: {}", e));
                                            g.search_loading = false;
                                        }
                                    }
                                }
                            }
                            KeyCode::Tab => {
                                g.cycle_search_mode();
                            }
                            KeyCode::Backspace => {
                                g.search_query.pop();
                                g.schedule_search();
                            }
                            KeyCode::Up => {
                                if g.search_results_visible {
                                    g.search_select_prev();
                                }
                            }
                            KeyCode::Down => {
                                if g.search_results_visible {
                                    g.search_select_next();
                                }
                            }
                            KeyCode::Char(c) => {
                                if g.search_query.len() < 100 {
                                    g.search_query.push(c);
                                    g.schedule_search();
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }

                    // Handle search detail view
                    if g.search_detail_visible {
                        match key.code {
                            KeyCode::Esc | KeyCode::Backspace => {
                                g.search_detail_visible = false;
                            }
                            _ => {}
                        }
                        continue;
                    }

                    // Handle search results navigation
                    if g.search_results_visible {
                        match key.code {
                            KeyCode::Esc => {
                                g.search_results_visible = false;
                                g.search_results.clear();
                                g.search_active = false;
                            }
                            KeyCode::Enter => {
                                if !g.search_results.is_empty() {
                                    g.search_detail_visible = true;
                                }
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                g.search_select_prev();
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                g.search_select_next();
                            }
                            KeyCode::Char('/') => {
                                g.start_search();
                            }
                            _ => {}
                        }
                        continue;
                    }

                    // Handle file preview navigation when visible
                    if g.file_preview_visible {
                        match key.code {
                            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Backspace => {
                                g.close_file_preview();
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                g.file_preview_scroll_up();
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                // Estimate visible lines (~30 for typical terminal)
                                g.file_preview_scroll_down(30);
                            }
                            KeyCode::PageUp => {
                                for _ in 0..10 {
                                    g.file_preview_scroll_up();
                                }
                            }
                            KeyCode::PageDown => {
                                for _ in 0..10 {
                                    g.file_preview_scroll_down(30);
                                }
                            }
                            KeyCode::Home => {
                                g.file_preview_scroll = 0;
                            }
                            KeyCode::End => {
                                g.file_preview_scroll = g.file_preview_content.len().saturating_sub(30);
                            }
                            _ => {}
                        }
                        continue;
                    }

                    // Handle file popup navigation when visible
                    if g.file_popup_visible {
                        // Build tree to get current nodes for navigation
                        let tree_nodes: Vec<(bool, String, String)> = if let Some(pid) = g.selected_project_id() {
                            if let Some(files) = g.project_files.get(&pid) {
                                crate::widgets::get_tree_node_info(files, &g.expanded_folders)
                            } else {
                                vec![]
                            }
                        } else {
                            vec![]
                        };
                        let tree_len = tree_nodes.len();

                        match key.code {
                            KeyCode::Esc | KeyCode::Char('f') => {
                                g.file_popup_visible = false;
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                if g.selected_file > 0 {
                                    g.selected_file -= 1;
                                    // Adjust scroll if needed
                                    if g.selected_file < g.file_popup_scroll {
                                        g.file_popup_scroll = g.selected_file;
                                    }
                                }
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                if g.selected_file + 1 < tree_len {
                                    g.selected_file += 1;
                                    // Adjust scroll if needed (assuming ~20 visible lines)
                                    if g.selected_file >= g.file_popup_scroll + 20 {
                                        g.file_popup_scroll = g.selected_file.saturating_sub(19);
                                    }
                                }
                            }
                            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                                // Toggle folder if folder, open preview if file
                                if let Some((is_dir, path, absolute_path)) = tree_nodes.get(g.selected_file) {
                                    if *is_dir {
                                        g.toggle_folder(path);
                                    } else {
                                        // Find file info and open preview
                                        // Clone the file data to avoid borrow conflict
                                        let file_clone = g.selected_project_id()
                                            .and_then(|pid| g.project_files.get(&pid))
                                            .and_then(|files| files.iter().find(|f| f.path == *path))
                                            .cloned();

                                        if let Some(file) = file_clone {
                                            // Read file content
                                            let file_path = if absolute_path.is_empty() { path } else { absolute_path };
                                            match crate::stream::read_file_content(file_path, 2000) {
                                                Ok(content) => {
                                                    g.open_file_preview(&file, content);
                                                }
                                                Err(e) => {
                                                    g.set_error(format!("Cannot read '{}': {}", file_path, e));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            KeyCode::Left | KeyCode::Char('h') => {
                                // Collapse folder if selected item is a folder
                                if let Some((is_dir, folder_path, _)) = tree_nodes.get(g.selected_file) {
                                    if *is_dir && g.is_folder_expanded(folder_path) {
                                        g.toggle_folder(folder_path);
                                    }
                                }
                            }
                            KeyCode::Char('q') => break,
                            _ => {}
                        }
                        continue;
                    }

                    // Handle codebase path input
                    if g.codebase_input_active {
                        match key.code {
                            KeyCode::Esc => {
                                g.codebase_input_active = false;
                                g.codebase_input_path.clear();
                                g.codebase_input_project_id = None;
                            }
                            KeyCode::Enter => {
                                // Start scanning with entered path (spawn as background task)
                                if let Some(pid) = g.codebase_input_project_id.take() {
                                    let root_path = g.codebase_input_path.clone();
                                    let user_id = g.current_user.clone();
                                    g.codebase_input_active = false;
                                    g.codebase_input_path.clear();
                                    g.start_scanning(&pid);
                                    g.set_error(format!("Scanning: {}...", root_path));
                                    drop(g); // Release lock before spawning

                                    // Spawn background task so UI can continue updating
                                    let scan_state = Arc::clone(&state);
                                    let scan_base_url = base_url.clone();
                                    let scan_api_key = api_key.clone();
                                    let scan_pid = pid.clone();
                                    let scan_root = root_path.clone();
                                    tokio::spawn(async move {
                                        // Scan
                                        let scan_result = crate::stream::scan_project_codebase(
                                            &scan_base_url, &scan_api_key, &user_id, &scan_pid, &scan_root,
                                        )
                                        .await;

                                        let mut g = scan_state.lock().await;
                                        match scan_result {
                                            Ok(count) => {
                                                g.set_error(format!("Scanned {} files, indexing...", count));
                                                drop(g);

                                                // Index
                                                let index_result = crate::stream::index_project_codebase(
                                                    &scan_base_url, &scan_api_key, &user_id, &scan_pid, &scan_root,
                                                )
                                                .await;

                                                let mut g = scan_state.lock().await;
                                                g.stop_scanning();
                                                match index_result {
                                                    Ok(indexed) => {
                                                        g.set_error(format!(
                                                            "✓ Indexed {} files. Press f to view.",
                                                            indexed
                                                        ));
                                                        g.mark_project_indexed(&scan_pid);
                                                        g.project_files.remove(&scan_pid);
                                                        g.files_expanded_projects.insert(scan_pid.clone());
                                                    }
                                                    Err(e) => {
                                                        g.set_error(format!("Index failed: {}", e));
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                g.stop_scanning();
                                                g.set_error(format!("Scan failed: {}", e));
                                            }
                                        }
                                    });
                                }
                            }
                            KeyCode::Backspace => {
                                g.codebase_input_path.pop();
                            }
                            KeyCode::Char(c) => {
                                g.codebase_input_path.push(c);
                            }
                            _ => {}
                        }
                        continue;
                    }

                    // Ctrl+U: switch user
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('u')
                    {
                        exit_action = TuiExitAction::SwitchUser;
                        break;
                    }

                    // Normal mode keybindings
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Esc => {
                            // Exit detail panel focus first, then clear event selection, then quit
                            if g.focus_panel == FocusPanel::Detail {
                                // In Projects, go back to Right panel (todos); in Dashboard, go to Left
                                if matches!(g.view_mode, ViewMode::Projects) {
                                    g.focus_panel = FocusPanel::Right;
                                } else {
                                    g.focus_panel = FocusPanel::Left;
                                }
                            } else if g.selected_event.is_some() {
                                g.clear_event_selection();
                            } else {
                                break;
                            }
                        }
                        KeyCode::Char('/') => {
                            g.start_search();
                        }
                        KeyCode::Char('c') => g.events.clear(),
                        KeyCode::Char('1') => g.set_view(ViewMode::Dashboard),
                        KeyCode::Char('2') => g.set_view(ViewMode::Projects),
                        KeyCode::Char('3') => g.set_view(ViewMode::ActivityLogs),
                        KeyCode::Char('4') => g.set_view(ViewMode::GraphMap),
                        KeyCode::Char('d') => g.set_view(ViewMode::Dashboard),
                        KeyCode::Char('p') => g.set_view(ViewMode::Projects),
                        KeyCode::Char('a') => g.set_view(ViewMode::ActivityLogs),
                        KeyCode::Char('g') => g.set_view(ViewMode::GraphMap),
                        KeyCode::Char('t') => g.toggle_theme(),
                        KeyCode::Char('e') => g.toggle_expand_sections(),
                        KeyCode::F(5) => {
                            // Manual refresh of todos and projects
                            let user_id = g.current_user.clone();
                            g.set_error("Refreshing...".to_string());
                            drop(g);

                            match refresh_todos(&base_url, &api_key, &user_id, &search_state).await
                            {
                                Ok(()) => {
                                    let mut g = search_state.lock().await;
                                    g.clear_error();
                                }
                                Err(e) => {
                                    let mut g = search_state.lock().await;
                                    g.set_error(format!("Refresh failed: {}", e));
                                }
                            }
                        }
                        KeyCode::Char('o') => {
                            // Toggle auto-rotate in graph map view
                            if matches!(g.view_mode, ViewMode::GraphMap) {
                                g.toggle_graph_auto_rotate();
                            }
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            g.zoom_in();
                        }
                        KeyCode::Char('-') => {
                            g.zoom_out();
                        }
                        // 3D Graph controls
                        KeyCode::Char('h') => {
                            if matches!(g.view_mode, ViewMode::GraphMap) {
                                g.rotate_graph_left();
                            }
                        }
                        KeyCode::Char('l') => {
                            if matches!(g.view_mode, ViewMode::GraphMap) {
                                g.rotate_graph_right();
                            }
                        }
                        KeyCode::Char('L') => {
                            // Trace lineage in Projects view
                            if matches!(g.view_mode, ViewMode::Projects) {
                                // Get memory_id from selected todo
                                let memory_id: Option<String> =
                                    if g.focus_panel == FocusPanel::Right {
                                        // Get selected todo's ID (use as memory ID for lineage)
                                        g.get_selected_dashboard_todo().map(|t| t.id.clone())
                                    } else {
                                        None
                                    };

                                if let Some(ref mem_id) = memory_id {
                                    let user_id = g.current_user.clone();
                                    g.set_error("Tracing lineage...".to_string());
                                    drop(g);

                                    // Fetch lineage trace
                                    let lineage_result = crate::stream::fetch_lineage_trace(
                                        &base_url, &api_key, &user_id, &mem_id, "backward", &state,
                                    )
                                    .await;

                                    let mut g = state.lock().await;
                                    match lineage_result {
                                        Ok(()) => {
                                            g.clear_error();
                                        }
                                        Err(e) => {
                                            g.set_error(format!("Lineage: {}", e));
                                        }
                                    }
                                } else {
                                    g.set_error("Select a todo to trace lineage".to_string());
                                }
                            }
                        }
                        KeyCode::Char('C') => {
                            // Confirm lineage edge in Projects view
                            if matches!(g.view_mode, ViewMode::Projects) {
                                if let Some(ref trace) = g.lineage_trace {
                                    // Get first inferred edge to confirm
                                    let edge_to_confirm = trace
                                        .edges
                                        .iter()
                                        .find(|e| e.source == "Inferred")
                                        .map(|e| e.id.clone());

                                    if let Some(edge_id) = edge_to_confirm {
                                        let user_id = g.current_user.clone();
                                        drop(g);

                                        let confirm_result = crate::stream::confirm_lineage_edge(
                                            &base_url, &api_key, &user_id, &edge_id,
                                        )
                                        .await;

                                        let mut g = state.lock().await;
                                        match confirm_result {
                                            Ok(msg) => {
                                                g.set_error(format!("✓ {}", msg));
                                            }
                                            Err(e) => {
                                                g.set_error(format!("Confirm failed: {}", e));
                                            }
                                        }
                                    } else {
                                        g.set_error("No inferred edges to confirm".to_string());
                                    }
                                } else {
                                    g.set_error("No lineage loaded. Press L first".to_string());
                                }
                            }
                        }
                        KeyCode::Char('X') => {
                            // Reject lineage edge in Projects view
                            if matches!(g.view_mode, ViewMode::Projects) {
                                if let Some(ref trace) = g.lineage_trace {
                                    // Get first inferred edge to reject
                                    let edge_to_reject = trace
                                        .edges
                                        .iter()
                                        .find(|e| e.source == "Inferred")
                                        .map(|e| e.id.clone());

                                    if let Some(edge_id) = edge_to_reject {
                                        let user_id = g.current_user.clone();
                                        drop(g);

                                        let reject_result = crate::stream::reject_lineage_edge(
                                            &base_url, &api_key, &user_id, &edge_id,
                                        )
                                        .await;

                                        let mut g = state.lock().await;
                                        match reject_result {
                                            Ok(msg) => {
                                                g.set_error(format!("✗ {}", msg));
                                            }
                                            Err(e) => {
                                                g.set_error(format!("Reject failed: {}", e));
                                            }
                                        }
                                    } else {
                                        g.set_error("No inferred edges to reject".to_string());
                                    }
                                } else {
                                    g.set_error("No lineage loaded. Press L first".to_string());
                                }
                            }
                        }
                        KeyCode::Char('<') | KeyCode::Char(',') => {
                            // Scroll lineage left in Projects view
                            if matches!(g.view_mode, ViewMode::Projects)
                                && g.lineage_trace.is_some()
                            {
                                g.lineage_scroll_left();
                            }
                        }
                        KeyCode::Char('>') | KeyCode::Char('.') => {
                            // Scroll lineage right in Projects view
                            if matches!(g.view_mode, ViewMode::Projects)
                                && g.lineage_trace.is_some()
                            {
                                g.lineage_scroll_right();
                            }
                        }
                        KeyCode::Char('f') => {
                            // Open file popup for indexed project in Projects view
                            if matches!(g.view_mode, ViewMode::Projects)
                                && g.focus_panel == FocusPanel::Left
                            {
                                if let Some(project_id) = g.selected_project_id() {
                                    if g.is_project_indexed(&project_id) {
                                        // Fetch files if not already loaded
                                        if g.get_project_files(&project_id).is_none()
                                            && !g.is_files_loading(&project_id)
                                        {
                                            g.start_files_loading(&project_id);
                                            let user_id = g.current_user.clone();
                                            let pid = project_id.clone();
                                            drop(g);

                                            let files_result = crate::stream::fetch_project_files(
                                                &base_url, &api_key, &user_id, &pid,
                                            )
                                            .await;

                                            let mut g = state.lock().await;
                                            match files_result {
                                                Ok(files) => {
                                                    g.set_project_files(&pid, files);
                                                    // Open popup after loading files
                                                    g.file_popup_visible = true;
                                                    g.file_popup_scroll = 0;
                                                }
                                                Err(e) => {
                                                    g.set_error(format!("Files: {}", e));
                                                    g.files_loading = None;
                                                }
                                            }
                                        } else {
                                            // Files already loaded, just open popup
                                            g.file_popup_visible = true;
                                            g.file_popup_scroll = 0;
                                        }
                                    } else {
                                        g.set_error("Press S to scan codebase first".to_string());
                                    }
                                }
                            }
                        }
                        KeyCode::Char('S') => {
                            // Open codebase path input for selected project (uppercase S)
                            // Only allow indexing for root projects (not sub-projects)
                            if matches!(g.view_mode, ViewMode::Projects)
                                && g.focus_panel == FocusPanel::Left
                            {
                                if let Some(project_id) = g.selected_project_id() {
                                    // Check if this is a root project (no parent)
                                    let is_root = g.projects.iter()
                                        .find(|p| p.id == project_id)
                                        .is_some_and(|p| p.parent_id.is_none());

                                    if is_root {
                                        // Pre-fill with current directory
                                        let default_path = std::env::current_dir()
                                            .map(|p| p.display().to_string())
                                            .unwrap_or_else(|_| ".".to_string());
                                        g.codebase_input_active = true;
                                        g.codebase_input_path = default_path;
                                        g.codebase_input_project_id = Some(project_id);
                                    } else {
                                        g.set_error("Indexing only available for root projects".to_string());
                                    }
                                }
                            }
                        }
                        KeyCode::Char('w') => {
                            if matches!(g.view_mode, ViewMode::GraphMap) {
                                g.tilt_graph(-0.1);
                            }
                        }
                        KeyCode::Char('s') => {
                            if matches!(g.view_mode, ViewMode::GraphMap) {
                                g.tilt_graph(0.1);
                            }
                        }
                        KeyCode::Char('r') => {
                            // Rebuild graph in graph view (lowercase r = rebuild)
                            if matches!(g.view_mode, ViewMode::GraphMap) {
                                let user_id = g.current_user.clone();
                                g.set_error("Rebuilding graph...".to_string());
                                drop(g);

                                // Trigger graph rebuild
                                let rebuild_result =
                                    rebuild_graph(&base_url, &api_key, &user_id).await;

                                let mut g = search_state.lock().await;
                                match rebuild_result {
                                    Ok(msg) => {
                                        g.set_error(msg.clone());
                                        drop(g);

                                        // Fetch fresh graph data
                                        let graph_result =
                                            fetch_graph_data(&base_url, &api_key, &user_id).await;

                                        let mut g = search_state.lock().await;
                                        match graph_result {
                                            Ok((nodes, edges, stats)) => {
                                                g.graph_data.nodes = nodes;
                                                g.graph_data.edges = edges;
                                                g.graph_stats = stats;
                                                g.graph_data.selected_node = 0;
                                            }
                                            Err(e) => {
                                                g.set_error(format!("Fetch failed: {}", e));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        g.set_error(format!("Rebuild failed: {}", e));
                                    }
                                }
                            }
                        }
                        KeyCode::Char('R') => {
                            // Refresh graph data without rebuilding (uppercase R)
                            if matches!(g.view_mode, ViewMode::GraphMap) {
                                let user_id = g.current_user.clone();
                                g.set_error("Refreshing graph...".to_string());
                                drop(g);

                                // Fetch fresh graph data
                                let graph_result =
                                    fetch_graph_data(&base_url, &api_key, &user_id).await;

                                let mut g = search_state.lock().await;
                                match graph_result {
                                    Ok((nodes, edges, stats)) => {
                                        let msg = format!(
                                            "Loaded {} entities, {} edges",
                                            nodes.len(),
                                            edges.len()
                                        );
                                        g.graph_data.nodes = nodes;
                                        g.graph_data.edges = edges;
                                        g.graph_stats = stats;
                                        g.graph_data.selected_node = 0;
                                        g.set_error(msg);
                                    }
                                    Err(e) => {
                                        g.set_error(format!("Refresh failed: {}", e));
                                    }
                                }
                            }
                        }
                        // Complete selected todo (mark as done)
                        KeyCode::Char('x') => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects) {
                                if let Some(todo) = g.get_selected_dashboard_todo() {
                                    let todo_id = todo.id.clone();
                                    let user_id = g.current_user.clone();
                                    drop(g);

                                    // Call API to complete todo
                                    match complete_todo(&base_url, &api_key, &user_id, &todo_id)
                                        .await
                                    {
                                        Ok(()) => {
                                            // SSE will auto-refresh todos
                                        }
                                        Err(e) => {
                                            let mut g = search_state.lock().await;
                                            g.set_error(format!("Complete failed: {}", e));
                                        }
                                    }
                                }
                            }
                        }
                        // Cycle todo status (backlog -> todo -> in_progress -> done)
                        KeyCode::Char(' ') => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects)
                                && g.focus_panel == FocusPanel::Left
                            {
                                if let Some(todo) = g.get_selected_dashboard_todo() {
                                    let todo_id = todo.id.clone();
                                    let current_status = todo.status.as_str();
                                    let new_status = next_status(current_status);
                                    let user_id = g.current_user.clone();
                                    drop(g);

                                    // Call API to update status
                                    match update_todo_status(
                                        &base_url, &api_key, &user_id, &todo_id, new_status,
                                    )
                                    .await
                                    {
                                        Ok(()) => {
                                            // SSE will auto-refresh todos
                                        }
                                        Err(e) => {
                                            let mut g = search_state.lock().await;
                                            g.set_error(format!("Status update failed: {}", e));
                                        }
                                    }
                                }
                            }
                        }
                        // Priority shortcuts: !=Urgent, @=High, #=Medium, $=Low
                        KeyCode::Char('!') => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects) {
                                if let Some(todo) = g.get_selected_dashboard_todo() {
                                    let todo_id = todo.id.clone();
                                    let user_id = g.current_user.clone();
                                    drop(g);
                                    let _ = update_todo_priority(
                                        &base_url, &api_key, &user_id, &todo_id, "urgent",
                                    )
                                    .await;
                                }
                            }
                        }
                        KeyCode::Char('@') => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects) {
                                if let Some(todo) = g.get_selected_dashboard_todo() {
                                    let todo_id = todo.id.clone();
                                    let user_id = g.current_user.clone();
                                    drop(g);
                                    let _ = update_todo_priority(
                                        &base_url, &api_key, &user_id, &todo_id, "high",
                                    )
                                    .await;
                                }
                            }
                        }
                        KeyCode::Char('#') => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects) {
                                if let Some(todo) = g.get_selected_dashboard_todo() {
                                    let todo_id = todo.id.clone();
                                    let user_id = g.current_user.clone();
                                    drop(g);
                                    let _ = update_todo_priority(
                                        &base_url, &api_key, &user_id, &todo_id, "medium",
                                    )
                                    .await;
                                }
                            }
                        }
                        KeyCode::Char('$') => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects) {
                                if let Some(todo) = g.get_selected_dashboard_todo() {
                                    let todo_id = todo.id.clone();
                                    let user_id = g.current_user.clone();
                                    drop(g);
                                    let _ = update_todo_priority(
                                        &base_url, &api_key, &user_id, &todo_id, "low",
                                    )
                                    .await;
                                }
                            }
                        }
                        // Reorder shortcuts: [ = move up, ] = move down
                        KeyCode::Char('[') => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects) {
                                if let Some(todo) = g.get_selected_dashboard_todo() {
                                    let todo_id = todo.id.clone();
                                    let user_id = g.current_user.clone();
                                    drop(g);
                                    let _ =
                                        reorder_todo(&base_url, &api_key, &user_id, &todo_id, "up")
                                            .await;
                                }
                            }
                        }
                        KeyCode::Char(']') => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects) {
                                if let Some(todo) = g.get_selected_dashboard_todo() {
                                    let todo_id = todo.id.clone();
                                    let user_id = g.current_user.clone();
                                    drop(g);
                                    let _ = reorder_todo(
                                        &base_url, &api_key, &user_id, &todo_id, "down",
                                    )
                                    .await;
                                }
                            }
                        }
                        KeyCode::Tab => {
                            // In Detail panel, Tab toggles between columns
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects)
                                && g.focus_panel == FocusPanel::Detail
                            {
                                g.toggle_detail_column();
                            } else if matches!(g.view_mode, ViewMode::GraphMap) {
                                g.toggle_graph_map_focus();
                            } else {
                                g.cycle_view();
                            }
                        }
                        KeyCode::Up | KeyCode::Char('k') => match g.view_mode {
                            ViewMode::Dashboard => match g.focus_panel {
                                FocusPanel::Left => g.dashboard_todo_up(),
                                FocusPanel::Right => g.select_event_prev(),
                                FocusPanel::Detail => g.detail_scroll_up(),
                            },
                            ViewMode::ActivityLogs => g.select_event_prev(),
                            ViewMode::Projects => match g.focus_panel {
                                FocusPanel::Left => {
                                    if g.projects_selected > 0 {
                                        g.projects_selected -= 1;
                                    }
                                }
                                FocusPanel::Right => g.right_panel_up(),
                                FocusPanel::Detail => g.detail_scroll_up(),
                            },
                            _ => g.scroll_up(),
                        },
                        KeyCode::Down | KeyCode::Char('j') => match g.view_mode {
                            ViewMode::Dashboard => {
                                match g.focus_panel {
                                    FocusPanel::Left => g.dashboard_todo_down(),
                                    FocusPanel::Right => g.select_event_next(),
                                    FocusPanel::Detail => {
                                        // Calculate max scroll based on selected todo
                                        let (max_notes, max_activity) =
                                            if let Some(todo) = g.get_selected_dashboard_todo() {
                                                let notes_lines = todo
                                                    .notes
                                                    .as_ref()
                                                    .map(|n| n.len() / 40 + 1)
                                                    .unwrap_or(0);
                                                let activity_count = todo.comments.len();
                                                (notes_lines, activity_count)
                                            } else {
                                                (0, 0)
                                            };
                                        g.detail_scroll_down(max_notes, max_activity);
                                    }
                                }
                            }
                            ViewMode::ActivityLogs => g.select_event_next(),
                            ViewMode::Projects => {
                                match g.focus_panel {
                                    FocusPanel::Left => {
                                        let total_items = g.left_panel_flat_count();
                                        if g.projects_selected < total_items.saturating_sub(1) {
                                            g.projects_selected += 1;
                                        }
                                    }
                                    FocusPanel::Right => g.right_panel_down(),
                                    FocusPanel::Detail => {
                                        // Calculate max scroll based on selected todo
                                        let (max_notes, max_activity) =
                                            if let Some(todo) = g.get_selected_dashboard_todo() {
                                                let notes_lines = todo
                                                    .notes
                                                    .as_ref()
                                                    .map(|n| n.len() / 40 + 1)
                                                    .unwrap_or(0);
                                                let activity_count = todo.comments.len();
                                                (notes_lines, activity_count)
                                            } else {
                                                (0, 0)
                                            };
                                        g.detail_scroll_down(max_notes, max_activity);
                                    }
                                }
                            }
                            _ => g.scroll_down(),
                        },
                        KeyCode::Left => {
                            if matches!(g.view_mode, ViewMode::Dashboard | ViewMode::Projects) {
                                match g.focus_panel {
                                    FocusPanel::Right => g.focus_panel = FocusPanel::Left,
                                    FocusPanel::Detail => {
                                        // In Detail, left/right toggle columns
                                        if g.detail_focus_column == 1 {
                                            g.detail_focus_column = 0;
                                        }
                                    }
                                    _ => {}
                                }
                            } else if matches!(g.view_mode, ViewMode::GraphMap) {
                                // Switch focus to entities panel
                                g.graph_map_focus = FocusPanel::Left;
                            }
                        }
                        KeyCode::Right => {
                            if matches!(g.view_mode, ViewMode::Dashboard) {
                                match g.focus_panel {
                                    FocusPanel::Left => {
                                        g.focus_panel = FocusPanel::Right;
                                        if g.selected_event.is_none() && !g.events.is_empty() {
                                            g.selected_event = Some(0);
                                        }
                                    }
                                    FocusPanel::Detail => {
                                        // In Detail, left/right toggle columns
                                        if g.detail_focus_column == 0 {
                                            g.detail_focus_column = 1;
                                        }
                                    }
                                    _ => {}
                                }
                            } else if matches!(g.view_mode, ViewMode::Projects) {
                                match g.focus_panel {
                                    FocusPanel::Left => {
                                        g.focus_panel = FocusPanel::Right;
                                        g.todos_selected = 0;
                                    }
                                    FocusPanel::Detail => {
                                        // In Detail, left/right toggle columns
                                        if g.detail_focus_column == 0 {
                                            g.detail_focus_column = 1;
                                        }
                                    }
                                    _ => {}
                                }
                            } else if matches!(g.view_mode, ViewMode::GraphMap) {
                                // Switch focus to connections panel
                                g.graph_map_focus = FocusPanel::Right;
                                g.selected_connection = 0;
                            }
                        }
                        KeyCode::Enter => match g.view_mode {
                            ViewMode::Dashboard => {
                                match g.focus_panel {
                                    FocusPanel::Left => {
                                        // Enter on a todo focuses the detail panel
                                        if g.get_selected_dashboard_todo().is_some() {
                                            g.focus_panel = FocusPanel::Detail;
                                            g.notes_scroll = 0;
                                            g.activity_scroll = 0;
                                            g.detail_focus_column = 0;
                                        }
                                    }
                                    FocusPanel::Right => {
                                        // Select event for detail view
                                        if g.selected_event.is_none() && !g.events.is_empty() {
                                            g.selected_event = Some(0);
                                        }
                                    }
                                    FocusPanel::Detail => {
                                        // Exit detail panel
                                        g.focus_panel = FocusPanel::Left;
                                    }
                                }
                            }
                            ViewMode::Projects => {
                                match g.focus_panel {
                                    FocusPanel::Left => {
                                        // Toggle expansion only if a project is selected
                                        if let Some(project_id) = g.selected_project_id() {
                                            g.toggle_project_expansion(&project_id);
                                        }
                                    }
                                    FocusPanel::Right => {
                                        // Enter on a todo focuses the detail panel
                                        if g.get_selected_dashboard_todo().is_some() {
                                            g.focus_panel = FocusPanel::Detail;
                                            g.notes_scroll = 0;
                                            g.activity_scroll = 0;
                                            g.detail_focus_column = 0;
                                        }
                                    }
                                    FocusPanel::Detail => {
                                        // Exit detail panel
                                        g.focus_panel = FocusPanel::Right;
                                    }
                                }
                            }
                            _ => {
                                if g.selected_event.is_none() && !g.events.is_empty() {
                                    g.selected_event = Some(0);
                                }
                            }
                        },
                        KeyCode::Backspace => g.clear_event_selection(),
                        KeyCode::PageUp => {
                            for _ in 0..5 {
                                match g.view_mode {
                                    ViewMode::Dashboard => {
                                        if g.focus_panel == FocusPanel::Detail {
                                            g.detail_scroll_up();
                                        } else {
                                            g.select_event_prev()
                                        }
                                    }
                                    ViewMode::ActivityLogs => g.select_event_prev(),
                                    ViewMode::Projects => match g.focus_panel {
                                        FocusPanel::Left => {
                                            if g.projects_selected > 0 {
                                                g.projects_selected -= 1;
                                            }
                                        }
                                        FocusPanel::Right => g.right_panel_up(),
                                        FocusPanel::Detail => {}
                                    },
                                    _ => g.scroll_up(),
                                }
                            }
                        }
                        KeyCode::PageDown => {
                            for _ in 0..5 {
                                match g.view_mode {
                                    ViewMode::Dashboard => {
                                        if g.focus_panel == FocusPanel::Detail {
                                            let (max_notes, max_activity) = if let Some(todo) =
                                                g.get_selected_dashboard_todo()
                                            {
                                                let notes_lines = todo
                                                    .notes
                                                    .as_ref()
                                                    .map(|n| n.len() / 40 + 1)
                                                    .unwrap_or(0);
                                                let activity_count = todo.comments.len();
                                                (notes_lines, activity_count)
                                            } else {
                                                (0, 0)
                                            };
                                            g.detail_scroll_down(max_notes, max_activity);
                                        } else {
                                            g.select_event_next()
                                        }
                                    }
                                    ViewMode::ActivityLogs => g.select_event_next(),
                                    ViewMode::Projects => match g.focus_panel {
                                        FocusPanel::Left => {
                                            let total_items = g.left_panel_flat_count();
                                            if g.projects_selected < total_items.saturating_sub(1) {
                                                g.projects_selected += 1;
                                            }
                                        }
                                        FocusPanel::Right => g.right_panel_down(),
                                        FocusPanel::Detail => {}
                                    },
                                    _ => g.scroll_down(),
                                }
                            }
                        }
                        KeyCode::Home => {
                            g.scroll_offset = 0;
                            match g.view_mode {
                                ViewMode::Dashboard | ViewMode::ActivityLogs => {
                                    if !g.events.is_empty() {
                                        g.selected_event = Some(0);
                                    }
                                }
                                ViewMode::Projects => match g.focus_panel {
                                    FocusPanel::Left => {
                                        g.projects_selected = 0;
                                        g.projects_scroll = 0;
                                    }
                                    FocusPanel::Right => {
                                        g.todos_selected = 0;
                                    }
                                    FocusPanel::Detail => {}
                                },
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        if last_tick.elapsed() >= tick_rate {
            let mut g = state.lock().await;
            g.tick();

            // Check if debounced search should execute
            if g.should_execute_search() {
                let query = g.search_query.clone();
                let mode = g.search_mode;
                let user_id = g.current_user.clone();
                g.mark_search_started();
                drop(g);

                // Execute search in background
                let results = execute_search(&base_url, &api_key, &user_id, &query, mode).await;

                let mut g = search_state.lock().await;
                match results {
                    Ok(r) => {
                        g.set_search_results(r);
                        g.search_results_visible = true;
                    }
                    Err(e) => {
                        g.set_error(format!("Search: {}", e));
                        g.search_loading = false;
                    }
                }
            }

            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(exit_action)
}

async fn execute_search(
    base_url: &str,
    api_key: &str,
    user_id: &str,
    query: &str,
    mode: SearchMode,
) -> Result<Vec<types::SearchResult>, String> {
    let client = reqwest::Client::new();

    let url = match mode {
        SearchMode::Keyword => format!("{}/api/list/{}?query={}", base_url, user_id, query),
        SearchMode::Semantic => format!("{}/api/recall", base_url),
        SearchMode::Date => format!("{}/api/recall/date", base_url),
    };

    match mode {
        SearchMode::Keyword => {
            // GET request for keyword search
            let resp = client
                .get(&url)
                .header("X-API-Key", api_key)
                .send()
                .await
                .map_err(|e| e.to_string())?;

            let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            parse_memory_list_response(data)
        }
        SearchMode::Semantic => {
            // POST request for semantic search
            let body = serde_json::json!({
                "user_id": user_id,
                "query": query,
                "mode": "semantic",
                "limit": 20
            });

            let resp = client
                .post(&url)
                .header("X-API-Key", api_key)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| e.to_string())?;

            let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            parse_recall_response(data)
        }
        SearchMode::Date => {
            // Parse date query - supports:
            // - "7d" / "7D" = last 7 days
            // - "2w" / "2W" = last 2 weeks
            // - "1m" / "1M" = last 1 month
            // - "2025-12-17" = from that date to now
            let now = chrono::Utc::now();
            let (start, end) = parse_date_query(query, now)?;

            let body = serde_json::json!({
                "user_id": user_id,
                "start": start.to_rfc3339(),
                "end": end.to_rfc3339(),
                "limit": 20
            });

            let resp = client
                .post(&url)
                .header("X-API-Key", api_key)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| e.to_string())?;

            let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
            parse_recall_response(data)
        }
    }
}

/// Parse date query string into start/end DateTime range
/// Supports: "7d" (days), "2w" (weeks), "1m" (months), or "YYYY-MM-DD" format
fn parse_date_query(
    query: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>), String> {
    let query = query.trim().to_lowercase();

    // Try relative format: Nd, Nw, Nm (days, weeks, months)
    if let Some(num_str) = query.strip_suffix('d') {
        let days: i64 = num_str.parse().map_err(|_| "Invalid number of days")?;
        let start = now - chrono::Duration::days(days);
        return Ok((start, now));
    }
    if let Some(num_str) = query.strip_suffix('w') {
        let weeks: i64 = num_str.parse().map_err(|_| "Invalid number of weeks")?;
        let start = now - chrono::Duration::weeks(weeks);
        return Ok((start, now));
    }
    if let Some(num_str) = query.strip_suffix('m') {
        let months: i64 = num_str.parse().map_err(|_| "Invalid number of months")?;
        let start = now - chrono::Duration::days(months * 30);
        return Ok((start, now));
    }

    // Try YYYY-MM-DD format
    if let Ok(date) = chrono::NaiveDate::parse_from_str(&query, "%Y-%m-%d") {
        if let Some(datetime) = date.and_hms_opt(0, 0, 0) {
            let start = datetime.and_utc();
            return Ok((start, now));
        }
    }

    // Try just a number as days
    if let Ok(days) = query.parse::<i64>() {
        let start = now - chrono::Duration::days(days);
        return Ok((start, now));
    }

    Err("Invalid date format. Use: 7d, 2w, 1m, or YYYY-MM-DD".to_string())
}

fn parse_memory_list_response(data: serde_json::Value) -> Result<Vec<types::SearchResult>, String> {
    let memories = data
        .get("memories")
        .and_then(|m| m.as_array())
        .ok_or("Invalid response format")?;

    let mut results = Vec::new();
    for mem in memories {
        let id = mem
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let content = mem
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let memory_type = mem
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();
        let created_at = mem
            .get("created_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(chrono::Utc::now);
        let tags = mem
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        results.push(types::SearchResult {
            id,
            content,
            memory_type,
            score: 1.0,
            created_at,
            tags,
        });
    }
    Ok(results)
}

async fn rebuild_graph(base_url: &str, api_key: &str, user_id: &str) -> Result<String, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/graph/{}/rebuild", base_url, user_id);

    let resp = client
        .post(&url)
        .header("X-API-Key", api_key)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let status = resp.status();
    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    if status.is_success() {
        let entities = data
            .get("entities_created")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let relationships = data
            .get("relationships_created")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        Ok(format!(
            "Graph rebuilt: {} entities, {} relationships",
            entities, relationships
        ))
    } else {
        let error = data
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");
        Err(error.to_string())
    }
}

async fn fetch_graph_data(
    base_url: &str,
    api_key: &str,
    user_id: &str,
) -> Result<
    (
        Vec<types::GraphNode>,
        Vec<types::GraphEdge>,
        types::GraphStats,
    ),
    String,
> {
    let client = reqwest::Client::new();

    // Fetch universe data (entities + connections)
    let url = format!("{}/api/graph/{}/universe", base_url, user_id);
    let resp = client
        .get(&url)
        .header("X-API-Key", api_key)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut stats = types::GraphStats::default();

    // Parse stars (entities)
    if let Some(stars) = data.get("stars").and_then(|s| s.as_array()) {
        for (i, star) in stars.iter().enumerate() {
            let id = star
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = star
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let entity_type = star
                .get("entity_type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mention_count = star
                .get("mention_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;

            // Filter out short/meaningless names
            if name.len() >= 3 && !name.chars().all(|c| c.is_lowercase()) {
                let short_id = if id.len() > 8 {
                    id[..8].to_string()
                } else {
                    id.clone()
                };

                // Position from server or generate
                let pos = star.get("position");
                let x = pos
                    .and_then(|p| p.get("x"))
                    .and_then(|v| v.as_f64())
                    .map(|v| (v as f32 / 200.0 + 0.5).clamp(0.1, 0.9))
                    .unwrap_or_else(|| {
                        let n = i as f32;
                        (n * 0.618).sin() * 0.35 + 0.5
                    });
                let y = pos
                    .and_then(|p| p.get("y"))
                    .and_then(|v| v.as_f64())
                    .map(|v| (v as f32 / 200.0 + 0.5).clamp(0.1, 0.9))
                    .unwrap_or_else(|| {
                        let n = i as f32;
                        (n * 0.618).cos() * 0.35 + 0.5
                    });
                let z = pos
                    .and_then(|p| p.get("z"))
                    .and_then(|v| v.as_f64())
                    .map(|v| (v as f32 / 200.0 + 0.5).clamp(0.1, 0.9))
                    .unwrap_or_else(|| {
                        let n = i as f32;
                        ((n * 0.3).sin() * 0.2 + 0.5).clamp(0.1, 0.9)
                    });

                nodes.push(types::GraphNode {
                    id,
                    short_id,
                    content: name,
                    memory_type: entity_type,
                    connections: mention_count,
                    x,
                    y,
                    z,
                });
            }
        }
    }

    stats.nodes = nodes.len() as u32;

    // Parse connections (edges)
    if let Some(connections) = data.get("connections").and_then(|c| c.as_array()) {
        for conn in connections {
            let from_id = conn
                .get("from_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let to_id = conn
                .get("to_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let strength = conn.get("strength").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;

            // Only add edge if both nodes exist
            if nodes.iter().any(|n| n.id == from_id) && nodes.iter().any(|n| n.id == to_id) {
                edges.push(types::GraphEdge {
                    from_id,
                    to_id,
                    weight: strength,
                });
            }
        }
    }

    stats.edges = edges.len() as u32;
    if !edges.is_empty() {
        stats.avg_weight = edges.iter().map(|e| e.weight).sum::<f32>() / edges.len() as f32;
    }
    let n = nodes.len() as f64;
    if n > 1.0 {
        let max_edges = n * (n - 1.0) / 2.0;
        stats.density = (edges.len() as f64 / max_edges) as f32;
    }

    Ok((nodes, edges, stats))
}

fn parse_recall_response(data: serde_json::Value) -> Result<Vec<types::SearchResult>, String> {
    let memories = data
        .get("memories")
        .and_then(|m| m.as_array())
        .ok_or("Invalid response format")?;

    let mut results = Vec::new();
    for mem in memories {
        let id = mem
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Fields are nested inside "experience" object
        let experience = mem.get("experience");
        let content = experience
            .and_then(|e| e.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let memory_type = experience
            .and_then(|e| e.get("memory_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();
        let score = mem
            .get("score")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(0.0);
        let created_at = mem
            .get("created_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(chrono::Utc::now);
        // Tags are inside experience.tags
        let tags = experience
            .and_then(|e| e.get("tags"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        results.push(types::SearchResult {
            id,
            content,
            memory_type,
            score,
            created_at,
            tags,
        });
    }
    Ok(results)
}

fn ui(f: &mut Frame, state: &AppState) {
    // Clear background with theme color
    f.render_widget(
        Block::default().style(Style::default().bg(state.theme.bg())),
        f.area(),
    );

    // Dynamic header height: base 9 + extra lines for additional context sessions
    let extra_context_lines = state.context_sessions.len().saturating_sub(1).min(3) as u16;
    let header_height = 9 + extra_context_lines;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header_height), // Header with logo (dynamic for multi-session)
            Constraint::Length(1),             // Spacer
            Constraint::Min(10),               // Main content
            Constraint::Length(3),             // Footer
        ])
        .split(f.area());
    render_header(f, chunks[0], state);
    // chunks[1] is spacer - empty
    render_main(f, chunks[2], state);
    render_footer(f, chunks[3], state);
}
