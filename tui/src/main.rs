use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::{io, sync::Arc, time::{Duration, Instant}};
use tokio::sync::Mutex;

mod logo;
mod stream;
mod types;
mod widgets;

use logo::ELEPHANT;
use stream::MemoryStream;
use types::{AppState, ViewMode};
use widgets::{render_header, render_main, render_footer};

struct UserSelector {
    users: Vec<String>,
    selected: usize,
    loading: bool,
    error: Option<String>,
}

impl UserSelector {
    fn new() -> Self { Self { users: vec![], selected: 0, loading: true, error: None } }
    fn select_next(&mut self) { if !self.users.is_empty() { self.selected = (self.selected + 1) % self.users.len(); } }
    fn select_prev(&mut self) { if !self.users.is_empty() { self.selected = self.selected.checked_sub(1).unwrap_or(self.users.len() - 1); } }
    fn selected_user(&self) -> Option<&String> { self.users.get(self.selected) }
}

async fn fetch_users(base_url: &str, api_key: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/users", base_url);
    match client.get(&url).header("X-API-Key", api_key).send().await {
        Ok(resp) => resp.json::<Vec<String>>().await.map_err(|e| format!("Parse: {}", e)),
        Err(e) => Err(format!("Connection: {}", e)),
    }
}

fn render_user_selector(f: &mut Frame, selector: &UserSelector) {
    let area = f.area();
    f.render_widget(Block::default().style(Style::default().bg(Color::Black)), area);
    let chunks = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Length(3), Constraint::Min(10), Constraint::Length(3)])
        .margin(2).split(area);

    let logo_lines: Vec<Line> = ELEPHANT.iter().enumerate().map(|(i, l)| {
        let g = [(255,180,50),(255,160,40),(255,140,30),(255,120,20),(255,100,10),(255,80,0)];
        Line::from(Span::styled(*l, Style::default().fg(Color::Rgb(g[i%6].0, g[i%6].1, g[i%6].2))))
    }).collect();
    f.render_widget(Paragraph::new(logo_lines).alignment(Alignment::Center), chunks[0]);

    f.render_widget(Paragraph::new(Line::from(Span::styled("SELECT USER", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)))).alignment(Alignment::Center), chunks[1]);

    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)).title(Span::styled(" Users ", Style::default().fg(Color::Cyan)));
    if selector.loading {
        f.render_widget(Paragraph::new("Loading...").fg(Color::Yellow).alignment(Alignment::Center).block(block), chunks[2]);
    } else if let Some(ref e) = selector.error {
        f.render_widget(Paragraph::new(e.as_str()).fg(Color::Red).alignment(Alignment::Center).block(block), chunks[2]);
    } else if selector.users.is_empty() {
        f.render_widget(Paragraph::new("No users found.").fg(Color::DarkGray).alignment(Alignment::Center).block(block), chunks[2]);
    } else {
        let items: Vec<ListItem> = selector.users.iter().enumerate().map(|(i, u)| {
            let s = i == selector.selected;
            let st = if s { Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) };
            ListItem::new(Line::from(vec![Span::styled(if s { "> " } else { "  " }, st), Span::styled(u, st)]))
        }).collect();
        f.render_widget(List::new(items).block(block), chunks[2]);
    }

    f.render_widget(Paragraph::new(Line::from(vec![
        Span::styled(" j/k ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::styled("select ", Style::default().fg(Color::DarkGray)),
        Span::styled(" Enter ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::styled("confirm ", Style::default().fg(Color::DarkGray)),
        Span::styled(" q ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::styled("quit", Style::default().fg(Color::DarkGray)),
    ])).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray))), chunks[3]);
}

async fn run_user_selector(base_url: &str, api_key: &str) -> Result<Option<String>> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let mut selector = UserSelector::new();
    match fetch_users(base_url, api_key).await {
        Ok(u) => { selector.users = u; selector.loading = false; }
        Err(e) => { selector.error = Some(e); selector.loading = false; }
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
                        KeyCode::Enter => if let Some(u) = selector.selected_user() { break Some(u.clone()); },
                        _ => {}
                    }
                }
            }
        }
    };
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<()> {
    let base_url = std::env::var("SHODH_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:3030".to_string()).trim_end_matches("/api/events").to_string();
    let api_key = std::env::var("SHODH_API_KEY").unwrap_or_else(|_| "sk-shodh-dev-local-testing-key".to_string());

    let user = match run_user_selector(&base_url, &api_key).await? {
        Some(u) => u,
        None => return Ok(()),
    };

    let state = Arc::new(Mutex::new(AppState::new()));
    { let mut s = state.lock().await; s.current_user = user.clone(); }

    let stream = MemoryStream::new(&format!("{}/api/events", base_url), &api_key, &user, Arc::clone(&state));
    let h = tokio::spawn(async move { stream.run().await; });
    let r = run_tui(state).await;
    h.abort();
    r
}

async fn run_tui(state: Arc<Mutex<AppState>>) -> Result<()> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        { let g = state.lock().await; terminal.draw(|f| ui(f, &g))?; }
        if crossterm::event::poll(tick_rate.saturating_sub(last_tick.elapsed()))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    let mut g = state.lock().await;
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('c') => g.events.clear(),
                        KeyCode::Char('d') => g.set_view(ViewMode::Dashboard),
                        KeyCode::Char('a') => g.set_view(ViewMode::ActivityLogs),
                        KeyCode::Char('g') => g.set_view(ViewMode::GraphList),
                        KeyCode::Char('m') => g.set_view(ViewMode::GraphMap),
                        KeyCode::Tab => g.cycle_view(),
                        KeyCode::Up | KeyCode::Char('k') => g.scroll_up(),
                        KeyCode::Down | KeyCode::Char('j') => g.scroll_down(),
                        KeyCode::PageUp => for _ in 0..5 { g.scroll_up(); },
                        KeyCode::PageDown => for _ in 0..5 { g.scroll_down(); },
                        KeyCode::Home => g.scroll_offset = 0,
                        _ => {}
                    }
                }
                Event::Mouse(m) => {
                    let mut g = state.lock().await;
                    match m.kind {
                        MouseEventKind::ScrollUp => g.scroll_up(),
                        MouseEventKind::ScrollDown => g.scroll_down(),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        if last_tick.elapsed() >= tick_rate { state.lock().await.tick(); last_tick = Instant::now(); }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    Ok(())
}

fn ui(f: &mut Frame, state: &AppState) {
    let chunks = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(10), Constraint::Length(3)]).split(f.area());
    render_header(f, chunks[0], state);
    render_main(f, chunks[1], state);
    render_footer(f, chunks[2], state);
}
