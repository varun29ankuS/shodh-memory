//! Linear-style CLI formatting for todos
//!
//! Produces clean, minimal output inspired by Linear's design:
//! - Status icons: ◌ ○ ◐ ⊘ ● ⊗
//! - Priority indicators: !!! !! ! (none)
//! - Short IDs: SHO-xxxx
//! - Status-grouped lists with horizontal rules

use chrono::{DateTime, Datelike, Utc};

use super::todos::{ProjectStats, UserTodoStats};
use super::types::{Project, ProjectStatus, Todo, TodoStatus};

/// Width of formatted output
const LINE_WIDTH: usize = 70;

/// Format a single todo line (Linear-style)
pub fn format_todo_line(todo: &Todo, project_name: Option<&str>, show_meta: bool) -> String {
    let status = todo.status.icon();
    let priority = format!("{:3}", todo.priority.indicator());
    let short_id = todo.id.short();

    // First line: status, priority, ID, content, project
    let mut line = format!("  {} {} {}  {}", status, priority, short_id, todo.content);

    if let Some(proj) = project_name {
        // Pad to align project name on the right
        let content_width = line.chars().count();
        if content_width < LINE_WIDTH - proj.len() - 2 {
            let padding = LINE_WIDTH - content_width - proj.len() - 2;
            line.push_str(&" ".repeat(padding));
        } else {
            line.push_str("  ");
        }
        line.push_str(proj);
    }

    // Second line: metadata (contexts, due date, blocked info)
    if show_meta {
        let mut meta = Vec::new();

        if !todo.contexts.is_empty() {
            meta.push(todo.contexts.join(" "));
        }

        if let Some(ref due) = todo.due_date {
            meta.push(format_due_date(due));
        }

        if todo.status == TodoStatus::Blocked {
            if let Some(ref blocked_on) = todo.blocked_on {
                meta.push(format!("Blocked on {}", blocked_on));
            }
        }

        if !meta.is_empty() {
            line.push_str(&format!("\n                  {}", meta.join(" · ")));
        }
    }

    line
}

/// Format due date relative to now
pub fn format_due_date(due: &DateTime<Utc>) -> String {
    let now = Utc::now();
    let diff_secs = (due.timestamp() - now.timestamp()) as i64;
    let diff_hours = diff_secs / 3600;
    let diff_days = diff_secs / 86400;

    if diff_secs < 0 {
        // Overdue
        let hours = (-diff_hours) as u32;
        if hours < 24 {
            format!("⚠ Overdue {}h", hours)
        } else {
            format!("⚠ Overdue {}d", hours / 24)
        }
    } else if diff_hours < 24 {
        // Due today
        format!("Due {}", due.format("%I:%M %p"))
    } else if diff_days < 7 {
        // Due this week
        format!("Due {}", due.format("%a"))
    } else {
        // Future
        format!("Due {}", due.format("%b %d"))
    }
}

/// Format a horizontal rule with label
fn format_section_header(label: &str) -> String {
    let dashes = LINE_WIDTH.saturating_sub(label.len() + 1);
    format!("{} {}", label, "─".repeat(dashes))
}

/// Format a subtask line with indentation
pub fn format_subtask_line(todo: &Todo, project_name: Option<&str>) -> String {
    let status = todo.status.icon();
    let priority = format!("{:3}", todo.priority.indicator());
    let short_id = todo.id.short();

    // Indented by 4 extra spaces for subtasks
    let mut line = format!("      {} {} {}  {}", status, priority, short_id, todo.content);

    if let Some(proj) = project_name {
        let content_width = line.chars().count();
        if content_width < LINE_WIDTH - proj.len() - 2 {
            let padding = LINE_WIDTH - content_width - proj.len() - 2;
            line.push_str(&" ".repeat(padding));
        } else {
            line.push_str("  ");
        }
        line.push_str(proj);
    }

    line
}

/// Format list of todos grouped by status (Linear-style main view)
/// Subtasks are shown indented under their parent todos
pub fn format_todo_list(todos: &[Todo], projects: &[Project]) -> String {
    if todos.is_empty() {
        return "No todos found.".to_string();
    }

    let mut output = format!(
        "SHO · My Todos{:>width$} items\n\n",
        todos.len(),
        width = LINE_WIDTH - 14
    );

    // Separate parent todos and subtasks
    let parent_todos: Vec<_> = todos.iter().filter(|t| t.parent_id.is_none()).collect();
    let subtasks: Vec<_> = todos.iter().filter(|t| t.parent_id.is_some()).collect();

    // Group by status in workflow order
    let status_order = [
        (TodoStatus::InProgress, "In Progress ◐"),
        (TodoStatus::Todo, "Todo ○"),
        (TodoStatus::Blocked, "Blocked ⊘"),
        (TodoStatus::Backlog, "Backlog ◌"),
    ];

    for (status, label) in status_order {
        let items: Vec<_> = parent_todos
            .iter()
            .filter(|t| t.status == status)
            .collect();

        if !items.is_empty() {
            output.push_str(&format_section_header(label));
            output.push('\n');

            for todo in items {
                let project_name = todo
                    .project_id
                    .as_ref()
                    .and_then(|pid| projects.iter().find(|p| p.id == *pid))
                    .map(|p| p.name.as_str());

                output.push_str(&format_todo_line(todo, project_name, true));
                output.push('\n');

                // Find and render subtasks of this todo
                let todo_subtasks: Vec<_> = subtasks
                    .iter()
                    .filter(|st| st.parent_id.as_ref() == Some(&todo.id))
                    .collect();

                for subtask in todo_subtasks {
                    let subtask_project = subtask
                        .project_id
                        .as_ref()
                        .and_then(|pid| projects.iter().find(|p| p.id == *pid))
                        .map(|p| p.name.as_str());

                    output.push_str(&format_subtask_line(subtask, subtask_project));
                    output.push('\n');
                }
            }
            output.push('\n');
        }
    }

    output
}

/// Format list of todos for a single project
pub fn format_project_todos(project: &Project, todos: &[Todo], stats: &ProjectStats) -> String {
    let mut output = format!(
        "{}{:>width$} items\n",
        project.name,
        stats.total,
        width = LINE_WIDTH - project.name.len()
    );
    output.push_str(&"─".repeat(LINE_WIDTH));
    output.push('\n');

    // Status summary
    let mut status_parts = Vec::new();
    if stats.in_progress > 0 {
        status_parts.push(format!("In Progress ({})", stats.in_progress));
    }
    if stats.todo > 0 {
        status_parts.push(format!("Todo ({})", stats.todo));
    }
    if stats.blocked > 0 {
        status_parts.push(format!("Blocked ({})", stats.blocked));
    }
    if stats.backlog > 0 {
        status_parts.push(format!("Backlog ({})", stats.backlog));
    }
    if !status_parts.is_empty() {
        output.push_str(&format!("  {}\n\n", status_parts.join("    ")));
    }

    // List todos
    for todo in todos {
        output.push_str(&format_todo_line(todo, None, true));
        output.push('\n');
    }

    output
}

/// Format due todos for proactive_context (compact view)
pub fn format_due_todos(todos: &[Todo]) -> String {
    if todos.is_empty() {
        return String::new();
    }

    let mut output = format!(
        "📋 Due Today ({})\n{}\n",
        todos.len(),
        "─".repeat(LINE_WIDTH)
    );

    for todo in todos {
        let status = todo.status.icon();
        let priority = format!("{:3}", todo.priority.indicator());
        let short_id = todo.id.short();

        let due_text = todo
            .due_date
            .as_ref()
            .map(|d| format_due_date(d))
            .unwrap_or_default();

        output.push_str(&format!(
            "  {} {} {}  {:<40} {}\n",
            status, priority, short_id, todo.content, due_text
        ));
    }

    output
}

/// Format created todo confirmation
pub fn format_todo_created(todo: &Todo, project_name: Option<&str>) -> String {
    let mut output = format!("✓ Created {}\n\n", todo.id.short());
    output.push_str(&format_todo_line(todo, project_name, true));
    output
}

/// Format completed todo confirmation
pub fn format_todo_completed(todo: &Todo, next: Option<&Todo>) -> String {
    let duration = todo
        .completed_at
        .and_then(|completed| {
            let created = todo.created_at;
            let diff = completed - created;
            let hours = diff.num_hours();
            let mins = diff.num_minutes() % 60;
            if hours > 0 {
                Some(format!("{}h {}m", hours, mins))
            } else if mins > 0 {
                Some(format!("{}m", mins))
            } else {
                Some("< 1m".to_string())
            }
        })
        .unwrap_or_default();

    let mut output = format!("✓ Completed {}\n\n", todo.id.short());
    output.push_str(&format!(
        "  {} {}  {}",
        todo.status.icon(),
        todo.content,
        if !duration.is_empty() {
            format!("✓ {}", duration)
        } else {
            String::new()
        }
    ));

    if let Some(next_todo) = next {
        output.push_str(&format!(
            "\n\n  → Next occurrence: {} ({})",
            next_todo.id.short(),
            next_todo
                .due_date
                .map(|d| format_due_date(&d))
                .unwrap_or_default()
        ));
    }

    output
}

/// Format todo updated confirmation
pub fn format_todo_updated(todo: &Todo, project_name: Option<&str>) -> String {
    let mut output = format!("✓ Updated {}\n\n", todo.id.short());
    output.push_str(&format_todo_line(todo, project_name, true));
    output
}

/// Format todo deleted confirmation
pub fn format_todo_deleted(todo_id: &str) -> String {
    format!("✓ Deleted {}", todo_id)
}

/// Format project list
pub fn format_project_list(projects: &[(Project, ProjectStats)]) -> String {
    if projects.is_empty() {
        return "No projects found.".to_string();
    }

    let mut output = format!(
        "SHO · Projects{:>width$} total\n\n",
        projects.len(),
        width = LINE_WIDTH - 14
    );

    for (project, stats) in projects {
        let active = stats.in_progress + stats.todo;
        output.push_str(&format!(
            "  📁 {:<40} {} active / {} total\n",
            project.name, active, stats.total
        ));

        // Status breakdown
        let mut parts = Vec::new();
        if stats.in_progress > 0 {
            parts.push(format!("◐ {}", stats.in_progress));
        }
        if stats.todo > 0 {
            parts.push(format!("○ {}", stats.todo));
        }
        if stats.blocked > 0 {
            parts.push(format!("⊘ {}", stats.blocked));
        }
        if stats.done > 0 {
            parts.push(format!("● {}", stats.done));
        }
        if !parts.is_empty() {
            output.push_str(&format!("     {}\n", parts.join("  ")));
        }
        output.push('\n');
    }

    output
}

/// Format project created confirmation
pub fn format_project_created(project: &Project) -> String {
    format!(
        "✓ Created project '{}' (ID: {})",
        project.name,
        &project.id.0.to_string()[..8]
    )
}

/// Format project update output
pub fn format_project_updated(project: &Project) -> String {
    let status_str = match project.status {
        ProjectStatus::Active => "Active",
        ProjectStatus::OnHold => "On Hold",
        ProjectStatus::Completed => "Completed",
        ProjectStatus::Archived => "Archived",
    };
    format!(
        "✓ Updated project '{}' (Status: {})",
        project.name, status_str
    )
}

/// Format project delete output
pub fn format_project_deleted(project: &Project, todos_deleted: usize) -> String {
    if todos_deleted > 0 {
        format!(
            "✓ Deleted project '{}' and {} todos",
            project.name, todos_deleted
        )
    } else {
        format!("✓ Deleted project '{}'", project.name)
    }
}

/// Format user todo stats
pub fn format_user_stats(stats: &UserTodoStats) -> String {
    let mut output = format!(
        "SHO · Stats{:>width$} todos\n",
        stats.total,
        width = LINE_WIDTH - 11
    );
    output.push_str(&"─".repeat(LINE_WIDTH));
    output.push('\n');

    output.push_str(&format!("\n  ◐ In Progress    {:>4}\n", stats.in_progress));
    output.push_str(&format!("  ○ Todo           {:>4}\n", stats.todo));
    output.push_str(&format!("  ⊘ Blocked        {:>4}\n", stats.blocked));
    output.push_str(&format!("  ◌ Backlog        {:>4}\n", stats.backlog));
    output.push_str(&format!("  ● Done           {:>4}\n", stats.done));
    output.push_str(&format!("  ⊗ Cancelled      {:>4}\n", stats.cancelled));

    output.push_str(&format!("\n  ⚠ Overdue        {:>4}\n", stats.overdue));
    output.push_str(&format!("  📅 Due Today     {:>4}\n", stats.due_today));
    output.push_str(&format!("  📁 Projects      {:>4}\n", stats.projects));

    output
}

/// Parse natural date strings to DateTime
pub fn parse_due_date(input: &str) -> Option<DateTime<Utc>> {
    use chrono::Duration;

    let input_lower = input.to_lowercase();
    let now = Utc::now();

    // End of day helper
    let end_of_day = |dt: DateTime<Utc>| {
        dt.date_naive()
            .and_hms_opt(23, 59, 59)
            .map(|t| t.and_utc())
            .unwrap_or(dt)
    };

    // Handle "next <day>" patterns
    if input_lower.starts_with("next ") {
        let day = input_lower.strip_prefix("next ").unwrap_or("");
        return match day {
            "monday" | "mon" => Some(next_weekday(now + Duration::days(7), 1)),
            "tuesday" | "tue" => Some(next_weekday(now + Duration::days(7), 2)),
            "wednesday" | "wed" => Some(next_weekday(now + Duration::days(7), 3)),
            "thursday" | "thu" => Some(next_weekday(now + Duration::days(7), 4)),
            "friday" | "fri" => Some(next_weekday(now + Duration::days(7), 5)),
            "saturday" | "sat" => Some(next_weekday(now + Duration::days(7), 6)),
            "sunday" | "sun" => Some(next_weekday(now + Duration::days(7), 0)),
            "week" => Some(end_of_day(now + Duration::weeks(1))),
            "month" => Some(end_of_day(now + Duration::days(30))),
            _ => None,
        };
    }

    // Handle "in X days/weeks" patterns
    if input_lower.starts_with("in ") {
        let rest = input_lower.strip_prefix("in ").unwrap_or("");
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() == 2 {
            if let Ok(num) = parts[0].parse::<i64>() {
                return match parts[1] {
                    "day" | "days" => Some(end_of_day(now + Duration::days(num))),
                    "week" | "weeks" => Some(end_of_day(now + Duration::weeks(num))),
                    "month" | "months" => Some(end_of_day(now + Duration::days(num * 30))),
                    _ => None,
                };
            }
        }
    }

    match input_lower.as_str() {
        "today" => Some(end_of_day(now)),
        "tomorrow" => Some(end_of_day(now + Duration::days(1))),
        "next week" => Some(end_of_day(now + Duration::weeks(1))),
        "next month" => Some(end_of_day(now + Duration::days(30))),
        "monday" | "mon" => Some(next_weekday(now, 1)),
        "tuesday" | "tue" => Some(next_weekday(now, 2)),
        "wednesday" | "wed" => Some(next_weekday(now, 3)),
        "thursday" | "thu" => Some(next_weekday(now, 4)),
        "friday" | "fri" => Some(next_weekday(now, 5)),
        "saturday" | "sat" => Some(next_weekday(now, 6)),
        "sunday" | "sun" => Some(next_weekday(now, 0)),
        "eod" | "end of day" => Some(end_of_day(now)),
        "eow" | "end of week" => Some(next_weekday(now, 5)), // Friday
        "eom" | "end of month" => {
            let next_month = now.with_day(1).unwrap() + Duration::days(32);
            let first_of_next = next_month.with_day(1).unwrap();
            Some(end_of_day(first_of_next - Duration::days(1)))
        }
        _ => {
            // Try parsing as ISO date
            chrono::DateTime::parse_from_rfc3339(input)
                .ok()
                .map(|d| d.with_timezone(&Utc))
                .or_else(|| {
                    // Try parsing as date only (YYYY-MM-DD)
                    chrono::NaiveDate::parse_from_str(input, "%Y-%m-%d")
                        .ok()
                        .and_then(|d| d.and_hms_opt(23, 59, 59))
                        .map(|dt| dt.and_utc())
                })
        }
    }
}

/// Get next occurrence of a weekday (0=Sun, 1=Mon, ..., 6=Sat)
fn next_weekday(from: DateTime<Utc>, target_dow: u32) -> DateTime<Utc> {
    use chrono::Duration;

    let current_dow = from.weekday().num_days_from_sunday();
    let days_until = if target_dow > current_dow {
        target_dow - current_dow
    } else if target_dow < current_dow {
        7 - current_dow + target_dow
    } else {
        7 // Same day = next week
    };

    let target = from + Duration::days(days_until as i64);
    target
        .date_naive()
        .and_hms_opt(23, 59, 59)
        .map(|t| t.and_utc())
        .unwrap_or(target)
}

/// Extract @contexts from content string
pub fn extract_contexts(content: &str) -> Vec<String> {
    let re = regex::Regex::new(r"@(\w+)").unwrap();
    re.captures_iter(content)
        .map(|c| format!("@{}", &c[1].to_lowercase()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_due_date() {
        assert!(parse_due_date("today").is_some());
        assert!(parse_due_date("tomorrow").is_some());
        assert!(parse_due_date("next week").is_some());
        assert!(parse_due_date("monday").is_some());
        assert!(parse_due_date("2024-12-25").is_some());
    }

    #[test]
    fn test_extract_contexts() {
        let contexts = extract_contexts("Buy groceries @errands @home");
        assert_eq!(contexts.len(), 2);
        assert!(contexts.contains(&"@errands".to_string()));
        assert!(contexts.contains(&"@home".to_string()));
    }

    #[test]
    fn test_format_due_date() {
        use chrono::Duration;

        let now = Utc::now();

        // Overdue
        let overdue = now - Duration::hours(2);
        let text = format_due_date(&overdue);
        assert!(text.contains("Overdue"));

        // Future
        let future = now + Duration::days(3);
        let text = format_due_date(&future);
        assert!(text.contains("Due"));
    }
}
