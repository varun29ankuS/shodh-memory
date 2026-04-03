//! Utility Functions for Memory Processing
//!
//! Text classification, content filtering, and regex helpers used across handlers.

use std::sync::OnceLock;

use crate::memory::ExperienceType;

// Static regexes for entity extraction (compiled once at startup)
static ALLCAPS_REGEX: OnceLock<regex::Regex> = OnceLock::new();
static ISSUE_ID_REGEX: OnceLock<regex::Regex> = OnceLock::new();

/// Get the all-caps regex (e.g., API, TUI, NER)
pub fn get_allcaps_regex() -> &'static regex::Regex {
    ALLCAPS_REGEX.get_or_init(|| regex::Regex::new(r"[A-Z]{2,}[A-Z0-9]*").unwrap())
}

/// Get the issue ID regex (e.g., SHO-123, JIRA-456)
pub fn get_issue_id_regex() -> &'static regex::Regex {
    ISSUE_ID_REGEX.get_or_init(|| regex::Regex::new(r"([A-Z]{2,10}-\d+)").unwrap())
}

/// Classify experience type from text content using keyword patterns.
/// Returns the most likely ExperienceType based on linguistic signals.
pub fn classify_experience_type(content: &str) -> ExperienceType {
    let lower = content.to_lowercase();

    // Decision signals - choices, preferences, commitments
    const DECISION_PATTERNS: &[&str] = &[
        "decided",
        "will use",
        "going with",
        "chose",
        "chosen",
        "prefer",
        "i'll",
        "we'll",
        "let's use",
        "selected",
        "picking",
        "opting for",
        "the approach is",
        "strategy is",
        "plan is to",
        "going to use",
    ];

    // Learning signals - new knowledge acquired
    const LEARNING_PATTERNS: &[&str] = &[
        "learned",
        "realized",
        "discovered",
        "found out",
        "turns out",
        "til ",
        "today i learned",
        "now i know",
        "understanding is",
        "figured out",
        "the reason is",
        "because",
        "works because",
        "key insight",
        "important to note",
        "remember that",
    ];

    // Error signals - bugs, issues, problems
    const ERROR_PATTERNS: &[&str] = &[
        "bug",
        "error",
        "fix",
        "fixed",
        "broken",
        "issue",
        "problem",
        "crash",
        "fail",
        "exception",
        "resolved",
        "workaround",
        "the solution was",
        "root cause",
        "debugging",
    ];

    // Discovery signals - findings, observations
    const DISCOVERY_PATTERNS: &[&str] = &[
        "found",
        "noticed",
        "interesting",
        "surprisingly",
        "unexpected",
        "turns out",
        "apparently",
        "it seems",
        "observation",
    ];

    // Context signals - user preferences, settings, environment
    const CONTEXT_PATTERNS: &[&str] = &[
        "prefers",
        "preference",
        "wants",
        "likes",
        "user",
        "setting",
        "configuration",
        "environment",
        "workspace",
        "setup",
    ];

    // Pattern signals - recurring behaviors, habits
    const PATTERN_PATTERNS: &[&str] = &[
        "pattern",
        "always",
        "usually",
        "tends to",
        "whenever",
        "every time",
        "consistently",
        "habit",
        "recurring",
    ];

    // Score each type
    let decision_score = DECISION_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let learning_score = LEARNING_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let error_score = ERROR_PATTERNS.iter().filter(|p| lower.contains(*p)).count();
    let discovery_score = DISCOVERY_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let context_score = CONTEXT_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let pattern_score = PATTERN_PATTERNS
        .iter()
        .filter(|p| lower.contains(*p))
        .count();

    // Find highest scoring type (require at least 1 match)
    let scores = [
        (decision_score, ExperienceType::Decision),
        (learning_score, ExperienceType::Learning),
        (error_score, ExperienceType::Error),
        (discovery_score, ExperienceType::Discovery),
        (context_score, ExperienceType::Context),
        (pattern_score, ExperienceType::Pattern),
    ];

    scores
        .into_iter()
        .filter(|(score, _)| *score > 0)
        .max_by_key(|(score, _)| *score)
        .map(|(_, typ)| typ)
        .unwrap_or(ExperienceType::Conversation)
}

/// Strip system noise from context to extract meaningful user content.
/// Removes <system-reminder>, <shodh-context>, Claude Code system prompts, and code blocks.
pub fn strip_system_noise(content: &str) -> String {
    let mut result = content.to_string();

    // Remove <system-reminder>...</system-reminder> blocks (handles multiline)
    while let Some(start) = result.find("<system-reminder>") {
        if let Some(end) = result.find("</system-reminder>") {
            let end_pos = end + "</system-reminder>".len();
            if end_pos <= result.len() && start < end_pos {
                result = format!("{}{}", &result[..start], &result[end_pos..]);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Remove <shodh-context>...</shodh-context> blocks (our own injected context)
    while let Some(start) = result.find("<shodh-context") {
        if let Some(end) = result.find("</shodh-context>") {
            let end_pos = end + "</shodh-context>".len();
            if end_pos <= result.len() && start < end_pos {
                result = format!("{}{}", &result[..start], &result[end_pos..]);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Remove <task-notification>...</task-notification> blocks
    while let Some(start) = result.find("<task-notification>") {
        if let Some(end) = result.find("</task-notification>") {
            let end_pos = end + "</task-notification>".len();
            if end_pos <= result.len() && start < end_pos {
                result = format!("{}{}", &result[..start], &result[end_pos..]);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Remove <shodh-memory>...</shodh-memory> blocks (hook-injected context that shouldn't be re-ingested)
    while let Some(start) = result.find("<shodh-memory") {
        if let Some(end) = result.find("</shodh-memory>") {
            let end_pos = end + "</shodh-memory>".len();
            if end_pos <= result.len() && start < end_pos {
                result = format!("{}{}", &result[..start], &result[end_pos..]);
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Remove session lifecycle messages (low-value noise from hooks)
    // These match patterns like "Session ended: user_stop" or "Session started in project-name"
    let lines: Vec<&str> = result.lines().collect();
    let filtered_lines: Vec<&str> = lines
        .into_iter()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("Session ended:")
                && !trimmed.starts_with("Session started")
                && !trimmed.starts_with("Modified file:")
        })
        .collect();
    result = filtered_lines.join("\n");

    // Remove Claude Code file content blocks - Windows paths
    while let Some(start) = result.find("Contents of C:\\") {
        let search_area = &result[start..];
        let end_offset = search_area
            .find("\n\n")
            .or_else(|| search_area.find("\r\n\r\n"))
            .unwrap_or(search_area.len().min(2000));
        let cut_pos = start + end_offset;
        if cut_pos <= result.len() {
            result = format!("{}{}", &result[..start], &result[cut_pos..]);
        } else {
            break;
        }
    }

    // Remove Claude Code file content blocks - Unix paths
    while let Some(start) = result.find("Contents of /") {
        let search_area = &result[start..];
        let end_offset = search_area
            .find("\n\n")
            .or_else(|| search_area.find("\r\n\r\n"))
            .unwrap_or(search_area.len().min(2000));
        let cut_pos = start + end_offset;
        if cut_pos <= result.len() {
            result = format!("{}{}", &result[..start], &result[cut_pos..]);
        } else {
            break;
        }
    }

    // Remove fenced code blocks (```...```) - these are often tool outputs, not memories
    while let Some(start) = result.find("```") {
        if start + 3 > result.len() {
            break;
        }
        if let Some(end) = result[start + 3..].find("```") {
            let end_pos = start + 3 + end + 3;
            if end_pos <= result.len() {
                result = format!("{}{}", &result[..start], &result[end_pos..]);
            } else {
                break;
            }
        } else {
            // Unclosed code block - remove from start to end
            result = result[..start].to_string();
            break;
        }
    }

    // Clean up excessive whitespace
    let result = result.split_whitespace().collect::<Vec<_>>().join(" ");

    // If result is mostly empty or very short after cleaning, return empty
    let trimmed = result.trim();
    if trimmed.len() < 10 || trimmed.chars().filter(|c| c.is_alphabetic()).count() < 5 {
        return String::new();
    }

    trimmed.to_string()
}

/// Check if content is a bare question (not worth storing as memory).
/// Questions like "what is X?" or "how do I Y?" without context are low-value.
pub fn is_bare_question(content: &str) -> bool {
    let trimmed = content.trim();
    let lower = trimmed.to_lowercase();

    // Question word starters
    let question_starters = [
        "what", "how", "why", "where", "when", "who", "can", "could", "is", "are", "do", "does",
        "will", "would", "should", "have",
    ];
    let starts_with_question = question_starters.iter().any(|q| lower.starts_with(q));
    let ends_with_question = trimmed.ends_with('?');

    // Short content - apply looser filter
    if trimmed.len() < 100 && (starts_with_question || ends_with_question) {
        return true;
    }

    // Medium content (100-300 chars) - check if it's purely a question without context
    if trimmed.len() < 300 && (starts_with_question || ends_with_question) {
        // Check for substance indicators that make it worth storing
        let has_substance = lower.contains("because")
            || lower.contains("the reason")
            || lower.contains("i think")
            || lower.contains("i believe")
            || lower.contains("we should")
            || lower.contains("decided")
            || lower.contains("learned")
            || lower.contains("found that")
            || lower.contains("the issue")
            || lower.contains("the problem")
            || lower.contains("the solution");

        if !has_substance {
            // Count sentences - pure questions are typically single sentence
            let sentence_count = trimmed.matches('.').count()
                + trimmed.matches('!').count()
                + trimmed.matches('?').count();

            if sentence_count <= 2 {
                return true;
            }
        }
    }

    false
}

/// Check if assistant response is boilerplate/low-value content.
/// Filters out generic greetings, offers to help, and repetitive patterns.
pub fn is_boilerplate_response(content: &str) -> bool {
    let lower = content.to_lowercase();

    // Generic greeting/ready-to-help patterns (high confidence noise)
    let boilerplate_starts = [
        "i'm ready to help",
        "i am ready to help",
        "i'm here to help",
        "i am here to help",
        "i can help you",
        "i'd be happy to help",
        "i would be happy to help",
        "let me help you",
        "i understand. i'm ready",
        "i understand. i am ready",
        "sure, i can",
        "sure! i can",
        "absolutely! i",
        "of course! i",
        "great question!",
        "good question!",
    ];

    if boilerplate_starts.iter().any(|p| lower.starts_with(p)) {
        return true;
    }

    // Generic offer patterns anywhere in short responses (<500 chars)
    if lower.len() < 500 {
        let generic_offers = [
            "what would you like me to",
            "let me know if you",
            "let me know what you",
            "feel free to ask",
            "don't hesitate to",
            "i'm happy to",
            "just let me know",
            "how can i assist",
            "how may i help",
            "is there anything else",
        ];

        let offer_count = generic_offers.iter().filter(|p| lower.contains(*p)).count();
        if offer_count >= 2 {
            return true;
        }
    }

    // Check for responses that are mostly bullet points of capabilities
    if lower.contains("i can:") || lower.contains("i'm able to:") {
        let bullet_count = content.matches("\n-").count() + content.matches("\n•").count();
        let has_substance = lower.contains("because")
            || lower.contains("the reason")
            || lower.contains("specifically")
            || lower.contains("for example");

        if bullet_count >= 3 && !has_substance {
            return true;
        }
    }

    false
}

/// Check if content is formatted tool/recall output that survived `strip_system_noise`.
/// Detects MCP-formatted memory listings, score bars, and structured output that
/// should never be stored as a memory.
pub fn is_tool_output_noise(content: &str) -> bool {
    let trimmed = content.trim();

    // Formatted recall output markers
    let noise_prefixes = [
        "─◎",
        "━━",
        "📝 MEMORIES",
        "✅ TODOS",
        "🐘 Recalled",
        "🐘 Memory",
    ];
    if noise_prefixes.iter().any(|p| trimmed.starts_with(p)) {
        return true;
    }

    // Memory tier markers from recall output re-ingestion
    let tier_markers = ["│ Working │", "│ Session │", "│ LongTerm │"];
    if tier_markers.iter().any(|m| trimmed.contains(m)) {
        return true;
    }

    // Score bar patterns (e.g. "█░░░░░░░░░ 5% │")
    if (trimmed.contains('█') || trimmed.contains('░')) && trimmed.contains('│') {
        return true;
    }

    // Memory ID patterns at end of lines (UUID suffix after ┗━)
    if trimmed.contains("┗━") {
        let memory_types = [
            "Observation",
            "Learning",
            "Decision",
            "Error",
            "Discovery",
            "Pattern",
            "Context",
            "Task",
            "CodeEdit",
            "FileAccess",
            "Search",
            "Command",
            "Conversation",
        ];
        if memory_types.iter().any(|t| trimmed.contains(t)) {
            return true;
        }
    }

    false
}

/// Check if content has a sufficient ratio of alphabetic characters to be meaningful.
/// Catches JSON dumps, emoji-heavy formatted output, hex strings, and control characters.
/// Returns `true` if the ratio is acceptable (≥40% alphabetic), `false` if too noisy.
pub fn has_sufficient_alpha_ratio(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return false;
    }

    let total_chars = trimmed.chars().count();
    let alpha_chars = trimmed.chars().filter(|c| c.is_alphabetic()).count();

    // At least 40% of characters must be alphabetic
    (alpha_chars as f64 / total_chars as f64) >= 0.4
}

/// Check if content is a re-ingested recall result (formatted memory listing).
/// This happens when the MCP proactive_context output gets fed back as input context
/// by hooks, and strip_system_noise doesn't fully clean it.
pub fn is_formatted_recall_output(content: &str) -> bool {
    let trimmed = content.trim();

    // Progress bar characters from score display
    let has_progress_bars = trimmed.contains('█') || trimmed.contains('░');

    // Memory tree connector from formatted output
    let has_tree_connector = trimmed.contains("┗━");

    // UUID-like pattern after │ separator (memory ID display)
    let has_pipe_id = trimmed.contains("│") && {
        // Look for hex UUID fragments after pipe
        trimmed.split('│').any(|part| {
            let t = part.trim();
            t.len() >= 8 && t.len() <= 40 && t.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
        })
    };

    // Need at least 2 indicators to be confident
    let indicators = has_progress_bars as u8 + has_tree_connector as u8 + has_pipe_id as u8;
    indicators >= 2
}

/// Strip MCP response formatting noise before embedding for feedback.
///
/// MCP tools (proactive_context, recall, etc.) wrap semantic content in
/// box-drawing art, emoji decorators, progress bars, and latency annotations.
/// This visual formatting is great for display but poisons the embedding —
/// MiniLM tokenizes "━━━" and "🧠" into noise tokens that dilute cosine
/// similarity, degrading Hebbian feedback signal quality.
///
/// This function strips the formatting layer while preserving all semantic
/// content. It does NOT affect what the user sees — only what gets embedded.
pub fn strip_mcp_response_noise(content: &str) -> String {
    let mut out = String::with_capacity(content.len());

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip pure box-art lines (━━━, ┃ header ┃, ┣━━━┫, etc.)
        if trimmed
            .chars()
            .all(|c| is_box_drawing(c) || c.is_whitespace())
            && !trimmed.is_empty()
        {
            continue;
        }

        // Skip latency/diagnostic annotations
        // e.g. "[Latency: 421ms | Threshold: 65%]", "[Feedback loop: ...]"
        if trimmed.starts_with("[Latency:")
            || trimmed.starts_with("[Feedback loop:")
            || trimmed.starts_with("[Token budget:")
        {
            continue;
        }

        // Skip "Surfaced N facts" / "Surfaced N memories" headers
        if trimmed.starts_with("Surfaced ") && trimmed.ends_with(" facts")
            || trimmed.starts_with("Surfaced ") && trimmed.ends_with(" memories")
        {
            continue;
        }

        // Skip progress bar lines (█████░░░ patterns)
        if trimmed.contains('█') || trimmed.contains('░') {
            continue;
        }

        // Skip "semantic: -X%" annotations
        if trimmed.starts_with("semantic:") {
            continue;
        }

        // Strip box-drawing and emoji decorators from lines that have real content
        let cleaned: String = trimmed
            .chars()
            .filter(|c| !is_box_drawing(*c) && !is_mcp_decorator(*c))
            .collect();

        let cleaned = cleaned.trim();
        if !cleaned.is_empty() {
            out.push_str(cleaned);
            out.push('\n');
        }
    }

    // Collapse multiple blank lines
    while out.contains("\n\n\n") {
        out = out.replace("\n\n\n", "\n\n");
    }

    out.trim().to_string()
}

/// Box-drawing characters used in MCP response formatting.
#[inline]
fn is_box_drawing(c: char) -> bool {
    matches!(
        c,
        '━' | '┃'
            | '┏'
            | '┓'
            | '┗'
            | '┛'
            | '┣'
            | '┫'
            | '┳'
            | '┻'
            | '╋'
            | '─'
            | '│'
            | '┌'
            | '┐'
            | '└'
            | '┘'
            | '├'
            | '┤'
            | '┬'
            | '┴'
            | '┼'
            | '╔'
            | '╗'
            | '╚'
            | '╝'
            | '║'
            | '═'
            | '╠'
            | '╣'
            | '╦'
            | '╩'
            | '╬'
    )
}

/// Emoji decorators commonly injected by MCP formatting.
/// These are visual indicators (not content) — stripping them improves embedding quality.
#[inline]
fn is_mcp_decorator(c: char) -> bool {
    matches!(
        c,
        '🧠' | '📅'
            | '📋'
            | '📌'
            | '💡'
            | '🐘'
            | '⚡'
            | '🔍'
            | '✅'
            | '❌'
            | '⏰'
            | '🎯'
            | '📊'
            | '🔗'
            | '💾'
            | '🏷'
            | '📝'
            | '🔔'
            | '⭐'
            | '🚀'
            | '⚠'
            | '🔄'
            | '📦'
            | '🛠'
            | '💬'
            | '🗂'
            | '📂'
            | '🏗'
            | '✨'
            | '🪝'
            | '▸'
            | '▹'
            | '▪'
            | '▫'
            | '◆'
            | '◇'
            | '●'
            | '○'
            | '◉'
            | '◎'
    )
}

/// Check if content is essentially empty or meaningless after preprocessing.
pub fn is_empty_content(content: &str) -> bool {
    let trimmed = content.trim();
    trimmed.is_empty() || trimmed.len() < 5
}

/// Default function for recall limit
pub fn default_recall_limit() -> usize {
    5
}

/// Default function for recall mode
pub fn default_recall_mode() -> String {
    "hybrid".to_string()
}

/// Default function for batch options (extract_entities, create_edges)
pub fn default_true() -> bool {
    true
}

/// Default change type for upsert
pub fn default_change_type() -> String {
    "content_updated".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_decision() {
        let content = "I decided to use Rust for this project";
        assert!(matches!(
            classify_experience_type(content),
            ExperienceType::Decision
        ));
    }

    #[test]
    fn test_classify_learning() {
        let content = "I learned that Rust's borrow checker prevents data races";
        assert!(matches!(
            classify_experience_type(content),
            ExperienceType::Learning
        ));
    }

    #[test]
    fn test_classify_error() {
        let content = "Found a bug in the authentication flow, fixed it by adding validation";
        assert!(matches!(
            classify_experience_type(content),
            ExperienceType::Error
        ));
    }

    #[test]
    fn test_strip_system_noise() {
        let content = "Hello <system-reminder>ignore this</system-reminder> world";
        let cleaned = strip_system_noise(content);
        assert!(!cleaned.contains("system-reminder"));
        assert!(!cleaned.contains("ignore this"));
    }

    #[test]
    fn test_is_bare_question() {
        assert!(is_bare_question("What is Rust?"));
        assert!(is_bare_question("How do I install npm?"));
        assert!(!is_bare_question(
            "I learned that Rust is a systems programming language because it provides memory safety without garbage collection."
        ));
    }

    #[test]
    fn test_strip_mcp_response_noise() {
        let mcp_output = "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n┃ 🧠 SHODH MEMORY ┃\n┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\nSurfaced 3 facts\n📌 User prefers dark mode\n📋 Project uses Rust for backend\n💡 Authentication uses JWT\n█████████░ 90%\n[Latency: 421ms | Threshold: 65%]\nsemantic: -12%";
        let cleaned = strip_mcp_response_noise(mcp_output);
        assert!(cleaned.contains("User prefers dark mode"));
        assert!(cleaned.contains("Project uses Rust for backend"));
        assert!(cleaned.contains("Authentication uses JWT"));
        assert!(!cleaned.contains("━━"));
        assert!(!cleaned.contains("Latency:"));
        assert!(!cleaned.contains("█"));
        assert!(!cleaned.contains("Surfaced 3 facts"));
    }

    #[test]
    fn test_strip_mcp_preserves_plain_text() {
        let plain = "The user decided to use PostgreSQL for the database. This is a good choice because it supports JSON and full-text search.";
        let cleaned = strip_mcp_response_noise(plain);
        assert_eq!(cleaned, plain);
    }

    #[test]
    fn test_is_boilerplate() {
        assert!(is_boilerplate_response(
            "I'm ready to help! What would you like me to do?"
        ));
        assert!(!is_boilerplate_response(
            "The issue is caused by a race condition in the authentication middleware."
        ));
    }

    #[test]
    fn test_is_tool_output_noise() {
        // Should detect formatted recall output
        assert!(is_tool_output_noise(
            "─◎ RETRIEVE 2m ▲ │ ○ !!!GROW-2 Rewrite GitHub R.."
        ));
        assert!(is_tool_output_noise("📝 MEMORIES\n• some memory here"));
        assert!(is_tool_output_noise("🐘 Recalled 4 Results"));
        assert!(is_tool_output_noise(
            "Something │ Working │ with tier marker"
        ));
        assert!(is_tool_output_noise(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ));

        // Should NOT detect legitimate content
        assert!(!is_tool_output_noise("User prefers Rust for memory safety"));
        assert!(!is_tool_output_noise(
            "The pipeline audit found 8 bugs that need fixing"
        ));
    }

    #[test]
    fn test_has_sufficient_alpha_ratio() {
        // Good alpha ratio — legitimate content
        assert!(has_sufficient_alpha_ratio(
            "User prefers Rust for memory safety guarantees"
        ));
        assert!(has_sufficient_alpha_ratio("The bug was in authentication"));

        // Bad alpha ratio — noise
        assert!(!has_sufficient_alpha_ratio("█░░░░░░░░░ 3% │ 06:04 PM"));
        assert!(!has_sufficient_alpha_ratio("━━━━━━━━━━━━━━━━━━━━━━━"));
        assert!(!has_sufficient_alpha_ratio(""));
        assert!(!has_sufficient_alpha_ratio("│ │ │ │ │ │ │"));
    }

    #[test]
    fn test_is_formatted_recall_output() {
        // Should detect re-ingested recall output (needs 2+ indicators)
        assert!(is_formatted_recall_output(
            "█░░░░░░░░░ 5% │ 02:08 PM\n  Some memory\n  ┗━ Observation │ Working │ abc12345"
        ));

        // Single indicator is not enough
        assert!(!is_formatted_recall_output("Just has ┗━ but nothing else"));
        assert!(!is_formatted_recall_output(
            "Has █ bar but no tree connector"
        ));

        // Should NOT detect legitimate content
        assert!(!is_formatted_recall_output(
            "The retrieval pipeline has 5 stages and processes memories in order"
        ));
    }
}
