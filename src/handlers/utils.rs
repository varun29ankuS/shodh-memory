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
    fn test_is_boilerplate() {
        assert!(is_boilerplate_response(
            "I'm ready to help! What would you like me to do?"
        ));
        assert!(!is_boilerplate_response(
            "The issue is caused by a race condition in the authentication middleware."
        ));
    }
}
