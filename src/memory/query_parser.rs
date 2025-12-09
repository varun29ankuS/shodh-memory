//! Linguistic Query Parser
//!
//! Based on:
//! - Lioma & Ounis (2006): "Content Load of Part of Speech Blocks"
//! - Bendersky & Croft (2008): "Discovering Key Concepts in Verbose Queries"
//!
//! Extracts focal entities (nouns), discriminative modifiers (adjectives),
//! and relational context (verbs) from natural language queries.

use crate::constants::{IC_ADJECTIVE, IC_NOUN, IC_VERB};

/// Focal entity extracted from query (noun)
#[derive(Debug, Clone)]
pub struct FocalEntity {
    pub text: String,
    pub ic_weight: f32,
}

/// Discriminative modifier (adjective/qualifier)
#[derive(Debug, Clone)]
#[allow(unused)] // Public API - fields exposed for analysis consumers
pub struct Modifier {
    pub text: String,
    /// IC weight for importance scoring (Lioma & Ounis 2006)
    pub ic_weight: f32,
}

/// Relational context (verb)
#[derive(Debug, Clone)]
#[allow(unused)] // Public API - fields exposed for analysis consumers
pub struct Relation {
    pub text: String,
    /// IC weight for importance scoring (Lioma & Ounis 2006)
    pub ic_weight: f32,
}

/// Complete linguistic analysis of a query
#[derive(Debug, Clone)]
#[allow(unused)] // Public API - fields exposed for analysis consumers
pub struct QueryAnalysis {
    /// Focal entities (nouns) - primary search targets
    pub focal_entities: Vec<FocalEntity>,

    /// Discriminative modifiers (adjectives) - quality refiners
    pub discriminative_modifiers: Vec<Modifier>,

    /// Relational context (verbs) - graph traversal guides
    pub relational_context: Vec<Relation>,

    /// Original query text (retained for logging/debugging)
    pub original_query: String,
}

#[allow(unused)] // Public API
impl QueryAnalysis {
    /// Calculate weighted importance of this query (for ranking)
    pub fn total_weight(&self) -> f32 {
        let entity_weight: f32 = self.focal_entities.iter().map(|e| e.ic_weight).sum();

        let modifier_weight: f32 = self
            .discriminative_modifiers
            .iter()
            .map(|m| m.ic_weight)
            .sum();

        let relation_weight: f32 = self.relational_context.iter().map(|r| r.ic_weight).sum();

        entity_weight + modifier_weight + relation_weight
    }
}

/// Parse query using linguistic heuristics
///
/// This is a heuristic-based implementation. Future improvements:
/// - Use proper POS tagger (e.g., rust-stemmers + POS library)
/// - Train on domain-specific corpus
/// - Add NER (Named Entity Recognition)
pub fn analyze_query(query_text: &str) -> QueryAnalysis {
    let words: Vec<&str> = query_text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();

    let mut focal_entities = Vec::new();
    let mut discriminative_modifiers = Vec::new();
    let mut relational_context = Vec::new();

    for (i, word) in words.iter().enumerate() {
        let word_lower = word.to_lowercase();

        // Skip stop words (no information content)
        if is_stop_word(&word_lower) {
            continue;
        }

        // Classify by linguistic patterns
        if is_noun(&word_lower, i, &words) {
            focal_entities.push(FocalEntity {
                text: word_lower.clone(),
                ic_weight: IC_NOUN,
            });
        } else if is_adjective(&word_lower) {
            discriminative_modifiers.push(Modifier {
                text: word_lower.clone(),
                ic_weight: IC_ADJECTIVE,
            });
        } else if is_verb(&word_lower) {
            relational_context.push(Relation {
                text: word_lower.clone(),
                ic_weight: IC_VERB,
            });
        } else {
            // Unknown words likely nouns (domain-specific terms)
            focal_entities.push(FocalEntity {
                text: word_lower,
                ic_weight: IC_NOUN,
            });
        }
    }

    QueryAnalysis {
        focal_entities,
        discriminative_modifiers,
        relational_context,
        original_query: query_text.to_string(),
    }
}

/// Check if word is a noun (entity)
///
/// Heuristics:
/// - Common noun indicators
/// - Technical domain terms (robotics, navigation, etc.)
/// - Capitalized words (proper nouns)
fn is_noun(word: &str, position: usize, context: &[&str]) -> bool {
    // Common nouns in robotics/memory domain
    const NOUN_INDICATORS: &[&str] = &[
        // Robotics domain
        "robot",
        "drone",
        "sensor",
        "lidar",
        "camera",
        "motor",
        "actuator",
        "obstacle",
        "path",
        "waypoint",
        "location",
        "coordinates",
        "position",
        "battery",
        "power",
        "energy",
        "voltage",
        "current",
        "system",
        "module",
        "component",
        "unit",
        "device",
        "temperature",
        "pressure",
        "humidity",
        "speed",
        "velocity",
        "signal",
        "communication",
        "network",
        "link",
        "connection",
        "navigation",
        "guidance",
        "control",
        "steering",
        "data",
        "information",
        "message",
        "command",
        "response",
        // General nouns
        "person",
        "people",
        "user",
        "agent",
        "operator",
        "time",
        "date",
        "day",
        "hour",
        "minute",
        "second",
        "area",
        "zone",
        "region",
        "sector",
        "space",
        "task",
        "mission",
        "goal",
        "objective",
        "target",
        "error",
        "warning",
        "alert",
        "notification",
        "level",
        "status",
        "state",
        "condition",
        "mode",
    ];

    if NOUN_INDICATORS.contains(&word) {
        return true;
    }

    // Check for noun suffixes
    if word.ends_with("tion")
        || word.ends_with("sion")
        || word.ends_with("ment")
        || word.ends_with("ness")
        || word.ends_with("ity")
        || word.ends_with("ance")
        || word.ends_with("ence")
    {
        return true;
    }

    // Check if preceded by determiner (a, an, the)
    if position > 0 {
        let prev = context.get(position - 1).unwrap_or(&"").to_lowercase();
        if prev == "a" || prev == "an" || prev == "the" {
            return true;
        }
    }

    false
}

/// Check if word is an adjective (qualifier)
fn is_adjective(word: &str) -> bool {
    const ADJECTIVE_INDICATORS: &[&str] = &[
        // Colors
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "black",
        "white",
        "gray",
        "grey",
        "pink",
        "brown",
        // Sizes
        "big",
        "small",
        "large",
        "tiny",
        "huge",
        "massive",
        "mini",
        "micro",
        "high",
        "low",
        "tall",
        "short",
        "long",
        "wide",
        "narrow",
        // States
        "hot",
        "cold",
        "warm",
        "cool",
        "frozen",
        "heated",
        "fast",
        "slow",
        "quick",
        "rapid",
        "gradual",
        "active",
        "inactive",
        "enabled",
        "disabled",
        "on",
        "off",
        "open",
        "closed",
        "locked",
        "unlocked",
        "full",
        "empty",
        "partial",
        "complete",
        // Quality
        "good",
        "bad",
        "excellent",
        "poor",
        "optimal",
        "suboptimal",
        "normal",
        "abnormal",
        "stable",
        "unstable",
        "safe",
        "unsafe",
        "dangerous",
        "hazardous",
        "new",
        "old",
        "recent",
        "ancient",
        "current",
        "latest",
        // Robotics-specific
        "autonomous",
        "manual",
        "automatic",
        "remote",
        "digital",
        "analog",
        "electronic",
        "mechanical",
        "wireless",
        "wired",
        "connected",
        "disconnected",
    ];

    if ADJECTIVE_INDICATORS.contains(&word) {
        return true;
    }

    // Common adjective suffixes
    if word.ends_with("ful")
        || word.ends_with("less")
        || word.ends_with("ous")
        || word.ends_with("ive")
        || word.ends_with("able")
        || word.ends_with("ible")
        || word.ends_with("al")
        || word.ends_with("ic")
    {
        return true;
    }

    // Past participles used as adjectives
    if word.ends_with("ed") && !is_verb(word) {
        return true;
    }

    // Present participles used as adjectives
    if word.ends_with("ing") && !is_verb(word) {
        return true;
    }

    false
}

/// Check if word is a verb (relational, lower priority)
///
/// Verbs are "bus stops" - common across many queries, low discrimination
fn is_verb(word: &str) -> bool {
    const VERB_INDICATORS: &[&str] = &[
        // Common verbs (high frequency, low information)
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        // Action verbs (still common)
        "go",
        "goes",
        "went",
        "gone",
        "going",
        "get",
        "gets",
        "got",
        "gotten",
        "getting",
        "make",
        "makes",
        "made",
        "making",
        "take",
        "takes",
        "took",
        "taken",
        "taking",
        "see",
        "sees",
        "saw",
        "seen",
        "seeing",
        // Robotics action verbs
        "detected",
        "detect",
        "detects",
        "detecting",
        "found",
        "find",
        "finds",
        "finding",
        "observed",
        "observe",
        "observes",
        "observing",
        "measured",
        "measure",
        "measures",
        "measuring",
        "sensed",
        "sense",
        "senses",
        "sensing",
        "scanned",
        "scan",
        "scans",
        "scanning",
        "navigated",
        "navigate",
        "navigates",
        "navigating",
        "moved",
        "move",
        "moves",
        "moving",
        "stopped",
        "stop",
        "stops",
        "stopping",
        "started",
        "start",
        "starts",
        "starting",
        "reached",
        "reach",
        "reaches",
        "reaching",
        "avoided",
        "avoid",
        "avoids",
        "avoiding",
        "blocked",
        "block",
        "blocks",
        "blocking",
    ];

    VERB_INDICATORS.contains(&word)
}

/// Check if word is a stop word (no information content)
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "the", "this", "that", "these", "those", "at", "in", "on", "to", "for", "of",
        "from", "by", "with", "and", "or", "but", "not", "as", "if", "when", "where", "i", "you",
        "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "her", "its", "our", "their",
    ];

    STOP_WORDS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noun_detection() {
        let query = "robot detected obstacle at coordinates";
        let analysis = analyze_query(query);

        let noun_texts: Vec<String> = analysis
            .focal_entities
            .iter()
            .map(|e| e.text.clone())
            .collect();

        assert!(noun_texts.contains(&"robot".to_string()));
        assert!(noun_texts.contains(&"obstacle".to_string()));
        assert!(noun_texts.contains(&"coordinates".to_string()));
    }

    #[test]
    fn test_adjective_detection() {
        let query = "red large obstacle in path";
        let analysis = analyze_query(query);

        let adj_texts: Vec<String> = analysis
            .discriminative_modifiers
            .iter()
            .map(|m| m.text.clone())
            .collect();

        assert!(adj_texts.contains(&"red".to_string()));
        assert!(adj_texts.contains(&"large".to_string()));
    }

    #[test]
    fn test_verb_detection() {
        let query = "robot detected obstacle";
        let analysis = analyze_query(query);

        let verb_texts: Vec<String> = analysis
            .relational_context
            .iter()
            .map(|r| r.text.clone())
            .collect();

        assert!(verb_texts.contains(&"detected".to_string()));
    }

    #[test]
    fn test_information_content_weights() {
        let query = "sensor detected red obstacle";
        let analysis = analyze_query(query);

        // Nouns should have IC weight 2.3
        for entity in &analysis.focal_entities {
            assert_eq!(entity.ic_weight, IC_NOUN);
        }

        // Adjectives should have IC weight 1.7
        for modifier in &analysis.discriminative_modifiers {
            assert_eq!(modifier.ic_weight, IC_ADJECTIVE);
        }

        // Verbs should have IC weight 1.0
        for relation in &analysis.relational_context {
            assert_eq!(relation.ic_weight, IC_VERB);
        }
    }
}
