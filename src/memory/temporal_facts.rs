//! Temporal Fact Extraction and Storage
//!
//! Extracts and stores temporal facts from conversations for multi-hop reasoning.
//! Key insight: Multi-hop temporal queries like "When is X planning Y?" require:
//! 1. Finding the FIRST/PLANNING mention, not any mention
//! 2. Resolving relative dates ("next month", "last Saturday") to absolute dates
//! 3. Indexing by entity + event for fast lookup
//!
//! Storage schema:
//! - `temporal_facts:{user_id}:{fact_id}` - Primary storage
//! - `temporal_by_entity:{user_id}:{entity}:{fact_id}` - Entity index
//! - `temporal_by_event:{user_id}:{event_stem}:{fact_id}` - Event index

use anyhow::Result;
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc, Weekday};
use rocksdb::{IteratorMode, DB};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

use super::types::MemoryId;

/// Type of temporal event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Event was planned/scheduled ("We're planning to go camping next month")
    Planned,
    /// Event occurred/happened ("I ran a charity race last Saturday")
    Occurred,
    /// Event was mentioned in past tense referring to history ("I painted that in 2022")
    Historical,
    /// Recurring event ("We always go camping in summer")
    Recurring,
}

/// A temporal fact extracted from conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalFact {
    /// Unique identifier
    pub id: String,
    /// Entity involved (e.g., "Melanie", "Caroline")
    pub entity: String,
    /// Event/action (e.g., "camping", "charity race", "painted sunrise")
    pub event: String,
    /// Stemmed event keywords for matching
    pub event_stems: Vec<String>,
    /// Type of event (planned, occurred, historical)
    pub event_type: EventType,
    /// Original relative time expression ("next month", "last Saturday")
    pub relative_time: Option<String>,
    /// Resolved absolute time
    pub resolved_time: ResolvedTime,
    /// Source memory ID
    pub source_memory_id: MemoryId,
    /// Conversation timestamp (used to resolve relative dates)
    pub conversation_date: DateTime<Utc>,
    /// Confidence in extraction (0.0-1.0)
    pub confidence: f32,
    /// Original sentence fragment
    pub source_text: String,
}

/// Resolved time representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolvedTime {
    /// Exact date (e.g., "May 7, 2023")
    ExactDate(NaiveDate),
    /// Month and year (e.g., "June 2023")
    MonthYear { month: u32, year: i32 },
    /// Just year (e.g., "2022")
    Year(i32),
    /// Relative to conversation ("the week before", "next month")
    RelativeDescription(String),
    /// Unknown/couldn't resolve
    Unknown,
}

impl ResolvedTime {
    /// Convert to a sortable string for comparison
    pub fn to_sortable_string(&self) -> String {
        match self {
            ResolvedTime::ExactDate(d) => d.format("%Y-%m-%d").to_string(),
            ResolvedTime::MonthYear { month, year } => format!("{:04}-{:02}", year, month),
            ResolvedTime::Year(y) => format!("{:04}", y),
            ResolvedTime::RelativeDescription(s) => s.clone(),
            ResolvedTime::Unknown => "unknown".to_string(),
        }
    }

    /// Check if this time matches a query time expression
    pub fn matches_query(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();
        let self_str = self.to_sortable_string().to_lowercase();

        // Direct match
        if self_str.contains(&query_lower) || query_lower.contains(&self_str) {
            return true;
        }

        // Month name matching
        match self {
            ResolvedTime::MonthYear { month, year } => {
                let month_name = month_name(*month).to_lowercase();
                query_lower.contains(&month_name) && query_lower.contains(&year.to_string())
            }
            ResolvedTime::Year(y) => query_lower.contains(&y.to_string()),
            _ => false,
        }
    }
}

/// Storage for temporal facts
pub struct TemporalFactStore {
    db: Arc<DB>,
}

impl TemporalFactStore {
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Store a temporal fact
    pub fn store(&self, user_id: &str, fact: &TemporalFact) -> Result<()> {
        // Primary storage
        let key = format!("temporal_facts:{}:{}", user_id, fact.id);
        let value = bincode::serde::encode_to_vec(fact, bincode::config::standard())?;
        self.db.put(key.as_bytes(), &value)?;

        // Entity index
        let entity_key = format!(
            "temporal_by_entity:{}:{}:{}",
            user_id,
            fact.entity.to_lowercase(),
            fact.id
        );
        self.db.put(entity_key.as_bytes(), fact.id.as_bytes())?;

        // Event index (by each stem)
        for stem in &fact.event_stems {
            let event_key = format!("temporal_by_event:{}:{}:{}", user_id, stem, fact.id);
            self.db.put(event_key.as_bytes(), fact.id.as_bytes())?;
        }

        Ok(())
    }

    /// Store multiple facts
    pub fn store_batch(&self, user_id: &str, facts: &[TemporalFact]) -> Result<usize> {
        let mut stored = 0;
        for fact in facts {
            if self.store(user_id, fact).is_ok() {
                stored += 1;
            }
        }
        Ok(stored)
    }

    /// Get a fact by ID
    pub fn get(&self, user_id: &str, fact_id: &str) -> Result<Option<TemporalFact>> {
        let key = format!("temporal_facts:{}:{}", user_id, fact_id);
        match self.db.get(key.as_bytes())? {
            Some(data) => {
                let (fact, _): (TemporalFact, _) =
                    bincode::serde::decode_from_slice(&data, bincode::config::standard())?;
                Ok(Some(fact))
            }
            None => Ok(None),
        }
    }

    /// Find facts by entity
    pub fn find_by_entity(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
    ) -> Result<Vec<TemporalFact>> {
        let prefix = format!("temporal_by_entity:{}:{}:", user_id, entity.to_lowercase());
        self.find_by_prefix(&prefix, user_id, limit)
    }

    /// Find facts by event keyword
    pub fn find_by_event(
        &self,
        user_id: &str,
        event: &str,
        limit: usize,
    ) -> Result<Vec<TemporalFact>> {
        let stemmer = Stemmer::create(Algorithm::English);
        let stem = stemmer.stem(&event.to_lowercase()).to_string();
        let prefix = format!("temporal_by_event:{}:{}:", user_id, stem);
        self.find_by_prefix(&prefix, user_id, limit)
    }

    /// Find facts matching entity AND event
    pub fn find_by_entity_and_event(
        &self,
        user_id: &str,
        entity: &str,
        event_keywords: &[&str],
        event_type: Option<EventType>,
    ) -> Result<Vec<TemporalFact>> {
        // Get facts by entity
        let entity_facts = self.find_by_entity(user_id, entity, 100)?;

        // Filter by event keywords
        let stemmer = Stemmer::create(Algorithm::English);
        let event_stems: HashSet<String> = event_keywords
            .iter()
            .map(|kw| stemmer.stem(&kw.to_lowercase()).to_string())
            .collect();

        let mut matching: Vec<TemporalFact> = entity_facts
            .into_iter()
            .filter(|f| {
                // Check if any event stem matches
                let has_event_match = f.event_stems.iter().any(|s| event_stems.contains(s));
                // Check event type if specified
                let type_matches = event_type.map_or(true, |t| f.event_type == t);
                has_event_match && type_matches
            })
            .collect();

        // Sort by conversation date (earliest first for "planning" queries)
        matching.sort_by_key(|f| f.conversation_date);

        Ok(matching)
    }

    fn find_by_prefix(
        &self,
        prefix: &str,
        user_id: &str,
        limit: usize,
    ) -> Result<Vec<TemporalFact>> {
        let mut facts = Vec::new();
        let mut seen_ids = HashSet::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(prefix) {
                break;
            }

            let fact_id = String::from_utf8_lossy(&value);
            if seen_ids.insert(fact_id.to_string()) {
                if let Some(fact) = self.get(user_id, &fact_id)? {
                    facts.push(fact);
                    if facts.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(facts)
    }

    /// List all temporal facts for a user
    pub fn list(&self, user_id: &str, limit: usize) -> Result<Vec<TemporalFact>> {
        let prefix = format!("temporal_facts:{}:", user_id);
        let mut facts = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            if let Ok((fact, _)) = bincode::serde::decode_from_slice::<TemporalFact, _>(
                &value,
                bincode::config::standard(),
            ) {
                facts.push(fact);
                if facts.len() >= limit {
                    break;
                }
            }
        }

        Ok(facts)
    }
}

// ============================================================================
// TEMPORAL FACT EXTRACTION
// ============================================================================

/// Extract temporal facts from memory content
pub fn extract_temporal_facts(
    content: &str,
    memory_id: &MemoryId,
    conversation_date: DateTime<Utc>,
    entities: &[String],
) -> Vec<TemporalFact> {
    let mut facts = Vec::new();
    let stemmer = Stemmer::create(Algorithm::English);

    // Split into sentences
    let sentences: Vec<&str> = content
        .split(|c| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .collect();

    for sentence in sentences {
        let sentence_lower = sentence.to_lowercase();

        // Find which entity is mentioned in this sentence
        let mentioned_entity = entities
            .iter()
            .find(|e| sentence_lower.contains(&e.to_lowercase()));

        if mentioned_entity.is_none() {
            continue;
        }
        let entity = mentioned_entity.unwrap().clone();

        // Try to extract temporal facts using patterns
        if let Some(fact) =
            extract_planning_fact(sentence, &entity, memory_id, conversation_date, &stemmer)
        {
            facts.push(fact);
        } else if let Some(fact) =
            extract_occurred_fact(sentence, &entity, memory_id, conversation_date, &stemmer)
        {
            facts.push(fact);
        } else if let Some(fact) =
            extract_historical_fact(sentence, &entity, memory_id, conversation_date, &stemmer)
        {
            facts.push(fact);
        }
    }

    facts
}

/// Extract "planning" facts (future events)
fn extract_planning_fact(
    sentence: &str,
    entity: &str,
    memory_id: &MemoryId,
    conversation_date: DateTime<Utc>,
    stemmer: &Stemmer,
) -> Option<TemporalFact> {
    let sentence_lower = sentence.to_lowercase();

    // Planning patterns
    let planning_patterns = [
        "planning to",
        "planning on",
        "going to",
        "thinking about",
        "want to",
        "hope to",
        "looking forward to",
        "excited to",
        "next month",
        "next week",
        "this weekend",
        "soon",
    ];

    let has_planning = planning_patterns.iter().any(|p| sentence_lower.contains(p));
    if !has_planning {
        return None;
    }

    // Extract event and time
    let (event, relative_time) = extract_event_and_time(sentence, &planning_patterns);
    if event.is_empty() {
        return None;
    }

    // Resolve relative time
    let resolved_time = resolve_relative_time(&relative_time, conversation_date);

    let event_stems: Vec<String> = event
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
        .collect();

    Some(TemporalFact {
        id: format!("tf-{}-{}", memory_id.0, uuid::Uuid::new_v4()),
        entity: entity.to_string(),
        event: event.clone(),
        event_stems,
        event_type: EventType::Planned,
        relative_time: if relative_time.is_empty() {
            None
        } else {
            Some(relative_time)
        },
        resolved_time,
        source_memory_id: memory_id.clone(),
        conversation_date,
        confidence: 0.8,
        source_text: sentence.to_string(),
    })
}

/// Extract "occurred" facts (past events)
fn extract_occurred_fact(
    sentence: &str,
    entity: &str,
    memory_id: &MemoryId,
    conversation_date: DateTime<Utc>,
    stemmer: &Stemmer,
) -> Option<TemporalFact> {
    let sentence_lower = sentence.to_lowercase();

    // Past event patterns
    let occurred_patterns = [
        "last saturday",
        "last sunday",
        "last monday",
        "last tuesday",
        "last wednesday",
        "last thursday",
        "last friday",
        "last week",
        "last weekend",
        "last month",
        "yesterday",
        "this morning",
        "earlier today",
        "recently",
        "just",
        "ago",
        "two weeks ago",
        "a few days ago",
    ];

    // Check for past tense verbs combined with time patterns
    let past_verbs = [
        "ran", "went", "did", "took", "made", "had", "got", "saw", "met",
    ];
    let has_past_event = occurred_patterns.iter().any(|p| sentence_lower.contains(p))
        || past_verbs.iter().any(|v| sentence_lower.contains(v));

    if !has_past_event {
        return None;
    }

    // Extract event and time
    let (event, relative_time) = extract_event_and_time(sentence, &occurred_patterns);
    if event.is_empty() {
        return None;
    }

    // Resolve relative time
    let resolved_time = resolve_relative_time(&relative_time, conversation_date);

    let event_stems: Vec<String> = event
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
        .collect();

    Some(TemporalFact {
        id: format!("tf-{}-{}", memory_id.0, uuid::Uuid::new_v4()),
        entity: entity.to_string(),
        event: event.clone(),
        event_stems,
        event_type: EventType::Occurred,
        relative_time: if relative_time.is_empty() {
            None
        } else {
            Some(relative_time)
        },
        resolved_time,
        source_memory_id: memory_id.clone(),
        conversation_date,
        confidence: 0.75,
        source_text: sentence.to_string(),
    })
}

/// Extract "historical" facts (events in distant past)
fn extract_historical_fact(
    sentence: &str,
    entity: &str,
    memory_id: &MemoryId,
    conversation_date: DateTime<Utc>,
    stemmer: &Stemmer,
) -> Option<TemporalFact> {
    let sentence_lower = sentence.to_lowercase();

    // Look for year references (historical)
    let year_pattern = regex::Regex::new(r"\b(19|20)\d{2}\b").ok()?;
    let year_match = year_pattern.find(&sentence_lower)?;
    let year: i32 = year_match.as_str().parse().ok()?;

    // Only treat as historical if year is before conversation year
    if year >= conversation_date.year() {
        return None;
    }

    // Extract what happened
    let event = extract_event_from_sentence(sentence);
    if event.is_empty() {
        return None;
    }

    let event_stems: Vec<String> = event
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| stemmer.stem(&w.to_lowercase()).to_string())
        .collect();

    Some(TemporalFact {
        id: format!("tf-{}-{}", memory_id.0, uuid::Uuid::new_v4()),
        entity: entity.to_string(),
        event: event.clone(),
        event_stems,
        event_type: EventType::Historical,
        relative_time: Some(year.to_string()),
        resolved_time: ResolvedTime::Year(year),
        source_memory_id: memory_id.clone(),
        conversation_date,
        confidence: 0.85,
        source_text: sentence.to_string(),
    })
}

/// Extract event description and time expression from sentence
fn extract_event_and_time(sentence: &str, patterns: &[&str]) -> (String, String) {
    let sentence_lower = sentence.to_lowercase();

    // Find which pattern matched (used to verify a pattern exists)
    let _matched_pattern = patterns
        .iter()
        .find(|p| sentence_lower.contains(*p))
        .map(|s| *s)
        .unwrap_or("");

    // Extract time expression
    let relative_time = extract_time_expression(&sentence_lower);

    // Extract event (words around the pattern)
    let event = extract_event_from_sentence(sentence);

    (event, relative_time)
}

/// Extract time expression from sentence
fn extract_time_expression(sentence: &str) -> String {
    let time_patterns = [
        "next month",
        "next week",
        "next year",
        "this weekend",
        "this month",
        "last saturday",
        "last sunday",
        "last monday",
        "last tuesday",
        "last wednesday",
        "last thursday",
        "last friday",
        "last week",
        "last weekend",
        "last month",
        "last year",
        "yesterday",
        "today",
        "tomorrow",
        "two weeks ago",
        "a few days ago",
        "a week ago",
    ];

    for pattern in time_patterns {
        if sentence.contains(pattern) {
            return pattern.to_string();
        }
    }

    // Check for "X ago" pattern
    if let Some(idx) = sentence.find(" ago") {
        let before = &sentence[..idx];
        let words: Vec<&str> = before.split_whitespace().collect();
        if words.len() >= 2 {
            let last_two = words[words.len() - 2..].join(" ");
            return format!("{} ago", last_two);
        }
    }

    String::new()
}

/// Extract main event/action from sentence
fn extract_event_from_sentence(sentence: &str) -> String {
    let stopwords: HashSet<&str> = [
        "i",
        "we",
        "you",
        "they",
        "he",
        "she",
        "it",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "really",
        "also",
        "that",
        "this",
        "these",
        "those",
        "am",
        "going",
        "planning",
        "thinking",
        "want",
        "hope",
        "looking",
        "forward",
        "excited",
        "last",
        "next",
        "week",
        "month",
        "year",
        "saturday",
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "weekend",
        "ago",
        "recently",
    ]
    .into_iter()
    .collect();

    // Extract content words
    let words: Vec<&str> = sentence
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let w_lower = w.to_lowercase();
            w.len() > 2 && !stopwords.contains(w_lower.as_str())
        })
        .take(5)
        .collect();

    words.join(" ")
}

// ============================================================================
// RELATIVE DATE RESOLUTION
// ============================================================================

/// Resolve relative time expression to absolute time
pub fn resolve_relative_time(relative: &str, conversation_date: DateTime<Utc>) -> ResolvedTime {
    let relative_lower = relative.to_lowercase();
    let conv_date = conversation_date.date_naive();

    // "next month"
    if relative_lower.contains("next month") {
        let next_month = if conv_date.month() == 12 {
            (1, conv_date.year() + 1)
        } else {
            (conv_date.month() + 1, conv_date.year())
        };
        return ResolvedTime::MonthYear {
            month: next_month.0,
            year: next_month.1,
        };
    }

    // "last month"
    if relative_lower.contains("last month") {
        let last_month = if conv_date.month() == 1 {
            (12, conv_date.year() - 1)
        } else {
            (conv_date.month() - 1, conv_date.year())
        };
        return ResolvedTime::MonthYear {
            month: last_month.0,
            year: last_month.1,
        };
    }

    // "this month"
    if relative_lower.contains("this month") {
        return ResolvedTime::MonthYear {
            month: conv_date.month(),
            year: conv_date.year(),
        };
    }

    // "next week"
    if relative_lower.contains("next week") {
        let next_week = conv_date + Duration::days(7);
        return ResolvedTime::ExactDate(next_week);
    }

    // "last week"
    if relative_lower.contains("last week") {
        let last_week = conv_date - Duration::days(7);
        return ResolvedTime::ExactDate(last_week);
    }

    // "yesterday"
    if relative_lower.contains("yesterday") {
        return ResolvedTime::ExactDate(conv_date - Duration::days(1));
    }

    // "today"
    if relative_lower.contains("today") {
        return ResolvedTime::ExactDate(conv_date);
    }

    // "tomorrow"
    if relative_lower.contains("tomorrow") {
        return ResolvedTime::ExactDate(conv_date + Duration::days(1));
    }

    // "last saturday", "last sunday", etc.
    let weekdays = [
        ("sunday", Weekday::Sun),
        ("monday", Weekday::Mon),
        ("tuesday", Weekday::Tue),
        ("wednesday", Weekday::Wed),
        ("thursday", Weekday::Thu),
        ("friday", Weekday::Fri),
        ("saturday", Weekday::Sat),
    ];

    for (name, weekday) in weekdays {
        if relative_lower.contains(&format!("last {}", name)) {
            let resolved = last_weekday(conv_date, weekday);
            return ResolvedTime::ExactDate(resolved);
        }
        if relative_lower.contains(&format!("this {}", name))
            || relative_lower.contains(&format!("next {}", name))
        {
            let resolved = next_weekday(conv_date, weekday);
            return ResolvedTime::ExactDate(resolved);
        }
    }

    // "X weeks/days ago"
    if relative_lower.contains("ago") {
        if let Some(days) = parse_ago_expression(&relative_lower) {
            return ResolvedTime::ExactDate(conv_date - Duration::days(days));
        }
    }

    // "this weekend"
    if relative_lower.contains("this weekend") {
        let days_until_saturday = (Weekday::Sat.num_days_from_monday() as i64
            - conv_date.weekday().num_days_from_monday() as i64
            + 7)
            % 7;
        let saturday = conv_date + Duration::days(days_until_saturday);
        return ResolvedTime::ExactDate(saturday);
    }

    // Year only (e.g., "2022")
    if let Ok(year) = relative_lower.trim().parse::<i32>() {
        if (1900..2100).contains(&year) {
            return ResolvedTime::Year(year);
        }
    }

    ResolvedTime::RelativeDescription(relative.to_string())
}

/// Find the last occurrence of a weekday before the given date
fn last_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let current_weekday = from.weekday();
    let days_back =
        (current_weekday.num_days_from_monday() as i64 - target.num_days_from_monday() as i64 + 7)
            % 7;
    let days_back = if days_back == 0 { 7 } else { days_back };
    from - Duration::days(days_back)
}

/// Find the next occurrence of a weekday after the given date
fn next_weekday(from: NaiveDate, target: Weekday) -> NaiveDate {
    let current_weekday = from.weekday();
    let days_forward =
        (target.num_days_from_monday() as i64 - current_weekday.num_days_from_monday() as i64 + 7)
            % 7;
    let days_forward = if days_forward == 0 { 7 } else { days_forward };
    from + Duration::days(days_forward)
}

/// Parse "X days/weeks ago" expressions
fn parse_ago_expression(expr: &str) -> Option<i64> {
    let words: Vec<&str> = expr.split_whitespace().collect();

    // Find "ago" position
    let ago_pos = words.iter().position(|w| *w == "ago")?;
    if ago_pos < 2 {
        return None;
    }

    let unit = words[ago_pos - 1];
    let num_str = words[ago_pos - 2];

    let num: i64 = match num_str {
        "a" | "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "few" => 3,
        "couple" => 2,
        _ => num_str.parse().ok()?,
    };

    let multiplier = match unit {
        "day" | "days" => 1,
        "week" | "weeks" => 7,
        "month" | "months" => 30,
        "year" | "years" => 365,
        _ => return None,
    };

    Some(num * multiplier)
}

/// Get month name from number
fn month_name(month: u32) -> &'static str {
    match month {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        12 => "December",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_next_month() {
        let conv_date = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let resolved = resolve_relative_time("next month", conv_date);

        match resolved {
            ResolvedTime::MonthYear { month, year } => {
                assert_eq!(month, 6);
                assert_eq!(year, 2023);
            }
            _ => panic!("Expected MonthYear"),
        }
    }

    #[test]
    fn test_resolve_last_saturday() {
        // May 25, 2023 is a Thursday
        let conv_date = DateTime::parse_from_rfc3339("2023-05-25T13:14:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let resolved = resolve_relative_time("last saturday", conv_date);

        match resolved {
            ResolvedTime::ExactDate(d) => {
                // Last Saturday before May 25 (Thursday) is May 20
                assert_eq!(d.to_string(), "2023-05-20");
            }
            _ => panic!("Expected ExactDate"),
        }
    }

    #[test]
    fn test_extract_event() {
        let sentence = "We're thinking about going camping next month";
        let event = extract_event_from_sentence(sentence);
        assert!(event.to_lowercase().contains("camping"));
    }
}
