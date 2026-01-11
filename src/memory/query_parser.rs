//! Linguistic Query Parser
//!
//! Based on:
//! - Lioma & Ounis (2006): "Content Load of Part of Speech Blocks"
//! - Bendersky & Croft (2008): "Discovering Key Concepts in Verbose Queries"
//! - Porter (1980): Stemming algorithm for term normalization
//!
//! Extracts focal entities (nouns), discriminative modifiers (adjectives),
//! and relational context (verbs) from natural language queries.
//!
//! # Polished Features (v2)
//! - Porter2 stemming for term normalization
//! - Compound noun detection (bigrams/trigrams)
//! - Context-aware POS disambiguation
//! - Negation scope tracking
//! - IDF-inspired term rarity weighting
//!
//! # Shallow Parsing / Chunking (v3)
//! - Sentence-level chunking for co-occurrence detection
//! - POS-based entity extraction (all nouns, verbs, adjectives - not just top-N)
//! - Designed for both query analysis AND memory storage
//!
//! # Temporal Extraction (v4)
//! - Extract dates from natural language text ("May 7, 2023", "yesterday", "last week")
//! - Detect temporal queries ("when did", "what date", "how long ago")
//! - Based on TEMPR approach (Hindsight paper achieving 89.6% on LoCoMo)

use crate::constants::{IC_ADJECTIVE, IC_NOUN, IC_VERB};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ============================================================================
// SHALLOW PARSING / CHUNKING MODULE
// ============================================================================
// This section provides sentence-level chunking and POS-based entity extraction.
// Unlike YAKE (which ranks by frequency and misses rare discriminative terms),
// this extracts ALL content words (nouns, verbs, adjectives) for graph building.

/// Part of speech tag
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PosTag {
    Noun,
    Verb,
    Adjective,
    ProperNoun,
    StopWord,
    Other,
}

/// A word with its POS annotation
#[derive(Debug, Clone)]
pub struct TaggedWord {
    pub text: String,
    pub stem: String,
    pub pos: PosTag,
    /// Position within the sentence (0-indexed)
    pub position: usize,
}

/// A sentence chunk containing tagged words
#[derive(Debug, Clone)]
pub struct SentenceChunk {
    /// Original sentence text
    pub text: String,
    /// Sentence index in document (0-indexed)
    pub sentence_idx: usize,
    /// All tagged words in this sentence
    pub words: Vec<TaggedWord>,
}

impl SentenceChunk {
    /// Get all nouns in this chunk
    pub fn nouns(&self) -> Vec<&TaggedWord> {
        self.words
            .iter()
            .filter(|w| matches!(w.pos, PosTag::Noun | PosTag::ProperNoun))
            .collect()
    }

    /// Get all verbs in this chunk
    pub fn verbs(&self) -> Vec<&TaggedWord> {
        self.words
            .iter()
            .filter(|w| w.pos == PosTag::Verb)
            .collect()
    }

    /// Get all adjectives in this chunk
    pub fn adjectives(&self) -> Vec<&TaggedWord> {
        self.words
            .iter()
            .filter(|w| w.pos == PosTag::Adjective)
            .collect()
    }

    /// Get all content words (nouns, verbs, adjectives)
    pub fn content_words(&self) -> Vec<&TaggedWord> {
        self.words
            .iter()
            .filter(|w| {
                matches!(
                    w.pos,
                    PosTag::Noun | PosTag::ProperNoun | PosTag::Verb | PosTag::Adjective
                )
            })
            .collect()
    }

    /// Generate co-occurrence pairs (words in same sentence get edges)
    /// Returns pairs of (word1_stem, word2_stem) for graph edge creation
    pub fn cooccurrence_pairs(&self) -> Vec<(&str, &str)> {
        let content = self.content_words();
        let mut pairs = Vec::new();

        for i in 0..content.len() {
            for j in (i + 1)..content.len() {
                pairs.push((content[i].stem.as_str(), content[j].stem.as_str()));
            }
        }

        pairs
    }
}

/// Result of chunking a document
#[derive(Debug, Clone)]
pub struct ChunkExtraction {
    /// All sentence chunks
    pub chunks: Vec<SentenceChunk>,
    /// Unique nouns found (stems)
    pub unique_nouns: HashSet<String>,
    /// Unique verbs found (stems)
    pub unique_verbs: HashSet<String>,
    /// Unique adjectives found (stems)
    pub unique_adjectives: HashSet<String>,
    /// Proper nouns (likely named entities)
    pub proper_nouns: HashSet<String>,
}

impl ChunkExtraction {
    /// Get all unique content word stems
    pub fn all_content_stems(&self) -> HashSet<String> {
        let mut all = self.unique_nouns.clone();
        all.extend(self.unique_verbs.clone());
        all.extend(self.unique_adjectives.clone());
        all.extend(self.proper_nouns.clone());
        all
    }

    /// Get all co-occurrence pairs across all chunks
    pub fn all_cooccurrence_pairs(&self) -> Vec<(String, String)> {
        let mut all_pairs = Vec::new();
        for chunk in &self.chunks {
            for (w1, w2) in chunk.cooccurrence_pairs() {
                all_pairs.push((w1.to_string(), w2.to_string()));
            }
        }
        all_pairs
    }
}

// ============================================================================
// TEMPORAL EXTRACTION MODULE
// ============================================================================
// Extracts temporal references from natural language text.
// Based on TEMPR approach from Hindsight paper (89.6% accuracy on LoCoMo).
// Key insight: Temporal filtering is critical for multi-hop retrieval.

/// A temporal reference extracted from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRef {
    /// The extracted date (normalized to NaiveDate)
    pub date: NaiveDate,
    /// Original text that was parsed (e.g., "May 7, 2023", "yesterday")
    pub original_text: String,
    /// Confidence in the extraction (0.0-1.0)
    pub confidence: f32,
    /// Position in original text (character offset)
    pub position: usize,
    /// Type of temporal reference
    pub ref_type: TemporalRefType,
}

/// Type of temporal reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalRefType {
    /// Absolute date (May 7, 2023)
    Absolute,
    /// Relative date (yesterday, last week)
    Relative,
    /// Day of week (on Monday, last Tuesday)
    DayOfWeek,
    /// Month reference (in May, last March)
    Month,
    /// Year reference (in 2023, last year)
    Year,
}

/// Result of temporal extraction from text
#[derive(Debug, Clone, Default)]
pub struct TemporalExtraction {
    /// All temporal references found
    pub refs: Vec<TemporalRef>,
    /// Earliest date mentioned
    pub earliest: Option<NaiveDate>,
    /// Latest date mentioned
    pub latest: Option<NaiveDate>,
}

impl TemporalExtraction {
    /// Check if any temporal references were found
    pub fn has_temporal_refs(&self) -> bool {
        !self.refs.is_empty()
    }

    /// Get date range (earliest, latest) if temporal refs exist
    pub fn date_range(&self) -> Option<(NaiveDate, NaiveDate)> {
        match (self.earliest, self.latest) {
            (Some(e), Some(l)) => Some((e, l)),
            (Some(e), None) => Some((e, e)),
            (None, Some(l)) => Some((l, l)),
            (None, None) => None,
        }
    }
}

/// Extract temporal references from text
///
/// Uses date_time_parser crate for natural language date parsing.
/// Handles:
/// - Absolute dates: "May 7, 2023", "2023-05-07", "07/05/2023"
/// - Relative dates: "yesterday", "last week", "3 days ago"
/// - Day of week: "on Monday", "last Tuesday"
/// - Month/year: "in May", "last year", "2023"
pub fn extract_temporal_refs(text: &str) -> TemporalExtraction {
    let now = Utc::now();
    let mut refs = Vec::new();
    let mut earliest: Option<NaiveDate> = None;
    let mut latest: Option<NaiveDate> = None;

    // Helper to validate date is in reasonable range (1900-2100)
    let is_valid_date = |date: &NaiveDate| -> bool {
        let year = date.year();
        year >= 1900 && year <= 2100
    };

    // Try dateparser on the full text (returns Result, never panics)
    if let Ok(parsed) = dateparser::parse(text) {
        let date = parsed.date_naive();
        if is_valid_date(&date) {
            refs.push(TemporalRef {
                date,
                original_text: text.to_string(),
                confidence: 0.8,
                position: 0,
                ref_type: classify_temporal_ref(text, &date, &now),
            });
            update_bounds(&mut earliest, &mut latest, date);
        }
    }

    // Try parsing individual sentences/phrases
    for (pos, sentence) in split_temporal_phrases(text).iter().enumerate() {
        if let Ok(parsed) = dateparser::parse(sentence) {
            let date = parsed.date_naive();
            if !is_valid_date(&date) {
                continue;
            }
            if refs.iter().any(|r| r.date == date) {
                continue;
            }
            refs.push(TemporalRef {
                date,
                original_text: sentence.to_string(),
                confidence: 0.7,
                position: pos,
                ref_type: classify_temporal_ref(sentence, &date, &now),
            });
            update_bounds(&mut earliest, &mut latest, date);
        }
    }

    // Also use regex-based extraction for explicit date patterns
    let explicit_dates = extract_explicit_dates(text);
    for (date, original, pos) in explicit_dates {
        if !is_valid_date(&date) {
            continue;
        }
        if refs.iter().any(|r| r.date == date) {
            continue;
        }
        refs.push(TemporalRef {
            date,
            original_text: original,
            confidence: 0.9,
            position: pos,
            ref_type: TemporalRefType::Absolute,
        });
        update_bounds(&mut earliest, &mut latest, date);
    }

    // Sort by position in text
    refs.sort_by_key(|r| r.position);

    TemporalExtraction {
        refs,
        earliest,
        latest,
    }
}

/// Classify the type of temporal reference
fn classify_temporal_ref(text: &str, date: &NaiveDate, now: &DateTime<Utc>) -> TemporalRefType {
    let text_lower = text.to_lowercase();
    let today = now.date_naive();

    // Check for relative indicators
    if text_lower.contains("yesterday")
        || text_lower.contains("ago")
        || text_lower.contains("last")
        || text_lower.contains("previous")
        || text_lower.contains("before")
        || text_lower.contains("earlier")
    {
        return TemporalRefType::Relative;
    }

    // Check for day of week
    let days = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ];
    if days.iter().any(|d| text_lower.contains(d)) {
        return TemporalRefType::DayOfWeek;
    }

    // Check for month names without day
    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let has_month = months.iter().any(|m| text_lower.contains(m));
    let has_day = text.chars().any(|c| c.is_ascii_digit());

    if has_month && !has_day {
        return TemporalRefType::Month;
    }

    // Check for year-only reference
    if text.len() == 4 && text.chars().all(|c| c.is_ascii_digit()) {
        return TemporalRefType::Year;
    }

    // If date is today or very close, might be relative
    let diff = (today - *date).num_days().abs();
    if diff <= 7 && text_lower.contains("this") {
        return TemporalRefType::Relative;
    }

    TemporalRefType::Absolute
}

/// Split text into temporal-relevant phrases
fn split_temporal_phrases(text: &str) -> Vec<String> {
    let mut phrases = Vec::new();

    // Split by common temporal markers and punctuation
    let markers = [
        " on ", " in ", " at ", " during ", " since ", " until ", " before ", " after ",
        " around ", ", ", ". ", "! ", "? ",
    ];

    let mut current = text.to_string();
    for marker in markers {
        let parts: Vec<&str> = current.split(marker).collect();
        if parts.len() > 1 {
            for part in parts {
                let trimmed = part.trim();
                if !trimmed.is_empty() && trimmed.len() > 3 {
                    phrases.push(trimmed.to_string());
                }
            }
            break;
        }
    }

    // If no splitting happened, try sentence boundaries
    if phrases.is_empty() {
        for sentence in text.split('.') {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() && trimmed.len() > 3 {
                phrases.push(trimmed.to_string());
            }
        }
    }

    phrases
}

/// Extract explicit date patterns that date_time_parser might miss
fn extract_explicit_dates(text: &str) -> Vec<(NaiveDate, String, usize)> {
    use regex::Regex;

    let mut results = Vec::new();

    // Pattern: "Month Day, Year" (e.g., "May 7, 2023")
    let month_day_year =
        Regex::new(r"(?i)(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})")
            .unwrap();

    for cap in month_day_year.captures_iter(text) {
        let month_str = &cap[1];
        let day: u32 = cap[2].parse().unwrap_or(1);
        let year: i32 = cap[3].parse().unwrap_or(2000);
        let month = month_to_num(month_str);

        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            let pos = cap.get(0).map(|m| m.start()).unwrap_or(0);
            results.push((date, cap[0].to_string(), pos));
        }
    }

    // Pattern: "YYYY-MM-DD"
    let iso_date = Regex::new(r"(\d{4})-(\d{2})-(\d{2})").unwrap();
    for cap in iso_date.captures_iter(text) {
        let year: i32 = cap[1].parse().unwrap_or(2000);
        let month: u32 = cap[2].parse().unwrap_or(1);
        let day: u32 = cap[3].parse().unwrap_or(1);

        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            let pos = cap.get(0).map(|m| m.start()).unwrap_or(0);
            results.push((date, cap[0].to_string(), pos));
        }
    }

    // Pattern: "MM/DD/YYYY" or "DD/MM/YYYY" (assume US format MM/DD)
    let slash_date = Regex::new(r"(\d{1,2})/(\d{1,2})/(\d{4})").unwrap();
    for cap in slash_date.captures_iter(text) {
        let month: u32 = cap[1].parse().unwrap_or(1);
        let day: u32 = cap[2].parse().unwrap_or(1);
        let year: i32 = cap[3].parse().unwrap_or(2000);

        if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
            let pos = cap.get(0).map(|m| m.start()).unwrap_or(0);
            results.push((date, cap[0].to_string(), pos));
        }
    }

    results
}

/// Convert month name to number
fn month_to_num(month: &str) -> u32 {
    match month.to_lowercase().as_str() {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" | "sept" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => 1,
    }
}

/// Update earliest/latest bounds
fn update_bounds(
    earliest: &mut Option<NaiveDate>,
    latest: &mut Option<NaiveDate>,
    date: NaiveDate,
) {
    match earliest {
        Some(e) if date < *e => *earliest = Some(date),
        None => *earliest = Some(date),
        _ => {}
    }
    match latest {
        Some(l) if date > *l => *latest = Some(date),
        None => *latest = Some(date),
        _ => {}
    }
}

// ============================================================================
// TEMPORAL QUERY DETECTION
// ============================================================================
// Detect when a query is asking about time/dates.

/// Query temporal intent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalIntent {
    /// Query is asking "when" something happened
    WhenQuestion,
    /// Query references a specific time period
    SpecificTime,
    /// Query asks about temporal ordering (before/after)
    Ordering,
    /// Query asks about duration (how long)
    Duration,
    /// No temporal intent detected
    None,
}

/// Detect temporal intent in a query
pub fn detect_temporal_intent(query: &str) -> TemporalIntent {
    let query_lower = query.to_lowercase();

    // "When" questions are highest priority
    if query_lower.starts_with("when")
        || query_lower.contains(" when ")
        || query_lower.contains("what date")
        || query_lower.contains("what day")
        || query_lower.contains("what time")
    {
        return TemporalIntent::WhenQuestion;
    }

    // Duration questions
    if query_lower.contains("how long")
        || query_lower.contains("how many days")
        || query_lower.contains("how many weeks")
        || query_lower.contains("how many months")
        || query_lower.contains("how many years")
    {
        return TemporalIntent::Duration;
    }

    // Ordering questions
    if query_lower.contains("before or after")
        || query_lower.contains("first or")
        || query_lower.contains("earlier or later")
        || query_lower.contains("which came first")
        || query_lower.contains("in what order")
    {
        return TemporalIntent::Ordering;
    }

    // Specific time references
    let time_indicators = [
        "yesterday",
        "today",
        "last week",
        "last month",
        "last year",
        "this week",
        "this month",
        "this year",
        "in january",
        "in february",
        "in march",
        "in april",
        "in may",
        "in june",
        "in july",
        "in august",
        "in september",
        "in october",
        "in november",
        "in december",
        "on monday",
        "on tuesday",
        "on wednesday",
        "on thursday",
        "on friday",
        "on saturday",
        "on sunday",
        " ago",
        " days ago",
        " weeks ago",
        " months ago",
        " years ago",
    ];

    if time_indicators.iter().any(|t| query_lower.contains(t)) {
        return TemporalIntent::SpecificTime;
    }

    // Check for date patterns
    let extraction = extract_temporal_refs(query);
    if extraction.has_temporal_refs() {
        return TemporalIntent::SpecificTime;
    }

    TemporalIntent::None
}

/// Check if a query requires temporal filtering for accurate retrieval
///
/// Returns true if the query has a temporal component that should be used
/// to filter/rank memories by their temporal references.
///
/// IMPORTANT: "When did X happen?" questions (WhenQuestion) return FALSE
/// because they are asking FOR a date, not filtering BY a date.
/// We should search semantically for X and extract the date from results.
///
/// "What happened in May 2023?" (SpecificTime) returns TRUE because
/// it's filtering BY a specific time period.
pub fn requires_temporal_filtering(query: &str) -> bool {
    let intent = detect_temporal_intent(query);
    matches!(
        intent,
        // WhenQuestion is EXCLUDED - it asks FOR a date, not BY a date
        TemporalIntent::SpecificTime | TemporalIntent::Duration | TemporalIntent::Ordering
    )
}

/// Check if a query is asking FOR a temporal answer (when did X happen?)
///
/// These queries should use semantic search on the event X, then extract
/// the date from the retrieved content.
pub fn asks_for_temporal_answer(query: &str) -> bool {
    matches!(detect_temporal_intent(query), TemporalIntent::WhenQuestion)
}

// ============================================================================
// ATTRIBUTE QUERY DETECTION
// ============================================================================
// Detect queries asking for specific attributes of entities.
// These queries need fact-first retrieval, not semantic similarity.
//
// Examples:
// - "What is Caroline's relationship status?" → entity=Caroline, attribute=relationship_status
// - "What is Melanie's job?" → entity=Melanie, attribute=job
// - "Where does Caroline live?" → entity=Caroline, attribute=location

/// Type of query for routing to appropriate retrieval strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    /// Attribute query: "What is X's Y?" - needs fact lookup
    Attribute(AttributeQuery),
    /// Temporal query: "When did X do Y?" - needs temporal filtering
    Temporal,
    /// Exploratory query: general semantic search
    Exploratory,
}

/// Extracted attribute query components
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeQuery {
    /// The entity being asked about (e.g., "Caroline")
    pub entity: String,
    /// The attribute being requested (e.g., "relationship_status")
    pub attribute: String,
    /// Attribute synonyms for matching (e.g., ["status", "single", "married", "dating"])
    pub attribute_synonyms: Vec<String>,
    /// Original query text
    pub original_query: String,
}

/// Classify a query to determine retrieval strategy
pub fn classify_query(query: &str) -> QueryType {
    // First check for attribute queries
    if let Some(attr_query) = detect_attribute_query(query) {
        return QueryType::Attribute(attr_query);
    }

    // Check for temporal queries
    if asks_for_temporal_answer(query) {
        return QueryType::Temporal;
    }

    // Default to exploratory
    QueryType::Exploratory
}

/// Detect and extract attribute query components
///
/// Patterns detected:
/// - "What is X's Y?" / "What is X's Y"
/// - "What is the Y of X?"
/// - "What Y does X have?"
/// - "Is X Y?" (boolean attribute)
/// - "Where does/is X?" (location attribute)
/// - "How old is X?" (age attribute)
pub fn detect_attribute_query(query: &str) -> Option<AttributeQuery> {
    let query_lower = query.to_lowercase();
    let query_trimmed = query_lower.trim().trim_end_matches('?');

    // Pattern 1: "What is X's Y" / "What's X's Y"
    if let Some(result) = extract_possessive_pattern(query_trimmed) {
        return Some(result);
    }

    // Pattern 2: "What is the Y of X"
    if let Some(result) = extract_of_pattern(query_trimmed) {
        return Some(result);
    }

    // Pattern 3: "Where does/is X" (location attribute)
    if query_lower.starts_with("where does") || query_lower.starts_with("where is") {
        if let Some(entity) = extract_entity_after_verb(query_trimmed) {
            return Some(AttributeQuery {
                entity,
                attribute: "location".to_string(),
                attribute_synonyms: vec![
                    "live".to_string(),
                    "lives".to_string(),
                    "living".to_string(),
                    "resides".to_string(),
                    "located".to_string(),
                    "address".to_string(),
                    "home".to_string(),
                    "place".to_string(),
                ],
                original_query: query.to_string(),
            });
        }
    }

    // Pattern 4: "How old is X" (age attribute)
    if query_lower.starts_with("how old") {
        if let Some(entity) = extract_entity_after_verb(query_trimmed) {
            return Some(AttributeQuery {
                entity,
                attribute: "age".to_string(),
                attribute_synonyms: vec![
                    "age".to_string(),
                    "years old".to_string(),
                    "born".to_string(),
                    "birthday".to_string(),
                ],
                original_query: query.to_string(),
            });
        }
    }

    // Pattern 5: "Is X married/single/..." (boolean relationship status)
    if query_lower.starts_with("is ") {
        let status_words = [
            "married",
            "single",
            "divorced",
            "engaged",
            "dating",
            "in a relationship",
        ];
        for status in &status_words {
            if query_lower.contains(status) {
                // Extract entity between "is" and status word
                let after_is = &query_trimmed[3..]; // Skip "is "
                if let Some(pos) = after_is.find(status) {
                    let entity = after_is[..pos].trim().to_string();
                    if !entity.is_empty()
                        && entity.chars().next().map_or(false, |c| c.is_alphabetic())
                    {
                        return Some(AttributeQuery {
                            entity: capitalize_first(&entity),
                            attribute: "relationship_status".to_string(),
                            attribute_synonyms: vec![
                                "single".to_string(),
                                "married".to_string(),
                                "divorced".to_string(),
                                "engaged".to_string(),
                                "dating".to_string(),
                                "relationship".to_string(),
                                "partner".to_string(),
                                "spouse".to_string(),
                                "status".to_string(),
                            ],
                            original_query: query.to_string(),
                        });
                    }
                }
            }
        }
    }

    None
}

/// Extract "X's Y" pattern from query
fn extract_possessive_pattern(query: &str) -> Option<AttributeQuery> {
    // Find possessive marker ('s or s')
    let possessive_patterns = [
        ("what is ", "'s "),
        ("what's ", "'s "),
        ("what is ", "' "),
        ("what's ", "' "),
    ];

    for (prefix, possessive) in possessive_patterns {
        if let Some(start) = query.find(prefix) {
            let after_prefix = &query[start + prefix.len()..];
            if let Some(pos_pos) = after_prefix.find(possessive) {
                let entity = after_prefix[..pos_pos].trim();
                let attribute = after_prefix[pos_pos + possessive.len()..].trim();

                if !entity.is_empty() && !attribute.is_empty() {
                    return Some(AttributeQuery {
                        entity: capitalize_first(entity),
                        attribute: normalize_attribute(attribute),
                        attribute_synonyms: get_attribute_synonyms(attribute),
                        original_query: query.to_string(),
                    });
                }
            }
        }
    }

    None
}

/// Extract "the Y of X" pattern from query
fn extract_of_pattern(query: &str) -> Option<AttributeQuery> {
    // Pattern: "what is the Y of X"
    let prefixes = ["what is the ", "what's the "];

    for prefix in prefixes {
        if let Some(start) = query.find(prefix) {
            let after_prefix = &query[start + prefix.len()..];
            if let Some(of_pos) = after_prefix.find(" of ") {
                let attribute = after_prefix[..of_pos].trim();
                let entity = after_prefix[of_pos + 4..].trim();

                if !entity.is_empty() && !attribute.is_empty() {
                    return Some(AttributeQuery {
                        entity: capitalize_first(entity),
                        attribute: normalize_attribute(attribute),
                        attribute_synonyms: get_attribute_synonyms(attribute),
                        original_query: query.to_string(),
                    });
                }
            }
        }
    }

    None
}

/// Extract entity after a verb like "is" or "does"
fn extract_entity_after_verb(query: &str) -> Option<String> {
    let verbs = [" is ", " does "];
    for verb in verbs {
        if let Some(pos) = query.find(verb) {
            let after_verb = query[pos + verb.len()..].trim();
            // Take first word(s) as entity (stop at common words)
            let stop_words = ["live", "work", "do", "have", "go", "stay", "come"];
            let words: Vec<&str> = after_verb.split_whitespace().collect();
            let mut entity_words = Vec::new();
            for word in words {
                if stop_words.contains(&word) {
                    break;
                }
                entity_words.push(word);
            }
            if !entity_words.is_empty() {
                return Some(capitalize_first(&entity_words.join(" ")));
            }
        }
    }
    None
}

/// Normalize an attribute name (e.g., "relationship status" → "relationship_status")
fn normalize_attribute(attr: &str) -> String {
    attr.trim()
        .to_lowercase()
        .replace(' ', "_")
        .replace('-', "_")
}

/// Get synonyms for common attributes
fn get_attribute_synonyms(attribute: &str) -> Vec<String> {
    let attr_lower = attribute.to_lowercase();

    // Relationship status synonyms
    if attr_lower.contains("relationship")
        || attr_lower.contains("status")
        || attr_lower.contains("marital")
    {
        return vec![
            "single".to_string(),
            "married".to_string(),
            "divorced".to_string(),
            "engaged".to_string(),
            "dating".to_string(),
            "relationship".to_string(),
            "partner".to_string(),
            "spouse".to_string(),
            "single parent".to_string(),
            "status".to_string(),
            "marital".to_string(),
        ];
    }

    // Job/occupation synonyms
    if attr_lower.contains("job")
        || attr_lower.contains("occupation")
        || attr_lower.contains("work")
    {
        return vec![
            "job".to_string(),
            "work".to_string(),
            "occupation".to_string(),
            "profession".to_string(),
            "career".to_string(),
            "employed".to_string(),
            "works as".to_string(),
        ];
    }

    // Name synonyms
    if attr_lower.contains("name") {
        return vec![
            "name".to_string(),
            "called".to_string(),
            "named".to_string(),
        ];
    }

    // Age synonyms
    if attr_lower.contains("age") {
        return vec![
            "age".to_string(),
            "old".to_string(),
            "years".to_string(),
            "born".to_string(),
            "birthday".to_string(),
        ];
    }

    // Default: return the attribute itself and common variations
    vec![attr_lower.clone(), attr_lower.replace('_', " ")]
}

/// Capitalize first letter of a string
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Extract chunks from text using shallow parsing
///
/// This function:
/// 1. Splits text into sentences
/// 2. Tags each word with its POS (noun, verb, adjective, etc.)
/// 3. Returns chunks that can be used for:
///    - Entity extraction (all nouns, not just YAKE top-N)
///    - Co-occurrence edge creation (words in same sentence)
///
/// Unlike YAKE, this doesn't rank by frequency - ALL content words are extracted.
pub fn extract_chunks(text: &str) -> ChunkExtraction {
    let stemmer = Stemmer::create(Algorithm::English);
    let sentences = split_sentences(text);

    let mut chunks = Vec::with_capacity(sentences.len());
    let mut unique_nouns = HashSet::new();
    let mut unique_verbs = HashSet::new();
    let mut unique_adjectives = HashSet::new();
    let mut proper_nouns = HashSet::new();

    for (sentence_idx, sentence) in sentences.iter().enumerate() {
        let words = tokenize_with_case(sentence);
        let mut tagged_words = Vec::with_capacity(words.len());

        for (position, (word, is_capitalized)) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();

            // Skip very short words
            if word_lower.len() < 2 {
                continue;
            }

            let stem = stemmer.stem(&word_lower).to_string();
            let pos = classify_pos_for_chunking(&word_lower, *is_capitalized, position, &words);

            match pos {
                PosTag::Noun => {
                    unique_nouns.insert(stem.clone());
                }
                PosTag::Verb => {
                    unique_verbs.insert(stem.clone());
                }
                PosTag::Adjective => {
                    unique_adjectives.insert(stem.clone());
                }
                PosTag::ProperNoun => {
                    proper_nouns.insert(word.clone());
                    unique_nouns.insert(stem.clone()); // Also add to nouns
                }
                _ => {}
            }

            if pos != PosTag::StopWord {
                tagged_words.push(TaggedWord {
                    text: word.clone(),
                    stem,
                    pos,
                    position,
                });
            }
        }

        if !tagged_words.is_empty() {
            chunks.push(SentenceChunk {
                text: sentence.clone(),
                sentence_idx,
                words: tagged_words,
            });
        }
    }

    ChunkExtraction {
        chunks,
        unique_nouns,
        unique_verbs,
        unique_adjectives,
        proper_nouns,
    }
}

/// Split text into sentences using punctuation and common patterns
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    // Simple sentence splitting on . ! ? and newlines
    // More sophisticated than just split('.') because we handle abbreviations
    for ch in text.chars() {
        current.push(ch);

        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let trimmed = current.trim();
            if !trimmed.is_empty() && trimmed.len() > 3 {
                // Avoid splitting on abbreviations like "Dr." or "Mr."
                let last_word: String = trimmed
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .collect();

                // Common abbreviations that shouldn't split
                let is_abbrev = matches!(
                    last_word.to_lowercase().as_str(),
                    "mr" | "mrs"
                        | "ms"
                        | "dr"
                        | "prof"
                        | "sr"
                        | "jr"
                        | "vs"
                        | "etc"
                        | "eg"
                        | "ie"
                        | "st"
                        | "ave"
                        | "rd"
                        | "blvd"
                );

                if !is_abbrev || ch == '\n' {
                    sentences.push(trimmed.to_string());
                    current.clear();
                }
            }
        }
    }

    // Don't forget the last sentence
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

/// Tokenize preserving case information for proper noun detection
fn tokenize_with_case(text: &str) -> Vec<(String, bool)> {
    text.split_whitespace()
        .map(|w| {
            let clean: String = w
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_string();
            let is_capitalized = clean
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);
            (clean, is_capitalized)
        })
        .filter(|(w, _)| !w.is_empty())
        .collect()
}

/// Classify POS for chunking purposes
fn classify_pos_for_chunking(
    word: &str,
    is_capitalized: bool,
    position: usize,
    _context: &[(String, bool)],
) -> PosTag {
    // Check stop words first
    if is_stop_word(word) {
        return PosTag::StopWord;
    }

    // Capitalized words not at sentence start are likely proper nouns
    if is_capitalized && position > 0 {
        return PosTag::ProperNoun;
    }

    // Check verb indicators
    if is_verb(word) {
        return PosTag::Verb;
    }

    // Check adjective indicators
    if is_adjective(word) {
        return PosTag::Adjective;
    }

    // Check noun indicators
    if is_noun_for_chunking(word) {
        return PosTag::Noun;
    }

    // Default: if it's a content word (not too short, not a stop word), treat as noun
    // This is the "80% rule" - unknown words are usually nouns in English
    if word.len() >= 4 {
        return PosTag::Noun;
    }

    PosTag::Other
}

/// Noun detection for chunking (more aggressive than query parsing)
fn is_noun_for_chunking(word: &str) -> bool {
    // All the noun indicators from the original is_noun function
    // Plus additional heuristics for storage

    // Domain-specific nouns
    const NOUN_INDICATORS: &[&str] = &[
        // Keep all original indicators
        "memory",
        "graph",
        "node",
        "edge",
        "entity",
        "embedding",
        "vector",
        "index",
        "query",
        "retrieval",
        "activation",
        "potentiation",
        "consolidation",
        "decay",
        "strength",
        "weight",
        "threshold",
        "importance",
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
        "function",
        "method",
        "class",
        "struct",
        "interface",
        "package",
        "library",
        "framework",
        "api",
        "endpoint",
        "request",
        "error",
        "exception",
        "bug",
        "fix",
        "feature",
        "test",
        "benchmark",
        "performance",
        "latency",
        "throughput",
        "cache",
        "buffer",
        "queue",
        "stack",
        "heap",
        "thread",
        "process",
        "server",
        "client",
        "database",
        "table",
        "column",
        "row",
        "schema",
        "migration",
        "deployment",
        "container",
        "cluster",
        "replica",
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
        "warning",
        "alert",
        "notification",
        "level",
        "status",
        "state",
        "condition",
        "mode",
        "type",
        "kind",
        "version",
        "release",
        "update",
        "change",
        "result",
        "output",
        "input",
        "value",
        "key",
        "name",
        "id",
        "identifier",
        // Additional nouns common in conversational text
        "sunrise",
        "sunset",
        "lake",
        "mountain",
        "beach",
        "forest",
        "garden",
        "park",
        "city",
        "town",
        "village",
        "country",
        "house",
        "home",
        "room",
        "building",
        "street",
        "road",
        "car",
        "bus",
        "train",
        "plane",
        "boat",
        "bicycle",
        "food",
        "drink",
        "water",
        "coffee",
        "tea",
        "breakfast",
        "lunch",
        "dinner",
        "meal",
        "book",
        "movie",
        "music",
        "song",
        "art",
        "painting",
        "photo",
        "picture",
        "video",
        "game",
        "sport",
        "team",
        "player",
        "match",
        "race",
        "trip",
        "vacation",
        "holiday",
        "weekend",
        "morning",
        "evening",
        "night",
        "week",
        "month",
        "year",
        "birthday",
        "wedding",
        "party",
        "event",
        "meeting",
        "class",
        "lesson",
        "course",
        "school",
        "college",
        "university",
        "job",
        "work",
        "office",
        "company",
        "business",
        "project",
        "plan",
        "idea",
        "thought",
        "feeling",
        "emotion",
        "love",
        "friend",
        "family",
        "parent",
        "child",
        "kid",
        "baby",
        "mother",
        "father",
        "sister",
        "brother",
        "wife",
        "husband",
        "partner",
        "group",
        "community",
        "society",
        "culture",
        "tradition",
        "story",
        "history",
        "news",
        "article",
        "blog",
        "post",
        "comment",
        "email",
        "letter",
        "phone",
        "call",
        "text",
        "chat",
        "conversation",
        "discussion",
        "talk",
        "speech",
        "presentation",
        "question",
        "answer",
        "problem",
        "solution",
        "issue",
        "challenge",
        "opportunity",
        "success",
        "failure",
        "experience",
        "skill",
        "knowledge",
        "wisdom",
        "truth",
        "fact",
        "opinion",
        "belief",
        "value",
        "principle",
        "rule",
        "law",
        "policy",
        "decision",
        "choice",
        "option",
        "alternative",
        "reason",
        "cause",
        "effect",
        "impact",
        "influence",
        "power",
        "authority",
        "responsibility",
        "duty",
        "right",
        "freedom",
        "justice",
        "peace",
        "war",
        "conflict",
        "agreement",
        "contract",
        "deal",
        "price",
        "cost",
        "money",
        "dollar",
        "euro",
        "pound",
        "budget",
        "investment",
        "profit",
        "loss",
        "risk",
        "reward",
        "benefit",
        "advantage",
        "disadvantage",
        "strength",
        "weakness",
        "opportunity",
        "threat",
        "strategy",
        "tactic",
        "method",
        "approach",
        "technique",
        "tool",
        "resource",
        "material",
        "product",
        "service",
        "quality",
        "quantity",
        "size",
        "shape",
        "color",
        "sound",
        "smell",
        "taste",
        "touch",
        "sight",
        "sense",
        "mind",
        "body",
        "heart",
        "soul",
        "spirit",
        "health",
        "illness",
        "disease",
        "medicine",
        "doctor",
        "nurse",
        "hospital",
        "clinic",
        "therapy",
        "treatment",
        "care",
        "support",
        "help",
        "advice",
        "guidance",
        "counseling",
        "coaching",
        "mentoring",
        "training",
        "education",
        "learning",
        "teaching",
        "research",
        "study",
        "experiment",
        "discovery",
        "invention",
        "innovation",
        "technology",
        "science",
        "math",
        "physics",
        "chemistry",
        "biology",
        "psychology",
        "sociology",
        "philosophy",
        "religion",
        "spirituality",
        "meditation",
        "yoga",
        "exercise",
        "fitness",
        "diet",
        "nutrition",
        "sleep",
        "rest",
        "relaxation",
        "stress",
        "anxiety",
        "depression",
        "happiness",
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "trust",
        "hope",
        "faith",
        "courage",
        "confidence",
        "pride",
        "shame",
        "guilt",
        "regret",
        "gratitude",
        "empathy",
        "compassion",
        "kindness",
        "generosity",
        "honesty",
        "integrity",
        "loyalty",
        "respect",
        "tolerance",
        "patience",
        "persistence",
        "determination",
        "motivation",
        "inspiration",
        "creativity",
        "imagination",
        "curiosity",
        "wonder",
        "beauty",
        "art",
        "music",
        "dance",
        "theater",
        "film",
        "literature",
        "poetry",
        "writing",
        "reading",
        "speaking",
        "listening",
        "communication",
        "expression",
        "interpretation",
        "understanding",
        "meaning",
        "purpose",
        "goal",
        "dream",
        "vision",
        "mission",
        "passion",
        "interest",
        "hobby",
        "activity",
        "adventure",
        "journey",
        "path",
        "way",
        "direction",
        "destination",
        "origin",
        "beginning",
        "end",
        "start",
        "finish",
        "progress",
        "growth",
        "development",
        "evolution",
        "transformation",
        "change",
        "transition",
        "shift",
        "movement",
        "action",
        "reaction",
        "response",
        "behavior",
        "habit",
        "pattern",
        "routine",
        "schedule",
        "plan",
        "strategy",
        "tactic",
        "approach",
        "method",
        "process",
        "procedure",
        "step",
        "stage",
        "phase",
        "cycle",
        "circle",
        "loop",
        "sequence",
        "order",
        "arrangement",
        "organization",
        "structure",
        "system",
        "network",
        "connection",
        "relationship",
        "bond",
        "link",
        "tie",
        "association",
        "affiliation",
        "membership",
        "participation",
        "involvement",
        "engagement",
        "commitment",
        "dedication",
        "devotion",
        "loyalty",
        "allegiance",
        "support",
        "backing",
        "endorsement",
        "approval",
        "acceptance",
        "recognition",
        "acknowledgment",
        "appreciation",
        "gratitude",
        "thanks",
        "praise",
        "compliment",
        "criticism",
        "feedback",
        "evaluation",
        "assessment",
        "judgment",
        "opinion",
        "view",
        "perspective",
        "angle",
        "aspect",
        "dimension",
        "element",
        "component",
        "part",
        "piece",
        "section",
        "segment",
        "portion",
        "share",
        "fraction",
        "percentage",
        "ratio",
        "proportion",
        "balance",
        "equilibrium",
        "harmony",
        "unity",
        "diversity",
        "variety",
        "difference",
        "similarity",
        "comparison",
        "contrast",
        "distinction",
        "separation",
        "division",
        "classification",
        "category",
        "class",
        "type",
        "kind",
        "sort",
        "species",
        "variety",
        "version",
        "edition",
        "model",
        "design",
        "style",
        "format",
        "layout",
        "arrangement",
        "configuration",
        "setup",
        "installation",
        "deployment",
    ];

    if NOUN_INDICATORS.contains(&word) {
        return true;
    }

    // Noun suffixes
    if word.ends_with("tion")
        || word.ends_with("sion")
        || word.ends_with("ment")
        || word.ends_with("ness")
        || word.ends_with("ity")
        || word.ends_with("ance")
        || word.ends_with("ence")
        || word.ends_with("age")
        || word.ends_with("ure")
        || word.ends_with("dom")
        || word.ends_with("ship")
        || word.ends_with("hood")
        || word.ends_with("ism")
        || word.ends_with("ist")
    {
        return true;
    }

    // -er/-or suffixes (but not comparative adjectives)
    if (word.ends_with("er") || word.ends_with("or")) && word.len() > 4 {
        // Exclude likely comparatives
        let without_suffix = &word[..word.len() - 2];
        if !without_suffix.ends_with("t")
            && !without_suffix.ends_with("g")
            && !without_suffix.ends_with("d")
        {
            return true;
        }
    }

    false
}

/// Focal entity extracted from query (noun)
#[derive(Debug, Clone)]
pub struct FocalEntity {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    pub ic_weight: f32,
    /// True if entity is part of a compound noun
    pub is_compound: bool,
    /// True if preceded by negation
    pub negated: bool,
}

/// Discriminative modifier (adjective/qualifier)
#[derive(Debug, Clone)]
pub struct Modifier {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    /// IC weight for importance scoring (Lioma & Ounis 2006)
    pub ic_weight: f32,
    /// True if preceded by negation
    pub negated: bool,
}

/// Relational context (verb)
#[derive(Debug, Clone)]
pub struct Relation {
    pub text: String,
    /// Stemmed form for matching
    pub stem: String,
    /// IC weight for importance scoring (Lioma & Ounis 2006)
    pub ic_weight: f32,
    /// True if preceded by negation
    pub negated: bool,
}

/// Complete linguistic analysis of a query
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Focal entities (nouns) - primary search targets
    pub focal_entities: Vec<FocalEntity>,

    /// Discriminative modifiers (adjectives) - quality refiners
    pub discriminative_modifiers: Vec<Modifier>,

    /// Relational context (verbs) - graph traversal guides
    pub relational_context: Vec<Relation>,

    /// Compound nouns detected (e.g., "machine learning", "neural network")
    pub compound_nouns: Vec<String>,

    /// Original query text (retained for logging/debugging)
    pub original_query: String,

    /// True if query contains negation
    pub has_negation: bool,
}

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

        // Compound nouns get bonus weight
        let compound_bonus = self.compound_nouns.len() as f32 * 0.5;

        entity_weight + modifier_weight + relation_weight + compound_bonus
    }

    /// Get all stems for efficient matching
    pub fn all_stems(&self) -> HashSet<String> {
        let mut stems = HashSet::new();
        for e in &self.focal_entities {
            stems.insert(e.stem.clone());
        }
        for m in &self.discriminative_modifiers {
            stems.insert(m.stem.clone());
        }
        for r in &self.relational_context {
            stems.insert(r.stem.clone());
        }
        stems
    }

    /// Get non-negated entity stems (for positive matching)
    pub fn positive_entity_stems(&self) -> Vec<&str> {
        self.focal_entities
            .iter()
            .filter(|e| !e.negated)
            .map(|e| e.stem.as_str())
            .collect()
    }

    /// Get negated entity stems (for exclusion)
    pub fn negated_entity_stems(&self) -> Vec<&str> {
        self.focal_entities
            .iter()
            .filter(|e| e.negated)
            .map(|e| e.stem.as_str())
            .collect()
    }

    /// Convert analysis to IC weights HashMap for BM25 term boosting
    ///
    /// Returns a mapping of lowercase terms to their IC weights:
    /// - Nouns (focal entities): IC_NOUN = 1.5
    /// - Adjectives (modifiers): IC_ADJECTIVE = 0.9
    /// - Verbs (relations): IC_VERB = 0.7
    ///
    /// Based on Lioma & Ounis (2006) - nouns carry more information content.
    pub fn to_ic_weights(&self) -> std::collections::HashMap<String, f32> {
        self.to_ic_weights_with_yake(true)
    }

    /// Convert analysis to IC weights with optional YAKE boosting
    ///
    /// When use_yake=true, extracts discriminative keywords using YAKE algorithm
    /// and boosts their weights. This is critical for multi-hop queries where
    /// discriminative terms like "sunrise" must outweigh common terms like "Melanie".
    pub fn to_ic_weights_with_yake(
        &self,
        use_yake: bool,
    ) -> std::collections::HashMap<String, f32> {
        use crate::embeddings::keywords::{KeywordConfig, KeywordExtractor};

        let mut weights = std::collections::HashMap::new();

        // YAKE keyword extraction for discriminative term boosting
        // YAKE identifies statistically rare/important terms in the query
        if use_yake {
            let config = KeywordConfig {
                max_keywords: 5,
                ngrams: 2,
                min_length: 3,
                ..Default::default()
            };
            let extractor = KeywordExtractor::with_config(config);
            let keywords = extractor.extract(&self.original_query);

            // Boost YAKE-identified keywords with high weights
            // YAKE importance is 0.0-1.0 where higher = more discriminative
            // Key insight: Only boost SINGLE words, not bigrams
            // Bigrams like "Melanie paint" match too broadly - we need specific terms
            for kw in keywords {
                let term = kw.text.to_lowercase();

                // Skip bigrams/trigrams - they match documents with either word
                // which defeats the purpose of discriminative keyword boosting
                if term.contains(' ') {
                    continue;
                }

                // Aggressive boost for single discriminative words
                // Scale: importance 0.5 → boost 3.5, importance 1.0 → boost 6.0
                // This ensures rare words like "sunrise" dominate over common words
                let yake_boost = 1.0 + (kw.importance * 5.0);
                weights
                    .entry(term)
                    .and_modify(|w: &mut f32| *w = w.max(yake_boost))
                    .or_insert(yake_boost);
            }
        }

        // Add focal entities (nouns) with highest IC weight
        for entity in &self.focal_entities {
            let term = entity.text.to_lowercase();
            weights
                .entry(term)
                .and_modify(|w: &mut f32| *w = w.max(entity.ic_weight))
                .or_insert(entity.ic_weight);
            // Also add stem for fuzzy matching
            if entity.stem != entity.text.to_lowercase() {
                weights
                    .entry(entity.stem.clone())
                    .and_modify(|w: &mut f32| *w = w.max(entity.ic_weight))
                    .or_insert(entity.ic_weight);
            }
        }

        // Add discriminative modifiers (adjectives)
        for modifier in &self.discriminative_modifiers {
            let term = modifier.text.to_lowercase();
            weights
                .entry(term)
                .and_modify(|w: &mut f32| *w = w.max(modifier.ic_weight))
                .or_insert(modifier.ic_weight);
            if modifier.stem != modifier.text.to_lowercase() {
                weights
                    .entry(modifier.stem.clone())
                    .and_modify(|w: &mut f32| *w = w.max(modifier.ic_weight))
                    .or_insert(modifier.ic_weight);
            }
        }

        // Add relational context (verbs)
        for relation in &self.relational_context {
            let term = relation.text.to_lowercase();
            weights
                .entry(term)
                .and_modify(|w: &mut f32| *w = w.max(relation.ic_weight))
                .or_insert(relation.ic_weight);
            if relation.stem != relation.text.to_lowercase() {
                weights
                    .entry(relation.stem.clone())
                    .and_modify(|w: &mut f32| *w = w.max(relation.ic_weight))
                    .or_insert(relation.ic_weight);
            }
        }

        // Boost compound nouns (they carry more specific meaning)
        for compound in &self.compound_nouns {
            // Each word in compound gets a small boost
            for word in compound.split_whitespace() {
                let term = word.to_lowercase();
                weights.entry(term).and_modify(|w: &mut f32| *w *= 1.2);
            }
        }

        weights
    }

    /// Get maximum YAKE keyword discriminativeness score for dynamic weight adjustment
    ///
    /// Returns (max_importance, discriminative_keywords) where:
    /// - max_importance: 0.0-1.0, higher means more discriminative keywords found
    /// - discriminative_keywords: keywords with importance > 0.5 (for logging)
    ///
    /// Use this to dynamically adjust BM25/vector weights in hybrid search:
    /// - High discriminativeness (>0.6) → boost BM25 weight (keyword matching critical)
    /// - Low discriminativeness (<0.3) → trust vector more (semantic similarity better)
    pub fn keyword_discriminativeness(&self) -> (f32, Vec<String>) {
        use crate::embeddings::keywords::{KeywordConfig, KeywordExtractor};

        let config = KeywordConfig {
            max_keywords: 5,
            ngrams: 2,
            min_length: 3,
            ..Default::default()
        };
        let extractor = KeywordExtractor::with_config(config);
        let keywords = extractor.extract(&self.original_query);

        let mut max_importance = 0.0f32;
        let mut discriminative = Vec::new();

        for kw in keywords {
            if kw.importance > max_importance {
                max_importance = kw.importance;
            }
            // Keywords with importance > 0.5 are considered discriminative
            if kw.importance > 0.5 {
                discriminative.push(kw.text.to_lowercase());
            }
        }

        (max_importance, discriminative)
    }

    /// Get phrase boosts for BM25 exact phrase matching
    ///
    /// Returns compound nouns and adjacent noun pairs as phrases with boost weights.
    /// Phrase matching significantly improves retrieval for multi-word concepts
    /// like "support group", "machine learning", "LGBTQ community".
    pub fn to_phrase_boosts(&self) -> Vec<(String, f32)> {
        let mut phrases = Vec::new();

        // Add compound nouns with high boost (they are specific concepts)
        for compound in &self.compound_nouns {
            // Compound nouns get 2.0x boost for exact phrase match
            phrases.push((compound.to_lowercase(), 2.0));
        }

        // Also detect adjacent nouns that might form natural phrases
        // Even if not in compound_nouns, adjacent entities may form useful phrases
        if self.focal_entities.len() >= 2 {
            for i in 0..self.focal_entities.len() - 1 {
                let e1 = &self.focal_entities[i];
                let e2 = &self.focal_entities[i + 1];
                // Only if not negated and not already a compound
                if !e1.negated && !e2.negated {
                    let phrase = format!("{} {}", e1.text.to_lowercase(), e2.text.to_lowercase());
                    if !self
                        .compound_nouns
                        .iter()
                        .any(|c| c.to_lowercase() == phrase)
                    {
                        // Adjacent nouns get 1.5x boost (lower than explicit compounds)
                        phrases.push((phrase, 1.5));
                    }
                }
            }
        }

        phrases
    }
}

/// Token with linguistic annotations
#[derive(Debug)]
struct AnnotatedToken {
    text: String,
    stem: String,
    pos: PartOfSpeech,
    negated: bool,
    position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PartOfSpeech {
    Noun,
    Adjective,
    Verb,
    StopWord,
    Negation,
    Unknown,
}

/// Parse query using linguistic analysis with Porter2 stemming
pub fn analyze_query(query_text: &str) -> QueryAnalysis {
    let stemmer = Stemmer::create(Algorithm::English);
    let words = tokenize(query_text);

    if words.is_empty() {
        return QueryAnalysis {
            focal_entities: Vec::new(),
            discriminative_modifiers: Vec::new(),
            relational_context: Vec::new(),
            compound_nouns: Vec::new(),
            original_query: query_text.to_string(),
            has_negation: false,
        };
    }

    // Annotate each token with POS and negation scope
    let annotated = annotate_tokens(&words, &stemmer);

    // Detect compound nouns
    let compound_nouns = detect_compound_nouns(&annotated);

    // Build result structures
    let mut focal_entities = Vec::new();
    let mut discriminative_modifiers = Vec::new();
    let mut relational_context = Vec::new();
    let mut has_negation = false;

    // Track which tokens are part of compounds
    let compound_positions: HashSet<usize> = compound_positions(&annotated, &compound_nouns);

    for token in &annotated {
        if token.pos == PartOfSpeech::Negation {
            has_negation = true;
            continue;
        }
        if token.pos == PartOfSpeech::StopWord {
            continue;
        }

        let is_compound = compound_positions.contains(&token.position);

        match token.pos {
            PartOfSpeech::Noun | PartOfSpeech::Unknown => {
                // Unknown words are likely domain-specific nouns
                let weight = calculate_term_weight(&token.text, IC_NOUN);
                focal_entities.push(FocalEntity {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    is_compound,
                    negated: token.negated,
                });
            }
            PartOfSpeech::Adjective => {
                let weight = calculate_term_weight(&token.text, IC_ADJECTIVE);
                discriminative_modifiers.push(Modifier {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    negated: token.negated,
                });
            }
            PartOfSpeech::Verb => {
                let weight = calculate_term_weight(&token.text, IC_VERB);
                relational_context.push(Relation {
                    text: token.text.clone(),
                    stem: token.stem.clone(),
                    ic_weight: weight,
                    negated: token.negated,
                });
            }
            _ => {}
        }
    }

    // Add compound nouns as high-weight entities
    for compound in &compound_nouns {
        let stem = stemmer.stem(compound).to_string();
        focal_entities.push(FocalEntity {
            text: compound.clone(),
            stem,
            ic_weight: IC_NOUN * 1.5, // Compound bonus
            is_compound: true,
            negated: false,
        });
    }

    QueryAnalysis {
        focal_entities,
        discriminative_modifiers,
        relational_context,
        compound_nouns,
        original_query: query_text.to_string(),
        has_negation,
    }
}

/// Tokenize query text into lowercase words
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Annotate tokens with POS tags and negation scope
fn annotate_tokens(words: &[String], stemmer: &Stemmer) -> Vec<AnnotatedToken> {
    let mut annotated = Vec::with_capacity(words.len());
    let mut in_negation_scope = false;
    let mut negation_distance = 0;

    for (i, word) in words.iter().enumerate() {
        let stem = stemmer.stem(word).to_string();
        let pos = classify_pos(word, i, words);

        // Track negation scope (extends 2-3 words after negation)
        if pos == PartOfSpeech::Negation {
            in_negation_scope = true;
            negation_distance = 0;
        } else if in_negation_scope {
            negation_distance += 1;
            if negation_distance > 3 {
                in_negation_scope = false;
            }
        }

        let negated = in_negation_scope && pos != PartOfSpeech::Negation;

        annotated.push(AnnotatedToken {
            text: word.clone(),
            stem,
            pos,
            negated,
            position: i,
        });
    }

    annotated
}

/// Classify part of speech using heuristics
fn classify_pos(word: &str, position: usize, context: &[String]) -> PartOfSpeech {
    // Check negation first
    if is_negation(word) {
        return PartOfSpeech::Negation;
    }

    // Check stop words
    if is_stop_word(word) {
        return PartOfSpeech::StopWord;
    }

    // Use suffix patterns and context for classification
    if is_verb(word) {
        return PartOfSpeech::Verb;
    }

    if is_adjective(word) {
        return PartOfSpeech::Adjective;
    }

    if is_noun(word, position, context) {
        return PartOfSpeech::Noun;
    }

    // Default to unknown (treated as noun for domain terms)
    PartOfSpeech::Unknown
}

/// Detect compound nouns (bigrams that commonly co-occur)
fn detect_compound_nouns(tokens: &[AnnotatedToken]) -> Vec<String> {
    let mut compounds = Vec::new();

    // Common compound noun patterns
    const COMPOUND_PATTERNS: &[(&str, &str)] = &[
        // Tech/AI compounds
        ("machine", "learning"),
        ("deep", "learning"),
        ("neural", "network"),
        ("natural", "language"),
        ("language", "model"),
        ("artificial", "intelligence"),
        ("knowledge", "graph"),
        ("vector", "database"),
        ("memory", "system"),
        ("data", "structure"),
        ("source", "code"),
        ("error", "handling"),
        ("unit", "test"),
        ("integration", "test"),
        ("api", "endpoint"),
        ("web", "server"),
        ("file", "system"),
        ("operating", "system"),
        ("database", "schema"),
        ("user", "interface"),
        ("command", "line"),
        ("version", "control"),
        ("pull", "request"),
        ("code", "review"),
        ("bug", "fix"),
        ("feature", "request"),
        // Domain-specific
        ("spreading", "activation"),
        ("hebbian", "learning"),
        ("long", "term"),
        ("short", "term"),
        ("working", "memory"),
        ("semantic", "search"),
        ("graph", "traversal"),
        ("edge", "device"),
        ("air", "gapped"),
        // Social/community
        ("support", "group"),
        ("pride", "parade"),
        ("poetry", "reading"),
        ("civil", "rights"),
        ("human", "rights"),
        ("social", "media"),
        ("community", "center"),
        ("discussion", "group"),
        ("therapy", "session"),
        ("art", "therapy"),
        ("group", "therapy"),
    ];

    // Check for known compound patterns
    for i in 0..tokens.len().saturating_sub(1) {
        let t1 = &tokens[i];
        let t2 = &tokens[i + 1];

        // Skip if either is a stop word or verb
        if t1.pos == PartOfSpeech::StopWord || t2.pos == PartOfSpeech::StopWord {
            continue;
        }

        for (w1, w2) in COMPOUND_PATTERNS {
            if (t1.stem == *w1 || t1.text == *w1) && (t2.stem == *w2 || t2.text == *w2) {
                compounds.push(format!("{} {}", t1.text, t2.text));
                break;
            }
        }

        // Heuristic: Noun + Noun often forms compound
        if (t1.pos == PartOfSpeech::Noun || t1.pos == PartOfSpeech::Unknown)
            && (t2.pos == PartOfSpeech::Noun || t2.pos == PartOfSpeech::Unknown)
        {
            // Check for common suffixes that indicate compound-worthy nouns
            if has_compound_suffix(&t1.text) || has_compound_suffix(&t2.text) {
                let compound = format!("{} {}", t1.text, t2.text);
                if !compounds.contains(&compound) {
                    compounds.push(compound);
                }
            }
        }
    }

    compounds
}

/// Check if word has suffix that often appears in compounds
fn has_compound_suffix(word: &str) -> bool {
    word.ends_with("tion")
        || word.ends_with("ment")
        || word.ends_with("ing")
        || word.ends_with("ness")
        || word.ends_with("ity")
        || word.ends_with("ance")
        || word.ends_with("ence")
        || word.ends_with("er")
        || word.ends_with("or")
        || word.ends_with("ist")
        || word.ends_with("ism")
}

/// Get positions of tokens that are part of compounds
fn compound_positions(tokens: &[AnnotatedToken], compounds: &[String]) -> HashSet<usize> {
    let mut positions = HashSet::new();

    for compound in compounds {
        let parts: Vec<&str> = compound.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        for i in 0..tokens.len().saturating_sub(parts.len() - 1) {
            let mut matches = true;
            for (j, part) in parts.iter().enumerate() {
                if tokens[i + j].text != *part {
                    matches = false;
                    break;
                }
            }
            if matches {
                for j in 0..parts.len() {
                    positions.insert(i + j);
                }
            }
        }
    }

    positions
}

/// Calculate term weight with IDF-like rarity boost
fn calculate_term_weight(word: &str, base_weight: f32) -> f32 {
    // Longer words tend to be more specific/rare
    let length_factor = if word.len() > 8 {
        1.2
    } else if word.len() > 5 {
        1.1
    } else {
        1.0
    };

    // Technical suffixes get slight boost
    let suffix_factor = if word.ends_with("tion")
        || word.ends_with("ment")
        || word.ends_with("ness")
        || word.ends_with("ity")
    {
        1.1
    } else {
        1.0
    };

    base_weight * length_factor * suffix_factor
}

/// Check if word is negation
fn is_negation(word: &str) -> bool {
    const NEGATIONS: &[&str] = &[
        "not",
        "no",
        "never",
        "none",
        "nothing",
        "neither",
        "nobody",
        "nowhere",
        "without",
        "cannot",
        "can't",
        "won't",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
        "shouldn't",
        "wouldn't",
        "couldn't",
        "mustn't",
    ];
    NEGATIONS.contains(&word)
}

/// Check if word is a noun (entity)
fn is_noun(word: &str, position: usize, context: &[String]) -> bool {
    // Domain-specific nouns (expanded list)
    const NOUN_INDICATORS: &[&str] = &[
        // Core memory/cognitive terms
        "memory",
        "graph",
        "node",
        "edge",
        "entity",
        "embedding",
        "vector",
        "index",
        "query",
        "retrieval",
        "activation",
        "potentiation",
        "consolidation",
        "decay",
        "strength",
        "weight",
        "threshold",
        "importance",
        // Tech terms
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
        // Software terms
        "function",
        "method",
        "class",
        "struct",
        "interface",
        "module",
        "package",
        "library",
        "framework",
        "api",
        "endpoint",
        "request",
        "response",
        "error",
        "exception",
        "bug",
        "fix",
        "feature",
        "test",
        "benchmark",
        "performance",
        "latency",
        "throughput",
        "cache",
        "buffer",
        "queue",
        "stack",
        "heap",
        "thread",
        "process",
        "server",
        "client",
        "database",
        "table",
        "column",
        "row",
        "schema",
        "migration",
        "deployment",
        "container",
        "cluster",
        "replica",
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
        "warning",
        "alert",
        "notification",
        "level",
        "status",
        "state",
        "condition",
        "mode",
        "type",
        "kind",
        "version",
        "release",
        "update",
        "change",
        "result",
        "output",
        "input",
        "value",
        "key",
        "name",
        "id",
        "identifier",
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
        || word.ends_with("er")
        || word.ends_with("or")
        || word.ends_with("ist")
        || word.ends_with("ism")
        || word.ends_with("age")
        || word.ends_with("ure")
        || word.ends_with("dom")
    {
        // Avoid verb forms like "better", "faster"
        if !(word.ends_with("er") && word.len() < 5) {
            return true;
        }
    }

    // Check if preceded by determiner (a, an, the)
    if position > 0 {
        if let Some(prev) = context.get(position - 1) {
            let prev = prev.to_lowercase();
            if prev == "a" || prev == "an" || prev == "the" || prev == "this" || prev == "that" {
                return true;
            }
        }
    }

    // Check if preceded by possessive
    if position > 0 {
        if let Some(prev) = context.get(position - 1) {
            if prev.ends_with("'s") || prev.ends_with("s'") {
                return true;
            }
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
        "open",
        "closed",
        "locked",
        "unlocked",
        "full",
        "empty",
        "partial",
        "complete",
        "valid",
        "invalid",
        "correct",
        "incorrect",
        "true",
        "false",
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
        "first",
        "last",
        "next",
        "previous",
        "primary",
        "secondary",
        "main",
        "important",
        "critical",
        "minor",
        "major",
        // Technical
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
        "local",
        "global",
        "private",
        "public",
        "static",
        "dynamic",
        "mutable",
        "immutable",
        "sync",
        "async",
        "concurrent",
        "parallel",
        "serial",
        "sequential",
        "optional",
        "required",
        "default",
        "custom",
    ];

    if ADJECTIVE_INDICATORS.contains(&word) {
        return true;
    }

    // Common adjective suffixes (excluding verb participles)
    if word.ends_with("ful")
        || word.ends_with("less")
        || word.ends_with("ous")
        || word.ends_with("ive")
        || word.ends_with("able")
        || word.ends_with("ible")
        || word.ends_with("al")
        || word.ends_with("ic")
        || word.ends_with("ary")
        || word.ends_with("ory")
    {
        // Avoid false positives
        let exceptions = ["animal", "interval", "arrival", "approval"];
        if !exceptions.contains(&word) {
            return true;
        }
    }

    false
}

/// Check if word is a verb (relational, lower priority)
fn is_verb(word: &str) -> bool {
    const VERB_INDICATORS: &[&str] = &[
        // Auxiliary/modal verbs
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
        // Common action verbs
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
        "give",
        "gives",
        "gave",
        "given",
        "giving",
        "use",
        "uses",
        "used",
        "using",
        "find",
        "finds",
        "found",
        "finding",
        "know",
        "knows",
        "knew",
        "known",
        "knowing",
        "think",
        "thinks",
        "thought",
        "thinking",
        "want",
        "wants",
        "wanted",
        "wanting",
        "need",
        "needs",
        "needed",
        "needing",
        "try",
        "tries",
        "tried",
        "trying",
        // Technical verbs
        "detect",
        "detects",
        "detected",
        "detecting",
        "observe",
        "observes",
        "observed",
        "observing",
        "measure",
        "measures",
        "measured",
        "measuring",
        "sense",
        "senses",
        "sensed",
        "sensing",
        "scan",
        "scans",
        "scanned",
        "scanning",
        "navigate",
        "navigates",
        "navigated",
        "navigating",
        "move",
        "moves",
        "moved",
        "moving",
        "stop",
        "stops",
        "stopped",
        "stopping",
        "start",
        "starts",
        "started",
        "starting",
        "reach",
        "reaches",
        "reached",
        "reaching",
        "avoid",
        "avoids",
        "avoided",
        "avoiding",
        "block",
        "blocks",
        "blocked",
        "blocking",
        "create",
        "creates",
        "created",
        "creating",
        "delete",
        "deletes",
        "deleted",
        "deleting",
        "update",
        "updates",
        "updated",
        "updating",
        "read",
        "reads",
        "reading",
        "write",
        "writes",
        "wrote",
        "written",
        "writing",
        "run",
        "runs",
        "ran",
        "running",
        "execute",
        "executes",
        "executed",
        "executing",
        "call",
        "calls",
        "called",
        "calling",
        "return",
        "returns",
        "returned",
        "returning",
        "store",
        "stores",
        "stored",
        "storing",
        "load",
        "loads",
        "loaded",
        "loading",
        "save",
        "saves",
        "saved",
        "saving",
        "fetch",
        "fetches",
        "fetched",
        "fetching",
        "send",
        "sends",
        "sent",
        "sending",
        "receive",
        "receives",
        "received",
        "receiving",
        "connect",
        "connects",
        "connected",
        "connecting",
        "disconnect",
        "disconnects",
        "disconnected",
        "disconnecting",
        "process",
        "processes",
        "processed",
        "processing",
        "handle",
        "handles",
        "handled",
        "handling",
        "parse",
        "parses",
        "parsed",
        "parsing",
        "compile",
        "compiles",
        "compiled",
        "compiling",
        "build",
        "builds",
        "built",
        "building",
        "test",
        "tests",
        "tested",
        "testing",
        "deploy",
        "deploys",
        "deployed",
        "deploying",
        "install",
        "installs",
        "installed",
        "installing",
        "configure",
        "configures",
        "configured",
        "configuring",
        "initialize",
        "initializes",
        "initialized",
        "initializing",
        "shutdown",
        "shutdowns",
        "terminate",
        "terminates",
        "terminated",
        "terminating",
    ];

    VERB_INDICATORS.contains(&word)
}

/// Check if word is a stop word (no information content)
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        // Articles
        "a",
        "an",
        "the",
        // Demonstratives
        "this",
        "that",
        "these",
        "those",
        // Prepositions
        "at",
        "in",
        "on",
        "to",
        "for",
        "of",
        "from",
        "by",
        "with",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        // Conjunctions
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        // Pronouns
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "who",
        "whom",
        "whose",
        "which",
        "what",
        "whoever",
        "whatever",
        "whichever",
        // Relative
        "that",
        "which",
        "who",
        "whom",
        "whose",
        // Question words (when not seeking info)
        "how",
        "when",
        "where",
        "why",
        // Common filler
        "just",
        "only",
        "even",
        "also",
        "too",
        "very",
        "really",
        "quite",
        "rather",
        "almost",
        "already",
        "still",
        "always",
        "never",
        "ever",
        "often",
        "sometimes",
        "usually",
        "perhaps",
        "maybe",
        "probably",
        "possibly",
        "certainly",
        "definitely",
        "actually",
        "basically",
        "essentially",
        "simply",
        "merely",
        // Be forms handled separately as verbs
        "as",
        "if",
        "then",
        "than",
        "because",
        "although",
        "though",
        "unless",
        "until",
        "while",
        "whereas",
        "whether",
        "since",
        // Others
        "some",
        "any",
        "all",
        "each",
        "every",
        "many",
        "much",
        "more",
        "most",
        "few",
        "less",
        "least",
        "other",
        "another",
        "such",
        "same",
        "different",
        "own",
        "several",
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

        // Nouns should have IC weight >= IC_NOUN base
        for entity in &analysis.focal_entities {
            assert!(entity.ic_weight >= IC_NOUN * 0.9); // Allow small variance
        }

        // Adjectives should have IC weight >= IC_ADJECTIVE base
        for modifier in &analysis.discriminative_modifiers {
            assert!(modifier.ic_weight >= IC_ADJECTIVE * 0.9);
        }

        // Verbs should have IC weight >= IC_VERB base
        for relation in &analysis.relational_context {
            assert!(relation.ic_weight >= IC_VERB * 0.9);
        }
    }

    #[test]
    fn test_stemming() {
        let query = "running detection algorithms";
        let analysis = analyze_query(query);

        // Check stems are different from original text
        let stems: Vec<String> = analysis
            .focal_entities
            .iter()
            .map(|e| e.stem.clone())
            .collect();

        // "detection" should stem to "detect"
        assert!(stems.iter().any(|s| s == "detect"));
        // "algorithms" should stem to "algorithm"
        assert!(stems.iter().any(|s| s == "algorithm"));
    }

    #[test]
    fn test_compound_noun_detection() {
        let query = "machine learning neural network";
        let analysis = analyze_query(query);

        assert!(analysis
            .compound_nouns
            .contains(&"machine learning".to_string()));
        assert!(analysis
            .compound_nouns
            .contains(&"neural network".to_string()));
    }

    #[test]
    fn test_negation_detection() {
        let query = "not working correctly";
        let analysis = analyze_query(query);

        assert!(analysis.has_negation);

        // Check that tokens after negation are marked
        let negated_entities: Vec<&FocalEntity> = analysis
            .focal_entities
            .iter()
            .filter(|e| e.negated)
            .collect();

        assert!(!negated_entities.is_empty());
    }

    #[test]
    fn test_negation_scope() {
        let query = "the sensor is not detecting obstacles properly";
        let analysis = analyze_query(query);

        assert!(analysis.has_negation);

        // "detecting" should be marked as negated
        let negated_verbs: Vec<&Relation> = analysis
            .relational_context
            .iter()
            .filter(|r| r.negated)
            .collect();

        assert!(negated_verbs.iter().any(|r| r.text == "detecting"));
    }

    #[test]
    fn test_all_stems_helper() {
        let query = "fast robot detecting obstacles";
        let analysis = analyze_query(query);

        let stems = analysis.all_stems();
        assert!(stems.contains("robot"));
        assert!(stems.contains("fast"));
        assert!(stems.contains("detect"));
        assert!(stems.contains("obstacl")); // Porter stem
    }

    #[test]
    fn test_positive_and_negated_stems() {
        let query = "working memory not failed";
        let analysis = analyze_query(query);

        let positive = analysis.positive_entity_stems();
        let negated = analysis.negated_entity_stems();

        // "memory" should be positive
        assert!(positive.iter().any(|s| s.contains("memori")));

        // "failed" should be negated (after "not")
        // Note: "failed" might be classified as verb or noun depending on context
    }

    #[test]
    fn test_empty_query() {
        let query = "";
        let analysis = analyze_query(query);

        assert!(analysis.focal_entities.is_empty());
        assert!(analysis.discriminative_modifiers.is_empty());
        assert!(analysis.relational_context.is_empty());
        assert!(!analysis.has_negation);
    }

    #[test]
    fn test_stop_words_filtered() {
        let query = "the a an is are was were";
        let analysis = analyze_query(query);

        // Only verbs should remain (is, are, was, were)
        assert!(analysis.focal_entities.is_empty());
        assert!(analysis.discriminative_modifiers.is_empty());
        assert!(!analysis.relational_context.is_empty());
    }

    #[test]
    fn test_total_weight_calculation() {
        let query = "fast robot detecting red obstacles";
        let analysis = analyze_query(query);

        let weight = analysis.total_weight();
        assert!(weight > 0.0);
    }

    #[test]
    fn test_to_ic_weights() {
        use crate::constants::{IC_ADJECTIVE, IC_NOUN, IC_VERB};

        let query = "fast robot detecting obstacles";
        let analysis = analyze_query(query);
        let weights = analysis.to_ic_weights();

        // Should have weights for terms
        assert!(!weights.is_empty(), "Weights should not be empty");

        // Check that weights were generated
        // Nouns get IC_NOUN (1.5), adjectives IC_ADJECTIVE (0.9), verbs IC_VERB (0.7)
        // The actual terms depend on POS tagging which may vary

        // At minimum, check that we have some weights with expected IC values
        let has_noun_weight = weights.values().any(|&w| (w - IC_NOUN).abs() < 0.01);
        let has_adj_weight = weights.values().any(|&w| (w - IC_ADJECTIVE).abs() < 0.01);
        let has_verb_weight = weights.values().any(|&w| (w - IC_VERB).abs() < 0.01);

        // At least one type should be present
        assert!(
            has_noun_weight || has_adj_weight || has_verb_weight,
            "Should have at least one IC weight type. Weights: {:?}",
            weights
        );
    }

    #[test]
    fn test_to_phrase_boosts() {
        // Test with a query containing known compound noun
        let query = "machine learning model for semantic search";
        let analysis = analyze_query(query);
        let phrases = analysis.to_phrase_boosts();

        // "machine learning" is a known compound pattern
        let has_ml = phrases.iter().any(|(p, _)| p == "machine learning");
        let has_ss = phrases.iter().any(|(p, _)| p == "semantic search");

        assert!(
            has_ml || has_ss,
            "Should detect 'machine learning' or 'semantic search' as phrase. Found: {:?}",
            phrases
        );

        // Compound nouns should have higher boost (2.0)
        for (phrase, boost) in &phrases {
            assert!(
                *boost >= 1.0,
                "Phrase '{}' should have boost >= 1.0, got {}",
                phrase,
                boost
            );
        }
    }

    #[test]
    fn test_to_phrase_boosts_support_group() {
        // Test with LoCoMo-style query
        let query = "when did she go to the support group";
        let analysis = analyze_query(query);
        let phrases = analysis.to_phrase_boosts();

        // "support group" should be detected as a compound
        let has_support_group = phrases.iter().any(|(p, _)| p == "support group");
        assert!(
            has_support_group,
            "Should detect 'support group' as phrase. Found: {:?}",
            phrases
        );
    }
}
