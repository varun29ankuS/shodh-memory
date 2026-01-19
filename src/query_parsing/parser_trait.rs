//! Query Parser Trait Definition
//!
//! Defines the interface that all query parsers must implement.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of parsing a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedQuery {
    /// Original query text
    pub original: String,

    /// Extracted entities (people, places, things)
    pub entities: Vec<Entity>,

    /// Extracted events/actions (verbs)
    pub events: Vec<Event>,

    /// Modifiers (adjectives, descriptors)
    pub modifiers: Vec<String>,

    /// Temporal information extracted from the query
    pub temporal: TemporalInfo,

    /// Whether this is an attribute query (asking about a property)
    pub is_attribute_query: bool,

    /// The attribute being asked about (if is_attribute_query)
    pub attribute: Option<AttributeQuery>,

    /// Compound terms detected (e.g., "machine learning")
    pub compounds: Vec<String>,

    /// IC weights for BM25 boosting
    pub ic_weights: HashMap<String, f32>,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// An extracted entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Original text
    pub text: String,
    /// Stemmed form
    pub stem: String,
    /// Entity type if detected
    pub entity_type: EntityType,
    /// Information content weight
    pub ic_weight: f32,
    /// Whether this entity is negated
    pub negated: bool,
}

/// Entity type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Place,
    Thing,
    Event,
    Time,
    Unknown,
}

/// An extracted event/action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Original text (verb)
    pub text: String,
    /// Stemmed form
    pub stem: String,
    /// IC weight
    pub ic_weight: f32,
}

/// Temporal information extracted from query
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalInfo {
    /// Whether the query has temporal intent
    pub has_temporal_intent: bool,

    /// Type of temporal query
    pub intent: TemporalIntent,

    /// Relative time references found ("last year", "next month")
    pub relative_refs: Vec<RelativeTimeRef>,

    /// Resolved absolute dates (if context date provided)
    pub resolved_dates: Vec<NaiveDate>,

    /// Absolute dates mentioned directly ("May 7, 2023")
    pub absolute_dates: Vec<NaiveDate>,
}

/// Type of temporal intent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TemporalIntent {
    /// "When did X happen?"
    WhenQuestion,
    /// "What happened in [time period]?"
    SpecificTime,
    /// "Did X happen before/after Y?"
    Ordering,
    /// "How long did X take?"
    Duration,
    /// No temporal intent
    #[default]
    None,
}

/// A relative time reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeTimeRef {
    /// Original text ("last year", "next month")
    pub text: String,
    /// Resolved date (if context available)
    pub resolved: Option<NaiveDate>,
    /// Direction (past/future)
    pub direction: TimeDirection,
    /// Unit (day, week, month, year)
    pub unit: TimeUnit,
    /// Offset amount (1 for "last", 2 for "two weeks ago")
    pub offset: i32,
}

/// Direction of time reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeDirection {
    Past,
    Future,
    Current,
}

/// Time unit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeUnit {
    Day,
    Week,
    Month,
    Year,
    Unknown,
}

/// Attribute query details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeQuery {
    /// The entity being queried about
    pub entity: String,
    /// The attribute being asked (e.g., "relationship status")
    pub attribute: String,
    /// Synonyms for the attribute value
    pub synonyms: Vec<String>,
}

/// Query parser trait - implement this for different parsing strategies
pub trait QueryParser: Send + Sync {
    /// Parse a query into structured components
    ///
    /// # Arguments
    /// * `query` - The natural language query
    /// * `context_date` - Optional date for resolving relative time references
    ///
    /// # Returns
    /// Parsed query structure with entities, events, temporal info, etc.
    fn parse(&self, query: &str, context_date: Option<DateTime<Utc>>) -> ParsedQuery;

    /// Get the parser type name (for logging/debugging)
    fn name(&self) -> &'static str;

    /// Check if this parser is available/loaded
    fn is_available(&self) -> bool {
        true
    }
}

impl ParsedQuery {
    /// Create an empty parsed query
    pub fn empty(original: &str) -> Self {
        Self {
            original: original.to_string(),
            entities: Vec::new(),
            events: Vec::new(),
            modifiers: Vec::new(),
            temporal: TemporalInfo::default(),
            is_attribute_query: false,
            attribute: None,
            compounds: Vec::new(),
            ic_weights: HashMap::new(),
            confidence: 0.0,
        }
    }

    /// Get all entity texts
    pub fn entity_texts(&self) -> Vec<&str> {
        self.entities.iter().map(|e| e.text.as_str()).collect()
    }

    /// Get all event stems
    pub fn event_stems(&self) -> Vec<&str> {
        self.events.iter().map(|e| e.stem.as_str()).collect()
    }

    /// Check if query is asking about time
    pub fn is_temporal_query(&self) -> bool {
        self.temporal.has_temporal_intent
    }
}
