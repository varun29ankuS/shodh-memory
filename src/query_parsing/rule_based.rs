//! Rule-Based Query Parser
//!
//! Wraps the existing YAKE/regex-based query_parser module into the QueryParser trait.
//! This is the default, battle-tested implementation.

use super::parser_trait::*;
use crate::memory::query_parser as legacy;
use chrono::{DateTime, Datelike, NaiveDate, Utc};

/// Rule-based query parser using YAKE, regex patterns, and heuristics
pub struct RuleBasedParser {
    _private: (),
}

impl RuleBasedParser {
    /// Create a new rule-based parser
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for RuleBasedParser {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryParser for RuleBasedParser {
    fn parse(&self, query: &str, context_date: Option<DateTime<Utc>>) -> ParsedQuery {
        // Use the existing analyze_query function
        let analysis = legacy::analyze_query(query);

        // Detect temporal intent
        let temporal_intent = legacy::detect_temporal_intent(query);
        let has_temporal = !matches!(temporal_intent, legacy::TemporalIntent::None);

        // Extract temporal references
        let temporal_refs = legacy::extract_temporal_refs(query);
        let relative_refs = extract_relative_refs(query, context_date);

        // Resolve relative dates if context provided
        let resolved_dates = if context_date.is_some() {
            relative_refs
                .iter()
                .filter_map(|r| r.resolved)
                .collect()
        } else {
            Vec::new()
        };

        // Extract absolute dates from temporal refs
        let absolute_dates: Vec<NaiveDate> = temporal_refs
            .refs
            .iter()
            .map(|r| r.date)
            .collect();

        // Convert focal entities
        let entities: Vec<Entity> = analysis
            .focal_entities
            .iter()
            .map(|e| Entity {
                text: e.text.clone(),
                stem: e.stem.clone(),
                entity_type: detect_entity_type(&e.text),
                ic_weight: e.ic_weight,
                negated: e.negated,
            })
            .collect();

        // Convert relational context to events
        let events: Vec<Event> = analysis
            .relational_context
            .iter()
            .map(|r| Event {
                text: r.text.clone(),
                stem: r.stem.clone(),
                ic_weight: r.ic_weight,
            })
            .collect();

        // Get modifiers
        let modifiers: Vec<String> = analysis
            .discriminative_modifiers
            .iter()
            .map(|m| m.text.clone())
            .collect();

        // Check for attribute query
        let (is_attribute_query, attribute) = match legacy::detect_attribute_query(query) {
            Some(aq) => (
                true,
                Some(AttributeQuery {
                    entity: aq.entity.clone(),
                    attribute: aq.attribute.clone(),
                    synonyms: aq.attribute_synonyms.clone(),
                }),
            ),
            None => (false, None),
        };

        // Get IC weights
        let ic_weights = analysis.to_ic_weights();

        ParsedQuery {
            original: query.to_string(),
            entities,
            events,
            modifiers,
            temporal: TemporalInfo {
                has_temporal_intent: has_temporal,
                intent: convert_temporal_intent(temporal_intent),
                relative_refs,
                resolved_dates,
                absolute_dates,
            },
            is_attribute_query,
            attribute,
            compounds: analysis.compound_nouns.clone(),
            ic_weights,
            confidence: 0.85, // Rule-based has consistent but not perfect accuracy
        }
    }

    fn name(&self) -> &'static str {
        "RuleBasedParser"
    }
}

/// Convert legacy TemporalIntent to new format
fn convert_temporal_intent(intent: legacy::TemporalIntent) -> TemporalIntent {
    match intent {
        legacy::TemporalIntent::WhenQuestion => TemporalIntent::WhenQuestion,
        legacy::TemporalIntent::SpecificTime => TemporalIntent::SpecificTime,
        legacy::TemporalIntent::Ordering => TemporalIntent::Ordering,
        legacy::TemporalIntent::Duration => TemporalIntent::Duration,
        legacy::TemporalIntent::None => TemporalIntent::None,
    }
}

/// Detect entity type from text (basic heuristics)
fn detect_entity_type(text: &str) -> EntityType {
    let text_lower = text.to_lowercase();

    // Check if it starts with capital (likely proper noun / person)
    if text.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        && !text.chars().all(|c| c.is_uppercase())
    {
        // Common person name patterns
        let first_word = text.split_whitespace().next().unwrap_or("");
        if is_likely_person_name(first_word) {
            return EntityType::Person;
        }
    }

    // Time-related words
    if ["morning", "evening", "afternoon", "night", "day", "week", "month", "year"]
        .iter()
        .any(|t| text_lower.contains(t))
    {
        return EntityType::Time;
    }

    // Event-related words
    if ["meeting", "party", "wedding", "concert", "race", "trip", "vacation"]
        .iter()
        .any(|e| text_lower.contains(e))
    {
        return EntityType::Event;
    }

    EntityType::Unknown
}

/// Check if a word is likely a person's name
fn is_likely_person_name(word: &str) -> bool {
    // Simple heuristic: capitalized, not a common noun, reasonable length
    if word.len() < 2 || word.len() > 20 {
        return false;
    }

    let first_char = word.chars().next().unwrap_or(' ');
    if !first_char.is_uppercase() {
        return false;
    }

    // Common non-person capitalized words
    let non_names = [
        "The", "This", "That", "What", "When", "Where", "Who", "How", "Why",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    ];

    !non_names.iter().any(|n| n.eq_ignore_ascii_case(word))
}

/// Extract relative time references from query
fn extract_relative_refs(query: &str, context_date: Option<DateTime<Utc>>) -> Vec<RelativeTimeRef> {
    let query_lower = query.to_lowercase();
    let mut refs = Vec::new();

    // Patterns for relative time references
    let patterns = [
        ("last year", TimeDirection::Past, TimeUnit::Year, 1),
        ("last month", TimeDirection::Past, TimeUnit::Month, 1),
        ("last week", TimeDirection::Past, TimeUnit::Week, 1),
        ("last saturday", TimeDirection::Past, TimeUnit::Day, -1), // Special handling
        ("last sunday", TimeDirection::Past, TimeUnit::Day, -1),
        ("last friday", TimeDirection::Past, TimeUnit::Day, -1),
        ("yesterday", TimeDirection::Past, TimeUnit::Day, 1),
        ("next year", TimeDirection::Future, TimeUnit::Year, 1),
        ("next month", TimeDirection::Future, TimeUnit::Month, 1),
        ("next week", TimeDirection::Future, TimeUnit::Week, 1),
        ("tomorrow", TimeDirection::Future, TimeUnit::Day, 1),
        ("two weeks ago", TimeDirection::Past, TimeUnit::Week, 2),
        ("three weeks ago", TimeDirection::Past, TimeUnit::Week, 3),
        ("two months ago", TimeDirection::Past, TimeUnit::Month, 2),
        ("a week ago", TimeDirection::Past, TimeUnit::Week, 1),
        ("a month ago", TimeDirection::Past, TimeUnit::Month, 1),
        ("a year ago", TimeDirection::Past, TimeUnit::Year, 1),
    ];

    for (pattern, direction, unit, offset) in patterns {
        if query_lower.contains(pattern) {
            let resolved = context_date.and_then(|ctx| {
                resolve_relative_date(ctx, direction, unit, offset, pattern)
            });

            refs.push(RelativeTimeRef {
                text: pattern.to_string(),
                resolved,
                direction,
                unit,
                offset,
            });
        }
    }

    refs
}

/// Resolve a relative date reference to an absolute date
fn resolve_relative_date(
    context: DateTime<Utc>,
    direction: TimeDirection,
    unit: TimeUnit,
    offset: i32,
    pattern: &str,
) -> Option<NaiveDate> {
    use chrono::Duration;

    let base_date = context.date_naive();

    // Handle "last <weekday>" specially
    if pattern.starts_with("last ") && pattern.len() > 5 {
        let weekday_str = &pattern[5..];
        if let Some(target_weekday) = parse_weekday(weekday_str) {
            // Find the most recent occurrence of this weekday before context date
            let current_weekday = base_date.weekday();
            let days_back = (current_weekday.num_days_from_monday() as i32
                - target_weekday.num_days_from_monday() as i32
                + 7) % 7;
            let days_back = if days_back == 0 { 7 } else { days_back };
            return Some(base_date - Duration::days(days_back as i64));
        }
    }

    let result = match (direction, unit) {
        (TimeDirection::Past, TimeUnit::Day) => base_date - Duration::days(offset as i64),
        (TimeDirection::Past, TimeUnit::Week) => base_date - Duration::weeks(offset as i64),
        (TimeDirection::Past, TimeUnit::Month) => {
            // Approximate month subtraction
            let months_back = offset as i64;
            let new_month = (base_date.month() as i64 - months_back - 1).rem_euclid(12) + 1;
            let year_offset = (base_date.month() as i64 - months_back - 1).div_euclid(12);
            NaiveDate::from_ymd_opt(
                base_date.year() + year_offset as i32,
                new_month as u32,
                base_date.day().min(28),
            )?
        }
        (TimeDirection::Past, TimeUnit::Year) => {
            NaiveDate::from_ymd_opt(base_date.year() - offset, base_date.month(), base_date.day())?
        }
        (TimeDirection::Future, TimeUnit::Day) => base_date + Duration::days(offset as i64),
        (TimeDirection::Future, TimeUnit::Week) => base_date + Duration::weeks(offset as i64),
        (TimeDirection::Future, TimeUnit::Month) => {
            let months_forward = offset as i64;
            let new_month = (base_date.month() as i64 + months_forward - 1).rem_euclid(12) + 1;
            let year_offset = (base_date.month() as i64 + months_forward - 1).div_euclid(12);
            NaiveDate::from_ymd_opt(
                base_date.year() + year_offset as i32,
                new_month as u32,
                base_date.day().min(28),
            )?
        }
        (TimeDirection::Future, TimeUnit::Year) => {
            NaiveDate::from_ymd_opt(base_date.year() + offset, base_date.month(), base_date.day())?
        }
        _ => return None,
    };

    Some(result)
}

/// Parse a weekday string
fn parse_weekday(s: &str) -> Option<chrono::Weekday> {
    use chrono::Weekday;
    match s.to_lowercase().as_str() {
        "monday" | "mon" => Some(Weekday::Mon),
        "tuesday" | "tue" | "tues" => Some(Weekday::Tue),
        "wednesday" | "wed" => Some(Weekday::Wed),
        "thursday" | "thu" | "thur" | "thurs" => Some(Weekday::Thu),
        "friday" | "fri" => Some(Weekday::Fri),
        "saturday" | "sat" => Some(Weekday::Sat),
        "sunday" | "sun" => Some(Weekday::Sun),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_parse_basic_query() {
        let parser = RuleBasedParser::new();
        let parsed = parser.parse("What is Caroline's relationship status?", None);

        assert!(parsed.is_attribute_query);
        assert!(!parsed.entities.is_empty());
    }

    #[test]
    fn test_parse_temporal_query() {
        let parser = RuleBasedParser::new();
        let parsed = parser.parse("When did Melanie paint a sunrise?", None);

        assert!(parsed.temporal.has_temporal_intent);
        assert_eq!(parsed.temporal.intent, TemporalIntent::WhenQuestion);
    }

    #[test]
    fn test_resolve_last_year() {
        let parser = RuleBasedParser::new();
        let context = chrono::Utc.with_ymd_and_hms(2023, 5, 8, 12, 0, 0).unwrap();
        let parsed = parser.parse("Melanie painted it last year", Some(context));

        assert!(!parsed.temporal.relative_refs.is_empty());
        let ref_ = &parsed.temporal.relative_refs[0];
        assert_eq!(ref_.text, "last year");
        assert_eq!(ref_.resolved, Some(NaiveDate::from_ymd_opt(2022, 5, 8).unwrap()));
    }
}
