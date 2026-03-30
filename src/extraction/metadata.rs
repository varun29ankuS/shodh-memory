use regex::Regex;
use std::sync::LazyLock;

use super::types::{ExtractedEntity, ExtractionSource};

static ISSUE_ID_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Z]+-\d+$").unwrap());

pub fn extract_metadata(tags: &[String], issue_ids: &[String]) -> Vec<ExtractedEntity> {
    let mut entities = Vec::new();

    // Process Tags
    for tag in tags {
        let (entity_type, text) = if let Some(idx) = tag.find(':') {
            let namespace = &tag[..idx];
            let value = &tag[idx + 1..];
            
            // Map common namespaces, fallback to capitalized namespace
            let mapped_type = match namespace.to_lowercase().as_str() {
                "cve" => "Cve".to_string(),
                "operator" => "OperatorId".to_string(),
                "campaign" => "Campaign".to_string(),
                _ => {
                    let mut chars = namespace.chars();
                    match chars.next() {
                        None => "Tag".to_string(),
                        Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
                    }
                }
            };
            (mapped_type, value.to_string())
        } else {
            ("Tag".to_string(), tag.clone())
        };

        entities.push(ExtractedEntity {
            text,
            entity_type,
            confidence: 1.0,
            spans: vec![],
            source: ExtractionSource::Metadata,
        });
    }

    // Process Issue IDs
    for issue_id in issue_ids {
        if ISSUE_ID_REGEX.is_match(issue_id) {
            entities.push(ExtractedEntity {
                text: issue_id.clone(),
                entity_type: "IssueId".to_string(),
                confidence: 1.0,
                spans: vec![],
                source: ExtractionSource::Metadata,
            });
        }
    }

    entities
}
