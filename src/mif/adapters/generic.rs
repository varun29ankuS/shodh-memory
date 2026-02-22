//! Generic JSON adapter.
//!
//! Handles any JSON array of objects with at least a `content` field:
//! ```json
//! [
//!   {
//!     "content": "Some memory text",
//!     "timestamp": "2024-01-15T10:30:00Z",
//!     "tags": ["tag1", "tag2"],
//!     "type": "observation",
//!     "metadata": {"key": "value"}
//!   }
//! ]
//! ```
//! This is the most permissive adapter — it accepts anything that looks like
//! a JSON array of objects. It's tried last in auto-detection.

use std::collections::HashMap;

use anyhow::Result;
use chrono::Utc;

use super::MifAdapter;
use crate::mif::schema::*;

pub struct GenericJsonAdapter;

impl MifAdapter for GenericJsonAdapter {
    fn name(&self) -> &str {
        "Generic JSON"
    }

    fn format_id(&self) -> &str {
        "generic"
    }

    fn detect(&self, data: &[u8]) -> bool {
        let s = match std::str::from_utf8(data) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let trimmed = s.trim_start();
        // Must be a JSON array with at least one object containing "content"
        trimmed.starts_with('[') && trimmed.contains("\"content\"")
    }

    fn to_mif(&self, data: &[u8]) -> Result<MifDocument> {
        let s = std::str::from_utf8(data)?;
        let items: Vec<serde_json::Value> = serde_json::from_str(s)?;

        let mut memories = Vec::new();

        for item in &items {
            let content = match item.get("content").and_then(|v| v.as_str()) {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };

            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .and_then(|s| uuid::Uuid::parse_str(s).ok())
                .unwrap_or_else(uuid::Uuid::new_v4);

            let created_at = item
                .get("timestamp")
                .or_else(|| item.get("created_at"))
                .or_else(|| item.get("date"))
                .and_then(|v| v.as_str())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            let memory_type = item
                .get("type")
                .or_else(|| item.get("memory_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("observation")
                .to_lowercase();

            let tags: Vec<String> = item
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let mut metadata: HashMap<String, String> = HashMap::new();
            if let Some(meta) = item.get("metadata").and_then(|v| v.as_object()) {
                for (k, v) in meta {
                    metadata.insert(
                        k.clone(),
                        v.as_str()
                            .map(String::from)
                            .unwrap_or_else(|| v.to_string()),
                    );
                }
            }

            memories.push(MifMemory {
                id,
                content: content.to_string(),
                memory_type,
                created_at,
                tags,
                entities: Vec::new(),
                metadata,
                embeddings: None,
                source: Some(MifSource {
                    source_type: "generic_json".to_string(),
                    session_id: None,
                    agent: None,
                }),
                parent_id: None,
                related_memory_ids: Vec::new(),
                related_todo_ids: Vec::new(),
                agent_id: None,
                external_id: item.get("id").and_then(|v| v.as_str()).map(String::from),
                version: 1,
            });
        }

        let now = Utc::now();
        Ok(MifDocument {
            mif_version: "2.0".to_string(),
            generator: MifGenerator {
                name: "generic-json-import".to_string(),
                version: "1.0".to_string(),
            },
            export_meta: MifExportMeta {
                id: uuid::Uuid::new_v4().to_string(),
                created_at: now,
                user_id: String::new(),
                checksum: String::new(),
                privacy: None,
            },
            memories,
            knowledge_graph: None,
            todos: Vec::new(),
            projects: Vec::new(),
            reminders: Vec::new(),
            vendor_extensions: HashMap::new(),
        })
    }

    fn from_mif(&self, doc: &MifDocument) -> Result<Vec<u8>> {
        let items: Vec<serde_json::Value> = doc
            .memories
            .iter()
            .map(|m| {
                let mut obj = serde_json::json!({
                    "id": m.id.to_string(),
                    "content": m.content,
                    "type": m.memory_type,
                    "timestamp": m.created_at.to_rfc3339(),
                });
                if !m.tags.is_empty() {
                    obj["tags"] = serde_json::to_value(&m.tags).unwrap_or_default();
                }
                if !m.metadata.is_empty() {
                    obj["metadata"] = serde_json::to_value(&m.metadata).unwrap_or_default();
                }
                obj
            })
            .collect();

        let json = serde_json::to_vec_pretty(&items)?;
        Ok(json)
    }
}
