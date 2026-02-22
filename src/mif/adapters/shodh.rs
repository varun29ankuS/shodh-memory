//! Shodh-native MIF adapter.
//!
//! Handles MIF v2 JSON (lossless round-trip) and MIF v1 backward compatibility.
//! v1 documents have `mif_version: "1.0"` and use `mem_`/`todo_` prefixed IDs.
//! v2 documents have `mif_version: "2.0"` and use raw UUIDs.

use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

use super::MifAdapter;
use crate::mif::schema::*;

pub struct ShodhAdapter;

impl MifAdapter for ShodhAdapter {
    fn name(&self) -> &str {
        "Shodh Memory (MIF v2/v1)"
    }

    fn format_id(&self) -> &str {
        "shodh"
    }

    fn detect(&self, data: &[u8]) -> bool {
        let s = match std::str::from_utf8(data) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let trimmed = s.trim_start();
        if !trimmed.starts_with('{') {
            return false;
        }
        // Look for MIF markers: mif_version field or shodh-memory generator
        trimmed.contains("\"mif_version\"") || trimmed.contains("\"shodh-memory\"")
    }

    fn to_mif(&self, data: &[u8]) -> Result<MifDocument> {
        let s = std::str::from_utf8(data)?;

        // Try v2 first (direct deserialization)
        if let Ok(doc) = serde_json::from_str::<MifDocument>(s) {
            return Ok(doc);
        }

        // Try v1 format and convert
        let v1: serde_json::Value = serde_json::from_str(s)?;
        convert_v1_to_v2(&v1)
    }

    fn from_mif(&self, doc: &MifDocument) -> Result<Vec<u8>> {
        let json = serde_json::to_vec_pretty(doc)?;
        Ok(json)
    }
}

/// Convert MIF v1 JSON to MIF v2 document.
///
/// v1 format differences:
/// - IDs are strings with `mem_`/`todo_` prefixes
/// - Memory types use PascalCase ("Observation" vs "observation")
/// - No vendor_extensions section
/// - Graph uses adjacency list with `entity:name` node IDs
fn convert_v1_to_v2(v1: &serde_json::Value) -> Result<MifDocument> {
    let version = v1
        .get("mif_version")
        .and_then(|v| v.as_str())
        .unwrap_or("1.0");

    if !version.starts_with("1.") {
        anyhow::bail!("Expected MIF v1.x, got: {version}");
    }

    let mut memories = Vec::new();
    if let Some(mems) = v1.get("memories").and_then(|v| v.as_array()) {
        for m in mems {
            if let Some(mem) = convert_v1_memory(m) {
                memories.push(mem);
            }
        }
    }

    let mut todos = Vec::new();
    if let Some(ts) = v1.get("todos").and_then(|v| v.as_array()) {
        for t in ts {
            if let Some(todo) = convert_v1_todo(t) {
                todos.push(todo);
            }
        }
    }

    let now = Utc::now();
    let export_id = v1
        .get("export")
        .and_then(|e| e.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("v1-import")
        .to_string();
    let user_id = v1
        .get("export")
        .and_then(|e| e.get("user_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(MifDocument {
        mif_version: "2.0".to_string(),
        generator: MifGenerator {
            name: "shodh-memory-v1-import".to_string(),
            version: version.to_string(),
        },
        export_meta: MifExportMeta {
            id: export_id,
            created_at: now,
            user_id,
            checksum: String::new(),
            privacy: None,
        },
        memories,
        knowledge_graph: None,
        todos,
        projects: Vec::new(),
        reminders: Vec::new(),
        vendor_extensions: HashMap::new(),
    })
}

fn convert_v1_memory(m: &serde_json::Value) -> Option<MifMemory> {
    let content = m.get("content")?.as_str()?;
    let id_str = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let uuid = strip_prefix_to_uuid(id_str, "mem_");

    let memory_type = m
        .get("type")
        .or_else(|| m.get("memory_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("observation")
        .to_lowercase();

    let created_at = m
        .get("created_at")
        .and_then(|v| v.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let entities: Vec<MifEntityRef> = m
        .get("entities")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let text = e.get("text").and_then(|v| v.as_str())?;
                    let entity_type = e
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_lowercase();
                    Some(MifEntityRef {
                        name: text.to_string(),
                        entity_type,
                        confidence: e.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0)
                            as f32,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let tags: Vec<String> = m
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Some(MifMemory {
        id: uuid,
        content: content.to_string(),
        memory_type,
        created_at,
        tags,
        entities,
        metadata: HashMap::new(),
        embeddings: None,
        source: None,
        parent_id: None,
        related_memory_ids: extract_uuid_list(m, "relations", "related_memories", "mem_"),
        related_todo_ids: extract_uuid_list(m, "relations", "related_todos", "todo_"),
        agent_id: None,
        external_id: None,
        version: 1,
    })
}

fn convert_v1_todo(t: &serde_json::Value) -> Option<MifTodo> {
    let content = t.get("content")?.as_str()?;
    let id_str = t.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let uuid = strip_prefix_to_uuid(id_str, "todo_");

    let created_at = t
        .get("created_at")
        .and_then(|v| v.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let updated_at = t
        .get("updated_at")
        .and_then(|v| v.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or(created_at);

    Some(MifTodo {
        id: uuid,
        content: content.to_string(),
        status: t
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("todo")
            .to_string(),
        priority: t
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("medium")
            .to_string(),
        created_at,
        updated_at,
        due_date: None,
        completed_at: None,
        project_id: None,
        parent_id: None,
        tags: Vec::new(),
        contexts: Vec::new(),
        notes: t.get("notes").and_then(|v| v.as_str()).map(String::from),
        blocked_on: None,
        recurrence: None,
        comments: Vec::new(),
        related_memory_ids: Vec::new(),
        external_id: None,
    })
}

fn strip_prefix_to_uuid(s: &str, prefix: &str) -> uuid::Uuid {
    let stripped = s.strip_prefix(prefix).unwrap_or(s);
    uuid::Uuid::parse_str(stripped).unwrap_or_else(|_| uuid::Uuid::new_v4())
}

fn extract_uuid_list(
    val: &serde_json::Value,
    container_key: &str,
    array_key: &str,
    prefix: &str,
) -> Vec<uuid::Uuid> {
    val.get(container_key)
        .and_then(|c| c.get(array_key))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| strip_prefix_to_uuid(s, prefix))
                .collect()
        })
        .unwrap_or_default()
}
