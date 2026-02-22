//! mem0 format adapter.
//!
//! mem0 exports memories as a JSON array of objects:
//! ```json
//! [
//!   {
//!     "id": "...",
//!     "memory": "The user prefers dark mode",
//!     "hash": "...",
//!     "metadata": {"category": "preference"},
//!     "created_at": "2024-01-15T10:30:00Z",
//!     "updated_at": "2024-01-15T10:30:00Z",
//!     "user_id": "user-123"
//!   }
//! ]
//! ```

use std::collections::HashMap;

use anyhow::Result;
use chrono::Utc;

use super::MifAdapter;
use crate::mif::schema::*;

pub struct Mem0Adapter;

impl MifAdapter for Mem0Adapter {
    fn name(&self) -> &str {
        "mem0"
    }

    fn format_id(&self) -> &str {
        "mem0"
    }

    fn detect(&self, data: &[u8]) -> bool {
        let s = match std::str::from_utf8(data) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let trimmed = s.trim_start();
        if !trimmed.starts_with('[') {
            return false;
        }
        // mem0 arrays have objects with a "memory" field (not "content")
        trimmed.contains("\"memory\"") && !trimmed.contains("\"mif_version\"")
    }

    fn to_mif(&self, data: &[u8]) -> Result<MifDocument> {
        let s = std::str::from_utf8(data)?;
        let items: Vec<serde_json::Value> = serde_json::from_str(s)?;

        let mut memories = Vec::new();
        let mut user_id = String::new();

        for item in &items {
            let memory_text = item
                .get("memory")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            if memory_text.is_empty() {
                continue;
            }

            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .and_then(|s| uuid::Uuid::parse_str(s).ok())
                .unwrap_or_else(uuid::Uuid::new_v4);

            if user_id.is_empty() {
                if let Some(uid) = item.get("user_id").and_then(|v| v.as_str()) {
                    user_id = uid.to_string();
                }
            }

            let created_at = item
                .get("created_at")
                .and_then(|v| v.as_str())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            // Extract metadata
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

            // Determine memory type from metadata category
            let memory_type = metadata
                .get("category")
                .map(|c| match c.as_str() {
                    "preference" | "preferences" => "observation",
                    "decision" => "decision",
                    "learning" | "fact" => "learning",
                    "error" | "mistake" => "error",
                    "task" | "todo" => "task",
                    _ => "observation",
                })
                .unwrap_or("observation")
                .to_string();

            let tags: Vec<String> = metadata
                .get("tags")
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            memories.push(MifMemory {
                id,
                content: memory_text.to_string(),
                memory_type,
                created_at,
                tags,
                entities: Vec::new(),
                metadata,
                embeddings: None,
                source: Some(MifSource {
                    source_type: "mem0".to_string(),
                    session_id: None,
                    agent: None,
                }),
                parent_id: None,
                related_memory_ids: Vec::new(),
                related_todo_ids: Vec::new(),
                agent_id: item
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                external_id: item.get("id").and_then(|v| v.as_str()).map(String::from),
                version: 1,
            });
        }

        let now = Utc::now();
        Ok(MifDocument {
            mif_version: "2.0".to_string(),
            generator: MifGenerator {
                name: "mem0-import".to_string(),
                version: "1.0".to_string(),
            },
            export_meta: MifExportMeta {
                id: uuid::Uuid::new_v4().to_string(),
                created_at: now,
                user_id,
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
        // Convert MIF memories to mem0 format
        let items: Vec<serde_json::Value> = doc
            .memories
            .iter()
            .map(|m| {
                let mut obj = serde_json::json!({
                    "id": m.id.to_string(),
                    "memory": m.content,
                    "created_at": m.created_at.to_rfc3339(),
                    "updated_at": m.created_at.to_rfc3339(),
                    "user_id": &doc.export_meta.user_id,
                });
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
