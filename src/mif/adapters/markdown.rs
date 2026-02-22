//! Markdown adapter with YAML frontmatter.
//!
//! Supports Letta/Obsidian-style memory files:
//! ```markdown
//! ---
//! type: observation
//! tags: [rust, programming]
//! created_at: 2024-01-15T10:30:00Z
//! ---
//! The user prefers Rust for systems programming.
//! ---
//! ---
//! type: decision
//! tags: [architecture]
//! ---
//! We decided to use RocksDB for storage.
//! ```
//!
//! Multiple memories are separated by double `---` (frontmatter end + next frontmatter start).

use std::collections::HashMap;

use anyhow::Result;
use chrono::Utc;

use super::MifAdapter;
use crate::mif::schema::*;

pub struct MarkdownAdapter;

impl MifAdapter for MarkdownAdapter {
    fn name(&self) -> &str {
        "Markdown (YAML frontmatter)"
    }

    fn format_id(&self) -> &str {
        "markdown"
    }

    fn detect(&self, data: &[u8]) -> bool {
        let s = match std::str::from_utf8(data) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let trimmed = s.trim_start();
        // Must start with YAML frontmatter delimiter
        trimmed.starts_with("---")
    }

    fn to_mif(&self, data: &[u8]) -> Result<MifDocument> {
        let s = std::str::from_utf8(data)?;
        let blocks = split_frontmatter_blocks(s);

        let mut memories = Vec::new();

        for (frontmatter, body) in blocks {
            let content = body.trim();
            if content.is_empty() {
                continue;
            }

            let fm = parse_frontmatter(&frontmatter);

            let id = fm
                .get("id")
                .and_then(|s| uuid::Uuid::parse_str(s).ok())
                .unwrap_or_else(uuid::Uuid::new_v4);

            let memory_type = fm
                .get("type")
                .cloned()
                .unwrap_or_else(|| "observation".to_string());

            let created_at = fm
                .get("created_at")
                .or_else(|| fm.get("date"))
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            let tags: Vec<String> = fm
                .get("tags")
                .map(|t| {
                    // Handle both "[tag1, tag2]" and "tag1, tag2" formats
                    let cleaned = t.trim_start_matches('[').trim_end_matches(']');
                    cleaned
                        .split(',')
                        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default();

            // Collect remaining frontmatter fields as metadata
            let metadata: HashMap<String, String> = fm
                .iter()
                .filter(|(k, _)| {
                    !matches!(k.as_str(), "type" | "tags" | "created_at" | "date" | "id")
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

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
                    source_type: "markdown".to_string(),
                    session_id: None,
                    agent: None,
                }),
                parent_id: None,
                related_memory_ids: Vec::new(),
                related_todo_ids: Vec::new(),
                agent_id: None,
                external_id: None,
                version: 1,
            });
        }

        let now = Utc::now();
        Ok(MifDocument {
            mif_version: "2.0".to_string(),
            generator: MifGenerator {
                name: "markdown-import".to_string(),
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
        let mut output = String::new();

        for (i, m) in doc.memories.iter().enumerate() {
            if i > 0 {
                output.push_str("\n---\n");
            }
            output.push_str("---\n");
            output.push_str(&format!("type: {}\n", m.memory_type));
            output.push_str(&format!("created_at: {}\n", m.created_at.to_rfc3339()));
            if !m.tags.is_empty() {
                output.push_str(&format!("tags: [{}]\n", m.tags.join(", ")));
            }
            output.push_str("---\n");
            output.push_str(&m.content);
            output.push('\n');
        }

        Ok(output.into_bytes())
    }
}

/// Split markdown into (frontmatter, body) blocks.
///
/// Each block starts with `---` (frontmatter), ends with `---` (delimiter),
/// followed by the body content. Multiple blocks are separated by `---\n---`.
fn split_frontmatter_blocks(input: &str) -> Vec<(String, String)> {
    let mut blocks = Vec::new();
    let lines: Vec<&str> = input.lines().collect();

    if lines.is_empty() || lines[0].trim() != "---" {
        return blocks;
    }

    let mut i = 0;
    while i < lines.len() {
        // Expect opening ---
        if lines[i].trim() != "---" {
            i += 1;
            continue;
        }
        i += 1; // skip opening ---

        // Collect frontmatter until closing ---
        let mut frontmatter = String::new();
        while i < lines.len() && lines[i].trim() != "---" {
            frontmatter.push_str(lines[i]);
            frontmatter.push('\n');
            i += 1;
        }
        if i < lines.len() {
            i += 1; // skip closing ---
        }

        // Collect body until next opening --- or end
        let mut body = String::new();
        while i < lines.len() && lines[i].trim() != "---" {
            body.push_str(lines[i]);
            body.push('\n');
            i += 1;
        }

        if !frontmatter.is_empty() || !body.is_empty() {
            blocks.push((frontmatter, body));
        }
    }

    blocks
}

/// Parse simple YAML frontmatter into key-value pairs.
///
/// Handles `key: value` format. Does not support nested YAML.
fn parse_frontmatter(fm: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in fm.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(pos) = line.find(':') {
            let key = line[..pos].trim().to_string();
            let value = line[pos + 1..].trim().to_string();
            if !key.is_empty() {
                map.insert(key, value);
            }
        }
    }
    map
}
