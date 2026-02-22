//! MIF v2 Schema — vendor-neutral types for memory interchange.
//!
//! Design principles:
//! 1. Core layer is universal: content, timestamps, types, tags, metadata.
//! 2. Knowledge graph uses property graph model (nodes + edges with properties).
//! 3. UUIDs preserved across round-trips for lossless import/export.
//! 4. Vendor extensions section for system-specific metadata (Hebbian state, decay, LTP).
//! 5. All string enums use lowercase snake_case for cross-system compatibility.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// =============================================================================
// TOP-LEVEL DOCUMENT
// =============================================================================

/// MIF v2 document — the top-level interchange container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifDocument {
    pub mif_version: String,
    pub generator: MifGenerator,
    pub export_meta: MifExportMeta,
    #[serde(default)]
    pub memories: Vec<MifMemory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knowledge_graph: Option<MifKnowledgeGraph>,
    #[serde(default)]
    pub todos: Vec<MifTodo>,
    #[serde(default)]
    pub projects: Vec<MifProject>,
    #[serde(default)]
    pub reminders: Vec<MifReminder>,
    /// Vendor-specific metadata. Key = vendor name (e.g., "shodh-memory").
    /// Allows lossless round-trip of system-specific data without polluting the core schema.
    #[serde(default)]
    pub vendor_extensions: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifGenerator {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifExportMeta {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub user_id: String,
    pub checksum: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy: Option<MifPrivacy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifPrivacy {
    pub pii_detected: bool,
    pub secrets_detected: bool,
    #[serde(default)]
    pub redacted_fields: Vec<String>,
}

// =============================================================================
// MEMORIES
// =============================================================================

/// A single memory in vendor-neutral form.
///
/// All IDs are raw UUIDs — no `mem_` or `todo_` prefixes.
/// Entity types are preserved (not "UNKNOWN").
/// Embeddings are model-tagged for cross-system portability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifMemory {
    pub id: Uuid,
    pub content: String,
    /// Lowercase type string: "observation", "decision", "learning", "error",
    /// "discovery", "pattern", "context", "task", "code_edit", "file_access",
    /// "search", "command", "conversation", "intention".
    pub memory_type: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub entities: Vec<MifEntityRef>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<MifEmbedding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<MifSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Uuid>,
    #[serde(default)]
    pub related_memory_ids: Vec<Uuid>,
    #[serde(default)]
    pub related_todo_ids: Vec<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    #[serde(default = "default_version")]
    pub version: u32,
}

fn default_version() -> u32 {
    1
}

/// An entity mentioned in a memory, with its type preserved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifEntityRef {
    pub name: String,
    /// Lowercase entity type: "person", "organization", "location", "technology",
    /// "concept", "event", "date", "product", "skill", "keyword", "unknown".
    #[serde(default = "default_entity_type")]
    pub entity_type: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_entity_type() -> String {
    "unknown".to_string()
}

fn default_confidence() -> f32 {
    1.0
}

/// Model-tagged embedding vector for cross-system portability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifEmbedding {
    pub model: String,
    pub dimensions: usize,
    pub vector: Vec<f32>,
    #[serde(default = "default_true")]
    pub normalized: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MifSource {
    #[serde(default)]
    pub source_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

// =============================================================================
// KNOWLEDGE GRAPH
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifKnowledgeGraph {
    #[serde(default)]
    pub entities: Vec<MifGraphEntity>,
    #[serde(default)]
    pub relationships: Vec<MifGraphRelationship>,
    #[serde(default)]
    pub episodes: Vec<MifGraphEpisode>,
}

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifGraphEntity {
    pub id: Uuid,
    pub name: String,
    /// Lowercase type strings: ["person"], ["technology", "concept"], etc.
    #[serde(default)]
    pub types: Vec<String>,
    #[serde(default)]
    pub attributes: HashMap<String, String>,
    #[serde(default)]
    pub summary: String,
    pub created_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
}

/// An edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifGraphRelationship {
    pub id: Uuid,
    pub source_entity_id: Uuid,
    pub target_entity_id: Uuid,
    /// Lowercase snake_case: "works_with", "part_of", "located_in", "causes", etc.
    pub relation_type: String,
    #[serde(default)]
    pub context: String,
    /// Normalized confidence/strength (0.0–1.0). Vendor-neutral.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    pub created_at: DateTime<Utc>,
    pub valid_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalidated_at: Option<DateTime<Utc>>,
}

/// An episodic node — a temporal grouping of entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifGraphEpisode {
    pub id: Uuid,
    pub name: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub entity_ids: Vec<Uuid>,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
}

// =============================================================================
// TODOS
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifTodo {
    pub id: Uuid,
    pub content: String,
    /// "backlog", "todo", "in_progress", "blocked", "done", "cancelled"
    pub status: String,
    /// "urgent", "high", "medium", "low", "none"
    pub priority: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub due_date: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Uuid>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub contexts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_on: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recurrence: Option<String>,
    #[serde(default)]
    pub comments: Vec<MifTodoComment>,
    #[serde(default)]
    pub related_memory_ids: Vec<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifTodoComment {
    pub id: Uuid,
    pub content: String,
    /// "comment", "progress", "resolution", "activity"
    #[serde(default = "default_comment_type")]
    pub comment_type: String,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}

fn default_comment_type() -> String {
    "comment".to_string()
}

// =============================================================================
// PROJECTS
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifProject {
    pub id: Uuid,
    pub name: String,
    #[serde(default)]
    pub prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// "active", "archived", "completed"
    #[serde(default = "default_project_status")]
    pub status: String,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
}

fn default_project_status() -> String {
    "active".to_string()
}

// =============================================================================
// REMINDERS
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifReminder {
    pub id: Uuid,
    pub content: String,
    pub trigger: MifTrigger,
    /// "pending", "triggered", "dismissed", "expired"
    pub status: String,
    #[serde(default = "default_priority")]
    pub priority: u8,
    #[serde(default)]
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triggered_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dismissed_at: Option<DateTime<Utc>>,
}

fn default_priority() -> u8 {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MifTrigger {
    Time {
        at: DateTime<Utc>,
    },
    Duration {
        seconds: u64,
        from: DateTime<Utc>,
    },
    Context {
        keywords: Vec<String>,
        #[serde(default = "default_threshold")]
        threshold: f32,
    },
}

fn default_threshold() -> f32 {
    0.65
}

// =============================================================================
// PII REDACTION RECORD
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifRedaction {
    pub redaction_type: String,
    pub original_length: usize,
    pub position: (usize, usize),
}
