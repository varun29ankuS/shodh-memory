//! Memory Interchange Format (MIF) and Multimodal Search Handlers
//!
//! This module handles:
//! - MIF export with optional PII redaction and encryption
//! - MIF import with merge strategies
//! - Multimodal search (similarity, temporal, causal, associative, hybrid)
//! - Robotics-specific search (spatial, mission, action_outcome)

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::state::MultiUserMemoryManager;
use super::types::RetrieveResponse;
use crate::errors::{AppError, ValidationErrorExt};
use crate::graph_memory;
use crate::memory::{self, Memory, Query as MemoryQuery};
use crate::validation;

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// MIF EXPORT TYPES
// =============================================================================

/// Request for MIF export
#[derive(Debug, Deserialize)]
pub struct MifExportRequest {
    pub user_id: String,
    #[serde(default)]
    pub include_embeddings: bool,
    #[serde(default)]
    pub include_graph: bool,
    #[serde(default)]
    pub since: Option<String>,
    #[serde(default)]
    pub redact_pii: bool,
    #[serde(default)]
    pub encrypt: bool,
    #[serde(default)]
    pub encryption_key: Option<String>,
    #[serde(default = "default_mif_format")]
    pub format: String,
}

fn default_mif_format() -> String {
    "json".to_string()
}

/// Request for MIF import
#[derive(Debug, Deserialize)]
pub struct MifImportRequest {
    pub user_id: String,
    #[serde(default = "default_merge_strategy")]
    pub merge_strategy: String,
    #[serde(default)]
    pub decrypt: bool,
    #[serde(default)]
    pub decryption_key: Option<String>,
    pub data: MifImportData,
}

fn default_merge_strategy() -> String {
    "skip_duplicates".to_string()
}

/// The actual MIF data being imported
#[derive(Debug, Deserialize)]
pub struct MifImportData {
    pub mif_version: String,
    #[serde(default)]
    pub memories: Vec<MifMemory>,
    #[serde(default)]
    pub todos: Vec<MifTodo>,
    #[serde(default)]
    pub graph: Option<MifGraph>,
}

/// MIF Memory object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifMemory {
    pub id: String,
    pub content: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    #[serde(default = "default_importance")]
    pub importance: f32,
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub accessed_at: String,
    #[serde(default)]
    pub access_count: u32,
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f32,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: MifSource,
    #[serde(default)]
    pub entities: Vec<MifEntity>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub embedding: Option<MifEmbedding>,
    #[serde(default)]
    pub relations: MifRelations,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub redactions: Option<Vec<MifRedaction>>,
}

fn default_importance() -> f32 {
    0.5
}

fn default_decay_rate() -> f32 {
    0.1
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MifSource {
    #[serde(rename = "type", default)]
    pub source_type: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub agent: Option<String>,
}

/// PII redaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifRedaction {
    #[serde(rename = "type")]
    pub redaction_type: String,
    pub original_length: usize,
    pub position: (usize, usize),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MifEntity {
    #[serde(default)]
    pub text: String,
    #[serde(rename = "type", default)]
    pub entity_type: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifEmbedding {
    pub model: String,
    pub dimensions: usize,
    pub vector: Vec<f32>,
    #[serde(default = "default_normalized")]
    pub normalized: bool,
}

fn default_normalized() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MifRelations {
    #[serde(default)]
    pub related_memories: Vec<String>,
    #[serde(default)]
    pub related_todos: Vec<String>,
}

/// MIF Todo object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifTodo {
    pub id: String,
    #[serde(default)]
    pub short_id: String,
    pub content: String,
    #[serde(default = "default_todo_status")]
    pub status: String,
    #[serde(default = "default_priority")]
    pub priority: String,
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub due_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub completed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub project: Option<MifProject>,
    #[serde(default)]
    pub contexts: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub subtask_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub blocked_on: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub recurrence: Option<String>,
    #[serde(default)]
    pub related_memory_ids: Vec<String>,
    #[serde(default)]
    pub comments: Vec<MifComment>,
}

fn default_todo_status() -> String {
    "todo".to_string()
}

fn default_priority() -> String {
    "medium".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifProject {
    pub id: String,
    pub name: String,
    #[serde(default = "default_prefix")]
    pub prefix: String,
}

fn default_prefix() -> String {
    "TODO".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifComment {
    pub id: String,
    pub content: String,
    #[serde(rename = "type", default = "default_comment_type")]
    pub comment_type: String,
    pub created_at: String,
}

fn default_comment_type() -> String {
    "comment".to_string()
}

/// MIF Graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifGraph {
    #[serde(default = "default_graph_format")]
    pub format: String,
    #[serde(default)]
    pub node_count: usize,
    #[serde(default)]
    pub edge_count: usize,
    #[serde(default)]
    pub nodes: Vec<MifNode>,
    #[serde(default)]
    pub edges: Vec<MifEdge>,
    #[serde(default)]
    pub hebbian_config: MifHebbianConfig,
}

fn default_graph_format() -> String {
    "adjacency_list".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub entity_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MifEdge {
    pub source: String,
    pub target: String,
    #[serde(default = "default_edge_weight")]
    pub weight: f32,
    #[serde(rename = "type", default = "default_edge_type")]
    pub edge_type: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub strengthened_count: u32,
}

fn default_edge_weight() -> f32 {
    0.5
}

fn default_edge_type() -> String {
    "semantic_association".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MifHebbianConfig {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,
    #[serde(default = "default_hebbian_decay")]
    pub decay_rate: f32,
    #[serde(default = "default_ltp_threshold")]
    pub ltp_threshold: f32,
    #[serde(default = "default_max_weight")]
    pub max_weight: f32,
}

fn default_learning_rate() -> f32 {
    0.1
}

fn default_hebbian_decay() -> f32 {
    0.05
}

fn default_ltp_threshold() -> f32 {
    0.8
}

fn default_max_weight() -> f32 {
    1.0
}

/// MIF Metadata
#[derive(Debug, Serialize)]
pub struct MifMetadata {
    pub total_memories: usize,
    pub total_todos: usize,
    pub date_range: MifDateRange,
    pub memory_types: HashMap<String, usize>,
    pub top_entities: Vec<MifTopEntity>,
    pub projects: Vec<MifProjectStats>,
    pub privacy: MifPrivacy,
}

#[derive(Debug, Serialize)]
pub struct MifDateRange {
    pub earliest: String,
    pub latest: String,
}

#[derive(Debug, Serialize)]
pub struct MifTopEntity {
    pub text: String,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct MifProjectStats {
    pub id: String,
    pub name: String,
    pub todo_count: usize,
}

#[derive(Debug, Serialize)]
pub struct MifPrivacy {
    pub pii_detected: bool,
    pub secrets_detected: bool,
    pub redacted_fields: Vec<String>,
}

/// Full MIF export document
#[derive(Debug, Serialize)]
pub struct MifExport {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub mif_version: String,
    pub generator: MifGenerator,
    pub export: MifExportMeta,
    pub memories: Vec<MifMemory>,
    pub todos: Vec<MifTodo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<MifGraph>,
    pub metadata: MifMetadata,
}

#[derive(Debug, Serialize)]
pub struct MifGenerator {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Serialize)]
pub struct MifExportMeta {
    pub id: String,
    pub created_at: String,
    pub user_id: String,
    pub checksum: String,
}

// =============================================================================
// MIF IMPORT TYPES
// =============================================================================

/// Response for MIF import
#[derive(Debug, Serialize)]
pub struct MifImportResponse {
    pub success: bool,
    pub imported: MifImportCounts,
    pub skipped: MifSkipCounts,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct MifImportCounts {
    pub memories: usize,
    pub todos: usize,
    pub edges: usize,
}

#[derive(Debug, Serialize, Default)]
pub struct MifSkipCounts {
    pub duplicates: usize,
    pub invalid: usize,
}

// =============================================================================
// PII DETECTION AND REDACTION
// =============================================================================

/// PII patterns for detection
pub struct PiiPatterns {
    email: regex::Regex,
    phone: regex::Regex,
    ssn: regex::Regex,
    api_key: regex::Regex,
    credit_card: regex::Regex,
    #[allow(dead_code)]
    ip_address: regex::Regex,
}

impl PiiPatterns {
    pub fn new() -> Self {
        Self {
            email: regex::Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
            phone: regex::Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b").unwrap(),
            ssn: regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            api_key: regex::Regex::new(
                r#"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['"]?[\w-]{16,}['"]?"#,
            )
            .unwrap(),
            credit_card: regex::Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b").unwrap(),
            ip_address: regex::Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap(),
        }
    }

    /// Detect and redact PII from content
    pub fn redact(&self, content: &str) -> (String, Vec<MifRedaction>, bool) {
        let mut redacted = content.to_string();
        let mut redactions = Vec::new();
        let mut pii_found = false;

        // Email
        for m in self.email.find_iter(content) {
            pii_found = true;
            redactions.push(MifRedaction {
                redaction_type: "email".to_string(),
                original_length: m.as_str().len(),
                position: (m.start(), m.end()),
            });
        }
        redacted = self
            .email
            .replace_all(&redacted, "[REDACTED:email]")
            .to_string();

        // Phone
        for m in self.phone.find_iter(content) {
            pii_found = true;
            redactions.push(MifRedaction {
                redaction_type: "phone".to_string(),
                original_length: m.as_str().len(),
                position: (m.start(), m.end()),
            });
        }
        redacted = self
            .phone
            .replace_all(&redacted, "[REDACTED:phone]")
            .to_string();

        // SSN
        for m in self.ssn.find_iter(content) {
            pii_found = true;
            redactions.push(MifRedaction {
                redaction_type: "ssn".to_string(),
                original_length: m.as_str().len(),
                position: (m.start(), m.end()),
            });
        }
        redacted = self
            .ssn
            .replace_all(&redacted, "[REDACTED:ssn]")
            .to_string();

        // API keys
        for m in self.api_key.find_iter(content) {
            pii_found = true;
            redactions.push(MifRedaction {
                redaction_type: "api_key".to_string(),
                original_length: m.as_str().len(),
                position: (m.start(), m.end()),
            });
        }
        redacted = self
            .api_key
            .replace_all(&redacted, "[REDACTED:api_key]")
            .to_string();

        // Credit card
        for m in self.credit_card.find_iter(content) {
            pii_found = true;
            redactions.push(MifRedaction {
                redaction_type: "credit_card".to_string(),
                original_length: m.as_str().len(),
                position: (m.start(), m.end()),
            });
        }
        redacted = self
            .credit_card
            .replace_all(&redacted, "[REDACTED:credit_card]")
            .to_string();

        (redacted, redactions, pii_found)
    }
}

impl Default for PiiPatterns {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ENCRYPTION (AES-256-GCM)
// =============================================================================

/// Encrypted MIF export wrapper
#[derive(Debug, Serialize)]
pub struct MifEncryptedExport {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub mif_version: String,
    pub encryption: MifEncryptionMeta,
}

#[derive(Debug, Serialize)]
pub struct MifEncryptionMeta {
    pub algorithm: String,
    pub key_derivation: String,
    pub encrypted_payload: String,
    pub iv: String,
    pub auth_tag: String,
}

/// Encrypt MIF export data using AES-256-GCM
#[allow(dead_code)]
pub fn encrypt_mif_data(
    data: &[u8],
    key: &[u8; 32],
) -> Result<(Vec<u8>, [u8; 12], [u8; 16]), anyhow::Error> {
    use aes_gcm::{
        aead::{generic_array::GenericArray, Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use rand::RngCore;

    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let cipher = Aes256Gcm::new(GenericArray::from_slice(key));

    let ciphertext = cipher
        .encrypt(nonce, data)
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    let (ct, tag) = ciphertext.split_at(ciphertext.len() - 16);
    let mut auth_tag = [0u8; 16];
    auth_tag.copy_from_slice(tag);

    Ok((ct.to_vec(), nonce_bytes, auth_tag))
}

/// Decrypt MIF export data using AES-256-GCM
#[allow(dead_code)]
pub fn decrypt_mif_data(
    ciphertext: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 12],
    auth_tag: &[u8; 16],
) -> Result<Vec<u8>, anyhow::Error> {
    use aes_gcm::{
        aead::{generic_array::GenericArray, Aead, KeyInit},
        Aes256Gcm, Nonce,
    };

    let mut ct_with_tag = ciphertext.to_vec();
    ct_with_tag.extend_from_slice(auth_tag);

    let cipher = Aes256Gcm::new(GenericArray::from_slice(key));
    let nonce = Nonce::from_slice(nonce);

    cipher
        .decrypt(nonce, ct_with_tag.as_slice())
        .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))
}

// =============================================================================
// MIF EXPORT HANDLER
// =============================================================================

/// Export memories in MIF (Memory Interchange Format)
#[tracing::instrument(skip(state), fields(user_id = %req.user_id))]
pub async fn export_mif(
    State(state): State<AppState>,
    Json(req): Json<MifExportRequest>,
) -> Result<Json<MifExport>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let user_id = req.user_id.clone();
    let include_embeddings = req.include_embeddings;
    let redact_pii = req.redact_pii;

    let pii_patterns = if redact_pii {
        Some(PiiPatterns::new())
    } else {
        None
    };

    use std::cell::Cell;
    let pii_detected = Cell::new(false);
    let secrets_detected = Cell::new(false);
    let mut redacted_fields: Vec<String> = Vec::new();

    // Get all memories
    let memories: Vec<MifMemory> = {
        let guard = memory_sys.read();
        guard
            .get_all_memories()
            .unwrap_or_default()
            .into_iter()
            .map(|m| {
                let embedding = if include_embeddings {
                    m.experience.embeddings.as_ref().map(|e| MifEmbedding {
                        model: "minilm-l6-v2".to_string(),
                        dimensions: e.len(),
                        vector: e.clone(),
                        normalized: true,
                    })
                } else {
                    None
                };

                let (content, redactions) = if let Some(ref patterns) = pii_patterns {
                    let (redacted_content, redaction_records, found_pii) =
                        patterns.redact(&m.experience.content);
                    if found_pii {
                        pii_detected.set(true);
                        if redaction_records
                            .iter()
                            .any(|r| r.redaction_type == "api_key")
                        {
                            secrets_detected.set(true);
                        }
                    }
                    (
                        redacted_content,
                        if redaction_records.is_empty() {
                            None
                        } else {
                            Some(redaction_records)
                        },
                    )
                } else {
                    (m.experience.content.clone(), None)
                };

                let entities: Vec<MifEntity> = m
                    .experience
                    .entities
                    .iter()
                    .map(|e| MifEntity {
                        text: e.clone(),
                        entity_type: "UNKNOWN".to_string(),
                        confidence: 1.0,
                    })
                    .collect();

                let (source_type, session_id) = m
                    .experience
                    .context
                    .as_ref()
                    .map(|ctx| {
                        let src = format!("{:?}", ctx.source.source_type).to_lowercase();
                        let sess = ctx.episode.episode_id.clone();
                        (src, sess)
                    })
                    .unwrap_or_else(|| ("conversation".to_string(), None));

                let tags: Vec<String> = m
                    .experience
                    .metadata
                    .get("tags")
                    .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                    .unwrap_or_default();

                MifMemory {
                    id: format!("mem_{}", m.id.0),
                    content,
                    memory_type: format!("{:?}", m.experience.experience_type),
                    importance: m.importance(),
                    created_at: m.created_at.to_rfc3339(),
                    updated_at: m.created_at.to_rfc3339(),
                    accessed_at: m.last_accessed().to_rfc3339(),
                    access_count: m.access_count(),
                    decay_rate: 0.1,
                    tags,
                    source: MifSource {
                        source_type,
                        session_id,
                        agent: Some("shodh-memory".to_string()),
                    },
                    entities,
                    embedding,
                    relations: MifRelations {
                        related_memories: m
                            .experience
                            .related_memories
                            .iter()
                            .map(|id| format!("mem_{}", id.0))
                            .collect(),
                        related_todos: m
                            .related_todo_ids
                            .iter()
                            .map(|id| format!("todo_{}", id.0))
                            .collect(),
                    },
                    redactions,
                }
            })
            .collect()
    };

    if redact_pii {
        for mem in &memories {
            if let Some(ref redacts) = mem.redactions {
                for r in redacts {
                    if !redacted_fields.contains(&r.redaction_type) {
                        redacted_fields.push(r.redaction_type.clone());
                    }
                }
            }
        }
    }

    // Get all todos
    let todos: Vec<MifTodo> = state
        .todo_store
        .list_todos_for_user(&user_id, None)
        .unwrap_or_default()
        .into_iter()
        .map(|t| {
            let project = t.project_id.as_ref().and_then(|pid| {
                state
                    .todo_store
                    .get_project(&user_id, pid)
                    .ok()
                    .flatten()
                    .map(|p| MifProject {
                        id: p.id.0.to_string(),
                        name: p.name,
                        prefix: p.prefix.unwrap_or_else(|| "TODO".to_string()),
                    })
            });

            MifTodo {
                id: format!("todo_{}", t.id.0),
                short_id: t.short_id(),
                content: t.content,
                status: format!("{:?}", t.status).to_lowercase(),
                priority: t.priority.indicator().to_string(),
                created_at: t.created_at.to_rfc3339(),
                updated_at: t.updated_at.to_rfc3339(),
                due_date: t.due_date.map(|d| d.to_rfc3339()),
                completed_at: t.completed_at.map(|d| d.to_rfc3339()),
                project,
                contexts: t.contexts,
                tags: t.tags,
                notes: t.notes,
                parent_id: t.parent_id.map(|id| format!("todo_{}", id.0)),
                subtask_ids: vec![],
                blocked_on: t.blocked_on,
                recurrence: t.recurrence.map(|r| format!("{:?}", r).to_lowercase()),
                related_memory_ids: t
                    .related_memory_ids
                    .iter()
                    .map(|id| format!("mem_{}", id.0))
                    .collect(),
                comments: vec![],
            }
        })
        .collect();

    // Build graph if requested
    let graph = if req.include_graph {
        let graph_memory = state.get_user_graph(&user_id).map_err(AppError::Internal)?;
        let graph_guard = graph_memory.read();

        let mut nodes = Vec::new();

        for m in &memories {
            nodes.push(MifNode {
                id: m.id.clone(),
                node_type: "memory".to_string(),
                entity_type: None,
            });
        }

        if let Ok(entities) = graph_guard.get_all_entities() {
            for entity in entities {
                let entity_type = entity
                    .labels
                    .first()
                    .map(|l| format!("{:?}", l))
                    .unwrap_or_else(|| "UNKNOWN".to_string());
                nodes.push(MifNode {
                    id: format!("entity:{}", entity.name),
                    node_type: "entity".to_string(),
                    entity_type: Some(entity_type),
                });
            }
        }

        for t in &todos {
            nodes.push(MifNode {
                id: t.id.clone(),
                node_type: "todo".to_string(),
                entity_type: None,
            });
        }

        let mut edges = Vec::new();
        if let Ok(relationships) = graph_guard.get_all_relationships() {
            for rel in relationships {
                edges.push(MifEdge {
                    source: format!("entity:{}", rel.from_entity),
                    target: format!("entity:{}", rel.to_entity),
                    weight: rel.strength,
                    edge_type: format!("{:?}", rel.relation_type).to_lowercase(),
                    created_at: rel.created_at.to_rfc3339(),
                    strengthened_count: rel.activation_count,
                });
            }
        }

        Some(MifGraph {
            format: "adjacency_list".to_string(),
            node_count: nodes.len(),
            edge_count: edges.len(),
            nodes,
            edges,
            hebbian_config: MifHebbianConfig {
                learning_rate: 0.1,
                decay_rate: 0.05,
                ltp_threshold: 0.8,
                max_weight: 1.0,
            },
        })
    } else {
        None
    };

    // Build metadata
    let mut memory_types: HashMap<String, usize> = HashMap::new();
    for m in &memories {
        *memory_types.entry(m.memory_type.clone()).or_insert(0) += 1;
    }

    let all_todos_for_count = state
        .todo_store
        .list_todos_for_user(&user_id, None)
        .unwrap_or_default();
    let projects: Vec<MifProjectStats> = state
        .todo_store
        .list_projects(&user_id)
        .unwrap_or_default()
        .into_iter()
        .map(|p| {
            let todo_count = all_todos_for_count
                .iter()
                .filter(|t| t.project_id.as_ref() == Some(&p.id))
                .count();
            MifProjectStats {
                id: p.id.0.to_string(),
                name: p.name,
                todo_count,
            }
        })
        .collect();

    let earliest = memories
        .iter()
        .map(|m| &m.created_at)
        .min()
        .cloned()
        .unwrap_or_default();
    let latest = memories
        .iter()
        .map(|m| &m.created_at)
        .max()
        .cloned()
        .unwrap_or_default();

    let metadata = MifMetadata {
        total_memories: memories.len(),
        total_todos: todos.len(),
        date_range: MifDateRange { earliest, latest },
        memory_types,
        top_entities: vec![],
        projects,
        privacy: MifPrivacy {
            pii_detected: pii_detected.get(),
            secrets_detected: secrets_detected.get(),
            redacted_fields,
        },
    };

    let export_id = format!(
        "exp_{}",
        uuid::Uuid::new_v4().to_string().replace('-', "")[..12].to_string()
    );
    let now = chrono::Utc::now();

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(format!(
        "{}{}{}",
        memories.len(),
        todos.len(),
        now.to_rfc3339()
    ));
    let checksum = format!("sha256:{}", hex::encode(hasher.finalize()));

    let export = MifExport {
        schema: "https://shodh-memory.dev/schemas/mif-v1.json".to_string(),
        mif_version: "1.0".to_string(),
        generator: MifGenerator {
            name: "shodh-memory".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
        export: MifExportMeta {
            id: export_id,
            created_at: now.to_rfc3339(),
            user_id: user_id.clone(),
            checksum,
        },
        memories,
        todos,
        graph,
        metadata,
    };

    state.log_event(
        &user_id,
        "MIF_EXPORT",
        &export.export.id,
        &format!(
            "Exported {} memories, {} todos",
            export.metadata.total_memories, export.metadata.total_todos
        ),
    );

    Ok(Json(export))
}

// =============================================================================
// MIF IMPORT HANDLER
// =============================================================================

/// Import memories from MIF format
pub async fn import_mif(
    State(state): State<AppState>,
    Json(req): Json<MifImportRequest>,
) -> Result<Json<MifImportResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    if !req.data.mif_version.starts_with("1.") {
        return Err(AppError::InvalidInput {
            field: "mif_version".to_string(),
            reason: format!(
                "Unsupported MIF version: {}. Only 1.x is supported.",
                req.data.mif_version
            ),
        });
    }

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let mut imported = MifImportCounts::default();
    let mut skipped = MifSkipCounts::default();
    let mut warnings = Vec::new();

    // Import memories
    for mif_mem in &req.data.memories {
        let exp_type = match mif_mem.memory_type.as_str() {
            "Observation" => memory::ExperienceType::Observation,
            "Decision" => memory::ExperienceType::Decision,
            "Learning" => memory::ExperienceType::Learning,
            "Error" => memory::ExperienceType::Error,
            "Discovery" => memory::ExperienceType::Discovery,
            "Pattern" => memory::ExperienceType::Pattern,
            "Context" => memory::ExperienceType::Context,
            "Task" => memory::ExperienceType::Task,
            "CodeEdit" => memory::ExperienceType::CodeEdit,
            "FileAccess" => memory::ExperienceType::FileAccess,
            "Search" => memory::ExperienceType::Search,
            "Command" => memory::ExperienceType::Command,
            "Conversation" => memory::ExperienceType::Conversation,
            _ => memory::ExperienceType::Observation,
        };

        let created_at = chrono::DateTime::parse_from_rfc3339(&mif_mem.created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .ok();

        let mut metadata: HashMap<String, String> = HashMap::new();
        if !mif_mem.tags.is_empty() {
            metadata.insert("tags".to_string(), mif_mem.tags.join(","));
        }

        let experience = memory::Experience {
            experience_type: exp_type,
            content: mif_mem.content.clone(),
            entities: mif_mem.entities.iter().map(|e| e.text.clone()).collect(),
            metadata,
            embeddings: mif_mem.embedding.as_ref().map(|e| e.vector.clone()),
            ..Default::default()
        };

        let should_import = match req.merge_strategy.as_str() {
            "skip_duplicates" => {
                let guard = memory_sys.read();
                let query = memory::Query {
                    query_text: Some(mif_mem.content.clone()),
                    max_results: 5,
                    ..Default::default()
                };
                let existing = guard.recall(&query).unwrap_or_default();

                if existing
                    .iter()
                    .any(|m| m.experience.content == mif_mem.content)
                {
                    skipped.duplicates += 1;
                    false
                } else {
                    true
                }
            }
            "overwrite" => true,
            "rename" => true,
            _ => true,
        };

        if should_import {
            let guard = memory_sys.read();
            match guard.remember(experience, created_at) {
                Ok(_) => imported.memories += 1,
                Err(e) => {
                    warnings.push(format!("Failed to import memory {}: {}", mif_mem.id, e));
                    skipped.invalid += 1;
                }
            }
        }
    }

    // Import todos
    for mif_todo in &req.data.todos {
        let status = match mif_todo.status.as_str() {
            "backlog" => memory::TodoStatus::Backlog,
            "todo" => memory::TodoStatus::Todo,
            "in_progress" => memory::TodoStatus::InProgress,
            "blocked" => memory::TodoStatus::Blocked,
            "done" => memory::TodoStatus::Done,
            "cancelled" => memory::TodoStatus::Cancelled,
            _ => memory::TodoStatus::Todo,
        };

        let priority = match mif_todo.priority.as_str() {
            "urgent" | "!!!" => memory::TodoPriority::Urgent,
            "high" | "!!" => memory::TodoPriority::High,
            "medium" | "!" => memory::TodoPriority::Medium,
            "low" | "-" => memory::TodoPriority::Low,
            _ => memory::TodoPriority::None,
        };

        let created_at = chrono::DateTime::parse_from_rfc3339(&mif_todo.created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());

        let due_date = mif_todo
            .due_date
            .as_ref()
            .and_then(|d| chrono::DateTime::parse_from_rfc3339(d).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        let todo = memory::Todo {
            id: memory::TodoId(uuid::Uuid::new_v4()),
            seq_num: 0,
            project_prefix: None,
            user_id: req.user_id.clone(),
            content: mif_todo.content.clone(),
            status,
            priority,
            project_id: None,
            parent_id: None,
            created_at,
            updated_at: created_at,
            due_date,
            completed_at: None,
            contexts: mif_todo.contexts.clone(),
            tags: mif_todo.tags.clone(),
            notes: mif_todo.notes.clone(),
            blocked_on: mif_todo.blocked_on.clone(),
            recurrence: None,
            sort_order: 0,
            comments: Vec::new(),
            embedding: None,
            related_memory_ids: Vec::new(),
            external_id: None,
        };

        match state.todo_store.store_todo(&todo) {
            Ok(_) => imported.todos += 1,
            Err(e) => {
                warnings.push(format!("Failed to import todo {}: {}", mif_todo.id, e));
                skipped.invalid += 1;
            }
        }
    }

    // Import graph edges if present
    if let Some(ref graph) = req.data.graph {
        for edge in &graph.edges {
            if !edge.source.starts_with("mem_") || !edge.target.starts_with("mem_") {
                continue;
            }
            imported.edges += 1;
        }
    }

    state.log_event(
        &req.user_id,
        "MIF_IMPORT",
        "import",
        &format!(
            "Imported {} memories, {} todos, {} edges. Skipped: {} duplicates, {} invalid",
            imported.memories, imported.todos, imported.edges, skipped.duplicates, skipped.invalid
        ),
    );

    Ok(Json(MifImportResponse {
        success: true,
        imported,
        skipped,
        warnings,
    }))
}

// =============================================================================
// MULTIMODAL SEARCH HANDLERS
// =============================================================================

/// Request for multimodal search
#[derive(Debug, Deserialize)]
pub struct MultiModalSearchRequest {
    pub user_id: String,
    pub query_text: String,
    pub mode: String,
    pub limit: Option<usize>,
}

/// Advanced multi-modal retrieval
pub async fn multimodal_search(
    State(state): State<AppState>,
    Json(req): Json<MultiModalSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let retrieval_mode = match req.mode.as_str() {
        "similarity" => memory::RetrievalMode::Similarity,
        "temporal" => memory::RetrievalMode::Temporal,
        "causal" => memory::RetrievalMode::Causal,
        "associative" => memory::RetrievalMode::Associative,
        "hybrid" => memory::RetrievalMode::Hybrid,
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        _ => {
            return Err(AppError::InvalidInput {
                field: "mode".to_string(),
                reason: format!(
                    "Invalid mode: {}. Must be one of: similarity, temporal, causal, associative, hybrid, spatial, mission, action_outcome",
                    req.mode
                ),
            })
        }
    };

    let query = MemoryQuery {
        query_text: Some(req.query_text.clone()),
        max_results: req.limit.unwrap_or(10),
        retrieval_mode,
        ..Default::default()
    };

    let shared_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;
    let raw_memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();
    let count = raw_memories.len();

    // Serialize memories to JSON for response
    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    state.log_event(
        &req.user_id,
        "MULTIMODAL_SEARCH",
        &req.mode,
        &format!("Retrieved {} memories using {} mode", count, req.mode),
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

// =============================================================================
// ROBOTICS SEARCH HANDLER
// =============================================================================

/// Request for robotics-specific memory search
#[derive(Debug, Deserialize)]
pub struct RoboticsSearchRequest {
    pub user_id: String,
    pub mode: String,
    pub query_text: Option<String>,
    pub robot_id: Option<String>,
    pub mission_id: Option<String>,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub radius_meters: Option<f64>,
    pub action_type: Option<String>,
    pub min_reward: Option<f32>,
    pub max_reward: Option<f32>,
    pub limit: Option<usize>,
}

/// Robotics-specific memory search
pub async fn robotics_search(
    State(state): State<AppState>,
    Json(req): Json<RoboticsSearchRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let retrieval_mode = match req.mode.as_str() {
        "spatial" => memory::RetrievalMode::Spatial,
        "mission" => memory::RetrievalMode::Mission,
        "action_outcome" => memory::RetrievalMode::ActionOutcome,
        "hybrid" => memory::RetrievalMode::Hybrid,
        "similarity" => memory::RetrievalMode::Similarity,
        _ => {
            return Err(AppError::InvalidInput {
                field: "mode".to_string(),
                reason: "Invalid mode. Use: spatial, mission, action_outcome, hybrid, similarity"
                    .to_string(),
            })
        }
    };

    let geo_filter = match (req.lat, req.lon, req.radius_meters) {
        (Some(lat), Some(lon), Some(radius)) => Some(memory::GeoFilter::new(lat, lon, radius)),
        _ => None,
    };

    let reward_range = match (req.min_reward, req.max_reward) {
        (Some(min), Some(max)) => Some((min, max)),
        (Some(min), None) => Some((min, 1.0)),
        (None, Some(max)) => Some((-1.0, max)),
        _ => None,
    };

    if matches!(retrieval_mode, memory::RetrievalMode::Spatial) && geo_filter.is_none() {
        return Err(AppError::InvalidInput {
            field: "lat/lon/radius_meters".to_string(),
            reason: "Spatial mode requires lat, lon, and radius_meters".to_string(),
        });
    }

    if matches!(retrieval_mode, memory::RetrievalMode::Mission) && req.mission_id.is_none() {
        return Err(AppError::InvalidInput {
            field: "mission_id".to_string(),
            reason: "Mission mode requires mission_id".to_string(),
        });
    }

    let query = MemoryQuery {
        query_text: req.query_text,
        robot_id: req.robot_id.clone(),
        mission_id: req.mission_id.clone(),
        geo_filter,
        action_type: req.action_type.clone(),
        reward_range,
        max_results: req.limit.unwrap_or(10),
        retrieval_mode,
        ..Default::default()
    };

    let shared_memories = memory_guard.recall(&query).map_err(AppError::Internal)?;
    let raw_memories: Vec<Memory> = shared_memories.iter().map(|m| (**m).clone()).collect();
    let count = raw_memories.len();

    // Serialize memories to JSON for response
    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    state.log_event(
        &req.user_id,
        "ROBOTICS_SEARCH",
        &req.mode,
        &format!(
            "Retrieved {} robotics memories (robot={:?}, mission={:?})",
            count, req.robot_id, req.mission_id
        ),
    );

    Ok(Json(RetrieveResponse { memories, count }))
}

// =============================================================================
// GRAPH ENTITY HANDLERS
// =============================================================================

/// Request to add entity to graph
#[derive(Debug, Deserialize)]
pub struct AddEntityRequest {
    pub user_id: String,
    pub name: String,
    pub label: String,
    pub attributes: Option<HashMap<String, String>>,
}

/// Add entity to knowledge graph
pub async fn add_entity(
    State(state): State<AppState>,
    Json(req): Json<AddEntityRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entity(&req.name).map_validation_err("name")?;
    validation::validate_entity(&req.label).map_validation_err("label")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    let entity_label = match req.label.as_str() {
        "Person" => graph_memory::EntityLabel::Person,
        "Organization" => graph_memory::EntityLabel::Organization,
        "Location" => graph_memory::EntityLabel::Location,
        "Event" => graph_memory::EntityLabel::Event,
        "Concept" => graph_memory::EntityLabel::Concept,
        "Technology" => graph_memory::EntityLabel::Technology,
        "Product" => graph_memory::EntityLabel::Product,
        "Skill" => graph_memory::EntityLabel::Skill,
        "Date" => graph_memory::EntityLabel::Date,
        // Document, Code, Time mapped to Other for extensibility
        "Document" | "Code" | "Time" => graph_memory::EntityLabel::Other(req.label.clone()),
        _ => graph_memory::EntityLabel::Other(req.label.clone()),
    };

    // Create EntityNode with attributes
    let mut attributes = std::collections::HashMap::new();
    if let Some(attrs) = &req.attributes {
        for (key, value) in attrs {
            attributes.insert(
                key.clone(),
                serde_json::to_string(value).unwrap_or_default(),
            );
        }
    }

    let entity = graph_memory::EntityNode {
        uuid: uuid::Uuid::new_v4(),
        name: req.name.clone(),
        labels: vec![entity_label.clone()],
        created_at: chrono::Utc::now(),
        last_seen_at: chrono::Utc::now(),
        mention_count: 1,
        summary: String::new(),
        attributes,
        name_embedding: None,
        salience: 0.5,
        is_proper_noun: true,
    };

    graph_guard.add_entity(entity).map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "ADD_ENTITY",
        &req.name,
        &format!("Added entity '{}' with label {:?}", req.name, entity_label),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "entity": req.name,
        "label": format!("{:?}", entity_label)
    })))
}

/// Request to add relationship between entities
#[derive(Debug, Deserialize)]
pub struct AddRelationshipRequest {
    pub user_id: String,
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
    pub strength: Option<f32>,
}

/// Add relationship to knowledge graph
pub async fn add_relationship(
    State(state): State<AppState>,
    Json(req): Json<AddRelationshipRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;
    validation::validate_entity(&req.from_entity).map_validation_err("from_entity")?;
    validation::validate_entity(&req.to_entity).map_validation_err("to_entity")?;

    let graph = state
        .get_user_graph(&req.user_id)
        .map_err(AppError::Internal)?;
    let graph_guard = graph.write();

    let relation_type = match req.relation_type.as_str() {
        "RelatedTo" => graph_memory::RelationType::RelatedTo,
        "AssociatedWith" => graph_memory::RelationType::AssociatedWith,
        "PartOf" => graph_memory::RelationType::PartOf,
        "Contains" => graph_memory::RelationType::Contains,
        "OwnedBy" => graph_memory::RelationType::OwnedBy,
        "CreatedBy" => graph_memory::RelationType::CreatedBy,
        "DevelopedBy" => graph_memory::RelationType::DevelopedBy,
        "Uses" => graph_memory::RelationType::Uses,
        "WorksWith" => graph_memory::RelationType::WorksWith,
        "WorksAt" => graph_memory::RelationType::WorksAt,
        "EmployedBy" => graph_memory::RelationType::EmployedBy,
        "LocatedIn" => graph_memory::RelationType::LocatedIn,
        "LocatedAt" => graph_memory::RelationType::LocatedAt,
        "Causes" => graph_memory::RelationType::Causes,
        "ResultsIn" => graph_memory::RelationType::ResultsIn,
        "Learned" => graph_memory::RelationType::Learned,
        "Knows" => graph_memory::RelationType::Knows,
        "Teaches" => graph_memory::RelationType::Teaches,
        "CoRetrieved" => graph_memory::RelationType::CoRetrieved,
        // Map non-existent variants to Custom for extensibility
        _ => graph_memory::RelationType::Custom(req.relation_type.clone()),
    };

    let strength = req.strength.unwrap_or(0.5);

    // Need to find entity UUIDs by name first
    let from_entity = graph_guard
        .find_entity_by_name(&req.from_entity)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::InvalidInput {
            field: "from_entity".to_string(),
            reason: format!("Entity '{}' not found", req.from_entity),
        })?;
    let from_uuid = from_entity.uuid;

    let to_entity = graph_guard
        .find_entity_by_name(&req.to_entity)
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::InvalidInput {
            field: "to_entity".to_string(),
            reason: format!("Entity '{}' not found", req.to_entity),
        })?;
    let to_uuid = to_entity.uuid;

    let edge = graph_memory::RelationshipEdge {
        uuid: uuid::Uuid::new_v4(),
        from_entity: from_uuid,
        to_entity: to_uuid,
        relation_type: relation_type.clone(),
        strength,
        created_at: chrono::Utc::now(),
        valid_at: chrono::Utc::now(),
        invalidated_at: None,
        source_episode_id: None,
        context: String::new(),
        last_activated: chrono::Utc::now(),
        activation_count: 1,
        potentiated: false,
        tier: graph_memory::EdgeTier::L1Working,
    };

    graph_guard
        .add_relationship(edge)
        .map_err(AppError::Internal)?;

    state.log_event(
        &req.user_id,
        "ADD_RELATIONSHIP",
        &format!("{}->{}", req.from_entity, req.to_entity),
        &format!(
            "Added {:?} relationship from '{}' to '{}'",
            relation_type, req.from_entity, req.to_entity
        ),
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "from": req.from_entity,
        "to": req.to_entity,
        "type": format!("{:?}", relation_type),
        "strength": strength
    })))
}

/// Get uncompressed old memories (for manual compression review)
#[derive(Debug, Deserialize)]
pub struct GetUncompressedRequest {
    pub user_id: String,
    pub days_old: u32,
}

pub async fn get_uncompressed_old(
    State(state): State<AppState>,
    Json(req): Json<GetUncompressedRequest>,
) -> Result<Json<RetrieveResponse>, AppError> {
    validation::validate_user_id(&req.user_id).map_validation_err("user_id")?;

    let memory_sys = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    let memory_guard = memory_sys.read();

    let cutoff = chrono::Utc::now() - chrono::Duration::days(req.days_old as i64);
    let raw_memories = memory_guard
        .get_uncompressed_older_than(cutoff)
        .map_err(AppError::Internal)?;

    let count = raw_memories.len();

    // Serialize memories to JSON for response
    let memories: Vec<serde_json::Value> = raw_memories
        .into_iter()
        .filter_map(|m| serde_json::to_value(&m).ok())
        .collect();

    Ok(Json(RetrieveResponse { memories, count }))
}
