//! API Request/Response Types
//!
//! All HTTP API request and response structures for the shodh-memory server.
//! Extracted from main.rs for better organization.

use serde::{Deserialize, Serialize};

// =============================================================================
// HEALTH & INFRASTRUCTURE
// =============================================================================

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub memory_mb: f64,
    pub active_users: usize,
}

// =============================================================================
// AUDIT & EVENTS
// =============================================================================

/// Audit event for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: String,
    pub memory_id: String,
    pub details: String,
}

/// SSE Memory Event - lightweight event for real-time streaming
#[derive(Debug, Clone, Serialize)]
pub struct MemoryEvent {
    pub event_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user_id: String,
    pub memory_id: Option<String>,
    pub content_preview: Option<String>,
    pub memory_type: Option<String>,
    pub importance: Option<f32>,
    pub count: Option<usize>,
}

/// Context status from Claude Code (MCP server reports this via status line)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStatus {
    pub session_id: Option<String>,
    pub tokens_used: u64,
    pub tokens_budget: u64,
    pub percent_used: u8,
    pub current_task: Option<String>,
    pub model: Option<String>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

// =============================================================================
// RECORD/REMEMBER API
// =============================================================================

#[derive(Deserialize)]
pub struct RecordRequest {
    pub user_id: String,
    pub content: String,
    #[serde(default)]
    pub experience_type: Option<String>,
    #[serde(default)]
    pub entities: Vec<String>,
}

#[derive(Serialize)]
pub struct RecordResponse {
    pub id: String,
    pub created_at: String,
}

/// Simplified remember request - just content, auto-creates Experience
#[derive(Deserialize)]
pub struct RememberRequest {
    pub user_id: String,
    pub content: String,
    /// Optional memory type (default: auto-classified)
    #[serde(default)]
    pub memory_type: Option<String>,
    /// Optional tags/entities
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional override timestamp (ISO 8601)
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Optional emotional valence (-1.0 to 1.0)
    #[serde(default)]
    pub emotional_valence: Option<f32>,
    /// Optional emotional arousal (0.0 to 1.0)
    #[serde(default)]
    pub emotional_arousal: Option<f32>,
    /// Optional dominant emotion label
    #[serde(default)]
    pub emotion: Option<String>,
    /// Optional source type
    #[serde(default)]
    pub source_type: Option<String>,
    /// Optional credibility score (0.0 to 1.0)
    #[serde(default)]
    pub credibility: Option<f32>,
    /// Optional episode ID for grouping related memories
    #[serde(default)]
    pub episode_id: Option<String>,
    /// Optional sequence number within episode
    #[serde(default)]
    pub sequence_number: Option<u32>,
    /// Optional preceding memory ID (for temporal chains)
    #[serde(default)]
    pub preceding_memory_id: Option<String>,
}

/// Simplified remember response
#[derive(Serialize)]
pub struct RememberResponse {
    pub id: String,
    pub stored: bool,
}

// =============================================================================
// RECALL API
// =============================================================================

/// Simplified recall request - just query text
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub user_id: String,
    pub query: String,
    #[serde(default = "default_recall_limit")]
    pub limit: usize,
    /// Retrieval mode: "semantic", "associative", or "hybrid" (default)
    #[serde(default = "default_recall_mode")]
    pub mode: String,
}

pub fn default_recall_limit() -> usize {
    5
}

pub fn default_recall_mode() -> String {
    "hybrid".to_string()
}

/// Simplified recall response - returns just text snippets
#[derive(Serialize)]
pub struct RecallResponse {
    pub memories: Vec<RecallMemory>,
    pub count: usize,
    /// Retrieval statistics (for observability)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_stats: Option<crate::memory::types::RetrievalStats>,
    /// Related todos found via semantic search
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub todos: Vec<RecallTodo>,
    /// Number of todos found
    #[serde(skip_serializing_if = "Option::is_none")]
    pub todo_count: Option<usize>,
    /// Related semantic facts extracted from memories
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub facts: Vec<RecallFact>,
    /// Number of facts found
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fact_count: Option<usize>,
    /// Triggered reminders (future intentions that influenced retrieval)
    /// These are prospective tasks whose context matched the query
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triggered_reminders: Vec<RecallReminder>,
    /// Number of triggered reminders
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reminder_count: Option<usize>,
}

/// Reminder/prospective task returned in recall results
#[derive(Serialize)]
pub struct RecallReminder {
    pub id: String,
    pub content: String,
    /// Keywords that triggered this reminder (from OnContext trigger)
    pub keywords: Vec<String>,
    /// How the reminder was matched (keyword_match, semantic_match)
    pub match_type: String,
    pub priority: u8,
    pub created_at: String,
}

/// Semantic fact returned in recall results
#[derive(Serialize)]
pub struct RecallFact {
    pub id: String,
    pub fact: String,
    pub confidence: f32,
    pub support_count: usize,
    pub related_entities: Vec<String>,
}

/// Todo returned in recall results
#[derive(Serialize)]
pub struct RecallTodo {
    pub id: String,
    pub short_id: String,
    pub content: String,
    pub status: String,
    pub priority: String,
    pub project: Option<String>,
    pub due_date: Option<String>,
    pub score: f32,
}

#[derive(Serialize)]
pub struct RecallMemory {
    pub id: String,
    pub experience: RecallExperience,
    pub importance: f32,
    pub created_at: String,
    pub score: f32,
}

#[derive(Serialize)]
pub struct RecallExperience {
    pub content: String,
    pub memory_type: Option<String>,
    pub tags: Vec<String>,
}

// =============================================================================
// BATCH REMEMBER API
// =============================================================================

/// Batch remember request for bulk inserts
#[derive(Deserialize)]
pub struct BatchRememberRequest {
    pub user_id: String,
    pub memories: Vec<BatchMemoryItem>,
    #[serde(default)]
    pub options: BatchRememberOptions,
}

/// Options for batch remember operation
#[derive(Deserialize)]
pub struct BatchRememberOptions {
    #[serde(default = "default_true")]
    pub extract_entities: bool,
    #[serde(default = "default_true")]
    pub create_edges: bool,
}

fn default_true() -> bool {
    true
}

impl Default for BatchRememberOptions {
    fn default() -> Self {
        Self {
            extract_entities: true,
            create_edges: true,
        }
    }
}

#[derive(Deserialize, Clone)]
pub struct BatchMemoryItem {
    pub content: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub emotional_valence: Option<f32>,
    #[serde(default)]
    pub emotional_arousal: Option<f32>,
    #[serde(default)]
    pub emotion: Option<String>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub credibility: Option<f32>,
    #[serde(default)]
    pub episode_id: Option<String>,
    #[serde(default)]
    pub sequence_number: Option<u32>,
    #[serde(default)]
    pub preceding_memory_id: Option<String>,
}

/// Error detail for a single item in batch
#[derive(Serialize)]
pub struct BatchErrorItem {
    pub index: usize,
    pub error: String,
}

#[derive(Serialize)]
pub struct BatchRememberResponse {
    pub created: usize,
    pub failed: usize,
    pub memory_ids: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<BatchErrorItem>,
}

// =============================================================================
// UPSERT API
// =============================================================================

/// Upsert request - create or update memory with external linking
#[derive(Deserialize)]
pub struct UpsertRequest {
    pub user_id: String,
    pub external_id: String,
    pub content: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_change_type")]
    pub change_type: String,
    #[serde(default)]
    pub changed_by: Option<String>,
    #[serde(default)]
    pub change_reason: Option<String>,
}

fn default_change_type() -> String {
    "update".to_string()
}

#[derive(Serialize)]
pub struct UpsertResponse {
    pub id: String,
    pub external_id: String,
    pub created: bool,
    pub updated: bool,
    pub revision: usize,
}

// =============================================================================
// MEMORY HISTORY API
// =============================================================================

/// Request to get memory history (audit trail)
#[derive(Deserialize)]
pub struct MemoryHistoryRequest {
    pub user_id: String,
    pub memory_id: String,
}

/// Response with memory revision history
#[derive(Serialize)]
pub struct MemoryHistoryResponse {
    pub memory_id: String,
    pub external_id: Option<String>,
    pub current_content: String,
    pub revision_count: usize,
    pub revisions: Vec<MemoryRevisionInfo>,
}

#[derive(Serialize)]
pub struct MemoryRevisionInfo {
    pub revision: usize,
    pub content: String,
    pub changed_at: String,
    pub change_type: String,
    pub changed_by: Option<String>,
}

// =============================================================================
// RETRIEVAL RESPONSE
// =============================================================================

/// Response for list/search operations returning multiple memories
#[derive(Serialize)]
pub struct RetrieveResponse {
    pub memories: Vec<serde_json::Value>,
    pub count: usize,
}

// =============================================================================
// TRACKED RETRIEVAL & FEEDBACK
// =============================================================================

/// Request for tracked retrieval (returns tracking ID for later feedback)
#[derive(Debug, Deserialize)]
pub struct TrackedRetrieveRequest {
    pub user_id: String,
    pub query: String,
    #[serde(default = "default_recall_limit")]
    pub limit: usize,
}

/// Response with tracking ID for feedback
#[derive(Serialize)]
pub struct TrackedRetrieveResponse {
    pub tracking_id: String,
    pub ids: Vec<String>,
    pub memories: Vec<RecallMemory>,
}

/// Request to provide feedback on retrieval outcome
#[derive(Debug, Deserialize)]
pub struct ReinforceFeedbackRequest {
    pub user_id: String,
    /// Memory IDs to reinforce
    pub ids: Vec<String>,
    /// "helpful", "misleading", or "neutral"
    pub outcome: String,
}

// =============================================================================
// CONSOLIDATION
// =============================================================================

/// Request to trigger consolidation
#[derive(Debug, Deserialize)]
pub struct ConsolidateRequest {
    pub user_id: String,
    /// Minimum supporting memories required for fact extraction
    #[serde(default = "default_min_support")]
    pub min_support: usize,
    /// Minimum age in days for memories to be considered for consolidation
    #[serde(default = "default_min_age_days")]
    pub min_age_days: i64,
}

fn default_min_support() -> usize {
    2
}

fn default_min_age_days() -> i64 {
    1
}

/// Response from consolidation
#[derive(Serialize)]
pub struct ConsolidateResponse {
    pub memories_analyzed: usize,
    pub facts_extracted: usize,
    pub facts_reinforced: usize,
    pub fact_ids: Vec<String>,
    pub memories_replayed: usize,
    pub edges_strengthened: usize,
    pub memories_decayed: usize,
}

// =============================================================================
// INDEX MAINTENANCE
// =============================================================================

#[derive(Deserialize)]
pub struct VerifyIndexRequest {
    pub user_id: String,
}

#[derive(Deserialize)]
pub struct RepairIndexRequest {
    pub user_id: String,
}

#[derive(Serialize)]
pub struct RepairIndexResponse {
    pub success: bool,
    pub total_storage: usize,
    pub total_indexed: usize,
    pub repaired: usize,
    pub failed: usize,
    pub is_healthy: bool,
}

#[derive(Deserialize)]
pub struct CleanupCorruptedRequest {
    pub user_id: String,
}

#[derive(Serialize)]
pub struct CleanupCorruptedResponse {
    pub success: bool,
    pub deleted_count: usize,
}

#[derive(Deserialize)]
pub struct RebuildIndexRequest {
    pub user_id: String,
}

#[derive(Serialize)]
pub struct RebuildIndexResponse {
    pub success: bool,
    pub storage_count: usize,
    pub indexed_count: usize,
    pub is_healthy: bool,
}

// =============================================================================
// BACKUP & RESTORE
// =============================================================================

#[derive(Deserialize)]
pub struct CreateBackupRequest {
    pub user_id: String,
}

#[derive(Serialize)]
pub struct BackupResponse {
    pub success: bool,
    pub backup: Option<crate::backup::BackupMetadata>,
    pub message: String,
}

#[derive(Deserialize)]
pub struct ListBackupsRequest {
    pub user_id: String,
}

#[derive(Serialize)]
pub struct ListBackupsResponse {
    pub success: bool,
    pub backups: Vec<crate::backup::BackupMetadata>,
    pub count: usize,
}

#[derive(Deserialize)]
pub struct VerifyBackupRequest {
    pub user_id: String,
    pub backup_id: u32,
}

#[derive(Serialize)]
pub struct VerifyBackupResponse {
    pub success: bool,
    pub is_valid: bool,
    pub message: String,
}

#[derive(Deserialize)]
pub struct PurgeBackupsRequest {
    pub user_id: String,
    pub keep_count: usize,
}

#[derive(Serialize)]
pub struct PurgeBackupsResponse {
    pub success: bool,
    pub purged_count: usize,
}

// =============================================================================
// CONTEXT STATUS
// =============================================================================

#[derive(Deserialize)]
pub struct ContextStatusRequest {
    pub session_id: String,
    pub tokens_used: u64,
    pub tokens_limit: u64,
    #[serde(default)]
    pub current_task: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

// =============================================================================
// DELETE / FORGET APIs
// =============================================================================

#[derive(Deserialize)]
pub struct ForgetByAgeRequest {
    pub user_id: String,
    pub older_than_days: i64,
}

#[derive(Deserialize)]
pub struct ForgetByImportanceRequest {
    pub user_id: String,
    pub below_importance: f32,
    #[serde(default)]
    pub older_than_days: Option<i64>,
}

#[derive(Deserialize)]
pub struct ForgetByPatternRequest {
    pub user_id: String,
    pub pattern: String,
}

#[derive(Deserialize)]
pub struct BulkDeleteRequest {
    pub user_id: String,
    pub memory_ids: Vec<String>,
    /// Optional: require all deletes to succeed (default: false, best-effort)
    #[serde(default)]
    pub atomic: bool,
}

#[derive(Deserialize)]
pub struct ClearAllRequest {
    pub user_id: String,
    /// Must be "DELETE_ALL_MEMORIES" to confirm
    pub confirm: String,
}

// =============================================================================
// RECALL BY TAGS/DATE
// =============================================================================

#[derive(Deserialize)]
pub struct RecallByTagsRequest {
    pub user_id: String,
    pub tags: Vec<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct RecallByDateRequest {
    pub user_id: String,
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct ForgetByTagsRequest {
    pub user_id: String,
    pub tags: Vec<String>,
}

#[derive(Deserialize)]
pub struct ForgetByDateRequest {
    pub user_id: String,
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
}

// =============================================================================
// PATCH MEMORY
// =============================================================================

#[derive(Deserialize)]
pub struct PatchMemoryRequest {
    pub user_id: String,
    pub memory_id: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub importance: Option<f32>,
}

// =============================================================================
// MULTIMODAL SEARCH
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct MultiModalSearchRequest {
    pub user_id: String,
    pub query_text: String,
    pub mode: String, // "similarity", "temporal", "causal", "associative", "hybrid"
    pub limit: Option<usize>,
}

// =============================================================================
// ROBOTICS SEARCH
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct RoboticsSearchRequest {
    pub user_id: String,
    pub query_text: String,
    #[serde(default)]
    pub robot_id: Option<String>,
    #[serde(default)]
    pub mission_id: Option<String>,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub time_range_start: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub time_range_end: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub include_spatial: bool,
    #[serde(default)]
    pub include_mission: bool,
    #[serde(default)]
    pub include_actions: bool,
    #[serde(default = "default_robotics_limit")]
    pub limit: usize,
}

fn default_robotics_limit() -> usize {
    10
}

// =============================================================================
// GRAPH API
// =============================================================================

#[derive(Deserialize)]
pub struct GetUncompressedRequest {
    pub user_id: String,
    pub memory_id: String,
}

#[derive(Deserialize)]
pub struct AddEntityRequest {
    pub user_id: String,
    pub name: String,
    pub label: String,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Deserialize)]
pub struct AddRelationshipRequest {
    pub user_id: String,
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
    #[serde(default)]
    pub strength: Option<f32>,
    #[serde(default)]
    pub context: Option<String>,
}

#[derive(Deserialize)]
pub struct GetAllEntitiesRequest {
    pub user_id: String,
}

// =============================================================================
// VISUALIZATION
// =============================================================================

#[derive(Serialize)]
pub struct BrainStateResponse {
    pub user_id: String,
    pub neurons: Vec<MemoryNeuron>,
    pub connections: Vec<(String, String, f32)>,
    pub stats: BrainStats,
}

#[derive(Serialize)]
pub struct MemoryNeuron {
    pub id: String,
    pub content_preview: String,
    pub memory_type: String,
    pub importance: f32,
    pub activation: f32,
    pub tier: String,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct BrainStats {
    pub total_neurons: usize,
    pub total_connections: usize,
    pub avg_importance: f32,
    pub memory_by_type: std::collections::HashMap<String, usize>,
}

#[derive(Deserialize)]
pub struct BuildVisualizationRequest {
    pub user_id: String,
}
