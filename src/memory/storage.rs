//! Storage backend for the memory system

use anyhow::{anyhow, Context, Result};
use bincode;
use chrono::{DateTime, Utc};
use rocksdb::{
    ColumnFamily, ColumnFamilyDescriptor, IteratorMode, Options, WriteBatch, WriteOptions, DB,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::oplog::{self, OpRecord, OpRecordDraft};
use super::types::*;

/// Helper trait to safely iterate over RocksDB results with error logging.
/// Unlike `.flatten()` which silently ignores errors, this logs them.
trait LogErrors<T> {
    fn log_errors(self) -> impl Iterator<Item = T>;
}

impl<I, T, E> LogErrors<T> for I
where
    I: Iterator<Item = Result<T, E>>,
    E: std::fmt::Display,
{
    fn log_errors(self) -> impl Iterator<Item = T> {
        self.filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("RocksDB iterator error (continuing): {}", e);
                None
            }
        })
    }
}

/// Write mode for storage operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Sync writes - fsync() on every write (durable but slow: 2-10ms per write)
    /// Use for: shutdown, critical data, compliance requirements
    Sync,
    /// Async writes - no fsync(), data buffered in OS page cache (fast: <1ms per write)
    /// Use for: robotics, edge, high-throughput scenarios
    /// Data survives process crashes but NOT power loss before next fsync
    Async,
}

impl Default for WriteMode {
    fn default() -> Self {
        // Default to async for robotics-grade latency
        // Override with SHODH_WRITE_MODE=sync for durability-critical deployments
        match std::env::var("SHODH_WRITE_MODE") {
            Ok(mode) if mode.to_lowercase() == "sync" => WriteMode::Sync,
            _ => WriteMode::Async,
        }
    }
}

// ============================================================================
// BACKWARD-COMPATIBLE DESERIALIZATION
// Handles versioned format (SHO magic + checksum), current, and legacy formats
// ============================================================================

pub(crate) const STORAGE_MAGIC: &[u8; 3] = b"SHO";

use std::collections::HashMap;

/// Default experience type for legacy deserialization
fn default_legacy_experience_type() -> ExperienceType {
    ExperienceType::Observation
}

/// Minimal memory format - EXACT match for hex pattern: UUID (16 bytes) + varint string length + string
/// This is the simplest possible format with no extra fields. bincode doesn't support #[serde(default)]
/// so we can't have optional fields - the struct must match the binary data exactly.
#[derive(Deserialize)]
struct MinimalMemory {
    id: MemoryId,
    content: String,
}

/// Memory with experience type prefix - format: UUID + u8 (unknown) + u8 (experience_type) + String
/// Matches the failing entries that have 2 extra bytes before content:
/// - Byte 16: u8 value (seen as 28/0x1c)
/// - Byte 17: u8 experience type index (seen as 7 = Task)
/// - Byte 18+: varint length + string content
#[derive(Deserialize)]
struct MemoryWithTypePrefix {
    id: MemoryId,
    _unknown_field: u8,  // byte 16 - purpose unclear, maybe version?
    experience_type: u8, // byte 17 - experience type enum index
    content: String,     // byte 18+
}

/// Memory with 3-byte header - format: UUID + 3 bytes + raw content
/// Hex analysis shows: bytes 16-18 are header, byte 19+ is UTF-8 content starting with `[`
/// - Byte 16: 0x1c (28) - unknown
/// - Byte 17: 0x07 (7) - possibly experience type
/// - Byte 18: 0xa4 (164) - unknown (NOT valid UTF-8 start)
/// - Byte 19+: UTF-8 content
#[derive(Deserialize)]
struct MemoryWith3ByteHeader {
    id: MemoryId,
    _header1: u8,    // byte 16
    _header2: u8,    // byte 17
    _header3: u8,    // byte 18
    content: String, // byte 19+
}

impl MemoryWith3ByteHeader {
    fn into_memory(self) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: self.content,
            ..Default::default()
        };
        Memory::from_legacy(
            self.id,
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Check if data could plausibly be a MessagePack-encoded struct.
///
/// Memory structs serialize as maps in MessagePack. Without this guard,
/// arbitrary binary data (e.g. bincode, raw UTF-8) fed to `rmp_serde::from_slice`
/// can be misinterpreted: a byte that happens to be a str32/bin32/array32 format
/// code causes rmp to read subsequent bytes as a u32 length prefix, triggering
/// multi-gigabyte (or larger) allocation attempts that crash the process.
///
/// We check that the first byte is a valid MessagePack map format code (fixmap,
/// map16, or map32) since serde structs always encode as maps.
fn is_plausible_msgpack_struct(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }
    let first = data[0];
    // rmp_serde serializes structs as arrays (to_vec) or maps (to_vec_named).
    // Both are valid, so accept either container type.
    //
    // fixarray: 0x90-0x9f | array16: 0xdc | array32: 0xdd
    // fixmap:   0x80-0x8f | map16:   0xde | map32:   0xdf
    matches!(first, 0x80..=0x9f | 0xdc..=0xdf)
}

/// Maximum size for data eligible for MessagePack deserialization.
///
/// rmp_serde has no built-in allocation limits. Corrupted data with valid-looking
/// MessagePack headers can declare multi-exabyte string/bin lengths, causing the
/// allocator to abort the process. Since catch_unwind doesn't catch abort (Windows),
/// the only defense is refusing to attempt deserialization on data larger than a
/// sane maximum. Legitimate memory entries serialized as MessagePack are typically
/// under 100KB. 1MB provides generous headroom.
const MAX_MSGPACK_DESER_SIZE: usize = 1024 * 1024;

/// Safely attempt MessagePack deserialization with pre-flight validation.
///
/// Returns None (skip) if the data doesn't look like a MessagePack struct
/// or exceeds the safe size limit.
fn try_msgpack_deserialize<T: serde::de::DeserializeOwned>(
    data: &[u8],
) -> Option<Result<T, String>> {
    if !is_plausible_msgpack_struct(data) {
        return None;
    }
    if data.len() > MAX_MSGPACK_DESER_SIZE {
        return Some(Err(format!(
            "msgpack data too large ({} bytes, max {})",
            data.len(),
            MAX_MSGPACK_DESER_SIZE
        )));
    }
    match rmp_serde::from_slice::<T>(data) {
        Ok(val) => Some(Ok(val)),
        Err(e) => Some(Err(e.to_string())),
    }
}

/// Try to parse as raw bytes: UUID (16) + skip header bytes + raw UTF-8 string
/// This is a last-resort fallback for entries that don't match any standard format
fn try_raw_memory_parse(data: &[u8]) -> Option<Memory> {
    if data.len() < 20 {
        return None;
    }

    // Extract UUID from first 16 bytes
    let uuid_bytes: [u8; 16] = data[0..16].try_into().ok()?;
    let id = MemoryId(uuid::Uuid::from_bytes(uuid_bytes));

    // Try different header sizes (2, 3, 4 bytes) and find valid UTF-8
    for header_skip in [2, 3, 4, 5, 6] {
        let content_start = 16 + header_skip;
        if content_start >= data.len() {
            continue;
        }
        if let Ok(content) = std::str::from_utf8(&data[content_start..]) {
            if !content.is_empty()
                && content
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_graphic())
                    .unwrap_or(false)
            {
                let now = Utc::now();
                let experience = Experience {
                    experience_type: ExperienceType::Observation,
                    content: content.to_string(),
                    ..Default::default()
                };
                // Log only at debug level to avoid spam
                tracing::debug!(
                    "Recovered memory with raw parsing (header_skip={}, content_len={})",
                    header_skip,
                    content.len()
                );
                return Some(Memory::from_legacy(
                    id,
                    experience,
                    0.5,
                    0,
                    now,
                    now,
                    false,
                    MemoryTier::LongTerm,
                    Vec::new(),
                    1.0,
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    None,
                    None,
                    1,
                    Vec::new(),
                    Vec::new(),
                ));
            }
        }
    }
    None
}

impl MemoryWithTypePrefix {
    fn into_memory(self) -> Memory {
        let now = Utc::now();
        let exp_type = match self.experience_type {
            0 => ExperienceType::Observation,
            1 => ExperienceType::Decision,
            2 => ExperienceType::Learning,
            3 => ExperienceType::Error,
            4 => ExperienceType::Discovery,
            5 => ExperienceType::Pattern,
            6 => ExperienceType::Context,
            7 => ExperienceType::Task,
            8 => ExperienceType::CodeEdit,
            9 => ExperienceType::FileAccess,
            10 => ExperienceType::Search,
            11 => ExperienceType::Command,
            12 => ExperienceType::Conversation,
            _ => ExperienceType::Observation,
        };
        let experience = Experience {
            experience_type: exp_type,
            content: self.content,
            ..Default::default()
        };
        Memory::from_legacy(
            self.id,
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

impl MinimalMemory {
    fn into_memory(self) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: self.content,
            ..Default::default()
        };
        Memory::from_legacy(
            self.id,
            experience,
            0.5, // default importance
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Very simple legacy Memory - some early versions stored content directly without Experience wrapper
/// This matches the hex pattern: UUID (16 bytes) + varint string length + string content
#[derive(Deserialize)]
struct SimpleLegacyMemory {
    id: MemoryId,
    content: String, // Direct content field, no Experience wrapper
    #[serde(default)]
    importance: f32,
    #[serde(default)]
    access_count: u32,
    #[serde(default)]
    created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    last_accessed: Option<DateTime<Utc>>,
    #[serde(default)]
    compressed: bool,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    actor_id: Option<String>,
    #[serde(default)]
    temporal_relevance: f32,
    #[serde(default)]
    score: Option<f32>,
}

impl SimpleLegacyMemory {
    fn into_memory(self) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: self.content,
            ..Default::default()
        };
        Memory::from_legacy(
            self.id,
            experience,
            if self.importance > 0.0 {
                self.importance
            } else {
                0.5
            },
            self.access_count,
            self.created_at.unwrap_or(now),
            self.last_accessed.unwrap_or(now),
            self.compressed,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Legacy Experience type from v0.1.0 - EXACT match for bincode 1.x deserialization
/// Includes all fields that were in the original Experience struct in EXACT ORDER.
/// bincode 1.x serializes fields positionally, so order matters!
#[derive(Deserialize)]
struct LegacyExperienceV1 {
    // Core fields (always present)
    #[serde(default = "default_legacy_experience_type")]
    experience_type: ExperienceType,
    content: String,
    #[serde(default)]
    context: Option<RichContext>,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    metadata: HashMap<String, String>,
    #[serde(default)]
    embeddings: Option<Vec<f32>>,
    #[serde(default)]
    related_memories: Vec<MemoryId>,
    #[serde(default)]
    causal_chain: Vec<MemoryId>,
    #[serde(default)]
    outcomes: Vec<String>,
    // Robotics fields
    #[serde(default)]
    robot_id: Option<String>,
    #[serde(default)]
    mission_id: Option<String>,
    #[serde(default)]
    geo_location: Option<[f64; 3]>,
    #[serde(default)]
    local_position: Option<[f32; 3]>,
    #[serde(default)]
    heading: Option<f32>,
    #[serde(default)]
    action_type: Option<String>,
    #[serde(default)]
    reward: Option<f32>,
    #[serde(default)]
    sensor_data: HashMap<String, f64>,
    // Decision & learning fields
    #[serde(default)]
    decision_context: Option<HashMap<String, String>>,
    #[serde(default)]
    action_params: Option<HashMap<String, String>>,
    #[serde(default)]
    outcome_type: Option<String>,
    #[serde(default)]
    outcome_details: Option<String>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    alternatives_considered: Vec<String>,
    // Environmental context
    #[serde(default)]
    weather: Option<HashMap<String, String>>,
    #[serde(default)]
    terrain_type: Option<String>,
    #[serde(default)]
    lighting: Option<String>,
    #[serde(default)]
    nearby_agents: Vec<HashMap<String, String>>,
    // Failure & anomaly tracking
    #[serde(default)]
    is_failure: bool,
    #[serde(default)]
    is_anomaly: bool,
    #[serde(default)]
    severity: Option<String>,
    #[serde(default)]
    recovery_action: Option<String>,
    #[serde(default)]
    root_cause: Option<String>,
    // Learned patterns & predictions
    #[serde(default)]
    pattern_id: Option<String>,
    #[serde(default)]
    predicted_outcome: Option<String>,
    #[serde(default)]
    prediction_accurate: Option<bool>,
    #[serde(default)]
    tags: Vec<String>,
}

impl LegacyExperienceV1 {
    fn into_experience(self) -> Experience {
        Experience {
            experience_type: self.experience_type,
            content: self.content,
            context: self.context,
            entities: self.entities,
            metadata: self.metadata,
            embeddings: self.embeddings,
            image_embeddings: None,
            audio_embeddings: None,
            video_embeddings: None,
            media_refs: Vec::new(),
            related_memories: self.related_memories,
            causal_chain: self.causal_chain,
            outcomes: self.outcomes,
            robot_id: self.robot_id,
            mission_id: self.mission_id,
            geo_location: self.geo_location,
            local_position: self.local_position,
            heading: self.heading,
            action_type: self.action_type,
            reward: self.reward,
            sensor_data: self.sensor_data,
            decision_context: self.decision_context,
            action_params: self.action_params,
            outcome_type: self.outcome_type,
            outcome_details: self.outcome_details,
            confidence: self.confidence,
            alternatives_considered: self.alternatives_considered,
            weather: self.weather,
            terrain_type: self.terrain_type,
            lighting: self.lighting,
            nearby_agents: self.nearby_agents,
            is_failure: self.is_failure,
            is_anomaly: self.is_anomaly,
            severity: self.severity,
            recovery_action: self.recovery_action,
            root_cause: self.root_cause,
            pattern_id: self.pattern_id,
            predicted_outcome: self.predicted_outcome,
            prediction_accurate: self.prediction_accurate,
            tags: self.tags,
            temporal_refs: Vec::new(),
            ner_entities: Vec::new(),
            cooccurrence_pairs: Vec::new(),
            importance_override: None,
            // Pre-dates toponym resolution by several schema versions. Left
            // empty rather than resolved on read: resolution belongs at
            // remember time, and silently minting coordinates during a legacy
            // decode would make old records look like they always had them.
            toponyms: Vec::new(),
            // Same reasoning: a v0.1.0 record has no record of which of its
            // `entities` the caller asserted, and inventing that provenance on
            // read would let a legacy keyphrase enter the graph as a caller
            // claim.
            declared_entities: Vec::new(),
            // A v1 bincode record was written years before any write path
            // stamped an origin, so there is nothing to recover. `Unknown` is
            // the truth; guessing "this looks like it came from the API" and
            // persisting that on the migration rewrite would mint provenance
            // that never existed.
            origin: crate::memory::types::MemoryOrigin::Unknown,
        }
    }
}

/// Legacy v0.1.0 Memory with full Experience - for bincode 1.x deserialization
#[derive(Deserialize)]
struct LegacyMemoryV1Full {
    #[serde(rename = "memory_id")]
    id: MemoryId,
    experience: LegacyExperienceV1,
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
}

impl LegacyMemoryV1Full {
    fn into_memory(self) -> Memory {
        Memory::from_legacy(
            self.id,
            self.experience.into_experience(),
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Legacy v0.1.0 format - matches the initial release serialization
/// Uses LegacyExperienceV1 (no multimodal fields) because bincode is positional
#[derive(Deserialize)]
struct LegacyMemoryV1 {
    #[serde(rename = "memory_id")]
    id: MemoryId,
    experience: LegacyExperienceV1, // Must use legacy Experience - bincode ignores #[serde(default)]
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
}

impl LegacyMemoryV1 {
    /// Convert legacy v1 format to current Memory format
    fn into_memory(self) -> Memory {
        Memory::from_legacy(
            self.id,
            self.experience.into_experience(),
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }
}

/// Legacy v2 format - after cognitive extensions but before external linking
/// Has tier, entity_refs, activation but no external_id/version/history
#[derive(Deserialize)]
struct LegacyMemoryV2 {
    id: MemoryId,
    experience: LegacyExperienceV1, // bincode is positional - must match original fields
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    tier: MemoryTier,
    entity_refs: Vec<EntityRef>,
    activation: f32,
    last_retrieval_id: Option<uuid::Uuid>,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
}

impl LegacyMemoryV2 {
    /// Convert legacy v2 format to current Memory format
    fn into_memory(self) -> Memory {
        Memory::from_legacy(
            self.id,
            self.experience.into_experience(),
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            self.tier,
            self.entity_refs,
            self.activation,
            self.last_retrieval_id,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            None,       // external_id not in v2
            1,          // version
            Vec::new(), // history not in v2
            Vec::new(), // related_todo_ids not in v2
        )
    }
}

/// Deserialize memory with multi-version fallback for backwards compatibility
///
/// Tries formats in order from newest to oldest:
/// 1. Current format (with external linking, todos)
/// 2. Legacy v2 (cognitive extensions, no external linking)
/// 3. Legacy v1 (original v0.1.0 format)
///
/// Returns (Memory, needs_migration) where needs_migration=true means the data
/// was in a legacy format and should be re-written for future performance.
/// Postcard defaults for every trailing `MemoryFlat` field added after the
/// postcard cutover (#192), in field order: `toponyms: Vec<Toponym>` (empty =
/// varint `0x00`), then `declared_entities: Vec<String>` (empty = varint
/// `0x00`), then `origin: MemoryOrigin` (`Unknown` = variant index varint
/// `0x00`).
///
/// `parent_id` is NOT listed: it was added in January, before the April
/// postcard cutover, so every postcard-era record already carries it.
///
/// `decode_raw_compat` appends these one at a time, so a record missing any
/// suffix of these fields decodes (postcard has no `#[serde(default)]` EOF
/// tolerance). Keep in sync with any new trailing field — appending a field to
/// `MemoryFlat` means appending its postcard default here, in the SAME ORDER as
/// the struct declares it. The order is a contract: `toponyms`, then
/// `declared_entities`, then `origin` — the order `MemoryFlat` declares them.
/// Bytes appended in the wrong order default the wrong fields.
///
/// All three defaults happen to encode as the same byte (`0x00`: an empty
/// `Vec` length, an empty `Vec` length, and `MemoryOrigin::Unknown`'s variant
/// index), so the SUFFIX is order-agnostic and a mis-ordered suffix would not
/// be caught by inspecting it. What is NOT order-agnostic is the declaration
/// order in `MemoryFlat`, which decides which stored bytes are read as which
/// field. `declared_entities` shipped first and live records already carry it,
/// so `origin` goes after it — putting `origin` first would make every such
/// record read `declared_entities`' length varint as the origin discriminant.
///
/// Note that this mechanism ONLY works for fields appended at the end of
/// `MemoryFlat`. A field added to `Experience` instead lands mid-payload, where
/// an old record decodes to silently wrong values rather than failing; that is
/// why `Experience::toponyms` is `#[serde(skip)]` and carried at the
/// `MemoryFlat` tail. `NerEntityRecord::fine_label` is what happens when that
/// rule is broken — see [`decode_postcard_memory`].
const MEMORY_DEFAULT_SUFFIX: &[u8] = &[0x00, 0x00, 0x00];

/// Decode a SHO v2 (postcard) payload, tolerating both compatible-change classes
/// the stored format has accumulated.
///
/// 1. **Missing tail fields** — repaired by appending `MEMORY_DEFAULT_SUFFIX`
///    one default at a time (see [`crate::serialization::decode_raw_compat`]).
/// 2. **A mid-payload field added to a nested struct** — NOT repairable by any
///    suffix, because the missing bytes are in the middle. `NerEntityRecord`
///    gained `fine_label` in July 2026, and `Experience::ner_entities` sits
///    ahead of `cooccurrence_pairs`, `importance_override` and the whole of
///    `MemoryFlat` after `experience`. Every memory written before that commit
///    with a non-empty `ner_entities` therefore desynchronises the decoder by
///    one byte per NER record. On a live 19,394-record store that is ~5% of
///    memories, surfacing as `Found an Option discriminant that wasn't 0 or 1`,
///    `Tried to parse invalid utf-8`, or a spurious end-of-buffer — whichever
///    the shifted bytes happened to hit.
///
/// The second class is recovered by re-running the decode under
/// [`NerWireGeneration::PreFineLabel`]. The retry is attempted only after the
/// current layout has failed, so a record written today never pays for it, and
/// the error reported on total failure is the CURRENT-format one — the honest
/// answer to "why can this store not be read by this build".
///
/// A record recovered through the retry is reported as needing migration, so
/// [`MemoryStorage::get`] rewrites it in the current format and no later read
/// has to guess again.
fn decode_postcard_memory(payload: &[u8]) -> Result<(Memory, bool)> {
    let current_err =
        match crate::serialization::decode_raw_compat::<Memory>(payload, MEMORY_DEFAULT_SUFFIX) {
            Ok((memory, defaulted)) => return Ok((memory, defaulted)),
            Err(e) => e,
        };

    let _generation = NerWireGeneration::PreFineLabel.enter();
    match crate::serialization::decode_raw_compat::<Memory>(payload, MEMORY_DEFAULT_SUFFIX) {
        Ok((memory, _)) => Ok((memory, true)),
        Err(_) => Err(current_err),
    }
}

/// Decode a stored memory value.
///
/// # A failed checksum is an error, not a hint to try harder
///
/// The SHO envelope carries a CRC32 per record. It used to be decorative: on a
/// mismatch the envelope reader returned `None`, and `None` here meant "no
/// envelope, try the legacy chain" — so a record whose checksum FAILED was
/// handed, magic bytes and all, to [`deserialize_with_fallback`], whose last
/// branch ([`try_raw_memory_parse`]) fabricates a `Memory` out of any
/// twenty-one bytes it is given. The one mechanism able to detect real
/// corruption had its failure signal laundered into a decode attempt by the
/// path most likely to invent a plausible wrong answer.
///
/// [`crate::serialization::read_sho_envelope`] now separates the two states,
/// and they are treated as opposites here:
///
/// - `Absent` — a pre-cutover record. Legitimately falls through to the legacy
///   chain; every record written before the envelope existed depends on that,
///   which is why the distinction had to be drawn rather than the fallback
///   simply removed.
/// - `ChecksumMismatch` / `Truncated` — damaged bytes. A hard error that never
///   reaches a fallback decoder.
///
/// An envelope carrying a version this build does not know is an error too: its
/// payload is by definition in a format the legacy chain predates, so running
/// seventeen legacy branches over it can only produce a fabrication.
fn deserialize_memory(data: &[u8]) -> Result<(Memory, bool)> {
    use crate::serialization::{ShoEnvelope, SHO_VERSION_BINCODE2, SHO_VERSION_POSTCARD};

    match crate::serialization::read_sho_envelope(data) {
        ShoEnvelope::Valid {
            version: SHO_VERSION_POSTCARD,
            payload,
        } => {
            // Current format: postcard, tolerating records written before
            // the trailing fields listed in MEMORY_DEFAULT_SUFFIX existed,
            // and before `NerEntityRecord::fine_label` existed.
            let (memory, defaulted) = decode_postcard_memory(payload)
                .map_err(|e| anyhow!("SHO v2 postcard decode failed: {e}"))?;
            // `defaulted` marks the record for rewrite in the current schema,
            // the same signal the legacy-format branches return.
            Ok((memory, defaulted))
        }
        ShoEnvelope::Valid {
            version: SHO_VERSION_BINCODE2,
            payload,
        } => {
            // Legacy SHO v1: bincode 2.x — decode and mark for migration
            let (memory, _): (Memory, _) =
                bincode::serde::decode_from_slice(payload, crate::bincode_safe_config())
                    .map_err(|e| anyhow!("SHO v1 bincode decode failed: {e}"))?;
            Ok((memory, true))
        }
        ShoEnvelope::Valid { version, .. } => Err(anyhow!(
            "SHO envelope version {version} is not readable by this build ({} \
             bytes); refusing to guess at it with the legacy decoders",
            data.len()
        )),
        ShoEnvelope::ChecksumMismatch { stored, computed } => Err(anyhow!(
            "SHO envelope checksum mismatch (stored {stored:08x}, computed \
             {computed:08x}, {} bytes): the bytes on disk are not the bytes that \
             were written. Refusing to decode — a fallback decode of corrupted \
             bytes returns a fabricated memory, not a recovered one",
            data.len()
        )),
        ShoEnvelope::Truncated { len } => Err(anyhow!(
            "SHO envelope truncated to {len} bytes, below the minimum envelope \
             size; refusing to decode"
        )),
        ShoEnvelope::Absent => {
            // No SHO header — legacy format (raw bincode/msgpack)
            deserialize_with_fallback(data)
                .map_err(|e| anyhow!("legacy (no SHO header) decode failed: {e}"))
        }
    }
}

/// Decode a stored memory value and verify it is the record its key claims.
///
/// # Why this is on the read path
///
/// Corruption on the memory side rarely announces itself. Every shape in the
/// legacy fallback chain derives the id from the VALUE; the writer derives the
/// key from the RECORD. A fabrication therefore cannot agree with its own key
/// — `try_raw_memory_parse` reads the first sixteen bytes of the value as a
/// UUID, which for an enveloped record is `SHO` + version + payload, and for
/// arbitrary garbage is arbitrary garbage. The check has no false positive by
/// construction: nothing in this codebase stores a memory under a key other
/// than its own id.
///
/// The cost is a sixteen-byte comparison against a value that has already cost
/// a RocksDB read and a full postcard decode — far below the noise floor of
/// the work done to produce the thing being checked. There is no perf argument
/// for serving a record that disagrees with its own key, and serving one is
/// worse than serving nothing: the caller cannot tell it apart from fact.
fn deserialize_memory_checked(key: &[u8], data: &[u8]) -> Result<(Memory, bool)> {
    let (memory, needs_migration) = deserialize_memory(data)?;
    if memory.id.0.as_bytes() != key {
        tracing::warn!(
            decoded_id = %memory.id.0,
            key = %hex::encode(key),
            bytes = data.len(),
            "decoded memory id disagrees with the key it was stored under — \
             treating it as corruption, not as a memory"
        );
        return Err(anyhow!(
            "decoded memory id {} does not match its key {} ({} bytes): the \
             record decoded into something the writer could not have written",
            memory.id.0,
            hex::encode(key),
            data.len()
        ));
    }
    Ok((memory, needs_migration))
}

/// Public wrapper around the full legacy fallback chain, used by the migration module.
///
/// Tries SHO v2 (postcard), SHO v1 (bincode 2.x), then the 17-path legacy
/// fallback for raw bincode/msgpack data. Returns just the Memory on success.
pub fn deserialize_memory_for_migration(data: &[u8]) -> Result<Memory> {
    deserialize_memory(data).map(|(m, _)| m)
}

/// Legacy MemoryFlat for bincode 2.x data written BEFORE multimodal Experience fields
/// This matches the format at MIF commit (dee4b03) - has tier/entity_refs but Experience without multimodal
#[derive(Deserialize)]
struct LegacyMemoryFlatV2 {
    id: MemoryId,
    experience: LegacyExperienceV1, // No multimodal fields
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    tier: MemoryTier,
    entity_refs: Vec<EntityRef>,
    activation: f32,
    last_retrieval_id: Option<uuid::Uuid>,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
    external_id: Option<String>,
    version: u32,
    history: Vec<MemoryRevision>,
    #[serde(default)]
    related_todo_ids: Vec<TodoId>,
}

impl LegacyMemoryFlatV2 {
    fn into_memory(self) -> Memory {
        Memory::from_legacy(
            self.id,
            self.experience.into_experience(),
            self.importance,
            self.access_count,
            self.created_at,
            self.last_accessed,
            self.compressed,
            self.tier,
            self.entity_refs,
            self.activation,
            self.last_retrieval_id,
            self.agent_id,
            self.run_id,
            self.actor_id,
            self.temporal_relevance,
            self.score,
            self.external_id,
            self.version,
            self.history,
            self.related_todo_ids,
        )
    }
}

/// Try deserializing with multiple format fallbacks
/// Supports bincode 2.x (current), MessagePack, and bincode 1.x (legacy) wire formats
///
/// Returns (Memory, is_legacy) where is_legacy=true means the data was in an old format
/// and should be re-written to current format for future performance.
fn deserialize_with_fallback(data: &[u8]) -> Result<(Memory, bool)> {
    fn record_branch(branch: &str) {
        crate::metrics::LEGACY_FALLBACK_BRANCH_TOTAL
            .with_label_values(&[branch])
            .inc();
    }

    // Try current format first (bincode 2.x with current Memory/Experience)
    // This is the hot path — avoid any allocations before this check.
    match bincode::serde::decode_from_slice::<Memory, _>(data, crate::bincode_safe_config()) {
        Ok((memory, _)) => Ok((memory, false)), // Current format, no migration needed
        Err(e) => {
            // Current format failed — enter fallback chain.
            // Bincode 1.x paths use with_limit() to cap allocations.
            // MessagePack paths use try_msgpack_deserialize() which scans for
            // oversized length prefixes before calling rmp_serde::from_slice.
            deserialize_legacy_fallback(data, e, record_branch)
        }
    }
}

/// Fallback deserialization chain for legacy memory formats.
///
/// Separated from `deserialize_with_fallback` so the hot path (current format)
/// stays allocation-free and the compiler can inline/optimize it independently.
fn deserialize_legacy_fallback(
    data: &[u8],
    first_error: bincode::error::DecodeError,
    record_branch: fn(&str),
) -> Result<(Memory, bool)> {
    // Log detailed errors for first entry only to help debug format issues
    static DEBUG_ENTRY_LOGGED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    let is_first_failure = !DEBUG_ENTRY_LOGGED.load(std::sync::atomic::Ordering::Relaxed);

    // Collect all errors for debugging
    let mut errors: Vec<(&str, String)> = Vec::new();
    errors.push(("bincode2 Memory", first_error.to_string()));

    // Try bincode 2.x MINIMAL format (just UUID + content string)
    // This matches the hex pattern: 16-byte UUID + varint length + string bytes
    match bincode::serde::decode_from_slice::<MinimalMemory, _>(data, crate::bincode_safe_config())
    {
        Ok((minimal, _)) => {
            tracing::debug!("Migrated memory from bincode 2.x minimal format");
            record_branch("bincode2_minimal");
            return Ok((minimal.into_memory(), true));
        }
        Err(e) => errors.push(("bincode2 MinimalMemory", e.to_string())),
    }

    // Try bincode 2.x with type prefix: UUID + u8 + u8 (experience_type) + String
    // Matches entries with 2 extra bytes before content (byte 16=unknown, byte 17=exp_type)
    match bincode::serde::decode_from_slice::<MemoryWithTypePrefix, _>(
        data,
        crate::bincode_safe_config(),
    ) {
        Ok((typed, _)) => {
            tracing::debug!("Migrated memory from bincode 2.x with type prefix");
            record_branch("bincode2_type_prefix");
            return Ok((typed.into_memory(), true));
        }
        Err(e) => errors.push(("bincode2 MemoryWithTypePrefix", e.to_string())),
    }

    // Try bincode 2.x with OLD Experience (before multimodal fields were added)
    match bincode::serde::decode_from_slice::<LegacyMemoryFlatV2, _>(
        data,
        crate::bincode_safe_config(),
    ) {
        Ok((legacy, _)) => {
            tracing::debug!("Migrated memory from bincode 2.x pre-multimodal format");
            record_branch("bincode2_legacy_flat_v2");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode2 LegacyMemoryFlatV2", e.to_string())),
    }

    // Bincode 1.x with size limit — prevents OOM from corrupted u64 length prefixes.
    // Bincode 1.x reads string/vec lengths as u64; corrupted data can declare exabyte allocations.
    // Must keep allow_trailing_bytes() to match bincode1::deserialize() default behavior.
    use bincode1::Options;
    // bincode1::deserialize() uses FixintEncoding internally, so we must match that.
    let bincode1_safe = bincode1::options()
        .with_fixint_encoding()
        .with_limit(data.len() as u64 + 1024)
        .allow_trailing_bytes();

    // Try bincode 1.x with LegacyMemoryV1 (v0.1.0 format)
    match bincode1_safe.deserialize::<LegacyMemoryV1>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x v0.1.0 format");
            record_branch("bincode1_legacy_v1");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 LegacyMemoryV1", e.to_string())),
    }

    // Try bincode 1.x MINIMAL format (just UUID + content)
    match bincode1_safe.deserialize::<MinimalMemory>(data) {
        Ok(minimal) => {
            tracing::debug!("Migrated memory from bincode 1.x minimal format");
            record_branch("bincode1_minimal");
            return Ok((minimal.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 MinimalMemory", e.to_string())),
    }

    // Try bincode 1.x with SIMPLE legacy format (content as direct field, no Experience wrapper)
    match bincode1_safe.deserialize::<SimpleLegacyMemory>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x simple format");
            record_branch("bincode1_simple");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 SimpleLegacyMemory", e.to_string())),
    }

    // Try bincode 1.x with fixint encoding (u64 lengths instead of varint)
    let fixint_config = bincode1::options()
        .with_fixint_encoding()
        .with_limit(data.len() as u64 + 1024)
        .allow_trailing_bytes();

    // Try bincode 1.x fixint MinimalMemory
    match fixint_config.deserialize::<MinimalMemory>(data) {
        Ok(minimal) => {
            tracing::debug!("Migrated memory from bincode 1.x fixint minimal format");
            record_branch("bincode1_fixint_minimal");
            return Ok((minimal.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 fixint MinimalMemory", e.to_string())),
    }

    match fixint_config.deserialize::<SimpleLegacyMemory>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x fixint simple format");
            record_branch("bincode1_fixint_simple");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 fixint SimpleLegacyMemory", e.to_string())),
    }

    // Try MessagePack minimal format (guarded: non-msgpack data can trigger OOM)
    match try_msgpack_deserialize::<MinimalMemory>(data) {
        Some(Ok(minimal)) => {
            tracing::debug!("Migrated memory from MessagePack minimal format");
            record_branch("msgpack_minimal");
            return Ok((minimal.into_memory(), true));
        }
        Some(Err(e)) => errors.push(("msgpack MinimalMemory", e)),
        None => errors.push(("msgpack MinimalMemory", "not msgpack format".to_string())),
    }

    // Try MessagePack format (rmp-serde) - self-describing format
    match try_msgpack_deserialize::<SimpleLegacyMemory>(data) {
        Some(Ok(legacy)) => {
            tracing::debug!("Migrated memory from MessagePack simple format");
            record_branch("msgpack_simple");
            return Ok((legacy.into_memory(), true));
        }
        Some(Err(e)) => errors.push(("msgpack SimpleLegacyMemory", e)),
        None => errors.push((
            "msgpack SimpleLegacyMemory",
            "not msgpack format".to_string(),
        )),
    }

    // Try bincode 1.x format with original Experience (no multimodal fields)
    match bincode1_safe.deserialize::<LegacyMemoryV1Full>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x v1 full format");
            record_branch("bincode1_legacy_v1_full");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 LegacyMemoryV1Full", e.to_string())),
    }

    // Try bincode 1.x with fixint encoding for full legacy format
    match fixint_config.deserialize::<LegacyMemoryV1Full>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x fixint v1 full format");
            record_branch("bincode1_fixint_legacy_v1_full");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 fixint LegacyMemoryV1Full", e.to_string())),
    }

    // Try MessagePack with full legacy format (guarded: non-msgpack data can trigger OOM)
    match try_msgpack_deserialize::<LegacyMemoryV1Full>(data) {
        Some(Ok(legacy)) => {
            tracing::debug!("Migrated memory from MessagePack v1 full format");
            record_branch("msgpack_legacy_v1_full");
            return Ok((legacy.into_memory(), true));
        }
        Some(Err(e)) => errors.push(("msgpack LegacyMemoryV1Full", e)),
        None => errors.push((
            "msgpack LegacyMemoryV1Full",
            "not msgpack format".to_string(),
        )),
    }

    // Try bincode 1.x format (used in versions prior to bincode 2.0 migration)
    if let Ok(legacy) = bincode1_safe.deserialize::<LegacyMemoryV1>(data) {
        tracing::debug!("Migrated memory from bincode 1.x format");
        record_branch("bincode1_legacy_v1_repeat");
        return Ok((legacy.into_memory(), true));
    }

    // Try legacy v2 format with bincode 1.x (cognitive extensions era)
    match bincode1_safe.deserialize::<LegacyMemoryV2>(data) {
        Ok(legacy) => {
            tracing::debug!("Migrated memory from bincode 1.x v2 format");
            record_branch("bincode1_legacy_v2");
            return Ok((legacy.into_memory(), true));
        }
        Err(e) => errors.push(("bincode1 LegacyMemoryV2", e.to_string())),
    }

    // Try bincode 2.x with 3-byte header: UUID + 3 bytes + String
    // Hex analysis shows content starts at byte 19 (byte 18 is 0xa4, not valid UTF-8 start)
    match bincode::serde::decode_from_slice::<MemoryWith3ByteHeader, _>(
        data,
        crate::bincode_safe_config(),
    ) {
        Ok((mem, _)) => {
            tracing::debug!("Migrated memory from bincode 2.x with 3-byte header");
            record_branch("bincode2_3byte_header");
            return Ok((mem.into_memory(), true));
        }
        Err(e) => errors.push(("bincode2 MemoryWith3ByteHeader", e.to_string())),
    }

    // LAST RESORT: Try raw byte parsing with different header skip sizes
    // This handles non-standard formats by finding where valid UTF-8 content starts
    if let Some(memory) = try_raw_memory_parse(data) {
        record_branch("raw_parse");
        return Ok((memory, true));
    }
    errors.push(("raw parse", "no valid UTF-8 content found".to_string()));

    // Log debug info on first failure only (at debug level to reduce noise)
    if is_first_failure {
        DEBUG_ENTRY_LOGGED.store(true, std::sync::atomic::Ordering::Relaxed);

        let hex_preview: String = data
            .iter()
            .take(32)
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::debug!(
            "Unknown memory format ({} bytes): {}...",
            data.len(),
            hex_preview
        );
    }

    // All formats failed
    record_branch("decode_failed");
    Err(anyhow!(
        "Failed to deserialize memory: incompatible format ({} bytes)",
        data.len()
    ))
}

/// Simple CRC32 implementation (IEEE polynomial)
pub(crate) fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Column family name for secondary indices (tags, types, timestamps, etc.)
const CF_INDEX: &str = "memory_index";

/// Column family for the append-only agent-traceability operation log
/// (`memory::oplog::OpRecord`, spec `docs/superpowers/specs/2026-07-30-agent-traceability-design.md`).
///
/// `pub` (unlike [`CF_INDEX`]) so other openers of the live storage path —
/// notably `migration.rs`'s hardcoded-CF-list opener — can reference the
/// constant instead of duplicating the string literal (audit
/// `2026-07-30-traceability-slice1-audit.md` Finding A / amendment 3).
///
/// Keyspace within this CF:
/// - `op:{session_id}:{seq:016x}` — one entry per [`super::oplog::OpRecord`],
///   `seq` zero-padded to 16 **hex** digits (`u64`) so lexicographic order
///   equals numeric order across the *entire* `u64` range. Decimal padding
///   would break above 10^16 (`u64::MAX` is 20 decimal digits but only 16
///   hex digits), so hex is exact at identical cost.
/// - `head:{session_id}` — cached `(last_seq, last_hash, last_ts)` for O(1)
///   append without scanning the session's full history.
/// - `incomplete:{session_id}` — integrity flag set by
///   [`MemoryStorage::oplog_mark_incomplete`]; never touches `op:`/`head:` keys.
///
/// `session_id` is validated (`crate::validation::validate_session_id`) to
/// reject `:` before it ever reaches a key, so these three prefixes cannot
/// collide with each other or with a foreign session's `op:` keys.
pub const CF_OPLOG: &str = "oplog";

/// Key prefix for oplog records: full key is `op:{session_id}:{seq:016x}`
/// (hex, not decimal — see [`CF_OPLOG`]'s doc comment for why).
const OPLOG_RECORD_PREFIX: &str = "op:";

/// Key prefix for the cached per-session head pointer: `head:{session_id}`.
const OPLOG_HEAD_PREFIX: &str = "head:";

/// Key prefix for the per-session integrity flag: `incomplete:{session_id}`.
const OPLOG_INCOMPLETE_PREFIX: &str = "incomplete:";

/// Cached per-session head pointer, stored under `head:{session_id}` in
/// [`CF_OPLOG`]. `last_ts` mirrors the head record's `ts` so
/// [`MemoryStorage::oplog_sessions`] can order sessions "newest-first by
/// head write time" without decoding each session's full record history.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OplogHead {
    last_seq: u64,
    last_hash: String,
    last_ts: DateTime<Utc>,
}

/// Maximum number of failed writes to buffer for retry.
/// Small enough to bound memory usage (~100 memories × ~2KB ≈ 200KB)
/// but large enough to absorb transient RocksDB contention bursts.
const WRITE_RETRY_BUFFER_CAPACITY: usize = 100;

/// Storage engine for long-term memory persistence
///
/// Uses a single RocksDB instance with 2 column families:
/// - default: main memory data (also shared by LearningHistoryStore, TemporalFactStore, etc. via key prefixes)
/// - `memory_index`: secondary indices for tag/type/timestamp queries
pub struct MemoryStorage {
    db: Arc<DB>,
    /// Base storage path for all memory data
    storage_path: PathBuf,
    /// Write mode (sync vs async) - affects latency vs durability tradeoff
    write_mode: WriteMode,
    /// Bounded retry buffer for failed writes.
    /// Memories that fail to store (transient RocksDB lock contention, disk pressure)
    /// are queued here and retried on the next maintenance tick or successful store.
    write_retry_buffer: parking_lot::Mutex<std::collections::VecDeque<Memory>>,
    /// Counter of total write failures (for /api/health metrics)
    write_failure_count: std::sync::atomic::AtomicU64,
    /// Serializes oplog head-read -> `WriteBatch`-write so concurrent appends
    /// to the same session cannot race and produce duplicate `seq` values or
    /// forked chains. Storage methods are NOT otherwise externally serialized
    /// — handlers take `.read()` guards on `MemorySystem` (audit
    /// `2026-07-30-traceability-slice1-audit.md` Finding G) — so this lock is
    /// the only thing making "seq = head.last_seq + 1" atomic. Held only
    /// across the append path (read head, build record, write batch); never
    /// taken by reads. Capture is low-frequency relative to reads, so a
    /// single per-storage (i.e. per-user) mutex is acceptable for v1.
    oplog_append_lock: parking_lot::Mutex<()>,
    /// Serializes read-modify-write updates of a memory record (see
    /// [`Self::modify`]). Same reasoning as `oplog_append_lock`: handlers hold
    /// only `.read()` guards on `MemorySystem`, so two link/parent/metadata
    /// writes to the same memory genuinely run at the same time, and a
    /// `get` -> mutate -> `update` sequence loses whichever write read first.
    ///
    /// A process-local mutex is sufficient rather than a shortcut: RocksDB
    /// takes an exclusive file lock, so exactly one process ever has this DB
    /// open, and every writer to it goes through this struct.
    record_mutation_lock: parking_lot::Mutex<()>,
}

impl MemoryStorage {
    /// CF accessor for the memory_index column family
    fn index_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_INDEX)
            .expect("memory_index CF must exist")
    }

    /// CF accessor for the agent-traceability oplog column family.
    fn oplog_cf(&self) -> &ColumnFamily {
        self.db.cf_handle(CF_OPLOG).expect("oplog CF must exist")
    }

    /// Create a new memory storage.
    ///
    /// If `shared_cache` is provided, all block-cache reads are charged against
    /// the shared LRU cache (recommended for multi-tenant server mode). When
    /// `None`, a small per-instance cache is created (standalone / test use).
    pub fn new(path: &Path, shared_cache: Option<&rocksdb::Cache>) -> Result<Self> {
        use crate::constants::ROCKSDB_MEMORY_WRITE_BUFFER_BYTES;

        // Create directories if they don't exist
        let storage_path = path.join("storage");
        std::fs::create_dir_all(&storage_path)?;

        // Configure RocksDB options for PRODUCTION durability + performance
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // ========================================================================
        // DURABILITY SETTINGS - Critical for data persistence across restarts
        // ========================================================================
        //
        // RocksDB data flow: Write → WAL → Memtable → SST files
        // Without proper sync, data in memtable can be lost on crash/restart
        //
        // Our approach: Sync WAL on every write (most durable option)
        // This ensures data survives even if process crashes before memtable flush
        // ========================================================================

        // WAL stays in default location (same as data dir) - avoids corruption issues
        opts.set_manual_wal_flush(false); // Auto-flush WAL entries

        // Write performance — sized for edge deployment (tune up via env for heavy workloads)
        opts.set_max_write_buffer_number(2);
        opts.set_write_buffer_size(ROCKSDB_MEMORY_WRITE_BUFFER_BYTES);
        opts.set_level_zero_file_num_compaction_trigger(4);
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB SST files
        opts.set_max_bytes_for_level_base(256 * 1024 * 1024); // 256MB L1
        opts.set_max_background_jobs(4);
        opts.set_level_compaction_dynamic_level_bytes(true);

        // Read performance — shared block cache for multi-tenant, small local for standalone
        use rocksdb::{BlockBasedOptions, Cache};
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false); // 10 bits/key = ~1% FPR
        let local_cache;
        let cache = match shared_cache {
            Some(c) => c,
            None => {
                local_cache = Cache::new_lru_cache(16 * 1024 * 1024); // 16MB standalone
                &local_cache
            }
        };
        block_opts.set_block_cache(cache);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true); // Pin L0 for fast reads
        opts.set_block_based_table_factory(&block_opts);

        // Open single database with column families (index CF uses lighter settings)
        let main_opts = opts.clone();
        let db = Arc::new(Self::open_or_repair_cf(&opts, &storage_path, move || {
            vec![
                ColumnFamilyDescriptor::new("default", main_opts.clone()),
                ColumnFamilyDescriptor::new(CF_INDEX, {
                    let mut idx_opts = Options::default();
                    idx_opts.create_if_missing(true);
                    idx_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                    idx_opts.set_max_write_buffer_number(2);
                    idx_opts.set_write_buffer_size(ROCKSDB_MEMORY_WRITE_BUFFER_BYTES);
                    idx_opts
                }),
                ColumnFamilyDescriptor::new(CF_OPLOG, {
                    let mut oplog_opts = Options::default();
                    oplog_opts.create_if_missing(true);
                    oplog_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                    oplog_opts.set_max_write_buffer_number(2);
                    oplog_opts.set_write_buffer_size(ROCKSDB_MEMORY_WRITE_BUFFER_BYTES);
                    oplog_opts
                }),
            ]
        })?);

        // Migrate from old separate-DB layout if needed
        Self::migrate_from_separate_dbs(path, &db)?;

        let write_mode = WriteMode::default();
        tracing::info!(
            "Storage initialized with {:?} write mode (latency: {})",
            write_mode,
            if write_mode == WriteMode::Sync {
                "2-10ms per write"
            } else {
                "<1ms per write"
            }
        );

        Ok(Self {
            db,
            storage_path: path.to_path_buf(),
            write_mode,
            write_retry_buffer: parking_lot::Mutex::new(std::collections::VecDeque::new()),
            write_failure_count: std::sync::atomic::AtomicU64::new(0),
            oplog_append_lock: parking_lot::Mutex::new(()),
            record_mutation_lock: parking_lot::Mutex::new(()),
        })
    }

    /// Open a RocksDB database with column families, automatically repairing if corruption is detected.
    ///
    /// On hard kills (ONNX deadlock, OOM, kill -9), RocksDB SST files can be left
    /// in a partially written state. This attempts repair before giving up.
    ///
    /// Takes a builder closure because `ColumnFamilyDescriptor` is not `Clone` —
    /// we need to rebuild descriptors for the retry path after repair.
    fn open_or_repair_cf<F>(opts: &Options, path: &Path, build_cfs: F) -> Result<DB>
    where
        F: Fn() -> Vec<ColumnFamilyDescriptor>,
    {
        match DB::open_cf_descriptors(opts, path, build_cfs()) {
            Ok(db) => Ok(db),
            Err(open_err) => {
                let err_str = open_err.to_string();
                // Only attempt repair for corruption-related errors
                if err_str.contains("Corruption")
                    || err_str.contains("bad block")
                    || err_str.contains("checksum mismatch")
                    || err_str.contains("MANIFEST")
                    || err_str.contains("CURRENT")
                {
                    tracing::warn!(
                        error = %open_err,
                        "RocksDB corruption detected in memory storage, attempting repair"
                    );
                    if let Err(repair_err) = DB::repair(opts, path) {
                        tracing::error!(
                            error = %repair_err,
                            "RocksDB repair failed for memory storage"
                        );
                        return Err(anyhow::anyhow!(
                            "Failed to open or repair memory storage: open={open_err}, repair={repair_err}"
                        ));
                    }
                    tracing::info!("RocksDB repair succeeded for memory storage, reopening");
                    DB::open_cf_descriptors(opts, path, build_cfs()).map_err(|e| {
                        anyhow::anyhow!("Failed to open memory storage after repair: {e}")
                    })
                } else {
                    Err(anyhow::anyhow!("Failed to open memory storage: {open_err}"))
                }
            }
        }
    }

    /// Migrate from old separate-DB layout (memories/ + memory_index/) to single CF-based DB.
    ///
    /// Detects whether old directories exist and the new CFs are empty, then bulk-copies
    /// all data into the unified DB. Old directories are renamed to *.pre_cf_migration
    /// for rollback safety.
    fn migrate_from_separate_dbs(base_path: &Path, db: &DB) -> Result<()> {
        let old_memories_dir = base_path.join("memories");
        let old_index_dir = base_path.join("memory_index");

        let has_old_memories = old_memories_dir.is_dir();
        let has_old_index = old_index_dir.is_dir();

        if !has_old_memories && !has_old_index {
            return Ok(());
        }

        tracing::info!("Detected old separate-DB layout, migrating to column families...");
        let mut total_migrated = 0usize;

        // Migrate main memories → default CF
        if has_old_memories {
            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_memories_dir, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;
                    for item in old_db.iterator(IteratorMode::Start) {
                        let (key, value) = item.map_err(|e| {
                            anyhow::anyhow!("RocksDB iterator error during memory migration: {e}")
                        })?;
                        batch.put(&key, &value);
                        count += 1;
                        if count.is_multiple_of(10_000) {
                            db.write(std::mem::take(&mut batch))?;
                            tracing::info!("  memories: migrated {count} entries...");
                        }
                    }
                    if !batch.is_empty() {
                        db.write(batch)?;
                    }
                    drop(old_db);
                    total_migrated += count;
                    tracing::info!("  memories: migrated {count} entries to default CF");

                    let backup_name = base_path.join("memories.pre_cf_migration");
                    if backup_name.exists() {
                        let _ = std::fs::remove_dir_all(&backup_name);
                    }
                    if let Err(e) = std::fs::rename(&old_memories_dir, &backup_name) {
                        tracing::warn!("Could not rename old memories dir: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Could not open old memories DB for migration: {e}");
                }
            }
        }

        // Migrate index → memory_index CF
        if has_old_index {
            let index_cf = db
                .cf_handle(CF_INDEX)
                .expect("memory_index CF must exist during migration");

            let old_opts = Options::default();
            match DB::open_for_read_only(&old_opts, &old_index_dir, false) {
                Ok(old_db) => {
                    let mut batch = WriteBatch::default();
                    let mut count = 0usize;
                    for item in old_db.iterator(IteratorMode::Start) {
                        let (key, value) = item.map_err(|e| {
                            anyhow::anyhow!("RocksDB iterator error during index migration: {e}")
                        })?;
                        batch.put_cf(&index_cf, &key, &value);
                        count += 1;
                        if count.is_multiple_of(10_000) {
                            db.write(std::mem::take(&mut batch))?;
                            tracing::info!("  index: migrated {count} entries...");
                        }
                    }
                    if !batch.is_empty() {
                        db.write(batch)?;
                    }
                    drop(old_db);
                    total_migrated += count;
                    tracing::info!("  index: migrated {count} entries to {CF_INDEX} CF");

                    let backup_name = base_path.join("memory_index.pre_cf_migration");
                    if backup_name.exists() {
                        let _ = std::fs::remove_dir_all(&backup_name);
                    }
                    if let Err(e) = std::fs::rename(&old_index_dir, &backup_name) {
                        tracing::warn!("Could not rename old memory_index dir: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Could not open old memory_index DB for migration: {e}");
                }
            }
        }

        if total_migrated > 0 {
            tracing::info!(
                "Memory storage migration complete: {total_migrated} total entries migrated"
            );
        }

        Ok(())
    }

    /// Get the base storage path
    pub fn path(&self) -> &Path {
        &self.storage_path
    }

    /// Store a memory with configurable write durability
    ///
    /// ROBOTICS OPTIMIZATION: Write mode is configurable via SHODH_WRITE_MODE env var.
    /// - Async (default): <1ms per write, data survives process crashes
    /// - Sync: 2-10ms per write, data survives power loss
    ///
    /// For robotics/edge: Use async mode + periodic flush() calls for best latency.
    /// For compliance/critical: Set SHODH_WRITE_MODE=sync for full durability.
    pub fn store(&self, memory: &Memory) -> Result<()> {
        // Opportunistically drain retry buffer on successful writes
        self.drain_retry_buffer();

        match self.store_inner(memory) {
            Ok(()) => Ok(()),
            Err(e) => {
                let err_str = e.to_string();
                // Buffer transient failures (lock contention, disk pressure)
                // but not serialization errors (those are permanent)
                if err_str.contains("lock")
                    || err_str.contains("LOCK")
                    || err_str.contains("disk")
                    || err_str.contains("space")
                    || err_str.contains("I/O")
                {
                    self.write_failure_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let mut buffer = self.write_retry_buffer.lock();
                    if buffer.len() < WRITE_RETRY_BUFFER_CAPACITY {
                        tracing::warn!(
                            memory_id = %memory.id.0,
                            buffer_len = buffer.len() + 1,
                            error = %e,
                            "Write failed, buffered for retry"
                        );
                        buffer.push_back(memory.clone());
                    } else {
                        tracing::error!(
                            memory_id = %memory.id.0,
                            error = %e,
                            "Write failed and retry buffer full — memory dropped"
                        );
                    }
                }
                Err(e)
            }
        }
    }

    /// Internal store implementation (two-phase commit with rollback)
    fn store_inner(&self, memory: &Memory) -> Result<()> {
        let key = memory.id.0.as_bytes();

        // Serialize memory (postcard + SHO v2 envelope)
        let value = crate::serialization::encode_sho(memory)
            .context(format!("Failed to serialize memory {}", memory.id.0))?;

        // Use write mode based on configuration
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);

        // Store in main database
        self.db
            .put_opt(key, &value, &write_opts)
            .context(format!("Failed to put memory {} in RocksDB", memory.id.0))?;

        // Update indices - rollback main write on failure for consistency
        if let Err(e) = self.update_indices(memory) {
            // Rollback: delete the memory we just wrote
            if let Err(del_err) = self.db.delete(key) {
                tracing::error!(
                    "Index write failed AND rollback failed for memory {}: index_err={}, delete_err={}",
                    memory.id.0, e, del_err
                );
            }
            return Err(e.context(format!(
                "Failed to update indices for memory {} (rolled back)",
                memory.id.0
            )));
        }

        Ok(())
    }

    /// Drain the retry buffer, re-attempting failed writes.
    /// Called opportunistically on every successful store() and explicitly
    /// from the maintenance cycle.
    pub fn drain_retry_buffer(&self) -> usize {
        let mut buffer = self.write_retry_buffer.lock();
        if buffer.is_empty() {
            return 0;
        }

        let count = buffer.len();
        let mut succeeded = 0;
        let mut still_failing = std::collections::VecDeque::new();

        for memory in buffer.drain(..) {
            match self.store_inner(&memory) {
                Ok(()) => {
                    succeeded += 1;
                    tracing::info!(
                        memory_id = %memory.id.0,
                        "Retried write succeeded"
                    );
                }
                Err(e) => {
                    tracing::debug!(
                        memory_id = %memory.id.0,
                        error = %e,
                        "Retry write still failing"
                    );
                    if still_failing.len() < WRITE_RETRY_BUFFER_CAPACITY {
                        still_failing.push_back(memory);
                    }
                }
            }
        }

        *buffer = still_failing;

        if succeeded > 0 || !buffer.is_empty() {
            tracing::info!(
                "Write retry drain: {}/{} succeeded, {} still pending",
                succeeded,
                count,
                buffer.len()
            );
        }

        succeeded
    }

    /// Number of writes currently buffered for retry
    pub fn pending_retry_count(&self) -> usize {
        self.write_retry_buffer.lock().len()
    }

    /// Total number of write failures since server start
    pub fn total_write_failures(&self) -> u64 {
        self.write_failure_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Update secondary indices for efficient retrieval
    fn update_indices(&self, memory: &Memory) -> Result<()> {
        let idx = self.index_cf();
        let mut batch = WriteBatch::default();

        // === Standard Indices ===

        // Index by date (for temporal queries)
        // BUG-001 FIX: Include memory_id in key to allow multiple memories per day
        // Old format: date:YYYYMMDD (overwrites on same day)
        // New format: date:YYYYMMDD:uuid (unique per memory)
        let date_key = format!(
            "date:{}:{}",
            memory.created_at.format("%Y%m%d"),
            memory.id.0
        );
        batch.put_cf(idx, date_key.as_bytes(), b"1");

        // Index by type
        let type_key = format!(
            "type:{:?}:{}",
            memory.experience.experience_type, memory.id.0
        );
        batch.put_cf(idx, type_key.as_bytes(), b"1");

        // Index by importance (quantized into buckets)
        let importance_bucket = (memory.importance() * 10.0) as u32;
        let importance_key = format!("importance:{}:{}", importance_bucket, memory.id.0);
        batch.put_cf(idx, importance_key.as_bytes(), b"1");

        // Index by entities (case-insensitive for tag search compatibility)
        for entity in &memory.experience.entities {
            let normalized_entity = entity.to_lowercase();
            let entity_key = format!("entity:{}:{}", normalized_entity, memory.id.0);
            batch.put_cf(idx, entity_key.as_bytes(), b"1");
        }

        // Index by tags (separate from entities for explicit tag queries)
        for tag in &memory.experience.tags {
            let normalized_tag = tag.to_lowercase();
            let tag_key = format!("tag:{}:{}", normalized_tag, memory.id.0);
            batch.put_cf(idx, tag_key.as_bytes(), b"1");
        }

        // Index by episode_id (for temporal/episodic retrieval)
        // Episode is the primary temporal grouping - memories in same episode are highly related
        if let Some(ctx) = &memory.experience.context {
            if let Some(episode_id) = &ctx.episode.episode_id {
                let episode_key = format!("episode:{}:{}", episode_id, memory.id.0);
                batch.put_cf(idx, episode_key.as_bytes(), b"1");

                // Also index by sequence within episode for temporal ordering
                if let Some(seq) = ctx.episode.sequence_number {
                    // Zero-pad sequence number for correct lexicographic ordering in RocksDB
                    // Without padding: 1, 10, 100, 2, 20... With {:010}: 0000000001, 0000000002...
                    let seq_key = format!("episode_seq:{}:{:010}:{}", episode_id, seq, memory.id.0);
                    batch.put_cf(idx, seq_key.as_bytes(), b"1");
                }
            }
        }

        // === Robotics Indices ===

        // Index by robot_id (for multi-robot systems)
        if let Some(ref robot_id) = memory.experience.robot_id {
            let robot_key = format!("robot:{}:{}", robot_id, memory.id.0);
            batch.put_cf(idx, robot_key.as_bytes(), b"1");
        }

        // Index by mission_id (for mission context retrieval)
        if let Some(ref mission_id) = memory.experience.mission_id {
            let mission_key = format!("mission:{}:{}", mission_id, memory.id.0);
            batch.put_cf(idx, mission_key.as_bytes(), b"1");
        }

        // Index by geo_location (for spatial queries) using geohash
        // Key format: geo:GEOHASH:memory_id (geohash at precision 10 = ~1.2m x 60cm)
        // Geohash enables efficient prefix-based spatial queries
        if let Some(geo) = memory.experience.geo_location {
            let lat = geo[0];
            let lon = geo[1];
            // Use precision 10 for warehouse-level accuracy (~1.2m cells)
            let geohash = super::types::geohash_encode(lat, lon, 10);
            let geo_key = format!("geo:{}:{}", geohash, memory.id.0);
            batch.put_cf(idx, geo_key.as_bytes(), b"1");
        }

        // Index by action_type (for action-based retrieval)
        if let Some(ref action_type) = memory.experience.action_type {
            let action_key = format!("action:{}:{}", action_type, memory.id.0);
            batch.put_cf(idx, action_key.as_bytes(), b"1");
        }

        // Index by reward (bucketed, for RL-style queries)
        // Bucket: -1.0 to 1.0 mapped to 0-20
        if let Some(reward) = memory.experience.reward {
            let clamped_reward = reward.clamp(-1.0, 1.0);
            let reward_bucket = ((clamped_reward + 1.0) * 10.0) as i32;
            let reward_key = format!("reward:{}:{}", reward_bucket, memory.id.0);
            batch.put_cf(idx, reward_key.as_bytes(), b"1");
        }

        // === Content Hash Index (idempotency) ===
        // Index by SHA256 content hash for dedup (issue #109)
        // Key format: content_hash:{hex} -> memory_id (16 bytes UUID)
        // Enables O(1) duplicate detection on remember()
        {
            let content_hash = Self::sha256_content_hash(&memory.experience.content);
            let hash_key = format!("content_hash:{}", content_hash);
            batch.put_cf(idx, hash_key.as_bytes(), memory.id.0.as_bytes());
        }

        // === External Linking Index ===
        // Index by external_id for upsert operations (Linear, GitHub, etc.)
        // Key format: external:{source}:{id}:{memory_id} -> memory_id
        // Enables O(1) lookup when syncing from external systems
        if let Some(ref external_id) = memory.external_id {
            let external_key = format!("external:{}:{}", external_id, memory.id.0);
            // Store memory_id as value for direct lookup
            batch.put_cf(idx, external_key.as_bytes(), memory.id.0.as_bytes());
        }

        // === Hierarchy Index ===
        // Index by parent_id for tree queries (list children, build tree)
        // Key format: parent:{parent_id}:{child_id} -> 1
        // Enables O(1) lookup of all children for a parent
        if let Some(ref parent_id) = memory.parent_id {
            let parent_key = format!("parent:{}:{}", parent_id.0, memory.id.0);
            batch.put_cf(idx, parent_key.as_bytes(), b"1");
        }

        // Use write mode based on configuration
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db.write_opt(batch, &write_opts)?;
        Ok(())
    }

    /// Compute SHA256 hex digest for content dedup indexing (issue #109)
    fn sha256_content_hash(content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Look up an existing memory by content hash (idempotency dedup, issue #109).
    /// Returns the MemoryId if identical content was already stored.
    pub fn get_by_content_hash(&self, content: &str) -> Option<MemoryId> {
        let content_hash = Self::sha256_content_hash(content);
        let hash_key = format!("content_hash:{}", content_hash);
        let idx = self.index_cf();
        match self.db.get_cf(idx, hash_key.as_bytes()) {
            Ok(Some(value)) if value.len() == 16 => {
                let uuid = uuid::Uuid::from_slice(&value).ok()?;
                // Verify the memory still exists (might have been deleted)
                let memory_id = MemoryId(uuid);
                match self.get_opt(&memory_id) {
                    Ok(Some(_)) => Some(memory_id),
                    Ok(None) => {
                        // Proven stale: RocksDB reports no record at that id,
                        // so the memory really was deleted. Clean up.
                        let _ = self.db.delete_cf(idx, hash_key.as_bytes());
                        None
                    }
                    Err(e) => {
                        // Inconclusive read — the memory may well exist. Do NOT
                        // delete the dedup entry: destroying it on a transient
                        // failure loses the only pointer from this content to
                        // its memory, and the caller then writes a duplicate
                        // whose store overwrites the entry with the new id.
                        tracing::warn!(
                            memory_id = %memory_id.0,
                            error = %e,
                            "content-hash dedup lookup could not read the target \
                             memory — keeping the dedup entry and reporting no \
                             match for this call only"
                        );
                        None
                    }
                }
            }
            _ => None,
        }
    }

    /// Retrieve a memory by ID, separating "this key does not exist" from
    /// "this read did not succeed".
    ///
    /// `Ok(None)` is returned in exactly one case: RocksDB itself reported no
    /// value at the key. That is the only outcome that proves the memory is
    /// gone. Every other failure — a RocksDB/IO error, or bytes that are
    /// present but do not decode — is `Err`, because the record may well still
    /// be there. Callers that delete index entries, dedup entries or any other
    /// pointer *to* a memory must key that deletion off `Ok(None)` and never
    /// off `Err`: an inconclusive read is not proof of absence, and treating it
    /// as one turns a transient blip into permanent, silent data loss.
    ///
    /// Performs the same lazy migration as [`Self::get`]: if a memory is in a
    /// legacy format it is re-written in the current format for faster future
    /// reads.
    pub fn get_opt(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let key = id.0.as_bytes();
        match self.db.get(key)? {
            Some(value) => {
                // Checked against the key it was fetched under: a decode whose
                // id disagrees with its key is a fabrication, and the error
                // returns BEFORE the lazy migration below so a fabrication is
                // never written anywhere.
                let (memory, needs_migration) =
                    deserialize_memory_checked(key, &value).with_context(|| {
                        format!(
                            "Failed to deserialize memory {} ({} bytes)",
                            id.0,
                            value.len()
                        )
                    })?;

                // Lazy migration: re-write legacy formats in current format
                if needs_migration {
                    if let Err(e) = self.migrate_memory_format(&memory) {
                        // Migration failure is non-fatal - log and continue
                        tracing::debug!("Lazy migration skipped for memory {}: {}", memory.id.0, e);
                    }
                }

                Ok(Some(memory))
            }
            None => Ok(None),
        }
    }

    /// Retrieve a memory by ID, where a missing memory is itself an error.
    ///
    /// This is the right call when the record IS the answer — a single-id read
    /// has nothing to return if the memory is absent. Callers that are
    /// *scanning* (and would react to the failure by discarding a pointer to
    /// the memory) must use [`Self::get_opt`] instead, which distinguishes
    /// absence from an unsuccessful read.
    pub fn get(&self, id: &MemoryId) -> Result<Memory> {
        self.get_opt(id)?
            .ok_or_else(|| anyhow!("Memory not found: {id:?}"))
    }

    /// Atomically read a memory, apply `f`, and write it back.
    ///
    /// The read and the write happen under one lock, so two concurrent callers
    /// cannot both read the pre-update record and have the second overwrite the
    /// first's change. Every read-modify-write of a memory record must go
    /// through here; doing the same three steps inline is the lost-update bug
    /// this exists to remove, and no amount of shrinking the gap between them
    /// fixes it.
    ///
    /// `f` sees the CURRENT stored record, not a snapshot the caller took
    /// earlier — that is the whole point, so callers must express their change
    /// as a mutation rather than as a whole replacement record.
    ///
    /// Returns the memory as written. `Ok(None)` if the memory does not exist;
    /// an unreadable record is an `Err`, never a silent no-op (see
    /// [`Self::get_opt`]).
    pub fn modify<F>(&self, id: &MemoryId, f: F) -> Result<Option<Memory>>
    where
        F: FnOnce(&mut Memory),
    {
        let _guard = self.record_mutation_lock.lock();
        let mut memory = match self.get_opt(id)? {
            Some(memory) => memory,
            None => return Ok(None),
        };
        f(&mut memory);
        self.update(&memory)?;
        Ok(Some(memory))
    }

    /// Re-write a memory in current format (lazy migration helper)
    fn migrate_memory_format(&self, memory: &Memory) -> Result<()> {
        let key = memory.id.0.as_bytes();
        let value = crate::serialization::encode_sho(memory)
            .context("Failed to serialize for migration")?;

        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(false); // Async is fine for migration

        self.db.put_opt(key, &value, &write_opts)?;
        tracing::debug!("Migrated memory {} to current format", memory.id.0);
        Ok(())
    }

    /// Find a memory by its external_id (e.g., "linear:SHO-39", "github:pr-123")
    ///
    /// Returns the memory if found, None if no memory with this external_id exists.
    /// Used for upsert operations when syncing from external sources.
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<Memory>> {
        // Index key format: external:{external_id}:{memory_id}
        let prefix = format!("external:{external_id}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );

        for (key, _value) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Extract memory_id from key (format: external:{external_id}:{memory_id})
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    return Ok(Some(self.get(&MemoryId(uuid))?));
                }
            }
        }

        Ok(None)
    }

    /// Update an existing memory
    ///
    /// ALGO-004 FIX: Re-indexes memory after update to handle importance drift.
    /// When Hebbian feedback changes importance, the old bucket index becomes stale.
    /// We remove old indices before storing to ensure index consistency.
    pub fn update(&self, memory: &Memory) -> Result<()> {
        // Remove old indices first (they may have stale importance buckets)
        self.remove_from_indices(&memory.id)?;
        // Store with fresh indices
        self.store(memory)
    }

    /// Persist access-metadata bumps for a batch of just-recalled memories in a
    /// single WriteBatch — the read-path-optimized alternative to calling
    /// `update()` per candidate.
    ///
    /// On a recall, `Memory::update_access` only changes `last_accessed`,
    /// `access_count`, and `importance`. Of the indexed fields, ONLY the
    /// importance bucket (`(importance * 10.0) as u32`) can change — date, type,
    /// entities, tags, episode, etc. are immutable across an access. So the old
    /// per-candidate `update()` (a full `remove_from_indices` re-read plus a
    /// rewrite of all ~14 index keys, then N separate DB writes) was almost
    /// entirely redundant write amplification on the hot recall path.
    ///
    /// This rewrites each main record once and touches the importance index ONLY
    /// when a memory's bucket actually crossed, coalescing everything into one
    /// write. `items` is `(memory, importance_before_the_access)`.
    pub fn persist_access_updates(&self, items: &[(&Memory, f32)]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let idx = self.index_cf();
        let mut batch = WriteBatch::default();
        for (memory, importance_before) in items {
            // Main record (carries the new access_count / last_accessed /
            // importance). Same default-CF + SHO-v2 encoding as `store_inner`.
            let value = crate::serialization::encode_sho(memory).context(format!(
                "serialize memory {} for access update",
                memory.id.0
            ))?;
            batch.put(memory.id.0.as_bytes(), value);

            // Importance index: rewrite only when the coarse bucket changed.
            let old_bucket = (*importance_before * 10.0) as u32;
            let new_bucket = (memory.importance() * 10.0) as u32;
            if old_bucket != new_bucket {
                let old_key = format!("importance:{}:{}", old_bucket, memory.id.0);
                let new_key = format!("importance:{}:{}", new_bucket, memory.id.0);
                batch.delete_cf(idx, old_key.as_bytes());
                batch.put_cf(idx, new_key.as_bytes(), b"1");
            }
        }
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .write_opt(batch, &write_opts)
            .context("persist_access_updates batch write")?;
        Ok(())
    }

    /// Delete a memory with configurable durability
    #[allow(unused)] // Public API - available for memory management
    pub fn delete(&self, id: &MemoryId) -> Result<()> {
        // Clean up indices FIRST while the memory still exists in the main DB,
        // since remove_from_indices() needs to read the memory to reconstruct index keys.
        self.remove_from_indices(id)?;

        // Then delete from main database
        let key = id.0.as_bytes();
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db.delete_opt(key, &write_opts)?;

        // Delete vector mapping if present
        let mapping_key = format!("vmapping:{}", id.0);
        let _ = self.db.delete_opt(mapping_key.as_bytes(), &write_opts);

        Ok(())
    }

    /// Remove memory from all indices
    /// BUG-005 FIX: Direct key deletion instead of O(n) scan with contains()
    /// We reconstruct index keys from memory metadata for O(k) deletion
    fn remove_from_indices(&self, id: &MemoryId) -> Result<()> {
        // Fetch memory to reconstruct index keys
        let memory = match self.get(id) {
            Ok(m) => m,
            Err(_) => {
                tracing::debug!("Memory {} not found, skipping index cleanup", id.0);
                return Ok(());
            }
        };

        let idx = self.index_cf();
        let mut batch = WriteBatch::default();

        // Reconstruct and delete all index keys directly (O(k) instead of O(n))

        // Date index
        let date_key = format!("date:{}:{}", memory.created_at.format("%Y%m%d"), id.0);
        batch.delete_cf(idx, date_key.as_bytes());

        // Type index
        let type_key = format!("type:{:?}:{}", memory.experience.experience_type, id.0);
        batch.delete_cf(idx, type_key.as_bytes());

        // Importance index
        let importance_bucket = (memory.importance() * 10.0) as u32;
        let importance_key = format!("importance:{}:{}", importance_bucket, id.0);
        batch.delete_cf(idx, importance_key.as_bytes());

        // Entity indices (must match the to_lowercase() normalization in update_indices)
        for entity in &memory.experience.entities {
            let normalized_entity = entity.to_lowercase();
            let entity_key = format!("entity:{}:{}", normalized_entity, id.0);
            batch.delete_cf(idx, entity_key.as_bytes());
        }

        // Tag indices
        for tag in &memory.experience.tags {
            let normalized_tag = tag.to_lowercase();
            let tag_key = format!("tag:{}:{}", normalized_tag, id.0);
            batch.delete_cf(idx, tag_key.as_bytes());
        }

        // Episode indices
        if let Some(ctx) = &memory.experience.context {
            if let Some(episode_id) = &ctx.episode.episode_id {
                let episode_key = format!("episode:{}:{}", episode_id, id.0);
                batch.delete_cf(idx, episode_key.as_bytes());

                if let Some(seq) = ctx.episode.sequence_number {
                    let seq_key = format!("episode_seq:{}:{:010}:{}", episode_id, seq, id.0);
                    batch.delete_cf(idx, seq_key.as_bytes());
                }
            }
        }

        // Robot index
        if let Some(ref robot_id) = memory.experience.robot_id {
            let robot_key = format!("robot:{}:{}", robot_id, id.0);
            batch.delete_cf(idx, robot_key.as_bytes());
        }

        // Mission index
        if let Some(ref mission_id) = memory.experience.mission_id {
            let mission_key = format!("mission:{}:{}", mission_id, id.0);
            batch.delete_cf(idx, mission_key.as_bytes());
        }

        // Geo index
        if let Some(geo) = memory.experience.geo_location {
            let geohash = super::types::geohash_encode(geo[0], geo[1], 10);
            let geo_key = format!("geo:{}:{}", geohash, id.0);
            batch.delete_cf(idx, geo_key.as_bytes());
        }

        // Action index
        if let Some(ref action_type) = memory.experience.action_type {
            let action_key = format!("action:{}:{}", action_type, id.0);
            batch.delete_cf(idx, action_key.as_bytes());
        }

        // Reward index (must match the clamp in update_indices)
        if let Some(reward) = memory.experience.reward {
            let clamped_reward = reward.clamp(-1.0, 1.0);
            let reward_bucket = ((clamped_reward + 1.0) * 10.0) as i32;
            let reward_key = format!("reward:{}:{}", reward_bucket, id.0);
            batch.delete_cf(idx, reward_key.as_bytes());
        }

        // Content hash index (idempotency dedup)
        {
            let content_hash = Self::sha256_content_hash(&memory.experience.content);
            let hash_key = format!("content_hash:{}", content_hash);
            batch.delete_cf(idx, hash_key.as_bytes());
        }

        // External linking index
        if let Some(ref external_id) = memory.external_id {
            let external_key = format!("external:{}:{}", external_id, id.0);
            batch.delete_cf(idx, external_key.as_bytes());
        }

        // Parent index (hierarchy)
        if let Some(ref parent_id) = memory.parent_id {
            let parent_key = format!("parent:{}:{}", parent_id.0, id.0);
            batch.delete_cf(idx, parent_key.as_bytes());
        }

        // Use write mode based on configuration
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db.write_opt(batch, &write_opts)?;
        Ok(())
    }

    /// Search memories by various criteria
    pub fn search(&self, criteria: SearchCriteria) -> Result<Vec<Memory>> {
        let mut memory_ids = Vec::new();

        match criteria {
            // === Standard Criteria ===
            SearchCriteria::ByDate { start, end } => {
                memory_ids = self.search_by_date_range(start, end)?;
            }
            SearchCriteria::ByType(exp_type) => {
                memory_ids = self.search_by_type(exp_type)?;
            }
            SearchCriteria::ByImportance { min, max } => {
                memory_ids = self.search_by_importance(min, max)?;
            }
            SearchCriteria::ByEntity(entity) => {
                memory_ids = self.search_by_entity(&entity)?;
            }
            SearchCriteria::ByTags(tags) => {
                memory_ids = self.search_by_tags(&tags)?;
            }

            // === Temporal/Episode Criteria ===
            SearchCriteria::ByEpisode(episode_id) => {
                memory_ids = self.search_by_episode(&episode_id)?;
            }
            SearchCriteria::ByEpisodeSequence {
                episode_id,
                min_sequence,
                max_sequence,
            } => {
                memory_ids =
                    self.search_by_episode_sequence(&episode_id, min_sequence, max_sequence)?;
            }

            // === Robotics Criteria ===
            SearchCriteria::ByRobot(robot_id) => {
                memory_ids = self.search_by_robot(&robot_id)?;
            }
            SearchCriteria::ByMission(mission_id) => {
                memory_ids = self.search_by_mission(&mission_id)?;
            }
            SearchCriteria::ByLocation {
                lat,
                lon,
                radius_meters,
                limit,
            } => {
                memory_ids = self.search_by_location(lat, lon, radius_meters, limit)?;
            }
            SearchCriteria::ByActionType(action_type) => {
                memory_ids = self.search_by_action_type(&action_type)?;
            }
            SearchCriteria::ByReward { min, max } => {
                memory_ids = self.search_by_reward(min, max)?;
            }

            // === Compound Criteria ===
            SearchCriteria::Combined(criterias) => {
                // Intersection of all criteria results
                // Use HashSet for O(1) lookups instead of O(n) Vec::contains
                use std::collections::HashSet;
                let mut result_sets: Vec<HashSet<MemoryId>> = Vec::new();
                for c in criterias {
                    result_sets.push(
                        self.search(c)?
                            .into_iter()
                            .map(|m| m.id)
                            .collect::<HashSet<_>>(),
                    );
                }

                if !result_sets.is_empty() {
                    let first_set = result_sets.remove(0);
                    memory_ids = first_set
                        .into_iter()
                        .filter(|id| result_sets.iter().all(|set| set.contains(id)))
                        .collect();
                }
            }

            // === Hierarchy Criteria ===
            SearchCriteria::ByParent(parent_id) => {
                memory_ids = self.search_by_parent(&parent_id)?;
            }
            SearchCriteria::RootsOnly => {
                memory_ids = self.search_roots()?;
            }
        }

        // Fetch full memories, filtering out forgotten ones
        let mut memories = Vec::new();
        for id in memory_ids {
            if let Ok(memory) = self.get(&id) {
                if !memory.is_forgotten() {
                    memories.push(memory);
                }
            }
        }

        Ok(memories)
    }

    fn search_by_date_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let start_key = format!("date:{}", start.format("%Y%m%d"));
        // BUG-001 FIX: End key needs ~ suffix to include all UUIDs for that date
        // Keys are: date:YYYYMMDD:uuid, so date:20251207~ comes after all Dec 7 entries
        let end_key = format!("date:{}~", end.format("%Y%m%d"));

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(start_key.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _value) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if &*key_str > end_key.as_str() {
                break;
            }
            // BUG-001 FIX: Extract memory_id from key (format: date:YYYYMMDD:uuid)
            if key_str.starts_with("date:") {
                let parts: Vec<&str> = key_str.split(':').collect();
                if parts.len() >= 3 {
                    // parts[0] = "date", parts[1] = "YYYYMMDD", parts[2] = uuid
                    if let Ok(uuid) = uuid::Uuid::parse_str(parts[2]) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    fn search_by_type(&self, exp_type: ExperienceType) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("type:{exp_type:?}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Extract ID from key
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    fn search_by_importance(&self, min: f32, max: f32) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let min_bucket = (min * 10.0) as u32;
        let max_bucket = (max * 10.0) as u32;

        for bucket in min_bucket..=max_bucket {
            let prefix = format!("importance:{bucket}:");
            let iter = self.db.iterator_cf(
                self.index_cf(),
                IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
            );

            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                // Extract ID from key
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    fn search_by_entity(&self, entity: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        // Normalize to lowercase for case-insensitive matching
        let normalized_entity = entity.to_lowercase();
        let prefix = format!("entity:{normalized_entity}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Extract ID from key
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by tags (returns memories matching ANY of the provided tags)
    fn search_by_tags(&self, tags: &[String]) -> Result<Vec<MemoryId>> {
        use std::collections::HashSet;

        // Union of all tag matches
        let mut all_ids = HashSet::new();

        for tag in tags {
            // Normalize to lowercase for case-insensitive matching
            let normalized_tag = tag.to_lowercase();
            let prefix = format!("tag:{normalized_tag}:");
            let iter = self.db.iterator_cf(
                self.index_cf(),
                IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
            );
            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        all_ids.insert(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(all_ids.into_iter().collect())
    }

    /// Search memories by episode ID
    /// Returns all memories in the specified episode
    fn search_by_episode(&self, episode_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("episode:{episode_id}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by episode with sequence filtering
    /// Returns memories in temporal order within the episode
    fn search_by_episode_sequence(
        &self,
        episode_id: &str,
        min_sequence: Option<u32>,
        max_sequence: Option<u32>,
    ) -> Result<Vec<MemoryId>> {
        let mut results: Vec<(u32, MemoryId)> = Vec::new();

        // Scan the episode_seq index which has format: episode_seq:{episode_id}:{seq}:{memory_id}
        let prefix = format!("episode_seq:{episode_id}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );

        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }

            // Parse: episode_seq:{episode_id}:{seq}:{memory_id}
            if let Some(rest) = key_str.strip_prefix(&prefix) {
                let parts: Vec<&str> = rest.splitn(2, ':').collect();
                if parts.len() == 2 {
                    if let (Ok(seq), Ok(uuid)) =
                        (parts[0].parse::<u32>(), uuid::Uuid::parse_str(parts[1]))
                    {
                        // Apply sequence filters
                        let passes_min = min_sequence.is_none_or(|min| seq >= min);
                        let passes_max = max_sequence.is_none_or(|max| seq <= max);

                        if passes_min && passes_max {
                            results.push((seq, MemoryId(uuid)));
                        }
                    }
                }
            }
        }

        // Sort by sequence number for temporal ordering
        results.sort_by_key(|(seq, _)| *seq);

        Ok(results.into_iter().map(|(_, id)| id).collect())
    }

    // ========================================================================
    // ROBOTICS SEARCH METHODS
    // ========================================================================

    /// Search memories by robot/drone identifier
    fn search_by_robot(&self, robot_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("robot:{robot_id}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by mission identifier
    fn search_by_mission(&self, mission_id: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("mission:{mission_id}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by geographic location using geohash prefix scanning
    ///
    /// Performance: O(k) where k = memories in ~9 geohash cells covering the radius
    /// Previous approach was O(n) where n = all geo-indexed memories
    ///
    /// `limit`, if `Some`, bounds the number of ids returned to the nearest
    /// `limit` by geohash-decoded APPROXIMATE distance (cell-center at
    /// geohash precision 10 is within ~1m of the true position — plenty
    /// accurate for this sort/cap; nothing here is hydrated). Capping must
    /// happen here, before the caller's `search()` hydrates (get + full
    /// deserialize) every returned id — capping after hydration doesn't
    /// bound the hydration cost, which is the expensive part. `None`
    /// preserves the historical uncapped, geohash-scan-order behavior.
    fn search_by_location(
        &self,
        center_lat: f64,
        center_lon: f64,
        radius_meters: f64,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryId>> {
        use super::types::{geohash_decode, geohash_search_prefixes, GeoFilter};

        let geo_filter = GeoFilter::new(center_lat, center_lon, radius_meters);
        // (id, approx_distance_meters) — distance kept so we can sort
        // nearest-first before capping, without hydrating anything.
        let mut candidates: Vec<(MemoryId, f64)> = Vec::new();

        // Get geohash prefixes for center + neighbors at appropriate precision
        let prefixes = geohash_search_prefixes(center_lat, center_lon, radius_meters);

        // Scan only the relevant geohash cells (9 cells = center + 8 neighbors)
        for geohash_prefix in prefixes {
            // Use prefix WITHOUT trailing colon so shorter search prefixes (e.g. precision 6)
            // still match longer stored geohashes (precision 10).
            // Stored keys: "geo:s02jksd91f:{uuid}", search prefix: "geo:s02jks"
            let prefix = format!("geo:{}", geohash_prefix);
            let iter = self.db.iterator_cf(
                self.index_cf(),
                IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
            );

            for (key, _value) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }

                // Key format: geo:GEOHASH:memory_id
                let parts: Vec<&str> = key_str.split(':').collect();
                if parts.len() >= 3 {
                    let geohash = parts[1];
                    // Decode geohash to get approximate lat/lon for distance check
                    let (min_lat, min_lon, max_lat, max_lon) = geohash_decode(geohash);
                    let approx_lat = (min_lat + max_lat) / 2.0;
                    let approx_lon = (min_lon + max_lon) / 2.0;
                    let approx_distance = geo_filter.haversine_distance(approx_lat, approx_lon);

                    // Final haversine check for edge cases at cell boundaries
                    if approx_distance <= radius_meters {
                        if let Ok(uuid) = uuid::Uuid::parse_str(parts[2]) {
                            candidates.push((MemoryId(uuid), approx_distance));
                        }
                    }
                }
            }
        }

        // Nearest-first, MemoryId tie-break (never raw geohash scan order,
        // which is a RocksDB iteration artifact, not a meaningful ordering).
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        if let Some(n) = limit {
            candidates.truncate(n);
        }

        Ok(candidates.into_iter().map(|(id, _)| id).collect())
    }

    /// Search memories by action type
    fn search_by_action_type(&self, action_type: &str) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("action:{action_type}:");

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if let Some(id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Search memories by reward range (for RL-style queries)
    fn search_by_reward(&self, min: f32, max: f32) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();

        // Reward is bucketed similar to importance (-10 to 10 buckets)
        // Clamp to prevent bucket overflow from out-of-range values
        let clamped_min = min.clamp(-1.0, 1.0);
        let clamped_max = max.clamp(-1.0, 1.0);
        let min_bucket = ((clamped_min + 1.0) * 10.0) as i32; // -1.0 -> 0, 1.0 -> 20
        let max_bucket = ((clamped_max + 1.0) * 10.0) as i32;

        for bucket in min_bucket..=max_bucket {
            let prefix = format!("reward:{bucket}:");
            let iter = self.db.iterator_cf(
                self.index_cf(),
                IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
            );

            for (key, _) in iter.log_errors() {
                let key_str = String::from_utf8_lossy(&key);
                if !key_str.starts_with(&prefix) {
                    break;
                }
                if let Some(id_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        ids.push(MemoryId(uuid));
                    }
                }
            }
        }

        Ok(ids)
    }

    // =========================================================================
    // HIERARCHY SEARCH METHODS
    // =========================================================================

    /// Get all children of a parent memory
    fn search_by_parent(&self, parent_id: &MemoryId) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let prefix = format!("parent:{}:", parent_id.0);

        let iter = self.db.iterator_cf(
            self.index_cf(),
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );

        for (key, _) in iter.log_errors() {
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            // Key format: parent:{parent_uuid}:{child_uuid}
            if let Some(child_id_str) = key_str.strip_prefix(&prefix) {
                if let Ok(uuid) = uuid::Uuid::parse_str(child_id_str) {
                    ids.push(MemoryId(uuid));
                }
            }
        }

        Ok(ids)
    }

    /// Get all root memories (memories with no parent)
    fn search_roots(&self) -> Result<Vec<MemoryId>> {
        let mut roots = Vec::new();

        // Iterate all memories and check for parent_id = None
        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            if let Ok((memory, _)) = deserialize_memory_checked(&key, &value) {
                if memory.parent_id.is_none() {
                    roots.push(memory.id);
                }
            }
        }

        Ok(roots)
    }

    /// Get children of a memory (public API)
    pub fn get_children(&self, parent_id: &MemoryId) -> Result<Vec<Memory>> {
        let child_ids = self.search_by_parent(parent_id)?;
        let mut children = Vec::new();
        for id in child_ids {
            if let Ok(memory) = self.get(&id) {
                children.push(memory);
            }
        }
        Ok(children)
    }

    /// Get the parent chain (ancestors) of a memory
    pub fn get_ancestors(&self, memory_id: &MemoryId) -> Result<Vec<Memory>> {
        let mut ancestors = Vec::new();
        let mut current_id = memory_id.clone();

        // Walk up the parent chain (max 100 to prevent infinite loops)
        for _ in 0..100 {
            let memory = self.get(&current_id)?;
            if let Some(parent_id) = &memory.parent_id {
                let parent = self.get(parent_id)?;
                ancestors.push(parent.clone());
                current_id = parent_id.clone();
            } else {
                break; // Reached root
            }
        }

        Ok(ancestors)
    }

    /// Get the full hierarchy context for a memory
    /// Returns (ancestors, memory, children)
    pub fn get_hierarchy_context(
        &self,
        memory_id: &MemoryId,
    ) -> Result<(Vec<Memory>, Memory, Vec<Memory>)> {
        let memory = self.get(memory_id)?;
        let ancestors = self.get_ancestors(memory_id)?;
        let children = self.get_children(memory_id)?;
        Ok((ancestors, memory, children))
    }

    /// Get all memories in a subtree rooted at the given memory
    pub fn get_subtree(&self, root_id: &MemoryId, max_depth: usize) -> Result<Vec<Memory>> {
        let mut result = Vec::new();
        let mut queue = vec![(root_id.clone(), 0usize)];

        while let Some((id, depth)) = queue.pop() {
            if depth > max_depth {
                continue;
            }
            if let Ok(memory) = self.get(&id) {
                result.push(memory);
                // Add children to queue
                if depth < max_depth {
                    let child_ids = self.search_by_parent(&id)?;
                    for child_id in child_ids {
                        queue.push((child_id, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get all memory IDs without loading full Memory objects.
    ///
    /// Returns only 16-byte UUID keys (lightweight). Use with `get()` to load
    /// individual memories in batches to avoid OOM on large datasets.
    pub fn get_all_ids(&self) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_opt(IteratorMode::Start, read_opts);
        for (key, _) in iter.flatten() {
            if key.len() == 16 {
                let uuid_bytes: [u8; 16] = key[..16].try_into().unwrap();
                ids.push(MemoryId(uuid::Uuid::from_bytes(uuid_bytes)));
            }
        }
        Ok(ids)
    }

    /// Get all memories from long-term storage
    ///
    /// Only returns entries with valid 16-byte UUID keys (consistent with get_stats).
    /// WARNING: Loads entire DB into memory. For large stores, prefer `for_each_memory`.
    pub fn get_all(&self) -> Result<Vec<Memory>> {
        let mut memories = Vec::new();

        // fill_cache(false) prevents this maintenance scan from polluting
        // the block cache with cold data, reducing C++ peak memory
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_opt(IteratorMode::Start, read_opts);
        for (key, value) in iter.flatten() {
            // Only process valid 16-byte UUID keys (consistent with get_stats)
            if key.len() != 16 {
                continue;
            }
            if let Ok((memory, _)) = deserialize_memory_checked(&key, &value) {
                if !memory.is_forgotten() {
                    memories.push(memory);
                }
            }
        }

        Ok(memories)
    }

    /// Iterate over all memories without loading them all into memory at once.
    ///
    /// Calls `f` for each non-forgotten memory. Returns early if `f` returns an error.
    /// Use this instead of `get_all()` when you don't need all memories simultaneously.
    pub fn for_each_memory<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(Memory) -> Result<()>,
    {
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.fill_cache(false);
        let iter = self.db.iterator_opt(IteratorMode::Start, read_opts);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            if let Ok((memory, _)) = deserialize_memory_checked(&key, &value) {
                if !memory.is_forgotten() {
                    f(memory)?;
                }
            }
        }
        Ok(())
    }

    pub fn get_uncompressed_older_than(&self, cutoff: DateTime<Utc>) -> Result<Vec<Memory>> {
        let mut memories = Vec::new();

        // Iterate through all memories
        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            if let Ok((memory, _)) = deserialize_memory_checked(&key, &value) {
                if !memory.compressed && !memory.is_forgotten() && memory.created_at < cutoff {
                    memories.push(memory);
                }
            }
        }

        Ok(memories)
    }

    /// Mark memories as forgotten (soft delete) with atomic batch write.
    /// Returns the IDs of memories that were flagged, so callers can clean up
    /// secondary indices (vector, BM25, graph).
    ///
    /// Also deletes each flagged memory's `geo:` index entry, if it has one,
    /// in the SAME batch as the flag update (review finding, geotemporal Task
    /// 2 round 4): the scan already has the full `Memory` in hand, so this
    /// costs no extra RocksDB read/deserialize, and riding in the existing
    /// per-sweep `WriteBatch` costs no extra fsync — a sweep flagging N
    /// memories still does exactly one `write_opt`, not N. (Round 3 instead
    /// added a `remove_geo_index(&self, id)` function, called per id in the
    /// caller's loop over `flagged_ids` — that re-fetched and
    /// re-deserialized each memory via `self.get(id)`, even though the
    /// scan right here already had it, and issued its own
    /// WriteBatch+write_opt per id, so under `WriteMode::Sync` a single
    /// `OlderThan`/`LowImportance` sweep could cost up to N fsync'd writes.
    /// `remove_geo_index` has been removed — this function's own batch is
    /// the only place the geo key needs deleting, since nothing else in
    /// this codebase soft-forgets memories one at a time; hard-delete
    /// (`ForgetCriteria::ById`, `delete()`/`remove_from_indices`) already
    /// handles geo cleanup for the single-id case.)
    ///
    /// NOTE on restoration (still true, carried over from the round-3 doc
    /// comment this replaces): there is no un-forget/restore path anywhere
    /// in this codebase for ANY secondary index — grepped for
    /// restore/recover/unforget across src/; nothing reverses
    /// `mark_forgotten_by_*`, and the `"forgotten"` metadata flag is only
    /// ever set to `"true"`, never cleared. Vector, BM25, and graph cleanup
    /// on soft-forget are therefore already one-way, unrestorable
    /// operations; deleting the geo index entry here matches that existing
    /// precedent rather than inventing a new (and currently unsupported)
    /// restoration contract for geo alone.
    pub fn mark_forgotten_by_age(&self, cutoff: DateTime<Utc>) -> Result<Vec<MemoryId>> {
        let mut batch = rocksdb::WriteBatch::default();
        let mut flagged_ids = Vec::new();
        let now = Utc::now().to_rfc3339();
        let idx = self.index_cf();

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            if let Ok((mut memory, _)) = deserialize_memory_checked(&key, &value) {
                if memory.is_forgotten() {
                    continue;
                }
                if memory.created_at < cutoff {
                    flagged_ids.push(memory.id.clone());
                    if let Some(geo) = memory.experience.geo_location {
                        let geohash = super::types::geohash_encode(geo[0], geo[1], 10);
                        let geo_key = format!("geo:{}:{}", geohash, memory.id.0);
                        batch.delete_cf(idx, geo_key.as_bytes());
                    }
                    memory
                        .experience
                        .metadata
                        .insert("forgotten".to_string(), "true".to_string());
                    memory
                        .experience
                        .metadata
                        .insert("forgotten_at".to_string(), now.clone());

                    let updated_value = crate::serialization::encode_sho(&memory)?;
                    batch.put(&key, updated_value);
                }
            }
        }

        if !flagged_ids.is_empty() {
            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(true);
            self.db.write_opt(batch, &write_opts)?;
        }

        Ok(flagged_ids)
    }

    /// Mark memories with low importance as forgotten with atomic batch write.
    /// Returns the IDs of memories that were flagged, so callers can clean up
    /// secondary indices (vector, BM25, graph).
    ///
    /// Also deletes each flagged memory's `geo:` index entry, if it has one,
    /// in the same batch as the flag update — see `mark_forgotten_by_age`'s
    /// doc comment for the full rationale (one sweep, one write, no
    /// redundant re-fetch; this mirrors it exactly).
    pub fn mark_forgotten_by_importance(&self, threshold: f32) -> Result<Vec<MemoryId>> {
        let mut batch = rocksdb::WriteBatch::default();
        let mut flagged_ids = Vec::new();
        let now = Utc::now().to_rfc3339();
        let idx = self.index_cf();

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            if let Ok((mut memory, _)) = deserialize_memory_checked(&key, &value) {
                if memory.is_forgotten() {
                    continue;
                }
                if memory.importance() < threshold {
                    flagged_ids.push(memory.id.clone());
                    if let Some(geo) = memory.experience.geo_location {
                        let geohash = super::types::geohash_encode(geo[0], geo[1], 10);
                        let geo_key = format!("geo:{}:{}", geohash, memory.id.0);
                        batch.delete_cf(idx, geo_key.as_bytes());
                    }
                    memory
                        .experience
                        .metadata
                        .insert("forgotten".to_string(), "true".to_string());
                    memory
                        .experience
                        .metadata
                        .insert("forgotten_at".to_string(), now.clone());

                    let updated_value = crate::serialization::encode_sho(&memory)?;
                    batch.put(&key, updated_value);
                }
            }
        }

        if !flagged_ids.is_empty() {
            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(true);
            self.db.write_opt(batch, &write_opts)?;
        }

        Ok(flagged_ids)
    }

    /// Remove memories matching a pattern with durable writes
    pub fn remove_matching(&self, regex: &regex::Regex) -> Result<usize> {
        let mut count = 0;
        let mut to_delete: Vec<MemoryId> = Vec::new();

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }
            if let Ok((memory, _)) = deserialize_memory_checked(&key, &value) {
                if regex.is_match(&memory.experience.content) {
                    to_delete.push(memory.id);
                    count += 1;
                }
            }
        }

        // Delete each memory through the proper delete() path which cleans up indices first
        for memory_id in to_delete {
            if let Err(e) = self.delete(&memory_id) {
                tracing::warn!("Failed to delete matching memory {}: {}", memory_id.0, e);
            }
        }

        Ok(count)
    }

    /// Update access count for a memory
    pub fn update_access(&self, id: &MemoryId) -> Result<()> {
        if let Ok(memory) = self.get(id) {
            // ZERO-COPY: Update metadata through interior mutability
            memory.update_access();

            // Persist updated metadata
            self.update(&memory)?;
        }
        Ok(())
    }

    /// Get statistics about stored memories
    pub fn get_stats(&self) -> Result<StorageStats> {
        let mut stats = StorageStats::default();
        let mut raw_count = 0;
        let mut skipped_non_memory = 0;
        let mut deserialize_errors = 0;
        let stats_prefix = b"stats:";

        let iter = self.db.iterator(IteratorMode::Start);
        for item in iter {
            match item {
                Ok((key, value)) => {
                    raw_count += 1;

                    // Skip stats entries - they use a different format
                    if key.starts_with(stats_prefix) {
                        skipped_non_memory += 1;
                        continue;
                    }

                    // Valid memory keys should be exactly 16 bytes (UUID bytes)
                    if key.len() != 16 {
                        skipped_non_memory += 1;
                        continue;
                    }

                    match deserialize_memory_checked(&key, &value) {
                        Ok((memory, _)) => {
                            if memory.is_forgotten() {
                                continue;
                            }
                            stats.total_count += 1;
                            stats.total_size_bytes += value.len();
                            if memory.compressed {
                                stats.compressed_count += 1;
                            }
                            stats.importance_sum += memory.importance();
                        }
                        Err(e) => {
                            deserialize_errors += 1;
                            tracing::warn!(
                                "Corrupted memory entry (key len: {}, value len: {}): {}",
                                key.len(),
                                value.len(),
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Iterator error: {}", e);
                }
            }
        }

        tracing::debug!(
            "get_stats: raw_count={}, memories={}, skipped={}, corrupted={}",
            raw_count,
            stats.total_count,
            skipped_non_memory,
            deserialize_errors
        );

        if stats.total_count > 0 {
            stats.average_importance = stats.importance_sum / stats.total_count as f32;
        }

        // Load persisted retrieval counter
        stats.total_retrievals = self.get_retrieval_count().unwrap_or(0);

        Ok(stats)
    }

    /// Get the persisted retrieval counter
    pub fn get_retrieval_count(&self) -> Result<usize> {
        const RETRIEVAL_KEY: &[u8] = b"stats:total_retrievals";
        match self.db.get(RETRIEVAL_KEY)? {
            Some(data) => {
                if data.len() >= 8 {
                    Ok(usize::from_le_bytes(data[..8].try_into().unwrap_or([0; 8])))
                } else {
                    Ok(0)
                }
            }
            None => Ok(0),
        }
    }

    /// Increment and persist the retrieval counter, returns new value
    pub fn increment_retrieval_count(&self) -> Result<usize> {
        const RETRIEVAL_KEY: &[u8] = b"stats:total_retrievals";
        let current = self.get_retrieval_count().unwrap_or(0);
        let new_count = current + 1;
        self.db.put(RETRIEVAL_KEY, new_count.to_le_bytes())?;
        Ok(new_count)
    }

    /// Remove corrupted memories that fail to deserialize even with legacy fallbacks
    /// Returns the number of entries deleted
    ///
    /// This function safely cleans up:
    /// 1. Entries with keys that are not valid 16-byte UUIDs (corrupted/misplaced)
    /// 2. Entries with valid UUID keys but corrupted values that fail ALL format fallbacks
    ///
    /// It preserves:
    /// - Valid Memory entries (any format - current or legacy)
    /// - Stats entries (keys starting with "stats:")
    pub fn cleanup_corrupted(&self) -> Result<usize> {
        let mut to_delete = Vec::new();

        // Known non-memory prefixes in the default CF that must be preserved.
        // These are legitimate data entries stored by subsystems that share the
        // default column family via storage.db(): SemanticFactStore, TemporalFactStore,
        // LineageGraph, LearningHistoryStore, plus vector mappings and interference.
        let skip_prefixes: &[&[u8]] = &[
            b"stats:",
            b"vmapping:",
            b"interference:",
            b"interference_meta:",
            b"_watermark:",
            b"facts:",
            b"facts_by_entity:",
            b"facts_by_type:",
            b"facts_embedding:",
            b"temporal_facts:",
            b"temporal_by_time:",
            b"temporal_by_entity:",
            b"lineage:",
            b"learning:",
            b"geo:",
        ];

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            // Skip known non-memory prefixed entries
            if skip_prefixes.iter().any(|p| key.starts_with(p)) {
                continue;
            }

            // Valid memory keys should be exactly 16 bytes (UUID bytes)
            let is_valid_memory_key = key.len() == 16;

            if !is_valid_memory_key {
                tracing::debug!(
                    "Marking for deletion: invalid key length {} (expected 16)",
                    key.len()
                );
                to_delete.push(key.to_vec());
            } else if deserialize_memory(&value).is_err() {
                // Deliberately the UNCHECKED decode. This branch deletes, and a
                // record that decodes but disagrees with its key is exactly the
                // evidence an operator needs to see; turning that disagreement
                // into a delete would destroy it without anyone in the loop.
                // Undecodable records are a different matter — they are already
                // unreadable, so deleting them loses nothing that was readable.
                tracing::debug!(
                    "Marking for deletion: valid key but corrupted value ({} bytes)",
                    value.len()
                );
                to_delete.push(key.to_vec());
            }
        }

        let count = to_delete.len();
        if count > 0 {
            tracing::info!("Cleaning up {} corrupted memory entries", count);

            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(self.write_mode == WriteMode::Sync);

            for key in to_delete {
                if let Err(e) = self.db.delete_opt(&key, &write_opts) {
                    tracing::warn!("Failed to delete corrupted entry: {}", e);
                }
            }

            // Flush to persist deletions
            self.flush()?;
        }

        Ok(count)
    }

    /// Migrate legacy memories to current format for improved performance
    /// Returns (migrated_count, already_current_count, failed_count)
    ///
    /// This function:
    /// 1. Iterates all memories in storage
    /// 2. Attempts to deserialize with format fallback
    /// 3. Re-saves successfully deserialized legacy memories in current format
    /// 4. Reports migration statistics
    pub fn migrate_legacy(&self) -> Result<(usize, usize, usize)> {
        let mut migrated = 0;
        let mut already_current = 0;
        let mut failed = 0;
        let stats_prefix = b"stats:";

        let iter = self.db.iterator(IteratorMode::Start);
        let mut to_migrate = Vec::new();

        for (key, value) in iter.flatten() {
            // Skip stats entries
            if key.starts_with(stats_prefix) {
                continue;
            }

            // Skip non-UUID keys
            if key.len() != 16 {
                continue;
            }

            // "Current" for this migration means SHO v2 (postcard) — the format
            // every write path now produces. The old `deserialize_memory(..).is_ok()`
            // check was wrong: deserialize_memory returns Ok for legacy records
            // too (that is its whole purpose), so EVERY decodable record looked
            // "current" and nothing was ever migrated. The `needs_migration`
            // bool is also not a reliable discriminator here — its fallback path
            // reports raw bincode 2.x as "current" (false) even though that data
            // still needs converting to postcard. So gate on the SHO envelope
            // version directly.
            let is_current_postcard = matches!(
                crate::serialization::read_sho_envelope(&value),
                crate::serialization::ShoEnvelope::Valid {
                    version: crate::serialization::SHO_VERSION_POSTCARD,
                    ..
                }
            );
            if is_current_postcard {
                already_current += 1;
                continue;
            }

            // Anything else (SHO v1 bincode, raw bincode/msgpack) is legacy:
            // decode via the full fallback chain and queue it for re-encode to
            // postcard. A decode failure is counted as `failed`, never silently
            // skipped.
            // Checked: this loop re-encodes what it decodes and writes it
            // back OVER the original bytes at `key`. An unchecked pseudo-decode
            // here does not merely serve a fabrication, it persists one and
            // destroys the evidence of what the record used to be.
            match deserialize_memory_checked(&key, &value) {
                Ok((memory, _)) => {
                    to_migrate.push((key.to_vec(), memory));
                }
                Err(_) => {
                    failed += 1;
                }
            }
        }

        // Re-save migrated memories in current format
        if !to_migrate.is_empty() {
            tracing::info!(
                "Migrating {} legacy memories to current format",
                to_migrate.len()
            );

            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(self.write_mode == WriteMode::Sync);

            for (key, memory) in to_migrate {
                match crate::serialization::encode_sho(&memory) {
                    Ok(serialized) => {
                        if let Err(e) = self.db.put_opt(&key, &serialized, &write_opts) {
                            tracing::warn!("Failed to migrate memory: {e}");
                            failed += 1;
                        } else {
                            migrated += 1;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to serialize migrated memory: {e}");
                        failed += 1;
                    }
                }
            }

            // Flush to persist migrations
            self.flush()?;
        }

        tracing::info!(
            "Migration complete: {} migrated, {} already current, {} failed",
            migrated,
            already_current,
            failed
        );

        Ok((migrated, already_current, failed))
    }

    /// Flush all column families to ensure data is persisted (critical for graceful shutdown)
    /// RocksDB in-process memory for this storage DB: (memtable bytes,
    /// table-reader bytes), summed over its column families. Block cache is
    /// NOT included here — it is shared across all DBs and reported once at
    /// the manager level (summing it per-CF would count the same pool many
    /// times over).
    pub fn rocksdb_memory_breakdown(&self) -> (u64, u64) {
        let mut memtables = 0u64;
        let mut readers = 0u64;
        for cf_name in ["default", CF_INDEX, CF_OPLOG] {
            if let Some(cf) = self.db.cf_handle(cf_name) {
                if let Ok(Some(v)) = self
                    .db
                    .property_int_value_cf(cf, "rocksdb.cur-size-all-mem-tables")
                {
                    memtables += v;
                }
                if let Ok(Some(v)) = self
                    .db
                    .property_int_value_cf(cf, "rocksdb.estimate-table-readers-mem")
                {
                    readers += v;
                }
            }
        }
        (memtables, readers)
    }

    pub fn flush(&self) -> Result<()> {
        use rocksdb::FlushOptions;

        let mut flush_opts = FlushOptions::default();
        flush_opts.set_wait(true); // Block until flush is complete

        // Single DB flush covers both default and index CFs
        self.db
            .flush_opt(&flush_opts)
            .map_err(|e| anyhow::anyhow!("Failed to flush memory storage: {e}"))?;

        // Explicitly flush the index CF (RocksDB flush_opt only flushes default CF)
        self.db
            .flush_cf_opt(self.index_cf(), &flush_opts)
            .map_err(|e| anyhow::anyhow!("Failed to flush index CF: {e}"))?;

        // Explicitly flush the oplog CF for the same reason (audit Finding H).
        self.db
            .flush_cf_opt(self.oplog_cf(), &flush_opts)
            .map_err(|e| anyhow::anyhow!("Failed to flush oplog CF: {e}"))?;

        Ok(())
    }

    /// Get a reference to the underlying RocksDB instance
    ///
    /// Used by SemanticFactStore to share the same database for fact storage.
    /// Facts use a different key prefix ("facts:") to avoid collisions.
    pub fn db(&self) -> Arc<DB> {
        self.db.clone()
    }

    // ========================================================================
    // Agent-traceability oplog (spec
    // docs/superpowers/specs/2026-07-30-agent-traceability-design.md §3.1-3.2)
    // ========================================================================

    /// Appends `draft` to its session's hash chain and persists it in
    /// [`CF_OPLOG`], returning the sealed record (`seq`/`prev_hash`/`hash`
    /// filled in). See [`CF_OPLOG`]'s doc comment for the key scheme.
    ///
    /// Atomic per session: reads the cached head, builds `seq =
    /// head.last_seq + 1` (or `0` against [`super::oplog::genesis_hash`] if
    /// this is the session's first record), then writes the record and the
    /// updated head in one `WriteBatch`.
    ///
    /// Concurrency: guarded by `oplog_append_lock` for the whole
    /// head-read → batch-write span. Storage methods are NOT externally
    /// serialized — handlers take `.read()` guards on `MemorySystem`
    /// (audit `2026-07-30-traceability-slice1-audit.md` Finding G) — so
    /// without this lock, two concurrent appends to the same session could
    /// both read the same head and produce duplicate `seq` values / forked
    /// chains.
    ///
    /// Durability: inherits `self.write_mode` exactly like every other
    /// storage write (audit amendment 6, resolving a constraint conflict
    /// with Task 5's <100 µs median append gate) — appends are NOT forced
    /// to `set_sync(true)`. See `oplog.rs`'s module docs for the resulting
    /// crash window.
    pub fn oplog_append(&self, draft: OpRecordDraft) -> Result<OpRecord> {
        crate::validation::validate_session_id(&draft.session_id)
            .context("oplog_append: invalid session_id")?;
        // oplog.rs's DOMAIN_SEP invariant ("0x00 cannot occur in a validated
        // session_id/user_id") covers user_id too, and user_id is
        // self-asserted (see OpRecord::user_id's doc comment) — an
        // unvalidated value with an embedded NUL/control char or unbounded
        // length would land verbatim in the permanent tamper-evident record
        // and break that guarantee.
        crate::validation::validate_user_id(&draft.user_id)
            .context("oplog_append: invalid user_id")?;

        let OpRecordDraft {
            ts,
            session_id,
            user_id,
            op,
            attestation,
            payload_summary,
            evidence_refs,
            outcome,
            reported_ts,
            source,
        } = draft;

        // Guard the whole head-read -> batch-write span (see method docs).
        let _guard = self.oplog_append_lock.lock();
        let cf = self.oplog_cf();
        let head_key = format!("{OPLOG_HEAD_PREFIX}{session_id}");

        let (seq, prev_hash) = match self
            .db
            .get_cf(cf, head_key.as_bytes())
            .context("oplog_append: reading session head")?
        {
            Some(bytes) => {
                let head: OplogHead = serde_json::from_slice(&bytes)
                    .context("oplog_append: decoding session head")?;
                (head.last_seq + 1, head.last_hash)
            }
            None => (0u64, oplog::genesis_hash(&session_id, &user_id)),
        };

        let mut record = OpRecord {
            seq,
            ts,
            session_id: session_id.clone(),
            user_id,
            op,
            attestation,
            payload_summary,
            evidence_refs,
            outcome,
            reported_ts,
            source,
            prev_hash: prev_hash.clone(),
            hash: String::new(),
        };
        record.hash = oplog::chain_hash(&prev_hash, &oplog::canonical_bytes(&record));

        let record_key = format!("{OPLOG_RECORD_PREFIX}{session_id}:{seq:016x}");
        // serde_json, matching oplog.rs's canonical-bytes format: the
        // persistence round-trip invariant (oplog.rs module docs) requires
        // any storage format to restore every field bit-exactly, and
        // serde_json over `OpRecord` is the one already proven to do so
        // (`serde_round_trip_preserves_verification`).
        let record_bytes =
            serde_json::to_vec(&record).context("oplog_append: serializing record")?;

        let new_head = OplogHead {
            last_seq: seq,
            last_hash: record.hash.clone(),
            last_ts: record.ts,
        };
        let head_bytes =
            serde_json::to_vec(&new_head).context("oplog_append: serializing session head")?;

        let mut batch = WriteBatch::default();
        batch.put_cf(cf, record_key.as_bytes(), &record_bytes);
        batch.put_cf(cf, head_key.as_bytes(), &head_bytes);

        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .write_opt(batch, &write_opts)
            .context("oplog_append: writing record + head batch")?;

        Ok(record)
    }

    /// Reads `session_id`'s records starting at `from_seq` (inclusive), in
    /// ascending `seq` order, capped at `limit`.
    pub fn oplog_read(
        &self,
        session_id: &str,
        from_seq: u64,
        limit: usize,
    ) -> Result<Vec<OpRecord>> {
        crate::validation::validate_session_id(session_id)
            .context("oplog_read: invalid session_id")?;
        if limit == 0 {
            return Ok(Vec::new());
        }

        let cf = self.oplog_cf();
        let prefix = format!("{OPLOG_RECORD_PREFIX}{session_id}:");
        let start_key = format!("{OPLOG_RECORD_PREFIX}{session_id}:{from_seq:016x}");

        let iter = self.db.iterator_cf(
            cf,
            IteratorMode::From(start_key.as_bytes(), rocksdb::Direction::Forward),
        );

        let mut records = Vec::new();
        for item in iter {
            let (key, value) = item.context("oplog_read: iterator error")?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            let record: OpRecord =
                serde_json::from_slice(&value).context("oplog_read: decoding record")?;
            records.push(record);
            if records.len() >= limit {
                break;
            }
        }
        Ok(records)
    }

    /// Sets the `incomplete:{session_id}` integrity flag. NEVER touches
    /// `op:`/`head:` keys — the flag is a separate key prefix precisely so
    /// this can never rewrite or delete a sealed record (audit amendment 15:
    /// this is the one intentional, out-of-band oplog write site, used by
    /// backup restore to mark sessions whose tail was rolled back).
    pub fn oplog_mark_incomplete(&self, session_id: &str) -> Result<()> {
        crate::validation::validate_session_id(session_id)
            .context("oplog_mark_incomplete: invalid session_id")?;
        let key = format!("{OPLOG_INCOMPLETE_PREFIX}{session_id}");

        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .put_cf_opt(self.oplog_cf(), key.as_bytes(), b"1", &write_opts)
            .context("oplog_mark_incomplete: writing flag")?;
        Ok(())
    }

    /// Whether `session_id`'s integrity flag is set (see
    /// [`Self::oplog_mark_incomplete`]).
    pub fn oplog_is_incomplete(&self, session_id: &str) -> Result<bool> {
        crate::validation::validate_session_id(session_id)
            .context("oplog_is_incomplete: invalid session_id")?;
        let key = format!("{OPLOG_INCOMPLETE_PREFIX}{session_id}");
        Ok(self
            .db
            .get_cf(self.oplog_cf(), key.as_bytes())
            .context("oplog_is_incomplete: reading flag")?
            .is_some())
    }

    /// Lists distinct session ids with at least one oplog record,
    /// newest-first by head write time (the `ts` of each session's most
    /// recent append; `session_id` breaks ties for determinism).
    ///
    /// No `user_id` filter (audit amendment 7): `MemoryStorage` is already
    /// per-user (`{base}/{user_id}/storage`), so a `user_id` parameter here
    /// would be vestigial and would invite a false cross-user-listing
    /// expectation.
    pub fn oplog_sessions(&self, limit: usize, offset: usize) -> Result<Vec<String>> {
        let cf = self.oplog_cf();
        let iter = self.db.iterator_cf(
            cf,
            IteratorMode::From(OPLOG_HEAD_PREFIX.as_bytes(), rocksdb::Direction::Forward),
        );

        let mut sessions: Vec<(String, DateTime<Utc>)> = Vec::new();
        for item in iter {
            let (key, value) = item.context("oplog_sessions: iterator error")?;
            if !key.starts_with(OPLOG_HEAD_PREFIX.as_bytes()) {
                break;
            }
            let session_id = String::from_utf8_lossy(&key[OPLOG_HEAD_PREFIX.len()..]).to_string();
            let head: OplogHead =
                serde_json::from_slice(&value).context("oplog_sessions: decoding session head")?;
            sessions.push((session_id, head.last_ts));
        }

        sessions.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        Ok(sessions
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(id, _)| id)
            .collect())
    }
}

/// Search criteria for memory retrieval
#[derive(Debug, Clone)]
pub enum SearchCriteria {
    // === Standard Criteria ===
    ByDate {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    ByType(ExperienceType),
    ByImportance {
        min: f32,
        max: f32,
    },
    ByEntity(String),
    /// Filter by tags (matches memories containing ANY of these tags)
    ByTags(Vec<String>),

    // === Temporal/Episode Criteria ===
    /// Filter by episode ID - memories in the same episode are highly related
    ByEpisode(String),
    /// Filter by episode with sequence ordering - returns memories in temporal order
    ByEpisodeSequence {
        episode_id: String,
        /// If provided, only return memories with sequence >= this value
        min_sequence: Option<u32>,
        /// If provided, only return memories with sequence <= this value
        max_sequence: Option<u32>,
    },

    // === Robotics Criteria ===
    /// Filter by robot/drone identifier
    ByRobot(String),
    /// Filter by mission identifier
    ByMission(String),
    /// Spatial filter: memories within radius of (lat, lon)
    ByLocation {
        lat: f64,
        lon: f64,
        radius_meters: f64,
        /// Optional cap on the number of matching ids returned, nearest-first
        /// by geohash-decoded approximate distance (tie-break MemoryId).
        /// Bounding happens INSIDE `search_by_location`, before any
        /// hydration — capping post-hydration is too late, since hydration
        /// (get + deserialize) is the expensive part. `None` preserves the
        /// historical uncapped behavior (used by robotics `spatial_search`,
        /// which already re-sorts/re-filters/truncates the fully hydrated
        /// result itself).
        limit: Option<usize>,
    },
    /// Filter by action type
    ByActionType(String),
    /// Filter by reward range (for RL-style queries)
    ByReward {
        min: f32,
        max: f32,
    },

    // === Compound Criteria ===
    Combined(Vec<SearchCriteria>),

    // === Hierarchy Criteria ===
    /// Filter by parent memory ID - returns all children of a memory
    ByParent(MemoryId),
    /// Filter for root memories (no parent)
    RootsOnly,
}

/// Storage statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_count: usize,
    pub compressed_count: usize,
    pub total_size_bytes: usize,
    pub average_importance: f32,
    pub importance_sum: f32,
    /// Total number of recall/retrieval operations (persisted)
    #[serde(default)]
    pub total_retrievals: usize,
}

// =============================================================================
// ATOMIC VECTOR INDEX MAPPING STORAGE
// =============================================================================
//
// This module provides atomic storage for vector index mappings alongside memory data.
// By storing IdMapping in RocksDB (not separate files), we ensure:
//
// 1. ATOMIC WRITES: Memory + vector mapping written in single WriteBatch
// 2. NO ORPHANS: If memory exists, its vector mapping exists (or can be rebuilt)
// 3. CRASH SAFETY: RocksDB WAL protects both memory data and mappings
// 4. SINGLE SOURCE OF TRUTH: RocksDB is THE authority, Vamana is just a cache
//
// MULTIMODALITY READY:
// - Each modality (text, image, audio, video) has separate vector space
// - Text: 384-dim MiniLM (current)
// - Image: 1024-dim ImageBind (future)
// - Audio: 1024-dim ImageBind (future)
// - Video: 1024-dim ImageBind (future)
// - Cross-modal search possible via ImageBind's unified embedding space
//
// Key format: "vmapping:{memory_id}" -> bincode(VectorMappingEntry)
// =============================================================================

/// Supported embedding modalities
///
/// When adding a new modality:
/// 1. Add variant here
/// 2. Create corresponding Vamana index with correct dimension
/// 3. Implement embedder for the modality
/// 4. Update search to include the modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text embeddings (MiniLM-L6-v2, 384-dim)
    Text,
    /// Image embeddings (future: ImageBind, 1024-dim)
    Image,
    /// Audio embeddings (future: ImageBind, 1024-dim)
    Audio,
    /// Video embeddings (future: ImageBind, 1024-dim)
    Video,
    /// Multi-modal unified embeddings (future: ImageBind, 1024-dim)
    /// Used when content has multiple modalities fused together
    Unified,
}

impl Modality {
    /// Get embedding dimension for this modality
    pub fn dimension(&self) -> usize {
        match self {
            // Single source of truth: 384 by default (MiniLM/bge/gte/mxbai and
            // nomic-Matryoshka), 768 when SHODH_TEXT_DIM=768 (native nomic).
            Modality::Text => crate::embeddings::minilm::configured_text_dim(),
            // ImageBind projects all modalities to 1024-dim shared space
            Modality::Image => 1024,
            Modality::Audio => 1024,
            Modality::Video => 1024,
            Modality::Unified => 1024,
        }
    }

    /// Get the string key for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
            Modality::Unified => "unified",
        }
    }
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Vector IDs for a specific modality
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModalityVectors {
    /// Vector IDs in this modality's Vamana index
    pub vector_ids: Vec<u32>,
    /// Embedding dimension (for validation)
    pub dimension: usize,
    /// Chunk boundaries (for long content)
    /// Each entry is (start_char, end_char) in original content
    pub chunk_ranges: Option<Vec<(usize, usize)>>,
}

/// Vector mapping entry for a single memory - MULTIMODALITY READY
///
/// Stores vector IDs for each modality separately, allowing:
/// - Text-only memories (current)
/// - Image-only memories (future)
/// - Multi-modal memories (text + image + audio)
/// - Cross-modal search via unified embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMappingEntry {
    /// Vector IDs per modality
    /// Key: Modality enum (serializes as string)
    /// Value: Vector IDs + metadata for that modality
    pub modalities: HashMap<Modality, ModalityVectors>,
    /// Timestamp when mapping was created (for debugging)
    pub created_at: i64,
    /// Schema version for forward compatibility
    pub version: u8,
}

impl Default for VectorMappingEntry {
    fn default() -> Self {
        Self {
            modalities: HashMap::new(),
            created_at: chrono::Utc::now().timestamp_millis(),
            version: 1,
        }
    }
}

impl VectorMappingEntry {
    /// Create a new mapping with text vectors (most common case)
    pub fn with_text(vector_ids: Vec<u32>) -> Self {
        let mut modalities = HashMap::new();
        modalities.insert(
            Modality::Text,
            ModalityVectors {
                vector_ids,
                dimension: Modality::Text.dimension(),
                chunk_ranges: None,
            },
        );
        Self {
            modalities,
            created_at: chrono::Utc::now().timestamp_millis(),
            version: 1,
        }
    }

    /// Get text vector IDs (convenience method for current text-only usage)
    pub fn text_vectors(&self) -> Option<&Vec<u32>> {
        self.modalities.get(&Modality::Text).map(|m| &m.vector_ids)
    }

    /// Get all vector IDs across all modalities (for deletion)
    pub fn all_vector_ids(&self) -> Vec<(Modality, u32)> {
        self.modalities
            .iter()
            .flat_map(|(modality, mv)| mv.vector_ids.iter().map(|id| (*modality, *id)))
            .collect()
    }

    /// Check if this entry has any vectors
    pub fn is_empty(&self) -> bool {
        self.modalities.values().all(|mv| mv.vector_ids.is_empty())
    }

    /// Add vectors for a modality
    pub fn add_modality(&mut self, modality: Modality, vector_ids: Vec<u32>) {
        self.modalities.insert(
            modality,
            ModalityVectors {
                dimension: modality.dimension(),
                vector_ids,
                chunk_ranges: None,
            },
        );
    }

    /// Add image vectors
    pub fn with_image(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Image, vector_ids);
        self
    }

    /// Add audio vectors
    pub fn with_audio(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Audio, vector_ids);
        self
    }

    /// Add video vectors
    pub fn with_video(mut self, vector_ids: Vec<u32>) -> Self {
        self.add_modality(Modality::Video, vector_ids);
        self
    }
}

impl MemoryStorage {
    // =========================================================================
    // ATOMIC VECTOR MAPPING OPERATIONS
    // =========================================================================

    /// Store memory and its text vector mapping atomically
    ///
    /// Uses WriteBatch to ensure both operations succeed or both fail.
    /// This is the ONLY way orphaned memories can be prevented.
    ///
    /// For text-only memories (current implementation). Use store_with_multimodal_vectors
    /// for memories with image/audio/video content.
    pub fn store_with_vectors(&self, memory: &Memory, vector_ids: Vec<u32>) -> Result<()> {
        self.store_with_multimodal_vectors(memory, Modality::Text, vector_ids)
    }

    /// Store memory with vectors for a specific modality
    ///
    /// MULTIMODALITY READY: Supports text, image, audio, video modalities.
    /// Each modality is stored separately, allowing cross-modal search.
    pub fn store_with_multimodal_vectors(
        &self,
        memory: &Memory,
        modality: Modality,
        vector_ids: Vec<u32>,
    ) -> Result<()> {
        let mut batch = WriteBatch::default();

        // 1. Serialize memory
        let memory_key = memory.id.0.as_bytes();
        let memory_value = crate::serialization::encode_sho(memory)
            .context(format!("Failed to serialize memory {}", memory.id.0))?;
        batch.put(memory_key, &memory_value);

        // 2. Serialize vector mapping with modality support
        let mapping_key = format!("vmapping:{}", memory.id.0);

        // Load existing mapping (for adding new modality to existing memory)
        let mut mapping_entry = self.get_vector_mapping(&memory.id)?.unwrap_or_default();

        // Add/update the modality vectors
        mapping_entry.add_modality(modality, vector_ids);

        let mapping_value = crate::serialization::encode(&mapping_entry)
            .context("Failed to serialize vector mapping")?;
        batch.put(mapping_key.as_bytes(), &mapping_value);

        // 3. Atomic write - both succeed or both fail
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .write_opt(batch, &write_opts)
            .context("Atomic write of memory + vector mapping failed")?;

        // 4. Update secondary indices (separate operation, but non-critical)
        if let Err(e) = self.update_indices(memory) {
            tracing::warn!("Secondary index update failed (non-fatal): {}", e);
        }

        Ok(())
    }

    /// Get vector mapping for a memory
    pub fn get_vector_mapping(&self, memory_id: &MemoryId) -> Result<Option<VectorMappingEntry>> {
        let mapping_key = format!("vmapping:{}", memory_id.0);
        match self.db.get(mapping_key.as_bytes())? {
            Some(data) => {
                let (entry, _) = crate::serialization::try_decode::<VectorMappingEntry>(&data)
                    .context("Failed to deserialize vector mapping")?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    /// Get all vector mappings (for rebuilding Vamana index on startup)
    ///
    /// Returns iterator-style results to avoid loading everything into memory at once.
    /// Sorted by memory_id for deterministic Vamana rebuilding.
    pub fn get_all_vector_mappings(&self) -> Result<Vec<(MemoryId, VectorMappingEntry)>> {
        let mut mappings = Vec::new();
        let prefix = b"vmapping:";

        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        for item in iter {
            match item {
                Ok((key, value)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if !key_str.starts_with("vmapping:") {
                        break;
                    }

                    // Extract memory_id from key
                    if let Some(id_str) = key_str.strip_prefix("vmapping:") {
                        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                            if let Ok((entry, _)) =
                                crate::serialization::try_decode::<VectorMappingEntry>(&value)
                            {
                                mappings.push((MemoryId(uuid), entry));
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Error reading vector mapping: {}", e);
                }
            }
        }

        Ok(mappings)
    }

    /// Delete vector mapping for a memory (called when deleting memory)
    pub fn delete_vector_mapping(&self, memory_id: &MemoryId) -> Result<()> {
        let mapping_key = format!("vmapping:{}", memory_id.0);
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db.delete_opt(mapping_key.as_bytes(), &write_opts)?;
        Ok(())
    }

    /// Update text vector mapping for a memory (for reindex operations)
    ///
    /// Convenience method for text-only reindexing.
    pub fn update_vector_mapping(&self, memory_id: &MemoryId, vector_ids: Vec<u32>) -> Result<()> {
        self.update_modality_vectors(memory_id, Modality::Text, vector_ids)
    }

    /// Update vector mapping for a specific modality
    ///
    /// MULTIMODALITY READY: Preserves vectors for other modalities while updating one.
    pub fn update_modality_vectors(
        &self,
        memory_id: &MemoryId,
        modality: Modality,
        vector_ids: Vec<u32>,
    ) -> Result<()> {
        let mapping_key = format!("vmapping:{}", memory_id.0);

        // Load existing mapping to preserve other modalities
        let mut mapping_entry = self.get_vector_mapping(memory_id)?.unwrap_or_default();

        // Update the specific modality
        mapping_entry.add_modality(modality, vector_ids);

        let mapping_value = crate::serialization::encode(&mapping_entry)?;

        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .put_opt(mapping_key.as_bytes(), &mapping_value, &write_opts)?;
        Ok(())
    }

    /// Delete memory and its vector mapping atomically
    pub fn delete_with_vectors(&self, id: &MemoryId) -> Result<()> {
        let mut batch = WriteBatch::default();

        // 1. Delete memory
        batch.delete(id.0.as_bytes());

        // 2. Delete vector mapping
        let mapping_key = format!("vmapping:{}", id.0);
        batch.delete(mapping_key.as_bytes());

        // 3. Atomic delete
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db.write_opt(batch, &write_opts)?;

        // 4. Clean up indices (non-critical)
        if let Err(e) = self.remove_from_indices(id) {
            tracing::warn!("Index cleanup failed (non-fatal): {}", e);
        }

        Ok(())
    }

    /// Count memories with vector mappings (for health checks)
    pub fn count_vector_mappings(&self) -> usize {
        let prefix = b"vmapping:";
        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        let mut count = 0;
        for (key, _) in iter.flatten() {
            if key.starts_with(prefix) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Check integrity: find memories without vector mappings
    ///
    /// Returns memories that have embeddings but no corresponding vector mapping.
    /// These need to be reindexed.
    pub fn find_memories_without_mappings(&self) -> Result<Vec<MemoryId>> {
        let mut orphans = Vec::new();

        let iter = self.db.iterator(IteratorMode::Start);
        for (key, value) in iter.flatten() {
            if key.len() != 16 {
                continue;
            }

            if let Ok((memory, _)) = deserialize_memory_checked(&key, &value) {
                let has_mapping = match self.get_vector_mapping(&memory.id) {
                    Ok(Some(entry)) => entry.text_vectors().is_some_and(|v| !v.is_empty()),
                    _ => false,
                };

                if !has_mapping && memory.experience.embeddings.is_some() {
                    orphans.push(memory.id);
                }
            }
        }

        Ok(orphans)
    }

    /// Get all text vector IDs from mappings (for Vamana statistics)
    pub fn get_all_text_vector_ids(&self) -> Result<Vec<u32>> {
        let mut all_ids = Vec::new();
        let mappings = self.get_all_vector_mappings()?;

        for (_, entry) in mappings {
            if let Some(text_vecs) = entry.text_vectors() {
                all_ids.extend(text_vecs.iter().copied());
            }
        }

        Ok(all_ids)
    }

    /// Get vector count per modality (for health monitoring)
    pub fn get_modality_stats(&self) -> Result<HashMap<Modality, usize>> {
        let mut stats: HashMap<Modality, usize> = HashMap::new();
        let mappings = self.get_all_vector_mappings()?;

        for (_, entry) in mappings {
            for (modality, mv) in entry.modalities {
                *stats.entry(modality).or_insert(0) += mv.vector_ids.len();
            }
        }

        Ok(stats)
    }

    // =========================================================================
    // INTERFERENCE PERSISTENCE (SHO-106 RIF)
    // =========================================================================
    //
    // Persists InterferenceDetector state to the main RocksDB database using
    // key prefix "interference:{memory_id}" for per-memory records and
    // "interference_meta:total" for the aggregate event counter.
    //
    // This ensures retrieval-induced forgetting history survives server restarts.
    // =========================================================================

    /// Persist interference records for a single memory
    ///
    /// Key format: `interference:{memory_id}` → JSON `Vec<InterferenceRecord>`
    pub fn save_interference_records(
        &self,
        memory_id: &str,
        records: &[super::replay::InterferenceRecord],
    ) -> Result<()> {
        let key = format!("interference:{memory_id}");
        let value =
            serde_json::to_vec(records).context("Failed to serialize interference records")?;

        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .put_opt(key.as_bytes(), &value, &write_opts)
            .context("Failed to persist interference records")?;

        Ok(())
    }

    /// Load all interference records from storage on startup
    ///
    /// Scans all `interference:` prefixed keys and deserializes records.
    /// Returns `(history_map, total_event_count)` for bulk-loading into InterferenceDetector.
    pub fn load_all_interference_records(
        &self,
    ) -> Result<(
        HashMap<String, Vec<super::replay::InterferenceRecord>>,
        usize,
    )> {
        let prefix = b"interference:";
        let mut history: HashMap<String, Vec<super::replay::InterferenceRecord>> = HashMap::new();
        let mut total_events: usize = 0;

        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        for item in iter.log_errors() {
            let (key, value) = item;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with("interference:") {
                break;
            }

            if let Some(memory_id) = key_str.strip_prefix("interference:") {
                match serde_json::from_slice::<Vec<super::replay::InterferenceRecord>>(&value) {
                    Ok(records) => {
                        total_events += records.len();
                        history.insert(memory_id.to_string(), records);
                    }
                    Err(e) => {
                        tracing::warn!(
                            key = %key_str,
                            error = %e,
                            "Failed to deserialize interference records, skipping"
                        );
                    }
                }
            }
        }

        // Load persisted total count (may be higher than sum of records due to eviction)
        let persisted_total = self
            .db
            .get(b"interference_meta:total")
            .ok()
            .flatten()
            .and_then(|v| {
                if v.len() == 8 {
                    Some(u64::from_le_bytes(v[..8].try_into().unwrap()) as usize)
                } else {
                    None
                }
            })
            .unwrap_or(total_events);

        Ok((history, persisted_total.max(total_events)))
    }

    /// Delete interference records for a single memory (called on forget/delete)
    pub fn delete_interference_records(&self, memory_id: &str) -> Result<()> {
        let key = format!("interference:{memory_id}");
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .delete_opt(key.as_bytes(), &write_opts)
            .context("Failed to delete interference records")?;
        Ok(())
    }

    /// Persist the total interference event count
    ///
    /// Key: `interference_meta:total` → 8-byte little-endian u64
    pub fn save_interference_event_count(&self, count: usize) -> Result<()> {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(self.write_mode == WriteMode::Sync);
        self.db
            .put_opt(
                b"interference_meta:total",
                (count as u64).to_le_bytes(),
                &write_opts,
            )
            .context("Failed to persist interference event count")?;
        Ok(())
    }

    // NOTE on `_watermark:fact_extraction:{user_id}` keys: the incremental
    // fact-extraction watermark was removed (it silently excluded every
    // age-ineligible memory from consolidation forever — the zero-facts
    // defect). Persisted keys may still exist in older stores; they are
    // simply never read again. The `_watermark:` prefix stays reserved in
    // the system-key lists (see `migration.rs`) so residue is skipped.

    /// Delete ALL interference records (GDPR forget_all)
    ///
    /// Batch-deletes all `interference:` and `interference_meta:` keys.
    pub fn clear_all_interference_records(&self) -> Result<usize> {
        let prefix = b"interference";
        let mut batch = WriteBatch::default();
        let mut count = 0;

        let iter = self
            .db
            .iterator(IteratorMode::From(prefix, rocksdb::Direction::Forward));

        for item in iter.log_errors() {
            let (key, _) = item;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with("interference") {
                break;
            }
            batch.delete(&key);
            count += 1;
        }

        if count > 0 {
            let mut write_opts = WriteOptions::default();
            write_opts.set_sync(self.write_mode == WriteMode::Sync);
            self.db.write_opt(batch, &write_opts)?;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct LegacyMinimalFixture {
        id: MemoryId,
        content: String,
    }

    fn sample_memory(id: MemoryId, content: &str) -> Memory {
        let now = Utc::now();
        let experience = Experience {
            experience_type: ExperienceType::Observation,
            content: content.to_string(),
            ..Default::default()
        };
        Memory::from_legacy(
            id,
            experience,
            0.5,
            0,
            now,
            now,
            false,
            MemoryTier::LongTerm,
            Vec::new(),
            1.0,
            None,
            None,
            None,
            None,
            0.0,
            None,
            None,
            1,
            Vec::new(),
            Vec::new(),
        )
    }

    // -----------------------------------------------------------------------
    // A failed checksum must not be laundered into a decode attempt.
    //
    // These bytes are constructed so the fallback chain PROVABLY fabricates a
    // memory from them: printable ASCII end to end, so `try_raw_memory_parse`
    // reads the first sixteen bytes as a UUID and the rest as content. If the
    // test flipped a bit in a real record instead, the fallback might fail for
    // an unrelated reason and the "it used to succeed" premise would hold only
    // by luck.
    // -----------------------------------------------------------------------

    /// A complete SHO v2 envelope whose CRC32 trailer is `XXXX` — wrong by
    /// construction — around an all-printable payload.
    fn envelope_with_a_broken_checksum() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(crate::memory::storage::STORAGE_MAGIC);
        bytes.push(crate::serialization::SHO_VERSION_POSTCARD);
        bytes.extend_from_slice(b"corrupted-record-payload-bytes");
        bytes.extend_from_slice(b"XXXX");
        bytes
    }

    #[test]
    fn a_failed_checksum_is_an_error_not_a_trip_through_the_fallback_chain() {
        let bytes = envelope_with_a_broken_checksum();

        // Premise 1: the envelope really is corrupt, not merely unparsed.
        assert!(
            matches!(
                crate::serialization::read_sho_envelope(&bytes),
                crate::serialization::ShoEnvelope::ChecksumMismatch { .. }
            ),
            "fixture must present a complete envelope with a wrong checksum"
        );

        // Premise 2, and the whole reason this defect mattered: the fallback
        // chain these bytes used to be handed to does NOT reject them. It
        // invents a Memory. Asserted permanently — if this ever starts failing,
        // the fabrication has been fixed at its source and this test needs
        // revisiting, but it does not mean the discrimination below is wrong.
        let fabricated = deserialize_with_fallback(&bytes);
        assert!(
            fabricated.is_ok(),
            "premise: the legacy chain fabricates a Memory from these bytes; \
             it errored instead: {:?}",
            fabricated.err()
        );

        // The fix: a record whose checksum fails never reaches that chain.
        let err = deserialize_memory(&bytes)
            .expect_err("a record with a failed checksum must be a hard error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("checksum mismatch"),
            "the error must name the checksum, so it cannot pass by way of some \
             unrelated decode failure; got: {msg}"
        );
    }

    #[test]
    fn a_record_with_no_envelope_still_decodes_through_the_legacy_chain() {
        // The other half of the discrimination, and the reason it is a
        // discrimination rather than a removal: every record written before the
        // SHO cutover carries no envelope, and must keep decoding.
        let id = MemoryId(
            uuid::Uuid::parse_str("6f0b1c2d-3e4f-4a5b-8c9d-0e1f2a3b4c5d").expect("static uuid"),
        );
        let memory = sample_memory(id.clone(), "written before the envelope existed");
        let legacy_bytes =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).expect("encode");

        assert_eq!(
            crate::serialization::read_sho_envelope(&legacy_bytes),
            crate::serialization::ShoEnvelope::Absent,
            "fixture must carry no envelope at all"
        );

        let (decoded, needs_migration) = deserialize_memory(&legacy_bytes)
            .expect("a pre-envelope record must still decode via the legacy chain");
        assert_eq!(decoded.id, id);
        assert_eq!(
            decoded.experience.content,
            "written before the envelope existed"
        );
        // Pinned as-is, not as it ought to be: the fallback chain reports raw
        // bincode 2.x of the CURRENT struct shape as needing no migration, even
        // though it still wants re-encoding as postcard. That is the wart
        // `migrate_legacy` documents and works around by gating on the envelope
        // version rather than on this flag. Asserting it here keeps the two
        // statements consistent.
        assert!(
            !needs_migration,
            "raw bincode 2.x of the current shape is reported as current"
        );

        // A genuinely older shape, from a deeper branch of the same chain, so
        // the guarantee is not resting on the one branch nearest the surface.
        let minimal = rmp_serde::to_vec(&LegacyMinimalFixture {
            id: id.clone(),
            content: "an even older record".to_string(),
        })
        .expect("msgpack");
        assert_eq!(
            crate::serialization::read_sho_envelope(&minimal),
            crate::serialization::ShoEnvelope::Absent
        );
        let (from_minimal, needs_migration) =
            deserialize_memory(&minimal).expect("a deep-legacy record must still decode");
        assert_eq!(from_minimal.id, id);
        assert_eq!(from_minimal.experience.content, "an even older record");
        assert!(
            needs_migration,
            "a deep-legacy record must be marked for rewrite"
        );
    }

    #[test]
    fn legacy_wire_formats_never_begin_with_the_sho_magic() {
        // The premise under "absent vs corrupt": no pre-envelope encoding can
        // be mistaken for a damaged envelope, because none of them start with
        // an ASCII S. Both legacy encoders length-prefix the leading UUID, so
        // byte 0 is a small integer, never 0x53. Pinned rather than reasoned
        // about, because the whole discrimination rests on it.
        // 53 48 4f = "SHO": the worst case for the magic test.
        let id = MemoryId(
            uuid::Uuid::parse_str("53484f01-0000-4000-8000-000000000000").expect("static uuid"),
        );
        let memory = sample_memory(id, "a uuid whose first three bytes ARE the magic");

        let bincode2 =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).expect("bincode");
        let msgpack = rmp_serde::to_vec(&memory).expect("msgpack");

        for (name, bytes) in [("bincode2", &bincode2), ("msgpack", &msgpack)] {
            assert_ne!(
                &bytes[0..3],
                &crate::memory::storage::STORAGE_MAGIC[..],
                "{name} encoding of a memory whose id starts with 53 48 4f must \
                 still not begin with the SHO magic, or absent-vs-corrupt has a \
                 false-positive population"
            );
            assert_eq!(
                crate::serialization::read_sho_envelope(bytes),
                crate::serialization::ShoEnvelope::Absent,
                "{name} record must read as Absent"
            );
        }
    }

    #[test]
    fn a_pseudo_decode_is_refused_and_its_bytes_are_left_where_they_are() {
        // Twenty-one bytes of garbage: enough for `try_raw_memory_parse` to
        // take the first sixteen as a UUID and the rest as content, and
        // fabricate every other field. It decodes. That is the defect.
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = MemoryStorage::new(dir.path(), None).expect("storage");

        let key = uuid::Uuid::parse_str("11112222-3333-4444-8555-666677778888").expect("uuid");
        let mut garbage = vec![0xAB_u8; 16];
        garbage.extend_from_slice(b"hello");
        assert_eq!(garbage.len(), 21);

        // The id the fabrication carries: derived from the VALUE, so it cannot
        // agree with the key the record was fetched under.
        let fabricated_id = MemoryId(uuid::Uuid::from_bytes([0xAB_u8; 16]));
        assert_ne!(fabricated_id.0, key);

        // Premise: unchecked, this decodes into a Memory with no error.
        assert!(
            deserialize_memory(&garbage).is_ok(),
            "premise: garbage pseudo-decodes through the fallback chain"
        );

        storage
            .db
            .put(key.as_bytes(), &garbage)
            .expect("seed the corrupted record");

        let err = storage
            .get_opt(&MemoryId(key))
            .expect_err("a decode that disagrees with its key must not be served");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("does not match its key"),
            "the error must name the id/key disagreement; got: {msg}"
        );

        // The evidence survives: the original bytes are still there, unchanged.
        let on_disk = storage
            .db
            .get(key.as_bytes())
            .expect("read back")
            .expect("the corrupted record must still exist");
        assert_eq!(
            on_disk, garbage,
            "the read path must not rewrite a record it could not trust"
        );

        // And no phantom was written at the fabricated id. `migrate_memory_format`
        // keys its rewrite off the DECODED id, so an unguarded lazy migration of
        // a pseudo-decode does not overwrite the original — it MANUFACTURES a
        // second record, at a key nothing ever wrote to, visible to every
        // full-CF scan in this file.
        assert!(
            storage
                .db
                .get(fabricated_id.0.as_bytes())
                .expect("read back")
                .is_none(),
            "the lazy migration must not have persisted the fabrication under \
             its invented id"
        );
    }

    #[test]
    fn a_scan_skips_the_record_whose_id_disagrees_with_its_key() {
        // The same check on the paths that SERVE results in bulk. A fabrication
        // that survives here is indistinguishable from a memory to every caller
        // downstream.
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = MemoryStorage::new(dir.path(), None).expect("storage");

        let good = sample_memory(
            MemoryId(uuid::Uuid::parse_str("aaaabbbb-cccc-4ddd-8eee-ffff00001111").unwrap()),
            "a real record",
        );
        storage.store(&good).expect("store");

        let key = uuid::Uuid::parse_str("99998888-7777-4666-8555-444433332222").expect("uuid");
        let mut garbage = vec![0xCD_u8; 16];
        garbage.extend_from_slice(b"fabricated content");
        storage.db.put(key.as_bytes(), &garbage).expect("seed");

        let all = storage.get_all().expect("get_all");
        assert_eq!(
            all.len(),
            1,
            "only the real record may be returned; got {:?}",
            all.iter().map(|m| m.id.0).collect::<Vec<_>>()
        );
        assert_eq!(all[0].id, good.id);

        // And the record is still on disk afterwards, untouched.
        assert_eq!(
            storage.db.get(key.as_bytes()).expect("read back"),
            Some(garbage)
        );
    }

    /// Build the SHO v2 bytes a pre-2026-07-12 build would have written for
    /// `memory`, whose experience must carry exactly one `NerEntityRecord` with
    /// `fine_label: None`.
    ///
    /// Today's encoder cannot produce that layout, and replicating `Experience`
    /// (50+ fields) to serialize it would be a second source of truth that rots.
    /// So the old bytes are derived by DIFFERENCE: encode the same memory twice,
    /// once with an empty `ner_entities` and once with the one record, and the
    /// encodings agree on every byte outside the span the record occupies. That
    /// span ends with the `0x00` that encodes `fine_label: None` — the byte the
    /// old format did not have — so removing it yields the pre-fine_label
    /// payload without duplicating a single field declaration.
    fn encode_pre_fine_label(memory: &Memory) -> Vec<u8> {
        assert_eq!(
            memory.experience.ner_entities.len(),
            1,
            "fixture expects exactly one NER record"
        );
        assert!(
            memory.experience.ner_entities[0].fine_label.is_none(),
            "the pre-fine_label format cannot carry a fine label"
        );

        let with_record = crate::serialization::encode_raw(memory).expect("encode with record");

        let mut without = memory.clone();
        without.experience.ner_entities.clear();
        let without_record = crate::serialization::encode_raw(&without).expect("encode without");

        // The encodings agree up to the `ner_entities` length varint, where one
        // says 1 and the other 0. Everything after that varint in the
        // record-less encoding is the shared tail (cooccurrence_pairs onward),
        // so its length pins where the record ends in the other encoding.
        // Deriving the tail by length rather than by scanning backwards for a
        // common suffix matters: the record's final byte IS 0x00, so a backward
        // scan runs straight past it into the record.
        let prefix = with_record
            .iter()
            .zip(without_record.iter())
            .take_while(|(a, b)| a == b)
            .count();
        assert!(
            prefix < without_record.len(),
            "the two encodings must differ at the ner_entities length varint"
        );
        assert_eq!(
            without_record[prefix], 0x00,
            "empty ner_entities is length 0"
        );
        assert_eq!(with_record[prefix], 0x01, "one NER record is length 1");

        let tail_len = without_record.len() - prefix - 1;
        let record_end = with_record.len() - tail_len;
        assert_eq!(
            with_record[record_end - 1],
            0x00,
            "the last byte of the NER record must be the fine_label None marker"
        );

        let mut legacy = Vec::with_capacity(with_record.len() - 1);
        legacy.extend_from_slice(&with_record[..record_end - 1]);
        legacy.extend_from_slice(&with_record[record_end..]);
        crate::serialization::wrap_sho_v2(&legacy)
    }

    #[test]
    fn memory_written_before_ner_fine_label_still_decodes() {
        // `NerEntityRecord::fine_label` was appended on 2026-07-12. The record
        // sits MID-PAYLOAD inside `Experience::ner_entities`, so a memory
        // written before that date desynchronises today's decoder by one byte
        // per NER record — everything after it (cooccurrence_pairs,
        // importance_override, and the whole of MemoryFlat past `experience`)
        // is read from the wrong offset. No tail-default suffix can repair a
        // hole in the middle, which is why this needs a generation retry.
        let id = MemoryId(
            uuid::Uuid::parse_str("7c1e5a6b-9d24-4f31-8a55-2b6c7d8e9f01").expect("static uuid"),
        );
        let mut memory = sample_memory(id.clone(), "the Dali struck the Key Bridge");
        memory.experience.ner_entities = vec![NerEntityRecord {
            text: "Key Bridge".to_string(),
            entity_type: "LOC".to_string(),
            confidence: 0.87,
            start_char: Some(19),
            end_char: Some(29),
            fine_label: None,
        }];
        memory.experience.cooccurrence_pairs = vec![("Dali".to_string(), "Key Bridge".to_string())];
        memory.experience.importance_override = Some(0.9);
        let created_at = memory.created_at;

        let legacy_bytes = encode_pre_fine_label(&memory);

        let (decoded, needs_migration) = deserialize_memory(&legacy_bytes)
            .expect("a memory written before fine_label existed must still be readable");

        assert_eq!(decoded.id, id);
        assert_eq!(decoded.experience.content, "the Dali struck the Key Bridge");
        assert_eq!(decoded.experience.ner_entities.len(), 1);
        assert_eq!(decoded.experience.ner_entities[0].text, "Key Bridge");
        assert_eq!(decoded.experience.ner_entities[0].end_char, Some(29));
        assert_eq!(
            decoded.experience.ner_entities[0].fine_label, None,
            "a record written before the field existed asserts no fine label"
        );
        // The fields AFTER the NER record are what a byte shift destroys, so
        // they are the real assertion: if the decoder is off by one, these come
        // back as garbage rather than these values.
        assert_eq!(
            decoded.experience.cooccurrence_pairs,
            vec![("Dali".to_string(), "Key Bridge".to_string())]
        );
        assert_eq!(decoded.experience.importance_override, Some(0.9));
        assert_eq!(decoded.created_at, created_at);
        assert!(
            needs_migration,
            "a record recovered through the legacy generation must be rewritten"
        );

        // The retry is scoped to the decode call and must not leak.
        assert_eq!(
            NerWireGeneration::current(),
            NerWireGeneration::Current,
            "the decode generation must be restored when the guard drops"
        );
    }

    #[test]
    fn current_memory_with_fine_label_is_unaffected_by_the_legacy_retry() {
        // A memory written today keeps its fine labels and must decode on the
        // first attempt, with no migration flag — the retry is a fallback, not
        // a second opinion.
        let id = MemoryId(
            uuid::Uuid::parse_str("2d4f6a8b-0c13-4e57-9a2b-3c5d7e9f0a1b").expect("static uuid"),
        );
        let mut memory = sample_memory(id.clone(), "GLiNER typed the span");
        memory.experience.ner_entities = vec![NerEntityRecord {
            text: "Key Bridge".to_string(),
            entity_type: "LOC".to_string(),
            confidence: 0.87,
            start_char: Some(0),
            end_char: Some(10),
            fine_label: Some("bridge".to_string()),
        }];

        let bytes = crate::serialization::encode_sho(&memory).unwrap();
        let (decoded, needs_migration) = deserialize_memory(&bytes).unwrap();

        assert_eq!(
            decoded.experience.ner_entities[0].fine_label.as_deref(),
            Some("bridge")
        );
        assert!(
            !needs_migration,
            "a current-format record must not be flagged for rewrite"
        );
    }

    #[test]
    fn memory_missing_its_tail_fields_is_defaulted_in_declaration_order() {
        // MEMORY_DEFAULT_SUFFIX repairs fields appended to the END of
        // MemoryFlat, one at a time, and the ORDER is a contract: `toponyms`,
        // then `declared_entities`, then `origin`. A record written before all
        // three, before the last two, before only the last, and a complete
        // record must all decode — and a record short by one field must not
        // have the WRONG field defaulted.
        //
        // The suffix bytes cannot catch a mis-ordering on their own: all three
        // defaults encode as `0x00` (two empty `Vec` lengths and
        // `MemoryOrigin::Unknown`'s variant index), so the suffix is
        // order-agnostic and reordering `MemoryFlat` still produces a decode
        // that succeeds. What catches it is truncating a REAL encoding one tail
        // field at a time and checking that the fields BEFORE the missing one
        // came back intact, which is what this test does. `origin` is therefore
        // set to a non-default variant: a defaulted `origin` is indistinguishable
        // from a correctly decoded `Unknown`.
        let id = MemoryId(
            uuid::Uuid::parse_str("8a7b6c5d-4e3f-4a2b-9c1d-0e1f2a3b4c5d").expect("static uuid"),
        );
        let mut memory = sample_memory(id.clone(), "a record with a populated tail");
        memory.experience.toponyms = vec![Toponym {
            mention: "Baltimore".to_string(),
            name: "Baltimore".to_string(),
            lat: 39.2904,
            lon: -76.6122,
            country: "US".to_string(),
            population: 585_708,
        }];
        memory.experience.declared_entities = vec!["Baltimore".to_string()];
        memory.experience.origin = MemoryOrigin::Api;

        // The property MEMORY_DEFAULT_SUFFIX depends on: `Unknown` must encode
        // as exactly the one `0x00` byte the decoder appends for a record
        // written before the field existed. Renumbering the enum so `Unknown`
        // is no longer variant 0 would break every such record silently.
        assert_eq!(
            crate::serialization::encode_raw(&MemoryOrigin::Unknown).expect("encode origin"),
            vec![0x00],
            "MemoryOrigin::Unknown must encode as the single byte the tail-default supplies"
        );

        let full = crate::serialization::encode_raw(&memory).expect("encode");

        // Complete record: no defaults applied.
        let (decoded, defaulted) =
            crate::serialization::decode_raw_compat::<Memory>(&full, MEMORY_DEFAULT_SUFFIX)
                .unwrap();
        assert!(!defaulted);
        assert_eq!(decoded.experience.declared_entities, vec!["Baltimore"]);
        assert_eq!(decoded.experience.toponyms.len(), 1);
        assert_eq!(decoded.experience.origin, MemoryOrigin::Api);

        // Written before `origin`: the tail ends after `declared_entities`. A
        // non-Unknown origin is one varint byte, so dropping the last byte
        // leaves exactly the shape the live `declared_entities`-era build wrote
        // — the records that already exist on disk.
        let without_origin = &full[..full.len() - 1];
        let (decoded, defaulted) =
            crate::serialization::decode_raw_compat::<Memory>(without_origin, MEMORY_DEFAULT_SUFFIX)
                .expect("a record written before origin must decode");
        assert!(
            defaulted,
            "the missing tail field must be reported as defaulted"
        );
        assert_eq!(
            decoded.experience.origin,
            MemoryOrigin::Unknown,
            "a record written before the field existed has no origin to recover"
        );
        assert_eq!(
            decoded.experience.declared_entities,
            vec!["Baltimore"],
            "the field before the missing one must survive intact — declaring \
             `origin` ahead of `declared_entities` would empty this instead"
        );
        assert_eq!(decoded.experience.toponyms.len(), 1);

        // Written before `declared_entities` too: the tail ends after
        // `toponyms`. A one-entry Vec<String> encodes as [len=1, strlen,
        // bytes...], so dropping "Baltimore" plus its two varints (on top of
        // the origin byte already dropped) leaves the pre-field shape.
        let without_declared = &full[..full.len() - 1 - ("Baltimore".len() + 2)];
        let (decoded, defaulted) = crate::serialization::decode_raw_compat::<Memory>(
            without_declared,
            MEMORY_DEFAULT_SUFFIX,
        )
        .expect("a record written before declared_entities must decode");
        assert!(
            defaulted,
            "the missing tail fields must be reported as defaulted"
        );
        assert!(decoded.experience.declared_entities.is_empty());
        assert_eq!(decoded.experience.origin, MemoryOrigin::Unknown);
        assert_eq!(
            decoded.experience.toponyms.len(),
            1,
            "the field before the missing ones must survive intact — a suffix \
             applied in the wrong order would empty this instead"
        );
    }

    #[test]
    fn test_deserialize_with_fallback_records_current_bincode2_branch() {
        // Deterministic id. The legacy fallback chain tries many decoders in
        // order, and the bincode1 attempts allow trailing bytes, so for certain
        // uuid byte patterns an earlier decoder spuriously matches the same
        // buffer before the intended branch — a random uuid made these fixture
        // tests flaky. Fixed, representative ids keep branch routing
        // reproducible without weakening any assertion below.
        let id = MemoryId(
            uuid::Uuid::parse_str("3f2a1b0c-1234-4abc-8def-0123456789ab").expect("static uuid"),
        );
        let memory = sample_memory(id.clone(), "current format memory");
        let bytes = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();

        let counter =
            crate::metrics::LEGACY_FALLBACK_BRANCH_TOTAL.with_label_values(&["bincode2_memory"]);
        let before = counter.get();

        let (decoded, is_legacy) = deserialize_with_fallback(&bytes).unwrap();
        let after = counter.get();

        assert_eq!(decoded.id, id);
        assert!(!is_legacy);
        // Current format is not a fallback — metric should NOT increment
        assert_eq!(after, before);
    }

    #[test]
    fn test_migrate_legacy_converts_raw_bincode_and_is_idempotent() {
        // Regression: migrate_legacy() decided "already current" via
        // `deserialize_memory(..).is_ok()`, but that returns Ok for legacy
        // records too (its whole purpose), so EVERY decodable record looked
        // current and `migrated` was always 0 — a silent no-op. Verify a
        // genuinely-legacy record (raw bincode 2.x, no SHO envelope) is migrated
        // to postcard and that a second pass is idempotent.
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = MemoryStorage::new(dir.path(), None).expect("storage");

        let id = MemoryId(
            uuid::Uuid::parse_str("5a6b7c8d-4567-4def-8234-56789abcdef0").expect("static uuid"),
        );
        let memory = sample_memory(id.clone(), "legacy raw bincode record");
        // Raw bincode 2.x with NO SHO envelope — the pre-migration on-disk
        // shape. (encode_sho would emit current postcard and defeat the test.)
        let legacy_bytes =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).expect("encode");
        assert_eq!(
            crate::serialization::read_sho_envelope(&legacy_bytes),
            crate::serialization::ShoEnvelope::Absent,
            "fixture must NOT carry an SHO header"
        );
        storage
            .db
            .put(id.0.as_bytes(), &legacy_bytes)
            .expect("seed legacy record");

        // First pass must actually migrate the legacy record (was 0 before fix).
        let (migrated, _already, failed) = storage.migrate_legacy().expect("migrate");
        assert_eq!(
            migrated, 1,
            "the legacy record must be migrated to postcard"
        );
        assert_eq!(failed, 0, "no decode/write failures");

        // The on-disk record is now SHO v2 postcard.
        let raw = storage
            .db
            .get(id.0.as_bytes())
            .expect("get")
            .expect("record present");
        assert!(
            matches!(
                crate::serialization::read_sho_envelope(&raw),
                crate::serialization::ShoEnvelope::Valid {
                    version: crate::serialization::SHO_VERSION_POSTCARD,
                    ..
                }
            ),
            "record must be rewritten as SHO v2 postcard"
        );

        // Second pass is a no-op: the record is already current.
        let (migrated2, already_current2, failed2) = storage.migrate_legacy().expect("migrate2");
        assert_eq!(migrated2, 0, "second pass migrates nothing");
        assert!(
            already_current2 >= 1,
            "the now-postcard record must count as already current"
        );
        assert_eq!(failed2, 0);
    }

    #[test]
    fn persist_access_updates_persists_record_and_moves_importance_bucket() {
        // Verifies the read-path batched access-update: the main record is
        // rewritten and the importance index is moved ONLY when the bucket
        // changed (date/type/etc. indices are untouched because they don't
        // change on access).
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = MemoryStorage::new(dir.path(), None).expect("storage");
        let id = MemoryId(
            uuid::Uuid::parse_str("a1a1a1a1-0000-4000-8000-000000000001").expect("static uuid"),
        );
        let memory = sample_memory(id.clone(), "access-update batching test");
        memory.set_importance(0.55); // bucket 5
        storage.store(&memory).expect("store");

        let idx = storage.index_cf();
        let key5 = format!("importance:5:{}", id.0);
        let key6 = format!("importance:6:{}", id.0);
        assert!(
            storage.db.get_cf(idx, key5.as_bytes()).unwrap().is_some(),
            "bucket 5 indexed after store"
        );

        // Access bumps importance across the bucket boundary.
        let before = memory.importance(); // 0.55
        memory.set_importance(0.65); // bucket 6
        storage
            .persist_access_updates(&[(&memory, before)])
            .expect("persist");

        // Main record carries the new importance.
        let reread = storage.get(&id).expect("get");
        assert!(
            (reread.importance() - 0.65).abs() < 1e-6,
            "new importance must be persisted, got {}",
            reread.importance()
        );
        // Importance index moved 5 -> 6.
        assert!(
            storage.db.get_cf(idx, key6.as_bytes()).unwrap().is_some(),
            "bucket 6 present after update"
        );
        assert!(
            storage.db.get_cf(idx, key5.as_bytes()).unwrap().is_none(),
            "stale bucket 5 removed"
        );

        // Empty input is a no-op.
        storage.persist_access_updates(&[]).expect("empty no-op");

        // Same-bucket bump persists the record but leaves the index untouched.
        let before2 = memory.importance(); // 0.65
        memory.set_importance(0.68); // still bucket 6
        storage
            .persist_access_updates(&[(&memory, before2)])
            .expect("persist same bucket");
        assert!(
            storage.db.get_cf(idx, key6.as_bytes()).unwrap().is_some(),
            "bucket 6 still present after same-bucket update"
        );
        assert!(
            (storage.get(&id).unwrap().importance() - 0.68).abs() < 1e-6,
            "importance still persisted on a same-bucket update"
        );
    }

    #[test]
    fn test_deserialize_with_fallback_bincode1_minimal_fixture() {
        let id = MemoryId(
            uuid::Uuid::parse_str("7a8b9c0d-2345-4bcd-9012-3456789abcde").expect("static uuid"),
        );
        let fixture = LegacyMinimalFixture {
            id: id.clone(),
            content: "legacy bincode1 minimal".to_string(),
        };
        let bytes = bincode1::serialize(&fixture).unwrap();

        let counter =
            crate::metrics::LEGACY_FALLBACK_BRANCH_TOTAL.with_label_values(&["bincode1_minimal"]);
        let before = counter.get();

        let (decoded, is_legacy) = deserialize_with_fallback(&bytes).unwrap();
        let after = counter.get();

        assert_eq!(decoded.id, id);
        assert!(is_legacy);
        assert_eq!(after, before + 1);
    }

    #[test]
    fn test_deserialize_with_fallback_msgpack_minimal_fixture() {
        let id = MemoryId(
            uuid::Uuid::parse_str("1e2d3c4b-3456-4cde-8123-456789abcdef").expect("static uuid"),
        );
        let fixture = LegacyMinimalFixture {
            id: id.clone(),
            content: "legacy msgpack minimal".to_string(),
        };
        let bytes = rmp_serde::to_vec(&fixture).unwrap();

        let counter =
            crate::metrics::LEGACY_FALLBACK_BRANCH_TOTAL.with_label_values(&["msgpack_minimal"]);
        let before = counter.get();

        let (decoded, is_legacy) = deserialize_with_fallback(&bytes).unwrap();
        let after = counter.get();

        assert_eq!(decoded.id, id);
        assert!(is_legacy);
        assert_eq!(after, before + 1);
    }

    #[test]
    fn test_write_mode_default_async() {
        std::env::remove_var("SHODH_WRITE_MODE");
        let mode = WriteMode::default();
        assert_eq!(mode, WriteMode::Async);
    }

    #[test]
    fn test_crc32_simple() {
        let data = b"test data for CRC32";
        let crc1 = crc32_simple(data);
        let crc2 = crc32_simple(data);

        assert_eq!(crc1, crc2);
        assert_ne!(crc1, 0);

        let crc3 = crc32_simple(b"different data");
        assert_ne!(crc1, crc3);
    }

    #[test]
    fn test_crc32_empty() {
        let crc = crc32_simple(b"");
        assert_eq!(
            crc, 0,
            "IEEE CRC32 of empty input is 0 (init 0xFFFFFFFF XOR final 0xFFFFFFFF)"
        );
    }

    #[test]
    fn test_modality_dimension() {
        assert_eq!(Modality::Text.dimension(), 384);
        // ImageBind projects all non-text modalities to 1024-dim shared space
        assert_eq!(Modality::Image.dimension(), 1024);
        assert_eq!(Modality::Audio.dimension(), 1024);
        assert_eq!(Modality::Video.dimension(), 1024);
        assert_eq!(Modality::Unified.dimension(), 1024);
    }

    #[test]
    fn test_modality_as_str() {
        assert_eq!(Modality::Text.as_str(), "text");
        assert_eq!(Modality::Image.as_str(), "image");
        assert_eq!(Modality::Audio.as_str(), "audio");
        assert_eq!(Modality::Video.as_str(), "video");
    }

    #[test]
    fn test_vector_mapping_entry_with_text() {
        let entry = VectorMappingEntry::with_text(vec![1, 2, 3]);

        assert_eq!(entry.text_vectors(), Some(&vec![1, 2, 3]));
        assert!(!entry.is_empty());
    }

    #[test]
    fn test_vector_mapping_entry_multimodal() {
        let entry = VectorMappingEntry::with_text(vec![1])
            .with_image(vec![2])
            .with_audio(vec![3])
            .with_video(vec![4]);

        let all = entry.all_vector_ids();
        assert_eq!(all.len(), 4);

        assert!(all.contains(&(Modality::Text, 1)));
        assert!(all.contains(&(Modality::Image, 2)));
        assert!(all.contains(&(Modality::Audio, 3)));
        assert!(all.contains(&(Modality::Video, 4)));
    }

    #[test]
    fn test_vector_mapping_entry_empty() {
        let entry = VectorMappingEntry::default();

        assert!(entry.is_empty());
        assert!(entry.text_vectors().is_none());
        assert!(entry.all_vector_ids().is_empty());
    }

    #[test]
    fn test_vector_mapping_entry_add_modality() {
        let mut entry = VectorMappingEntry::default();
        entry.add_modality(Modality::Text, vec![1, 2]);

        assert_eq!(entry.text_vectors(), Some(&vec![1, 2]));
    }

    #[test]
    fn test_storage_stats_default() {
        let stats = StorageStats::default();

        assert_eq!(stats.total_count, 0);
        assert_eq!(stats.compressed_count, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert_eq!(stats.total_retrievals, 0);
    }

    #[test]
    fn test_search_criteria_variants() {
        let criteria1 = SearchCriteria::ByEntity("test".to_string());
        let criteria2 = SearchCriteria::ByImportance { min: 0.5, max: 1.0 };
        let criteria3 = SearchCriteria::ByType(ExperienceType::Observation);

        assert!(matches!(criteria1, SearchCriteria::ByEntity(_)));
        assert!(matches!(criteria2, SearchCriteria::ByImportance { .. }));
        assert!(matches!(criteria3, SearchCriteria::ByType(_)));
    }

    #[test]
    fn test_search_criteria_by_date() {
        let now = Utc::now();
        let start = now - chrono::Duration::days(7);
        let criteria = SearchCriteria::ByDate { start, end: now };

        if let SearchCriteria::ByDate { start: s, end: e } = criteria {
            assert!(s < e);
        } else {
            panic!("Expected ByDate");
        }
    }

    #[test]
    fn test_search_criteria_combined() {
        let criteria = SearchCriteria::Combined(vec![
            SearchCriteria::ByEntity("test".to_string()),
            SearchCriteria::ByImportance { min: 0.5, max: 1.0 },
        ]);

        if let SearchCriteria::Combined(inner) = criteria {
            assert_eq!(inner.len(), 2);
        } else {
            panic!("Expected Combined");
        }
    }

    #[test]
    fn test_modality_vectors_struct() {
        let mv = ModalityVectors {
            vector_ids: vec![1, 2, 3],
            dimension: 384,
            chunk_ranges: None,
        };

        assert_eq!(mv.vector_ids.len(), 3);
        assert_eq!(mv.dimension, 384);
        assert!(mv.chunk_ranges.is_none());
    }

    /// Two concurrent read-modify-write updates of the SAME memory must both
    /// survive. Before `modify()` existed, callers did `get` -> mutate ->
    /// `update` inline while holding only a `.read()` guard on `MemorySystem`,
    /// so both readers saw the pre-update record and the second write silently
    /// erased the first.
    ///
    /// The sleep inside each closure widens the window deterministically: with
    /// the lock it only serializes the two calls, without it the loss is
    /// reliable rather than occasional.
    #[test]
    fn concurrent_modify_does_not_lose_an_update() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = Arc::new(MemoryStorage::new(dir.path(), None).expect("storage"));

        let id = MemoryId(uuid::Uuid::new_v4());
        storage
            .store(&sample_memory(id.clone(), "a memory two writers will touch"))
            .expect("store");

        let parent = MemoryId(uuid::Uuid::new_v4());

        let a = {
            let storage = Arc::clone(&storage);
            let id = id.clone();
            let parent = parent.clone();
            std::thread::spawn(move || {
                storage
                    .modify(&id, |m| {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        m.set_parent(Some(parent));
                    })
                    .expect("modify parent")
            })
        };
        let b = {
            let storage = Arc::clone(&storage);
            let id = id.clone();
            std::thread::spawn(move || {
                storage
                    .modify(&id, |m| {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        m.experience.tags.push("tagged-by-the-other-writer".to_string());
                    })
                    .expect("modify tags")
            })
        };
        a.join().expect("thread a");
        b.join().expect("thread b");

        let final_memory = storage.get(&id).expect("read back");
        assert_eq!(
            final_memory.get_parent(),
            Some(&parent),
            "the parent write was lost by the concurrent tag write"
        );
        assert!(
            final_memory
                .experience
                .tags
                .iter()
                .any(|t| t == "tagged-by-the-other-writer"),
            "the tag write was lost by the concurrent parent write"
        );
    }

    /// `modify` must not invent a memory, and must not report absence for a
    /// record it simply failed to read.
    #[test]
    fn modify_reports_a_missing_memory_without_writing_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = MemoryStorage::new(dir.path(), None).expect("storage");

        let missing = MemoryId(uuid::Uuid::new_v4());
        let mut ran = false;
        let result = storage
            .modify(&missing, |_| ran = true)
            .expect("modify a missing memory is not an error");
        assert!(result.is_none(), "modify reported a memory that never existed");
        assert!(!ran, "the mutation ran against a memory that does not exist");
        assert!(storage.get_opt(&missing).expect("get_opt").is_none());
    }

    /// `modify()` is a decode → mutate → re-encode round trip, so it can drop
    /// any field the round trip does not carry. The three `MemoryFlat` tail
    /// fields are exactly that shape: they are `#[serde(skip)]` on `Experience`
    /// and ride at the tail of `MemoryFlat` instead, which is why the LZ4 path
    /// has to copy them back by hand.
    ///
    /// `modify()` was written on a branch that had no `origin` and no
    /// `declared_entities` at all, so it could not have been checked against
    /// them, and its live caller is `set_memory_parent` — a path that runs on
    /// memories carrying exactly this authority. If the round trip dropped
    /// them, reparenting a memory would silently retract every caller entity
    /// assertion the graph admits on, and reset its provenance to `Unknown`,
    /// while changing nothing a caller of `set_memory_parent` would think to
    /// look at.
    ///
    /// `origin` is set to a non-default variant deliberately: a dropped
    /// `origin` decodes as `Unknown`, which is indistinguishable from a record
    /// that never had one.
    #[test]
    fn modify_preserves_the_memory_flat_tail_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = MemoryStorage::new(dir.path(), None).expect("storage");

        let id = MemoryId(uuid::Uuid::new_v4());
        let mut memory = sample_memory(id.clone(), "a memory whose tail must survive a modify");
        memory.experience.toponyms = vec![Toponym {
            mention: "Baltimore".to_string(),
            name: "Baltimore".to_string(),
            lat: 39.2904,
            lon: -76.6122,
            country: "US".to_string(),
            population: 585_708,
        }];
        memory.experience.declared_entities = vec!["Baltimore".to_string()];
        memory.experience.origin = MemoryOrigin::Api;
        storage.store(&memory).expect("store");

        // The live mutation: reparenting touches none of the three.
        let parent = MemoryId(uuid::Uuid::new_v4());
        storage
            .modify(&id, |m| m.set_parent(Some(parent.clone())))
            .expect("modify")
            .expect("the memory exists");

        let after = storage.get(&id).expect("read back");
        assert_eq!(after.get_parent(), Some(&parent), "the mutation itself");
        assert_eq!(
            after.experience.declared_entities,
            vec!["Baltimore".to_string()],
            "modify() retracted the caller's declared entities — graph \
             admission runs on exactly this authority"
        );
        assert_eq!(
            after.experience.origin,
            MemoryOrigin::Api,
            "modify() reset the memory's origin to Unknown, which reads as a \
             record that never had one"
        );
        assert_eq!(
            after.experience.toponyms.len(),
            1,
            "modify() dropped the resolved toponyms"
        );
        assert_eq!(after.experience.toponyms[0].name, "Baltimore");
    }
}
