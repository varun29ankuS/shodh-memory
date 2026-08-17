//! Per-memory origin: which write path put a memory in the store.
//!
//! Four things are pinned here:
//!
//! 1. **Old records read as `Unknown`, not as something wrong.** Storage is
//!    postcard — positional, no per-field presence marker — so adding a field
//!    is a wire-format change and has to be proven against bytes written
//!    before the field existed. A mis-placed field does not fail to decode; it
//!    reads the next field's first byte as its own and shifts everything after
//!    it, which is why this is the test that matters most.
//! 2. **Nothing backfills.** `Unknown` survives a decode, a re-encode, a
//!    compression round trip and a dedup merge. Inferring an origin for a
//!    historical memory would be manufacturing provenance.
//! 3. **First write wins.** A duplicate arriving down a different path does
//!    not get to relabel where a memory came from.
//! 4. **The value is reachable on the read path**, at a pinned location.
//!
//! Storage lives under `std::env::temp_dir()` via `tempfile::TempDir`, NOT
//! under any watched path — the tantivy file-watcher corrupts commits under
//! watched directories (see the BM25 file-watcher finding).

use chrono::{DateTime, Utc};
use serde::Serialize;
use tempfile::TempDir;
use uuid::Uuid;

use shodh_memory::memory::compression::CompressionPipeline;
use shodh_memory::memory::storage::deserialize_memory_for_migration;
use shodh_memory::memory::types::{
    ChangeType, EntityRef, Experience, ExperienceType, MemoryOrigin, MemoryRevision, MemoryTier,
    Toponym,
};
use shodh_memory::memory::{Memory, MemoryConfig, MemoryId, MemorySystem, TodoId};
use shodh_memory::serialization::{encode_raw, wrap_sho_v2};

fn setup_memory_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 200,
        ..Default::default()
    };
    let system = MemorySystem::new(config, None).expect("Failed to create memory system");
    (system, temp_dir)
}

// ============================================================================
// THE WIRE NAMES
// ============================================================================

/// Every variant must have a distinct wire name that survives a parse.
///
/// `as_str` is written by hand rather than derived from `Debug`, precisely so
/// that renaming a Rust variant cannot silently change the API contract. That
/// makes it a table someone can get wrong, so it is checked exhaustively
/// against `MemoryOrigin::ALL` rather than on a couple of samples.
#[test]
fn every_origin_has_a_unique_wire_name_that_round_trips() {
    let mut seen: Vec<&'static str> = Vec::new();
    for origin in MemoryOrigin::ALL {
        let name = origin.as_str();
        assert!(
            !name.is_empty(),
            "{origin:?} has an empty wire name — it would be unfilterable"
        );
        assert!(
            !seen.contains(&name),
            "wire name {name:?} is used by two variants; a filter could not tell them apart"
        );
        seen.push(name);

        assert_eq!(
            MemoryOrigin::parse(name),
            Some(*origin),
            "{origin:?} does not survive as_str -> parse"
        );
    }

    // `ALL` is what `parse` searches, so a variant missing from it is
    // unparseable. Pinning the count makes adding a variant without updating
    // the list a failing test rather than a silently unfilterable origin.
    assert_eq!(
        MemoryOrigin::ALL.len(),
        15,
        "a variant was added or removed; update this count deliberately, and \
         remember that reordering variants re-points every stored record"
    );
    assert_eq!(
        MemoryOrigin::ALL[0],
        MemoryOrigin::Unknown,
        "Unknown must stay at index 0: its postcard encoding is the 0x00 byte \
         MEMORY_DEFAULT_SUFFIX supplies for records written before the field existed"
    );
}

/// Query strings are typed by humans, so parsing is deliberately tolerant of
/// case and of `-` for `_` — and deliberately intolerant of anything else, so
/// a typo is rejected instead of silently matching nothing.
#[test]
fn origin_parse_is_tolerant_of_case_and_dashes_but_not_of_typos() {
    assert_eq!(
        MemoryOrigin::parse("TODO_LIFECYCLE"),
        Some(MemoryOrigin::TodoLifecycle)
    );
    assert_eq!(
        MemoryOrigin::parse("todo-lifecycle"),
        Some(MemoryOrigin::TodoLifecycle)
    );
    assert_eq!(
        MemoryOrigin::parse("  todo_lifecycle  "),
        Some(MemoryOrigin::TodoLifecycle)
    );

    assert_eq!(MemoryOrigin::parse("todo"), None);
    assert_eq!(MemoryOrigin::parse("TodoLifecycle"), None);
    assert_eq!(MemoryOrigin::parse(""), None);
    assert_eq!(MemoryOrigin::parse("hook"), None);
}

/// The default is the honest one. A `Default::default()` construction site is
/// one that does NOT know its write path (a legacy decoder, a test fixture),
/// and it must not pick a plausible-looking origin on its behalf.
#[test]
fn default_origin_is_unknown() {
    assert_eq!(MemoryOrigin::default(), MemoryOrigin::Unknown);
    assert_eq!(Experience::default().origin, MemoryOrigin::Unknown);
}

// ============================================================================
// OLD-FORMAT COMPATIBILITY — the load-bearing test
// ============================================================================

/// `MemoryFlat` exactly as it was encoded BEFORE the trailing `origin` field
/// was added — same fields, same order, same types, ending at `toponyms`.
///
/// `Experience` is embedded here as the real type on purpose: its `origin`
/// field is `#[serde(skip)]`, so encoding it reproduces the historical bytes
/// byte-for-byte. If someone ever removes that `skip`, this struct starts
/// emitting the field mid-payload and the tests below fail — which is exactly
/// the regression we want to catch, because that is the shape that decodes to
/// silently wrong values in production rather than erroring.
#[derive(Serialize)]
struct MemoryFlatPreOrigin {
    id: MemoryId,
    experience: Experience,
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    tier: MemoryTier,
    entity_refs: Vec<EntityRef>,
    activation: f32,
    last_retrieval_id: Option<Uuid>,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
    external_id: Option<String>,
    version: u32,
    history: Vec<MemoryRevision>,
    related_todo_ids: Vec<TodoId>,
    parent_id: Option<MemoryId>,
    toponyms: Vec<Toponym>,
}

/// The same, one schema revision older still: written before `toponyms` too,
/// so it is short by BOTH tail fields. `decode_raw_compat` appends the default
/// suffix one byte at a time, and this is the case that proves it.
#[derive(Serialize)]
struct MemoryFlatPreToponyms {
    id: MemoryId,
    experience: Experience,
    importance: f32,
    access_count: u32,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    compressed: bool,
    tier: MemoryTier,
    entity_refs: Vec<EntityRef>,
    activation: f32,
    last_retrieval_id: Option<Uuid>,
    agent_id: Option<String>,
    run_id: Option<String>,
    actor_id: Option<String>,
    temporal_relevance: f32,
    score: Option<f32>,
    external_id: Option<String>,
    version: u32,
    history: Vec<MemoryRevision>,
    related_todo_ids: Vec<TodoId>,
    parent_id: Option<MemoryId>,
}

/// Distinctive scalar values, chosen because these are precisely the fields a
/// mis-decode corrupts: a field read from the MIDDLE of the payload consumes
/// the first byte of `importance` as its own and shifts everything after it.
/// Asserting only `origin == Unknown` would pass under that corruption, since
/// a shifted read also tends to yield the zero variant.
const PROBE_IMPORTANCE: f32 = 0.625;
const PROBE_ACCESS_COUNT: u32 = 7;
const PROBE_ACTIVATION: f32 = 0.375;
const PROBE_VERSION: u32 = 3;

fn probe_experience(content: &str) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Learning,
        entities: vec!["gripper".to_string()],
        ..Default::default()
    }
}

fn assert_nothing_shifted(decoded: &Memory, id: &MemoryId, content: &str) {
    assert_eq!(decoded.id, *id);
    assert_eq!(decoded.tier, MemoryTier::Working);
    assert_eq!(decoded.experience.content, content);
    assert_eq!(decoded.experience.entities, vec!["gripper".to_string()]);
    assert_eq!(decoded.experience.experience_type, ExperienceType::Learning);
    assert!(
        (decoded.importance() - PROBE_IMPORTANCE).abs() < 1e-6,
        "importance corrupted: got {}",
        decoded.importance()
    );
    assert_eq!(
        decoded.access_count(),
        PROBE_ACCESS_COUNT,
        "access_count corrupted"
    );
    assert_eq!(decoded.agent_id.as_deref(), Some("agent-7"));
    assert_eq!(decoded.external_id.as_deref(), Some("ext-42"));
    assert_eq!(decoded.version, PROBE_VERSION);
}

#[test]
fn records_written_before_origin_existed_decode_as_unknown() {
    let id = MemoryId(Uuid::new_v4());
    let now = Utc::now();
    let content = "A memory written before origins were recorded";

    let legacy = MemoryFlatPreOrigin {
        id: id.clone(),
        experience: probe_experience(content),
        importance: PROBE_IMPORTANCE,
        access_count: PROBE_ACCESS_COUNT,
        created_at: now,
        last_accessed: now,
        compressed: false,
        tier: MemoryTier::Working,
        entity_refs: Vec::new(),
        activation: PROBE_ACTIVATION,
        last_retrieval_id: None,
        agent_id: Some("agent-7".to_string()),
        run_id: None,
        actor_id: None,
        temporal_relevance: 0.5,
        score: None,
        external_id: Some("ext-42".to_string()),
        version: PROBE_VERSION,
        history: Vec::new(),
        related_todo_ids: Vec::new(),
        parent_id: None,
        // A non-empty toponym list on purpose: if `origin` were ever declared
        // BEFORE this field, the decoder would read this Vec's length varint as
        // the origin discriminant. That is the exact merge hazard documented on
        // `MemoryFlat::origin`, and an empty Vec here would hide it.
        toponyms: vec![Toponym {
            mention: "Baltimore".to_string(),
            name: "Baltimore".to_string(),
            lat: 39.29038,
            lon: -76.61219,
            country: "US".to_string(),
            population: 585_708,
        }],
    };

    // Encode exactly as the storage layer did before the field existed: raw
    // postcard payload inside a SHO v2 envelope.
    let payload = encode_raw(&legacy).expect("encode legacy record");
    let record = wrap_sho_v2(&payload);

    let decoded = deserialize_memory_for_migration(&record).expect(
        "a record written before `origin` existed must still decode — postcard \
         has no #[serde(default)] EOF tolerance, so this only works if the \
         field is appended at the tail and defaulted by the decoder",
    );

    assert_eq!(
        decoded.experience.origin,
        MemoryOrigin::Unknown,
        "an old record has no recorded origin, and nothing may infer one"
    );

    // The preceding tail field must still be intact — this is what proves the
    // new field was appended AFTER it rather than in front of it.
    assert_eq!(
        decoded.experience.toponyms.len(),
        1,
        "toponyms was consumed by the new field: origin is declared in the wrong position"
    );
    assert_eq!(decoded.experience.toponyms[0].name, "Baltimore");

    assert_nothing_shifted(&decoded, &id, content);
}

#[test]
fn records_missing_both_tail_fields_decode_as_unknown() {
    let id = MemoryId(Uuid::new_v4());
    let now = Utc::now();
    let content = "A memory written before toponyms or origins existed";

    let legacy = MemoryFlatPreToponyms {
        id: id.clone(),
        experience: probe_experience(content),
        importance: PROBE_IMPORTANCE,
        access_count: PROBE_ACCESS_COUNT,
        created_at: now,
        last_accessed: now,
        compressed: false,
        tier: MemoryTier::Working,
        entity_refs: Vec::new(),
        activation: PROBE_ACTIVATION,
        last_retrieval_id: None,
        agent_id: Some("agent-7".to_string()),
        run_id: None,
        actor_id: None,
        temporal_relevance: 0.5,
        score: None,
        external_id: Some("ext-42".to_string()),
        version: PROBE_VERSION,
        history: Vec::new(),
        related_todo_ids: Vec::new(),
        parent_id: None,
    };

    let payload = encode_raw(&legacy).expect("encode legacy record");
    let record = wrap_sho_v2(&payload);

    let decoded = deserialize_memory_for_migration(&record).expect(
        "a record short by TWO trailing fields must still decode — \
         MEMORY_DEFAULT_SUFFIX is applied one byte at a time for exactly this case",
    );

    assert_eq!(decoded.experience.origin, MemoryOrigin::Unknown);
    assert!(decoded.experience.toponyms.is_empty());
    assert_nothing_shifted(&decoded, &id, content);
}

#[test]
fn current_records_round_trip_with_origin_intact() {
    // The other half of the compat story: a record written NOW must decode
    // with its origin, through the same entry point. A non-zero discriminant
    // is used deliberately — `Unknown` would round-trip even if the field were
    // dropped entirely.
    let id = MemoryId(Uuid::new_v4());
    let memory = Memory::new(
        id.clone(),
        Experience {
            content: "Todo completed echo".to_string(),
            experience_type: ExperienceType::Task,
            origin: MemoryOrigin::TodoLifecycle,
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );

    let payload = encode_raw(&memory).expect("encode current record");
    let record = wrap_sho_v2(&payload);
    let decoded = deserialize_memory_for_migration(&record).expect("decode current record");

    assert_eq!(decoded.experience.origin, MemoryOrigin::TodoLifecycle);
    assert_eq!(decoded.id, id);
}

/// A record decoded from the old format and re-encoded in the current one must
/// still say `Unknown`. This is the no-backfill guarantee at the point where it
/// would be easiest to violate: `deserialize_memory` returns
/// `needs_migration = true` for these records, and the migration rewrite is
/// exactly the place someone might be tempted to "helpfully" stamp an origin.
#[test]
fn the_migration_rewrite_does_not_invent_an_origin() {
    let id = MemoryId(Uuid::new_v4());
    let now = Utc::now();
    let content = "A memory that will be rewritten in the current schema";

    let legacy = MemoryFlatPreOrigin {
        id: id.clone(),
        experience: probe_experience(content),
        importance: PROBE_IMPORTANCE,
        access_count: PROBE_ACCESS_COUNT,
        created_at: now,
        last_accessed: now,
        compressed: false,
        tier: MemoryTier::Working,
        entity_refs: Vec::new(),
        activation: PROBE_ACTIVATION,
        last_retrieval_id: None,
        agent_id: Some("agent-7".to_string()),
        run_id: None,
        actor_id: None,
        temporal_relevance: 0.5,
        score: None,
        external_id: Some("ext-42".to_string()),
        version: PROBE_VERSION,
        history: Vec::new(),
        related_todo_ids: Vec::new(),
        parent_id: None,
        toponyms: Vec::new(),
    };

    let old_record = wrap_sho_v2(&encode_raw(&legacy).expect("encode legacy record"));
    let decoded = deserialize_memory_for_migration(&old_record).expect("decode legacy record");

    // Re-encode in the CURRENT schema, as the migration rewrite does...
    let rewritten = wrap_sho_v2(&encode_raw(&decoded).expect("re-encode in current schema"));
    let reread = deserialize_memory_for_migration(&rewritten).expect("decode rewritten record");

    assert_eq!(
        reread.experience.origin,
        MemoryOrigin::Unknown,
        "the rewrite manufactured provenance for a record that never had any"
    );
    assert_nothing_shifted(&reread, &id, content);
}

// ============================================================================
// SURVIVAL THROUGH THE REST OF THE PIPELINE
// ============================================================================

/// `Experience::origin` is `#[serde(skip)]`, and the LZ4 compressor encodes
/// `experience` STANDALONE — so the compressed blob does not contain the origin
/// at all. Decompression must carry it across from the memory's outer
/// experience, or every compressed memory silently reads as `Unknown`, which is
/// indistinguishable from a pre-origin record.
#[test]
fn origin_survives_lz4_compression_round_trip() {
    let compressor = CompressionPipeline::new();

    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: "Coordination between the Baltimore and London teams".repeat(20),
            experience_type: ExperienceType::Observation,
            origin: MemoryOrigin::GithubConnector,
            ..Default::default()
        },
        // High importance selects the lossless LZ4 strategy — the one that
        // encodes `experience` standalone and so loses skipped fields.
        0.95,
        None,
        None,
        None,
        None,
    );

    let compressed = compressor.compress(&memory).expect("compress");
    assert_eq!(
        compressed
            .experience
            .metadata
            .get("compression_strategy")
            .map(String::as_str),
        Some("lz4"),
        "precondition: this memory must take the LZ4 path"
    );

    let restored = compressor.decompress(&compressed).expect("decompress");
    assert_eq!(
        restored.experience.origin,
        MemoryOrigin::GithubConnector,
        "decompression wiped the origin"
    );
    assert_eq!(restored.experience.content, memory.experience.content);
}

/// First write wins. `remember()` dedups on a content hash and merges the
/// duplicate's enrichment into the stored memory; a duplicate arriving down a
/// different write path did not create the memory and must not relabel it.
#[test]
fn a_duplicate_from_another_path_does_not_relabel_the_origin() {
    let (system, _dir) = setup_memory_system();
    let content = "The gripper stalled at 40% travel on the third attempt";

    let first = system
        .remember(
            Experience {
                content: content.to_string(),
                origin: MemoryOrigin::Api,
                ..Default::default()
            },
            None,
        )
        .expect("first write");

    // Byte-identical content arriving from a different path, carrying real
    // enrichment so the merge branch actually runs rather than short-circuiting.
    let second = system
        .remember(
            Experience {
                content: content.to_string(),
                entities: vec!["gripper".to_string()],
                origin: MemoryOrigin::MifImport,
                ..Default::default()
            },
            None,
        )
        .expect("duplicate write");

    assert_eq!(first, second, "precondition: the dedup path must have run");

    let stored = system.get_memory(&first).expect("stored memory readable");
    assert_eq!(
        stored.experience.origin,
        MemoryOrigin::Api,
        "the duplicate relabelled where the memory came from"
    );
    assert!(
        stored.experience.entities.contains(&"gripper".to_string()),
        "precondition: the merge must have folded in the duplicate's enrichment, \
         otherwise this test proves nothing about the merge"
    );
}

/// The update half of an upsert mutates the memory that is already stored. It
/// must leave that memory's origin alone: the record was created by whatever
/// path first saw the `external_id`, and a later update did not create it.
#[test]
fn upsert_update_preserves_the_stored_origin() {
    let (system, _dir) = setup_memory_system();

    let (created_id, was_update) = system
        .upsert(
            "linear:SHO-1".to_string(),
            Experience {
                content: "SHO-1: gripper calibration".to_string(),
                origin: MemoryOrigin::LinearConnector,
                ..Default::default()
            },
            ChangeType::Created,
            Some("linear-webhook".to_string()),
            None,
        )
        .expect("create via upsert");
    assert!(!was_update, "precondition: first upsert must create");

    let (updated_id, was_update) = system
        .upsert(
            "linear:SHO-1".to_string(),
            Experience {
                content: "SHO-1: gripper calibration (done)".to_string(),
                // A different origin on the incoming copy: if the update path
                // ever copied `experience.origin` across, this is what would
                // land.
                origin: MemoryOrigin::Api,
                ..Default::default()
            },
            ChangeType::ContentUpdated,
            Some("someone-else".to_string()),
            None,
        )
        .expect("update via upsert");
    assert!(was_update, "precondition: second upsert must update");
    assert_eq!(created_id, updated_id);

    let stored = system
        .get_memory(&updated_id)
        .expect("stored memory readable");
    assert_eq!(
        stored.experience.content, "SHO-1: gripper calibration (done)",
        "precondition: the update must actually have applied"
    );
    assert_eq!(
        stored.experience.origin,
        MemoryOrigin::LinearConnector,
        "the update relabelled where the memory came from"
    );
}

/// A stamped origin must survive the full store-and-read-back path, not just an
/// in-memory encode. This is the round trip a caller actually observes.
#[test]
fn a_stamped_origin_survives_storage_and_readback() {
    let (system, _dir) = setup_memory_system();

    let id = system
        .remember(
            Experience {
                content: "Session digest for 2026-08-16".to_string(),
                experience_type: ExperienceType::Context,
                origin: MemoryOrigin::SessionSummary,
                ..Default::default()
            },
            None,
        )
        .expect("write");

    let stored = system.get_memory(&id).expect("readable");
    assert_eq!(stored.experience.origin, MemoryOrigin::SessionSummary);
}

// ============================================================================
// READ PATH
// ============================================================================

/// `GET /api/memory/{id}` returns the whole `Memory`, serialized through
/// `MemoryFlat`. Because `origin` is carried at the flat struct's tail, it
/// surfaces as a TOP-LEVEL key rather than nested under `experience` — the same
/// placement as `toponyms`. This pins where, so anyone moving it does so
/// deliberately.
#[test]
fn get_memory_json_exposes_origin_at_the_top_level() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience {
            content: "Auto-captured assistant turn".to_string(),
            experience_type: ExperienceType::Observation,
            origin: MemoryOrigin::AutoIngest,
            ..Default::default()
        },
        0.5,
        None,
        None,
        None,
        None,
    );

    let json: serde_json::Value =
        serde_json::to_value(&memory).expect("Memory must serialize to JSON");

    assert_eq!(
        json.get("origin").and_then(|o| o.as_str()),
        Some("auto_ingest"),
        "origin must be present at the top level of the memory object"
    );
    assert!(
        json["experience"].get("origin").is_none(),
        "origin is carried on the flat struct, not inside `experience`"
    );
}

/// The JSON name is the same string `as_str` produces, so a client can send
/// back what it received as an `origin=` filter value without translation.
#[test]
fn json_names_match_the_filter_names() {
    for origin in MemoryOrigin::ALL {
        let json = serde_json::to_value(origin).expect("origin serializes");
        assert_eq!(
            json.as_str(),
            Some(origin.as_str()),
            "{origin:?} serializes to a different string than it filters by"
        );
    }
}
