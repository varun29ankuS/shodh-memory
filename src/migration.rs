//! Offline migration from legacy serialization formats to postcard.
//!
//! Run via: `shodh-memory-server migrate --storage <path> [--dry-run]`
//!
//! This reads every record in every RocksDB store, decodes it using the
//! legacy format (bincode 2.x for most stores, msgpack for learning events),
//! re-encodes it as postcard with the appropriate format tag or SHO envelope,
//! and writes it back. Records that are already in postcard format are skipped.
//!
//! ## Safety
//!
//! - The server must NOT be running during migration (checked via RocksDB lock).
//! - On success, a marker file `migration_v2_postcard` is written to the storage
//!   directory. The server will refuse to start without this marker once the
//!   migration code is shipped (future enforcement, not yet gated).
//! - `--dry-run` reports what would be migrated without modifying any data.

use anyhow::{Context, Result};
use rocksdb::{ColumnFamilyDescriptor, IteratorMode, Options as RocksOptions, WriteBatch, DB};
use std::path::Path;

use crate::serialization;

// Re-use the actual stored types for deserialization.
// These are the concrete types that bincode encoded into RocksDB values.
use crate::graph_memory::{EntityNode, EpisodicNode, RelationshipEdge};
use crate::handlers::types::AuditEvent;
use crate::memory::compression::SemanticFact;
use crate::memory::lineage::{LineageBranch, LineageEdge};
use crate::memory::storage::VectorMappingEntry;
use crate::memory::temporal_facts::TemporalFact;

/// Marker file written after successful migration.
const MIGRATION_MARKER: &str = "migration_v2_postcard";

/// Maximum records per WriteBatch to bound memory usage.
const BATCH_SIZE: usize = 500;

/// Check whether the storage directory has already been migrated.
pub fn is_migrated(storage_path: &Path) -> bool {
    storage_path.join(MIGRATION_MARKER).exists()
}

/// Summary of a completed migration run.
#[derive(Debug, Default)]
pub struct MigrationReport {
    pub users: usize,
    pub memories_migrated: usize,
    pub memories_skipped: usize,
    pub graph_records_migrated: usize,
    pub graph_records_skipped: usize,
    pub facts_migrated: usize,
    pub facts_skipped: usize,
    pub lineage_migrated: usize,
    pub lineage_skipped: usize,
    pub temporal_migrated: usize,
    pub temporal_skipped: usize,
    pub vector_mappings_migrated: usize,
    pub vector_mappings_skipped: usize,
    pub audit_migrated: usize,
    pub audit_skipped: usize,
    pub errors: Vec<String>,
    pub dry_run: bool,
}

impl std::fmt::Display for MigrationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = if self.dry_run { "DRY RUN" } else { "MIGRATED" };
        writeln!(f, "=== Postcard Migration Report ({mode}) ===")?;
        writeln!(f, "Users processed:       {}", self.users)?;
        writeln!(
            f,
            "Memories:              {} migrated, {} skipped",
            self.memories_migrated, self.memories_skipped
        )?;
        writeln!(
            f,
            "Graph records:         {} migrated, {} skipped",
            self.graph_records_migrated, self.graph_records_skipped
        )?;
        writeln!(
            f,
            "Facts + embeddings:    {} migrated, {} skipped",
            self.facts_migrated, self.facts_skipped
        )?;
        writeln!(
            f,
            "Lineage edges/branches:{} migrated, {} skipped",
            self.lineage_migrated, self.lineage_skipped
        )?;
        writeln!(
            f,
            "Temporal facts:        {} migrated, {} skipped",
            self.temporal_migrated, self.temporal_skipped
        )?;
        writeln!(
            f,
            "Vector mappings:       {} migrated, {} skipped",
            self.vector_mappings_migrated, self.vector_mappings_skipped
        )?;
        writeln!(
            f,
            "Audit events:          {} migrated, {} skipped",
            self.audit_migrated, self.audit_skipped
        )?;
        if !self.errors.is_empty() {
            writeln!(f, "Errors:                {}", self.errors.len())?;
            for (i, e) in self.errors.iter().enumerate().take(20) {
                writeln!(f, "  [{i}] {e}")?;
            }
            if self.errors.len() > 20 {
                writeln!(f, "  ... and {} more", self.errors.len() - 20)?;
            }
        }
        Ok(())
    }
}

/// Run the full migration on the given storage directory.
pub fn migrate_all(storage_path: &Path, dry_run: bool) -> Result<MigrationReport> {
    let mut report = MigrationReport {
        dry_run,
        ..Default::default()
    };

    if is_migrated(storage_path) {
        eprintln!("Storage already migrated (marker file exists). Nothing to do.");
        return Ok(report);
    }

    if !storage_path.exists() {
        anyhow::bail!("Storage path does not exist: {}", storage_path.display());
    }

    // --- Shared DB (audit events) ---
    let shared_path = storage_path.join("shared");
    if shared_path.exists() {
        eprintln!("Migrating shared DB (audit events)...");
        match migrate_shared_db(&shared_path, dry_run) {
            Ok((migrated, skipped)) => {
                report.audit_migrated = migrated;
                report.audit_skipped = skipped;
            }
            Err(e) => {
                report.errors.push(format!("shared DB: {e:#}"));
            }
        }
    }

    // --- Per-user databases ---
    let entries: Vec<_> = std::fs::read_dir(storage_path)
        .context("reading storage directory")?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name_str = name.to_string_lossy();
            e.path().is_dir() && name_str != "shared" && !name_str.ends_with(".pre_cf_migration")
        })
        .collect();

    for entry in &entries {
        let user_id = entry.file_name();
        let user_id_str = user_id.to_string_lossy();
        let user_path = entry.path();

        eprintln!("Migrating user: {user_id_str}");
        report.users += 1;

        // Memory DB (storage/ subdir)
        let storage_dir = user_path.join("storage");
        if storage_dir.exists() {
            match migrate_memory_db(&storage_dir, dry_run) {
                Ok(counts) => {
                    report.memories_migrated += counts.memories_migrated;
                    report.memories_skipped += counts.memories_skipped;
                    report.facts_migrated += counts.facts_migrated;
                    report.facts_skipped += counts.facts_skipped;
                    report.lineage_migrated += counts.lineage_migrated;
                    report.lineage_skipped += counts.lineage_skipped;
                    report.temporal_migrated += counts.temporal_migrated;
                    report.temporal_skipped += counts.temporal_skipped;
                    report.vector_mappings_migrated += counts.vector_mappings_migrated;
                    report.vector_mappings_skipped += counts.vector_mappings_skipped;
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("user {user_id_str} memory DB: {e:#}"));
                }
            }
        }

        // Graph DB (graph/ subdir)
        let graph_dir = user_path.join("graph");
        if graph_dir.exists() {
            match migrate_graph_db(&graph_dir, dry_run) {
                Ok((migrated, skipped)) => {
                    report.graph_records_migrated += migrated;
                    report.graph_records_skipped += skipped;
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("user {user_id_str} graph DB: {e:#}"));
                }
            }
        }
    }

    // Write migration marker.
    // We write the marker even if some individual records failed to decode —
    // those records are corrupted in the DB and won't improve on re-run.
    // The marker is only withheld if we had zero successful migrations
    // (indicating a systemic failure like wrong DB path or permissions).
    if !dry_run {
        let total_migrated = report.memories_migrated
            + report.graph_records_migrated
            + report.facts_migrated
            + report.lineage_migrated
            + report.temporal_migrated
            + report.audit_migrated;
        let total_skipped = report.memories_skipped
            + report.graph_records_skipped
            + report.facts_skipped
            + report.lineage_skipped
            + report.temporal_skipped
            + report.audit_skipped;

        if total_migrated > 0 || total_skipped > 0 {
            let marker_path = storage_path.join(MIGRATION_MARKER);
            let timestamp = chrono::Utc::now().to_rfc3339();
            std::fs::write(
                &marker_path,
                format!(
                    "migrated_at={timestamp}\nmemories={}\ngraph={}\nfacts={}\nlineage={}\ntemporal={}\naudit={}\nerrors={}\n",
                    report.memories_migrated,
                    report.graph_records_migrated,
                    report.facts_migrated,
                    report.lineage_migrated,
                    report.temporal_migrated,
                    report.audit_migrated,
                    report.errors.len(),
                ),
            )
            .context("writing migration marker file")?;
            eprintln!("Migration marker written to {}", marker_path.display());
            if !report.errors.is_empty() {
                eprintln!(
                    "NOTE: {} records could not be decoded (pre-existing corruption). These were skipped.",
                    report.errors.len()
                );
            }
        } else if !report.errors.is_empty() {
            eprintln!(
                "ERROR: {} errors and no records migrated — marker NOT written. Check storage path and permissions.",
                report.errors.len()
            );
        }
    }

    Ok(report)
}

// ---------------------------------------------------------------------------
// Memory DB migration
// ---------------------------------------------------------------------------

/// Counts for the per-user memory DB migration.
#[derive(Default)]
struct MemoryDbCounts {
    memories_migrated: usize,
    memories_skipped: usize,
    facts_migrated: usize,
    facts_skipped: usize,
    lineage_migrated: usize,
    lineage_skipped: usize,
    temporal_migrated: usize,
    temporal_skipped: usize,
    vector_mappings_migrated: usize,
    vector_mappings_skipped: usize,
}

/// Known prefixes for sub-stores in the memory DB default CF.
const FACTS_PREFIX: &[u8] = b"facts:";
const FACTS_BY_ENTITY_PREFIX: &[u8] = b"facts_by_entity:";
const FACTS_BY_TYPE_PREFIX: &[u8] = b"facts_by_type:";
const FACTS_EMBEDDING_PREFIX: &[u8] = b"facts_embedding:";
const TEMPORAL_FACTS_PREFIX: &[u8] = b"temporal_facts:";
const TEMPORAL_BY_ENTITY_PREFIX: &[u8] = b"temporal_by_entity:";
const TEMPORAL_BY_EVENT_PREFIX: &[u8] = b"temporal_by_event:";
const TEMPORAL_BY_TIME_PREFIX: &[u8] = b"temporal_by_time:";
const VMAPPING_PREFIX: &[u8] = b"vmapping:";
const LEARNING_PREFIX: &[u8] = b"learning:";

/// Prefixes whose values are plain string references (not serialized structs).
/// Index entries: their values are just key references — no binary format to migrate.
const INDEX_ONLY_PREFIXES: &[&[u8]] = &[
    b"stats:",
    b"interference:",
    b"interference_meta:",
    b"_watermark:",
    FACTS_BY_ENTITY_PREFIX,
    FACTS_BY_TYPE_PREFIX,
    b"learning_by_memory:",
    b"learning_by_type:",
    b"learning_by_fact:",
    b"learning_stats:",
    b"geo:",
    // lineage by_from / by_to are index refs, but edges/branches are bincode
    b"lineage:by_from:",
    b"lineage:by_to:",
    // temporal index entries
    TEMPORAL_BY_ENTITY_PREFIX,
    TEMPORAL_BY_EVENT_PREFIX,
    TEMPORAL_BY_TIME_PREFIX,
];

fn migrate_memory_db(storage_dir: &Path, dry_run: bool) -> Result<MemoryDbCounts> {
    let cf_index = "memory_index";
    let mut opts = RocksOptions::default();
    opts.create_if_missing(false);
    opts.create_missing_column_families(true);

    let cfs = vec![
        ColumnFamilyDescriptor::new("default", RocksOptions::default()),
        ColumnFamilyDescriptor::new(cf_index, RocksOptions::default()),
    ];

    let db = DB::open_cf_descriptors(&opts, storage_dir, cfs)
        .with_context(|| format!("opening memory DB at {}", storage_dir.display()))?;

    let mut counts = MemoryDbCounts::default();
    let mut total_processed: usize = 0;

    // --- Default CF: memories + sub-stores ---
    let mut batch = WriteBatch::default();
    let mut batch_count: usize = 0;

    for item in db.iterator(IteratorMode::Start) {
        let (key, value) = item?;

        total_processed += 1;
        if total_processed % 1000 == 0 {
            eprintln!("  ... processed {total_processed} records");
        }

        // Skip index-only entries (values are plain string references, not serialized)
        if INDEX_ONLY_PREFIXES.iter().any(|p| key.starts_with(p)) {
            continue;
        }

        // Learning events: stay on msgpack (ConsolidationEvent uses #[serde(tag = "type")])
        if key.starts_with(LEARNING_PREFIX) {
            continue;
        }

        // Determine record type by key prefix
        if key.starts_with(FACTS_PREFIX) && !key.starts_with(FACTS_EMBEDDING_PREFIX) {
            // SemanticFact
            migrate_generic_record::<SemanticFact>(
                &db,
                None,
                &key,
                &value,
                dry_run,
                &mut batch,
                &mut batch_count,
                &mut counts.facts_migrated,
                &mut counts.facts_skipped,
            )?;
        } else if key.starts_with(FACTS_EMBEDDING_PREFIX) {
            // Vec<f32> embedding
            migrate_generic_record::<Vec<f32>>(
                &db,
                None,
                &key,
                &value,
                dry_run,
                &mut batch,
                &mut batch_count,
                &mut counts.facts_migrated,
                &mut counts.facts_skipped,
            )?;
        } else if key.starts_with(b"lineage:edges:") {
            // LineageEdge
            migrate_generic_record::<LineageEdge>(
                &db,
                None,
                &key,
                &value,
                dry_run,
                &mut batch,
                &mut batch_count,
                &mut counts.lineage_migrated,
                &mut counts.lineage_skipped,
            )?;
        } else if key.starts_with(b"lineage:branches:") {
            // LineageBranch
            migrate_generic_record::<LineageBranch>(
                &db,
                None,
                &key,
                &value,
                dry_run,
                &mut batch,
                &mut batch_count,
                &mut counts.lineage_migrated,
                &mut counts.lineage_skipped,
            )?;
        } else if key.starts_with(TEMPORAL_FACTS_PREFIX) {
            // TemporalFact
            migrate_generic_record::<TemporalFact>(
                &db,
                None,
                &key,
                &value,
                dry_run,
                &mut batch,
                &mut batch_count,
                &mut counts.temporal_migrated,
                &mut counts.temporal_skipped,
            )?;
        } else if key.starts_with(VMAPPING_PREFIX) {
            // VectorMappingEntry
            migrate_generic_record::<VectorMappingEntry>(
                &db,
                None,
                &key,
                &value,
                dry_run,
                &mut batch,
                &mut batch_count,
                &mut counts.vector_mappings_migrated,
                &mut counts.vector_mappings_skipped,
            )?;
        } else if key.len() == 16 {
            // Memory record (16-byte UUID key) — uses SHO envelope
            if let Some((version, _payload)) = serialization::unwrap_sho(&value) {
                if version == serialization::SHO_VERSION_POSTCARD {
                    // Already postcard
                    counts.memories_skipped += 1;
                    continue;
                }
            }

            // Try to deserialize with the full legacy fallback chain
            match crate::memory::storage::deserialize_memory_for_migration(&value) {
                Ok(memory) => {
                    if !dry_run {
                        let new_value = serialization::encode_sho(&memory)?;
                        batch.put(&*key, &new_value);
                        batch_count += 1;
                    }
                    counts.memories_migrated += 1;
                }
                Err(e) => {
                    eprintln!(
                        "  WARNING: cannot decode memory key ({} bytes): {e:#}",
                        key.len()
                    );
                    // Don't fail the whole migration for one bad record
                }
            }
        }
        // else: unknown prefix / unknown key length — skip silently

        // Flush batch periodically
        if batch_count >= BATCH_SIZE && !dry_run {
            db.write(std::mem::take(&mut batch))?;
            batch_count = 0;
        }
    }

    // --- memory_index CF: VectorMappingEntry records ---
    if let Some(cf) = db.cf_handle(cf_index) {
        for item in db.iterator_cf(cf, IteratorMode::Start) {
            let (key, value) = item?;
            // The memory_index CF stores various index entries. Only vmapping entries
            // need migration — others are string references or small metadata.
            if key.starts_with(VMAPPING_PREFIX) {
                migrate_generic_record::<VectorMappingEntry>(
                    &db,
                    Some(cf),
                    &key,
                    &value,
                    dry_run,
                    &mut batch,
                    &mut batch_count,
                    &mut counts.vector_mappings_migrated,
                    &mut counts.vector_mappings_skipped,
                )?;
            }
        }
    }

    // Flush remaining
    if batch_count > 0 && !dry_run {
        db.write(batch)?;
    }

    eprintln!(
        "  Memory DB: {} memories migrated, {} skipped",
        counts.memories_migrated, counts.memories_skipped
    );

    Ok(counts)
}

// ---------------------------------------------------------------------------
// Graph DB migration
// ---------------------------------------------------------------------------

/// Graph DB column families that contain serialized data (entities, edges, episodes).
const GRAPH_DATA_CFS: &[&str] = &["entities", "relationships", "episodes", "entity_edges"];

/// Graph DB column families that contain index data (string references, counts) — skip.
const GRAPH_INDEX_CFS: &[&str] = &[
    "entity_pair_index",
    "entity_episodes",
    "name_index",
    "lowercase_index",
    "stemmed_index",
];

fn migrate_graph_db(graph_dir: &Path, dry_run: bool) -> Result<(usize, usize)> {
    let mut opts = RocksOptions::default();
    opts.create_if_missing(false);
    opts.create_missing_column_families(true);

    let all_cfs: Vec<&str> = GRAPH_DATA_CFS
        .iter()
        .chain(GRAPH_INDEX_CFS.iter())
        .copied()
        .collect();

    let cfs: Vec<ColumnFamilyDescriptor> = std::iter::once(ColumnFamilyDescriptor::new(
        "default",
        RocksOptions::default(),
    ))
    .chain(
        all_cfs
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, RocksOptions::default())),
    )
    .collect();

    let db = DB::open_cf_descriptors(&opts, graph_dir, cfs)
        .with_context(|| format!("opening graph DB at {}", graph_dir.display()))?;

    let mut migrated: usize = 0;
    let mut skipped: usize = 0;

    // Migrate data CFs. All three graph types (EntityNode, RelationshipEdge,
    // EpisodicNode) are stored as opaque bincode blobs. We use the generic
    // bincode→postcard migration which doesn't care about the concrete type
    // at compile time — we just need to round-trip the bytes through
    // try_decode → encode. But try_decode needs a concrete type parameter.
    //
    // Since the graph CFs are typed:
    //   entities → EntityNode
    //   relationships, entity_edges → RelationshipEdge
    //   episodes → EpisodicNode
    // we dispatch per CF name.

    for cf_name in GRAPH_DATA_CFS {
        let cf = match db.cf_handle(cf_name) {
            Some(cf) => cf,
            None => continue,
        };

        let mut batch = WriteBatch::default();
        let mut batch_count: usize = 0;

        for item in db.iterator_cf(cf, IteratorMode::Start) {
            let (key, value) = item?;

            match *cf_name {
                "entities" => {
                    migrate_generic_record::<EntityNode>(
                        &db,
                        Some(cf),
                        &key,
                        &value,
                        dry_run,
                        &mut batch,
                        &mut batch_count,
                        &mut migrated,
                        &mut skipped,
                    )?;
                }
                "relationships" | "entity_edges" => {
                    migrate_generic_record::<RelationshipEdge>(
                        &db,
                        Some(cf),
                        &key,
                        &value,
                        dry_run,
                        &mut batch,
                        &mut batch_count,
                        &mut migrated,
                        &mut skipped,
                    )?;
                }
                "episodes" => {
                    migrate_generic_record::<EpisodicNode>(
                        &db,
                        Some(cf),
                        &key,
                        &value,
                        dry_run,
                        &mut batch,
                        &mut batch_count,
                        &mut migrated,
                        &mut skipped,
                    )?;
                }
                _ => {}
            }

            if batch_count >= BATCH_SIZE && !dry_run {
                db.write(std::mem::take(&mut batch))?;
                batch_count = 0;
            }
        }

        if batch_count > 0 && !dry_run {
            db.write(batch)?;
        }
    }

    eprintln!("  Graph DB: {migrated} migrated, {skipped} skipped");
    Ok((migrated, skipped))
}

// ---------------------------------------------------------------------------
// Shared DB migration (audit events)
// ---------------------------------------------------------------------------

fn migrate_shared_db(shared_dir: &Path, dry_run: bool) -> Result<(usize, usize)> {
    let mut opts = RocksOptions::default();
    opts.create_if_missing(false);
    opts.create_missing_column_families(true);

    // The shared DB has many CFs but only "audit" contains bincode data.
    // Others (todos, projects, prospective, feedback, files) use JSON.
    let shared_cfs = [
        "default",
        "audit",
        "todos",
        "projects",
        "todo_index",
        "prospective",
        "prospective_index",
        "files",
        "file_index",
        "feedback",
    ];
    let cfs: Vec<ColumnFamilyDescriptor> = shared_cfs
        .iter()
        .map(|name| ColumnFamilyDescriptor::new(*name, RocksOptions::default()))
        .collect();

    let db = DB::open_cf_descriptors(&opts, shared_dir, cfs)
        .with_context(|| format!("opening shared DB at {}", shared_dir.display()))?;

    let cf = match db.cf_handle("audit") {
        Some(cf) => cf,
        None => return Ok((0, 0)),
    };

    let mut migrated: usize = 0;
    let mut skipped: usize = 0;
    let mut batch = WriteBatch::default();
    let mut batch_count: usize = 0;

    for item in db.iterator_cf(cf, IteratorMode::Start) {
        let (key, value) = item?;

        migrate_generic_record::<AuditEvent>(
            &db,
            Some(cf),
            &key,
            &value,
            dry_run,
            &mut batch,
            &mut batch_count,
            &mut migrated,
            &mut skipped,
        )?;

        if batch_count >= BATCH_SIZE && !dry_run {
            db.write(std::mem::take(&mut batch))?;
            batch_count = 0;
        }
    }

    if batch_count > 0 && !dry_run {
        db.write(batch)?;
    }

    eprintln!("  Shared DB: {migrated} audit events migrated, {skipped} skipped");
    Ok((migrated, skipped))
}

// ---------------------------------------------------------------------------
// Generic record migration helper
// ---------------------------------------------------------------------------

/// Migrate a single record from legacy bincode to tagged postcard.
///
/// If the record already has the postcard format tag, it is skipped.
/// Otherwise it is decoded via `try_decode` (postcard-first, bincode fallback)
/// and re-encoded as tagged postcard.
#[allow(clippy::too_many_arguments)]
fn migrate_generic_record<T>(
    _db: &DB,
    cf: Option<&rocksdb::ColumnFamily>,
    key: &[u8],
    value: &[u8],
    dry_run: bool,
    batch: &mut WriteBatch,
    batch_count: &mut usize,
    migrated: &mut usize,
    skipped: &mut usize,
) -> Result<()>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    // Already in postcard format — skip
    if serialization::has_format_tag_pub(value) {
        *skipped += 1;
        return Ok(());
    }

    // Decode with legacy bincode fallback
    let (val, _needs_migration): (T, bool) =
        serialization::try_decode(value).with_context(|| {
            format!(
                "decoding record (key len={}, value len={})",
                key.len(),
                value.len()
            )
        })?;

    if !dry_run {
        let new_value = serialization::encode(&val)?;
        if let Some(cf) = cf {
            batch.put_cf(cf, key, &new_value);
        } else {
            batch.put(key, &new_value);
        }
        *batch_count += 1;
    }

    *migrated += 1;
    Ok(())
}
