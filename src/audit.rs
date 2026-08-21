//! Read-only audit of the RocksDB stores.
//!
//! Classifies every record in the memory, graph, and shared DBs as
//! decodable (current schema), legacy (decodes via fallback, needs
//! migration), or undecodable (current and fallback schemas both fail).
//! Undecodable records are archived with their raw bytes (hex) to a JSONL
//! file so nothing is lost before any purge/migration decision.
//!
//! Mirror of the iteration/dispatch logic in `migration.rs`, with the
//! difference that tagged/postcard records are decode-VERIFIED instead of
//! being skipped on sight (the blind spot that let schema drift go
//! undetected for two days).

use anyhow::{Context, Result};
use rocksdb::{ColumnFamilyDescriptor, IteratorMode, Options as RocksOptions, DB};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::graph_memory::{EntityNode, EpisodicNode, RelationshipEdge};
use crate::handlers::types::AuditEvent;
use crate::memory::compression::SemanticFact;
use crate::memory::lineage::{LineageBranch, LineageEdge};
use crate::memory::storage::{deserialize_memory_for_migration, VectorMappingEntry, CF_OPLOG};
use crate::memory::temporal_facts::TemporalFact;
use crate::memory::types::Memory;
use crate::graph_memory::GRAPH_CF_NAMES;
use crate::migration::{
    FACTS_EMBEDDING_PREFIX, FACTS_PREFIX, GRAPH_DATA_CFS, INDEX_ONLY_PREFIXES, LEARNING_PREFIX,
    TEMPORAL_FACTS_PREFIX, VMAPPING_PREFIX,
};
use crate::serialization;

const STORAGE_MAGIC: &[u8] = b"SHO";

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct AuditReport {
    pub shared: DbAudit,
    pub users: Vec<UserAudit>,
    pub archive_path: Option<String>,
    pub archived_records: usize,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct UserAudit {
    pub user: String,
    pub memory: DbAudit,
    pub graph: DbAudit,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct DbAudit {
    pub total: usize,
    pub decodable: usize,
    pub legacy: usize,
    pub undecodable: usize,
    pub skipped: usize,
    pub cfs: Vec<CfAudit>,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct CfAudit {
    pub cf: String,
    pub total: usize,
    pub decodable: usize,
    pub legacy: usize,
    pub undecodable: usize,
    pub skipped: usize,
    pub errors: BTreeMap<String, usize>,
}

impl std::fmt::Display for AuditReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "==== shodh-memory audit ====")?;
        writeln!(
            f,
            "shared:   total={} decodable={} legacy={} undecodable={} skipped={}",
            self.shared.total,
            self.shared.decodable,
            self.shared.legacy,
            self.shared.undecodable,
            self.shared.skipped
        )?;
        for u in &self.users {
            writeln!(f, "user {}:", u.user)?;
            writeln!(
                f,
                "  memory: total={} decodable={} legacy={} undecodable={} skipped={}",
                u.memory.total, u.memory.decodable, u.memory.legacy, u.memory.undecodable, u.memory.skipped
            )?;
            for cf in &u.memory.cfs {
                writeln!(
                    f,
                    "    cf {:<16} total={:<6} decodable={:<6} legacy={:<6} undecodable={:<6} skipped={}",
                    cf.cf, cf.total, cf.decodable, cf.legacy, cf.undecodable, cf.skipped
                )?;
                for (err, n) in &cf.errors {
                    writeln!(f, "      x{:<4} {err}", n)?;
                }
            }
            writeln!(
                f,
                "  graph:  total={} decodable={} legacy={} undecodable={} skipped={}",
                u.graph.total, u.graph.decodable, u.graph.legacy, u.graph.undecodable, u.graph.skipped
            )?;
            for cf in &u.graph.cfs {
                writeln!(
                    f,
                    "    cf {:<16} total={:<6} decodable={:<6} legacy={:<6} undecodable={:<6} skipped={}",
                    cf.cf, cf.total, cf.decodable, cf.legacy, cf.undecodable, cf.skipped
                )?;
                for (err, n) in &cf.errors {
                    writeln!(f, "      x{:<4} {err}", n)?;
                }
            }
        }
        writeln!(
            f,
            "archive: {} ({} undecodable records)",
            self.archive_path.as_deref().unwrap_or("(none)"),
            self.archived_records
        )
    }
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

enum Verdict {
    Decodable,
    Legacy,
    Undecodable(String),
}

fn classify_memory(value: &[u8]) -> Verdict {
    match serialization::unwrap_sho(value) {
        Some((version, payload)) => {
            if version == serialization::SHO_VERSION_POSTCARD {
                match postcard::from_bytes::<Memory>(payload) {
                    Ok(_) => Verdict::Decodable,
                    Err(e) => Verdict::Undecodable(format!("SHO v2 postcard payload: {e}")),
                }
            } else {
                match deserialize_memory_for_migration(value) {
                    Ok(_) => Verdict::Legacy,
                    Err(e) => Verdict::Undecodable(format!("SHO v{version} legacy: {e:#}")),
                }
            }
        }
        None => {
            if value.len() >= 3 && &value[..3] == STORAGE_MAGIC {
                Verdict::Undecodable("SHO magic present but envelope/CRC invalid".into())
            } else {
                match deserialize_memory_for_migration(value) {
                    Ok(_) => Verdict::Legacy,
                    Err(e) => Verdict::Undecodable(format!("legacy chain: {e:#}")),
                }
            }
        }
    }
}

fn classify_generic<T: serde::de::DeserializeOwned>(value: &[u8]) -> Verdict {
    match serialization::try_decode::<T>(value) {
        // try_decode returns (val, needs_migration): true = legacy bincode,
        // false = current tagged postcard.
        Ok((_, true)) => Verdict::Legacy,
        Ok((_, false)) => Verdict::Decodable,
        Err(e) => Verdict::Undecodable(format!("{e}")),
    }
}

// ---------------------------------------------------------------------------
// Archive (raw bytes of undecodable records)
// ---------------------------------------------------------------------------

struct Archive {
    path: Option<PathBuf>,
    writer: Option<std::fs::File>,
    count: usize,
}

impl Archive {
    fn open(path: Option<&Path>) -> Result<Self> {
        let writer = match path {
            Some(p) => Some(std::fs::File::create(p).with_context(|| format!("creating archive {}", p.display()))?),
            None => None,
        };
        Ok(Self {
            path: path.map(|p| p.to_path_buf()),
            writer,
            count: 0,
        })
    }

    fn push(&mut self, db: &str, cf: &str, key: &[u8], value: &[u8], error: &str) -> Result<()> {
        self.count += 1;
        if let Some(w) = self.writer.as_mut() {
            let rec = serde_json::json!({
                "db": db,
                "cf": cf,
                "key_hex": hex(key),
                "value_hex": hex(value),
                "value_len": value.len(),
                "error": error,
            });
            serde_json::to_writer(&mut *w, &rec)?;
            w.write_all(b"\n")?;
        }
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// Per-DB audits
// ---------------------------------------------------------------------------

fn open_db(db_dir: &Path, cfs: &[&str]) -> Result<DB> {
    let mut opts = RocksOptions::default();
    opts.create_if_missing(false);
    opts.create_missing_column_families(true);
    // Read-only open: the audit must be able to run against a live data
    // dir (the serving process holds the RocksDB lock). Read-only mode
    // skips the exclusive lock; error_if_log_file_exist=false tolerates
    // the live WAL.
    DB::open_cf_for_read_only(&opts, db_dir, cfs.iter().copied(), false)
        .with_context(|| format!("opening DB at {}", db_dir.display()))
}

fn audit_shared_db(shared_dir: &Path, archive: &mut Archive) -> Result<DbAudit> {
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
    let db = open_db(shared_dir, &shared_cfs)?;
    let mut audit = DbAudit::default();

    for cf_name in shared_cfs {
        let cf = match db.cf_handle(cf_name) {
            Some(cf) => cf,
            None => continue,
        };
        let mut cf_audit = CfAudit {
            cf: cf_name.to_string(),
            ..Default::default()
        };
        for item in db.iterator_cf(cf, IteratorMode::Start) {
            let (key, value) = item?;
            cf_audit.total += 1;
            if cf_name == "audit" {
                match classify_generic::<AuditEvent>(&value) {
                    Verdict::Decodable => cf_audit.decodable += 1,
                    Verdict::Legacy => cf_audit.legacy += 1,
                    Verdict::Undecodable(e) => {
                        cf_audit.undecodable += 1;
                        *cf_audit.errors.entry(e.clone()).or_insert(0) += 1;
                        archive.push("shared", cf_name, &key, &value, &e)?;
                    }
                }
            } else {
                cf_audit.skipped += 1;
            }
        }
        audit.total += cf_audit.total;
        audit.decodable += cf_audit.decodable;
        audit.legacy += cf_audit.legacy;
        audit.undecodable += cf_audit.undecodable;
        audit.skipped += cf_audit.skipped;
        audit.cfs.push(cf_audit);
    }
    Ok(audit)
}

fn audit_memory_db(mem_dir: &Path, archive: &mut Archive) -> Result<DbAudit> {
    let cfs = ["default", "memory_index", CF_OPLOG];
    let db = open_db(mem_dir, &cfs)?;
    let mut audit = DbAudit::default();

    for cf_name in cfs {
        let cf = match db.cf_handle(cf_name) {
            Some(cf) => cf,
            None => continue,
        };
        let mut cf_audit = CfAudit {
            cf: cf_name.to_string(),
            ..Default::default()
        };
        for item in db.iterator_cf(cf, IteratorMode::Start) {
            let (key, value) = item?;
            cf_audit.total += 1;
            if cf_name != "default" {
                cf_audit.skipped += 1;
                continue;
            }
            let verdict = if key.len() == 16 {
                classify_memory(&value)
            } else if key.starts_with(FACTS_PREFIX) {
                classify_generic::<SemanticFact>(&value)
            } else if key.starts_with(FACTS_EMBEDDING_PREFIX) {
                classify_generic::<Vec<f32>>(&value)
            } else if key.starts_with(b"lineage:edges:") {
                classify_generic::<LineageEdge>(&value)
            } else if key.starts_with(b"lineage:branches:") {
                classify_generic::<LineageBranch>(&value)
            } else if key.starts_with(TEMPORAL_FACTS_PREFIX) {
                classify_generic::<TemporalFact>(&value)
            } else if key.starts_with(VMAPPING_PREFIX) {
                classify_generic::<VectorMappingEntry>(&value)
            } else if key.starts_with(LEARNING_PREFIX)
                || INDEX_ONLY_PREFIXES.iter().any(|p| key.starts_with(p))
            {
                Verdict::Decodable // index/ref records: nothing to verify
            } else {
                Verdict::Decodable // opaque/unknown: not an error
            };
            match verdict {
                Verdict::Decodable => cf_audit.decodable += 1,
                Verdict::Legacy => cf_audit.legacy += 1,
                Verdict::Undecodable(e) => {
                    cf_audit.undecodable += 1;
                    *cf_audit.errors.entry(e.clone()).or_insert(0) += 1;
                    archive.push("memory", cf_name, &key, &value, &e)?;
                }
            }
        }
        audit.total += cf_audit.total;
        audit.decodable += cf_audit.decodable;
        audit.legacy += cf_audit.legacy;
        audit.undecodable += cf_audit.undecodable;
        audit.skipped += cf_audit.skipped;
        audit.cfs.push(cf_audit);
    }
    Ok(audit)
}

fn audit_graph_db(graph_dir: &Path, archive: &mut Archive) -> Result<DbAudit> {
    let all_cfs: Vec<&str> = GRAPH_CF_NAMES.to_vec();
    let cfs: Vec<&str> = std::iter::once("default").chain(all_cfs.iter().copied()).collect();
    let db = open_db(graph_dir, &cfs)?;
    let mut audit = DbAudit::default();

    for cf_name in cfs {
        let cf = match db.cf_handle(cf_name) {
            Some(cf) => cf,
            None => continue,
        };
        let mut cf_audit = CfAudit {
            cf: cf_name.to_string(),
            ..Default::default()
        };
        let data_cf = GRAPH_DATA_CFS.contains(&cf_name);
        for item in db.iterator_cf(cf, IteratorMode::Start) {
            let (key, value) = item?;
            cf_audit.total += 1;
            let verdict = if !data_cf {
                Verdict::Decodable // index CF: skipped by design
            } else {
                match cf_name {
                    "entities" => classify_generic::<EntityNode>(&value),
                    "relationships" | "entity_edges" => classify_generic::<RelationshipEdge>(&value),
                    "episodes" => classify_generic::<EpisodicNode>(&value),
                    _ => Verdict::Decodable,
                }
            };
            match verdict {
                Verdict::Decodable => cf_audit.decodable += 1,
                Verdict::Legacy => cf_audit.legacy += 1,
                Verdict::Undecodable(e) => {
                    cf_audit.undecodable += 1;
                    *cf_audit.errors.entry(e.clone()).or_insert(0) += 1;
                    archive.push("graph", cf_name, &key, &value, &e)?;
                }
            }
        }
        audit.total += cf_audit.total;
        audit.decodable += cf_audit.decodable;
        audit.legacy += cf_audit.legacy;
        audit.undecodable += cf_audit.undecodable;
        audit.skipped += cf_audit.skipped;
        audit.cfs.push(cf_audit);
    }
    Ok(audit)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn audit_all(storage_path: &Path, archive_path: Option<&Path>) -> Result<AuditReport> {
    let mut archive = Archive::open(archive_path)?;
    let mut report = AuditReport {
        archive_path: archive_path.map(|p| p.display().to_string()),
        ..Default::default()
    };

    let shared_dir = storage_path.join("shared");
    if shared_dir.is_dir() {
        report.shared = audit_shared_db(&shared_dir, &mut archive)?;
    }

    let mut users: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(storage_path) {
        for entry in rd.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') && name != "backups" && entry.path().join("storage").is_dir() {
                    users.push(name);
                }
            }
        }
    }
    users.sort();

    for user in users {
        let base = storage_path.join(&user);
        let mut ua = UserAudit {
            user,
            memory: DbAudit::default(),
            graph: DbAudit::default(),
        };
        let mem_dir = base.join("storage");
        if mem_dir.is_dir() {
            ua.memory = audit_memory_db(&mem_dir, &mut archive)?;
        }
        let graph_dir = base.join("graph").join("graph");
        if graph_dir.is_dir() {
            ua.graph = audit_graph_db(&graph_dir, &mut archive)?;
        }
        report.users.push(ua);
    }

    report.archived_records = archive.count;
    Ok(report)
}
