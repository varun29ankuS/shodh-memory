//! The source registry: the durable list of things that produce data for this
//! store, and how far ingestion has got with each one.
//!
//! # The rule this store exists to obey
//!
//! `SessionStore` (`src/memory/sessions.rs`) keeps its sessions in two
//! `RwLock<HashMap>`s and never reads or writes disk. The measured consequence
//! is that a profile holding 18,032 memories and 230 recorded sessions answers
//! `{"sessions":[],"count":0}` immediately after a restart. The defect is not
//! the `RwLock<HashMap>` — the audit log has an in-memory `VecDeque` and is
//! fine, because RocksDB is the authority and the deque is a cache. The defect
//! is that the **only** copy of the state was in the process.
//!
//! > **No registry fact — a source definition, a cursor, a run record, a
//! > failure counter — may have its only copy in process memory.** Every read
//! > path in this module answers from RocksDB. There is nothing to "hydrate on
//! > startup" because nothing is ever loaded into an authoritative map.
//!
//! In-memory structures remain permitted as *caches* and as *locks*; the
//! per-source run lock on `MultiUserMemoryManager` is the latter.
//!
//! # Why two new column families in the shared DB
//!
//! Not a per-user `MemoryStorage`: per-user DBs sit behind a `moka` LRU with
//! idle eviction, so anything that needs to see every source for every user
//! would have to force-open every user's RocksDB.
//!
//! Not the `audit` CF: that CF *actively deletes*. `rotate_user_audit_logs`
//! removes entries past `audit_retention_days` and past
//! `audit_max_entries_per_user`. A source definition that evaporates after 30
//! days is not a registry. The audit CF still gets a human-timeline entry per
//! run — it is just never the authority.
//!
//! Not `CF_OPLOG`: it is a hash-chained, per-user, per-*session*,
//! tamper-evident log with a documented wire contract that forbids adding
//! fields. It answers "was this history altered", not "where is this source up
//! to". If ingestion ever wants tamper-evidence the answer is appending
//! witnessed records to that chain, not growing a second one here.
//!
//! Not a JSON file: committing "this item is ingested" and "this run's
//! counters" together needs a `WriteBatch`. A file gives torn writes and a
//! rewrite-the-world cost per item.
//!
//! Two CFs rather than one because cursors are high-cardinality — one key per
//! file, so a 50k-document folder is 50k keys — while definitions and run
//! records are low-cardinality and are scanned whole on every dashboard poll.
//! Sharing a CF would make that listing walk every cursor.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, Options, WriteBatch, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Definitions, run history, leases and per-source runtime state.
pub const CF_SOURCES: &str = "sources";
/// One key per tracked item. Written on the hot path of every run.
pub const CF_SOURCE_CURSOR: &str = "source_cursor";

/// Run records kept per source. Older ones are pruned when a run finishes.
pub const RUN_HISTORY_KEEP: usize = 50;
/// Verbatim failures carried on a run record. `items_failed` is the true count;
/// this is a sample and the API says so on the wire.
pub const RUN_FAILURE_SAMPLE: usize = 50;
/// Consecutive failures on one item before it stops being retried.
pub const QUARANTINE_THRESHOLD: u16 = 3;
/// Memory ids from earlier versions of an item kept on its cursor.
pub const SUPERSEDED_KEEP: usize = 20;

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Newtype over a source's UUID, mirroring `MemoryId` / `ProspectiveTaskId`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SourceId(pub uuid::Uuid);

impl std::fmt::Display for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Record types
// ---------------------------------------------------------------------------

/// What kind of thing a source is.
///
/// # Wire format: APPEND ONLY
///
/// Encoded by postcard as its declaration index. Reordering or removing a
/// variant silently re-points every stored definition at a different kind. New
/// kinds go at the END — the same contract `MemoryOrigin` carries, and
/// [`SourceKind::ALL`] is pinned to declaration order by a test for the same
/// reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    WatchedFolder,
}

impl SourceKind {
    /// Stable wire name. Written by hand so renaming the Rust variant cannot
    /// silently change the API contract.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WatchedFolder => "watched_folder",
        }
    }

    /// Every variant in declaration (and therefore postcard-discriminant)
    /// order.
    pub const ALL: &'static [Self] = &[Self::WatchedFolder];

    pub fn parse(s: &str) -> Option<Self> {
        let normalized = s.trim().to_ascii_lowercase().replace('-', "_");
        Self::ALL.iter().copied().find(|k| k.as_str() == normalized)
    }
}

/// Per-kind configuration. One variant per [`SourceKind`], same APPEND-ONLY
/// rule, and deliberately **externally** tagged: postcard has no
/// `deserialize_any`, so serde's `#[serde(tag = "...")]` representation cannot
/// round-trip through it. The JSON shape the API speaks is built separately in
/// the handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceConfig {
    WatchedFolder(WatchedFolderConfig),
}

impl SourceConfig {
    pub fn kind(&self) -> SourceKind {
        match self {
            Self::WatchedFolder(_) => SourceKind::WatchedFolder,
        }
    }

    pub fn as_watched_folder(&self) -> &WatchedFolderConfig {
        match self {
            Self::WatchedFolder(c) => c,
        }
    }
}

/// Default include globs: text a person wrote, not a build artefact.
pub fn default_include_globs() -> Vec<String> {
    vec!["**/*.md".to_string(), "**/*.txt".to_string()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchedFolderConfig {
    /// The CANONICAL root, resolved once at registration and stored resolved.
    /// Never re-resolved at run time: re-resolving would let a path that
    /// passed the deny-list at registration become a different directory
    /// later.
    pub root: String,
    /// Globs applied to the ROOT-RELATIVE path of an already-enumerated entry.
    /// **Never used to construct a path.** See the security note on
    /// `crate::ingest::folder`.
    pub include_globs: Vec<String>,
    /// User exclusions. The credential deny-list is applied on top of these
    /// and is not user-editable.
    pub exclude_globs: Vec<String>,
    pub max_depth: u16,
    pub max_files_per_run: u32,
    pub max_file_bytes: u64,
    pub max_run_bytes: u64,
    /// Every N-th run ignores the (size, mtime) fast path and re-reads
    /// everything. 0 = never. Covers the case borg's cache TTL covers: content
    /// changed while size and mtime were restored.
    pub rehash_every_n_runs: u32,
    /// `ExperienceType` stamped on every memory this source writes.
    pub memory_type: String,
    /// Tags added to every memory this source writes.
    pub tags: Vec<String>,
}

impl WatchedFolderConfig {
    /// Defaults chosen to be survivable rather than generous: a first run on a
    /// mis-pointed folder stops at 2,000 files and 256 MiB and says so.
    pub fn with_root(root: String) -> Self {
        Self {
            root,
            include_globs: default_include_globs(),
            exclude_globs: Vec::new(),
            max_depth: 8,
            max_files_per_run: 2_000,
            max_file_bytes: 1_048_576,
            max_run_bytes: 268_435_456,
            rehash_every_n_runs: 0,
            memory_type: "observation".to_string(),
            tags: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDefinition {
    pub id: SourceId,
    pub user_id: String,
    /// Human label, unique per user.
    pub name: String,
    pub kind: SourceKind,
    pub config: SourceConfig,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Bumped when the record's semantics change. Starts at 1.
    pub schema_version: u16,
}

/// What has happened to one item — for a folder source, one file — under one
/// source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemCursor {
    /// Root-relative, `/`-separated. The only human-readable copy of the path:
    /// the key holds a hash of it, so a Windows path's `:` and `\` can never
    /// corrupt a delimited key and key size is bounded regardless of depth.
    pub path: String,
    pub size_bytes: u64,
    /// mtime in nanoseconds since the Unix epoch. `None` when the platform
    /// refused to report one, which forces the slow path — a missing signal is
    /// never read as "unchanged".
    pub mtime_unix_nanos: Option<i64>,
    /// sha256 of the RAW FILE BYTES, hex. Distinct from the store's own
    /// `content_hash`, which is sha256 of the memory's content string.
    pub content_sha256: String,
    /// Memories this version of the item produced, in part order. Length > 1
    /// means the file was split.
    pub memory_ids: Vec<uuid::Uuid>,
    /// Memory ids earlier versions produced, newest first, capped at
    /// [`SUPERSEDED_KEEP`]. Without this, overwriting `memory_ids` is the
    /// moment the registry loses the only link between a stale memory and the
    /// file it came from.
    pub superseded_memory_ids: Vec<uuid::Uuid>,
    pub first_ingested_at: DateTime<Utc>,
    pub last_ingested_at: DateTime<Utc>,
    /// Updated on every run that ENUMERATED this item, even when unchanged.
    /// This is what makes "disappeared" detectable.
    pub last_seen_at: DateTime<Utc>,
    pub last_run_id: uuid::Uuid,
    pub state: ItemState,
    /// Consecutive failures. Reset to 0 on any success. Drives quarantine.
    pub consecutive_failures: u16,
    /// The `external_id` this item's single-part memory is bound to, when the
    /// current version was written by upsert. `None` for a split document,
    /// whose parts are appended and carry no external id.
    pub external_id: Option<String>,
}

/// Externally tagged for the same postcard reason as [`SourceConfig`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ItemState {
    /// Written, as a new memory or as a new version of one.
    Ingested,
    /// Content identical to a memory already in the store; no new memory made.
    Deduped,
    /// Deliberately not ingested and NOT an error: too large, binary, not
    /// UTF-8, denied by a security rule. Retried only when size or mtime move.
    Skipped { reason: String },
    /// Transient failure. Retried on the next run.
    Failed { reason: String },
    /// Failed [`QUARANTINE_THRESHOLD`] runs running. Never retried
    /// automatically; cleared by a forced run.
    Quarantined { reason: String },
}

impl ItemState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ingested => "ingested",
            Self::Deduped => "deduped",
            Self::Skipped { .. } => "skipped",
            Self::Failed { .. } => "failed",
            Self::Quarantined { .. } => "quarantined",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Ingested | Self::Deduped => None,
            Self::Skipped { reason } | Self::Failed { reason } | Self::Quarantined { reason } => {
                Some(reason)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunTrigger {
    Manual,
}

impl RunTrigger {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Manual => "manual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    Running,
    Succeeded,
    PartiallyFailed,
    Failed,
    Aborted,
}

impl RunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::PartiallyFailed => "partially_failed",
            Self::Failed => "failed",
            Self::Aborted => "aborted",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunFailure {
    /// Root-relative path.
    pub item: String,
    pub reason: String,
    pub at: DateTime<Utc>,
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRun {
    pub run_id: uuid::Uuid,
    pub source_id: SourceId,
    pub user_id: String,
    pub trigger: RunTrigger,
    pub started_at: DateTime<Utc>,
    /// The exact integer this run's key was built from. Carried so nothing
    /// ever has to reconstruct the key by re-formatting `started_at`.
    pub started_nanos: i64,
    pub finished_at: Option<DateTime<Utc>>,
    pub status: RunStatus,
    pub items_seen: u32,
    pub items_unchanged: u32,
    pub items_ingested: u32,
    pub items_deduped: u32,
    pub items_skipped: u32,
    pub items_failed: u32,
    pub items_disappeared: u32,
    /// Entries a security rule refused before any read: a symlink or reparse
    /// point, a credential-shaped name, an escape from the canonical root.
    /// Separate from `items_skipped` because "what is this thing reading" is
    /// the first question asked of a connector on a machine holding a corpus,
    /// and it cannot be answered by a number that also counts big files.
    pub items_denied_by_policy: u32,
    pub memories_written: u32,
    pub bytes_read: u64,
    /// First [`RUN_FAILURE_SAMPLE`] failures verbatim.
    pub failures: Vec<RunFailure>,
    /// Set only on a run-level fatal: root gone, permission denied on root.
    pub error: Option<String>,
    /// Set when a cap ended the run early. Never a silent truncation.
    pub truncated_by: Option<String>,
}

impl SourceRun {
    pub fn start(def: &SourceDefinition, trigger: RunTrigger, started_at: DateTime<Utc>) -> Self {
        Self {
            run_id: uuid::Uuid::new_v4(),
            source_id: def.id.clone(),
            user_id: def.user_id.clone(),
            trigger,
            started_at,
            started_nanos: started_at
                .timestamp_nanos_opt()
                .unwrap_or_else(|| started_at.timestamp_millis().saturating_mul(1_000_000)),
            finished_at: None,
            status: RunStatus::Running,
            items_seen: 0,
            items_unchanged: 0,
            items_ingested: 0,
            items_deduped: 0,
            items_skipped: 0,
            items_failed: 0,
            items_disappeared: 0,
            items_denied_by_policy: 0,
            memories_written: 0,
            bytes_read: 0,
            failures: Vec::new(),
            error: None,
            truncated_by: None,
        }
    }

    /// Record a failure, keeping the first [`RUN_FAILURE_SAMPLE`] verbatim.
    /// `items_failed` is incremented by the caller and is the true count.
    pub fn push_failure(&mut self, item: &str, reason: &str, retryable: bool) {
        if self.failures.len() < RUN_FAILURE_SAMPLE {
            self.failures.push(RunFailure {
                item: item.to_string(),
                reason: reason.to_string(),
                at: Utc::now(),
                retryable,
            });
        }
    }
}

/// Presence of this key means a run believes it is in flight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunLease {
    pub run_id: uuid::Uuid,
    pub started_at: DateTime<Utc>,
    /// The EXACT integer used to build this run's `run:` key. The startup
    /// sweep locates the run record by key, and reconstructing that key by
    /// re-formatting `started_at` would make crash recovery depend on a
    /// `DateTime` round-tripping bit-exactly through postcard — the failure
    /// class `src/memory/oplog.rs` already documents. Storing the key material
    /// removes the class.
    pub run_started_nanos: i64,
    pub heartbeat_at: DateTime<Utc>,
    /// The OS pid that took the lease.
    ///
    /// **Diagnostic, not a correctness mechanism.** RocksDB takes an exclusive
    /// OS lock on the DB directory, so only one process can hold the shared DB
    /// open. If this process just opened it, no other process holds it, and
    /// every `active:` key present at startup is therefore provably stale. The
    /// field exists so the record reads truthfully and so a future
    /// multi-process deployment has what it needs. Do not mistake this lease
    /// for a distributed lock.
    pub pid: u32,
}

/// Per-source state a run writes. Deliberately separate from
/// [`SourceDefinition`] so a run never rewrites the record the user authored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceRuntime {
    pub last_run_id: Option<uuid::Uuid>,
    pub last_run_started_at: Option<DateTime<Utc>>,
    pub last_run_finished_at: Option<DateTime<Utc>>,
    pub last_run_status: Option<RunStatus>,
    pub last_success_at: Option<DateTime<Utc>>,
    pub consecutive_failures: u16,
    /// Monotonic count of completed runs; drives `rehash_every_n_runs`.
    pub run_count: u64,
    pub memories_written_total: u64,
    /// Item totals as of the last completed run. Kept here rather than
    /// computed per request so the dashboard listing stays O(sources): the run
    /// already walks every cursor once to detect disappearances, and these
    /// come out of that same walk.
    pub items_tracked: u64,
    pub items_failed: u64,
    pub items_quarantined: u64,
}

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

/// `user_id` is `validate_user_id`-checked, which rejects `:`, so these
/// prefixes cannot be forged by a user id — the same guarantee `CF_OPLOG`
/// relies on. `source_id` is a server-minted UUID.
fn def_key(user_id: &str, source_id: &SourceId) -> String {
    format!("def:{}:{}", user_id, source_id.0)
}

fn def_prefix(user_id: &str) -> String {
    format!("def:{}:", user_id)
}

/// Zero-padded so lexicographic order is chronological order. The prospective
/// store had to migrate to this convention after `"9" > "10"` broke its due
/// index; there is no reason to repeat it.
fn run_key(user_id: &str, source_id: &SourceId, started_nanos: i64) -> String {
    format!("run:{}:{}:{:020}", user_id, source_id.0, started_nanos)
}

fn run_prefix(user_id: &str, source_id: &SourceId) -> String {
    format!("run:{}:{}:", user_id, source_id.0)
}

fn lease_key(user_id: &str, source_id: &SourceId) -> String {
    format!("active:{}:{}", user_id, source_id.0)
}

fn runtime_key(user_id: &str, source_id: &SourceId) -> String {
    format!("rt:{}:{}", user_id, source_id.0)
}

/// Scoped by `source_id` only. A `source_id` is a server-minted UUID and a
/// definition binds it to exactly one `user_id`; putting the user in the key
/// would make a user rename impossible and buys nothing.
fn cursor_key(source_id: &SourceId, item_hash: &str) -> String {
    format!("cur:{}:{}", source_id.0, item_hash)
}

fn cursor_prefix(source_id: &SourceId) -> String {
    format!("cur:{}:", source_id.0)
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// Reads and writes the registry. Holds no state of its own beyond the DB
/// handle — by construction, not by discipline.
pub struct SourceStore {
    db: Arc<DB>,
}

impl SourceStore {
    /// Column families this store needs. Call when opening the shared DB.
    pub fn cf_descriptors() -> Vec<ColumnFamilyDescriptor> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        vec![
            ColumnFamilyDescriptor::new(CF_SOURCES, opts.clone()),
            ColumnFamilyDescriptor::new(CF_SOURCE_CURSOR, opts),
        ]
    }

    /// Open the store and sweep any lease left behind by a process that died
    /// mid-run.
    pub fn new(db: Arc<DB>) -> Result<Self> {
        let store = Self { db };
        let swept = store.sweep_stale_leases()?;
        if swept > 0 {
            tracing::warn!(
                runs = swept,
                "Marked source runs Aborted: the process exited while they were in flight"
            );
        }
        Ok(store)
    }

    fn sources_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_SOURCES)
            .expect("sources CF must exist in shared DB")
    }

    fn cursor_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(CF_SOURCE_CURSOR)
            .expect("source_cursor CF must exist in shared DB")
    }

    // -- definitions --------------------------------------------------------

    pub fn put_source(&self, def: &SourceDefinition) -> Result<()> {
        let bytes = crate::serialization::encode(def).context("encode source definition")?;
        self.db
            .put_cf(self.sources_cf(), def_key(&def.user_id, &def.id), bytes)
            .context("write source definition")
    }

    pub fn get_source(&self, user_id: &str, source_id: &SourceId) -> Result<Option<SourceDefinition>> {
        let raw = self
            .db
            .get_cf(self.sources_cf(), def_key(user_id, source_id))
            .context("read source definition")?;
        match raw {
            None => Ok(None),
            Some(bytes) => Ok(Some(
                crate::serialization::decode(&bytes).context("decode source definition")?,
            )),
        }
    }

    /// Every source for a user, oldest first. O(sources) — cursors live in
    /// their own CF and are not walked here.
    pub fn list_sources(&self, user_id: &str) -> Result<Vec<SourceDefinition>> {
        let prefix = def_prefix(user_id);
        let mut out = Vec::new();
        for item in self
            .db
            .prefix_iterator_cf(self.sources_cf(), prefix.as_bytes())
        {
            let (key, value) = item.context("iterate source definitions")?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            out.push(crate::serialization::decode(&value).context("decode source definition")?);
        }
        out.sort_by(|a: &SourceDefinition, b: &SourceDefinition| {
            a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id))
        });
        Ok(out)
    }

    /// Delete a definition and everything derived from it: runs, lease,
    /// runtime, and every cursor. Returns the number of cursors removed.
    pub fn delete_source(&self, user_id: &str, source_id: &SourceId) -> Result<usize> {
        let mut batch = WriteBatch::default();
        batch.delete_cf(self.sources_cf(), def_key(user_id, source_id));
        batch.delete_cf(self.sources_cf(), lease_key(user_id, source_id));
        batch.delete_cf(self.sources_cf(), runtime_key(user_id, source_id));

        let rp = run_prefix(user_id, source_id);
        for item in self.db.prefix_iterator_cf(self.sources_cf(), rp.as_bytes()) {
            let (key, _) = item.context("iterate runs for delete")?;
            if !key.starts_with(rp.as_bytes()) {
                break;
            }
            batch.delete_cf(self.sources_cf(), &*key);
        }

        let cp = cursor_prefix(source_id);
        let mut cursors = 0usize;
        for item in self.db.prefix_iterator_cf(self.cursor_cf(), cp.as_bytes()) {
            let (key, _) = item.context("iterate cursors for delete")?;
            if !key.starts_with(cp.as_bytes()) {
                break;
            }
            batch.delete_cf(self.cursor_cf(), &*key);
            cursors += 1;
        }

        self.db.write(batch).context("delete source")?;
        Ok(cursors)
    }

    // -- runtime ------------------------------------------------------------

    pub fn get_runtime(&self, user_id: &str, source_id: &SourceId) -> Result<SourceRuntime> {
        let raw = self
            .db
            .get_cf(self.sources_cf(), runtime_key(user_id, source_id))
            .context("read source runtime")?;
        match raw {
            None => Ok(SourceRuntime::default()),
            Some(bytes) => {
                crate::serialization::decode(&bytes).context("decode source runtime")
            }
        }
    }

    // -- runs ---------------------------------------------------------------

    /// Record the intent before performing the side effect: the `Running` run
    /// record and the lease go down in ONE batch, before the first directory
    /// read. A crash after this point is detectable as "started, never
    /// finished" instead of invisible.
    pub fn begin_run(&self, run: &SourceRun, lease: &RunLease) -> Result<()> {
        let mut batch = WriteBatch::default();
        batch.put_cf(
            self.sources_cf(),
            run_key(&run.user_id, &run.source_id, run.started_nanos),
            crate::serialization::encode(run).context("encode run")?,
        );
        batch.put_cf(
            self.sources_cf(),
            lease_key(&run.user_id, &run.source_id),
            crate::serialization::encode(lease).context("encode lease")?,
        );
        self.db.write(batch).context("begin run")
    }

    /// Rewrite the run record and refresh the lease heartbeat, so a dashboard
    /// shows live progress and a hung run is visible as a stale heartbeat.
    pub fn heartbeat(&self, run: &SourceRun, lease: &mut RunLease) -> Result<()> {
        lease.heartbeat_at = Utc::now();
        let mut batch = WriteBatch::default();
        batch.put_cf(
            self.sources_cf(),
            run_key(&run.user_id, &run.source_id, run.started_nanos),
            crate::serialization::encode(run).context("encode run")?,
        );
        batch.put_cf(
            self.sources_cf(),
            lease_key(&run.user_id, &run.source_id),
            crate::serialization::encode(lease).context("encode lease")?,
        );
        self.db.write(batch).context("heartbeat run")
    }

    /// The cursor advance. Written in ONE batch with the run counters, and
    /// **only after** every part of the item has been written to the store.
    ///
    /// A crash between the memory write and this call loses the cursor
    /// advance, not the memory: the next run re-reads the file, derives
    /// identical content, and the sink absorbs the duplicate. That is Kafka
    /// Connect's "records written, offsets not yet committed" window, made
    /// harmless by an idempotent sink. The reverse order — cursor first —
    /// would lose data instead, which is why it is never done.
    pub fn commit_item(
        &self,
        source_id: &SourceId,
        item_hash: &str,
        cursor: &ItemCursor,
        run: &SourceRun,
    ) -> Result<()> {
        let mut batch = WriteBatch::default();
        batch.put_cf(
            self.cursor_cf(),
            cursor_key(source_id, item_hash),
            crate::serialization::encode(cursor).context("encode cursor")?,
        );
        batch.put_cf(
            self.sources_cf(),
            run_key(&run.user_id, &run.source_id, run.started_nanos),
            crate::serialization::encode(run).context("encode run")?,
        );
        self.db.write(batch).context("commit item cursor")
    }

    /// Final run record, runtime counters, and lease release — one batch —
    /// followed by history pruning.
    pub fn finish_run(&self, run: &SourceRun, runtime: &SourceRuntime) -> Result<()> {
        let mut batch = WriteBatch::default();
        batch.put_cf(
            self.sources_cf(),
            run_key(&run.user_id, &run.source_id, run.started_nanos),
            crate::serialization::encode(run).context("encode run")?,
        );
        batch.put_cf(
            self.sources_cf(),
            runtime_key(&run.user_id, &run.source_id),
            crate::serialization::encode(runtime).context("encode runtime")?,
        );
        batch.delete_cf(self.sources_cf(), lease_key(&run.user_id, &run.source_id));
        self.db.write(batch).context("finish run")?;
        self.prune_runs(&run.user_id, &run.source_id)?;
        Ok(())
    }

    /// Newest first. `total` is a true count, not the page length.
    pub fn list_runs(
        &self,
        user_id: &str,
        source_id: &SourceId,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<SourceRun>, usize)> {
        let mut all = self.all_runs(user_id, source_id)?;
        all.reverse();
        let total = all.len();
        let page = all.into_iter().skip(offset).take(limit).collect();
        Ok((page, total))
    }

    /// Oldest first, which is key order.
    fn all_runs(&self, user_id: &str, source_id: &SourceId) -> Result<Vec<SourceRun>> {
        let prefix = run_prefix(user_id, source_id);
        let mut out = Vec::new();
        for item in self
            .db
            .prefix_iterator_cf(self.sources_cf(), prefix.as_bytes())
        {
            let (key, value) = item.context("iterate runs")?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            out.push(crate::serialization::decode(&value).context("decode run")?);
        }
        Ok(out)
    }

    fn prune_runs(&self, user_id: &str, source_id: &SourceId) -> Result<()> {
        let prefix = run_prefix(user_id, source_id);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for item in self
            .db
            .prefix_iterator_cf(self.sources_cf(), prefix.as_bytes())
        {
            let (key, _) = item.context("iterate runs for prune")?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            keys.push(key.to_vec());
        }
        if keys.len() <= RUN_HISTORY_KEEP {
            return Ok(());
        }
        let drop_count = keys.len() - RUN_HISTORY_KEEP;
        let mut batch = WriteBatch::default();
        for key in keys.into_iter().take(drop_count) {
            batch.delete_cf(self.sources_cf(), key);
        }
        self.db.write(batch).context("prune run history")
    }

    // -- cursors ------------------------------------------------------------

    pub fn get_cursor(&self, source_id: &SourceId, item_hash: &str) -> Result<Option<ItemCursor>> {
        let raw = self
            .db
            .get_cf(self.cursor_cf(), cursor_key(source_id, item_hash))
            .context("read item cursor")?;
        match raw {
            None => Ok(None),
            Some(bytes) => Ok(Some(
                crate::serialization::decode(&bytes).context("decode item cursor")?,
            )),
        }
    }

    /// A `last_seen_at` refresh on an item the run enumerated but did not
    /// re-read. Not a cursor advance: losing this write cannot lose data, it
    /// can only make a present file look absent for one run.
    pub fn touch_cursor(
        &self,
        source_id: &SourceId,
        item_hash: &str,
        cursor: &ItemCursor,
    ) -> Result<()> {
        let bytes = crate::serialization::encode(cursor).context("encode item cursor")?;
        self.db
            .put_cf(self.cursor_cf(), cursor_key(source_id, item_hash), bytes)
            .context("touch item cursor")
    }

    /// Every cursor for a source. Walked once per run for disappearance
    /// detection and item totals; the API's item list reads it too.
    pub fn list_cursors(&self, source_id: &SourceId) -> Result<Vec<ItemCursor>> {
        let prefix = cursor_prefix(source_id);
        let mut out = Vec::new();
        for item in self
            .db
            .prefix_iterator_cf(self.cursor_cf(), prefix.as_bytes())
        {
            let (key, value) = item.context("iterate item cursors")?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            out.push(crate::serialization::decode(&value).context("decode item cursor")?);
        }
        out.sort_by(|a: &ItemCursor, b: &ItemCursor| a.path.cmp(&b.path));
        Ok(out)
    }

    // -- crash recovery -----------------------------------------------------

    /// Mark every run whose lease survived a process exit as `Aborted`.
    ///
    /// Called from `new()`. Every live `active:` key at this point is provably
    /// stale — see [`RunLease::pid`] — so this needs no timeout heuristic.
    fn sweep_stale_leases(&self) -> Result<usize> {
        let mut swept = 0usize;
        let mut batch = WriteBatch::default();

        for item in self.db.prefix_iterator_cf(self.sources_cf(), b"active:") {
            let (key, value) = item.context("iterate run leases")?;
            if !key.starts_with(b"active:") {
                break;
            }
            let lease: RunLease = match crate::serialization::decode(&value) {
                Ok(l) => l,
                Err(e) => {
                    // An undecodable lease still has to be cleared, or it
                    // blocks the source forever with no way to see why.
                    tracing::error!(error = %e, "Undecodable run lease dropped during startup sweep");
                    batch.delete_cf(self.sources_cf(), &*key);
                    swept += 1;
                    continue;
                }
            };

            // key = active:{user_id}:{source_id}
            let key_str = String::from_utf8_lossy(&key).to_string();
            let rest = &key_str["active:".len()..];
            let Some(split) = rest.rfind(':') else {
                tracing::error!(key = %key_str, "Malformed run lease key dropped during startup sweep");
                batch.delete_cf(self.sources_cf(), &*key);
                swept += 1;
                continue;
            };
            let user_id = &rest[..split];
            let Ok(uuid) = uuid::Uuid::parse_str(&rest[split + 1..]) else {
                tracing::error!(key = %key_str, "Run lease key holds no source id; dropped");
                batch.delete_cf(self.sources_cf(), &*key);
                swept += 1;
                continue;
            };
            let source_id = SourceId(uuid);

            let rkey = run_key(user_id, &source_id, lease.run_started_nanos);
            if let Some(raw) = self
                .db
                .get_cf(self.sources_cf(), &rkey)
                .context("read run for lease sweep")?
            {
                match crate::serialization::decode::<SourceRun>(&raw) {
                    Ok(mut run) => {
                        run.status = RunStatus::Aborted;
                        run.finished_at = Some(Utc::now());
                        run.error = Some(format!(
                            "process exited during run (lease taken {}, last heartbeat {})",
                            lease.started_at.to_rfc3339(),
                            lease.heartbeat_at.to_rfc3339()
                        ));
                        batch.put_cf(
                            self.sources_cf(),
                            &rkey,
                            crate::serialization::encode(&run).context("encode aborted run")?,
                        );
                    }
                    Err(e) => {
                        tracing::error!(error = %e, run = %rkey, "Undecodable run record left as-is during sweep");
                    }
                }
            }

            // An abort counts against the source exactly as a failure does.
            let mut runtime = self.get_runtime(user_id, &source_id)?;
            runtime.consecutive_failures = runtime.consecutive_failures.saturating_add(1);
            runtime.last_run_id = Some(lease.run_id);
            runtime.last_run_started_at = Some(lease.started_at);
            runtime.last_run_finished_at = Some(Utc::now());
            runtime.last_run_status = Some(RunStatus::Aborted);
            batch.put_cf(
                self.sources_cf(),
                runtime_key(user_id, &source_id),
                crate::serialization::encode(&runtime).context("encode runtime")?,
            );

            batch.delete_cf(self.sources_cf(), &*key);
            swept += 1;
        }

        if swept > 0 {
            self.db.write(batch).context("sweep stale run leases")?;
        }
        Ok(swept)
    }

    /// True while a lease is held. Read from RocksDB, like everything else.
    pub fn is_running(&self, user_id: &str, source_id: &SourceId) -> Result<bool> {
        Ok(self
            .db
            .get_cf(self.sources_cf(), lease_key(user_id, source_id))
            .context("read run lease")?
            .is_some())
    }

    /// Remove every registry record belonging to a user. Called when a user is
    /// deleted, so a purged profile does not leave a folder connector behind.
    pub fn purge_user(&self, user_id: &str) -> Result<usize> {
        let sources = self.list_sources(user_id)?;
        let mut removed = 0usize;
        for def in &sources {
            self.delete_source(user_id, &def.id)?;
            removed += 1;
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SourceKind::ALL` claims declaration order, which is postcard
    /// discriminant order, which is what is on disk. A kind appended to the
    /// enum but forgotten in `ALL` cannot be parsed from the wire; a kind
    /// inserted rather than appended re-points every stored definition.
    #[test]
    fn source_kind_all_is_in_wire_discriminant_order() {
        for (index, kind) in SourceKind::ALL.iter().enumerate() {
            let encoded = crate::serialization::encode_raw(kind).expect("encode kind");
            assert_eq!(
                encoded,
                vec![index as u8],
                "SourceKind::ALL[{}] ({}) does not encode as that discriminant",
                index,
                kind.as_str()
            );
            assert_eq!(SourceKind::parse(kind.as_str()), Some(*kind));
        }
    }

    /// Every stored type in this module goes through the crate's postcard
    /// helpers. Postcard has no `deserialize_any`, so serde's internally
    /// tagged enum representation (`#[serde(tag = "...")]`) silently fails to
    /// decode — the shape the design sketch reached for. This pins the shapes
    /// that are actually used.
    #[test]
    fn registry_records_round_trip_through_postcard() {
        let id = SourceId(uuid::Uuid::new_v4());
        let def = SourceDefinition {
            id: id.clone(),
            user_id: "tester".to_string(),
            name: "Field notes".to_string(),
            kind: SourceKind::WatchedFolder,
            config: SourceConfig::WatchedFolder(WatchedFolderConfig::with_root(
                "/corpus".to_string(),
            )),
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            schema_version: 1,
        };
        let bytes = crate::serialization::encode(&def).expect("encode definition");
        let back: SourceDefinition =
            crate::serialization::decode(&bytes).expect("decode definition");
        assert_eq!(back.name, def.name);
        assert_eq!(back.config.kind(), SourceKind::WatchedFolder);
        assert_eq!(back.config.as_watched_folder().max_depth, 8);

        for state in [
            ItemState::Ingested,
            ItemState::Deduped,
            ItemState::Skipped {
                reason: "binary content".to_string(),
            },
            ItemState::Failed {
                reason: "locked".to_string(),
            },
            ItemState::Quarantined {
                reason: "locked".to_string(),
            },
        ] {
            let bytes = crate::serialization::encode(&state).expect("encode state");
            let back: ItemState = crate::serialization::decode(&bytes).expect("decode state");
            assert_eq!(back, state, "ItemState must survive a postcard round trip");
        }
    }
}
