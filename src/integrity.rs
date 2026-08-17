//! Read-only integrity scrub for persisted records.
//!
//! # Why this exists
//!
//! In July 2026 two positional-format changes made 982 memories and 6,821 graph
//! nodes unreadable. They stayed broken for over a month and were found by
//! accident while chasing an unrelated 500. Nothing in the system noticed,
//! because nothing in the system was looking.
//!
//! # Why "does it decode?" is the wrong question
//!
//! A corrupted record frequently does *not* fail to decode. `Memory` has an
//! eighteen-branch fallback chain ending in [`try_raw_memory_parse`], which
//! takes the first sixteen bytes of the value as a UUID, the rest as content,
//! and fabricates every other field. Twenty-one bytes of garbage decode into a
//! `Memory` through it. Worse, [`crate::serialization::unwrap_sho`] returns
//! `None` on a CRC32 mismatch, and `deserialize_memory` treats `None` as "no
//! envelope, try the legacy chain" — so a bit-rotted current-format record is
//! handed, magic bytes and all, to the very chain that will invent a plausible
//! `Memory` from it. The read succeeds, no warning is logged past the first
//! one, and a fabricated memory is served as fact.
//!
//! A scrub that asks "does every record decode?" reports that population as
//! perfectly healthy. This module therefore classifies into four states, not
//! two:
//!
//! | state | meaning |
//! |---|---|
//! | [`RecordClass::Clean`] | decoded under the current schema, envelope intact |
//! | [`RecordClass::Legacy`] | decoded only via an older wire generation — readable, aging, should be rewritten |
//! | [`RecordClass::Undecodable`] | no generation could read it — broken, but *honestly* broken: it errors rather than lying |
//! | [`RecordClass::Implausible`] | decoded successfully into values the writer could not have produced |
//!
//! The fourth class is the point of this module.
//!
//! # Read-only means read-only
//!
//! This module never calls [`crate::memory::storage::MemoryStorage::get`],
//! which lazily *rewrites* any record it decodes via a legacy path — on a
//! pseudo-decode that would overwrite the original bytes with the fabrication,
//! destroying the evidence. It iterates raw column families with
//! `fill_cache(false)` and does its own envelope parsing.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use chrono::{DateTime, TimeZone, Utc};
use rocksdb::{ColumnFamily, ReadOptions, DB};
use serde::{Deserialize, Serialize};

use crate::graph_memory::EntityNode;
use crate::memory::types::Memory;

// ---------------------------------------------------------------------------
// Plausibility bounds
// ---------------------------------------------------------------------------

/// Earliest timestamp any record in this system can carry.
///
/// shodh-memory's RocksDB store did not exist before this date, so a persisted
/// `created_at` below it cannot have been written by any version of the writer.
/// Deliberately far below the real first-write date: the bound exists to catch
/// arbitrary bytes reinterpreted as a timestamp, not to police data age. A
/// signal that fires on healthy records is worse than no signal.
const TIMESTAMP_FLOOR_UNIX: i64 = 1_577_836_800; // 2020-01-01T00:00:00Z

/// Slack allowed above wall clock, absorbing clock skew between the writing
/// host and the scrubbing host.
const TIMESTAMP_CEILING_SLACK: Duration = Duration::from_secs(86_400);

/// Slack allowed on `last_accessed < created_at` before it counts as inverted.
const TIMESTAMP_INVERSION_SLACK_SECS: i64 = 60;

/// Tolerance on the L2 norm of a stored embedding.
///
/// The write path L2-normalises every embedding it produces
/// (`embeddings::minilm::normalize`), whose own round-trip test asserts
/// `(norm - 1.0).abs() < 1e-5`. The tolerance here is four orders of magnitude
/// looser so that f32 accumulation order and Matryoshka truncation cannot
/// produce a false positive.
const EMBEDDING_NORM_TOLERANCE: f32 = 0.05;

/// Loose absolute bounds on embedding dimension.
///
/// No `EMBEDDING_DIM` constant exists — dimension is a runtime property of the
/// configured embedder (384 for MiniLM-L6-v2, 768 for the nomic family,
/// truncatable by Matryoshka). `types.rs` documents the supported span as
/// 384–1536. These bounds are far outside that on both sides so that switching
/// embedder can never trip the alarm; they exist to catch a length varint read
/// out of the middle of some other field.
const EMBEDDING_DIM_MIN: usize = 16;
const EMBEDDING_DIM_MAX: usize = 16_384;

/// Maximum number of per-record findings retained in a report.
///
/// Findings are evidence for a human, not a data export. Beyond this the counts
/// carry the signal and `findings_truncated` says so out loud.
/// Retained findings per (record kind, classification) pair.
///
/// Deliberately a per-class quota rather than one shared budget: see
/// [`Sweep::push_finding`].
pub const FINDINGS_PER_CLASS: usize = 25;

/// Default wall-clock ceiling for one scrub, in milliseconds.
///
/// Below the server's `TimeoutLayer` (`request_timeout_secs`, default 60), so a
/// scrub that would outrun the request budget reports itself incomplete rather
/// than being killed and returning nothing at all.
///
/// Sized against measurement, not taste: the largest live profile sweeps in
/// ~14s cold. A 20s ceiling would leave that profile one growth spurt away
/// from permanently reporting `indeterminate` — honest, but useless. 45s keeps
/// roughly 3x headroom over the measured cost while still leaving 15s of margin
/// under the request timeout.
pub const DEFAULT_MAX_DURATION_MS: u64 = 45_000;

/// Readahead window for the sequential scans.
///
/// The memory default column family is shared with several prefixed subsystem
/// keyspaces; on the live claude-code store 97% of its 679,118 keys are not
/// memories. Scanning it is I/O bound on sequential block reads, which is
/// exactly what readahead is for.
const SCAN_READAHEAD_BYTES: usize = 4 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Classification of a single persisted record.
///
/// Ordered by severity: a record that satisfies several of these is reported
/// under the worst one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordClass {
    /// Decoded under the current schema with an intact envelope.
    Clean,
    /// Decoded only by falling back to an older wire generation. Readable and
    /// correct as far as we can tell, but one schema change away from the
    /// July failure mode. Should be rewritten in the current format.
    Legacy,
    /// Decoded, but into values the writer could not have produced. This is the
    /// dangerous class: the system serves these as fact.
    Implausible,
    /// No decode generation could read it. Broken, but it fails loudly.
    Undecodable,
    /// The stored CRC32 does not match the payload. The bytes on disk are not
    /// the bytes that were written.
    ChecksumMismatch,
}

/// Per-record-class tallies for one column family.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassCounts {
    /// Keys the iterator produced, including ones that are not records of this
    /// class (e.g. the prefixed subsystem keyspaces sharing the default CF).
    pub keys_seen: u64,
    /// Keys that were actually classified.
    pub scanned: u64,
    pub clean: u64,
    pub legacy: u64,
    pub implausible: u64,
    pub undecodable: u64,
    pub checksum_mismatch: u64,
    /// Records soft-deleted by the user (`metadata["forgotten"] == "true"`).
    /// Counted, never treated as a defect — they decoded fine.
    pub forgotten: u64,
    /// Iterator-level failures. A non-zero value means the sweep could not see
    /// every record, so it cannot certify health.
    pub read_errors: u64,
    /// Which decode generation each record needed, by path name.
    pub decode_paths: BTreeMap<String, u64>,
    /// Which plausibility check fired, by check name. A record failing several
    /// checks increments each of them, so these sum to at least `implausible`.
    pub checks_failed: BTreeMap<String, u64>,
    /// Observed embedding dimensions and their frequencies. Informational: a
    /// second peak here is embedder drift, which no threshold can judge for you.
    pub embedding_dims: BTreeMap<usize, u64>,
}

impl ClassCounts {
    fn record(&mut self, class: RecordClass, path: &str, checks: &[&'static str]) {
        self.scanned += 1;
        match class {
            RecordClass::Clean => self.clean += 1,
            RecordClass::Legacy => self.legacy += 1,
            RecordClass::Implausible => self.implausible += 1,
            RecordClass::Undecodable => self.undecodable += 1,
            RecordClass::ChecksumMismatch => self.checksum_mismatch += 1,
        }
        *self.decode_paths.entry(path.to_string()).or_insert(0) += 1;
        for check in checks {
            *self.checks_failed.entry((*check).to_string()).or_insert(0) += 1;
        }
    }

    /// Records that are broken or lying, as opposed to merely old.
    pub fn defects(&self) -> u64 {
        self.implausible + self.undecodable + self.checksum_mismatch
    }
}

/// One record's worth of evidence, retained up to [`MAX_FINDINGS`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// `"memory"` or `"graph_node"`.
    pub record_class: String,
    /// The RocksDB key, hex-encoded. Always present — the decoded id is not
    /// trustworthy on a pseudo-decode, the key is.
    pub key: String,
    /// The id the record decoded to, when it decoded at all. A value differing
    /// from `key` is itself the finding.
    pub decoded_id: Option<String>,
    pub classification: RecordClass,
    /// Which decode generation produced this record.
    pub decode_path: String,
    /// Which plausibility checks failed, or the decode error for undecodables.
    pub detail: String,
    /// `created_at` as decoded, for cohort analysis. A defect population that
    /// clusters in a date range is a schema change, not bit rot.
    pub created_at: Option<DateTime<Utc>>,
}

/// The scrub's judgement. Not a number for someone to interpret later — the
/// point of this work is that a number nobody reads is how the July breakage
/// survived a month.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Every record decoded cleanly under the current schema.
    Healthy,
    /// Everything is readable, but some records need an older wire generation.
    /// Not an outage; a countdown. The next positional schema change breaks
    /// exactly this population.
    Aging,
    /// Records that fail to decode exist, but under the alarm rate. They error
    /// rather than lie, so they are contained.
    Degraded,
    /// Either records decode into impossible values (the system is serving
    /// fabrications), or the on-disk bytes fail their checksum, or the
    /// undecodable rate is above the alarm threshold.
    Unhealthy,
    /// The sweep could not see every record — it was cut short, or the
    /// iterator errored. No health claim is possible and none is made.
    Indeterminate,
}

/// Rate of undecodable records above which the verdict escalates from
/// `Degraded` to `Unhealthy`.
///
/// 0.1% of a 19k-record store is nineteen records. Below that, a handful of
/// records written by a long-dead format is a maintenance item. Above it, a
/// systematic breakage is in progress.
pub const UNDECODABLE_ALARM_RATE: f64 = 0.001;

/// Bounds on how much work one scrub may do.
#[derive(Debug, Clone)]
pub struct ScrubBudget {
    /// Stop after this many records. `None` = full sweep, which is the default
    /// and the recommended setting; see the module docs on sampling.
    pub max_records: Option<u64>,
    /// Stop after this much wall clock.
    pub max_duration: Option<Duration>,
}

impl Default for ScrubBudget {
    fn default() -> Self {
        Self {
            max_records: None,
            max_duration: Some(Duration::from_millis(DEFAULT_MAX_DURATION_MS)),
        }
    }
}

/// The full result of one scrub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityScrubReport {
    pub user_id: String,
    pub started_at: DateTime<Utc>,
    pub duration_ms: u64,

    /// `true` only if every record in every scanned column family was
    /// classified. A `false` here invalidates every "0 defects" claim below it,
    /// and [`Verdict::Indeterminate`] enforces that in the verdict.
    pub complete: bool,
    /// Why the sweep stopped early, when it did.
    pub stop_reason: Option<String>,
    /// Column families the sweep never reached at all (e.g. the graph database
    /// could not be opened). Named, not silently omitted.
    pub skipped: Vec<String>,

    pub memories: ClassCounts,
    pub graph_nodes: ClassCounts,

    /// Every plausibility check this build applies, by name. A check absent
    /// from this list was not run, whatever the counts say.
    pub checks_applied: Vec<String>,

    pub findings: Vec<Finding>,
    pub findings_truncated: bool,

    pub verdict: Verdict,
    /// The numeric rule that produced `verdict`, spelled out in the response so
    /// the judgement travels with the numbers.
    pub verdict_rule: String,
    /// Flat boolean for alerting hooks, matching `IndexIntegrityReport`'s
    /// convention. False for anything that is not [`Verdict::Healthy`] —
    /// including [`Verdict::Indeterminate`], because "we could not check" must
    /// never read as "fine".
    pub is_healthy: bool,
}

/// Every check this build applies, in the order they are evaluated.
pub const MEMORY_CHECKS: &[&str] = &[
    "id_key_mismatch",
    "timestamp_out_of_bounds",
    "timestamp_inverted",
    "unit_range",
    "reward_range",
    "content_empty",
    "embedding_non_finite",
    "embedding_dimension",
    "embedding_not_unit_norm",
];

/// Every check applied to graph nodes.
pub const GRAPH_CHECKS: &[&str] = &[
    "id_key_mismatch",
    "timestamp_out_of_bounds",
    "timestamp_inverted",
    "unit_range",
    "name_empty",
    "embedding_non_finite",
    "embedding_dimension",
    "embedding_not_unit_norm",
];

// ---------------------------------------------------------------------------
// Budget tracking
// ---------------------------------------------------------------------------

/// Budget and evidence accumulator, threaded through every column-family
/// sweep so that a truncated scan is truncated *everywhere* and says so once.
pub struct Sweep {
    started: Instant,
    budget: ScrubBudget,
    records: u64,
    keys: u64,
    stop_reason: Option<String>,
    findings: Vec<Finding>,
    finding_quota: BTreeMap<(String, RecordClass), usize>,
    findings_truncated: bool,
}

impl Sweep {
    /// Start a sweep under `budget`.
    pub fn new(budget: ScrubBudget) -> Self {
        Self {
            started: Instant::now(),
            budget,
            records: 0,
            keys: 0,
            stop_reason: None,
            findings: Vec::new(),
            finding_quota: BTreeMap::new(),
            findings_truncated: false,
        }
    }

    /// Poll the wall-clock deadline. Called on EVERY key, not only on record
    /// keys.
    ///
    /// The memory default column family is shared with subsystem keyspaces
    /// that outnumber memories 35 to 1, and on some profiles they arrive in
    /// long uninterrupted runs. A deadline checked only when a record is
    /// classified can be overrun by seconds while the iterator walks a stretch
    /// of `facts:` keys — and on a pathological store the request would then
    /// die at the server timeout returning nothing at all, which is precisely
    /// the outcome the budget exists to prevent. ~20ns per key buys a deadline
    /// that actually binds.
    ///
    /// Returns `false` once the sweep must stop.
    fn poll_deadline(&mut self) -> bool {
        if self.stop_reason.is_some() {
            return false;
        }
        if let Some(max) = self.budget.max_duration {
            if self.started.elapsed() >= max {
                self.stop_reason = Some(format!(
                    "time budget of {}ms exhausted after {} keys ({} records \
                     classified); the counts below describe only what was scanned",
                    max.as_millis(),
                    self.keys,
                    self.records
                ));
                return false;
            }
        }
        self.keys += 1;
        true
    }

    /// Charge one classified record against the record budget.
    ///
    /// Returns `false` when the budget is exhausted, recording why.
    fn tick(&mut self) -> bool {
        if self.stop_reason.is_some() {
            return false;
        }
        if let Some(max) = self.budget.max_records {
            if self.records >= max {
                self.stop_reason = Some(format!(
                    "record budget exhausted after {} records; the counts below \
                     describe only what was scanned",
                    self.records
                ));
                return false;
            }
        }
        self.records += 1;
        true
    }

    /// Evidence collected so far, capped at [`FINDINGS_PER_CLASS`] per class.
    pub fn findings(&self) -> &[Finding] {
        &self.findings
    }

    /// Whether evidence was dropped because the cap was reached.
    pub fn findings_truncated(&self) -> bool {
        self.findings_truncated
    }

    /// Why the sweep stopped early, if it did. `None` means it ran to the end.
    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }

    fn push_finding(&mut self, f: Finding) {
        // Quota per (record kind, classification) rather than one global cap.
        // Measured on the live claude-code store: 301 undecodable graph nodes
        // and 19 undecodable memories filled a flat 100-slot budget before a
        // single implausible record reached it, hiding the exact class this
        // module exists to surface. A shared budget lets the loudest failure
        // mode censor the quietest one.
        let bucket = (f.record_class.clone(), f.classification);
        let used = self.finding_quota.entry(bucket).or_insert(0);
        if *used < FINDINGS_PER_CLASS {
            *used += 1;
            self.findings.push(f);
        } else {
            self.findings_truncated = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Envelope-level classification
// ---------------------------------------------------------------------------

/// Outcome of decoding one value, before plausibility is considered.
enum DecodeOutcome<T> {
    /// Decoded under the current schema with an intact envelope.
    Current(T, &'static str),
    /// Decoded only via an older wire generation.
    Legacy(T, String),
    /// Nothing could read it.
    Failed(String, &'static str),
    /// The envelope's CRC32 does not match its payload.
    ChecksumMismatch,
}

/// Verify a SHO envelope's CRC32 without going through
/// [`crate::serialization::unwrap_sho`].
///
/// `unwrap_sho` collapses "no envelope" and "corrupt envelope" into `None`, and
/// its caller reacts to `None` by feeding the whole buffer — magic bytes
/// included — to the legacy fallback chain, which will happily invent a
/// `Memory` from it. The scrub needs those two cases kept apart, so it parses
/// the envelope itself.
fn split_sho(data: &[u8]) -> Option<Result<(u8, &[u8]), ()>> {
    if data.len() < 8 || &data[0..3] != crate::memory::storage::STORAGE_MAGIC {
        return None;
    }
    let version = data[3];
    let payload_end = data.len() - 4;
    let stored = u32::from_le_bytes([
        data[payload_end],
        data[payload_end + 1],
        data[payload_end + 2],
        data[payload_end + 3],
    ]);
    if stored != crate::memory::storage::crc32_simple(&data[..payload_end]) {
        return Some(Err(()));
    }
    Some(Ok((version, &data[4..payload_end])))
}

/// Classify one memory value at the wire level.
fn decode_memory_value(value: &[u8]) -> DecodeOutcome<Memory> {
    use crate::serialization::{SHO_VERSION_BINCODE2, SHO_VERSION_POSTCARD};

    match split_sho(value) {
        Some(Err(())) => DecodeOutcome::ChecksumMismatch,
        Some(Ok((SHO_VERSION_POSTCARD, payload))) => {
            match crate::serialization::decode_raw_compat::<Memory>(
                payload,
                crate::memory::storage::MEMORY_DEFAULT_SUFFIX,
            ) {
                Ok((m, false)) => DecodeOutcome::Current(m, "sho_v2_postcard"),
                // Decoded only after supplying defaults for trailing fields the
                // record predates. Readable, but written against an older
                // schema — exactly the population a positional change breaks.
                Ok((m, true)) => DecodeOutcome::Legacy(m, "sho_v2_postcard_defaulted".to_string()),
                Err(e) => DecodeOutcome::Failed(e.to_string(), "sho_v2_postcard"),
            }
        }
        Some(Ok((SHO_VERSION_BINCODE2, payload))) => {
            match bincode::serde::decode_from_slice::<Memory, _>(
                payload,
                crate::bincode_safe_config(),
            ) {
                Ok((m, _)) => DecodeOutcome::Legacy(m, "sho_v1_bincode2".to_string()),
                Err(e) => DecodeOutcome::Failed(e.to_string(), "sho_v1_bincode2"),
            }
        }
        Some(Ok((other, _))) => DecodeOutcome::Failed(
            format!("unknown SHO envelope version {other}"),
            "sho_unknown_version",
        ),
        None => {
            // No envelope: a record written before the SHO cutover. Try the
            // current bincode shape first so it is distinguishable from the
            // deep fallback chain, whose later branches fabricate fields.
            match bincode::serde::decode_from_slice::<Memory, _>(
                value,
                crate::bincode_safe_config(),
            ) {
                Ok((m, _)) => DecodeOutcome::Legacy(m, "raw_bincode2".to_string()),
                Err(_) => {
                    // Delegate to the production fallback chain rather than
                    // duplicating its eighteen branches — a second copy would
                    // drift from the one that actually serves reads, and then
                    // the scrub would be measuring a decoder nobody uses.
                    match crate::memory::storage::deserialize_memory_for_migration(value) {
                        Ok(m) => DecodeOutcome::Legacy(m, "legacy_fallback_chain".to_string()),
                        Err(e) => DecodeOutcome::Failed(e.to_string(), "legacy_fallback_chain"),
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plausibility
// ---------------------------------------------------------------------------

fn timestamp_bounds() -> (DateTime<Utc>, DateTime<Utc>) {
    let floor = Utc
        .timestamp_opt(TIMESTAMP_FLOOR_UNIX, 0)
        .single()
        .unwrap_or(DateTime::<Utc>::MIN_UTC);
    let ceiling = Utc::now()
        + chrono::Duration::from_std(TIMESTAMP_CEILING_SLACK)
            .unwrap_or_else(|_| chrono::Duration::days(1));
    (floor, ceiling)
}

/// Check a float that the schema declares to lie in `[0, 1]`.
fn unit_ok(v: f32) -> bool {
    v.is_finite() && (0.0..=1.0).contains(&v)
}

/// Shared embedding checks. Returns the observed dimension, if any, so the
/// caller can build a histogram.
///
/// An all-zero vector is *not* flagged: `embeddings::minilm` writes one
/// deliberately when normalisation fails, so it is a legitimately-written
/// value. Flagging it would be a false positive on healthy data.
fn check_embedding(
    emb: Option<&Vec<f32>>,
    failed: &mut Vec<&'static str>,
    detail: &mut Vec<String>,
) -> Option<usize> {
    let v = emb?;
    let dim = v.len();

    if !(EMBEDDING_DIM_MIN..=EMBEDDING_DIM_MAX).contains(&dim) {
        failed.push("embedding_dimension");
        detail.push(format!(
            "embedding dimension {dim} outside [{EMBEDDING_DIM_MIN},{EMBEDDING_DIM_MAX}]"
        ));
        // A length this wrong means the length varint was read out of the wrong
        // place; the floats behind it are not worth further inspection.
        return Some(dim);
    }

    if v.iter().any(|x| !x.is_finite()) {
        failed.push("embedding_non_finite");
        detail.push("embedding contains NaN or Inf".to_string());
        return Some(dim);
    }

    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Zero vectors are the documented normalisation-failure fallback.
    if norm > f32::EPSILON && (norm - 1.0).abs() > EMBEDDING_NORM_TOLERANCE {
        failed.push("embedding_not_unit_norm");
        detail.push(format!("embedding L2 norm {norm:.4}, expected 1.0"));
    }
    Some(dim)
}

/// Plausibility verdict for a decoded `Memory`.
///
/// Returns the names of every check that failed and a human-readable detail
/// string. An empty result means the record is plausible.
fn check_memory(m: &Memory, key: &[u8]) -> (Vec<&'static str>, String, Option<usize>) {
    let mut failed: Vec<&'static str> = Vec::new();
    let mut detail: Vec<String> = Vec::new();

    // The writer puts the memory's own id in the key. Every fallback shape in
    // the legacy chain — including `try_raw_memory_parse`, which fabricates the
    // rest of the record — derives its id from the first sixteen bytes of the
    // VALUE. On a pseudo-decode of an enveloped record those bytes are "SHO" +
    // version + payload, which cannot equal the key. This check has no false
    // positive by construction: the writer never stores a record under a key
    // other than its own id.
    if m.id.0.as_bytes() != key {
        failed.push("id_key_mismatch");
        detail.push(format!(
            "decoded id {} does not match key {}",
            m.id.0,
            hex::encode(key)
        ));
    }

    let (floor, ceiling) = timestamp_bounds();
    let last_accessed = m.last_accessed();
    for (name, ts) in [
        ("created_at", m.created_at),
        ("last_accessed", last_accessed),
    ] {
        if ts < floor || ts > ceiling {
            failed.push("timestamp_out_of_bounds");
            detail.push(format!("{name} = {ts} outside [{floor}, {ceiling}]"));
        }
    }
    if last_accessed < m.created_at - chrono::Duration::seconds(TIMESTAMP_INVERSION_SLACK_SECS) {
        failed.push("timestamp_inverted");
        detail.push(format!(
            "last_accessed {last_accessed} precedes created_at {}",
            m.created_at
        ));
    }

    for (name, v) in [
        ("importance", m.importance()),
        ("activation", m.activation()),
        ("temporal_relevance", m.temporal_relevance()),
    ] {
        if !unit_ok(v) {
            failed.push("unit_range");
            detail.push(format!("{name} = {v}, declared range [0.0, 1.0]"));
        }
    }
    if let Some(c) = m.experience.confidence {
        if !unit_ok(c) {
            failed.push("unit_range");
            detail.push(format!("confidence = {c}, declared range [0.0, 1.0]"));
        }
    }
    if let Some(r) = m.experience.reward {
        if !r.is_finite() || !(-1.0..=1.0).contains(&r) {
            failed.push("reward_range");
            detail.push(format!("reward = {r}, declared range [-1.0, 1.0]"));
        }
    }

    // `content` is the one field of `Experience` with no serde default: the
    // writer requires it. An empty one means the string length was read from
    // the wrong offset.
    if m.experience.content.is_empty() {
        failed.push("content_empty");
        detail.push("experience.content is empty".to_string());
    }

    let dim = check_embedding(m.experience.embeddings.as_ref(), &mut failed, &mut detail);

    failed.sort_unstable();
    failed.dedup();
    (failed, detail.join("; "), dim)
}

/// Plausibility verdict for a decoded `EntityNode`.
fn check_entity(n: &EntityNode, key: &[u8]) -> (Vec<&'static str>, String, Option<usize>) {
    let mut failed: Vec<&'static str> = Vec::new();
    let mut detail: Vec<String> = Vec::new();

    if n.uuid.as_bytes() != key {
        failed.push("id_key_mismatch");
        detail.push(format!(
            "decoded uuid {} does not match key {}",
            n.uuid,
            hex::encode(key)
        ));
    }

    // NOT the check that catches the July `EntityLabel` renumbering — measured,
    // not assumed. That desync always dies inside postcard, because
    // `DateTime<Utc>` serialises as a string and every following length varint
    // lands misaligned; see
    // `renumbered_entity_label_surfaces_as_undecodable_and_trips_the_alarm`.
    // What this bound earns its place for is the case postcard cannot reject:
    // `try_decode_compat` falls back to raw bincode 2.x for untagged records,
    // and bincode tolerates trailing bytes, so a shifted read there can
    // complete and hand back a node with an impossible date.
    let (floor, ceiling) = timestamp_bounds();
    for (name, ts) in [
        ("created_at", n.created_at),
        ("last_seen_at", n.last_seen_at),
    ] {
        if ts < floor || ts > ceiling {
            failed.push("timestamp_out_of_bounds");
            detail.push(format!("{name} = {ts} outside [{floor}, {ceiling}]"));
        }
    }
    if n.last_seen_at < n.created_at - chrono::Duration::seconds(TIMESTAMP_INVERSION_SLACK_SECS) {
        failed.push("timestamp_inverted");
        detail.push(format!(
            "last_seen_at {} precedes created_at {}",
            n.last_seen_at, n.created_at
        ));
    }

    if !unit_ok(n.salience) {
        failed.push("unit_range");
        detail.push(format!(
            "salience = {}, declared range [0.0, 1.0]",
            n.salience
        ));
    }
    if let Some(s) = n.selectivity {
        if !s.is_finite() {
            failed.push("unit_range");
            detail.push(format!("selectivity = {s} is not finite"));
        }
    }

    if n.name.is_empty() {
        failed.push("name_empty");
        detail.push("entity name is empty".to_string());
    }

    let dim = check_embedding(n.name_embedding.as_ref(), &mut failed, &mut detail);

    failed.sort_unstable();
    failed.dedup();
    (failed, detail.join("; "), dim)
}

// ---------------------------------------------------------------------------
// Column-family sweeps
// ---------------------------------------------------------------------------

/// Sequential-scan read options.
///
/// `fill_cache(false)` because a scrub is cold-data maintenance: charging every
/// record it touches to the shared block cache would evict the working set of
/// every other tenant. `set_readahead_size` because that same flag defeats
/// RocksDB own prefetching heuristics, and a full scan without readahead
/// degenerates into one synchronous block read at a time.
fn scan_opts() -> ReadOptions {
    let mut o = ReadOptions::default();
    o.fill_cache(false);
    o.set_readahead_size(SCAN_READAHEAD_BYTES);
    o
}

/// Scrub every memory record in a memory database.
///
/// Memories live in the default column family under raw 16-byte UUID keys,
/// sharing that keyspace with prefixed subsystem keys (`facts:`, `vmapping:`,
/// `_watermark:`, …). Those are counted in `keys_seen` and skipped: they are
/// not memory records and this scrub makes no claim about them.
pub fn scrub_memories(db: &DB, sweep: &mut Sweep) -> ClassCounts {
    let mut counts = ClassCounts::default();
    // A raw iterator, so the value is only materialised for keys that are
    // actually memory records. Measured on the live claude-code store: reading
    // every value in the shared default column family took 23.2s, and 97% of
    // that work was fact, watermark and vmapping payloads this scrub makes no
    // claim about.
    let mut iter = db.raw_iterator_opt(scan_opts());
    iter.seek_to_first();

    loop {
        if let Err(e) = iter.status() {
            // An iterator error is not "no more records". Per the
            // structural-emptiness rule, an inconclusive read must never read
            // as absence -- count it, and let it force Indeterminate.
            counts.read_errors += 1;
            tracing::warn!(error = %e, "integrity scrub: memory iterator error");
            break;
        }
        let Some(key) = iter.key() else { break };
        counts.keys_seen += 1;
        if !sweep.poll_deadline() {
            break;
        }
        if key.len() == 16 {
            if !sweep.tick() {
                break;
            }
            let key = key.to_vec();
            match iter.value() {
                Some(value) => classify_memory(&key, value, &mut counts, sweep),
                None => counts.read_errors += 1,
            }
        }
        iter.next();
    }
    counts
}

fn classify_memory(key: &[u8], value: &[u8], counts: &mut ClassCounts, sweep: &mut Sweep) {
    let (class, path, decoded_id, detail, created_at, checks) = match decode_memory_value(value) {
        DecodeOutcome::ChecksumMismatch => (
            RecordClass::ChecksumMismatch,
            "sho_crc_mismatch".to_string(),
            None,
            "stored CRC32 does not match payload; the bytes on disk are not the \
             bytes that were written"
                .to_string(),
            None,
            Vec::new(),
        ),
        DecodeOutcome::Failed(err, path) => (
            RecordClass::Undecodable,
            path.to_string(),
            None,
            err,
            None,
            Vec::new(),
        ),
        DecodeOutcome::Current(m, path) => {
            let (failed, detail, dim) = check_memory(&m, key);
            if let Some(d) = dim {
                *counts.embedding_dims.entry(d).or_insert(0) += 1;
            }
            if m.is_forgotten() {
                counts.forgotten += 1;
            }
            let class = if failed.is_empty() {
                RecordClass::Clean
            } else {
                RecordClass::Implausible
            };
            (
                class,
                path.to_string(),
                Some(m.id.0.to_string()),
                detail,
                Some(m.created_at),
                failed,
            )
        }
        DecodeOutcome::Legacy(m, path) => {
            let (failed, detail, dim) = check_memory(&m, key);
            if let Some(d) = dim {
                *counts.embedding_dims.entry(d).or_insert(0) += 1;
            }
            if m.is_forgotten() {
                counts.forgotten += 1;
            }
            // Implausibility outranks age. A record that decoded only via a
            // legacy generation AND into impossible values is not "aging" — it
            // is the pseudo-decode this module exists to find.
            let class = if failed.is_empty() {
                RecordClass::Legacy
            } else {
                RecordClass::Implausible
            };
            (
                class,
                path,
                Some(m.id.0.to_string()),
                detail,
                Some(m.created_at),
                failed,
            )
        }
    };

    counts.record(class, &path, &checks);
    if class != RecordClass::Clean && class != RecordClass::Legacy {
        sweep.push_finding(Finding {
            record_class: "memory".to_string(),
            key: hex::encode(key),
            decoded_id,
            classification: class,
            decode_path: path,
            detail,
            created_at,
        });
    }
}

/// Scrub every entity node in a graph database.
///
/// Graph records carry no CRC32 — they use the two-byte postcard format tag,
/// not the SHO envelope — so [`RecordClass::ChecksumMismatch`] is structurally
/// unreachable here and always reports zero. That is a gap in the storage
/// format, not in the scrub, and it is why the plausibility checks carry the
/// whole load on graph nodes.
pub fn scrub_graph_nodes(db: &DB, cf: &ColumnFamily, sweep: &mut Sweep) -> ClassCounts {
    let mut counts = ClassCounts::default();
    let mut iter = db.raw_iterator_cf_opt(cf, scan_opts());
    iter.seek_to_first();

    loop {
        if let Err(e) = iter.status() {
            counts.read_errors += 1;
            tracing::warn!(error = %e, "integrity scrub: graph iterator error");
            break;
        }
        let Some(key) = iter.key() else { break };
        counts.keys_seen += 1;
        if !sweep.poll_deadline() {
            break;
        }
        if key.len() == 16 {
            if !sweep.tick() {
                break;
            }
            let key = key.to_vec();
            match iter.value() {
                Some(value) => classify_entity(&key, value, &mut counts, sweep),
                None => counts.read_errors += 1,
            }
        }
        iter.next();
    }
    counts
}

fn classify_entity(key: &[u8], value: &[u8], counts: &mut ClassCounts, sweep: &mut Sweep) {
    let (node, path) = match crate::graph_memory::decode_entity_node_for_scrub(value) {
        Ok((n, false)) => (n, "postcard_tagged".to_string()),
        Ok((n, true)) => (n, "legacy_or_defaulted".to_string()),
        Err(e) => {
            counts.record(RecordClass::Undecodable, "postcard_tagged", &[]);
            sweep.push_finding(Finding {
                record_class: "graph_node".to_string(),
                key: hex::encode(key),
                decoded_id: None,
                classification: RecordClass::Undecodable,
                decode_path: "postcard_tagged".to_string(),
                detail: e.to_string(),
                created_at: None,
            });
            return;
        }
    };

    let (failed, detail, dim) = check_entity(&node, key);
    if let Some(d) = dim {
        *counts.embedding_dims.entry(d).or_insert(0) += 1;
    }
    let class = if !failed.is_empty() {
        RecordClass::Implausible
    } else if path == "legacy_or_defaulted" {
        RecordClass::Legacy
    } else {
        RecordClass::Clean
    };

    counts.record(class, &path, &failed);
    if class == RecordClass::Implausible {
        sweep.push_finding(Finding {
            record_class: "graph_node".to_string(),
            key: hex::encode(key),
            decoded_id: Some(node.uuid.to_string()),
            classification: class,
            decode_path: path,
            detail,
            created_at: Some(node.created_at),
        });
    }
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// The numeric rule, rendered into the response so the judgement travels with
/// the numbers instead of living in a runbook nobody opens.
pub fn verdict_rule_text() -> String {
    format!(
        "unhealthy if implausible > 0 OR checksum_mismatch > 0 OR undecodable/scanned > {rate}; \
         degraded if undecodable > 0 at or below that rate; \
         aging if legacy > 0; \
         indeterminate if the sweep was incomplete or the iterator errored; \
         healthy only on a complete, error-free sweep with no defects. \
         Implausible is weighted hardest because an undecodable record fails \
         loudly while an implausible one is served as fact.",
        rate = UNDECODABLE_ALARM_RATE
    )
}

fn decide(memories: &ClassCounts, nodes: &ClassCounts, complete: bool) -> Verdict {
    let scanned = memories.scanned + nodes.scanned;
    let implausible = memories.implausible + nodes.implausible;
    let checksum = memories.checksum_mismatch + nodes.checksum_mismatch;
    let undecodable = memories.undecodable + nodes.undecodable;
    let read_errors = memories.read_errors + nodes.read_errors;
    let legacy = memories.legacy + nodes.legacy;

    let undecodable_rate = if scanned == 0 {
        0.0
    } else {
        undecodable as f64 / scanned as f64
    };

    // Evidence found is evidence, complete sweep or not: a partial sweep that
    // saw a fabricated record still saw it.
    if implausible > 0 || checksum > 0 || undecodable_rate > UNDECODABLE_ALARM_RATE {
        return Verdict::Unhealthy;
    }
    if undecodable > 0 {
        return Verdict::Degraded;
    }
    // Only a sweep that saw everything may report an absence of defects.
    // "0 corrupt" that means "we stopped at 200 records" is the exact defect
    // this module exists to catch, so it is refused here rather than explained
    // in a footnote.
    if !complete || read_errors > 0 {
        return Verdict::Indeterminate;
    }
    if legacy > 0 {
        return Verdict::Aging;
    }
    Verdict::Healthy
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Run a full scrub over one user's memory store and knowledge graph.
///
/// # Sampling
///
/// This is a full sweep, deliberately. Every defect population found so far is
/// a *write-date cohort*: the July NER desync hit every memory with a non-empty
/// NER list written before 2026-07-12; the `EntityLabel` renumbering hit every
/// node carrying `Other(...)` written before 2026-07-11; and the twenty-one
/// empty-content records this scrub found on the live claude-code store arrived
/// in two bursts, on 2026-04-01 and 2026-04-07, and nowhere else.
///
/// Uniform sampling is the wrong instrument for a cohort. A 5% sample that
/// happens to draw none of a twenty-one-record population reports zero defects
/// with complete confidence — and a sampled "0 corrupt" is worse than no scrub
/// at all, because it manufactures exactly the confidence that let the July
/// breakage survive a month.
///
/// # Cost, measured
///
/// On the live claude-code store (190MB, cold cache, read-only handle) the
/// memory sweep takes ~12.8s and the graph sweep ~0.5s. The other three
/// profiles finish in 25–400ms. That cost is *not* record decoding: it is
/// iterating the shared default column family, where 659,865 of 679,374 keys
/// belong to the fact, watermark, lineage and vmapping keyspaces rather than to
/// memories. Per-key cost is ~15–19µs on every profile regardless of how many
/// records are actually classified, which is the signature of a scan bound by
/// block reads rather than by work. Giving memories their own column family
/// would cut this roughly 35-fold; that is a storage-layout change, not a
/// change to this module.
///
/// 13s of `fill_cache(false)` sequential reading, off the request path, once an
/// hour, is a 0.4% duty cycle and evicts nothing. `ScrubBudget` exists to bound
/// pathological cases and to make a truncated sweep *say so* — not as a
/// sampling knob.
pub fn scrub_user(
    user_id: &str,
    memory_db: &DB,
    graph: Option<(&DB, &ColumnFamily)>,
    budget: ScrubBudget,
) -> IntegrityScrubReport {
    let started_at = Utc::now();
    let mut sweep = Sweep::new(budget);

    let memories = scrub_memories(memory_db, &mut sweep);

    let mut skipped = Vec::new();
    let graph_nodes = match graph {
        Some((gdb, cf)) => scrub_graph_nodes(gdb, cf, &mut sweep),
        None => {
            skipped.push("graph:entities".to_string());
            ClassCounts::default()
        }
    };

    let complete = sweep.stop_reason.is_none() && skipped.is_empty();
    let verdict = decide(&memories, &graph_nodes, complete);

    let mut checks_applied: Vec<String> = MEMORY_CHECKS.iter().map(|c| c.to_string()).collect();
    for c in GRAPH_CHECKS {
        let c = (*c).to_string();
        if !checks_applied.contains(&c) {
            checks_applied.push(c);
        }
    }

    let report = IntegrityScrubReport {
        user_id: user_id.to_string(),
        started_at,
        duration_ms: sweep.started.elapsed().as_millis() as u64,
        complete,
        stop_reason: sweep.stop_reason.clone(),
        skipped,
        memories,
        graph_nodes,
        checks_applied,
        findings: std::mem::take(&mut sweep.findings),
        findings_truncated: sweep.findings_truncated,
        verdict,
        verdict_rule: verdict_rule_text(),
        is_healthy: verdict == Verdict::Healthy,
    };

    // Alarm, not merely inform. A defence buyer does not ask whether you have
    // bugs; they ask how you would know. This is the line that answers it.
    match report.verdict {
        Verdict::Unhealthy => tracing::error!(
            user_id = %report.user_id,
            implausible = report.memories.implausible + report.graph_nodes.implausible,
            undecodable = report.memories.undecodable + report.graph_nodes.undecodable,
            checksum_mismatch =
                report.memories.checksum_mismatch + report.graph_nodes.checksum_mismatch,
            "integrity scrub: UNHEALTHY — stored records decode into impossible \
             values or fail their checksum"
        ),
        Verdict::Degraded | Verdict::Indeterminate => tracing::warn!(
            user_id = %report.user_id,
            verdict = ?report.verdict,
            complete = report.complete,
            stop_reason = ?report.stop_reason,
            "integrity scrub: health could not be certified"
        ),
        Verdict::Aging => tracing::info!(
            user_id = %report.user_id,
            legacy = report.memories.legacy + report.graph_nodes.legacy,
            "integrity scrub: records readable only via a legacy wire generation"
        ),
        Verdict::Healthy => tracing::info!(
            user_id = %report.user_id,
            scanned = report.memories.scanned + report.graph_nodes.scanned,
            "integrity scrub: clean"
        ),
    }

    report
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{Experience, Memory, MemoryId};
    use uuid::Uuid;

    fn budget() -> ScrubBudget {
        ScrubBudget {
            max_records: None,
            max_duration: None,
        }
    }

    fn memory_with(content: &str) -> Memory {
        Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience {
                content: content.to_string(),
                ..Default::default()
            },
            0.5,
            None,
            None,
            None,
            None,
        )
    }

    /// A memory database with the same column families the writer creates.
    fn memory_db(dir: &std::path::Path) -> DB {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        DB::open_cf(&opts, dir, ["default", "memory_index", "oplog"]).expect("open memory db")
    }

    fn graph_db(dir: &std::path::Path) -> DB {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        DB::open_cf(
            &opts,
            dir,
            ["default", crate::graph_memory::ENTITIES_CF_NAME],
        )
        .expect("open graph db")
    }

    fn entity_with(name: &str) -> EntityNode {
        let now = Utc::now();
        EntityNode {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            labels: vec![crate::graph_memory::EntityLabel::Person],
            created_at: now,
            last_seen_at: now,
            mention_count: 1,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        }
    }

    // -----------------------------------------------------------------------
    // THE DECISIVE TEST
    //
    // A record that decodes SUCCESSFULLY into a fabricated `Memory` must be
    // flagged. This is the exact shape described in the incident: a handful of
    // bytes of garbage that the legacy fallback chain turns into an
    // `Ok(Some(memory))` with no warning, no skip and no signal.
    // -----------------------------------------------------------------------

    #[test]
    fn pseudo_decode_is_flagged_implausible_not_clean() {
        let dir = tempfile::tempdir().unwrap();
        let db = memory_db(dir.path());

        let key = Uuid::new_v4();
        // 16 bytes of anything plus a few more is enough for
        // `try_raw_memory_parse`, which takes the first sixteen bytes as a
        // UUID and the rest as content, then fabricates every remaining field.
        let mut garbage = vec![0xAB_u8; 16];
        garbage.extend_from_slice(b"hello");
        db.put(key.as_bytes(), &garbage).unwrap();

        // Fail-first evidence, permanently asserted: the PRODUCTION decoder
        // reports this garbage as a perfectly good memory. Any scrub that asks
        // only "does it decode?" reports this store as healthy. If this
        // assertion ever fails, the pseudo-decode has been fixed at the source
        // and this test's premise needs revisiting -- it does not mean the
        // scrub is wrong.
        let laundered = crate::memory::storage::deserialize_memory_for_migration(&garbage);
        assert!(
            laundered.is_ok(),
            "premise of this test: the production fallback chain pseudo-decodes \
             garbage into a Memory. It returned an error instead: {:?}",
            laundered.err()
        );

        let mut sweep = Sweep::new(budget());
        let counts = scrub_memories(&db, &mut sweep);

        assert_eq!(counts.scanned, 1);
        assert_eq!(
            counts.clean, 0,
            "a fabricated record must never be counted clean"
        );
        assert_eq!(
            counts.undecodable, 0,
            "it decoded -- that is the whole point"
        );
        assert_eq!(
            counts.implausible, 1,
            "the fabricated record must land in the implausible class"
        );
        assert_eq!(counts.checks_failed.get("id_key_mismatch"), Some(&1));
        assert_eq!(sweep.findings.len(), 1);
        assert_eq!(sweep.findings[0].classification, RecordClass::Implausible);
    }

    // -----------------------------------------------------------------------
    // No false positives on healthy records. A signal that fires on good data
    // trains people to ignore the alarm, which is worse than no signal.
    // -----------------------------------------------------------------------

    #[test]
    fn records_written_by_the_real_writer_are_clean() {
        let dir = tempfile::tempdir().unwrap();
        let storage =
            crate::memory::storage::MemoryStorage::new(dir.path(), None).expect("storage");
        for i in 0..50 {
            storage
                .store(&memory_with(&format!("healthy record {i}")))
                .unwrap();
        }

        let mut sweep = Sweep::new(budget());
        let counts = scrub_memories(storage.raw_db(), &mut sweep);

        assert_eq!(counts.scanned, 50);
        assert_eq!(
            counts.clean, 50,
            "every check must pass on records the writer produced; \
             checks that fired: {:?}",
            counts.checks_failed
        );
        assert_eq!(counts.implausible, 0);
        assert_eq!(counts.legacy, 0);
        assert_eq!(counts.undecodable, 0);

        assert_eq!(
            decide(&counts, &ClassCounts::default(), true),
            Verdict::Healthy
        );
    }

    #[test]
    fn prefixed_subsystem_keys_are_not_mistaken_for_memories() {
        let dir = tempfile::tempdir().unwrap();
        let db = memory_db(dir.path());
        // The default CF is shared with several prefixed keyspaces.
        db.put(b"facts:some-entity", b"opaque").unwrap();
        db.put(b"_watermark:fact_extraction:u", b"12345").unwrap();
        db.put(b"stats:total", b"7").unwrap();

        let mut sweep = Sweep::new(budget());
        let counts = scrub_memories(&db, &mut sweep);
        assert_eq!(counts.keys_seen, 3);
        assert_eq!(
            counts.scanned, 0,
            "non-memory keyspaces must be skipped, not judged"
        );
        assert_eq!(counts.defects(), 0);
    }

    // -----------------------------------------------------------------------
    // Checksum mismatch must not be laundered into a legacy decode.
    // -----------------------------------------------------------------------

    #[test]
    fn crc_mismatch_is_reported_not_laundered_through_the_fallback_chain() {
        let dir = tempfile::tempdir().unwrap();
        let db = memory_db(dir.path());

        let m = memory_with("a record whose bytes later rotted");
        let mut bytes = crate::serialization::encode_sho(&m).unwrap();
        // Flip a bit in the payload, leaving the CRC trailer stale.
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        db.put(m.id.0.as_bytes(), &bytes).unwrap();

        let mut sweep = Sweep::new(budget());
        let counts = scrub_memories(&db, &mut sweep);
        assert_eq!(
            counts.checksum_mismatch, 1,
            "decode paths: {:?}",
            counts.decode_paths
        );
        assert_eq!(counts.clean, 0);
        assert_eq!(counts.legacy, 0);
        assert_eq!(
            sweep.findings[0].classification,
            RecordClass::ChecksumMismatch
        );

        assert_eq!(
            decide(&counts, &ClassCounts::default(), true),
            Verdict::Unhealthy,
            "bytes that do not match their own checksum are never a warning"
        );
    }

    // -----------------------------------------------------------------------
    // The July defect, reproduced: EntityLabel variant renumbering.
    // -----------------------------------------------------------------------

    /// `EntityLabel` as it was before twelve variants were inserted above
    /// `Other(String)` (c365eef3, 2026-07-11), moving it from index 23 to 35.
    /// Only the discriminant positions matter -- postcard addresses variants by
    /// declaration index, not by name.
    #[derive(Serialize)]
    #[allow(dead_code)]
    enum PreJulyEntityLabel {
        V0,
        V1,
        V2,
        V3,
        V4,
        V5,
        V6,
        V7,
        V8,
        V9,
        V10,
        V11,
        V12,
        V13,
        V14,
        V15,
        V16,
        V17,
        V18,
        V19,
        V20,
        V21,
        V22,
        /// Index 23 -- where `Other(String)` lived before the renumbering.
        Other(String),
    }

    /// `EntityNode` with the pre-renumbering label type. Field order and count
    /// are identical to the live struct, so the only wire difference is the
    /// label discriminant -- exactly the July change.
    #[derive(Serialize)]
    struct PreJulyEntityNode {
        uuid: Uuid,
        name: String,
        labels: Vec<PreJulyEntityLabel>,
        created_at: DateTime<Utc>,
        last_seen_at: DateTime<Utc>,
        mention_count: usize,
        summary: String,
        attributes: std::collections::HashMap<String, String>,
        name_embedding: Option<Vec<f32>>,
        salience: f32,
        is_proper_noun: bool,
        selectivity: Option<f32>,
        fine_type: Option<String>,
        kb_id: Option<String>,
    }

    fn pre_july_node(label_payload: &str) -> (Uuid, Vec<u8>) {
        let uuid = Uuid::new_v4();
        let now = Utc::now();
        let node = PreJulyEntityNode {
            uuid,
            name: "Maersk".to_string(),
            labels: vec![PreJulyEntityLabel::Other(label_payload.to_string())],
            created_at: now,
            last_seen_at: now,
            mention_count: 3,
            summary: String::new(),
            attributes: std::collections::HashMap::new(),
            name_embedding: None,
            salience: 0.5,
            is_proper_noun: true,
            selectivity: None,
            fine_type: None,
            kb_id: None,
        };
        (uuid, crate::serialization::encode(&node).unwrap())
    }

    #[test]
    fn renumbered_entity_label_is_never_reported_clean() {
        let dir = tempfile::tempdir().unwrap();
        let db = graph_db(dir.path());
        let cf = db.cf_handle(crate::graph_memory::ENTITIES_CF_NAME).unwrap();

        let (uuid, bytes) = pre_july_node("shipping_line");
        db.put_cf(cf, uuid.as_bytes(), &bytes).unwrap();

        let mut sweep = Sweep::new(budget());
        let counts = scrub_graph_nodes(&db, cf, &mut sweep);

        assert_eq!(counts.scanned, 1);
        assert_eq!(
            counts.clean, 0,
            "a node written before the EntityLabel renumbering is not clean; \
             decode paths: {:?}, checks: {:?}",
            counts.decode_paths, counts.checks_failed
        );
        assert_eq!(counts.legacy, 0, "this is corruption, not aging");
        assert_eq!(counts.defects(), 1);
    }

    #[test]
    fn renumbered_entity_label_surfaces_as_undecodable_and_trips_the_alarm() {
        // EMPIRICAL CORRECTION, and it matters.
        //
        // The obvious expectation is that the renumbering produces a silent
        // wrong decode: index 23 reads as the unit variant `Norp`, the string
        // payload lands in `created_at`, the node decodes into a wrong value.
        // It does not. `DateTime<Utc>` serialises as a STRING, so the desync
        // makes every subsequent length varint misaligned; the decode dies on
        // "Hit the end of buffer" or "Tried to parse invalid utf-8" long
        // before any field can be inspected. Probed across payloads and tail
        // sizes from 0 to 20,000 bytes, every single shape errored.
        //
        // That agrees with the live symptom: graph traverse returned 500, an
        // error, not wrong answers. So on graph nodes this defect class is
        // caught as UNDECODABLE, and it is the undecodable RATE that raises
        // the alarm — 74% of the live graph, three orders of magnitude above
        // the threshold. The timestamp bound does not catch this one, and the
        // scrub must not claim it does.
        let dir = tempfile::tempdir().unwrap();
        let db = graph_db(dir.path());
        let cf = db.cf_handle(crate::graph_memory::ENTITIES_CF_NAME).unwrap();

        for payload in ["shipping_line", "1970-01-01T00:00:00Z", "x"] {
            let (uuid, bytes) = pre_july_node(payload);
            assert!(
                crate::graph_memory::decode_entity_node_for_scrub(&bytes).is_err(),
                "payload {payload:?} decoded; if this ever succeeds the class                  moves from undecodable to implausible and the plausibility                  checks become the load-bearing signal for it"
            );
            db.put_cf(cf, uuid.as_bytes(), &bytes).unwrap();
        }

        let mut sweep = Sweep::new(budget());
        let counts = scrub_graph_nodes(&db, cf, &mut sweep);
        assert_eq!(counts.scanned, 3);
        assert_eq!(counts.undecodable, 3);
        assert_eq!(counts.clean, 0);
        assert_eq!(sweep.findings.len(), 3);

        // A whole-cohort breakage is far above the alarm rate, so it reports
        // unhealthy rather than degraded.
        assert_eq!(
            decide(&ClassCounts::default(), &counts, true),
            Verdict::Unhealthy
        );
    }

    #[test]
    fn healthy_entity_nodes_are_clean() {
        let dir = tempfile::tempdir().unwrap();
        let db = graph_db(dir.path());
        let cf = db.cf_handle(crate::graph_memory::ENTITIES_CF_NAME).unwrap();
        for i in 0..20 {
            let n = entity_with(&format!("Entity {i}"));
            let bytes = crate::serialization::encode(&n).unwrap();
            db.put_cf(cf, n.uuid.as_bytes(), &bytes).unwrap();
        }

        let mut sweep = Sweep::new(budget());
        let counts = scrub_graph_nodes(&db, cf, &mut sweep);
        assert_eq!(counts.clean, 20, "checks fired: {:?}", counts.checks_failed);
        assert_eq!(counts.defects(), 0);
    }

    // -----------------------------------------------------------------------
    // A partial sweep must say so, and must never certify health.
    // -----------------------------------------------------------------------

    #[test]
    fn truncated_sweep_reports_incomplete_and_refuses_to_certify_health() {
        let dir = tempfile::tempdir().unwrap();
        let storage =
            crate::memory::storage::MemoryStorage::new(dir.path(), None).expect("storage");
        for i in 0..20 {
            storage.store(&memory_with(&format!("record {i}"))).unwrap();
        }

        let mut sweep = Sweep::new(ScrubBudget {
            max_records: Some(3),
            max_duration: None,
        });
        let counts = scrub_memories(storage.raw_db(), &mut sweep);

        assert_eq!(counts.scanned, 3);
        assert_eq!(counts.defects(), 0, "the three it saw were fine");
        assert!(sweep.stop_reason.is_some(), "it must say why it stopped");

        assert_eq!(
            decide(&counts, &ClassCounts::default(), false),
            Verdict::Indeterminate,
            "'0 corrupt' that means 'we stopped at 3 records' must never read \
             as healthy"
        );
    }

    #[test]
    fn a_budget_spent_on_memories_does_not_leave_the_graph_looking_clean() {
        // The budget is shared across every column family in one report. If
        // the memory sweep spends it, the graph sweep must come back visibly
        // empty and the report must be incomplete -- an untouched section
        // reading "0 defects" beside a truncated one is the same lie this
        // module exists to prevent, one level down.
        let dir = tempfile::tempdir().unwrap();
        let storage =
            crate::memory::storage::MemoryStorage::new(dir.path(), None).expect("storage");
        for i in 0..10 {
            storage.store(&memory_with(&format!("record {i}"))).unwrap();
        }

        let gdir = tempfile::tempdir().unwrap();
        let gdb = graph_db(gdir.path());
        let cf = gdb
            .cf_handle(crate::graph_memory::ENTITIES_CF_NAME)
            .unwrap();
        // A node that would be flagged if it were ever reached.
        let (uuid, bytes) = pre_july_node("shipping_line");
        gdb.put_cf(cf, uuid.as_bytes(), &bytes).unwrap();

        let report = scrub_user(
            "u",
            storage.raw_db(),
            Some((&gdb, cf)),
            ScrubBudget {
                max_records: Some(2),
                max_duration: None,
            },
        );

        assert_eq!(report.memories.scanned, 2);
        assert_eq!(report.graph_nodes.scanned, 0, "the graph was never reached");
        assert_eq!(report.graph_nodes.defects(), 0);
        assert!(!report.complete);
        assert!(report.stop_reason.is_some());
        assert_eq!(report.verdict, Verdict::Indeterminate);
        assert!(!report.is_healthy);
    }

    #[test]
    fn a_skipped_column_family_forbids_a_healthy_verdict() {
        let dir = tempfile::tempdir().unwrap();
        let storage =
            crate::memory::storage::MemoryStorage::new(dir.path(), None).expect("storage");
        storage.store(&memory_with("only record")).unwrap();

        // No graph handed in: the report must name the gap and refuse to
        // certify, rather than reporting health for half the data.
        let report = scrub_user("u", storage.raw_db(), None, budget());
        assert_eq!(report.skipped, vec!["graph:entities".to_string()]);
        assert!(!report.complete);
        assert_eq!(report.verdict, Verdict::Indeterminate);
        assert!(!report.is_healthy);
    }

    #[test]
    fn read_errors_forbid_a_healthy_verdict() {
        let counts = ClassCounts {
            scanned: 100,
            clean: 100,
            read_errors: 1,
            ..Default::default()
        };
        assert_eq!(
            decide(&counts, &ClassCounts::default(), true),
            Verdict::Indeterminate,
            "an inconclusive read is not an absent record"
        );
    }

    #[test]
    fn the_report_names_every_check_it_applied() {
        let dir = tempfile::tempdir().unwrap();
        let storage =
            crate::memory::storage::MemoryStorage::new(dir.path(), None).expect("storage");
        storage.store(&memory_with("x")).unwrap();
        let report = scrub_user("u", storage.raw_db(), None, budget());
        for c in MEMORY_CHECKS.iter().chain(GRAPH_CHECKS.iter()) {
            assert!(
                report.checks_applied.iter().any(|a| a == c),
                "check {c} ran but is not declared in the report"
            );
        }
        assert!(!report.verdict_rule.is_empty());
    }

    // -----------------------------------------------------------------------
    // Verdict arithmetic.
    // -----------------------------------------------------------------------

    #[test]
    fn one_implausible_record_outweighs_any_number_of_clean_ones() {
        let c = ClassCounts {
            scanned: 100_000,
            clean: 99_999,
            implausible: 1,
            ..Default::default()
        };
        assert_eq!(
            decide(&c, &ClassCounts::default(), true),
            Verdict::Unhealthy
        );
    }

    #[test]
    fn a_few_undecodable_records_are_degraded_not_unhealthy() {
        // Undecodable records fail loudly: they error rather than serving a
        // fabrication. Below the alarm rate that is a maintenance item.
        let c = ClassCounts {
            scanned: 19_438,
            clean: 19_432,
            undecodable: 6,
            ..Default::default()
        };
        assert!((c.undecodable as f64 / c.scanned as f64) < UNDECODABLE_ALARM_RATE);
        assert_eq!(decide(&c, &ClassCounts::default(), true), Verdict::Degraded);
    }

    #[test]
    fn a_systematic_undecodable_population_is_unhealthy() {
        let c = ClassCounts {
            scanned: 19_438,
            clean: 18_456,
            undecodable: 982, // the July NER cohort
            ..Default::default()
        };
        assert!((c.undecodable as f64 / c.scanned as f64) > UNDECODABLE_ALARM_RATE);
        assert_eq!(
            decide(&c, &ClassCounts::default(), true),
            Verdict::Unhealthy
        );
    }

    #[test]
    fn legacy_records_are_aging_not_broken() {
        let c = ClassCounts {
            scanned: 100,
            clean: 90,
            legacy: 10,
            ..Default::default()
        };
        assert_eq!(decide(&c, &ClassCounts::default(), true), Verdict::Aging);
    }

    // -----------------------------------------------------------------------
    // Individual plausibility checks: each fires on a bad value and stays
    // silent on a good one.
    // -----------------------------------------------------------------------

    fn checks_for(m: &Memory) -> Vec<&'static str> {
        check_memory(m, m.id.0.as_bytes()).0
    }

    fn memory_from(e: Experience) -> Memory {
        Memory::new(MemoryId(Uuid::new_v4()), e, 0.5, None, None, None, None)
    }

    #[test]
    fn check_timestamp_out_of_bounds() {
        let mut m = memory_with("x");
        assert!(checks_for(&m).is_empty());
        m.created_at = Utc.timestamp_opt(0, 0).single().unwrap();
        assert!(checks_for(&m).contains(&"timestamp_out_of_bounds"));

        let mut m = memory_with("x");
        m.created_at = Utc::now() + chrono::Duration::days(30);
        assert!(checks_for(&m).contains(&"timestamp_out_of_bounds"));
    }

    #[test]
    fn check_timestamp_inverted() {
        let mut m = memory_with("x");
        // created_at moved forward while last_accessed stays at construction
        // time, i.e. the record claims it was read before it existed.
        m.created_at = Utc::now() + chrono::Duration::hours(1);
        let fired = checks_for(&m);
        assert!(fired.contains(&"timestamp_inverted"), "fired: {fired:?}");
    }

    #[test]
    fn check_unit_range() {
        assert!(checks_for(&memory_with("x")).is_empty());

        let bad = Experience {
            content: "x".to_string(),
            confidence: Some(42.0),
            ..Default::default()
        };
        assert!(checks_for(&memory_from(bad)).contains(&"unit_range"));

        let nan = Experience {
            content: "x".to_string(),
            confidence: Some(f32::NAN),
            ..Default::default()
        };
        assert!(checks_for(&memory_from(nan)).contains(&"unit_range"));
    }

    #[test]
    fn check_reward_range() {
        let ok = Experience {
            content: "x".to_string(),
            reward: Some(0.5),
            ..Default::default()
        };
        assert!(checks_for(&memory_from(ok)).is_empty());

        let bad = Experience {
            content: "x".to_string(),
            reward: Some(1e9),
            ..Default::default()
        };
        assert!(checks_for(&memory_from(bad)).contains(&"reward_range"));
    }

    #[test]
    fn check_content_empty() {
        assert!(checks_for(&memory_from(Experience::default())).contains(&"content_empty"));
    }

    #[test]
    fn check_embedding_signals() {
        let unit: Vec<f32> = {
            let mut v = vec![0.0_f32; 384];
            v[0] = 1.0;
            v
        };
        let mut e = Experience {
            content: "x".to_string(),
            embeddings: Some(unit.clone()),
            ..Default::default()
        };
        assert!(
            checks_for(&memory_from(e.clone())).is_empty(),
            "unit-norm embedding is healthy"
        );

        // An all-zero vector is what the write path deliberately stores when
        // normalisation fails. It is not corruption and must not be flagged.
        e.embeddings = Some(vec![0.0_f32; 384]);
        assert!(
            checks_for(&memory_from(e.clone())).is_empty(),
            "the documented zero-vector fallback must not raise a false alarm"
        );

        e.embeddings = Some({
            let mut v = unit.clone();
            v[3] = f32::NAN;
            v
        });
        assert!(checks_for(&memory_from(e.clone())).contains(&"embedding_non_finite"));

        e.embeddings = Some(vec![7.0_f32; 384]);
        assert!(checks_for(&memory_from(e.clone())).contains(&"embedding_not_unit_norm"));

        e.embeddings = Some(vec![1.0_f32; 4]);
        assert!(checks_for(&memory_from(e)).contains(&"embedding_dimension"));
    }

    #[test]
    fn check_entity_name_and_salience() {
        let n = entity_with("Maersk");
        assert!(check_entity(&n, n.uuid.as_bytes()).0.is_empty());

        let empty = entity_with("");
        assert!(check_entity(&empty, empty.uuid.as_bytes())
            .0
            .contains(&"name_empty"));

        let mut bad = entity_with("Maersk");
        bad.salience = 12.0;
        assert!(check_entity(&bad, bad.uuid.as_bytes())
            .0
            .contains(&"unit_range"));

        let mut nan = entity_with("Maersk");
        nan.salience = f32::NAN;
        assert!(check_entity(&nan, nan.uuid.as_bytes())
            .0
            .contains(&"unit_range"));
    }

    // -----------------------------------------------------------------------
    // Candidate signals we REJECTED, with the evidence that rejects them.
    // -----------------------------------------------------------------------

    #[test]
    fn enum_discriminant_range_checks_would_be_vacuous() {
        // A post-decode "is this tier/label discriminant in range?" check can
        // never fire: postcard rejects an out-of-range variant index during
        // decode, so an implausible discriminant surfaces as Undecodable, not
        // as a decoded-but-wrong value.
        let mut bytes = postcard::to_allocvec(&crate::memory::types::MemoryTier::Working).unwrap();
        bytes[0] = 200;
        let decoded: Result<crate::memory::types::MemoryTier, _> = postcard::from_bytes(&bytes);
        assert!(
            decoded.is_err(),
            "postcard accepted an out-of-range discriminant; if this ever \
             passes, an enum-range plausibility check becomes worth adding"
        );
    }

    #[test]
    fn content_hash_index_is_not_a_per_record_checksum() {
        // The dedup index is `content_hash:{sha256} -> memory_id`, one slot per
        // CONTENT, last writer wins. Two memories with identical content
        // legitimately leave the index pointing at only one of them, so a
        // "hash maps back to my id" check false-positives on every duplicate.
        // This test pins that behaviour so the signal is not adopted later on
        // the assumption that it is a per-record integrity field.
        let dir = tempfile::tempdir().unwrap();
        let storage =
            crate::memory::storage::MemoryStorage::new(dir.path(), None).expect("storage");
        let a = memory_with("identical content");
        let b = memory_with("identical content");
        storage.store(&a).unwrap();
        storage.store(&b).unwrap();

        let resolved = storage.get_by_content_hash("identical content");
        assert_eq!(
            resolved,
            Some(b.id),
            "the dedup index resolves content to the LAST writer"
        );
        assert_ne!(
            resolved,
            Some(a.id),
            "so the first memory would be falsely flagged by a hash round-trip \
             check despite being perfectly healthy"
        );

        // Both records are in fact clean.
        let mut sweep = Sweep::new(budget());
        let counts = scrub_memories(storage.raw_db(), &mut sweep);
        assert_eq!(counts.clean, 2);
    }
}
