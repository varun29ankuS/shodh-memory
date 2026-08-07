//! Agent-traceability operation log: record type, hash chain, integrity verification.
//!
//! This module owns the *source of truth* for agent traceability (spec
//! `docs/superpowers/specs/2026-07-30-agent-traceability-design.md` §3.2): an
//! append-only log of operations, each record cryptographically chained to its
//! predecessor so that any post-hoc edit, deletion, or reordering is detectable.
//!
//! # What the chain proves — and what it does not
//!
//! `verify_chain` proves that a *presented* sequence of records is exactly the
//! sequence that was sealed, in order, for one `(session_id, user_id)` pair.
//! It cannot prove that nothing was ever removed from the *store*: an attacker
//! (or an operator) with write access to the column family can truncate the log
//! and the remaining prefix still verifies. Spec §3.2's corrected immutability
//! scope names the two legitimate whole-store, chain-terminating events —
//! user deletion (`forget_user`) and backup restore — so audit language must
//! say "tamper-evident within the recorded history", never "nothing was ever
//! deleted".
//!
//! # Durability
//!
//! Appends inherit the storage layer's configured write mode rather than
//! forcing an fsync per record: a forced fsync per append is irreconcilable
//! with the <100 µs median append budget. RocksDB's WAL is written on every
//! append (manual WAL flush is disabled), so records survive a **process**
//! crash; a power loss or OS crash can lose the last unflushed appends.
//! Operators who require power-loss durability for the audit trail set
//! `SHODH_WRITE_MODE=sync`, which then also covers the oplog. Deliberate,
//! documented tradeoff — not a silent default.
//!
//! # Persistence round-trip invariant
//!
//! `verify_chain` recomputes hashes over records that were deserialized from
//! disk. Any persistence format used for `OpRecord` MUST round-trip every field
//! bit-exactly — including `ts`/`reported_ts` sub-second precision — or
//! verification reports **false tampering** on untouched data.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// `attestation` value for operations the engine observed itself.
pub const ATTESTATION_WITNESSED: &str = "witnessed";

/// `attestation` value for operations reported by an external party
/// (Claude Code hooks today; Buzz/OTel connectors later). Self-asserted:
/// the engine vouches only for the fact that the report arrived.
pub const ATTESTATION_REPORTED: &str = "reported";

/// Maximum `payload_summary` length in **characters** (not bytes) — a
/// multi-byte summary at the limit can exceed 2048 bytes on the wire.
pub const PAYLOAD_SUMMARY_MAX_CHARS: usize = 2048;

/// Domain tag mixed into the genesis digest. Bumping the version deliberately
/// invalidates every existing chain's genesis link, so treat it as WIRE-BREAKING.
const GENESIS_DOMAIN: &[u8] = b"shodh.oplog.genesis.v1";

/// Domain tag mixed into every link digest. WIRE-BREAKING to change.
const CHAIN_DOMAIN: &[u8] = b"shodh.oplog.chain.v1";

/// Field separator for domain-tagged digest inputs. `0x00` cannot occur in a
/// validated `session_id`/`user_id`, so tag/field concatenation is unambiguous:
/// without it `("ab", "c")` and `("a", "bc")` would hash identically.
const DOMAIN_SEP: u8 = 0x00;

/// One traced operation.
///
/// # WIRE FORMAT WARNING — field order IS the wire format
///
/// [`canonical_bytes`] serializes this struct **directly** with `serde_json`,
/// and `serde_json` emits struct fields in **declaration order**. That byte
/// sequence is the pre-image of every `hash` in every persisted chain.
/// Therefore:
///
/// - **Reordering these fields is a WIRE-BREAKING change.** Every stored record
///   would fail `verify_chain` with a hash mismatch, and the log — whose entire
///   purpose is tamper evidence — would report tampering on untouched data. There
///   is no migration: the digests cannot be recomputed without the original order.
/// - **Renaming a field, or adding/removing one, is equally WIRE-BREAKING**, for
///   the same reason (JSON keys are part of the pre-image).
/// - New fields may only be introduced behind a new chain domain version
///   (`CHAIN_DOMAIN`) with an explicit, logged chain-termination for existing
///   sessions — never as a quiet additive change.
///
/// This is the same lesson as `RelationType`'s postcard discriminants
/// (`graph_memory.rs`): a derived serde impl silently couples declaration order
/// to on-disk bytes. `canonical_bytes` deliberately does *not* round-trip
/// through `serde_json::Value`, because `Value`'s map would reorder keys.
///
/// A unit test (`canonical_bytes_field_order_is_the_wire_contract`) pins the
/// order. If that test fails, the diff is wire-breaking — do not update the test.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpRecord {
    /// Position in this session's chain. Genesis is `0`, then strictly `+1`.
    pub seq: u64,
    /// Engine arrival time (server clock) — authoritative ordering source.
    pub ts: chrono::DateTime<chrono::Utc>,
    /// Session this record belongs to. Verified against the chain's header.
    pub session_id: String,
    /// Acting agent identity. Verified against the chain's header.
    ///
    /// Note: HTTP auth establishes a valid API key, not a verified user, so
    /// this is the identity the request *asserted*, witnessed by the engine.
    pub user_id: String,
    /// Operation name, e.g. `"recall"`, `"remember"`, `"report:file_edit"`.
    pub op: String,
    /// [`ATTESTATION_WITNESSED`] or [`ATTESTATION_REPORTED`]. Never blended.
    pub attestation: String,
    /// Bounded request summary — build it with [`bound_payload_summary`].
    pub payload_summary: String,
    /// Memory ids touched (returned by a recall, stored by a write).
    pub evidence_refs: Vec<String>,
    /// `"ok"` or `"error:<status>"`.
    pub outcome: String,
    /// Reporter's own clock, preserved alongside `ts`. Reported records only.
    pub reported_ts: Option<chrono::DateTime<chrono::Utc>>,
    /// Reporter identifier (e.g. `"claude-code-hook"`). Reported records only.
    pub source: Option<String>,
    /// Hex SHA-256 of the predecessor's `hash`-chained digest; for genesis, the
    /// output of [`genesis_hash`] over the session header.
    pub prev_hash: String,
    /// Hex SHA-256 of THIS record: `chain_hash(prev_hash, canonical_bytes(self))`,
    /// where `canonical_bytes` blanks this very field.
    pub hash: String,
}

/// Caller-supplied fields for a new [`OpRecord`], i.e. every field except the
/// three the storage layer computes at append time under the per-session
/// head lock: `seq`, `prev_hash`, and `hash`.
///
/// See `MemoryStorage::oplog_append` (`memory/storage.rs`), which turns a
/// draft into a sealed, chained `OpRecord` and persists it in `CF_OPLOG`.
#[derive(Debug, Clone)]
pub struct OpRecordDraft {
    /// Engine arrival time (server clock) — authoritative ordering source.
    pub ts: chrono::DateTime<chrono::Utc>,
    /// Session this record belongs to. Must pass `validate_session_id`
    /// (rejects `:`, which would otherwise break the `op:{session_id}:`
    /// prefix scan) — enforced by `oplog_append`, not by this type.
    pub session_id: String,
    /// Acting agent identity, as asserted by the request (see [`OpRecord::user_id`]).
    pub user_id: String,
    /// Operation name, e.g. `"recall"`, `"remember"`, `"report:file_edit"`.
    pub op: String,
    /// [`ATTESTATION_WITNESSED`] or [`ATTESTATION_REPORTED`]. Never blended.
    pub attestation: String,
    /// Bounded request summary — build it with [`bound_payload_summary`].
    pub payload_summary: String,
    /// Memory ids touched (returned by a recall, stored by a write).
    pub evidence_refs: Vec<String>,
    /// `"ok"` or `"error:<status>"`.
    pub outcome: String,
    /// Reporter's own clock, preserved alongside `ts`. Reported records only.
    pub reported_ts: Option<chrono::DateTime<chrono::Utc>>,
    /// Reporter identifier (e.g. `"claude-code-hook"`). Reported records only.
    pub source: Option<String>,
}

/// Why chain verification failed, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainError {
    /// The **expected** sequence number at the failing position (i.e. the index
    /// into the verified slice), not the `seq` read off the offending record —
    /// a tampered or misplaced record's own `seq` is untrustworthy by definition.
    pub at_seq: u64,
    /// Human-readable cause, naming the mismatching field.
    pub reason: String,
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "oplog chain verification failed at seq {}: {}",
            self.at_seq, self.reason
        )
    }
}

impl std::error::Error for ChainError {}

/// Canonical hash pre-image for `r`: `serde_json` bytes of the record with the
/// `hash` field blanked (a record cannot commit to its own digest).
///
/// Determinism rests on two properties, both load-bearing:
/// 1. `serde_json` emits derived-struct fields in **declaration order** — see
///    the WIRE FORMAT WARNING on [`OpRecord`].
/// 2. The struct is serialized **directly**, never via `serde_json::Value`,
///    whose map representation would reorder keys.
pub fn canonical_bytes(r: &OpRecord) -> Vec<u8> {
    let mut blanked = r.clone();
    blanked.hash = String::new();
    // Serializing a struct of owned Strings/ints/Vec<String> cannot fail: there
    // are no non-string map keys and no non-finite floats, the only two
    // documented serde_json failure modes.
    serde_json::to_vec(&blanked)
        .expect("OpRecord contains no non-string map keys or non-finite floats")
}

/// Links `canonical` to its predecessor: hex SHA-256 over
/// `CHAIN_DOMAIN || 0x00 || prev_hash || 0x00 || canonical`.
///
/// `prev_hash` is fixed-width (64 hex chars from [`genesis_hash`] or a prior
/// `chain_hash`), so the concatenation is unambiguous even without the
/// separator; the separator and domain tag are belt-and-braces against a
/// future caller passing a variable-width value.
pub fn chain_hash(prev_hash: &str, canonical: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CHAIN_DOMAIN);
    hasher.update([DOMAIN_SEP]);
    hasher.update(prev_hash.as_bytes());
    hasher.update([DOMAIN_SEP]);
    hasher.update(canonical);
    hex::encode(hasher.finalize())
}

/// `prev_hash` for a session's genesis record: hex SHA-256 over the session
/// header `GENESIS_DOMAIN || 0x00 || session_id || 0x00 || user_id`.
///
/// The domain tag and separators bind the digest to this purpose and prevent
/// field-boundary collisions (`("ab","c")` vs `("a","bc")`).
pub fn genesis_hash(session_id: &str, user_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(GENESIS_DOMAIN);
    hasher.update([DOMAIN_SEP]);
    hasher.update(session_id.as_bytes());
    hasher.update([DOMAIN_SEP]);
    hasher.update(user_id.as_bytes());
    hex::encode(hasher.finalize())
}

/// Bounds a summary to [`PAYLOAD_SUMMARY_MAX_CHARS`] characters, marking
/// truncation in-band.
///
/// In-band marking (rather than a `truncated: bool` field, the convention used
/// by `handlers/crud.rs`) is forced by [`OpRecord`]'s frozen wire format: adding
/// a field is wire-breaking. The marker records the *original* character count
/// so the omitted amount is recoverable from the record alone.
///
/// The bound is in **characters**; truncation always lands on a `char`
/// boundary, so multi-byte input can produce more than 2048 bytes.
pub fn bound_payload_summary(summary: &str) -> String {
    let total = summary.chars().count();
    if total <= PAYLOAD_SUMMARY_MAX_CHARS {
        return summary.to_string();
    }
    // The marker names the original length (a fixed, already-known quantity),
    // so its own length is knowable before truncating — no circular sizing.
    let marker = format!("[truncated: original {total} chars]");
    let keep = PAYLOAD_SUMMARY_MAX_CHARS.saturating_sub(marker.chars().count());
    let mut out: String = summary.chars().take(keep).collect();
    out.push_str(&marker);
    out
}

/// Verifies that `records` is the exact, complete, in-order chain sealed for
/// `(session_id, user_id)`.
///
/// Checks, per position `i` (failing fast with `at_seq = i`):
/// 1. **Identity** — `session_id` and `user_id` match the arguments. This is
///    what turns a prefix-scan bleed (a `:`-bearing `session_id` letting a
///    foreign session's keys match `op:{sid}:`) into a *detected* error instead
///    of a hash mismatch misread as tampering.
/// 2. **Sequence** — `records[0].seq == 0` and every `seq` is `i`.
/// 3. **Linkage** — `records[i].prev_hash` equals `records[i-1].hash`, and
///    `records[0].prev_hash` equals [`genesis_hash`].
/// 4. **Integrity** — `records[i].hash` equals the recomputed
///    `chain_hash(prev_hash, canonical_bytes(record))`.
///
/// An empty slice is `Ok`: a session with no recorded operations has nothing to
/// contradict. Callers that require a non-empty trace must check separately —
/// "verified" here never means "complete" (see the module docs).
pub fn verify_chain(
    records: &[OpRecord],
    session_id: &str,
    user_id: &str,
) -> Result<(), ChainError> {
    let mut expected_prev = genesis_hash(session_id, user_id);

    for (i, r) in records.iter().enumerate() {
        let at_seq = i as u64;
        let fail = |reason: String| Err(ChainError { at_seq, reason });

        if r.session_id != session_id {
            return fail(format!(
                "session_id mismatch: record claims {:?}, chain is {:?}",
                r.session_id, session_id
            ));
        }
        if r.user_id != user_id {
            return fail(format!(
                "user_id mismatch: record claims {:?}, chain is {:?}",
                r.user_id, user_id
            ));
        }
        if r.seq != at_seq {
            return fail(format!(
                "sequence mismatch: expected seq {at_seq}, record carries {}",
                r.seq
            ));
        }
        if r.prev_hash != expected_prev {
            let what = if i == 0 { "genesis" } else { "prev_hash" };
            return fail(format!(
                "{what} link broken: expected {expected_prev}, record carries {}",
                r.prev_hash
            ));
        }
        let recomputed = chain_hash(&r.prev_hash, &canonical_bytes(r));
        if recomputed != r.hash {
            return fail(format!(
                "record hash mismatch (content altered after sealing): expected {recomputed}, record carries {}",
                r.hash
            ));
        }

        expected_prev = r.hash.clone();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, TimeZone, Utc};

    /// Fixed epoch second for deterministic timestamps: 2026-07-30T12:00:00Z.
    ///
    /// Tests MUST NOT use `Utc::now()` — canonical bytes and therefore every
    /// hash in the chain depend on `ts`, so a wall-clock timestamp makes the
    /// expected digests irreproducible.
    const FIXED_EPOCH_SECS: i64 = 1_785_412_800;

    fn fixed_ts(offset_secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(FIXED_EPOCH_SECS + offset_secs, 0)
            .single()
            .expect("fixed test timestamp must be valid")
    }

    /// Builds a record for `session_id`/`user_id` and seals it: `hash` is set to
    /// `chain_hash(prev_hash, canonical_bytes(record-with-blank-hash))`.
    fn mk_record_for(
        seq: u64,
        op: &str,
        prev_hash: String,
        session_id: &str,
        user_id: &str,
    ) -> OpRecord {
        let mut r = OpRecord {
            seq,
            ts: fixed_ts(seq as i64),
            session_id: session_id.to_string(),
            user_id: user_id.to_string(),
            op: op.to_string(),
            attestation: ATTESTATION_WITNESSED.to_string(),
            payload_summary: format!("payload for {op}"),
            evidence_refs: vec![format!("mem-{seq}")],
            outcome: "ok".to_string(),
            reported_ts: None,
            source: None,
            prev_hash,
            hash: String::new(),
        };
        r.hash = chain_hash(&r.prev_hash, &canonical_bytes(&r));
        r
    }

    fn mk_record(seq: u64, op: &str, prev_hash: String) -> OpRecord {
        mk_record_for(seq, op, prev_hash, "s1", "u1")
    }

    #[test]
    fn chain_links_and_verifies() {
        let r1 = mk_record(0, "recall", genesis_hash("s1", "u1"));
        let r2 = mk_record(1, "remember", r1.hash.clone());
        assert!(verify_chain(&[r1.clone(), r2.clone()], "s1", "u1").is_ok());
    }

    #[test]
    fn tamper_breaks_chain() {
        let r1 = mk_record(0, "recall", genesis_hash("s1", "u1"));
        let mut r2 = mk_record(1, "remember", r1.hash.clone());
        r2.payload_summary = "tampered".into(); // content changed, hash not recomputed
        let err = verify_chain(&[r1, r2], "s1", "u1").unwrap_err();
        assert_eq!(err.at_seq, 1);
    }

    #[test]
    fn reorder_breaks_chain() {
        let r1 = mk_record(0, "recall", genesis_hash("s1", "u1"));
        let r2 = mk_record(1, "remember", r1.hash.clone());
        let r3 = mk_record(2, "recall", r2.hash.clone());
        // In-order chain is valid.
        assert!(verify_chain(&[r1.clone(), r2.clone(), r3.clone()], "s1", "u1").is_ok());
        // Swap the last two: position 1 now holds seq 2.
        let err = verify_chain(&[r1, r3, r2], "s1", "u1").unwrap_err();
        // `at_seq` is the EXPECTED sequence number at the failing position.
        assert_eq!(err.at_seq, 1);
    }

    #[test]
    fn canonical_bytes_stable_and_hash_field_blanked() {
        let r = mk_record(0, "recall", genesis_hash("s", "u"));
        let b1 = canonical_bytes(&r);
        let mut r2 = r.clone();
        r2.hash = "different".into();
        assert_eq!(
            b1,
            canonical_bytes(&r2),
            "hash field must not affect canonical bytes"
        );
        // Repeated calls are byte-identical (no map reordering).
        assert_eq!(b1, canonical_bytes(&r));
    }

    /// Pins the wire order. Struct field order IS the canonical byte order, so
    /// reordering `OpRecord`'s fields invalidates every stored hash. If this
    /// test fails, the change is WIRE-BREAKING — not a test to update.
    #[test]
    fn canonical_bytes_field_order_is_the_wire_contract() {
        let r = mk_record(0, "recall", genesis_hash("s1", "u1"));
        let json = String::from_utf8(canonical_bytes(&r)).expect("canonical bytes are UTF-8 JSON");
        let keys: Vec<&str> = json
            .match_indices("\":")
            .filter_map(|(i, _)| json[..i].rfind('"').map(|s| &json[s + 1..i]))
            .collect();
        assert_eq!(
            keys,
            vec![
                "seq",
                "ts",
                "session_id",
                "user_id",
                "op",
                "attestation",
                "payload_summary",
                "evidence_refs",
                "outcome",
                "reported_ts",
                "source",
                "prev_hash",
                "hash",
            ],
            "OpRecord field order is the on-disk wire order"
        );
        assert!(
            json.ends_with(r#""hash":""}"#),
            "hash must be blanked in canonical bytes, got tail: {}",
            &json[json.len().saturating_sub(32)..]
        );
    }

    #[test]
    fn empty_chain_verifies() {
        assert!(verify_chain(&[], "s1", "u1").is_ok());
    }

    #[test]
    fn wrong_genesis_breaks_chain_at_zero() {
        // Record sealed against a genesis for a DIFFERENT session header.
        let r1 = mk_record(0, "recall", genesis_hash("other", "u1"));
        let err = verify_chain(&[r1], "s1", "u1").unwrap_err();
        assert_eq!(err.at_seq, 0);
        assert!(
            err.reason.contains("genesis"),
            "reason should name the genesis mismatch, got: {}",
            err.reason
        );
    }

    /// Audit Finding F: a `:`-bearing `session_id` can make `op:{sid}:` prefix
    /// scans bleed foreign records into a session's chain. Identity checking
    /// turns that into a DETECTED error rather than false tampering.
    #[test]
    fn foreign_session_record_is_detected() {
        let r1 = mk_record(0, "recall", genesis_hash("s1", "u1"));
        let foreign = mk_record_for(1, "remember", r1.hash.clone(), "s2", "u1");
        let err = verify_chain(&[r1, foreign], "s1", "u1").unwrap_err();
        assert_eq!(err.at_seq, 1);
        assert!(
            err.reason.contains("session_id"),
            "reason should name session_id, got: {}",
            err.reason
        );
    }

    #[test]
    fn foreign_user_record_is_detected() {
        let r1 = mk_record(0, "recall", genesis_hash("s1", "u1"));
        let foreign = mk_record_for(1, "remember", r1.hash.clone(), "s1", "u2");
        let err = verify_chain(&[r1, foreign], "s1", "u1").unwrap_err();
        assert_eq!(err.at_seq, 1);
        assert!(
            err.reason.contains("user_id"),
            "reason should name user_id, got: {}",
            err.reason
        );
    }

    /// `genesis_hash("ab", "c")` must not equal `genesis_hash("a", "bc")`.
    #[test]
    fn genesis_hash_is_domain_separated() {
        assert_ne!(genesis_hash("ab", "c"), genesis_hash("a", "bc"));
        assert_eq!(genesis_hash("s1", "u1").len(), 64);
        assert!(genesis_hash("s1", "u1")
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn chain_hash_is_fixed_width_hex() {
        let h = chain_hash(&genesis_hash("s1", "u1"), b"payload");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        // Deterministic.
        assert_eq!(h, chain_hash(&genesis_hash("s1", "u1"), b"payload"));
        assert_ne!(h, chain_hash(&genesis_hash("s1", "u1"), b"payload2"));
    }

    #[test]
    fn payload_summary_is_bounded_and_marks_truncation() {
        let short = "a".repeat(10);
        assert_eq!(bound_payload_summary(&short), short);

        let long = "b".repeat(PAYLOAD_SUMMARY_MAX_CHARS + 500);
        let bounded = bound_payload_summary(&long);
        assert!(
            bounded.chars().count() <= PAYLOAD_SUMMARY_MAX_CHARS,
            "bounded summary is {} chars",
            bounded.chars().count()
        );
        assert!(
            bounded.contains("truncated"),
            "truncation must be marked in-band"
        );
        assert!(bounded.contains(&(PAYLOAD_SUMMARY_MAX_CHARS + 500).to_string()));

        // Exactly at the limit is untouched.
        let exact = "c".repeat(PAYLOAD_SUMMARY_MAX_CHARS);
        assert_eq!(bound_payload_summary(&exact), exact);

        // Multi-byte input must not panic or split a char.
        let multi = "é".repeat(PAYLOAD_SUMMARY_MAX_CHARS + 10);
        let bounded_multi = bound_payload_summary(&multi);
        assert!(bounded_multi.chars().count() <= PAYLOAD_SUMMARY_MAX_CHARS);
    }

    /// Task-2 round-trip invariant: any persistence format for `OpRecord` must
    /// restore every field bit-exactly, or `verify_chain` reports false tampering.
    #[test]
    fn serde_round_trip_preserves_verification() {
        let r1 = mk_record(0, "recall", genesis_hash("s1", "u1"));
        let mut r2 = mk_record(1, "report:file_edit", r1.hash.clone());
        r2.attestation = ATTESTATION_REPORTED.to_string();
        r2.reported_ts = Some(fixed_ts(7));
        r2.source = Some("claude-code-hook".to_string());
        r2.hash = chain_hash(&r2.prev_hash, &canonical_bytes(&r2));

        let encoded = serde_json::to_vec(&[r1.clone(), r2.clone()]).expect("serialize");
        let decoded: Vec<OpRecord> = serde_json::from_slice(&encoded).expect("deserialize");
        assert_eq!(decoded, vec![r1, r2]);
        assert!(verify_chain(&decoded, "s1", "u1").is_ok());
    }

    #[test]
    fn attestation_constants_are_stable_strings() {
        assert_eq!(ATTESTATION_WITNESSED, "witnessed");
        assert_eq!(ATTESTATION_REPORTED, "reported");
    }

    #[test]
    fn chain_error_displays_seq_and_reason() {
        let r1 = mk_record(0, "recall", genesis_hash("s1", "u1"));
        let mut r2 = mk_record(1, "remember", r1.hash.clone());
        r2.outcome = "error:500".into();
        let err = verify_chain(&[r1, r2], "s1", "u1").unwrap_err();
        let text = err.to_string();
        assert!(text.contains('1'), "display must name the seq: {text}");
        assert!(!err.reason.is_empty());
    }
}
