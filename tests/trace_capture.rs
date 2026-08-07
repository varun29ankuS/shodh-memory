//! Integration tests for Task 2 of the agent-traceability slice-1 plan:
//! `CF_OPLOG` storage — append, read, integrity flag (audit
//! `docs/superpowers/audits/2026-07-30-traceability-slice1-audit.md`).
//!
//! Storage lives under `std::env::temp_dir()` via `tempfile::TempDir`, NOT
//! under any OneDrive-watched path — mirrors `tests/geo_composition.rs`'s
//! temp-dir setup (see the BM25 onedrive finding for why).
//!
//! `MemoryStorage` is constructed directly (it is `pub`, and
//! `MemoryStorage::new` is a `pub fn`) rather than through `MemorySystem`:
//! the oplog interface (`oplog_append`/`oplog_read`/...) lives on
//! `MemoryStorage` itself, so testing at that layer avoids pulling in the
//! full memory system (embedder, retriever, etc.) for a storage-only test.

use chrono::{TimeZone, Timelike, Utc};
use shodh_memory::memory::oplog::{
    self, OpRecordDraft, ATTESTATION_REPORTED, ATTESTATION_WITNESSED,
};
use shodh_memory::memory::storage::MemoryStorage;
use tempfile::TempDir;

/// Fixed epoch second so test data (and any failure output) is reproducible.
fn fixed_ts(offset_secs: i64) -> chrono::DateTime<chrono::Utc> {
    Utc.timestamp_opt(1_785_412_800 + offset_secs, 0)
        .single()
        .expect("fixed test timestamp must be valid")
}

fn draft(session_id: &str, user_id: &str, op: &str, offset_secs: i64) -> OpRecordDraft {
    OpRecordDraft {
        ts: fixed_ts(offset_secs),
        session_id: session_id.to_string(),
        user_id: user_id.to_string(),
        op: op.to_string(),
        attestation: ATTESTATION_WITNESSED.to_string(),
        payload_summary: format!("payload for {op}"),
        evidence_refs: vec![format!("mem-{offset_secs}")],
        outcome: "ok".to_string(),
        reported_ts: None,
        source: None,
    }
}

fn setup_storage() -> (MemoryStorage, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let storage = MemoryStorage::new(temp_dir.path(), None).expect("Failed to create storage");
    (storage, temp_dir)
}

#[test]
fn append_read_roundtrip_and_chain() {
    let (storage, _temp) = setup_storage();

    let r0 = storage
        .oplog_append(draft("s1", "alice", "recall", 0))
        .expect("append 0 should succeed");
    let r1 = storage
        .oplog_append(draft("s1", "alice", "remember", 1))
        .expect("append 1 should succeed");
    let r2 = storage
        .oplog_append(draft("s1", "alice", "recall", 2))
        .expect("append 2 should succeed");

    assert_eq!(r0.seq, 0);
    assert_eq!(r1.seq, 1);
    assert_eq!(r2.seq, 2);
    assert_eq!(r0.prev_hash, oplog::genesis_hash("s1", "alice"));
    assert_eq!(r1.prev_hash, r0.hash);
    assert_eq!(r2.prev_hash, r1.hash);

    let read_back = storage
        .oplog_read("s1", 0, 10)
        .expect("read should succeed");
    assert_eq!(read_back, vec![r0, r1, r2]);
    assert!(oplog::verify_chain(&read_back, "s1", "alice").is_ok());
}

#[test]
fn sessions_isolated() {
    let (storage, _temp) = setup_storage();

    // Interleave appends to two different sessions.
    let a0 = storage
        .oplog_append(draft("session-a", "alice", "recall", 0))
        .expect("append a0");
    let b0 = storage
        .oplog_append(draft("session-b", "bob", "remember", 1))
        .expect("append b0");
    let a1 = storage
        .oplog_append(draft("session-a", "alice", "remember", 2))
        .expect("append a1");
    let b1 = storage
        .oplog_append(draft("session-b", "bob", "recall", 3))
        .expect("append b1");

    let reads_a = storage
        .oplog_read("session-a", 0, 100)
        .expect("read session-a");
    let reads_b = storage
        .oplog_read("session-b", 0, 100)
        .expect("read session-b");

    assert_eq!(reads_a, vec![a0, a1]);
    assert_eq!(reads_b, vec![b0, b1]);

    // No cross-session bleed: each session's records carry only its own
    // session_id/user_id.
    assert!(reads_a
        .iter()
        .all(|r| r.session_id == "session-a" && r.user_id == "alice"));
    assert!(reads_b
        .iter()
        .all(|r| r.session_id == "session-b" && r.user_id == "bob"));

    // Each chain verifies independently against its own header.
    assert!(oplog::verify_chain(&reads_a, "session-a", "alice").is_ok());
    assert!(oplog::verify_chain(&reads_b, "session-b", "bob").is_ok());

    // Distinct session listing surfaces both, newest-first by head write time.
    let sessions = storage.oplog_sessions(10, 0).expect("oplog_sessions");
    assert_eq!(
        sessions,
        vec!["session-b".to_string(), "session-a".to_string()]
    );
}

#[test]
fn incomplete_flag_roundtrip() {
    let (storage, _temp) = setup_storage();

    let r0 = storage
        .oplog_append(draft("s1", "alice", "recall", 0))
        .expect("append should succeed");

    assert!(
        !storage.oplog_is_incomplete("s1").expect("is_incomplete"),
        "flag must be unset before marking"
    );

    storage
        .oplog_mark_incomplete("s1")
        .expect("mark_incomplete should succeed");

    assert!(
        storage.oplog_is_incomplete("s1").expect("is_incomplete"),
        "flag must be set after marking"
    );

    // Marking incomplete must NEVER touch records: re-read equals pre-mark bytes.
    let reread = storage.oplog_read("s1", 0, 10).expect("re-read");
    assert_eq!(reread, vec![r0]);
}

#[test]
fn oplog_survives_reopen() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let (r0, r1) = {
        let storage = MemoryStorage::new(temp_dir.path(), None).expect("Failed to create storage");
        let r0 = storage
            .oplog_append(draft("s1", "alice", "recall", 0))
            .expect("append 0");
        let r1 = storage
            .oplog_append(draft("s1", "alice", "remember", 1))
            .expect("append 1");
        (r0, r1)
        // storage dropped here
    };

    let reopened = MemoryStorage::new(temp_dir.path(), None).expect("Failed to reopen storage");
    let read_back = reopened.oplog_read("s1", 0, 10).expect("read after reopen");

    assert_eq!(read_back, vec![r0, r1]);
    assert!(oplog::verify_chain(&read_back, "s1", "alice").is_ok());
}

#[test]
fn invalid_session_id_is_rejected() {
    let (storage, _temp) = setup_storage();

    // A colon would bleed op:{session}: prefix scans across sessions (audit
    // Finding F) — the storage layer must reject it outright, not merely
    // rely on verify_chain's identity check as a backstop.
    let bad = draft("has:colon", "alice", "recall", 0);
    assert!(storage.oplog_append(bad).is_err());

    assert!(storage.oplog_read("has:colon", 0, 10).is_err());
    assert!(storage.oplog_mark_incomplete("has:colon").is_err());
    assert!(storage.oplog_is_incomplete("has:colon").is_err());
}

#[test]
fn invalid_user_id_is_rejected() {
    let (storage, _temp) = setup_storage();

    // A NUL byte would land verbatim in the permanent tamper-evident record
    // and break oplog.rs's DOMAIN_SEP invariant ("0x00 cannot occur in a
    // validated session_id/user_id") if it weren't rejected here.
    let null_user = draft("s1", "bad\0user", "recall", 0);
    assert!(storage.oplog_append(null_user).is_err());

    // Over validation::MAX_USER_ID_LENGTH (128 chars).
    let long_user_id = "a".repeat(300);
    let long_user = draft("s1", &long_user_id, "recall", 1);
    assert!(storage.oplog_append(long_user).is_err());

    // Nothing written by either rejected append: no records, and the
    // session has no head (absent from the session listing).
    assert!(storage.oplog_read("s1", 0, 10).expect("read").is_empty());
    assert!(!storage
        .oplog_sessions(10, 0)
        .expect("oplog_sessions")
        .contains(&"s1".to_string()));
}

#[test]
fn nanosecond_precision_round_trips_bit_exact() {
    let (storage, _temp) = setup_storage();

    let ts_with_nanos = Utc
        .timestamp_opt(1_700_000_000, 123_456_789)
        .single()
        .expect("valid timestamp with nanoseconds");
    let reported_ts_with_nanos = Utc
        .timestamp_opt(1_700_000_100, 987_654_321)
        .single()
        .expect("valid reported timestamp with nanoseconds");

    let mut d = draft("s1", "alice", "report:file_edit", 0);
    d.ts = ts_with_nanos;
    d.attestation = ATTESTATION_REPORTED.to_string();
    d.reported_ts = Some(reported_ts_with_nanos);
    d.source = Some("claude-code-hook".to_string());

    let sealed = storage
        .oplog_append(d)
        .expect("append with nanosecond timestamps");
    let read_back = storage
        .oplog_read("s1", 0, 10)
        .expect("read back")
        .into_iter()
        .next()
        .expect("exactly one record");

    // Bit-exact: any lossy persistence format for ts/reported_ts would make
    // verify_chain report FALSE TAMPERING on untouched data — this is the
    // "Persistence round-trip invariant" from oplog.rs's module docs.
    assert_eq!(read_back.ts, ts_with_nanos);
    assert_eq!(read_back.ts.timestamp_subsec_nanos(), 123_456_789);
    assert_eq!(read_back.reported_ts, Some(reported_ts_with_nanos));
    assert_eq!(
        read_back
            .reported_ts
            .expect("reported_ts must round-trip")
            .timestamp_subsec_nanos(),
        987_654_321
    );
    assert_eq!(read_back, sealed);
    assert!(oplog::verify_chain(&[read_back], "s1", "alice").is_ok());
}

#[test]
fn reported_attestation_round_trips() {
    let (storage, _temp) = setup_storage();

    let mut d = draft("s1", "alice", "report:file_edit", 0);
    d.attestation = ATTESTATION_REPORTED.to_string();
    d.reported_ts = Some(fixed_ts(7));
    d.source = Some("claude-code-hook".to_string());

    let sealed = storage.oplog_append(d).expect("append reported record");
    let read_back = storage.oplog_read("s1", 0, 10).expect("read back");

    assert_eq!(read_back, vec![sealed.clone()]);
    assert_eq!(sealed.attestation, ATTESTATION_REPORTED);
    assert_eq!(sealed.source.as_deref(), Some("claude-code-hook"));
    assert!(oplog::verify_chain(&read_back, "s1", "alice").is_ok());
}

// ═══════════════════════════════════════════════════════════════════════
// Task 3: end-to-end witnessed-op capture through the protected router
// (middleware inserts OpTrace → handlers enrich → middleware appends).
// build_protected_routes is used directly WITHOUT the auth layer: capture
// mounts inside that function (audit amendment 9), so this exercises the
// exact production wiring while keeping the test self-contained.
// ═══════════════════════════════════════════════════════════════════════

mod capture_e2e {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::json;
    use shodh_memory::config::ServerConfig;
    use shodh_memory::handlers::{build_protected_routes, MultiUserMemoryManager};
    use shodh_memory::memory::oplog::{self, ATTESTATION_WITNESSED};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tower::ServiceExt;

    fn post_json(path: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builds")
    }

    #[tokio::test]
    async fn remember_then_recall_are_witnessed_with_evidence() {
        let dir = TempDir::new().expect("temp dir");
        let cfg = ServerConfig {
            storage_path: dir.path().to_path_buf(),
            ..ServerConfig::default()
        };
        let mgr = Arc::new(
            MultiUserMemoryManager::new(dir.path().to_path_buf(), cfg)
                .expect("create MultiUserMemoryManager"),
        );
        let app = build_protected_routes(mgr.clone());
        let user = "trace-e2e";

        // 1) remember (no session_id field exists on RememberRequest → the
        //    middleware's session-store fallback covers it).
        let resp = app
            .clone()
            .oneshot(post_json(
                "/api/remember",
                json!({"user_id": user, "content": "trace e2e: the harbor crane moved at dawn"}),
            ))
            .await
            .expect("remember request");
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let remember_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let stored_id = remember_json["id"]
            .as_str()
            .expect("id in response")
            .to_string();

        // 2) recall WITH an explicit session id. /api/recall parses session_id
        //    as a UUID (src/handlers/recall.rs:475) and 400s otherwise, so the
        //    fixture uses a fixed UUID literal — fixed, not random, to keep the
        //    test deterministic. No such session exists in the session store,
        //    so get_session_time_range returns None and recall stays in its
        //    default mode; the id still reaches the oplog verbatim.
        let recall_session = "3f2b7c14-9d5e-4a61-8b02-6e7d1c4f9a83";
        let resp = app
            .clone()
            .oneshot(post_json(
                "/api/recall",
                json!({"user_id": user, "query": "harbor crane", "session_id": recall_session}),
            ))
            .await
            .expect("recall request");
        assert_eq!(resp.status(), StatusCode::OK);

        // 3) Inspect the oplog via the cache-only accessor (never creates).
        let system = mgr
            .cached_user_memory(user)
            .expect("user must be cached after its own ops");
        let guard = system.read();
        let storage = guard.storage();

        // recall's record: explicit session, witnessed, evidence ⊆ stored ids.
        let recall_records = storage
            .oplog_read(recall_session, 0, 10)
            .expect("read recall session");
        assert_eq!(
            recall_records.len(),
            1,
            "exactly one op in the recall session"
        );
        let r = &recall_records[0];
        assert_eq!(r.op, "post:recall");
        assert_eq!(r.attestation, ATTESTATION_WITNESSED);
        assert_eq!(r.user_id, user);
        assert_eq!(r.outcome, "ok");
        assert!(
            !r.evidence_refs.is_empty(),
            "recall must carry surfaced memory ids as evidence"
        );
        assert!(
            r.evidence_refs.contains(&stored_id),
            "recall evidence {:?} must include the stored id {stored_id}",
            r.evidence_refs
        );
        assert!(oplog::verify_chain(&recall_records, recall_session, user).is_ok());

        // remember's record: lives in the session-store fallback session.
        let sessions = storage.oplog_sessions(10, 0).expect("session list");
        let fallback: Vec<_> = sessions.iter().filter(|s| *s != recall_session).collect();
        assert_eq!(
            fallback.len(),
            1,
            "exactly one fallback session for the remember op, got {sessions:?}"
        );
        let remember_records = storage
            .oplog_read(fallback[0], 0, 10)
            .expect("read fallback");
        assert_eq!(remember_records.len(), 1);
        let m = &remember_records[0];
        assert_eq!(m.op, "post:remember");
        assert_eq!(m.attestation, ATTESTATION_WITNESSED);
        assert_eq!(m.evidence_refs, vec![stored_id.clone()]);
        assert!(oplog::verify_chain(&remember_records, fallback[0], user).is_ok());
    }
}
