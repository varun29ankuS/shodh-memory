# Agent Traceability Slice 1 — Capture + Oplog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. All executor subagents run `model: opus`.

**Goal:** Every agent operation against shodh is captured into an append-only, hash-chained, lifecycle-excluded operation log, with a reported-event ingest endpoint for hook-sourced actions and a paginated trace read API.

**Architecture:** A tower/axum middleware on the protected router is the single capture choke point: it opens an `OpTrace` request extension, handlers enrich it with evidence (memory ids), and the middleware finalizes exactly one oplog record per request into a dedicated RocksDB column family (`CF_OPLOG`), chained by SHA-256. Reported events (Claude Code hooks) enter the same log via `POST /api/trace/report` with `attestation: reported`. Spec: `docs/superpowers/specs/2026-07-30-agent-traceability-design.md` §3.1–§3.2, slice 1 of §4.

**Tech Stack:** Rust (axum middleware, rocksdb CF, sha2), existing auth/validation stack. No new external deps except `sha2` if not already present (check Cargo.toml — `sha2` is likely already a dependency via other crates; if absent, add workspace-pinned).

## Global Constraints

- cargo check / clippy / test ONLY (never build/run); all cargo `-j 2`; test runs `-- --test-threads=2`; targeted runs, never the full suite locally (honest CI covers it).
- Capture failure policy (spec §3.1 verbatim): "a failed log append must never fail the underlying operation; it must also never be silent — the failure increments a loud metric and permanently marks that session's trace `integrity: incomplete`".
- Log CF has NO update or delete code paths — append and read only (spec §3.2).
- Attestation values: exactly `witnessed` | `reported` — never blended (spec §2).
- Overhead gate: capture append median < 100µs in the unit bench; recall-latency overhead < 2% observed on the honest CI run (report, don't hand-tune).
- LongMemEval smoke bit-neutral. `cargo fmt --all -- --check` clean before every push (two branches failed CI on this — do not repeat).
- No attribution footers. PR workflow; branch `feat/traceability-capture-oplog` off origin/main.
- Line anchors below verified 2026-07-30 but drift — re-locate by quoted code, not numbers.

## File Structure

- Create `src/memory/oplog.rs` — record type, hash chain, append/read, integrity marking. Self-contained; storage-agnostic core functions + a thin RocksDB adapter trait implemented by `MemoryStorage`.
- Modify `src/memory/storage.rs` — `CF_OPLOG` descriptor + accessor + append/read methods (~1176 descriptor vec; follow `CF_INDEX` pattern).
- Create `src/handlers/trace.rs` — capture middleware, `OpTrace` extension type, `POST /api/trace/report`, `GET /api/trace/{session_id}`.
- Modify `src/handlers/router.rs` — mount middleware on protected routes (~:100 `build_protected_routes`), register the two routes.
- Modify `src/handlers/recall.rs` + `src/handlers/remember.rs` — evidence enrichment (push returned/stored memory ids into `OpTrace`).
- Test `tests/trace_capture.rs` — end-to-end capture/report/read/chain tests.

---

### Task 0: Substrate audit (no code)

**Files:** Create: `docs/superpowers/audits/2026-07-30-traceability-slice1-audit.md`

**Interfaces:** Produces the audit doc consumed by Tasks 1–5; findings amend them before execution.

- [ ] **Step 1:** Lifecycle-writer enumeration (spec §3.2 requirement): grep every site that writes/deletes in RocksDB — `delete_cf|put_cf|write_opt|WriteBatch|compact|drop_cf` across `src/` — and produce a table: writer → CFs touched. Prove by enumeration that adding `CF_OPLOG` is untouched by: decay sweeps, `mark_forgotten_*`, `remove_from_indices`, index rebuilds, canonicalization/consolidation, backup/restore (`backup.rs` — CHECK: does backup copy whole DB incl. new CFs? It should — oplog must be backed up; verify restore doesn't rewrite records), `repair_index`, migration.rs.
- [ ] **Step 2:** Verify `open_or_repair_cf` (storage.rs ~:1174) opens an EXISTING database when a NEW CF descriptor is added (rocksdb needs `create_missing_column_families(true)` — find where opts set it; if absent, note the exact change needed so existing deployments upgrade cleanly). This is the data-safety-critical check of the slice.
- [ ] **Step 3:** Router/auth audit: confirm `build_protected_routes` (router.rs ~:100) is the complete surface of agent ops (list every route; flag any op route registered in `build_public_routes` that should be captured — e.g., webhooks are NOT agent ops, `/api/context/*` is not either; the capture set = protected `/api/*` op routes). Confirm how auth middleware is layered by the caller (grep `build_protected_routes(` call sites) so the capture layer slots INSIDE auth (only authenticated ops are traced).
- [ ] **Step 4:** Session identity audit: which request types carry `session_id` today (grep `session_id` in handlers/types.rs); document the fallback rule from the plan (absent → `adhoc-{user_id}-{UTC date}`) and confirm no collision with real session-id formats.
- [ ] **Step 5:** `sha2` presence in Cargo.toml/lock; metrics facility for the loud-failure counter (grep `metrics::` in handlers — reuse the existing pattern, name the exact macro/fn).
- [ ] **Step 6:** Commit audit doc on branch `feat/traceability-capture-oplog` (created off origin/main): `docs: traceability slice-1 substrate audit`.

### Task 1: Oplog core — record, hash chain, integrity

**Files:** Create: `src/memory/oplog.rs`; register module in `src/memory/mod.rs` (`pub mod oplog;` near the other module declarations at the top).

**Interfaces (Produces — later tasks rely on these exact signatures):**
```rust
pub const ATTESTATION_WITNESSED: &str = "witnessed";
pub const ATTESTATION_REPORTED: &str = "reported";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpRecord {
    pub seq: u64,
    pub ts: chrono::DateTime<chrono::Utc>,
    pub session_id: String,
    pub user_id: String,
    pub op: String,                    // e.g. "recall", "remember", "report:file_edit"
    pub attestation: String,           // ATTESTATION_* constant
    pub payload_summary: String,       // bounded: truncate to 2048 chars, mark truncation
    pub evidence_refs: Vec<String>,    // memory ids touched (returned/stored)
    pub outcome: String,               // "ok" | "error:<status>"
    pub reported_ts: Option<chrono::DateTime<chrono::Utc>>, // reporter's own clock, reported-only
    pub source: Option<String>,        // reporter identifier, reported-only
    pub prev_hash: String,             // hex SHA-256 of predecessor's canonical bytes; genesis: hash of session header
    pub hash: String,                  // hex SHA-256 of THIS record's canonical bytes (with hash field empty during computation)
}

pub fn canonical_bytes(r: &OpRecord) -> Vec<u8>;          // serde_json with hash field blanked, stable field order
pub fn chain_hash(prev_hash: &str, canonical: &[u8]) -> String;
pub fn genesis_hash(session_id: &str, user_id: &str) -> String;
pub fn verify_chain(records: &[OpRecord], session_id: &str, user_id: &str) -> Result<(), ChainError>; // ChainError { at_seq, reason }
```

- [ ] **Step 1: Write failing unit tests** in `#[cfg(test)]` inside oplog.rs:

```rust
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
    r2.payload_summary = "tampered".into();   // content changed, hash not recomputed
    let err = verify_chain(&[r1, r2], "s1", "u1").unwrap_err();
    assert_eq!(err.at_seq, 1);
}

#[test]
fn reorder_breaks_chain() { /* swap two valid records; verify fails at the swap point */ }

#[test]
fn canonical_bytes_stable_and_hash_field_blanked() {
    let r = mk_record(0, "recall", genesis_hash("s", "u"));
    let b1 = canonical_bytes(&r);
    let mut r2 = r.clone(); r2.hash = "different".into();
    assert_eq!(b1, canonical_bytes(&r2), "hash field must not affect canonical bytes");
}
```
(`mk_record` helper builds a record then sets `hash = chain_hash(&prev, &canonical_bytes(...))` — write it in the test module; timestamps fixed constants, never `Utc::now()`, for determinism.)

- [ ] **Step 2:** Run red: `cargo test -j 2 --lib oplog -- --test-threads=2` → compile error (module missing).
- [ ] **Step 3:** Implement `oplog.rs` per the interface block: serde_json canonicalization (struct field order is declaration order in serde_json — document that reordering struct fields is a WIRE-BREAKING change, same postcard lesson as RelationType), sha2 for hashes, `verify_chain` walking seq/prev_hash/recomputed-hash with precise `ChainError { at_seq, reason: String }`.
- [ ] **Step 4:** Run green; `cargo fmt --all`; commit `feat(trace): oplog record + hash chain core`.

### Task 2: Storage — CF_OPLOG, append, read, integrity flag

**Files:** Modify: `src/memory/storage.rs` (descriptor vec ~:1176; new methods near `search_by_location`'s region or the misc-accessor area; constant near `CF_INDEX` declaration — grep `CF_INDEX =`).

**Interfaces (Produces):**
```rust
pub const CF_OPLOG: &str = "oplog";
impl MemoryStorage {
    /// Appends with chain linkage; returns the stored record (with seq/hashes filled).
    /// Key: "op:{session_id}:{seq:016}" — u64 zero-padded for lexicographic order.
    /// Session head cached: "head:{session_id}" -> (last_seq, last_hash) stored in CF_OPLOG too
    /// (head entries are the ONLY non-record keys; prefix "head:" vs "op:").
    pub fn oplog_append(&self, partial: OpRecordDraft) -> Result<OpRecord>;
    pub fn oplog_read(&self, session_id: &str, from_seq: u64, limit: usize) -> Result<Vec<OpRecord>>;
    pub fn oplog_mark_incomplete(&self, session_id: &str) -> Result<()>;  // sets flag key "incomplete:{session_id}"; NEVER touches records
    pub fn oplog_is_incomplete(&self, session_id: &str) -> Result<bool>;
    pub fn oplog_sessions(&self, user_id: Option<&str>, limit: usize, offset: usize) -> Result<Vec<String>>; // distinct session ids, newest-first by head write time
}
pub struct OpRecordDraft { /* all OpRecord fields except seq, prev_hash, hash */ }
```
Append is atomic per session: read head → build record with `seq = head.last_seq + 1` (or 0 + genesis hash) → single WriteBatch writes record + updated head. Concurrency: take the existing storage write path's locking approach — audit Task 0 documents whether `MemoryStorage` methods are already externally serialized (e.g., behind a lock in `MultiUserMemoryManager`); if not, guard head-read→write with a per-storage `Mutex<()>` for the append path only (document: capture is low-frequency relative to reads; a single mutex is acceptable for v1 and measured in Task 5's bench).

- [ ] **Step 1: Failing tests** in `tests/trace_capture.rs` (integration; construct storage the way existing tests do — read `tests/` for the `MemoryStorage`/`MemorySystem` construction pattern, mirror `tests/geo_composition.rs` temp-dir setup):

```rust
#[test]
fn append_read_roundtrip_and_chain() {
    // 3 appends to one session; oplog_read returns them in order; verify_chain passes;
    // seq = 0,1,2; r0.prev_hash == genesis_hash(session, user).
}
#[test]
fn sessions_isolated() { /* two sessions interleaved appends; reads don't bleed; chains verify independently */ }
#[test]
fn incomplete_flag_roundtrip() { /* mark → is_incomplete true; records untouched (re-read equals pre-mark bytes) */ }
#[test]
fn oplog_survives_reopen() { /* append, drop storage, reopen same path, read + verify_chain still pass */ }
```
- [ ] **Step 2:** Run red (methods missing). **Step 3:** Implement (CF descriptor added to the vec at ~:1176 with the CF_INDEX-style lighter options; `create_missing_column_families` confirmed per audit). **Step 4:** Green + targeted re-run of one existing storage-touching test file to prove no regression: `cargo test -j 2 --test geo_composition -- --test-threads=2`. **Step 5:** fmt; commit `feat(trace): CF_OPLOG append-only storage with per-session chains`.

### Task 3: Capture middleware + evidence enrichment

**Files:** Create: `src/handlers/trace.rs` (middleware + `OpTrace`); Modify: `src/handlers/router.rs` (`build_protected_routes` ~:100), `src/handlers/recall.rs` (recall + paginated_recall success paths), `src/handlers/remember.rs` (remember + batch success paths).

**Interfaces (Produces):**
```rust
/// Request extension; handlers enrich it. Cheap clone-free interior mutability.
#[derive(Clone, Default)]
pub struct OpTrace(pub Arc<Mutex<OpTraceInner>>);
pub struct OpTraceInner {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub evidence_refs: Vec<String>,
    pub payload_summary: Option<String>,
}
pub async fn capture_middleware(State(state): State<AppState>, req: Request, next: Next) -> Response;
```
Middleware behavior: insert `OpTrace::default()` extension → run handler → derive `op` from method+path (strip `/api/`, e.g. `POST /api/recall` → `recall`), `outcome` from status → build `OpRecordDraft { attestation: ATTESTATION_WITNESSED, .. }` with session fallback `adhoc-{user_id}-{YYYY-MM-DD}` → `oplog_append`. On append error: `metrics` loud counter (exact call per audit Task 0) + `oplog_mark_incomplete(session)` best-effort + response returned unchanged. Capture set: protected routes only (mounted via `.layer(axum::middleware::from_fn_with_state(state.clone(), trace::capture_middleware))` on the router returned by `build_protected_routes` — inside auth per audit Step 3 finding). Exclusion: `GET /api/trace/*` reads are NOT captured (avoid self-amplifying logs) — early-return in the middleware on path prefix `/api/trace`.

Handler enrichment (exact insertions):
- `recall.rs` recall: after results are final (grep `RecallResponse` construction), `if let Some(trace) = req_extensions_optiontrace { trace.0.lock().push evidence ids }` — the plan's implementer reads how extensions are accessible in these axum handlers (`Extension<OpTrace>` extractor added to the handler signature; axum makes the extension available because the middleware inserted it) and pushes `memories.iter().map(|m| m.id...)`.
- `remember.rs`: push the created memory id.
- Both also set `session_id`/`user_id` from the validated request fields (the middleware cannot parse bodies — handlers own body-derived identity; middleware falls back to `unknown` user only if a handler never set it, and such records are still appended, flagged in payload_summary `identity:unset`).

- [ ] **Step 1: Failing integration tests** (extend `tests/trace_capture.rs`): spin the axum app the way existing handler tests do (read `tests/handler_tests.rs` construction — it exists per W0's report reference to `handler_tests.rs:1402`); POST /api/remember then /api/recall with a session_id; then GET /api/trace/{session} (Task 4 endpoint — for THIS task assert via direct `storage.oplog_read`): exactly 2 records, ops `remember`/`recall`, both `witnessed`, recall's `evidence_refs` non-empty and ⊆ stored ids, chain verifies.
- [ ] **Step 2:** red. **Step 3:** implement middleware + router mount + enrichment. **Step 4:** green + `cargo test -j 2 --test handler_tests -- --test-threads=2` (no handler regressions). **Step 5:** fmt; commit `feat(trace): witnessed-op capture middleware with handler evidence enrichment`.

### Task 4: Reported-event ingest + trace read API + hooks wiring

**Files:** Modify: `src/handlers/trace.rs` (two handlers), `src/handlers/router.rs` (routes), `hooks/claude-code-ingest.sh` + `hooks/stop.sh` + `hooks/user-prompt.sh` (add best-effort trace-report POST alongside existing behavior); Test: `tests/trace_capture.rs`.

**Interfaces (Produces):**
```rust
// POST /api/trace/report  (protected route, captured=NO — it IS capture)
#[derive(Deserialize)] pub struct TraceReportRequest {
    pub user_id: String, pub session_id: String,
    pub op: String,                     // free-form, prefixed by server as "report:{op}"
    pub payload_summary: String,        // server truncates to 2048
    pub reported_ts: Option<String>,    // RFC3339; parsed, else null
    pub source: String,                 // e.g. "claude-code-hook/stop"
}
// -> 200 {"seq": n, "hash": "..."}  ; attestation forced to ATTESTATION_REPORTED server-side.
// GET /api/trace/{session_id}?from_seq=0&limit=200  (protected)
// -> {"records":[OpRecord...], "integrity": "ok"|"incomplete", "chain_verified": bool, "next_seq": n|null}
// GET /api/trace?user_id=&limit=&offset=   -> {"sessions":[...]}  (session list)
```
`chain_verified` computed over the returned page ONLY when `from_seq==0` and the page is complete; otherwise `null` (never claim verification over a partial window — spec §5 export-honesty rule applied to the API).

- [ ] **Step 1: failing tests:** report → record present with `attestation=="reported"`, op `report:file_edit`, `reported` never `witnessed`; GET trace returns both witnessed+reported interleaved by seq with correct `chain_verified: true`; tampering a record via raw CF write in the test → `chain_verified: false` (this test writes raw bytes deliberately to simulate tamper — the ONLY place raw CF writes are acceptable, in tests).
- [ ] **Step 2:** red. **Step 3:** implement handlers + routes (`/api/trace/report` post, `/api/trace/:session_id` get, `/api/trace` get — protected). **Step 4:** hooks: in each of the three shell hooks, after the existing ingest call, add a fire-and-forget `curl -s -X POST "$API_URL/api/trace/report" ... || true` with `source` naming the hook; MUST NOT change existing behavior or exit codes (`|| true`, no `set -e` interference — note `claude-code-ingest.sh` uses `set -e`: place the call with explicit `|| true`). **Step 5:** green; fmt; commit `feat(trace): reported-event ingest, trace read API, hook wiring`.

### Task 5: Overhead bench + capture-completeness sweep

**Files:** Test: `tests/trace_capture.rs` additions; Create: `benches/` entry ONLY if a bench harness already exists (audit Task 0 checks — if none, implement as an ignored-by-default test `#[ignore]` run explicitly, never in CI timing-sensitive paths).

- [ ] **Step 1:** Capture-completeness: a test iterating every captured protected route with a minimal valid request (read router.rs's protected list from audit Step 3; for routes needing complex payloads use their minimal valid body from existing handler tests) asserting records == requests, one per op. This test is the moving-parts contract: a future route added without capture shows up as a count mismatch. If minimal-body coverage of ALL routes is impractical, cover ≥ the top 10 agent ops (remember, recall, paginated recall, batch, forget, todos CRUD, context) and assert the middleware's mounting point covers the rest by construction (comment linking to router.rs mount).
- [ ] **Step 2:** Append-cost micro-measure as `#[ignore]`d test: 1,000 appends to one session on temp storage, assert median < 100µs and p99 < 1ms, print distribution (informational; run locally via `cargo test -j 2 --test trace_capture -- --ignored --test-threads=1`, numbers into the PR body — the <2% recall-overhead observation comes from comparing honest-CI durations pre/post merge, reported not gated in v1).
- [ ] **Step 3:** fmt; commit `test(trace): capture completeness sweep + append-cost measurement`.
- [ ] **Step 4:** Push branch; PR with body containing: spec link, audit doc link, bench numbers, the §3.2 lifecycle-exclusion audit table, and the explicit statement that CI's honest run is the final gate.

## Self-review (done at write time)

- **Spec coverage:** §3.1 capture/choke/enrichment/failure-policy → Tasks 3, 2; §3.2 log/CF/chain/lifecycle-exclusion → Tasks 0, 1, 2; reported-ingest + hooks → Task 4; trace read API (slice-1 deliverable) → Task 4; gates (completeness, overhead, LongMemEval-neutral) → Task 5 + CI. Graph index/pane/time-travel/export = slices 2–4, out of scope here.
- **Placeholder scan:** all steps carry code or exact grep-anchored instructions; the two deliberate audit-dependencies (create_missing_column_families, metrics macro name) are pinned to audit outputs.
- **Type consistency:** `OpRecord`/`OpRecordDraft`/`OpTrace` names and the `oplog_*` method set are used identically across Tasks 1–5; attestation constants referenced everywhere, never string literals.
