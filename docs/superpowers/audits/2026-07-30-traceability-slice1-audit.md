# Traceability Slice-1 Substrate Audit (Task 0)

**Date:** 2026-07-30
**Branch:** `feat/traceability-capture-oplog` (off `origin/main` @ `f4164612`)
**Spec:** `docs/superpowers/specs/2026-07-30-agent-traceability-design.md` §3.1–§3.2
**Plan:** `docs/superpowers/plans/2026-07-30-traceability-slice1-capture-oplog.md`
**Method:** Pure source read. No production code changed, no cargo commands run. Every claim below carries a `file:line` anchor. Exactly one claim is derived from library contract rather than code-read; it is labelled as such in Step 2.

## Verdict summary

| Step | Subject | Verdict |
|---|---|---|
| 1 | Lifecycle-writer enumeration | **NEEDS-CHANGE** — no lifecycle writer *touches* a new CF, but two out-of-band paths *destroy* oplog data (`forget_user`, backup restore). Spec §3.2's "no delete paths exist" is false as written. |
| 2 | `open_or_repair_cf` / `create_missing_column_families` | **NEEDS-CHANGE** — the adding opener is SAFE (`storage.rs:1129`); a *different* opener (`migration.rs:329`) breaks the moment the CF exists. Reframed below. |
| 3 | Router / auth layering | **NEEDS-CHANGE** — two transports reach the protected routes; the plan's mount point covers only one. Exclusion set must be larger than `/api/trace/*`. |
| 4 | Session identity | **NEEDS-CHANGE** — `session_id` is unvalidated free-form (breaks the key scheme), and neither hooks nor MCP `remember` send one. The plan's `adhoc-` fallback should be replaced with the existing `SessionStore`. |
| 5 | `sha2` + metrics facility | **SAFE** — `sha2 = "0.10"` and `hex = "0.4"` already direct deps; metrics pattern named exactly below (with one gotcha: declare *and* register). |
| 6 | Commit audit doc | **SAFE** — done; see end of doc. |

**Amendments required to Tasks 1–5: 18** (numbered list at end — 2 for Task 1, 6 for Task 2, 5 for Task 3, 1 for Task 4, 4 for Task 5). Two of them (6 and 18) resolve constraint conflicts *inside the plan itself*: forced-fsync durability vs. the <100 µs append gate, and `records == requests` vs. dropping identity-less records.

---

## Step 1 — Lifecycle-writer enumeration

### 1.1 The DB topology (needed to read the table)

There are **three** distinct RocksDB databases, not one:

| DB | Path | CFs today | Opened at |
|---|---|---|---|
| Per-user memory ("storage DB") | `{base}/{user_id}/storage` | `default`, `memory_index` | `storage.rs:1174` |
| Per-user graph | `{base}/{user_id}/graph/graph` | 11 CFs (`graph_memory.rs:23-37`) | `graph_memory.rs:2366` |
| Shared multi-user | `{base}/shared` | `todos`, `projects`, `todo_index`, `prospective`, `prospective_index`, `files`, `file_index`, `feedback`, `audit` | `state.rs:920` |

`CF_OPLOG` belongs in the **per-user storage DB** (`storage.rs:1080` declares `CF_INDEX`; the new constant sits beside it). This is load-bearing for three later findings: the oplog is automatically per-user isolated (no cross-user key collision is possible), it is automatically inside the per-user backup, and it is automatically inside the per-user delete blast radius.

Note the storage DB's `default` CF is a **shared keyspace** partitioned by key prefix — memories (bare UUID keys), `facts:`, `facts_by_entity:`, `facts_by_type:`, `facts_embedding:`, `temporal_facts:`, `temporal_by_*:`, `lineage:*`, `learning:*`, `vmapping:`, `interference*:`, `geo:`, `stats:`, `_watermark:` (`migration.rs:284-316` enumerates them). Every "sweep" style writer below operates by **prefix iteration on `default`**, which is why a new CF is structurally invisible to them.

### 1.2 Writer → CF table (per-user storage DB)

All sites found by `put_cf|delete_cf|delete_range|write_opt|WriteBatch|drop_cf|compact_range|.delete(|.put(|db.write(` across `src/`, mapped to enclosing function.

| Writer (file:line) | CFs written | Mechanism | Touches a new CF? |
|---|---|---|---|
| `storage.rs:1421 store_inner` | `default` | `put_opt` on bare id key | No |
| `storage.rs:1516 update_indices` | `memory_index` | 15× `batch.put_cf(idx, …)` + `write_opt` | No — explicit `idx` handle |
| `storage.rs:1663 get_by_content_hash` | `memory_index` | `delete_cf(idx, hash_key)` (stale-hash repair) | No |
| `storage.rs:1716 migrate_memory_format` | `default` | `put_opt` | No |
| `storage.rs:1785 persist_access_updates` | `default` + `memory_index` | `batch.put`, `batch.delete_cf(idx,…)` | No |
| `storage.rs:1820 delete` | `default` | `delete_opt` (record + `vmapping:`) | No |
| `storage.rs:1841 remove_from_indices` | `memory_index` | 15× `batch.delete_cf(idx, …)` | **No** — enumerated key list, never a scan |
| `storage.rs:2702 mark_forgotten_by_age` | `default` (+ `memory_index` geo key) | iterate `default`, `batch.put`, one `delete_cf(idx, geo_key)` | No |
| `storage.rs:2756 mark_forgotten_by_importance` | same as above | same | No |
| `storage.rs:2803 remove_matching` | via `delete()` | — | No |
| `storage.rs:2931 increment_retrieval_count` | `default` | `put(RETRIEVAL_KEY)` | No |
| `storage.rs:2949 cleanup_corrupted` | `default` | iterate `default`, `delete_opt` | No |
| `storage.rs:3027 migrate_legacy` | `default` | iterate `default`, `put_opt` | No |
| `storage.rs:3463 store_with_multimodal_vectors` | `default` | `WriteBatch` puts | No |
| `storage.rs:3559 delete_vector_mapping` / `3577 update_modality_vectors` / `3601 delete_with_vectors` | `default` | `vmapping:` keys | No |
| `storage.rs:3712 save_interference_records` / `3792 delete_interference_records` / `3805 save_interference_event_count` / `3850 clear_all_interference_records` | `default` | `interference*:` prefix | No |
| `storage.rs:3835 set_fact_watermark` | `default` | `_watermark:` key | No |
| `storage.rs:1262 migrate_from_separate_dbs` | `default` + `memory_index` | one-time legacy import | No |
| `memory/facts.rs:52/103/125/472/492` (`FactStore`) | `default` | `facts*:` prefix | No |
| `memory/temporal_facts.rs:124` | `default` | `temporal_*:` prefix | No |
| `memory/lineage.rs:384/414/507` | `default` | `lineage:*` prefix | No |
| `memory/learning_history.rs:139 record`, `:712 prune_old_events` | `default` | `learning*:` prefix | No |
| `memory/mod.rs:5608 forget`, `:6691 forget_by_pattern`, `:6785 forget_by_tags`, `:6865 forget_by_date_range`, `:6960 forget_by_type` | via `storage.delete()` | — | No |
| `memory/mod.rs:7054 forget_all` | `default` | `db.prefix_iterator("facts:"…, "temporal_facts:")` + `batch.delete(&key)` (**bare `delete`, i.e. default CF only**, `mod.rs:7139-7167`) | **No** |
| `memory/mod.rs:10062 decay_facts_for_all_users` | `default` (via `fact_store.delete`) | — | No |
| `memory/hybrid_search.rs:629 remove_memory` | none (tantivy BM25) | — | No |

Shared DB writers (`state.rs:480 rotate_user_audit_logs`, `:1028 migrate_audit_db`, `:1103 log_event`, `:1423 delete_by_prefix`, `:1447 purge_user_from_shared_db`) all name their CF via `cf_handle("todos"|"audit"|…)` on `self.shared_db` — a **different database**; they cannot reach the storage DB at all.

Graph writers (`graph_memory.rs`, 73 sites) operate on the graph DB — again a different database. Canonicalization and consolidation (`handlers/graph.rs canonicalize_user_graph`, `handlers/consolidation.rs consolidate_memories`) run through those same typed stores.

### 1.3 Global negative results (the strongest evidence)

- **No `drop_cf` anywhere in `src/`.** No `DB::destroy` anywhere in `src/`. Verified by repo-wide grep; the only `cf_handle(` sites are the 18 listed at `state.rs:470/1022/1035/1124/1454/1463/1495/1522/1525/1545/1553`, `storage.rs:1110/1318/3130`, `migration.rs:475/566/675`, `mif.rs:626` — every one names a specific CF string. No code enumerates "all CFs" via `DB::list_cf` and acts on them.
- **No `compact_range` / `compact_range_cf` and no compaction filter is configured** anywhere. `storage.rs:1126-1170` sets only compression, WAL, write-buffer, level and block-cache options — there is no `set_compaction_filter*` call, so the "compaction filters" writer class named in spec §3.2 does not exist in this codebase. An append-only CF therefore cannot be silently GC'd.
- **No `delete_range` / `delete_range_cf` anywhere.** Every delete is key-enumerated or prefix-scanned.

**Verdict on the plan's named list:** `mark_forgotten_*`, `remove_from_indices`, canonicalization/consolidation, and `migration.rs` record rewriting all appear in the table above and **none writes to a CF it does not name explicitly.** For the remaining two named classes the proof is the enumeration's negative: `src/decay.rs` and `src/handlers/consolidation.rs` (which owns `rebuild_index`, `repair_vector_index`, `cleanup_corrupted`, `migrate_legacy`, `consolidate_memories`) contain **zero direct RocksDB write sites** — neither file appears among the 17 files with ≥1 write site in the repo-wide grep described in §1.2. Decay sweeps and index repair/rebuild therefore reach storage only through the enumerated `MemoryStorage` methods above, every one of which names its CF. Adding `CF_OPLOG` is invisible to all of them. **SAFE.**

### 1.4 Backup — SAFE, and better than the plan hoped

`create_backup` (`backup.rs:90-101`) uses the RocksDB **BackupEngine** against the whole `&DB` handle: `backup_engine.create_new_backup(db)` at `backup.rs:101`. BackupEngine operates at the SST/WAL file level for the entire database, so **all column families are captured, including any CF added later — no code change needed for the oplog to be backed up.**

The `db` handle passed in is the per-user storage DB: `state.rs:2519-2521` → `self.get_user_memory(name)` → `memory.get_db()` (`memory/mod.rs:7228`) → `create_comprehensive_backup_with_graph(&db, …)` at `state.rs:2540`. Secondary stores (`shared`) and the graph DB use the `Checkpoint` API (`backup.rs:189`, `:216`) — also whole-DB, also CF-complete.

Incidental resolution: `state.rs:2512` (`let db_path = path.join("storage")`) is only an existence check to decide whether a directory is a user; it does not open the DB. Not a second hardcoded-CF-list site.

### 1.5 Restore — NEEDS-CHANGE (Finding E)

`restore_backup` (`backup.rs:304-347`) calls `backup_engine.restore_from_backup(restore_path, restore_path, …)`. This is a **file-level** restore: it does not read, rewrite, or re-serialize records, so it cannot corrupt oplog record bytes or re-hash a chain. In that narrow sense the plan's question ("verify restore doesn't rewrite records") answers **SAFE**.

But the honest finding is the other direction: **restore rolls the oplog backwards.** Any oplog record appended after the restored backup was taken is deleted by the restore, silently, together with the head pointer. The chain that remains is internally valid — which is worse than a broken chain, because the trace then *claims integrity over a truncated history*. This directly contradicts spec §3.2 ("Retention: none in v1 — the log is permanent… never a silent prune").

The restore path is reachable as an authenticated agent op: `POST /api/backup/restore` → `handlers/consolidation.rs:586 restore_backup` → `state.evict_user(&user_id)` (`:599`) → `restore_comprehensive_backup(…, &memory_db_path, …)` (`:610-619`) where `memory_db_path = base/{user_id}/storage` (`:595`).

### 1.6 User deletion — NEEDS-CHANGE (Finding D): spec §3.2 is false as written

Spec §3.2: *"No update or delete paths exist for it in the codebase (enforced by construction and by contract test)."*

That statement cannot be made true. `MultiUserMemoryManager::forget_user` (`state.rs:1362`) ends with:

```rust
let user_path = self.base_path.join(user_id);
if user_path.exists() { … std::fs::remove_dir_all(&user_path) … }   // state.rs:1391-1414
```

This deletes `{base}/{user_id}` wholesale — memory DB, graph DB, vector index, and therefore `CF_OPLOG`. It is reachable as an authenticated agent op: `DELETE /api/users/{user_id}` → `users::delete_user`. It is deliberate GDPR erasure (`state.rs:1447 purge_user_from_shared_db` is commented "GDPR").

The contract test Task 5 writes must therefore be scoped honestly: **"no per-record update or delete API exists on `CF_OPLOG`, and no lifecycle writer names it"** — provable, and what actually matters. Whole-user erasure and backup restore must be documented in the spec as the two **out-of-band, chain-terminating** operations, exactly as §3.2 pre-committed for a future retention policy. A test asserting "no delete path exists" would be a false claim in a compliance-facing feature.

### 1.7 Durability gaps for the new CF (Finding H)

- `MemoryStorage::flush` (`storage.rs:3148-3165`) flushes the default CF via `flush_opt` then **explicitly** the index CF via `flush_cf_opt(self.index_cf(), …)`, with the comment "RocksDB flush_opt only flushes default CF". A new CF is not flushed. Called from `state.rs:1689 flush_all_databases` (graceful shutdown). Durability still rests on the WAL (`storage.rs:1144 set_manual_wal_flush(false)`), so this is not data loss — but it is an inconsistency the oplog should not inherit silently.
- `rocksdb_memory_breakdown` (`storage.rs:3126-3146`) hardcodes `for cf_name in ["default", CF_INDEX]`. Oplog memtable/table-reader bytes would be invisible to `/api/health` and `/metrics` RSS accounting.
- The plan never states the oplog's write durability. `WriteMode::default()` (`storage.rs:1191`) is async unless `SHODH_WRITE_MODE=sync`. There is precedent for forcing durability on integrity-critical writes (`storage.rs:2741`, `:2795`, where the forget sweeps set `write_opts.set_sync(true)`), **but a forced fsync per append is irreconcilable with the plan's own <100 µs median append gate** — an fsync costs 100 µs to several ms depending on device and write cache. This is a choice, not an oversight to fix twice; it is resolved decisively in amendment 6.

---

## Step 2 — `open_or_repair_cf` and `create_missing_column_families`

### 2.1 The primary opener: SAFE

`MemoryStorage::new` sets both flags before opening:

```
storage.rs:1128    opts.create_if_missing(true);
storage.rs:1129    opts.create_missing_column_families(true);
```

and `opts` is the value handed to `open_or_repair_cf` at `storage.rs:1174`, which calls `DB::open_cf_descriptors(opts, path, build_cfs())` at `storage.rs:1222`. Adding a third `ColumnFamilyDescriptor` to the vec at `storage.rs:1175-1185` therefore **creates the CF on an existing database on first open, with no migration step.** The repair retry path at `storage.rs:1247` re-invokes the same `build_cfs` closure, so the descriptor set is identical on both attempts. Existing deployments upgrade cleanly. **SAFE — no change needed here.**

### 2.2 The plan's framing is too narrow — the real risk is every *other* opener (Finding A, CRITICAL)

`create_missing_column_families(true)` covers *listed-but-absent*. It does **not** cover *present-but-unlisted*: a read-write `DB::open_cf_descriptors` that omits a CF the database already contains fails with `Invalid argument: You have to open all column families. Column families not opened: <name>`.

> **Honesty flag:** this one sentence is the audit's only claim derived from the RocksDB library contract rather than from a code read, and it cannot be empirically tested inside a no-cargo-run audit. It is the documented read-write open contract (RocksDB `DBImpl::Open`, surfaced by the `rocksdb` crate's `open_cf_descriptors`). The asymmetry that makes it easy to miss: **read-only** opens (`DB::open_for_read_only`) *do* tolerate a subset, which is why the five `open_for_read_only` sites found (`storage.rs:1279`, `:1322`, `graph_memory.rs:2489`, `files.rs:92`, `feedback.rs:1643`, `prospective.rs:157`, `todos.rs:161`, `state.rs:1056`) are all safe: they open *legacy* directories, not the live storage path, and read-only anyway.

Every opener of the live per-user storage path was enumerated. Exactly one is a read-write open with a hardcoded CF list:

```rust
// migration.rs:318-330  (fn migrate_memory_db)
let cf_index = "memory_index";
let mut opts = RocksOptions::default();
opts.create_if_missing(false);
opts.create_missing_column_families(true);
let cfs = vec![
    ColumnFamilyDescriptor::new("default", RocksOptions::default()),
    ColumnFamilyDescriptor::new(cf_index, RocksOptions::default()),
];
let db = DB::open_cf_descriptors(&opts, storage_dir, cfs)
```

`storage_dir` is `{base}/{user_id}/storage` (`migration.rs:170`) — the exact live path. This is **reachable from a shipped CLI subcommand**: `main.rs:159` → `migration::migrate_all(&cli.storage_path, dry_run)`, and `main.rs:161-164` exits `1` if the report has errors.

Failure sequence on a real deployment:
1. Operator has pre-postcard data and has not yet run `shodh migrate` (gated by the marker file, `migration.rs:127`).
2. Operator upgrades and starts the server. `MemoryStorage::new` creates `CF_OPLOG` in every per-user storage DB.
3. Operator then runs `shodh migrate`. Every per-user open now fails; `migrate_all` records one error per user (`migration.rs:186-189`) and the process exits 1. The legacy-format records are **permanently unreachable by the migration tool** — exactly the "silent data loss" class this repo has been closing.

The fix is one line per descriptor list and is forward- and backward-safe (the function already sets `create_missing_column_families(true)`, so listing `CF_OPLOG` also works on DBs that predate it). `migration.rs:527` (graph) and `:650` (shared) open *different* databases and need no change.

`migration.rs:475` (`cf_handle(cf_index)`) is a lookup on the handle just opened, not an independent open.

Also checked and clear: `tests/file_memory_tests.rs:37` opens its own temp DB with the files CFs, not the storage path. `tui/`, `python/`, and `mcp-server/` contain no RocksDB opens (grep for `open_cf_descriptors|memory_index` in those trees returns nothing).

**Verdict: NEEDS-CHANGE — add `ColumnFamilyDescriptor::new(CF_OPLOG, RocksOptions::default())` to `migration.rs:324-327` in the same commit that adds the CF descriptor to `storage.rs:1175`.** These two edits must land together; splitting them across commits leaves a broken `shodh migrate` on `main`.

---

## Step 3 — Router / auth audit

### 3.1 Route inventory

`build_protected_routes` (`router.rs:100-475`) registers **162 method+path pairs over 153 distinct path patterns**, including `/metrics` when `SHODH_METRICS_PUBLIC` is unset (`router.rs:467-469`, secure default `false` per `router.rs:27-34`). Full extraction of the table (method, path, handler) is reproducible with:

```
awk 'NR>=100 && NR<=470' src/handlers/router.rs | tr -d '\n' \
  | sed 's/\.route(/\n.route(/g' | grep '^\.route('
```

Domain groups: remember/upsert (4), recall + proactive context (11), memory CRUD (10), forget (6), users/stats (4), compression (3), search (3), storage/index management (5), consolidation + backups (12), facts (6), lineage (9), graph advanced (12) + basic (2), visualization (5), todos (21), projects (7), files (5), reminders (8), sessions (7), A/B testing (13), integrations sync (2), SSE/WS streaming (4), MIF (3), `/metrics` (1).

`build_public_routes` (`router.rs:60-94`) registers 10: `/api/context/status` (GET+POST), `/api/context_status` (GET+POST TUI aliases), `/api/context/sse`, `/webhook/linear`, `/webhook/github`, `/graph/view`, `/dashboard`, and `/metrics` only when explicitly made public. `build_probe_routes` (`router.rs:42-49`) registers the 4 `/health*` probes.

**No agent op is registered in `build_public_routes`.** Webhooks are third-party ingress (HMAC-verified, not agent-attributable). `/api/context*` is status-line telemetry, not a memory op — consistent with the plan's exclusions. **The protected router is the complete surface of authenticated agent ops. Confirmed.**

### 3.2 Two transports reach `build_protected_routes` (Finding B, HIGH)

`build_protected_routes` and `build_router` are both public (`handlers/mod.rs:61`), and `server.rs` uses **both**:

```rust
// server.rs:260-268  — HTTP: auth layered by the caller
let routes = handlers::build_protected_routes(Arc::clone(&manager))
    .layer(axum::middleware::from_fn(auth::auth_middleware));
if rate_limit_enabled { routes.layer(make_governor_layer()) } else { routes }

// server.rs:339-340  — local IPC: build_router merges protected routes with NO auth layer
let ipc_router = handlers::build_router(Arc::clone(&manager))
    .layer(axum::middleware::from_fn(middleware::track_metrics));
```

`build_router` (`router.rs:482-488`) merges probe + public + protected and its own doc comment states it applies no auth. IPC auth is enforced inside `local_ipc.rs` (`use crate::auth::{self, AuthError}` at `local_ipc.rs:36`, challenge/response at `:223-259`, `:435`), before dispatch — so IPC traffic is authenticated, just not by the axum layer.

IPC is **enabled by default** (`server.rs:298-305`, `unwrap_or(true)`), and the MCP server prefers it when configured: `mcp-server/index.ts:130-133` builds `IPC_CLIENT` from `SHODH_IPC_ENDPOINT` and `:510-517` routes every `apiCall` through IPC when present, HTTP otherwise — hitting the same `/api/*` paths either way (`index.ts:1807` `/api/remember`, `:1969` `/api/recall`, …). This repo's own `.mcp.json` does not set `SHODH_IPC_ENDPOINT`, so the checked-in config currently runs over HTTP; but IPC mode is a first-class supported deployment (`SHODH_IPC_REQUIRED` exists on both sides).

**Consequence for the plan:** the plan mounts capture "on the router returned by `build_protected_routes`". Read at the `server.rs:261` call site, that captures HTTP only and leaves the entire IPC surface untraced — an audit log with a silent hole is worse than no audit log. The fix costs nothing: mount the layer **inside `router.rs::build_protected_routes`**, just before/at `router.with_state(state)` (`router.rs:474`). Both transports then inherit it, and for HTTP it is structurally *inside* auth because the caller's `.layer(auth_middleware)` wraps the whole router.

**Layer-ordering note (must be stated in Task 3 so it is not inverted):** on an axum `Router`, each `.layer()` wraps what came before, so the **last** layer applied is the **outermost**. `build_protected_routes(state)` (capture inside) `.layer(auth)` ⇒ auth outer, capture inner ⇒ capture runs only for requests auth let through. Adding `.layer(capture)` *after* `.layer(auth)` at the call site would place capture **outside** auth and trace rejected requests. Mounting inside `build_protected_routes` makes the correct ordering structural rather than call-site-dependent.

### 3.3 Auth establishes an API key, not a user identity (Finding C, part 1)

`auth::auth_middleware` (`auth.rs:188-248`) extracts `X-API-Key` / `Authorization: Bearer` / (for WS + SSE only) `?api_key=`, calls `validate_api_key`, and runs the next layer. It **inserts nothing into request extensions** and performs no user binding — one valid key may act as any `user_id`. Therefore:

- The capture middleware cannot learn `user_id` from auth; identity must come from the handler (body-parsed), exactly as the plan's enrichment design assumes. Good.
- "Authenticated" in the trace means "presented a valid API key", not "is user X". The audit-export honesty rule (spec §5) should say so; `witnessed` attests *engine observation*, not *identity verification*.

### 3.4 The exclusion set must be larger than `/api/trace/*` (Finding C, part 2 — HIGH)

Three route classes inside the protected router are not agent ops and must not be captured:

1. **`/metrics`** (`router.rs:468`). A Prometheus scrape is an authenticated `GET` with no body and therefore **no `user_id` and no `session_id`**. Combined with the plan's "records with unset identity are still appended" rule and spec §3.2's zero-retention policy, every scrape appends a permanent oplog record under a junk identity, at scrape interval, forever. That is an unbounded oplog-flooding vector, not a cosmetic issue.
2. **SSE / WebSocket routes**: `GET /api/events/sse`, `GET /api/events`, `GET /api/stream`, `GET /api/context/monitor` (`router.rs:449-452`). `next.run(req)` returns when the response *starts* (stream open / upgrade), so `outcome` and any duration are meaningless for a connection that then lives for hours. Capture at most a single `stream_open` op, or exclude.
3. **`/api/trace/*` reads** — as the plan already says.

Additionally, **identity-less requests must be dropped, not appended under a synthetic user**, because `MultiUserMemoryManager::get_user_memory` (`state.rs:1242-1332`) **creates on miss**: it builds `MemoryConfig { storage_path: base/{user_id} }` (`:1262-1265`) and calls `MemorySystem::new` (`:1275`), which runs `create_dir_all` + `create_if_missing(true)` (`storage.rs:1124-1128`), wires a graph (`:1319`), and inserts into the cache (`:1327`). So a capture attempt for user `unknown` **materialises a junk user directory and a full MemorySystem** — with a 4-attempt lock-contention retry loop and up to ~750 ms of sleeps on contention (`state.rs:1288-1296`). That is fatal both to the "capture never affects the operation" rule and to the <100 µs append budget.

**Task 3 must therefore look the user up cache-only.** `MultiUserMemoryManager` currently exposes no cache-only accessor: `user_memories.get()` is private, and the only public read-through options are `get_user_memory` (creates) and `cached_user_memories()` (`state.rs:1627`, returns *all* cached users). A small accessor is needed — see amendment 8. In practice the user is always already cached, because the handler that just ran used it.

### 3.5 Non-router transports

- **Zenoh** (`src/zenoh_transport/handlers.rs`) implements `handle_remember` (`:290`) and `handle_recall` (`:616`) as full memory ops that never touch axum — `handle_remember` even mirrors `handlers/remember.rs` including `session_store().get_or_create_session` (`:524`). It is behind an **optional, non-default** feature (`Cargo.toml:298 zenoh = ["dep:zenoh"]`, `default = []` at `Cargo.toml:289`, gated at `lib.rs:80-81`), so slice 1 does not have a live hole here. **Document it as a known, feature-gated gap** in the PR body and spec §7 rather than treating it as a slice-1 defect; note that enabling `zenoh` silently reduces trace completeness.
- **Hooks** call HTTP endpoints and are covered by the router. One is broken today: `hooks/claude-code-ingest.sh:88` POSTs to `$API_URL/api/record`, and **no `/api/record` route exists** anywhere in `src/` (repo-wide grep for `"/api/record"` returns nothing). With `-s`, `>/dev/null 2>&1`, and `|| true` the 404 is invisible. Task 4's instruction to add the trace-report call "after the existing ingest call" must not be read as an endorsement that the existing call works; the correct ingest path is `/api/remember` (as `hooks/stop.sh:28` already uses). The plan's other hook claim **is** correct and was verified: `claude-code-ingest.sh:28` does contain `set -e` (`stop.sh` and `user-prompt.sh` do not).

---

## Step 4 — Session identity

### 4.1 What carries `session_id` today

In `src/handlers/types.rs`, exactly three declarations (`grep session_id src/handlers/types.rs`):

| Location | Type | Route | Notes |
|---|---|---|---|
| `types.rs:56` `ContextStatus.session_id` | `Option<String>` | response/broadcast DTO | status-line telemetry |
| `types.rs:101` `RecallRequest.session_id` | `Option<String>` | `POST /api/recall` (+ aliases) | doc-commented "session-scoped retrieval (used with mode=temporal)" — optional, retrieval-scoped |
| `types.rs:547` `ContextStatusRequest.session_id` | `String` (required) | `POST /api/context/status` — **public route** | not an agent op |

Elsewhere: `handlers/sessions.rs:104` takes `session_id` as a path param and parses it as a UUID (`:109 uuid::Uuid::parse_str`); `sessions.rs:368` carries `Option<String>` in a history entry; `sessions.rs:270/429/631` stuff `session_id` into memory *metadata*. **`RememberRequest` has no `session_id` field** — `remember.rs:121` only mentions it as an example metadata key.

So the answer to "which request types carry `session_id` today" is: **recall (optional) and the public context-status route (required). Nothing else.**

### 4.2 The server already owns a session identity — use it (Finding F)

`remember.rs:692` and `todos.rs:1120`, `:1791` do not read a client session; they call:

```rust
let session_id = state.session_store().get_or_create_session(&req.user_id);
```

`SessionStore::get_or_create_session` (`memory/sessions.rs:686-698`) returns the user's active session or starts one. `SessionId` is a UUID newtype (`sessions.rs:182 Self(Uuid::new_v4())`), consistent with `sessions.rs:109` parsing path params as UUIDs. This identity is what `/api/sessions*`, session digests, and `session_history` already key on — i.e. what slice 2's transcript pane and session picker will need to join against.

**Recommendation (replaces the plan's fallback):** when the request carries no `session_id`, use `state.session_store().get_or_create_session(user_id)` rather than minting `adhoc-{user_id}-{UTC date}`. Rationale:

- `adhoc-…` invents a **second session namespace** that nothing else in the codebase can resolve; slice 2's picker would have to reconcile two kinds of session identity.
- The date-bucketed form collapses a day's activity into one "session", which is precisely the granularity the transcript pane exists to avoid.
- `SessionStore` is in-memory, so it resets on restart — acceptable and arguably correct (a restart *is* a new session), and strictly better than a per-day bucket.

Cost to note in Task 3: `get_or_create_session` takes a write lock on a `HashMap` and may create an entry — a real (small) side effect on read paths. `remember`/`todos` already pay it, and it is cheap relative to the ~100 µs budget. If a truly side-effect-free fallback is required, prefix with `adhoc-` **only** as a last resort and reserve the prefix in validation.

### 4.3 Collision analysis (the brief's explicit question)

- **Structural collision: none.** Real session ids are hyphenated UUIDs (36 chars, `sessions.rs:182` + `:109`); `adhoc-{user_id}-{YYYY-MM-DD}` cannot parse as a UUID.
- **Namespace is not reserved, though.** `RecallRequest.session_id` is unvalidated free-form `Option<String>` — a client can literally send `adhoc-alice-2026-07-30`. Because `CF_OPLOG` lives in the **per-user** storage DB (§1.1), this is same-user self-forgery only; cross-user injection is structurally impossible. Reported events are self-asserted by definition (spec §2), so this is a documentation matter, not a hole — but if the `adhoc-` form is kept, validation must reject a client-supplied `session_id` starting with `adhoc-`.

### 4.4 `session_id` is unvalidated — and that breaks the key scheme (Finding F, HIGH)

There is **no `validate_session_id`** in `src/validation.rs` (full `pub fn validate*` list read; 27 validators, none for session ids) and `grep session_id src/validation.rs` returns nothing. `session_id` is unbounded free-form text and may contain `:`.

The plan's keyspace is `op:{session_id}:{seq:016}`, with `head:{session_id}` and `incomplete:{session_id}` as the only other prefixes, and `oplog_read` prefix-scans `op:{session_id}:`. With `:` permitted:

- Session `"x"` read with prefix `op:x:` also matches records of session `"x:0000000000000000"` → **foreign records bleed into another session's trace and `verify_chain` fails on a chain that was never broken.**
- Unbounded length → unbounded RocksDB keys on an append-only CF.

**Fix:** add `validate_session_id` (max 128 chars, charset `[A-Za-z0-9._-]`, non-empty, reject `:`) and apply it at `POST /api/trace/report` and wherever a client-supplied `session_id` enters the oplog; alternatively hex-encode `session_id` in the key. Validation is the better choice — it also keeps `GET /api/trace/{session_id}` path params sane.

### 4.5 Hook and MCP session coverage — the finding that decides slice-1 usefulness

- **No hook sends a `session_id`.** `grep session_id hooks/*.sh` returns nothing across `claude-code-ingest.sh`, `stop.sh`, `user-prompt.sh`, `session-start.sh`.
- **MCP sends one only on `recall`**, and only when the model chooses to pass it: declared in the tool schema at `mcp-server/index.ts:810`, destructured at `:1857-1862`, forwarded conditionally at `:1974` (`...(session_id ? { session_id } : {})`). `remember` (`index.ts:1807`) sends none.

So without an amendment, essentially **every** slice-1 record — witnessed MCP ops and all reported hook events — lands in the fallback bucket, and witnessed/reported records for the same real conversation land in *different* buckets. The `InformedBy`/`NextAction` intra-session ordering that spec §3.3 builds on would be joining noise.

Claude Code hook payloads do carry `session_id` in their JSON input, and the hooks already parse that JSON (`claude-code-ingest.sh:85` reads `.cwd` from `$INPUT` via `jq`). Task 4 must extract `.session_id` from `$INPUT` and send it; and the MCP `remember` call should forward its session id too. This is the difference between a transcript and a pile.

---

## Step 5 — `sha2` and the metrics facility

### 5.1 Dependencies: SAFE, no additions

```
Cargo.toml:195   sha2 = "0.10"
Cargo.toml:196   hex = "0.4"
```

Both are direct dependencies; `Cargo.lock:3996` pins `sha2 0.10.9` and `Cargo.lock:1637` pins `hex 0.4.3`. `backup.rs:666 hash_directory_sorted` already uses `Sha256`, so the crate is compiled today. **No `Cargo.toml` change is needed for Task 1.** Use `hex::encode` for the hex digests rather than hand-rolling formatting.

### 5.2 Metrics: prometheus statics + a registration list (SAFE, with one gotcha)

The facility is the `prometheus` crate with a process-global registry, **not** the `metrics` crate — so there is no `counter!`/`metrics::` macro. The exact pattern (`src/metrics.rs`):

1. **Declare** a `LazyLock` static, e.g. `metrics.rs:436-442`:
```rust
pub static ERRORS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new("shodh_errors_total", "Total errors by type"),
        &["error_type", "endpoint"],
    ).expect("ERRORS_TOTAL metric must be valid at compile time")
});
```
2. **Register** it in `do_register_metrics()` using the local `register!` macro declared at `metrics.rs:675-681`:
```rust
register!(TRACE_CAPTURE_FAILURES_TOTAL, "TRACE_CAPTURE_FAILURES_TOTAL");
```
`register_metrics()` (`metrics.rs:661-670`) is idempotent via `METRICS_INIT: OnceLock` and is called once at `server.rs:124`.
3. **Increment** from call sites as `crate::metrics::NAME.with_label_values(&[…]).inc()` (labelled) or `.inc()` (plain) — e.g. `circuit_breaker.rs:228`, `memory/mod.rs:2331`, `storage.rs:799`.

**Gotcha to state in Task 3:** a metric that is declared but **not** added to the `register!` list never appears in `/metrics` — the "loud failure" requirement of spec §3.1 would itself fail silently. Both edits are mandatory.

**Recommendation:** add a dedicated `TRACE_CAPTURE_FAILURES_TOTAL: IntCounter` (or `IntCounterVec` labelled `["reason"]` with values like `append_error`, `identity_unset`) rather than reusing `ERRORS_TOTAL`. Spec §3.1 wants a metric an operator can alert on by name; `ERRORS_TOTAL`'s `endpoint` label also carries cardinality risk if raw request paths (with ids) are passed. `metrics.rs:9-10` explicitly warns against high-cardinality labels — never label by `user_id` or `session_id`.

### 5.3 Bench harness (Task 5's open question): exists, but still prefer the `#[ignore]`d test

A criterion harness is already configured: `Cargo.toml:224 criterion = { version = "0.5", features = ["html_reports"] }` with **11** `[[bench]]` entries (`Cargo.toml:228-271`) and files in `benches/`. So Task 5's condition "ONLY if a bench harness already exists" is satisfied.

**Recommendation anyway: implement the append-cost measure as the `#[ignore]`d test in `tests/trace_capture.rs`.** Running a criterion bench requires `cargo bench`, which is outside the allowed command set for this repo (`cargo check`/`clippy`/`test` only) and outside CI's timing-sensitive path, whereas `cargo test -- --ignored` is allowed and reproducible. Adding a 12th `[[bench]]` entry buys nothing for a single median/p99 number.

### 5.4 Concurrency: the plan's open question resolved

The plan asks whether `MemoryStorage` methods are already externally serialized. **They are not.** `get_user_memory` returns `Arc<parking_lot::RwLock<MemorySystem>>` (`state.rs:1242`), and the hot handlers take **read** guards — `recall.rs:513/648/727/840/918/969`, `remember.rs:673/753/815/836/858/1036`. Concurrent readers therefore call storage methods in parallel. A `read → seq+1 → write` head update would race and produce duplicate `seq` values / forked chains.

`MemoryStorage` already owns a `parking_lot::Mutex` (`storage.rs:1101 write_retry_buffer`), so the pattern is in-house. Task 2 must add a dedicated `oplog_append_lock: parking_lot::Mutex<()>` held across head-read→batch-write only. Note that this also means the "atomic per session" claim needs the mutex to be *per storage instance* (per user), which is sufficient because sessions never span users.

---

## Step 6 — Commit

Audit doc written to `docs/superpowers/audits/2026-07-30-traceability-slice1-audit.md` and committed on `feat/traceability-capture-oplog` (created off `origin/main` @ `f4164612`) with message `docs: traceability slice-1 substrate audit`. No production files touched; no cargo commands run. **SAFE.**

---

## Findings index

| # | Severity | Finding | Where |
|---|---|---|---|
| A | **CRITICAL** | `migrate_memory_db` opens the live storage DB read-write with a hardcoded 2-CF list; breaks (exit 1, legacy data stranded) once `CF_OPLOG` exists | `migration.rs:318-330`, reachable `main.rs:159` |
| B | **HIGH** | Two transports reach the protected routes; capture mounted at the `server.rs:261` call site would miss the entire IPC surface | `server.rs:261` vs `:339`, `router.rs:482-488` |
| C | **HIGH** | `/metrics` + SSE/WS are inside the protected router; identity-less capture would flood the oplog forever and, via `get_user_memory`, materialise junk users | `router.rs:468`, `:449-452`, `state.rs:1242-1332` |
| D | **HIGH** | Spec §3.2 "no delete paths exist" is **false**: `forget_user` `remove_dir_all`s the user directory including `CF_OPLOG` | `state.rs:1362`, `:1391-1414` |
| E | **MEDIUM** | Backup restore silently rolls the oplog backwards, leaving a self-consistent but truncated chain that falsely claims integrity | `backup.rs:304-347`, `consolidation.rs:586-619` |
| F | **HIGH** | `session_id` unvalidated (`:` breaks the `op:{sid}:{seq}` prefix scan); no hook and no MCP `remember` sends one; server already owns a better identity | `validation.rs` (absent), `types.rs:101`, `hooks/*.sh`, `index.ts:1974`, `sessions.rs:686` |
| G | **MEDIUM** | `MemoryStorage` is unreachable from handlers (`long_term_memory` private, no accessor) and appends are not externally serialized (`.read()` guards) | `memory/mod.rs:214`, `recall.rs:513`, `remember.rs:673` |
| H | **LOW** | `flush()` and `rocksdb_memory_breakdown` hardcode `default`+`memory_index`; oplog append durability unspecified | `storage.rs:3148-3165`, `:3126-3146`, `:1191` |
| I | **LOW** | Zenoh transport performs memory ops outside the router (feature-gated, non-default) — documented known gap | `zenoh_transport/handlers.rs:290/616`, `Cargo.toml:289/298` |
| J | **INFO** | `hooks/claude-code-ingest.sh:88` POSTs to `/api/record`, which does not exist — pre-existing silent 404 | `claude-code-ingest.sh:88` |
| K | **INFO** | Bench harness exists (criterion, 11 `[[bench]]`), but `#[ignore]`d test remains the right vehicle | `Cargo.toml:224-271` |

Verified-and-clear (no action): no `drop_cf`/`DB::destroy`/`delete_range`/`compact_range`/compaction filter anywhere in `src/`; all `open_for_read_only` sites target legacy dirs; backup **does** include new CFs automatically; `state.rs:2512` is an existence check, not an opener; the plan's `set -e` claim about `claude-code-ingest.sh` is correct.

---

## Amendments required to Tasks 1–5

**Task 1 (oplog core)**
1. **No `Cargo.toml` change.** `sha2 0.10` (`Cargo.toml:195`) and `hex 0.4` (`:196`) are already direct deps; use `hex::encode`. Delete the plan's "add workspace-pinned if absent" branch.
2. Module registration goes at `src/memory/mod.rs` between `pub mod lineage;` (`:21`) and `pub mod pattern_detection;` (`:22`) to keep the list alphabetical.

**Task 2 (storage / `CF_OPLOG`)**
3. **(Finding A, blocking)** In the *same commit* that adds the `CF_OPLOG` descriptor to `storage.rs:1175-1185`, add `ColumnFamilyDescriptor::new(CF_OPLOG, RocksOptions::default())` to the `cfs` vec at `migration.rs:324-327`. Splitting these across commits leaves `shodh migrate` broken on `main`. Note `CF_OPLOG` must be `pub` (as the plan's Task 2 interface block already declares) — do **not** copy the `CF_INDEX` pattern, which is private (`storage.rs:1080`) and is precisely why `migration.rs:319` duplicates the `"memory_index"` string literal; `migration.rs` must reference the constant, not a second literal.
4. **(Finding G)** Add `oplog_append_lock: parking_lot::Mutex<()>` to `MemoryStorage` (`storage.rs:1092-1104`) and hold it across head-read → `WriteBatch` write. Storage methods are **not** externally serialized (`.read()` guards at `recall.rs:513`, `remember.rs:673`) — the plan's "audit documents whether…" is hereby answered: they are not.
5. **(Finding F)** Add `validate_session_id` to `src/validation.rs` (≤128 chars, `[A-Za-z0-9._-]`, non-empty, **reject `:`**) and call it before any key construction. Without it, `oplog_read`'s `op:{sid}:` prefix scan bleeds across sessions and `verify_chain` reports false tampering.
6. **(Finding H — durability decision, resolves a constraint conflict)** Extend `MemoryStorage::flush` (`storage.rs:3159-3162`) with a `flush_cf_opt` for the oplog CF, and add `CF_OPLOG` to `rocksdb_memory_breakdown`'s array (`storage.rs:3129`). **Do NOT force `set_sync(true)` on oplog appends.** Oplog appends inherit `self.write_mode` exactly like every other storage write, for one reason: a forced fsync per append cannot coexist with Task 5's <100 µs median gate (§1.7). Document the resulting crash window in `oplog.rs`'s module docs: the WAL is written on every append (`storage.rs:1144 set_manual_wal_flush(false)`), so records survive a **process** crash; a power loss or OS crash can lose the last unflushed appends, and operators requiring power-loss durability for the audit trail set `SHODH_WRITE_MODE=sync` (which then also applies to the oplog). This is a deliberate, documented tradeoff — not a silent default.
7. **Drop the `user_id` parameter from `oplog_sessions`** (plan interface line) or document it as a record-field filter: `MemoryStorage` is already per-user (`{base}/{user_id}/storage`, `storage.rs:1123`), so the parameter is vestigial and invites a false cross-user-listing expectation.
8. **(Finding G)** Add `pub fn storage(&self) -> &Arc<MemoryStorage>` (or `oplog_*` pass-throughs) to `MemorySystem`: `long_term_memory` is private (`memory/mod.rs:214`) with no accessor, so Task 3's middleware cannot reach the new methods as the plan is written. `get_db()` (`memory/mod.rs:7228`) returns the raw `Arc<DB>` and must **not** be used for oplog writes — raw-CF writes are the one thing the append-only contract forbids outside tests.

**Task 3 (capture middleware)**
9. **(Finding B)** Mount `capture_middleware` **inside `router.rs::build_protected_routes`** (at `router.rs:474`), not at the `server.rs:261` call site. This covers HTTP *and* local IPC (`server.rs:339`, enabled by default per `server.rs:298-305`) and makes "inside auth" structural. Record in the code comment that `.layer()` order is inner-first: last-applied is outermost. **Consequence to plan for:** four existing test harnesses build the protected router directly and will therefore inherit capture — `handlers/test_helpers.rs:72`, `tests/handler_tests.rs:74`, `tests/handler_pipeline_tests.rs:63`, `tests/pipeline_integration_tests.rs:63`. Task 3's `cargo test --test handler_tests` re-run will exercise all of them; treat this list as the checklist rather than debugging a surprise.
10. **(Finding C)** Exclusion list must be `/api/trace/*` **plus** `/metrics` (`router.rs:468`) **plus** the four SSE/WS routes (`router.rs:449-452`). `/metrics` alone would append one permanent record per Prometheus scrape under a junk identity.
11. **(Finding C)** Identity-less requests: increment the loud counter and **drop the record**. Do **not** append under a synthetic user — `get_user_memory` (`state.rs:1242`) creates the user directory and a full `MemorySystem` on miss (with up to ~750 ms of retry sleeps at `:1288-1296`). Add a cache-only accessor to `MultiUserMemoryManager` (`user_memories.get()` is private; `cached_user_memories()` at `state.rs:1627` returns all users) and use it. This replaces the plan's `identity:unset` / `user unknown` rule. **Because "drop" is a real capture hole, the no-identity route set must be named, not discovered:** some protected routes have no user identity by construction — verified examples are `GET /api/users` (`users.rs:71`, `State` only), `GET /api/mif/adapters` (`mif.rs:311`, no arguments at all), and the `/api/ab/*` family (e.g. `ab_testing.rs:81`, `State` only). Task 3 must export that set as a named constant (e.g. `NO_IDENTITY_OPS`) alongside the exclusion list so Task 5 can consume it — see amendment 18. (For contrast, `GET /api/stats` *does* carry identity: `users.rs:36-40` validates `query.user_id`.)
12. **(Finding F)** Session fallback: use `state.session_store().get_or_create_session(user_id)` (`memory/sessions.rs:686`, the identity `remember.rs:692` and `todos.rs:1120` already use) instead of `adhoc-{user_id}-{UTC date}`. If `adhoc-` is retained for any path, reserve the prefix in `validate_session_id`.
13. **(Step 5)** The loud metric is prometheus, not the `metrics` crate: declare `pub static TRACE_CAPTURE_FAILURES_TOTAL: LazyLock<IntCounter…>` in `src/metrics.rs` **and** add `register!(TRACE_CAPTURE_FAILURES_TOTAL, "TRACE_CAPTURE_FAILURES_TOTAL");` inside `do_register_metrics()` (`metrics.rs:672+`). Declared-but-unregistered never reaches `/metrics`. Never label by `user_id`/`session_id` (`metrics.rs:9-10`).

**Task 4 (reported ingest + hooks)**
14. **(Finding F / §4.5)** Hooks must send the Claude Code `session_id`: extract it from the hook payload with `jq -r '.session_id // empty'` on `$INPUT` (the hooks already parse that JSON — `claude-code-ingest.sh:85`). Without it every reported event lands in the fallback bucket, separated from the witnessed records of the same conversation, and slice 2's session feed has nothing to join on. Also forward a session id from MCP `remember` (`mcp-server/index.ts:1807`), which currently sends none. Separately, do **not** assume the "existing ingest call" in `claude-code-ingest.sh:88` works — it POSTs to `/api/record`, which does not exist (Finding J); either fix it to `/api/remember` in this task or leave it untouched and say so, but do not describe it as working.

**Task 5 (gates)**
15. **(Findings D, E)** Scope the lifecycle-exclusion contract test to what is provable: *no per-record update/delete API exists on `CF_OPLOG`, and no lifecycle writer names the CF*. Do **not** assert "no delete path exists" — `forget_user` (`state.rs:1362`) `remove_dir_all`s the user directory and backup restore (`backup.rs:304`) rolls the log back. Both must be documented in the PR body and proposed as a spec §3.2 amendment naming them as explicit out-of-band chain-terminating operations, with restore additionally setting `integrity: incomplete` for affected sessions. **Note that this creates a second, intentional oplog write site** — `oplog_mark_incomplete` called from `handlers/consolidation.rs::restore_backup` (~`:610-619`, after `restore_comprehensive_backup` returns). It writes only the `incomplete:{session_id}` flag key, never a record, and must be listed in the contract test's enumerated-allowed write set or the test will flag it as a violation.
16. **(Finding K)** Keep the append-cost measure as an `#[ignore]`d test. A criterion harness does exist (`Cargo.toml:224`, 11 `[[bench]]` entries at `:228-271`), but `cargo bench` is outside this repo's allowed command set; `cargo test -- --ignored` is not.
17. **(Finding I)** The capture-completeness test asserts coverage of the *router* surface. State explicitly in the test's doc comment that the `zenoh` feature (`Cargo.toml:298`, non-default) performs memory ops outside the router (`zenoh_transport/handlers.rs:290/616`) and is a documented slice-1 gap, so a future reader does not mistake the passing test for total coverage.
18. **(Finding C / amendment 11 — makes Task 5 Step 1 writable)** The plan's `records == requests` assertion is **unsatisfiable as written** once amendment 11 lands: identity-less routes produce zero records by design. Task 5 Step 1 must assert `records == requests - requests_to(NO_IDENTITY_OPS ∪ EXCLUDED_OPS)`, importing both sets from `handlers/trace.rs` (amendments 10 and 11) rather than re-listing paths in the test. The excluded/no-identity sets are then themselves the moving-parts contract: a future route added to either set is a visible, reviewed diff in `trace.rs`, and a future route added to neither shows up as the intended count mismatch. If any route in `NO_IDENTITY_OPS` is later judged to be a genuine agent op (the `/api/ab/*` family is the likely candidate), the fix is a server-identity sentinel `user_id`, decided explicitly — not a widened tolerance in the test.
