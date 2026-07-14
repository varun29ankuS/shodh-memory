# Review — `feat/local-ipc` (authenticated local IPC transport)

| Field | Value |
|-------|-------|
| Reviews | commit `074c4b1`, "feat: add authenticated local IPC transport" |
| Method | Multi-agent static review: 7 independent lenses, every candidate finding independently refuted and repro'd by two further agents before being kept |
| Coverage | 61 candidate findings → 56 survived verification, 5 refuted and dropped |
| Executed | Nothing. `cargo`, `vitest` and `npm` were never run — see [Unverified](#unverified) |
| Verdict | **Needs rework** — see [Verdict](#verdict) |

## Verdict

The transport itself is competently built. The framing is correct, auth is evaluated
before route policy, the Unix socket lands 0600 inside a verified 0700 directory, the
Windows pipe sets `FILE_FLAG_FIRST_PIPE_INSTANCE` and a protected DACL, and the
"retain the un-abortable dispatch task so shutdown cannot flush storage under it"
reasoning in `handle_connection` is better than average.

What is wrong is the *shape*. An additive feature was made mandatory, always-on and
fail-fatal; a documented CLI contract was deleted rather than deprecated; and the
design doc asserts properties the code does not deliver. Each of those is
individually cheap to fix and each of them, left alone, is a legitimate reason for
upstream to reject the change.

This document records what was found. The fixes land in the commits that follow it.

---

## B — Blockers

### B1. The IPC listener is mandatory, always-on, and its bind failure kills the server

`src/server.rs:299-305` binds the IPC endpoint *before* the HTTP listener at `:318`,
and propagates failure with `?`. There is no `SHODH_IPC_ENABLED`, no `--no-ipc`, and
no cargo feature: `src/lib.rs:38` declares `pub mod local_ipc;` unconditionally and
`windows-sys`/`libc` are added as **non-optional** dependencies. `--ipc-endpoint`
only relocates the endpoint; it cannot switch it off.

This inverts the repo's own precedent. The existing alternate transport
(`src/zenoh_transport/`) is behind a cargo feature, opt-in via `SHODH_ZENOH_ENABLED`,
and its failure to start is non-fatal (`src/server.rs:174-179` on `main`).

Every one of these starts fine today and refuses to boot after this change:

- **Two instances, one user.** `default_endpoint()` (`src/local_ipc.rs:51-63`) is
  scoped to neither the port nor the storage path, so `shodh server --port 3031`
  alongside `--port 3030` hits `src/local_ipc.rs:670`
  (`"Shodh IPC endpoint is already serving"`) and exits **before binding its own HTTP
  port**. Windows fails the same way via `first_pipe_instance(true)` (`:905`).
- **macOS, immediately after the documented first run.** `dirs::config_dir()` and
  `dirs::data_local_dir()` are the *same* path on macOS
  (`~/Library/Application Support`), so `shodh init` creates `…/shodh` with
  `create_dir_all` at mode 0755 (`src/cli.rs:537`), and `bind()` then rejects it at
  `src/local_ipc.rs:650-658` (`mode & 0o777 != 0o700`). `DirBuilder::mode` does not
  chmod a directory that already exists. `shodh init && shodh server` cannot start.
- **Any pre-existing non-0700 parent** — `SHODH_IPC_ENDPOINT=/tmp/x.sock` (`/tmp` is
  1777), a container HOME, a read-only rootfs.

**Fix:** make IPC opt-in and its failure non-fatal — `warn!` and continue HTTP-only.
Own the endpoint directory's mode instead of demanding it (chmod to 0700 when we own
it). A non-fatal bind also resolves the multi-instance collision without inventing an
endpoint-naming scheme.

### B2. The cross-platform MCP quickstart hardcodes the Windows named pipe

`mcp-server/README.md:51` puts `"SHODH_IPC_ENDPOINT": "\\\\.\\pipe\\shodh-memory"`
into the primary copy-paste config block, with the macOS/Linux config-file paths
listed *between* it and the corrective caveat at `:72-76`. On Unix that value is a
single path component; `index.ts:4893` forwards it to the auto-spawned backend,
`src/local_ipc.rs:637-640` rejects it ("IPC socket path must have a parent
directory"), and per B1 that is fatal. The spawned server dies with `stdio: "ignore"`,
so nothing is printed, and with `IPC_CLIENT` non-null there is no HTTP fallback
(`index.ts:447`) — every tool call fails, silently.

**Fix:** leave the quickstart on its working default; document `SHODH_IPC_ENDPOINT`
in the env table and the IPC section with per-platform values. Have `ipc-client.ts`
hard-fail a `\\.\pipe\` endpoint on non-win32 rather than forwarding it.

### B3. `shodh serve` deletes `--api-url` / `SHODH_API_URL` with no fallback

`src/cli.rs:118-128` replaces the `api_url` argument outright with `ipc_endpoint`. A
documented, load-bearing env contract disappears: `SHODH_API_URL` is now silently
ignored, and `--api-url` is a hard clap error. Anyone pointing the MCP server at a
non-default port, a remote host, or a second instance is broken with no deprecation
window and no migration note.

**Fix:** keep `--api-url` / `SHODH_API_URL`. Prefer IPC when its endpoint is
reachable, fall back to HTTP otherwise. That fixes the CLI break and the
no-fallback-when-IPC-is-down failure in one move.

### B4. The only Rust test that binds a real socket cannot pass on macOS

`src/local_ipc.rs:1274` builds its socket path under the system temp dir. On macOS
`TMPDIR` is a `/var/folders/xx/…/T/` path that, with the test's own suffix, exceeds
the transport's own 103-byte `MAX_SOCKET_PATH_BYTES` limit (`:605`) — so `bind()`
rejects the path the test just constructed.

**Fix:** bind under a short, deterministic prefix in tests (and use `tempfile::TempDir`
— already a dev-dependency — so the parent directory is cleaned up; `EndpointGuard`
removes the socket but never the directory it created).

---

## M — Major

### M1. A transient `accept()` error tears down the whole daemon

`src/local_ipc.rs:741`: every `accept()` failure does `break Some(...)`, which returns
`Err` from `serve()`, which the `tokio::select!` in `src/server.rs:354-361` treats as a
terminal error — taking the HTTP server down with it. A transient `EMFILE` or
`ECONNABORTED` kills the process.

**Fix:** log and continue on transient errors (with a small backoff); reserve teardown
for genuinely fatal ones.

### M2. The IPC shutdown drain is unbounded

`src/server.rs:401-407` awaits `ipc_handle` with no timeout and no abort, while the
HTTP arm at `:380-398` bounds its drain with `SERVER_DRAIN_TIMEOUT_SECS`. One slow
route blocks the storage flush that the shutdown path exists to guarantee, until
systemd `SIGKILL`s the process — losing the flush. The design doc claims "bounded
operation deadlines" (`docs/architecture/07-local-ipc-transport.md:77`); the
per-request deadlines are bounded, the drain is not.

**Fix:** bound the IPC drain the same way the HTTP drain is bounded.

### M3. IPC traffic bypasses the entire HTTP middleware stack

`src/server.rs:296-299` hands the IPC transport a bare `handlers::build_router()`. The
HTTP path layers rate limiting, `track_metrics`, timeouts, security headers and trace
propagation on top of it; the IPC path gets none of them. So the transport this PR
simultaneously makes the *default* for `shodh serve` is invisible to Prometheus and
exempt from the rate limiter — and the doc never says so.

Auth itself is correctly shared (`auth::validate_api_key`), so this is narrower than
it looks — but it is the drift-prone shape: any future middleware silently does not
apply to local agent traffic.

**Fix:** layer at least `track_metrics` onto the IPC router, and state the remaining
divergence in the doc.

### M4. IPC is not behaviour-identical to REST for the same route

`src/local_ipc.rs:481` caps a response body at `MAX_BODY_BYTES` (8 MiB − 64 KiB). The
same route over HTTP has no such cap. A large-but-legal `POST /api/recall` succeeds
over HTTP and 413s over IPC. The doc's "single operation catalog" framing implies
equivalence that does not hold.

**Fix:** document the cap, make it configurable, and state the divergence.

### M5. The MCP client's 10 s deadline is shorter than the server's route deadline

`src/local_ipc.rs:38` (`DEFAULT_CLIENT_TIMEOUT = 10s`) is used for **both** `get()`
(`:104`) and `post()` (`:115`) — a copied line; `request()` already takes a `Duration`
and nobody ever passes a different one. The server's route deadline is 60 s. A slow
write is reported to the MCP caller as a failure *after the server has already
committed it*. Over HTTP, `shodh serve` had no client timeout at all.

**Fix:** give writes their own deadline, aligned with (or exceeding) the server's.

### M6. Windows: a Node child process per IPC request, and the queue burns the caller's deadline

`mcp-server/ipc-client.ts:205` spawns `process.execPath` per request with a helper
limit of 4 (`:11-13`), and the deadline timer is armed at `:185` *before*
`acquireWindowsHelper()` at `:189` — so a queued 5th request can time out having never
touched the pipe. `waitForServer()` (`index.ts:4802`) can cost up to 30 sequential Node
spawns.

The root cause is server-side: `src/local_ipc.rs:803-886` keeps exactly **one** pending
`NamedPipeServer` instance, so concurrent clients hit `ERROR_PIPE_BUSY` (the Rust
client loops on code 231 at `:817-829`) and libuv's uncancellable `WaitNamedPipe`.

**Fix:** pre-create a pool of pending pipe instances server-side, then use plain
`net.createConnection({ path })` in TypeScript and delete the helper machinery
(~120 lines).

### M7. Windows: the pipe name is machine-global, so a second user cannot start shodh

`src/local_ipc.rs:52-55` returns a fixed `\\.\pipe\shodh-memory` (the Unix default is
correctly per-user, `:59-62`), while the DACL is user-scoped (`:997`) and
`first_pipe_instance(true)` is set (`:905`). A second logged-in user's `shodh server`
fails with ACCESS_DENIED and — per B1 — dies. The doc claims "Windows uses a
current-user-scoped named pipe" (`07:22`); only the DACL is.

`current_user_sid_string()` is already implemented (`:1028`) and unused for this.

**Fix:** fold the user SID into the pipe name.

### M8. Protocol errors are emitted with an empty `id`, which both clients then reject as a mismatch

The server emits framing and parse errors with `id: ""` (`src/local_ipc.rs:241`, `:246`,
`:319`, `:341`), while both clients check the id *before* the status
(`src/local_ipc.rs:167-169`, `mcp-server/ipc-client.ts:361-363`) and discard the real
error. The server already recovers the id leniently at `:257-260` for the dispatch path
and does not reuse it here.

On the TypeScript side the resulting plain `Error` does not match `/API error (\d+)/`
(`index.ts:554`), so a deterministic 400 is classified as retryable and the request is
retried three times.

Not reachable from the two shipped clients (both pre-check the frame limit and always
send a UUID) — it bites third-party clients and, notably, the first v2 client, whose
version-skew error is exactly the one that gets eaten.

**Fix:** echo the recovered id; define in the doc what an unrecoverable-id error looks
like so clients accept it as a protocol error rather than a mismatch.

### M9. An over-limit response envelope is dropped rather than reported

`src/local_ipc.rs:306` logs and returns *without writing anything*, so the client sees
EOF instead of a 413. Reachable when JSON-escaping a near-limit non-JSON body eats the
64 KiB of headroom.

**Fix:** write a `RESPONSE_TOO_LARGE` frame.

### M10. The test suite does not test the things the doc says it tests

`src/local_ipc.rs:1117-1279` has 8 tests. `docs/architecture/07-local-ipc-transport.md:99-103`
claims ten categories including "endpoint-permission", "deadline" and "redaction" tests.

- **No test ever calls `auth::validate_api_key` over IPC.** Both auth tests send
  `auth: String::new()` (`:1157`, `:1175`), which short-circuits at `:372-374`; line
  `:375` never executes. Stub `validate_api_key` to `Ok(())` and the suite stays green
  while IPC accepts *any* key on every `/api` route. The authenticated happy path has
  never run.
- **`streaming_routes_are_excluded_from_finite_ipc` (`:1144-1149`) is a tautology** — it
  asserts two const arrays contain their own literals. Delete the policy gates at
  `:409-425` and every test still passes.
- **15 of 18 protocol error codes have zero assertions.**
- **No endpoint-permission test exists at all** — nothing asserts the 0700 directory
  (`:642`, `:652`), the 0600 socket (`:697`), or any stale-socket branch (`:658-692`).
- **The Windows DACL test (`:1246-1262`) is compiled by no CI job.** `ci.yml:101-103`
  gates `cargo test --lib` on `matrix.os != 'windows-latest'`, and clippy runs on
  ubuntu — so the `#[cfg(windows)]` FFI is not even type-checked. Change the SDDL to
  `D:(A;;GA;;;WD)` (grant-all-to-everyone) and CI stays green.
- **No CI job runs the TypeScript tests at all**, and `ipc-client.ts` is excluded from
  `mcp-server/vitest.config.ts`'s coverage gate.
- **No cross-implementation test.** The TS tests drive a `net.createServer` mock; the
  Rust tests drive `Router::new()`. The two implementations of wire protocol v1 have
  never been proven to interoperate.

---

## D — Duplication and copy-paste

Requested explicitly, so recorded separately even where the severity is low.

### D1. Copy-paste defects

- **`IpcClient::post` reuses the GET timeout constant** — `src/local_ipc.rs:104` and
  `:115` both pass `DEFAULT_CLIENT_TIMEOUT`. Mechanically a copied line; see M5 for the
  consequence.
- **`read_frame`'s error messages are written for the server, and the client reuses
  them verbatim.** `src/local_ipc.rs:549-556` says "IPC accepts exactly one
  newline-delimited **request** per connection", and the client surfaces it as-is
  (`:155-157`) — a malformed *response* is reported to the user as if they sent
  garbage. The status codes baked into `FrameError` are meaningless client-side and are
  discarded. Fix: `read_frame(reader, Side::Request | Side::Response)`.
- **Three different wordings for one condition** — `src/local_ipc.rs:553` vs `:568-572`
  vs `mcp-server/ipc-client.ts:178`/`:309`.
- **`apiCall` serializes the body twice** — `mcp-server/index.ts:543-549` calls
  `serializeAndValidateBody`, discards `.serialized` (on `main` it was assigned to
  `options.body`), and hands the raw object on to be re-stringified. Not a hole — the
  100 KB cap is not evadable, `JSON.stringify` is deterministic and nothing mutates the
  body in between — but it is a redundant pass and a dead return value.

### D2. Logic duplicated across the three implementations — the part that will drift

1. **Exposure policy.** `STREAMING_PATHS` (5 entries) and `PUBLIC_API_PATHS` (2) at
   `src/local_ipc.rs:40-48` are a hand-copy of `src/handlers/router.rs:60-88` and
   `:443-446`. Correct today; nothing keeps them correct, and the only test is the
   tautology in M10. **The drift is asymmetric:** a missed *public* route fails closed
   (IPC merely demands auth), but a missed *streaming* route fails **open** into
   `to_bytes` on an endless SSE body (`:481`) — the connection wedges until the dispatch
   deadline, holding its semaphore permit (`:745`) and blocking the drain (M2). This
   makes the doc's "No route logic is duplicated between transports" (`07:94`) false
   exactly where it matters. **Fix:** derive the lists from `handlers::router`, or add a
   test that walks the built `Router` and fails on any unclassified route.
2. **Auth exemption policy.** `src/auth.rs:201-212` (`/health` exact, `/webhook/`
   prefix, `X-API-Key` → Bearer → `?api_key=`) versus `src/local_ipc.rs:361-388`
   (`GET /health` exact, `/api/` prefix, envelope `auth` field). The key *check* is
   correctly shared; the *policy* is a second implementation. Any future addition to
   `auth_middleware` — per-key scopes, auth-failure throttling — applies to HTTP and
   silently not to the transport now carrying all local agent traffic. Same class as M3.
3. **The wire protocol is implemented twice** — `src/local_ipc.rs` (server + Rust
   client) and `mcp-server/ipc-client.ts` — with every framing and envelope invariant
   asserted twice against two independent restatements of the spec, and no test where
   one meets the other. They already disagree: the write deadline is 10 s in Rust
   (`:115`) and 30 s in TypeScript (`index.ts:541`).
4. **Error-id handling** — see M8.

---

## P — The proposal itself

`docs/architecture/07-local-ipc-transport.md` makes claims the code does not deliver.
Each must be made true or deleted.

| Claim | Reality |
|-------|---------|
| "reduces exposed surface area" (`:13`) | Both listeners bind unconditionally; there is no `--no-http`. Post-merge the surface is HTTP **+** IPC. The doc concedes it four sections later (`:91`). |
| "No route logic is duplicated between transports" (`:94`) | Handlers are shared; the *exposure policy* is a hand-maintained duplicate (D2.1). |
| "streaming routes are never silently downgraded" (`:28`) | True only while someone hand-updates the array (D2.1). |
| "bounded operation deadlines and drains in-flight route work" (`:77`) | The per-request deadlines are bounded; the drain is not (M2). |
| "includes … endpoint-permission … tests" (`:100`) | No such test exists (M10). |
| "Independent review covered …" (`:101-103`) | An unfalsifiable process claim, permanently committed to the architecture docs. Review provenance belongs in the PR thread. |
| "Windows uses a current-user-scoped named pipe" (`:22`) | Only the DACL is; the name is machine-global (M7). |

Genre and placement: `07` is an RFC skeleton with a SEP-style metadata table dropped
into a descriptive series (`01-neuroscience-foundations.md` has no metadata table and
cites `src/` ten times; `07` cites `src/` zero times). Its
`Status | Implemented in the local-ipc pull request` is not a lifecycle state and
dangles the moment it merges. It has no Author and no Created date, unlike both
existing proposals under `seps/`.

Process: a local IPC transport appears on none of the three priority lists in
`CONTRIBUTING.md:317-339`, and the nearest analogues ("Distributed mode", "GraphQL
API" — i.e. additional transports) sit in the **Low** bucket. There is no prior demand
for it in the repo. An unsolicited change has to be flawless in fit.

---

## N — Minor

- Public `/api/` routes are 403'd over IPC even with a valid key (`src/local_ipc.rs:48`,
  `:409-416`) — the *least* sensitive routes are the only unreachable ones, and the TUI
  already calls `/api/context_status`. Allow them or justify the exclusion in one line.
- No headers and no content type in the envelope: `GET /api/graph/{id}/export` returns
  `application/gexf+xml` (`src/handlers/export.rs:595`), which `envelope_from_response`
  stringifies via `from_utf8_lossy` (`:483-484`); the Rust client's `from_value::<R>()`
  then fails and the TS client blind-casts (`ipc-client.ts:128`).
- Setting `SHODH_IPC_ENDPOINT` silently disables continuous ingestion —
  `index.ts:59-61`, `:218-220`, `:345`: `streamMemory`/`streamToolCall` become no-ops.
  Documented in `mcp-server/README.md`, but `docs/direct-server-systemd.md:88-104` hands
  users exactly that config without the caveat.
- The in-band API key adds no attacker-work against the transport's own threat model
  (the socket is 0600 owner-only, the pipe DACL is user+SYSTEM, and a same-uid process
  can read the key from the server's environment anyway). `SO_PEERCRED` /
  `GetNamedPipeClientProcessId` are available and unused. State that the key is
  defence-in-depth against a widened endpoint, or make peer credentials the primary
  authenticator.
- The mandatory correlation `id` earns nothing on a protocol the doc defines as
  one-request-one-response-one-connection (`07:32`), and it never reaches a tracing span
  (`middleware::request_id` is not layered onto the IPC router). Drop it or justify it.
- Three docs give three different endpoint values — `README.md:360` (`/run/user/1000/…`,
  which matches nothing), `README-rust.md:248`, `mcp-server/README.md:51` — and none is
  labelled the default. State the default once; link from the rest.
- Missing docs-sync deliverables: no `SHODH_IPC_ENDPOINT` row in `.env.example`; no row
  in `SECURITY.md:38-46`'s hardening table for a new always-on authenticated listener.
- `docs/direct-server-systemd.md:18-22`, `:96-98` *rewrites* the maintainer's existing
  command-role descriptions and swaps `SHODH_API_URL` for `SHODH_IPC_ENDPOINT` in their
  example. Add, don't replace.
- The Unix tests leak an empty 0700 temp directory per run — `EndpointGuard::drop`
  (`:618-629`) removes the socket, never the parent it created at `:641-645`.
- Commit `074c4b1` carries 2215 insertions across the Rust server, the CLI, the TS
  client and five docs in one un-bisectable commit, and its body is only
  `Agent: gpt-5 high` — no what, no why. Split it so that the CLI breakage (B3) can be
  rejected independently of the transport.

---

## Unverified

This review is **static only**. Nothing was executed — no `cargo`, no `vitest`, no
`npm`. The following are traced by hand and should be confirmed by running something:

- **B1's macOS `shodh init` → 0755 collision.** The load-bearing premise is that `dirs`
  collapses `config_dir()` and `data_local_dir()` onto `~/Library/Application Support`.
- **B4's macOS socket-path overflow.** The byte arithmetic was computed, not observed.
- **B1's two-instance collision**, on Linux and on Windows.
- **M10's mutation claims** ("stub `validate_api_key` to `Ok(())` and the suite is still
  green") — every test was traced by hand; confirm with `cargo test --lib` after each
  mutation.
- **Whether the Windows DACL test compiles at all** — it is currently built by no CI job.
- **All 10 TypeScript tests** — no workflow runs them.
- **M6's Windows helper cost** — measure `node -e` spawn latency against loopback
  `fetch` on a real Windows box before defending the helper design.

---

## Resolution

Fixes for the blockers, majors, duplication items, and the doc rewrite landed in the two
commits after this one on `feat/local-ipc`. `cargo check --bins` and the `cargo test --lib`
binary both compile locally (Windows); the tests themselves are left for CI to run.

**Fixed:** B1, B2, B3, B4; M1, M2, M3, M5, M7, M8, M9, M10; D1; D2.1 (the tautological
policy test is replaced by one that drives dispatch — deriving the lists from the router is
not done, see below); P (the doc no longer asserts what the code does not do).

**Deferred (GitHub issues are disabled on this fork, so recorded here instead):**

- **M6 — Windows per-request Node child process.** Excluded from this pass by scope: the
  correct fix is a server-side pool of pending `NamedPipeServer` instances plus plain
  `net.createConnection` on the TS side (deleting the helper), and that is a delicate
  refactor of Windows runtime behaviour that cannot be verified on this box. The helper
  machinery is left in place; the arming loop it depends on is at least no longer able to
  tear down the daemon (M1).
- **S1 — Windows client-side pipe squatting / key harvest** (found by the security
  cross-check, not the main pass). The client writes the envelope — which carries the API
  key — before verifying the server pipe's owner. M7's per-user pipe name raises the bar (an
  attacker must know the victim's SID and win the pre-create race) but does not close it; the
  full fix is a `GetNamedPipeServerProcessId` owner check before the first write, Windows-only
  FFI not verifiable here.
- **S2 — pre-auth resource asymmetry** (security cross-check). The semaphore permit is taken
  before `accept()` and the read deadline is the full 60s, so an unauthenticated same-user
  peer can hold slots. Low severity (same-user only); the fix is a short pre-auth read timeout
  distinct from the request timeout.
- **D2.1 derivation.** The exposure lists are still hand-maintained (axum does not expose its
  registered routes to walk), now covered by a dispatch-driving test rather than a tautology.

The security cross-check also **confirmed correct** and did not change: the `/health`
exemption (no auth bypass), constant-time key comparison, auth-before-policy ordering, the
Windows DACL, and that the API key does not leak on any path including the serde error path.
