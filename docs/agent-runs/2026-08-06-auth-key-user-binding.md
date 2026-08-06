# Agent run record — bind API keys to user identity (T18)

**Date:** 2026-08-06 · **Branch:** `fix/auth-bind-keys-to-users`
**Commits:** `694371da` (implementation), `a56ace35` (verification fixes)

The diff survives in git; the reasoning does not. This records what was proved,
what the verification pass found that the change itself missed, and what was
deliberately left unfixed — so none of it is re-derived or mistaken for oversight.

## The defect

`validate_api_key(key) -> Result<(), AuthError>` answered "is this key valid",
never "who is this". `user_id` arrived from the request body. Net effect: **every
valid API key was root over every user.** This blocked multi-seat entirely.

## The change

`AuthIdentity{Unscoped, User(String)}`; `resolve_api_key`; opt-in
`SHODH_SCOPED_API_KEYS` (`user_id:key`, comma-separated); `auth_middleware`
attaches identity to request extensions; `scope_enforcement_middleware` authorizes
the named `user_id` against it. Layered on HTTP **and** IPC.

Legacy posture is the default: with no scoped keys configured, the table is
unscoped-only with precedence identical to `origin/main`, and the scope middleware
passes `Unscoped` through without buffering the body.

## Verification — adversarial, briefed to refute

| Claim | Verdict | Evidence |
|---|---|---|
| Every per-user route covered | **REFUTED → fixed** | `GET /api/sessions/{id}` ignored its required `user_id`; `Session` is keyed by `SessionId` only. A scoped key could read any user's session by UUID. Fixed in `a56ace35`. |
| Bypass via Form/multipart/alt field names | **REFUTED (none)** | Zero `Multipart`/`Form<` in `src/handlers/`; all bodies `Json<T>`. No alternate user-id field names. |
| Body-limit differential (scoped 413 where unscoped passes) | **REFUTED** | No `DefaultBodyLimit`; `MAX_SCANNED_BODY_BYTES` = 2MB = axum's `Json` default. Aligned. |
| Layer ordering / extension stripping | **REFUTED (safe)** | `server.rs:265-269` — last `.layer()` is outermost, so `auth_middleware` runs first and always inserts identity. Early-returns are only `/health` and `/webhook/*`, neither protected. Typed extensions cannot be spoofed over HTTP. |
| Malformed scoped entries fail closed | **PARTIALLY REFUTED** | True alone. But a malformed scoped entry for key `K` **plus** `K` in `SHODH_API_KEYS` resolves `K` as `Unscoped` — fail-open vs intent. Compound misconfig; now pinned by a test. |
| Parsing edge cases | **CONFIRMED** | `validate_user_id` permits only `alphanumeric,-,_,@,.` so `:` is genuinely forbidden; `split_once(':')` lets keys contain `:` but not user IDs. Exact `str` equality — no case/unicode normalization, mismatches fail closed. |
| No enumeration | **MOSTLY** | `/api/users` denied for scoped keys. Residual: `get_session_stats` returns a global `users_with_sessions` count and takes no `user_id`, so the middleware structurally cannot see it. Metadata only. |
| Byte-identical with no scoped keys | **CONFIRMED** | Unscoped precedence replicates `origin/main` exactly; prod never falls back to the dev key. |
| MCP / hooks unbroken | **CONFIRMED** | Both use `SHODH_API_KEY` → `Unscoped` → pass-through. |
| Probe signing intact | **CONFIRMED** | `configured_api_keys` only *adds* proofs; frame cap returns 413, not silent EOF — not a T16 silent-capture risk. |
| 11 added tests fail on revert | **CONFIRMED** | All substantive; no tautologies, no `Default` assertions. |

Result: **42 auth tests + 2 session tests green**, `fmt` clean, `clippy --lib` clean.

## Deliberately not fixed

- **`get_session_stats` global tenant count.** Pre-existing, no `user_id` parameter,
  structurally invisible to the middleware. Metadata, no identities or content.
  Product decision, not a T18-scope bug.
- **Compound-misconfig fail-open** (malformed scoped entry + same key unscoped).
  Requires two mistakes at once. Pinned by test so a future change is deliberate.

## Operator footgun — document before shipping

If the shared MCP/hooks key is ever added to `SHODH_SCOPED_API_KEYS`, those clients
are silently confined to one user. Capture would degrade invisibly — the T16 failure
shape. This needs an explicit note in the env-var docs.

## Environment facts (cost an agent hours; do not rediscover)

- **Builds inside the repo path fail deterministically**: `cl.exe` exits 1 with no
  diagnostic while compiling the same rocksdb artifact
  (`remove_emptyvalue_compactionfilter.o`), 5/5 retries. All green results above required
  `CARGO_TARGET_DIR` pointed outside the repo tree.

  **CAUSE NOT IDENTIFIED — do not repeat the original guess.** The verifying agent
  attributed this to OneDrive because the repo path contains `\OneDrive\`. That was an
  inference from a directory name, and it is **wrong**: no OneDrive process runs on this
  machine, and the folder carries no reparse-point or cloud-file attributes — it is an
  ordinary directory left from a previous setup.

  Candidates, none confirmed: Windows Defender real-time scanning (verified ON, no visible
  exclusions; scanning freshly-written object files is a well-known cause of exactly this
  silent `cl.exe` failure), or contention from several agents building concurrently.
  Whoever hits this next: identify the actual locker (e.g. Process Explorer / handle.exe)
  before acting, rather than inheriting a guess.
- **`cargo clippy --all-targets` is red on `origin/main`**, independent of this branch:
  `absurd_extreme_comparisons` deny-errors in `tests/brutal_stress_tests.rs:687,1303`
  (`unsigned >= 0`). Same class as the one fixed in #426 — that PR fixed one file, not
  the class. An all-targets clippy gate cannot currently pass; a lib-scoped one is clean.
