# Local IPC Transport

| Field | Value |
|-------|-------|
| Status | Implemented (enabled by default) |
| Author | Alex Yanchenko |
| Created | 2026-07-14 |
| Scope | Rust engine, Rust/TypeScript MCP clients, local integrations |
| Platforms | Unix-domain sockets and Windows named pipes |
| Protocol | Version 2, newline-delimited JSON + HMAC endpoint proof |

## Problem

The REST API is the right transport for remote clients, but local MCP and hook
integrations do not need to go through an HTTP listener. An authenticated local
transport avoids per-client URL configuration drift and lets the existing Axum
route table stay the single operation catalog. It is **additive**: it does not
remove the HTTP listener, so it does not by itself shrink the process's exposed
surface — it adds a second, OS-access-controlled entry point that local clients
can prefer.

## Proposal

When enabled, the engine binds one local endpoint at startup and dispatches local
requests through the same Axum router used by REST. By default, Unix uses a
filesystem socket under the platform data directory; Windows uses a named pipe
whose name is per-user and whose DACL grants the current user plus LocalSystem.
The endpoint is configurable
(`SHODH_IPC_ENDPOINT`) and independent of the memory storage path.

Default endpoints are:

- Linux: `${XDG_DATA_HOME}/shodh/shodh-memory.sock` when `XDG_DATA_HOME` is set,
  otherwise `~/.local/share/shodh/shodh-memory.sock`.
- macOS: `~/Library/Application Support/shodh/shodh-memory.sock`.
- Windows: `\\.\pipe\shodh-memory-<current-user-SID>` (falling back to the bare
  pipe name only when the SID lookup fails).

IPC is **enabled by default and non-fatal**:

- `SHODH_IPC_ENABLED=false` (also `0`, `no`, or `off`) disables the listener.
- A bind failure is logged and the process continues serving HTTP only — it never
  aborts startup. This is deliberate: a second instance for the same user, a data
  directory not at mode 0700, or a read-only endpoint parent must not take the
  whole daemon down.
- `SHODH_IPC_REQUIRED=true` switches both the server and native Rust client to
  fail-closed behavior. A disabled or unbindable listener aborts server startup,
  and the client refuses HTTP fallback. The TypeScript client also requires an
  explicit `SHODH_IPC_ENDPOINT` when this mode is set.

The transport is request/response only. Streaming routes remain on their existing
WebSocket / server-sent-event transports and are refused (`501`) over IPC rather
than being downgraded to a finite response.

## Wire contract

Each connection carries at most one request and one response. The server closes
the connection after the response, so a delayed second request can never cause a
second route dispatch.

Request envelope:

```json
{
  "v": 2,
  "id": "client-correlation-id",
  "auth": "request:<server-nonce>:<hmac-sha256>",
  "challenge": "",
  "method": "POST",
  "path": "/api/recall",
  "body": {}
}
```

Response envelope:

```json
{
  "v": 2,
  "id": "client-correlation-id",
  "auth": "response:<hmac-sha256>",
  "status": 200,
  "body": {}
}
```

Before an ordinary request, a client sends exact `GET /health` with empty `auth`
and a random 32-byte `challenge`. The server returns its per-process nonce plus an
HMAC proof for each configured API key. A client continues only when its key
verifies one proof. Ordinary requests then carry an HMAC bound to that server
nonce, request ID, method, path, and raw JSON body. The reusable API key is never
put on the IPC wire. Responses carry a proof bound to the request proof, ID, and
status.

Frames are UTF-8 JSON terminated by a newline and bounded at eight MiB. The
version is mandatory. Clients reject malformed UTF-8, unknown envelope fields,
invalid status codes, missing frame delimiters, trailing data, oversized frames,
and missing or invalid server proofs. A response whose `id` is empty is a
pre-authentication protocol error; authenticated clients reject it as untrusted.

## Security and lifecycle

- Only an exact `GET /health` probe has empty `auth`. A challenged probe proves
  the server knows a configured key without disclosing that key; an unchallenged
  probe remains available for basic local health tooling.
- All ordinary `/api/*` routes verify their keyed request proof **before** method,
  route-policy, or body-value processing. The body remains `RawValue` until that
  check succeeds.
- Public context-status, streaming, webhook, and HTML routes are not exposed
  through IPC.
- Unix: the endpoint's parent directory is opened with `O_DIRECTORY|O_NOFOLLOW`,
  verified as current-user-owned, and tightened by handle to 0700; the socket is
  0600. Both sides assert peer UID. Stale-socket cleanup checks ownership, type,
  liveness, and device/inode identity before unlinking.
- Windows: the pipe uses `FILE_FLAG_FIRST_PIPE_INSTANCE`, rejects remote clients,
  and carries a protected DACL granting access only to the current user and
  LocalSystem. Rust clients explicitly request `SECURITY_IDENTIFICATION`; both
  sides verify the peer process account. The TypeScript client routes pipe I/O
  through the bundled Rust binary because Node's file APIs cannot set those SQOS
  flags safely.
- At most 16 connections may occupy the pre-authentication stage, each for at most
  two seconds. Authenticated dispatch has its own configured concurrency limit.
  Shutdown drain remains bounded by the HTTP drain deadline.

## Client integration

The Rust MCP client (`shodh serve`) prefers IPC when the endpoint passes the
authenticated challenge and falls back to HTTP at `SHODH_API_URL` otherwise,
unless `SHODH_IPC_REQUIRED=true` selects fail-closed behavior.

The TypeScript MCP client selects IPC when `SHODH_IPC_ENDPOINT` is set and retains
HTTP behavior when it is absent. A `\\.\pipe\` endpoint on a non-Windows platform is
rejected rather than forwarded to the auto-spawned backend. WebSocket streaming in
IPC mode requires explicit opt-in (`SHODH_STREAM_WEBSOCKET=true`).

## Known limitations

- **Manually classified routes.** The streaming and public-route exclusion lists
  are maintained beside the IPC transport because Axum does not expose registered
  routes for enumeration. Dispatch-driving tests cover the current lists, but a new
  streaming or public HTTP route still requires an explicit IPC exclusion.
- **Middleware divergence.** IPC dispatches into the route table with only
  `track_metrics` layered on. Rate limiting, CORS, and security headers apply to
  HTTP only; auth, bounds, and concurrency are enforced inside the IPC layer. New
  HTTP middleware does not automatically cover IPC.
- **Response shape and size.** IPC carries a JSON value and no HTTP headers or
  content type, so non-JSON exports are not equivalent to their HTTP responses.
  It also caps a response body at about eight MiB; the same route over HTTP has no
  such cap and can succeed where IPC returns `413`.
- **Logical namespaces are not authorization tenants.** `user_id` selects a memory
  namespace. API keys remain full-authority credentials across namespaces for both
  HTTP and IPC; peer UID/SID cannot be mapped safely to application identifiers
  such as `claude-code` or a robot ID. Deploy separate server/key instances when
  mutually untrusted tenants require isolation.
- **Windows client shape.** The TypeScript client uses a bounded pool of helper
  processes so Windows pipe I/O stays killable and receives Rust's SQOS and peer
  checks. This limits throughput, and queued time counts against the request
  deadline.

## Verification

The suite includes challenge/response and request-proof parity, key non-disclosure,
body-tamper rejection, authentication order, exact-`GET /health`, frame boundaries,
delayed-second-frame handling, streaming/public-route policy, API-key redaction,
and platform-gated endpoint permission/DACL and peer-account checks. The Rust and
TypeScript implementations exercise the same length-prefixed HMAC field framing.
