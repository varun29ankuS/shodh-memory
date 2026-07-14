# Local IPC Transport

| Field | Value |
|-------|-------|
| Status | Implemented (enabled by default) |
| Author | Alex Yanchenko |
| Created | 2026-07-14 |
| Scope | Rust engine, Rust/TypeScript MCP clients, local integrations |
| Platforms | Unix-domain sockets and Windows named pipes |
| Protocol | Version 1, newline-delimited JSON |

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
  "v": 1,
  "id": "client-correlation-id",
  "auth": "configured-api-key",
  "method": "POST",
  "path": "/api/recall",
  "body": {}
}
```

Response envelope:

```json
{
  "v": 1,
  "id": "client-correlation-id",
  "status": 200,
  "body": {}
}
```

Frames are UTF-8 JSON terminated by a newline and bounded at eight MiB. The
version is mandatory. Clients reject malformed UTF-8, unknown envelope fields,
invalid status codes, missing frame delimiters, trailing data, and oversized
frames. A response whose `id` is empty is treated as a server-side protocol error
(the server could not recover a correlation id from the request) and surfaced via
its status, **not** rejected as an id mismatch.

## Security and lifecycle

- Only an exact unauthenticated `GET /health` probe is public. The exemption
  matches the raw request path byte-for-byte, which is strictly narrower than the
  parsed `/api/` gate, so it cannot be widened into an `/api` route by dot-segments
  or encoding.
- All ordinary `/api/*` routes authenticate with the configured API key (constant-
  time comparison, `auth::validate_api_key`) **before** method or route-policy
  evaluation.
- Public context-status, streaming, webhook, and HTML routes are not exposed
  through IPC.
- Unix: the endpoint's parent directory is owned by the current user and forced to
  mode 0700 (tightened if it pre-exists looser); the socket is 0600. Stale-socket
  cleanup checks ownership, type, liveness, and device/inode identity before
  unlinking.
- Windows: the pipe uses `FILE_FLAG_FIRST_PIPE_INSTANCE`, rejects remote clients,
  and carries a protected DACL granting access only to the current user and
  LocalSystem. The default pipe name includes the current user's SID so different
  users do not collide. A SID is discoverable and the name does not by itself stop
  pipe squatting; client-side owner verification remains a known limitation.
- Per-request read and dispatch deadlines are bounded, and the IPC drain at
  shutdown is bounded by the same timeout as the HTTP drain, so a slow route cannot
  hold the process open and cost the storage flush.

## Client integration

The Rust MCP client (`shodh serve`) prefers IPC when the endpoint answers a health
probe and falls back to HTTP at `SHODH_API_URL` otherwise — so `SHODH_API_URL`
keeps working and a client never hard-fails every tool call merely because IPC is
unavailable while HTTP is up.

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
- **In-band key.** On the legitimate 0600 socket or DACL-protected pipe, OS access
  is already user-scoped, so the API key is defense-in-depth (e.g. against a widened
  endpoint), not an independent second factor. Peer-credential checks (`SO_PEERCRED` /
  `GetNamedPipeClientProcessId`) are not yet used.
- **Windows client shape.** The TypeScript client uses a bounded pool of helper
  processes because Node's named-pipe connect cannot be cancelled reliably. This
  limits throughput, and queued time counts against the request deadline.
- **Windows server-owner verification.** Clients do not verify the named-pipe
  server owner before sending the in-band API key. The per-user name prevents
  cross-user collisions but does not close this pre-creation race.
- **Pre-auth occupancy.** A same-user peer can occupy a concurrency slot for the
  full request-read timeout before authenticating; there is no shorter pre-auth
  deadline.

## Verification

The suite includes protocol, authentication-order and key-enforcement (valid and
wrong key over the real validator), exact-`GET /health` probe, frame-boundary,
delayed-second-frame, streaming/public-route policy (driving dispatch, not the
const arrays), protocol-error-code, API-key redaction, and — platform-gated — a
Unix endpoint-permission test (0700 parent, 0600 socket) and a Windows DACL test.
A cross-implementation test that runs the real server against the real TypeScript
client does not yet exist; the two protocol implementations are verified
independently.
