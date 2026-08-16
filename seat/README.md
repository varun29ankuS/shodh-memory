# seat — conversation harness

Server-side agent harness for the shodh-memory conversation seat. It owns the
agent loop, streams structured events to a client, and closes two learning
loops against the Rust backend on every turn. Memory operations are never
opaque: every recall carries ids, scores, and full per-memory
`ScoreAttribution`; every learning update is a reviewable, revertible ledger
event.

Built on [`pi`](https://github.com/earendil-works/pi) (`@earendil-works/pi-agent-core`
for the agent loop, `@earendil-works/pi-ai` for the unified multi-provider LLM
API). Provider credentials resolve from environment variables inside this
process and never reach a client.

## Architecture

```
client (SSE) ──► SeatServer (node:http)
                    │
                    ├─ ModelRegistry ── pi builtin providers (env-key auth)
                    │                └─ local providers: Ollama / LM Studio / vLLM
                    │                   (custom pi provider, openai-completions
                    │                    API + per-model baseUrl)
                    │
                    ├─ Conversation ── pi Agent (loop, tools, streaming)
                    │     ├─ memory tools (native HTTP → Rust backend)
                    │     ├─ MCP host tools (stdio / streamable HTTP / SSE)
                    │     └─ learning loops (see below)
                    │
                    ├─ ShodhBackend ── HTTP client for the Rust API
                    │                  (X-API-Key, shapes transcribed from
                    │                   src/handlers/*)
                    └─ LearningLedger ── append-only JSONL, revert support
```

## Memory as a first-class tool

`recall_memory` calls `POST /api/recall` with `debug: true`, so every result
carries the backend's full score attribution (RRF base, graph/hybrid RRF,
hebbian boost, importance/recency/arousal/credibility factors,
`feedback_multiplier`, quality gate, final score, contributing sources). The
tool returns a compact text rendering to the model and emits a
`memory_recall` SSE event with the complete structured payload — retrieved
ids, scores, attribution, related facts, todos, and lineage edges — for the UI
to render as its own element.

The system prompt asks the model to cite memories inline as `[mem:<id>]`;
citations feed the reinforcement loop below.

## Two learning loops, one substrate

Both loops store and strengthen state through the same shodh-memory machinery.
There is no second state store.

**Loop 1 — memory-level (user scope), two legs with strict ownership.**
`/api/reinforce` moves importance, Hebbian associations, and entity salience —
but **not** feedback momentum. The only backend path that writes momentum (the
`feedback_multiplier` in score attribution) is `POST /api/proactive_context`
(`src/handlers/recall.rs:1310-1720`). The seat drives both:

- *Implicit/momentum leg* — every turn calls `proactive_context` with the new
  user message as `context`, the previous assistant message as
  `previous_response`, the new user message as `user_followup`, and the
  previous run's tool actions (`ToolAction` shape from
  `src/memory/feedback.rs`; native memory tools excluded — a recall cue
  trivially overlaps memory content and would fake a usage signal). The
  backend evaluates the previous turn's pending surfaced set (momentum,
  context fingerprints, temporal credits) and applies its own
  `reinforce_recall` + Hebbian pass for helpful/misleading classifications.
  The memories it surfaces are all injected into the system prompt (surfaced
  set == seen set, otherwise the implicit loop penalizes memories the model
  never saw) and are **owned by this leg**. The `proactive_context` SSE event
  exposes the surfaced set and the feedback outcome (reinforced/weakened ids).
- *Explicit leg* — memories recalled by the `recall_memory` tool and **not**
  proactive-surfaced that turn are reinforced through `POST /api/reinforce`:
  - *helpful* — cited (`[mem:id]`) or token overlap ≥ 0.1 (mirrors
    `calculate_entity_overlap` / `OVERLAP_WEAK_THRESHOLD`);
  - *neutral* — surfaced but unused;
  - *misleading* — negative follow-up keywords (verbatim `NEGATIVE_KEYWORDS`
    list), applied to the previous turn's surfaced set minus proactive-owned
    ids.

The id-level ownership split is what prevents double-counting: a memory
surfaced by both channels in one turn is reinforced exactly once (by the
implicit leg). Known seam: tool-recalled-only memories get no momentum, because
the backend ties momentum to its own pending-set lifecycle
(`set_pending`/`take_pending`, single slot per `user_id`) — moving them would
require a backend change, not a harness workaround. For the same reason the
seat guards concurrent proactive calls per `user_id` in-process (feedback
fields are skipped when a call is in flight, mirroring `mcp-server/index.ts`);
a second process using the same `user_id` cannot be guarded from here.

`auto_ingest` is explicitly `false` on every call: the backend default (true)
silently ingests the previous response as memories, which would bypass the
ledger. Seat writes stay deliberate and ledgered.

**Loop 2 — harness-level (harness scope).** The harness gets better at its own
job by storing operational lessons *as memories* in an isolated namespace,
`<user_id>.seat-harness`. Scope isolation uses the backend's existing
per-`user_id` seam: memory store, knowledge graph, and feedback state are all
keyed by `user_id` directory (`src/handlers/state.rs`), so the two scopes can
never share retrieval, Hebbian co-activation, or feedback statistics.

Harness learnings enter the scope three ways:

- automatic capture of empty recalls (cue recorded with rephrasing advice);
- automatic capture of tool failures;
- the model's own `record_seat_learning` tool (operational lessons only).

Before each turn the harness recalls from its scope with the user message as
cue and injects strong matches (score ≥ 0.25, max 3) as a labeled
system-prompt block for that turn only, emitting `harness_learning_applied`.
Injected learnings are reinforced by the same rules as user memories — the
loop is closed in both scopes.

## Learning ledger

Every update either loop makes is appended to a JSONL ledger
(`<data-dir>/learning-ledger.jsonl`) *before* the conversation moves on:
memory writes, every reinforcement (with per-memory overlap values and
trigger), and reverts. Reverts are appended events referencing the original —
nothing is mutated.

Revert semantics are honest about what the backend supports:

- memory writes revert exactly (`DELETE /api/memory/{id}`);
- helpful/misleading reinforcements revert by a *compensating* opposite
  outcome through the same `/api/reinforce` path — the backend's EMA-with-
  inertia momentum update is not exactly invertible, and the revert event says
  so;
- neutral reinforcements record access only; nothing to compensate.

Every entry carries an `actor` — `agent` (the model emitted the tool call),
`system` (an automatic seat loop: citation/overlap/negative-followup
reinforcement, the backend's implicit-feedback pass, a deterministic harness
capture) or `user` (a human acting through the HTTP surface, today only
`POST /v1/learning/revert`). This is orthogonal to `scope`, which names the
memory namespace touched, not the initiator; an `actor: "agent"`,
`scope: "user"` entry is the model writing into the human's memory.

Entries appended before `actor` existed have no such field and are **not**
backfilled — inferring an actor after the fact and recording it as fact is the
kind of invention an audit log exists to prevent. The read path reports them
as `unknown`.

## Audit trail

The ledger records *changes to memory state*. Two other things a reviewer needs
are recorded elsewhere and were previously readable only one conversation at a
time: which tool ran when (`tool_call_start`/`tool_call_end`) and what was
retrieved with what scores (`memory_recall`, `proactive_context`). Both live in
the `events` table of `<data-dir>/seat.db`, persisted atomically at the end of
each turn — so an aborted process loses at most the turn in flight.

`GET /v1/audit/export` merges all three into one trail, sorted by
`(ts, source, ref)`. That is a total order, so exporting the same window twice
is byte-identical and two exports can be diffed. Row shape:

| column | meaning |
|---|---|
| `ts` | ISO-8601 UTC |
| `source` | `ledger`, `tool_call` or `retrieval` |
| `actor` | `user` / `agent` / `system` / `unknown` |
| `kind` | ledger kind, tool name, or event type |
| `user_id`, `conversation_id`, `turn` | where it happened |
| `ref` | ledger entry id, tool call id, or memory-operation identity |
| `detail` | JSON: tool arguments and duration, or the scored result set |

A tool call that never returned is kept with `ended_at`, `duration_ms` and
`is_error` all null — an invoked-and-never-returned tool is exactly what a
reviewer is looking for, and `is_error: false` would assert a success that
never happened.

Recorded tool-call `detail` holds the arguments, not the result: results are
forwarded to the backend's feedback pass and are not persisted seat-side.

## HTTP API

| Method | Path | Body / notes |
|---|---|---|
| GET | `/healthz` | seat + backend health (unauthenticated) |
| GET | `/v1/models?refresh=1` | available models; `refresh` re-probes local endpoints |
| GET | `/v1/providers` | provider auth status — configured/source/stored, never key material |
| PUT | `/v1/providers/{id}/key` | `{api_key}` — stored server-side (`provider-credentials.json`, 0600); a stored key beats env in pi's resolution order |
| DELETE | `/v1/providers/{id}/key` | remove the stored key; env-var auth, if present, remains |
| POST | `/v1/providers/{id}/oauth/start` | run pi's browser-OAuth login for the provider; SSE stream of `oauth_notify` (auth URLs, device codes, progress), `oauth_prompt` (pasted codes, selections), `oauth_complete`/`oauth_error`. One at a time per provider; disconnecting aborts |
| POST | `/v1/providers/{id}/oauth/input` | `{prompt_id, value}` — answer a pending login prompt |
| GET | `/v1/mcp/servers` | every configured MCP server: status, transport, tool list, connection error, endpoint (query-stripped) and the NAMES of any auth headers sent |
| POST | `/v1/mcp/servers/{name}/reconnect` | re-run one server's connection; returns its post-attempt state, failure included |
| GET | `/v1/conversations?user_id` | persisted session list with turn counts and accumulated token/cost totals |
| POST | `/v1/conversations` | `{user_id, provider, model, system_prompt?}` |
| GET | `/v1/conversations/{id}` | metadata + transcript + durable events (evidence replay) |
| PATCH | `/v1/conversations/{id}` | `{title}` — rename |
| DELETE | `/v1/conversations/{id}` | delete conversation, transcript and events |
| POST | `/v1/conversations/{id}/messages` | `{text}` → SSE stream of events |
| PATCH | `/v1/conversations/{id}/model` | `{provider, model}` — swap model mid-conversation; transcript and retrieved evidence unchanged |
| GET | `/v1/learning/events?limit&conversation_id` | ledger review |
| POST | `/v1/learning/revert` | `{event_id}` |
| GET | `/v1/audit/tool-calls?user_id&conversation_id&tool_name&since&until&limit` | tool invocations across every conversation, start/end joined into one record with a duration |
| GET | `/v1/audit/export?format=jsonl\|csv&user_id&conversation_id&since&until` | the merged audit trail as a downloadable file; `X-Audit-Rows` carries the row count |

SSE event types: `turn_start`, `text_delta`, `thinking_delta`,
`tool_call_start`, `tool_call_end`, `memory_recall`, `proactive_context`,
`memory_write`, `memory_reinforce`, `harness_learning_applied`,
`model_changed`, `usage`
(per-call token counts and cost from pi's `Usage`), `turn_end`, `agent_end`,
`error`. Payloads are defined in `src/events.ts`.

## Local models

pi ships no Ollama/LM Studio/vLLM provider, but every pi `Model` carries its
own `baseUrl` and the `openai-completions` implementation dials it directly.
The registry therefore registers three custom providers (`ollama`, `lmstudio`,
`vllm`) built with `createProvider` + `openAICompletionsApi()`, keyless auth,
and dynamic model discovery via `GET {baseUrl}/models`. Any other
OpenAI-compatible endpoint works the same way.

All three are listed in `LOCAL_PROVIDER_IDS` (`src/models-registry.ts`), and
that membership is what makes them keyless, billed as `none` and flagged
`local` — those three properties are derived from the list rather than restated
per provider, so registering another local endpoint is a one-line change plus
its base-URL config knob.

## MCP host

pi has no MCP client (its README: "No MCP"), so `src/mcp.ts` is the seat's
own: it connects to MCP servers, lists their tools, and exposes each one to the
agent loop as a pi `AgentTool` named `mcp__<server>__<tool>` with the server's
plain JSON Schema passed through (pi's validator handles non-TypeBox schemas).
Configure with `SEAT_MCP_SERVERS=/path/to/servers.json`:

```json
{
  "servers": [
    { "name": "shodh-memory", "command": "node", "args": ["mcp-server/dist/index.js"] },
    { "name": "issues", "url": "https://mcp.example.com/mcp",
      "headerEnv": { "Authorization": "ISSUES_MCP_TOKEN" } }
  ]
}
```

**Transports.** Three, matching what `@modelcontextprotocol/sdk` ships:
`stdio` (a command this machine runs), streamable HTTP (`"transport": "http"`,
the current standard for remote servers) and HTTP+SSE (`"transport": "sse"`,
superseded and deprecated in the SDK, but still the only thing many deployed
servers speak). A remote server defaults to `"auto"`: streamable HTTP first,
falling back to SSE only on 404/405/406 — the statuses that mean "this endpoint
does not implement that verb". A 401/403/429 or a dead host is reported as
itself rather than retried over an older transport.

**Credentials.** `headers` sets request headers verbatim; `headerEnv` maps a
header name to an environment variable read in this process, which keeps the
token out of the config file. A named variable that is not set fails the
connection with the variable's name, rather than dialling out unauthenticated.
Header **values** never leave the process: `GET /v1/mcp/servers` reports header
NAMES, and a server's URL is reported with its query string and any userinfo
stripped, because `?key=…` is a real way MCP endpoints are authenticated.

**Lifecycle.** Every configured server is retained with a status whether or not
it connected — `connecting`, `ready`, `failed` (never came up) or
`disconnected` (was up, went away) — because a dead server and a server with no
tools are indistinguishable if all you keep is a tool count. Connections are
established behind the HTTP listener, so a hung endpoint cannot delay startup.
A stdio server's stderr is piped and its tail is folded into the failure
message (a child that cannot start says why there and nowhere else); the tail
is cleared once the server is working, so a later disconnect quotes the crash
rather than the startup banner. `notifications/tools/list_changed` re-lists
that server's tools in place, and the agent re-reads the bridged tool set at
the start of every turn, so a tool list is not frozen at boot.

There is **no automatic reconnection loop**, deliberately. Two of the three
ways a server dies — a command that does not exist, a credential that is not
accepted — can never be fixed by retrying, and the third, a restarted endpoint,
is one `POST /v1/mcp/servers/{name}/reconnect` away from a surface that is
already showing the failure. A background retry against a crash-looping child
would hide the problem and add load. A tool call that arrives for a server that
is down fails with a named `McpServerUnavailableError` naming the server and
the remedy, not a socket error from three layers down.

The native memory tools remain the recall/remember path of record — they carry
attribution that MCP text framing cannot.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SHODH_API_URL` | `http://127.0.0.1:3030` | Rust backend (or `SHODH_HOST`/`SHODH_PORT`) |
| `SHODH_API_KEY` | — (required) | backend `X-API-Key` (falls back to `SHODH_DEV_API_KEY`, `SHODH_API_KEYS`) |
| `SEAT_HOST` / `SEAT_PORT` | `127.0.0.1` / `3141` | bind address |
| `SEAT_AUTH_TOKEN` | — | bearer token; mandatory for non-loopback binds |
| `SEAT_DATA_DIR` | `%LOCALAPPDATA%\shodh\seat-harness` (win) / XDG data dir | ledger location — deliberately outside watched/synced folders |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | Ollama OpenAI-compat endpoint |
| `LMSTUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio endpoint |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | vLLM endpoint (`vllm serve` default port) |
| `SEAT_LOCAL_CONTEXT_WINDOW` / `SEAT_LOCAL_MAX_TOKENS` | `32768` / `8192` | advertised limits for local models |
| `SEAT_MCP_SERVERS` | — | path to MCP servers JSON |
| `SEAT_MCP_CONNECT_TIMEOUT_MS` | `30000` | per-server budget to start, handshake and list tools (an `npx`/`uvx` server may be fetching its own package on first run) |
| Provider keys | — | pi's env conventions (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, …) |

## Build and run

```
npm install
npm run build
SHODH_API_KEY=... node dist/index.js
```

Requires Node ≥ 22.19 (pi's floor).

## Persistence

Conversations are durable across restarts. One SQLite database
(`<data-dir>/seat.db`, `node:sqlite`, WAL) holds three tables:

- `conversations` — metadata plus accumulated token/cost totals per
  conversation, so the session list shows real numbers without replaying
  transcripts;
- `transcripts` — the pi `AgentMessage[]` snapshot after each turn, which
  re-seeds `Agent.state.messages` when a conversation is reopened;
- `events` — every SeatEvent except the delta streams (whose final form lives
  in the transcript), so the UI can rebuild the full evidence surface —
  recalls with attribution, proactive context, reinforcements, ledger ids —
  for any reopened conversation.

Each turn persists atomically (transcript + events + totals in one
transaction), including aborted turns. Live `Conversation` objects are a cache
over the store; a conversation not live in this process is rehydrated on its
next message. If its stored model no longer resolves (key removed, local
endpoint gone), reads still work and message attempts get a 409 naming the
remedy: switch the model.

pi's `@earendil-works/pi-session-backend-sqlite-node` was evaluated and not
used: it implements pi-agent-core's `SessionRepository` — cwd-keyed session
trees with entry lanes, branch caches and leases — for pi's own session layer,
which the seat does not use, and it has no representation for seat events, so
adopting it would still have required a second store for the evidence stream.

## Provider credentials

`FileCredentialStore` (`src/credentials.ts`) implements pi's `CredentialStore`
over `<data-dir>/provider-credentials.json` (0600, temp-file + rename writes)
and is injected into `builtinModels({credentials})`. pi resolves a stored
credential before ambient env vars, so a key submitted through
`PUT /v1/providers/{id}/key` becomes the working credential immediately, and
`DELETE` falls back to env. `GET /v1/providers` reports, per provider: whether
auth is configured, pi's own source label (`ANTHROPIC_API_KEY`, `OAuth`,
`local endpoint (keyless)`, …), whether a seat-stored key exists, whether
the provider meaningfully accepts an API key at all (ambient-only providers
like Bedrock/Vertex do not), and whether pi ships a browser OAuth flow for it
— including whether that flow is subscription-backed (`isSubscription`:
Claude Pro/Max, ChatGPT, Copilot, …). Key material never appears in any
response.

Auth is three genuinely different shapes, and `/v1/models` labels every model
with the consequence under its *effective* credential (`billing`):
`"none"` (local endpoint — nothing leaves the machine), `"subscription"`
(OAuth flat-rate plan — token counts are plan consumption, pi's per-token
cost numbers do not describe a bill), `"metered"` (API key — they do).

## Deliberately not built

- **Python/IPython kernel, installable skill packages, recursive sub-agents** —
  external design choices that do not fit this product.
- **A second store for harness behaviors** — harness learnings are memories in
  an isolated scope, on purpose; see above.
