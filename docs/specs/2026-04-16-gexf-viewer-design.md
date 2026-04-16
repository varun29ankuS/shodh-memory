# GEXF Viewer — Shodh-Semantic Graph Browser

## Goal

Ship a browser-based viewer that makes shodh's Hebbian memory dynamics legible: spreading activation, edge weights, and LTP tier promotions become visually obvious at a glance. Works against shodh's GEXF export endpoint (live or snapshot), in-tree and bundled into the shodh binary.

## Non-Goals (v1)

- Write-curation actions (pin / forget / tier changes via UI). Viewer is read-only; curation surfaces *candidates* and lets you export ID lists for out-of-band action.
- Generic GEXF viewer. This viewer is shodh-shaped from the start. Extraction of a domain-neutral engine from `../viewer/` is a separate effort, intentionally deferred until we have three concrete implementations (HTTP Graph, ISAP, Shodh) to refactor toward.
- Fine-grained graph SSE events (`NODE_ACTIVATED`, `EDGE_STRENGTHENED`, …). v1 uses the existing `/api/events/sse` stream as a refetch trigger; any event = "something changed, re-pull GEXF." Fine-grained events are an additive follow-up.
- Replacing the d3 viewer on day one. Ship at a new route; retire the old route after a grace period.

### Regressions from the existing d3 viewer

- **3D view removed.** sigma.js is 2D WebGL. The d3/three viewer's 3D toggle does not carry over. Debugging Hebbian dynamics (the primary reason to build this) is a 2D task; the 3D orbit was a novelty, not a debugging surface. If 3D proves load-bearing in practice, it comes back as a follow-up using a separate renderer behind a mode toggle, not a blocker for v1.

## Context

**What exists today:**

- `GET /api/graph/{user_id}/export?format=gexf` — GEXF 1.3 export of the full knowledge graph. Nodes: entity / memory / episode. Edges: relationship / entity_ref. Merged upstream from our work.
- `GET /graph/view` — existing d3.js + three.js viewer (`src/handlers/graph_view.html`, 636 lines). Consumes a custom JSON shape at `/api/graph/data/{user_id}`, not GEXF. Caps at 200 entities + 100 memories. **Polls every 10 seconds** — not SSE-driven; an explicit comment at `graph_view.html:634` notes SSE was removed because browser `EventSource` cannot send custom headers so `X-API-Key` auth fails. API key is server-injected via `{{API_KEY}}` template substitution from `SHODH_DEV_API_KEY` env var.
- `GET /graph/assets/{file}` — static-asset handler for vendored JS (d3, three.js, OrbitControls), baked into the binary via `include_bytes!`. Per-request nonce + CSP pattern established in `visualization::graph_view`.
- `GET /api/events/sse` — shared SSE stream consumed by the TUI via `reqwest_eventsource` (which supports custom headers). Protected by `X-API-Key` (registered under `build_protected_routes` in `router.rs:405`). Supports `?user_id=X` query filter. Does **not** currently support query-param auth, so browser `EventSource` cannot subscribe — see "Authentication" below for the v1 change.
- `../viewer/` — sibling project, sigma.js + graphology, already does GEXF parse/write. Currently hosts HTTP-Graph and ISAP domain modules. Not bundled with shodh; lives in its own repo.

**Gap:**

No viewer consumes the GEXF export. The d3 viewer uses an older custom JSON contract, renders at most 300 nodes, and has no concept of LTP status, Hebbian edge weight as a first-class visual, or spreading-activation animation. Shipping a viewer for GEXF is the forcing function; debugging Hebbian dynamics is the payoff.

## Architecture

Three pieces:

1. **Browser app** — static HTML + vanilla JS + CSS at `src/handlers/viewer/`. Dependencies loaded via importmap + ESM from `/graph/assets/`. No build step. No bundler. No `package.json`. Matches the pattern already established for the d3 viewer.
2. **Rust glue** — handler in `src/handlers/visualization.rs` (or a sibling `graph_view2.rs`), serving `index.html` at `GET /graph/view2`. Follows the existing nonce-per-request CSP pattern: generate a 128-bit random nonce, substitute `{{NONCE}}` in the HTML, attach `CspNonce` to response extensions, middleware widens CSP for that one response.
3. **No new backend routes.** Consumes `GET /api/graph/{user_id}/export?format=gexf` for the initial snapshot and `GET /api/events/sse?user_id=X` for the refetch trigger. Two existing routes gain additive capabilities:
   - The export route gains one additional query parameter (`include_content`, see "GEXF Export Additions" below) **and** `ETag` / `If-Modified-Since` response headers so the client can short-circuit unchanged refetches with a `304`.
   - The SSE route gains optional `?api_key=X` query-param auth as a fallback when the `X-API-Key` header is absent (see "Authentication" below). This is the unblock for browser `EventSource`, which cannot send custom headers. Existing `reqwest_eventsource` consumers (the TUI) continue to use the header — unchanged.

   No new routes are added.

## Scale Target

Design for ~50k nodes and ~200k edges. WebGL rendering required (sigma.js). In-place graph diff + mutate for live updates; full-replace per event is a non-starter at this scale.

## Fork Base

Fork `../viewer/` at commit **`827c191`** (`refactor(viewer): simplify dashed edge program and tune redirect edges`), the last commit before ISAP landed (first ISAP commit is `0ac5187 feat(viewer): extract ISAP investigation indexes on GEXF load`). If HTTP-Graph stripping proves heavier than expected, drop back to `75327d2` (`replace 3s polling timer with event-driven filter refresh, fix GEXF load crash`), a more generic earlier state.

## Authentication

The backend is gated by `X-API-Key`. Browser `EventSource` cannot set custom headers, which is why the existing d3 viewer had to drop SSE in favor of 10-second polling (see `graph_view.html:634`). For the new viewer, two changes unblock browser auth without introducing a new auth model:

1. **HTML template substitution (unchanged pattern).** The `view2` handler substitutes `{{API_KEY}}` into `index.html` at render time, sourced from `SHODH_DEV_API_KEY`. This matches the existing d3 viewer. Fetch calls use the `X-API-Key` header as usual.

2. **SSE query-param fallback (new).** `webhooks::memory_events_sse` (the `/api/events/sse` handler) gains an optional `?api_key=X` query parameter. Precedence: `X-API-Key` header wins if present; otherwise fall back to `?api_key`. If neither is present or either is wrong, the handler responds `401` as today. The viewer's `EventSource` client appends `&api_key={{API_KEY}}` to the SSE URL.

**Security tradeoff.** The API key travels in the URL and therefore ends up in server logs, HTTP `Referer` headers, and browser history. The existing viewer already embeds the key in the rendered HTML and sends it in `fetch` headers, so the attack surface is already "anyone with access to the browser session has the key." Query-param auth widens that to "anyone reading server logs." Acceptable for the local-dev use case this viewer targets. If a remote / multi-tenant deployment lands later, the fix is a short-lived signed session token minted per-view — not a reason to block v1.

## GEXF Export Additions (Prerequisite)

The existing GEXF export carries most of what the viewer needs (all node attrs, edge `weight=` as Hebbian strength, `ltp_status`, `tier`, `activation_count`). The following attrs exist on the internal model and should be added to the emitted GEXF before viewer work begins. Same commit/branch; the viewer depends on them.

**Edges** (in `relationship_to_edge` emission loop):

- `last_activated` — DateTime. Required for spreading-activation debugging (animate recently-fired edges).
- `created_at` — DateTime. Age filter.
- `valid_at` — DateTime. Bitemporal validity.
- `entity_confidence` — Optional f64. Quality signal.

**Memory nodes** (in `memory_to_node` emission loop):

- `last_accessed` — DateTime. Recency highlight.
- `temporal_relevance` — f32. Derived decay-aware score.
- `created_at` — DateTime. Age.
- `agent_id`, `run_id` — Optional strings. Provenance.

**Entity nodes** (in `entity_to_node` emission loop):

- `last_seen_at`, `created_at` — DateTimes.
- `summary` — String.
- `labels` — array of strings, serialized as comma-separated string attvalue.
- `is_proper_noun` — bool.

**Episode nodes** (in `episode_to_node` emission loop):

- `source` — string.
- `valid_at`, `created_at` — DateTimes.

**Content gating.** Memory `content` (full text) and Entity `summary` and Episode `content` are substantially larger than the numeric/enum attrs. Gate behind a new `?include_content=bool` query param, default `false`, mirroring the existing `?include_embeddings=bool` pattern. Snapshot stays compact; the viewer lazy-fetches full content on node click.

**Attribute declarations.** Update the `<attributes class="node">` and `<attributes class="edge">` declaration blocks at the top of the GEXF document with new IDs for each added attribute.

**`<meta>` block.** Add a `server_time` field (RFC 3339) to the GEXF `<meta>` element, emitted at export time. The viewer displays server-vs-client skew in the sidebar when they differ by >1s, so stale snapshots and pulse-animation timing glitches are diagnosable. Keeps the viewer honest about what "now" means in animations that key off `last_activated`.

**ETag / If-Modified-Since.** The export handler emits a weak `ETag` derived from the graph's max `last_modified` timestamp (or a monotonic version counter if that's cheaper). Refetch sends `If-None-Match`; unchanged snapshots return `304` with no body. Cuts refetch cost during idle streams where SSE fires but nothing material has changed.

**Node decay rate.** Not emitted. Confirmed per-tier-constant (lives in `decay.rs`); the viewer derives it from `tier`.

## Implementation Prerequisites

Before any viewer code lands in this repo, the fork of `../viewer/` is vendored into `src/handlers/viewer/` as step 0. Concretely:

1. Clone `../viewer/` at commit `827c191` (fallback `75327d2`).
2. Delete HTTP-Graph and ISAP-specific modules; keep the generic graphology / sigma / GEXF / FA2 / interaction scaffolding.
3. Move the stripped tree into `src/handlers/viewer/` matching the structure in "Component Breakdown" below.
4. Commit as the first change of the viewer branch with a message pointing at the source commit, so the provenance and what-was-removed are both legible in `git log`.

Every subsequent commit on the branch modifies this vendored base rather than authoring fresh sigma.js scaffolding. This keeps us honest about what's been tested upstream vs what's new.

## Component Breakdown

```
src/handlers/viewer/
  index.html               # entry page; nonce-substituted by handler
  css/
    style.css              # theme, panels, tooltip, legend
  js/
    boot.js                # URL parsing, mode detection, wire-up
    graph/
      loader.js            # fetch GEXF (API / file / drag-drop) → graphology Graph
      layout.js            # ForceAtlas2 worker (inherited from viewer, re-tuned)
      renderer.js          # sigma init, reducers, lifecycle
    domain/
      node-style.js        # tier → color, activation → glow, size → f(importance)
      edge-style.js        # weight → thickness, ltp_status → dash, tier → hue
      filters.js           # tier / type / weight / activation / LTP / recency filters
    ui/
      sidebar.js           # filter controls, stats, legend, live indicator
      detail-panel.js      # node/edge detail; lazy-loads full content
      legend.js            # visual vocabulary reference
    live/
      sse-client.js        # EventSource wrapper, exponential backoff reconnect
      refetch.js           # debounced refetch, in-place graph diff + mutate
    config/
      api-client.js        # base URL, X-API-Key header, path builders
```

Vendored JS deps go in `src/handlers/assets/` alongside the existing d3/three files: `sigma.min.js`, `graphology.umd.min.js`, `graphology-gexf.umd.min.js`, `graphology-layout-forceatlas2.umd.min.js`. Served by the existing `graph_asset` handler at `/graph/assets/{file}`. Expected size: ~1–1.5 MB additional binary weight.

## Data Flow

### Boot

URL parsing determines mode:

- `?user_id=X` with no `file` → **live**.
- `?file=<url>` → **snapshot-remote**.
- Drag-dropped `.gexf` file → **snapshot-local**.
- No params, no file → blank drop zone + "Connect to shodh" form (base URL + user_id).

### Live mode

1. `loader.fetchFromApi(user_id)` → GEXF text → `graphology.gexf.parse` → `Graph`. Store `ETag` from response for subsequent refetches.
2. `renderer.mount(Graph, container)` — sigma initialized, FA2 worker started. A "Laying out..." overlay is visible while FA2 converges to prevent users thinking the graph is broken during the first 1–3 seconds; dismissed when simulation energy drops below a threshold or a timeout fires, whichever first.
3. `sseClient.connect(base_url, user_id)` — `EventSource('/api/events/sse?user_id=X&api_key={{API_KEY}}')` (query-param auth; see "Authentication").
4. On any SSE message, `refetch.trigger()`.
5. `refetch.trigger` implements a 2000ms debounce: set `pending = true`, schedule fetch. While a fetch is in flight, new triggers set `pending = true` again; the next debounce window starts after completion. At most one in-flight + one queued. The longer debounce bounds refetch bandwidth on high-activity streams; 2s is imperceptible for the debugging use case and reduces re-parse cost on larger graphs.
6. Refetch sends `If-None-Match` with the stored ETag. `304` → no work beyond releasing the pending flag. `200` → parse the body, compute set-diff against the current `Graph`, apply add/update/remove in place, store the new ETag:

```
added_nodes   = new \ old → addNode with attrs; new nodes spawn at a neighbor's position with jitter
removed_nodes = old \ new → dropNode
common_nodes  = old ∩ new → mergeNodeAttributes (updates activation, tier, etc.)
same pattern for edges.
```

7. Viewport, selection, FA2 simulation state preserved. New nodes enter FA2 cold and are reintegrated on the next tick.

### Snapshot modes

`loader.parseFromText(text)` → `Graph` → `renderer.mount`. No SSE. Graph is immutable until the user drops another file.

### Interaction

- **Hover** → tooltip with label, tier, key quantitative attrs.
- **Click** → `detailPanel.show(id)`. Panel reads from Graph attributes first; if `content` is absent (snapshot was fetched with `include_content=false`), lazy-fetches `GET /api/memories/{id}` for full text.
- **Selection** → hop-aware highlight decay (inherited from `../viewer/` commit `fc87754`): dim non-neighbors to ~10% opacity, decay by distance.
- **Filter change** → re-run reducers; no graph mutation, just visual.

### SSE Reconnect

On error or close, exponential backoff: 1s, 2s, 4s, 8s, capped at 30s. Sidebar indicator flips to "disconnected / reconnecting." On successful reconnect, trigger one refetch to resync changes missed during disconnect.

## Visual Vocabulary

### Nodes

| Attribute | Channel | Detail |
|---|---|---|
| `type` | Shape | Circle = memory, square = entity, diamond = episode |
| `tier` (memories: working/session/longterm) | Color family | Warm orange → yellow → cool blue (working = hot, longterm = cool) |
| `importance` | Base size | 6–24px radius, linear |
| `activation` | Halo intensity | Bright ring at high activation; animated pulse when > 0.7 |
| `access_count` | Halo thickness | Log-scaled |
| `last_accessed` < 60s | Border pulse | Recency badge |

### Edges

| Attribute | Channel | Detail |
|---|---|---|
| `weight` (Hebbian strength, from GEXF `weight=`) | Line thickness | 0.5–5px, linear |
| `tier` (L1 / L2 / L3) | Hue | L1 hot red, L2 amber, L3 cool blue (volatile → established) |
| `ltp_status` | Dash pattern | dashed = pending, solid = consolidated, thick-solid = just-promoted |
| `last_activated` < 5s | Animated pulse | Source → target trail; the "spreading activation" debug channel |
| `activation_count` | Tooltip only | No additional channel |
| `relation_type` | Label on hover | Already emitted as GEXF edge `label=` |

**Performance budget for animated pulses.** Only animate edges whose `last_activated` is within the last 5 seconds. At steady state this is typically fewer than 100 edges. During a burst it may hit low thousands; still within sigma's WebGL capacity. If it degrades, throttle to top-N most-recent.

**Color choice is opinionated.** L1/Working = warm, L3/Longterm = cool. Rationale: matches the existing d3 viewer's direction and reads as "active/volatile → established." Flip if preference differs — the change is one constant table.

### Filters (left sidebar)

- Tier checkboxes (working / session / longterm + L1 / L2 / L3).
- Type checkboxes (memory / entity / episode).
- Weight threshold slider (hide edges below W).
- Activation threshold slider (hide nodes below A).
- LTP status filter.
- Recency window selector ("last 60s / 5m / 1h / all").
- Find-by-id and find-by-label text input.

### Curation (read-only v1)

- "Weakness" highlight toggle: importance < 0.2 **and** access_count < 5 — red ring. Candidate prunes.
- "Orphans" highlight: zero-degree nodes.
- "Dead edges" highlight: `last_activated` older than 7d or `activation_count == 0`.
- Tier histogram in sidebar.
- **Export filtered subset as GEXF** — download of currently-visible subgraph.
- **Export selected node IDs as text** — so users can pipe into `shodh forget <id>` etc. Curation-by-exported-list, no UI writes.

## Serving and Route Migration

### New route

Add `GET /graph/view2` served by a new handler, following `visualization::graph_view`'s nonce pattern. Static assets (JS modules and CSS) served via the existing `graph_asset` handler or an expanded variant at `/graph/viewer/{file}`, baked in via `include_bytes!` macros over the `src/handlers/viewer/` tree.

### Coexistence period

d3 viewer stays at `/graph/view`. New viewer at `/graph/view2`. No deprecation warning initially. Users pick.

### Grace period, then flip

After the new viewer has been in real use long enough to catch issues (target: a handful of release cycles), flip `/graph/view` to the new handler. Move the old `graph_view.html` and its supporting code out of the tree. Drop `/api/graph/data/{user_id}` if nothing else consumes it (TUI and scripts need a quick audit first). Announce in CHANGELOG.

3D view is not reintroduced. Document this in CHANGELOG and README when the flip happens.

## Error Handling

| Failure | Behavior |
|---|---|
| Initial GEXF fetch 4xx/5xx | Empty canvas + banner: "Failed to load graph: {status}." Retry button. |
| Initial fetch timeout (>15s) | Banner: "Load timed out." Retry button. |
| Malformed GEXF | Caught at `graphology.gexf.parse`. Banner: "GEXF parse error: {message}." Empty canvas. |
| SSE connection fails | Sidebar indicator: "disconnected." Exponential backoff reconnect. Graph stays rendered with last-loaded state. |
| SSE event with unrecognized type | Ignored; logged at `console.debug`. Viewer does not introspect event types in v1. |
| Refetch fails mid-session | Sidebar indicator: "live updates paused: {reason}." Graph unchanged. Retry with same backoff. |
| File >100 MB dropped | Warning banner: "File too large ({size})." Proceeds anyway. |
| Detail-panel lazy fetch fails | Panel shows truncated label + "Full content unavailable." No popup. |
| FA2 worker crash | Caught, logged, sim halts. Banner: "Layout stopped. Reload to retry." Graph interactive at last positions. |

Explicitly **not** handled:

- Shodh server not running: returns the initial-fetch banner, stops. No retry storms, no guessing.
- User authorization failures: 401/403/404 surfaced as-is in the banner.

## Testing

### Rust

- Unit tests for each new GEXF attribute: emitted correctly, well-formed, round-trips through `graphology.gexf.parse` in a Node-based test harness or via a manual fixture check.
- Integration test for the new `/graph/view2` handler: 200 response, content-type HTML, nonce substituted, CSP header present.
- Integration test for any expanded asset-handler allowlist: allowed files return 200 with the right content-type and cache header; disallowed paths return 404.
- Extend existing `test_gexf_export` assertions with the new attrs.

### JavaScript

- **Vitest + jsdom** for pure-logic modules (`filters.js`, `refetch.js` debounce, the graph-diff function inside `refetch.js`). No rendering, no sigma.
- **Playwright** end-to-end: run shodh with a seeded small test graph, load `/graph/view2?user_id=test`, assert:
  - Canvas renders non-empty after initial fetch.
  - Hover shows tooltip with expected fields.
  - Click opens detail panel with tier / activation / weight matching fixture.
  - Filter toggles hide/show expected node counts.
  - An SSE event (triggered via a test-only memory-write endpoint or a direct DB write) causes refetch within 5s (covers the 2s debounce + transport + parse).
  - Drag-drop a fixture `.gexf` → graph updates, SSE detaches.
- Fixtures: one hand-crafted 100-node GEXF covering every tier, every `ltp_status`, and at least one of every node/edge type, at `test/fixtures/viewer/`.

### Out of Scope for v1

- Visual regression (screenshot diffs).
- Strict performance SLAs at 50k nodes — smoke-level only.
- Layout quality assertions.

### Manual Smoke Test (before the flip in Migration)

1. Load a real shodh instance with >1k nodes. Viewer opens, renders within 5s.
2. Trigger a burst of activity (50 recalls in a minute). Pulses appear on active edges; no layout jitter.
3. Close and reopen mid-session. Reconnect works; state consistent.
4. Drag-drop a snapshot file. SSE detaches. Graph replaced.

## Open Items

None. All design decisions settled during brainstorming. Implementation plan to be produced next via `writing-plans`.
