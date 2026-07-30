# W1 — Map + Timeline Pane — Design

**Date:** 2026-07-30
**Status:** Approved design, pending implementation plan
**Scope:** `front/` (canonical surface), `front/src/main.rs` (serving crate), tile-asset pipeline. Engine untouched — the geotemporal capability merged in #418/#419 provides every API this pane consumes.

## 1. The human questions this pane answers

1. **"What happened near here?"** — events pinned on a real map of the analyst's area
2. **"When — and in what order?"** — a time scrubber that filters the map live; ordering chains drawn on it
3. **"What led to this?"** — the `Precedes` chain rendered as a walkable storyline on the map

Every screen element must serve one of these; the acceptance test is a person unfamiliar with shodh answering all three unaided on a geo-tagged corpus. Directive line (top of pane): *"Drag the time sliders, or click any glowing event."*

## 2. Decisions (ratified)

| Decision | Choice | Provenance |
|---|---|---|
| Stack | MapLibre GL JS 6 + PMTiles 4 + Protomaps v4 dark extract | PoC verdict; verified zero external calls at ~46 MiB total incl. z0–15 regional tiles |
| Gating | Triple: config flag (`SHODH_UI_MAP`, disable-only URL param) + asset-absence graceful hide + lazy loading (zero map bytes fetched when off) | Gating requirement; PoC-tested |
| Tiles | Optional separate artifact (like model bundles), never in repo/default install; per-deployment swappable | Offline/sovereign deployments; boundary-depiction strategy (§6) |
| Time control | The PoC scrubber (FROM/TO + density histogram) — becomes the workbench-standard time control (trace spec §3.5 reuses it) | PoC + trace spec |
| Attribution | ODbL notice rendered, docked (never occluded), plain text | Legal obligation, PoC-verified pattern |
| Framework | None — vanilla JS in `front/index.html`'s existing structure, vendored libs only | W0 decision; single-file structure question deferred |

## 3. Data wiring (replaces the PoC's hardcoded events)

- **Events:** `POST /api/recall` with `mode:"hybrid"`, `geo_lat/geo_lon/geo_radius_meters` (the composed path from #418) seeded from the current viewport center/extent; plus `GET /api/memories` pagination for initial corpus overview. Each memory with `geo_location` becomes a pin: `{lat, lon, created_at, content-snippet, id}`.
- **Time filter:** client-side on `created_at` over fetched results (the API has no `time_range` param — documented limitation, same as the eval harness; a server-side window param is a follow-up, not faked).
- **Precedes chains:** `Precedes` edges fetched via the graph API for entities/memories in view; rendered as the PoC's dashed ordinal chain. If the graph exposes no clean edge query for this yet, the plan's audit task decides between the existing graph endpoints and a small additive read endpoint — never a bespoke parallel store.
- **Classes/colors:** derive pin color from entity fine-type of the memory's primary location/entity (gpe/facility/event classes), falling back to memory type. Deliberate palette reuse from the existing visual system (warm = activity rule).
- **Empty/no-geo state:** if the corpus has zero geo-tagged memories, the pane says so in words ("No located events in this corpus yet — geo-tagged data appears here after ingest") — never an empty black map.

## 4. Serving requirements (front crate)

- `tower-http` `ServeDir`/`ServeFile` with **HTTP Range support** for the `.pmtiles` artifact (full-body serving breaks PMTiles — PoC-measured); correct `.mjs` → `text/javascript` MIME (wrong MIME = silent module refusal).
- Vendored assets (maplibre mjs/css/worker, pmtiles js, style JSON, sprites, glyph ranges) served same-origin; tile artifact path configurable (`SHODH_MAP_TILES`), absence = gate trigger.
- Security: popups/labels use `setDOMContent`/text nodes — never `setHTML` — since content is ingested data (XSS surface the PoC flagged).

## 5. Reference scenario

Acceptance scenario on a geo-tagged corpus (e.g., the GDELT harbor dataset): scrub to a chosen week → located events appear → toggle the Precedes chain → click through the storyline in time order. The PoC page retires once this pane renders the same scene from live data.

## 6. Boundary depiction is a deployment decision

MapLibre is depiction-neutral; political boundaries live in the tile/style asset, and different jurisdictions have different legal requirements for how boundaries must be drawn. The tile-artifact swap (§2) is the mechanism: per-deployment tile/style builds (style-level boundary suppression; point-of-view boundary datasets; official national geodata where certification requires it) with no code change per market.

## 7. Testing

- Gating contract tests mirror the PoC's harness where portable: `?map=0` fetches zero map assets; missing tiles → graceful card; no external hosts in any fetched asset.
- Data wiring: stubbed `/api/recall` fixtures → pins render with correct positions/times; scrubber filters counts; chain assembles/disassembles across window edges (the PoC's proven behavior, asserted against fixtures).
- Human-questions acceptance: §1, answered unaided.

## 8. Out of scope

Server-side time-range param (follow-up); tiles beyond the reference AOI (artifact decision per deployment); Spatial-mode robotics view changes; case pinning (W3); trace overlay (trace slice 3 composes later via shared selection).
