# Agent Traceability via Graphs — Design

**Date:** 2026-07-30
**Status:** Approved design, pending implementation plan
**Scope:** shodh-memory engine (capture, log, graph index, time-travel), MCP server, workbench surfaces

## 1. Motivation

Agents increasingly act as first-class collaborators, and their operations deserve the same auditability as any other system of record. Transport-level traceability (signed events: who said what) answers attribution; **knowledge-level traceability** — what did the agent know, where did that knowledge come from, and what did it do with it — requires a memory substrate with provenance, deterministic replay, and lineage, which shodh already provides. This design delivers an inspectable chronological session transcript per agent, backed by the graph so the trail is *traversable*, not just scrollable.

## 2. Decisions (ratified in brainstorm, 2026-07-30)

| Decision | Choice | Rationale |
|---|---|---|
| Trace scope | Full activity from day one | Memory ops witnessed server-side for ANY agent (zero integration); non-memory actions (edits/commands/prompts) via the SHIPPED Claude Code hooks; Buzz/OTel connectors additive later |
| Witnessed vs reported | Hard attestation distinction | Engine-observed ops = `witnessed`; hook/connector events = `reported`. Never blended — rendered with distinct badges; audit claims only ever assert what was witnessed |
| Storage | Log-as-truth + partitioned graph index | Knowledge graph decays/prunes BY DESIGN; audit trails must never. Append-only immutable log is source of truth; graph entries are derived, rebuildable, lifecycle-excluded |
| Tamper evidence | Hash chain from day one | Each record carries prev_hash (SHA-256); near-free at write, transformative for the regulated-audience audit story |
| V1 capabilities | Transcript pane, graph overlay, time-travel, audit export | All four ratified; sliced 1→4 so each lands independently; slices 1–2 target pre-demo |
| Track priority | Parallel to demo-critical path, not instead | Map pane (W1), live eval, and corpus re-ingest keep the Aug-12 critical path; this track must not consume it |

## 3. Architecture

Flow is one-directional: **capture → log → {index, surfaces}**. Trace writes never mutate knowledge-graph state; knowledge-graph lifecycle (decay, prune, canonicalize, consolidation) never touches trace data. Consumers read only.

### 3.1 Capture layer
- **Single choke point** wrapping every MCP/HTTP operation at the request router (the decode-choke-point lesson applied to writes): op type, `session_id`, agent identity (`user_id`), timestamp, request summary, outcome status.
- **Retrieval evidence on recalls:** returned memory ids always; per-layer attribution when the diagnostics path ran (existing `ScoreAttribution` machinery reused, not duplicated).
- **Reported-event ingest endpoint:** hook/connector events (Claude Code `hooks/` today; Buzz/OTel later) enter the SAME log with `attestation: reported`, source identifier, and the reporter's own timestamp preserved alongside arrival time.
- **Capture failure policy:** a failed log append must never fail the underlying operation; it must also never be silent — the failure increments a loud metric and permanently marks that session's trace `integrity: incomplete`, which every surface must display. Silent-data-loss rule applied to the audit trail itself.

### 3.2 The operation log (source of truth)
- Keyspace `oplog:{session_id}:{seq}` in a **dedicated column family**, append-only. No update or delete paths exist for it in the codebase (enforced by construction and by contract test).
- Each record: `{seq, ts, session_id, user_id, op, attestation, payload_summary, evidence_refs, prev_hash, integrity_flags}`. `prev_hash` = SHA-256 over the canonical serialization of the predecessor record; genesis record hashes the session header.
- **`payload_summary` is a knowing PII commitment (slice-1 review I5):** it carries bounded request content (e.g. the first 200 chars of a recall query) into a log that is permanent and per-record-undeletable by design — `/api/forget` cannot remove it; only whole-user erasure can. This is deliberate (an audit trail that can't say what was asked can't audit), but it must be stated wherever the log is described, and any future redaction capability is a chain-terminating design change per the retention clause below, never a quiet edit.
- **Lifecycle exclusion is provable:** the implementation plan's Phase-0 audit enumerates every RocksDB lifecycle writer (decay sweeps, forget sweeps, compaction filters, index rebuilds, canonicalization, backup/restore) and proves none touch this CF. A contract test pins it.
- Retention: none in v1 — the log is permanent. (A future retention policy, if ever added, must be an explicit, logged, chain-terminating operation — never a silent prune. Recorded here so it cannot arrive as a quiet default later.)
- **Immutability scope, corrected by the slice-1 substrate audit (2026-07-30):** "no delete paths" holds *within a user's lifetime under normal operation*. Two pre-existing whole-store events legitimately bound a chain and are explicit chain-terminating facts, not violations: (a) **user deletion** (`forget_user` removes the entire user directory — the right-to-erasure semantics the product already has; the trace goes with the user, by design); (b) **backup restore** (rolls the store — log included — back to a self-consistent but truncated chain; exports taken afterward report verified-but-shorter, with the restore event as the documented explanation). Audit claims and export verdicts speak within these bounds — "tamper-proof within the recorded history," never "nothing was ever deleted."

### 3.3 Graph index (derived, partitioned)
- **Node kinds:** `Session`, `Action`. **Edge kinds:** `PerformedBy` (Action→Session/agent), `Retrieved` (Action→Memory), `Wrote` (Action→Memory), `InformedBy` (Action→Memory, for evidence surfaced into responses), `NextAction` (Action→Action, intra-session order).
- Materialized **asynchronously** from the log (capture latency stays flat); materializer is idempotent, resumable from any log position, and fully rebuildable — the index is cache, the log is truth.
- **Partition rule at the read choke point:** one predicate excludes trace node/edge kinds from semantic retrieval, spreading activation, decay, pruning, and consolidation. Contract test: trace entities never appear in recall results, never receive decay updates, never get pruned. (Same test discipline as geo composition's exclusion assertions.)
- **The payoff query class:** walk `InformedBy` backward from any action into the knowledge graph, then continue through existing provenance (edge → attesting episodes) — answering "which stored belief informed this action, and what attested that belief" as a single traversal. Cross-session variants ("this belief influenced actions in sessions A, B, C") are the same walk with a wider frontier.

### 3.4 Surfaces
- **Transcript pane** (canonical front/, gated exactly like the map pane: config flag + graceful absence + lazy asset loading): chronological session feed; each action expands — recalls show query → layers → scores → returned memories with links; writes show what was stored; reported actions show source + badge. Session picker composes with the existing identity selector (agent = `user_id`).
- **Graph overlay:** selecting an action highlights its touched subgraph on the existing canvas graph (ids arrive from the action's `evidence_refs`; no new graph query engine needed for v1 overlay).
- **Time-travel:** `as_of: T` filter on retrieval — a memory is visible iff `created_at <= T` and not tombstoned at T; graph edges respect existing bi-temporal fields (`valid_at`/`invalidated_at`). Surfaced as a "view as of" control (scrubber pattern from the map PoC). Explicitly documented limitation for v1: `as_of` reconstructs *content availability*, not historical index states (embeddings/ANN graph as they were at T) — ranking at T is approximated with current indexes over T-filtered candidates; the audit answer ("could the agent have known X at T") is exact, the ordering replay is approximate. This distinction appears in the UI copy and the export.
- **Audit export:** one call produces a session bundle — full log slice, hash chain, evidence memories (MIF for the knowledge parts), integrity verdict (chain verified / incomplete flags) — verifiable offline by a standalone checker script shipped in-repo.

### 3.5 Visual grammar — from the human's point of view (ratified 2026-07-30; amended same day: "make sure the ui takes humans pov")

**Governing rule:** every screen is organized around plain-language questions a person brings — never around the system's data model. Schema vocabulary (attestation, provenance, DAG, hash chain) is banned from first-glance UI copy; it lives behind expansion for expert users. **Acceptance test written into every UI plan: a person unfamiliar with shodh answers the pane's stated questions unaided.**

The trace surface answers four human questions, in this order of prominence:

1. **"What did this agent do?"** — the session reads as a *story*: a one-line plain-English summary per step ("Looked up 4 memories about the Key Bridge collapse", "Saved a note about the manifest mismatch", "Edited src/parser.rs — reported by Claude Code"). Claude-style feed: each line expands to detail only when asked. The engineer's view (query → layers → scores) is the *expanded* state, never the resting state.
2. **"What was it looking at?"** — expanding a step shows the actual memories consulted as readable snippets with dates — never id lists. Trust labels in words: "Verified by the engine ✓" vs "Reported by Claude Code" (the witnessed/reported enum stays in the API, not the label).
3. **"Where did that come from?"** — from any memory snippet, one click walks its origin as a short readable chain ("First recorded July 8 · confirmed twice since"), with the graph overlay (slice 3) lighting the same trail on the canvas for those who want the picture.
4. **"What did it know back then?"** — slice 4's time-travel control is labeled in human terms ("Show this agent's knowledge as of July 12, 09:00") using the map PoC's scrubber; the v1 ranking approximation appears as a plain caveat ("shows what was knowable then — not exactly how it would have ranked").

The *session shape* element (formerly a DAG strip) survives only in service of question 1: a slim activity sparkline over time — bursts and gaps a human reads instantly — dots colored by "looked things up / saved something / acted outside shodh". No node-edge diagram at rest; the structural view is an expansion, not a landing state.

Cross-cutting: shared selection (touch a step anywhere, it highlights everywhere); tamper-evidence surfaces as "Tamper-proof record ✓" / "⚠ this session's record has gaps" (hash chains stay in the export, not the copy); same gating/lazy-load contract as the map pane; dark-theme consistent; every visual element traceable to a log record — no decorative data.

## 4. Slices (each an independent SDD plan+build)

1. **Capture + log:** choke-point middleware, oplog CF, hash chain, reported-event endpoint, hooks wired to it, trace read API (`GET /api/trace/{session}` paginated). Gate: capture-completeness test (every routed op → exactly one record); overhead < 2% on recall latency (benchmarked); LongMemEval smoke bit-neutral.
2. **Graph index + transcript pane:** materializer, partitioned kinds, partition contract tests, pane in front/ (gated). Gate: partition-exclusion proofs; pane renders real sessions end-to-end.
3. **Graph overlay:** selection → subgraph highlight wiring. Gate: overlay driven by evidence_refs only, no bespoke queries.
4. **Time-travel + audit export:** `as_of` retrieval filter + UI control; export bundle + offline verifier. Gate: point-in-time correctness on fixtures (memory visible/invisible across T boundaries incl. tombstones); export chain-verification round-trip.

Slices 1–2 target pre-demo; 3–4 follow. Every slice runs the full discipline: plan → Opus implementers → adversarial review → scoped re-reviews → honest CI.

## 5. Error handling

Capture: never fails the op; never silent (metric + `integrity: incomplete`). Materializer: crash-safe resume; a gap in materialization is self-healing (rebuild from log) and never visible as missing truth — only as delayed index. Export: refuses to produce a bundle claiming integrity over a broken chain; produces it with an explicit `incomplete` verdict instead. UI: absence of trace data for a session renders as "not captured" (distinct from "empty session").

## 6. Testing

Unit: hash-chain construction/verification incl. tamper cases; record serialization round-trip; `as_of` boundary fixtures. Contract: lifecycle-exclusion (no writer touches oplog CF; no trace kind in recall/decay/prune paths); capture completeness per routed op; witnessed/reported never blended in any API response field. Integration: end-to-end session → transcript → overlay ids → export → offline verify. Eval gates: LongMemEval smoke bit-neutral per slice; recall-latency overhead < 2%.

## 7. Out of scope (v1)

Buzz/OTel connectors (design allows, later); retention/redaction policies (explicitly deferred, see §3.2); historical index-state replay (approximation documented instead); cross-agent *privacy* controls beyond existing user_id isolation; the remove-vs-revive Hebbian decision (independent; noted because both touch graph lifecycle).

## 8. Risks

| Risk | Mitigation |
|---|---|
| Trace data leaks into retrieval/rankings | Single read-choke-point predicate + contract tests (geo-composition discipline) |
| Log write overhead on hot recall path | Async evidence enrichment where possible; <2% latency gate, benchmarked not asserted |
| Audit trail mutability creep (future features) | Log CF has no update/delete paths; retention policy pre-committed to be explicit and chain-terminating (§3.2) |
| Witnessed/reported blur undermines audit claims | Attestation is a mandatory record field, distinct rendering, export verdicts only assert witnessed ops |
| Track competes with demo-critical path | Slice plans scheduled around W1/eval/ingest; slices 3–4 explicitly post-demo |
