# Phase-1 verdict: why the seat's causal-chain cases fail, and the fix shipped for it

Date: 2026-08-10 · Branch: `feat/causal-chain-lineage` (implementation in `0f385c0b`) ·
Baseline comparator: mech1 (PR #484, arm B = 111/126, forward-trace 1/6, propulsion-chain 2/6).
Status: **implementation committed + unit-tested; measured run NOT executed** (machine ceded
for demo-day rebuild before the census-with-fix and the pre-registered 6-repeat run).

## Verdict, in one paragraph

The causal-language signal exists and extraction works; nothing joins it across memories or
routes it into the lineage graph, and every edge the type-pair path does mint is too weak to
be delivered by recall. The chain cases fail because exactly one memory — "The drifting
vessel struck a support pier…" — is a lexical island reachable only through a causal edge
that did not exist. This is the cheap routing/join/calibration problem, **not** the
CATENA-extraction research wall.

## Evidence (all from a fresh store seeded with the frozen demo corpus, post-#484 backend)

1. **`catena::extract_event_links` fires on the demo corpus.** The knowledge-graph export of
   the seeded store contains the event nodes (`strike`, `collapse`, `trip`, `lose`, `loss`,
   `drift`, `closure`, `wreckage`) and the within-sentence causal event edges CATENA minted:
   `trip→lose` (m0 "lost propulsion **because** an electrical breaker tripped"),
   `loss→drift` (m1 "**led to** the Dali drifting"), `strike→collapse` (m2 "**which
   triggered** the collapse"), `closure→divert` (m5 "**as a result of** the closure").
   Extraction is per-sentence only, and its sole consumer is the knowledge graph
   (`graph_memory.rs:3178`) — the lineage graph never sees it.

2. **`infer_by_types` HAS an Observation→Observation path — since April.** Added in
   `b761f342` ("causal chain density", 2026-04-10): `(Observation, Observation) →
   InformedBy` at 0.7 × 0.75 = 0.525 base (`lineage.rs:945`). The "missing type mapping"
   theory is **falsified**; do not re-derive it. The drift→strike pair dies later in the
   pipeline: `confidence = 0.525 × semantic_signal × temporal`, and m1/m2 share no entities
   and almost no vocabulary, so `semantic_signal` falls below what the 0.20 store floor
   needs (≥0.381) — or below the 0.30 semantic gate outright. Causal continuation across a
   lexical topic shift is precisely where co-occurrence signals go to zero.

3. **The delivery machinery exists but is dead at real confidences.** The seeded store has
   101 inferred lineage edges: mean confidence 0.240, max 0.375. Recall's lineage machinery
   gates at 0.5 (score boost) and 0.7 (candidate expansion, `recall.rs` /
   `LINEAGE_EXPANSION_MIN_CONFIDENCE`) — **every edge in the store is below both gates**, so
   the boost, the expansion, and the seat's `recall_lineage` rendering (which only carries
   edges among returned results) can never deliver anything. This matches the #484 revert
   verdict ("drift→strike absent in 7 of 7 probed user graphs").

4. **One unreachable memory accounts for all nine chain failures.** Rescoring mech1's raw
   transcripts with main's own scorer and decomposing per gold id: in every one of the 9
   failing runs (forward-trace 5, propulsion-chain 4), m2 ("drifting vessel struck a support
   pier") was **never retrieved by any mechanism** — not proactive, not native recall, not
   MCP. m0 and m3 were retrieved and cited almost always (m3 in 8/9 failures). Direct recall
   probes of both chain questions against the seeded store confirm: m2 absent from top-8 —
   its text contains no "Dali", no "propulsion", no "port".

## What was built (committed in `0f385c0b`, pushed)

Cross-memory causal-language join, routed into lineage:

- `catena::CausalProfile` / `causal_profile()` — per-memory asserted causes, asserted
  effects, narrated event triggers, in normalized lemmas. `Precedes` contributes nothing.
- `causal_vocab::normalize_event_lemma()` — deverbal noun → verb lemma (`loss`→`lose`) so
  the nominal and verbal mention of one event unify. New lookup; `NOMINAL_EVENTS`/`SIGNALS`/
  `links_from_tokens` untouched → knowledge-graph spine extraction stays byte-identical.
- `LineageGraph::infer_language_relation()` — handshake tier 0.80 (both memories assert
  causation through a shared event: m0→m1), continuation tier 0.70 (one asserts, the other
  narrates: m1→m2 via `drift`). Wired through `infer_relation_with_profiles()` into
  `infer_lineage_for_memory()`; degrades to the type path when the parser is unavailable.
- Unit tests pin the tiers, the temporal constraints, the degradation, and the delivery
  contract (`language_confidence_tiers_clear_the_expansion_gate`).

## The confidence-calibration question (open — argued, not yet measured)

Raising edges above the delivery gates is **not obviously safe**; the PMI edge-gate finding
(−97.4% edges was a *win*) says more edges is not the goal. The argued position:

- Confidence here is **not** scaled by entity/embedding overlap, deliberately: the shared
  *asserted* event is the semantic evidence, and the target case is exactly where overlap is
  zero. Evidence hierarchy: explicit 1.0 > handshake 0.80 > continuation 0.70 (= expansion
  gate, pinned by test) > type-prior ≤ 0.375 measured.
- The flood guard is structural: **narration alone can never mint an edge** — at least one
  side must contain an explicit causal signal ("because", "led to", "triggered", …) whose
  linked event the other side carries. The haystack (50+ routine memories) is signal-free by
  construction and contributes nothing (unit-tested).
- **Unverified until the census re-runs against the fixed backend** (pre-registered in
  `%LOCALAPPDATA%/shodh-guidance-eval/results/causal1/PREREGISTRATION.md`): ≤10 language
  edges on the 71-memory corpus; zero ≥0.5 edges between haystack pairs; m1→m2 Caused@0.70
  present; recall probe returns m2 with its edge in the payload. If the census shows a
  flood, the calibration is wrong regardless of how good the argument sounds.

Known residual risks, accepted and documented: (a) `proactive_context` is a separate
pipeline with no lineage expansion — zero-recall runs (1 of 12 mech1 chain runs) stay
unfixable by this change; (b) m2→m3 has no language bridge (m3's sentence-initial "Because"
is a real CATENA-lite extraction gap — `nearest_left` finds no event left of token 0 — but
m3 was cited in 11/12 runs, so it is not load-bearing); (c) generic recurring events
("delay", "closure") in signal-bearing memories could over-link on other corpora — the
census flood check is the tripwire, and a lemma stoplist is the fallback lever.

## Recall-gate statement

`tests/recall/locomo-gate-baseline.json` is **unaffected**: the recall harness ingests via
`remember()` + `process_experience_into_graph` and queries `MemorySystem::recall` directly —
`infer_lineage_for_memory` never runs there, and no shared extraction path changed (verified:
the diff is pure-additive in `catena.rs`/`causal_vocab.rs`).

## To resume (exact steps)

1. Rebuild backend from this branch: `cargo test --no-run` with `CARGO_TARGET_DIR` at the
   repo's `target/` (build was interrupted mid-way; incremental resume). **Verify the exe
   mtime refreshed** and pass it via `BACKEND_EXE` explicitly — `resolveBackendExe()`
   otherwise prefers the stale `target/x86_64-pc-windows-msvc/release` exe from Aug 9.
2. Re-run the census (script preserved in the session scratchpad; trivially recreatable:
   seed `seed-demo-corpus.mjs` MEMORIES into a fresh user, dump `/api/lineage/edges`, run
   the two chain-question recall probes) and check the four pre-registered census gates.
3. Seat: use the **worktree's** `seat/dist` (built from main's seat source; `npm ci && npm
   run build`) — the main repo's `seat/dist` (Aug 9 22:54) contains unmerged R1-branch
   changes. `mcp-server/dist` from the main repo is current (no source drift).
4. Measured run per `PREREGISTRATION.md`: arm B only, `--mech on --guidance off --provider
   anthropic --model claude-haiku-4-5`, users `causal1-b1..b6`, full frozen 21-case set,
   results dir `results/causal1`, rescore with main's untouched scorer. Bar: forward-trace
   ≥4/6 AND propulsion-chain ≥4/6 AND no non-chain case −2 vs mech1 AND overall ≥85%.
   Scorer and case set must show zero diffs under `seat/eval/`.
