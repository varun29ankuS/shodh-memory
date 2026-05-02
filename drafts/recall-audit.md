# Recall Harness Audit — RH-1

**Issue:** #263
**Branch:** `feat/recall-harness-audit`
**Date:** 2026-05-02
**Status:** Audit complete, decision recorded

## Purpose

Before adding NDCG@k metrics, an L1 smoke suite, a CI gate, or swapping the embedder
to arctic-embed-xs, we need to know what recall infrastructure already exists in the
repo, what is reusable, what is missing, and whether new work should be grafted onto
the existing benchmark file or extracted into a new module.

## Inventory — what exists today

### `benches/associative_retrieval_benchmarks.rs` (629 lines)

A Criterion benchmark suite for the **knowledge graph layer only**. It exercises
`GraphMemory::traverse_from_entity` and `GraphMemory::traverse_weighted` directly,
not `MemorySystem::recall`.

**Three fixture scenarios** (all graph-only, no embeddings, no full pipeline):
- `CrossDomainScenario` — meeting notes that bridge two unrelated topics.
- `TemporalChainScenario` — sequence of events with temporal ordering.
- `CooccurrenceScenario` — entities that frequently co-occur, no direct edge.

Each scenario captures a `relevant_episode_ids: HashSet<Uuid>` ground-truth field.

**Three metric helpers** at lines 411–438:
```rust
fn precision_at_k(retrieved: &[Uuid], relevant: &HashSet<Uuid>, k: usize) -> f64
fn recall_at_k(retrieved: &[Uuid], relevant: &HashSet<Uuid>, k: usize) -> f64
fn mean_reciprocal_rank(retrieved: &[Uuid], relevant: &HashSet<Uuid>) -> f64
```

**Critical finding:** the metric helpers are **never called** by any benchmark in
the file. The `relevant_episode_ids` field is **never read**. Every benchmark from
line 444 onward is shaped like:

```rust
b.iter(|| {
    let _retrieved = scenario.graph.traverse_from_entity(...).expect("Failed");
});
```

The retrieval result is dropped on the floor. Only wall-clock latency is measured.
The lines 577–587 "expected improvement" table at the bottom of the file is
hand-typed and aspirational — no code asserts those numbers.

### Existing recall API surfaces in `src/memory/mod.rs`

- `recall(&self, query: &Query) -> Result<Vec<SharedMemory>>` — line 1174, the
  public path used by the HTTP server and MCP tools.
- `recall_with_diagnostics(&self, query: &Query) -> Result<RetrievalResult>` —
  line 1183, returns memories **plus** stage telemetry.
- `recall_tracked(&self, query: &Query) -> Result<TrackedRetrieval>` — line 5337,
  emits a retrieval id + query fingerprint for downstream feedback wiring.

### Existing telemetry types

- `RetrievalResult` (`src/memory/types.rs:3000`) wraps `memories + stats`.
- `RetrievalStats` (`src/memory/types.rs:2872`) already tracks per-stage data:
  mode, semantic_candidates, graph_candidates, graph_density, semantic/graph/
  linguistic weights, graph_hops, entities_activated, avg_salience_boost,
  retrieval_time_us, embedding_time_us.
- `TrackedRetrieval` (`src/memory/retrieval.rs:1543`) carries
  retrieval_id, query_fingerprint, retrieved_at.

### Other benchmarks

12 bench files in `benches/`, 24 test files in `tests/`. Sampled inspection
matches the pattern in associative_retrieval_benchmarks.rs: latency-only,
no quality gates, no NDCG anywhere in the tree (`grep -ri ndcg src/ benches/
tests/` returns zero hits).

`benchmark_report.json` at the repo root contains 22 latency rows from a
Criterion run. No quality columns.

## Gap analysis — what is missing

| Capability                                  | Present? | Notes                                                       |
|---------------------------------------------|----------|-------------------------------------------------------------|
| NDCG@k                                      | No       | Not implemented anywhere                                    |
| precision@k / recall@k / MRR helpers        | Partial  | Defined in benches file, never invoked, not reusable        |
| Full-pipeline fixtures (query → memories)   | No       | Existing fixtures hit `GraphMemory` directly                |
| Per-stage attribution (vamana / +graph / …) | Partial  | `RetrievalStats` reports candidate counts but not isolated  |
|                                             |          | runs of each layer; no mode toggle in the public API        |
| `recall-eval` binary                        | No       | No CLI for batch query → metrics → JSON                     |
| baseline.json on current MiniLM-L6          | No       | Nothing to regress against                                  |
| CI gate on regression                       | No       | No GitHub Action wired to recall metrics                    |
| LoCoMo (or any external) dataset loader     | No       | Only synthetic graph fixtures exist                         |

## Decision — extend vs extract

**Extract: create a new module `src/recall_harness.rs` and a new binary
`src/bin/recall-eval.rs`. Leave `benches/associative_retrieval_benchmarks.rs`
alone.**

### Why not extend the existing bench file

1. **Wrong layer.** The bench file builds a `GraphMemory` directly with
   hand-crafted `EntityNode`/`RelationEdge` records. The recall harness needs the
   full `MemorySystem` pipeline — embeddings, Vamana, spreading activation, BM25,
   reranker, fact-aware boost — which only `MemorySystem::recall_*` exposes.
   Stuffing the harness into a graph-only fixture would force us to either
   duplicate the whole pipeline in test setup or weaken what we measure.

2. **Wrong runtime.** Criterion is for micro-benchmark statistics
   (warmup, sample size, throughput estimation). We want a pass/fail metric
   harness that emits machine-readable JSON for CI and historical baselines.
   Criterion's output format and lifecycle do not match that need.

3. **Wrong reach.** Helpers locked inside a `#[bench]` file are not callable
   from a binary, an integration test, or a future Python wrapper. A library
   module is reusable.

4. **Audit trail.** The existing file's "expected improvement" table and dead
   metric helpers should stay as-is for now — they document graph-layer intent.
   Tearing them out as part of this work would mix concerns and bloat the diff.

### Module shape (preview, not implemented in this PR)

```
src/recall_harness.rs
    pub mod metrics      // ndcg_at_k, recall_at_k, precision_at_k, mrr, p_at_1
    pub mod fixtures     // L1 smoke cases (RH-3), LoCoMo loader (RH-7)
    pub mod runner       // run_query_set(memory, queries) -> Vec<QueryResult>
    pub mod report       // serialize to baseline.json / report.json
src/bin/recall-eval.rs   // CLI: --suite l1 --baseline baseline.json --out report.json
```

The runner calls `MemorySystem::recall_with_diagnostics` so per-query
`RetrievalStats` rides along into the report — RH-8 (per-layer attribution)
extends `Query` with a mode flag rather than re-implementing the pipeline.

## What this PR contains

This audit doc only. No code changes. RH-1's acceptance criterion in #263 is
"short markdown doc with decision + rationale", which this satisfies. RH-2
(metrics module) is the first PR that ships code under the chosen layout.

## Follow-up tasks unblocked by this decision

- **RH-2** can start: create `src/recall_harness.rs` with `mod metrics` and port
  `precision_at_k`/`recall_at_k`/`mrr` semantics, add `ndcg_at_k`, write unit
  tests with known-answer cases.
- **RH-4** (binary) and **RH-3** (L1 fixtures) become straightforward consumers
  of the module from RH-2.
- **RH-8** (per-layer attribution) can extend `Query` with an optional
  `pipeline_stages: Option<RecallStageMask>` field rather than introducing a
  parallel API.

## Open question (deferred, not blocking)

The dead helpers and unread `relevant_episode_ids` field in
`benches/associative_retrieval_benchmarks.rs` are technical debt. Once the
new harness lands, that file should either start using the harness's metric
module to assert quality (preferred) or have the dead fields removed. Tracked
informally for now; not creating a separate issue until RH-2 is merged.
