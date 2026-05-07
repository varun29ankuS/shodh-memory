# Recall Harness Fixtures & Baseline

This directory holds the fixtures, ground truth, and frozen baseline for
the `recall-eval` binary (see `src/bin/recall_eval.rs`).

## Files

- `corpora/shodh-smoke.jsonl` — L1 smoke corpus (memories ingested by the runner).
- `smoke_cases.jsonl` — 30 hand-crafted query cases with graded relevance labels,
  spanning six categories (decision / code / temporal / entity / multi_hop / negation).
- `baseline.json` — frozen per-metric scores against current `main`. Every PR is
  measured against this; regressions beyond the configured tolerance fail CI
  (see issue #267 / RH-5).

## Regenerating the baseline

Only regenerate `baseline.json` when you have *intentionally* changed retrieval
quality (embedder swap, scoring tweak, pipeline refactor) and want to freeze the
new numbers. Routine refactors should leave it untouched — that is the whole point.

```bash
cargo run --release --bin recall-eval -- \
    --suite smoke \
    --output tests/recall/baseline.json
```

The binary records the current git SHA, embedder identifier, and timestamp into
the report header so the baseline is self-describing.

After regenerating, sanity-check the diff:

```bash
git diff tests/recall/baseline.json
```

If a metric moved by more than ~2% in either direction, write a one-paragraph
justification in the PR description so future bisects have context.

## Per-pipeline-layer attribution (`--layer`)

`recall-eval` accepts a `--layer` flag (RH-8, #270) that selects which
subset of the retrieval pipeline runs. Modes are **cumulative**: each row
adds one stage on top of the row above it.

| Mode             | Stages added                                                           |
| ---------------- | ---------------------------------------------------------------------- |
| `vamana-only`    | Layer 3 vector ANN only (cosine + tie-break).                          |
| `+spreading`     | + Layer 2 graph spreading activation (RRF over vector ⊕ graph).        |
| `+bm25`          | + Layer 4 BM25 leg of three-way RRF fusion.                            |
| `+rerank`        | + Layer 4.9 ontological re-rank (see naming caveat below).             |
| `+facts`         | + Layer 0.7 / 4.8 fact-source boost from consolidated knowledge.       |
| `full`           | + Layer 0.4/0.5/0.6 pre-filters, Layer 4.6 interference, Layer 4.7     |
|                  | prospective signal, full Layer 5 unified scoring (recency × importance |
|                  | × arousal × credibility × tags × feedback × quality), retrieval       |
|                  | competition, Hebbian coactivation, hierarchy expansion.                |

Pass `--layer all` to run every mode in one harness invocation:

```bash
cargo run --release --bin recall-eval -- \
    --suite smoke \
    --repeats 1 \
    --layer all \
    --output /tmp/rh8_all.json
```

The report's `layers` map gains one entry per mode; `scripts/recall_diff.py`
renders a per-layer `ndcg@10`/`recall@10` delta table when both reports
share more than `full`.

### Caveats — read these before staring at the numbers

1. **`+rerank` is a misnomer in this codebase.** Issue #270 specs the mode
   as a cross-encoder rerank stage. shodh has no cross-encoder. The gate
   wraps the **ontological re-ranker** at Layer 4.9 (multiplicative boost
   when episode entity types match the query's expected ontology labels).
   The label is preserved for spec fidelity; if a cross-encoder ever
   lands, it joins this same gate.

2. **Modes below `full` skip Layer 5 unified scoring.** Per-layer ndcg
   numbers will look strictly *lower* than `full` for reasons that are
   *not* "this stage didn't help" — they include the absence of recency,
   importance, arousal, credibility, feedback, quality-gate, and Hebbian
   multipliers. Read the table as **deltas between adjacent rows**, not
   as standalone absolute values.

3. **Cumulative-only by design.** You cannot ask for "BM25 without
   spreading" or "rerank without facts" — `--layer` accepts only the six
   cumulative modes. If you need a non-cumulative ablation, that's a
   different feature, not this flag.

4. **CI gating still keys on `full` only.** The `.github/workflows/recall.yml`
   workflow runs `--layer full --repeats 5`; per-layer numbers are
   diagnostic and not regression-gated. Lower modes have no baseline yet
   because no production caller ever runs them.

5. **30 cases is a small sample.** A single document rank flip moves
   per-category recall by ~0.20 and per-mode ndcg by ~0.05. A `+0.01`
   per-layer delta is noise; trust direction, not magnitude.

## Regeneration history

| Date       | SHA       | Embedder       | Notes                          |
| ---------- | --------- | -------------- | ------------------------------ |
| 2026-05-03 | `6756665` | minilm-l6-v2   | Initial capture (RH-6, #268).  |
