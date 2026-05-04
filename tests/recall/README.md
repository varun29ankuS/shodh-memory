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

## Regeneration history

| Date       | SHA       | Embedder       | Notes                          |
| ---------- | --------- | -------------- | ------------------------------ |
| 2026-05-03 | `6756665` | minilm-l6-v2   | Initial capture (RH-6, #268).  |
