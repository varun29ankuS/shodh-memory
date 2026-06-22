# Archived experiment workflows

These are one-shot ablation / sweep / confirm studies that have already run and
whose conclusions are banked. They live here (not under `.github/workflows/`) so
**GitHub Actions no longer schedules them** — only files directly under
`.github/workflows/` are treated as workflows. They are kept in-tree as a
reproducible record of how each result was produced.

## Why they were retired

`locomo-recall.yml` (still active) is a **parametric ablation runner**: it takes
arbitrary `SHODH_*` flags and `max_cases` as `workflow_dispatch` inputs. Every
study below is reproducible through it, so the hardcoded per-experiment YAML was
redundant clutter that only triggered on experiment branches.

## Read these numbers with three caveats

Every archived study shares the same harness configuration, and it has known
measurement limits. Do **not** treat a banked delta as production-grade without
re-confirming against them:

1. **`--repeats 1`** — a single pass over `SHODH_MAX_CASES=300`. One flipped case
   is ±0.0033 recall, so any delta below ~0.01 is at or under single-case
   granularity with **no variance estimate**. Determinism (frozen clock) makes a
   run reproducible, not statistically significant.
2. **`SHODH_MAX_CASES=300`, `SHODH_MAX_CORPUS=1500`** — a capped, gold-preserving
   fast regime. Recall here is **inflated vs full scale / held-out** (~16pp on at
   least one measured case). Winners must be re-confirmed at full corpus.
3. **NER mode unpinned** — most of these run on CI's *fallback* NER, while
   production uses *neural* NER. Entity-resolution and graph studies in
   particular may not transfer, because extraction quality is load-bearing.

To re-run any of them, dispatch `locomo-recall.yml` with the flag(s) from the
study's header comment, ideally with a higher `--repeats`, full corpus, and a
pinned NER mode.
