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

## The PR gate baseline (`locomo-gate`)

CI's L1 Smoke Suite (`.github/workflows/recall.yml`) runs the 100-case
`locomo-gate` suite and compares against two checked-in files:

- `locomo-gate-baseline.json`: aggregate metrics. The `recall-eval` binary
  gates on these; its exit code is the pass/fail.
- `locomo-gate-baseline.per-case.json`: the per-case records from the same run.
  `scripts/recall_diff.py` uses them to list, in the PR comment, every case
  whose result moved, bucketed into lost gold / gained gold / p@1 flips /
  rank-only.

On this suite the 2% tolerance is below one case for p@1 (n=100, one case =
0.01), so for that metric the gate is a no-net-loss check. Within one CPU
kernel class the runs are deterministic: five repeats must be byte-identical,
and identical code gives identical per-case results on every runner of that
class. So within a class, any case that moves is a real change, not noise. The
case list is how a reviewer decides whether a change is worth what it moved.

**Across kernel classes, identical code does not give identical results.**
ONNX Runtime picks its fp32 kernels by CPUID, and no setting pins the choice.
GitHub's `ubuntu-latest` pool mixes AVX2-only runners (mostly AMD EPYC 7763)
with AVX-512 ones (EPYC 9V74, Xeon Platinum 8573C, Xeon 6973P-C). The two
classes give different low bits in all three models. In the cross-encoder the
difference reaches 0.06 in logit, and on the gate it moved 1 of 100 cases,
with 14 reranker pools changing
(runs 35901281838 and 35902501577, 2026-09-23). A hypervisor can hide AVX-512
from a chip that has it, and such a runner lands in the AVX2 class. So the
class is what the runner *exposes*. Each report records it as `kernel.class`,
with the CPU model and features alongside. `recall-eval` refuses to compare
two classes as `infrastructure`, and the workflow fails a PR run on the wrong
class within seconds. Re-run the job: GitHub assigns runner hardware per job.
One gate run on the same tree (35894635781) matched neither class and was
never reproduced across 16 sampled runners. If a same-class comparison ever
moves cases on unchanged code, the recorded `cpu_model` and `features` are the
first thing to compare.

**Regenerate both files together, from the same run, on `main`, after a merged
change that intentionally moved quality.** A stale baseline hides regressions:
while main sat 3.5pp of recall@10 above the baseline, a PR could lose up to
that much and still pass.

1. `gh workflow run recall.yml --ref main` (defaults: `locomo-gate`, `full`,
   5 repeats, `ce_rerank=1`). The gate measures the pipeline the server
   ships, which reranks with the cross-encoder by default, so the baseline is
   recorded with it on. `-f ce_rerank=0` measures the library default for
   comparison and must not be committed as the baseline.
2. `gh run download <run-id> -n recall-eval-report`.
3. Copy `current.json` to `locomo-gate-baseline.json` and `per-case.json` to
   `locomo-gate-baseline.per-case.json`. Check that `git_sha` names the main
   commit you meant, that `repeats` is 5, and that `kernel.class` is
   `x86_64-avx2-fma`. That is the most common class in the pool, so it is
   the one PR runs land on most often. If the run landed on another class,
   dispatch again.
4. Add a row to the regeneration history below, and in the PR list the cases
   that moved since the previous baseline.

## Regenerating the baseline

Only regenerate `baseline.json` when you have *intentionally* changed retrieval
quality (embedder swap, scoring tweak, pipeline refactor) and want to freeze the
new numbers. Routine refactors should leave it untouched — that is the whole point.

```bash
cargo run --release --bin recall-eval -- \
    --suite smoke \
    --storage "$LOCALAPPDATA/shodh-eval/baseline" \
    --output tests/recall/baseline.json
```

The binary records the current git SHA, embedder identifier, and timestamp into
the report header so the baseline is self-describing.

**`--storage` MUST point outside directories watched by a search indexer,
antivirus, or sync daemon** — in practice: avoid the user-profile Documents
tree, use `%LOCALAPPDATA%` (excluded from Windows Search indexing by default).
Watcher processes intermittently lock freshly written tantivy segment files,
which silently drops BM25 commit batches and makes rankings depend on ambient
machine state (root-caused 2026-06-12: 432/432 smoke comparisons diverged with
storage under the Documents tree; 0–1 under `%LOCALAPPDATA%`). The harness now
hard-fails ingest when a commit batch is lost after retries, so a watched path
errors out rather than producing garbage numbers.

The eval scoring clock is frozen (`SHODH_EVAL_NOW`, pinned to a fixed anchor in
`pin_harness_threads`) so recency components cannot drift between repeat
passes or rot the baseline as the static corpus ages against wall-clock time.

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
   as a cross-encoder rerank stage, but it wraps the **ontological
   re-ranker** at Layer 4.9 (multiplicative boost when episode entity types
   match the query's expected ontology labels). The cross-encoder that
   landed in #536 is a separate stage behind `SHODH_CE_RERANK`, off by
   default, and is not part of any `--layer` mode.

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
| 2026-08-08 | `ec7abd2` | minilm-l6-v2   | `locomo-gate` aggregate baseline, 1 repeat, no per-case file. |
| 2026-09-22 | `cef6721` | minilm-l6-v2   | `locomo-gate` + per-case, 5 repeats (workflow run 35693977402). After #509 (keyphrases stop becoming graph nodes; q62 lost, q46/q52 gained) and #560 (BM25 stemming; q62, q129, q85 gained, 18 rank-only moves, none lost). recall@10 0.5268 → 0.5618, ndcg@10 0.4111 → 0.4248, p@1 0.31 → 0.31. |
| 2026-09-24 | `628cbeb` | minilm-l6-v2   | `locomo-gate` + per-case, 5 repeats, cross-encoder on, **kernel class `x86_64-avx2-fma`** (AMD EPYC 7763, workflow run 35956028151). First baseline recorded with the reranker on and the first to record its CPU kernel class. Measured on the PR branch that adds the class field, because a baseline without the field is refused, so it could not come from main first. Per-case results are identical to main `681b2d8` on an AVX2 runner (run 35891724716, 0 of 100 differ). `failures` was cleared by hand: it held that run's refusal of the previous class-less baseline. 43 cases moved vs `cef6721`: 14 gained gold, 3 lost gold (q125, q30, q43), p@1 +13/−1 (q62), 12 rank-only. recall@10 0.5618 → 0.6368, ndcg@10 0.4248 → 0.5409, mrr 0.4042 → 0.5372, p@1 0.31 → 0.46. Latency p50 535 ms. AVX2 runners are the slow class: the same tree ran at 216–240 ms p50 on AVX-512 runners. |
