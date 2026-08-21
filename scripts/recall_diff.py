#!/usr/bin/env python3
"""Format a markdown diff table comparing a recall-eval baseline to a current run.

Used by the RH-5 CI gate (`.github/workflows/recall.yml`) to post a
human-readable summary as a PR comment. Intentionally stdlib-only so the
workflow does not need a `pip install` step.

Usage:
    python scripts/recall_diff.py <baseline.json> <current.json> [--tolerance 2.0]

Exits 0 always; the gating decision belongs to the `recall-eval` binary's
exit code, not to this formatter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

GATING_METRICS = ("ndcg@10", "recall@10", "mrr", "p@1")
INFO_METRICS = ("map", "precision@10")
LATENCY_METRICS = ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms")
# RH-12 (#272): per-case median latency distribution stats. Absent on
# pre-RH-12 reports, so the renderer skips them when both sides are zero.
LATENCY_DIST_METRICS = ("latency_min_ms", "latency_max_ms", "latency_iqr_ms")


# --- Uncertainty -------------------------------------------------------------
#
# Mirrors `src/recall_harness/uncertainty.rs`; kept in sync by
# `renderer_matches_the_rust_estimators` in that module's tests.
#
# The interval is NOT run-to-run noise — retrieval here is deterministic and the
# harness hard-fails if rank lists diverge across repeats (RH-12). It is the
# generalisation interval: if this same system met a different draw of questions
# from the same distribution, how far could the number move? That is what a
# reader assumes a bare point estimate has already accounted for, and it has not.
Z_95 = 1.959963984540054


def wilson_ci95(p: float, n: int) -> float:
    """95% Wilson half-width for a proportion (`p@1`).

    The normal approximation is unusable in this regime: at p=0 it reports a
    half-width of exactly zero, claiming certainty from a handful of cases.
    """
    if n <= 0:
        return 1.0
    p = min(max(p, 0.0), 1.0)
    z2 = Z_95 * Z_95
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    spread = (Z_95 / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(abs(p - max(centre - spread, 0.0)), abs(min(centre + spread, 1.0) - p))


def bounded_mean_ci95(n: int) -> float:
    """95% half-width for a mean of per-case scores in [0,1].

    Popoviciu's bound (variance <= 1/4) — a rigorous upper bound, since the
    report does not carry per-case variance. Never optimistic.
    """
    if n <= 0:
        return 1.0
    return min(Z_95 * 0.5 / math.sqrt(n), 1.0)


def ci_for(metric: str, value: float, n: int) -> float:
    return wilson_ci95(value, n) if metric == "p@1" else bounded_mean_ci95(n)


def fmt_uncertainty(metric: str, base: float, cur: float, n: int) -> str:
    """`±0.0980 (n=100)`, plus a flag when the delta is under the resolution."""
    ci = ci_for(metric, base, n)
    within = abs(cur - base) < ci
    tag = " · **within sampling error**" if within else ""
    return f"±{ci:.4f} (n={n}){tag}"


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(base: float, cur: float, tolerance_pct: float, gating: bool) -> str:
    """Render `base → cur (Δ)` with a status marker for gating metrics."""
    delta = cur - base
    delta_str = f"{delta:+.4f}"
    base_str = f"{base:.4f}"
    cur_str = f"{cur:.4f}"
    if not gating or base <= 0.0:
        return f"{base_str} → {cur_str} ({delta_str})"
    allowed_drop = base * (tolerance_pct / 100.0)
    # Match the rust comparator exactly: regression iff cur + allowed_drop < base.
    if cur + allowed_drop < base:
        marker = "❌"
    elif delta < 0.0:
        marker = "⚠️"
    else:
        marker = "✅"
    return f"{base_str} → {cur_str} ({delta_str}) {marker}"


def fmt_latency(base: float, cur: float) -> str:
    delta = cur - base
    return f"{base:.1f} → {cur:.1f} ({delta:+.1f})"


# Stable marker so the CI workflow can find and edit its own prior comment
# instead of stacking a fresh comment per push. Kept as an HTML comment so it
# is invisible in the rendered PR view.
COMMENT_MARKER = "<!-- recall-harness-comment-marker:rh-5 -->"


def render(baseline: dict[str, Any], current: dict[str, Any], tolerance_pct: float) -> str:
    lines: list[str] = []
    lines.append(COMMENT_MARKER)
    lines.append("## Recall harness — smoke suite")
    lines.append("")
    lines.append(
        f"Baseline `{baseline.get('git_sha', '?')[:7]}` "
        f"({baseline.get('embedder', '?')}) → "
        f"current `{current.get('git_sha', '?')[:7]}` "
        f"({current.get('embedder', '?')}) · tolerance **{tolerance_pct:.1f}%**"
    )
    base_repeats = baseline.get("repeats", 1)
    cur_repeats = current.get("repeats", 1)
    lines.append(
        f"Repeats: baseline **{base_repeats}** → current **{cur_repeats}** "
        f"(per-case latency is the median across repeats; rank lists must "
        f"be byte-identical across all repeats — see RH-12, #272)"
    )
    lines.append("")

    base_full = baseline.get("layers", {}).get("full", {})
    cur_full = current.get("layers", {}).get("full", {})
    if not base_full or not cur_full:
        lines.append("> **Infrastructure failure:** one or both reports are missing the `full` layer.")
        return "\n".join(lines)

    n_cases = int(current.get("case_count", 0) or baseline.get("case_count", 0) or 0)

    lines.append("### Quality (gated)")
    lines.append("")
    lines.append("| metric | baseline → current (Δ) | 95% CI |")
    lines.append("| ------ | ---------------------- | ------ |")
    for m in GATING_METRICS:
        b, c = base_full.get(m, 0.0), cur_full.get(m, 0.0)
        lines.append(
            f"| `{m}` | {fmt_metric(b, c, tolerance_pct, gating=True)} "
            f"| {fmt_uncertainty(m, b, c, n_cases)} |"
        )
    for m in INFO_METRICS:
        b, c = base_full.get(m, 0.0), cur_full.get(m, 0.0)
        lines.append(
            f"| `{m}` | {fmt_metric(b, c, tolerance_pct, gating=False)} "
            f"| {fmt_uncertainty(m, b, c, n_cases)} |"
        )
    lines.append("")
    smallest = bounded_mean_ci95(n_cases)
    lines.append(
        f"> Deltas smaller than **±{smallest:.4f}** are below what {n_cases} questions can "
        f"resolve. They are still real changes *on this fixed benchmark* — retrieval is "
        f"deterministic — but they are not evidence about quality in general. The fix for an "
        f"under-resolved metric is more questions, never a looser tolerance."
    )
    lines.append("")

    lines.append("### Latency (informational)")
    lines.append("")
    lines.append("| metric | baseline → current (Δ ms) |")
    lines.append("| ------ | ------------------------- |")
    for m in LATENCY_METRICS:
        lines.append(f"| `{m}` | {fmt_latency(base_full.get(m, 0.0), cur_full.get(m, 0.0))} |")
    # RH-12 distribution stats: only render when at least one side reports
    # them, so old baselines (all zeros) don't pollute the table.
    has_dist_stats = any(
        base_full.get(m, 0.0) != 0.0 or cur_full.get(m, 0.0) != 0.0
        for m in LATENCY_DIST_METRICS
    )
    if has_dist_stats:
        for m in LATENCY_DIST_METRICS:
            lines.append(
                f"| `{m}` | {fmt_latency(base_full.get(m, 0.0), cur_full.get(m, 0.0))} |"
            )
    lines.append("")

    # RH-8 (#270): per-pipeline-layer attribution. The harness can run the
    # smoke suite under up to six cumulative modes; render a delta table
    # only for modes shared between baseline and current. When only `full`
    # is present (the default CI path), this section gracefully degrades.
    # Mode order is the production pipeline order, not BTreeMap order, so
    # ndcg deltas read top-to-bottom as additive stages.
    PIPELINE_MODE_ORDER = (
        "vamana_only",
        "+spreading",
        "+bm25",
        "+rerank",
        "+facts",
        "full",
    )
    base_layers = baseline.get("layers", {})
    cur_layers = current.get("layers", {})
    shared_modes = [m for m in PIPELINE_MODE_ORDER if m in base_layers and m in cur_layers]
    # Surface this section only when there is something beyond `full` to
    # show — otherwise it duplicates the gated table above.
    if len(shared_modes) > 1:
        lines.append("### Per-layer attribution `ndcg@10` / `recall@10`")
        lines.append("")
        lines.append(
            "Cumulative modes (each row adds one stage to the row above). "
            "Per-layer numbers are diagnostic — only `full` is gated by CI. "
            "`+rerank` covers the ontological re-ranker at Layer 4.9; this "
            "codebase has no cross-encoder despite the spec label."
        )
        lines.append("")
        lines.append("| mode | `ndcg@10` (Δ) | `recall@10` (Δ) |")
        lines.append("| ---- | ------------- | --------------- |")
        for m in shared_modes:
            b_layer = base_layers.get(m, {})
            c_layer = cur_layers.get(m, {})
            ndcg = fmt_metric(
                b_layer.get("ndcg@10", 0.0),
                c_layer.get("ndcg@10", 0.0),
                tolerance_pct,
                gating=False,
            )
            recall = fmt_metric(
                b_layer.get("recall@10", 0.0),
                c_layer.get("recall@10", 0.0),
                tolerance_pct,
                gating=False,
            )
            lines.append(f"| `{m}` | {ndcg} | {recall} |")
        lines.append("")

    base_cats = baseline.get("by_category", {})
    cur_cats = current.get("by_category", {})
    cats = sorted(set(base_cats) | set(cur_cats))
    if cats:
        lines.append("### Per-category `ndcg@10`")
        lines.append("")
        lines.append("| category | baseline → current (Δ) |")
        lines.append("| -------- | ---------------------- |")
        for c in cats:
            b = base_cats.get(c, {}).get("ndcg@10", 0.0)
            cur_v = cur_cats.get(c, {}).get("ndcg@10", 0.0)
            lines.append(f"| `{c}` | {fmt_metric(b, cur_v, tolerance_pct, gating=False)} |")
        lines.append("")

    failures = current.get("failures") or []
    regressions = [f for f in failures if f.get("kind") == "regression"]
    infra = [f for f in failures if f.get("kind") == "infrastructure"]
    cases = [f for f in failures if f.get("kind") == "case"]

    if regressions:
        lines.append(f"### ❌ Regressions ({len(regressions)})")
        lines.append("")
        for f in regressions:
            lines.append(f"- {f.get('detail', '')}")
        lines.append("")
    if infra:
        lines.append(f"### ⚠️ Infrastructure failures ({len(infra)})")
        lines.append("")
        for f in infra:
            lines.append(f"- {f.get('detail', '')}")
        lines.append("")
    if cases:
        lines.append(f"### ⚠️ Per-case failures ({len(cases)})")
        lines.append("")
        for f in cases[:10]:
            lines.append(f"- {f.get('detail', '')}")
        if len(cases) > 10:
            lines.append(f"- …and {len(cases) - 10} more")
        lines.append("")

    if not (regressions or infra or cases):
        lines.append("### ✅ No regressions")
        lines.append("")
        lines.append(f"All {len(GATING_METRICS)} gated metrics within {tolerance_pct:.1f}% tolerance.")
        lines.append("")

    lines.append("<sub>Generated by `.github/workflows/recall.yml` (RH-5, #267).</sub>")
    return "\n".join(lines)


__all__ = ["COMMENT_MARKER", "render", "load_report"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("--tolerance", type=float, default=2.0)
    args = p.parse_args()

    baseline = load_report(args.baseline)
    current = load_report(args.current)
    sys.stdout.write(render(baseline, current, args.tolerance))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
