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
import sys
from pathlib import Path
from typing import Any

GATING_METRICS = ("ndcg@10", "recall@10", "mrr", "p@1")
INFO_METRICS = ("map", "precision@10")
LATENCY_METRICS = ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms")


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
    lines.append("")

    base_full = baseline.get("layers", {}).get("full", {})
    cur_full = current.get("layers", {}).get("full", {})
    if not base_full or not cur_full:
        lines.append("> **Infrastructure failure:** one or both reports are missing the `full` layer.")
        return "\n".join(lines)

    lines.append("### Quality (gated)")
    lines.append("")
    lines.append("| metric | baseline → current (Δ) |")
    lines.append("| ------ | ---------------------- |")
    for m in GATING_METRICS:
        lines.append(
            f"| `{m}` | {fmt_metric(base_full.get(m, 0.0), cur_full.get(m, 0.0), tolerance_pct, gating=True)} |"
        )
    for m in INFO_METRICS:
        lines.append(
            f"| `{m}` | {fmt_metric(base_full.get(m, 0.0), cur_full.get(m, 0.0), tolerance_pct, gating=False)} |"
        )
    lines.append("")

    lines.append("### Latency (informational)")
    lines.append("")
    lines.append("| metric | baseline → current (Δ ms) |")
    lines.append("| ------ | ------------------------- |")
    for m in LATENCY_METRICS:
        lines.append(f"| `{m}` | {fmt_latency(base_full.get(m, 0.0), cur_full.get(m, 0.0))} |")
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
