#!/usr/bin/env python3
"""Format a markdown diff table comparing a recall-eval baseline to a current run.

Used by the RH-5 CI gate (`.github/workflows/recall.yml`) to post a
human-readable summary as a PR comment. Intentionally stdlib-only so the
workflow does not need a `pip install` step.

Usage:
    python scripts/recall_diff.py <baseline.json> <current.json> [--tolerance 2.0]
        [--baseline-per-case <baseline.per-case.json> --current-per-case <per-case.json>]

With both per-case files it also lists every case whose result moved, which is
what makes a red or green gate reviewable: the aggregate says THAT something
moved, the case list says WHAT.

Exits 0 always; the gating decision belongs to the `recall-eval` binary's
exit code, not to this formatter.
"""

from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Any

GATING_METRICS = ("ndcg@10", "recall@10", "mrr", "p@1")

# Metrics that count WHOLE CASES, so the smallest change they can express is
# 1/n. `p@1` is a hit count over cases; `recall@10` and `ndcg@10` are averages
# of continuous per-case values and can move arbitrarily little.
#
# This matters because the gate's tolerance is a percentage of the baseline,
# and on this suite that percentage lands BELOW one case for every quantized
# metric: at n=100 a 2% tolerance on p@1=0.31 allows 0.0062 while one case is
# 0.0100, and per-category it is worse — multi_hop has 25 cases (one case =
# 0.0400 against an allowance of 0.0056) and open_domain has 10 (one case =
# 0.1000). A "2% tolerance" that cannot admit the smallest possible change is
# not a tolerance, it is an equality check, and a single case flipping either
# way reads as a regression indistinguishable from a real one.
#
# The renderer does not get to decide that question — the Rust comparator owns
# the exit code and nothing here changes it. What it can do is stop the reader
# mistaking a resolution artefact for a measurement, so every quantized
# regression is annotated with how many cases it actually is.
QUANTIZED_METRICS = ("p@1", "precision@10", "hit@10")
INFO_METRICS = ("map", "precision@10")
LATENCY_METRICS = ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms")
# RH-12 (#272): per-case median latency distribution stats. Absent on
# pre-RH-12 reports, so the renderer skips them when both sides are zero.
LATENCY_DIST_METRICS = ("latency_min_ms", "latency_max_ms", "latency_iqr_ms")


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


def case_counts(current: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Total cases and per-category cases, as the report itself recorded them."""
    total = int(current.get("case_count") or 0)
    per_cat = {
        cat: int(v.get("case_count") or 0)
        for cat, v in (current.get("by_category") or {}).items()
        if isinstance(v, dict)
    }
    return total, per_cat


def resolution_note(detail: str, total: int, per_cat: dict[str, int]) -> str:
    """Given a regression detail line, say how many cases the drop represents.

    Detail lines look like `p@1: baseline 0.3100, current 0.3000, allowed drop
    0.0062` or `p@1[multi_hop]: ...`. Parsed rather than restructured because
    the Rust side owns this string and a renderer should not require it to
    change shape to stay readable.
    """
    m = re.match(
        r"^([a-z@0-9_]+)(?:\[([a-z_]+)\])?: baseline ([0-9.]+), current ([0-9.]+), allowed drop ([0-9.]+)",
        detail,
    )
    if not m:
        return ""
    metric, category, base, cur, allowed = m.group(1), m.group(2), *map(float, m.groups()[2:])
    if metric not in QUANTIZED_METRICS:
        return ""
    n = per_cat.get(category, 0) if category else total
    if n <= 0:
        return ""
    quantum = 1.0 / n
    drop = base - cur
    cases = drop / quantum
    note = f"n={n}, one case = {quantum:.4f}, this drop = {cases:.1f} case(s)"
    if allowed < quantum:
        note += (
            f" — the allowance ({allowed:.4f}) is BELOW one case, so on this metric "
            f"the gate is a no-net-loss check: a change passes only if it loses no "
            f"more cases than it gains. See the case list for which ones moved"
        )
    return note


# Per-case fields compared between baseline and current. `recall_at_k` is the
# production top-k recall, `p_at_1` the whole-case hit, `ndcg_at_k` the ranking
# quality that moves when a gold document changes rank without entering or
# leaving the top k.
PER_CASE_EPS = 1e-9
QUERY_PREVIEW_CHARS = 80


def load_per_case(path: Path) -> dict[str, dict[str, Any]]:
    """Index a `--per-case-output` file by `case_id`, `full` layer only.

    The file is `{"full": [record, ...]}`; only `full` is gated, so only `full`
    is compared.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("full") if isinstance(data, dict) else None
    if isinstance(records, dict):
        records = records.get("cases")
    if not isinstance(records, list):
        return {}
    return {r["case_id"]: r for r in records if isinstance(r, dict) and "case_id" in r}


def _num(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def classify_case_moves(
    baseline: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Sort every case whose result changed into exactly one bucket.

    Precedence is by consequence: a case that lost a gold document from the
    top k is reported as that even if its p@1 also moved, because losing
    reach is the more severe change and the one a reviewer must see first.
    """
    buckets: dict[str, list[dict[str, Any]]] = {
        "lost_reach": [],
        "gained_reach": [],
        "lost_p1": [],
        "gained_p1": [],
        "rank_only": [],
        "only_in_baseline": [],
        "only_in_current": [],
    }
    for case_id in sorted(set(baseline) | set(current)):
        b, c = baseline.get(case_id), current.get(case_id)
        if b is None:
            buckets["only_in_current"].append({"case_id": case_id, "record": c})
            continue
        if c is None:
            buckets["only_in_baseline"].append({"case_id": case_id, "record": b})
            continue
        d_recall = _num(c, "recall_at_k") - _num(b, "recall_at_k")
        d_p1 = _num(c, "p_at_1") - _num(b, "p_at_1")
        d_ndcg = _num(c, "ndcg_at_k") - _num(b, "ndcg_at_k")
        if max(abs(d_recall), abs(d_p1), abs(d_ndcg)) <= PER_CASE_EPS:
            continue
        move = {
            "case_id": case_id,
            "category": c.get("category") or b.get("category") or "?",
            "query": c.get("query") or b.get("query") or "",
            "recall": (_num(b, "recall_at_k"), _num(c, "recall_at_k")),
            "p1": (_num(b, "p_at_1"), _num(c, "p_at_1")),
            "ndcg": (_num(b, "ndcg_at_k"), _num(c, "ndcg_at_k")),
            "missed": c.get("missed") or [],
        }
        if d_recall < -PER_CASE_EPS:
            buckets["lost_reach"].append(move)
        elif d_recall > PER_CASE_EPS:
            buckets["gained_reach"].append(move)
        elif d_p1 < -PER_CASE_EPS:
            buckets["lost_p1"].append(move)
        elif d_p1 > PER_CASE_EPS:
            buckets["gained_p1"].append(move)
        else:
            buckets["rank_only"].append(move)
    return buckets


def _fmt_move(move: dict[str, Any]) -> str:
    query = move["query"]
    if len(query) > QUERY_PREVIEW_CHARS:
        query = query[: QUERY_PREVIEW_CHARS - 1] + "…"
    (rb, rc), (pb, pc), (nb, nc) = move["recall"], move["p1"], move["ndcg"]
    line = (
        f"- `{move['case_id']}` ({move['category']}) — recall@k {rb:.2f}→{rc:.2f}, "
        f"p@1 {pb:.0f}→{pc:.0f}, ndcg {nb:.3f}→{nc:.3f} · _{query}_"
    )
    if move["missed"]:
        line += f" · missed: {', '.join(f'`{m}`' for m in move['missed'])}"
    return line


def render_case_moves(buckets: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    moved = sum(
        len(buckets[k])
        for k in ("lost_reach", "gained_reach", "lost_p1", "gained_p1", "rank_only")
    )
    lines.append(f"### Cases that moved vs baseline ({moved})")
    lines.append("")
    if moved == 0 and not (buckets["only_in_baseline"] or buckets["only_in_current"]):
        lines.append("No case changed its result. Every per-case metric is identical to the baseline.")
        lines.append("")
        return lines
    lines.append(
        f"**{len(buckets['lost_reach'])}** lost a gold document from the top k · "
        f"**{len(buckets['gained_reach'])}** gained one · "
        f"p@1 lost **{len(buckets['lost_p1'])}** / gained **{len(buckets['gained_p1'])}** · "
        f"**{len(buckets['rank_only'])}** moved rank only"
    )
    lines.append("")
    for key, title in (
        ("lost_reach", "Lost a gold document from the top k"),
        ("lost_p1", "Lost p@1 (reach unchanged)"),
        ("gained_reach", "Gained a gold document in the top k"),
        ("gained_p1", "Gained p@1 (reach unchanged)"),
    ):
        if buckets[key]:
            lines.append(f"**{title} ({len(buckets[key])})**")
            lines.append("")
            lines.extend(_fmt_move(m) for m in buckets[key])
            lines.append("")
    if buckets["rank_only"]:
        lines.append("<details>")
        lines.append(
            f"<summary>Rank-only moves ({len(buckets['rank_only'])}): a gold document "
            f"changed position without entering or leaving the top k</summary>"
        )
        lines.append("")
        lines.extend(_fmt_move(m) for m in buckets["rank_only"])
        lines.append("")
        lines.append("</details>")
        lines.append("")
    for key, what in (
        ("only_in_baseline", "in the baseline but not in this run"),
        ("only_in_current", "in this run but not in the baseline"),
    ):
        if buckets[key]:
            ids = ", ".join(f"`{m['case_id']}`" for m in buckets[key])
            lines.append(
                f"⚠️ {len(buckets[key])} case(s) {what}: {ids}. The suite changed, so the "
                f"baseline must be regenerated for the comparison to mean anything."
            )
            lines.append("")
    return lines


def fmt_latency(base: float, cur: float) -> str:
    delta = cur - base
    return f"{base:.1f} → {cur:.1f} ({delta:+.1f})"


# Stable marker so the CI workflow can find and edit its own prior comment
# instead of stacking a fresh comment per push. Kept as an HTML comment so it
# is invisible in the rendered PR view.
COMMENT_MARKER = "<!-- recall-harness-comment-marker:rh-5 -->"


def kernel_note(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Say which CPU kernel class each side ran on, and warn when they differ.

    ONNX Runtime picks its fp32 kernels by CPUID, so identical code ranks
    differently on an AVX-512 runner and an AVX2-only one. The Rust comparator
    refuses such a pair as `infrastructure`; this makes the comment say why,
    so nobody reads the case list below it as something the change did.
    """
    def describe(report: dict[str, Any]) -> tuple[str, str]:
        kernel = report.get("kernel")
        if not isinstance(kernel, dict) or not kernel.get("class"):
            return "", "not recorded"
        model = kernel.get("cpu_model") or "unknown CPU"
        return kernel["class"], f"`{kernel['class']}` ({model})"

    base_class, base_desc = describe(baseline)
    cur_class, cur_desc = describe(current)
    lines = [f"CPU kernel class: baseline {base_desc} → current {cur_desc}"]
    if not base_class or not cur_class or base_class != cur_class:
        lines += [
            "",
            "> [!WARNING]",
            "> **Not comparable: the two runs used different, or unrecorded, CPU "
            "kernel classes.** Identical code ranks differently across classes, "
            "so any case listed below may have moved because of the runner, not "
            "the change. The gate refuses this comparison as infrastructure. "
            "Re-run the job until it lands on the baseline's class, or regenerate "
            "the baseline (tests/recall/README.md).",
        ]
    return lines


def rerank_note(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Say whether each side reranked with the cross-encoder, and warn when they differ.

    The reranker moves p@1 by about 15pp, so a `ce_rerank=0` run against the
    reranked baseline reads as a large regression no code caused. The Rust
    comparator refuses that pair; this says why in the comment.
    """
    def describe(report: dict[str, Any]) -> tuple[Any, str]:
        r = report.get("rerank")
        if not isinstance(r, dict) or "enabled" not in r:
            return None, "not recorded"
        if r["enabled"]:
            return (True, r.get("depth")), f"cross-encoder on, depth {r.get('depth')}"
        return (False, None), "cross-encoder off"

    base_key, base_desc = describe(baseline)
    cur_key, cur_desc = describe(current)
    lines = [f"Reranker: baseline {base_desc} → current {cur_desc}"]
    if base_key is None or cur_key is None or base_key != cur_key:
        lines += [
            "",
            "> [!WARNING]",
            "> **Not comparable: the two runs used different, or unrecorded, reranker "
            "settings.** They measure two pipelines, so the numbers below are not a "
            "regression or an improvement of either. The gate refuses this comparison "
            "as infrastructure.",
        ]
    return lines


def render(
    baseline: dict[str, Any],
    current: dict[str, Any],
    tolerance_pct: float,
    case_moves: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
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
    lines.extend(kernel_note(baseline, current))
    lines.extend(rerank_note(baseline, current))
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

    total_cases, per_cat_cases = case_counts(current)
    lines.append("### Quality (gated)")
    lines.append("")
    if total_cases:
        lines.append(
            f"Suite holds **{total_cases}** cases, so a whole-case metric moves in steps of "
            f"**{1.0 / total_cases:.4f}** overall"
            + (
                " ("
                + ", ".join(
                    f"{c} n={n} → {1.0 / n:.4f}"
                    for c, n in sorted(per_cat_cases.items())
                    if n > 0
                )
                + ")"
                if per_cat_cases
                else ""
            )
            + ". A drop smaller than that step cannot occur; one equal to it is a single case."
        )
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
            "`+rerank` covers the ontological re-ranker at Layer 4.9. The "
            "cross-encoder (`SHODH_CE_RERANK`, #536) is off by default and is "
            "not one of these modes."
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
        total_cases, per_cat_cases = case_counts(current)
        lines.append(f"### ❌ Regressions ({len(regressions)})")
        lines.append("")
        for f in regressions:
            detail = f.get("detail", "")
            note = resolution_note(detail, total_cases, per_cat_cases)
            lines.append(f"- {detail}")
            if note:
                lines.append(f"  - _resolution: {note}_")
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

    if case_moves is not None:
        lines.extend(render_case_moves(case_moves))
    else:
        lines.append(
            "_No per-case baseline, so which cases moved cannot be shown. The baseline "
            "is regenerated together with its `.per-case.json` (see `tests/recall/README.md`)._"
        )
        lines.append("")

    lines.append("<sub>Generated by `.github/workflows/recall.yml` (RH-5, #267).</sub>")
    return "\n".join(lines)


__all__ = [
    "COMMENT_MARKER",
    "classify_case_moves",
    "load_per_case",
    "load_report",
    "render",
    "render_case_moves",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument("--baseline-per-case", type=Path)
    p.add_argument("--current-per-case", type=Path)
    args = p.parse_args()

    baseline = load_report(args.baseline)
    current = load_report(args.current)
    case_moves = None
    if (
        args.baseline_per_case
        and args.current_per_case
        and args.baseline_per_case.is_file()
        and args.current_per_case.is_file()
    ):
        case_moves = classify_case_moves(
            load_per_case(args.baseline_per_case), load_per_case(args.current_per_case)
        )
    sys.stdout.write(render(baseline, current, args.tolerance, case_moves))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
