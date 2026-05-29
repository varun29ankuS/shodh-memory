#!/usr/bin/env python3
"""LoCoMo-MC10 per-layer attribution benchmark for shodh-memory.

Runs the LoCoMo-MC10 multiple-choice benchmark across the six cumulative
RH-8 pipeline layer modes (`vamana_only` -> `+spreading` -> `+bm25` ->
`+rerank` -> `+facts` -> `full`) and reports, per layer, both the
retrieval characteristics (latency, retrieval-set diff vs `full`) and
the end-to-end answer accuracy (LLM judges the retrieved context).

Why this exists:
  The 30-case L1 smoke harness measures regression on a tiny synthetic
  fixture where Layer 5's recency/importance/feedback signals have no
  variance to amplify. LoCoMo-MC10 ships ~1,986 multi-turn dialogue
  questions with a real ground truth (correct multiple-choice index),
  which is closer to the agent-memory regime shodh is built for.

What it does NOT do:
  - It does not measure recall@k against gold-evidence memory IDs;
    LoCoMo-MC10 has no gold memory annotation. Quality per layer is
    inferred from end-to-end LLM accuracy on the retrieved context.
  - It does not gate CI; this is a diagnostic harness.

Usage:
  Server must be running on http://127.0.0.1:3030 before invocation.

  # Cheap default — judge `full` only, latency + retrieval drift for the rest
  python locomo_layer_eval.py --provider openai-compatible \\
      --api-base https://api.groq.com/openai/v1 \\
      --model llama-3.1-8b-instant --limit 100

  # Full per-layer accuracy (6x the LLM calls)
  python locomo_layer_eval.py --provider openai-compatible \\
      --api-base https://api.groq.com/openai/v1 \\
      --model llama-3.1-8b-instant --limit 100 --judge-all-layers
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Reuse the working pieces from the existing eval; this script is the
# per-layer wrapper, not a fork. If `locomo_mc10_eval.py` evolves, this
# script picks up the changes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomo_mc10_eval import (  # noqa: E402  (intentional sys.path hack)
    ShodhMemoryClient,
    create_provider,
    select_answer_with_llm,
    store_conversations,
)

# RH-8 cumulative ladder. Ordering matters: each row adds one stage to
# the row above it. The HTTP server normalises both `vamana_only` and
# `vamana-only`; we send the canonical underscore form.
LAYER_MODES: tuple[str, ...] = (
    "vamana_only",
    "+spreading",
    "+bm25",
    "+rerank",
    "+facts",
    "full",
)


def recall_with_layer(
    client: ShodhMemoryClient,
    query: str,
    limit: int,
    layer: str,
) -> tuple[list[dict], float]:
    """Recall against a specific RH-8 layer mode.

    Returns the raw API memory list (each entry has experience.{id,content})
    plus wall-clock latency in milliseconds. Errors propagate — this is a
    diagnostic harness, swallowing them would lie about layer behaviour.
    """
    payload = {
        "user_id": client.user_id,
        "query": query,
        "limit": limit,
        "mode": "hybrid",
        "layers": layer,
    }
    start = time.perf_counter()
    resp = client.session.post(f"{client.base_url}/api/recall", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    resp.raise_for_status()
    return resp.json().get("memories", []), elapsed_ms


def memories_to_id_set(memories: list[dict]) -> tuple[str, ...]:
    """Stable ordered tuple of memory IDs for set-comparison + hashing.

    The API returns `experience.id` per memory; falls back to content
    hash if a server build hides the id (older snapshots).
    """
    ids: list[str] = []
    for m in memories:
        exp = m.get("experience") or {}
        mid = exp.get("id")
        if mid is None:
            content = (exp.get("content") or "")[:200]
            mid = f"content:{hash(content)}"
        ids.append(str(mid))
    return tuple(ids)


def memories_to_context(memories: list[dict]) -> str:
    """Format retrieved memories as the LLM-judge prompt context.

    Identical formatting to `locomo_mc10_eval.recall_context` so the
    accuracy numbers from this script can be compared head-to-head with
    the single-mode eval.
    """
    if not memories:
        return "(No relevant memories found)"
    return "\n\n".join(
        f"[Memory {i + 1}]: {m.get('experience', {}).get('content', '')}"
        for i, m in enumerate(memories)
    )


def jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Jaccard overlap between two retrieval sets. 1.0 = identical sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


@dataclass
class PerLayerResult:
    layer: str
    latency_ms: float
    retrieved_ids: tuple[str, ...]
    judged: bool = False
    predicted_idx: int = -1
    correct: bool = False
    context_chars: int = 0


@dataclass
class ItemResult:
    question_id: str
    question_type: str
    correct_idx: int
    num_stored: int
    store_latency_ms: float
    layers: dict[str, PerLayerResult] = field(default_factory=dict)


def evaluate_item(
    item: dict,
    provider,
    client: ShodhMemoryClient,
    limit: int,
    judge_all_layers: bool,
) -> ItemResult:
    """Store sessions once, recall under each layer mode, optionally judge each."""
    client.user_id = f"locomo_layer_{item['question_id']}"

    sessions = item.get("haystack_sessions", [])
    summaries = item.get("haystack_session_summaries", [])
    datetimes = item.get("haystack_session_datetimes", [])
    num_stored, store_latency = store_conversations(client, sessions, summaries, datetimes)

    result = ItemResult(
        question_id=item["question_id"],
        question_type=item["question_type"],
        correct_idx=item["correct_choice_index"],
        num_stored=num_stored,
        store_latency_ms=store_latency,
    )

    for layer in LAYER_MODES:
        memories, latency = recall_with_layer(client, item["question"], limit, layer)
        ids = memories_to_id_set(memories)
        context = memories_to_context(memories)
        per = PerLayerResult(
            layer=layer,
            latency_ms=latency,
            retrieved_ids=ids,
            context_chars=len(context),
        )
        # Judge `full` always (so single-LLM-call mode still returns one
        # quality number) and the rest only when the caller asks for it.
        if judge_all_layers or layer == "full":
            predicted = select_answer_with_llm(
                provider=provider,
                question=item["question"],
                choices=item["choices"],
                context=context,
            )
            per.judged = True
            per.predicted_idx = predicted
            per.correct = predicted == item["correct_choice_index"]
        result.layers[layer] = per

    return result


def percentile(values: list[float], pct: float) -> float:
    """statistics-only percentile so the script stays stdlib-only."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    # statistics.quantiles cuts into n equal-frequency intervals; use n=100
    # and pick the appropriate cut. inclusive method matches numpy.percentile
    # at integer percentiles closely enough for diagnostic latency reporting.
    cuts = statistics.quantiles(sorted(values), n=100, method="inclusive")
    # cuts has 99 entries (Q1..Q99); index pct-1 for 1<=pct<=99.
    idx = max(0, min(98, int(round(pct)) - 1))
    return cuts[idx]


def aggregate(results: list[ItemResult], judge_all_layers: bool) -> dict:
    """Build the per-layer x per-question_type breakdown from raw results."""
    out: dict = {
        "total_items": len(results),
        "judge_all_layers": judge_all_layers,
        "layers": {},
        "by_question_type": {},
    }

    full_ids_by_qid = {r.question_id: r.layers["full"].retrieved_ids for r in results}

    for layer in LAYER_MODES:
        latencies = [r.layers[layer].latency_ms for r in results]
        # Jaccard vs full for the same question — meaningful for non-full layers
        jaccs = [
            jaccard(r.layers[layer].retrieved_ids, full_ids_by_qid[r.question_id])
            for r in results
        ]
        judged = [r.layers[layer] for r in results if r.layers[layer].judged]
        accuracy = (
            sum(1 for j in judged if j.correct) / len(judged) * 100.0 if judged else None
        )

        out["layers"][layer] = {
            "latency_ms_avg": sum(latencies) / len(latencies),
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
            "latency_ms_p99": percentile(latencies, 99),
            "jaccard_vs_full_avg": sum(jaccs) / len(jaccs),
            "judged_count": len(judged),
            "accuracy_pct": accuracy,
        }

    # Per question_type x layer accuracy table — only meaningful when judged
    by_type_layers: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_type_count: dict[str, int] = defaultdict(int)
    for r in results:
        by_type_count[r.question_type] += 1
        for layer, per in r.layers.items():
            if per.judged:
                by_type_layers[r.question_type][layer].append(per.correct)

    for qtype, count in by_type_count.items():
        layer_accuracy: dict[str, Optional[float]] = {}
        for layer in LAYER_MODES:
            judged_for_layer = by_type_layers[qtype].get(layer, [])
            if judged_for_layer:
                layer_accuracy[layer] = (
                    sum(judged_for_layer) / len(judged_for_layer) * 100.0
                )
            else:
                layer_accuracy[layer] = None
        out["by_question_type"][qtype] = {
            "count": count,
            "layer_accuracy_pct": layer_accuracy,
        }

    return out


def print_summary(agg: dict) -> None:
    print("\n" + "=" * 72)
    print("LoCoMo-MC10 — per-layer attribution")
    print("=" * 72)
    print(
        f"\nTotal items: {agg['total_items']}   "
        f"judge_all_layers: {agg['judge_all_layers']}\n"
    )
    print(
        f"{'layer':<14} {'latency p50':>12} {'p95':>8} {'avg':>8}"
        f"   {'jaccard@full':>12}   {'accuracy':>10}"
    )
    print("-" * 72)
    for layer in LAYER_MODES:
        s = agg["layers"][layer]
        acc = "—" if s["accuracy_pct"] is None else f"{s['accuracy_pct']:>8.2f}%"
        print(
            f"{layer:<14} {s['latency_ms_p50']:>10.1f}ms "
            f"{s['latency_ms_p95']:>6.1f}ms "
            f"{s['latency_ms_avg']:>6.1f}ms"
            f"     {s['jaccard_vs_full_avg']:>9.3f}     {acc}"
        )

    print("\nAccuracy by question_type x layer:")
    print("-" * 72)
    qtypes = sorted(agg["by_question_type"].keys())
    header = f"{'qtype':<22} {'n':>4}  " + "  ".join(f"{l:>10}" for l in LAYER_MODES)
    print(header)
    for qtype in qtypes:
        row = agg["by_question_type"][qtype]
        cells = []
        for layer in LAYER_MODES:
            v = row["layer_accuracy_pct"][layer]
            cells.append("    —    " if v is None else f"{v:>8.2f}% ")
        print(f"{qtype:<22} {row['count']:>4}  " + "  ".join(cells))
    print()


def load_dataset(path: str, limit: Optional[int]) -> list[dict]:
    if not os.path.exists(path):
        print(f"Dataset not found at {path}")
        print(
            "Download with:\n"
            "  python -c \"from datasets import load_dataset; "
            "load_dataset('Percena/locomo-mc10')\""
        )
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if limit:
        data = data[: min(limit, len(data))]
    return data


def serialise_results(results: list[ItemResult]) -> list[dict]:
    """Per-item dump suitable for JSON; tuples -> lists for stability."""
    out = []
    for r in results:
        d = asdict(r)
        for layer, per in d["layers"].items():
            per["retrieved_ids"] = list(per["retrieved_ids"])
        out.append(d)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", default="openai-compatible")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of LoCoMo-MC10 items to evaluate (default: 100). "
        "Smaller numbers are noisy — see RH-9 for the rationale.",
    )
    parser.add_argument(
        "--recall-limit",
        type=int,
        default=5,
        help="top-k passed to /api/recall per layer (default: 5).",
    )
    parser.add_argument(
        "--judge-all-layers",
        action="store_true",
        help="Run the LLM judge on every layer's context (6x LLM cost). "
        "Without this flag, only `full` is judged and the cheaper layers "
        "report latency + Jaccard-vs-full only.",
    )
    parser.add_argument("--shodh-url", default="http://127.0.0.1:3030")
    parser.add_argument(
        "--shodh-api-key",
        default="sk-shodh-dev-local-testing-key",
        help="Server's X-API-Key. Default works against the dev server.",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/"
            "snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/data/locomo_mc10.json"
        ),
    )
    parser.add_argument("--output", default="locomo_layer_results.json")
    args = parser.parse_args()

    # Test server reachability before anything else — otherwise tqdm
    # spins for an hour before the first connection error.
    try:
        resp = requests.get(f"{args.shodh_url}/health", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: cannot reach shodh server at {args.shodh_url}: {e}")
        return 2

    client = ShodhMemoryClient(base_url=args.shodh_url, api_key=args.shodh_api_key)
    provider = create_provider(args.provider, args.model, args.api_base, args.api_key)

    dataset = load_dataset(args.dataset_path, args.limit)
    print(f"Evaluating {len(dataset)} items across {len(LAYER_MODES)} layers")
    print(f"Provider: {args.provider} / model: {args.model}")
    print(f"Judge all layers: {args.judge_all_layers}\n")

    results: list[ItemResult] = []
    for item in tqdm(dataset, desc="layer-eval"):
        try:
            results.append(
                evaluate_item(
                    item=item,
                    provider=provider,
                    client=client,
                    limit=args.recall_limit,
                    judge_all_layers=args.judge_all_layers,
                )
            )
        except requests.HTTPError as e:
            # Server-side rejection (e.g. unknown layer mode) — abort
            # rather than silently continuing with skewed aggregates.
            print(f"\nFATAL: HTTP error on item {item.get('question_id')}: {e}")
            return 2

    agg = aggregate(results, args.judge_all_layers)
    print_summary(agg)

    out = {
        "config": {
            "provider": args.provider,
            "model": args.model,
            "limit": args.limit,
            "recall_limit": args.recall_limit,
            "judge_all_layers": args.judge_all_layers,
            "shodh_url": args.shodh_url,
            "layer_modes": list(LAYER_MODES),
        },
        "summary": agg,
        "results": serialise_results(results),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Detailed results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
