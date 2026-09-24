#!/usr/bin/env python3
"""Diff two `SHODH_STAGE_EXPORT` files stage by stage.

The recall harness writes, under `SHODH_STAGE_EXPORT=<path>`, one JSON line per
corpus item (`ingest`: stored embedding and NER spans), one for the graph
(`graph`: node and edge sets keyed by entity name), and one per case (`case`:
each retrieval leg, the fused ranking, the reranker's pool and scores, and the
final ranking). Scores are f32 bit patterns, so any move is visible.

Given two such files, from two machines, commits or settings, this reports the
first stage at which they part, and for each stage whether the ORDER changed or
only the SCORES did. That is how the cross-runner divergence in #566 was found:
embeddings and NER scores moved with the CPU kernel class, while NER span sets
and graph membership did not.

Usage:
    python scripts/stage_diff.py <a.jsonl> <b.jsonl>

Stdlib only, like recall_diff.py.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Per-case keys that hold a plain ranked id list, in pipeline order.
RANK_KEYS = ("pool", "deep", "ranks", "final_captured")


def load(path: Path) -> dict[str, Any]:
    ingest: dict[str, dict[str, Any]] = {}
    graph: dict[str, Any] | None = None
    cases: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "ingest":
                ingest[record["id"]] = record
            elif kind == "graph":
                graph = record
            elif kind == "case":
                cases[record["case_id"]] = record
    return {"ingest": ingest, "graph": graph, "cases": cases}


def span_set(ner: list[str]) -> set[str]:
    """NER spans without their confidence bits: which entities, not how sure."""
    out = set()
    for span in ner:
        parts = span.split("|")
        # text | type | fine label | confidence bits | start | end
        out.add("|".join(parts[:3] + parts[4:]))
    return out


def membership(items: list[str]) -> set[str]:
    """Graph node or edge records without their trailing numeric field."""
    return {x.rsplit("|", 1)[0] for x in items}


def scored_stages(case: dict[str, Any]) -> dict[str, list[list[Any]]]:
    stages = {"leg:" + s["stage"]: s["items"] for s in case.get("legs", [])}
    stages.update({s["stage"]: s["items"] for s in case.get("ce", [])})
    return stages


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    shared = [k for k in a["ingest"] if k in b["ingest"]]
    result: dict[str, Any] = {
        "items": len(shared),
        "embedding_differs": [
            k for k in shared
            if a["ingest"][k]["embedding_sha256"] != b["ingest"][k]["embedding_sha256"]
        ],
        "ner_scores_differ": [
            k for k in shared if a["ingest"][k]["ner_sha256"] != b["ingest"][k]["ner_sha256"]
        ],
        "ner_spans_differ": [
            k for k in shared
            if span_set(a["ingest"][k]["ner"]) != span_set(b["ingest"][k]["ner"])
        ],
    }
    ga, gb = a["graph"], b["graph"]
    if ga and gb:
        result["nodes"] = (
            sorted(membership(ga["nodes"]) - membership(gb["nodes"])),
            sorted(membership(gb["nodes"]) - membership(ga["nodes"])),
        )
        result["edges"] = (
            sorted(membership(ga["edges"]) - membership(gb["edges"])),
            sorted(membership(gb["edges"]) - membership(ga["edges"])),
        )
    order: dict[str, list[str]] = defaultdict(list)
    scores: dict[str, list[str]] = defaultdict(list)
    shared_cases = [c for c in a["cases"] if c in b["cases"]]
    for cid in shared_cases:
        ca, cb = a["cases"][cid], b["cases"][cid]
        for key in RANK_KEYS:
            if ca.get(key) != cb.get(key):
                order[key].append(cid)
        sa, sb = scored_stages(ca), scored_stages(cb)
        for stage in sa.keys() & sb.keys():
            ids_a = [x[0] for x in sa[stage]]
            ids_b = [x[0] for x in sb[stage]]
            if ids_a != ids_b:
                order[stage].append(cid)
            elif sa[stage] != sb[stage]:
                scores[stage].append(cid)
    result["cases"] = len(shared_cases)
    result["order"] = dict(order)
    result["scores"] = dict(scores)
    return result


def render(r: dict[str, Any]) -> str:
    n = r["items"]
    lines = [
        f"ingest ({n} items):",
        f"  embedding bytes differ   {len(r['embedding_differs'])}/{n}",
        f"  NER scores differ        {len(r['ner_scores_differ'])}/{n}",
        f"  NER span sets differ     {len(r['ner_spans_differ'])}/{n}  {r['ner_spans_differ'][:5]}",
    ]
    for key in ("nodes", "edges"):
        if key in r:
            only_a, only_b = r[key]
            lines.append(f"  graph {key}: only in A {len(only_a)}, only in B {len(only_b)}")
            lines += [f"    A: {x}" for x in only_a[:5]] + [f"    B: {x}" for x in only_b[:5]]
    lines.append(f"cases ({r['cases']}):")
    stages = sorted(r["order"].keys() | r["scores"].keys())
    if not stages:
        lines.append("  every stage identical")
    for stage in stages:
        o, s = r["order"].get(stage, []), r["scores"].get(stage, [])
        lines.append(f"  {stage:<18} order differs {len(o):>3}   scores only {len(s):>3}  {o[:6]}")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().split("\n\n")[-2], file=sys.stderr)
        return 2
    a, b = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    print(render(compare(a, b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
