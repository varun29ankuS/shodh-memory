#!/usr/bin/env python3
"""Convert the original LoCoMo dataset into recall-harness fixtures.

LoCoMo (snap-research/locomo, `data/locomo10.json`) ships 10 multi-session
dialogues plus QA pairs whose `evidence` lists the dialogue turns (`dia_id`)
that support each answer. That evidence is a GOLD memory-ID annotation — so we
can measure recall@k directly, with no LLM judge, exactly like the smoke suite.

This is the HELD-OUT generalization set: none of the pipeline changes were
diagnosed against it. recall.yml tunes on the 108 smoke cases; running these
LoCoMo fixtures through the same `recall-eval` tells us whether a recall fix
generalized or just fit the smoke corpus.

Output (recall-harness JSONL):
  tests/recall/corpora/locomo.jsonl  — one CorpusItem per dialogue turn
  tests/recall/locomo_cases.jsonl    — one SmokeCase per QA pair (cat 1-4)

Usage:
  curl -sL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o locomo10.json
  python benchmarks/locomo_to_harness.py --input locomo10.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

# LoCoMo category code -> harness category name. 5 (adversarial) is excluded:
# its "answer" is deliberately not supported by the cited turn, so retrieving
# that turn is not a clean recall target.
CATEGORY = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}


def parse_dt(raw: str) -> str:
    """Parse LoCoMo's '1:56 pm on 8 May, 2023' into an RFC3339 UTC string."""
    if raw:
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
            try:
                dt = datetime.strptime(raw.strip(), fmt)
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                continue
    # Fallback: a fixed epoch so the fixture is deterministic.
    return datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat()


def convert(data: list) -> tuple[list, list, dict]:
    corpus: list[dict] = []
    corpus_ids: set[str] = set()
    cases: list[dict] = []
    stats = {"turns": 0, "qa_total": 0, "qa_kept": 0, "qa_no_evidence": 0,
             "qa_unresolved": 0, "by_category": {}}

    for conv in data:
        sid = conv["sample_id"]
        c = conv["conversation"]
        # 1. Corpus: every dialogue turn becomes a memory.
        for key, val in c.items():
            if not (key.startswith("session_") and isinstance(val, list)):
                continue
            created = parse_dt(c.get(f"{key}_date_time", ""))
            for turn in val:
                dia = turn.get("dia_id")
                if not dia:
                    continue
                cid = f"{sid}:{dia}"
                if cid in corpus_ids:
                    continue
                corpus_ids.add(cid)
                corpus.append({
                    "id": cid,
                    "content": f"{turn.get('speaker', '')}: {turn.get('text', '')}".strip(),
                    "memory_type": "conversation",
                    "tags": [sid, turn.get("speaker", "")],
                    "created_at": created,
                })
                stats["turns"] += 1

        # 2. Cases: QA pairs whose evidence resolves to real turns.
        for i, qa in enumerate(conv["qa"]):
            stats["qa_total"] += 1
            cat = CATEGORY.get(qa.get("category"))
            if cat is None:  # adversarial / unknown
                continue
            ev = qa.get("evidence") or []
            if not ev:
                stats["qa_no_evidence"] += 1
                continue
            relevant = [
                {"corpus_item_id": f"{sid}:{e}", "grade": 3}
                for e in ev
                if isinstance(e, str) and f"{sid}:{e}" in corpus_ids
            ]
            if not relevant:
                stats["qa_unresolved"] += 1
                continue
            cases.append({
                "id": f"{sid}_q{i}",
                "category": cat,
                "query": qa["question"],
                "fixture_corpus_id": "locomo",
                "relevant": relevant,
            })
            stats["qa_kept"] += 1
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

    return corpus, cases, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="locomo10.json")
    ap.add_argument("--corpus-out", default="tests/recall/corpora/locomo.jsonl")
    ap.add_argument("--cases-out", default="tests/recall/locomo_cases.jsonl")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    corpus, cases, stats = convert(data)

    os.makedirs(os.path.dirname(args.corpus_out), exist_ok=True)
    with open(args.corpus_out, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(args.cases_out, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"corpus -> {args.corpus_out} ({len(corpus)} turns)")
    print(f"cases  -> {args.cases_out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
