#!/usr/bin/env python3
"""Convert LongMemEval-S into per-question recall-harness fixtures.

LongMemEval (xiaowu0162/LongMemEval, ICLR 2025) is the current SOTA long-term
memory benchmark: 500 questions, each with its OWN haystack of ~48 chat
sessions (~115K tokens). This differs structurally from LoCoMo, where 10
dialogues are shared across all questions — here every question carries a
private haystack, so the harness must ingest one haystack, run one query, and
score, then move to the next question (a loop of mini-evals, not one shared
corpus).

Gold evidence is annotated two ways in the source:
  - turn-level: turns that support the answer carry `"has_answer": true`
  - session-level: `answer_session_ids` lists the evidence sessions
We emit the TURN-LEVEL target — the strictest, cleanest recall@k signal, with
no LLM judge — matching how recall-eval scores the smoke and LoCoMo suites.

Source schema (longmemeval_s.json, one object per question):
  question_id, question, answer, question_type, question_date,
  haystack_session_ids: [sid, ...],
  haystack_dates:       [iso, ...],
  haystack_sessions:    [[{role, content, has_answer?}, ...], ...],
  answer_session_ids:   [sid, ...]

Output (recall-harness fixtures, under tests/recall/longmemeval/):
  manifest.jsonl          — one line per question:
      {id, question, category, gold_ids:[turn_id,...], corpus: "corpora/<qid>.jsonl"}
  corpora/<question_id>.jsonl — one CorpusItem per user/assistant turn:
      {id, content, memory_type, tags, created_at}

A turn id is `<question_id>::<session_id>::t<turn_index>` so it is globally
unique and traceable back to the source.

Usage:
  pip install huggingface_hub
  python -c "from huggingface_hub import hf_hub_download as d; \
      print(d(repo_id='xiaowu0162/longmemeval-cleaned', \
      filename='longmemeval_s_cleaned.json', repo_type='dataset'))"
  python benchmarks/longmemeval_to_harness.py --input <path-to-json> --limit 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

# LongMemEval question_type -> harness category. The harness category is used
# only for per-category recall breakdowns; unknown types fall back to "other".
CATEGORY = {
    "single-session-user": "single_hop",
    "single-session-assistant": "single_hop",
    "single-session-preference": "single_hop",
    "multi-session": "multi_hop",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge_update",
}


def iso(raw: str) -> str:
    """Normalize a haystack date to RFC3339 UTC; fall back to a fixed epoch."""
    if raw:
        for fmt in (
            "%Y/%m/%d (%a) %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return (
                    datetime.strptime(raw.strip(), fmt)
                    .replace(tzinfo=timezone.utc)
                    .isoformat()
                )
            except ValueError:
                continue
        # Some releases already ship RFC3339.
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
                timezone.utc
            ).isoformat()
        except ValueError:
            pass
    return datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat()


def convert(data: list, out_dir: str, limit: int | None) -> dict:
    corpora_dir = os.path.join(out_dir, "corpora")
    os.makedirs(corpora_dir, exist_ok=True)
    manifest: list[dict] = []
    stats = {"questions": 0, "turns": 0, "gold_turns": 0, "no_gold": 0}

    for q in data:
        if limit is not None and stats["questions"] >= limit:
            break
        qid = q["question_id"]
        sessions = q.get("haystack_sessions", [])
        sids = q.get("haystack_session_ids", [])
        dates = q.get("haystack_dates", [])

        corpus: list[dict] = []
        gold_ids: list[str] = []
        for si, session in enumerate(sessions):
            sid = sids[si] if si < len(sids) else f"s{si}"
            created = iso(dates[si] if si < len(dates) else "")
            for ti, turn in enumerate(session):
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                tid = f"{qid}::{sid}::t{ti}"
                corpus.append(
                    {
                        "id": tid,
                        "content": content,
                        "memory_type": "conversation",
                        "tags": [sid, turn.get("role", "user")],
                        "created_at": created,
                    }
                )
                if turn.get("has_answer") is True:
                    gold_ids.append(tid)

        if not corpus:
            continue
        if not gold_ids:
            # No turn-level gold (some releases only annotate at session level).
            # Skip rather than emit an unscorable case — keeps recall honest.
            stats["no_gold"] += 1
            continue

        corpus_rel = os.path.join("corpora", f"{qid}.jsonl")
        with open(os.path.join(out_dir, corpus_rel), "w", encoding="utf-8") as f:
            for item in corpus:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        manifest.append(
            {
                "id": qid,
                "question": q.get("question", ""),
                "category": CATEGORY.get(q.get("question_type", ""), "other"),
                "gold_ids": gold_ids,
                "corpus": corpus_rel.replace(os.sep, "/"),
            }
        )
        stats["questions"] += 1
        stats["turns"] += len(corpus)
        stats["gold_turns"] += len(gold_ids)

    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to longmemeval_s(.cleaned).json")
    ap.add_argument(
        "--out",
        default="tests/recall/longmemeval",
        help="output fixture directory",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only convert the first N scorable questions (cheap pilot)",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="deterministically reorder so any --limit prefix is a representative "
        "category mix (the source is type-ordered)",
    )
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # some releases wrap in {"questions": [...]}
        data = data.get("questions", list(data.values()))

    if args.shuffle:
        # The source is type-ordered (all single-session questions first), so a
        # naive --limit prefix would be all easy single-hop. Order by a stable
        # hash of question_id: deterministic and reproducible, but any prefix is
        # a representative mix of categories.
        data.sort(key=lambda q: hashlib.md5(q["question_id"].encode()).hexdigest())

    stats = convert(data, args.out, args.limit)
    print(
        f"LongMemEval -> harness: {stats['questions']} questions, "
        f"{stats['turns']} turns, {stats['gold_turns']} gold turns, "
        f"{stats['no_gold']} skipped (no turn-level gold). Out: {args.out}"
    )


if __name__ == "__main__":
    main()
