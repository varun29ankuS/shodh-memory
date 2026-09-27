#!/usr/bin/env python3
"""Build the `directional` recall suite: can retrieval tell who did what to whom?

Every gold memory has a TWIN made of exactly the same words with the roles
swapped: "Nate invited Joanna to the housewarming." and "Joanna invited Nate to
the housewarming." A question such as "Who did Nate invite to the
housewarming?" has one gold (the first) and one decisive distractor (the twin).
No lexical signal separates them, so only direction can.

Categories:
  dir_object   "Who did A <verb> ...?" asks for the object; the gold says "A <verbed> B".
  dir_subject  "Who <verbed> B ...?" asks for the subject; the gold says "A <verbed> B".
  dir_passive  the memories are passive ("B was <verbed> ... by A") and the
               questions are the two active forms above.

Both twins of every pair are asked about in both directions, so a system that
always prefers one twin scores exactly 50%. Within a verb, a name appears in only
one pair, so every question has exactly one gold. Every name also appears in one
unrelated filler memory, so matching names alone does not reduce the candidates
to the twins.

Deterministic (fixed seed). Stdlib only.

Usage:
    python benchmarks/build_directional_suite.py
Writes tests/recall/corpora/directional.jsonl and tests/recall/directional_cases.jsonl.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 20260926
CORPUS_ID = "directional"
ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "tests/recall/corpora/directional.jsonl"
CASES_PATH = ROOT / "tests/recall/directional_cases.jsonl"

NAMES = [
    "Nate", "Joanna", "Caroline", "Melanie", "Priya", "Tomas", "Aisha", "Daniel",
    "Mei", "Oliver", "Fatima", "Lucas", "Sofia", "Kwame", "Hana", "Ethan",
    "Leila", "Marco", "Yuki", "Grace", "Omar", "Ingrid", "Rafael", "Chloe",
    "Arjun", "Nadia", "Felix", "Zara", "Diego", "Elena", "Samir", "Clara",
    "Jonas", "Amara", "Victor", "Isla", "Mateo", "Rosa", "Hugo", "Tara",
]

# (past, base, passive participle, complement). Questions put the complement
# after the verb, so "Who did A <base> <complement>?" and
# "Who <past> B <complement>?".
VERBS = [
    ("invited", "invite", "invited", "to the housewarming"),
    ("helped", "help", "helped", "with the move"),
    ("called", "call", "called", "on Sunday night"),
    ("visited", "visit", "visited", "in the hospital"),
    ("hired", "hire", "hired", "for the bakery"),
    ("forgave", "forgive", "forgiven", "after the argument"),
    ("taught", "teach", "taught", "to play guitar"),
    ("beat", "beat", "beaten", "at chess"),
    ("thanked", "thank", "thanked", "for the birthday gift"),
    ("interviewed", "interview", "interviewed", "for the podcast"),
]
ACTIVE_PAIRS_PER_VERB = 4
PASSIVE_PAIRS_PER_VERB = 2

FILLERS = [
    "{n} has been training for a half marathon since spring.",
    "{n} started learning to bake sourdough bread.",
    "{n} adopted a rescue dog named Biscuit.",
    "{n} is saving up for a trip to Portugal.",
    "{n} repainted the kitchen a pale green.",
    "{n} joined a pottery class on Thursday evenings.",
    "{n} has been reading a lot of science fiction lately.",
    "{n} switched to cycling to work.",
]


def active(a: str, past: str, b: str, comp: str) -> str:
    return f"{a} {past} {b} {comp}."


def passive(a: str, part: str, b: str, comp: str) -> str:
    # "B was <part> <comp> by A": A is the agent.
    return f"{b} was {part} {comp} by {a}."


def main() -> None:
    rng = random.Random(SEED)
    start = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    corpus: list[dict] = []
    cases: list[dict] = []

    def add_memory(content: str) -> str:
        mid = f"dir-m{len(corpus):03d}"
        corpus.append(
            {
                "id": mid,
                "content": content,
                "memory_type": "conversation",
                "tags": ["directional"],
                "created_at": (start + timedelta(hours=len(corpus))).isoformat(),
            }
        )
        return mid

    def add_case(category: str, query: str, gold: str) -> None:
        cases.append(
            {
                "id": f"dir-q{len(cases):03d}",
                "category": category,
                "query": query,
                "fixture_corpus_id": CORPUS_ID,
                "relevant": [{"corpus_item_id": gold, "grade": 3}],
            }
        )

    for past, base, part, comp in VERBS:
        n_pairs = ACTIVE_PAIRS_PER_VERB + PASSIVE_PAIRS_PER_VERB
        names = rng.sample(NAMES, 2 * n_pairs)
        for k in range(n_pairs):
            a, b = names[2 * k], names[2 * k + 1]
            is_passive = k >= ACTIVE_PAIRS_PER_VERB
            render = (lambda x, y: passive(x, part, y, comp)) if is_passive else (
                lambda x, y: active(x, past, y, comp)
            )
            # fwd: a acts on b. rev: b acts on a. Same words, roles swapped.
            fwd = add_memory(render(a, b))
            rev = add_memory(render(b, a))
            obj_cat = "dir_passive" if is_passive else "dir_object"
            subj_cat = "dir_passive" if is_passive else "dir_subject"
            for agent, patient, gold in ((a, b, fwd), (b, a, rev)):
                add_case(obj_cat, f"Who did {agent} {base} {comp}?", gold)
                add_case(subj_cat, f"Who {past} {patient} {comp}?", gold)

    for i, n in enumerate(NAMES):
        add_memory(FILLERS[i % len(FILLERS)].format(n=n))

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with CASES_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    print(f"{len(corpus)} memories, {len(cases)} cases {by_cat}")


if __name__ == "__main__":
    main()
