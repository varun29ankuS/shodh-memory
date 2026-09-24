"""Tests for the stage-export differ.

Runs standalone, like test_recall_diff.py:

    python scripts/test_stage_diff.py
"""

import json
import tempfile
from pathlib import Path

from stage_diff import compare, load, render, span_set


def export(ner_conf="3e9d9943", emb="aa", edge_strength="3f000000", leg=None, ce=None):
    leg = leg or [["m1", "3f800000"], ["m2", "3f000000"]]
    ce = ce or [["m1", "c0000000"], ["m2", "c0100000"]]
    return [
        {"kind": "ingest", "id": "m1", "embedding_sha256": emb, "ner_sha256": ner_conf,
         "ner": [f"Nate|PER|Some(\"diplomat\")|{ner_conf}|12|16"]},
        {"kind": "graph", "nodes": ["Nate|[\"PER\"]|3"],
         "edges": [f"Nate|Joanna|RelatedTo|{edge_strength}"]},
        {"kind": "case", "case_id": "q1", "pool": ["m1", "m2"], "deep": ["m1", "m2"],
         "ranks": ["m1", "m2"], "final_captured": ["m1", "m2"],
         "legs": [{"stage": "vector", "items": leg}], "ce": [{"stage": "ce", "items": ce}]},
    ]


def roundtrip(records):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return load(p)


class TestStageDiff:
    def test_identical_exports_report_nothing(self):
        r = compare(roundtrip(export()), roundtrip(export()))
        assert r["embedding_differs"] == [] and r["ner_scores_differ"] == []
        assert r["order"] == {} and r["scores"] == {}
        assert "every stage identical" in render(r)

    def test_score_drift_is_not_a_span_change(self):
        # The cross-runner pattern: NER confidence bits move, the spans found do not.
        r = compare(roundtrip(export(ner_conf="3e9d9943")), roundtrip(export(ner_conf="3e9d9946")))
        assert r["ner_scores_differ"] == ["m1"]
        assert r["ner_spans_differ"] == []

    def test_edge_strength_drift_is_not_a_membership_change(self):
        r = compare(roundtrip(export(edge_strength="3f000000")),
                    roundtrip(export(edge_strength="3f000001")))
        assert r["edges"] == ([], [])

    def test_order_and_score_changes_are_told_apart(self):
        swapped = [["m2", "3f000000"], ["m1", "3f800000"]]
        r = compare(roundtrip(export()), roundtrip(export(leg=swapped)))
        assert r["order"] == {"leg:vector": ["q1"]}
        drift = [["m1", "c0000000"], ["m2", "c0100001"]]
        r = compare(roundtrip(export()), roundtrip(export(ce=drift)))
        assert r["scores"] == {"ce": ["q1"]} and r["order"] == {}

    def test_span_set_drops_only_the_confidence(self):
        assert span_set(["a|PER|None|3e9d9943|0|1"]) == {"a|PER|None|0|1"}


if __name__ == "__main__":
    failed = 0
    for name in sorted(n for n in dir(TestStageDiff) if n.startswith("test_")):
        try:
            getattr(TestStageDiff(), name)()
            print(f"  ok   {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{'FAILED' if failed else 'all passed'} ({failed} failure(s))")
    raise SystemExit(1 if failed else 0)
