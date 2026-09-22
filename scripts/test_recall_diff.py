"""Tests for the recall diff renderer's resolution reporting.

`resolution_note` exists to stop a reader mistaking a quantisation artefact for
a measurement, so its arithmetic is the claim and has to be checked. Runs standalone so it needs nothing installed, and is still collectable by
pytest if that is present:

    python scripts/test_recall_diff.py
    python -m pytest scripts/test_recall_diff.py

The renderer never changes pass/fail — the Rust comparator owns the exit code —
so nothing here asserts about gating. It asserts about honesty.
"""

import json
import tempfile
from pathlib import Path

from recall_diff import (
    QUANTIZED_METRICS,
    case_counts,
    classify_case_moves,
    load_per_case,
    render,
    render_case_moves,
    resolution_note,
)

CASES = {"multi_hop": 25, "open_domain": 10, "single_hop": 35, "temporal": 30}


class TestCaseCounts:
    def test_reads_the_counts_the_report_recorded(self):
        report = {
            "case_count": 100,
            "by_category": {c: {"case_count": n, "p@1": 0.3} for c, n in CASES.items()},
        }
        total, per_cat = case_counts(report)
        assert total == 100
        assert per_cat == CASES

    def test_survives_a_report_with_neither(self):
        # Pre-RH-12 reports predate these fields. A renderer that raises here
        # would take down the PR comment for every historical baseline.
        total, per_cat = case_counts({})
        assert total == 0
        assert per_cat == {}

    def test_ignores_non_dict_category_entries(self):
        total, per_cat = case_counts({"case_count": 10, "by_category": {"x": 3}})
        assert (total, per_cat) == (10, {})


class TestResolutionNote:
    def test_reports_a_one_case_drop_as_one_case(self):
        note = resolution_note(
            "p@1: baseline 0.3100, current 0.3000, allowed drop 0.0062", 100, CASES
        )
        assert "n=100" in note
        assert "one case = 0.0100" in note
        assert "1.0 case(s)" in note

    def test_uses_the_category_count_for_a_per_category_metric(self):
        # The load-bearing case: multi_hop has 25 cases, so ONE case is 0.0400 —
        # sixteen times the overall step. Reading this against n=100 would call
        # a single flip a four-case collapse.
        note = resolution_note(
            "p@1[multi_hop]: baseline 0.2800, current 0.2400, allowed drop 0.0056",
            100,
            CASES,
        )
        assert "n=25" in note
        assert "one case = 0.0400" in note
        assert "1.0 case(s)" in note

    def test_says_so_when_the_allowance_cannot_admit_one_case(self):
        note = resolution_note(
            "p@1: baseline 0.3100, current 0.3000, allowed drop 0.0062", 100, CASES
        )
        assert "BELOW one case" in note

    def test_stays_quiet_when_the_allowance_can_admit_a_case(self):
        # A suite large enough for its own tolerance gets no warning — the note
        # must not cry wolf on a gate that is working.
        note = resolution_note(
            "p@1: baseline 0.3100, current 0.3000, allowed drop 0.0500", 100, CASES
        )
        assert "1.0 case(s)" in note
        assert "BELOW one case" not in note

    def test_says_nothing_about_continuous_metrics(self):
        # These average continuous per-case values and can move arbitrarily
        # little, so "how many cases is this" is not a meaningful question and
        # answering it would invent a precision the metric does not have.
        for metric in ("ndcg@10", "recall@10", "mrr", "map"):
            assert metric not in QUANTIZED_METRICS
            note = resolution_note(
                f"{metric}: baseline 0.4111, current 0.4105, allowed drop 0.0082", 100, CASES
            )
            assert note == "", metric

    def test_returns_empty_on_an_unparseable_detail(self):
        # The Rust side owns this string. If it changes shape, the renderer must
        # degrade to printing the detail alone rather than crash the comment.
        assert resolution_note("something else entirely", 100, CASES) == ""

    def test_returns_empty_when_the_count_is_unknown(self):
        assert (
            resolution_note(
                "p@1: baseline 0.3100, current 0.3000, allowed drop 0.0062", 0, {}
            )
            == ""
        )
        assert (
            resolution_note(
                "p@1[nonesuch]: baseline 0.2800, current 0.2400, allowed drop 0.0056",
                100,
                CASES,
            )
            == ""
        )

    def test_a_multi_case_drop_is_not_reported_as_one(self):
        # Guards the arithmetic itself: three cases out of 100 is 0.03.
        note = resolution_note(
            "p@1: baseline 0.3100, current 0.2800, allowed drop 0.0062", 100, CASES
        )
        assert "3.0 case(s)" in note


def _record(case_id, recall, p1, ndcg, category="multi_hop", query="q", missed=()):
    return {
        "case_id": case_id,
        "category": category,
        "query": query,
        "recall_at_k": recall,
        "p_at_1": p1,
        "ndcg_at_k": ndcg,
        "missed": list(missed),
    }


class TestCaseMoves:
    """The per-case list is what makes a gate result reviewable.

    The fixtures are the real #509 shapes: `conv-42_q62` lost its gold from the
    top 10 while `conv-42_q46` and `q52` gained theirs. The aggregate netted
    that to one lost p@1 case, and with no case list nobody reviewing #509 saw
    which query paid for it.
    """

    BASE = {
        "conv-42_q62": _record("conv-42_q62", 0.5, 1.0, 0.613, query="How many letters has Joanna recieved?"),
        "conv-42_q46": _record("conv-42_q46", 0.0, 0.0, 0.0),
        "conv-42_q52": _record("conv-42_q52", 0.5, 1.0, 0.61),
        "conv-42_q20": _record("conv-42_q20", 1.0, 0.0, 0.631, category="temporal"),
        "conv-42_q1": _record("conv-42_q1", 1.0, 1.0, 1.0),
    }
    CUR = {
        "conv-42_q62": _record(
            "conv-42_q62",
            0.0,
            0.0,
            0.0,
            query="How many letters has Joanna recieved?",
            missed=("conv-42:D14:1", "conv-42:D18:5"),
        ),
        "conv-42_q46": _record("conv-42_q46", 0.5, 0.0, 0.333),
        "conv-42_q52": _record("conv-42_q52", 1.0, 1.0, 0.9),
        "conv-42_q20": _record("conv-42_q20", 1.0, 0.0, 0.5, category="temporal"),
        "conv-42_q1": _record("conv-42_q1", 1.0, 1.0, 1.0),
    }

    def test_each_moved_case_lands_in_exactly_one_bucket(self):
        b = classify_case_moves(self.BASE, self.CUR)
        ids = lambda k: [m["case_id"] for m in b[k]]
        assert ids("lost_reach") == ["conv-42_q62"]
        assert ids("gained_reach") == ["conv-42_q46", "conv-42_q52"]
        assert ids("rank_only") == ["conv-42_q20"]
        assert ids("lost_p1") == [] and ids("gained_p1") == []

    def test_an_unchanged_case_is_not_listed(self):
        b = classify_case_moves(self.BASE, self.CUR)
        listed = {m["case_id"] for moves in b.values() for m in moves}
        assert "conv-42_q1" not in listed

    def test_losing_reach_outranks_losing_p1(self):
        # q62 lost both its top-10 gold and its p@1. It must be reported as the
        # graver change, not filed under p@1 where it reads as a rank wobble.
        b = classify_case_moves(self.BASE, self.CUR)
        assert [m["case_id"] for m in b["lost_reach"]] == ["conv-42_q62"]

    def test_a_p1_flip_with_unchanged_reach_is_its_own_bucket(self):
        base = {"a": _record("a", 1.0, 1.0, 1.0)}
        cur = {"a": _record("a", 1.0, 0.0, 0.63)}
        b = classify_case_moves(base, cur)
        assert [m["case_id"] for m in b["lost_p1"]] == ["a"]
        b = classify_case_moves(cur, base)
        assert [m["case_id"] for m in b["gained_p1"]] == ["a"]

    def test_a_changed_suite_is_reported_not_silently_compared(self):
        b = classify_case_moves({"old": _record("old", 1, 1, 1)}, {"new": _record("new", 1, 1, 1)})
        assert [m["case_id"] for m in b["only_in_baseline"]] == ["old"]
        assert [m["case_id"] for m in b["only_in_current"]] == ["new"]
        text = "\n".join(render_case_moves(b))
        assert "regenerated" in text

    def test_rendering_names_the_case_its_query_and_what_it_missed(self):
        text = "\n".join(render_case_moves(classify_case_moves(self.BASE, self.CUR)))
        assert "**1** lost a gold document" in text
        assert "**2** gained one" in text
        assert "`conv-42_q62`" in text
        assert "How many letters has Joanna recieved?" in text
        assert "`conv-42:D18:5`" in text
        # Rank-only moves are collapsed, not dropped.
        assert "<details>" in text and "`conv-42_q20`" in text

    def test_identical_runs_say_so(self):
        text = "\n".join(render_case_moves(classify_case_moves(self.BASE, self.BASE)))
        assert "(0)" in text
        assert "No case changed its result" in text

    def test_loads_the_harness_per_case_file_shape(self):
        # `recall-eval --per-case-output` writes {"full": [record, ...]}.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "per-case.json"
            path.write_text(json.dumps({"full": list(self.CUR.values())}), encoding="utf-8")
            loaded = load_per_case(path)
        assert set(loaded) == set(self.CUR)
        assert loaded["conv-42_q62"]["recall_at_k"] == 0.0

    def test_render_without_a_per_case_baseline_says_so(self):
        report = {"layers": {"full": {m: 0.5 for m in ("ndcg@10", "recall@10", "mrr", "p@1")}}}
        text = render(report, report, 2.0, None)
        assert "No per-case baseline" in text
        text = render(report, report, 2.0, classify_case_moves(self.BASE, self.CUR))
        assert "Cases that moved vs baseline (4)" in text


if __name__ == "__main__":
    # A plain runner, because this repo ships no python test dependency and a
    # test that cannot be run is documentation.
    failed = 0
    for cls in (TestCaseCounts, TestResolutionNote, TestCaseMoves):
        instance = cls()
        for name in sorted(n for n in dir(cls) if n.startswith("test_")):
            try:
                getattr(instance, name)()
                print(f"  ok   {cls.__name__}.{name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL {cls.__name__}.{name}: {e}")
    print()
    print(f"{'FAILED' if failed else 'all passed'} ({failed} failure(s))")
    raise SystemExit(1 if failed else 0)
