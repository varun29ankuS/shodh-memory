"""Tests for the recall diff renderer's resolution reporting.

`resolution_note` exists to stop a reader mistaking a quantisation artefact for
a measurement, so its arithmetic is the claim and has to be checked. Runs standalone so it needs nothing installed, and is still collectable by
pytest if that is present:

    python scripts/test_recall_diff.py
    python -m pytest scripts/test_recall_diff.py

The renderer never changes pass/fail — the Rust comparator owns the exit code —
so nothing here asserts about gating. It asserts about honesty.
"""

from recall_diff import QUANTIZED_METRICS, case_counts, resolution_note

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


if __name__ == "__main__":
    # A plain runner, because this repo ships no python test dependency and a
    # test that cannot be run is documentation.
    failed = 0
    for cls in (TestCaseCounts, TestResolutionNote):
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
