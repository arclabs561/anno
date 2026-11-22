import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from check_matrix_outcomes import RANDOM_TASKS, check, check_wikigold


def outcome(task, strategy="random", hard=False, examples=1):
    return {
        "record_type": "outcome",
        "strategy": strategy,
        "slice": f"{task}.lang=en",
        "hard_junk": hard,
        "cost_units": examples,
        "primary_f1": 0,
        "ok": False,  # A valid zero score counts as an execution.
    }


class CoverageTest(unittest.TestCase):
    def accepted(self, rows):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return check(map(json.dumps, rows))

    def test_zero_scores_and_a_failed_worst_first_attempt(self):
        rows = [outcome(task) for task in RANDOM_TASKS]
        rows.append(outcome("ner", "worst-first", hard=True, examples=0))
        self.assertTrue(self.accepted(rows))

    def test_repeated_ner_cannot_cover_other_tasks(self):
        rows = [outcome("ner")] * 9 + [outcome("ner", "worst-first")] * 2
        self.assertFalse(self.accepted(rows))

    def test_hard_failures_and_empty_samples_do_not_cover_a_task(self):
        rows = [outcome(task) for task in RANDOM_TASKS if task != "re"]
        rows.extend([outcome("re", hard=True), outcome("re", examples=0)])
        rows.append(outcome("re", "worst-first"))
        self.assertFalse(self.accepted(rows))

    def test_policy_decision_alone_does_not_count_as_an_outcome(self):
        rows = [outcome(task) for task in RANDOM_TASKS]
        rows.append({"strategy": "worst-first", "chosen": "tplinker"})
        self.assertFalse(self.accepted(rows))

    def test_invalid_log_identifies_its_line(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            check([json.dumps(outcome("re")), "{"])

    def test_empty_cache_is_visible_in_exploratory_report(self):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertTrue(check([], report=True))
        self.assertIn("temporal (no outcome)", output.getvalue())

    def test_smoke_requires_exact_fixture_result(self):
        row = dict(outcome("ner"), backend="heuristic", dataset="WikiGold", primary_f1=1.0)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertTrue(check_wikigold([json.dumps(row)]))
            self.assertFalse(check_wikigold([]))
            for mutation in ({"primary_f1": 0.0}, {"backend": "crf"}, {"cost_units": 0}):
                self.assertFalse(check_wikigold([json.dumps(row | mutation)]))

    def test_reports_first_decision_and_matching_observation(self):
        decision = {"strategy": "worst-first", "chosen": "heuristic", "slice": "ner",
                    "run_id": "run", "datasets": ["WikiGold"], "explore_first": False,
                    "top_candidates": {"rows": [{"arm": "heuristic", "calls": 2}]}}
        decision["worst_first_round"] = {"retry": {"observation_id": "prior", "reason": "recurrent_after_recovery", "risk": 0.25, "observations": 4}}
        wrong = dict(outcome("ner", "worst-first"), backend="crf", dataset="WikiGold", run_id="run")
        matching = dict(wrong, backend="heuristic", hard_junk=True, fail_kind="io")
        for rows, result in (([decision, wrong], "no outcome"), ([decision, wrong, matching], "hard failure")):
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                check(map(json.dumps, rows), report=True)
            if matching in rows:
                self.assertIn("Hard failures by recorded kind: io 1", output.getvalue())
            self.assertIn("Retry evidence: prior (risk weight 0.250 over 4 observations)", output.getvalue())
            self.assertIn(f"heuristic (repeated failures after recovery, prior calls 2) — {result}", output.getvalue())
