import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from check_matrix_outcomes import RANDOM_TASKS, check, check_smoke


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
            self.assertTrue(check_smoke([json.dumps(row)]))
            self.assertFalse(check_smoke([]))
            for mutation in ({"primary_f1": 0.0}, {"backend": "crf"}, {"cost_units": 0}):
                self.assertFalse(check_smoke([json.dumps(row | mutation)]))
