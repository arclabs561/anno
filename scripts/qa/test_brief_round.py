import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from brief_round import evidence


class RoundEvidenceTest(unittest.TestCase):
    def test_only_completed_measurements_present_scores(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan = {"slot": 2, "requested_strategy": "estimate", "seed": 9}
            (directory / "schedule.json").write_text(json.dumps(plan))
            self.assertIn("did not produce", evidence(directory))
            result = directory / "result.json"
            result.write_text(json.dumps(dict(plan, strategy_matches=False, plan_only=False)))
            self.assertIn("differed", evidence(directory))
            result.write_text(json.dumps(dict(plan, strategy_matches=True, plan_only=True)))
            self.assertIn("No backend was evaluated", evidence(directory))
            result.write_text(json.dumps(dict(plan, strategy_matches=True, plan_only=False)))
            (directory / "decisions.jsonl").write_text(json.dumps(dict(
                record_type="outcome", dataset="WikiGold", backend="heuristic", ok=True, primary_f1=0.0)))
            self.assertIn("ok=True, primary F1=0.0", evidence(directory))

    def test_unknown_directory_fails(self):
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "expected panel"):
            evidence(Path(tmp))
