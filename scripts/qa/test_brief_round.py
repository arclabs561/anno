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

    def test_real_measurements_select_different_challenges_without_forwarding_source_text(self):
        base = dict(record_type="outcome", dataset="Synthetic", backend="control", ok=False,
                    primary_f1=0.25, cost_units=2, source_text="PRIVATE_SENTINEL")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"

            def brief(counts):
                path.write_text(json.dumps(dict(base, structural_counts=counts)))
                output = evidence(path)
                self.assertNotIn("PRIVATE_SENTINEL", output)
                return output

            coref = dict(gold_coref_links=100, matched_coref_links=3, predicted_coref_links=8)
            self.assertIn("mention-proposal comparison", brief(dict(coref, reachable_gold_links=10)))
            self.assertIn("At least 57 reachable links", brief(dict(coref, reachable_gold_links=60)))
            self.assertIn("input coverage was unobserved", brief(coref))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                brief(dict(coref, reachable_gold_links=101))
            temporal = dict(unmatched_predictions_overlapping_missed_gold=3, matched_spans=1,
                            predicted_spans=5, gold_empty_sentences=0, gold_empty_sentences_with_predictions=0)
            self.assertIn("3/4 unmatched predictions", brief(temporal))
            self.assertIn("Reserve a labeled negative panel", brief(temporal))
            self.assertNotIn("gold-empty", brief({}))

    def test_unknown_directory_fails(self):
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "expected panel"):
            evidence(Path(tmp))
