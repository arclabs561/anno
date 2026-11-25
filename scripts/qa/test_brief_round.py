import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from brief_round import evidence


class RoundEvidenceTest(unittest.TestCase):
    def test_benchmark_handoff_uses_recorded_source_and_digest(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            digest = "sha256:" + "a" * 64
            row = dict(dataset="WikiGold", backend="heuristic", success=True)
            path.write_text(json.dumps({"summary": {}, "results": [
                dict(row, provenance={"dataset": {"data_source": "S3 cache", "version": digest,
                                                   "source_url": "PRIVATE_SENTINEL"}}),
                dict(row, provenance={"dataset": {"data_source": "local cache", "content_sha256": "b" * 64}}),
                dict(row, success=False, error="PRIVATE_SENTINEL"),
            ]}))
            script = Path(__file__).with_name("brief_round.py")
            command = [sys.executable, str(script), str(path), "Which input did the run use?"]
            brief = subprocess.check_output([*command, "--compact"], text=True)
            full = subprocess.check_output(command, text=True)
            self.assertIn(f"success=True, source=S3 cache, digest={digest}", brief)
            self.assertIn("source=local cache, digest=sha256:" + "b" * 64, brief)
            self.assertIn("success=False, source=unrecorded, digest=unrecorded", brief)
            self.assertNotIn("PRIVATE_SENTINEL", brief)
            self.assertLess(len(brief), len(full))
            row["dataset"] = "invalid\nidentifier"
            path.write_text(json.dumps({"summary": {}, "results": [row]}))
            with self.assertRaises(ValueError):
                evidence(path)

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

    def test_crossdoc_handoff_accepts_only_consistent_bounded_assertion_counts(self):
        digests = {k + "_sha256": "a" * 64 for k in ("source", "oracle", "producer")}
        receipt = dict(digests, kind="anno.qa.crossdoc-case.v1", runner="cli",
                       passed=2, failed=0, outcome="pass")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"

            def brief(**changes):
                path.write_text(json.dumps(dict(receipt, **changes)))
                return evidence(path)

            self.assertIn("2 assertions (2 passed, 0 failed)", brief())
            self.assertIn("eval case outcome fail", brief(runner="eval", failed=254, outcome="fail"))
            for bad in (dict(passed=True), dict(failed=0.0), dict(passed=-1), dict(passed=0),
                        dict(passed=257), dict(outcome="fail"), dict(failed=1),
                        dict(runner=["cli"]), dict(source_sha256="A" * 64),
                        dict(expected="PRIVATE_EXPECTATION")):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    brief(**bad)
            receipt.pop("oracle_sha256")
            with self.assertRaises(ValueError):
                brief()

    def test_direct_reviewed_replay_distinguishes_unmeasured_and_hides_case_details(self):
        digests = {k + "_sha256": "b" * 64 for k in ("source", "oracle", "producer")}
        receipt = dict(digests, corpus="typed_spans", case="PRIVATE_CASE",
                       outcome="fail", signature="failed_rules:2")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"

            def brief(row):
                path.write_text(json.dumps(row))
                return evidence(path)

            self.assertEqual(brief(receipt), "Reviewed replay evaluated 1 case. Outcome fail.")
            self.assertIn("Outcome pass", brief(dict(receipt, outcome="pass", signature="all_assertions_passed")))
            for bad in (dict(outcome="error"), dict(signature="PRIVATE"), dict(corpus=[]),
                        dict(producer_sha256="bad"), dict(case=3), dict(extra=True)):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    brief(dict(receipt, **bad))
            for outcome in ("unavailable", "error"):
                self.assertEqual(brief(dict(outcome=outcome, detail="PRIVATE_REASON")),
                                 f"Reviewed replay reported {outcome}. Evaluated 0 cases.")

    def test_reviewed_matrix_slot_keeps_selector_and_case_details_out_of_handoff(self):
        from replay_feedback import key

        case = dict(scope="synthetic", case="PRIVATE_CASE", corpus="typed_spans", family="PRIVATE_CASE",
                    source_sha256="a" * 64, oracle_sha256="b" * 64)
        plan = dict(slot=1, requested_strategy="reviewed-replay", seed=7, selected_case=case)
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            schedule, result = (directory / name for name in ("schedule.json", "result.json"))
            schedule.write_text(json.dumps(dict(plan, plan_only=True)))
            result.write_text("{}")
            self.assertNotIn("PRIVATE_CASE", evidence(directory))
            schedule.write_text(json.dumps(dict(plan, plan_only=False)))
            for outcome, replays in (("fail", 2), ("pass", 1), ("pass", 2)):
                event = dict(case, outcome=outcome, replays=replays)
                observation = key(event)
                result.write_text(json.dumps(dict(outcome_observation_id=observation)))
                (directory / "replay.json").write_text(json.dumps(dict(event, observation_id=observation)))
                if outcome == "pass" and replays == 2:
                    with self.assertRaisesRegex(ValueError, "attempt"):
                        evidence(directory)
                else:
                    brief = evidence(directory)
                    self.assertIn(f"Evaluated 1 reviewed case in {replays}", brief)
                    self.assertNotIn("PRIVATE_CASE", brief)

    def test_crossdoc_cli_initializes_trace_from_aggregate_handoff(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "PRIVATE_CASE_FILENAME.json"
            trace = Path(tmp) / "run.sqlite"
            receipt = dict(kind="anno.qa.crossdoc-case.v1", runner="eval", passed=1,
                           failed=2, outcome="fail", source_sha256="a" * 64,
                           oracle_sha256="b" * 64, producer_sha256="c" * 64)
            path.write_text(json.dumps(receipt))
            cli = [sys.executable, str(Path(__file__).with_name("brief_round.py")),
                   str(path), "Compare revisions", "--trace", str(trace)]
            run = subprocess.run(cli, capture_output=True, check=True)
            self.assertIn(b"3 assertions (1 passed, 2 failed)", run.stdout)
            self.assertNotIn(b"PRIVATE_CASE_FILENAME", run.stdout)
            self.assertNotIn(receipt["oracle_sha256"].encode(), run.stdout)
            with sqlite3.connect(trace) as db:
                prompt_digest, = db.execute("SELECT prompt_sha256 FROM spans WHERE kind='round'").fetchone()
            self.assertEqual(prompt_digest, hashlib.sha256(run.stdout).hexdigest())

    def test_unknown_directory_fails(self):
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "expected panel"):
            evidence(Path(tmp))
