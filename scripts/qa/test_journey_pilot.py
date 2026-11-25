import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import journey_pilot as pilot


class JourneyPilotTests(unittest.TestCase):
    def test_confirmed_receipts_require_matching_failures(self):
        bad = "a" * 64
        for outputs, expected, attempts in [(["pass"], "pass", 1), ([bad, bad], bad, 2),
                ([bad, "pass"], "unavailable", 2), ([bad, "b" * 64], "unavailable", 2),
                ([ValueError()], "unavailable", 1)]:
            with patch.object(pilot, "measure_case", side_effect=outputs) as run:
                result = pilot.confirmed(run)
            self.assertEqual((result["observation"], result["attempts"]), (expected, attempts))

    def test_restore_rejects_oversize_unknown_duplicate_or_foreign_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state"
            row = dict.fromkeys(pilot.POLICIES, ["span-roundtrip", "pass"])
            good = dict(schema=1, cohort="abc", rounds=1, last_run=10, history=[row])
            path.write_text(json.dumps(good))
            self.assertEqual(pilot.read_state(path, "abc"), good)
            for bad in [json.dumps({**good, "extra": 1}), json.dumps({**good, "cohort": "old"}),
                        json.dumps({**good, "schema": True}), '{"schema":1,"schema":1}',
                        json.dumps({**good, "history": [dict.fromkeys(pilot.JOURNEYS, "free text")]}),
                        json.dumps({**good, "rounds": 17}), " " * (pilot.MAX_BYTES + 1)]:
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    pilot.read_state(path, "abc")

    def test_prospective_picks_census_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = argparse.Namespace(state=directory / "state.json", output=directory / "first",
                    run_id=101, seed=12, anno=Path(__file__), selector=directory / "selector")
            adaptive_inputs = []
            def select(command, **kwargs):
                if command[0] == "git":
                    return "a" * 40
                adaptive_inputs.append(json.loads(kwargs["input"]))
                return "stale-revision"

            unavailable = True
            def measure(name, _binary):
                self.assertTrue((args.output / "plan.json").is_file())
                if unavailable and name == "span-roundtrip":
                    raise FileNotFoundError()
                return "a" * 64 if name == "catalog-update" else "pass"

            with patch.object(pilot, "cohort", return_value="abc"), patch.dict(pilot.os.environ, {}, clear=True), \
                    patch.object(pilot.subprocess, "check_output", side_effect=select), \
                    patch.object(pilot, "measure_case", side_effect=measure) as measure_case:
                self.assertEqual(pilot.pilot(args), 2)
                report = json.loads((args.output / "report.json").read_text())
                self.assertEqual((report["census_size"], report["budget_per_policy"], report["measured"]), (3, 1, 2))
                self.assertEqual(measure_case.call_count, 4)
                self.assertEqual(report["policy_results"]["balanced"]["outcome"], "unavailable")
                self.assertEqual(report["policy_results"]["muxer"]["outcome"], "pass")
                self.assertFalse(args.state.exists())
                unavailable = False
                args.output = directory / "recovered"
                self.assertEqual(pilot.pilot(args), 0)
                self.assertEqual(measure_case.call_count, 8)
                first = args.state.read_bytes()
                args.output = directory / "retry"
                self.assertEqual(pilot.pilot(args), 0)
                self.assertEqual(args.state.read_bytes(), first)
                self.assertEqual(measure_case.call_count, 8)
                args.output, args.run_id = directory / "next", 102
                self.assertEqual(pilot.pilot(args), 0)
                second = json.loads((args.output / "report.json").read_text())
                self.assertEqual(second["policy_results"]["balanced"], dict(
                    case="catalog-update", outcome="confirmed-failure", unseen_in_window=True))
                # Only the census saw the catalog failure in the first round for these
                # two policies. Neither earns a past observation from another policy.
                counts = {a["name"]: (a["measured"], a["failures"]) for a in adaptive_inputs[-1]}
                self.assertEqual(counts["catalog-update"], (0, 0))
                self.assertEqual(counts["stale-revision"], (1, 0))
                self.assertEqual(pilot.read_state(args.state, "abc")["history"][0]["muxer"],
                                 ["stale-revision", "pass"])
