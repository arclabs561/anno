# SPDX-License-Identifier: MIT OR Apache-2.0
import hashlib
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import journey_policy_panel as panel


class PanelTests(unittest.TestCase):
    def test_policy_selects_before_reveal_and_only_receives_selected_results(self):
        events = []
        arms = panel.pilot.JOURNEYS

        def pick(state, seed, selector):
            events.append("pick")
            if state["rounds"]:
                previous, = state["history"]
                self.assertEqual(previous["muxer"], [arms[2], "pass"])
                self.assertTrue(panel.pilot.failure(previous["balanced"][1]))
            return dict(zip(panel.pilot.POLICIES, arms, strict=True))

        def reveal():
            events.append("reveal")
            return {name: {"failure": "same" if name == arms[0] else None, "elapsed_ms": 2}
                    for name in arms}

        with patch.object(panel.pilot, "picks", pick):
            scores = panel.summarize(panel.simulate(None, [reveal, reveal], 3))
        self.assertEqual(events, ["pick", "reveal", "pick", "reveal"])
        self.assertEqual(scores["balanced"]["distinct_failures"], 1)
        self.assertEqual(scores["balanced"]["hits"], 2)
        self.assertEqual(scores["muxer"]["hits"], 0)
        self.assertAlmostEqual(scores["uniform_expectation"]["distinct_failures"], 5 / 9)
        self.assertAlmostEqual(scores["uniform_expectation"]["hits"], 2 / 3)

    def test_faulted_round_requires_the_scheduled_signature(self):
        for observed in (None, "another-assertion"):
            with self.subTest(observed=observed), patch.object(
                    panel, "observe", return_value={"failure": observed}):
                with self.assertRaisesRegex(ValueError, "expected 'brat-output-segments'"):
                    panel.mutation_census(None, ["span-roundtrip"])

    def test_clean_round_rejects_unexpected_failure(self):
        with patch.object(panel, "observe", return_value={"failure": "unexpected"}):
            with self.assertRaisesRegex(ValueError, "expected None, observed 'unexpected'"):
                panel.mutation_census(None, [])

    def test_stale_probe_needs_an_advanced_head_and_a_specific_rejection(self):
        def result(value):
            return subprocess.CompletedProcess([], 0, json.dumps(value).encode(), b"")

        before = {"head": "old", "documents": [{"id": "first", "text": "Reed"}]}
        after = {**before, "head": "second"}
        stale = subprocess.CompletedProcess([], 1, b"", b"corpus head changed or belongs to a different journal")
        unrelated = subprocess.CompletedProcess([], 1, b"", b"error reading journal header")
        for refreshed, rejection, diagnostic in [
                (before, stale, "did not advance"),
                ({"head": "second", "documents": []}, stale, "reviewed documents"),
                (after, unrelated, "unexpected reason")]:
            for mutate in (False, True):
                with self.subTest(diagnostic=diagnostic, mutate=mutate), tempfile.TemporaryDirectory() as temp:
                    deliveries = []

                    def corpus(_dir, _bin, *args):
                        if "apply" in args:
                            deliveries.append(json.loads((Path(temp) / "doc").read_text())["id"])
                            return result({"receipt": {"document_id": deliveries[-1]}})
                        if len(deliveries) == 1:
                            return result(before)
                        return rejection if "--at-head" in args else result(refreshed)

                    with patch.object(panel, "cli", side_effect=corpus):
                        with self.assertRaisesRegex(ValueError, diagnostic):
                            panel.stale(Path(temp), "anno", mutate)

    def test_schedule_checks_commitment_and_arm_set(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "panel.json"
            path.write_text(json.dumps([[]] * 12))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(panel.sealed_schedule(path, digest), [[]] * 12)
            path.write_text(json.dumps([["bad-arm"]] * 12))
            with self.assertRaisesRegex(ValueError, "digest"):
                panel.sealed_schedule(path, digest)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "arm"):
                panel.sealed_schedule(path, digest)


if __name__ == "__main__":
    unittest.main()
