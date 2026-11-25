# SPDX-License-Identifier: MIT OR Apache-2.0
import hashlib
import json
from pathlib import Path
import sys
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

    @unittest.skipIf(sys.platform == "win32", "test executables use a shell shebang")
    def test_panel_preserves_sibling_lookup_and_refuses_replaced_build_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def tool(name, output):
                path = root / name
                path.write_text('#!/bin/sh\ncat "$0.value"\n')
                path.chmod(0o700)
                (root / f"{name}.value").write_text(output)
                return path

            current = tool("current", "fixed")
            historical = tool("historical", "old")
            selector = tool("picker", "span-roundtrip")
            schedule = root / "schedule.json"
            schedule.write_text(json.dumps([[]] * 12))
            dest = root / "report.json"
            argv = ["panel", "--anno", str(current), "--historical", str(historical),
                    "--schedule", str(schedule), panel.pilot.native.digest(schedule), "--out", str(dest)]
            old_causes = {"span-roundtrip": "brat-incomplete-input-overwrites-review",
                          "catalog-update": "catalog-incomplete-source-selection"}
            replace_role = None

            def observe(binary, name, mutate):
                nonlocal replace_role
                # Read a sibling through the supplied executable's original path.
                producer = panel.cli(root, binary).stdout.decode().strip()
                self.assertIn(producer, ("old", "fixed"))
                if replace_role is not None:
                    # Simulate an atomic Cargo replacement, with identical bytes.
                    replacement = root / "replacement"
                    replacement.write_bytes(replace_role.read_bytes())
                    replacement.chmod(0o700)
                    replacement.replace(replace_role)
                    replace_role = None
                failure = old_causes.get(name) if producer == "old" else None
                if mutate:
                    failure = panel.MUTATION_SIGNATURES[name]
                return {"failure": failure, "elapsed_ms": 0}

            with patch.object(sys, "argv", argv), patch.object(panel.pilot, "built_selector",
                    return_value=(selector, panel.pilot.native.digest(selector))), patch.object(
                    panel, "observe", observe):
                panel.main()
                self.assertIn("historical", json.loads(dest.read_bytes())["panels"])
                expected = b"previously reviewed receipt\n"
                dest.write_bytes(expected)
                for role in (current, historical, selector):
                    with self.subTest(role=role.name):
                        replace_role = role
                        with self.assertRaisesRegex(ValueError, "executable changed during panel"):
                            panel.main()
                        self.assertEqual(dest.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
