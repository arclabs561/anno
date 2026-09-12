"""Profiling contracts exercised with nextest's libtest-json-plus event shape."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analyze_test_profile import analyze_timing_file


class TestProfileAnalysis(unittest.TestCase):
    def test_ignored_events_do_not_change_measured_counts_or_module_buckets(self):
        events = [
            {"type": "suite", "event": "started", "nextest": {"test_binary": "anno"}},
            {
                "type": "test",
                "event": "ignored",
                "name": "anno::anno$other::tests::ignored",
            },
            {
                "type": "test",
                "event": "ok",
                "exec_time": 0.25,
                "name": "anno::anno$coalesce::resolver::tests::first",
            },
            {
                "type": "test",
                "event": "failed",
                "exec_time": 0.75,
                "name": "anno::anno$coalesce::resolver::tests::second",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "nextest.json"
            recording.write_text("\n".join(json.dumps(event) for event in events))
            analysis = analyze_timing_file(recording)

        self.assertEqual(analysis["summary"]["total_tests"], 2)
        self.assertEqual(analysis["summary"]["total_time_secs"], 1.0)
        self.assertEqual(analysis["summary"]["avg_time_secs"], 0.5)
        self.assertEqual(
            [(row["binary"], row["count"]) for row in analysis["by_binary"]],
            [("anno", 2)],
        )
        self.assertEqual(
            [(row["module"], row["count"]) for row in analysis["by_module"]],
            [("coalesce::resolver", 2)],
        )

    def test_cli_rejects_diagnostics_in_the_json_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "nextest.json"
            recording.write_text("Compiling anno\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("analyze_test_profile.py")),
                    str(recording),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Invalid JSON on line 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
