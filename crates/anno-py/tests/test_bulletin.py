"""Replay the README with the distribution installed in the active interpreter."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTICES = [EXAMPLES / "bulletins" / f"bulletin-edition-{n}.txt" for n in (1, 2)]


class BulletinTest(unittest.TestCase):
    def run_example(self, before, after):
        with tempfile.TemporaryDirectory() as workdir:
            output = subprocess.check_output(
                [sys.executable, "-I", EXAMPLES / "bulletin_diff.py", before, after],
                cwd=workdir,
                env={**os.environ, "ANNO_NO_DOWNLOADS": "1"},
            )
        report = json.loads(output)
        self.assertEqual(report["sources"], [
            {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (before, after)
        ])
        return report["date_count_changes"]

    def test_changed_deadline(self):
        changes = self.run_example(*NOTICES)
        self.assertEqual([c["date_pieces"] for c in changes], [["2026-12-04"], ["2026-12-07"]])
        self.assertEqual(changes[0]["after"], [])
        self.assertEqual(changes[1]["before"], [])
        surface = os.environ.get("ANNO_BULLETIN_SURFACE", "checkout")
        detector = {"checkout": "pattern", "pypi-0.1.0": None}[surface]
        locations = zip(NOTICES, changes, ("before", "after"), ([98, 108], [110, 120]))
        for path, change, side, span in locations:
            source = path.read_bytes().decode("utf-8")
            self.assertEqual(source[slice(*span)], change["date_pieces"][0])
            self.assertEqual(change[side], [{
                "line": 2, "source_line": source.splitlines()[1],
                "segments": [span], "detector": detector,
            }])
            self.assertEqual(
                source.splitlines()[2], "The public drawing is scheduled for 2026-12-09."
            )

    def test_identical_editions(self):
        self.assertEqual(self.run_example(NOTICES[0], NOTICES[0]), [])


if __name__ == "__main__":
    unittest.main()
