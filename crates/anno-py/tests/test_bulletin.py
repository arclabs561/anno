# SPDX-License-Identifier: MIT OR Apache-2.0
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
                timeout=30,
                env={**os.environ, "ANNO_NO_DOWNLOADS": "1"},
            )
        report = json.loads(output)
        self.assertEqual(report["sources"], [
            {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (before, after)
        ])
        return report["date_count_changes"]

    def check_deadline(self, paths, dates, spans):
        changes = self.run_example(*paths)
        self.assertEqual([c["date_pieces"] for c in changes], [[date] for date in dates])
        self.assertEqual(changes[0]["after"], [])
        self.assertEqual(changes[1]["before"], [])
        surface = os.environ.get("ANNO_BULLETIN_SURFACE", "checkout")
        detector = {"checkout": "pattern", "pypi-0.1.0": None}[surface]
        for path, change, side, span in zip(paths, changes, ("before", "after"), spans):
            source = path.read_bytes().decode("utf-8")
            self.assertEqual(source[slice(*span)], change["date_pieces"][0])
            line = source.count("\n", 0, span[0]) + 1
            self.assertEqual(change[side], [{
                "line": line, "source_line": source.splitlines()[line - 1],
                "segments": [span], "detector": detector,
            }])

    def test_changed_deadline(self):
        self.check_deadline(NOTICES, ["2026-12-04", "2026-12-07"], [[98, 108], [110, 120]])
        for path in NOTICES:
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines()[2],
                "The public drawing is scheduled for 2026-12-09.",
            )

    def test_caller_replaces_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, amended = [Path(temporary) / name for name in ("first.txt", "amended.txt")]
            first.write_bytes("🛶 Édition\nDock closes 2027-02-08.\n".encode("utf-8"))
            amended.write_bytes("🛶 Édition\nRévision\nPier closes 2027-02-11.\n".encode("utf-8"))
            self.check_deadline(
                [first, amended], ["2027-02-08", "2027-02-11"], [[22, 32], [31, 41]]
            )

    def test_identical_editions(self):
        self.assertEqual(self.run_example(NOTICES[0], NOTICES[0]), [])


if __name__ == "__main__":
    unittest.main()
