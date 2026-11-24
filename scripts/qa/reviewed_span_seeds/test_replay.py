"""Offline trust-boundary checks; real historical traces stay local."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from replay import check_round, digest, verified


class ReviewedSeedAdmission(unittest.TestCase):
    def test_agent_verdict_cannot_admit_an_unreviewed_or_reserved_fixture(self):
        with TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "input.json"
            fixture.write_text(json.dumps({"judge": "confirmed", "scope": "exposed-regression",
                                           "origin": "project-authored-synthetic"}))
            with self.assertRaisesRegex(ValueError, "outside the reviewed"):
                check_round(fixture, Path(tmp))

    def test_changed_evidence_is_rejected(self):
        with TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.json"
            result.write_bytes(b"changed outcome")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verified(result, digest(b"independently frozen outcome"))


if __name__ == "__main__":
    unittest.main()
