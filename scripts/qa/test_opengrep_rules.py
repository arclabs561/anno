import json
import tempfile
import unittest
from pathlib import Path

from opengrep_rules import scan_counts


class ScanCountsTests(unittest.TestCase):
    def test_findings_are_advisory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps({"errors": [], "results": [{"check_id": "example"}],
                                        "paths": {"scanned": ["lib.rs"]}}))
            self.assertEqual(scan_counts(path), (1, 1))

    def test_incomplete_reports_are_rejected(self):
        reports = (
            {"errors": [{"type": "InvalidRuleSchemaError"}], "results": [], "paths": {"scanned": []}},
            {"errors": [{"type": "Parse error"}], "results": [], "paths": {"scanned": ["lib.rs"]}},
            {"errors": [], "results": [], "paths": {"scanned": []}},
            {"errors": [], "results": None, "paths": {"scanned": ["lib.rs"]}},
            [],  # historical CI placeholder for a missing report
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            for report in reports:
                with self.subTest(report=report):
                    path.write_text(json.dumps(report))
                    with self.assertRaises((KeyError, TypeError, ValueError)):
                        scan_counts(path)
