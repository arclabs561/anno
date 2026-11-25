import json
import tempfile
import unittest
from pathlib import Path

from opengrep_rules import check_fixture_ids, check_target_coverage, scan_counts


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


class ProductionCoverageTests(unittest.TestCase):
    def test_each_configured_rule_has_a_target(self):
        status = "  Scanning 3 files tracked by git with 2 Code rules:\n"
        self.assertEqual(check_target_coverage(status + "Ran 2 rules on 3 files: 0 findings.", 3), 2)
        with self.assertRaisesRegex(ValueError, "applied 1/2"):
            check_target_coverage(status + "Ran 1 rule on 3 files: 0 findings.", 3)
        with self.assertRaisesRegex(ValueError, "did not report"):
            check_target_coverage(status, 3)


class FixtureCoverageTests(unittest.TestCase):
    def test_another_passing_rule_cannot_hide_a_missing_case(self):
        checks = {"present": {"passed": True, "matches": {"graph.rs": {"expected_lines": [4]}}}}
        check_fixture_ids("config.yaml", {"present"}, checks)
        with self.assertRaisesRegex(ValueError, "omitted"):
            check_fixture_ids("config.yaml", {"present", "omitted"}, checks)

    def test_an_empty_positive_case_is_rejected(self):
        checks = {"present": {"passed": True, "matches": {"graph.rs": {"expected_lines": []}}}}
        with self.assertRaisesRegex(ValueError, "no passing positive case"):
            check_fixture_ids("config.yaml", {"present"}, checks)
