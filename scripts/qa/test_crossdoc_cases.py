"""Contract checks for source-addressed cross-document QA cases."""
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import crossdoc_cases as cases

SEEDS = Path(__file__).with_name("fixtures") / "crossdoc"


def seed(name="coverage-control"):
    return json.loads((SEEDS / (name + ".json")).read_text())


class CrossDocCaseTests(unittest.TestCase):
    def test_ambiguous_source_spans_require_labels(self):
        case = seed()
        case["assertions"][0]["mentions"][0].pop()
        with self.assertRaisesRegex(ValueError, "ambiguous source span"):
            cases.validate_case(case)
        case = seed()
        case["documents"][0]["signals"][1]["label"] = "PER"
        with self.assertRaisesRegex(ValueError, "duplicate source span"):
            cases.validate_case(case)

    def test_pair_check_cannot_alias_the_same_occurrence(self):
        case = seed()
        case["assertions"][0]["mentions"] = [["note", 0, 3], ["note", 0, 3, "PER"]]
        with self.assertRaisesRegex(ValueError, "two distinct"):
            cases.validate_case(case)

    def test_unknown_document_out_of_range_and_future_edition_are_invalid(self):
        for ref, after in [(["absent-doc", 0, 3], 2), (["note", 0, 99], 2), (["note", 0, 3], 1)]:
            case = seed()
            case["assertions"] = [{"after": after, "kind": "absent", "mentions": [ref]}]
            with self.subTest(ref=ref, after=after), self.assertRaises(ValueError):
                cases.validate_case(case)

    def test_validation_rejects_executable_fields_and_unbounded_work(self):
        case = seed()
        variants = [
            {**case, "command": ["sh", "-c", "false"]},
            {**case, "assertions": case["assertions"] * 52},
            {**case, "threshold": float("nan")},
            {**case, "documents": [{**case["documents"][0], "text": "x" * 4097}]},
        ]
        for data in variants:
            with self.assertRaises(ValueError):
                cases.validate_case(data)

    def test_absent_and_unresolved_cannot_satisfy_separation_or_record_checks(self):
        case = seed()
        case["assertions"] = [
            {"after": 2, "kind": "separate", "mentions": [["note", 0, 3], ["card", 0, 3, "LOC"]]},
            {"after": 2, "kind": "separate", "mentions": [["note", 0, 3], ["card", 11, 14]]},
            {"after": 2, "kind": "record", "record": None, "mentions": [["card", 0, 3, "LOC"]]},
            {"after": 2, "kind": "resolved", "mentions": [["note", 0, 3]]},
        ]
        report = {"metadata": {"documents_processed": 2, "total_entities": 3},
                  "clusters": [{"documents": ["note"], "mentions": [["note", 0]]}]}
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(report).encode(), stderr=b"")
        with patch.object(subprocess, "run", return_value=completed) as cli:
            checks = cases.run_case(case, Path(sys.executable))["checks"]
        self.assertEqual([check["passed"] for check in checks], [False, False, False, True])
        self.assertEqual([check["observed"][-1]["state"] for check in checks[:2]], ["unresolved", "absent"])
        cli.assert_called_once()
        self.assertEqual(cli.call_args.args[0][1:4], ["cross-doc", "--stdin", "--format"])
        self.assertEqual(cli.call_args.kwargs["env"]["ANNO_NO_DOWNLOADS"], "1")
        self.assertNotIn("shell", cli.call_args.kwargs)

    def test_malformed_success_report_is_an_execution_error(self):
        doc = cases.validate_case(seed())
        editions = cases._editions(doc, 2)
        row = {"documents": ["note"], "mentions": [["note", 0]]}
        report = {"metadata": {"documents_processed": 2, "total_entities": 3}, "clusters": [row]}
        for key, value in (("mentions", [["note", 10]]), ("mentions", [["missing", 0]]),
                           ("mentions", [["note", 0], ["note", 0]]), ("documents", ["card"])):
            bad = copy.deepcopy(report)
            bad["clusters"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                cases._membership(bad, editions)

    def test_input_revisions_resolve_the_anchor_in_the_effective_document(self):
        wires = cases.validate_case(seed("catalog-revision"))
        ref = ["matinee", 5, 12]
        self.assertEqual(cases._locate(ref, cases._editions(wires, 3)), (1, ("matinee", 0)))
        self.assertEqual(cases._locate(ref, cases._editions(wires, 4)), (4, None))


if __name__ == "__main__":
    unittest.main()
