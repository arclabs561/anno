"""Reader-oracle validation and native producer mutation controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import brief_round
import crossdoc_query_checks as query
import replay_crossdoc_case as replay

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures/crossdoc"
CASE = FIXTURES / "lamp-ledger.json"
HOLDOUT = FIXTURES / "lamp-ledger.queries.json"
NATIVE = os.environ.get("ANNO_BIN")


class QueryHoldoutTest(unittest.TestCase):
    def test_binding_ignores_authored_assertions_and_checks_source_revisions(self):
        case = replay.load_case(CASE)
        holdout, _ = replay.read_json(HOLDOUT)
        def validate():
            return query.validate(holdout, case, replay.fingerprints(case)["source_sha256"])
        validate()
        case["assertions"].pop()
        validate()
        case["documents"][0]["text"] += " Correction."
        with self.assertRaisesRegex(ValueError, "frozen source"):
            validate()

    def test_rejects_unobserved_or_duplicate_expected_postings_at_checkpoint(self):
        case = replay.load_case(CASE)
        holdout, _ = replay.read_json(HOLDOUT)
        for expected in ([["receipt", 0]], [["lamp", 1]], [["lamp", 0], ["lamp", 0]], [["lamp", True]]):
            holdout["queries"] = [{"after": 1, "entity": "Nacre", "postings": expected}]
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                query.validate(holdout, case, holdout["source_sha256"])

    def test_query_counts_extra_duplicates(self):
        rule = {"after": 1, "entity": "Nacre", "postings": [["lamp", 0]]}
        native = subprocess.CompletedProcess([], 0, stdout=b'[{"mentions":[["lamp",0],["lamp",0]]}]')
        with patch.object(subprocess, "run", return_value=native):
            [measured] = query.run([rule], {1: Path("report.json")}, Path("anno"))
        self.assertFalse(measured["passed"])
        self.assertEqual(measured["actual"], [["lamp", 0], ["lamp", 0]])

    def test_cli_source_digest_does_not_execute_a_producer(self):
        output = subprocess.check_output([sys.executable, HERE/"replay_crossdoc_case.py", CASE, "--source-digest"])
        self.assertEqual(output.decode().strip(), json.loads(HOLDOUT.read_text())["source_sha256"])


@unittest.skipUnless(NATIVE, "set ANNO_BIN to the built anno CLI for native mutation and reader checks")
class NativeQueryHoldoutTest(unittest.TestCase):
    def replay_case(self, producer, holdout, expected_status):
        p = subprocess.run([sys.executable, HERE/"replay_crossdoc_case.py", CASE, producer,
                            "--query-holdout", holdout, "--details"], capture_output=True)
        self.assertEqual(p.returncode, expected_status, p.stderr.decode())
        receipt = json.loads(p.stdout)
        self.assertEqual(receipt["proposal_sha256"], hashlib.sha256(CASE.read_bytes()).hexdigest())
        self.assertEqual(receipt["consumer"]["holdout_sha256"], hashlib.sha256(holdout.read_bytes()).hexdigest())
        self.assertIn("reader queries", brief_round.query_case_evidence(receipt))
        return receipt, json.loads(p.stderr)

    def test_target_mutations_distinguish_two_contracts(self):
        native = Path(NATIVE).resolve(strict=True)
        with tempfile.TemporaryDirectory() as d:
            for field, counts, actual in (
                (None, (4, 2), [["lamp", 0], ["receipt", 0]]),
                ("kb_name", (2, 2), [["lamp", 0], ["receipt", 0]]),
                ("canonical_name", (4, 0), [["lift", 0]]),
            ):
                producer = native
                if field:
                    producer = Path(d) / "producer"
                    # Only modify the report. The query command always runs the native reader.
                    producer.write_text(f"""#!/usr/bin/env python3
import json, os, subprocess, sys
if sys.argv[1] != "cross-doc":
    os.execv({str(native)!r}, [{str(native)!r}, *sys.argv[1:]])
report = json.loads(subprocess.check_output([{str(native)!r}, *sys.argv[1:]], stdin=sys.stdin.buffer))
left, right = report["clusters"]
left[{field!r}], right[{field!r}] = right[{field!r}], left[{field!r}]
print(json.dumps(report))
""")
                    producer.chmod(0o700)
                with self.subTest(field=field):
                    receipt, details = self.replay_case(producer, HOLDOUT, int(bool(field)))
                    self.assertEqual((receipt["authored"]["passed"], receipt["consumer"]["passed"]), counts)
                    self.assertEqual(details["consumer"][0]["actual"], actual)
                    altered = copy.deepcopy(receipt)
                    altered["outcome"] = "fail" if receipt["outcome"] == "pass" else "pass"
                    with self.assertRaisesRegex(ValueError, "counts"):
                        brief_round.query_case_evidence(altered)

    def test_reader_only_checkpoint_catalog_filters_and_empty_answer(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "queries.json"
            holdout = json.loads(HOLDOUT.read_bytes())
            holdout["queries"] = [
                {"after": 1, "catalog_record": ["inventory", "19"], "postings": [["lamp", 0]]},
                {"after": 1, "entity": "Nacre", "catalog_record": ["rigging", "19"], "postings": []},
            ]
            path.write_text(json.dumps(holdout))
            receipt, _ = self.replay_case(Path(NATIVE).resolve(), path, 0)
            self.assertEqual(receipt["consumer"]["passed"], 2)


if __name__ == "__main__":
    unittest.main()
