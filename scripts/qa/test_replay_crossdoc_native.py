import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import replay_crossdoc_native as adapter


CASE = "query-catalog-publication"
CONTRACT = adapter.CASES[CASE]


def result(status="ok", name=CONTRACT.test, counts="1 passed; 0 failed; 0 ignored", detail="", code=0):
    output = (f"running 1 test\ntest {name} ... {status}\n{detail}\n"
              f"test result: {status}. {counts}; 0 measured; 4 filtered out; finished in 0.01s\n")
    return subprocess.CompletedProcess([], code, output)


class CrossdocNativeTests(unittest.TestCase):
    def test_only_registered_cases_have_a_family(self):
        for case in adapter.CASES:
            self.assertEqual(adapter.family_for_case(None, case), case)
        with self.assertRaises(ValueError):
            adapter.family_for_case(None, "--test unknown")

    def test_pass_requires_one_named_executed_case(self):
        self.assertEqual(adapter._verdict(CONTRACT, result()), ("pass", adapter.PASS_SIGNATURE))
        for output in [
            result(name="different_test"), result(counts="0 passed; 0 failed; 1 ignored"),
            result(counts="0 passed; 0 failed; 0 ignored"),
            subprocess.CompletedProcess([], 0, "running 0 tests\ntest result: ok."),
            result(code=-9),
        ]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                adapter._verdict(CONTRACT, output)

    def test_named_contract_failure_is_distinct_from_infrastructure(self):
        def failed(reason, source=CONTRACT.source):
            detail = f"thread '{CONTRACT.test}' (123) panicked at {source}:19:7:\n{reason}"
            return result("FAILED", counts="0 passed; 1 failed; 0 ignored", detail=detail, code=101)

        status, signature = adapter._verdict(CONTRACT, failed("assertion `left == right` failed"))
        self.assertEqual(status, "fail")
        self.assertRegex(signature, "^" + adapter.FAIL_SIGNATURE + "$")
        for invalid in [failed("called Result::unwrap() on an Err value: Os { code: 28 }"),
                        failed("assertion failed", source="other.rs"), result(code=101)]:
            with self.assertRaises(ValueError):
                adapter._verdict(CONTRACT, invalid)

    def test_cli_producer_is_the_actual_binary_and_build_failures_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / CONTRACT.source
            source.parent.mkdir(parents=True)
            source.write_text("fixture and oracle")
            harness, cli = root / "harness", root / "anno"
            harness.write_bytes(b"test executable")
            cli.write_bytes(b"actual CLI executable")
            def artifact(kind, source, path, is_test):
                return json.dumps({"reason": "compiler-artifact", "target": {"kind": [kind], "src_path": str(source)},
                                   "executable": str(path), "profile": {"test": is_test}})
            receipts = "\n".join([artifact("test", source, harness, True),
                                  artifact("bin", root / "crates/anno-cli/src/main.rs", cli, False)])
            with patch.object(subprocess, "check_output", return_value=receipts) as build, \
                    patch.object(subprocess, "run", return_value=result()) as run:
                observed = adapter.replay(root, CASE)
                self.assertIn("--offline", build.call_args.args[0])
                self.assertEqual(run.call_args.args[0][:3], [harness, "--exact", CONTRACT.test])
                self.assertEqual(observed["producer_sha256"], hashlib.sha256(cli.read_bytes()).hexdigest())
                with patch.object(subprocess, "check_output", return_value=receipts.splitlines()[0]), \
                        self.assertRaises(ValueError):
                    adapter.replay(root, CASE)
                with patch.object(subprocess, "check_output", side_effect=subprocess.CalledProcessError(101, "cargo")), \
                        self.assertRaises(subprocess.CalledProcessError):
                    adapter.replay(root, CASE)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
