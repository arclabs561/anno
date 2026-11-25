import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import replay_feedback as feedback
import replay_reader_cases as adapter

ROOT = Path(__file__).resolve().parents[2]
CASE = "issuer-local-credit-citations"


class ReaderReplay(unittest.TestCase):
    def fixture_copy(self, root):
        directory = root / adapter.DIRECTORY / "fixtures"
        directory.mkdir(parents=True)
        for name in ("cases.json", "answer_key.json"):
            shutil.copyfile(ROOT / adapter.DIRECTORY / "fixtures" / name, directory / name)
        return directory

    def test_history_and_answer_key_have_independent_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = self.fixture_copy(root)
            original = adapter.describe(root, CASE)
            source = directory / "cases.json"
            suite = json.loads(source.read_bytes())
            next(c for c in suite["cases"] if c["id"] == CASE)["work"] += "-new"
            source.write_bytes(adapter.encode(suite))
            edited = adapter.describe(root, CASE)
            self.assertNotEqual(original["source_sha256"], edited["source_sha256"])
            self.assertEqual(original["oracle_sha256"], edited["oracle_sha256"])
            key = directory / "answer_key.json"
            labels = json.loads(key.read_bytes())
            labels["answers"][CASE]["both-citations"].pop()
            key.write_bytes(adapter.encode(labels))
            relabeled = adapter.describe(root, CASE)
            self.assertEqual(edited["source_sha256"], relabeled["source_sha256"])
            self.assertNotEqual(edited["oracle_sha256"], relabeled["oracle_sha256"])
            with self.assertRaises(ValueError):
                adapter.reviewed(root, "../local-proposal")

    def test_incomplete_or_malformed_reader_responses_are_execution_errors(self):
        _, questions, key = adapter.reviewed(ROOT, CASE)
        incomplete = {q: rows for q, rows in key.items() if q != "both-citations"}
        for body in ({"version": True, "answers": {CASE: key}},
                     {"version": 1, "answers": {CASE: incomplete}},
                     {"version": 1, "answers": {CASE: {**key, "both-citations": None}}}):
            with self.subTest(), self.assertRaises(ValueError):
                adapter.answers([sys.executable, "-c", "import sys; print(sys.argv[1])", json.dumps(body)],
                                ROOT, CASE, questions)
        with self.assertRaises(subprocess.CalledProcessError):
            adapter.answers([sys.executable, "-c", "raise SystemExit(1)"], ROOT, CASE, questions)

    def test_native_readers_catch_one_lost_issuer(self):
        try:
            paths = adapter.executables(ROOT)
        except FileNotFoundError:
            self.skipTest("build anno and reader_answer to run the native mutation control")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture_copy(root)
            scripts = root / "scripts/qa"
            scripts.mkdir(parents=True)
            shutil.copyfile(adapter.__file__, scripts / Path(adapter.__file__).name)
            sqlite_reader = root / adapter.DIRECTORY / "check.py"
            shutil.copyfile(paths["sqlite_reader"], sqlite_reader)
            with patch.dict(os.environ, {"ANNO_BIN": str(paths["anno"]), "ANNO_READER_BIN": str(paths["rust"])}):
                status, passing = feedback.replay_once(root, "reader_cases", CASE)
                self.assertEqual((status, passing["outcome"]), (0, "pass"))
                # Transform one completed SQLite answer. Rust and the independent key
                # still contain both publishers; an adapter checking only Rust would pass.
                sqlite_reader.write_text(
                    "import json, subprocess, sys\n"
                    f"real = {str(paths['sqlite_reader'])!r}\n"
                    "result = json.loads(subprocess.check_output([sys.executable, '-I', real, *sys.argv[1:]]))\n"
                    f"result['answers'][{CASE!r}]['both-citations'].pop()\n"
                    "print(json.dumps(result))\n")
                status, failed = feedback.replay_once(root, "reader_cases", CASE)
                self.assertEqual((status, failed["outcome"]), (1, "fail"))
                self.assertEqual(failed["signature"], "answers:" + adapter.fingerprint([["sqlite", "both-citations"]]))
                self.assertNotEqual(passing["producer_sha256"], failed["producer_sha256"])
                for pin in ("source_sha256", "oracle_sha256"):
                    self.assertEqual(passing[pin], failed[pin])


if __name__ == "__main__":
    unittest.main()
