import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay_document_delivery as adapter


ROOT = Path(__file__).resolve().parents[2]
CASE = "historical-receipt"


class DeliveryCorpusTest(unittest.TestCase):
    def test_reviewed_seed_is_pinned_separately_from_the_oracle(self):
        cases = [adapter.describe(ROOT, case) for case in sorted(adapter.CASES)]
        self.assertEqual(len({case["source_sha256"] for case in cases}), 5)
        self.assertEqual(len({case["oracle_sha256"] for case in cases}), 1)
        with self.assertRaisesRegex(ValueError, "unknown reviewed delivery seed"):
            adapter.pins(ROOT, "../unreviewed-history")

    def test_only_a_completed_pinned_native_assertion_creates_a_failure(self):
        name = adapter.CONTRACT.test
        marker = "delivery-oracle " + " ".join(adapter.pins(ROOT, CASE))
        detail = f"thread '{name}' panicked at {adapter.SUPPORT.source}:101:9:\nassertion failed"
        summary = "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s"

        def failed(lines):
            return subprocess.CompletedProcess([], 101, "\n".join([
                "running 1 test", f"test {name} ... FAILED", *lines, summary,
            ]))

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "native"
            binary.write_bytes(b"synthetic binary")
            with patch.object(adapter.native, "_artifacts", return_value=(binary, binary)), \
                    patch.object(adapter, "pins", return_value=marker.split()[1:]), \
                    patch.object(subprocess, "run") as run:
                run.return_value = failed([marker, detail])
                self.assertEqual(adapter.replay(ROOT, CASE)["outcome"], "fail")
                for lines in ([detail], [marker, marker, detail], [marker, "setup failed"]):
                    with self.subTest(lines=lines), self.assertRaises(ValueError):
                        run.return_value = failed(lines)
                        adapter.replay(ROOT, CASE)
