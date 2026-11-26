# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check the evidence stream independently of model weights."""
import tempfile
from pathlib import Path
import unittest

from run_fcoref_endpoint_stages import observations


class CaptureTests(unittest.TestCase):
    def capture(self, rank_lines):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            runner = directory / "runner"
            runner.write_text(
                "#!/usr/bin/env python3\n"
                "import json,sys\n"
                "for i, line in enumerate(sys.stdin, 1):\n"
                "    doc = json.loads(line)\n"
                f"    for _ in range({rank_lines}):\n"
                "        print('FCOREF_RANKS', json.dumps([1, [[0, 0, i]]]), file=sys.stderr)\n"
                "    print(json.dumps(dict(id=doc['id'], producer='fcoref', clusters=[])), flush=True)\n"
            )
            runner.chmod(0o700)
            rows = [{"ID": key, "Text": "Mira"} for key in ("first", "second")]
            return list(observations(rows, runner, directory))

    def test_equal_length_documents_have_independent_traces(self):
        records = self.capture(1)
        self.assertEqual([(row["id"], row["trace"][1][0][-1]) for row in records],
                         [("first", 1), ("second", 2)])

    def test_missing_or_duplicate_trace_rejects_run(self):
        for count in (0, 2):
            with self.subTest(rank_lines=count), self.assertRaisesRegex(ValueError, "one rank table"):
                self.capture(count)
