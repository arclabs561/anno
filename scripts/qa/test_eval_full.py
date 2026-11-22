"""Exercise the full-evaluation shell gate with offline benchmark receipts."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "eval-full.sh"


class FullEvaluationTests(unittest.TestCase):
    def test_at_least_one_successful_nonempty_row_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cargo = root / "cargo"
            cargo.write_text(
                '#!/bin/sh\n'
                'if [ "$1" = build ]; then exit 0; fi\n'
                'while [ "$#" -gt 0 ]; do\n'
                '  case "$1" in\n'
                '    --output) shift; echo "Evaluation summary" > "$1";;\n'
                '    --output-json) shift; echo "$RESULT" > "$1";;\n'
                '  esac\n'
                '  shift\n'
                'done\n'
            )
            cargo.chmod(0o755)
            passed = {"success": True, "num_examples": 1}
            failed = {"success": False, "num_examples": 0}
            for rows, expected in [
                ([], False), ([failed], False),
                ([{"success": True, "num_examples": 0}], False),
                ([{"success": True, "num_examples": "1"}], False),
                ([failed, passed], True),
            ]:
                with self.subTest(rows=rows):
                    result = subprocess.run(
                        ["bash", SCRIPT], cwd=root,
                        env={**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}",
                             "RESULT": json.dumps({"results": rows})},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode == 0, expected, result.stderr)
                    self.assertEqual("Full evaluation complete" in result.stdout, expected)
