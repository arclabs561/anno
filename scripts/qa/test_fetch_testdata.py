"""Exercise local capture replacement without connecting to the source websites."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "fetch_testdata.sh"
MOCK_CURL = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]
assert '--location' in args
output = Path(args[args.index('--output') + 1])
url = args[-1]
body = b'%PDF-1.7\\n' if '/pdf/' in url else b'<html>'
body += b'x' * 120
status = 200
if os.environ.get('FAIL_CAPTURES'):
    if 'climate.ec.europa' in url:
        status = 404
    elif 'consilium' in url:
        status = 302
    elif 'news.un.org' in url:
        body = b'tiny'
    elif url.endswith('/pdf/2101.00884'):
        body = b'access notice ' * 20
    elif url.endswith('/abs/2309.14084'):
        output.write_bytes(b'partial download')
        sys.exit(18)
if status >= 400 and '--fail' in args:
    sys.exit(22)
output.write_bytes(body)
print(status, end='')
"""


class CaptureTests(unittest.TestCase):
    def test_failed_refresh_preserves_good_captures_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curl = root / "curl"
            curl.write_text(MOCK_CURL)
            curl.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"])
            command = ["bash", str(SCRIPT)]
            run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            captures = root / "testdata/real_world"
            self.assertEqual(len(list(captures.glob("*.html"))), 6)
            pdfs = sorted(captures.glob("*.pdf"))
            self.assertEqual(len(pdfs), 2)
            self.assertTrue(all(path.read_bytes().startswith(b"%PDF-") for path in pdfs))
            self.assertIn("arxiv_org_pdf_2309_14084.pdf (129 bytes)", run.stdout)

            # Use distinct prior bodies so a failed replacement is observable.
            paths = sorted(captures.iterdir())
            for path in paths:
                path.write_bytes(b"previous capture")
            run = subprocess.run(
                command, cwd=root, env={**env, "FAIL_CAPTURES": "1"}, capture_output=True, text=True
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(sorted(captures.iterdir()), paths)
            for path in paths:
                refreshed = path.name in {
                    "openai_com_policies_usage-policies_.html",
                    "arxiv_org_pdf_2309_14084.pdf",
                    "arxiv_org_abs_2101_00884.html",
                }
                self.assertEqual(path.read_bytes() != b"previous capture", refreshed, path.name)
