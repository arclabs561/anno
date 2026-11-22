"""Offline checks for the dataset URL validation entry point."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("validate_urls.py")


class RegistryInputTests(unittest.TestCase):
    def test_empty_registry_cannot_produce_a_passing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "registry.json")
            output = Path(tmp, "report.json")
            for value in ({}, [], {"datasets": []}):
                with self.subTest(value=value):
                    source.write_text(json.dumps(value))
                    result = subprocess.run(
                        [sys.executable, SCRIPT, "--input", source, "--output", output],
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("nonempty datasets array", result.stderr)
                    self.assertFalse(output.exists())

    def test_registry_row_reaches_the_report_without_a_network_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "registry.json")
            output = Path(tmp, "report.json")
            source.write_text(json.dumps({"datasets": [{"id": "Fixture", "url": ""}]}))
            subprocess.run(
                [sys.executable, SCRIPT, "--input", source, "--output", output],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(output.read_text()), [
                {"name": "Fixture", "url": "", "status": "no_url", "code": None},
            ])
