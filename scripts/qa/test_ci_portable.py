"""Provider-independent exit status and failure receipt tests; no compilation."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ci-portable.sh"


class PortableFailureTests(unittest.TestCase):
    def exercise(self, lane: str, cargo_exit: int, python_exit: int) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            programs = {
                "cargo": f'if [ "$1" = --version ]; then echo cargo-test; exit 0; fi\necho deliberate-cargo-failure\nexit {cargo_exit}\n',
                "rustc": "echo rustc-test\n",
                "just": "exit 0\n",
                "python3": f"echo deliberate-python-failure\nexit {python_exit}\n",
                "cache-wrapper": "echo cache-statistics-test\n",
            }
            for name, body in programs.items():
                executable = bin_dir / name
                executable.write_text("#!/bin/sh\n" + body)
                executable.chmod(0o755)
            receipts = root / "receipts"
            environment = dict(os.environ)
            environment.update(
                PATH=str(bin_dir) + os.pathsep + environment.get("PATH", ""),
                ANNO_ARTIFACT_DIR=str(receipts),
                RUSTC_WRAPPER=str(bin_dir / "cache-wrapper"),
            )
            completed = subprocess.run(
                ["bash", str(SCRIPT), lane], env=environment, cwd=root,
                capture_output=True, text=True, timeout=10, check=False,
            )
            metadata = (receipts / "metadata.txt").read_text()
            logs = "\n".join(path.read_text() for path in (receipts / "logs").glob("*.log"))
            self.assertTrue((receipts / "Cargo.lock").is_file())
            self.assertIn("cache-statistics-test", (receipts / "sccache-stats.txt").read_text())
            return completed.returncode, metadata, logs

    def test_compiler_failure_survives_tee_and_exit_receipts(self) -> None:
        code, metadata, logs = self.exercise("check", 23, 0)
        self.assertEqual(code, 23)
        self.assertIn("exit_status=23", metadata)
        self.assertIn("deliberate-cargo-failure", logs)

    def test_blocking_lane_fails_when_panel_contract_tests_fail(self) -> None:
        code, metadata, logs = self.exercise("blocking", 0, 29)
        self.assertEqual(code, 29)
        self.assertIn("exit_status=29", metadata)
        self.assertIn("deliberate-python-failure", logs)
        self.assertNotIn("deliberate-cargo-failure", logs)


if __name__ == "__main__":
    unittest.main()
