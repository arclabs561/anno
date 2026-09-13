"""Negative contract tests for scripts.qa.run_panel."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import math
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("run_panel.py")
MANIFEST_PATH = MODULE_PATH.parents[2] / "scripts/qa/core-panel.json"
SPEC = importlib.util.spec_from_file_location("run_panel", MODULE_PATH)
assert SPEC and SPEC.loader
panel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = panel
SPEC.loader.exec_module(panel)


SUITE = {
    "id": "test",
    "tasks": ["ner"],
    "datasets": ["WikiGold"],
    "backends": ["heuristic", "stacked"],
    "seeds": [42],
    "max_examples": 20,
    "allowed_incompatibilities": [],
}


def result(backend: str, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "task": "NER",
        "dataset": "WikiGold",
        "backend": backend,
        "seed": 42,
        "success": True,
        "error": None,
        "num_examples": 20,
        "metrics": {"f1": 0.5, "precision": 0.5},
        "provenance": {
            "schema_version": 1,
            "build": {"package_version": "0.13.0", "enabled_features": ["eval"]},
            "dataset": {
                "source_url": "https://example.invalid/wikigold",
                "data_source": "local cache",
                "sentence_count": 20,
            },
            "backend": {"requested": backend, "effective": backend},
            "runtime": {"cached_only": True, "seed": 42, "max_examples": 20},
        },
    }
    value.update(overrides)
    return value


class ValidateSuiteTests(unittest.TestCase):
    def valid(self) -> dict[str, object]:
        return {"results": [result("heuristic"), result("stacked")]}

    def assert_contract_fails(self, artifact: dict[str, object]) -> None:
        with self.assertRaises(panel.ContractError):
            panel.validate_suite(SUITE, artifact)

    def test_accepts_exact_nonempty_cached_results(self) -> None:
        rows = panel.validate_suite(SUITE, self.valid())
        self.assertEqual(2, len(rows))

    def test_rejects_zero_results_even_when_process_would_exit_zero(self) -> None:
        self.assert_contract_fails({"results": []})

    def test_rejects_non_object_artifact(self) -> None:
        self.assert_contract_fails([])

    def test_rejects_duplicate_cell(self) -> None:
        artifact = self.valid()
        artifact["results"].append(result("heuristic"))
        self.assert_contract_fails(artifact)

    def test_rejects_missing_expected_cell(self) -> None:
        self.assert_contract_fails({"results": [result("heuristic")]})

    def test_rejects_unexpected_incompatibility(self) -> None:
        artifact = self.valid()
        artifact["results"][0] = result("heuristic", success=False, error="incompatible: labels")
        self.assert_contract_fails(artifact)

    def test_accepts_only_declared_incompatibility(self) -> None:
        suite = copy.deepcopy(SUITE)
        suite["allowed_incompatibilities"] = [{"task": "NER", "dataset": "WikiGold", "backend": "heuristic"}]
        artifact = self.valid()
        artifact["results"][0] = result("heuristic", success=False, error="incompatible: labels")
        self.assertEqual(2, len(panel.validate_suite(suite, artifact)))

    def test_rejects_hard_failure_for_declared_incompatibility(self) -> None:
        suite = copy.deepcopy(SUITE)
        suite["allowed_incompatibilities"] = [{"task": "NER", "dataset": "WikiGold", "backend": "heuristic"}]
        artifact = self.valid()
        artifact["results"][0] = result("heuristic", success=False, error="model unavailable")
        with self.assertRaises(panel.ContractError):
            panel.validate_suite(suite, artifact)

    def test_rejects_nonfinite_metric(self) -> None:
        artifact = self.valid()
        artifact["results"][0] = result("heuristic", metrics={"f1": math.nan})
        self.assert_contract_fails(artifact)

    def test_rejects_legacy_provenance(self) -> None:
        artifact = self.valid()
        artifact["results"][0] = result("heuristic", provenance={"schema_version": 0})
        self.assert_contract_fails(artifact)

    def test_rejects_missing_primary_metric(self) -> None:
        artifact = self.valid()
        artifact["results"][0] = result("heuristic", metrics={"precision": 0.5})
        self.assert_contract_fails(artifact)

    def test_rejects_provenance_that_does_not_match_result_identity(self) -> None:
        artifact = self.valid()
        bad = copy.deepcopy(result("heuristic"))
        bad["provenance"]["runtime"]["seed"] = 73
        artifact["results"][0] = bad
        self.assert_contract_fails(artifact)

    def test_refuses_stale_artifact_before_running_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "test-seed42.json").write_text("{}")
            with self.assertRaises(panel.ContractError):
                panel.run_suite(Path(sys.executable), SUITE, output, 1)

    def test_run_mode_refuses_stale_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "panel-summary.json").write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    1,
                    panel.main(
                        [
                            "--manifest",
                            str(MANIFEST_PATH),
                            "--output-dir",
                            str(output),
                            "--anno-bin",
                            sys.executable,
                            "--suite",
                            "smoke",
                        ]
                    ),
                )

    def fixture_executable(self, directory: Path, body: str) -> Path:
        fixture = directory / "fake-anno"
        fixture.write_text("#!/bin/sh\n" + body)
        fixture.chmod(fixture.stat().st_mode | stat.S_IXUSR)
        return fixture

    def test_exit_zero_without_result_fails_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt"
            binary = self.fixture_executable(Path(directory), "exit 0\n")
            with contextlib.redirect_stderr(io.StringIO()):
                code = panel.main(
                    [
                        "--manifest",
                        str(MANIFEST_PATH),
                        "--output-dir",
                        str(output),
                        "--anno-bin",
                        str(binary),
                        "--suite",
                        "smoke",
                    ]
                )
            self.assertEqual(1, code)
            self.assertTrue((output / "smoke-seed42.log").is_file())
            self.assertFalse((output / "panel-summary.json").exists())
            self.assertFalse((output / "panel-summary.md").exists())

    def test_timeout_retains_partial_process_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt"
            binary = self.fixture_executable(Path(directory), "printf 'partial output\\n'\nsleep 2\n")
            with contextlib.redirect_stderr(io.StringIO()):
                code = panel.main(
                    [
                        "--manifest",
                        str(MANIFEST_PATH),
                        "--output-dir",
                        str(output),
                        "--anno-bin",
                        str(binary),
                        "--suite",
                        "smoke",
                        "--timeout-seconds",
                        "1",
                    ]
                )
            self.assertEqual(1, code)
            log = (output / "smoke-seed42.log").read_text()
            self.assertIn("# exit-status: unknown", log)
            self.assertIn("timed out after 1s", log)
            self.assertIn("partial output", log)
            self.assertFalse((output / "panel-summary.json").exists())

    def test_child_forces_model_downloads_off_without_mutating_parent_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt"
            binary = self.fixture_executable(
                Path(directory),
                "printf 'ANNO_NO_DOWNLOADS=%s HF_HUB_OFFLINE=%s\\n' \"$ANNO_NO_DOWNLOADS\" \"$HF_HUB_OFFLINE\"\nexit 0\n",
            )
            with mock.patch.dict(os.environ, {"ANNO_NO_DOWNLOADS": "0", "HF_HUB_OFFLINE": "0"}):
                with contextlib.redirect_stderr(io.StringIO()):
                    code = panel.main(
                        [
                            "--manifest",
                            str(MANIFEST_PATH),
                            "--output-dir",
                            str(output),
                            "--anno-bin",
                            str(binary),
                            "--suite",
                            "smoke",
                        ]
                    )
                self.assertEqual("0", os.environ["ANNO_NO_DOWNLOADS"])
                self.assertEqual("0", os.environ["HF_HUB_OFFLINE"])
            log = (output / "smoke-seed42.log").read_text()
            self.assertIn("ANNO_NO_DOWNLOADS=1 HF_HUB_OFFLINE=1", log)
            self.assertEqual(1, code)


if __name__ == "__main__":
    unittest.main()
