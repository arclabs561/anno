"""Regression tests for BERT trace artifact identity checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


_SCRIPT = Path(__file__).with_name("compare_bert_onnx_traces.py")
_SPEC = importlib.util.spec_from_file_location("bert_trace_comparator", _SCRIPT)
assert _SPEC and _SPEC.loader
comparator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparator)


def trace(graph: str | None, tokenizer: str | None, config: str | None) -> dict:
    return {
        "artifacts": {
            "graph": None if graph is None else {"sha256": graph},
            "tokenizer": None if tokenizer is None else {"sha256": tokenizer},
            "config": None if config is None else {"sha256": config},
        }
    }


class ArtifactIdentityTests(unittest.TestCase):
    def test_rejects_missing_graph_and_tokenizer_on_both_sides(self) -> None:
        failures = comparator.compare_artifact_identity({}, {})
        self.assertIn("artifact graph is missing or has an invalid SHA-256", failures)
        self.assertIn("artifact tokenizer is missing or has an invalid SHA-256", failures)

    def test_rejects_a_hash_mismatch(self) -> None:
        left = trace("a" * 64, "b" * 64, "c" * 64)
        right = trace("d" * 64, "b" * 64, "c" * 64)
        self.assertEqual(
            comparator.compare_artifact_identity(left, right),
            [f"artifact graph differs: rust={'a' * 64!r} reference={'d' * 64!r}"],
        )


if __name__ == "__main__":
    unittest.main()
