# SPDX-License-Identifier: MIT OR Apache-2.0
import json
from subprocess import CompletedProcess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_gliner_relex_cli import FIXTURE, check_case, stock_decision_digest


class RelexParityTests(unittest.TestCase):
    def test_stock_digest_preserves_direction_and_document_assignment(self):
        case = {"name": "row1", "entities": [], "relations": [
            {"head": [0, 1], "tail": [2, 3], "type": "example", "confidence": 0.9}]}
        baseline = stock_decision_digest([case])
        self.assertNotEqual(baseline, stock_decision_digest([dict(case, name="row2")]))
        edge = case["relations"][0]
        edge["confidence"] = 0.89
        self.assertEqual(baseline, stock_decision_digest([case]))
        edge["head"], edge["tail"] = edge["tail"], edge["head"]
        self.assertNotEqual(baseline, stock_decision_digest([case]))

    def test_empty_output_still_requires_the_trained_runtime_and_generation(self):
        case = next(case for case in json.loads(FIXTURE.read_text())["cases"] if case["name"] == "empty")
        generation = "a" * 64
        producer = {"source": "knowledgator/gliner-relex-large-v1.0", "method": "Neural",
                    "model_version": "sha256:" + generation}
        expected = {"model": "gliner-relex", "runtime_model": "GLiNER-ONNX", "relation_producer": producer}
        with tempfile.TemporaryDirectory() as directory:
            export = Path(directory, "export.json")
            export.write_text('{"signals": []}')
            for changes, error in (({}, None), ({"runtime_model": "fallback"}, "runtime"),
                                   ({"relation_producer": dict(producer, model_version="old")}, "producer")):
                output = {"entities": [], "relations": [], "provenance": expected | changes}
                replies = [CompletedProcess([], 0, stdout=data, stderr=b"")
                           for data in (json.dumps(output).encode(), export.read_bytes())]
                with self.subTest(changes=changes), patch("subprocess.run", side_effect=replies):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            check_case("anno", case, generation, export)
                    else:
                        self.assertEqual(check_case("anno", case, generation, export)["directed_edges"], 0)

    def test_failure_reports_the_stage_and_exit_code_without_source_or_process_output(self):
        case = json.loads(FIXTURE.read_bytes())["cases"][0] | {"text": "PRIVATE_FIXTURE_SENTENCE"}
        reply = CompletedProcess([], 17, stdout=b"PRIVATE_STDOUT", stderr=b"PRIVATE_STDERR")
        with tempfile.TemporaryDirectory() as directory, patch("subprocess.run", return_value=reply) as run:
            with self.assertRaises(ValueError) as failure:
                check_case("anno", case, "a" * 64, Path(directory, "export.json"))
        self.assertEqual(str(failure.exception), "Native extraction exited 17")
        self.assertTrue(run.call_args.kwargs["capture_output"])


if __name__ == "__main__":
    unittest.main()
