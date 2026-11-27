# SPDX-License-Identifier: MIT OR Apache-2.0
import json
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
                with self.subTest(changes=changes), patch("subprocess.check_output",
                        side_effect=[json.dumps(output), export.read_text()]):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            check_case("anno", case, generation, export)
                    else:
                        self.assertEqual(check_case("anno", case, generation, export)["directed_edges"], 0)


if __name__ == "__main__":
    unittest.main()
