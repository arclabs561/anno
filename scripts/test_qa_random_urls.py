import json
import subprocess
import unittest
from unittest.mock import patch

import qa_random_urls as qa


class ExtractionVerdict(unittest.TestCase):
    def run_extractor(self, output):
        proc = subprocess.CompletedProcess([], 0, json.dumps(output), "")
        with patch.object(qa.subprocess, "run", return_value=proc):
            return qa.run_anno("anno", "https://example.org", "pattern", 1, False)

    def test_zero_matches_in_nonempty_text(self):
        output = {"entities": [], "provenance": {"model": "regex", "text_chars": 4, "entity_count": 0}}
        result = self.run_extractor(output)
        self.assertEqual((result.status, result.backend, result.reported_model), ("OK", "pattern", "regex"))

    def test_empty_or_malformed_result(self):
        for output in ({}, [], {"entities": []}, {"entities": {}, "provenance": {}}):
            with self.subTest(output=output):
                self.assertEqual(self.run_extractor(output).status, "error")

    def test_counts_and_character_bounds(self):
        entity = {"text": "東京", "type": "LOC", "start": 2, "end": 4}
        prov = {"model": "test", "text_chars": 4, "entity_count": 1}
        self.assertEqual(self.run_extractor({"entities": [entity], "provenance": prov}).status, "OK")
        for row, metadata in (
            (None, prov), (dict(entity, end=6), prov), (dict(entity, start=True), prov),
            (dict(entity, type=""), prov), (entity, dict(prov, entity_count=0)),
            (entity, dict(prov, text_chars=0)),
        ):
            with self.subTest(row=row, metadata=metadata):
                self.assertEqual(self.run_extractor({"entities": [row], "provenance": metadata}).status, "error")

    def test_empty_selection_fails(self):
        with patch("sys.argv", ["qa_random_urls", "--count", "0"]):
            self.assertEqual(qa.main(), 1)
