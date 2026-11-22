import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import qa_random_urls as qa


class ExtractionVerdict(unittest.TestCase):
    def run_extractor(self, output, source=qa.WIKIPEDIA_SOURCES[0]):
        proc = subprocess.CompletedProcess([], 0, json.dumps(output), "")
        with patch.object(qa.subprocess, "run", return_value=proc):
            return qa.run_anno(source, "anno", "https://example.org", "pattern", 1, False)

    def test_zero_matches_in_nonempty_text(self):
        output = {"entities": [], "provenance": {"model": "regex", "text_chars": 4, "entity_count": 0}}
        result = self.run_extractor(output)
        self.assertEqual((result.status, result.backend, result.reported_model), ("OK", "pattern", "regex"))

    def test_fixed_japanese_document_requires_japanese_extraction(self):
        source = next(s for s in qa.DOCUMENT_SOURCES if s.lang == "ja")
        prov = {"model": "pattern", "text_chars": 200, "entity_count": 0}
        for language in ("en", None, "ja"):
            with self.subTest(language=language):
                data = {"entities": [], "provenance": dict(prov)}
                if language:
                    data["provenance"]["language"] = language
                result = self.run_extractor(data, source)
                self.assertEqual(result.status, "OK" if language == "ja" else "error")
        # A sampled feed may link to an article in a different language.
        self.assertEqual(self.run_extractor(data, qa.RSS_SOURCES[0]).status, "OK")

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

    def test_second_backend_reuses_final_article_url(self):
        article = "https://example.org/article"
        def extract(source, binary, url, backend, timeout, verbose):
            return qa.RunResult(qa.DOCUMENT_SOURCES[0], article, "OK", backend=backend)
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", [
            "qa_random_urls", "--category", "wikipedia", "--count", "1",
            "--seed", "42", "--backends", "pattern,heuristic", "--output", tmp + "/qa.md",
        ]), patch.object(qa, "run_anno", side_effect=extract) as runner:
            self.assertEqual(qa.main(), 0)
        self.assertIn("Special:Random", runner.call_args_list[0].args[2])
        self.assertEqual(runner.call_args_list[1].args[2], article)

    def test_empty_selection_fails(self):
        with patch("sys.argv", ["qa_random_urls", "--count", "0"]):
            self.assertEqual(qa.main(), 1)
