import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import qa_random_urls as qa


class ExtractionVerdict(unittest.TestCase):
    def test_source_inventory(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = qa.self_test()
        self.assertEqual(status, 0, output.getvalue())

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

    def test_comparison_requires_matching_input_from_each_backend(self):
        source = qa.DOCUMENT_SOURCES[0]
        first = qa.RunResult(source, source.url, "OK", backend="pattern", text_hash="a")
        for second, message in [
            (qa.RunResult(source, source.url, "OK", backend="heuristic", text_hash="a"), "1/1 sources"),
            (qa.RunResult(source, source.url, "OK", backend="heuristic", text_hash="b"), "Different input text"),
            (qa.RunResult(source, source.url, "OK", backend="heuristic"), "Incomplete input comparison"),
            (qa.RunResult(source, source.url, "error", backend="heuristic"), "Incomplete input comparison"),
        ]:
            with self.subTest(message=message):
                report = qa.generate_report([first, second], 42, ["pattern", "heuristic"], qa.datetime.now())
                self.assertIn(message, report)
                if second.text_hash != first.text_hash:
                    self.assertIn("0/1 sources", report)

    def test_empty_selection_fails(self):
        with patch("sys.argv", ["qa_random_urls", "--count", "0"]):
            self.assertEqual(qa.main(), 1)


class CrossDomainSampling(unittest.TestCase):
    def test_rotates_publishers_before_reusing_hosts(self):
        pool = [qa.Source(f"{host}-{n}", f"https://{host}.example/{n}", "cross-domain", "en", "Latin")
                for host in ("science", "law", "arts") for n in (1, 2)]
        sampled = qa.sample_publishers(pool, 6, qa.random.Random(17))
        self.assertEqual(sampled, qa.sample_publishers(pool, 6, qa.random.Random(17)))
        self.assertEqual(set(sampled), set(pool))
        for group in (sampled[:3], sampled[3:]):
            self.assertEqual(len({qa.urlsplit(source.url).hostname for source in group}), 3)
        self.assertEqual(len(qa.CROSS_DOMAIN_SOURCES), len({s.url for s in qa.CROSS_DOMAIN_SOURCES}))
        self.assertGreaterEqual(len({qa.urlsplit(s.url).hostname for s in qa.CROSS_DOMAIN_SOURCES}), 6)

    def test_oversized_filtered_request_fails_before_fetch(self):
        with patch("sys.argv", ["qa", "--cross-domain", "--lang", "ja", "--count", "3"]), \
             patch.object(qa, "resolve_url") as resolve:
            self.assertEqual(qa.main(), 1)
        resolve.assert_not_called()

    def test_partial_failure_records_hosts_final_url_and_hash(self):
        pool = [qa.Source("Test A", "https://a.example/doc", "cross-domain", "en", "Latin"),
                qa.Source("Test B", "https://b.example/doc", "cross-domain", "en", "Latin")]

        def extract(source, binary, url, backend, timeout, verbose):
            if source == pool[0]:
                return qa.RunResult(source, "https://final.example/new", "OK", backend=backend, text_hash="xxh3:abc")
            return qa.RunResult(source, url, "timeout", backend=backend, error_msg="timed out")

        with tempfile.TemporaryDirectory() as tmp:
            report_path = qa.Path(tmp) / "report.md"
            with patch("sys.argv", ["qa", "--cross-domain", "--count", "2", "--seed", "5",
                                    "--delay", "0", "--output", str(report_path)]), \
                 patch.object(qa, "CROSS_DOMAIN_SOURCES", pool), \
                 patch.object(qa, "run_anno", side_effect=extract) as calls:
                self.assertEqual(qa.main(), 1)
            self.assertEqual(calls.call_count, 2)
            report = report_path.read_text()
            for expected in ("| a.example | 1 | 0 |", "| b.example | 0 | 1 |", "https://a.example/doc",
                             "https://final.example/new", "xxh3:abc", "timed out"):
                self.assertIn(expected, report)

    def test_cross_domain_success_requires_text_hash(self):
        source = qa.CROSS_DOMAIN_SOURCES[0]
        data = {"entities": [], "provenance": {"model": "pattern", "text_chars": 4, "entity_count": 0}}
        proc = subprocess.CompletedProcess([], 0, json.dumps(data), "")
        with patch.object(qa.subprocess, "run", return_value=proc):
            result = qa.run_anno(source, "anno", source.url, "pattern", 1, False)
        self.assertEqual(result.status, "error")
        self.assertIn("hash", result.error_msg)
