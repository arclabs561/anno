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

    def test_every_topic_before_reusing_a_topic_in_the_first_publisher_pass(self):
        pool = [qa.Source(host, f"https://{host}.example/doc", "cross-domain", "en", "Latin", domain=topic)
                for topic, hosts in [("arts", ["museum"]), ("law", ["court"]),
                                     ("science", ["nasa", "mit", "lab", "college"])] for host in hosts]
        for seed in range(80):
            with self.subTest(seed=seed):
                chosen = qa.sample_publishers(pool, 3, qa.random.Random(seed))
                self.assertEqual({s.domain for s in chosen}, {"arts", "law", "science"})
                self.assertEqual(len(set(chosen)), len(chosen))

    def test_inventory_flags_mixed_topics_on_a_publisher(self):
        first = qa.CROSS_DOMAIN_SOURCES[0]
        alternate = qa.Source("Different section", first.url + "-different", "cross-domain",
                              first.lang, first.script, domain="different")
        with patch.object(qa, "CROSS_DOMAIN_SOURCES", [first, alternate]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(qa.self_test(), 1)
        self.assertIn("Multiple catalog topics for publisher", output.getvalue())

    def test_language_only_selection_retains_the_multilingual_pool(self):
        with patch("sys.argv", ["qa", "--lang", "hi", "--no-rss", "--count", "1", "--dry-run"]), \
                patch.object(qa, "resolve_url", side_effect=lambda source: source.url) as resolve:
            self.assertEqual(qa.main(), 0)
        self.assertEqual(resolve.call_args.args[0].lang, "hi")

    def test_default_coverage_and_replay_use_captured_articles(self):
        fixed = qa.Source("Museum", "https://museum.example/object", "cross-domain", "fr", "Latin", domain="arts")
        feed = qa.Source("Lab", "https://lab.example/feed", "cross-domain", "en", "Latin", True, domain="science")
        article = "https://lab.example/story"
        digest = "xxh3:0000000000000001"
        def extract(source, binary, url, backend, timeout, verbose):
            return qa.RunResult(source, url, "OK", backend=backend, text_hash=digest)
        with tempfile.TemporaryDirectory() as tmp, patch.object(qa, "CROSS_DOMAIN_SOURCES", [feed, fixed]), \
                patch.object(qa, "discover_publisher_article", return_value=article) as discover, \
                patch.object(qa, "run_anno", side_effect=extract) as run:
            output = qa.Path(tmp) / "discovery.md"
            replay = qa.Path(tmp) / "replay.md"
            def call(*args):
                with patch("sys.argv", ["qa", "--delay", "0", *map(str, args)]):
                    return qa.main()
            self.assertEqual(call("--count", 2, "--seed", 42, "--backends", "pattern,heuristic", "--output", output), 0)
            self.assertEqual(discover.call_count, 1)
            receipt = output.with_suffix(".json")
            rows = json.loads(receipt.read_bytes())
            self.assertEqual(rows["eligible_sources"], {"arts": 1, "science": 1})
            self.assertEqual({s["selected_url"] for s in rows["sources"]}, {fixed.url, article})
            self.assertIn("1 text hashes appeared under multiple catalog topics", output.read_text())
            self.assertEqual(call("--replay", receipt, "--output", replay), 0)
            results = json.loads(replay.with_suffix(".json").read_bytes())
            self.assertEqual(results["replay"]["outcomes"], {s.url: "matched" for s in (fixed, feed)})
            self.assertEqual(results["backends"], ["pattern", "heuristic"])
            digest = "xxh3:0000000000000002"
            with contextlib.redirect_stdout(io.StringIO()) as changed:
                self.assertEqual(call("--replay", receipt, "--backends", "pattern", "--output", replay), 1)
            self.assertIn("Verified inputs: 2/2 selected sources", changed.getvalue())
            self.assertIn("0 matched, 2 changed, 0 unverified", changed.getvalue())
            self.assertEqual(set(json.loads(replay.with_suffix(".json").read_bytes())["replay"]["outcomes"].values()), {"changed"})
            self.assertEqual(discover.call_count, 1)
            calls = run.call_count
            self.assertEqual(call("--replay", receipt, "--output", output), 1)
            self.assertEqual(call("--count", 2, "--output", receipt), 1)
            source = next(s for s in rows["sources"] if s["source_url"] == feed.url)
            for address in ["https://other.example/story", "https://lab.example:8080/story",
                            "https://lab.example@other.example/story", "https://lab.example/feed"]:
                source["selected_url"] = address
                receipt.write_text(json.dumps(rows))
                self.assertEqual(call("--replay", receipt, "--output", replay), 1)
            self.assertEqual(run.call_count, calls)

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
                return qa.RunResult(source, "https://final.example/new", "OK", backend=backend, text_hash="xxh3:0000000000000abc")
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
                             "https://final.example/new", "xxh3:0000000000000abc", "timed out"):
                self.assertIn(expected, report)

    def test_public_summary_excludes_fetched_fields_and_keeps_failure_counts(self):
        source = qa.Source("Catalog", "https://source.example/feed", "cross-domain", "en", "Latin",
                           True, domain="arts")
        private = "PRIVATE-CANARY"
        with tempfile.TemporaryDirectory() as tmp:
            root = qa.Path(tmp)
            report, summary = root / "local.md", root / "public.json"
            run = qa.RunResult(source, "https://source.example/" + private, "error", backend="pattern",
                               error_msg=private, stderr_tail=private, detected_lang=private)
            with patch("sys.argv", ["qa", "--count", "1", "--seed", "42", "--backends", "pattern",
                                    "--output", str(report), "--public-summary", str(summary)]), patch.object(
                qa, "CROSS_DOMAIN_SOURCES", [source]
            ), patch.object(qa, "discover_publisher_article", return_value=run.url), patch.object(
                qa, "run_anno", return_value=run
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(qa.main(), 1)
            self.assertIn(private, report.read_text())
            self.assertNotIn(private, summary.read_text())
            public = json.loads(summary.read_text())
            self.assertEqual(public["outcomes"], {"error": 1})
            self.assertEqual(public["coverage"]["failures"], {"backend": 1})
            self.assertEqual(public["sources"], [{"catalog_name": "Catalog", "resolved": True,
                                                  "checks": {"pattern": "error"}}])

    def test_public_summary_rejects_unreviewed_names_from_replay(self):
        source = qa.Source("Catalog", "https://source.example/doc", "cross-domain", "en", "Latin",
                           domain="arts")
        with tempfile.TemporaryDirectory() as tmp, patch.object(qa, "CROSS_DOMAIN_SOURCES", [source]):
            root = qa.Path(tmp)
            ref, summary, report = root / "source.json", root / "public.json", root / "local.md"
            for backend, eligible in [("unreviewed", {"arts": 1}), ("pattern", {"arts": 1, "private": 1})]:
                receipt = qa.cross_domain_receipt(1, [backend], eligible, {source: source.url}, [
                    qa.RunResult(source, source.url, "OK", backend=backend, text_hash="xxh3:0000000000000001")
                ])
                ref.write_text(json.dumps(receipt))
                with patch("sys.argv", ["qa", "--replay", str(ref), "--output", str(report),
                                        "--public-summary", str(summary)]), patch.object(qa, "run_anno") as run:
                    self.assertEqual(qa.main(), 1)
                    run.assert_not_called()
                    self.assertFalse(summary.exists())

    def test_cross_domain_success_requires_text_hash(self):
        source = qa.CROSS_DOMAIN_SOURCES[0]
        data = {"entities": [], "provenance": {"model": "pattern", "text_chars": 4, "entity_count": 0}}
        proc = subprocess.CompletedProcess([], 0, json.dumps(data), "")
        with patch.object(qa.subprocess, "run", return_value=proc):
            result = qa.run_anno(source, "anno", source.url, "pattern", 1, False)
        self.assertEqual(result.status, "error")
        self.assertIn("hash", result.error_msg)


class PublisherFeeds(unittest.TestCase):
    source = qa.Source("Test feed", "https://news.example/rss", "cross-domain", "en", "Latin", True)

    def test_seed_selects_same_host_articles_independently_of_entry_order(self):
        def choose(entries):
            data = ("<feed xmlns='http://www.w3.org/2005/Atom'><entry>"
                    "<link rel='self' href='https://news.example/self'/>" +
                    "".join(f"<link href='{url}'/>" for url in entries) +
                    "</entry></feed>").encode()
            with patch.object(qa.urllib.request, "urlopen", return_value=io.BytesIO(data)) as fetch:
                page = qa.discover_publisher_article(self.source, 7, 60)
                self.assertEqual(fetch.call_args.kwargs["timeout"], 15)
                return page
        links = ["https://news.example/a", "https://news.example/b", "https://other.example/c"]
        self.assertEqual(choose(["https://news.example/invalid path", "https://[broken", links[0]]), links[0])
        self.assertIn(choose(links), links[:2])
        self.assertEqual(choose(links), choose(reversed(links)))

    def test_feed_size_empty_and_parse_failures(self):
        cases = [(b"<rss><channel/></rss>", "same-host"),
                 (b"<rss>", "no element found"), (b"x" * (qa.MAX_PUBLISHER_FEED_BYTES + 1), "2 MiB")]
        for data, reason in cases:
            with self.subTest(reason=reason), patch.object(qa.urllib.request, "urlopen", return_value=io.BytesIO(data)):
                with self.assertRaisesRegex((ValueError, qa.ET.ParseError), reason):
                    qa.discover_publisher_article(self.source, 1, 3)

    def test_mixed_report_preserves_discovery_failure_and_redirect(self):
        empty_feed = qa.Source("Empty feed", "https://empty.example/rss", "cross-domain", "en", "Latin", True)
        fixed = qa.Source("Fixed", "https://fixed.example/doc", "cross-domain", "en", "Latin")
        pool = [self.source, fixed, empty_feed]

        def fetch(req, timeout):
            if req.full_url == empty_feed.url:
                return io.BytesIO(b"<rss><channel/></rss>")
            self.assertEqual(req.full_url, self.source.url)
            return io.BytesIO(b"<rss><channel><item><link>https://news.example/entry</link></item></channel></rss>")

        def extract(source, binary, url, backend, timeout, verbose):
            return qa.RunResult(source, url + "-final", "OK", backend=backend, text_hash="xxh3:0000000000000123")

        with tempfile.TemporaryDirectory() as tmp, patch.object(qa, "CROSS_DOMAIN_SOURCES", pool), \
                patch.object(qa.urllib.request, "urlopen", side_effect=fetch) as feeds, \
                patch.object(qa, "run_anno", side_effect=extract) as docs:
            report = qa.Path(tmp) / "qa.md"
            with patch("sys.argv", ["qa", "--cross-domain", "--count", "3", "--seed", "42",
                                    "--delay", "0", "--output", str(report)]):
                self.assertEqual(qa.main(), 1)
            self.assertEqual(feeds.call_count, 2)
            self.assertEqual(docs.call_count, 2)
            for value in (self.source.url, "](https://news.example/entry)", "https://news.example/entry-final",
                          "xxh3:0000000000000123", "empty.example | 0 | 1", "feed has no same-host"):
                self.assertIn(value, report.read_text())
            with patch("sys.argv", ["qa", "--cross-domain", "--no-rss", "--count", "1", "--dry-run"]):
                self.assertEqual(qa.main(), 0)
            self.assertEqual(feeds.call_count, 2)
