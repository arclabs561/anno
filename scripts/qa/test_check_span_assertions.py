import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import check_span_assertions as oracle


def prediction(row, entities):
    return dict(id=row["id"], entities=entities,
                source_sha256=hashlib.sha256(row["text"].encode("utf-8")).hexdigest())


def entity(start, end, text, label):
    return dict(start=start, end=end, text=text, type=label)


class TestSpanAssertions(unittest.TestCase):
    def setUp(self):
        self.text = "🪶Zoë (CPA)"
        self.positive = dict(kind="exact_span", start=1, end=4, text="Zoë", types=["PER"])
        self.negative = dict(kind="no_overlap", start=6, end=9, text="CPA", types=["ORG", "PER"])
        self.row = dict(id="unicode", text=self.text, assertions=[self.positive, self.negative])

    def score(self, entities):
        return oracle.score(oracle.load_cases([self.row]), [prediction(self.row, entities)])

    def test_unicode_exact_and_half_open_typed_overlap(self):
        report = self.score([entity(1, 4, "Zoë", "PER"), entity(4, 6, " (", "ORG"),
                             entity(6, 9, "CPA", "LOC")])
        self.assertEqual((report["passed"], report["total"]), (2, 2))

    def test_swallowed_partial_and_exact_negative(self):
        for prediction in [entity(1, 9, "Zoë (CPA", "ORG"), entity(8, 10, "A)", "PER"),
                           entity(6, 9, "CPA", "ORG")]:
            with self.subTest(prediction=prediction):
                report = self.score([prediction])
                self.assertEqual(report["passed"], 0)
                self.assertEqual(report["findings"][1]["overlaps"], [prediction])

    def test_partial_or_wrong_type_fails_positive(self):
        for prediction in [entity(1, 3, "Zo", "PER"), entity(1, 4, "Zoë", "ORG")]:
            self.assertFalse(self.score([prediction])["findings"][0]["passed"])

    def test_invalid_labels_or_ranges_raise(self):
        for patch in [dict(kind="not_exact"), dict(types=[]), dict(types=["ORGG"]),
                      dict(types=["PER", "PER"]), dict(types="PER"), dict(start=True),
                      dict(start=-1), dict(end=20), dict(text="Zo")]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                oracle.load_cases([{**self.row, "assertions": [{**self.positive, **patch}]}])
        for prediction in [entity(4, 8, "Zoë", "PER"), entity(1, 4, "Zoë", "  PER")]:
            with self.assertRaises(ValueError):
                self.score([prediction])

    def test_valid_custom_prediction_and_declared_assertion_type(self):
        # Real `anno batch --model pattern` output on this source.
        hashtag = entity(2, 9, "#Launch", "Hashtag")
        check = dict(kind="no_overlap", start=2, end=9, text="#Launch", types=["ORG"])
        case = dict(id="tag", text="🦊 #Launch at LHA.", assertions=[check])
        predictions = [prediction(case, [hashtag])]
        finding = oracle.score(oracle.load_cases([case]), predictions)["findings"][0]
        self.assertTrue(finding["passed"])
        self.assertEqual(finding["overlaps"], [hashtag])
        check["types"] = ["Hashtag"]
        with self.assertRaises(ValueError):
            oracle.load_cases([case])
        self.assertEqual(oracle.score(oracle.load_cases([case], ["Hashtag"]), predictions)["passed"], 0)
        check["kind"] = "exact_span"
        self.assertEqual(oracle.score(oracle.load_cases([case], ["Hashtag"]), predictions)["passed"], 1)
        check["types"] = ["HashTag"]
        with self.assertRaises(ValueError):
            oracle.load_cases([case], ["Hashtag"])
        hashtag["end"] = 8
        check["types"] = ["ORG"]
        with self.assertRaises(ValueError):
            oracle.score(oracle.load_cases([case]), predictions)

    def test_replay_requires_each_document_once(self):
        cases = oracle.load_cases([self.row])
        good = dict(id="unicode", entities=[])
        for predictions in [[], [good, good], [dict(id="unknown", entities=[])]]:
            with self.assertRaises(ValueError):
                oracle.score(cases, predictions)

    def test_saved_output_must_bind_the_complete_input(self):
        cases = oracle.load_cases([self.row])
        for entities in ([], [entity(1, 4, "Zoë", "PER")]):
            stale = prediction({**self.row, "text": "🪶Zoë waits"}, entities)
            for value in (stale["source_sha256"], None, True, "invalid"):
                with self.subTest(entities=entities, digest=value), self.assertRaisesRegex(
                        ValueError, "source_sha256 differs"):
                    oracle.score(cases, [{**stale, "source_sha256": value}])
            legacy = dict(id=self.row["id"], entities=entities)
            with self.assertRaisesRegex(ValueError, "lacks source_sha256"):
                oracle.score(cases, [legacy])
            with patch("check_span_assertions.subprocess.run") as run:
                run.return_value.stdout = json.dumps(legacy)
                live = oracle.run_batch(cases, "old-anno", "heuristic")
                self.assertEqual(json.loads(run.call_args.kwargs["input"]),
                                 {"id": self.row["id"], "text": self.row["text"]})
                self.assertEqual(live, [prediction(self.row, entities)])
                self.assertEqual(oracle.score(cases, live), self.score(entities))
                run.return_value.stdout = json.dumps(stale)
                with self.assertRaisesRegex(ValueError, "source_sha256 differs"):
                    oracle.score(cases, oracle.run_batch(cases, "anno", "heuristic"))

    def test_jsonl_preserves_unicode_line_separators_inside_values(self):
        row = {"text": "a\u2028b\u0085c"}
        self.assertEqual(oracle.jsonl_rows(json.dumps(row, ensure_ascii=False) + "\r\n"), [row])

    def test_comparison_detects_loss_even_when_total_improves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{**self.row, "id": str(i), "assertions": [self.positive]} for i in range(4)]
            rows[0]["domain"] = "arts"
            for row in rows[1:3]:
                row["domain"] = "finance"
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps(rows))
            old, new = root / "old.jsonl", root / "new.jsonl"
            for target, good in [(old, {0}), (new, {1, 2})]:
                target.write_text("".join(json.dumps(prediction(row,
                    [entity(1, 4, "Zoë", "PER")] if int(row["id"]) in good else [])) + "\n"
                    for row in reversed(rows)))
            cmd = [sys.executable, oracle.__file__, "--fixture", str(fixture),
                   "--baseline-predictions", str(old), "--predictions"]
            changed = subprocess.run([*cmd, str(new)], capture_output=True, text=True)
            self.assertEqual(changed.returncode, 1, changed.stderr)
            report = json.loads(changed.stdout)
            self.assertEqual(report["comparison"],
                             {"baseline_passed": 1, "gained": 2, "lost": 1})
            groups = {row["domain"]: row for row in report["domains"]}
            self.assertEqual(groups["arts"]["comparison"], {"baseline_passed": 1, "gained": 0, "lost": 1})
            self.assertEqual(groups["finance"]["documents"], 2)
            self.assertEqual(groups["finance"]["comparison"], {"baseline_passed": 0, "gained": 2, "lost": 0})
            self.assertEqual(groups[None]["exact_span"], {"passed": 0, "total": 1})
            self.assertEqual(groups[None]["no_overlap"], {"passed": 0, "total": 0})
            same = subprocess.run([*cmd, str(old)], capture_output=True, text=True)
            self.assertEqual(same.returncode, 0, same.stderr)
            self.assertEqual(json.loads(same.stdout)["passed"], 1)
            old.write_text(old.read_text().replace(prediction(self.row, [])["source_sha256"], "bad"))
            stale_baseline = subprocess.run([*cmd, str(new)], capture_output=True, text=True)
            self.assertEqual(stale_baseline.returncode, 2)
            self.assertFalse(stale_baseline.stdout)

    def test_domain_counts_keep_empty_predictions_and_separate_negative_controls(self):
        fixture = oracle.DEFAULT_FIXTURE.with_name("domain-span-assertions.json")
        cases = oracle.load_cases(json.loads(fixture.read_text()))
        report = oracle.score(cases, [prediction(row, []) for row in cases.values()])
        domains = oracle.summarize_domains(cases, report["findings"])
        self.assertGreaterEqual(len(domains), 8)
        for row in domains:
            self.assertGreaterEqual(row["documents"], 2)
            self.assertGreaterEqual(row["exact_span"]["total"], 2)
            self.assertEqual(row["exact_span"]["passed"], 0)
            self.assertGreaterEqual(row["no_overlap"]["total"], 2)
            self.assertEqual(row["no_overlap"]["passed"], row["no_overlap"]["total"])
            self.assertNotIn("comparison", row)
        for invalid in ["", "law\npolicy", 4]:
            with self.subTest(domain=invalid), self.assertRaisesRegex(ValueError, "domain"):
                oracle.load_cases([{**self.row, "domain": invalid}])

    def test_replay_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps([self.row]))
            saved = root / "saved.jsonl"
            cmd = [sys.executable, oracle.__file__, "--fixture", str(fixture), "--predictions", str(saved)]
            for entities, code in [([entity(1, 4, "Zoë", "PER")], 0), ([], 1),
                                   ([entity(2, 5, "Zoë", "PER")], 2)]:
                saved.write_text(json.dumps(prediction(self.row, entities)) + "\n")
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                if code < 2:
                    self.assertEqual(json.loads(result.stdout)["total"], 2)
