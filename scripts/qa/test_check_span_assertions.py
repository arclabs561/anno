import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import check_span_assertions as oracle


def entity(start, end, text, label):
    return dict(start=start, end=end, text=text, type=label)


class TestSpanAssertions(unittest.TestCase):
    def setUp(self):
        self.text = "🪶Zoë (CPA)"
        self.positive = dict(kind="exact_span", start=1, end=4, text="Zoë", types=["PER"])
        self.negative = dict(kind="no_overlap", start=6, end=9, text="CPA", types=["ORG", "PER"])
        self.row = dict(id="unicode", text=self.text, assertions=[self.positive, self.negative])

    def score(self, entities):
        return oracle.score(oracle.load_cases([self.row]), [dict(id="unicode", entities=entities)])

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
        predictions = [dict(id="tag", entities=[hashtag])]
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

    def test_replay_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps([self.row]))
            saved = root / "saved.jsonl"
            cmd = [sys.executable, oracle.__file__, "--fixture", str(fixture), "--predictions", str(saved)]
            for entities, code in [([entity(1, 4, "Zoë", "PER")], 0), ([], 1),
                                   ([entity(2, 5, "Zoë", "PER")], 2)]:
                saved.write_text(json.dumps(dict(id="unicode", entities=entities)) + "\n")
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                if code < 2:
                    self.assertEqual(json.loads(result.stdout)["total"], 2)
