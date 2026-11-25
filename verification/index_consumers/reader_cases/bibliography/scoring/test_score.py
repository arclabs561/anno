# SPDX-License-Identifier: MIT OR Apache-2.0
"""Original examples. No publisher holdout or key is used in the unit tests."""
from copy import deepcopy
from unittest import TestCase

from .common import fingerprint
from .predict import generate
from .score import score


def author(name):
    return [[["surname", name]]]


def source(version, day, name):
    return {"edition": version, "published_on": day, "byline": author(name),
            "source_url": f"https://example.org/example/v{version}",
            "source_sha256": "0" * 64, "license": "https://creativecommons.org/licenses/by/4.0/"}


class BibliographyScoringTest(TestCase):
    def setUp(self):
        self.frame = [{"notice": "17", "target": "2", "day": "2022-03-01"}]
        self.work = {"doi": "10.7554/eLife.2", "through": "2022-04-01",
                     "editions": [source(1, "2022-01-01", "Novak"), source(2, "2022-02-01", "Novák"),
                                  source(3, "2022-03-01", "Novák"), source(4, "2022-04-01", "Team")]}
        self.gold = []
        for cut, edition, name in [("first", 1, "Novak"), ("pre", 2, "Novák"), ("post", 3, "Novák")]:
            self.gold.append({"notice": "17", "cut": cut, "status": "verified", "edition": edition,
                              "byline": fingerprint(author(name)), "printed_byline": fingerprint([name]),
                              "sources": [{"url": "https://example.org/example"}]})

    def test_latest_postdates_notice_and_delivery_agrees(self):
        predictions = generate(self.frame, [self.work])
        summary = score(predictions, self.gold)
        self.assertEqual(summary["delivery_pairs"], 6)
        self.assertEqual(summary["delivery_mismatches"], [])
        arms = summary["deliveries"]["bulk"]["arms"]
        self.assertEqual(arms["dated"]["all"]["printed_byline"]["correct"], 3)
        self.assertEqual(arms["mutable"]["all"]["printed_byline"]["wrong"], 3)
        self.assertEqual(arms["mutable"]["all"]["byline"]["works_with_wrong_answer"], 1)
        self.assertEqual(summary["deliveries"]["bulk"]["notice_changes"]["byline"]["changed_at_notice"], 0)

    def test_unknown_reference_stays_in_coverage_and_does_not_score(self):
        predictions = generate(self.frame, [self.work])
        self.gold[0].update(status="partial", byline=None, printed_byline=None)
        self.gold[1].update(status="unavailable", edition=None, byline=None, printed_byline=None)
        stats = score(predictions, self.gold)["deliveries"]["online"]["arms"]["dated"]["all"]
        self.assertEqual((stats["selected"], stats["answered"]), (3, 3))
        self.assertEqual(stats["edition"]["reference_known"], 2)
        self.assertEqual(stats["byline"]["reference_known"], 1)
        self.assertEqual(stats["byline"]["unjudged_answered"], 2)

    def test_missing_current_body_abstains_even_when_earlier_is_supported(self):
        self.work["editions"][-1].pop("byline")
        stats = score(generate(self.frame, [self.work]), self.gold)["deliveries"]["bulk"]["arms"]
        self.assertEqual(stats["dated"]["all"]["answered"], 3)
        self.assertEqual(stats["mutable"]["all"]["printed_byline"]["abstained_known"], 3)

    def test_validation_rejects_missing_rows_and_detects_evidence_delivery_mismatch(self):
        predictions = generate(self.frame, [self.work])
        with self.assertRaisesRegex(ValueError, "missing/unexpected"):
            score(predictions, self.gold[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score(predictions, self.gold + [self.gold[0]])
        different = deepcopy(predictions)
        different["rows"][0]["evidence"]["source_url"] = "https://example.org/wrong"
        self.assertEqual(len(score(different, self.gold)["delivery_mismatches"]), 1)
