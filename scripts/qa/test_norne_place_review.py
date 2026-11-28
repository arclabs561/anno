"""Check routing with no gold, a punctuation boundary and a match inside a word."""

import json
import unittest

from norne_place_review import ROOT, SPLITS, measure, tradeoff
from norne_source_replay import Observation, Sentence


class PlaceReviewTest(unittest.TestCase):
    def test_flags_require_the_exact_previous_word_and_do_not_require_gold(self):
        text = "I Øv fra, Zed fra Vox"
        words = ["I", "Øv", "fra", ",", "Zed", "fra", "Vox"]
        case = Sentence(
            "one",
            text,
            "",
            [(w, "O") for w in words],
            [(2, 4, "LOC"), (10, 13, "PER")],
            [],
            [],
        )
        spans = [
            (2, 4, "PER"),
            (10, 13, "PER"),
            (18, 21, "PER"),
            (19, 21, "PER"),
            (2, 4, "LOC"),
        ]
        queues = measure([case], [Observation(spans, set())])
        self.assertEqual(queues["per_start_inside_token"], 1)
        self.assertEqual(queues["all"]["suggestions"], 4)
        self.assertEqual(queues["all"]["outcomes"]["PER"], 1)
        flags = queues["flagged"]
        self.assertEqual(
            (flags["suggestions"], flags["exact_places"], flags["outcomes"]["no_overlap"]),
            (2, 1, 1),
        )

        case.source_gold = []
        empty = measure([case], [Observation(spans, set())])
        self.assertEqual(empty["sentences_without_gold"], 1)
        self.assertEqual(empty["flagged"]["outcomes"]["no_overlap"], 2)

    def test_all_published_queues_use_the_original_splits_and_actual_per_counts(self):
        docs = ROOT / "docs/eval"
        panels = json.loads((docs / "norne-place-review.json").read_bytes())["splits"]
        replay = json.loads((docs / "norne-source-replay.json").read_bytes())["splits"]
        self.assertEqual(set(panels), set(SPLITS))
        for key, row in panels.items():
            native = replay[key]
            self.assertEqual(row["source_sha256"], native["source_sha256"])
            self.assertEqual(
                row["ordered_predictions_sha256"], native["after"]["ordered_predictions_sha256"]
            )
            self.assertEqual(row["all"]["suggestions"], native["after_predictions_by_type"]["PER"])
            for arm in ("all", "flagged"):
                groups = row[arm]["outcomes"]
                self.assertEqual(row[arm]["suggestions"], sum(groups.values()))
                self.assertEqual(row[arm]["exact_places"], groups["LOC"] + groups["GPE_LOC"])
            for group, count in row["flagged"]["outcomes"].items():
                self.assertLessEqual(count, row["all"]["outcomes"][group])

    def test_chart_uses_each_test_queue_and_all_reviewable_places(self):
        docs = ROOT / "docs/eval"
        receipt = json.loads((docs / "norne-place-review.json").read_bytes())
        chart = json.loads((docs / "norne-place-tradeoff.json").read_bytes())
        self.assertEqual(chart, tradeoff(receipt))
        self.assertEqual(chart["x_axis"], "suggestions")
        for series, key in zip(chart["series"], ("nob-test", "nno-test"), strict=True):
            all_per = receipt["splits"][key]["all"]
            self.assertEqual(series["observations"], all_per["suggestions"])
            self.assertEqual(series["gold"], all_per["exact_places"])
            for point, arm in zip(series["points"], ("flagged", "all"), strict=True):
                queue = receipt["splits"][key][arm]
                self.assertEqual(point["correct"], queue["exact_places"])
                self.assertEqual(point["unmatched"] + point["correct"], queue["suggestions"])
        receipt["splits"]["nob-test"]["flagged"]["exact_places"] += 1
        with self.assertRaisesRegex(ValueError, "Review counts disagree"):
            tradeoff(receipt)
