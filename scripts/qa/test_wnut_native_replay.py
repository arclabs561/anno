"""Check the WNUT projection, exposure strata and native provenance contract."""

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wnut_native_replay import (
    ARMS,
    check_prediction,
    count_rows,
    first_cased_character,
    score_sentence,
)


class NativeWnutTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "figure requires Matplotlib"
    )
    def test_changed_predictions_with_unchanged_hits_still_change_release_headline(
        self,
    ):
        from matplotlib import pyplot as plt
        from plot_wnut_native import draw

        receipt = (
            Path(__file__).resolve().parents[2]
            / "docs/eval/wnut-native-comparison.json"
        )
        report = json.loads(receipt.read_text())
        pairs = report["paired"]["all"]
        pairs.update(false_positives_added=1, false_positives_removed=1)
        with self.assertRaisesRegex(ValueError, "Identical typed spans"):
            draw(report)
        pairs["changed_sentences"] = 1
        figure = draw(report)
        try:
            self.assertIn("Typed spans changed on 1 of", figure.texts[0].get_text())
        finally:
            plt.close(figure)

    def test_projected_span_changes_and_unseen_false_positives(self):
        by_arm = {r: defaultdict(Counter) for r in ARMS}
        paired = defaultdict(Counter)
        # The corporation «Org» is seen; the group is new. All categories remain in gold.
        text = "Org clan Gizmo Book"
        gold = {
            (0, 3, "corporation"),
            (4, 8, "group"),
            (9, 14, "product"),
            (15, 19, "creative-work"),
        }
        before = {(0, 3, "ORG"), (4, 6, "ORG"), (9, 14, "LOC")}
        after = {(4, 8, "ORG"), (15, 19, "WORK_OF_ART")}
        score_sentence(
            text, gold, dict(baseline=before, candidate=after), {"org"}, by_arm, paired
        )
        score_sentence(
            "Extra",
            set(),
            dict(baseline=set(), candidate={(0, 5, "PER")}),
            {"org"},
            by_arm,
            paired,
        )
        release, candidate = (count_rows(by_arm[r]) for r in ARMS)
        self.assertEqual(
            [
                release["unseen"][key]
                for key in ("exact", "wrong_type", "partial_overlap", "no_overlap")
            ],
            [0, 1, 1, 1],
        )
        self.assertEqual(candidate["all"]["false_positives"], 1)
        self.assertEqual(
            candidate["unseen"]["false_positives_in_gold_empty_sentences"], 1
        )
        self.assertEqual(candidate["unseen/group"]["exact"], 1)
        self.assertEqual(candidate["unseen/case-lower"]["exact"], 1)
        self.assertEqual(candidate["seen/corporation"]["no_overlap"], 1)
        self.assertEqual(paired["unseen"]["exact_gained"], 2)
        self.assertEqual(paired["seen"]["exact_lost"], 1)
        self.assertEqual(paired["unseen"]["false_positives_removed"], 2)
        self.assertEqual(paired["all"]["changed_sentences"], 2)

    def test_binds_emitted_span_and_producer_to_each_native_input(self):
        row = {
            "type": "LOC",
            "text": "Oslo",
            "start": 2,
            "end": 6,
            "confidence": 0.6,
            "provenance": {"source": "heuristic"},
        }
        prov = {
            "model": "heuristic",
            "entity_count": 1,
            "text_hash": "h",
            "text_chars": 6,
        }
        raw = {"entities": [row], "provenance": prov}
        self.assertEqual(
            check_prediction(raw, "🌿 Oslo", "h", False),
            ({(2, 6, "LOC")}, Counter(heuristic=1), "unreported"),
        )
        with self.assertRaisesRegex(ValueError, "backend"):
            check_prediction(raw, "🌿 Oslo", "h", True)
        prov["runtime_model"] = "heuristic"
        with self.assertRaisesRegex(ValueError, "Native input"):
            check_prediction(raw, "🌿 Oslo", "other", True)
        row["start"] = 3
        with self.assertRaisesRegex(ValueError, "Unicode"):
            check_prediction(raw, "🌿 Oslo", "h", True)
        row["start"] = 2
        row["provenance"]["source"] = "fallback"
        with self.assertRaisesRegex(ValueError, "per-entity producer"):
            check_prediction(raw, "🌿 Oslo", "h", True)

    def test_casing_uses_first_letter_with_upper_or_lower_case(self):
        self.assertEqual(first_cased_character("@9éX"), "lower")
        self.assertEqual(first_cased_character("!9Éx"), "upper")
        self.assertEqual(first_cased_character("99"), "uncased")

    def test_default_stack_requires_the_native_layers_and_runtime(self):
        row = dict(
            text="2020",
            type="DATE",
            start=0,
            end=4,
            confidence=0.95,
            provenance={"source": "pattern"},
        )
        prov = dict(
            model="stacked",
            runtime_model="stacked(regex+heuristic)",
            text_chars=4,
            entity_count=1,
            text_hash="h",
        )
        record = {"entities": [row], "provenance": prov}
        _, producers, runtime = check_prediction(record, "2020", "h", True, "stacked")
        self.assertEqual(producers, Counter(pattern=1))
        self.assertEqual(runtime, prov["runtime_model"])
        prov["runtime_model"] = "heuristic"
        with self.assertRaisesRegex(ValueError, "backend"):
            check_prediction(record, "2020", "h", True, "stacked")


if __name__ == "__main__":
    unittest.main()
