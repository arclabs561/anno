"""Check reconstruction and abstention when the native candidate order matters."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from native_gap_link_retention import Decision, Observation, chart, select, validate_trace


class NativeMarginTest(unittest.TestCase):
    def test_rival_includes_subthreshold_scores_and_one_candidate_has_no_rival(self):
        two = Decision((8, 11), "pronominal", 0.225, [(1, 0.21), (0, 0.24)], 0)
        self.assertIsNone(two.parent("rival", 0.04))
        self.assertEqual(two.parent("rival", 0.025), 0)
        one = Decision((8, 11), "pronominal", 0.225, [(0, 0.24)], 0)
        self.assertEqual(one.parent("rival", 0.30), 0)
        self.assertIsNone(one.parent("threshold", 0.025))
        proper = Decision((8, 11), "proper", 0.45, [(1, 0.46), (0, 0.46)], 1)
        self.assertEqual(proper.parent("rival", 0.30), 1)

    def test_removing_direct_edge_preserves_unaffected_incoming_and_transitive_links(
        self,
    ):
        obs = Observation(
            [
                Decision((0, 3), "proper", 0.45, [], None),
                Decision((4, 7), "pronominal", 0.225, [(0, 0.23)], 0),
                Decision((8, 11), "proper", 0.45, [(1, 0.75), (0, 0.2)], 1),
            ],
            (1, 0, 2),
        )
        self.assertEqual(obs.mask(), "11111")
        self.assertEqual(obs.mask("threshold", 0.1), "11101")

    def test_zero_margin_reconciles_native_ties_and_rejects_corrupted_trace(self):
        row = {
            "ID": "t",
            "Text": "Ana & Ada say she",
            "Pronoun": "she",
            "Pronoun-offset": 14,
            "A": "Ana",
            "A-offset": 0,
            "B": "Ada",
            "B-offset": 6,
        }
        labels = [(0, 3), (6, 9), (14, 17)]
        scored = [[], [[0, 0.45]], [[1, 0.5], [0, 0.5]]]
        recs = [
            dict(
                index=i,
                span=span,
                mention_type="pronominal" if i == 2 else "proper",
                threshold=0.225 if i == 2 else 0.45,
                candidates=scores,
                selected=1 if i == 2 else None,
                max_distance=100,
            )
            for i, (span, scores) in enumerate(zip(labels, scored))
        ]
        output = {
            "id": "t",
            "producer": "mention-ranking",
            "clusters": [[labels[0]], [labels[1], labels[2]]],
        }
        obs, _ = validate_trace(row, recs, output)
        self.assertEqual(obs.mask(), "11101")
        wrong = deepcopy(recs)
        wrong[2]["selected"] = 0
        with self.assertRaisesRegex(ValueError, "winner"):
            validate_trace(row, wrong, output)
        with self.assertRaisesRegex(ValueError, "partitions"):
            validate_trace(row, recs, {**output, "clusters": [[s] for s in labels]})

    def test_choose_only_eligible_dev_point_and_prefer_more_true_links_at_equal_cost(
        self,
    ):
        grid = [
            dict(margin=0.0, tp=10, fp=10),
            dict(margin=0.05, tp=9, fp=3),
            dict(margin=0.1, tp=8, fp=3),
            dict(margin=0.2, tp=7, fp=0),
        ]
        self.assertEqual(select(grid), 0.05)

    def test_chart_and_classified_masks_reconcile_to_published_results(self):
        root = Path(__file__).resolve().parents[2] / "docs/eval"
        receipt = json.loads((root / "native-gap-link-retention.json").read_text())
        self.assertEqual(
            chart(receipt["panels"]),
            json.loads((root / "native-gap-margin-tradeoff.json").read_text()),
        )
        for group in receipt["panels"].values():
            neutral = group["operating_points"]["Neutral type"]
            for hashes in group["checked_native_margin_partitions_sha256"].values():
                self.assertEqual(len(hashes), len(receipt["checked_native_margin_grid"]))
                self.assertEqual(hashes[0], group["native_partitions_sha256"]["neutral"])
            self.assertEqual(
                sum(group["diagnostic"]["wrong_links_by_target_search"].values()),
                neutral["fp"],
            )
            for name, points in group["operating_points"].items():
                self.assertEqual(sum(points.values()), group["rows"] * 2)
                if name == "Neutral type":
                    continue
                delta = group["paired_vs_neutral"][name]
                self.assertEqual(
                    points["tp"] - neutral["tp"],
                    delta["correct_gained"] - delta["correct_lost"],
                )
                self.assertEqual(
                    points["fp"] - neutral["fp"],
                    delta["incorrect_added"] - delta["incorrect_removed"],
                )


if __name__ == "__main__":
    unittest.main()
