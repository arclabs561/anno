# SPDX-License-Identifier: MIT OR Apache-2.0
import copy
import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent / "joint_baseline"))
from jerex_controls import ARMS, paired_intervals, partition_detected, rescore, summarize
from jerex_revised_reference import FILES, REVISION as REVISED_REVISION, classify
from jerex_stages import CHECKPOINT, REVISION, SPLIT


class JerexControlsTests(unittest.TestCase):
    def test_public_followups_contain_aggregate_numbers_and_pins(self):
        root = Path(__file__).parent / "joint_baseline" / "receipts" / "jerex_controls"
        review = json.loads((root / "revised_reference.json").read_bytes())
        self.assertEqual(set(review), {"source_sha256", "revised_revision", "documents_by_revised_split",
                                      "aligned_predictions", "outcomes"})
        self.assertEqual((review["source_sha256"], review["revised_revision"]), (SPLIT, REVISED_REVISION))
        self.assertEqual(set(review["documents_by_revised_split"]), set(FILES))
        self.assertEqual(sum(review["documents_by_revised_split"].values()), 32)
        self.assertEqual(review["outcomes"], {"exact_directed_predicate_added": 65,
            "another_predicate_on_revised_pair": 2, "directed_pair_still_unannotated": 14})
        self.assertEqual(sum(review["outcomes"].values()), review["aligned_predictions"])
        expected_keys = {"source_sha256", "model_sha256", "revision", "document_start", "documents",
                         "reference_threshold", "arms", "paired_f1_difference_95pct_points", "revised_reference"}
        for start in (0, 32):
            report = json.loads((root / f"docred_{start}_{start + 31}.json").read_bytes())
            self.assertEqual(set(report), expected_keys)
            self.assertEqual((report["source_sha256"], report["model_sha256"], report["revision"]),
                             (SPLIT, CHECKPOINT["model.ckpt"], REVISION))
            self.assertEqual((report["document_start"], report["documents"], report["reference_threshold"]),
                             (start, 32, 0.6))
            if start == 0:
                stock = report["arms"][ARMS[0]]["0.6"]
                expected = json.loads((root.parent / "jerex_docred32.json").read_bytes())["stages"]["labeled_relations"]
                self.assertEqual((stock["gold"], stock["matched"] + stock["extras"], stock["matched"]),
                                 (expected["gold"], expected["predicted"], expected["matched"]))
            revised = report["revised_reference"]
            self.assertEqual(set(revised), {"revision", "arms", "paired_f1_difference_95pct_points"})
            self.assertEqual(revised["revision"], REVISED_REVISION)
            for edition in (report, revised):
                baseline = edition["arms"][ARMS[0]]["0.6"]
                self.assertEqual(set(edition["arms"]), set(ARMS))
                self.assertEqual(set(edition["paired_f1_difference_95pct_points"]), set(ARMS))
                for arm, thresholds in edition["arms"].items():
                    self.assertEqual(set(thresholds), {"0.25", "0.5", "0.6", "0.75"})
                    intervals = edition["paired_f1_difference_95pct_points"][arm]
                    self.assertEqual(set(intervals), set(thresholds))
                    for threshold, counts in thresholds.items():
                        self.assertEqual(set(counts), {"gold", "matched", "missed", "extras", "gained", "lost"})
                        self.assertTrue(all(type(v) is int and v >= 0 for v in counts.values()))
                        self.assertEqual(counts["gold"], counts["matched"] + counts["missed"])
                        self.assertEqual(counts["matched"], baseline["matched"] + counts["gained"] - counts["lost"])
                        low, high = intervals[threshold]
                        self.assertTrue(-100 <= low <= high <= 100)

    def test_partition_keeps_false_detections_and_cannot_add_missing_mentions(self):
        spans = {2: (0, 1), 5: (3, 4), 7: (5, 6)}
        gold = [{(0, 1), (3, 4), (8, 9)}, {(4, 5)}]
        self.assertEqual(partition_detected(spans, gold), [[2, 5], [7]])
        with self.assertRaisesRegex(ValueError, "multiple reference"):
            partition_detected(spans, [*gold, {(3, 4)}])

    def test_paired_gains_and_losses_are_scored_against_production_decisions(self):
        gold = {"one", "two"}
        base = ({(0, 1)}, gold, {0.5: ({"one"}, set())}, 0.5)
        changed = ({(0, 1)}, gold, {0.5: ({"two"}, {"new"})}, 0.5)
        totals = summarize([base, changed])
        self.assertEqual(totals[ARMS[1]]["0.5"], dict(gold=2, matched=1, extras=1, missed=1, gained=1, lost=1))
        with self.assertRaisesRegex(ValueError, "mention detections"):
            summarize([base, (set(), *changed[1:])])

    def test_revised_relabels_an_extra_without_changing_a_prediction(self):
        original = {"title": "Example", "sents": [["Ada", "met", "Bea"]],
                    "vertexSet": [[{"sent_id": 0, "pos": [0, 1]}], [{"sent_id": 0, "pos": [2, 3]}]],
                    "labels": []}
        revised = copy.deepcopy(original)
        revised["labels"].append({"h": 0, "r": "met", "t": 1})
        edge = (frozenset({(0, 1)}), "met", frozenset({(2, 3)}))
        base = ({(0, 1), (2, 3)}, set(), {0.5: (set(), {edge})}, 0.5)
        candidate = ({(0, 1), (2, 3)}, set(), {0.5: (set(), set())}, 0.5)
        runs = rescore([base, candidate], [original], {"Example": ("dev", revised)})
        self.assertEqual(runs[0][2][0.5], ({edge}, set()))
        self.assertEqual(runs[1][2][0.5], (set(), set()))
        self.assertEqual(paired_intervals(runs)[ARMS[1]]["0.5"], [-100.0, -100.0])

    def test_revised_reference_joins_cluster_membership_and_direction(self):
        original = {"title": "Example", "sents": [["Ada", "met", "Bea"]],
                    "vertexSet": [[{"sent_id": 0, "pos": [0, 1]}], [{"sent_id": 0, "pos": [2, 3]}]],
                    "labels": []}
        revised = copy.deepcopy(original)
        revised["vertexSet"].reverse()
        edge = {"predicate": "met", "head_alignment": {"exact": [0]}, "tail_alignment": {"exact": [1]},
                "outcome": "directed_pair_unannotated"}
        revised["labels"] = [{"h": 1, "t": 0, "r": "met"}]
        self.assertEqual(classify(original, [edge], revised), {"exact_directed_predicate_added": 1})
        revised["labels"] = [{"h": 0, "t": 1, "r": "met"}]
        self.assertEqual(classify(original, [edge], revised), {"directed_pair_still_unannotated": 1})
        revised["vertexSet"][1].append({"sent_id": 0, "pos": [1, 2]})
        self.assertEqual(classify(original, [edge], revised), {"endpoint_membership_changed_or_ambiguous": 1})
        revised["sents"] = [["Ada", "saw", "Bea"]]
        with self.assertRaisesRegex(ValueError, "different title or source"):
            classify(original, [edge], revised)


if __name__ == "__main__":
    unittest.main()
