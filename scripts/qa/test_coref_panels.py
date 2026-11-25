"""Regression checks for targeted GAP decisions and LitBank corpus scoring."""

import math
import os
import random
import runpy
import unittest
from coref_gap import check_file, index_partition, measure
from coref_litbank import score_doc, summarize


class PanelScores(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("COVAL_EVALUATOR"), "optional pinned CoVal reference"
    )
    def test_cluster_counts_against_coval(self):
        path = os.environ["COVAL_EVALUATOR"]
        check_file(
            path, "bae143609c4648a6b68ac43266872a293794bc7c1c420abf7405eec5c07958ba"
        )
        reference = runpy.run_path(path)
        rng = random.Random(20261125)
        for _ in range(256):
            partitions = []
            for _side in range(2):
                clusters = {}
                for span in range(12):
                    if rng.random() < 0.8:
                        clusters.setdefault(rng.randrange(6), set()).add(
                            (span, span + 1)
                        )
                partitions.append(list(clusters.values()))
            gold, predicted = partitions
            actual = score_doc(gold, predicted)
            gold = [c for c in gold if len(c) > 1]
            predicted = [c for c in predicted if len(c) > 1]
            for metric, name in [("b3", "b_cubed"), ("muc", "muc")]:
                fn = reference[name]
                expected = (
                    *fn(predicted, index_partition(gold)),
                    *fn(gold, index_partition(predicted)),
                )
                for value, ref in zip(actual[metric], expected, strict=True):
                    self.assertTrue(math.isclose(value, ref, abs_tol=1e-10))
            for value, ref in zip(
                actual["ceafe"], reference["ceafe"](predicted, gold), strict=True
            ):
                self.assertTrue(math.isclose(value, ref, abs_tol=1e-10))

    def test_targeted_gap_decision_ignores_unlabeled_pairs(self):
        doc = ("t", "P A B X", [(0, 1), (2, 3), (4, 5)], [True, False])
        out = {"producer": "fixture", "id": "t", "clusters": [[(0, 1), (2, 3), (6, 7)]]}
        result = measure([doc], [out], "fixture")
        self.assertEqual(result["per_id"], ["11010"])
        self.assertEqual(result["counts"]["f1"], 1)

    def test_gap_rejects_a_mention_assigned_to_two_clusters(self):
        p, a, b = (0, 1), (2, 3), (4, 5)
        doc = ("t", "P A B", [p, a, b], [True, False])
        out = {"id": "t", "producer": "fixture", "clusters": [[p, a], [p, b]]}
        with self.assertRaisesRegex(ValueError, "overlapping cluster assignment"):
            measure([doc], [out], "fixture")

    def test_gap_supplied_refusal_keeps_attempt_outside_scored_confusion_matrix(self):
        doc = ("t", "P A B", [(0, 1), (2, 3), (4, 5)], [True, False])
        empty = {"id": "t", "producer": "maverick", "clusters": []}
        refused = {"id": "t", "producer": "maverick", "unsupported": "supplied_span_splits_token"}
        result = measure([doc, doc], [refused, empty], "maverick", "supplied")
        counts = result["counts"]
        self.assertEqual((counts["attempted_documents"], counts["documents"], counts["refused_documents"]), (2, 1, 1))
        self.assertEqual((counts["gold_links"], counts["refused_gold_links"], counts["fn"], counts["tn"]), (1, 1, 1, 1))
        self.assertEqual(result["per_id"], [{"unsupported": "supplied_span_splits_token"}, "00000"])
        for row, mode in [(refused, "predicted"), ({**refused, "clusters": []}, "supplied")]:
            with self.assertRaisesRegex(ValueError, "invalid refusal"):
                measure([doc], [row], "maverick", mode)

    def test_gap_with_no_gold_or_predicted_links(self):
        doc = ("t", "P A B", [(0, 1), (2, 3), (4, 5)], [False, False])
        out = {"id": "t", "producer": "fixture", "clusters": []}
        counts = measure([doc], [out], "fixture")["counts"]
        self.assertEqual((counts["tn"], counts["exact_decisions"]), (2, 1))
        for metric in ("precision", "recall", "f1"):
            self.assertIsNone(counts[metric])

    def test_missing_mention_and_singleton_have_distinct_denominators(self):
        a, b, c, d = ((i, i + 1) for i in range(4))
        result = score_doc([{a, b, c}, {d}], [{a, b}, {d}])
        self.assertEqual(result["mentions"], [3, 3, 4])
        self.assertEqual(result["linked_mentions"], [2, 2, 3])
        self.assertEqual(result["muc"], [1, 1, 1, 2])
        self.assertAlmostEqual(result["b3"][2], 4 / 3)
        self.assertAlmostEqual(result["ceafe"][0], 4 / 5)
        self.assertAlmostEqual(summarize(result)["b3"][1], 4 / 9, places=6)

    def test_identical_numeric_coordinates_in_separate_documents(self):
        a, b, c, d = ((i, i + 1) for i in range(4))
        one = score_doc([{a, b}], [{a, b}])
        two = score_doc([{a, b}], [{c, d}])
        combined = {k: [i + j for i, j in zip(one[k], two[k])] for k in one}
        self.assertEqual(summarize(combined)["conll_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
