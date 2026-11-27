# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest
from collections import Counter

from relex_roles import classify, gold, suppressed_edge_scores


class RelexTests(unittest.TestCase):
    def test_directed_occurrences_and_duplicate_multiplicity(self):
        doc = {"sentence": ["李", "met", "李"], "relations": [[0, 0, 2, 2, "met"]] * 2}
        a, b = (0, 1), (6, 7)
        reference = gold(doc)
        self.assertEqual(reference, Counter({(a, "met", b): 2}))
        graph = Counter({(a, "met", b): 1})
        self.assertEqual(classify(reference, graph, ["met"], {a, b}, {a, b}, {(a, b)}, {a, b}),
                         {"duplicate_reference_occurrence": 1})

    def test_mention_pair_and_predicate_loss_are_separate(self):
        a, b = (0, 1), (4, 5)
        reference = Counter({(a, "met", b): 1, (b, "met", a): 1})
        both = {a, b}
        def misses(proposed, pairs, kept):
            return classify(reference, Counter(), ["met"], both, proposed, pairs, kept)
        self.assertEqual(misses({b}, set(), {b}), {"below_mention_threshold_head": 1, "below_mention_threshold_tail": 1})
        self.assertEqual(misses(both, {(a, b)}, both), {"ordered_pair_not_generated": 1, "predicate_below_threshold": 1})
        self.assertEqual(misses(both, {(a, b), (b, a)}, {a}), {"flat_overlap_suppression": 2})

    def test_suppressed_edges_require_disjoint_ends_and_the_directed_predicate(self):
        a, b, nested, other = (0, 2), (2, 5), (1, 2), (5, 7)
        reference = Counter({
            (a, "left", b): 2, (b, "left", a): 1, (b, "right", a): 1,
            (a, "left", nested): 1, (nested, "left", a): 1,
            (b, "left", other): 1, (a, "left", other): 1,
            (a, "missing", b): 1, (other, "left", b): 1,
        })
        scores = {(a, b): [0.51, 0.1], (b, a): [0.5, 0.8],
                  (a, nested): [0.9, 0.2], (nested, a): [0.2, 0.9],
                  (b, other): [float("nan"), 0.9], (a, other): [0.8, 0.8]}
        counts = suppressed_edge_scores(reference, {a, other}, ["left", "right"], scores, 0.5)
        self.assertEqual(counts, {
            "recoverable_distinct_edges": 2,
            "duplicate_occurrences_of_recoverable_edges": 1,
            "disjoint_endpoints_predicate_at_or_below_gate": 1,
            "overlapping_endpoints_predicate_above_gate": 1,
            "overlapping_endpoints_predicate_at_or_below_gate": 1,
            "unavailable_predicate_score": 1,
        })


if __name__ == "__main__":
    unittest.main()
