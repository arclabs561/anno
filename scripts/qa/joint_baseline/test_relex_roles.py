# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest
from collections import Counter

from relex_roles import classify, gold


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


if __name__ == "__main__":
    unittest.main()
