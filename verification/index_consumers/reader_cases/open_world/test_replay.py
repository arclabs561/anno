# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest

from .page_control import page_control, page_fold
from .replay import evaluate, unique_rows


class OpenWorldEvidence(unittest.TestCase):
    def test_page_control_separates_same_page_and_distinguishes_window_pooling(self):
        rows = [
            (0, 1, ("blue", "Zal", "Q1", False)), (0, 2, ("red", "Zál", "Q1", False)),
            (0, 3, ("blue", "Own", "Q2", False)), (0, 4, ("blue", "Own", "Q2", False)),
            (20, 5, ("blue", "Solo", "Q8", False)), (20, 6, ("red", "Own", "Q2", False)),
            (20, 7, ("red", "Solo", "Q9", True)),
        ]
        self.assertEqual((page_fold("blue"), page_fold("red")), (0, 1))
        result = page_control(rows)
        own = result["within_window"]["by_label"]
        self.assertEqual(own["reachable"], {"exact": {"no_hit": 2}, "fold": {"correct": 2}})
        self.assertEqual(own["sampled_out"]["exact"], {"no_hit": 4})
        self.assertEqual(own["true_nil"]["fold"], {"wrong": 1})
        pooled = result["pooled_windows"]["by_label"]
        self.assertEqual(pooled["reachable"]["exact"], {"no_hit": 2, "correct": 3})
        self.assertEqual(pooled["reachable"]["fold"], {"correct": 5})
        with self.assertRaises(ValueError):
            page_control([*rows, rows[0]])

    def test_known_unsampled_and_nil_are_different_decisions(self):
        train = [("a", "Selo", "Q1", False), ("b", "Sélo", "Q2", False),
                 ("a", "Ar", "Q3", False), ("a", "Private", "Q7", True)]
        test = [("c", "Ar", "Q3", False), ("d", "Ar", "Q5", False),
                ("e", "Ar", "Q8", True), ("e", "Ar", "Q8", True),
                ("f", "Ár", "Q8", True), ("g", "Sélo", "Q2", False),
                ("a", "Ar", "Q3", False), ("h", "", "Q4", False)]
        result = evaluate(train, test)
        self.assertEqual(result["excluded_records_from_shared_pages"], 1)
        exact, folded = (result["policies"][k] for k in ("exact", "fold"))
        self.assertEqual(exact["links"], {"id_sampled.correct": 2, "id_unsampled.wrong": 1,
                         "id_unsampled.empty": 1, "nil.wrong": 2, "nil.empty": 1})
        self.assertEqual(folded["links"]["id_sampled.ambiguous"], 1)
        self.assertEqual(folded["nil_pairs"], {"gold": 2, "correct": 2, "wrong": 0, "missed": 0})
        self.assertEqual(exact["nil_pairs"]["missed"], 2)
        # A new entity with the identical printed spelling creates wrong cross-page pairs.
        other = evaluate(train, [*test, ("i", "Ar", "Q9", True)])
        self.assertEqual(other["policies"]["fold"]["nil_pairs"]["wrong"], 3)

    def test_observation_id_conflict(self):
        row = ("a", "Ar", "Q1", False)
        self.assertEqual(unique_rows([(1, row), (1, row)]), ([row], 1))
        with self.assertRaises(ValueError):
            unique_rows([(1, row), (1, ("a", "Ar", "Q2", False))])
