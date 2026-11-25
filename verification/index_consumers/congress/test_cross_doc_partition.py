# SPDX-License-Identifier: MIT OR Apache-2.0
"""Controls for the cross-document pair denominator and the dated reader policy."""
import unittest

from cross_doc_partition import Observation, evaluate, reader_metrics, resolve_chamber_conflict


def score(placements, latest):
    """Each occurrence supplies its withheld person ID and its assigned track group."""
    docs = {doc: [Observation("Vale", person, "venue") for person, _ in rows]
            for doc, rows in placements.items()}
    predictions = {(doc, i): group for doc, rows in placements.items()
                   for i, (_, group) in enumerate(rows)}
    return reader_metrics(docs, predictions, latest)


class CrossDocumentControls(unittest.TestCase):
    def test_prior_lookup_requires_distinct_dates(self):
        lookup = score({
            "2021-04-02/bulletin/Larch": [("a", 1), ("b", 1), ("c", 2)],
            "2021-04-02/bulletin/Silt": [("c", 2)],
            "2022-04-02/bulletin/Larch": [("b", 1), ("d", 3)],
            "2023-04-02/bulletin/Silt": [("a", 1), ("e", 2), ("c", 2), ("d", 4)],
            "2024-04-02/bulletin/Larch": [("a", 1), ("e", 2)],
        }, "2023-04-02")["prior_lookup"]
        self.assertEqual(lookup["queries"], 4)
        self.assertEqual(lookup["queries_with_history"], 3)  # e first appears at the query date.
        # Group 1 repeats the same mistaken b on two dates. Group 2 has two
        # documents on one date. Its two true hits for c disappear at the higher floor.
        self.assertEqual(lookup["minimum_dates"], {
            "1": {"shown_pages": 3, "pages_with_wrong_hits": 2,
                  "correct_hits": 3, "wrong_hits": 4, "missed_hits": 1},
            "2": {"shown_pages": 1, "pages_with_wrong_hits": 1,
                  "correct_hits": 1, "wrong_hits": 2, "missed_hits": 3},
        })

    def test_b3_counts_same_document_occurrences_and_singletons(self):
        result = score({
            "2021-04-02/bulletin/Larch": [("a", 1), ("b", 1)],
            "2022-04-02/bulletin/Silt": [("a", 1), ("c", 2)],
        }, "2022-04-02")["b3"]
        # Per-occurrence precisions are 2/3, 1/3, 2/3 and 1.
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1)
        self.assertAlmostEqual(result["f1"], 4 / 5)
        empty = score({}, "2022-04-02")
        self.assertEqual(empty["b3"], {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        self.assertEqual(empty["prior_lookup"]["queries"], 0)

    def test_same_document_pairs_are_excluded(self):
        rows = {
            "a": [Observation("same", "one", "rep"), Observation("same", "two", "rep")],
            "b": [Observation("variant", "one", "rep"), Observation("new", "three", "rep")],
        }
        duplicate = {("a", 0): 1, ("a", 1): 1, ("b", 0): 2, ("b", 1): 1}
        counts = evaluate(rows, duplicate)
        self.assertEqual([counts[k] for k in ("cross_document_pairs", "tp", "fp", "fn", "tn")],
                         [4, 0, 2, 1, 1])

    def test_roles_can_change_between_cuts(self):
        a, b, later = "old/roster/rep", "old/roster/sen", "new/roster/sen"
        rows = {a: [Observation("N", "1", "rep")], later: [Observation("N", "1", "sen")]}
        groups = {(a, 0): 1, (later, 0): 1}
        changed_role = resolve_chamber_conflict(rows, groups)
        self.assertEqual(changed_role[a, 0], changed_role[later, 0])
        rows[b] = [Observation("N", "2", "sen")]
        groups[b, 0] = 1
        conflict = resolve_chamber_conflict(rows, groups)
        self.assertNotEqual(conflict[a, 0], conflict[b, 0])
        # A later role transition and an overlapping namesake can occur together.
        # Dividing the entire historical cluster by chamber would misgroup this future row.
        self.assertEqual(conflict[b, 0], conflict[later, 0])


if __name__ == "__main__":
    unittest.main()
