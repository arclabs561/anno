# SPDX-License-Identifier: MIT OR Apache-2.0
"""Controls for the cross-document pair denominator and the dated reader policy."""
import unittest

from cross_doc_partition import Observation, evaluate, resolve_chamber_conflict


class CrossDocumentControls(unittest.TestCase):
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
