"""Authored controls for exact, nested, partial and unassessed name boundaries."""

import unittest

from litbank_name_boundaries import Document, classify, compare, inventory


class NameBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document("control", "Zoë Day near Elm", {(0, 7), (13, 16)},
                            {(0, 7), (13, 16), (0, 12)})

    def test_added_proposal_preserves_raw_parts_but_can_displace_retained_name(self):
        stock = {"id": "control", "proposed": [[13, 16], [0, 3]], "retained": [[13, 16], [0, 3]]}
        trial = {"id": "control", "proposed": stock["proposed"] + [[0, 7], [8, 16]],
                 "retained": [[0, 7], [8, 16]]}
        result = compare([self.doc], [stock], [trial])
        raw = result["proposed"]
        self.assertEqual(raw["counts"]["gained_exact_multiword"], 1)
        self.assertEqual(raw["counts"]["lost_exact_one_word"], 0)
        self.assertEqual(raw["removed"], {})
        self.assertEqual(raw["added"]["contains_proper_person_without_matching_annotated_boundary"], 1)
        self.assertEqual(result["retained"]["counts"]["lost_exact_one_word"], 1)

    def test_valid_nested_boundary_precedes_enclosure_and_overlap_is_distinct(self):
        self.assertEqual(classify((0, 12), self.doc), "exact_other_annotated")
        self.assertEqual(classify((0, 3), self.doc), "other_overlap_with_proper_person")
        self.assertEqual(classify((8, 12), self.doc), "unassessed")

    def test_trace_ids_and_scalar_bounds_are_checked(self):
        for trace in ({"id": "other", "proposed": []}, {"id": "control", "proposed": [[0, 17]]}):
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                inventory(self.doc, trace, "proposed")


if __name__ == "__main__":
    unittest.main()
