# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest

from fcoref_optional_highlight import outcome


class OptionalHighlightTests(unittest.TestCase):
    def test_exact_name_can_still_misidentify_a_phrase(self):
        name = (0, 9)
        phrase = (0, 20)
        self.assertEqual(outcome(phrase, name, {name}, "same"), "exact_highlight")
        self.assertEqual(outcome(phrase, name, {name}, "different"), "unsafe_emission")
        self.assertEqual(outcome(phrase, name, {name}, "uncertain"), "unassessed_emission")

    def test_only_one_contained_occurrence_can_supply_a_highlight(self):
        phrase, name, other = (2, 14), (2, 5), (10, 14)
        self.assertEqual(outcome(phrase, name, {other}, "same"), "other_span")
        self.assertEqual(outcome(phrase, name, {name, other}, "same"), "abstained")
        self.assertEqual(outcome(phrase, name, {(20, 23)}, "same"), "abstained")
