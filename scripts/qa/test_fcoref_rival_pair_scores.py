# SPDX-License-Identifier: MIT OR Apache-2.0
"""Authored controls for score decomposition, native order and null ties."""
import unittest

from fcoref_rival_pair_scores import edge, retained_indices


class PairScores(unittest.TestCase):
    def setUp(self):
        self.text = "Zoë met Ada"
        self.offsets = [(0, 3), (4, 7), (8, 11)]
        self.table = [
            (0, 0, 1, 0, []),
            (1, 1, -1, 0, [(0, 0)]),
            (2, 2, -1, 2, [(-2, -2), (1, -1)]),
        ]

    def test_direction_and_null_in_source_order(self):
        idx = retained_indices(self.table, self.text, self.offsets)
        older, later = (0, 3), (8, 11)
        for pronoun, name, query in ((older, later, 'name'), (later, older, 'pronoun')):
            result = edge(self.table, idx, pronoun, name)
            self.assertEqual((result["query"], result["winner"], result["total"]),
                             (query, "null", -2))
            self.assertEqual((result["winning_score"], result["best_real"]), (0, -1))
        tie = edge(self.table, idx, older, (4, 7))
        self.assertEqual((tie["winner"], tie["total"]), ("direct", 0))

    def test_wrong_component_and_reversed_null_tie_fail(self):
        for row in ((1, 1, -1, 1, [(0, 0)]), (1, 1, -1, 0, [(7, 0)])):
            with self.subTest(row=row), self.assertRaises(ValueError):
                retained_indices([self.table[0], row], self.text, self.offsets)

    def test_best_token_realization_of_trimmed_surface(self):
        offsets = [(0, 0), (0, 3), (3, 3)]
        table = [(0, 1, -1, 0, []), (1, 1, 2, 0, [(0, 1)])]
        self.assertEqual(retained_indices(table, "Zoë", offsets), {(0, 3): 1})
