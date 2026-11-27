"""Paired controls for the decision to keep the existing possessive-name rule."""
import unittest

from native_gap_possessive_splits import compare, follow_possessive_pairs


class PossessiveSplitTests(unittest.TestCase):
    def test_proposal_gain_without_a_link_and_offsetting_correct_links(self):
        rows = [{"A-coref": "TRUE", "B-coref": "FALSE"}] * 3
        frozen = {"gold_links": 3, "tp": 1, "fp": 0}
        stock = {"counts": frozen, "per_id": ["11110", "10100", "10100"]}
        trial = {"counts": dict(frozen, fp=1), "per_id": ["11101", "11110", "11100"]}
        measured = compare(rows, (stock, ["111", "101", "101"]), (trial, ["111"] * 3), frozen)
        self.assertEqual(measured["paired_changes"], {
            "gold_proposals_gained": 2, "gold_proposals_lost": 0, "gained_proposals_linked": 1,
            "correct_links_gained": 1, "correct_links_lost": 1,
            "wrong_links_gained": 1, "wrong_links_lost": 0,
        })
        self.assertFalse(measured["within_stock_wrong_link_budget"])

    def test_follow_only_original_possessive_misses(self):
        old = [
            {"id": "a", "candidate": "A", "name_boundary_observation": "including_possessive_clitic"},
            {"id": "b", "candidate": "B", "name_boundary_observation": "multiword_only_subspans"},
        ]
        new = [
            {"id": "a", "candidate": "A", "first_loss_or_link": "cluster_emission",
             "direct_pair": "at_or_below_threshold"},
            {"id": "b", "candidate": "B", "first_loss_or_link": "linked", "direct_pair": "selected"},
        ]
        self.assertEqual(follow_possessive_pairs(old, new), {
            "candidate_outcome": {"cluster_emission": 1},
            "candidate_direct_pair_when_retained": {"at_or_below_threshold": 1},
        })
        with self.assertRaisesRegex(ValueError, "different labeled pair"):
            follow_possessive_pairs(old, new[::-1])
