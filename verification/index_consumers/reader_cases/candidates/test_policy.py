# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest

from ..ajmc import Mention
from .evaluate import empty_counts, evaluate_page, outcomes
from .policy import CandidateIndex, one_edit, spelling


class PolicyTest(unittest.TestCase):
    def test_empty_native_gate_uses_visible_hits_including_unscored_ones(self):
        q, donor, native = ("q", 0), ("d", 0), ("d", 1)
        rows = {
            q: Mention("Groven", "Q1", False),
            donor: Mention("Grovon", "Q1", False),
            native: Mention("Groven", "NIL", False),
        }
        groups = {q: 1, native: 1, donor: 2}
        counts = empty_counts()
        evaluate_page("q", [donor, native], rows, groups, counts, {})
        self.assertEqual(counts["expanded"]["any_correct"], 1)
        self.assertEqual(counts["empty_native"]["any_correct"], 0)
        self.assertEqual(counts["empty_native"]["with_unscored_hit"], 1)
        counts = empty_counts()
        evaluate_page("q", [donor], rows, groups, counts, {})
        self.assertEqual(counts["empty_native"]["exact_supported"], 1)

    def test_native_matches_are_outside_the_supplemental_ambiguity_gate(self):
        both = CandidateIndex({"base": ("Téra", 4), "other": ("Tera", 7)})
        extra = both.suggest("Téra", 4, "expanded")
        self.assertEqual((extra.tier, extra.hits), ("fold", {"other"}))
        base_only = CandidateIndex({"base": ("Groven", 4), "other": ("Grovon", 7)})
        extra = base_only.suggest("Groven", 4, "expanded")
        self.assertEqual((extra.tier, extra.hits), ("edit", {"other"}))

    def test_fold_preserves_occurrence_and_omits_own_group(self):
        gallery = CandidateIndex(
            {"already": ("ÉLÈNE", 1), "a": ("Elène", 2), "b": ("ELENE", 2)}
        )
        self.assertEqual(gallery.suggest("élène", 1, "fold").hits, {"a", "b"})
        self.assertEqual(gallery.suggest("élène", 1, "native").hits, set())
        self.assertNotEqual(spelling("ΐ")[0], spelling("ι")[0])

    def test_ambiguity_at_abbreviation_tier_stops_edit_fallback(self):
        index = CandidateIndex(
            {"a": ("Norenova", 2), "b": ("Norenton", 3), "c": ("Norel", 4)}
        )
        decision = index.suggest("Noren.", 1, "expanded")
        self.assertEqual(
            (decision.tier, decision.decision, decision.hits),
            ("abbreviation", "ambiguous", set()),
        )
        self.assertEqual(index.suggest("Noren", 1, "expanded").hits, {"c"})

    def test_edit_is_bounded_and_no_transitive_expansion(self):
        for left, right in [
            ("brenna", "brena"),
            ("brenna", "brinna"),
            ("brenna", "brennan"),
        ]:
            self.assertTrue(one_edit(left, right))
            self.assertTrue(one_edit(right, left))
        self.assertFalse(one_edit("brinna", "birnna"))
        index = CandidateIndex({"a": ("Fennar", 2), "b": ("Fennor", 2)})
        self.assertEqual(index.suggest("Fennan", 1, "expanded").hits, {"a"})
        self.assertEqual(index.suggest("Όλγα", 1, "expanded").hits, set())

    def test_unscored_extras_and_page_errors(self):
        a, b, c, d = (("p", 0), ("p", 1), ("r", 0), ("s", 0))
        rows = {
            a: Mention("a", "Q1", False),
            b: Mention("b", "Q2", False),
            c: Mention("c", "_", True),
            d: Mention("d", "Q3", False),
        }
        report = outcomes({a, b, c, d}, {a}, rows)
        self.assertEqual(report["exact_supported"], 0)
        self.assertEqual(report["with_unscored_hit"], 1)
        self.assertEqual(report["wrong_hits_on_supported_pages"], 1)
        self.assertEqual(report["wrong_only_pages"], 1)


if __name__ == "__main__":
    unittest.main()
