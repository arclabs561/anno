# SPDX-License-Identifier: MIT OR Apache-2.0
"""Original controls for the independent pronoun transfer selector."""
from collections import Counter
import unittest

from coref_corpora.litbank import Mention
from litbank_pronoun_transfer import choices, phrases


class TransferChoices(unittest.TestCase):
    def test_choices_are_independent_of_gold_and_recover_scalar_spans(self):
        text = "🦦 Neri greeted Sol. Neri waved at her."
        first, sol, neri, her = [text.index("Neri"), text.index("Sol"), text.rindex("Neri"), text.index("her")]
        names = [Mention((i, i + width), "PROP_PER", chain) for i, width, chain in
                 [(first, 4, "n"), (sol, 3, "s"), (neri, 4, "n")]]
        for chain, expected_a, expected_b in (("s", "TRUE", "FALSE"), ("n", "FALSE", "TRUE"),
                                              ("elsewhere", "FALSE", "FALSE")):
            pronoun = Mention((her, her + 3), "PRON_PER", chain)
            row, = choices("synthetic", text, [*names, pronoun], Counter())
            self.assertEqual((row["A"], row["A-offset"], row["B"], row["B-offset"],
                              row["Pronoun-offset"]), ("Sol", sol, "Neri", neri, her))
            self.assertEqual((row["A-coref"], row["B-coref"]), (expected_a, expected_b))

    def test_aliases_of_one_person_and_distant_candidates_are_excluded(self):
        text = "Ilo met Kai. She waited."
        mentions = [Mention((0, 3), "PROP_PER", "one"), Mention((8, 11), "PROP_PER", "one"),
                    Mention((13, 16), "PRON_PER", "one")]
        counts = Counter()
        self.assertEqual(list(choices("x", text, mentions, counts)), [])
        self.assertEqual((counts["same_candidate_identity"], counts["both_true_excluded"]), (1, 1))
        padded = text[:8] + " " * 500 + text[8:]
        shifted = [mentions[0], *(Mention((a + 500, b + 500), tag, chain)
                                  for (a, b), tag, chain in mentions[1:])]
        counts = Counter()
        self.assertEqual(list(choices("x", padded, shifted, counts)), [])
        self.assertEqual(counts["no_second_distinct_name"], 1)

    def test_overlap_check_uses_twenty_contiguous_casefolded_words(self):
        words = " ".join("word" + str(i) for i in range(20))
        self.assertEqual(phrases(words), phrases(words.upper().replace(" ", ", ")))
        self.assertEqual(len(phrases(words)), 1)
        self.assertFalse(phrases(words.replace("word5 ", "")))


if __name__ == "__main__":
    unittest.main()
