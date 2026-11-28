# SPDX-License-Identifier: MIT OR Apache-2.0
"""Original controls for the independent pronoun transfer selector."""
from collections import Counter
import os
from pathlib import Path
import stat
import tempfile
import unittest

from coref_corpora.litbank import Mention
from litbank_pronoun_transfer import choices, export_panel, phrases


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

    def test_reflexive_slice_is_disjoint_from_the_original_target_forms(self):
        text = "Ada greeted Béa. She congratulated herself."
        mentions = [Mention((text.index(word), text.index(word) + len(word)), tag, chain)
                    for word, tag, chain in [("Ada", "PROP_PER", "a"), ("Béa", "PROP_PER", "b"),
                                             ("She", "PRON_PER", "a"), ("herself", "PRON_PER", "a")]]
        original, = choices("w", text, mentions, Counter())
        reflexive, = choices("w", text, mentions, Counter(), pronouns={"herself", "himself"})
        self.assertEqual((original["Pronoun"], reflexive["Pronoun"]), ("She", "herself"))
        self.assertEqual((reflexive["A"], reflexive["B"], reflexive["A-coref"]), ("Ada", "Béa", "TRUE"))

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

    def test_private_export_checks_before_publish_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.tsv"
            with self.assertRaisesRegex(ValueError, "frozen selection"):
                export_panel(path, [{"ID": "synthetic"}], {}, {"wrong": "digest"})
            self.assertEqual(list(path.parent.iterdir()), [])
            saved_umask = os.umask(0)
            try:
                export_panel(path, [{"ID": "synthetic"}], {})
            finally:
                os.umask(saved_umask)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            old = path.read_bytes()
            with self.assertRaises(FileExistsError):
                export_panel(path, [{"ID": "replacement"}], {})
            self.assertEqual(old, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
