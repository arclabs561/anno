# SPDX-License-Identifier: MIT OR Apache-2.0
"""Guard the reader denominators using original mentions, including two kinds of unknown."""
import unittest

from .ajmc import Mention, mention, score


class ReaderQueries(unittest.TestCase):
    def test_missing_annotations_cannot_count_as_wrong_people_or_as_exact_answers(self):
        docs = {
            "en/1": [Mention("Ora", "Q1", False)],
            "de/2": [Mention("Öra", "Q1", False), Mention("Oro", "Q2", False),
                     Mention("Or", "NIL", False), Mention("O", "_", True)],
        }
        merged = {("en/1", 0): 1, **{("de/2", i): 1 for i in range(4)}}
        result = score(docs, merged)
        pairs = result["pairs"]["cross_volume"]
        self.assertEqual((pairs["pairs"], pairs["tp"], pairs["fp"], pairs["tp_different_spelling"]),
                         (2, 1, 1, 1))
        queries = result["reader_queries"]
        self.assertEqual((queries["with_other_page_gold"], queries["retrieved_any_correct"],
                          queries["retrieved_exact_supported_set"]), (2, 2, 1))
        self.assertEqual((queries["shown_with_wrong_qid"], queries["shown_with_unscorable_hit"]), (2, 1))
        self.assertEqual((result["explicit_nil"], result["unspecified_secondary"]), (1, 1))

        clean = dict(merged)
        for i in (1, 2, 3):
            clean["de/2", i] = i + 1
        self.assertEqual(score(docs, clean)["reader_queries"]["retrieved_exact_supported_set"], 2)

    def test_partial_ocr_token(self):
        value = mention([("«Év", "Q1", ["Partial-1:3", "NoSpaceAfter"]), ("a", "Q1", [])])
        self.assertEqual(value, Mention("Éva", "Q1", False))
