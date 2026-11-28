# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest

from litbank_reflexives import score


class ReflexiveDecisions(unittest.TestCase):
    def test_enclosing_both_names_or_enclosing_a_wrong_name_never_earns_exact_credit(self):
        template = {"Text": "Mia met Zoe and excused herself.", "A": "Mia", "A-offset": 0,
                    "B": "Zoe", "B-offset": 8, "Pronoun": "herself", "Pronoun-offset": 24,
                    "A-coref": "FALSE", "B-coref": "TRUE"}
        rows = [dict(template, ID="both"), dict(template, ID="wrong"), dict(template, ID="elsewhere")]
        groups = [[[[0, 11], [24, 31]]], [[[0, 7], [24, 31]]], [[[0, 11], [8, 15]], [[24, 31]]]]
        outputs = [{"producer": "fcoref", "id": row["ID"], "clusters": clusters}
                   for row, clusters in zip(rows, groups, strict=True)]
        receipt = score(rows, outputs, "fcoref", "predicted")
        self.assertEqual(receipt["outcomes"]["no_named_link"], 3)
        self.assertEqual(receipt["enclosing_names_without_exact_link"],
                         {"both_names_enclosed": 1, "one_name_enclosed_wrong_distractor": 1,
                          "neither_name_enclosed": 1})

    def test_both_candidates_and_missing_query_are_distinct_from_unique_answers(self):
        template = {"Text": "Mia met Zoe and excused herself.", "A": "Mia", "A-offset": 0,
                    "B": "Zoe", "B-offset": 8, "Pronoun": "herself", "Pronoun-offset": 24,
                    "A-coref": "FALSE", "B-coref": "FALSE"}
        rows = [dict(template, ID=str(i)) for i in range(4)]
        a, b, p = [0, 3], [8, 11], [24, 31]
        outputs = [{"producer": "mention-ranking", "id": row["ID"], "clusters": groups}
                   for row, groups in zip(rows, [[[a, b, p]], [[a, b]], [[a, p]], []], strict=True)]
        receipt = score(rows, outputs, "mention-ranking", "predicted")
        counts = receipt["outcomes"]
        self.assertEqual((counts["both"], counts["no_named_link"], counts["wrong_neither"]), (1, 2, 1))
        self.assertEqual(receipt["exact_query_A_B_returned"], {"111": 1, "011": 1, "110": 1, "000": 1})
