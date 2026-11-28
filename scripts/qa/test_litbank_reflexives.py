# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest

from litbank_reflexives import score


class ReflexiveDecisions(unittest.TestCase):
    def test_enclosing_both_names_or_enclosing_a_wrong_name_never_earns_exact_credit(self):
        template = {"Text": "Mia met Zoe and excused herself.", "A": "Mia", "A-offset": 0,
                    "B": "Zoe", "B-offset": 8, "Pronoun": "herself", "Pronoun-offset": 24,
                    "A-coref": "FALSE", "B-coref": "TRUE"}
        rows = [dict(template, ID=f"litbank-{i}-124-131") for i in (1, 2, 3)]
        groups = [[[[0, 11], [24, 31]]], [[[0, 7], [24, 31]]], [[[0, 11], [8, 15]], [[24, 31]]]]
        outputs = [{"producer": "fcoref", "id": row["ID"], "clusters": clusters}
                   for row, clusters in zip(rows, groups, strict=True)]
        receipt = score(rows, outputs, "fcoref", "predicted", {"2": {(100, 103): "person", (100, 107): "other"}})
        self.assertEqual(receipt["enclosing_span_gold_chain_vs_only_covered_name"], {"different": 1})
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

    def test_maverick_refusal_and_omitted_singletons_remain_distinct(self):
        template = {"Text": "Mia met Zoe and excused herself.", "A": "Mia", "A-offset": 0,
                    "B": "Zoe", "B-offset": 8, "Pronoun": "herself", "Pronoun-offset": 24,
                    "A-coref": "FALSE", "B-coref": "TRUE"}
        rows = [dict(template, ID="refusal"), dict(template, ID="singletons")]
        outputs = [{"producer": "maverick", "id": "refusal", "unsupported": "supplied_span_splits_token"},
                   {"producer": "maverick", "id": "singletons", "clusters": []}]
        receipt = score(rows, outputs, "maverick", "supplied")
        self.assertEqual(receipt["answers_by_gold"]["B"], {"refused": 1, "no_named_link": 1})
        self.assertEqual(receipt["exact_query_A_B_returned"], {"000": 1})
