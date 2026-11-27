# SPDX-License-Identifier: MIT OR Apache-2.0
import unittest

from audit_gap_decisions import spans
from litbank_pronoun_distance import arm, boundary_probe
from native_gap_margin import lookup


class SearchDistanceTests(unittest.TestCase):
    def test_newly_eligible_name_can_still_be_rejected(self):
        row = boundary_probe(101)
        pronoun, name, later_name = spans(row)
        self.assertEqual(pronoun[0] - name[1], 101)
        self.assertEqual(len(row["Text"][name[1]:pronoun[0]].encode()), 202)
        members = [[*name, 0], [*pronoun, 1], [*later_name, 2]]
        for cap, scores, expected in [(100, [], 0), (500, [[0, .20, .225]], 1)]:
            trace = {"id": row["ID"], "members": members, "max_distance": cap,
                     "decisions": [{}, {"candidates": scores, "selected": None}, {}]}
            result = arm([lookup(row, trace)], 1)
            self.assertEqual(result["gold_eligible"], expected)
            self.assertEqual(result["scored_pronoun_name_pairs"], expected)
            self.assertEqual(result["gold_selected_directly"], 0)
            self.assertEqual(result["returned"]["abstained"], 1)


if __name__ == "__main__":
    unittest.main()
