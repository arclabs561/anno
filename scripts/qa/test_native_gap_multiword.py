import unittest

import gap_receipts
from native_gap_multiword import compare, linking_edges


class PairedDecisionTests(unittest.TestCase):
    def test_indirect_path_preserves_direction_on_both_sides_of_common_ancestor(self):
        root, name, pronoun = (0, 2), (3, 6), (7, 10)
        trace = {"retained": [root, name, pronoun], "decisions": [
            {"selected": None, "candidates": []},
            {"selected": 0, "candidates": [(0, .8, .5)]},
            {"selected": 0, "candidates": [(1, .4, .5), (0, .7, .5)]},
        ]}
        self.assertEqual(linking_edges(trace, pronoun, name), [
            {"mention": pronoun, "antecedent": root, "score": .7, "threshold": .5},
            {"mention": name, "antecedent": root, "score": .8, "threshold": .5},
        ])

    def test_new_wrong_link_is_charged_even_when_another_wrong_link_disappears(self):
        spans = [(0, 2), (3, 6), (7, 10)]
        row = {
            "Text": "Jo Kim she", "URL": "fixture",
            "A": "Jo", "A-offset": "0",
            "B": "Kim", "B-offset": "3",
            "Pronoun": "she", "Pronoun-offset": "7",
            "A-coref": "FALSE", "B-coref": "FALSE",
        }

        def trace(doc, selected):
            earlier = [1, 0]
            return {
                "id": doc, "proposed": spans, "retained": spans, "max_distance": 50,
                "decisions": [
                    {"selected": None, "max_antecedents": 2, "candidates": []},
                    {"selected": None, "max_antecedents": 2, "candidates": [(0, 0, 0.5)]},
                    {"selected": selected, "max_antecedents": 2,
                     "candidates": [(j, float(j == selected), 0.5) for j in earlier]},
                ],
                "clusters": [] if selected is None else [[spans[selected], spans[2]]],
            }

        rows = [dict(row, ID="swap"), dict(row, ID="gain", **{"B-coref": "TRUE"})]
        stock = [trace("swap", 0), trace("gain", None)]
        candidate = [trace("swap", 1), trace("gain", 1)]
        frozen = {
            "gold_pairs": 1,
            "partitions_sha256": gap_receipts.digest([t["clusters"] for t in stock]),
            "name_boundary_observations": {"multiword_only_subspans": 0},
        }
        changes = []
        result = compare(rows, stock, candidate, frozen, changes=changes)
        self.assertEqual([(d["id"], d["candidate"], d["gold_link"], d["stock"],
                           d["candidate_outcome"]) for d in changes], [
            ("swap", "A", False, "linked", "isolated_endpoint"),
            ("swap", "B", False, "isolated_endpoint", "linked"),
            ("gain", "B", True, "isolated_endpoint", "linked"),
        ])
        self.assertEqual(changes[0]["name"], {"text": "Jo", "chars": (0, 2)})
        self.assertEqual(result["correct_links"], dict(stock=0, candidate=1, gained=1, lost=0))
        self.assertEqual(result["wrong_links"], dict(stock=1, candidate=1, gained=1, lost=1))
        self.assertEqual(result["newly_wrong_links_by_candidate_outcome"], {"neither_name_is_gold": 1})
        self.assertFalse(result["passes_gate"])


if __name__ == "__main__":
    unittest.main()
