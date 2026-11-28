import unittest

import gap_receipts
from native_gap_multiword import compare


class PairedDecisionTests(unittest.TestCase):
    def test_new_wrong_link_is_charged_even_when_another_wrong_link_disappears(self):
        spans = [(0, 2), (3, 6), (7, 10)]
        row = {
            "Text": "Jo Kim she",
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
        result = compare(rows, stock, candidate, frozen)
        self.assertEqual(result["correct_links"], dict(stock=0, candidate=1, gained=1, lost=0))
        self.assertEqual(result["wrong_links"], dict(stock=1, candidate=1, gained=1, lost=1))
        self.assertEqual(result["newly_wrong_links_by_candidate_outcome"], {"neither_name_is_gold": 1})
        self.assertFalse(result["passes_gate"])


if __name__ == "__main__":
    unittest.main()
