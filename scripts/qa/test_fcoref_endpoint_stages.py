# SPDX-License-Identifier: MIT OR Apache-2.0
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from fcoref_endpoint_stages import classify, score, span_ranks, rival_directions
from gap_receipts import digest


def trace(encoding, first):
    legal = [(i, j) for i in range(len(encoding.offsets)) for j in range(i, min(i + 30, len(encoding.offsets)))
             if all(encoding.attention_mask[i:j + 1])]
    ordered = first + [pair for pair in legal if pair not in first]
    return [(sum(encoding.attention_mask) + 3) // 4,
            [[i, j, -n] for n, (i, j) in enumerate(ordered)]]


class EndpointStagesTests(unittest.TestCase):
    def test_distinguishes_each_first_loss_and_its_roles(self):
        p, n = (1, 2), (5, 7)
        for ranks, clusters, expected in [
            ({p: 7}, {}, ("no_exact_candidate", "name")),
            ({p: 7, n: 8}, {}, ("below_stock_budget", "both")),
            ({p: 1, n: 8}, {}, ("below_stock_budget", "name")),
            ({p: 1, n: 2}, {n: 4}, ("retained_but_not_emitted", "pronoun")),
            ({p: 1, n: 2}, {n: 3, p: 4}, ("emitted_separately", None)),
            ({p: 1, n: 2}, {n: 3, p: 3}, ("linked", None)),
        ]:
            with self.subTest(expected=expected):
                self.assertEqual(classify(p, n, ranks, 2, clusters), expected)

    def test_rivals_are_strict_containments_of_the_same_occurrence(self):
        name = (10, 18)
        self.assertEqual(rival_directions([(7, 18), (10, 23)], name), "longer")
        self.assertEqual(rival_directions([(10, 12), (13, 18)], name), "shorter")
        self.assertEqual(rival_directions([(7, 18), (13, 18)], name), "both")
        self.assertEqual(rival_directions([name, (0, 8), (18, 26), (14, 22)], name), "none")

    def test_projection_uses_native_trim_and_best_duplicate_rank(self):
        text = "\u2003É\x1c\u00a0"
        enc = SimpleNamespace(offsets=[(0, 0), (0, 1), (1, 3), (3, 4), (0, 0)],
                              attention_mask=[1] * 5)
        ranked = trace(enc, [(0, 3), (1, 3), (2, 2)])
        self.assertEqual(span_ranks(text, enc, ranked), {(1, 3): 1})

    def test_rejects_incomplete_illegal_nonfinite_and_misordered_traces(self):
        enc = SimpleNamespace(offsets=[(0, 1), (1, 2), (2, 3)], attention_mask=[1, 0, 1])
        baseline = trace(enc, [])
        for bad in (
            [1, baseline[1][:1]],
            [1, [[0, 0, 1], [0, 0, 0]]],
            [1, [[0, 2, 1], [2, 2, 0]]],
            [1, [[0, 0, float("nan")], [2, 2, 0]]],
            [2, baseline[1]],
            [1, [[0, 0, 0], [2, 2, 1]]],
            [1, [[2, 2, 1], [0, 0, 1]]],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                span_ranks("abc", enc, bad)

    def test_exact_double_budget_boundary_and_frozen_partition(self):
        text = "A B she."
        enc = SimpleNamespace(offsets=[(0, 0), (0, 1), (2, 3), (4, 7), (7, 8), (0, 0)],
                              attention_mask=[1] * 6)
        tokenizer = SimpleNamespace(encode=lambda _: enc)
        row = {"ID": "one", "Text": text, "Pronoun": "she", "A": "A", "B": "B",
               "Pronoun-offset": "4", "A-offset": "0", "B-offset": "2", "A-coref": "TRUE", "B-coref": "FALSE"}
        # K=2, pronoun is first and A is at exactly 2K. Both remain absent from final clusters.
        ranked = trace(enc, [(3, 3), (2, 2), (4, 4), (1, 1)])
        record = {"id": "one", "stock": {"id": "one", "producer": "fcoref", "clusters": []}, "trace": ranked}
        reference = {"cohort_ids_sha256": hashlib.sha256(b"one").hexdigest(),
                     "stock_partitions_sha256": digest([["one", []]]),
                     "stock": {"fp": 0, "tp": 0, "tn": 1, "fn": 1, "gold_both_covered": 0}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "reference.json")
            path.write_text(json.dumps(reference))
            def scored():
                return score([row], [record], tokenizer, reference_path=path)
            summary = scored()
            self.assertEqual(summary["below_budget_pairs_with_both_in_first_2k"], 1)
            self.assertEqual(summary["endpoint_at_first_loss"]["below_stock_budget"]["name"], 1)
            record["stock"]["clusters"] = [[[4, 7], [2, 3]]]
            with self.assertRaisesRegex(ValueError, "frozen stock run"):
                scored()
            record["stock"]["clusters"] = [[[0, 1], [2, 3]]]
            with self.assertRaisesRegex(ValueError, "absent from retained"):
                scored()
            containing = [[(4, 7), (0, 3)]]
            record["stock"]["clusters"] = containing
            record["trace"] = trace(enc, [(3, 3), (1, 2), (4, 4), (1, 1)])
            reference["stock_partitions_sha256"] = digest([["one", [[(0, 3), (4, 7)]]]])
            path.write_text(json.dumps(reference))
            rivals = scored()["below_budget_name_rivals"]
            self.assertEqual(rivals["gold_pairs"], 1)
            self.assertEqual(rivals["among_retained"]["longer"], 1)
            self.assertEqual(rivals["among_pronoun_cluster"]["longer"], 1)
            record["id"] = "two"
            with self.assertRaisesRegex(ValueError, "Trace document"):
                scored()


if __name__ == "__main__":
    unittest.main()
