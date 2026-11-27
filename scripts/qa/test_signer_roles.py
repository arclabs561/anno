# SPDX-License-Identifier: MIT OR Apache-2.0
from collections import defaultdict
import unittest

import signer_roles as signer


class SignerRolesTests(unittest.TestCase):
    def test_balanced_minimal_pairs_and_unicode_spans(self):
        rows = signer.panel()
        self.assertEqual(len({r["ID"] for r in rows}), 52)
        self.assertEqual(signer.digest(rows), "486ffe3a6876b2ed70cb70593c2e15b749d8f116640151e11cfcdd71a0a83f57")
        pairs = defaultdict(list)
        for row in rows:
            if row["gold"] is not None:
                pairs[row["pair"]].append(row)
        self.assertEqual(len(pairs), 24)
        for a, b in pairs.values():
            self.assertEqual((a["gold"], b["gold"]), ("A", "B"))
            self.assertEqual(signer.gap.spans(a), signer.gap.spans(b))
            if a["family"] != "repeated_name":
                self.assertEqual(a["Pronoun"], b["Pronoun"])
                self.assertEqual(sum(x != y for x, y in zip(a["Text"].split(), b["Text"].split(), strict=True)), 1)
        self.assertTrue(any(len(r["Text"][:r["Pronoun-offset"]].encode()) > r["Pronoun-offset"] for r in rows))

    def test_exact_bounds_and_competing_answers_have_separate_verdicts(self):
        row = signer.panel()[0]
        p, a, b = map(list, signer.gap.spans(row))
        def score(groups):
            return signer.observe(row, {"id": row["ID"], "producer": "fcoref", "clusters": groups},
                                  "fcoref", "predicted")
        self.assertEqual(score([[p, a]])["status"], "correct")
        self.assertEqual(score([[p, b]])["status"], "wrong")
        self.assertEqual(score([[p, a, b]])["status"], "both_linked")
        self.assertEqual(score([[p], [a], [b]])["status"], "abstained")
        self.assertEqual(score([[p], [b]])["status"], "candidate_missing")
        broad = score([[[p[0], p[1]+4], a]])
        self.assertEqual((broad["status"], broad["names_linked_to_strictly_enclosing_query"]), ("query_missing", "A"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score([[p, a], [p, b]])

    def test_invariant_position_cannot_get_pair_credit(self):
        rows = signer.panel()
        observations = [dict(choice="A", status="correct" if r["gold"] == "A" else "wrong") for r in rows]
        summary = signer.summarize(rows, observations)["paired_unique_decisions"]
        self.assertEqual(sum(c["same_unique_choice"] for c in summary.values()), 24)
        self.assertEqual(sum(c["both_correct"] for c in summary.values()), 0)


if __name__ == "__main__":
    unittest.main()
