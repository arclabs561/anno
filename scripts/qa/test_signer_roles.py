# SPDX-License-Identifier: MIT OR Apache-2.0
from collections import defaultdict
from pathlib import Path
from subprocess import CompletedProcess
import unittest
from unittest.mock import patch

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

    def test_reflexive_name_order_cannot_solve_both_clauses(self):
        rows = signer.reflexive_panel()
        self.assertEqual(signer.digest(rows), "9ed58ba2b32c83a10c9554a8acf5594fe89054f76b12844a3a3c2537437a7b36")
        outcomes = []
        for a, b in zip(rows[::2], rows[1::2], strict=True):
            self.assertEqual(signer.gap.spans(a), signer.gap.spans(b))
            self.assertEqual([a["gold"], b["gold"]], ["A", "B"])
            for row in (a, b):
                p, _, nearest = signer.gap.spans(row)
                output = {"id": row["ID"], "producer": "fcoref", "clusters": [[list(nearest), list(p)]]}
                outcomes.append(signer.observe(row, output, "fcoref", "predicted"))
        summary = signer.summarize(rows, outcomes)
        self.assertEqual(summary["by_family"]["reflexive_clause"], {"correct": 12, "wrong": 12})
        self.assertEqual(summary["paired_unique_decisions"]["reflexive_clause"], {"same_unique_choice": 12})

    def test_authored_outer_phrase_is_distinct_from_a_named_link(self):
        rows = signer.reflexive_panel(infinitive=True)
        self.assertEqual(signer.digest(rows), "68bfd07f9f539e38bbcdcd73a736d4bf11c1bca25c4fde1a5e54d896a495333e")
        for row in rows:
            p, a, b = signer.gap.spans(row)
            group = [[a[0], b[1]], list(p)]
            result = signer.observe(row, {"id": row["ID"], "producer": "fcoref", "clusters": [group]},
                                    "fcoref", "predicted")
            self.assertEqual(result["status"], "candidate_missing")
            if row["gold"] == "A":
                self.assertEqual(result["diagnostic"], "exact_query_linked_to_outer_phrase_headed_by_A")
            else:
                self.assertNotIn("diagnostic", result)

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

    def test_named_answer_and_intermediate_role_evidence_remain_distinct(self):
        rows = signer.panel()
        row = next(r for r in rows if r["family"] == "exclusive")
        p, _, _ = signer.gap.spans(row)
        start = row["Text"].index("Only the ")
        end = row["Text"].index(" signed the register", start)
        out = {"id": row["ID"], "producer": "fcoref", "clusters": [[list(p), [start, end]]]}
        result = signer.observe(row, out, "fcoref", "predicted")
        self.assertEqual(result["status"], "candidate_missing")
        self.assertEqual(result["diagnostic"], "exact_query_linked_to_cue_phrase_without_named_answer")
        with self.assertRaisesRegex(ValueError, "unrequested span"):
            signer.observe(row, out, "fcoref", "supplied")
        row = next(r for r in rows if r["family"] == "repeated_name" and r["gold"] == "B")
        p, _, (start, end) = signer.gap.spans(row)
        out.update(id=row["ID"], clusters=[[list(p), [start, end + 1]]])
        result = signer.observe(row, out, "fcoref", "predicted")
        self.assertEqual(result["status"], "candidate_missing")
        self.assertEqual(result["diagnostic"], "exact_query_linked_to_expected_name_with_period")

    def test_unsupported_feature_and_worker_failure_are_distinct(self):
        row = signer.panel()[0]
        reason = f"fcoref on {row['ID']}: Feature not available: fcoref cannot cluster supplied mentions"
        with patch.object(signer, "FC_FILES", {}), patch.object(signer.subprocess, "run") as call:
            call.return_value = CompletedProcess([], 1, "", reason)
            receipt, details = signer.replay([row], Path("worker"), "fcoref-supplied")
            self.assertEqual((receipt["status"], details), ("unsupported_operation", []))
            call.return_value = CompletedProcess([], 1, "", "model initialization failed")
            with self.assertRaisesRegex(ValueError, "initialization failed"):
                signer.replay([row], Path("worker"), "fcoref-supplied")

    def test_invariant_position_cannot_get_pair_credit(self):
        rows = signer.panel()
        observations = [dict(choice="A", status="correct" if r["gold"] == "A" else "wrong") for r in rows]
        summary = signer.summarize(rows, observations)["paired_unique_decisions"]
        self.assertEqual(sum(c["same_unique_choice"] for c in summary.values()), 24)
        self.assertEqual(sum(c["both_correct"] for c in summary.values()), 0)


if __name__ == "__main__":
    unittest.main()
