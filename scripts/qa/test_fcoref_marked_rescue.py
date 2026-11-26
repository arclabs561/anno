# SPDX-License-Identifier: MIT OR Apache-2.0
import hashlib
import types
import unittest
from unittest.mock import patch

from fcoref_marked_rescue import digest, documents, measure, run_arm, summarize


class MarkedRescueTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(ID="doc", Text="Zoë met Bo; she left.", A="Zoë", B="Bo", Pronoun="she")
        self.row.update({"A-offset": "0", "B-offset": "8", "Pronoun-offset": "12",
                         "A-coref": "TRUE", "B-coref": "FALSE"})
        self.docs = documents([self.row])
        self.a, self.b, self.pronoun = (0, 3), (8, 10), (12, 15)

    def test_reservation_sends_exact_utf8_ranges_without_labels(self):
        out = types.SimpleNamespace(stdout='{"id":"doc","producer":"fcoref","clusters":[]}\n',
                                    stderr="FCOREF_RESERVE 3 3 1 5\n")
        with patch("fcoref_marked_rescue.subprocess.run", return_value=out) as run:
            _, stats = run_arm("binary", {}, self.docs, True)
            from json import loads
            self.assertEqual(loads(run.call_args.kwargs["input"]), {
                "id": "doc", "text": self.row["Text"], "marked_bytes": [[13, 16], [0, 4], [9, 11]]})
            self.assertEqual(stats, [(3, 3, 1, 5)])

    def test_preserves_lost_links_and_rejects_different_stock_partition(self):
        def pred(*mentions):
            return [{"id": "doc", "producer": "fcoref", "clusters": [list(mentions)]}]
        stock = pred(self.a, self.pronoun)
        marked = pred(self.b, self.pronoun)
        expected = dict(stock_partitions_sha256=digest([("doc", [[self.a, self.pronoun]])]),
                        stock=measure(self.docs, stock, "fcoref")["counts"],
                        cohort_ids_sha256=hashlib.sha256(b"doc").hexdigest())
        result = summarize(self.docs, stock, marked, [(3, 3, 1, 5)], expected)
        self.assertEqual(result["paired"]["gold_links_lost"], 1)
        self.assertEqual(result["paired"]["unmatched_links_added"], 1)
        self.assertEqual(result["output_pair_changes"]["added_unscored_pairs"], 0)
        with self.assertRaisesRegex(ValueError, "stock panel"):
            summarize(self.docs, marked, stock, [(3, 3, 1, 5)], expected)

    def test_bad_native_candidate_budget_fails(self):
        out = types.SimpleNamespace(stdout="", stderr="FCOREF_RESERVE 3 2 3 2\n")
        with patch("fcoref_marked_rescue.subprocess.run", return_value=out):
            with self.assertRaisesRegex(ValueError, "Invalid native"):
                run_arm("binary", {}, self.docs, True)


if __name__ == "__main__":
    unittest.main()
