"""Guard semantics of the independent JEREX native-output audit."""
import unittest

from audit_jerex_edges import compare


class JerexEdgeAuditTest(unittest.TestCase):
    def setUp(self):
        mention = {"pos": [0, 1], "sent_id": 0, "type": "PER"}
        self.gold = {"title": "Example", "sents": [["Ada", "read"]], "vertexSet": [[mention]], "labels": []}
        self.pred = {
            "title": "Example", "tokens": ["Ada", "read"],
            "mentions": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
            "entities": [{"mentions": [0], "type": "PER"}], "relations": [],
        }

    def test_detection_counts_unclustered_mentions(self):
        totals = compare([self.gold], [self.pred])["totals_on_identifiable_docs"]
        self.assertEqual(totals["mentions"], {"gold": 1, "predicted": 2, "matched": 1})
        self.assertEqual(totals["clusters"], {"gold": 1, "predicted": 1, "matched": 1})

    def test_duplicate_membership_keeps_attempted_document(self):
        self.pred["entities"] *= 2
        self.pred["relations"] = [{"head": 0, "tail": 1, "type": "knows"}]
        report = compare([self.gold], [self.pred])
        self.assertEqual((report["attempted"], report["unique_membership_docs"]), (1, 0))
        self.assertEqual(report["documents"][0]["predicted_indexed_edges"], 1)
        self.assertEqual(report["totals_on_identifiable_docs"], {})

    def test_negative_mention_index_is_invalid(self):
        self.pred["entities"][0]["mentions"] = [-1]
        with self.assertRaisesRegex(ValueError, "invalid predicted mention index"):
            compare([self.gold], [self.pred])
